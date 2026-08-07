"""
LLM에게 노출하는 도구.

각 도구는 다른 도구가 못 하는 일을 한다. 경계는 저장소 단위다.

- search_movies      : TMDB 조건 조회·정렬·개수·OTT 편성·상영 상태
- get_movie_details  : TMDB 한 편의 상세(감독·출연·시청처)
- search_by_vibe     : 로컬 Chroma 무드 검색            (P4에서 추가)
- web_search         : 평단·수상·화제·비하인드·해석      (P5에서 추가)

도구의 docstring이 곧 라우팅 규칙이자 프롬프트다. 예시를 넉넉히 넣을수록 정확해진다.
결과가 없을 때 "조건을 완화해보세요" 같은 안내를 반환하면 LLM이 알아서 재호출한다.
그래프에 재시도 루프를 새길 필요가 없다.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Literal

from langchain_core.documents import Document
from langchain_core.tools import tool

from rag import catalog, tmdb
from rag.sources import doc_to_source, tmdb_to_source

logger = logging.getLogger(__name__)

# 표본이 부족한 영화를 후보에서 배제한다. 이게 없으면 1표에 10점 받은 무명작이
# 최상단에 온다(실측: ROBLOX_OOF.mp3, 10.0점 1표).
_MIN_VOTE_COUNT = 300

# 되묻기 힌트를 만들 때 쓰는 표본 하한. 여기까지 올리면 고전 명작만 남는다.
_HINT_VOTE_COUNT = 3000

# 답변 밑에 "무엇을 보고 답했는지"를 적기 위한 표시. 도구가 어느 데이터에
# 기대는지는 도구 자신이 가장 잘 알므로 여기에 둔다.
TMDB_ATTRIBUTION = "tmdb"
LOCAL_ATTRIBUTION = "local"
WEB_ATTRIBUTION = "web"
# TMDB의 시청처 데이터는 JustWatch가 제공하며 **표기가 의무다.** 다른 표시와 달리
# 빠뜨리면 약관 위반이라, 애매하면 붙이는 쪽으로 판단한다.
JUSTWATCH_ATTRIBUTION = "justwatch"

# 화면에 늘어놓는 순서. 무엇을 검색했는지(TMDB·로컬·웹)를 먼저 보여주고,
# 제공자 표기는 뒤에 붙인다.
ATTRIBUTION_ORDER = (
    TMDB_ATTRIBUTION,
    LOCAL_ATTRIBUTION,
    WEB_ATTRIBUTION,
    JUSTWATCH_ATTRIBUTION,
)

_TOOL_ATTRIBUTIONS = {
    "search_movies": (TMDB_ATTRIBUTION,),
    # 상세 응답에는 시청처 줄이 언제나 들어간다. 답변이 그 줄을 옮겨 적었는지는
    # 알 수 없으므로, 불렀다면 표기한다.
    "get_movie_details": (TMDB_ATTRIBUTION, JUSTWATCH_ATTRIBUTION),
    "search_by_vibe": (LOCAL_ATTRIBUTION,),
    "web_search": (WEB_ATTRIBUTION,),
}


def _artifact(
    *,
    success: bool,
    sources: list[dict] | None = None,
    web_sources: list[dict] | None = None,
) -> dict:
    """도구 실행 결과의 구조화된 근거.

    본문 문자열은 LLM용이고, API 출처는 이 artifact만 신뢰한다. 실패·빈 결과도
    같은 모양으로 남겨 호출 시도와 성공한 조회를 구분한다.
    """
    return {
        "success": success,
        "sources": list(sources or []),
        "web_sources": list(web_sources or []),
    }


def attributions_for(name: str, args: dict | None = None) -> tuple[str, ...]:
    """도구 호출 하나가 어떤 출처를 쓴 것인지."""
    marks = _TOOL_ATTRIBUTIONS.get(name, ())
    # 편성으로 거르는 검색은 JustWatch 데이터를 조건으로 쓴 것이다.
    if name == "search_movies" and (args or {}).get("watch_provider"):
        marks += (JUSTWATCH_ATTRIBUTION,)
    return marks


_SORT_MAP = {
    "popularity": "popularity.desc",
    "rating_desc": "vote_average.desc",
    "rating_asc": "vote_average.asc",
    "year_desc": "primary_release_date.desc",
    "year_asc": "primary_release_date.asc",
}

_STATUS_PATHS = {
    "now_playing": "/movie/now_playing",
    "upcoming": "/movie/upcoming",
    "trending": "/trending/movie/week",
}


def _format_movies(
    movies: list[dict],
    *,
    detail_shape: bool = False,
    sources: list[dict] | None = None,
) -> str:
    """LLM이 읽을 목록 텍스트. 출처 카드와 달리 사람이 읽는 형태다.

    `sources`를 주면 감독을 함께 적는다. **제목과 연도만으로는 원작과 리메이크를
    가를 수 없다.** TMDB의 with_people은 배역·각본·제작까지 모두 걸려서, '박찬욱
    영화'를 물으면 그가 원작자로만 이름을 올린 스파이크 리의 올드보이(2013)가
    함께 온다. 감독이 적혀 있지 않으면 LLM은 그것을 자기가 아는 올드보이(2003)로
    읽고 그 줄에 2003년 설명을 붙인다(실측).

    카드를 만들며 이미 확보한 값이라 추가 조회는 없다.
    """
    id_to_name = tmdb.genre_id_to_name()
    directors = [(s.get("director") or "") for s in (sources or [])]
    lines = []
    for index, m in enumerate(movies):
        if detail_shape:
            genres = ", ".join(g["name"] for g in m.get("genres", []))
        else:
            genres = ", ".join(
                id_to_name.get(i, "") for i in m.get("genre_ids", []) if i in id_to_name
            )
        year = (m.get("release_date") or "")[:4] or "연도 미상"
        rating = round(float(m.get("vote_average") or 0.0), 1)
        director = directors[index] if index < len(directors) else ""
        line = (
            f"- {m.get('title', '')} ({year}) | {genres or '장르 정보 없음'} | "
            f"평점 {rating} ({m.get('vote_count', 0)}명)"
        )
        lines.append(f"{line} | 감독 {director}" if director else line)
    return "\n".join(lines)


def _hydrated_sources(results: list[dict]) -> list[dict]:
    """완전한 출처 카드를 만든다. 카탈로그에 없는 영화만 TMDB 상세를 부른다.

    /discover와 /search 응답에는 감독·국가·출연진이 없다. 그대로 카드를 만들면
    "감독 정보 없음"이 매번 뜨고, 답한 도구에 따라 카드 모양이 달라진다.

    대부분은 로컬 카탈로그에서 해결된다(실측 적중률 97%). 남은 소수만 상세를
    부르므로 5편 질의에서 추가 요청은 평균 0.15회다.

    LLM에게 주는 본문(_format_movies)에는 여기서 얻은 감독만 넘긴다. 나머지 상세를
    다 넣으면 컨텍스트만 커지고, 감독은 같은 제목의 다른 작품을 가르는 데 필요하다.
    """
    misses = [m for m in results if not catalog.lookup(m.get("id"))]
    if misses:
        # 병렬로 미리 받아 lru_cache에 채워둔다. 5건 기준 순차 2.1초 → 병렬 1.2초.
        # tmdb_to_source가 곧바로 캐시 히트로 읽는다.
        with ThreadPoolExecutor(max_workers=len(misses)) as pool:
            list(pool.map(_prefetch_detail, misses))

    return [_source_for(m) for m in results]


def _prefetch_detail(movie: dict) -> None:
    try:
        tmdb.movie_detail(movie["id"])
    except (tmdb.TmdbError, KeyError):
        logger.warning("출처 카드 보강 실패: %s", movie.get("title"))


def _source_for(movie: dict) -> dict:
    """카탈로그 → 상세 캐시 → 목록 정보 순으로 카드를 만든다."""
    if catalog.lookup(movie.get("id")):
        return tmdb_to_source(movie)
    try:
        return tmdb_to_source(tmdb.movie_detail(movie["id"]), detail_shape=True)
    except (tmdb.TmdbError, KeyError):
        return tmdb_to_source(movie)


def _resolve_genres(genre: str) -> tuple[list[int], list[str]]:
    """'로맨스, 코미디' → ([10749, 35], []). 모르는 이름은 두 번째로 돌려준다.

    TMDB에는 '로맨틱 코미디' 같은 복합 장르가 없다. 그래서 예전에는 그 요청이
    통째로 거절당했고, 모델이 로맨스와 코미디를 따로 검색하다 결국 제 기억으로
    답했다(실측: 답변의 다섯 편이 도구 결과에 하나도 없었다). 둘을 함께 넘길 수
    있으면 한 번에 풀린다.
    """
    name_to_id = tmdb.genre_name_to_id()
    ids: list[int] = []
    unknown: list[str] = []
    for raw in genre.split(","):
        name = raw.strip()
        if not name:
            continue
        genre_id = name_to_id.get(name)
        if genre_id is None:
            unknown.append(name)
        else:
            ids.append(genre_id)
    return ids, unknown


def _scope_hint() -> str:
    """범위 없는 최상급 질의에 되물을 때 쓸 구체적 예시를 만든다."""
    try:
        top = tmdb.discover(
            sort_by="vote_average.desc", **{"vote_count.gte": _HINT_VOTE_COUNT}
        ).get("results", [])[:3]
    except tmdb.TmdbError:
        return ""
    if not top:
        return ""
    names = ", ".join(
        f"{m['title']}({round(float(m.get('vote_average') or 0), 1)}점)" for m in top
    )
    return f" 표가 매우 많은 작품만 보면 {names} 같은 고전이 상위입니다."


@tool(response_format="content_and_artifact")
def search_movies(
    person: str | None = None,
    # 여러 장르는 쉼표로 나눠 준다. 둘 다 해당하는 영화만 나온다(AND).
    genre: str | None = None,
    country: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    min_rating: float | None = None,
    watch_provider: str | None = None,
    status: Literal["now_playing", "upcoming", "trending"] | None = None,
    sort_by: Literal[
        "popularity", "rating_desc", "rating_asc", "year_desc", "year_asc"
    ] = "popularity",
    limit: int = 5,
    count_only: bool = False,
):
    """조건으로 영화 목록을 찾거나, 정렬하거나, 개수를 셉니다.

    사용 예:
    - "봉준호 영화"                  → person="봉준호"
    - "한국 스릴러 영화 추천"         → genre="스릴러", country="한국"
    - "일본 애니메이션"               → genre="애니메이션", country="일본"
    - "2020년 이후 평점 8점 이상 SF"  → genre="SF", year_from=2020, min_rating=8.0
    - "로맨틱 코미디"                 → genre="로맨스, 코미디"
    - "액션 코미디"                   → genre="액션, 코미디"
    - "가장 오래된 한국 영화"         → genre 없이 sort_by="year_asc", limit=1
    - "2020년 이후 SF 몇 편?"        → genre="SF", year_from=2020, count_only=True
    - "넷플릭스에 볼만한 액션"        → genre="액션", watch_provider="넷플릭스"
    - "지금 극장에서 뭐 해?"          → status="now_playing"
    - "개봉 예정작"                  → status="upcoming"
    - "요즘 인기 있는 영화"           → status="trending"
    - "평점 높은 액션 5편"            → genre="액션", sort_by="rating_desc", limit=5

    국적이 언급되면 반드시 country 인자를 쓰세요. 결과를 받아 국적을 하나씩
    확인하려고 get_movie_details를 반복 호출하지 마세요.

    person은 감독·배우의 이름입니다. 영화 제목이나 캐릭터 이름을 넣지 마세요.
    특정 제목을 찾을 때는 get_movie_details를 쓰세요.

    평점 순위를 물으면 장르·연도·국가·인물 중 하나로 범위를 좁혀서 호출하세요.
    범위 없이 "평점 가장 높은 영화"만 물으면 순위를 확정할 수 없습니다.

    특정 영화 한 편의 상세 정보(감독, 출연, 시청처)는 get_movie_details를 쓰세요.
    분위기나 느낌으로 찾는 질문에는 search_by_vibe를 쓰세요.
    """
    # 범위 없는 최상급은 답이 표본 하한에 따라 뒤집힌다. 단정하지 말고 되묻는다.
    if sort_by.startswith("rating") and not any(
        (person, genre, country, year_from, year_to, min_rating, watch_provider, status)
    ):
        return (
            "범위가 지정되지 않아 순위를 확정할 수 없습니다. 기준 표본을 얼마로 두느냐에 "
            f"따라 1위가 바뀝니다.{_scope_hint()} "
            "장르·연도·국가·인물 중 하나로 좁혀 주시면 정확히 답할 수 있습니다.",
            _artifact(success=False),
        )

    # TMDB의 상영 상태 전용 endpoint는 아래 조건들을 받지 않는다. 예전에는 인자를
    # 넘겨받고도 조용히 버려서 "지금 상영 중인 한국 영화"가 전 세계 목록이 됐다.
    # 첫 페이지만 받아 로컬에서 거르면 정확한 결과·개수를 보장할 수 없으므로,
    # 지원하는 척하지 않고 조건을 나눠 달라고 명시한다.
    if status:
        combined = [
            label
            for enabled, label in (
                (person, "인물"),
                (genre, "장르"),
                (country, "국가"),
                (year_from or year_to, "연도"),
                (watch_provider, "OTT"),
                (min_rating is not None, "평점"),
                (sort_by != "popularity", "정렬"),
                (count_only, "개수 집계"),
            )
            if enabled
        ]
        if combined:
            return (
                f"상영 상태 검색은 {', '.join(combined)} 조건을 함께 지원하지 않습니다. "
                "상영 상태와 세부 조건을 나눠서 질문해주세요.",
                _artifact(success=False),
            )

    try:
        if status:
            data = tmdb.list_endpoint(_STATUS_PATHS[status], region=tmdb.WATCH_REGION)
        else:
            params: dict = {
                "vote_count.gte": _MIN_VOTE_COUNT,
                "sort_by": _SORT_MAP[sort_by],
            }

            if person:
                person_id = tmdb.find_person_id(person)
                if person_id is None:
                    return (
                        f"'{person}'을(를) TMDB에서 찾지 못했습니다.",
                        _artifact(success=False),
                    )
                params["with_people"] = person_id

            if genre:
                genre_ids, unknown = _resolve_genres(genre)
                if unknown:
                    supported = ", ".join(tmdb.genre_name_to_id())
                    return (
                        f"'{', '.join(unknown)}'는 지원하지 않는 장르입니다. "
                        "'로맨틱 코미디'처럼 두 장르가 섞인 요청은 "
                        "genre=\"로맨스, 코미디\"처럼 쉼표로 나눠 주세요. "
                        f"사용 가능한 장르: {supported}",
                        _artifact(success=False),
                    )
                # 쉼표로 이으면 TMDB는 AND로 읽는다. 파이프(|)면 OR이 되어
                # 로맨스'거나' 코미디인 영화가 다 나온다 — 그건 좁히는 게 아니다.
                params["with_genres"] = ",".join(str(i) for i in genre_ids)

            if country:
                code = tmdb.resolve_country_code(country)
                if code is None:
                    return (
                        f"'{country}'는 지원하지 않는 국가입니다. "
                        f"사용 가능: {tmdb.supported_country_names()}",
                        _artifact(success=False),
                    )
                params["with_origin_country"] = code

            if watch_provider:
                provider_id = tmdb.resolve_provider_id(watch_provider)
                if provider_id is None:
                    return (
                        f"'{watch_provider}'의 편성 정보는 확인되지 않습니다. "
                        "국내에서 확인 가능한 서비스는 넷플릭스, 왓챠, 웨이브, 티빙, "
                        "디즈니플러스 등입니다.",
                        _artifact(success=False),
                    )
                params["with_watch_providers"] = provider_id
                params["watch_region"] = tmdb.WATCH_REGION

            if year_from:
                params["primary_release_date.gte"] = f"{year_from}-01-01"
            if year_to:
                params["primary_release_date.lte"] = f"{year_to}-12-31"
            if min_rating:
                params["vote_average.gte"] = min_rating

            data = tmdb.discover(**params)
    except tmdb.TmdbError as exc:
        return f"영화 정보를 가져오지 못했습니다: {exc}", _artifact(success=False)

    if count_only:
        return (
            f"조건에 맞는 영화는 총 {data.get('total_results', 0)}편입니다.",
            _artifact(success=True),
        )

    results = data.get("results", [])[:limit]
    if not results:
        return (
            "조건에 맞는 영화가 없습니다. 조건을 완화해 다시 시도해보세요.",
            _artifact(success=False),
        )

    sources = _hydrated_sources(results)
    return (
        _format_movies(results, sources=sources),
        _artifact(success=True, sources=sources),
    )


def _best_hit(hits: list[dict], title: str, year: int | None = None) -> dict:
    """제목 검색 결과에서 한 편을 고른다.

    TMDB가 주는 첫 번째를 그냥 쓰면 원작과 리메이크가 뒤집힌다. `/search/movie`는
    인기순이라 'Oldboy'로 물으면 스파이크 리의 2013년작이 먼저 온다(실측: 2003년
    원작은 표 10,033개, 리메이크는 2,162개인데도).

    연도를 알면 그것으로 확정하고, 모르면 제목이 정확히 같은 것을, 그다음은 표가
    많은 쪽을 택한다. 표 수는 "어느 쪽을 물었을 가능성이 큰가"의 대리 지표다.
    """
    query = title.strip().casefold()

    def rank(hit: dict) -> tuple:
        names = {
            (hit.get("title") or "").strip().casefold(),
            (hit.get("original_title") or "").strip().casefold(),
        }
        same_year = year is not None and (hit.get("release_date") or "")[:4] == str(year)
        return (same_year, query in names, hit.get("vote_count") or 0)

    return max(hits, key=rank)


@tool(response_format="content_and_artifact")
def get_movie_details(title: str, year: int | None = None):
    """특정 영화 한 편의 상세 정보를 가져옵니다.
    감독, 주요 출연진, 줄거리, 상영시간, 국내 OTT 시청처를 포함합니다.

    사용 예:
    - "인터스텔라 감독 누구야"   → title="인터스텔라"
    - "기생충 어디서 볼 수 있어"  → title="기생충"
    - "올드보이 출연진 알려줘"    → title="올드보이"
    - "부산행 몇 분짜리야"       → title="부산행"

    리메이크·재개봉처럼 같은 제목의 다른 작품이 있으면 연도를 함께 주세요.
    연도를 제목에 붙여 쓰면 검색이 실패합니다.
    - "박찬욱 올드보이 알려줘"    → title="올드보이", year=2003
    - "스파이크 리 올드보이"      → title="올드보이", year=2013

    여러 편의 목록이 필요하면 search_movies를 쓰세요.
    """
    try:
        hits = tmdb.search_by_title(title).get("results", [])
        if not hits:
            return (
                f"'{title}'을(를) TMDB에서 찾지 못했습니다.",
                _artifact(success=False),
            )
        detail = tmdb.movie_detail(_best_hit(hits, title, year)["id"])
    except tmdb.TmdbError as exc:
        return f"영화 정보를 가져오지 못했습니다: {exc}", _artifact(success=False)

    directors = [
        c["name"]
        for c in detail.get("credits", {}).get("crew", [])
        if c.get("job") == "Director"
    ]
    providers = tmdb.flatrate_providers(detail)

    lines = [
        f"{detail.get('title', '')} ({(detail.get('release_date') or '')[:4]})",
        f"감독: {', '.join(directors) or '정보 없음'}",
        f"출연: {', '.join(tmdb.top_cast(detail)) or '정보 없음'}",
        f"장르: {', '.join(g['name'] for g in detail.get('genres', []))}",
        f"평점: {round(float(detail.get('vote_average') or 0.0), 1)} "
        f"({detail.get('vote_count', 0)}명)",
        f"상영시간: {detail.get('runtime') or '정보 없음'}분",
        # 데이터가 불완전하므로 '없음'이 아니라 '확인되지 않음'이다.
        f"구독형 시청처(한국): {', '.join(providers) or '확인되지 않음'}",
        f"줄거리: {detail.get('overview', '')}",
    ]
    return (
        "\n".join(lines),
        _artifact(
            success=True,
            sources=[tmdb_to_source(detail, detail_shape=True)],
        ),
    )


# --- search_by_vibe ---------------------------------------------------------

# 파이프 문자열(genres/mood_tags)은 Chroma where로 거를 수 없어 파이썬에서 후처리한다.
# 후처리로 걸러지는 만큼 넉넉히 뽑아야 하는데, 컬렉션이 350편 규모라 k=100도 거의 공짜다.
_TAG_FILTER_FETCH_K = 100

# (인자 이름, 메타데이터 필드, 연산자) — Chroma where로 정확히 동작하는 축만.
_NUMERIC_FILTERS = (
    ("max_violence", "violence", "$lte"),
    ("max_sadness", "sadness", "$lte"),
    ("min_tension", "tension", "$gte"),
    ("max_complexity", "complexity", "$lte"),
    ("min_rating", "vote_average", "$gte"),
)


@lru_cache(maxsize=1)
def _vectorstore():
    """색인을 처음 쓰는 시점에 로드한다.

    모듈 import 시점에 로드하면 색인이 없는 환경(테스트, 색인 전 기동)에서
    tools.py를 import 하는 것만으로 실패한다.
    """
    from rag.store import get_vectorstore

    return get_vectorstore()


def _build_where(args: dict) -> dict | None:
    """수치·범주형만 Chroma where로 만든다.

    파이프 문자열은 절대 여기 넣지 않는다. 메타데이터 $like는 미지원이고
    $contains는 문법만 통과한 뒤 조용히 빈 결과를 준다(chromadb 1.5.9 검증).
    """
    conditions: list[dict] = []
    for arg_name, field, operator in _NUMERIC_FILTERS:
        value = args.get(arg_name)
        if value is not None:
            conditions.append({field: {operator: value}})
    if args.get("pacing"):
        conditions.append({"pacing": {"$eq": args["pacing"]}})

    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def _matches_genre(doc: Document, genre: str | None) -> bool:
    """파이프 경계까지 포함해 부분일치 오탐을 막는다('액션'이 '액션코미디'에 걸리지 않게)."""
    if not genre:
        return True
    return f"|{genre}|" in (doc.metadata.get("genres") or "")


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


@tool(response_format="content_and_artifact")
def search_by_vibe(
    vibe: str,
    max_violence: int | None = None,
    max_sadness: int | None = None,
    min_tension: int | None = None,
    max_complexity: int | None = None,
    pacing: Literal["느림", "보통", "빠름"] | None = None,
    genre: str | None = None,
    min_rating: float | None = None,
    exclude_titles: list[str] | None = None,
    limit: int = 5,
):
    """분위기·감정·감상 상황으로 영화를 찾습니다.
    엄선된 영화 안에서 검색하며, 각 영화의 감상 경험이 미리 분석되어 있습니다.

    vibe에는 사용자가 원하는 느낌을 자연어 문장으로 넣으세요.
    부정 조건은 vibe가 아니라 수치 인자로 옮기세요. 중요합니다.
    "잔인하지 않은"을 vibe에 넣으면 오히려 잔인한 영화가 나옵니다.

    사용 예:
    - "일요일 오후에 보기 좋은 잔잔한 영화"
      → vibe="주말 낮에 편하게 볼 수 있는 잔잔하고 따뜻한 영화"
    - "잔인하지 않은 액션"
      → vibe="통쾌하고 긴장감 있는 액션", genre="액션", max_violence=2
    - "슬프지만 마지막엔 위로되는"
      → vibe="무겁게 시작하지만 조용한 위로로 끝나는 감정선"
    - "머리 안 쓰고 볼 영화"
      → vibe="가볍게 틀어두고 즐길 수 있는", max_complexity=2
    - "긴장감 넘치는 스릴러"
      → vibe="숨 막히는 긴장이 이어지는", genre="스릴러", min_tension=4

    구체적인 제목·감독·연도 조회에는 이 도구를 쓰지 마세요.
    타인의 반응("사람들이 많이 운 영화", "인생영화로 꼽히는")은 web_search를 쓰세요.
    """
    try:
        store = _vectorstore()
    except (FileNotFoundError, RuntimeError) as exc:
        return f"무드 검색을 사용할 수 없습니다: {exc}", _artifact(success=False)

    where = _build_where(locals())
    fetch_k = _TAG_FILTER_FETCH_K if genre else max(limit * 3, limit)
    docs = store.similarity_search(vibe, k=fetch_k, filter=where)

    excluded = {title.strip() for title in (exclude_titles or [])}
    kept = [
        doc
        for doc in docs
        if doc.metadata.get("title") not in excluded and _matches_genre(doc, genre)
    ][:limit]

    if not kept:
        return (
            "조건에 맞는 영화를 찾지 못했습니다. 수치 조건을 완화하거나 "
            "장르를 빼고 다시 시도해보세요.",
            _artifact(success=False),
        )
    return (
        _format_docs(kept),
        _artifact(success=True, sources=[doc_to_source(doc) for doc in kept]),
    )


# --- web_search -------------------------------------------------------------

# 검색 결과 본문에서 LLM에 넘길 길이. 넘치면 컨텍스트만 먹고 정확도는 안 오른다.
_WEB_SNIPPET_LIMIT = 500


@lru_cache(maxsize=1)
def _web_search_client():
    """Tavily 클라이언트를 처음 쓰는 시점에 만든다.

    키가 없어도 tools.py import와 그래프 구성은 성공해야 한다. 웹 검색만 못 쓰는
    상태로 나머지 도구는 정상 동작하는 게 낫다.
    """
    from rag.config import settings

    if not settings.tavily_api_key:
        raise RuntimeError(
            "TAVILY_API_KEY가 설정되지 않았습니다. .env에 키를 입력하세요."
        )

    from langchain_tavily import TavilySearch

    return TavilySearch(
        max_results=settings.web_search_max_results,
        tavily_api_key=settings.tavily_api_key,
    )


@tool(response_format="content_and_artifact")
def web_search(query: str):
    """평단 반응, 시상식, 화제성, 제작 비하인드, 작품 해석을 웹에서 찾습니다.
    TMDB가 구조화해서 주지 않는 정보만 담당합니다.

    사용 예:
    - "기생충 평단 반응 어땠어"        → query="기생충 영화 평단 반응 평론가"
    - "올해 아카데미 작품상"           → query="2026 아카데미 작품상 수상작"
    - "인터스텔라 결말 무슨 뜻이야"     → query="인터스텔라 결말 해석"
    - "사람들이 인생영화로 꼽는 작품"   → query="인생영화 추천 많이 꼽히는 영화"
    - "사람들이 많이 운 영화"          → query="가장 슬픈 영화 관객 반응"
    - "그 영화 왜 논란됐어"            → query="<제목> 논란 이유"
    - "촬영에 얼마나 걸렸어"           → query="<제목> 제작 기간 비하인드"

    이 도구를 쓰지 마세요. 아래는 전부 다른 도구로 해결됩니다:
    - "넷플릭스에 뭐 있어"      → search_movies(watch_provider="넷플릭스")
    - "지금 상영 중인 영화"      → search_movies(status="now_playing")
    - "요즘 인기 있는 영화"      → search_movies(status="trending")
    - "평점 높은 영화"          → search_movies(sort_by="rating_desc")
    - "봉준호 영화"             → search_movies(person="봉준호")
    - "인터스텔라 감독 누구야"   → get_movie_details(title="인터스텔라")
    - "기생충 어디서 봐"        → get_movie_details(title="기생충")
    - "잔잔한 영화 추천"        → search_by_vibe

    웹 검색으로 영화 제목을 알아냈다면, 상세 정보와 시청처는 get_movie_details로
    다시 확인하세요.
    """
    try:
        response = _web_search_client().invoke({"query": query})
    except RuntimeError as exc:
        logger.warning("웹 검색 사용 불가: %s", exc)
        return f"웹 검색을 사용할 수 없습니다: {exc}", _artifact(success=False)
    except Exception:  # noqa: BLE001 - 외부 서비스 장애를 답변 실패로 만들지 않는다
        # 예외를 문자열로만 흘리면 서버 로그에 아무것도 안 남아 원인을 못 찾는다.
        # Tavily 무료 티어는 순간적으로 rate limit이 걸린다.
        logger.exception("웹 검색 호출 실패 (query=%r)", query)
        return (
            "웹 검색이 일시적으로 실패했습니다. 다른 도구로 답할 수 있는 질문이면 "
            "그 도구를 쓰고, 아니면 사용자에게 잠시 후 다시 시도해 달라고 하세요.",
            _artifact(success=False),
        )

    results = response.get("results", []) if isinstance(response, dict) else []
    if not results:
        return f"'{query}'에 대한 웹 검색 결과가 없습니다.", _artifact(success=False)

    blocks = []
    web_sources = []
    for item in results:
        content = (item.get("content") or "").strip()
        title = item.get("title") or "제목 없음"
        url = (item.get("url") or "").strip()
        blocks.append(
            f"[{title}]\n"
            f"{content[:_WEB_SNIPPET_LIMIT]}\n"
            f"출처: {url}"
        )
        if url:
            web_sources.append({"title": title, "url": url})
    if not web_sources:
        return "\n\n".join(blocks), _artifact(success=False)
    return (
        "\n\n".join(blocks),
        _artifact(success=True, web_sources=web_sources),
    )


# 웹 결과는 영화 카드와 분리된 web_sources artifact로 보존한다. 기사·리뷰 페이지를
# 영화 카드 스키마에 억지로 넣지 않으면서도 구체적인 URL을 API와 UI까지 전달한다.
TOOLS = [search_movies, get_movie_details, search_by_vibe, web_search]
