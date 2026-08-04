"""
TMDB 수집 스크립트 (데이터 파이프라인 1단계).

TMDB API에서 한국 인기작 + 해외 인기/고평점작을 모아 data/movies.json 으로 저장한다.
원본 API를 매번 호출하지 않고, 한 번 수집해 로컬 JSON으로 떨어뜨린 뒤 그 파일을
색인(indexing)에 사용한다(재현성 ↑, 호출 최소화, 수집/색인 책임 분리).

사용법:
    python -m scripts.fetch_tmdb

사전 준비:
    .env 에 TMDB_API_KEY 를 채운다(themoviedb.org → 설정 → API, 무료).

수집 흐름:
1) 목록 엔드포인트에서 영화 ID 후보를 모은다.
   - discover(with_origin_country=KR) : 한국 인기작
   - movie/popular                    : 해외 인기작
   - movie/top_rated                  : 해외 고평점작
2) ID를 중복 제거한다.
3) 각 ID의 상세(append_to_response=credits,keywords)를 받아 장르명/감독/주연/국가와
   무드 인리치먼트용 키워드, 정렬 신뢰도용 vote_count, 포스터 경로를 보강한다.
4) data/movies.json 으로 저장한다.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from rag.config import settings

_BASE = "https://api.themoviedb.org/3"


def _api_get(path: str, **params) -> dict:
    """TMDB API GET 요청(쿼리 파라미터로 api_key/language 자동 첨부)."""
    params.setdefault("api_key", settings.tmdb_api_key)
    params.setdefault("language", settings.tmdb_language)
    url = f"{_BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


# 잔잔함을 해치는 장르. SF·모험·판타지까지 빼야 마션·인터스텔라 같은 대작이 걸러진다.
_NOT_QUIET_GENRES = ("액션", "스릴러", "공포", "범죄", "전쟁", "SF", "모험", "판타지")

# 이 축들은 평점순으로 뽑는다. popularity는 잔잔한 계열에서도 블록버스터를 올려
# 잡음이 섞인다(실측: 애니메이션 축 1페이지가 토이스토리·미니언즈·마리오).
_QUIET_MIN_VOTES = 500


def _quiet_axes() -> dict[str, dict]:
    """무드 스펙트럼을 넓히는 수집축. 장르 ID는 캐싱된 rag.tmdb 매핑을 쓴다."""
    from rag import tmdb

    genre = tmdb.genre_name_to_id()
    common = {
        "without_genres": ",".join(str(genre[g]) for g in _NOT_QUIET_GENRES),
        "vote_count.gte": _QUIET_MIN_VOTES,
        "sort_by": "vote_average.desc",
    }
    return {
        # 고전 드라마 명작 — 시네마 천국, 12명의 성난 사람들, 인생은 아름다워
        "드라마 평점순": {**common, "with_genres": genre["드라마"]},
        # 생활극·성장물 — 퍼펙트 데이즈, 이키루, 목소리의 형태, 룩백
        "일본 드라마": {
            **common,
            "with_genres": genre["드라마"],
            "with_origin_country": "JP",
            # 풀이 141편뿐이라 하한을 낮춰야 페이지가 채워진다.
            "vote_count.gte": settings.tmdb_min_vote_count,
        },
        # 한국 생활극 — 소원, 도가니, 20세기 소녀
        "한국 드라마": {
            **common,
            "with_genres": genre["드라마"],
            "with_origin_country": "KR",
            "vote_count.gte": settings.tmdb_min_vote_count,
        },
        "로맨스 평점순": {**common, "with_genres": genre["로맨스"]},
    }


def _collect_ids() -> list[int]:
    """목록 엔드포인트들을 돌며 영화 ID 후보를 모아 중복 제거한다."""
    ids: list[int] = []

    def add_from(path: str, pages: int, **extra) -> None:
        for page in range(1, pages + 1):
            try:
                data = _api_get(path, page=page, **extra)
            except urllib.error.HTTPError as exc:
                print(f"[fetch] {path} p{page} 실패: {exc}", file=sys.stderr)
                break
            for m in data.get("results", []):
                ids.append(m["id"])
            time.sleep(0.05)  # API 예의상 약간의 간격

    # 한국 인기작. vote_count 하한을 쿼리에 걸어 트래픽만 높은 작품을 애초에 제외한다.
    # 이게 없으면 popularity 1페이지 2위가 성인물이고, 수집분의 54%가 50표 미만이 된다.
    add_from(
        "/discover/movie",
        settings.tmdb_kr_pages,
        with_origin_country="KR",
        sort_by="popularity.desc",
        **{"vote_count.gte": settings.tmdb_min_vote_count},
    )
    # 해외 인기작 / 고평점작. 목록 엔드포인트는 vote_count.gte를 받지 않으므로
    # 하한은 _to_movie 이후 사후 필터로 건다(해외분은 저표본이 1% 미만이라 안전망 성격).
    add_from("/movie/popular", settings.tmdb_intl_pages)
    add_from("/movie/top_rated", settings.tmdb_intl_pages)

    # 잔잔한 계열. 위 세 축만으로는 액션·스릴러가 67%를 차지해 무드 검색의 대표
    # 질의("비 오는 날 볼 잔잔한 영화")에 후보가 몇 편밖에 안 남는다.
    for label, params in _quiet_axes().items():
        before = len(ids)
        add_from("/discover/movie", settings.tmdb_quiet_pages, **params)
        print(f"[fetch] 잔잔한 축 '{label}': +{len(ids) - before}건")

    # 순서 유지 중복 제거
    return list(dict.fromkeys(ids))


def _director_of(credits: dict) -> str:
    """crew 목록에서 job == 'Director' 인 사람을 찾는다."""
    for c in credits.get("crew", []):
        if c.get("job") == "Director":
            return c.get("name", "")
    return ""


def _top_cast(credits: dict, n: int = 5) -> list[str]:
    """출연진 상위 n명의 이름."""
    cast = sorted(credits.get("cast", []), key=lambda c: c.get("order", 999))
    return [c.get("name", "") for c in cast[:n] if c.get("name")]


def _country_of(detail: dict) -> str:
    """원산지 국가 코드(없으면 제작 국가 첫 항목)."""
    origin = detail.get("origin_country") or []
    if origin:
        return origin[0]
    prod = detail.get("production_countries") or []
    return prod[0]["iso_3166_1"] if prod else ""


def _keywords_of(detail: dict, n: int = 20) -> list[str]:
    """무드 인리치먼트 입력으로 쓸 TMDB 키워드 상위 n개."""
    keywords = detail.get("keywords", {}).get("keywords", [])
    return [k.get("name", "") for k in keywords[:n] if k.get("name")]


def _to_movie(detail: dict) -> dict:
    """TMDB 상세 응답을 색인에 쓸 평평한 dict로 변환한다."""
    credits = detail.get("credits", {})
    release = detail.get("release_date") or ""
    year = int(release[:4]) if release[:4].isdigit() else 0
    return {
        "movie_id": detail["id"],
        "title": detail.get("title", ""),
        "original_title": detail.get("original_title", ""),
        "year": year,
        "genres": [g["name"] for g in detail.get("genres", [])],
        "director": _director_of(credits),
        "cast": _top_cast(credits),
        "country": _country_of(detail),
        "vote_average": round(float(detail.get("vote_average", 0.0)), 1),
        # 표본이 적은 영화를 정렬 상위에서 배제하는 데 쓴다. 이게 없으면
        # "평점 가장 높은 영화"에 3표짜리 무명작이 올라온다.
        "vote_count": int(detail.get("vote_count") or 0),
        "runtime": int(detail.get("runtime") or 0),
        # 무드 프로파일 생성 시 줄거리만으로는 부족한 층위를 보완한다.
        "keywords": _keywords_of(detail),
        # 경로만 저장한다. 이미지 크기(w300/w500)는 렌더링 시점에 결정한다.
        "poster_path": detail.get("poster_path") or "",
        "overview": detail.get("overview", ""),
    }


def fetch_movies() -> list[dict]:
    """ID 수집 → 상세 보강 → movie dict 리스트."""
    if not settings.tmdb_api_key:
        raise RuntimeError(
            "TMDB_API_KEY 가 설정되지 않았습니다. .env 에 키를 채우세요 "
            "(themoviedb.org → 설정 → API)."
        )

    ids = _collect_ids()
    print(f"[fetch] 수집 대상 영화 수(중복 제거): {len(ids)}")

    movies: list[dict] = []
    skipped_no_overview = 0
    skipped_low_votes = 0
    for i, mid in enumerate(ids, 1):
        try:
            detail = _api_get(f"/movie/{mid}", append_to_response="credits,keywords")
        except urllib.error.HTTPError as exc:
            print(f"[fetch] id={mid} 상세 실패: {exc}", file=sys.stderr)
            continue
        movie = _to_movie(detail)
        # 줄거리(한국어)가 비어 있으면 검색 가치가 낮으므로 건너뛴다.
        if not movie["overview"]:
            skipped_no_overview += 1
            continue
        # 표본이 부족한 영화는 평점 정렬을 오염시키고 무드 분석 입력도 부실하다.
        # 목록 엔드포인트(popular/top_rated) 유입분을 여기서 함께 거른다.
        if movie["vote_count"] < settings.tmdb_min_vote_count:
            skipped_low_votes += 1
            continue
        movies.append(movie)
        if i % 50 == 0:
            print(f"[fetch] 진행: {i}/{len(ids)}")
        time.sleep(0.05)

    print(
        f"[fetch] 제외: 줄거리 없음 {skipped_no_overview}편 / "
        f"투표 {settings.tmdb_min_vote_count}표 미만 {skipped_low_votes}편"
    )
    print(f"[fetch] 최종 영화 수: {len(movies)}")
    return movies


def main() -> None:
    movies = fetch_movies()
    out = settings.movies_file
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(movies, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[fetch] 저장 완료: {out} ({len(movies)}편)")
    print("다음 단계: python -m scripts.enrich")


if __name__ == "__main__":
    main()
