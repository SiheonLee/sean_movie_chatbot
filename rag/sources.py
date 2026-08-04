"""
출처 카드 변환기.

도구가 찾은 영화를 UI가 그대로 렌더링할 수 있는 dict로 바꾼다. 변환기는 두 개뿐이고
반환 키가 서로 정확히 같아야 한다. 키가 어긋나면 어느 도구가 답했느냐에 따라 UI가
깨진다.

- tmdb_to_source : TMDB 응답 (search_movies, get_movie_details 공용)
- doc_to_source  : Chroma 문서 (search_by_vibe)

웹 검색 결과는 여기를 거치지 않는다. 기사·리뷰 페이지는 영화가 아니라서 이 스키마에
넣으면 빈 카드가 된다.
"""

from __future__ import annotations

from langchain_core.documents import Document

from rag import catalog, tmdb

# 카드에 넣을 주요 출연진 수. 배역명은 카드에 넣지 않는다 — 카드는 빠른 시각
# 참조용이고, 배역은 답변 본문(get_movie_details)이 담당한다.
_CAST_LIMIT = 4

# 포스터는 경로만 저장하고 크기는 렌더링 시점에 정한다. w300/w500 교체가 데이터
# 변경 없이 가능하도록.
POSTER_BASE_URL = f"{tmdb.IMAGE_BASE_URL}/w300"

_SNIPPET_LIMIT = 160

# ISO 3166-1 국가 코드 → 한국어 표기(자주 등장하는 것만; 없으면 코드 그대로).
_COUNTRY_KO = {
    "KR": "한국", "US": "미국", "JP": "일본", "GB": "영국", "FR": "프랑스",
    "CN": "중국", "HK": "홍콩", "DE": "독일", "IT": "이탈리아", "ES": "스페인",
    "IN": "인도", "CA": "캐나다", "AU": "호주", "TW": "대만", "TH": "태국",
}


def country_ko(code: str) -> str:
    return _COUNTRY_KO.get(code, code)


def _truncate(text: str) -> str:
    text = text.strip().replace("\n", " ")
    return (text[:_SNIPPET_LIMIT] + "…") if len(text) > _SNIPPET_LIMIT else text


def _rating(value: object) -> float:
    """소수점 1자리로 통일한다.

    TMDB는 소수점 3자리를 주고, 같은 영화라도 discover와 detail의 값이 다르다
    (실측: 9.317 vs 9.333). 반올림하면 둘 다 9.3이 되어 불일치가 사라진다.
    """
    return round(float(value or 0.0), 1)


def _year(release_date: str | None) -> int:
    head = (release_date or "")[:4]
    return int(head) if head.isdigit() else 0


def tmdb_to_source(movie: dict, *, detail_shape: bool = False) -> dict:
    """TMDB 응답 → 출처 카드.

    목록 응답(/discover, /search)과 상세 응답(/movie/{id})은 구조가 다르다.

        목록: genre_ids(정수 배열), credits 없음
        상세: genres(객체 배열), credits 있음

    목록 결과에는 감독·국가·출연진이 없다. 그래서 먼저 로컬 카탈로그를 조회한다
    (실측 적중률 97%). 카탈로그에 없을 때만 목록 응답의 정보로 채우며, 이때
    감독·출연은 빈 값이 된다.

    국가를 original_language로 추정하지 않는다. 'en'은 미국·영국·호주를 구분하지
    못해서 추측이 오답이 된다. 모르는 건 비워 두는 편이 낫다.
    """
    catalogued = catalog.lookup(movie.get("id"))
    if catalogued:
        return _from_catalog(catalogued)

    if detail_shape:
        genres = [g["name"] for g in movie.get("genres", [])]
        director = tmdb.director_of(movie)
        cast = ", ".join(tmdb.top_cast(movie, n=_CAST_LIMIT))
    else:
        id_to_name = tmdb.genre_id_to_name()
        genres = [id_to_name.get(i, "") for i in movie.get("genre_ids", [])]
        director = ""
        cast = ""

    return {
        "title": movie.get("title", ""),
        "year": _year(movie.get("release_date")),
        "director": director,
        "cast": cast,
        "genres": ", ".join(g for g in genres if g),
        "country": country_ko((movie.get("origin_country") or [""])[0]),
        "vote_average": _rating(movie.get("vote_average")),
        "poster_path": movie.get("poster_path") or "",
        "snippet": _truncate(movie.get("overview") or ""),
    }


def _from_catalog(movie: dict) -> dict:
    """로컬 카탈로그 항목 → 출처 카드. API 호출 없음."""
    return {
        "title": movie.get("title", ""),
        "year": int(movie.get("year") or 0),
        "director": movie.get("director") or "",
        "cast": ", ".join((movie.get("cast") or [])[:_CAST_LIMIT]),
        "genres": ", ".join(movie.get("genres") or []),
        "country": country_ko(movie.get("country") or ""),
        "vote_average": _rating(movie.get("vote_average")),
        "poster_path": movie.get("poster_path") or "",
        "snippet": _truncate(movie.get("overview") or ""),
    }


def doc_to_source(doc: Document) -> dict:
    """Chroma 문서 → 출처 카드.

    무드 검색은 카탈로그 자체를 색인한 것이라 tmdb_id 조회가 100% 적중한다.
    색인에 없는 출연진까지 카드에 채울 수 있다.
    """
    catalogued = catalog.lookup(doc.metadata.get("tmdb_id"))
    if catalogued:
        return _from_catalog(catalogued)

    meta = doc.metadata
    return {
        "title": meta.get("title", ""),
        "year": int(meta.get("year") or 0),
        "director": meta.get("director", ""),
        "cast": "",
        "genres": (meta.get("genres") or "").strip("|").replace("|", ", "),
        "country": country_ko(meta.get("country", "")),
        "vote_average": _rating(meta.get("vote_average")),
        "poster_path": meta.get("poster_path", ""),
        "snippet": _truncate(doc.page_content),
    }
