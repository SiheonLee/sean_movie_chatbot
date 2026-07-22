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
3) 각 ID의 상세(append_to_response=credits)를 받아 장르명/감독/주연/국가를 보강한다.
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

    # 한국 인기작
    add_from(
        "/discover/movie",
        settings.tmdb_kr_pages,
        with_origin_country="KR",
        sort_by="popularity.desc",
    )
    # 해외 인기작 / 고평점작
    add_from("/movie/popular", settings.tmdb_intl_pages)
    add_from("/movie/top_rated", settings.tmdb_intl_pages)

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
    for i, mid in enumerate(ids, 1):
        try:
            detail = _api_get(f"/movie/{mid}", append_to_response="credits")
        except urllib.error.HTTPError as exc:
            print(f"[fetch] id={mid} 상세 실패: {exc}", file=sys.stderr)
            continue
        movie = _to_movie(detail)
        # 줄거리(한국어)가 비어 있으면 검색 가치가 낮으므로 건너뛴다.
        if not movie["overview"]:
            continue
        movies.append(movie)
        if i % 50 == 0:
            print(f"[fetch] 진행: {i}/{len(ids)}")
        time.sleep(0.05)

    print(f"[fetch] 줄거리가 있는 최종 영화 수: {len(movies)}")
    return movies


def main() -> None:
    movies = fetch_movies()
    out = settings.movies_file
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(movies, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[fetch] 저장 완료: {out} ({len(movies)}편)")
    print("다음 단계: python -m scripts.ingest")


if __name__ == "__main__":
    main()
