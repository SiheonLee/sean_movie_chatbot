"""
TMDB API 클라이언트.

도구(rag/tools.py)가 쓰는 얇은 조회 계층이다. LLM에게 노출되는 개념(장르명, 인물명,
OTT 서비스명)을 TMDB가 요구하는 ID로 바꾸는 책임이 여기에 있다.

설계 요점:
- httpx.Client를 모듈 수준에 하나만 두어 커넥션을 재사용한다. 질의마다 TMDB 호출이
  붙으므로 핸드셰이크 비용이 그대로 응답 지연이 된다.
- ID 매핑(장르/프로바이더/인명)과 영화 상세는 lru_cache로 프로세스 수명 내내 재사용한다.
  거의 변하지 않는 데이터이고, 매핑을 위해 매번 왕복하면 도구 호출 한 번이 3~4배가 된다.
- append_to_response로 상세·크레딧·시청처를 한 번에 받는다. 따로 부르면 요청이 3배다.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import httpx

from rag.config import settings

BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p"

# 국내 시청처 조회 기준 지역. TMDB는 국가별로 편성이 완전히 다르다.
WATCH_REGION = "KR"

_client = httpx.Client(
    base_url=BASE_URL,
    params={
        "api_key": settings.tmdb_api_key,
        "language": settings.tmdb_language,
    },
    timeout=10.0,
)


class TmdbError(RuntimeError):
    """TMDB 응답이 실패했을 때. 도구 경계에서 사용자용 메시지로 변환한다."""


def _get(path: str, **params: Any) -> dict:
    """TMDB GET. 클라이언트 기본 params(api_key/language)와 자동 병합된다."""
    try:
        response = _client.get(path, params=params)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise TmdbError(
            f"TMDB 요청 실패({exc.response.status_code}): {path}"
        ) from exc
    except httpx.HTTPError as exc:
        raise TmdbError(f"TMDB 연결 실패: {path}") from exc
    return response.json()


# --- ID 매핑 (LLM이 쓰는 이름 → TMDB가 요구하는 ID) ------------------------


@lru_cache(maxsize=1)
def genre_name_to_id() -> dict[str, int]:
    """장르명 → 장르 ID. `with_genres=액션`은 통하지 않는다."""
    data = _get("/genre/movie/list")
    return {g["name"]: g["id"] for g in data.get("genres", [])}


@lru_cache(maxsize=1)
def genre_id_to_name() -> dict[int, str]:
    """장르 ID → 장르명. 목록 응답의 genre_ids를 복원할 때 쓴다."""
    return {v: k for k, v in genre_name_to_id().items()}


# TMDB의 프로바이더명은 영문이지만 사용자와 LLM은 한국어로 말한다.
# 한국어 표기 → TMDB provider_name. 값은 provider_name_to_id()로 다시 ID가 된다.
_PROVIDER_ALIASES = {
    "넷플릭스": "Netflix",
    "왓챠": "Watcha",
    "웨이브": "wavve",
    "티빙": "TVING",
    "디즈니플러스": "Disney Plus",
    "디즈니+": "Disney Plus",
    "애플티비": "Apple TV",
    "아마존프라임": "Amazon Prime Video",
    "프라임비디오": "Amazon Prime Video",
    "왓챠피디아": "Watcha",
    "크런치롤": "Crunchyroll",
}


@lru_cache(maxsize=1)
def provider_name_to_id() -> dict[str, int]:
    """국내 OTT 서비스명 → 프로바이더 ID. 하드코딩하면 서비스 개편에 깨진다."""
    data = _get("/watch/providers/movie", watch_region=WATCH_REGION)
    return {p["provider_name"]: p["provider_id"] for p in data.get("results", [])}


# 사용자와 LLM은 "한국 영화"라고 말하고 TMDB는 ISO 3166-1 코드를 받는다.
# 장르·프로바이더와 같은 이름→ID 해석 계층이다.
_COUNTRY_ALIASES = {
    "한국": "KR", "대한민국": "KR", "국내": "KR", "한국영화": "KR",
    "미국": "US", "할리우드": "US",
    "일본": "JP", "중국": "CN", "홍콩": "HK", "대만": "TW",
    "영국": "GB", "프랑스": "FR", "독일": "DE", "이탈리아": "IT",
    "스페인": "ES", "인도": "IN", "태국": "TH", "캐나다": "CA",
    "호주": "AU", "러시아": "RU", "스웨덴": "SE", "덴마크": "DK",
}


def resolve_country_code(name: str) -> str | None:
    """'한국' → 'KR'. 이미 코드면 그대로 통과시킨다."""
    cleaned = name.strip()
    if cleaned in _COUNTRY_ALIASES:
        return _COUNTRY_ALIASES[cleaned]
    upper = cleaned.upper()
    return upper if len(upper) == 2 and upper.isalpha() else None


def supported_country_names() -> str:
    return ", ".join(sorted({k for k in _COUNTRY_ALIASES if len(k) > 1}))


def resolve_provider_id(name: str) -> int | None:
    """'넷플릭스', 'Netflix', 'netflix' 어느 쪽으로 와도 ID를 찾는다.

    쿠팡플레이처럼 TMDB 국내 목록에 아예 없는 서비스는 None이다. 도구는 이때
    '지원하지 않는다'가 아니라 '확인되지 않는다'로 답해야 한다.
    """
    providers = provider_name_to_id()
    canonical = _PROVIDER_ALIASES.get(name.strip(), name.strip())
    if canonical in providers:
        return providers[canonical]
    folded = canonical.casefold()
    return next(
        (pid for pname, pid in providers.items() if pname.casefold() == folded),
        None,
    )


@lru_cache(maxsize=512)
def find_person_id(name: str) -> int | None:
    """인명 → person_id. with_crew/with_people은 이름을 받지 않아 2-hop이 필요하다.

    동명이인은 TMDB의 인기순 1위를 택한다.
    """
    results = _get("/search/person", query=name).get("results", [])
    return results[0]["id"] if results else None


# --- 조회 -----------------------------------------------------------------


def discover(**params: Any) -> dict:
    """조건 조회. 필터와 정렬을 한 엔드포인트에서 함께 받는다."""
    return _get("/discover/movie", **params)


def search_by_title(title: str) -> dict:
    """제목 검색. 감독 정보는 주지 않으므로 상세를 한 번 더 불러야 한다."""
    return _get("/search/movie", query=title)


def list_endpoint(path: str, **params: Any) -> dict:
    """/movie/now_playing, /movie/upcoming, /trending/movie/week 등."""
    return _get(path, **params)


@lru_cache(maxsize=512)
def movie_detail(movie_id: int) -> dict:
    """상세 + 크레딧 + 국내 시청처를 한 번의 요청으로 받는다.

    시청처는 `detail["watch/providers"]`로 접근한다. 키에 슬래시가 들어간다.
    """
    return _get(
        f"/movie/{movie_id}",
        append_to_response="credits,watch/providers",
    )


# --- 상세 응답에서 자주 쓰는 조각 ------------------------------------------


def director_of(detail: dict) -> str:
    """상세 응답의 crew에서 감독을 찾는다. 목록 응답에는 crew가 없다."""
    crew = detail.get("credits", {}).get("crew", [])
    return next((c["name"] for c in crew if c.get("job") == "Director"), "")


def top_cast(detail: dict, n: int = 5) -> list[str]:
    """주요 출연진을 '배우(배역)' 형태로.

    배역명을 빼고 배우 이름만 주면 LLM이 줄거리에서 인물명을 긁어와 제멋대로
    짝짓는다(실측: '이선균 - 기정: 기택의 아내'. 실제 이선균은 박동익 역).
    TMDB가 character를 주고 있으므로 반드시 함께 넘긴다.

    배역명은 로마자로 오는 경우가 많지만(Kim Ki-taek), 없는 것보다 낫다.
    """
    entries = []
    for member in detail.get("credits", {}).get("cast", [])[:n]:
        name = member.get("name")
        if not name:
            continue
        character = (member.get("character") or "").strip()
        entries.append(f"{name}({character} 역)" if character else name)
    return entries


def flatrate_providers(detail: dict) -> list[str]:
    """국내 구독형(정액제) 시청처.

    JustWatch 제공 데이터라 국내 서비스 커버리지가 완전하지 않다. 결과가 비어 있어도
    '제공하지 않는다'가 아니라 '확인되지 않는다'로 다뤄야 한다.
    """
    region = detail.get("watch/providers", {}).get("results", {}).get(WATCH_REGION, {})
    return [p["provider_name"] for p in region.get("flatrate", [])]
