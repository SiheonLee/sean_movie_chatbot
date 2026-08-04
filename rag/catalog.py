"""
로컬 영화 카탈로그.

data/movies.json을 tmdb_id로 조회할 수 있게 메모리에 올린다. 출처 카드를 만들 때
TMDB 상세를 다시 부르지 않기 위한 계층이다.

왜 이게 통하는가:
    코퍼스는 TMDB의 popularity/top_rated/평점순에서 수집했고, 사용자 질의도 대개
    같은 축을 탄다. 실측 적중률 97%(한국 스릴러·평점높은 액션·봉준호·2020년 이후
    SF·지금 상영중·요즘 뜨는 각 상위 5편 기준 29/30).

    search_by_vibe는 이 코퍼스 자체를 검색하므로 적중률이 100%다.

이 파일이 없어도 동작해야 한다. 카탈로그는 최적화지 필수 의존이 아니다.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache

from rag.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _by_id() -> dict[int, dict]:
    """movie_id → 영화 dict. 파일이 없으면 빈 카탈로그로 동작한다."""
    try:
        movies = json.loads(settings.movies_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning(
            "로컬 카탈로그를 읽지 못했습니다: %s. 출처 카드는 TMDB 상세 조회로 "
            "채워집니다(느려집니다).",
            settings.movies_file,
        )
        return {}
    return {m["movie_id"]: m for m in movies if "movie_id" in m}


def lookup(movie_id: int | None) -> dict | None:
    """카탈로그에서 영화를 찾는다. 없으면 None."""
    return _by_id().get(movie_id) if movie_id else None
