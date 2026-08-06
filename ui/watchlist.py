"""보고싶은 영화를 담아두고 다시 꺼낸다.

한 사용자당 파일 하나다. 카드를 다시 그려야 하므로 출처 카드 전체를 그대로
담는다(한 편에 1KB 남짓). 저장 방식은 ui/history.py와 같은 규율을 따른다 —
임시 파일에 쓰고 자리를 바꾸고, user_id로 경계를 긋고, 스키마 버전을 붙인다.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
WATCHLIST_DIR = Path(
    os.getenv("CINEBOT_WATCHLIST_DIR", str(ROOT_DIR / "data" / "watchlist"))
)

SCHEMA_VERSION = 1
MAX_SAVED = 200
DEFAULT_USER_ID = "local"

# 파일 이름이 되는 값이라 반드시 검사한다.
_USER_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


def movie_key(source: dict[str, Any]) -> str:
    """같은 영화인지 가르는 값.

    제목만으로는 안 된다. `괴물`(2006)과 `괴물`(2023)은 다른 영화다.
    """
    return f"{source.get('title') or ''}|{source.get('year') or ''}"


def _path(user_id: str) -> Path:
    return WATCHLIST_DIR / f"{user_id}.json"


def _read(user_id: str) -> list[dict[str, Any]]:
    if not _USER_PATTERN.match(user_id):
        return []
    try:
        record = json.loads(_path(user_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(record, dict) or record.get("version") != SCHEMA_VERSION:
        return []
    movies = record.get("movies")
    return movies if isinstance(movies, list) else []


def _write(user_id: str, movies: list[dict[str, Any]]) -> None:
    """저장에 실패해도 대화는 계속돼야 하므로 삼킨다."""
    if not _USER_PATTERN.match(user_id):
        return
    record = {
        "version": SCHEMA_VERSION,
        "user_id": user_id,
        "movies": movies[:MAX_SAVED],
    }
    try:
        WATCHLIST_DIR.mkdir(parents=True, exist_ok=True)
        temporary = _path(user_id).with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, _path(user_id))
    except OSError:
        logger.exception("보고싶은 영화를 저장하지 못했습니다 (user=%s)", user_id)


def saved_movies(*, user_id: str = DEFAULT_USER_ID) -> list[dict[str, Any]]:
    """담은 순서의 역순(최근에 담은 것이 앞)으로 출처 카드를 돌려준다."""
    return [
        movie["source"]
        for movie in _read(user_id)
        if isinstance(movie.get("source"), dict)
    ]


def saved_keys(*, user_id: str = DEFAULT_USER_ID) -> set[str]:
    return {str(movie.get("key")) for movie in _read(user_id)}


def add(source: dict[str, Any], *, user_id: str = DEFAULT_USER_ID) -> None:
    key = movie_key(source)
    movies = [m for m in _read(user_id) if m.get("key") != key]
    movies.insert(
        0,
        {
            "key": key,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "source": source,
        },
    )
    _write(user_id, movies)


def remove(key: str, *, user_id: str = DEFAULT_USER_ID) -> None:
    _write(user_id, [m for m in _read(user_id) if m.get("key") != key])


def sync(
    shown: list[dict[str, Any]],
    selected_keys: set[str],
    *,
    user_id: str = DEFAULT_USER_ID,
) -> bool:
    """한 답변에 보인 영화들에 대해서만 담기/빼기를 맞춘다.

    **그 답변에 없는 영화는 건드리지 않는다.** 여러 답변에 같은 영화가 나올 수
    있는데, 한 곳에서의 선택이 다른 곳의 기록을 지우면 안 된다.

    바뀐 게 있으면 True. 없으면 파일을 건드리지 않는다 — Streamlit은 재실행이
    잦아서 매번 쓰면 디스크만 계속 두드린다.
    """
    movies = _read(user_id)
    existing = {str(m.get("key")) for m in movies}
    shown_by_key = {movie_key(s): s for s in shown}

    add_keys = {k for k in selected_keys if k in shown_by_key} - existing
    drop_keys = (set(shown_by_key) - selected_keys) & existing
    if not add_keys and not drop_keys:
        return False

    movies = [m for m in movies if str(m.get("key")) not in drop_keys]
    now = datetime.now().isoformat(timespec="seconds")
    for key in add_keys:
        movies.insert(0, {"key": key, "saved_at": now, "source": shown_by_key[key]})
    _write(user_id, movies)
    return True


def clear(*, user_id: str = DEFAULT_USER_ID) -> None:
    _write(user_id, [])
