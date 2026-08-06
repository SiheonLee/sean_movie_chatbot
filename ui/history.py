"""지난 대화를 파일에 남기고 다시 읽는다.

**저장하는 것은 화면에 그린 내용뿐이다.** 텍스트와 출처 카드만 남기고, LLM이 보는
대화 상태(LangGraph 체크포인트)는 저장하지 않는다. 지난 대화는 읽기 전용이라
이어서 물어볼 일이 없고, 그래서 두 저장소의 정합성을 맞출 필요도 없다.

한 대화가 파일 하나다. JSON이라 문제가 생기면 그냥 열어보면 된다.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
HISTORY_DIR = Path(
    os.getenv("CINEBOT_HISTORY_DIR", str(ROOT_DIR / "data" / "conversations"))
)

# 레코드 형식이 바뀌면 올린다. 옛 파일을 만났을 때 무시할지 옮길지 판단하는 근거다.
SCHEMA_VERSION = 1

# 최근 몇 개까지 남길지. 어시스턴트 한 턴이 출처 5장 포함 약 2.7KB라 넉넉한 값이다.
MAX_CONVERSATIONS = 50

# 지금은 사용자가 한 명이라 고정값이다. 멀티 유저가 되면 이 자리에 실제 신원이
# 들어가고, 파일도 사용자별로 나뉜다. 자리를 미리 비워두면 그때 마이그레이션이
# 거의 없다.
DEFAULT_USER_ID = "local"

_TITLE_LIMIT = 40
# 파일 이름이 되는 값이라 반드시 검사한다. 검사 없이 경로에 붙이면 "../"로
# 저장소 밖 파일을 건드릴 수 있다.
_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


@dataclass(frozen=True)
class ConversationSummary:
    """사이드바 목록에 필요한 것만."""

    conversation_id: str
    title: str
    updated_at: str
    turns: int


def _is_valid_id(conversation_id: str) -> bool:
    return bool(_ID_PATTERN.match(conversation_id))


def _path(conversation_id: str) -> Path:
    return HISTORY_DIR / f"{conversation_id}.json"


def conversation_title(messages: list[dict[str, Any]]) -> str:
    """목록에 보여줄 이름. 첫 질문을 그대로 쓴다."""
    for message in messages:
        if message.get("role") == "user":
            title = " ".join((message.get("content") or "").split())
            if title:
                return title[:_TITLE_LIMIT]
    return "제목 없는 대화"


def save_conversation(
    conversation_id: str,
    messages: list[dict[str, Any]],
    *,
    user_id: str = DEFAULT_USER_ID,
) -> None:
    """대화를 덮어쓴다. 저장에 실패해도 대화 자체는 계속돼야 하므로 삼킨다."""
    if not _is_valid_id(conversation_id) or not messages:
        return

    record = {
        "version": SCHEMA_VERSION,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "title": conversation_title(messages),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "messages": messages,
    }
    try:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        # 임시 파일에 다 쓰고 나서 자리를 바꾼다. Streamlit은 재실행이 잦아서
        # 쓰는 도중에 끊기면 반쪽짜리 JSON이 남는다. os.replace는 원자적이다.
        temporary = _path(conversation_id).with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary, _path(conversation_id))
    except OSError:
        logger.exception("대화 기록을 저장하지 못했습니다 (id=%s)", conversation_id)
        return

    _prune(user_id=user_id)


def load_conversation(
    conversation_id: str, *, user_id: str = DEFAULT_USER_ID
) -> list[dict[str, Any]] | None:
    """저장된 대화의 메시지. 없거나 읽을 수 없으면 None."""
    if not _is_valid_id(conversation_id):
        return None
    record = _read(_path(conversation_id))
    if record is None or record.get("user_id") != user_id:
        return None
    messages = record.get("messages")
    return messages if isinstance(messages, list) else None


def list_conversations(
    *, user_id: str = DEFAULT_USER_ID
) -> list[ConversationSummary]:
    """최근에 갱신된 것부터."""
    summaries: list[ConversationSummary] = []
    for path in _record_paths():
        record = _read(path)
        if record is None or record.get("user_id") != user_id:
            continue
        messages = record.get("messages") or []
        summaries.append(
            ConversationSummary(
                conversation_id=str(record.get("conversation_id") or path.stem),
                title=str(record.get("title") or "제목 없는 대화"),
                updated_at=str(record.get("updated_at") or ""),
                turns=sum(1 for m in messages if m.get("role") == "user"),
            )
        )
    summaries.sort(key=lambda s: s.updated_at, reverse=True)
    return summaries


def _record_paths() -> list[Path]:
    if not HISTORY_DIR.exists():
        return []
    return sorted(HISTORY_DIR.glob("*.json"))


def _read(path: Path) -> dict[str, Any] | None:
    """읽을 수 없는 파일 하나가 목록 전체를 막지 않게 한다."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("대화 기록을 읽지 못했습니다: %s", path.name)
        return None
    if not isinstance(record, dict) or record.get("version") != SCHEMA_VERSION:
        return None
    return record


def _prune(*, user_id: str) -> None:
    """오래된 것부터 지워 MAX_CONVERSATIONS개만 남긴다."""
    summaries = list_conversations(user_id=user_id)
    for summary in summaries[MAX_CONVERSATIONS:]:
        try:
            _path(summary.conversation_id).unlink(missing_ok=True)
        except OSError:
            logger.warning("오래된 기록을 지우지 못했습니다: %s", summary.conversation_id)
