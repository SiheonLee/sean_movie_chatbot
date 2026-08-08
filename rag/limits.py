"""Small, process-safe request limits for the invited-user demo."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


class RequestLimitError(RuntimeError):
    """A request was rejected before any external API work started."""


class RequestLease:
    """One concurrency slot. ``release`` is safe to call more than once."""

    def __init__(self, release: Callable[[], None] | None = None) -> None:
        self._release = release
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
            if self._release is not None:
                self._release()


class UsageLimiter:
    """Persist daily user counts and per-conversation counts in SQLite."""

    def __init__(
        self,
        db_path: Path,
        *,
        daily_limit: int,
        session_limit: int,
        day: Callable[[], str] | None = None,
    ) -> None:
        if daily_limit < 1 or session_limit < 1:
            raise ValueError("question limits must be positive")
        self.db_path = Path(db_path)
        self.daily_limit = daily_limit
        self.session_limit = session_limit
        self._day = day or (lambda: datetime.now(timezone.utc).date().isoformat())
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS daily_question_usage (
                        day TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        question_count INTEGER NOT NULL,
                        PRIMARY KEY (day, user_id)
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_question_usage (
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        question_count INTEGER NOT NULL,
                        PRIMARY KEY (user_id, session_id)
                    )
                    """
                )

    def consume(self, user_id: str, session_id: str) -> None:
        """Atomically reserve one question or raise without changing either count."""
        today = self._day()
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                daily_row = connection.execute(
                    "SELECT question_count FROM daily_question_usage "
                    "WHERE day = ? AND user_id = ?",
                    (today, user_id),
                ).fetchone()
                session_row = connection.execute(
                    "SELECT question_count FROM session_question_usage "
                    "WHERE user_id = ? AND session_id = ?",
                    (user_id, session_id),
                ).fetchone()
                daily_count = daily_row[0] if daily_row else 0
                session_count = session_row[0] if session_row else 0

                if daily_count >= self.daily_limit:
                    raise RequestLimitError(
                        f"오늘 사용할 수 있는 질문 {self.daily_limit}회를 모두 사용했습니다. "
                        "UTC 자정 이후 다시 시도해주세요."
                    )
                if session_count >= self.session_limit:
                    raise RequestLimitError(
                        f"이 대화에서 사용할 수 있는 질문 {self.session_limit}회를 모두 "
                        "사용했습니다. 새 대화를 시작해주세요."
                    )

                connection.execute(
                    """
                    INSERT INTO daily_question_usage(day, user_id, question_count)
                    VALUES (?, ?, 1)
                    ON CONFLICT(day, user_id) DO UPDATE SET
                        question_count = question_count + 1
                    """,
                    (today, user_id),
                )
                connection.execute(
                    """
                    INSERT INTO session_question_usage(user_id, session_id, question_count)
                    VALUES (?, ?, 1)
                    ON CONFLICT(user_id, session_id) DO UPDATE SET
                        question_count = question_count + 1
                    """,
                    (user_id, session_id),
                )


class RequestGate:
    """Apply the process-local concurrency cap before consuming persistent quota."""

    def __init__(self, usage: UsageLimiter, *, max_concurrent: int) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")
        self.usage = usage
        self._slots = threading.BoundedSemaphore(max_concurrent)

    def admit(self, user_id: str, session_id: str) -> RequestLease:
        if not self._slots.acquire(blocking=False):
            raise RequestLimitError(
                "현재 다른 답변을 생성 중입니다. 잠시 후 다시 시도해주세요."
            )
        lease = RequestLease(self._slots.release)
        try:
            self.usage.consume(user_id, session_id)
        except Exception:
            lease.release()
            raise
        return lease
