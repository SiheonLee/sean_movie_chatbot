from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag.limits import RequestGate, RequestLimitError, UsageLimiter


class UsageLimiterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.db_path = Path(self.temporary.name) / "limits.sqlite"

    def limiter(self, *, daily: int = 3, session: int = 2, day=None) -> UsageLimiter:
        return UsageLimiter(
            self.db_path,
            daily_limit=daily,
            session_limit=session,
            day=day,
        )

    def test_session_limit_rejects_without_consuming_daily_quota(self):
        limiter = self.limiter(daily=3, session=2)
        limiter.consume("user-1", "session-1")
        limiter.consume("user-1", "session-1")

        with self.assertRaisesRegex(RequestLimitError, "새 대화"):
            limiter.consume("user-1", "session-1")

        # 거절된 요청이 일일 한도까지 먹지 않았으므로 다른 대화의 한 번은 된다.
        limiter.consume("user-1", "session-2")
        with self.assertRaisesRegex(RequestLimitError, "UTC 자정"):
            limiter.consume("user-1", "session-2")

    def test_daily_limit_is_shared_across_sessions_and_persists(self):
        self.limiter(daily=2, session=2).consume("user-1", "session-1")
        restarted = self.limiter(daily=2, session=2)
        restarted.consume("user-1", "session-2")

        with self.assertRaisesRegex(RequestLimitError, "오늘"):
            restarted.consume("user-1", "session-3")

        # 다른 익명 사용자에게는 별도 한도가 있다.
        restarted.consume("user-2", "session-3")

    def test_daily_limit_resets_on_the_next_utc_day(self):
        today = ["2026-08-07"]
        limiter = self.limiter(daily=1, session=2, day=lambda: today[0])
        limiter.consume("user-1", "session-1")
        with self.assertRaises(RequestLimitError):
            limiter.consume("user-1", "session-2")

        today[0] = "2026-08-08"
        limiter.consume("user-1", "session-2")


class RequestGateTests(unittest.TestCase):
    def test_concurrency_rejection_does_not_consume_quota(self):
        with tempfile.TemporaryDirectory() as directory:
            usage = UsageLimiter(
                Path(directory) / "limits.sqlite",
                daily_limit=2,
                session_limit=2,
                day=lambda: "2026-08-07",
            )
            gate = RequestGate(usage, max_concurrent=1)
            first = gate.admit("user-1", "session-1")

            with self.assertRaisesRegex(RequestLimitError, "다른 답변"):
                gate.admit("user-1", "session-1")

            first.release()
            second = gate.admit("user-1", "session-1")
            second.release()
            with self.assertRaisesRegex(RequestLimitError, "오늘"):
                gate.admit("user-1", "session-2")


if __name__ == "__main__":
    unittest.main()
