"""익명 신원과 사용자 격리 테스트.

격리가 깨지면 남의 대화가 사이드바에 뜨고, 한 사람의 '모두 삭제'가 모두의
기록을 지운다. 조용히 일어나는 종류의 사고라 여기서 붙잡는다.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ui import history, identity, watchlist


class UserIdTests(unittest.TestCase):
    def test_a_fresh_id_is_usable(self):
        self.assertTrue(identity.is_valid(identity.new_user_id()))

    def test_ids_are_not_guessable(self):
        """남의 id를 찍어서 맞힐 수 있으면 격리가 아니다."""
        ids = {identity.new_user_id() for _ in range(200)}

        self.assertEqual(len(ids), 200)
        self.assertTrue(all(len(i) == 32 for i in ids))

    def test_a_good_cookie_is_kept(self):
        """같은 브라우저로 돌아오면 제 기록을 다시 봐야 한다."""
        existing = identity.new_user_id()

        self.assertEqual(identity.resolve_user_id(existing), existing)

    def test_a_missing_cookie_gets_a_new_id(self):
        self.assertTrue(identity.is_valid(identity.resolve_user_id(None)))

    def test_a_tampered_cookie_is_thrown_away(self):
        """쿠키는 사용자가 고칠 수 있다. **파일 이름이 되는 값이라 더 위험하다.**"""
        for bad in (
            "../../etc/passwd",
            "local",
            "",
            "a" * 33,
            "ABCDEF01234567890123456789012345",  # 대문자는 우리 형식이 아니다
            "../" + "a" * 29,
            123,
            None,
            object(),
        ):
            with self.subTest(bad=bad):
                self.assertFalse(identity.is_valid(bad))
                # 그래도 화면은 떠야 한다. 새 신원을 발급해 계속 간다.
                self.assertTrue(identity.is_valid(identity.resolve_user_id(bad)))


class CookieScriptTests(unittest.TestCase):
    def test_the_script_writes_the_id(self):
        script = identity.remember_user_id_script("a" * 32)

        self.assertIn(identity.COOKIE_NAME, script)
        self.assertIn("a" * 32, script)

    def test_the_cookie_outlives_the_session(self):
        """세션 쿠키로 두면 브라우저를 닫는 순간 기록을 잃는다."""
        script = identity.remember_user_id_script("a" * 32)

        self.assertIn(f"max-age={identity.COOKIE_MAX_AGE}", script)
        self.assertGreaterEqual(identity.COOKIE_MAX_AGE, 60 * 60 * 24 * 30)

    def test_it_does_not_leak_across_sites(self):
        self.assertIn("SameSite=Lax", identity.remember_user_id_script("a" * 32))

    def test_secure_is_decided_at_run_time(self):
        """http에서 Secure를 붙이면 브라우저가 쿠키를 버려 매번 새 사람이 된다."""
        script = identity.remember_user_id_script("a" * 32)

        self.assertIn('protocol === "https:"', script)


class IsolationTests(unittest.TestCase):
    """두 사람이 서로의 것을 보지 못해야 한다."""

    def setUp(self):
        self.history_tmp = tempfile.TemporaryDirectory()
        self.watchlist_tmp = tempfile.TemporaryDirectory()
        for module, name, tmp in (
            (history, "HISTORY_DIR", self.history_tmp),
            (watchlist, "WATCHLIST_DIR", self.watchlist_tmp),
        ):
            patcher = patch.object(module, name, Path(tmp.name))
            patcher.start()
            self.addCleanup(patcher.stop)
            self.addCleanup(tmp.cleanup)

        self.민수 = identity.new_user_id()
        self.영희 = identity.new_user_id()

    def movie(self, title: str) -> dict:
        return {"title": title, "year": 2019, "vote_average": 8.5, "poster_path": "/x.jpg"}

    def test_conversations_stay_with_their_owner(self):
        history.save_conversation(
            "c1", [{"role": "user", "content": "민수의 질문"}], user_id=self.민수
        )

        self.assertEqual(len(history.list_conversations(user_id=self.민수)), 1)
        self.assertEqual(history.list_conversations(user_id=self.영희), [])
        self.assertIsNone(history.load_conversation("c1", user_id=self.영희))

    def test_one_person_cannot_delete_anothers_conversation(self):
        history.save_conversation(
            "c1", [{"role": "user", "content": "민수의 질문"}], user_id=self.민수
        )

        self.assertFalse(history.delete_conversation("c1", user_id=self.영희))
        self.assertIsNotNone(history.load_conversation("c1", user_id=self.민수))

    def test_delete_all_only_empties_your_own(self):
        """실측 위험: 한 사람의 '모두 삭제'가 모두의 기록을 지우던 구조였다."""
        history.save_conversation("c1", [{"role": "user", "content": "민수"}], user_id=self.민수)
        history.save_conversation("c2", [{"role": "user", "content": "영희"}], user_id=self.영희)

        self.assertEqual(history.delete_all_conversations(user_id=self.민수), 1)
        self.assertEqual(len(history.list_conversations(user_id=self.영희)), 1)

    def test_watchlists_stay_apart(self):
        watchlist.add(self.movie("기생충"), user_id=self.민수)

        self.assertEqual(len(watchlist.saved_movies(user_id=self.민수)), 1)
        self.assertEqual(watchlist.saved_movies(user_id=self.영희), [])

    def test_removing_a_movie_touches_only_your_shelf(self):
        watchlist.add(self.movie("기생충"), user_id=self.민수)
        watchlist.add(self.movie("기생충"), user_id=self.영희)

        watchlist.remove("기생충|2019", user_id=self.민수)

        self.assertEqual(watchlist.saved_movies(user_id=self.민수), [])
        self.assertEqual(len(watchlist.saved_movies(user_id=self.영희)), 1)

    def test_each_person_gets_the_full_quota(self):
        """한도는 사람마다 센다. 남이 채운 만큼 내 자리가 줄면 안 된다."""
        with patch.object(history, "MAX_CONVERSATIONS", 2):
            for i in range(3):
                history.save_conversation(
                    f"m{i}", [{"role": "user", "content": f"질문{i}"}], user_id=self.민수
                )
            history.save_conversation(
                "y1", [{"role": "user", "content": "영희 질문"}], user_id=self.영희
            )

        self.assertEqual(len(history.list_conversations(user_id=self.민수)), 2)
        self.assertEqual(len(history.list_conversations(user_id=self.영희)), 1)


if __name__ == "__main__":
    unittest.main()
