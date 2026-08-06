from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ui import watchlist


def source(title: str, year: int = 2019) -> dict:
    return {"title": title, "year": year, "poster_path": "/x.jpg", "vote_average": 8.0}


class WatchlistTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patcher = patch.object(watchlist, "WATCHLIST_DIR", Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_added_movie_comes_back_with_its_card(self):
        watchlist.add(source("기생충"))

        saved = watchlist.saved_movies()

        self.assertEqual(len(saved), 1)
        # 카드를 다시 그려야 하므로 포스터까지 살아 있어야 한다.
        self.assertEqual(saved[0]["poster_path"], "/x.jpg")

    def test_same_title_different_year_is_a_different_movie(self):
        watchlist.add(source("괴물", 2006))
        watchlist.add(source("괴물", 2023))

        self.assertEqual(len(watchlist.saved_movies()), 2)

    def test_adding_twice_does_not_duplicate(self):
        watchlist.add(source("기생충"))
        watchlist.add(source("기생충"))

        self.assertEqual(len(watchlist.saved_movies()), 1)

    def test_remove(self):
        watchlist.add(source("기생충"))
        watchlist.remove(watchlist.movie_key(source("기생충")))

        self.assertEqual(watchlist.saved_movies(), [])

    def test_sync_adds_and_drops_only_what_was_shown(self):
        """다른 답변에서 담은 영화를 여기서 지우면 안 된다."""
        watchlist.add(source("옥자", 2017))
        shown = [source("기생충"), source("올드보이", 2003)]

        changed = watchlist.sync(shown, {watchlist.movie_key(source("기생충"))})

        titles = {m["title"] for m in watchlist.saved_movies()}
        self.assertTrue(changed)
        self.assertEqual(titles, {"옥자", "기생충"})

    def test_sync_drops_deselected(self):
        watchlist.add(source("기생충"))
        shown = [source("기생충")]

        changed = watchlist.sync(shown, set())

        self.assertTrue(changed)
        self.assertEqual(watchlist.saved_movies(), [])

    def test_sync_is_a_noop_when_nothing_changed(self):
        """Streamlit은 재실행이 잦다. 바뀐 게 없으면 파일을 건드리지 않는다."""
        watchlist.add(source("기생충"))
        before = (watchlist.WATCHLIST_DIR / "local.json").stat().st_mtime_ns

        changed = watchlist.sync(
            [source("기생충")], {watchlist.movie_key(source("기생충"))}
        )

        after = (watchlist.WATCHLIST_DIR / "local.json").stat().st_mtime_ns
        self.assertFalse(changed)
        self.assertEqual(before, after)

    def test_other_users_list_is_separate(self):
        watchlist.add(source("기생충"), user_id="someone")

        self.assertEqual(watchlist.saved_movies(), [])
        self.assertEqual(len(watchlist.saved_movies(user_id="someone")), 1)

    def test_bad_user_id_is_rejected(self):
        watchlist.add(source("기생충"), user_id="../escape")

        self.assertEqual(list(watchlist.WATCHLIST_DIR.glob("*.json")), [])

    def test_broken_file_reads_as_empty(self):
        watchlist.add(source("기생충"))
        (watchlist.WATCHLIST_DIR / "local.json").write_text("{망가짐", encoding="utf-8")

        self.assertEqual(watchlist.saved_movies(), [])

    def test_cap_is_enforced(self):
        with patch.object(watchlist, "MAX_SAVED", 3):
            for index in range(5):
                watchlist.add(source(f"영화{index}"))

        saved = watchlist.saved_movies()
        self.assertEqual(len(saved), 3)
        # 최근에 담은 것이 남는다.
        self.assertEqual(saved[0]["title"], "영화4")


if __name__ == "__main__":
    unittest.main()
