from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest

from ui.app import POSTER_BASE_URL, poster_url


class PosterUrlTests(unittest.TestCase):
    def test_path_becomes_cdn_url(self):
        self.assertEqual(
            poster_url({"poster_path": "/abc.jpg"}), f"{POSTER_BASE_URL}/abc.jpg"
        )

    def test_missing_poster_yields_none(self):
        """포스터가 없는 영화도 있다. 빈 URL로 깨진 이미지를 띄우면 안 된다."""
        self.assertIsNone(poster_url({"poster_path": ""}))
        self.assertIsNone(poster_url({}))

    def test_size_is_decided_at_render_time(self):
        """경로만 저장하므로 w185↔w300 교체가 데이터 변경 없이 가능해야 한다."""
        self.assertIn("/t/p/", POSTER_BASE_URL)


class CineBotAppTests(unittest.TestCase):
    def test_initial_screen_renders_without_error(self):
        app = AppTest.from_file("ui/app.py", default_timeout=10).run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(
            [button.label for button in app.button[:4]],
            [
                "기생충의 감독과 줄거리를 알려줘",
                "평점이 가장 높은 영화는 무엇이야?",
                "한국 스릴러 영화를 추천해줘",
                "봉준호 감독의 다른 영화도 알려줘",
            ],
        )
        self.assertEqual(
            app.chat_input[0].placeholder,
            "영화 제목, 감독, 장르 또는 추천 조건을 입력하세요",
        )


if __name__ == "__main__":
    unittest.main()
