from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

import ui.api_client
import ui.app
from ui import history

from ui.app import (
    POSTER_BASE_URL,
    SUGGESTED_QUESTIONS,
    poster_url,
    source_card_html,
    status_html,
)


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


def card(**overrides) -> dict:
    source = {
        "title": "기생충",
        "year": 2019,
        "director": "봉준호",
        "cast": "송강호(Kim Ki-taek 역)",
        "genres": "드라마",
        "country": "한국",
        "vote_average": 8.517,
        "poster_path": "/abc.jpg",
        "snippet": "요약",
    }
    source.update(overrides)
    return source


def summary_of(markup: str) -> str:
    """접혔을 때 보이는 부분(<summary> 안)."""
    return markup.split("</summary>")[0]


def collapsed_body_of(markup: str) -> str:
    """눌러야 보이는 부분(<summary> 뒤)."""
    return markup.split("</summary>")[1]


class SourceCardTests(unittest.TestCase):
    """접었을 때는 포스터·제목·연도·평점만, 나머지는 눌러야 나온다."""

    def test_card_is_a_native_details_element(self):
        """st.expander면 클릭마다 서버 왕복과 rerun이 든다. 브라우저에 맡긴다."""
        markup = source_card_html(card())

        self.assertTrue(markup.startswith("<details"))
        self.assertIn("<summary", markup)

    def test_collapsed_view_shows_only_poster_and_identity(self):
        head = summary_of(source_card_html(card()))

        self.assertIn(f'src="{POSTER_BASE_URL}/abc.jpg"', head)
        self.assertIn("기생충", head)
        self.assertIn("⭐ 8.5", head)
        # 세로로 길어지던 원인들은 접힌 화면에 없어야 한다.
        self.assertNotIn("출연", head)
        self.assertNotIn("줄거리", head)
        self.assertNotIn("요약", head)

    def test_details_live_outside_the_summary(self):
        body = collapsed_body_of(source_card_html(card()))

        self.assertIn("감독", body)
        self.assertIn("출연", body)
        self.assertIn("요약", body)

    def test_poster_is_rendered_inline(self):
        html = source_card_html(card())

        self.assertIn(f'src="{POSTER_BASE_URL}/abc.jpg"', html)
        self.assertIn("기생충", html)

    def test_missing_poster_gets_a_placeholder(self):
        """빈 src로 깨진 이미지를 띄우면 그리드 높이가 무너진다."""
        html = source_card_html(card(poster_path=""))

        self.assertNotIn("<img", html)
        self.assertIn("cine-card-poster-empty", html)

    def test_rating_is_rounded_to_one_decimal(self):
        """엔드포인트마다 8.517/8.533으로 달라진다. 표시는 한 자리로 맞춘다."""
        self.assertIn("⭐ 8.5", source_card_html(card()))

    def test_empty_fields_are_dropped(self):
        """무드 검색 결과에는 출연진이 없다. '정보 없음' 줄만 늘리지 않는다."""
        html = source_card_html(card(cast="", director=""))

        self.assertNotIn("출연", html)
        self.assertNotIn("감독", html)
        self.assertIn("장르", html)

    def test_titles_are_escaped(self):
        """제목·줄거리는 TMDB에서 온 문자열이다. 그대로 HTML에 박으면 안 된다."""
        html = source_card_html(card(title="<script>alert(1)</script>"))

        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_status_text_is_escaped(self):
        self.assertNotIn("<b>", status_html("<b>찾는 중</b>"))


class CineBotAppTests(unittest.TestCase):
    def setUp(self):
        # 실제 대화 기록을 읽지 않게 격리한다. 안 하면 개발자 기계에 쌓인 지난
        # 대화가 사이드바에 뜨면서 버튼 목록 검사가 흔들린다(실측).
        self.tmp = tempfile.TemporaryDirectory()
        patcher = patch.object(history, "HISTORY_DIR", Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_initial_screen_renders_without_error(self):
        app = AppTest.from_file("ui/app.py", default_timeout=10).run()

        self.assertEqual(len(app.exception), 0)
        # 문구를 그대로 박지 않는다. 추천 질문은 바뀌기 마련이고, 여기서 확인할
        # 것은 "첫 화면에 추천 질문이 버튼으로 다 뜨는가"다.
        labels = [button.label for button in app.button]
        for question in SUGGESTED_QUESTIONS:
            self.assertIn(question, labels)
        self.assertEqual(
            app.chat_input[0].placeholder,
            "영화 제목, 감독, 장르 또는 추천 조건을 입력하세요",
        )

    def test_non_ascii_input_does_not_crash(self):
        """compare_digest에 str을 넘기면 한글 입력에서 TypeError가 나 화면이 죽는다."""
        with patch.object(ui.app, "PASSCODE", "cinebot-test"):
            self.assertFalse(ui.app.passcode_matches("틀린값"))
            self.assertTrue(ui.app.passcode_matches("cinebot-test"))

        with patch.object(ui.app, "PASSCODE", "열려라참깨"):
            self.assertTrue(ui.app.passcode_matches("열려라참깨"))
            self.assertFalse(ui.app.passcode_matches("열려라"))

    def test_no_passcode_means_no_lock(self):
        """기본값은 잠그지 않음. 값을 넣어야 잠긴다."""
        with patch.object(ui.app, "PASSCODE", ""):
            app = AppTest.from_file("ui/app.py", default_timeout=10).run()

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.chat_input)

    def test_passcode_blocks_the_chat_screen(self):
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        # PASSCODE는 스크립트가 실행될 때 환경 변수에서 읽힌다.
        with patch.dict(os.environ, {"CINEBOT_PASSCODE": "열려라참깨"}):
            app.run()

        # 잠긴 동안에는 채팅 입력창이 아예 만들어지지 않는다.
        self.assertEqual(len(app.chat_input), 0)
        self.assertTrue(app.text_input)

    def test_wrong_passcode_shows_an_error_and_stays_locked(self):
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        with patch.dict(os.environ, {"CINEBOT_PASSCODE": "열려라참깨"}):
            app.run()
            app.text_input[0].set_value("틀린값")
            app.button[0].click().run()

            self.assertEqual(len(app.exception), 0)
            self.assertTrue(app.error)
            self.assertEqual(len(app.chat_input), 0)

    def test_right_passcode_opens_the_chat_screen(self):
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        with patch.dict(os.environ, {"CINEBOT_PASSCODE": "열려라참깨"}):
            app.run()
            app.text_input[0].set_value("열려라참깨")
            app.button[0].click().run()

            self.assertEqual(len(app.exception), 0)
            self.assertTrue(app.session_state["unlocked"])
            self.assertTrue(app.chat_input)

    def test_past_conversation_is_read_only(self):
        """지난 대화는 볼 수만 있어야 한다 — 이어서 물어보는 길을 막는다."""
        history.save_conversation(
            "past-1",
            [
                {"role": "user", "content": "기생충 감독은?"},
                {"role": "assistant", "content": "봉준호입니다.", "sources": []},
            ],
        )
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.viewing = "past-1"
        app.run()

        self.assertEqual(len(app.exception), 0)
        # 지난 대화 내용이 보이고
        self.assertIn("봉준호입니다.", [m.value for m in app.markdown])
        # 입력은 막혀 있다
        self.assertTrue(app.chat_input[0].disabled)

    def test_opening_a_past_conversation_drops_the_unanswered_question(self):
        """남겨두면 돌아왔을 때 뒤늦게 전송돼 같은 질문에 요금이 두 번 나간다."""
        history.save_conversation(
            "past-2", [{"role": "user", "content": "지난 질문"}]
        )
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.pending = "아직 답 못 받은 질문"
        app.session_state.viewing = "past-2"
        app.run()

        self.assertEqual(len(app.exception), 0)
        self.assertIsNone(app.session_state.pending)

    def test_widgets_are_locked_while_the_answer_streams(self):
        """중단되면 LangGraph 실행이 통째로 사라진다. 눌리지 않게 막는 게 먼저다.

        답변을 받는 실행에서는 사이드바와 입력창이 잠긴 채로 그려져야 한다.
        그 실행이 끝난 뒤의 요소 트리를 확인한다 — 스트리밍 도중에 읽으면 직전
        실행의 트리를 보게 된다.
        """

        class Answering:
            def __init__(self, *args, **kwargs):
                pass

            def stream_query(self, question, session_id):
                yield {"type": "done", "answer": "답변", "sources": []}

        history.save_conversation("past-3", [{"role": "user", "content": "지난 질문"}])
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.pending = "질문"
        with patch.object(ui.api_client, "RagApiClient", Answering):
            app.run()

        self.assertEqual(len(app.exception), 0)
        # 답변이 끝나면 스스로 다시 그려 잠금을 푼다. 안 그러면 위젯이 전부
        # 잠긴 채로 남아 사용자가 다시 실행시킬 방법이 없다.
        self.assertFalse(app.chat_input[0].disabled)
        self.assertFalse(all(b.disabled for b in app.sidebar.button))
        self.assertEqual([m["role"] for m in app.session_state.messages][-1], "assistant")

    def test_widgets_stay_locked_while_the_question_is_pending(self):
        """스트리밍이 시작되기 전 실행에서도 잠겨 있어야 한다."""
        history.save_conversation("past-4", [{"role": "user", "content": "지난 질문"}])
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        # viewing을 켜면 스트리밍은 건너뛰지만 busy 판정은 그대로다.
        app.session_state.pending = "질문"
        app.session_state.viewing = "past-4"
        app.run()

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.chat_input[0].disabled)

    def test_interrupted_turn_gets_a_notice(self):
        """뚫렸을 때 질문만 남으면 화면이 멈춘 것처럼 보인다. 무슨 일인지 남긴다."""
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        # 스트리밍이 끊긴 뒤의 상태: 질문만 있고 pending은 비어 있다.
        app.session_state.messages = [{"role": "user", "content": "기생충 감독은?"}]
        app.session_state.pending = None
        app.run()

        self.assertEqual(len(app.exception), 0)
        last = app.session_state.messages[-1]
        self.assertEqual(last["role"], "assistant")
        self.assertTrue(last["error"])
        self.assertIn("답변을 받지 못했습니다", last["content"])
        # 다시 물어볼 수 있어야 한다.
        self.assertFalse(app.chat_input[0].disabled)

    def test_notice_is_not_added_while_a_question_is_pending(self):
        """답변을 받으러 가는 중인 질문에까지 안내를 붙이면 안 된다."""
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.messages = [{"role": "user", "content": "기생충 감독은?"}]
        app.session_state.pending = "기생충 감독은?"
        app.session_state.viewing = None

        class Answering:
            def __init__(self, *args, **kwargs):
                pass

            def stream_query(self, question, session_id):
                yield {"type": "done", "answer": "봉준호입니다.", "sources": []}

        with patch.object(ui.api_client, "RagApiClient", Answering):
            app.run()

        contents = [m["content"] for m in app.session_state.messages]
        self.assertNotIn("답변을 받지 못했습니다. 다시 물어봐 주세요.", contents)
        self.assertIn("봉준호입니다.", contents)

    def test_question_is_claimed_before_the_api_call(self):
        """스트리밍이 중단돼도 다시 전송되지 않도록, 부르기 전에 pending을 비운다."""
        seen: list[Any] = []

        class Recording:
            def __init__(self, *args, **kwargs):
                pass

            def stream_query(self, question, session_id):
                # API를 부르는 시점의 pending 값을 기록한다.
                seen.append(st.session_state.get("pending"))
                yield {"type": "done", "answer": "답변", "sources": []}

        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.pending = "질문"
        # ui/app.py는 실행될 때마다 `from ui.api_client import RagApiClient`로
        # 이름을 다시 가져온다. 그래서 원본 모듈 쪽을 갈아끼워야 한다.
        with patch.object(ui.api_client, "RagApiClient", Recording):
            app.run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(seen, [None])

    def test_missing_past_conversation_falls_back_to_the_live_one(self):
        """파일이 지워졌는데 화면이 죽으면 안 된다."""
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.viewing = "없는-대화"
        app.run()

        self.assertEqual(len(app.exception), 0)
        self.assertIsNone(app.session_state.viewing)
        self.assertFalse(app.chat_input[0].disabled)

    def test_welcome_disappears_once_a_conversation_starts(self):
        """첫 질문 뒤에도 히어로와 추천 버튼이 남던 문제의 회귀 방지."""
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.messages = [{"role": "user", "content": "기생충 감독은?"}]
        app.session_state.pending = None
        app.run()

        labels = [button.label for button in app.button]
        for question in SUGGESTED_QUESTIONS:
            self.assertNotIn(question, labels)


if __name__ == "__main__":
    unittest.main()
