from __future__ import annotations

import json
import os
import random
import re
import tempfile
import unittest
from pathlib import Path
from datetime import date
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

import ui.api_client
import ui.app
from ui import history, identity, watchlist

from ui.app import (
    CHAT_VIEW,
    attribution_html,
    movie_labels,
    history_container_key,
    PAST_PAGE_SIZE,
    POSTER_BASE_URL,
    SUGGESTED_QUESTIONS,
    SUGGESTION_COUNT,
    WATCHLIST_VIEW,
    past_id,
    past_view,
    pick_suggestions,
    poster_url,
    resolved_view,
    scroll_to_bottom_script,
    SHELF_POSTER_BASE_URL,
    shelf_card_html,
    source_card_html,
    sources_html,
    status_html,
    unique_sources,
    watchlist_filename,
    watchlist_markdown,
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


def shelf_cards(app) -> list[str]:
    """보관함 카드의 HTML만. 스타일시트도 st.markdown으로 들어오므로 걸러낸다."""
    return [
        m.value
        for m in app.markdown
        if (m.value or "").startswith('<div class="cine-shelf-card"')
    ]


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
        # 상자는 남는다. CSS가 여기에 🎬를 깔고 자리를 지킨다.
        self.assertIn('class="cine-card-poster"', html)

    def test_a_poster_sits_inside_the_placeholder_box(self):
        """이미지가 404여도 상자가 남아야 🎬가 대신 보인다.

        onerror로 갈아끼울 수 없어서(Streamlit이 인라인 핸들러를 지운다) 상자를
        깔고 그 위에 이미지를 덮는 구조다.
        """
        html = source_card_html(card())

        self.assertIn('class="cine-card-poster"', html)
        self.assertIn('class="cine-card-poster-img"', html)

    def test_the_box_carries_the_name_not_the_image(self):
        """깨진 이미지의 대체 텍스트가 🎬 위에 겹쳐 보이면 안 된다."""
        html = source_card_html(card())

        self.assertIn('aria-label="기생충 포스터"', html)
        self.assertIn('alt=""', html)

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

    def test_card_group_preserves_source_order(self):
        markup = sources_html(
            [card(title="옥자", year=2017), card(title="기생충", year=2019)]
        )

        self.assertLess(markup.index("옥자"), markup.index("기생충"))

    def test_status_text_is_escaped(self):
        self.assertNotIn("<b>", status_html("<b>찾는 중</b>"))


class AttributionTests(unittest.TestCase):
    """답변 밑의 출처 표기. 무엇을 보고 답했는지 답변만으로는 알 수 없다."""

    def test_labels_are_readable_not_identifiers(self):
        markup = attribution_html(["tmdb", "web"])

        self.assertIn("TMDB", markup)
        self.assertIn("웹 검색", markup)
        self.assertNotIn("tmdb", markup)

    def test_justwatch_is_named_in_full(self):
        """TMDB의 시청처 데이터는 JustWatch 제공이고 표기가 의무다."""
        self.assertIn("JustWatch", attribution_html(["tmdb", "justwatch"]))

    def test_order_follows_the_given_list(self):
        markup = attribution_html(["tmdb", "web", "justwatch"])

        self.assertLess(markup.index("TMDB"), markup.index("웹 검색"))
        self.assertLess(markup.index("웹 검색"), markup.index("JustWatch"))

    def test_nothing_to_credit_draws_nothing(self):
        """도구 없이 답한 턴에 빈 상자를 남기지 않는다."""
        self.assertEqual(attribution_html([]), "")

    def test_unknown_marks_are_dropped_silently(self):
        """서버가 새 출처를 추가해도 화면이 깨지지 않아야 한다."""
        markup = attribution_html(["tmdb", "새로운출처"])

        self.assertIn("TMDB", markup)
        self.assertNotIn("새로운출처", markup)

    def test_web_source_url_is_rendered_as_a_clickable_link(self):
        markup = attribution_html(
            ["web"],
            [{"title": "기생충 평가", "url": "https://example.com/review"}],
        )

        self.assertIn('href="https://example.com/review"', markup)
        self.assertIn("기생충 평가", markup)
        self.assertIn('rel="noopener noreferrer"', markup)

    def test_non_http_web_source_url_is_not_rendered(self):
        markup = attribution_html(
            ["web"],
            [{"title": "위험한 링크", "url": "javascript:alert(1)"}],
        )

        self.assertNotIn("javascript:", markup)
        self.assertNotIn("위험한 링크", markup)


# 주석을 걷어낸 스타일시트. 주석 안에도 `margin-bottom: -16px` 같은 문장이
# 있어서(그 값을 설명하는 주석이다) 그대로 두면 파서가 그것을 선언으로 읽는다.
STYLES = re.sub(
    r"/\*.*?\*/",
    "",
    (Path(__file__).resolve().parent.parent / "ui" / "styles.css").read_text(
        encoding="utf-8"
    ),
    flags=re.S,
)


def rule_body(selector: str) -> str:
    """styles.css에서 그 셀렉터 블록의 본문."""
    match = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", STYLES)
    assert match, f"{selector} 규칙을 찾지 못했습니다"
    return match.group(1)


def bottom_margin(selector: str) -> str | None:
    """축약형(margin: a b c)과 개별 지정(margin-bottom: c) 둘 다에서 아래 여백."""
    body = rule_body(selector)
    single = re.search(r"margin-bottom:\s*([^;]+);", body)
    if single:
        return single.group(1).strip()
    shorthand = re.search(r"(?<!-)\bmargin:\s*([^;]+);", body)
    if not shorthand:
        return None
    parts = shorthand.group(1).split()
    # 1개=사방, 2개=세로/가로, 3개 이상=위/가로/아래
    return parts[0] if len(parts) == 1 else parts[2] if len(parts) >= 3 else parts[0]


class MarkdownContainerOffsetTests(unittest.TestCase):
    """Streamlit은 stMarkdownContainer에 margin-bottom: -16px를 건다.

    마지막 자식이 제 몫의 아래 여백을 가진 <p>라고 가정한 값이다. 우리가 넣는
    커스텀 블록은 <p>가 아니라서 그 음수가 그대로 남고, 요소가 아래로 끌려가
    말풍선 바닥을 파고든다(실측: 출처 표기가 말풍선 아래 1px까지 내려왔다).
    """

    #  st.markdown(unsafe_allow_html=True)으로 말풍선 안에 넣는 블록들
    CUSTOM_BLOCKS = (".cine-attribution", ".cine-cards", ".cine-thinking")

    def test_every_custom_block_offsets_the_negative_margin(self):
        for selector in self.CUSTOM_BLOCKS:
            with self.subTest(selector=selector):
                self.assertEqual(
                    bottom_margin(selector),
                    "1rem",
                    f"{selector}에 -16px 상쇄가 없습니다. 말풍선 바닥에 붙습니다.",
                )

    def test_the_offset_matches_what_streamlit_subtracts(self):
        """상쇄값은 16px여야 한다. 스케일 변수로 바꾸면 조용히 어긋난다."""
        self.assertEqual(bottom_margin(".cine-attribution"), "1rem")


class ChipLabelTests(unittest.TestCase):
    """칩 이름이 같으면 st.pills가 두 영화를 가르지 못한다."""

    def test_plain_titles_stay_plain(self):
        labels = movie_labels([card(title="기생충", year=2019), card(title="옥자", year=2017)])

        self.assertEqual(set(labels.values()), {"기생충", "옥자"})

    def test_same_title_gets_its_year(self):
        """올드보이 원작과 리메이크. 이름이 같으면 화면에서도 고를 수 없다."""
        labels = movie_labels([card(title="올드보이", year=2003), card(title="올드보이", year=2013)])

        self.assertEqual(sorted(labels.values()), ["올드보이 (2003)", "올드보이 (2013)"])

    def test_only_the_clashing_title_is_annotated(self):
        """구분이 필요한 만큼만 붙인다. 늘 달면 칩이 길어져 목록이 답답해진다."""
        labels = movie_labels(
            [
                card(title="올드보이", year=2003),
                card(title="올드보이", year=2013),
                card(title="기생충", year=2019),
            ]
        )

        self.assertEqual(labels["기생충|2019"], "기생충")

    def test_labels_are_unique_per_key(self):
        """같은 이름이 둘 남으면 고친 의미가 없다."""
        labels = movie_labels([card(title="괴물", year=2006), card(title="괴물", year=2023)])

        self.assertEqual(len(set(labels.values())), 2)

    def test_a_missing_year_falls_back_to_the_title(self):
        """연도를 모르면 붙일 것이 없다. 이름이라도 보여야 한다."""
        labels = movie_labels([card(title="제목만", year=0), card(title="제목만", year=2020)])

        self.assertIn("제목만", labels.values())

    def test_the_same_movie_twice_becomes_one_chip(self):
        sources = [card(title="기생충", year=2019), card(title="기생충", year=2019)]

        self.assertEqual(len(unique_sources(sources)), 1)

    def test_order_survives_deduplication(self):
        sources = [card(title="옥자", year=2017), card(title="기생충", year=2019)]

        self.assertEqual([s["title"] for s in unique_sources(sources)], ["옥자", "기생충"])


class ViewTransitionTests(unittest.TestCase):
    """화면은 view 하나로 정해진다. 여기서는 그 전이 규칙만 본다."""

    def test_past_view_carries_the_conversation_id(self):
        self.assertEqual(past_id(past_view("past-1")), "past-1")
        self.assertIsNone(past_id(CHAT_VIEW))
        self.assertIsNone(past_id(WATCHLIST_VIEW))

    def test_past_conversation_replaces_the_watchlist(self):
        """저장 목록이 우선이라 지난 대화를 눌러도 안 바뀌던 문제의 회귀 방지."""
        opened = past_view("past-1")

        self.assertEqual(past_id(opened), "past-1")
        self.assertNotEqual(opened, WATCHLIST_VIEW)

    def test_empty_watchlist_falls_back_to_chat(self):
        self.assertEqual(
            resolved_view(WATCHLIST_VIEW, has_saved_movies=False, past_exists=False),
            CHAT_VIEW,
        )

    def test_watchlist_survives_while_it_has_movies(self):
        self.assertEqual(
            resolved_view(WATCHLIST_VIEW, has_saved_movies=True, past_exists=False),
            WATCHLIST_VIEW,
        )

    def test_missing_past_conversation_falls_back_to_chat(self):
        self.assertEqual(
            resolved_view(past_view("없는-대화"), has_saved_movies=True, past_exists=False),
            CHAT_VIEW,
        )

    def test_existing_past_conversation_stays_open(self):
        view = past_view("past-1")

        self.assertEqual(
            resolved_view(view, has_saved_movies=True, past_exists=True), view
        )

    def test_chat_is_never_resolved_away(self):
        for has_saved in (True, False):
            with self.subTest(has_saved=has_saved):
                self.assertEqual(
                    resolved_view(
                        CHAT_VIEW, has_saved_movies=has_saved, past_exists=False
                    ),
                    CHAT_VIEW,
                )


class ScrollPreservationTests(unittest.TestCase):
    """칩을 누르면 화면이 튀던 문제. 스크롤을 건드리는 두 경로를 각각 막는다."""

    def test_history_container_key_changes_only_with_the_screen(self):
        """키가 같아야 DOM이 살아남고, 그래야 스크롤 위치가 유지된다."""
        session = "s-1"

        self.assertEqual(
            history_container_key(CHAT_VIEW, session),
            history_container_key(CHAT_VIEW, session),
        )
        self.assertNotEqual(
            history_container_key(CHAT_VIEW, session),
            history_container_key(WATCHLIST_VIEW, session),
        )
        self.assertNotEqual(
            history_container_key(past_view("past-1"), session),
            history_container_key(past_view("past-2"), session),
        )

    def test_a_new_conversation_gets_a_new_container(self):
        """'새 대화'는 화면 종류가 그대로다. 키가 같으면 지난 말풍선이 남는다."""
        self.assertNotEqual(
            history_container_key(CHAT_VIEW, "s-1"),
            history_container_key(CHAT_VIEW, "s-2"),
        )

    def test_the_same_token_scrolls_only_once(self):
        """iframe이 다시 붙어도 스크립트가 또 돌면 화면이 맨 아래로 끌려간다."""
        script = scroll_to_bottom_script(3)

        # 부모 문서에 처리한 토큰을 남기고, 같은 값이면 건너뛴다.
        self.assertIn("__cineScrollToken", script)
        self.assertIn("!==", script)
        self.assertIn("const token = 3", script)

    def test_a_new_question_scrolls_again(self):
        """토큰이 오르면 다시 내려가야 한다 — 방금 보낸 질문이 보여야 하므로."""
        self.assertIn("const token = 4", scroll_to_bottom_script(4))


class IsolatedStorageTests(unittest.TestCase):
    """실제 저장소를 건드리지 않는 AppTest용 바탕."""

    def setUp(self):
        # 실제 대화 기록을 읽지 않게 격리한다. 안 하면 개발자 기계에 쌓인 지난
        # 대화가 사이드바에 뜨면서 버튼 목록 검사가 흔들린다(실측).
        self.tmp = tempfile.TemporaryDirectory()
        patcher = patch.object(history, "HISTORY_DIR", Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

        # 보고싶은 영화도 같은 이유로 격리한다 — 개발자 기계에 담아둔 영화가
        # 있으면 사이드바에 버튼이 하나 더 붙는다.
        self.watchlist_tmp = tempfile.TemporaryDirectory()
        watchlist_patcher = patch.object(
            watchlist, "WATCHLIST_DIR", Path(self.watchlist_tmp.name)
        )
        watchlist_patcher.start()
        self.addCleanup(watchlist_patcher.stop)
        self.addCleanup(self.watchlist_tmp.cleanup)

        # 앱이 발급할 신원을 저장 계층의 기본값과 같게 고정한다. 그래야 테스트가
        # 심어 둔 데이터(user_id 없이 저장 → "local")를 앱이 제 것으로 본다.
        id_patcher = patch.object(
            identity, "new_user_id", return_value=history.DEFAULT_USER_ID
        )
        id_patcher.start()
        self.addCleanup(id_patcher.stop)


class CineBotAppTests(IsolatedStorageTests):
    def test_initial_screen_renders_without_error(self):
        app = AppTest.from_file("ui/app.py", default_timeout=10).run()

        self.assertEqual(len(app.exception), 0)
        # 문구를 그대로 박지 않는다. 추천 질문은 바뀌기 마련이고, 여기서 확인할
        # 것은 "첫 화면에 추천 질문이 버튼으로 뜨는가"다.
        labels = [button.label for button in app.button]
        self.assertTrue(set(labels) & set(SUGGESTED_QUESTIONS))
        self.assertEqual(
            app.chat_input[0].placeholder,
            "영화 제목, 감독, 장르 또는 추천 조건을 입력하세요",
        )

    def test_help_discloses_data_freshness_and_usage_limits(self):
        app = AppTest.from_file("ui/app.py", default_timeout=10).run()
        markdown = "\n".join(m.value or "" for m in app.markdown)

        self.assertIn("약 500편", markdown)
        self.assertIn("OTT 정보는 누락되거나 바뀔 수", markdown)
        self.assertIn("하루 30회", markdown)
        self.assertIn("대화당 12회", markdown)
        self.assertIn("쿠키를 지우면", markdown)

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
        app.session_state.view = past_view("past-1")
        app.run()

        self.assertEqual(len(app.exception), 0)
        # 지난 대화 내용이 보이고
        self.assertIn("봉준호입니다.", [m.value for m in app.markdown])
        # 물어볼 자리는 아예 없다. 잠긴 입력창을 남겨 봐야 "왜 막혔지"만 남는다.
        self.assertEqual(len(app.chat_input), 0)

    def test_opening_a_past_conversation_drops_the_unanswered_question(self):
        """남겨두면 돌아왔을 때 뒤늦게 전송돼 같은 질문에 요금이 두 번 나간다."""
        history.save_conversation(
            "past-2", [{"role": "user", "content": "지난 질문"}]
        )
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.pending = "아직 답 못 받은 질문"
        app.session_state.view = past_view("past-2")
        app.run()

        self.assertEqual(len(app.exception), 0)
        self.assertIsNone(app.session_state.pending)

    def test_past_conversation_opens_while_the_watchlist_is_showing(self):
        """저장 목록이 우선이라 지난 대화를 눌러도 화면이 안 바뀌던 문제의 회귀 방지."""
        history.save_conversation(
            "past-5",
            [
                {"role": "user", "content": "기생충 감독은?"},
                {"role": "assistant", "content": "봉준호입니다.", "sources": []},
            ],
        )
        watchlist.add(card())
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.view = WATCHLIST_VIEW
        app.run()
        # 담아둔 영화 화면이 떠 있는 상태에서 지난 대화를 누른다.
        app.sidebar.button(key="past-past-5").click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state.view, past_view("past-5"))
        self.assertIn("봉준호입니다.", [m.value for m in app.markdown])

    def test_emptying_the_watchlist_returns_to_the_chat_screen(self):
        """목록이 비면 상태까지 되돌린다. 안 그러면 다음에 담는 순간 화면이 튄다."""
        watchlist.add(card())
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.view = WATCHLIST_VIEW
        app.run()
        self.assertEqual(app.session_state.view, WATCHLIST_VIEW)

        watchlist.clear()
        app.run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state.view, CHAT_VIEW)
        # 대화 화면이므로 다시 물어볼 수 있어야 한다.
        self.assertFalse(app.chat_input[0].disabled)

    def test_saving_a_movie_again_does_not_jump_to_the_watchlist(self):
        """비운 뒤에도 showing_watchlist가 True로 남아 화면이 튀던 문제의 회귀 방지."""
        watchlist.add(card())
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.view = WATCHLIST_VIEW
        app.run()
        watchlist.clear()
        app.run()

        # 나중에 다시 영화를 담아도 화면은 대화 그대로여야 한다.
        watchlist.add(card(title="괴물", year=2006))
        app.run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state.view, CHAT_VIEW)
        self.assertFalse(app.chat_input[0].disabled)

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
        """스트리밍이 시작되기 전 실행에서도 잠겨 있어야 한다.

        답변을 받아올 예정(pending)인 실행에서 확인한다. 예전에는 지난 대화를
        열어 스트리밍만 건너뛰게 했는데, 이제 읽기 전용 화면에는 입력창이 아예
        없어서 잠금을 볼 수 없다.
        """
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.pending = "질문"

        class Hanging:
            def __init__(self, *args, **kwargs):
                pass

            def stream_query(self, question, session_id):
                raise ui.api_client.ApiClientError("연결 실패")

        with patch.object(ui.api_client, "RagApiClient", Hanging):
            app.run()

        self.assertEqual(len(app.exception), 0)
        # 답변이 끝나면 스스로 다시 그려 잠금을 푼다.
        self.assertFalse(app.chat_input[0].disabled)

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
        app.session_state.view = CHAT_VIEW

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

    def test_streamed_answer_keeps_its_attributions(self):
        """지난 대화를 다시 열어도 근거가 보여야 하고, JustWatch 표기도 남아야 한다."""

        class Answering:
            def __init__(self, *args, **kwargs):
                pass

            def stream_query(self, question, session_id):
                yield {
                    "type": "done",
                    "answer": "넷플릭스에서 볼 수 있습니다.",
                    "sources": [],
                    "attributions": ["tmdb", "justwatch"],
                }

        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.pending = "기생충 어디서 봐?"
        with patch.object(ui.api_client, "RagApiClient", Answering):
            app.run()

        self.assertEqual(len(app.exception), 0)
        answered = app.session_state.messages[-1]
        self.assertEqual(answered["attributions"], ["tmdb", "justwatch"])
        self.assertIn("JustWatch", "\n".join(m.value for m in app.markdown))

    def test_old_answers_without_attributions_still_render(self):
        """표기 기능 이전에 저장된 대화에는 그 필드가 없다."""
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.messages = [
            {"role": "user", "content": "기생충 감독은?"},
            {"role": "assistant", "content": "봉준호입니다.", "sources": []},
        ]
        app.run()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("봉준호입니다.", [m.value for m in app.markdown])

    def test_question_is_claimed_before_the_api_call(self):
        """스트리밍이 중단돼도 다시 전송되지 않도록, 부르기 전에 pending을 비운다."""
        seen: list[Any] = []
        seen_user_ids: list[str | None] = []

        class Recording:
            def __init__(self, *args, **kwargs):
                seen_user_ids.append(kwargs.get("user_id"))

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
        self.assertEqual(seen_user_ids, [history.DEFAULT_USER_ID])

    def test_missing_past_conversation_falls_back_to_the_live_one(self):
        """파일이 지워졌는데 화면이 죽으면 안 된다."""
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.view = past_view("없는-대화")
        app.run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state.view, CHAT_VIEW)
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


class SuggestionTests(IsolatedStorageTests):
    """첫 화면 추천 질문: 후보 중 넷을 뽑되, 뽑은 뒤에는 흔들리지 않는다."""

    def shown_suggestions(self, app) -> list[str]:
        return [
            button.label
            for button in app.button
            if button.key and button.key.startswith("suggestion-")
        ]

    def test_pick_is_a_subset_of_the_pool(self):
        for _ in range(20):
            picked = pick_suggestions()

            self.assertEqual(len(picked), SUGGESTION_COUNT)
            # 같은 질문이 두 번 걸리면 위젯 키까지 겹친다.
            self.assertEqual(len(set(picked)), SUGGESTION_COUNT)
            self.assertTrue(set(picked) <= set(SUGGESTED_QUESTIONS))

    def test_pick_does_not_ask_for_more_than_it_has(self):
        """후보를 줄였을 때 random.sample이 ValueError로 화면을 죽이지 않게."""
        picked = pick_suggestions(("하나", "둘"), 4)

        self.assertEqual(set(picked), {"하나", "둘"})

    def test_pool_has_more_than_it_shows(self):
        """후보가 보여줄 수와 같으면 매번 같은 넷이 나온다 — 바꾼 이유가 사라진다."""
        self.assertGreater(len(SUGGESTED_QUESTIONS), SUGGESTION_COUNT)

    def test_welcome_shows_four_from_the_pool(self):
        app = AppTest.from_file("ui/app.py", default_timeout=10).run()
        shown = self.shown_suggestions(app)

        self.assertEqual(len(shown), SUGGESTION_COUNT)
        self.assertTrue(set(shown) <= set(SUGGESTED_QUESTIONS))

    def test_the_same_four_survive_a_rerun(self):
        """재실행마다 다시 뽑으면 누르려던 버튼이 손가락 밑에서 바뀐다."""
        app = AppTest.from_file("ui/app.py", default_timeout=10).run()
        first = self.shown_suggestions(app)
        app.run()

        self.assertEqual(self.shown_suggestions(app), first)

    def test_the_same_four_survive_opening_the_sidebar_screens(self):
        """사이드바를 건드리기만 해도 재실행된다. 첫 화면은 그대로여야 한다."""
        history.save_conversation(
            "past-9", [{"role": "user", "content": "지난 질문"}]
        )
        app = AppTest.from_file("ui/app.py", default_timeout=10).run()
        first = self.shown_suggestions(app)

        app.session_state.view = past_view("past-9")
        app.run()
        app.session_state.view = CHAT_VIEW
        app.run()

        self.assertEqual(self.shown_suggestions(app), first)

    def test_clicking_a_suggestion_asks_that_question(self):
        """보여주는 목록이 바뀌었어도 누르면 그 질문이 그대로 전송돼야 한다."""

        class Answering:
            def __init__(self, *args, **kwargs):
                pass

            def stream_query(self, question, session_id):
                yield {"type": "done", "answer": "답변입니다.", "sources": []}

        app = AppTest.from_file("ui/app.py", default_timeout=10).run()
        asked = self.shown_suggestions(app)[0]
        with patch.object(ui.api_client, "RagApiClient", Answering):
            app.button(key=f"suggestion-0-{asked}").click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state.messages[0]["content"], asked)

    def test_a_new_conversation_draws_again(self):
        """첫 화면을 다시 보는 유일한 순간이다. 같은 넷이 또 걸리면 고를 게 그것뿐인 줄 안다."""
        drawn = [
            list(SUGGESTED_QUESTIONS[:SUGGESTION_COUNT]),
            list(SUGGESTED_QUESTIONS[SUGGESTION_COUNT:]),
        ]
        with patch.object(random, "sample", side_effect=drawn):
            app = AppTest.from_file("ui/app.py", default_timeout=10).run()
            self.assertEqual(self.shown_suggestions(app), drawn[0])

            # 사이드바 첫 버튼이 '새 대화'다.
            app.sidebar.button[0].click().run()
            app.run()

            self.assertEqual(self.shown_suggestions(app), drawn[1])


class PastConversationSidebarTests(IsolatedStorageTests):
    """지난 대화 목록: 몇 개만 보이고, 나머지는 더보기로, 삭제는 물어보고."""

    def save_past(self, number: int) -> str:
        """지난 대화 하나. 목록 순서가 흔들리지 않게 시각을 직접 박는다.

        save_conversation은 updated_at을 초 단위로 남긴다. 한 테스트에서 여러 개를
        저장하면 값이 같아져 어느 것이 위에 오는지 정해지지 않는다.
        """
        conversation_id = f"c{number}"
        history.save_conversation(
            conversation_id,
            [
                {"role": "user", "content": f"질문 {number}"},
                {"role": "assistant", "content": f"답변 {number}", "sources": []},
            ],
        )
        path = history.HISTORY_DIR / f"{conversation_id}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["updated_at"] = f"2026-01-{number:02d}T00:00:00"
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        return conversation_id

    def open_rows(self, app) -> list[str]:
        """목록에 실제로 보이는 대화 버튼의 키."""
        return [
            button.key
            for button in app.sidebar.button
            if button.key and button.key.startswith("past-c")
        ]

    def button_keys(self, app) -> list[str]:
        return [button.key for button in app.sidebar.button if button.key]

    def settle(self, app):
        """st.rerun()이 걸린 뒤 브라우저가 실제로 보게 될 화면.

        버튼 처리가 st.rerun()을 부르면 스크립트는 그 자리에서 멈춘다. AppTest는
        **중단된 실행의 요소 트리를 그대로 들고 있어서**, 새 화면이 더 짧으면
        지난 실행의 요소가 뒤에 남는다(실측: 접었는데 목록이 6줄로 보였다).
        세션 상태는 이미 바뀌어 있으므로 한 번 더 실행하면 정리된 화면이 나온다.
        """
        return app.run()

    def sidebar_markdown(self, app) -> str:
        return "\n".join(element.value for element in app.sidebar.markdown)

    def test_only_one_page_is_shown(self):
        """50개를 사이드바에 늘어놓으면 '새 대화'가 스크롤 밖으로 밀린다."""
        for number in range(1, 9):
            self.save_past(number)
        app = AppTest.from_file("ui/app.py", default_timeout=10).run()

        self.assertEqual(len(self.open_rows(app)), PAST_PAGE_SIZE)
        # 최근 것부터 보인다.
        self.assertEqual(self.open_rows(app)[0], "past-c8")
        self.assertEqual(app.sidebar.button(key="past-more").label, "더보기 3개")

    def test_short_list_has_no_more_button(self):
        for number in range(1, 4):
            self.save_past(number)
        app = AppTest.from_file("ui/app.py", default_timeout=10).run()

        self.assertEqual(len(self.open_rows(app)), 3)
        self.assertNotIn("past-more", self.button_keys(app))

    def test_more_reveals_the_rest_and_can_be_folded_back(self):
        for number in range(1, 9):
            self.save_past(number)
        app = AppTest.from_file("ui/app.py", default_timeout=10).run()
        app.sidebar.button(key="past-more").click().run()
        self.settle(app)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(self.open_rows(app)), 8)
        self.assertNotIn("past-more", self.button_keys(app))

        app.sidebar.button(key="past-less").click().run()
        self.settle(app)

        self.assertEqual(len(self.open_rows(app)), PAST_PAGE_SIZE)
        self.assertIn("past-more", self.button_keys(app))

    def test_header_shows_how_many_are_stored(self):
        """저장 개수는 목록에 보이는 수가 아니라 파일 수다 — 한도가 걸리는 대상."""
        for number in range(1, 8):
            self.save_past(number)
        app = AppTest.from_file("ui/app.py", default_timeout=10).run()

        self.assertIn(f"7/{history.MAX_CONVERSATIONS}", self.sidebar_markdown(app))

    def test_policy_note_states_the_limit_and_what_happens(self):
        self.save_past(1)
        app = AppTest.from_file("ui/app.py", default_timeout=10).run()
        markdown = self.sidebar_markdown(app)

        self.assertIn(f"최대 {history.MAX_CONVERSATIONS}개까지 저장", markdown)
        self.assertIn("가장 오래된 대화부터 자동으로 삭제", markdown)

    def test_deleting_asks_before_it_removes_anything(self):
        """되돌릴 수 없는 일이다. 누른 것만으로 지워지면 안 된다."""
        self.save_past(1)
        app = AppTest.from_file("ui/app.py", default_timeout=10).run()
        app.sidebar.button(key="delete-past-c1").click().run()

        self.assertEqual(len(app.exception), 0)
        # 모달이 무엇을 지울지 말해주고, 파일은 아직 그대로다.
        self.assertIn("질문 1", "\n".join(m.value for m in app.markdown))
        self.assertIsNotNone(history.load_conversation("c1"))

    def test_cancelling_keeps_the_conversation(self):
        self.save_past(1)
        app = AppTest.from_file("ui/app.py", default_timeout=10).run()
        app.sidebar.button(key="delete-past-c1").click().run()
        app.button(key="cancel-delete").click().run()
        self.settle(app)

        self.assertEqual(len(app.exception), 0)
        self.assertIsNone(app.session_state.pending_delete)
        self.assertIsNotNone(history.load_conversation("c1"))
        self.assertEqual(self.open_rows(app), ["past-c1"])

    def test_confirming_removes_one_conversation(self):
        self.save_past(1)
        self.save_past(2)
        app = AppTest.from_file("ui/app.py", default_timeout=10).run()
        app.sidebar.button(key="delete-past-c1").click().run()
        app.button(key="confirm-delete").click().run()
        self.settle(app)

        self.assertEqual(len(app.exception), 0)
        self.assertIsNone(history.load_conversation("c1"))
        self.assertIsNotNone(history.load_conversation("c2"))
        self.assertEqual(self.open_rows(app), ["past-c2"])

    def test_deleting_the_open_conversation_returns_to_the_chat(self):
        """보고 있던 대화를 지웠는데 화면이 그대로면 빈 화면을 보게 된다."""
        self.save_past(1)
        self.save_past(2)
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.view = past_view("c1")
        app.run()
        app.sidebar.button(key="delete-past-c1").click().run()
        app.button(key="confirm-delete").click().run()
        self.settle(app)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state.view, CHAT_VIEW)
        self.assertFalse(app.chat_input[0].disabled)

    def test_delete_all_asks_with_the_count(self):
        for number in range(1, 4):
            self.save_past(number)
        app = AppTest.from_file("ui/app.py", default_timeout=10).run()
        app.sidebar.button(key="past-delete-all").click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("3개", "\n".join(m.value for m in app.markdown))
        self.assertEqual(len(history.list_conversations()), 3)

    def test_confirming_delete_all_empties_the_list(self):
        for number in range(1, 4):
            self.save_past(number)
        app = AppTest.from_file("ui/app.py", default_timeout=10).run()
        app.sidebar.button(key="past-delete-all").click().run()
        app.button(key="confirm-delete").click().run()
        self.settle(app)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(history.list_conversations(), [])
        # 목록 자체가 사라지므로 머리말도 없어야 한다.
        self.assertNotIn("지난 대화", self.sidebar_markdown(app))

    def test_dismissing_the_modal_forgets_what_it_was_going_to_delete(self):
        """바깥 클릭·X·ESC는 '취소' 버튼을 거치지 않는다.

        대상이 남아 있으면 다음에 아무 버튼이나 눌러 화면이 다시 그려질 때
        닫았던 모달이 되살아난다. 그 길을 st.dialog(on_dismiss=)로 막았고,
        여기서는 그 콜백이 실제로 대상을 지우는지만 본다.
        """
        state = SimpleNamespace(pending_delete=("one", "c1", "질문 1"))
        with patch.object(st, "session_state", state):
            ui.app.forget_delete_target()

        self.assertIsNone(state.pending_delete)

    def test_delete_buttons_are_locked_while_the_answer_streams(self):
        """스트리밍 중에 사이드바를 누르면 실행이 통째로 끊긴다(render_sidebar 참고)."""
        self.save_past(1)
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.pending = "질문"
        app.session_state.view = past_view("c1")
        app.run()

        self.assertTrue(app.sidebar.button(key="delete-past-c1").disabled)
        self.assertTrue(app.sidebar.button(key="past-delete-all").disabled)


class ShelfCardTests(unittest.TestCase):
    """보관함 카드는 접히지 않는다. 훑어보는 곳이라 포스터·제목·연도·평점이면 된다."""

    def test_card_has_no_fold(self):
        """답변 카드의 <details>와 달리 여기서는 펼칠 것이 없다."""
        markup = shelf_card_html(card())

        self.assertNotIn("<details", markup)
        self.assertNotIn("<summary", markup)

    def test_card_shows_identity_only(self):
        markup = shelf_card_html(card())

        self.assertIn("기생충", markup)
        self.assertIn("2019", markup)
        self.assertIn("⭐ 8.5", markup)
        # 감독·출연·줄거리는 답변 카드의 몫이다.
        self.assertNotIn("봉준호", markup)
        self.assertNotIn("요약", markup)

    def test_poster_is_larger_than_the_answer_card(self):
        """보관함에서는 카드가 주인공이라 더 큰 판을 받는다."""
        self.assertIn(SHELF_POSTER_BASE_URL, shelf_card_html(card()))
        self.assertNotEqual(SHELF_POSTER_BASE_URL, POSTER_BASE_URL)

    def test_missing_poster_gets_a_placeholder(self):
        markup = shelf_card_html(card(poster_path=""))

        self.assertNotIn("<img", markup)
        self.assertIn('class="cine-shelf-poster"', markup)

    def test_a_poster_sits_inside_the_placeholder_box(self):
        """보관함도 답변 카드와 같은 구조라야 404에서 함께 버틴다."""
        markup = shelf_card_html(card())

        self.assertIn('class="cine-shelf-poster"', markup)
        self.assertIn('class="cine-shelf-poster-img"', markup)

    def test_titles_are_escaped(self):
        markup = shelf_card_html(card(title="<script>alert(1)</script>"))

        self.assertNotIn("<script>", markup)
        self.assertIn("&lt;script&gt;", markup)


class WatchlistMarkdownTests(unittest.TestCase):
    """담아둔 영화 제목만 불렛으로 옮겨 적는다."""

    def test_titles_become_bullets(self):
        markdown = watchlist_markdown(
            [card(title="기생충", year=2019), card(title="옥자", year=2017)]
        )

        self.assertEqual(markdown, "- 기생충\n- 옥자\n")

    def test_same_title_carries_its_year(self):
        """받아본 사람이 어느 올드보이인지 알 수 있어야 한다."""
        markdown = watchlist_markdown(
            [card(title="올드보이", year=2003), card(title="올드보이", year=2013)]
        )

        self.assertEqual(markdown, "- 올드보이 (2003)\n- 올드보이 (2013)\n")

    def test_only_the_clashing_title_gets_a_year(self):
        markdown = watchlist_markdown(
            [
                card(title="올드보이", year=2003),
                card(title="올드보이", year=2013),
                card(title="기생충", year=2019),
            ]
        )

        self.assertIn("- 기생충\n", markdown)
        self.assertNotIn("기생충 (2019)", markdown)

    def test_the_same_movie_twice_is_one_line(self):
        markdown = watchlist_markdown(
            [card(title="기생충", year=2019), card(title="기생충", year=2019)]
        )

        self.assertEqual(markdown, "- 기생충\n")

    def test_order_follows_the_shelf(self):
        """화면에 놓인 순서 그대로 적힌다."""
        markdown = watchlist_markdown(
            [card(title="옥자", year=2017), card(title="기생충", year=2019)]
        )

        self.assertLess(markdown.index("옥자"), markdown.index("기생충"))

    def test_nothing_saved_yields_nothing(self):
        self.assertEqual(watchlist_markdown([]), "")

    def test_filename_carries_the_date(self):
        """여러 번 받아도 덮어쓰지 않게 한다."""
        name = watchlist_filename(date(2026, 8, 7))

        self.assertTrue(name.endswith(".md"))
        self.assertIn("2026-08-07", name)


class WatchlistShelfTests(IsolatedStorageTests):
    """카드마다 빼기 버튼. 지우기 전에 한 번 묻는다."""

    def open_shelf(self, *movies):
        for movie in movies:
            watchlist.add(movie)
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.view = WATCHLIST_VIEW
        app.run()
        return app

    def test_every_saved_movie_gets_a_card(self):
        app = self.open_shelf(card(title="기생충", year=2019), card(title="옥자", year=2017))

        self.assertEqual(len(shelf_cards(app)), 2)
        self.assertTrue(app.button(key="shelf-delete-0"))
        self.assertTrue(app.button(key="shelf-delete-1"))

    def test_the_chips_are_gone(self):
        """카드에서 바로 뺄 수 있으니 칩은 더 필요 없다."""
        app = self.open_shelf(card())

        self.assertEqual(len(app.pills), 0)

    def test_removing_asks_first(self):
        """되돌릴 수 있는 일이지만, 훑다가 잘못 누르는 자리다."""
        app = self.open_shelf(card(title="기생충", year=2019))

        app.button(key="shelf-delete-0").click().run()

        self.assertEqual(len(app.exception), 0)
        # 아직 지우지 않았다.
        self.assertEqual(watchlist.saved_keys(), {"기생충|2019"})
        self.assertIn("담아둔 영화에서 뺄까요?", "\n".join(m.value for m in app.markdown))

    def test_confirming_removes_only_that_movie(self):
        app = self.open_shelf(
            card(title="기생충", year=2019), card(title="옥자", year=2017)
        )

        # 최근에 담은 것(옥자)이 앞에 온다.
        app.button(key="shelf-delete-0").click().run()
        app.button(key="confirm-delete").click().run()

        self.assertEqual(watchlist.saved_keys(), {"기생충|2019"})

    def test_cancelling_keeps_everything(self):
        app = self.open_shelf(card(title="기생충", year=2019))

        app.button(key="shelf-delete-0").click().run()
        app.button(key="cancel-delete").click().run()

        self.assertEqual(watchlist.saved_keys(), {"기생충|2019"})
        self.assertIsNone(app.session_state.pending_delete)

    def test_removing_refreshes_the_chips_in_answers(self):
        """답변에 달린 칩도 뺀 상태로 다시 그려져야 한다."""
        app = self.open_shelf(card(title="기생충", year=2019))
        before = app.session_state.watchlist_rev

        app.button(key="shelf-delete-0").click().run()
        app.button(key="confirm-delete").click().run()

        self.assertGreater(app.session_state.watchlist_rev, before)

    def test_emptying_the_shelf_returns_to_the_chat(self):
        """마지막 한 편을 빼면 보여줄 것이 없다. 대화로 돌아간다."""
        app = self.open_shelf(card(title="기생충", year=2019))

        app.button(key="shelf-delete-0").click().run()
        app.button(key="confirm-delete").click().run()

        self.assertEqual(app.session_state.view, CHAT_VIEW)

    def test_the_shelf_offers_a_markdown_download(self):
        app = self.open_shelf(
            card(title="기생충", year=2019), card(title="옥자", year=2017)
        )

        self.assertEqual(len(app.exception), 0)
        self.assertIn("마크다운", app.download_button[0].label)

    def test_the_container_key_follows_the_count(self):
        """카드 수가 곧 요소 수다. 줄어드는데 컨테이너가 그대로면 잔여물이 남는다."""
        self.assertNotEqual(
            history_container_key(WATCHLIST_VIEW, "s-1", 3),
            history_container_key(WATCHLIST_VIEW, "s-1", 2),
        )
        self.assertEqual(
            history_container_key(WATCHLIST_VIEW, "s-1", 3),
            history_container_key(WATCHLIST_VIEW, "s-1", 3),
        )


class SidebarNavigationTests(IsolatedStorageTests):
    """사이드바는 내비게이션이다. 항목을 누르면 그 화면으로 가고, 구성은 안 바뀐다."""

    def nav(self, app, key: str):
        return app.sidebar.button(key=f"nav-{key}")

    def test_both_destinations_are_always_listed(self):
        """어느 화면에 있든 같은 항목이 같은 자리에 있어야 위치를 기억할 수 있다."""
        watchlist.add(card())
        app = AppTest.from_file("ui/app.py", default_timeout=10).run()

        self.assertIn("지금 대화", self.nav(app, "chat").label)
        self.assertIn("담아둔 영화", self.nav(app, "watchlist").label)

        app.session_state.view = WATCHLIST_VIEW
        app.run()

        self.assertIn("지금 대화", self.nav(app, "chat").label)
        self.assertIn("담아둔 영화", self.nav(app, "watchlist").label)

    def test_the_label_never_becomes_an_action(self):
        """'목록 닫기'처럼 라벨이 동작으로 바뀌면 지금 어디인지 글자로 역산해야 한다."""
        watchlist.add(card())
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.view = WATCHLIST_VIEW
        app.run()

        self.assertEqual(self.nav(app, "watchlist").label, "담아둔 영화 · 1편")

    def test_clicking_a_destination_goes_there(self):
        watchlist.add(card())
        app = AppTest.from_file("ui/app.py", default_timeout=10).run()

        self.nav(app, "watchlist").click().run()

        self.assertEqual(app.session_state.view, WATCHLIST_VIEW)

    def test_clicking_the_same_destination_stays_put(self):
        """예전에는 같은 버튼이 토글이라 두 번째 클릭에 대화로 튕겨 나갔다."""
        watchlist.add(card())
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.view = WATCHLIST_VIEW
        app.run()

        self.nav(app, "watchlist").click().run()

        self.assertEqual(app.session_state.view, WATCHLIST_VIEW)

    def test_now_talking_brings_you_back_from_anywhere(self):
        history.save_conversation("past-nav", [{"role": "user", "content": "지난 질문"}])
        watchlist.add(card())
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()

        for view in (WATCHLIST_VIEW, past_view("past-nav")):
            with self.subTest(view=view):
                app.session_state.view = view
                app.run()
                self.nav(app, "chat").click().run()
                self.assertEqual(app.session_state.view, CHAT_VIEW)

    def test_an_empty_watchlist_keeps_its_place(self):
        """자리가 사라졌다 생기면 그때마다 아래 항목이 밀린다."""
        app = AppTest.from_file("ui/app.py", default_timeout=10).run()

        item = self.nav(app, "watchlist")
        self.assertEqual(item.label, "담아둔 영화 · 0편")
        self.assertTrue(item.disabled)

    def test_the_back_button_is_gone(self):
        """돌아가는 길은 '지금 대화' 항목 하나로 모았다."""
        history.save_conversation("past-back", [{"role": "user", "content": "지난 질문"}])
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.view = past_view("past-back")
        app.run()

        labels = [button.label for button in app.sidebar.button]
        self.assertNotIn("지금 대화로 돌아가기", labels)

    def test_the_open_conversation_is_still_clickable(self):
        """비활성으로 흐리게 두면 지금 있는 곳이 오히려 가장 안 보인다."""
        history.save_conversation("past-open", [{"role": "user", "content": "지난 질문"}])
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.view = past_view("past-open")
        app.run()

        row = app.sidebar.button(key="past-past-open")
        self.assertFalse(row.disabled)

        row.click().run()

        self.assertEqual(app.session_state.view, past_view("past-open"))


class SameTitleWatchlistTests(IsolatedStorageTests):
    """원작과 리메이크는 제목이 같다. 담고 빼는 길이 서로 막히면 안 된다."""

    원작 = staticmethod(lambda: card(title="올드보이", year=2003, vote_average=8.2))
    리메이크 = staticmethod(lambda: card(title="올드보이", year=2013, vote_average=5.9))

    def answer_with_both(self, app):
        app.session_state.messages = [
            {"role": "user", "content": "올드보이 알려줘"},
            {
                "role": "assistant",
                "content": "올드보이(2003)와 올드보이(2013)입니다.",
                "sources": [self.원작(), self.리메이크()],
                "attributions": ["tmdb"],
            },
        ]
        app.run()

    def test_both_can_be_saved(self):
        """실측 회귀: 둘 다 골라도 파일에는 한 편만 들어갔다."""
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        self.answer_with_both(app)

        app.pills[0].set_value(
            [watchlist.movie_key(self.원작()), watchlist.movie_key(self.리메이크())]
        ).run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(
            sorted(watchlist.saved_keys()), ["올드보이|2003", "올드보이|2013"]
        )

    def test_removing_one_leaves_the_other(self):
        """실측 회귀: 지워도 동명의 한 편이 파일에 남았다."""
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        self.answer_with_both(app)
        app.pills[0].set_value(
            [watchlist.movie_key(self.원작()), watchlist.movie_key(self.리메이크())]
        ).run()

        app.pills[0].set_value([watchlist.movie_key(self.원작())]).run()

        self.assertEqual(sorted(watchlist.saved_keys()), ["올드보이|2003"])

    def test_both_can_be_removed(self):
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        self.answer_with_both(app)
        app.pills[0].set_value(
            [watchlist.movie_key(self.원작()), watchlist.movie_key(self.리메이크())]
        ).run()

        app.pills[0].set_value([]).run()

        self.assertEqual(watchlist.saved_keys(), set())

    def test_the_watchlist_screen_shows_both_apart(self):
        """담아둔 영화 화면에서도 둘이 각각 카드로 보이고 따로 빠져야 한다."""
        watchlist.add(self.원작())
        watchlist.add(self.리메이크())
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.view = WATCHLIST_VIEW
        app.run()

        cards = shelf_cards(app)
        self.assertEqual(len(cards), 2)
        self.assertIn("2003", cards[1])
        self.assertIn("2013", cards[0])

        # 최근에 담은 것(리메이크)이 앞에 온다. 그것만 뺀다.
        app.button(key="shelf-delete-0").click().run()
        app.button(key="confirm-delete").click().run()

        self.assertEqual(sorted(watchlist.saved_keys()), ["올드보이|2003"])


class ScrollTimingTests(IsolatedStorageTests):
    """언제 내려가는가. 토큰이 오르는 순간이 곧 스크롤하는 순간이다."""

    class Answering:
        def __init__(self, *args, **kwargs):
            pass

        def stream_query(self, question, session_id):
            yield {"type": "done", "answer": "답변입니다.", "sources": [], "attributions": []}

    def test_a_finished_answer_scrolls_to_itself(self):
        """위로 올려 읽던 중이라도 답변이 확정되면 그 말풍선으로 돌아와야 한다."""
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.pending = "질문"
        before = app.session_state.scroll_token
        with patch.object(ui.api_client, "RagApiClient", self.Answering):
            app.run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state.scroll_token, before + 1)

    def test_saving_a_movie_does_not_scroll(self):
        """칩을 눌렀다고 화면이 움직이면 읽던 자리를 잃는다."""
        history.save_conversation(
            "past-scroll",
            [
                {"role": "user", "content": "추천해줘"},
                {"role": "assistant", "content": "기생충입니다.", "sources": [card()]},
            ],
        )
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        app.session_state.view = past_view("past-scroll")
        app.run()
        before = app.session_state.scroll_token

        # 답변에 달린 칩으로 영화를 담는다.
        app.pills[0].set_value([watchlist.movie_key(card())]).run()

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(watchlist.saved_keys())
        self.assertEqual(app.session_state.scroll_token, before)

    def test_opening_a_past_conversation_does_not_scroll(self):
        history.save_conversation(
            "past-quiet", [{"role": "user", "content": "지난 질문"}]
        )
        app = AppTest.from_file("ui/app.py", default_timeout=10)
        app.run()
        before = app.session_state.scroll_token
        app.session_state.view = past_view("past-quiet")
        app.run()

        self.assertEqual(app.session_state.scroll_token, before)


if __name__ == "__main__":
    unittest.main()
