"""그래프 계층 테스트: 출처 수집의 턴 경계와 중복 제거, 스트리밍 이벤트."""

from __future__ import annotations

import unittest
import uuid
from datetime import date
from unittest.mock import patch

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)
from langgraph.errors import GraphRecursionError

import rag.graph
from rag.graph import (
    _PREAMBLE_HOLD_CHARS,
    MovieRagGraph,
    build_system_prompt,
    collect_turn_attributions,
    collect_turn_sources,
    collect_turn_tool_calls,
    collect_turn_tool_results,
    collect_turn_web_sources,
    tool_status,
)


def source(title: str, year: int = 2020, movie_id: int = 0) -> dict:
    return {
        "movie_id": movie_id,
        "title": title,
        "year": year,
        "director": "",
        "genres": "드라마",
        "country": "한국",
        "vote_average": 8.0,
        "poster_path": "/x.jpg",
        "snippet": "요약",
    }


def tool_message(*sources: dict) -> ToolMessage:
    return ToolMessage(content="결과", tool_call_id="call-1", artifact=list(sources))


def result_message(
    call_id: str,
    *,
    sources: list[dict] | None = None,
    web_sources: list[dict] | None = None,
    success: bool = True,
) -> ToolMessage:
    return ToolMessage(
        content="결과" if success else "실패",
        tool_call_id=call_id,
        artifact={
            "success": success,
            "sources": list(sources or []),
            "web_sources": list(web_sources or []),
        },
    )


class CollectTurnSourcesTests(unittest.TestCase):
    def test_collects_sources_from_current_turn(self):
        messages = [
            HumanMessage(content="봉준호 영화"),
            AIMessage(content=""),
            tool_message(source("기생충", 2019), source("살인의 추억", 2003)),
            AIMessage(content="답변"),
        ]
        titles = [s["title"] for s in collect_turn_sources(messages)]
        self.assertEqual(titles, ["기생충", "살인의 추억"])

    def test_stops_at_previous_turn(self):
        """체크포인터가 보존한 이전 턴의 출처가 섞이면 안 된다."""
        messages = [
            HumanMessage(content="1턴 질문"),
            tool_message(source("이전턴영화")),
            AIMessage(content="1턴 답변"),
            HumanMessage(content="2턴 질문"),
            tool_message(source("이번턴영화")),
            AIMessage(content="2턴 답변"),
        ]
        titles = [s["title"] for s in collect_turn_sources(messages)]
        self.assertEqual(titles, ["이번턴영화"])

    def test_deduplicates_by_title_and_year(self):
        """도구를 여러 번 불러 같은 영화가 겹쳐도 카드는 하나여야 한다."""
        messages = [
            HumanMessage(content="질문"),
            tool_message(source("기생충", 2019)),
            tool_message(source("기생충", 2019), source("옥자", 2017)),
            AIMessage(content="답변"),
        ]
        titles = [s["title"] for s in collect_turn_sources(messages)]
        self.assertEqual(titles, ["기생충", "옥자"])

    def test_same_title_different_year_is_kept(self):
        messages = [
            HumanMessage(content="질문"),
            tool_message(source("괴물", 2006), source("괴물", 2023)),
            AIMessage(content="답변"),
        ]
        self.assertEqual(len(collect_turn_sources(messages)), 2)

    def test_no_tool_call_yields_no_sources(self):
        """대화 자체에 대한 질문은 도구를 안 부르므로 출처가 없다."""
        messages = [
            HumanMessage(content="내가 방금 뭐라고 했지?"),
            AIMessage(content="봉준호 영화를 물으셨습니다."),
        ]
        self.assertEqual(collect_turn_sources(messages), [])

    def test_ignores_tool_message_without_artifact(self):
        """되묻기·오류 응답은 artifact가 비어 있어 출처를 만들지 않는다."""
        messages = [
            HumanMessage(content="평점 가장 높은 영화"),
            ToolMessage(content="범위를 좁혀 주세요.", tool_call_id="c", artifact=[]),
            AIMessage(content="어떤 장르로 좁힐까요?"),
        ]
        self.assertEqual(collect_turn_sources(messages), [])


def ai_with_calls(*calls: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": n, "args": a, "id": f"c{i}", "type": "tool_call"}
        for i, (n, a) in enumerate(calls)
    ])


class CollectTurnToolCallsTests(unittest.TestCase):
    """라우팅 평가의 근거. 어떤 도구를 어떤 인자로 불렀는지 봐야 판단할 수 있다."""

    def test_collects_name_and_args(self):
        messages = [
            HumanMessage(content="잔인하지 않은 액션"),
            ai_with_calls(("search_by_vibe", {"vibe": "통쾌한", "max_violence": 2})),
            tool_message(source("엑시트")),
            AIMessage(content="답변"),
        ]
        calls = collect_turn_tool_calls(messages)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "search_by_vibe")
        self.assertEqual(calls[0]["args"]["max_violence"], 2)

    def test_stops_at_previous_turn(self):
        messages = [
            HumanMessage(content="1턴"),
            ai_with_calls(("search_movies", {"person": "봉준호"})),
            AIMessage(content="1턴 답변"),
            HumanMessage(content="2턴"),
            ai_with_calls(("get_movie_details", {"title": "기생충"})),
            AIMessage(content="2턴 답변"),
        ]
        self.assertEqual(
            [c["name"] for c in collect_turn_tool_calls(messages)],
            ["get_movie_details"],
        )

    def test_collects_multiple_calls_in_order(self):
        """웹으로 제목을 알아낸 뒤 상세를 잇는 연결 패턴."""
        messages = [
            HumanMessage(content="아카데미 수상작 어디서 봐"),
            ai_with_calls(("web_search", {"query": "아카데미 작품상"})),
            AIMessage(content=""),
            ai_with_calls(("get_movie_details", {"title": "기생충"})),
            AIMessage(content="답변"),
        ]
        self.assertEqual(
            [c["name"] for c in collect_turn_tool_calls(messages)],
            ["web_search", "get_movie_details"],
        )

    def test_no_tool_call_yields_empty(self):
        messages = [
            HumanMessage(content="안녕?"),
            AIMessage(content="영화에 대해 물어보세요."),
        ]
        self.assertEqual(collect_turn_tool_calls(messages), [])


class CollectTurnToolResultsTests(unittest.TestCase):
    def test_connects_calls_to_successful_structured_evidence(self):
        movie = source("기생충", 2019, 10)
        web = {"title": "기생충 평가", "url": "https://example.com/review"}
        messages = [
            HumanMessage(content="질문"),
            ai_with_calls(
                ("search_movies", {"person": "봉준호"}),
                ("web_search", {"query": "기생충 평단"}),
            ),
            result_message("c0", sources=[movie]),
            result_message("c1", web_sources=[web]),
            AIMessage(content="답변"),
        ]

        results = collect_turn_tool_results(messages)

        self.assertEqual([item["name"] for item in results], ["search_movies", "web_search"])
        self.assertEqual(results[0]["args"], {"person": "봉준호"})
        self.assertTrue(results[0]["success"])
        self.assertEqual(results[0]["sources"], [movie])
        self.assertEqual(results[1]["web_sources"], [web])

    def test_failed_result_keeps_diagnostic_text_but_not_evidence(self):
        messages = [
            HumanMessage(content="질문"),
            ai_with_calls(("web_search", {"query": "q"})),
            ToolMessage(
                content="웹 검색 실패",
                tool_call_id="c0",
                artifact={
                    "success": False,
                    "sources": [source("잘못된 카드")],
                    "web_sources": [{"title": "x", "url": "https://x"}],
                },
            ),
            AIMessage(content="실패했습니다."),
        ]

        result = collect_turn_tool_results(messages)[0]

        self.assertFalse(result["success"])
        self.assertEqual(result["content"], "웹 검색 실패")
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["web_sources"], [])

    def test_stops_at_previous_turn(self):
        messages = [
            HumanMessage(content="이전"),
            ai_with_calls(("search_movies", {"genre": "액션"})),
            result_message("c0", sources=[source("이전 영화")]),
            AIMessage(content="이전 답변"),
            HumanMessage(content="현재"),
            ai_with_calls(("search_by_vibe", {"vibe": "잔잔한"})),
            result_message("c0", sources=[source("현재 영화")]),
            AIMessage(content="현재 답변"),
        ]

        results = collect_turn_tool_results(messages)

        self.assertEqual([item["name"] for item in results], ["search_by_vibe"])
        self.assertEqual(results[0]["sources"][0]["title"], "현재 영화")

    def test_trace_returns_final_fields_calls_and_tool_evidence(self):
        movie = source("기생충", 2019, 10)
        messages = [
            HumanMessage(content="봉준호 영화"),
            ai_with_calls(("search_movies", {"person": "봉준호"})),
            result_message("c0", sources=[movie]),
            AIMessage(content="**기생충**(2019)을 추천합니다."),
        ]
        graph = object.__new__(MovieRagGraph)

        with patch.object(MovieRagGraph, "_run", return_value=messages):
            result = graph.trace("봉준호 영화")

        self.assertEqual(result["answer"], "**기생충**(2019)을 추천합니다.")
        self.assertEqual(result["sources"], [movie])
        self.assertEqual(result["tool_calls"][0]["name"], "search_movies")
        self.assertEqual(result["tool_results"][0]["sources"], [movie])


class TurnAttributionsTests(unittest.TestCase):
    """답변 밑의 출처 표기는 성공한 도구 결과에서만 얻는다."""

    def test_each_tool_names_its_data(self):
        messages = [
            HumanMessage(content="분위기 좋은 영화"),
            ai_with_calls(("search_by_vibe", {"vibe": "잔잔한"})),
            result_message("c0", sources=[source("기생충")]),
            AIMessage(content="답변"),
        ]

        self.assertEqual(collect_turn_attributions(messages), ["local"])

    def test_watch_provider_data_credits_justwatch(self):
        """TMDB의 시청처는 JustWatch 제공이라 표기가 의무다."""
        messages = [
            HumanMessage(content="기생충 어디서 봐?"),
            ai_with_calls(("get_movie_details", {"title": "기생충"})),
            result_message("c0", sources=[source("기생충")]),
            AIMessage(content="답변"),
        ]

        self.assertEqual(collect_turn_attributions(messages), ["tmdb", "justwatch"])

    def test_filtering_by_ott_also_credits_justwatch(self):
        """편성으로 걸렀다면 그 데이터를 조건으로 쓴 것이다."""
        messages = [
            HumanMessage(content="넷플릭스 액션"),
            ai_with_calls(("search_movies", {"genre": "액션", "watch_provider": "넷플릭스"})),
            result_message("c0", sources=[source("기생충")]),
            AIMessage(content="답변"),
        ]

        self.assertEqual(collect_turn_attributions(messages), ["tmdb", "justwatch"])

    def test_plain_search_does_not_credit_justwatch(self):
        messages = [
            HumanMessage(content="봉준호 영화"),
            ai_with_calls(("search_movies", {"person": "봉준호"})),
            result_message("c0", sources=[source("기생충")]),
            AIMessage(content="답변"),
        ]

        self.assertEqual(collect_turn_attributions(messages), ["tmdb"])

    def test_multiple_tools_are_deduped_and_ordered(self):
        """무엇을 검색했는지가 먼저, 제공자 표기는 뒤."""
        messages = [
            HumanMessage(content="아카데미 수상작 어디서 봐"),
            ai_with_calls(("web_search", {"query": "아카데미"})),
            result_message(
                "c0",
                web_sources=[{"title": "수상 결과", "url": "https://example.com/a"}],
            ),
            AIMessage(content=""),
            ai_with_calls(
                ("get_movie_details", {"title": "기생충"}),
                ("search_movies", {"person": "봉준호"}),
            ),
            result_message("c0", sources=[source("기생충")]),
            result_message("c1", sources=[source("옥자")]),
            AIMessage(content="답변"),
        ]

        self.assertEqual(
            collect_turn_attributions(messages), ["tmdb", "web", "justwatch"]
        )

    def test_no_tool_means_nothing_to_credit(self):
        """도구 없이 답한 턴(이전 대화에 대한 질문 등)에는 표기가 없다."""
        messages = [
            HumanMessage(content="방금 첫 번째로 추천한 게 뭐였지?"),
            AIMessage(content="기생충이었습니다."),
        ]

        self.assertEqual(collect_turn_attributions(messages), [])

    def test_failed_and_empty_tools_are_not_credited(self):
        messages = [
            HumanMessage(content="평단 반응"),
            ai_with_calls(("web_search", {"query": "평단 반응"})),
            result_message("c0", success=False),
            AIMessage(content="웹 검색이 실패했습니다."),
        ]

        self.assertEqual(collect_turn_attributions(messages), [])
        self.assertEqual(collect_turn_web_sources(messages), [])

    def test_successful_count_without_movie_cards_is_credited(self):
        messages = [
            HumanMessage(content="2020년 이후 몇 편?"),
            ai_with_calls(("search_movies", {"year_from": 2020, "count_only": True})),
            result_message("c0"),
            AIMessage(content="266편입니다."),
        ]

        self.assertEqual(collect_turn_attributions(messages), ["tmdb"])


class SystemPromptTests(unittest.TestCase):
    """프롬프트가 지켜야 할 두 가지 경계.

    LLM 동작 자체는 여기서 확인할 수 없다. 규칙이 프롬프트에서 조용히 빠지는
    것만 막는다 — 둘 다 실측으로 드러난 문제라 사라지면 곧바로 재발한다.
    """

    def test_the_answer_is_confined_to_tool_results(self):
        """실측: 도구 후보에 맞는 것이 없자 제 기억으로 다섯 편을 추천했다."""
        prompt = build_system_prompt(date(2026, 8, 7))

        self.assertIn("도구 결과에 있던 것이어야", prompt)
        self.assertIn("제목조차", prompt)

    def test_off_topic_questions_are_refused(self):
        """실측: '오늘 뭐먹지?'에 식사 추천을 했다."""
        prompt = build_system_prompt(date(2026, 8, 7))

        self.assertIn("영화와 이 서비스에 대한 질문에만", prompt)

    def test_todays_date_is_written_in(self):
        """날짜가 없으면 '올해'를 학습 시점으로 읽는다(기존 규칙)."""
        prompt = build_system_prompt(date(2026, 8, 7))

        self.assertIn("2026", prompt)


class StructuredSourcesTests(unittest.TestCase):
    """artifact는 카드의 자격·내용, 답변은 카드의 선택·순서를 정한다."""

    def test_short_title_substrings_do_not_change_sources(self):
        """시·밤·형·콜·섬·활이 일반 문장에 있어도 문자열로 출처를 만들지 않는다."""
        returned = [
            source(title, 2020 + index, 100 + index)
            for index, title in enumerate(("시", "밤", "형", "콜", "섬", "활"))
        ]
        with_coincidences = [
            HumanMessage(content="추천해줘"),
            result_message("c0", sources=returned),
            AIMessage(content="새로운 시도를 해볼 만한 밤입니다. 활기차게 골라봤어요."),
        ]
        without_coincidences = [
            HumanMessage(content="추천해줘"),
            result_message("c0", sources=returned),
            AIMessage(content="검색 결과를 정리했습니다."),
        ]

        self.assertEqual(
            MovieRagGraph._to_result(with_coincidences)["sources"],
            MovieRagGraph._to_result(without_coincidences)["sources"],
        )
        self.assertEqual(MovieRagGraph._to_result(with_coincidences)["sources"], [])

    def test_cards_follow_answer_selection_and_order(self):
        returned = [
            source("시", 2015, 1),
            source("기생충", 2019, 2),
            source("옥자", 2017, 3),
        ]
        messages = [
            HumanMessage(content="추천해줘"),
            result_message("c0", sources=returned),
            AIMessage(
                content="**옥자**(2017)를 먼저, **기생충**(2019)을 다음으로 추천합니다."
            ),
        ]

        result = MovieRagGraph._to_result(messages)

        self.assertEqual([item["movie_id"] for item in result["sources"]], [3, 2])

    def test_plain_title_and_year_are_an_explicit_mention(self):
        messages = [
            HumanMessage(content="추천해줘"),
            result_message("c0", sources=[source("기생충", 2019, 2)]),
            AIMessage(content="기생충(2019)을 추천합니다."),
        ]

        result = MovieRagGraph._to_result(messages)

        self.assertEqual(result["sources"][0]["movie_id"], 2)

    def test_short_title_is_kept_when_title_and_year_are_explicit(self):
        messages = [
            HumanMessage(content="추천해줘"),
            result_message("c0", sources=[source("시", 2010, 10)]),
            AIMessage(content="**시**(2010)를 추천합니다."),
        ]

        result = MovieRagGraph._to_result(messages)

        self.assertEqual([item["movie_id"] for item in result["sources"]], [10])

    def test_same_title_different_year_uses_the_stated_year(self):
        returned = [source("올드보이", 2013, 13), source("올드보이", 2003, 3)]
        messages = [
            HumanMessage(content="박찬욱 영화 추천해줘"),
            result_message("c0", sources=returned),
            AIMessage(content="**올드보이**(2003)를 추천합니다."),
        ]

        result = MovieRagGraph._to_result(messages)

        self.assertEqual([item["movie_id"] for item in result["sources"]], [3])

    def test_punctuation_difference_in_title_still_matches(self):
        returned = [source("너의 이름은.", 2016, 16)]
        messages = [
            HumanMessage(content="추천해줘"),
            result_message("c0", sources=returned),
            AIMessage(content="**너의 이름은**(2016)을 추천합니다."),
        ]

        self.assertEqual(MovieRagGraph._to_result(messages)["sources"], returned)

    def test_same_textual_mention_does_not_create_two_cards(self):
        returned = [source("괴물", 2006, 1), source("괴물", 2006, 2)]
        messages = [
            HumanMessage(content="추천해줘"),
            result_message("c0", sources=returned),
            AIMessage(content="**괴물**(2006)을 추천합니다."),
        ]

        result = MovieRagGraph._to_result(messages)

        self.assertEqual([item["movie_id"] for item in result["sources"]], [1])

    def test_movie_not_returned_by_a_tool_cannot_get_a_card(self):
        messages = [
            HumanMessage(content="추천해줘"),
            result_message("c0", sources=[source("기생충", 2019, 2)]),
            AIMessage(content="**옥자**(2017)를 추천합니다."),
        ]

        self.assertEqual(MovieRagGraph._to_result(messages)["sources"], [])

    def test_failed_result_cannot_supply_a_short_title(self):
        messages = [
            HumanMessage(content="추천해줘"),
            result_message("c0", sources=[source("밤", 2012, 3)], success=False),
            AIMessage(content="오늘 밤 보기 좋은 작품을 찾지 못했습니다."),
        ]

        self.assertEqual(MovieRagGraph._to_result(messages)["sources"], [])

    def test_movie_id_is_the_primary_deduplication_key(self):
        messages = [
            HumanMessage(content="질문"),
            result_message("c0", sources=[source("기생충", 2019, 10)]),
            result_message("c1", sources=[source("Parasite", 2019, 10)]),
            AIMessage(content="답변"),
        ]

        self.assertEqual(len(collect_turn_sources(messages)), 1)

    def test_web_urls_are_deduplicated_structurally(self):
        web = {"title": "기생충 평가", "url": "https://example.com/review"}
        messages = [
            HumanMessage(content="평단 반응"),
            result_message("c0", web_sources=[web]),
            result_message("c1", web_sources=[web]),
            AIMessage(content="호평입니다."),
        ]

        self.assertEqual(collect_turn_web_sources(messages), [web])
        self.assertEqual(MovieRagGraph._to_result(messages)["web_sources"], [web])


class ToolStatusTests(unittest.TestCase):
    def test_known_tool_gets_its_own_label(self):
        self.assertEqual(tool_status(["search_by_vibe"]), "분위기에 맞는 영화를 고르는 중")

    def test_parallel_calls_of_same_tool_collapse(self):
        """get_movie_details ×3은 같은 문구를 세 번 잇지 않는다."""
        self.assertEqual(
            tool_status(["get_movie_details"] * 3), "상세 정보를 확인하는 중"
        )

    def test_different_tools_are_joined(self):
        self.assertEqual(
            tool_status(["search_movies", "web_search"]),
            "영화를 찾는 중 · 웹에서 최신 정보를 찾는 중",
        )

    def test_unknown_tool_falls_back(self):
        """도구를 새로 추가해도 문구가 비지 않아야 한다."""
        self.assertEqual(tool_status(["search_by_actor"]), "정보를 확인하는 중")


def text_chunk(text: str, node: str = "agent") -> tuple:
    return ("messages", (AIMessageChunk(content=[{"type": "text", "text": text}]), {"langgraph_node": node}))


def tool_arg_chunk() -> tuple:
    """도구 인자가 실려오는 청크. Anthropic은 partial_json으로 흘려보낸다."""
    chunk = AIMessageChunk(
        content=[{"type": "tool_use", "index": 0, "partial_json": '{"genre":'}]
    )
    return ("messages", (chunk, {"langgraph_node": "agent"}))


def update(node: str, *messages) -> tuple:
    return ("updates", {node: {"messages": list(messages)}})


class FakeApp:
    """compile()된 그래프 대신 정해진 이벤트를 흘려보낸다."""

    def __init__(self, events: list, error: Exception | None = None) -> None:
        self.events = events
        self.error = error
        self.updated: list = []

    def stream(self, inputs, config, stream_mode):
        yield from self.events
        if self.error:
            raise self.error

    def update_state(self, config, values):
        self.updated.append(values)


def graph_with(events: list, error: Exception | None = None) -> MovieRagGraph:
    """LLM을 만들지 않고 스트리밍 경로만 떼어내 시험한다."""
    graph = object.__new__(MovieRagGraph)
    graph.app = FakeApp(events, error)
    return graph


class StreamAnswerTests(unittest.TestCase):
    """이벤트 순서가 UI 렌더링 규약이다. 특히 reset의 위치가 중요하다."""

    def run_stream(self, events: list, error: Exception | None = None) -> list[dict]:
        graph = graph_with(events, error)
        collected = list(graph.stream_answer("한국 스릴러 추천해줘", "s-1"))
        self.last_app = graph.app
        return collected

    def test_short_preamble_never_reaches_the_client(self):
        """도구 호출 전 서두는 화면에 뜨지도 않아야 한다.

        실측: '이제 첫 번째 영화의 상세 정보를 확인하겠습니다.' 같은 문장이
        도구 호출 직전에 나온다. 흘렸다가 지우면 사용자 눈에 깜빡이므로,
        한도 미만은 쥐고 있다가 도구 라운드로 밝혀지면 조용히 버린다.
        """
        events = self.run_stream(
            [
                text_chunk("찾아드릴게요."),
                tool_arg_chunk(),
                update("agent", ai_with_calls(("search_movies", {"genre": "스릴러"}))),
                update("tools", tool_message(source("기생충", 2019))),
                text_chunk("기생충(2019)을 추천합니다."),
                update("agent", AIMessage(content="기생충(2019)을 추천합니다.")),
            ]
        )

        self.assertEqual(
            [(e["type"], e.get("text")) for e in events],
            [
                ("status", "질문을 살펴보는 중"),
                ("status", "영화를 찾는 중"),
                ("token", "기생충(2019)을 추천합니다."),
                ("done", None),
            ],
        )
        # 아무것도 안 나갔으니 취소할 것도 없다.
        self.assertNotIn("reset", [e["type"] for e in events])
        self.assertEqual(events[-1]["answer"], "기생충(2019)을 추천합니다.")

    def test_long_preamble_leaks_and_is_cancelled(self):
        """한도를 넘긴 서두는 새어나간다. 그때는 reset이 안전망으로 뒤를 받는다."""
        preamble = "가" * (_PREAMBLE_HOLD_CHARS + 10)
        events = self.run_stream(
            [
                text_chunk(preamble),
                update("agent", ai_with_calls(("search_movies", {}))),
                update("tools", tool_message(source("기생충", 2019))),
                text_chunk("기생충입니다."),
                update("agent", AIMessage(content="기생충입니다.")),
            ]
        )

        types = [e["type"] for e in events]
        self.assertIn("reset", types)
        self.assertLess(types.index("reset"), types.index("done"))
        # 취소 전에 서두가 나갔고, 취소 후 최종 답변이 다시 나간다.
        self.assertEqual(
            [e["text"] for e in events if e["type"] == "token"],
            [preamble, "기생충입니다."],
        )

    def test_long_answer_streams_after_the_hold(self):
        """최종 답변은 한도를 넘는 순간 흘러나가야 한다. 통째로 모았다 주면 안 된다."""
        head = "나" * _PREAMBLE_HOLD_CHARS
        events = self.run_stream(
            [
                update("agent", ai_with_calls(("search_movies", {}))),
                update("tools", tool_message(source("기생충", 2019))),
                text_chunk(head),
                text_chunk("뒤에 이어지는 문장."),
                update("agent", AIMessage(content=head + "뒤에 이어지는 문장.")),
            ]
        )

        # 첫 덩어리는 쥐고 있던 몫, 그 뒤로는 오는 대로 흘린다.
        self.assertEqual(
            [e["text"] for e in events if e["type"] == "token"],
            [head, "뒤에 이어지는 문장."],
        )

    def test_tool_argument_chunks_are_not_tokens(self):
        """partial_json 청크가 답변 텍스트로 새면 화면에 JSON이 찍힌다."""
        events = self.run_stream(
            [
                tool_arg_chunk(),
                update("agent", ai_with_calls(("search_movies", {}))),
                update("tools", tool_message(source("기생충", 2019))),
                text_chunk("기생충입니다."),
                update("agent", AIMessage(content="기생충입니다.")),
            ]
        )
        self.assertEqual([e["text"] for e in events if e["type"] == "token"], ["기생충입니다."])

    def test_tool_output_is_not_streamed_as_answer(self):
        """ToolMessage도 messages 모드로 흘러나온다. 원문을 답변인 양 뿌리면 안 된다."""
        raw = ("messages", (ToolMessage(content="- 기생충 (2019) | 평점 8.5", tool_call_id="c"), {"langgraph_node": "tools"}))
        events = self.run_stream(
            [
                update("agent", ai_with_calls(("search_movies", {}))),
                raw,
                update("tools", tool_message(source("기생충", 2019))),
                text_chunk("기생충입니다."),
                update("agent", AIMessage(content="기생충입니다.")),
            ]
        )
        self.assertEqual([e["text"] for e in events if e["type"] == "token"], ["기생충입니다."])

    def test_done_carries_only_sources_in_answer_order(self):
        """도구 후보 중 답변이 실제 소개한 카드만 같은 순서로 확정한다."""
        events = self.run_stream(
            [
                update("agent", ai_with_calls(("search_movies", {}))),
                update("tools", tool_message(source("기생충", 2019), source("옥자", 2017))),
                text_chunk("**기생충**(2019)을 추천합니다."),
                update("agent", AIMessage(content="**기생충**(2019)을 추천합니다.")),
            ]
        )
        done = events[-1]
        self.assertEqual([s["title"] for s in done["sources"]], ["기생충"])

    def test_no_tool_call_streams_straight_through(self):
        """대화 자체에 대한 질문은 도구를 안 부르므로 reset이 없어야 한다."""
        events = self.run_stream(
            [
                text_chunk("영화에 대해 물어보세요."),
                update("agent", AIMessage(content="영화에 대해 물어보세요.")),
            ]
        )
        self.assertNotIn("reset", [e["type"] for e in events])
        self.assertEqual(events[-1]["sources"], [])

    def test_recursion_limit_ends_the_turn_with_a_message(self):
        """예외를 그대로 올리면 체크포인트에 답변 없는 도구 메시지만 남는다."""
        events = self.run_stream(
            [
                text_chunk("찾아볼게요."),
                update("agent", ai_with_calls(("search_movies", {}))),
            ],
            error=GraphRecursionError("limit"),
        )

        # 서두는 쥐고 있다 버려졌으므로 취소할 것이 없다.
        self.assertEqual([e["type"] for e in events[-2:]], ["token", "done"])
        self.assertNotIn("reset", [e["type"] for e in events])
        self.assertIn("중단했습니다", events[-1]["answer"])
        self.assertEqual(events[-1]["sources"], [])
        # 다음 턴이 잔여물을 근거로 답하지 않도록 답변을 하나 붙여둔다.
        self.assertEqual(len(self.last_app.updated), 1)

    def test_disconnect_closes_the_langsmith_trace(self):
        """안 닫으면 대시보드에 'pending'으로 영원히 남는다(실측 3개).

        GeneratorExit는 Exception이 아니라 BaseException이라 LangChain의 오류
        보고 경로를 안 탄다. 우리가 직접 닫아야 한다.
        """
        기록된_run = uuid.uuid4()

        class RecordingApp(FakeApp):
            def stream(self, inputs, config, stream_mode):
                # LangGraph가 루트 실행을 시작했다고 알린다.
                for handler in config["callbacks"]:
                    handler.on_chain_start({}, {}, run_id=기록된_run, parent_run_id=None)
                yield from self.events

        graph = object.__new__(MovieRagGraph)
        graph.app = RecordingApp(
            [text_chunk("답변 앞부분"), update("agent", AIMessage(content="답변"))]
        )

        닫힌 = []
        with patch.object(rag.graph, "close_abandoned_run", 닫힌.append):
            stream = graph.stream_answer("질문", "s-1")
            next(stream)  # status
            next(stream)  # 첫 토큰까지 흘린 뒤
            stream.close()  # 클라이언트가 사라진 상황

        self.assertEqual(닫힌, [기록된_run])

    def test_normal_finish_does_not_close_anything(self):
        """정상 종료는 LangChain이 알아서 닫는다. 우리가 또 건드리면 안 된다."""
        graph = graph_with(
            [text_chunk("답변"), update("agent", AIMessage(content="답변"))]
        )
        닫힌 = []
        with patch.object(rag.graph, "close_abandoned_run", 닫힌.append):
            list(graph.stream_answer("질문", "s-1"))

        self.assertEqual(닫힌, [])

    def test_done_is_always_last_and_single(self):
        events = self.run_stream(
            [
                text_chunk("답변"),
                update("agent", AIMessage(content="답변")),
            ]
        )
        self.assertEqual([e["type"] for e in events].count("done"), 1)
        self.assertEqual(events[-1]["type"], "done")


if __name__ == "__main__":
    unittest.main()
