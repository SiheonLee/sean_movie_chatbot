"""그래프 계층 테스트: 출처 수집의 턴 경계와 중복 제거, 스트리밍 이벤트."""

from __future__ import annotations

import unittest
import uuid
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
    collect_turn_sources,
    collect_turn_tool_calls,
    tool_status,
)


def source(title: str, year: int = 2020) -> dict:
    return {
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


class UsedSourcesTests(unittest.TestCase):
    """카드는 답변이 소개한 순서를 따라가야 한다."""

    def test_cards_follow_the_answer_not_the_tool(self):
        """도구는 평점순, 답변은 질문에 맞는 순. 둘이 어긋나면 카드가 헷갈린다."""
        sources = [source("기생충", 2019), source("올드보이", 2003), source("아가씨", 2016)]
        answer = "아가씨(2016)를 먼저 추천합니다. 다음은 기생충(2019), 마지막으로 올드보이(2003)입니다."

        used = MovieRagGraph._used_sources(answer, sources)

        self.assertEqual([s["title"] for s in used], ["아가씨", "기생충", "올드보이"])

    def test_unmentioned_movies_are_still_dropped(self):
        sources = [source("기생충", 2019), source("옥자", 2017)]
        answer = "기생충(2019)만 추천합니다."

        used = MovieRagGraph._used_sources(answer, sources)

        self.assertEqual([s["title"] for s in used], ["기생충"])

    def test_ties_keep_tool_order(self):
        """같은 제목 다른 연도는 언급 위치가 같다. 도구 순서를 뒤집지 않는다."""
        sources = [source("괴물", 2006), source("괴물", 2023)]
        answer = "괴물을 추천합니다."

        used = MovieRagGraph._used_sources(answer, sources)

        self.assertEqual([s["year"] for s in used], [2006, 2023])

    def test_falls_back_to_tool_order_when_nothing_matches(self):
        """제목이 하나도 안 걸리면 카드 0개보다 도구 순서 그대로가 낫다."""
        sources = [source("기생충", 2019), source("옥자", 2017)]

        used = MovieRagGraph._used_sources("추천드릴 작품을 찾지 못했습니다.", sources)

        self.assertEqual([s["title"] for s in used], ["기생충", "옥자"])


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

    def test_done_carries_only_sources_the_answer_used(self):
        """도구는 넓게 훑고 답변은 일부만 쓴다. 카드는 답변을 따라가야 한다."""
        events = self.run_stream(
            [
                update("agent", ai_with_calls(("search_movies", {}))),
                update("tools", tool_message(source("기생충", 2019), source("옥자", 2017))),
                text_chunk("기생충을 추천합니다."),
                update("agent", AIMessage(content="기생충을 추천합니다.")),
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
