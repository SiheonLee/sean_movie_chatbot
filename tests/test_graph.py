"""그래프 계층 테스트: 출처 수집의 턴 경계와 중복 제거."""

from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rag.graph import collect_turn_sources, collect_turn_tool_calls


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


if __name__ == "__main__":
    unittest.main()
