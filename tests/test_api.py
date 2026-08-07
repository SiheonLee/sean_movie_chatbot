from __future__ import annotations

import json
import unittest
from unittest.mock import Mock

from fastapi import HTTPException
from pydantic import ValidationError

from rag.api import STREAM_ERROR_MESSAGE, QueryRequest, _stream_events, app, query


class QueryRequestTests(unittest.TestCase):
    def test_question_is_trimmed(self):
        request = QueryRequest(question="  기생충 감독은?  ", session_id="test-session")

        self.assertEqual(request.question, "기생충 감독은?")

    def test_whitespace_question_is_rejected(self):
        with self.assertRaises(ValidationError):
            QueryRequest(question="   ")

    def test_invalid_session_id_is_rejected(self):
        with self.assertRaises(ValidationError):
            QueryRequest(question="기생충 감독은?", session_id="공백 포함")


class QueryEndpointTests(unittest.TestCase):
    def setUp(self):
        self.original_graph = getattr(app.state, "graph", None)

    def tearDown(self):
        if self.original_graph is None:
            app.state._state.pop("graph", None)
        else:
            app.state.graph = self.original_graph

    def test_query_returns_answer_and_sources(self):
        app.state.graph = Mock()
        app.state.graph.answer.return_value = {
            "answer": "기생충의 감독은 봉준호입니다.",
            "sources": [
                {
                    "title": "기생충",
                    "year": 2019,
                    "director": "봉준호",
                    "genres": "코미디, 스릴러, 드라마",
                    "country": "한국",
                    "vote_average": 8.5,
                    "snippet": "제목: 기생충",
                }
            ],
        }

        response = query(QueryRequest(question="기생충 감독은?", session_id="session-1"))

        self.assertEqual(response.answer, "기생충의 감독은 봉준호입니다.")
        self.assertEqual(response.sources[0].title, "기생충")

    def test_attributions_ride_along_with_the_answer(self):
        """답변만 봐서는 웹에서 찾아온 것인지 로컬 색인에서 고른 것인지 알 수 없다."""
        app.state.graph = Mock()
        app.state.graph.answer.return_value = {
            "answer": "넷플릭스에서 볼 수 있습니다.",
            "sources": [],
            "attributions": ["tmdb", "justwatch"],
        }

        response = query(QueryRequest(question="기생충 어디서 봐?"))

        self.assertEqual(response.attributions, ["tmdb", "justwatch"])

    def test_missing_attributions_default_to_empty(self):
        app.state.graph = Mock()
        app.state.graph.answer.return_value = {"answer": "안녕하세요.", "sources": []}

        response = query(QueryRequest(question="안녕?"))

        self.assertEqual(response.attributions, [])

    def test_internal_error_is_not_exposed(self):
        app.state.graph = Mock()
        app.state.graph.answer.side_effect = RuntimeError("secret internal error")

        with self.assertRaises(HTTPException) as raised:
            query(QueryRequest(question="기생충 감독은?"))

        self.assertEqual(raised.exception.status_code, 500)
        self.assertNotIn("secret internal error", raised.exception.detail)


def frames(events: list) -> list[dict]:
    graph = Mock()
    graph.stream_answer.return_value = iter(events)
    return [
        json.loads(frame.removeprefix("data: ").strip())
        for frame in _stream_events(graph, "기생충 감독은?", "session-1")
    ]


class StreamEventTests(unittest.TestCase):
    def test_done_carries_attributions(self):
        """스트리밍 답변에도 출처 표기가 실려야 한다 — JustWatch는 표기가 의무다."""
        events = frames(
            [
                {
                    "type": "done",
                    "answer": "넷플릭스에서 볼 수 있습니다.",
                    "sources": [],
                    "attributions": ["tmdb", "justwatch"],
                }
            ]
        )

        self.assertEqual(events[0]["attributions"], ["tmdb", "justwatch"])

    def test_events_are_sse_frames(self):
        graph = Mock()
        graph.stream_answer.return_value = iter([{"type": "token", "text": "봉준호"}])

        raw = list(_stream_events(graph, "기생충 감독은?", "session-1"))

        self.assertEqual(raw, ['data: {"type": "token", "text": "봉준호"}\n\n'])

    def test_korean_is_not_escaped(self):
        """\\uXXXX로 부풀면 프레임이 3배가 된다."""
        graph = Mock()
        graph.stream_answer.return_value = iter([{"type": "status", "text": "찾는 중"}])

        self.assertIn("찾는 중", next(iter(_stream_events(graph, "q", None))))

    def test_done_sources_pass_through_source_model(self):
        """/query와 같은 스키마여야 UI 카드 렌더링 코드가 하나로 유지된다."""
        events = frames(
            [
                {
                    "type": "done",
                    "answer": "기생충입니다.",
                    "sources": [
                        {
                            "title": "기생충",
                            "year": 2019,
                            "director": "봉준호",
                            "genres": "드라마",
                            "country": "한국",
                            "vote_average": 8.5,
                            "snippet": "요약",
                        }
                    ],
                }
            ]
        )

        source = events[0]["sources"][0]
        # 도구가 안 채운 선택 필드도 기본값으로 채워져 나간다.
        self.assertEqual(source["cast"], "")
        self.assertEqual(source["poster_path"], "")
        self.assertEqual(source["title"], "기생충")

    def test_internal_error_becomes_an_error_event(self):
        """응답이 시작된 뒤에는 500을 보낼 수 없다. 오류도 이벤트로 알린다."""

        def explode(**kwargs):
            yield {"type": "token", "text": "부"}
            raise RuntimeError("secret internal error")

        graph = Mock()
        graph.stream_answer.side_effect = explode

        events = [
            json.loads(frame.removeprefix("data: ").strip())
            for frame in _stream_events(graph, "q", None)
        ]

        self.assertEqual(events[-1], {"type": "error", "message": STREAM_ERROR_MESSAGE})
        self.assertNotIn("secret internal error", json.dumps(events, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
