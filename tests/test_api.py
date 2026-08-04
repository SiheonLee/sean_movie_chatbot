from __future__ import annotations

import unittest
from unittest.mock import Mock

from fastapi import HTTPException
from pydantic import ValidationError

from rag.api import QueryRequest, app, query


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

    def test_internal_error_is_not_exposed(self):
        app.state.graph = Mock()
        app.state.graph.answer.side_effect = RuntimeError("secret internal error")

        with self.assertRaises(HTTPException) as raised:
            query(QueryRequest(question="기생충 감독은?"))

        self.assertEqual(raised.exception.status_code, 500)
        self.assertNotIn("secret internal error", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()
