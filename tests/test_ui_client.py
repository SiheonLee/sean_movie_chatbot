from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import httpx

from ui.api_client import ApiClientError, RagApiClient


class RagApiClientTests(unittest.TestCase):
    def setUp(self):
        self.client = RagApiClient("http://127.0.0.1:8000/")

    @patch("ui.api_client.httpx.get")
    def test_health_returns_true_for_ok_response(self, get: Mock):
        response = Mock()
        response.json.return_value = {"status": "ok"}
        get.return_value = response

        self.assertTrue(self.client.is_healthy())
        get.assert_called_once_with("http://127.0.0.1:8000/health", timeout=3.0)

    @patch("ui.api_client.httpx.get")
    def test_health_returns_false_when_api_is_unreachable(self, get: Mock):
        get.side_effect = httpx.ConnectError("offline")

        self.assertFalse(self.client.is_healthy())

    @patch("ui.api_client.httpx.post")
    def test_query_sends_question_and_session(self, post: Mock):
        response = Mock()
        response.json.return_value = {
            "answer": "기생충의 감독은 봉준호입니다.",
            "sources": [],
        }
        post.return_value = response

        result = self.client.query("기생충 감독은?", "session-1")

        self.assertEqual(result["answer"], "기생충의 감독은 봉준호입니다.")
        post.assert_called_once_with(
            "http://127.0.0.1:8000/query",
            json={"question": "기생충 감독은?", "session_id": "session-1"},
            timeout=120.0,
        )

    @patch("ui.api_client.httpx.post")
    def test_query_translates_timeout_to_safe_message(self, post: Mock):
        post.side_effect = httpx.ReadTimeout("slow")

        with self.assertRaisesRegex(ApiClientError, "시간"):
            self.client.query("기생충 감독은?", "session-1")


if __name__ == "__main__":
    unittest.main()
