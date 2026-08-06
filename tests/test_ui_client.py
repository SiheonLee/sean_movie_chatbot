from __future__ import annotations

import json
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


def sse(*events: dict) -> list[str]:
    """SSE 프레임을 줄 단위로. 프레임 사이의 빈 줄까지 그대로 흉내낸다."""
    lines: list[str] = []
    for event in events:
        lines.append(f"data: {json.dumps(event, ensure_ascii=False)}")
        lines.append("")
    return lines


class FakeStream:
    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self.lines = lines
        self.status_code = status_code
        self.was_read = False

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *exc_info) -> bool:
        return False

    def read(self) -> bytes:
        self.was_read = True
        return b""

    def raise_for_status(self) -> None:
        raise httpx.HTTPStatusError("boom", request=Mock(), response=self)

    def iter_lines(self):
        return iter(self.lines)


class StreamQueryTests(unittest.TestCase):
    def setUp(self):
        self.client = RagApiClient("http://127.0.0.1:8000/")

    @patch("ui.api_client.httpx.stream")
    def test_parses_frames_in_order(self, stream: Mock):
        stream.return_value = FakeStream(
            sse(
                {"type": "status", "text": "영화를 찾는 중"},
                {"type": "token", "text": "기생충"},
                {"type": "done", "answer": "기생충입니다.", "sources": []},
            )
        )

        events = list(self.client.stream_query("기생충 감독은?", "session-1"))

        self.assertEqual([e["type"] for e in events], ["status", "token", "done"])
        self.assertEqual(events[-1]["answer"], "기생충입니다.")
        stream.assert_called_once_with(
            "POST",
            "http://127.0.0.1:8000/query/stream",
            json={"question": "기생충 감독은?", "session_id": "session-1"},
            timeout=120.0,
        )

    @patch("ui.api_client.httpx.stream")
    def test_error_event_becomes_client_error(self, stream: Mock):
        stream.return_value = FakeStream(
            sse({"type": "error", "message": "답변을 생성하지 못했습니다."})
        )

        with self.assertRaisesRegex(ApiClientError, "생성하지 못했습니다"):
            list(self.client.stream_query("기생충 감독은?", "session-1"))

    @patch("ui.api_client.httpx.stream")
    def test_stream_cut_before_done_is_an_error(self, stream: Mock):
        """반쪽짜리 텍스트를 완성된 답변인 것처럼 남기면 안 된다."""
        stream.return_value = FakeStream(sse({"type": "token", "text": "기생"}))

        with self.assertRaises(ApiClientError):
            list(self.client.stream_query("기생충 감독은?", "session-1"))

    @patch("ui.api_client.httpx.stream")
    def test_http_error_is_translated(self, stream: Mock):
        response = FakeStream([], status_code=422)
        stream.return_value = response

        with self.assertRaisesRegex(ApiClientError, "질문 내용"):
            list(self.client.stream_query("기생충 감독은?", "session-1"))
        # 본문을 읽지 않은 스트림은 raise_for_status()가 상태를 조립하지 못한다.
        self.assertTrue(response.was_read)

    @patch("ui.api_client.httpx.stream")
    def test_timeout_is_translated(self, stream: Mock):
        stream.side_effect = httpx.ReadTimeout("slow")

        with self.assertRaisesRegex(ApiClientError, "시간"):
            list(self.client.stream_query("기생충 감독은?", "session-1"))

    @patch("ui.api_client.httpx.stream")
    def test_malformed_frame_is_reported(self, stream: Mock):
        stream.return_value = FakeStream(["data: {not json", ""])

        with self.assertRaisesRegex(ApiClientError, "형식"):
            list(self.client.stream_query("기생충 감독은?", "session-1"))


if __name__ == "__main__":
    unittest.main()
