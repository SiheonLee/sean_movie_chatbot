"""Streamlit UI에서 FastAPI 영화 RAG 서버를 호출하는 작은 HTTP 클라이언트."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import TypedDict

import httpx


class Source(TypedDict):
    title: str
    year: int
    director: str
    cast: str
    genres: str
    country: str
    vote_average: float
    poster_path: str  # TMDB 이미지 경로. 렌더링은 UI가 크기를 붙여 사용한다.
    snippet: str


class QueryResult(TypedDict):
    answer: str
    sources: list[Source]


class StreamEvent(TypedDict, total=False):
    """`/query/stream`이 흘려보내는 이벤트.

    type이 무엇이냐에 따라 채워지는 키가 다르다.

    ==========  ==============================================================
    ``status``  ``text`` — 진행 상태 한 줄. 계속 교체해 보여준다.
    ``token``   ``text`` — 답변 텍스트 조각.
    ``reset``   (없음) — 지금까지 받은 token을 버린다.
    ``done``    ``answer``, ``sources`` — 확정된 답변. 마지막에 한 번 온다.
    ==========  ==============================================================
    """

    type: str
    text: str
    answer: str
    sources: list[Source]


class ApiClientError(RuntimeError):
    """UI에 안전하게 표시할 수 있는 API 호출 오류."""


_TIMEOUT_MESSAGE = "답변 생성 시간이 너무 길어 요청을 종료했습니다. 다시 시도해주세요."
_INVALID_QUESTION_MESSAGE = "질문 내용을 확인한 뒤 다시 입력해주세요."
_SERVER_MESSAGE = "CineBot이 답변을 만들지 못했습니다. 잠시 후 다시 시도해주세요."
_OFFLINE_MESSAGE = "영화 정보 서버에 연결할 수 없습니다. API 연결 상태를 확인해주세요."
_MALFORMED_MESSAGE = "서버 응답 형식을 확인할 수 없습니다."

# SSE 프레임은 "data: {...}" 한 줄과 빈 줄로 온다.
_SSE_DATA_PREFIX = "data: "


def _to_client_error(exc: Exception) -> ApiClientError:
    """httpx 예외를 UI에 그대로 띄울 수 있는 문구로 옮긴다."""
    if isinstance(exc, httpx.TimeoutException):
        return ApiClientError(_TIMEOUT_MESSAGE)
    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response.status_code == 422:
            return ApiClientError(_INVALID_QUESTION_MESSAGE)
        return ApiClientError(_SERVER_MESSAGE)
    return ApiClientError(_OFFLINE_MESSAGE)


class RagApiClient:
    """`/health`, `/query`, `/query/stream` 호출만 담당한다."""

    def __init__(
        self,
        base_url: str,
        *,
        query_timeout: float = 120.0,
        health_timeout: float = 3.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.query_timeout = query_timeout
        self.health_timeout = health_timeout

    def is_healthy(self) -> bool:
        try:
            response = httpx.get(
                f"{self.base_url}/health",
                timeout=self.health_timeout,
            )
            response.raise_for_status()
            return response.json().get("status") == "ok"
        except (httpx.HTTPError, ValueError):
            return False

    def query(self, question: str, session_id: str) -> QueryResult:
        try:
            response = httpx.post(
                f"{self.base_url}/query",
                json={"question": question, "session_id": session_id},
                timeout=self.query_timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise _to_client_error(exc) from exc

        answer = payload.get("answer")
        sources = payload.get("sources")
        if not isinstance(answer, str) or not isinstance(sources, list):
            raise ApiClientError(_MALFORMED_MESSAGE)
        return {"answer": answer, "sources": sources}

    def stream_query(self, question: str, session_id: str) -> Iterator[StreamEvent]:
        """`/query/stream`의 SSE를 이벤트 단위로 흘려준다.

        서버가 ``error`` 이벤트를 보내거나 ``done`` 없이 끊기면 ApiClientError를
        올린다. 호출부가 query()와 같은 방식으로 오류를 처리할 수 있게 한다.
        """
        completed = False
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/query/stream",
                json={"question": question, "session_id": session_id},
                timeout=self.query_timeout,
            ) as response:
                if response.status_code >= 400:
                    # 본문을 읽지 않은 스트림은 raise_for_status()가 상태를
                    # 조립하지 못한다. 먼저 비운 뒤 올린다.
                    response.read()
                    response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith(_SSE_DATA_PREFIX):
                        continue
                    try:
                        event: StreamEvent = json.loads(
                            line[len(_SSE_DATA_PREFIX) :]
                        )
                    except json.JSONDecodeError as exc:
                        raise ApiClientError(_MALFORMED_MESSAGE) from exc
                    kind = event.get("type")
                    if kind == "error":
                        # 서버가 이미 사용자에게 보여도 되는 문장만 담아 보낸다.
                        message = event.get("message")
                        raise ApiClientError(
                            message if isinstance(message, str) else _SERVER_MESSAGE
                        )
                    if kind == "done":
                        completed = True
                    yield event
        except httpx.HTTPError as exc:
            raise _to_client_error(exc) from exc

        if not completed:
            # 답변이 확정되기 전에 연결이 끊겼다. 반쪽짜리 텍스트를 완성된 답변인
            # 것처럼 남기지 않는다.
            raise ApiClientError(_SERVER_MESSAGE)
