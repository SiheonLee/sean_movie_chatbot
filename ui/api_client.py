"""Streamlit UI에서 FastAPI 영화 RAG 서버를 호출하는 작은 HTTP 클라이언트."""

from __future__ import annotations

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


class ApiClientError(RuntimeError):
    """UI에 안전하게 표시할 수 있는 API 호출 오류."""


class RagApiClient:
    """`/health`와 `/query` 호출만 담당한다."""

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
        except httpx.TimeoutException as exc:
            raise ApiClientError(
                "답변 생성 시간이 너무 길어 요청을 종료했습니다. 다시 시도해주세요."
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 422:
                message = "질문 내용을 확인한 뒤 다시 입력해주세요."
            else:
                message = "CineBot이 답변을 만들지 못했습니다. 잠시 후 다시 시도해주세요."
            raise ApiClientError(message) from exc
        except (httpx.RequestError, ValueError) as exc:
            raise ApiClientError(
                "영화 정보 서버에 연결할 수 없습니다. API 연결 상태를 확인해주세요."
            ) from exc

        answer = payload.get("answer")
        sources = payload.get("sources")
        if not isinstance(answer, str) or not isinstance(sources, list):
            raise ApiClientError("서버 응답 형식을 확인할 수 없습니다.")
        return {"answer": answer, "sources": sources}
