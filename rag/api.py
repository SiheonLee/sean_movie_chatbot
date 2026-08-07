"""
FastAPI 래핑 (LangGraph 버전).

- 앱이 뜰 때(lifespan) StateGraph를 1회 컴파일해 파이프라인을 준비한다.
  (매 기동·매 요청마다 재인덱싱/모델 재로드를 하지 않는다.)
- POST /query        : 질문 → 답변 + 출처. session_id 로 멀티턴 대화 맥락 유지.
- POST /query/stream : 같은 답변을 SSE로 흘려보낸다(진행 상태 + 토큰).
- GET  /health       : 헬스 체크.

실행:
    uvicorn rag.api:app --reload
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from rag.graph import MovieRagGraph

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 그래프를 1회 컴파일(임베딩 모델·LLM·벡터스토어 1회 로드).
    app.state.graph = MovieRagGraph()
    yield


app = FastAPI(title="영화 정보 RAG API (LangGraph)", version="2.0.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="1~500자의 질문",
    )
    session_id: str | None = Field(
        None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
        description="대화 세션 ID. 같은 값을 보내면 이전 대화 맥락이 이어짐(멀티턴).",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "기생충 감독이 누구야?", "session_id": "user-123"},
            ]
        }
    }

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("질문을 입력해주세요.")
        return question


class SourceModel(BaseModel):
    # D3 이전 응답에는 없던 필드라 기본값을 둔다. 새 도구 결과는 실제 TMDB ID를
    # 채워 제목이 같거나 짧아도 문자열 비교 없이 식별할 수 있다.
    movie_id: int = 0
    title: str
    year: int
    director: str
    # 배역명까지 함께 온다("송강호(Kim Ki-taek 역)"). 무드 검색 결과는 색인에
    # 출연진이 없어 빈 문자열이므로 기본값을 둔다.
    cast: str = ""
    genres: str
    country: str
    vote_average: float
    # 포스터가 없는 영화가 응답 전체를 실패시키지 않도록 기본값을 둔다.
    # 값은 경로("/abc.jpg")이고 이미지 크기는 UI가 정한다.
    poster_path: str = ""
    snippet: str


class WebSourceModel(BaseModel):
    title: str
    url: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceModel]
    # 기존 sources는 영화 카드 전용으로 유지하고 웹 페이지는 별도 필드로 추가한다.
    web_sources: list[WebSourceModel] = Field(default_factory=list)
    # 무엇을 보고 답했는지("tmdb", "local", "web", "justwatch"). 표시 문구는 UI가
    # 정하고 여기서는 식별자만 넘긴다. JustWatch는 표기가 의무라 반드시 실린다.
    attributions: list[str] = []


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    try:
        result = app.state.graph.answer(
            question=req.question,
            session_id=req.session_id,
        )
    except Exception as exc:  # noqa: BLE001 - API 경계에서 내부 오류를 기록하고 숨김
        logger.exception("영화 질문 처리 중 오류가 발생했습니다.")
        raise HTTPException(
            status_code=500,
            detail="답변을 생성하지 못했습니다. 잠시 후 다시 시도해주세요.",
        ) from exc

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceModel(**s) for s in result["sources"]],
        web_sources=[
            WebSourceModel(**s) for s in result.get("web_sources", [])
        ],
        attributions=result.get("attributions", []),
    )


def _sse(event: dict) -> str:
    """이벤트 하나를 SSE 프레임으로. 한글이 \\uXXXX로 부풀지 않게 그대로 싣는다."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


STREAM_ERROR_MESSAGE = "답변을 생성하지 못했습니다. 잠시 후 다시 시도해주세요."


def _stream_events(graph, question: str, session_id: str | None) -> Iterator[str]:
    """그래프 이벤트를 SSE 프레임으로 옮긴다.

    출처는 /query와 같은 SourceModel을 통과시킨다. 두 엔드포인트가 같은 스키마를
    내보내야 UI의 카드 렌더링 코드를 하나로 유지할 수 있다.
    """
    try:
        for event in graph.stream_answer(question=question, session_id=session_id):
            if event.get("type") == "done":
                event = {
                    **event,
                    "sources": [
                        SourceModel(**s).model_dump() for s in event.get("sources", [])
                    ],
                    "web_sources": [
                        WebSourceModel(**s).model_dump()
                        for s in event.get("web_sources", [])
                    ],
                }
            yield _sse(event)
    except Exception:  # noqa: BLE001 - API 경계에서 내부 오류를 기록하고 숨김
        # 응답이 이미 시작됐으면 500을 보낼 수 없다. 오류도 이벤트로 알린다.
        logger.exception("영화 질문 스트리밍 중 오류가 발생했습니다.")
        yield _sse({"type": "error", "message": STREAM_ERROR_MESSAGE})


@app.post("/query/stream")
def query_stream(req: QueryRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_events(app.state.graph, req.question, req.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # 프록시가 응답을 모아뒀다 한 번에 보내면 스트리밍이 의미를 잃는다.
            "X-Accel-Buffering": "no",
        },
    )
