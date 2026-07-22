"""
FastAPI 래핑 (LangGraph 버전).

- 앱이 뜰 때(lifespan) StateGraph를 1회 컴파일해 파이프라인을 준비한다.
  (매 기동·매 요청마다 재인덱싱/모델 재로드를 하지 않는다.)
- POST /query : 질문 → 답변 + 출처. session_id 로 멀티턴 대화 맥락 유지.
- GET  /health: 헬스 체크.

실행:
    uvicorn rag.api:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.graph import MovieRagGraph


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 그래프를 1회 컴파일(임베딩 모델·LLM·벡터스토어 1회 로드).
    app.state.graph = MovieRagGraph()
    yield


app = FastAPI(title="영화 정보 RAG API (LangGraph)", version="2.0.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(..., description="질문")
    session_id: str | None = Field(
        None, description="대화 세션 ID. 같은 값을 보내면 이전 대화 맥락이 이어짐(멀티턴)."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "기생충 감독이 누구야?"},
                {"question": "그 감독의 다른 영화도 알려줘", "session_id": "user-123"},
            ]
        }
    }


class SourceModel(BaseModel):
    title: str
    year: int
    director: str
    genres: str
    country: str
    vote_average: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceModel]


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
    except Exception as exc:  # noqa: BLE001 - API 경계에서 에러를 메시지로 변환
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceModel(**s) for s in result["sources"]],
    )
