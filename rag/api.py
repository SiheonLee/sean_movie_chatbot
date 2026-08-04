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

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
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
    except Exception as exc:  # noqa: BLE001 - API 경계에서 내부 오류를 기록하고 숨김
        logger.exception("영화 질문 처리 중 오류가 발생했습니다.")
        raise HTTPException(
            status_code=500,
            detail="답변을 생성하지 못했습니다. 잠시 후 다시 시도해주세요.",
        ) from exc

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceModel(**s) for s in result["sources"]],
    )
