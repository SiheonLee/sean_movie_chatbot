"""LangGraph StateGraph 기반 영화 RAG 그래프.

질문 유형을 먼저 분류한 뒤 서로 다른 경로로 처리한다.

- 영화 사실/추천: 대화 맥락화 → 필터 분석 → 점수 검색 → 문서별 평가 → 근거 답변
- 대화/메타: 검색 없이 대화 이력으로 답변
- 집계/정렬: 전체 영화 데이터에서 결정적으로 계산

검색 결과가 부족하면 검색용 질문만 재작성하고 필터부터 다시 계산한다. 사용자 원문과
대화 이력은 바꾸지 않는다.
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from rag.config import settings
from rag.indexing import country_ko, get_vectorstore, load_movies
from rag.providers import get_llm

logger = logging.getLogger(__name__)

QuestionRoute = Literal["fact", "conversation", "aggregate"]


class DocumentGrade(BaseModel):
    """검색 문서 하나의 관련성 판정."""

    document_id: int = Field(description="0부터 시작하는 문서 번호")
    relevant: Literal["yes", "no"] = Field(
        description="질문에 직접 답할 근거가 있으면 yes, 아니면 no"
    )


class DocumentGrades(BaseModel):
    """한 번의 LLM 호출로 받는 문서별 관련성 판정."""

    grades: list[DocumentGrade]


class GroundedAnswer(BaseModel):
    """답변과 실제 사용한 문서 번호."""

    answer: str = Field(description="사용자에게 보여줄 한국어 답변")
    source_ids: list[int] = Field(
        default_factory=list,
        description="답변 작성에 실제 사용한 문서 번호. 근거가 없으면 빈 목록",
    )


class RagState(TypedDict, total=False):
    question: str                              # 외부 입력과의 하위 호환용 사용자 원문
    original_question: str                     # 절대 재작성하지 않는 현재 사용자 원문
    search_query: str                          # 맥락화/재작성된 검색 전용 질문
    route: QuestionRoute                       # fact | conversation | aggregate
    messages: Annotated[list, add_messages]    # 멀티턴 대화 이력
    filters: dict                              # 검색용 질문에서 추출한 구조 필터
    documents: list[Document]                  # 거리/관련성 평가를 통과한 문서만 보존
    document_scores: list[float]               # documents와 같은 순서의 Chroma 거리
    relevant: bool
    retries: int
    answer: str
    sources: list


# ---------------------------------------------------------------------------
# 규칙 기반 질의 분석과 라우팅
# ---------------------------------------------------------------------------

_GENRES = [
    "액션", "모험", "애니메이션", "코미디", "범죄", "다큐멘터리", "드라마", "가족",
    "판타지", "역사", "공포", "음악", "미스터리", "로맨스", "SF", "스릴러", "전쟁", "서부",
]
_COUNTRY = {
    "한국": "KR", "국내": "KR", "미국": "US", "할리우드": "US", "일본": "JP",
    "영국": "GB", "프랑스": "FR", "중국": "CN", "홍콩": "HK", "인도": "IN",
}
_YEAR_RE = re.compile(r"(19\d{2}|20\d{2})")
_RATING_RE = re.compile(r"평점\s*(\d+(?:\.\d+)?)\s*이상")

_CONVERSATION_RE = re.compile(
    r"(내가|우리가).{0,16}(말|질문|물어|언급)|"
    r"(이전|아까|방금).{0,16}(대화|말|질문|언급)|"
    r"(우리 대화|대화 내용|내 질문|무슨 질문)|"
    r"^(안녕|반가워|고마워|감사|도움말|무엇을 할 수)",
    re.IGNORECASE,
)
_AGGREGATE_RES = (
    re.compile(r"(평점|별점).*(가장|제일|최고|최저|높|낮|순)"),
    re.compile(r"(가장|제일|최고|최저|높|낮).*(평점|별점)"),
    re.compile(r"(가장|제일).*(오래된|최신|최근)"),
    re.compile(r"(오래된|최신|최근).*(순|영화)"),
    re.compile(r"(몇\s*편|몇\s*개|개수|총\s*몇)"),
)
_GRADE_LINE_RE = re.compile(r"^(?:문서\s*)?(\d+)\s*:\s*(yes|no)\s*$", re.IGNORECASE)
_SOURCE_LINE_RE = re.compile(r"^SOURCE_IDS:\s*(NONE|\d+(?:\s*,\s*\d+)*)\s*$", re.IGNORECASE)


def classify_question(question: str) -> QuestionRoute:
    """질문을 대화/집계/영화 사실 경로로 결정적으로 분류한다."""
    q = question.strip()
    if _CONVERSATION_RE.search(q):
        return "conversation"
    if any(pattern.search(q) for pattern in _AGGREGATE_RES):
        return "aggregate"
    return "fact"


def parse_document_grades(text: str, document_count: int) -> set[int]:
    """`문서 N: yes|no`만 허용하고 누락·중복·설명 텍스트는 거부한다."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    parsed: dict[int, str] = {}
    for line in lines:
        match = _GRADE_LINE_RE.fullmatch(line)
        if not match:
            raise ValueError(f"허용되지 않은 관련성 판정 형식: {line!r}")
        document_id, verdict = int(match.group(1)), match.group(2).lower()
        if not 0 <= document_id < document_count or document_id in parsed:
            raise ValueError(f"잘못되거나 중복된 문서 번호: {document_id}")
        parsed[document_id] = verdict
    if set(parsed) != set(range(document_count)):
        raise ValueError("모든 검색 문서에 대한 판정이 필요합니다.")
    return {document_id for document_id, verdict in parsed.items() if verdict == "yes"}


def parse_grounded_answer(text: str, document_count: int) -> GroundedAnswer:
    """마지막 `SOURCE_IDS:` 줄을 엄격히 파싱해 답변과 사용 문서를 분리한다."""
    lines = text.strip().splitlines()
    if len(lines) < 2:
        raise ValueError("답변과 SOURCE_IDS 줄이 모두 필요합니다.")
    match = _SOURCE_LINE_RE.fullmatch(lines[-1].strip())
    if not match:
        raise ValueError("응답 마지막 줄에 올바른 SOURCE_IDS가 없습니다.")
    source_text = match.group(1).upper()
    source_ids = [] if source_text == "NONE" else [int(value.strip()) for value in source_text.split(",")]
    if any(not 0 <= document_id < document_count for document_id in source_ids):
        raise ValueError("존재하지 않는 출처 번호가 포함되었습니다.")
    answer = "\n".join(lines[:-1]).strip()
    if not answer:
        raise ValueError("답변 본문이 비어 있습니다.")
    return GroundedAnswer(answer=answer, source_ids=source_ids)


def infer_grounded_answer_from_titles(text: str, docs: list[Document]) -> GroundedAnswer:
    """SOURCE_IDS가 없을 때 답변에 명시된 정확한 영화 제목만 출처로 인정한다."""
    answer_lines = [
        line for line in text.strip().splitlines() if not line.strip().upper().startswith("SOURCE_IDS:")
    ]
    answer = "\n".join(answer_lines).strip()
    source_ids = [
        index
        for index, doc in enumerate(docs)
        if (title := str(doc.metadata.get("title", "")).strip()) and title in answer
    ]
    if not answer or not source_ids:
        raise ValueError("답변에서 검색 문서의 정확한 영화 제목을 확인할 수 없습니다.")
    return GroundedAnswer(answer=answer, source_ids=source_ids)


def extract_filters(question: str) -> dict:
    """질문 문자열에서 연도/장르/국가/평점 조건을 규칙 기반으로 추출한다."""
    filters: dict = {}

    years = [int(y) for y in _YEAR_RE.findall(question)]
    if years:
        if len(years) >= 2:
            filters["year_from"], filters["year_to"] = min(years), max(years)
        else:
            y = years[0]
            if any(k in question for k in ("이후", "부터", "이래")):
                filters["year_from"] = y
            elif any(k in question for k in ("이전", "까지", "이하")):
                filters["year_to"] = y
            else:
                filters["year_from"] = filters["year_to"] = y

    for genre in _GENRES:
        if genre in question:
            filters["genre"] = genre
            break

    for word, code in _COUNTRY.items():
        if word in question:
            filters["country"] = code
            break

    match = _RATING_RE.search(question)
    if match:
        filters["min_rating"] = float(match.group(1))

    return filters


def _metadata_filter(filters: dict) -> dict | None:
    """연도/국가/평점 조건을 Chroma 메타데이터 where 필터로 변환한다."""
    conds: list[dict] = []
    if "year_from" in filters:
        conds.append({"year": {"$gte": filters["year_from"]}})
    if "year_to" in filters:
        conds.append({"year": {"$lte": filters["year_to"]}})
    if filters.get("country"):
        conds.append({"country": {"$eq": filters["country"]}})
    if "min_rating" in filters:
        conds.append({"vote_average": {"$gte": filters["min_rating"]}})

    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def _format_docs(docs: list[Document], *, numbered: bool = False) -> str:
    if numbered:
        return "\n\n".join(
            f"[문서 {index}]\n{doc.page_content}" for index, doc in enumerate(docs)
        )
    return "\n\n".join(doc.page_content for doc in docs)


def _has_exact_metadata_match(search_query: str, doc: Document) -> bool:
    """검색어에 문서의 제목 또는 감독명이 정확히 있으면 결정적 관련 문서로 본다."""
    query = search_query.casefold()
    for key in ("title", "director"):
        value = str(doc.metadata.get(key, "")).strip()
        if len(value) >= 2 and value.casefold() in query:
            return True
    return False


def _is_explicitly_excluded(search_query: str, doc: Document) -> bool:
    """`영화명 제외/을 제외/를 제외`로 명시한 문서를 검색 결과에서 제거한다."""
    title = str(doc.metadata.get("title", "")).strip()
    if not title:
        return False
    return any(
        pattern in search_query
        for pattern in (f"{title} 제외", f"{title}을 제외", f"{title}를 제외")
    )


def _to_source(doc: Document) -> dict:
    text = doc.page_content.strip().replace("\n", " ")
    genres = (doc.metadata.get("genres", "") or "").strip("|").replace("|", ", ")
    return {
        "title": doc.metadata.get("title", ""),
        "year": int(doc.metadata.get("year", 0) or 0),
        "director": doc.metadata.get("director", ""),
        "genres": genres,
        "country": country_ko(doc.metadata.get("country", "")),
        "vote_average": float(doc.metadata.get("vote_average", 0.0) or 0.0),
        "snippet": (text[:160] + "…") if len(text) > 160 else text,
    }


def _movie_to_source(movie: dict) -> dict:
    genres = ", ".join(movie.get("genres") or [])
    parts = [
        f"제목: {movie.get('title', '')}",
        f"개봉연도: {movie.get('year', 0)}",
        f"감독: {movie.get('director', '')}",
        f"장르: {genres}",
        f"평점: {movie.get('vote_average', 0.0)}",
    ]
    text = " | ".join(parts)
    return {
        "title": movie.get("title", ""),
        "year": int(movie.get("year", 0) or 0),
        "director": movie.get("director", ""),
        "genres": genres,
        "country": country_ko(movie.get("country", "")),
        "vote_average": float(movie.get("vote_average", 0.0) or 0.0),
        "snippet": (text[:160] + "…") if len(text) > 160 else text,
    }


def _matches_filters(movie: dict, filters: dict) -> bool:
    year = int(movie.get("year", 0) or 0)
    if "year_from" in filters and year < filters["year_from"]:
        return False
    if "year_to" in filters and year > filters["year_to"]:
        return False
    if filters.get("country") and movie.get("country") != filters["country"]:
        return False
    if "min_rating" in filters and float(movie.get("vote_average", 0.0)) < filters["min_rating"]:
        return False
    if filters.get("genre") and filters["genre"] not in (movie.get("genres") or []):
        return False
    return True


def answer_aggregate_question(
    question: str, movies: list[dict], top_k: int
) -> tuple[str, list[dict]]:
    """전체 영화 데이터에서 count/min/max/order를 결정적으로 계산한다."""
    filters = extract_filters(question)
    candidates = [movie for movie in movies if _matches_filters(movie, filters)]
    if not candidates:
        return "조건에 맞는 영화가 현재 데이터에 없습니다.", []

    if re.search(r"(몇\s*편|몇\s*개|개수|총\s*몇)", question):
        return f"현재 저장된 데이터에서 조건에 맞는 영화는 총 {len(candidates)}편입니다.", []

    if "평점" in question or "별점" in question:
        rated = [movie for movie in candidates if float(movie.get("vote_average", 0.0)) > 0]
        if not rated:
            return "조건에 맞는 영화의 평점 정보가 없습니다.", []
        ascending = any(word in question for word in ("낮", "최저"))
        ranked = sorted(
            rated,
            key=lambda movie: float(movie.get("vote_average", 0.0)),
            reverse=not ascending,
        )
        if "순" in question:
            selected = ranked[:top_k]
        else:
            best_score = float(ranked[0].get("vote_average", 0.0))
            selected = [
                movie
                for movie in ranked
                if float(movie.get("vote_average", 0.0)) == best_score
            ][:top_k]
        direction = "낮은" if ascending else "높은"
        titles = ", ".join(
            f"{movie['title']}({movie.get('year', 0)}, {movie.get('vote_average', 0.0)}점)"
            for movie in selected
        )
        return f"현재 저장된 TMDB 평점 기준으로 가장 {direction} 영화는 {titles}입니다.", selected

    dated = [movie for movie in candidates if int(movie.get("year", 0) or 0) > 0]
    if not dated:
        return "조건에 맞는 영화의 개봉연도 정보가 없습니다.", []

    newest = any(word in question for word in ("최신", "최근"))
    ranked = sorted(dated, key=lambda movie: int(movie.get("year", 0)), reverse=newest)
    target_year = int(ranked[0].get("year", 0))
    selected = [movie for movie in ranked if int(movie.get("year", 0)) == target_year][:top_k]
    qualifier = "최신" if newest else "가장 오래된"
    titles = ", ".join(f"{movie['title']}({movie.get('year', 0)})" for movie in selected)
    return f"현재 저장된 데이터에서 {qualifier} 영화는 {titles}입니다.", selected


_CONTEXTUALIZE_PROMPT = (
    "당신은 영화 검색 질의 재작성기입니다. 이전 대화를 참고해 현재 질문을 대화 없이도 "
    "이해되는 독립적인 한국어 검색 문장 한 줄로 바꾸세요. 지시대명사는 실제 영화명·인명으로 "
    "해소하세요. '다른 영화'나 '또 다른 작품'을 묻는 경우 기준이 된 이전 영화명도 "
    "'인셉션을 제외한'처럼 명시하세요. 답변하거나 대화에 없던 조건을 추가하지 마세요."
)

_GRADE_PROMPT = (
    "각 영화 문서를 서로 독립적으로 평가하세요. 질문에 직접 답할 사실이 문서에 있을 때만 "
    "relevant=yes입니다. 주제가 비슷하거나 영화라는 이유만으로 yes를 주면 안 됩니다. "
    "모든 문서 번호를 정확히 한 번씩 판정하세요."
)

_GRADE_LINE_PROMPT = (
    _GRADE_PROMPT
    + " 각 줄을 정확히 '문서 0: yes' 또는 '문서 0: no' 형식으로만 출력하세요. "
    "설명, 머리말, 코드 블록은 금지합니다."
)

_SYSTEM_PROMPT = (
    "당신은 영화 정보를 안내하는 한국어 도우미입니다.\n"
    "아래 번호가 붙은 영화 정보만 사실 근거로 사용해 간결하고 정확하게 답하세요.\n"
    "정보가 부족하면 추측하지 말고 '주어진 정보에서는 확인할 수 없습니다.'라고 답하세요.\n"
    "답변에 영화 제목과 개봉연도를 함께 언급하세요.\n"
    "source_ids에는 답변의 사실을 작성할 때 실제 사용한 문서 번호만 넣으세요.\n\n"
    "영화 정보:\n{context}"
)

_SOURCE_LINE_INSTRUCTION = (
    "\n응답의 마지막 줄에는 반드시 `SOURCE_IDS: 0,1`처럼 실제 사용한 문서 번호만 쓰세요. "
    "근거가 없으면 `SOURCE_IDS: NONE`을 쓰고, 이 줄 뒤에는 아무것도 쓰지 마세요."
)

_CONVERSATION_PROMPT = (
    "당신은 영화 챗봇의 대화 도우미입니다. 사용자의 현재 질문이 이전 대화 자체에 관한 "
    "질문이므로 아래 대화 이력만 사용해 한국어로 간결하게 답하세요. 검색하지 않은 새로운 "
    "영화 사실은 만들지 마세요. 이력에서 확인되지 않으면 확인할 수 없다고 답하세요."
)


class MovieRagGraph:
    """StateGraph를 1회 컴파일해 재사용하는 영화 RAG 파이프라인."""

    def __init__(self) -> None:
        self.vectorstore = get_vectorstore()
        self.movies = load_movies(settings.movies_file)
        self.llm = get_llm()
        if settings.graph_native_structured_output:
            self.document_grader = self._structured_model(DocumentGrades)
            self.grounded_generator = self._structured_model(GroundedAnswer)
        else:
            self.document_grader = None
            self.grounded_generator = None
        self.app = self._build()

    def _structured_model(self, schema: type[BaseModel]):
        try:
            if settings.llm_provider == "anthropic":
                # Anthropic의 네이티브 JSON Schema 경로를 명시한다. 기본값인
                # function_calling보다 스키마 준수 여부가 명확하다.
                return self.llm.with_structured_output(schema, method="json_schema")
            return self.llm.with_structured_output(schema)
        except (NotImplementedError, ValueError):
            logger.warning("현재 LLM이 구조화 출력을 지원하지 않습니다: %s", schema.__name__)
            return None

    # --- 라우팅/질의 준비 ----------------------------------------------------

    def _route_question(self, state: RagState) -> dict:
        question = state.get("original_question") or state["question"]
        return {
            "original_question": question,
            "search_query": question,
            "route": classify_question(question),
            "messages": [HumanMessage(content=question)],
            "filters": {},
            "documents": [],
            "document_scores": [],
            "relevant": False,
            "retries": 0,
            "sources": [],
        }

    def _route_decision(self, state: RagState) -> QuestionRoute:
        return state.get("route", "fact")

    def _contextualize_query(self, state: RagState) -> dict:
        """이전 대화로 현재 질문의 지시대명사를 해소해 검색 전용 문장을 만든다."""
        question = state["original_question"]
        messages = list(state.get("messages", []))
        history = messages[:-1]
        if not history:
            return {"search_query": question}

        prompt = [SystemMessage(_CONTEXTUALIZE_PROMPT)]
        prompt.extend(history[-8:])
        prompt.append(HumanMessage(content=f"현재 질문: {question}"))
        try:
            rewritten = self.llm.invoke(prompt).content.strip().splitlines()[0].strip('"\' ')
        except Exception:  # 맥락화 실패 시 사용자 원문으로 안전하게 검색
            logger.exception("대화 기반 검색 질의 맥락화에 실패했습니다.")
            rewritten = question
        if "다른" in question:
            history_text = "\n".join(str(message.content) for message in history)
            mentioned_titles = [
                movie.get("title", "")
                for movie in getattr(self, "movies", [])
                if movie.get("title") and movie["title"] in history_text
            ]
            if mentioned_titles and "제외" not in rewritten:
                rewritten = f"{rewritten} ({mentioned_titles[-1]} 제외)"
        return {"search_query": rewritten or question}

    def _analyze_filters(self, state: RagState) -> dict:
        return {"filters": extract_filters(state["search_query"])}

    # --- 검색/평가/재작성 ----------------------------------------------------

    def _retrieve(self, state: RagState) -> dict:
        filters = state.get("filters", {})
        search_kwargs: dict = {"k": max(settings.retrieval_fetch_k, settings.top_k)}
        where = _metadata_filter(filters)
        if where:
            search_kwargs["filter"] = where
        if filters.get("genre"):
            search_kwargs["where_document"] = {"$contains": filters["genre"]}

        results = self.vectorstore.similarity_search_with_score(
            state["search_query"], **search_kwargs
        )
        accepted = [
            (doc, float(distance))
            for doc, distance in results
            if float(distance) <= settings.retrieval_max_distance
            and not _is_explicitly_excluded(state["search_query"], doc)
        ][: settings.top_k]
        return {
            "documents": [doc for doc, _ in accepted],
            "document_scores": [distance for _, distance in accepted],
        }

    def _grade_documents(self, state: RagState) -> dict:
        """거리 필터를 통과한 각 문서를 구조화 출력으로 독립 판정한다."""
        docs = state.get("documents", [])
        scores = state.get("document_scores", [])
        if not docs:
            return {"documents": [], "document_scores": [], "relevant": False}
        exact_ids = {
            index
            for index, doc in enumerate(docs)
            if _has_exact_metadata_match(state["search_query"], doc)
        }
        if exact_ids:
            kept_docs = [doc for index, doc in enumerate(docs) if index in exact_ids]
            kept_scores = [score for index, score in enumerate(scores) if index in exact_ids]
            return {
                "documents": kept_docs,
                "document_scores": kept_scores,
                "relevant": True,
            }
        if not settings.graph_grade_with_llm:
            return {"relevant": True}

        human_prompt = HumanMessage(
            "검색 질문: " + state["search_query"] + "\n"
            "판정할 문서 번호: " + ", ".join(str(index) for index in range(len(docs))) + "\n\n"
            + _format_docs(docs, numbered=True)
        )
        relevant_ids: set[int] | None = None
        if self.document_grader is not None:
            try:
                result = self.document_grader.invoke([SystemMessage(_GRADE_PROMPT), human_prompt])
                raw_grades = (
                    result.grades if isinstance(result, DocumentGrades) else result["grades"]
                )
                relevant_ids = set()
                seen_ids: set[int] = set()
                for grade in raw_grades:
                    document_id = (
                        grade.document_id
                        if isinstance(grade, DocumentGrade)
                        else grade["document_id"]
                    )
                    verdict = (
                        grade.relevant if isinstance(grade, DocumentGrade) else grade["relevant"]
                    )
                    if not 0 <= document_id < len(docs) or document_id in seen_ids:
                        raise ValueError("잘못되거나 중복된 문서 번호")
                    seen_ids.add(document_id)
                    if verdict == "yes":
                        relevant_ids.add(document_id)
                if seen_ids != set(range(len(docs))):
                    raise ValueError("모든 문서 판정이 필요합니다.")
            except Exception:
                logger.warning("네이티브 구조화 문서 판정 실패; strict text로 전환합니다.")
                self.document_grader = None

        if relevant_ids is None:
            try:
                reply = self.llm.invoke([SystemMessage(_GRADE_LINE_PROMPT), human_prompt]).content
                relevant_ids = parse_document_grades(reply, len(docs))
            except Exception:
                # 관련성을 검증할 수 없을 때 무관 문서를 통과시키지 않는다.
                logger.exception("검색 문서 strict 관련성 판정에 실패했습니다.")
                relevant_ids = set()

        kept_docs = [doc for index, doc in enumerate(docs) if index in relevant_ids]
        kept_scores = [score for index, score in enumerate(scores) if index in relevant_ids]
        return {
            "documents": kept_docs,
            "document_scores": kept_scores,
            "relevant": bool(kept_docs),
        }

    def _rewrite_query(self, state: RagState) -> dict:
        """검색 전용 질문만 재작성한다. 다음 노드에서 필터도 다시 계산한다."""
        prompt = [
            SystemMessage(
                "검색 결과가 없었습니다. 사용자의 의도를 유지하면서 영화 제목·인명·핵심 조건이 "
                "잘 드러나는 한국어 벡터 검색 문장 한 줄로 다시 쓰세요. 답변하지 마세요."
            ),
            HumanMessage(
                f"사용자 원문: {state['original_question']}\n"
                f"현재 검색 문장: {state['search_query']}"
            ),
        ]
        try:
            new_query = self.llm.invoke(prompt).content.strip().splitlines()[0].strip('"\' ')
        except Exception:
            logger.exception("검색 질의 재작성에 실패했습니다.")
            new_query = state["search_query"]
        return {
            "search_query": new_query or state["search_query"],
            "retries": state.get("retries", 0) + 1,
        }

    def _decide_after_grade(self, state: RagState) -> Literal["generate", "rewrite"]:
        if state.get("relevant"):
            return "generate"
        if state.get("retries", 0) < settings.graph_max_retries:
            return "rewrite"
        return "generate"

    # --- 응답 생성 -----------------------------------------------------------

    @staticmethod
    def _refusal() -> dict:
        answer = "주어진 정보에서는 확인할 수 없습니다."
        return {"answer": answer, "sources": [], "messages": [AIMessage(content=answer)]}

    def _generate_rag(self, state: RagState) -> dict:
        docs = state.get("documents", [])
        if not docs:
            return self._refusal()

        base_system = (
            f"검색을 위해 독립적으로 해석한 현재 질문: {state['search_query']}\n\n"
            + _SYSTEM_PROMPT.format(context=_format_docs(docs, numbered=True))
        )
        history = list(state.get("messages", []))
        parsed_answer: GroundedAnswer | None = None

        if self.grounded_generator is not None:
            try:
                result = self.grounded_generator.invoke([SystemMessage(base_system)] + history)
                parsed_answer = (
                    result
                    if isinstance(result, GroundedAnswer)
                    else GroundedAnswer(**result)
                )
            except Exception:
                logger.warning("네이티브 구조화 답변 실패; strict text로 전환합니다.")
                self.grounded_generator = None

        if parsed_answer is None:
            attempts = [
                history,
                [HumanMessage(content=state["original_question"])],
            ]
            for attempt_number, attempt_messages in enumerate(attempts, start=1):
                raw_answer = self.llm.invoke(
                    [SystemMessage(base_system + _SOURCE_LINE_INSTRUCTION)] + attempt_messages
                ).content
                try:
                    parsed_answer = parse_grounded_answer(raw_answer, len(docs))
                except ValueError:
                    try:
                        parsed_answer = infer_grounded_answer_from_titles(raw_answer, docs)
                    except ValueError:
                        logger.warning(
                            "근거 답변 출처 검증 실패(%s/%s)", attempt_number, len(attempts)
                        )
                        continue
                break
            if parsed_answer is None:
                return self._refusal()

        answer, source_ids = parsed_answer.answer, parsed_answer.source_ids
        if "확인할 수 없습니다" in answer:
            source_ids = []

        valid_source_ids = sorted({index for index in source_ids if 0 <= index < len(docs)})
        used_docs = [docs[index] for index in valid_source_ids]
        return {
            "answer": answer,
            "sources": [_to_source(doc) for doc in used_docs],
            "messages": [AIMessage(content=answer)],
        }

    def _generate_conversation(self, state: RagState) -> dict:
        messages = [SystemMessage(_CONVERSATION_PROMPT)] + list(state.get("messages", []))
        answer = self.llm.invoke(messages).content
        return {"answer": answer, "sources": [], "messages": [AIMessage(content=answer)]}

    def _answer_aggregate(self, state: RagState) -> dict:
        answer, selected = answer_aggregate_question(
            state["original_question"], self.movies, settings.top_k
        )
        return {
            "answer": answer,
            "sources": [_movie_to_source(movie) for movie in selected],
            "messages": [AIMessage(content=answer)],
        }

    # --- 그래프 구성/체크포인터 ---------------------------------------------

    def _checkpointer(self):
        if settings.checkpointer == "sqlite":
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver

            conn = sqlite3.connect(str(settings.checkpoint_db), check_same_thread=False)
            return SqliteSaver(conn)

        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    def _build(self):
        graph = StateGraph(RagState)
        graph.add_node("route_question", self._route_question)
        graph.add_node("contextualize_query", self._contextualize_query)
        graph.add_node("analyze_filters", self._analyze_filters)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade_documents", self._grade_documents)
        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("generate_rag", self._generate_rag)
        graph.add_node("generate_conversation", self._generate_conversation)
        graph.add_node("answer_aggregate", self._answer_aggregate)

        graph.add_edge(START, "route_question")
        graph.add_conditional_edges(
            "route_question",
            self._route_decision,
            {
                "fact": "contextualize_query",
                "conversation": "generate_conversation",
                "aggregate": "answer_aggregate",
            },
        )
        graph.add_edge("contextualize_query", "analyze_filters")
        graph.add_edge("analyze_filters", "retrieve")
        graph.add_edge("retrieve", "grade_documents")
        graph.add_conditional_edges(
            "grade_documents",
            self._decide_after_grade,
            {"generate": "generate_rag", "rewrite": "rewrite_query"},
        )
        # 재작성으로 새 조건이 생길 수 있으므로 retrieve가 아니라 필터 분석으로 돌아간다.
        graph.add_edge("rewrite_query", "analyze_filters")
        graph.add_edge("generate_rag", END)
        graph.add_edge("generate_conversation", END)
        graph.add_edge("answer_aggregate", END)

        return graph.compile(checkpointer=self._checkpointer())

    # --- 외부 진입점 ---------------------------------------------------------

    def answer(self, question: str, session_id: str | None = None) -> dict:
        """질문에 답한다. 같은 session_id는 이전 대화 상태를 이어받는다."""
        thread_id = session_id or f"once-{uuid.uuid4()}"
        state_in: RagState = {
            "question": question,
            "original_question": question,
            "search_query": question,
            "filters": {},
            "documents": [],
            "document_scores": [],
            "relevant": False,
            "retries": 0,
            "sources": [],
        }
        result = self.app.invoke(
            state_in, config={"configurable": {"thread_id": thread_id}}
        )
        return {"answer": result.get("answer", ""), "sources": result.get("sources", [])}
