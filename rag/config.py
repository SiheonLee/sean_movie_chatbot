"""
설정 모듈.

모든 설정은 환경 변수(.env)로 주입한다. 코드 곳곳에 os.getenv를 흩어두지 않고
이 한 곳에서 Settings 객체로 모아 관리한다(설정 분리). 다른 모듈은
`from rag.config import settings` 로 가져다 쓴다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# .env 파일을 읽어 환경 변수로 등록한다.
load_dotenv()

# 프로젝트 루트 (이 파일 기준 한 단계 위)
ROOT_DIR = Path(__file__).resolve().parent.parent


def _get(name: str, default: str) -> str:
    return os.getenv(name, default)


def _optional_float(name: str, default: str) -> float | None:
    """빈 문자열이면 None. '설정 안 함'과 '0으로 설정'을 구분해야 할 때 쓴다."""
    raw = os.getenv(name, default).strip()
    return float(raw) if raw else None


@dataclass
class Settings:
    # --- 경로 ---
    # TMDB 수집 결과(JSON)가 저장될 위치. fetch_tmdb 가 만들고 indexing 이 읽는다.
    movies_file: Path = field(default_factory=lambda: ROOT_DIR / _get("MOVIES_FILE", "data/movies.json"))
    # 무드 프로파일(LLM 결과) 캐시. 임베딩이나 텍스트 조립을 손볼 때 LLM을 다시
    # 부르지 않도록 배치 결과를 여기서 한 번 끊는다.
    enriched_file: Path = field(default_factory=lambda: ROOT_DIR / _get("ENRICHED_FILE", "data/enriched.json"))
    # 영속 벡터스토어 저장 위치. 한 번 색인하면 재기동 시 재인덱싱하지 않는다.
    chroma_dir: Path = field(default_factory=lambda: ROOT_DIR / _get("CHROMA_DIR", "chroma_db"))
    collection_name: str = field(default_factory=lambda: _get("CHROMA_COLLECTION", "movies"))

    # 청킹 설정(CHUNK_SIZE/CHUNK_OVERLAP)은 제거했다. 무드 문서는 500편 규모에
    # 영화당 한 문서라 짧아서 분할하지 않는다.

    # --- 검색 ---
    # 도구가 기본으로 돌려줄 영화 편수. 거리 상한(retrieval_max_distance)은 제거했다.
    # 임베딩을 바꿀 때마다 재보정해야 하는 매직 넘버였고, 후보가 부족하면 도구가
    # "조건을 완화하세요"를 반환해 LLM이 재호출하는 편이 낫다.
    top_k: int = field(default_factory=lambda: int(_get("TOP_K", "4")))

    # --- TMDB 수집 ---
    tmdb_api_key: str | None = field(default_factory=lambda: os.getenv("TMDB_API_KEY"))
    tmdb_language: str = field(default_factory=lambda: _get("TMDB_LANGUAGE", "ko-KR"))
    # 한국 인기작 / 해외 인기작·고평점작에서 각각 몇 페이지(페이지당 20편)를 모을지.
    tmdb_kr_pages: int = field(default_factory=lambda: int(_get("TMDB_KR_PAGES", "8")))
    tmdb_intl_pages: int = field(default_factory=lambda: int(_get("TMDB_INTL_PAGES", "7")))
    # 잔잔한 계열 수집축. popularity/top_rated만 쓰면 액션·스릴러가 67%를 차지해
    # "비 오는 날 볼 잔잔한 영화" 같은 무드 질의에 후보가 몇 편 남지 않는다.
    tmdb_quiet_pages: int = field(default_factory=lambda: int(_get("TMDB_QUIET_PAGES", "3")))
    # 최소 투표 수. TMDB의 popularity는 페이지뷰·API 트래픽 기반 지표라 실제 인기와
    # 무관하게 조회만 많은 작품(특히 한국 성인물)이 상위에 올라온다. discover 쿼리에
    # vote_count.gte로 걸어 애초에 수집하지 않고, 목록 엔드포인트에는 사후 필터로 쓴다.
    tmdb_min_vote_count: int = field(
        default_factory=lambda: int(_get("TMDB_MIN_VOTE_COUNT", "100"))
    )

    # --- 임베딩 제공자 ---
    # 색인과 검색은 같은 모델·차원을 사용해야 한다.
    embedding_provider: str = field(default_factory=lambda: _get("EMBEDDING_PROVIDER", "google").lower())
    google_embedding_model: str = field(
        default_factory=lambda: _get("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")
    )
    google_embedding_dimensions: int = field(
        default_factory=lambda: int(_get("GOOGLE_EMBEDDING_DIMENSIONS", "3072"))
    )
    # 한 요청에 담을 문서 수. embed_documents는 건수와 무관하게 요청 1건으로
    # 나가므로 이 값이 페이로드 크기를 결정한다. 실측: 50건 성공 / 90건 429.
    google_embedding_batch_size: int = field(
        default_factory=lambda: int(_get("GOOGLE_EMBEDDING_BATCH_SIZE", "50"))
    )
    # 분당 요청 수. 무료 티어 한도(100 RPM)에 여유를 둔다.
    google_embedding_rpm: int = field(
        default_factory=lambda: int(_get("GOOGLE_EMBEDDING_RPM", "90"))
    )
    google_embedding_max_retries: int = field(
        default_factory=lambda: int(_get("GOOGLE_EMBEDDING_MAX_RETRIES", "5"))
    )

    # --- LLM 제공자 ---
    # 도구 호출(bind_tools)이 전제라 openai(기본) | anthropic | google 만 지원한다.
    llm_provider: str = field(default_factory=lambda: _get("LLM_PROVIDER", "openai").lower())
    openai_model: str = field(default_factory=lambda: _get("OPENAI_MODEL", "gpt-5.6-luna"))
    anthropic_model: str = field(
        default_factory=lambda: _get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    )
    google_llm_model: str = field(default_factory=lambda: _get("GOOGLE_MODEL", "gemini-2.5-flash"))
    # 0.1에서도 골든셋이 실행할 때마다 다른 케이스에서 실패했다(라우팅 변동성).
    # 도구 선택은 창의성이 필요한 작업이 아니므로 0으로 내린다.
    #
    # None이 될 수 있다. GPT-5 계열 추론 모델은 temperature를 아예 거부하기도 해서
    # (400) 파라미터 자체를 빼야 할 때가 있다. LLM_TEMPERATURE를 빈 값으로 두면
    # None이 되고, 제공자에 넘기지 않는다.
    llm_temperature: float | None = field(default_factory=lambda: _optional_float("LLM_TEMPERATURE", "0"))
    # 512는 부족하다. 영화 5편을 소개하면 stop_reason=max_tokens로 문장 중간에서
    # 잘린다(실측). 웹 검색 기반 평단 반응 요약은 더 길다.
    #
    # 추론 모델에서는 이 값이 "추론 + 답변" 합계라서 더 크게 잡아야 한다. 모자라면
    # 추론에 다 쓰고 답변이 빈 문자열로 온다.
    llm_max_tokens: int = field(default_factory=lambda: int(_get("LLM_MAX_TOKENS", "2048")))
    # 추론 강도. 비워 두면 파라미터를 안 보내고 API 기본값을 쓴다.
    #
    # "none"에는 부수 효과가 있다. langchain-openai는 gpt-5 계열(chat 제외)에서
    # reasoning_effort가 "none"이 아니면 temperature를 payload에서 빼버린다.
    # 즉 **"none"으로 둬야 LLM_TEMPERATURE=0이 실제로 전달된다**(§4-17의 라우팅
    # 변동성 완화책). 대신 추론을 끄므로 함정 질의의 라우팅 정확도가 달라질 수
    # 있다 — 골든셋으로 양쪽을 재보고 정하는 게 맞다.
    openai_reasoning_effort: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_REASONING_EFFORT", "").strip() or None
    )

    # --- 웹 검색 ---
    # 평단·수상·화제·비하인드·해석만 담당한다. 결과 수를 늘려도 정확도가 오르지
    # 않고 컨텍스트만 커진다.
    web_search_max_results: int = field(
        default_factory=lambda: int(_get("WEB_SEARCH_MAX_RESULTS", "3"))
    )

    # --- 키 ---
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    google_api_key: str | None = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    tavily_api_key: str | None = field(default_factory=lambda: os.getenv("TAVILY_API_KEY"))

    # --- 평가(LangSmith) ---
    dataset_name: str = field(default_factory=lambda: _get("LANGSMITH_DATASET", "movies-rag-eval"))
    eval_file: Path = field(default_factory=lambda: ROOT_DIR / _get("EVAL_FILE", "eval/dataset.jsonl"))

    # --- LangGraph ---
    # 멀티턴 대화 메모리 저장 방식: memory(휘발) | sqlite(파일 영속).
    checkpointer: str = field(default_factory=lambda: _get("CHECKPOINTER", "memory").lower())
    checkpoint_db: Path = field(default_factory=lambda: ROOT_DIR / _get("CHECKPOINT_DB", "graph_checkpoints.sqlite"))

    # --- 초대형 데모 보호 장치 ---
    # 익명 브라우저 ID별 하루 질문 수와 대화별 질문 수는 SQLite에 보존한다.
    # 동시 처리 수는 단일 API 프로세스 안에서 외부 LLM 호출이 겹치는 정도를 막는다.
    rate_limit_db: Path = field(
        default_factory=lambda: ROOT_DIR / _get("RATE_LIMIT_DB", "rate_limits.sqlite")
    )
    daily_question_limit: int = field(
        default_factory=lambda: int(_get("DAILY_QUESTION_LIMIT", "30"))
    )
    session_question_limit: int = field(
        default_factory=lambda: int(_get("SESSION_QUESTION_LIMIT", "12"))
    )
    max_concurrent_requests: int = field(
        default_factory=lambda: int(_get("MAX_CONCURRENT_REQUESTS", "2"))
    )


settings = Settings()
