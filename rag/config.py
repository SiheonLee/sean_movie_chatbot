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


@dataclass
class Settings:
    # --- 경로 ---
    # TMDB 수집 결과(JSON)가 저장될 위치. fetch_tmdb 가 만들고 indexing 이 읽는다.
    movies_file: Path = field(default_factory=lambda: ROOT_DIR / _get("MOVIES_FILE", "data/movies.json"))
    # 영속 벡터스토어 저장 위치. 한 번 색인하면 재기동 시 재인덱싱하지 않는다.
    chroma_dir: Path = field(default_factory=lambda: ROOT_DIR / _get("CHROMA_DIR", "chroma_db"))
    collection_name: str = field(default_factory=lambda: _get("CHROMA_COLLECTION", "movies"))

    # --- 청킹(줄거리가 매우 길 때만 추가 분할) ---
    chunk_size: int = field(default_factory=lambda: int(_get("CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(_get("CHUNK_OVERLAP", "100")))

    # --- 검색 ---
    top_k: int = field(default_factory=lambda: int(_get("TOP_K", "4")))
    # Chroma에서 먼저 가져올 후보 수. 거리/LLM 관련성 필터 후 top_k개만 남긴다.
    retrieval_fetch_k: int = field(default_factory=lambda: int(_get("RETRIEVAL_FETCH_K", "8")))
    # Chroma L2 거리 상한(낮을수록 유사). BGE-M3 현재 색인 기준 기본값이며 평가로 보정한다.
    retrieval_max_distance: float = field(
        default_factory=lambda: float(_get("RETRIEVAL_MAX_DISTANCE", "1.0"))
    )

    # --- TMDB 수집 ---
    tmdb_api_key: str | None = field(default_factory=lambda: os.getenv("TMDB_API_KEY"))
    tmdb_language: str = field(default_factory=lambda: _get("TMDB_LANGUAGE", "ko-KR"))
    # 한국 인기작 / 해외 인기작·고평점작에서 각각 몇 페이지(페이지당 20편)를 모을지.
    tmdb_kr_pages: int = field(default_factory=lambda: int(_get("TMDB_KR_PAGES", "8")))
    tmdb_intl_pages: int = field(default_factory=lambda: int(_get("TMDB_INTL_PAGES", "7")))

    # --- 임베딩 제공자 ---
    # huggingface(기본) | google
    embedding_provider: str = field(default_factory=lambda: _get("EMBEDDING_PROVIDER", "huggingface").lower())
    hf_embedding_model: str = field(default_factory=lambda: _get("HF_EMBEDDING_MODEL", "BAAI/bge-m3"))
    google_embedding_model: str = field(default_factory=lambda: _get("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001"))

    # --- LLM 제공자 ---
    # anthropic(기본, Claude Haiku) | ollama | google | hf_endpoint | hf_pipeline
    llm_provider: str = field(default_factory=lambda: _get("LLM_PROVIDER", "anthropic").lower())
    anthropic_model: str = field(
        default_factory=lambda: _get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    )
    hf_llm_model: str = field(default_factory=lambda: _get("HF_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    ollama_model: str = field(default_factory=lambda: _get("OLLAMA_MODEL", "gemma4:e2b-mlx"))
    ollama_base_url: str = field(default_factory=lambda: _get("OLLAMA_BASE_URL", "http://localhost:11434"))
    google_llm_model: str = field(default_factory=lambda: _get("GOOGLE_MODEL", "gemini-2.5-flash"))
    llm_temperature: float = field(default_factory=lambda: float(_get("LLM_TEMPERATURE", "0.1")))
    llm_max_tokens: int = field(default_factory=lambda: int(_get("LLM_MAX_TOKENS", "512")))

    # --- 키 ---
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    google_api_key: str | None = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    hf_api_token: str | None = field(default_factory=lambda: os.getenv("HUGGINGFACEHUB_API_TOKEN"))

    # --- 평가(LangSmith) ---
    dataset_name: str = field(default_factory=lambda: _get("LANGSMITH_DATASET", "movies-rag-eval"))
    eval_file: Path = field(default_factory=lambda: ROOT_DIR / _get("EVAL_FILE", "eval/dataset.jsonl"))

    # --- LangGraph ---
    # 검색결과가 부적합할 때 질의를 재작성해 재검색하는 최대 횟수(무한 루프 방지).
    graph_max_retries: int = field(default_factory=lambda: int(_get("GRAPH_MAX_RETRIES", "2")))
    # 검색결과 관련성 평가에 LLM을 쓸지(끄면 "문서 존재 여부"만으로 판단해 호출 절약).
    graph_grade_with_llm: bool = field(default_factory=lambda: _get("GRAPH_GRADE_WITH_LLM", "true").lower() == "true")
    # Claude Haiku 4.5는 JSON Schema 출력을 지원한다. 미지원 모델은 .env에서 false로 끈다.
    graph_native_structured_output: bool = field(
        default_factory=lambda: _get("GRAPH_NATIVE_STRUCTURED_OUTPUT", "true").lower() == "true"
    )
    # 멀티턴 대화 메모리 저장 방식: memory(휘발) | sqlite(파일 영속).
    checkpointer: str = field(default_factory=lambda: _get("CHECKPOINTER", "memory").lower())
    checkpoint_db: Path = field(default_factory=lambda: ROOT_DIR / _get("CHECKPOINT_DB", "graph_checkpoints.sqlite"))


settings = Settings()
