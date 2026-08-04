"""
벡터스토어 접근과 색인 호환성 검증.

이 모듈은 **런타임이 import 한다**(search_by_vibe가 벡터스토어를 얻는다). 색인을
*만드는* 쪽은 scripts/build_store.py이고, 여기서는 만들지 않는다. 그래서 이름이
indexing이 아니라 store다.

핵심 설계 — 색인 지문(fingerprint):
    원본 데이터와 벡터 공간을 결정하는 설정을 함께 해시해 chroma_dir/.source_hash에
    저장한다. 기동 시 이 값을 대조해, 데이터·어휘·문서 스키마·임베딩 설정이 모두
    같을 때만 기존 색인을 신뢰한다. 하나라도 다르면 재색인을 요구한다.

    해시 계산은 양쪽이 쓴다. scripts/build_store.py가 색인 후 기록하고, 여기의
    get_vectorstore()가 기동 시 검증한다. scripts/ → rag/ 방향이므로 배치가 이
    모듈을 import 하는 것은 허용된다(그 반대는 금지).
"""

from __future__ import annotations

import hashlib
import json

from langchain_chroma import Chroma

from rag.config import settings
from rag.providers import get_embeddings
from rag.vocab import VOCAB_VERSION

# 임베딩 텍스트 조립 방식이나 메타데이터 구성을 바꾸면 올린다.
DOCUMENT_SCHEMA_VERSION = 2

_HASH_FILENAME = ".source_hash"
_CHROMA_DB_FILENAME = "chroma.sqlite3"

_REINDEX_HINT = "`python -m scripts.build_store` 로 색인하세요."


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_index_hash(movies_raw: str, enriched_raw: str) -> str:
    """원본과 벡터 공간을 결정하는 모든 입력을 묶어 색인 식별자를 만든다."""
    fingerprint = {
        "movies_sha256": _hash(movies_raw),
        "enriched_sha256": _hash(enriched_raw),
        "vocab_version": VOCAB_VERSION,
        "document_schema_version": DOCUMENT_SCHEMA_VERSION,
        "embedding": {
            "provider": settings.embedding_provider,
            "model": settings.google_embedding_model,
            "dimensions": settings.google_embedding_dimensions,
        },
    }
    return _hash(json.dumps(fingerprint, ensure_ascii=False, sort_keys=True))


def hash_path():
    return settings.chroma_dir / _HASH_FILENAME


def _read_sources() -> tuple[str, str]:
    """movies.json과 enriched.json 원문을 읽는다. 없으면 안내와 함께 실패한다."""
    if not settings.movies_file.exists():
        raise FileNotFoundError(
            f"영화 데이터가 없습니다: {settings.movies_file}\n"
            f"먼저 `python -m scripts.fetch_tmdb` 로 수집하세요."
        )
    if not settings.enriched_file.exists():
        raise FileNotFoundError(
            f"무드 프로파일이 없습니다: {settings.enriched_file}\n"
            f"먼저 `python -m scripts.enrich` 로 생성하세요."
        )
    return (
        settings.movies_file.read_text(encoding="utf-8"),
        settings.enriched_file.read_text(encoding="utf-8"),
    )


def current_index_hash() -> str:
    """현재 데이터·설정 기준의 색인 식별자."""
    return compute_index_hash(*_read_sources())


def index_is_current() -> bool:
    database_path = settings.chroma_dir / _CHROMA_DB_FILENAME
    path = hash_path()
    if not (database_path.exists() and path.exists()):
        return False
    try:
        return path.read_text(encoding="utf-8").strip() == current_index_hash()
    except FileNotFoundError:
        # 원본이 사라졌으면 색인을 신뢰할 수 없다.
        return False


def get_vectorstore() -> Chroma:
    """이미 색인된 영속 Chroma 컬렉션을 로드한다(재색인 X)."""
    database_path = settings.chroma_dir / _CHROMA_DB_FILENAME
    if not database_path.exists():
        raise FileNotFoundError(
            f"벡터스토어가 없습니다: {settings.chroma_dir}\n먼저 {_REINDEX_HINT}"
        )
    if not index_is_current():
        raise RuntimeError(
            "벡터스토어가 현재 데이터·어휘·임베딩 설정과 호환되지 않습니다.\n"
            f"다시 {_REINDEX_HINT}"
        )
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )
