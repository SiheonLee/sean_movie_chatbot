"""
인덱싱 모듈.

data/movies.json → 영화 1편을 검색용 텍스트 문서로 조립 → 영속 Chroma 저장.

핵심 설계:
- 영화는 필드가 짧고 구조적이라 '영화 1편 = 검색 문서 1개'로 다룬다.
  제목/장르/감독/출연/국가/평점/줄거리를 한 텍스트로 합쳐 임베딩한다.
- year/genres/country/vote_average 를 메타데이터로 저장해 필터 검색을 지원한다.
  Chroma는 리스트 메타데이터를 못 받으므로 genres 는 "|드라마|스릴러|" 형태
  문자열로 저장해 부분일치($contains) 필터에 쓴다.
- movies.json 내용 해시를 chroma_db/.source_hash 에 저장한다. 내용이 그대로면
  재색인을 건너뛰어 코퍼스 임베딩을 데이터가 바뀐 경우에만 1회 수행한다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import settings
from rag.providers import get_embeddings

# ISO 3166-1 국가 코드 → 한국어 표기(자주 등장하는 것만; 없으면 코드 그대로).
_COUNTRY_KO = {
    "KR": "한국", "US": "미국", "JP": "일본", "GB": "영국", "FR": "프랑스",
    "CN": "중국", "HK": "홍콩", "DE": "독일", "IT": "이탈리아", "ES": "스페인",
    "IN": "인도", "CA": "캐나다", "AU": "호주", "TW": "대만", "TH": "태국",
}


def _source_hash_path() -> Path:
    return settings.chroma_dir / ".source_hash"


def _compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def country_ko(code: str) -> str:
    return _COUNTRY_KO.get(code, code)


def _movie_text(m: dict) -> str:
    """영화 dict를 임베딩할 검색 문서 텍스트로 조립한다."""
    lines = [
        f"제목: {m['title']} ({m['year']})" if m.get("year") else f"제목: {m['title']}",
    ]
    if m.get("original_title") and m["original_title"] != m["title"]:
        lines.append(f"원제: {m['original_title']}")
    if m.get("genres"):
        lines.append(f"장르: {', '.join(m['genres'])}")
    if m.get("director"):
        lines.append(f"감독: {m['director']}")
    if m.get("cast"):
        lines.append(f"출연: {', '.join(m['cast'])}")
    meta_line = []
    if m.get("country"):
        meta_line.append(f"국가: {country_ko(m['country'])}")
    if m.get("vote_average"):
        meta_line.append(f"평점: {m['vote_average']}")
    if meta_line:
        lines.append(" | ".join(meta_line))
    if m.get("overview"):
        lines.append(f"줄거리: {m['overview']}")
    return "\n".join(lines)


def _movie_metadata(m: dict) -> dict:
    """필터·출처 표시에 쓸 메타데이터. genres 는 부분일치용 문자열로 저장."""
    genres = m.get("genres") or []
    return {
        "movie_id": m.get("movie_id", 0),
        "title": m.get("title", ""),
        "year": m.get("year", 0),
        "genres": ("|" + "|".join(genres) + "|") if genres else "",
        "director": m.get("director", ""),
        "country": m.get("country", ""),
        "vote_average": float(m.get("vote_average", 0.0)),
    }


def load_movies(movies_file: Path) -> list[dict]:
    """수집된 movies.json 을 읽는다."""
    if not movies_file.exists():
        raise FileNotFoundError(
            f"영화 데이터가 없습니다: {movies_file}\n"
            f"먼저 `python -m scripts.fetch_tmdb` 로 수집하세요."
        )
    return json.loads(movies_file.read_text(encoding="utf-8"))


def build_documents(movies: list[dict]) -> list[Document]:
    """영화 리스트를 색인용 Document 리스트로 만든다(긴 줄거리만 추가 분할)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    docs: list[Document] = []
    for m in movies:
        text = _movie_text(m)
        meta = _movie_metadata(m)
        if len(text) <= settings.chunk_size:
            docs.append(Document(page_content=text, metadata=meta))
        else:
            # 줄거리가 매우 긴 드문 경우에만 분할(각 조각에 동일 메타 부여)
            for piece in splitter.split_text(text):
                docs.append(Document(page_content=piece, metadata=dict(meta)))
    return docs


def build_index(force: bool = False) -> Chroma | None:
    """
    movies.json 을 색인해 영속 Chroma 컬렉션을 만든다.

    내용 해시가 기존과 같고 force=False 면 재색인을 건너뛴다(임베딩 생략).
    재색인이 일어난 경우 Chroma 를, 스킵한 경우 None 을 반환한다.
    """
    movies_file = settings.movies_file
    raw = movies_file.read_text(encoding="utf-8") if movies_file.exists() else ""
    if not raw:
        raise FileNotFoundError(
            f"영화 데이터가 없습니다: {movies_file}\n"
            f"먼저 `python -m scripts.fetch_tmdb` 로 수집하세요."
        )
    new_hash = _compute_hash(raw)
    hash_path = _source_hash_path()

    if not force and hash_path.exists() and hash_path.read_text().strip() == new_hash:
        print("[indexing] 데이터 변경 없음 → 재색인 스킵(이미 최신).")
        return None

    movies = json.loads(raw)
    docs = build_documents(movies)
    print(f"[indexing] 영화 수: {len(movies)} / 생성된 문서 수: {len(docs)}")

    embeddings = get_embeddings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    # 기존 컬렉션이 있으면 깨끗이 새로 만든다(중복 방지).
    existing = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )
    try:
        existing.delete_collection()
    except Exception:  # noqa: BLE001 - 최초 실행 등 컬렉션이 없을 수 있음
        pass

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
    )
    hash_path.write_text(new_hash, encoding="utf-8")
    print(f"[indexing] 영속 저장 완료: {settings.chroma_dir}")
    return vectorstore


def get_vectorstore() -> Chroma:
    """이미 색인된 영속 Chroma 컬렉션을 로드한다(재인덱싱 X)."""
    if not settings.chroma_dir.exists():
        raise FileNotFoundError(
            f"벡터스토어가 없습니다: {settings.chroma_dir}\n"
            f"먼저 `python -m scripts.ingest` 로 색인하세요."
        )
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )
