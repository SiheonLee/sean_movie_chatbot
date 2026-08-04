"""
임베딩 텍스트 조립 + Chroma 구축 (데이터 파이프라인 3단계).

movies.json + enriched.json → 검색 문서 조립 → 영속 Chroma 저장.

사용법:
    python -m scripts.build_store          # 데이터·설정이 바뀐 경우에만 재색인
    python -m scripts.build_store --force  # 변경 여부와 무관하게 강제 재색인

설계 요점:
- 무드 서술을 문서 앞에 둔다. 임베딩 모델은 앞부분 토큰에 더 민감한 경향이 있고,
  무드 질의에 걸리는 게 이 컬렉션의 목적이다.
- 줄거리는 뒤에 붙인다. "우주 배경의 잔잔한 영화" 같은 혼합 질의에 유리하다.
- 수치는 임베딩 텍스트에 넣지 않는다. 임베딩은 숫자의 대소를 이해하지 못한다.
  메타데이터로만 보내 범위 필터에 쓴다.
- 350편 규모라 문서가 짧아 청킹하지 않는다.
"""

from __future__ import annotations

import json
import sys

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.config import settings
from rag.providers import get_embeddings
from rag.store import compute_index_hash, hash_path, index_is_current


def build_page_content(movie: dict, vibe: dict) -> str:
    """임베딩할 검색 문서. 무드를 앞에, 줄거리를 뒤에."""
    return "\n".join([
        f"{movie['title']} ({movie.get('year')})",
        "",
        f"분위기: {vibe['one_line_vibe']}",
        f"감정선: {vibe['emotional_arc']}",
        f"느낌: {', '.join(vibe['mood_tags'])}",
        f"이럴 때 보기 좋습니다: {', '.join(vibe['watch_situations'])}",
        "",
        f"장르: {', '.join(movie.get('genres', []))}",
        f"줄거리: {movie.get('overview', '')}",
    ])


def build_metadata(movie: dict, vibe: dict) -> dict:
    """Chroma 메타데이터는 스칼라만 허용 — 리스트는 문자열로 직렬화한다.

    파이프 문자열(genres/mood_tags/situations)은 Chroma where 필터로 쓸 수 없다.
    메타데이터 $like는 미지원이고 $contains는 조용히 빈 결과를 준다(검증됨).
    파이썬 후처리 필터용이다.
    """
    return {
        # 필터용 수치 — $lte/$gte가 정상 동작하는 유일한 축
        "violence": vibe["violence"],
        "sadness": vibe["sadness"],
        "tension": vibe["tension"],
        "complexity": vibe["complexity"],
        "pacing": vibe["pacing"],
        # 출처 카드 + 후속 TMDB 조회용
        "tmdb_id": movie["movie_id"],
        "title": movie["title"],
        "year": int(movie.get("year") or 0),
        "director": movie.get("director") or "",
        "country": movie.get("country") or "",
        "vote_average": float(movie.get("vote_average") or 0.0),
        "poster_path": movie.get("poster_path") or "",
        # 리스트 → 파이프 구분 문자열 (양쪽 경계 포함해 부분일치 오탐 방지)
        "genres": "|" + "|".join(movie.get("genres", [])) + "|",
        "mood_tags": "|" + "|".join(vibe["mood_tags"]) + "|",
        "situations": "|" + "|".join(vibe["watch_situations"]) + "|",
    }


def build_documents(movies: list[dict], enriched: list[dict]) -> list[Document]:
    by_id = {m["movie_id"]: m for m in movies}
    documents = []
    orphans = 0
    for vibe in enriched:
        movie = by_id.get(vibe["movie_id"])
        if movie is None:
            # movies.json이 재수집되면서 빠진 영화. enriched는 이어하기용 캐시라
            # 과거 항목이 남아 있을 수 있다.
            orphans += 1
            continue
        documents.append(
            Document(
                page_content=build_page_content(movie, vibe),
                metadata=build_metadata(movie, vibe),
            )
        )
    if orphans:
        print(f"[build] 현재 movies.json에 없는 무드 프로파일 {orphans}건은 건너뜁니다.")
    return documents


def main(force: bool = False) -> None:
    movies_raw = settings.movies_file.read_text(encoding="utf-8")
    enriched_raw = settings.enriched_file.read_text(encoding="utf-8")

    if not force and index_is_current():
        print("[build] 데이터·어휘·임베딩 설정 변경 없음 → 재색인 스킵(이미 최신).")
        return

    documents = build_documents(json.loads(movies_raw), json.loads(enriched_raw))
    if not documents:
        raise RuntimeError(
            "색인할 문서가 없습니다. movies.json과 enriched.json의 movie_id가 "
            "일치하는지 확인하세요."
        )
    print(f"[build] 색인할 문서 수: {len(documents)}")

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

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
    )
    hash_path().write_text(
        compute_index_hash(movies_raw, enriched_raw), encoding="utf-8"
    )
    print(f"[build] 영속 저장 완료: {settings.chroma_dir}")
    print("완료. 이제 `uvicorn rag.api:app --reload` 로 API를 띄울 수 있습니다.")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
