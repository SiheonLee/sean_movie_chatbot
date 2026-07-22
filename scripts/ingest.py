"""
색인 실행 스크립트.

사용법:
    python -m scripts.ingest            # 데이터가 바뀐 경우에만 재색인
    python -m scripts.ingest --force    # 변경 여부와 무관하게 강제 재색인

data/movies.json 을 읽어 chroma_db/ 에 영속 벡터스토어를 만든다.
movies.json 내용 해시를 비교해, 변경이 없으면 임베딩을 건너뛴다(시간·비용 절약).
"""
import sys

from rag.indexing import build_index

if __name__ == "__main__":
    force = "--force" in sys.argv
    result = build_index(force=force)
    if result is not None:
        print("완료. 이제 `uvicorn rag.api:app --reload` 로 API를 띄울 수 있습니다.")
