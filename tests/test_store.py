"""색인 지문(fingerprint) 테스트.

지문이 놓치는 변경이 있으면 낡은 벡터스토어를 새 데이터로 착각해 조용히 틀린
검색 결과를 낸다. 각 입력이 실제로 해시를 바꾸는지 확인한다.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from rag import store
from rag.config import settings

MOVIES = '[{"movie_id": 1, "title": "기생충"}]'
ENRICHED = '[{"movie_id": 1, "mood_tags": ["묵직한"]}]'


class IndexFingerprintTests(unittest.TestCase):
    def test_identical_inputs_have_same_hash(self):
        self.assertEqual(
            store.compute_index_hash(MOVIES, ENRICHED),
            store.compute_index_hash(MOVIES, ENRICHED),
        )

    def test_movies_change_invalidates_hash(self):
        self.assertNotEqual(
            store.compute_index_hash(MOVIES, ENRICHED),
            store.compute_index_hash('[{"movie_id": 2, "title": "괴물"}]', ENRICHED),
        )

    def test_enriched_change_invalidates_hash(self):
        """무드 프로파일만 다시 만들어도 재색인이 필요하다."""
        self.assertNotEqual(
            store.compute_index_hash(MOVIES, ENRICHED),
            store.compute_index_hash(MOVIES, '[{"movie_id": 1, "mood_tags": ["잔잔한"]}]'),
        )

    def test_vocab_version_change_invalidates_hash(self):
        """어휘를 바꾸면 기존 색인의 태그 체계와 어긋난다."""
        with patch.object(store, "VOCAB_VERSION", 1):
            first = store.compute_index_hash(MOVIES, ENRICHED)
        with patch.object(store, "VOCAB_VERSION", 2):
            second = store.compute_index_hash(MOVIES, ENRICHED)
        self.assertNotEqual(first, second)

    def test_document_schema_version_change_invalidates_hash(self):
        """임베딩 텍스트 조립 방식이 바뀌면 재색인해야 한다."""
        with patch.object(store, "DOCUMENT_SCHEMA_VERSION", 2):
            first = store.compute_index_hash(MOVIES, ENRICHED)
        with patch.object(store, "DOCUMENT_SCHEMA_VERSION", 3):
            second = store.compute_index_hash(MOVIES, ENRICHED)
        self.assertNotEqual(first, second)

    def test_embedding_model_change_invalidates_hash(self):
        with patch.object(settings, "google_embedding_model", "gemini-embedding-001"):
            first = store.compute_index_hash(MOVIES, ENRICHED)
        with patch.object(settings, "google_embedding_model", "gemini-embedding-2"):
            second = store.compute_index_hash(MOVIES, ENRICHED)
        self.assertNotEqual(first, second)

    def test_embedding_dimension_change_invalidates_hash(self):
        with patch.object(settings, "google_embedding_dimensions", 3072):
            first = store.compute_index_hash(MOVIES, ENRICHED)
        with patch.object(settings, "google_embedding_dimensions", 768):
            second = store.compute_index_hash(MOVIES, ENRICHED)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
