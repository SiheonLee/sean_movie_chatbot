from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from langchain_core.embeddings import Embeddings

from rag.config import settings
from rag.providers import QuotaAwareGoogleEmbeddings, get_embeddings, get_llm


class FakeEmbeddings(Embeddings):
    def __init__(self):
        self.document_batches: list[list[str]] = []
        self.query_results: list[list[float] | Exception] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_batches.append(texts)
        return [[float(index)] for index, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        result = self.query_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def quota_aware(delegate, *, batch_size=50, requests_per_minute=90, max_retries=2):
    return QuotaAwareGoogleEmbeddings(
        delegate,
        batch_size=batch_size,
        requests_per_minute=requests_per_minute,
        max_retries=max_retries,
    )


class QuotaAwareEmbeddingTests(unittest.TestCase):
    @patch("rag.providers.time.sleep")
    def test_documents_are_split_at_configured_batch_size(self, _sleep: Mock):
        """배치 크기는 요청당 페이로드를 정한다. 너무 크면 429가 난다(실측 90건)."""
        delegate = FakeEmbeddings()
        embeddings = quota_aware(delegate, batch_size=2)

        result = embeddings.embed_documents(["a", "b", "c", "d", "e"])

        self.assertEqual(delegate.document_batches, [["a", "b"], ["c", "d"], ["e"]])
        self.assertEqual(len(result), 5)

    @patch("rag.providers.time.sleep")
    def test_requests_are_paced_by_rpm_not_by_document_count(self, sleep: Mock):
        """배치 1개 = 요청 1개. 60/RPM 간격으로 편다(예전엔 배치마다 61초를 쉬었다)."""
        delegate = FakeEmbeddings()
        embeddings = quota_aware(delegate, batch_size=2, requests_per_minute=60)

        embeddings.embed_documents(["a", "b", "c", "d"])

        # 배치 2개 → 사이 간격 1회, 60/60 = 1.0초
        sleep.assert_called_once_with(1.0)

    @patch("rag.providers.time.sleep")
    def test_retries_after_server_supplied_delay(self, sleep: Mock):
        delegate = FakeEmbeddings()
        delegate.query_results = [
            RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 2.5s."),
            [0.1, 0.2],
        ]
        embeddings = quota_aware(delegate)

        self.assertEqual(embeddings.embed_query("기생충 감독"), [0.1, 0.2])
        sleep.assert_called_once_with(3.5)

    @patch("rag.providers.time.sleep")
    def test_retries_with_backoff_when_server_gives_no_delay(self, sleep: Mock):
        """Google은 retryDelay 없는 429도 보낸다. 여기서 포기하면 색인이 통째로 날아간다."""
        delegate = FakeEmbeddings()
        delegate.query_results = [
            RuntimeError("429 RESOURCE_EXHAUSTED. You exceeded your current quota."),
            RuntimeError("429 RESOURCE_EXHAUSTED. You exceeded your current quota."),
            [0.3],
        ]
        embeddings = quota_aware(delegate, max_retries=3)

        self.assertEqual(embeddings.embed_query("잔잔한 영화"), [0.3])
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [20.0, 40.0])

    def test_non_rate_limit_errors_are_not_retried(self):
        delegate = FakeEmbeddings()
        delegate.query_results = [ValueError("잘못된 입력")]
        embeddings = quota_aware(delegate)

        with self.assertRaises(ValueError):
            embeddings.embed_query("x")

    def test_oversized_batch_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "BATCH_SIZE"):
            quota_aware(FakeEmbeddings(), batch_size=200)


class GoogleEmbeddingProviderTests(unittest.TestCase):
    def test_google_embeddings_require_api_key(self):
        with (
            patch.object(settings, "embedding_provider", "google"),
            patch.object(settings, "google_api_key", None),
        ):
            with self.assertRaisesRegex(RuntimeError, "GOOGLE_API_KEY"):
                get_embeddings()

    def test_google_embeddings_use_retrieval_model_and_dimensions(self):
        with (
            patch.object(settings, "embedding_provider", "google"),
            patch.object(settings, "google_api_key", "test-key"),
            patch.object(settings, "google_embedding_model", "gemini-embedding-001"),
            patch.object(settings, "google_embedding_dimensions", 3072),
            patch.object(settings, "google_embedding_rpm", 90),
            patch.object(settings, "google_embedding_max_retries", 5),
            patch(
                "langchain_google_genai.GoogleGenerativeAIEmbeddings"
            ) as google_embeddings,
        ):
            result = get_embeddings()

        google_embeddings.assert_called_once_with(
            model="gemini-embedding-001",
            api_key="test-key",
            output_dimensionality=3072,
        )
        self.assertIs(result.delegate, google_embeddings.return_value)
        self.assertEqual(result.requests_per_minute, 90)
        self.assertEqual(result.max_retries, 5)

    def test_legacy_embedding_provider_is_rejected(self):
        with patch.object(settings, "embedding_provider", "huggingface"):
            with self.assertRaisesRegex(RuntimeError, "EMBEDDING_PROVIDER=google"):
                get_embeddings()


class AnthropicProviderTests(unittest.TestCase):
    def test_anthropic_requires_api_key(self):
        with (
            patch.object(settings, "llm_provider", "anthropic"),
            patch.object(settings, "anthropic_api_key", None),
        ):
            with self.assertRaisesRegex(RuntimeError, "ANTHROPIC_API_KEY"):
                get_llm()

    def test_anthropic_uses_configured_haiku_model(self):
        with (
            patch.object(settings, "llm_provider", "anthropic"),
            patch.object(settings, "anthropic_api_key", "test-key"),
            patch.object(settings, "anthropic_model", "claude-haiku-4-5-20251001"),
            patch.object(settings, "llm_temperature", 0.1),
            patch.object(settings, "llm_max_tokens", 512),
            patch("langchain_anthropic.ChatAnthropic") as chat_anthropic,
        ):
            get_llm()

        chat_anthropic.assert_called_once_with(
            model="claude-haiku-4-5-20251001",
            api_key="test-key",
            temperature=0.1,
            max_tokens=512,
        )


if __name__ == "__main__":
    unittest.main()
