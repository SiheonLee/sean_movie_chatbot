"""
모델 제공자 추상화.

임베딩과 LLM을 만드는 책임을 이 한 곳에 모은다. 임베딩은 Google Gemini로
통일하고, 답변 LLM은 환경 변수로 OpenAI ↔ Claude ↔ Gemini를 독립적으로
교체할 수 있다.

**임베딩과 답변 LLM은 따로 논다.** 답변 LLM을 바꿔도 색인 지문(rag/store.py)에는
임베딩 설정만 들어가므로 재색인이 필요 없다.

import 는 각 분기 안에서 한다. 안 쓰는 제공자의 무거운 패키지(torch 등)를
불필요하게 로딩하지 않기 위함이다.
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import TypeVar

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from rag.config import settings

_T = TypeVar("_T")
_RETRY_DELAY_RE = re.compile(
    r"(?:Please retry in\s+|['\"]retryDelay['\"]:\s*['\"])(\d+(?:\.\d+)?)s",
    re.IGNORECASE,
)
_RETRY_MARGIN_SECONDS = 1.0

# Google이 대기 시간을 알려주지 않는 429도 있다. 그때 쓸 백오프 기준값.
_DEFAULT_BACKOFF_SECONDS = 20.0


class QuotaAwareGoogleEmbeddings(Embeddings):
    """Google 무료 티어 한도를 지키고 429를 재시도한다.

    두 가지를 구분해야 한다. 초기 구현은 이 둘을 혼동해서 실패했다.

    - batch_size: **한 요청에 담을 문서 수**. embed_documents(texts)는 텍스트가
      몇 건이든 batchEmbedContents 요청 1건으로 나간다. 페이로드가 크면 건수와
      무관하게 429가 난다(실측: 50건 성공 1.9초 / 90건 실패).
    - requests_per_minute: **분당 요청 수**. 위 요청의 개수를 제한한다.

    초기 구현은 "문서 1건 = 요청 1건"으로 보고 90건마다 61초를 쉬었다. 실제로는
    499건이 요청 10건이라 쉴 필요가 없었고, 대신 배치가 커서 페이로드 한도를
    넘겼다. 즉 정확히 반대로 튜닝돼 있었다.
    """

    def __init__(
        self,
        delegate: Embeddings,
        *,
        batch_size: int,
        requests_per_minute: int,
        max_retries: int,
    ) -> None:
        if not 1 <= batch_size <= 100:
            raise ValueError("GOOGLE_EMBEDDING_BATCH_SIZE는 1~100 사이여야 합니다.")
        if not 1 <= requests_per_minute <= 100:
            raise ValueError("GOOGLE_EMBEDDING_RPM은 1~100 사이여야 합니다.")
        if max_retries < 0:
            raise ValueError("GOOGLE_EMBEDDING_MAX_RETRIES는 0 이상이어야 합니다.")
        self.delegate = delegate
        self.batch_size = batch_size
        self.requests_per_minute = requests_per_minute
        self.max_retries = max_retries

    @property
    def _seconds_between_requests(self) -> float:
        """요청 간 간격. 60초를 RPM으로 나눠 균등하게 편다."""
        return 60.0 / self.requests_per_minute

    @staticmethod
    def _is_rate_limit(error: Exception) -> bool:
        current: BaseException | None = error
        while current is not None:
            if getattr(current, "code", None) == 429:
                return True
            current = current.__cause__
        message = str(error)
        return "429" in message and "RESOURCE_EXHAUSTED" in message

    @staticmethod
    def _retry_delay(error: Exception) -> float | None:
        """429 응답이 알려준 대기 시간. 명시되지 않았으면 None."""
        match = _RETRY_DELAY_RE.search(str(error))
        if not match:
            return None
        return float(match.group(1)) + _RETRY_MARGIN_SECONDS

    def _with_quota_retry(self, operation: Callable[[], _T]) -> _T:
        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except Exception as error:
                if not self._is_rate_limit(error) or attempt == self.max_retries:
                    raise
                # retryDelay 없는 429에서 포기하면 색인 전체가 날아간다.
                # 알려준 값이 없으면 지수 백오프로 물러난다.
                delay = self._retry_delay(error)
                if delay is None:
                    delay = _DEFAULT_BACKOFF_SECONDS * (2**attempt)
                print(
                    "[embeddings] Google API 요청 한도 도달 → "
                    f"{delay:.1f}초 후 재시도 ({attempt + 1}/{self.max_retries})"
                )
                time.sleep(delay)
        raise AssertionError("도달할 수 없는 재시도 상태입니다.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        total = len(texts)
        batches = [
            texts[start : start + self.batch_size]
            for start in range(0, total, self.batch_size)
        ]
        for batch_index, batch in enumerate(batches):
            batch_embeddings = self._with_quota_retry(
                lambda batch=batch: self.delegate.embed_documents(batch)
            )
            if len(batch_embeddings) != len(batch):
                raise RuntimeError(
                    "Google 임베딩 응답 수가 요청한 문서 수와 일치하지 않습니다."
                )
            embeddings.extend(batch_embeddings)
            print(f"[embeddings] 문서 임베딩 진행: {len(embeddings)}/{total}")

            if batch_index < len(batches) - 1:
                time.sleep(self._seconds_between_requests)

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._with_quota_retry(lambda: self.delegate.embed_query(text))


def get_embeddings() -> Embeddings:
    """검색 용도에 맞춘 Google Gemini 임베딩 객체를 반환한다."""
    if settings.embedding_provider != "google":
        raise RuntimeError(
            "지원하지 않는 EMBEDDING_PROVIDER입니다. Google 마이그레이션 이후에는 "
            "EMBEDDING_PROVIDER=google만 사용할 수 있습니다."
        )
    if not settings.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY가 설정되지 않았습니다. "
            ".env에 Google AI Studio API 키를 입력하세요."
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    # task_type을 객체 수준에서 고정하지 않는다. LangChain이 문서 색인에는
    # RETRIEVAL_DOCUMENT, 사용자 질의에는 RETRIEVAL_QUERY를 각각 적용한다.
    delegate = GoogleGenerativeAIEmbeddings(
        model=settings.google_embedding_model,
        api_key=settings.google_api_key,
        output_dimensionality=settings.google_embedding_dimensions,
    )
    return QuotaAwareGoogleEmbeddings(
        delegate,
        batch_size=settings.google_embedding_batch_size,
        requests_per_minute=settings.google_embedding_rpm,
        max_retries=settings.google_embedding_max_retries,
    )


def get_llm() -> BaseChatModel:
    """설정에 맞는 Chat LLM 객체를 반환한다.

    도구 호출(bind_tools)이 이 파이프라인의 전제라서 제공자를 셋으로 좁혔다.
    HuggingFace Endpoint/Pipeline과 Ollama의 소형 모델은 tool calling을 안정적으로
    지원하지 않아 그래프가 도구를 아예 못 부르는 상태가 된다.
    """
    provider = settings.llm_provider

    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY가 설정되지 않았습니다. .env에 OpenAI API 키를 입력하세요."
            )

        from langchain_openai import ChatOpenAI

        # max_tokens는 langchain-openai가 max_completion_tokens로 바꿔 보낸다.
        # GPT-5 계열에서는 이 값이 "추론 + 답변" 합계다.
        options: dict = {
            "model": settings.openai_model,
            "api_key": settings.openai_api_key,
            "max_tokens": settings.llm_max_tokens,
        }
        # temperature와 reasoning_effort는 넣지 않으면 payload에서 아예 빠진다.
        # 추론 모델이 temperature를 거부하는 경우가 있어(400) 뺄 수 있어야 한다.
        if settings.llm_temperature is not None:
            options["temperature"] = settings.llm_temperature
        if settings.openai_reasoning_effort:
            options["reasoning_effort"] = settings.openai_reasoning_effort
        return ChatOpenAI(**options)

    if provider == "google":
        if not settings.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY가 설정되지 않았습니다. .env에 Google API 키를 입력하세요."
            )

        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.google_llm_model,
            google_api_key=settings.google_api_key,
            temperature=settings.llm_temperature,
        )

    if provider != "anthropic":
        raise RuntimeError(
            f"지원하지 않는 LLM_PROVIDER입니다: {provider!r}. "
            "도구 호출이 필요하므로 openai, anthropic, google만 사용할 수 있습니다."
        )

    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY가 설정되지 않았습니다. .env에 Anthropic API 키를 입력하세요."
        )

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
