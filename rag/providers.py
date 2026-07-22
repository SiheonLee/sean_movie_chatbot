"""
모델 제공자 추상화.

임베딩과 LLM을 만드는 책임을 이 한 곳에 모은다. 환경 변수만 바꾸면
Claude ↔ Ollama ↔ Gemini ↔ HuggingFace 로 갈아끼울 수 있다(의존성 역전).
임베딩 제공자는 LLM 제공자와 독립적이므로 Claude를 사용해도 기존 BGE-M3 색인을 유지한다.

import 는 각 분기 안에서 한다. 안 쓰는 제공자의 무거운 패키지(torch 등)를
불필요하게 로딩하지 않기 위함이다.
"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from rag.config import settings


def get_embeddings() -> Embeddings:
    """설정에 맞는 임베딩 객체를 반환한다."""
    provider = settings.embedding_provider

    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=settings.google_embedding_model,
            google_api_key=settings.google_api_key,
        )

    # 기본값: HuggingFace (로컬 sentence-transformers, API 키 불필요)
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.hf_embedding_model,
        # 코사인 유사도 검색을 위해 임베딩을 정규화한다.
        encode_kwargs={"normalize_embeddings": True},
    )


def get_llm() -> BaseChatModel:
    """설정에 맞는 Chat LLM 객체를 반환한다."""
    provider = settings.llm_provider

    if provider == "anthropic":
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

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.google_llm_model,
            google_api_key=settings.google_api_key,
            temperature=settings.llm_temperature,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
        )

    if provider == "hf_pipeline":
        # 로컬에서 transformers 파이프라인으로 직접 추론한다(GPU/시간 필요).
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

        llm = HuggingFacePipeline.from_model_id(
            model_id=settings.hf_llm_model,
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": settings.llm_max_tokens,
                "temperature": settings.llm_temperature,
                "do_sample": settings.llm_temperature > 0,
            },
        )
        return ChatHuggingFace(llm=llm)

    # 기본값: HuggingFace Inference API (HUGGINGFACEHUB_API_TOKEN 필요)
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    llm = HuggingFaceEndpoint(
        repo_id=settings.hf_llm_model,
        task="text-generation",
        huggingfacehub_api_token=settings.hf_api_token,
        max_new_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )
    return ChatHuggingFace(llm=llm)
