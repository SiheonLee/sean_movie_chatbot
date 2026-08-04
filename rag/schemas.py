"""
무드 프로파일 스키마.

이 프로젝트에서 가장 중요한 설계 결정이 여기 있다. 어떤 정보를 임베딩 텍스트로
보내고 어떤 정보를 메타데이터로 보낼지의 구분이다.

    자연어 서술 → 임베딩 텍스트   사용자 질의도 자연어라 같은 의미 공간에 있어야 함
    1~5 수치    → 메타데이터      부정·비교·범위 조건을 정확히 처리
    범주형      → 메타데이터      정확 일치 필터

"잔인하지 않은 액션"을 임베딩으로 풀면 잔인한 영화가 나온다. 임베딩은 부정어를
제대로 다루지 못해서 "잔인하지 않은"과 "잔인한"이 벡터 공간에서 가깝게 놓인다.
violence <= 2 메타데이터 필터로 풀어야 한다.

같은 이유로 수치를 임베딩 텍스트에 "폭력성 2점"처럼 넣지 않는다. 임베딩은 숫자의
대소를 이해하지 못한다.

이 클래스는 배치(scripts/enrich.py)만 import 하지만 rag/ 에 둔다. Chroma 메타데이터의
계약이고, search_by_vibe의 max_violence/max_sadness 인자가 이 필드와 1:1 대응하므로
한쪽만 고치면 조용히 깨진다.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from rag.vocab import (
    MAX_MOOD_TAGS,
    MAX_WATCH_SITUATIONS,
    MOOD_TAGS,
    PACING,
    WATCH_SITUATIONS,
)

# 어휘를 타입으로 강제한다. list[str]로 두면 프롬프트에 허용 목록을 적어도 LLM이
# 목록 밖 값을 만들어낸다(1차 실측: '감동적인', '섬뜩한', '흥분' 5건).
# Literal로 바꾸면 스키마가 도구 정의로 전달되어 모델이 애초에 벗어나지 못한다.
MoodTag = Literal[tuple(MOOD_TAGS)]  # type: ignore[valid-type]
WatchSituation = Literal[tuple(WATCH_SITUATIONS)]  # type: ignore[valid-type]
Pacing = Literal[tuple(PACING)]  # type: ignore[valid-type]


class MovieVibe(BaseModel):
    """영화 한 편의 분위기 프로파일."""

    # --- 임베딩에 들어갈 자연어 ---
    one_line_vibe: str = Field(
        description="이 영화를 볼 때의 느낌을 한 문장으로. 줄거리 요약이 아니라 감상 경험."
    )
    emotional_arc: str = Field(
        description="감정의 흐름을 한 문장으로. 예: '무겁게 시작해 조용한 위로로 끝난다'"
    )
    mood_tags: list[MoodTag] = Field(
        min_length=2,
        max_length=MAX_MOOD_TAGS,
        description=f"가장 특징적인 것만 2~{MAX_MOOD_TAGS}개. 애매하면 적게 고르세요.",
    )
    watch_situations: list[WatchSituation] = Field(
        min_length=1,
        max_length=MAX_WATCH_SITUATIONS,
        description=f"가장 잘 맞는 상황 1~{MAX_WATCH_SITUATIONS}개",
    )

    # --- 메타데이터 필터로 쓸 수치 ---
    violence: int = Field(ge=1, le=5, description="폭력·유혈 수위. 1=전혀없음, 5=매우잔인")
    sadness: int = Field(ge=1, le=5, description="슬픔의 강도")
    tension: int = Field(ge=1, le=5, description="긴장·스릴의 강도")
    complexity: int = Field(ge=1, le=5, description="이해 난이도. 1=가볍게 봐도 됨")
    pacing: Pacing
