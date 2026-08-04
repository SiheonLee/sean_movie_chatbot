"""
무드 프로파일 검수 (데이터 파이프라인 2.5단계).

사용법:
    python -m scripts.inspect_enriched

여기가 잘못되면 아래 전부가 무너지고, 되돌리려면 전체 재임베딩이다. 배치가 끝나면
반드시 눈으로 확인한다.

무엇을 보는가:
- 태그 빈도: 한 번도 안 쓰인 태그는 어휘에서 빼거나 프롬프트에 설명을 추가한다.
  절반 넘게 붙은 태그는 변별력이 없으니 쪼개거나 제거한다.
- 수치 분포: 전부 3점 근처에 몰려 있으면 필터로 못 쓴다. 가장 흔한 실패다.
  프롬프트의 절대 기준 앵커를 강화하면 개선된다.
- 어휘 이탈: 허용 목록 밖의 값이 나왔는지.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter

from rag.config import settings
from rag.vocab import MOOD_TAGS, PACING, WATCH_SITUATIONS

_NUMERIC_FIELDS = ("violence", "sadness", "tension", "complexity")

# 표준편차가 이보다 작으면 값이 한곳에 몰려 필터 역할을 못 한다.
_MIN_USEFUL_STDEV = 0.8


def _bar(count: int, total: int, width: int = 30) -> str:
    filled = round(width * count / total) if total else 0
    return "█" * filled + "·" * (width - filled)


def main() -> None:
    enriched = json.loads(settings.enriched_file.read_text(encoding="utf-8"))
    total = len(enriched)
    print(f"검수 대상: {total}건\n")

    problems: list[str] = []

    # --- 태그 빈도 -----------------------------------------------------------
    print("=" * 64)
    print("무드 태그 빈도")
    print("=" * 64)
    tags = Counter(t for m in enriched for t in m["mood_tags"])
    for tag in MOOD_TAGS:
        count = tags.get(tag, 0)
        share = count / total
        flag = ""
        if count == 0:
            flag = "  ← 미사용"
            problems.append(f"태그 '{tag}'가 한 번도 쓰이지 않았습니다.")
        elif share > 0.5:
            flag = "  ← 변별력 없음"
            problems.append(f"태그 '{tag}'가 {share:.0%}에 붙었습니다.")
        print(f"  {tag:8} {count:>4} {_bar(count, total)}{flag}")

    unknown = set(tags) - set(MOOD_TAGS)
    if unknown:
        problems.append(f"어휘 밖 태그가 생성되었습니다: {', '.join(sorted(unknown))}")

    # --- 감상 상황 -----------------------------------------------------------
    print("\n" + "=" * 64)
    print("감상 상황 빈도")
    print("=" * 64)
    situations = Counter(s for m in enriched for s in m["watch_situations"])
    for situation in WATCH_SITUATIONS:
        count = situations.get(situation, 0)
        print(f"  {situation:14} {count:>4} {_bar(count, total)}")
    unknown_situations = set(situations) - set(WATCH_SITUATIONS)
    if unknown_situations:
        problems.append(
            f"어휘 밖 감상 상황: {', '.join(sorted(unknown_situations))}"
        )

    # --- 수치 분포 -----------------------------------------------------------
    print("\n" + "=" * 64)
    print("수치 분포 (필터로 쓸 수 있는가)")
    print("=" * 64)
    for field in _NUMERIC_FIELDS:
        values = [m[field] for m in enriched]
        counts = Counter(values)
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        spread = " ".join(f"{score}:{counts.get(score, 0):>3}" for score in range(1, 6))
        flag = ""
        if stdev < _MIN_USEFUL_STDEV:
            flag = "  ← 몰려 있음"
            problems.append(
                f"{field}의 표준편차가 {stdev:.2f}입니다. 필터로 쓰기 어렵습니다."
            )
        print(f"  {field:11} {spread}   평균 {mean:.2f}  표준편차 {stdev:.2f}{flag}")

    pacing = Counter(m["pacing"] for m in enriched)
    print(f"  {'pacing':11} " + " ".join(f"{p}:{pacing.get(p, 0):>3}" for p in PACING))
    unknown_pacing = set(pacing) - set(PACING)
    if unknown_pacing:
        problems.append(f"어휘 밖 pacing: {', '.join(sorted(unknown_pacing))}")

    # --- 자연어 표본 ---------------------------------------------------------
    print("\n" + "=" * 64)
    print("자연어 서술 표본 (줄거리 요약이 아니라 감상 경험인가)")
    print("=" * 64)
    step = max(1, total // 5)
    for movie in enriched[::step][:5]:
        print(f"\n  [{movie.get('title', movie['movie_id'])}]")
        print(f"    분위기: {movie['one_line_vibe']}")
        print(f"    감정선: {movie['emotional_arc']}")
        print(
            f"    태그: {', '.join(movie['mood_tags'])} | "
            f"상황: {', '.join(movie['watch_situations'])}"
        )
        print(
            f"    violence={movie['violence']} sadness={movie['sadness']} "
            f"tension={movie['tension']} complexity={movie['complexity']} "
            f"pacing={movie['pacing']}"
        )

    # --- 판정 ---------------------------------------------------------------
    print("\n" + "=" * 64)
    if problems:
        print(f"확인이 필요한 항목 {len(problems)}건")
        print("=" * 64)
        for problem in problems:
            print(f"  - {problem}")
        print(
            "\n수치가 몰려 있으면 scripts/enrich.py의 절대 기준 앵커를 강화하고 "
            "--force로 다시 돌리세요."
        )
    else:
        print("이상 없음. python -m scripts.build_store 로 색인하세요.")
    print("=" * 64)


if __name__ == "__main__":
    main()
