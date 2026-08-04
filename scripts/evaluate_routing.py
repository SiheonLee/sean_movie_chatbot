"""
골든셋 라우팅 평가.

사용법:
    python -m scripts.evaluate_routing              # 전체
    python -m scripts.evaluate_routing --group 함정  # 특정 그룹만

**검색 품질보다 도구 선택이 먼저입니다.** 라우팅이 틀리면 아무리 인덱싱을 잘해도
소용없고, 라우팅은 docstring 한 줄로 고칠 수 있습니다. 그래서 답변 내용이 아니라
'어떤 도구를 어떤 인자로 불렀는가'를 채점합니다.

골든셋에는 **함정 구역을 반드시 포함**합니다. 기본 라우팅은 웬만하면 맞고, 실제로
틀리는 건 "넷플릭스에 뭐 있어"(웹처럼 보이지만 TMDB)나 "올해 최고의 영화"(TMDB처럼
보이지만 웹) 같은 경계 사례입니다.

답변 품질(사실성·유용성) 평가는 scripts/evaluate.py(LangSmith)가 담당합니다.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from typing import Callable

from rag.graph import MovieRagGraph

THIS_YEAR = date.today().year


@dataclass
class Case:
    question: str
    expect_tool: str | None  # None = 도구를 부르지 않아야 함
    group: str
    # 인자까지 봐야 하는 경우. 도구 인자 dict를 받아 bool을 반환한다.
    check_args: Callable[[dict], bool] | None = None
    check_label: str = ""
    # 출처 카드가 없어야 하는 경우(되묻기, 웹 검색).
    expect_no_sources: bool = False


def _has(*names: str) -> Callable[[dict], bool]:
    """지정한 인자가 전부 채워졌는지."""
    return lambda args: all(args.get(n) is not None for n in names)


GOLDEN: list[Case] = [
    # --- 기본 라우팅 ---------------------------------------------------------
    Case("봉준호 영화 알려줘", "search_movies", "기본",
         _has("person"), "person 인자"),
    Case("인터스텔라 감독이 누구야", "get_movie_details", "기본"),
    Case("기생충 어디서 볼 수 있어", "get_movie_details", "기본"),
    Case("2020년 이후 SF 영화 몇 편이야?", "search_movies", "기본",
         lambda a: a.get("count_only") is True, "count_only"),
    Case("비 오는 날 혼자 보기 좋은 잔잔한 영화", "search_by_vibe", "기본"),
    Case("슬프지만 마지막엔 위로되는 영화", "search_by_vibe", "기본"),

    # --- 함정: 웹처럼 보이지만 TMDB ------------------------------------------
    Case("넷플릭스에서 볼만한 액션 영화", "search_movies", "함정-TMDB",
         _has("watch_provider"), "watch_provider 인자"),
    Case("지금 극장에서 뭐 해?", "search_movies", "함정-TMDB",
         lambda a: a.get("status") == "now_playing", "status=now_playing"),
    Case("요즘 인기 있는 영화 알려줘", "search_movies", "함정-TMDB",
         lambda a: a.get("status") == "trending", "status=trending"),
    # 도구 이름만 보면 통과하지만 연도가 틀린다. 시스템 프롬프트에 오늘 날짜가
    # 없으면 LLM이 학습 시점 연도(2024)를 넣는다 — 실제로 겪은 버그다.
    Case("올해 개봉한 한국 영화", "search_movies", "함정-TMDB",
         lambda a: a.get("year_from") == THIS_YEAR, f"year_from={THIS_YEAR}"),
    Case(f"{THIS_YEAR}년에 개봉한 영화", "search_movies", "함정-TMDB",
         lambda a: a.get("year_from") == THIS_YEAR, f"year_from={THIS_YEAR}"),

    # --- 함정: TMDB처럼 보이지만 웹 ------------------------------------------
    Case("기생충 평단 반응 어땠어", "web_search", "함정-웹"),
    Case("올해 아카데미 작품상 받은 영화가 뭐야", "web_search", "함정-웹"),
    Case("인터스텔라 결말이 무슨 뜻이야", "web_search", "함정-웹"),
    Case("사람들이 인생영화로 꼽는 작품은?", "web_search", "함정-웹"),

    # --- 부정 조건은 수치 인자로 ---------------------------------------------
    Case("잔인하지 않은 액션 영화", "search_by_vibe", "부정조건",
         _has("max_violence"), "max_violence 인자"),
    Case("머리 안 쓰고 볼 수 있는 영화", "search_by_vibe", "부정조건",
         _has("max_complexity"), "max_complexity 인자"),

    # --- 국적 질의 -------------------------------------------------------------
    # country 인자가 없던 시절 이 질문 하나가 도구를 11번 호출하고, web_search로
    # 새고, 답변이 잘리고, 출처 카드가 14개 생겼다.
    Case("한국 스릴러 영화를 추천해줘", "search_movies", "국적",
         lambda a: a.get("country") is not None, "country 인자"),
    Case("일본 애니메이션 추천해줘", "search_movies", "국적",
         lambda a: a.get("country") is not None, "country 인자"),

    # --- 모르는 제목이라도 먼저 검색 -----------------------------------------
    # 학습 데이터에 없는 최신작을 "어떤 작품인지 알려달라"고 되묻고 끝낸 적이 있다.
    Case("오디세이 평단 반응 어때?", "web_search", "검색우선"),
    Case("군체 평가 어때?", "get_movie_details", "검색우선"),

    # --- 되묻기 --------------------------------------------------------------
    Case("평점 가장 높은 영화", "search_movies", "되묻기",
         lambda a: a.get("sort_by", "").startswith("rating"), "rating 정렬",
         expect_no_sources=True),

    # --- 도구를 부르면 안 되는 것 --------------------------------------------
    Case("안녕? 뭘 할 수 있어?", None, "도구없음"),
]


def run_case(graph: MovieRagGraph, case: Case) -> tuple[bool, str]:
    result = graph.trace(case.question)
    calls = result["tool_calls"]
    names = [c["name"] for c in calls]

    if case.expect_tool is None:
        if calls:
            return False, f"도구를 부르지 않아야 하는데 {names} 호출"
        return True, "도구 미호출"

    if case.expect_tool not in names:
        return False, f"기대 {case.expect_tool} / 실제 {names or '호출 없음'}"

    detail = case.expect_tool
    if case.check_args:
        args = next(c["args"] for c in calls if c["name"] == case.expect_tool)
        if not case.check_args(args):
            return False, f"{case.expect_tool}는 맞지만 {case.check_label} 누락: {args}"
        detail += f" ({case.check_label})"

    if case.expect_no_sources and result["sources"]:
        titles = [s["title"] for s in result["sources"]]
        return False, f"출처가 없어야 하는데 {len(titles)}건: {titles[:3]}"

    return True, detail


def main(group: str | None = None) -> None:
    cases = [c for c in GOLDEN if group is None or group in c.group]
    if not cases:
        groups = sorted({c.group for c in GOLDEN})
        print(f"'{group}'에 해당하는 케이스가 없습니다. 그룹: {', '.join(groups)}")
        return

    graph = MovieRagGraph()
    failures: list[tuple[Case, str]] = []
    current_group = None

    for case in cases:
        if case.group != current_group:
            current_group = case.group
            print(f"\n[{current_group}]")
        try:
            passed, detail = run_case(graph, case)
        except Exception as exc:  # noqa: BLE001 - 한 건 실패로 평가를 멈추지 않는다
            passed, detail = False, f"실행 오류: {exc}"
        print(f"  {'PASS' if passed else 'FAIL'}  {case.question}")
        print(f"        → {detail}")
        if not passed:
            failures.append((case, detail))

    total = len(cases)
    print(f"\n{'=' * 62}")
    print(f"{total - len(failures)}/{total} 통과")
    if failures:
        print(f"{'=' * 62}")
        for case, detail in failures:
            print(f"  [{case.group}] {case.question}\n      {detail}")
        print(
            "\n라우팅 실패는 대개 docstring으로 고칩니다. 해당 도구의 사용 예를 "
            "늘리거나, 잘못 불린 도구의 docstring에 반례를 추가하세요."
        )
    print("=" * 62)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    args = sys.argv[1:]
    selected = args[args.index("--group") + 1] if "--group" in args else None
    main(selected)
