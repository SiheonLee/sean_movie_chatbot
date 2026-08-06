"""
끝나지 않은 LangSmith 트레이스를 사후에 닫습니다.

사용법:
    python -m scripts.close_stale_traces                 # 미리보기(아무것도 안 고침)
    python -m scripts.close_stale_traces --apply         # 실제로 닫기
    python -m scripts.close_stale_traces --older-than 60 # 60분 넘은 것만

**왜 필요한가.** UI가 답변을 받는 도중에 연결이 끊기면(다른 위젯 클릭·새로고침·
네트워크 끊김) Starlette가 SSE 제너레이터를 닫고, 그 닫힘이 LangGraph까지 전파돼
실행이 통째로 사라집니다. 이때 파이썬이 던지는 `GeneratorExit`는 `Exception`이
아니라 `BaseException`이라 보통의 오류 처리 경로를 타지 않고, 결과적으로 LangSmith에
**종료 이벤트가 영영 도착하지 않습니다.** 그 트레이스는 대시보드에서 계속
'pending'으로 남습니다.

`--older-than`이 안전장치입니다. 지금 돌아가는 중인 실행까지 닫아버리면 멀쩡한
트레이스를 망칩니다. 기본 30분은 어떤 답변보다도 넉넉합니다(실측 답변 10초 내외).

끝난 시각은 **같은 트레이스에서 마지막으로 시작된 run의 시각**으로 잡습니다.
지금 시각을 쓰면 몇 시간짜리 실행처럼 보이고, 시작 시각을 쓰면 0초처럼 보입니다.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv()

CLOSE_REASON = "클라이언트 연결이 끊겨 실행이 중단되었습니다(사후에 닫음)."


def main() -> int:
    parser = argparse.ArgumentParser(description="끝나지 않은 LangSmith 트레이스 정리")
    parser.add_argument(
        "--project",
        default=os.getenv("LANGSMITH_PROJECT", "movie-rag"),
        help="LangSmith 프로젝트 이름",
    )
    parser.add_argument(
        "--older-than",
        type=int,
        default=30,
        help="이 분보다 오래된 것만 닫습니다(진행 중인 실행 보호). 기본 30",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="실제로 닫습니다. 없으면 미리보기만 합니다",
    )
    args = parser.parse_args()

    if not os.getenv("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY가 없습니다. .env를 확인하세요.")
        return 1

    from langsmith import Client

    client = Client()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=args.older_than)

    stale = [
        run
        for run in client.list_runs(project_name=args.project, limit=100)
        if run.end_time is None and run.start_time.replace(tzinfo=timezone.utc) < cutoff
    ]
    if not stale:
        print(f"[{args.project}] 닫을 트레이스가 없습니다.")
        return 0

    # 트레이스 단위로 묶어, 그 안에서 마지막으로 시작된 시각을 끝난 시각으로 삼는다.
    by_trace: dict[str, list] = defaultdict(list)
    for run in stale:
        by_trace[str(run.trace_id)].append(run)

    print(f"[{args.project}] 끝나지 않은 run {len(stale)}개 / 트레이스 {len(by_trace)}개")
    if not args.apply:
        print("(미리보기입니다. 실제로 닫으려면 --apply)\n")

    closed = 0
    for trace_id, runs in by_trace.items():
        end_time = max(run.start_time for run in runs)
        print(f"\n  트레이스 {trace_id[:8]} → 종료 시각 {end_time:%Y-%m-%d %H:%M:%S}")
        for run in sorted(runs, key=lambda r: r.start_time):
            print(f"    {run.start_time:%H:%M:%S}  {run.name}")
            if not args.apply:
                continue
            try:
                client.update_run(run.id, end_time=end_time, error=CLOSE_REASON)
                closed += 1
            except Exception as exc:  # noqa: BLE001 - 하나 실패해도 나머지는 닫는다
                print(f"      실패: {type(exc).__name__}: {exc}")

    if args.apply:
        print(f"\n{closed}개를 닫았습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
