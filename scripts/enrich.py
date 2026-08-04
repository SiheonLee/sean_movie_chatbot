"""
무드 프로파일 생성 배치 (데이터 파이프라인 2단계).

data/movies.json → 영화별로 LLM에 '감상 경험'을 물어봄 → data/enriched.json

사용법:
    python -m scripts.enrich          # 아직 처리 안 된 영화만
    python -m scripts.enrich --force  # 전부 다시

설계 요점:
- 결과를 반드시 JSON으로 떨어뜨린다. 임베딩 모델을 바꾸거나 텍스트 조립 방식을
  손볼 때마다 LLM을 다시 부르면 낭비다. 실제로 이 부분을 여러 번 고치게 된다.
- 이어하기가 되게 만든다. 353건 중 300건에서 rate limit에 걸렸을 때 처음부터
  다시 하면 뼈아프다.
- with_structured_output으로 스키마를 강제한다. 파싱 폴백이 필요 없어진다.

다음 단계: python -m scripts.build_store
"""

from __future__ import annotations

import asyncio
import json
import sys

from langchain_core.messages import HumanMessage, SystemMessage

from rag.config import settings
from rag.providers import get_llm
from rag.schemas import MovieVibe
from rag.vocab import MAX_MOOD_TAGS, MOOD_TAGS, WATCH_SITUATIONS

# 동시 요청 수. rate limit 방어이자 실패 시 손실 범위를 좁히는 장치.
_CONCURRENCY = 5

# 무드를 붙일 가치가 있는 영화의 하한. movies.json 수집 단계에서 이미 걸렀지만
# 임계값을 바꿔 재수집하지 않고 여기서 좁히고 싶을 때를 위해 남겨둔다.
_MIN_VOTE_COUNT = 0

ENRICH_PROMPT = f"""당신은 영화의 분위기를 분석하는 큐레이터입니다.
주어진 영화 정보를 보고 '이 영화를 볼 때 어떤 느낌인가'를 분석하세요.

중요한 원칙:
- 줄거리를 요약하지 마세요. 감상 경험을 서술하세요.
- mood_tags는 반드시 다음에서만 고르세요: {", ".join(MOOD_TAGS)}
- watch_situations는 반드시 다음에서만 고르세요: {", ".join(WATCH_SITUATIONS)}
- mood_tags는 최대 {MAX_MOOD_TAGS}개입니다. 칸을 채우려 하지 말고 이 영화를 다른
  영화와 구별해 주는 것만 고르세요. 두 개로 충분하면 두 개만 쓰세요.
  '긴장감있는', '묵직한', '비장한'은 남발하기 쉬운 태그입니다. 정말 그 영화의
  핵심 특징일 때만 쓰세요.
- 수치는 상대적 기준이 아니라 절대 기준입니다. 다른 영화와 비교하지 말고
  아래 앵커에 맞추세요.
  violence 1은 폭력 묘사가 전혀 없음, 3은 액션 장르의 평균,
  5는 직접적 유혈 묘사가 반복됨을 뜻합니다.
  애니메이션·가족 영화의 violence는 보통 1~2입니다.
  1점과 5점을 쓰기를 주저하지 마세요. 전부 3점 근처로 몰리면 쓸모가 없습니다.
- 정보가 부족하면 장르 관례에 따라 보수적으로 추정하세요."""


def build_input(movie: dict) -> str:
    """LLM에 보낼 영화 정보. keywords가 줄거리만으로는 부족한 층위를 보완한다."""
    return (
        f"제목: {movie['title']} ({movie.get('year')})\n"
        f"장르: {', '.join(movie.get('genres', []))}\n"
        f"감독: {movie.get('director', '')}\n"
        f"평점: {movie.get('vote_average')}\n"
        f"상영시간: {movie.get('runtime')}분\n"
        f"키워드: {', '.join(movie.get('keywords', [])[:15])}\n"
        f"줄거리: {movie.get('overview', '')}"
    )


async def enrich_one(llm, movie: dict) -> dict | None:
    try:
        vibe: MovieVibe = await llm.ainvoke(
            [SystemMessage(ENRICH_PROMPT), HumanMessage(build_input(movie))]
        )
    except Exception as exc:  # noqa: BLE001 - 한 건 실패로 배치를 멈추지 않는다
        print(f"[enrich] 실패: {movie['title']} — {exc}", file=sys.stderr)
        return None
    return {"movie_id": movie["movie_id"], "title": movie["title"], **vibe.model_dump()}


async def main(force: bool = False) -> None:
    movies = json.loads(settings.movies_file.read_text(encoding="utf-8"))
    movies = [m for m in movies if m.get("vote_count", 0) >= _MIN_VOTE_COUNT]
    out_path = settings.enriched_file

    # 이미 처리된 것은 건너뛴다 — 중단 후 재개 가능하게.
    done: dict[int, dict] = {}
    if out_path.exists() and not force:
        done = {
            item["movie_id"]: item
            for item in json.loads(out_path.read_text(encoding="utf-8"))
        }

    todo = [m for m in movies if m["movie_id"] not in done]
    print(f"[enrich] 처리 대상 {len(todo)}건 (완료 {len(done)}건 / 전체 {len(movies)}건)")
    if not todo:
        print("[enrich] 처리할 영화가 없습니다.")
        return

    llm = get_llm().with_structured_output(MovieVibe)
    semaphore = asyncio.Semaphore(_CONCURRENCY)
    completed = 0

    async def guarded(movie: dict) -> dict | None:
        nonlocal completed
        async with semaphore:
            result = await enrich_one(llm, movie)
        completed += 1
        if completed % 50 == 0:
            print(f"[enrich] 진행: {completed}/{len(todo)}")
        return result

    results = await asyncio.gather(*[guarded(m) for m in todo])
    failed = sum(1 for item in results if item is None)
    for item in results:
        if item:
            done[item["movie_id"]] = item

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(list(done.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[enrich] 저장 완료: {out_path} ({len(done)}건, 실패 {failed}건)")
    if failed:
        print("[enrich] 실패분은 다시 실행하면 이어서 처리됩니다.")
    print("다음 단계: python -m scripts.inspect_enriched 로 분포를 검수하세요.")


if __name__ == "__main__":
    asyncio.run(main(force="--force" in sys.argv))
