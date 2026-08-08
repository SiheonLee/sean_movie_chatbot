# 영화 정보 검색 한국어 챗봇

LangGraph 도구 호출 기반 영화 챗봇입니다. TMDB API·로컬 벡터 검색·웹 검색을
질문 성격에 따라 골라 씁니다.

```
"봉준호 영화 알려줘"              → TMDB 조건 조회
"비 오는 날 볼 잔잔한 영화"        → 로컬 무드 검색
"잔인하지 않은 액션"              → 무드 검색 + violence ≤ 2 필터
"기생충 평단 반응 어땠어"          → 웹 검색
"평점 가장 높은 영화"             → 범위를 좁혀 달라고 되물음
```

---

## Description

영화에 대한 질문을 받아 **도구를 골라 호출하고**, 그 결과만 근거로 답합니다.
답변과 함께 근거가 된 영화 카드(포스터·감독·출연·평점)를 보여줍니다.

핵심 설계는 **저장소 단위로 도구를 나눈 것**입니다. 각 도구는 다른 도구가 못 하는
일을 합니다.

| 도구 | 소스 | 답하는 질문 |
|---|---|---|
| `search_movies` | TMDB API | "무엇이 있나" — 조건 조회·정렬·개수·OTT 편성·상영 상태 |
| `get_movie_details` | TMDB API | "이 영화 정보" — 감독·출연·배역·시청처 |
| `search_by_vibe` | 로컬 Chroma | "어떤 느낌인가" — 분위기·감정·감상 상황 |
| `web_search` | Tavily | 평단·수상·화제·비하인드·해석 |

**로컬 벡터스토어가 존재하는 유일한 이유는 무드 프로파일입니다.** TMDB는
"일요일 오후에 보기 좋은"이라는 층위를 주지 않습니다. 그건 LLM으로 만들어
붙여야 하고, 그러려면 대상 집합이 유한해야 합니다.

## Features

- **도구 라우팅** — 질문 유형을 정규식으로 분류하지 않고 LLM이 도구를 고릅니다.
  도구를 늘려도 그래프 구조는 그대로입니다.
- **무드 검색** — 영화 499편에 LLM이 생성한 감상 프로파일(분위기·감정선·태그·
  폭력성/슬픔/긴장/난이도 1~5)을 붙여 색인했습니다.
- **부정 조건 처리** — "잔인하지 않은 액션"을 임베딩으로 풀면 오히려 잔인한 영화가
  나옵니다. 임베딩은 부정어를 다루지 못하기 때문입니다. 수치를 메타데이터로
  분리해 `violence ≤ 2` 필터로 해결합니다.
- **되묻기** — "평점 가장 높은 영화"는 표본 하한을 얼마로 두느냐에 따라 1위가
  뒤집힙니다. 단정하지 않고 범위를 좁혀 달라고 되묻습니다.
- **멀티턴** — 체크포인터가 대화를 보존합니다. "그중에 첫 번째 영화 어디서 볼 수
  있어?" 같은 지시대명사를 LLM이 직접 해소합니다.
- **출처 카드** — 답변에 실제로 언급된 영화만 카드로 보여줍니다.

## Architecture

### 그래프

```
START → agent → (도구 호출이 있으면) tools → agent → ... → END
```

2노드 3엣지입니다. 도구가 4개인데도 그래프가 커지지 않는 게 요점입니다.

정규식 질문 분류, 필터 추출, 문서 관련성 평가, 질의 재작성 루프, 출처 번호 파싱은
전부 사라졌습니다. 결과가 부족하면 도구가 *"조건을 완화해보세요"*를 반환하고
LLM이 알아서 재호출합니다.

### 파일 구조

배치 기준은 **실행 시점이 아니라 import 방향**입니다. 서버가 import 하면 `rag/`,
아무도 import 하지 않고 실행만 되면 `scripts/`.

```
rag/                     서버가 import 하는 것만
├── config.py            설정 일원화 (.env → Settings)
├── providers.py         LLM·임베딩 (openai | anthropic | google)
├── vocab.py             무드 태그 어휘 + VOCAB_VERSION
├── schemas.py           MovieVibe (Literal로 어휘 강제)
├── store.py             get_vectorstore() + 색인 지문 검증
├── catalog.py           movies.json 조회 (출처 카드 API 호출 제거)
├── tmdb.py              TMDB 클라이언트 (캐싱·이름→ID 해석)
├── tools.py             도구 4개
├── sources.py           출처 카드 변환기
├── graph.py             도구 호출 그래프
└── api.py               FastAPI

scripts/                 서버 기동 전 1회 실행 — 파이프라인 순서대로
├── fetch_tmdb.py        1. 수집
├── enrich.py            2. 무드 프로파일 생성
├── inspect_enriched.py  2.5 검수
├── build_store.py       3. 색인
├── evaluate_routing.py  평가: 도구 선택 (골든셋)
└── evaluate.py          평가: 답변 품질 (LangSmith)

ui/                      Streamlit 챗 UI
data/                    movies.json · enriched.json (gitignore)
chroma_db/               영속 벡터스토어 (gitignore, 재생성 가능)
```

**불변식: `rag/` → `scripts/` import 금지.** 배치가 `rag/` 안에 있으면 서빙
경로가 인리치먼트용 설정까지 딸려 로드하게 됩니다.

```bash
grep -rn "from scripts\|import scripts" rag/ ui/   # 결과가 없어야 정상
```

### 색인 지문

원본 데이터·어휘 버전·문서 스키마·임베딩 설정을 함께 해시해
`chroma_db/.source_hash`에 저장합니다. 하나라도 다르면 재색인을 요구합니다.
낡은 벡터스토어를 새 데이터로 착각해 조용히 틀린 결과를 내는 걸 막습니다.

## Data

### 수집 (`data/movies.json`, 499편)

TMDB에서 여섯 축으로 모읍니다.

| 축 | 목적 |
|---|---|
| 한국 인기작 (`discover`) | 국내 작품 |
| 해외 인기작 / 고평점작 | `popular`, `top_rated` |
| 드라마·로맨스 평점순 | 잔잔한 계열 확보 |
| 일본·한국 드라마 | 생활극·성장물 |

**`vote_count.gte`를 쿼리에 겁니다.** TMDB의 `popularity`는 페이지뷰·API 트래픽
지표라 조회만 많은 작품이 상위에 올라옵니다. 필터 없이 받으면 수집분의 절반이
표본 부족 작품이 됩니다.

**잔잔한 계열 축이 따로 있는 이유**는 인기순만 쓰면 액션·스릴러가 코퍼스의 67%를
차지해 "비 오는 날 볼 잔잔한 영화" 같은 대표 질의에 후보가 몇 편 안 남기 때문입니다.

### 무드 프로파일 (`data/enriched.json`, 499건)

영화당 LLM 1회 호출로 생성합니다. 약 $2, 10분.

```json
{
  "one_line_vibe": "도시의 소음 속에서 발견하는 작은 아름다움들에 마음이 차분히 가라앉는 경험",
  "emotional_arc": "고요한 일상의 소중함을 깨닫다가, 예상치 못한 관계 속에서...",
  "mood_tags": ["잔잔한", "따뜻한", "서정적인"],
  "watch_situations": ["혼자 밤에", "집중해서 볼 때"],
  "violence": 1, "sadness": 3, "tension": 1, "complexity": 2, "pacing": "느림"
}
```

**자연어는 임베딩 텍스트로, 수치는 메타데이터로 보냅니다.** 사용자 질의도
자연어라 같은 의미 공간에 있어야 하고, 부정·비교·범위 조건은 메타데이터 필터가
정확합니다.

## Prerequisites

- Python 3.14
- [uv](https://docs.astral.sh/uv/)
- API 키
  - **TMDB** — [themoviedb.org](https://www.themoviedb.org/) → 설정 → API (무료)
  - **OpenAI** — 답변 생성 (기본: `gpt-5.6-luna`)
  - **Google AI Studio** — 임베딩 (무료 티어)
  - **Tavily** — [tavily.com](https://tavily.com) 웹 검색 (무료 티어)
  - LangSmith — 추적·평가 (선택)

## How to Run

### 1. 의존성 설치

```bash
uv sync
```

### 2. 환경 변수

```bash
cp .env.example .env
```

값이 비어 있는 항목만 채우면 됩니다.

```bash
TMDB_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
TAVILY_API_KEY=
```

나머지는 코드 기본값과 같으니 바꿀 이유가 생겼을 때만 건드리세요.

### 3. 데이터 파이프라인

순서대로 한 번씩 실행합니다.

```bash
uv run python -m scripts.fetch_tmdb          # 1. 수집 (~7분)
uv run python -m scripts.enrich              # 2. 무드 프로파일 (~10분, 약 $2)
uv run python -m scripts.inspect_enriched    # 2.5 검수
uv run python -m scripts.build_store         # 3. 색인 (~8분)
```

**검수를 건너뛰지 마세요.** 여기가 잘못되면 아래 전부가 무너지고, 되돌리려면
전체 재임베딩입니다. 태그 빈도·수치 분포·어휘 이탈을 자동 판정합니다.

```
violence    1: 198  2: 84  3:108  4:103  5:  6   평균 2.27  표준편차 1.22
sadness     1:  49  2:104  3:104  4:192  5: 50   평균 3.18  표준편차 1.16
```

수치가 전부 3점 근처에 몰리면 필터로 쓸 수 없습니다. 그럴 땐 `scripts/enrich.py`의
절대 기준 앵커를 강화하고 `--force`로 다시 돌리세요.

`enrich`는 **이어하기가 됩니다.** 중단해도 완료분은 보존됩니다. 어휘나 스키마를
바꿨을 때만 `--force`가 필요합니다.

`build_store`는 데이터·어휘·임베딩 설정이 그대로면 **자동으로 스킵**합니다.

### 4. 실행

```bash
uv run uvicorn rag.api:app --reload     # API  → http://127.0.0.1:8000
uv run streamlit run ui/app.py          # UI   → http://localhost:8501
```

### 5. 질문 요청

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "비 오는 날 혼자 보기 좋은 영화 추천해줘", "session_id": "user-1"}'
```

```json
{
  "answer": "비 오는 날 혼자 보기 좋은 영화들을 추천해드립니다...",
  "sources": [
    {
      "title": "언어의 정원", "year": 2013, "director": "신카이 마코토",
      "cast": "이리노 미유, 하나자와 카나", "genres": "애니메이션, 드라마, 로맨스",
      "country": "일본", "vote_average": 7.5,
      "poster_path": "/xxx.jpg", "snippet": "..."
    }
  ]
}
```

같은 `session_id`를 보내면 이전 대화 맥락이 이어집니다.
`GET /health`로 헬스 체크할 수 있습니다.

### 6. 테스트 · 평가

```bash
uv run python -m unittest discover -s tests -t .   # 단위 테스트
uv run python -m scripts.evaluate_routing          # 골든셋 라우팅
uv run python -m scripts.evaluate                  # 답변 품질 (LangSmith)
```

평가가 둘로 나뉘어 있습니다. **`evaluate_routing.py`가 먼저입니다** — 라우팅이
틀리면 검색 품질을 볼 이유가 없고, 라우팅은 docstring 한 줄로 고칠 수 있습니다.

골든셋에는 **함정 구역**이 포함돼 있습니다. 웹처럼 보이지만 TMDB인 질문
("넷플릭스에 뭐 있어", "지금 상영 중"), TMDB처럼 보이지만 웹인 질문
("올해 최고의 영화", "인생영화로 꼽히는") 등 실제로 틀리는 지점입니다.

### 7. Docker Compose

```bash
# .env의 CINEBOT_PASSCODE를 먼저 채워야 합니다.
docker compose up --build
```

`.env`가 그대로 주입됩니다. 데이터와 벡터스토어는 볼륨으로 마운트되므로
컨테이너 안에서 파이프라인을 다시 돌릴 필요는 없습니다. UI는
`http://127.0.0.1:8501`에 열리고, API는 호스트 포트를 게시하지 않아 Compose
내부 네트워크에서만 접근할 수 있습니다. 질문 사용량 DB와 대화 체크포인트는 기존
`checkpoint_data` 볼륨에 보존됩니다.

## Configuration

자주 건드리는 것만 추립니다. 전체는 [.env.example](.env.example)에 주석과 함께
있습니다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` \| `anthropic` \| `google`. 도구 호출이 전제라 이 셋만 지원 |
| `OPENAI_MODEL` | `gpt-5.6-luna` | 답변 LLM. 바꿔도 재색인 불필요(색인 지문은 임베딩만 봄) |
| `OPENAI_REASONING_EFFORT` | (없음) | 비우면 API 기본값. `none`이라야 `LLM_TEMPERATURE`가 전달됨 |
| `LLM_TEMPERATURE` | `0` | 도구 선택은 창의성이 필요 없음 |
| `LLM_MAX_TOKENS` | `8192` | 상한이지 목표가 아님. 추론 모델은 "추론+답변" 합계라 넉넉히 |
| `TMDB_MIN_VOTE_COUNT` | `100` | 표본 부족 작품을 수집 단계에서 배제 |
| `TMDB_QUIET_PAGES` | `3` | 잔잔한 계열 수집축 페이지 수 |
| `GOOGLE_EMBEDDING_BATCH_SIZE` | `50` | 요청당 문서 수. 90이면 429가 남 |
| `CHECKPOINTER` | `memory` | `memory`(휘발) \| `sqlite`(파일 영속) |
| `CINEBOT_PASSCODE` | (없음) | Compose 실행 시 필수인 공유 UI 패스코드. 정식 인증은 아님 |
| `DAILY_QUESTION_LIMIT` | `30` | 익명 브라우저 ID별 UTC 일일 질문 수 |
| `SESSION_QUESTION_LIMIT` | `12` | 한 대화에서 허용할 질문 수 |
| `MAX_CONCURRENT_REQUESTS` | `2` | API 프로세스가 동시에 처리할 질문 수 |

## Notes

- **TMDB 시청처 정보는 JustWatch 제공이라 국내 커버리지가 완전하지 않습니다.**
  쿠팡플레이는 목록에 없습니다. 정보가 없을 때 "제공하지 않는다"가 아니라
  **"확인되지 않는다"**로 답하도록 처리했습니다.
- **라우팅이 결정적이지 않습니다.** 같은 질문에도 도구 선택이 실행마다 달라질 수
  있습니다. `temperature=0`으로 완화했지만 남아 있으며, 모델 상향이 근본 대응입니다.
- 진행 경과와 설계 판단의 근거는 `docs/`에 있습니다.

## References

- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [TMDB API](https://developer.themoviedb.org/docs)
- [Chroma](https://docs.trychroma.com/)
- [Tavily](https://docs.tavily.com/)
- [OpenAI API](https://platform.openai.com/docs/)
- [Anthropic API](https://docs.anthropic.com/)
