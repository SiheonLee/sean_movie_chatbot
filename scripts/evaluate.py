"""D8 고정 평가셋을 한 번의 trace 실행으로 평가한다.

공식 실행 전제:
- 12개 질문과 기준은 ``eval/dataset.jsonl``의 SHA-256으로 고정한다.
- 기존 ``movies-rag-eval``을 재사용하지 않는다.
- target은 GPT-5.6 Luna, judge는 별도 GPT-5.6 Terra를 사용한다.
- 개발자 규칙 점수와 AI judge 점수를 별도 feedback key로 기록한다.
- 원본 Chroma는 임시 복사본으로 대체해 런타임 쓰기로부터 보호한다.

사용법:
    LANGSMITH_DATASET=movies-rag-eval-d8-v1 python -m scripts.evaluate
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client
from langsmith.evaluation import evaluate
from pydantic import BaseModel, Field

from rag.config import ROOT_DIR, settings
from rag.graph import (
    _RECURSION_FALLBACK,
    _answer_mention_span,
    MovieRagGraph,
)

D8_DATASET_NAME = "movies-rag-eval-d8-v1"
D8_DATASET_VERSION = "d8-v1"
EXPECTED_GROUPS = {"tmdb": 4, "local": 4, "web": 4}
EXPECTED_REQUIRED_ARG_COUNT = 33
EXPECTED_CARD_CASES = 7
EXPECTED_WEB_CASES = 4

_PREDICATE_OPS = {"eq", "one_of", "gte", "present", "nonempty", "contains_groups"}
_BOLD_MOVIE_RE = re.compile(
    r"\*\*(?P<title>[^*\n]{1,80})\*\*\s*[（(]\s*"
    r"(?P<year>\d{4})\s*년?\s*[)）]"
)


def load_cases(path: Path | None = None) -> list[dict]:
    """고정 JSONL을 읽고 D8의 표본 수·그룹·채점 계약을 검증한다."""
    path = path or settings.eval_file
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"평가셋 {line_number}행 JSON 오류: {exc}") from exc
        rows.append(row)

    if len(rows) != 12:
        raise ValueError(f"D8 평가셋은 정확히 12건이어야 합니다: {len(rows)}건")
    ids = [row.get("id") for row in rows]
    questions = [row.get("question") for row in rows]
    if None in ids or len(ids) != len(set(ids)):
        raise ValueError("평가 ID는 비어 있지 않고 중복되지 않아야 합니다.")
    if None in questions or len(questions) != len(set(questions)):
        raise ValueError("평가 질문은 비어 있지 않고 중복되지 않아야 합니다.")
    groups = Counter(row.get("group") for row in rows)
    if dict(groups) != EXPECTED_GROUPS:
        raise ValueError(f"검색 경로별 4건 구성이 아닙니다: {dict(groups)}")

    predicate_count = 0
    card_cases = 0
    web_cases = 0
    for row in rows:
        if "answer" in row:
            raise ValueError(f"{row['id']}: D8에는 고정 answer 문자열을 두지 않습니다.")
        expectation = row.get("expectation") or {}
        expected_tool = expectation.get("expected_tool")
        allowed_tools = expectation.get("allowed_tools") or []
        if not expected_tool or expected_tool not in allowed_tools:
            raise ValueError(f"{row['id']}: expected_tool/allowed_tools 계약이 잘못됐습니다.")
        for predicate in expectation.get("required_args") or []:
            if predicate.get("op") not in _PREDICATE_OPS or not predicate.get("name"):
                raise ValueError(f"{row['id']}: 지원하지 않는 필수 인자 predicate입니다.")
            predicate_count += 1
        card_cases += expectation.get("movie_cards") == "required"
        web_cases += expectation.get("web_sources") == "required"

    if predicate_count != EXPECTED_REQUIRED_ARG_COUNT:
        raise ValueError(
            f"필수 인자 분모가 고정값과 다릅니다: {predicate_count}/"
            f"{EXPECTED_REQUIRED_ARG_COUNT}"
        )
    if card_cases != EXPECTED_CARD_CASES or web_cases != EXPECTED_WEB_CASES:
        raise ValueError(
            f"구조화 출처 사례 수가 잘못됐습니다: card={card_cases}, web={web_cases}"
        )
    return rows


def dataset_checksum(path: Path | None = None) -> str:
    path = path or settings.eval_file
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _example_payload(case: dict) -> dict:
    return {
        "inputs": {"question": case["question"], "case_id": case["id"]},
        "outputs": {
            "dataset_version": D8_DATASET_VERSION,
            "group": case["group"],
            "expectation": case["expectation"],
        },
        "metadata": {
            "case_id": case["id"],
            "group": case["group"],
            "dataset_version": D8_DATASET_VERSION,
        },
    }


def _stored_example_payload(example) -> dict:
    metadata = example.metadata or {}
    return {
        "inputs": example.inputs,
        "outputs": example.outputs,
        "metadata": {
            "case_id": metadata.get("case_id"),
            "group": metadata.get("group"),
            "dataset_version": metadata.get("dataset_version"),
        },
    }


def sync_dataset(
    client: Client,
    cases: list[dict],
    *,
    dataset_name: str = D8_DATASET_NAME,
):
    """버전 dataset을 만들거나 원격 내용이 로컬 고정본과 같은지 검증한다.

    질문만 같으면 기존 reference를 방치하던 과거 동기화와 달리 inputs, outputs,
    metadata를 모두 비교한다. 이미 있는 dataset이 다르면 수정·추가·삭제하지 않고
    중단해 공식 실행 뒤 기준이 바뀌는 일을 막는다.
    """
    if dataset_name == "movies-rag-eval":
        raise ValueError("기존 movies-rag-eval dataset은 D8에 재사용할 수 없습니다.")

    datasets = list(client.list_datasets(dataset_name=dataset_name))
    if len(datasets) > 1:
        raise RuntimeError(f"동일한 이름의 LangSmith dataset이 여러 개입니다: {dataset_name}")
    expected = {case["id"]: _example_payload(case) for case in cases}

    if not datasets:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="D8 영화 추천 RAG 고정 평가셋: TMDB 4, 로컬 Chroma 4, 웹 4",
            metadata={
                "dataset_version": D8_DATASET_VERSION,
                "sha256": dataset_checksum(),
            },
        )
        client.create_examples(
            dataset_id=dataset.id,
            examples=list(expected.values()),
        )
        print(f"[eval] 새 고정 Dataset 생성: {dataset.name} (12건)")
        return dataset

    dataset = datasets[0]
    stored_examples = list(client.list_examples(dataset_id=dataset.id))
    actual: dict[str, dict] = {}
    duplicates: list[str] = []
    for example in stored_examples:
        case_id = (example.inputs or {}).get("case_id")
        if case_id in actual:
            duplicates.append(str(case_id))
        actual[str(case_id)] = _stored_example_payload(example)

    mismatches = sorted(
        case_id
        for case_id in set(expected) | set(actual)
        if expected.get(case_id) != actual.get(case_id)
    )
    if duplicates or mismatches:
        details = sorted(set(duplicates + mismatches))
        raise RuntimeError(
            "기존 D8 dataset이 로컬 고정본과 다릅니다. 원격 내용을 자동 변경하지 "
            f"않습니다: {', '.join(details)}"
        )
    print(f"[eval] 기존 고정 Dataset 검증 완료: {dataset.name} (12건 동일)")
    return dataset


@contextmanager
def isolated_chroma(source: Path | None = None) -> Iterator[Path]:
    """원본 Chroma를 임시 복사본으로 바꿔 평가 중 런타임 쓰기를 격리한다."""
    source = (source or settings.chroma_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Chroma 원본 디렉터리가 없습니다: {source}")

    from rag import tools as rag_tools

    original = settings.chroma_dir
    with tempfile.TemporaryDirectory(prefix="movie-rag-d8-chroma-") as tmp:
        copied = Path(tmp) / "chroma_db"
        shutil.copytree(source, copied)
        rag_tools._vectorstore.cache_clear()
        settings.chroma_dir = copied
        try:
            yield copied
        finally:
            rag_tools._vectorstore.cache_clear()
            settings.chroma_dir = original


def make_target(graph: MovieRagGraph) -> Callable[[dict], dict]:
    """한 질문의 answer·routing·구조화 evidence를 한 실행에서 보존한다."""

    def target(inputs: dict) -> dict:
        try:
            result = graph.trace(inputs["question"])
        except Exception as exc:  # noqa: BLE001 - 한 사례 오류도 점수 분모에 남긴다
            return {
                "answer": "",
                "sources": [],
                "web_sources": [],
                "attributions": [],
                "tool_calls": [],
                "tool_results": [],
                "error": type(exc).__name__,
            }
        return {**result, "error": None}

    return target


@dataclass
class ExternalCallCounter:
    """공식 실행의 논리적 외부 호출 수.

    LangSmith 전송 자체와 OpenAI SDK 내부의 투명한 HTTP 재시도는 포함하지 않는다.
    Google embedding은 프로젝트가 직접 수행하는 429 재시도 시도까지 센다.
    """

    target_llm: int = 0
    judge_llm: int = 0
    tmdb_http: int = 0
    tavily_search: int = 0
    google_embedding_attempts: int = 0

    def as_dict(self) -> dict[str, int]:
        calls = {
            "target_llm": self.target_llm,
            "judge_llm": self.judge_llm,
            "tmdb_http": self.tmdb_http,
            "tavily_search": self.tavily_search,
            "google_embedding_attempts": self.google_embedding_attempts,
        }
        return {**calls, "logical_total_excluding_langsmith": sum(calls.values())}


class _CountingInvoker:
    def __init__(self, delegate, on_invoke: Callable[[], None]) -> None:
        self.delegate = delegate
        self.on_invoke = on_invoke

    def invoke(self, *args, **kwargs):
        self.on_invoke()
        return self.delegate.invoke(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.delegate, name)


@contextmanager
def count_external_calls(
    graph: MovieRagGraph, counter: ExternalCallCounter
) -> Iterator[None]:
    """Target LLM·TMDB·Tavily·Google embedding 호출을 실행 중 집계한다."""
    from rag import providers as rag_providers
    from rag import tmdb as rag_tmdb
    from rag import tools as rag_tools

    original_graph_llm = graph.llm
    original_tmdb_get = rag_tmdb._get
    original_web_factory = rag_tools._web_search_client
    original_embedding_retry = rag_providers.QuotaAwareGoogleEmbeddings._with_quota_retry

    def count_target() -> None:
        counter.target_llm += 1

    def counted_tmdb_get(*args, **kwargs):
        counter.tmdb_http += 1
        return original_tmdb_get(*args, **kwargs)

    def counted_web_factory():
        client = original_web_factory()
        return _CountingInvoker(
            client,
            lambda: setattr(counter, "tavily_search", counter.tavily_search + 1),
        )

    def counted_embedding_retry(self, operation):
        def counted_operation():
            counter.google_embedding_attempts += 1
            return operation()

        return original_embedding_retry(self, counted_operation)

    graph.llm = _CountingInvoker(original_graph_llm, count_target)
    rag_tmdb._get = counted_tmdb_get
    rag_tools._web_search_client = counted_web_factory
    rag_providers.QuotaAwareGoogleEmbeddings._with_quota_retry = counted_embedding_retry
    try:
        yield
    finally:
        graph.llm = original_graph_llm
        rag_tmdb._get = original_tmdb_get
        rag_tools._web_search_client = original_web_factory
        rag_providers.QuotaAwareGoogleEmbeddings._with_quota_retry = original_embedding_retry


@dataclass(frozen=True)
class FractionScore:
    numerator: int
    denominator: int
    comment: str

    @property
    def score(self) -> float:
        return self.numerator / self.denominator if self.denominator else 0.0


def _normalized(value) -> str:
    return re.sub(r"\s+", "", str(value)).casefold()


def _matches_predicate(args: dict, predicate: dict) -> bool:
    name = predicate["name"]
    operation = predicate["op"]
    value = args.get(name)
    if operation == "present":
        return value is not None
    if operation == "nonempty":
        return bool(str(value or "").strip())
    if operation == "eq":
        expected = predicate.get("value")
        if isinstance(expected, str):
            return _normalized(value) == _normalized(expected)
        return value == expected
    if operation == "one_of":
        return any(_normalized(value) == _normalized(item) for item in predicate["value"])
    if operation == "gte":
        try:
            return float(value) >= float(predicate["value"])
        except (TypeError, ValueError):
            return False
    if operation == "contains_groups":
        text = _normalized(value or "")
        return all(
            any(_normalized(alternative) in text for alternative in group)
            for group in predicate["groups"]
        )
    raise ValueError(f"지원하지 않는 predicate op: {operation}")


def _movie_key(source: dict) -> tuple:
    movie_id = int(source.get("movie_id") or 0)
    if movie_id:
        return ("id", movie_id)
    return (
        "title-year",
        _normalized(source.get("title") or ""),
        int(source.get("year") or 0),
    )


def _answer_movie_mentions(answer: str, candidates: list[dict]) -> list[tuple]:
    """명시적인 제목(연도) 언급을 답변 순서로 찾는다.

    후보 영화는 제품과 같은 매칭 함수를 쓴다. 시스템 형식인 굵은 제목·연도는
    별도로 파싱해 도구 후보에 없는 제목도 개발자 규칙에서 잡는다.
    """
    mentions: list[tuple[int, int, tuple]] = []
    candidate_spans: list[tuple[int, int, tuple]] = []
    for source in candidates:
        span = _answer_mention_span(answer, source)
        if span is not None:
            item = (*span, _movie_key(source))
            candidate_spans.append(item)
            mentions.append(item)

    for match in _BOLD_MOVIE_RE.finditer(answer):
        span = match.span()
        if any(start == span[0] and end == span[1] for start, end, _ in candidate_spans):
            continue
        overlapping = next(
            (
                key
                for start, end, key in candidate_spans
                if not (span[1] <= start or span[0] >= end)
            ),
            None,
        )
        key = overlapping or (
            "title-year",
            _normalized(match.group("title").strip(" -–—:·")),
            int(match.group("year")),
        )
        mentions.append((span[0], span[1], key))

    ordered: list[tuple] = []
    seen: set[tuple] = set()
    for _, _, key in sorted(mentions, key=lambda item: item[0]):
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def score_developer(outputs: dict, expectation: dict) -> dict[str, FractionScore]:
    """한 사례의 개발자 규칙 점수와 정확한 분자·분모를 계산한다."""
    expected_tool = expectation["expected_tool"]
    calls = [call for call in outputs.get("tool_calls") or [] if isinstance(call, dict)]
    names = [call.get("name") for call in calls]
    expected_calls = [call for call in calls if call.get("name") == expected_tool]
    tool_results = [
        result for result in outputs.get("tool_results") or [] if isinstance(result, dict)
    ]
    successful_expected = [
        result
        for result in tool_results
        if result.get("name") == expected_tool and result.get("success") is True
    ]

    metrics: dict[str, FractionScore] = {}
    routed = expected_tool in names
    metrics["routing"] = FractionScore(int(routed), 1, f"expected={expected_tool}, actual={names}")

    predicates = expectation.get("required_args") or []
    args = (expected_calls[0].get("args") or {}) if expected_calls else {}
    arg_hits = sum(_matches_predicate(args, predicate) for predicate in predicates)
    metrics["required_args"] = FractionScore(
        arg_hits,
        len(predicates),
        f"{arg_hits}/{len(predicates)} predicates, args={args}",
    )

    allowed = set(expectation.get("allowed_tools") or [])
    max_calls = int(expectation.get("max_calls") or 1)
    disciplined = (
        names == [expected_tool]
        and len(calls) <= max_calls
        and all(name in allowed for name in names)
    )
    metrics["call_discipline"] = FractionScore(
        int(disciplined), 1, f"calls={names}, max={max_calls}"
    )

    answer = str(outputs.get("answer") or "")
    completed = (
        not outputs.get("error")
        and bool(answer.strip())
        and answer != _RECURSION_FALLBACK
        and bool(successful_expected)
    )
    metrics["error_free_completion"] = FractionScore(
        int(completed),
        1,
        f"error={outputs.get('error')!r}, successful_expected={bool(successful_expected)}",
    )

    expected_attributions = expectation.get("expected_attributions") or []
    actual_attributions = outputs.get("attributions") or []
    metrics["attribution"] = FractionScore(
        int(actual_attributions == expected_attributions),
        1,
        f"expected={expected_attributions}, actual={actual_attributions}",
    )

    cards = outputs.get("sources") or []
    web_sources = outputs.get("web_sources") or []
    movie_policy = expectation.get("movie_cards")
    web_policy = expectation.get("web_sources")
    source_shape = (
        (movie_policy != "required" or bool(cards))
        and (movie_policy != "forbidden" or not cards)
        and (web_policy != "required" or bool(web_sources))
        and (web_policy != "forbidden" or not web_sources)
    )
    metrics["structured_source_shape"] = FractionScore(
        int(source_shape),
        1,
        f"cards={len(cards)}, web_sources={len(web_sources)}",
    )

    if movie_policy == "required":
        candidate_sources = [
            source
            for result in tool_results
            if result.get("success") is True
            for source in result.get("sources") or []
            if isinstance(source, dict)
        ]
        candidate_keys = {_movie_key(source) for source in candidate_sources}
        card_keys = [_movie_key(source) for source in cards if isinstance(source, dict)]
        mention_keys = _answer_movie_mentions(answer, candidate_sources)

        provenance = bool(card_keys) and all(key in candidate_keys for key in card_keys)
        grounding = bool(mention_keys) and all(key in candidate_keys for key in mention_keys)
        membership = bool(card_keys) and set(card_keys) == set(mention_keys)
        alignment = bool(card_keys) and card_keys == mention_keys
        metrics["card_provenance"] = FractionScore(
            int(provenance), 1, f"cards={card_keys}, candidates={sorted(candidate_keys, key=str)}"
        )
        metrics["answer_movie_grounding"] = FractionScore(
            int(grounding), 1, f"mentions={mention_keys}"
        )
        metrics["answer_card_membership"] = FractionScore(
            int(membership), 1, f"mentions={mention_keys}, cards={card_keys}"
        )
        metrics["answer_card_order"] = FractionScore(
            int(alignment), 1, f"mentions={mention_keys}, cards={card_keys}"
        )

    if web_policy == "required":
        evidence_urls = {
            source.get("url")
            for result in tool_results
            if result.get("success") is True
            for source in result.get("web_sources") or []
            if isinstance(source, dict) and source.get("url")
        }
        final_urls = [
            source.get("url")
            for source in web_sources
            if isinstance(source, dict) and source.get("url")
        ]
        separated = not cards and bool(final_urls) and all(url in evidence_urls for url in final_urls)
        metrics["web_source_separation"] = FractionScore(
            int(separated), 1, f"cards={len(cards)}, urls={final_urls}"
        )
    return metrics


def _expectation_from_example(example) -> dict:
    outputs = example.outputs or {}
    return outputs.get("expectation") or {}


def developer_rules(run, example) -> dict:
    metrics = score_developer(run.outputs or {}, _expectation_from_example(example))
    return {
        "results": [
            {
                "key": f"developer_{key}",
                "score": metric.score,
                "comment": metric.comment,
                "metadata": {
                    "numerator": metric.numerator,
                    "denominator": metric.denominator,
                },
            }
            for key, metric in metrics.items()
        ]
    }


def aggregate_developer_scores(runs: list, examples: list) -> dict[str, FractionScore]:
    totals: dict[str, list[int]] = {}
    for run, example in zip(runs, examples, strict=True):
        for key, metric in score_developer(
            run.outputs or {}, _expectation_from_example(example)
        ).items():
            total = totals.setdefault(key, [0, 0])
            total[0] += metric.numerator
            total[1] += metric.denominator
    return {
        key: FractionScore(numerator, denominator, f"{numerator}/{denominator}")
        for key, (numerator, denominator) in totals.items()
    }


def developer_summary(runs: list, examples: list) -> dict:
    totals = aggregate_developer_scores(runs, examples)
    return {
        "results": [
            {
                "key": f"developer_summary_{key}",
                "score": metric.score,
                "comment": metric.comment,
                "metadata": {
                    "numerator": metric.numerator,
                    "denominator": metric.denominator,
                },
            }
            for key, metric in totals.items()
        ]
    }


JudgeScore = Literal[0.0, 0.5, 1.0]


class JudgeCriterion(BaseModel):
    score: JudgeScore
    reason: str = Field(min_length=1, max_length=500)


class JudgeVerdict(BaseModel):
    direct_answer: JudgeCriterion
    evidence_grounding: JudgeCriterion
    condition_relevance: JudgeCriterion
    uncertainty_calibration: JudgeCriterion
    usefulness_clarity: JudgeCriterion


_JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 영화 추천 RAG의 독립 채점자입니다. 제공된 실행 evidence만 보고 "
            "평가하며, 모델 기억이나 외부 지식으로 사실을 보충하지 마세요. Target의 "
            "모델명은 알 필요가 없습니다. 각 축은 0, 0.5, 1 중 하나입니다.\n\n"
            "- direct_answer: 질문에 직접 답했는가\n"
            "- evidence_grounding: 답변 주장이 성공한 도구 결과로 뒷받침되는가\n"
            "- condition_relevance: 추천 이유·설명이 질문 조건과 관련 있는가\n"
            "- uncertainty_calibration: 확인되지 않은 사실을 단정하지 않았는가\n"
            "- usefulness_clarity: 전체적으로 유용하고 이해하기 쉬운가\n\n"
            "도구 실패나 빈 evidence를 솔직히 알린 답변은 불확실성 축에서 인정할 수 "
            "있지만, 근거 없는 사실에는 evidence_grounding 점수를 주지 마세요.",
        ),
        ("human", "다음 JSON 실행을 채점하세요.\n{payload}"),
    ]
)

_JUDGE_FIELDS = (
    "direct_answer",
    "evidence_grounding",
    "condition_relevance",
    "uncertainty_calibration",
    "usefulness_clarity",
)


def _judge_feedback(verdict: JudgeVerdict, *, error: str | None = None) -> dict:
    results = []
    scores = []
    for field_name in _JUDGE_FIELDS:
        criterion = getattr(verdict, field_name)
        score = float(criterion.score)
        scores.append(score)
        results.append(
            {
                "key": f"ai_judge_{field_name}",
                "score": score,
                "comment": criterion.reason,
                "metadata": {"judge_model": settings.judge_model, "error": error},
            }
        )
    results.append(
        {
            "key": "ai_judge_mean",
            "score": sum(scores) / len(scores),
            "comment": "5개 AI judge 축의 사례별 평균",
            "metadata": {"judge_model": settings.judge_model, "error": error},
        }
    )
    results.append(
        {
            "key": "ai_judge_valid",
            "score": 1,
            "comment": "AI judge 구조화 응답 정상",
            "metadata": {"judge_model": settings.judge_model, "error": error},
        }
    )
    return {"results": results}


def _failed_judge_feedback(error: Exception) -> dict:
    reason = f"judge 실행 오류: {type(error).__name__}"
    metadata = {"judge_model": settings.judge_model, "error": type(error).__name__}
    return {
        "results": [
            *[
                {
                    "key": f"ai_judge_{field_name}",
                    "score": None,
                    "comment": reason,
                    "metadata": metadata,
                }
                for field_name in _JUDGE_FIELDS
            ],
            {
                "key": "ai_judge_mean",
                "score": None,
                "comment": reason,
                "metadata": metadata,
            },
            {
                "key": "ai_judge_valid",
                "score": 0,
                "comment": reason,
                "metadata": metadata,
            },
        ]
    }


def make_llm_judge(chain=None, *, counter: ExternalCallCounter | None = None):
    """Target 설정을 재사용하지 않는 GPT-5.6 Terra judge를 만든다."""
    if chain is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY가 없어 Terra judge를 만들 수 없습니다.")
        from langchain_openai import ChatOpenAI

        judge_model = ChatOpenAI(
            model=settings.judge_model,
            api_key=settings.openai_api_key,
            max_tokens=settings.judge_max_tokens,
        ).with_structured_output(JudgeVerdict)
        chain = _JUDGE_PROMPT | judge_model

    def llm_judge(run, example) -> dict:
        outputs = run.outputs or {}
        payload = {
            "question": (example.inputs or {}).get("question", ""),
            "judge_focus": _expectation_from_example(example).get("judge_focus", ""),
            "answer": outputs.get("answer", ""),
            "tool_results": outputs.get("tool_results", []),
            "final_sources": outputs.get("sources", []),
            "final_web_sources": outputs.get("web_sources", []),
            "execution_error": outputs.get("error"),
        }
        try:
            if counter is not None:
                counter.judge_llm += 1
            raw = chain.invoke({"payload": json.dumps(payload, ensure_ascii=False)})
            verdict = raw if isinstance(raw, JudgeVerdict) else JudgeVerdict.model_validate(raw)
        except Exception as exc:  # noqa: BLE001 - judge 실패도 12건 분모에서 숨기지 않는다
            return _failed_judge_feedback(exc)
        return _judge_feedback(verdict)

    return llm_judge


def current_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_official_config() -> None:
    """실수로 다른 target/judge/dataset 설정으로 공식 실행하지 못하게 막는다."""
    expected = {
        "LANGSMITH_DATASET": D8_DATASET_NAME,
        "LLM_PROVIDER": "openai",
        "OPENAI_MODEL": "gpt-5.6-luna",
        "LLM_TEMPERATURE": 0.0,
        "OPENAI_REASONING_EFFORT": "none",
        "LLM_MAX_TOKENS": 8192,
        "JUDGE_MODEL": "gpt-5.6-terra",
        "JUDGE_MAX_TOKENS": 1024,
    }
    actual = {
        "LANGSMITH_DATASET": settings.dataset_name,
        "LLM_PROVIDER": settings.llm_provider,
        "OPENAI_MODEL": settings.openai_model,
        "LLM_TEMPERATURE": settings.llm_temperature,
        "OPENAI_REASONING_EFFORT": settings.openai_reasoning_effort,
        "LLM_MAX_TOKENS": settings.llm_max_tokens,
        "JUDGE_MODEL": settings.judge_model,
        "JUDGE_MAX_TOKENS": settings.judge_max_tokens,
    }
    mismatches = [
        f"{name}={actual[name]!r} (기대 {value!r})"
        for name, value in expected.items()
        if actual[name] != value
    ]
    if mismatches:
        raise RuntimeError("D8 고정 실행 설정이 다릅니다: " + "; ".join(mismatches))


def main() -> None:
    cases = load_cases()
    validate_official_config()

    commit_sha = current_commit()
    checksum = dataset_checksum()
    evaluator_checksum = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    experiment_prefix = (
        f"movie-rag-d8-{commit_sha[:7]}-{date.today().strftime('%Y%m%d')}-fixed"
    )
    metadata = {
        "d8_dataset_version": D8_DATASET_VERSION,
        "dataset_sha256": checksum,
        "evaluator_sha256": evaluator_checksum,
        "commit_sha": commit_sha,
        "target_provider": settings.llm_provider,
        "target_model": settings.openai_model,
        "target_temperature": settings.llm_temperature,
        "target_reasoning_effort": settings.openai_reasoning_effort,
        "target_max_tokens": settings.llm_max_tokens,
        "judge_provider": "openai",
        "judge_model": settings.judge_model,
        "judge_max_tokens": settings.judge_max_tokens,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.google_embedding_model,
        "isolated_chroma": True,
    }

    with isolated_chroma():
        graph = MovieRagGraph()
        external_calls = ExternalCallCounter()
        judge = make_llm_judge(counter=external_calls)
        client = Client()
        dataset = sync_dataset(client, cases, dataset_name=settings.dataset_name)
        with count_external_calls(graph, external_calls):
            result = evaluate(
                make_target(graph),
                data=dataset.name,
                evaluators=[developer_rules, judge],
                summary_evaluators=[developer_summary],
                experiment_prefix=experiment_prefix,
                description="D8 고정 12문항 단일 실행: 개발자 규칙 + GPT-5.6 Terra judge",
                metadata=metadata,
                max_concurrency=0,
                num_repetitions=1,
                client=client,
            )

    print(f"[eval] dataset={dataset.name}")
    print(f"[eval] dataset_sha256={checksum}")
    print(f"[eval] evaluator_sha256={evaluator_checksum}")
    print(f"[eval] commit={commit_sha}")
    print(f"[eval] experiment={result.experiment_name}")
    print(
        "[eval] external_calls="
        + json.dumps(external_calls.as_dict(), ensure_ascii=False, sort_keys=True)
    )


if __name__ == "__main__":
    main()
