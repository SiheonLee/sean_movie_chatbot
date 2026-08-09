"""D8 평가 도구 자체의 고정 dataset·규칙 점수·judge·동기화 테스트."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rag.config import settings
from scripts.evaluate import (
    D8_DATASET_NAME,
    D8_DATASET_VERSION,
    ExternalCallCounter,
    JudgeVerdict,
    _example_payload,
    _matches_predicate,
    aggregate_developer_scores,
    count_external_calls,
    dataset_checksum,
    isolated_chroma,
    load_cases,
    make_llm_judge,
    make_target,
    score_developer,
    sync_dataset,
    validate_official_config,
)


def movie(title: str, year: int, movie_id: int) -> dict:
    return {
        "movie_id": movie_id,
        "title": title,
        "year": year,
        "director": "감독",
        "cast": "배우",
        "genres": "액션",
        "country": "한국",
        "vote_average": 8.0,
        "poster_path": "/p.jpg",
        "snippet": "요약",
    }


def case(case_id: str) -> dict:
    return next(item for item in load_cases() if item["id"] == case_id)


def example_for(item: dict) -> SimpleNamespace:
    payload = _example_payload(item)
    return SimpleNamespace(inputs=payload["inputs"], outputs=payload["outputs"])


def empty_output(**overrides) -> dict:
    base = {
        "answer": "",
        "sources": [],
        "web_sources": [],
        "attributions": [],
        "tool_calls": [],
        "tool_results": [],
        "error": None,
    }
    return base | overrides


class FixedDatasetTests(unittest.TestCase):
    def test_dataset_has_fixed_groups_and_denominators(self):
        cases = load_cases()

        self.assertEqual(len(cases), 12)
        self.assertEqual(
            {group: sum(item["group"] == group for item in cases) for group in ("tmdb", "local", "web")},
            {"tmdb": 4, "local": 4, "web": 4},
        )
        self.assertEqual(
            sum(len(item["expectation"]["required_args"]) for item in cases), 33
        )
        self.assertEqual(
            sum(item["expectation"]["movie_cards"] == "required" for item in cases), 7
        )
        self.assertEqual(
            sum(item["expectation"]["web_sources"] == "required" for item in cases), 4
        )
        self.assertTrue(all("answer" not in item for item in cases))

    def test_dataset_checksum_is_stable_for_unchanged_bytes(self):
        self.assertEqual(dataset_checksum(), dataset_checksum())
        self.assertEqual(len(dataset_checksum()), 64)

    def test_fixed_answer_field_is_rejected(self):
        rows = load_cases()
        rows[0]["answer"] = "낡은 고정 정답"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text(
                "\n".join(__import__("json").dumps(row, ensure_ascii=False) for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "고정 answer"):
                load_cases(path)


class ArgumentPredicateTests(unittest.TestCase):
    def test_alias_and_numeric_predicates(self):
        self.assertTrue(
            _matches_predicate(
                {"watch_provider": "netflix"},
                {"name": "watch_provider", "op": "one_of", "value": ["넷플릭스", "Netflix"]},
            )
        )
        self.assertTrue(
            _matches_predicate(
                {"min_rating": 8.1},
                {"name": "min_rating", "op": "gte", "value": 8.0},
            )
        )

    def test_web_query_uses_semantic_token_groups_not_exact_string(self):
        predicate = case("web-01")["expectation"]["required_args"][0]
        self.assertTrue(
            _matches_predicate(
                {"query": "2026 오스카 작품상 공식 수상 결과"}, predicate
            )
        )
        self.assertFalse(
            _matches_predicate({"query": "최근 인기 영화"}, predicate)
        )


class DeveloperScoringTests(unittest.TestCase):
    def test_perfect_movie_case_preserves_fraction_denominators(self):
        item = case("tmdb-01")
        source = movie("액션 A", 2020, 1)
        args = {"genre": "액션", "watch_provider": "넷플릭스", "limit": 3}
        outputs = empty_output(
            answer="**액션 A**(2020)를 추천합니다.",
            sources=[source],
            attributions=["tmdb", "justwatch"],
            tool_calls=[{"name": "search_movies", "args": args}],
            tool_results=[
                {
                    "name": "search_movies",
                    "args": args,
                    "success": True,
                    "content": "액션 A (2020)",
                    "sources": [source],
                    "web_sources": [],
                }
            ],
        )

        scores = score_developer(outputs, item["expectation"])

        self.assertEqual(
            (scores["required_args"].numerator, scores["required_args"].denominator),
            (3, 3),
        )
        self.assertTrue(all(metric.numerator == metric.denominator for metric in scores.values()))

    def test_expected_route_can_pass_while_extra_call_fails_discipline(self):
        expectation = case("tmdb-01")["expectation"]
        outputs = empty_output(
            tool_calls=[
                {"name": "search_movies", "args": {}},
                {"name": "web_search", "args": {"query": "q"}},
            ]
        )

        scores = score_developer(outputs, expectation)

        self.assertEqual(scores["routing"].numerator, 1)
        self.assertEqual(scores["call_discipline"].numerator, 0)

    def test_web_sources_are_separate_and_derived_from_successful_evidence(self):
        item = case("web-01")
        web = {"title": "공식 결과", "url": "https://example.com/oscar"}
        args = {"query": "2026 아카데미 작품상"}
        outputs = empty_output(
            answer="검색 결과 작품상 수상작은 A입니다.",
            web_sources=[web],
            attributions=["web"],
            tool_calls=[{"name": "web_search", "args": args}],
            tool_results=[
                {
                    "name": "web_search",
                    "args": args,
                    "success": True,
                    "content": "작품상 수상 결과",
                    "sources": [],
                    "web_sources": [web],
                }
            ],
        )

        scores = score_developer(outputs, item["expectation"])

        self.assertEqual(scores["web_source_separation"].numerator, 1)
        self.assertEqual(scores["structured_source_shape"].numerator, 1)

    def test_summary_adds_predicate_denominators_instead_of_averaging_cases(self):
        items = [case("tmdb-01"), case("tmdb-02")]
        runs = [SimpleNamespace(outputs=empty_output()) for _ in items]
        examples = [example_for(item) for item in items]

        totals = aggregate_developer_scores(runs, examples)

        self.assertEqual(totals["routing"].denominator, 2)
        self.assertEqual(totals["required_args"].denominator, 7)
        self.assertEqual(totals["card_provenance"].denominator, 1)


class TargetTests(unittest.TestCase):
    def test_target_returns_trace_fields_once(self):
        result = empty_output(answer="답변", tool_calls=[{"name": "x", "args": {}}])
        result.pop("error")
        graph = SimpleNamespace(trace=lambda question: result)

        output = make_target(graph)({"question": "질문"})

        self.assertEqual(output["answer"], "답변")
        self.assertIsNone(output["error"])

    def test_target_turns_exception_into_a_scored_failure(self):
        def explode(_question):
            raise RuntimeError("secret detail")

        output = make_target(SimpleNamespace(trace=explode))({"question": "질문"})

        self.assertEqual(output["error"], "RuntimeError")
        self.assertNotIn("secret", str(output))
        self.assertEqual(output["tool_results"], [])


class ExternalCallCounterTests(unittest.TestCase):
    def test_counts_logical_services_and_restores_wrappers(self):
        from rag import providers, tmdb, tools

        llm = SimpleNamespace(invoke=lambda payload: "target")
        web = SimpleNamespace(invoke=lambda payload: {"results": []})
        graph = SimpleNamespace(llm=llm)
        counter = ExternalCallCounter()
        embeddings = providers.QuotaAwareGoogleEmbeddings(
            SimpleNamespace(), batch_size=1, requests_per_minute=1, max_retries=0
        )

        with (
            patch.object(tmdb, "_get", return_value={}),
            patch.object(tools, "_web_search_client", return_value=web),
        ):
            original_tmdb_get = tmdb._get
            original_web_factory = tools._web_search_client
            with count_external_calls(graph, counter):
                self.assertEqual(graph.llm.invoke({}), "target")
                tmdb._get("/movie/1")
                tools._web_search_client().invoke({"query": "q"})
                self.assertEqual(embeddings._with_quota_retry(lambda: "embedded"), "embedded")
            self.assertIs(graph.llm, llm)
            self.assertIs(tmdb._get, original_tmdb_get)
            self.assertIs(tools._web_search_client, original_web_factory)

        self.assertEqual(counter.target_llm, 1)
        self.assertEqual(counter.tmdb_http, 1)
        self.assertEqual(counter.tavily_search, 1)
        self.assertEqual(counter.google_embedding_attempts, 1)
        self.assertEqual(counter.as_dict()["logical_total_excluding_langsmith"], 4)


class FakeClient:
    def __init__(self, dataset=None, examples=None):
        self.dataset = dataset
        self.examples = list(examples or [])
        self.created_dataset = None
        self.created_examples = None

    def list_datasets(self, *, dataset_name):
        return iter([self.dataset] if self.dataset is not None else [])

    def create_dataset(self, *, dataset_name, description, metadata):
        self.created_dataset = SimpleNamespace(id="dataset-1", name=dataset_name)
        return self.created_dataset

    def create_examples(self, *, dataset_id, examples):
        self.created_examples = (dataset_id, examples)

    def list_examples(self, *, dataset_id):
        return iter(self.examples)


def stored_example(item: dict) -> SimpleNamespace:
    payload = _example_payload(item)
    return SimpleNamespace(**payload)


class DatasetSyncTests(unittest.TestCase):
    def test_creates_a_new_versioned_dataset_in_one_batch(self):
        client = FakeClient()
        cases = load_cases()

        dataset = sync_dataset(client, cases)

        self.assertEqual(dataset.name, D8_DATASET_NAME)
        self.assertEqual(client.created_examples[0], "dataset-1")
        self.assertEqual(len(client.created_examples[1]), 12)

    def test_matching_existing_dataset_is_reused_without_writes(self):
        dataset = SimpleNamespace(id="dataset-1", name=D8_DATASET_NAME)
        cases = load_cases()
        client = FakeClient(dataset, [stored_example(item) for item in cases])

        returned = sync_dataset(client, cases)

        self.assertIs(returned, dataset)
        self.assertIsNone(client.created_examples)

    def test_reference_or_expectation_change_aborts_instead_of_appending(self):
        dataset = SimpleNamespace(id="dataset-1", name=D8_DATASET_NAME)
        cases = load_cases()
        examples = [stored_example(item) for item in cases]
        examples[0].outputs = examples[0].outputs | {"group": "changed"}
        client = FakeClient(dataset, examples)

        with self.assertRaisesRegex(RuntimeError, "tmdb-01"):
            sync_dataset(client, cases)
        self.assertIsNone(client.created_examples)

    def test_legacy_dataset_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "재사용"):
            sync_dataset(FakeClient(), load_cases(), dataset_name="movies-rag-eval")


class JudgeTests(unittest.TestCase):
    def test_terra_judge_emits_five_axes_and_mean(self):
        verdict = JudgeVerdict.model_validate(
            {
                "direct_answer": {"score": 1, "reason": "직접 답함"},
                "evidence_grounding": {"score": 0.5, "reason": "일부 근거"},
                "condition_relevance": {"score": 1, "reason": "조건 관련"},
                "uncertainty_calibration": {"score": 0.5, "reason": "일부 단정"},
                "usefulness_clarity": {"score": 1, "reason": "명확함"},
            }
        )
        chain = SimpleNamespace(invoke=lambda payload: verdict)
        run = SimpleNamespace(outputs=empty_output(answer="답변"))
        counter = ExternalCallCounter()

        feedback = make_llm_judge(chain, counter=counter)(
            run, example_for(case("tmdb-01"))
        )

        self.assertEqual(len(feedback["results"]), 7)
        mean = next(item for item in feedback["results"] if item["key"] == "ai_judge_mean")
        self.assertEqual(mean["score"], 0.8)
        self.assertEqual(mean["metadata"]["judge_model"], settings.judge_model)
        self.assertEqual(counter.judge_llm, 1)

    def test_judge_error_is_recorded_as_zero_not_dropped(self):
        def explode(_payload):
            raise ValueError("invalid")

        feedback = make_llm_judge(SimpleNamespace(invoke=explode))(
            SimpleNamespace(outputs=empty_output()), example_for(case("tmdb-01"))
        )

        valid = next(item for item in feedback["results"] if item["key"] == "ai_judge_valid")
        axes = [item for item in feedback["results"] if item["key"] != "ai_judge_valid"]
        self.assertEqual(valid["score"], 0)
        self.assertTrue(all(item["score"] is None for item in axes))
        self.assertTrue(all(item["metadata"]["error"] == "ValueError" for item in feedback["results"]))


class IsolatedChromaTests(unittest.TestCase):
    def test_copy_is_used_and_original_bytes_are_unchanged(self):
        original_setting = settings.chroma_dir
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            database = source / "chroma.sqlite3"
            database.write_bytes(b"original")

            with isolated_chroma(source) as copied:
                self.assertEqual(settings.chroma_dir, copied)
                self.assertNotEqual(copied, source)
                (copied / "chroma.sqlite3").write_bytes(b"runtime change")

            self.assertEqual(database.read_bytes(), b"original")
            self.assertEqual(settings.chroma_dir, original_setting)


class EvaluationConfigTests(unittest.TestCase):
    def test_judge_is_separate_terra_model_with_small_output_cap(self):
        self.assertEqual(settings.judge_model, "gpt-5.6-terra")
        self.assertEqual(settings.judge_max_tokens, 1024)
        self.assertEqual(D8_DATASET_VERSION, "d8-v1")

    def test_official_config_rejects_the_legacy_dataset(self):
        with patch.object(settings, "dataset_name", "movies-rag-eval"):
            with self.assertRaisesRegex(RuntimeError, "LANGSMITH_DATASET"):
                validate_official_config()

    def test_official_config_accepts_the_frozen_models_and_limits(self):
        frozen = {
            "dataset_name": D8_DATASET_NAME,
            "llm_provider": "openai",
            "openai_model": "gpt-5.6-luna",
            "llm_temperature": 0.0,
            "openai_reasoning_effort": "none",
            "llm_max_tokens": 8192,
            "judge_model": "gpt-5.6-terra",
            "judge_max_tokens": 1024,
        }
        patches = [patch.object(settings, name, value) for name, value in frozen.items()]
        for context in patches:
            context.start()
            self.addCleanup(context.stop)

        validate_official_config()


if __name__ == "__main__":
    unittest.main()
