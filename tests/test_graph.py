from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from rag.config import settings
from rag.graph import (
    DocumentGrade,
    DocumentGrades,
    GroundedAnswer,
    MovieRagGraph,
    answer_aggregate_question,
    classify_question,
    infer_grounded_answer_from_titles,
    parse_document_grades,
    parse_grounded_answer,
)


def movie_doc(title: str, director: str = "", year: int = 2020) -> Document:
    return Document(
        page_content=f"제목: {title} ({year})\n감독: {director}",
        metadata={
            "title": title,
            "director": director,
            "year": year,
            "genres": "|드라마|",
            "country": "KR",
            "vote_average": 8.0,
        },
    )


class FakeLLM:
    def __init__(self, content: str):
        self.content = content
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return AIMessage(content=self.content)


class FakeRunnable:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = result
        self.error = error

    def invoke(self, prompt):
        if self.error:
            raise self.error
        return self.result


class FakeVectorStore:
    def __init__(self, results):
        self.results = results
        self.query = None
        self.kwargs = None

    def similarity_search_with_score(self, query, **kwargs):
        self.query = query
        self.kwargs = kwargs
        return self.results


class StructuredOutputTests(unittest.TestCase):
    def test_anthropic_uses_native_json_schema_method(self):
        graph = MovieRagGraph.__new__(MovieRagGraph)
        graph.llm = Mock()

        with patch.object(settings, "llm_provider", "anthropic"):
            graph._structured_model(DocumentGrades)

        graph.llm.with_structured_output.assert_called_once_with(
            DocumentGrades, method="json_schema"
        )


class GraphRoutingTests(unittest.TestCase):
    def test_classifies_conversation_aggregate_and_fact(self):
        self.assertEqual(classify_question("내가 이전에 말한 영화가 뭐야?"), "conversation")
        self.assertEqual(classify_question("평점이 가장 높은 영화는?"), "aggregate")
        self.assertEqual(classify_question("그 감독의 다른 영화도 알려줘"), "fact")

    def test_contextualizes_followup_with_history(self):
        graph = MovieRagGraph.__new__(MovieRagGraph)
        graph.llm = FakeLLM("크리스토퍼 놀란 감독의 다른 영화")
        graph.movies = [{"title": "인셉션"}]
        state = {
            "original_question": "그 감독의 다른 영화도 알려줘",
            "messages": [
                HumanMessage(content="인셉션 감독 누구야?"),
                AIMessage(content="인셉션 감독은 크리스토퍼 놀란입니다."),
                HumanMessage(content="그 감독의 다른 영화도 알려줘"),
            ],
        }

        result = graph._contextualize_query(state)

        self.assertEqual(
            result["search_query"], "크리스토퍼 놀란 감독의 다른 영화 (인셉션 제외)"
        )
        self.assertIn("현재 질문", graph.llm.last_prompt[-1].content)
        self.assertIn("제외한", graph.llm.last_prompt[0].content)


class RetrievalAndGradingTests(unittest.TestCase):
    def test_retrieve_discards_documents_above_distance_threshold(self):
        close_doc = movie_doc("기생충", "봉준호", 2019)
        far_doc = movie_doc("인셉션", "크리스토퍼 놀란", 2010)
        graph = MovieRagGraph.__new__(MovieRagGraph)
        graph.vectorstore = FakeVectorStore([(close_doc, 0.6), (far_doc, 1.2)])

        with (
            patch.object(settings, "retrieval_max_distance", 1.0),
            patch.object(settings, "retrieval_fetch_k", 8),
            patch.object(settings, "top_k", 4),
        ):
            result = graph._retrieve({"search_query": "기생충 감독", "filters": {}})

        self.assertEqual(result["documents"], [close_doc])
        self.assertEqual(result["document_scores"], [0.6])
        self.assertEqual(graph.vectorstore.query, "기생충 감독")

    def test_grades_each_document_and_keeps_only_yes(self):
        parasite = movie_doc("기생충", "봉준호", 2019)
        inception = movie_doc("인셉션", "크리스토퍼 놀란", 2010)
        graph = MovieRagGraph.__new__(MovieRagGraph)
        graph.document_grader = FakeRunnable(
            DocumentGrades(
                grades=[
                    DocumentGrade(document_id=0, relevant="yes"),
                    DocumentGrade(document_id=1, relevant="no"),
                ]
            )
        )
        state = {
            "original_question": "가난한 가족이 부잣집에 취업하는 영화 감독은?",
            "search_query": "가난한 가족이 부잣집에 취업하는 영화 감독",
            "documents": [parasite, inception],
            "document_scores": [0.6, 0.8],
        }

        with patch.object(settings, "graph_grade_with_llm", True):
            result = graph._grade_documents(state)

        self.assertTrue(result["relevant"])
        self.assertEqual(result["documents"], [parasite])
        self.assertEqual(result["document_scores"], [0.6])

    def test_exact_director_match_bypasses_unstable_llm_grader(self):
        interstellar = movie_doc("인터스텔라", "크리스토퍼 놀란", 2014)
        unrelated = movie_doc("기생충", "봉준호", 2019)
        graph = MovieRagGraph.__new__(MovieRagGraph)
        graph.document_grader = FakeRunnable(error=RuntimeError("must not be called"))

        result = graph._grade_documents(
            {
                "search_query": "크리스토퍼 놀란 감독의 다른 영화",
                "documents": [interstellar, unrelated],
                "document_scores": [0.7, 0.8],
            }
        )

        self.assertTrue(result["relevant"])
        self.assertEqual(result["documents"], [interstellar])

    def test_grader_error_fails_closed(self):
        graph = MovieRagGraph.__new__(MovieRagGraph)
        graph.document_grader = FakeRunnable(error=RuntimeError("invalid output"))
        graph.llm = FakeLLM("문서 0: no")
        state = {
            "original_question": "타이타닉 감독은?",
            "search_query": "타이타닉 감독",
            "documents": [movie_doc("인셉션")],
            "document_scores": [0.9],
        }

        with patch.object(settings, "graph_grade_with_llm", True):
            result = graph._grade_documents(state)

        self.assertFalse(result["relevant"])
        self.assertEqual(result["documents"], [])
        self.assertIsNone(graph.document_grader)

    def test_generate_returns_only_structured_used_sources(self):
        parasite = movie_doc("기생충", "봉준호", 2019)
        oldboy = movie_doc("올드보이", "박찬욱", 2003)
        graph = MovieRagGraph.__new__(MovieRagGraph)
        graph.grounded_generator = FakeRunnable(
            GroundedAnswer(
                answer="기생충(2019)의 감독은 봉준호입니다.",
                source_ids=[0, 0, 99],
            )
        )
        state = {
            "search_query": "기생충 감독",
            "documents": [parasite, oldboy],
            "messages": [HumanMessage(content="기생충 감독은?")],
        }

        result = graph._generate_rag(state)

        self.assertEqual([source["title"] for source in result["sources"]], ["기생충"])

    def test_refusal_never_returns_sources(self):
        graph = MovieRagGraph.__new__(MovieRagGraph)
        graph.grounded_generator = FakeRunnable(
            GroundedAnswer(answer="주어진 정보에서는 확인할 수 없습니다.", source_ids=[0])
        )
        result = graph._generate_rag(
            {
                "search_query": "없는 영화",
                "documents": [movie_doc("인셉션")],
                "messages": [HumanMessage(content="없는 영화 감독은?")],
            }
        )

        self.assertEqual(result["sources"], [])


class AggregateTests(unittest.TestCase):
    def test_highest_rating_uses_full_movie_list(self):
        movies = [
            {"title": "A", "year": 2000, "vote_average": 8.2, "genres": [], "country": "US"},
            {"title": "B", "year": 2020, "vote_average": 9.0, "genres": [], "country": "KR"},
            {"title": "C", "year": 2021, "vote_average": 7.5, "genres": [], "country": "KR"},
        ]

        answer, selected = answer_aggregate_question(
            "이 데이터베이스에서 평점이 가장 높은 영화는?", movies, top_k=4
        )

        self.assertIn("B", answer)
        self.assertEqual([movie["title"] for movie in selected], ["B"])

    def test_count_honors_country_filter(self):
        movies = [
            {"title": "A", "year": 2000, "vote_average": 8.2, "genres": [], "country": "US"},
            {"title": "B", "year": 2020, "vote_average": 9.0, "genres": [], "country": "KR"},
            {"title": "C", "year": 2021, "vote_average": 7.5, "genres": [], "country": "KR"},
        ]

        answer, selected = answer_aggregate_question("한국 영화는 총 몇 편이야?", movies, 4)

        self.assertIn("2편", answer)
        self.assertEqual(selected, [])


class StrictParserTests(unittest.TestCase):
    def test_korean_negative_sentence_is_not_misread_as_relevant(self):
        with self.assertRaises(ValueError):
            parse_document_grades("관련이 없습니다", document_count=1)

    def test_document_grade_requires_all_document_ids(self):
        with self.assertRaises(ValueError):
            parse_document_grades("문서 0: yes", document_count=2)

    def test_grounded_answer_extracts_only_declared_sources(self):
        result = parse_grounded_answer(
            "기생충(2019)의 감독은 봉준호입니다.\nSOURCE_IDS: 0", document_count=2
        )

        self.assertEqual(result.answer, "기생충(2019)의 감독은 봉준호입니다.")
        self.assertEqual(result.source_ids, [0])

    def test_title_fallback_returns_only_titles_named_in_answer(self):
        docs = [movie_doc("오디세이"), movie_doc("인셉션"), movie_doc("인터스텔라")]

        result = infer_grounded_answer_from_titles(
            "다른 작품으로 오디세이(2026)와 인터스텔라(2014)가 있습니다.", docs
        )

        self.assertEqual(result.source_ids, [0, 2])


if __name__ == "__main__":
    unittest.main()
