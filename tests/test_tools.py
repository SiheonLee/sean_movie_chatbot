"""도구와 출처 변환기 테스트. TMDB는 전부 모킹한다(네트워크 없음)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from rag import tools
from rag.sources import doc_to_source, tmdb_to_source

GENRE_IDS = {28: "액션", 878: "SF", 18: "드라마"}

LIST_MOVIE = {
    "id": 1,
    "title": "기생충",
    "release_date": "2019-05-30",
    "genre_ids": [18, 28],
    "vote_average": 8.514,
    "vote_count": 20989,
    "poster_path": "/p.jpg",
    "overview": "가난한 가족 이야기",
}

DETAIL_MOVIE = {
    "id": 1,
    "title": "기생충",
    "release_date": "2019-05-30",
    "genres": [{"name": "드라마"}, {"name": "액션"}],
    "vote_average": 8.533,
    "vote_count": 20989,
    "runtime": 132,
    "poster_path": "/p.jpg",
    "origin_country": ["KR"],
    "overview": "가난한 가족 이야기",
    "credits": {
        "crew": [{"job": "Director", "name": "봉준호"}, {"job": "Writer", "name": "한진원"}],
        "cast": [{"name": "송강호"}, {"name": "이선균"}],
    },
    "watch/providers": {"results": {"KR": {"flatrate": [{"provider_name": "wavve"}]}}},
}


def call(tool, **kwargs):
    result = tool.invoke(
        {"type": "tool_call", "name": tool.name, "args": kwargs, "id": "t"}
    )
    return result.content, result.artifact


class SourceConversionTests(unittest.TestCase):
    def test_list_shape_has_no_director_or_country(self):
        with patch("rag.tmdb.genre_id_to_name", return_value=GENRE_IDS):
            card = tmdb_to_source(LIST_MOVIE)
        self.assertEqual(card["director"], "")
        self.assertEqual(card["country"], "")
        self.assertEqual(card["genres"], "드라마, 액션")

    def test_detail_shape_fills_director_and_country(self):
        card = tmdb_to_source(DETAIL_MOVIE, detail_shape=True)
        self.assertEqual(card["director"], "봉준호")
        self.assertEqual(card["country"], "한국")

    def test_rating_rounded_so_endpoints_agree(self):
        """discover 8.514 / detail 8.533 → 둘 다 8.5로 보여야 한다."""
        with patch("rag.tmdb.genre_id_to_name", return_value=GENRE_IDS):
            listed = tmdb_to_source(LIST_MOVIE)
        detailed = tmdb_to_source(DETAIL_MOVIE, detail_shape=True)
        self.assertEqual(listed["vote_average"], detailed["vote_average"])
        self.assertEqual(listed["vote_average"], 8.5)

    def test_both_converters_produce_identical_keys(self):
        """키가 어긋나면 어느 도구가 답했느냐에 따라 UI가 깨진다."""
        doc = Document(
            page_content="기생충 (2019)",
            metadata={
                "title": "기생충", "year": 2019, "director": "봉준호",
                "genres": "|드라마|", "country": "KR", "vote_average": 8.5,
                "poster_path": "/p.jpg",
            },
        )
        self.assertEqual(
            set(doc_to_source(doc)), set(tmdb_to_source(DETAIL_MOVIE, detail_shape=True))
        )

    def test_missing_poster_becomes_empty_string(self):
        with patch("rag.tmdb.genre_id_to_name", return_value=GENRE_IDS):
            card = tmdb_to_source({**LIST_MOVIE, "poster_path": None})
        self.assertEqual(card["poster_path"], "")

    def test_missing_release_date_yields_zero_year(self):
        with patch("rag.tmdb.genre_id_to_name", return_value=GENRE_IDS):
            card = tmdb_to_source({**LIST_MOVIE, "release_date": ""})
        self.assertEqual(card["year"], 0)


class SearchMoviesTests(unittest.TestCase):
    def test_unscoped_rating_sort_asks_to_narrow(self):
        """범위 없는 최상급은 답이 임계값에 따라 뒤집히므로 되물어야 한다."""
        with patch("rag.tools._scope_hint", return_value=""):
            content, artifact = call(tools.search_movies, sort_by="rating_desc")
        self.assertIn("범위", content)
        self.assertEqual(artifact, [])

    def test_scoped_rating_sort_proceeds(self):
        with (
            patch("rag.tmdb.genre_name_to_id", return_value={"액션": 28}),
            patch("rag.tmdb.genre_id_to_name", return_value=GENRE_IDS),
            patch("rag.tmdb.discover", return_value={"results": [LIST_MOVIE]}) as disc,
        ):
            content, artifact = call(
                tools.search_movies, genre="액션", sort_by="rating_desc"
            )
        self.assertIn("기생충", content)
        self.assertEqual(len(artifact), 1)
        self.assertEqual(disc.call_args.kwargs["sort_by"], "vote_average.desc")

    def test_vote_count_floor_always_applied(self):
        with (
            patch("rag.tmdb.genre_id_to_name", return_value=GENRE_IDS),
            patch("rag.tmdb.discover", return_value={"results": [LIST_MOVIE]}) as disc,
        ):
            call(tools.search_movies, year_from=2020)
        self.assertEqual(disc.call_args.kwargs["vote_count.gte"], tools._MIN_VOTE_COUNT)

    def test_unknown_genre_lists_supported_ones(self):
        with patch("rag.tmdb.genre_name_to_id", return_value={"액션": 28, "SF": 878}):
            content, artifact = call(tools.search_movies, genre="없는장르")
        self.assertIn("액션", content)
        self.assertEqual(artifact, [])

    def test_unknown_provider_says_unconfirmed_not_unsupported(self):
        with patch("rag.tmdb.resolve_provider_id", return_value=None):
            content, _ = call(tools.search_movies, watch_provider="쿠팡플레이")
        self.assertIn("확인되지 않습니다", content)

    def test_person_not_found(self):
        with patch("rag.tmdb.find_person_id", return_value=None):
            content, artifact = call(tools.search_movies, person="없는사람")
        self.assertIn("찾지 못했습니다", content)
        self.assertEqual(artifact, [])

    def test_count_only_returns_total_without_sources(self):
        with patch("rag.tmdb.discover", return_value={"total_results": 266}):
            content, artifact = call(tools.search_movies, year_from=2020, count_only=True)
        self.assertIn("266편", content)
        self.assertEqual(artifact, [])

    def test_empty_result_suggests_relaxing(self):
        """LLM이 조건을 풀어 재호출하도록 유도하는 문구가 있어야 한다."""
        with patch("rag.tmdb.discover", return_value={"results": []}):
            content, artifact = call(tools.search_movies, year_from=2020)
        self.assertIn("완화", content)
        self.assertEqual(artifact, [])

    def test_status_uses_dedicated_endpoint(self):
        with (
            patch("rag.tmdb.genre_id_to_name", return_value=GENRE_IDS),
            patch("rag.tmdb.list_endpoint", return_value={"results": [LIST_MOVIE]}) as ep,
        ):
            call(tools.search_movies, status="now_playing")
        self.assertEqual(ep.call_args.args[0], "/movie/now_playing")

    def test_tmdb_error_becomes_user_message(self):
        with patch("rag.tmdb.discover", side_effect=tools.tmdb.TmdbError("503")):
            content, artifact = call(tools.search_movies, year_from=2020)
        self.assertIn("가져오지 못했습니다", content)
        self.assertEqual(artifact, [])


def vibe_doc(title: str, *, genres: str = "|드라마|", **meta) -> Document:
    base = {
        "title": title, "year": 2019, "director": "봉준호", "genres": genres,
        "country": "KR", "vote_average": 8.5, "poster_path": "/p.jpg",
        "violence": 3, "sadness": 3, "tension": 3, "complexity": 3, "pacing": "보통",
    }
    return Document(page_content=f"{title} (2019)\n분위기: 묵직하다", metadata=base | meta)


class SearchByVibeTests(unittest.TestCase):
    def setUp(self):
        tools._vectorstore.cache_clear()
        self.addCleanup(tools._vectorstore.cache_clear)

    def patched_store(self, docs):
        store = unittest.mock.Mock()
        store.similarity_search.return_value = docs
        return patch("rag.tools._vectorstore", return_value=store), store

    def test_numeric_filters_go_to_chroma_where(self):
        ctx, store = self.patched_store([vibe_doc("기생충")])
        with ctx:
            call(tools.search_by_vibe, vibe="긴장감", max_violence=2, min_tension=4)
        where = store.similarity_search.call_args.kwargs["filter"]
        self.assertEqual(
            where,
            {"$and": [{"violence": {"$lte": 2}}, {"tension": {"$gte": 4}}]},
        )

    def test_single_filter_is_not_wrapped_in_and(self):
        ctx, store = self.patched_store([vibe_doc("기생충")])
        with ctx:
            call(tools.search_by_vibe, vibe="잔잔한", max_violence=2)
        self.assertEqual(
            store.similarity_search.call_args.kwargs["filter"], {"violence": {"$lte": 2}}
        )

    def test_no_filters_passes_none(self):
        ctx, store = self.patched_store([vibe_doc("기생충")])
        with ctx:
            call(tools.search_by_vibe, vibe="잔잔한")
        self.assertIsNone(store.similarity_search.call_args.kwargs["filter"])

    def test_genre_never_reaches_chroma_where(self):
        """파이프 문자열은 Chroma에서 조용히 빈 결과를 준다 — 반드시 파이썬 후처리."""
        ctx, store = self.patched_store([vibe_doc("기생충", genres="|드라마|")])
        with ctx:
            call(tools.search_by_vibe, vibe="잔잔한", genre="드라마")
        self.assertIsNone(store.similarity_search.call_args.kwargs["filter"])

    def test_genre_filtered_in_python_with_pipe_boundaries(self):
        """'액션'이 '액션코미디'에 걸리면 안 된다."""
        docs = [vibe_doc("액션코미디물", genres="|액션코미디|"),
                vibe_doc("진짜액션", genres="|액션|SF|")]
        ctx, _ = self.patched_store(docs)
        with ctx:
            content, artifact = call(tools.search_by_vibe, vibe="통쾌한", genre="액션")
        self.assertEqual([s["title"] for s in artifact], ["진짜액션"])

    def test_genre_filter_widens_fetch_k(self):
        ctx, store = self.patched_store([vibe_doc("기생충")])
        with ctx:
            call(tools.search_by_vibe, vibe="잔잔한", genre="드라마", limit=5)
        self.assertEqual(store.similarity_search.call_args.kwargs["k"],
                         tools._TAG_FILTER_FETCH_K)

    def test_exclude_titles_removes_already_recommended(self):
        docs = [vibe_doc("인셉션"), vibe_doc("기생충")]
        ctx, _ = self.patched_store(docs)
        with ctx:
            _, artifact = call(tools.search_by_vibe, vibe="몽환적인",
                               exclude_titles=["인셉션"])
        self.assertEqual([s["title"] for s in artifact], ["기생충"])

    def test_limit_is_applied_after_filtering(self):
        docs = [vibe_doc(f"영화{i}") for i in range(10)]
        ctx, _ = self.patched_store(docs)
        with ctx:
            _, artifact = call(tools.search_by_vibe, vibe="잔잔한", limit=3)
        self.assertEqual(len(artifact), 3)

    def test_empty_result_suggests_relaxing(self):
        ctx, _ = self.patched_store([])
        with ctx:
            content, artifact = call(tools.search_by_vibe, vibe="잔잔한", max_violence=1)
        self.assertIn("완화", content)
        self.assertEqual(artifact, [])

    def test_missing_index_returns_message_not_crash(self):
        with patch("rag.tools._vectorstore", side_effect=RuntimeError("색인 없음")):
            content, artifact = call(tools.search_by_vibe, vibe="잔잔한")
        self.assertIn("사용할 수 없습니다", content)
        self.assertEqual(artifact, [])


class WebSearchTests(unittest.TestCase):
    def setUp(self):
        tools._web_search_client.cache_clear()
        self.addCleanup(tools._web_search_client.cache_clear)

    def patched_client(self, response):
        client = unittest.mock.Mock()
        client.invoke.return_value = response
        return patch("rag.tools._web_search_client", return_value=client)

    def test_formats_results_with_source_urls(self):
        response = {"results": [
            {"title": "기생충 평가", "content": "평단 호평", "url": "https://x/1"},
            {"title": "아카데미 4관왕", "content": "작품상 수상", "url": "https://x/2"},
        ]}
        with self.patched_client(response):
            content = tools.web_search.invoke({"query": "기생충 평단"})
        self.assertIn("기생충 평가", content)
        self.assertIn("https://x/1", content)
        self.assertIn("아카데미 4관왕", content)

    def test_produces_no_source_cards(self):
        """웹 결과는 영화가 아니라 기사다. 출처 카드 스키마에 넣으면 빈 카드가 된다."""
        response = {"results": [{"title": "t", "content": "c", "url": "u"}]}
        with self.patched_client(response):
            result = tools.web_search.invoke(
                {"type": "tool_call", "name": "web_search",
                 "args": {"query": "q"}, "id": "t"}
            )
        self.assertIsNone(getattr(result, "artifact", None))

    def test_long_content_is_truncated(self):
        response = {"results": [
            {"title": "t", "content": "가" * 3000, "url": "u"}
        ]}
        with self.patched_client(response):
            content = tools.web_search.invoke({"query": "q"})
        self.assertLess(len(content), 3000)

    def test_empty_results_reported(self):
        with self.patched_client({"results": []}):
            content = tools.web_search.invoke({"query": "없는영화"})
        self.assertIn("결과가 없습니다", content)

    def test_missing_api_key_returns_message_not_crash(self):
        """키가 없어도 나머지 도구는 정상 동작해야 한다."""
        with patch("rag.tools._web_search_client",
                   side_effect=RuntimeError("TAVILY_API_KEY가 설정되지 않았습니다.")):
            content = tools.web_search.invoke({"query": "q"})
        self.assertIn("사용할 수 없습니다", content)
        self.assertIn("TAVILY_API_KEY", content)

    def test_service_failure_is_absorbed(self):
        client = unittest.mock.Mock()
        client.invoke.side_effect = ConnectionError("timeout")
        with patch("rag.tools._web_search_client", return_value=client):
            content = tools.web_search.invoke({"query": "q"})
        self.assertIn("실패했습니다", content)


class GetMovieDetailsTests(unittest.TestCase):
    def test_returns_director_cast_and_provider(self):
        with (
            patch("rag.tmdb.search_by_title", return_value={"results": [{"id": 1}]}),
            patch("rag.tmdb.movie_detail", return_value=DETAIL_MOVIE),
        ):
            content, artifact = call(tools.get_movie_details, title="기생충")
        self.assertIn("봉준호", content)
        self.assertIn("송강호", content)
        self.assertIn("wavve", content)
        self.assertEqual(artifact[0]["director"], "봉준호")

    def test_missing_provider_says_unconfirmed(self):
        detail = {**DETAIL_MOVIE, "watch/providers": {"results": {}}}
        with (
            patch("rag.tmdb.search_by_title", return_value={"results": [{"id": 1}]}),
            patch("rag.tmdb.movie_detail", return_value=detail),
        ):
            content, _ = call(tools.get_movie_details, title="기생충")
        self.assertIn("확인되지 않음", content)
        self.assertNotIn("제공하지 않", content)

    def test_title_not_found(self):
        with patch("rag.tmdb.search_by_title", return_value={"results": []}):
            content, artifact = call(tools.get_movie_details, title="없는영화")
        self.assertIn("찾지 못했습니다", content)
        self.assertEqual(artifact, [])


if __name__ == "__main__":
    unittest.main()
