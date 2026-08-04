"""CineBot: 영화 정보 RAG를 사용하는 Streamlit 채팅 화면."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from dotenv import load_dotenv

from ui.api_client import ApiClientError, RagApiClient, Source

load_dotenv()


API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
SUGGESTED_QUESTIONS = (
    "기생충의 감독과 줄거리를 알려줘",
    "평점이 가장 높은 영화는 무엇이야?",
    "한국 스릴러 영화를 추천해줘",
    "봉준호 감독의 다른 영화도 알려줘",
)

st.set_page_config(
    page_title="CineBot",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded",
)


def load_styles() -> None:
    css = (ROOT_DIR / "ui" / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def initialize_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())


def reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())


@st.cache_data(ttl=10, show_spinner=False)
def check_api_health(api_url: str) -> bool:
    return RagApiClient(api_url).is_healthy()


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### CineBot")
        if st.button(
            "새 대화",
            icon=":material/add_comment:",
            use_container_width=True,
            type="primary",
        ):
            reset_conversation()
            st.rerun()

        healthy = check_api_health(API_URL)
        status_class = "" if healthy else " offline"
        status_text = "API 연결됨" if healthy else "API 연결 안 됨"
        st.markdown(
            (
                '<div class="cine-status">'
                f'<span class="cine-status-dot{status_class}"></span>'
                f"<span>{status_text}</span>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def render_welcome() -> str | None:
    st.markdown(
        """
        <section class="cine-hero">
          <div class="cine-eyebrow">Movie Conversation</div>
          <h1 class="cine-title">CineBot</h1>
          <p class="cine-subtitle">영화가 궁금할 땐, CineBot에게 물어보세요.</p>
        </section>
        <div class="cine-suggestions-label">이런 질문으로 시작해보세요</div>
        """,
        unsafe_allow_html=True,
    )

    selected: str | None = None
    for row_start in range(0, len(SUGGESTED_QUESTIONS), 2):
        columns = st.columns(2)
        for column, question in zip(
            columns, SUGGESTED_QUESTIONS[row_start : row_start + 2]
        ):
            if column.button(
                question,
                key=f"suggestion-{row_start}-{question}",
                use_container_width=True,
            ):
                selected = question
    return selected


def source_label(source: Source) -> str:
    title = source.get("title") or "제목 정보 없음"
    year = source.get("year") or "-"
    rating = float(source.get("vote_average") or 0.0)
    return f"🎬 {title} ({year}) · ⭐ {rating:.1f}"


# TMDB 이미지 CDN. 경로만 저장하고 크기는 여기서 정한다. w185는 카드용.
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w185"


def poster_url(source: Source) -> str | None:
    """포스터 경로를 표시용 URL로. 경로가 없는 영화도 있으므로 None을 허용한다."""
    path = (source.get("poster_path") or "").strip()
    return f"{POSTER_BASE_URL}{path}" if path else None


def render_sources(sources: list[Source]) -> None:
    if not sources:
        return

    st.caption(f"답변에 사용된 영화 정보 {len(sources)}개")
    for source in sources:
        with st.expander(source_label(source)):
            director = source.get("director") or "정보 없음"
            genres = source.get("genres") or "정보 없음"
            country = source.get("country") or "정보 없음"
            snippet = source.get("snippet") or "표시할 근거 문장이 없습니다."

            url = poster_url(source)
            if url:
                poster_column, body = st.columns([1, 3])
                poster_column.image(url, use_container_width=True)
            else:
                # 포스터가 없으면 텍스트가 좁은 칸에 갇히지 않도록 전체 폭을 쓴다.
                body = st.container()

            with body:
                st.caption(f"감독 · {director}")
                # 무드 검색 결과에는 출연진이 없다. 없을 때 "정보 없음"을 띄우면
                # 빈 줄만 늘어나므로 아예 감춘다.
                if source.get("cast"):
                    st.caption(f"출연 · {source['cast']}")
                st.caption(f"장르 · {genres}")
                st.caption(f"국가 · {country}")
                st.write(snippet)


def render_message(message: dict[str, Any]) -> None:
    role = message["role"]
    avatar = ":material/person:" if role == "user" else ":material/movie:"
    with st.chat_message(role, avatar=avatar):
        if message.get("error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])
            render_sources(message.get("sources", []))


def ask_cinebot(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=":material/movie:"):
        with st.spinner("CineBot이 영화를 찾고 있어요..."):
            try:
                result = RagApiClient(API_URL).query(
                    question=question,
                    session_id=st.session_state.session_id,
                )
                message = {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                }
            except ApiClientError as exc:
                message = {
                    "role": "assistant",
                    "content": str(exc),
                    "sources": [],
                    "error": True,
                }

    st.session_state.messages.append(message)
    st.rerun()


load_styles()
initialize_session()
render_sidebar()

selected_question = None
if not st.session_state.messages:
    selected_question = render_welcome()
else:
    for chat_message in st.session_state.messages:
        render_message(chat_message)

typed_question = st.chat_input(
    "영화 제목, 감독, 장르 또는 추천 조건을 입력하세요",
    max_chars=500,
)
question = selected_question or typed_question
if question:
    ask_cinebot(question.strip())
