"""CineBot: 영화 정보 RAG를 사용하는 Streamlit 채팅 화면."""

from __future__ import annotations

import hmac
import html
import os
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from ui import history, watchlist
from ui.api_client import ApiClientError, RagApiClient, Source

load_dotenv()


API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")

# 화면을 가리는 최소한의 자물쇠. 비워 두면 잠그지 않는다.
#
# 정식 인증이 아니다. 사용자 계정도, 만료도, 시도 횟수 제한도 없고 모두가 같은
# 값을 쓴다. **그리고 FastAPI 서버는 이 값과 무관하게 열려 있다** — 링크를 아는
# 사람이 API를 직접 부르는 것은 막지 못한다. "아무나 열어보지는 못하게" 정도의
# 용도로만 쓸 것.
PASSCODE = os.getenv("CINEBOT_PASSCODE", "").strip()
SUGGESTED_QUESTIONS = (
    "기생충의 감독과 줄거리를 알려줘",
    "평점이 가장 높은 영화는 무엇이야?",
    "한국 스릴러 영화를 추천해줘",
    "시끄러운 카페에서 보기 좋은 영화",
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
    # 아직 답변하지 않은 질문. 입력을 받은 실행과 답변을 그리는 실행을 나눈다.
    if "pending" not in st.session_state:
        st.session_state.pending = None
    # 값이 바뀔 때만 화면을 맨 아래로 내린다(scroll_to_bottom_script 참고).
    if "scroll_token" not in st.session_state:
        st.session_state.scroll_token = 0
    # 지난 대화를 열어보는 중이면 그 id. 지금 대화를 보고 있으면 None.
    if "viewing" not in st.session_state:
        st.session_state.viewing = None
    # 보고싶은 영화 목록이 바뀔 때마다 오른다. 칩 위젯 키에 섞어 넣어, 목록이
    # 바뀌면 모든 칩이 새로 만들어지며 파일 내용을 다시 읽게 한다.
    if "watchlist_rev" not in st.session_state:
        st.session_state.watchlist_rev = 0
    # 보고싶은 영화 목록을 펼쳐 보는 중인지.
    if "showing_watchlist" not in st.session_state:
        st.session_state.showing_watchlist = False


def reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.pending = None
    st.session_state.viewing = None
    st.session_state.showing_watchlist = False


@st.cache_data(ttl=10, show_spinner=False)
def check_api_health(api_url: str) -> bool:
    return RagApiClient(api_url).is_healthy()


def render_sidebar(busy: bool = False) -> None:
    """`busy`면 답변을 받는 중이라 모든 버튼을 잠근다.

    Streamlit은 위젯이 바뀌면 **실행 중인 스크립트를 끊는다.** 답변을 흘리는
    도중에 사이드바를 누르면 그 자리에서 중단되고, 이미 시작된 LangGraph 실행이
    통째로 사라진다. 그러면 답변도 못 받고 LangSmith에는 끝나지 않는 트레이스가
    남는다(실측). 눌리지 않게 막는 편이 뒷수습보다 낫다.
    """
    with st.sidebar:
        st.markdown("### CineBot")
        if st.button(
            "새 대화",
            icon=":material/add_comment:",
            use_container_width=True,
            type="primary",
            disabled=busy,
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
        render_watchlist_entry(busy)
        render_past_conversations(busy)


def render_watchlist_entry(busy: bool = False) -> None:
    """사이드바의 '보고싶은 영화' 입구. 담은 게 없으면 아무것도 안 보인다."""
    saved = watchlist.saved_movies()
    if not saved:
        return

    st.markdown(
        '<div class="cine-past-label">보고싶은 영화</div>', unsafe_allow_html=True
    )
    showing = st.session_state.showing_watchlist
    if st.button(
        f"{'목록 닫기' if showing else '담아둔 영화'} · {len(saved)}편",
        icon=":material/bookmark:",
        use_container_width=True,
        disabled=busy,
    ):
        st.session_state.showing_watchlist = not showing
        # 지난 대화를 보던 중이라면 그건 닫는다. 두 화면이 겹치면 헷갈린다.
        st.session_state.viewing = None
        st.rerun()


def render_past_conversations(busy: bool = False) -> None:
    """지난 대화 목록. 열어보기만 되고 이어서 물어볼 수는 없다."""
    # 지금 대화는 목록에 넣지 않는다. 사이드바에서 자기 자신을 여는 건 의미가 없다.
    past = [
        summary
        for summary in history.list_conversations()
        if summary.conversation_id != st.session_state.session_id
    ]
    if not past:
        return

    st.markdown('<div class="cine-past-label">지난 대화</div>', unsafe_allow_html=True)
    if st.session_state.viewing is not None and st.button(
        "지금 대화로 돌아가기",
        icon=":material/arrow_back:",
        use_container_width=True,
        disabled=busy,
    ):
        st.session_state.viewing = None
        st.rerun()

    for summary in past:
        opened = summary.conversation_id == st.session_state.viewing
        if st.button(
            summary.title,
            key=f"past-{summary.conversation_id}",
            help=f"{summary.updated_at.replace('T', ' ')} · {summary.turns}번 질문",
            use_container_width=True,
            type="secondary",
            disabled=opened or busy,
        ):
            st.session_state.viewing = summary.conversation_id
            st.rerun()


HELP_EXAMPLES = (
    ("정보", "기생충 감독이 누구야?"),
    ("추천", "한국 스릴러 영화를 추천해줘"),
    ("분위기", "비 오는 날 볼 잔잔한 영화"),
    ("시청처", "기생충 넷플릭스에서 볼 수 있어?"),
    ("평가·화제", "기생충 평단 반응 어땠어?"),
)


def render_help() -> None:
    """오른쪽 아래에 붙는 도움말.

    환영 화면은 첫 질문과 함께 사라져서 사용법을 다시 볼 방법이 없었다.

    st.popover나 st.dialog가 아니라 `<details>`를 쓴다. 둘 다 열고 닫을 때마다
    rerun이 걸리는데, 답변을 스트리밍하는 중에 rerun이 끼면 화면이 흔들린다.
    도움말 내용은 고정이라 브라우저에 맡기면 서버를 건드릴 일이 없다.
    """
    items = "".join(
        f'<li><span class="cine-help-tag">{html.escape(tag)}</span>'
        f"<span>{html.escape(example)}</span></li>"
        for tag, example in HELP_EXAMPLES
    )
    st.markdown(
        '<details class="cine-help">'
        '<summary class="cine-help-toggle" title="도움말">?</summary>'
        '<div class="cine-help-panel">'
        '<div class="cine-help-title">이렇게 물어보세요</div>'
        f'<ul class="cine-help-list">{items}</ul>'
        '<div class="cine-help-note">'
        "앞의 대화를 기억합니다. “그중 첫 번째 영화는?”처럼 이어서 물어도 됩니다."
        "</div>"
        "</div>"
        "</details>",
        unsafe_allow_html=True,
    )


def passcode_matches(code: str) -> bool:
    """입력값이 패스코드와 같은지.

    한 글자씩 비교하면 걸리는 시간으로 자릿수를 추측할 수 있어 상수 시간으로
    비교한다. 반드시 bytes로 넘긴다 — compare_digest에 str을 주면 ASCII만 받고,
    한글이 섞이면 TypeError가 나서 화면이 통째로 죽는다(실측).
    """
    return hmac.compare_digest(code.encode("utf-8"), PASSCODE.encode("utf-8"))


def render_lock() -> None:
    """패스코드 입력 화면.

    폼으로 묶어 제출할 때만 확인한다. 안 그러면 글자를 칠 때마다 재실행되면서
    입력 중인 값이 매번 대조된다.
    """
    st.markdown(
        """
        <section class="cine-hero">
          <div class="cine-eyebrow">Private</div>
          <h1 class="cine-title">CineBot</h1>
          <p class="cine-subtitle">패스코드를 입력하면 들어갈 수 있어요.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    with st.form("cine-lock", border=False):
        code = st.text_input(
            "패스코드",
            type="password",
            label_visibility="collapsed",
            placeholder="패스코드",
        )
        submitted = st.form_submit_button("들어가기", use_container_width=True)

    if not submitted:
        return
    if passcode_matches(code):
        st.session_state.unlocked = True
        st.rerun()
    else:
        st.error("패스코드가 올바르지 않습니다.")


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


# TMDB 이미지 CDN. 경로만 저장하고 크기는 여기서 정한다. w185는 카드용.
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w185"


def poster_url(source: Source) -> str | None:
    """포스터 경로를 표시용 URL로. 경로가 없는 영화도 있으므로 None을 허용한다."""
    path = (source.get("poster_path") or "").strip()
    return f"{POSTER_BASE_URL}{path}" if path else None


def _field_html(label: str, value: str) -> str:
    """카드의 한 줄. 값이 없으면 빈 줄을 남기지 않고 아예 지운다."""
    if not value:
        return ""
    return (
        '<div class="cine-card-field">'
        f'<span class="cine-card-key">{label}</span>'
        f'<span class="cine-card-value">{html.escape(value)}</span>'
        "</div>"
    )


def source_card_html(source: Source) -> str:
    """출처 카드 하나를 HTML로.

    접힌 상태에서는 포스터·제목·연도·평점만 보이고, 눌러야 나머지가 펼쳐진다.
    세로로 긴 카드가 답변보다 화면을 더 차지하던 문제를 이렇게 줄인다.

    `st.expander`가 아니라 `<details>`를 쓴다. expander는 클릭할 때마다 서버
    왕복과 rerun이 일어나서, 답변 하나에 카드가 5장이면 그만큼 느려진다.
    `<details>`는 브라우저가 자체적으로 여닫으므로 왕복이 없다. 대신 rerun이
    일어나면(다음 질문을 하면) 열어둔 카드가 닫힌다 — 지난 턴의 카드라 괜찮다.
    """
    title = html.escape(source.get("title") or "제목 정보 없음")
    year = html.escape(str(source.get("year") or "-"))
    rating = float(source.get("vote_average") or 0.0)

    url = poster_url(source)
    if url:
        poster = f'<img class="cine-card-poster" src="{html.escape(url)}" alt="{title} 포스터" loading="lazy">'
    else:
        poster = '<div class="cine-card-poster cine-card-poster-empty">🎬</div>'

    details = (
        _field_html("감독", source.get("director") or "")
        # 무드 검색 결과에는 출연진이 없다. 없을 때 "정보 없음"을 띄우면 빈 줄만
        # 늘어나므로 아예 감춘다.
        + _field_html("출연", source.get("cast") or "")
        + _field_html("장르", source.get("genres") or "")
        + _field_html("국가", source.get("country") or "")
        # <p>로 두면 Streamlit의 .stMarkdown p 규칙이 폰트 크기를 덮어써서
        # 출연진 줄보다 커진다. div는 그 규칙에 안 걸린다.
        + f'<div class="cine-card-snippet">{html.escape(source.get("snippet") or "")}</div>'
    )

    return (
        '<details class="cine-card">'
        '<summary class="cine-card-head">'
        f"{poster}"
        '<span class="cine-card-caption">'
        f'<span class="cine-card-title">{title}</span>'
        f'<span class="cine-card-meta">{year} · ⭐ {rating:.1f}</span>'
        "</span>"
        "</summary>"
        f'<div class="cine-card-body">{details}</div>'
        "</details>"
    )


def sources_html(sources: list[Source], label: str = "답변에 사용된 영화") -> str:
    """카드 묶음 전체를 HTML 한 덩어리로. 없으면 빈 문자열.

    저장 목록 화면에서도 같은 카드를 쓰므로 머리말은 밖에서 정한다.
    """
    if not sources:
        return ""

    cards = "".join(source_card_html(source) for source in sources)
    return (
        f'<div class="cine-sources-label">{html.escape(label)} {len(sources)}편</div>'
        f'<div class="cine-cards">{cards}</div>'
    )


def message_key(index: int) -> str:
    """말풍선을 구분하는 고유 키.

    키가 없으면 Streamlit은 **순서(인덱스)로만** 요소를 짝짓는다. 그래서 이력이
    늘어 자리가 밀리면, 직전 실행에서 어시스턴트 말풍선이 있던 자리에 이번엔
    사용자 말풍선이 앉는다. 사용자 말풍선은 요소를 하나만 그리므로 어시스턴트가
    쓰던 두 번째 요소(출처 카드)가 지워지지 않고 남는다 — 질문 말풍선에 카드가
    붙어 보이던 원인이다.

    키를 주면 말풍선마다 제 자리를 갖게 되어 이 문제가 생기지 않는다. 스트리밍
    중인 답변에도 **이력에 들어갈 때와 같은 키**를 줘야 화면이 이어진다.
    """
    return f"cine-msg-{index}"


# 어시스턴트 말풍선 안에 두는 요소 수. 스트리밍할 때와 이력에서 다시 그릴 때가
# **반드시 같아야 한다.** 같은 키(=같은 컨테이너) 안에서 개수가 달라지면 남는
# 요소가 지워지지 않는다. 실측: 스트리밍이 [상태줄, 답변, 카드] 3개를 만들고
# 이력이 [답변, 카드] 2개를 만들어 지난 답변의 카드가 두 번 보였다.
_ASSISTANT_SLOTS = 3  # 답변 · 출처 카드 · 보고싶은 영화 고르기


# 저장 목록 화면의 칩에 쓰는 인덱스. 말풍선 인덱스(0 이상)와 겹치지 않게 음수를 쓴다.
_WATCHLIST_VIEW_INDEX = -1


def _on_pick(widget_key: str, sources: list[Source]) -> None:
    """칩을 눌렀을 때만 저장한다.

    on_change로 묶는 이유가 있다. 매 재실행마다 저장하면, 같은 영화가 나온 다른
    답변의 낡은 위젯 값이 방금 뺀 영화를 도로 담아버린다.
    """
    selected = set(st.session_state.get(widget_key) or [])
    if watchlist.sync(sources, selected):
        # 키가 바뀌면 위젯이 새로 만들어지면서 파일 내용을 다시 읽는다. 그래야
        # 같은 영화를 보여주는 다른 답변의 칩도 함께 갱신된다.
        st.session_state.watchlist_rev += 1


def render_watchlist_picker(sources: list[Source], index: int) -> None:
    """이 답변에 나온 영화 중 보고싶은 것 고르기.

    고를 게 없어도 빈 자리를 채운다. 말풍선 안 요소 수가 실행마다 달라지면
    지난 답변의 내용이 지워지지 않고 남는다(_ASSISTANT_SLOTS 참고).
    """
    if not sources:
        st.markdown("")
        return

    keys = [watchlist.movie_key(source) for source in sources]
    labels = {
        watchlist.movie_key(source): source.get("title") or "제목 없음"
        for source in sources
    }
    already = watchlist.saved_keys()
    widget_key = f"pick-{index}-{st.session_state.watchlist_rev}"
    st.pills(
        "보고싶은 영화",
        options=keys,
        format_func=lambda key: labels.get(key, key),
        selection_mode="multi",
        default=[key for key in keys if key in already],
        key=widget_key,
        label_visibility="collapsed",
        on_change=_on_pick,
        args=(widget_key, sources),
    )


def render_message(message: dict[str, Any], index: int) -> None:
    role = message["role"]
    avatar = ":material/person:" if role == "user" else ":material/movie:"
    with st.container(key=message_key(index)), st.chat_message(role, avatar=avatar):
        if role == "user":
            st.markdown(message["content"])
            return

        # 아래 세 줄이 _ASSISTANT_SLOTS개다. 출처가 없어도 빈 자리를 그려서
        # stream_answer()가 만드는 자리 수와 맞춘다.
        if message.get("error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])
        sources = message.get("sources", [])
        st.markdown(sources_html(sources), unsafe_allow_html=True)
        render_watchlist_picker(sources, index)


def status_html(text: str) -> str:
    return (
        '<div class="cine-thinking">'
        '<span class="cine-thinking-dot"></span>'
        f"<span>{html.escape(text)}…</span>"
        "</div>"
    )


def stream_answer(question: str, index: int) -> dict[str, Any]:
    """답변을 흘려 그리고, 확정된 메시지 딕셔너리를 돌려준다.

    자리를 _ASSISTANT_SLOTS개만 잡고 그 안에서만 내용을 바꾼다. 도중에 새 요소를
    만들면 이력 렌더링과 개수가 어긋나 지난 답변이 중복으로 남는다.

    토큰을 이어 붙여 화면에 보여주되 최종 내용은 done 이벤트의 answer를 쓴다.
    reset은 도구 호출 전 서두를 지우라는 신호다(rag.graph.stream_answer 참고).
    """
    body = st.empty()  # 진행 상태 → 답변 텍스트
    cards = st.empty()  # 출처 카드
    picker = st.empty()  # 보고싶은 영화 고르기
    buffer: list[str] = []

    try:
        for event in RagApiClient(API_URL).stream_query(
            question=question,
            session_id=st.session_state.session_id,
        ):
            kind = event.get("type")
            if kind == "status":
                body.markdown(
                    status_html(event.get("text", "")), unsafe_allow_html=True
                )
            elif kind == "token":
                buffer.append(event.get("text", ""))
                # 커서를 붙여 아직 쓰는 중임을 보인다.
                body.markdown("".join(buffer) + "▌")
            elif kind == "reset":
                buffer.clear()
                body.empty()
            elif kind == "done":
                answer = event.get("answer", "")
                sources = event.get("sources", [])
                body.markdown(answer)
                cards.markdown(sources_html(sources), unsafe_allow_html=True)
                with picker.container():
                    render_watchlist_picker(sources, index)
                return {"role": "assistant", "content": answer, "sources": sources}
    except ApiClientError as exc:
        body.error(str(exc))
        cards.markdown("", unsafe_allow_html=True)
        picker.markdown("")
        return {
            "role": "assistant",
            "content": str(exc),
            "sources": [],
            "error": True,
        }

    # stream_query()가 done 없이 끝나면 ApiClientError를 올린다. 여기 오면 안 된다.
    raise AssertionError("스트림이 done 없이 정상 종료했습니다.")


# 부드러운 스크롤이 끝났어야 할 시간. 이보다 늦으면 애니메이션이 돌지 않은
# 것으로 보고 즉시 맞춘다.
_SCROLL_FALLBACK_MS = 1200


def scroll_to_bottom_script(token: int) -> str:
    """맨 아래로 부드럽게 스크롤하는 컴포넌트 HTML.

    Streamlit에 스크롤 API가 없어서 높이 0짜리 iframe 안에서 부모 문서를 직접
    만진다. `token`은 iframe의 내용을 바꿔 스크립트를 다시 실행시키는 용도다 —
    값이 그대로면 브라우저가 iframe을 다시 로드하지 않아 스크롤도 일어나지
    않는다. 그래서 사용자가 위로 올려둔 화면을 매 렌더마다 끌어내리지 않는다.

    스크롤은 브라우저의 `behavior: "smooth"`에 맡긴다. 다만 부드러운 스크롤은
    애니메이션 프레임을 필요로 해서, **탭이 백그라운드면 진행되지 않는다.**
    그래서 타이머로 한 번 확인해 안 움직였으면 즉시 맞춘다. 타이머는 백그라운드
    에서도(느려질 뿐) 돌기 때문에, 사용자가 다른 탭에 있는 동안에도 위치는
    맞춰진다.
    """
    return f"""
    <script>
      // token={token}
      const box = window.parent.document
        .querySelector('[data-testid="stAppScrollToBottomContainer"]');
      if (box) {{
        const bottom = () => box.scrollHeight - box.clientHeight;
        box.scrollTo({{top: bottom(), behavior: "smooth"}});
        setTimeout(() => {{
          if (bottom() - box.scrollTop > 4) box.scrollTop = bottom();
        }}, {_SCROLL_FALLBACK_MS});
      }}
    </script>
    """


load_styles()
initialize_session()

# 자물쇠는 사이드바보다 먼저. 잠긴 동안에는 채팅 화면을 아예 만들지 않는다.
# 환영 화면과 같은 이유로 st.empty()에 담는다 — 통과한 뒤 확실히 비우기 위해서다.
lock_slot = st.empty()
if PASSCODE and not st.session_state.get("unlocked"):
    with lock_slot.container():
        render_lock()
    st.stop()
lock_slot.empty()

# 이번 실행에서 답변을 받아올 예정이면 화면을 잠근다. 사이드바는 스트리밍보다
# 먼저 그려지므로 여기서 미리 알아야 한다.
busy = bool(st.session_state.pending)

render_sidebar(busy)
# 화면에 고정(position: fixed)되므로 DOM 어디에 있든 상관없다. 조건 없이 늘
# 같은 자리에서 그려 실행마다 요소 개수가 흔들리지 않게 한다.
render_help()

# 화면 위쪽을 세 덩어리로 **미리** 만들어 둔다. 개수도 순서도 실행마다 같으므로
# 이력이 늘어도 뒤엣것이 밀리지 않는다.
#
# 이게 없으면 이력이 두 개 늘 때마다 그 뒤의 요소들이 두 칸씩 밀리고, 직전 실행에
# 어시스턴트 말풍선이 있던 자리에 이번엔 질문 말풍선이 앉는다. 질문 말풍선은
# 요소를 하나만 그리므로 어시스턴트가 쓰던 출처 카드가 지워지지 않고 남는다 —
# 질문 말풍선에 카드가 붙어 보이던 원인이다.
# 환영 화면은 st.empty()에 담는다. 한 번에 하나만 들고 있다가 .empty()로 확실히
# 비울 수 있는 유일한 수단이라서다. 일반 컨테이너에 담으면 첫 질문 뒤에도
# 히어로와 추천 버튼이 화면에 남는다(실측).
welcome_slot = st.empty()
# 지난 대화를 열면 보여줄 메시지가 통째로 바뀐다. 개수가 줄어도 남는 말풍선이
# 없도록 st.empty()에 담는다 — 컨테이너째 갈아끼우는 유일한 방법이다.
history_slot = st.empty()
scroll_area = st.container(key="cine-scroll")
stream_area = st.container(key="cine-stream")

# 지난 대화를 보는 중이면 그 내용을, 아니면 지금 대화를 그린다.
saved_movies = watchlist.saved_movies()
showing_watchlist = st.session_state.showing_watchlist and bool(saved_movies)

viewing = st.session_state.viewing
past_messages = history.load_conversation(viewing) if viewing else None
if viewing and past_messages is None:
    # 파일이 지워졌거나 읽히지 않는다. 조용히 지금 대화로 되돌린다.
    st.session_state.viewing = viewing = None

# 지난 대화나 저장 목록을 열면 아직 답 못 받은 질문은 버린다. 남겨두면 지금
# 대화로 돌아왔을 때 뒤늦게 전송돼 같은 질문에 요금이 두 번 나간다.
if viewing or showing_watchlist:
    st.session_state.pending = None

# 새로고침 등으로 스트리밍이 뚫리면 질문만 남고 답변이 안 붙는다. 그 상태로 두면
# 화면에 물음표만 덩그러니 남고 아무 일도 일어나지 않는다(pending을 미리 비우므로
# 다시 시도하지도 않는다). 여기서 한 번 수습해 무슨 일이 있었는지 남긴다.
if (
    not st.session_state.pending
    and not viewing
    and st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
):
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": "답변을 받지 못했습니다. 다시 물어봐 주세요.",
            "sources": [],
            "error": True,
        }
    )
    history.save_conversation(
        st.session_state.session_id, st.session_state.messages
    )

shown_messages = past_messages if viewing else st.session_state.messages
# 대화가 아닌 화면(저장 목록)에서는 입력을 막는다.
read_only = bool(viewing or showing_watchlist)

selected_question = None
if shown_messages or st.session_state.pending or showing_watchlist:
    welcome_slot.empty()
else:
    with welcome_slot.container():
        selected_question = render_welcome()

with history_slot.container():
    if showing_watchlist:
        st.markdown(
            '<div class="cine-readonly">아래에서 이름을 눌러 목록에서 뺄 수 있어요.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            sources_html(saved_movies, label="담아둔 영화"), unsafe_allow_html=True
        )
        # 전부 선택된 상태로 보여준다. 눌러서 끄면 목록에서 빠진다.
        render_watchlist_picker(saved_movies, _WATCHLIST_VIEW_INDEX)
    else:
        if viewing:
            st.markdown(
                '<div class="cine-readonly">지난 대화입니다. 보기만 할 수 있어요.</div>',
                unsafe_allow_html=True,
            )
        for index, chat_message in enumerate(shown_messages):
            render_message(chat_message, index)

# 스트리밍보다 **먼저** 실행돼야 한다. 답변을 다 받은 뒤에 스크롤하면 몇 초 동안
# 화면이 위에 머물러 있다가 갑자기 내려간다. 여기에 두면 방금 보낸 질문이
# 그려지자마자 답변이 나올 자리로 내려간다.
with scroll_area:
    components.html(scroll_to_bottom_script(st.session_state.scroll_token), height=0)

# 입력창은 **스트리밍보다 먼저** 만든다. 화면 위치는 어차피 맨 아래로 고정되지만,
# 스크립트 순서상 뒤에 두면 잠금 상태가 답변이 끝난 뒤에야 브라우저에 도착해서
# 정작 잠가야 할 10초 동안 열려 있게 된다.
if busy:
    placeholder = "답변을 받는 중입니다"
elif showing_watchlist:
    placeholder = "담아둔 영화를 보는 중입니다"
elif viewing:
    placeholder = "지난 대화는 읽기 전용입니다"
else:
    placeholder = "영화 제목, 감독, 장르 또는 추천 조건을 입력하세요"

typed_question = st.chat_input(
    placeholder, max_chars=500, disabled=read_only or busy
)

# 대기 중인 질문의 답변만 이력 뒤에 이어서 그린다. 이 답변이 이력에 들어가면
# 받게 될 인덱스를 키로 미리 쓴다. 그래야 다음 실행에서 같은 컨테이너로 이어진다.
with stream_area:
    asking = st.session_state.pending
    if asking and not read_only:
        # **API를 부르기 전에** 먼저 비운다. 답변이 흘러나오는 동안 사용자가 다른
        # 위젯을 건드리면 Streamlit이 스크립트를 중단하고 다시 실행하는데, 그때
        # pending이 남아 있으면 같은 질문이 한 번 더 전송된다(같은 답변에 요금
        # 두 번). 중단되면 답을 못 받는 편이 몰래 두 번 과금되는 것보다 낫다.
        st.session_state.pending = None
        answer_index = len(st.session_state.messages)
        with (
            st.container(key=message_key(answer_index)),
            st.chat_message("assistant", avatar=":material/movie:"),
        ):
            answered = stream_answer(asking, answer_index)
        st.session_state.messages.append(answered)
        # 답변이 끝날 때마다 남긴다. 창을 그냥 닫아도 대화가 보존된다.
        history.save_conversation(
            st.session_state.session_id, st.session_state.messages
        )
        # 잠금을 풀려면 한 번 더 그려야 한다. 이번 실행의 위젯은 busy=True로 이미
        # 잠긴 채 만들어졌고, 위젯이 전부 잠겨 있으면 사용자가 다시 실행시킬
        # 방법이 없어 화면이 영영 잠긴다(실측).
        st.rerun()

question = selected_question or typed_question
if question:
    # 질문은 상태에만 넣고 즉시 다시 실행한다. 그래야 사용자 말풍선도 이력
    # 렌더링을 거쳐, 화면에 그려지는 경로가 하나로 유지된다.
    st.session_state.messages.append({"role": "user", "content": question.strip()})
    st.session_state.pending = question.strip()
    st.session_state.scroll_token += 1
    st.rerun()
