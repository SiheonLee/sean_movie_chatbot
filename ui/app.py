"""CineBot: 영화 정보 RAG를 사용하는 Streamlit 채팅 화면."""

from __future__ import annotations

import hmac
import html
import os
import random
import sys
import uuid
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from ui import history, identity, watchlist
from ui.api_client import ApiClientError, RagApiClient, Source, WebSource

load_dotenv()


API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")

# 화면을 가리는 최소한의 자물쇠. 비워 두면 잠그지 않는다.
#
# 정식 인증이 아니다. 사용자 계정도, 만료도, 시도 횟수 제한도 없고 모두가 같은
# 값을 쓴다. Compose에서는 FastAPI 포트를 호스트에 게시하지 않지만, API를 별도로
# 공개하면 이 값이 직접 호출을 막아주지는 않는다. "아무나 열어보지는 못하게"
# 정도의 용도로만 쓸 것.
PASSCODE = os.getenv("CINEBOT_PASSCODE", "").strip()

# API와 같은 기본값. Compose에서는 같은 환경 변수를 양쪽에 전달한다. 화면 안내용
# 숫자라 실제 집행은 FastAPI가 맡는다.
DAILY_QUESTION_LIMIT = int(os.getenv("DAILY_QUESTION_LIMIT", "30"))
SESSION_QUESTION_LIMIT = int(os.getenv("SESSION_QUESTION_LIMIT", "12"))

# 첫 화면에 보여줄 후보. 이 중 SUGGESTION_COUNT개를 뽑아 보여준다. 늘 같은 넷을
# 내걸면 "이런 것만 되는구나"로 읽혀서, 물어볼 수 있는 범위를 실제보다 좁게
# 보이게 한다. 후보는 정보·평가·무드·장르·시의성처럼 결이 다르게 섞어 둔다.
SUGGESTED_QUESTIONS = (
    "기생충의 감독이 누구인지랑 줄거리도 알려줘",
    "영화 'Hope'에 대한 이동진 평론가의 평가가 궁금해",
    "한여름 밤에 친구들이랑 보기 좋은 영화 추천해줘",
    "한국 스릴러 영화 중에 재밌는 작품 있을까?",
    "이번 황금종려상은 어떤 영화가 받았어?",
    "올해 개봉한 액션 영화 중 평점 높은 작품 알려줘",
    "2010년대 로맨틱 코미디 영화 추천 부탁해",
    "죽기 전에 꼭 봐야 할 인생 영화가 뭐야?",
    "비 오는 날 혼자 보기 좋은 영화 추천해줘",
    "결말이 충격적인 반전 영화 알려줄래?",
    "봉준호 감독의 대표작을 개봉순으로 소개해줘",
    "부모님과 함께 감상하기 좋은 한국 영화 있을까?",
    "러닝타임이 두 시간 안 넘는 액션 영화 찾아줘",
    "1990년대에 나온 명작 SF 영화 추천해줘",
    "실화를 바탕으로 만든 감동적인 영화가 보고 싶어",
    "영상미가 정말 뛰어난 영화들을 알려줘",
    "최근 아카데미 작품상 수상작이 궁금해",
    "평점 좋은 한국 범죄 스릴러 영화 뭐가 있어?",
    "가볍게 웃으면서 볼 수 있는 코미디 영화 추천해줘",
    "이번 주말에 연인과 보기 좋은 영화 골라줘",
)
SUGGESTION_COUNT = 4


def pick_suggestions(
    questions: tuple[str, ...] = SUGGESTED_QUESTIONS, count: int = SUGGESTION_COUNT
) -> tuple[str, ...]:
    """후보 중 count개. 후보가 모자라면 있는 만큼만."""
    return tuple(random.sample(questions, min(count, len(questions))))


st.set_page_config(
    page_title="CineBot",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded",
)


# 지금 어느 화면인가. 이 값 하나가 정한다.
#
# 예전에는 showing_watchlist와 viewing 두 불리언이 나눠 들고 있었다. 둘 다 켜질
# 수 있으니 어느 쪽이 이기는지 따로 규칙이 필요했고(저장 목록이 이겼다), 그래서
# 저장 목록을 보는 중에는 지난 대화를 눌러도 화면이 바뀌지 않았다. 목록을 다
# 비워도 showing_watchlist는 True로 남아, 나중에 영화를 담는 순간 담아둔 영화
# 화면으로 튀었다. 한 값으로 합치면 두 상태 모두 표현할 수 없게 된다.
CHAT_VIEW = "chat"
WATCHLIST_VIEW = "watchlist"

# "chat" | "watchlist" | ("past", 대화 id)
View = str | tuple[str, str]


def past_view(conversation_id: str) -> View:
    return ("past", conversation_id)


def past_id(view: View) -> str | None:
    """지난 대화를 보는 중이면 그 id, 아니면 None."""
    if isinstance(view, tuple) and len(view) == 2 and view[0] == "past":
        return view[1]
    return None


def history_container_key(view: View, session_id: str, saved_count: int = 0) -> str:
    """대화를 담는 컨테이너의 키. **화면이 바뀔 때만 바뀌어야 한다.**

    키가 같으면 Streamlit이 같은 컨테이너를 재사용해 DOM이 살아남는다. 그래야
    재실행 뒤에도 스크롤 위치가 유지된다. 예전에는 st.empty()에 담아 매번
    갈아끼웠는데, 그러면 대화가 통째로 사라졌다 다시 그려지면서 브라우저가
    스크롤을 되돌린다(실측: 보고싶은 영화 칩을 누르면 화면이 대화 맨 위로 튐).

    화면이 바뀌면 키도 바뀌어 컨테이너째 교체된다 — 지난 대화를 열었을 때
    이전 화면의 말풍선이 남지 않는 이유다.

    지금 대화에는 session_id까지 넣는다. '새 대화'는 화면 종류가 그대로라,
    이게 없으면 키가 같아 컨테이너가 재사용되고 지난 말풍선이 남는다.
    """
    conversation_id = past_id(view)
    if conversation_id is not None:
        return f"cine-history-past-{conversation_id}"
    if view == WATCHLIST_VIEW:
        # 보관함은 카드 하나가 요소 하나다. 편수가 줄면 요소도 줄어드는데, 같은
        # 컨테이너 안에서 요소가 줄면 남은 것이 지워지지 않는다. 편수를 키에 넣어
        # 그때만 컨테이너째 갈아끼운다(편수가 같으면 그대로 두어 DOM을 지킨다).
        return f"cine-history-watchlist-{saved_count}"
    return f"cine-history-chat-{session_id}"


def go_to(view: View) -> None:
    """화면을 옮긴다. **옮기는 길은 여기 하나뿐이다.**

    옮긴 화면은 제 시작점에서 시작해야 한다(scroll_target 참고). 그러려면 화면을
    바꿀 때마다 스크롤 토큰을 올려야 하는데, 전환이 여러 곳에 흩어져 있으면 한
    곳을 빠뜨린 채로도 멀쩡해 보인다 — 그 화면만 이전 스크롤 위치에 남는다.
    """
    st.session_state.view = view
    st.session_state.scroll_token += 1
    st.rerun()


def resolved_view(view: View, *, has_saved_movies: bool, past_exists: bool) -> View:
    """보여줄 내용이 없어진 화면은 지금 대화로 되돌린다.

    화면을 고르는 것뿐 아니라 **상태 자체를 되돌리는 것**이 요점이다. 그리기만
    지금 대화로 하고 값은 그대로 두면, 조건이 되살아나는 순간(영화를 다시 담는
    순간) 사용자가 부르지도 않은 화면이 튀어나온다.
    """
    if view == WATCHLIST_VIEW and not has_saved_movies:
        return CHAT_VIEW
    if past_id(view) is not None and not past_exists:
        return CHAT_VIEW
    return view


def load_styles() -> None:
    css = (ROOT_DIR / "ui" / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def initialize_session() -> None:
    # 신원을 가장 먼저 정한다. 아래 상태들이 전부 이 사람의 것이기 때문이다.
    identity.current_user_id()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    # 아직 답변하지 않은 질문. 입력을 받은 실행과 답변을 그리는 실행을 나눈다.
    if "pending" not in st.session_state:
        st.session_state.pending = None
    # 값이 바뀔 때만 화면을 맨 아래로 내린다. 질문을 보낼 때와 답변이 확정될 때만
    # 오른다(scroll_to_bottom_script 참고).
    if "scroll_token" not in st.session_state:
        st.session_state.scroll_token = 0
    # 첫 화면에 내걸 추천 질문. **여기서 한 번만 뽑는다.** 그릴 때마다 뽑으면
    # 재실행마다 목록이 바뀌어서, 누르려던 버튼이 손가락 밑에서 다른 질문으로
    # 바뀐다. Streamlit은 사이드바를 건드리기만 해도 재실행된다.
    if "suggestions" not in st.session_state:
        st.session_state.suggestions = pick_suggestions()
    # 지금 보고 있는 화면(View 참고).
    if "view" not in st.session_state:
        st.session_state.view = CHAT_VIEW
    # 지난 대화 목록을 "더보기"로 다 펼쳤는지.
    if "past_expanded" not in st.session_state:
        st.session_state.past_expanded = False
    # 삭제 확인 모달이 물어볼 대상(DeleteTarget). None이면 모달을 띄우지 않는다.
    if "pending_delete" not in st.session_state:
        st.session_state.pending_delete = None
    # 보고싶은 영화 목록이 바뀔 때마다 오른다. 칩 위젯 키에 섞어 넣어, 목록이
    # 바뀌면 모든 칩이 새로 만들어지며 파일 내용을 다시 읽게 한다.
    if "watchlist_rev" not in st.session_state:
        st.session_state.watchlist_rev = 0


def reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.pending = None
    st.session_state.view = CHAT_VIEW
    # 첫 화면을 다시 보게 되는 유일한 순간이라 여기서 새로 뽑는다. 방금 쓴 것과
    # 같은 넷이 또 걸려 있으면 고를 것이 그것뿐인 줄 알게 된다.
    st.session_state.suggestions = pick_suggestions()


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
        render_nav(busy)
        render_past_conversations(busy)


def render_nav_item(
    label: str,
    *,
    key: str,
    icon: str,
    target: View,
    active: bool,
    busy: bool,
    disabled: bool = False,
) -> None:
    """사이드바 이동 항목 하나.

    **누르면 그 화면으로 간다. 그것뿐이다.** 예전에는 담아둔 영화가 토글이라
    라벨이 '담아둔 영화 · 2편' ↔ '목록 닫기 · 2편'으로 바뀌었는데, 같은 자리에서
    이름과 동작이 번갈아 나오니 지금 어느 화면인지를 글자로 역산해야 했다.

    지금 있는 곳은 라벨이 아니라 **강조**로 알린다(styles.css의 nav-active 참고).
    이미 그 화면이면 눌러도 아무 일도 하지 않는다 — 같은 자리에 다시 가겠다고
    화면 전체를 다시 그릴 이유가 없다.
    """
    # 컨테이너 키와 위젯 키는 같은 이름 공간을 쓴다. 겹치면 그 자리에서 죽는다
    # (StreamlitDuplicateElementKey). 그래서 줄에는 -row-를 붙여 갈라 둔다.
    with st.container(key=f"nav-row-{key}{'-active' if active else ''}"):
        if (
            st.button(
                label,
                key=f"nav-{key}",
                icon=icon,
                use_container_width=True,
                disabled=disabled or busy,
            )
            and not active
        ):
            go_to(target)


def render_nav(busy: bool = False) -> None:
    """화면 사이를 오가는 항목들.

    **구성이 화면에 따라 달라지지 않는다.** 어느 화면에 있든 같은 항목이 같은
    자리에 있어야 방금 누른 곳이 어디였는지 기억할 수 있다. 담은 영화가 없어도
    자리를 비우지 않고 흐리게 남겨 두는 이유도 같다 — 첫 영화를 담는 순간 항목이
    생겨나며 아래가 밀리면, 목록이 움직인 것처럼 보인다.
    """
    view = st.session_state.view
    saved = watchlist.saved_movies(user_id=st.session_state.user_id)

    render_nav_item(
        "지금 대화",
        key="chat",
        icon=":material/forum:",
        target=CHAT_VIEW,
        active=view == CHAT_VIEW,
        busy=busy,
    )
    render_nav_item(
        f"담아둔 영화 · {len(saved)}편",
        key="watchlist",
        icon=":material/bookmark:",
        target=WATCHLIST_VIEW,
        active=view == WATCHLIST_VIEW,
        busy=busy,
        # 담은 게 없으면 갈 곳이 없다. 그래도 기능이 있다는 것은 보인다.
        disabled=not saved,
    )


# 한 번에 보여주는 지난 대화 수. 나머지는 "더보기"로 펼친다.
#
# 저장은 MAX_CONVERSATIONS(50)개까지 그대로 하되, 사이드바에 50줄을 늘어놓지
# 않는다. 다 펼치면 목록이 화면 높이를 넘겨서 정작 자주 쓰는 '새 대화'와 상태
# 표시가 스크롤 밖으로 밀린다.
PAST_PAGE_SIZE = 5

# ⓘ에 붙는 안내. 개수는 한도에서 직접 읽는다 — 한도를 바꿨는데 안내만 옛날
# 숫자로 남는 일이 없도록.
HISTORY_POLICY = (
    f"대화는 최대 {history.MAX_CONVERSATIONS}개까지 저장됩니다. "
    "한도를 초과하면 가장 오래된 대화부터 자동으로 삭제됩니다."
)

# 삭제 확인 모달이 물어볼 대상. ("one", 대화 id, 제목) 또는 ("all", "", "").
DeleteTarget = tuple[str, str, str]


def delete_one(conversation_id: str, title: str) -> DeleteTarget:
    return ("one", conversation_id, title)


def delete_all() -> DeleteTarget:
    return ("all", "", "")


def delete_movie(movie_key: str, title: str) -> DeleteTarget:
    """담아둔 영화 한 편. 대화 삭제와 같은 모달을 쓰되 문구만 달라진다."""
    return ("movie", movie_key, title)


def render_past_header(saved_count: int) -> None:
    """'지난 대화'와 저장 개수, 정책 안내(ⓘ).

    개수는 목록에 보이는 수가 아니라 **저장된 파일 수**다. 한도가 걸리는 대상이
    파일이라, 지금 대화까지 세야 37/50이 실제로 남은 자리를 뜻한다.

    ⓘ는 st.popover가 아니다. popover는 열고 닫을 때마다 rerun이 걸린다
    (render_help 참고). 고정된 안내문 하나를 보여주자고 서버를 왕복할 이유가 없다.

    `<details>`도 아니다. 닫힌 details의 내용은 크롬이 아예 그리지 않아서, CSS로
    display를 바꿔도 마우스를 올렸을 때 나오게 할 수 없다(실측). tabindex를 준
    span이면 :hover로 올렸을 때, :focus로 눌렀을 때 둘 다 CSS만으로 열린다.
    """
    policy = html.escape(HISTORY_POLICY)
    st.markdown(
        '<div class="cine-past-label cine-past-head">'
        "<span>지난 대화</span>"
        '<span class="cine-past-meta">'
        f'<span class="cine-past-count">{saved_count}/{history.MAX_CONVERSATIONS}</span>'
        '<span class="cine-policy" tabindex="0" role="note" aria-label="저장 정책">ⓘ'
        f'<span class="cine-policy-panel">{policy}</span>'
        "</span>"
        "</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_past_row(
    summary: history.ConversationSummary, *, viewing: str | None, busy: bool
) -> None:
    """대화 한 줄: 여는 버튼과, 그 위에 겹쳐 둔 삭제 버튼.

    **삭제는 평소에 숨어 있다.** 줄마다 휴지통이 늘어서 있으면 목록이 아니라
    버튼 밭으로 보이고, 정작 자주 쓰는 '열기'가 묻힌다. 줄에 마우스를 올렸을
    때만(또는 키보드 포커스가 왔을 때만) 오른쪽에 뜨게 한다 — 나타나고 사라지는
    것은 CSS가 하므로 서버를 왕복하지 않는다.

    두 버튼을 나란히 놓지 않고 **세로로 쌓은 뒤 CSS로 겹친다.** 칼럼으로 나누면
    삭제가 차지한 폭만큼 제목이 좁아져서, 숨겨 둔 동안에도 제목이 일찍 잘린다.

    삭제는 여기서 하지 않는다. 대상만 세션에 적어두고 모달에 넘긴다 — 되돌릴
    방법이 없는 일이라 반드시 한 번 더 묻는다.
    """
    # 담아둔 영화를 보는 중에 눌러도 그대로 지난 대화로 넘어간다. 화면이
    # 하나뿐이므로 어느 쪽이 우선인지 따질 일이 없다.
    opened = summary.conversation_id == viewing
    # 이 키가 CSS에서 한 줄의 경계가 된다(styles.css의 st-key-past-row- 참고).
    # 열어둔 대화는 -active를 달아 이동 항목과 같은 방식으로 강조한다. 예전처럼
    # 비활성으로 흐리게 두면, 지금 있는 곳이 오히려 가장 안 보인다.
    with st.container(key=f"past-row-{summary.conversation_id}{'-active' if opened else ''}"):
        if (
            st.button(
                summary.title,
                key=f"past-{summary.conversation_id}",
                help=f"{summary.updated_at.replace('T', ' ')} · {summary.turns}번 질문",
                use_container_width=True,
                type="secondary",
                disabled=busy,
            )
            and not opened
        ):
            go_to(past_view(summary.conversation_id))

        if st.button(
            "",
            key=f"delete-past-{summary.conversation_id}",
            icon=":material/delete:",
            help="이 대화를 삭제합니다",
            disabled=busy,
        ):
            st.session_state.pending_delete = delete_one(
                summary.conversation_id, summary.title
            )
            st.rerun()


def render_past_conversations(busy: bool = False) -> None:
    """지난 대화 목록. 열어보기만 되고 이어서 물어볼 수는 없다."""
    saved = history.list_conversations(user_id=st.session_state.user_id)
    # 지금 대화는 목록에 넣지 않는다. 사이드바에서 자기 자신을 여는 건 의미가 없다.
    past = [
        summary
        for summary in saved
        if summary.conversation_id != st.session_state.session_id
    ]
    if not past:
        return

    render_past_header(len(saved))

    # 돌아가는 길은 '지금 대화' 항목 하나로 모았다(render_nav 참고). 여기에
    # 따로 두면 지난 대화를 열 때만 버튼이 생겨나며 목록이 한 칸씩 밀린다.
    viewing = past_id(st.session_state.view)

    expanded = st.session_state.past_expanded
    shown = past if expanded else past[:PAST_PAGE_SIZE]
    for summary in shown:
        render_past_row(summary, viewing=viewing, busy=busy)

    hidden = len(past) - len(shown)
    if hidden and st.button(
        f"더보기 {hidden}개",
        key="past-more",
        icon=":material/expand_more:",
        use_container_width=True,
        disabled=busy,
    ):
        st.session_state.past_expanded = True
        st.rerun()
    elif expanded and st.button(
        "접기",
        key="past-less",
        icon=":material/expand_less:",
        use_container_width=True,
        disabled=busy,
    ):
        st.session_state.past_expanded = False
        st.rerun()

    if st.button(
        "지난 대화 모두 삭제",
        key="past-delete-all",
        icon=":material/delete_sweep:",
        use_container_width=True,
        disabled=busy,
    ):
        st.session_state.pending_delete = delete_all()
        st.rerun()


def forget_delete_target() -> None:
    """모달을 바깥 클릭·X·ESC로 닫았을 때 지울 대상을 잊는다.

    이렇게 닫는 길은 '취소' 버튼을 거치지 않는다. 기본값(on_dismiss="ignore")은
    서버에 알리지도 않아서, pending_delete가 그대로 남는다. 그러면 다음에 아무
    버튼이나 눌러 화면이 다시 그려지는 순간 닫았던 모달이 되살아난다(실측:
    바깥을 눌러 닫고 '더보기'를 누르면 삭제 모달이 다시 떴다).
    """
    st.session_state.pending_delete = None


@st.dialog("담아둔 영화에서 빼기", on_dismiss=forget_delete_target)
def render_movie_removal() -> None:
    """영화를 뺄 때의 모달. 본문은 대화 삭제와 같고 제목만 다르다.

    st.dialog의 제목은 데코레이터 인자라 실행 중에 바꿀 수 없다. 한 모달로
    돌려쓰면 영화를 빼는데 '지난 대화 삭제'라고 적힌 창이 뜬다.
    """
    render_delete_body()


@st.dialog("지난 대화 삭제", on_dismiss=forget_delete_target)
def render_delete_confirmation() -> None:
    """지우기 전에 한 번 더 묻는다.

    파일을 지우는 일이라 되돌릴 수 없다. 사이드바는 목록 버튼이 촘촘히 붙어
    있어서, 열어보려다 삭제를 누르는 일이 실제로 일어난다.
    """
    render_delete_body()


def render_delete_body() -> None:
    """모달 본문. 무엇을 지우는지에 따라 문구와 버튼 이름만 달라진다."""
    kind, target_id, title = st.session_state.pending_delete
    if kind == "all":
        # 여기서 다시 센다. 버튼을 누른 뒤 다른 탭에서 지웠을 수도 있다.
        count = len(history.list_conversations(user_id=st.session_state.user_id))
        st.markdown(f"지난 대화 **{count}개**를 모두 삭제할까요?")
        st.caption("지운 대화는 되돌릴 수 없습니다.")
    elif kind == "movie":
        st.markdown(f"**{title}**\n\n담아둔 영화에서 뺄까요?")
        # 대화와 달리 되돌릴 수 있다. 겁줄 일이 아니다.
        st.caption("답변에서 다시 담을 수 있습니다.")
    else:
        # 제목을 문장에 끼워 넣지 않는다. 제목이 무엇으로 끝나느냐에 따라
        # 조사가 달라져서(영화를/기생충을) "을(를)"처럼 어정쩡해진다.
        st.markdown(f"**{title}**\n\n이 대화를 삭제할까요?")
        st.caption("지운 대화는 되돌릴 수 없습니다.")

    confirm_column, cancel_column = st.columns(2)
    if confirm_column.button(
        "빼기" if kind == "movie" else "삭제",
        key="confirm-delete",
        type="primary",
        use_container_width=True,
    ):
        if kind == "all":
            history.delete_all_conversations(user_id=st.session_state.user_id)
        elif kind == "movie":
            watchlist.remove(target_id, user_id=st.session_state.user_id)
            # 답변에 달린 칩도 이 영화를 뺀 상태로 다시 그려져야 한다.
            st.session_state.watchlist_rev += 1
        else:
            history.delete_conversation(target_id, user_id=st.session_state.user_id)
        st.session_state.pending_delete = None
        # 지운 대화를 열어보던 중이었다면 resolved_view가 지금 대화로 되돌린다.
        st.rerun()

    if cancel_column.button("취소", key="cancel-delete", use_container_width=True):
        st.session_state.pending_delete = None
        st.rerun()


HELP_EXAMPLES = (
    ("정보", "기생충 감독이 누구야?"),
    ("추천", "한국 스릴러 영화를 추천해줘"),
    ("분위기", "비 오는 날 볼 잔잔한 영화"),
    ("시청처", "기생충 넷플릭스에서 볼 수 있어?"),
    ("평가·화제", "기생충 평단 반응 어땠어?"),
)

DATA_NOTICE = (
    "약 500편의 선별 영화 카탈로그와 TMDB·웹 검색 결과를 사용합니다. "
    "개봉·평점·OTT 정보는 누락되거나 바뀔 수 있으니 출처 원문도 확인해주세요. "
    f"익명 브라우저 ID당 하루 {DAILY_QUESTION_LIMIT}회(UTC 기준), "
    f"대화당 {SESSION_QUESTION_LIMIT}회까지 질문할 수 있습니다. "
    "쿠키를 지우면 지난 대화와 저장 목록의 연결이 끊길 수 있습니다."
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
        f'<div class="cine-help-note">{html.escape(DATA_NOTICE)}</div>'
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


def render_welcome(questions: tuple[str, ...]) -> str | None:
    """환영 화면. 누른 질문을 돌려준다.

    보여줄 질문은 밖에서 받는다. 여기서 뽑으면 재실행마다 바뀐다.
    """
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
    for row_start in range(0, len(questions), 2):
        columns = st.columns(2)
        for column, question in zip(columns, questions[row_start : row_start + 2]):
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


def poster_html(url: str | None, alt: str, css_class: str) -> str:
    """포스터 자리. 이미지가 없거나 **깨져도** 자리를 지키고 🎬가 대신 보인다.

    onerror로 갈아끼우는 흔한 방법은 여기서 못 쓴다 — Streamlit이 인라인 이벤트
    핸들러를 지운다(실측: unsafe_allow_html로 넣은 onerror가 DOM에서 사라졌다).

    그래서 상자에 🎬를 깔고 그 위에 이미지를 덮는다. 이미지가 뜨면 상자를 완전히
    가리고, 실패하면 아무것도 그리지 않아 깔아둔 🎬가 그대로 보인다. TMDB 경로가
    낡아 404가 나도 카드 높이가 무너지지 않는다.

    img의 alt는 비운다. 깨졌을 때 대체 텍스트가 🎬 위에 겹쳐 보이기 때문이다.
    읽어줄 이름은 상자가 aria-label로 들고 있다.
    """
    box = f'<div class="{css_class}" role="img" aria-label="{alt}">'
    if not url:
        return f"{box}</div>"
    return (
        f'{box}<img class="{css_class}-img" src="{html.escape(url)}" '
        'alt="" loading="lazy"></div>'
    )


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

    poster = poster_html(poster_url(source), f"{title} 포스터", "cine-card-poster")

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


def sources_html(sources: list[Source], label: str = "검색 근거 영화") -> str:
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
_ASSISTANT_SLOTS = 4  # 답변 · 출처 카드 · 보고싶은 영화 고르기 · 출처 표기


# 답변 아래에 "무엇을 보고 답했는지"를 적는다. 답변만 읽어서는 웹에서 찾아온
# 것인지 로컬 색인에서 고른 것인지 알 수 없다.
#
# JustWatch는 취향이 아니라 의무다. TMDB의 OTT 편성 데이터는 JustWatch가
# 제공하며 표기 없이 쓸 수 없다. 그래서 이 표시는 접거나 감추지 않는다.
ATTRIBUTION_LABELS = {
    "tmdb": "TMDB",
    "local": "로컬 DB",
    "web": "웹 검색",
    "justwatch": "JustWatch",
}


def attribution_html(
    attributions: list[str], web_sources: list[WebSource] | None = None
) -> str:
    """출처 표기와 웹 검색의 구체적인 링크. 안전한 HTTP(S) URL만 렌더링한다."""
    labels = [
        ATTRIBUTION_LABELS[mark] for mark in attributions if mark in ATTRIBUTION_LABELS
    ]
    links = []
    for source in web_sources or []:
        url = (source.get("url") or "").strip()
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        title = (source.get("title") or parsed.netloc).strip()
        links.append(
            f'<a href="{html.escape(url, quote=True)}" target="_blank" '
            f'rel="noopener noreferrer">{html.escape(title)}</a>'
        )
    if not labels and not links:
        return ""
    parts = [*(f"<span>{html.escape(label)}</span>" for label in labels), *links]
    return (
        '<div class="cine-attribution">출처 '
        + " · ".join(parts)
        + "</div>"
    )


# 보관함 카드의 포스터. 답변에 붙는 카드(w185)보다 크게 받는다 — 답변에서는
# 카드가 근거라 작아도 되지만, 여기서는 카드가 화면의 주인공이다.
SHELF_POSTER_BASE_URL = "https://image.tmdb.org/t/p/w342"

# 한 줄에 놓는 카드 수.
SHELF_COLUMNS = 4


def shelf_poster_url(source: Source) -> str | None:
    path = (source.get("poster_path") or "").strip()
    return f"{SHELF_POSTER_BASE_URL}{path}" if path else None


def shelf_card_html(source: Source) -> str:
    """보관함 카드 하나.

    접었다 펴는 부분이 없다. 답변의 카드는 '이 답변이 무엇을 근거로 했는가'라
    감독·출연·줄거리까지 담지만, 보관함은 '내가 무엇을 담아뒀는가'를 훑는
    곳이라 포스터와 제목·연도·평점이면 충분하다.
    """
    title = html.escape(source.get("title") or "제목 정보 없음")
    year = html.escape(str(source.get("year") or "-"))
    rating = float(source.get("vote_average") or 0.0)

    poster = poster_html(
        shelf_poster_url(source), f"{title} 포스터", "cine-shelf-poster"
    )

    return (
        '<div class="cine-shelf-card">'
        f"{poster}"
        # 제목과 메타는 한 덩어리로 묶는다. 삭제 아이콘이 이 덩어리의 첫 줄
        # 오른쪽에 앉아야 해서, 높이를 예측할 수 있는 상자가 필요하다.
        '<div class="cine-shelf-body">'
        f'<div class="cine-shelf-title">{title}</div>'
        f'<div class="cine-shelf-meta">{year} · ⭐ {rating:.1f}</div>'
        "</div>"
        "</div>"
    )


def watchlist_markdown(saved: list[Source]) -> str:
    """담아둔 영화를 불렛 목록으로. 제목만 적되 같은 제목은 연도로 가른다.

    이름 규칙을 movie_labels에 맡긴다 — 화면에서 두 올드보이를 가르는 방식과
    옮겨 적은 목록의 방식이 다르면, 받아본 사람이 어느 쪽인지 알 수 없다.
    """
    movies = unique_sources(saved)
    labels = movie_labels(movies)
    lines = [f"- {labels[watchlist.movie_key(source)]}" for source in movies]
    return "\n".join(lines) + "\n" if lines else ""


def watchlist_filename(today: date | None = None) -> str:
    """내려받는 파일 이름. 날짜를 붙여 여러 번 받아도 덮어쓰지 않게 한다."""
    return f"cinebot-담아둔-영화-{(today or date.today()).isoformat()}.md"


def render_watchlist_download(saved: list[Source], busy: bool = False) -> None:
    """제목 목록 내려받기.

    편수는 라벨에 넣지 않는다 — 바로 옆 안내가 이미 말하고 있고, 버튼 이름은
    무엇이 일어나는지만 짧게 알리면 된다.
    """
    st.download_button(
        "마크다운으로 내려받기",
        data=watchlist_markdown(saved).encode("utf-8"),
        file_name=watchlist_filename(),
        mime="text/markdown",
        key="shelf-download",
        icon=":material/download:",
        disabled=busy,
    )


def render_watchlist_shelf(saved: list[Source], busy: bool = False) -> None:
    """담아둔 영화를 카드로 늘어놓고, 카드마다 뺄 수 있게 한다.

    카드 본문은 HTML이지만 삭제는 진짜 위젯이라야 서버에 닿는다. 그래서 카드
    하나를 컨테이너 하나로 감싸고 그 안에 [카드 HTML, 삭제 버튼] 둘을 넣은 뒤,
    CSS로 버튼만 포스터 위에 얹는다(지난 대화 줄과 같은 구조). 카드 묶음 전체를
    HTML 한 덩어리로 그리면 카드마다 버튼을 앉힐 자리를 잡을 수 없다.
    """
    for row_start in range(0, len(saved), SHELF_COLUMNS):
        columns = st.columns(SHELF_COLUMNS, gap="medium")
        row = saved[row_start : row_start + SHELF_COLUMNS]
        for offset, (column, source) in enumerate(zip(columns, row)):
            index = row_start + offset
            with column, st.container(key=f"shelf-card-{index}"):
                st.markdown(shelf_card_html(source), unsafe_allow_html=True)
                if st.button(
                    "",
                    key=f"shelf-delete-{index}",
                    icon=":material/delete:",
                    help="담아둔 영화에서 뺍니다",
                    disabled=busy,
                ):
                    st.session_state.pending_delete = delete_movie(
                        watchlist.movie_key(source), source.get("title") or "이 영화"
                    )
                    st.rerun()


def _on_pick(widget_key: str, sources: list[Source]) -> None:
    """칩을 눌렀을 때만 저장한다.

    on_change로 묶는 이유가 있다. 매 재실행마다 저장하면, 같은 영화가 나온 다른
    답변의 낡은 위젯 값이 방금 뺀 영화를 도로 담아버린다.
    """
    selected = set(st.session_state.get(widget_key) or [])
    if watchlist.sync(sources, selected, user_id=st.session_state.user_id):
        # 키가 바뀌면 위젯이 새로 만들어지면서 파일 내용을 다시 읽는다. 그래야
        # 같은 영화를 보여주는 다른 답변의 칩도 함께 갱신된다.
        st.session_state.watchlist_rev += 1


def unique_sources(sources: list[Source]) -> list[Source]:
    """같은 영화(제목·연도가 같은)를 한 편으로 접는다. 순서는 그대로 둔다."""
    seen: set[str] = set()
    unique: list[Source] = []
    for source in sources:
        key = watchlist.movie_key(source)
        if key not in seen:
            seen.add(key)
            unique.append(source)
    return unique


def movie_labels(sources: list[Source]) -> dict[str, str]:
    """사람에게 보여줄 이름. 같은 제목이 둘 이상이면 연도를 붙여 가른다.

    칩과 내려받는 목록이 함께 쓴다 — 화면에서 가려지는 두 영화는 텍스트로
    옮겨 적어도 똑같이 가려져야 한다.

    **st.pills는 표시 이름이 같은 두 항목을 구분하지 못한다.** 넘기는 값(movie_key)이
    달라도 화면에 같은 이름으로 보이면 하나로 접힌다. 실측: 올드보이 원작(2003)과
    스파이크 리의 리메이크(2013)를 함께 담으면 파일에는 한 편만 들어가고, 지우려
    해도 다른 한 편이 남았다.

    구분이 필요한 만큼만 붙인다. 늘 연도를 달면 칩이 길어져 목록이 답답해진다.
    """
    titles = [source.get("title") or "제목 없음" for source in sources]
    duplicated = {title for title in titles if titles.count(title) > 1}

    labels: dict[str, str] = {}
    for source in sources:
        title = source.get("title") or "제목 없음"
        year = source.get("year")
        labels[watchlist.movie_key(source)] = (
            f"{title} ({year})" if title in duplicated and year else title
        )
    return labels


def render_watchlist_picker(sources: list[Source], index: int) -> None:
    """이 답변에 나온 영화 중 보고싶은 것 고르기.

    고를 게 없어도 빈 자리를 채운다. 말풍선 안 요소 수가 실행마다 달라지면
    지난 답변의 내용이 지워지지 않고 남는다(_ASSISTANT_SLOTS 참고).
    """
    if not sources:
        st.markdown("")
        return

    # 같은 영화가 두 번 들어오면 칩도 두 개가 되고, 그 둘은 이름까지 같아 서로를
    # 가릴 수 없다. 먼저 한 편으로 접는다.
    sources = unique_sources(sources)
    keys = [watchlist.movie_key(source) for source in sources]
    labels = movie_labels(sources)
    already = watchlist.saved_keys(user_id=st.session_state.user_id)
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

        # 아래 네 줄이 _ASSISTANT_SLOTS개다. 출처가 없어도 빈 자리를 그려서
        # stream_answer()가 만드는 자리 수와 맞춘다.
        if message.get("error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])
        sources = message.get("sources", [])
        st.markdown(sources_html(sources), unsafe_allow_html=True)
        render_watchlist_picker(sources, index)
        st.markdown(
            attribution_html(
                message.get("attributions", []), message.get("web_sources", [])
            ),
            unsafe_allow_html=True,
        )


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
    attribution = st.empty()  # 출처 표기
    buffer: list[str] = []

    try:
        for event in RagApiClient(
            API_URL, user_id=st.session_state.user_id
        ).stream_query(
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
                web_sources = event.get("web_sources", [])
                attributions = event.get("attributions", [])
                body.markdown(answer)
                cards.markdown(sources_html(sources), unsafe_allow_html=True)
                with picker.container():
                    render_watchlist_picker(sources, index)
                attribution.markdown(
                    attribution_html(attributions, web_sources), unsafe_allow_html=True
                )
                return {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "web_sources": web_sources,
                    # 이력에 함께 남긴다. 지난 대화를 열어봐도 근거가 보여야 하고,
                    # JustWatch 표기는 그때도 붙어 있어야 한다.
                    "attributions": attributions,
                }
    except ApiClientError as exc:
        body.error(str(exc))
        cards.markdown("", unsafe_allow_html=True)
        picker.markdown("")
        attribution.markdown("")
        return {
            "role": "assistant",
            "content": str(exc),
            "sources": [],
            "web_sources": [],
            "attributions": [],
            "error": True,
        }

    # stream_query()가 done 없이 끝나면 ApiClientError를 올린다. 여기 오면 안 된다.
    raise AssertionError("스트림이 done 없이 정상 종료했습니다.")


# 부드러운 스크롤이 끝났어야 할 시간. 이보다 늦으면 애니메이션이 돌지 않은
# 것으로 보고 즉시 맞춘다.
_SCROLL_FALLBACK_MS = 1200


def scroll_target(view: View) -> str:
    """그 화면을 열었을 때 가야 할 곳.

    **화면을 옮기면 그 화면의 시작점으로 간다**는 한 가지 규칙이다. 대화의
    시작점은 최근 메시지가 있는 맨 아래이고, 보관함과 지난 대화는 맨 위다 —
    보관함은 위에 내려받기가 있고, 지난 대화는 처음부터 읽는 곳이다.
    """
    return "bottom" if view == CHAT_VIEW else "top"


def scroll_to_bottom_script(token: int, target: str = "bottom") -> str:
    """화면을 그 자리로 부드럽게 옮기는 컴포넌트 HTML.

    Streamlit에 스크롤 API가 없어서 높이 0짜리 iframe 안에서 부모 문서를 직접
    만진다. `token`은 "이번에 스크롤할 차례인가"를 가르는 값으로, **질문을 보낼
    때, 답변이 확정될 때, 화면을 옮길 때** 오른다. 질문 시점에 한 번 더 올리는
    이유가 있다 — 그때는 답변이 아직 비어 있어서, 답변이 길면 끝이 화면 밖에
    남는다. 나머지 재실행(칩 클릭, 영화 빼기 등)에서는 오르지 않으므로 사용자가
    올려둔 화면을 건드리지 않는다.

    **토큰을 srcdoc에 적어두는 것만으로는 부족하다.** 그건 "내용이 같으면
    브라우저가 iframe을 다시 로드하지 않는다"는 데 기댄 것인데, 재실행 중에
    iframe이 DOM에서 떨어졌다 다시 붙으면 내용이 같아도 스크립트가 다시 돈다.
    그러면 사용자가 올려둔 화면이 영문 모를 이유로 맨 아래로 끌려간다(실측:
    보고싶은 영화 칩을 누르자 화면이 아래로 튐).

    그래서 처리한 토큰을 **부모 문서에** 적어둔다. iframe은 다시 만들어져도
    부모는 그대로라, 같은 토큰이면 두 번째부터는 아무 일도 하지 않는다.

    스크롤은 브라우저의 `behavior: "smooth"`에 맡긴다. 다만 부드러운 스크롤은
    애니메이션 프레임을 필요로 해서, **탭이 백그라운드면 진행되지 않는다.**
    그래서 타이머로 한 번 확인해 안 움직였으면 즉시 맞춘다. 타이머는 백그라운드
    에서도(느려질 뿐) 돌기 때문에, 사용자가 다른 탭에 있는 동안에도 위치는
    맞춰진다.
    """
    return f"""
    <script>
      const token = {token};
      const parent = window.parent;
      // 스크롤되는 상자가 화면마다 다르다. 입력창이 있으면 Streamlit이
      // stAppScrollToBottomContainer를 만들지만, 읽기 전용 화면에는 입력창이
      // 없어서 그 상자도 없다(실측: 보관함에서 querySelector가 null). 그때는
      // 본문(stMain)이 직접 스크롤한다.
      const box = parent.document
        .querySelector('[data-testid="stAppScrollToBottomContainer"]')
        || parent.document.querySelector('[data-testid="stMain"]');
      // 이미 처리한 토큰이면 이번 실행은 iframe이 다시 붙은 것뿐이다.
      if (box && parent.__cineScrollToken !== token) {{
        parent.__cineScrollToken = token;
        const bottom = () => "{target}" === "top" ? 0 : box.scrollHeight - box.clientHeight;
        box.scrollTo({{top: bottom(), behavior: "smooth"}});
        setTimeout(() => {{
          if (Math.abs(bottom() - box.scrollTop) > 4) box.scrollTop = bottom();
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
# 모달은 사이드바가 아니라 화면 한가운데에 뜨는 앱 전체의 것이라 밖에서 띄운다.
if st.session_state.pending_delete:
    if st.session_state.pending_delete[0] == "movie":
        render_movie_removal()
    else:
        render_delete_confirmation()
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
# 어느 화면을 그릴지 **슬롯을 만들기 전에** 정한다. 대화를 담을 컨테이너의 키에
# 화면이 들어가기 때문이다(history_container_key 참고).
#
# 지난 대화를 보는 중이면 그 내용을, 아니면 지금 대화를 그린다.
saved_movies = watchlist.saved_movies(user_id=st.session_state.user_id)
viewing = past_id(st.session_state.view)
# 파일이 지워졌거나 읽히지 않을 수 있다. 그때는 조용히 지금 대화로 되돌린다.
past_messages = (
    history.load_conversation(viewing, user_id=st.session_state.user_id)
    if viewing
    else None
)

st.session_state.view = resolved_view(
    st.session_state.view,
    has_saved_movies=bool(saved_movies),
    past_exists=past_messages is not None,
)
viewing = past_id(st.session_state.view)
past_messages = past_messages if viewing else None
showing_watchlist = st.session_state.view == WATCHLIST_VIEW

welcome_slot = st.empty()
history_slot = st.container(
    key=history_container_key(
        st.session_state.view, st.session_state.session_id, len(saved_movies)
    )
)
scroll_area = st.container(key="cine-scroll")
stream_area = st.container(key="cine-stream")

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
        st.session_state.session_id,
        st.session_state.messages,
        user_id=st.session_state.user_id,
    )

shown_messages = past_messages if viewing else st.session_state.messages
# 대화가 아닌 화면(저장 목록)에서는 입력을 막는다.
read_only = bool(viewing or showing_watchlist)

selected_question = None
if shown_messages or st.session_state.pending or showing_watchlist:
    welcome_slot.empty()
else:
    with welcome_slot.container():
        selected_question = render_welcome(st.session_state.suggestions)

with history_slot:
    if showing_watchlist:
        # 안내와 내려받기를 한 줄에 둔다. 세로 가운데로 맞추지 않으면 높이가 서로
        # 달라(안내는 여백 있는 상자, 버튼은 한 줄) 위아래로 어긋나 보인다.
        notice_column, download_column = st.columns(
            [3, 2], gap="medium", vertical_alignment="center"
        )
        notice_column.markdown(
            '<div class="cine-readonly">'
            f"담아둔 영화 {len(saved_movies)}편입니다. "
            "카드에 마우스를 올리면 뺄 수 있어요."
            "</div>",
            unsafe_allow_html=True,
        )
        with download_column:
            render_watchlist_download(saved_movies, busy)
        render_watchlist_shelf(saved_movies, busy)
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
    # 이 브라우저의 id를 쿠키에 적어 둔다. 조건 없이 늘 그린다 — 값은 같고,
    # 그릴 때마다 만료가 1년 뒤로 밀려서 자주 오는 사람의 기록이 더 오래 남는다.
    components.html(
        identity.remember_user_id_script(st.session_state.user_id), height=0
    )
    components.html(
        scroll_to_bottom_script(
            st.session_state.scroll_token, scroll_target(st.session_state.view)
        ),
        height=0,
    )

# 입력창은 **스트리밍보다 먼저** 만든다. 화면 위치는 어차피 맨 아래로 고정되지만,
# 스크립트 순서상 뒤에 두면 잠금 상태가 답변이 끝난 뒤에야 브라우저에 도착해서
# 정작 잠가야 할 10초 동안 열려 있게 된다.
#
# **읽기 전용 화면에는 입력창을 아예 두지 않는다.** 담아둔 영화와 지난 대화에서는
# 물어볼 수 없으니, 잠긴 입력창을 남겨 봐야 "왜 막혔지"만 남는다. 화면 아래를
# 비우면 보고 있는 내용이 그만큼 넓어진다.
placeholder = (
    "답변을 받는 중입니다"
    if busy
    else "영화 제목, 감독, 장르 또는 추천 조건을 입력하세요"
)
typed_question = (
    None if read_only else st.chat_input(placeholder, max_chars=500, disabled=busy)
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
            st.session_state.session_id,
            st.session_state.messages,
            user_id=st.session_state.user_id,
        )
        # 답변이 확정됐으니 그 말풍선까지 내려간다. 질문을 보낼 때 한 번 내려가지만
        # 그때는 답변이 아직 비어 있어서, 답변이 길면 끝이 화면 밖에 남는다.
        # 위로 올려 지난 대화를 읽던 중이라면 여기서 답변 자리로 돌아온다.
        st.session_state.scroll_token += 1
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
