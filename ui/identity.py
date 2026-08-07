"""접속자를 가르는 익명 신원.

브라우저마다 임의의 id를 하나 발급해 쿠키에 적어두고, 그 id로 대화 기록과
보고싶은 영화를 나눈다.

**인증이 아니라 격리다.** 쿠키를 복사하면 그 사람의 기록을 볼 수 있고, 쿠키를
지우면 제 기록을 잃는다. 여러 사람이 접속해도 한 사람인 것처럼 기록을 공유하던
상태(모두가 user_id="local")를 벗어나는 것이 목적이고, 여기에 비밀을 담아서는
안 된다. 제대로 된 계정이 필요해지면 이 모듈만 갈아끼우면 된다 — 저장 계층은
이미 user_id로 경계를 긋고 있다.

쿠키를 **읽는 것**은 Streamlit이 해주지만(st.context.cookies) **쓰는 것**은
해주지 않는다. 그래서 값을 적는 일은 높이 0짜리 iframe 안의 스크립트에 맡긴다
(ui/app.py의 스크롤 스크립트와 같은 방법이다).
"""

from __future__ import annotations

import re
import uuid

import streamlit as st

COOKIE_NAME = "cinebot_uid"

# 1년. 짧게 잡으면 어느 날 갑자기 대화 기록이 사라진 것처럼 보인다.
COOKIE_MAX_AGE = 60 * 60 * 24 * 365

# 발급하는 형식은 uuid4의 hex 32자다. **파일 이름이 되는 값이라 반드시 검사한다.**
# 쿠키는 사용자가 브라우저에서 고칠 수 있어서, 여기 들어오는 값은 남이 써 준
# 문자열이라고 봐야 한다. 검사 없이 경로에 붙이면 "../"로 저장소 밖을 건드린다.
_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def new_user_id() -> str:
    return uuid.uuid4().hex


def is_valid(user_id: object) -> bool:
    """쓸 수 있는 id인지.

    타입까지 본다. 쿠키에서 온 값은 문자열이라는 보장이 없다 — 실측: 테스트
    환경(AppTest)의 st.context.cookies는 MagicMock을 돌려줘서, 정규식에 그대로
    넘기면 그 자리에서 TypeError로 화면이 죽었다.
    """
    return isinstance(user_id, str) and bool(_ID_PATTERN.match(user_id))


def resolve_user_id(cookie_value: object) -> str:
    """쿠키에서 읽은 값으로 쓸 id를 정한다. 못 쓸 값이면 새로 발급한다."""
    return cookie_value if is_valid(cookie_value) else new_user_id()


def current_user_id() -> str:
    """이 브라우저의 id.

    한 세션 안에서는 값이 바뀌지 않아야 한다. 재실행마다 새로 뽑으면 방금 담은
    영화가 다음 실행에서 사라진다 — 다른 사람의 보관함을 보게 되는 것과 같다.
    """
    if "user_id" not in st.session_state:
        st.session_state.user_id = resolve_user_id(
            st.context.cookies.get(COOKIE_NAME)
        )
    return st.session_state.user_id


def remember_user_id_script(user_id: str) -> str:
    """id를 쿠키에 적는 컴포넌트 HTML.

    매 실행마다 그려도 된다. 같은 값을 다시 적을 뿐이고, 그때마다 만료가 1년
    뒤로 밀려서 자주 오는 사람의 기록이 더 오래 남는다.

    Secure는 https일 때만 붙인다. http에서 붙이면 브라우저가 쿠키를 통째로
    버려서, 로컬 개발 중에는 매번 새 사람이 된다.
    """
    return f"""
    <script>
      const secure = window.parent.location.protocol === "https:" ? "; Secure" : "";
      window.parent.document.cookie =
        "{COOKIE_NAME}={user_id}; path=/; max-age={COOKIE_MAX_AGE}; SameSite=Lax" + secure;
    </script>
    """
