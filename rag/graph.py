"""
도구 호출 기반 LangGraph 파이프라인.

질문 유형을 규칙으로 분류하고 경로를 나누는 대신, LLM이 도구를 고르게 한다.
그래프는 2노드 3엣지로 고정되고, 도구를 늘려도 구조는 그대로다.

    START → agent → (도구 호출이 있으면) tools → agent → ... → END

이전 구조(9노드 12엣지)에서 사라진 것들:
- 질문 분류 정규식        → LLM의 도구 선택
- 필터 추출 정규식        → 도구 인자
- 검색 문서 관련성 평가    → 도구가 이미 조건에 맞는 것만 반환
- 질의 재작성 재시도 루프  → 도구가 "조건을 완화하세요"를 반환하면 LLM이 재호출
- 출처 번호 파싱 3중 폴백  → ToolMessage.artifact
"""

from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from collections.abc import Iterator
from datetime import date, datetime, timezone

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from rag.config import settings
from rag.providers import get_llm
from rag.tools import ATTRIBUTION_ORDER, TOOLS, attributions_for

logger = logging.getLogger(__name__)

# 무한 도구 호출 방어. 도구 → agent → 도구 왕복이 이 횟수를 넘으면 중단된다.
# LangGraph는 노드 실행 수로 세므로 도구 호출 라운드는 대략 이 값의 절반이다.
_RECURSION_LIMIT = 12

# 한도에 걸렸을 때 사용자에게 보여줄 답변. 예외를 그대로 올리면 UI가 500을 받고
# 체크포인트에는 최종 답변 없는 도구 메시지만 남아, 다음 턴이 그 잔여물을 근거로
# 답하게 된다(실측: 같은 질문이 두 번 들어간 트레이스).
_RECURSION_FALLBACK = (
    "정보를 찾는 과정이 너무 길어져 중단했습니다. "
    "질문을 조금 더 구체적으로 나눠서 다시 물어봐 주세요."
)

# 스트리밍 중 사용자에게 보여줄 중간 상태. 첫 답변 토큰까지 LLM 추론 1회 + 도구
# 실행이 선행되므로(실측 4.7초), 그동안 화면이 비어 있으면 멈춘 것처럼 보인다.
_TOOL_STATUS = {
    "search_movies": "영화를 찾는 중",
    "get_movie_details": "상세 정보를 확인하는 중",
    "search_by_vibe": "분위기에 맞는 영화를 고르는 중",
    "web_search": "웹에서 최신 정보를 찾는 중",
}
_DEFAULT_TOOL_STATUS = "정보를 확인하는 중"
_THINKING_STATUS = "질문을 살펴보는 중"

# 도구 호출 전 서두를 화면에 내보내지 않기 위해 앞부분을 쥐고 있는 양(글자 수).
#
# 라운드가 도구 호출로 끝날지는 라운드가 끝나야 알 수 있다(§4-21). 그래서 앞부분을
# 잠깐 쥐고 있다가, 서두라기엔 너무 길어지면 그때부터 최종 답변으로 보고 흘린다.
# 시간이 아니라 글자 수로 재는 이유는 재현 가능해서다 — 시간 기준은 API 응답 속도에
# 따라 결과가 달라져 테스트로 고정할 수 없다.
#
# 실측 서두 길이(프롬프트로 억제한 뒤): 27·37·45·63자. 최종 답변은 수백 자다.
# 양쪽으로 부드럽게 무너진다 — 이보다 긴 서두는 기존처럼 reset으로 취소하고,
# 이보다 짧은 답변은 라운드가 끝날 때 한 번에 나타난다(짧아서 스트리밍 이득도 없다).
_PREAMBLE_HOLD_CHARS = 80


def tool_status(names: list[str]) -> str:
    """도구 이름을 사용자용 한 줄 상태 문구로. 병렬 호출은 중복을 접는다."""
    labels: list[str] = []
    for name in names:
        label = _TOOL_STATUS.get(name, _DEFAULT_TOOL_STATUS)
        if label not in labels:
            labels.append(label)
    return " · ".join(labels) if labels else _DEFAULT_TOOL_STATUS


_SYSTEM_PROMPT_TEMPLATE = (
    "당신은 영화 정보를 안내하는 한국어 도우미입니다.\n\n"
    # 실측: '오늘 뭐먹지?'에 식사 추천을 했다. 도구가 없는 주제라 답이 전부
    # 학습 지식에서 나오고, 출처 카드도 붙지 않아 근거를 확인할 길이 없다.
    "**영화와 이 서비스에 대한 질문에만 답합니다.**\n"
    "영화·감독·배우·장르·평점·시청처·수상·평단 반응, 그리고 이 챗봇 사용법이 "
    "답할 수 있는 범위입니다. 그 밖의 주제(식사·날씨·건강·코딩·일반 상식·인생 "
    "상담 등)는 아무리 간단해도 답하지 마세요. 짧게 사양하고 어떤 영화 질문을 "
    "할 수 있는지 한 줄로 안내하세요. 영화 이야기로 자연스럽게 이어지는 질문만 "
    "받으세요.\n\n"
    # 날짜를 안 주면 학습 시점 지식에 의존한다. 실측: '올해 개봉한 한국 영화'가
    # year_from=2024로 나가고, 2026년 영화를 돌려줘도 "아직 개봉 안 함"이라고 답했다.
    "오늘은 {today}입니다. '올해'는 {year}년, '작년'은 {last_year}년입니다.\n"
    "당신의 학습 데이터는 오늘보다 오래됐습니다. 연도·최신작·개봉 예정작에 대해서는 "
    "당신의 기억이 아니라 도구 결과를 신뢰하세요. 도구가 {year}년 영화를 돌려주면 "
    "그것이 사실입니다. '아직 개봉하지 않았다', '정보가 없다'고 답하지 마세요.\n\n"
    "질문에 답하기 전에 반드시 적절한 도구로 정보를 확인하세요. "
    "유명한 영화라도 기억에 의존하지 말고 도구 결과를 근거로 답하세요.\n"
    # 모르는 제목일수록 검색해야 하는데, 오히려 되묻고 끝내는 경우가 있었다.
    # 최신작은 학습 데이터에 없으니 '모르는 제목'과 '존재하지 않는 영화'는 다르다.
    "제목이 모호하거나 처음 듣는 영화라도 먼저 도구로 검색하세요. 검색 결과를 "
    "확인한 뒤에도 어느 작품인지 가릴 수 없을 때만 되물으세요. 검색해보지 않고 "
    "'어떤 작품인지 알려달라'고 답하지 마세요.\n"
    # 실측: '2010년대 로맨틱 코미디'에서 도구가 준 후보에 맞는 것이 없자 러브
    # 로지·어바웃 타임 같은 다섯 편을 제 기억으로 추천했다. 답변은 그럴듯했지만
    # 출처 카드는 도구가 준 15편(기생충·토르…)이라 서로 어긋났다.
    "**답변에 등장하는 영화는 모두 도구 결과에 있던 것이어야 합니다.** 도구가 "
    "돌려주지 않은 영화는 제목조차 꺼내지 마세요. 아무리 잘 아는 작품이라도, "
    "질문에 더 잘 맞아 보여도 마찬가지입니다.\n"
    "도구 결과에 마땅한 영화가 없으면 **없다고 말하세요.** 기억으로 채우지 말고, "
    "찾은 것 중 가까운 작품을 이유와 함께 제안하거나 다른 조건을 권하세요. "
    "예: '로맨틱 코미디로 좁히면 결과가 적습니다. 로맨스 쪽으로 넓혀볼까요?'\n"
    "도구 결과에 없는 사실은 지어내지 마세요. 확인되지 않으면 확인할 수 없다고 답하세요.\n"
    "OTT 시청처 정보가 없을 때는 '제공하지 않는다'가 아니라 '확인되지 않는다'라고 "
    "답하세요. 데이터가 불완전할 수 있습니다.\n"
    "영화를 언급할 때는 제목과 개봉연도를 함께 쓰세요.\n"
    "도구가 범위를 좁혀 달라고 요청하면, 그 이유를 사용자에게 그대로 전달하고 "
    "어떤 조건으로 좁힐지 되물으세요. 임의로 조건을 지어내 다시 호출하지 마세요.\n"
    "사용자가 '다른 영화'나 '그거 말고'를 요청하면, 이미 추천한 제목을 "
    "search_by_vibe의 exclude_titles 인자에 넣어 다시 호출하세요.\n"
    "'잔인하지 않은', '슬프지 않은' 같은 부정 조건은 검색 문장이 아니라 "
    "max_violence, max_sadness 같은 수치 인자로 옮기세요.\n"
    "이전 대화 자체에 대한 질문에는 도구 없이 답하세요.\n"
    # 답변은 스트리밍으로 나간다. 도구를 부르기 전에 쓴 문장은 화면에 잠깐 떴다가
    # 지워지므로(§4-21) 애초에 안 쓰게 막는다. 도구 호출이 뒤따를지는 라운드가
    # 끝나야 알 수 있어서, 이미 생성된 서두는 취소하는 것 말고 방법이 없다.
    "도구를 부르기로 했다면 아무 말도 쓰지 말고 곧바로 도구만 호출하세요. "
    "'찾아보겠습니다', '알아볼게요', '검색해드릴게요' 같은 예고 문장은 쓰지 마세요. "
    "설명은 도구 결과를 받은 뒤 최종 답변에서만 하세요.\n"
    # 실측: '##' 헤더를 남발해 답변이 과하게 커지고, 후보를 10편씩 늘어놓았다.
    "\n답변 형식:\n"
    "- 채팅 답변입니다. '##' 같은 제목 서식은 쓰지 마세요. 강조는 **굵게**로 충분합니다.\n"
    # 굵게 표시가 깨지는 조건이 있다. 닫는 ** 바로 앞이 문장부호()나 》)이고
    # 바로 뒤가 한글이면, 마크다운이 그것을 닫는 표시로 인정하지 않아 별표가
    # 화면에 그대로 보인다. 실측: '**엑시트 (2019)**는' → `**`가 노출됨.
    # 연도를 굵게 밖으로 빼면 닫는 ** 앞이 글자라서 정상 렌더링된다.
    "- 굵게는 **제목**까지만 감싸고 연도는 밖에 두세요. "
    "`**엑시트**(2019)는`처럼 씁니다. "
    "`**엑시트 (2019)**는`처럼 닫는 별표 앞에 괄호가 오면 화면에 별표가 그대로 "
    "보이니 절대 그렇게 쓰지 마세요.\n"
    "- 영화 제목에 《》나 〈〉 같은 괄호는 쓰지 마세요.\n"
    "- 영화는 3~5편만 소개하세요. 도구가 더 많이 돌려줘도 가장 잘 맞는 것만 고르세요.\n"
    "- 편당 두세 문장이면 충분합니다. 줄거리를 그대로 옮기지 말고 왜 추천하는지 쓰세요.\n"
    "- 도구가 준 정보를 빠짐없이 나열하지 마세요. 질문에 답하는 데 필요한 것만 쓰세요."
)


def build_system_prompt(today: date | None = None) -> str:
    """현재 날짜를 넣은 시스템 프롬프트.

    매 요청마다 만든다. 모듈 로드 시점에 고정하면 서버가 며칠 떠 있을 때 날짜가
    낡는다.
    """
    today = today or date.today()
    return _SYSTEM_PROMPT_TEMPLATE.format(
        today=today.strftime("%Y년 %m월 %d일"),
        year=today.year,
        last_year=today.year - 1,
    )


_DISCONNECT_REASON = "클라이언트 연결이 끊겨 실행이 중단되었습니다."


class _RootRunRecorder(BaseCallbackHandler):
    """LangSmith 루트 run id만 받아 적는다.

    연결이 끊겼을 때 그 트레이스를 닫아주려면 id가 필요한데, 실행이 중간에
    사라지면 어디서도 알려주지 않는다. 시작할 때 미리 적어둔다.
    """

    def __init__(self) -> None:
        self.root_run_id = None

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        if parent_run_id is None and self.root_run_id is None:
            self.root_run_id = run_id


def close_abandoned_run(run_id) -> None:
    """중단된 실행의 LangSmith 트레이스를 닫는다.

    안 닫으면 대시보드에 'pending'으로 영원히 남는다. 클라이언트가 사라지면
    파이썬이 `GeneratorExit`를 던지는데, 이건 `Exception`이 아니라
    `BaseException`이라 LangChain의 오류 보고 경로를 타지 않아 종료 이벤트가
    영영 발생하지 않는다(실측: 루트·agent·ChatOpenAI 3개가 pending으로 남음).

    실패해도 조용히 넘어간다. 이미 끊어진 요청을 정리하는 중이라 여기서 예외를
    올려봐야 받아줄 곳이 없다.
    """
    if run_id is None or not os.getenv("LANGSMITH_API_KEY"):
        return
    try:
        from langsmith import Client

        Client().update_run(
            run_id,
            end_time=datetime.now(timezone.utc),
            error=_DISCONNECT_REASON,
        )
    except Exception:  # noqa: BLE001 - 정리 작업이 본 흐름을 방해하면 안 된다
        logger.warning("중단된 트레이스를 닫지 못했습니다 (run=%s)", run_id)


def _checkpointer():
    """멀티턴 대화 메모리.

    SqliteSaver.from_conn_string()은 컨텍스트 매니저를 반환하므로 그대로 쓰면 안 된다.
    """
    if settings.checkpointer == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(str(settings.checkpoint_db), check_same_thread=False)
        return SqliteSaver(conn)

    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()


def _turn_start(messages: list) -> int:
    """이번 턴의 시작 위치. 마지막 HumanMessage 다음부터가 이번 턴이다."""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return index + 1
    return 0


def collect_turn_tool_calls(messages: list) -> list[dict]:
    """이번 턴에 LLM이 부른 도구와 인자.

    라우팅 평가와 디버깅에 쓴다. 검색 품질보다 도구 선택이 맞았는지가 먼저이고,
    그건 인자까지 봐야 판단할 수 있다(예: '잔인하지 않은'이 vibe 문장에 남았는지
    max_violence로 옮겨졌는지).
    """
    calls: list[dict] = []
    for message in messages[_turn_start(messages) :]:
        if isinstance(message, AIMessage):
            calls.extend(message.tool_calls or [])
    return calls


def collect_turn_attributions(messages: list) -> list[str]:
    """이번 턴이 무엇을 보고 답했는지(TMDB·로컬 색인·웹·JustWatch).

    답변 텍스트나 호출 시도가 아니라 **성공한 ToolMessage artifact**에서 얻는다.
    도구 호출 ID로 원래 호출의 이름·인자와 연결하므로 실패·빈 결과에는 표기가
    붙지 않고, OTT 검색의 JustWatch 표기도 유지된다.
    """
    marks = set()
    calls_by_id: dict[str, dict] = {}
    for message in messages[_turn_start(messages) :]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                if call.get("id"):
                    calls_by_id[call["id"]] = call
            continue
        if not isinstance(message, ToolMessage) or not _artifact_succeeded(message):
            continue
        call = calls_by_id.get(message.tool_call_id)
        if call:
            marks.update(attributions_for(call.get("name", ""), call.get("args")))
    return [mark for mark in ATTRIBUTION_ORDER if mark in marks]


def _artifact_succeeded(message: ToolMessage) -> bool:
    """새 artifact와 기존 체크포인트의 목록 artifact를 함께 읽는다."""
    if getattr(message, "status", "success") == "error":
        return False
    artifact = message.artifact
    if isinstance(artifact, dict):
        return artifact.get("success") is True
    # D3 이전 체크포인트의 영화 도구 artifact는 영화 dict 목록이었다.
    return isinstance(artifact, list) and bool(artifact)


def _movie_sources(message: ToolMessage) -> list[dict]:
    if not _artifact_succeeded(message):
        return []
    artifact = message.artifact
    if isinstance(artifact, dict):
        sources = artifact.get("sources", [])
    else:
        sources = artifact
    return [source for source in sources if isinstance(source, dict)]


def _web_sources(message: ToolMessage) -> list[dict]:
    if not _artifact_succeeded(message) or not isinstance(message.artifact, dict):
        return []
    return [
        source
        for source in message.artifact.get("web_sources", [])
        if isinstance(source, dict) and source.get("url")
    ]


def collect_turn_sources(messages: list) -> list[dict]:
    """이번 턴에 성공한 도구가 반환한 영화만 구조적으로 모은다.

    체크포인터가 전체 대화를 보존하므로 그냥 순회하면 몇 턴 전 출처까지 딸려온다.
    뒤에서부터 훑어 이번 턴의 시작(마지막 HumanMessage)을 찾고, 거기서부터 정방향으로
    모은다. 역순으로 모아 마지막에 뒤집으면 한 도구가 돌려준 목록의 내부 순서까지
    뒤집혀서 순위가 거꾸로 표시된다.
    """
    sources: list[dict] = []
    seen: set[tuple] = set()
    for message in messages[_turn_start(messages) :]:
        if isinstance(message, ToolMessage):
            for source in _movie_sources(message):
                movie_id = int(source.get("movie_id") or 0)
                key = ("id", movie_id) if movie_id else (
                    "title-year",
                    source.get("title"),
                    source.get("year"),
                )
                if key not in seen:
                    seen.add(key)
                    sources.append(source)
    return sources


def collect_turn_web_sources(messages: list) -> list[dict]:
    """이번 턴에 성공한 웹 검색이 실제 반환한 제목·URL을 모은다."""
    sources: list[dict] = []
    seen_urls: set[str] = set()
    for message in messages[_turn_start(messages) :]:
        if not isinstance(message, ToolMessage):
            continue
        for source in _web_sources(message):
            url = source["url"]
            if url not in seen_urls:
                seen_urls.add(url)
                sources.append(source)
    return sources


class MovieRagGraph:
    """StateGraph를 1회 컴파일해 재사용하는 영화 도우미."""

    def __init__(self) -> None:
        self.llm = get_llm().bind_tools(TOOLS)
        self.app = self._build()

    def _agent(self, state: MessagesState) -> dict:
        system = SystemMessage(build_system_prompt())
        return {"messages": [self.llm.invoke([system] + state["messages"])]}

    def _build(self):
        builder = StateGraph(MessagesState)
        builder.add_node("agent", self._agent)
        builder.add_node("tools", ToolNode(TOOLS))
        builder.add_edge(START, "agent")
        # tools_condition: 마지막 AIMessage에 tool_calls가 있으면 "tools", 없으면 END.
        builder.add_conditional_edges("agent", tools_condition)
        builder.add_edge("tools", "agent")
        return builder.compile(checkpointer=_checkpointer())

    def _run(self, question: str, session_id: str | None, prefix: str) -> list:
        thread_id = session_id or f"{prefix}-{uuid.uuid4()}"
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": _RECURSION_LIMIT,
        }
        try:
            result = self.app.invoke(
                {"messages": [HumanMessage(content=question)]}, config
            )
        except GraphRecursionError:
            # 예외를 올리면 체크포인트에 최종 답변 없는 도구 메시지만 남아 다음
            # 턴이 그 잔여물로 답한다. 답변을 하나 붙여 턴을 정상 종료시킨다.
            logger.warning("도구 호출 한도 초과 (thread=%s, q=%r)", thread_id, question)
            self.app.update_state(
                config, {"messages": [AIMessage(content=_RECURSION_FALLBACK)]}
            )
            return list(self.app.get_state(config).values.get("messages", []))
        return result.get("messages", [])

    def answer(self, question: str, session_id: str | None = None) -> dict:
        """질문에 답한다. 같은 session_id는 이전 대화 상태를 이어받는다."""
        return self._to_result(self._run(question, session_id, "once"))

    def trace(self, question: str, session_id: str | None = None) -> dict:
        """answer()와 같되 이번 턴의 도구 호출까지 함께 돌려준다(평가·디버깅용)."""
        messages = self._run(question, session_id, "trace")
        return {
            **self._to_result(messages),
            "tool_calls": collect_turn_tool_calls(messages),
        }

    def stream_answer(
        self, question: str, session_id: str | None = None
    ) -> Iterator[dict]:
        """answer()와 같은 결과를 진행 상황과 함께 흘려보낸다.

        이벤트는 네 종류다.

        ============  ==========================================================
        ``status``    도구 실행 같은 중간 상태. 한 줄을 계속 교체해 보여주면 된다.
        ``token``     답변 텍스트 조각.
        ``reset``     지금까지 받은 ``token``을 버리라는 신호. 드물게만 온다.
        ``done``      확정된 ``answer``와 ``sources``. 항상 마지막에 한 번 온다.
        ============  ==========================================================

        **서두를 거르는 방법.** 모델은 도구를 부르기 전에 예고 문장을 먼저 쓴다
        (실측: "이제 첫 번째 영화의 상세 정보를 확인하겠습니다."). 그 서두는 답변이
        아니라서 화면에 남으면 안 되는데, 도구 호출이 뒤따를지는 라운드가 끝나야
        알 수 있다(§4-21). 두 겹으로 막는다.

        1. 시스템 프롬프트로 **서두를 아예 쓰지 말라**고 지시한다. 1라운드 흐름은
           이걸로 잡히지만, 도구 결과를 받은 뒤 다음 라운드에서 여전히 샌다.
        2. 앞 ``_PREAMBLE_HOLD_CHARS``자를 **쥐고 있다가** 넘어가면 그때 흘린다.
           도구 라운드로 밝혀지면 쥔 채로 버리므로 화면에 뜨지 않는다.

        ``reset``은 2가 뚫렸을 때(서두가 한도보다 길었을 때)만 나가는 안전망이다.

        출처는 성공한 ToolMessage artifact에서 모아 ``done``에 한 번에 싣는다.
        답변 문자열에서 제목이나 URL을 다시 추측하지 않는다.

        비동기로 만들지 않았다. ``SqliteSaver``가 async 메서드에서
        ``NotImplementedError``를 던지므로 ``CHECKPOINTER=sqlite``가 깨진다.
        동기 제너레이터는 Starlette가 스레드풀에서 돌려주므로 서버도 막지 않는다.
        """
        thread_id = session_id or f"stream-{uuid.uuid4()}"
        # 연결이 끊겼을 때 트레이스를 닫아주려면 루트 run id가 필요하다.
        recorder = _RootRunRecorder()
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": _RECURSION_LIMIT,
            "callbacks": [recorder],
        }
        # _to_result()를 그대로 재사용하려고 이번 턴 메시지를 모아둔다. 맨 앞의
        # HumanMessage가 _turn_start()의 기준점이 된다.
        turn: list = [HumanMessage(content=question)]
        held: list[str] = []  # 서두일지 답변일지 아직 못 가린 텍스트
        flushed = False  # 이번 라운드의 텍스트를 내보내기 시작했는가

        yield {"type": "status", "text": _THINKING_STATUS}
        try:
            for mode, payload in self.app.stream(
                {"messages": [turn[0]]},
                config,
                stream_mode=["updates", "messages"],
            ):
                if mode == "messages":
                    chunk, meta = payload
                    # tools 노드의 ToolMessage도 이 모드로 나온다. 도구 결과 원문을
                    # 답변인 양 흘리면 안 되므로 agent 노드만 받는다.
                    if meta.get("langgraph_node") != "agent":
                        continue
                    # .text는 텍스트 블록만 잇는다. 도구 인자가 실려오는 청크
                    # (partial_json)는 빈 문자열이 되어 저절로 걸러진다.
                    text = getattr(chunk, "text", "")
                    if not text:
                        continue
                    if flushed:
                        yield {"type": "token", "text": text}
                        continue
                    held.append(text)
                    if sum(map(len, held)) >= _PREAMBLE_HOLD_CHARS:
                        # 서두라기엔 너무 길다. 최종 답변으로 보고 흘리기 시작한다.
                        yield {"type": "token", "text": "".join(held)}
                        held.clear()
                        flushed = True
                    continue

                for node, update in payload.items():
                    if not isinstance(update, dict):
                        continue
                    messages = update.get("messages") or []
                    turn.extend(messages)
                    if node != "agent":
                        continue
                    names = [
                        call["name"]
                        for message in messages
                        for call in (getattr(message, "tool_calls", None) or [])
                    ]
                    if not names:
                        # 도구 호출이 없었으니 이 라운드가 최종 답변이다. 한도에
                        # 못 미쳐 쥐고 있던 나머지를 마저 내보낸다.
                        if held:
                            yield {"type": "token", "text": "".join(held)}
                            held.clear()
                            flushed = True
                        continue
                    # 도구 라운드였다. 쥐고 있던 서두는 조용히 버리고, 한도를 넘겨
                    # 이미 내보낸 게 있을 때만 취소를 통지한다.
                    held.clear()
                    if flushed:
                        yield {"type": "reset"}
                        flushed = False
                    yield {"type": "status", "text": tool_status(names)}
        except GeneratorExit:
            # 클라이언트가 사라져 이 제너레이터가 뜯기는 중이다. 여기서 닫아주지
            # 않으면 LangSmith에 끝나지 않는 트레이스가 남는다.
            logger.info("스트림이 중단됐습니다 (thread=%s)", thread_id)
            close_abandoned_run(recorder.root_run_id)
            raise
        except GraphRecursionError:
            logger.warning("도구 호출 한도 초과 (thread=%s, q=%r)", thread_id, question)
            self.app.update_state(
                config, {"messages": [AIMessage(content=_RECURSION_FALLBACK)]}
            )
            # 쥐고 있던 텍스트는 화면에 안 나갔으니 그냥 버리고, 나간 게 있을
            # 때만 취소를 통지한다.
            held.clear()
            if flushed:
                yield {"type": "reset"}
            yield {"type": "token", "text": _RECURSION_FALLBACK}
            yield {
                "type": "done",
                "answer": _RECURSION_FALLBACK,
                "sources": [],
                "web_sources": [],
                "attributions": [],
            }
            return

        # 토큰을 이어 붙이지 않고 최종 메시지에서 다시 뽑는다. 클라이언트가 놓친
        # reset 때문에 답변이 어긋나는 일이 없도록 done이 언제나 정본이다.
        yield {"type": "done", **self._to_result(turn)}

    @staticmethod
    def _to_result(messages: list) -> dict:
        answer = messages[-1].content if messages else ""
        if isinstance(answer, list):
            # 일부 제공자는 content를 블록 리스트로 준다. 텍스트 블록만 잇는다.
            answer = "".join(
                block.get("text", "")
                for block in answer
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return {
            "answer": answer,
            "sources": collect_turn_sources(messages),
            "web_sources": collect_turn_web_sources(messages),
            "attributions": collect_turn_attributions(messages),
        }
