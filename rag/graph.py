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
import sqlite3
import uuid
from datetime import date

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
from rag.tools import TOOLS

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

_SYSTEM_PROMPT_TEMPLATE = (
    "당신은 영화 정보를 안내하는 한국어 도우미입니다.\n\n"
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
    # 실측: '##' 헤더를 남발해 답변이 과하게 커지고, 후보를 10편씩 늘어놓았다.
    "\n답변 형식:\n"
    "- 채팅 답변입니다. '##' 같은 제목 서식은 쓰지 마세요. 강조는 **굵게**로 충분합니다.\n"
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


def collect_turn_sources(messages: list) -> list[dict]:
    """이번 턴에 호출된 도구의 출처만 모은다.

    체크포인터가 전체 대화를 보존하므로 그냥 순회하면 몇 턴 전 출처까지 딸려온다.
    뒤에서부터 훑어 이번 턴의 시작(마지막 HumanMessage)을 찾고, 거기서부터 정방향으로
    모은다. 역순으로 모아 마지막에 뒤집으면 한 도구가 돌려준 목록의 내부 순서까지
    뒤집혀서 순위가 거꾸로 표시된다.
    """
    sources: list[dict] = []
    seen: set[tuple] = set()
    for message in messages[_turn_start(messages) :]:
        if isinstance(message, ToolMessage) and message.artifact:
            for source in message.artifact:
                key = (source.get("title"), source.get("year"))
                if key not in seen:
                    seen.add(key)
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

    @staticmethod
    def _used_sources(answer: str, sources: list[dict]) -> list[dict]:
        """답변이 실제로 언급한 영화만 남긴다.

        LLM은 후보를 넓게 훑고 그중 일부만 답변에 쓴다. 도구가 돌려준 것을 전부
        카드로 내보내면 "답변에 없는 영화가 출처로 뜨는" 상태가 된다(실측: 후보
        14편 수집, 답변엔 3편 언급).

        제목이 하나도 안 걸리면 필터를 적용하지 않는다. 도구는 분명히 결과를
        줬는데 카드가 0개가 되는 편이 더 나쁘다.
        """
        used = [s for s in sources if s.get("title") and s["title"] in answer]
        return used or sources

    @classmethod
    def _to_result(cls, messages: list) -> dict:
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
            "sources": cls._used_sources(answer, collect_turn_sources(messages)),
        }
