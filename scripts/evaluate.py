"""
LangSmith 평가 스크립트 (요구사항 3).

흐름:
1. eval/dataset.jsonl 을 읽어 LangSmith Dataset 을 생성/동기화한다.
2. RAG 파이프라인을 target 으로 삼아 모든 예제에 대해 실행한다(자동 추적).
3. 평가기 2종으로 채점한다.
   - keyword_recall : 기대 답변의 핵심어가 모델 답변에 얼마나 포함됐는지(0~1).
     baseline 의 단순 split() 방식을 한국어 조사 제거 + 정규화로 개선했다.
   - llm_judge      : LLM 이 기대 답변과 모델 답변의 의미 일치도를 0/0.5/1 로 채점.

사용법:
    python -m scripts.evaluate
"""
from __future__ import annotations

import json
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client
from langsmith.evaluation import evaluate

from rag.config import settings
from rag.graph import MovieRagGraph
from rag.providers import get_llm

# 핵심어 추출 시 제거할 흔한 한국어 조사/어미 꼬리.
_PARTICLES = ("은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "도", "로", "으로", "다", "이다")
_TOKEN_RE = re.compile(r"[A-Za-z가-힣0-9]+")


def _keywords(text: str) -> list[str]:
    """기대 답변에서 길이 2자 이상 토큰을 핵심어로 뽑고 조사 꼬리를 제거한다."""
    words = []
    for w in _TOKEN_RE.findall(text):
        for p in _PARTICLES:
            if len(w) > len(p) + 1 and w.endswith(p):
                w = w[: -len(p)]
                break
        if len(w) >= 2:
            words.append(w)
    # 중복 제거(순서 유지)
    return list(dict.fromkeys(words))


def keyword_recall(run, example) -> dict:
    """기대 답변 핵심어가 모델 답변에 포함된 비율(0~1)."""
    pred = (run.outputs or {}).get("answer", "")
    expected = (example.outputs or {}).get("answer", "")
    keys = _keywords(expected)
    if not keys:
        return {"key": "keyword_recall", "score": 0.0}
    hit = sum(1 for k in keys if k in pred)
    score = hit / len(keys)
    return {
        "key": "keyword_recall",
        "score": score,
        "comment": f"{hit}/{len(keys)} 핵심어 포함: {keys}",
    }


_JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 답변 품질을 평가하는 채점자입니다.\n"
            "기대 답변(reference)과 모델 답변(prediction)을 비교해, 의미가 일치하면 1,\n"
            "부분적으로만 일치하면 0.5, 무관하거나 틀리면 0을 매기세요.\n"
            "반드시 첫 줄에 0, 0.5, 1 중 하나의 숫자만 쓰고, 둘째 줄부터 짧은 이유를 적으세요.",
        ),
        (
            "human",
            "질문: {question}\n\n기대 답변: {reference}\n\n모델 답변: {prediction}",
        ),
    ]
)


def make_llm_judge():
    """LLM-as-judge 평가기를 생성한다(LLM 1회 로드 후 재사용)."""
    judge_chain = _JUDGE_PROMPT | get_llm() | StrOutputParser()

    def llm_judge(run, example) -> dict:
        reply = judge_chain.invoke(
            {
                "question": example.inputs["question"],
                "reference": (example.outputs or {}).get("answer", ""),
                "prediction": (run.outputs or {}).get("answer", ""),
            }
        )
        first = reply.strip().splitlines()[0].strip() if reply.strip() else "0"
        try:
            score = float(re.findall(r"[0-9.]+", first)[0])
        except (ValueError, IndexError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        return {"key": "llm_judge_semantic_match", "score": score, "comment": reply}

    return llm_judge


def sync_dataset(client: Client):
    """jsonl 평가셋을 읽어 LangSmith Dataset 을 생성/동기화한다."""
    rows = [json.loads(line) for line in settings.eval_file.read_text("utf-8").splitlines() if line.strip()]

    existing = list(client.list_datasets(dataset_name=settings.dataset_name))
    if existing:
        dataset = existing[0]
        print(f"[eval] 기존 Dataset 사용: {dataset.name} ({dataset.id})")
        have = {ex.inputs["question"] for ex in client.list_examples(dataset_id=dataset.id)}
        new = [r for r in rows if r["question"] not in have]
        if new:
            client.create_examples(
                dataset_id=dataset.id,
                inputs=[{"question": r["question"]} for r in new],
                outputs=[{"answer": r["answer"]} for r in new],
            )
            print(f"[eval] 신규 예제 {len(new)}건 추가")
    else:
        dataset = client.create_dataset(
            dataset_name=settings.dataset_name,
            description="영화 정보 RAG 답변 품질 평가셋",
        )
        client.create_examples(
            dataset_id=dataset.id,
            inputs=[{"question": r["question"]} for r in rows],
            outputs=[{"answer": r["answer"]} for r in rows],
        )
        print(f"[eval] 새 Dataset 생성 + 예제 {len(rows)}건 추가: {dataset.id}")
    return dataset


def main():
    client = Client()
    sync_dataset(client)

    graph = MovieRagGraph()

    def target(inputs: dict) -> dict:
        # 평가는 단일 턴 → session_id 없이 매번 새 스레드로 실행.
        result = graph.answer(inputs["question"])
        return {"answer": result["answer"]}

    result = evaluate(
        target,
        data=settings.dataset_name,
        evaluators=[keyword_recall, make_llm_judge()],
        experiment_prefix="movie-rag-langgraph",
    )
    print(result)


if __name__ == "__main__":
    main()
