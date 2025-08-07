# medicine_related_filter_node.py

from qa_state import QAState
from retrievers import llm

def medicine_related_filter_node(state: QAState) -> QAState:
    """
    사용자의 질문이 약품 또는 의약품과 관련 있는지 여부를 판단하여
    is_medicine_related 필드에 저장합니다.
    """
    query = state["query"]

    prompt = f"""
다음 질문이 약품 또는 의약품과 관련 있나요?

[질문]
{query}

의약품, 복용, 성분, 효능, 부작용, 질병 치료 관련 질문이면 "예",
운동, 식단, 건강 상식 등 일반 질문이면 "아니오"로 답해주세요.

"예" 또는 "아니오"로만 답변하세요.
"""

    try:
        response = llm.invoke(prompt).content.strip().lower()
        is_related = response.startswith("예") or response.startswith("yes")
        state["is_medicine_related"] = is_related
    except Exception as e:
        state["is_medicine_related"] = None

    return state

