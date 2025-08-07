from qa_state import QAState
from retrievers import llm

def requery_node(state: QAState) -> QAState:
    """
    문서가 부족하거나 환각이 감지된 경우, 원 질문을 개선해서 re_query에 저장합니다.
    """
    query = state.get("query")
    hallucinated = state.get("hallucination_flag")
    relevant_docs = state.get("relevant_docs") or []

    # 재질문 조건 확인
    if hallucinated is not True and len(relevant_docs) > 0:
        state["re_query"] = None
        return state

    # LLM에게 재질문 요청
    prompt = f"""
아래 질문이 문서에서 원하는 정보를 찾지 못했습니다.
사용자가 원하는 정보를 더 명확히 찾을 수 있도록, 이 질문을 구체적이고 명확하게 다시 작성해 주세요.

원래 질문: {query}

개선된 질문:
"""

    try:
        new_query = llm.invoke(prompt).content.strip()
        state["re_query"] = new_query
    except Exception as e:
        state["re_query"] = None

    return state
