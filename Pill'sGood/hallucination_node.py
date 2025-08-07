from qa_state import QAState
from langchain_core.documents import Document
from typing import List
from retrievers import llm

def hallucination_check_node(state: QAState) -> QAState:
    """
    LLM을 사용해 답변이 문서에 기반했는지 여부를 판별하고, 환각 여부를 판단합니다.
    """
    docs: List[Document] = state.get("relevant_docs") or []
    answer = state.get("final_answer")
    query = state.get("query")

    if not docs or not answer or not query:
        state["hallucination_flag"] = None
        return state

    # 문서 내용 압축
    context_text = "\n".join([doc.page_content for doc in docs])[:3000]

    prompt = f"""
다음은 사용자 질문, 모델이 생성한 답변, 그리고 관련 문서입니다.

[질문]
{query}

[모델이 생성한 답변]
{answer}

[관련 문서]
{context_text}

이 답변이 문서 내용에 기반하고 있나요? 
"예" 또는 "아니오"만 출력하세요. 추가 설명 없이.
"""

    try:
        response = llm.invoke(prompt).content.strip().lower()
        hallucinated = response.startswith("아니오") or response.startswith("no")
        state["hallucination_flag"] = hallucinated
    except Exception as e:
        state["hallucination_flag"] = None

    return state
