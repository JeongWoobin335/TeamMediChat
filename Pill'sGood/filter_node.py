from qa_state import QAState
from langchain_core.documents import Document
from typing import List

def filter_relevant_node(state: QAState) -> QAState:
    """
    리랭크된 문서 중 질문과 요청 항목에 관련된 문서만 relevant_docs에 저장합니다.
    """
    docs: List[Document] = state.get("reranked_docs") or []
    if not docs:
        state["relevant_docs"] = []
        return state

    # 👇 약품명 비교를 위해 기준 이름 정리
    target_name = (state.get("normalized_query") or state.get("cleaned_query") or "").replace(" ", "").lower()
    requested_fields = state.get("requested_fields") or ["효능", "부작용", "사용법"]

    def is_doc_relevant(doc: Document, query: str, requested_fields: List[str]) -> bool:
        text = doc.page_content.lower()
        product_name = doc.metadata.get("제품명", "").replace(" ", "").lower()

        # ✅ 약품명이 일치하지 않으면 relevance 0점
        if product_name != target_name:
            return False

        score = 0
        if any(field.lower() in text for field in requested_fields):
            score += 1
        if any(token in text for token in query.lower().split()):
            score += 1

        return score >= 0

    # 🔍 실제 filtering
    filtered_docs = [
        doc for doc in docs if is_doc_relevant(doc, state["query"], requested_fields)
    ]

    state["relevant_docs"] = filtered_docs
    return state
