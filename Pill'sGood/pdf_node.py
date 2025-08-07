from qa_state import QAState
from typing import List
from langchain_core.documents import Document
from retrievers import pdf_retriever  # 미리 만들어둔 pdf_retriever import
from cache_manager import cache_manager

def pdf_search_node(state: QAState) -> QAState:
    query = state.get("cleaned_query") or state.get("normalized_query")
    if not query:
        state["pdf_results"] = []
        return state

    # 캐시 확인
    cached_results = cache_manager.get_search_cache(query, "pdf")
    if cached_results is not None:
        state["pdf_results"] = cached_results
        return state

    try:
        results: List[Document] = pdf_retriever.invoke(query)

        # ✅ 제품명 메타데이터 보정
        for doc in results:
            if "제품명" not in doc.metadata or not doc.metadata["제품명"]:
                doc.metadata["제품명"] = query  # 최소한 검색어라도 넣어줌

        state["pdf_results"] = results
        # 캐시 저장
        cache_manager.save_search_cache(query, "pdf", results)
    except Exception as e:
        state["pdf_results"] = []

    return state

