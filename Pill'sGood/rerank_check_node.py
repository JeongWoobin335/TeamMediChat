from qa_state import QAState
from retrievers import compressor  # CrossEncoder 기반 문서 압축기
from langchain_core.documents import Document
from typing import List
from answer_utils import generate_response_llm_from_prompt
import re
import json

def normalize(text: str) -> str:
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"[^\w가-힣]", "", text)
    return re.sub(r"\s+", "", text.strip().lower())

def contains_product_name(doc: Document, product_name: str) -> bool:
    return normalize(product_name) == normalize(doc.metadata.get("제품명", ""))

def rerank_node(state: QAState) -> QAState:
    """
    원래 시스템의 리랭킹 노드
    검색 결과를 처리하고 관련성 높은 문서를 선택
    """
    print("🔄 리랭킹 노드 시작")
    
    all_docs: List[Document] = []
    if state.get("pdf_results"):
        all_docs.extend(state["pdf_results"])
    if state.get("excel_results"):
        all_docs.extend(state["excel_results"])
    if state.get("sns_results"):
        all_docs.extend(state["sns_results"])

    if not all_docs:
        print("⚠️ 검색 결과가 없음")
        state["reranked_docs"] = []
        state["relevant_docs"] = []
        return state

    try:
        query = state.get("query", "")
        product_name = state.get("normalized_query") or state.get("cleaned_query")
        product_name = normalize(product_name or "")

        # 제품명 기반 필터링 (하드코딩)
        excel_docs = state.get("excel_results", [])
        excel_matched = [doc for doc in excel_docs if contains_product_name(doc, product_name)]

        if excel_matched:
            print("✅ 제품명 기반 매칭 성공")
            state["relevant_docs"] = excel_matched[:3]
            state["reranked_docs"] = []
            return state

        # 전체 문서 리랭킹
        print("🔄 전체 문서 리랭킹 실행")
        reranked = compressor.compress_documents(all_docs, query=query)
        state["reranked_docs"] = reranked

        # 제품명 기반 추가 필터링
        filtered = [doc for doc in reranked if contains_product_name(doc, product_name)]

        if filtered:
            # 중복 제거
            deduped = []
            seen = set()
            for doc in filtered + reranked:
                key = doc.page_content[:100]
                if key not in seen:
                    deduped.append(doc)
                    seen.add(key)
                if len(deduped) >= 3:
                    break
            state["relevant_docs"] = deduped
        else:
            state["relevant_docs"] = reranked[:3]

        print(f"✅ 검색 결과 처리 완료: {len(state['relevant_docs'])}개 문서")

    except Exception as e:
        print(f"❌ 검색 결과 처리 중 오류 발생: {e}")
        state["reranked_docs"] = []
        state["relevant_docs"] = []

    return state
