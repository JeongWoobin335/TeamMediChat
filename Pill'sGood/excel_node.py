from typing import List
from langchain_core.documents import Document
from retrievers import excel_retriever, excel_docs, product_names, product_names_normalized
from qa_state import QAState
from cache_manager import cache_manager
import re

def normalize(text: str) -> str:
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"[^\w가-힣]", "", text)
    return re.sub(r"\s+", "", text.strip().lower())

def keyword_search(query: str, docs: List[Document]) -> List[Document]:
    """키워드 기반 검색으로 정확한 매칭 수행"""
    query_lower = query.lower()
    matched_docs = []
    
    # 효능 관련 키워드 매핑 (더 포괄적으로)
    effect_keywords = {
        "피곤": ["피로", "육체피로", "기운", "활력", "원기", "체력", "지치", "무기력", "피곤", "피곤함", "피곤증", "피로증"],
        "감기": ["감기", "기침", "콧물", "해열", "열", "몸살", "감기증상"],
        "두통": ["두통", "머리", "편두통", "긴장성", "두통증"],
        "소화": ["소화", "위장", "속", "복통", "소화불량", "소화장애"],
        "불면": ["불면", "수면", "잠", "불안", "긴장", "수면장애"],
        "관절": ["관절", "근육", "통증", "염증", "부종", "관절통"]
    }
    
    # 쿼리에서 효능 키워드 추출
    detected_effects = []
    for effect, keywords in effect_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            detected_effects.append(effect)
    
    for doc in docs:
        content = doc.page_content.lower()
        score = 0
        
        # 효능 키워드 매칭
        for effect in detected_effects:
            if effect in content:
                score += 2
            # 관련 키워드도 확인
            for keyword in effect_keywords[effect]:
                if keyword in content:
                    score += 1
        
        # 약품명 정확 매칭 (높은 우선순위)
        if doc.metadata.get("제품명", "").lower() in query_lower:
            score += 5
        
        if score > 0:
            matched_docs.append((doc, score))
    
    # 점수순 정렬
    matched_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in matched_docs[:10]]  # 상위 10개 반환

def excel_search_node(state: QAState) -> QAState:
    query = state.get("cleaned_query") or state.get("normalized_query")
    if not query:
        state["excel_results"] = []
        return state

    # 사용법 관련 질문인지 감지
    usage_keywords = ["사용법", "복용법", "먹는법", "복용", "먹는", "복용량", "용량", "횟수", "하루", "일일"]
    is_usage_query = any(keyword in query.lower() for keyword in usage_keywords)

    # 캐시 확인
    cached_results = cache_manager.get_search_cache(query, "excel")
    if cached_results is not None:
        state["excel_results"] = cached_results
        return state

    try:
        # 1. 키워드 기반 검색 (정확한 매칭)
        keyword_results = keyword_search(query, excel_docs)
        
        # 2. 벡터 기반 검색 (의미적 유사도)
        normalized_query = normalize(query)
        vector_results: List[Document] = excel_retriever.invoke(normalized_query)
        
        # 3. 사용법 관련 질문인 경우 type: "usage" 문서 우선
        if is_usage_query:
            usage_docs = [doc for doc in vector_results if doc.metadata.get("type") == "usage"]
            main_docs = [doc for doc in vector_results if doc.metadata.get("type") == "main"]
            
            # usage 문서를 먼저, 그 다음 main 문서 순으로 정렬
            prioritized_results = usage_docs + main_docs
            vector_results = prioritized_results[:20]  # 상위 20개만 유지
        
        # 4. 결과 통합 (키워드 결과 우선)
        combined_results = []
        seen_products = set()
        
        # 키워드 검색 결과 먼저 추가
        for doc in keyword_results:
            product_name = doc.metadata.get("제품명", "")
            if product_name not in seen_products:
                combined_results.append(doc)
                seen_products.add(product_name)
        
        # 벡터 검색 결과 추가 (중복 제거)
        for doc in vector_results:
            product_name = doc.metadata.get("제품명", "")
            if product_name not in seen_products:
                combined_results.append(doc)
                seen_products.add(product_name)
        
        state["excel_results"] = combined_results
        # 캐시 저장
        cache_manager.save_search_cache(query, "excel", combined_results)
    except Exception as e:
        state["excel_results"] = []

    return state

