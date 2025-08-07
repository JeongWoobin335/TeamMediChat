from qa_state import QAState
from retrievers import search_agent, summarize_structured_json  # 외부 검색기 & 요약기
from cache_manager import cache_manager

def external_search_node(state: QAState) -> QAState:
    """
    외부 검색 도구를 사용하여 약품 관련 정보를 수집하고 요약합니다.
    결과는 raw 텍스트와 JSON 구조 모두로 저장됩니다.
    """
    query = state.get("cleaned_query") or state.get("normalized_query")
    if not query:
        return state

    # 캐시 확인
    cached_raw = cache_manager.get_search_cache(query, "external_raw")
    cached_parsed = cache_manager.get_search_cache(query, "external_parsed")
    
    if cached_raw is not None and cached_parsed is not None:
        state["external_raw"] = cached_raw[0].page_content if cached_raw else None
        state["external_parsed"] = cached_parsed[0].page_content if cached_parsed else None
        return state

    try:
        # 최신 정보 요청인 경우 외부 검색 생략
        category = state.get("category", "")
        if category == "최신 약품" or any(keyword in query.lower() for keyword in ["2024", "2023", "새로", "신약", "fda", "승인"]):
            state["external_raw"] = None
            state["external_parsed"] = None
            return state
        
        search_query = f"site:mfds.go.kr OR site:health.kr {query}"
        raw_result = search_agent.run(search_query)
        parsed_result = summarize_structured_json(raw_result)

        state["external_raw"] = raw_result
        state["external_parsed"] = parsed_result

        # 캐시 저장
        from langchain_core.documents import Document
        cache_manager.save_search_cache(query, "external_raw", [Document(page_content=raw_result)])
        cache_manager.save_search_cache(query, "external_parsed", [Document(page_content=str(parsed_result))])

    except Exception as e:
        state["external_raw"] = None
        state["external_parsed"] = None

    return state
