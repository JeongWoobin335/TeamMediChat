from qa_state import QAState
from retrievers import search_agent, summarize_structured_json  # 외부 검색기 & 요약기
from cache_manager import cache_manager
from answer_utils import generate_response_llm_from_prompt
import json

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
        state["external_parsed"] = cached_raw[0].page_content if cached_parsed else None
        return state

    try:
        # LLM이 최신 정보 요청인지 판단
        category = state.get("category", "")
        current_query = state.get("query", "")
        
        latest_info_prompt = f"""
당신은 의약품 정보 요청 분석 전문가입니다.
사용자의 질문이 최신 약품 정보나 경험담을 요청하는 것인지 판단해주세요.

**사용자 질문:**
{current_query}

**질문 카테고리:**
{category if category else "미분류"}

**판단 기준:**
- 최신 약품 정보를 요청하는지 (2024, 2023, 신약, FDA 승인 등)
- 실제 사용자 경험이나 후기를 원하는지
- 최신 연구 결과나 임상시험 정보를 원하는지

JSON 형식으로 응답해주세요:
{{
    "is_latest_info_request": true/false,
    "reasoning": "판단 근거"
}}
"""
        
        try:
            response = generate_response_llm_from_prompt(
                prompt=latest_info_prompt,
                temperature=0.1,
                max_tokens=200
            )
            
            # JSON 응답 파싱
            try:
                analysis_result = json.loads(response)
                is_latest_info_request = analysis_result.get("is_latest_info_request", False)
                reasoning = analysis_result.get("reasoning", "")
                
                print(f"🔍 LLM 최신 정보 요청 분석:")
                print(f"  - 최신 정보 요청: {is_latest_info_request}")
                print(f"  - 판단 근거: {reasoning}")
                
                # 최신 정보 요청인 경우 외부 검색 생략
                if is_latest_info_request:
                    print("📡 최신 정보 요청으로 인식하여 외부 검색 생략")
                    state["external_raw"] = None
                    state["external_parsed"] = None
                    return state
                    
            except json.JSONDecodeError:
                print("⚠️ 최신 정보 요청 분석 결과를 JSON으로 파싱할 수 없음")
                # 파싱 실패 시 기본 검색 진행
        
        except Exception as e:
            print(f"❌ 최신 정보 요청 분석 중 오류 발생: {e}")
            # 오류 발생 시 기본 검색 진행
        
        # 일반 검색 진행
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
        print(f"❌ 외부 검색 중 오류 발생: {e}")
        state["external_raw"] = None
        state["external_parsed"] = None

    return state
