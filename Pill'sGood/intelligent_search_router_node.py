from qa_state import QAState
from answer_utils import generate_response_llm_from_prompt
import json

def intelligent_search_router_node(state: QAState) -> QAState:
    """
    LLM이 직접 검색 전략을 결정하는 노드
    하드코딩된 키워드 매칭 대신 LLM의 자연스러운 이해 능력 활용
    """
    print("🔍 지능형 검색 라우터 노드 시작")
    
    # 현재 상태 정보 수집
    current_query = state.get("query", "")
    conversation_context = state.get("conversation_context", "")
    category = state.get("category", "")
    excel_results = state.get("excel_results", [])
    pdf_results = state.get("pdf_results", [])
    
    # LLM에게 검색 전략 결정 요청
    search_strategy_prompt = f"""
당신은 의약품 정보 검색 시스템의 검색 전략 담당자입니다.
현재 상황을 분석하여 가장 적절한 검색 경로를 결정해야 합니다.

**현재 상황:**
- 사용자 질문: {current_query}
- 대화 맥락: {conversation_context[:300] if conversation_context else "없음"}
- 질문 카테고리: {category if category else "미분류"}
- Excel 검색 결과: {len(excel_results)}개
- PDF 검색 결과: {len(pdf_results)}개

**검색 경로 옵션:**
1. "rerank" - 충분한 정보가 수집되어 리랭킹으로 넘어가는 경우
2. "pdf_search" - Excel에서 정보를 찾지 못해 PDF 검색이 필요한 경우
3. "external_search" - PDF에서도 정보를 찾지 못해 외부 검색이 필요한 경우

**판단 기준:**
- 현재 수집된 정보가 사용자 질문에 충분한지
- 어떤 추가 검색이 필요한지
- 가장 효율적인 검색 순서는 무엇인지

JSON 형식으로 응답해주세요:
{{
    "search_decision": "검색_경로",
    "reasoning": "판단 근거",
    "current_info_status": "현재 정보 상태",
    "next_search_strategy": "다음 검색 전략"
}}
"""
    
    try:
        # LLM이 검색 전략을 결정
        response = generate_response_llm_from_prompt(
            prompt=search_strategy_prompt,
            temperature=0.1,
            max_tokens=400
        )
        
        # JSON 응답 파싱
        try:
            strategy_result = json.loads(response)
            search_decision = strategy_result.get("search_decision", "rerank")
            reasoning = strategy_result.get("reasoning", "분석 실패")
            current_info_status = strategy_result.get("current_info_status", "")
            next_search_strategy = strategy_result.get("next_search_strategy", "")
            
            print(f"🔍 LLM 검색 전략 분석 결과:")
            print(f"  - 검색 결정: {search_decision}")
            print(f"  - 판단 근거: {reasoning}")
            print(f"  - 현재 정보 상태: {current_info_status}")
            print(f"  - 다음 검색 전략: {next_search_strategy}")
            
            # 상태에 검색 결정과 분석 결과 저장
            state["search_decision"] = search_decision
            state["search_reasoning"] = reasoning
            state["search_strategy"] = next_search_strategy
            
        except json.JSONDecodeError:
            print("⚠️ LLM 응답을 JSON으로 파싱할 수 없음, 기본값 사용")
            # JSON 파싱 실패 시 기본 검색 전략
            if len(excel_results) > 0:
                state["search_decision"] = "rerank"
            elif len(pdf_results) > 0:
                state["search_decision"] = "rerank"
            else:
                state["search_decision"] = "external_search"
            state["search_reasoning"] = "JSON 파싱 실패로 인한 기본 검색 전략"
    
    except Exception as e:
        print(f"❌ LLM 검색 전략 분석 중 오류 발생: {e}")
        # 오류 발생 시 기본 검색 전략
        if len(excel_results) > 0:
            state["search_decision"] = "rerank"
        elif len(pdf_results) > 0:
            state["search_decision"] = "rerank"
        else:
            state["search_decision"] = "external_search"
        state["search_reasoning"] = f"오류 발생으로 인한 기본 검색 전략: {str(e)}"
    
    return state
