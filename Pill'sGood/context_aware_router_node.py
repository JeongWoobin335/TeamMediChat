from qa_state import QAState
from answer_utils import generate_response_llm_from_prompt
import json
import re

def extract_json_from_response(response: str) -> dict:
    """
    LLM 응답에서 JSON 부분을 추출하는 함수
    """
    try:
        # 직접 JSON 파싱 시도
        return json.loads(response)
    except json.JSONDecodeError:
        # JSON 부분만 추출 시도
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # JSON 형식이 아닌 경우 키워드 기반 분석
        return analyze_response_by_keywords(response)

def analyze_response_by_keywords(response: str) -> dict:
    """
    LLM 응답을 키워드 기반으로 분석하여 라우팅 결정
    """
    response_lower = response.lower()
    
    # 라우팅 결정을 위한 키워드 매칭
    if any(word in response_lower for word in ["부작용", "효능", "효과", "정보"]):
        return {
            "routing_decision": "excel_search",
            "confidence": "medium",
            "reasoning": "키워드 기반 정보 검색 라우팅",
            "user_intent": "약품 정보 요청",
            "context_relevance": "정보 검색 관련 질문"
        }
    elif any(word in response_lower for word in ["연구", "데이터", "분석", "논문"]):
        return {
            "routing_decision": "pdf_search",
            "confidence": "medium",
            "reasoning": "키워드 기반 연구 자료 검색 라우팅",
            "user_intent": "연구 자료 요청",
            "context_relevance": "연구 자료 관련 질문"
        }
    elif any(word in response_lower for word in ["최신", "신약", "2024", "2023", "FDA"]):
        return {
            "routing_decision": "external_search",
            "confidence": "medium",
            "reasoning": "키워드 기반 최신 정보 검색 라우팅",
            "user_intent": "최신 정보 요청",
            "context_relevance": "최신 정보 관련 질문"
        }
    else:
        return {
            "routing_decision": "excel_search",
            "confidence": "low",
            "reasoning": "키워드 분석 실패로 인한 기본 라우팅",
            "user_intent": "일반 정보 요청",
            "context_relevance": "기본 정보 검색"
        }

def context_aware_router_node(state: QAState) -> QAState:
    """
    LLM이 직접 맥락을 이해하고 적절한 처리 경로를 결정하는 노드
    스마트 하이브리드 접근법으로 안정성과 정확성을 모두 확보
    """
    print("🧠 맥락 인식 라우터 노드 시작")
    
    # 현재 질문과 대화 맥락 수집
    current_query = state.get("query", "")
    conversation_context = state.get("conversation_context", "")
    user_context = state.get("user_context", "")
    category = state.get("category", "")
    
    # 1차: 빠른 패턴 매칭 (하드코딩)
    print("🔍 1차: 빠른 패턴 매칭 시작")
    pattern_result = quick_pattern_analysis(current_query, category)
    
    if pattern_result['confidence'] == 'high':
        print("✅ 높은 신뢰도 패턴 매칭으로 빠른 처리")
        state.update(pattern_result)
        return state
    
    # 2차: LLM 맥락 분석
    print("🧠 2차: LLM 맥락 분석 시작")
    llm_result = llm_context_analysis(current_query, conversation_context, user_context, category)
    
    # 3차: 결과 비교 및 최종 결정
    final_decision = compare_and_decide(pattern_result, llm_result)
    
    print(f"📊 최종 라우팅 결정:")
    print(f"  - 경로: {final_decision['route']}")
    print(f"  - 신뢰도: {final_decision['confidence']}")
    print(f"  - 방법: {final_decision['method']}")
    print(f"  - 판단 근거: {final_decision.get('reasoning', '')}")
    
    # 상태에 최종 결정 저장
    state.update(final_decision)
    
    return state

def quick_pattern_analysis(query: str, category: str) -> dict:
    """
    1차 필터링: 빠른 패턴 매칭 (하드코딩)
    """
    query_lower = query.lower()
    
    # 명확한 패턴들 (하드코딩)
    patterns = {
        "excel_search": [
            "부작용", "효능", "효과", "성분", "가격", "제조사", "보험", "급여",
            "복용", "투여", "섭취", "먹는법", "약물", "약품", "정보", "알려줘",
            "사용", "복용법", "용법", "어떻게"
        ],
        "pdf_search": [
            "연구", "논문", "임상", "시험", "데이터", "통계", "분석", "결과",
            "보고서", "문서", "자료", "논문", "연구결과"
        ],
        "external_search": [
            "최신", "신약", "2024", "2023", "FDA", "승인", "시판", "출시",
            "뉴스", "소식", "업데이트", "변경", "새로운", "최근"
        ]
    }
    
    # 패턴 매칭
    for route, keywords in patterns.items():
        for keyword in keywords:
            if keyword in query_lower:
                return {
                    "route": route,
                    "confidence": "high",
                    "method": "pattern_matching",
                    "matched_keyword": keyword,
                    "routing_decision": route,
                    "reasoning": f"키워드 '{keyword}' 매칭"
                }
    
    # 카테고리 기반 기본 라우팅
    if category:
        category_mapping = {
            "정보": "excel_search", 
            "연구": "pdf_search",
            "최신": "external_search"
        }
        if category in category_mapping:
            return {
                "route": category_mapping[category],
                "confidence": "medium",
                "method": "category_based",
                "matched_category": category,
                "routing_decision": category_mapping[category],
                "reasoning": f"카테고리 '{category}' 기반 라우팅"
            }
    
    return {
        "route": "excel_search",
        "confidence": "low",
        "method": "default",
        "reason": "패턴 매칭 실패",
        "routing_decision": "excel_search",
        "reasoning": "기본 라우팅"
    }

def llm_context_analysis(query: str, context: str, user_context: str, category: str) -> dict:
    """
    2차 분석: LLM 기반 맥락 이해 (재시도 로직 포함)
    """
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            print(f"🧠 LLM 맥락 분석 시도 {attempt + 1}/{max_retries}")
            
            context_prompt = f"""
당신은 의약품 상담 시스템의 라우팅 담당자입니다.
사용자의 질문과 맥락을 분석하여 가장 적절한 처리 경로를 결정해주세요.

**사용자 질문:**
{query}

**대화 맥락:**
{context[:500] if context else "없음"}

**사용자 맥락:**
{user_context[:300] if user_context else "없음"}

**질문 카테고리:**
{category if category else "미분류"}

**처리 경로 옵션:**
1. "excel_search" - 약품 정보, 부작용, 효능 등 기본 정보가 필요한 경우  
2. "pdf_search" - 연구 자료, 임상 데이터, 상세 분석이 필요한 경우
3. "external_search" - 최신 정보, 신약, 외부 소식이 필요한 경우

**분석 기준:**
- 사용자의 구체적인 의도 파악
- 이전 대화와의 연관성
- 필요한 정보의 종류와 깊이
- 맥락적 이해

**중요:** 반드시 JSON 형식으로만 응답해주세요.

JSON 형식:
{{
    "routing_decision": "처리_경로",
    "confidence": "high/medium/low",
    "reasoning": "판단 근거",
    "user_intent": "사용자 의도",
    "context_relevance": "맥락 관련성"
}}
"""
            
            response = generate_response_llm_from_prompt(
                prompt=context_prompt,
                temperature=0.1,
                max_tokens=400
            )
            
            # JSON 파싱 시도
            result = extract_json_from_response(response)
            
            if result and "routing_decision" in result:
                print("✅ LLM 맥락 분석 성공")
                return {
                    "route": result.get("routing_decision", "excel_search"),
                    "confidence": result.get("confidence", "medium"),
                    "method": "llm_analysis",
                    "reasoning": result.get("reasoning", ""),
                    "user_intent": result.get("user_intent", ""),
                    "context_relevance": result.get("context_relevance", ""),
                    "routing_decision": result.get("routing_decision", "excel_search")
                }
            else:
                print(f"⚠️ LLM 응답에서 유효한 라우팅 정보를 찾을 수 없음 (시도 {attempt + 1})")
                
        except Exception as e:
            print(f"❌ LLM 맥락 분석 시도 {attempt + 1} 실패: {e}")
    
    # 모든 시도 실패 시 폴백
    print("🔄 모든 LLM 분석 시도 실패, 폴백 시스템 사용")
    return llm_fallback_analysis(query, context, user_context, category)

def llm_fallback_analysis(query: str, context: str, user_context: str, category: str) -> dict:
    """
    LLM 분석 실패 시 폴백 분석
    """
    print("🔄 폴백 분석 시스템 실행")
    
    # 간단한 키워드 기반 분석
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["부작용", "효능", "정보", "알려줘"]):
        return {
            "route": "excel_search",
            "confidence": "low",
            "method": "fallback",
            "routing_decision": "excel_search",
            "reasoning": "폴백 키워드 분석"
        }
    elif any(word in query_lower for word in ["연구", "데이터", "분석", "논문"]):
        return {
            "route": "pdf_search",
            "confidence": "low",
            "method": "fallback",
            "routing_decision": "pdf_search",
            "reasoning": "폴백 키워드 분석"
        }
    else:
        return {
            "route": "excel_search",
            "confidence": "low",
            "method": "fallback",
            "routing_decision": "excel_search",
            "reasoning": "폴백 기본값"
        }

def compare_and_decide(pattern_result: dict, llm_result: dict) -> dict:
    """
    패턴 매칭과 LLM 분석 결과를 비교하여 최종 결정
    """
    print("⚖️ 결과 비교 및 최종 결정")
    
    # 신뢰도 가중치 계산
    confidence_weights = {"high": 3, "medium": 2, "low": 1}
    
    pattern_score = confidence_weights.get(pattern_result.get("confidence", "low"), 1)
    llm_score = confidence_weights.get(llm_result.get("confidence", "low"), 1)
    
    print(f"📊 점수 비교:")
    print(f"  - 패턴 매칭: {pattern_score}점 ({pattern_result.get('confidence', 'low')})")
    print(f"  - LLM 분석: {llm_score}점 ({llm_result.get('confidence', 'low')})")
    
    # 더 높은 신뢰도를 가진 결과 선택
    if pattern_score >= llm_score:
        print("✅ 패턴 매칭 결과 선택")
        return pattern_result
    else:
        print("✅ LLM 분석 결과 선택")
        return llm_result
