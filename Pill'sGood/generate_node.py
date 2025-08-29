from qa_state import QAState
from answer_utils import generate_response_llm_from_prompt
from langchain_core.documents import Document
import re
import json
from typing import List

def contains_exact_product_name(doc: Document, product_name: str) -> bool:
    return re.search(rf"\[제품명\]:\s*{re.escape(product_name)}\b", doc.page_content) is not None

def extract_medicine_from_context(conversation_context: str) -> list:
    """대화 맥락에서 약품 정보를 추출하는 함수 (LLM 기반)"""
    if not conversation_context:
        return []
    
    # LLM에게 약품명 추출 요청
    extraction_prompt = f"""
다음 대화 맥락에서 언급된 약품명들을 추출해주세요.

**대화 맥락:**
{conversation_context}

**응답 형식:**
약품명만 쉼표로 구분하여 나열해주세요.
예: 아스피린, 이부프로펜, 파라세타몰
"""
    
    try:
        response = generate_response_llm_from_prompt(
            prompt=extraction_prompt,
            temperature=0.1,
            max_tokens=200
        )
        
        # 쉼표로 구분된 약품명들을 리스트로 변환
        medicines = [name.strip() for name in response.split(',') if name.strip()]
        return medicines
        
    except Exception as e:
        print(f"❌ 약품명 추출 중 오류 발생: {e}")
        return []

def extract_medicine_details_from_context(conversation_context: str) -> dict:
    """대화 맥락에서 약품의 상세 정보를 추출하는 함수 (LLM 기반)"""
    if not conversation_context:
        return {}
    
    # LLM에게 상세 정보 추출 요청
    extraction_prompt = f"""
다음 대화 맥락에서 언급된 약품들의 상세 정보를 추출해주세요.

**대화 맥락:**
{conversation_context}

**추출할 정보:**
- 약품명
- 효능/효과
- 부작용
- 사용법
- 주의사항

JSON 형식으로 응답해주세요:
{{
    "medicines": [
        {{
            "name": "약품명",
            "effects": ["효능1", "효능2"],
            "side_effects": ["부작용1", "부작용2"],
            "usage": "사용법",
            "precautions": ["주의사항1", "주의사항2"]
        }}
    ]
}}
"""
    
    try:
        response = generate_response_llm_from_prompt(
            prompt=extraction_prompt,
            temperature=0.1,
            max_tokens=800
        )
        
        # JSON 응답 파싱
        try:
            result = json.loads(response)
            return result.get("medicines", {})
        except json.JSONDecodeError:
            print("⚠️ 상세 정보 추출 결과를 JSON으로 파싱할 수 없음")
            return {}
            
    except Exception as e:
        print(f"❌ 상세 정보 추출 중 오류 발생: {e}")
        return {}

def extract_medicine_context(conversation_context: str, medicine_name: str) -> str:
    """대화 맥락에서 특정 약품 주변의 문맥을 추출"""
    # config에서 설정된 최대 문맥 길이 사용
    context_length = 100 # 예시 값, 실제 사용 시 환경 변수나 설정에서 가져옴
    
    pattern = rf".{{0,{context_length}}}{re.escape(medicine_name)}.{{0,{context_length}}}"
    matches = re.findall(pattern, conversation_context, re.IGNORECASE)
    
    if matches:
        return matches[0]
    return ""

def extract_effect_from_context(context: str) -> str:
    """문맥에서 효능 정보를 추출"""
    effect_keywords = ["효능", "효과", "도움", "개선", "완화", "치료", "예방"]
    
    for keyword in effect_keywords:
        if keyword in context:
            # 키워드 주변 문맥 추출
            start = max(0, context.find(keyword) - 50)
            end = min(len(context), context.find(keyword) + 100)
            return context[start:end].strip()
    
    return "효능 정보를 찾을 수 없습니다"

def extract_side_effects_from_context(context: str) -> str:
    """문맥에서 부작용 정보를 추출"""
    side_effect_keywords = ["부작용", "주의사항", "경고", "증상", "불편"]
    
    for keyword in side_effect_keywords:
        if keyword in context:
            start = max(0, context.find(keyword) - 50)
            end = min(len(context), context.find(keyword) + 100)
            return context[start:end].strip()
    
    return "부작용 정보를 찾을 수 없습니다"

def extract_usage_from_context(context: str) -> str:
    """문맥에서 사용법 정보를 추출"""
    usage_keywords = ["사용법", "복용법", "용법", "복용", "섭취", "투여"]
    
    for keyword in usage_keywords:
        if keyword in context:
            start = max(0, context.find(keyword) - 50)
            end = min(len(context), context.find(keyword) + 100)
            return context[start:end].strip()
    
    return "사용법 정보를 찾을 수 없습니다"

def generate_final_answer_node(state: QAState) -> QAState:
    print("🎯 최종 답변 생성 노드 시작")
    print(f"📊 상태 정보:")
    print(f"  - final_answer: {state.get('final_answer', '없음')}")
    print(f"  - recommendation_answer: {state.get('recommendation_answer', '없음')}")
    print(f"  - relevant_docs: {len(state.get('relevant_docs', []))}개")
    print(f"  - external_parsed: {state.get('external_parsed', '없음')}")
    print(f"  - sns_results: {len(state.get('sns_results', []))}개")
    print(f"  - sns_analysis: {state.get('sns_analysis', '없음')}")
    print(f"  - conversation_context: {state.get('conversation_context', '없음')[:100] if state.get('conversation_context') else '없음'}...")
    print(f"  - user_context: {state.get('user_context', '없음')}")
    
    # ✅ 이미 final_answer가 설정된 경우 (최신 정보 요청 등)
    if state.get("final_answer"):
        print("✅ 이미 final_answer가 설정되어 있음")
        return state

    # ✅ 병력 기반 추천이 있는 경우 먼저 반환하고 종료 (우선순위 1순위)
    if state.get("recommendation_answer"):
        print("✅ recommendation_answer 사용")
        state["final_answer"] = state["recommendation_answer"]
        return state

    # 🔍 LLM 기반 맥락 분석 및 답변 생성
    conversation_context = state.get("conversation_context", "")
    user_context = state.get("user_context", "")
    current_query = state.get("query", "")
    relevant_docs = state.get("relevant_docs", [])
    
    if conversation_context and current_query:
        print("🔄 LLM이 맥락을 분석하여 답변 생성")
        
        # LLM에게 맥락 기반 답변 생성 요청
        context_aware_prompt = f"""
당신은 의약품 상담 전문가입니다.
사용자의 질문과 대화 맥락을 분석하여 자연스럽고 유용한 답변을 생성해주세요.

**사용자 질문:**
{current_query}

**대화 맥락:**
{conversation_context[:800] if conversation_context else "없음"}

**사용자 질문 맥락:**
{user_context[:400] if user_context else "없음"}

**검색된 문서 정보:**
{len(relevant_docs)}개의 관련 문서가 있습니다.

**답변 요구사항:**
1. 사용자의 질문에 직접적으로 답변
2. 이전 대화 맥락과 자연스럽게 연결
3. "그 약", "이거" 같은 대명사가 있다면 맥락에 맞게 해석
4. 자연스럽고 대화적인 톤으로 응답
5. 필요시 추가 질문을 유도하는 방식으로 마무리

**주의사항:**
- 하드코딩된 템플릿이나 키워드 매칭을 사용하지 말 것
- 맥락을 자연스럽게 이해하고 응답할 것
- 사용자가 원하는 정보를 정확히 파악할 것
"""
        
        try:
            # LLM이 맥락을 이해하고 자연스러운 답변 생성
            final_answer = generate_response_llm_from_prompt(
                prompt=context_aware_prompt,
                temperature=0.7,  # 자연스러운 대화를 위해 적당한 temperature
                max_tokens=1000
            )
            
            state["final_answer"] = final_answer
            print("✅ LLM 기반 맥락 인식 답변 생성 완료")
            
        except Exception as e:
            print(f"❌ LLM 답변 생성 중 오류 발생: {e}")
            # 오류 발생 시 기본 답변
            state["final_answer"] = f"죄송합니다. 답변을 생성하는 중 오류가 발생했습니다: {str(e)}"
    
    else:
        print("❌ 대화 맥락 정보가 부족하여 답변을 생성할 수 없음")
        state["final_answer"] = "죄송합니다. 대화 맥락 정보가 부족하여 적절한 답변을 생성할 수 없습니다."
    
    return state
