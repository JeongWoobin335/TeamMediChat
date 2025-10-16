from qa_state import QAState
from chat_session_manager import ChatSessionManager
from answer_utils import generate_response_llm_from_prompt
import re
import json

def extract_medicine_info_from_context(context: str) -> dict:
    """대화 맥락에서 약품 정보를 추출하는 함수 (LLM 기반)"""
    if not context:
        return {}
    
    # LLM에게 약품 정보 추출 요청
    extraction_prompt = f"""
당신은 의약품 정보 추출 전문가입니다.
다음 대화 맥락에서 언급된 약품들과 관련 정보를 추출해주세요.

**대화 맥락:**
{context}

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
            print("⚠️ 약품 정보 추출 결과를 JSON으로 파싱할 수 없음")
            return {}
            
    except Exception as e:
        print(f"❌ 약품 정보 추출 중 오류 발생: {e}")
        return {}

def remember_previous_context_node(state: QAState) -> QAState:
    """
    대화 세션 관리자를 통해 이전 대화 맥락을 활용하여 더 정확한 응답을 생성합니다.
    LLM 기반 맥락 이해로 하드코딩된 키워드 매칭을 대체합니다.
    """
    print("🧠 이전 대화 맥락 분석 노드 시작")
    
    # 이전 응답이 있다면 저장
    final_answer = state.get("final_answer")
    if final_answer:
        state["previous_context"] = final_answer
    
    # 대화 맥락 정보가 있다면 활용
    conversation_context = state.get("conversation_context", "")
    if conversation_context:
        print(f"🔍 대화 맥락 분석 중... (길이: {len(conversation_context)} 문자)")
        
        # LLM 기반 약품 정보 추출
        medicine_info = extract_medicine_info_from_context(conversation_context)
        if medicine_info:
            print(f"💊 대화 맥락에서 약품 정보 발견: {len(medicine_info)}개")
            state["extracted_medicines"] = medicine_info
        
        # 이전 대화 맥락을 previous_context에 추가
        if state.get("previous_context"):
            state["previous_context"] = f"{state['previous_context']}\n\n이전 대화 맥락:\n{conversation_context}"
        else:
            state["previous_context"] = f"이전 대화 맥락:\n{conversation_context}"
    
    # 사용자 질문 맥락도 활용
    user_context = state.get("user_context", "")
    if user_context:
        print(f"👤 사용자 질문 맥락: {user_context[:100]}...")
        if state.get("previous_context"):
            state["previous_context"] = f"{state['previous_context']}\n\n사용자 질문 맥락:\n{user_context}"
        else:
            state["previous_context"] = f"사용자 질문 맥락:\n{user_context}"
    
    # LLM 기반 맥락 분석으로 이전 대화 내용 질문 여부 판단
    current_query = state.get("query", "")
    if current_query and conversation_context:
        context_analysis_prompt = f"""
당신은 대화 맥락 분석 전문가입니다.
현재 질문이 이전 대화 내용을 참조하는 질문인지 판단해주세요.

**이전 대화 맥락:**
{conversation_context[:500]}

**현재 질문:**
{current_query}

**판단 기준:**
- 현재 질문이 이전 대화에서 언급된 내용을 참조하는지
- "그 약", "이거", "아까 말한" 같은 표현이 있는지
- 이전 대화와 자연스럽게 연결되는지

**중요: 코드 블록 없이 순수 JSON만 반환하세요!**

출력 형식:
{{
    "is_asking_about_previous": true,
    "reasoning": "현재 질문이 이전 대화를 참조하는 이유",
    "referenced_content": "참조된 구체적인 내용"
}}
"""
        
        try:
            response = generate_response_llm_from_prompt(
                prompt=context_analysis_prompt,
                temperature=0.1,
                max_tokens=400
            )
            
            # JSON 코드 블록 제거 (```json ... ``` 형태 처리)
            cleaned_response = response.strip()
            if cleaned_response.startswith('```'):
                # 첫 번째 줄 제거 (```json)
                lines = cleaned_response.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                # 마지막 줄 제거 (```)
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                cleaned_response = '\n'.join(lines).strip()
            
            # JSON 응답 파싱
            try:
                analysis_result = json.loads(cleaned_response)
                is_asking_about_previous = analysis_result.get("is_asking_about_previous", False)
                reasoning = analysis_result.get("reasoning", "")
                referenced_content = analysis_result.get("referenced_content", "")
                
                print(f"🧠 LLM 맥락 분석 결과:")
                print(f"  - 이전 대화 참조 질문: {is_asking_about_previous}")
                print(f"  - 판단 근거: {reasoning}")
                if referenced_content:
                    print(f"  - 참조된 내용: {referenced_content[:100]}...")
                
                state["is_asking_about_previous"] = is_asking_about_previous
                state["context_reasoning"] = reasoning
                
                # 약품 정보가 있다면 이를 활용할 수 있도록 표시
                if state.get("extracted_medicines") and is_asking_about_previous:
                    print(f"💊 추출된 약품 정보: {len(state['extracted_medicines'])}개")
                    
            except json.JSONDecodeError as e:
                print(f"⚠️ 맥락 분석 결과를 JSON으로 파싱할 수 없음: {e}")
                print(f"🔍 원본 응답 (처음 200자): {response[:200]}...")
                print(f"🔍 정리된 응답 (처음 200자): {cleaned_response[:200]}...")
                state["is_asking_about_previous"] = False
                
        except Exception as e:
            print(f"❌ 맥락 분석 중 오류 발생: {e}")
            state["is_asking_about_previous"] = False
    
    return state

