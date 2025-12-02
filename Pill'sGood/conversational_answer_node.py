# conversational_answer_node.py - GPT 기반 최종 답변 자연스럽게 재구성 노드

from qa_state import QAState
from answer_utils import generate_response_llm_from_prompt
from config import PromptConfig
from prompt_utils import (
    get_role_definition, get_section_structure, get_common_instructions,
    get_conversational_tone_examples
)

def conversational_answer_node(state: QAState) -> QAState:
    """
    GPT를 사용하여 최종 답변을 자연스럽고 대화형으로 재구성합니다.
    - 구조화된 답변을 자연스러운 대화로 변환
    - 이전 대화 맥락과 자연스럽게 연결
    - 연계적인 질문에 대화하듯이 답변
    
    🚀 성능 최적화: 연속 질문일 때만 재구성 (첫 질문은 스킵)
    """
    print("💬 대화형 답변 재구성 노드 시작")
    
    # 신약 관련 질문은 재구성 건너뛰기 (링크 보존)
    routing_decision = state.get("routing_decision", "")
    if routing_decision == "new_medicine_search":
        print("✅ 신약 관련 질문이므로 재구성 건너뛰기 (링크 보존)")
        return state
    
    # 기존 최종 답변 가져오기
    current_answer = state.get("final_answer", "")
    conversation_context = state.get("conversation_context", "")
    current_query = state.get("query", "")
    original_query = state.get("original_query", current_query)
    
    if not current_answer or not current_answer.strip():
        print("⚠️ 최종 답변이 없어 재구성 건너뜀")
        return state
    
    # 🚀 성능 최적화: 연속 질문일 때만 재구성 (첫 질문은 스킵)
    is_follow_up = state.get("is_follow_up", False)
    has_conversation_context = bool(conversation_context and len(conversation_context) > 50)
    
    # 연속 질문 판단: is_follow_up 플래그 또는 conversation_context 존재
    is_continuation = is_follow_up or has_conversation_context
    
    if not is_continuation:
        print("✅ 첫 질문이므로 재구성 건너뛰기 (enhanced_rag_answer 그대로 사용)")
        print(f"   - is_follow_up: {is_follow_up}")
        print(f"   - conversation_context 길이: {len(conversation_context) if conversation_context else 0}")
        state["answer_was_polished"] = False
        return state
    
    print(f"🔄 연속 질문 감지 → 재구성 실행 (is_follow_up: {is_follow_up}, context 길이: {len(conversation_context) if conversation_context else 0})")
    
    print(f"🔍 기존 답변 길이: {len(current_answer)}자")
    print(f"🔍 기존 답변 미리보기: {current_answer[:100]}...")
    
    # GPT에게 자연스러운 대화형 답변으로 재구성 요청
    conversational_prompt = f"""{get_role_definition("pharmacist_friendly")}

**사용자 질문:**
{current_query}

**이전 대화:**
{conversation_context[:800] if conversation_context else "없음"}

**기존 답변 (구조화된 형식):**
{current_answer}

**⚠️ 매우 중요: 기존 답변의 모든 정보를 빠짐없이 포함하세요**
- 기존 답변에는 여러 데이터 소스(Excel DB, PubChem, YouTube, 네이버 뉴스, 용량주의 성분, 연령대 금기, 일일 최대 투여량 등)에서 수집한 정보가 모두 포함되어 있습니다
- 재구성 시 이 모든 정보를 빠짐없이 포함해야 합니다
- 요약하거나 생략하지 말고, 모든 세부사항을 그대로 유지하세요

---

**당신의 임무:** 위의 기존 답변을 **실제 대화하듯이** 재구성하세요.

**⚠️ 연속 질문 판단:**
- 이전 대화가 있고, 사용자 질문이 "주의사항", "사용법", "효능", "부작용" 등 특정 부분만 요청하는 경우 → 연속 질문으로 판단
- 연속 질문인 경우, 질문한 부분만 답변하고 나머지는 생략하세요
- 첫 질문이거나 전체 정보를 요청하는 경우 → 모든 섹션 포함

**핵심 규칙:**

1. **구조적으로 구분하되 대화체 톤 유지:**
   - 이모지와 번호를 사용해서 내용을 명확히 구분
   - 사용 가능한 이모지: 💊 📋 ⚠️ 💡 ✅ ❌ 
   {get_section_structure(["efficacy", "precautions", "alternatives", "additional_info"])}
   - 각 섹션 내에서는 대화하듯이 자연스럽게 작성

2. **모든 정보와 전문용어를 그대로 유지:**
   - 기존 답변의 모든 내용을 빠짐없이 포함
   - 전문용어, 성분명, 수치, 근거 등 모두 그대로 유지
   - 예: "아세트아미노펜은 중추신경계에서 COX-2 효소를 억제하여 프로스타글란딘 생성을 감소시킴으로써..." 이런 전문적인 설명도 그대로 포함
   - 요약하거나 단순화하지 말고 모든 세부사항 유지

3. **각 섹션은 짧은 문단으로:**
   - 한 문단에 3-4문장 정도로 끊기
   - 각 섹션 사이에 한 줄 띄우기
   - 섹션 내부에서도 필요하면 문단 나누기

4. **대화하듯이 시작하세요:**
{get_conversational_tone_examples()}

5. **⚠️ 연속 질문 처리 (매우 중요!):**
   - 이전 대화 맥락을 확인하여, 사용자가 질문한 특정 부분만 답변하세요
   - 예: "주의사항에 대해 더 알려줄래?" → 주의사항 섹션만 답변, 효능/작용 원리는 생략
   - 예: "사용법은 어떻게 되나요?" → 사용법 섹션만 답변, 다른 섹션은 생략
   - **⚠️ 핵심: 연속 질문인 경우, 해당 섹션을 이전 대화보다 훨씬 더 상세하게 확장하여 설명하세요!**
   - 이전 대화의 해당 섹션 내용을 단순 반복하지 말고, 더 깊이 있고 구체적인 정보를 추가하세요
   - 예를 들어, 주의사항을 물었을 때:
     * 이전 대화: "아세트아미노펜은 하루 최대 4,000mg을 초과하지 않도록 주의하세요."
     * 연속 질문 답변: "아세트아미노펜은 하루 최대 4,000mg을 초과하지 않도록 주의해야 해요. 이는 성인 기준으로 하루 8정(정당 500mg 기준)을 넘지 않는다는 의미예요. 만약 다른 약품에도 아세트아미노펜이 포함되어 있다면, 그 용량도 함께 계산해야 합니다. 예를 들어, 감기약과 진통제를 동시에 복용하는 경우, 두 약품의 아세트아미노펜 함량을 합산하여 4,000mg을 초과하지 않도록 주의해야 해요. 과다 복용 시 급성 간부전, 간독성, 메트헤모글로빈혈증 등의 심각한 부작용이 발생할 수 있으며, 특히 간 질환이 있거나 알코올을 자주 섭취하는 경우 더욱 위험합니다."
   - 사용자가 "더 알려달라"고 한 것은 더 상세한 정보를 원한다는 의미이므로, 기존 답변보다 훨씬 더 자세하고 구체적으로 설명하세요
   - 만약 이전 대화가 없거나 첫 질문인 경우, 모든 섹션을 포함하세요

6. **마무리는 짧고 친근하게:**
   - "궁금한 거 있으면 언제든 물어보세요!"
   - "도움이 되셨길 바랍니다!"
   - 긴 인사말이나 격식적인 마무리 금지

**금지사항:**
- 정보 요약이나 생략 (단, 연속 질문의 경우 질문하지 않은 부분은 생략 가능)
- {PromptConfig.COMMON_INSTRUCTIONS['no_source_mention']}
- 과도하게 격식적인 표현
- 너무 긴 문단 (적절히 끊기)
- **연속 질문에서 이전 대화의 해당 섹션을 단순 반복 (대신 더 상세하게 확장해야 함)**

**중요:**
- 첫 질문인 경우: 모든 내용을 포함하되 대화체 톤을 유지하세요
- 연속 질문인 경우: 
  * 질문한 부분만 답변하고, 이전에 이미 설명한 섹션은 생략하세요
  * **하지만 질문한 섹션은 이전 대화보다 훨씬 더 상세하게 확장하여 설명하세요!**
  * 이전 대화의 해당 섹션을 단순 반복하지 말고, 더 구체적이고 깊이 있는 정보를 추가하세요
  * 사용자가 "더 알려달라"고 한 것은 더 많은 정보를 원한다는 의미이므로, 기존 답변의 2-3배 이상의 상세함으로 설명하세요
- 구조적으로 구분하되 대화체 톤을 유지하세요

**예시 비교:**

❌ 나쁜 예 (너무 형식적):
"✅ **욱씬정 사용 가능 여부**

**판단 근거:**
- 아세트아미노펜은 중추신경계에서 COX-2 효소를 억제하여 프로스타글란딘 생성을 감소시킴
- 카페인무수물은 중추신경계 자극으로 피로 감소

**⚠️ 주의사항:**
1. 아세트아미노펜: 하루 최대 4,000mg
2. 카페인무수물: 하루 최대 90mg"

✅ 좋은 예 (구조적이지만 대화체):
"안녕하세요! 욱씬정 궁금하시군요.

네, 감기 증상에 드셔도 돼요.

💊 **효능 및 작용 원리**
욱씬정의 주성분인 아세트아미노펜은 중추신경계에서 COX-2 효소를 억제하여 프로스타글란딘 생성을 감소시킴으로써 통증을 완화하고 열을 내리는 효과가 있답니다. 카페인무수물은 중추신경계를 자극하여 피로를 줄이고 각성을 증가시키는 역할을 해요. 푸르설티아민은 비타민 B1 유도체로 신경 및 근육 기능을 개선하는 데 도움을 줍니다.

욱씬정은 감기의 제증상인 인후통, 오한, 발열, 두통, 관절통, 근육통을 완화하는 데 효과적이에요.

⚠️ **주의사항**
욱씬정에는 아세트아미노펜과 카페인무수물이 포함되어 있어서 용량을 주의해야 해요. 아세트아미노펜은 하루 최대 4,000mg, 카페인무수물은 하루 최대 90mg을 초과하지 않도록 주의하세요. 과다 복용 시 간 손상이나 불면증을 유발할 수 있으니, 반드시 의사나 약사의 처방에 따라 사용해야 합니다.

간 질환 환자나 카페인 과민증 환자는 복용을 피하는 것이 좋습니다.

💡 **대안 약품**
만약 다른 선택지를 원하신다면, 타이레놀이나 포펜정을 고려해보실 수 있어요. 타이레놀은 아세트아미노펜을 주성분으로 하여 유사한 효과를 제공합니다.

📋 **최신 정보**
최근에는 아세트아미노펜이 간독성을 유발할 수 있다는 연구가 있어, 과다 복용을 피하는 것이 중요해요. 특히 다른 약과 중복 복용 시 주의가 필요합니다.

궁금한 거 있으면 언제든 물어보세요!"

---

**이제 위의 기존 답변을 자연스러운 대화체로 재구성하세요.**
설명 없이 재구성된 답변만 출력하세요:
"""
    
    try:
        print("GPT로 답변 재구성 중...")
        
        # ChatGPT 호출 (자연스러운 대화를 위해 적당한 temperature)
        conversational_answer = generate_response_llm_from_prompt(
            prompt=conversational_prompt,
            temperature=0.7,  # 자연스러운 대화를 위해 적당한 temperature
            max_tokens=2000
        )
        
        # 응답 정제
        conversational_answer = conversational_answer.strip()
        
        # 응답이 너무 짧거나 이상하면 원본 유지
        if len(conversational_answer) < len(current_answer) * 0.3:  # 원본의 30% 미만이면 이상함
            print(f"⚠️ 재구성된 답변이 너무 짧아서 원본 유지")
            conversational_answer = current_answer
        elif not conversational_answer:
            print(f"⚠️ 재구성된 답변이 비어있어서 원본 유지")
            conversational_answer = current_answer
        
        print(f"✅ 재구성된 답변 길이: {len(conversational_answer)}자")
        print(f"✅ 재구성된 답변 미리보기: {conversational_answer[:100]}...")
        
        # 상태 업데이트
        state["final_answer"] = conversational_answer
        state["original_answer"] = current_answer  # 원본 답변 보존 (디버깅용)
        state["answer_was_polished"] = True
        
        print("대화형 답변 재구성 완료")
        
    except Exception as e:
        print(f"❌ 답변 재구성 중 오류 발생: {e}")
        # 오류 발생 시 원본 답변 유지
        state["final_answer"] = current_answer
        state["original_answer"] = current_answer
        state["answer_was_polished"] = False
    
    return state

