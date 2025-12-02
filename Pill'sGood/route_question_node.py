# route_question_node.py

from qa_state import QAState
from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_teddynote.models import get_model_name, LLMs

# LLM 초기화
MODEL_NAME = get_model_name(LLMs.GPT4)
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

# 구조화된 라우팅 출력 모델 정의
class RouteQuery(BaseModel):
    datasource: Literal["OCR_IMAGE", "MEDICINE_USAGE_CHECK", "MEDICINE_INFO", "SNS_SEARCH", "NEW_MEDICINE_SEARCH", "FOLLOW_UP_QUESTION"] = Field(...)
    reason: str
    condition: list[str] = []
    category: str = ""
    requested_fields: list[str] = []
    medicine_name: str = ""  # 사용자가 언급한 약품명
    usage_context: str = ""  # 사용하려는 상황/증상
    has_image: bool = False  # 이미지가 포함된 질문인지 여부
    is_follow_up: bool = False  # 이전 답변에 대한 추가 질문인지 여부
    follow_up_type: str = ""  # 추가 질문 유형 (usage, ingredient, side_effect 등)

# 프롬프트 정의 - 최적화된 버전 (Few-shot + Chain-of-Thought)
system_prompt = """당신은 약품 질문 분류 전문가입니다. 단계별로 분석하여 정확한 경로로 안내하세요.

**중요:** 질문이 "원본 질문: ... 보정된 질문: ..." 형식으로 제공되면, 원본 질문의 패턴(예: "그럼", "그러면" 등)을 우선 확인하세요.

**⚠️ 약품명 추출 시 주의사항:**
- 약품명의 일부인 "정", "연고", "캡슐", "시럽", "액", "주사" 등은 절대 제거하지 마세요.
- 조사(은, 는, 이, 가, 을, 를 등)만 제거하세요.
- 예: "마그틴정은" → "마그틴정" (조사 "은"만 제거, "정"은 유지)
- 예: "욱씬정을" → "욱씬정" (조사 "을"만 제거, "정"은 유지)

## 📋 분류 카테고리
1. OCR_IMAGE: "이 약/연고" 등 지시대명사 사용 (이미지 필요)
2. MEDICINE_USAGE_CHECK: 명시적 약품명 + 사용 가능 여부 질문
3. FOLLOW_UP_QUESTION: 이전 답변 관련 추가 질문
4. NEW_MEDICINE_SEARCH: 신약 관련 질문 (신약 뉴스, 신약 개발, 신약 승인 등)
5. SNS_SEARCH: 최신 정보/경험담/뉴스 검색 필요 (기존 약품의 보조 정보)
6. MEDICINE_INFO: 일반적인 약품 정보 문의

## 🔍 단계별 분석 프로세스

### STEP 1: 지시대명사 확인 (최우선)
- "이 약", "이 연고", "이거", "그 약" 포함? → OCR_IMAGE (has_image: true)

### STEP 2: 연속 질문 패턴 확인
- "그럼", "그러면", "그런데"로 시작?
- 단편적 질문? ("사용법은?", "부작용은?", "~이 뭔데?")
→ FOLLOW_UP_QUESTION (is_follow_up: true)

### STEP 3: 약품명 존재 여부
- 명시적 약품명 있음? → MEDICINE_USAGE_CHECK 또는 MEDICINE_INFO
- 없음 + 정보 요청? → MEDICINE_INFO

### STEP 4: 신약 관련 질문 확인 (최우선)
- "신약" + "뉴스/소식/개발/승인/출시" 포함? → NEW_MEDICINE_SEARCH
- "치매 신약", "알츠하이머 신약", "당뇨 신약" 등 질병명 + 신약? → NEW_MEDICINE_SEARCH
- "최근 나온 약", "새로 나온 약", "신약 알려줘" → NEW_MEDICINE_SEARCH
- **중요**: 신약 관련 질문은 약품 사용 가능성 판단이 아닌 신약 정보 요청

### STEP 5: SNS 검색 키워드 확인
- "최신", "경험담", "후기", "뉴스", "출시" 포함? (신약 키워드 제외)
→ SNS_SEARCH (기존 약품의 보조 정보 수집용)

### STEP 6: 최종 판단
- 사용 가능 여부 질문? ("~해도 되나?", "먹어도 될까?")
- 정보 요청? ("~에 대해 알려줘", "~은 뭐야?")

## 💡 Few-Shot 예시 (추론 과정 포함)

### 예시 1
질문: "이 연고 습진에 발라도 되나?"
[추론]
- STEP 1: "이 연고" 발견 → 이미지 필요
- 결론: OCR_IMAGE
출력: {{"datasource": "OCR_IMAGE", "medicine_name": "", "usage_context": "습진", "has_image": true, "reason": "지시대명사 '이 연고' 사용, 이미지 필요"}}

### 예시 2
질문: "바스포라는 연고 상처에 발라도 될까?"
[추론]
- STEP 1: 지시대명사 없음
- STEP 3: 약품명 "바스포" 발견 (조사 "라는" 제거)
- STEP 5: 사용 가능 여부 질문
출력: {{"datasource": "MEDICINE_USAGE_CHECK", "medicine_name": "바스포", "usage_context": "상처", "has_image": false, "reason": "명시적 약품명과 사용 상황 제시"}}

### 예시 2-1
질문: "마그틴정은 체했을 때 먹어도 되나?"
[추론]
- STEP 1: 지시대명사 없음
- STEP 3: 약품명 "마그틴정" 발견 (조사 "은"만 제거, "정"은 약품명의 일부이므로 유지)
- STEP 5: 사용 가능 여부 질문
출력: {{"datasource": "MEDICINE_USAGE_CHECK", "medicine_name": "마그틴정", "usage_context": "체함", "has_image": false, "reason": "명시적 약품명과 사용 상황 제시"}}

### 예시 3
질문: "그럼 부작용은 뭐야?"
[추론]
- STEP 2: "그럼"으로 시작 + 단편적 질문
- 결론: FOLLOW_UP_QUESTION
출력: {{"datasource": "FOLLOW_UP_QUESTION", "is_follow_up": true, "follow_up_type": "side_effect", "reason": "이전 답변에 대한 추가 질문"}}

### 예시 4
질문: "최근에 나온 신약 소식 알려줘"
[추론]
- STEP 4: "신약" + "소식" 키워드 → 신약 관련 질문
- 결론: NEW_MEDICINE_SEARCH
출력: {{"datasource": "NEW_MEDICINE_SEARCH", "category": "신약", "reason": "신약 관련 뉴스 검색 필요"}}

### 예시 4-1
질문: "치매 신약에 관한 뉴스를 알려줘"
[추론]
- STEP 4: "치매" + "신약" + "뉴스" → 신약 관련 질문
- 결론: NEW_MEDICINE_SEARCH
출력: {{"datasource": "NEW_MEDICINE_SEARCH", "category": "신약", "reason": "신약 관련 뉴스 검색 필요"}}

### 예시 5
질문: "푸르설티아민이 뭔데?"
[추론]
- STEP 2: 단편적 질문 + 성분명 의문
- 결론: FOLLOW_UP_QUESTION
출력: {{"datasource": "FOLLOW_UP_QUESTION", "is_follow_up": true, "follow_up_type": "ingredient", "reason": "성분명에 대한 질문"}}

### 예시 6
질문: "원본 질문: 그럼 어떻게 먹으면 되는데?\n보정된 질문: 욱씬정을 어떻게 복용하면 되나요?"
[추론]
- STEP 2: 원본 질문에서 "그럼"으로 시작 + "어떻게" 사용법 질문
- 결론: FOLLOW_UP_QUESTION (원본 질문의 패턴 우선)
출력: {{"datasource": "FOLLOW_UP_QUESTION", "is_follow_up": true, "follow_up_type": "usage", "medicine_name": "욱씬정", "reason": "원본 질문의 '그럼' 패턴과 사용법 질문"}}

## ⚙️ 필드 추출 가이드

**medicine_name**: 약품명만 추출 (조사만 제거, 약품명의 일부인 "정", "연고", "캡슐" 등은 유지)
- "타이레놀정이" → "타이레놀정" (조사 "이"만 제거, "정"은 유지)
- "바스포라는" → "바스포" (조사 "라는"만 제거)
- "마그틴정은" → "마그틴정" (조사 "은"만 제거, "정"은 유지)
- "욱씬정을" → "욱씬정" (조사 "을"만 제거, "정"은 유지)

**condition**: 증상을 표준 용어로 변환
- "지쳐서" → "피곤함"
- "속이 안 좋아서" → "체함"
- "머리가 아파서" → "두통"

**usage_context**: 사용 상황/증상
- "상처에 발라도 될까?" → "상처"
- "감기에 먹어도 되나?" → "감기"

**follow_up_type**: 연속 질문 유형
- usage, ingredient, side_effect, mechanism, precaution
- alternative_medicines, similar_medicines, new_medicine

## 📤 출력 형식
{{
  "datasource": "...",
  "reason": "...",
  "condition": [...],
  "category": "...",
  "requested_fields": [...],
  "medicine_name": "...",
  "usage_context": "...",
  "has_image": true/false,
  "is_follow_up": true/false,
  "follow_up_type": "..."
}}

**중요**: 
- 질문에 명시되지 않은 정보는 빈 값으로 두세요. 추측하지 마세요.
- 약품명에서 "정", "연고", "캡슐" 등은 약품명의 일부이므로 절대 제거하지 마세요.
- 조사(은, 는, 이, 가, 을, 를, 에, 의, 와, 과 등)만 제거하세요.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

# 구조화된 LLM 라우터
question_router = prompt | llm.with_structured_output(RouteQuery)

# route_question_node.py 내부 수정
def route_question_node(state: QAState) -> QAState:
    # preprocess 직후이므로 원본 쿼리 사용
    query = state.get("query", "")
    original_query = state.get("original_query", query)
    
    # 원본 질문 우선 사용 (신약 관련 질문은 보정 없이 원본 그대로 사용)
    result = question_router.invoke({"question": query})

    # 약품명에서 조사 제거 (정규식 기반)
    import re
    medicine_name = result.medicine_name
    if medicine_name:
        # 원본 약품명 보존 (나중에 "정" 복원을 위해)
        original_medicine_name = medicine_name
        
        # 한글 조사 제거
        medicine_name = re.sub(r'[은는이가을를에의와과도부터까지에서부터]$', '', medicine_name)
        # 연속된 공백 제거
        medicine_name = re.sub(r'\s+', ' ', medicine_name).strip()
        
        # "정", "연고", "캡슐" 등이 제거되었는지 확인하고 복원
        # 원본 질문에서 직접 약품명 형태 확인
        query_lower = query.lower()
        medicine_name_lower = medicine_name.lower()
        
        # 원본 질문에서 "약품명+형태" 패턴 직접 추출
        medicine_forms = ['정', '연고', '캡슐', '시럽', '액', '주사', '분말', '가루']
        for form in medicine_forms:
            # 원본 질문에서 "약품명+형태" 패턴 찾기
            # 예: "마그틴정은" → "마그틴정"
            pattern = rf'({re.escape(medicine_name_lower)})\s*{form}[은는이가을를에의와과도부터까지에서부터]?'
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                # 약품명 리스트에서 "약품명+형태" 형태로 검색
                from retrievers import excel_docs
                candidate_name = medicine_name_lower + form
                for doc in excel_docs:
                    product_name = doc.metadata.get("제품명", "")
                    if product_name and product_name.lower() == candidate_name:
                        medicine_name = product_name
                        print(f"✅ 약품명 형태 복원: '{medicine_name_lower}' → '{medicine_name}' (원본 질문에서 '{form}' 발견)")
                        break
                if medicine_name != medicine_name_lower:  # 복원되었으면
                    break
        
        print(f"🔍 약품명 조사 제거: '{result.medicine_name}' → '{medicine_name}'")
    
    # 상태에 저장
    state["condition"] = result.condition
    state["category"] = result.category
    state["medicine_name"] = medicine_name  # 조사 제거된 약품명
    state["usage_context"] = result.usage_context
    state["has_image"] = result.has_image  # has_image 필드도 저장
    state["is_follow_up"] = result.is_follow_up  # 연속 질문 여부
    state["follow_up_type"] = result.follow_up_type  # 연속 질문 유형

    # ❗ requested_fields fallback 추가
    state["requested_fields"] = result.requested_fields if result.requested_fields else ["효능", "부작용", "사용법"]
    
    # 디버깅용 로그 추가
    print(f"🔍 라우팅 분석 결과:")
    print(f"  - datasource: {result.datasource}")
    print(f"  - has_image: {result.has_image}")
    print(f"  - medicine_name: {result.medicine_name}")
    print(f"  - usage_context: {result.usage_context}")
    
    # 라우팅 결정 로직 개선
    if result.datasource == "FOLLOW_UP_QUESTION":
        routing_decision = "follow_up"  # 연속 질문 처리
    elif result.datasource == "OCR_IMAGE" and result.has_image:
        routing_decision = "ocr_image"  # OCR 이미지 처리 (이미지가 있을 때만)
    elif result.datasource == "OCR_IMAGE" and not result.has_image:
        routing_decision = "usage_check"  # 이미지가 없으면 사용 가능성 판단으로
    elif result.datasource == "MEDICINE_USAGE_CHECK":
        routing_decision = "usage_check"  # 약품 사용 가능성 판단
    elif result.datasource == "NEW_MEDICINE_SEARCH":
        routing_decision = "new_medicine_search"  # 신약 관련 질문 전용 검색
    elif result.datasource == "SNS_SEARCH":
        routing_decision = "sns_search"  # 기존 약품의 보조 정보 검색 (enhanced_rag에서 사용)
    else:
        routing_decision = "search"
    
    state["routing_decision"] = routing_decision
    return state
