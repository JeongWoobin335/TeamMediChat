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
    datasource: Literal["OCR_IMAGE", "MEDICINE_RECOMMENDATION", "MEDICINE_USAGE_CHECK", "MEDICINE_INFO", "SNS_SEARCH"] = Field(...)
    reason: str
    condition: list[str] = []
    category: str = ""
    requested_fields: list[str] = []
    medicine_name: str = ""  # 사용자가 언급한 약품명
    usage_context: str = ""  # 사용하려는 상황/증상
    has_image: bool = False  # 이미지가 포함된 질문인지 여부

# 프롬프트 정의
system_prompt = """
당신은 사용자의 약품 관련 질문을 분석하여 가장 적절한 처리 방법을 결정하는 전문가입니다.

사용자의 질문을 분석하여 다음을 판단하세요:

1. **datasource**: 
   - "OCR_IMAGE": 이미지가 포함된 질문으로, OCR로 약품명을 추출해야 하는 경우 (예: "이 연고 상처에 발라도 되나?" + 이미지)
   - "MEDICINE_USAGE_CHECK": 특정 약품의 사용 가능성 판단 요청 (예: "베타딘 연고가 있는데 상처에 발라도 되나?")
   - "MEDICINE_RECOMMENDATION": 특정 질병/증상에 대한 약품 추천 요청
   - "MEDICINE_INFO": 약품 정보, 사용법, 부작용 등 기본 정보 요청
   - "SNS_SEARCH": 최신 신약 소식, 사용자 경험담, 실시간 정보, 뉴스 등 SNS/미디어 검색이 필요한 질문

2. **condition**: 질병/증상 또는 신체 부위가 언급된 경우 추출
   - 질병/증상: 다양한 표현을 의미적으로 이해하여 표준 의학 용어로 변환
     * 피곤함 관련: "지쳐서", "기운이 없어서", "체력이 떨어져서", "너무 피곤해", "힘들어서" → "피곤함"
     * 체함 관련: "속이 안 좋아서", "소화가 안 돼서", "배가 불편해서", "위가 안 좋아서" → "체함"
     * 변비 관련: "배가 안 나와서", "장이 안 좋아서", "변비가 심해서" → "변비"
     * 기타: "감기", "두통", "위장염" 등
   - 신체 부위: "위장", "심장", "간", "신장" 등 (부담/자극 관련 키워드와 함께 언급된 경우)
   - 예시: "너무 지쳐서 약 먹고 싶은데" → condition: ["피곤함"]
   - 예시: "속이 안 좋아서 약 먹을까 하는데" → condition: ["체함"]
   - 예시: "위에 부담이 적은 감기약" → condition: ["감기", "위장"]

3. **category**: 질문의 주제 카테고리 (예: "약품 정보", "사용법", "최신 약품", "부작용 비교" 등)

4. **requested_fields**: 질문에서 요구한 항목들만 나열 (예: "효능", "부작용", "사용법")

5. **medicine_name**: 사용자가 언급한 약품명 (MEDICINE_USAGE_CHECK인 경우 필수)
   - 예: "베타딘 연고", "타이레놀", "이부프로펜" 등
   - 주의: "바스포라는 연고" → "바스포" (조사 제거)
   - 주의: "타이레놀정이 있는데" → "타이레놀" (조사 제거)
   - 주의: "이부프로펜을 먹어도" → "이부프로펜" (조사 제거)

6. **usage_context**: 약품을 사용하려는 상황/증상 (MEDICINE_USAGE_CHECK인 경우 필수)
   - 예: "상처", "두통", "감기", "위가 안 좋을 때" 등

7. **has_image**: 질문에 이미지가 포함되어 있는지 여부 (true/false)

8. **reason**: 왜 이렇게 분류했는지 간단한 이유

**중요**: 
- **"이 연고", "이 약", "이거", "이 약은", "이 연고는" 등의 표현이 있으면 반드시 OCR_IMAGE로 분류하고 has_image: true로 설정**
- "이 약은 감기에 먹어도 되나?" = OCR_IMAGE (has_image: true) - 무조건 이미지 포함으로 간주
- "이 연고는 상처에 발라도 되나?" = OCR_IMAGE (has_image: true) - 무조건 이미지 포함으로 간주
- "이 약 감기에 먹어도 되나?" = OCR_IMAGE (has_image: true) - 무조건 이미지 포함으로 간주
- **명시적인 약품명이 언급된 경우만** MEDICINE_USAGE_CHECK로 분류 (예: "타이레놀을 감기에 먹어도 되나?")
- **"이" + 약품 관련 단어 = 무조건 이미지 포함으로 간주**
- "최신", "신약", "새로운", "최근", "상륙", "출시" 등의 키워드가 있으면 SNS_SEARCH로 분류
- "경험담", "후기", "사용해본", "복용해본" 등의 키워드가 있으면 SNS_SEARCH로 분류
- "뉴스", "소식", "정보" 등의 키워드가 있으면 SNS_SEARCH로 분류
- 질문에 명시되지 않은 항목은 절대 포함하지 마세요
- 사용자의 의도를 정확히 파악하여 가장 적절한 경로로 안내하세요
- 이전 대화 내용을 묻는 질문은 MEDICINE_INFO로 분류하세요
- 신체 부위와 부담/자극 관련 키워드가 함께 언급되면 해당 신체 부위도 condition에 포함하세요

**약품명 추출 예시**:
- "바스포라는 연고 상처에 발라도 될까?" → datasource: "MEDICINE_USAGE_CHECK", medicine_name: "바스포", has_image: false
- "타이레놀정이 있는데 두통에 먹어도 될까?" → datasource: "MEDICINE_USAGE_CHECK", medicine_name: "타이레놀", has_image: false
- "이부프로펜을 감기에 먹어도 될까?" → datasource: "MEDICINE_USAGE_CHECK", medicine_name: "이부프로펜", has_image: false
- "이 연고 상처에 발라도 될까?" → datasource: "OCR_IMAGE", medicine_name: "", has_image: true
- "이 약은 감기에 먹어도 되나?" → datasource: "OCR_IMAGE", medicine_name: "", has_image: true
- "이 연고는 습진에 발라도 되나?" → datasource: "OCR_IMAGE", medicine_name: "", has_image: true
- "이 약 감기에 먹어도 되나?" → datasource: "OCR_IMAGE", medicine_name: "", has_image: true

다음 JSON 형식으로만 응답하세요:

{{
  "datasource": "...",
  "reason": "...",
  "condition": [...],
  "category": "...",
  "requested_fields": [...],
  "medicine_name": "...",
  "usage_context": "...",
  "has_image": true/false
}}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

# 구조화된 LLM 라우터
question_router = prompt | llm.with_structured_output(RouteQuery)

# route_question_node.py 내부 수정
def route_question_node(state: QAState) -> QAState:
    query = state["query"]
    result = question_router.invoke({"question": query})

    # 약품명에서 조사 제거 (정규식 기반)
    import re
    medicine_name = result.medicine_name
    if medicine_name:
        # 한글 조사 제거
        medicine_name = re.sub(r'[은는이가을를에의와과도부터까지에서부터]$', '', medicine_name)
        # 연속된 공백 제거
        medicine_name = re.sub(r'\s+', ' ', medicine_name).strip()
        print(f"🔍 약품명 조사 제거: '{result.medicine_name}' → '{medicine_name}'")
    
    # 상태에 저장
    state["condition"] = result.condition
    state["category"] = result.category
    state["medicine_name"] = medicine_name  # 조사 제거된 약품명
    state["usage_context"] = result.usage_context
    state["has_image"] = result.has_image  # has_image 필드도 저장

    # ❗ requested_fields fallback 추가
    state["requested_fields"] = result.requested_fields if result.requested_fields else ["효능", "부작용", "사용법"]
    
    # 디버깅용 로그 추가
    print(f"🔍 라우팅 분석 결과:")
    print(f"  - datasource: {result.datasource}")
    print(f"  - has_image: {result.has_image}")
    print(f"  - medicine_name: {result.medicine_name}")
    print(f"  - usage_context: {result.usage_context}")
    
    # 라우팅 결정 로직 개선
    if result.datasource == "OCR_IMAGE" and result.has_image:
        routing_decision = "ocr_image"  # OCR 이미지 처리 (이미지가 있을 때만)
    elif result.datasource == "OCR_IMAGE" and not result.has_image:
        routing_decision = "usage_check"  # 이미지가 없으면 사용 가능성 판단으로
    elif result.datasource == "MEDICINE_USAGE_CHECK":
        routing_decision = "usage_check"  # 새로운 약품 사용 가능성 판단
    elif result.datasource == "MEDICINE_RECOMMENDATION":
        routing_decision = "recommend"
    elif result.datasource == "SNS_SEARCH":
        routing_decision = "sns_search"  # 새로운 라우팅 옵션
    else:
        routing_decision = "search"
    
    state["routing_decision"] = routing_decision
    return state
