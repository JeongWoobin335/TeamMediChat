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
    datasource: Literal["MEDICINE_RECOMMENDATION", "MEDICINE_INFO", "SNS_SEARCH"] = Field(...)
    reason: str
    condition: list[str] = []
    category: str = ""
    requested_fields: list[str] = []

# 프롬프트 정의
system_prompt = """
당신은 사용자의 약품 관련 질문을 분석하여 가장 적절한 처리 방법을 결정하는 전문가입니다.

사용자의 질문을 분석하여 다음을 판단하세요:

1. **datasource**: 
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

5. **reason**: 왜 이렇게 분류했는지 간단한 이유

**중요**: 
- "최신", "신약", "새로운", "최근", "상륙", "출시" 등의 키워드가 있으면 SNS_SEARCH로 분류
- "경험담", "후기", "사용해본", "복용해본" 등의 키워드가 있으면 SNS_SEARCH로 분류
- "뉴스", "소식", "정보" 등의 키워드가 있으면 SNS_SEARCH로 분류
- 질문에 명시되지 않은 항목은 절대 포함하지 마세요
- 사용자의 의도를 정확히 파악하여 가장 적절한 경로로 안내하세요
- 이전 대화 내용을 묻는 질문은 MEDICINE_INFO로 분류하세요
- 신체 부위와 부담/자극 관련 키워드가 함께 언급되면 해당 신체 부위도 condition에 포함하세요

다음 JSON 형식으로만 응답하세요:

{{
  "datasource": "...",
  "reason": "...",
  "condition": [...],
  "category": "...",
  "requested_fields": [...]
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

    # 상태에 저장
    state["condition"] = result.condition
    state["category"] = result.category

    # ❗ requested_fields fallback 추가
    state["requested_fields"] = result.requested_fields if result.requested_fields else ["효능", "부작용", "사용법"]
    
    # 라우팅 결정 로직 개선
    if result.datasource == "MEDICINE_RECOMMENDATION":
        routing_decision = "recommend"
    elif result.datasource == "SNS_SEARCH":
        routing_decision = "sns_search"  # 새로운 라우팅 옵션
    else:
        routing_decision = "search"
    
    state["routing_decision"] = routing_decision
    return state
