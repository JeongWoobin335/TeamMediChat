import json
import re
from typing import List
from retrievers import llm


# ✅ 텍스트 정규화 유틸
def normalize(text: str) -> str:
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"[^\w가-힣]", "", text)
    return re.sub(r"\s+", "", text.strip().lower())


# ✅ 문서에서 특정 필드 추출
def extract_field(docs, label, product_name=None):
    # 더 간단하고 확실한 패턴으로 변경
    pattern = rf"\[{label}\]\s*[:：]?\s*(.*?)(?=\n\[|\Z)"

    # 1순위: 정확한 제품명 필터링
    for doc in docs:
        content = doc.page_content.replace("", "")
        meta_name = doc.metadata.get("제품명", "")
        if product_name and normalize(meta_name) != normalize(product_name):
            continue
        match = re.search(pattern, content, re.DOTALL)  # DOTALL 플래그 추가
        if match:
            result = match.group(1).strip()
            if result and result != "정보 없음":
                return result

    # 2순위: 전체 문서에서 탐색
    for doc in docs:
        content = doc.page_content.replace("", "")
        match = re.search(pattern, content, re.DOTALL)  # DOTALL 플래그 추가
        if match:
            result = match.group(1).strip()
            if result and result != "정보 없음":
                return result

    return "정보 없음"


# ✅ LLM 응답 생성 (기존 함수)
def generate_response_llm(name: str, fields: List[str], eff: str, side: str, usage: str, 
                         conversation_context: str = "", user_context: str = "") -> str:
    context = {
        "제품명": name,
        "효능": eff,
        "부작용": side,
        "사용법": usage
    }

    field_info = {
        k: v for k, v in context.items()
        if k in fields and v not in ["정보 없음", ""]
    }

    # 대화 맥락 정보 구성
    context_info = ""
    if conversation_context:
        context_info += f"\n이전 대화 맥락:\n{conversation_context}\n"
    if user_context:
        context_info += f"\n사용자 질문 맥락:\n{user_context}\n"

    prompt = f"""
당신은 따뜻하고 신뢰감 있는 건강 상담사입니다.
다음 정보에 기반하여 사용자 질문에 응답해주세요.

요청 항목: {fields}
정보:
{json.dumps(field_info, ensure_ascii=False)}{context_info}

답변:
"""
    return llm.invoke(prompt).content.strip()


# ✅ 새로운 LLM 응답 생성 함수 (prompt 기반)
def generate_response_llm_from_prompt(prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
    """
    프롬프트를 직접 받아서 LLM 응답을 생성하는 함수
    
    Args:
        prompt: LLM에게 전달할 프롬프트
        temperature: 응답의 창의성 (0.0 ~ 1.0)
        max_tokens: 최대 토큰 수
        
    Returns:
        LLM이 생성한 응답 텍스트
    """
    try:
        # LLM 호출 (temperature와 max_tokens는 현재 llm 객체에서 지원하지 않을 수 있음)
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"❌ LLM 응답 생성 중 오류 발생: {e}")
        return f"죄송합니다. 응답을 생성하는 중 오류가 발생했습니다: {str(e)}"
