import json
import re
from typing import List
from retrievers import llm
from cache_manager import cache_manager
from langchain_openai import ChatOpenAI


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


# ✅ 새로운 LLM 응답 생성 함수 (prompt 기반, 캐싱 포함)
def generate_response_llm_from_prompt(prompt: str, temperature: float = 0.7, max_tokens: int = 1000, cache_type: str = "general", use_cache: bool = True) -> str:
    """
    프롬프트를 직접 받아서 LLM 응답을 생성하는 함수 (캐싱 지원)
    
    Args:
        prompt: LLM에게 전달할 프롬프트
        temperature: 응답의 창의성 (0.0 ~ 1.0)
        max_tokens: 최대 토큰 수
        cache_type: 캐시 타입 (기본값: "general")
        use_cache: 캐시 사용 여부 (기본값: True)
        
    Returns:
        LLM이 생성한 응답 텍스트
    """
    try:
        # 캐시 확인 (temperature가 0.3 이하일 때만 캐시 사용 - 일관성 있는 응답만 캐싱)
        # 캐시 키에 temperature를 포함시켜서 다른 temperature의 응답과 구분
        cache_key = f"{prompt}__temp_{temperature}" if use_cache and temperature <= 0.3 else None
        if cache_key:
            cached_response = cache_manager.get_llm_response_cache(cache_key, cache_type)
            if cached_response:
                return cached_response
        
        # LLM 호출 - temperature를 실제로 적용하기 위해 새로운 객체 생성
        # 기존 llm 객체의 설정을 가져와서 temperature만 변경
        # LangChain ChatOpenAI 객체에서 model 정보 가져오기
        model_name = "gpt-4o"  # 기본값
        if hasattr(llm, 'model_name'):
            model_name = llm.model_name
        elif hasattr(llm, 'model'):
            model_name = llm.model
        elif hasattr(llm, '_default_params') and 'model' in llm._default_params:
            model_name = llm._default_params['model']
        
        # temperature와 max_tokens를 적용한 새로운 LLM 객체 생성
        llm_kwargs = {"model": model_name, "temperature": temperature}
        if max_tokens:
            llm_kwargs["max_tokens"] = max_tokens
        
        llm_with_temp = ChatOpenAI(**llm_kwargs)
        
        response = llm_with_temp.invoke(prompt)
        result = response.content.strip()
        
        # 캐시 저장 (temperature가 0.3 이하일 때만)
        if cache_key:
            cache_manager.save_llm_response_cache(cache_key, result, cache_type)
        
        return result
    except Exception as e:
        print(f"❌ LLM 응답 생성 중 오류 발생: {e}")
        return f"죄송합니다. 응답을 생성하는 중 오류가 발생했습니다: {str(e)}"
