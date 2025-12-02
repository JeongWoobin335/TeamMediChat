# prompt_utils.py - 공통 프롬프트 유틸리티 함수들

from config import PromptConfig
from typing import Optional, List, Dict

def get_role_definition(role_type: str) -> str:
    """
    역할 정의 반환
    
    Args:
        role_type: 역할 타입 (pharmacist, expert, classifier 등)
    
    Returns:
        역할 정의 문자열
    """
    return PromptConfig.ROLES.get(role_type, "당신은 전문가입니다.")


def get_common_instructions(include_source_mention: bool = True, 
                           include_tone: bool = True,
                           include_comprehensive: bool = True) -> str:
    """
    공통 지시사항 반환
    
    Args:
        include_source_mention: 출처 언급 금지 포함 여부
        include_tone: 톤 지시사항 포함 여부
        include_comprehensive: 종합 정보 활용 지시사항 포함 여부
    
    Returns:
        공통 지시사항 문자열
    """
    instructions = []
    
    if include_source_mention:
        instructions.append(f"- {PromptConfig.COMMON_INSTRUCTIONS['no_source_mention']}")
    
    if include_tone:
        instructions.append(f"- {PromptConfig.COMMON_INSTRUCTIONS['natural_tone']}")
    
    if include_comprehensive:
        instructions.append(f"- {PromptConfig.COMMON_INSTRUCTIONS['comprehensive_info']}")
    
    if instructions:
        return "**중요 지침:**\n" + "\n".join(instructions)
    return ""


def get_section_structure(sections: Optional[List[str]] = None) -> str:
    """
    공통 섹션 구조 반환
    
    Args:
        sections: 포함할 섹션 리스트 (None이면 모든 섹션)
    
    Returns:
        섹션 구조 문자열
    """
    if sections is None:
        sections = ["efficacy", "precautions", "usage", "alternatives", "latest_info"]
    
    section_list = []
    for section in sections:
        if section in PromptConfig.SECTION_STRUCTURE:
            section_list.append(f"   - {PromptConfig.SECTION_STRUCTURE[section]}")
    
    if section_list:
        return "**이모지로 섹션을 나누어서 답변하세요:**\n" + "\n".join(section_list)
    return ""


def get_medical_consultation_footer(style: str = "standard") -> str:
    """
    의료진 상담 권고 마무리 반환
    
    Args:
        style: 스타일 (standard, friendly, warning)
    
    Returns:
        마무리 문자열
    """
    if style == "friendly":
        return f"\n\n💡 {PromptConfig.COMMON_INSTRUCTIONS['medical_consultation']}"
    elif style == "warning":
        return f"\n\n⚠️ **중요**: {PromptConfig.COMMON_INSTRUCTIONS['medical_consultation']}"
    else:
        return f"\n\n{PromptConfig.FOOTERS['medical_consultation']}"


def get_friendly_closing() -> str:
    """
    친근한 마무리 문구 반환
    """
    return PromptConfig.FOOTERS['friendly_closing']


def build_answer_prompt_structure(
    role_type: str,
    user_question: str,
    context: Optional[str] = None,
    collected_data: Optional[str] = None,
    include_sections: bool = True,
    include_common_instructions: bool = True,
    include_footer: bool = True,
    footer_style: str = "standard"
) -> str:
    """
    답변 생성 프롬프트의 기본 구조를 생성
    
    Args:
        role_type: 역할 타입
        user_question: 사용자 질문
        context: 대화 맥락 (선택)
        collected_data: 수집된 데이터 (선택)
        include_sections: 섹션 구조 포함 여부
        include_common_instructions: 공통 지시사항 포함 여부
        include_footer: 마무리 포함 여부
        footer_style: 마무리 스타일
    
    Returns:
        프롬프트 문자열
    """
    parts = []
    
    # 역할 정의
    parts.append(get_role_definition(role_type))
    
    # 사용자 질문
    parts.append(f"**사용자 질문:**\n{user_question}")
    
    # 대화 맥락
    if context:
        parts.append(f"**이전 대화 맥락:**\n{context[:1000] if len(context) > 1000 else context}")
    
    # 수집된 데이터
    if collected_data:
        parts.append(f"**수집된 정보:**\n{collected_data}")
    
    # 섹션 구조
    if include_sections:
        parts.append(get_section_structure())
    
    # 공통 지시사항
    if include_common_instructions:
        parts.append(get_common_instructions())
    
    # 마무리
    if include_footer:
        parts.append(get_medical_consultation_footer(footer_style))
    
    return "\n\n".join(parts)


def get_source_mention_examples() -> str:
    """
    출처 언급 예시 반환 (올바른/잘못된 예시)
    """
    return """
**출처 언급 예시:**
- ✅ "전문가들에 따르면...", "알려진 바로는...", "일반적으로...", "최근에는..."
- ❌ "YouTube에서 봤는데...", "DB에 따르면...", "Excel DB에서...", "PubChem에서..."
"""


def get_conversational_tone_examples() -> str:
    """
    대화형 톤 예시 반환
    """
    return """
**대화형 톤 예시:**
- 처음 질문: "안녕하세요! [약품명] 궁금하시군요.", "안녕하세요! [약품명]에 대해 알려드릴게요."
- 연속 질문: "네, 그 부분 설명해드릴게요.", "아, 그거요?"
- 섹션 시작: "이유는 이래요.", "주의하실 점은 이거예요.", "다른 선택지도 있어요."
- 마무리: "궁금한 거 있으면 언제든 물어보세요!", "도움이 되셨길 바랍니다!"
"""

