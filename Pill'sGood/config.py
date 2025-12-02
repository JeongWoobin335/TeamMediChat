# 설정 관리 파일 - LLM 기반 자동 판단을 위한 최소한의 설정

class SearchConfig:
    """검색 관련 설정"""
    
    # 검색 결과 제한
    MAX_SEARCH_RESULTS = 3
    MAX_CONTEXT_LENGTH = 200
    MAX_DISPLAY_LENGTH = 100
    
    # 캐시 설정
    ENABLE_CACHE = True
    CACHE_EXPIRY_DAYS = 7
    
    # 벡터 DB 설정
    VECTOR_DB_SIMILARITY_THRESHOLD = 0.7
    
class ModelConfig:
    """모델 관련 설정"""
    
    # LLM 설정
    DEFAULT_TEMPERATURE = 0
    DEFAULT_MODEL = "gpt-4"
    
    # 토큰 제한
    MAX_INPUT_TOKENS = 4000
    MAX_OUTPUT_TOKENS = 1000

class PromptConfig:
    """프롬프트 관련 설정"""
    
    # 답변 길이 제한
    MIN_ANSWER_LENGTH = 1500  # 최종 답변 최소 길이
    MIN_DETAILED_ANSWER_LENGTH = 1200  # 상세 답변 최소 길이
    MIN_SECTION_LENGTH = 400  # 섹션별 최소 길이
    MAX_SECTION_LENGTH = 800  # 섹션별 최대 길이
    MIN_NEWS_SECTION_LENGTH = 800  # 뉴스 섹션 최소 길이 (증가)
    MIN_VIDEO_SECTION_LENGTH = 700  # 영상 섹션 최소 길이 (증가)
    MIN_SUMMARY_LENGTH = 200  # 요약 최소 길이
    MAX_SUMMARY_LENGTH = 300  # 요약 최대 길이
    
    # 일반 답변 길이
    MIN_CONVERSATIONAL_LENGTH = 200  # 대화형 답변 최소 길이
    MAX_CONVERSATIONAL_LENGTH = 400  # 대화형 답변 최대 길이
    MIN_INGREDIENT_ANSWER_LENGTH = 500  # 성분 답변 최소 길이
    MAX_INGREDIENT_ANSWER_LENGTH = 700  # 성분 답변 최대 길이
    
    # 공통 지시사항
    COMMON_INSTRUCTIONS = {
        "no_source_mention": "출처 언급 금지 (YouTube, Excel DB, PubChem 등 언급하지 마세요)",
        "natural_tone": "자연스럽고 친근한 톤으로 답변",
        "medical_consultation": "정확한 진단을 위해서는 의사나 약사와 상담하시기 바랍니다",
        "comprehensive_info": "모든 수집된 정보를 종합적으로 활용하세요"
    }
    
    # 역할 정의
    ROLES = {
        "pharmacist": "당신은 친근하고 전문적인 약사입니다.",
        "pharmacist_friendly": "당신은 친근한 약사로, 사용자와 편하게 대화하듯이 의약품 정보를 제공합니다.",
        "expert": "당신은 의약품 정보 전문가입니다.",
        "safety_expert": "당신은 의약품 안전성 평가 전문가입니다.",
        "classifier": "당신은 약품 질문 분류 전문가입니다.",
        "refinement_expert": "당신은 의약품 상담 시스템의 질문 보정 전문가입니다.",
        "router": "당신은 의약품 상담 시스템의 라우팅 담당자입니다.",
        "context_analyst": "당신은 대화 맥락 분석 전문가입니다.",
        "medicine_classifier": "당신은 약품명/성분명 분류 전문가입니다.",
        "integration_expert": "당신은 다중 소스 의약품 정보 통합 전문가입니다."
    }
    
    # 섹션 구조
    SECTION_STRUCTURE = {
        "efficacy": "💊 효능 및 작용 원리",
        "precautions": "⚠️ 주의사항 및 부작용",
        "usage": "📋 사용법",
        "alternatives": "💡 대안 약품",
        "latest_info": "💡 추가 정보",
        "additional_info": "💡 추가 정보",
        "summary": "💡 주요 내용 요약"
    }
    
    # 마무리 문구
    FOOTERS = {
        "medical_consultation": "⚠️ **중요**: 정확한 진단과 처방을 위해서는 의사나 약사와 상담하시기 바랍니다.",
        "friendly_closing": "궁금한 거 있으면 언제든 물어보세요!",
        "helpful_closing": "도움이 되셨길 바랍니다!"
    }

# 동적으로 설정을 추가할 수 있는 함수들 (필요시에만 사용)
def update_search_config(key: str, value):
    """검색 설정 업데이트"""
    if hasattr(SearchConfig, key):
        setattr(SearchConfig, key, value)

def update_model_config(key: str, value):
    """모델 설정 업데이트"""
    if hasattr(ModelConfig, key):
        setattr(ModelConfig, key, value)