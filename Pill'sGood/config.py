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

# 동적으로 설정을 추가할 수 있는 함수들 (필요시에만 사용)
def update_search_config(key: str, value):
    """검색 설정 업데이트"""
    if hasattr(SearchConfig, key):
        setattr(SearchConfig, key, value)

def update_model_config(key: str, value):
    """모델 설정 업데이트"""
    if hasattr(ModelConfig, key):
        setattr(ModelConfig, key, value)