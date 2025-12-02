from typing import Optional, List, TypedDict
from langchain_core.documents import Document

class QAState(TypedDict, total=False):
    """
    LangGraph 기반 의약품 QA 시스템의 전체 상태를 관리하는 딕셔너리 정의입니다.
    각 노드에서 참조하거나 수정할 수 있는 주요 필드들은 다음과 같습니다:

    - 사용자 입력 관련
        - query: 사용자 원 질문 (ChatGPT 보정 후)
        - original_query: 사용자 원본 질문 (보정 전)
        - query_was_refined: 질문 보정 여부 (Optional[bool])
        - cleaned_query: 약품명 정제 결과
        - normalized_query: 약품명 정규화 결과

    - 전처리 정보
        - condition: 병력 정보 (ex. 위장염)
        - category: 약물 카테고리 (ex. 감기약)
        - requested_fields: 사용자가 궁금해하는 항목들 (효능, 부작용 등)

    - 추천 흐름
        - recommendation_answer: 병력 기반 약품 추천 결과
    
    - 약품 사용 가능성 판단 흐름 (새로 추가)
        - medicine_name: 사용자가 언급한 약품명
        - usage_context: 약품을 사용하려는 상황/증상
        - usage_check_answer: 약품 사용 가능성 판단 결과

    - 관련성 판단
        - is_medicine_related: 약 관련 질문인지 여부 판단 결과

    - 과거 질문 활용
        - previous_context: 이전 질문 맥락
        - conversation_context: 전체 대화 맥락
        - user_context: 사용자 질문 맥락
        - session_id: 현재 대화 세션 ID

    - 대화 맥락 분석 (새로 추가)
        - has_medicine_recommendation: Optional[bool]
        - is_asking_about_previous: Optional[bool]
        - extracted_medicines: Optional[dict]
        - context_reasoning: Optional[str]

    - 검색 결과
        - pdf_results: PDF 검색 결과 문서 리스트
        - excel_results: Excel 검색 결과 문서 리스트
        - external_raw: 외부 검색의 원문 텍스트
        - external_parsed: 외부 검색을 LLM이 정제한 JSON 구조
        - sns_results: SNS(레딧) 검색 결과 문서 리스트
        - sns_count: SNS 검색 결과 개수

    - 후처리 및 평가
        - reranked_docs: 리랭킹 결과 문서 리스트
        - relevant_docs: 문서 평가 후 관련성 높은 문서 리스트
        - hallucination_flag: 환각 여부 판단 결과
        - re_query: 재검색 시 새로 생성된 질문

    - LLM 기반 라우팅 (새로 추가)
        - routing_decision: Optional[str]
        - routing_reasoning: Optional[str]
        - context_analysis: Optional[str]
        - user_intent: Optional[str]
        - search_decision: Optional[str]
        - search_reasoning: Optional[str]
        - search_strategy: Optional[str]

    - 최종 생성 결과
        - final_answer: LLM이 생성한 최종 응답 텍스트 (ChatGPT 재구성 후)
        - original_answer: 원본 답변 (ChatGPT 재구성 전)
        - answer_was_polished: 답변 재구성 여부 (Optional[bool])
    """

    # 사용자 입력 관련
    query: str
    original_query: Optional[str]  # 원본 질문 (ChatGPT 보정 전)
    query_was_refined: Optional[bool]  # 질문 보정 여부
    cleaned_query: Optional[str]
    normalized_query: Optional[str]
    
    # OCR 이미지 처리 관련
    image_data: Optional[bytes]  # 업로드된 이미지 데이터
    extracted_text: Optional[str]  # OCR로 추출된 텍스트
    
    # 사용자 위치 정보
    user_location: Optional[dict]  # 사용자 위치 정보 (lat, lng, timestamp)

    # 전처리 정보
    condition: Optional[str]
    category: Optional[str]
    requested_fields: Optional[List[str]]

    # 추천 흐름
    recommendation_answer: Optional[str]
    
    # 약품 사용 가능성 판단 흐름 (새로 추가)
    medicine_name: Optional[str]
    usage_context: Optional[str]
    usage_check_answer: Optional[str]

    # 관련성 판단
    is_medicine_related: Optional[bool]

    # 과거 질문 활용
    previous_context: Optional[str]
    conversation_context: Optional[str]
    user_context: Optional[str]
    session_id: Optional[str]
    
    # 대화 맥락 분석 (새로 추가)
    has_medicine_recommendation: Optional[bool]
    is_asking_about_previous: Optional[bool]
    extracted_medicines: Optional[dict]
    context_reasoning: Optional[str]

    # 검색 결과
    pdf_results: Optional[List[Document]]
    excel_results: Optional[List[Document]]
    external_raw: Optional[str]
    external_parsed: Optional[dict]
    sns_results: Optional[List[Document]]
    sns_count: Optional[int]

    # 후처리 및 평가
    reranked_docs: Optional[List[Document]]
    relevant_docs: Optional[List[Document]]
    hallucination_flag: Optional[bool]
    re_query: Optional[str]

    # LLM 기반 라우팅 (새로 추가)
    routing_decision: Optional[str]
    routing_reasoning: Optional[str]
    context_analysis: Optional[str]
    user_intent: Optional[str]
    search_decision: Optional[str]
    search_reasoning: Optional[str]
    search_strategy: Optional[str]
    
    # 연속 질문 처리 (새로 추가)
    is_follow_up: Optional[bool]
    follow_up_type: Optional[str]

    # 향상된 RAG 결과
    enhanced_rag_answer: Optional[str]
    enhanced_rag_analysis: Optional[dict]
    follow_up_questions: Optional[List[str]]
    excel_info: Optional[dict]
    pdf_info: Optional[dict]
    korean_ingredient_info: Optional[dict]
    international_ingredient_info: Optional[dict]
    combined_analysis: Optional[dict]
    merged_medicine_info: Optional[dict]  # 병합된 약품 정보 (PDF 포함)

    # 최종 생성 결과
    final_answer: Optional[str]
    original_answer: Optional[str]  # 원본 답변 (ChatGPT 재구성 전)
    answer_was_polished: Optional[bool]  # 답변 재구성 여부