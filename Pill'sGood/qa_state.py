from typing import Optional, List, TypedDict
from langchain_core.documents import Document

class QAState(TypedDict, total=False):
    """
    LangGraph 기반 의약품 QA 시스템의 전체 상태를 관리하는 딕셔너리 정의입니다.
    각 노드에서 참조하거나 수정할 수 있는 주요 필드들은 다음과 같습니다:

    - 사용자 입력 관련
        - query: 사용자 원 질문
        - cleaned_query: 약품명 정제 결과
        - normalized_query: 약품명 정규화 결과

    - 전처리 정보
        - condition: 병력 정보 (ex. 위장염)
        - category: 약물 카테고리 (ex. 감기약)
        - requested_fields: 사용자가 궁금해하는 항목들 (효능, 부작용 등)

    - 추천 흐름
        - recommendation_answer: 병력 기반 약품 추천 결과

    - 관련성 판단
        - is_medicine_related: 약 관련 질문인지 여부 판단 결과

    - 과거 질문 활용
        - previous_context: 이전 질문 맥락

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

    - 최종 생성 결과
        - final_answer: LLM이 생성한 최종 응답 텍스트
    """

    # 사용자 입력 관련
    query: str
    cleaned_query: Optional[str]
    normalized_query: Optional[str]

    # 전처리 정보
    condition: Optional[str]
    category: Optional[str]
    requested_fields: Optional[List[str]]

    # 추천 흐름
    recommendation_answer: Optional[str]

    # 관련성 판단
    is_medicine_related: Optional[bool]

    # 과거 질문 활용
    previous_context: Optional[str]

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

    routing_decision: Optional[str]

    # 최종 생성 결과
    final_answer: Optional[str]