# preprocess_node.py

from qa_state import QAState
import re

# 유틸 함수
def clean_product_name(query: str) -> str:
    query = re.sub(r"(의|에 대해.*|알려줘|무엇입니까|뭔가요|뭐야|어떻게.*|사용법|부작용|효능|복용.*|섭취.*|투여.*)", "", query)
    query = re.sub(r"[^\w가-힣]", "", query)
    query = re.sub(r"(은|는|이|가|을|를)$", "", query)
    return query.strip()

def normalize(text: str) -> str:
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"[^\w가-힣]", "", text)
    return re.sub(r"\s+", "", text.strip().lower())

# 노드 함수
def preprocess_query_node(state: QAState) -> QAState:
    """
    사용자 질문에서 약품명을 정제(cleaned_query) 및 정규화(normalized_query)하여 상태에 저장합니다.
    """
    query = state["query"]
    cleaned = clean_product_name(query)
    normalized_query = normalize(cleaned)

    state["cleaned_query"] = cleaned
    state["normalized_query"] = normalized_query

    return state
