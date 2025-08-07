from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph
from qa_state import QAState

# 노드 import 
from preprocess_node import preprocess_query_node
from medicine_related_filter_node import medicine_related_filter_node
from route_question_node import route_question_node
from recommend_medicine_node import recommend_medicine_node
from remember_clean_node import remember_previous_context_node
from pdf_node import pdf_search_node
from excel_node import excel_search_node
from external_node import external_search_node
from rerank_check_node import rerank_node
from hallucination_node import hallucination_check_node
from requery_answer_node import requery_node
from generate_node import generate_final_answer_node
from sns_node import sns_search_node
from dotenv import load_dotenv
from cache_manager import print_cache_stats



load_dotenv()

# 그래프 초기화
builder = StateGraph(QAState)

# 노드 등록
builder.add_node("preprocess", preprocess_query_node)
builder.add_node("medicine_filter", medicine_related_filter_node)
builder.add_node("route", route_question_node)
builder.add_node("recommend", recommend_medicine_node)
builder.add_node("search", remember_previous_context_node)
builder.add_node("pdf_search", pdf_search_node)
builder.add_node("excel_search", excel_search_node)
builder.add_node("external_search", external_search_node)
builder.add_node("sns_search", sns_search_node)
builder.add_node("rerank", rerank_node)
builder.add_node("hallucination", hallucination_check_node)
builder.add_node("requery", requery_node)
builder.add_node("generate", generate_final_answer_node)

# 진입점 설정
builder.set_entry_point("preprocess")

# 흐름 연결
builder.add_edge("preprocess", "medicine_filter")
builder.add_edge("medicine_filter", "route")

# 라우팅 분기
def route_decision_router(state: QAState):
    return state["routing_decision"]

builder.add_conditional_edges("route", route_decision_router)

# 추천 흐름: 추천 후 곧바로 generate
builder.add_edge("recommend", "generate")

# 정보 검색 흐름 - 최신 정보 요청 감지
def search_router(state: QAState):
    category = state.get("category", "")
    query = (state.get("query", "") or "").lower()
    
    # 최신 약품 정보 요청인 경우 SNS 검색 우선 (PDF/Excel 건너뛰기)
    latest_keywords = ["2024", "2023", "새로", "신약", "fda", "승인", "최신", "새로운", "경험담", "후기", 
                      "latest", "new", "recent", "experience", "review", "side effect"]
    if (category == "최신 약품" or 
        any(keyword in query for keyword in latest_keywords)):
        return "external_search"
    
    # 일반 정보 요청인 경우 PDF 검색부터 시작
    return "pdf_search"

builder.add_conditional_edges("search", search_router)

def pdf_router(state: QAState):
    query = (state.get("cleaned_query") or state.get("normalized_query") or "").lower()
    docs = state.get("pdf_results") or []
    return "rerank" if any(query in doc.page_content.lower() for doc in docs) else "excel_search"

builder.add_conditional_edges("pdf_search", pdf_router)

def excel_router(state: QAState):
    docs = state.get("excel_results") or []
    return "rerank" if len(docs) > 0 else "external_search"

builder.add_conditional_edges("excel_search", excel_router)

# 공통 후속 흐름
builder.add_edge("external_search", "sns_search")
builder.add_edge("sns_search", "rerank")
builder.add_edge("rerank", "hallucination")

def hallucination_router(state: QAState):
    flag = state.get("hallucination_flag")
    return "requery" if flag else "generate"

builder.add_conditional_edges("hallucination", hallucination_router)
builder.add_edge("requery", "generate")

# 종료점 설정
builder.set_finish_point("generate")

# 그래프 컴파일
graph = builder.compile()

# 테스트 실행
if __name__ == "__main__":
    # 캐시 통계 출력
    print_cache_stats()
    
    # 테스트 쿼리 (피곤함만 테스트)
    test_queries = [
        "너무 지쳐서 약 먹고 싶은데 뭐가 좋을까?"
    ]
    
    for i, test_query in enumerate(test_queries, 1):
        test_state = QAState(query=test_query)
        result = graph.invoke(test_state)
        print(f"\n=== 테스트 {i} 결과 ===")
        print(f"질문: {test_query}")
        print(f"답변: {result.get('final_answer', '답변 없음')}")
        print("=" * 50)
    
    # 테스트 후 캐시 통계 다시 출력
    print_cache_stats()
