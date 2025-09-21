from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph
from qa_state import QAState

# 노드 import 
from preprocess_node import preprocess_query_node
from medicine_related_filter_node import medicine_related_filter_node
from route_question_node import route_question_node
from recommend_medicine_node import recommend_medicine_node
from medicine_usage_check_node import medicine_usage_check_node  # 새로운 노드 추가
from ocr_node import ocr_image_node  # OCR 이미지 처리 노드 추가
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
builder.add_node("usage_check", medicine_usage_check_node)  # 새로운 노드 추가
builder.add_node("ocr_image", ocr_image_node)  # OCR 이미지 처리 노드 추가
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

# route_question_node에서 분기
def route_decision(state: QAState):
    routing_decision = state.get("routing_decision", "search")
    print(f"🎯 라우팅 결정: {routing_decision}")
    return routing_decision

builder.add_conditional_edges("route", route_decision)

# 추천 흐름: 추천 후 곧바로 generate
builder.add_edge("recommend", "generate")

# 약품 사용 가능성 판단 흐름: 사용 가능성 판단 후 곧바로 generate
builder.add_edge("usage_check", "generate")

# OCR 이미지 처리 흐름: OCR 처리 후 사용 가능성 판단으로 연결
builder.add_edge("ocr_image", "usage_check")

# SNS 검색 흐름 
builder.add_edge("sns_search", "generate")

# 일반 검색 흐름
builder.add_edge("excel_search", "rerank")
builder.add_edge("pdf_search", "rerank")
builder.add_edge("external_search", "rerank")
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

# 실시간 대화 모드
if __name__ == "__main__":
    import sys
    
    print("🏥 TeamMediChat - 실시간 대화 모드")
    print("=" * 60)
    
    try:
        from chat_interface import ChatInterface
        chat_interface = ChatInterface()
        chat_interface.run()
    except ImportError as e:
        print(f"❌ 채팅 인터페이스를 불러올 수 없습니다: {e}")
        print("💡 채팅 인터페이스를 사용하려면 필요한 파일들이 모두 있는지 확인하세요.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 채팅 인터페이스 실행 중 오류 발생: {e}")
        sys.exit(1)
