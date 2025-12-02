from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph
from qa_state import QAState

# 노드 import 
from preprocess_node import preprocess_query_node
from question_refinement_node import question_refinement_node  # ChatGPT 기반 질문 보정 노드
from medicine_related_filter_node import medicine_related_filter_node
from route_question_node import route_question_node
from medicine_usage_check_node import medicine_usage_check_node  # 새로운 노드 추가
from ocr_node import ocr_image_node  # OCR 이미지 처리 노드 추가
from remember_clean_node import remember_previous_context_node
from pdf_node import pdf_search_node
from excel_node import excel_search_node
from external_node import external_search_node
from parallel_search_node import parallel_search_node
from rerank_check_node import rerank_node
from hallucination_node import hallucination_check_node
from requery_answer_node import requery_node
from generate_node import generate_final_answer_node
from conversational_answer_node import conversational_answer_node  # ChatGPT 기반 대화형 답변 재구성 노드
from sns_node import sns_search_node
from new_medicine_search_node import new_medicine_search_node  # 신약 관련 질문 전용 검색 노드
from enhanced_rag_node import enhanced_rag_node
from follow_up_question_node import follow_up_question_node

from dotenv import load_dotenv
from cache_manager import print_cache_stats

load_dotenv()

# 그래프 초기화
builder = StateGraph(QAState)

# 노드 등록
builder.add_node("preprocess", preprocess_query_node)
builder.add_node("question_refinement", question_refinement_node)  # ChatGPT 기반 질문 보정
builder.add_node("medicine_filter", medicine_related_filter_node)
builder.add_node("route", route_question_node)
builder.add_node("usage_check", medicine_usage_check_node)  # 새로운 노드 추가
builder.add_node("ocr_image", ocr_image_node)  # OCR 이미지 처리 노드 추가
builder.add_node("search", remember_previous_context_node)
builder.add_node("pdf_search", pdf_search_node)
builder.add_node("excel_search", excel_search_node)
builder.add_node("external_search", external_search_node)
builder.add_node("parallel_search", parallel_search_node)  # 병렬 검색 노드
builder.add_node("sns_search", sns_search_node)  # 기존 약품의 보조 정보 검색 (enhanced_rag에서 사용)
builder.add_node("new_medicine_search", new_medicine_search_node)  # 신약 관련 질문 전용 검색
builder.add_node("rerank", rerank_node)
builder.add_node("hallucination", hallucination_check_node)
builder.add_node("requery", requery_node)
builder.add_node("enhanced_rag", enhanced_rag_node)
builder.add_node("follow_up", follow_up_question_node)
builder.add_node("generate", generate_final_answer_node)
builder.add_node("conversational_answer", conversational_answer_node)  # ChatGPT 기반 대화형 답변 재구성

# 진입점 설정
builder.set_entry_point("preprocess")

# 흐름 연결 
# preprocess 후 바로 route로 이동 (신약 관련 질문을 먼저 감지)
builder.add_edge("preprocess", "route")

# route_question_node에서 분기
def route_decision(state: QAState):
    """신약 관련 질문인지 확인하여 분기"""
    routing_decision = state.get("routing_decision", "search")
    print(f"🎯 라우팅 결정: {routing_decision}")
    
    # 신약 관련 질문은 바로 new_medicine_search로 (question_refinement 건너뛰기)
    if routing_decision == "new_medicine_search":
        return "new_medicine_search"
    # 일반 질문은 question_refinement를 거쳐야 함
    else:
        return "question_refinement"

builder.add_conditional_edges("route", route_decision)

# 일반 질문 흐름: question_refinement → medicine_filter
builder.add_edge("question_refinement", "medicine_filter")

# medicine_filter 후 다시 분기 (원래 라우팅 결정 사용)
def route_after_refinement(state: QAState):
    """question_refinement 후 원래 라우팅 결정으로 분기"""
    routing_decision = state.get("routing_decision", "search")
    return routing_decision

builder.add_conditional_edges("medicine_filter", route_after_refinement)


# 약품 사용 가능성 판단 흐름: 사용 가능성 판단 후 향상된 RAG로
builder.add_edge("usage_check", "enhanced_rag")

# OCR 이미지 처리 흐름: OCR 처리 후 사용 가능성 판단으로 연결
builder.add_edge("ocr_image", "usage_check")

# 향상된 RAG 흐름: 향상된 RAG 후 hallucination 검사 후 generate로
builder.add_edge("enhanced_rag", "hallucination")

# 연속 질문 흐름: 연속 질문 처리 후 generate로
builder.add_edge("follow_up", "generate")

# 신약 검색 흐름: 신약 관련 질문 전용 검색 후 generate로
builder.add_edge("new_medicine_search", "generate")

# SNS 검색 흐름: 기존 약품의 보조 정보 검색 (enhanced_rag에서 사용)
builder.add_edge("sns_search", "generate")

# 일반 검색 흐름 (병렬 검색 사용)
# search 노드(remember_previous_context_node) 다음에 병렬 검색 실행
builder.add_edge("search", "parallel_search")
builder.add_edge("parallel_search", "rerank")
# 기존 순차 검색 경로는 유지하지 않음 (병렬 검색으로 대체)
builder.add_edge("rerank", "hallucination")

def hallucination_router(state: QAState):
    flag = state.get("hallucination_flag")
    return "requery" if flag else "generate"

builder.add_conditional_edges("hallucination", hallucination_router)
builder.add_edge("requery", "generate")

# 최종 답변 재구성: generate 후 ChatGPT로 자연스럽게 변환
builder.add_edge("generate", "conversational_answer")

# 종료점 설정
builder.set_finish_point("conversational_answer")

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
