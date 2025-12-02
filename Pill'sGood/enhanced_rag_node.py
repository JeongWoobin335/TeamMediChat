# enhanced_rag_node.py - 향상된 RAG 노드

from qa_state import QAState
from enhanced_rag_system import EnhancedRAGSystem
from typing import Dict, List

def enhanced_rag_node(state: QAState) -> QAState:
    """향상된 RAG 노드 - 여러 DB에서 정보를 수집하고 조합하여 근거 있는 답변 생성"""
    
    # ⚠️ 중요: question_refinement_node에서 보정된 약품명이 있으면 우선 사용
    medicine_name = state.get("extracted_medicine_name") or state.get("medicine_name", "")
    usage_context = state.get("usage_context", "")
    
    if not medicine_name or not usage_context:
        state["enhanced_rag_answer"] = "죄송합니다. 약품명이나 사용 상황 정보가 부족하여 분석할 수 없습니다."
        return state
    
    # 보정된 약품명으로 state 업데이트 (다음 노드에서도 사용하도록)
    if state.get("extracted_medicine_name") and state.get("extracted_medicine_name") != state.get("medicine_name"):
        state["medicine_name"] = medicine_name
        print(f"✅ 보정된 약품명으로 state 업데이트: '{state.get('medicine_name', '')}' → '{medicine_name}'")
    
    print(f"🔍 향상된 RAG 분석 시작: {medicine_name} → {usage_context}")
    
    # 디버깅: state 전체 키 확인
    print(f"🔍 state에 저장된 키들: {list(state.keys())}")
    
    try:
        # 통합 RAG 시스템 초기화
        rag_system = EnhancedRAGSystem()
        
        # 병합된 약품 정보 확인 (medicine_usage_check_node에서 생성된 정보)
        merged_medicine_info = state.get("merged_medicine_info")
        print(f"🔍 merged_medicine_info 타입: {type(merged_medicine_info)}, 값: {merged_medicine_info is not None}")
        if merged_medicine_info:
            print(f"✅ 병합된 약품 정보 발견: {medicine_name} (효능: {len(str(merged_medicine_info.get('효능', '')))}자, 부작용: {len(str(merged_medicine_info.get('부작용', '')))}자)")
            print(f"📋 병합된 정보 미리보기 - 효능: {str(merged_medicine_info.get('효능', ''))[:100]}...")
            print(f"📋 병합된 정보 미리보기 - 부작용: {str(merged_medicine_info.get('부작용', ''))[:100]}...")
        else:
            print(f"⚠️ 병합된 약품 정보 없음, 직접 수집")
        
        # 종합 분석 수행 (병합된 정보 전달)
        analysis_result = rag_system.analyze_medicine_comprehensively(medicine_name, usage_context, merged_medicine_info)
        
        # 결과를 state에 저장
        state["enhanced_rag_analysis"] = analysis_result
        evidence_response = analysis_result.get("evidence_based_response", "분석을 완료할 수 없습니다.")
        state["enhanced_rag_answer"] = evidence_response
        state["follow_up_questions"] = analysis_result.get("follow_up_questions", [])
        
        # 디버깅: 생성된 답변 확인
        print(f"🔍 생성된 enhanced_rag_answer: {evidence_response[:200]}...")
        print(f"🔍 combined_analysis 존재: {'combined_analysis' in analysis_result}")
        if 'combined_analysis' in analysis_result:
            print(f"🔍 combined_analysis 내용: {analysis_result['combined_analysis']}")
        
        # 추가 정보 저장
        state["excel_info"] = analysis_result.get("excel_info", {})
        state["pdf_info"] = analysis_result.get("pdf_info", {})
        state["korean_ingredient_info"] = analysis_result.get("korean_ingredient_info", {})
        state["international_ingredient_info"] = analysis_result.get("international_ingredient_info", {})
        state["combined_analysis"] = analysis_result.get("combined_analysis", {})
        # 최신 정보 저장 (hallucination 노드에서 사용)
        state["youtube_info"] = analysis_result.get("youtube_info", {})
        state["naver_news_info"] = analysis_result.get("naver_news_info", {})
        
        print(f"✅ 향상된 RAG 분석 완료: {medicine_name}")
        
    except Exception as e:
        print(f"❌ 향상된 RAG 분석 오류: {e}")
        state["enhanced_rag_answer"] = f"분석 중 오류가 발생했습니다: {str(e)}"
        state["enhanced_rag_analysis"] = {"error": str(e)}
    
    return state

def generate_conversational_response(state: QAState) -> str:
    """대화형 응답 생성"""
    
    enhanced_answer = state.get("enhanced_rag_answer", "")
    follow_up_questions = state.get("follow_up_questions", [])
    
    if not enhanced_answer:
        return "죄송합니다. 답변을 생성할 수 없습니다."
    
    # 기본 답변
    response = enhanced_answer
    
    # 추가 질문이 있으면 추가
    if follow_up_questions:
        response += "\n\n**추가로 궁금한 점이 있으시다면:**\n"
        for i, question in enumerate(follow_up_questions[:3], 1):
            response += f"{i}. {question}\n"
        
        response += "\n💬 언제든지 질문해주세요!"
    
    return response
