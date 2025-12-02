# parallel_search_node.py - 병렬 검색 노드
# PDF, Excel, External 검색을 병렬로 실행

from qa_state import QAState
from concurrent.futures import ThreadPoolExecutor, as_completed
from pdf_node import pdf_search_node
from excel_node import excel_search_node
from external_node import external_search_node
from typing import Dict, Any

def parallel_search_node(state: QAState) -> QAState:
    """PDF, Excel, External 검색을 병렬로 실행하는 노드"""
    query = state.get("cleaned_query") or state.get("normalized_query")
    if not query:
        state["pdf_results"] = []
        state["excel_results"] = []
        state["external_raw"] = None
        state["external_parsed"] = None
        return state
    
    print("🔄 병렬 검색 시작 (PDF, Excel, External)...")
    
    # 각 검색 작업을 병렬로 실행
    def run_pdf_search():
        """PDF 검색 실행"""
        try:
            # state를 dict로 변환하여 전달 (각 노드는 state를 수정하고 반환)
            pdf_state = pdf_search_node(dict(state))
            return ('pdf', pdf_state.get("pdf_results", []))
        except Exception as e:
            print(f"⚠️ PDF 검색 오류: {e}")
            return ('pdf', [])
    
    def run_excel_search():
        """Excel 검색 실행"""
        try:
            excel_state = excel_search_node(dict(state))
            return ('excel', excel_state.get("excel_results", []))
        except Exception as e:
            print(f"⚠️ Excel 검색 오류: {e}")
            return ('excel', [])
    
    def run_external_search():
        """External 검색 실행"""
        try:
            external_state = external_search_node(dict(state))
            return ('external', {
                'raw': external_state.get("external_raw"),
                'parsed': external_state.get("external_parsed")
            })
        except Exception as e:
            print(f"⚠️ External 검색 오류: {e}")
            return ('external', {'raw': None, 'parsed': None})
    
    # 병렬 실행
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_pdf_search): 'pdf',
            executor.submit(run_excel_search): 'excel',
            executor.submit(run_external_search): 'external'
        }
        
        # 결과 수집
        for future in as_completed(futures):
            search_type = futures[future]
            try:
                result_type, result_data = future.result()
                
                if result_type == 'pdf':
                    state["pdf_results"] = result_data
                    print(f"  ✅ PDF 검색 완료: {len(result_data)}개 결과")
                elif result_type == 'excel':
                    state["excel_results"] = result_data
                    print(f"  ✅ Excel 검색 완료: {len(result_data)}개 결과")
                elif result_type == 'external':
                    state["external_raw"] = result_data.get('raw')
                    state["external_parsed"] = result_data.get('parsed')
                    print(f"  ✅ External 검색 완료")
                    
            except Exception as e:
                print(f"  ❌ {search_type} 검색 실패: {e}")
                # 기본값 설정
                if search_type == 'pdf':
                    state["pdf_results"] = []
                elif search_type == 'excel':
                    state["excel_results"] = []
                elif search_type == 'external':
                    state["external_raw"] = None
                    state["external_parsed"] = None
    
    print("✅ 병렬 검색 완료")
    return state

