from qa_state import QAState
from retrievers import compressor  # CrossEncoder 기반 문서 압축기
from langchain_core.documents import Document
from typing import List
import re

def normalize(text: str) -> str:
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"[^\w가-힣]", "", text)
    return re.sub(r"\s+", "", text.strip().lower())

def contains_product_name(doc: Document, product_name: str) -> bool:
    return normalize(product_name) == normalize(doc.metadata.get("제품명", ""))

def rerank_node(state: QAState) -> QAState:
    all_docs: List[Document] = []
    if state.get("pdf_results"):
        all_docs.extend(state["pdf_results"])
    if state.get("excel_results"):
        all_docs.extend(state["excel_results"])
    if state.get("sns_results"):
        all_docs.extend(state["sns_results"])

    if not all_docs:
        # 최신 정보 요청인 경우 특별한 응답 생성
        query = state.get("query", "")
        category = state.get("category", "")
        
        if "최신" in query or "새로" in query or "2024" in query or "2023" in query:
            latest_info_response = f"""안녕하세요! '{query}'에 대한 최신 정보를 요청하셨군요.

현재 시스템에서 해당 정보를 찾을 수 없어서, 다음과 같은 방법으로 최신 정보를 확인하실 수 있습니다:

🔍 **추천 검색 방법:**
1. **의료 전문 사이트**: 식약처, FDA 공식 웹사이트
2. **의료 데이터베이스**: PubMed, ClinicalTrials.gov
3. **제약사 공식 발표**: 관련 제약회사 공식 보도자료
4. **의료 전문 매체**: 의학 저널, 의료 뉴스

💡 **참고사항:**
- 최신 약품 정보는 지속적으로 업데이트됩니다
- 정확한 정보는 의사나 약사와 상담하시는 것을 권장합니다
- 임상시험 중인 약품의 경우 승인 상태가 변경될 수 있습니다

더 구체적인 정보가 필요하시면 언제든지 다시 질문해 주세요!"""
            
            state["final_answer"] = latest_info_response
            state["relevant_docs"] = []
            state["reranked_docs"] = []
            return state
        
        state["reranked_docs"] = []
        state["relevant_docs"] = []
        return state

    try:
        query = state.get("query", "")
        category = state.get("category", "")

        # ✅ LLM 기반 문서 필터링
        
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        # 문서 관련성 판단 프롬프트
        relevance_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 사용자의 질문과 문서의 관련성을 판단하는 전문가입니다.

사용자 질문: {query}

다음 문서들이 사용자의 질문과 관련이 있는지 판단하세요.
관련성이 높은 문서만 선택하여 JSON 형태로 응답하세요.

응답 형식:
{{
  "relevant_docs": [
    {{
      "index": 0,
      "reason": "이유"
    }}
  ]
}}

관련성이 없다면 빈 배열을 반환하세요."""),
            ("human", "문서들:\n{docs}")
        ])
        
        # 문서들을 문자열로 변환
        docs_text = ""
        for i, doc in enumerate(all_docs[:10]):  # 최대 10개만
            docs_text += f"[{i}] {doc.page_content[:300]}...\n\n"
        
        try:
            # LLM이 관련 문서 판단
            relevance_result = llm.invoke(relevance_prompt.format(
                query=query,
                docs=docs_text
            ))
            
            # JSON 파싱 (더 안전한 방식)
            import json
            import re
            
            try:
                # JSON 부분만 추출
                content = relevance_result.content
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                
                if json_match:
                    result_data = json.loads(json_match.group())
                    relevant_indices = [item["index"] for item in result_data.get("relevant_docs", [])]
                    
                    if relevant_indices:
                        filtered_docs = [all_docs[i] for i in relevant_indices if i < len(all_docs)]
                        state["relevant_docs"] = filtered_docs[:3]
                        state["reranked_docs"] = []
                        return state
                    else:
                        # SNS 검색 결과가 있으면 우선 사용
                        sns_docs = state.get("sns_results", [])
                        if sns_docs:
                            state["relevant_docs"] = sns_docs[:3]
                            state["reranked_docs"] = []
                            return state
                        else:
                            # 최신 정보 요청에 대한 응답 직접 생성
                            query = state.get("query", "")
                            latest_info_response = f"""안녕하세요! '{query}'에 대한 최신 정보를 요청하셨군요.

현재 시스템에서 해당 정보를 찾을 수 없어서, 다음과 같은 방법으로 최신 정보를 확인하실 수 있습니다:

🔍 **추천 검색 방법:**
1. **의료 전문 사이트**: 식약처, FDA 공식 웹사이트
2. **의료 데이터베이스**: PubMed, ClinicalTrials.gov
3. **제약사 공식 발표**: 관련 제약회사 공식 보도자료
4. **의료 전문 매체**: 의학 저널, 의료 뉴스

💡 **참고사항:**
- 최신 약품 정보는 지속적으로 업데이트됩니다
- 정확한 정보는 의사나 약사와 상담하시는 것을 권장합니다
- 임상시험 중인 약품의 경우 승인 상태가 변경될 수 있습니다

더 구체적인 정보가 필요하시면 언제든지 다시 질문해 주세요!"""
                            
                            state["final_answer"] = latest_info_response
                            state["relevant_docs"] = []
                            state["reranked_docs"] = []
                            return state
                else:
                    pass
            except Exception as e:
                pass
        except Exception as e:
            pass

        # ✅ 기존 제품명 매칭 로직
        product_name = state.get("normalized_query") or state.get("cleaned_query")
        product_name = normalize(product_name or "")

        excel_docs = state.get("excel_results", [])
        excel_matched = [doc for doc in excel_docs if contains_product_name(doc, product_name)]

        if excel_matched:
            state["relevant_docs"] = excel_matched[:3]
            state["reranked_docs"] = []
            return state

        # Excel에 없으면 전체 문서 리랭킹
        reranked = compressor.compress_documents(all_docs, query=query)
        state["reranked_docs"] = reranked

        filtered = [doc for doc in reranked if contains_product_name(doc, product_name)]

        if filtered:
            deduped = []
            seen = set()
            for doc in filtered + reranked:
                key = doc.page_content[:100]
                if key not in seen:
                    deduped.append(doc)
                    seen.add(key)
                if len(deduped) >= 3:
                    break
            state["relevant_docs"] = deduped
        else:
            state["relevant_docs"] = reranked[:3]

    except Exception as e:
        state["reranked_docs"] = []
        state["relevant_docs"] = []

    return state
