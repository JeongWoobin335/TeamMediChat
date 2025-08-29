# recommend_medicine_node.py

from qa_state import QAState
from retrievers import llm, pdf_structured_docs, excel_docs
from langchain_core.documents import Document
from typing import List
import json
import re
from cache_manager import cache_manager

# 신체 부위별 위험 키워드 맵
body_part_risk_map = {
    "위장": ["위장 자극", "속쓰림", "구토", "소화불량", "위통", "복통", "메스꺼움"],
    "심장": ["심장 부담", "혈압 상승", "심박수 증가", "부정맥", "협심증"],
    "간": ["간독성", "간 손상", "간수치 상승", "황달"],
    "신장": ["신장 독성", "신장 손상", "소변량 감소", "부종"],
    "폐": ["호흡곤란", "기침", "천식 악화", "폐부종"],
    "뇌": ["두통", "어지럼증", "졸음", "집중력 저하", "기억력 저하"]
}

# 병력 위험 키워드 맵 (기존 호환성 유지)
condition_risk_map = {
    "위장염": body_part_risk_map["위장"],
    "간질환": body_part_risk_map["간"],
    "고혈압": body_part_risk_map["심장"],
    "감기": []  # 감기는 기본적으로 위험 키워드가 없음
}

# 신체 부위 감지 함수
def detect_body_part_concern(query: str) -> tuple[str, list[str]]:
    """질문에서 신체 부위별 우려사항을 감지합니다."""
    query_lower = query.lower()
    
    # 신체 부위별 키워드 매핑
    body_part_keywords = {
        "위장": ["위장", "위", "속", "배", "복부"],
        "심장": ["심장", "심", "혈압", "혈관"],
        "간": ["간", "간장"],
        "신장": ["신장", "콩팥", "소변"],
        "폐": ["폐", "호흡", "숨"],
        "뇌": ["뇌", "머리", "정신", "집중"]
    }
    
    # 부담/자극 관련 키워드
    concern_keywords = ["부담", "자극", "나쁨", "안좋음", "민감", "약함", "상처"]
    
    detected_parts = []
    for part, keywords in body_part_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            if any(concern in query_lower for concern in concern_keywords):
                detected_parts.append(part)
    
    return detected_parts[0] if detected_parts else "", detected_parts

# 필드 추출 함수 (개선된 버전)
def extract_field_from_doc(text: str, label: str) -> str:
    pattern = rf"\[{label}\]:\s*((?:.|\n)*?)(?=\n\[|\Z)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else "정보 없음"

# 약품별 정보 수집 함수 (새로운 청크 구조 지원)
def collect_medicine_info(product_name: str, all_docs: List[Document]) -> dict:
    """약품별로 효능, 부작용, 사용법 정보를 수집"""
    info = {
        "제품명": product_name,
        "효능": "정보 없음",
        "부작용": "정보 없음", 
        "사용법": "정보 없음"
    }
    
    # 해당 약품의 모든 문서 찾기
    product_docs = [doc for doc in all_docs if doc.metadata.get("제품명") == product_name]
    
    for doc in product_docs:
        content = doc.page_content
        doc_type = doc.metadata.get("type", "")
        
        # 효능과 부작용은 main 타입에서 추출
        if doc_type == "main" or doc_type == "":
            efficacy = extract_field_from_doc(content, "효능")
            side_effects = extract_field_from_doc(content, "부작용")
            
            if efficacy != "정보 없음":
                info["효능"] = efficacy
            if side_effects != "정보 없음":
                info["부작용"] = side_effects
        
        # 사용법은 usage 타입에서 추출
        if doc_type == "usage":
            usage = extract_field_from_doc(content, "사용법")
            if usage != "정보 없음":
                info["사용법"] = usage
    
    return info

# 배치 처리로 약품-증상 매칭 판단
def batch_medicine_matching(medicines_info, condition, batch_size=20):
    """여러 약품을 한 번에 묶어서 LLM이 증상 관련성을 판단"""
    if not medicines_info:
        return {}
    
    # 캐시에서 먼저 확인
    cached_result = cache_manager.get_matching_cache(condition, medicines_info)
    if cached_result is not None:
        return cached_result
    
    # 약품이 너무 많으면 배치로 나누어 처리
    medicines_list = list(medicines_info.items())
    all_results = {}
    
    for i in range(0, len(medicines_list), batch_size):
        batch_medicines = dict(medicines_list[i:i + batch_size])
        
        # 배치별 캐시 확인
        batch_cached = cache_manager.get_matching_cache(condition, batch_medicines)
        if batch_cached is not None:
            all_results.update(batch_cached)
            continue
        
        # 배치 프롬프트 생성
        medicines_text = ""
        for j, (name, info) in enumerate(batch_medicines.items(), 1):
            medicines_text += f"{j}. {name}: {info['효능']}\n"
        
        batch_prompt = f"""
        다음 약품들 중 '{condition}' 증상에 도움이 될 수 있는 약품들을 찾아주세요.
        
        {medicines_text}
        
        답변은 번호만 쉼표로 구분해서 작성해주세요. (예: 1,3,5)
        도움이 되는 약품이 없으면 '없음'이라고 답변해주세요.
        """
        
        try:
            response = llm.invoke(batch_prompt).content.strip()
            
            # 응답 파싱
            if "없음" in response or "none" in response.lower():
                batch_result = {}
            else:
                # 번호 추출
                import re
                numbers = re.findall(r'\d+', response)
                batch_result = {}
                
                for num in numbers:
                    idx = int(num) - 1
                    if idx < len(batch_medicines):
                        medicine_name = list(batch_medicines.keys())[idx]
                        batch_result[medicine_name] = True
            
            # 배치 결과를 캐시에 저장
            cache_manager.save_matching_cache(condition, batch_medicines, batch_result)
            
            # 전체 결과에 추가
            all_results.update(batch_result)
            
        except Exception as e:
            print(f"배치 처리 실패 (배치 {i//batch_size + 1}): {e}")
            continue
    
    return all_results

# 응답 생성 프롬프트
def generate_recommendation_response(condition, category, candidates, primary_concern=None):
    # 신체 부위별 근거 설명 추가
    concern_explanation = ""
    if primary_concern:
        concern_explanation = f"""
**중요**: 사용자가 '{primary_concern}'에 부담이 적은 약을 요청했습니다. 
부작용 섹션에서 '{primary_concern}' 관련 키워드가 적은 약품을 우선적으로 추천하고, 
왜 해당 약품이 '{primary_concern}'에 부담이 적은지 구체적인 근거를 제시해주세요.
"""
    
    prompt = f"""
당신은 건강 상태에 맞는 약을 추천해주는 건강 상담사입니다.
사용자는 '{condition}' 병력을 가지고 있으며, '{category}'에 도움이 되는 약을 찾고 있습니다.

{concern_explanation}

다음은 추천 가능한 후보 목록입니다. 각 약품을 친절하게 소개하고 병력과 관련된 이유를 함께 설명해주세요.
특히 부작용 정보를 바탕으로 왜 해당 약품이 적합한지 구체적인 근거를 제시해주세요.

{json.dumps(candidates[:3], ensure_ascii=False)}

**응답 형식**:
1. 각 약품별로 효능, 부작용, 사용법을 명확히 설명
2. 부작용 섹션에서 실제 언급된 내용을 바탕으로 신체 부위별 부담도 설명
3. 구체적인 근거 없이 "부담이 적다"고 함부로 말하지 말고, 실제 부작용 내용을 인용
4. 데이터 소스(PDF/Excel)도 언급하여 신뢰성 표시

답변:
"""
    return llm.invoke(prompt).content.strip()

# LangGraph 노드 정의
def recommend_medicine_node(state: QAState) -> QAState:
    condition = state.get("condition", "")
    category = state.get("category", "")
    query = state.get("query", "").lower()
    
    if not condition or not category:
        state["recommendation_answer"] = "죄송합니다. 병력 또는 약물 종류 정보가 없어 추천이 어렵습니다."
        return state

    # condition이 리스트인 경우 처리
    if isinstance(condition, list):
        conditions = condition
    else:
        conditions = [condition] if condition else []
    
    # 신체 부위별 우려사항 감지
    primary_concern, all_concerns = detect_body_part_concern(query)

    # 모든 조건에 대한 위험 키워드 수집
    all_risk_keywords = []
    for cond in conditions:
        if cond in condition_risk_map:
            all_risk_keywords.extend(condition_risk_map[cond])
    
    # 중복 제거
    all_risk_keywords = list(set(all_risk_keywords))
    
    candidates = []
    
    # Excel DB를 우선으로 검색 (Excel 우선 정책)
    print(f"🔍 Excel DB 우선 검색 시작: {len(excel_docs)}개 문서")
    
    # Excel에서 먼저 약품 정보 수집
    excel_medicines_info = {}
    for doc in excel_docs:
        name = doc.metadata.get("제품명", "")
        if name and name not in excel_medicines_info:
            medicine_info = collect_medicine_info(name, excel_docs)
            excel_medicines_info[name] = medicine_info
    
    print(f"✅ Excel DB에서 {len(excel_medicines_info)}개 약품 정보 수집")
    
    # Excel DB에서 먼저 매칭 시도
    for condition in conditions:
        if excel_medicines_info:
            print(f"🔍 Excel DB에서 {condition} 증상 매칭 시도...")
            excel_relevant_medicines = batch_medicine_matching(excel_medicines_info, condition, batch_size=15)
            
            # Excel DB에서 매칭된 약품들을 candidates에 추가
            for name, is_relevant in excel_relevant_medicines.items():
                if is_relevant and name in excel_medicines_info:
                    medicine_info = excel_medicines_info[name]
                    
                    # 신체 부위별 우려사항이 있는 경우 해당 부위 부작용이 적은 약품 우선
                    if primary_concern:
                        concern_risk_keywords = body_part_risk_map.get(primary_concern, [])
                        concern_risk_count = sum(1 for risk in concern_risk_keywords if risk in medicine_info["부작용"].lower())
                        
                        # 해당 부위 부작용이 적은 약품만 선택
                        if concern_risk_count <= 1:
                            candidates.append({
                                "제품명": name,
                                "효능": medicine_info["효능"],
                                "부작용": medicine_info["부작용"],
                                "사용법": medicine_info["사용법"],
                                f"{primary_concern}_부담도": concern_risk_count,
                                "데이터_소스": "Excel"
                            })
                            print(f"✅ Excel 후보 추가: {name}")
                    else:
                        # 일반적인 위험 키워드 필터링
                        if not any(risk in medicine_info["부작용"].lower() for risk in all_risk_keywords):
                            candidates.append({
                                "제품명": name,
                                "효능": medicine_info["효능"],
                                "부작용": medicine_info["부작용"],
                                "사용법": medicine_info["사용법"],
                                "데이터_소스": "Excel"
                            })
                            print(f"✅ Excel 후보 추가: {name}")
    
    # Excel DB에서 충분한 정보를 찾지 못한 경우에만 PDF 보완
    if len(candidates) < 3:  # 최소 3개 이상의 약품이 필요
        print(f"🔍 Excel DB에서 {len(candidates)}개 약품만 찾음, PDF DB 보완 검색...")
        
        # PDF에서 추가 검색
        pdf_medicines_info = {}
        for doc in pdf_structured_docs:
            name = doc.metadata.get("제품명", "")
            if name and name not in pdf_medicines_info:
                medicine_info = collect_medicine_info(name, pdf_structured_docs)
                pdf_medicines_info[name] = medicine_info
        
        print(f"✅ PDF DB에서 {len(pdf_medicines_info)}개 약품 정보 수집")
        
        # PDF에서 추가 매칭
        for condition in conditions:
            if pdf_medicines_info:
                print(f"🔍 PDF DB에서 {condition} 증상 추가 매칭...")
                pdf_relevant_medicines = batch_medicine_matching(pdf_medicines_info, condition, batch_size=10)
                
                # 이미 Excel에서 찾은 약품은 제외하고 PDF에서만 찾은 약품 추가
                for name, is_relevant in pdf_relevant_medicines.items():
                    if is_relevant and name in pdf_medicines_info and name not in [c["제품명"] for c in candidates]:
                        medicine_info = pdf_medicines_info[name]
                        
                        # 신체 부위별 우려사항이 있는 경우 해당 부위 부작용이 적은 약품 우선
                        if primary_concern:
                            concern_risk_keywords = body_part_risk_map.get(primary_concern, [])
                            concern_risk_count = sum(1 for risk in concern_risk_keywords if risk in medicine_info["부작용"].lower())
                            
                            if concern_risk_count <= 1:
                                candidates.append({
                                    "제품명": name,
                                    "효능": medicine_info["효능"],
                                    "부작용": medicine_info["부작용"],
                                    "사용법": medicine_info["사용법"],
                                    f"{primary_concern}_부담도": concern_risk_count,
                                    "데이터_소스": "PDF"
                                })
                                print(f"✅ PDF 후보 추가: {name}")
                        else:
                            if not any(risk in medicine_info["부작용"].lower() for risk in all_risk_keywords):
                                candidates.append({
                                    "제품명": name,
                                    "효능": medicine_info["효능"],
                                    "부작용": medicine_info["부작용"],
                                    "사용법": medicine_info["사용법"],
                                    "데이터_소스": "PDF"
                                })
                                print(f"✅ PDF 후보 추가: {name}")
                        
                        # 충분한 약품을 찾았으면 중단
                        if len(candidates) >= 5:
                            break
    else:
        print(f"✅ Excel DB에서 충분한 약품을 찾음: {len(candidates)}개, PDF 검색 건너뜀")

    print(f"🔍 디버깅: 최종 candidates 개수 = {len(candidates)}")
    print(f"🔍 데이터 소스별 통계:")
    excel_count = sum(1 for c in candidates if c["데이터_소스"] == "Excel")
    pdf_count = sum(1 for c in candidates if c["데이터_소스"] == "PDF")
    print(f"  - Excel: {excel_count}개")
    print(f"  - PDF: {pdf_count}개")
    
    if not candidates:
        state["recommendation_answer"] = f"죄송합니다. '{condition}' 병력에 적합한 {category} 관련 약품을 찾지 못했습니다."
        return state

    # 신체 부위별 우려사항이 있는 경우 해당 부위 부담도로 정렬
    if primary_concern:
        candidates.sort(key=lambda x: x.get(f"{primary_concern}_부담도", 0))

    try:
        response = generate_recommendation_response(condition, category, candidates, primary_concern)
        state["recommendation_answer"] = response
    except Exception as e:
        state["recommendation_answer"] = "추천 생성 실패: LLM 호출 중 문제가 발생했습니다."

    return state
