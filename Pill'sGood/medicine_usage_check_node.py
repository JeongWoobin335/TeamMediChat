# medicine_usage_check_node.py

from qa_state import QAState
from retrievers import llm, pdf_structured_docs, excel_docs
from langchain_core.documents import Document
from typing import List, Optional
import json
import re
from cache_manager import cache_manager
from difflib import get_close_matches

def normalize_medicine_name(name: str) -> str:
    """
    약품명 정규화 (유사도 매칭을 위해)
    """
    if not name:
        return ""
    
    # 소문자 변환
    normalized = name.lower()
    
    # 특수문자, 공백, 숫자 제거 (한글과 영문만 유지)
    normalized = re.sub(r'[^\w가-힣]', '', normalized)
    
    # 연속된 공백 제거
    normalized = re.sub(r'\s+', '', normalized)
    
    return normalized.strip()

def calculate_similarity(str1: str, str2: str) -> float:
    """
    두 문자열의 유사도 계산 (0.0 ~ 1.0)
    """
    if not str1 or not str2:
        return 0.0
    
    if str1 == str2:
        return 1.0
    
    # 길이가 너무 다르면 유사도 낮음
    len_diff = abs(len(str1) - len(str2))
    if len_diff > max(len(str1), len(str2)) * 0.5:
        return 0.0
    
    # Levenshtein distance 기반 유사도 계산
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    distance = levenshtein_distance(str1, str2)
    max_len = max(len(str1), len(str2))
    similarity = 1.0 - (distance / max_len)
    
    return similarity

def find_similar_medicine_name(ocr_result: str, medicine_list: List[str], cutoff: float = 0.8) -> Optional[str]:
    """
    OCR 결과와 유사한 약품명 찾기
    """
    if not ocr_result or not medicine_list:
        return None
    
    # OCR 결과 정규화
    normalized_ocr = normalize_medicine_name(ocr_result)
    print(f"🔍 정규화된 OCR 결과: '{normalized_ocr}'")
    
    # 약품명 리스트도 정규화
    normalized_medicines = [(normalize_medicine_name(med), med) for med in medicine_list]
    
    
    # 직접 유사도 계산
    best_match = None
    best_similarity = 0.0
    
    for norm, orig in normalized_medicines:
        similarity = calculate_similarity(normalized_ocr, norm)
        
        # 유사도가 높은 경우만 로그 출력 (성능 개선)
        if similarity > 0.3:
            print(f"🔍 '{orig}' 유사도: {similarity:.3f}")
        
        if similarity > best_similarity and similarity >= cutoff:
            best_similarity = similarity
            best_match = orig
    
    if best_match:
        print(f"✅ 유사도 매칭 성공: '{ocr_result}' → '{best_match}' (유사도: {best_similarity:.3f})")
        return best_match
    
    # cutoff를 낮춰서 다시 시도
    if cutoff > 0.5:
        print(f"🔍 cutoff를 낮춰서 재시도 (0.5)")
        for norm, orig in normalized_medicines:
            similarity = calculate_similarity(normalized_ocr, norm)
            if similarity > best_similarity and similarity >= 0.5:
                best_similarity = similarity
                best_match = orig
        
        if best_match:
            print(f"✅ 낮은 cutoff 매칭 성공: '{ocr_result}' → '{best_match}' (유사도: {best_similarity:.3f})")
            return best_match
    
    print(f"❌ 유사도 매칭 실패: '{ocr_result}' (최고 유사도: {best_similarity:.3f})")
    return None

def find_medicine_info(medicine_name: str, all_docs: List[Document]) -> dict:
    """약품명으로 약품 정보를 찾아서 반환"""
    medicine_info = {
        "제품명": medicine_name,
        "효능": "정보 없음",
        "부작용": "정보 없음", 
        "사용법": "정보 없음",
        "주의사항": "정보 없음"
    }
    
    # 정확한 제품명 매칭 시도
    exact_matches = [doc for doc in all_docs if doc.metadata.get("제품명") == medicine_name]
    
    if not exact_matches:
        # 부분 매칭 시도 (약품명이 포함된 경우)
        partial_matches = []
        for doc in all_docs:
            doc_name = doc.metadata.get("제품명", "")
            if medicine_name in doc_name or doc_name in medicine_name:
                partial_matches.append(doc)
        
        if partial_matches:
            exact_matches = partial_matches
        else:
            # 유사도 기반 매칭 시도
            print(f"🔍 유사도 기반 약품명 매칭 시도: '{medicine_name}'")
            
            # 모든 약품명 리스트 생성
            medicine_list = [doc.metadata.get("제품명", "") for doc in all_docs if doc.metadata.get("제품명")]
            medicine_list = list(set(medicine_list))  # 중복 제거
            
            # 유사도 매칭 시도
            similar_medicine = find_similar_medicine_name(medicine_name, medicine_list, cutoff=0.8)
            if similar_medicine:
                print(f"✅ 유사도 매칭 성공: '{medicine_name}' → '{similar_medicine}'")
                # 유사한 약품명으로 다시 검색
                exact_matches = [doc for doc in all_docs if doc.metadata.get("제품명") == similar_medicine]
                # medicine_info의 제품명을 올바른 약품명으로 업데이트
                medicine_info["제품명"] = similar_medicine
            else:
                print(f"🔍 유사도 매칭 실패: '{medicine_name}'")
                # 기존 정규화 방식도 시도
                normalized_medicine = re.sub(r"[^\w가-힣]", "", medicine_name.lower())
                for doc in all_docs:
                    doc_name = doc.metadata.get("제품명", "")
                    normalized_doc_name = re.sub(r"[^\w가-힣]", "", doc_name.lower())
                    if normalized_medicine in normalized_doc_name or normalized_doc_name in normalized_medicine:
                        exact_matches.append(doc)
                        break
    
    if not exact_matches:
        return medicine_info
    
    # 약품 정보 수집
    for doc in exact_matches:
        content = doc.page_content
        doc_type = doc.metadata.get("type", "")
        
        # 효능과 부작용은 main 타입에서 추출
        if doc_type == "main" or doc_type == "":
            efficacy = extract_field_from_doc(content, "효능")
            side_effects = extract_field_from_doc(content, "부작용")
            
            if efficacy != "정보 없음":
                medicine_info["효능"] = efficacy
            if side_effects != "정보 없음":
                medicine_info["부작용"] = side_effects
        
        # 사용법은 usage 타입에서 추출
        if doc_type == "usage":
            usage = extract_field_from_doc(content, "사용법")
            if usage != "정보 없음":
                medicine_info["사용법"] = usage
    
    return medicine_info

def extract_field_from_doc(text: str, label: str) -> str:
    """문서에서 특정 필드 추출"""
    pattern = rf"\[{label}\]:\s*((?:.|\n)*?)(?=\n\[|\Z)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else "정보 없음"

def check_medicine_usage_safety(medicine_info: dict, usage_context: str) -> dict:
    """약품 사용 안전성 판단"""
    
    # 캐시에서 먼저 확인
    cache_key = f"{medicine_info['제품명']}_{usage_context}"
    cache_file = cache_manager.matching_cache_dir / f"{cache_key}.pkl"
    
    if cache_file.exists():
        try:
            import pickle
            with open(cache_file, 'rb') as f:
                cached_result = pickle.load(f)
            print(f"📂 사용 가능성 캐시 히트: {cache_key}")
            return cached_result
        except Exception as e:
            print(f"❌ 사용 가능성 캐시 로드 실패: {e}")
    
    # 캐시에 없으면 None 반환
    cached_result = None
    
    # LLM을 사용한 안전성 판단
    prompt = f"""
당신은 의약품 안전성 전문가입니다. 다음 약품을 주어진 상황에서 사용해도 안전한지 판단해주세요.

**약품 정보:**
- 제품명: {medicine_info['제품명']}
- 효능: {medicine_info['효능']}
- 부작용: {medicine_info['부작용']}
- 사용법: {medicine_info['사용법']}

**사용 상황:**
{usage_context}

**판단 기준:**
1. 약품의 효능이 해당 상황에 적합한가?
   - 효능에 명시된 증상/상황과 사용자가 언급한 상황이 의미적으로 일치하는가?
   - 예: "육체피로" ↔ "피곤할 때", "두통" ↔ "머리가 아플 때", "감기" ↔ "감기에 걸렸을 때"
2. 부작용이 해당 상황에서 위험하지 않은가?
3. 사용법이 올바른가?
4. 특별한 주의사항이 있는가?

**중요한 지침:**
- 효능 정보를 꼼꼼히 분석하여 사용 상황과의 연관성을 찾으세요
- 동의어나 유사한 표현을 고려하세요 (예: 피로 ↔ 피곤함, 두통 ↔ 머리 아픔, 습진 ↔ 피부염 ↔ 아토피)
- 의학적으로 관련된 증상들을 의미적으로 이해하세요 (예: 습진, 아토피, 피부염, 발진, 가려움 등은 모두 피부 관련)
- 약품의 주요 효능이 사용 상황과 일치하면 사용 가능으로 판단하세요
- 부작용이 심각하지 않고 사용법이 적절하면 사용 가능으로 판단하세요
- 명확한 근거를 제시하세요

다음 JSON 형식으로 응답해주세요:
{{
    "safe_to_use": true/false,
    "reason": "사용 가능/불가능한 구체적인 이유 (효능과 사용 상황의 연관성 포함)",
    "precautions": "주의사항 (있는 경우)",
    "alternative_suggestion": "대안 제안 (필요한 경우)"
}}
"""
    
    try:
        response = llm.invoke(prompt).content.strip()
        print(f"🔍 LLM 응답: {response[:200]}...")
        
        # JSON 응답 파싱 (```json 제거 처리)
        try:
            # ```json과 ``` 제거
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                if json_end != -1:
                    response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                if json_end != -1:
                    response = response[json_start:json_end].strip()
            
            result = json.loads(response)
            print(f"✅ JSON 파싱 성공: {result}")
        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 실패: {e}")
            print(f"🔍 원본 응답: {response}")
            # JSON 파싱 실패 시 기본 응답
            result = {
                "safe_to_use": False,
                "reason": "약품 정보를 분석할 수 없습니다.",
                "precautions": "의사나 약사와 상담하세요.",
                "alternative_suggestion": ""
            }
        
        # 캐시에 저장
        try:
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            print(f"💾 사용 가능성 캐시 저장됨: {cache_key}")
        except Exception as e:
            print(f"❌ 사용 가능성 캐시 저장 실패: {e}")
        
        return result
        
    except Exception as e:
        print(f"❌ 약품 사용 안전성 판단 중 오류 발생: {e}")
        return {
            "safe_to_use": False,
            "reason": "안전성 판단 중 오류가 발생했습니다.",
            "precautions": "의사나 약사와 상담하세요.",
            "alternative_suggestion": ""
        }

def generate_usage_check_response(medicine_name: str, usage_context: str, medicine_info: dict, safety_result: dict) -> str:
    """사용 가능성 판단 결과를 사용자 친화적인 응답으로 변환"""
    
    # usage_context에서 질문 형태의 문장을 정리하여 자연스러운 표현으로 변환
    clean_context = usage_context
    if "?" in usage_context:
        import re
        # 질문 형태에서 핵심 증상/상황만 추출
        # "이 연고 습진에 발라도 되나?" → "습진에"
        # "박테로신이라는 연고 습진에 발라도 되나?" → "습진에"
        # "두통에 먹어도 되나?" → "두통에"
        # "상처에 발라도 되나?" → "상처에"
        
        # 더 정확한 패턴 매칭
        patterns = [
            r'([가-힣]+에)\s+[가-힣\s]*발라도\s+되나\?',  # "습진에 발라도 되나?"
            r'([가-힣]+에)\s+[가-힣\s]*먹어도\s+되나\?',   # "두통에 먹어도 되나?"
            r'([가-힣]+에)\s+[가-힣\s]*써도\s+되나\?',     # "상처에 써도 되나?"
            r'([가-힣]+에)\s+[가-힣\s]*사용해도\s+되나\?', # "상처에 사용해도 되나?"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, usage_context)
            if match:
                clean_context = match.group(1)
                break
        
        # 패턴 매칭이 실패한 경우 기본 처리
        if clean_context == usage_context:
            clean_context = usage_context.replace("?", "").strip()
    
    if safety_result["safe_to_use"]:
        response = f"✅ **{medicine_name}**을(를) {clean_context} 사용하는 것은 **가능**합니다.\n\n"
        response += f"**이유:** {safety_result['reason']}\n\n"
        
        if safety_result.get("precautions"):
            response += f"**주의사항:** {safety_result['precautions']}\n\n"
    else:
        response = f"❌ **{medicine_name}**을(를) {clean_context} 사용하는 것은 **권장하지 않습니다**.\n\n"
        response += f"**이유:** {safety_result['reason']}\n\n"
        
        if safety_result.get("precautions"):
            response += f"**주의사항:** {safety_result['precautions']}\n\n"
        
        if safety_result.get("alternative_suggestion"):
            response += f"**대안 제안:** {safety_result['alternative_suggestion']}\n\n"
    
    # 약품 정보 요약 추가
    response += "**약품 정보 요약:**\n"
    response += f"- 효능: {medicine_info['효능']}\n"
    response += f"- 부작용: {medicine_info['부작용']}\n"
    response += f"- 사용법: {medicine_info['사용법']}\n\n"
    
    response += "⚠️ **중요:** 이 정보는 참고용이며, 정확한 진단과 처방을 위해서는 의사나 약사와 상담하시기 바랍니다."
    
    return response

def medicine_usage_check_node(state: QAState) -> QAState:
    """약품 사용 가능성 판단 노드"""
    
    medicine_name = state.get("medicine_name", "")
    usage_context = state.get("usage_context", "")
    
    if not medicine_name or not usage_context:
        state["usage_check_answer"] = "죄송합니다. 약품명이나 사용 상황 정보가 부족하여 판단할 수 없습니다."
        return state
    
    print(f"🔍 약품 사용 가능성 판단 시작: {medicine_name} → {usage_context}")
    
    # Excel DB에서 먼저 검색
    print("📊 Excel DB에서 약품 정보 검색 중...")
    medicine_info = find_medicine_info(medicine_name, excel_docs)
    
    # Excel에서 정보를 찾지 못한 경우 PDF에서 검색
    if medicine_info["효능"] == "정보 없음":
        print("📄 PDF DB에서 약품 정보 검색 중...")
        pdf_medicine_info = find_medicine_info(medicine_name, pdf_structured_docs)
        
        # PDF에서 찾은 정보로 업데이트
        if pdf_medicine_info["효능"] != "정보 없음":
            medicine_info = pdf_medicine_info
            print(f"✅ PDF에서 {medicine_name} 정보 발견")
        else:
            print(f"❌ {medicine_name} 정보를 찾을 수 없음")
    
    # 약품 정보를 찾지 못한 경우
    if medicine_info["효능"] == "정보 없음":
        state["usage_check_answer"] = f"죄송합니다. '{medicine_name}'에 대한 정보를 찾을 수 없습니다. 정확한 약품명을 확인하시거나 의사/약사와 상담하시기 바랍니다."
        return state
    
    print(f"✅ 약품 정보 발견: {medicine_info['제품명']}")
    
    # 사용 안전성 판단
    print("🔍 사용 안전성 판단 중...")
    safety_result = check_medicine_usage_safety(medicine_info, usage_context)
    
    # 최종 응답 생성
    print("📝 최종 응답 생성 중...")
    final_response = generate_usage_check_response(medicine_name, usage_context, medicine_info, safety_result)
    
    state["usage_check_answer"] = final_response
    
    print("✅ 약품 사용 가능성 판단 완료")
    return state
