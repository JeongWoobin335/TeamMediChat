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

def find_medicine_info(medicine_name: str, all_docs: List[Document], is_ocr_result: bool = False) -> dict:
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
            # OCR 결과인 경우에만 유사도 기반 매칭 시도
            if is_ocr_result:
                print(f"🔍 OCR 결과 유사도 기반 약품명 매칭 시도: '{medicine_name}'")
                
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
            else:
                print(f"🔍 단일 텍스트 질문: 유사도 매칭 건너뜀")
    
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
    
    # LLM을 사용한 안전성 판단 - 최적화된 프롬프트
    prompt = f"""당신은 의약품 안전성 평가 전문가입니다. 단계별로 분석하여 근거 있는 판단을 내리세요.

## 📋 약품 정보
- 제품명: {medicine_info['제품명']}
- 효능: {medicine_info['효능']}
- 부작용: {medicine_info['부작용']}
- 사용법: {medicine_info['사용법']}

## 🎯 사용 상황
{usage_context}

## 🔍 3단계 평가 프로세스

### STEP 1: 효능-증상 매칭 분석 (가장 중요)
아래 의학적 증상 매핑을 참고하여 약품 효능과 사용 상황의 연관성을 평가하세요.

**의학적 증상 매핑:**
- 피부 질환: 습진 ↔ 아토피 ↔ 피부염 ↔ 발진 ↔ 가려움 ↔ 두드러기
- 상처/외상: 상처 ↔ 찰과상 ↔ 긁힘 ↔ 베인 상처 ↔ 외상 ↔ 화상
- 통증: 두통 ↔ 편두통 ↔ 머리 아픔 / 근육통 ↔ 몸살 / 치통 ↔ 잇몸 통증
- 피로: 피로 ↔ 피곤함 ↔ 무기력 ↔ 기운 없음 ↔ 체력 저하 ↔ 육체 피로
- 소화: 소화불량 ↔ 체함 ↔ 속 불편 ↔ 위장 장애 ↔ 더부룩함
- 감염: 세균 감염 ↔ 화농 ↔ 염증 ↔ 고름
- 감기: 감기 ↔ 코감기 ↔ 목감기 ↔ 기침 ↔ 콧물 ↔ 인후통

**분석 질문:**
1. 약품 효능에 명시된 증상이 사용 상황과 직접 일치하는가?
2. 위 매핑에서 의미적으로 유사한 증상인가?
3. 매칭 강도: 완전일치(100%) / 강한연관(80%) / 중간연관(50%) / 약한연관(30%) / 무관(0%)

**STEP 1 결과:**
- 매칭 강도: ___%
- 근거: [효능의 어떤 부분이 사용 상황과 연관되는지 구체적으로 설명]

### STEP 2: 위험도 평가
**부작용 심각도 점검:**
- 심각한 부작용 있음? (쇼크, 중증 알레르기 등) → 위험
- 일반적 부작용만 있음? (졸음, 가벼운 소화불량 등) → 보통
- 부작용 미미 또는 없음? → 안전

**사용 상황 적합성:**
- 해당 상황에서 부작용이 치명적인가?
- 사용법이 상황에 맞는가? (경구/외용 등)

**STEP 2 결과:**
- 위험 수준: 높음 / 보통 / 낮음
- 근거: [구체적 설명]

### STEP 3: 최종 판단
**종합 점수 계산:**
- 매칭 강도 ≥ 50% + 위험 수준 낮음/보통 → 사용 가능
- 매칭 강도 < 50% 또는 위험 수준 높음 → 사용 불가

**신뢰도 평가:**
- 높음: 명확한 효능 일치 + 안전성 확인됨
- 중간: 유사 증상 + 큰 위험 없음
- 낮음: 효능 불명확하거나 위험 요소 있음

## 💡 판단 예시

### 예시 1: 베타딘 연고 + 상처
- STEP 1: 효능 "상처 소독, 세균 감염 예방" vs 사용 "상처" → 100% 일치
- STEP 2: 부작용 "피부 자극" → 경미, 위험 낮음
- STEP 3: 사용 가능 (신뢰도: 높음)

### 예시 2: 감기약 + 두통
- STEP 1: 효능 "감기 증상 완화(두통, 발열)" vs 사용 "두통" → 80% 강한 연관
- STEP 2: 부작용 "졸음" → 경미, 위험 낮음
- STEP 3: 사용 가능 (신뢰도: 높음)

### 예시 3: 피부 연고 + 근육통
- STEP 1: 효능 "습진, 피부염 완화" vs 사용 "근육통" → 0% 무관
- STEP 2: 효능 불일치
- STEP 3: 사용 불가 (신뢰도: 높음)

## 📤 출력 형식 (JSON)
{{
    "safe_to_use": true/false,
    "confidence_score": 0.0~1.0,
    "matching_strength": 0~100,
    "reason": "STEP 1-3 분석 결과를 바탕으로 한 구체적 근거 (2-3문장)",
    "precautions": "주의사항 (필요시)",
    "alternative_suggestion": "대안 제안 (사용 불가 시)"
}}

**중요**: 추측하지 말고 주어진 약품 정보만으로 판단하세요. 불확실하면 confidence_score를 낮추세요.
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
            
            # 새로운 필드가 없으면 기본값 추가 (하위 호환성)
            if "confidence_score" not in result:
                result["confidence_score"] = 0.7  # 기본 중간 신뢰도
            if "matching_strength" not in result:
                result["matching_strength"] = 50  # 기본 중간 매칭
            
            print(f"✅ JSON 파싱 성공: safe_to_use={result.get('safe_to_use')}, confidence={result.get('confidence_score')}, matching={result.get('matching_strength')}%")
        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 실패: {e}")
            print(f"🔍 원본 응답: {response}")
            # JSON 파싱 실패 시 기본 응답
            result = {
                "safe_to_use": False,
                "confidence_score": 0.3,
                "matching_strength": 0,
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
            "confidence_score": 0.0,
            "matching_strength": 0,
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
    
    # 신뢰도 및 매칭 강도 정보
    confidence = safety_result.get("confidence_score", 0.7)
    matching = safety_result.get("matching_strength", 50)
    
    # 신뢰도 레벨 표시
    if confidence >= 0.8:
        confidence_text = "높음 🟢"
    elif confidence >= 0.5:
        confidence_text = "중간 🟡"
    else:
        confidence_text = "낮음 🔴"
    
    if safety_result["safe_to_use"]:
        response = f"✅ **{medicine_name}**을(를) {clean_context} 사용하는 것은 **가능**합니다.\n\n"
        response += f"**판단 근거:** {safety_result['reason']}\n\n"
        response += f"**신뢰도:** {confidence_text} (효능 매칭: {matching}%)\n\n"
        
        if safety_result.get("precautions"):
            response += f"**⚠️ 주의사항:** {safety_result['precautions']}\n\n"
    else:
        response = f"❌ **{medicine_name}**을(를) {clean_context} 사용하는 것은 **권장하지 않습니다**.\n\n"
        response += f"**판단 근거:** {safety_result['reason']}\n\n"
        response += f"**신뢰도:** {confidence_text} (효능 매칭: {matching}%)\n\n"
        
        if safety_result.get("precautions"):
            response += f"**⚠️ 주의사항:** {safety_result['precautions']}\n\n"
        
        if safety_result.get("alternative_suggestion"):
            response += f"**💡 대안 제안:** {safety_result['alternative_suggestion']}\n\n"
    
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
    
    # Excel DB에서만 검색 (PDF DB 제거)
    print("📊 Excel DB에서 약품 정보 검색 중...")
    # 이미지가 포함된 경우(OCR 결과)인지 확인
    is_ocr_result = state.get("has_image", False) or state.get("extracted_text") is not None
    medicine_info = find_medicine_info(medicine_name, excel_docs, is_ocr_result)
    
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
