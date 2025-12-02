# question_refinement_node.py - GPT 기반 질문 보정 노드

from qa_state import QAState
from answer_utils import generate_response_llm_from_prompt
from retrievers import excel_docs, known_ingredients, product_names  # 🚀 성능 최적화: 전역 변수 사용
import re
from typing import Optional, List

def normalize_medicine_name(name: str) -> str:
    """약품명 정규화 (유사도 매칭을 위해)"""
    if not name:
        return ""
    normalized = name.lower()
    normalized = re.sub(r'[^\w가-힣]', '', normalized)
    normalized = re.sub(r'\s+', '', normalized)
    return normalized.strip()

def calculate_similarity(str1: str, str2: str) -> float:
    """두 문자열의 유사도 계산 (0.0 ~ 1.0)"""
    if not str1 or not str2:
        return 0.0
    
    if str1 == str2:
        return 1.0
    
    len_diff = abs(len(str1) - len(str2))
    if len_diff > max(len(str1), len(str2)) * 0.5:
        return 0.0
    
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
    similarity = 1.0 - (distance / max_len) if max_len > 0 else 0.0
    return similarity

def find_similar_ingredient_name(query: str, ingredient_list: set, cutoff: float = 0.6) -> Optional[str]:
    """질문에서 성분명 후보를 추출하고 유사도 기반으로 가장 유사한 성분명 찾기"""
    if not query or not ingredient_list:
        return None
    
    # 질문에서 성분명 후보 추출
    pattern1 = re.findall(r'([가-힣]{2,10})(?:은|는|이|가|을|를)', query)
    pattern2 = re.findall(r'([가-힣]{2,10})(?:의)', query)
    pattern3 = re.findall(r'[가-힣]{2,10}', query)
    
    priority_candidates = list(set(pattern1 + pattern2))
    other_candidates = list(set(pattern3))
    
    if priority_candidates:
        all_candidates = priority_candidates
    else:
        all_candidates = other_candidates
    
    if not all_candidates:
        return None
    
    valid_candidates = []
    for candidate in all_candidates:
        clean_candidate = re.sub(r'[은는이가을를에의와과도부터까지에서부터]$', '', candidate).strip()
        if len(clean_candidate) < 2:
            continue
        
        normalized_candidate = normalize_medicine_name(clean_candidate)
        
        max_similarity = 0.0
        matched_ingredient = None
        for ingredient in ingredient_list:
            normalized_ingredient = normalize_medicine_name(ingredient)
            similarity = calculate_similarity(normalized_candidate, normalized_ingredient)
            if similarity > max_similarity:
                max_similarity = similarity
                matched_ingredient = ingredient
        
        if max_similarity >= 0.4:
            valid_candidates.append((clean_candidate, max_similarity, matched_ingredient))
    
    if not valid_candidates:
        return None
    
    candidate, max_sim, best_match = max(valid_candidates, key=lambda x: x[1])
    
    print(f"🔍 성분명 후보 추출: '{candidate}' (정규화: '{normalize_medicine_name(candidate)}')")
    
    if max_sim >= cutoff:
        print(f"✅ 성분명 유사도 매칭 성공: '{candidate}' → '{best_match}' (유사도: {max_sim:.3f})")
        return best_match
    elif max_sim >= 0.4:
        print(f"✅ 낮은 cutoff 성분명 매칭 성공: '{candidate}' → '{best_match}' (유사도: {max_sim:.3f})")
        return best_match
    
    print(f"❌ 성분명 유사도 매칭 실패: '{candidate}' (최고 유사도: {max_sim:.3f})")
    return None

def find_similar_medicine_name(query: str, medicine_list: List[str], cutoff: float = 0.6) -> Optional[str]:
    """질문에서 약품명 후보를 추출하고 유사도 기반으로 가장 유사한 약품명 찾기"""
    if not query or not medicine_list:
        return None
    
    # 질문에서 약품명 후보 추출 (더 정확한 패턴)
    # 패턴 1: "약품명은/는/이/가/을/를" 형태 (우선순위 높음)
    pattern1 = re.findall(r'([가-힣]{2,10})(?:은|는|이|가|을|를)', query)
    # 패턴 2: "약품명정", "약품명연고" 등 형태 포함 (우선순위 높음)
    pattern2 = re.findall(r'([가-힣]{2,8})(?:정|연고|크림|젤|캡슐|시럽|액|주사)', query)
    # 패턴 3: "약품명의" 형태
    pattern3 = re.findall(r'([가-힣]{2,10})(?:의)', query)
    # 패턴 4: 일반 한글 단어
    pattern4 = re.findall(r'[가-힣]{2,10}', query)
    
    # 우선순위가 높은 패턴부터 후보 수집
    priority_candidates = list(set(pattern1 + pattern2 + pattern3))
    other_candidates = list(set(pattern4))
    
    # 우선순위 후보가 있으면 그것부터 사용, 없으면 일반 후보 사용
    if priority_candidates:
        all_candidates = priority_candidates
    else:
        all_candidates = other_candidates
    
    if not all_candidates:
        return None
    
    # 각 후보에 대해 약품명 리스트와의 유사도 계산하여 필터링
    # 유사도가 일정 수준 이상인 것만 약품명 후보로 인정 (하드코딩 필터 대신)
    valid_candidates = []
    for candidate in all_candidates:
        # 조사 제거
        clean_candidate = re.sub(r'[은는이가을를에의와과도부터까지에서부터]$', '', candidate).strip()
        if len(clean_candidate) < 2:
            continue
        
        normalized_candidate = normalize_medicine_name(clean_candidate)
        
        # 약품명 리스트와의 최고 유사도 및 매칭된 약품명 계산
        max_similarity = 0.0
        matched_medicine = None
        for medicine in medicine_list:
            normalized_medicine = normalize_medicine_name(medicine)
            similarity = calculate_similarity(normalized_candidate, normalized_medicine)
            if similarity > max_similarity:
                max_similarity = similarity
                matched_medicine = medicine
        
        # 유사도가 일정 수준 이상이면 약품명 후보로 인정
        if max_similarity >= 0.4:  # 낮은 기준으로 필터링 (하드코딩 필터 대신)
            valid_candidates.append((clean_candidate, max_similarity, matched_medicine))
    
    if not valid_candidates:
        return None
    
    # 가장 유사도가 높은 후보 선택
    candidate, max_sim, best_match = max(valid_candidates, key=lambda x: x[1])
    
    print(f"🔍 약품명 후보 추출: '{candidate}' (정규화: '{normalize_medicine_name(candidate)}')")
    
    # cutoff 기준 확인
    if max_sim >= cutoff:
        print(f"✅ 유사도 매칭 성공: '{candidate}' → '{best_match}' (유사도: {max_sim:.3f})")
        return best_match
    elif max_sim >= 0.4:
        print(f"✅ 낮은 cutoff 매칭 성공: '{candidate}' → '{best_match}' (유사도: {max_sim:.3f})")
        return best_match
    
    print(f"❌ 유사도 매칭 실패: '{candidate}' (최고 유사도: {max_sim:.3f})")
    return None

def question_refinement_node(state: QAState) -> QAState:
    """
    GPT를 사용하여 사용자 질문을 보정합니다.
    - 오타 보정 (약품명, 증상명 등)
    - 불완전한 질문 완성
    - 의도 명확화
    - 맥락 이해 (이전 대화 참조)
    """
    print("🔍 질문 보정 노드 시작")
    
    raw_query = state.get("query", "")
    conversation_context = state.get("conversation_context", "")
    
    # 원본 질문 보존 (이미 original_query가 있으면 그것을 사용, 없으면 현재 query를 원본으로 저장)
    if "original_query" not in state or not state.get("original_query"):
        original_query = raw_query
    else:
        original_query = state.get("original_query", raw_query)
    
    if not raw_query or not raw_query.strip():
        print("⚠️ 질문이 비어있어 보정 건너뜀")
        return state
    
    # 🚀 성능 최적화: 전역 변수 product_names 사용 (매번 생성하지 않음)
    medicine_list = []
    try:
        if product_names:
            medicine_list = product_names
            print(f"📊 약품명 리스트 사용 (전역 변수): {len(medicine_list)}개")
        else:
            # 폴백: product_names가 없으면 직접 생성 (최초 1회만)
            for doc in excel_docs:
                product_name = doc.metadata.get("제품명", "")
                if product_name and product_name not in medicine_list:
                    medicine_list.append(product_name)
            print(f"📊 약품명 리스트 생성: {len(medicine_list)}개")
    except Exception as e:
        print(f"⚠️ 약품명 리스트 로드 실패: {e}")
        medicine_list = []
    
    # 연속 질문인지 확인 (이전 대화 맥락이 있는지)
    is_follow_up = bool(conversation_context and len(conversation_context) > 50)
    
    # 이전 대화에서 언급된 성분명 추출 (연속 질문인 경우)
    mentioned_ingredients = set()
    if is_follow_up:
        # 이전 대화에서 성분명 패턴 찾기 (더 정확하게)
        # 패턴 1: "성분명은/는/이/가/을/를" 형태
        ingredient_patterns1 = re.findall(r'([가-힣]{2,15})(?:은|는|이|가|을|를)', conversation_context)
        # 패턴 2: "성분명의" 형태
        ingredient_patterns2 = re.findall(r'([가-힣]{2,15})(?:의)', conversation_context)
        # 패턴 3: "주성분: 성분명" 형태
        ingredient_patterns3 = re.findall(r'주성분[:\s]*([가-힣]{2,15})', conversation_context)
        # 패턴 4: "성분명," 형태 (쉼표로 구분된 성분 목록)
        ingredient_patterns4 = re.findall(r'([가-힣]{2,15}),', conversation_context)
        
        all_patterns = ingredient_patterns1 + ingredient_patterns2 + ingredient_patterns3 + ingredient_patterns4
        
        for pattern in all_patterns:
            # 정규화하여 성분명 리스트와 비교
            normalized_pattern = normalize_medicine_name(pattern)
            for ingredient in known_ingredients:
                normalized_ingredient = normalize_medicine_name(ingredient)
                # 정확히 일치하거나 포함 관계인 경우
                if normalized_pattern == normalized_ingredient or normalized_pattern in normalized_ingredient or normalized_ingredient in normalized_pattern:
                    mentioned_ingredients.add(ingredient)
                    break
        
        if mentioned_ingredients:
            print(f"🔍 이전 대화에서 언급된 성분명: {list(mentioned_ingredients)[:5]}")
    
    # ⚠️ 중요: LLM 보정 전에 원본 질문에서 약품명이 정확히 존재하는지 먼저 확인
    exact_medicine_match = None
    if medicine_list:
        # 원본 질문에서 약품명 후보 추출
        raw_candidates = re.findall(r'([가-힣]{2,10})(?:은|는|이|가|을|를|의|정|연고)', raw_query)
        raw_candidates += re.findall(r'([가-힣]{2,8})(?:정|연고|크림|젤|캡슐|시럽|액|주사)', raw_query)
        
        # 각 후보에 대해 약품명 리스트에서 정확한 매칭 확인
        for candidate in raw_candidates:
            clean_candidate = re.sub(r'[은는이가을를에의와과도부터까지에서부터]$', '', candidate).strip()
            if len(clean_candidate) < 2:
                continue
            
            normalized_candidate = normalize_medicine_name(clean_candidate)
            
            # 약품명 리스트에서 정확한 매칭 확인 (정규화 후 비교)
            for medicine in medicine_list:
                normalized_medicine = normalize_medicine_name(medicine)
                # 정확히 일치하거나 매우 높은 유사도(0.95 이상)인 경우
                if normalized_candidate == normalized_medicine:
                    exact_medicine_match = medicine
                    print(f"✅ 원본 질문에서 정확한 약품명 발견: '{clean_candidate}' → '{medicine}'")
                    break
                elif calculate_similarity(normalized_candidate, normalized_medicine) >= 0.95:
                    exact_medicine_match = medicine
                    print(f"✅ 원본 질문에서 매우 유사한 약품명 발견: '{clean_candidate}' → '{medicine}'")
                    break
            
            if exact_medicine_match:
                break
    
    # 🚀 성능 최적화: 정확한 약품명이 있고, 질문이 명확하면 LLM 보정 스킵
    if exact_medicine_match and not is_follow_up:
        # 질문이 간단하고 명확한지 확인 (오타나 불완전한 질문이 아닌지)
        escaped_medicine = re.escape(exact_medicine_match)
        simple_query_patterns = [
            rf'.*?{escaped_medicine}.*?(?:먹어도|사용해도|써도|복용해도).*?',
            rf'.*?{escaped_medicine}.*?(?:효능|부작용|사용법|주의사항).*?',
            rf'.*?{escaped_medicine}.*?(?:어떤|무엇|알려|설명).*?'
        ]
        
        is_simple_query = any(re.search(pattern, raw_query, re.IGNORECASE) for pattern in simple_query_patterns)
        
        if is_simple_query:
            print(f"⚡ 성능 최적화: 정확한 약품명 발견 + 명확한 질문 → LLM 보정 스킵")
            # 약품명을 state에 저장하고 원본 질문 유지
            state["extracted_medicine_name"] = exact_medicine_match
            state["medicine_name"] = exact_medicine_match
            state["query"] = raw_query
            state["original_query"] = original_query
            state["query_was_refined"] = False
            print(f"✅ 약품명 추출 완료 (LLM 없이): '{exact_medicine_match}'")
            return state
    
    # ChatGPT에게 질문 보정 요청 (약품명 힌트 없이 먼저 보정)
    refinement_prompt = f"""당신은 의약품 상담 시스템의 질문 보정 전문가입니다.
사용자의 질문을 분석하여 오타를 보정하고, 불완전한 질문을 완성하며, 의도를 명확히 해주세요.

**원본 질문:**
{raw_query}

**이전 대화 맥락:**
{conversation_context[:500] if conversation_context else "없음"}

**보정 작업:**

1. **오타 보정:**
   - 약품명 오타 보정 (예: "타이라놀" → "타이레놀")
   - **성분명 오타 보정** (예: "아세트아미노펜" → "아세트아미노펜")
   - 증상명 오타 보정 (예: "두통" → "두통")
   - **중요**: 약품명과 성분명을 구분하여 보정하세요. 성분명은 약품명과 다릅니다.
   - **⚠️ 매우 중요**: 원본 질문에 이미 정확한 약품명이 포함되어 있으면 약품명을 변경하지 마세요.

2. **불완전한 질문 완성:**
   - 중간에 끊긴 질문 완성 (예: "타이레놀 먹으..." → "타이레놀을 먹어도 되나요?")
   - 불명확한 표현 명확화 (예: "이거" → 이전 대화 맥락 참조하여 실제 약품명/성분명으로 변환)

3. **의도 명확화:**
   - 질문 의도 파악 및 명확한 질문으로 변환
   - 예: "타이레놀" → "타이레놀에 대해 알려주세요" 또는 "타이레놀 사용 가능 여부 확인"
   - **성분명 질문**: "푸르설티아민이 뭔데?" → "푸르설티아민이 무엇인가요?" (약품명으로 변환하지 말 것)

4. **맥락 이해:**
   - 이전 대화가 존재할 경우 이전 대화에서 언급된 약품명, **성분명**, 증상 등을 참조
   - "이거", "그거", "그 약" 같은 지시대명사를 실제 약품명으로 변환
   - **연속 질문의 경우**: 이전 대화에서 언급된 성분명에 대한 질문일 수 있으므로 성분명으로 인식

**중요 지침:**
- 원본 질문의 의도와 의미를 최대한 보존
- 약품명과 성분명을 구분하여 보정 (성분명을 약품명으로 변환하지 말 것)
- 연속 질문의 경우 이전 대화에서 언급된 성분명에 대한 질문일 가능성이 높음
- 약품명은 이전 대화 맥락을 참조하여 정확하게 보정
- **⚠️ 원본 질문에 이미 정확한 약품명이 있으면 약품명을 변경하지 마세요**
- 불필요하게 질문을 길게 만들지 말고, 핵심만 명확히
- 이전 대화 맥락이 없으면 추측하지 말고 원본 질문 그대로 유지

**출력 형식:**
보정된 질문만 반환하세요. 설명이나 추가 텍스트 없이 질문만 출력하세요.

**보정된 질문:**
"""
    
    try:
        print(f"🔍 원본 질문: '{raw_query}'")
        
        # ChatGPT 호출 (temperature를 높여서 더 자연스러운 보정)
        refined_query = generate_response_llm_from_prompt(
            prompt=refinement_prompt,
            temperature=0.3,  # 오타 보정은 낮은 temperature로 정확성 확보
            max_tokens=200,
            cache_type="question_refinement"  # 캐싱 타입 지정
        )
        
        # 응답 정제 (불필요한 공백, 줄바꿈 제거)
        refined_query = refined_query.strip()
        
        # 응답이 너무 길거나 이상하면 원본 유지
        if len(refined_query) > len(raw_query) * 3:  # 원본의 3배 이상이면 이상함
            print(f"⚠️ 보정된 질문이 너무 길어서 원본 유지")
            refined_query = raw_query
        elif not refined_query or len(refined_query) < 2:
            print(f"⚠️ 보정된 질문이 비어있어서 원본 유지")
            refined_query = raw_query
        
        print(f"✅ 보정된 질문: '{refined_query}'")
        
        # ⚠️ 중요: 원본에서 정확한 약품명이 발견된 경우, LLM 보정 결과에서 약품명이 변경되었는지 확인
        if exact_medicine_match:
            # 보정된 질문에서 정확한 약품명이 여전히 있는지 확인
            refined_normalized = normalize_medicine_name(exact_medicine_match)
            refined_contains_exact = False
            
            # 보정된 질문에서 약품명 후보 추출
            refined_candidates = re.findall(r'([가-힣]{2,10})(?:은|는|이|가|을|를|의|정|연고)', refined_query)
            refined_candidates += re.findall(r'([가-힣]{2,8})(?:정|연고|크림|젤|캡슐|시럽|액|주사)', refined_query)
            
            for candidate in refined_candidates:
                clean_candidate = re.sub(r'[은는이가을를에의와과도부터까지에서부터]$', '', candidate).strip()
                normalized_candidate = normalize_medicine_name(clean_candidate)
                if normalized_candidate == refined_normalized:
                    refined_contains_exact = True
                    break
            
            # LLM이 정확한 약품명을 잘못 변경한 경우, 원본 약품명으로 복원
            if not refined_contains_exact:
                print(f"⚠️ LLM이 정확한 약품명을 변경함. 원본 약품명으로 복원: '{exact_medicine_match}'")
                # 보정된 질문에서 잘못된 약품명을 찾아서 원본 약품명으로 교체
                for candidate in refined_candidates:
                    clean_candidate = re.sub(r'[은는이가을를에의와과도부터까지에서부터]$', '', candidate).strip()
                    normalized_candidate = normalize_medicine_name(clean_candidate)
                    # 유사도가 낮으면 잘못된 약품명으로 간주
                    similarity = calculate_similarity(normalized_candidate, refined_normalized)
                    if similarity < 0.7:
                        refined_query = refined_query.replace(candidate, exact_medicine_match)
                        print(f"✅ 약품명 복원: '{candidate}' → '{exact_medicine_match}'")
                        break
        
        # 1단계: 성분명 매칭 시도 (연속 질문이거나 성분 관련 질문인 경우 우선)
        extracted_ingredient = None
        if known_ingredients:
            # 연속 질문이고 이전 대화에서 언급된 성분이 있으면 그것을 우선 고려
            if is_follow_up and mentioned_ingredients:
                # 이전 대화에서 언급된 성분명과 매칭 시도
                for mentioned_ing in mentioned_ingredients:
                    if mentioned_ing in refined_query or normalize_medicine_name(mentioned_ing) in normalize_medicine_name(refined_query):
                        extracted_ingredient = mentioned_ing
                        print(f"✅ 이전 대화 맥락에서 성분명 발견: '{extracted_ingredient}'")
                        break
            
            # 이전 대화 맥락에서 찾지 못했으면 일반 성분명 매칭 시도
            if not extracted_ingredient:
                extracted_ingredient = find_similar_ingredient_name(refined_query, known_ingredients, cutoff=0.6)
                if extracted_ingredient:
                    print(f"✅ 보정된 질문에서 성분명 추출: '{extracted_ingredient}'")
        
        # 2단계: 성분명이 매칭되지 않았을 때만 약품명 매칭 시도
        extracted_medicine = None
        if not extracted_ingredient and medicine_list:
            # ⚠️ 중요: 원본에서 정확한 약품명이 발견된 경우 우선 사용
            if exact_medicine_match:
                extracted_medicine = exact_medicine_match
                print(f"✅ 원본 질문의 정확한 약품명 사용: '{extracted_medicine}'")
            else:
                # 원본에서 정확한 매칭이 없을 때만 보정된 질문에서 약품명 추출
                extracted_medicine = find_similar_medicine_name(refined_query, medicine_list, cutoff=0.6)
                if extracted_medicine:
                    print(f"✅ 보정된 질문에서 약품명 추출: '{extracted_medicine}'")
            
            if extracted_medicine:
                # 약품명을 state에 저장 (다른 노드에서 사용 가능)
                state["extracted_medicine_name"] = extracted_medicine
                # ⚠️ 중요: state["medicine_name"]도 업데이트하여 이후 노드에서 보정된 약품명 사용
                if state.get("medicine_name"):
                    old_medicine_name = state.get("medicine_name")
                    state["medicine_name"] = extracted_medicine
                    print(f"✅ state['medicine_name'] 업데이트: '{old_medicine_name}' → '{extracted_medicine}'")
                
                # 보정된 질문에서 오타가 있는 약품명을 정확한 약품명으로 교체
                # 단, 원본에서 정확한 약품명이 발견된 경우에만 교체 (LLM이 잘못 변경한 경우 복원)
                if exact_medicine_match:
                    # 원본 약품명이 보정된 질문에 없으면 교체
                    refined_normalized = normalize_medicine_name(exact_medicine_match)
                    refined_candidates = re.findall(r'([가-힣]{2,10})(?:은|는|이|가|을|를|의|정|연고)', refined_query)
                    refined_candidates += re.findall(r'([가-힣]{2,8})(?:정|연고|크림|젤|캡슐|시럽|액|주사)', refined_query)
                    
                    found_exact = False
                    for candidate in refined_candidates:
                        clean_candidate = re.sub(r'[은는이가을를에의와과도부터까지에서부터]$', '', candidate).strip()
                        normalized_candidate = normalize_medicine_name(clean_candidate)
                        if normalized_candidate == refined_normalized:
                            found_exact = True
                            break
                    
                    if not found_exact:
                        # 잘못된 약품명을 원본 약품명으로 교체
                        for candidate in refined_candidates:
                            clean_candidate = re.sub(r'[은는이가을를에의와과도부터까지에서부터]$', '', candidate).strip()
                            normalized_candidate = normalize_medicine_name(clean_candidate)
                            similarity = calculate_similarity(normalized_candidate, refined_normalized)
                            if similarity < 0.7:  # 유사도가 낮으면 잘못된 약품명
                                refined_query = refined_query.replace(candidate, extracted_medicine)
                                print(f"✅ 약품명 복원: '{candidate}' → '{extracted_medicine}' (유사도: {similarity:.3f})")
                                print(f"✅ 최종 보정된 질문: '{refined_query}'")
                                break
                else:
                    # 일반적인 오타 보정 (원본에 정확한 약품명이 없었던 경우)
                    refined_candidates = re.findall(r'([가-힣]{2,10})(?:은|는|이|가|을|를|의|정|연고)', refined_query)
                    for candidate in refined_candidates:
                        normalized_candidate = normalize_medicine_name(candidate)
                        normalized_extracted = normalize_medicine_name(extracted_medicine)
                        similarity = calculate_similarity(normalized_candidate, normalized_extracted)
                        if 0.5 <= similarity < 1.0 and candidate != extracted_medicine:
                            refined_query = refined_query.replace(candidate, extracted_medicine)
                            print(f"✅ 약품명 오타 보정: '{candidate}' → '{extracted_medicine}' (유사도: {similarity:.3f})")
                            print(f"✅ 최종 보정된 질문: '{refined_query}'")
                            break
            else:
                print(f"⚠️ 보정된 질문에서 약품명을 찾지 못했습니다")
        
        # 성분명이 매칭된 경우 state에 저장 (다른 노드에서 사용 가능)
        if extracted_ingredient:
            state["extracted_ingredient_name"] = extracted_ingredient
            print(f"📝 성분명으로 인식: '{extracted_ingredient}' (약품명 매칭 건너뜀)")
        
        # 상태 업데이트
        state["query"] = refined_query
        state["original_query"] = original_query  # 원본 보존 (디버깅용)
        
        # 보정 여부 플래그 추가
        if refined_query != raw_query:
            state["query_was_refined"] = True
            print(f"📝 질문이 보정되었습니다: '{raw_query}' → '{refined_query}'")
        else:
            state["query_was_refined"] = False
            print(f"📝 질문 보정 불필요 (원본 유지)")
        
    except Exception as e:
        print(f"❌ 질문 보정 중 오류 발생: {e}")
        # 오류 발생 시 원본 질문 유지
        state["query"] = raw_query
        state["original_query"] = original_query
        state["query_was_refined"] = False
    
    return state

