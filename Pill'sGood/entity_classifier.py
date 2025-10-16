# entity_classifier.py - 약품명/성분명 동적 분류기

import re
import json
from typing import Dict, Set
from retrievers import llm, known_ingredients, ingredient_to_products_map

def classify_medicine_vs_ingredient(query: str, pubchem_result: Dict = None) -> Dict:
    """
    약품명인지 성분명인지 동적으로 판단 (하드코딩 없음)
    
    Args:
        query: 사용자가 질문한 약품명 또는 성분명
        pubchem_result: PubChem 검색 결과 (선택적)
    
    Returns:
        {
            "type": "product" | "ingredient" | "unknown",
            "name": "정확한 명칭",
            "confidence": "high" | "medium" | "low" | "unknown",
            "method": "exact_match" | "partial_match" | "llm_inference" | "pubchem_hint" | "fallback",
            "products": [...],  # 성분인 경우 해당 성분이 포함된 제품 목록
            "reasoning": "판단 근거"
        }
    """
    
    # 조사 제거
    query_clean = re.sub(r'[은는이가을를에의와과도부터까지에서부터]$', '', query.strip())
    
    print(f"🔍 엔티티 분류 시작: '{query}' → '{query_clean}'")
    
    # === 1단계: 빠른 휴리스틱 체크 (Excel DB의 성분 리스트와 정확 매칭) ===
    if query_clean in known_ingredients:
        products = ingredient_to_products_map.get(query_clean, [])
        print(f"✅ 성분 정확 매칭: {query_clean} (제품 {len(products)}개)")
        return {
            "type": "ingredient",
            "name": query_clean,
            "confidence": "high",
            "method": "exact_match",
            "products": products,
            "reasoning": f"Excel DB에서 성분으로 정확히 매칭됨 (제품 {len(products)}개에서 사용)"
        }
    
    # === 2단계: 부분 매칭 (예: "푸르설티아민" in "푸르설티아민질산염") ===
    for ingredient in known_ingredients:
        if query_clean in ingredient or ingredient in query_clean:
            products = ingredient_to_products_map.get(ingredient, [])
            print(f"✅ 성분 부분 매칭: '{query_clean}' → '{ingredient}' (제품 {len(products)}개)")
            return {
                "type": "ingredient",
                "name": ingredient,
                "confidence": "medium",
                "method": "partial_match",
                "products": products,
                "reasoning": f"Excel DB에서 '{ingredient}'으로 부분 매칭됨 (제품 {len(products)}개에서 사용)"
            }
    
    # === 3단계: PubChem 힌트 활용 ===
    if pubchem_result and pubchem_result.get("cid"):
        print(f"💡 PubChem에서 발견 (CID: {pubchem_result['cid']}) → 성분으로 추정")
        return {
            "type": "ingredient",
            "name": query_clean,
            "confidence": "high",
            "method": "pubchem_hint",
            "products": [],  # PubChem에만 있는 성분
            "reasoning": f"PubChem에서 발견됨 (CID: {pubchem_result['cid']}), 국제적으로 인정된 성분명"
        }
    
    # === 4단계: LLM 기반 판단 ===
    print(f"🧠 LLM 기반 분류 시도: {query_clean}")
    llm_result = _classify_with_llm(query_clean)
    
    if llm_result["type"] != "unknown":
        # LLM이 성공적으로 분류함
        if llm_result["type"] == "ingredient":
            # 성분으로 판단된 경우, Excel DB에서 해당 성분 포함 제품 찾기
            llm_result["products"] = ingredient_to_products_map.get(llm_result["name"], [])
        return llm_result
    
    # === 5단계: 기본값 (알 수 없음) ===
    print(f"⚠️ 분류 실패: {query_clean}")
    return {
        "type": "unknown",
        "name": query_clean,
        "confidence": "unknown",
        "method": "fallback",
        "products": [],
        "reasoning": "약품명인지 성분명인지 판단할 수 없습니다."
    }

def _classify_with_llm(query: str) -> Dict:
    """LLM을 사용하여 약품명/성분명 분류"""
    
    prompt = f"""당신은 약품명/성분명 분류 전문가입니다. 단계별로 분석하여 정확히 판단하세요.

## 🎯 분류 대상
용어: {query}

## 🔍 2단계 분류 프로세스

### STEP 1: 형태 분석
다음 특징을 확인하세요:
- 짧고 기억하기 쉬운가? (4-6글자) → 약품명 가능성
- 길고 복잡한 화학 용어인가? (8글자 이상) → 성분명 가능성
- 접미사: "~정", "~연고", "~캡슐", "~액" → 약품명
- 접미사: "~엔", "~민", "~올", "~산", "~염" → 성분명 가능성

### STEP 2: 의미 판단
**약품명 (제품명):**
- 브랜드명, 상표명
- 예: 타이레놀, 게보린, 베타딘, 박카스

**성분명 (화학명):**
- 국제 표준 성분명
- 예: 아세트아미노펜, 이부프로펜, 푸르설티아민, 카페인무수물

## 💡 Few-Shot 예시

### 예시 1: "타이레놀"
[분석]
- STEP 1: 4글자, 짧고 기억하기 쉬움
- STEP 2: 브랜드명, 제약회사 상표
[결과] product

### 예시 2: "아세트아미노펜"
[분석]
- STEP 1: 8글자, 복잡한 화학 용어
- STEP 2: 국제 표준 성분명
[결과] ingredient

### 예시 3: "푸르설티아민"
[분석]
- STEP 1: 7글자, "~민" 접미사
- STEP 2: 비타민 B1 유도체, 화학 성분명
[결과] ingredient

### 예시 4: "베타딘연고"
[분석]
- STEP 1: "~연고" 접미사 → 약품명
- STEP 2: 브랜드명
[결과] product

## 📤 출력 형식 (JSON)
{{
    "type": "product|ingredient|unknown",
    "name": "정확한 명칭",
    "reasoning": "STEP 1-2 분석 근거 (1문장)"
}}

**중요:** 확실하지 않으면 "unknown"으로 응답하세요.
"""
    
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # JSON 마크다운 제거
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            if json_end != -1:
                content = content[json_start:json_end].strip()
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            if json_end != -1:
                content = content[json_start:json_end].strip()
        
        result = json.loads(content)
        
        print(f"🧠 LLM 분류 결과: {result['type']} (근거: {result.get('reasoning', '')})")
        
        return {
            "type": result.get("type", "unknown"),
            "name": result.get("name", query),
            "confidence": "medium",
            "method": "llm_inference",
            "products": [],  # 여기서는 빈 리스트, 나중에 채워짐
            "reasoning": result.get("reasoning", "LLM이 분류함")
        }
        
    except json.JSONDecodeError as e:
        print(f"⚠️ LLM JSON 파싱 실패: {e}")
        return {
            "type": "unknown",
            "name": query,
            "confidence": "low",
            "method": "llm_inference",
            "products": [],
            "reasoning": "LLM 응답을 파싱할 수 없음"
        }
    except Exception as e:
        print(f"❌ LLM 분류 오류: {e}")
        return {
            "type": "unknown",
            "name": query,
            "confidence": "low",
            "method": "llm_inference",
            "products": [],
            "reasoning": f"LLM 분류 중 오류 발생: {str(e)}"
        }

def extract_target_from_query(query: str) -> str:
    """질문에서 대상 약품명/성분명 추출"""
    # "푸르설티아민이 뭐야?" → "푸르설티아민"
    # "아세트아미노펜은 뭐야?" → "아세트아미노펜"
    # "타이레놀 부작용은?" → "타이레놀"
    
    patterns = [
        r'([가-힣a-zA-Z]+)[이가은는을를]?\s*(뭐|무엇|어떤|어떻게)',
        r'([가-힣a-zA-Z]+)[의]?\s*(부작용|효능|사용법|작용기전)',
        r'([가-힣a-zA-Z]+)',  # 기본 패턴
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            target = match.group(1).strip()
            # 조사 제거
            target = re.sub(r'[은는이가을를에의와과도부터까지에서부터]$', '', target)
            return target
    
    return query.strip()


