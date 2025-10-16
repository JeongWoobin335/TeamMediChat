# enhanced_rag_system.py - 통합 RAG 시스템

import time
import json
from typing import Dict, List, Optional
from qa_state import QAState
from retrievers import (
    excel_docs, pdf_structured_docs, 
    extract_active_ingredients_from_medicine,
    llm
)
from pubchem_api import PubChemAPI
from translation_rag import TranslationRAG
from answer_utils import generate_response_llm_from_prompt

# YouTube 검색 함수 import
from sns_node import search_youtube_videos, get_video_transcript, summarize_video_content

# 네이버 뉴스 API import
from naver_news_api import NaverNewsAPI

class EnhancedRAGSystem:
    """통합 RAG 시스템 - 여러 DB에서 정보를 수집하고 조합하여 근거 있는 답변 생성"""
    
    def __init__(self):
        self.pubchem_api = PubChemAPI()
        self.translation_rag = TranslationRAG()
        self.naver_news_api = NaverNewsAPI()  # 인스턴스 생성
        self.llm = llm
    
    def analyze_medicine_comprehensively(self, medicine_name: str, usage_context: str) -> Dict:
        """약품 종합 분석 - 진정한 RAG 구현 (YouTube 통합)"""
        print(f"🔍 종합 약품 분석 시작: {medicine_name} → {usage_context}")
        
        analysis_result = {
            'medicine_name': medicine_name,
            'usage_context': usage_context,
            'excel_info': {},
            'pdf_info': {},
            'korean_ingredient_info': {},
            'international_ingredient_info': {},
            'youtube_info': {},  # ✅ YouTube 정보 추가
            'naver_news_info': {},  # ✅ 네이버 뉴스 정보 추가
            'combined_analysis': {},
            'evidence_based_response': '',
            'follow_up_questions': [],
            'analysis_timestamp': time.time()
        }
        
        try:
            # 1단계: Excel DB에서 기본 약품 정보 수집
            print("📊 1단계: Excel DB에서 기본 정보 수집...")
            excel_info = self._get_excel_medicine_info(medicine_name)
            analysis_result['excel_info'] = excel_info
            
            # 2단계: PDF DB 검색 제거 (Excel DB만 사용)
            print("📄 2단계: PDF DB 검색 건너뜀 (Excel DB만 사용)")
            analysis_result['pdf_info'] = {}
            
            # 3단계: 주성분 추출
            print("🧪 3단계: 주성분 추출...")
            active_ingredients = self._extract_active_ingredients(medicine_name, excel_info)
            print(f"  추출된 주성분: {active_ingredients}")
            
            # 4단계: 각 주성분에 대한 상세 분석
            korean_ingredient_info = {}
            international_ingredient_info = {}
            
            for ingredient in active_ingredients:
                print(f"🔍 주성분 분석: {ingredient}")
                
                # PubChem에서 국제 정보 수집 (한국어명 자동 변환)
                print(f"  🌍 PubChem에서 {ingredient} 정보 수집...")
                international_info = self.pubchem_api.analyze_ingredient_comprehensive(ingredient)
                
                # 번역 RAG로 영어 정보를 한국어로 번역
                print(f"  🔄 {ingredient} 정보 번역 중...")
                translated_info = self.translation_rag.translate_pharmacology_info(international_info)
                international_ingredient_info[ingredient] = {
                    'original': international_info,
                    'translated': translated_info
                }
            
            analysis_result['korean_ingredient_info'] = korean_ingredient_info
            analysis_result['international_ingredient_info'] = international_ingredient_info
            
            # 4.5단계: YouTube에서 실전 정보 수집
            print("📺 4.5단계: YouTube에서 실전 정보 수집...")
            youtube_info = self._search_youtube_info(medicine_name, usage_context, active_ingredients)
            analysis_result['youtube_info'] = youtube_info
            
            # 4.6단계: 네이버 뉴스에서 추가 정보 수집 (✅ 신제품, 트렌드 등)
            print("📰 4.6단계: 네이버 뉴스에서 추가 정보 수집...")
            naver_news_info = self._search_naver_news_info(medicine_name, active_ingredients)
            analysis_result['naver_news_info'] = naver_news_info
            
            # 5단계: LLM이 모든 정보를 조합하여 근거 있는 분석 수행
            print("🧠 5단계: LLM 종합 분석 (YouTube, 네이버 뉴스 정보 포함)...")
            combined_analysis = self._perform_llm_analysis(
                medicine_name, usage_context, analysis_result
            )
            analysis_result['combined_analysis'] = combined_analysis
            
            # 6단계: 근거 기반 답변 생성
            print("📝 6단계: 근거 기반 답변 생성...")
            evidence_based_response = self._generate_evidence_based_response(
                medicine_name, usage_context, analysis_result
            )
            analysis_result['evidence_based_response'] = evidence_based_response
            
            # 7단계: 추가 질문 생성
            print("❓ 7단계: 추가 질문 생성...")
            follow_up_questions = self._generate_follow_up_questions(analysis_result)
            analysis_result['follow_up_questions'] = follow_up_questions
            
            print(f"✅ 종합 분석 완료: {medicine_name}")
            
        except Exception as e:
            print(f"❌ 종합 분석 오류: {e}")
            analysis_result['error'] = str(e)
        
        return analysis_result
    
    def _get_excel_medicine_info(self, medicine_name: str) -> Dict:
        """Excel DB에서 약품 정보 수집 (부분 매칭 지원)"""
        # 정확한 매칭 시도
        for doc in excel_docs:
            if doc.metadata.get("제품명") == medicine_name:
                return {
                    'product_name': doc.metadata.get("제품명", ""),
                    'main_ingredient': doc.metadata.get("주성분", ""),
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
        
        # 정확한 매칭이 없으면 부분 매칭 시도 (수출명 문제 해결)
        print(f"🔍 Enhanced RAG 정확한 매칭 실패, 부분 매칭 시도: {medicine_name}")
        for doc in excel_docs:
            product_name = doc.metadata.get("제품명", "")
            # 약품명이 제품명의 시작 부분과 일치하는지 확인
            if product_name.startswith(medicine_name) or medicine_name in product_name:
                print(f"  Enhanced RAG 부분 매칭 발견: '{product_name}' (검색어: '{medicine_name}')")
                return {
                    'product_name': doc.metadata.get("제품명", ""),
                    'main_ingredient': doc.metadata.get("주성분", ""),
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
        
        print(f"❌ Enhanced RAG에서 '{medicine_name}' 약품 정보를 찾을 수 없음")
        return {}
    
    def _get_pdf_medicine_info(self, medicine_name: str) -> Dict:
        """PDF DB에서 약품 정보 수집"""
        for doc in pdf_structured_docs:
            if doc.metadata.get("제품명") == medicine_name:
                return {
                    'product_name': doc.metadata.get("제품명", ""),
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
        return {}
    
    def _extract_active_ingredients(self, medicine_name: str, excel_info: Dict) -> List[str]:
        """주성분 추출"""
        ingredients = []
        
        # Excel 정보에서 주성분 추출
        if excel_info.get('main_ingredient') and excel_info['main_ingredient'] != '정보 없음':
            main_ingredient = excel_info['main_ingredient']
            print(f"🔍 {medicine_name} 주성분 추출: {main_ingredient}")
            
            # 쉼표로 구분된 성분들을 개별적으로 분리
            if ',' in main_ingredient:
                ingredients = [ing.strip() for ing in main_ingredient.split(',') if ing.strip()]
                print(f"  분리된 성분들: {ingredients}")
            else:
                ingredients = [main_ingredient.strip()]
                print(f"  단일 성분: {ingredients}")
        else:
            print(f"  주성분 정보 없음")
        
        # 기존 함수 사용 (백업)
        if not ingredients:
            ingredients = extract_active_ingredients_from_medicine(medicine_name)
        
        return ingredients
    
    def _perform_llm_analysis(self, medicine_name: str, usage_context: str, analysis_result: Dict) -> Dict:
        """LLM이 모든 정보를 조합하여 분석 수행 (YouTube, 네이버 뉴스 포함)"""
        
        # 모든 수집된 정보를 정리 (번역된 정보 우선 사용)
        collected_info = {
            'medicine_name': medicine_name,
            'usage_context': usage_context,
            'excel_info': analysis_result['excel_info'],
            'pdf_info': analysis_result['pdf_info'],
            'korean_ingredient_info': analysis_result['korean_ingredient_info'],
            'international_ingredient_info': analysis_result['international_ingredient_info'],
            'youtube_info': analysis_result.get('youtube_info', {}),
            'naver_news_info': analysis_result.get('naver_news_info', {})  # ✅ 네이버 뉴스 정보 추가
        }
        
        # 번역된 정보를 별도로 정리
        translated_summaries = {}
        for ingredient, info in analysis_result['international_ingredient_info'].items():
            if 'translated' in info and 'summary_kr' in info['translated']:
                translated_summaries[ingredient] = info['translated']['summary_kr']
        
        # ✅ YouTube 정보 요약
        youtube_summary = self._format_youtube_info(analysis_result.get('youtube_info', {}))
        
        # ✅ 네이버 뉴스 정보 요약
        naver_news_summary = self._format_naver_news_info(analysis_result.get('naver_news_info', {}))
        
        analysis_prompt = f"""당신은 다중 소스 의약품 정보 통합 전문가입니다. 여러 소스의 정보를 종합하여 근거 있는 분석을 제공하세요.

## 🎯 분석 목표
- 약품: {medicine_name}
- 사용 목적: {usage_context}

## 📚 수집된 정보 (다중 소스)

### 소스 1: 한국 의약품 정보 DB - Excel (신뢰도: 높음)
{json.dumps(collected_info['excel_info'], indent=2, ensure_ascii=False)}

### 소스 2: 국제 성분 DB (PubChem, 신뢰도: 높음)
{json.dumps(translated_summaries, indent=2, ensure_ascii=False)}

### 소스 3: 전문가 의견 & 실사용 경험 (신뢰도: 중간~높음)
{youtube_summary}

### 소스 4: 최신 뉴스 & 추가 정보 (신뢰도: 중간, 참고용)
{naver_news_summary}

## 🔍 4단계 통합 분석 프로세스

### STEP 1: 소스 신뢰도 평가
각 소스의 정보 품질을 평가하세요:
- 한국 의약품 정보 DB: 공식 의약품 정보 (최우선)
- PubChem: 국제 표준 약리학 데이터
- YouTube/전문가: 실전 경험 (약사/의사 검증 필요)

출력: 어느 소스가 가장 신뢰할 만한지 판단

### STEP 2: 정보 일관성 검증
다중 소스 간 정보 교차 검증:
1. 효능/작용기전 일치 여부
2. 부작용 정보 일치 여부
3. 사용법 일치 여부

**모순 탐지:**
- 소스 간 모순 발견 시 명시하고, 더 신뢰할 소스를 우선
- 예: Excel "두통 완화" vs YouTube "근육통 완화" → Excel 우선

출력: 모순 있음/없음, 모순 내용, 해결 방안

### STEP 3: 작용기전 및 안전성 종합 분석
**주성분별 상세 분석:**
각 주성분에 대해:
1. 약리학적 작용기전 (어떻게 작용하는가?)
2. 사용 목적과의 연관성 점수 (0~100%)
3. 부작용 심각도 (경미/보통/심각)

**예시:**
- 아세트아미노펜: COX-2 억제 → 프로스타글란딘 감소 → 통증 완화
- 두통 사용: 95% 연관 (직접 효과)
- 부작용: 경미 (과다 복용 시 간 손상 주의)

출력: 각 성분의 메커니즘, 연관성, 안전성

### STEP 4: 근거 기반 최종 결론
**종합 판단 기준:**
- 사용 가능: 연관성 ≥ 50% + 안전성 경미~보통
- 사용 불가: 연관성 < 50% 또는 안전성 심각

**신뢰도 레벨:**
- high: 모든 소스 일치 + 명확한 과학적 근거
- medium: 일부 소스만 또는 간접적 근거
- low: 정보 부족 또는 모순 존재

## 💡 분석 예시

### 예시 1: 타이레놀 (아세트아미노펜) + 두통
STEP 1: Excel(높음), PubChem(높음) 신뢰
STEP 2: 모순 없음 - 모두 "두통 완화" 명시
STEP 3: COX 억제 → 통증 감소, 연관성 95%, 부작용 경미
STEP 4: 사용 가능 (신뢰도: high)

### 예시 2: 습진 연고 + 근육통
STEP 1: Excel(높음) 신뢰
STEP 2: 모순 없음
STEP 3: 항염증(피부용) → 근육통 무관, 연관성 5%, 부작용 보통
STEP 4: 사용 불가 (신뢰도: high)

## 📤 출력 형식 (JSON)
{{
    "safe_to_use": true/false,
    "confidence_level": "high/medium/low",
    "source_reliability": {{
        "korean_db": "high/medium/low",
        "pubchem": "high/medium/low",
        "expert_videos": "high/medium/low"
    }},
    "contradiction_detected": true/false,
    "contradiction_details": "모순 내용 상세 설명 (없으면 빈 문자열)",
    "mechanism_analysis": "각 주성분의 약리학적 작용기전 상세 설명 (2-3문장, 구체적 메커니즘 포함)",
    "efficacy_match_score": 0~100,
    "safety_level": "mild/moderate/severe",
    "safety_assessment": "안전성 종합 평가 (1-2문장)",
    "contraindications": ["금기사항1", "금기사항2"],
    "precautions": ["주의사항1", "주의사항2"],
    "evidence_summary": "판단 근거 요약 (어느 소스에서 어떤 정보 활용했는지 명시)",
    "alternative_suggestions": ["대안1", "대안2"],
    "expert_recommendation": "최종 전문가 권고사항"
}}

**중요 지침:**
- 반드시 STEP 1-4 순서로 사고하세요
- mechanism_analysis는 구체적 메커니즘 필수 (예: "COX-2 억제", "세로토닌 재흡수 차단")
- 추측 금지 - 주어진 정보만 사용
- 모순 발견 시 신뢰도 높은 소스 우선
- 불확실하면 confidence_level 낮추고 이유 명시
"""
        
        try:
            response = self.llm.invoke(analysis_prompt)
            
            # JSON 응답 파싱
            try:
                if "```json" in response.content:
                    json_start = response.content.find("```json") + 7
                    json_end = response.content.find("```", json_start)
                    if json_end != -1:
                        json_str = response.content[json_start:json_end].strip()
                    else:
                        json_str = response.content[json_start:].strip()
                else:
                    json_str = response.content.strip()
                
                analysis = json.loads(json_str)
                return analysis
                
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 기본 응답 (새 필드 포함)
                return {
                    "safe_to_use": False,
                    "confidence_level": "low",
                    "source_reliability": {
                        "korean_db": "unknown",
                        "pubchem": "unknown",
                        "expert_videos": "unknown"
                    },
                    "contradiction_detected": False,
                    "contradiction_details": "",
                    "mechanism_analysis": "분석 중 오류 발생",
                    "efficacy_match_score": 0,
                    "safety_level": "unknown",
                    "safety_assessment": "안전성 평가를 완료할 수 없습니다",
                    "contraindications": [],
                    "precautions": ["의사나 약사와 상담하세요"],
                    "evidence_summary": "정보 분석 중 오류가 발생했습니다",
                    "alternative_suggestions": [],
                    "expert_recommendation": "의료진과 상담을 권장합니다"
                }
                
        except Exception as e:
            print(f"❌ LLM 분석 오류: {e}")
            return {
                "safe_to_use": False,
                "confidence_level": "low",
                "source_reliability": {
                    "korean_db": "unknown",
                    "pubchem": "unknown",
                    "expert_videos": "unknown"
                },
                "contradiction_detected": False,
                "contradiction_details": "",
                "mechanism_analysis": f"분석 오류: {str(e)}",
                "efficacy_match_score": 0,
                "safety_level": "unknown",
                "safety_assessment": "안전성 평가를 완료할 수 없습니다",
                "contraindications": [],
                "precautions": ["의사나 약사와 상담하세요"],
                "evidence_summary": "정보 분석 중 오류가 발생했습니다",
                "alternative_suggestions": [],
                "expert_recommendation": "의료진과 상담을 권장합니다"
            }
    
    def _generate_evidence_based_response(self, medicine_name: str, usage_context: str, analysis_result: Dict) -> str:
        """근거 기반 답변 생성 - 자연스러운 대화형 답변 (YouTube, 네이버 뉴스 통합)"""
        
        # 수집된 모든 정보를 정리
        excel_info = analysis_result.get('excel_info', {})
        korean_info = analysis_result.get('korean_ingredient_info', {})
        international_info = analysis_result.get('international_ingredient_info', {})
        youtube_info = analysis_result.get('youtube_info', {})
        naver_news_info = analysis_result.get('naver_news_info', {})  # ✅ 네이버 뉴스 정보 추가
        combined_analysis = analysis_result.get('combined_analysis', {})
        
        # 동적 대안 약품 검색
        print("🔍 동적 대안 약품 검색 중...")
        alternative_medicines = self._find_similar_medicines_dynamically(medicine_name, usage_context, excel_info)
        print(f"✅ 발견된 대안 약품: {[alt['name'] for alt in alternative_medicines]}")
        
        # LLM에게 자연스러운 답변 생성 요청
        prompt = f"""
당신은 친근하고 전문적인 약사입니다. 사용자의 질문에 대해 수집된 정보를 바탕으로 자연스럽고 대화형으로 답변해주세요.

**사용자 질문:** {medicine_name}은(는) {usage_context}에 먹어도 되나?

**수집된 정보:**

1. **Excel DB 정보:**
{excel_info.get('content', '정보 없음')}

2. **한국 의약품 DB 정보:**
{self._format_korean_info(korean_info)}

3. **PubChem 국제 정보:**
{self._format_international_info(international_info)}

4. **추가 실전 정보 (전문가 의견, 사용 팁, 경험담):**
{self._format_youtube_info(youtube_info)}

5. **최신 뉴스 & 추가 정보 (신제품, 트렌드 등):**
{self._format_naver_news_info(naver_news_info)}

6. **종합 분석 결과:**
- 사용 가능성: {combined_analysis.get('safe_to_use', 'Unknown')}
- 신뢰도: {combined_analysis.get('confidence_level', 'Unknown')}
- 작용기전: {combined_analysis.get('mechanism_analysis', '정보 없음')}
- 안전성 평가: {combined_analysis.get('safety_assessment', '정보 없음')}
- 주의사항: {combined_analysis.get('precautions', [])}
- 금기사항: {combined_analysis.get('contraindications', [])}
- 대안 제안: {combined_analysis.get('alternative_suggestions', [])}
- 전문가 권고: {combined_analysis.get('expert_recommendation', '정보 없음')}

6. **동적 대안 약품 분석:**
{self._format_alternative_medicines(alternative_medicines)}

**답변 요구사항:**
1. 친근하고 대화하는 톤으로 답변
2. **수집된 모든 정보를 자연스럽게 조합**하여 설명 (출처 언급 금지!)
3. **반드시 구체적인 작용기전을 포함하여 상세한 근거 제시**
4. **실전 사용 팁, 전문가 의견, 주의사항 등을 자연스럽게 녹여서 설명** (있는 경우)
5. **최신 뉴스 정보가 있다면 자연스럽게 추가 정보로 제공** (신제품, 트렌드 등)
6. 주의사항과 금기사항을 자연스럽게 언급
7. 필요시 대안도 제안
8. 마지막에 의료진 상담 권고

**중요 지침:**
- "YouTube에서는...", "뉴스에서...", "영상에서...", "Excel DB에서...", "PubChem에서..." 같은 **출처 언급 절대 금지**
- 모든 정보를 **하나의 통합된 지식**처럼 자연스럽게 설명
- 예: "전문가들은...", "알려진 바로는...", "일반적으로...", "최근에는..." 같은 표현 사용
- 뉴스 정보는 "참고로..." 또는 "💡 알아두면 좋은 정보" 섹션에 자연스럽게 추가

**답변 구조 (반드시 이 순서로):**
1. **결론**: "네, {medicine_name}은(는) {usage_context}에 사용하실 수 있습니다" 또는 "아니요, 권장하지 않습니다"
2. **상세한 작용기전**: 각 주성분의 구체적인 작용 메커니즘을 설명
3. **효과**: 해당 증상에 어떤 효과가 있는지 구체적으로 설명
4. **주의사항**: 구체적인 주의사항과 금기사항
5. **대안**: 위에서 제공된 동적 대안 약품 분석 결과를 바탕으로 구체적인 대안 약품 제안 (실제 약품명만 사용, 이부프로펜/나프록센 같은 성분명 사용 금지, 각 대안의 주성분과 효과 근거 포함)
6. **💡 알아두면 좋은 정보** (⚠️ 중요: 이 섹션을 풍부하게 작성하세요):
   - 추가 실전 정보에서 발견한 **모든 흥미로운 사실** (치매 예방, 뇌세포 보호, 면역력 강화 등)
   - **최신 뉴스 정보 모두 포함** (신제품 출시, 리뉴얼, 성분 연구, 트렌드, 소비자 반응 등)
   - **YouTube에서 발견한 실전 팁** (복용 시간, 음식 궁합, 효과 극대화 방법 등)
   - **성분 관련 최신 연구** (효능 입증, 새로운 발견 등)
   - 위의 정보들을 **자연스럽게 여러 문단으로** 작성하되, 출처는 언급하지 말고 "알려진 바로는", "최근에는", "전문가들은" 등의 표현 사용
   - **최소 3-5개의 구체적인 추가 정보**를 제공할 것
7. **마무리**: 의료진 상담 권고

**중요 지침:**
- 작용기전 설명 시 "중추신경계에서 프로스타글란딘 합성을 억제하여..." 같은 구체적인 메커니즘 포함
- 단순히 "통증을 줄이고 열을 내린다"가 아닌 "어떻게" 작용하는지 설명
- 모든 약품에 대해 동일한 수준의 상세함을 유지
- 의학적으로 정확하면서도 이해하기 쉽게 설명
- **대안 약품 제시 시 반드시 위에서 제공된 동적 대안 약품 분석 결과만 사용**
- **이부프로펜, 나프록센 같은 성분명을 대안으로 제시하지 말고, 실제 약품명(포펜정, 타이레놀 등)만 사용**

자연스럽고 친근한 톤으로 답변해주세요.
"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"❌ 자연스러운 답변 생성 오류: {e}")
            # 오류 시 기본 템플릿 답변
            return self._generate_fallback_response(medicine_name, usage_context, combined_analysis)
    
    def _format_korean_info(self, korean_info: Dict) -> str:
        """한국 의약품 DB 정보 포맷팅"""
        if not korean_info:
            return "정보 없음"
        
        formatted = []
        for ingredient, info in korean_info.items():
            if info.get('detail_info'):
                detail = info['detail_info']
                # 더 상세한 정보 포함
                mechanism = detail.get('작용기전', '정보 없음')
                pharmacology = detail.get('약동학', '정보 없음')
                if mechanism != '정보 없음':
                    formatted.append(f"- {ingredient} 작용기전: {mechanism}")
                if pharmacology != '정보 없음':
                    formatted.append(f"- {ingredient} 약동학: {pharmacology}")
        
        return "\n".join(formatted) if formatted else "정보 없음"
    
    def _format_international_info(self, international_info: Dict) -> str:
        """PubChem 정보 포맷팅"""
        if not international_info:
            return "정보 없음"
        
        formatted = []
        for ingredient, info in international_info.items():
            # 더 상세한 정보 포함
            if info.get('description'):
                formatted.append(f"- {ingredient} 설명: {info['description'][:300]}...")
            if info.get('basic_info', {}).get('MechanismOfAction'):
                formatted.append(f"- {ingredient} 작용기전: {info['basic_info']['MechanismOfAction']}")
            if info.get('detailed_info', {}).get('MechanismOfAction'):
                formatted.append(f"- {ingredient} 상세 작용기전: {info['detailed_info']['MechanismOfAction']}")
        
        return "\n".join(formatted) if formatted else "정보 없음"
    
    def _format_youtube_info(self, youtube_info: Dict) -> str:
        """실전 정보 포맷팅 (전문가 의견, 사용 경험 등 - 출처 숨김)"""
        if not youtube_info or youtube_info.get('total_videos', 0) == 0:
            return "추가 실전 정보 없음"
        
        formatted = []
        formatted.append(f"총 {youtube_info['total_videos']}개 전문 정보원 참조 (상세 자료: {youtube_info.get('has_transcript_count', 0)}개)")
        
        # 약품 관련 정보 (더 많이, 더 길게)
        medicine_videos = youtube_info.get('medicine_videos', [])
        if medicine_videos:
            formatted.append("\n💊 약품 관련 실전 정보:")
            for i, video in enumerate(medicine_videos[:8], 1):  # 8개로 증가
                formatted.append(f"  {i}. {video['title']}")
                if video.get('has_transcript'):
                    formatted.append(f"     핵심 내용: {video.get('summary', '')[:600]}...")  # 600자로 증가
                else:
                    formatted.append(f"     개요: {video.get('description', '')[:300]}...")
        
        # 성분 관련 정보 (더 많이, 더 길게)
        ingredient_videos = youtube_info.get('ingredient_videos', [])
        if ingredient_videos:
            formatted.append("\n🧪 성분 관련 전문 정보:")
            for i, video in enumerate(ingredient_videos[:6], 1):  # 6개로 증가
                formatted.append(f"  {i}. {video['title']}")
                if video.get('has_transcript'):
                    formatted.append(f"     핵심 내용: {video.get('summary', '')[:600]}...")  # 600자로 증가
        
        # 사용법 관련 정보 (더 많이, 더 길게)
        usage_videos = youtube_info.get('usage_videos', [])
        if usage_videos:
            formatted.append("\n💡 사용법 및 팁:")
            for i, video in enumerate(usage_videos[:4], 1):  # 4개로 증가
                formatted.append(f"  {i}. {video['title']}")
                if video.get('has_transcript'):
                    formatted.append(f"     핵심 내용: {video.get('summary', '')[:600]}...")  # 600자로 증가
        
        return "\n".join(formatted) if formatted else "추가 실전 정보 없음"
    
    def _generate_fallback_response(self, medicine_name: str, usage_context: str, combined_analysis: Dict) -> str:
        """오류 시 기본 답변"""
        if combined_analysis.get('safe_to_use'):
            response = f"네, {medicine_name}은(는) {usage_context}에 사용하실 수 있습니다.\n\n"
        else:
            response = f"아니요, {medicine_name}은(는) {usage_context}에 사용을 권장하지 않습니다.\n\n"
        
        if combined_analysis.get('mechanism_analysis'):
            response += f"이유는 {combined_analysis['mechanism_analysis']}\n\n"
        
        if combined_analysis.get('precautions'):
            response += "주의하실 점은:\n"
            for precaution in combined_analysis['precautions']:
                response += f"- {precaution}\n"
            response += "\n"
        
        response += "정확한 진단을 위해서는 의사나 약사와 상담하시기 바랍니다."
        
        return response
    
    def _generate_follow_up_questions(self, analysis_result: Dict) -> List[str]:
        """추가 질문 생성"""
        questions = []
        
        # 주성분 관련 질문
        korean_info = analysis_result.get('korean_ingredient_info', {})
        for ingredient, info in korean_info.items():
            if info.get('detail_info'):
                questions.append(f"{ingredient}의 작용기전이 궁금하신가요?")
                questions.append(f"{ingredient}의 부작용에 대해 더 자세히 알고 싶으신가요?")
        
        # 사용법 관련 질문
        questions.append("이 약의 정확한 사용법이 궁금하신가요?")
        questions.append("다른 약과 함께 복용해도 되는지 궁금하신가요?")
        
        # 대안 관련 질문
        questions.append("비슷한 효과의 다른 약품이 궁금하신가요?")
        questions.append("자연 치료법에 대해 알고 싶으신가요?")
        
        return questions[:5]  # 최대 5개 질문
    
    def _find_similar_medicines_dynamically(self, medicine_name: str, usage_context: str, excel_info: Dict) -> List[Dict]:
        """Excel DB에서 동적으로 유사 약품 검색 (동일 성분 우선순위)"""
        print(f"🔍 동적 유사 약품 검색: {medicine_name} → {usage_context}")
        
        # 대상 약품의 주성분 추출
        target_ingredients = self._extract_ingredients_from_excel_info(excel_info)
        print(f"  대상 약품 주성분: {target_ingredients}")
        
        # 1단계: 동일 성분 약품 검색 (최고 우선순위)
        same_ingredient_medicines = self._find_medicines_with_same_ingredients(medicine_name, target_ingredients)
        print(f"  동일 성분 약품: {[med['name'] for med in same_ingredient_medicines]}")
        
        # 2단계: 유사 성분 약품 검색 (2순위)
        similar_ingredient_medicines = self._find_medicines_with_similar_ingredients(medicine_name, target_ingredients)
        print(f"  유사 성분 약품: {[med['name'] for med in similar_ingredient_medicines]}")
        
        # 3단계: 효능 기반 약품 검색 (3순위)
        efficacy_based_medicines = self._find_medicines_by_efficacy(medicine_name, usage_context, target_ingredients)
        print(f"  효능 기반 약품: {[med['name'] for med in efficacy_based_medicines]}")
        
        # 우선순위별로 정렬하여 상위 3개 반환
        all_medicines = same_ingredient_medicines + similar_ingredient_medicines + efficacy_based_medicines
        
        # 우선순위와 유사도를 모두 고려하여 정렬 (동일 성분 > 유사 성분 > 효능 기반)
        all_medicines.sort(key=lambda x: (x.get("priority", 999), -x["similarity_score"]))
        
        # 상위 3개 반환하되, 동일/유사 성분이 있으면 그것을 우선
        result = []
        if same_ingredient_medicines:
            result.extend(same_ingredient_medicines[:2])  # 동일 성분 최대 2개
        if similar_ingredient_medicines and len(result) < 3:
            remaining = 3 - len(result)
            result.extend(similar_ingredient_medicines[:remaining])
        if len(result) < 3:
            remaining = 3 - len(result)
            result.extend(efficacy_based_medicines[:remaining])
        
        return result[:3]
    
    def _extract_ingredients_from_excel_info(self, excel_info: Dict) -> List[str]:
        """Excel 정보에서 주성분 추출"""
        ingredients = []
        
        if excel_info.get('main_ingredient') and excel_info['main_ingredient'] != '정보 없음':
            main_ingredient = excel_info['main_ingredient']
            if ',' in main_ingredient:
                ingredients = [ing.strip() for ing in main_ingredient.split(',') if ing.strip()]
            else:
                ingredients = [main_ingredient.strip()]
        
        return ingredients
    
    def _extract_ingredients_from_doc(self, doc) -> List[str]:
        """문서에서 주성분 추출"""
        ingredients = []
        
        # 메타데이터에서 주성분 추출
        if doc.metadata.get("주성분") and doc.metadata["주성분"] != "정보 없음":
            main_ingredient = doc.metadata["주성분"]
            if ',' in main_ingredient:
                ingredients = [ing.strip() for ing in main_ingredient.split(',') if ing.strip()]
            else:
                ingredients = [main_ingredient.strip()]
        
        return ingredients
    
    def _calculate_ingredient_similarity(self, target_ingredients: List[str], doc_ingredients: List[str]) -> float:
        """주성분 유사도 계산"""
        if not target_ingredients or not doc_ingredients:
            return 0.0
        
        # 정규화된 성분명으로 변환
        target_normalized = [self._normalize_ingredient_name(ing) for ing in target_ingredients]
        doc_normalized = [self._normalize_ingredient_name(ing) for ing in doc_ingredients]
        
        # 교집합 계산
        common_ingredients = set(target_normalized) & set(doc_normalized)
        
        if not common_ingredients:
            return 0.0
        
        # 유사도 = 교집합 크기 / 합집합 크기
        union_size = len(set(target_normalized) | set(doc_normalized))
        similarity = len(common_ingredients) / union_size
        
        return similarity
    
    def _normalize_ingredient_name(self, ingredient: str) -> str:
        """성분명 정규화"""
        if not ingredient:
            return ""
        
        # 소문자 변환 및 특수문자 제거
        normalized = ingredient.lower().strip()
        normalized = ''.join(c for c in normalized if c.isalnum() or c in '가-힣')
        
        return normalized
    
    def _extract_efficacy_from_doc(self, doc) -> str:
        """문서에서 효능 추출"""
        content = doc.page_content
        
        # 효능 패턴 찾기
        import re
        efficacy_patterns = [
            r'\[효능\]:\s*([^\[\n]+)',
            r'효능[:\s]*([^\[\n]+)',
            r'이 약의 효능은 무엇입니까\?\s*([^\[\n]+)'
        ]
        
        for pattern in efficacy_patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1).strip()
        
        return "정보 없음"
    
    def _format_alternative_medicines(self, alternative_medicines: List[Dict]) -> str:
        """대안 약품 정보 포맷팅 (실제 약품명 우선)"""
        if not alternative_medicines:
            return "대안 약품 없음"
        
        formatted = []
        for i, alt in enumerate(alternative_medicines, 1):
            # 우선순위에 따른 표시
            priority_text = ""
            if alt.get("priority") == 1:
                priority_text = " (동일 성분)"
            elif alt.get("priority") == 2:
                priority_text = " (유사 성분)"
            elif alt.get("priority") == 3:
                priority_text = " (효능 기반)"
            
            formatted.append(f"- {alt['name']}{priority_text}: {', '.join(alt['ingredients'])}")
            formatted.append(f"  효능: {alt['efficacy']}")
        
        return "\n".join(formatted)
    
    def _find_medicines_with_same_ingredients(self, medicine_name: str, target_ingredients: List[str]) -> List[Dict]:
        """동일 성분을 가진 약품 검색 (최고 우선순위)"""
        same_ingredient_medicines = []
        
        for doc in excel_docs:
            doc_name = doc.metadata.get("제품명", "")
            if doc_name == medicine_name:  # 자기 자신은 제외
                continue
                
            doc_ingredients = self._extract_ingredients_from_doc(doc)
            if not doc_ingredients:
                continue
            
            # 동일 성분 확인 (순서 무관)
            if set(target_ingredients) == set(doc_ingredients):
                same_ingredient_medicines.append({
                    "name": doc_name,
                    "ingredients": doc_ingredients,
                    "similarity_score": 1.0,  # 완전 일치
                    "efficacy": self._extract_efficacy_from_doc(doc),
                    "content": doc.page_content,
                    "priority": 1  # 최고 우선순위
                })
        
        return same_ingredient_medicines
    
    def _find_medicines_with_similar_ingredients(self, medicine_name: str, target_ingredients: List[str]) -> List[Dict]:
        """유사 성분을 가진 약품 검색 (2순위)"""
        similar_ingredient_medicines = []
        
        for doc in excel_docs:
            doc_name = doc.metadata.get("제품명", "")
            if doc_name == medicine_name:  # 자기 자신은 제외
                continue
                
            doc_ingredients = self._extract_ingredients_from_doc(doc)
            if not doc_ingredients:
                continue
            
            # 유사도 계산
            similarity_score = self._calculate_ingredient_similarity(target_ingredients, doc_ingredients)
            
            # 50% 이상 유사하고 완전 일치가 아닌 경우
            if 0.5 <= similarity_score < 1.0:
                similar_ingredient_medicines.append({
                    "name": doc_name,
                    "ingredients": doc_ingredients,
                    "similarity_score": similarity_score,
                    "efficacy": self._extract_efficacy_from_doc(doc),
                    "content": doc.page_content,
                    "priority": 2  # 2순위
                })
        
        return similar_ingredient_medicines
    
    def _find_medicines_by_efficacy(self, medicine_name: str, usage_context: str, target_ingredients: List[str]) -> List[Dict]:
        """효능 기반 약품 검색 (3순위)"""
        efficacy_based_medicines = []
        
        for doc in excel_docs:
            doc_name = doc.metadata.get("제품명", "")
            if doc_name == medicine_name:  # 자기 자신은 제외
                continue
                
            doc_ingredients = self._extract_ingredients_from_doc(doc)
            if not doc_ingredients:
                continue
            
            # 효능 기반 유사도 계산
            efficacy_similarity = self._calculate_efficacy_similarity(usage_context, doc)
            
            # 30% 이상 유사한 경우
            if efficacy_similarity > 0.3:
                efficacy_based_medicines.append({
                    "name": doc_name,
                    "ingredients": doc_ingredients,
                    "similarity_score": efficacy_similarity,
                    "efficacy": self._extract_efficacy_from_doc(doc),
                    "content": doc.page_content,
                    "priority": 3  # 3순위
                })
        
        return efficacy_based_medicines
    
    def _calculate_efficacy_similarity(self, usage_context: str, doc) -> float:
        """효능 기반 유사도 계산"""
        efficacy = self._extract_efficacy_from_doc(doc)
        if efficacy == "정보 없음":
            return 0.0
        
        # 간단한 키워드 매칭 (향후 LLM 기반으로 개선 가능)
        context_keywords = self._extract_keywords_from_context(usage_context)
        efficacy_keywords = self._extract_keywords_from_efficacy(efficacy)
        
        if not context_keywords or not efficacy_keywords:
            return 0.0
        
        # 교집합 계산
        common_keywords = set(context_keywords) & set(efficacy_keywords)
        union_keywords = set(context_keywords) | set(efficacy_keywords)
        
        if not union_keywords:
            return 0.0
        
        return len(common_keywords) / len(union_keywords)
    
    def _extract_keywords_from_context(self, usage_context: str) -> List[str]:
        """사용 맥락에서 키워드 추출"""
        # 간단한 키워드 매핑
        keyword_mapping = {
            "두통": ["두통", "머리", "편두통", "통증"],
            "감기": ["감기", "몸살", "인후통", "기침", "콧물", "발열"],
            "치통": ["치통", "치아", "잇몸", "통증"],
            "생리통": ["생리통", "월경통", "생리", "통증"],
            "근육통": ["근육통", "어깨", "요통", "목", "통증"],
            "관절통": ["관절통", "무릎", "관절염", "통증"],
            "발열": ["발열", "열", "고열", "해열"],
            "소화불량": ["소화불량", "속쓰림", "위장", "소화"],
            "상처": ["상처", "외상", "염증", "치유"],
            "습진": ["습진", "피부염", "발진", "가려움", "아토피"]
        }
        
        for key, keywords in keyword_mapping.items():
            if key in usage_context:
                return keywords
        
        return [usage_context]
    
    def _extract_keywords_from_efficacy(self, efficacy: str) -> List[str]:
        """효능에서 키워드 추출"""
        # 간단한 키워드 추출 (향후 더 정교하게 개선 가능)
        keywords = []
        efficacy_lower = efficacy.lower()
        
        if "두통" in efficacy_lower or "머리" in efficacy_lower:
            keywords.append("두통")
        if "감기" in efficacy_lower or "몸살" in efficacy_lower:
            keywords.append("감기")
        if "통증" in efficacy_lower:
            keywords.append("통증")
        if "해열" in efficacy_lower or "열" in efficacy_lower:
            keywords.append("발열")
        if "소화" in efficacy_lower or "위장" in efficacy_lower:
            keywords.append("소화불량")
        if "피부" in efficacy_lower or "습진" in efficacy_lower:
            keywords.append("습진")
        
        return keywords if keywords else [efficacy]
    
    def _search_youtube_info(self, medicine_name: str, usage_context: str, ingredients: List[str]) -> Dict:
        """YouTube에서 약품/성분 관련 실전 정보 수집 (범용화)"""
        print(f"📺 YouTube 정보 수집: {medicine_name}")
        
        youtube_result = {
            'medicine_videos': [],
            'ingredient_videos': [],
            'usage_videos': [],
            'total_videos': 0,
            'has_transcript_count': 0
        }
        
        try:
            # 1. 약품명 직접 검색 (기본 정보) - 더 많은 검색어
            search_queries = [
                f"{medicine_name} 효능 효과",
                f"{medicine_name} 사용법",
                f"{medicine_name} 약사 설명",
                f"{medicine_name} 자세히",
                f"{medicine_name} 리뷰"
            ]
            
            # 2. 성분명 검색 (더 깊은 정보) - 3개로 증가
            for ingredient in ingredients[:3]:  # 상위 3개 성분
                search_queries.extend([
                    f"{ingredient} 성분 설명",
                    f"{ingredient} 작용기전",
                    f"{ingredient} 효능",
                    f"{ingredient} 효과"
                ])
            
            # 3. 사용 맥락 검색 - 더 구체적으로
            if usage_context:
                search_queries.extend([
                    f"{medicine_name} {usage_context}",
                    f"{medicine_name} {usage_context} 효과"
                ])
            
            # 🆕 4. 부가 정보 검색 (실사용 팁, 주의사항, 경험담) - 대폭 증가
            search_queries.extend([
                f"{medicine_name} 복용 팁",
                f"{medicine_name} 주의사항",
                f"{medicine_name} 실제 효과",
                f"{medicine_name} 먹는 법",
                f"{medicine_name} 장점",
                f"{medicine_name} 차이",
                f"{medicine_name} 언제"
            ])
            
            # 성분 부가 정보 - 2개로 증가
            for ingredient in ingredients[:2]:  # 대표 성분 2개
                search_queries.extend([
                    f"{ingredient} 부작용",
                    f"{ingredient} 복용법",
                    f"{ingredient} 효과"
                ])
            
            print(f"  검색어 목록: 총 {len(search_queries)}개")
            print(f"  주요 검색어: {search_queries[:5]}...")
            
            all_videos = []
            
            # 각 검색어로 YouTube 검색 - 10개로 증가, 영상당 개수도 증가
            for query in search_queries[:10]:  # 상위 10개 검색어
                try:
                    videos = search_youtube_videos(query, max_videos=8)  # 8개로 증가
                    
                    for video in videos:
                        # 자막 추출 시도
                        transcript = get_video_transcript(video["video_id"])
                        
                        if transcript:
                            # 자막이 있으면 요약 (길이 증가)
                            summary = summarize_video_content(transcript, max_length=800)  # 800자로 증가
                            video['transcript'] = transcript
                            video['summary'] = summary
                            video['has_transcript'] = True
                            youtube_result['has_transcript_count'] += 1
                        else:
                            # 자막 없으면 제목+설명만 (더 길게)
                            video['transcript'] = ''
                            video['summary'] = f"{video['title']} - {video['description'][:300]}"
                            video['has_transcript'] = False
                        
                        video['search_query'] = query
                        all_videos.append(video)
                        
                except Exception as e:
                    print(f"  ⚠️ '{query}' 검색 실패: {e}")
                    continue
            
            # 중복 제거 (video_id 기준)
            unique_videos = {}
            for video in all_videos:
                vid = video["video_id"]
                if vid not in unique_videos:
                    unique_videos[vid] = video
            
            # 분류
            medicine_videos = []
            ingredient_videos = []
            usage_videos = []
            
            for video in unique_videos.values():
                query = video.get('search_query', '')
                if medicine_name in query:
                    medicine_videos.append(video)
                elif any(ing in query for ing in ingredients):
                    ingredient_videos.append(video)
                elif usage_context in query:
                    usage_videos.append(video)
                else:
                    medicine_videos.append(video)  # 기본은 약품 정보
            
                youtube_result['medicine_videos'] = medicine_videos[:10]  # 10개로 증가
                youtube_result['ingredient_videos'] = ingredient_videos[:8]  # 8개로 증가
                youtube_result['usage_videos'] = usage_videos[:5]  # 5개로 증가
                youtube_result['total_videos'] = len(unique_videos)
            
            print(f"  ✅ YouTube 정보 수집 완료:")
            print(f"     - 약품 영상: {len(medicine_videos)}개")
            print(f"     - 성분 영상: {len(ingredient_videos)}개")
            print(f"     - 사용법 영상: {len(usage_videos)}개")
            print(f"     - 자막 있음: {youtube_result['has_transcript_count']}개")
            
        except Exception as e:
            print(f"  ❌ YouTube 검색 오류: {e}")
        
        return youtube_result
    
    def _search_naver_news_info(self, medicine_name: str, ingredients: List[str]) -> Dict:
        """네이버 뉴스에서 약품 관련 추가 정보 수집 (신제품, 트렌드 등)"""
        print(f"📰 네이버 뉴스 정보 수집: {medicine_name}")
        
        try:
            # 네이버 뉴스 API로 추가 정보 검색 (개수 증가)
            news_result = self.naver_news_api.search_medicine_additional_info(
                medicine_name=medicine_name,
                ingredients=ingredients,
                max_results=30  # 30개로 증가
            )
            
            return news_result
            
        except Exception as e:
            print(f"❌ 네이버 뉴스 검색 오류: {e}")
            return {
                "medicine_news": [],
                "product_news": [],
                "ingredient_news": [],
                "trend_news": [],
                "total_count": 0
            }
    
    def _format_naver_news_info(self, naver_news_result: Dict) -> str:
        """네이버 뉴스 정보 포맷팅 (추가 정보 중심)"""
        if not naver_news_result or naver_news_result.get('total_count', 0) == 0:
            return "최신 뉴스 정보 없음"
        
        formatted = []
        formatted.append(f"총 {naver_news_result['total_count']}건의 관련 뉴스 발견")
        
        # 신제품 정보 (가장 중요!) - 더 많이, 더 길게
        product_news = naver_news_result.get('product_news', [])
        if product_news:
            formatted.append("\n🆕 신제품 & 출시 소식:")
            for i, news in enumerate(product_news[:5], 1):  # 5개로 증가
                formatted.append(f"  {i}. {news['title']}")
                formatted.append(f"     {news['description'][:400]}...")  # 400자로 증가
                formatted.append(f"     ({news['pub_date_parsed']})")
        
        # 약품 일반 뉴스 (더 많이, 더 길게)
        medicine_news = naver_news_result.get('medicine_news', [])
        if medicine_news:
            formatted.append("\n📰 관련 뉴스:")
            for i, news in enumerate(medicine_news[:6], 1):  # 6개로 증가
                formatted.append(f"  {i}. {news['title']}")
                formatted.append(f"     {news['description'][:300]}...")  # 300자로 증가
        
        # 트렌드 & 연구 정보 (더 많이, 더 길게)
        trend_news = naver_news_result.get('trend_news', [])
        if trend_news:
            formatted.append("\n📈 트렌드 & 연구:")
            for i, news in enumerate(trend_news[:5], 1):  # 5개로 증가
                formatted.append(f"  {i}. {news['title']}")
                formatted.append(f"     {news['description'][:300]}...")  # 300자로 증가
        
        # 성분 관련 뉴스 (더 많이, 더 길게)
        ingredient_news = naver_news_result.get('ingredient_news', [])
        if ingredient_news:
            formatted.append("\n🧪 성분 관련:")
            for i, news in enumerate(ingredient_news[:4], 1):  # 4개로 증가
                formatted.append(f"  {i}. {news['title']}")
                formatted.append(f"     {news['description'][:250]}...")  # 설명 추가
        
        return "\n".join(formatted) if formatted else "최신 뉴스 정보 없음"
