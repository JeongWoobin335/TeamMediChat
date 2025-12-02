# enhanced_rag_system.py - 통합 RAG 시스템

import time
import json
import os
import re
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from qa_state import QAState
from retrievers import (
    excel_docs, pdf_structured_docs, 
    extract_active_ingredients_from_medicine,
    get_medicine_dosage_warnings,
    llm
)
from pubchem_api import PubChemAPI
from translation_rag import TranslationRAG
from answer_utils import generate_response_llm_from_prompt
from cache_manager import cache_manager
from config import PromptConfig
from prompt_utils import (
    get_role_definition, get_common_instructions, get_section_structure,
    get_medical_consultation_footer, get_source_mention_examples
)

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
    
    def analyze_medicine_comprehensively(self, medicine_name: str, usage_context: str, merged_medicine_info: Optional[Dict] = None) -> Dict:
        """약품 종합 분석 - 진정한 RAG 구현 (YouTube 통합)
        
        Args:
            medicine_name: 약품명
            usage_context: 사용 맥락
            merged_medicine_info: 병합된 약품 정보 (PDF 포함, 선택적)
        """
        # 종합 약품 분석 시작
        
        analysis_result = {
            'medicine_name': medicine_name,
            'usage_context': usage_context,
            'excel_info': {},
            'pdf_info': {},
            'korean_ingredient_info': {},
            'international_ingredient_info': {},
            'dosage_warning_info': {},  # ✅ 용량주의 성분 정보 추가
            'youtube_info': {},  # ✅ YouTube 정보 추가
            'naver_news_info': {},  # ✅ 네이버 뉴스 정보 추가
            'combined_analysis': {},
            'evidence_based_response': '',
            'follow_up_questions': [],
            'analysis_timestamp': time.time()
        }
        
        try:
            # 1단계: Excel DB에서 기본 약품 정보 수집
            # 병합된 정보가 있으면 우선 사용, 없으면 직접 수집
            if merged_medicine_info:
                print(f"📋 병합된 약품 정보 사용: {medicine_name}")
                # 병합된 정보를 excel_info 형식으로 변환
                excel_info = {
                    'product_name': merged_medicine_info.get('제품명', medicine_name),
                    'main_ingredient': merged_medicine_info.get('주성분', '정보 없음'),
                    'content': f"[제품명]: {merged_medicine_info.get('제품명', medicine_name)}\n"
                              f"[주성분]: {merged_medicine_info.get('주성분', '정보 없음')}\n"
                              f"[효능]: {merged_medicine_info.get('효능', '정보 없음')}\n"
                              f"[부작용]: {merged_medicine_info.get('부작용', '정보 없음')}\n"
                              f"[사용법]: {merged_medicine_info.get('사용법', '정보 없음')}",
                    'efficacy': merged_medicine_info.get('효능', '정보 없음'),
                    'side_effects': merged_medicine_info.get('부작용', '정보 없음'),
                    'usage': merged_medicine_info.get('사용법', '정보 없음')
                }
            else:
                # 1단계: Excel DB에서 기본 정보 수집
                excel_info = self._get_excel_medicine_info(medicine_name)
            analysis_result['excel_info'] = excel_info
            
            # 2단계: PDF DB 검색 제거 (Excel DB만 사용)
            # 2단계: PDF DB 검색 제거 (Excel DB만 사용)
            analysis_result['pdf_info'] = {}
            
            # 3단계: 주성분 추출
            # 3단계: 주성분 추출
            active_ingredients = self._extract_active_ingredients(medicine_name, excel_info)
            # 추출된 주성분
            
            # 3.5단계: 용량주의 성분 정보 수집
            # 3.5단계: 용량주의 성분 정보 수집
            dosage_warnings = get_medicine_dosage_warnings(medicine_name)
            analysis_result['dosage_warning_info'] = {
                'warnings': dosage_warnings,
                'has_warnings': len(dosage_warnings) > 0,
                'warning_count': len(dosage_warnings)
            }
            # 용량주의 성분 발견
            
            # 3.6단계: 연령대 금기 성분 정보 수집
            from retrievers import get_medicine_age_contraindications
            age_contraindications = get_medicine_age_contraindications(medicine_name)
            analysis_result['age_contraindication_info'] = {
                'contraindications': age_contraindications,
                'has_contraindications': len(age_contraindications) > 0,
                'contraindication_count': len(age_contraindications)
            }
            
            # 3.7단계: 일일 최대 투여량 정보 수집
            from retrievers import get_medicine_daily_max_dosage
            daily_max_dosage = get_medicine_daily_max_dosage(medicine_name)
            analysis_result['daily_max_dosage_info'] = {
                'dosage_infos': daily_max_dosage,
                'has_dosage_info': len(daily_max_dosage) > 0,
                'dosage_info_count': len(daily_max_dosage)
            }
            
            # 4단계: 각 주성분에 대한 상세 분석 (병렬 처리)
            korean_ingredient_info = {}
            international_ingredient_info = {}
            
            # PubChem 정보 수집을 병렬로 처리하는 헬퍼 함수
            def process_ingredient(ingredient: str) -> tuple:
                """주성분 정보를 수집하고 번역하는 함수 (병렬 처리용)"""
                try:
                    # PubChem에서 국제 정보 수집
                    international_info = self.pubchem_api.analyze_ingredient_comprehensive(ingredient)
                    
                    # 번역 RAG로 영어 정보를 한국어로 번역
                    translated_info = self.translation_rag.translate_pharmacology_info(international_info)
                    
                    return (ingredient, {
                        'original': international_info,
                        'translated': translated_info
                    })
                except Exception as e:
                    print(f"⚠️ 성분 {ingredient} 처리 중 오류: {e}")
                    return (ingredient, {
                        'original': {},
                        'translated': {}
                    })
            
            # 주성분 정보 수집을 병렬로 실행
            if active_ingredients:
                print(f"🔄 {len(active_ingredients)}개 성분 정보 병렬 수집 중...")
                with ThreadPoolExecutor(max_workers=min(len(active_ingredients), 5)) as executor:
                    # 모든 성분에 대해 병렬로 PubChem 정보 수집 및 번역
                    future_to_ingredient = {
                        executor.submit(process_ingredient, ingredient): ingredient 
                        for ingredient in active_ingredients
                    }
                    
                    for future in as_completed(future_to_ingredient):
                        ingredient, info = future.result()
                        international_ingredient_info[ingredient] = info
                        print(f"✅ 성분 {ingredient} 정보 수집 완료")
            
            analysis_result['korean_ingredient_info'] = korean_ingredient_info
            analysis_result['international_ingredient_info'] = international_ingredient_info
            
            # 4.5-4.6단계: YouTube, 네이버 뉴스, PubChem을 병렬로 수집
            print("🔄 외부 API 병렬 수집 시작 (YouTube, 네이버 뉴스)...")
            
            def collect_youtube_info():
                """YouTube 정보 수집 (병렬 처리용)"""
                try:
                    return self._search_youtube_info(medicine_name, usage_context, active_ingredients)
                except Exception as e:
                    print(f"⚠️ YouTube 정보 수집 오류: {e}")
                    return {
                        'medicine_videos': [],
                        'ingredient_videos': [],
                        'usage_videos': [],
                        'total_videos': 0,
                        'has_transcript_count': 0
                    }
            
            def collect_naver_news_info():
                """네이버 뉴스 정보 수집 (병렬 처리용)"""
                try:
                    return self._search_naver_news_info(medicine_name, active_ingredients)
                except Exception as e:
                    print(f"⚠️ 네이버 뉴스 정보 수집 오류: {e}")
                    return {
                        "medicine_news": [],
                        "product_news": [],
                        "ingredient_news": [],
                        "trend_news": [],
                        "total_count": 0
                    }
            
            # YouTube와 네이버 뉴스를 동시에 실행
            with ThreadPoolExecutor(max_workers=2) as executor:
                youtube_future = executor.submit(collect_youtube_info)
                naver_news_future = executor.submit(collect_naver_news_info)
                
                # 결과 대기
                youtube_info = youtube_future.result()
                naver_news_info = naver_news_future.result()
            
            analysis_result['youtube_info'] = youtube_info
            analysis_result['naver_news_info'] = naver_news_info
            print("✅ 외부 API 병렬 수집 완료")
            
            # 5단계: LLM이 모든 정보를 조합하여 근거 있는 분석 수행
            # 5단계: LLM 종합 분석
            combined_analysis = self._perform_llm_analysis(
                medicine_name, usage_context, analysis_result
            )
            analysis_result['combined_analysis'] = combined_analysis
            
            # 6단계: 근거 기반 답변 생성
            # 6단계: 근거 기반 답변 생성
            evidence_based_response = self._generate_evidence_based_response(
                medicine_name, usage_context, analysis_result
            )
            analysis_result['evidence_based_response'] = evidence_based_response
            
            # 7단계: 추가 질문 생성
            # 7단계: 추가 질문 생성
            follow_up_questions = self._generate_follow_up_questions(analysis_result)
            analysis_result['follow_up_questions'] = follow_up_questions
            
            # 종합 분석 완료
            
        except Exception as e:
            print(f"❌ 종합 분석 오류: {e}")
            analysis_result['error'] = str(e)
        
        return analysis_result
    
    def _get_excel_medicine_info(self, medicine_name: str) -> Dict:
        """Excel DB에서 약품 정보 수집 (여러 파일에서 모두 수집하여 병합) - 🚀 성능 최적화: 인덱스 활용"""
        # 🚀 성능 최적화: excel_product_index 사용 (전체 순회 대신)
        from retrievers import excel_product_index
        
        matched_docs = []
        
        # 인덱스에서 직접 가져오기 (O(1) 조회)
        if medicine_name in excel_product_index:
            matched_docs = excel_product_index[medicine_name]
        else:
            # 인덱스에 없으면 전체 순회 (폴백)
            for doc in excel_docs:
                if doc.metadata.get("제품명") == medicine_name:
                    matched_docs.append(doc)
        
        # 정확한 매칭이 없으면 부분 매칭 시도 (수출명 문제 해결)
        if not matched_docs:
            for doc in excel_docs:
                product_name = doc.metadata.get("제품명", "")
                # 약품명이 제품명의 시작 부분과 일치하는지 확인
                if product_name.startswith(medicine_name) or medicine_name in product_name:
                    matched_docs.append(doc)
        
        if not matched_docs:
            return {}
        
        print(f"📊 Enhanced RAG: {len(matched_docs)}개 문서에서 정보 수집 중...")
        
        # 파일별로 그룹화
        docs_by_file = {}
        for doc in matched_docs:
            excel_file = doc.metadata.get("excel_file")
            if excel_file:
                if excel_file not in docs_by_file:
                    docs_by_file[excel_file] = []
                docs_by_file[excel_file].append(doc)
        
        # 각 파일에서 정보 수집
        all_efficacy = []
        all_side_effects = []
        all_usage = []
        main_ingredient = ""
        product_name = medicine_name
        
        import re
        url_pattern = r'https?://[^\s]+'
        
        for excel_file, file_docs in docs_by_file.items():
            file_name = os.path.basename(excel_file)
            file_efficacy = None
            file_side_effects = None
            file_usage = None
            
            for doc in file_docs:
                content = doc.page_content
                doc_type = doc.metadata.get("type", "")
                
                # 제품명과 주성분은 첫 번째 문서에서 가져오기
                if not product_name or product_name == medicine_name:
                    product_name = doc.metadata.get("제품명", medicine_name)
                if not main_ingredient:
                    main_ingredient = doc.metadata.get("주성분", "정보 없음")
                
                # 효능과 부작용은 main 타입에서 추출
                if doc_type == "main" or doc_type == "":
                    efficacy_match = re.search(r'\[효능\]:\s*((?:.|\n)*?)(?=\n\[|\Z)', content)
                    side_effects_match = re.search(r'\[부작용\]:\s*((?:.|\n)*?)(?=\n\[|\Z)', content)
                    
                    if efficacy_match:
                        efficacy = efficacy_match.group(1).strip()
                        if efficacy != "정보 없음" and not re.search(url_pattern, efficacy):
                            if file_efficacy is None or len(efficacy) > len(file_efficacy or ""):
                                file_efficacy = efficacy
                    
                    if side_effects_match:
                        side_effects = side_effects_match.group(1).strip()
                        if side_effects != "정보 없음" and not re.search(url_pattern, side_effects):
                            if file_side_effects is None or len(side_effects) > len(file_side_effects or ""):
                                file_side_effects = side_effects
                
                # 사용법은 usage 타입에서 추출
                if doc_type == "usage":
                    usage_match = re.search(r'\[사용법\]:\s*((?:.|\n)*?)(?=\n\[|\Z)', content)
                    if usage_match:
                        usage = usage_match.group(1).strip()
                        if usage != "정보 없음" and not re.search(url_pattern, usage):
                            if file_usage is None or len(usage) > len(file_usage or ""):
                                file_usage = usage
            
            # 파일별 정보 수집
            if file_efficacy:
                all_efficacy.append((file_name, file_efficacy))
            if file_side_effects:
                all_side_effects.append((file_name, file_side_effects))
            if file_usage:
                all_usage.append((file_name, file_usage))
        
        # 여러 소스의 정보를 병합
        from medicine_usage_check_node import merge_multiple_sources_with_llm
        
        final_efficacy = "정보 없음"
        if len(all_efficacy) > 1:
            final_efficacy = merge_multiple_sources_with_llm(all_efficacy, "효능")
        elif len(all_efficacy) == 1:
            final_efficacy = all_efficacy[0][1]
        
        final_side_effects = "정보 없음"
        if len(all_side_effects) > 1:
            final_side_effects = merge_multiple_sources_with_llm(all_side_effects, "부작용")
        elif len(all_side_effects) == 1:
            final_side_effects = all_side_effects[0][1]
        
        final_usage = "정보 없음"
        if len(all_usage) > 1:
            final_usage = merge_multiple_sources_with_llm(all_usage, "사용법")
        elif len(all_usage) == 1:
            final_usage = all_usage[0][1]
        
        # 통합 content 생성
        content = f"""[제품명]: {product_name}
[주성분]: {main_ingredient}
[효능]: {final_efficacy}
[부작용]: {final_side_effects}
[사용법]: {final_usage}"""
        
        return {
            'product_name': product_name,
            'main_ingredient': main_ingredient,
            'efficacy': final_efficacy,
            'side_effects': final_side_effects,
            'usage': final_usage,
            'content': content,
            'metadata': matched_docs[0].metadata if matched_docs else {}
        }
    
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
            # 주성분 추출
            
            # 쉼표로 구분된 성분들을 개별적으로 분리
            if ',' in main_ingredient:
                ingredients = [ing.strip() for ing in main_ingredient.split(',') if ing.strip()]
                # 분리된 성분들
            else:
                ingredients = [main_ingredient.strip()]
                # 단일 성분
        else:
            pass  # 주성분 정보 없음
        
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
            'dosage_warning_info': analysis_result.get('dosage_warning_info', {}),  # ✅ 용량주의 성분 정보 추가
            'age_contraindication_info': analysis_result.get('age_contraindication_info', {}),  # ✅ 연령대 금기 성분 정보 추가
            'daily_max_dosage_info': analysis_result.get('daily_max_dosage_info', {}),  # ✅ 일일 최대 투여량 정보 추가
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
        
        # ✅ 용량주의 성분 정보 요약
        dosage_warning_summary = self._format_dosage_warning_info(analysis_result.get('dosage_warning_info', {}))
        
        # ✅ 연령대 금기 성분 정보 요약
        age_contraindication_summary = self._format_age_contraindication_info(analysis_result.get('age_contraindication_info', {}))
        
        # ✅ 일일 최대 투여량 정보 요약
        daily_max_dosage_summary = self._format_daily_max_dosage_info(analysis_result.get('daily_max_dosage_info', {}))
        
        analysis_prompt = f"""당신은 다중 소스 의약품 정보 통합 전문가입니다. 여러 소스의 정보를 종합하여 근거 있는 분석을 제공하세요.

## 🎯 분석 목표
- 약품: {medicine_name}
- 사용 목적: {usage_context}

## 📚 수집된 정보 (다중 소스)

### 소스 1: 한국 의약품 정보 DB - Excel (신뢰도: 높음, 여러 파일에서 병합된 정보)
**중요**: 이 정보는 여러 Excel 파일(OpenData_ItemPermitDetail20251115.xls, e약은요정보검색1-5.xlsx)에서 수집하여 병합한 것입니다. 각 파일의 고유한 정보가 모두 포함되어 있습니다.
{json.dumps(collected_info['excel_info'], indent=2, ensure_ascii=False)}

### 소스 2: 국제 성분 DB (PubChem, 신뢰도: 높음)
**중요**: 이 정보는 Excel DB의 주성분 정보를 기반으로 PubChem에서 수집한 국제 표준 약리학 데이터입니다.
{json.dumps(translated_summaries, indent=2, ensure_ascii=False)}

### 소스 3: 전문가 의견 & 실사용 경험 (신뢰도: 중간~높음)
{youtube_summary}

### 소스 4: 최신 뉴스 & 추가 정보 (신뢰도: 중간, 참고용)
{naver_news_summary}

### 소스 5: 용량주의 성분 정보 (신뢰도: 높음, 식약처 공고 기준)
{dosage_warning_summary}

### 소스 6: 연령대 금기 성분 정보 (신뢰도: 높음, 식약처 공고 기준)
{age_contraindication_summary}

### 소스 7: 일일 최대 투여량 정보 (신뢰도: 높음, 식약처 공고 기준, 정 개수 단위)
{daily_max_dosage_summary}

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
- 사용 가능: 연관성 ≥ 50% + 안전성 경미~보통 + 용량주의 성분 없음
- 사용 불가: 연관성 < 50% 또는 안전성 심각 또는 용량주의 성분 있음

**⚠️ 용량주의 성분 특별 고려사항:**
- 용량주의 성분이 포함된 약품은 반드시 의사/약사 처방 필요
- 용량주의 성분이 있으면 사용 가능성과 관계없이 처방 의무 강조

**⚠️ 연령대 금기 성분 특별 고려사항:**
- 연령대 금기 성분이 포함된 약품은 해당 연령대에서 사용 금지 또는 주의 필요
- 연령대 금기 성분이 있으면 사용 가능성과 관계없이 해당 연령대에서의 사용 금지 또는 주의사항 강조

**📊 일일 최대 투여량 특별 고려사항:**
- 일일 최대 투여량 정보가 있으면 반드시 주의사항에 포함
- 특히 정(개수) 단위의 경우 개수를 명확히 명시하여 초과 복용 방지
- 1일 최대용량 정보를 반드시 답변에 포함
- **단일 성분 vs 복합 성분 구분:**
  * 단일 성분: 해당 성분만의 용량을 체크하면 됨 (예: 아세트아미노펜 단독)
  * 복합 성분: 복합제 전체의 용량을 고려해야 하며, 다른 성분과의 상호작용도 주의 필요
  * 복합 성분인 경우 답변에서 "이 성분은 복합제로 사용될 수 있으며, 다른 성분과 함께 복용 시 용량을 더욱 주의해야 합니다"라고 명시

**신뢰도 레벨:**
- high: 모든 소스 일치 + 명확한 과학적 근거
- medium: 일부 소스만 또는 간접적 근거
- low: 정보 부족 또는 모순 존재

## 💡 분석 예시

### 예시 1: 타이레놀 (아세트아미노펜) + 두통 (경구제)
STEP 1: Excel(높음), PubChem(높음) 신뢰
STEP 2: 모순 없음 - 모두 "두통 완화" 명시
STEP 3: COX 억제 → 통증 감소, 연관성 95%, 부작용 경미
STEP 4: 사용 가능 (신뢰도: high)
**표현**: "타이레놀은 두통에 먹어도 됩니다"

### 예시 2: 마데카솔 연고 + 상처 (외용제)
STEP 1: Excel(높음) 신뢰
STEP 2: 모순 없음 - 모두 "상처 치유" 명시
STEP 3: 항염증(피부용) → 상처 치유, 연관성 90%, 부작용 경미
STEP 4: 사용 가능 (신뢰도: high)
**표현**: "마데카솔 연고는 상처에 발라도 됩니다"

### 예시 3: 습진 연고 + 근육통 (외용제, 사용 불가)
STEP 1: Excel(높음) 신뢰
STEP 2: 모순 없음
STEP 3: 항염증(피부용) → 근육통 무관, 연관성 5%, 부작용 보통
STEP 4: 사용 불가 (신뢰도: high)
**표현**: "습진 연고는 근육통에 발라면 안 됩니다"

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
    "dosage_warning_info": {{
        "has_dosage_warning": true/false,
        "warning_ingredients": ["성분1", "성분2"],
        "max_daily_doses": ["성분1: 용량", "성분2: 용량"],
        "prescription_required": true/false,
    }},
    "age_contraindication_info": {{
        "has_age_contraindication": true/false,
        "contraindication_ingredients": ["성분1", "성분2"],
        "age_restrictions": ["연령기준1: 금기내용", "연령기준2: 금기내용"],
        "age_restriction_required": true/false,
    }},
    "daily_max_dosage_info": {{
        "has_dosage_info": true/false,
        "dosage_ingredients": ["성분1", "성분2"],
        "max_daily_dosages": ["성분1: 제형/단위/용량", "성분2: 제형/단위/용량"],
        "dosage_unit_type": "정(개수)/기타",
        "single_ingredients": ["단일 성분 목록"],
        "complex_ingredients": ["복합 성분 목록"],
        "complex_details": {{"복합성분명": "복합제 정보"}}
    }},
    "evidence_summary": "판단 근거 요약 (어느 소스에서 어떤 정보 활용했는지 명시)",
    "alternative_suggestions": ["대안1", "대안2"],
    "expert_recommendation": "최종 전문가 권고사항 (용량주의 성분이 있으면 반드시 처방 필요성 강조)"
}}

**중요 지침:**
- 반드시 STEP 1-4 순서로 사고하세요
- mechanism_analysis는 구체적 메커니즘 필수 (예: "COX-2 억제", "세로토닌 재흡수 차단")
- ⚠️ 용량주의 성분이 있으면 반드시 답변에 포함하고 처방 필요성 강조
- 추측 금지 - 주어진 정보만 사용
- 모순 발견 시 신뢰도 높은 소스 우선
- 불확실하면 confidence_level 낮추고 이유 명시
- **모든 소스의 정보를 종합적으로 활용하세요. 각 소스마다 약간씩 다른 정보가 있을 수 있으므로, 모든 고유한 정보를 포함하여 분석하세요.**
"""
        
        try:
            # 캐시 확인
            cached_response = cache_manager.get_llm_response_cache(analysis_prompt, "combined_analysis")
            if cached_response:
                response_content = cached_response
            else:
                response = self.llm.invoke(analysis_prompt)
                response_content = response.content if hasattr(response, 'content') else str(response)
                # 캐시 저장
                cache_manager.save_llm_response_cache(analysis_prompt, response_content, "combined_analysis")
            
            # JSON 응답 파싱
            try:
                if "```json" in response_content:
                    json_start = response_content.find("```json") + 7
                    json_end = response_content.find("```", json_start)
                    if json_end != -1:
                        json_str = response_content[json_start:json_end].strip()
                    else:
                        json_str = response_content[json_start:].strip()
                else:
                    json_str = response_content.strip()
                
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
        dosage_warning_info = analysis_result.get('dosage_warning_info', {})  # ✅ 용량주의 성분 정보 추가
        youtube_info = analysis_result.get('youtube_info', {})
        naver_news_info = analysis_result.get('naver_news_info', {})  # ✅ 네이버 뉴스 정보 추가
        combined_analysis = analysis_result.get('combined_analysis', {})
        
        # 동적 대안 약품 검색
        # 동적 대안 약품 검색 중
        alternative_medicines = self._find_similar_medicines_dynamically(medicine_name, usage_context, excel_info)
        # 발견된 대안 약품
        
        # 디버깅: 용량주의 정보 확인
        dosage_warning_formatted = self._format_dosage_warning_info(dosage_warning_info)
        
        # 연령대 금기 정보 포맷팅
        age_contraindication_info = analysis_result.get('age_contraindication_info', {})
        age_contraindication_formatted = self._format_age_contraindication_info(age_contraindication_info)
        
        # 일일 최대 투여량 정보 포맷팅
        daily_max_dosage_info = analysis_result.get('daily_max_dosage_info', {})
        daily_max_dosage_formatted = self._format_daily_max_dosage_info(daily_max_dosage_info)
        
        # LLM에게 자연스러운 답변 생성 요청
        # 병합된 정보가 있으면 더 상세하게 표시
        excel_content = excel_info.get('content', '정보 없음')
        if excel_info.get('efficacy') and excel_info.get('efficacy') != '정보 없음':
            # 병합된 상세 정보가 있는 경우 - 개별 필드로 명확하게 표시
            excel_content = f"""[제품명]: {excel_info.get('product_name', medicine_name)}
[주성분]: {excel_info.get('main_ingredient', '정보 없음')}

[효능]: 
{excel_info.get('efficacy', '정보 없음')}

[부작용 및 주의사항]: 
{excel_info.get('side_effects', '정보 없음')}

[사용법]: 
{excel_info.get('usage', '정보 없음')}"""
        
        # 디버깅: Excel 정보 확인
        efficacy_len = len(str(excel_info.get('efficacy', '')))
        side_effects_len = len(str(excel_info.get('side_effects', '')))
        print(f"🔍 프롬프트에 전달되는 Excel 정보 - 효능: {efficacy_len}자, 부작용: {side_effects_len}자")
        if side_effects_len > 500:
            print(f"📋 부작용 정보 미리보기 (처음 200자): {str(excel_info.get('side_effects', ''))[:200]}...")
        
        # 디버깅: 네이버 뉴스 정보 확인
        naver_news_formatted = self._format_naver_news_info(naver_news_info)
        naver_news_len = len(naver_news_formatted)
        print(f"🔍 프롬프트에 전달되는 네이버 뉴스 정보 - 길이: {naver_news_len}자")
        if naver_news_len > 100:
            print(f"📰 네이버 뉴스 정보 미리보기 (처음 300자): {naver_news_formatted[:300]}...")
        else:
            print(f"⚠️ 네이버 뉴스 정보가 너무 짧거나 없습니다: '{naver_news_formatted}'")
        
        # 주성분 목록 추출 (네이버 뉴스 필터링용)
        active_ingredients = list(international_info.keys()) if international_info else []
        if not active_ingredients:
            # excel_info에서 주성분 추출 시도
            main_ingredient = excel_info.get('main_ingredient', '')
            if main_ingredient and main_ingredient != '정보 없음':
                if ',' in main_ingredient:
                    active_ingredients = [ing.strip() for ing in main_ingredient.split(',') if ing.strip()]
                else:
                    active_ingredients = [main_ingredient.strip()]
        
        active_ingredients_str = ', '.join(active_ingredients) if active_ingredients else '없음'
        
        # 🚀 약품 타입 판단 (경구/외용) - 동적 표현 생성
        def determine_medicine_type(med_name: str, usage_ctx: str) -> dict:
            """약품 타입을 판단하여 적절한 표현 반환"""
            import re
            # 외용제 제형 키워드
            topical_forms = ['연고', '크림', '젤', '로션', '밴드', '파스', '플라스터', '패치']
            # 경구제 제형 키워드
            oral_forms = ['정', '캡슐', '시럽', '액', '분말', '가루', '정제']
            
            # 사용 맥락에서 외용 키워드 확인
            topical_keywords = ['발라', '도포', '바르', '상처에', '습진에', '피부에', '외용']
            oral_keywords = ['먹어', '복용', '섭취', '경구']
            
            # 약품명에서 제형 확인
            is_topical_by_name = any(form in med_name for form in topical_forms)
            is_oral_by_name = any(form in med_name for form in oral_forms)
            
            # 사용 맥락에서 키워드 확인
            is_topical_by_context = any(keyword in usage_ctx for keyword in topical_keywords)
            is_oral_by_context = any(keyword in usage_ctx for keyword in oral_keywords)
            
            # 판단 로직
            if is_topical_by_name or is_topical_by_context:
                return {
                    'type': 'topical',
                    'question': '발라도 되나?',
                    'positive': '발라도 됩니다',
                    'negative': '발라면 안 됩니다',
                    'caution': '주의해서 발라야 합니다',
                    'usage_verb': '도포',
                    'usage_noun': '도포'
                }
            elif is_oral_by_name or is_oral_by_context:
                return {
                    'type': 'oral',
                    'question': '먹어도 되나?',
                    'positive': '먹어도 됩니다',
                    'negative': '먹으면 안 됩니다',
                    'caution': '주의해서 먹어야 합니다',
                    'usage_verb': '복용',
                    'usage_noun': '복용'
                }
            else:
                # 기본값은 경구 (대부분의 약품이 경구제이므로)
                return {
                    'type': 'oral',
                    'question': '먹어도 되나?',
                    'positive': '먹어도 됩니다',
                    'negative': '먹으면 안 됩니다',
                    'caution': '주의해서 먹어야 합니다',
                    'usage_verb': '복용',
                    'usage_noun': '복용'
                }
        
        # 약품 타입 판단
        medicine_type_info = determine_medicine_type(medicine_name, usage_context)
        
        prompt = f"""
{get_role_definition("pharmacist")} 사용자의 질문에 대해 아래 모든 정보 소스를 종합하여 답변해주세요.

**사용자 질문:** {medicine_name}은(는) {usage_context}에 {medicine_type_info['question']}

**📋 정보 소스:**

## STEP 1: Excel DB 정보 (공식 약품 정보)

**Excel DB 정보:**
{excel_content}

## STEP 2: PubChem 국제 정보
{self._format_international_info(international_info)}

## STEP 3: YouTube 실전 정보 (수집된 정보만 사용)
{self._format_youtube_info(youtube_info)}

## STEP 4: 네이버 뉴스 정보 (약품명/주성분 관련만)
주성분: {active_ingredients_str}
{self._format_naver_news_info(naver_news_info)}

## STEP 5: 용량주의 성분 정보
{dosage_warning_formatted}

## STEP 6: 연령대 금기 성분 정보
{age_contraindication_formatted}

## STEP 7: 일일 최대 투여량 정보
{daily_max_dosage_formatted}

## STEP 8: 종합 분석 결과
- 사용 가능성: {combined_analysis.get('safe_to_use', 'Unknown')}
- 작용기전: {combined_analysis.get('mechanism_analysis', '정보 없음')}
- 안전성: {combined_analysis.get('safety_assessment', '정보 없음')}
- 주의사항: {combined_analysis.get('precautions', [])}
- 금기사항: {combined_analysis.get('contraindications', [])}
- 대안: {combined_analysis.get('alternative_suggestions', [])}

## STEP 9: 대안 약품
{self._format_alternative_medicines(alternative_medicines)}

**답변 요구사항:**
1. 모든 소스의 정보를 빠짐없이 포함
2. {PromptConfig.COMMON_INSTRUCTIONS['natural_tone']}
3. {get_section_structure()}
4. {PromptConfig.COMMON_INSTRUCTIONS['no_source_mention']}
5. 구체적인 작용기전 포함
6. 용량주의/연령대 금기/일일 최대 투여량 정보 있으면 주의사항에 명시
7. 최소 {PromptConfig.MIN_ANSWER_LENGTH}자 이상 작성

**중요 지침:**
{get_common_instructions()}
- 모든 정보를 **하나의 통합된 지식**처럼 자연스럽게 설명
{get_source_mention_examples()}
- 뉴스 정보는 "참고로..." 또는 "💡 알아두면 좋은 정보" 섹션에 자연스럽게 추가
- ⚠️ **용량주의 성분이 있으면 반드시 주의사항에 포함하고 처방 필요성 강조**
- ⚠️ **연령대 금기 성분이 있으면 해당 연령대에서의 사용 금지 또는 주의사항 명시**
- 📊 **일일 최대 투여량 정보가 있으면 주의사항에 포함 (정 개수 단위의 경우 개수 명시)**

**⚠️ 매우 중요: 약품 타입에 따른 표현 사용**
- 이 약품은 **{medicine_type_info['type']}제**입니다.
- 경구제(정, 캡슐 등)인 경우: "먹어도 됩니다", "먹으면 안 됩니다", "복용"
- 외용제(연고, 크림 등)인 경우: "발라도 됩니다", "발라면 안 됩니다", "도포"
- 사용 맥락에 "발라도", "상처에", "습진에" 등이 있으면 외용제로 판단

**답변 구조 (이모지로 섹션 구분):**
0. **✅ 먼저 질문에 대한 직접적인 답변 (가장 중요!):**
   - 반드시 답변의 맨 처음에 "{medicine_name}은(는) {usage_context}에 [{medicine_type_info['positive']}/{medicine_type_info['negative']}/{medicine_type_info['caution']}]" 같은 명확한 판단을 제시하세요
   - 사용 가능성 정보(safe_to_use)를 바탕으로 "네, {medicine_type_info['positive']}", "아니요, {medicine_type_info['negative']}", "{medicine_type_info['caution']}" 중 하나를 명확히 답변하세요
   - 간단한 이유도 함께 제시하세요 (예: "네, {medicine_type_info['positive']}. 메가본정은 근육통 완화에 효과적이지만, 용량주의 성분이 포함되어 있어 주의가 필요합니다." 또는 "네, 발라도 됩니다. 마데카솔 연고는 상처 치유에 효과적입니다.")
1. **💊 효능 및 작용 원리**: 
   - 위 정보에서 수집한 모든 효능 정보를 포함
   - 각 주성분의 구체적인 작용 메커니즘을 설명
   - 해당 증상에 어떤 효과가 있는지 구체적으로 설명
2. **⚠️ 주의사항 및 부작용**: 
   - 위 정보에서 수집한 모든 부작용 및 주의사항 정보를 포함 (요약하지 말고 상세 내용 포함)
   - 용량주의 성분이 있으면 반드시 포함하고 처방 필요성 강조, 구체적인 용량 정보 명시
   - 연령대 금기 성분이 있으면 해당 연령대에서의 사용 금지 또는 주의사항 명시
   - 일일 최대 투여량 정보가 있으면 주의사항에 포함 (정 개수 단위의 경우 개수 명시)
3. **📋 사용법**: 
   - 위 정보에서 수집한 모든 사용법 정보를 포함
4. **💡 대안 약품**: 
   - 위에서 제공된 동적 대안 약품 분석 결과를 바탕으로 구체적인 대안 약품 제안 (실제 약품명만 사용)
5. **💡 추가 정보**: 
   - **⚠️ 이 섹션은 오직 수집된 YouTube 영상과 네이버 뉴스에서 실제로 확인된 사실적인 내용만 포함합니다.**
   - **절대 일반적인 약리학 지식이나 의학 상식을 포함하지 마세요.**
   - **수집된 정보가 없거나 적으면 "추가 정보 없음" 또는 "수집된 추가 정보가 제한적입니다"라고만 작성하세요.**
   
   **포함 가능한 내용 (반드시 수집된 정보에 기반):**
   - 질문한 약품({medicine_name})이 명시적으로 언급된 뉴스에서 확인된 신제품 출시 정보 - **뉴스 제목, 발행일, 구체적인 내용을 명시**
   - 질문한 약품({medicine_name})의 리뉴얼이나 성분 변경 정보 - **뉴스 제목, 발행일, 구체적인 변경 사항을 명시**
   - 질문한 약품의 주성분({active_ingredients_str}) 중 하나 이상이 언급된 뉴스 - **뉴스 제목, 발행일, 구체적인 내용을 명시**
   - 주성분({active_ingredients_str}) 중 하나를 포함한 다른 약품에 대한 뉴스 (같은 성분을 설명하는 것이므로 포함 가능) - **뉴스 제목, 발행일, 구체적인 내용을 명시**
   - 질문한 약품({medicine_name})과 직접 관련된 최신 연구 결과 - **연구 제목, 발행일, 구체적인 연구 내용을 명시**
   - 질문한 약품({medicine_name})에 대한 YouTube 영상에서 확인된 실전 팁 - **영상 제목, 채널명, 구체적인 팁 내용을 명시**
   - 질문한 약품({medicine_name})과 직접 관련된 트렌드나 시장 동향 - **뉴스/영상 제목, 발행일, 구체적인 내용을 명시**
   - 질문한 약품({medicine_name})에 대한 사용자 경험담이나 후기 - **영상/뉴스 제목, 구체적인 경험 내용을 명시**
   
   **⚠️ 절대 포함하지 말 것:**
   - 질문한 약품의 주성분({active_ingredients_str}) 목록에 포함되지 않은 성분이 언급된 뉴스 (예: DL-메틸에페드린염산염은 주성분이 아니므로 제외)
   - 질문한 약품({medicine_name})과 무관하고 주성분도 언급되지 않은 다른 약품명이 언급된 뉴스
   - 질문한 약품({medicine_name})과 직접 관련이 없고 주성분도 언급되지 않은 일반적인 감기약 시장 동향
   - 일반적인 약리학 지식 (예: "비타민 B6는 신경 기능을 개선합니다", "카페인은 불안을 유발할 수 있습니다")
   - 일반적인 의학 상식
   - 수집된 정보에 없는 내용을 일반화한 표현
   - "최근 뉴스에 따르면", "최근 연구에서는" 같은 모호한 표현 (반드시 구체적인 뉴스/영상 제목과 내용을 언급)
   
   **작성 원칙:**
   - **⚠️ 매우 중요: 수집된 YouTube 영상과 네이버 뉴스 정보를 빠짐없이 모두 포함하세요. 절대 생략하거나 요약하지 마세요.**
   - **⚠️ 특히 네이버 뉴스 정보가 있으면 반드시 "💡 추가 정보" 섹션에 포함하세요. 뉴스가 1건이라도 반드시 언급하세요.**
   - 각 정보의 출처(뉴스 제목, 영상 제목, 발행일 등)를 명시하세요
   - 각 뉴스/영상의 구체적인 내용을 요약하지 말고 가능한 한 상세히 설명하세요
   - 위의 정보들을 자연스럽게 여러 문단으로 작성하되, 수집된 정보가 많으면 더 많은 문단으로 작성하세요
   - **수집된 네이버 뉴스가 20건이면 20건 모두 언급하고, YouTube 영상이 50개면 가능한 한 많은 영상을 언급하세요**
   - **수집된 정보가 많을수록 더 상세하고 긴 섹션을 작성하세요**
   - **네이버 뉴스 정보가 "최신 뉴스 정보 없음"이 아니면 반드시 답변에 포함하세요**
6. **마무리**: {get_medical_consultation_footer("friendly").strip()}

**⚠️ 답변 길이 요구사항:**
- 답변은 최소 {PromptConfig.MIN_ANSWER_LENGTH}자 이상 작성하세요
- 모든 중요한 정보를 빠짐없이 포함하세요
- 각 섹션을 충분히 상세하게 작성하세요 (섹션당 최소 {PromptConfig.MIN_SECTION_LENGTH}자)
- 요약하지 말고 필요한 모든 세부사항을 포함하세요

**중요 지침:**
- 작용기전 설명 시 "중추신경계에서 프로스타글란딘 합성을 억제하여..." 같은 구체적인 메커니즘 포함
- 단순히 "통증을 줄이고 열을 내린다"가 아닌 "어떻게" 작용하는지 설명
- 모든 약품에 대해 동일한 수준의 상세함을 유지
- 의학적으로 정확하면서도 이해하기 쉽게 설명
- **대안 약품 제시 시 반드시 위에서 제공된 동적 대안 약품 분석 결과만 사용**
- **이부프로펜, 나프록센 같은 성분명을 대안으로 제시하지 말고, 실제 약품명(포펜정, 타이레놀 등)만 사용**

{PromptConfig.COMMON_INSTRUCTIONS['natural_tone']}으로 답변해주세요.
"""
        
        try:
            # 캐시 확인 (최종 답변은 캐싱하지 않음 - 매번 다른 답변이 필요할 수 있음)
            # 하지만 같은 입력이면 같은 답변이 나와야 하므로 캐싱 적용
            cached_response = cache_manager.get_llm_response_cache(prompt, "final_answer")
            if cached_response:
                return cached_response
            
            response = self.llm.invoke(prompt)
            result = response.content.strip()
            
            # 캐시 저장
            cache_manager.save_llm_response_cache(prompt, result, "final_answer")
            return result
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
        
        # 🚀 성능 최적화: 표시할 영상 수 및 길이 감소 (품질 영향 최소)
        medicine_videos = youtube_info.get('medicine_videos', [])
        if medicine_videos:
            formatted.append("\n💊 약품 관련 실전 정보:")
            for i, video in enumerate(medicine_videos[:5], 1):  # 12개 → 5개
                formatted.append(f"  {i}. {video['title']}")
                if video.get('has_transcript'):
                    formatted.append(f"     핵심 내용: {video.get('summary', '')[:600]}...")  # 1200자 → 600자
                else:
                    formatted.append(f"     개요: {video.get('description', '')[:300]}...")  # 600자 → 300자
        
        # 성분 관련 정보
        ingredient_videos = youtube_info.get('ingredient_videos', [])
        if ingredient_videos:
            formatted.append("\n🧪 성분 관련 전문 정보:")
            for i, video in enumerate(ingredient_videos[:4], 1):  # 10개 → 4개
                formatted.append(f"  {i}. {video['title']}")
                if video.get('has_transcript'):
                    formatted.append(f"     핵심 내용: {video.get('summary', '')[:600]}...")  # 1200자 → 600자
                else:
                    formatted.append(f"     개요: {video.get('description', '')[:300]}...")  # 600자 → 300자
        
        # 사용법 관련 정보
        usage_videos = youtube_info.get('usage_videos', [])
        if usage_videos:
            formatted.append("\n💡 사용법 및 팁:")
            for i, video in enumerate(usage_videos[:3], 1):  # 8개 → 3개
                formatted.append(f"  {i}. {video['title']}")
                if video.get('has_transcript'):
                    formatted.append(f"     핵심 내용: {video.get('summary', '')[:600]}...")  # 1200자 → 600자
                else:
                    formatted.append(f"     개요: {video.get('description', '')[:300]}...")  # 600자 → 300자
        
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
        
        response += get_medical_consultation_footer("standard").strip()
        
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
        # 동적 유사 약품 검색
        
        # 대상 약품의 주성분 추출
        target_ingredients = self._extract_ingredients_from_excel_info(excel_info)
        # 대상 약품 주성분
        
        # 1단계: 동일 성분 약품 검색 (최고 우선순위)
        same_ingredient_medicines = self._find_medicines_with_same_ingredients(medicine_name, target_ingredients)
        # 동일 성분 약품
        
        # 2단계: 유사 성분 약품 검색 (2순위)
        similar_ingredient_medicines = self._find_medicines_with_similar_ingredients(medicine_name, target_ingredients)
        # 유사 성분 약품
        
        # 3단계: 효능 기반 약품 검색 (3순위)
        efficacy_based_medicines = self._find_medicines_by_efficacy(medicine_name, usage_context, target_ingredients)
        # 효능 기반 약품
        
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
        # YouTube 정보 수집
        
        youtube_result = {
            'medicine_videos': [],
            'ingredient_videos': [],
            'usage_videos': [],
            'total_videos': 0,
            'has_transcript_count': 0
        }
        
        try:
            # 🚀 성능 최적화: 검색어 수 감소 (품질 영향 최소)
            search_queries = [
                f"{medicine_name}",  # 기본 약품명만
            ]
            
            # 2. 성분명 검색 - 상위 2개 성분만 (3개 → 2개)
            for ingredient in ingredients[:2]:
                search_queries.append(f"{ingredient}")
            
            # 3. 사용 맥락 검색 (선택적)
            if usage_context and len(search_queries) < 3:  # 최대 3개 검색어
                search_queries.append(f"{medicine_name} {usage_context}")
            
            # 검색어 목록
            
            all_videos = []
            
            # YouTube 검색을 병렬로 처리하는 헬퍼 함수
            def process_youtube_query(query: str) -> List[Dict]:
                """YouTube 검색어 처리 (병렬 처리용)"""
                try:
                    # 🚀 성능 최적화: 검색 결과 수 감소 (15개 → 8개)
                    videos = search_youtube_videos(query, max_videos=8)
                    
                    # 자막 추출도 병렬로 처리
                    def process_video(video: Dict) -> Dict:
                        """비디오 자막 추출 및 요약 (병렬 처리용)"""
                        try:
                            transcript = get_video_transcript(video["video_id"])
                            
                            if transcript:
                                # 자막이 있으면 요약
                                summary = summarize_video_content(transcript, max_length=800)
                                video['transcript'] = transcript
                                video['summary'] = summary
                                video['has_transcript'] = True
                            else:
                                # 자막 없으면 제목+설명만
                                video['transcript'] = ''
                                video['summary'] = f"{video['title']} - {video['description'][:300]}"
                                video['has_transcript'] = False
                            
                            video['search_query'] = query
                            return video
                        except Exception as e:
                            print(f"⚠️ 비디오 {video.get('video_id', 'unknown')} 처리 오류: {e}")
                            video['transcript'] = ''
                            video['summary'] = f"{video['title']} - {video['description'][:300]}"
                            video['has_transcript'] = False
                            video['search_query'] = query
                            return video
                    
                    # 🚀 성능 최적화: 병렬 처리 워커 수 감소 (10개 → 5개)
                    processed_videos = []
                    transcript_count = 0
                    if videos:
                        with ThreadPoolExecutor(max_workers=min(len(videos), 5)) as executor:
                            future_to_video = {
                                executor.submit(process_video, video): video 
                                for video in videos
                            }
                            
                            for future in as_completed(future_to_video):
                                processed_video = future.result()
                                processed_videos.append(processed_video)
                                if processed_video.get('has_transcript'):
                                    transcript_count += 1
                    else:
                        processed_videos = videos
                    
                    return (processed_videos, transcript_count)
                    
                except Exception as e:
                    print(f"⚠️ YouTube 검색어 '{query}' 처리 오류: {e}")
                    return ([], 0)
            
            # 🚀 성능 최적화: 검색어 수 제한 (10개 → 3개)
            search_queries_limited = search_queries[:3]
            print(f"🔄 {len(search_queries_limited)}개 YouTube 검색어 병렬 처리 중...")
            with ThreadPoolExecutor(max_workers=min(len(search_queries_limited), 3)) as executor:
                future_to_query = {
                    executor.submit(process_youtube_query, query): query 
                    for query in search_queries_limited
                }
                
                for future in as_completed(future_to_query):
                    videos, count = future.result()
                    all_videos.extend(videos)
                    youtube_result['has_transcript_count'] += count
            
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
            
                # 🚀 성능 최적화: 결과 수 감소 (품질 영향 최소)
                youtube_result['medicine_videos'] = medicine_videos[:6]  # 10개 → 6개
                youtube_result['ingredient_videos'] = ingredient_videos[:5]  # 8개 → 5개
                youtube_result['usage_videos'] = usage_videos[:3]  # 5개 → 3개
                youtube_result['total_videos'] = len(unique_videos)
            
            # YouTube 정보 수집 완료
            
        except Exception as e:
            print(f"  ❌ YouTube 검색 오류: {e}")
        
        return youtube_result
    
    def _search_naver_news_info(self, medicine_name: str, ingredients: List[str]) -> Dict:
        """네이버 뉴스에서 약품 관련 추가 정보 수집 (신제품, 트렌드 등)"""
        # 네이버 뉴스 정보 수집
        
        try:
            # 🚀 성능 최적화: 검색 결과 수 감소 (품질 영향 최소)
            news_result = self.naver_news_api.search_medicine_additional_info(
                medicine_name=medicine_name,
                ingredients=ingredients,
                max_results=20  # 50개 → 20개로 감소 (품질 영향 최소)
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
        if not naver_news_result:
            return "최신 뉴스 정보 없음"
        
        # 모든 뉴스를 하나의 리스트로 통합
        all_news = []
        all_news.extend(naver_news_result.get('product_news', []))
        all_news.extend(naver_news_result.get('medicine_news', []))
        all_news.extend(naver_news_result.get('trend_news', []))
        all_news.extend(naver_news_result.get('ingredient_news', []))
        
        total_count = len(all_news)
        
        if total_count == 0:
            return "최신 뉴스 정보 없음"
        
        formatted = []
        formatted.append(f"총 {total_count}건의 관련 뉴스 발견")
        
        # 🚀 성능 최적화: 표시할 뉴스 수 및 길이 감소 (품질 영향 최소)
        product_news = naver_news_result.get('product_news', [])
        if product_news:
            formatted.append("\n🆕 신제품 & 출시 소식:")
            for i, news in enumerate(product_news[:5], 1):  # 10개 → 5개
                formatted.append(f"  {i}. {news['title']}")
                formatted.append(f"     {news['description'][:500]}...")  # 1000자 → 500자
                formatted.append(f"     발행일: {news.get('pub_date_parsed', news.get('pub_date', '날짜 정보 없음'))}")
        
        # 약품 일반 뉴스
        medicine_news = naver_news_result.get('medicine_news', [])
        if medicine_news:
            formatted.append("\n📰 관련 뉴스:")
            for i, news in enumerate(medicine_news[:5], 1):  # 12개 → 5개
                formatted.append(f"  {i}. {news['title']}")
                formatted.append(f"     {news['description'][:400]}...")  # 800자 → 400자
                formatted.append(f"     발행일: {news.get('pub_date_parsed', news.get('pub_date', '날짜 정보 없음'))}")
        
        # 트렌드 & 연구 정보
        trend_news = naver_news_result.get('trend_news', [])
        if trend_news:
            formatted.append("\n📈 트렌드 & 연구:")
            for i, news in enumerate(trend_news[:4], 1):  # 10개 → 4개
                formatted.append(f"  {i}. {news['title']}")
                formatted.append(f"     {news['description'][:400]}...")  # 700자 → 400자
                formatted.append(f"     발행일: {news.get('pub_date_parsed', news.get('pub_date', '날짜 정보 없음'))}")
        
        # 성분 관련 뉴스
        ingredient_news = naver_news_result.get('ingredient_news', [])
        if ingredient_news:
            formatted.append("\n🧪 성분 관련:")
            for i, news in enumerate(ingredient_news[:4], 1):  # 8개 → 4개
                formatted.append(f"  {i}. {news['title']}")
                formatted.append(f"     {news['description'][:400]}...")  # 700자 → 400자
                formatted.append(f"     발행일: {news.get('pub_date_parsed', news.get('pub_date', '날짜 정보 없음'))}")
        
        # 모든 뉴스가 비어있지만 total_count가 있는 경우 (카테고리 분류 문제)
        if not formatted or len(formatted) == 1:  # "총 X건"만 있는 경우
            formatted.append("\n📰 수집된 뉴스 목록:")
            for i, news in enumerate(all_news[:20], 1):  # 최대 20개
                formatted.append(f"  {i}. {news.get('title', '제목 없음')}")
                formatted.append(f"     {news.get('description', '내용 없음')[:500]}...")
                formatted.append(f"     발행일: {news.get('pub_date_parsed', news.get('pub_date', '날짜 정보 없음'))}")
        
        return "\n".join(formatted) if formatted else "최신 뉴스 정보 없음"
    
    def _format_dosage_warning_info(self, dosage_warning_info: Dict) -> str:
        """용량주의 성분 정보를 포맷팅 (단일/복합 구분 포함)"""
        if not dosage_warning_info or not dosage_warning_info.get('has_warnings', False):
            return "용량주의 성분 없음"
        
        warnings = dosage_warning_info.get('warnings', [])
        if not warnings:
            return "용량주의 성분 없음"
        
        formatted = ["⚠️ 용량주의 성분 발견:"]
        
        # 단일 성분과 복합 성분을 구분하여 표시
        single_ingredients = []
        complex_ingredients = []
        
        for warning in warnings:
            ingredient = warning.get('ingredient', '')
            dosage_info = warning.get('dosage_info', {})
            single_complex = dosage_info.get('single_complex', '')
            complex_medicine = dosage_info.get('complex_medicine', '')
            max_dose = dosage_info.get('max_daily_dose', '정보 없음')
            
            if single_complex == '복합' and complex_medicine:
                # 복합 성분인 경우
                complex_ingredients.append({
                    'ingredient': ingredient,
                    'max_dose': max_dose,
                    'complex_medicine': complex_medicine
                })
            else:
                # 단일 성분인 경우
                single_ingredients.append({
                    'ingredient': ingredient,
                    'max_dose': max_dose
                })
        
        # 단일 성분 정보 표시
        if single_ingredients:
            formatted.append("\n【단일 성분】")
            for item in single_ingredients:
                formatted.append(f"  - {item['ingredient']}: 1일 최대용량 {item['max_dose']}")
        
        # 복합 성분 정보 표시
        if complex_ingredients:
            formatted.append("\n【복합 성분】")
            for item in complex_ingredients:
                formatted.append(f"  - {item['ingredient']}: 1일 최대용량 {item['max_dose']}")
                if item['complex_medicine']:
                    # 복합제 정보에서 한글명 추출
                    korean_match = re.search(r'\(([가-힣]+)\)', item['complex_medicine'])
                    if korean_match:
                        complex_name = korean_match.group(1)
                        formatted.append(f"    (복합제 구성: {complex_name})")
                    else:
                        formatted.append(f"    (복합제: {item['complex_medicine']})")
        
        formatted.append("\n⚠️ 중요: 용량주의 성분이 포함된 약품은 반드시 의사나 약사의 처방에 따라 사용하세요.")
        formatted.append("   - 단일 성분: 해당 성분의 용량만 체크하면 됩니다.")
        formatted.append("   - 복합 성분: 복합제 전체의 용량을 고려해야 하며, 다른 성분과의 상호작용도 주의해야 합니다.")
        
        return "\n".join(formatted)
    
    def _format_age_contraindication_info(self, age_contraindication_info: Dict) -> str:
        """연령대 금기 성분 정보를 포맷팅"""
        if not age_contraindication_info or not age_contraindication_info.get('has_contraindications', False):
            return "연령대 금기 성분 없음"
        
        contraindications = age_contraindication_info.get('contraindications', [])
        if not contraindications:
            return "연령대 금기 성분 없음"
        
        formatted = ["⚠️ 연령대 금기 성분 발견:"]
        
        for contra in contraindications:
            ingredient = contra.get('ingredient', '')
            age_info = contra.get('age_contraindication_info', {})
            age_contraindications = age_info.get('age_contraindications', [])
            
            if age_contraindications:
                formatted.append(f"\n【{ingredient}】")
                for age_contra in age_contraindications:
                    age_criteria = age_contra.get('age_criteria', '정보 없음')
                    contraindication = age_contra.get('contraindication', '정보 없음')
                    formatted.append(f"  - 연령기준: {age_criteria}")
                    formatted.append(f"    금기내용: {contraindication}")
        
        formatted.append("\n⚠️ 중요: 연령대 금기 성분이 포함된 약품은 해당 연령대에서 사용 시 반드시 의사나 약사와 상담하세요.")
        
        return "\n".join(formatted)
    
    def _format_daily_max_dosage_info(self, daily_max_dosage_info: Dict) -> str:
        """일일 최대 투여량 정보를 포맷팅"""
        if not daily_max_dosage_info or not daily_max_dosage_info.get('has_dosage_info', False):
            return "일일 최대 투여량 정보 없음"
        
        dosage_infos = daily_max_dosage_info.get('dosage_infos', [])
        if not dosage_infos:
            return "일일 최대 투여량 정보 없음"
        
        formatted = ["📊 일일 최대 투여량 정보:"]
        
        for dosage_info in dosage_infos:
            ingredient = dosage_info.get('ingredient', '')
            formulation = dosage_info.get('formulation', '')
            dosage_unit = dosage_info.get('dosage_unit', '')
            max_daily_dosage = dosage_info.get('max_daily_dosage', '정보 없음')
            all_formulations = dosage_info.get('all_formulations', [])
            
            formatted.append(f"\n【{ingredient}】")
            if formulation:
                formatted.append(f"  - 제형: {formulation}")
            if dosage_unit:
                formatted.append(f"  - 투여단위: {dosage_unit}")
            formatted.append(f"  - 1일 최대 투여량: {max_daily_dosage}")
            
            # 여러 제형 정보가 있는 경우
            if all_formulations and len(all_formulations) > 1:
                formatted.append(f"  - 기타 제형 정보:")
                for form_info in all_formulations[1:]:  # 첫 번째는 이미 표시했으므로
                    form_formulation = form_info.get('formulation', '')
                    form_unit = form_info.get('dosage_unit', '')
                    form_max = form_info.get('max_daily_dosage', '')
                    if form_formulation or form_unit or form_max:
                        formatted.append(f"    • {form_formulation} ({form_unit}): {form_max}")
        
        formatted.append("\n⚠️ 중요: 일일 최대 투여량을 초과하지 않도록 주의하세요. 특히 정(개수) 단위의 약품은 개수를 정확히 확인하세요.")
        
        return "\n".join(formatted)
