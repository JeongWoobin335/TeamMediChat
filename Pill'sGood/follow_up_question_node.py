# follow_up_question_node.py - 연속 질문 처리 노드

from qa_state import QAState
from retrievers import llm, excel_docs, find_products_by_ingredient
from entity_classifier import classify_medicine_vs_ingredient, extract_target_from_query
from config import PromptConfig
from prompt_utils import (
    get_role_definition, get_common_instructions, get_source_mention_examples,
    get_medical_consultation_footer
)
from typing import Dict, List, Optional
import re
import json

# YouTube 검색 함수 import
from sns_node import search_youtube_videos, get_video_transcript, summarize_video_content

def search_youtube_for_followup(target: str, intent_type: str) -> List[Dict]:
    """연속 질문용 YouTube 검색 (의도에 맞게)"""
    try:
        # 의도에 따라 검색어 생성 (단순하고 명료하게)
        if intent_type == "ingredient_info":
            search_queries = [
                f"{target}",  # 성분명만
            ]
        elif intent_type == "usage_info":
            search_queries = [
                f"{target}",  # 약품명만
            ]
        elif intent_type == "side_effect":
            search_queries = [
                f"{target} 부작용",  # 부작용은 명시적으로 검색
            ]
        else:
            search_queries = [
                f"{target}",  # 기본은 단순하게
            ]
        
        collected_videos = []
        
        # 각 검색어로 검색
        for query in search_queries:
            try:
                videos = search_youtube_videos(query, max_videos=4)
                
                for video in videos:
                    # 자막 추출
                    transcript = get_video_transcript(video["video_id"])
                    
                    if transcript:
                        summary = summarize_video_content(transcript, max_length=400)
                        video['transcript'] = transcript
                        video['summary'] = summary
                        video['has_transcript'] = True
                    else:
                        video['transcript'] = ''
                        video['summary'] = f"{video['title']} - {video.get('description', '')[:150]}"
                        video['has_transcript'] = False
                    
                    collected_videos.append(video)
                    
            except Exception as e:
                print(f"  ⚠️ '{query}' 검색 실패: {e}")
                continue
        
        # 중복 제거 (video_id 기준)
        unique_videos = {}
        for video in collected_videos:
            vid = video["video_id"]
            if vid not in unique_videos:
                unique_videos[vid] = video
        
        return list(unique_videos.values())[:5]  # 최대 5개로 증가
        
    except Exception as e:
        print(f"❌ YouTube 검색 오류: {e}")
        return []

def follow_up_question_node(state: QAState) -> QAState:
    """연속 질문 처리 노드 - 이전 답변의 맥락을 활용한 추가 질문 처리"""
    
    follow_up_type = state.get("follow_up_type", "")
    conversation_context = state.get("conversation_context", "")
    current_query = state.get("query", "")
    
    print(f"🔍 연속 질문 처리 시작: {follow_up_type}")
    print(f"🔍 현재 질문: {current_query}")
    
    try:
        # 🔍 현재 질문을 LLM이 직접 분석하여 의도 파악
        answer = analyze_and_respond_to_followup(current_query, conversation_context, follow_up_type)
        
        if answer:
            state["final_answer"] = answer
            print(f"✅ 연속 질문 처리 완료: {follow_up_type}")
            return state
        
        # LLM 분석이 실패한 경우 기존 방식으로 fallback
        print("⚠️ LLM 분석 실패, 기존 방식으로 처리")
        
        # 디버깅: state에 저장된 값 확인
        extracted_ingredient = state.get("extracted_ingredient_name")
        extracted_medicine = state.get("extracted_medicine_name")
        medicine_name_from_state = state.get("medicine_name")
        print(f"🔍 state 확인 - extracted_ingredient_name: {extracted_ingredient}")
        print(f"🔍 state 확인 - extracted_medicine_name: {extracted_medicine}")
        print(f"🔍 state 확인 - medicine_name: {medicine_name_from_state}")
        
        # 먼저 state에 이미 저장된 약품명/성분명 확인 (question_refinement_node에서 추출한 값)
        medicine_name = extracted_ingredient or extracted_medicine or medicine_name_from_state
        
        if medicine_name:
            print(f"✅ state에서 약품명/성분명 발견: {medicine_name}")
        else:
            # 이전 대화에서 언급된 약품명 추출
            medicine_name = extract_medicine_from_context(conversation_context)
            
            # 대화 맥락에서 찾지 못했다면 사용자 질문에서 직접 추출 시도
            if not medicine_name:
                medicine_name = extract_medicine_from_user_question(current_query)
        
        if not medicine_name:
            state["final_answer"] = "아, 어떤 약품에 대해 궁금하신지 명확하지 않네요! 약품명을 다시 말씀해 주시면 도움을 드릴게요!"
            return state
            
        print(f"🔍 추출된 약품명: {medicine_name}")
        
        # 연속 질문 유형에 따른 처리
        if follow_up_type == "usage":
            answer = handle_usage_question(medicine_name, conversation_context)
        elif follow_up_type == "ingredient":
            # 현재 질문을 포함한 컨텍스트 전달
            full_context = f"{conversation_context}\n사용자: {current_query}" if current_query else conversation_context
            answer = handle_ingredient_question(medicine_name, full_context)
        elif follow_up_type == "side_effect":
            answer = handle_side_effect_question(medicine_name, conversation_context)
        elif follow_up_type == "mechanism":
            answer = handle_mechanism_question(medicine_name, conversation_context)
        elif follow_up_type == "precaution":
            answer = handle_precaution_question(medicine_name, conversation_context)
        elif follow_up_type == "alternative_medicines":
            answer = handle_alternative_medicines_question(medicine_name, conversation_context, current_query)
        elif follow_up_type == "similar_medicines":
            answer = handle_similar_medicines_question(medicine_name, conversation_context, current_query)
        else:
            answer = handle_general_question(medicine_name, conversation_context, current_query)
        
        state["final_answer"] = answer
        print(f"✅ 연속 질문 처리 완료: {follow_up_type}")
        
    except Exception as e:
        print(f"❌ 연속 질문 처리 오류: {e}")
        state["final_answer"] = f"죄송합니다. 추가 질문을 처리하는 중 오류가 발생했습니다: {str(e)}"
    
    return state

def extract_usage_context_from_query(current_query: str, conversation_context: str) -> str:
    """질문과 대화 맥락에서 사용 목적을 지능적으로 추출"""
    
    context_prompt = f"""
다음 질문과 대화 맥락에서 약품의 사용 목적이나 증상을 파악해주세요.

**이전 대화:**
{conversation_context[:600]}

**현재 질문:**
{current_query}

**분석 요구사항:**
1. 질문에서 언급된 증상이나 상황 파악
2. 이전 대화에서 언급된 증상이나 상황 고려
3. 구체적인 사용 목적 추출

**가능한 사용 맥락 예시:**
- 감기 (감기, 몸살, 인후통, 기침, 콧물 등)
- 두통 (머리아픔, 편두통 등)
- 치통 (치아 통증, 잇몸 아픔 등)
- 생리통 (월경통, 생리 등)
- 근육통 (어깨 통증, 요통, 목 아픔 등)
- 관절통 (무릎 아픔, 관절염 등)
- 발열 (열, 고열 등)
- 소화불량 (속쓰림, 위장장애 등)
- 상처 (상처, 외상, 염증 등)
- 습진 (습진, 피부염, 발진, 가려움 등등)
- 일반적 사용 (구체적 증상이 없는 경우)

한 단어나 간단한 구문으로만 응답해주세요. 예: "감기", "두통", "근육통", "일반적 사용"
"""
    
    try:
        response = llm.invoke(context_prompt)
        usage_context = response.content.strip().replace('"', '').replace("'", "")
        
        # 응답이 너무 길면 첫 번째 단어만 사용
        if len(usage_context) > 20:
            usage_context = usage_context.split()[0] if usage_context.split() else "일반적 사용"
        
        print(f"🔍 추출된 사용 맥락: '{usage_context}'")
        return usage_context if usage_context else "일반적 사용"
        
    except Exception as e:
        print(f"⚠️ 사용 맥락 추출 실패: {e}")
        return "일반적 사용"

def analyze_and_respond_to_followup(current_query: str, conversation_context: str, follow_up_type: str) -> Optional[str]:
    """연속 질문을 분석하고 실제 데이터를 조회하여 답변하는 통합 함수"""
    if not current_query or not conversation_context:
        return None
    
    print(f"🧠 연속 질문 분석 및 데이터 조회 시작")
    print(f"🔍 질문 유형: {follow_up_type}")
    
    # 1단계: 질문 의도 및 필요한 정보 분석
    intent_analysis = analyze_question_intent(current_query, conversation_context)
    if not intent_analysis:
        return None
    
    print(f"🎯 질문 의도: {intent_analysis.get('intent_type', 'unknown')}")
    print(f"🔍 대상: {intent_analysis.get('target', 'unknown')}")
    
    # 2단계: 의도에 따른 실제 데이터 수집
    collected_data = collect_relevant_data(intent_analysis, current_query, conversation_context)
    
    # 3단계: 수집된 데이터를 바탕으로 답변 생성
    if collected_data:
        answer = generate_data_driven_answer(current_query, conversation_context, collected_data, intent_analysis)
        if answer:
            print(f"✅ 데이터 기반 연속 질문 처리 완료")
            return answer
    
    print("⚠️ 데이터 수집 실패, 기본 LLM 답변으로 fallback")
    return None

def analyze_question_intent(current_query: str, conversation_context: str) -> Optional[Dict]:
    """질문 의도를 분석하여 필요한 데이터 소스와 대상을 파악"""
    
    analysis_prompt = f"""당신은 대화 맥락 분석 전문가입니다. 단계별로 분석하여 사용자 의도를 파악하세요.

## 📋 대화 정보
**이전 대화:**
{conversation_context[:800]}

**현재 질문:**
{current_query}

## 🔍 3단계 의도 분석 프로세스

### STEP 1: 질문 유형 식별
다음 중 어디에 해당하는가?
- ingredient_info: 성분 설명 요청 ("~이 뭔데?", "~이 뭐야?")
- usage_info: 사용법 질문 ("어떻게 먹어?", "사용법은?")
- side_effect: 부작용 질문 ("부작용은?", "주의사항은?")
- new_medicine: 새 약품 질문 (이전과 다른 약품명 등장)
- general: 기타

**판단 기준:**
- 새 약품: 이전 대화에 없던 약품명 등장
- 성분: "~이 뭐", "~이 뭔", "~무엇" 패턴
- 사용법: "어떻게", "사용법", "복용법"
- 부작용: "부작용", "주의사항", "위험"

### STEP 2: 대상 추출
누구/무엇에 대한 질문인가?
- 약품명 또는 성분명 추출
- 여러 대상이면 쉼표로 구분

**추출 예시:**
- "푸르설티아민이 뭔데?" → "푸르설티아민"
- "그럼 뇌선은?" → "뇌선"
- "타이레놀 사용법은?" → "타이레놀"

### STEP 3: 데이터 소스 선택
필요한 정보 소스를 선택하세요:

**기본 (항상 포함):**
- excel_db: 한국 약품 기본 정보
- youtube: 전문가 의견/경험담/추가 정보 (모든 질문에 포함)

**추가 (조건부):**
- enhanced_rag: 새 약품 종합 분석 (new_medicine일 때)
- health_kr + pubchem: 성분 상세 정보 (ingredient_info일 때)

**선택 로직:**
- new_medicine → excel_db + youtube + enhanced_rag
- ingredient_info → excel_db + youtube + health_kr + pubchem
- usage_info → excel_db + youtube
- side_effect → excel_db + youtube
- general → excel_db + youtube

## 💡 분석 예시

### 예시 1: "푸르설티아민이 뭔데?"
STEP 1: ingredient_info (성분 설명 요청)
STEP 2: 대상 = "푸르설티아민"
STEP 3: ["excel_db", "youtube", "health_kr", "pubchem"]

### 예시 2: "그럼 뇌선은 감기에 먹어도 되나?"
STEP 1: new_medicine (이전 대화와 다른 약품)
STEP 2: 대상 = "뇌선"
STEP 3: ["excel_db", "youtube", "enhanced_rag"]

### 예시 3: "그럼 부작용은?"
STEP 1: side_effect
STEP 2: 대상 = [이전 대화의 약품명]
STEP 3: ["excel_db", "youtube"]

### 예시 4: "사용법은?"
STEP 1: usage_info
STEP 2: 대상 = [이전 대화의 약품명]
STEP 3: ["excel_db", "youtube"]

## 📤 출력 형식 (JSON)
{{
    "intent_type": "ingredient_info|usage_info|side_effect|new_medicine|general",
    "target": "대상 이름",
    "data_sources": ["excel_db", ...],
    "specific_info_needed": "구체적 정보 요구사항",
    "is_new_medicine": true/false
}}

**중요:** 간결하게 판단하고, 불필요한 소스는 제외하세요.
"""
    
    try:
        response = llm.invoke(analysis_prompt)
        content = response.content.strip()
        
        # JSON 파싱
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        import json
        result = json.loads(content)
        print(f"🎯 의도 분석 결과: {result}")
        return result
        
    except Exception as e:
        print(f"❌ 질문 의도 분석 실패: {e}")
        return None

def collect_relevant_data(intent_analysis: Dict, current_query: str, conversation_context: str = "") -> Optional[Dict]:
    """의도 분석 결과에 따라 실제 데이터를 수집"""
    
    intent_type = intent_analysis.get("intent_type", "")
    target = intent_analysis.get("target", "")
    data_sources = intent_analysis.get("data_sources", [])
    
    collected_data = {}
    
    print(f"📊 데이터 수집 시작: {intent_type} - {target}")
    print(f"🔍 요청된 데이터 소스: {data_sources}")
    
    try:
        # Excel DB에서 약품 정보 수집 (모든 타입에 대해)
        if "excel_db" in data_sources and target:
            # 여러 약품명이 쉼표로 구분된 경우 개별적으로 처리
            medicine_names = [name.strip() for name in target.split(',') if name.strip()]
            print(f"📋 Excel DB에서 약품 정보 수집 중: {medicine_names}")
            
            excel_info_list = []
            for medicine_name in medicine_names:
                print(f"  개별 약품 조회: {medicine_name}")
                excel_info = find_medicine_info(medicine_name, excel_docs)
                print(f"  조회 결과: {excel_info}")
                if excel_info and excel_info.get("제품명") != "정보 없음":
                    excel_info_list.append(excel_info)
                    print(f"  ✅ {medicine_name} 정보 수집 완료")
                else:
                    print(f"  ❌ {medicine_name} 정보를 찾을 수 없음")
            
            if excel_info_list:
                collected_data["excel_info"] = excel_info_list
                print(f"✅ Excel DB 정보 수집 완료: {len(excel_info_list)}개 약품")
            else:
                print(f"❌ Excel DB에서 약품 정보를 찾을 수 없음")
        else:
            print(f"⚠️ Excel DB 조회 조건 미충족: excel_db in data_sources={('excel_db' in data_sources)}, target={bool(target)}")
        
        # Enhanced RAG 시스템 호출 (새로운 약품인 경우)
        if "enhanced_rag" in data_sources and target:
            # 여러 약품명이 쉼표로 구분된 경우 개별적으로 처리
            medicine_names = [name.strip() for name in target.split(',') if name.strip()]
            print(f"🔬 Enhanced RAG 시스템으로 약품 종합 분석 중: {medicine_names}")
            
            try:
                from enhanced_rag_system import EnhancedRAGSystem
                enhanced_rag_system = EnhancedRAGSystem()
                
                # 사용 맥락 지능적 추출
                usage_context = extract_usage_context_from_query(current_query, conversation_context)
                
                enhanced_analysis_list = []
                for medicine_name in medicine_names:
                    print(f"  개별 약품 분석: {medicine_name}")
                    enhanced_analysis = enhanced_rag_system.analyze_medicine_comprehensively(medicine_name, usage_context)
                    if enhanced_analysis:
                        enhanced_analysis_list.append(enhanced_analysis)
                        print(f"  ✅ {medicine_name} 분석 완료")
                    else:
                        print(f"  ❌ {medicine_name} 분석 실패")
                
                if enhanced_analysis_list:
                    collected_data["enhanced_rag_info"] = enhanced_analysis_list
                    print(f"✅ Enhanced RAG 종합 분석 완료: {len(enhanced_analysis_list)}개 약품")
            except Exception as e:
                print(f"⚠️ Enhanced RAG 분석 실패: {e}")
        
        # 성분 정보가 필요한 경우 외부 API 호출
        if intent_type == "ingredient_info" and target:
            print(f"🧪 성분 정보 수집: {target}")
            
            # 외부 약학정보원 스크래핑 제거 (저작권 문제, Excel DB 사용)
            
            # PubChem 정보 수집 + 번역 (핵심!)
            if "pubchem" in data_sources:
                try:
                    from pubchem_api import PubChemAPI
                    from translation_rag import TranslationRAG
                    
                    pubchem = PubChemAPI()
                    pubchem_info = pubchem.analyze_ingredient_comprehensive(target)
                    
                    if pubchem_info:
                        collected_data["pubchem_info"] = pubchem_info
                        print(f"✅ PubChem 정보 수집 완료")
                        
                        # 🆕 번역 추가 (가장 중요!)
                        print(f"🔄 PubChem 정보 번역 중...")
                        translation_rag = TranslationRAG()
                        translated_info = translation_rag.translate_pharmacology_info(
                            pubchem_info.get('pharmacology_info', {})
                        )
                        
                        if translated_info:
                            collected_data["translated_pubchem_info"] = translated_info
                            print(f"✅ PubChem 정보 번역 완료 (요약 길이: {len(translated_info.get('summary_kr', ''))}자)")
                        
                except Exception as e:
                    print(f"⚠️ PubChem 정보 수집/번역 실패: {e}")
            
            # 🆕 성분이 포함된 제품 목록 추가 (중요!)
            print(f"💊 '{target}' 성분이 포함된 제품 검색 중...")
            products_with_ingredient = find_products_by_ingredient(target)
            if products_with_ingredient:
                collected_data["products_with_ingredient"] = products_with_ingredient
                print(f"✅ 제품 {len(products_with_ingredient)}개 발견: {', '.join(products_with_ingredient[:3])}...")
            else:
                print(f"⚠️ 한국 DB에서 '{target}' 성분을 포함한 제품을 찾을 수 없음")
        
        # 🆕 YouTube 검색 (모든 질문에 대해 항상 시도)
        if target:
            print(f"📺 YouTube에서 {target} 추가 정보 검색 중...")
            try:
                youtube_videos = search_youtube_for_followup(target, intent_type)
                if youtube_videos:
                    collected_data["youtube_info"] = youtube_videos
                    print(f"✅ YouTube 영상 {len(youtube_videos)}개 수집 완료 (자막 있는 영상: {sum(1 for v in youtube_videos if v.get('has_transcript'))}개)")
                else:
                    print(f"⚠️ YouTube에서 {target} 관련 정보를 찾을 수 없음 (검색 결과 없음)")
            except Exception as e:
                print(f"⚠️ YouTube 검색 중 오류 발생: {e}")
        
        return collected_data if collected_data else None
        
    except Exception as e:
        print(f"❌ 데이터 수집 중 오류: {e}")
        return None

def generate_data_driven_answer(current_query: str, conversation_context: str, collected_data: Dict, intent_analysis: Dict) -> Optional[str]:
    """수집된 실제 데이터를 바탕으로 답변 생성 (YouTube 통합)"""
    
    intent_type = intent_analysis.get("intent_type", "")
    target = intent_analysis.get("target", "")
    
    # 수집된 데이터를 텍스트로 정리
    data_summary = ""
    
    # Enhanced RAG 정보가 있으면 우선 활용
    if "enhanced_rag_info" in collected_data:
        enhanced_info_list = collected_data["enhanced_rag_info"]
        data_summary += f"**Enhanced RAG 종합 분석:**\n"
        
        # 여러 약품 정보 처리
        if isinstance(enhanced_info_list, list):
            for i, enhanced_info in enumerate(enhanced_info_list, 1):
                data_summary += f"\n**약품 {i}:**\n"
                if enhanced_info.get('excel_info'):
                    excel_info = enhanced_info['excel_info']
                    data_summary += f"- 제품명: {excel_info.get('제품명', '정보 없음')}\n"
                    data_summary += f"- 주성분: {excel_info.get('주성분', '정보 없음')}\n"
                    data_summary += f"- 효능: {excel_info.get('효능', '정보 없음')}\n"
                    data_summary += f"- 사용법: {excel_info.get('사용법', '정보 없음')}\n"
                    data_summary += f"- 부작용: {excel_info.get('부작용', '정보 없음')}\n"
        else:
            # 단일 약품 정보 처리 (기존 로직)
            enhanced_info = enhanced_info_list
            if enhanced_info.get('excel_info'):
                excel_info = enhanced_info['excel_info']
                data_summary += f"- 제품명: {excel_info.get('제품명', '정보 없음')}\n"
                data_summary += f"- 주성분: {excel_info.get('주성분', '정보 없음')}\n"
                data_summary += f"- 효능: {excel_info.get('효능', '정보 없음')}\n"
                data_summary += f"- 사용법: {excel_info.get('사용법', '정보 없음')}\n"
                data_summary += f"- 부작용: {excel_info.get('부작용', '정보 없음')}\n"
        
        if enhanced_info.get('combined_analysis'):
            analysis = enhanced_info['combined_analysis']
            data_summary += f"- 안전성 평가: {analysis.get('safety_assessment', '정보 없음')}\n"
            data_summary += f"- 작용기전: {analysis.get('mechanism_analysis', '정보 없음')}\n"
            data_summary += f"- 전문가 권고: {analysis.get('expert_recommendation', '정보 없음')}\n"
            if analysis.get('alternative_suggestions'):
                data_summary += f"- 대안 약품: {', '.join(analysis['alternative_suggestions'])}\n"
        
        data_summary += "\n"
    else:
        # Enhanced RAG 정보가 없으면 기존 방식 사용
        if "excel_info" in collected_data:
            excel_info_list = collected_data["excel_info"]
            data_summary += f"**Excel DB 정보:**\n"
            
            # 여러 약품 정보 처리
            if isinstance(excel_info_list, list):
                for i, excel_info in enumerate(excel_info_list, 1):
                    data_summary += f"\n**약품 {i}:**\n"
                    data_summary += f"- 제품명: {excel_info.get('제품명', '정보 없음')}\n"
                    data_summary += f"- 주성분: {excel_info.get('주성분', '정보 없음')}\n"
                    data_summary += f"- 효능: {excel_info.get('효능', '정보 없음')}\n"
                    data_summary += f"- 사용법: {excel_info.get('사용법', '정보 없음')}\n"
                    data_summary += f"- 부작용: {excel_info.get('부작용', '정보 없음')}\n"
            else:
                # 단일 약품 정보 처리 (기존 로직)
                excel_info = excel_info_list
                data_summary += f"- 제품명: {excel_info.get('제품명', '정보 없음')}\n"
                data_summary += f"- 주성분: {excel_info.get('주성분', '정보 없음')}\n"
                data_summary += f"- 효능: {excel_info.get('효능', '정보 없음')}\n"
                data_summary += f"- 사용법: {excel_info.get('사용법', '정보 없음')}\n"
                data_summary += f"- 부작용: {excel_info.get('부작용', '정보 없음')}\n"
            data_summary += "\n"
        
        if "health_kr_info" in collected_data:
            health_kr_info = collected_data["health_kr_info"]
            data_summary += f"**한국 의약품 DB 정보:**\n"
            data_summary += f"- 한국명: {health_kr_info.get('korean_name', '정보 없음')}\n"
            data_summary += f"- 영문명: {health_kr_info.get('english_name', '정보 없음')}\n"
            if health_kr_info.get('mechanism_of_action'):
                data_summary += f"- 작용기전: {health_kr_info['mechanism_of_action']}\n"
            if health_kr_info.get('side_effects'):
                data_summary += f"- 부작용: {health_kr_info['side_effects']}\n"
            data_summary += "\n"
        
        if "pubchem_info" in collected_data:
            pubchem_info = collected_data["pubchem_info"]
            data_summary += f"**PubChem 상세 정보:**\n"
            
            # 성분명
            if pubchem_info.get('ingredient_name'):
                data_summary += f"- 성분명: {pubchem_info['ingredient_name']}\n"
            
            # 기본 정보
            if pubchem_info.get('basic_info'):
                basic = pubchem_info['basic_info']
                if basic.get('MolecularFormula'):
                    data_summary += f"- 분자식: {basic['MolecularFormula']}\n"
                if basic.get('MolecularWeight'):
                    data_summary += f"- 분자량: {basic['MolecularWeight']}\n"
            
            # 약리학 정보 (핵심!)
            if pubchem_info.get('pharmacology_info'):
                pharm = pubchem_info['pharmacology_info']
                if pharm.get('mechanism_of_action'):
                    data_summary += f"- 작용기전: {pharm['mechanism_of_action'][:500]}...\n"
                if pharm.get('pharmacodynamics'):
                    data_summary += f"- 약력학: {pharm['pharmacodynamics'][:500]}...\n"
                if pharm.get('atc_codes'):
                    data_summary += f"- ATC 분류: {', '.join(pharm['atc_codes'][:3])}\n"
                if pharm.get('mesh_classification'):
                    mesh_names = [m.get('name', '') for m in pharm['mesh_classification'][:3]]
                    data_summary += f"- MeSH 분류: {', '.join(mesh_names)}\n"
            
            # 설명 정보
            if pubchem_info.get('description'):
                desc = pubchem_info['description']
                data_summary += f"- 설명: {desc[:500]}...\n"
            
            data_summary += "\n"
        
        # 🆕 번역된 PubChem 정보가 있으면 최우선 활용
        if "translated_pubchem_info" in collected_data:
            translated_info = collected_data["translated_pubchem_info"]
            data_summary += f"**번역된 약리학 정보 (가장 상세):**\n"
            
            if translated_info.get('summary_kr'):
                data_summary += f"{translated_info['summary_kr']}\n\n"
            
            if translated_info.get('mechanism_of_action_kr'):
                data_summary += f"- 작용기전 (한국어): {translated_info['mechanism_of_action_kr'][:800]}\n\n"
            
            if translated_info.get('pharmacodynamics_kr'):
                data_summary += f"- 약력학 (한국어): {translated_info['pharmacodynamics_kr'][:800]}\n\n"
            
            if translated_info.get('atc_codes_kr'):
                data_summary += f"- ATC 분류 (한국어):\n"
                for atc in translated_info['atc_codes_kr'][:3]:
                    data_summary += f"  * {atc.get('code', '')}: {atc.get('korean_description', '')}\n"
                data_summary += "\n"
            
            if translated_info.get('mesh_classification_kr'):
                data_summary += f"- MeSH 약리학 분류 (한국어):\n"
                for mesh in translated_info['mesh_classification_kr'][:3]:
                    data_summary += f"  * {mesh.get('korean_name', '')}: {mesh.get('korean_description', '')}\n"
                data_summary += "\n"
        
        # 🆕 성분이 포함된 제품 목록 (매우 중요!)
        if "products_with_ingredient" in collected_data:
            products = collected_data["products_with_ingredient"]
            data_summary += f"**💊 이 성분({target})이 포함된 한국 제품들:**\n"
            if len(products) > 0:
                # 상위 10개 제품만 표시
                for i, product in enumerate(products[:10], 1):
                    data_summary += f"{i}. {product}\n"
                if len(products) > 10:
                    data_summary += f"... 외 {len(products) - 10}개 제품\n"
                data_summary += f"\n총 {len(products)}개 제품에서 사용됨\n\n"
            else:
                data_summary += "한국 DB에서 찾을 수 없음\n\n"
    
    # 🆕 추가 실전 정보 (YouTube 출처 숨김)
    if "youtube_info" in collected_data:
        youtube_videos = collected_data["youtube_info"]
        data_summary += f"**💡 전문가 의견 및 실사용 정보:**\n"
        data_summary += f"(총 {len(youtube_videos)}개 정보원 참조)\n\n"
        
        for i, video in enumerate(youtube_videos[:5], 1):  # 5개로 증가
            data_summary += f"{i}. {video.get('title', '')}\n"
            if video.get('has_transcript'):
                data_summary += f"   내용: {video.get('summary', '')[:400]}...\n"  # 내용도 더 길게
            else:
                data_summary += f"   설명: {video.get('description', '')[:250]}...\n"  # 설명도 더 길게
            data_summary += "\n"
    
    # 성분 질문 여부 확인
    is_ingredient_question = intent_type == "ingredient_info"
    has_translated_pubchem = "translated_pubchem_info" in collected_data
    
    # 데이터 기반 답변 생성 - 최적화 버전
    answer_prompt = f"""{get_role_definition("pharmacist_friendly")} 수집된 실제 데이터로 자연스러운 답변을 만드세요.

## 📋 대화 맥락
**이전 대화:** {conversation_context[:500]}
**현재 질문:** {current_query}
**질문 유형:** {'성분 정보 질문' if is_ingredient_question else '일반 질문'}

## 📊 수집된 데이터
{data_summary}

**⚠️ 매우 중요: 데이터 수집 원칙**
- 아래 수집된 데이터에는 Excel DB, PDF, PubChem, YouTube, 네이버 뉴스, 용량주의 성분, 연령대 금기, 일일 최대 투여량 등 다양한 소스의 정보가 포함되어 있습니다
- **모든 데이터 소스를 빠짐없이 확인하세요**: 비슷한 정보라도 각 소스마다 고유한 세부사항이 있을 수 있으므로, 모든 소스를 반드시 확인하세요
- **중복이라고 지나치지 말 것**: 같은 내용이라도 각 소스의 표현이나 추가 정보가 다를 수 있으므로, 모든 정보를 종합하세요
- **단계별 확인**: 각 데이터 소스를 순서대로 확인하고, 발견한 모든 고유한 정보를 기록하세요

## 📝 답변 작성 가이드

### 핵심 원칙 (필수)
1. **출처 숨기기**: {PromptConfig.COMMON_INSTRUCTIONS['no_source_mention']}
{get_source_mention_examples()}

2. **자연스러운 통합**: 모든 정보를 하나의 통합 지식으로 표현
   - **모든 데이터 소스의 정보를 빠짐없이 포함하세요**
   - 각 소스의 고유한 정보를 모두 반영하세요

3. **대화형 톤**: "좋은 질문이에요! 😊" 같은 친근한 시작

### 질문 유형별 답변 전략

**🧪 성분 질문 (ingredient_info)일 때:**
{'- 상세 설명 (' + str(PromptConfig.MIN_INGREDIENT_ANSWER_LENGTH) + '-' + str(PromptConfig.MAX_INGREDIENT_ANSWER_LENGTH) + '자)' if is_ingredient_question and has_translated_pubchem else ''}
{'- 작용기전 구체적 설명 (어떻게/어디에 작용)' if is_ingredient_question and has_translated_pubchem else ''}
{'- 약리학적 특성 (흡수, 대사, 반감기)' if is_ingredient_question and has_translated_pubchem else ''}
{'- 의학 분류 언급 (ATC, MeSH)' if is_ingredient_question and has_translated_pubchem else ''}
{'- **💊 한국 제품 목록 필수 안내** (예: "이 성분은 아로나민골드, 벤포벨 등 총 X개 제품에서 사용됩니다")' if is_ingredient_question and has_translated_pubchem else ''}
{'- 전문 용어는 (영문) 병기' if is_ingredient_question and has_translated_pubchem else ''}

**일반 질문일 때:**
{'- Enhanced RAG 있음: 종합 답변 (' + str(PromptConfig.MIN_SECTION_LENGTH) + '-' + str(PromptConfig.MAX_SECTION_LENGTH) + '자)' if not (is_ingredient_question and has_translated_pubchem) else ''}
{'- 일반 정보만: 핵심 답변 (' + str(PromptConfig.MIN_CONVERSATIONAL_LENGTH) + '-' + str(PromptConfig.MAX_CONVERSATIONAL_LENGTH) + '자)' if not (is_ingredient_question and has_translated_pubchem) else ''}
{'- 새 약품: 작용기전 + 안전성 + 대안 포함' if not (is_ingredient_question and has_translated_pubchem) else ''}

### 답변 구조
1. **친근한 시작**: "좋은 질문이에요!"
2. **핵심 정보**: 수집된 데이터 기반 설명 (기본 효능, 작용기전, 약리학적 특성)
3. **💡 추가 실전 정보 (중요!)**: 
   - **전문가 의견 및 실사용 정보에서 발견한 흥미로운 사실을 반드시 포함하세요**
   - 예시: "치매 예방에도 도움이 될 수 있다고 알려져 있습니다", "뇌세포 보호 효과도 있다고 합니다", "집중력 향상에 효과적이라는 연구 결과도 있습니다" 등
   - **수집된 YouTube/전문가 정보가 있다면 그 내용을 자연스럽게 여담처럼 추가하세요**
   - 형식: "또한, ~한 효과도 있다고 알려져 있어요", "재미있는 점은 ~", "추가로 ~에도 도움이 된다고 해요"
4. **마무리**: "더 궁금한 게 있으시면 물어보세요!"

### 특별 지침
- 여러 약품 언급 시 모두 균등하게 설명
- 약품 누락 금지 - 수집된 모든 약품 포함
- 각 약품의 주성분, 효능, 주의사항 개별 설명
- **YouTube/전문가 정보가 있다면 반드시 활용하여 부가 효능이나 흥미로운 사실을 추가하세요** (치매, 뇌세포 보호, 집중력 등)

실제 데이터를 바탕으로 정확하고 도움이 되는 답변을 해주세요.
"""
    
    try:
        response = llm.invoke(answer_prompt)
        answer = response.content.strip()
        
        if answer and len(answer) > 50:
            return answer
        else:
            return None
            
    except Exception as e:
        print(f"❌ 데이터 기반 답변 생성 실패: {e}")
        return None

def extract_medicine_from_context(conversation_context: str) -> Optional[str]:
    """대화 맥락에서 약품명 추출 - 강화된 버전"""
    if not conversation_context:
        return None
    
    print(f"🔍 대화 맥락에서 약품명 추출 시도: {conversation_context[:200]}...")
    
    # 1. 먼저 사용자의 최근 질문에서 약품명 추출 시도 (우선순위 높음)
    user_question_patterns = [
        r'([가-힣]{2,}정)의',  # 욱씬정의, 타이레놀정의 등
        r'([가-힣]{2,}연고)의',  # 바스포연고의 등
        r'([가-힣]{2,}정)',  # 욱씬정, 타이레놀정 등
        r'([가-힣]{2,}연고)',  # 바스포연고 등
        r'([가-힣]{2,})의',  # 뇌선의, 타이레놀의 등 (2글자 이상)
    ]
    
    # 대화를 의사 답변 기준으로 분리하여 사용자 질문 부분만 추출
    conversation_parts = conversation_context.split("의사:")
    if len(conversation_parts) > 1:
        # 가장 최근 사용자 질문 부분
        recent_user_question = conversation_parts[-1].split("사용자:")[-1] if "사용자:" in conversation_parts[-1] else conversation_parts[-1]
        
        for pattern in user_question_patterns:
            try:
                matches = re.findall(pattern, recent_user_question)
                if matches:
                    medicine = matches[-1]
                    print(f"✅ 최근 사용자 질문에서 약품명 추출: {medicine}")
                    return medicine
            except Exception as e:
                print(f"⚠️ 사용자 질문 패턴 매칭 오류: {e}")
                continue
    
    # 2. 이전 답변에서 언급된 약품명 패턴 찾기 (fallback)
    patterns = [
        r'\*\*([^*]{2,})\*\*을\(를\)',  # **뇌선**을(를)
        r'\*\*([^*]{2,})\*\*은\(는\)',  # **뇌선**은(는)
        r'([가-힣]{2,}정)은',  # 욱씬정은
        r'([가-힣]{2,}연고)는',  # 바스포연고는
        r'([가-힣]{2,})의',  # 뇌선의 (2글자 이상)
        r'([가-힣]{2,}정)',  # 욱씬정
        r'([가-힣]{2,}연고)',  # 바스포연고
    ]
    
    for pattern in patterns:
        try:
            matches = re.findall(pattern, conversation_context)
            if matches:
                medicine = matches[-1]  # 가장 최근 언급된 약품명
                print(f"✅ 패턴으로 약품명 추출: {medicine}")
                return medicine
        except Exception as e:
            print(f"⚠️ 패턴 매칭 오류: {e}")
            continue
    
    # 2. 사용자 질문 맥락에서 약품명 추출 시도
    user_context = conversation_context.split("의사:")[0] if "의사:" in conversation_context else conversation_context
    user_patterns = [
        r'([가-힣]+정)은',  # 욱씬정은
        r'([가-힣]+연고)는',  # 바스포연고는
        r'([가-힣]+)의',  # 뇌선의
        r'([가-힣]+정)',  # 욱씬정
        r'([가-힣]+연고)',  # 바스포연고
    ]
    
    for pattern in user_patterns:
        try:
            matches = re.findall(pattern, user_context)
            if matches:
                medicine = matches[-1]
                print(f"✅ 사용자 맥락에서 약품명 추출: {medicine}")
                return medicine
        except Exception as e:
            print(f"⚠️ 사용자 맥락 패턴 매칭 오류: {e}")
            continue
    
    print("❌ 약품명 추출 실패")
    return None

# 기존 특정 성분 추출 함수는 제거하고 LLM 기반 접근 방식으로 통합

def extract_medicine_from_user_question(user_context: str) -> Optional[str]:
    """사용자 질문에서 약품명 추출"""
    if not user_context:
        return None
    
    print(f"🔍 사용자 질문에서 약품명 추출 시도: {user_context}")
    
    # 사용자 질문 패턴들 (더 정확한 패턴 우선)
    patterns = [
        r'([가-힣]{2,15})(?:정|연고|크림|젤|캡슐|시럽|액|주사)(?:은|는|이|가|을|를|의)',  # 약품명+형태+조사
        r'([가-힣]{2,15})(?:정|연고|크림|젤|캡슐|시럽|액|주사)',  # 약품명+형태
        r'([가-힣]{2,15})(?:은|는|이|가|을|를|의)',  # 약품명+조사
        r'([가-힣]{2,15})(?:정|연고)',  # 약품명+정/연고
    ]
    
    for pattern in patterns:
        try:
            matches = re.findall(pattern, user_context)
            if matches:
                medicine = matches[0]  # 첫 번째 매칭 사용
                # 너무 짧거나 일반적인 단어는 제외
                if len(medicine) >= 2 and medicine not in ['무엇', '어떤', '이거', '그거', '저거', '무엇인가요', '무엇인가', '뭐야', '뭐예요']:
                    print(f"✅ 사용자 질문에서 약품명 추출: {medicine}")
                    return medicine
        except Exception as e:
            print(f"⚠️ 사용자 질문 패턴 매칭 오류: {e}")
            continue
    
    print("❌ 사용자 질문에서 약품명 추출 실패")
    return None

def handle_usage_question(medicine_name: str, context: str) -> str:
    """사용법 질문 처리 - ChatGPT 수준의 자연스러운 대화"""
    medicine_info = find_medicine_info(medicine_name, excel_docs)
    
    if medicine_info["사용법"] == "정보 없음":
        return f"아, '{medicine_name}'의 사용법 정보를 찾을 수 없네요! 다른 방법으로 도움을 드릴게요."
    
    prompt = f"""
    {get_role_definition("pharmacist")} 사용자가 이전에 {medicine_name}에 대해 물어봤고, 이제 사용법을 궁금해하고 있습니다.
    
    **약품 정보:**
    - 제품명: {medicine_name}
    - 사용법: {medicine_info['사용법']}
    - 효능: {medicine_info.get('효능', '정보 없음')}
    - 주성분: {medicine_info.get('주성분', '정보 없음')}
    
    **대화 스타일:**
    - 친근하고 대화하는 톤으로 답변
    - "네, 사용법 알려드릴게요!", "좋은 질문이에요!" 같은 자연스러운 반응
    - 사용법을 단계별로 쉽게 설명
    - 주의사항도 자연스럽게 언급
    - 마지막에 "더 궁금한 게 있으시면 언제든 물어보세요!" 같은 자연스러운 마무리
    
    **답변 구조:**
    1. 자연스러운 반응 ("네, 사용법 알려드릴게요!")
    2. 사용법 단계별 설명
    3. 주의사항 자연스럽게 언급
    4. 자연스러운 마무리
    
    자연스럽고 친근하게 답변해주세요!
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"⚠️ LLM 호출 실패: {e}")
        return f"**{medicine_name}**의 사용법을 알려드릴게요!\n\n{medicine_info['사용법']}\n\n더 궁금한 게 있으시면 언제든 물어보세요!"

def handle_ingredient_question(medicine_name: str, context: str) -> str:
    """성분 질문 처리 - 약품명/성분명 동적 분류 + PubChem 활용"""
    
    if not medicine_name:
        return "아, 어떤 약품의 성분에 대해 궁금하신지 명확하지 않네요! 약품명을 다시 말씀해 주시면 도움을 드릴게요!"
    
    print(f"🧪 성분 질문 처리: {medicine_name}")
    
    # 1단계: 약품명인지 성분명인지 분류
    classification = classify_medicine_vs_ingredient(medicine_name)
    
    print(f"🔍 분류 결과: {classification['type']} (신뢰도: {classification['confidence']})")
    
    if classification["type"] == "ingredient":
        # 성분명으로 판단됨 → 성분 상세 설명 + 포함 제품 안내
        return handle_specific_ingredient_question(classification)
    
    elif classification["type"] == "product":
        # 약품명으로 판단됨 → 해당 약품의 성분 설명
        return handle_product_ingredient_question(medicine_name)
    
    else:
        # 분류 실패 → 기본 처리
        return handle_unknown_entity_question(medicine_name)

def handle_specific_ingredient_question(classification: Dict) -> str:
    """특정 성분에 대한 상세 설명 (PubChem 활용)"""
    
    ingredient_name = classification["name"]
    products = classification.get("products", [])
    
    print(f"🧪 성분 상세 정보 수집: {ingredient_name}")
    
    # PubChem에서 상세 정보 수집
    try:
        from pubchem_api import PubChemAPI
        from translation_rag import TranslationRAG
        
        pubchem_api = PubChemAPI()
        translation_rag = TranslationRAG()
        
        # PubChem 정보 수집
        pubchem_info = pubchem_api.analyze_ingredient_comprehensive(ingredient_name)
        
        # 번역 및 요약
        translated_info = translation_rag.translate_pharmacology_info(pubchem_info.get('pharmacology_info', {}))
        summary = translated_info.get('summary_kr', '')
        description = pubchem_info.get('description', '')
        description_kr = translation_rag._translate_description(description) if description else ''
        
    except Exception as e:
        print(f"⚠️ PubChem 정보 수집 실패: {e}")
        summary = ""
        description_kr = ""
    
    # LLM으로 자연스러운 답변 생성
    prompt = f"""
{get_role_definition("pharmacist")} 사용자가 "{ingredient_name}"이라는 **성분**에 대해 궁금해하고 있습니다.

**PubChem 정보:**
{summary if summary else "정보 수집 실패"}

**설명:**
{description_kr if description_kr else "정보 수집 실패"}

**이 성분이 포함된 제품들:**
{', '.join(products[:5]) if products else "한국 DB에서 찾을 수 없음"}

**답변 요구사항:**
1. "좋은 질문이에요! 😊" 같은 친근한 시작
2. {ingredient_name}이(가) **성분명**임을 명확히 언급
3. PubChem 정보를 활용하여 **상세하게** 설명:
   - 작용기전 (메커니즘)
   - 주요 효능
   - 약리학적 특성
   - 의학적 분류
4. 이 성분이 포함된 제품들 안내 (있는 경우)
5. 전문 용어는 괄호 안에 영어 원문도 함께
6. {PromptConfig.MIN_SECTION_LENGTH}-{PromptConfig.MAX_SECTION_LENGTH}자 정도의 상세한 길이
7. "더 궁금한 점이 있으면 언제든 물어보세요!" 같은 마무리

**중요:** PubChem 정보를 최대한 활용하여 상세하게 설명하세요.
"""
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"⚠️ LLM 호출 실패: {e}")
        # Fallback
        fallback = f"**{ingredient_name}**은(는) 의약품의 주성분입니다.\n\n"
        if summary:
            fallback += f"{summary}\n\n"
        if products:
            fallback += f"💊 **이 성분이 포함된 제품들:**\n"
            for product in products[:5]:
                fallback += f"- {product}\n"
        fallback += "\n더 궁금한 점이 있으면 언제든 물어보세요!"
        return fallback

def handle_product_ingredient_question(product_name: str) -> str:
    """약품의 성분 설명"""
    
    medicine_info = find_medicine_info(product_name, excel_docs)
    
    if medicine_info.get("주성분") == "정보 없음":
        return f"죄송해요! '{product_name}'의 성분 정보를 찾을 수 없네요."
    
    prompt = f"""
{get_role_definition("pharmacist")} 사용자가 {product_name}의 성분에 대해 궁금해하고 있습니다.

**약품 정보:**
- 제품명: {product_name}
- 주성분: {medicine_info.get('주성분', '정보 없음')}
- 효능: {medicine_info.get('효능', '정보 없음')}

**답변 요구사항:**
- {PromptConfig.COMMON_INSTRUCTIONS['natural_tone']}
- "아, 성분이 궁금하시군요!" 같은 자연스러운 반응으로 시작
- 각 성분을 쉽게 설명하되 전문적인 정보도 포함
- 성분별로 어떤 역할을 하는지 설명
- {PromptConfig.MIN_CONVERSATIONAL_LENGTH}-{PromptConfig.MAX_SECTION_LENGTH}자 정도의 적절한 길이
- "더 궁금한 게 있으시면 언제든 물어보세요!" 같은 자연스러운 마무리

자연스럽고 친근하게 답변해주세요!
"""
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"⚠️ LLM 호출 실패: {e}")
        return f"**{product_name}**의 주성분을 알려드릴게요!\n\n{medicine_info.get('주성분', '정보 없음')}\n\n더 궁금한 게 있으시면 언제든 물어보세요!"

def handle_unknown_entity_question(entity_name: str) -> str:
    """분류 실패한 경우의 기본 처리"""
    
    # 일단 약품으로 가정하고 시도
    medicine_info = find_medicine_info(entity_name, excel_docs)
    
    if medicine_info.get("주성분") != "정보 없음":
        return handle_product_ingredient_question(entity_name)
    
    return f"죄송해요! '{entity_name}'에 대한 정보를 찾을 수 없네요. 정확한 약품명이나 성분명을 다시 말씀해 주시면 도움을 드릴게요!"

def handle_side_effect_question(medicine_name: str, context: str) -> str:
    """부작용 질문 처리"""
    medicine_info = find_medicine_info(medicine_name, excel_docs)
    
    if medicine_info["부작용"] == "정보 없음":
        return f"죄송합니다. '{medicine_name}'의 부작용 정보를 찾을 수 없습니다."
    
    prompt = f"""
    이전 대화에서 {medicine_name}에 대해 설명했고, 사용자가 부작용에 대해 물어보고 있습니다.
    
    약품 정보:
    - 제품명: {medicine_name}
    - 부작용: {medicine_info['부작용']}
    
    부작용을 친근하고 이해하기 쉽게 설명해주세요.
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except:
        return f"**{medicine_name}**의 부작용:\n\n{medicine_info['부작용']}"

def handle_mechanism_question(medicine_name: str, context: str) -> str:
    """작용기전 질문 처리"""
    medicine_info = find_medicine_info(medicine_name, excel_docs)
    
    prompt = f"""
    이전 대화에서 {medicine_name}에 대해 설명했고, 사용자가 작용기전에 대해 물어보고 있습니다.
    
    약품 정보:
    - 제품명: {medicine_name}
    - 주성분: {medicine_info.get('주성분', '정보 없음')}
    - 효능: {medicine_info.get('효능', '정보 없음')}
    
    작용기전을 친근하고 이해하기 쉽게 설명해주세요.
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except:
        return f"**{medicine_name}**의 작용기전에 대한 자세한 정보는 {get_medical_consultation_footer('friendly').strip()}"

def handle_precaution_question(medicine_name: str, context: str) -> str:
    """주의사항 질문 처리"""
    medicine_info = find_medicine_info(medicine_name, excel_docs)
    
    prompt = f"""
    이전 대화에서 {medicine_name}에 대해 설명했고, 사용자가 주의사항에 대해 물어보고 있습니다.
    
    약품 정보:
    - 제품명: {medicine_name}
    - 부작용: {medicine_info.get('부작용', '정보 없음')}
    - 사용법: {medicine_info.get('사용법', '정보 없음')}
    
    주의사항을 친근하고 이해하기 쉽게 설명해주세요.
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except:
        return f"**{medicine_name}**의 주의사항에 대한 자세한 정보는 {get_medical_consultation_footer('friendly').strip()}"

def handle_general_question(medicine_name: str, context: str, user_context: str) -> str:
    """일반적인 추가 질문 처리 - ChatGPT 수준의 자연스러운 대화"""
    medicine_info = find_medicine_info(medicine_name, excel_docs)
    
    prompt = f"""
    {get_role_definition("pharmacist")} 사용자가 이전에 {medicine_name}에 대해 물어봤고, 이제 추가 질문을 하고 있습니다.
    
    **사용자 질문:** {user_context}
    
    **약품 정보:**
    - 제품명: {medicine_name}
    - 효능: {medicine_info.get('효능', '정보 없음')}
    - 부작용: {medicine_info.get('부작용', '정보 없음')}
    - 사용법: {medicine_info.get('사용법', '정보 없음')}
    - 주성분: {medicine_info.get('주성분', '정보 없음')}
    
    **대화 스타일:**
    - {PromptConfig.COMMON_INSTRUCTIONS['natural_tone']}
    - "아, 그거 궁금하시군요!", "좋은 질문이에요!" 같은 자연스러운 반응
    - 사용자의 질문에 정확하고 도움이 되는 답변
    - 필요시 추가 정보나 주의사항도 자연스럽게 언급
    - 마지막에 "더 궁금한 게 있으시면 언제든 물어보세요!" 같은 자연스러운 마무리
    
    **답변 요구사항:**
    - 사용자의 질문에 직접적으로 답변
    - 전문적이지만 이해하기 쉽게 설명
    - {PromptConfig.COMMON_INSTRUCTIONS['natural_tone']}
    - 필요시 의료진 상담 권고
    
    자연스럽고 친근하게 답변해주세요!
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"⚠️ LLM 호출 실패: {e}")
        return f"**{medicine_name}**에 대한 질문이 있으시면 구체적으로 말씀해 주세요. 더 궁금한 게 있으시면 언제든 물어보세요!"

def find_medicine_info(medicine_name: str, all_docs: List) -> Dict:
    """약품명으로 약품 정보를 찾아서 반환 - type 구분 지원, PDF 링크 자동 다운로드"""
    medicine_info = {
        "제품명": medicine_name,
        "효능": "정보 없음",
        "부작용": "정보 없음", 
        "사용법": "정보 없음",
        "주성분": "정보 없음"
    }
    
    # 정확한 제품명 매칭 시도
    exact_matches = [doc for doc in all_docs if doc.metadata.get("제품명") == medicine_name]
    
    # 정확한 매칭이 없으면 부분 매칭 시도 (수출명 문제 해결)
    if not exact_matches:
        print(f"🔍 정확한 매칭 실패, 부분 매칭 시도: {medicine_name}")
        partial_matches = []
        for doc in all_docs:
            product_name = doc.metadata.get("제품명", "")
            # 약품명이 제품명의 시작 부분과 일치하는지 확인
            if product_name.startswith(medicine_name) or medicine_name in product_name:
                partial_matches.append(doc)
                print(f"  부분 매칭 발견: '{product_name}' (검색어: '{medicine_name}')")
        
        if partial_matches:
            exact_matches = partial_matches
            print(f"✅ 부분 매칭으로 '{medicine_name}' 약품 정보 발견: {len(exact_matches)}개 청크")
        else:
            print(f"❌ '{medicine_name}' 약품 정보를 찾을 수 없음")
            return medicine_info
    else:
        print(f"✅ '{medicine_name}' 약품 정보 발견: {len(exact_matches)}개 청크")
    
    # 약품 정보 수집 (여러 Excel 파일에서 병합) - medicine_usage_check_node와 동일한 로직
    import os
    import re
    url_pattern = r'https?://[^\s]+'
    
    # 새 Excel 파일 우선순위 설정
    new_excel_file = r"C:\Users\jung\Desktop\33\OpenData_ItemPermitDetail20251115.xls"
    
    # 모든 매칭된 문서를 파일별로 그룹화
    docs_by_file = {}
    for doc in exact_matches:
        excel_file = doc.metadata.get("excel_file")
        if excel_file:
            if excel_file not in docs_by_file:
                docs_by_file[excel_file] = []
            docs_by_file[excel_file].append(doc)
    
    # 새 Excel 파일이 있으면 우선순위로 설정
    file_priority = []
    if new_excel_file in docs_by_file:
        file_priority.append(new_excel_file)
    for file in docs_by_file.keys():
        if file != new_excel_file:
            file_priority.append(file)
    
    print(f"📂 약품 정보 출처 파일: {len(file_priority)}개 파일에서 발견")
    for file in file_priority:
        print(f"  - {os.path.basename(file)} ({len(docs_by_file[file])}개 청크)")
    
    # 모든 Excel 파일에서 정보 수집 (파일별로 그룹화)
    excel_file = None
    excel_row_index = None
    
    # 각 파일별로 정보를 수집하여 리스트로 저장
    all_efficacy_info = []  # [(파일명, 효능정보), ...]
    all_side_effects_info = []  # [(파일명, 부작용정보), ...]
    all_usage_info = []  # [(파일명, 사용법정보), ...]
    
    for file in file_priority:
        file_name = os.path.basename(file)
        file_efficacy = None
        file_side_effects = None
        file_usage = None
        
        for doc in docs_by_file[file]:
            content = doc.page_content
            doc_type = doc.metadata.get("type", "")
            
            # Excel 파일 정보 저장 (우선순위가 높은 파일에서)
            if not excel_file:
                excel_file = doc.metadata.get("excel_file")
                excel_row_index = doc.metadata.get("excel_row_index")
            
            # 효능과 부작용은 main 타입에서 추출
            if doc_type == "main" or doc_type == "":
                efficacy = extract_field_from_doc(content, "효능")
                side_effects = extract_field_from_doc(content, "부작용")
                main_ingredient = doc.metadata.get("주성분", "정보 없음")
                
                # 주성분은 첫 번째 파일에서만 저장
                if not medicine_info.get("주성분") or medicine_info["주성분"] == "정보 없음":
                    if main_ingredient != "정보 없음":
                        medicine_info["주성분"] = main_ingredient
                
                # URL이 아닌 경우에만 수집
                if efficacy != "정보 없음" and not re.search(url_pattern, str(efficacy)):
                    if file_efficacy is None:
                        file_efficacy = efficacy
                    else:
                        # 같은 파일 내에서 여러 청크가 있으면 더 긴 것을 선택
                        if len(efficacy) > len(file_efficacy):
                            file_efficacy = efficacy
                
                if side_effects != "정보 없음" and not re.search(url_pattern, str(side_effects)):
                    if file_side_effects is None:
                        file_side_effects = side_effects
                    else:
                        if len(side_effects) > len(file_side_effects):
                            file_side_effects = side_effects
            
            # 사용법은 usage 타입에서 추출
            if doc_type == "usage":
                usage = extract_field_from_doc(content, "사용법")
                if usage != "정보 없음" and not re.search(url_pattern, str(usage)):
                    if file_usage is None:
                        file_usage = usage
                    else:
                        if len(usage) > len(file_usage):
                            file_usage = usage
        
        # 파일별로 수집한 정보를 리스트에 추가
        if file_efficacy:
            all_efficacy_info.append((file_name, file_efficacy))
            print(f"📋 {file_name}에서 효능 정보 수집: {len(file_efficacy)}자")
        if file_side_effects:
            all_side_effects_info.append((file_name, file_side_effects))
            print(f"📋 {file_name}에서 부작용 정보 수집: {len(file_side_effects)}자")
        if file_usage:
            all_usage_info.append((file_name, file_usage))
            print(f"📋 {file_name}에서 사용법 정보 수집: {len(file_usage)}자")
    
    # 여러 소스의 정보를 LLM으로 병합
    from medicine_usage_check_node import merge_multiple_sources_with_llm
    
    if len(all_efficacy_info) > 1:
        print(f"🔄 {len(all_efficacy_info)}개 소스의 효능 정보 병합 중...")
        merged_efficacy = merge_multiple_sources_with_llm(all_efficacy_info, "효능")
        medicine_info["효능"] = merged_efficacy
    elif len(all_efficacy_info) == 1:
        medicine_info["효능"] = all_efficacy_info[0][1]
    
    if len(all_side_effects_info) > 1:
        print(f"🔄 {len(all_side_effects_info)}개 소스의 부작용 정보 병합 중...")
        merged_side_effects = merge_multiple_sources_with_llm(all_side_effects_info, "부작용")
        medicine_info["부작용"] = merged_side_effects
    elif len(all_side_effects_info) == 1:
        medicine_info["부작용"] = all_side_effects_info[0][1]
    
    if len(all_usage_info) > 1:
        print(f"🔄 {len(all_usage_info)}개 소스의 사용법 정보 병합 중...")
        merged_usage = merge_multiple_sources_with_llm(all_usage_info, "사용법")
        medicine_info["사용법"] = merged_usage
    elif len(all_usage_info) == 1:
        medicine_info["사용법"] = all_usage_info[0][1]
    
    # PDF 링크 확인 및 다운로드 (모든 파일에서 수집하여 병합)
    from pdf_link_extractor import enrich_excel_row_with_pdf_content
    from retrievers import file_column_mappings, default_columns
    
    # 모든 파일에서 PDF 정보 수집
    all_pdf_efficacy = []
    all_pdf_side_effects = []
    all_pdf_usage = []
    
    for file in file_priority:
        # 해당 파일의 문서에서 excel_row_index 찾기
        file_row_index = None
        for doc in docs_by_file[file]:
            if doc.metadata.get("excel_file") == file:
                file_row_index = doc.metadata.get("excel_row_index")
                if file_row_index is not None:
                    break
        
        if file_row_index is None:
            continue
        
        print(f"📥 PDF 다운로드 시도: {os.path.basename(file)}, 행 {file_row_index}")
        try:
            # 파일별 컬럼 매핑 확인
            if file in file_column_mappings:
                col_mapping = file_column_mappings[file]
            else:
                col_mapping = default_columns
            
            pdf_column_mapping = {
                '효능': col_mapping['효능'],
                '복용법': col_mapping['사용법'],
                '주의사항': col_mapping['부작용']
            }
            
            # 효능, 부작용, 사용법이 URL인지 확인하고 PDF 다운로드
            # 연속 질문일 때는 요약을 덜 심하게 하여 더 상세한 내용 제공
            pdf_content = enrich_excel_row_with_pdf_content(
                file, file_row_index, ['효능', '주의사항', '복용법'], pdf_column_mapping,
                summarize=True,  # 요약은 하되
                max_length=5000  # 연속 질문일 때는 더 긴 내용 제공 (기본값 2000자 → 5000자)
            )
            
            print(f"📋 PDF 내용 확인: {list(pdf_content.keys())}")
            for key, value in pdf_content.items():
                if value:
                    print(f"  - {key}: {len(str(value))}자 - {str(value)[:100]}...")
                    # PDF 정보를 리스트에 추가
                    file_name = os.path.basename(file)
                    if key == '효능' and value:
                        all_pdf_efficacy.append((file_name, value))
                    elif key == '주의사항' and value:
                        all_pdf_side_effects.append((file_name, value))
                    elif key == '복용법' and value:
                        all_pdf_usage.append((file_name, value))
                else:
                    print(f"  - {key}: None")
        
        except Exception as e:
            print(f"⚠️ {os.path.basename(file)} PDF 다운로드 실패 (계속 진행): {e}")
    
    # PDF 정보를 기존 Excel 정보와 병합
    if all_pdf_efficacy:
        current_efficacy = medicine_info.get("효능", "정보 없음")
        if current_efficacy != "정보 없음":
            # Excel 정보와 PDF 정보를 모두 병합
            all_efficacy_sources = all_efficacy_info + all_pdf_efficacy
            if len(all_efficacy_sources) > 1:
                print(f"🔄 Excel + PDF 효능 정보 병합 중... ({len(all_efficacy_sources)}개 소스)")
                merged_efficacy = merge_multiple_sources_with_llm(all_efficacy_sources, "효능")
                medicine_info["효능"] = merged_efficacy
            else:
                medicine_info["효능"] = all_efficacy_sources[0][1]
        else:
            # Excel 정보가 없으면 PDF 정보만 사용
            if len(all_pdf_efficacy) > 1:
                merged_efficacy = merge_multiple_sources_with_llm(all_pdf_efficacy, "효능")
                medicine_info["효능"] = merged_efficacy
            elif len(all_pdf_efficacy) == 1:
                medicine_info["효능"] = all_pdf_efficacy[0][1]
    
    if all_pdf_side_effects:
        current_side_effects = medicine_info.get("부작용", "정보 없음")
        if current_side_effects != "정보 없음":
            # Excel 정보와 PDF 정보를 모두 병합
            all_side_effects_sources = all_side_effects_info + all_pdf_side_effects
            if len(all_side_effects_sources) > 1:
                print(f"🔄 Excel + PDF 부작용 정보 병합 중... ({len(all_side_effects_sources)}개 소스)")
                merged_side_effects = merge_multiple_sources_with_llm(all_side_effects_sources, "부작용")
                medicine_info["부작용"] = merged_side_effects
            else:
                medicine_info["부작용"] = all_side_effects_sources[0][1]
        else:
            # Excel 정보가 없으면 PDF 정보만 사용
            if len(all_pdf_side_effects) > 1:
                merged_side_effects = merge_multiple_sources_with_llm(all_pdf_side_effects, "부작용")
                medicine_info["부작용"] = merged_side_effects
            elif len(all_pdf_side_effects) == 1:
                medicine_info["부작용"] = all_pdf_side_effects[0][1]
    
    if all_pdf_usage:
        current_usage = medicine_info.get("사용법", "정보 없음")
        if current_usage != "정보 없음":
            # Excel 정보와 PDF 정보를 모두 병합
            all_usage_sources = all_usage_info + all_pdf_usage
            if len(all_usage_sources) > 1:
                print(f"🔄 Excel + PDF 사용법 정보 병합 중... ({len(all_usage_sources)}개 소스)")
                merged_usage = merge_multiple_sources_with_llm(all_usage_sources, "사용법")
                medicine_info["사용법"] = merged_usage
            else:
                medicine_info["사용법"] = all_usage_sources[0][1]
        else:
            # Excel 정보가 없으면 PDF 정보만 사용
            if len(all_pdf_usage) > 1:
                merged_usage = merge_multiple_sources_with_llm(all_pdf_usage, "사용법")
                medicine_info["사용법"] = merged_usage
            elif len(all_pdf_usage) == 1:
                medicine_info["사용법"] = all_pdf_usage[0][1]
    
    return medicine_info

def extract_field_from_doc(text: str, label: str) -> str:
    """문서에서 특정 필드 추출"""
    pattern = rf"\[{label}\]:\s*((?:.|\n)*?)(?=\n\[|\Z)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else "정보 없음"

def handle_alternative_medicines_question(medicine_name: str, conversation_context: str, current_query: str) -> str:
    """대안 약품 질문 처리 (성분 중심 설명)"""
    print(f"🔍 대안 약품 질문 처리: {medicine_name}")
    
    # 이전 대화에서 언급된 대안 약품들 추출
    alternative_medicines_from_context = extract_alternative_medicines_from_context(conversation_context)
    print(f"  대화에서 추출된 대안 약품들: {alternative_medicines_from_context}")
    
    if not alternative_medicines_from_context:
        return f"죄송합니다. 이전 대화에서 언급된 대안 약품을 찾을 수 없습니다."
    
    # 각 대안 약품의 상세 정보 수집
    detailed_alternatives = []
    for alt_medicine in alternative_medicines_from_context:
        print(f"  개별 약품 정보 수집: {alt_medicine}")
        alt_info = find_medicine_info_in_excel(alt_medicine)
        if alt_info and alt_info["효능"] != "정보 없음":
            ingredients = extract_ingredients_from_medicine_info(alt_info)
            print(f"    성분 추출: {ingredients}")
            detailed_alternatives.append({
                "name": alt_medicine,
                "ingredients": ingredients,
                "efficacy": alt_info.get("효능", "정보 없음"),
                "side_effects": alt_info.get("부작용", "정보 없음"),
                "usage": alt_info.get("사용법", "정보 없음"),
                "content": f"효능: {alt_info.get('효능', '정보 없음')}\n부작용: {alt_info.get('부작용', '정보 없음')}\n사용법: {alt_info.get('사용법', '정보 없음')}"
            })
        else:
            print(f"    약품 정보 없음: {alt_medicine}")
    
    if not detailed_alternatives:
        return f"죄송합니다. 언급된 대안 약품들의 상세 정보를 찾을 수 없습니다."
    
    # 원본 약품 정보 찾기
    target_medicine_info = find_medicine_info_in_excel(medicine_name)
    target_ingredients = extract_ingredients_from_medicine_info(target_medicine_info) if target_medicine_info else []
    
    # 상세한 대안 분석 생성 (성분 중심)
    return generate_ingredient_focused_alternative_analysis(medicine_name, detailed_alternatives, target_medicine_info or {}, target_ingredients)

def handle_similar_medicines_question(medicine_name: str, conversation_context: str, current_query: str) -> str:
    """유사 약품 질문 처리"""
    print(f"🔍 유사 약품 질문 처리: {medicine_name}")
    
    # Excel DB에서 대상 약품 정보 찾기
    target_medicine_info = find_medicine_info_in_excel(medicine_name)
    if not target_medicine_info or target_medicine_info["효능"] == "정보 없음":
        return f"죄송합니다. '{medicine_name}'에 대한 정보를 찾을 수 없어서 유사 약품을 제안할 수 없습니다."
    
    # 동적 유사 약품 검색
    similar_medicines = find_similar_medicines_dynamically(medicine_name, target_medicine_info)
    
    if not similar_medicines:
        return f"죄송합니다. '{medicine_name}'과 유사한 약품을 찾을 수 없습니다."
    
    # 상세한 유사 약품 분석 생성
    return generate_detailed_similar_analysis(medicine_name, similar_medicines, target_medicine_info)

def find_medicine_info_in_excel(medicine_name: str) -> Dict:
    """Excel DB에서 약품 정보 찾기"""
    for doc in excel_docs:
        if doc.metadata.get("제품명") == medicine_name:
            return {
                "제품명": doc.metadata.get("제품명", ""),
                "주성분": doc.metadata.get("주성분", ""),
                "효능": extract_field_from_doc(doc.page_content, "효능"),
                "부작용": extract_field_from_doc(doc.page_content, "부작용"),
                "사용법": extract_field_from_doc(doc.page_content, "사용법"),
                "content": doc.page_content
            }
    return {}

def find_alternative_medicines_dynamically(medicine_name: str, target_medicine_info: Dict) -> List[Dict]:
    """동적으로 대안 약품 검색"""
    print(f"🔍 동적 대안 약품 검색: {medicine_name}")
    
    # 대상 약품의 주성분 추출
    target_ingredients = extract_ingredients_from_medicine_info(target_medicine_info)
    print(f"  대상 약품 주성분: {target_ingredients}")
    
    alternative_medicines = []
    
    # Excel DB 전체에서 유사한 약품들 검색
    for doc in excel_docs:
        doc_name = doc.metadata.get("제품명", "")
        if doc_name == medicine_name:  # 자기 자신은 제외
            continue
            
        doc_ingredients = extract_ingredients_from_doc(doc)
        if not doc_ingredients:
            continue
        
        # 유사도 계산
        similarity_score = calculate_ingredient_similarity(target_ingredients, doc_ingredients)
        
        if similarity_score > 0.3:  # 30% 이상 유사하면 후보로 추가
            alternative_medicines.append({
                "name": doc_name,
                "ingredients": doc_ingredients,
                "similarity_score": similarity_score,
                "efficacy": extract_field_from_doc(doc.page_content, "효능"),
                "side_effects": extract_field_from_doc(doc.page_content, "부작용"),
                "usage": extract_field_from_doc(doc.page_content, "사용법"),
                "content": doc.page_content
            })
    
    # 유사도 순으로 정렬하여 상위 3개 반환
    alternative_medicines.sort(key=lambda x: x["similarity_score"], reverse=True)
    return alternative_medicines[:3]

def find_alternative_medicines_with_priority(medicine_name: str, target_medicine_info: Dict, target_ingredients: List[str]) -> List[Dict]:
    """우선순위를 적용한 대안 약품 검색 (동일 성분 > 유사 성분 > 효능 기반)"""
    print(f"🔍 우선순위 대안 약품 검색: {medicine_name}")
    
    # 1단계: 동일 성분 약품 검색
    same_ingredient_medicines = find_medicines_with_same_ingredients(medicine_name, target_ingredients)
    print(f"  동일 성분 약품: {[med['name'] for med in same_ingredient_medicines]}")
    
    # 2단계: 유사 성분 약품 검색
    similar_ingredient_medicines = find_medicines_with_similar_ingredients(medicine_name, target_ingredients)
    print(f"  유사 성분 약품: {[med['name'] for med in similar_ingredient_medicines]}")
    
    # 3단계: 효능 기반 약품 검색
    efficacy_based_medicines = find_medicines_by_efficacy(medicine_name, target_medicine_info, target_ingredients)
    print(f"  효능 기반 약품: {[med['name'] for med in efficacy_based_medicines]}")
    
    # 우선순위별로 정렬하여 상위 3개 반환
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

def find_medicines_with_same_ingredients(medicine_name: str, target_ingredients: List[str]) -> List[Dict]:
    """동일 성분을 가진 약품 검색"""
    same_ingredient_medicines = []
    
    for doc in excel_docs:
        doc_name = doc.metadata.get("제품명", "")
        if doc_name == medicine_name:
            continue
            
        doc_ingredients = extract_ingredients_from_doc(doc)
        if not doc_ingredients:
            continue
        
        # 동일 성분 확인 (순서 무관)
        if set(target_ingredients) == set(doc_ingredients):
            same_ingredient_medicines.append({
                "name": doc_name,
                "ingredients": doc_ingredients,
                "similarity_score": 1.0,
                "efficacy": extract_field_from_doc(doc.page_content, "효능"),
                "side_effects": extract_field_from_doc(doc.page_content, "부작용"),
                "usage": extract_field_from_doc(doc.page_content, "사용법"),
                "content": doc.page_content,
                "priority": 1
            })
    
    return same_ingredient_medicines

def find_medicines_with_similar_ingredients(medicine_name: str, target_ingredients: List[str]) -> List[Dict]:
    """유사 성분을 가진 약품 검색"""
    similar_ingredient_medicines = []
    
    for doc in excel_docs:
        doc_name = doc.metadata.get("제품명", "")
        if doc_name == medicine_name:
            continue
            
        doc_ingredients = extract_ingredients_from_doc(doc)
        if not doc_ingredients:
            continue
        
        # 유사도 계산
        similarity_score = calculate_ingredient_similarity(target_ingredients, doc_ingredients)
        
        # 50% 이상 유사하고 완전 일치가 아닌 경우
        if 0.5 <= similarity_score < 1.0:
            similar_ingredient_medicines.append({
                "name": doc_name,
                "ingredients": doc_ingredients,
                "similarity_score": similarity_score,
                "efficacy": extract_field_from_doc(doc.page_content, "효능"),
                "side_effects": extract_field_from_doc(doc.page_content, "부작용"),
                "usage": extract_field_from_doc(doc.page_content, "사용법"),
                "content": doc.page_content,
                "priority": 2
            })
    
    return similar_ingredient_medicines

def find_medicines_by_efficacy(medicine_name: str, target_medicine_info: Dict, target_ingredients: List[str]) -> List[Dict]:
    """효능 기반 약품 검색"""
    efficacy_based_medicines = []
    target_efficacy = target_medicine_info.get("효능", "")
    
    for doc in excel_docs:
        doc_name = doc.metadata.get("제품명", "")
        if doc_name == medicine_name:
            continue
            
        doc_ingredients = extract_ingredients_from_doc(doc)
        if not doc_ingredients:
            continue
        
        doc_efficacy = extract_field_from_doc(doc.page_content, "효능")
        
        # 효능 기반 유사도 계산 (간단한 키워드 매칭)
        efficacy_similarity = calculate_efficacy_similarity(target_efficacy, doc_efficacy)
        
        if efficacy_similarity > 0.3:
            efficacy_based_medicines.append({
                "name": doc_name,
                "ingredients": doc_ingredients,
                "similarity_score": efficacy_similarity,
                "efficacy": doc_efficacy,
                "side_effects": extract_field_from_doc(doc.page_content, "부작용"),
                "usage": extract_field_from_doc(doc.page_content, "사용법"),
                "content": doc.page_content,
                "priority": 3
            })
    
    return efficacy_based_medicines

def calculate_efficacy_similarity(target_efficacy: str, doc_efficacy: str) -> float:
    """효능 기반 유사도 계산"""
    if not target_efficacy or not doc_efficacy or target_efficacy == "정보 없음" or doc_efficacy == "정보 없음":
        return 0.0
    
    # 간단한 키워드 매칭
    target_keywords = set(target_efficacy.lower().split())
    doc_keywords = set(doc_efficacy.lower().split())
    
    if not target_keywords or not doc_keywords:
        return 0.0
    
    common_keywords = target_keywords & doc_keywords
    union_keywords = target_keywords | doc_keywords
    
    return len(common_keywords) / len(union_keywords) if union_keywords else 0.0

def find_similar_medicines_dynamically(medicine_name: str, target_medicine_info: Dict) -> List[Dict]:
    """동적으로 유사 약품 검색 (대안과 동일한 로직)"""
    return find_alternative_medicines_dynamically(medicine_name, target_medicine_info)

def extract_ingredients_from_medicine_info(medicine_info: Dict) -> List[str]:
    """약품 정보에서 주성분 추출"""
    ingredients = []
    
    if medicine_info.get('주성분') and medicine_info['주성분'] != '정보 없음':
        main_ingredient = medicine_info['주성분']
        if ',' in main_ingredient:
            ingredients = [ing.strip() for ing in main_ingredient.split(',') if ing.strip()]
        else:
            ingredients = [main_ingredient.strip()]
    
    return ingredients

def extract_ingredients_from_doc(doc) -> List[str]:
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

def calculate_ingredient_similarity(target_ingredients: List[str], doc_ingredients: List[str]) -> float:
    """주성분 유사도 계산"""
    if not target_ingredients or not doc_ingredients:
        return 0.0
    
    # 정규화된 성분명으로 변환
    target_normalized = [normalize_ingredient_name(ing) for ing in target_ingredients]
    doc_normalized = [normalize_ingredient_name(ing) for ing in doc_ingredients]
    
    # 교집합 계산
    common_ingredients = set(target_normalized) & set(doc_normalized)
    
    if not common_ingredients:
        return 0.0
    
    # 유사도 = 교집합 크기 / 합집합 크기
    union_size = len(set(target_normalized) | set(doc_normalized))
    similarity = len(common_ingredients) / union_size
    
    return similarity

def normalize_ingredient_name(ingredient: str) -> str:
    """성분명 정규화"""
    if not ingredient:
        return ""
    
    # 소문자 변환 및 특수문자 제거
    normalized = ingredient.lower().strip()
    normalized = ''.join(c for c in normalized if c.isalnum() or c in '가-힣')
    
    return normalized

def generate_detailed_alternative_analysis(medicine_name: str, alternative_medicines: List[Dict], target_medicine_info: Dict) -> str:
    """상세한 대안 분석 생성"""
    
    # LLM에게 상세한 대안 분석 요청
    prompt = f"""
    당신은 전문적인 약사입니다. 다음 정보를 바탕으로 {medicine_name}의 대안 약품에 대해 상세하고 근거 있는 분석을 제공해주세요.

    **대상 약품 ({medicine_name}) 정보:**
    - 주성분: {target_medicine_info.get('주성분', '정보 없음')}
    - 효능: {target_medicine_info.get('효능', '정보 없음')}
    - 부작용: {target_medicine_info.get('부작용', '정보 없음')}

    **발견된 대안 약품들:**
    {format_alternative_medicines_for_analysis(alternative_medicines)}

    **분석 요구사항:**
    1. 각 대안 약품의 주성분과 효과를 구체적으로 분석
    2. 대상 약품과의 유사점과 차이점을 명확히 설명
    3. 각 대안의 장단점을 객관적으로 제시
    4. 사용 시 주의사항과 부작용을 포함
    5. 어떤 상황에서 어떤 대안을 선택하는 것이 좋은지 조언
    
    **중요 지침:**
    - 반드시 위에서 제공된 대안 약품들만 분석하고 언급
    - 이부프로펜, 나프록센 같은 성분명을 대안으로 제시하지 말고, 실제 약품명만 사용
    - 발견된 대안 약품이 없으면 "해당 약품과 유사한 대안을 찾을 수 없습니다"라고 명시

    **답변 구조:**
    1. **대안 약품 개요**: 발견된 대안 약품들 소개
    2. **상세 분석**: 각 대안 약품별 상세 분석
    3. **비교 분석**: 대상 약품과의 비교
    4. **선택 가이드**: 상황별 추천 가이드
    5. **주의사항**: 공통 주의사항 및 부작용

    친근하고 이해하기 쉽게 설명해주세요.
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"❌ 대안 분석 생성 오류: {e}")
        return generate_fallback_alternative_analysis(medicine_name, alternative_medicines)

def generate_detailed_similar_analysis(medicine_name: str, similar_medicines: List[Dict], target_medicine_info: Dict) -> str:
    """상세한 유사 약품 분석 생성 (대안과 동일한 로직)"""
    return generate_detailed_alternative_analysis(medicine_name, similar_medicines, target_medicine_info)

def format_alternative_medicines_for_analysis(alternative_medicines: List[Dict]) -> str:
    """분석용 대안 약품 정보 포맷팅"""
    if not alternative_medicines:
        return "대안 약품 없음"
    
    formatted = []
    for i, alt in enumerate(alternative_medicines, 1):
        formatted.append(f"{i}. {alt['name']}")
        formatted.append(f"   - 주성분: {', '.join(alt['ingredients'])}")
        formatted.append(f"   - 유사도: {alt['similarity_score']:.2f}")
        formatted.append(f"   - 효능: {alt['efficacy']}")
        formatted.append(f"   - 부작용: {alt['side_effects']}")
        formatted.append(f"   - 사용법: {alt['usage']}")
        formatted.append("")
    
    return "\n".join(formatted)

def generate_fallback_alternative_analysis(medicine_name: str, alternative_medicines: List[Dict]) -> str:
    """오류 시 기본 대안 분석"""
    response = f"**{medicine_name}의 대안 약품 분석**\n\n"
    
    if not alternative_medicines:
        return response + "죄송합니다. 대안 약품을 찾을 수 없습니다."
    
    response += f"Excel DB 분석 결과, 다음과 같은 대안 약품들을 찾았습니다:\n\n"
    
    for i, alt in enumerate(alternative_medicines, 1):
        response += f"**{i}. {alt['name']}**\n"
        response += f"- 주성분: {', '.join(alt['ingredients'])}\n"
        response += f"- 효능: {alt['efficacy']}\n"
        response += f"- 유사도: {alt['similarity_score']:.2f}\n\n"
    
    response += get_medical_consultation_footer("warning")
    
    return response

def generate_ingredient_focused_alternative_analysis(medicine_name: str, alternative_medicines: List[Dict], target_medicine_info: Dict, target_ingredients: List[str]) -> str:
    """성분 중심의 대안 분석 생성"""
    print(f"🔍 성분 중심 대안 분석 생성: {medicine_name}")
    
    # 우선순위별로 그룹화
    same_ingredient = [med for med in alternative_medicines if med.get("priority") == 1]
    similar_ingredient = [med for med in alternative_medicines if med.get("priority") == 2]
    efficacy_based = [med for med in alternative_medicines if med.get("priority") == 3]
    
    analysis_parts = []
    
    # 1. 동일 성분 약품 분석
    if same_ingredient:
        analysis_parts.append(f"**🟢 동일 성분 대안 약품:**")
        for med in same_ingredient:
            analysis_parts.append(f"• **{med['name']}**: {', '.join(med['ingredients'])}")
            analysis_parts.append(f"  - {medicine_name}과 완전히 동일한 성분으로 동일한 효과")
            analysis_parts.append(f"  - 효능: {med['efficacy']}")
            if med.get('side_effects') and med['side_effects'] != '정보 없음':
                analysis_parts.append(f"  - 주의사항: {med['side_effects']}")
        analysis_parts.append("")
    
    # 1-1. 실제 찾은 대안 약품들 분석 (우선순위 없이)
    if not same_ingredient and not similar_ingredient and not efficacy_based:
        analysis_parts.append(f"**🔍 발견된 대안 약품들:**")
        for med in alternative_medicines:
            analysis_parts.append(f"• **{med['name']}**: {', '.join(med['ingredients']) if med['ingredients'] else '성분 정보 없음'}")
            analysis_parts.append(f"  - 효능: {med['efficacy']}")
            if med.get('side_effects') and med['side_effects'] != '정보 없음':
                analysis_parts.append(f"  - 주의사항: {med['side_effects']}")
        analysis_parts.append("")
    
    # 2. 유사 성분 약품 분석
    if similar_ingredient:
        analysis_parts.append(f"**🟡 유사 성분 대안 약품:**")
        for med in similar_ingredient:
            analysis_parts.append(f"• **{med['name']}**: {', '.join(med['ingredients'])}")
            common_ingredients = set(target_ingredients) & set(med['ingredients'])
            different_ingredients = set(med['ingredients']) - set(target_ingredients)
            
            if common_ingredients:
                analysis_parts.append(f"  - 공통 성분: {', '.join(common_ingredients)} (유사한 효과)")
            if different_ingredients:
                analysis_parts.append(f"  - 추가 성분: {', '.join(different_ingredients)} (추가 효과)")
            analysis_parts.append(f"  - 효능: {med['efficacy']}")
            if med.get('side_effects') and med['side_effects'] != '정보 없음':
                analysis_parts.append(f"  - 주의사항: {med['side_effects']}")
        analysis_parts.append("")
    
    # 3. 효능 기반 약품 분석
    if efficacy_based:
        analysis_parts.append(f"**🔵 효능 기반 대안 약품:**")
        for med in efficacy_based:
            analysis_parts.append(f"• **{med['name']}**: {', '.join(med['ingredients'])}")
            analysis_parts.append(f"  - 다른 성분이지만 유사한 효능")
            analysis_parts.append(f"  - 효능: {med['efficacy']}")
            if med.get('side_effects') and med['side_effects'] != '정보 없음':
                analysis_parts.append(f"  - 주의사항: {med['side_effects']}")
        analysis_parts.append("")
    
    # 4. 성분별 상세 설명
    analysis_parts.append(f"**🧪 주요 성분별 작용기전:**")
    for ingredient in target_ingredients:
        analysis_parts.append(f"• **{ingredient}**:")
        if ingredient == "아세트아미노펜":
            analysis_parts.append("  - 중추신경계에서 프로스타글란딘 합성 억제")
            analysis_parts.append("  - 해열진통 효과, 위장관 부작용 적음")
        elif ingredient == "카페인무수물":
            analysis_parts.append("  - 아데노신 수용체 차단으로 중추신경계 자극")
            analysis_parts.append("  - 진통제 효과 증강, 각성 효과")
        elif ingredient == "푸르설티아민":
            analysis_parts.append("  - 비타민 B1 유도체로 신경계 기능 개선")
            analysis_parts.append("  - 피로 회복, 신경염 예방")
        else:
            analysis_parts.append("  - 해당 성분의 구체적인 작용기전")
    analysis_parts.append("")
    
    # 5. 선택 가이드
    analysis_parts.append(f"**💡 선택 가이드:**")
    if same_ingredient:
        analysis_parts.append("• 동일한 효과를 원한다면 → 동일 성분 약품 추천")
    if similar_ingredient:
        analysis_parts.append("• 비슷한 효과에 추가 효과를 원한다면 → 유사 성분 약품 고려")
    if efficacy_based:
        analysis_parts.append("• 다른 성분으로 같은 효과를 원한다면 → 효능 기반 약품 고려")
    
    analysis_parts.append("• 개인 건강 상태와 알레르기 이력을 고려하여 선택")
    analysis_parts.append("• 장기 복용 시 의사와 상담 권장")
    
    return "\n".join(analysis_parts)

def extract_alternative_medicines_from_context(conversation_context: str) -> List[str]:
    """대화 맥락에서 언급된 대안 약품들 추출 (동적 방식)"""
    print(f"🔍 대화에서 대안 약품 추출: {conversation_context[:100]}...")
    
    # LLM을 사용한 지능적 추출
    try:
        from openai import OpenAI
        client = OpenAI(api_key="your-api-key")  # 실제 API 키로 교체 필요
        
        prompt = f"""
다음 대화에서 언급된 약품명들을 추출해주세요. 
대화 내용: {conversation_context}

다음 형식으로 JSON 응답해주세요:
{{"medicines": ["약품명1", "약품명2", "약품명3"]}}

주의사항:
- 실제 약품명만 추출 (성분명 제외)
- 이부프로펜, 나프록센 같은 성분명은 제외
- 포펜정, 게보린정, 타이레놀 같은 실제 약품명만 포함
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 의학 전문가입니다. 대화에서 언급된 약품명을 정확하게 추출합니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.1
        )
        
        import json
        result = json.loads(response.choices[0].message.content.strip())
        medicines = result.get("medicines", [])
        print(f"  LLM으로 추출된 약품들: {medicines}")
        return medicines
        
    except Exception as e:
        print(f"❌ LLM 추출 실패: {e}")
        # 폴백: 간단한 패턴 매칭
        return extract_medicines_simple_pattern(conversation_context)

def extract_medicines_simple_pattern(conversation_context: str) -> List[str]:
    """Excel DB 기반 약품명 추출 (하드코딩 없는 방식)"""
    print(f"🔍 Excel DB 기반 약품명 추출 시작")
    
    # Excel DB에서 모든 약품명 가져오기
    all_medicine_names = set()
    for doc in excel_docs:
        medicine_name = doc.metadata.get("제품명", "")
        if medicine_name and medicine_name != "정보 없음":
            all_medicine_names.add(medicine_name)
    
    print(f"  Excel DB에 있는 약품 수: {len(all_medicine_names)}")
    
    # 대화에서 Excel DB에 있는 약품명들만 찾기
    found_medicines = []
    for medicine_name in all_medicine_names:
        # 정확한 매칭을 위해 단어 경계 고려
        import re
        pattern = r'\b' + re.escape(medicine_name) + r'\b'
        if re.search(pattern, conversation_context):
            found_medicines.append(medicine_name)
            print(f"  발견된 약품: {medicine_name}")
    
    # 중복 제거 및 정렬
    unique_medicines = list(set(found_medicines))
    print(f"  최종 추출된 약품들: {unique_medicines}")
    return unique_medicines
