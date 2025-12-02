from qa_state import QAState
from answer_utils import generate_response_llm_from_prompt
from langchain_core.documents import Document
from config import PromptConfig
from prompt_utils import (
    get_role_definition, get_section_structure, get_common_instructions,
    get_medical_consultation_footer
)
import re
import json
from typing import List

def contains_exact_product_name(doc: Document, product_name: str) -> bool:
    return re.search(rf"\[제품명\]:\s*{re.escape(product_name)}\b", doc.page_content) is not None

def extract_medicine_from_context(conversation_context: str) -> list:
    """대화 맥락에서 약품 정보를 추출하는 함수 (LLM 기반)"""
    if not conversation_context:
        return []
    
    # LLM에게 약품명 추출 요청
    extraction_prompt = f"""
다음 대화 맥락에서 언급된 약품명들을 추출해주세요.

**대화 맥락:**
{conversation_context}

**응답 형식:**
약품명만 쉼표로 구분하여 나열해주세요.
예: 아스피린, 이부프로펜, 파라세타몰
"""
    
    try:
        response = generate_response_llm_from_prompt(
            prompt=extraction_prompt,
            temperature=0.1,
            max_tokens=200
        )
        
        # 쉼표로 구분된 약품명들을 리스트로 변환
        medicines = [name.strip() for name in response.split(',') if name.strip()]
        return medicines
        
    except Exception as e:
        print(f"❌ 약품명 추출 중 오류 발생: {e}")
        return []

def extract_medicine_details_from_context(conversation_context: str) -> dict:
    """대화 맥락에서 약품의 상세 정보를 추출하는 함수 (LLM 기반)"""
    if not conversation_context:
        return {}
    
    # LLM에게 상세 정보 추출 요청
    extraction_prompt = f"""
다음 대화 맥락에서 언급된 약품들의 상세 정보를 추출해주세요.

**대화 맥락:**
{conversation_context}

**추출할 정보:**
- 약품명
- 효능/효과
- 부작용
- 사용법
- 주의사항

JSON 형식으로 응답해주세요:
{{
    "medicines": [
        {{
            "name": "약품명",
            "effects": ["효능1", "효능2"],
            "side_effects": ["부작용1", "부작용2"],
            "usage": "사용법",
            "precautions": ["주의사항1", "주의사항2"]
        }}
    ]
}}
"""
    
    try:
        response = generate_response_llm_from_prompt(
            prompt=extraction_prompt,
            temperature=0.1,
            max_tokens=800
        )
        
        # JSON 응답 파싱
        try:
            result = json.loads(response)
            return result.get("medicines", {})
        except json.JSONDecodeError:
            print("⚠️ 상세 정보 추출 결과를 JSON으로 파싱할 수 없음")
            return {}
            
    except Exception as e:
        print(f"❌ 상세 정보 추출 중 오류 발생: {e}")
        return {}

def extract_medicine_context(conversation_context: str, medicine_name: str) -> str:
    """대화 맥락에서 특정 약품 주변의 문맥을 추출"""
    # config에서 설정된 최대 문맥 길이 사용
    context_length = 100 # 예시 값, 실제 사용 시 환경 변수나 설정에서 가져옴
    
    pattern = rf".{{0,{context_length}}}{re.escape(medicine_name)}.{{0,{context_length}}}"
    matches = re.findall(pattern, conversation_context, re.IGNORECASE)
    
    if matches:
        return matches[0]
    return ""

def extract_effect_from_context(context: str) -> str:
    """문맥에서 효능 정보를 추출"""
    effect_keywords = ["효능", "효과", "도움", "개선", "완화", "치료", "예방"]
    
    for keyword in effect_keywords:
        if keyword in context:
            # 키워드 주변 문맥 추출
            start = max(0, context.find(keyword) - 50)
            end = min(len(context), context.find(keyword) + 100)
            return context[start:end].strip()
    
    return "효능 정보를 찾을 수 없습니다"

def extract_side_effects_from_context(context: str) -> str:
    """문맥에서 부작용 정보를 추출"""
    side_effect_keywords = ["부작용", "주의사항", "경고", "증상", "불편"]
    
    for keyword in side_effect_keywords:
        if keyword in context:
            start = max(0, context.find(keyword) - 50)
            end = min(len(context), context.find(keyword) + 100)
            return context[start:end].strip()
    
    return "부작용 정보를 찾을 수 없습니다"

def extract_usage_from_context(context: str) -> str:
    """문맥에서 사용법 정보를 추출"""
    usage_keywords = ["사용법", "복용법", "용법", "복용", "섭취", "투여"]
    
    for keyword in usage_keywords:
        if keyword in context:
            start = max(0, context.find(keyword) - 50)
            end = min(len(context), context.find(keyword) + 100)
            return context[start:end].strip()
    
    return "사용법 정보를 찾을 수 없습니다"

def generate_final_answer_node(state: QAState) -> QAState:
    print("🎯 최종 답변 생성 노드 시작")
    print(f"📊 상태 정보:")
    print(f"  - final_answer: {state.get('final_answer', '없음')}")
    print(f"  - recommendation_answer: {state.get('recommendation_answer', '없음')}")
    print(f"  - relevant_docs: {len(state.get('relevant_docs', []))}개")
    print(f"  - external_parsed: {state.get('external_parsed', '없음')}")
    print(f"  - sns_results: {len(state.get('sns_results', []))}개")
    print(f"  - sns_analysis: {state.get('sns_analysis', '없음')}")
    print(f"  - conversation_context: {state.get('conversation_context', '없음')[:100] if state.get('conversation_context') else '없음'}...")
    print(f"  - user_context: {state.get('user_context', '없음')}")
    
    # ✅ 이미 final_answer가 설정된 경우 (최신 정보 요청 등)
    if state.get("final_answer"):
        print("✅ 이미 final_answer가 설정되어 있음")
        return state

    # ✅ 향상된 RAG 답변이 있는 경우 먼저 반환하고 종료 (최고 우선순위)
    enhanced_answer = state.get("enhanced_rag_answer")
    print(f"🔍 enhanced_rag_answer 확인: {enhanced_answer is not None}, 길이: {len(enhanced_answer) if enhanced_answer else 0}")
    if enhanced_answer:
        # 환각이 발견된 경우 추가 정보 섹션 수정 (제거하지 않고 경고 메시지 추가)
        hallucination_flag = state.get("hallucination_flag")
        hallucination_details = state.get("hallucination_details", {})
        if hallucination_flag is True:
            print("⚠️ 환각이 발견되어 추가 정보 섹션에 경고 메시지 추가 중...")
            import re
            # "💡 추가 정보" 또는 "📰 최신 정보" 섹션 찾기
            additional_info_pattern = r'(💡\s*\*\*추가 정보\*\*|📰\s*\*\*최신 정보\*\*)(.*?)(?=\n\n(?:💡|🏥|📋|⚠️|💊|✅|❌|$))'
            def add_warning(match):
                section_header = match.group(1)
                section_content = match.group(2)
                # 섹션 내용 끝에 경고 메시지 추가
                warning = "\n\n⚠️ **참고**: 위 정보 중 일부는 수집된 자료에서 명시적으로 확인되지 않았거나 일반적인 약리학 지식일 수 있습니다."
                return section_header + section_content + warning
            
            modified_answer = re.sub(additional_info_pattern, add_warning, enhanced_answer, flags=re.DOTALL)
            if modified_answer != enhanced_answer:
                print("✅ 추가 정보 섹션에 경고 메시지 추가 완료")
                enhanced_answer = modified_answer
            else:
                # 패턴이 매칭되지 않으면 다른 패턴 시도
                additional_info_pattern2 = r'(💡.*?추가 정보|📰.*?최신 정보)(.*?)(?=\n\n|$)'
                modified_answer = re.sub(additional_info_pattern2, add_warning, enhanced_answer, flags=re.DOTALL)
                if modified_answer != enhanced_answer:
                    print("✅ 추가 정보 섹션에 경고 메시지 추가 완료 (패턴2)")
                    enhanced_answer = modified_answer
        
        print("✅ enhanced_rag_answer 사용")
        state["final_answer"] = enhanced_answer
        return state
    
    # ✅ 약품 사용 가능성 판단이 있는 경우 (기존 방식)
    if state.get("usage_check_answer"):
        print("✅ usage_check_answer 사용")
        state["final_answer"] = state["usage_check_answer"]
        return state

    # ✅ 병력 기반 추천이 있는 경우 반환 (우선순위 2순위)
    if state.get("recommendation_answer"):
        print("✅ recommendation_answer 사용")
        state["final_answer"] = state["recommendation_answer"]
        return state

    # ✅ 신약 검색 결과가 있는 경우 (신약 관련 질문 전용)
    sns_results = state.get("sns_results", [])
    sns_analysis = state.get("sns_analysis", {})
    routing_decision = state.get("routing_decision", "")
    current_query = state.get("query", "") or state.get("original_query", "")
    
    if sns_results and len(sns_results) > 0 and routing_decision == "new_medicine_search":
        print("✅ 신약 검색 결과 처리 시작")
        print(f"📊 신약 검색 결과: {len(sns_results)}개")
        
        # 신약 검색 결과를 기반으로 답변 생성
        try:
            # YouTube와 네이버 뉴스 결과 분리
            youtube_docs = [doc for doc in sns_results if doc.metadata.get("source") == "youtube"]
            news_docs = [doc for doc in sns_results if doc.metadata.get("source") == "naver_news"]
            
            # 검색 의도 확인
            intent = sns_analysis.get("intent", "latest_info")
            detected_drugs = sns_analysis.get("potential_drugs", [])
            
            # 신약 관련 답변 생성 프롬프트
            new_medicine_prompt = f"""{get_role_definition("expert")} 신약 관련 최신 정보를 사용자에게 친근하고 전문적으로 제공하세요.

**사용자 질문:**
{current_query}

**검색 의도:**
{intent}

**감지된 약품/키워드:**
{', '.join(detected_drugs) if detected_drugs else '없음'}

**📺 YouTube 영상 정보 ({len(youtube_docs)}개):**
"""
            
            # YouTube 영상 정보 수집 (링크 포함)
            youtube_info_list = []
            for i, doc in enumerate(youtube_docs[:5], 1):
                title = doc.metadata.get("title", "제목 없음")
                summary = doc.metadata.get("summary", doc.page_content[:300])
                channel = doc.metadata.get("channel_title", "")
                video_id = doc.metadata.get("video_id", "")
                video_url = f"https://www.youtube.com/watch?v={video_id}" if video_id else ""
                
                new_medicine_prompt += f"\n{i}. **{title}**\n"
                if channel:
                    new_medicine_prompt += f"   채널: {channel}\n"
                new_medicine_prompt += f"   내용: {summary[:500]}...\n"
                
                youtube_info_list.append({
                    "title": title,
                    "channel": channel,
                    "url": video_url,
                    "summary": summary
                })
            
            new_medicine_prompt += f"\n**📰 네이버 뉴스 정보 ({len(news_docs)}개):**\n"
            
            # 네이버 뉴스 정보 수집 (링크 포함)
            news_info_list = []
            for i, doc in enumerate(news_docs[:10], 1):
                title = doc.metadata.get("title", "제목 없음")
                description = doc.page_content.split("\n내용: ")[1] if "\n내용: " in doc.page_content else doc.page_content[:200]
                pub_date = doc.metadata.get("pub_date", "")
                link = doc.metadata.get("link", "") or doc.metadata.get("original_link", "")
                
                new_medicine_prompt += f"\n{i}. **{title}**\n"
                if pub_date:
                    new_medicine_prompt += f"   발행일: {pub_date}\n"
                new_medicine_prompt += f"   내용: {description[:400]}...\n"
                
                news_info_list.append({
                    "title": title,
                    "pub_date": pub_date,
                    "url": link,
                    "description": description
                })
            
            new_medicine_prompt += f"""
**답변 요구사항:**
1. 위의 영상과 뉴스 정보를 종합하여 신약 관련 최신 정보를 제공
2. {PromptConfig.COMMON_INSTRUCTIONS['natural_tone']}
3. 이모지로 섹션을 나누어서 답변:
   - 📰 **최신 뉴스**: 뉴스 정보를 종합하여 신약의 개발 현황, 승인 상태, 출시 소식 등을 상세히 설명 (최소 {PromptConfig.MIN_NEWS_SECTION_LENGTH}자)
   - 📺 **관련 영상 정보**: 영상 내용을 바탕으로 전문가 의견, 임상 결과, 환자 경험 등을 풍부하게 설명 (최소 {PromptConfig.MIN_VIDEO_SECTION_LENGTH}자)
   - {PromptConfig.SECTION_STRUCTURE['summary']}: 전체 정보를 요약하여 핵심 포인트를 정리 ({PromptConfig.MIN_SUMMARY_LENGTH}-{PromptConfig.MAX_SUMMARY_LENGTH}자)
4. 출처는 자연스럽게 언급하되, 플랫폼명(YouTube, 네이버 뉴스 등)은 언급하지 마세요
   - 예: "최근 뉴스에 따르면...", "전문가 의견에 따르면...", "전문가들은...", "연구 결과에 따르면..."
   - ❌ "YouTube 영상에서...", "네이버 뉴스에 따르면..." 같은 표현 금지
5. 신약의 개발 현황, 승인 상태, 임상 시험 결과, 출시 일정 등을 포함
6. 최소 {PromptConfig.MIN_DETAILED_ANSWER_LENGTH}자 이상의 상세하고 풍부한 답변 작성

**중요:**
- 효능 및 작용원리, 주의사항 섹션은 포함하지 마세요 (신약 정보 중심)
- 모든 정보를 자연스럽게 통합하여 설명
- 사용자가 궁금해하는 신약 정보를 중심으로 답변
- 최신 정보임을 강조하되, 과장하지 않기
- 구체적인 뉴스 제목과 영상 제목을 자연스럽게 언급
- 플랫폼명(YouTube, 네이버 뉴스 등)은 절대 언급하지 마세요
"""
            
            final_answer = generate_response_llm_from_prompt(
                prompt=new_medicine_prompt,
                temperature=0.7,
                max_tokens=2500
            )
            
            # 관련 링크 섹션 추가
            links_section = "\n\n---\n\n🔗 **관련 링크**\n\n"
            
            # 뉴스 링크
            if news_info_list:
                links_section += "📰 **관련 뉴스:**\n"
                for news in news_info_list[:5]:  # 상위 5개만
                    if news["url"]:
                        links_section += f"- [{news['title']}]({news['url']})"
                        if news.get("pub_date"):
                            links_section += f" ({news['pub_date']})"
                        links_section += "\n"
                links_section += "\n"
            
            # YouTube 영상 링크
            if youtube_info_list:
                links_section += "📺 **관련 영상:**\n"
                for video in youtube_info_list[:5]:  # 상위 5개만
                    if video["url"]:
                        links_section += f"- [{video['title']}]({video['url']})"
                        if video.get("channel"):
                            links_section += f" - {video['channel']}"
                        links_section += "\n"
            
            final_answer += links_section
            
            state["final_answer"] = final_answer
            print("✅ 신약 검색 결과 기반 답변 생성 완료")
            return state
            
        except Exception as e:
            print(f"❌ 신약 검색 결과 처리 중 오류 발생: {e}")
            # 오류 발생 시 기본 처리로 넘어감

    # 🔍 LLM 기반 맥락 분석 및 답변 생성
    conversation_context = state.get("conversation_context", "")
    user_context = state.get("user_context", "")
    if not current_query:  # 위에서 정의되지 않았다면 여기서 정의
        current_query = state.get("query", "") or state.get("original_query", "")
    relevant_docs = state.get("relevant_docs", [])
    
    if conversation_context and current_query:
        print("🔄 LLM이 맥락을 분석하여 답변 생성")
        
        # LLM에게 맥락 기반 답변 생성 요청
        context_aware_prompt = f"""
{get_role_definition("pharmacist")} 
사용자와 자연스러운 대화를 나누며 의약품에 대한 정확한 정보를 제공합니다.

**현재 사용자 질문:**
{current_query}

**이전 대화 맥락:**
{conversation_context[:1000] if conversation_context else "없음"}

**관련 문서 정보:**
{len(relevant_docs)}개의 관련 문서가 수집되었습니다. 이 문서들에는 Excel DB, PDF, PubChem, YouTube, 네이버 뉴스 등 다양한 소스의 정보가 포함되어 있습니다.

**⚠️ 매우 중요: 정보 수집 원칙**
- 이전 대화 맥락과 관련 문서를 모두 확인하세요
- 각 문서에서 발견한 모든 고유한 정보를 빠짐없이 포함하세요
- 비슷한 정보라도 각 소스의 표현이나 추가 세부사항이 다를 수 있으므로, 모든 정보를 종합하세요

**답변 스타일:**
- 마치 친구나 가족과 대화하듯 자연스럽고 친근하게
- "아, 그거 궁금하시군요!", "좋은 질문이에요!" 같은 자연스러운 반응
- 전문적이지만 이해하기 쉬운 설명
- 필요시 "더 궁금한 게 있으시면 언제든 물어보세요!" 같은 마무리

**답변 요구사항:**
1. 이전 대화의 맥락을 정확히 파악하고 연결
2. 사용자의 구체적인 질문에 직접적으로 답변
3. 대명사나 모호한 표현이 있다면 맥락에서 추론하여 명확히 해석
4. 티키타카가 가능한 대화형 답변
5. {PromptConfig.MIN_CONVERSATIONAL_LENGTH}-{PromptConfig.MAX_CONVERSATIONAL_LENGTH}자 정도의 적절한 길이
6. **관련 문서의 모든 정보를 빠짐없이 확인하고 활용하세요**

**중요:**
- 이전 대화에서 언급된 약품이나 성분이 있다면 그것을 기반으로 답변
- 사용자가 특정 성분에 대해 물어봤다면 그 성분에만 집중해서 답변
- 불필요하게 모든 정보를 다 나열하지 말고 질문에 맞는 정보만 제공
- **하지만 질문에 관련된 모든 소스의 정보는 빠짐없이 포함하세요**
"""
        
        try:
            # LLM이 맥락을 이해하고 자연스러운 답변 생성
            final_answer = generate_response_llm_from_prompt(
                prompt=context_aware_prompt,
                temperature=0.7,  # 자연스러운 대화를 위해 적당한 temperature
                max_tokens=1000
            )
            
            state["final_answer"] = final_answer
            print("✅ LLM 기반 맥락 인식 답변 생성 완료")
            
        except Exception as e:
            print(f"❌ LLM 답변 생성 중 오류 발생: {e}")
            # 오류 발생 시 기본 답변
            state["final_answer"] = f"죄송합니다. 답변을 생성하는 중 오류가 발생했습니다: {str(e)}"
    
    else:
        print("❌ 대화 맥락 정보가 부족하여 답변을 생성할 수 없음")
        state["final_answer"] = "죄송합니다. 대화 맥락 정보가 부족하여 적절한 답변을 생성할 수 없습니다."
    
    return state
