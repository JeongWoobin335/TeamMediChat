# new_medicine_search_node.py - 신약 관련 질문 전용 검색 노드
# PLUS 폴더의 sns_node.py를 기반으로 생성
# 기존 sns_search_node는 약품 사용 가능성 판단의 보조 정보 수집용으로 그대로 유지

import os
import re
from typing import List, Dict, Optional
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qa_state import QAState
from medical_patterns import *
from dotenv import load_dotenv
from answer_utils import generate_response_llm_from_prompt

# 환경 변수 로드
load_dotenv()

# 네이버 뉴스 API
from naver_news_api import NaverNewsAPI

# ==================== API 설정 함수 ====================

def setup_youtube_api():
    """유튜브 API 설정"""
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY가 .env 파일에 설정되지 않았습니다.")
    return api_key

# ==================== 유튜브 검색 함수 ====================

def extract_keywords(text: str) -> List[str]:
    """텍스트에서 키워드 추출"""
    keywords = re.findall(r'\b\w+\b', text.lower())
    return keywords

def search_youtube_videos(query: str, max_videos: int = 10) -> List[Dict]:
    """유튜브에서 약품 관련 영상 검색"""
    try:
        api_key = setup_youtube_api()
        videos = []
        
        # 유튜브 검색 API 엔드포인트
        search_url = "https://www.googleapis.com/youtube/v3/search"
        
        # 검색 파라미터
        params = {
            'part': 'snippet',
            'q': query,
            'key': api_key,
            'maxResults': max_videos,
            'type': 'video',
            'relevanceLanguage': 'ko',  # 한국어 우선
            'videoDuration': 'medium',  # 중간 길이 영상 (5-20분)
            'order': 'relevance'
        }
        
        # 검색 요청
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        
        search_results = response.json()
        
        if 'items' not in search_results:
            print(f"❌ 검색 결과가 없습니다: {query}")
            return []
        
        for item in search_results['items']:
            snippet = item['snippet']
            video_id = item['id']['videoId']
            
            # 영상 정보 추출
            video_info = {
                "title": snippet['title'],
                "description": snippet['description'],
                "channel_title": snippet['channelTitle'],
                "published_at": snippet['publishedAt'],
                "video_id": video_id,
                "thumbnail": snippet['thumbnails']['medium']['url'],
                "source": "youtube",
                "keywords": extract_keywords(snippet['title'] + " " + snippet['description'])
            }
            
            videos.append(video_info)
        
        print(f"✅ '{query}' 검색 결과: {len(videos)}개 영상")
        return videos
        
    except Exception as e:
        print(f"❌ 유튜브 검색 실패: {e}")
        return []

def get_video_transcript(video_id: str) -> str:
    """유튜브 영상의 자막/내용 가져오기"""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
        
        # YouTubeTranscriptApi 인스턴스 생성
        ytt_api = YouTubeTranscriptApi()
        
        # 방법 1: fetch 메서드로 직접 가져오기 (가장 간단한 방법)
        try:
            transcript = ytt_api.fetch(video_id, languages=['ko', 'en'])
            
            if transcript:
                # 자막 텍스트를 하나로 합치기
                # transcript는 FetchedTranscript 객체이고, 각 item은 FetchedTranscriptSnippet 객체
                full_transcript = ""
                for snippet in transcript:
                    # FetchedTranscriptSnippet 객체는 text 속성을 가짐
                    if hasattr(snippet, 'text'):
                        full_transcript += snippet.text + " "
                    elif isinstance(snippet, dict) and 'text' in snippet:
                        full_transcript += snippet['text'] + " "
                    elif isinstance(snippet, str):
                        full_transcript += snippet + " "
                
                if full_transcript.strip():
                    print(f"✅ 영상 {video_id} 자막 추출 성공: {len(full_transcript)}자")
                    return full_transcript.strip()
                else:
                    return ""
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
            # 자막이 비활성화되었거나 없는 경우
            print(f"⚠️ 영상 {video_id} 자막 없음: {type(e).__name__}")
            return ""
        except Exception as e1:
            # 다른 예외 발생 시 list 메서드로 시도
            print(f"⚠️ fetch 실패, list 메서드로 시도: {type(e1).__name__}")
            
            # 방법 2: list 메서드로 자막 목록 가져온 후 선택
            try:
                transcript_list = ytt_api.list(video_id)
                
                # 한국어 자막 우선, 없으면 영어 자막
                transcript = transcript_list.find_transcript(['ko', 'en'])
                transcript_data = transcript.fetch()
                
                if transcript_data:
                    # 자막 텍스트를 하나로 합치기
                    # transcript_data는 FetchedTranscript 객체
                    full_transcript = ""
                    for snippet in transcript_data:
                        # FetchedTranscriptSnippet 객체는 text 속성을 가짐
                        if hasattr(snippet, 'text'):
                            full_transcript += snippet.text + " "
                        elif isinstance(snippet, dict) and 'text' in snippet:
                            full_transcript += snippet['text'] + " "
                        elif isinstance(snippet, str):
                            full_transcript += snippet + " "
                    
                    if full_transcript.strip():
                        print(f"✅ 영상 {video_id} 자막 추출 성공 (list): {len(full_transcript)}자")
                        return full_transcript.strip()
                    else:
                        return ""
            except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e2:
                print(f"⚠️ 영상 {video_id} 자막 없음 (list): {type(e2).__name__}")
                return ""
            except Exception as e2:
                print(f"❌ 자막 가져오기 실패 (list): {type(e2).__name__}: {e2}")
                return ""
        
        print(f"⚠️ 영상 {video_id} 자막이 없습니다")
        return ""
            
    except Exception as e:
        print(f"❌ 자막 가져오기 실패: {type(e).__name__}: {e}")
        return ""

# ==================== 요약 및 분석 함수 ====================

def summarize_video_content(content: str, max_length: int = 500) -> str:
    """영상 내용을 요약"""
    try:
        if len(content) <= max_length:
            return content
        
        # 텍스트 분할기 사용
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(content)
        
        # 첫 번째 청크와 마지막 청크를 사용하여 요약
        if len(chunks) >= 2:
            summary = chunks[0][:max_length//2] + "...\n\n" + chunks[-1][:max_length//2]
        else:
            summary = chunks[0][:max_length]
        
        return summary
        
    except Exception as e:
        print(f"❌ 내용 요약 실패: {e}")
        return content[:max_length] if len(content) > max_length else content

def extract_disease_name_with_llm(query: str) -> Optional[str]:
    """LLM을 사용하여 질문에서 질병명 추출"""
    try:
        extraction_prompt = f"""다음 질문에서 신약과 관련된 질병명을 추출해주세요.

**질문:** {query}

**지시사항:**
1. 질문에서 질병명(예: 치매, 당뇨, 당뇨병, 고혈압, 암, 알츠하이머 등)을 찾아주세요.
2. "그럼", "그리고", "또한", "관련", "대한", "에 관한", "에 대한", "정보", "뉴스", "알려줘" 같은 조사나 일반 단어는 무시하세요.
3. 질병명만 추출하고, 다른 단어는 포함하지 마세요.
4. 질병명을 찾을 수 없으면 "없음"이라고 답하세요.

**응답 형식:**
질병명만 답하세요. 예: 치매, 당뇨, 고혈압, 없음
"""
        
        response = generate_response_llm_from_prompt(
            prompt=extraction_prompt,
            temperature=0.1,
            max_tokens=50,
            cache_type="disease_extraction",
            use_cache=True
        )
        
        # 응답 정리
        disease_name = response.strip()
        
        # "없음" 또는 빈 문자열 체크
        if not disease_name or disease_name.lower() in ["없음", "none", "없어", "찾을 수 없음"]:
            return None
        
        # 불필요한 설명 제거 (예: "질병명: 치매" → "치매")
        if ":" in disease_name:
            disease_name = disease_name.split(":")[-1].strip()
        
        # 공백 제거
        disease_name = disease_name.strip()
        
        # 너무 긴 경우 (잘못된 추출 가능성) 제외
        if len(disease_name) > 10:
            print(f"⚠️ 추출된 질병명이 너무 깁니다: '{disease_name}', 무시합니다.")
            return None
        
        return disease_name if disease_name else None
        
    except Exception as e:
        print(f"❌ LLM 질병명 추출 중 오류 발생: {e}")
        return None

def analyze_query_intent(query: str) -> Dict[str, any]:
    """쿼리의 의도와 핵심 요소를 점수 기반으로 분석 (PLUS 개선 버전)"""
    query_lower = query.lower()
    
    # 1. 의도별 점수 계산
    intent_scores = {
        "pain_relief": 0,
        "discomfort_relief": 0,
        "side_effect": 0,
        "experience_review": 0,
        "efficacy": 0,
        "latest_info": 0,
        "general_info": 0
    }
    
    # 통증 관련 의도 점수
    for pattern in PAIN_PATTERNS:
        if re.search(pattern, query_lower):
            intent_scores["pain_relief"] += 3
            if re.search(r'너무|매우|정말|엄청|심하게', query_lower):
                intent_scores["pain_relief"] += 2
    
    # 불편함 관련 의도 점수
    for pattern in DISCOMFORT_PATTERNS:
        if re.search(pattern, query_lower):
            intent_scores["discomfort_relief"] += 3
    
    # 부작용 관련 의도 점수
    for pattern in SIDE_EFFECT_PATTERNS:
        if re.search(pattern, query_lower):
            intent_scores["side_effect"] += 5
            if re.search(r'부작용|나빠졌어|악화|새로\s*생겼어', query_lower):
                intent_scores["side_effect"] += 2
    
    # 경험담 관련 의도 점수
    for pattern in EXPERIENCE_PATTERNS:
        if re.search(pattern, query_lower):
            intent_scores["experience_review"] += 3
            if re.search(r'경험담|후기|경험|사용후기|복용후기', query_lower):
                intent_scores["experience_review"] += 1
    
    # 효능 관련 의도 점수
    for pattern in EFFICACY_PATTERNS:
        if re.search(pattern, query_lower):
            intent_scores["efficacy"] += 3
    
    # 최신 정보 관련 의도 점수 (신약 관련 질문에 중요)
    for pattern in LATEST_PATTERNS:
        if re.search(pattern, query_lower):
            intent_scores["latest_info"] += 3
            if re.search(r'2024|2023|새로|신약', query_lower):
                intent_scores["latest_info"] += 1
    
    # 일반 정보 기본 점수
    intent_scores["general_info"] = 1
    
    # 2. 가장 높은 점수의 의도 선택
    intent = max(intent_scores, key=intent_scores.get)
    
    # 3. 부작용 의도가 있는 경우 우선순위 조정
    print(f"🔍 부작용 키워드 체크: '부작용' in '{query_lower}' = {'부작용' in query_lower}")
    if "부작용" in query_lower:
        print(f"✅ 부작용 키워드 발견! 현재 의도 점수: {intent_scores}")
        if intent_scores["side_effect"] > 0 and intent_scores["experience_review"] > 0:
            if intent_scores["side_effect"] >= intent_scores["experience_review"]:
                intent = "side_effect"
                print(f"🎯 부작용 의도로 설정 (점수 비교)")
            else:
                intent = "side_effect_experience"
                print(f"🎯 복합 의도로 설정: side_effect_experience")
        elif intent_scores["side_effect"] > 0:
            intent = "side_effect"
            print(f"🎯 부작용 의도로 설정 (기존 점수)")
        else:
            intent = "side_effect"
            intent_scores["side_effect"] = 6
            print(f"🎯 부작용 의도로 강제 설정 (키워드 기반)")
    else:
        print(f"❌ 부작용 키워드 없음")
    
    # 4. 핵심 키워드 추출 (LLM 기반 질병명 추출)
    potential_drugs = []
    
    # LLM을 사용하여 질병명 추출
    if '신약' in query_lower:
        disease_name = extract_disease_name_with_llm(query)
        
        if disease_name:
            potential_drugs.append(f"{disease_name} 신약")
            print(f"✅ LLM 기반 질병명 추출: '{disease_name} 신약'")
        else:
            potential_drugs.append("신약")
            print(f"⚠️ LLM 질병명 추출 실패, 신약 단독 사용")
    else:
        # 신약 키워드가 없으면 일반적인 약품명 추출 시도
        disease_name = extract_disease_name_with_llm(query)
        if disease_name:
            potential_drugs.append(disease_name)
            print(f"✅ LLM 기반 질병명 추출: '{disease_name}'")
    
    # LLM 기반 추출이 실패한 경우에만 폴백 (하지만 이제는 LLM이 대부분 처리)
    if not potential_drugs:
        print(f"⚠️ LLM 기반 추출 실패, potential_drugs가 비어있음")
    
    # 5. 증상 부위/성격 추출
    body_parts = []
    
    for part_name, patterns in BODY_PART_PATTERNS.items():
        if any(re.search(pattern, query_lower) for pattern in patterns):
            body_parts.append(part_name)
    
    # 6. 증상 강도/성격
    intensity = "moderate"
    
    for intensity_level, patterns in INTENSITY_PATTERNS.items():
        if any(re.search(pattern, query_lower) for pattern in patterns):
            intensity = intensity_level
            break
    
    return {
        "intent": intent,
        "intent_scores": intent_scores,
        "potential_drugs": potential_drugs,
        "body_parts": body_parts,
        "intensity": intensity,
        "original_query": query
    }

def create_search_terms(analysis: Dict[str, any]) -> List[str]:
    """분석 결과를 바탕으로 검색어 생성 (신약 관련 질문 전용)"""
    search_terms = []
    
    intent = analysis.get("intent")
    potential_drugs = analysis.get("potential_drugs", [])
    body_parts = analysis.get("body_parts", [])
    intensity = analysis.get("intensity")
    
    # 1. 핵심 키워드 기반 검색어 생성 (신약 관련 질문 전용)
    if potential_drugs:
        for keyword in potential_drugs[:2]:  # "치매 신약" 같은 복합 키워드 그대로 사용
            if intent == "latest_info":
                # 신약 관련 최신 정보 검색
                search_terms.extend([
                    f"{keyword}",
                    f"{keyword} 뉴스",
                    f"{keyword} 최신",
                    f"{keyword} 개발",
                    f"{keyword} 승인",
                    f"{keyword} 출시"
                ])
            elif intent == "side_effect":
                search_terms.extend([
                    f"{keyword} 부작용",
                    f"{keyword} 부작용 경험"
                ])
            elif intent == "experience_review":
                search_terms.extend([
                    f"{keyword} 경험담",
                    f"{keyword} 사용 후기"
                ])
            elif intent == "efficacy":
                search_terms.extend([
                    f"{keyword} 효과",
                    f"{keyword} 효능"
                ])
    
    # 2. 일반적인 의도별 검색어 (핵심 키워드가 없을 때만 제한적으로 사용)
    if not potential_drugs:
        if intent == "latest_info":
            search_terms.extend([
                "신약 승인",
                "신약 출시",
                "신약 개발"
            ])
        elif intent == "side_effect":
            search_terms.extend([
                "신약 부작용"
            ])
        elif intent == "experience_review":
            search_terms.extend([
                "신약 경험담"
            ])
    
    # 3. 부위별 검색어 추가
    if body_parts:
        for part in body_parts:
            if intent == "side_effect":
                search_terms.append(f"{part} 부작용 경험")
            elif intent == "experience_review":
                search_terms.append(f"{part} 치료 경험")
    
    # 4. 강도에 따른 검색어
    if intensity == "severe":
        search_terms.append("심한 부작용 경험")
    elif intensity == "mild":
        search_terms.append("가벼운 부작용 경험")
    
    # 중복 제거 및 우선순위 정렬
    unique_terms = list(dict.fromkeys(search_terms))
    
    # ✅ 약품명이 포함된 검색어를 우선으로 정렬
    if potential_drugs:
        drug_terms = [t for t in unique_terms if any(drug in t for drug in potential_drugs)]
        other_terms = [t for t in unique_terms if not any(drug in t for drug in potential_drugs)]
        unique_terms = drug_terms + other_terms[:2]  # 약품명 검색어 + 일반 검색어 최대 2개만
    else:
        # 약품명이 없으면 검색어를 더 제한
        unique_terms = unique_terms[:3]
    
    print(f"📊 최종 검색어 목록 (우선순위 정렬): {unique_terms[:8]}")
    return unique_terms[:8]  # 검색어를 8개로 제한

def filter_relevant_videos(videos: List[Dict], analysis: Dict[str, any]) -> List[Dict]:
    """원본 질문과 관련성에 따라 영상 필터링"""
    relevant_videos = []
    
    intent = analysis.get("intent")
    potential_drugs = analysis.get("potential_drugs", [])
    body_parts = analysis.get("body_parts", [])
    
    for video in videos:
        content_lower = (video["title"] + " " + video["description"]).lower()
        relevance_score = 0
        
        # 1. 의도별 관련성 점수
        if intent == "side_effect":
            side_effect_keywords = ['부작용', 'adverse', 'negative', 'problem', 'issue', 'trouble', 'bad', 'unwanted', 'reaction']
            if any(keyword in content_lower for keyword in side_effect_keywords):
                relevance_score += 3
            else:
                continue
        
        elif intent == "experience_review":
            experience_keywords = ['경험', 'review', '후기', 'testimonial', 'story', 'used', '복용', '사용', 'took', 'tried']
            if any(keyword in content_lower for keyword in experience_keywords):
                relevance_score += 3
            else:
                continue
        
        elif intent == "latest_info":
            latest_keywords = ['신약', '새로운', '최신', '개발', '승인', '상륙', '출시', 'new', 'latest', 'development']
            if any(keyword in content_lower for keyword in latest_keywords):
                relevance_score += 3
            else:
                continue
        
        # 2. 약품명 관련성
        if potential_drugs:
            for drug in potential_drugs:
                if drug.lower() in content_lower:
                    relevance_score += 4
                    break
        
        # 3. 부위 관련성
        if body_parts:
            for part in body_parts:
                if part in content_lower:
                    relevance_score += 2
        
        # 4. 제목 관련성 점수
        if any(keyword in video["title"].lower() for keyword in ['약', '감기', 'cold', 'flu', 'medicine', 'drug', '신약', '치료']):
            relevance_score += 1
        
        # 관련성 점수가 일정 수준 이상인 영상만 포함
        if relevance_score >= 3:
            video["relevance_score"] = relevance_score
            relevant_videos.append(video)
    
    # 관련성 점수 순으로 정렬
    relevant_videos.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    # 최대 5개로 제한
    return relevant_videos[:5]

def filter_relevant_news(news_items: List[Dict], analysis: Dict[str, any]) -> List[Dict]:
    """원본 질문과 관련성에 따라 네이버 뉴스 필터링 (PLUS 개선 버전)"""
    relevant_news = []
    
    intent = analysis.get("intent")
    potential_drugs = analysis.get("potential_drugs", [])
    body_parts = analysis.get("body_parts", [])
    
    print(f"\n🔍 뉴스 필터링 시작")
    print(f"   - 약품명: {potential_drugs}")
    print(f"   - 의도: {intent}")
    print(f"   - 총 뉴스 수: {len(news_items)}")
    
    for idx, news in enumerate(news_items, 1):
        title = news.get("title", "")
        description = news.get("description", "")
        title_lower = title.lower()
        desc_lower = description.lower()
        content_lower = title_lower + " " + desc_lower
        relevance_score = 0
        score_details = []  # 점수 상세 정보
        
        # 1. 약품명 체크 (약품명이 있는 경우만 필수)
        drug_mentioned = False
        if potential_drugs:
            for drug in potential_drugs:
                # 제목에 약품명이 있으면 높은 점수
                if drug.lower() in title_lower:
                    relevance_score += 10
                    drug_mentioned = True
                    score_details.append(f"약품명(제목):+10")
                    break
                # 내용에만 있으면 중간 점수
                elif drug.lower() in desc_lower:
                    relevance_score += 5
                    drug_mentioned = True
                    score_details.append(f"약품명(내용):+5")
                    break
            
            # 약품명이 있는 쿼리인데 기사에 없으면 의학 관련이면 약간의 점수
            if not drug_mentioned:
                medical_general = ['약', '의약품', '제약', '성분', '복용', '처방']
                if any(kw in content_lower for kw in medical_general):
                    relevance_score += 2
                    score_details.append("의학관련:+2")
                else:
                    score_details.append("약품명없음:제외")
                    print(f"  [{idx}] ❌ 제외 (약품명 없음): {title[:40]}...")
                    continue
        else:
            # 약품명이 없는 쿼리면 기본 점수
            relevance_score += 3
            score_details.append("기본:+3")
        
        # 2. 의도별 관련성 점수 (완화)
        intent_matched = False
        if intent == "side_effect":
            side_effect_keywords = ['부작용', '이상반응', '위험', '주의', '경고', '리콜', '문제']
            matched_keywords = [kw for kw in side_effect_keywords if kw in content_lower]
            if matched_keywords:
                score = len(matched_keywords) * 3
                relevance_score += score
                intent_matched = True
                score_details.append(f"부작용키워드({len(matched_keywords)}):+{score}")
        
        elif intent == "experience_review":
            experience_keywords = ['사용', '복용', '효과', '결과', '사례', '임상', '후기', '경험']
            matched_keywords = [kw for kw in experience_keywords if kw in content_lower]
            if matched_keywords:
                score = len(matched_keywords) * 2
                relevance_score += score
                intent_matched = True
                score_details.append(f"경험키워드({len(matched_keywords)}):+{score}")
        
        elif intent == "latest_info":
            latest_keywords = ['신약', '새로운', '최신', '개발', '승인', '출시', '론칭', '허가', '발매']
            matched_keywords = [kw for kw in latest_keywords if kw in content_lower]
            if matched_keywords:
                score = len(matched_keywords) * 3
                relevance_score += score
                intent_matched = True
                score_details.append(f"최신키워드({len(matched_keywords)}):+{score}")
        
        elif intent == "efficacy":
            efficacy_keywords = ['효능', '효과', '작용', '치료', '개선', '완화', '임상', '도움']
            matched_keywords = [kw for kw in efficacy_keywords if kw in content_lower]
            if matched_keywords:
                score = len(matched_keywords) * 2
                relevance_score += score
                intent_matched = True
                score_details.append(f"효능키워드({len(matched_keywords)}):+{score}")
        
        # 의도 키워드가 없어도 의학 관련이면 약간 가산
        if not intent_matched:
            medical_keywords = ['의약품', '제약', '성분', '약국', '의사', '병원', '환자', '질환']
            matched_medical = [kw for kw in medical_keywords if kw in content_lower]
            if matched_medical:
                relevance_score += 2
                score_details.append(f"의학키워드:+2")
        
        # 3. 부위 관련성
        if body_parts:
            for part in body_parts:
                if part in content_lower:
                    relevance_score += 2
                    score_details.append(f"부위({part}):+2")
        
        # 4. 무관한 키워드 강력 감점 (주식/투자 관련 강화)
        irrelevant_keywords = ['정치', '선거', '스포츠', '연예', '게임', '주식', '부동산', 
                              '경제전망', '금융시장', '투자', '증권', '코인', '가상화폐',
                              '상장', '주가', '관련주', '특징주', '증시', '시장', '거래',
                              '매수', '매도', '종목', '기업분석', '실적', '배당']
        matched_irrelevant = [kw for kw in irrelevant_keywords if kw in content_lower]
        if matched_irrelevant:
            # 제목에 무관 키워드가 있으면 더 강하게 감점
            if any(kw in title_lower for kw in matched_irrelevant):
                relevance_score -= 15  # 제목에 있으면 더 강하게 감점
                score_details.append(f"무관(제목:{matched_irrelevant[0]}):−15")
            else:
                relevance_score -= 10
                score_details.append(f"무관({matched_irrelevant[0]}):−10")
        
        # 5. 광고성 키워드 감점
        ad_keywords = ['할인', '이벤트', '특가', '프로모션', '쿠폰', '특별가']
        matched_ad = [kw for kw in ad_keywords if kw in content_lower]
        if matched_ad:
            relevance_score -= 5
            score_details.append(f"광고({matched_ad[0]}):−5")
        
        # 6. 핵심 키워드가 제목에 명확히 있는 경우 추가 점수 (신약 관련 질문에 중요)
        if potential_drugs:
            for drug in potential_drugs:
                # 제목에 핵심 키워드가 명확히 포함되어 있으면 추가 점수
                if drug.lower() in title_lower:
                    # 이미 위에서 점수를 받았지만, 더 명확한 매칭인 경우 추가 점수
                    if len(drug.split()) > 1:  # "치매 신약" 같은 복합 키워드
                        relevance_score += 5
                        score_details.append(f"핵심키워드(제목):+5")
        
        # 관련성 점수가 일정 수준 이상인 뉴스만 포함 (임계값 강화: 8점)
        # 핵심 키워드가 제목에 있으면 5점 이상도 허용
        min_score = 8
        if potential_drugs:
            # 핵심 키워드가 제목에 있으면 최소 점수 완화
            for drug in potential_drugs:
                if drug.lower() in title_lower:
                    min_score = 5
                    break
        
        if relevance_score >= min_score:
            news["relevance_score"] = relevance_score
            relevant_news.append(news)
            score_str = ", ".join(score_details)
            print(f"  [{idx}] ✅ 선택 [{relevance_score}점] ({score_str})")
            print(f"        제목: {title[:50]}...")
        else:
            score_str = ", ".join(score_details)
            print(f"  [{idx}] ❌ 제외 [{relevance_score}점] ({score_str})")
            print(f"        제목: {title[:50]}...")
    
    # 관련성 점수 순으로 정렬
    relevant_news.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    # 최대 10개로 제한 (좀 더 많이)
    print(f"\n🎯 필터링 완료: {len(relevant_news)}개 뉴스 중 상위 {min(len(relevant_news), 10)}개 선택")
    return relevant_news[:10]

# ==================== 신약 검색 노드 ====================

def new_medicine_search_node(state: QAState) -> QAState:
    """신약 관련 질문 전용 검색 노드 (유튜브 + 네이버 뉴스 기반) - 영상 내용 추출 및 요약 포함"""
    
    print("🔍 신약 검색 노드 실행 시작 (유튜브 + 네이버 뉴스)")
    
    # 쿼리에서 검색 키워드 추출 (원본 쿼리 우선 사용)
    query = state.get("query", "") or state.get("original_query", "")
    
    print(f"📝 분석할 쿼리: {query}")
    
    if not query:
        print("❌ 쿼리가 없어서 신약 검색 건너뜀")
        state["sns_results"] = []
        state["sns_count"] = 0
        return state
    
    # 1. 쿼리 의도 분석
    print("🧠 쿼리 의도 분석 시작")
    analysis = analyze_query_intent(query)
    print(f"🎯 감지된 의도: {analysis['intent']}")
    print(f"📊 의도 점수: {analysis['intent_scores']}")
    print(f"💊 감지된 약품: {analysis['potential_drugs']}")
    print(f"🦴 감지된 부위: {analysis['body_parts']}")
    
    # 2. 검색어 생성
    print("🔎 검색어 생성 시작")
    search_terms = create_search_terms(analysis)
    print(f"🔍 생성된 검색어: {search_terms}")
    
    all_videos = []
    all_news = []
    
    # 3. 각 검색어로 유튜브 검색
    print("📺 유튜브 검색 시작")
    for search_term in search_terms[:3]:  # 최대 3개 검색어만 사용
        try:
            print(f"🔍 유튜브 '{search_term}' 검색 중...")
            videos = search_youtube_videos(search_term, max_videos=5)
            print(f"📝 '{search_term}' 검색 결과: {len(videos)}개 영상")
            all_videos.extend(videos)
        except Exception as e:
            print(f"❌ 유튜브 '{search_term}' 검색 실패: {e}")
            continue
    
    # 4. 네이버 뉴스 검색 (관련성 우선, 정확도순 + 최신순 혼합)
    potential_drugs = analysis.get("potential_drugs", [])
    print("📰 네이버 뉴스 검색 시작")
    print(f"   감지된 핵심 키워드: {potential_drugs}")
    try:
        naver_api = NaverNewsAPI()
        
        # 핵심 키워드가 있으면 키워드로 검색 (예: "치매 신약")
        if potential_drugs:
            for keyword in potential_drugs[:2]:  # 최대 2개 키워드만 사용
                # 정확도순 검색 (관련성 높은 뉴스 우선)
                print(f"🔍 네이버 뉴스 '{keyword}' 검색 중... (정확도순)")
                news_items_sim = naver_api.search_news(keyword, display=15, sort="sim")
                print(f"📝 '{keyword}' 정확도순 검색 결과: {len(news_items_sim)}개 뉴스")
                all_news.extend(news_items_sim)
                
                # 최신순 검색 (최신 뉴스도 일부 포함)
                print(f"🔍 네이버 뉴스 '{keyword}' 검색 중... (최신순)")
                news_items_date = naver_api.search_news(keyword, display=10, sort="date")
                print(f"📝 '{keyword}' 최신순 검색 결과: {len(news_items_date)}개 뉴스")
                all_news.extend(news_items_date)
        else:
            # 핵심 키워드가 없으면 검색어 사용
            if search_terms:
                # 첫 번째 검색어만 사용
                search_term = search_terms[0]
                print(f"🔍 네이버 뉴스 '{search_term}' 검색 중... (정확도순)")
                news_items = naver_api.search_news(search_term, display=15, sort="sim")
                print(f"📝 '{search_term}' 검색 결과: {len(news_items)}개 뉴스")
                all_news.extend(news_items)
            else:
                print("⚠️ 검색어가 없어 뉴스 검색 건너뜀")
    except Exception as e:
        print(f"❌ 네이버 뉴스 검색 실패: {e}")
    
    print(f"📊 총 수집된 영상: {len(all_videos)}개")
    print(f"📊 총 수집된 뉴스: {len(all_news)}개")
    
    # 중복 뉴스 제거 (link 기준)
    if all_news:
        seen_links = set()
        unique_news = []
        for news in all_news:
            link = news.get("link", "") or news.get("original_link", "")
            if link and link not in seen_links:
                seen_links.add(link)
                unique_news.append(news)
            elif not link:  # link가 없으면 제목으로 중복 체크
                title = news.get("title", "")
                if title not in [n.get("title", "") for n in unique_news]:
                    unique_news.append(news)
        all_news = unique_news
        print(f"📊 중복 제거 후 뉴스: {len(all_news)}개")
    
    # 5. 영상 필터링
    print("🔍 영상 필터링 시작")
    filtered_videos = filter_relevant_videos(all_videos, analysis)
    print(f"✅ 필터링 후 영상: {len(filtered_videos)}개")
    
    # 6. 뉴스 필터링
    print("🔍 뉴스 필터링 시작")
    filtered_news = filter_relevant_news(all_news, analysis)
    print(f"✅ 필터링 후 뉴스: {len(filtered_news)}개")
    
    # 7. 영상 내용 추출 및 요약
    print("📹 영상 내용 추출 및 요약 시작")
    enriched_videos = []
    for video in filtered_videos:
        try:
            # 자막 추출
            transcript = get_video_transcript(video["video_id"])
            
            if transcript:
                # 자막이 있으면 요약
                summarized_content = summarize_video_content(transcript, max_length=800)
                video["transcript"] = transcript
                video["summarized_content"] = summarized_content
                video["has_transcript"] = True
                print(f"✅ 영상 {video['video_id']} 자막 추출 및 요약 완료")
            else:
                # 자막이 없으면 제목과 설명만 사용
                content = f"제목: {video['title']}\n설명: {video['description']}"
                video["transcript"] = ""
                video["summarized_content"] = content
                video["has_transcript"] = False
                print(f"⚠️ 영상 {video['video_id']} 자막 없음, 기본 정보만 사용")
            
            enriched_videos.append(video)
            
        except Exception as e:
            print(f"❌ 영상 {video['video_id']} 내용 추출 실패: {e}")
            # 실패해도 기본 정보는 포함
            content = f"제목: {video['title']}\n설명: {video['description']}"
            video["transcript"] = ""
            video["summarized_content"] = content
            video["has_transcript"] = False
            enriched_videos.append(video)
    
    # 8. Document 형태로 변환
    print("📄 Document 변환 시작")
    sns_docs = []
    
    # 유튜브 영상을 Document로 변환
    for video in enriched_videos:
        # 요약된 내용을 주요 콘텐츠로 사용
        content = video["summarized_content"]
        
        doc = Document(
            page_content=content,
            metadata={
                "source": "youtube",
                "title": video.get("title", "제목 없음"),  # 제목 추가!
                "video_id": video["video_id"],
                "channel_title": video["channel_title"],
                "keywords": video["keywords"],
                "relevance_score": video.get("relevance_score", 0),
                "type": "youtube_video",
                "search_intent": analysis["intent"],
                "detected_drugs": analysis.get("potential_drugs", []),
                "body_parts": analysis.get("body_parts", []),
                "thumbnail": video["thumbnail"],
                "published_at": video["published_at"],
                "has_transcript": video.get("has_transcript", False),
                "transcript_length": len(video.get("transcript", "")),
                "summary_length": len(video.get("summarized_content", "")),
                "summary": video.get("summarized_content", "")  # summary도 추가
            }
        )
        sns_docs.append(doc)
    
    # 네이버 뉴스를 Document로 변환 (필터링된 뉴스만 사용)
    for news in filtered_news:
        content = f"제목: {news['title']}\n내용: {news['description']}\n발행일: {news.get('pub_date_parsed', news.get('pub_date', ''))}"
        
        doc = Document(
            page_content=content,
            metadata={
                "source": "naver_news",
                "title": news["title"],
                "link": news.get("link", ""),
                "original_link": news.get("original_link", ""),
                "type": "news_article",
                "search_intent": analysis["intent"],
                "detected_drugs": analysis.get("potential_drugs", []),
                "pub_date": news.get("pub_date_parsed", news.get("pub_date", "")),
                "relevance_score": news.get("relevance_score", 0)
            }
        )
        sns_docs.append(doc)
    
    # 결과를 state에 저장
    state["sns_results"] = sns_docs
    state["sns_count"] = len(sns_docs)
    state["sns_analysis"] = analysis
    
    print(f"🎉 신약 검색 완료: {len(sns_docs)}개 결과")
    print(f"📺 유튜브: {len(enriched_videos)}개 영상")
    print(f"📰 네이버 뉴스: {len(filtered_news)}개 기사")
    print(f"📊 자막 있는 영상: {sum(1 for v in enriched_videos if v.get('has_transcript', False))}개")
    
    return state

