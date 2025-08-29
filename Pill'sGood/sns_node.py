import os
import re
from typing import List, Dict
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qa_state import QAState
from medical_patterns import *

def setup_youtube_api():
    """유튜브 API 설정"""
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY가 .env 파일에 설정되지 않았습니다.")
    return api_key

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
        
        # 한국어 자막 우선, 없으면 영어 자막
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
        
        if transcript_list:
            # 자막 텍스트를 하나로 합치기
            full_transcript = ""
            for transcript in transcript_list:
                full_transcript += transcript['text'] + " "
            
            print(f"✅ 영상 {video_id} 자막 추출 성공: {len(full_transcript)}자")
            return full_transcript.strip()
        else:
            print(f"⚠️ 영상 {video_id} 자막이 없습니다")
            return ""
            
    except Exception as e:
        print(f"❌ 자막 가져오기 실패: {e}")
        return ""

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

def analyze_query_intent(query: str) -> Dict[str, any]:
    """쿼리의 의도와 핵심 요소를 점수 기반으로 분석"""
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
    
    # 최신 정보 관련 의도 점수
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
    
    # 4. 약품명 추출
    words = re.findall(r'\b\w+\b', query_lower)
    potential_drugs = []
    
    common_words = ['the', 'and', 'for', 'with', 'this', 'that', 'what', 'when', 'where', 'how', 'why', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should', 'may', 'might']
    
    for word in words:
        if len(word) > 2 and word not in common_words:
            potential_drugs.append(word)
    
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
    """분석 결과를 바탕으로 검색어 생성"""
    search_terms = []
    
    intent = analysis.get("intent")
    potential_drugs = analysis.get("potential_drugs", [])
    body_parts = analysis.get("body_parts", [])
    intensity = analysis.get("intensity")
    
    # 1. 한국어 검색어 우선 생성
    if potential_drugs:
        for drug in potential_drugs[:2]:
            if intent == "side_effect":
                search_terms.extend([
                    f"{drug} 부작용",
                    f"{drug} 부작용 경험",
                    f"{drug} 부작용 후기"
                ])
            elif intent == "experience_review":
                search_terms.extend([
                    f"{drug} 경험담",
                    f"{drug} 사용 후기",
                    f"{drug} 복용 경험"
                ])
            elif intent == "efficacy":
                search_terms.extend([
                    f"{drug} 효과",
                    f"{drug} 효능",
                    f"{drug} 복용 결과"
                ])
    
    # 2. 일반적인 의도별 검색어 (한국어)
    if intent == "side_effect":
        search_terms.extend([
            "감기약 부작용 경험",
            "감기약 부작용 후기",
            "약물 부작용 경험담",
            "부작용 경험담",
            "감기약 복용 후 부작용",
            "약물 부작용 경험",
            "부작용 경험"
        ])
    elif intent == "experience_review":
        search_terms.extend([
            "감기약 경험담",
            "약물 사용 후기",
            "복용 경험담",
            "약물 경험"
        ])
    elif intent == "latest_info":
        search_terms.extend([
            "신약 소식",
            "신약 뉴스",
            "신약 개발",
            "신약 승인",
            "신약 상륙"
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
    
    # 중복 제거
    unique_terms = list(dict.fromkeys(search_terms))
    print(f"📊 최종 검색어 목록: {unique_terms}")
    return unique_terms[:20]

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

def sns_search_node(state: QAState) -> QAState:
    """SNS 검색 노드 (유튜브 기반) - 영상 내용 추출 및 요약 포함"""
    
    print("🔍 SNS 검색 노드 실행 시작 (유튜브)")
    
    # 쿼리에서 검색 키워드 추출
    query = state.get("cleaned_query", "") or state.get("normalized_query", "") or state.get("query", "")
    
    print(f"📝 분석할 쿼리: {query}")
    
    if not query:
        print("❌ 쿼리가 없어서 SNS 검색 건너뜀")
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
    
    # 3. 각 검색어로 유튜브 검색
    print("📺 유튜브 검색 시작")
    for search_term in search_terms:
        try:
            print(f"🔍 '{search_term}' 검색 중...")
            videos = search_youtube_videos(search_term, max_videos=5)
            print(f"📝 '{search_term}' 검색 결과: {len(videos)}개 영상")
            all_videos.extend(videos)
        except Exception as e:
            print(f"❌ '{search_term}' 검색 실패: {e}")
            continue
    
    print(f"📊 총 수집된 영상: {len(all_videos)}개")
    
    # 4. 영상 필터링
    print("🔍 영상 필터링 시작")
    filtered_videos = filter_relevant_videos(all_videos, analysis)
    print(f"✅ 필터링 후 영상: {len(filtered_videos)}개")
    
    # 5. 영상 내용 추출 및 요약
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
    
    # 6. Document 형태로 변환
    print("📄 Document 변환 시작")
    sns_docs = []
    for video in enriched_videos:
        # 요약된 내용을 주요 콘텐츠로 사용
        content = video["summarized_content"]
        
        doc = Document(
            page_content=content,
            metadata={
                "source": "youtube",
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
                "summary_length": len(video.get("summarized_content", ""))
            }
        )
        sns_docs.append(doc)
    
    # 결과를 state에 저장
    state["sns_results"] = sns_docs
    state["sns_count"] = len(sns_docs)
    state["sns_analysis"] = analysis
    
    print(f"🎉 SNS 검색 완료: {len(sns_docs)}개 결과")
    print(f"📊 자막 있는 영상: {sum(1 for v in enriched_videos if v.get('has_transcript', False))}개")
    
    return state 