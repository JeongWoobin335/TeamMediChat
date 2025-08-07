import os
import re
from typing import List, Dict
import praw
from langchain_core.documents import Document
from qa_state import QAState

def setup_reddit_api():
    """레딧 API 설정"""
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT")
    )
    return reddit

def extract_keywords(text: str) -> List[str]:
    """텍스트에서 키워드 추출"""
    keywords = re.findall(r'\b\w+\b', text.lower())
    return keywords

def search_reddit_posts(query: str, max_posts: int = 10) -> List[Dict]:
    """레딧에서 약품 관련 포스트 검색"""
    try:
        reddit = setup_reddit_api()
        posts = []
        
        # 약품 관련 서브레딧들
        subreddits = [
            "medicine", "pharmacy", "AskDocs", "health", 
            "Supplements", "Drugs", "medical"
        ]
        
        for subreddit_name in subreddits[:4]:  # 최대 4개 서브레딧 검색
            try:
                subreddit = reddit.subreddit(subreddit_name)
                
                # 인기 포스트와 최신 포스트 모두 가져오기 (검색 범위 확장)
                hot_posts = subreddit.hot(limit=10)  # 인기 포스트 10개
                
                new_posts = subreddit.new(limit=10)  # 최신 포스트 10개
                
                # 모든 포스트를 합치기
                all_subreddit_posts = list(hot_posts) + list(new_posts)
                
                for post in all_subreddit_posts:
                    # 제목과 내용 결합
                    content = f"제목: {post.title}\n내용: {post.selftext}"
                    
                    # 모든 포스트를 수집 (필터링은 나중에)
                    keywords = extract_keywords(content)
                    
                    posts.append({
                        "content": content,
                        "keywords": keywords,
                        "source": "reddit",
                        "subreddit": subreddit_name,
                        "score": post.score,
                        "created_utc": post.created_utc
                    })
                    
            except Exception as e:
                continue
                
    except Exception as e:
        return []
    
    return posts[:max_posts]

def filter_medicine_related_posts(posts: List[Dict]) -> List[Dict]:
    """약품 관련 포스트 필터링 (더 관대하게)"""
    medicine_keywords = [
        "약", "감기약", "진통제", "소화제", "비타민", "영양제",
        "부작용", "효능", "복용법", "처방", "약국", "제약",
        "medicine", "drug", "pill", "side effect", "efficacy",
        "cold", "fever", "headache", "pain", "symptom", "treatment",
        "doctor", "pharmacy", "prescription", "medication"
    ]
    
    filtered_posts = []
    for post in posts:
        content_lower = post["content"].lower()
        # 더 관대한 필터링: 키워드가 있거나 점수가 높은 포스트 포함
        if (any(keyword in content_lower for keyword in medicine_keywords) or 
            post.get("score", 0) > 5):  # 점수가 5 이상이면 포함
            filtered_posts.append(post)
    
    # 필터링 결과가 너무 적으면 모든 포스트 반환 (더 관대하게)
    if len(filtered_posts) < 5 and len(posts) > 0:
        return posts[:10]  # 최대 10개까지
    
    # 필터링 결과가 없어도 일부 포스트 반환
    if len(filtered_posts) == 0 and len(posts) > 0:
        return posts[:5]
    
    return filtered_posts

def translate_korean_to_english(query: str) -> List[str]:
    """한국어 키워드를 영어로 변환"""
    korean_to_english = {
        # 약품명
        "타이레놀": ["tylenol", "acetaminophen", "paracetamol"],
        "아스피린": ["aspirin"],
        "이부프로펜": ["ibuprofen", "advil", "motrin"],
        "파라세타몰": ["paracetamol", "acetaminophen"],
        "판콜": ["pancol", "cold medicine"],
        "판피린": ["panpyrin", "cold medicine"],
        "베아제": ["bease", "digestive enzyme"],
        "가스모틴": ["gasmotin", "motilium"],
        "훼스탈": ["festal", "digestive enzyme"],
        
        # 증상/질병
        "감기약": ["cold medicine", "flu medicine"],
        "약": ["medicine", "drug", "medication"],
        "진통제": ["painkiller", "analgesic", "pain medicine"],
        "소화제": ["digestive medicine", "antacid"],
        "비타민": ["vitamin", "supplement"],
        "부작용": ["side effect", "adverse effect"],
        "효능": ["efficacy", "effectiveness"],
        "복용법": ["dosage", "how to take"],
        "두통": ["headache", "migraine"],
        "열": ["fever", "temperature"],
        "기침": ["cough", "coughing"],
        "콧물": ["runny nose", "nasal discharge"],
        "추천": ["recommendation", "suggestion"],
        "위장": ["stomach", "gastric"],
        "위장염": ["gastritis", "stomach inflammation"],
        "감기": ["cold", "flu", "common cold"],
        "복통": ["stomach pain", "abdominal pain"],
        "소화불량": ["indigestion", "dyspepsia"],
        "열": ["fever", "temperature"],
        # 최신 정보 관련
        "최신": ["latest", "new", "recent"],
        "새로운": ["new", "latest", "recent"],
        "경험담": ["experience", "review", "testimonial"],
        "후기": ["review", "experience", "feedback"],
        "부작용": ["side effect", "adverse effect", "reaction"]
    }
    
    translated_keywords = []
    for korean, english_list in korean_to_english.items():
        if korean in query:
            translated_keywords.extend(english_list)
    
    return translated_keywords

def sns_search_node(state: QAState) -> QAState:
    """SNS 검색 노드 (레딧 기반)"""
    
    # 쿼리에서 검색 키워드 추출
    query = state.get("cleaned_query", "") or state.get("normalized_query", "") or state.get("query", "")
    
    # 더 구체적인 약품 관련 키워드 추출
    medicine_keywords = [
        # 일반적인 약품명
        "타이레놀", "아스피린", "이부프로펜", "파라세타몰", "아세트아미노펜",
        "판콜", "판피린", "베아제", "가스모틴", "훼스탈",
        # 영어 약품명
        "tylenol", "aspirin", "ibuprofen", "acetaminophen", "paracetamol",
        # 증상/질병
        "감기", "두통", "복통", "소화불량", "위장", "열", "기침", "콧물",
        "cold", "headache", "stomach", "digestion", "fever", "cough",
        # 약품 종류
        "감기약", "진통제", "소화제", "비타민", "영양제",
        "medicine", "drug", "pill", "supplement", "vitamin",
        # 최신 정보 관련 키워드
        "2024", "2023", "새로", "신약", "최신", "새로운", "경험담", "후기",
        "new", "latest", "recent", "experience", "review", "side effect"
    ]
    
    search_keywords = []
    
    # 쿼리에서 약품 관련 키워드 찾기 (더 정확하게)
    query_lower = query.lower()
    for keyword in medicine_keywords:
        if keyword.lower() in query_lower:
            search_keywords.append(keyword)
    
    # 한국어 키워드가 있으면 영어로 변환
    if any(keyword in ["감기약", "약", "진통제", "소화제", "비타민", "타이레놀", "감기", "두통", "소화불량", "최신", "새로운", "경험담", "후기"] for keyword in search_keywords):
        english_keywords = translate_korean_to_english(query)
        search_keywords.extend(english_keywords)
    
    # 키워드가 없으면 기본 키워드 사용
    if not search_keywords:
        search_keywords = ["medicine", "drug"]  # 기본값
    
    all_posts = []
    
    # 각 키워드로 레딧 검색 (더 간단한 키워드부터)
    # 기본 키워드들로 먼저 테스트
    basic_keywords = ["medicine", "headache", "pain"]
    for keyword in basic_keywords:
        try:
            posts = search_reddit_posts(keyword, max_posts=5)
            all_posts.extend(posts)
        except Exception as e:
            pass
    
    # 추가로 쿼리에서 추출한 키워드도 검색
    for keyword in search_keywords[:2]:  # 최대 2개 키워드만 검색
        if keyword not in basic_keywords:  # 중복 방지
            try:
                posts = search_reddit_posts(keyword, max_posts=5)
                all_posts.extend(posts)
            except Exception as e:
                pass
    
    # 약품 관련 포스트 필터링 (더 관대하게)
    filtered_posts = filter_medicine_related_posts(all_posts)
    
    # Document 형태로 변환
    sns_docs = []
    for post in filtered_posts:
        doc = Document(
            page_content=post["content"],
            metadata={
                "source": "reddit",
                "subreddit": post["subreddit"],
                "keywords": post["keywords"],
                "score": post["score"],
                "type": "sns_post"
            }
        )
        sns_docs.append(doc)
    
    # 결과를 state에 저장
    state["sns_results"] = sns_docs
    state["sns_count"] = len(sns_docs)
    
    return state 