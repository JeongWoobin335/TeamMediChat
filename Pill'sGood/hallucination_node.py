from qa_state import QAState
from langchain_core.documents import Document
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from answer_utils import generate_response_llm_from_prompt
import json

# GPT-5 모델 초기화 - 환각 검사 전용
_hallucination_llm = None

def _get_hallucination_llm():
    """환각 검사용 GPT-5 모델을 지연 초기화"""
    global _hallucination_llm
    if _hallucination_llm is None:
        # GPT-5 사용 (가장 정확한 환각 검사)
        # GPT-5는 temperature 파라미터를 지원하지 않으므로 제거
        _hallucination_llm = ChatOpenAI(model="gpt-5")
    return _hallucination_llm

def hallucination_check_node(state: QAState) -> QAState:
    """
    GPT-5 기반 최신 정보(YouTube, 네이버 뉴스) 전용 환각 검사 노드
    
    **중요**: 효능, 부작용, 주의사항 등은 신뢰도 높은 DB(Excel, PDF, PubChem)에서 수집되므로
    환각 검사가 필요하지 않습니다. 오직 외부 소스인 YouTube와 네이버 뉴스 최신 정보만 검사합니다.
    
    최신 정보의 정확성과 신뢰성을 검증하여:
    1. 수집된 최신 정보가 실제로 존재하는지 확인
    2. 답변에 포함된 최신 정보가 수집된 정보와 일치하는지 확인
    3. 과장되거나 잘못된 정보가 포함되어 있는지 확인
    """
    print("🔍 최신 정보 환각 검사 시작 (GPT-5 기반, 최신 정보 전용)")
    
    # enhanced_rag_answer 또는 final_answer 확인
    answer = state.get("enhanced_rag_answer") or state.get("final_answer", "")
    query = state.get("query", "")
    
    if not answer or not query:
        print("⚠️ 답변 또는 질문이 없어 환각 검사 건너뜀")
        state["hallucination_flag"] = None
        state["hallucination_details"] = {}
        return state
    
    # 최신 정보 소스 수집
    youtube_info = state.get("youtube_info") or {}
    naver_news_info = state.get("naver_news_info") or {}
    enhanced_rag_analysis = state.get("enhanced_rag_analysis", {})
    
    # enhanced_rag_analysis에서 최신 정보 추출
    if enhanced_rag_analysis:
        if not youtube_info:
            youtube_info = enhanced_rag_analysis.get("youtube_info", {})
        if not naver_news_info:
            naver_news_info = enhanced_rag_analysis.get("naver_news_info", {})
    
    # 최신 정보가 있는지 확인
    has_latest_info = (
        (youtube_info and youtube_info.get('total_videos', 0) > 0) or
        (naver_news_info and naver_news_info.get('total_count', 0) > 0)
    )
    
    if not has_latest_info:
        print("ℹ️ 최신 정보(YouTube, 네이버 뉴스)가 없어 환각 검사 건너뜀")
        print("   (효능, 부작용 등은 신뢰도 높은 DB에서 수집되므로 검사 불필요)")
        # 최신 정보가 없으면 환각 검사 불필요 (효능, 부작용은 신뢰도 높은 DB에서 왔으므로)
        state["hallucination_flag"] = False  # 환각 없음으로 표시
        state["hallucination_details"] = {
            "has_latest_info": False,
            "check_type": "skipped",
            "reason": "최신 정보 없음, 신뢰도 높은 DB 정보만 사용"
        }
        return state
    
    # 최신 정보가 있는 경우 상세 환각 검사
    print("📰 최신 정보 기반 상세 환각 검사 수행")
    
    # YouTube 정보 요약
    youtube_summary = _format_youtube_info_for_check(youtube_info)
    
    # 네이버 뉴스 정보 요약
    naver_news_summary = _format_naver_news_info_for_check(naver_news_info)
    
    # 상세 환각 검사 프롬프트 (최신 정보 전용)
    hallucination_check_prompt = f"""
당신은 의약품 정보의 정확성을 검증하는 전문가입니다. 
**오직 답변의 "📰 최신 정보" 섹션만 검증하세요.** 효능, 부작용, 주의사항 등은 신뢰도 높은 DB에서 수집되었으므로 검증 대상이 아닙니다.

**사용자 질문:**
{query}

**시스템이 생성한 답변 (전체):**
{answer}

**수집된 YouTube 정보:**
{youtube_summary if youtube_summary else "YouTube 정보 없음"}

**수집된 네이버 뉴스 정보:**
{naver_news_summary if naver_news_summary else "네이버 뉴스 정보 없음"}

**⚠️ 검증 대상:**
- **오직 "💡 추가 정보" 또는 "📰 최신 정보" 섹션만 검증하세요**
- 효능, 부작용, 주의사항, 사용법 등은 검증 대상이 아닙니다 (신뢰도 높은 DB 정보)

**검증 항목 (추가 정보 섹션만):**
1. 답변의 "💡 추가 정보" 또는 "📰 최신 정보" 섹션에 포함된 내용이 실제로 수집된 YouTube/네이버 뉴스 정보에 존재하는가?
2. 추가 정보 섹션의 사실 주장이 수집된 정보와 일치하는가?
3. 추가 정보 섹션에 과장되거나 잘못된 정보가 포함되어 있는가?
4. 추가 정보 섹션에 수집된 정보에 없는 내용을 마치 확실한 정보인 것처럼 표현했는가?

**⚠️ 매우 중요: 환각 판단 기준**
- **일반적인 약리학 지식이나 의학 상식은 환각이 아닙니다**
  - 예: "비타민 B6는 신경 기능을 개선하고 피로 회복에 도움을 줍니다" → 일반적인 약리학 지식이므로 환각 아님
  - 예: "카페인은 과다 복용 시 불안이나 불면증을 유발할 수 있습니다" → 일반적인 약리학 지식이므로 환각 아님
  - 예: "아세트아미노펜은 간독성을 유발할 수 있습니다" → 일반적인 약리학 지식이므로 환각 아님

- **다음 경우만 환각으로 판단:**
  1. **구체적인 사실 주장이 수집된 정보에 없고, 일반 상식도 아닌 경우**
     - 예: "최근 뉴스에 따르면, 욱씬정은 2014년 6월에 마더스제약에서 출시되었습니다" → 수집된 뉴스에 이 정보가 없으면 환각
  2. **과장되거나 왜곡된 정보**
     - 예: "최근 연구에서 이 약이 완전히 안전하다고 밝혀졌습니다" → 수집된 정보에 이런 주장이 없으면 환각
  3. **시간성 왜곡**
     - 예: 2014년 뉴스를 "최근 뉴스"로 표현 → 환각

- **수집된 정보와 일치하는 내용은 환각이 아님**
- **효능, 부작용 등 다른 섹션은 검증하지 마세요**

**응답 형식 (JSON):**
{{
    "is_hallucinated": true/false,
    "confidence": "high/medium/low",
    "reasons": ["이유1", "이유2"],
    "specific_issues": [
        {{
            "issue_type": "과장/왜곡/없는정보",
            "content": "문제가 된 답변 내용",
            "evidence": "수집된 정보와의 비교 결과"
        }}
    ],
    "verified_info": [
        {{
            "content": "검증된 답변 내용",
            "source": "youtube/naver_news"
        }}
    ]
}}
"""

    try:
        llm = _get_hallucination_llm()
        response = llm.invoke(hallucination_check_prompt)
        response_text = response.content.strip()
        
        # JSON 코드 블록 제거
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            response_text = '\n'.join(lines).strip()
        
        # JSON 파싱
        try:
            result = json.loads(response_text)
            is_hallucinated = result.get("is_hallucinated", False)
            confidence = result.get("confidence", "medium")
            reasons = result.get("reasons", [])
            specific_issues = result.get("specific_issues", [])
            verified_info = result.get("verified_info", [])
            
            print(f"🔍 환각 검사 결과:")
            print(f"  - 환각 여부: {is_hallucinated}")
            print(f"  - 신뢰도: {confidence}")
            print(f"  - 이유: {', '.join(reasons[:3])}")
            
            if specific_issues:
                print(f"  - 발견된 문제: {len(specific_issues)}개")
                for issue in specific_issues[:2]:
                    print(f"    • {issue.get('issue_type', '')}: {issue.get('content', '')[:50]}...")
            
            state["hallucination_flag"] = is_hallucinated
            state["hallucination_details"] = {
                "has_latest_info": True,
                "check_type": "latest_info_verification",
                "confidence": confidence,
                "reasons": reasons,
                "specific_issues": specific_issues,
                "verified_info": verified_info
            }
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON 파싱 실패: {e}")
            print(f"원본 응답: {response_text[:200]}...")
            # 기본 환각 검사로 fallback
            state["hallucination_flag"] = None
            state["hallucination_details"] = {
                "has_latest_info": True,
                "check_type": "latest_info_verification",
                "error": "JSON 파싱 실패"
            }
            
    except Exception as e:
        print(f"❌ 환각 검사 오류: {e}")
        state["hallucination_flag"] = None
        state["hallucination_details"] = {
            "has_latest_info": True,
            "check_type": "latest_info_verification",
            "error": str(e)
        }
    
    return state

def _format_youtube_info_for_check(youtube_info: Dict) -> str:
    """YouTube 정보를 환각 검사용으로 포맷팅"""
    if not youtube_info or youtube_info.get('total_videos', 0) == 0:
        return "YouTube 정보 없음"
    
    formatted = []
    formatted.append(f"총 {youtube_info['total_videos']}개 영상")
    
    # 약품 관련 영상
    medicine_videos = youtube_info.get('medicine_videos', [])
    if medicine_videos:
        formatted.append("\n약품 관련 영상:")
        for video in medicine_videos[:10]:
            formatted.append(f"- 제목: {video.get('title', '')}")
            if video.get('summary'):
                formatted.append(f"  요약: {video.get('summary', '')[:500]}")
            elif video.get('description'):
                formatted.append(f"  설명: {video.get('description', '')[:300]}")
    
    # 성분 관련 영상
    ingredient_videos = youtube_info.get('ingredient_videos', [])
    if ingredient_videos:
        formatted.append("\n성분 관련 영상:")
        for video in ingredient_videos[:8]:
            formatted.append(f"- 제목: {video.get('title', '')}")
            if video.get('summary'):
                formatted.append(f"  요약: {video.get('summary', '')[:500]}")
    
    # 사용법 관련 영상
    usage_videos = youtube_info.get('usage_videos', [])
    if usage_videos:
        formatted.append("\n사용법 관련 영상:")
        for video in usage_videos[:6]:
            formatted.append(f"- 제목: {video.get('title', '')}")
            if video.get('summary'):
                formatted.append(f"  요약: {video.get('summary', '')[:500]}")
    
    return "\n".join(formatted)

def _format_naver_news_info_for_check(naver_news_info: Dict) -> str:
    """네이버 뉴스 정보를 환각 검사용으로 포맷팅"""
    if not naver_news_info or naver_news_info.get('total_count', 0) == 0:
        return "네이버 뉴스 정보 없음"
    
    formatted = []
    formatted.append(f"총 {naver_news_info['total_count']}건의 뉴스")
    
    # 신제품 뉴스
    product_news = naver_news_info.get('product_news', [])
    if product_news:
        formatted.append("\n신제품 뉴스:")
        for news in product_news[:10]:
            formatted.append(f"- 제목: {news.get('title', '')}")
            formatted.append(f"  내용: {news.get('description', '')[:500]}")
            formatted.append(f"  날짜: {news.get('pub_date_parsed', '')}")
    
    # 일반 뉴스
    medicine_news = naver_news_info.get('medicine_news', [])
    if medicine_news:
        formatted.append("\n일반 뉴스:")
        for news in medicine_news[:12]:
            formatted.append(f"- 제목: {news.get('title', '')}")
            formatted.append(f"  내용: {news.get('description', '')[:400]}")
    
    # 트렌드 뉴스
    trend_news = naver_news_info.get('trend_news', [])
    if trend_news:
        formatted.append("\n트렌드 뉴스:")
        for news in trend_news[:10]:
            formatted.append(f"- 제목: {news.get('title', '')}")
            formatted.append(f"  내용: {news.get('description', '')[:400]}")
    
    # 성분 관련 뉴스
    ingredient_news = naver_news_info.get('ingredient_news', [])
    if ingredient_news:
        formatted.append("\n성분 관련 뉴스:")
        for news in ingredient_news[:8]:
            formatted.append(f"- 제목: {news.get('title', '')}")
            formatted.append(f"  내용: {news.get('description', '')[:300]}")
    
    return "\n".join(formatted)
