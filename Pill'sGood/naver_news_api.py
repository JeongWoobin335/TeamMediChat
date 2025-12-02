# naver_news_api.py - 네이버 뉴스 API 연동 (추가 정보 수집용)

import os
import requests
import time
from typing import List, Dict, Optional
from dotenv import load_dotenv
from cache_manager import cache_manager

# 환경 변수 로드
load_dotenv()

class NaverNewsAPI:
    """네이버 뉴스 API 클래스 - 약품 관련 추가 정보 수집"""
    
    def __init__(self):
        # 강제로 .env 다시 로드 (디버깅용)
        load_dotenv(override=True)
        
        self.client_id = os.getenv("NAVER_CLIENT_ID")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET")
        self.base_url = "https://openapi.naver.com/v1/search/news.json"
        self.request_delay = 0.1  # API 요청 간격 (초)
        
        # 디버깅: 모든 환경 변수 키 출력
        print(f"🔍 환경 변수 확인 (NaverNewsAPI 초기화):")
        print(f"   .env 파일에서 읽은 NAVER_CLIENT_ID: {self.client_id if self.client_id else '❌ None'}")
        print(f"   .env 파일에서 읽은 NAVER_CLIENT_SECRET: {'***' + self.client_secret[-4:] if self.client_secret and len(self.client_secret) > 4 else '❌ None'}")
        
        # 모든 환경 변수 중 NAVER로 시작하는 것들 출력
        naver_vars = {k: v for k, v in os.environ.items() if 'NAVER' in k.upper()}
        if naver_vars:
            print(f"   환경 변수 중 NAVER 관련: {list(naver_vars.keys())}")
        else:
            print(f"   ⚠️ 환경 변수에 NAVER 관련 변수가 없습니다!")
        
        if not self.client_id or not self.client_secret:
            print("⚠️ 네이버 API 키가 설정되지 않았습니다. (.env 파일에 NAVER_CLIENT_ID, NAVER_CLIENT_SECRET 추가 필요)")
    
    def search_news(
        self, 
        query: str, 
        display: int = 10, 
        start: int = 1, 
        sort: str = "date"  # "date" 또는 "sim"
    ) -> List[Dict]:
        """
        네이버 뉴스 검색
        
        Args:
            query: 검색어 (약품명 등)
            display: 검색 결과 개수 (최대 100)
            start: 검색 시작 위치 (최대 1000)
            sort: 정렬 방식 ("date": 최신순, "sim": 정확도순)
        
        Returns:
            뉴스 기사 리스트
        """
        # API 키 확인
        if not self.client_id or not self.client_secret:
            print("❌ 네이버 API 키가 설정되지 않았습니다!")
            print(f"   NAVER_CLIENT_ID: {'설정됨' if self.client_id else '❌ 없음'}")
            print(f"   NAVER_CLIENT_SECRET: {'설정됨' if self.client_secret else '❌ 없음'}")
            print("   .env 파일에 NAVER_CLIENT_ID와 NAVER_CLIENT_SECRET를 추가하세요.")
            return []
        
        # 캐시 확인
        cache_key = f"naver_news_{query}_{display}_{start}_{sort}"
        cached_result = cache_manager.get_search_cache(cache_key, "naver_news")
        if cached_result is not None:
            print(f"📂 네이버 뉴스 캐시 히트: {query}")
            return cached_result
        
        try:
            # API 요청 헤더
            headers = {
                "X-Naver-Client-Id": self.client_id,
                "X-Naver-Client-Secret": self.client_secret
            }
            
            # 파라미터
            params = {
                "query": query,
                "display": min(display, 100),  # 최대 100개
                "start": min(start, 1000),     # 최대 1000
                "sort": sort
            }
            
            print(f"🔍 네이버 뉴스 검색: '{query}' (정렬: {sort})")
            print(f"   URL: {self.base_url}")
            print(f"   파라미터: {params}")
            
            # API 호출
            response = requests.get(
                self.base_url,
                headers=headers,
                params=params,
                timeout=10
            )
            
            print(f"   응답 상태: {response.status_code}")
            
            response.raise_for_status()
            
            data = response.json()
            items = data.get("items", [])
            
            # 결과 가공
            processed_items = []
            for item in items:
                processed_item = {
                    "title": self._remove_html_tags(item.get("title", "")),
                    "original_link": item.get("originallink", ""),
                    "link": item.get("link", ""),
                    "description": self._remove_html_tags(item.get("description", "")),
                    "pub_date": item.get("pubDate", ""),
                    "pub_date_parsed": self._parse_date(item.get("pubDate", ""))
                }
                processed_items.append(processed_item)
            
            print(f"✅ 네이버 뉴스 검색 완료: {len(processed_items)}건")
            
            # 캐시 저장
            cache_manager.save_search_cache(cache_key, "naver_news", processed_items)
            
            # API 요청 간격
            time.sleep(self.request_delay)
            
            return processed_items
            
        except requests.exceptions.HTTPError as e:
            print(f"❌ 네이버 뉴스 API HTTP 오류: {e}")
            print(f"   응답 내용: {e.response.text if hasattr(e, 'response') else 'N/A'}")
            if hasattr(e, 'response') and e.response.status_code == 403:
                print("   💡 403 오류: 네이버 개발자 센터에서 '검색 API'를 활성화했는지 확인하세요.")
            elif hasattr(e, 'response') and e.response.status_code == 401:
                print("   💡 401 오류: Client ID 또는 Client Secret이 잘못되었습니다.")
            return []
        except requests.exceptions.RequestException as e:
            print(f"❌ 네이버 뉴스 API 요청 오류: {e}")
            return []
        except Exception as e:
            print(f"❌ 네이버 뉴스 처리 오류: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def search_medicine_additional_info(
        self, 
        medicine_name: str, 
        ingredients: List[str] = None,
        max_results: int = 15
    ) -> Dict:
        """
        약품 관련 추가 정보 검색 (신제품, 트렌드, 이슈 등)
        
        Args:
            medicine_name: 약품명
            ingredients: 주성분 리스트
            max_results: 최대 결과 수
        
        Returns:
            {
                "medicine_news": [...],      # 약품 직접 관련 뉴스
                "product_news": [...],       # 신제품/출시 관련
                "ingredient_news": [...],    # 성분 관련 뉴스
                "trend_news": [...],         # 트렌드/이슈
                "total_count": int
            }
        """
        result = {
            "medicine_news": [],
            "product_news": [],
            "ingredient_news": [],
            "trend_news": [],
            "total_count": 0
        }
        
        try:
            # 🚀 성능 최적화: 검색 결과 수 감소
            print(f"📰 약품 최신 소식 검색: {medicine_name}")
            medicine_news = self.search_news(
                query=medicine_name,
                display=min(max_results, 15),  # 100개 → 15개로 감소
                sort="date"
            )
            
            # 신제품/출시 관련 필터링
            product_keywords = ["출시", "신제품", "새로운", "론칭", "리뉴얼", "파워", "플러스"]
            for news in medicine_news[:10]:
                title_desc = (news.get("title", "") + " " + news.get("description", "")).lower()
                if any(keyword in title_desc for keyword in product_keywords):
                    result["product_news"].append(news)
                else:
                    result["medicine_news"].append(news)
            
            # 최대 개수 제한 (더 많이)
            result["medicine_news"] = result["medicine_news"][:8]  # 8개로 증가
            result["product_news"] = result["product_news"][:5]  # 5개로 증가
            
            # 🚀 성능 최적화: 성분 검색 수 감소 (3개 → 2개, 10개 → 5개)
            if ingredients:
                for ingredient in ingredients[:2]:  # 3개 → 2개
                    print(f"📰 성분 트렌드 검색: {ingredient}")
                    ingredient_news = self.search_news(
                        query=ingredient,
                        display=5,  # 10개 → 5개
                        sort="date"
                    )
                    
                    # 트렌드 키워드 필터링
                    trend_keywords = ["효과", "연구", "발견", "밝혀", "도움", "예방", "개선"]
                    for news in ingredient_news[:5]:
                        title_desc = (news.get("title", "") + " " + news.get("description", "")).lower()
                        if any(keyword in title_desc for keyword in trend_keywords):
                            result["trend_news"].append(news)
                        else:
                            result["ingredient_news"].append(news)
            
            # 최대 개수 제한 (더 많이)
            result["ingredient_news"] = result["ingredient_news"][:5]  # 5개로 증가
            result["trend_news"] = result["trend_news"][:5]  # 5개로 증가
            
            # 중복 제거
            result = self._remove_duplicates(result)
            
            # 총 개수
            result["total_count"] = (
                len(result["medicine_news"]) +
                len(result["product_news"]) +
                len(result["ingredient_news"]) +
                len(result["trend_news"])
            )
            
            print(f"✅ 네이버 뉴스 추가 정보 검색 완료: 총 {result['total_count']}건")
            print(f"   - 약품 뉴스: {len(result['medicine_news'])}건")
            print(f"   - 신제품 정보: {len(result['product_news'])}건")
            print(f"   - 성분 뉴스: {len(result['ingredient_news'])}건")
            print(f"   - 트렌드 정보: {len(result['trend_news'])}건")
            
        except Exception as e:
            print(f"❌ 약품 추가 정보 검색 오류: {e}")
        
        return result
    
    def _remove_html_tags(self, text: str) -> str:
        """HTML 태그 제거 (<b>, </b> 등)"""
        import re
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    def _parse_date(self, date_str: str) -> str:
        """날짜 파싱 (RFC 2822 형식 → 읽기 쉬운 형식)"""
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(date_str)
            return dt.strftime("%Y-%m-%d")
        except:
            return date_str
    
    def _remove_duplicates(self, result: Dict) -> Dict:
        """중복 뉴스 제거"""
        seen_links = set()
        
        for category in ["medicine_news", "product_news", "ingredient_news", "trend_news"]:
            unique_items = []
            for item in result[category]:
                link = item.get("original_link") or item.get("link")
                if link not in seen_links:
                    seen_links.add(link)
                    unique_items.append(item)
            result[category] = unique_items
        
        return result
