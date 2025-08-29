# TeamMediChat - 의약품 정보 검색 및 추천 시스템

## 개요

TeamMediChat은 의약품 정보 검색, 약품 추천, 그리고 실시간 대화를 지원하는 AI 기반 시스템입니다. LangGraph를 활용하여 복잡한 의료 질문을 단계별로 처리하고, 다양한 데이터 소스에서 정확한 정보를 제공합니다.

## 시스템 아키텍처

### 핵심 구성 요소

- **LangGraph 기반 워크플로우**: 상태 기반 그래프 구조로 질문 처리 흐름 관리
- **다중 데이터 소스**: Excel, PDF, 외부 API, YouTube 등 다양한 정보 소스 활용
- **AI 기반 라우팅**: 사용자 질문을 분석하여 적절한 처리 경로로 안내
- **실시간 대화 인터페이스**: 연속적인 대화를 통한 사용자 경험 제공
- **웹 기반 UI**: ChatGPT 스타일의 현대적인 채팅 인터페이스

## 주요 노드 구성

### 1. 전처리 및 라우팅 노드
- **preprocess_node**: 사용자 입력 정규화 및 기본 처리
- **medicine_related_filter_node**: 의약품 관련 질문 여부 판단
- **route_question_node**: 질문 유형에 따른 처리 경로 결정 (추천/검색/SNS 검색)

### 2. 검색 및 정보 수집 노드
- **excel_search_node**: Excel 데이터베이스에서 구조화된 의약품 정보 검색
- **pdf_search_node**: PDF 문서에서 상세한 의학 문헌 및 연구 자료 검색
- **external_search_node**: 외부 API를 통한 최신 약품 정보 및 시장 데이터 수집
- **sns_search_node**: YouTube API를 통한 최신 신약 소식, 사용자 경험, 뉴스 검색

### 3. 품질 관리 및 생성 노드
- **rerank_check_node**: 검색 결과의 관련성 및 품질 평가
- **hallucination_node**: AI 응답의 사실 여부 검증
- **requery_answer_node**: 부족한 정보가 있을 경우 추가 검색 수행
- **generate_node**: 사용자에게 최종 답변 제공

### 4. 특화 노드
- **recommend_medicine_node**: 특정 증상이나 질병에 대한 약품 추천
- **remember_clean_node**: 이전 대화 맥락 분석 및 기억
- **context_aware_router_node**: 맥락을 고려한 지능형 라우팅

## 데이터 처리 흐름

### 1. 질문 분석 및 분류

사용자의 질문은 다음과 같이 분류됩니다:
- **약품 추천 요청**: 특정 증상이나 질병에 대한 약품 추천
- **기본 정보 요청**: 약품 정보, 부작용, 사용법 등 상세 정보
- **SNS 검색 요청**: 최신 신약 소식, 사용자 경험담, 실시간 정보, 뉴스 등

### 2. 검색 우선순위 및 경로

#### 추천 경로
- 사용자 질문 → 약품 추천 → 최종 답변

#### SNS 검색 경로 (YouTube)
- 사용자 질문 → YouTube 검색 → 자막 추출 → 내용 요약 → 최종 답변

#### 일반 검색 경로
- 사용자 질문 → Excel/PDF/외부 검색 → 재순위화 → 환각 검사 → 최종 답변

### 3. 품질 관리

- **재순위화**: 검색 결과의 관련성 및 신뢰도 평가
- **환각 검사**: AI 응답의 사실 여부 검증
- **재질의**: 부족한 정보가 있을 경우 추가 검색 수행

## YouTube 통합 기능

### SNS 검색 노드 (sns_node.py)

- **YouTube API 연동**: 공식 YouTube Data API v3 사용
- **자막 추출**: youtube-transcript-api를 통한 한국어/영어 자막 자동 추출
- **내용 요약**: LangChain 텍스트 분할기를 사용한 스마트 요약
- **의도별 검색**: 부작용, 경험담, 최신 정보 등 질문 의도에 따른 검색어 자동 생성
- **관련성 필터링**: 원본 질문과의 관련성을 점수 기반으로 평가

### 검색 최적화

- 한국어 우선 검색 (relevanceLanguage: 'ko')
- 중간 길이 영상 우선 (videoDuration: 'medium')
- 관련성 순 정렬 (order: 'relevance')
- 최대 5개 영상으로 제한하여 품질 보장

## 웹 인터페이스

### 주요 기능

- **ChatGPT 스타일 채팅**: 실시간 양방향 대화
- **세션 관리**: 새 대화 세션 생성, 기존 세션 복구 및 전환
- **대화 맥락 기억**: 이전 대화 내용을 자동으로 기억
- **반응형 디자인**: 데스크톱, 태블릿, 모바일 지원

### 기술 스택

- **백엔드**: FastAPI, WebSocket, 기존 LangGraph 시스템
- **프론트엔드**: HTML5, CSS3, JavaScript, Font Awesome
- **통신**: WebSocket (실시간), REST API (세션 관리)

## 설치 및 실행

### 필수 요구사항

- Python 3.8 이상
- OpenAI API 키 (환경 변수로 설정)
- YouTube API 키 (SNS 검색 기능 사용 시)
- 필요한 Python 패키지들 (requirements.txt 참조)

### 환경 변수 설정

```bash
# .env 파일에 설정
OPENAI_API_KEY=your_openai_api_key_here
YOUTUBE_API_KEY=your_youtube_api_key_here
```

### 실행 방법

#### 터미널 모드
```bash
# 의존성 설치
pip install -r requirements.txt

# 시스템 실행
python main_graph.py
```

#### 웹 서버 모드
```bash
# Windows
run_web_server.bat

# Linux/Mac
chmod +x run_web_server.sh
./run_web_server.sh

# 직접 실행
python web_server.py
```

#### 웹 접속
```
http://localhost:8000
```

## 설정 및 커스터마이징

### 기본 설정 (config.py)

- **검색 설정**: 최대 결과 수, 컨텍스트 길이, 캐시 만료 시간
- **모델 설정**: LLM 온도, 토큰 제한, 기본 모델
- **벡터 DB 설정**: 유사도 임계값

### 키워드 패턴 설정 (medical_patterns.py)

- 통증 관련 패턴
- 불편함 관련 패턴
- 부작용 관련 패턴
- 경험담 관련 패턴
- 신체 부위 패턴
- 강도 표현 패턴

## 캐시 시스템

### 캐시 구조

- **질문 캐시**: 동일한 질문에 대한 응답 저장
- **검색 결과 캐시**: 데이터베이스 검색 결과 임시 저장
- **벡터 DB 캐시**: 임베딩 및 벡터 검색 결과 저장
- **세션 캐시**: 대화 세션별 컨텍스트 정보 관리

### 캐시 관리

- 자동 만료 시간 설정
- 메모리 사용량 모니터링
- 필요시 수동 캐시 정리

## 확장성 및 유지보수

### 새로운 노드 추가

1. 노드 함수 구현
2. main_graph.py에 노드 등록
3. 적절한 위치에 엣지 연결
4. 라우팅 로직 업데이트

### 데이터 소스 확장

- 새로운 데이터베이스 연결
- API 엔드포인트 추가
- 데이터 전처리 로직 구현

### 새로운 검색 소스 추가

- SNS 플랫폼 확장 (Twitter, Instagram 등)
- 뉴스 API 연동
- 의학 저널 API 연동

## 성능 최적화

### 검색 효율성

- 우선순위 기반 검색 경로 최적화
- 캐시를 통한 중복 검색 방지
- 비동기 처리로 응답 시간 단축

### 메모리 관리

- 세션별 메모리 사용량 제한
- 주기적인 캐시 정리
- 불필요한 데이터 자동 제거

## 보안 및 개인정보 보호

- API 키 환경 변수 관리
- 사용자 데이터 암호화
- 세션 정보 보안 처리
- 로그 데이터 개인정보 제거
- CORS 설정 (개발/프로덕션 환경별 조정)

## 문제 해결

### 일반적인 문제

- **WebSocket 연결 실패**: 방화벽 설정, 포트 확인
- **세션 로드 실패**: 폴더 권한, 기존 시스템 동작 확인
- **메시지 전송 실패**: 네트워크 연결, 서버 로그 확인

### SNS 검색 관련 문제

- **YouTube API 키 오류**: .env 파일 설정 확인
- **자막 추출 실패**: 영상 자막 존재 여부 확인
- **검색 결과 부족**: 검색어 다양화 필요

## 향후 개선 계획

- 사용자 인증 시스템
- 대화 내용 내보내기 (PDF, Excel)
- 약품 정보 시각화 (차트, 그래프)
- 다국어 지원
- 다크 모드 테마
- PWA (Progressive Web App) 지원
- 추가 SNS 플랫폼 연동
- 음성 입력/출력 지원

## 파일 구조

```
Pill'sGood/
├── main_graph.py                    # 메인 그래프 및 노드 연결
├── web_server.py                    # FastAPI 웹 서버
├── chat_interface.py                # 터미널 채팅 인터페이스
├── chat_session_manager.py          # 대화 세션 관리
├── cache_manager.py                 # 캐시 시스템 관리
├── config.py                        # 시스템 설정
├── qa_state.py                      # 질문-답변 상태 관리
├── medical_patterns.py              # 의학 관련 키워드 패턴
├── retrievers.py                    # 벡터 검색 및 리트리버
├── static/                          # 웹 UI 정적 파일
│   ├── chat.html                   # 메인 HTML 페이지
│   ├── chat.css                    # 스타일시트
│   └── chat.js                     # JavaScript 기능
├── chat_sessions/                   # 대화 세션 저장소
├── cache/                           # 캐시 데이터 저장소
├── requirements.txt                 # Python 의존성
├── README.md                        # 이 파일
├── run_web_server.bat              # Windows 실행 스크립트
└── run_web_server.sh               # Linux/Mac 실행 스크립트
```

## 지원 및 문의

시스템 사용 중 문제가 발생하거나 질문이 있으시면:
1. 브라우저 개발자 도구 콘솔 확인
2. 서버 터미널 로그 확인
3. 기존 터미널 버전과 비교하여 문제 파악
4. 설정 파일 및 환경 변수 확인

---

**TeamMediChat**으로 정확하고 신뢰할 수 있는 의약품 정보를 얻어보세요!
