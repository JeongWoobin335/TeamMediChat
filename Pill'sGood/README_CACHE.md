# 🗂️ 캐싱 시스템 가이드

## 개요

이 캐싱 시스템은 테스트 시 OpenAI API 크레딧을 절약하기 위해 설계되었습니다. 매번 실행할 때마다 반복되는 비용이 많이 드는 작업들을 캐싱하여 재사용할 수 있습니다.

## 🎯 캐싱 대상

### 1. 벡터 DB 캐싱
- **PDF 문서 임베딩**: 한 번 생성된 FAISS 벡터 DB를 파일로 저장
- **Excel 문서 임베딩**: 10개 Excel 파일의 모든 데이터 임베딩 결과 저장
- **파일 해시 기반 무효화**: 원본 파일이 변경되면 자동으로 재생성

### 2. 검색 결과 캐싱
- **PDF 검색 결과**: 동일한 쿼리에 대한 검색 결과 저장
- **Excel 검색 결과**: 정규화된 쿼리 기반 검색 결과 저장
- **외부 검색 결과**: Tavily API 호출 결과 및 요약 저장

### 3. LLM 매칭 결과 캐싱 (신규)
- **약품-증상 매칭**: LLM이 판단한 약품과 증상의 관련성 결과 저장
- **배치 처리 최적화**: 여러 약품을 묶어서 처리한 결과 캐싱
- **중복 호출 방지**: 동일한 약품-증상 조합에 대한 LLM 호출 방지

## 📁 캐시 구조

```
cache/
├── vectors/          # 벡터 DB 캐시
│   ├── pdf_vector_db.pkl
│   ├── pdf_vector_db_hash.json
│   ├── excel_vector_db.pkl
│   └── excel_vector_db_hash.json
├── search/           # 검색 결과 캐시
│   ├── search_pdf_[hash].pkl
│   ├── search_excel_[hash].pkl
│   ├── search_external_raw_[hash].pkl
│   └── search_external_parsed_[hash].pkl
├── matching/         # LLM 매칭 결과 캐시 (신규)
│   ├── matching_[condition_hash]_[medicines_hash].pkl
│   └── ...
└── embeddings/       # 임베딩 캐시 (향후 확장)
```

## 🚀 사용법

### 1. 자동 캐싱 (기본)
```python
# main_graph.py 실행 시 자동으로 캐싱 적용
python main_graph.py
```

### 2. 캐시 관리 도구
```python
# 캐시 관리 메뉴 실행
python cache_test.py

# 캐싱 시스템 테스트만 실행
python cache_test.py test
```

### 3. 프로그래밍 방식
```python
from cache_manager import cache_manager

# 캐시 통계 확인
stats = cache_manager.get_cache_stats()
print(f"캐시 크기: {stats['total_cache_size_mb']}MB")

# 만료된 캐시 정리 (7일 이상)
cache_manager.clear_expired_cache()

# 모든 캐시 삭제
cache_manager.clear_all_cache()
```

## 💰 비용 절약 효과

### 첫 번째 실행
- PDF 임베딩: ~$0.50-1.00 (문서 크기에 따라)
- Excel 임베딩: ~$2.00-5.00 (10개 파일)
- 검색 API 호출: ~$0.10-0.50 (쿼리당)
- LLM 매칭 호출: ~$1.00-3.00 (약품 수에 따라)

### 두 번째 실행부터
- PDF 임베딩: $0 (캐시 사용)
- Excel 임베딩: $0 (캐시 사용)
- 검색 API 호출: $0 (캐시 사용)
- LLM 매칭 호출: $0 (캐시 사용)

**예상 절약 효과**: 테스트당 $4-10 정도의 크레딧 절약

## ⚙️ 설정 옵션

### 캐시 디렉토리 변경
```python
from cache_manager import CacheManager

# 사용자 정의 캐시 디렉토리
custom_cache = CacheManager(cache_dir="my_cache")
```

### 캐시 만료 기간 조정
```python
# 30일로 만료 기간 연장
cache_manager.clear_expired_cache(max_age_days=30)
```

## 🔧 문제 해결

### 캐시 로드 실패
```python
# 모든 캐시 삭제 후 재생성
cache_manager.clear_all_cache()
# main_graph.py 재실행
```

### 파일 경로 변경
- PDF나 Excel 파일 경로가 변경되면 자동으로 캐시가 무효화됩니다
- 파일 해시가 변경되면 벡터 DB가 재생성됩니다

### 메모리 부족
```python
# 오래된 캐시 정리
cache_manager.clear_expired_cache(max_age_days=3)
```

## 📊 모니터링

### 캐시 히트율 확인
- 로그에서 `📂 캐시 사용` 메시지 확인
- `📂 PDF 검색 캐시 히트` 등의 메시지로 캐시 효과 확인

### 캐시 크기 모니터링
```python
stats = cache_manager.get_cache_stats()
print(f"캐시 크기: {stats['total_cache_size_mb']}MB")
```

## ⚠️ 주의사항

1. **첫 실행 시간**: 캐시 생성으로 인해 첫 실행은 오래 걸릴 수 있습니다
2. **디스크 공간**: 캐시 파일들이 디스크 공간을 사용합니다
3. **파일 변경**: 원본 파일이 변경되면 관련 캐시가 자동으로 무효화됩니다
4. **API 키**: 외부 검색 캐시는 Tavily API 키가 필요합니다

## 🎉 효과

- **테스트 속도**: 2-3배 빨라짐
- **API 비용**: 80-90% 절약
- **개발 효율성**: 빠른 반복 테스트 가능

## 🆕 새로운 기능: LLM 매칭 캐싱

### 배치 처리 최적화
- 약품을 15개씩 묶어서 LLM에 전송
- 개별 약품 호출 대비 90% 이상의 API 호출 감소
- 응답 시간 단축 및 비용 절약

### 스마트 캐싱
- 약품 정보와 증상의 해시 기반 캐시 키 생성
- 동일한 약품-증상 조합에 대한 중복 LLM 호출 방지
- 캐시 히트 시 즉시 결과 반환

### 사용 예시
```python
# 배치 크기 조정 (기본값: 15)
relevant_medicines = batch_medicine_matching(
    medicines_info, 
    condition, 
    batch_size=20  # 더 큰 배치로 처리
)

# 캐시 통계 확인
from cache_manager import print_cache_stats
print_cache_stats()
``` 