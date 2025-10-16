import os
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class CacheManager:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 캐시 하위 디렉토리들
        self.vector_cache_dir = self.cache_dir / "vectors"
        self.search_cache_dir = self.cache_dir / "search"
        self.embedding_cache_dir = self.cache_dir / "embeddings"
        self.matching_cache_dir = self.cache_dir / "matching"  # LLM 매칭 결과 캐시
        
        for dir_path in [self.vector_cache_dir, self.search_cache_dir, self.embedding_cache_dir, self.matching_cache_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def _get_file_hash(self, file_path: str) -> str:
        """파일의 해시값을 계산하여 캐시 키로 사용"""
        if not os.path.exists(file_path):
            return "nonexistent"
        
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_data_hash(self, data: Any) -> str:
        """데이터의 해시값을 계산"""
        if isinstance(data, str):
            return hashlib.md5(data.encode()).hexdigest()
        elif isinstance(data, list):
            return hashlib.md5(str(sorted(data)).encode()).hexdigest()
        elif isinstance(data, dict):
            # 딕셔너리의 경우 키-값 쌍을 정렬하여 일관된 해시 생성
            sorted_items = sorted(data.items())
            return hashlib.md5(str(sorted_items).encode()).hexdigest()
        else:
            return hashlib.md5(str(data).encode()).hexdigest()
    
    def get_cache_key(self, source_type: str, identifier: str, data_hash: str = None) -> str:
        """캐시 키 생성"""
        if data_hash:
            return f"{source_type}_{identifier}_{data_hash}"
        return f"{source_type}_{identifier}"
    
    def is_vector_cache_valid(self, source_type: str, file_paths: List[str]) -> bool:
        """벡터 캐시가 유효한지 확인"""
        cache_key = self.get_cache_key(source_type, "vector_db")
        cache_dir = self.vector_cache_dir / cache_key
        hash_file = self.vector_cache_dir / f"{cache_key}_hash.json"
        
        if not cache_dir.exists() or not hash_file.exists():
            return False
        
        try:
            with open(hash_file, 'r') as f:
                stored_hashes = json.load(f)
            
            # 파일 해시 비교
            current_hashes = {path: self._get_file_hash(path) for path in file_paths}
            return stored_hashes == current_hashes
        except:
            return False
    
    def is_docs_cache_valid(self, source_type: str) -> bool:
        """문서 캐시가 유효한지 확인"""
        if source_type == "excel":
            cache_key = self.get_cache_key(source_type, "excel_docs")
        elif source_type == "pdf":
            cache_key = self.get_cache_key(source_type, "pdf_docs")
        else:
            return False
            
        cache_file = self.vector_cache_dir / f"{cache_key}.pkl"
        return cache_file.exists()
    
    def save_vector_cache(self, source_type: str, file_paths: List[str], vector_db: FAISS):
        """벡터 DB 캐싱"""
        cache_key = self.get_cache_key(source_type, "vector_db")
        cache_dir = self.vector_cache_dir / cache_key
        hash_file = self.vector_cache_dir / f"{cache_key}_hash.json"
        
        try:
            # FAISS 벡터 DB를 디렉토리로 저장
            cache_dir.mkdir(exist_ok=True)
            vector_db.save_local(str(cache_dir))
            
            # 파일 해시 저장
            current_hashes = {path: self._get_file_hash(path) for path in file_paths}
            with open(hash_file, 'w') as f:
                json.dump(current_hashes, f)
            
            print(f"💾 {source_type} 벡터 DB 캐시 저장됨")
        except Exception as e:
            print(f"❌ {source_type} 벡터 DB 캐시 저장 실패: {e}")
            # 캐시 디렉토리가 부분적으로 생성된 경우 삭제
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
            if hash_file.exists():
                hash_file.unlink()
    
    def load_vector_cache(self, source_type: str, embedding_model=None) -> Optional[FAISS]:
        """벡터 DB 캐시 로드"""
        cache_key = self.get_cache_key(source_type, "vector_db")
        cache_dir = self.vector_cache_dir / cache_key
        
        if cache_dir.exists():
            try:
                # FAISS 벡터 DB를 디렉토리에서 로드 (보안 허용)
                if embedding_model is None:
                    from langchain_openai import OpenAIEmbeddings
                    embedding_model = OpenAIEmbeddings()
                vector_db = FAISS.load_local(str(cache_dir), embedding_model, allow_dangerous_deserialization=True)
                print(f"📂 {source_type} 벡터 DB 캐시 로드됨")
                return vector_db
            except Exception as e:
                print(f"❌ {source_type} 벡터 DB 캐시 로드 실패: {e}")
        
        return None
    
    def get_search_cache_key(self, query: str, source_type: str) -> str:
        """검색 캐시 키 생성"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return self.get_cache_key("search", f"{source_type}_{query_hash}")
    
    def get_search_cache(self, query: str, source_type: str) -> Optional[List[Document]]:
        """검색 결과 캐시 조회"""
        cache_key = self.get_search_cache_key(query, source_type)
        cache_file = self.search_cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    results = pickle.load(f)
                print(f"📂 {source_type} 검색 캐시 히트: {query[:30]}...")
                return results
            except Exception as e:
                print(f"❌ {source_type} 검색 캐시 로드 실패: {e}")
        
        return None
    
    def save_search_cache(self, query: str, source_type: str, results: List[Document]):
        """검색 결과 캐싱"""
        cache_key = self.get_search_cache_key(query, source_type)
        cache_file = self.search_cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
            print(f"💾 {source_type} 검색 결과 캐시 저장됨")
        except Exception as e:
            print(f"❌ {source_type} 검색 캐시 저장 실패: {e}")
    
    def get_matching_cache_key(self, condition: str, medicines_info: Dict[str, Any]) -> str:
        """약품-증상 매칭 캐시 키 생성"""
        # 조건과 약품 정보의 해시를 조합하여 캐시 키 생성
        condition_hash = hashlib.md5(condition.encode()).hexdigest()
        medicines_hash = self._get_data_hash(medicines_info)
        return f"matching_{condition_hash}_{medicines_hash[:16]}"
    
    def get_matching_cache(self, condition: str, medicines_info: Dict[str, Any]) -> Optional[Dict[str, bool]]:
        """약품-증상 매칭 결과 캐시 조회"""
        cache_key = self.get_matching_cache_key(condition, medicines_info)
        cache_file = self.matching_cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                print(f"📂 매칭 캐시 히트: {condition} - {len(medicines_info)}개 약품")
                return result
            except Exception as e:
                print(f"❌ 매칭 캐시 로드 실패: {e}")
        
        return None
    
    def save_matching_cache(self, condition: str, medicines_info: Dict[str, Any], matching_result: Dict[str, bool]):
        """약품-증상 매칭 결과 캐싱"""
        cache_key = self.get_matching_cache_key(condition, medicines_info)
        cache_file = self.matching_cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(matching_result, f)
            print(f"💾 매칭 결과 캐시 저장됨: {condition} - {len(medicines_info)}개 약품")
        except Exception as e:
            print(f"❌ 매칭 캐시 저장 실패: {e}")
    
    def save_excel_docs_cache(self, source_type: str, excel_docs: List[Document]):
        """Excel 문서 리스트 캐싱"""
        cache_key = self.get_cache_key(source_type, "excel_docs")
        cache_file = self.vector_cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(excel_docs, f)
            print(f"💾 Excel 문서 캐시 저장됨: {len(excel_docs)}개 문서")
        except Exception as e:
            print(f"❌ Excel 문서 캐시 저장 실패: {e}")
    
    def load_excel_docs_cache(self, source_type: str) -> Optional[List[Document]]:
        """Excel 문서 리스트 캐시 로드"""
        cache_key = self.get_cache_key(source_type, "excel_docs")
        cache_file = self.vector_cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    excel_docs = pickle.load(f)
                print(f"📂 Excel 문서 캐시 로드됨: {len(excel_docs)}개 문서")
                return excel_docs
            except Exception as e:
                print(f"❌ Excel 문서 캐시 로드 실패: {e}")
        
        return None
    
    def save_pdf_docs_cache(self, source_type: str, pdf_docs: List[Document]):
        """PDF 문서 리스트 캐싱"""
        cache_key = self.get_cache_key(source_type, "pdf_docs")
        cache_file = self.vector_cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(pdf_docs, f)
            print(f"💾 PDF 문서 캐시 저장됨: {len(pdf_docs)}개 문서")
        except Exception as e:
            print(f"❌ PDF 문서 캐시 저장 실패: {e}")
    
    def load_pdf_docs_cache(self, source_type: str) -> Optional[List[Document]]:
        """PDF 문서 리스트 캐시 로드"""
        cache_key = self.get_cache_key(source_type, "pdf_docs")
        cache_file = self.vector_cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    pdf_docs = pickle.load(f)
                print(f"📂 PDF 문서 캐시 로드됨: {len(pdf_docs)}개 문서")
                return pdf_docs
            except Exception as e:
                print(f"❌ PDF 문서 캐시 로드 실패: {e}")
        
        return None
    
    def clear_expired_cache(self, max_age_days: int = 7):
        """만료된 캐시 정리"""
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        for cache_dir in [self.search_cache_dir, self.embedding_cache_dir]:
            for cache_file in cache_dir.glob("*.pkl"):
                if cache_file.stat().st_mtime < cutoff_time.timestamp():
                    cache_file.unlink()
                    print(f"🗑️ 만료된 캐시 삭제: {cache_file.name}")
    
    def clear_all_cache(self):
        """모든 캐시 삭제"""
        for cache_dir in [self.vector_cache_dir, self.search_cache_dir, self.embedding_cache_dir, self.matching_cache_dir]:
            for cache_file in cache_dir.glob("*"):
                if cache_file.is_file():
                    cache_file.unlink()
                elif cache_file.is_dir():
                    import shutil
                    shutil.rmtree(cache_file)
        print("🗑️ 모든 캐시 삭제됨")
    
    def clear_docs_cache(self, source_type: str):
        """특정 소스의 문서 캐시만 삭제"""
        if source_type == "excel":
            cache_key = self.get_cache_key(source_type, "excel_docs")
        elif source_type == "pdf":
            cache_key = self.get_cache_key(source_type, "pdf_docs")
        else:
            print(f"❌ 지원하지 않는 소스 타입: {source_type}")
            return
            
        cache_file = self.vector_cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            cache_file.unlink()
            print(f"🗑️ {source_type} 문서 캐시 삭제됨")
        else:
            print(f"📝 {source_type} 문서 캐시가 이미 없음")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보"""
        # 벡터 캐시는 디렉토리로 저장되므로 디렉토리 개수로 계산
        vector_cache_count = len([d for d in self.vector_cache_dir.iterdir() if d.is_dir()])
        
        stats = {
            "vector_cache_count": vector_cache_count,
            "search_cache_count": len(list(self.search_cache_dir.glob("*.pkl"))),
            "embedding_cache_count": len(list(self.embedding_cache_dir.glob("*.pkl"))),
            "matching_cache_count": len(list(self.matching_cache_dir.glob("*.pkl"))),
            "total_cache_size_mb": 0
        }
        
        total_size = 0
        for cache_dir in [self.vector_cache_dir, self.search_cache_dir, self.embedding_cache_dir, self.matching_cache_dir]:
            for cache_file in cache_dir.glob("*"):
                if cache_file.is_file():
                    total_size += cache_file.stat().st_size
        
        stats["total_cache_size_mb"] = round(total_size / (1024 * 1024), 2)
        return stats

# 전역 캐시 매니저 인스턴스
cache_manager = CacheManager() 

def print_cache_stats():
    """캐시 통계를 출력하는 유틸리티 함수"""
    stats = cache_manager.get_cache_stats()
    print("\n📊 캐시 통계:")
    print(f"  - 벡터 캐시: {stats['vector_cache_count']}개")
    print(f"  - 검색 캐시: {stats['search_cache_count']}개")
    print(f"  - 임베딩 캐시: {stats['embedding_cache_count']}개")
    print(f"  - 매칭 캐시: {stats['matching_cache_count']}개")
    print(f"  - 총 캐시 크기: {stats['total_cache_size_mb']}MB")
    print() 