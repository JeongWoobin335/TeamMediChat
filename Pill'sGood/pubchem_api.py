# pubchem_api.py - PubChem API 연동 모듈 (개선된 버전)

import requests
import json
import time
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from cache_manager import cache_manager
from translation_rag import TranslationRAG

class PubChemAPI:
    """PubChem API 연동 클래스 (개선된 버전)"""
    
    def __init__(self):
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.pug_view_base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound"
        self.request_delay = 0.5  # API 요청 간격 (초)
        self.translation_rag = TranslationRAG()
    
    def _get_english_name(self, ingredient_name: str) -> str:
        """한국어 성분명을 영어명으로 변환 (LLM 기반)"""
        try:
            english_name = self.translation_rag.translate_korean_to_english(ingredient_name)
            print(f"🔄 성분명 변환: {ingredient_name} → {english_name}")
            return english_name
        except Exception as e:
            print(f"⚠️ 성분명 변환 오류: {e}")
            return ingredient_name  # 실패 시 원본 반환
    
    def _make_request(self, url: str, cache_key: str = None) -> Dict:
        """API 요청 실행 (캐시 포함)"""
        try:
            # 캐시 확인
            if cache_key:
                cached_result = cache_manager.get_search_cache(cache_key, "pubchem")
                if cached_result is not None:
                    print(f"📂 PubChem 캐시 히트: {cache_key}")
                    return cached_result
            
            print(f"🔍 PubChem API 요청: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # PropertyTable 구조에서 데이터 추출
            if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                result = data['PropertyTable']['Properties'][0]
            else:
                result = data
            
            # 캐시 저장
            if cache_key:
                cache_manager.save_search_cache(cache_key, "pubchem", result)
            
            # API 요청 간격
            time.sleep(self.request_delay)
            
            return result
            
        except Exception as e:
            print(f"❌ PubChem API 오류: {e}")
            return {}
    
    def _make_pug_view_request(self, cid: str, heading: str, cache_key: str = None) -> Dict:
        """PUG View API 요청 (작용기전, 효능 등 상세 정보)"""
        try:
            # 캐시 확인
            if cache_key:
                cached_result = cache_manager.get_search_cache(cache_key, "pubchem")
                if cached_result is not None:
                    print(f"📂 PubChem PUG View 캐시 히트: {cache_key}")
                    return cached_result
            
            url = f"{self.pug_view_base}/{cid}/JSON/?heading={heading}"
            print(f"🔍 PubChem PUG View API 요청: {url}")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # 캐시 저장
            if cache_key:
                cache_manager.save_search_cache(cache_key, "pubchem", data)
            
            # API 요청 간격
            time.sleep(self.request_delay)
            
            return data
            
        except Exception as e:
            print(f"❌ PubChem PUG View API 오류: {e}")
            return {}
    
    def get_compound_cid(self, compound_name: str) -> Optional[str]:
        """화합물 CID (Compound ID) 가져오기"""
        english_name = self._get_english_name(compound_name)
        cache_key = f"pubchem_cid_{english_name}"
        
        try:
            url = f"{self.base_url}/compound/name/{english_name}/cids/JSON"
            data = self._make_request(url, cache_key)
            
            if 'IdentifierList' in data and 'CID' in data['IdentifierList']:
                cids = data['IdentifierList']['CID']
                if cids:
                    return str(cids[0])
            return None
        except:
            return None
    
    def get_compound_basic_info(self, compound_name: str) -> Dict:
        """화합물 기본 정보 가져오기 (개선된 버전)"""
        english_name = self._get_english_name(compound_name)
        cache_key = f"pubchem_basic_{english_name}"
        
        # 실제 존재하는 속성만 요청
        properties = [
            'MolecularFormula', 'MolecularWeight', 'IUPACName',
            'CanonicalSMILES', 'IsomericSMILES', 'InChI', 'InChIKey'
        ]
        
        properties_str = ','.join(properties)
        url = f"{self.base_url}/compound/name/{english_name}/property/{properties_str}/JSON"
        
        return self._make_request(url, cache_key)
    
    def get_compound_pharmacology_info(self, compound_name: str) -> Dict:
        """화합물 약리학 정보 가져오기 (PUG View API 사용)"""
        english_name = self._get_english_name(compound_name)
        cid = self.get_compound_cid(english_name)
        
        if not cid:
            return {}
        
        cache_key = f"pubchem_pharmacology_{english_name}"
        
        # Pharmacology and Biochemistry 섹션 요청
        data = self._make_pug_view_request(cid, "Pharmacology+and+Biochemistry", cache_key)
        
        return self._extract_pharmacology_data(data)
    
    def _extract_pharmacology_data(self, data: Dict) -> Dict:
        """PUG View 데이터에서 약리학 정보 추출"""
        result = {
            'mechanism_of_action': '',
            'pharmacodynamics': '',
            'pharmacokinetics': '',
            'therapeutic_uses': [],
            'side_effects': [],
            'drug_interactions': [],
            'atc_codes': [],
            'mesh_classification': []
        }
        
        try:
            if 'Record' in data and 'Section' in data['Record']:
                for section in data['Record']['Section']:
                    if section.get('TOCHeading') == 'Pharmacology and Biochemistry':
                        for subsection in section.get('Section', []):
                            heading = subsection.get('TOCHeading', '')
                            
                            if heading == 'Pharmacodynamics':
                                result['pharmacodynamics'] = self._extract_text_from_section(subsection)
                            elif heading == 'Mechanism of Action':
                                result['mechanism_of_action'] = self._extract_text_from_section(subsection)
                            elif heading == 'ATC Code':
                                result['atc_codes'] = self._extract_atc_codes(subsection)
                            elif heading == 'MeSH Pharmacological Classification':
                                result['mesh_classification'] = self._extract_mesh_classification(subsection)
                                
        except Exception as e:
            print(f"⚠️ 약리학 데이터 추출 오류: {e}")
        
        return result
    
    def _extract_text_from_section(self, section: Dict) -> str:
        """섹션에서 텍스트 추출"""
        try:
            if 'Information' in section:
                for info in section['Information']:
                    if 'Value' in info and 'StringWithMarkup' in info['Value']:
                        for markup in info['Value']['StringWithMarkup']:
                            if 'String' in markup:
                                return markup['String']
        except:
            pass
        return ''
    
    def _extract_atc_codes(self, section: Dict) -> List[str]:
        """ATC 코드 추출"""
        codes = []
        try:
            if 'Information' in section:
                for info in section['Information']:
                    if 'Value' in info and 'StringWithMarkup' in info['Value']:
                        for markup in info['Value']['StringWithMarkup']:
                            if 'String' in markup:
                                codes.append(markup['String'])
        except:
            pass
        return codes
    
    def _extract_mesh_classification(self, section: Dict) -> List[Dict]:
        """MeSH 분류 추출"""
        classifications = []
        try:
            if 'Information' in section:
                for info in section['Information']:
                    classification = {
                        'name': info.get('Name', ''),
                        'description': ''
                    }
                    if 'Value' in info and 'StringWithMarkup' in info['Value']:
                        for markup in info['Value']['StringWithMarkup']:
                            if 'String' in markup:
                                classification['description'] = markup['String']
                                break
                    classifications.append(classification)
        except:
            pass
        return classifications
    
    def get_compound_description(self, compound_name: str) -> str:
        """화합물 설명 정보 가져오기 (개선된 버전)"""
        english_name = self._get_english_name(compound_name)
        cache_key = f"pubchem_description_{english_name}"
        url = f"{self.base_url}/compound/name/{english_name}/description/JSON"
        
        try:
            data = self._make_request(url, cache_key)
            if 'InformationList' in data and 'Information' in data['InformationList']:
                return data['InformationList']['Information'][0].get('Description', '')
            return ''
        except:
            return ''
    
    def get_compound_synonyms(self, compound_name: str) -> List[str]:
        """화합물 동의어 목록 가져오기 (개선된 버전)"""
        english_name = self._get_english_name(compound_name)
        cache_key = f"pubchem_synonyms_{english_name}"
        url = f"{self.base_url}/compound/name/{english_name}/synonyms/JSON"
        
        try:
            data = self._make_request(url, cache_key)
            if 'InformationList' in data and 'Information' in data['InformationList']:
                return data['InformationList']['Information'][0].get('Synonym', [])
            return []
        except:
            return []
    
    def search_compounds_by_smiles(self, smiles: str) -> List[Dict]:
        """SMILES 구조로 화합물 검색"""
        cache_key = f"pubchem_smiles_{smiles}"
        url = f"{self.base_url}/compound/smiles/{smiles}/property/MolecularFormula,MolecularWeight,IUPACName/JSON"
        
        try:
            data = self._make_request(url, cache_key)
            if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                return data['PropertyTable']['Properties']
            return []
        except:
            return []
    
    def get_compound_xrefs(self, compound_name: str) -> Dict:
        """외부 데이터베이스 참조 정보 가져오기"""
        english_name = self._get_english_name(compound_name)
        cache_key = f"pubchem_xrefs_{english_name}"
        url = f"{self.base_url}/compound/name/{english_name}/xrefs/JSON"
        
        return self._make_request(url, cache_key)
    
    def analyze_ingredient_comprehensive(self, ingredient_name: str) -> Dict:
        """성분 종합 분석 (병렬 처리 버전)"""
        print(f"🔍 PubChem 종합 분석: {ingredient_name}")
        
        result = {
            'ingredient_name': ingredient_name,
            'english_name': self._get_english_name(ingredient_name),
            'basic_info': {},
            'pharmacology_info': {},
            'description': '',
            'synonyms': [],
            'xrefs': {},
            'cid': None,
            'analysis_timestamp': time.time()
        }
        
        try:
            # 1. CID 가져오기 (먼저 실행, 다른 정보 수집에 필요)
            result['cid'] = self.get_compound_cid(ingredient_name)
            english_name = result['english_name']
            
            if not result['cid']:
                print(f"⚠️ CID를 찾을 수 없어 추가 정보 수집 불가: {ingredient_name}")
                return result
            
            # 2-6. 나머지 정보들을 병렬로 수집
            print("  🔄 병렬 정보 수집 시작...")
            
            def collect_basic_info():
                """기본 정보 수집"""
                try:
                    print("  📊 기본 정보 수집 중...")
                    return self.get_compound_basic_info(ingredient_name)
                except Exception as e:
                    print(f"⚠️ 기본 정보 수집 오류: {e}")
                    return {}
            
            def collect_pharmacology_info():
                """약리학 정보 수집"""
                try:
                    print("  📋 약리학 정보 수집 중...")
                    return self.get_compound_pharmacology_info(ingredient_name)
                except Exception as e:
                    print(f"⚠️ 약리학 정보 수집 오류: {e}")
                    return {}
            
            def collect_description():
                """설명 정보 수집"""
                try:
                    print("  📝 설명 정보 수집 중...")
                    return self.get_compound_description(ingredient_name)
                except Exception as e:
                    print(f"⚠️ 설명 정보 수집 오류: {e}")
                    return ''
            
            def collect_synonyms():
                """동의어 목록 수집"""
                try:
                    print("  🔤 동의어 목록 수집 중...")
                    return self.get_compound_synonyms(ingredient_name)
                except Exception as e:
                    print(f"⚠️ 동의어 목록 수집 오류: {e}")
                    return []
            
            def collect_xrefs():
                """외부 참조 수집"""
                try:
                    print("  🔗 외부 참조 수집 중...")
                    return self.get_compound_xrefs(ingredient_name)
                except Exception as e:
                    print(f"⚠️ 외부 참조 수집 오류: {e}")
                    return {}
            
            # 병렬 실행
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(collect_basic_info): 'basic_info',
                    executor.submit(collect_pharmacology_info): 'pharmacology_info',
                    executor.submit(collect_description): 'description',
                    executor.submit(collect_synonyms): 'synonyms',
                    executor.submit(collect_xrefs): 'xrefs'
                }
                
                # 결과 수집
                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        result[key] = future.result()
                        print(f"  ✅ {key} 수집 완료")
                    except Exception as e:
                        print(f"  ❌ {key} 수집 실패: {e}")
                        # 기본값 설정
                        if key == 'description':
                            result[key] = ''
                        elif key == 'synonyms':
                            result[key] = []
                        else:
                            result[key] = {}
            
            print(f"✅ PubChem 분석 완료: {ingredient_name}")
            
        except Exception as e:
            print(f"❌ PubChem 분석 오류: {e}")
            result['error'] = str(e)
        
        return result
