# translation_rag.py - 영어 약리학 정보를 한국어로 번역하는 전용 RAG

import time
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# answer_utils 대신 직접 LLM 사용
from dotenv import load_dotenv

load_dotenv()

class TranslationRAG:
    """영어 약리학 정보를 한국어로 번역하는 전용 RAG 시스템"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        # 🚀 성능 최적화: 자주 쓰는 성분명 사전 구축 (LLM 호출 없이 즉시 반환)
        self.korean_to_english_dict = {
            "아세트아미노펜": "acetaminophen",
            "이부프로펜": "ibuprofen",
            "나프록센": "naproxen",
            "디클로페낙": "diclofenac",
            "케토프로펜": "ketoprofen",
            "멜록시캠": "meloxicam",
            "셀레콕시브": "celecoxib",
            "카페인": "caffeine",
            "카페인무수물": "anhydrous caffeine",
            "푸르설티아민": "fursultiamine",
            "푸르설티아민염산염": "fursultiamine hydrochloride",
            "디펜히드라민": "diphenhydramine",
            "클로르페니라민": "chlorpheniramine",
            "로라타딘": "loratadine",
            "세티리진": "cetirizine",
            "펙소페나딘": "fexofenadine",
            "덱스부프로펜": "dexibuprofen",
            "트라마돌": "tramadol",
            "코데인": "codeine",
            "옥시코돈": "oxycodone",
            "모르핀": "morphine",
            "프레드니솔론": "prednisolone",
            "덱사메타손": "dexamethasone",
            "하이드로코르티손": "hydrocortisone",
            "베타메타손": "betamethasone",
            "아목시실린": "amoxicillin",
            "아목시실린트리하이드레이트": "amoxicillin trihydrate",
            "세팔렉신": "cephalexin",
            "아지트로마이신": "azithromycin",
            "클라리트로마이신": "clarithromycin",
            "도시사이클린": "doxycycline",
            "테트라사이클린": "tetracycline",
            "시프로플록사신": "ciprofloxacin",
            "레보플록사신": "levofloxacin",
            "메트로니다졸": "metronidazole",
            "클린다마이신": "clindamycin",
            "옥시부티닌": "oxybutynin",
            "톨테로딘": "tolterodine",
            "솔리페나신": "solifenacin",
            "다리페나신": "darifenacin",
            "옴페프라졸": "omeprazole",
            "란소프라졸": "lansoprazole",
            "에소메프라졸": "esomeprazole",
            "판토프라졸": "pantoprazole",
            "라베프라졸": "rabeprazole",
            "란티딘": "ranitidine",
            "파모티딘": "famotidine",
            "시메티딘": "cimetidine",
            "니자티딘": "nizatidine",
            "도메페리돈": "domperidone",
            "메토클로프라미드": "metoclopramide",
            "시사프리드": "cisapride",
            "모사프리드": "mosapride",
            "비스무트": "bismuth",
            "수크랄페이트": "sucralfate",
            "알긴산나트륨": "sodium alginate",
            "알마겔": "aluminum hydroxide",
            "마그네슘하이드록사이드": "magnesium hydroxide",
            "시메티콘": "simethicone",
            "디메티콘": "dimethicone",
            "락툴로스": "lactulose",
            "비사코딜": "bisacodyl",
            "세나": "senna",
            "프로바이오틱스": "probiotics",
            "락토바실러스": "lactobacillus",
            "비피도박테리움": "bifidobacterium",
            "파라세타몰": "paracetamol",  # 아세트아미노펜의 다른 이름
            "아스피린": "aspirin",
            "살리실산": "salicylic acid",
            "살리실아마이드": "salicylamide",
            "인도메타신": "indomethacin",
            "피로시캠": "piroxicam",
            "테노시캠": "tenoxicam",
            "로페콕시브": "rofecoxib",
            "발데콕시브": "valdecoxib",
            "에토리콕시브": "etoricoxib",
            "파레콕시브": "parecoxib",
            "부프로펜": "buprofen",
            "플루비프로펜": "flurbiprofen",
            "옥사프로진": "oxaprozin",
            "피록시캠": "piroxicam",
            "펜타조신": "pentazocine",
            "부프레노르핀": "buprenorphine",
            "펜타닐": "fentanyl",
            "히드로모르폰": "hydromorphone",
            "메타돈": "methadone",
            "부프레노르핀": "buprenorphine",
            "날트렉손": "naltrexone",
            "날록손": "naloxone",
            "부프레노르핀": "buprenorphine",
            "펜타닐": "fentanyl",
            "히드로모르폰": "hydromorphone",
            "메타돈": "methadone",
            "부프레노르핀": "buprenorphine",
            "날트렉손": "naltrexone",
            "날록손": "naloxone",
        }
    
    def _generate_response(self, prompt: str, temperature: float = 0.1, max_tokens: int = 1000) -> str:
        """LLM을 사용하여 응답 생성"""
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"⚠️ LLM 응답 생성 오류: {e}")
            return ""
    
    def translate_pharmacology_info(self, english_info: Dict) -> Dict:
        """영어 약리학 정보를 한국어로 번역"""
        print(f"🔄 약리학 정보 번역 시작...")
        
        result = {
            'mechanism_of_action_kr': '',
            'pharmacodynamics_kr': '',
            'therapeutic_uses_kr': [],
            'side_effects_kr': [],
            'drug_interactions_kr': [],
            'atc_codes_kr': [],
            'mesh_classification_kr': [],
            'translation_timestamp': time.time()
        }
        
        try:
            # 1. 작용기전 번역
            if english_info.get('mechanism_of_action'):
                result['mechanism_of_action_kr'] = self._translate_mechanism_of_action(
                    english_info['mechanism_of_action']
                )
            
            # 2. 약력학 번역
            if english_info.get('pharmacodynamics'):
                result['pharmacodynamics_kr'] = self._translate_pharmacodynamics(
                    english_info['pharmacodynamics']
                )
            
            # 3. ATC 코드 한국어 설명
            if english_info.get('atc_codes'):
                result['atc_codes_kr'] = self._translate_atc_codes(
                    english_info['atc_codes']
                )
            
            # 4. MeSH 분류 한국어 설명
            if english_info.get('mesh_classification'):
                result['mesh_classification_kr'] = self._translate_mesh_classification(
                    english_info['mesh_classification']
                )
            
            # 핵심 정보 요약 생성
            result['summary_kr'] = self._create_summary(result)
            
            print(f"✅ 약리학 정보 번역 완료")
            
        except Exception as e:
            print(f"❌ 번역 중 오류 발생: {e}")
            result['error'] = str(e)
        
        return result
    
    def _translate_mechanism_of_action(self, english_text: str) -> str:
        """작용기전 영어 → 한국어 번역 (LLM 기반)"""
        prompt = f"""
당신은 의학 전문 번역가입니다. 다음 영어 작용기전 정보를 정확하고 자연스러운 한국어로 번역해주세요.

**번역 원칙:**
1. **의학 용어는 정확한 한국어 의학 용어로 자동 번역** (사전 없이 LLM이 판단)
2. 일반인도 이해할 수 있도록 설명하되, 전문성은 유지
3. 문맥에 맞는 자연스러운 한국어로 번역
4. 핵심 내용은 놓치지 않도록 주의
5. 복잡한 의학 개념은 이해하기 쉽게 풀어서 설명

**번역 스타일:**
- 전문적이지만 읽기 쉽게
- 문장을 너무 길지 않게 (3-4문장으로 나누어)
- 중요한 키워드는 괄호 안에 영어 원문도 함께 표기

**영어 원문:**
{english_text}

**한국어 번역:**
"""
        
        try:
            response = self._generate_response(prompt)
            return response.strip()
        except Exception as e:
            print(f"⚠️ 작용기전 번역 오류: {e}")
            return "작용기전 정보를 번역할 수 없습니다."
    
    def _translate_pharmacodynamics(self, english_text: str) -> str:
        """약력학 영어 → 한국어 번역 (LLM 기반)"""
        prompt = f"""
당신은 의학 전문 번역가입니다. 다음 영어 약력학 정보를 정확하고 자연스러운 한국어로 번역해주세요.

**번역 원칙:**
1. **의학 용어는 정확한 한국어 의학 용어로 자동 번역** (사전 없이 LLM이 판단)
2. 약물의 효과와 작용을 명확하고 이해하기 쉽게 설명
3. 일반인도 이해할 수 있도록 번역하되, 전문성은 유지
4. 문맥에 맞는 자연스러운 한국어로 번역
5. 약물의 효과와 부작용을 구분하여 명확하게 설명

**번역 스타일:**
- 전문적이지만 읽기 쉽게
- 문장을 너무 길지 않게 (3-4문장으로 나누어)
- 중요한 키워드는 괄호 안에 영어 원문도 함께 표기
- 약물의 주요 효과는 강조하여 표기

**영어 원문:**
{english_text}

**한국어 번역:**
"""
        
        try:
            response = self._generate_response(prompt)
            return response.strip()
        except Exception as e:
            print(f"⚠️ 약력학 번역 오류: {e}")
            return "약력학 정보를 번역할 수 없습니다."
    
    def _translate_atc_codes(self, atc_codes: List[str]) -> List[Dict]:
        """ATC 코드를 한국어 설명으로 번역 (LLM 기반)"""
        if not atc_codes:
            return []
        
        # ATC 코드들을 하나의 문자열로 결합
        codes_text = ', '.join(atc_codes)
        
        prompt = f"""
당신은 의학 전문가입니다. 다음 ATC 코드들을 한국어로 번역하고 설명해주세요.

**ATC 코드:**
{codes_text}

**번역 원칙:**
1. 각 ATC 코드의 의미를 정확한 한국어로 번역
2. 의학 분야별 분류를 명확하게 설명
3. 일반인도 이해할 수 있도록 설명

**출력 형식:**
각 코드에 대해 다음과 같이 설명해주세요:
- 코드: 한국어 설명

**한국어 번역:**
"""
        
        try:
            response = self._generate_response(prompt)
            
            # 응답을 파싱하여 구조화된 데이터로 변환
            translated_codes = []
            lines = response.strip().split('\n')
            
            for line in lines:
                if ':' in line and line.strip():
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        code = parts[0].strip().replace('-', '').replace('*', '')
                        description = parts[1].strip()
                        translated_codes.append({
                            'code': code,
                            'korean_description': description
                        })
            
            return translated_codes
            
        except Exception as e:
            print(f"⚠️ ATC 코드 번역 오류: {e}")
            return [{'code': code, 'korean_description': f"ATC 코드 {code}"} for code in atc_codes]
    
    def _translate_mesh_classification(self, mesh_classifications: List[Dict]) -> List[Dict]:
        """MeSH 분류를 한국어로 번역 (LLM 기반)"""
        if not mesh_classifications:
            return []
        
        # MeSH 분류들을 하나의 문자열로 결합
        classifications_text = ""
        for i, classification in enumerate(mesh_classifications, 1):
            name = classification.get('name', '')
            description = classification.get('description', '')
            classifications_text += f"{i}. {name}: {description}\n"
        
        prompt = f"""
당신은 의학 전문가입니다. 다음 MeSH 분류들을 한국어로 번역해주세요.

**MeSH 분류:**
{classifications_text}

**번역 원칙:**
1. 각 분류명을 정확한 한국어 의학 용어로 번역
2. 설명을 이해하기 쉽고 자연스러운 한국어로 번역
3. 전문적이지만 일반인도 이해할 수 있도록 번역

**출력 형식:**
각 분류에 대해 다음과 같이 번역해주세요:
- 분류명: 한국어 설명

**한국어 번역:**
"""
        
        try:
            response = self._generate_response(prompt)
            
            # 응답을 파싱하여 구조화된 데이터로 변환
            translated_classifications = []
            lines = response.strip().split('\n')
            
            for i, line in enumerate(lines):
                if ':' in line and line.strip():
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        korean_name = parts[0].strip().replace('-', '').replace('*', '')
                        korean_description = parts[1].strip()
                        
                        # 원본 정보와 매칭
                        original = mesh_classifications[i] if i < len(mesh_classifications) else {}
                        
                        translated_classifications.append({
                            'korean_name': korean_name,
                            'korean_description': korean_description,
                            'original_name': original.get('name', ''),
                            'original_description': original.get('description', '')
                        })
            
            return translated_classifications
            
        except Exception as e:
            print(f"⚠️ MeSH 분류 번역 오류: {e}")
            return [{
                'korean_name': classification.get('name', ''),
                'korean_description': classification.get('description', ''),
                'original_name': classification.get('name', ''),
                'original_description': classification.get('description', '')
            } for classification in mesh_classifications]
    
    def translate_comprehensive_ingredient_info(self, pubchem_result: Dict) -> Dict:
        """PubChem 결과 전체를 한국어로 번역"""
        print(f"🔄 종합 성분 정보 번역 시작...")
        
        result = {
            'ingredient_name': pubchem_result.get('ingredient_name', ''),
            'english_name': pubchem_result.get('english_name', ''),
            'basic_info_kr': {},
            'pharmacology_info_kr': {},
            'description_kr': '',
            'synonyms_kr': [],
            'translation_timestamp': time.time()
        }
        
        try:
            # 1. 기본 정보 한국어 설명
            if pubchem_result.get('basic_info'):
                result['basic_info_kr'] = self._translate_basic_info(
                    pubchem_result['basic_info']
                )
            
            # 2. 약리학 정보 번역
            if pubchem_result.get('pharmacology_info'):
                result['pharmacology_info_kr'] = self.translate_pharmacology_info(
                    pubchem_result['pharmacology_info']
                )
            
            # 3. 설명 정보 번역
            if pubchem_result.get('description'):
                result['description_kr'] = self._translate_description(
                    pubchem_result['description']
                )
            
            # 4. 동의어 목록 (영어 그대로 유지)
            result['synonyms_kr'] = pubchem_result.get('synonyms', [])
            
            print(f"✅ 종합 성분 정보 번역 완료")
            
        except Exception as e:
            print(f"❌ 종합 번역 중 오류 발생: {e}")
            result['error'] = str(e)
        
        return result
    
    def _translate_basic_info(self, basic_info: Dict) -> Dict:
        """기본 정보를 한국어로 번역 (LLM 기반)"""
        if not basic_info:
            return {}
        
        # 기본 정보를 문자열로 변환
        info_text = ""
        for key, value in basic_info.items():
            if value:
                info_text += f"{key}: {value}\n"
        
        prompt = f"""
당신은 화학 전문가입니다. 다음 화합물의 기본 정보를 한국어로 번역해주세요.

**영어 원문:**
{info_text}

**번역 원칙:**
1. 화학 용어는 정확한 한국어로 번역
2. 일반인도 이해할 수 있도록 설명
3. 전문적이지만 읽기 쉽게 번역

**출력 형식:**
각 항목에 대해 다음과 같이 번역해주세요:
- 한국어 항목명: 값 또는 설명

**한국어 번역:**
"""
        
        try:
            response = self._generate_response(prompt)
            
            # 응답을 파싱하여 구조화된 데이터로 변환
            korean_info = {}
            lines = response.strip().split('\n')
            
            for line in lines:
                if ':' in line and line.strip():
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        korean_key = parts[0].strip()
                        korean_value = parts[1].strip()
                        korean_info[korean_key] = korean_value
            
            return korean_info
            
        except Exception as e:
            print(f"⚠️ 기본 정보 번역 오류: {e}")
            # 기본 매핑으로 폴백
            korean_info = {}
            for key, value in basic_info.items():
                if key == 'MolecularFormula':
                    korean_info['분자식'] = value
                elif key == 'MolecularWeight':
                    korean_info['분자량'] = f"{value} g/mol"
                elif key == 'IUPACName':
                    korean_info['IUPAC명'] = value
                else:
                    korean_info[key] = value
            return korean_info
    
    def _translate_description(self, english_description: str) -> str:
        """설명 정보를 한국어로 번역 (LLM 기반)"""
        if not english_description:
            return ""
        
        prompt = f"""
당신은 의학 전문 번역가입니다. 다음 영어 약물 설명을 정확하고 자연스러운 한국어로 번역해주세요.

**번역 원칙:**
1. 의학 용어는 정확한 한국어 의학 용어로 번역
2. 일반인도 이해할 수 있도록 설명
3. 전문적이지만 읽기 쉽게 번역
4. 문맥에 맞는 자연스러운 한국어로 번역

**영어 원문:**
{english_description}

**한국어 번역:**
"""
        
        try:
            response = self._generate_response(prompt)
            return response.strip()
        except Exception as e:
            print(f"⚠️ 설명 번역 오류: {e}")
            return english_description
    
    def translate_korean_to_english(self, korean_text: str) -> str:
        """한국어를 영어로 번역 (성분명 변환용) - 🚀 성능 최적화: 사전 우선 사용"""
        if not korean_text:
            return ""
        
        # 🚀 성능 최적화: 사전에 있으면 즉시 반환 (LLM 호출 없음)
        korean_clean = korean_text.strip()
        if korean_clean in self.korean_to_english_dict:
            english_name = self.korean_to_english_dict[korean_clean]
            print(f"📚 사전에서 발견 (LLM 스킵): '{korean_clean}' → '{english_name}'")
            return english_name
        
        # 사전에 없으면 LLM 호출
        prompt = f"""
당신은 의학 전문가입니다. 다음 한국어 성분명을 정확한 영어명으로 변환해주세요.

**한국어 성분명:** {korean_text}

**변환 원칙:**
1. 정확한 영어 의학 용어 사용
2. 일반적인 상품명이 아닌 성분명으로 변환
3. 표준화된 영어명 사용
4. 복합 성분의 경우 각각을 영어로 변환
5. 불필요한 설명 없이 영어명만 반환

**영어명:**
"""
        
        try:
            response = self._generate_response(prompt)
            english_name = response.strip()
            # 🚀 성능 최적화: 변환 결과를 사전에 추가 (다음번에는 LLM 호출 없음)
            if english_name and english_name != korean_text:
                self.korean_to_english_dict[korean_clean] = english_name
                print(f"💾 사전에 추가: '{korean_clean}' → '{english_name}'")
            return english_name
        except Exception as e:
            print(f"⚠️ 한국어→영어 번역 오류: {e}")
            return korean_text  # 실패 시 원본 반환
    
    def _create_summary(self, translated_info: Dict) -> str:
        """번역된 정보를 상세하게 정리 (덜 요약, 더 많은 정보 활용)"""
        print(f"🔄 상세 정보 정리 중...")
        
        # 모든 가용 정보 추출
        mechanism = translated_info.get('mechanism_of_action_kr', '')
        pharmacodynamics = translated_info.get('pharmacodynamics_kr', '')
        atc_codes = translated_info.get('atc_codes_kr', [])
        mesh_classifications = translated_info.get('mesh_classification_kr', [])
        
        if not mechanism and not pharmacodynamics and not atc_codes and not mesh_classifications:
            return "약리학 정보가 없습니다."
        
        # ATC 코드 정리
        atc_info = ""
        if atc_codes:
            atc_info = "**의약품 분류 (ATC):**\n"
            for atc in atc_codes[:3]:  # 상위 3개만
                code = atc.get('code', '')
                description = atc.get('korean_description', '')
                if code and description:
                    atc_info += f"- {code}: {description}\n"
        
        # MeSH 분류 정리
        mesh_info = ""
        if mesh_classifications:
            mesh_info = "**약리학적 분류 (MeSH):**\n"
            for mesh in mesh_classifications[:3]:  # 상위 3개만
                name = mesh.get('korean_name', '')
                description = mesh.get('korean_description', '')
                if name and description:
                    mesh_info += f"- {name}: {description}\n"
        
        prompt = f"""
당신은 의학 전문가입니다. 다음 약물의 약리학 정보를 **상세하고 포괄적으로** 정리해주세요.

**작용기전 (Mechanism of Action):**
{mechanism if mechanism else "정보 없음"}

**약력학 (Pharmacodynamics):**
{pharmacodynamics if pharmacodynamics else "정보 없음"}

{atc_info}

{mesh_info}

**정리 원칙:**
1. **상세하게 설명** - 4-6문장으로 충분히 설명 (2-3문장 X)
2. **모든 제공된 정보를 최대한 활용** - 작용기전, 약력학, 분류 정보 모두 포함
3. 작용 위치 + 작용 방식 + 주요 효과 + 약리학적 특성 모두 설명
4. **전문적인 의학 용어도 포함** (분류, 억제 메커니즘, 수용체 등)
5. 전문 용어는 괄호 안에 영어 원문도 함께 표기
6. 일반인이 이해할 수 있도록 설명은 추가하되, 정보는 생략하지 않음
7. ATC 코드나 MeSH 분류가 있다면 이것도 자연스럽게 언급
8. 작용기전 → 약력학 → 분류 → 주요 효과 순으로 논리적으로 연결

**상세 정리:**
"""
        
        try:
            response = self._generate_response(prompt, max_tokens=2000)  # 토큰 증가
            print(f"✅ 상세 정보 정리 완료 (길이: {len(response)}자)")
            return response
        except Exception as e:
            print(f"⚠️ 정리 생성 오류: {e}")
            # 폴백: 모든 정보를 단순 연결
            fallback_text = ""
            if mechanism:
                fallback_text += f"**작용기전:** {mechanism}\n\n"
            if pharmacodynamics:
                fallback_text += f"**약력학:** {pharmacodynamics}\n\n"
            if atc_codes:
                fallback_text += f"**의약품 분류:** {', '.join([atc.get('korean_description', '') for atc in atc_codes[:3]])}\n\n"
            if mesh_classifications:
                fallback_text += f"**약리학적 분류:** {', '.join([mesh.get('korean_name', '') for mesh in mesh_classifications[:3]])}\n\n"
            
            return fallback_text if fallback_text else "약리학 정보가 없습니다."
