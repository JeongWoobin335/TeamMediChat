import os
import re
import json
import pandas as pd
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools.tavily_search import TavilySearchResults
from cache_manager import cache_manager

# === 공통 설정 ===
splitter = TokenTextSplitter(chunk_size=600, chunk_overlap=100)
embedding_model = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o", temperature=0)
hf_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
compressor = CrossEncoderReranker(model=hf_model, top_n=5)

# === PDF 인덱싱 및 검색기 ===
pdf_path = r"C:\Users\jung\Desktop\pdf\한국에서 널리 쓰이는 일반의약품 20선.pdf"

# 전역 변수 초기화
pdf_structured_docs = []
pdf_product_index = {}

# 캐시 확인
if cache_manager.is_vector_cache_valid("pdf", [pdf_path]) and cache_manager.is_docs_cache_valid("pdf"):
    print("📂 PDF 벡터 DB 및 문서 캐시 사용")
    pdf_vectordb = cache_manager.load_vector_cache("pdf", embedding_model)
    pdf_structured_docs = cache_manager.load_pdf_docs_cache("pdf")
    
    if pdf_vectordb is None or pdf_structured_docs is None:
        print("⚠️ PDF 캐시 로드 실패, 새로 생성합니다")
        pdf_vectordb = None
        pdf_structured_docs = []
        pdf_product_index = {}
    else:
        print("📂 PDF 벡터 DB 및 문서 캐시 로드됨")
        # pdf_product_index도 복원
        pdf_product_index = {}
        for doc in pdf_structured_docs:
            name = doc.metadata.get("제품명", "")
            if name:
                pdf_product_index.setdefault(name, []).append(doc)
else:
    print("🔄 PDF 벡터 DB 새로 생성")
    pdf_vectordb = None

if pdf_vectordb is None:
    pdf_docs_raw = PyPDFLoader(pdf_path).load()
    pdf_structured_docs = []
    pdf_product_index = {}

    for doc in pdf_docs_raw:
        blocks = re.findall(r"(\d+\.\s*.+?)(?=\n\d+\.|\Z)", doc.page_content, re.DOTALL)
        for block in blocks:
            name_match = re.match(r"\d+\.\s*([^\n(]+)", block)
            if name_match:
                name = name_match.group(1).strip()
                eff = re.search(r"주요 효능[:：]\s*(.*?)(?:\n|일반적인 부작용[:：])", block, re.DOTALL)
                side = re.search(r"일반적인 부작용[:：]\s*(.*?)(?:\n|성인 기준 복용법[:：])", block, re.DOTALL)
                usage = re.search(r"성인 기준 복용법[:：]\s*(.*?)(?:\n|$)", block, re.DOTALL)
                content = f"[제품명]: {name}\n[효능]: {eff.group(1).strip() if eff else '정보 없음'}\n[부작용]: {side.group(1).strip() if side else '정보 없음'}\n[사용법]: {usage.group(1).strip() if usage else '정보 없음'}"

                for chunk in splitter.split_text(content):
                    doc_obj = Document(page_content=chunk, metadata={"제품명": name})
                    pdf_structured_docs.append(doc_obj)

                doc_full = Document(page_content=content, metadata={"제품명": name})
                pdf_product_index.setdefault(name, []).append(doc_full)

    pdf_vectordb = FAISS.from_documents(pdf_structured_docs, embedding_model)
    # 캐시 저장 (실패해도 계속 진행)
    try:
        cache_manager.save_vector_cache("pdf", [pdf_path], pdf_vectordb)
        cache_manager.save_pdf_docs_cache("pdf", pdf_structured_docs)
        print("✅ PDF 벡터 DB 및 문서 캐시 저장 완료")
    except Exception as e:
        print(f"⚠️ PDF 캐시 저장 실패, 계속 진행: {e}")

pdf_retriever = ContextualCompressionRetriever(
    base_retriever=pdf_vectordb.as_retriever(search_type="similarity", k=20),
    base_compressor=compressor
)

# === Excel 인덱싱 및 검색기 ===
# 기존 Excel 파일들
excel_files = [rf"C:\Users\jung\Desktop\11\e약은요정보검색{i}.xlsx" for i in range(1, 6)]

# ============================================
# 새 Excel 파일 추가하기
# ============================================
# 새 Excel 파일 경로 추가
new_excel_file = r"C:\Users\jung\Desktop\33\OpenData_ItemPermitDetail20251115.xls"
excel_files.append(new_excel_file)

# 파일별 컬럼명 매핑 (파일 경로를 키로 사용)
file_column_mappings = {}  # {파일경로: 컬럼매핑}

# 기본 컬럼명 (기존 파일용)
default_columns = {
    "제품명": "제품명",
    "효능": "이 약의 효능은 무엇입니까?",
    "부작용": "이 약은 어떤 이상반응이 나타날 수 있습니까?",
    "사용법": "이 약은 어떻게 사용합니까?",
    "주성분": "주성분"
}

# 새 Excel 파일의 컬럼명 매핑 (효능효과, 용법용량, 주의사항만 추출)
file_column_mappings[new_excel_file] = {
    "제품명": "품목명",      # 제품명 컬럼 (필수)
    "효능": "효능효과",      # 효능효과 컬럼
    "부작용": "주의사항",    # 주의사항 컬럼
    "사용법": "용법용량",    # 용법용량 컬럼
    "주성분": ""             # 주성분은 사용하지 않음 (빈 문자열)
}

# 전역 변수 초기화
excel_docs = []
excel_product_index = {}
product_names = []
product_names_normalized = []

# 캐시 확인
if cache_manager.is_vector_cache_valid("excel", excel_files) and cache_manager.is_docs_cache_valid("excel"):
    print("📂 Excel 벡터 DB 및 문서 캐시 사용")
    excel_vectordb = cache_manager.load_vector_cache("excel", embedding_model)
    excel_docs = cache_manager.load_excel_docs_cache("excel")
    
    if excel_vectordb is None or excel_docs is None:
        print("⚠️ Excel 캐시 로드 실패, 새로 생성합니다")
        excel_vectordb = None
        excel_docs = []
        excel_product_index = {}
        product_names = []
        product_names_normalized = []
    else:
        print("📂 Excel 벡터 DB 및 문서 캐시 로드됨")
        # product_names와 product_names_normalized도 복원
        product_names = [doc.metadata.get("제품명", "") for doc in excel_docs if doc.metadata.get("제품명")]
        product_names = list(set(product_names))  # 중복 제거
        product_names_normalized = [re.sub(r"[^\w가-힣]", "", name.lower()) for name in product_names]
        
        # excel_product_index도 복원
        excel_product_index = {}
        for doc in excel_docs:
            name = doc.metadata.get("제품명", "")
            if name:
                excel_product_index.setdefault(name, []).append(doc)
else:
    print("🔄 Excel 벡터 DB 새로 생성")
    excel_vectordb = None

if excel_vectordb is None:
    excel_docs = []
    excel_product_index = {}
    product_names = []
    product_names_normalized = []

    for file in excel_files:
        if not os.path.exists(file): 
            print(f"❌ 파일이 존재하지 않음: {file}")
            continue
        
        df = pd.read_excel(file)
        
        # 파일별 컬럼 매핑 확인
        if file in file_column_mappings:
            col_mapping = file_column_mappings[file]
        else:
            # 기본 매핑 사용 (기존 파일)
            col_mapping = default_columns
        
        # 실제 컬럼명 확인 (빈 문자열 제외)
        required_cols = [col for col in col_mapping.values() if col]  # 빈 문자열 제외
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️ 파일 '{os.path.basename(file)}'에서 컬럼 누락: {missing_cols}")
            print(f"   사용 가능한 컬럼: {list(df.columns)[:10]}...")  # 처음 10개만 표시
            # 누락된 컬럼이 있으면 건너뛰기
            continue
        
        # 매핑된 컬럼으로 데이터 추출
        # 주성분이 빈 문자열인 경우를 위해 실제 사용할 컬럼만 선택
        actual_cols = []
        actual_keys = []
        for key, col_name in col_mapping.items():
            if col_name:  # 빈 문자열이 아닌 경우만
                actual_cols.append(col_name)
                actual_keys.append(key)
        
        df_selected = df[actual_cols].fillna("정보 없음")
        df_selected.columns = actual_keys  # 표준 컬럼명으로 변경
        
        # 주성분이 없는 경우를 위해 기본값 추가
        if '주성분' not in df_selected.columns:
            df_selected['주성분'] = '정보 없음'
        
        for row_idx, row in df_selected.iterrows():
            name = row["제품명"].strip()
            if not name or name == "정보 없음":
                continue
                
            product_names.append(name)
            product_names_normalized.append(re.sub(r"[^\w가-힣]", "", name.lower()))

            # 스마트 청크 분할: 사용법을 별도 청크로 분리하여 보존
            efficacy = row['효능']
            side_effects = row['부작용']
            usage = row['사용법']
            main_ingredient = row.get('주성분', '정보 없음')
            
            # PDF 링크는 나중에 필요할 때 다운로드하도록 URL만 저장
            # Excel 로드 시에는 PDF 다운로드하지 않음 (성능 향상)
            
            # 메인 내용 (효능 + 부작용)
            content_main = (
                f"[제품명]: {name}\n"
                f"[주성분]: {main_ingredient}\n"
                f"[효능]: {efficacy}\n"
                f"[부작용]: {side_effects}"
            )
            
            # 사용법 내용 (별도 청크)
            content_usage = (
                f"[제품명]: {name}\n"
                f"[주성분]: {main_ingredient}\n"
                f"[사용법]: {usage}"
            )
            
            # 메인 청크 분할
            main_chunks = splitter.split_text(content_main)
            for chunk in main_chunks:
                doc_obj = Document(page_content=chunk, metadata={
                    "제품명": name, 
                    "주성분": main_ingredient,
                    "type": "main",
                    "excel_file": file,  # 원본 Excel 파일 경로
                    "excel_row_index": row_idx  # Excel 행 인덱스
                })
                excel_docs.append(doc_obj)
            
            # 사용법 청크 분할 (더 큰 청크 크기 사용)
            usage_chunks = splitter.split_text(content_usage)
            for chunk in usage_chunks:
                doc_obj = Document(page_content=chunk, metadata={
                    "제품명": name, 
                    "주성분": main_ingredient,
                    "type": "usage",
                    "excel_file": file,  # 원본 Excel 파일 경로
                    "excel_row_index": row_idx  # Excel 행 인덱스
                })
                excel_docs.append(doc_obj)

            # 전체 내용도 보존 (검색용)
            doc_full = Document(page_content=f"{content_main}\n{content_usage}", metadata={"제품명": name})
            excel_product_index.setdefault(name, []).append(doc_full)

    # 배치별 임베딩 처리 (토큰 제한 방지)
    print(f"🔄 Excel 데이터 배치별 임베딩 처리: 총 {len(excel_docs)}개 문서")
    batch_size = 50
    excel_vectordb = None
    
    for i in range(0, len(excel_docs), batch_size):
        batch = excel_docs[i:i+batch_size]
        
        try:
            if excel_vectordb is None:
                # 첫 번째 배치로 벡터 DB 초기화
                excel_vectordb = FAISS.from_documents(batch, embedding_model)
            else:
                # 기존 벡터 DB에 배치 추가
                excel_vectordb.add_documents(batch)
        except Exception as e:
            print(f"⚠️ 배치 {i//batch_size + 1} 처리 실패: {e}")
            continue
    
    if excel_vectordb is None:
        print("❌ 모든 배치 처리 실패")
        excel_vectordb = FAISS.from_documents([], embedding_model)  # 빈 벡터 DB 생성
    
    # 캐시 저장 (실패해도 계속 진행)
    try:
        cache_manager.save_vector_cache("excel", excel_files, excel_vectordb)
        cache_manager.save_excel_docs_cache("excel", excel_docs)
        print("✅ Excel 벡터 DB 및 문서 캐시 저장 완료")
    except Exception as e:
        print(f"⚠️ Excel 캐시 저장 실패, 계속 진행: {e}")

excel_retriever = ContextualCompressionRetriever(
    base_retriever=excel_vectordb.as_retriever(search_type="similarity", k=20),
    base_compressor=compressor
)

# === 외부 검색기 ===
search_agent = initialize_agent(
    tools=[TavilySearchResults()],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=False
)

# === LLM 요약기 ===
def extract_active_ingredients_from_medicine(medicine_name: str) -> List[str]:
    """약품명으로부터 주성분 추출"""
    ingredients = []
    
    try:
        # Excel DB에서 해당 약품의 주성분 찾기
        for doc in excel_docs:
            if doc.metadata.get("제품명") == medicine_name:
                # 주성분 정보가 메타데이터에 있는지 확인
                if "주성분" in doc.metadata:
                    ingredient = doc.metadata["주성분"]
                    if ingredient and ingredient != "정보 없음":
                        # 쉼표로 구분된 성분들을 분리
                        if ',' in ingredient:
                            ingredients = [ing.strip() for ing in ingredient.split(',') if ing.strip()]
                        else:
                            ingredients = [ingredient.strip()]
                        break
        
        # 주성분이 없으면 문서 내용에서 추출 시도
        if not ingredients:
            for doc in excel_docs:
                if doc.metadata.get("제품명") == medicine_name:
                    content = doc.page_content
                    # 주성분 관련 패턴 찾기
                    import re
                    patterns = [
                        r'주성분[:\s]*([^,\n]+)',
                        r'성분[:\s]*([^,\n]+)',
                        r'주요성분[:\s]*([^,\n]+)'
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, content)
                        if matches:
                            # 쉼표로 구분된 성분들을 분리
                            for match in matches:
                                if ',' in match:
                                    ingredients.extend([ing.strip() for ing in match.split(',') if ing.strip()])
                                else:
                                    ingredients.append(match.strip())
                            break
        
        print(f"🔍 {medicine_name} 주성분 추출: {ingredients}")
        return ingredients
        
    except Exception as e:
        print(f"❌ 주성분 추출 오류: {e}")
        return []

def summarize_structured_json(text: str) -> dict:
    prompt = f"""
    다음 약품 관련 텍스트에서 항목별 정보를 JSON 형식으로 정리해줘.
    항목은 '제품명', '효능', '부작용', '사용법'이며, 없으면 "정보 없음"으로 표기해줘.

    텍스트:
    {text}

    결과 형식:
    {{
      "제품명": "...",
      "효능": "...",
      "부작용": "...",
      "사용법": "..."
    }}
    """
    try:
        response = llm.invoke(prompt)
        return json.loads(response.content.strip())
    except:
        return {
            "제품명": "",
            "효능": "정보 없음",
            "부작용": "정보 없음",
            "사용법": "정보 없음"
        }

# === 성분 색인 구축 (동적) ===
def build_ingredient_index():
    """Excel DB에서 모든 성분명을 동적으로 추출하고 성분→제품 매핑 생성"""
    all_ingredients = set()
    ingredient_to_products = {}
    
    print("📊 성분 색인 구축 중...")
    
    for doc in excel_docs:
        product_name = doc.metadata.get("제품명", "")
        ingredients_str = doc.metadata.get("주성분", "")
        
        if ingredients_str and ingredients_str != "정보 없음":
            # 쉼표로 구분된 성분들 분리
            ingredients = [ing.strip() for ing in ingredients_str.split(',') if ing.strip()]
            
            for ingredient in ingredients:
                all_ingredients.add(ingredient)
                
                # 성분 → 제품 매핑
                if ingredient not in ingredient_to_products:
                    ingredient_to_products[ingredient] = []
                if product_name and product_name not in ingredient_to_products[ingredient]:
                    ingredient_to_products[ingredient].append(product_name)
    
    print(f"✅ 추출된 성분 총 {len(all_ingredients)}개")
    print(f"✅ 성분→제품 매핑 {len(ingredient_to_products)}개 생성")
    
    return all_ingredients, ingredient_to_products

# 전역 변수로 저장 (시작 시 한 번만 실행)
known_ingredients, ingredient_to_products_map = build_ingredient_index()

def find_products_by_ingredient(ingredient_name: str) -> List[str]:
    """특정 성분이 포함된 제품 목록 반환"""
    return ingredient_to_products_map.get(ingredient_name, [])

# === 용량주의 성분 데이터 처리 ===
dosage_warning_ingredients = {}  # 성분명 -> 용량 정보 매핑
dosage_warning_loaded = False

def load_dosage_warning_data():
    """용량주의 성분 리스트 로드 (새 파일 형식: OpenData_PotOpenDurIngr_D20251115.xls)"""
    global dosage_warning_ingredients, dosage_warning_loaded
    
    print(f"🔍 용량주의 성분 리스트 로드 시도 - 현재 상태: loaded={dosage_warning_loaded}")
    
    if dosage_warning_loaded:
        print(f"📂 이미 로드됨 - 총 {len(dosage_warning_ingredients)}개 성분")
        return dosage_warning_ingredients
    
    try:
        # 새 용량주의 성분 리스트 파일 경로
        dosage_file_path = r"C:\Users\jung\Desktop\22\OpenData_PotOpenDurIngr_D20251115.xls"
        
        print(f"🔍 파일 존재 확인: {dosage_file_path}")
        print(f"🔍 파일 존재 여부: {os.path.exists(dosage_file_path)}")
        
        if not os.path.exists(dosage_file_path):
            print(f"⚠️ 용량주의 성분 리스트 파일을 찾을 수 없습니다: {dosage_file_path}")
            dosage_warning_loaded = True
            return dosage_warning_ingredients
        
        print("📊 용량주의 성분 리스트 로드 중...")
        df = pd.read_excel(dosage_file_path)
        print(f"📊 엑셀 파일 로드 완료 - 행 수: {len(df)}, 컬럼: {list(df.columns)}")
        
        # 사용할 컬럼 확인
        required_columns = ['단일복합구분코드', 'DUR성분명', '복합제', '1일최대용량']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ 필수 컬럼이 없습니다: {missing_columns}")
            print(f"   사용 가능한 컬럼: {list(df.columns)}")
            dosage_warning_loaded = True
            return dosage_warning_ingredients
        
        # 데이터 처리
        processed_count = 0
        print(f"🔍 데이터 처리 시작 - 총 {len(df)}행")
        
        for idx, row in df.iterrows():
            # 사용할 컬럼 추출
            single_complex = str(row.get('단일복합구분코드', '')).strip()
            ingredient_name = str(row.get('DUR성분명', '')).strip()
            complex_medicine = str(row.get('복합제', '')).strip()
            max_dose = str(row.get('1일최대용량', '')).strip()
            
            # NaN 값 처리
            if pd.isna(row.get('DUR성분명')) or ingredient_name == 'nan' or not ingredient_name:
                continue
            
            if idx < 5:  # 처음 5개 행만 로그 출력
                print(f"🔍 행 {idx}: 성분명='{ingredient_name}', 단일/복합='{single_complex}', 복합제='{complex_medicine}', 용량='{max_dose}'")
            
            # 데이터 구조 구성 (기존 구조와 호환성 유지)
            ingredient_data = {
                'korean_name': ingredient_name,
                'english_name': '',  # 새 파일에는 영문명이 별도 컬럼으로 없음
                'formulation': '',  # 제형 정보는 별도 컬럼으로 없음
                'max_daily_dose': max_dose if max_dose != 'nan' else '',
                'remarks': f"단일/복합: {single_complex}" + (f", 복합제: {complex_medicine}" if complex_medicine != 'nan' and complex_medicine else ""),
                'single_complex': single_complex,  # 새 필드 추가
                'complex_medicine': complex_medicine if complex_medicine != 'nan' else ''  # 새 필드 추가
            }
            
            # 한국어 성분명으로 매핑
            dosage_warning_ingredients[ingredient_name] = ingredient_data
            
            # 복합제인 경우 복합제 성분명도 매핑 (관계성분 정보 활용)
            if single_complex == '복합' and complex_medicine and complex_medicine != 'nan':
                # 복합제 정보에서 성분명 추출 시도 (예: "[D001312]Naltrexone(날트렉손)" 형식)
                # 괄호 안의 한글명 추출
                korean_match = re.search(r'\(([가-힣]+)\)', complex_medicine)
                if korean_match:
                    complex_ingredient_name = korean_match.group(1)
                    # 복합제 성분도 별도로 매핑 (용량 정보는 주성분과 동일)
                    if complex_ingredient_name not in dosage_warning_ingredients:
                        dosage_warning_ingredients[complex_ingredient_name] = {
                            **ingredient_data,
                            'korean_name': complex_ingredient_name,
                            'remarks': f"복합제 구성 성분 (주성분: {ingredient_name})"
                        }
            
            processed_count += 1
        
        print(f"✅ 용량주의 성분 {len(dosage_warning_ingredients)}개 로드 완료 (처리된 행: {processed_count}개)")
        print(f"🔍 로드된 성분 예시: {list(dosage_warning_ingredients.keys())[:5]}")
        dosage_warning_loaded = True
        
    except Exception as e:
        print(f"❌ 용량주의 성분 리스트 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        dosage_warning_loaded = True
    
    return dosage_warning_ingredients

def find_dosage_warning_info(ingredient_name: str) -> dict:
    """특정 성분의 용량주의 정보 찾기"""
    if not dosage_warning_loaded:
        load_dosage_warning_data()
    
    # 정확한 매칭 시도
    if ingredient_name in dosage_warning_ingredients:
        return dosage_warning_ingredients[ingredient_name]
    
    # 부분 매칭 시도 (성분명이 포함된 경우)
    for key, value in dosage_warning_ingredients.items():
        if ingredient_name in key or key in ingredient_name:
            return value
    
    # 정규화된 매칭 시도
    normalized_ingredient = re.sub(r'[^\w가-힣]', '', ingredient_name.lower())
    for key, value in dosage_warning_ingredients.items():
        normalized_key = re.sub(r'[^\w가-힣]', '', key.lower())
        if normalized_ingredient in normalized_key or normalized_key in normalized_ingredient:
            return value
    
    return None

def get_medicine_dosage_warnings(medicine_name: str) -> List[dict]:
    """약품의 주성분들 중 용량주의 성분이 있는지 확인"""
    print(f"🔍 용량주의 성분 확인 시작: '{medicine_name}'")
    
    warnings = []
    
    # 약품의 주성분 추출
    ingredients = extract_active_ingredients_from_medicine(medicine_name)
    print(f"🔍 추출된 주성분: {ingredients}")
    
    for ingredient in ingredients:
        print(f"🔍 성분 '{ingredient}' 용량주의 확인 중...")
        dosage_info = find_dosage_warning_info(ingredient)
        if dosage_info:
            print(f"✅ 용량주의 성분 발견: '{ingredient}' - {dosage_info['max_daily_dose']}")
            warnings.append({
                'ingredient': ingredient,
                'dosage_info': dosage_info
            })
        else:
            print(f"❌ 용량주의 성분 아님: '{ingredient}'")
    
    print(f"🔍 최종 용량주의 성분 개수: {len(warnings)}")
    return warnings

# === 연령대 금기 성분 데이터 처리 ===
age_contraindication_ingredients = {}  # 성분명 -> 연령대별 금기 정보 매핑
age_contraindication_loaded = False

def load_age_contraindication_data():
    """연령대 금기 성분 리스트 로드 (OpenData_PotOpenDurIngr_B20251117.xls)"""
    global age_contraindication_ingredients, age_contraindication_loaded
    
    print(f"🔍 연령대 금기 성분 리스트 로드 시도 - 현재 상태: loaded={age_contraindication_loaded}")
    
    if age_contraindication_loaded:
        print(f"📂 이미 로드됨 - 총 {len(age_contraindication_ingredients)}개 성분")
        return age_contraindication_ingredients
    
    try:
        # 연령대 금기 성분 파일 경로
        age_contraindication_file = r"C:\Users\jung\Desktop\44\OpenData_PotOpenDurIngr_B20251117.xls"
        
        print(f"🔍 파일 존재 확인: {age_contraindication_file}")
        
        if not os.path.exists(age_contraindication_file):
            print(f"⚠️ 연령대 금기 성분 파일을 찾을 수 없습니다: {age_contraindication_file}")
            age_contraindication_loaded = True
            return age_contraindication_ingredients
        
        print("📊 연령대 금기 성분 리스트 로드 중...")
        df = pd.read_excel(age_contraindication_file)
        print(f"📊 엑셀 파일 로드 완료 - 행 수: {len(df)}, 컬럼: {list(df.columns)}")
        
        # 사용할 컬럼 확인
        required_columns = ['DUR성분명', '연령기준', '금기내용']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ 필수 컬럼이 없습니다: {missing_columns}")
            print(f"   사용 가능한 컬럼: {list(df.columns)}")
            age_contraindication_loaded = True
            return age_contraindication_ingredients
        
        # 데이터 처리
        processed_count = 0
        print(f"🔍 데이터 처리 시작 - 총 {len(df)}행")
        
        for idx, row in df.iterrows():
            ingredient_name = str(row.get('DUR성분명', '')).strip()
            age_criteria = str(row.get('연령기준', '')).strip()
            contraindication = str(row.get('금기내용', '')).strip()
            
            # NaN 값 처리
            if pd.isna(row.get('DUR성분명')) or ingredient_name == 'nan' or not ingredient_name:
                continue
            
            if idx < 5:  # 처음 5개 행만 로그 출력
                print(f"🔍 행 {idx}: 성분명='{ingredient_name}', 연령기준='{age_criteria}', 금기내용='{contraindication[:50] if contraindication else '없음'}...'")
            
            # 성분명이 이미 있으면 리스트에 추가, 없으면 새로 생성
            if ingredient_name not in age_contraindication_ingredients:
                age_contraindication_ingredients[ingredient_name] = {
                    'korean_name': ingredient_name,
                    'age_contraindications': []  # 여러 연령대별 금기 정보를 리스트로 저장
                }
            
            # 연령대별 금기 정보 추가
            if age_criteria and age_criteria != 'nan' and contraindication and contraindication != 'nan':
                age_contraindication_ingredients[ingredient_name]['age_contraindications'].append({
                    'age_criteria': age_criteria,
                    'contraindication': contraindication
                })
            
            processed_count += 1
        
        print(f"✅ 연령대 금기 성분 {len(age_contraindication_ingredients)}개 로드 완료 (처리된 행: {processed_count}개)")
        print(f"🔍 로드된 성분 예시: {list(age_contraindication_ingredients.keys())[:5]}")
        age_contraindication_loaded = True
        
    except Exception as e:
        print(f"❌ 연령대 금기 성분 리스트 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        age_contraindication_loaded = True
    
    return age_contraindication_ingredients

def find_age_contraindication_info(ingredient_name: str) -> dict:
    """특정 성분의 연령대별 금기 정보 찾기"""
    if not age_contraindication_loaded:
        load_age_contraindication_data()
    
    # 정확한 매칭 시도
    if ingredient_name in age_contraindication_ingredients:
        return age_contraindication_ingredients[ingredient_name]
    
    # 부분 매칭 시도
    for key, value in age_contraindication_ingredients.items():
        if ingredient_name in key or key in ingredient_name:
            return value
    
    # 정규화된 매칭 시도
    normalized_ingredient = re.sub(r'[^\w가-힣]', '', ingredient_name.lower())
    for key, value in age_contraindication_ingredients.items():
        normalized_key = re.sub(r'[^\w가-힣]', '', key.lower())
        if normalized_ingredient in normalized_key or normalized_key in normalized_ingredient:
            return value
    
    return None

def get_medicine_age_contraindications(medicine_name: str) -> List[dict]:
    """약품의 주성분들 중 연령대 금기 성분이 있는지 확인"""
    print(f"🔍 연령대 금기 성분 확인 시작: '{medicine_name}'")
    
    contraindications = []
    
    # 약품의 주성분 추출
    ingredients = extract_active_ingredients_from_medicine(medicine_name)
    print(f"🔍 추출된 주성분: {ingredients}")
    
    for ingredient in ingredients:
        print(f"🔍 성분 '{ingredient}' 연령대 금기 확인 중...")
        age_info = find_age_contraindication_info(ingredient)
        if age_info and age_info.get('age_contraindications'):
            print(f"✅ 연령대 금기 성분 발견: '{ingredient}' - {len(age_info['age_contraindications'])}개 금기 정보")
            contraindications.append({
                'ingredient': ingredient,
                'age_contraindication_info': age_info
            })
        else:
            print(f"❌ 연령대 금기 성분 아님: '{ingredient}'")
    
    print(f"🔍 최종 연령대 금기 성분 개수: {len(contraindications)}")
    return contraindications

# === 일일 최대 투여량 데이터 처리 ===
daily_max_dosage_ingredients = {}  # 성분명 -> 일일 최대 투여량 정보 매핑
daily_max_dosage_loaded = False

def load_daily_max_dosage_data():
    """일일 최대 투여량 정보 로드 (OpenData_DayMaxDosgQyInfo20251116.xls)"""
    global daily_max_dosage_ingredients, daily_max_dosage_loaded
    
    print(f"🔍 일일 최대 투여량 정보 로드 시도 - 현재 상태: loaded={daily_max_dosage_loaded}")
    
    if daily_max_dosage_loaded:
        print(f"📂 이미 로드됨 - 총 {len(daily_max_dosage_ingredients)}개 성분")
        return daily_max_dosage_ingredients
    
    try:
        # 일일 최대 투여량 파일 경로
        daily_max_dosage_file = r"C:\Users\jung\Desktop\55\OpenData_DayMaxDosgQyInfo20251116.xls"
        
        print(f"🔍 파일 존재 확인: {daily_max_dosage_file}")
        
        if not os.path.exists(daily_max_dosage_file):
            print(f"⚠️ 일일 최대 투여량 파일을 찾을 수 없습니다: {daily_max_dosage_file}")
            daily_max_dosage_loaded = True
            return daily_max_dosage_ingredients
        
        print("📊 일일 최대 투여량 정보 로드 중...")
        df = pd.read_excel(daily_max_dosage_file)
        print(f"📊 엑셀 파일 로드 완료 - 행 수: {len(df)}, 컬럼: {list(df.columns)}")
        
        # 사용할 컬럼 확인
        required_columns = ['성분명(한글)', '제형명', '투여단위', '1일최대투여량']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ 필수 컬럼이 없습니다: {missing_columns}")
            print(f"   사용 가능한 컬럼: {list(df.columns)}")
            daily_max_dosage_loaded = True
            return daily_max_dosage_ingredients
        
        # 데이터 처리
        processed_count = 0
        print(f"🔍 데이터 처리 시작 - 총 {len(df)}행")
        
        for idx, row in df.iterrows():
            ingredient_name = str(row.get('성분명(한글)', '')).strip()
            formulation = str(row.get('제형명', '')).strip()
            dosage_unit = str(row.get('투여단위', '')).strip()
            max_daily_dosage = str(row.get('1일최대투여량', '')).strip()
            
            # NaN 값 처리
            if pd.isna(row.get('성분명(한글)')) or ingredient_name == 'nan' or not ingredient_name:
                continue
            
            if idx < 5:  # 처음 5개 행만 로그 출력
                print(f"🔍 행 {idx}: 성분명='{ingredient_name}', 제형='{formulation}', 단위='{dosage_unit}', 최대투여량='{max_daily_dosage}'")
            
            # 성분명이 이미 있으면 리스트에 추가, 없으면 새로 생성
            if ingredient_name not in daily_max_dosage_ingredients:
                daily_max_dosage_ingredients[ingredient_name] = {
                    'korean_name': ingredient_name,
                    'dosage_info': []  # 여러 제형별 투여량 정보를 리스트로 저장
                }
            
            # 제형별 투여량 정보 추가
            if max_daily_dosage and max_daily_dosage != 'nan':
                daily_max_dosage_ingredients[ingredient_name]['dosage_info'].append({
                    'formulation': formulation if formulation != 'nan' else '',
                    'dosage_unit': dosage_unit if dosage_unit != 'nan' else '',
                    'max_daily_dosage': max_daily_dosage
                })
            
            processed_count += 1
        
        print(f"✅ 일일 최대 투여량 정보 {len(daily_max_dosage_ingredients)}개 성분 로드 완료 (처리된 행: {processed_count}개)")
        print(f"🔍 로드된 성분 예시: {list(daily_max_dosage_ingredients.keys())[:5]}")
        daily_max_dosage_loaded = True
        
    except Exception as e:
        print(f"❌ 일일 최대 투여량 정보 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        daily_max_dosage_loaded = True
    
    return daily_max_dosage_ingredients

def find_daily_max_dosage_info(ingredient_name: str, formulation: str = None) -> dict:
    """특정 성분의 일일 최대 투여량 정보 찾기"""
    if not daily_max_dosage_loaded:
        load_daily_max_dosage_data()
    
    # 정확한 매칭 시도
    if ingredient_name in daily_max_dosage_ingredients:
        ingredient_info = daily_max_dosage_ingredients[ingredient_name]
        
        # 제형이 지정된 경우 해당 제형 정보만 반환
        if formulation:
            for dosage_info in ingredient_info.get('dosage_info', []):
                if formulation in dosage_info.get('formulation', '') or dosage_info.get('formulation', '') in formulation:
                    return {
                        'ingredient': ingredient_name,
                        'formulation': dosage_info.get('formulation', ''),
                        'dosage_unit': dosage_info.get('dosage_unit', ''),
                        'max_daily_dosage': dosage_info.get('max_daily_dosage', '')
                    }
        
        # 제형이 지정되지 않았거나 매칭되지 않은 경우 첫 번째 정보 반환
        if ingredient_info.get('dosage_info'):
            first_info = ingredient_info['dosage_info'][0]
            return {
                'ingredient': ingredient_name,
                'formulation': first_info.get('formulation', ''),
                'dosage_unit': first_info.get('dosage_unit', ''),
                'max_daily_dosage': first_info.get('max_daily_dosage', ''),
                'all_formulations': ingredient_info.get('dosage_info', [])  # 모든 제형 정보도 포함
            }
        
        return ingredient_info
    
    # 부분 매칭 시도
    for key, value in daily_max_dosage_ingredients.items():
        if ingredient_name in key or key in ingredient_name:
            if value.get('dosage_info'):
                first_info = value['dosage_info'][0]
                return {
                    'ingredient': key,
                    'formulation': first_info.get('formulation', ''),
                    'dosage_unit': first_info.get('dosage_unit', ''),
                    'max_daily_dosage': first_info.get('max_daily_dosage', ''),
                    'all_formulations': value.get('dosage_info', [])
                }
    
    # 정규화된 매칭 시도
    normalized_ingredient = re.sub(r'[^\w가-힣]', '', ingredient_name.lower())
    for key, value in daily_max_dosage_ingredients.items():
        normalized_key = re.sub(r'[^\w가-힣]', '', key.lower())
        if normalized_ingredient in normalized_key or normalized_key in normalized_ingredient:
            if value.get('dosage_info'):
                first_info = value['dosage_info'][0]
                return {
                    'ingredient': key,
                    'formulation': first_info.get('formulation', ''),
                    'dosage_unit': first_info.get('dosage_unit', ''),
                    'max_daily_dosage': first_info.get('max_daily_dosage', ''),
                    'all_formulations': value.get('dosage_info', [])
                }
    
    return None

def get_medicine_daily_max_dosage(medicine_name: str) -> List[dict]:
    """약품의 주성분들 중 일일 최대 투여량 정보가 있는지 확인"""
    print(f"🔍 일일 최대 투여량 정보 확인 시작: '{medicine_name}'")
    
    dosage_infos = []
    
    # 약품의 주성분 추출
    ingredients = extract_active_ingredients_from_medicine(medicine_name)
    print(f"🔍 추출된 주성분: {ingredients}")
    
    for ingredient in ingredients:
        print(f"🔍 성분 '{ingredient}' 일일 최대 투여량 확인 중...")
        dosage_info = find_daily_max_dosage_info(ingredient)
        if dosage_info:
            print(f"✅ 일일 최대 투여량 정보 발견: '{ingredient}' - {dosage_info.get('max_daily_dosage', '정보 없음')}")
            dosage_infos.append(dosage_info)
        else:
            print(f"❌ 일일 최대 투여량 정보 없음: '{ingredient}'")
    
    print(f"🔍 최종 일일 최대 투여량 정보 개수: {len(dosage_infos)}")
    return dosage_infos

# === Export 대상 ===
__all__ = [
    "pdf_retriever",
    "excel_retriever",
    "product_names",
    "product_names_normalized",
    "search_agent",
    "summarize_structured_json",
    "extract_active_ingredients_from_medicine",
    "pdf_product_index",
    "excel_product_index",
    "pdf_structured_docs",
    "excel_docs",
    "known_ingredients",
    "ingredient_to_products_map",
    "find_products_by_ingredient",
    "load_dosage_warning_data",
    "find_dosage_warning_info",
    "get_medicine_dosage_warnings",
    "load_age_contraindication_data",
    "find_age_contraindication_info",
    "get_medicine_age_contraindications",
    "load_daily_max_dosage_data",
    "find_daily_max_dosage_info",
    "get_medicine_daily_max_dosage"
]
