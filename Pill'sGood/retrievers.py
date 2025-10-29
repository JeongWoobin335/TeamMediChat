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
excel_files = [rf"C:\Users\jung\Desktop\11\e약은요정보검색{i}.xlsx" for i in range(1, 6)]
required_columns = [
    "제품명",
    "이 약의 효능은 무엇입니까?",  # 정확한 컬럼명
    "이 약은 어떤 이상반응이 나타날 수 있습니까?",  # 정확한 컬럼명
    "이 약은 어떻게 사용합니까?",
    "주성분"  # 주성분 컬럼 추가
]

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
        if not all(col in df.columns for col in required_columns): 
            print(f"❌ 필수 컬럼 누락: {[col for col in required_columns if col not in df.columns]}")
            continue

        df = df[required_columns].fillna("정보 없음")
        for _, row in df.iterrows():
            name = row["제품명"].strip()
            product_names.append(name)
            product_names_normalized.append(re.sub(r"[^\w가-힣]", "", name.lower()))

            # 스마트 청크 분할: 사용법을 별도 청크로 분리하여 보존
            efficacy = row['이 약의 효능은 무엇입니까?']
            side_effects = row['이 약은 어떤 이상반응이 나타날 수 있습니까?']
            usage = row['이 약은 어떻게 사용합니까?']
            main_ingredient = row.get('주성분', '정보 없음')  # 주성분 추가
            
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
                    "type": "main"
                })
                excel_docs.append(doc_obj)
            
            # 사용법 청크 분할 (더 큰 청크 크기 사용)
            usage_chunks = splitter.split_text(content_usage)
            for chunk in usage_chunks:
                doc_obj = Document(page_content=chunk, metadata={
                    "제품명": name, 
                    "주성분": main_ingredient,
                    "type": "usage"
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
    """용량주의 성분 리스트 로드"""
    global dosage_warning_ingredients, dosage_warning_loaded
    
    print(f"🔍 용량주의 성분 리스트 로드 시도 - 현재 상태: loaded={dosage_warning_loaded}")
    
    if dosage_warning_loaded:
        print(f"📂 이미 로드됨 - 총 {len(dosage_warning_ingredients)}개 성분")
        return dosage_warning_ingredients
    
    try:
        # 용량주의 성분 리스트 파일 경로 (실제 파일 위치에 맞게 수정 필요)
        dosage_file_path = r"C:\Users\jung\Desktop\22\용량주의 성분리스트_250530.xlsx"
        
        print(f"🔍 파일 존재 확인: {dosage_file_path}")
        print(f"🔍 파일 존재 여부: {os.path.exists(dosage_file_path)}")
        
        if not os.path.exists(dosage_file_path):
            print(f"⚠️ 용량주의 성분 리스트 파일을 찾을 수 없습니다: {dosage_file_path}")
            dosage_warning_loaded = True
            return dosage_warning_ingredients
        
        print("📊 용량주의 성분 리스트 로드 중...")
        df = pd.read_excel(dosage_file_path)
        print(f"📊 엑셀 파일 로드 완료 - 행 수: {len(df)}, 컬럼: {list(df.columns)}")
        
        # 실제 데이터가 시작되는 행 찾기 (헤더 행 건너뛰기)
        data_start_row = 0
        for idx, row in df.iterrows():
            # 첫 번째 컬럼에 숫자가 있는 행을 찾기 (연번)
            first_col = str(row.iloc[0]).strip()
            if first_col.isdigit():
                data_start_row = idx
                break
        
        print(f"🔍 데이터 시작 행: {data_start_row}")
        
        # 실제 데이터만 사용 (헤더 행 제외)
        if data_start_row > 0:
            df = df.iloc[data_start_row:].reset_index(drop=True)
            print(f"🔍 헤더 제거 후 행 수: {len(df)}")
        
        # 컬럼명을 수동으로 매핑 (Unnamed 컬럼들)
        # 일반적으로 용량주의 성분 리스트는 다음 순서: 연번, 성분명(국문), 성분명(영문), 제형, 1일 최대용량, 비고
        actual_columns = {
            'korean_name': df.columns[1] if len(df.columns) > 1 else None,  # 두 번째 컬럼
            'english_name': df.columns[2] if len(df.columns) > 2 else None,  # 세 번째 컬럼
            'formulation': df.columns[3] if len(df.columns) > 3 else None,    # 네 번째 컬럼
            'max_daily_dose': df.columns[4] if len(df.columns) > 4 else None, # 다섯 번째 컬럼
            'remarks': df.columns[5] if len(df.columns) > 5 else None         # 여섯 번째 컬럼
        }
        
        print(f"🔍 수동 컬럼 매핑 결과: {actual_columns}")
        
        # None 값 제거
        actual_columns = {k: v for k, v in actual_columns.items() if v is not None}
        
        if not actual_columns:
            print("❌ 용량주의 성분 리스트 컬럼을 찾을 수 없습니다")
            dosage_warning_loaded = True
            return dosage_warning_ingredients
        
        # 데이터 처리
        processed_count = 0
        print(f"🔍 데이터 처리 시작 - 총 {len(df)}행")
        
        for idx, row in df.iterrows():
            korean_name = str(row.get(actual_columns.get('korean_name', ''), '')).strip()
            english_name = str(row.get(actual_columns.get('english_name', ''), '')).strip()
            formulation = str(row.get(actual_columns.get('formulation', ''), '')).strip()
            max_dose = str(row.get(actual_columns.get('max_daily_dose', ''), '')).strip()
            remarks = str(row.get(actual_columns.get('remarks', ''), '')).strip()
            
            if idx < 5:  # 처음 5개 행만 로그 출력
                print(f"🔍 행 {idx}: 한글='{korean_name}', 영문='{english_name}', 용량='{max_dose}'")
            
            if not korean_name or korean_name == 'nan':
                continue
            
            # 한국어 성분명으로 매핑
            dosage_warning_ingredients[korean_name] = {
                'korean_name': korean_name,
                'english_name': english_name,
                'formulation': formulation,
                'max_daily_dose': max_dose,
                'remarks': remarks
            }
            
            # 영어 성분명으로도 매핑 (있는 경우)
            if english_name and english_name != 'nan':
                dosage_warning_ingredients[english_name] = {
                    'korean_name': korean_name,
                    'english_name': english_name,
                    'formulation': formulation,
                    'max_daily_dose': max_dose,
                    'remarks': remarks
                }
            
            processed_count += 1
        
        print(f"✅ 용량주의 성분 {len(dosage_warning_ingredients)}개 로드 완료 (처리된 행: {processed_count}개)")
        print(f"🔍 로드된 성분 예시: {list(dosage_warning_ingredients.keys())[:5]}")
        dosage_warning_loaded = True
        
    except Exception as e:
        print(f"❌ 용량주의 성분 리스트 로드 실패: {e}")
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
    "get_medicine_dosage_warnings"
]
