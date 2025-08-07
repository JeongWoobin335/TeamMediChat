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
if cache_manager.is_vector_cache_valid("pdf", [pdf_path]):
    print("📂 PDF 벡터 DB 캐시 사용")
    pdf_vectordb = cache_manager.load_vector_cache("pdf", embedding_model)
    if pdf_vectordb is None:
        print("⚠️ PDF 캐시 로드 실패, 새로 생성합니다")
        pdf_vectordb = None
    else:
        print("📂 pdf 벡터 DB 캐시 로드됨")
        # 캐시에서 로드된 경우에도 pdf_structured_docs 설정
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
    "이 약의 효능은 무엇입니까?",
    "이 약은 어떤 이상반응이 나타날 수 있습니까?",
    "이 약은 어떻게 사용합니까?"
]

# 전역 변수 초기화
excel_docs = []
excel_product_index = {}
product_names = []
product_names_normalized = []

# 캐시 확인
if cache_manager.is_vector_cache_valid("excel", excel_files):
    print("📂 Excel 벡터 DB 캐시 사용")
    excel_vectordb = cache_manager.load_vector_cache("excel", embedding_model)
    if excel_vectordb is None:
        print("⚠️ Excel 캐시 로드 실패, 새로 생성합니다")
        excel_vectordb = None
    else:
        print("📂 Excel 벡터 DB 캐시 로드됨")
        # 캐시에서 로드된 경우에도 excel_docs 설정
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
                
                # 메인 내용 (효능 + 부작용)
                content_main = (
                    f"[제품명]: {name}\n"
                    f"[효능]: {efficacy}\n"
                    f"[부작용]: {side_effects}"
                )
                
                # 사용법 내용 (별도 청크)
                content_usage = (
                    f"[제품명]: {name}\n"
                    f"[사용법]: {usage}"
                )
                
                # 메인 청크 분할
                main_chunks = splitter.split_text(content_main)
                for chunk in main_chunks:
                    doc_obj = Document(page_content=chunk, metadata={"제품명": name, "type": "main"})
                    excel_docs.append(doc_obj)
                
                # 사용법 청크 분할
                usage_chunks = splitter.split_text(content_usage)
                for chunk in usage_chunks:
                    doc_obj = Document(page_content=chunk, metadata={"제품명": name, "type": "usage"})
                    excel_docs.append(doc_obj)

                # 전체 내용도 보존 (검색용)
                doc_full = Document(page_content=f"{content_main}\n{content_usage}", metadata={"제품명": name})
                excel_product_index.setdefault(name, []).append(doc_full)
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
            
            # 메인 내용 (효능 + 부작용)
            content_main = (
                f"[제품명]: {name}\n"
                f"[효능]: {efficacy}\n"
                f"[부작용]: {side_effects}"
            )
            
            # 사용법 내용 (별도 청크)
            content_usage = (
                f"[제품명]: {name}\n"
                f"[사용법]: {usage}"
            )
            
            # 메인 청크 분할
            main_chunks = splitter.split_text(content_main)
            for chunk in main_chunks:
                doc_obj = Document(page_content=chunk, metadata={"제품명": name, "type": "main"})
                excel_docs.append(doc_obj)
            
            # 사용법 청크 분할 (더 큰 청크 크기 사용)
            usage_chunks = splitter.split_text(content_usage)
            for chunk in usage_chunks:
                doc_obj = Document(page_content=chunk, metadata={"제품명": name, "type": "usage"})
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
        print("✅ Excel 벡터 DB 캐시 저장 완료")
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

# === Export 대상 ===
__all__ = [
    "pdf_retriever",
    "excel_retriever",
    "product_names",
    "product_names_normalized",
    "search_agent",
    "summarize_structured_json",
    "pdf_product_index",
    "excel_product_index",
    "pdf_structured_docs",
    "excel_docs"
]
