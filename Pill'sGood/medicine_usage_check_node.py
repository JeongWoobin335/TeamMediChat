# medicine_usage_check_node.py

from qa_state import QAState
from retrievers import llm, pdf_structured_docs, excel_docs, get_medicine_dosage_warnings, load_dosage_warning_data
from langchain_core.documents import Document
from typing import List, Optional
import json
import re
from cache_manager import cache_manager
from difflib import get_close_matches

def normalize_medicine_name(name: str) -> str:
    """
    약품명 정규화 (유사도 매칭을 위해)
    """
    if not name:
        return ""
    
    # 소문자 변환
    normalized = name.lower()
    
    # 특수문자, 공백, 숫자 제거 (한글과 영문만 유지)
    normalized = re.sub(r'[^\w가-힣]', '', normalized)
    
    # 연속된 공백 제거
    normalized = re.sub(r'\s+', '', normalized)
    
    return normalized.strip()

def calculate_similarity(str1: str, str2: str) -> float:
    """
    두 문자열의 유사도 계산 (0.0 ~ 1.0)
    """
    if not str1 or not str2:
        return 0.0
    
    if str1 == str2:
        return 1.0
    
    # 길이가 너무 다르면 유사도 낮음
    len_diff = abs(len(str1) - len(str2))
    if len_diff > max(len(str1), len(str2)) * 0.5:
        return 0.0
    
    # Levenshtein distance 기반 유사도 계산
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    distance = levenshtein_distance(str1, str2)
    max_len = max(len(str1), len(str2))
    similarity = 1.0 - (distance / max_len)
    
    return similarity

def find_similar_medicine_name(ocr_result: str, medicine_list: List[str], cutoff: float = 0.8) -> Optional[str]:
    """
    OCR 결과와 유사한 약품명 찾기
    """
    if not ocr_result or not medicine_list:
        return None
    
    # OCR 결과 정규화
    normalized_ocr = normalize_medicine_name(ocr_result)
    print(f"🔍 정규화된 OCR 결과: '{normalized_ocr}'")
    
    # 약품명 리스트도 정규화
    normalized_medicines = [(normalize_medicine_name(med), med) for med in medicine_list]
    
    
    # 직접 유사도 계산
    best_match = None
    best_similarity = 0.0
    
    for norm, orig in normalized_medicines:
        similarity = calculate_similarity(normalized_ocr, norm)
        
        # 유사도가 높은 경우만 로그 출력 (성능 개선)
        if similarity > 0.3:
            print(f"🔍 '{orig}' 유사도: {similarity:.3f}")
        
        if similarity > best_similarity and similarity >= cutoff:
            best_similarity = similarity
            best_match = orig
    
    if best_match:
        print(f"✅ 유사도 매칭 성공: '{ocr_result}' → '{best_match}' (유사도: {best_similarity:.3f})")
        return best_match
    
    # cutoff를 낮춰서 다시 시도
    if cutoff > 0.5:
        print(f"🔍 cutoff를 낮춰서 재시도 (0.5)")
        for norm, orig in normalized_medicines:
            similarity = calculate_similarity(normalized_ocr, norm)
            if similarity > best_similarity and similarity >= 0.5:
                best_similarity = similarity
                best_match = orig
        
        if best_match:
            print(f"✅ 낮은 cutoff 매칭 성공: '{ocr_result}' → '{best_match}' (유사도: {best_similarity:.3f})")
            return best_match
    
    print(f"❌ 유사도 매칭 실패: '{ocr_result}' (최고 유사도: {best_similarity:.3f})")
    return None

def find_medicine_info(medicine_name: str, all_docs: List[Document], is_ocr_result: bool = False) -> dict:
    """약품명으로 약품 정보를 찾아서 반환"""
    medicine_info = {
        "제품명": medicine_name,
        "효능": "정보 없음",
        "부작용": "정보 없음", 
        "사용법": "정보 없음",
        "주의사항": "정보 없음"
    }
    
    # 정확한 제품명 매칭 시도
    exact_matches = [doc for doc in all_docs if doc.metadata.get("제품명") == medicine_name]
    
    if not exact_matches:
        # 부분 매칭 시도 (약품명이 포함된 경우)
        partial_matches = []
        for doc in all_docs:
            doc_name = doc.metadata.get("제품명", "")
            if medicine_name in doc_name or doc_name in medicine_name:
                partial_matches.append(doc)
        
        if partial_matches:
            exact_matches = partial_matches
        else:
            # OCR 결과인 경우에만 유사도 기반 매칭 시도
            if is_ocr_result:
                print(f"🔍 OCR 결과 유사도 기반 약품명 매칭 시도: '{medicine_name}'")
                
                # 모든 약품명 리스트 생성
                medicine_list = [doc.metadata.get("제품명", "") for doc in all_docs if doc.metadata.get("제품명")]
                medicine_list = list(set(medicine_list))  # 중복 제거
                
                # 유사도 매칭 시도
                similar_medicine = find_similar_medicine_name(medicine_name, medicine_list, cutoff=0.8)
                if similar_medicine:
                    print(f"✅ 유사도 매칭 성공: '{medicine_name}' → '{similar_medicine}'")
                    # 유사한 약품명으로 다시 검색
                    exact_matches = [doc for doc in all_docs if doc.metadata.get("제품명") == similar_medicine]
                    # medicine_info의 제품명을 올바른 약품명으로 업데이트
                    medicine_info["제품명"] = similar_medicine
                else:
                    print(f"🔍 유사도 매칭 실패: '{medicine_name}'")
                    # 기존 정규화 방식도 시도
                    normalized_medicine = re.sub(r"[^\w가-힣]", "", medicine_name.lower())
                    for doc in all_docs:
                        doc_name = doc.metadata.get("제품명", "")
                        normalized_doc_name = re.sub(r"[^\w가-힣]", "", doc_name.lower())
                        if normalized_medicine in normalized_doc_name or normalized_doc_name in normalized_medicine:
                            exact_matches.append(doc)
                            break
            else:
                print(f"🔍 단일 텍스트 질문: 유사도 매칭 건너뜀")
    
    if not exact_matches:
        return medicine_info
    
    # 약품 정보 수집 (여러 Excel 파일에서 병합)
    import os
    import re
    url_pattern = r'https?://[^\s]+'
    
    # 새 Excel 파일 우선순위 설정
    new_excel_file = r"C:\Users\jung\Desktop\33\OpenData_ItemPermitDetail20251115.xls"
    
    # 모든 매칭된 문서를 파일별로 그룹화
    docs_by_file = {}
    for doc in exact_matches:
        excel_file = doc.metadata.get("excel_file")
        if excel_file:
            if excel_file not in docs_by_file:
                docs_by_file[excel_file] = []
            docs_by_file[excel_file].append(doc)
    
    # 새 Excel 파일이 있으면 우선순위로 설정
    file_priority = []
    if new_excel_file in docs_by_file:
        file_priority.append(new_excel_file)
    for file in docs_by_file.keys():
        if file != new_excel_file:
            file_priority.append(file)
    
    print(f"📂 약품 정보 출처 파일: {len(file_priority)}개 파일에서 발견")
    for file in file_priority:
        print(f"  - {os.path.basename(file)} ({len(docs_by_file[file])}개 청크)")
    
    # 모든 Excel 파일에서 정보 수집 (파일별로 그룹화)
    excel_file = None
    excel_row_index = None
    
    # 각 파일별로 정보를 수집하여 리스트로 저장
    all_efficacy_info = []  # [(파일명, 효능정보), ...]
    all_side_effects_info = []  # [(파일명, 부작용정보), ...]
    all_usage_info = []  # [(파일명, 사용법정보), ...]
    
    for file in file_priority:
        file_name = os.path.basename(file)
        file_efficacy = None
        file_side_effects = None
        file_usage = None
        
        for doc in docs_by_file[file]:
            content = doc.page_content
            doc_type = doc.metadata.get("type", "")
            
            # Excel 파일 정보 저장 (우선순위가 높은 파일에서)
            if not excel_file:
                excel_file = doc.metadata.get("excel_file")
                excel_row_index = doc.metadata.get("excel_row_index")
            
            # 효능과 부작용은 main 타입에서 추출
            if doc_type == "main" or doc_type == "":
                efficacy = extract_field_from_doc(content, "효능")
                side_effects = extract_field_from_doc(content, "부작용")
                
                # URL이 아닌 경우에만 수집
                if efficacy != "정보 없음" and not re.search(url_pattern, str(efficacy)):
                    if file_efficacy is None:
                        file_efficacy = efficacy
                    else:
                        # 같은 파일 내에서 여러 청크가 있으면 더 긴 것을 선택
                        if len(efficacy) > len(file_efficacy):
                            file_efficacy = efficacy
                
                if side_effects != "정보 없음" and not re.search(url_pattern, str(side_effects)):
                    if file_side_effects is None:
                        file_side_effects = side_effects
                    else:
                        if len(side_effects) > len(file_side_effects):
                            file_side_effects = side_effects
            
            # 사용법은 usage 타입에서 추출
            if doc_type == "usage":
                usage = extract_field_from_doc(content, "사용법")
                if usage != "정보 없음" and not re.search(url_pattern, str(usage)):
                    if file_usage is None:
                        file_usage = usage
                    else:
                        if len(usage) > len(file_usage):
                            file_usage = usage
        
        # 파일별로 수집한 정보를 리스트에 추가
        if file_efficacy:
            all_efficacy_info.append((file_name, file_efficacy))
            print(f"📋 {file_name}에서 효능 정보 수집: {len(file_efficacy)}자")
        if file_side_effects:
            all_side_effects_info.append((file_name, file_side_effects))
            print(f"📋 {file_name}에서 부작용 정보 수집: {len(file_side_effects)}자")
        if file_usage:
            all_usage_info.append((file_name, file_usage))
            print(f"📋 {file_name}에서 사용법 정보 수집: {len(file_usage)}자")
    
    # 여러 소스의 정보를 LLM으로 병합
    if len(all_efficacy_info) > 1:
        print(f"🔄 {len(all_efficacy_info)}개 소스의 효능 정보 병합 중...")
        merged_efficacy = merge_multiple_sources_with_llm(all_efficacy_info, "효능")
        medicine_info["효능"] = merged_efficacy
    elif len(all_efficacy_info) == 1:
        medicine_info["효능"] = all_efficacy_info[0][1]
    
    if len(all_side_effects_info) > 1:
        print(f"🔄 {len(all_side_effects_info)}개 소스의 부작용 정보 병합 중...")
        merged_side_effects = merge_multiple_sources_with_llm(all_side_effects_info, "부작용")
        medicine_info["부작용"] = merged_side_effects
    elif len(all_side_effects_info) == 1:
        medicine_info["부작용"] = all_side_effects_info[0][1]
    
    if len(all_usage_info) > 1:
        print(f"🔄 {len(all_usage_info)}개 소스의 사용법 정보 병합 중...")
        merged_usage = merge_multiple_sources_with_llm(all_usage_info, "사용법")
        medicine_info["사용법"] = merged_usage
    elif len(all_usage_info) == 1:
        medicine_info["사용법"] = all_usage_info[0][1]
    
    # PDF 링크 확인 및 다운로드 (모든 파일에서 수집하여 병합)
    from pdf_link_extractor import enrich_excel_row_with_pdf_content
    from retrievers import file_column_mappings, default_columns
    
    # 모든 파일에서 PDF 정보 수집
    all_pdf_efficacy = []
    all_pdf_side_effects = []
    all_pdf_usage = []
    
    for file in file_priority:
        # 해당 파일의 문서에서 excel_row_index 찾기
        file_row_index = None
        for doc in docs_by_file[file]:
            if doc.metadata.get("excel_file") == file:
                file_row_index = doc.metadata.get("excel_row_index")
                if file_row_index is not None:
                    break
        
        if file_row_index is None:
            continue
        
        print(f"📥 PDF 다운로드 시도: {os.path.basename(file)}, 행 {file_row_index}")
        try:
            # 파일별 컬럼 매핑 확인
            if file in file_column_mappings:
                col_mapping = file_column_mappings[file]
            else:
                col_mapping = default_columns
            
            pdf_column_mapping = {
                '효능': col_mapping['효능'],
                '복용법': col_mapping['사용법'],
                '주의사항': col_mapping['부작용']
            }
            
            # 효능, 부작용, 사용법이 URL인지 확인하고 PDF 다운로드
            pdf_content = enrich_excel_row_with_pdf_content(
                file, file_row_index, ['효능', '주의사항', '복용법'], pdf_column_mapping
            )
            
            print(f"📋 PDF 내용 확인: {list(pdf_content.keys())}")
            for key, value in pdf_content.items():
                if value:
                    print(f"  - {key}: {len(str(value))}자 - {str(value)[:100]}...")
                    # PDF 정보를 리스트에 추가
                    file_name = os.path.basename(file)
                    if key == '효능' and value:
                        all_pdf_efficacy.append((file_name, value))
                    elif key == '주의사항' and value:
                        all_pdf_side_effects.append((file_name, value))
                    elif key == '복용법' and value:
                        all_pdf_usage.append((file_name, value))
                else:
                    print(f"  - {key}: None")
        
        except Exception as e:
            print(f"⚠️ {os.path.basename(file)} PDF 다운로드 실패 (계속 진행): {e}")
    
    # PDF 정보를 기존 Excel 정보와 병합
    if all_pdf_efficacy:
        current_efficacy = medicine_info.get("효능", "정보 없음")
        if current_efficacy != "정보 없음":
            # Excel 정보와 PDF 정보를 모두 병합
            all_efficacy_sources = all_efficacy_info + all_pdf_efficacy
            if len(all_efficacy_sources) > 1:
                print(f"🔄 Excel + PDF 효능 정보 병합 중... ({len(all_efficacy_sources)}개 소스)")
                merged_efficacy = merge_multiple_sources_with_llm(all_efficacy_sources, "효능")
                medicine_info["효능"] = merged_efficacy
            else:
                medicine_info["효능"] = all_efficacy_sources[0][1]
        else:
            # Excel 정보가 없으면 PDF 정보만 사용
            if len(all_pdf_efficacy) > 1:
                merged_efficacy = merge_multiple_sources_with_llm(all_pdf_efficacy, "효능")
                medicine_info["효능"] = merged_efficacy
            elif len(all_pdf_efficacy) == 1:
                medicine_info["효능"] = all_pdf_efficacy[0][1]
    
    if all_pdf_side_effects:
        current_side_effects = medicine_info.get("부작용", "정보 없음")
        if current_side_effects != "정보 없음":
            # Excel 정보와 PDF 정보를 모두 병합
            all_side_effects_sources = all_side_effects_info + all_pdf_side_effects
            if len(all_side_effects_sources) > 1:
                print(f"🔄 Excel + PDF 부작용 정보 병합 중... ({len(all_side_effects_sources)}개 소스)")
                merged_side_effects = merge_multiple_sources_with_llm(all_side_effects_sources, "부작용")
                medicine_info["부작용"] = merged_side_effects
            else:
                medicine_info["부작용"] = all_side_effects_sources[0][1]
        else:
            # Excel 정보가 없으면 PDF 정보만 사용
            if len(all_pdf_side_effects) > 1:
                merged_side_effects = merge_multiple_sources_with_llm(all_pdf_side_effects, "부작용")
                medicine_info["부작용"] = merged_side_effects
            elif len(all_pdf_side_effects) == 1:
                medicine_info["부작용"] = all_pdf_side_effects[0][1]
    
    if all_pdf_usage:
        current_usage = medicine_info.get("사용법", "정보 없음")
        if current_usage != "정보 없음":
            # Excel 정보와 PDF 정보를 모두 병합
            all_usage_sources = all_usage_info + all_pdf_usage
            if len(all_usage_sources) > 1:
                print(f"🔄 Excel + PDF 사용법 정보 병합 중... ({len(all_usage_sources)}개 소스)")
                merged_usage = merge_multiple_sources_with_llm(all_usage_sources, "사용법")
                medicine_info["사용법"] = merged_usage
            else:
                medicine_info["사용법"] = all_usage_sources[0][1]
        else:
            # Excel 정보가 없으면 PDF 정보만 사용
            if len(all_pdf_usage) > 1:
                merged_usage = merge_multiple_sources_with_llm(all_pdf_usage, "사용법")
                medicine_info["사용법"] = merged_usage
            elif len(all_pdf_usage) == 1:
                medicine_info["사용법"] = all_pdf_usage[0][1]
    
    # 연령대 금기 성분 정보 추가
    try:
        from retrievers import get_medicine_age_contraindications
        age_contraindications = get_medicine_age_contraindications(medicine_name)
        if age_contraindications:
            medicine_info["연령대_금기_정보"] = age_contraindications
            print(f"✅ 연령대 금기 정보 추가: {len(age_contraindications)}개 성분")
    except Exception as e:
        print(f"⚠️ 연령대 금기 정보 수집 실패: {e}")
    
    # 일일 최대 투여량 정보 추가
    try:
        from retrievers import get_medicine_daily_max_dosage
        daily_max_dosage = get_medicine_daily_max_dosage(medicine_name)
        if daily_max_dosage:
            medicine_info["일일_최대_투여량_정보"] = daily_max_dosage
            print(f"✅ 일일 최대 투여량 정보 추가: {len(daily_max_dosage)}개 성분")
    except Exception as e:
        print(f"⚠️ 일일 최대 투여량 정보 수집 실패: {e}")
    
    return medicine_info

def extract_field_from_doc(text: str, label: str) -> str:
    """문서에서 특정 필드 추출"""
    pattern = rf"\[{label}\]:\s*((?:.|\n)*?)(?=\n\[|\Z)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else "정보 없음"

def merge_multiple_sources_with_llm(sources_info: List[tuple], field_name: str) -> str:
    """
    여러 소스의 정보를 LLM으로 병합합니다.
    중복 내용은 제거하고, 각 소스의 고유한 내용은 모두 포함합니다.
    
    Args:
        sources_info: [(소스명, 정보), ...] 형식의 리스트
        field_name: 필드명 (효능, 부작용, 사용법 등)
    
    Returns:
        병합된 정보
    """
    if not sources_info:
        return "정보 없음"
    
    if len(sources_info) == 1:
        return sources_info[0][1]
    
    try:
        print(f"🔄 {len(sources_info)}개 소스의 {field_name} 정보 병합 중...")
        
        # 소스별 정보를 정리
        sources_text = ""
        for i, (source_name, info) in enumerate(sources_info, 1):
            sources_text += f"\n**소스 {i} ({source_name}):**\n{info}\n"
        
        merge_prompt = f"""당신은 의약품 정보 전문가입니다. 여러 소스에서 수집한 {field_name} 정보를 병합하여 완전한 정보를 만들어주세요.

**병합 원칙:**
1. 중복되는 내용은 하나로 통합 (같은 의미의 내용이 여러 소스에 있으면 하나만 유지)
2. 각 소스의 고유한 내용은 반드시 모두 포함 (소스별로 다른 정보가 있으면 모두 추가)
3. 모든 중요한 정보를 포함 (금기사항, 주의사항, 용량 정보, 특수 사용법 등)
4. 구체적인 수치나 용량 정보는 모두 유지
5. 자연스러운 문장으로 통합
6. 소스별로 약간씩 다른 표현이라도 의미가 다르면 모두 포함

**수집된 {field_name} 정보 (여러 소스):**
{sources_text}

**병합된 {field_name} 정보 (중복 제거, 모든 고유 정보 포함):**
"""
        
        # 캐시 확인
        cached_response = cache_manager.get_llm_response_cache(merge_prompt, f"merge_multiple_{field_name}")
        if cached_response:
            merged = cached_response
        else:
            response = llm.invoke(merge_prompt)
            merged = response.content if hasattr(response, 'content') else str(response)
            # 캐시 저장
            if merged and len(merged) > 50:
                cache_manager.save_llm_response_cache(merge_prompt, merged, f"merge_multiple_{field_name}")
        
        if merged and len(merged) > 50:
            print(f"✅ {field_name} 정보 병합 완료: {len(merged)}자 (원본: {sum(len(info) for _, info in sources_info)}자)")
            return merged.strip()
        else:
            print(f"⚠️ 병합 결과가 너무 짧아 첫 번째 소스 정보 유지")
            return sources_info[0][1]
    
    except Exception as e:
        print(f"⚠️ {field_name} 정보 병합 실패, 첫 번째 소스 정보 유지: {e}")
        return sources_info[0][1]

def merge_medicine_info_with_llm(current_info: str, pdf_info: str, field_name: str) -> str:
    """
    LLM을 사용하여 기존 정보와 PDF 정보를 병합합니다.
    중복 내용은 제거하고, 새로운 내용은 추가합니다.
    
    Args:
        current_info: 기존 정보
        pdf_info: PDF에서 추출한 정보
        field_name: 필드명 (효능, 부작용, 사용법 등)
    
    Returns:
        병합된 정보
    """
    # URL이거나 정보 없음이면 PDF 정보로 교체
    url_pattern = r'https?://[^\s]+'
    if current_info == "정보 없음" or re.search(url_pattern, str(current_info)):
        return pdf_info
    
    # 두 정보가 비슷하면 그냥 기존 정보 유지 (불필요한 LLM 호출 방지)
    if current_info.strip() == pdf_info.strip():
        return current_info
    
    try:
        print(f"🔄 {field_name} 정보 병합 중... (기존: {len(current_info)}자, PDF: {len(pdf_info)}자)")
        
        merge_prompt = f"""당신은 의약품 정보 전문가입니다. 기존 정보와 PDF에서 추출한 정보를 병합하여 완전한 {field_name} 정보를 만들어주세요.

**병합 원칙:**
1. 중복되는 내용은 하나로 통합
2. 기존 정보에 없는 새로운 내용은 반드시 추가
3. 모든 중요한 정보를 포함 (금기사항, 주의사항, 용량 정보 등)
4. 구체적인 수치나 용량 정보는 모두 유지
5. 자연스러운 문장으로 통합

**기존 {field_name} 정보:**
{current_info}

**PDF에서 추출한 {field_name} 정보:**
{pdf_info}

**병합된 {field_name} 정보 (중복 제거, 신규 내용 추가):**
"""
        
        # 캐시 확인
        cached_response = cache_manager.get_llm_response_cache(merge_prompt, f"merge_{field_name}")
        if cached_response:
            merged = cached_response
        else:
            response = llm.invoke(merge_prompt)
            merged = response.content if hasattr(response, 'content') else str(response)
            # 캐시 저장
            if merged and len(merged) > 50:
                cache_manager.save_llm_response_cache(merge_prompt, merged, f"merge_{field_name}")
        
        if merged and len(merged) > 50:
            print(f"✅ {field_name} 정보 병합 완료: {len(merged)}자")
            return merged.strip()
        else:
            print(f"⚠️ 병합 결과가 너무 짧아 기존 정보 유지")
            return current_info
    
    except Exception as e:
        print(f"⚠️ {field_name} 정보 병합 실패, 기존 정보 유지: {e}")
        return current_info

def check_medicine_usage_safety(medicine_info: dict, usage_context: str) -> dict:
    """약품 사용 안전성 판단"""
    
    # 캐시에서 먼저 확인 (용량주의 성분 리스트 통합으로 인해 캐시 비활성화)
    cache_key = f"{medicine_info['제품명']}_{usage_context}"
    cache_file = cache_manager.matching_cache_dir / f"{cache_key}.pkl"
    
    # 용량주의 성분 리스트가 통합되었으므로 캐시를 무시하고 새로 계산
    print(f"🔍 용량주의 성분 리스트 통합으로 인해 캐시 무시: {cache_key}")
    cached_result = None
    
    # 용량주의 성분 정보 확인
    dosage_warnings = get_medicine_dosage_warnings(medicine_info['제품명'])
    dosage_warning_text = ""
    if dosage_warnings:
        dosage_warning_text = "\n\n## ⚠️ 용량주의 성분 정보\n"
        for warning in dosage_warnings:
            ingredient = warning['ingredient']
            dosage_info = warning['dosage_info']
            dosage_warning_text += f"- **{ingredient}**: 1일 최대용량 {dosage_info['max_daily_dose']}\n"
            if dosage_info['remarks'] and dosage_info['remarks'] != 'nan':
                dosage_warning_text += f"  - 비고: {dosage_info['remarks']}\n"
        dosage_warning_text += "\n**중요**: 용량주의 성분이 포함된 약품은 반드시 의사나 약사의 처방에 따라 사용하세요.\n"
    
    # LLM을 사용한 안전성 판단 - 최적화된 프롬프트
    prompt = f"""당신은 의약품 안전성 평가 전문가입니다. 단계별로 분석하여 근거 있는 판단을 내리세요.

## 📋 약품 정보
- 제품명: {medicine_info['제품명']}
- 효능: {medicine_info['효능']}
- 부작용: {medicine_info['부작용']}
- 사용법: {medicine_info['사용법']}{dosage_warning_text}

## 🎯 사용 상황
{usage_context}

## 🔍 3단계 평가 프로세스

### STEP 1: 효능-증상 매칭 분석 (가장 중요)
아래 의학적 증상 매핑을 참고하여 약품 효능과 사용 상황의 연관성을 평가하세요.

**의학적 증상 매핑:**
- 피부 질환: 습진 ↔ 아토피 ↔ 피부염 ↔ 발진 ↔ 가려움 ↔ 두드러기
- 상처/외상: 상처 ↔ 찰과상 ↔ 긁힘 ↔ 베인 상처 ↔ 외상 ↔ 화상
- 통증: 두통 ↔ 편두통 ↔ 머리 아픔 / 근육통 ↔ 몸살 / 치통 ↔ 잇몸 통증
- 피로: 피로 ↔ 피곤함 ↔ 무기력 ↔ 기운 없음 ↔ 체력 저하 ↔ 육체 피로
- 소화: 소화불량 ↔ 체함 ↔ 속 불편 ↔ 위장 장애 ↔ 더부룩함
- 감염: 세균 감염 ↔ 화농 ↔ 염증 ↔ 고름
- 감기: 감기 ↔ 코감기 ↔ 목감기 ↔ 기침 ↔ 콧물 ↔ 인후통

**분석 질문:**
1. 약품 효능에 명시된 증상이 사용 상황과 직접 일치하는가?
2. 위 매핑에서 의미적으로 유사한 증상인가?
3. 매칭 강도: 완전일치(100%) / 강한연관(80%) / 중간연관(50%) / 약한연관(30%) / 무관(0%)

**STEP 1 결과:**
- 매칭 강도: ___%
- 근거: [효능의 어떤 부분이 사용 상황과 연관되는지 구체적으로 설명]

### STEP 2: 용량주의 성분 평가 (새로 추가)
**용량주의 성분 점검:**
- 용량주의 성분이 포함되어 있는가?
- 1일 최대용량 정보가 제공되었는가?
- 복합제인 경우 각 성분별 용량 고려 필요

**용량주의 성분이 있는 경우:**
- 반드시 의사나 약사 처방 필요
- 자가 처방 금지
- 용량 초과 시 심각한 부작용 가능성

**STEP 2 결과:**
- 용량주의 여부: 있음 / 없음
- 처방 필요성: 필수 / 권장 / 불필요
- 근거: [구체적 설명]

### STEP 3: 위험도 평가
**부작용 심각도 점검:**
- 심각한 부작용 있음? (쇼크, 중증 알레르기 등) → 위험
- 일반적 부작용만 있음? (졸음, 가벼운 소화불량 등) → 보통
- 부작용 미미 또는 없음? → 안전

**사용 상황 적합성:**
- 해당 상황에서 부작용이 치명적인가?
- 사용법이 상황에 맞는가? (경구/외용 등)

**STEP 3 결과:**
- 위험 수준: 높음 / 보통 / 낮음
- 근거: [구체적 설명]

### STEP 4: 최종 판단
**종합 점수 계산:**
- 용량주의 성분 있음 → 반드시 의사/약사 처방 필요
- 매칭 강도 ≥ 50% + 위험 수준 낮음/보통 + 용량주의 없음 → 사용 가능
- 매칭 강도 < 50% 또는 위험 수준 높음 → 사용 불가

**신뢰도 평가:**
- 높음: 명확한 효능 일치 + 안전성 확인됨 + 용량주의 정보 확인됨
- 중간: 유사 증상 + 큰 위험 없음 + 용량주의 없음
- 낮음: 효능 불명확하거나 위험 요소 있음 또는 용량주의 성분 포함

## 💡 판단 예시

### 예시 1: 베타딘 연고 + 상처
- STEP 1: 효능 "상처 소독, 세균 감염 예방" vs 사용 "상처" → 100% 일치
- STEP 2: 부작용 "피부 자극" → 경미, 위험 낮음
- STEP 3: 사용 가능 (신뢰도: 높음)

### 예시 2: 감기약 + 두통
- STEP 1: 효능 "감기 증상 완화(두통, 발열)" vs 사용 "두통" → 80% 강한 연관
- STEP 2: 부작용 "졸음" → 경미, 위험 낮음
- STEP 3: 사용 가능 (신뢰도: 높음)

### 예시 3: 피부 연고 + 근육통
- STEP 1: 효능 "습진, 피부염 완화" vs 사용 "근육통" → 0% 무관
- STEP 2: 효능 불일치
- STEP 3: 사용 불가 (신뢰도: 높음)

## 📤 출력 형식 (JSON)
{{
    "safe_to_use": true/false,
    "confidence_score": 0.0~1.0,
    "matching_strength": 0~100,
    "has_dosage_warning": true/false,
    "prescription_required": true/false,
    "reason": "STEP 1-4 분석 결과를 바탕으로 한 구체적 근거 (2-3문장)",
    "precautions": "주의사항 (필요시)",
    "dosage_warnings": ["용량주의 성분 정보 (있는 경우)"],
    "alternative_suggestion": "대안 제안 (사용 불가 시)"
}}

**중요**: 추측하지 말고 주어진 약품 정보만으로 판단하세요. 불확실하면 confidence_score를 낮추세요.
"""
    
    try:
        # 캐시 확인
        cached_response = cache_manager.get_llm_response_cache(prompt, "usage_check")
        if cached_response:
            response = cached_response
        else:
            response = llm.invoke(prompt).content.strip()
            # 캐시 저장
            cache_manager.save_llm_response_cache(prompt, response, "usage_check")
        print(f"🔍 LLM 응답: {response[:200]}...")
        
        # JSON 응답 파싱 (```json 제거 처리)
        try:
            # ```json과 ``` 제거
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                if json_end != -1:
                    response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                if json_end != -1:
                    response = response[json_start:json_end].strip()
            
            result = json.loads(response)
            
            # 새로운 필드가 없으면 기본값 추가 (하위 호환성)
            if "confidence_score" not in result:
                result["confidence_score"] = 0.7  # 기본 중간 신뢰도
            if "matching_strength" not in result:
                result["matching_strength"] = 50  # 기본 중간 매칭
            if "has_dosage_warning" not in result:
                result["has_dosage_warning"] = len(dosage_warnings) > 0
            if "prescription_required" not in result:
                result["prescription_required"] = len(dosage_warnings) > 0
            if "dosage_warnings" not in result:
                result["dosage_warnings"] = [f"{w['ingredient']}: {w['dosage_info']['max_daily_dose']}" for w in dosage_warnings]
            
            print(f"✅ JSON 파싱 성공: safe_to_use={result.get('safe_to_use')}, confidence={result.get('confidence_score')}, matching={result.get('matching_strength')}%, dosage_warning={result.get('has_dosage_warning')}")
        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 실패: {e}")
            print(f"🔍 원본 응답: {response}")
            # JSON 파싱 실패 시 기본 응답
            result = {
                "safe_to_use": False,
                "confidence_score": 0.3,
                "matching_strength": 0,
                "has_dosage_warning": len(dosage_warnings) > 0,
                "prescription_required": len(dosage_warnings) > 0,
                "reason": "약품 정보를 분석할 수 없습니다.",
                "precautions": "의사나 약사와 상담하세요.",
                "dosage_warnings": [f"{w['ingredient']}: {w['dosage_info']['max_daily_dose']}" for w in dosage_warnings],
                "alternative_suggestion": ""
            }
        
        # 캐시에 저장
        try:
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            print(f"💾 사용 가능성 캐시 저장됨: {cache_key}")
        except Exception as e:
            print(f"❌ 사용 가능성 캐시 저장 실패: {e}")
        
        return result
        
    except Exception as e:
        print(f"❌ 약품 사용 안전성 판단 중 오류 발생: {e}")
        return {
            "safe_to_use": False,
            "confidence_score": 0.0,
            "matching_strength": 0,
            "has_dosage_warning": len(dosage_warnings) > 0,
            "prescription_required": len(dosage_warnings) > 0,
            "reason": "안전성 판단 중 오류가 발생했습니다.",
            "precautions": "의사나 약사와 상담하세요.",
            "dosage_warnings": [f"{w['ingredient']}: {w['dosage_info']['max_daily_dose']}" for w in dosage_warnings],
            "alternative_suggestion": ""
        }

def generate_usage_check_response(medicine_name: str, usage_context: str, medicine_info: dict, safety_result: dict) -> str:
    """사용 가능성 판단 결과를 사용자 친화적인 응답으로 변환"""
    
    # usage_context에서 질문 형태의 문장을 정리하여 자연스러운 표현으로 변환
    clean_context = usage_context
    if "?" in usage_context:
        import re
        # 질문 형태에서 핵심 증상/상황만 추출
        # "이 연고 습진에 발라도 되나?" → "습진에"
        # "박테로신이라는 연고 습진에 발라도 되나?" → "습진에"
        # "두통에 먹어도 되나?" → "두통에"
        # "상처에 발라도 되나?" → "상처에"
        
        # 더 정확한 패턴 매칭
        patterns = [
            r'([가-힣]+에)\s+[가-힣\s]*발라도\s+되나\?',  # "습진에 발라도 되나?"
            r'([가-힣]+에)\s+[가-힣\s]*먹어도\s+되나\?',   # "두통에 먹어도 되나?"
            r'([가-힣]+에)\s+[가-힣\s]*써도\s+되나\?',     # "상처에 써도 되나?"
            r'([가-힣]+에)\s+[가-힣\s]*사용해도\s+되나\?', # "상처에 사용해도 되나?"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, usage_context)
            if match:
                clean_context = match.group(1)
                break
        
        # 패턴 매칭이 실패한 경우 기본 처리
        if clean_context == usage_context:
            clean_context = usage_context.replace("?", "").strip()
    
    # 신뢰도 및 매칭 강도 정보
    confidence = safety_result.get("confidence_score", 0.7)
    matching = safety_result.get("matching_strength", 50)
    
    # 신뢰도 레벨 표시
    if confidence >= 0.8:
        confidence_text = "높음 🟢"
    elif confidence >= 0.5:
        confidence_text = "중간 🟡"
    else:
        confidence_text = "낮음 🔴"
    
    # 용량주의 정보 확인
    has_dosage_warning = safety_result.get("has_dosage_warning", False)
    prescription_required = safety_result.get("prescription_required", False)
    dosage_warnings = safety_result.get("dosage_warnings", [])
    
    if safety_result["safe_to_use"]:
        response = f"✅ **{medicine_name}**을(를) {clean_context} 사용하는 것은 **가능**합니다.\n\n"
        response += f"**판단 근거:** {safety_result['reason']}\n\n"
        response += f"**신뢰도:** {confidence_text} (효능 매칭: {matching}%)\n\n"
        
        # 용량주의 정보 추가
        if has_dosage_warning:
            response += f"**⚠️ 용량주의 성분 포함:**\n"
            for warning in dosage_warnings:
                response += f"- {warning}\n"
            response += f"\n**중요:** 용량주의 성분이 포함된 약품은 반드시 의사나 약사의 처방에 따라 사용하세요.\n\n"
        
        if safety_result.get("precautions"):
            response += f"**⚠️ 주의사항:** {safety_result['precautions']}\n\n"
    else:
        response = f"❌ **{medicine_name}**을(를) {clean_context} 사용하는 것은 **권장하지 않습니다**.\n\n"
        response += f"**판단 근거:** {safety_result['reason']}\n\n"
        response += f"**신뢰도:** {confidence_text} (효능 매칭: {matching}%)\n\n"
        
        # 용량주의 정보 추가
        if has_dosage_warning:
            response += f"**⚠️ 용량주의 성분 포함:**\n"
            for warning in dosage_warnings:
                response += f"- {warning}\n"
            response += f"\n**중요:** 용량주의 성분이 포함된 약품은 반드시 의사나 약사의 처방에 따라 사용하세요.\n\n"
        
        if safety_result.get("precautions"):
            response += f"**⚠️ 주의사항:** {safety_result['precautions']}\n\n"
        
        if safety_result.get("alternative_suggestion"):
            response += f"**💡 대안 제안:** {safety_result['alternative_suggestion']}\n\n"
    
    # 약품 정보 요약 추가
    response += "**약품 정보 요약:**\n"
    response += f"- 효능: {medicine_info['효능']}\n"
    response += f"- 부작용: {medicine_info['부작용']}\n"
    response += f"- 사용법: {medicine_info['사용법']}\n\n"
    
    response += "⚠️ **중요:** 이 정보는 참고용이며, 정확한 진단과 처방을 위해서는 의사나 약사와 상담하시기 바랍니다."
    
    return response

def medicine_usage_check_node(state: QAState) -> QAState:
    """약품 사용 가능성 판단 노드"""
    
    # ⚠️ 중요: question_refinement_node에서 보정된 약품명이 있으면 우선 사용
    medicine_name = state.get("extracted_medicine_name") or state.get("medicine_name", "")
    usage_context = state.get("usage_context", "")
    
    if not medicine_name or not usage_context:
        state["usage_check_answer"] = "죄송합니다. 약품명이나 사용 상황 정보가 부족하여 판단할 수 없습니다."
        return state
    
    # 보정된 약품명으로 state 업데이트 (다음 노드에서도 사용하도록)
    if state.get("extracted_medicine_name") and state.get("extracted_medicine_name") != state.get("medicine_name"):
        state["medicine_name"] = medicine_name
        print(f"✅ 보정된 약품명으로 state 업데이트: '{state.get('medicine_name', '')}' → '{medicine_name}'")
    
    print(f"🔍 약품 사용 가능성 판단 시작: {medicine_name} → {usage_context}")
    
    # Excel DB에서만 검색 (PDF DB 제거)
    print("📊 Excel DB에서 약품 정보 검색 중...")
    # 이미지가 포함된 경우(OCR 결과)인지 확인
    is_ocr_result = state.get("has_image", False) or state.get("extracted_text") is not None
    medicine_info = find_medicine_info(medicine_name, excel_docs, is_ocr_result)
    
    # 약품 정보를 찾지 못한 경우
    if medicine_info["효능"] == "정보 없음":
        state["usage_check_answer"] = f"죄송합니다. '{medicine_name}'에 대한 정보를 찾을 수 없습니다. 정확한 약품명을 확인하시거나 의사/약사와 상담하시기 바랍니다."
        return state
    
    print(f"✅ 약품 정보 발견: {medicine_info['제품명']}")
    print(f"📊 최종 medicine_info 상태:")
    print(f"  - 효능: {medicine_info.get('효능', '정보 없음')[:100]}... (길이: {len(str(medicine_info.get('효능', '')))})")
    print(f"  - 부작용: {medicine_info.get('부작용', '정보 없음')[:100]}... (길이: {len(str(medicine_info.get('부작용', '')))})")
    print(f"  - 사용법: {medicine_info.get('사용법', '정보 없음')[:100]}... (길이: {len(str(medicine_info.get('사용법', '')))})")
    
    # 병합된 약품 정보를 state에 저장 (enhanced_rag_system에서 사용)
    state["merged_medicine_info"] = medicine_info
    print(f"💾 병합된 약품 정보 state 저장 완료: {medicine_info.get('제품명', '')} (효능: {len(str(medicine_info.get('효능', '')))}자, 부작용: {len(str(medicine_info.get('부작용', '')))}자)")
    
    # 사용 안전성 판단
    print("🔍 사용 안전성 판단 중...")
    safety_result = check_medicine_usage_safety(medicine_info, usage_context)
    
    # 최종 응답 생성
    print("📝 최종 응답 생성 중...")
    final_response = generate_usage_check_response(medicine_name, usage_context, medicine_info, safety_result)
    
    state["usage_check_answer"] = final_response
    
    print("✅ 약품 사용 가능성 판단 완료")
    return state
