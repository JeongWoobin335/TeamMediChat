# ocr_node.py
import cv2
import numpy as np
from PIL import Image
import io
import re
from typing import Optional, Tuple, List
from qa_state import QAState
from answer_utils import generate_response_llm_from_prompt
from difflib import get_close_matches
from retrievers import excel_docs

# EasyOCR import
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    print("✅ EasyOCR 사용 가능")
except ImportError:
    EASYOCR_AVAILABLE = False
    print("❌ EasyOCR 사용 불가 - 설치가 필요합니다")

def preprocess_image(image_data: bytes) -> np.ndarray:
    """
    이미지 전처리 함수
    - 회전 보정
    - 노이즈 제거
    - 대비 향상
    """
    try:
        # 바이트 데이터를 이미지로 변환
        image = Image.open(io.BytesIO(image_data))
        
        # 이미지 크기 확인 및 리사이즈
        width, height = image.size
        print(f"📏 원본 이미지 크기: {width}x{height}")
        
        # 너무 작은 이미지는 확대
        if width < 300 or height < 300:
            scale_factor = max(300/width, 300/height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"📏 리사이즈된 이미지 크기: {new_width}x{new_height}")
        
        # OpenCV 형식으로 변환
        img_array = np.array(image)
        
        # RGB를 BGR로 변환 (OpenCV는 BGR 사용)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        # 간단한 전처리
        # 노이즈 제거
        denoised = cv2.medianBlur(gray, 3)
        
        # 대비 향상
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 간단한 이진화
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
        
    except Exception as e:
        print(f"❌ 이미지 전처리 중 오류 발생: {e}")
        return None

def extract_text_from_image(image_data: bytes) -> str:
    """
    이미지에서 텍스트 추출 (다중 OCR 엔진 + ROI 기반 처리)
    """
    try:
        # 원본 이미지 직접 사용
        image = Image.open(io.BytesIO(image_data))
        print(f"📏 원본 이미지 크기: {image.size[0]}x{image.size[1]}")
        
        
        # 이미지 크기 확대 (OCR 정확도 향상)
        if image.size[0] < 2000 or image.size[1] < 2000:
            scale_factor = max(2000/image.size[0], 2000/image.size[1])
            new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            print(f"🔄 이미지 리사이즈: {new_size}")
        
        # OpenCV로 변환
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # 대비향상 방법만 사용 (가장 정확한 결과)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        print("✅ 대비향상 전처리 완료")
        
        # EasyOCR로 텍스트 추출
        text = ""
        
        if EASYOCR_AVAILABLE:
            try:
                print("🔍 EasyOCR로 시도...")
                reader = easyocr.Reader(['ko', 'en'], gpu=False)
                result = reader.readtext(enhanced)
                
                if result:
                    texts = []
                    for (bbox, text, confidence) in result:
                        if confidence > 0.2:  # 신뢰도 20% 이상
                            texts.append(text)
                            print(f"  🔍 EasyOCR: '{text}' (신뢰도: {confidence:.2f})")
                    
                    if texts:
                        text = ' '.join(texts)
                        print(f"✅ EasyOCR 결과: '{text}'")
                    else:
                        print("⚠️ EasyOCR에서 신뢰도가 낮은 결과만 발견됨")
                else:
                    print("⚠️ EasyOCR에서 텍스트를 찾을 수 없음")
                        
            except Exception as e:
                print(f"❌ EasyOCR 오류: {e}")
        else:
            print("❌ EasyOCR을 사용할 수 없습니다")
        
        if not text.strip():
            print("❌ OCR 처리 실패")
        
        # 텍스트 정제
        cleaned_text = clean_extracted_text(text)
        
        # 조사 제거 (정규식 기반)
        if cleaned_text:
            # 한글 조사 제거
            cleaned_text = re.sub(r'[은는이가을를에의와과도부터까지에서부터]$', '', cleaned_text)
            # 연속된 공백 제거
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            print(f"🔍 조사 제거 후 OCR 결과: '{cleaned_text}'")
        
        print(f"🔍 최종 OCR 결과: '{cleaned_text}'")
        
        if not cleaned_text.strip():
            print("❌ 모든 OCR 시도가 실패했습니다")
            return ""
        
        return cleaned_text
        
    except Exception as e:
        print(f"❌ OCR 처리 중 오류 발생: {e}")
        return ""

def clean_extracted_text(text: str) -> str:
    """
    OCR로 추출된 텍스트 정제
    """
    if not text:
        return ""
    
    # 불필요한 문자 제거
    text = re.sub(r'[^\w가-힣\s\-\.]', ' ', text)
    
    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 줄바꿈을 공백으로 변환
    text = text.replace('\n', ' ')
    
    return text.strip()

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

def extract_medicine_name_from_text(text: str) -> str:
    """
    추출된 텍스트에서 약품명 추출 (패턴 매칭 + 유사도 매칭 + LLM 기반)
    """
    if not text:
        return ""
    
    # Excel DB에서 약품명 리스트 추출
    medicine_list = []
    try:
        for doc in excel_docs:
            product_name = doc.metadata.get("제품명", "")
            if product_name and product_name not in medicine_list:
                medicine_list.append(product_name)
        print(f"📊 Excel DB에서 {len(medicine_list)}개 약품명 로드")
    except Exception as e:
        print(f"⚠️ Excel DB 로드 실패: {e}")
        medicine_list = []
    
    # 먼저 패턴 매칭으로 약품명 찾기 (더 포괄적으로)
    medicine_patterns = [
        # 구체적인 약품명 패턴 (우선순위 높음)
        r'([가-힣]{2,10})\s*(연고|크림|젤|정|캡슐|시럽|주사|액|분말|가루)',
        r'([가-힣]{2,10})\s*(3중|복합|처방)',
        r'([가-힣]{2,10})\s*(일반의약품|처방약)',
        r'([가-힣]{2,10})\s*(치료|감염|예방)',
        r'([가-힣]{2,10})\s*(외상|상처|화상)',
        r'([가-힣]{2,10})\s*(10g|20g|30g|50g|100g)',
        r'([가-힣]{2,10})\s*(mg|g|ml)',
        
        # 일반적인 한글 패턴
        r'([가-힣]{2,10})\s*[0-9]',  # 한글 + 숫자
        r'([가-힣]{2,10})\s*[a-zA-Z]',  # 한글 + 영문
        r'([가-힣]{2,10})',  # 단순히 한글 2-10자
        
        # 더 넓은 범위의 한글 패턴
        r'([가-힣]{1,15})',  # 한글 1-15자 (더 넓게)
        
        # 특수 문자 포함 패턴
        r'([가-힣]{2,10})[^\w\s]',  # 한글 + 특수문자
        r'[^\w\s]([가-힣]{2,10})',  # 특수문자 + 한글
    ]
    
    # 제외할 단어들 (약품명이 아닌 것들) - 기본 단어만
    exclude_words = [
        '약학정보원', '정보원', '약학', '정보', '원', '치료', '예방', '감염', '외상', '상처', '화상',
        '복합', '처방', '일반의약품', '처방약', '및', '의', '와', '과', '을', '를', '이', '가',
        '3중', '2차', '10g', '20g', '30g', '50g', '100g', 'mg', 'g', 'ml', 'KPIC'
    ]
    
    # 제외할 패턴들 (하드코딩 대신 패턴 매칭)
    exclude_patterns = [
        r'.*복합.*처방.*',  # "3중복합처방의", "중복합전방의" 등
        r'.*복합.*전방.*',  # "3중복합전방의" 등  
        r'.*중복.*',        # "중복"이 포함된 모든 단어
        r'.*처방.*',        # "처방"이 포함된 모든 단어
        r'.*전방.*',        # "전방"이 포함된 모든 단어
        r'.*복합.*',        # "복합"이 포함된 모든 단어
        r'^\d+$',           # 숫자만 있는 단어 (8, 10 등)
        r'.*정보원.*',      # "약학정보원" 등
        r'.*치료.*',        # "치료" 관련 단어
        r'.*감염.*',        # "감염" 관련 단어
    ]
    
    
    print(f"🔍 약품명 추출 시도 - 입력 텍스트: '{text}'")
    
    # 일반적인 약품명 패턴 우선 검색 (형태 포함)
    common_medicine_patterns = [
        r'([가-힣]{2,8})\s*(연고|크림|젤)',  # 연고류
        r'([가-힣]{2,8})\s*(정|캡슐)',      # 정제/캡슐류
        r'([가-힣]{2,8})\s*(시럽|액)',      # 액체류
        r'([가-힣]{2,8})\s*(주사|주)',      # 주사제
        r'([가-힣]{2,8})\s*(분말|가루)',    # 분말류
    ]
    
    # 구체적인 약품명 패턴 먼저 검색 (형태 포함)
    for pattern in common_medicine_patterns:
        matches = re.findall(pattern, text)
        if matches:
            # 가장 긴 약품명 선택 (형태 포함)
            best_match = max(matches, key=lambda x: len(x[0]))
            medicine_name = f"{best_match[0]}{best_match[1]}"  # 약품명 + 형태
            if best_match[0] not in exclude_words and len(best_match[0]) >= 2:
                print(f"🔍 약품명 패턴으로 발견: '{medicine_name}' (패턴: {pattern})")
                # 패턴 매칭 성공 후에도 유사도 매칭 시도
                if medicine_list:
                    similar_medicine = find_similar_medicine_name(medicine_name, medicine_list, cutoff=0.8)
                    if similar_medicine:
                        print(f"✅ 패턴 매칭 후 유사도 매칭 성공: '{medicine_name}' → '{similar_medicine}'")
                        return similar_medicine
                return medicine_name
    
    # 스마트한 약품명 선택 (패턴 기반 필터링)
    # OCR 결과에서 한글 단어들을 추출하고 점수 계산
    korean_words = re.findall(r'[가-힣]{2,10}', text)
    if korean_words:
        # 제외 단어와 패턴 필터링
        valid_words = []
        for word in korean_words:
            # 기본 제외 단어 체크
            if word in exclude_words:
                print(f"🔍 제외 단어: '{word}' (기본 제외 목록)")
                continue
            
            # 패턴 기반 제외 체크
            is_excluded = False
            for pattern in exclude_patterns:
                if re.match(pattern, word):
                    print(f"🔍 제외 단어: '{word}' (패턴: {pattern})")
                    is_excluded = True
                    break
            
            if not is_excluded and len(word) >= 2:
                valid_words.append(word)
        
        if valid_words:
            # 길이 기반 점수 계산
            scored_words = []
            for word in valid_words:
                score = len(word)  # 기본 점수: 길이만 고려
                scored_words.append((word, score))
            
            # 점수 순으로 정렬하여 가장 높은 점수의 약품명 선택
            scored_words.sort(key=lambda x: x[1], reverse=True)
            best_word = scored_words[0][0]
            print(f"🔍 스마트 선택: '{best_word}' (점수: {scored_words[0][1]})")
            
            # 스마트 선택 후에도 유사도 매칭 시도
            if medicine_list:
                similar_medicine = find_similar_medicine_name(best_word, medicine_list, cutoff=0.8)
                if similar_medicine:
                    print(f"✅ 스마트 선택 후 유사도 매칭 성공: '{best_word}' → '{similar_medicine}'")
                    return similar_medicine
            
            return best_word
    
    # 정규식 기반 조사 제거는 route_question_node에서 처리
    
    # OCR 오타 수정 제거 - 하드코딩 방식은 확장성 없음
    
    # 텍스트에서 가장 긴 한글 단어 찾기 (약품명 후보)
    korean_words = re.findall(r'[가-힣]{2,10}', text)
    if korean_words:
        # 제외 단어가 아닌 가장 긴 단어 선택
        valid_words = [word for word in korean_words if word not in exclude_words and len(word) >= 2]
        if valid_words:
            medicine_name = max(valid_words, key=len)
            print(f"🔍 가장 긴 한글 단어로 약품명 추정: '{medicine_name}'")
            return medicine_name
    
    for i, pattern in enumerate(medicine_patterns):
        matches = re.findall(pattern, text)
        if matches:
            # 가장 긴 약품명 선택
            medicine_name = max(matches, key=lambda x: len(x[0]))[0]
            
            # 제외 단어에 포함되지 않은 경우만 선택
            if medicine_name not in exclude_words and len(medicine_name) >= 2:
                print(f"🔍 패턴 {i+1} 매칭으로 약품명 발견: '{medicine_name}' (패턴: {pattern})")
                return medicine_name
            else:
                print(f"🔍 패턴 {i+1} 매칭 결과 제외: '{medicine_name}' (제외 단어 또는 너무 짧음)")
        else:
            print(f"🔍 패턴 {i+1} 매칭 실패 (패턴: {pattern})")
    
    # 패턴 매칭 실패시 유사도 매칭 시도
    if medicine_list:
        print("🔍 유사도 기반 약품명 매칭 시도...")
        
        # 추출된 한글 단어들로 유사도 매칭 시도
        korean_words = re.findall(r'[가-힣]{2,10}', text)
        for word in korean_words:
            if word not in exclude_words and len(word) >= 2:
                similar_medicine = find_similar_medicine_name(word, medicine_list, cutoff=0.8)
                if similar_medicine:
                    print(f"✅ 유사도 매칭 성공: '{word}' → '{similar_medicine}'")
                    return similar_medicine
                else:
                    print(f"🔍 '{word}' 유사도 매칭 실패")
        
        # 전체 텍스트로도 유사도 매칭 시도
        similar_medicine = find_similar_medicine_name(text, medicine_list, cutoff=0.7)
        if similar_medicine:
            print(f"✅ 전체 텍스트 유사도 매칭 성공: '{text}' → '{similar_medicine}'")
            return similar_medicine
        else:
            print("🔍 전체 텍스트 유사도 매칭 실패")
    
    # 유사도 매칭이 실패했다면 계속 진행
    
    # 유사도 매칭도 실패시 LLM 사용
    prompt = f"""
다음은 약품 포장지에서 OCR로 추출한 텍스트입니다.
이 텍스트에서 약품명(상품명)을 찾아주세요.

**추출된 텍스트:**
{text}

**찾을 약품명 특징:**
- 정제, 캡슐, 연고, 크림, 젤, 시럽, 주사제 등 모든 약품 형태
- 상품명, 브랜드명, 의약품명, 제품명
- 처방약, 일반의약품, 건강기능식품 등 모든 의약품류
- 예: "바스포", "타이레놀정", "베타딘 연고", "이부프로펜 캡슐", "판콜에이", "게보린"

**제외할 단어들:**
약학정보원, 정보원, 치료, 예방, 감염, 외상, 상처, 화상, 복합, 처방, 일반의약품, 처방약, 3중, 2차, 10g, 20g, 30g, 50g, 100g, mg, g, ml, KPIC, 치료및, 2차감염, 3중복합처방의

**응답 형식:**
약품명만 정확히 추출해서 알려주세요. 여러 개가 있다면 가장 주요한 것 하나만 선택하세요.
약품명을 찾을 수 없으면 "없음"이라고 답하세요.

**약품명:**
"""
    
    try:
        response = generate_response_llm_from_prompt(
            prompt=prompt,
            temperature=0.1,
            max_tokens=50
        )
        
        # "없음"이나 빈 응답 처리
        if not response or response.strip() in ["없음", "찾을 수 없음", "없습니다"]:
            return ""
        
        return response.strip()
        
    except Exception as e:
        print(f"❌ 약품명 추출 중 오류 발생: {e}")
        return ""

def ocr_image_node(state: QAState) -> QAState:
    """
    OCR 이미지 처리 노드
    이미지에서 약품명을 추출하고 사용 가능성 판단을 위한 질문으로 변환
    """
    print("📸 OCR 이미지 처리 노드 시작")
    
    # 이미지 데이터 확인
    image_data = state.get("image_data")
    if not image_data:
        print("❌ 이미지 데이터가 없습니다")
        state["final_answer"] = "죄송합니다. 이미지를 업로드해주세요."
        return state
    
    # OCR로 텍스트 추출
    extracted_text = extract_text_from_image(image_data)
    if not extracted_text:
        print("❌ 이미지에서 텍스트를 추출할 수 없습니다")
        state["final_answer"] = "죄송합니다. 이미지에서 텍스트를 읽을 수 없습니다. 더 선명한 이미지를 업로드해주세요."
        return state
    
    # 약품명 추출 (유사도 매칭 포함)
    medicine_name = extract_medicine_name_from_text(extracted_text)
    if not medicine_name:
        print("❌ 약품명을 찾을 수 없습니다")
        state["final_answer"] = "죄송합니다. 이미지에서 약품명을 찾을 수 없습니다. 약품명이 명확히 보이는 이미지를 업로드해주세요."
        return state
    
    print(f"✅ 최종 약품명: {medicine_name}")
    
    # 원래 질문에서 사용 맥락 추출
    original_query = state.get("query", "")
    usage_context = extract_usage_context_from_query(original_query)
    
    # 상태 업데이트
    state["medicine_name"] = medicine_name
    state["usage_context"] = usage_context
    state["extracted_text"] = extracted_text
    state["routing_decision"] = "usage_check"  # 약품 사용 가능성 판단으로 라우팅
    
    print(f"🎯 OCR 처리 완료 - 약품명: {medicine_name}, 사용 맥락: {usage_context}")
    
    return state

def extract_usage_context_from_query(query: str) -> str:
    """
    사용자 질문에서 사용 맥락 추출
    """
    if not query:
        return "일반적인 사용"
    
    # 질문 형태를 정리하여 자연스러운 표현으로 변환 (medicine_usage_check_node와 동일한 로직)
    clean_context = query
    if "?" in query:
        import re
        # 질문 형태에서 핵심 증상/상황만 추출
        # "이 연고 습진에 발라도 되나?" → "습진에"
        # "이 연고 상처에 발라도 되나?" → "상처에"
        
        # 더 정확한 패턴 매칭
        patterns = [
            r'([가-힣]+에)\s+[가-힣\s]*발라도\s+되나\?',  # "습진에 발라도 되나?"
            r'([가-힣]+에)\s+[가-힣\s]*먹어도\s+되나\?',   # "두통에 먹어도 되나?"
            r'([가-힣]+에)\s+[가-힣\s]*써도\s+되나\?',     # "상처에 써도 되나?"
            r'([가-힣]+에)\s+[가-힣\s]*사용해도\s+되나\?', # "상처에 사용해도 되나?"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                clean_context = match.group(1)
                break
        
        # 패턴 매칭이 실패한 경우 기본 처리
        if clean_context == query:
            clean_context = query.replace("?", "").strip()
    
    print(f"🔍 사용 맥락 정리: '{query}' → '{clean_context}'")
    return clean_context

# 테스트용 함수
def test_ocr_with_image_file(image_path: str) -> str:
    """
    이미지 파일로 OCR 테스트
    """
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # OCR 처리
        extracted_text = extract_text_from_image(image_data)
        medicine_name = extract_medicine_name_from_text(extracted_text)
        
        print(f"추출된 텍스트: {extracted_text}")
        print(f"추출된 약품명: {medicine_name}")
        
        return medicine_name
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        return ""

if __name__ == "__main__":
    # 테스트 실행
    print("🧪 OCR 노드 테스트")
    # test_ocr_with_image_file("test_image.jpg")
