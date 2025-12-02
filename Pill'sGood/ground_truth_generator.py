"""
답지 생성 스크립트: 평가용 질문에 대한 답지 자동 생성
- 성분 질문: PubChem에서 정보 수집 후 번역
- 약품 사용 가능성 질문: Excel DB에서 정보 수집
"""
import os
import json
import re
from typing import Dict, List, Optional
from dotenv import load_dotenv

# 시스템 모듈 import
from pubchem_api import PubChemAPI
from translation_rag import TranslationRAG
from retrievers import excel_docs

load_dotenv()

class GroundTruthGenerator:
    """답지 생성기"""
    
    def __init__(self):
        self.pubchem_api = PubChemAPI()
        self.translation_rag = TranslationRAG()
    
    def extract_field_from_doc(self, content: str, field_name: str) -> str:
        """문서에서 특정 필드 추출"""
        pattern = rf"\[{field_name}\]\s*[:：]?\s*(.*?)(?=\n\[|\Z)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            result = match.group(1).strip()
            if result and result != "정보 없음":
                return result
        return "정보 없음"
    
    def find_medicine_info(self, medicine_name: str) -> Dict:
        """Excel DB에서 약품 정보 찾기"""
        medicine_info = {
            "제품명": medicine_name,
            "효능": "정보 없음",
            "부작용": "정보 없음",
            "사용법": "정보 없음",
            "주성분": "정보 없음"
        }
        
        # 정확한 제품명 매칭 시도
        exact_matches = [doc for doc in excel_docs if doc.metadata.get("제품명") == medicine_name]
        
        # 정확한 매칭이 없으면 부분 매칭 시도
        if not exact_matches:
            partial_matches = []
            for doc in excel_docs:
                product_name = doc.metadata.get("제품명", "")
                if product_name.startswith(medicine_name) or medicine_name in product_name:
                    partial_matches.append(doc)
            
            if partial_matches:
                exact_matches = partial_matches
        
        if not exact_matches:
            return medicine_info
        
        # 약품 정보 수집
        for doc in exact_matches:
            content = doc.page_content
            doc_type = doc.metadata.get("type", "")
            
            if doc_type == "main" or doc_type == "":
                efficacy = self.extract_field_from_doc(content, "효능")
                side_effects = self.extract_field_from_doc(content, "부작용")
                usage = self.extract_field_from_doc(content, "사용법")
                main_ingredient = doc.metadata.get("주성분", "정보 없음")
                
                if efficacy != "정보 없음" and medicine_info["효능"] == "정보 없음":
                    medicine_info["효능"] = efficacy
                if side_effects != "정보 없음" and medicine_info["부작용"] == "정보 없음":
                    medicine_info["부작용"] = side_effects
                if usage != "정보 없음" and medicine_info["사용법"] == "정보 없음":
                    medicine_info["사용법"] = usage
                if main_ingredient != "정보 없음" and medicine_info["주성분"] == "정보 없음":
                    medicine_info["주성분"] = main_ingredient
        
        return medicine_info
    
    def generate_ingredient_ground_truth(self, ingredient_name: str) -> str:
        """성분 질문에 대한 답지 생성"""
        print(f"\n🔍 성분 정보 수집 중: {ingredient_name}")
        
        # 1. PubChem에서 성분 정보 수집
        pubchem_result = self.pubchem_api.analyze_ingredient_comprehensive(ingredient_name)
        
        if not pubchem_result or not pubchem_result.get('cid'):
            return f"{ingredient_name}에 대한 정보를 찾을 수 없습니다."
        
        # 2. 영어 정보를 한국어로 번역
        print(f"🔄 영어 정보 번역 중...")
        translated_result = self.translation_rag.translate_comprehensive_ingredient_info(pubchem_result)
        
        # 3. 답지 생성
        answer_parts = []
        
        # 기본 정보
        answer_parts.append(f"**{ingredient_name}**은(는) 다음과 같은 성분입니다:\n")
        
        # 설명 정보
        if translated_result.get('description_kr'):
            answer_parts.append(f"**설명:**\n{translated_result['description_kr']}\n")
        
        # 약리학 정보 요약
        if translated_result.get('pharmacology_info_kr'):
            pharm_info = translated_result['pharmacology_info_kr']
            
            if pharm_info.get('summary_kr'):
                answer_parts.append(f"**약리학적 특성:**\n{pharm_info['summary_kr']}\n")
            elif pharm_info.get('mechanism_of_action_kr'):
                answer_parts.append(f"**작용기전:**\n{pharm_info['mechanism_of_action_kr']}\n")
            elif pharm_info.get('pharmacodynamics_kr'):
                answer_parts.append(f"**약력학:**\n{pharm_info['pharmacodynamics_kr']}\n")
        
        # 기본 정보 (분자식, 분자량 등)
        if translated_result.get('basic_info_kr'):
            basic_info = translated_result['basic_info_kr']
            if basic_info:
                answer_parts.append("**기본 정보:**\n")
                for key, value in basic_info.items():
                    answer_parts.append(f"- {key}: {value}\n")
        
        # 마무리
        answer_parts.append("\n⚠️ **중요**: 정확한 진단과 처방을 위해서는 의사나 약사와 상담하시기 바랍니다.")
        
        return "\n".join(answer_parts)
    
    def generate_usage_ground_truth(self, medicine_name: str, usage_context: str) -> str:
        """약품 사용 가능성 질문에 대한 답지 생성"""
        print(f"\n🔍 약품 정보 수집 중: {medicine_name} (사용 상황: {usage_context})")
        
        # Excel DB에서 약품 정보 수집
        medicine_info = self.find_medicine_info(medicine_name)
        
        if medicine_info["효능"] == "정보 없음":
            return f"'{medicine_name}'에 대한 정보를 찾을 수 없습니다."
        
        # 답지 생성
        answer_parts = []
        
        # 약품명과 사용 상황
        answer_parts.append(f"**{medicine_name}**을(를) **{usage_context}**에 사용하는 것에 대해 설명드리겠습니다.\n")
        
        # 효능 정보 확인
        efficacy = medicine_info.get("효능", "정보 없음")
        if efficacy != "정보 없음":
            # 효능에 사용 상황이 포함되어 있는지 확인
            usage_context_lower = usage_context.lower()
            efficacy_lower = efficacy.lower()
            
            # 사용 가능 여부 판단
            can_use = False
            reason = ""
            
            # 효능에 직접 언급된 경우
            if usage_context_lower in efficacy_lower:
                can_use = True
                reason = f"{medicine_name}의 효능에 {usage_context}이(가) 포함되어 있습니다."
            else:
                # 유사 키워드 매칭
                context_keywords = {
                    "코감기": ["감기", "비염", "코막힘", "콧물"],
                    "감기": ["감기", "감염", "바이러스"],
                    "근육통": ["근육", "통증", "염증"],
                    "신경통": ["신경", "통증", "염증"],
                    "치질": ["치질", "항문", "출혈"],
                    "체함": ["소화", "위장", "식욕"],
                    "습진": ["피부", "염증", "가려움"]
                }
                
                if usage_context_lower in context_keywords:
                    keywords = context_keywords[usage_context_lower]
                    if any(keyword in efficacy_lower for keyword in keywords):
                        can_use = True
                        reason = f"{medicine_name}의 효능이 {usage_context}과(와) 관련이 있을 수 있습니다."
            
            # 답변 생성
            if can_use:
                answer_parts.append(f"✅ **사용 가능합니다.**\n")
                answer_parts.append(f"{reason}\n")
            else:
                answer_parts.append(f"⚠️ **사용 전 의사/약사 상담 권장**\n")
                answer_parts.append(f"{medicine_name}의 효능과 {usage_context}의 관련성을 정확히 확인하기 위해서는 의사나 약사와 상담하시는 것이 좋습니다.\n")
            
            # 효능 정보
            answer_parts.append(f"**{medicine_name}의 효능:**\n{efficacy}\n")
        
        # 주성분 정보
        main_ingredient = medicine_info.get("주성분", "정보 없음")
        if main_ingredient != "정보 없음":
            answer_parts.append(f"**주성분:** {main_ingredient}\n")
        
        # 사용법 정보
        usage = medicine_info.get("사용법", "정보 없음")
        if usage != "정보 없음":
            answer_parts.append(f"**사용법:**\n{usage}\n")
        
        # 부작용 정보
        side_effects = medicine_info.get("부작용", "정보 없음")
        if side_effects != "정보 없음":
            answer_parts.append(f"**주의사항 및 부작용:**\n{side_effects}\n")
        
        # 마무리
        answer_parts.append("\n⚠️ **중요**: 정확한 진단과 처방을 위해서는 의사나 약사와 상담하시기 바랍니다.")
        
        return "\n".join(answer_parts)
    
    def generate_all_ground_truths(self) -> List[Dict]:
        """모든 질문에 대한 답지 생성"""
        questions = [
            {
                "id": 1,
                "question": "노플정은 코감기에 먹어도 되나?",
                "type": "usage",
                "medicine_name": "노플정",
                "usage_context": "코감기"
            },
            {
                "id": 2,
                "question": "욱씬정은 감기에 먹어도 되나?",
                "type": "usage",
                "medicine_name": "욱씬정",
                "usage_context": "감기"
            },
            {
                "id": 3,
                "question": "푸르설티아민이 뭐야?",
                "type": "ingredient",
                "ingredient_name": "푸르설티아민"
            },
            {
                "id": 4,
                "question": "삐콤씨정은 근육통에 먹어도 되나?",
                "type": "usage",
                "medicine_name": "삐콤씨정",
                "usage_context": "근육통"
            },
            {
                "id": 5,
                "question": "아세트아미노펜이 뭐야?",
                "type": "ingredient",
                "ingredient_name": "아세트아미노펜"
            },
            {
                "id": 6,
                "question": "맥타정은 신경통에 먹어도 되나?",
                "type": "usage",
                "medicine_name": "맥타정",
                "usage_context": "신경통"
            },
            {
                "id": 7,
                "question": "덱시부프로펜이 뭐야?",
                "type": "ingredient",
                "ingredient_name": "덱시부프로펜"
            },
            {
                "id": 8,
                "question": "마노엘정은 치질에 먹어도 되나?",
                "type": "usage",
                "medicine_name": "마노엘정",
                "usage_context": "치질"
            },
            {
                "id": 9,
                "question": "아네모정은 체했을 때 먹어도 되나?",
                "type": "usage",
                "medicine_name": "아네모정",
                "usage_context": "체함"
            },
            {
                "id": 10,
                "question": "구아내정은 습진에 먹어도 되나?",
                "type": "usage",
                "medicine_name": "구아내정",
                "usage_context": "습진"
            }
        ]
        
        results = []
        
        print("=" * 60)
        print("📝 답지 생성 시작")
        print("=" * 60)
        
        for q_data in questions:
            print(f"\n[{q_data['id']}/10] 처리 중: {q_data['question']}")
            
            try:
                if q_data['type'] == 'ingredient':
                    ground_truth = self.generate_ingredient_ground_truth(q_data['ingredient_name'])
                else:
                    ground_truth = self.generate_usage_ground_truth(
                        q_data['medicine_name'],
                        q_data['usage_context']
                    )
                
                result = {
                    "id": q_data['id'],
                    "question": q_data['question'],
                    "type": q_data['type'],
                    "ground_truth": ground_truth
                }
                
                if q_data['type'] == 'ingredient':
                    result["ingredient_name"] = q_data['ingredient_name']
                else:
                    result["medicine_name"] = q_data['medicine_name']
                    result["usage_context"] = q_data['usage_context']
                
                results.append(result)
                print(f"✅ 답지 생성 완료 ({len(ground_truth)}자)")
                
            except Exception as e:
                print(f"❌ 답지 생성 실패: {e}")
                results.append({
                    "id": q_data['id'],
                    "question": q_data['question'],
                    "type": q_data['type'],
                    "ground_truth": f"답지 생성 중 오류 발생: {str(e)}",
                    "error": str(e)
                })
        
        print("\n" + "=" * 60)
        print("✅ 모든 답지 생성 완료!")
        print("=" * 60)
        
        return results
    
    def save_ground_truths(self, results: List[Dict], filename: str = "ground_truth.json"):
        """답지를 JSON 파일로 저장"""
        os.makedirs("evaluation_charts", exist_ok=True)
        filepath = os.path.join("evaluation_charts", filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 답지 저장 완료: {filepath}")
        return filepath


def main():
    """메인 실행 함수"""
    generator = GroundTruthGenerator()
    
    # 모든 답지 생성
    results = generator.generate_all_ground_truths()
    
    # 답지 저장
    generator.save_ground_truths(results)
    
    # 요약 출력
    print("\n📊 생성된 답지 요약:")
    print(f"  - 총 질문 수: {len(results)}")
    print(f"  - 성분 질문: {sum(1 for r in results if r['type'] == 'ingredient')}")
    print(f"  - 사용 가능성 질문: {sum(1 for r in results if r['type'] == 'usage')}")


if __name__ == "__main__":
    main()

