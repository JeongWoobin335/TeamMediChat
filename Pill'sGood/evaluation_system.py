"""
평가 시스템: 본 시스템(Pill'sGood)과 GPT-5의 응답을 Gemini 3.0 Pro로 비교 평가
"""
import os
import json
import time
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# 본 시스템 import
from main_graph import graph
from qa_state import QAState

load_dotenv()

class EvaluationSystem:
    """평가 시스템: 본 시스템과 GPT-5 응답을 Gemini 3.0 Pro로 평가"""
    
    def __init__(self):
        """평가 시스템 초기화"""
        # GPT-5 모델 (추가 기능 없이 순수 모델 응답)
        # 참고: GPT-5가 아직 출시되지 않았다면 "gpt-4o" 또는 다른 모델로 변경 필요
        # GPT-5는 temperature와 max_tokens를 지원하지 않을 수 있으므로 제거
        # hallucination_node.py와 동일한 방식으로 초기화
        self.gpt5_llm = ChatOpenAI(model="gpt-5")
        
        # Gemini 3.0 Pro 평가자 모델
        self.evaluator_llm = ChatGoogleGenerativeAI(
            model="gemini-3-pro-preview",  # Gemini 3.0 Pro Preview
            temperature=1.0,  # 평가는 일관성 있게
            max_output_tokens=4000  # JSON 응답이 길 수 있으므로 증가
        )
        
        # 평가 프롬프트 템플릿
        self.evaluation_prompt_template = ChatPromptTemplate.from_messages([
            ("system", """당신은 의약품 정보 시스템의 전문 평가자입니다. 
두 시스템의 응답을 객관적이고 공정하게 평가해주세요.

평가 기준:
1. 정확성 (Accuracy): 의학적 정보의 정확성 (0-10점)
2. 관련성 (Relevance): 질문과 답변의 관련성 (0-10점)
3. 완전성 (Completeness): 필요한 정보의 완전성 (0-10점)
4. 유용성 (Usefulness): 사용자에게 도움이 되는 정도 (0-10점)
5. 전체 점수 (Overall): 종합 평가 (0-10점)

각 지표에 대해 점수와 간단한 이유를 제공해주세요."""),
            ("human", """다음 질문에 대한 두 시스템의 응답을 평가해주세요.

**질문:**
{question}

**답지 (정답):**
{ground_truth}

**시스템 A 응답 (본 시스템):**
{system_a_response}

**시스템 B 응답 (GPT-5):**
{system_b_response}

다음 JSON 형식으로 평가 결과를 제공해주세요:
{{
    "system_a": {{
        "accuracy": 점수,
        "relevance": 점수,
        "completeness": 점수,
        "usefulness": 점수,
        "overall": 점수,
        "accuracy_reason": "이유",
        "relevance_reason": "이유",
        "completeness_reason": "이유",
        "usefulness_reason": "이유",
        "overall_reason": "이유"
    }},
    "system_b": {{
        "accuracy": 점수,
        "relevance": 점수,
        "completeness": 점수,
        "usefulness": 점수,
        "overall": 점수,
        "accuracy_reason": "이유",
        "relevance_reason": "이유",
        "completeness_reason": "이유",
        "usefulness_reason": "이유",
        "overall_reason": "이유"
    }},
    "comparison": {{
        "winner": "system_a" 또는 "system_b" 또는 "tie",
        "reason": "승자 선정 이유"
    }}
}}""")
        ])
    
    def get_our_system_response(self, question: str) -> str:
        """
        본 시스템(Pill'sGood)의 응답 생성
        
        Args:
            question: 사용자 질문
            
        Returns:
            시스템 응답
        """
        try:
            # QAState 초기화
            initial_state = QAState(
                query=question,
                session_id=f"eval_{int(time.time())}",
                conversation_context="",
                user_context=""
            )
            
            # 그래프 실행
            result = graph.invoke(initial_state)
            
            # 최종 답변 추출
            answer = result.get("final_answer", "죄송합니다. 답변을 생성할 수 없습니다.")
            
            return answer
            
        except Exception as e:
            print(f"❌ 본 시스템 응답 생성 오류: {e}")
            return f"오류 발생: {str(e)}"
    
    def get_gpt5_response(self, question: str) -> str:
        """
        GPT-5의 응답 생성 (추가 기능 없이 순수 모델 응답)
        
        Args:
            question: 사용자 질문
            
        Returns:
            GPT-5 응답
        """
        try:
            prompt = f"""당신은 의약품 정보 전문가입니다. 다음 질문에 대해 정확하고 도움이 되는 답변을 제공해주세요.

질문: {question}

답변:"""
            
            print(f"  🔍 GPT-5 모델 호출 중... (모델: {self.gpt5_llm.model_name if hasattr(self.gpt5_llm, 'model_name') else getattr(self.gpt5_llm, 'model', 'unknown')})")
            response = self.gpt5_llm.invoke(prompt)
            
            # 응답 객체 확인
            if not response:
                print(f"  ⚠️ GPT-5 응답 객체가 None입니다")
                return "응답 객체가 None입니다"
            
            # content 속성 확인
            if not hasattr(response, 'content'):
                print(f"  ⚠️ GPT-5 응답에 content 속성이 없습니다. 응답 타입: {type(response)}")
                print(f"  📝 응답 내용: {str(response)[:200]}")
                return str(response)
            
            content = response.content.strip() if response.content else ""
            
            if not content:
                print(f"  ⚠️ GPT-5 응답이 비어있습니다.")
                print(f"  📝 응답 객체: {type(response)}")
                print(f"  📝 content 속성 값: {repr(response.content)[:200]}")
            
            return content
            
        except Exception as e:
            print(f"  ❌ GPT-5 응답 생성 오류: {e}")
            import traceback
            traceback.print_exc()
            return f"오류 발생: {str(e)}"
    
    def evaluate_responses(self, question: str, ground_truth: str, 
                          system_a_response: str, system_b_response: str) -> Dict:
        """
        Gemini 3.0 Pro를 사용하여 두 응답 평가
        
        Args:
            question: 원본 질문
            ground_truth: 답지 (정답)
            system_a_response: 본 시스템 응답
            system_b_response: GPT-5 응답
            
        Returns:
            평가 결과 딕셔너리
        """
        try:
            # 평가 프롬프트 생성
            prompt = self.evaluation_prompt_template.format_messages(
                question=question,
                ground_truth=ground_truth,
                system_a_response=system_a_response,
                system_b_response=system_b_response
            )
            
            # Gemini 3.0 Pro로 평가
            response = self.evaluator_llm.invoke(prompt)
            evaluation_text = response.content.strip() if response.content else ""
            
            # 디버깅: 응답이 비어있는지 확인
            if not evaluation_text:
                print(f"⚠️ Gemini 응답이 비어있습니다")
                return {
                    "raw_evaluation": "",
                    "error": "Gemini 응답 없음"
                }
            
            # 디버깅: 응답의 처음 부분 출력
            print(f"  📝 Gemini 응답 (처음 300자): {evaluation_text[:300]}")
            
            # JSON 파싱 시도
            try:
                # JSON 블록 추출
                json_text = evaluation_text
                
                # ```json 또는 ``` 블록 추출
                if "```json" in evaluation_text:
                    json_start = evaluation_text.find("```json") + 7
                    remaining_text = evaluation_text[json_start:]
                    json_end = remaining_text.find("```")
                    if json_end != -1:
                        json_text = remaining_text[:json_end].strip()
                    else:
                        json_text = remaining_text.strip()
                elif "```" in evaluation_text:
                    json_start = evaluation_text.find("```") + 3
                    remaining_text = evaluation_text[json_start:]
                    json_end = remaining_text.find("```")
                    if json_end != -1:
                        json_text = remaining_text[:json_end].strip()
                    else:
                        json_text = remaining_text.strip()
                
                # JSON 객체 시작 찾기
                json_start_idx = json_text.find("{")
                if json_start_idx != -1:
                    json_text = json_text[json_start_idx:]
                    
                    # JSON 객체만 추출 (중괄호 매칭)
                    brace_count = 0
                    end_pos = -1
                    for i, char in enumerate(json_text):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i + 1
                                break
                    
                    if end_pos > 0:
                        json_text = json_text[:end_pos]
                    else:
                        # 닫는 중괄호를 찾지 못한 경우, 마지막 } 찾기
                        last_brace = json_text.rfind("}")
                        if last_brace != -1:
                            json_text = json_text[:last_brace + 1]
                
                # 최종 JSON 텍스트가 비어있지 않은지 확인
                if not json_text or not json_text.strip():
                    raise json.JSONDecodeError("JSON 텍스트가 비어있음", json_text, 0)
                
                # JSON 파싱 시도
                evaluation_result = json.loads(json_text)
                print(f"  ✅ JSON 파싱 성공")
                return evaluation_result
                
            except json.JSONDecodeError as e:
                # JSON 파싱 실패 시 복구 시도
                print(f"  ⚠️ JSON 파싱 실패: {e}")
                print(f"  🔧 JSON 복구 시도 중...")
                
                # 복구 시도: 불완전한 JSON 수정
                try:
                    if 'json_text' in locals() and json_text:
                        # 불완전한 문자열 필드 닫기
                        fixed_json = json_text
                        
                        # 마지막 불완전한 문자열 필드 찾아서 닫기
                        # "key": "value 형태로 끝나는 경우
                        import re
                        # 불완전한 문자열 필드 패턴 찾기
                        incomplete_string_pattern = r'"([^"]*)"\s*:\s*"([^"]*)$'
                        matches = list(re.finditer(incomplete_string_pattern, fixed_json, re.MULTILINE))
                        
                        if matches:
                            # 마지막 매치의 불완전한 문자열 닫기
                            last_match = matches[-1]
                            fixed_json = fixed_json[:last_match.end()] + '"'
                            
                            # 닫는 중괄호 추가
                            open_braces = fixed_json.count('{')
                            close_braces = fixed_json.count('}')
                            missing_braces = open_braces - close_braces
                            if missing_braces > 0:
                                fixed_json += '\n' + '    ' * (missing_braces - 1) + '}' * missing_braces
                            
                            # 다시 파싱 시도
                            evaluation_result = json.loads(fixed_json)
                            print(f"  ✅ JSON 복구 성공")
                            return evaluation_result
                except:
                    pass
                
                # 복구 실패 시 원본 텍스트 반환
                print(f"  📝 추출된 JSON 텍스트 (처음 1000자): {json_text[:1000] if 'json_text' in locals() else 'N/A'}")
                print(f"  📝 원본 응답 (처음 500자): {evaluation_text[:500]}")
                return {
                    "raw_evaluation": evaluation_text,
                    "error": f"JSON 파싱 실패: {str(e)}",
                    "extracted_json": json_text[:2000] if 'json_text' in locals() else ""
                }
                
        except Exception as e:
            print(f"❌ 평가 오류: {e}")
            return {
                "error": str(e)
            }
    
    def run_evaluation(self, questions: List[Dict[str, str]]) -> List[Dict]:
        """
        질문 리스트에 대해 평가 실행
        
        Args:
            questions: 질문 리스트, 각 항목은 {"question": "...", "ground_truth": "..."} 형식
            
        Returns:
            평가 결과 리스트
        """
        results = []
        
        print(f"📊 평가 시작: 총 {len(questions)}개 질문")
        print("=" * 60)
        
        for idx, q_data in enumerate(questions, 1):
            question = q_data.get("question", "")
            ground_truth = q_data.get("ground_truth", "")
            
            print(f"\n[{idx}/{len(questions)}] 질문: {question[:50]}...")
            
            # 1. 본 시스템 응답 생성
            print("  🔄 본 시스템 응답 생성 중...")
            system_a_response = self.get_our_system_response(question)
            print(f"  ✅ 본 시스템 응답 완료 ({len(system_a_response)}자)")
            
            # 2. GPT-5 응답 생성
            print("  🔄 GPT-5 응답 생성 중...")
            system_b_response = self.get_gpt5_response(question)
            print(f"  ✅ GPT-5 응답 완료 ({len(system_b_response)}자)")
            
            # 3. Gemini 3.0 Pro로 평가
            print("  🔄 Gemini 3.0 Pro 평가 중...")
            evaluation = self.evaluate_responses(
                question=question,
                ground_truth=ground_truth,
                system_a_response=system_a_response,
                system_b_response=system_b_response
            )
            print("  ✅ 평가 완료")
            
            # 결과 저장
            result = {
                "id": q_data.get("id"),  # 질문 ID 포함
                "question": question,
                "ground_truth": ground_truth,
                "system_a_response": system_a_response,
                "system_b_response": system_b_response,
                "evaluation": evaluation,
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            
            # 중간 결과 저장 (백업)
            self.save_results(results, f"evaluation_results_backup_{idx}.json")
            
            # 잠시 대기 (API rate limit 방지)
            time.sleep(1)
        
        print("\n" + "=" * 60)
        print("✅ 모든 평가 완료!")
        
        return results
    
    def save_results(self, results: List[Dict], filename: Optional[str] = None):
        """
        평가 결과를 JSON 파일로 저장
        
        Args:
            results: 평가 결과 리스트
            filename: 저장할 파일명 (None이면 자동 생성)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        filepath = os.path.join("evaluation_charts", filename)
        os.makedirs("evaluation_charts", exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 결과 저장: {filepath}")
    
    def generate_summary(self, results: List[Dict]) -> Dict:
        """
        평가 결과 요약 통계 생성
        
        Args:
            results: 평가 결과 리스트
            
        Returns:
            요약 통계 딕셔너리
        """
        if not results:
            return {}
        
        system_a_scores = {
            "accuracy": [],
            "relevance": [],
            "completeness": [],
            "usefulness": [],
            "overall": []
        }
        
        system_b_scores = {
            "accuracy": [],
            "relevance": [],
            "completeness": [],
            "usefulness": [],
            "overall": []
        }
        
        winners = {"system_a": 0, "system_b": 0, "tie": 0}
        
        for result in results:
            evaluation = result.get("evaluation", {})
            
            if "error" in evaluation:
                continue
            
            # System A 점수 수집
            if "system_a" in evaluation:
                for metric in system_a_scores.keys():
                    score = evaluation["system_a"].get(metric, 0)
                    if isinstance(score, (int, float)):
                        system_a_scores[metric].append(score)
            
            # System B 점수 수집
            if "system_b" in evaluation:
                for metric in system_b_scores.keys():
                    score = evaluation["system_b"].get(metric, 0)
                    if isinstance(score, (int, float)):
                        system_b_scores[metric].append(score)
            
            # 승자 집계
            if "comparison" in evaluation:
                winner = evaluation["comparison"].get("winner", "tie")
                winners[winner] = winners.get(winner, 0) + 1
        
        # 평균 계산
        def calculate_avg(scores):
            return sum(scores) / len(scores) if scores else 0
        
        summary = {
            "total_questions": len(results),
            "system_a_averages": {
                metric: calculate_avg(scores)
                for metric, scores in system_a_scores.items()
            },
            "system_b_averages": {
                metric: calculate_avg(scores)
                for metric, scores in system_b_scores.items()
            },
            "winners": winners
        }
        
        return summary


def load_ground_truth(filepath: str = "evaluation_charts/ground_truth.json") -> List[Dict]:
    """답안지 파일 로드"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✅ 답안지 로드 완료: {len(data)}개 질문")
        return data
    except FileNotFoundError:
        print(f"❌ 답안지 파일을 찾을 수 없습니다: {filepath}")
        return []
    except Exception as e:
        print(f"❌ 답안지 로드 오류: {e}")
        return []


def main():
    """평가 시스템 실행"""
    # 평가 시스템 초기화
    evaluator = EvaluationSystem()
    
    # 답안지 파일 로드
    ground_truth_data = load_ground_truth()
    
    if not ground_truth_data:
        print("❌ 답안지 파일을 로드할 수 없어 평가를 진행할 수 없습니다.")
        return
    
    # 질문 리스트 준비 (답안지 형식에 맞춰 변환)
    # 7, 9, 10번만 평가
    target_ids = [9, 10]
    questions = []
    for item in ground_truth_data:
        item_id = item.get("id")
        if item_id in target_ids:
            questions.append({
                "question": item.get("question", ""),
                "ground_truth": item.get("ground_truth", ""),
                "id": item_id
            })
    
    print(f"\n📊 추가 평가 시작: {target_ids}번 질문 ({len(questions)}개)")
    print("=" * 60)
    
    # 평가 실행
    results = evaluator.run_evaluation(questions)
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluator.save_results(results, f"evaluation_results_{timestamp}.json")
    
    # 요약 통계 생성
    summary = evaluator.generate_summary(results)
    print("\n📊 평가 요약:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    
    # 요약도 저장
    evaluator.save_results([summary], f"evaluation_summary_{timestamp}.json")


if __name__ == "__main__":
    main()

