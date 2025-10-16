import os
import sys
import re
from typing import Optional
from chat_session_manager import ChatSessionManager
from main_graph import builder
from qa_state import QAState
from answer_utils import generate_response_llm_from_prompt
import json

class ChatInterface:
    """실시간 대화 인터페이스"""
    
    def __init__(self):
        self.session_manager = ChatSessionManager()
        self.graph = builder.compile()
        self.current_session_id = None
        
        # 기존 세션이 있으면 가장 최근 세션을 로드, 없으면 새 세션 시작
        self.load_or_create_session()
    
    def load_or_create_session(self):
        """기존 세션이 있으면 가장 최근 세션을 로드, 없으면 새 세션 시작"""
        sessions = self.session_manager.list_sessions()
        
        if sessions:
            # 가장 최근에 업데이트된 세션을 찾기
            latest_session = max(sessions, key=lambda x: x["last_updated"])
            session_id = latest_session["session_id"]
            
            # 해당 세션으로 전환
            if self.session_manager.switch_session(session_id):
                self.current_session_id = session_id
                print(f"\n🔄 이전 세션을 복구했습니다. (세션 ID: {session_id})")
                print(f"📚 이전 대화 내용: {latest_session['message_count']}개 메시지")
                self.show_conversation_history()
                print("💬 계속해서 대화를 이어가세요!")
                print("📝 명령어: /help (도움말), /new (새 세션), /sessions (세션 목록), /quit (종료)")
                print("-" * 60)
            else:
                self.start_new_session()
        else:
            # 저장된 세션이 없으면 새 세션 시작
            self.start_new_session()
    
    def start_new_session(self):
        """새로운 대화 세션 시작"""
        self.current_session_id = self.session_manager.create_new_session()
        print(f"\n🆕 새로운 대화 세션이 시작되었습니다. (세션 ID: {self.current_session_id})")
        print("💬 의약품에 대한 질문을 자유롭게 해주세요!")
        print("📝 명령어: /help (도움말), /new (새 세션), /sessions (세션 목록), /quit (종료)")
        print("-" * 60)
    
    def switch_session(self, session_id: str):
        """다른 세션으로 전환"""
        if self.session_manager.switch_session(session_id):
            self.current_session_id = session_id
            print(f"\n🔄 세션을 전환했습니다: {session_id}")
            self.show_conversation_history()
        else:
            print(f"❌ 세션을 찾을 수 없습니다: {session_id}")
    
    def show_conversation_history(self, max_messages: int = 10):
        """현재 세션의 대화 히스토리 표시"""
        history = self.session_manager.get_conversation_context(max_messages)
        if history:
            print(f"\n📚 대화 히스토리 (최근 {max_messages}개):")
            print("-" * 40)
            print(history)
            print("-" * 40)
        else:
            print("\n📚 아직 대화 기록이 없습니다.")
    
    def list_sessions(self):
        """모든 세션 목록 표시"""
        sessions = self.session_manager.list_sessions()
        if not sessions:
            print("\n📋 저장된 세션이 없습니다.")
            return
        
        print(f"\n📋 저장된 세션 목록 ({len(sessions)}개):")
        print("-" * 60)
        for session in sessions:
            status = "🟢 현재" if session["is_current"] else "⚪"
            print(f"{status} {session['session_id']}")
            print(f"    메시지: {session['message_count']}개")
            print(f"    생성: {session['created_at'].strftime('%Y-%m-%d %H:%M')}")
            print(f"    최근: {session['last_updated'].strftime('%Y-%m-%d %H:%M')}")
            print()
    
    def show_help(self):
        """도움말 표시"""
        help_text = """
📖 도움말

💬 대화 명령어:
    /help      - 이 도움말을 표시합니다
    /new       - 새로운 대화 세션을 시작합니다
    /sessions  - 저장된 세션 목록을 표시합니다
    /switch    - 다른 세션으로 전환합니다
    /history   - 현재 세션의 대화 히스토리를 표시합니다
    /quit      - 프로그램을 종료합니다

💊 의약품 질문 예시:
    - "두통약의 부작용이 궁금해요"
    - "감기약 추천해주세요"
    - "혈압약 복용 중인데 주의사항이 있나요?"
    - "2024년 새로 나온 당뇨약이 있나요?"

🔄 대화 맥락:
    - 이전 대화 내용을 기억하여 연속된 질문에 답변합니다
    - "그 약의 효능은?" 같은 추상적인 질문도 이전 맥락을 참고하여 답변합니다
        """
        print(help_text)
    
    def process_query(self, query: str) -> str:
        """사용자 질문을 처리하고 답변 생성"""
        try:
            # 전체 대화 맥락을 가져오기 (더 많은 메시지 포함)
            current_context = self.session_manager.get_conversation_context(max_messages=20)
            
            # 현재 질문이 이전 대화 맥락에 포함되어 있는지 확인
            if query not in current_context:
                # 이전 대화 맥락에 현재 질문 추가
                full_context = f"{current_context}\n사용자: {query}" if current_context else f"사용자: {query}"
            else:
                full_context = current_context
            
            print(f"🔍 대화 맥락 분석:")
            print(f"  - 전체 맥락 길이: {len(full_context)} 문자")
            
            # LLM 기반 맥락 분석
            context_analysis_prompt = f"""
당신은 대화 맥락 분석 전문가입니다.
다음 대화 맥락을 분석하여 사용자의 의도를 파악해주세요.

**대화 맥락:**
{full_context[:1000] if full_context else "없음"}

**분석 요구사항:**
1. 이전 대화에서 약품 추천이 있었는지
2. 현재 질문이 이전 대화 내용을 참조하는지
3. 대화 맥락에서 발견된 주요 약품 정보

**중요: 코드 블록 없이 순수 JSON만 반환하세요!**

출력 형식:
{{
    "has_medicine_recommendation": true/false,
    "is_asking_about_previous": true/false,
    "found_medicines": ["약품1", "약품2"],
    "reasoning": "분석 근거"
}}
"""
            
            try:
                response = generate_response_llm_from_prompt(
                    prompt=context_analysis_prompt,
                    temperature=0.1,
                    max_tokens=400
                )
                
                # JSON 코드 블록 제거 (```json ... ``` 형태 처리)
                cleaned_response = response.strip()
                if cleaned_response.startswith('```'):
                    # 첫 번째 줄 제거 (```json)
                    lines = cleaned_response.split('\n')
                    if lines[0].startswith('```'):
                        lines = lines[1:]
                    # 마지막 줄 제거 (```)
                    if lines and lines[-1].strip() == '```':
                        lines = lines[:-1]
                    cleaned_response = '\n'.join(lines).strip()
                
                # JSON 응답 파싱
                try:
                    analysis_result = json.loads(cleaned_response)
                    has_medicine_recommendation = analysis_result.get("has_medicine_recommendation", False)
                    is_asking_about_previous = analysis_result.get("is_asking_about_previous", False)
                    found_medicines = analysis_result.get("found_medicines", [])
                    reasoning = analysis_result.get("reasoning", "")
                    
                    print(f"🧠 LLM 맥락 분석 결과:")
                    print(f"  - 약품 추천 포함: {has_medicine_recommendation}")
                    print(f"  - 이전 대화 참조: {is_asking_about_previous}")
                    print(f"  - 발견된 약품: {found_medicines[:3] if found_medicines else '없음'}")
                    print(f"  - 분석 근거: {reasoning[:100] if reasoning else '없음'}...")
                    
                except json.JSONDecodeError as e:
                    print(f"⚠️ 맥락 분석 결과를 JSON으로 파싱할 수 없음: {e}")
                    print(f"🔍 원본 응답 (처음 200자): {response[:200]}...")
                    print(f"🔍 정리된 응답 (처음 200자): {cleaned_response[:200]}...")
                    has_medicine_recommendation = False
                    is_asking_about_previous = False
                    found_medicines = []
                    
            except Exception as e:
                print(f"❌ 맥락 분석 중 오류 발생: {e}, 기본값 사용")
                has_medicine_recommendation = False
                is_asking_about_previous = False
                found_medicines = []
            
            # 세션 정보를 state에 추가
            initial_state = QAState(
                query=query,
                session_id=self.current_session_id,
                conversation_context=full_context,
                user_context=self.session_manager.get_user_context(),
                has_medicine_recommendation=has_medicine_recommendation,
                is_asking_about_previous=is_asking_about_previous
            )
            
            # 그래프 실행
            result = self.graph.invoke(initial_state)
            
            # 답변 추출
            answer = result.get("final_answer", "죄송합니다. 답변을 생성할 수 없습니다.")
            
            # 세션에 메시지 추가
            self.session_manager.add_user_message(query)
            self.session_manager.add_assistant_message(answer)
            
            # 세션 저장
            self.session_manager.save_session(self.current_session_id)
            
            return answer
            
        except Exception as e:
            error_msg = f"오류가 발생했습니다: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def run(self):
        """대화 인터페이스 실행"""
        print("🏥 TeamMediChat - 의약품 상담 시스템")
        print("=" * 60)
        
        while True:
            try:
                # 사용자 입력 받기
                user_input = input("\n💬 질문을 입력하세요: ").strip()
                
                # 빈 입력 처리
                if not user_input:
                    continue
                
                # 명령어 처리
                if user_input.startswith('/'):
                    self.handle_command(user_input)
                    continue
                
                # 일반 질문 처리
                print("\n🤔 질문을 분석하고 있습니다...")
                answer = self.process_query(user_input)
                
                print(f"\n💊 답변:")
                print("-" * 40)
                print(answer)
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\n\n👋 프로그램을 종료합니다.")
                break
            except EOFError:
                print("\n\n👋 프로그램을 종료합니다.")
                break
    
    def handle_command(self, command: str):
        """명령어 처리"""
        cmd = command.lower().strip()
        
        if cmd == '/help':
            self.show_help()
        elif cmd == '/new':
            self.start_new_session()
        elif cmd == '/sessions':
            self.list_sessions()
        elif cmd == '/history':
            self.show_conversation_history()
        elif cmd == '/quit':
            print("\n👋 프로그램을 종료합니다.")
            self.session_manager.save_all_sessions()
            sys.exit(0)
        elif cmd.startswith('/switch'):
            # /switch session_id 형식
            parts = command.split()
            if len(parts) == 2:
                session_id = parts[1]
                self.switch_session(session_id)
            else:
                print("❌ 사용법: /switch <세션ID>")
        else:
            print(f"❌ 알 수 없는 명령어입니다: {command}")
            print("💡 /help를 입력하여 사용 가능한 명령어를 확인하세요.")

def main():
    """메인 함수"""
    try:
        chat_interface = ChatInterface()
        chat_interface.run()
    except Exception as e:
        print(f"❌ 프로그램 실행 중 오류가 발생했습니다: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
