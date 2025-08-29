from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
from typing import Dict, List
import uuid
from datetime import datetime

# 기존 시스템 import
from main_graph import graph
from qa_state import QAState
from chat_session_manager import ChatSessionManager
from answer_utils import generate_response_llm_from_prompt
import re

app = FastAPI(title="TeamMediChat API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# WebSocket 연결 관리
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, List[str]] = {}  # session_id -> connection_ids
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = websocket
        
        if session_id not in self.session_connections:
            self.session_connections[session_id] = []
        self.session_connections[session_id].append(connection_id)
        
        return connection_id
    
    def disconnect(self, connection_id: str, session_id: str):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if session_id in self.session_connections:
            if connection_id in self.session_connections[session_id]:
                self.session_connections[session_id].remove(connection_id)
    
    async def send_personal_message(self, message: dict, connection_id: str):
        if connection_id in self.active_connections:
            await self.active_connections[connection_id].send_text(json.dumps(message, ensure_ascii=False))
    
    async def broadcast_to_session(self, message: dict, session_id: str):
        if session_id in self.session_connections:
            for connection_id in self.session_connections[session_id]:
                await self.send_personal_message(message, connection_id)

manager = ConnectionManager()
chat_manager = ChatSessionManager()

@app.get("/", response_class=HTMLResponse)
async def get_chat_page():
    """채팅 페이지 HTML 반환"""
    with open("static/chat.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/sessions")
async def get_sessions():
    """저장된 세션 목록 반환"""
    try:
        sessions = chat_manager.list_sessions()
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sessions")
async def create_session():
    """새 세션 생성"""
    try:
        session_id = chat_manager.create_new_session()
        return {"session_id": session_id, "message": "새 세션이 생성되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, limit: int = 50):
    """특정 세션의 메시지 히스토리 반환"""
    try:
        # 세션 존재 확인
        if not chat_manager.session_exists(session_id):
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        # 대화 맥락 가져오기
        context = chat_manager.get_conversation_context(max_messages=limit)
        
        # 메시지 형식으로 변환
        messages = []
        if context:
            lines = context.strip().split('\n')
            current_role = None
            current_content = []
            
            for line in lines:
                if line.startswith('사용자: '):
                    if current_role and current_content:
                        messages.append({
                            "role": current_role,
                            "content": '\n'.join(current_content).strip(),
                            "timestamp": datetime.now().isoformat()
                        })
                    current_role = "user"
                    current_content = [line[4:]]  # "사용자: " 제거
                elif line.startswith('AI: '):
                    if current_role and current_content:
                        messages.append({
                            "role": current_role,
                            "content": '\n'.join(current_content).strip(),
                            "timestamp": datetime.now().isoformat()
                        })
                    current_role = "assistant"
                    current_content = [line[4:]]  # "AI: " 제거
                else:
                    if current_content:
                        current_content.append(line)
            
            # 마지막 메시지 추가
            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": '\n'.join(current_content).strip(),
                    "timestamp": datetime.now().isoformat()
                })
        
        return {"messages": messages, "session_id": session_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket 연결 처리"""
    connection_id = None
    try:
        # 연결 수락
        connection_id = await manager.connect(websocket, session_id)
        
        # 연결 성공 메시지 전송
        await manager.send_personal_message({
            "type": "connection_established",
            "session_id": session_id,
            "connection_id": connection_id,
            "message": "채팅 서버에 연결되었습니다."
        }, connection_id)
        
        # 기존 메시지 히스토리 전송
        try:
            context = chat_manager.get_conversation_context(max_messages=50)
            if context:
                await manager.send_personal_message({
                    "type": "chat_history",
                    "session_id": session_id,
                    "history": context
                }, connection_id)
        except Exception as e:
            print(f"히스토리 로드 오류: {e}")
        
        # 메시지 수신 대기
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data["type"] == "chat_message":
                await handle_chat_message(websocket, session_id, message_data)
            elif message_data["type"] == "typing_start":
                await manager.broadcast_to_session({
                    "type": "user_typing",
                    "session_id": session_id
                }, session_id)
            elif message_data["type"] == "typing_stop":
                await manager.broadcast_to_session({
                    "type": "user_typing_stop",
                    "session_id": session_id
                }, session_id)
                
    except WebSocketDisconnect:
        if connection_id:
            manager.disconnect(connection_id, session_id)
    except Exception as e:
        print(f"WebSocket 오류: {e}")
        if connection_id:
            manager.disconnect(connection_id, session_id)

async def handle_chat_message(websocket: WebSocket, session_id: str, message_data: dict):
    """채팅 메시지 처리"""
    try:
        user_message = message_data["content"]
        
        # 사용자 메시지 브로드캐스트
        await manager.broadcast_to_session({
            "type": "chat_message",
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id
        }, session_id)
        
        # AI 답변 생성
        try:
            # 전체 대화 맥락을 가져오기
            current_context = chat_manager.get_conversation_context(max_messages=20)
            
            # 현재 질문이 이전 대화 맥락에 포함되어 있는지 확인
            if user_message not in current_context:
                full_context = f"{current_context}\n사용자: {user_message}" if current_context else f"사용자: {user_message}"
            else:
                full_context = current_context
            
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

JSON 형식으로 응답해주세요:
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
                
                # JSON 응답 파싱
                try:
                    analysis_result = json.loads(response)
                    has_medicine_recommendation = analysis_result.get("has_medicine_recommendation", False)
                    is_asking_about_previous = analysis_result.get("is_asking_about_previous", False)
                    found_medicines = analysis_result.get("found_medicines", [])
                    reasoning = analysis_result.get("reasoning", "")
                    
                    print(f"🧠 LLM 맥락 분석 결과:")
                    print(f"  - 약품 추천 포함: {has_medicine_recommendation}")
                    print(f"  - 이전 대화 참조: {is_asking_about_previous}")
                    print(f"  - 발견된 약품: {found_medicines[:3] if found_medicines else '없음'}")
                    print(f"  - 분석 근거: {reasoning[:100]}...")
                    
                except json.JSONDecodeError:
                    print("⚠️ 맥락 분석 결과를 JSON으로 파싱할 수 없음, 기본값 사용")
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
                query=user_message,
                session_id=session_id,
                conversation_context=full_context,
                user_context=chat_manager.get_user_context(),
                has_medicine_recommendation=has_medicine_recommendation,
                is_asking_about_previous=is_asking_about_previous
            )
            
            # 그래프 실행
            result = graph.invoke(initial_state)
            
            # 답변 추출
            ai_answer = result.get("final_answer", "죄송합니다. 답변을 생성할 수 없습니다.")
            
            # 세션에 메시지 추가
            chat_manager.add_user_message(user_message)
            chat_manager.add_assistant_message(ai_answer)
            
            # 세션 저장
            chat_manager.save_session(session_id)
            
            # AI 답변 브로드캐스트
            await manager.broadcast_to_session({
                "type": "chat_message",
                "role": "assistant",
                "content": ai_answer,
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id
            }, session_id)
            
        except Exception as e:
            error_message = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
            await manager.broadcast_to_session({
                "type": "error",
                "message": error_message,
                "session_id": session_id
            }, session_id)
            
    except Exception as e:
        print(f"메시지 처리 오류: {e}")
        await manager.broadcast_to_session({
            "type": "error",
            "message": "메시지 처리 중 오류가 발생했습니다.",
            "session_id": session_id
        }, session_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
