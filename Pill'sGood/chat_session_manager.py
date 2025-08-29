from typing import List, Dict, Optional
from datetime import datetime
import json
import os

class ChatMessage:
    """대화 메시지를 나타내는 클래스"""
    def __init__(self, role: str, content: str, timestamp: Optional[datetime] = None):
        self.role = role  # "user" 또는 "assistant"
        self.content = content
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatMessage':
        timestamp = datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now()
        return cls(data["role"], data["content"], timestamp)

class ChatSession:
    """단일 대화 세션을 관리하는 클래스"""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[ChatMessage] = []
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
    
    def add_message(self, role: str, content: str):
        """새 메시지 추가"""
        message = ChatMessage(role, content)
        self.messages.append(message)
        self.last_updated = datetime.now()
    
    def get_conversation_history(self, max_messages: int = 10) -> str:
        """대화 히스토리를 문자열로 반환 (최근 N개 메시지)"""
        if not self.messages:
            return ""
        
        recent_messages = self.messages[-max_messages:]
        history = []
        
        for msg in recent_messages:
            role_text = "사용자" if msg.role == "user" else "의사"
            history.append(f"{role_text}: {msg.content}")
        
        return "\n".join(history)
    
    def get_user_context(self, max_messages: int = 5) -> str:
        """사용자 질문 맥락을 추출하여 반환"""
        user_messages = [msg for msg in self.messages if msg.role == "user"]
        if not user_messages:
            return ""
        
        recent_user_messages = user_messages[-max_messages:]
        context = []
        
        for msg in recent_user_messages:
            context.append(msg.content)
        
        return " | ".join(context)
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatSession':
        session = cls(data["session_id"])
        session.messages = [ChatMessage.from_dict(msg_data) for msg_data in data.get("messages", [])]
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.last_updated = datetime.fromisoformat(data["last_updated"])
        return session

class ChatSessionManager:
    """전체 대화 세션을 관리하는 클래스"""
    def __init__(self, storage_dir: str = "chat_sessions"):
        self.storage_dir = storage_dir
        self.sessions: Dict[str, ChatSession] = {}
        self.current_session_id: Optional[str] = None
        
        # 저장 디렉토리 생성
        os.makedirs(storage_dir, exist_ok=True)
        
        # 기존 세션 로드
        self._load_sessions()
    
    def create_new_session(self) -> str:
        """새로운 대화 세션 생성"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = ChatSession(session_id)
        self.sessions[session_id] = session
        self.current_session_id = session_id
        return session_id
    
    def get_current_session(self) -> Optional[ChatSession]:
        """현재 활성 세션 반환"""
        if self.current_session_id and self.current_session_id in self.sessions:
            return self.sessions[self.current_session_id]
        return None
    
    def add_user_message(self, content: str):
        """현재 세션에 사용자 메시지 추가"""
        session = self.get_current_session()
        if session:
            session.add_message("user", content)
    
    def add_assistant_message(self, content: str):
        """현재 세션에 어시스턴트 메시지 추가"""
        session = self.get_current_session()
        if session:
            session.add_message("assistant", content)
    
    def get_conversation_context(self, max_messages: int = 10) -> str:
        """현재 세션의 대화 맥락 반환"""
        session = self.get_current_session()
        if session:
            return session.get_conversation_history(max_messages)
        return ""
    
    def get_user_context(self, max_messages: int = 5) -> str:
        """사용자 질문 맥락 반환"""
        session = self.get_current_session()
        if session:
            return session.get_user_context(max_messages)
        return ""
    
    def switch_session(self, session_id: str) -> bool:
        """다른 세션으로 전환"""
        if session_id in self.sessions:
            self.current_session_id = session_id
            return True
        return False
    
    def list_sessions(self) -> List[Dict]:
        """모든 세션 목록 반환"""
        session_list = []
        for session_id, session in self.sessions.items():
            session_list.append({
                "session_id": session_id,
                "message_count": len(session.messages),
                "created_at": session.created_at,
                "last_updated": session.last_updated,
                "is_current": session_id == self.current_session_id
            })
        return session_list
    
    def save_session(self, session_id: str):
        """특정 세션을 파일로 저장"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            file_path = os.path.join(self.storage_dir, f"{session_id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
    
    def save_all_sessions(self):
        """모든 세션을 파일로 저장"""
        for session_id in self.sessions:
            self.save_session(session_id)
    
    def _load_sessions(self):
        """저장된 세션들을 로드"""
        if not os.path.exists(self.storage_dir):
            return
        
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.storage_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    session = ChatSession.from_dict(session_data)
                    self.sessions[session.session_id] = session
                except Exception as e:
                    print(f"세션 로드 실패: {filename}, 오류: {e}")
    
    def delete_session(self, session_id: str) -> bool:
        """세션 삭제"""
        if session_id in self.sessions:
            # 파일도 삭제
            file_path = os.path.join(self.storage_dir, f"{session_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # 메모리에서도 삭제
            del self.sessions[session_id]
            
            # 현재 세션이 삭제된 경우 새로운 세션 생성
            if self.current_session_id == session_id:
                if self.sessions:
                    self.current_session_id = list(self.sessions.keys())[0]
                else:
                    self.current_session_id = None
            
            return True
        return False
