from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
from typing import Dict, List
import uuid
from datetime import datetime
import base64

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

@app.get("/map", response_class=HTMLResponse)
async def get_map_page():
    """카카오 맵 페이지 HTML 반환"""
    with open("static/index.html", "r", encoding="utf-8") as f:
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

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """세션 삭제"""
    try:
        # 세션 존재 확인
        if not chat_manager.session_exists(session_id):
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        # 세션 삭제
        success = chat_manager.delete_session(session_id)
        if success:
            return {"message": "세션이 삭제되었습니다."}
        else:
            raise HTTPException(status_code=500, detail="세션 삭제에 실패했습니다.")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ocr")
async def process_image_ocr(image: UploadFile = File(...), query: str = ""):
    """이미지 OCR 처리 API"""
    try:
        # 이미지 파일 읽기
        image_data = await image.read()
        
        # 이미지 크기 검증 (5MB 제한)
        if len(image_data) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="이미지 크기는 5MB 이하여야 합니다.")
        
        # 이미지 타입 검증
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")
        
        # OCR API 호출
        
        # OCR 처리
        from ocr_node import extract_text_from_image, extract_medicine_name_from_text
        
        # 텍스트 추출
        extracted_text = extract_text_from_image(image_data)
        if not extracted_text:
            return {
                "success": False,
                "message": "이미지에서 텍스트를 추출할 수 없습니다.",
                "extracted_text": "",
                "medicine_name": ""
            }
        
        # 약품명 추출
        medicine_name = extract_medicine_name_from_text(extracted_text)
        
        return {
            "success": True,
            "message": "OCR 처리 완료",
            "extracted_text": extracted_text,
            "medicine_name": medicine_name,
            "filename": image.filename
        }
        
    except Exception as e:
        print(f"❌ OCR 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"OCR 처리 중 오류가 발생했습니다: {str(e)}")

@app.post("/api/pharmacy/search")
async def search_nearby_pharmacies(latitude: float, longitude: float, radius: int = 1000):
    """근처 약국 검색 API (카카오 주소 검색 API 활용)"""
    try:
        import requests
        
        # 카카오 주소 검색 API를 사용한 약국 검색
        kakao_api_key = "c6cd8abf935c72e801367bc8249c4f1f"  # 실제 API 키 사용
        url = "https://dapi.kakao.com/v2/local/search/category.json"
        
        headers = {
            "Authorization": f"KakaoAK {kakao_api_key}"
        }
        
        params = {
            "category_group_code": "PM9",  # 약국 카테고리 코드
            "x": longitude,
            "y": latitude,
            "radius": radius,
            "sort": "distance"  # 거리순 정렬
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            pharmacies = []
            
            for place in data.get("documents", [])[:5]:  # 상위 5개만 반환
                # 거리 계산 (간단한 하버사인 공식) - 타입 변환 추가
                distance = calculate_distance(
                    float(latitude), float(longitude),
                    float(place["y"]), float(place["x"])
                )
                
                pharmacy_info = {
                    "name": place["place_name"],
                    "address": place["address_name"],
                    "road_address": place.get("road_address_name", ""),
                    "phone": place.get("phone", ""),
                    "distance": round(distance * 1000, 1),  # km를 m로 변환
                    "latitude": float(place["y"]),
                    "longitude": float(place["x"]),
                    "place_url": place.get("place_url", "")
                }
                pharmacies.append(pharmacy_info)
            
            return {
                "success": True,
                "pharmacies": pharmacies,
                "total_count": len(pharmacies)
            }
        else:
            print(f"❌ 카카오 API 오류: {response.status_code}")
            return {
                "success": False,
                "message": "약국 검색 중 오류가 발생했습니다.",
                "pharmacies": []
            }
            
    except Exception as e:
        print(f"❌ 약국 검색 중 오류 발생: {e}")
        return {
            "success": False,
            "message": "약국 검색 중 오류가 발생했습니다.",
            "pharmacies": []
        }

def calculate_distance(lat1, lon1, lat2, lon2):
    """두 지점 간의 거리 계산 (km) - 타입 검증 추가"""
    import math
    
    # 타입 검증 및 변환
    try:
        lat1 = float(lat1)
        lon1 = float(lon1)
        lat2 = float(lat2)
        lon2 = float(lon2)
    except (ValueError, TypeError) as e:
        print(f"❌ 좌표 타입 변환 오류: {e}")
        return 0.0
    
    # 하버사인 공식
    R = 6371  # 지구 반지름 (km)
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat/2) * math.sin(dlat/2) + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon/2) * math.sin(dlon/2))
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, limit: int = 50):
    """특정 세션의 메시지 히스토리 반환"""
    try:
        # 세션 존재 확인
        if not chat_manager.session_exists(session_id):
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        # 특정 세션의 메시지 가져오기
        session = chat_manager.sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        # 메시지 형식으로 변환
        messages = []
        for msg in session.messages[-limit:]:  # 최근 N개 메시지만
            messages.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
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
        image_data = message_data.get("image_data")  # 이미지 데이터 추출
        user_location = message_data.get("user_location")  # 사용자 위치 정보 추출
        
        # 디버깅: 사용자 위치 정보 로그
        if user_location:
            print(f"📍 사용자 위치 정보 수신됨: {user_location}")
        else:
            print("⚠️ 사용자 위치 정보 없음")
        
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
                    
                    # LLM 맥락 분석 결과 처리
                    
                except json.JSONDecodeError as e:
                    # JSON 파싱 실패 시 기본값 사용
                    has_medicine_recommendation = False
                    is_asking_about_previous = False
                    found_medicines = []
                    
            except Exception as e:
                print(f"❌ 맥락 분석 중 오류 발생: {e}, 기본값 사용")
                has_medicine_recommendation = False
                is_asking_about_previous = False
                found_medicines = []
            
            # 이미지 데이터를 바이트로 변환
            image_bytes = None
            if image_data:
                try:
                    image_bytes = bytes(image_data)
        # 이미지 데이터 수신
                except Exception as e:
                    print(f"❌ 이미지 데이터 변환 오류: {e}")
            
            # 세션 정보를 state에 추가
            initial_state = QAState(
                query=user_message,
                session_id=session_id,
                conversation_context=full_context,
                user_context=chat_manager.get_user_context(),
                has_medicine_recommendation=has_medicine_recommendation,
                is_asking_about_previous=is_asking_about_previous,
                image_data=image_bytes,  # 이미지 데이터 추가
                user_location=user_location  # 사용자 위치 정보 추가
            )
            
            # 그래프 실행
            result = graph.invoke(initial_state)
            
            # 답변 추출
            ai_answer = result.get("final_answer", "죄송합니다. 답변을 생성할 수 없습니다.")
            
            # 근처 약국 정보 추가 (사용자 위치가 있고 의약품 관련 질문인 경우)
            if user_location and is_medicine_related_question(user_message):
                try:
                    # 근처 약국 검색
                    pharmacy_response = await search_nearby_pharmacies(
                        latitude=user_location["lat"],
                        longitude=user_location["lng"],
                        radius=1000
                    )
                    
                    if pharmacy_response["success"] and pharmacy_response["pharmacies"]:
                        # 약국 정보를 답변에 추가
                        ai_answer = add_pharmacy_info_to_answer(ai_answer, pharmacy_response["pharmacies"])
                        print(f"✅ 근처 약국 정보 추가됨: {len(pharmacy_response['pharmacies'])}개")
                    
                except Exception as e:
                    print(f"❌ 약국 정보 추가 중 오류: {e}")
            
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

def is_medicine_related_question(message: str) -> bool:
    """의약품 관련 질문인지 판단"""
    medicine_keywords = [
        "약", "약품", "약국", "처방", "복용", "부작용", "효능", "성분",
        "두통", "감기", "해열", "소화", "통증", "염증", "알레르기",
        "타이레놀", "아스피린", "이부프로펜", "감기약", "두통약"
    ]
    
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in medicine_keywords)

def add_pharmacy_info_to_answer(answer: str, pharmacies: list) -> str:
    """답변에 약국 정보 추가"""
    if not pharmacies:
        return answer
    
    pharmacy_info = "\n\n🏥 **근처 약국 정보:**\n"
    
    for i, pharmacy in enumerate(pharmacies[:3], 1):  # 상위 3개만 표시
        pharmacy_info += f"{i}. **{pharmacy['name']}**\n"
        pharmacy_info += f"   📍 {pharmacy['road_address'] or pharmacy['address']}\n"
        if pharmacy['phone']:
            pharmacy_info += f"   📞 {pharmacy['phone']}\n"
        pharmacy_info += f"   📏 거리: {pharmacy['distance']}m\n\n"
    
    pharmacy_info += "💡 **참고:** 위 약국들은 현재 위치 기준으로 가장 가까운 곳들입니다. 정확한 약품 구매 가능 여부는 약국에 직접 문의하시기 바랍니다."
    
    return answer + pharmacy_info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)
