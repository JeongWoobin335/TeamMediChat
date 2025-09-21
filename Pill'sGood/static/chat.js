// 전역 변수
let currentSessionId = null;
let websocket = null;
let isConnected = false;
let typingTimer = null;

// DOM 요소들
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendButton = document.getElementById('sendButton');
const charCount = document.getElementById('charCount');
const typingIndicator = document.getElementById('typingIndicator');
const loadingOverlay = document.getElementById('loadingOverlay');
const errorModal = document.getElementById('errorModal');
const errorMessage = document.getElementById('errorMessage');
const errorCloseBtn = document.getElementById('errorCloseBtn');
const sidebarToggle = document.getElementById('sidebarToggle');
const sidebarToggleMobile = document.getElementById('sidebarToggleMobile');
const sidebar = document.getElementById('sidebar');
const newSessionBtn = document.getElementById('newSessionBtn');
const sessionsList = document.getElementById('sessionsList');
const sessionInfo = document.getElementById('sessionInfo');

// 이미지 업로드 관련 요소들
const imageUploadBtn = document.getElementById('imageUploadBtn');
const imageInput = document.getElementById('imageInput');
const imageUploadSection = document.getElementById('imageUploadSection');
const imagePreview = document.getElementById('imagePreview');
const previewImage = document.getElementById('previewImage');
const removeImageBtn = document.getElementById('removeImageBtn');

// 전역 변수
let currentImageData = null;

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    loadSessions();
});

// 앱 초기화
function initializeApp() {
    // 새 세션 생성
    createNewSession();
    
    // 입력 필드 자동 크기 조정
    autoResizeTextarea();
    
    // 모바일 사이드바 토글
    if (window.innerWidth <= 768) {
        sidebar.classList.remove('show');
    }
}

// 이벤트 리스너 설정
function setupEventListeners() {
    // 메시지 전송
    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keydown', handleKeyDown);
    chatInput.addEventListener('input', handleInput);
    
    // 이미지 업로드
    imageUploadBtn.addEventListener('click', () => imageInput.click());
    imageInput.addEventListener('change', handleImageUpload);
    removeImageBtn.addEventListener('click', removeImage);
    
    // 사이드바 토글
    sidebarToggle.addEventListener('click', toggleSidebar);
    sidebarToggleMobile.addEventListener('click', toggleSidebar);
    
    // 새 세션
    newSessionBtn.addEventListener('click', createNewSession);
    
    // 오류 모달
    errorCloseBtn.addEventListener('click', hideErrorModal);
    
    // 윈도우 리사이즈
    window.addEventListener('resize', handleResize);
}

// WebSocket 연결
function connectWebSocket(sessionId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${sessionId}`;
    
    try {
        websocket = new WebSocket(wsUrl);
        
        websocket.onopen = function(event) {
            console.log('WebSocket 연결됨');
            isConnected = true;
            updateConnectionStatus(true);
        };
        
        websocket.onmessage = function(event) {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        };
        
        websocket.onclose = function(event) {
            console.log('WebSocket 연결 끊어짐');
            isConnected = false;
            updateConnectionStatus(false);
            
            // 자동 재연결 시도
            if (event.code !== 1000) {
                setTimeout(() => {
                    if (currentSessionId) {
                        connectWebSocket(currentSessionId);
                    }
                }, 3000);
            }
        };
        
        websocket.onerror = function(error) {
            console.error('WebSocket 오류:', error);
            showError('WebSocket 연결 오류가 발생했습니다.');
        };
        
    } catch (error) {
        console.error('WebSocket 연결 실패:', error);
        showError('WebSocket 연결을 설정할 수 없습니다.');
    }
}

// WebSocket 메시지 처리
function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'connection_established':
            console.log('연결 성공:', data.message);
            break;
            
        case 'chat_message':
            displayMessage(data.role, data.content, data.timestamp);
            // AI 답변을 받은 후 로딩 화면 숨기기
            if (data.role === 'assistant') {
                hideLoading();
            }
            break;
            
        case 'chat_history':
            displayChatHistory(data.history);
            break;
            
        case 'user_typing':
            showTypingIndicator();
            break;
            
        case 'user_typing_stop':
            hideTypingIndicator();
            break;
            
        case 'error':
            showError(data.message);
            break;
            
        default:
            console.log('알 수 없는 메시지 타입:', data.type);
    }
}

// 채팅 히스토리 표시
function displayChatHistory(history) {
    if (!history) return;
    
    // 기존 메시지들 제거 (환영 메시지 제외)
    const welcomeMessage = chatMessages.querySelector('.assistant-message');
    chatMessages.innerHTML = '';
    if (welcomeMessage) {
        chatMessages.appendChild(welcomeMessage);
    }
    
    // 히스토리 파싱 및 표시
    const lines = history.trim().split('\n');
    let currentRole = null;
    let currentContent = [];
    
    for (const line of lines) {
        if (line.startsWith('사용자: ')) {
            if (currentRole && currentContent.length > 0) {
                displayMessage(currentRole, currentContent.join('\n'), new Date().toISOString());
            }
            currentRole = 'user';
            currentContent = [line.substring(4)];
        } else if (line.startsWith('AI: ')) {
            if (currentRole && currentContent.length > 0) {
                displayMessage(currentRole, currentContent.join('\n'), new Date().toISOString());
            }
            currentRole = 'assistant';
            currentContent = [line.substring(4)];
        } else {
            if (currentContent.length > 0) {
                currentContent.push(line);
            }
        }
    }
    
    // 마지막 메시지 추가
    if (currentRole && currentContent.length > 0) {
        displayMessage(currentRole, currentContent.join('\n'), new Date().toISOString());
    }
    
    scrollToBottom();
}

// 메시지 표시
function displayMessage(role, content, timestamp) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    
    if (role === 'user') {
        avatar.innerHTML = '<i class="fas fa-user"></i>';
    } else {
        avatar.innerHTML = '<i class="fas fa-hospital"></i>';
    }
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    const messageText = document.createElement('div');
    messageText.className = 'message-text';
    messageText.textContent = content;
    
    const messageTimestamp = document.createElement('div');
    messageTimestamp.className = 'message-timestamp';
    messageTimestamp.textContent = formatTimestamp(timestamp);
    
    messageContent.appendChild(messageText);
    messageContent.appendChild(messageTimestamp);
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// 메시지 전송
function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || !isConnected) return;
    
    // 입력 필드 비우기
    chatInput.value = '';
    updateCharCount();
    autoResizeTextarea();
    
    // 전송 버튼 비활성화
    sendButton.disabled = true;
    
    // 로딩 표시
    showLoading();
    
    // WebSocket으로 메시지 전송
    const messageData = {
        type: 'chat_message',
        content: message,
        image_data: currentImageData ? Array.from(currentImageData) : null  // 이미지 데이터 포함
    };
    
    websocket.send(JSON.stringify(messageData));
    
    // 사용자 메시지 즉시 표시 (이미지 포함)
    displayMessageWithImage('user', message, new Date().toISOString(), currentImageData);
    
    // 이미지 초기화
    removeImage();
    
    // 타이핑 표시 숨기기
    hideTypingIndicator();
}

// 키보드 이벤트 처리
function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        if (!sendButton.disabled) {
            sendMessage();
        }
    }
}

// 입력 이벤트 처리
function handleInput() {
    updateCharCount();
    autoResizeTextarea();
    
    // 타이핑 표시
    if (isConnected) {
        clearTimeout(typingTimer);
        showTypingIndicator();
        
        typingTimer = setTimeout(() => {
            hideTypingIndicator();
        }, 1000);
    }
    
    // 전송 버튼 활성화/비활성화
    sendButton.disabled = chatInput.value.trim().length === 0;
}

// 문자 수 업데이트
function updateCharCount() {
    const count = chatInput.value.length;
    charCount.textContent = `${count}/2000`;
    
    if (count > 1800) {
        charCount.style.color = '#dc3545';
    } else if (count > 1500) {
        charCount.style.color = '#ffc107';
    } else {
        charCount.style.color = '#6c757d';
    }
}

// 텍스트 영역 자동 크기 조정
function autoResizeTextarea() {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
}

// 타이핑 표시
function showTypingIndicator() {
    if (isConnected) {
        typingIndicator.style.display = 'flex';
        websocket.send(JSON.stringify({ type: 'typing_start' }));
    }
}

// 타이핑 표시 숨기기
function hideTypingIndicator() {
    typingIndicator.style.display = 'none';
    if (isConnected) {
        websocket.send(JSON.stringify({ type: 'typing_stop' }));
    }
}

// 로딩 표시
function showLoading() {
    loadingOverlay.classList.add('show');
    
    // 30초 후 자동으로 로딩 숨기기 (안전장치)
    setTimeout(() => {
        if (loadingOverlay.classList.contains('show')) {
            console.log('로딩 타임아웃 - 자동으로 숨김');
            hideLoading();
        }
    }, 30000);
}

// 로딩 숨기기
function hideLoading() {
    loadingOverlay.classList.remove('show');
}

// 오류 표시
function showError(message) {
    errorMessage.textContent = message;
    errorModal.classList.add('show');
    hideLoading();
}

// 오류 모달 숨기기
function hideErrorModal() {
    errorModal.classList.remove('show');
}

// 사이드바 토글
function toggleSidebar() {
    sidebar.classList.toggle('show');
}

// 새 세션 생성
async function createNewSession() {
    try {
        const response = await fetch('/api/sessions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            currentSessionId = data.session_id;
            
            // WebSocket 연결
            if (websocket) {
                websocket.close();
            }
            connectWebSocket(currentSessionId);
            
            // 채팅 영역 초기화
            clearChatMessages();
            
            // 세션 목록 새로고침
            loadSessions();
            
            console.log('새 세션 생성됨:', currentSessionId);
        } else {
            throw new Error('세션 생성 실패');
        }
    } catch (error) {
        console.error('세션 생성 오류:', error);
        showError('새 세션을 생성할 수 없습니다.');
    }
}

// 세션 목록 로드
async function loadSessions() {
    try {
        const response = await fetch('/api/sessions');
        if (response.ok) {
            const data = await response.json();
            displaySessions(data.sessions);
        }
    } catch (error) {
        console.error('세션 목록 로드 오류:', error);
    }
}

// 세션 목록 표시
function displaySessions(sessions) {
    sessionsList.innerHTML = '';
    
    if (sessions.length === 0) {
        sessionsList.innerHTML = '<p style="color: #6c757d; text-align: center; padding: 20px;">저장된 세션이 없습니다.</p>';
        return;
    }
    
    sessions.forEach(session => {
        const sessionItem = document.createElement('div');
        sessionItem.className = 'session-item';
        if (session.session_id === currentSessionId) {
            sessionItem.classList.add('active');
        }
        
        const sessionTitle = document.createElement('div');
        sessionTitle.className = 'session-title';
        sessionTitle.textContent = `세션 ${session.session_id.substring(0, 8)}...`;
        
        const sessionMeta = document.createElement('div');
        sessionMeta.className = 'session-meta';
        sessionMeta.textContent = `${session.message_count}개 메시지`;
        
        sessionItem.appendChild(sessionTitle);
        sessionItem.appendChild(sessionMeta);
        
        sessionItem.addEventListener('click', () => {
            switchSession(session.session_id);
        });
        
        sessionsList.appendChild(sessionItem);
    });
}

// 세션 전환
function switchSession(sessionId) {
    if (sessionId === currentSessionId) return;
    
    currentSessionId = sessionId;
    
    // WebSocket 재연결
    if (websocket) {
        websocket.close();
    }
    connectWebSocket(currentSessionId);
    
    // 채팅 영역 초기화
    clearChatMessages();
    
    // 세션 목록 새로고침
    loadSessions();
    
    // 모바일에서 사이드바 닫기
    if (window.innerWidth <= 768) {
        sidebar.classList.remove('show');
    }
}

// 채팅 메시지 초기화
function clearChatMessages() {
    chatMessages.innerHTML = `
        <div class="message assistant-message">
            <div class="message-avatar">
                <i class="fas fa-hospital"></i>
            </div>
            <div class="message-content">
                <div class="message-text">
                    안녕하세요! 🏥 TeamMediChat입니다.<br>
                    의약품에 대한 질문을 자유롭게 해주세요.<br>
                    이전 대화 내용을 기억하여 연속된 질문에도 답변할 수 있습니다.
                </div>
                <div class="message-timestamp">지금</div>
            </div>
        </div>
    `;
}

// 연결 상태 업데이트
function updateConnectionStatus(connected) {
    if (connected) {
        sessionInfo.innerHTML = '<i class="fas fa-circle"></i> 연결됨';
        sessionInfo.style.color = '#28a745';
    } else {
        sessionInfo.innerHTML = '<i class="fas fa-circle"></i> 연결 끊어짐';
        sessionInfo.style.color = '#dc3545';
    }
}

// 스크롤을 맨 아래로
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// 타임스탬프 포맷
function formatTimestamp(timestamp) {
    if (!timestamp) return '지금';
    
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;
    
    if (diff < 60000) { // 1분 미만
        return '방금 전';
    } else if (diff < 3600000) { // 1시간 미만
        const minutes = Math.floor(diff / 60000);
        return `${minutes}분 전`;
    } else if (diff < 86400000) { // 24시간 미만
        const hours = Math.floor(diff / 3600000);
        return `${hours}시간 전`;
    } else {
        return date.toLocaleDateString('ko-KR');
    }
}

// 윈도우 리사이즈 처리
function handleResize() {
    if (window.innerWidth > 768) {
        sidebar.classList.remove('show');
    }
}

// 이미지 업로드 처리
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    // 파일 타입 검증
    if (!file.type.startsWith('image/')) {
        showError('이미지 파일만 업로드할 수 있습니다.');
        return;
    }
    
    // 파일 크기 검증 (5MB 제한)
    if (file.size > 5 * 1024 * 1024) {
        showError('이미지 크기는 5MB 이하여야 합니다.');
        return;
    }
    
    // FileReader로 이미지 읽기
    const reader = new FileReader();
    reader.onload = function(e) {
        const imageData = e.target.result;
        const byteArray = new Uint8Array(e.target.result);
        currentImageData = byteArray;
        
        // 이미지 미리보기 표시
        previewImage.src = imageData;
        imageUploadSection.style.display = 'block';
        
        console.log('이미지 업로드 완료:', file.name, file.size, 'bytes');
    };
    
    reader.readAsArrayBuffer(file);
}

// 이미지 제거
function removeImage() {
    currentImageData = null;
    imageUploadSection.style.display = 'none';
    previewImage.src = '';
    imageInput.value = '';
}

// 이미지가 포함된 메시지 표시
function displayMessageWithImage(role, content, timestamp, imageData) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    
    if (role === 'user') {
        avatar.innerHTML = '<i class="fas fa-user"></i>';
    } else {
        avatar.innerHTML = '<i class="fas fa-hospital"></i>';
    }
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    const messageText = document.createElement('div');
    messageText.className = 'message-text';
    messageText.textContent = content;
    
    // 이미지가 있는 경우 이미지 표시
    if (imageData && role === 'user') {
        const imageBlob = new Blob([imageData], { type: 'image/jpeg' });
        const imageUrl = URL.createObjectURL(imageBlob);
        
        const messageImage = document.createElement('div');
        messageImage.className = 'message-image';
        messageImage.innerHTML = `
            <img src="${imageUrl}" style="max-width: 200px; max-height: 150px; border-radius: 8px; margin-top: 10px;">
        `;
        messageText.appendChild(messageImage);
    }
    
    const messageTimestamp = document.createElement('div');
    messageTimestamp.className = 'message-timestamp';
    messageTimestamp.textContent = formatTimestamp(timestamp);
    
    messageContent.appendChild(messageText);
    messageContent.appendChild(messageTimestamp);
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// 페이지 언로드 시 정리
window.addEventListener('beforeunload', function() {
    if (websocket) {
        websocket.close();
    }
});
