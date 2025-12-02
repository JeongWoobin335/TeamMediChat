// 전역 변수
let currentSessionId = null;
let websocket = null;
let isConnected = false;
let typingTimer = null;
let userLocation = null; // 사용자 위치 정보 저장
let messagesLoaded = false; // 메시지 로드 상태 추적 (중복 방지용)

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
const sessionsToggleBtn = document.getElementById('sessionsToggleBtn');
const sessionsToggleIcon = document.getElementById('sessionsToggleIcon');
const sessionInfo = document.getElementById('sessionInfo');

// 이미지 업로드 관련 요소들
const imageUploadBtn = document.getElementById('imageUploadBtn');
const imageInput = document.getElementById('imageInput');
const imageUploadSection = document.getElementById('imageUploadSection');
const imagePreview = document.getElementById('imagePreview');
const previewImage = document.getElementById('previewImage');
const removeImageBtn = document.getElementById('removeImageBtn');

// 카카오 맵 모달 관련 요소들
const pharmacyFindBtn = document.getElementById('pharmacyFindBtn');
const mapModal = document.getElementById('mapModal');
const mapModalClose = document.getElementById('mapModalClose');

// 주소 입력 관련 요소들
const addressInputSection = document.getElementById('addressInputSection');
const addressInput = document.getElementById('addressInput');
const searchAddressBtn = document.getElementById('searchAddressBtn');
const addressSuggestions = document.getElementById('addressSuggestions');

// 전역 변수
let currentImageData = null;


// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', function() {
    // 스플래시 화면 제거
    setTimeout(() => {
        const splashScreen = document.getElementById('splashScreen');
        if (splashScreen) {
            splashScreen.remove();
        }
    }, 2500); // 2초 + 페이드아웃 애니메이션 시간
    
    initializeApp();
    setupEventListeners();
    loadSessions();
});

// 앱 초기화
async function initializeApp() {
    // 기존 세션이 있는지 확인 후 로드
    await loadExistingSessions();
    
    // 입력 필드 자동 크기 조정
    autoResizeTextarea();
    
    // 모바일 사이드바 토글
    if (window.innerWidth <= 768) {
        sidebar.classList.remove('show');
    }
    
    // 사용자 위치 정보 로드
    loadUserLocationFromStorage();
}

// 기존 세션 로드 또는 새 세션 생성
async function loadExistingSessions() {
    try {
        const response = await fetch('/api/sessions');
        if (response.ok) {
            const data = await response.json();
            if (data.sessions && data.sessions.length > 0) {
                // 기존 세션이 있으면 가장 최근 세션 사용
                const latestSession = data.sessions[0]; // 가장 최근 세션
                currentSessionId = latestSession.session_id;
                
                // 해당 세션의 대화 내용 먼저 로드 (API 사용)
                await loadSessionMessages(currentSessionId);
                
                // WebSocket 연결 (히스토리는 이미 로드했으므로 중복 방지)
                connectWebSocket(currentSessionId);
            } else {
                // 기존 세션이 없으면 새로 생성
                await createNewSession();
            }
        } else {
            // API 오류 시 새 세션 생성
            await createNewSession();
        }
    } catch (error) {
        console.error('세션 로드 오류:', error);
        // 오류 시 새 세션 생성
        await createNewSession();
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
    
    // 세션 목록 토글
    sessionsToggleBtn.addEventListener('click', toggleSessionsList);
    
    // 카카오 맵 모달
    pharmacyFindBtn.addEventListener('click', openMapModal);
    mapModalClose.addEventListener('click', closeMapModal);
    mapModal.addEventListener('click', function(e) {
        if (e.target === mapModal) {
            closeMapModal();
        }
    });
    
    // 주소 입력 관련 이벤트
    searchAddressBtn.addEventListener('click', searchAddress);
    addressInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            searchAddress();
        }
    });
    addressInput.addEventListener('input', handleAddressInput);
    
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
            isConnected = true;
            updateConnectionStatus(true);
        };
        
        websocket.onmessage = function(event) {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        };
        
        websocket.onclose = function(event) {
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
            break;
            
        case 'chat_message':
            // 사용자 메시지는 이미 클라이언트에서 표시했으므로 서버에서 받은 사용자 메시지는 무시
            // (중복 방지: 서버는 브로드캐스트를 위해 사용자 메시지를 다시 보내지만, 
            //  클라이언트에서는 이미 표시했으므로 assistant 메시지만 표시)
            if (data.role === 'user') {
                // 사용자 메시지는 무시 (이미 sendMessage()에서 표시함)
                break;
            }
            
            // assistant 메시지만 표시
            displayMessage(data.role, data.content, data.timestamp);
            // AI 답변을 받은 후 로딩 화면 숨기기
            if (data.role === 'assistant') {
                hideLoading();
            }
            break;
            
        case 'chat_history':
            // API로 이미 메시지를 로드했다면 WebSocket 히스토리는 무시 (중복 방지)
            if (!messagesLoaded) {
                displayChatHistory(data.history);
                messagesLoaded = true;
            }
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
            break;
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
        } else if (line.startsWith('의사: ')) {
            // 서버에서 "의사"로 보내므로 "AI" 대신 "의사" 확인
            if (currentRole && currentContent.length > 0) {
                displayMessage(currentRole, currentContent.join('\n'), new Date().toISOString());
            }
            currentRole = 'assistant';
            currentContent = [line.substring(4)];
        } else if (line.startsWith('AI: ')) {
            // 호환성을 위해 "AI: "도 지원
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
        avatar.innerHTML = '<img src="/static/assets/logo_white.png" alt="TeamMediChat" style="width: 35px; height: 35px; vertical-align: middle;">';
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
    
    // WebSocket으로 메시지 전송 (위치 정보 포함)
    const messageData = {
        type: 'chat_message',
        content: message,
        image_data: currentImageData ? Array.from(currentImageData) : null,  // 이미지 데이터 포함
        user_location: userLocation // 사용자 위치 정보 추가
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
    // 모바일에서는 show 클래스 토글
    if (window.innerWidth <= 768) {
        sidebar.classList.toggle('show');
    } else {
        // 데스크톱에서는 hidden 클래스 토글
        sidebar.classList.toggle('hidden');
        // 버튼 아이콘 변경
        const icon = sidebarToggle.querySelector('i');
        if (sidebar.classList.contains('hidden')) {
            icon.className = 'fas fa-chevron-right';
        } else {
            icon.className = 'fas fa-bars';
        }
    }
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
            
            // 메시지 로드 상태 초기화
            messagesLoaded = false;
            
            // WebSocket 연결
            if (websocket) {
                websocket.close();
            }
            connectWebSocket(currentSessionId);
            
            // 채팅 영역 초기화
            clearChatMessages();
            
            // 세션 목록 새로고침
            loadSessions();
            
            // 새 세션 생성됨
        } else {
            throw new Error('세션 생성 실패');
        }
    } catch (error) {
        console.error('세션 생성 오류:', error);
        showError('새 세션을 생성할 수 없습니다.');
    }
}

// 세션 목록 토글
function toggleSessionsList() {
    const isHidden = sessionsList.style.display === 'none';
    
    if (isHidden) {
        sessionsList.style.display = 'block';
        sessionsToggleBtn.classList.add('active');
    } else {
        sessionsList.style.display = 'none';
        sessionsToggleBtn.classList.remove('active');
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
    
    sessions.forEach((session, index) => {
        const sessionItem = document.createElement('div');
        sessionItem.className = 'session-item';
        if (session.session_id === currentSessionId) {
            sessionItem.classList.add('active');
        }
        
        const sessionTitle = document.createElement('div');
        sessionTitle.className = 'session-title';
        sessionTitle.textContent = `${sessions.length - index}번째 대화`;
        
        const sessionMeta = document.createElement('div');
        sessionMeta.className = 'session-meta';
        sessionMeta.textContent = `${session.message_count}개 메시지`;
        
        // 삭제 버튼 추가
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'session-delete-btn';
        deleteBtn.title = '대화 삭제';
        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation(); // 클릭 이벤트 전파 방지
            deleteSession(session.session_id);
        });
        
        sessionItem.appendChild(sessionTitle);
        sessionItem.appendChild(sessionMeta);
        sessionItem.appendChild(deleteBtn);
        
        sessionItem.addEventListener('click', () => {
            switchSession(session.session_id);
        });
        
        sessionsList.appendChild(sessionItem);
    });
}

// 세션 전환
async function switchSession(sessionId) {
    if (sessionId === currentSessionId) return;
    
    currentSessionId = sessionId;
    
    // 메시지 로드 상태 초기화
    messagesLoaded = false;
    
    // 채팅 영역 초기화
    clearChatMessages();
    
    // 해당 세션의 대화 내용 먼저 로드
    await loadSessionMessages(sessionId);
    
    // WebSocket 재연결 (히스토리는 이미 로드했으므로 중복 방지)
    if (websocket) {
        websocket.close();
    }
    connectWebSocket(currentSessionId);
    
    // 세션 목록 새로고침
    loadSessions();
    
    // 모바일에서 사이드바 닫기
    if (window.innerWidth <= 768) {
        sidebar.classList.remove('show');
    }
}

// 세션의 대화 내용 로드
async function loadSessionMessages(sessionId) {
    try {
        const response = await fetch(`/api/sessions/${sessionId}/messages`);
        if (response.ok) {
            const data = await response.json();
            if (data.messages && data.messages.length > 0) {
                // 기존 환영 메시지 제거
                clearChatMessages();
                
                // 대화 내용 표시
                data.messages.forEach(message => {
                    displayMessage(message.role, message.content, message.timestamp);
                });
                
                // 메시지 로드 완료 표시 (WebSocket 히스토리 중복 방지)
                messagesLoaded = true;
                
                // 스크롤을 맨 아래로
                scrollToBottom();
            } else {
                // 메시지가 없어도 로드 완료로 표시
                messagesLoaded = true;
            }
        } else {
            console.error('세션 메시지 로드 실패:', response.status);
            messagesLoaded = true; // 오류가 있어도 로드 시도 완료로 표시
        }
    } catch (error) {
        console.error('세션 메시지 로드 오류:', error);
        messagesLoaded = true; // 오류가 있어도 로드 시도 완료로 표시
    }
}

// 세션 삭제
async function deleteSession(sessionId) {
    if (confirm('이 대화를 삭제하시겠습니까?')) {
        try {
            const response = await fetch(`/api/sessions/${sessionId}`, {
                method: 'DELETE'
            });
            
            if (response.ok) {
                // 현재 세션이 삭제된 세션이면 새 세션 생성
                if (sessionId === currentSessionId) {
                    await createNewSession();
                }
                
                // 세션 목록 새로고침
                loadSessions();
            } else {
                console.error('세션 삭제 실패:', response.status);
                alert('대화 삭제에 실패했습니다.');
            }
        } catch (error) {
            console.error('세션 삭제 오류:', error);
            alert('대화 삭제 중 오류가 발생했습니다.');
        }
    }
}

// 채팅 메시지 초기화
function clearChatMessages() {
    chatMessages.innerHTML = `
        <div class="message assistant-message">
            <div class="message-avatar">
                <img src="/static/assets/logo_white.png" alt="TeamMediChat" style="width: 35px; height: 35px; vertical-align: middle;">
            </div>
            <div class="message-content">
                <div class="message-text">안녕하세요! 💬 TeamMediChat입니다.<br>
의약품에 대해 궁금한 점이 있다면 편하게 질문해주세요.<br>
주변 병원이나 약국을 찾고 싶으시다면 상단의 <b>약국 찾기</b> 버튼을 눌러주세요.<br><br>
<small>※ TeamMediChat은 전문 의학 상담을 제공하지 않습니다. 증상에 따른 처방이나 정확한 진단이 필요하다면 의료 전문가와 상담하시길 권장드립니다.</small></div>
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
        
        // 이미지 업로드 완료
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
        avatar.innerHTML = '<img src="/static/assets/logo_white.png" alt="TeamMediChat" style="width: 35px; height: 35px; vertical-align: middle;">';
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

// ==================== 카카오 맵 관련 함수들 ====================

// 카카오 맵 API 로딩 대기 함수
function waitForKakaoMapAPI() {
    return new Promise((resolve, reject) => {
        if (typeof kakao !== 'undefined' && kakao.maps && kakao.maps.services && kakao.maps.services.Places) {
            // 카카오 맵 API 및 Places 서비스 이미 로드됨
            resolve();
            return;
        }
        
        // 카카오 맵 API 및 Places 서비스 로딩 대기 중
        const checkInterval = setInterval(() => {
            if (typeof kakao !== 'undefined' && kakao.maps && kakao.maps.services && kakao.maps.services.Places) {
                clearInterval(checkInterval);
                // 카카오 맵 API 및 Places 서비스 로딩 완료
                resolve();
            }
        }, 100);
        
        // 10초 타임아웃
        setTimeout(() => {
            clearInterval(checkInterval);
            reject(new Error('카카오 맵 API 또는 Places 서비스 로딩 타임아웃'));
        }, 10000);
    });
}

// test.js 동적 로딩 함수
function loadKakaoMapScript() {
    return new Promise(async (resolve, reject) => {
        try {
            // 먼저 카카오 맵 API 로딩 대기
            await waitForKakaoMapAPI();
            
            // 이미 로드되었는지 확인
            if (typeof initializeKakaoMap === 'function') {
                // test.js 이미 로드됨
                resolve();
                return;
            }
            
            // 이미 로딩 중인지 확인
            if (window.kakaoMapLoading) {
                // test.js 로딩 중
                // 로딩 완료까지 대기
                const checkInterval = setInterval(() => {
                    if (typeof initializeKakaoMap === 'function') {
                        clearInterval(checkInterval);
                        resolve();
                    }
                }, 100);
                return;
            }
            
            // test.js 동적 로딩 시작
            window.kakaoMapLoading = true;
            
            const script = document.createElement('script');
            script.src = '/static/test.js';
            script.onload = () => {
                // test.js 로드 완료
                window.kakaoMapLoading = false;
                resolve();
            };
            script.onerror = (error) => {
                console.error('❌ test.js 로드 실패:', error);
                window.kakaoMapLoading = false;
                reject(error);
            };
            document.head.appendChild(script);
        } catch (error) {
            reject(error);
        }
    });
}

// 카카오 맵 모달 열기
async function openMapModal() {
    // 카카오 맵 모달 열기 시작
    
    mapModal.classList.add('show');
    
    try {
        // test.js 로드 대기
        await loadKakaoMapScript();
        
        // 맵 초기화
        if (typeof initializeKakaoMap === 'function') {
            // test.js의 맵 초기화 함수 사용
            const map = initializeKakaoMap('map');
            if (map) {
                // 모달이 완전히 표시된 후 맵 크기 재조정
                setTimeout(() => {
                    map.relayout();
                    // 맵 크기 재조정 완료
                }, 100);
                // 맵 초기화 완료
            } else {
                console.error('❌ 맵 초기화 실패');
                showError('맵을 초기화할 수 없습니다.');
            }
        } else {
            console.error('❌ initializeKakaoMap 함수를 찾을 수 없음');
            showError('맵 초기화 함수를 찾을 수 없습니다.');
        }
    } catch (error) {
        console.error('❌ 맵 로딩 중 오류:', error);
        showError('맵을 로드할 수 없습니다: ' + error.message);
    }
}

// ===== 위치 정보 관련 함수들 =====

// 저장된 사용자 위치 정보 로드 (세션 기반)
function loadUserLocationFromStorage() {
    try {
        const savedLocation = sessionStorage.getItem('userLocation');
        if (savedLocation) {
            userLocation = JSON.parse(savedLocation);
            return userLocation;
        }
    } catch (error) {
        console.error('❌ 위치 정보 로드 실패:', error);
    }
    
    userLocation = null;
    return null;
}

// 사용자 위치 정보 업데이트 (카카오 맵에서 호출)
function updateUserLocation(lat, lng) {
    userLocation = {
        lat: lat,
        lng: lng,
        timestamp: Date.now()
    };
    
    // 세션 스토리지에 저장 (브라우저 탭 닫으면 삭제됨)
    sessionStorage.setItem('userLocation', JSON.stringify(userLocation));
}

// 근처 약국 정보를 답변에 추가하는 함수
function addPharmacyInfoToAnswer(answer, pharmacies) {
    if (!pharmacies || pharmacies.length === 0) {
        return answer;
    }
    
    let pharmacyInfo = '\n\n🏥 **근처 약국 정보:**\n';
    
    pharmacies.forEach((pharmacy, index) => {
        pharmacyInfo += `${index + 1}. **${pharmacy.name}**\n`;
        pharmacyInfo += `   📍 ${pharmacy.road_address || pharmacy.address}\n`;
        if (pharmacy.phone) {
            pharmacyInfo += `   📞 ${pharmacy.phone}\n`;
        }
        pharmacyInfo += `   📏 거리: ${pharmacy.distance}m\n\n`;
    });
    
    pharmacyInfo += '💡 **참고:** 위 약국들은 현재 위치 기준으로 가장 가까운 곳들입니다. 정확한 약품 구매 가능 여부는 약국에 직접 문의하시기 바랍니다.';
    
    return answer + pharmacyInfo;
}

// 약국 정보가 포함된 답변을 표시하는 함수
function displayAnswerWithPharmacy(answer, pharmacies) {
    const enhancedAnswer = addPharmacyInfoToAnswer(answer, pharmacies);
    displayMessage('assistant', enhancedAnswer, new Date().toISOString());
}

// 카카오 맵 모달 닫기
function closeMapModal() {
    mapModal.classList.remove('show');
}

// ==================== 주소 검색 관련 함수들 ====================

// 주소 검색 함수
async function searchAddress() {
    const query = addressInput.value.trim();
    if (!query) {
        showError('주소를 입력해주세요.');
        return;
    }
    
    try {
        searchAddressBtn.disabled = true;
        searchAddressBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 검색 중...';
        
        // 카카오 주소 검색 API 사용
        const geocoder = new kakao.maps.services.Geocoder();
        
        geocoder.addressSearch(query, function(result, status) {
            searchAddressBtn.disabled = false;
            searchAddressBtn.innerHTML = '<i class="fas fa-search"></i> 검색';
            
            if (status === kakao.maps.services.Status.OK) {
                displayAddressSuggestions(result);
            } else {
                showError('주소를 찾을 수 없습니다. 다른 주소를 입력해주세요.');
            }
        });
        
    } catch (error) {
        searchAddressBtn.disabled = false;
        searchAddressBtn.innerHTML = '<i class="fas fa-search"></i> 검색';
        console.error('주소 검색 오류:', error);
        showError('주소 검색 중 오류가 발생했습니다.');
    }
}

// 주소 검색 결과 표시
function displayAddressSuggestions(results) {
    addressSuggestions.innerHTML = '';
    
    if (results.length === 0) {
        addressSuggestions.style.display = 'none';
        return;
    }
    
    results.forEach((result, index) => {
        const suggestionItem = document.createElement('div');
        suggestionItem.className = 'address-suggestion-item';
        suggestionItem.innerHTML = `
            <div class="address-name">${result.place_name || result.address_name}</div>
            <div class="address-detail">${result.address_name}</div>
        `;
        
        suggestionItem.addEventListener('click', function() {
            selectAddress(result);
        });
        
        addressSuggestions.appendChild(suggestionItem);
    });
    
    addressSuggestions.style.display = 'block';
}

// 주소 선택 함수
function selectAddress(addressResult) {
    const position = new kakao.maps.LatLng(addressResult.y, addressResult.x);
    
    // 주소 입력 섹션 숨기기
    addressInputSection.style.display = 'none';
    
    // 맵에 마커 표시
    if (typeof displayMarker === 'function') {
        displayMarker(position);
    }
    
    // 사용자 위치 정보 업데이트
    updateUserLocation(addressResult.y, addressResult.x);
    
    // 주소 위치 정보 저장 (다음에 맵을 열 때 복원용)
    saveAddressLocation(addressResult);
    
    // 검색 결과 숨기기
    addressSuggestions.style.display = 'none';
}

// 주소 위치 정보 저장 (세션 기반)
function saveAddressLocation(addressResult) {
    try {
        const addressData = {
            lat: addressResult.y,
            lng: addressResult.x,
            address: addressResult.address_name,
            timestamp: Date.now()
        };
        
        sessionStorage.setItem('savedAddressLocation', JSON.stringify(addressData));
    } catch (error) {
        console.error('❌ 주소 위치 정보 저장 실패:', error);
    }
}

// 주소 입력 처리
function handleAddressInput() {
    const query = addressInput.value.trim();
    if (query.length < 2) {
        addressSuggestions.style.display = 'none';
        return;
    }
    
    // 실시간 검색 (디바운싱)
    clearTimeout(window.addressSearchTimeout);
    window.addressSearchTimeout = setTimeout(() => {
        if (query.length >= 2) {
            searchAddress();
        }
    }, 300);
}


