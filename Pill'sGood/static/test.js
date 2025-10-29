var placeOverlay, contentNode, markers = [], currCategory = '';
var map, ps;
var currentLocationMarker = null; // 현재 위치 마커 관리

// 카카오 맵 초기화 함수
window.initializeKakaoMap = function(containerId) {
    
    // 카카오 맵 API 로딩 확인
    if (typeof kakao === 'undefined' || !kakao.maps) {
        console.error('카카오 맵 API가 로드되지 않았습니다');
        return null;
    }
    
    // Places 서비스 확인
    if (!kakao.maps.services || !kakao.maps.services.Places) {
        console.error('카카오 맵 Places 서비스가 로드되지 않았습니다');
        return null;
    }
    
    // 컨테이너 확인
    var mapContainer = document.getElementById(containerId);
    if (!mapContainer) {
        console.error('맵 컨테이너를 찾을 수 없습니다:', containerId);
        return null;
    }
    
    // 기존 맵이 있으면 제거
    if (map) {
        map = null;
    }
    
    try {
        // 마커 클릭 시 해당 장소의 상세정보 출력하는 커스텀오버레이
        placeOverlay = new kakao.maps.CustomOverlay({zIndex: 1});
        contentNode = document.createElement('div'); // 커스텀 오버레이의 컨텐츠 엘리먼트
        markers = []; // 마커 담을 배열
        currCategory = ''; // 현재 선택된 카테고리 담는 변수
        
        var mapOption = { 
            center: new kakao.maps.LatLng(33.450701, 126.570667), // 지도의 중심좌표
            level: 6 // 지도의 확대 레벨 
        }; 
        
        map = new kakao.maps.Map(mapContainer, mapOption); // 지도를 생성합니다
        ps = new kakao.maps.services.Places(map); // 장소 검색 객체 생성
        
        // 이벤트 리스너 등록
        setupMapEvents();
        
        return map;
    } catch (error) {
        console.error('맵 초기화 중 오류:', error);
        return null;
    }
};

// 맵 이벤트 설정 함수
function setupMapEvents() {
    // 지도에 idle 이벤트 등록
    kakao.maps.event.addListener(map, 'idle', searchPlaces);
    
    //커스텀 오버레이의 컨텐츠 노드에 css class 추가
    contentNode.className = 'placeinfo_wrap';
    
    // 커스텀 오버레이의 컨텐츠 노드에 mousedown, touchstart 이벤트가 발생했을때
    // 지도 객체에 이벤트가 전달되지 않도록 이벤트 핸들러로 kakao.maps.event.preventMap 메소드를 등록합니다 
    addEventHandle(contentNode, 'mousedown', kakao.maps.event.preventMap);
    addEventHandle(contentNode, 'touchstart', kakao.maps.event.preventMap);
    
    // 커스텀 오버레이 컨텐츠 설정
    placeOverlay.setContent(contentNode);
    
    // 각 카테고리에 클릭 이벤트 등록
    addCategoryClickEvent();
    
    // 주소 입력 우선 모드 - GPS 위치로 맵 중심 설정 (마커 없이)
    setMapCenterFromGPS();
    
    // 지도 클릭 이벤트 등록
    setupMapClickEvent();
    
    // 위치 재요청 버튼 제거됨 (주소 입력 우선 모드)
}

// 현재 위치 표시 함수 (정확도 개선)
function showCurrentLocation() {
    // HTML5의 geolocation으로 사용할 수 있는지 확인합니다 
    if (navigator.geolocation) {
        
        // 정확도 옵션 설정
        var options = {
            enableHighAccuracy: true,    // 높은 정확도 요청
            timeout: 10000,              // 10초 타임아웃
            maximumAge: 300000           // 5분간 캐시된 위치 허용
        };
        
        // 현재 위치 요청 중
        
        // GeoLocation을 이용해서 접속 위치를 얻어옵니다
        navigator.geolocation.getCurrentPosition(
            function(position) {
                var lat = position.coords.latitude, // 위도
                    lon = position.coords.longitude, // 경도
                    accuracy = position.coords.accuracy; // 정확도 (미터)
                
                // 위치 획득 완료
                
                var locPosition = new kakao.maps.LatLng(lat, lon), // 마커가 표시될 위치를 geolocation으로 얻어온 좌표로 생성합니다
                    message = `<div style="padding:5px;">
                        <strong>현위치</strong><br>
                        <small>정확도: ±${Math.round(accuracy)}m</small>
                    </div>`; // 인포윈도우에 표시될 내용입니다
                
                // 마커와 인포윈도우를 표시합니다
                displayMarker(locPosition, message);
                
                // 스마트 위치 설정 로직
                if (accuracy <= 100) {
                    // 정확도가 좋으면 위치 설정 제안
                    // 위치 설정 안내 제거됨 (주소 입력 우선 모드)
                } else {
                    // 정확도가 낮으면 수동 설정 안내
                    // 위치 설정 안내 제거됨 (주소 입력 우선 모드)
                }
            },
            function(error) {
                console.error('위치 정보 획득 실패:', error.message);
                
                // 오류에 따른 기본 위치 설정
                var locPosition, message;
                
                switch(error.code) {
                    case error.PERMISSION_DENIED:
                        message = '위치 접근이 거부되었습니다.';
                        locPosition = new kakao.maps.LatLng(37.5665, 126.9780); // 서울시청
                        break;
                    case error.POSITION_UNAVAILABLE:
                        message = '위치 정보를 사용할 수 없습니다.';
                        locPosition = new kakao.maps.LatLng(37.5665, 126.9780); // 서울시청
                        break;
                    case error.TIMEOUT:
                        message = '위치 요청 시간 초과. 실외에서 다시 시도해주세요.';
                        locPosition = new kakao.maps.LatLng(37.5665, 126.9780); // 서울시청
                        break;
                    default:
                        message = '알 수 없는 오류가 발생했습니다.';
                        locPosition = new kakao.maps.LatLng(37.5665, 126.9780); // 서울시청
                        break;
                }
                
                displayMarker(locPosition, `<div style="padding:5px;">${message}</div>`);
            },
            options
        );
        
    } else { // HTML5의 GeoLocation을 사용할 수 없을때 마커 표시 위치와 인포윈도우 내용을 설정합니다
        
        var locPosition = new kakao.maps.LatLng(37.5665, 126.9780), // 서울시청으로 변경
            message = 'geolocation을 사용할 수 없습니다.'
            
        displayMarker(locPosition, `<div style="padding:5px;">${message}</div>`);
    }
}

// 엘리먼트에 이벤트 핸들러를 등록하는 함수입니다
function addEventHandle(target, type, callback) {
    if (target.addEventListener) {
        target.addEventListener(type, callback);
    } else {
        target.attachEvent('on' + type, callback);
    }
}

// 카테고리 검색을 요청하는 함수입니다
function searchPlaces() {
    if (!currCategory) {
        return;
    }
    
    // 커스텀 오버레이를 숨깁니다 
    placeOverlay.setMap(null);

    // 지도에 표시되고 있는 마커를 제거합니다
    removeMarker();
    
    ps.categorySearch(currCategory, placesSearchCB, {useMapBounds:true}); 
}

// 장소검색이 완료됐을 때 호출되는 콜백함수 입니다
function placesSearchCB(data, status, pagination) {
    if (status === kakao.maps.services.Status.OK) {

        // 정상적으로 검색이 완료됐으면 지도에 마커를 표출합니다
        displayPlaces(data);
    } else if (status === kakao.maps.services.Status.ZERO_RESULT) {
        // 검색결과가 없는경우 해야할 처리가 있다면 이곳에 작성해 주세요

    } else if (status === kakao.maps.services.Status.ERROR) {
        // 에러로 인해 검색결과가 나오지 않은 경우 해야할 처리가 있다면 이곳에 작성해 주세요
        
    }
}

// 지도에 마커를 표출하는 함수입니다
function displayPlaces(places) {

    // 몇번째 카테고리가 선택되어 있는지 얻어옵니다
    // 이 순서는 스프라이트 이미지에서의 위치를 계산하는데 사용됩니다
    var order = document.getElementById(currCategory).getAttribute('data-order');

    

    for ( var i=0; i<places.length; i++ ) {

            // 마커를 생성하고 지도에 표시합니다
            var marker = addMarker(new kakao.maps.LatLng(places[i].y, places[i].x), order);

            // 마커와 검색결과 항목을 클릭 했을 때
            // 장소정보를 표출하도록 클릭 이벤트를 등록합니다
            (function(marker, place) {
                kakao.maps.event.addListener(marker, 'click', function() {
                    displayPlaceInfo(place);
                });
            })(marker, places[i]);
    }
}

// 마커를 생성하고 지도 위에 마커를 표시하는 함수입니다
function addMarker(position, order) {
    var imageSrc = 'https://t1.daumcdn.net/localimg/localimages/07/mapapidoc/places_category.png', // 마커 이미지 url, 스프라이트 이미지를 씁니다
        imageSize = new kakao.maps.Size(27, 28),  // 마커 이미지의 크기
        imgOptions =  {
            spriteSize : new kakao.maps.Size(72, 208), // 스프라이트 이미지의 크기
            spriteOrigin : new kakao.maps.Point(46, (order*36)), // 스프라이트 이미지 중 사용할 영역의 좌상단 좌표
            offset: new kakao.maps.Point(11, 28) // 마커 좌표에 일치시킬 이미지 내에서의 좌표
        },
        markerImage = new kakao.maps.MarkerImage(imageSrc, imageSize, imgOptions),
            marker = new kakao.maps.Marker({
            position: position, // 마커의 위치
            image: markerImage 
        });

    marker.setMap(map); // 지도 위에 마커를 표출합니다
    markers.push(marker);  // 배열에 생성된 마커를 추가합니다

    return marker;
}

// 지도 위에 표시되고 있는 마커를 모두 제거합니다
function removeMarker() {
    for ( var i = 0; i < markers.length; i++ ) {
        markers[i].setMap(null);
    }   
    markers = [];
}

// 클릭한 마커에 대한 장소 상세정보를 커스텀 오버레이로 표시하는 함수입니다
function displayPlaceInfo (place) {
    var content = '<div class="placeinfo">' +
                    '   <a class="title" href="' + place.place_url + '" target="_blank" title="' + place.place_name + '">' + place.place_name + '</a>';   

    if (place.road_address_name) {
        content += '    <span title="' + place.road_address_name + '">' + place.road_address_name + '</span>' +
                    '  <span class="jibun" title="' + place.address_name + '">(지번 : ' + place.address_name + ')</span>';
    }  else {
        content += '    <span title="' + place.address_name + '">' + place.address_name + '</span>';
    }                
   
    content += '    <span class="tel">' + place.phone + '</span>' + 
                '</div>' + 
                '<div class="after"></div>';

    contentNode.innerHTML = content;
    placeOverlay.setPosition(new kakao.maps.LatLng(place.y, place.x));
    placeOverlay.setMap(map);  
}


// 각 카테고리에 클릭 이벤트를 등록합니다
function addCategoryClickEvent() {
    var category = document.getElementById('category'),
        children = category.children;

    for (var i=0; i<children.length; i++) {
        children[i].onclick = onClickCategory;
    }
}

// 카테고리를 클릭했을 때 호출되는 함수입니다
function onClickCategory() {
    var id = this.id,
        className = this.className;

    placeOverlay.setMap(null);

    if (className === 'on') {
        currCategory = '';
        changeCategoryClass();
        removeMarker();
    } else {
        currCategory = id;
        changeCategoryClass(this);
        searchPlaces();
    }
}

// 클릭된 카테고리에만 클릭된 스타일을 적용하는 함수입니다
function changeCategoryClass(el) {
    var category = document.getElementById('category'),
        children = category.children,
        i;

    for ( i=0; i<children.length; i++ ) {
        children[i].className = '';
    }

    if (el) {
        el.className = 'on';
    } 
} 




// 지도에 마커와 인포윈도우를 표시하는 함수입니다 (개선된 버전)
function displayMarker(locPosition, message) {
    // 기존 현재 위치 마커가 있으면 제거
    if (currentLocationMarker) {
        currentLocationMarker.setMap(null);
    }

    // 새로운 마커를 생성합니다
    currentLocationMarker = new kakao.maps.Marker({  
        map: map, 
        position: locPosition
    }); 
    
    var iwContent = message, // 인포윈도우에 표시할 내용
        iwRemoveable = true;

    // 인포윈도우를 생성합니다
    var infowindow = new kakao.maps.InfoWindow({
        content : iwContent,
        removable : iwRemoveable
    });
    
    // 인포윈도우를 마커위에 표시합니다 
    infowindow.open(map, currentLocationMarker);
    
    // 지도 중심좌표를 접속위치로 변경합니다
    map.setCenter(locPosition);
}

// 위치 설정 안내 함수 제거됨 (주소 입력 우선 모드)

// 사용자 위치 저장 함수 제거됨 (주소 입력 우선 모드)

// GPS 위치로 맵 중심 설정 (마커 없이)
function setMapCenterFromGPS() {
    if (navigator.geolocation) {
        var options = {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 0
        };
        
        navigator.geolocation.getCurrentPosition(
            function(position) {
                var lat = position.coords.latitude;
                var lon = position.coords.longitude;
                var accuracy = position.coords.accuracy;
                
                // 맵 중심을 GPS 위치로 설정 (마커는 표시하지 않음)
                var gpsPosition = new kakao.maps.LatLng(lat, lon);
                map.setCenter(gpsPosition);
                
                // 저장된 주소 위치가 있으면 복원
                restoreSavedAddressLocation();
            },
            function(error) {
                // GPS 실패 시 기본 위치 (서울시청)
                var defaultPosition = new kakao.maps.LatLng(37.5665, 126.9780);
                map.setCenter(defaultPosition);
                
                // 저장된 주소 위치가 있으면 복원
                restoreSavedAddressLocation();
            },
            options
        );
    } else {
        var defaultPosition = new kakao.maps.LatLng(37.5665, 126.9780);
        map.setCenter(defaultPosition);
        
        // 저장된 주소 위치가 있으면 복원
        restoreSavedAddressLocation();
    }
}

// 저장된 주소 위치 복원 (세션 기반)
function restoreSavedAddressLocation() {
    try {
        var savedAddress = sessionStorage.getItem('savedAddressLocation');
        if (savedAddress) {
            var addressData = JSON.parse(savedAddress);
            
            // 주소 입력 섹션 숨기기
            var addressInputSection = document.getElementById('addressInputSection');
            if (addressInputSection) {
                addressInputSection.style.display = 'none';
            }
            
            // 맵에 마커 표시
            var position = new kakao.maps.LatLng(addressData.lat, addressData.lng);
            var message = `<div style="padding:5px;">
                <strong>설정된 위치</strong><br>
                <small>${addressData.address}</small>
            </div>`;
            displayMarker(position, message);
            
            // 맵 중심을 저장된 위치로 설정
            map.setCenter(position);
        }
    } catch (error) {
        console.error('저장된 주소 위치 복원 실패:', error);
    }
}

// 저장된 사용자 위치 불러오기 (완전 비활성화)
function loadUserLocation() {
    // 주소 입력 우선 모드에서는 저장된 위치를 완전히 무시
    return null;
}

// 지도 클릭 이벤트 설정
function setupMapClickEvent() {
    // 지도 클릭 이벤트 등록
    kakao.maps.event.addListener(map, 'click', function(mouseEvent) {
        var latlng = mouseEvent.latLng;
        
        // 클릭한 위치에 마커 표시
        displayMarker(latlng, '<div style="padding:5px;"><strong>선택한 위치</strong></div>');
        
        // 위치 설정 안내 표시
        // 위치 설정 안내 제거됨 (주소 입력 우선 모드)
    });
}

// 위치 재요청 버튼 함수 제거됨 (주소 입력 우선 모드)

// 현재 위치 마커 제거 함수 (개선된 버전)
function removeCurrentLocationMarker() {
    if (currentLocationMarker) {
        currentLocationMarker.setMap(null);
        currentLocationMarker = null;
    }
}    