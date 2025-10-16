# check_env.py - .env 파일 디버깅 도구 (키 값은 숨김)

import os
from pathlib import Path
from dotenv import load_dotenv

print("=" * 60)
print("🔍 .env 파일 디버깅 도구")
print("=" * 60)

# 1. 현재 작업 디렉토리 확인
print(f"\n1️⃣ 현재 작업 디렉토리:")
print(f"   {os.getcwd()}")

# 2. .env 파일 위치 확인
env_paths = [
    Path(".env"),
    Path("../.env"),
    Path("Pill'sGood/.env"),
]

print(f"\n2️⃣ .env 파일 존재 여부:")
env_file_found = None
for env_path in env_paths:
    exists = env_path.exists()
    print(f"   {env_path}: {'✅ 존재' if exists else '❌ 없음'}")
    if exists and env_file_found is None:
        env_file_found = env_path

# 3. .env 파일 내용 확인 (키는 마스킹)
if env_file_found:
    print(f"\n3️⃣ .env 파일 내용 분석 ({env_file_found}):")
    try:
        with open(env_file_found, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"   총 {len(lines)}줄")
        print(f"   파일 인코딩: UTF-8 ✅")
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # 빈 줄
            if not line_stripped:
                print(f"   줄 {i}: (빈 줄)")
                continue
            
            # 주석
            if line_stripped.startswith('#'):
                print(f"   줄 {i}: {line_stripped}")
                continue
            
            # 환경 변수
            if '=' in line_stripped:
                key, value = line_stripped.split('=', 1)
                
                # 공백 체크
                has_leading_space = key != key.lstrip()
                has_trailing_space = key != key.rstrip()
                key_clean = key.strip()
                
                # 따옴표 체크
                has_quotes = value.startswith('"') or value.startswith("'")
                
                # 값 마스킹
                if value:
                    masked_value = value[:3] + '***' + value[-3:] if len(value) > 6 else '***'
                else:
                    masked_value = "(빈 값!)"
                
                print(f"   줄 {i}: {key_clean}={masked_value}")
                
                # 문제 체크
                if has_leading_space or has_trailing_space:
                    print(f"        ⚠️ 키 앞뒤에 공백 있음!")
                if has_quotes:
                    print(f"        ⚠️ 값에 따옴표 있음 (제거 필요)")
                if not value:
                    print(f"        ❌ 값이 비어있음!")
                
                # NAVER 관련 키 하이라이트
                if 'NAVER' in key_clean.upper():
                    print(f"        🎯 네이버 API 키 발견!")
            else:
                print(f"   줄 {i}: {line_stripped[:50]}... (형식 오류!)")
    
    except UnicodeDecodeError:
        print(f"   ❌ UTF-8 인코딩 오류! 파일을 UTF-8로 저장하세요.")
    except Exception as e:
        print(f"   ❌ 파일 읽기 오류: {e}")
else:
    print(f"\n❌ .env 파일을 찾을 수 없습니다!")
    print(f"   .env 파일을 다음 위치에 생성하세요:")
    print(f"   {Path.cwd() / '.env'}")

# 4. load_dotenv() 실행 후 환경 변수 확인
print(f"\n4️⃣ load_dotenv() 실행 후 환경 변수:")
load_dotenv(override=True)

naver_client_id = os.getenv("NAVER_CLIENT_ID")
naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")
openai_key = os.getenv("OPENAI_API_KEY")

print(f"   OPENAI_API_KEY: {'✅ 설정됨' if openai_key else '❌ 없음'}")
print(f"   NAVER_CLIENT_ID: {'✅ 설정됨 (' + naver_client_id[:5] + '***)' if naver_client_id else '❌ 없음'}")
print(f"   NAVER_CLIENT_SECRET: {'✅ 설정됨 (***' + naver_client_secret[-5:] + ')' if naver_client_secret else '❌ 없음'}")

# 5. 모든 환경 변수 중 NAVER 포함된 것
print(f"\n5️⃣ 환경 변수 중 NAVER 관련:")
naver_vars = {k: v for k, v in os.environ.items() if 'NAVER' in k.upper()}
if naver_vars:
    for key in naver_vars.keys():
        print(f"   - {key}: ✅ 설정됨")
else:
    print(f"   ❌ NAVER 관련 환경 변수가 하나도 없습니다!")

print("\n" + "=" * 60)
print("🎯 결론:")
if naver_client_id and naver_client_secret:
    print("✅ 네이버 API 키가 올바르게 설정되었습니다!")
else:
    print("❌ 네이버 API 키가 환경 변수에 로드되지 않았습니다.")
    print("\n📋 체크리스트:")
    print("   1. .env 파일이 올바른 위치에 있는가?")
    print("   2. .env 파일에 NAVER_CLIENT_ID=값 형식으로 작성했는가?")
    print("   3. 키와 값 사이에 공백이 없는가?")
    print("   4. 값에 따옴표가 없는가?")
    print("   5. 파일이 UTF-8로 저장되었는가?")
print("=" * 60)

