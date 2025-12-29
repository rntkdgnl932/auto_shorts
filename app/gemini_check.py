import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. .env 파일 로드
load_dotenv()

# 2. 환경변수에서 키 가져오기
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# [중요] 키가 제대로 불러와졌는지 확인
if not GEMINI_API_KEY:
    print("❌ 오류: .env 파일을 찾을 수 없거나 GEMINI_API_KEY가 없습니다.")
    print("      이 파일 안에 API 키를 직접 적어서 다시 시도해보세요.")
else:
    # 3. 가져온 키로 설정 (여기가 중요합니다!)
    genai.configure(api_key=GEMINI_API_KEY)

    print(f"🔑 감지된 키: {GEMINI_API_KEY[:5]}... (앞 5자리만 표시)")
    print("------ [내 API 키로 사용 가능한 모델 목록] ------")

    try:
        found = False
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"모델명: {m.name}")
                found = True

        if not found:
            print("⚠️ 목록 조회는 성공했으나, 'generateContent' 지원 모델이 없습니다.")
            print("   라이브러리 업데이트가 필요할 수 있습니다.")

    except Exception as e:
        print(f"❌ 목록 조회 실패: {e}")
        print("   (API 키가 틀렸거나, 인터넷 연결, 혹은 라이브러리 버전 문제일 수 있습니다.)")

    print("------------------------------------------------")