import subprocess
import sys
import os

# 1. 실행 위치 기준 잡기
if getattr(sys, 'frozen', False):
    # EXE 실행 중일 때: EXE 파일이 있는 폴더 기준
    base_dir = os.path.dirname(sys.executable)
else:
    # 파이참 테스트 중일 때: run.py 파일이 있는 폴더 기준
    base_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 실행할 목표 (main.py)
target_script = os.path.join(base_dir, "main.py")

# 3. [핵심] .venv 폴더 자동 감지 로직
# 우선순위 1: 내 옆에 .venv 폴더가 있나? (새 컴퓨터 배포 환경)
venv_python = os.path.join(base_dir, ".venv", "Scripts", "python.exe")

# 우선순위 2: 혹시 venv 라는 이름으로 만들었나?
venv_python_alt = os.path.join(base_dir, "venv", "Scripts", "python.exe")

if os.path.exists(venv_python):
    python_exe = venv_python  # 옆에 있는 .venv 사용 (성공!)
elif os.path.exists(venv_python_alt):
    python_exe = venv_python_alt
elif not getattr(sys, 'frozen', False):
    # 개발 중일 때는 지금 파이참이 쓰는 파이썬 사용
    python_exe = sys.executable
else:
    # EXE인데 옆에 .venv도 없으면 어쩔 수 없이 시스템 파이썬 사용
    python_exe = "python"

# 4. 실행
try:
    # console=True 대신 subprocess 설정으로 콘솔창 제어 가능
    subprocess.call([python_exe, target_script])
except Exception as e:
    import ctypes
    msg = f"실행 실패!\n\n[시도한 파이썬 경로]\n{python_exe}\n\n[에러 내용]\n{e}"
    ctypes.windll.user32.MessageBoxW(0, msg, "Error", 0)