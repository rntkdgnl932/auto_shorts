import subprocess
import sys
import os
import ctypes


# -----------------------------
# 공통: 에러 메시지 박스 헬퍼
# -----------------------------
def show_error_box(msg_err: str, title: str = "Error") -> None:
    """
    PyInstaller EXE / 콘솔 유무에 상관없이
    윈도우 메시지 박스로 에러 내용을 표시하는 헬퍼.
    """
    try:
        # user32.dll 직접 로드
        user32 = ctypes.WinDLL("user32", use_last_error=True)

        # 시그니처 정의 (HWND, LPCWSTR, LPCWSTR, UINT)
        user32.MessageBoxW.argtypes = (
            ctypes.c_void_p,  # hWnd
            ctypes.c_wchar_p, # lpText
            ctypes.c_wchar_p, # lpCaption
            ctypes.c_uint     # uType
        )
        user32.MessageBoxW.restype = ctypes.c_int

        MB_OK = 0x00000000
        user32.MessageBoxW(None, msg_err, title, MB_OK)
    except Exception as ee:
        # MessageBoxW 자체가 실패한 경우: 최소한 stderr로라도 남기기
        print("MessageBoxW 호출 실패:", ee, file=sys.stderr)
        print(msg_err, file=sys.stderr)


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
venv_python = os.path.join(base_dir, ".venv", "Scripts", "python.exe")
venv_python_alt = os.path.join(base_dir, "venv", "Scripts", "python.exe")

if os.path.exists(venv_python):
    python_exe = venv_python  # 옆에 있는 .venv 사용
elif os.path.exists(venv_python_alt):
    python_exe = venv_python_alt
elif not getattr(sys, 'frozen', False):
    # 개발 중일 때는 지금 파이참이 쓰는 파이썬 사용
    python_exe = sys.executable
else:
    # EXE인데 옆에 .venv도 없으면 어쩔 수 없이 시스템 파이썬
    python_exe = "python"

# 4. 실행
try:
    subprocess.call([python_exe, target_script])
except Exception as e:
    msg = f"실행 실패!\n\n[시도한 파이썬 경로]\n{python_exe}\n\n[에러 내용]\n{e!r}"
    show_error_box(msg, "shorts_make 런처 오류")
