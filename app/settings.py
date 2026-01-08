# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# ───────── 기본값 (하드코어 디폴트) ─────────
# 가사 분석 관련
# ===== Whisper / Sync 기본값 =====
WHISPER_MODEL_SIZE = "medium"   # tiny/small/medium/large
WHISPER_BEAM_SIZE = 5           # 3~10 권장
WHISPER_VAD_FILTER = False      # Windows 안정성 위해 False 권장
WHISPER_USE_VOCAL_SEP = False   # 보컬 분리 기본 off (느림)

MASTER_TARGET_I   = -12.0
MASTER_TARGET_TP  = -1.0
MASTER_TARGET_LRA = 11.0

# initial_prompt는 코드에서 80토큰/1000바이트로 자동 절삭

# 타임 배분/정렬 튜닝
SYNC_AVG_MIN_SEC_PER_UNIT = 2.0   # 라인당 최소 평균 길이
SYNC_END_BIAS_SEC = 2.5           # 끝쪽 여유
SYNC_START_PREROLL = 0.30         # 첫 온셋 앞 여유

# 길이 스냅(초) — 20s → 30s로 상향
ALLOWED_TARGET_SECONDS = (30, 60, 120, 180)
MIN_TARGET_SECONDS = 30


# (선택) 전처리
AUDIO_PRE_NORMALIZE = False       # 피크 노멀라이즈
AUDIO_PRE_HIGHPASS_HZ = 0         # 0=끄기, 80~120 권장


# ai 관련
OPENAI_MIN_BALANCE_USD: float = float(os.getenv("OPENAI_MIN_BALANCE_USD", "5"))
OPENAI_BALANCE_USD_ENV: str | None = os.getenv("OPENAI_BALANCE_USD")  # 선택: 수동 입력용
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# 쿠팡 관련
COUPANG_ACCESS_KEY = os.getenv("COUPANG_ACCESS_KEY")
COUPANG_SECRET_KEY = os.getenv("COUPANG_SECRET_KEY")
COUPANG_PARTNER_ID = os.getenv("COUPANG_PARTNER_ID")
COUPANG_SUB_ID = os.getenv("COUPANG_SUB_ID", "")

# 프로젝트 루트
BASE_DIR: Path = Path(r"C:\my_games\shorts_make\maked_title")
COMFY_RESULT_ROOT = r"C:\comfyResult\shorts_make"  # ComfyUI가 현재 저장하는 폴더(제목 하위)

# Ace-Step 요청 대기시간(초) / 폴링 간격(초)
ACE_WAIT_TIMEOUT_SEC = 1800    # 30분 (필요시 900→1800으로 늘려둠)
ACE_POLL_INTERVAL_SEC = 2.0
# 생성 파일 최종 저장 루트(제목 치환 지원)
FINAL_OUT: str = r"C:\my_games\shorts_make\maked_title\[title]"

# ComfyUI 로그 tail (원하면 환경변수로 덮어쓰기)
COMFY_LOG_FILE: str | None = os.getenv("COMFY_LOG_FILE", r"C:\comfy310\ComfyUI\user\comfyui.log")
CHARACTER_DIR = r"C:\my_games\shorts_make\character"
COMFY_INPUT_DIR   = r"C:\comfy310\ComfyUI\input"       # (원하면) Comfy input
COMFY_OUTPUT_DIR  = r"C:\my_games\shorts_make\comfy_output"      # (원하면) Comfy output
# ComfyUI 서버 주소
COMFY_HOST: str = os.environ.get("COMFY_HOST", "http://127.0.0.1:8188")

# ComfyUI 자동 탐색 후보 (기존 music_gen.py 내부에 있던 것 이동)
DEFAULT_HOST_CANDIDATES = [
    "http://127.0.0.1:8188",
    "http://localhost:8188",
    "http://127.0.0.1:8189",
    "http://localhost:8189",
    "http://127.0.0.1:8187",
    "http://localhost:8187",
]


# 경로들 (프롬프트/워크플로 JSON)
JSONS_DIR: Path = BASE_DIR.parent / "app" / "jsons"  # 기존 구조 유지
JSONS_DIR.mkdir(parents=True, exist_ok=True)




ACE_STEP_PROMPT_JSON: Path = JSONS_DIR / "ace_step_1_t2m.json"  # 음악 생성 프롬프트 JSON
# I2V_WORKFLOW:         Path = JSONS_DIR / "guff_movie.json"         # i2v 워크플로 JSON
I2V_WORKFLOW:         Path = JSONS_DIR / "No.48.WAN2.2-LightX2V-I2V.json"


# 음악생성 대기시간 등
_DEFAULT_ACE_WAIT_TIMEOUT_SEC = 900.0   # 15분
_DEFAULT_ACE_POLL_INTERVAL_SEC = 2.0    # 2초

_DEFAULT_WEIGHTS = {
    "verse": 1.0,
    "chorus": 1.2,
    "bridge": 1.0,
    "pre-chorus": 1.0,
    "intro": 0.6,
    "outro": 0.8,
    "unknown": 1.0,
}

SECTION_HEADER_RE = re.compile(r"^\s*\[(?P<name>[^]]+)]\s*$", re.IGNORECASE)

# Wan / SeedVR2 보간용 세그먼트 최대 프레임 수 (메모리 안전 상한)
# - 0 또는 음수면 "제한 없음"
# - 기본값 48 (너무 빡세면 40, 더 여유 있으면 56 등으로 조절 가능)
WAN_MAX_FRAMES_PER_SEGMENT: int = int(os.environ.get("WAN_MAX_FRAMES_PER_SEGMENT", "500"))


# 오디오/비디오 저장 관련
# - AUDIO_SAVE_FORMAT 은 "wav"|"mp3"|"opus" 중 하나
AUDIO_SAVE_FORMAT: str = os.environ.get("AUDIO_SAVE_FORMAT", "mp3").lower()

# ffmpeg/ffprobe 실행 파일 경로
FFMPEG_EXE:   str = os.environ.get("FFMPEG_EXE", "ffmpeg")
# FFPROBE_EXE: 명시되면 그대로 사용, 아니면 FFMPEG_EXE 옆의 ffprobe.exe를 추정, 끝으로 PATH의 ffprobe 사용
FFPROBE_EXE:  str = os.environ.get("FFPROBE_EXE", "")

USE_HWACCEL: bool = True


get_style_list = ["electronic","rock","pop","funk","soul","cyberpunk","acid jazz","edm","soft electric drums","melodic"]
get_scene_list = ["background music for parties","radio broadcasts","workout playlists"]
get_instr_list = ["saxophone","jazz","piano","violin","acoustic guitar","electric bass"]
get_tempo_list = ["110 bpm","fast tempo","slow tempo","loops","fills"]



# ───────── I2V (Image-to-Video) 생성 상수 (Wan 2.2 / ComfyUI) ─────────
# 영상 분할 생성 시 프레임 제어 상수
I2V_CHUNK_BASE_FRAMES = 82    # 기본 청크 길이 (유효 구간)
I2V_OVERLAP_FRAMES = 6        # 세그먼트 간 겹침/루프백 길이 (Cross-fade용)
I2V_PAD_TAIL_FRAMES = 5       # 생성 안정성을 위한 끝부분 여유분 (잘라낼 부분)


# ───────── 사용자 정의 UI 기본값 설정 ─────────

# 1. 렌더링 FPS (예: 24, 30, 60)
DEFAULT_MOVIE_FPS = 16

# 2. 이미지/영상 해상도 (가로, 세로)
DEFAULT_IMG_SIZE = (405, 720)

# 3. 이미지 생성 품질 (스텝 수)
DEFAULT_T2I_STEPS = 6

# ───────── 로컬 오버라이드 파일 ─────────
_SETTINGS_DIR: Path = BASE_DIR.parent / "app" / "_local"
_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_LOCAL_JSON: Path = _SETTINGS_DIR / "settings_local.json"


# ───────── 오버라이드 로직 ─────────
# settings.py 안에 이 3개 함수만 교체하세요.

def _apply_overrides(d: dict):
    """settings.py 모듈 전역에 덮어쓰기. 경로형은 Path로 재해석."""
    g = globals()
    for k, v in (d or {}).items():
        if not k.isupper():
            continue
        if k in g:
            if isinstance(v, str) and (k.endswith("_DIR") or k.endswith("_JSON") or k.endswith("_WORKFLOW")):
                try:
                    g[k] = Path(v)
                except Exception:
                    g[k] = v
            else:
                g[k] = v

def load_overrides() -> dict:
    """설정 파일을 읽어 settings 전역에 적용하고 dict 반환."""
    try:
        if SETTINGS_LOCAL_JSON.exists():
            with open(SETTINGS_LOCAL_JSON, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
                _apply_overrides(data)
                return data
    except Exception as e:
        # 문제 있으면 조용히 무시하지 말고 로그 남김
        print(f"[settings] load_overrides error: {e}")
    return {}

def save_overrides(**kwargs) -> Path:
    """
    주어진 키들만 settings_local.json에 반영하고,
    저장 직후 현재 런타임(settings 전역)에도 즉시 반영한다.
    반환: 실제 저장된 json 경로(Path)
    """
    _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)

    current = {}
    if SETTINGS_LOCAL_JSON.exists():
        try:
            with open(SETTINGS_LOCAL_JSON, "r", encoding="utf-8") as f:
                current = json.load(f) or {}
        except Exception:
            current = {}

    for k, v in kwargs.items():
        if not k.isupper():
            continue
        current[k] = str(v) if isinstance(v, Path) else v

    with open(SETTINGS_LOCAL_JSON, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2)

    # 저장 직후 런타임에도 적용
    _apply_overrides(current)
    print(f"[settings] overrides saved -> {SETTINGS_LOCAL_JSON}")
    return SETTINGS_LOCAL_JSON



# ───────── 모듈 임포트 시 오버라이드 자동 적용 ─────────
load_overrides()

# ───────── ffprobe 경로 보정(필요 시) ─────────
if not FFPROBE_EXE:
    # FFMPEG_EXE가 파일 경로로 지정되어 있으면 같은 폴더의 ffprobe를 추정
    try:
        ff = Path(FFMPEG_EXE)
        if ff.suffix.lower() == ".exe":
            probe_cand = ff.with_name("ffprobe.exe")
        else:
            probe_cand = ff.parent / "ffprobe"
        if probe_cand.exists():
            FFPROBE_EXE = str(probe_cand)
        else:
            FFPROBE_EXE = "ffprobe"
    except Exception:
        FFPROBE_EXE = "ffprobe"
