﻿# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
from pathlib import Path

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

# UI 기본값 (영상 분할/프레임)
DEFAULT_CHUNK: int = 300
DEFAULT_OVERLAP: int = 5
DEFAULT_INPUT_FPS: int = 60
DEFAULT_TARGET_FPS: int = 60

# 경로들 (프롬프트/워크플로 JSON)
JSONS_DIR: Path = BASE_DIR.parent / "app" / "jsons"  # 기존 구조 유지
JSONS_DIR.mkdir(parents=True, exist_ok=True)

ACE_STEP_PROMPT_JSON: Path = JSONS_DIR / "ace_step_1_t2m.json"  # 음악 생성 프롬프트 JSON
I2V_WORKFLOW:         Path = JSONS_DIR / "guff_movie.json"         # i2v 워크플로 JSON

# 오디오/비디오 저장 관련
# - AUDIO_SAVE_FORMAT 은 "wav"|"mp3"|"opus" 중 하나
AUDIO_SAVE_FORMAT: str = os.environ.get("AUDIO_SAVE_FORMAT", "mp3").lower()

# ffmpeg/ffprobe 실행 파일 경로
FFMPEG_EXE:   str = os.environ.get("FFMPEG_EXE", "ffmpeg")
# FFPROBE_EXE: 명시되면 그대로 사용, 아니면 FFMPEG_EXE 옆의 ffprobe.exe를 추정, 끝으로 PATH의 ffprobe 사용
FFPROBE_EXE:  str = os.environ.get("FFPROBE_EXE", "")

USE_HWACCEL: bool = True


# FILE: settings.py
# INSERT AFTER: the block where these already exist:
#   DEFAULT_CHUNK: int = 300
#   DEFAULT_OVERLAP: int = 12
#   DEFAULT_INPUT_FPS: int = 60
#   DEFAULT_TARGET_FPS: int = 60
#
# PURPOSE: 드롭다운에서 선택할 이미지 크기/FPS 후보와 기본값,
#          타임라인 보정 파라미터를 '설정'으로 명확히 둡니다.

# 1) 이미지 기본 크기 (가로, 세로)
#    예: (832, 1472)는 9:16 세로형에 가깝습니다. (w, h)
DEFAULT_IMG_SIZE: tuple[int, int] = (720, 1080)

# 드롭다운 후보(원하는 값 추가/삭제 가능)
IMAGE_SIZE_CHOICES: list[int] = [240, 480, 520, 720, 960, 1080, 1280, 1440, 1920]

# 2) 렌더 FPS 기본/후보 (24 또는 60 선택)
DEFAULT_MOVIE_FPS: int = 30
MOVIE_FPS_CHOICES: list[int] = [24, 30, 60]

# 3) 청크 오버랩(프레임) — i2v를 여러 덩어리로 만들 때 경계 끊김을 줄이기 위해
#    앞/뒤 청크가 겹치는 프레임 수. 60fps에서 12프레임은 약 0.2초.
DEFAULT_MOVIE_OVERLAP: int = 5

# 4) 최소 장면 길이(초) — 분석/보정 시 너무 짧은 컷을 '삭제'가 아니라
#    '최소 길이로 승격'하는 기준값. (삭제로 바꾸고 싶으면 알려주세요.)
MIN_SCENE_SEC_DEFAULT: float = 0.20

# 5) 시간 반올림 자릿수 — start/end/duration을 소수점 몇 자리까지 반올림할지
ROUND_SEC_DEFAULT: int = 3

# 6) 전역 네거티브 프롬프트 — 품질 저하 요소를 한 번에 관리
NEGATIVE_BANK_DEFAULT: str = (
    "text, letters, typography, watermark, logo, signature, caption, subtitles, closed captions, "
    "korean letters, hangul, hangeul, handwriting, calligraphy, "
    "텍스트, 글자, 글씨, 문구, 활자, 자막, 캡션, 로고, 워터마크, 서명, 낙관, 표지판, 간판, 스티커, "
    "low quality, worst quality, lowres, jpeg artifacts, noise, grainy, overexposed, underexposed, "
    "artifact, compression artifacts, deformed hands, extra fingers, fused fingers, long fingers, "
    "mutated hands, deformed face, distorted face, ugly face, asymmetry, "
    "nsfw, nude, nudity, sexual content"
)


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
