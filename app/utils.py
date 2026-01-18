# -*- coding: utf-8 -*-
"""
공용 유틸 함수 모음 (외부 의존성 최소화)
- 프로젝트 폴더 규칙: BASE_DIR / "maked_title" / <title>
- 결과 복사 규칙: settings.FINAL_OUT 경로의 [title] 치환
"""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
import importlib.util
import shutil
import time
import subprocess
import math
from typing import List, Tuple, Callable, Optional, Any, Dict, Union, Iterable, cast
from app import settings as settings
from dataclasses import dataclass
from typing import Literal, Sequence
import re, json
from dotenv import load_dotenv
from pydub import AudioSegment  # type: ignore
from mutagen import File as MutagenFile
from types import SimpleNamespace
from google.generativeai.types import GenerationConfig
from pathlib import Path
import io, traceback
import os
from os import PathLike
from PyQt5 import QtWidgets, QtCore
import requests

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # 설치 전이라면 None으로 두고 에러 메시지에서 안내

# ── Gemini SDK (선택) ────────────────────────────────────────────────
try:
    import google.generativeai as genai
    try:
        from google.api_core.exceptions import GoogleAPIError  # type: ignore
    except Exception:
        GoogleAPIError = Exception  # type: ignore
except Exception:
    genai = None
    GoogleAPIError = Exception  # type: ignore

# OpenAI 예외 (결제/한도 감지용)
try:
    from openai import BadRequestError
except Exception:
    class BadRequestError(Exception):
        pass
from app.settings import (
    BASE_DIR, COMFY_HOST, DEFAULT_HOST_CANDIDATES, _DEFAULT_ACE_WAIT_TIMEOUT_SEC, _DEFAULT_ACE_POLL_INTERVAL_SEC, _DEFAULT_WEIGHTS,
    ACE_STEP_PROMPT_JSON, FFMPEG_EXE, FINAL_OUT, SECTION_HEADER_RE
)
from app.settings import BASE_DIR

# 자막 관련

_FONTFILE_CACHE: Dict[str, Optional[str]] = {}

# --- 한글→영문 매핑 ---
_KO2EN_CHAR: Dict[str, str] = {
    "긴 웨이브펌": "long wavy hair",
    "갈색 머리": "brown hair",
    "검정 머리": "black hair",
    "짧은 머리": "short hair",
    "슬림 체형": "slim body",
    "넓은 어깨": "broad shoulders",
    "일관성 유지": "consistent appearance",
}

_KO2EN_STYLE: Dict[str, str] = {
    "서정": "lyrical mood",
    "도시": "urban",
    "야간": "night",
    "희미한 추억": "faint nostalgia",
    "실사": "photorealistic",
    "정면": "facing camera",
    "정면 얼굴": "facing camera",
    "자연스러운 조명": "natural light",
    "보케": "bokeh",
    "도입": "intro",
    "암시": "hint",
    "미니멀": "minimal",
    "잔잔함": "calm",
    "근접": "close-up",
    "친밀감": "intimate",
    "개방감": "open space",
    "광각": "wide angle",
    "확장": "expansive",
    "전환감": "transition",
    "대비": "contrast",
    "변화": "change",
    "여운": "lingering",
    "잔상": "afterimage",
    "감쇠": "decay",
}

_EFFECT_MAP: Dict[str, str] = {
    "soft light": "soft light",
    "film grain": "film grain",
    "gentle camera pan": "gentle camera pan",
    "bokeh": "bokeh",
    "slow push-in": "slow push-in",
    "bloom": "bloom",
    "wide angle": "wide angle",
}

_NEGATIVE_MAP: Dict[str, str] = {
    "손가락 왜곡": "deformed hands",
    "눈 왜곡": "distorted eyes",
    "과도한 보정": "overprocessed",
    "노이즈": "noisy",
    "흐릿함": "blurry",
    "텍스트 워터마크": "watermark, text",
}
_DEFAULT_NEGATIVE: List[str] = [
    "lowres", "bad anatomy", "bad proportions", "extra limbs", "extra fingers",
    "missing fingers", "jpeg artifacts", "signature", "logo", "nsfw"
]

def _ko_to_en_tokens(text: str) -> List[str]:
    toks: List[str] = []
    for token in re.split(r"[,\s/+]+", (text or "").strip()):
        if not token:
            continue
        toks.append(_KO2EN_STYLE.get(token, token))
    return toks

def _parse_char_refs(char_refs: List[str]) -> List[Dict[str, int]]:
    out: List[Dict[str, int]] = []
    for ref in (char_refs or []):
        if ":" in ref:
            i = ref.index(":")
            cid = ref[:i]
            try:
                idx = int(ref[i+1:])
            except Exception:
                idx = 0
        else:
            cid, idx = ref, 0
        out.append({"id": cid, "index": idx})
    return out

def _effects_to_tags(effects: List[str]) -> List[str]:
    out: List[str] = []
    for e in (effects or []):
        e = (e or "").strip()
        if not e:
            continue
        out.append(_EFFECT_MAP.get(e, e))
    return out

def _compose_negative(global_ctx: Dict[str, Any]) -> str:
    items: List[str] = []
    neg_src = (global_ctx or {}).get("negative_bank", "")
    if neg_src:
        parts = [s.strip() for s in re.split(r"[,\s/]+", neg_src) if s.strip()]
        for p in parts:
            items.append(_NEGATIVE_MAP.get(p, p))
    items.extend(_DEFAULT_NEGATIVE)
    uniq, seen = [], set()
    for t in items:
        if t and t not in seen:
            uniq.append(t); seen.add(t)
    return ", ".join(uniq)

def _section_mood_tags(section: str, global_ctx: Dict[str, Any]) -> List[str]:
    moods = (global_ctx or {}).get("section_moods", {}) or {}
    m = moods.get(section, "")
    return _ko_to_en_tokens(m) if m else []

def _themes_palette_tags(global_ctx: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    themes = (global_ctx or {}).get("themes", []) or []
    for t in themes:
        if t:
            tags.append(_KO2EN_STYLE.get(t, t))
    palette = (global_ctx or {}).get("palette", "")
    if palette:
        palette = palette.replace("+", " ")
        tags.extend(_ko_to_en_tokens(palette))
    style_guide = (global_ctx or {}).get("style_guide", "")
    if style_guide:
        tags.extend(_ko_to_en_tokens(style_guide))
    uniq, seen = [], set()
    for t in tags:
        if t and t not in seen:
            uniq.append(t); seen.add(t)
    return uniq

def _extract_char_tags(char_id: str, style_str: str) -> List[str]:
    base: List[str] = []
    if char_id.startswith("female"):
        base.append("young woman")
        # [수정] UI에서 설정한 환경 변수가 '1'일 때만 주입
        if os.environ.get("FORCE_HUGE_BREASTS") == "1":
            base.append("huge breasts")
    elif char_id.startswith("male"):
        base.append("young man")
    else:
        base.append("person")
    if style_str:
        parts = [s.strip() for s in re.split(r"[,\s/]+", style_str) if s.strip()]
        for p in parts:
            base.append(_KO2EN_CHAR.get(p, p))
    uniq, seen = [], set()
    for t in base:
        if t and t not in seen:
            uniq.append(t); seen.add(t)
    return uniq


def _build_scene_prompts(scene: Dict[str, Any], story: Dict[str, Any]) -> Tuple[str, str]:
    """
    [수정됨] story.json의 정보를 조합하여 최종 영어 태그 기반 프롬프트를 생성합니다.
    - 네거티브 프롬프트('--neg')를 여기서 덧붙이지 않습니다. (story_enrich 단계에서 prompt_negative 필드에 저장)
    - 조합 로직은 기존 방식을 유지하되, 네거티브 부분만 제거합니다.
    """
    global_ctx = story.get("global_context", {}) or {}
    char_styles = story.get("character_styles", {}) or {}

    # 네거티브 조합 로직은 유지 (하지만 최종 프롬프트 문자열에 --neg로 붙이지 않음)
    # negative_text = _compose_negative(global_ctx) # 필요 시 내부 로깅 등에 사용 가능

    effects = _effects_to_tags(scene.get("effect", []))
    section_tags = _section_mood_tags(scene.get("section", ""), global_ctx)
    theme_palette = _themes_palette_tags(global_ctx)

    # 캐릭터 태그
    char_tags: List[str] = []
    char_refs = _parse_char_refs(scene.get("characters", []))
    for ref in char_refs:
        rid_str = str(ref.get("id", ""))
        ctags = _extract_char_tags(rid_str, char_styles.get(rid_str, ""))
        if isinstance(ctags, list):
            char_tags.extend(ctags)
        elif ctags:
            char_tags.append(str(ctags))

    # 배경 힌트 (기존 로직 유지)
    bg_hint = ""
    if scene.get("prompt_img"): # scene['prompt_img']가 AI 강화 후 이미 영어 태그일 수 있음
        # 간단히 첫 5개 단어 정도만 배경 힌트로 간주 (더 정교한 추출 필요 시 개선)
        bg_tags_candidate = str(scene["prompt_img"]).split(',')[:5]
        # 한국어 설명 필드('prompt')가 있으면 그것을 우선 사용
        prompt_ko = str(scene.get("prompt", "")).strip()
        if prompt_ko:
             # 간단히 첫 부분만 사용
             bg_hint = prompt_ko.split('.')[0] # 첫 문장 정도
        else:
             bg_hint = " ".join(bg_tags_candidate)

    bg_tags = _ko_to_en_tokens(bg_hint) if bg_hint else []

    # 이미지 프롬프트 조합 (기존 방식 유지)
    img_prompt_parts = [
        ", ".join(char_tags) if char_tags else None,
        ", ".join(bg_tags) if bg_tags else None,
        ", ".join(theme_palette) if theme_palette else None,
        ", ".join(section_tags) if section_tags else None,
        ", ".join(effects) if effects else None,
        "photorealistic, cinematic lighting, high detail, 8k, masterpiece", # 품질 태그
    ]
    img_prompt = ", ".join([p for p in img_prompt_parts if p])
    # [수정] 네거티브 프롬프트(--neg) 추가 로직 제거

    # 무비 프롬프트 (모션 힌트 추가 로직 유지)
    motion_hint = ""
    # scene['prompt_movie']에서 모션 힌트 추출 시도 (AI 강화 단계에서 이미 추가되었을 수 있음)
    raw_movie_prompt = scene.get("prompt_movie", "")
    if isinstance(raw_movie_prompt, str):
        # 'motion:' 또는 'camera:' 키워드 이후 부분 추출
        motion_match = re.search(r'(motion:|camera:)(.*)', raw_movie_prompt, re.IGNORECASE)
        if motion_match:
            motion_hint = motion_match.group(2).strip()

    # [수정] 네거티브 프롬프트(--neg) 추가 로직 제거
    movie_prompt = f"{img_prompt}, motion: {motion_hint}" if motion_hint else img_prompt

    # 최종 반환 시 앞뒤 공백 제거 및 중복 쉼표 정리
    img_prompt = re.sub(r'\s*,\s*', ', ', img_prompt).strip(', ')
    movie_prompt = re.sub(r'\s*,\s*', ', ', movie_prompt).strip(', ')

    return img_prompt, movie_prompt


# real_use
def save_story_overwrite_with_prompts(story_path: Path) -> Path:
    """
    [수정됨 v2] story.json을 읽어 audit 정보만 추가하고 같은 파일에 덮어쓴다.
    - 프롬프트(prompt_img/prompt_movie) 재작성 로직을 완전히 제거합니다.
      (프롬프트 생성 책임은 story_enrich.py의 apply_gpt_to_story_v11로 일원화)
    - 실패 시 예외를 그대로 올립니다.
    """
    print(f"[PROMPTS_SAVE] load → {story_path}", flush=True)
    try:
        story_raw = load_json(story_path, None)
        if not isinstance(story_raw, dict):
             # 로드 실패 또는 형식이 dict가 아니면 오류 발생
             raise ValueError(f"Failed to load or parse story.json as dict: {story_path}")
        story = story_raw # 타입 검증 후 할당
    except Exception as e_load:
        print(f"[PROMPTS_SAVE] ERROR loading story.json: {e_load}", flush=True)
        raise # 오류를 상위로 전파

    scenes = story.get("scenes", []) or []
    print(f"[PROMPTS_SAVE] scenes={len(scenes)} | chars={story.get('characters', [])}", flush=True)

    # audit 정보만 추가 (덮어쓰기 여부, 시간)
    story.setdefault("audit", {})
    story["audit"]["prompts_finalized_at_enrich"] = True # 플래그 이름 변경 (enrich 단계에서 완료됨을 명시)
    story["audit"]["finalized_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        # save_json 함수를 사용하여 안전하게 저장 시도
        save_json(story_path, story) # save_json은 Path 객체를 반환하지 않으므로 수정
        print(f"[PROMPTS_SAVE] wrote (audit updated) → {story_path}", flush=True)
        return story_path # 저장 성공 시 경로 반환
    except Exception as e_save:
        print(f"[PROMPTS_SAVE] ERROR writing story.json: {e_save}", flush=True)
        raise # 오류를 상위로 전파

# ===== END =====

S = settings  # noqa: N816

# 선택 의존성: mutagen (MP3 등 길이 읽기)
_MUTAGEN_AVAILABLE = importlib.util.find_spec("mutagen") is not None
if _MUTAGEN_AVAILABLE:
    from mutagen import File as _MutagenFile  # type: ignore
else:
    _MutagenFile = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# 경로/파일 기본 유틸
def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def ensure_dir(p: os.PathLike | str) -> Path:
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d

# real_use
def load_json(p: os.PathLike | str, default: Optional[Any] = None) -> Union[Dict[str, Any], List[Any], None]:
    """
    JSON 파일을 로드합니다. 객체(dict) 또는 배열(list)을 반환할 수 있습니다.
    파일이 없거나 오류 발생 시 default 값을 반환합니다 (기본값은 None).
    - [수정] 반환 타입을 Union[Dict, List, None]으로 명시하여 seg.json 같은 리스트 루트 파일 지원.
    - [수정] default 반환 시 None 대신 {}을 반환하던 것을 default 인자(기본 None)를 따르도록 수정.
    """
    path = Path(p)
    if path.is_dir():
        # 폴더 경로에 대한 오류 처리를 명확하게 유지
        raise IsADirectoryError(f"load_json: 제공된 경로는 폴더입니다: {path}")
    if not path.exists():
        # 파일이 없으면 명시적으로 default 반환 (기본값은 None)
        return default

    try:
        with path.open("r", encoding="utf-8") as f:
            # json.load는 dict 또는 list 등 다양한 타입을 반환할 수 있음
            content: Any = json.load(f)
            # 로드된 내용이 dict 또는 list인 경우만 그대로 반환
            if isinstance(content, (dict, list)):
                return content
            else:
                # 예상치 못한 타입(예: 숫자, 문자열)이면 default 반환
                return default
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        # 파일 읽기 또는 JSON 파싱 오류 시 default 반환
        print(f"[WARN] load_json 실패 ({path.name}): {type(e).__name__}. 기본값 반환.") # 오류 로그 추가 (선택 사항)
        return default
    except Exception as e_unexpected:
        # 기타 예외 처리
        print(f"[ERROR] load_json 중 예상치 못한 오류 ({path.name}): {type(e_unexpected).__name__}. 기본값 반환.")
        return default

# real_use
def save_json(p: os.PathLike | str, obj: Any) -> Path:
    """
    JSON 객체(dict 또는 list)를 파일에 저장합니다.
    - [수정] obj 타입을 Any로 변경하여 리스트도 저장할 수 있도록 수정했습니다.
    """
    path = Path(p)
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path

# real_use
def write_text(p: os.PathLike | str, text: str) -> Path:
    path = Path(p)
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 문자열/날짜
# real_use
def sanitize_title(s: str) -> str:
    """윈도우 금지문자 제거 + 공백 정돈"""
    bad = r'\/:*?"<>|'
    out = "".join(c for c in (s or "").strip() if c not in bad)
    out = " ".join(out.split())
    return out or "untitled"
# real_use
def now_kr(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    kst = timezone(timedelta(hours=9))
    return datetime.now(tz=kst).strftime(fmt)

# real_use
def today_str(fmt: str = "%Y-%m-%d") -> str:
    return datetime.now().strftime(fmt)


# ─────────────────────────────────────────────────────────────────────────────




def _normalize_maked_title_root(base: Path) -> Path:
    """
    BASE_DIR 안에 'maked_title'이 하나만 들어가도록 정규화한다.
    - BASE_DIR에 이미 포함되어 있으면 추가로 붙이지 않음
    - 실수로 두 번 들어가 있으면 한 번만 남김
    """
    base = Path(base).resolve()
    parts = list(base.parts)
    # 'maked_title'이 들어있는지 찾기
    idxs = [i for i, p in enumerate(parts) if p.lower() == "maked_title"]
    if not idxs:
        # 없다면 한 번만 붙인다
        return base / "maked_title"
    # 있다면 첫 번째 것까지만 남기고 그 뒤의 중복은 제거
    first = idxs[0]
    return Path(*parts[:first + 1])

# real_use
def ensure_project_dir(title: str) -> Path:
    """
    프로젝트 폴더를 반환(없으면 생성). 항상:
      <BASE_DIR>/maked_title/<제목>
    형태가 되도록 보장한다. 'maked_title' 중복 방지.
    """


    # BASE_DIR 정규화 → maked_title 1회만 유지
    root = _normalize_maked_title_root(Path(getattr(S, "BASE_DIR", ".")))

    # 제목은 파일명 안전하게
    def _sanitize_title_for_path(s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"[\\/:*?\"<>|\r\n\t]+", "_", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s or "untitled"

    p = root / _sanitize_title_for_path(title)
    p.mkdir(parents=True, exist_ok=True)
    return p



# ─────────────────────────────────────────────────────────────────────────────

# 오디오 길이
def _wav_duration_sec(path: Path) -> float:
    import wave
    with wave.open(str(path), "rb") as w:
        frames, rate = w.getnframes(), w.getframerate()
        return 0.0 if rate <= 0 else frames / float(rate)
# real_use
def audio_duration_sec(path: os.PathLike | str) -> float:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return 0.0

    # WAV는 표준 라이브러리
    if p.suffix.lower() == ".wav":
        try:
            return _wav_duration_sec(p)
        except Exception:
            return 0.0

    # mutagen 있으면 MP3/FLAC 등
    if callable(_MutagenFile):
        try:
            m = _MutagenFile(str(p))
            length = getattr(getattr(m, "info", None), "length", None)
            return float(length) if isinstance(length, (int, float)) else 0.0
        except Exception:
            return 0.0

    return 0.0






# ─────────────────────────────────────────────────────────────────────────────
# 결과 복사: FINAL_OUT 의 [title] 치환 규칙 사용
def _resolve_out_dir_from_template(root_or_tpl: str, title: str) -> Path:
    tpl = (root_or_tpl or "").strip()
    safe_title = sanitize_title(title or "untitled")
    if not tpl:
        return Path()
    return Path(tpl.replace("[title]", safe_title)).resolve()

def save_to_user_library(kind: str, title: str, src_path: Path, *,
                         root: str | None = None, rename: bool = True) -> Path:
    """
    kind : 'audio' | 'video' | 'image' ...
    title: 프로젝트 타이틀 (폴더명 치환에 사용)
    src_path: 복사할 원본 파일
    root : 기본 None → settings.FINAL_OUT 사용
    rename: True면 kind별 관례 이름(vocal.mp3 / final.mp4 / cover.png 등)으로 저장
    """
    src_path = Path(src_path)
    if not src_path.exists():
        raise FileNotFoundError(f"원본이 없습니다: {src_path}")

    root_tpl = (root if root is not None else getattr(S, "FINAL_OUT", "") or "").strip()
    if not root_tpl:
        # 지정이 없으면 복사 생략 (원본 경로 반환)
        return src_path

    base_dir = _resolve_out_dir_from_template(root_tpl, title)
    base_dir.mkdir(parents=True, exist_ok=True)

    # 이름 규칙
    default_stem = {"audio": "vocal", "video": "final", "image": "cover"}.get(kind, "file")
    dst_name = f"{default_stem}{src_path.suffix.lower()}" if rename else src_path.name
    dst_path = base_dir / dst_name

    shutil.copy2(src_path, dst_path)  # 메타데이터 보존 복사
    try:
        print("[MUSIC] COPIED", str(src_path), "->", str(dst_path), flush=True)
    except Exception:
        pass
    return dst_path

def effective_title(meta_or_title) -> str:
    """
    project.json 메타(dict) 또는 문자열을 받아 '표준화된 제목'을 반환.
    우선순위:
      1) dict라면 meta["title"]
      2) dict라면 meta["paths"]["project_dir"] 폴더명
      3) 문자열이면 그대로
    모두 없으면 'untitled'
    """
    # dict(meta)인 경우
    if isinstance(meta_or_title, dict):
        t = (meta_or_title.get("title") or "").strip()

        if not t:
            # paths.project_dir가 있으면 폴더명 사용
            paths = meta_or_title.get("paths") or {}
            proj_dir = paths.get("project_dir") or ""
            if proj_dir:
                try:
                    t = Path(proj_dir).name
                except Exception:
                    t = ""

        return sanitize_title(t or "untitled")

    # 문자열/기타인 경우
    return sanitize_title(str(meta_or_title) if meta_or_title is not None else "untitled")

# ==== PATCH: scene default normalizer ====

def ensure_scene_defaults(scene: Dict[str, Any]) -> Dict[str, Any]:
    """story.json의 scene에 누락된 선택항목을 기본값으로 채운다."""
    sc = dict(scene)

    # 더 이상 강제하지 않지만, 과거 코드가 참조할 수 있으므로 기본 False로 보정
    if "needs_character_asset" not in sc:
        sc["needs_character_asset"] = False

    # 안전판: 자주 쓰는 선택 필드들 기본값도 함께 보정 (없으면 빈 값)
    sc.setdefault("prompt_img", "")
    sc.setdefault("prompt_movie", "")
    sc.setdefault("prompt_negative", "")
    sc.setdefault("characters", [])
    sc.setdefault("character_objs", [])
    sc.setdefault("effect", [])
    sc.setdefault("screen_transition", False)
    sc.setdefault("img_file", "")
    sc.setdefault("clip_file", "")

    return sc
# ==== PATCH END ====
# ==== PATCH: story scene auto-fix for legacy keys ====


def autofix_story_scene_flags(story_path: str | Path, *, save: bool = True) -> Dict[str, Any]:
    """
    story.json을 로드해 각 scene에 누락된 레거시 플래그를 기본값으로 보정한다.
    - needs_character_asset: 기본 False
    - prompt_img/prompt_movie/prompt_negative 등 선택 필드도 없으면 빈 값으로 채움(안전판)
    반환: 보정 적용된 story dict
    """
    p = Path(story_path)
    if not p.exists():
        raise FileNotFoundError(f"story.json not found: {p}")

    story = load_json(p, {})
    scenes: List[Dict[str, Any]] = list(story.get("scenes") or [])
    if not scenes:
        return story

    fixed = False
    for sc in scenes:
        # 레거시 강제 필드 기본값
        if "needs_character_asset" not in sc:
            sc["needs_character_asset"] = False
            fixed = True
        # 자주 참조되는 선택 필드 기본값(없으면 빈 값으로)
        sc.setdefault("prompt_img", "")
        sc.setdefault("prompt_movie", "")
        sc.setdefault("prompt_negative", "")
        sc.setdefault("characters", [])
        sc.setdefault("character_objs", [])
        sc.setdefault("effect", [])
        sc.setdefault("screen_transition", False)
        sc.setdefault("img_file", "")
        sc.setdefault("clip_file", "")

    if fixed and save:
        story["scenes"] = scenes
        save_json(p, story)
    return story
# ==== PATCH END ====
# ==== PATCH: purge legacy scene keys ====


# utils.py (새로 추가)
def purge_legacy_scene_keys(story: dict) -> dict:
    s = dict(story or {})
    scs = s.get("scenes") or []
    clean = []
    for sc in scs:
        d = dict(sc or {})
        # 더 이상 쓰지 않는 키 제거
        for k in ("needs_character_asset",):
            if k in d:
                d.pop(k, None)
        clean.append(d)
    s["scenes"] = clean
    return s


def purge_legacy_scene_keys_in_file(story_path: str | Path, *, save: bool = True) -> Dict[str, Any]:
    """파일에서 story.json을 읽어 레거시 키를 제거하고(있으면), 선택적으로 저장한다."""
    p = Path(story_path)
    story = load_json(p, {})
    new_story = purge_legacy_scene_keys(story)
    if save and new_story is not story:
        save_json(p, new_story)
    return new_story


# ── settings 유연 임포트 ─────────────────────────────────────────────
# 권장: ai.py의 임포트 블록 치환


# (선택) 하위 호환
S = settings

def _load_env_once():
    """가급적 .env를 자동 로드. 이미 값이 있으면 덮어쓰지 않음."""
    if any(os.getenv(k) for k in ("OPENAI_API_KEY", "OPENAI_APIKEY", "OPENAI_KEY")):
        return
    # 후보 경로들: CWD, settings.BASE_DIR, ai.py 상위, 그 조상 경로들
    candidates: List[Path] = []
    try:
        candidates.append(Path.cwd() / ".env")
    except Exception:
        pass
    if S and getattr(S, "BASE_DIR", None):
        candidates.append(Path(getattr(S, "BASE_DIR")) / ".env")  # type: ignore
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        candidates.append(p / ".env")
    # 중복 제거 + 존재하는 파일만
    seen, targets = set(), []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c.exists():
            targets.append(c)
    if load_dotenv:
        for dotenv_path in targets:
            try:
                load_dotenv(dotenv_path=dotenv_path, override=False)
            except Exception:
                pass
    else:
        # 폴백: 매우 단순 파서 (KEY=VALUE 줄만)
        for dotenv_path in targets:
            try:
                for line in dotenv_path.read_text(encoding="utf-8").splitlines():
                    if not line or line.strip().startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        os.environ.setdefault(k, v)
            except Exception:
                pass

_load_env_once()

# ── OpenAI SDK ───────────────────────────────────────────────────────


Provider = Literal["openai", "gemini"]

# ── 설정 컨테이너 ────────────────────────────────────────────────────
@dataclass
class AIConfig:
    provider: Provider = "openai"

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_model: str = "gpt-5-mini"

    # (옵션) Gemini
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.0-flash"

    def resolved(self) -> "AIConfig":
        def pick(*cands):
            for c in cands:
                if c:
                    return c
            return None

        provider = pick(
            getattr(S, "AI_PROVIDER", None) if S else None,
            os.getenv("AI_PROVIDER"),
            self.provider,
        )
        # OpenAI 키: 여러 이름 허용
        openai_api_key = pick(
            self.openai_api_key,
            getattr(S, "OPENAI_API_KEY", None) if S else None,
            os.getenv("OPENAI_API_KEY"),
            os.getenv("OPENAI_APIKEY"),
            os.getenv("OPENAI_KEY"),
        )
        return AIConfig(
            provider = (provider or "openai").lower(),  # type: ignore
            openai_api_key = openai_api_key,
            openai_base_url = pick(
                self.openai_base_url,
                getattr(S, "OPENAI_BASE_URL", None) if S else None,
                os.getenv("OPENAI_BASE_URL"),
            ),
            openai_model = pick(
                self.openai_model,
                getattr(S, "OPENAI_MODEL", None) if S else None,
                os.getenv("OPENAI_MODEL"),
                "gpt-5-mini",
            ),
            gemini_api_key = pick(
                self.gemini_api_key,
                getattr(S, "GEMINI_API_KEY", None) if S else None,
                os.getenv("GEMINI_API_KEY"),
            ),
            gemini_model = pick(
                self.gemini_model,
                getattr(S, "GEMINI_MODEL", None) if S else None,
                os.getenv("GEMINI_MODEL"),
                "gemini-2.0-flash",
            ),
        )

# ── 본체 ─────────────────────────────────────────────────────────────
# real_use
class AI:
    def __init__(self, cfg: AIConfig | None = None):
        self.cfg = (cfg or AIConfig()).resolved()
        self._openai = None
        self.default_prefer = (self.cfg.provider or 'openai').lower()
        self._gemini_ready = False
        self._init_clients()

        self.default_prefer = os.getenv("AI_PREFER", "openai").lower()  # "openai" / "gemini"
        self.gemini_model = getattr(self.cfg, "gemini_model", None) or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self._gemini_configured = False

    def _init_clients(self):
        """
        OpenAI/Gemini 클라이언트 초기화.
        - httpx가 OpenAI SDK 요구 버전보다 낮으면 OpenAI SDK 인스턴스를 '시도'조차 하지 않고
          REST 셤(Shim)으로 즉시 폴백하여 httpx 내부 __del__ 경고를 원천 차단한다.
        - 기존 호출부 호환: res.choices[0].message.content 형태 유지.
        """

        # ---- httpx 호환성 체크(사전) ----
        def _httpx_is_compatible() -> bool:
            try:
                import httpx  # type: ignore
            except Exception:
                return False
            ver = getattr(httpx, "__version__", "0.0.0")
            try:
                parts = [int(x) for x in ver.split(".")[:3]]
                while len(parts) < 3:
                    parts.append(0)
                major, minor, patch = parts
            except Exception:
                return False
            # OpenAI 최신 SDK는 httpx >= 0.27 권장
            if major > 0:
                return True
            return (minor, patch) >= (27, 0)

        # ---- OpenAI REST 셤(Shim) ----
        class _OpenAIShim:
            def __init__(self, api_key: str, base_url_param: str | None = None, timeout: float = 60.0):
                self.api_key = (api_key or "").strip()
                self.base_url = (base_url_param or "https://api.openai.com").rstrip("/")
                self.timeout = float(timeout)
                self.chat = self._Chat(self)

            class _Chat:
                def __init__(self, outer: "_OpenAIShim"):
                    self.completions = _OpenAIShim._Completions(outer)

            class _Completions:
                def __init__(self, outer: "_OpenAIShim"):
                    self._outer = outer

                def create(self, **params):

                    try:
                        import httpx  # type: ignore
                    except Exception as exc:
                        raise RuntimeError("httpx가 필요합니다.") from exc
                    url = f"{self._outer.base_url}/v1/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {self._outer.api_key}",
                        "Content-Type": "application/json",
                    }
                    with httpx.Client(timeout=self._outer.timeout) as client:
                        resp = client.post(url, headers=headers, json=params)
                        resp.raise_for_status()
                        data = resp.json()
                    try:
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
                    except Exception:
                        content = ""
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                    )

        # ---- OpenAI ----
        self._openai = None
        key = (self.cfg.openai_api_key or "").strip()
        base_url = (self.cfg.openai_base_url or "").strip() or None

        if key:
            use_sdk = _httpx_is_compatible() and ("OpenAI" in globals() and OpenAI is not None)
            if use_sdk:
                try:
                    if base_url:
                        self._openai = OpenAI(api_key=key, base_url=base_url)  # type: ignore[name-defined]
                    else:
                        self._openai = OpenAI(api_key=key)  # type: ignore[name-defined]
                except Exception:
                    # 어떤 이유로든 SDK 생성 실패하면 조용히 셤으로 폴백
                    self._openai = _OpenAIShim(api_key=key, base_url_param=base_url, timeout=60.0)
            else:
                # 사전 체크에서 불합격: SDK 시도 자체를 안 함 → __del__ 경고 원천 차단
                self._openai = _OpenAIShim(api_key=key, base_url_param=base_url, timeout=60.0)

        # ---- Gemini ----
        self._gemini_ready = False
        gkey = (self.cfg.gemini_api_key or "").strip()
        if gkey and ("genai" in globals() and genai is not None):
            try:
                genai.configure(api_key=gkey)  # type: ignore[name-defined]
                self._gemini_ready = True
            except Exception:
                self._gemini_ready = False

    # ---------- 내부 공용 호출 ----------
    def _ask_openai(self, system: str, prompt: str, **kwargs) -> str:
        """
        OpenAI 호출 (Chat Completions). 모델별 호환을 위해 temperature는 보내지 않는다.
        response_format 등 필요한 값만 kwargs로 전달 가능.
        """
        if self._openai is None:
            raise RuntimeError("OpenAI client is not initialized")

        params: Dict[str, Any] = {
            "model": getattr(self.cfg, "openai_model", None) or os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        rf = kwargs.get("response_format")
        if rf:
            params["response_format"] = rf

        res = self._openai.chat.completions.create(**params)
        return (res.choices[0].message.content or "").strip()

    def _ask_gemini(self, system: str, prompt: str, **kwargs) -> str:
        """
        Gemini 호출. google-generativeai 필요: pip install google-generativeai
        system + user를 하나의 프롬프트로 합쳐 전송.
        """

        api_key = getattr(self.cfg, "gemini_api_key", None) or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")

        try:
            import google.generativeai as ggen  # type: ignore
        except Exception as exc:
            raise RuntimeError("google-generativeai 패키지가 필요합니다. `pip install google-generativeai`") from exc

        if not getattr(self, "_gemini_configured", False):
            ggen.configure(api_key=api_key)
            self._gemini_configured = True

        model_name = self.gemini_model or "gemini-2.0-flash"
        model = ggen.GenerativeModel(model_name)

        # JSON을 더 잘 내도록 하고 싶으면 mime_type을 application/json으로 바꿔도 됨
        gen_cfg: Dict[str, Any] = {"response_mime_type": "text/plain"}
        if isinstance(kwargs.get("response_format"), dict) and kwargs["response_format"].get("type") == "json_object":
            gen_cfg["response_mime_type"] = "application/json"

        # 타입 경고 제거: 분기 내에서만 설정하고, 최종적으로 없으면 dict 사용
        generation_config_obj: Any = None
        try:

            generation_config_obj = GenerationConfig(**gen_cfg)
        except Exception:
            try:
                generation_config_obj = getattr(ggen, "GenerationConfig")(**gen_cfg)  # type: ignore[attr-defined]
            except Exception:
                pass
        if generation_config_obj is None:
            generation_config_obj = gen_cfg  # 호환 버전 대비: dict도 허용

        resp = model.generate_content(
            f"{system.strip()}\n\n{prompt.strip()}",
            generation_config=generation_config_obj,
        )

        # 표준 추출
        try:
            text = (resp.text or "").strip()
            if text:
                return text
        except Exception:
            pass

        # 후보/파츠 폴백
        try:
            parts: List[str] = []
            for cand in getattr(resp, "candidates", []) or []:
                cont = getattr(cand, "content", None)
                for part in getattr(cont, "parts", []) or []:
                    t = getattr(part, "text", "")
                    if t:
                        parts.append(t)
            return "\n".join(parts).strip()
        except Exception:
            return ""

    # real_use
    def ask_smart(
            self,
            system: str,
            user: str,
            *,
            prefer: str = "openai",
            allow_fallback: bool = True,
            trace=None,
            **kwargs,
    ) -> str:
        """
        공급자 우선순위에 따라 호출하고, trace 라벨을 '실제 시도한 공급자'로 표준화해 남긴다.
        최종적으로 선택/성공한 공급자는 'provider:selected ...' 한 줄로 요약한다.
        """

        def _t(provider_tag: str, kind: str, msg: str) -> None:
            if trace:
                try:
                    trace(f"{provider_tag}:{kind}", msg)
                except Exception:
                    pass

        # 시도 순서 결정
        order = ["openai", "gemini"] if (prefer or "openai").lower() == "openai" else ["gemini", "openai"]
        last_err: BaseException | None = None

        for provider in order:
            # prefer="gemini" & allow_fallback=False 면 gemini만 시도
            if prefer.lower() == "gemini" and not allow_fallback and provider != "gemini":
                continue

            if provider == "openai":
                model_name = getattr(self.cfg, "openai_model", None) or os.getenv("OPENAI_MODEL", "gpt-5-mini")
                _t("openai", "request", f"model={model_name}")
                try:
                    out = self._ask_openai(system, user, **kwargs)
                    _t("openai", "success", f"len={len(out)}")
                    _t("provider", "selected", f"openai model={model_name}")
                    return out
                except Exception as e:
                    _t("openai", "error", f"{type(e).__name__}: {e}")
                    last_err = e
                    if prefer.lower() == "openai" and allow_fallback:
                        continue
                    raise
            else:
                model_name = self.gemini_model or "gemini-2.0-flash"
                _t("gemini", "request", f"model={model_name}")
                try:
                    out = self._ask_gemini(system, user, **kwargs)
                    _t("gemini", "success", f"len={len(out)}")
                    _t("provider", "selected", f"gemini model={model_name}")
                    return out
                except Exception as e:
                    _t("gemini", "error", f"{type(e).__name__}: {e}")
                    last_err = e
                    if prefer.lower() == "gemini" and allow_fallback:
                        continue
                    raise

        if last_err:
            raise last_err
        raise RuntimeError("ask_smart: no provider attempted")

    def _ask(self, system: str, prompt: str) -> str:
        """
        기본: OpenAI 먼저, 결제/한도 오류면 Gemini 폴백.
        provider가 'gemini'로 설정된 경우 Gemini만 사용.
        """
        prefer = (getattr(self, "default_prefer", None) or getattr(self.cfg, "provider", "openai")).lower()
        if prefer == "gemini":
            return self._ask_gemini(system, prompt)
        try:
            return self._ask_openai(system, prompt)
        except BadRequestError as err:
            msg = str(err).lower()
            # 결제/한도/크레딧 관련 메시지에서만 폴백
            if any(k in msg for k in ["insufficient_quota", "billing", "credit", "payment", "402"]) and getattr(self, "_gemini_ready", False):
                return self._ask_gemini(system, prompt)
            raise

    # ---------- 프롬프트(장면 1문장) ----------
    def scene_prompt_kor(
        self,
        *,
        section: str,
        scene_hint: str,
        characters: Sequence[str],
        tags: Sequence[str] = (),
        effect: Optional[str] = None,
        motion: Optional[str] = None,
    ) -> str:
        has_female = any("female" in (c or "").lower() for c in characters)
        has_male   = any("male"   in (c or "").lower() for c in characters)
        people_txt = (
            "여자 단독" if has_female and not has_male else
            "남자 단독" if has_male and not has_female else
            "남녀 투샷" if (has_female and has_male) else
            "인물 없음"
        )
        # 태그 4~8개만 (실제 사용자 컨텍스트로 프롬프트에 녹임)
        tags_used = [t for t in (tags or []) if t][:8]

        system_rules = (
            "너는 뮤직비디오 한 장면을 묘사하는 프롬프트 생성기다. "
            "문장 한 줄로, '배경/인물/행동'이 자연스럽게 포함되도록 요약해. "
            "가사 원문을 그대로 쓰지 말고, 장면의 시각적 정보를 압축해라. "
            "인물은 {people}이며 **정면 또는 3/4 각도**, **얼굴 프레임 중앙**, **선명한 조명**을 명시해라. "
            "배경/분위기/렌즈/시간대/조명 등 시각 키워드만 사용하고, ‘노래’·‘가사’ 같은 단어는 금지. "
            "세로 영상에 맞는 안정 구도를 권장한다."
        ).format(people=people_txt)

        user_ctx = {
            "section": section,
            "scene_hint": scene_hint,
            "characters": list(characters),
            "tags": list(tags_used),
            "effect": effect,
            "motion": motion,
        }
        prompt_template = (
            "아래 컨텍스트로 1문장 프롬프트를 만들어줘.\n"
            f"[컨텍스트]\n{json.dumps(user_ctx, ensure_ascii=False)}\n"
            "출력은 딱 한 문장 한국어. 예: "
            "“노을 진 창가에서 여자가 카메라를 정면으로 바라보며 고요히 숨을 고르는 장면, "
            "부드러운 보케와 따뜻한 톤, 얼굴 중앙, 3/4 각도”."
        )
        return self._ask(system_rules, prompt_template)

    # ---------- 제목/가사/태그 생성 ----------
    # 가사생성 (lyrics_gen.py) 여기에 같은 이름 있음 추후 확인해야함
    def generate_title_lyrics_tags(
            *,
            prompt: str,
            duration_min: int,
            title_in: str = "",
            allowed_tags=None,
            duration_sec: int | None = None,
            trace=None,
            prefer: str | None = None,  # "openai" | "gemini"
            allow_fallback: bool | None = None,  # True/False
    ) -> dict:
        """
        가사 생성:
          - 1줄≈5초 기준으로 '본문 줄수(헤더 제외)' 가이드를 제시
          - 허용 섹션: [verse], [bridge] (그 외 헤더는 제거/치환)
          - 본문 라인에 [ko]/[en] 언어 태그를 붙이지 않음
          - 변환 '최종본'을 BASE_DIR/_debug/lyrics_gen.log 에 기록
        출력: {"title":".", "lyrics":".", "tags":[...], "tags_pick":[...]}
        """

        def emit(ev: str, msg: str) -> None:
            if callable(trace):
                try:
                    trace(ev, msg)
                except (TypeError, ValueError):
                    pass

        allowed_tags = allowed_tags or []

        # ---- 목표 초 계산(초 우선) ----
        sec_val = None
        if duration_sec is not None:
            try:
                sec_val = int(duration_sec)
            except (TypeError, ValueError):
                sec_val = None
        if sec_val is None and isinstance(prompt, str):
            match_sec = re.search(r"(\d{1,3})\s*(초|s|sec|secs|second|seconds)\b", prompt, flags=re.I)
            if match_sec:
                try:
                    sec_val = int(match_sec.group(1))
                except (TypeError, ValueError):
                    sec_val = None
        if sec_val is None or sec_val <= 0:
            try:
                duration_min_val = int(duration_min)
            except (TypeError, ValueError):
                duration_min_val = 2
            duration_min_val = max(1, min(3, duration_min_val))
            sec_val = duration_min_val * 60

        # ---- 1줄≈5초: '본문 줄수' 가이드(헤더 제외) ----
        base_lines = max(1, round(sec_val / 5))
        if sec_val <= 35:
            min_lines, max_lines = 6, 8
        elif sec_val <= 75:
            min_lines, max_lines = 10, 12
        elif sec_val <= 150:
            min_lines, max_lines = max(8, base_lines - 2), base_lines + 2
        else:
            min_lines, max_lines = max(10, base_lines - 2), base_lines + 3

        sys_msg = (
            "You are a Korean lyricist and music director. Return ONE JSON object only:\n"
            '{"title":".", "lyrics_ko":".", "tags":[".", "."], "tags_pick":[".", "."]}\n'
            "- Allowed headers: [verse], [bridge]\n"
            f"- Body line budget (EXCLUDING headers): {min_lines}–{max_lines}\n"
            "- IMPORTANT:\n"
            "  1) Do NOT use [intro], [outro], [chorus], pre/post-chorus, hook, etc.\n"
            "  2) Keep lyric lines only under [verse]/[bridge].\n"
            "  3) No production notes or metadata.\n"
            f"- Target duration: ~{sec_val}s (≈5s per line)\n"
        )
        if allowed_tags:
            sys_msg += "ALLOWED_TAGS: " + ", ".join(sorted(set(allowed_tags))) + "\n"

        user_msg = (
                "[TASK]\n"
                "- Write natural Korean lyrics with the above constraints.\n"
                "- Title may be short and poetic.\n\n"
                "[PROMPT]\n" + (prompt or "")
        )

        # ---- 모델 호출 ----
        prefer_opt = "openai" if prefer is None else str(prefer)
        allow_opt = (allow_fallback if allow_fallback is not None else (prefer_opt == "openai"))
        emit("ai:prepare",
             f"prefer={prefer_opt}, allow_fallback={allow_opt}, sec={sec_val}, lines={min_lines}-{max_lines}")

        ai = AI()
        raw_reply = ai.ask_smart(sys_msg, user_msg, prefer=prefer_opt, allow_fallback=allow_opt, trace=trace)
        reply_text = str(raw_reply or "").strip()
        if not reply_text:
            raise RuntimeError("빈 응답입니다.")

        # ---- JSON 파싱(관대한 추출) ----
        emit("parse:begin", f"text_len={len(reply_text)}")
        try:
            a_pos = reply_text.find("{")
            b_pos = reply_text.rfind("}")
            if a_pos == -1 or b_pos == -1 or b_pos <= a_pos:
                data_obj = json.loads(reply_text)
            else:
                data_obj = json.loads(reply_text[a_pos:b_pos + 1])
        except json.JSONDecodeError:
            data_obj = {"title": title_in or "untitled", "lyrics_ko": reply_text}
        emit("parse:end", "ok")

        # ---- 필드 보정 ----
        title = str(data_obj.get("title", "")).strip() or (title_in or "untitled")
        lyrics_src = str(data_obj.get("lyrics_ko", "") or data_obj.get("lyrics", "")).strip()

        # ---- 금지 섹션 제거/치환 ----
        ban_head_pat = re.compile(
            r"^\s*\[(?:chorus|pre[- ]?chorus|post[- ]?chorus|hook|coda|break|tag|interlude|intro|outro)(?:\s+\d+)?]\s*$",
            re.IGNORECASE,
        )
        tmp_lines: List[str] = []
        for ln in lyrics_src.splitlines():
            s = ln.strip()
            if ban_head_pat.match(s):
                tmp_lines.append("[verse]")
            else:
                tmp_lines.append(ln)
        lyrics_mid = "\n".join(tmp_lines)

        # ---- 기본 노이즈 정리 ----
        keep_head_pat = re.compile(r"^\s*\[(?:verse|bridge)(?:\s+\d+)?]\s*$", re.IGNORECASE)
        paren_only_pat = re.compile(r"^\s*\(.+?\)\s*$")
        cleaned: List[str] = []
        for ln in lyrics_mid.splitlines():
            s = ln.strip()
            if not s:
                continue
            if keep_head_pat.match(s):
                cleaned.append(s)
                continue
            if paren_only_pat.match(s):
                continue
            cleaned.append(s)

        # 중복 제거
        uniq: List[str] = []
        seen = set()
        for s in cleaned:
            if s not in seen:
                uniq.append(s)
                seen.add(s)

        lyrics_body = "\n".join(uniq).strip()
        if not lyrics_body:
            raise RuntimeError("가사 내용이 비어 있습니다.")

        # ---- 태그 정규화 ----
        def _norm_tags(tags_in) -> List[str]:
            if isinstance(tags_in, str):
                parts = [p.strip() for p in re.split(r"[,\n/;]+", tags_in) if p.strip()]
            elif isinstance(tags_in, list):
                parts = [str(p).strip() for p in tags_in if str(p).strip()]
            else:
                parts = []
            base = [
                "clean vocals", "natural articulation", "warm emotional tone",
                "studio reverb light", "clear diction", "balanced mixing",
            ]
            if len(parts) < 5:
                parts = list(dict.fromkeys(parts + base))
            return parts[:12]

        tags = _norm_tags(data_obj.get("tags"))
        picks_raw = _norm_tags(data_obj.get("tags_pick"))
        if allowed_tags:
            allow_set = {str(t).lower() for t in allowed_tags}
            picks = [t for t in picks_raw if t.lower() in allow_set][:10]
        else:
            picks = picks_raw[:10]

        # ---- 가사 라인 정리 (언어 태그 없이) ----
        final_lines_generate: List[str] = []
        for line_item in lyrics_body.splitlines():
            # [verse], [bridge] 같은 섹션 헤더는 소문자로 통일하여 그대로 유지합니다.
            stripped_line = line_item.strip()
            if keep_head_pat.match(stripped_line):
                final_lines_generate.append(stripped_line.lower())
            # 나머지 가사 라인은 원본을 그대로 추가합니다.
            else:
                final_lines_generate.append(line_item)

        lyrics_out = "\n".join(final_lines_generate)

        # ---- 디버그 로그(최종본 기록) ----
        try:

            base_dir = Path(BASE_DIR)
            dbg_dir = base_dir / "_debug"
            dbg_dir.mkdir(parents=True, exist_ok=True)
            with (dbg_dir / "lyrics_gen.log").open("a", encoding="utf-8") as fp:
                fp.write("\n===== LYRICS GENERATED (no lang tags) =====\n")
                fp.write(f"title: {title}\n")
                fp.write(lyrics_out + "\n")
        except (OSError, ValueError, TypeError, ImportError):
            pass

        emit("normalize:done", "lyrics generated without language tags")

        return {"title": title, "lyrics": lyrics_out, "tags": tags, "tags_pick": picks}

    # ---------- JSON/정규화 유틸 ----------
    @staticmethod
    def _safe_json(text: str) -> Dict[str, Any]:
        """
        모델이 코드펜스나 설명을 섞어보내도 JSON만 뽑아 안전 파싱
        """
        t = (text or "").strip()
        # 코드펜스 제거
        if t.startswith("```"):
            t = t.strip("`").strip()
        # JSON 블록만 추출
        s, e = t.find("{"), t.rfind("}")
        if 0 <= s < e:
            frag = t[s:e+1]
            try:
                return json.loads(frag)
            except Exception:
                pass
        # 라스트 찬스
        try:
            return json.loads(t)
        except Exception:
            return {"title": "", "lyrics": "", "tags": [], "tags_pick": []}

    @staticmethod
    def _enforce_title(title: str, fallback_prompt: str) -> str:
        t = (title or "").strip()
        if not t or t in {"무제", "제목", "Untitled", "untitled"}:
            t = (fallback_prompt or "노래").strip()
            t = re.sub(r"[^ㄱ-ㅎ가-힣0-9A-Za-z\s]", "", t)
            t = t.split()[0][:12] if t else "노래"
        return t

    @staticmethod
    def _dedup_keep_order(items: List[str]) -> List[str]:
        seen, ret = set(), []
        for it in items:
            k = (it or "").strip()
            if not k:
                continue
            if k not in seen:
                seen.add(k)
                ret.append(k)
        return ret

    def _normalize_tags(self, tags) -> list[str]:
        if isinstance(tags, str):
            parts = [p.strip() for p in re.split(r"[,\n/;]+", tags) if p.strip()]
        elif isinstance(tags, list):
            parts = [str(p).strip() for p in tags if str(p).strip()]
        else:
            parts = []
        # 영문만
        parts = [p for p in parts if re.search(r"[A-Za-z]", p)]
        parts = self._dedup_keep_order(parts)

        # 5개 미만이면 보강할 기본 성향
        basic = [
            "clean vocals",
            "natural articulation",
            "warm emotional tone",
            "studio reverb light",
            "clear diction",
            "breath control",
            "balanced mixing",
        ]
        if len(parts) < 5:
            parts = self._dedup_keep_order(parts + basic)

        return parts[:12]

    @staticmethod
    def _normalize_sections(text: str) -> str:
        if not text:
            return text
        out_lines, has_tag = [], False
        for ln in text.splitlines():
            stripped = ln.strip()
            # 이미 정식 라벨이면 통과
            if re.match(r"^\[(verse|chorus|bridge|outro)(\s+\d+)?]\s*$", stripped, flags=re.IGNORECASE):
                out_lines.append(stripped.lower())
                has_tag = True
                continue
            # 한국어 라벨을 치환
            m = re.match(r"^\s*\(?\s*(\d+)\s*절\s*\)?\s*[:：)]*\s*$", stripped)
            if m:
                out_lines.append(f"[verse {m.group(1)}]")
                has_tag = True
                continue
            if re.match(r"^\s*\(?\s*후\s*렴\s*\)?\s*[:：)]*\s*$", stripped):
                out_lines.append("[chorus]")
                has_tag = True
                continue
            if re.match(r"^\s*\(?\s*브\s*릿\s*지\s*\)?\s*[:：)]*\s*$", stripped):
                out_lines.append("[bridge]")
                has_tag = True
                continue
            if re.match(r"^\s*\(?\s*아\s*웃\s*트\s*로\s*\)?\s*[:：)]*\s*$", stripped):
                out_lines.append("[outro]")
                has_tag = True
                continue
            out_lines.append(ln)
        if not has_tag and out_lines:
            out_lines.insert(0, "[verse]")
        return "\n".join(out_lines)

    @staticmethod
    def _extract_seconds_hint(text: str) -> Optional[int]:
        if not text:
            return None
        m = re.search(r"(\d{1,3})\s*(초|s|sec|secs|second|seconds)\b", text, flags=re.I)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    @staticmethod
    def _fix_inline_headers(text: str) -> str:
        """한 줄에 헤더와 가사가 붙어버린 경우 분리"""
        if not text:
            return ""
        lines = []
        for ln in text.splitlines():
            ln2 = re.sub(r"(?i)\[(verse|chorus|bridge|outro)]\s*", lambda m: m.group(0)+"\n", ln, count=1)
            parts = [p for p in ln2.splitlines() if p.strip()]
            lines.extend(parts)
        return "\n".join(lines)

    @staticmethod
    def _parse_sections(text: str) -> list[tuple[str, list[str]]]:
        """가사를 섹션 단위로 파싱 → [(section, [lines...])]"""
        cur_sec: Optional[str] = None
        cur_lines: list[str] = []
        ret: list[tuple[str, list[str]]] = []
        for ln in (text or "").splitlines():
            s = ln.strip()
            m = re.match(r"^\[(verse|chorus|bridge|outro)(?:\s+\d+)?]\s*$", s, flags=re.I)
            if m:
                if cur_sec is not None:
                    ret.append((cur_sec, [x for x in cur_lines if x.strip()]))
                cur_sec = m.group(1).lower()
                cur_lines = []
            else:
                if s:
                    cur_lines.append(s)
        if cur_sec is not None:
            ret.append((cur_sec, [ln for ln in cur_lines if ln.strip()]))
        if not ret:
            # 섹션이 하나도 없으면 verse로 묶어 반환
            lines = [ln for ln in (text or "").splitlines() if ln.strip()]
            return [("verse", lines)]
        return ret

    @staticmethod
    def _format_sections(sections: list[tuple[str, list[str]]]) -> str:
        """[(sec, lines)]를 정식 포맷 문자열로."""
        chunks: list[str] = []
        for sec, lines in sections:
            chunks.append(f"[{sec}]")
            chunks.extend(lines)
            chunks.append("")  # 섹션 사이 공백
        return "\n".join(chunks).strip()

    @staticmethod
    def _enforce_duration_structure(
            sections: list[tuple[str, list[str]]], seconds: int
    ) -> list[tuple[str, list[str]]]:
        """
        곡 길이에 맞춰 섹션/행 수를 '컷'한다.
        - ≤30초: [verse] 2–3줄 → [chorus] 2–3줄 (총 4–6줄)
        - 31–60초: [verse] 4–6줄 → [chorus] 4–6줄 (총 8–12줄)
        - 나머지: 그대로 두되 헤더 형식만 정리
        """
        sec = max(1, int(seconds or 0))
        if sec <= 30:
            verse, chorus = None, None
            for s, lines in sections:
                if s == "verse" and verse is None:
                    verse = ("verse", lines[:3])  # 2~3줄 목표, 넉넉히 3으로 컷
                elif s == "chorus" and chorus is None:
                    chorus = ("chorus", lines[:3])
                if verse and chorus:
                    break
            out: list[tuple[str, list[str]]] = []
            if verse:  out.append(verse)
            if chorus: out.append(chorus)
            if not out and sections:
                s0, l0 = sections[0]
                out = [(s0, l0[:8])]
            return out

        if sec <= 60:
            verse, chorus = None, None
            for s, lines in sections:
                if s == "verse" and verse is None:
                    verse = ("verse", lines[:6])  # 4~6줄 목표
                elif s == "chorus" and chorus is None:
                    chorus = ("chorus", lines[:6])
                if verse and chorus:
                    break
            out: list[tuple[str, list[str]]] = []
            if verse:  out.append(verse)
            if chorus: out.append(chorus)
            if not out and sections:
                s0, l0 = sections[0]
                out = [(s0, l0[:8])]
            return out

        # 60초 초과는 구조 가이드만 따르고 그대로 반환
        return sections


    class AI:
        # ... (기존 코드 유지)

        def segment_lyrics(self, sections: list) -> dict:
            """
            sections = [{"id":"S01","text":"<전체 가사 한 줄>"}]
            반환:
            {
              "segments": [
                {"text": "...의미1...", "reason": "..."},
                {"text": "...의미2...", "reason": "..."},
                ...
              ]
            }
            """
            _ = self  # keep as instance method (avoid "could be static" warning)

            full = " ".join(((sections[0].get("text") or "").strip()).split())
            if not full:
                return {"segments": []}

            # 1) OpenAI 사용 (환경변수 OPENAI_API_KEY 설정 시)
            if os.getenv("OPENAI_API_KEY"):
                try:
                    # OpenAI responses API (Responses) 사용 예시

                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    prompt = (
                        "다음 한국어 가사를 의미 전달 단위(한 구절이 온전한 의미, 임팩트, 행동단위, 묘사단위 등)로만 나눠줘. "
                        "되도록이면 문장이 길지 않게 짧게 나눠줘. "
                        "각 조각은 원문 그대로 보존하고, 불필요한 수정/삭제/치환 금지. "
                        "길이 기준이 아니라 의미 기준. JSON 배열로만 반환: [\"구절1\", \"구절2\", ...]\n\n"
                        f"가사:\n{full}"
                    )
                    res = client.responses.create(
                        model="gpt-4o-mini",
                        input=prompt,
                        temperature=0.2,
                    )
                    text = res.output_text.strip()
                    # JSON 배열 파싱 시도
                    arr = json.loads(text)
                    segs = [{"text": t, "reason": "openai-seg"} for t in arr if isinstance(t, str) and t.strip()]
                    return {"segments": segs}
                except Exception:
                    pass

            # 2) Gemini 사용 (환경변수 GOOGLE_API_KEY 설정 시)
            if os.getenv("GOOGLE_API_KEY"):
                try:
                    import google.generativeai as ggen  # type: ignore
                    ggen.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                    model = ggen.GenerativeModel("gemini-2.0-flash")
                    prompt = (
                        "다음 한국어 가사를 의미 전달 단위로만 나눠줘. "
                        "각 조각은 원문 그대로 보존하고, 불필요한 수정/삭제/치환 금지. "
                        "길이 기준이 아니라 의미 기준. JSON 배열만 반환: [\"구절1\", \"구절2\", ...]\n\n"
                        f"가사:\n{full}"
                    )
                    resp = model.generate_content(prompt)
                    text = (resp.text or "").strip()
                    arr = json.loads(text)
                    segs = [{"text": t, "reason": "gemini-seg"} for t in arr if isinstance(t, str) and t.strip()]
                    return {"segments": segs}
                except Exception:
                    pass

            # 3) 둘 다 실패시: 안전한 규칙 기반 백업(하드코딩 문구 금지, 문장부호/접속어 기준 휴리스틱)
            #  - 쉼표/마침표/의문문/접속어(그리고/하지만/혹시 등) 주변으로 나눔
            text = full
            # 문장부호 기준 1차 분리
            parts = re.split(r"[,.!?]\s*", text)

            # 추가 휴리스틱: '혹시', '지금', '같은', '하지만', '그리고' 등 접속/전환 단서 앞에서 자르기
            refined = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                chunks = re.split(r"\s+(?=(혹시|지금|같은|하지만|그리고|그래서|그러면))", p)
                buf = []
                for c in chunks:
                    c = c.strip()
                    if not c:
                        continue
                    if buf:
                        buf.append(c)
                        refined.append(" ".join(buf).strip())
                        buf = []
                    else:
                        buf.append(c)
                if buf:
                    refined.append(" ".join(buf).strip())
            refined = [r for r in refined if r]
            return {"segments": [{"text": r, "reason": "fallback"} for r in refined]}



def _qt_get(*names):
    """
    예: _qt_get("Qt","NonModal")  # PyQt5
        _qt_get("Qt","WindowModality","NonModal")  # PyQt6
    존재하면 그 값을, 없으면 None을 반환
    """
    obj = QtCore
    for n in names:
        obj = getattr(obj, n, None)
        if obj is None:
            return None
    return obj

flags = getattr(QtCore.QEventLoop, "AllEvents", None)
if flags is None:
    pef = getattr(QtCore.QEventLoop, "ProcessEventsFlag", None)
    flags = getattr(pef, "AllEvents", None) if pef else None

# 호환 상수
WINDOW_NONMODAL = _qt_get("Qt", "NonModal") or _qt_get("Qt", "WindowModality", "NonModal")
PEF_ALL_EVENTS = _qt_get("QEventLoop", "AllEvents") or _qt_get("QEventLoop", "ProcessEventsFlag", "AllEvents")
DOCK_BOTTOM = _qt_get("Qt", "BottomDockWidgetArea") or _qt_get("Qt", "DockWidgetArea", "BottomDockWidgetArea")

TailT = Optional[Union[str, PathLike[str]]]


# real_use
class ProgressLogDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, title="작업 진행"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self.setMinimumWidth(520)
        self._completed = False

        self.lbl = QtWidgets.QLabel("준비 중…")
        self.bar = QtWidgets.QProgressBar(); self.bar.setRange(0, 0)
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(200)
        self.btn = QtWidgets.QPushButton("숨기기")
        self.btn.clicked.connect(self._on_btn)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.lbl); lay.addWidget(self.bar); lay.addWidget(self.log)
        row = QtWidgets.QHBoxLayout(); row.addStretch(1); row.addWidget(self.btn); lay.addLayout(row)

    def _on_btn(self):
        if self._completed: self.accept()
        else: self.hide()

    def set_status(self, text: str):
        self.lbl.setText(text or "")

    def append_log(self, text: str):
        if not text: return
        self.log.appendPlainText(text.rstrip("\n"))
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def enter_determinate(self, total: int):
        self._completed = False
        total = max(0, int(total))
        self.bar.setRange(0, total); self.bar.setValue(0)
        self.setWindowTitle("진행 중")
        self.set_status(f"0 / {total}")

    def step(self, msg: str | None = None):
        if msg: self.append_log(msg)
        v = min(self.bar.value() + 1, self.bar.maximum())
        self.bar.setValue(v)
        if self.bar.maximum() > 0:
            self.set_status(f"{v} / {self.bar.maximum()}")

    def set_completed(self, text: str = "완료"):
        self._completed = True
        self.bar.setRange(0, 1); self.bar.setValue(1)
        self.setWindowTitle("완료")
        self.set_status(text)
        self.btn.setText("닫기")

    def finish(self, ok: bool, err_text: str | None) -> None:
        # 진행바 완료 상태
        try:
            self.bar.setRange(0, 1)
            self.bar.setValue(1)
        except Exception:
            pass

        # 타이틀/로그
        try:
            base = self.windowTitle()
            self.setWindowTitle(f"{base} — {'완료' if ok else '오류'}")
        except Exception:
            pass

        try:
            self.append_log("[done] 완료" if ok else f"[error] {err_text or '오류'}")
        except Exception:
            pass

        # '닫기' 버튼 확보 (기존 버튼 있으면 재활용)
        btn = None
        for name in ("btn_close", "btnCancel", "btn_cancel", "btnClose", "btn_ok", "closeButton"):
            b = getattr(self, name, None)
            if isinstance(b, QtWidgets.QPushButton):
                btn = b
                break

        # 버튼이 없으면 생성해서 우측 정렬로 레이아웃에 추가
        if btn is None:
            btn = QtWidgets.QPushButton("닫기", self)
            lay = self.layout()
            if lay is not None:
                hb = QtWidgets.QHBoxLayout()
                hb.addStretch(1)  # 왼쪽 공간으로 밀어 우측 정렬 효과
                hb.addWidget(btn)

                if isinstance(lay, (QtWidgets.QVBoxLayout, QtWidgets.QHBoxLayout, QtWidgets.QFormLayout)):
                    lay.addLayout(hb)
                elif isinstance(lay, QtWidgets.QGridLayout):
                    row = lay.rowCount()
                    cols = max(1, lay.columnCount())
                    lay.addLayout(hb, row, 0, 1, cols)
                else:
                    # 레이아웃 타입을 모를 때 안전판
                    cont = QtWidgets.QWidget(self)
                    cont.setLayout(hb)
                    try:
                        lay.addWidget(cont)
                    except Exception:
                        v = QtWidgets.QVBoxLayout(self)
                        v.addWidget(cont)
                        self.setLayout(v)

        # 버튼을 '닫기' 동작으로 전환
        try:
            btn.clicked.disconnect()
        except Exception:
            pass
        btn.setText("닫기")
        btn.setEnabled(True)
        btn.clicked.connect(self.accept)

        # ESC로도 닫히게(보강)
        try:
            self.setResult(0)
        except Exception:
            pass


class JobWorker(QtCore.QObject):
    """백그라운드에서 실제 작업을 실행하고, 진행 상황을 신호로 넘김"""
    progress = QtCore.pyqtSignal(dict)
    finished = QtCore.pyqtSignal(object, object)  # (result, err)

    def __init__(self, job_callable):
        super().__init__()
        self._job = job_callable

    @QtCore.pyqtSlot()
    def run(self):
        try:
            def progress_cb(info: dict):
                # 항상 dict 복사 후 emit (예상치 못한 뮤터블 공유 방지)
                try:
                    self.progress.emit(dict(info or {}))
                except Exception as abc:
                    print("[progress] emit error:", abc, flush=True)

            res = self._job(progress_cb)
            self.finished.emit(res, None)
        except Exception as e:
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            self.finished.emit(None, f"{e}\n{tb}")


class _Tailer(QtCore.QObject):
    """선택: 텍스트 로그 파일 tail. UI 스레드에서 QTimer로만 동작."""
    tick = QtCore.pyqtSignal(str)

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self._path = Path(path) if path else None
        self._fp = None
        self._pos = 0
        self._tm = QtCore.QTimer(self)
        self._tm.setInterval(500)
        self._tm.timeout.connect(self._poll)

    def start(self):
        if not (self._path and self._path.exists()):
            return
        try:
            self._fp = open(self._path, "r", encoding="utf-8", errors="ignore")
            self._fp.seek(0, io.SEEK_END)
            self._pos = self._fp.tell()
            self._tm.start()
            self.tick.emit(f"[tail] {self._path}")
        except Exception:
            pass

    def stop(self):
        try:
            self._tm.stop()
        except Exception:
            pass
        try:
            if self._fp:
                self._fp.close()
                self._fp = None
        except Exception:
            pass

    def _poll(self):
        try:
            if not self._fp:
                return
            self._fp.seek(self._pos)
            chunk = self._fp.read()
            self._pos = self._fp.tell()
            if chunk:
                self.tick.emit(chunk)
        except Exception:
            pass


def _mk_progress(owner: QtWidgets.QWidget, title: str, tail_file: TailT = None):
    tail_str = os.fspath(tail_file) if tail_file is not None else None

    dlg = ProgressLogDialog(parent=owner, title=title)
    try:
        flag = getattr(QtCore.Qt, "WindowStaysOnTopHint", None)
        if flag is None:
            flag = getattr(getattr(QtCore.Qt, "WindowType"), "WindowStaysOnTopHint")
        dlg.setWindowFlag(flag, False)
    except Exception:
        pass

    dlg.resize(780, 420)
    dlg.show()
    QtWidgets.QApplication.processEvents()
    try:
        dlg.raise_()
        dlg.activateWindow()
    except Exception:
        pass

    def _post_ui(fn) -> None:
        QtCore.QTimer.singleShot(0, fn)

    def on_progress(info: Dict[str, Any]) -> None:
        stage = str(info.get("stage") or "info")
        msg = str(info.get("msg") or "")
        step = info.get("step")

        if msg:
            _post_ui(lambda: dlg.append_log(f"[{stage}] {msg}"))
            if hasattr(owner, "log_line") and callable(getattr(owner, "log_line")):
                _post_ui(lambda: owner.log_line(stage, msg))

        if isinstance(step, int) and step > 0:
            def _advance():
                if dlg.bar.maximum() == 0:
                    dlg.enter_determinate(step)
                dlg.step(msg or "")
            _post_ui(_advance)

    def finalize(ok: bool, _payload: Any = None, err: Optional[BaseException] = None) -> None:
        _post_ui(lambda: dlg.finish(ok, (str(err) if err else None)))

    # (선택) 실시간 로그 테일링 — 이미 외부에서 tail_file을 주면 활성화
    if tail_str and os.path.exists(tail_str):
        try:
            fp = open(tail_str, "r", encoding="utf-8", errors="ignore")
            fp.seek(0, os.SEEK_END)
            timer = QtCore.QTimer(dlg)
            timer.setInterval(500)

            def _poll():
                try:
                    line = fp.readline()
                    flushed = False
                    while line:
                        dlg.append_log(line.rstrip("\n"))
                        flushed = True
                        line = fp.readline()
                    if flushed:
                        dlg.ensure_log_visible()
                except Exception:
                    pass

            timer.timeout.connect(_poll)
            timer.start()

            def _stop():
                try:
                    timer.stop()
                except Exception:
                    pass
                try:
                    fp.close()
                except Exception:
                    pass

            try:
                dlg.finished.connect(lambda _=None: _stop())
            except Exception:
                pass
        except Exception:
            pass

    return on_progress, finalize, dlg


#
# def run_job_with_progress(
#     owner: QtWidgets.QWidget,
#     title: str,
#     job: Callable[[Callable[[Dict[str, Any]], None]], Any],
#     *,
#     tail_file: TailT = None,
#     on_done: Optional[Callable[[bool, Any, Optional[BaseException]], None]] = None,
# ) -> None:
#     """동기 실행(간단 작업용)"""
#     on_progress, finalize, _dlg = _mk_progress(owner, title, tail_file=tail_file)
#     ok = True
#     payload: Any = None
#     err: Optional[BaseException] = None
#     try:
#         payload = job(on_progress)
#     except BaseException as exc:
#         ok = False
#         err = exc
#     finally:
#         finalize(ok, payload, err)
#         if on_done:
#             try:
#                 on_done(ok, payload, err)
#             except Exception:
#                 pass

def connect_worker_with_progress(
    owner: QtWidgets.QWidget,
    title: str,
    *,
    worker: QtCore.QObject,
    signal_name: str = "progress",  # 워커의 진행 시그널 이름(인자: dict)
    finished_signal: str = "finished",  # 종료 시그널 이름
    error_signal: Optional[str] = None,  # 에러 시그널 이름(인자: Exception 또는 str)
    tail_file: Optional[str] = None,
    on_done: Optional[Callable[[bool, Any, Optional[BaseException]], None]] = None,
) -> None:
    """
    스레드/워커 기반 작업을 진행창에 붙이고 싶은 경우 사용.
    - worker.progress(dict) → 진행 로그
    - worker.finished() → ok=True 종료 처리
    - worker.error(Exception|str) → ok=False 종료 처리
    """
    on_progress, finalize, _dlg = _mk_progress(owner, title, tail_file=tail_file)

    # progress(dict) 연결
    try:
        getattr(worker, signal_name).connect(lambda info: on_progress(info if isinstance(info, dict) else {"msg": str(info)}))  # type: ignore
    except Exception:
        pass

    def _done(ok: bool, payload: Any = None, exc: Optional[BaseException] = None):
        finalize(ok, payload, exc)
        if on_done:
            try:
                on_done(ok, payload, exc)
            except Exception:
                pass

    # finished()
    try:
        getattr(worker, finished_signal).connect(lambda: _done(True, None, None))  # type: ignore
    except Exception:
        pass

    # error(...)
    if error_signal:
        try:
            def _on_error(e: Any):
                _done(False, None, e if isinstance(e, BaseException) else RuntimeError(str(e)))
            getattr(worker, error_signal).connect(_on_error)  # type: ignore
        except Exception:
            pass

# ───────── 비동기(스레드) 실행기 추가 ─────────
class _JobWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(dict)
    finished = QtCore.pyqtSignal(object, object)  # payload, err
    def __init__(self, job):
        super().__init__()
        self._job = job
    @QtCore.pyqtSlot()
    def run(self):
        try:
            payload = self._job(self._emit_progress)
            self.finished.emit(payload, None)
        except BaseException as exc:
            self.finished.emit(None, exc)
    def _emit_progress(self, info: Dict[str, Any]) -> None:
        if not isinstance(info, dict):
            info = {"msg": str(info)}
        self.progress.emit(info)

# real_use
def run_job_with_progress_async(
    owner: QtWidgets.QWidget,
    title: str,
    job,
    *,
    tail_file=None,
    on_done=None,
) -> None:

    # 0) 기존 진행창 재사용 여부 확인
    reuse_ctx = getattr(owner, "_progress_ctx", None)
    on_progress_ui = finalize_ui = dlg = None
    reused = False

    if reuse_ctx is not None:
        try:
            old_on_progress, old_finalize, old_dlg = reuse_ctx
            if old_dlg is not None and old_dlg.isVisible():
                on_progress_ui, finalize_ui, dlg = old_on_progress, old_finalize, old_dlg
                reused = True
        except Exception:
            pass

    # 1) 재사용 불가하면 새로 만든다
    if dlg is None:
        on_progress_ui, finalize_ui, dlg = _mk_progress(owner, title, tail_file=tail_file)  # type: ignore
        setattr(owner, "_progress_ctx", (on_progress_ui, finalize_ui, dlg))
        reused = False  # 새 창이니까 원래 finalize 써도 됨

    # 2) 시작 로그
    try:
        on_progress_ui({"stage": "ui", "msg": "[ui] 작업 시작 준비"})
    except Exception:
        pass

    class _Worker(QtCore.QObject):
        progress = QtCore.pyqtSignal(dict)
        finished = QtCore.pyqtSignal(object, object)

        @QtCore.pyqtSlot()
        def run(self):
            payload = None
            err = None
            try:
                def on_progress(info: dict):
                    if not isinstance(info, dict):
                        info = {"msg": str(info)}
                    self.progress.emit(info)
                payload = job(on_progress)
            except Exception as ex:
                err = ex
            finally:
                self.finished.emit(payload, err)

    obj = _Worker()
    th = QtCore.QThread(dlg)
    obj.moveToThread(th)

    def _on_progress(info: dict):
        try:
            on_progress_ui(info)
        except Exception:
            pass

    def _on_finished(payload, err):
        ok = (err is None)

        # 새로 만든 창일 때만 원래 finalize 호출
        if not reused:
            try:
                finalize_ui(ok, payload, err)
            except Exception:
                pass
        else:
            # 재사용 창일 때는 닫기버튼/추가 UI 생성 막기 위해 아무것도 안 함
            # 필요하면 여기서 로그만 하나 찍자
            try:
                on_progress_ui({"stage": "done", "msg": "[ui] 작업 1건 완료 (재사용 중)"})
            except Exception:
                pass

        # 호출자가 준 on_done은 항상 불러줌
        if callable(on_done):
            try:
                on_done(ok, payload, err)
            except Exception:
                pass

        # 스레드 정리
        try:
            th.quit()
            th.wait(100)
        except Exception:
            pass

        # 소유자에 보관했던 스레드 참조 제거
        try:
            jobss = getattr(owner, "_progress_jobs", [])
            if th in jobss:
                jobss.remove(th)
            setattr(owner, "_progress_jobs", jobss)
        except Exception:
            pass

    obj.progress.connect(_on_progress)
    obj.finished.connect(_on_finished)
    th.started.connect(obj.run)

    # GC 방지
    try:
        jobs = getattr(owner, "_progress_jobs", None)
        if not isinstance(jobs, list):
            jobs = []
        jobs.append(th)
        setattr(owner, "_progress_jobs", jobs)
        setattr(th, "_worker_ref", obj)
    except Exception:
        pass

    # 시작 로그
    try:
        on_progress_ui({"stage": "ui", "msg": "[ui] 백그라운드 스레드 시작"})
    except Exception:
        pass

    # 스레드 시작
    try:
        th.start()
    except Exception as start_exc:
        try:
            on_progress_ui({"stage": "error", "msg": f"[error] thread start failed: {start_exc}"})
        except Exception:
            pass
        # 첫 창일 때만 finalize
        if not reused:
            try:
                finalize_ui(False, None, start_exc)
            except Exception:
                pass
        if callable(on_done):
            try:
                on_done(False, None, start_exc)
            except Exception:
                pass
        return












class _FallbackProgressDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, title: str = "진행", tail_file: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(720, 360)
        self.text = QtWidgets.QPlainTextEdit(self)
        self.text.setReadOnly(True)
        self.bar = QtWidgets.QProgressBar(self)
        self.bar.setRange(0, 0)  # indeterminate
        btn = QtWidgets.QPushButton("닫기", self)
        btn.clicked.connect(self.accept)
        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(self.text)
        v.addWidget(self.bar)
        v.addWidget(btn)
        self._tail_file = tail_file

    def append_log(self, line: str) -> None:
        self.text.appendPlainText(line)
        sb = self.text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def enter_determinate(self, steps: int) -> None:
        self.bar.setRange(0, max(steps, 1))
        self.bar.setValue(0)

    def step(self, _label: str = "") -> None:
        self.bar.setValue(self.bar.value() + 1)

    def finish(self, ok: bool, err_text: str | None) -> None:
        self.append_log("[done] 완료" if ok else f"[error] {err_text or '오류'}")
        try:
            self.bar.setRange(0, 1)
            self.bar.setValue(1)
        except Exception:
            pass
        # 버튼이 '닫기'로 보이도록(이미 닫기면 스킵)
        try:
            self.findChild(QtWidgets.QPushButton).setText("닫기")
        except Exception:
            pass
        self.close()  # 바로 닫고 싶다면 유지, 남겨두려면 이 줄을 제거




# 공식/커뮤니티에서 통용되는 영어 태그만 화이트리스트로 유지
ACE_TAG_WHITELIST = {
    # Style
    "electronic","rock","pop","funk","soul","cyberpunk","acid jazz","em",
    "soft electric drums","melodic",
    # Scene
    "background music for parties","radio broadcasts","workout playlists",
    # Instrument
    "saxophone","jazz","piano","violin","acoustic guitar","electric bass",
    # Tempo / Production
    "110 bpm","fast tempo","slow tempo","loops","fills",
    # Vocal
    "soft female voice","soft male voice","mixed vocals",
    # Basic vocal bundle
    "clean vocals","natural articulation","warm emotional tone",
    "studio reverb light","clear diction","breath control","balanced mixing",
}

# 한글/동의어 → 영문 표준 태그 매핑
KO_EN_TAG_MAP = {
    # Style
    "전자음악":"electronic", "록":"rock", "록 음악":"rock",
    "팝":"pop", "펑크":"funk", "소울":"soul",
    "사이버펑크":"cyberpunk", "애시드 재즈":"acid jazz",
    "이엠":"em", "소프트 전자 드럼":"soft electric drums",
    "멜로딕":"melodic",

    # Scene
    "파티 배경 음악":"background music for parties",
    "라디오 방송":"radio broadcasts",
    "운동용 플레이리스트":"workout playlists",

    # Instrument
    "색소폰":"saxophone","재즈":"jazz","피아노":"piano","바이올린":"violin",
    "어쿠스틱 기타":"acoustic guitar","일렉트릭 베이스":"electric bass",

    # Tempo / Production
    "110 비피엠":"110 bpm","빠른 템포":"fast tempo","느린 템포":"slow tempo",
    "루프":"loops","필":"fills",

    # Vocal
    "여성 보컬":"soft female voice","남성 보컬":"soft male voice","혼성 보컬":"mixed vocals",
    "클린 보컬":"clean vocals","자연스러운 발음":"natural articulation",
    "따뜻한 감정 톤":"warm emotional tone","가벼운 스튜디오 리버브":"studio reverb light",
    "명확한 딕션":"clear diction","호흡 컨트롤":"breath control","밸런스드 믹싱":"balanced mixing",
}

# 영어 동의어 → 표준화
EN_SYNONYM_MAP = {
    "female vocal":"soft female voice",
    "male vocal":"soft male voice",
    "female voice":"soft female voice",
    "male voice":"soft male voice",
    "mixed vocal":"mixed vocals",
}

def _dedup(seq: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for s in seq:
        k = (s or "").strip().lower()
        if not k: continue
        if k not in seen:
            seen.add(k); out.append(k)
    return out

def normalize_tags_to_english(tags: Iterable[str]) -> List[str]:
    """입력(한/영 뒤섞임) → 영문 표준 태그만, 소문자·중복 제거."""
    out = []
    for t in tags:
        if not t: continue
        s = t.strip()
        # 한글 → 영문
        s = KO_EN_TAG_MAP.get(s, s)
        # 영어 동의어 정규화
        s_l = s.lower()
        s = EN_SYNONYM_MAP.get(s_l, s_l)
        # 화이트리스트 필터
        if s in ACE_TAG_WHITELIST:
            out.append(s)
    return _dedup(out)



def _submit_and_wait(
    base: str,
    wf_graph: dict,
    timeout: int | float | None = None,
    poll: float | None = None,
    on_progress=None
) -> dict[str, Any]:
    """
    ComfyUI /prompt 제출 후 /history/<id>를 폴링하여 완료까지 대기.
    반환: history의 해당 entry(dict[str, Any]) — outputs/status 포함
    """


    # 안전한 기본값 결정
    timeout_val = float(timeout if timeout is not None else _DEFAULT_ACE_WAIT_TIMEOUT_SEC)
    poll_val = float(poll if poll is not None else _DEFAULT_ACE_POLL_INTERVAL_SEC)

    _dlog("WAIT-START", f"timeout={timeout_val}", f"poll={poll_val}")

    # 0) 서버 핑
    try:
        url_stats = f"{base.rstrip('/')}/system_stats"
        _dlog("PING", url_stats)
        r = requests.get(url_stats, timeout=5.0)
        _dlog("PING-RESP", f"status={r.status_code}")
        if r.status_code != 200:
            raise RuntimeError(f"ComfyUI 응답 코드 {r.status_code}")
    except Exception as e:
        _dlog("PING-FAIL", f"{type(e).__name__}: {e}")
        raise ConnectionError(f"ComfyUI에 연결 실패: {base} ({e})")

    # 1) 제출
    try:
        url_prompt = f"{base.rstrip('/')}/prompt"
        _dlog("POST-/prompt", f"url={url_prompt}", f"nodes={len(wf_graph)}")
        r = requests.post(url_prompt, json={"prompt": wf_graph}, timeout=(5.0, 25.0))
        _dlog("POST-RESP", f"ok={r.ok}", f"status={r.status_code}")
    except Exception as e:
        _dlog("POST-FAIL", f"{type(e).__name__}: {e}")
        raise

    if not r.ok:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        _dlog("POST-NOK-BODY", str(detail)[:400])
        raise RuntimeError(f"/prompt 제출 실패: {r.status_code} {detail}")

    try:
        resp: Dict[str, Any] = r.json() or {}
    except Exception as e:
        _dlog("POST-JSON-FAIL", f"{type(e).__name__}: {e}", f"text={r.text[:400]}")
        raise

    prompt_id = resp.get("prompt_id") or resp.get("promptId") or resp.get("id")
    if not prompt_id:
        raise RuntimeError("ComfyUI가 prompt_id를 반환하지 않았습니다.")
    prompt_key = str(prompt_id)
    _dlog("PROMPT-ID", prompt_key)

    t0 = time.time()
    last_outputs = 0
    last_hist_status = None
    tick = 0

    last_queue_pending = None
    last_queue_running = None
    idle_ticks_after_output = 0

    while True:
        tick += 1
        elapsed = time.time() - t0
        if elapsed > timeout_val:
            _dlog("TIMEOUT", f"elapsed={elapsed:.1f}s")
            raise TimeoutError(
                "ACE-Step 대기 시간 초과\n"
                "- ComfyUI 실행/COMFY_HOST 확인\n"
                "- SaveAudio 노드 타입/옵션 확인 (mp3/wav/opus)\n"
                "- 큐가 길면 대기 또는 서버 재시작\n"
                "- 첫 실행 시 모델 다운로드 완료까지 대기\n"
            )

        # 큐 상태
        try:
            q = requests.get(f"{base.rstrip('/')}/queue", timeout=5.0)
            if q.ok:
                qj: Dict[str, Any] = cast(Dict[str, Any], q.json() or {})
                last_queue_pending = len(cast(list, qj.get("queue_pending", []) or []))
                last_queue_running = len(cast(list, qj.get("queue_running", []) or []))
                if on_progress:
                    on_progress({
                        "stage": "queue",
                        "pending": last_queue_pending,
                        "running": last_queue_running,
                        "elapsed": elapsed,
                    })
                if tick % 5 == 0:
                    _dlog("QUEUE", f"pending={last_queue_pending}", f"running={last_queue_running}")
        except Exception as e:
            if tick % 5 == 0:
                _dlog("QUEUE-READ-FAIL", f"{type(e).__name__}: {e}")

        # 히스토리
        try:
            h = requests.get(f"{base.rstrip('/')}/history/{prompt_key}", timeout=10.0)
            if not h.ok:
                if tick % 5 == 0:
                    _dlog("HIST-HTTP", f"status={h.status_code}")
                time.sleep(poll_val)
                continue
            hist_raw: Any = h.json()
            hist: Dict[str, Any] = cast(Dict[str, Any], hist_raw or {})
        except Exception as e:
            if tick % 5 == 0:
                _dlog("HIST-READ-FAIL", f"{type(e).__name__}: {e}")
            time.sleep(poll_val)
            continue

        entry_obj = hist.get(prompt_key)
        if not isinstance(entry_obj, dict):
            if tick % 5 == 0:
                _dlog("HIST-NO-ENTRY-YET")
            time.sleep(poll_val)
            continue
        entry_dict: Dict[str, Any] = cast(Dict[str, Any], entry_obj)

        # 상태/키 로깅(1회성)
        st_obj = (entry_dict.get("status") or {})
        st = st_obj.get("status") or st_obj.get("status_str")
        if st != last_hist_status:
            _dlog("HIST-STATUS", st, "| keys:", list(entry_dict.keys()))
            if st_obj:
                _dlog("HIST-STATUS-OBJ", {k: st_obj.get(k) for k in list(st_obj.keys())[:6]})
            last_hist_status = st

        # 에러
        if (st or "").lower() == "error":
            err = st_obj.get("error") or {}
            node_errors = err.get("node_errors") or {}
            details = []
            for nid, ne in node_errors.items():
                details.append(f"node {nid}: {ne.get('message') or ne}")
            msg = err.get("message") or "ComfyUI 내부 에러"
            _dlog("HIST-ERROR", msg, "|", " / ".join(details))
            raise RuntimeError(f"ComfyUI 에러: {msg}\n" + ("\n".join(details) if details else ""))

        # 출력 수 변화
        outs = entry_dict.get("outputs") or {}
        n_outs = sum(len(v or []) for v in (outs.values() if isinstance(outs, dict) else []))
        if n_outs != last_outputs:
            last_outputs = n_outs
            _dlog("HIST-OUTPUTS", f"count={n_outs}")
            if on_progress:
                on_progress({"stage": "running", "outputs": n_outs, "elapsed": elapsed})

        # 정상 완료
        exec_info = (st_obj.get("exec_info") or {})
        queue_info = (exec_info.get("queue") or "")
        if (st in ("success", "completed", "ok")) or ("completed" in str(queue_info).lower()):
            _dlog("HIST-DONE", f"elapsed={elapsed:.1f}s", f"outputs={n_outs}")
            if on_progress:
                on_progress({"stage": "completed", "outputs": n_outs, "elapsed": elapsed})
            return entry_dict  # ✅

        # 휴리스틱 완료: 출력이 있고 큐가 비었으면 몇 틱 후 완료 간주
        if n_outs > 0 and last_queue_pending == 0 and last_queue_running == 0:
            idle_ticks_after_output += 1
            if idle_ticks_after_output == 1:
                _dlog("HIST-HEURISTIC-ARMED", "outputs>0 & queue empty -> waiting few ticks")
            if idle_ticks_after_output >= 3:
                _dlog("HIST-HEURISTIC-DONE", f"elapsed={elapsed:.1f}s", f"outputs={n_outs}")
                if on_progress:
                    on_progress({"stage": "completed(heuristic)", "outputs": n_outs, "elapsed": elapsed})
                return entry_dict
        else:
            idle_ticks_after_output = 0

        time.sleep(poll_val)

    # 타입 체커 안심용 (정상 흐름상 도달 불가)
    return {}  # type: ignore[return-value]



_LOG_PATH = Path(BASE_DIR) / "music_gen.log"  # 언제나 여기로도 기록
def _dlog(*args):
    msg = " ".join(str(a) for a in args)
    line = f"[MUSIC] {msg}"
    try:
        # 1) stdout (flush)
        print(line, flush=True)
    except Exception:
        pass
    try:
        # 2) stderr (일부 런처에서 stdout이 안 보일 때 대비)
        import sys
        print(line, file=sys.stderr, flush=True)
    except Exception:
        pass
    try:
        # 3) 파일 로그 (GUI에서도 100% 확인 가능)
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────
# 유틸: 오디오, 영살 길이 견고 획득(중앙값, 3배 튐 방지)
# ─────────────────────────────────────────────────────────────
# real_use
def _guess_ffprobe_exe() -> str:
    """
    ffprobe 실행 파일 경로 추정.
    1) settings.FFPROBE_EXE 있으면 사용
    2) settings.FFMPEG_EXE가 ffmpeg.exe면 같은 폴더의 ffprobe.exe로 추정
    3) 최후: 'ffprobe'
    """
    try:
        ffprobe_exe = getattr(settings, "FFPROBE_EXE", "")  # type: ignore
        if ffprobe_exe and os.path.isfile(ffprobe_exe):
            return ffprobe_exe
    except Exception:
        pass

    try:
        ffmpeg_exe = getattr(settings, "FFMPEG_EXE", "")  # type: ignore
        if ffmpeg_exe:
            ffmpeg_exe = str(ffmpeg_exe)
            if os.path.isfile(ffmpeg_exe):
                d = os.path.dirname(ffmpeg_exe)
                cand = os.path.join(d, "ffprobe.exe" if os.name == "nt" else "ffprobe")
                if os.path.isfile(cand):
                    return cand
    except Exception:
        pass

    return "ffprobe"


def _probe_duration_ffprobe(path: str) -> float:
    import json
    try:
        ffprobe = _guess_ffprobe_exe()
        out = subprocess.check_output(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
            stderr=subprocess.STDOUT,
        )
        data = json.loads(out.decode("utf-8", "ignore"))
        dur = float(data.get("format", {}).get("duration", 0.0) or 0.0)
        return dur if math.isfinite(dur) and dur > 0 else 0.0
    except Exception:
        return 0.0

# real_use
def _probe_duration_mutagen(path: str) -> float:
    try:
          # type: ignore
        mf = MutagenFile(path)
        dur = float(getattr(mf, "info", None).length if mf and getattr(mf, "info", None) else 0.0)
        return dur if math.isfinite(dur) and dur > 0 else 0.0
    except Exception:
        return 0.0
# real_use
def _probe_duration_pydub(path: str) -> float:
    try:

        dur_ms = len(AudioSegment.from_file(path))
        dur = float(dur_ms) / 1000.0
        return dur if math.isfinite(dur) and dur > 0 else 0.0
    except Exception:
        return 0.0
# real_use
def _probe_duration_soundfile(path: str) -> float:
    try:
        import soundfile as zsf  # type: ignore
        with zsf.SoundFile(path) as f:
            dur = float(len(f) / (f.samplerate or 1))
            return dur if math.isfinite(dur) and dur > 0 else 0.0
    except Exception:
        return 0.0
# real_use
def _probe_duration_wave(path: str) -> float:
    try:
        import wave
        with wave.open(path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 1
            dur = float(frames) / float(rate)
            return dur if math.isfinite(dur) and dur > 0 else 0.0
    except Exception:
        return 0.0
# real_use
def _probe_duration_librosa(path: str) -> float:
    try:
        import librosa  # type: ignore
        try:
            dur = float(librosa.get_duration(path=path))
        except TypeError:
            dur = float(librosa.get_duration(filename=path))
        return dur if math.isfinite(dur) and dur > 0 else 0.0
    except Exception:
        return 0.0
# real_use
def get_duration(path: str) -> float:
    p = str(path or "").strip()
    if not p or not os.path.isfile(p):
        return 0.0
    cands: List[float] = []
    for fn in (
        _probe_duration_ffprobe,
        _probe_duration_mutagen,
        _probe_duration_pydub,
        _probe_duration_soundfile,
        _probe_duration_wave,
        # _probe_duration_librosa,
    ):
        v = fn(p)
        if v and math.isfinite(v) and v > 0:
            cands.append(v)
    if not cands:
        return 0.0
    cands.sort()
    mid = len(cands) // 2
    if len(cands) % 2 == 1:
        median = cands[mid]
    else:
        median = 0.5 * (cands[mid - 1] + cands[mid])
    mn, mx = cands[0], cands[-1]
    if mx / max(mn, 1e-9) >= 2.4 and mn > 0:
        return mn
    return median
# ============================================================== #

# ================================================================================================= #
# ===========================================자막 관련============================================== #
# ================================================================================================= #

# ===========폰트 가져오기============== #
# 폰트 유틸
def resolve_windows_fontfile(font_family: str) -> Optional[str]:
    """
    Windows 폰트 패밀리/표시명(예: '맑은 고딕', 'Malgun Gothic')을
    실제 폰트 파일 경로(C:\\Windows\\Fonts\\...)로 해석한다.

    - 1) alias(한글/영문/별칭) 기반으로 Fonts 폴더에서 빠르게 탐색
    - 2) 레지스트리(Fonts)에서 표시명/값 매칭
    - 3) 최후수단: Fonts 폴더 파일명 근사검색

    설치되어 있지 않으면 None
    """
    if not font_family:
        return None

    s = str(font_family).strip()
    if not s:
        return None

    # 이미 파일 경로가 들어온 경우
    try:
        p = Path(s)
        if p.is_file():
            return str(p)
    except Exception:
        pass

    # 캐시(함수 속성 이용: 모듈 전역 오염 최소화)
    cache: Dict[str, Optional[str]] = getattr(resolve_windows_fontfile, "_cache", {})
    if s in cache:
        return cache[s]

    def _norm(x: str) -> str:
        return "".join(ch.lower() for ch in x if ch.isalnum())

    fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    target = _norm(s)

    # -----------------------------
    # 0) Alias/파일 후보 (핵심)
    # -----------------------------
    # UI가 한글/영문/다른 표기 무엇을 주더라도, 실제 파일로 떨어지게 만든다.
    alias_to_files: Dict[str, List[str]] = {
        # Gulim
        "굴림": ["gulim.ttc"],
        "gulim": ["gulim.ttc"],
        "gulimche": ["gulim.ttc"],

        # Malgun Gothic
        "맑은고딕": ["malgun.ttf", "malgunbd.ttf", "malgunsl.ttf"],
        "맑은": ["malgun.ttf", "malgunbd.ttf", "malgunsl.ttf"],
        "malgungothic": ["malgun.ttf", "malgunbd.ttf", "malgunsl.ttf"],
        "malgun": ["malgun.ttf", "malgunbd.ttf", "malgunsl.ttf"],

        # Dotum
        "돋움": ["dotum.ttf", "dotumche.ttf"],
        "dotum": ["dotum.ttf", "dotumche.ttf"],

        # Batang
        "바탕": ["batang.ttc"],
        "batang": ["batang.ttc"],

        # Gungsuh
        "궁서": ["gungsuh.ttc"],
        "gungsuh": ["gungsuh.ttc"],
    }

    # target이 완전히 일치하거나(예: '굴림'), 공백 제거 형태('맑은 고딕'->'맑은고딕')도 고려
    target_no_space = _norm(s.replace(" ", ""))

    # alias 키 후보(우선순위 높은 것부터)
    alias_keys = []
    for k in {target, target_no_space}:
        if k in alias_to_files:
            alias_keys.append(k)

    # alias 후보가 있으면 Fonts 폴더에서 즉시 탐색
    if fonts_dir.is_dir() and alias_keys:
        for ak in alias_keys:
            for fn in alias_to_files.get(ak, []):
                fp = (fonts_dir / fn)
                if fp.is_file():
                    cache[s] = str(fp)
                    setattr(resolve_windows_fontfile, "_cache", cache)
                    return str(fp)

    # -----------------------------
    # 1) 레지스트리에서 폰트 목록 수집
    # -----------------------------
    entries: Dict[str, str] = {}
    try:
        import winreg  # Windows 전용

        reg_paths = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"),
            (winreg.HKEY_CURRENT_USER,  r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"),
        ]

        for root, subkey in reg_paths:
            try:
                with winreg.OpenKey(root, subkey) as k:
                    i = 0
                    while True:
                        try:
                            name, value, _ = winreg.EnumValue(k, i)
                            i += 1
                            if not name or not value:
                                continue
                            entries[name] = str(value)
                        except OSError:
                            break
            except OSError:
                pass
    except Exception:
        entries = {}

    # -----------------------------
    # 2) 레지스트리 이름/값에서 매칭
    # -----------------------------
    best: Optional[Tuple[int, str]] = None  # (score, path)

    # alias가 있으면 그 alias도 같이 target 후보로 추가
    target_candidates = {target, target_no_space}
    if alias_keys:
        target_candidates.update(alias_keys)

    for disp_name, val in entries.items():
        disp_norm = _norm(disp_name)
        val_norm = _norm(val)

        score = 0
        # 여러 후보 중 하나라도 매칭되면 점수 부여
        for tc in target_candidates:
            if not tc:
                continue
            if disp_norm == tc:
                score = max(score, 100)
            elif tc in disp_norm:
                score = max(score, 80)
            elif tc in val_norm:
                score = max(score, 60)

        if score <= 0:
            continue

        vp = Path(val)
        if not vp.is_absolute():
            vp = (fonts_dir / val).resolve()

        if vp.is_file():
            if best is None or score > best[0]:
                best = (score, str(vp))

    if best:
        cache[s] = best[1]
        setattr(resolve_windows_fontfile, "_cache", cache)
        return best[1]

    # -----------------------------
    # 3) 최후수단: Fonts 폴더 파일명 근사검색
    # -----------------------------
    try:
        if fonts_dir.is_dir() and target:
            # (중요) target이 한글이면 파일명이 영문일 가능성이 높다.
            # 그래서 alias가 있으면 alias 후보 파일도 한 번 더 시도한다.
            if alias_keys:
                for ak in alias_keys:
                    for fn in alias_to_files.get(ak, []):
                        fp = (fonts_dir / fn)
                        if fp.is_file():
                            cache[s] = str(fp)
                            setattr(resolve_windows_fontfile, "_cache", cache)
                            return str(fp)

            for fp in fonts_dir.iterdir():
                if not fp.is_file():
                    continue
                if fp.suffix.lower() not in (".ttf", ".ttc", ".otf"):
                    continue
                if target in _norm(fp.name):
                    cache[s] = str(fp)
                    setattr(resolve_windows_fontfile, "_cache", cache)
                    return str(fp)
    except Exception:
        pass

    cache[s] = None
    setattr(resolve_windows_fontfile, "_cache", cache)
    return None

# ===========자막 두줄 처리============== #


def split_subtitle_two_lines(text: str, max_units: float = 20.0) -> Tuple[str, str]:
    """
    자막을 2줄로 분할한다.
    - 한글/비ASCII: 1.0
    - 영어/숫자/ASCII: 0.5
    - 공백: 0.5
    - max_units 초과하면 2줄로 나누되, 두 줄의 유닛 수가 최대한 비슷하게(균형)
    - 나누는 위치는 가능하면 쉼표/마침표/공백 등 자연스러운 경계 우선
    - 2줄로도 2번째 줄이 max_units를 넘으면, 2번째 줄을 max_units에 맞춰 잘라 '…' 추가
    """
    s = (text or "").strip()
    if not s:
        return "", ""

    # normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()

    def _unit(ch: str) -> float:
        if ch == " ":
            return 0.5
        o = ord(ch)
        # ASCII(영문/숫자/기호) = 0.5
        if o < 128:
            return 0.5
        # 그 외(한글 포함) = 1.0
        return 1.0

    def _units(t: str) -> float:
        return sum(_unit(c) for c in t)

    def _cut_to_units(t: str, lim: float) -> str:
        acc = 0.0
        out = []
        for c in t:
            u = _unit(c)
            if acc + u > lim:
                break
            out.append(c)
            acc += u
        return "".join(out).rstrip()

    total = _units(s)
    if total <= max_units:
        return s, ""

    # 후보 분할점: 문장부호/공백 근처를 우선
    # "자연스러움" 가중치: 문장부호(최고) > 쉼표류 > 공백 > 그 외
    punct_strong = set(".!?。！？")
    punct_mid = set(",，、;:：·")

    # 목표: 반으로 나누되 line1이 max_units 넘지 않게
    target = total * 0.5

    # 모든 인덱스에서 누적 유닛 계산(한 번만)
    cum = [0.0]
    for c in s:
        cum.append(cum[-1] + _unit(c))

    # 가능한 분할점: 1..len-1
    candidates = []
    for i in range(1, len(s)):
        left_u = cum[i]
        right_u = total - left_u
        if left_u > max_units:  # 1줄 max 초과는 불가
            break

        prev = s[i - 1]
        nxt = s[i] if i < len(s) else ""

        # 경계 점수(낮을수록 좋음)
        # 기본: 균형(왼/오 차이)
        balance_pen = abs(left_u - right_u)

        # 자연스러운 끊김 보너스(패널티를 줄임)
        cut_bonus = 0.0
        if prev in punct_strong:
            cut_bonus = 4.0
        elif prev in punct_mid:
            cut_bonus = 3.0
        elif prev == " ":
            cut_bonus = 2.0
        elif nxt == " ":
            cut_bonus = 1.5

        # 너무 짧은 줄 방지(각 줄 최소 4유닛 정도는 유지하려고 약한 패널티)
        short_pen = 0.0
        if left_u < 4.0 or right_u < 4.0:
            short_pen = 3.0

        score = balance_pen + short_pen - cut_bonus
        candidates.append((score, i))

    # 후보가 없으면 강제 컷
    if not candidates:
        line1 = _cut_to_units(s, max_units)
        line2 = s[len(line1):].lstrip()
        if _units(line2) > max_units:
            line2 = _cut_to_units(line2, max(0.0, max_units - 0.5)).rstrip() + "…"
        return line1, line2

    candidates.sort(key=lambda x: x[0])
    _, best_i = candidates[0]

    line1 = s[:best_i].rstrip()
    line2 = s[best_i:].lstrip()

    # 2번째 줄이 너무 길면 잘라서 말줄임
    if _units(line2) > max_units:
        line2 = _cut_to_units(line2, max(0.0, max_units - 0.5)).rstrip() + "…"

    return line1, line2


def _ffmpeg_escape_drawtext(s: str) -> str:
    """
    FFmpeg drawtext 안전 이스케이프(기본형).
    - 백슬래시/콜론/작은따옴표/퍼센트/개행 처리
    - 개행은 filtergraph 파서 단계 때문에 '\\\\n' 으로 넣어야 drawtext에서 '\n'으로 인식됨
    """
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\\", "\\\\")
    s = s.replace(":", "\\:")
    s = s.replace("'", "\\'")
    s = s.replace("%", "\\%")
    # ✅ 여기 중요: \n -> \\n 로 “필터 파서용” 이스케이프
    s = s.replace("\r\n", "\n")
    s = s.replace("\n", "\\\\n")
    s = s.replace("\r", "")
    return s



# 경로 유틸
def _ffmpeg_escape_filter_path(p: str) -> str:
    """
    FFmpeg filter 옵션 값으로 들어가는 '경로' 이스케이프
    - Windows 드라이브 콜론 C: 를 \: 로 이스케이프해야 함
    """
    if p is None:
        return ""
    p = str(p).replace("\\", "/")
    p = p.replace("\\", "\\\\")
    p = p.replace(":", "\\:")
    p = p.replace("'", "\\'")
    return p

# ====================================================자막 클래스 모음 ========================================




# ============================================================
# SubtitleComposer: drawtext 자막을 "재사용 가능"하게 구성
# - max_units(한줄 제한), max_lines(줄수 제한), line_gap_px(줄간격), lift_ratio(전체 올림)
# - Windows 폰트 resolve 포함(인스턴스 캐시)
# - 개행문자(\n)로 drawtext 하는 방식은 Windows에서 깨질 수 있으므로,
#   drawtext N개를 쌓는 방식으로 안정 구현
# ============================================================

@dataclass
class SubtitleStyle:
    font_family: str = "Gulim"
    fontsize: int = 36
    y: str = "h-140"

    box: bool = True
    boxcolor: str = "black@0.45"
    boxborderw: int = 18

    max_units: float = 20.0
    max_lines: int = 2

    line_gap_px: Optional[int] = None

    # ✅ 3줄 이상에서 화면 밖으로 나가는 문제를 줄이기 위해 기본 lift 강화
    # (원래 0.62는 2줄엔 좋지만 3줄에 부족한 케이스가 많음)
    lift_ratio: float = 0.88



class SubtitleComposer:
    """
    SubtitleComposer는 "자막(drawtext) 관련 책임"을 전부 가진다.

    외부(호출자)가 주는 값:
      - font_family / fontsize / y / box 옵션
      - max_units (한줄 제한)
      - max_lines (줄수 제한: 2/3/4...)
      - line_gap_px (줄간격: None이면 자동)
      - lift_ratio (줄수 많아질 때 y를 얼마나 위로 올릴지)
    """

    def __init__(
        self,
        *,
        font_family: str = "Gulim",
        fontsize: int = 36,
        y: str = "h-120",
        box: bool = True,
        boxcolor: str = "black@0.45",
        boxborderw: int = 18,
        max_units: float = 20.0,
        max_lines: int = 3,
        line_gap_px: Optional[int] = None,
        lift_ratio: float = 0.62,
    ) -> None:
        self.style = SubtitleStyle(
            font_family=str(font_family or "").strip() or "Gulim",
            fontsize=int(fontsize),
            y=str(y),
            box=bool(box),
            boxcolor=str(boxcolor),
            boxborderw=int(boxborderw),
            max_units=float(max_units),
            max_lines=max(1, int(max_lines)),
            line_gap_px=(None if line_gap_px is None else int(line_gap_px)),
            lift_ratio=float(lift_ratio),
        )

        # ✅ 인스턴스 캐시 (함수 속성 _cache 쓰지 않음 → '_cache' 에러 방지)
        self._font_cache: Dict[str, Optional[str]] = {}

        # ✅ font_arg를 미리 확정해두면, 호출자는 "그대로 drawtext에 쓴다"
        self.fontfile = self._resolve_windows_fontfile(self.style.font_family)
        if self.fontfile:
            fontfile_ff = _ffmpeg_escape_filter_path(str(self.fontfile))
            self.font_arg = f"fontfile='{fontfile_ff}'"
        else:
            # 최후수단(환경에 따라 fontconfig 문제 가능)
            self.font_arg = f"font='{self.style.font_family}'"

    # ----------------------------
    # (A) 폰트 resolve (Composer 내부)
    # ----------------------------
    def _resolve_windows_fontfile(self, font_family: str) -> Optional[str]:
        """
        Windows 폰트 패밀리/표시명(예: '굴림', 'Malgun Gothic')을 실제 폰트 파일 경로로 해석.
        - alias 기반 빠른 탐색
        - 레지스트리 Fonts 매칭
        - 최후수단: Fonts 폴더 파일명 근사검색
        """
        s = (font_family or "").strip()
        if not s:
            return None

        # 캐시
        if s in self._font_cache:
            return self._font_cache[s]

        # 이미 파일경로로 들어온 경우
        try:
            p = Path(s)
            if p.is_file():
                self._font_cache[s] = str(p)
                return str(p)
        except Exception:
            pass

        def _norm(x: str) -> str:
            return "".join(ch.lower() for ch in x if ch.isalnum())

        fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        target = _norm(s)
        target_no_space = _norm(s.replace(" ", ""))

        alias_to_files: Dict[str, List[str]] = {
            # Gulim
            "굴림": ["gulim.ttc"],
            "gulim": ["gulim.ttc"],
            "gulimche": ["gulim.ttc"],

            # Malgun Gothic
            "맑은고딕": ["malgun.ttf", "malgunbd.ttf", "malgunsl.ttf"],
            "맑은": ["malgun.ttf", "malgunbd.ttf", "malgunsl.ttf"],
            "malgungothic": ["malgun.ttf", "malgunbd.ttf", "malgunsl.ttf"],
            "malgun": ["malgun.ttf", "malgunbd.ttf", "malgunsl.ttf"],

            # Dotum
            "돋움": ["dotum.ttf", "dotumche.ttf"],
            "dotum": ["dotum.ttf", "dotumche.ttf"],

            # Batang
            "바탕": ["batang.ttc"],
            "batang": ["batang.ttc"],

            # Gungsuh
            "궁서": ["gungsuh.ttc"],
            "gungsuh": ["gungsuh.ttc"],
        }

        alias_keys: List[str] = []
        for k in {target, target_no_space, _norm(s.replace(" ", ""))}:
            if k in alias_to_files:
                alias_keys.append(k)

        # 0) alias로 즉시 탐색
        if fonts_dir.is_dir() and alias_keys:
            for ak in alias_keys:
                for fn in alias_to_files.get(ak, []):
                    fp = fonts_dir / fn
                    if fp.is_file():
                        self._font_cache[s] = str(fp)
                        return str(fp)

        # 1) 레지스트리에서 목록 수집
        entries: Dict[str, str] = {}
        try:
            import winreg  # Windows 전용

            reg_paths = [
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"),
                (winreg.HKEY_CURRENT_USER,  r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"),
            ]
            for root, subkey in reg_paths:
                try:
                    with winreg.OpenKey(root, subkey) as k:
                        i = 0
                        while True:
                            try:
                                name, value, _ = winreg.EnumValue(k, i)
                                i += 1
                                if name and value:
                                    entries[str(name)] = str(value)
                            except OSError:
                                break
                except OSError:
                    pass
        except Exception:
            entries = {}

        # 2) 레지스트리 기반 매칭
        best: Optional[Tuple[int, str]] = None
        target_candidates = {target, target_no_space}
        if alias_keys:
            target_candidates.update(alias_keys)

        for disp_name, val in entries.items():
            disp_norm = _norm(disp_name)
            val_norm = _norm(val)

            score = 0
            for tc in target_candidates:
                if not tc:
                    continue
                if disp_norm == tc:
                    score = max(score, 100)
                elif tc in disp_norm:
                    score = max(score, 80)
                elif tc in val_norm:
                    score = max(score, 60)

            if score <= 0:
                continue

            vp = Path(val)
            if not vp.is_absolute():
                vp = (fonts_dir / val).resolve()

            if vp.is_file():
                if best is None or score > best[0]:
                    best = (score, str(vp))

        if best:
            self._font_cache[s] = best[1]
            return best[1]

        # 3) 최후수단: Fonts 폴더 근사검색
        try:
            if fonts_dir.is_dir() and target:
                for fp in fonts_dir.iterdir():
                    if not fp.is_file():
                        continue
                    if fp.suffix.lower() not in (".ttf", ".ttc", ".otf"):
                        continue
                    if target in _norm(fp.name) or target_no_space in _norm(fp.name):
                        self._font_cache[s] = str(fp)
                        return str(fp)
        except Exception:
            pass

        self._font_cache[s] = None
        return None

    # ----------------------------
    # (B) 글자 유닛 계산(한글=1, ASCII=0.5, 공백=0.5)
    # ----------------------------
    @staticmethod
    def _unit(ch: str) -> float:
        if ch == " ":
            return 0.5
        if ord(ch) < 128:
            return 0.5
        return 1.0

    @classmethod
    def _units(cls, s: str) -> float:
        return sum(cls._unit(c) for c in s)

    @classmethod
    def _cut_to_units(cls, s: str, lim: float) -> str:
        acc = 0.0
        out: List[str] = []
        for c in s:
            u = cls._unit(c)
            if acc + u > lim:
                break
            out.append(c)
            acc += u
        return "".join(out).rstrip()

    # ----------------------------
    # (C) 텍스트 → 최대 max_lines 줄로 "단어 기준" 분할 + 균형
    # ----------------------------
    def split_lines(self, text: str) -> List[str]:
        """
        ✅ 요구사항 반영
        1) max_lines는 '최대 줄 수' (억지로 3줄 맞추지 않음)
        2) 2줄 이상이면 위/아래 글자수가 비슷하도록(균형) 분할
        3) max_lines로도 못 담으면 마지막 줄 끝에 '...' 처리
        4) 공백뿐 아니라 / - · : 등도 토큰 경계로 취급
        5) 공백 없는 긴 문자열도 강제 래핑(여러 줄) 가능
        """
        import math
        import re

        s = (text or "").strip()
        if not s:
            return []
        s = re.sub(r"\s+", " ", s).strip()

        max_units = float(self.style.max_units)  # 한 줄 폭 제한
        max_lines = max(1, int(self.style.max_lines))  # 최대 줄 수

        # -----------------------------
        # 토큰화: 공백 + 구분자 경계
        # -----------------------------
        seps = set("/-·:|,;()[]{}")
        tokens: List[str] = []
        buf: List[str] = []

        for ch in s:
            if ch == " ":
                if buf:
                    tokens.append("".join(buf))
                    buf = []
                continue
            if ch in seps:
                if buf:
                    buf.append(ch)  # 구분자는 앞 토큰에 붙임
                    tokens.append("".join(buf))
                    buf = []
                else:
                    tokens.append(ch)
                continue
            buf.append(ch)

        if buf:
            tokens.append("".join(buf))

        if not tokens:
            return []

        # ✅ 이 클래스에 존재하는 유닛 함수는 _units / _unit / _cut_to_units 이다.
        # ( _calc_units 같은 건 없음 )
        tok_units = [self._units(t) for t in tokens]
        n = len(tokens)

        # 토큰 사이 공백 유닛(이 클래스 정책: 공백 0.5)
        space_u = 0.5

        def seg_units(i: int, j: int) -> float:
            if i >= j:
                return 0.0
            u = sum(tok_units[i:j])
            u += space_u * max(0, (j - i - 1))
            return float(u)

        total_u = seg_units(0, n)

        # ✅ 1줄로 들어가면 1줄 (억지 분할 금지)
        if total_u <= max_units + 1e-9:
            return [" ".join(tokens).strip()]

        # ✅ 필요한 최소 줄 수 (max_lines 초과 금지)
        needed = int(math.ceil(total_u / max_units))
        target_max = min(max_lines, max(2, needed))

        INF = 10 ** 18

        def solve_exact_lines(L: int) -> Optional[List[str]]:
            # 정확히 L줄로 배치하는 DP (각 줄 slack^2 최소화)
            dp = [[INF] * (n + 1) for _ in range(L + 1)]
            nxt = [[-1] * (n + 1) for _ in range(L + 1)]
            dp[L][n] = 0

            for k in range(L - 1, -1, -1):
                for i in range(n - 1, -1, -1):
                    best_cost = INF
                    best_j = -1
                    for j in range(i + 1, n + 1):
                        u = seg_units(i, j)
                        if u > max_units + 1e-9:
                            break
                        slack = max_units - u
                        pen = slack * slack

                        prev_tok = tokens[j - 1]
                        if prev_tok and prev_tok[-1] in (".", "!", "?", ",", "…"):
                            pen *= 0.92

                        cand = pen + dp[k + 1][j]
                        if cand < best_cost:
                            best_cost = cand
                            best_j = j

                    dp[k][i] = best_cost
                    nxt[k][i] = best_j

            if nxt[0][0] == -1:
                return None

            out: List[str] = []
            i = 0
            for k in range(L):
                j = nxt[k][i]
                if j <= i:
                    return None
                out.append(" ".join(tokens[i:j]).strip())
                i = j
            if i != n:
                return None
            return out

        # ✅ 가능한 최소 줄 수부터 시도 (억지 max_lines 맞춤 금지)
        lines: Optional[List[str]] = None
        for L in range(2, target_max + 1):
            got = solve_exact_lines(L)
            if got is not None:
                lines = got
                break

        # DP 실패(주로 공백 없는 긴 토큰) → 문자 단위 강제 래핑
        if lines is None:
            hard: List[str] = []
            cur = ""
            cur_u = 0.0
            for ch in s:
                u = self._unit(ch)  # ✅ 단일 문자 유닛은 _unit 사용
                if cur and cur_u + u > max_units + 1e-9:
                    hard.append(cur.rstrip())
                    cur = ch
                    cur_u = u
                else:
                    cur += ch
                    cur_u += u
            if cur.strip():
                hard.append(cur.rstrip())
            lines = hard

        # ✅ max_lines 초과면: 마지막 줄에 ... (정확히 요구사항)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            last = lines[-1]
            last_cut = self._cut_to_units(last, max(0.0, max_units - 3.0)).rstrip()
            lines[-1] = (last_cut + "...") if last_cut else "..."
            return lines

        # ✅ 마지막 줄이 혹시 제한을 넘으면 안전하게 ... 처리
        if lines:
            last = lines[-1]
            if self._units(last) > max_units + 1e-9:
                last_cut = self._cut_to_units(last, max(0.0, max_units - 3.0)).rstrip()
                lines[-1] = (last_cut + "...") if last_cut else "..."

        return lines

    # ----------------------------
    # (D) 줄 간격 계산 + y 자동 올림(화면 잘림 방지)
    # ----------------------------
    def _calc_line_gap_px(self) -> int:
        """
        ✅ 줄간격: 더 넓게(체감 개선)
        - style.line_gap_px가 있으면 그 값을 최우선
        - 자동일 때는 fontsize 기반으로 이전보다 확실히 띄움
        """
        if self.style.line_gap_px is not None:
            return max(16, int(self.style.line_gap_px))

        # 이전이 빡빡했다면, 1.45 수준으로 올려야 체감이 납니다.
        base_gap = int(round(float(self.style.fontsize) * 1.45))
        border_pad = int(max(0, self.style.boxborderw))
        return max(34, base_gap + int(round(border_pad * 0.5)))

    def calc_y_lines(self, base_y: str, line_count: Optional[int] = None, line_gap_px: Optional[int] = None) -> List[str]:
        """
        줄 수가 늘어날수록 첫 줄을 더 위로 올려서 블록이 화면 아래로 밀리지 않게 한다.
        - base_y는 'h-140' 같은 ffmpeg 표현 또는 숫자 문자열/정수 모두 허용
        - line_count/line_gap_px가 None이면 안전한 기본값으로 계산 (호출부 실수로 인한 크래시 방지)
        """
        import re

        n = max(1, int(line_count) if line_count is not None else 1)
        gap = max(1, int(line_gap_px) if line_gap_px is not None else self._calc_line_gap_px())

        # base_y 파싱: "h-140" 형태면 140을 base로 사용
        m = re.match(r"^\s*h\s*-\s*(\d+)\s*$", str(base_y))
        if not m:
            # 파싱 실패(예: "200") → 그대로 사용, n줄이면 +gap 누적
            ys = [str(base_y)]
            for k in range(1, n):
                ys.append(f"{base_y}+{k * gap}")
            return ys

        base = int(m.group(1))

        # 전체 블록 높이(대략): n줄이면 (n-1)*gap 만큼 아래로 확장
        total_h = (n - 1) * gap

        # 기본 lift: 블록의 절반 정도를 위로 당김(3줄 이상 안전)
        lift_ratio = float(getattr(self.style, "lift_ratio", 0.88))
        lift_px = int(round(total_h * 0.5 * lift_ratio))

        # 3줄 이상은 추가로 더 올림(화면 밖으로 나가는 문제 방지)
        if n >= 3:
            lift_px += int(round((n - 2) * gap * 0.65))

        y0 = base + lift_px

        ys = [f"h-{y0}"]
        for k in range(1, n):
            ys.append(f"h-{y0}+{k * gap}")
        return ys


    # ----------------------------
    # (E) drawtext 문자열 생성(여기가 재사용의 핵심)
    # ----------------------------
    def build_subtitle_drawtexts(
            self,
            *,
            text: str,
            show_t: float,
            hide_t: float,
            alpha_expr: str,
            fontcolor: str = "white",
            x_expr: str = "(w-text_w)/2",
    ) -> List[str]:
        """
        입력 텍스트를 max_lines로 분할 → 각 줄 drawtext 필터를 만들어 반환.
        호출자는 vf_parts.extend(...)만 하면 된다.
        """
        txt = (text or "").strip()
        if not txt:
            return []

        lines = self.split_lines(txt)
        if not lines:
            return []

        # ✅ 강제 max_lines 적용(안전)
        lines = lines[: max(1, int(self.style.max_lines))]

        # ✅ line_gap / y 계산 (여기가 이번 오류의 원인 수정 포인트)
        gap = int(self._calc_line_gap_px())
        y_list = self.calc_y_lines(self.style.y, len(lines), gap)

        box = "1" if self.style.box else "0"

        out: List[str] = []
        for ln, y in zip(lines, y_list):
            ln_esc = _ffmpeg_escape_drawtext(ln)
            out.append(
                "drawtext="
                f"{self.font_arg}:"
                f"text='{ln_esc}':"
                f"fontsize={int(self.style.fontsize)}:"
                f"fontcolor={fontcolor}:"
                f"x={x_expr}:"
                f"y={y}:"
                f"box={box}:"
                f"boxcolor={self.style.boxcolor}:"
                f"boxborderw={int(self.style.boxborderw)}:"
                f"enable='between(t,{show_t:.3f},{hide_t:.3f})':"
                f"alpha='{alpha_expr}'"
            )
        return out

    def build_title_drawtext(
        self,
        *,
        title_text: str,
        title_fontsize: int,
        title_y: str,
        alpha_expr: str,
    ) -> Optional[str]:
        t = (title_text or "").strip()
        if not t:
            return None
        t_esc = _ffmpeg_escape_drawtext(t)
        return (
            "drawtext="
            f"{self.font_arg}:"
            f"text='{t_esc}':"
            f"fontsize={int(title_fontsize)}:"
            "fontcolor=white:"
            "box=1:boxcolor=black@0.5:boxborderw=6:"
            "x=(w-text_w)/2:"
            f"y={title_y}:"
            f"alpha='{alpha_expr}'"
        )



# ===============================================================================================
# ========================한영 변환기======================================================
def translate_kor_to_en_prompt(ai, kor_text: str) -> str:
    kor_text = (kor_text or "").strip()
    if not kor_text:
        return ""
    if ai is None:
        return kor_text

    sys_msg = (
        "You are a professional translator for AI image generation prompts.\n"
        "Translate Korean to natural, concise English.\n"
        "Do NOT add new details. Keep meaning faithful.\n"
        "Output English ONLY."
    )
    user_msg = f'Korean:\n"{kor_text}"\n\nEnglish:'
    try:
        out = ai.ask_smart(sys_msg, user_msg, prefer="openai")
        return (out or "").strip().replace("```", "").strip()
    except Exception:
        return kor_text



#