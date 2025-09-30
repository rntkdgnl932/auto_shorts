﻿# -*- coding: utf-8 -*-
"""
공용 유틸 함수 모음 (외부 의존성 최소화)
- 프로젝트 폴더 규칙: BASE_DIR / "maked_title" / <title>
- 결과 복사 규칙: settings.FINAL_OUT 경로의 [title] 치환
"""
from __future__ import annotations
from typing import Optional
from datetime import datetime, timezone, timedelta
import importlib.util
import shutil
import os

# ─────────────────────────────────────────────────────────────────────────────

# ===== GPT 프롬프트 재작성 유틸 (utils.py 내부에 추가) =====

# ---- 매핑 테이블 ----
# ===== GPT 프롬프트 재작성(Story Overwrite) =====
# ==== GPT 프롬프트 재작성( story.json 덮어쓰기 ) ====
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

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
        # 요청: 여성 캐릭터에는 필수 태그
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
    global_ctx = story.get("global_context", {}) or {}
    char_styles = story.get("character_styles", {}) or {}

    negative = _compose_negative(global_ctx)
    effects = _effects_to_tags(scene.get("effect", []))
    section_tags = _section_mood_tags(scene.get("section", ""), global_ctx)
    theme_palette = _themes_palette_tags(global_ctx)

    # 캐릭터 태그
    char_tags: List[str] = []
    char_refs = _parse_char_refs(scene.get("characters", []))
    for ref in char_refs:
        # ref["id"]가 int일 수도 있으므로 안전하게 문자열로 변환
        rid_str = str(ref.get("id", ""))
        ctags = _extract_char_tags(rid_str, char_styles.get(rid_str, ""))
        if isinstance(ctags, list):
            char_tags.extend(ctags)
        elif ctags:
            # 혹시 단일 문자열을 반환하는 구현이어도 안전하게 처리
            char_tags.append(str(ctags))

    # 배경 힌트(기존 prompt_img의 '배경:' 뒤만 뽑아 간단 영문 토큰화)
    bg_hint = ""
    if scene.get("prompt_img"):
        m = re.search(r"(배경[:：]\s*)(.+)", scene["prompt_img"])
        if m:
            bg_hint = m.group(2).strip()
    bg_tags = _ko_to_en_tokens(bg_hint) if bg_hint else []

    # 이미지 프롬프트
    img_prompt_parts = [
        ", ".join(char_tags) if char_tags else None,
        ", ".join(bg_tags) if bg_tags else None,
        ", ".join(theme_palette) if theme_palette else None,
        ", ".join(section_tags) if section_tags else None,
        ", ".join(effects) if effects else None,
        "photorealistic, cinematic lighting, high detail, 8k, masterpiece",
    ]
    img_prompt = ", ".join([p for p in img_prompt_parts if p])

    # 무비 프롬프트(모션)
    motion_hint = ""
    if scene.get("prompt_movie"):
        mm = re.search(r"(인물 동작[:：]\s*)(.+)", scene["prompt_movie"])
        if mm:
            motion_hint = mm.group(2).strip()
    movie_prompt = img_prompt + (f", motion: {motion_hint}" if motion_hint else "")

    if negative:
        img_prompt = f"{img_prompt} --neg {negative}"
        movie_prompt = f"{movie_prompt} --neg {negative}"

    return img_prompt, movie_prompt


def save_story_overwrite_with_prompts(story_path: Path) -> Path:
    """
    story.json을 읽어 각 scene의 prompt_img/prompt_movie를 재작성하여 같은 파일에 덮어쓴다.
    실패 시 예외를 그대로 올리되, [PROMPTS] 로그는 남긴다.
    """
    print(f"[PROMPTS] load → {story_path}", flush=True)
    story: Dict[str, Any] = json.loads(Path(story_path).read_text(encoding="utf-8"))

    scenes = story.get("scenes", []) or []
    print(f"[PROMPTS] scenes={len(scenes)} | chars={story.get('characters', [])}", flush=True)

    changed = 0
    for sc in scenes:
        sid = sc.get("id", "?")
        img_p, mov_p = _build_scene_prompts(sc, story)
        sc["prompt_img"] = img_p
        sc["prompt_movie"] = mov_p
        changed += 1
        if changed <= 2:  # 샘플만
            print(f"[PROMPTS] {sid} img='{img_p[:100]}...'", flush=True)
            print(f"[PROMPTS] {sid} mov='{mov_p[:100]}...'", flush=True)

    story.setdefault("audit", {})
    story["audit"]["prompts_overwritten"] = True
    story["audit"]["prompts_overwritten_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    Path(story_path).write_text(json.dumps(story, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[PROMPTS] wrote → {story_path} (changed={changed})", flush=True)
    return story_path
# ==== /GPT 프롬프트 재작성 ====

# ===== END =====

# ===== /GPT 프롬프트 재작성 유틸 =====
# =========]]]]]]]]]=============================

# settings 로드 (패키지/단독 실행 모두 대응)
try:
    from app import settings as settings    # 소문자 별칭
    from app.settings import BASE_DIR
except ImportError:
    import settings as settings             # 소문자 별칭
    from settings import BASE_DIR

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

def load_json(p: os.PathLike | str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    path = Path(p)
    if path.is_dir():
        raise IsADirectoryError(f"load_json: 폴더 경로입니다: {path}")
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(p: os.PathLike | str, obj: Dict[str, Any]) -> Path:
    path = Path(p)
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path

def write_text(p: os.PathLike | str, text: str) -> Path:
    path = Path(p)
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 문자열/날짜
def sanitize_title(s: str) -> str:
    """윈도우 금지문자 제거 + 공백 정돈"""
    bad = r'\/:*?"<>|'
    out = "".join(c for c in (s or "").strip() if c not in bad)
    out = " ".join(out.split())
    return out or "untitled"

def now_kr(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    kst = timezone(timedelta(hours=9))
    return datetime.now(tz=kst).strftime(fmt)

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
from pathlib import Path
from typing import Dict, Any

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
from pathlib import Path
from typing import Dict, Any

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
# ==== PATCH END ====


# === [MERGED FROM ai.py] ===
from dataclasses import dataclass
from typing import Literal, Sequence
import re, json

# ── settings 유연 임포트 ─────────────────────────────────────────────
# 권장: ai.py의 임포트 블록 치환
try:
    from app import settings as settings
except Exception:
    try:
        import settings as settings
    except Exception:
        settings = None

# (선택) 하위 호환
S = settings


# ── .env 로드 (python-dotenv) ────────────────────────────────────────
try:
    from dotenv import load_dotenv  # pip install python-dotenv
except Exception:
    load_dotenv = None

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
    gemini_model: str = "gemini-2.5-pro"

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
                "gemini-2.5-pro",
            ),
        )

# ── 본체 ─────────────────────────────────────────────────────────────
class AI:
    def __init__(self, cfg: AIConfig | None = None):
        self.cfg = (cfg or AIConfig()).resolved()
        self._openai = None
        self.default_prefer = (self.cfg.provider or 'openai').lower()
        self._gemini_ready = False
        self._init_clients()

        self.default_prefer = os.getenv("AI_PREFER", "openai").lower()  # "openai" / "gemini"
        self.gemini_model = getattr(self.cfg, "gemini_model", None) or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
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
                    from types import SimpleNamespace
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
        import os
        from typing import Any, Dict, List

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

        model_name = self.gemini_model or "gemini-2.5-pro"
        model = ggen.GenerativeModel(model_name)

        # JSON을 더 잘 내도록 하고 싶으면 mime_type을 application/json으로 바꿔도 됨
        gen_cfg: Dict[str, Any] = {"response_mime_type": "text/plain"}
        if isinstance(kwargs.get("response_format"), dict) and kwargs["response_format"].get("type") == "json_object":
            gen_cfg["response_mime_type"] = "application/json"

        # 타입 경고 제거: 분기 내에서만 설정하고, 최종적으로 없으면 dict 사용
        generation_config_obj: Any = None
        try:
            from google.generativeai.types import GenerationConfig  # type: ignore
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
        공급자 우선순위에 따라 호출하고, 실제 호출한 쪽의 trace 라벨을 남긴다.
        prefer="gemini" 이고 allow_fallback=False 이면 Gemini만 사용.
        """

        def _t(ev: str, msg: str) -> None:
            if trace:
                try:
                    trace(ev, msg)
                except Exception:
                    pass

        order = ["openai", "gemini"] if prefer == "openai" else ["gemini", "openai"]
        last_err: BaseException | None = None

        for provider in order:
            if prefer == "gemini" and not allow_fallback and provider != "gemini":
                continue

            if provider == "openai":
                _t(
                    "openai:request",
                    f"model={getattr(self.cfg, 'openai_model', None) or os.getenv('OPENAI_MODEL', 'gpt-5-mini')}"
                )
                try:
                    # 시그니처에 맞게 포지셔널 전달: (system, prompt)
                    out = self._ask_openai(system, user, **kwargs)
                    _t("openai:success", f"len={len(out)}")
                    return out
                except Exception as e:
                    _t("openai:error", f"{type(e).__name__}: {e}")
                    last_err = e
                    if prefer == "openai" and allow_fallback:
                        continue
                    raise
            else:
                _t("gemini:request", f"model={self.gemini_model}")
                try:
                    # 시그니처에 맞게 포지셔널 전달: (system, prompt)
                    out = self._ask_gemini(system, user, **kwargs)
                    _t("gemini:success", f"len={len(out)}")
                    return out
                except Exception as e:
                    _t("gemini:error", f"{type(e).__name__}: {e}")
                    last_err = e
                    if prefer == "gemini" and allow_fallback:
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
    def generate_title_lyrics_tags(
            self,
            *,
            prompt: str,
            duration_min: int,
            title_in: str = "",
            allowed_tags: Sequence[str] = (),
            language: str = "ko",
            duration_sec: int | None = None,
            trace: Optional[Any] = None,
    ) -> Dict[str, Any]:
        # ── 길이 스펙 결정 ──
        # 우선순위: duration_sec 인자 > 프롬프트 내 '20초' 등 힌트 > duration_min*60
        seconds_hint = (
            int(duration_sec) if duration_sec else
            (self._extract_seconds_hint(prompt) or (int(duration_min) * 60 if duration_min else 60))
        )
        # 기존 분(min) 스펙(60/120/180)은 그대로 유지
        dur_spec: Dict[int, Dict[str, str]] = {
            1: dict(
                target="~60s",
                structure=(
                    "[verse] 8–10 lines → [chorus] 6–8 lines → [bridge] 4–6 lines → [chorus] 6–8 lines\n"
                    "- Chorus twice, total ~24–32 lines"
                ),
            ),
            2: dict(
                target="~120s",
                structure=(
                    "[verse] 8–10 lines → [chorus] 6–8 lines → [verse] 8–10 lines → [chorus] 6–8 lines\n"
                    "- Chorus twice, total ~32–40 lines"
                ),
            ),
            3: dict(
                target="~180s",
                structure=(
                    "[verse] 8–10 lines → [chorus] 6–8 lines → [verse] 8–10 lines → "
                    "[chorus] 6–8 lines → [bridge] 4–6 lines → [chorus] 6–8 lines → [outro] 2–4 lines\n"
                    "- Chorus three times (last chorus can repeat), total ~44–56 lines"
                ),
            ),
        }
        # seconds_hint에 따라 요약 스펙
        if seconds_hint <= 30:
            spec = dict(target="≤30s", structure="[verse] 2–3 lines → [chorus] 2–3 lines (total 4–6 lines).")
        elif seconds_hint <= 60:
            spec = dict(target="31–60s", structure="[verse] 4–6 lines → [chorus] 4–6 lines (total 8–12 lines).")
        else:
            spec = dur_spec.get(max(1, min(3, int(duration_min or 2))), dur_spec[2])

        allowed_str = ", ".join(sorted({t for t in allowed_tags})) if allowed_tags else ""

        sys_rule = (
            "You are a Korean lyricist and music director. Return ONE JSON object only:\n"
            '{"title":"...", "lyrics":"...", "tags":["...", "..."], "tags_pick":["...", "..."]}\n'
            "- `lyrics` MUST use ONLY these headers: [verse], [chorus], [bridge], [outro].\n"
            f"- Target duration: {spec['target']}. Structure guideline:\n{spec['structure']}\n"
            "- Writing style: concise, singable Korean lines (natural prosody), everyday words.\n"
            "- TAGS MUST BE ENGLISH (ACE-Step style), 4–8 items.\n"
            "- If ALLOWED_TAGS are provided, pick 4–10 items ONLY from them that best match mood/instrumentation "
            "and put them in `tags_pick`.\n"
            "- Do NOT include any extra text outside the JSON."
        )
        if allowed_str:
            sys_rule += f"\nALLOWED_TAGS: {allowed_str}\n"

        user_req = {
            "prompt": prompt,
            "duration_min": duration_min,
            "title_hint": title_in,
            "language": language
        }
        ask = (
            "Generate title, lyrics, and tags for the request below. Output JSON ONLY, no code block.\n\n"
            f"[REQUEST]\n{json.dumps(user_req, ensure_ascii=False)}"
        )

        # prefer=None을 전달하지 않고 호출 (Optional 인자 제거로 타입 경고 해소)
        out = self.ask_smart(sys_rule, ask, trace=trace)
        data = self._safe_json(out)

        # ── 안전 보정 + 형식 정리 ──
        data["title"] = self._enforce_title(data.get("title", ""), prompt)
        raw_lyrics = str(data.get("lyrics", "")).strip()

        # 1) 헤더가 같은 줄에 붙은 경우 분리
        raw_lyrics = self._fix_inline_headers(raw_lyrics)
        # 2) 허용 헤더만 정상화
        raw_lyrics = self._normalize_sections(raw_lyrics)
        # 3) 파싱 → 길이 기반 섹션/줄수 강제 컷
        sections = self._parse_sections(raw_lyrics)
        sections = self._enforce_duration_structure(sections, seconds_hint)
        data["lyrics"] = self._format_sections(sections)

        # 태그 정리
        data["tags"] = self._normalize_tags(data.get("tags"))
        picks_raw = self._normalize_tags(data.get("tags_pick"))
        if allowed_tags:
            allowed_set = {t.lower() for t in allowed_tags}
            picks_raw = [t for t in picks_raw if t.lower() in allowed_set]
        data["tags_pick"] = list(dict.fromkeys(picks_raw))[:12]

        return data

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
                    from openai import OpenAI
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
                    model = ggen.GenerativeModel("gemini-1.5-flash")
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


# === [MERGED FROM progress.py] ===
# app/progress.py
# -*- coding: utf-8 -*-
from pathlib import Path
import io, traceback
import os
from typing import Callable, Optional, Any, Dict, Union
from os import PathLike
from PyQt5 import QtWidgets, QtCore

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

# (여기에 ProgressLogDialog 클래스가 먼저 옴)

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



def run_job_with_progress(
    owner: QtWidgets.QWidget,
    title: str,
    job: Callable[[Callable[[Dict[str, Any]], None]], Any],
    *,
    tail_file: TailT = None,
    on_done: Optional[Callable[[bool, Any, Optional[BaseException]], None]] = None,
) -> None:
    """동기 실행(간단 작업용)"""
    on_progress, finalize, _dlg = _mk_progress(owner, title, tail_file=tail_file)
    ok = True
    payload: Any = None
    err: Optional[BaseException] = None
    try:
        payload = job(on_progress)
    except BaseException as exc:
        ok = False
        err = exc
    finally:
        finalize(ok, payload, err)
        if on_done:
            try:
                on_done(ok, payload, err)
            except Exception:
                pass

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

# progress.py (하단에 추가 또는 기존 함수 교체)

# progress.py  ── 전체 교체
def run_job_with_progress_async(
    owner: QtWidgets.QWidget,
    title: str,
    job,  # Callable[[Callable[[dict], None]], Any]
    *,
    tail_file=None,
    on_done=None,  # Callable[[bool, Any, Optional[BaseException]], None]
) -> None:
    """
    비동기 작업 실행 + 진행창 + (옵션) 로그 테일링(tail_file).
    - owner: 부모 위젯
    - title: 진행창 제목
    - job(on_progress: Callable[[dict], None]) -> Any: 백그라운드에서 실행할 함수
    - tail_file: 실시간 테일링할 로그 파일 경로(선택)  → _mk_progress 내부에서 처리
    - on_done(ok: bool, payload: Any, err: Optional[Exception]): 완료 콜백
    """
    from PyQt5 import QtCore

    # --- 네가 제공한 _mk_progress는 (on_progress, finalize, dlg) 순서로 3개 반환 ---
    try:
        on_progress_ui, finalize_ui, dlg = _mk_progress(owner, title, tail_file=tail_file)  # type: ignore
    except Exception as e:
        # _mk_progress가 없거나 실패하면 바로 알리고 종료(불필요한 대체 구현 없이 명확히 처리)
        try:
            QtWidgets.QMessageBox.critical(owner, "진행창 초기화 실패", str(e))  # type: ignore
        except Exception:
            pass
        if callable(on_done):
            try:
                on_done(False, None, e)
            except Exception:
                pass
        return

    # 초기 로그 한 줄(네 _mk_progress.on_progress가 QTimer로 UI 스레드에 안전하게 반영)
    try:
        on_progress_ui({"stage": "ui", "msg": "[ui] 작업 시작 준비"})
    except Exception:
        pass

    class _Worker(QtCore.QObject):
        progress = QtCore.pyqtSignal(dict)
        finished = QtCore.pyqtSignal(object, object)  # (payload, err)

        @QtCore.pyqtSlot()
        def run(self):
            payload = None
            err = None
            try:
                def on_progress(info: dict):
                    # 워커 스레드 → 메인 스레드로 신호만 보냄(실제 UI 반영은 슬롯에서)
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
        # 메인 스레드: 네 _mk_progress가 넘긴 on_progress_ui 호출(내부가 QTimer로 UI 반영)
        try:
            on_progress_ui(info)
        except Exception:
            pass

    def _on_finished(payload, err):
        ok = (err is None)

        # finalize 호출로 진행창 마무리
        try:
            finalize_ui(ok, payload, err)
        except Exception:
            pass

        # 완료 콜백
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
            jobs = getattr(owner, "_progress_jobs", [])
            if th in jobs:
                jobs.remove(th)
            setattr(owner, "_progress_jobs", jobs)
        except Exception:
            pass

    obj.progress.connect(_on_progress)
    obj.finished.connect(_on_finished)
    th.started.connect(obj.run)

    # GC 방지: 소유자에 스레드 참조를 보관
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

    # 스레드 시작(실패 시 finalize + on_done(False, ...))
    try:
        th.start()
    except Exception as start_exc:
        try:
            on_progress_ui({"stage": "error", "msg": f"[error] thread start failed: {start_exc}"})
        except Exception:
            pass
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










# ───────── 백업 ProgressLogDialog (없을 때만) ─────────
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



# === [MERGED FROM tag_norm.py] ===
# app/tag_norm.py
from typing import Iterable, List

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
