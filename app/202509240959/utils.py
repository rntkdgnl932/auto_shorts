# -*- coding: utf-8 -*-
"""
공용 유틸 함수 모음 (외부 의존성 최소화)
- 프로젝트 폴더 규칙: BASE_DIR / "maked_title" / <title>
- 결과 복사 규칙: settings.FINAL_OUT 경로의 [title] 치환
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Iterable, List, Tuple
from pathlib import Path
from datetime import datetime, timezone, timedelta
import importlib.util
import shutil
import json
import os
import re
import time

# ─────────────────────────────────────────────────────────────────────────────

# ===== GPT 프롬프트 재작성 유틸 (utils.py 내부에 추가) =====

# ---- 매핑 테이블 ----
# ===== GPT 프롬프트 재작성(Story Overwrite) =====
# ==== GPT 프롬프트 재작성( story.json 덮어쓰기 ) ====
import json, re, time
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
        ctags = _extract_char_tags(ref["id"], char_styles.get(ref["id"], ""))
        char_tags.extend(ctags)

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
        img_prompt  = f"{img_prompt} --neg {negative}"
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
from typing import Dict, Any, List

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
from typing import Dict, Any, List

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
