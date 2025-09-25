# -*- coding: utf-8 -*-
# OpenAI GPT로 제목/가사/태그 생성
import re
from typing import List

try:
    from ai import AI
except Exception:
    from app.ai import AI  # type: ignore

# 유연한 임포트 (패키지/단일 파일 실행 모두 지원)
from app.utils import ensure_project_dir, save_json, write_text, now_kr, today_str

# ====== DEBUG LOGGING ======
from pathlib import Path as _Path
try:
    from app.settings import BASE_DIR as _BASE
except Exception:
    from settings import BASE_DIR as _BASE  # type: ignore

_DEBUG_DIR = _Path(_BASE) / "_debug"
_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
_DEBUG_LOG = _DEBUG_DIR / "lyrics_debug.log"

def _dlog(msg: str):
    """콘솔 + 파일에 동시에 남김"""
    line = f"[LYRICS_DEBUG] {msg}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as fp:
            fp.write(line + "\n")
    except Exception:
        pass
# ===========================


OPENAI_MODEL = "gpt-5-mini"

# ───────── 기본 보컬 태그(보강용) ─────────
BASIC_VOCAL_TAGS = [
    "clean vocals",
    "natural articulation",
    "warm emotional tone",
    "studio reverb light",
    "clear diction",
    "breath control",
    "balanced mixing",
]

# ───────── 유틸 ─────────
def _dedup_keep_order(items: List[str]) -> List[str]:
    seen, out = set(), []
    for it in items:
        t = (it or "").strip()
        if not t:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _normalize_tags(tags, ensure_basic: bool = False) -> list[str]:
    """
    - 문자열이면 쉼표/슬래시/줄바꿈으로 분할
    - 영문 토큰만 사용
    - 중복 제거
    - ensure_basic=True면 BASIC_VOCAL_TAGS 전부를 무조건 포함
    - 최소 5개 미만이면 BASIC_VOCAL_TAGS로 보강
    - 최종 최대 12개(여유)로 컷하되, BASIC_VOCAL_TAGS는 먼저 고정 포함
    """
    if isinstance(tags, str):
        parts = [p.strip() for p in re.split(r"[,\n/;]+", tags) if p.strip()]
    elif isinstance(tags, list):
        parts = [str(p).strip() for p in tags if str(p).strip()]
    else:
        parts = []

    # 영문만
    parts = [p for p in parts if re.search(r"[A-Za-z]", p)]
    parts = _dedup_keep_order(parts)

    if ensure_basic:
        # BASIC_VOCAL_TAGS를 항상 앞에 고정 포함
        parts = _dedup_keep_order(BASIC_VOCAL_TAGS + parts)

    # 5개 미만이면 보강
    if len(parts) < 5:
        parts = _dedup_keep_order(parts + BASIC_VOCAL_TAGS)

    # 최종 컷(여유롭게 12개까지, 필요시 8로 줄여도 됨)
    return parts[:12]


# 섹션 라벨 보강/정리 (한국어 표기 → [verse]/[chorus]/[bridge]/[outro])
_SECTION_PATTERNS = [
    (re.compile(r"^\s*\(?\s*(\d+)\s*절\s*\)?\s*[:：)]*\s*$"), lambda m: f"[verse {m.group(1)}]"),
    (re.compile(r"^\s*\(?\s*후\s*렴\s*\)?\s*[:：)]*\s*$"),     lambda m: "[chorus]"),
    (re.compile(r"^\s*\(?\s*브\s*릿\s*지\s*\)?\s*[:：)]*\s*$"), lambda m: "[bridge]"),
    (re.compile(r"^\s*\(?\s*아\s*웃\s*트\s*로\s*\)?\s*[:：)]*\s*$"), lambda m: "[outro]"),
]

# ─────────────────────────────────────────────────────────────
SECTION_ONLY = re.compile(
    r"^\s*\[(?:intro|verse|pre[- ]?chorus|chorus|bridge|outro|hook|coda|break|tag|interlude)(?:\s+\d+)?\]\s*$",
    re.IGNORECASE,
)
PAREN_ONLY  = re.compile(r"^\s*\(.+?\)\s*$")  # 줄 전체가 괄호 메모

# 촬영/메모/지시문 + '사진이름 -> 컨셉' 류 라인들
NOTE_LINE = re.compile(
    r"^\s*\[?\s*(?:촬영|촹영)?\s*컨셉\s*메모\]?\s*$|"      # [촬영 컨셉 메모], (오타 '촹영' 포함)
    r"^\s*\[?\s*촬영\s*메모\]?\s*$|"                        # [촬영 메모]
    r"^\s*(?:메모|노트|note)\s*[:：]?\s*$|"                 # 메모/노트 라벨성 문구
    r"^\s*(?:scene|camera|shot)\b.*$|"                      # scene/camera/shot로 시작하는 지시문
    r"^\s*(?:bpm|key|tempo)\s*[:：].*$|"                    # bpm:/key:/tempo:
    r"^\s*사진\s*이름\s*->\s*.*$",                          # 사진이름 -> 컨셉
    re.IGNORECASE,
)

# lyrics_gen.py

def _ensure_intro_in_lyrics(text: str, *, total_seconds: int) -> str:
    """
    총 길이가 60초 이상인데 [intro] 블록이 없으면 맨 앞에 삽입.
    ※ 기존 '(인트로 분위기/도입부 묘사)' 같은 괄호 문구는 더이상 넣지 않음.
    """
    try:
        lines = (text or "").splitlines()
        has_intro = any(ln.strip().lower().startswith("[intro]") for ln in lines)
        if total_seconds >= 60 and not has_intro:
            add = "[intro]\n\n"   # 괄호 문구 제거
            return add + (text or "")
        return text or ""
    except Exception:
        return text or ""





def normalize_sections(text: str) -> str:
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
        converted = None
        for pat, repl in _SECTION_PATTERNS:
            m = pat.match(stripped)
            if m:
                converted = repl(m)
                break
        out_lines.append(converted if converted else ln)
        has_tag = has_tag or bool(converted)
    if not has_tag and out_lines:
        out_lines.insert(0, "[verse]")
    return "\n".join(out_lines)



# ───────── 공개 API ─────────

# 맨 위에 이미 import re 되어 있으면 생략
import re
def generate_title_lyrics_tags(
    *,
    prompt: str,
    duration_min: int,
    title_in: str = "",
    allowed_tags=None,
    duration_sec: int | None = None,
    trace=None,
    prefer: str | None = None,          # "openai" | "gemini"
    allow_fallback: bool | None = None, # True/False
) -> dict:
    import json, re
    from typing import Any, Dict, List

    def t(ev: str, msg: str):
        if trace:
            try:
                trace(ev, msg)
            except Exception:
                pass

    # ---- 입력 정리
    prompt = str(prompt or "")
    title_in = str(title_in or "")
    allowed_tags = list(allowed_tags or [])

    # ---- 목표 초 결정(초 우선)
    sec = None
    if duration_sec is not None:
        try:
            sec = int(duration_sec)
        except ValueError:
            sec = None
    if sec is None and prompt:
        m = re.search(r"(\d{1,3})\s*(초|s|sec|secs|second|seconds)\b", prompt, flags=re.I)
        if m:
            try:
                sec = int(m.group(1))
            except ValueError:
                sec = None
    if sec is None or sec <= 0:
        try:
            duration_min = int(duration_min)
        except ValueError:
            duration_min = 2
        duration_min = max(1, min(3, duration_min))
        sec = duration_min * 60

    # ---- 프롬프트(시스템/유저)
    dmin_for_log = 0 if sec < 60 else max(1, min(3, sec // 60))
    system = (
        "You are a Korean lyricist and music director. Return ONE JSON object only:\n"
        '{"title":"...", "lyrics_ko":"...", "tags":["...", "..."], "tags_pick":["...", "..."]}\n'
        "- lyrics_ko MUST use ONLY these headers: [verse], [chorus], [bridge], [outro].\n"
    )
    # 길이 가이드
    if sec < 60:
        system += "- Target duration: ~{}s. Structure: [verse] 3–5 lines → [chorus] 3–5 lines (optional)\n".format(sec)
    else:
        system += "- Target duration: ~{}s. Use standard verse/chorus/bridge/outro structure.\n".format(sec)

    if allowed_tags:
        system += "ALLOWED_TAGS: " + ", ".join(sorted(set(allowed_tags))) + "\n"

    user = (
        f"[PROMPT]: {prompt or '(empty)'}\n"
        f"[TARGET_SEC]: {sec}\n"
        f"[DURATION_MIN_HINT]: {dmin_for_log}\n"
        f"[TITLE_HINT]: {title_in or '(none)'}"
    )

    # ---- 모델 호출
    try:
        from ai import AI
    except ImportError:
        from app.ai import AI  # type: ignore
    ai = AI()
    prefer0 = (prefer or getattr(ai, "default_prefer", "openai")).lower()
    allow0 = allow_fallback if allow_fallback is not None else (prefer0 == "openai")

    t("ai:prepare", f"prefer={prefer0}, allow_fallback={allow0}, target_sec={sec}")
    t("ai:call", "모델 호출 시작")

    raw = ai.ask_smart(system, user, prefer=prefer0, allow_fallback=allow0, trace=trace)

    if trace:
        try:
            prev = str(raw)[:80]
            t("ai:recv", f"응답 수신 ok, len={len(str(raw))}, preview={prev}")
        except Exception:
            pass

    text = str(raw or "").strip()
    if not text:
        t("ai:error", "빈 응답")
        raise RuntimeError("빈 응답입니다.")

    # ---- JSON 파싱
    t("parse:begin", f"text_len={len(text)}")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        data_obj = json.loads(text)  # 순수 JSON
    else:
        data_obj = json.loads(text[start:end + 1])
    t("parse:end", "JSON 파싱 완료")

    # ---- 필드 보정
    title = str(data_obj.get("title", "")).strip() or (title_in or "untitled")
    lyrics = str(data_obj.get("lyrics_ko", "") or data_obj.get("lyrics", "")).strip()

    # 섹션 라벨 정리/보강
    from lyrics_gen import normalize_sections, _ensure_intro_in_lyrics  # 자기 파일 내
    lyrics = normalize_sections(lyrics)
    if sec >= 60:
        lyrics = _ensure_intro_in_lyrics(lyrics, total_seconds=int(sec))

    # === [추가] 불필요 문구 클린업(섹션 라벨은 유지) ===
    SECTION_ONLY = re.compile(
        r"^\s*\[(?:intro|verse|pre[- ]?chorus|chorus|bridge|outro|hook|coda|break|tag|interlude)(?:\s+\d+)?\]\s*$",
        re.IGNORECASE,
    )
    PAREN_ONLY  = re.compile(r"^\s*\(.+?\)\s*$")  # 줄 전체가 괄호 메모
    NOTE_LINE = re.compile(
        r"^\s*\[?\s*(?:촬영|촹영)?\s*컨셉\s*메모\]?\s*$|"      # [촬영 컨셉 메모]
        r"^\s*\[?\s*촬영\s*메모\]?\s*$|"                        # [촬영 메모]
        r"^\s*(?:메모|노트|note)\s*[:：]?\s*$|"                 # 메모/노트
        r"^\s*(?:scene|camera|shot)\b.*$|"                      # scene/camera/shot...
        r"^\s*(?:bpm|key|tempo)\s*[:：].*$|"                    # bpm:/key:/tempo:
        r"^\s*사진\s*이름\s*->\s*.*$",                          # 사진이름 -> 컨셉
        re.IGNORECASE,
    )

    cleaned: List[str] = []
    for ln in lyrics.splitlines():
        s = ln.strip()
        if not s:
            continue
        if SECTION_ONLY.match(s):
            cleaned.append(s)
            continue
        if PAREN_ONLY.match(s):
            continue
        if NOTE_LINE.match(s):
            continue
        if "->" in s and ("사진" in s or "이미지" in s):
            continue
        s = re.sub(r"\s+", " ", s)  # 공백 축약
        cleaned.append(s)

    # 중복 라인 제거(선착순)
    seen, uniq = set(), []
    for s in cleaned:
        if s not in seen:
            uniq.append(s); seen.add(s)

    lyrics = "\n".join(uniq).strip()
    if not lyrics:
        raise RuntimeError("메모/태그 제거 후 남은 가사가 없습니다.")

    # ---- 태그 정규화(기존 로직 유지)
    def _norm_tags(tags_in: Any, ensure_basic: bool = False) -> List[str]:
        if isinstance(tags_in, str):
            parts = [p.strip() for p in re.split(r"[,\n/;]+", tags_in) if p.strip()]
        elif isinstance(tags_in, list):
            parts = [str(p).strip() for p in tags_in if str(p).strip()]
        else:
            parts = []
        parts = [p for p in parts if re.search(r"[A-Za-z]", p)]
        # 최소 5개 보강
        base = [
            "clean vocals","natural articulation","warm emotional tone",
            "studio reverb light","clear diction","breath control","balanced mixing",
        ]
        if ensure_basic:
            parts = list(dict.fromkeys(base + parts))
        if len(parts) < 5:
            parts = list(dict.fromkeys(parts + base))
        return parts[:12]

    tags = _norm_tags(data_obj.get("tags"))
    picks_raw = _norm_tags(data_obj.get("tags_pick"))

    if allowed_tags:
        allow_set = {t.lower() for t in allowed_tags}
        picks = [t for t in picks_raw if t.lower() in allow_set][:10]
    else:
        picks = picks_raw[:10]

    return {"title": title, "lyrics": lyrics, "tags": tags, "tags_pick": picks}









# lyrics_gen.py (또는 create_project_files가 있는 파일)


def create_project_files(title: str, lyrics: str, prompt: str) -> str:
    project_dir = ensure_project_dir(title)
    date = today_str()

    lyrics_norm = normalize_sections(lyrics)  # ← 섹션 유지

    txt_path = project_dir / f"{title}({date}).txt"
    # 🔧 "[가사]" 고정 헤더 제거: 섹션 헤더가 맨 위로 오게
    write_text(
        txt_path,
        f"노래제목: {title}\n생성일: {date}\n\n{lyrics_norm}\n\n[촬영 컨셉 메모]\n사진이름 -> 컨셉"
    )

    meta = {
        "title": title,
        "lyrics": lyrics_norm,
        "prompt": prompt,
        "created_at": now_kr(),
        "time": 60,
        "auto_tags": True,
        "ace_tags": [],
        "tags_in_use": [],
        "manual_tags": [],
        "paths": {
            "project_dir": str(project_dir),
            "lyrics_txt": str(txt_path),
            "txt": str(txt_path),
            "audio_out": str(project_dir / "vocal.mp3"),
            "video_dir": str(project_dir / "clips"),
        },
        "i2v_plan": {
            "input_fps": 60, "target_fps": 60, "base_chunk": 300, "overlap": 12,
            "upscale": {"enabled": True, "model": "RealESRGAN_x4plus"}
        },
        "concepts": []
    }
    save_json(project_dir / "project.json", meta)
    (project_dir / "clips").mkdir(exist_ok=True)
    return str(project_dir)



