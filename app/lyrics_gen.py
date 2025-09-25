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
# 맨 위에 이미 import re 되어 있으면 생략
import re

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
    """
    가사 생성:
      - 1줄≈5초 규칙으로 '본문 줄수(섹션 헤더 제외)' 범위만 제시
      - 허용 섹션은 [verse], [bridge]만 (intro/outro/chorus/pre-chorus 금지)
      - 모델이 실수로 금지 섹션을 내보내면 모두 제거하거나 [verse]로 치환
    출력 JSON: {"title": "...", "lyrics_ko":"...", "tags":["...", "..."], "tags_pick":["...", "..."]}
    """
    import json
    from typing import List
    from ai import AI  # 기존 구조 유지

    def t(ev: str, msg: str):
        try:
            print(f"[lyrics_gen:{ev}] {msg}")
        except Exception:
            pass

    allowed_tags = allowed_tags or []

    # ---- 목표 초 계산(초 우선) ----
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

    # ---- 1줄≈5초: '본문 줄수' 범위 가이드(헤더 제외) ----
    base_lines = max(1, round(sec / 5))
    if sec <= 35:
        min_lines, max_lines = 6, 8
    elif sec <= 75:
        min_lines, max_lines = 10, 12
    elif sec <= 150:
        min_lines, max_lines = max(8, base_lines - 2), base_lines + 2
    else:
        min_lines, max_lines = max(10, base_lines - 2), base_lines + 2

    # ---- 시스템/유저 프롬프트 (intro/outro/chorus 금지) ----
    system = (
        "You are a Korean lyricist and music director. Return ONE JSON object only:\n"
        '{"title":"...", "lyrics_ko":"...", "tags":["...", "..."], "tags_pick":["...", "..."]}\n'
        "- Allowed headers: [verse], [bridge] (lowercase)\n"
        f"- Body line budget (EXCLUDING headers): {min_lines}–{max_lines} total lines.\n"
        "- IMPORTANT:\n"
        "  1) Do NOT use [intro], [outro], [chorus], or any pre-chorus sections.\n"
        "  2) Keep lyric lines only under [verse]/[bridge].\n"
        "  3) Do NOT add production notes, camera/stage directions, or metadata.\n"
        f"- Target duration guide: ~{sec}s; assume ~5s per line.\n"
    )
    if allowed_tags:
        system += "ALLOWED_TAGS: " + ", ".join(sorted(set(allowed_tags))) + "\n"

    user = (
        "[TASK]\n"
        "- Write natural Korean lyrics with the above constraints.\n"
        "- Title may be short and poetic.\n\n"
        "[PROMPT]\n" + (prompt or "")
    )

    # ---- 모델 호출 (스마트 라우팅) ----
    prefer0 = "openai" if prefer is None else str(prefer)
    allow0 = (allow_fallback if allow_fallback is not None else (prefer0 == "openai"))
    t("ai:prepare", f"prefer={prefer0}, allow_fallback={allow0}, sec={sec}, lines={min_lines}-{max_lines}")

    ai = AI()
    raw = ai.ask_smart(system, user, prefer=prefer0, allow_fallback=allow0, trace=trace)
    text = str(raw or "").strip()
    if not text:
        raise RuntimeError("빈 응답입니다.")

    # ---- JSON 파싱(관대한 추출) ----
    t("parse:begin", f"text_len={len(text)}")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        data_obj = json.loads(text)
    else:
        data_obj = json.loads(text[start:end + 1])
    t("parse:end", "ok")

    # ---- 필드 보정 ----
    title = str(data_obj.get("title", "")).strip() or (title_in or "untitled")
    lyrics = str(data_obj.get("lyrics_ko", "") or data_obj.get("lyrics", "")).strip()

    # (선택) 섹션 라벨 정규화가 파일에 이미 있다면 활용 (없어도 무시)
    try:
        from lyrics_gen import normalize_sections as _norm_sections  # type: ignore
        lyrics = _norm_sections(lyrics)
    except (ImportError, AttributeError):
        pass

    # ---- 금지 섹션 제거/치환 ----
    # 1) 헤더만 있는 라인 패턴
    header_only = re.compile(
        r"^\s*\[(?:intro|outro|pre[- ]?chorus|chorus|hook|coda|break|tag|interlude|verse|bridge)(?:\s+\d+)?\]\s*$",
        re.IGNORECASE,
    )
    # 2) 금지 헤더(섹션 전체 제거) 블록 추출
    forbidden_heads = re.compile(r"^\s*\[(?:intro|outro|pre[- ]?chorus|chorus)\b[^\]]*\]\s*$", re.IGNORECASE)

    lines = lyrics.splitlines()
    cleaned_lines: list[str] = []
    skip_mode = False
    for ln in lines:
        s = ln.strip()
        if forbidden_heads.match(s):
            skip_mode = True
            continue
        if header_only.match(s):
            # 허용 헤더만 통과(verse/bridge). 금지 헤더는 위에서 잡힘.
            head = s.lower()
            if "[verse]" in head or "[bridge]" in head:
                cleaned_lines.append(s)
            skip_mode = False
            continue
        if skip_mode:
            # 금지 섹션의 본문은 통째로 제거
            continue
        cleaned_lines.append(ln)
    lyrics = "\n".join(cleaned_lines).strip()

    # 혹시 모델이 헤더 없이 코러스 단어를 본문에 넣는 경우는 그대로 두되,
    # "[pre-chorus]"나 "[chorus]" 형식의 헤더는 위에서 모두 제거됨.
    # 일부 모델이 "[Chorus 1]" 등 변형을 쓸 수 있어 추가 방어:
    chorus_head_pat = re.compile(r"^\s*\[(?:pre[- ]?chorus|chorus)(?:\s+\d+)?\]\s*$", re.IGNORECASE)
    lyrics = "\n".join("[verse]" if chorus_head_pat.match(x.strip()) else x for x in lyrics.splitlines())

    # ---- 기본 노이즈 정리 ----
    section_only = re.compile(
        r"^\s*\[(?:verse|bridge)(?:\s+\d+)?\]\s*$",
        re.IGNORECASE,
    )
    paren_only = re.compile(r"^\s*\(.+?\)\s*$")
    cleaned: List[str] = []
    for ln in lyrics.splitlines():
        s = ln.strip()
        if not s:
            continue
        if section_only.match(s):
            cleaned.append(s)
            continue
        if paren_only.match(s):
            continue
        cleaned.append(s)

    # 중복 제거
    uniq: List[str] = []
    seen = set()
    for s in cleaned:
        if s not in seen:
            uniq.append(s)
            seen.add(s)

    lyrics = "\n".join(uniq).strip()
    if not lyrics:
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

def _ensure_intro_outro_headers(text: str) -> str:
    """가사에 [intro]와 [outro] 헤더를 반드시 포함시키되, 그 아래에는 가사 줄을 두지 않는다."""
    lines = (text or "").splitlines()
    has_intro = any(ln.strip().lower().startswith("[intro]") for ln in lines)
    has_outro = any(ln.strip().lower().startswith("[outro]") for ln in lines)

    out_lines: list[str] = []
    if not has_intro:
        out_lines.append("[intro]")
        out_lines.append("")  # 무가사 영역
    out_lines.extend(lines)
    if not has_outro:
        if out_lines and out_lines[-1].strip():
            out_lines.append("")
        out_lines.append("[outro]")

    # 혹시 intro/outro 아래에 가사 줄이 있었다면 제거
    cleaned: list[str] = []
    current_header: str | None = None
    for ln in out_lines:
        s = (ln or "").strip()
        if s.startswith("[") and s.endswith("]"):
            current_header = s.lower()
            cleaned.append(s)
            continue
        if current_header in ("[intro]", "[outro]"):
            # 무가사: 비워둔다
            continue
        cleaned.append(ln)
    return "\n".join(cleaned)





