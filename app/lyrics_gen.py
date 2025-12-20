# -*- coding: utf-8 -*-
from app.utils import AI
import json
import re
from typing import List
from app.utils import ensure_project_dir, save_json, write_text, now_kr, today_str
from pathlib import Path as _Path
from app.settings import BASE_DIR as _BASE

from app import settings as settings_mod
from pathlib import Path



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
    r"^\s*\[(?:intro|verse|pre[- ]?chorus|chorus|bridge|outro|hook|coda|break|tag|interlude)(?:\s+\d+)?]\s*$",
    re.IGNORECASE,
)
PAREN_ONLY  = re.compile(r"^\s*\(.+?\)\s*$")  # 줄 전체가 괄호 메모

# 촬영/메모/지시문 + '사진이름 -> 컨셉' 류 라인들
NOTE_LINE = re.compile(
    r"^\s*\[?\s*(?:촬영|촹영)?\s*컨셉\s*메모]?\s*$|"      # [촬영 컨셉 메모], (오타 '촹영' 포함)
    r"^\s*\[?\s*촬영\s*메모]?\s*$|"                        # [촬영 메모]
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





def normalize_sections(lyrics: str) -> str:
    """
    섹션/가사 정규화.
    - 섹션 헤더([verse], [bridge], [chorus] 등)는 그대로 보존
    - 가사 라인은 언어 태그를 보수적으로 고정:
        * 한글(가-힣)이 하나라도 있으면 [ko] 강제
        * 한글이 전혀 없으면 [en] 기본
      -> [fr], [pl], [de] 등 비의도 언어 태그를 제거하여 LLS/Comfy 파이프라인에서
         한국어 라인이 무시되는 문제를 방지
    - 공백/빈 줄은 유지하되 불필요한 트레일링 스페이스는 제거
    """
    import re

    if not isinstance(lyrics, str):
        return ""

    # 허용 섹션 태그(필요 시 확장 가능)
    section_pat = re.compile(r"^\s*\[(verse|bridge|chorus|hook|intro|outro|pre-chorus|post-chorus)]\s*$",
                             re.IGNORECASE)
    # 기존 언어 태그 패턴
    lang_tag_pat = re.compile(r"^\s*\[([a-z]{2})]\s*", re.IGNORECASE)
    # 한글 존재 여부
    has_hangul = re.compile(r"[\uac00-\ud7a3]")

    # ko/en만 허용
    allowed_langs = {"ko", "en"}

    lines = lyrics.splitlines()
    out = []

    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line.strip():
            out.append(line)
            continue

        # 섹션 헤더는 그대로
        if section_pat.match(line):
            # 섹션명은 소문자로 통일
            sec = section_pat.match(line).group(1).lower()
            out.append(f"[{sec}]")
            continue

        # 선행 언어 태그 제거 후 재적용
        m = lang_tag_pat.match(line)
        if m:
            lang = m.group(1).lower()
            content = line[m.end():].lstrip()
        else:
            lang = ""
            content = line.lstrip()

        # 한글 포함 여부로 우선 판정
        if has_hangul.search(content):
            lang = "ko"
        else:
            # 한글이 없으면 기본 en (혼합 토큰으로 인한 오탐 방지)
            if lang not in allowed_langs:
                lang = "en"

        # 이미 ko/en 태그라면 유지, 그렇지 않으면 ko/en로 교정
        if lang not in allowed_langs:
            lang = "ko" if has_hangul.search(content) else "en"

        # 빈 컨텐츠는 그대로
        if not content:
            out.append(content)
            continue

        out.append(f"[{lang}]{content}")

    return "\n".join(out)




# ───────── 공개 API ─────────

# lyrics_gen.py 파일에서 이 함수를 찾아 아래 내용으로 전체를 교체하세요.

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
    emit("ai:prepare", f"prefer={prefer_opt}, allow_fallback={allow_opt}, sec={sec_val}, lines={min_lines}-{max_lines}")

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

        base_dir = Path(getattr(settings_mod, "BASE_DIR", "."))
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





