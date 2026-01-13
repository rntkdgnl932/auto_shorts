# -*- coding: utf-8 -*-
"""
audio_sync.py
- 오디오 길이와 (있으면) project.json의 가사를 이용해 타임라인 생성
- 결과는 segments.json / scene.json을 '항상' 오디오 파일 폴더에 저장(save=True일 때)
"""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, cast, Union, Iterable, Pattern
import random
from pydub import AudioSegment  # type: ignore
from app.settings import (
    BASE_DIR, COMFY_HOST, DEFAULT_HOST_CANDIDATES, _DEFAULT_ACE_WAIT_TIMEOUT_SEC, _DEFAULT_ACE_POLL_INTERVAL_SEC, _DEFAULT_WEIGHTS,
    ACE_STEP_PROMPT_JSON, FFMPEG_EXE, FINAL_OUT, SECTION_HEADER_RE
)
from app import settings as _s
from app.utils import load_json, save_json, sanitize_title, effective_title, save_to_user_library, audio_duration_sec, _dlog, _submit_and_wait, get_duration
from librosa.onset import onset_strength as lb_onset_strength, onset_detect as lb_onset_detect
from librosa.effects import hpss as lb_hpss
import tempfile
from mutagen import File as MutagenFile
import requests
import uuid
import time
import numpy as np
import shutil # Added for _ensure_vocal_wav
import sys
import subprocess
import difflib
import re
# ───────────────────────── utils 안전 import ─────────────────────────


# (선택) 런타임 설정 — 필요시 사용
import app.settings as settings
import soundfile as sf
import librosa
import noisereduce as nr
import pyloudnorm as pyln
from pathlib import Path
import json
import math
import os
from app import settings as app_settings_local
import app.settings as root_settings_local


# from app.video_build import build_image_json as _build_image_json, build_movie_json as _build_movie_json
# from app.video_build import normalize_to_v11, validate_story_v11_dict

# ───────────────────────── librosa(선택) ─────────────────────────
import librosa as _librosa_mod  # 실제 모듈
librosa = cast(Any, _librosa_mod)  # 정적 분석 경고 억제


# ───────────────────────── 오디오 유틸 ─────────────────────────
# real use
def _librosa_trim_edges(audio_path: Path) -> Tuple[float, float, float]:
    """앞뒤 무음 트리밍 추정: (offset, effective, full) 초 단위. 실패 시 (0, full, full)."""
    full = audio_duration_sec(audio_path)
    if librosa is None or full <= 0:
        return 0.0, full, full
    try:
        y, sr = librosa.load(str(audio_path), mono=True)  # type: ignore[attr-defined]
        yt, idx = librosa.effects.trim(y, top_db=25)      # type: ignore[attr-defined]
        off = idx[0] / sr
        eff = len(yt) / sr
        return float(off), float(eff), float(full)
    except (ValueError, RuntimeError, OSError, AttributeError):
        return 0.0, full, full

# real use
def _safe_load_audio_for_onsets(p: Path):
    if librosa is None:
        return None, None
    try:
        y, sr = librosa.load(str(p), mono=True)  # type: ignore[attr-defined]
        return y, sr
    except (FileNotFoundError, OSError, ValueError, RuntimeError):
        return None, None

# real use
def _detect_onsets_sec(p: Path) -> List[float]:
    """오디오에서 온셋(초) 리스트 검출. librosa 없으면 빈 리스트."""
    y, sr = _safe_load_audio_for_onsets(p)
    if y is None or sr is None:
        return []
    try:
        on = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)  # type: ignore[attr-defined]
        on = sorted(float(t) for t in on if t >= 0.0)
        dedup: List[float] = []
        for t in on:
            if not dedup or (t - dedup[-1]) >= 0.01:  # 10ms 이내 중복 제거
                dedup.append(round(t, 3))
        return dedup
    except (ValueError, RuntimeError, AttributeError):
        return []


# ───────────────────────── 가사 → 블록 파서 ─────────────────────────
# SECTION_HEADER_RE = re.compile(r"^\s*\[(?P<name>[^]]+)]\s*$", re.IGNORECASE)

# real use
def parse_lyrics_blocks(lyrics_text: str) -> List[Dict[str, Any]]:
    """[verse]/[chorus]/[bridge]/... 헤더로 블록 분리"""
    lines = (lyrics_text or "").splitlines()
    blocks: List[Dict[str, Any]] = []
    cur_name = "unknown"
    cur_label = "Unknown"
    cur_lines: List[str] = []

    def _flush() -> None:
        nonlocal blocks, cur_name, cur_label, cur_lines
        if cur_lines:
            pure = [line.strip() for line in cur_lines if line.strip()]
            if pure:
                blocks.append({
                    "section": cur_name.lower(),
                    "label": cur_label,
                    "lines": pure,
                })
            cur_lines = []

    for ln in lines:
        m = SECTION_HEADER_RE.match(ln)
        if m:
            _flush()
            raw = m.group("name").strip()
            cur_label = raw
            cur_name = raw.split(":")[0].strip().lower()
            continue
        cur_lines.append(ln)
    _flush()
    return blocks


# ───────────────────────── 길이 배분 ─────────────────────────
# _DEFAULT_WEIGHTS = {
#     "verse": 1.0,
#     "chorus": 1.2,
#     "bridge": 1.0,
#     "pre-chorus": 1.0,
#     "intro": 0.6,
#     "outro": 0.8,
#     "unknown": 1.0,
# }
# real use
def _section_weight(section: str) -> float:
    s = section.lower().strip()
    if s.startswith("chorus") or "hook" in s:
        return _DEFAULT_WEIGHTS["chorus"]
    if s.startswith("verse"):
        return _DEFAULT_WEIGHTS["verse"]
    if "bridge" in s:
        return _DEFAULT_WEIGHTS["bridge"]
    if "pre" in s and "chorus" in s:
        return _DEFAULT_WEIGHTS["pre-chorus"]
    if "intro" in s:
        return _DEFAULT_WEIGHTS["intro"]
    if "outro" in s:
        return _DEFAULT_WEIGHTS["outro"]
    return _DEFAULT_WEIGHTS.get(s, _DEFAULT_WEIGHTS["unknown"])

# real use
def _round_half(x: float) -> float:
    return round(x * 2.0) / 2.0  # 0.5초 스냅

# real use
def allocate_durations_by_lyrics(total_seconds: float, blocks: List[Dict[str, Any]]) -> List[float]:
    if total_seconds <= 0 or not blocks:
        return []
    weights: List[float] = []
    for b in blocks:
        n_lines = max(1, len(b.get("lines") or []))
        w = _section_weight(b.get("section", "unknown")) * n_lines
        weights.append(max(0.1, w))
    total_w = sum(weights) or float(len(blocks))
    raw = [(w / total_w) * total_seconds for w in weights]
    snapped = [_round_half(x) for x in raw]
    diff = total_seconds - sum(snapped)
    if abs(diff) >= 0.25:
        snapped[-1] = max(0.5, snapped[-1] + diff)
    snapped = [max(0.5, x) for x in snapped]
    diff2 = total_seconds - sum(snapped)
    if abs(diff2) >= 0.25:
        snapped[-1] = max(0.5, snapped[-1] + diff2)
    return snapped

# real use
def build_timeline(
    blocks: List[Dict[str, Any]],
    durations: List[float],
    *,
    offset: float = 0.0
) -> List[Dict[str, Any]]:
    """가사 블록과 각 블록 길이로 타임라인 생성."""
    timeline_out: List[Dict[str, Any]] = []
    t = float(offset)

    for block, dur in zip(blocks, durations):
        start = max(0.0, t)
        end = max(start, start + float(dur))

        section = str(block.get("section", "unknown"))
        label = str(block.get("label") or section.title())
        lines = list(block.get("lines", []))

        timeline_out.append({
            "section": section,
            "label": label,
            "lines": lines,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
        })
        t = end

    return timeline_out


# ───────────────────────── 의미 단위 분해 ─────────────────────────
# real use
def _meaning_units_from_lyrics(lines: List[str], *, section: str, ai: Any) -> List[Dict[str, Any]]:
    """
    GPT 우선 의미 단위 분해. 실패하면 [].
    출력 예: [{"lines":[0,1,2], "summary":"감정이 고조되는 장면"}, ...]
    - 내부 변수는 소문자 사용
    - 광범위한 예외 제거(가능한 구체적으로)
    """
    import json
    if not ai or not lines:
        return []

    ask = None
    for name in ("meaning_units_kor", "segment_meaning_units", "chat", "complete", "__call__"):
        fn = getattr(ai, name, None)
        if callable(fn):
            ask = fn
            break
    if not ask:
        return []

    numbered = "\n".join(f"{i}: {line}" for i, line in enumerate(lines))
    prompt = (
        "다음 가사 라인들을 섹션 내부의 '의미 단위'로 묶어줘.\n"
        "각 묶음은 lines(0-based index 배열)과 summary(가사 인용 없이 한국어 의미요약 한 문장)로.\n"
        "JSON 배열만 출력. 예) [{\"lines\":[0,1],\"summary\":\"잔잔한 회상\"}]\n\n"
        f"섹션: {section}\n"
        "가사 라인(번호: 내용):\n" + numbered
    )

    try:
        raw = ask(prompt)
    except (TypeError, ValueError, RuntimeError):
        return []

    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
    elif isinstance(raw, dict) and isinstance(raw.get("units"), list):
        data = raw["units"]
    elif isinstance(raw, list):
        data = raw
    else:
        return []

    units: List[Dict[str, Any]] = []

    def _to_int(v: Any) -> Optional[int]:
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    for group in data:
        if not isinstance(group, dict):
            continue
        raw_line_idxs = group.get("lines")
        if not isinstance(raw_line_idxs, list):
            continue
        line_idxs = [x for x in (_to_int(v) for v in raw_line_idxs) if isinstance(x, int)]
        if not line_idxs:
            continue

        summary = str(group.get("summary") or "").strip()
        if not summary:
            summary = "의미 단위 요약"

        units.append({"lines": line_idxs, "summary": summary})

    return units

# real use
def _rule_meaning_units_from_lines(lines: List[str]) -> List[Dict[str, Any]]:
    """규칙 기반 의미 단위(빈줄/구두점/접속사 단서). 가사 원문 인용 없이 일반 요약문 사용."""
    if not lines:
        return []

    groups: List[List[int]] = []
    current: List[int] = []
    breakers = ("그리고", "하지만", "그러나", "그래도", "그러니", "그래서")

    for idx, line in enumerate(lines):
        s = (line or "").strip()
        if not s:
            if current:
                groups.append(current)
                current = []
            continue

        current.append(idx)
        if any(w in s for w in breakers) or any(p in s for p in (".", "!", "?", "…")):
            groups.append(current)
            current = []

    if current:
        groups.append(current)
    if not groups:
        groups = [[i] for i in range(len(lines))]

    generic_summaries = ("감정 변주를 강조", "인물의 내면에 집중", "공간감 확장", "여운을 남기는 장면", "희미한 회상")
    result: List[Dict[str, Any]] = []
    for i, grp in enumerate(groups):
        result.append({"lines": grp, "summary": generic_summaries[i % len(generic_summaries)]})
    return result

# real use
def _unit_lyric_text(lines: List[str], idxs: List[int]) -> str:
    picked = [lines[i].strip() for i in idxs if 0 <= i < len(lines)]
    joined = " ".join(picked).strip()
    return re.sub(r"\s+([,.!?])", r"\1", joined)



# ───────────────────────── 섹션 내 길이 분배 & 스냅 ─────────────────────────
# real use
def _allocate_and_fix_durations(
    total: float,
    n: int,
    *,
    min_len: float,
    max_len: float,
    target_mean: float
) -> List[float]:
    """
    섹션 총길이(total)를 n개로 나눔.
    1) target_mean을 시드로 분배 → 합계(total) 스케일
    2) [min_len, max_len] 범위로 워터필 보정
    3) 불가능한 경우엔 균등 분배
    """
    total = max(0.0, float(total))
    n = max(1, int(n))
    min_len = float(min_len)
    max_len = float(max_len)
    tm = float(target_mean) if target_mean and target_mean > 0 else (total / n if n else 0.0)

    if n == 1:
        val = total if min_len <= total <= max_len else max(min_len, min(max_len, total))
        return [val]

    feasible_min = n * min_len
    feasible_max = n * max_len

    if total < feasible_min - 1e-9:
        return [total / n] * n
    if total > feasible_max + 1e-9:
        base = [max_len] * n
        scale = total / sum(base)
        return [x * scale for x in base]

    seed = max(min_len, min(max_len, tm))
    vals = [seed] * n

    s = sum(vals)
    vals = [v * (total / s) for v in vals] if s > 0 else [total / n] * n
    vals = [max(min_len, min(max_len, v)) for v in vals]

    def _waterfill(values: List[float], tgt: float, lo: float, hi: float) -> List[float]:
        values = values[:]
        for _ in range(12):
            cur = sum(values)
            diff = cur - tgt
            if abs(diff) <= 1e-6:
                break
            if diff > 0:
                room = [max(0.0, v - lo) for v in values]
                cap = sum(room)
                if cap <= 0:
                    break
                ratio = min(1.0, diff / cap)
                for i, r in enumerate(room):
                    values[i] = max(lo, values[i] - r * ratio)
            else:
                need = [max(0.0, hi - v) for v in values]
                cap = sum(need)
                if cap <= 0:
                    break
                ratio = min(1.0, (-diff) / cap)
                for i, r in enumerate(need):
                    values[i] = min(hi, values[i] + r * ratio)

        final_diff = tgt - sum(values)
        if abs(final_diff) > 1e-6:
            if final_diff > 0:
                for i in range(len(values)):
                    room = hi - values[i]
                    if room > 1e-9:
                        delta = min(room, final_diff)
                        values[i] += delta
                        final_diff -= delta
                        if final_diff <= 1e-9:
                            break
            else:
                need = -final_diff
                for i in range(len(values)):
                    room = values[i] - lo
                    if room > 1e-9:
                        delta = min(room, need)
                        values[i] -= delta
                        need -= delta
                        if need <= 1e-9:
                            break
        return values

    vals = _waterfill(vals, total, min_len, max_len)
    return vals

# real use
def _snap_boundaries_to_onsets(
    borders: List[float],
    onsets: List[float],
    *,
    window: float,
    min_len: float,
    section_start: float,
    section_end: float,
) -> List[float]:
    """
    첫/끝 제외 내부 경계를 가장 가까운 온셋으로 스냅(±window).
    스냅 후 섹션 경계로 앵커링, 정렬/중복 제거, 최소 길이 위반 시 내부 경계 제거.
    """
    if len(borders) <= 2 or not onsets:
        b = [float(x) for x in borders]
        if b:
            b[0] = float(section_start)
            b[-1] = float(section_end)
        return b

    b = [float(x) for x in borders]
    on = sorted(float(o) for o in onsets)

    for i in range(1, len(b) - 1):
        t = b[i]
        lo = section_start + min_len
        hi = section_end - min_len
        candidates = [o for o in on if lo <= o <= hi and abs(o - t) <= window]
        if candidates:
            nearest = min(candidates, key=lambda o: abs(o - t))
            b[i] = nearest

    b[0], b[-1] = float(section_start), float(section_end)
    for i in range(1, len(b) - 1):
        b[i] = max(section_start + 1e-9, min(section_end - 1e-9, b[i]))

    interior = sorted(set(b[1:-1]))
    b = [b[0]] + interior + [b[-1]]

    i = 1
    while i < len(b) - 1:
        left_gap = b[i] - b[i - 1]
        right_gap = b[i + 1] - b[i]
        if left_gap < min_len or right_gap < min_len:
            del b[i]
            continue
        i += 1

    return [round(x, 6) for x in b]

# real use
def _validate_and_fix_segments(
    segs: List[Dict[str, Any]],
    *,
    section: str,
    section_start: float,
    section_end: float,
    min_len: float,
    max_len: float,
) -> List[Dict[str, Any]]:
    """
    0/음수 길이 제거·역전 수정·경계 보정. 반환되는 모든 세그먼트에 'section' 필드를 주입.
    """
    sec = str(section or "unknown")

    fixed: List[Dict[str, Any]] = []
    for seg in segs:
        a = float(seg.get("start", 0.0))
        b = float(seg.get("end", 0.0))
        if b <= a:
            continue
        seg2 = dict(seg)
        seg2["section"] = sec
        fixed.append(seg2)

    if not fixed:
        return []

    result: List[Dict[str, Any]] = []
    i = 0
    while i < len(fixed):
        seg = fixed[i]
        a, b = float(seg["start"]), float(seg["end"])
        d = b - a

        if d < min_len and i + 1 < len(fixed):
            fixed[i + 1]["start"] = a
            fixed[i + 1]["section"] = sec
            fixed[i + 1]["duration"] = round(float(fixed[i + 1]["end"]) - a, 3)
            i += 1
            continue

        if d > max_len:
            k = int((d + max_len - 1e-6) // max_len) + 1
            step = d / k
            t = a
            for _ in range(k):
                u = min(b, t + step)
                seg_new = dict(seg)
                seg_new.update({
                    "section": sec,
                    "start": round(t, 3),
                    "end": round(u, 3),
                    "duration": round(u - t, 3),
                })
                result.append(seg_new)
                t = u
            i += 1
            continue

        seg_ok = dict(seg)
        seg_ok["section"] = sec
        seg_ok["duration"] = round(d, 3)
        result.append(seg_ok)
        i += 1

    if result:
        result[0]["start"] = round(section_start, 3)
        result[-1]["end"] = round(section_end, 3)
        result[0]["duration"] = round(result[0]["end"] - result[0]["start"], 3)
        for idx in range(1, len(result)):
            result[idx]["start"] = round(result[idx - 1]["end"], 3)
            result[idx]["duration"] = round(result[idx]["end"] - result[idx]["start"], 3)
            result[idx]["section"] = sec

    return result



# real use
def _merge_overflow_shots(segs: List[Dict[str, Any]], *, max_shots: int) -> List[Dict[str, Any]]:
    """샷 개수가 너무 많을 때 인접 샷 병합(섹션 유지)으로 감소."""
    if len(segs) <= max_shots:
        return segs

    # 리스트 얕은 복사로 시작(원본 보존). 타입을 명확히 지정해 정적 분석 경고를 방지.
    merged_shots: List[Dict[str, Any]] = list(segs)

    i = 0
    while len(merged_shots) > max_shots and i < len(merged_shots) - 1:
        if merged_shots[i].get("section") == merged_shots[i + 1].get("section"):
            merged_shots[i]["end"] = merged_shots[i + 1].get("end", merged_shots[i].get("end"))
            start_val = float(merged_shots[i].get("start", 0.0))
            end_val = float(merged_shots[i].get("end", start_val))
            merged_shots[i]["duration"] = round(end_val - start_val, 3)
            del merged_shots[i + 1]
        else:
            i += 1

    while len(merged_shots) > max_shots and len(merged_shots) >= 2:
        merged_shots[0]["end"] = merged_shots[1].get("end", merged_shots[0].get("end"))
        start_val0 = float(merged_shots[0].get("start", 0.0))
        end_val0 = float(merged_shots[0].get("end", start_val0))
        merged_shots[0]["duration"] = round(end_val0 - start_val0, 3)
        del merged_shots[1]

    return merged_shots




# real_use
def analyze_project(
    project_dir: str | Path = "",
    *,
    audio_path: str | Path,
    save: bool = True,
    save_vocal: bool = False,           # (보존) 미사용
    vocal_format: str | None = None,    # (보존) 미사용
    project_json_name: str = "project.json",
    # v1.1 옵션
    use_audio_segmentation: bool = True,
    min_shot_sec: float = 1.0,
    max_shot_sec: float = 5.0,
    target_mean_sec: float = 3.0,
    onset_snap_window: float = 0.15,
    max_shots: int = 400,
    ai: Any = None,
    # ▼ NEW: 오디오 대신 이 길이로 전체 타임라인을 강제
    force_total_sec: float | None = None,
) -> Dict[str, Any]:
    """오디오(또는 force_total_sec) 기반으로 timeline/segments 생성."""
    _ = (save_vocal, vocal_format)  # noqa: F841

    audio_file = Path(audio_path).resolve()
    if not audio_file.exists() or not audio_file.is_file():
        raise FileNotFoundError(f"오디오 없음: {audio_file}")

    outdir = audio_file.parent
    outdir.mkdir(parents=True, exist_ok=True)

    audio_dur = float(audio_duration_sec(audio_file))
    if audio_dur <= 0:
        raise RuntimeError(f"오디오 길이 읽기 실패: {audio_file}")

    effective_dur = float(force_total_sec) if (force_total_sec and force_total_sec > 0) else audio_dur

    lyrics_text = ""
    pj1 = outdir / project_json_name
    pj2 = Path(project_dir) / project_json_name if project_dir else None
    meta: Dict[str, Any] = {}
    for pj in (pj1, pj2):
        if pj and pj.exists():
            meta = load_json(pj, {}) or {}
            lyrics_text = (meta.get("lyrics") or "").strip()
            if lyrics_text:
                break

    if lyrics_text:
        lyr_blocks = parse_lyrics_blocks(lyrics_text) or []
        if not lyr_blocks:
            lyr_blocks = [{
                "section": "all",
                "label": "Full",
                "lines": [ln for ln in lyrics_text.splitlines() if ln.strip()],
            }]
        durs = allocate_durations_by_lyrics(effective_dur, lyr_blocks)
        timeline = build_timeline(lyr_blocks, durs, offset=0.0)
    else:
        timeline = [{
            "label": "full",
            "section": "all",
            "lines": [],
            "start": 0.0,
            "end": round(effective_dur, 3),
            "duration": round(effective_dur, 3),
        }]

    # 인트로 보장
    try:
        timeline = _ensure_intro_at_head(
            timeline, effective_dur,
            min_intro=2.5, max_intro=5.0,
            prefer_audio=True, audio_path=audio_file
        )
    except (ValueError, TypeError, RuntimeError):
        pass

    onsets = _detect_onsets_sec(audio_file) if use_audio_segmentation else []

    segments: List[Dict[str, Any]] = []
    for blk in timeline:
        section = str(blk.get("section", "unknown"))
        label = str(blk.get("label") or section)
        s0 = float(blk.get("start", 0.0))
        e0 = float(blk.get("end", 0.0))
        lines = list(blk.get("lines") or [])
        if e0 <= s0:
            continue

        units = _meaning_units_from_lyrics(lines, section=section, ai=ai)
        if not units and lines:
            units = _rule_meaning_units_from_lines(lines)
        if not lines and not units:
            units = [{"lines": [0], "summary": "리듬 포인트 중심"}]

        sec_duration = max(0.0, e0 - s0)
        parts = _allocate_and_fix_durations(
            total=sec_duration,
            n=len(units),
            min_len=min_shot_sec,
            max_len=max_shot_sec,
            target_mean=target_mean_sec
        )

        borders = [s0]
        t = s0
        for d in parts:
            t = min(e0, t + d)
            borders.append(round(t, 6))
        if len(borders) >= 2 and abs(e0 - borders[-1]) > 1e-6:
            borders[-1] = e0

        if onsets:
            borders = _snap_boundaries_to_onsets(
                borders=borders,
                onsets=onsets,
                window=onset_snap_window,
                min_len=min_shot_sec * 0.8,
                section_start=s0,
                section_end=e0
            )

        sec_segs: List[Dict[str, Any]] = []
        for i in range(len(borders) - 1):
            a, b = borders[i], borders[i + 1]
            unit_idx = min(i, len(units) - 1) if units else 0
            unit_info = units[unit_idx] if units else {"lines": [], "summary": ""}
            # 의미 단위에 해당하는 가사(블록 내부 라인 기준)
            lyric_txt = _unit_lyric_text(lines, unit_info.get("lines", [])) if lines else ""

            sec_segs.append({
                "label": label,
                "section": section,
                "start": round(a, 3),
                "end": round(b, 3),
                "duration": round(b - a, 3),
                "summary": str(unit_info.get("summary") or "감정/장면 변주 강조"),
                "lyric": lyric_txt,
            })

        sec_segs = _validate_and_fix_segments(
            segs=sec_segs,
            section=section,
            section_start=s0,
            section_end=e0,
            min_len=min_shot_sec,
            max_len=max_shot_sec
        )
        segments.extend(sec_segs)

    if len(segments) > max_shots:
        segments = _merge_overflow_shots(segments, max_shots=max_shots)

    scenes_for_scene_json = [{
        "label": it.get("label") or it.get("section") or "scene",
        "section": it.get("section", "unknown"),
        "start": float(it.get("start", 0.0)),
        "end": float(it.get("end", 0.0)),
        "duration": float(it.get("duration", 0.0)),
    } for it in timeline]

    scene_payload = {
        "audio": str(audio_file),
        "offset": 0.0,
        "duration": round(audio_dur, 3),          # 실제 오디오 길이
        "effective_duration": round(effective_dur, 3),
        "scenes": scenes_for_scene_json,
    }

    seg_path = (outdir / "segments.json") if save else None
    scn_path = (outdir / "scene.json") if save else None
    if save:
        segments_file_payload = {
            "audio": str(audio_file),
            "duration": round(effective_dur, 3),
            "timeline": timeline,
            "segments": segments,
        }
        save_json(seg_path, segments_file_payload); print(f"[ANALYZE] wrote {seg_path}", flush=True)
        save_json(scn_path, scene_payload);         print(f"[ANALYZE] wrote {scn_path}", flush=True)

    return {
        "audio": str(audio_file),
        "duration": float(effective_dur),
        "timeline": timeline,
        "segments": segments,
        "segments_path": str(seg_path) if seg_path else "",
        "segments_payload": segments,
        "scene_path": str(scn_path) if scn_path else "",
        "scene_payload": scene_payload,
        "meta": meta,
    }





# ───────────────────────── 인트로 삽입(섹션 타임라인 단계) ─────────────────────────
# real_use
def _ensure_intro_at_head(
    timeline: List[Dict[str, Any]],
    total_dur: float,
    *,
    intro_ratio: float = 0.10,          # 전체 길이의 비율(기본 10%)
    min_intro: float = 1.5,             # 하한(초)
    max_intro: Optional[float] = None,  # 상한(초). None이면 자동
    prefer_audio: bool = True,
    audio_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    인트로 섹션이 없으면, 전체 길이의 intro_ratio만큼 앞에 끼워 넣고 뒤 블록들을 밀어준다.
    (오디오의 초반 무음/비보컬 오프셋을 힌트로 사용 가능)
    """
    if total_dur <= 0:
        return timeline

    has_intro = any(str(b.get("section", "")).lower().startswith("intro") for b in timeline)
    if has_intro:
        return timeline

    base_guess = float(max(0.0, intro_ratio)) * float(total_dur)

    if prefer_audio and audio_path is not None:
        try:
            off, _eff, _full = _librosa_trim_edges(audio_path)
            off = float(off)
            if 0.2 <= off <= total_dur * 0.5:
                base_guess = max(base_guess, off)
        except (ValueError, RuntimeError, OSError):
            pass

    hi = max_intro if (max_intro is not None) else max(0.5, total_dur * 0.25)
    length = max(min_intro, min(hi, base_guess))
    length = float(max(0.5, min(float(total_dur) - 0.01, length)))

    new_tl: List[Dict[str, Any]] = [{
        "section": "intro",
        "label": "Intro",
        "lines": [],
        "start": 0.0,
        "end": round(length, 3),
        "duration": round(length, 3),
    }]

    for blk in timeline:
        s_val = float(blk.get("start", 0.0)) + length
        e_val = float(blk.get("end", 0.0)) + length
        if e_val <= length + 1e-6:
            continue
        s_val = max(s_val, length)
        e_val = min(e_val, float(total_dur))
        if e_val <= s_val:
            continue
        b2 = dict(blk)
        b2["start"] = round(s_val, 3)
        b2["end"] = round(e_val, 3)
        b2["duration"] = round(e_val - s_val, 3)
        new_tl.append(b2)

    if new_tl:
        new_tl[-1]["end"] = round(float(total_dur), 3)
        new_tl[-1]["duration"] = round(float(new_tl[-1]["end"]) - float(new_tl[-1]["start"]), 3)

    return new_tl





# ─────────────────────────────────────────────────────────────
# 0) 유틸: 오디오 길이 견고 획득(중앙값, 3배 튐 방지)
# ─────────────────────────────────────────────────────────────
# real_use
# def _probe_duration_ffprobe(path: str) -> float:
#     import json
#     try:
#         out = subprocess.check_output(
#             ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
#             stderr=subprocess.STDOUT,
#         )
#         data = json.loads(out.decode("utf-8", "ignore"))
#         dur = float(data.get("format", {}).get("duration", 0.0) or 0.0)
#         return dur if math.isfinite(dur) and dur > 0 else 0.0
#     except Exception:
#         return 0.0
# # real_use
# def _probe_duration_mutagen(path: str) -> float:
#     try:
#           # type: ignore
#         mf = MutagenFile(path)
#         dur = float(getattr(mf, "info", None).length if mf and getattr(mf, "info", None) else 0.0)
#         return dur if math.isfinite(dur) and dur > 0 else 0.0
#     except Exception:
#         return 0.0
# # real_use
# def _probe_duration_pydub(path: str) -> float:
#     try:
#
#         dur_ms = len(AudioSegment.from_file(path))
#         dur = float(dur_ms) / 1000.0
#         return dur if math.isfinite(dur) and dur > 0 else 0.0
#     except Exception:
#         return 0.0
# # real_use
# def _probe_duration_soundfile(path: str) -> float:
#     try:
#         import soundfile as zsf  # type: ignore
#         with zsf.SoundFile(path) as f:
#             dur = float(len(f) / (f.samplerate or 1))
#             return dur if math.isfinite(dur) and dur > 0 else 0.0
#     except Exception:
#         return 0.0
# # real_use
# def _probe_duration_wave(path: str) -> float:
#     try:
#         import wave
#         with wave.open(path, "rb") as wf:
#             frames = wf.getnframes()
#             rate = wf.getframerate() or 1
#             dur = float(frames) / float(rate)
#             return dur if math.isfinite(dur) and dur > 0 else 0.0
#     except Exception:
#         return 0.0
# # real_use
# def _probe_duration_librosa(path: str) -> float:
#     try:
#         import librosa  # type: ignore
#         try:
#             dur = float(librosa.get_duration(path=path))
#         except TypeError:
#             dur = float(librosa.get_duration(filename=path))
#         return dur if math.isfinite(dur) and dur > 0 else 0.0
#     except Exception:
#         return 0.0
# # real_use
# def get_audio_duration(path: str) -> float:
#     p = str(path or "").strip()
#     if not p or not os.path.isfile(p):
#         return 0.0
#     cands: List[float] = []
#     for fn in (
#         _probe_duration_ffprobe,
#         _probe_duration_mutagen,
#         _probe_duration_pydub,
#         _probe_duration_soundfile,
#         _probe_duration_wave,
#         _probe_duration_librosa,
#     ):
#         v = fn(p)
#         if v and math.isfinite(v) and v > 0:
#             cands.append(v)
#     if not cands:
#         return 0.0
#     cands.sort()
#     mid = len(cands) // 2
#     if len(cands) % 2 == 1:
#         median = cands[mid]
#     else:
#         median = 0.5 * (cands[mid - 1] + cands[mid])
#     mn, mx = cands[0], cands[-1]
#     if mx / max(mn, 1e-9) >= 2.4 and mn > 0:
#         return mn
#     return median


# real use
def separate_vocals_demucs(input_path: str) -> Optional[str]:
    try:
        if not os.path.isfile(input_path):
            return None
        # 단순 존재 확인만 하면 되므로 결과를 변수에 담지 않는다.
        subprocess.check_output(["demucs", "--help"], stderr=subprocess.STDOUT)
    except Exception:
        return None
    try:
        import tempfile
        outdir = tempfile.mkdtemp(prefix="demucs_")
        subprocess.check_call(["demucs", "-n", "htdemucs", "-o", outdir, input_path])
        vocals = None
        for root, _, files in os.walk(outdir):
            for fn in files:
                if fn.lower() in ("vocals.wav", "vocals.flac"):
                    vocals = os.path.join(root, fn)
                    break
        return vocals
    except Exception:
        return None
















# -*- coding: utf-8 -*-
"""
ACE-Step(ComfyUI) 음악 생성 — 태그/길이 병합·주입 + 포맷 보장(노드 적응 + 후처리 트랜스코딩)

기능 요약
- project.json의 가사/태그/길이를 주입하여 ComfyUI 워크플로 실행
- LyricsLangSwitch(가사), TextEncodeAceStepAudio(tags), EmptyAceStepLatentAudio(seconds) 주입
- KSampler seed 랜덤화로 변주
- SaveAudio 계열 filename_prefix 고정(프로젝트별 하위폴더: shorts_make/<title>/vocal_final*)
- /history 스키마 A/B 모두 지원(+ node_errors 감지)
- 서버의 지원 노드(/object_info) 확인 → 가능한 저장 노드로 '적응'해서 class_type 변경
  * 없으면 MP3 저장으로 폴백(quality='320k')
- /view로 결과 다운로드 → settings.AUDIO_SAVE_FORMAT으로 최종 보장(필요 시 ffmpeg 트랜스코딩)
- 결과를 프로젝트 폴더에 'vocal.<fmt>'로 저장하고,
  설정(FINAL_OUT)에 지정한 사용자 폴더([title] 치환)에 자동 복사
- project.json 갱신(tags_effective, comfy_debug)
- 디버그: _prompt_sent.json, _history_raw.json 저장
"""



# settings 상수(대기/폴링 주기)

# ── settings / utils 유연 임포트 ───────────────────────────────────────────────

#
# _DEFAULT_ACE_WAIT_TIMEOUT_SEC = 900.0   # 15분
# _DEFAULT_ACE_POLL_INTERVAL_SEC = 2.0    # 2초
# real use
def _ace_wait_timeout_sec():
    return (
        (getattr(_s, "ACE_STEP_WAIT_TIMEOUT_SEC", None) if _s else None)
        or (getattr(_s, "ACE_WAIT_TIMEOUT_SEC", None) if _s else None)
        or _DEFAULT_ACE_WAIT_TIMEOUT_SEC
    )
# real use
def _ace_poll_interval_sec():
    return (
        (getattr(_s, "ACE_STEP_POLL_INTERVAL_SEC", None) if _s else None)
        or (getattr(_s, "ACE_POLL_INTERVAL_SEC", None) if _s else None)
        or _DEFAULT_ACE_POLL_INTERVAL_SEC
    )



# ─────────────────────────────  ────────────────────────────────────






# ───────────────────────────── HTTP 유틸 ────────────────────────────────────
# real use
def _http_get(base: str, path: str, timeout: int = 30, params: Optional[dict] = None) -> requests.Response:
    return requests.get(base.rstrip("/") + path, params=params or {}, timeout=timeout)

# real use
def _probe_server(base: str, timeout: int = 3) -> bool:
    for p in ("/view", "/history"):
        try:
            r = requests.get(base.rstrip("/") + p, timeout=timeout)
            if r.status_code == 200:
                return True
        except Exception:
            pass
    return False


# real use
def _load_workflow_graph(json_path: str | Path) -> dict:
    """
    ComfyUI용 워크플로 JSON을 로드해, /prompt에 바로 넣을 수 있는
    '노드 딕셔너리' 형태로 정규화해서 반환한다.

    추가 보정(기능 불변, 참조만 정리):
      - 문자열 형태 노드 참조 '#74' → '74'
      - 잘못된 3원소 참조 예: ["문자태그", "17", 0] → ["17", 0]
      - {"nodes":[...]} / {id→node} 두 포맷 모두 정규화
    """


    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"워크플로 JSON 없음: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1) 'prompt' 키로 감싸져 있으면 내부만 꺼냄
    g: Any = data.get("prompt") if isinstance(data, dict) else None
    if not isinstance(g, (dict, list)):
        g = data  # prompt 키가 없으면 전체가 그래프라고 가정

    # 2) {"nodes":[{id:.., class_type:..}, ...]} → {id(str): node(dict)}
    if isinstance(g, dict) and "nodes" in g and isinstance(g["nodes"], list):
        nodes_list = g["nodes"]
        tmp: dict[str, dict] = {}
        for node_dict in nodes_list:
            if not isinstance(node_dict, dict):
                continue
            node_id = str(node_dict.get("id") or "").strip()
            if not node_id:
                raise ValueError("워크플로 형식 오류: nodes[*].id 가 없음")
            node_copy = {k: v for k, v in node_dict.items() if k != "id"}
            tmp[node_id] = node_copy
        g = tmp

    # 3) 최종은 dict 여야 함
    if not isinstance(g, dict):
        raise ValueError("워크플로 형식 오류: prompt 그래프가 dict 가 아님")

    # 4) 각 노드 검증
    bad = [nid for nid, node in g.items() if not (isinstance(node, dict) and "class_type" in node)]
    if bad:
        raise ValueError(
            f"워크플로 노드에 class_type 누락: {', '.join(bad[:10])}" + ("..." if len(bad) > 10 else "")
        )

    # ──────────────── ★ 참조 정규화 블록 ────────────────
    rx_hash_id = re.compile(r"^#(\d+)$")

    def _unhash(x: Any) -> Any:
        if isinstance(x, str):
            match_obj = rx_hash_id.match(x)
            if match_obj:
                return match_obj.group(1)
        return x

    def _fix_seq(seq: list[Any]) -> list[Any]:
        out = [_unhash(v) for v in seq]
        # ["문자", "17", 0] → ["17", 0]
        if (
            len(out) == 3
            and isinstance(out[0], str)
            and not out[0].isdigit()
            and isinstance(out[1], (str, int))
            and out[2] == 0
        ):
            return [str(out[1]), 0]
        # ["#17", 0] → ["17", 0]
        if len(out) == 2 and isinstance(out[0], str):
            return [str(_unhash(out[0])), out[1]]
        return out

    def _normalize_inputs(inputs_map: dict[str, Any]) -> None:
        for key, value in list(inputs_map.items()):
            if isinstance(value, str):
                inputs_map[key] = _unhash(value)
            elif isinstance(value, list):
                inputs_map[key] = _fix_seq(value)
            elif isinstance(value, dict):
                inner_map = value
                # 1단계 깊이만 정리(워크플로 입력 스키마 상 충분)
                for inner_key, inner_val in list(inner_map.items()):
                    if isinstance(inner_val, str):
                        inner_map[inner_key] = _unhash(inner_val)
                    elif isinstance(inner_val, list):
                        inner_map[inner_key] = _fix_seq(inner_val)

    # dict(id→node) 전체 순회하며 inputs 정규화
    for _nid, node in g.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, dict):
            _normalize_inputs(inputs)
            # TextEncodeAceStepAudio의 lyrics가 '#숫자'면 ["숫자", 0]로 보정
            if node.get("class_type") == "TextEncodeAceStepAudio":
                val = inputs.get("lyrics")
                if isinstance(val, list):
                    inputs["lyrics"] = _fix_seq(val)
                elif isinstance(val, str):
                    mo = rx_hash_id.match(val)
                    if mo:
                        inputs["lyrics"] = [mo.group(1), 0]
    # ─────────────────────────────────────────────

    return g





# real use
def _choose_host() -> str:
    """settings의 COMFY_HOST 우선, 그 다음 DEFAULT_HOST_CANDIDATES에서 살아있는 서버를 선택."""
    cand: List[str] = []
    if COMFY_HOST:
        cand.append(COMFY_HOST)
    for c in (getattr(settings, "DEFAULT_HOST_CANDIDATES", DEFAULT_HOST_CANDIDATES) or []):
        if c and c not in cand:
            cand.append(c)
    for host in cand:
        if host and _probe_server(host):
            return host
    return cand[0] if cand else "http://127.0.0.1:8188"



# real_use
def _collect_effective_tags(meta: dict) -> List[str]:
    """
    [수정 v15] project.json에서 실제 주입할 태그 리스트:
    - 1순위: AI가 긍정 프롬프트로 생성한 태그 (meta['prompt_user_ai_tags'])
    - 2순위 (폴백): auto_tags==True시 (UI태그) + (AI추천태그)
    - 3순위 (폴백): auto_tags==False시 (UI태그)
    - [수정 v16] 긍정 프롬프트(prompt_user) 텍스트를 태그로 분리하는 로직 *제거*.
    """

    # --- 1. [신규] 긍정 프롬프트 기반 AI 태그 (최우선) ---
    ai_tags = meta.get("prompt_user_ai_tags")

    # UI 체크박스 태그 (auto_tags==True일 때만 '추가'로 사용)
    ui_tags = []
    if meta.get("auto_tags", True):
        ui_tags = list(meta.get("ace_tags") or [])
        ui_tags.extend(meta.get("tags_in_use") or [])
        _dlog("[_collect_effective_tags] Using AI Tags (if present) + UI 'Auto' Tags")
    else:
        ui_tags = list(meta.get("manual_tags") or [])
        _dlog("[_collect_effective_tags] Using AI Tags (if present) + UI 'Manual' Tags")

    if isinstance(ai_tags, list) and ai_tags:
        # AI 태그가 있으면 UI 태그와 결합
        combined = ai_tags + ui_tags
    else:
        # AI 태그가 없으면 (v14 폴백 제거됨), UI 태그만 사용
        _dlog("[_collect_effective_tags] No AI tags found (prompt_user_ai_tags). Using UI tags only.")
        combined = ui_tags

    # 3. 모두 합치고 중복 제거 (공통)
    seen = set()
    unique_tags = []
    for tag in combined:
        t_lower = tag.lower()
        if t_lower not in seen:
            seen.add(t_lower)
            unique_tags.append(tag)

    _dlog(f"[_collect_effective_tags] Total tags collected: {len(unique_tags)}")
    return unique_tags


# real_use
def generate_music_with_acestep(
        project_dir: str,
        *,
        on_progress: Optional[Callable[[dict], None]] = None,
        target_seconds: int | None = None,
) -> str:
    """
    ComfyUI(ACE-Step) 단일 트랙 음악 생성 — project.json 단일 소스 정책.
    - [수정 v15] 요청하신 대로 세미콜론(;)을 제거하고 줄바꿈을 적용하여 구문 오류를 수정합니다.
    - [수정 v14] 태그가 적용되지 않는 문제 해결 (v13 로직 폐기):
        * 긍정 프롬프트 박스 + UI 태그를 결합하여 Node 14('tags')에 주입.
        * Node 74('lyrics')에는 순수 가사만 주입.
    - lyrics_lls_after 복원 등 이전 수정사항 유지.
    - [추가] 부정 프롬프트(negative tags)를 Node 80(CLIPTextEncode.text)에 주입하여 KSampler negative로 반영.
    """

    # --- 내부 헬퍼 함수 ---
    def notify(stage: str, **kw: Any) -> None:
        if on_progress:
            try:
                info: Dict[str, Any] = {"stage": stage}
                info.update(kw)
                on_progress(info)
            except (TypeError, ValueError, RuntimeError):
                pass

    def _iter_nodes(graph: dict) -> Any:
        if not isinstance(graph, dict):
            return
        nodes_list = graph.get("nodes")
        if isinstance(nodes_list, list):
            for nobj in nodes_list:
                if isinstance(nobj, dict):
                    yield nobj
        else:
            for _k, nobj in graph.items():
                if isinstance(nobj, dict) and "class_type" in nobj:
                    yield nobj
                elif isinstance(nobj, dict) and "prompt" in nobj and isinstance(nobj["prompt"], dict):
                    for _pk, p_nobj in nobj["prompt"].items():
                        if isinstance(p_nobj, dict) and "class_type" in p_nobj:
                            yield p_nobj

    def _force_wav_save_for_this_graph(graph: dict, *, proj_dir: Path, filename_prefix: str) -> str:
        for nobj in _iter_nodes(graph):
            ct_local = str(nobj.get("class_type", "")).lower()
            if ct_local.startswith("saveaudio"):
                inputs_map_local = nobj.setdefault("inputs", {})
                inputs_map_local["filename_prefix"] = filename_prefix
                if "output_path" in inputs_map_local:
                    inputs_map_local["output_path"] = str(proj_dir)
                for base_key in ("basename", "base_filename", "filename"):
                    if base_key in inputs_map_local:
                        inputs_map_local[base_key] = "vocal"
                for wav_key, wav_val in (("format", "wav"), ("container", "wav"), ("codec", "pcm_s16le")):
                    if wav_key in inputs_map_local:
                        inputs_map_local[wav_key] = wav_val
                if "sample_rate" in inputs_map_local:
                    inputs_map_local.setdefault("sample_rate", 44100)
                for bd_key in ("bit_depth", "bitdepth", "bits"):
                    if bd_key in inputs_map_local:
                        inputs_map_local[bd_key] = 16
        return ".wav"

    def _call_comfy_api_for_ko_chunk(text_chunk: str, host_address: str) -> str:
        """
        한글 덩어리를 전용 워크플로우로 변환 시도. (v12 파싱 로직 유지)
        """
        lyrics_node_id = "74"
        result_node_id = "75"  # PreviewAny 노드
        save_text_node_id = "79"

        transformed_result = text_chunk
        api_workflow_path: Optional[Path] = None

        try:
            jsons_dir_val = getattr(settings, "JSONS_DIR", "jsons")
            api_workflow_path = Path(jsons_dir_val) / "ace_step_text_change.json"
            if not api_workflow_path.exists():
                api_workflow_path = Path("jsons") / "ace_step_text_change.json"
                if not api_workflow_path.exists():
                    raise FileNotFoundError("API workflow not found")
            with open(api_workflow_path, 'r', encoding='utf-8') as f_api_wf:
                api_workflow_graph = json.load(f_api_wf)
        except Exception as load_api_wf_err:
            error_path_str = str(api_workflow_path) if api_workflow_path else "configured/default path"
            _dlog(f"[ERROR] Comfy API(Chunk): Failed load/parse API WF '{error_path_str}'. Err: {load_api_wf_err}.")
            return transformed_result

        try:
            if lyrics_node_id in api_workflow_graph:
                api_workflow_graph[lyrics_node_id].setdefault("inputs", {})["lyrics"] = text_chunk
            else:
                _dlog(f"[WARN] Comfy API(Chunk): Node {lyrics_node_id} not found.")
                return transformed_result
            if save_text_node_id in api_workflow_graph:
                save_inputs = api_workflow_graph[save_text_node_id].setdefault("inputs", {})
                save_inputs["text"] = [lyrics_node_id, 0]
                save_inputs["filename_prefix"] = f"comfy_api_transform_temp_{uuid.uuid4().hex[:8]}"
                save_inputs["append"] = "overwrite"

            prompt_payload_api = {"prompt": api_workflow_graph, "client_id": str(uuid.uuid4())}
            api_url_post = f"{host_address.rstrip('/')}/prompt"
            response_post_api = requests.post(api_url_post, json=prompt_payload_api, timeout=10)
            response_post_api.raise_for_status()
            response_json_api = response_post_api.json()
            prompt_id_api = response_json_api.get('prompt_id')
            if not prompt_id_api:
                _dlog("[WARN] Comfy API(Chunk): Prompt ID 없음.")
                return transformed_result

            start_time_poll_api = time.time()
            api_url_get = f"{host_address.rstrip('/')}/history/{prompt_id_api}"
            history_data_api: Optional[Dict] = None
            found_result_text: Optional[str] = None

            while time.time() - start_time_poll_api < 15:
                try:
                    response_get_api = requests.get(api_url_get, timeout=5)
                    if response_get_api.ok:
                        history_data_api = response_get_api.json()
                        if prompt_id_api in history_data_api:
                            history_entry = history_data_api[prompt_id_api]
                            status_obj = history_entry.get("status", {})
                            status_str = status_obj.get("status_str", "")
                            if status_str == "error":
                                _dlog(
                                    f"[ERROR] Comfy API(Chunk): Job failed. Err: {status_obj.get('exception_message')}")
                                return transformed_result

                            outputs_api = history_entry.get('outputs')
                            if isinstance(outputs_api, dict):
                                result_node_output = outputs_api.get(result_node_id)
                                if isinstance(result_node_output, dict):
                                    result_list = result_node_output.get('text')
                                    if isinstance(result_list, list) and result_list:
                                        potential_text = result_list[0]
                                        if isinstance(potential_text, str) and potential_text.strip():
                                            found_result_text = potential_text.strip()
                                            break

                            if status_obj.get("completed", False):
                                _dlog(
                                    f"[WARN] Comfy API(Chunk): Job completed but Node {result_node_id} output was missing or invalid.")
                                _dlog("       History Entry:", json.dumps(history_entry, indent=2, ensure_ascii=False))
                                break

                except requests.exceptions.Timeout:
                    pass
                except requests.exceptions.RequestException as poll_err:
                    _dlog(f"[WARN] Comfy API(Chunk): History polling error: {poll_err}")

                time.sleep(0.5)

            if found_result_text is not None:
                transformed_result = found_result_text
                _dlog("[DEBUG] Comfy API(Chunk) 변환 성공:", text_chunk, "->", transformed_result)
                return transformed_result
            else:
                _dlog("[WARN] Comfy API(Chunk): 폴링 시간 초과 또는 최종 결과 없음. 원본 반환:", text_chunk)
                if history_data_api:
                    _dlog("       Last History Response:",
                          json.dumps(history_data_api.get(prompt_id_api, {}), indent=2, ensure_ascii=False))
                return text_chunk

        except requests.exceptions.RequestException as req_err_api:
            _dlog(f"[ERROR] Comfy API(Chunk) 호출 실패: {req_err_api}.")
        except Exception as exy:
            _dlog(f"[ERROR] Comfy API(Chunk) 처리 오류: {exy}.")
        return text_chunk

        # --- 메인 로직 시작 ---

    _dlog("ENTER", f"project_dir={project_dir}")
    proj = Path(project_dir)
    proj.mkdir(parents=True, exist_ok=True)
    pj = proj / "project.json"
    meta: Dict[str, Any] = load_json(pj, {}) or {}
    title = effective_title(meta)
    lyrics_raw = (meta.get("lyrics") or "").strip()

    use_lls = meta.get("lls_enabled") is True
    _dlog("LLS_CHECK", f"lls_enabled is {meta.get('lls_enabled')}, use_lls set to {use_lls}")

    try:
        comfy_host_addr_local = _choose_host()
    except Exception as host_err:
        _dlog(f"[ERROR] ComfyUI host selection failed: {host_err}.")
        return f"Music generation failed: Cannot connect to ComfyUI host ({getattr(settings, 'COMFY_HOST', 'default')})."

    lyrics_eff: str
    if use_lls:
        lyrics_eff = (meta.get("lyrics_lls") or "").strip()
        _dlog("LYRICS_MODE", "Using LLS (from project.json)")
        if not lyrics_eff:
            _dlog("[ERROR] LLS enabled but 'lyrics_lls' is empty.")
            return "Music generation failed: LLS enabled but lyrics_lls is empty."
    else:
        _dlog("LYRICS_MODE", "Using Raw Lyrics (Applying split/API transform with dedicated workflow)")
        processed_lines_new: List[str] = []
        chunk_pattern = re.compile(r'[가-힣]+(?:[,\s]*[가-힣]+)*|[^가-힣]+(?:[,\s]*[^가-힣]+)*')
        original_lines = lyrics_raw.splitlines()
        total_lines_to_process = len(original_lines)
        _dlog("LYRICS_PROCESSING_START", f"Total lines: {total_lines_to_process}")

        for line_index, line in enumerate(original_lines):
            stripped_line = line.strip()
            if (line_index + 1) % 5 == 0 or (line_index + 1) == total_lines_to_process:
                _dlog("LYRICS_PROCESSING_PROGRESS", f"Processing line {line_index + 1}/{total_lines_to_process}")
            if not stripped_line:
                processed_lines_new.append("")
                continue
            if stripped_line.startswith('[') and stripped_line.endswith(']'):
                processed_lines_new.append(stripped_line.lower())
                continue

            tagged_chunks: List[str] = []
            found_chunks = chunk_pattern.findall(stripped_line)

            if not found_chunks and stripped_line:
                tagged_chunks.append(f"[en]{stripped_line}")
            else:
                for chunk_text in found_chunks:
                    stripped_chunk = chunk_text.strip()
                    if not stripped_chunk:
                        continue
                    if re.search(r'[가-힣]', stripped_chunk):
                        transformed_ko_chunk = _call_comfy_api_for_ko_chunk(stripped_chunk, comfy_host_addr_local)
                        if transformed_ko_chunk.startswith("[ko]"):
                            tagged_chunks.append(transformed_ko_chunk)
                        else:
                            tagged_chunks.append(f"[ko]{stripped_chunk}")
                            _dlog(f"[WARN] API result for '{stripped_chunk}' invalid. Used original.")
                    else:
                        tagged_chunks.append(f"[en]{stripped_chunk}")

            processed_line_result = " ".join(tagged_chunks)
            processed_lines_new.append(processed_line_result)

        lyrics_eff = "\n".join(processed_lines_new)
        _dlog("LYRICS_PROCESSING_END", f"Finished processing {total_lines_to_process} lines.")
        _dlog("[DEBUG] Final lyrics_eff preview (New Logic, first 200 chars):", lyrics_eff[:200])

    if not lyrics_eff:
        _dlog("[ERROR] No effective lyrics.")
        return "Music generation failed: No effective lyrics."

    seconds_val: int = 60
    positive_tags: List[str] = []
    negative_tags: List[str] = []
    main_wf_path: Union[str, Path] = getattr(settings, "ACE_STEP_PROMPT_JSON", "jsons/ace_step_1_t2m.json")
    graph_loaded: Optional[Dict] = None
    base_host: str = comfy_host_addr_local
    subfolder_path: str = ""
    save_prefix: str = ""
    output_ext: str = ".wav"
    history_result: Optional[Dict] = None
    final_audio_path: Optional[Path] = None
    library_saved_path: Optional[Path] = None
    lls_after_result: str = ""

    try:
        if target_seconds is not None:
            seconds_val = int(max(1, target_seconds))
        else:
            time_meta = meta.get("time")
            ts_meta = meta.get("target_seconds")
            try:
                seconds_val = int(ts_meta) if ts_meta is not None else int(time_meta)  # type: ignore
            except (ValueError, TypeError):
                seconds_val = 60
            seconds_val = int(max(1, seconds_val))

        meta["target_seconds"] = seconds_val
        meta["time"] = seconds_val
        meta["lyrics_lls_now"] = lyrics_eff

        try:
            save_json(pj, meta)
            _dlog("META_SAVE_PRE", f"Saved target_seconds={seconds_val} and lyrics_lls_now")
        except OSError as e:
            _dlog(f"[WARN] Failed save pre-gen meta: {e}")

        positive_tags = _collect_effective_tags(meta)
        tags_string = ", ".join(positive_tags)
        _dlog("TAGS_TO_INJECT (Combined)", tags_string)

        neg_raw = meta.get("prompt_neg") or ""
        negative_tags = [t.strip() for t in re.split(r'[,;\n]+', str(neg_raw)) if t.strip()]

        try:
            graph_loaded = _load_workflow_graph(main_wf_path)
        except Exception as e:
            _dlog(f"[ERROR] Failed load main WF: {e}")
            raise

        _dlog("HOST", base_host, "| DESIRED_FMT wav")
        sanitized_title = sanitize_title(title)
        subfolder_path = f"shorts_make/{sanitized_title}"
        save_prefix = f"{subfolder_path}/vocal_final"
        output_ext = _force_wav_save_for_this_graph(graph_loaded, proj_dir=proj, filename_prefix=save_prefix)

        try:
            lyrics_id = "74"
            txt_pos_id = "14"
            sec_id = "17"
            sampler_id = "52"

            if lyrics_id in graph_loaded:
                graph_loaded[lyrics_id].setdefault("inputs", {})["lyrics"] = lyrics_eff
                _dlog("INJECT", f"Node {lyrics_id} (Pure Lyrics)")

            if txt_pos_id in graph_loaded:
                graph_loaded[txt_pos_id].setdefault("inputs", {})["tags"] = tags_string
                _dlog("INJECT", f"Node {txt_pos_id} (Combined Tags)")

            if sec_id in graph_loaded:
                graph_loaded[sec_id].setdefault("inputs", {})["seconds"] = seconds_val
                _dlog("INJECT", f"Node {sec_id} (Seconds)")
            if sampler_id in graph_loaded:
                graph_loaded[sampler_id].setdefault("inputs", {})["seed"] = _rand_seed()
                _dlog("INJECT", f"Node {sampler_id} (Seed)")

            # (추가) 부정 프롬프트(negative tags)도 워크플로에 주입
            # ace_step_1_t2mm.json 기준: Node 80(CLIPTextEncode.text) → KSampler.negative 입력
            neg_txt_id = "80"
            neg_string = ", ".join(negative_tags).strip()
            if neg_txt_id in graph_loaded:
                graph_loaded[neg_txt_id].setdefault("inputs", {})["text"] = neg_string
                _dlog("INJECT", f"Node {neg_txt_id} (Negative Prompt) len={len(neg_string)}")
            else:
                if neg_string:
                    _dlog("WARN", f"Negative prompt node {neg_txt_id} not found. negative will be ignored.")
        except Exception as e:
            _dlog(f"[ERROR] Main WF injection failed: {e}")
            raise

        notify("submitting", host=base_host)
        progress_cb = on_progress if callable(on_progress) else (lambda i: _dlog("PROG", i))

        try:
            dbg_wf = proj / "_debug_workflow_sent.json"
            save_json(dbg_wf, {"prompt": graph_loaded})
            _dlog("DEBUG_WORKFLOW_SAVE", f"Saved final WF to {dbg_wf.name}")
        except Exception as e:
            _dlog(f"[WARN] Failed save debug WF: {e}")

        history_result = _submit_and_wait(base_host, graph_loaded, timeout=_ace_wait_timeout_sec(),
                                          poll=_ace_poll_interval_sec(), on_progress=progress_cb)

        saved_files_list: List[Path] = []
        outputs_hist = history_result.get("outputs") if isinstance(history_result, dict) else None

        if isinstance(outputs_hist, dict):
            for _nid, node_out in outputs_hist.items():
                for key in ("audio", "audios", "files", "wav", "mp3", "output"):
                    arr = node_out.get(key)
                    if not isinstance(arr, list):
                        continue
                    for item in arr:
                        if not isinstance(item, dict):
                            continue
                        fn = (item.get("filename") or "").strip()
                        sf = (item.get("subfolder") or "").strip()
                        if not fn or fn.startswith("ComfyUI_temp_"):
                            continue
                        sf_norm = sf.replace("\\", "/").lstrip("/")

                        try:
                            dl_file = _download_output_file(base_host, fn, sf_norm, out_dir=proj)
                            if isinstance(dl_file, Path) and dl_file.exists() and dl_file.stat().st_size > 0:
                                saved_files_list.append(dl_file)
                                _dlog("DOWNLOAD_SUCCESS", f"File: '{fn}' -> '{dl_file.name}'")
                        except Exception as e:
                            _dlog("DOWNLOAD_ERROR", f"'{fn}': {type(e).__name__}")

        if saved_files_list:
            wavs = sorted([p for p in saved_files_list if p.suffix.lower() == ".wav"], key=lambda p: p.stat().st_mtime,
                          reverse=True)
            others = sorted([p for p in saved_files_list if p.suffix.lower() != ".wav"],
                            key=lambda p: p.stat().st_mtime, reverse=True)
            candidates = wavs + others
            source_audio: Optional[Path] = candidates[0] if candidates else None

            if source_audio and source_audio.exists():
                _dlog("SELECTED_SOURCE_AUDIO", f"Selected '{source_audio.name}'.")
                ffmpeg_p = getattr(settings, "FFMPEG_EXE", "ffmpeg") or "ffmpeg"
                try:
                    final_audio_path = _ensure_vocal_wav(source_audio, proj, ffmpeg_exe=ffmpeg_p)
                    _dlog("ENSURED_WAV", f"Final WAV: '{final_audio_path.name}'")

                    master_fn = globals().get("_master_wav_precise")
                    if callable(master_fn) and isinstance(final_audio_path, Path) and final_audio_path.exists():
                        _dlog("MASTERING_START", "Applying mastering...")
                        try:
                            ti = float(getattr(settings, "MASTER_TARGET_I", -12.0))
                            tp = float(getattr(settings, "MASTER_TARGET_TP", -1.0))
                            tl = float(getattr(settings, "MASTER_TARGET_LRA", 11.0))
                            mastered_p = master_fn(final_audio_path, I=ti, TP=tp, LRA=tl, ffmpeg_exe=ffmpeg_p)

                            if isinstance(mastered_p, Path) and mastered_p.exists():
                                if mastered_p.resolve() != final_audio_path.resolve():
                                    try:
                                        final_audio_path.unlink()
                                        mastered_p.rename(final_audio_path)
                                        _dlog("MASTERING_SUCCESS_REPLACED", f"Mastered: '{final_audio_path.name}'")
                                    except OSError as e:
                                        _dlog(
                                            f"[WARN] Failed replace after mastering: {e}. Kept at '{mastered_p.name}'.")
                                        final_audio_path = mastered_p
                                else:
                                    _dlog("MASTERING_SUCCESS_INPLACE", f"Mastered in-place: '{final_audio_path.name}'")
                        except Exception as e:
                            _dlog("MASTERING_ERROR", f"Error: {type(e).__name__}")
                except Exception as e:
                    _dlog(f"[ERROR] Ensuring WAV failed: {e}")
                    final_audio_path = None

        if isinstance(outputs_hist, dict):
            buffer_lls_after: List[str] = []
            main_lyrics_node_output = outputs_hist.get("74", {})
            if main_lyrics_node_output:
                for key_lls_res in ("text", "result", "string", "strings"):
                    value_lls_res = main_lyrics_node_output.get(key_lls_res)
                    if isinstance(value_lls_res, str) and value_lls_res.strip():
                        buffer_lls_after.append(value_lls_res.strip())
                        break
                    elif isinstance(value_lls_res, list):
                        found = False
                        for item in value_lls_res:
                            if isinstance(item, str) and item.strip():
                                buffer_lls_after.append(item.strip())
                                found = True
                        if found:
                            break

            if not buffer_lls_after:
                preview_node_output = outputs_hist.get("75", {})
                if isinstance(preview_node_output, dict):
                    for key_preview in ("text", "previews"):
                        value_preview = preview_node_output.get(key_preview)
                        if isinstance(value_preview, str) and value_preview.strip():
                            buffer_lls_after.append(value_preview.strip())
                            break
                        elif isinstance(value_preview, list):
                            found_preview = False
                            for item_preview in value_preview:
                                if isinstance(item_preview, str) and item_preview.strip():
                                    buffer_lls_after.append(item_preview.strip())
                                    found_preview = True
                            if found_preview:
                                break

            if buffer_lls_after:
                lls_after_result = "\n".join(buffer_lls_after).strip()
                if lls_after_result:
                    meta["lyrics_lls_after"] = lls_after_result
                    _dlog("LLS_AFTER_CAPTURE", f"Captured Node 74/75 output: {len(lls_after_result)} chars")

        if isinstance(final_audio_path, Path) and final_audio_path.exists() and final_audio_path.stat().st_size > 0:
            meta.setdefault("paths", {})["vocal"] = str(final_audio_path)
            meta["audio"] = str(final_audio_path)
        else:
            final_audio_path = None

        comfy_debug_section = meta.setdefault("comfy_debug", {})
        comfy_debug_section.update({"host": base_host, "prompt_json": str(main_wf_path), "prompt_seconds": seconds_val,
                                    "requested_format": "wav", "requested_ext": output_ext,
                                    "subfolder": subfolder_path})
        meta["tags_effective"] = {"positive": positive_tags, "negative": negative_tags}

        try:
            save_json(pj, meta)
            _dlog("META_SAVE_FINAL", f"Final project.json saved: '{pj.name}'")
        except Exception as e:
            _dlog(f"[ERROR] Failed save final pj: {e}")

        if isinstance(final_audio_path, Path) and final_audio_path.exists():
            try:
                library_path = save_to_user_library("audio", title, final_audio_path, rename=True)
                if isinstance(library_path, Path) and library_path.exists():
                    library_saved_path = library_path
            except Exception as e:
                _dlog("LIBRARY_COPY_ERROR", f"Failed: {e}")

    except Exception as main_execution_err:
        _dlog(f"[FATAL] Music generation failed: {main_execution_err}")
        notify("error", error=f"음악 생성 실패: {main_execution_err}")
        summary_final = f"ACE-Step 실패 ❌\n- 오류: {main_execution_err}"
        _dlog("LEAVE", summary_final.replace("\n", " | "))
        return summary_final

    # --- 최종 요약 ---
    wf_name = Path(main_wf_path).name if main_wf_path else "Unknown WF"
    message_lines: List[str] = [f"ACE-Step 완료 ✅", f"- 프롬프트: {wf_name}", f"- 길이: {seconds_val}s",
                                f"- 태그: +{len(positive_tags)} / -{len(negative_tags)}"]

    if isinstance(final_audio_path, Path) and final_audio_path.exists():
        message_lines.append(f"- 저장:     '{final_audio_path.name}'")
    else:
        message_lines.append("- 저장:     (오류 또는 파일 없음)")

    if isinstance(library_saved_path, Path) and library_saved_path.exists():
        message_lines.append(f"- 라이브러리: '{library_saved_path.name}'")

    summary_final = "\n".join(message_lines)
    _dlog("LEAVE", summary_final.replace("\n", " | "))
    notify("finished", summary=summary_final)
    return summary_final




# real use
def _rand_seed() -> int:
    return random.randint(1, 2_147_483_646)



# ───────────────────────────── 결과 파일 처리 ─────────────────────────────────
# real use
def _download_output_file(base: str, filename: str, subfolder: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    _dlog("TRY-DOWNLOAD", f"filename={filename}", f"subfolder={subfolder}", f"out_dir={out_dir}")

    combos = [
        ("audio", subfolder or ""),
        ("output", subfolder or ""),
        ("audio", ""),   # subfolder 비움
        ("output", ""),  # subfolder 비움
    ]

    last_err = None
    for t, sf in combos:
        try:
            r = _http_get(base, "/view", params={
                "filename": filename,
                "subfolder": sf,
                "type": t
            }, timeout=20)
            _dlog("VIEW-RESP", f"type={t}", f"sf='{sf}'", f"status={r.status_code}", f"bytes={len(r.content) if r.ok else 0}")
            if r.ok and r.content:
                target = out_dir / filename
                with open(target, "wb") as f:
                    f.write(r.content)
                _dlog("DOWNLOADED", str(target), f"size={target.stat().st_size}")
                return target
        except Exception as e:
            last_err = e
            _dlog("VIEW-TRY-FAIL", f"type={t}", f"sf='{sf}'", f"{type(e).__name__}: {e}")

    if last_err:
        raise last_err
    raise RuntimeError(f"다운로드 실패: filename={filename}, subfolder='{subfolder}'")

# ───────────────────────────── 메인 함수 ──────────────────────────────────────

# real use
def _ensure_vocal_wav(src_path: Path, proj_dir: Path, ffmpeg_exe: str = "ffmpeg") -> Path:
    """
    입력이 wav가 아니면 wav(16-bit PCM, 44.1kHz, 2ch)로 트랜스코딩하여
    proj_dir/vocal.wav 로 저장. 입력이 이미 wav면 파일 이동/복사로 통일.
    """
    proj_dir.mkdir(parents=True, exist_ok=True)
    out_wav = proj_dir / "vocal.wav"

    if src_path.suffix.lower() == ".wav":
        if src_path.resolve() != out_wav.resolve():
            try:
                if out_wav.exists():
                    out_wav.unlink()
            except Exception:
                pass
            src_path.replace(out_wav)
        return out_wav

    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(src_path),
        "-ac", "2",
        "-ar", "44100",
        "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return out_wav




###################################################################
#######################테스트중##################################
###################################################################

# real_use
def sync_lyrics_with_whisper_pro(
        audio_path: str,
        lyrics_text: str,
        *,
        model_size: str = "large-v3",
        beam_size: int = 5
) -> dict:
    """
    [최종 수정] 올바른 후처리 함수 이름(_finalize_audio_and_update_time)을 호출하도록 수정합니다.
    """

    # ... (slp_to_float, slp_round3 등 다른 내부 유틸 함수들은 변경 없음) ...
    def slp_to_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    def slp_round3(val):
        try:
            return round(float(val), 3)
        except (TypeError, ValueError):
            return 0.0

    def slp_clean_lines(slp_text_in: str) -> list[str]:
        slp_out_lines: list[str] = []
        if isinstance(slp_text_in, str) and slp_text_in:
            for slp_each_line in slp_text_in.splitlines():
                slp_each_trimmed = slp_each_line.strip()
                if slp_each_trimmed:
                    slp_out_lines.append(slp_each_trimmed)
        return slp_out_lines

    def slp_normalize_list(raw_list: list) -> list[dict]:
        slp_norm_list: list[dict] = []
        if not isinstance(raw_list, list): return slp_norm_list
        for slp_item_local in raw_list:
            if isinstance(slp_item_local, dict):
                slp_start_val, slp_end_val, slp_text_val = slp_to_float(slp_item_local.get("start", 0.0)), slp_to_float(
                    slp_item_local.get("end", 0.0)), str(slp_item_local.get("text", "") or "")
            else:
                try:
                    slp_start_val, slp_end_val, slp_text_val = slp_to_float(slp_item_local[0]), slp_to_float(
                        slp_item_local[1]), str(slp_item_local[2] if len(slp_item_local) > 2 else "")
                except (TypeError, IndexError):
                    continue
            if slp_end_val < slp_start_val: slp_start_val, slp_end_val = slp_end_val, slp_start_val
            if slp_end_val == slp_start_val: continue
            slp_norm_list.append({"start": slp_start_val, "end": slp_end_val, "text": slp_text_val})
        return slp_norm_list

    def slp_is_punct_only(text):
        if not text: return True
        return not "".join(ch for ch in text if ch.isalnum())

    def slp_energy_bounds(path_obj):
        try:
            import soundfile as _sf_slp
            import numpy as _np_slp
            slp_data, slp_sr = _sf_slp.read(str(path_obj))
            if slp_sr <= 0: return 0.0, math.inf
            slp_x = _np_slp.asarray(slp_data, dtype=_np_slp.float32)
            if slp_x.ndim == 2: slp_x = slp_x.mean(axis=1)
            slp_win, slp_hop, slp_rms = max(1, int(slp_sr * 0.05)), max(1, int(slp_sr * 0.02)), []
            current_idx, slp_len = 0, slp_x.shape[0]
            while current_idx + slp_win <= slp_len:
                segment = slp_x[current_idx:current_idx + slp_win]
                sum_of_squares = _np_slp.sum(segment ** 2)
                segment_length = len(segment)
                mean_of_squares = sum_of_squares / segment_length if segment_length > 0 else 0.0
                value_to_sqrt = mean_of_squares + 1e-9
                rms_value = _np_slp.sqrt(value_to_sqrt)
                slp_rms.append(float(rms_value))
                current_idx += slp_hop
            if not slp_rms: return 0.0, math.inf
            slp_thr = _np_slp.percentile(slp_rms, 35.0)
            slp_first = next((i_r for i_r, v_r in enumerate(slp_rms) if v_r > slp_thr), 0)
            slp_last = next((j_r for j_r in range(len(slp_rms) - 1, -1, -1) if slp_rms[j_r] > slp_thr),
                            len(slp_rms) - 1)
            slp_start_sec, slp_end_sec = max(0.0, (slp_first * slp_hop) / float(slp_sr) - 0.5), ((
                                                                                                             slp_last * slp_hop) / float(
                slp_sr)) + 0.8
            return slp_start_sec, slp_end_sec if slp_end_sec > slp_start_sec else math.inf
        except (ImportError, RuntimeError, OSError, ValueError):
            print("[WARN] Failed to calculate energy bounds. Proceeding without boundary filtering.")
            return 0.0, math.inf

    slp_audio_path = Path(audio_path)
    slp_project_dir = slp_audio_path.parent
    slp_seg_ready_path = slp_project_dir / "seg_ready.json"
    slp_seg_json_path = slp_project_dir / "seg.json"
    slp_lyrics_lines = slp_clean_lines(lyrics_text)

    slp_lyrics_compare = []
    try:
        slp_pj_path = slp_project_dir / "project.json"
        if slp_pj_path.exists():
            with open(slp_pj_path, "r", encoding="utf-8") as f:
                slp_meta = json.load(f)
            slp_compare_data = slp_meta.get("lyrics_compare")
            if isinstance(slp_compare_data, list):
                slp_lyrics_compare = [str(line) for line in slp_compare_data if str(line).strip()]
                if slp_lyrics_compare:
                    print("[SYNC-PRO] project.json의 'lyrics_compare'를 기준으로 환각을 제거합니다.")
    except (json.JSONDecodeError, OSError, KeyError, FileNotFoundError) as e:
        print(f"[WARN] project.json 로드 실패 ({type(e).__name__}), 원본 가사 텍스트를 사용합니다.")

    slp_transcribe_target_path = str(slp_audio_path)
    try:
        slp_vocal_stem_path = separate_vocals_demucs(str(slp_audio_path))
        if slp_vocal_stem_path and os.path.exists(slp_vocal_stem_path):
            slp_transcribe_target_path = slp_vocal_stem_path
            print(f"[SYNC-PRO] Demucs 보컬 스템을 분석 대상으로 사용합니다: {Path(slp_vocal_stem_path).name}")
    except NameError:
        print("[INFO] separate_vocals_demucs 함수를 찾을 수 없어 원본 오디오로 분석합니다.")
    except Exception as e:
        print(f"[WARN] Demucs 실행 중 오류 발생 ({type(e).__name__}), 원본 오디오로 분석합니다.")

    slp_tr_ret = transcribe_words(path=slp_transcribe_target_path, model=model_size, beam_size=beam_size,
                                  vad_filter=True)
    slp_segments_raw = slp_tr_ret.get("segments", [])
    slp_segments_norm = slp_normalize_list(slp_segments_raw)

    slp_begin_sec, slp_valid_end_sec = slp_energy_bounds(slp_audio_path)
    slp_filtered = [
        {"start": max(slp_to_float(s.get("start")), slp_begin_sec),
         "end": min(slp_to_float(s.get("end")), slp_valid_end_sec),
         "text": str(s.get("text", ""))}
        for s in slp_segments_norm
        if (isinstance(s, dict) and
            slp_to_float(s.get("end", 0.0)) > slp_to_float(s.get("start", 0.0)) and
            slp_to_float(s.get("start", 0.0)) < slp_valid_end_sec and
            slp_to_float(s.get("end", 0.0)) > slp_begin_sec and
            not slp_is_punct_only(str(s.get("text", ""))))
    ]

    try:
        with open(slp_seg_ready_path, "w", encoding="utf-8") as f:
            json.dump(slp_filtered, f, ensure_ascii=False, indent=2)
        print(f"[SYNC-PRO] seg_ready.json 저장됨: {slp_seg_ready_path}")
    except OSError as e:
        print(f"[WARN] seg_ready.json 저장 실패: {e}")

    slp_intermediate_final = _create_final_segments_from_ready(
        seg_ready_payload=slp_filtered,
        clean_lyrics_lines=slp_lyrics_lines,
        lyrics_compare_lines=slp_lyrics_compare,
        project_dir=str(slp_project_dir)
    )

    # [핵심 수정] 올바른 함수 이름으로 호출하고, 반환값을 올바르게 처리합니다.
    print("\n--- [FINAL ORGANIZING PROCESS] ---")
    slp_final_organized = _finalize_audio_and_update_time(  # <-- 이름 수정됨
        final_segments=slp_intermediate_final,
        project_dir=str(slp_project_dir),
        audio_path=str(slp_audio_path)
    )

    try:
        with open(slp_seg_json_path, "w", encoding="utf-8") as f:
            json.dump(slp_final_organized, f, ensure_ascii=False, indent=2)
        print(f"[SYNC-PRO] 최종 seg.json 저장됨: {slp_seg_json_path} ({len(slp_final_organized)}줄)")
    except OSError as e:
        print(f"[WARN] 최종 seg.json 저장 실패: {e}")

    final_audio_duration_after_processing = get_duration(str(slp_audio_path))
    summary_lines = [
        f"\n[음악분석 결과]",
        f"파일: {slp_audio_path.name}",
        f"최종 오디오 길이: {slp_round3(final_audio_duration_after_processing)}s",
        f"최종 유효 세그먼트: {len(slp_final_organized)}개 (원본 가사 기준)\n",
        "=== 최종 줄별 정합 (seg.json) ==="
    ]
    for idx, seg in enumerate(slp_final_organized, start=1):
        summary_lines.append(
            f"[{idx:02d}] {seg.get('start', 0.0):5.2f}~{seg.get('end', 0.0):6.2f}  {str(seg.get('text', '') or '')}")
    summary_text = "\n".join(summary_lines)
    print(summary_text)

    return {
        "seg_ready_path": str(slp_seg_ready_path),
        "seg_json_path": str(slp_seg_json_path),
        "segments": slp_filtered,
        "final_segments": slp_final_organized,
        "summary_text": summary_text
    }


# real use
def normalize_times_in_text(text: str, ref_time_map: dict) -> str:
    """
    ASR 결과 한 줄에서 시간 표현만 찾아,
    ref_time_map(캐논 (h,m,s) -> 정답 표기)으로 보수적으로 치환한다.
    """

    def to_int_from_k_num(token: str) -> Optional[int]:
        token = token.strip()
        special = {'한': 1, '두': 2, '세': 3, '네': 4}
        if token in special:
            return special[token]
        digit = {'영': 0, '공': 0, '일': 1, '이': 2, '삼': 3, '사': 4, '오': 5,
                 '육': 6, '칠': 7, '팔': 8, '구': 9}
        if token.isdigit():
            return int(token)
        if '십' in token:
            parts = token.split('십')
            tens = digit.get(parts[0], 1) if parts[0] else 1
            ones = digit.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
            return tens * 10 + ones
        if len(token) == 1 and token in digit:
            return digit[token]
        return None

    # --- 1) HH:MM(:SS?) / M:SS 탐지 후 치환 ---
    colon_rx = re.compile(r'(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?(?!\d)')
    def repl_colon(m: re.Match) -> str:
        h_or_m = int(m.group(1))
        mm = int(m.group(2))
        ss = int(m.group(3)) if m.group(3) is not None else None
        key_3 = (h_or_m, mm, ss)
        key_2 = (h_or_m, mm, None)
        # 우선 3-튜플, 없으면 2-튜플
        if key_3 in ref_time_map:
            return ref_time_map[key_3]
        if key_2 in ref_time_map:
            return ref_time_map[key_2]
        return m.group(0)

    text_after_colon = colon_rx.sub(repl_colon, text)

    # --- 2) (숫자|한글수사) + (시|분|초) 연쇄 탐지 후 묶음 단위로 치환 ---
    # 예: '3시 5분 10초', '세 시 5분', '15시'
    num_unit_rx = re.compile(
    r'(?P<num>\d{1,2}|[영공일이삼사오육칠팔구]+|한|두|세|네)\s*(?P<unit>[시분초])'
)
    pairs = [(m.group('num'), m.group('unit'), m.span()) for m in num_unit_rx.finditer(text_after_colon)]
    if not pairs:
        return text_after_colon

    # 연속 구간으로 묶기(시/분/초가 연달아 나오면 하나의 덩어리로)
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(pairs):
        start_i = pairs[i][2][0]
        end_i = pairs[i][2][1]
        j = i + 1
        while j < len(pairs) and pairs[j][2][0] == end_i:
            end_i = pairs[j][2][1]
            j += 1
        spans.append((start_i, end_i))
        i = j

    # 뒤에서부터 치환해야 인덱스 안 어긋남
    out = text_after_colon
    for span_start, span_end in reversed(spans):
        chunk = out[span_start:span_end]
        # 이 chunk 내부에서 다시 시/분/초 값 추출하여 (h,m,s)로 캐논키 생성
        sub_pairs = list(num_unit_rx.finditer(chunk))
        h = m_val = s_val = None
        for sm in sub_pairs:
            num_raw = sm.group('num')
            unit = sm.group('unit')
            if num_raw.isdigit():
                val = int(num_raw)
            else:
                val = to_int_from_k_num(num_raw)
            if unit == '시':
                h = val
            elif unit == '분':
                m_val = val
            elif unit == '초':
                s_val = val
        key = (h, m_val, s_val)
        if key in ref_time_map and h is not None:
            out = out[:span_start] + ref_time_map[key] + out[span_end:]
        # ref에 없으면 그대로 둔다(보수적)

    return out


# real_use
def _create_final_segments_from_ready(
        seg_ready_payload: list,
        clean_lyrics_lines: list,  # 헤더 포함된 원본 가사 (입력)
        lyrics_compare_lines: list,  # 헤더 포함된 비교용 가사 (입력)
        project_dir: str
) -> list:
    """
    [헤더 제거 최종 수정 v5.1 + 안전한 시간표현 통일 주입 + 이웃기반 영단어 복원(내부 로컬) + 참조 bigram 문맥 가중치]
    - Boundary detection uses lyric lines *without* headers.
    - Word correction reference(`ref_all_words`)는 헤더 제거본 사용.
    - [수정] 환각 가사 제거를 위해 원본 가사 줄 수(N)를 초과하는 세그먼트를 강제 절단하며,
             시간 조정 후 '원본 단어 검증' 클린업 로직을 추가합니다.
    """


    # ---- 시간표현 통일 헬퍼 바인딩(있으면만 사용, 임포트 시도 안 함) ----
    build_time_variant_map_local = globals().get("build_time_variant_map")
    normalize_times_in_text_local = globals().get("normalize_times_in_text")

    time_map_local = None
    if build_time_variant_map_local:
        try:
            time_map_local = build_time_variant_map_local(clean_lyrics_lines)
        except Exception:
            time_map_local = None

    def _maybe_norm_time(s: str) -> str:
        if time_map_local and normalize_times_in_text_local:
            try:
                return normalize_times_in_text_local(s, time_map_local)
            except Exception:
                return s
        return s

    # --- constants and local utils (확인되지 않은 참조 해결) ---
    _match_type_kept_local = "KEPT"
    _match_type_authenticated_local = "AUTHENTICATED"
    _match_type_corrected_local = "CORRECTED"
    _filter_short_words = {'a', 'ah', 'oh', 'uh', 'hm', 'hmm', 'um', 'o', 'eh', '음', 'nan', 'neo'}

    def _safe_float(val: Any) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    def _norm_for_compare(text: str) -> str:
        s = str(text or "").lower().strip()
        s = re.sub(r"[^a-z0-9가-힣\s']+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _match_score(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(None, a, b).ratio()

    # kroman 처리 (ImportError 시 None 할당 및 except 블록에서도 정의)
    kroman_local: Optional[Any]
    try:
        import kroman  # type: ignore
        kroman_local = kroman
    except ImportError:
        kroman_local = None
        print("[WARN] kroman library not found. Romanization comparison will be limited.")

    _korean_pattern_local: Pattern[str] = re.compile(r'[\uac00-\ud7a3]')

    def _is_korean(word: str) -> bool:
        return bool(_korean_pattern_local.search(str(word or "")))

    def _romanize(text: str) -> str:
        text_norm = _norm_for_compare(text)
        if not text_norm:
            return ""
        korean_parts = _korean_pattern_local.findall(text)
        if korean_parts and kroman_local:
            try:
                korean_text_only = "".join(korean_parts)
                parsed: Optional[str] = kroman_local.parse(korean_text_only)  # type: ignore[attr-defined]
                return parsed.replace("-", "").lower() if parsed else text_norm
            except Exception as e_kroman_parse:
                print(f"[WARN] kroman parsing failed for '{text}': {type(e_kroman_parse).__name__}")
                return text_norm
        return text_norm

    # ---- 1. Prepare Reference Lines & Boundary Detection ----
    print("\n--- [BOUNDARY DETECTION] ---")
    correction_ref_lines_with_headers = lyrics_compare_lines if lyrics_compare_lines else clean_lyrics_lines
    if not correction_ref_lines_with_headers or not seg_ready_payload:
        print("[ERROR] Reference lyrics (with headers) or ready segments are empty.")
        return []

    # 헤더 제외한 가사 목록
    boundary_ref_lines_no_headers: List[str] = []
    header_pattern = re.compile(r"^\s*\[[^]]+]\s*$")
    for line_with_header in correction_ref_lines_with_headers:
        if not header_pattern.match(line_with_header):
            boundary_ref_lines_no_headers.append(line_with_header)

    if not boundary_ref_lines_no_headers:
        print("[ERROR] No actual lyric lines found after removing headers.")
        boundary_ref_lines_no_headers = correction_ref_lines_with_headers
        if not boundary_ref_lines_no_headers:
            return []
        print("[WARN] Fallback: Using lines *with* headers for boundary detection.")

    # 시간표현 통일(있으면만)
    boundary_ref_lines_no_headers = [_maybe_norm_time(x) for x in boundary_ref_lines_no_headers]

    # 경계 탐색용 첫/마지막 줄
    first_line_raw = boundary_ref_lines_no_headers[0] if boundary_ref_lines_no_headers else ""
    last_line_raw = boundary_ref_lines_no_headers[-1] if boundary_ref_lines_no_headers else ""
    first_line_norm = _norm_for_compare(first_line_raw)
    last_line_norm = _norm_for_compare(last_line_raw)
    print(f"[DEBUG] Target First Line (Raw, No Header): '{first_line_raw}'")
    print(f"[DEBUG] Target First Line (Norm, No Header): '{first_line_norm}'")
    print(f"[DEBUG] Target Last Line (Raw, No Header): '{last_line_raw}'")
    print(f"[DEBUG] Target Last Line (Norm, No Header): '{last_line_norm}'")

    first_idx, last_idx = -1, -1
    score_threshold_boundary = 0.25

    # 첫 줄 탐색
    print("\n[DEBUG] Searching for First Line Match...")
    if first_line_norm:
        for i, seg_item in enumerate(seg_ready_payload):
            if isinstance(seg_item, dict) and "text" in seg_item:
                seg_text_raw = _maybe_norm_time(str(seg_item.get("text", "")))
                seg_text_norm = _norm_for_compare(seg_text_raw)
                score_val = _match_score(seg_text_norm, first_line_norm)
                print(
                    f"  [DEBUG] Comparing seg[{i}] ('{seg_text_norm}') vs First ('{first_line_norm}') -> Score: {score_val:.4f}")
                if score_val > score_threshold_boundary:
                    first_idx = i
                    print(f"    => Match found! Setting first_idx = {i}")
                    break
    else:
        first_idx = 0
        print("[WARN] Target first line empty, setting first_idx = 0.")

    # 마지막 줄 탐색(역방향)
    print("\n[DEBUG] Searching for Last Line Match (reverse)...")
    if last_line_norm:
        payload_len = len(seg_ready_payload)
        for i_rev in range(payload_len - 1, -1, -1):
            current_seg_item = seg_ready_payload[i_rev]
            if isinstance(current_seg_item, dict) and "text" in current_seg_item:
                seg_text_norm_rev = _norm_for_compare(_maybe_norm_time(current_seg_item.get("text", "")))
                score_val_rev = _match_score(seg_text_norm_rev, last_line_norm)
                print(
                    f"  [DEBUG] Comparing seg[{i_rev}] ('{seg_text_norm_rev}') vs Last ('{last_line_norm}') -> Score: {score_val_rev:.4f}")
                if score_val_rev > score_threshold_boundary:
                    last_idx = i_rev
                    print(f"    => Match found! Setting last_idx = {i_rev}")
                    break
    else:
        last_idx = len(seg_ready_payload) - 1 if seg_ready_payload else -1
        print("[WARN] Target last line empty, setting last_idx = end.")

    print(f"\n[DEBUG] Final Boundary Indices: first_idx = {first_idx}, last_idx = {last_idx}")

    # seg_compare_payload 생성
    seg_compare_payload: list = []
    if first_idx != -1 and last_idx != -1 and last_idx >= first_idx:
        seg_compare_payload = seg_ready_payload[first_idx: last_idx + 1]
        print(f"[INFO] Slicing seg_ready from {first_idx} to {last_idx + 1}.")
    elif first_idx != -1:
        seg_compare_payload = seg_ready_payload[first_idx:]
        print(f"[WARN] Last line fail. Slicing from {first_idx} to end.")
    elif last_idx != -1:
        seg_compare_payload = seg_ready_payload[: last_idx + 1]
        print(f"[WARN] First line fail. Slicing from start to {last_idx + 1}.")
    else:
        seg_compare_payload = seg_ready_payload
        print("[WARN] Both boundary fails. Using all segments.")
    print(f"       => Resulting seg_compare_payload length: {len(seg_compare_payload)}")

    if not seg_compare_payload:
        print("[ERROR] No segments selected. Cannot proceed.")
        return []

    # seg_compare.json 저장
    try:
        compare_path = Path(project_dir) / "seg_compare.json"

        save_json(compare_path, seg_compare_payload)
        print(f"[SYNC-PRO] seg_compare.json saved: {compare_path} ({len(seg_compare_payload)} segments)")
    except Exception as e_save_compare:
        print(f"[WARN] Failed to save seg_compare.json: {type(e_save_compare).__name__}")

    # ---- 2 & 3. Word Correction 준비 (헤더 제거본 + 시간통일 적용) ----
    ref_source_lines = boundary_ref_lines_no_headers
    ref_source_lines = [_maybe_norm_time(x) for x in boundary_ref_lines_no_headers]

    # === 앵커 기반 영단어 복원용 맵 ===
    anchor_map: Dict[str, str] = {}
    rx_token = re.compile(r"[A-Za-z]+|[가-힣]+|\d+")
    for line_ref in ref_source_lines:
        toks = rx_token.findall(line_ref or "")
        for i_tok in range(len(toks)):
            anchor_tok = toks[i_tok]
            if not anchor_tok:
                continue
            eng_seq: List[str] = []
            j = i_tok + 1
            while j < len(toks) and len(eng_seq) < 2:
                t = toks[j]
                if re.fullmatch(r"[A-Za-z]+", t):
                    eng_seq.append(t.lower())
                    j += 1
                    continue
                break
            if eng_seq:
                key_anchor = anchor_tok.strip()
                if _is_korean(key_anchor) or re.search(r"\d", key_anchor):
                    phrase = " ".join(eng_seq).strip()
                    if phrase:
                        prev = anchor_map.get(key_anchor)
                        if not prev or len(phrase) > len(prev):
                            anchor_map[key_anchor] = phrase

    def _apply_bilingual_hint(text: str, anchors: Dict[str, str]) -> str:
        """
        세그먼트 한 줄에서, '앵커 다음에 붙은 한글 오인식 토큰 1개'를
        참조의 영단어 구절로 보수적으로 치환한다.
        예) '푸른 스크린라이트' -> '푸른 screen light'
        """
        if not text or not anchors:
            return text
        out = str(text)
        for anchor_key in anchors.keys():
            eng_phrase = anchors.get(anchor_key)
            pat = re.compile(r"(?:(?<=^)|(?<=\s))" + re.escape(anchor_key) + r"\s+([가-힣]{2,15})(?=\s|$)")

            def _repl(m: re.Match) -> str:
                return f"{anchor_key} {eng_phrase}"

            out_new = pat.sub(_repl, out)
            out = out_new
        return out

    # === bigram 문맥 맵 (ref 라인 순서 기준) ===
    bigram_counts: Dict[str, Dict[str, int]] = {}
    for line_ref in ref_source_lines:
        toks = rx_token.findall(line_ref or "")
        toks_l = [t.lower() for t in toks if t]
        for i_b in range(len(toks_l) - 1):
            a_token = toks_l[i_b]
            b_token = toks_l[i_b + 1]
            if a_token not in bigram_counts:
                bigram_counts[a_token] = {}
            if b_token not in bigram_counts[a_token]:
                bigram_counts[a_token][b_token] = 0
            bigram_counts[a_token][b_token] += 1

    top_next_map: Dict[str, str] = {}
    for k_prev_token, nxts in bigram_counts.items():
        best_next_token = ""
        best_cnt = -1
        for cand_nxt, cnt in nxts.items():
            if cnt > best_cnt:
                best_cnt = cnt
                best_next_token = cand_nxt
        if best_next_token:
            top_next_map[k_prev_token] = best_next_token

    ref_all_words_raw = " ".join(ref_source_lines).replace(",", "").replace("'", "")
    ref_all_words = ref_all_words_raw.split()
    ref_all_words = [w for w in ref_all_words if w]
    if not ref_all_words:
        print("[ERROR] Reference words list (no headers) is empty.")
        return []
    print(f"[DEBUG] Using {len(ref_all_words)} reference words for correction (headers excluded).")
    ref_words_norm = [_norm_for_compare(w) for w in ref_all_words]
    ref_words_roman = [_romanize(w) for w in ref_all_words]

    intermediate_segments: List[Dict] = []
    print("\n--- [SEGMENT CORRECTION PROCESS] ---")
    for seg_idx, seg_corr in enumerate(seg_compare_payload):
        if not isinstance(seg_corr, dict) or "text" not in seg_corr:
            continue
        whisper_text_corr = str(seg_corr.get("text", "")).strip()
        whisper_text_corr = _maybe_norm_time(whisper_text_corr)

        # 앵커 기반 영단어 복원(보수적, 전처리 1회)
        whisper_text_corr = _apply_bilingual_hint(whisper_text_corr, anchor_map)

        if not whisper_text_corr:
            continue

        segment_corrected_words: List[str] = []
        whisper_words_list = whisper_text_corr.split()
        whisper_cursor = 0
        leftover_whisper_part = ""

        while whisper_cursor < len(whisper_words_list):
            candidate_whisper_raw = leftover_whisper_part + whisper_words_list[whisper_cursor]
            candidate_whisper = candidate_whisper_raw.strip()
            candidate_norm = _norm_for_compare(candidate_whisper)
            if not candidate_norm:
                whisper_cursor += 1
                leftover_whisper_part = ""
                continue

            best_match: Dict[str, Any] = {
                "score": -1.0,
                "word": candidate_whisper,
                "type": _match_type_kept_local,
                "ref_origin": ""
            }

            # 직전 확정 단어 기반 문맥 보너스
            prev_out_lower = segment_corrected_words[-1].lower() if segment_corrected_words else ""

            for i_ref, loop_ref_word in enumerate(ref_all_words):
                ref_norm_loop = ref_words_norm[i_ref]
                ref_roman_loop = ref_words_roman[i_ref]
                if not ref_norm_loop:
                    continue

                auth_score = -1.0
                if candidate_norm and ref_norm_loop and candidate_norm in ref_norm_loop:
                    auth_score = 0.85 + (len(candidate_norm) / len(ref_norm_loop)) * 0.15

                spell_score = _match_score(candidate_norm, ref_norm_loop)
                roman_score = 0.0
                candidate_roman = _romanize(candidate_whisper)
                if candidate_roman and ref_roman_loop:
                    roman_score = _match_score(candidate_roman, ref_roman_loop)
                elif not _is_korean(candidate_whisper) and ref_roman_loop:
                    roman_score = _match_score(candidate_norm, ref_roman_loop) * 0.9

                # 문맥 보너스 (ref bigram 기반)
                context_bonus = 0.0
                loop_ref_lower = loop_ref_word.lower()
                if prev_out_lower and prev_out_lower in bigram_counts:
                    if loop_ref_lower in bigram_counts.get(prev_out_lower, {}):
                        context_bonus += 0.18
                        if top_next_map.get(prev_out_lower) == loop_ref_lower:
                            context_bonus += 0.05

                correction_base = spell_score if spell_score >= roman_score else roman_score
                correction_score = correction_base + context_bonus
                current_best_score = best_match["score"]

                if auth_score > correction_score and auth_score > current_best_score:
                    best_match.update({
                        "score": auth_score,
                        "word": candidate_whisper,
                        "type": _match_type_authenticated_local,
                        "ref_origin": loop_ref_word
                    })
                elif correction_score > current_best_score:
                    is_candidate_korean = _is_korean(candidate_whisper)
                    is_ref_single_eng = len(loop_ref_word) == 1 and 'a' <= loop_ref_word.lower() <= 'z'
                    if not (is_candidate_korean and is_ref_single_eng):
                        best_match.update({
                            "score": correction_score,
                            "word": loop_ref_word,
                            "type": _match_type_corrected_local,
                            "ref_origin": loop_ref_word
                        })

            match_type = best_match["type"]
            word_to_process = best_match["word"]
            best_score = best_match["score"]
            ref_origin = best_match["ref_origin"]
            score_threshold = 0.50
            word_to_append: Optional[str] = None
            leftover_whisper_part = ""

            if match_type == _match_type_kept_local:
                if candidate_whisper.lower() not in _filter_short_words:
                    word_to_append = candidate_whisper
            else:
                if best_score >= score_threshold:
                    word_to_append = word_to_process
                    ref_matched_norm = _norm_for_compare(ref_origin)
                    candidate_actual_norm = _norm_for_compare(candidate_whisper)
                    if len(candidate_actual_norm) > len(ref_matched_norm) and candidate_actual_norm.startswith(
                            ref_matched_norm):
                        try:
                            match_obj = re.match(re.escape(ref_matched_norm), candidate_whisper,
                                                 flags=re.IGNORECASE | re.UNICODE)
                            if match_obj:
                                leftover_whisper_part = candidate_whisper[match_obj.end():].strip()
                        except re.error:
                            leftover_whisper_part = ""
                else:
                    if candidate_whisper.lower() not in _filter_short_words:
                        word_to_append = candidate_whisper

            if word_to_append:
                segment_corrected_words.append(word_to_append)

            # === 추가 보완: 부분 일치로 남은 꼬리를 즉시 다음 토큰으로 재처리 ===
            if leftover_whisper_part:
                whisper_words_list.insert(whisper_cursor + 1, leftover_whisper_part)
                leftover_whisper_part = ""

            whisper_cursor += 1

        corrected_text = " ".join(segment_corrected_words).strip()
        if corrected_text:
            intermediate_segments.append({
                "start": _safe_float(seg_corr.get("start")),
                "end": _safe_float(seg_corr.get("end")),
                "text": corrected_text,
                "line_ko": corrected_text
            })

    print("--- [SEGMENT CORRECTION PROCESS COMPLETED] ---")
    print(f"Total intermediate segments generated: {len(intermediate_segments)}")

    # ---- 4. Final Boundary Check & Return ----
    if not intermediate_segments:
        print("[ERROR] No segments survived correction.")
        return []

    final_start_idx, final_end_idx = -1, -1
    final_boundary_threshold = 0.45

    print("\n--- [FINAL BOUNDARY CHECK] ---")
    if first_line_norm:
        print(f"[DEBUG] Final check - Searching for First Line (Norm): '{first_line_norm}'")
        for i_final, seg_final in enumerate(intermediate_segments):
            score_final = _match_score(_norm_for_compare(seg_final.get("text", "")), first_line_norm)
            if score_final > final_boundary_threshold:
                final_start_idx = i_final
                print(f"    => Match found! final_start_idx = {i_final}")
                break
    else:
        final_start_idx = 0
        print("[WARN] Final check - Target first line empty, using index 0.")

    if last_line_norm:
        print(f"[DEBUG] Final check - Searching for Last Line (Norm, reverse): '{last_line_norm}'")
        inter_len = len(intermediate_segments)
        for i_final_rev in range(inter_len - 1, -1, -1):
            seg_final_rev = intermediate_segments[i_final_rev]
            score_final_rev = _match_score(_norm_for_compare(seg_final_rev.get("text", "")), last_line_norm)
            if score_final_rev > final_boundary_threshold:
                final_end_idx = i_final_rev
                print(f"    => Match found! final_end_idx = {i_final_rev}")
                break
    else:
        final_end_idx = len(intermediate_segments) - 1 if intermediate_segments else -1
        print("[WARN] Final check - Target last line empty, using last index.")

    print(f"[DEBUG] Final Boundary Indices after check: start={final_start_idx}, end={final_end_idx}")

    final_segments: List[Dict] = []
    if final_start_idx != -1 and final_end_idx != -1 and final_end_idx >= final_start_idx:
        final_segments = intermediate_segments[final_start_idx: final_end_idx + 1]
        print(
            f"[SYNC-PRO] Final check: Kept segments {final_start_idx} to {final_end_idx} (Total: {len(final_segments)}).")
    elif intermediate_segments:
        final_segments = intermediate_segments
        print("[WARN] Final check failed, using all intermediate.")
    else:
        print("[ERROR] Final check failed, no intermediate segments.")
        return []

    # -----------------------------------------------------
    # 💡 [핵심 추가 로직] 원본 가사 줄 수(N) 초과 강제 절단
    # -----------------------------------------------------
    # boundary_ref_lines_no_headers는 헤더가 제거된 원본 가사입니다.
    N_SOURCE_LINES = len(boundary_ref_lines_no_headers)

    if len(final_segments) > N_SOURCE_LINES:
        # 세그먼트가 원본 줄 수보다 많으면, 앞에서부터 N개만 남기고 잘라냅니다.
        print(
            f"[INFO] Segment count ({len(final_segments)}) exceeds source line count ({N_SOURCE_LINES}). Trimming to {N_SOURCE_LINES} segments.")
        final_segments = final_segments[:N_SOURCE_LINES]
    # -----------------------------------------------------

    # Time Adjustment
    final_segments.sort(key=lambda x: _safe_float(x.get("start", 0.0)))
    print("\n--- [TIME ADJUSTMENT] ---")
    for i_adjust in range(1, len(final_segments)):
        prev_end_adjust = _safe_float(final_segments[i_adjust - 1].get("end", 0.0))
        curr_start_adjust = _safe_float(final_segments[i_adjust].get("start", 0.0))
        if curr_start_adjust < prev_end_adjust - 0.01:
            final_segments[i_adjust]["start"] = prev_end_adjust
            curr_end_adjust = _safe_float(final_segments[i_adjust].get("end", 0.0))
            if curr_end_adjust < prev_end_adjust + 0.1:
                final_segments[i_adjust]["end"] = prev_end_adjust + 0.1

    # -----------------------------------------------------
    # 💡 [핵심 추가 로직] 최종 가사 후처리: 원본 가사에 없는 단어 제거
    # -----------------------------------------------------
    final_segments_clean: List[Dict] = []

    # 원본 가사에서 사용된 모든 단어(헤더 제외)를 찾기 위한 set 생성
    # ref_all_words는 이미 위에서 계산됨
    source_words_set = set(w.lower() for w in ref_all_words)

    for seg in final_segments:
        current_text = str(seg.get("text", "")).strip()

        # current_text를 단어 단위로 분해 (공백 및 쉼표 기준)
        current_words = [w for w in re.split(r"[\s,]+", current_text) if w]

        # 원본에 포함된 단어만 남깁니다.
        corrected_words_list = []
        for word in current_words:
            # 원본 가사 단어 집합에 포함되는 경우에만 추가
            if word.lower() in source_words_set:
                corrected_words_list.append(word)
            # 그렇지 않은 경우, 즉 "볼"과 같은 환각 단어는 무시

        new_text = " ".join(corrected_words_list).strip()

        if new_text and new_text != current_text:
            # 텍스트가 변경되었고 (예: "볼 복잡했던" -> "복잡했던") 새 텍스트가 비어있지 않은 경우 업데이트
            seg["text"] = new_text
            seg["line_ko"] = new_text  # line_ko도 함께 업데이트
            print(f"[CLEANUP] Corrected line: '{current_text}' -> '{new_text}'")
            final_segments_clean.append(seg)
        elif not new_text and current_text:
            # 모든 단어가 제거되었다면 (빈 줄)
            print(f"[CLEANUP] Line completely removed: '{current_text}'")
            continue  # 최종 목록에 추가하지 않음
        else:
            # 변경 없음 (new_text == current_text) 또는 new_text가 비어있지 않은 경우
            final_segments_clean.append(seg)

    # -----------------------------------------------------

    print(f"\n[SYNC-PRO] Returning {len(final_segments_clean)} final segments after cleanup.")
    print("------------------------------------\n")
    return final_segments_clean






# real_use
def transcribe_words(
        path: str,
        model_size: str | None = None,
        *,
        model: str | None = None,
        beam_size: int = 5,
        initial_prompt: str | None = None,
        language: str | None = None,
        print_translate_view: bool = True,
        vad_filter: bool = True,  # 이 인자는 외부 호환성을 위해 남겨둡니다.
        **kwargs,
) -> dict:
    """
    오디오 → 단어 단위 타임라인.
    - [최종 수정] VAD 충돌 문제를 해결하기 위해 vad_filter 옵션을 내부적으로 False로 강제합니다.
    - Linter 경고를 해결하기 위해 except 블록에서 변수를 할당합니다.
    """


    # 사용되지 않는 매개변수를 명시적으로 처리 (Linter 경고 방지)
    _ = initial_prompt

    def _fmt(t: float) -> str:
        return f"{t:06.3f}"

    mdl = (model_size or model or "medium")
    bs = int(beam_size) if isinstance(beam_size, int) else 5
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"audio file not found: {path}")

    # [핵심 수정] VAD 충돌을 막기 위해 vad_filter를 False로 강제합니다.
    use_vad = False
    print(
        f"[transcribe_words] start path='{p.name}', model='{mdl}', beam_size={bs}, vad_filter={use_vad} (Forced False)")

    segments_list: List[Tuple[float, float, str]] = []
    words_list: List[Tuple[float, float, str]] = []

    fw_allow = {"temperature", "beam_size", "patience", "length_penalty", "compression_ratio_threshold",
                "logprob_threshold", "no_speech_threshold", "condition_on_previous_text", "task", "vad_filter",
                "language", "best_of"}
    ow_allow = {"temperature", "beam_size", "patience", "length_penalty", "compression_ratio_threshold",
                "logprob_threshold", "no_speech_threshold", "condition_on_previous_text", "task", "language", "best_of"}

    try:
        import faster_whisper
    except ImportError:
        faster_whisper = None
        print("[INFO] faster_whisper not found, falling back to openai-whisper.")

    if faster_whisper:
        try:
            fw_opts: Dict[str, Any] = {"beam_size": bs, "word_timestamps": True}
            if isinstance(language, str) and language.strip():
                fw_opts["language"] = language.strip()
            for k, v in kwargs.items():
                if k in fw_allow:
                    fw_opts[k] = v

            # [핵심 수정] VAD 옵션을 강제로 끕니다.
            fw_opts["vad_filter"] = use_vad

            wmodel = faster_whisper.WhisperModel(mdl, device="cpu", compute_type="int8")
            seg_iter, _info = wmodel.transcribe(str(p), **fw_opts)

            for seg_idx, seg in enumerate(seg_iter, 1):
                a, b, txt = float(getattr(seg, "start", 0.0)), float(getattr(seg, "end", 0.0)), str(
                    getattr(seg, "text", "")).strip()
                segments_list.append((a, b, txt))
                print(f"[SEG {seg_idx:04d}] {_fmt(a)} ~ {_fmt(b)} | {txt}")
                for w in getattr(seg, "words", []):
                    try:
                        wa, wb, wt = float(w.start), float(w.end), str(w.word).strip()
                        if wt:
                            words_list.append((wa, wb, wt))
                            print(f"  [WORD {len(words_list):05d}] {_fmt(wa)} ~ {_fmt(wb)} | {wt}")
                    except (TypeError, ValueError, AttributeError):
                        continue
            print(f"[transcribe_words] ko-pass done: segments={len(segments_list)}, words={len(words_list)}")

        except Exception as e:
            print(f"[WARN] faster-whisper failed during execution: {type(e).__name__}: {e}", file=sys.stderr)
            segments_list, words_list = [], []

    if not segments_list:
        try:
            import whisper
            try:
                wmodel2 = whisper.load_model(mdl)
            except Exception as load_err:
                raise RuntimeError(f"Failed to load openai-whisper model '{mdl}'.") from load_err

            ow_opts: Dict[str, Any] = {"beam_size": bs, "word_timestamps": True, "fp16": False}
            if isinstance(language, str) and language.strip():
                ow_opts["language"] = language.strip()
            for k, v in kwargs.items():
                if k in ow_allow:
                    ow_opts[k] = v

            res = whisper.transcribe(wmodel2, str(p), **ow_opts)

            for seg_idx, seg in enumerate(res.get("segments", []), 1):
                a, b, txt = float(seg.get("start", 0.0)), float(seg.get("end", 0.0)), str(seg.get("text", "")).strip()
                segments_list.append((a, b, txt))
                print(f"[SEG {seg_idx:04d}] {_fmt(a)} ~ {_fmt(b)} | {txt}")
                for w in seg.get("words", []):
                    try:
                        wa, wb, wt = float(w.get("start", 0.0)), float(w.get("end", 0.0)), str(
                            w.get("word", "")).strip()
                        if wt:
                            words_list.append((wa, wb, wt))
                            print(f"  [WORD {len(words_list):05d}] {_fmt(wa)} ~ {_fmt(wb)} | {wt}")
                    except (TypeError, ValueError):
                        continue
            print(f"[transcribe_words] ko-pass done: segments={len(segments_list)}, words={len(words_list)}")

        except ImportError as e:
            raise RuntimeError("Both faster-whisper and openai-whisper are not installed.") from e
        except Exception as e:
            raise RuntimeError(f"openai-whisper transcribe failed: {type(e).__name__}: {e}") from e

    if print_translate_view and segments_list:
        try:
            print("[transcribe_words] translate-pass (en view) start")
            for en_idx, (a, b, txt) in enumerate(segments_list, 1):
                print(f"[EN-SEG {en_idx:04d}] {_fmt(a)} ~ {_fmt(b)} | {txt}")
            print(f"[transcribe_words] translate-pass done: segments={len(segments_list)}")
        except Exception as e:
            print(f"[WARN] Failed to print translate view: {type(e).__name__}: {e}", file=sys.stderr)

    return {"segments": segments_list, "words": words_list}




# ============================================================
# 최종 오디오 처리 및 시간 업데이트 함수 (모든 경고 해결 최종 버전)
# ============================================================
# real use
def _finalize_audio_and_update_time(
        final_segments: List[Dict[str, Any]],  # 타입 힌트 명시
        project_dir: str,
        audio_path: str
) -> List[Dict[str, Any]]:  # [수정] 처리된 세그먼트 리스트를 반환하도록 변경
    """
    [최종 수정] 모든 Linter 경고를 해결하고, 처리된 세그먼트 리스트를 반환합니다.
    """

    # ---- 지역 유틸 함수 정의 ----
    def _safe_float(val: Any) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    # ---- 1. 무의미한 의성어 제거 ----
    _FILTER_WORDS = {'a', 'ah', 'oh', 'uh', 'hm', 'hmm', 'um', 'o', 'eh', '음'}

    organized_segments: List[Dict[str, Any]] = []
    if not final_segments:
        print("[FINALIZE] No segments to process.")
    else:
        for seg in final_segments:
            text_to_check = str(seg.get("text", "")).lower().strip()
            if text_to_check in _FILTER_WORDS:
                print(f"  - [FILTERED] Segment removed (short interjection): '{text_to_check}'")
                continue
            organized_segments.append(seg)

    if not organized_segments:
        print("[WARN] All segments were filtered out.")
        return []  # 빈 리스트 반환

    # ---- 2. 타임스탬프 포맷팅 ----
    for seg in organized_segments:
        seg["start"] = round(_safe_float(seg.get("start", 0.0)), 2)
        seg["end"] = round(_safe_float(seg.get("end", 0.0)), 2)

    # ---- ffmpeg 경로 처리 ----
    ffmpeg_exe = "ffmpeg"
    try:

        if app_settings_local and hasattr(app_settings_local, "FFMPEG_EXE"):
            ffmpeg_exe = getattr(app_settings_local, "FFMPEG_EXE") or "ffmpeg"
        elif root_settings_local and hasattr(root_settings_local, "FFMPEG_EXE"):
            ffmpeg_exe = getattr(root_settings_local, "FFMPEG_EXE") or "ffmpeg"
    except Exception as e:
        print(f"[WARN] Error loading settings for FFMPEG_EXE: {e}. Using default.")

    # ---- 3. 오디오 처리 및 'time' 업데이트 ----
    last_lyric_end = _safe_float(organized_segments[-1].get("end", 0.0))
    target_duration = last_lyric_end + 5.0
    actual_duration = get_duration(audio_path)
    final_audio_duration = actual_duration if actual_duration > 0 else target_duration

    print(
        f"[FINALIZE] Last lyric end: {last_lyric_end:.2f}s, Target: {target_duration:.2f}s, Actual: {actual_duration:.2f}s")

    ffmpeg_cmd: list[str] = []
    if actual_duration > 0:
        if target_duration < actual_duration and not (abs(target_duration - actual_duration) < 0.5):
            print("[FINALIZE] Case A: Trimming audio and applying fade-out.")
            final_audio_duration = target_duration
            fade_start = max(0.0, target_duration - 2.0)
            ffmpeg_cmd = [ffmpeg_exe, "-y", "-i", str(audio_path), "-to", f"{target_duration:.3f}", "-af",
                          f"afade=t=out:st={fade_start:.3f}:d=2.0", "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2"]
        elif abs(target_duration - actual_duration) < 0.5:
            print("[FINALIZE] Case B: Applying fade-out only.")
            final_audio_duration = actual_duration
            fade_start = max(0.0, actual_duration - 2.0)
            ffmpeg_cmd = [ffmpeg_exe, "-y", "-i", str(audio_path), "-af", f"afade=t=out:st={fade_start:.3f}:d=2.0",
                          "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2"]
        else:
            print("[FINALIZE] Case C: No audio processing needed.")
            final_audio_duration = actual_duration

    if ffmpeg_cmd:
        temp_output_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=Path(audio_path).suffix or ".wav", delete=False) as temp_file:
                temp_output_path = temp_file.name
            ffmpeg_cmd.append(temp_output_path)
            print(f"[FINALIZE] Running FFmpeg: {' '.join(ffmpeg_cmd)}")
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')

            processed_path = Path(temp_output_path)
            if processed_path.exists() and processed_path.stat().st_size > 0:
                shutil.move(str(processed_path), str(audio_path))
                print(f"[FINALIZE] Audio file processed: {Path(audio_path).name}")
            else:
                if processed_path.exists(): processed_path.unlink()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
            error_output = getattr(e, 'stderr', str(e))
            print(f"[ERROR] FFmpeg/File operation failed. Original file kept. Error: {error_output.strip()}")
            if temp_output_path and Path(temp_output_path).exists(): Path(temp_output_path).unlink()

    # ---- 4. project.json 업데이트 ----
    try:
        pj_path = Path(project_dir) / "project.json"
        meta = {}
        if pj_path.exists():
            try:
                with open(pj_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError) as read_err:
                meta = {}
                print(f"[WARN] Failed to read project.json: {read_err}.")

        meta["lyrics_result"] = "\n".join([str(seg.get("text", "")) for seg in organized_segments])
        new_time_int = int(round(final_audio_duration))
        meta["time"] = new_time_int

        try:
            with open(pj_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"[SYNC-PRO] 'lyrics_result' and 'time' ({new_time_int}s) updated in project.json.")
        except OSError as write_err:
            print(f"[ERROR] Failed to write to project.json: {write_err}")

    except (KeyError, FileNotFoundError) as e:
        print(f"[WARN] Could not update project.json: {e}")

    return organized_segments  # [수정] 처리된 세그먼트 리스트 반환

# # real use
# def _submit_and_wait(
#     base: str,
#     wf_graph: dict,
#     timeout: int | float | None = None,
#     poll: float | None = None,
#     on_progress=None
# ) -> dict[str, Any]:
#     """
#     ComfyUI /prompt 제출 후 /history/<id>를 폴링하여 완료까지 대기.
#     반환: history의 해당 entry(dict[str, Any]) — outputs/status 포함
#     """
#
#
#     # 안전한 기본값 결정
#     timeout_val = float(timeout if timeout is not None else _ace_wait_timeout_sec())
#     poll_val = float(poll if poll is not None else _ace_poll_interval_sec())
#
#     _dlog("WAIT-START", f"timeout={timeout_val}", f"poll={poll_val}")
#
#     # 0) 서버 핑
#     try:
#         url_stats = f"{base.rstrip('/')}/system_stats"
#         _dlog("PING", url_stats)
#         r = requests.get(url_stats, timeout=5.0)
#         _dlog("PING-RESP", f"status={r.status_code}")
#         if r.status_code != 200:
#             raise RuntimeError(f"ComfyUI 응답 코드 {r.status_code}")
#     except Exception as e:
#         _dlog("PING-FAIL", f"{type(e).__name__}: {e}")
#         raise ConnectionError(f"ComfyUI에 연결 실패: {base} ({e})")
#
#     # 1) 제출
#     try:
#         url_prompt = f"{base.rstrip('/')}/prompt"
#         _dlog("POST-/prompt", f"url={url_prompt}", f"nodes={len(wf_graph)}")
#         r = requests.post(url_prompt, json={"prompt": wf_graph}, timeout=(5.0, 25.0))
#         _dlog("POST-RESP", f"ok={r.ok}", f"status={r.status_code}")
#     except Exception as e:
#         _dlog("POST-FAIL", f"{type(e).__name__}: {e}")
#         raise
#
#     if not r.ok:
#         try:
#             detail = r.json()
#         except Exception:
#             detail = r.text
#         _dlog("POST-NOK-BODY", str(detail)[:400])
#         raise RuntimeError(f"/prompt 제출 실패: {r.status_code} {detail}")
#
#     try:
#         resp: Dict[str, Any] = r.json() or {}
#     except Exception as e:
#         _dlog("POST-JSON-FAIL", f"{type(e).__name__}: {e}", f"text={r.text[:400]}")
#         raise
#
#     prompt_id = resp.get("prompt_id") or resp.get("promptId") or resp.get("id")
#     if not prompt_id:
#         raise RuntimeError("ComfyUI가 prompt_id를 반환하지 않았습니다.")
#     prompt_key = str(prompt_id)
#     _dlog("PROMPT-ID", prompt_key)
#
#     t0 = time.time()
#     last_outputs = 0
#     last_hist_status = None
#     tick = 0
#
#     last_queue_pending = None
#     last_queue_running = None
#     idle_ticks_after_output = 0
#
#     while True:
#         tick += 1
#         elapsed = time.time() - t0
#         if elapsed > timeout_val:
#             _dlog("TIMEOUT", f"elapsed={elapsed:.1f}s")
#             raise TimeoutError(
#                 "ACE-Step 대기 시간 초과\n"
#                 "- ComfyUI 실행/COMFY_HOST 확인\n"
#                 "- SaveAudio 노드 타입/옵션 확인 (mp3/wav/opus)\n"
#                 "- 큐가 길면 대기 또는 서버 재시작\n"
#                 "- 첫 실행 시 모델 다운로드 완료까지 대기\n"
#             )
#
#         # 큐 상태
#         try:
#             q = requests.get(f"{base.rstrip('/')}/queue", timeout=5.0)
#             if q.ok:
#                 qj: Dict[str, Any] = cast(Dict[str, Any], q.json() or {})
#                 last_queue_pending = len(cast(list, qj.get("queue_pending", []) or []))
#                 last_queue_running = len(cast(list, qj.get("queue_running", []) or []))
#                 if on_progress:
#                     on_progress({
#                         "stage": "queue",
#                         "pending": last_queue_pending,
#                         "running": last_queue_running,
#                         "elapsed": elapsed,
#                     })
#                 if tick % 5 == 0:
#                     _dlog("QUEUE", f"pending={last_queue_pending}", f"running={last_queue_running}")
#         except Exception as e:
#             if tick % 5 == 0:
#                 _dlog("QUEUE-READ-FAIL", f"{type(e).__name__}: {e}")
#
#         # 히스토리
#         try:
#             h = requests.get(f"{base.rstrip('/')}/history/{prompt_key}", timeout=10.0)
#             if not h.ok:
#                 if tick % 5 == 0:
#                     _dlog("HIST-HTTP", f"status={h.status_code}")
#                 time.sleep(poll_val)
#                 continue
#             hist_raw: Any = h.json()
#             hist: Dict[str, Any] = cast(Dict[str, Any], hist_raw or {})
#         except Exception as e:
#             if tick % 5 == 0:
#                 _dlog("HIST-READ-FAIL", f"{type(e).__name__}: {e}")
#             time.sleep(poll_val)
#             continue
#
#         entry_obj = hist.get(prompt_key)
#         if not isinstance(entry_obj, dict):
#             if tick % 5 == 0:
#                 _dlog("HIST-NO-ENTRY-YET")
#             time.sleep(poll_val)
#             continue
#         entry_dict: Dict[str, Any] = cast(Dict[str, Any], entry_obj)
#
#         # 상태/키 로깅(1회성)
#         st_obj = (entry_dict.get("status") or {})
#         st = st_obj.get("status") or st_obj.get("status_str")
#         if st != last_hist_status:
#             _dlog("HIST-STATUS", st, "| keys:", list(entry_dict.keys()))
#             if st_obj:
#                 _dlog("HIST-STATUS-OBJ", {k: st_obj.get(k) for k in list(st_obj.keys())[:6]})
#             last_hist_status = st
#
#         # 에러
#         if (st or "").lower() == "error":
#             err = st_obj.get("error") or {}
#             node_errors = err.get("node_errors") or {}
#             details = []
#             for nid, ne in node_errors.items():
#                 details.append(f"node {nid}: {ne.get('message') or ne}")
#             msg = err.get("message") or "ComfyUI 내부 에러"
#             _dlog("HIST-ERROR", msg, "|", " / ".join(details))
#             raise RuntimeError(f"ComfyUI 에러: {msg}\n" + ("\n".join(details) if details else ""))
#
#         # 출력 수 변화
#         outs = entry_dict.get("outputs") or {}
#         n_outs = sum(len(v or []) for v in (outs.values() if isinstance(outs, dict) else []))
#         if n_outs != last_outputs:
#             last_outputs = n_outs
#             _dlog("HIST-OUTPUTS", f"count={n_outs}")
#             if on_progress:
#                 on_progress({"stage": "running", "outputs": n_outs, "elapsed": elapsed})
#
#         # 정상 완료
#         exec_info = (st_obj.get("exec_info") or {})
#         queue_info = (exec_info.get("queue") or "")
#         if (st in ("success", "completed", "ok")) or ("completed" in str(queue_info).lower()):
#             _dlog("HIST-DONE", f"elapsed={elapsed:.1f}s", f"outputs={n_outs}")
#             if on_progress:
#                 on_progress({"stage": "completed", "outputs": n_outs, "elapsed": elapsed})
#             return entry_dict  # ✅
#
#         # 휴리스틱 완료: 출력이 있고 큐가 비었으면 몇 틱 후 완료 간주
#         if n_outs > 0 and last_queue_pending == 0 and last_queue_running == 0:
#             idle_ticks_after_output += 1
#             if idle_ticks_after_output == 1:
#                 _dlog("HIST-HEURISTIC-ARMED", "outputs>0 & queue empty -> waiting few ticks")
#             if idle_ticks_after_output >= 3:
#                 _dlog("HIST-HEURISTIC-DONE", f"elapsed={elapsed:.1f}s", f"outputs={n_outs}")
#                 if on_progress:
#                     on_progress({"stage": "completed(heuristic)", "outputs": n_outs, "elapsed": elapsed})
#                 return entry_dict
#         else:
#             idle_ticks_after_output = 0
#
#         time.sleep(poll_val)
#
#     # 타입 체커 안심용 (정상 흐름상 도달 불가)
#     return {}  # type: ignore[return-value]


# ───────────────────────────── 디버그 ────────────────────────────────────
# _LOG_PATH = Path(BASE_DIR) / "music_gen.log"  # 언제나 여기로도 기록
# # real use
# def _dlog(*args):
#     msg = " ".join(str(a) for a in args)
#     line = f"[MUSIC] {msg}"
#     try:
#         # 1) stdout (flush)
#         print(line, flush=True)
#     except Exception:
#         pass
#     try:
#         # 2) stderr (일부 런처에서 stdout이 안 보일 때 대비)
#         import sys
#         print(line, file=sys.stderr, flush=True)
#     except Exception:
#         pass
#     try:
#         # 3) 파일 로그 (GUI에서도 100% 확인 가능)
#         _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
#         with open(_LOG_PATH, "a", encoding="utf-8") as f:
#             f.write(line + "\n")
#     except Exception:
#         pass
