# -*- coding: utf-8 -*-
"""
audio_sync.py
- 오디오 길이와 (있으면) project.json의 가사를 이용해 타임라인 생성
- 결과는 segments.json / scene.json을 '항상' 오디오 파일 폴더에 저장(save=True일 때)
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

# ───────────────────────── utils 안전 import ─────────────────────────
try:
    from app.utils import audio_duration_sec, save_json, load_json, ensure_dir, sanitize_title
except (ImportError, ModuleNotFoundError):
    # 최소 동작 대체
    import json as _json
    from pathlib import Path as _Path

    def load_json(p: _Path, default=None):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return _json.load(f)
        except (FileNotFoundError, OSError, _json.JSONDecodeError):
            return default

    def save_json(p: _Path, obj: Any):
        p = _Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            _json.dump(obj, f, ensure_ascii=False, indent=2)

    def audio_duration_sec(p: _Path) -> float:
        # mutagen 우선
        try:
            from mutagen import File as MFile  # type: ignore
            mf = MFile(str(p))
            if mf and mf.info and getattr(mf.info, "length", None):
                return float(mf.info.length)
        except (ImportError, ValueError, OSError):
            pass
        # 폴백 실패
        return 0.0

    def ensure_dir(p: _Path) -> _Path:
        p = _Path(p)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def sanitize_title(t: str) -> str:
        import re as _re
        t = (t or "").strip()
        t = _re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", t)
        t = _re.sub(r"\s+", " ", t).strip()
        return t or "untitled"

# (선택) image/movie 문서 빌더 존재 시 사용
try:
    from app.image_movie_docs import build_image_json as _build_image_json, build_movie_json as _build_movie_json
except (ImportError, ModuleNotFoundError):
    _build_image_json = None
    _build_movie_json = None

# (선택) 런타임 설정 — 필요시 사용
try:
    from app import settings as settings  # noqa: N812
except (ImportError, ModuleNotFoundError):
    try:
        import settings as settings  # type: ignore  # noqa: N812
    except (ImportError, ModuleNotFoundError):
        class _DummySettings:
            BASE_DIR = "."
            FINAL_OUT = "."
            JSONS_DIR = "."
        settings = _DummySettings()  # type: ignore

# ───────────────────────── librosa(선택) ─────────────────────────
try:
    import librosa as _librosa_mod  # 실제 모듈
except (ImportError, OSError):
    _librosa_mod = None
librosa = cast(Any, _librosa_mod)  # 정적 분석 경고 억제


# ───────────────────────── 오디오 유틸 ─────────────────────────
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


def _safe_load_audio_for_onsets(p: Path):
    if librosa is None:
        return None, None
    try:
        y, sr = librosa.load(str(p), mono=True)  # type: ignore[attr-defined]
        return y, sr
    except (FileNotFoundError, OSError, ValueError, RuntimeError):
        return None, None


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
SECTION_HEADER_RE = re.compile(r"^\s*\[(?P<name>[^]]+)]\s*$", re.IGNORECASE)

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
_DEFAULT_WEIGHTS = {
    "verse": 1.0,
    "chorus": 1.2,
    "bridge": 1.0,
    "pre-chorus": 1.0,
    "intro": 0.6,
    "outro": 0.8,
    "unknown": 1.0,
}

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

def _round_half(x: float) -> float:
    return round(x * 2.0) / 2.0  # 0.5초 스냅

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
def _meaning_units_from_lyrics(lines: List[str], *, section: str, ai: Any) -> List[Dict[str, Any]]:
    """
    GPT 우선 의미 단위 분해. 실패하면 [].
    출력 예: [{"lines":[0,1,2], "summary":"감정이 고조되는 장면"}, ...]
    - 내부 변수는 소문자 사용
    - 광범위한 예외 제거(가능한 구체적으로)
    """
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
def _unit_lyric_text(lines: List[str], idxs: List[int]) -> str:
    picked = [lines[i].strip() for i in idxs if 0 <= i < len(lines)]
    joined = " ".join(picked).strip()
    return re.sub(r"\s+([,.!?])", r"\1", joined)

def _even_slices(n: int, k: int) -> List[Tuple[int, int]]:
    # n개 아이템을 k등분(앞쪽부터 1개 차이 허용)
    cuts = [round(j * n / k) for j in range(k + 1)]
    return [(cuts[j], cuts[j + 1]) for j in range(k)]
# ---- 의미 단위로 가사 적용(토큰 분배보다 '마지막에' 실행) ----  # // NEW
def _apply_units_overwrite(
    scenes: List[Dict[str, Any]],
    units: List[Dict[str, Any]],
    lines: List[str],
    target_sections: List[str] = None
) -> None:
    """
    이미 생성된 씬 목록(scenes)에 대해, 지정된 섹션을 골라
    의미 단위(units) 단위로 가사를 '균등 분배' 방식으로 덮어쓴다.
    - units[*]["lines"] : 원본 lines에서 사용할 인덱스 리스트
    - lines            : 원본 가사(라인 단위)
    - target_sections  : 기본 ["verse"]
    """
    if not scenes or not units or not lines:
        return

    sects = set(target_sections or ["verse"])
    idxs = [i for i, s in enumerate(scenes) if s.get("section") in sects]
    if not idxs:
        return

    # 유닛 -> 텍스트
    def _text_for_unit(u: Dict[str, Any]) -> str:
        sel = []
        for k in (u.get("lines") or []):
            try:
                if isinstance(k, int) and 0 <= k < len(lines):
                    sel.append(lines[k])
            except (TypeError, ValueError):
                continue
        return " ".join(sel).strip()

    unit_texts = [_text_for_unit(u) for u in units]
    unit_texts = [t for t in unit_texts if t]  # 빈 문자열 제거
    if not unit_texts:
        return

    # idxs를 unit_texts 개수만큼 '가능한 균등'하게 슬라이스
    m, n = len(idxs), len(unit_texts)
    base, rem = divmod(m, n)
    slices = []
    start = 0
    for r in range(n):
        size = base + (1 if r < rem else 0)
        end = start + size
        slices.append((start, end))
        start = end

    # 덮어쓰기
    for uidx, (a, b) in enumerate(slices):
        for j in idxs[a:b]:
            scenes[j]["lyric"] = unit_texts[uidx]



# ───────────────────────── 섹션 내 길이 분배 & 스냅 ─────────────────────────
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


def _merge_overflow_shots(segs: List[Dict[str, Any]], *, max_shots: int) -> List[Dict[str, Any]]:
    """샷 개수가 너무 많을 때 인접 샷 병합(섹션 유지)으로 감소."""
    if len(segs) <= max_shots:
        return segs

    merged = segs[:]
    i = 0
    while len(merged) > max_shots and i < len(merged) - 1:
        if merged[i]["section"] == merged[i + 1]["section"]:
            merged[i]["end"] = merged[i + 1]["end"]
            merged[i]["duration"] = round(float(merged[i]["end"]) - float(merged[i]["start"]), 3)
            del merged[i + 1]
        else:
            i += 1

    while len(merged) > max_shots and len(merged) >= 2:
        merged[0]["end"] = merged[1]["end"]
        merged[0]["duration"] = round(float(merged[0]["end"]) - float(merged[0]["start"]), 3)
        del merged[1]

    return merged


# ───────────────────────── 공개 API: 분석 ─────────────────────────
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


# ───────────────────────── 전역 콘텍스트/GPT 프롬프트 ─────────────────────────
def _analyze_lyrics_global_kor(ai: Any, *, lyrics: str, title: str = "") -> Dict[str, Any]:
    """
    가사 전체를 한 번만 분석해 전역 컨텍스트를 만든다.
    반환 예:
    {
      "global_summary": "...", "themes": [...], "palette": "...", "style_guide": "...",
      "section_moods": {"intro":"암시/미니멀", "verse":"잔잔/근접", "chorus":"개방감/광각", ...},
      "avoid": ["가사 원문 인용 금지", "문자 텍스트 등장 금지"]
    }
    """
    lyrics = (lyrics or "").strip()
    if not ai or not lyrics:
        return {
            "global_summary": "도시적 감성의 서정적인 곡. 정면 얼굴과 일관된 스타일 유지.",
            "themes": ["서정", "도시", "야간", "희미한 추억"],
            "palette": "야간 네온 + 따뜻한 톤, 보케",
            "style_guide": "인물 정면, 헤어/의상/분위기 일관, 자연스러운 조명",
            "section_moods": {
                "intro": "도입/암시/미니멀",
                "verse": "잔잔함/근접/친밀감",
                "chorus": "개방감/광각/확장",
                "bridge": "전환감/대비/변화",
                "outro": "여운/잔상/감쇠",
            },
            "avoid": ["가사 원문 인용 금지", "문자 텍스트 삽입 금지"],
        }

    ask = None
    for name in ("lyrics_analysis_kor", "analyze_lyrics", "complete", "chat", "__call__"):
        fn = getattr(ai, name, None)
        if callable(fn):
            ask = fn
            break
    if not ask:
        return _analyze_lyrics_global_kor(None, lyrics=lyrics, title=title)

    sys_msg = (
        "다음 전체 가사를 분석해 영상 연출용 전역 컨텍스트를 JSON으로 만들어라. "
        "가사 원문 문구는 결과에 인용하지 말 것. 한국어만 사용. "
        "필드: global_summary(문장), themes(리스트), palette(문장), style_guide(문장), "
        "section_moods(섹션별 문장: intro/verse/chorus/bridge/outro), avoid(리스트)."
    )
    user_msg = f"[제목] {title}\n[가사]\n{lyrics}"
    try:
        raw = ask(f"{sys_msg}\n\n{user_msg}")
        data = json.loads(raw) if isinstance(raw, str) else (raw or {})
        if isinstance(data, dict) and data:
            return data
    except (TypeError, ValueError, RuntimeError, json.JSONDecodeError):
        pass
    return _analyze_lyrics_global_kor(None, lyrics=lyrics, title=title)


def _gpt_prompt_for_scene(
    ai: Any,
    *,
    section: str,
    summary: str | None,
    index: int,
    duration: float,
    characters: List[str] | None = None,
    meta: Dict[str, Any] | None = None,
    global_ctx: Dict[str, Any] | None = None,
    mode: str = "image",  # "image" | "movie" | "generic"
) -> str:
    try:
        if not ai:
            return ""
        ask = None
        for name in ("shot_prompt_kor", "scene_prompt_kor", "complete", "chat", "__call__"):
            fn = getattr(ai, name, None)
            if callable(fn):
                ask = fn
                break
        if not ask:
            return ""

        sec = (section or "").lower()
        moods = (global_ctx or {}).get("section_moods", {}) if isinstance(global_ctx, dict) else {}
        sec_mood = (
            moods.get("chorus") if sec.startswith("chorus") else
            moods.get("verse")  if sec.startswith("verse")  else
            moods.get("bridge") if "bridge" in sec          else
            moods.get("intro")  if "intro"  in sec          else
            moods.get("outro")  if "outro"  in sec          else
            "자연스러운 흐름"
        )
        palette = (global_ctx or {}).get("palette") or "자연광/네온 혼합, 보케"
        style_guide = (global_ctx or {}).get("style_guide") or "정면 얼굴, 실사화, 스타일 일관"
        avoid = (global_ctx or {}).get("avoid") or ["가사 원문 인용 금지"]

        has_people = bool(characters)
        frontal = "정면 얼굴, " if has_people else ""
        realism = "실사화, " if has_people else ""
        role = "영상용" if mode == "movie" else "이미지용"
        extra = "카메라/동작/조명을 간결히 포함" if mode == "movie" else "조명/분위기 중심"

        text_summary = (summary or "감정 변주를 강조").strip()
        title = str((meta or {}).get("title") or "").strip()
        genre = str((meta or {}).get("genre") or (meta or {}).get("style") or "").strip()
        tags = (meta or {}).get("tags_in_use") or (meta or {}).get("ace_tags") or []
        tagline = ", ".join([t for t in tags if isinstance(t, str)])[:100]

        sys_prompt = (
            f"{role} 한국어 프롬프트 한 문장을 생성하라. "
            "원칙: (1) 가사 원문 문구 인용 금지, (2) 섹션 무드/팔레트/스타일가이드를 반영, "
            f"(3) {extra}, (4) 따옴표/코드블록 없이 문장만 출력."
        )
        user_prompt = (
            f"- 프로젝트: {title or '무제'} / 장르: {genre or '미정'}\n"
            f"- 샷 번호: {index}\n"
            f"- 섹션: {section}\n"
            f"- 의미 요약: {text_summary}\n"
            f"- 전역 무드: {sec_mood}\n"
            f"- 팔레트: {palette}\n"
            f"- 스타일: {realism}{frontal}{style_guide}\n"
            f"- 길이(초): {max(0.0, float(duration)):.3f}\n"
            f"- 태그: {tagline}\n"
            f"- 피해야 할 것: {', '.join(map(str, avoid))}\n"
        )

        raw = ask(f"{sys_prompt}\n\n{user_prompt}")
        val = raw.get("text") if isinstance(raw, dict) else str(raw or "")
        return val.strip().strip("`'\"").strip()
    except (TypeError, ValueError, RuntimeError):
        return ""


# ───────────────────────── story.json 빌더 ─────────────────────────
def build_story_json(
    *,
    audio_file: str,
    project_json: str,
    fps: int = 24,
    max_shot_sec: float = 5.0,
    enable_transitions: bool = True,
    use_gpt: bool = True,
    prompt_provider: Optional[Callable[..., str]] = None,
    also_build_docs: bool = False,
    ai: Any = None,
    workflow_dir: Optional[str] = None,  # (호환 유지)
    hair_map: Optional[dict] = None,
    on_progress: Optional[Callable[[str], None]] = None,
    force_total_sec: Optional[float] = None,
    # ▼▼▼ 추가된 파라미터 ▼▼▼                                  # // NEW
    meaning_units_override: Optional[List[Dict[str, Any]]] = None,  # [{"lines":[...],"summary":"..."}]
    meaning_unit_lines: Optional[List[str]] = None,                 # 원문을 라인 단위로 나눈 리스트
) -> Dict[str, Optional[str]]:

    """
    segments 기반으로 제작용 story.json 생성 + (옵션) image.json/movie.json 생성
    - 최종 저장은 항상 v1.1 포맷(normalize_to_v11)을 강제
    """
    _ = workflow_dir  # 미사용(호환)

    # 진행 출력
    def _p(msg: str) -> None:
        if on_progress:
            try:
                on_progress(str(msg))
            except (TypeError, ValueError, RuntimeError):  # // CHANGED: broad-except 제거
                pass
        print(f"[BUILD-STORY] {msg}", flush=True)

    # 출력 폴더: project.json이 있는 폴더
    def _determine_out_dir(project_json_path: str, _title_text: str) -> Path:
        pj = Path(project_json_path)
        return pj.parent

    def _ensure_dir(p: Path) -> Path:
        p.mkdir(parents=True, exist_ok=True)
        return p

    # 입력
    audio_p = Path(audio_file)
    pj_p = Path(project_json)
    meta = load_json(pj_p, {}) or {}
    title = sanitize_title(meta.get("title") or pj_p.parent.name or "untitled")
    lyrics_all = str(meta.get("lyrics", "") or "")

    # 전역 가사 분석(한 번만)
    global_ctx = _analyze_lyrics_global_kor(ai, lyrics=lyrics_all, title=title)

    out_dir = _determine_out_dir(project_json, title)
    _ensure_dir(out_dir)
    imgs_dir = _ensure_dir(out_dir / "imgs")
    _ensure_dir(out_dir / "clips")
    _p(f"target project dir = {out_dir}")

    # segments 생성(분석기 호출)
    try:
        try:
            from app.audio_sync import analyze_project as _an  # type: ignore
        except (ImportError, ModuleNotFoundError):
            from audio_sync import analyze_project as _an  # type: ignore
    except (ImportError, ModuleNotFoundError) as e:  # pragma: no cover
        raise RuntimeError(f"analyze_project import 실패: {type(e).__name__}: {e}")

    try:
        res = _an(
            str(pj_p.parent),
            audio_path=str(audio_p),
            save=True,
            ai=ai,
            force_total_sec=force_total_sec,
        )
    except (TypeError, ValueError, RuntimeError, FileNotFoundError):
        res = _an(
            str(pj_p.parent),
            audio_path=str(audio_p),
            save=True,
            force_total_sec=force_total_sec,
        )

    seg_json_path = Path(res.get("segments_path") or (audio_p.parent / "segments.json"))
    segdoc = load_json(seg_json_path, {}) or {}
    segs: List[Dict[str, Any]] = list(segdoc.get("segments") or [])
    total_dur = float(segdoc.get("duration") or 0.0)

    # 샷 상한 보정 + 60s 이상이면 인트로 보장(세그먼트 레벨)
    segs = _limit_shot_lengths_for_story(segs, max_shot_sec)
    segs = _ensure_intro_for_story(segs, total_duration=total_dur, threshold_sec=60.0, intro_len=1.5)

    effect_map: Dict[str, List[str]] = {
        "intro":  ["soft light", "film grain", "gentle camera pan"],
        "verse":  ["bokeh", "slow push-in"],
        "chorus": ["bloom", "wide angle"],
        "bridge": ["rack-focus", "light streaks"],
        "outro":  ["fade-out", "soft glow"],
    }

    # 캐릭터 스타일
    try:
        hair = None
        if ai and hasattr(ai, "character_style_kor"):
            hair = ai.character_style_kor(
                char_ids=["female_01", "male_01"],
                song_title=title,
                lyrics=lyrics_all,
            )
        if not isinstance(hair, dict) or not hair:
            hair = hair_map or {
                "female_01": "긴 웨이브펌, 갈색 머리, 슬림 체형, 일관성 유지",
                "male_01":   "짧은 머리, 검정 머리, 넓은 어깨, 일관성 유지",
            }
    except (TypeError, ValueError, RuntimeError):
    # 폴백
        hair = hair_map or {
            "female_01": "긴 웨이브펌, 갈색 머리, 슬림 체형, 일관성 유지",
            "male_01":   "짧은 머리, 검정 머리, 넓은 어깨, 일관성 유지",
        }

    def mood_for_section(sec: str) -> str:
        s = (sec or "").lower()
        if s.startswith("chorus"):
            return "개방감, 광각, 확장되는 감정"
        if s.startswith("verse"):
            return "잔잔함, 근접, 친밀감"
        if "bridge" in s:
            return "전환감, 대비, 변화"
        if "intro" in s:
            return "도입, 암시, 미니멀"
        if "outro" in s:
            return "여운, 잔상, 감쇠"
        return "자연스러운 흐름"

    # 씬 구성
    scenes: List[Dict[str, Any]] = []
    last_section = None

    for i, g in enumerate(segs, start=1):
        section = str(g.get("section") or "verse")
        first_of_section = (section != last_section)
        sid = f"t_{i:03d}"
        characters = ["female_01", "male_01"] if section == "chorus" else ["female_01"]
        has_people = bool(characters)

        prompt_txt: Optional[str] = None
        if use_gpt and ai:
            prompt_txt = _gpt_prompt_for_scene(
                ai,
                section=section,
                summary=(g.get("summary") or ""),
                index=i,
                duration=float(g.get("duration", 0.0) or 0.0),
                characters=characters,
                meta=meta,
                global_ctx=global_ctx,
                mode="movie" if also_build_docs else "image",
            ) or None

        if (not prompt_txt) and callable(prompt_provider):
            try:
                prompt_txt = prompt_provider(
                    section=section,
                    start=float(g.get("start", 0.0)),
                    end=float(g.get("end", 0.0)),
                    duration=float(g.get("duration", 0.0)),
                    title=title,
                    meta=meta,
                    characters=characters,
                    summary=g.get("summary") or "",
                    index=i,
                    global_ctx=global_ctx,
                )
            except (TypeError, ValueError, RuntimeError):
                prompt_txt = None

        if not prompt_txt:
            summary_text = (g.get("summary") or "감정 변주를 강조").strip()
            mood_text = mood_for_section(section)
            musts = "실사화, 정면 얼굴, 일관된 스타일(헤어/의상/분위기 유지)" if has_people else "일관된 스타일"
            prompt_txt = f"{summary_text}, {mood_text}, {musts}, 자연스러운 조명과 보케, 고품질"

        scenes.append({
            "id": sid,
            "section": section,
            "start": float(g.get("start", 0.0)),
            "end": float(g.get("end", 0.0)),
            "duration": round(float(g.get("duration", 0.0)), 3),
            "scene": g.get("label") or section,
            "characters": characters,
            "effect": effect_map.get(section, effect_map["verse"]),
            "screen_transition": bool(first_of_section and enable_transitions),
            "img_file": str(imgs_dir / f"{sid}.png"),
            "prompt": prompt_txt,
            "needs_character_asset": True,
            "lyric": str(g.get("lyric", "")),
        })
        last_section = section

    # ---- 의미 단위로 가사 적용(토큰 분배보다 '마지막에' 실행) ----  # // NEW
    muo = locals().get("meaning_units_override")
    mul = locals().get("meaning_unit_lines")
    try:
        if muo and mul:
            _apply_units_overwrite(
                scenes,
                muo,
                mul,
                target_sections=["verse"]
            )
    except (TypeError, ValueError, KeyError) as e:
        _p(f"의미 단위 덮어쓰기 건너뜀(입력 형식 오류): {e}")

    # story 원본
    story_raw: Dict[str, Any] = {
        "audio": str(audio_p),
        "fps": int(fps),
        "duration": round(total_dur if total_dur else (scenes[-1]["end"] if scenes else 0.0), 3),
        "offset": 0.0,
        "title": title,
        "characters": ["female_01", "male_01"],
        "character_styles": hair,
        "global_context": global_ctx,
        "scenes": scenes,
    }

    story_path = out_dir / "story.json"
    try:
        try:
            from app.image_movie_docs import normalize_to_v11, validate_story_v11_dict
        except (ImportError, ModuleNotFoundError):
            from image_movie_docs import normalize_to_v11, validate_story_v11_dict  # type: ignore

        story_v11 = normalize_to_v11(story_raw)
        story_v11["fps"] = int(fps)
        save_json(story_path, story_v11)

        errs = []
        try:
            errs = validate_story_v11_dict(story_v11)
        except (TypeError, ValueError, RuntimeError):
            errs = []
        if errs:
            _p("⚠ v1.1 검증 경고: " + " | ".join(errs))
        else:
            _p("v1.1 저장 및 검증 통과")
    except (ImportError, ModuleNotFoundError, OSError, ValueError) as e:
        # 실패 시라도 원본을 남김
        save_json(story_path, story_raw)
        _p(f"정규화 실패 → 원본 저장 ({type(e).__name__}: {e})")

    _p(f"story.json saved → {story_path}")

    # image.json / movie.json (옵션)
    img_path: Optional[str] = None
    mov_path: Optional[str] = None
    if also_build_docs and (_build_image_json or _build_movie_json):
        try:
            if _build_image_json:
                img_path = _build_image_json(str(out_dir), hair_map=hair_map)
            if _build_movie_json:
                mov_path = _build_movie_json(str(out_dir), hair_map=hair_map)
            if img_path:
                _p(f"image.json -> {img_path}")
            if mov_path:
                _p(f"movie.json -> {mov_path}")
        except (ImportError, ModuleNotFoundError, OSError, ValueError) as e:
            _p(f"image/movie 문서 생성 실패: {type(e).__name__}: {e}")

    return {
        "story": str(story_path),
        "image": str(img_path) if img_path else None,
        "movie": str(mov_path) if mov_path else None,
    }


# ───────────────────────── story 빌더 보조(shot 길이/인트로) ─────────────────────────
def _split_segment_for_story(seg: Dict[str, Any], max_len: float) -> List[Dict[str, Any]]:
    """긴 세그먼트를 max_len 이하로 여러 개로 쪼갠다."""
    start_ts = float(seg.get("start", 0.0))
    end_ts = float(seg.get("end", 0.0)) or (start_ts + float(seg.get("duration", 0.0)))

    if end_ts <= start_ts:
        return []

    duration = end_ts - start_ts
    if duration <= max_len:
        seg2 = dict(seg)
        seg2["duration"] = round(duration, 3)
        return [seg2]

    parts: List[Dict[str, Any]] = []
    cursor = start_ts
    index = 0

    while cursor < end_ts - 1e-6:
        piece = min(max_len, end_ts - cursor)
        sub = dict(seg)
        sub["start"] = round(cursor, 3)
        sub["end"] = round(cursor + piece, 3)
        sub["duration"] = round(piece, 3)
        base_label = str(seg.get("label") or seg.get("section") or "scene").strip()
        sub["label"] = f"{base_label}_{index + 1:02d}"
        parts.append(sub)
        cursor += piece
        index += 1

    return parts


def _limit_shot_lengths_for_story(segments_in: List[Dict[str, Any]], max_len: float) -> List[Dict[str, Any]]:
    """각 샷의 길이를 max_len 이하로 제한하고, 순차적으로 시간(start/end)을 재계산한다."""
    max_secs = float(max(0.5, max_len))

    pieces: List[Dict[str, Any]] = []
    for seg in segments_in or []:
        pieces.extend(_split_segment_for_story(seg, max_secs))

    # 연속 재타이밍(겹침 방지)
    t = 0.0
    for seg in pieces:
        d = float(seg.get("duration", 0.0))
        seg["start"] = round(t, 3)
        seg["end"] = round(t + d, 3)
        t += d

    return pieces


def _ensure_intro_for_story(
    segments_in: List[Dict[str, Any]],
    *,
    total_duration: float,
    threshold_sec: float = 60.0,
    intro_len: float = 1.5,
) -> List[Dict[str, Any]]:
    """총 길이가 threshold 이상인데 첫 섹션이 intro가 아니면 intro 세그먼트를 1개 삽입."""
    try:
        total = float(total_duration)
    except (TypeError, ValueError):
        total = 0.0

    if total < float(threshold_sec) or not segments_in:
        return segments_in

    first = segments_in[0]
    section = str(first.get("section") or "").lower()
    if section.startswith("intro"):
        return segments_in

    intro_secs = float(max(0.5, intro_len))
    intro = {
        "id": "intro",
        "section": "intro",
        "label": "Intro",
        "start": 0.0,
        "end": round(intro_secs, 3),
        "duration": round(intro_secs, 3),
        "summary": "도입/암시/미니멀",
    }

    result: List[Dict[str, Any]] = [intro]
    for seg in segments_in:
        s = float(seg.get("start", 0.0)) + intro_secs
        e = float(seg.get("end", 0.0)) + intro_secs
        if e <= s:
            continue
        sub = dict(seg)
        sub["start"] = round(s, 3)
        sub["end"] = round(e, 3)
        sub["duration"] = round(e - s, 3)
        result.append(sub)

    return result


# ───────────────────────── 인트로 삽입(섹션 타임라인 단계) ─────────────────────────
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


# ───────────────────────── 호환 보조 ─────────────────────────



# ============================================================
# drop-in replacement for: split_lyrics_into_semantic_units_ai
# - 컷 수 강제 없음
# - 반복 보존
# - 섹션 태그 제거/한줄화는 호출부에서 이미 처리했다고 가정(그대로 받아서 분할)
# - AI가 가능하면 우선 시도 → 실패/비일관 시 규칙 기반으로 자연 호흡 분할
# ============================================================
# ============================================================
# Natural Korean lyric splitter (no hardcoded, no per-lyric helpers)
# - 컷 수 강제하지 않음
# - 반복 보존
# - 섹션 태그 제거/한줄화는 호출부에서 이미 처리했다고 가정(그대로 받아서 분할)
# - AI가 있으면 시도하되, 결과가 부자연스러우면 규칙 기반으로 교정
# ============================================================




# 가사# 가사# 가사# 가사# 가사# 가사# 가사# 가사# 가사# 가사# 가사# 가#


# audio_sync.py — Whisper 정렬 + 폴백 배분 통합판


from typing import List, Tuple, Optional
import os, math, json, subprocess, re

# ─────────────────────────────────────────────────────────────
# 0) 유틸: 오디오 길이 견고 획득(중앙값, 3배 튐 방지)
# ─────────────────────────────────────────────────────────────
def _probe_duration_ffprobe(path: str) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
            stderr=subprocess.STDOUT,
        )
        data = json.loads(out.decode("utf-8", "ignore"))
        dur = float(data.get("format", {}).get("duration", 0.0) or 0.0)
        return dur if math.isfinite(dur) and dur > 0 else 0.0
    except Exception:
        return 0.0

def _probe_duration_mutagen(path: str) -> float:
    try:
        from mutagen import File as MutagenFile  # type: ignore
        mf = MutagenFile(path)
        dur = float(getattr(mf, "info", None).length if mf and getattr(mf, "info", None) else 0.0)
        return dur if math.isfinite(dur) and dur > 0 else 0.0
    except Exception:
        return 0.0

def _probe_duration_pydub(path: str) -> float:
    try:
        from pydub import AudioSegment  # type: ignore
        dur_ms = len(AudioSegment.from_file(path))
        dur = float(dur_ms) / 1000.0
        return dur if math.isfinite(dur) and dur > 0 else 0.0
    except Exception:
        return 0.0

def _probe_duration_soundfile(path: str) -> float:
    try:
        import soundfile as sf  # type: ignore
        with sf.SoundFile(path) as f:
            dur = float(len(f) / (f.samplerate or 1))
            return dur if math.isfinite(dur) and dur > 0 else 0.0
    except Exception:
        return 0.0

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

def get_audio_duration(path: str) -> float:
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
        _probe_duration_librosa,
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

# ─────────────────────────────────────────────────────────────
# 1) 온셋 검출(없으면 빈 리스트)
# ─────────────────────────────────────────────────────────────
def detect_onsets_seconds(
    path: str,
    *,
    sr: int = 22050,
    hop_length: int = 512,
    backtrack: bool = True,
) -> List[float]:
    p = str(path or "").strip()
    if not p or not os.path.isfile(p):
        return []
    try:
        import numpy as np  # type: ignore
        import librosa  # type: ignore
        y, _sr = librosa.load(p, sr=sr, mono=True)
        if y is None or len(y) == 0:
            return []
        on_f = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=hop_length, backtrack=backtrack, units="frames"
        )
        times = librosa.frames_to_time(on_f, sr=sr, hop_length=hop_length)
        out = [float(t) for t in np.asarray(times).ravel().tolist() if math.isfinite(float(t)) and t >= 0.0]
        out_sorted = []
        last = -1.0
        for t in sorted(out):
            if last < 0 or (t - last) >= 0.01:
                out_sorted.append(t)
                last = t
        return out_sorted
    except Exception:
        return []

# ─────────────────────────────────────────────────────────────
# 2) 길이 배분(온셋 분위 + tail + end_bias) — 폴백에 사용
# ─────────────────────────────────────────────────────────────
def layout_time_by_weights(
    units: List[str],
    total_start: float,
    total_end: float,
    *,
    onsets: Optional[List[float]] = None,
    min_len: float = 0.5,
    end_bias_sec: float = 2.5,
    avg_min_sec_per_unit: float = 2.0,
) -> List[Tuple[float, float]]:
    import re as _re
    n = len(units or [])
    if n <= 0:
        return []
    start = float(total_start)
    end = float(total_end)
    if not math.isfinite(start):
        start = 0.0
    if not math.isfinite(end) or end <= start:
        end = start + max(1.0, n * min_len)

    usable_end = end
    if onsets:
        hz = sorted([float(t) for t in onsets if math.isfinite(float(t)) and start <= float(t) <= end])
        k = len(hz)
        if k >= 2:
            diffs = [hz[i + 1] - hz[i] for i in range(k - 1)]
            diffs = [d for d in diffs if d > 1e-3]
            if diffs:
                ds = sorted(diffs)
                mid = len(ds) // 2
                med = ds[mid] if len(ds) % 2 == 1 else 0.5 * (ds[mid - 1] + ds[mid])
            else:
                med = 0.6
            q_raw = n / float(n + 4.0)
            q = max(0.45, min(0.75, q_raw))
            idx = int(max(0, min(k - 1, round((k - 1) * q))))
            t_q = hz[idx]
            tail = med * 1.5
            guess = t_q + tail + max(0.0, float(end_bias_sec))
            floor_end = start + max(n * min_len, n * max(0.5, float(avg_min_sec_per_unit)))
            usable_end = max(floor_end, min(end, guess))

    def _w(s: str) -> float:
        core = _re.sub(r"[^0-9A-Za-z가-힣]+", "", (s or ""))
        return max(1.0, float(len(core)))

    weights = [_w(u) for u in units]
    wsum = sum(weights) or float(n)
    bounds = [start]
    acc = 0.0
    span = usable_end - start
    for w in weights:
        acc += w
        bounds.append(start + span * (acc / wsum))

    if onsets:
        hints = sorted([float(x) for x in onsets if math.isfinite(float(x))])
        hints = [h for h in hints if start <= h <= usable_end]
        tol = max(0.12, min(0.6, span * 0.02))
        for i in range(1, len(bounds) - 1):
            t = bounds[i]
            lo, hi = 0, len(hints) - 1
            best = None
            bestd = 1e9
            while lo <= hi:
                mid = (lo + hi) // 2
                d = abs(hints[mid] - t)
                if d < bestd:
                    bestd = d
                    best = hints[mid]
                if hints[mid] < t:
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best is not None and abs(best - t) <= tol:
                bounds[i] = best

    tiny = 1e-6
    for i in range(1, len(bounds)):
        if bounds[i] <= bounds[i - 1] + tiny:
            bounds[i] = bounds[i - 1] + tiny

    segs = [(bounds[i], bounds[i + 1]) for i in range(n)]

    for i in range(n):
        a, b = segs[i]
        if b - a < min_len:
            need = min_len - (b - a)
            push = min(need, usable_end - b)
            if push > 0:
                segs[i] = (a, b + push)
                for j in range(i + 1, n):
                    aj, bj = segs[j]
                    segs[j] = (aj + push, bj + push)

    la, lb = segs[-1]
    if abs(lb - usable_end) > 1e-6:
        segs[-1] = (la, usable_end)
    return segs

# ─────────────────────────────────────────────────────────────
# 3) (선택) 보컬 분리 — 없으면 None 반환(폴백)
# ─────────────────────────────────────────────────────────────
def separate_vocals_demucs(input_path: str) -> Optional[str]:
    try:
        if not os.path.isfile(input_path):
            return None
        out = subprocess.check_output(["demucs", "--help"], stderr=subprocess.STDOUT)
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






# ─────────────────────────────────────────────────────────────
# 5) 공식 가사 라인과 Whisper 단어 정렬
# ─────────────────────────────────────────────────────────────
def _norm_txt(s: str) -> str:
    s = re.sub(r"\s+", "", (s or ""))
    return s

def align_words_to_lyrics_lines(official_lyrics: str, words: List[Tuple[float, float, str]]) -> List[Tuple[Optional[float], Optional[float], str]]:
    """
    줄바꿈 기준 공식 가사 라인과 Whisper 단어 시퀀스를 그리디하게 정렬
    결과: (start, end, line_text) 리스트. 실패 라인은 (None, None, line_text)
    """
    lines = [ln.strip() for ln in (official_lyrics or "").replace("\r", "").split("\n")]
    lines = [ln for ln in lines if ln.strip()]
    i = 0
    n = len(words)
    aligned: List[Tuple[Optional[float], Optional[float], str]] = []

    # 거리 함수: rapidfuzz 우선, 실패 시 difflib
    def _dist(a: str, b: str) -> float:
        try:
            from rapidfuzz.distance import Levenshtein  # type: ignore
            return float(Levenshtein.distance(a, b))
        except Exception:
            import difflib
            # difflib는 1.0이 유사, 여기선 거리로 바꿔 사용
            ratio = difflib.SequenceMatcher(None, a, b).ratio()
            return float(max(len(a), len(b)) * (1.0 - ratio))

    for ln in lines:
        target = _norm_txt(ln)
        if not target:
            aligned.append((None, None, ln))
            continue

        best_j = i
        best_cost = 1e9
        best_span = None

        buf = ""
        start_t = None
        for j in range(i, min(i + 250, n)):
            ws, we, wt = words[j]
            token = (wt or "").strip()
            if not token:
                continue
            if start_t is None:
                start_t = ws
            buf += token
            cand = _norm_txt(buf)
            if len(cand) > len(target) * 1.6:
                break
            cost = _dist(cand, target)
            norm = cost / max(1, len(target))
            score = norm + abs(len(target) - len(cand)) * 0.01
            if score < best_cost:
                best_cost = score
                best_j = j + 1
                best_span = (start_t, we)
            if norm <= 0.22:
                break

        if best_span is None:
            aligned.append((None, None, ln))
        else:
            aligned.append((best_span[0], best_span[1], ln))
            i = best_j

    return aligned

# ─────────────────────────────────────────────────────────────
# 6) 상위 래퍼: Whisper 우선 정렬 → 실패 라인만 폴백 배분
# ─────────────────────────────────────────────────────────────
def prepare_pure_lyrics_lines(raw_lyrics: str, drop_section_tags: bool = True) -> List[str]:
    lines = []
    for ln in (raw_lyrics or "").replace("\r", "").split("\n"):
        s = ln.strip()
        if not s:
            continue
        if drop_section_tags and re.fullmatch(r"\s*\[[^]]+]\s*", s, flags=re.I):
            continue
        lines.append(s)
    return lines



# ------------------------------------------------------------
# 1) Whisper 전사: CPU 고정 + initial_prompt 안전 + 폴백
# ------------------------------------------------------------
from typing import Dict, Any  # 파일 상단에 이미 있다면 중복 무시

def transcribe_words(
    path: str,
    model_size: str = "medium",
    initial_prompt: str = "",
    *,
    beam_size: int = 5,
    vad_filter: bool = False,  # 인자 유지하지만 실제 호출은 항상 False로 강제
) -> Dict[str, Any]:
    """
    오디오를 Whisper로 전사하여 단어 타임스탬프를 반환한다.
    - CPU 전용(CUDA 완전 차단)
    - faster-whisper 우선, 실패 시 빈 dict 반환(상위 폴백 로직이 처리)
    반환 형식: {"segments":[{start,end,text}], "words":[(ws,we,token),...]}
    실패 시 {}.
    """
    import os
    import re
    import inspect

    p = (path or "").strip()
    if not p or not os.path.isfile(p):
        return {}

    def _trim_prompt(s: str) -> str:
        s = re.sub(r"\s+", " ", s or "").strip()
        parts = s.split(" ")
        parts = [w for w in parts if w]
        if len(parts) > 80:
            parts = parts[:80]
        s2 = " ".join(parts)
        if len(s2.encode("utf-8")) > 1000:
            # 바이트 1000 내에서 자르기
            b = s2.encode("utf-8")[:1000]
            try:
                s2 = b.decode("utf-8", errors="ignore")
            except Exception:
                s2 = ""
        return s2

    initial_prompt = _trim_prompt(initial_prompt)

    # faster-whisper (CPU, int8)
    try:
        from faster_whisper import WhisperModel  # type: ignore
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

        sig = inspect.signature(model.transcribe)
        # 항상 False로 강제(onnxruntime 경로 차단)
        call_kwargs: Dict[str, Any] = {
            "word_timestamps": True,
            "vad_filter": False,
            "beam_size": int(max(1, beam_size)),
        }
        if "initial_prompt" in sig.parameters and initial_prompt:
            call_kwargs["initial_prompt"] = initial_prompt

        segments, _ = model.transcribe(p, **call_kwargs)

        out_segs: List[Dict[str, Any]] = []
        out_words: List[Tuple[float, float, str]] = []
        for seg in (segments or []):
            s = float(seg.get("start", 0.0) or 0.0)
            e = float(seg.get("end", s + 0.01) or (s + 0.01))
            out_segs.append({"start": s, "end": e, "text": (seg.get("text") or "").strip()})
            for w in (seg.get("words") or []):
                ws = w.get("start"); we = w.get("end")
                if ws is None or we is None:
                    continue
                out_words.append((float(ws), float(we), (w.get("word") or "").strip()))
        return {"segments": out_segs, "words": out_words}
    except Exception:
        return {}




# ------------------------------------------------------------
# 2) 가사 싱크: 보컬분리(옵션) → 전사 → 정렬 → 폴백 배분
# ------------------------------------------------------------
def sync_lyrics_with_whisper(
    audio_path: str,
    lyrics_text: str,
    *,
    model_size: str = "large-v3",
    use_vocal_separation: bool = True,
    min_len: float = 0.5,
    end_bias_sec: float = 2.5,
    avg_min_sec_per_unit: float = 2.0,
    start_preroll: float = 0.30,
    beam_size: int = 5,
    vad_filter: bool = True,  # 외부 시그니처 보존(내부에서는 강제로 False 적용)
) -> dict:
    """
    라인 단위 가사 정렬(Whisper 단어 타임스탬프 기반, CPU 전용):
      - 어떤 분기에서도 **반드시 dict**를 반환한다.
      - faster-whisper는 **device='cpu', compute_type='int8'**로 고정.
      - **vad_filter는 내부에서 항상 False**로 호출하여 onnxruntime(Silero VAD) 경로를 원천 차단.
      - 라인→단어 시퀀스 정렬(경량 DP) + librosa RMS VAD 클램프 + 온셋 스냅 + 폴백 배분.
    반환 dict 스키마:
      {
        "segments": [(start: float, end: float, text: str), ...],
        "onsets": [float, ...],
        "duration_sec": float,
        "start_at": 0.0
      }
    """
    import re
    from pathlib import Path
    from typing import List, Tuple, Optional
    import numpy as np

    # 0) 필수 유틸 (기존에 존재하는 함수만 사용)
    try:
        duration_sec = float(get_audio_duration(audio_path))
    except Exception:
        duration_sec = 0.0

    try:
        onsets = list(detect_onsets_seconds(audio_path))
    except Exception:
        onsets = []

    try:
        raw_lines = list(prepare_pure_lyrics_lines(lyrics_text))
    except Exception:
        raw_lines = [(lyrics_text or "").strip()]

    # 1) 라인 전처리
    lines: List[str] = []
    for ln in raw_lines:
        s = (ln or "").strip()
        s = re.sub(r"[·…“”\"'`]+", "", s)
        if s:
            lines.append(s)
    if not lines:
        return {
            "segments": [],
            "onsets": onsets,
            "duration_sec": duration_sec,
            "start_at": 0.0,
        }

    # 2) 입력 오디오(보컬 스템 우선). 실패해도 원본 사용
    vocal_path = audio_path
    if use_vocal_separation:
        try:
            if callable(globals().get("try_extract_vocals")):
                vp = globals()["try_extract_vocals"](audio_path)  # type: ignore
                if isinstance(vp, str) and Path(vp).exists():
                    vocal_path = vp
        except Exception:
            vocal_path = audio_path

    # 3) 토큰 정규화
    def norm_token(s: str) -> str:
        x = s.strip().lower()
        x = re.sub(r"[^\w\s가-힣]", " ", x)
        x = re.sub(r"\s+", " ", x)
        return x

    line_tokens: List[List[str]] = []
    for ln in lines:
        toks = [z for z in norm_token(ln).split() if z]
        line_tokens.append(toks)

    # 4) ASR: faster-whisper (CPU 전용, vad_filter=False 강제)
    words: List[dict] = []
    asr_fail = False
    try:
        from faster_whisper import WhisperModel  # type: ignore
        import os
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segs, _ = model.transcribe(
            vocal_path,
            word_timestamps=True,
            vad_filter=False,          # ← 내부 강제: onnxruntime 경로 차단
            beam_size=int(beam_size),
        )
        for seg in (segs or []):
            for w in (seg.words or []):
                tok = (w.word or "").strip()
                if tok:
                    words.append({"tok": tok, "start": float(w.start or 0.0), "end": float(w.end or 0.0)})
    except Exception:
        asr_fail = True
        words = []

    # 5) 단어 토큰 시퀀스 구성
    word_tokens: List[Tuple[str, float, float]] = []
    for w in words:
        t = norm_token(w["tok"])
        if t:
            word_tokens.append((t, float(w["start"]), float(w["end"])))

    # 6) 경량 DP 기반 라인 정렬
    aligned: List[Tuple[Optional[float], Optional[float], str, float]] = []
    if word_tokens and line_tokens and not asr_fail:
        wi = 0
        n = len(word_tokens)
        for idx, toks in enumerate(line_tokens):
            if not toks:
                aligned.append((None, None, lines[idx], 0.0))
                continue

            m = len(toks)
            best_span = None
            best_score = -1.0

            max_window = min(n, wi + 300)
            for k in range(wi, max_window):
                matched = 0
                j = k
                tpos = 0
                skips = 0
                while j < n and tpos < m:
                    if toks[tpos] == word_tokens[j][0]:
                        matched += 1
                        tpos += 1
                        j += 1
                    else:
                        j += 1
                        skips += 1
                        if skips > 2 * m:
                            break
                if matched == 0:
                    continue
                match_ratio = matched / float(m)
                span_len = j - k
                penalty = max(0.0, (span_len - matched) / float(max(1, matched)))
                score = match_ratio - 0.15 * penalty
                if score > best_score:
                    best_score = score
                    st = word_tokens[k][1]
                    et = word_tokens[j - 1][2]
                    best_span = (st, et)

            if best_span:
                st, et = best_span
                aligned.append((float(st), float(et), lines[idx], float(max(0.0, min(1.0, best_score)))))
                wi = max(wi, next((p for p in range(wi, n) if word_tokens[p][1] >= st), wi))
            else:
                aligned.append((None, None, lines[idx], 0.0))
    else:
        aligned = [(None, None, ln, 0.0) for ln in lines]

    # 7) librosa RMS VAD로 경계 클램프
    vad_bounds: List[Tuple[float, float]] = []
    try:
        import librosa
        y, sr = librosa.load(vocal_path, sr=None, mono=True)
        hop = 512
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
        thr = float(np.percentile(rms, 60) * 0.6)
        active = rms > thr
        i = 0
        while i < len(active):
            if active[i]:
                s = times[i]
                while i < len(active) and active[i]:
                    i += 1
                e = times[min(i, len(times) - 1)]
                if e - s >= 0.12:
                    vad_bounds.append((float(s), float(e)))
            else:
                i += 1
    except Exception:
        vad_bounds = []

    def clamp_to_bounds(st: float, et: float) -> Tuple[float, float]:
        if not vad_bounds:
            return st, et
        best = None
        best_d = 1e9
        for s, e in vad_bounds:
            if et < s:
                d = s - et
            elif st > e:
                d = st - e
            else:
                d = 0.0
            if d < best_d:
                best_d = d
                best = (s, e)
        if not best:
            return st, et
        s, e = best
        st2 = max(st, s)
        et2 = min(et, e) if et > 0 else e
        if et2 <= st2:
            mid = (s + e) / 2.0
            return max(s, mid - 0.2), min(e, mid + 0.2)
        return st2, et2

    # 8) 온셋 스냅
    def snap(t: float, arr: List[float], tol: float = 0.12) -> float:
        if not arr:
            return t
        cand = min(arr, key=lambda x: abs(x - t))
        if abs(cand - t) <= tol:
            return cand
        return t

    # 9) 후처리 + 폴백 배분
    segments: List[Tuple[float, float, str]] = []
    prev_end = 0.0
    for st, et, text, score in aligned:
        if isinstance(st, float) and isinstance(et, float):
            st2, et2 = clamp_to_bounds(st, et)
            st2 = max(0.0, st2 - float(start_preroll))
            if onsets:
                st2 = snap(st2, onsets, 0.12)
                et2 = snap(et2, onsets, 0.12)
            if et2 - st2 < float(min_len):
                et2 = st2 + float(min_len)
            if st2 < prev_end:
                st2 = prev_end
            if et2 > duration_sec + end_bias_sec:
                et2 = min(duration_sec + end_bias_sec, st2 + max(min_len, 0.4))
            segments.append((float(st2), float(et2), text))
            prev_end = float(et2)
        else:
            st2 = prev_end
            if onsets:
                next_on = next((x for x in onsets if x > st2 + 0.2), None)
                if next_on is not None:
                    et2 = max(st2 + min_len, min(next_on + 0.3, st2 + 1.2))
                else:
                    et2 = min(duration_sec + end_bias_sec, st2 + max(min_len, 0.8))
            else:
                et2 = min(duration_sec + end_bias_sec, st2 + max(min_len, 0.8))
            if et2 <= st2:
                et2 = st2 + float(min_len)
            segments.append((float(st2), float(et2), text))
            prev_end = float(et2)

    # 평균 길이 하한 보장
    if segments:
        lens = [max(0.0, b - a) for a, b, _ in segments]
        avg_len = float(np.mean(lens)) if lens else 0.0
        target_min = float(avg_min_sec_per_unit)
        if avg_len < target_min:
            scale = (target_min / avg_len) if avg_len > 0 else 1.0
            scale = float(min(1.6, max(1.0, scale)))
            new_segments: List[Tuple[float, float, str]] = []
            cur = 0.0
            for a, b, t in segments:
                dur = (b - a) * scale
                a2 = max(cur, a)
                b2 = min(duration_sec + end_bias_sec, a2 + dur)
                if b2 - a2 < min_len:
                    b2 = a2 + min_len
                new_segments.append((float(a2), float(b2), t))
                cur = float(b2)
            segments = new_segments

    # 첫 라인 프리롤 보정
    if segments:
        a0, b0, t0 = segments[0]
        a0 = max(0.0, a0 - float(start_preroll))
        segments[0] = (a0, b0, t0)

    # 10) 어떤 경우에도 dict로 반환 (tuple 금지)
    return {
        "segments": segments,
        "onsets": onsets,
        "duration_sec": duration_sec,
        "start_at": 0.0,
    }





def cut_audio_to_last_lyric_with_outro(
    audio_path: str,
    lyrics_text: str,
    *,
    project_dir: str | None = None,
    outro_sec: int | None = None,
    model_size: str = "medium",
    fade_out: bool = True,
    export_path: str | None = None,
) -> dict:
    """
    Whisper로 마지막 가사 종료 시각을 찾고,
    그 시점 + outro_sec까지만 오디오를 잘라 새 파일로 저장한다. 마지막 구간은 페이드아웃(선택) 적용.
    반환 예:
      {"in":..., "out":..., "t_end":..., "outro_sec":..., "duration_in":..., "duration_out":..., "whisper_used": ...}
    """
    from pathlib import Path
    import math
    import random as _random

    src = Path(audio_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"audio not found: {src}")

    # project.json의 outro_sec 우선 사용
    if outro_sec is None and project_dir:
        pj = Path(project_dir).resolve() / "project.json"
        try:
            meta = load_json(str(pj), default={}) or {}
            music_meta = meta.get("music_meta") or {}
            v = music_meta.get("outro_sec")
            if isinstance(v, (int, float)) and v > 0:
                outro_sec = int(v)
        except Exception:
            pass
    if outro_sec is None or not isinstance(outro_sec, int) or outro_sec <= 0:
        outro_sec = int(_random.randint(5, 15))

    # Whisper 동기화
    sync = sync_lyrics_with_whisper(
        audio_path=str(src),
        lyrics_text=str(lyrics_text or ""),
        model_size=model_size,
        use_vocal_separation=False,
    )
    segments = sync.get("segments") or []
    duration_in = float(sync.get("duration_sec") or 0.0)

    # 마지막 가사 종료 시각
    t_end = 0.0
    for seg in segments:
        try:
            endv = float(seg.get("end") or 0.0)
        except (TypeError, ValueError):
            endv = 0.0
        if math.isfinite(endv) and endv > t_end:
            t_end = endv

    if duration_in <= 0.0:
        duration_in = float(get_audio_duration(str(src)) or 0.0)
    if duration_in <= 0.0:
        raise RuntimeError("오디오 길이를 파악할 수 없습니다.")

    # 컷 지점: 마지막 가사 + outro_sec (원본 길이 이내)
    t_cut = t_end + float(outro_sec)
    if t_cut > duration_in:
        t_cut = duration_in

    # Pydub 컷 + 페이드아웃
    try:
        from pydub import AudioSegment  # type: ignore
    except Exception as e:
        raise RuntimeError(f"pydub import failed: {type(e).__name__}: {e}")

    seg_a = AudioSegment.from_file(str(src))
    cut_ms = int(max(1, round(t_cut * 1000)))
    seg_a = seg_a[:cut_ms]

    if fade_out:
        fade_ms = int(max(500, outro_sec * 1000))
        if fade_ms > len(seg_a):
            fade_ms = int(len(seg_a) * 0.5)
        seg_a = seg_a.fade_out(fade_ms)

    # 저장 경로
    if export_path:
        out_path = Path(export_path)
    else:
        out_path = src.with_name("vocal_cut.wav")

    fmt = out_path.suffix.lower().lstrip(".") or "wav"
    if fmt not in ("wav", "mp3", "flac", "ogg"):
        fmt = "wav"
        out_path = out_path.with_suffix(".wav")

    # ★ format은 한 번만 전달한다
    if fmt == "wav":
        # 16-bit PCM
        seg_a.export(str(out_path), format="wav", parameters=["-acodec", "pcm_s16le"])
    else:
        seg_a.export(str(out_path), format=fmt)

    return {
        "in": str(src),
        "out": str(out_path),
        "t_end": float(t_end),
        "outro_sec": int(outro_sec),
        "duration_in": float(duration_in),
        "duration_out": float(cut_ms / 1000.0),
        "whisper_used": bool(segments),
    }


from typing import List, Optional, Tuple

def _normalize_korean_text(s: str) -> str:
    """가사/ASR 매칭 안정화를 위한 아주 가벼운 정규화."""
    import re
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[.,!?~…·:;\"'()`\[\]{}<>^/\\|-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _split_lyrics_body_lines(lyrics_text: str) -> List[str]:
    """섹션 헤더([intro]/[chorus]/...)와 빈 줄을 제외한 '본문 줄'만 리스트로 반환."""
    lines: List[str] = []
    for ln in (lyrics_text or "").splitlines():
        s = (ln or "").strip()
        if not s:
            continue
        if s.startswith("[") and s.endswith("]"):
            continue
        lines.append(s)
    return lines

def _soft_match_line_in_words(
    line_norm: str,
    words: List[Tuple[str, float, float]],
    win: int = 22,
) -> Optional[Tuple[int, int, float]]:
    """
    정규화된 가사 1줄(line_norm)을 (단어,시작,끝) 시퀀스에서 유사도 최고인 구간으로 매칭.
    개선:
      - 완전 부분포함/역방향 포함시 큰 가중치
      - 토큰 Jaccard
      - 윈도우 시작이 앞쪽일수록 소폭 가산(앞줄 우선 편향)
      - 임계값 완화(0.18)
    반환: (start_idx, end_idx, score) 또는 None
    """
    if not words:
        return None
    tokens = [_normalize_korean_text(w[0]) for w in words]
    target = line_norm.strip()
    if not target:
        return None

    best = (-1.0, 0, 0)  # (score, si, ei)
    n = len(tokens)

    tgt_len = max(1, len(target.split()))
    wsize = max(4, min(win, tgt_len + 6))

    for si in range(0, n):
        ei = min(n, si + wsize)
        sub = " ".join(tokens[si:ei]).strip()
        if not sub:
            continue

        # 포함 점수(강화)
        contain_score = 0.0
        if target and sub:
            if target in sub:
                contain_score = 1.0
            elif sub in target:
                contain_score = 0.55

        # Jaccard
        set_a = set(target.split()); set_b = set(sub.split())
        inter = len(set_a & set_b); union = len(set_a | set_b) or 1
        jacc = inter / union

        # 앞쪽 윈도우 보너스(초반 줄 성공률 개선)
        pos_bias = 1.0 - (si / max(1.0, n - 1))  # 1.0(앞) → 0.0(뒤)
        pos_bonus = 0.08 * pos_bias

        score = 0.35 * contain_score + 0.57 * jacc + pos_bonus
        if score > best[0]:
            best = (score, si, ei)

    if best[0] < 0.18:
        return None
    return (best[1], best[2], best[0])

from typing import List, Dict, Any, Tuple  # 파일 상단에 이미 있다면 중복 무시

def _interpolate_missing_lines(
    aligned: List[Dict[str, Any]],
    *,
    min_start_sec: float = 0.0,
    guard: float = 0.12,
    min_line_sec: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    앞/뒤로 이미 매칭된 줄 사이에 있는 (start/end=None) 줄들을
    라인 길이(문자 수) 비례로 나눠 채운다.
    - 첫 구간: min_start_sec ~ 첫 매칭 start 사이를 채움(인트로 제외 후 남은 부분)
    - 마지막 구간: 마지막 매칭 이후는 보간하지 않음(불필요 확장 방지)
    반환: 보간된 aligned (원본 리스트 수정)
    """

    def _is_num(x: Any) -> bool:
        return isinstance(x, (int, float))

    # known: (index, start, end) — 타입 안전하게 구축
    known: List[Tuple[int, float, float]] = []
    for i, it in enumerate(aligned):
        s_val = it.get("start")
        e_val = it.get("end")
        if _is_num(s_val) and _is_num(e_val):
            s = float(s_val)  # 이제 타입체커 OK
            e = float(e_val)
            if e >= s:
                known.append((i, s, e))

    if not known:
        # 전부 None이면 보간 불가
        return aligned

    # 선두 구간(인트로 이후 ~ 첫 매칭 전)
    first_i, first_s, _first_e = known[0]
    lead_start = float(max(min_start_sec, 0.0))
    lead_end = float(max(first_s - guard, lead_start))
    if first_i > 0 and (lead_end - lead_start) >= (min_line_sec * 0.5):
        lengths = [max(1, len(str(aligned[k].get("line") or ""))) for k in range(0, first_i)]
        total = float(sum(lengths)) if lengths else 1.0
        t = lead_start
        for k, L in zip(range(0, first_i), lengths):
            span = max(min_line_sec, (lead_end - lead_start) * (float(L) / total))
            s = t
            e = min(lead_end, t + span)
            if e < s:
                e = s
            aligned[k]["start"] = float(s)
            aligned[k]["end"] = float(e)
            aligned[k]["score"] = float(aligned[k].get("score") or 0.0)
            t = e

    # 중간 구간들
    for (a_idx, _a_s, a_e), (b_idx, b_s, _b_e) in zip(known, known[1:]):
        if b_idx - a_idx <= 1:
            continue  # 사이에 비어있는 줄 없음
        gap_s = float(max(a_e + guard, _a_s if '_a_s' in locals() else a_e))  # 안전
        gap_s = max(gap_s, a_e + guard)
        gap_e = float(max(gap_s, b_s - guard))
        mids = list(range(a_idx + 1, b_idx))
        if (gap_e - gap_s) < (min_line_sec * 0.5):
            continue
        lengths = [max(1, len(str(aligned[k].get("line") or ""))) for k in mids]
        total = float(sum(lengths)) if lengths else 1.0
        t = gap_s
        for k, L in zip(mids, lengths):
            span = max(min_line_sec, (gap_e - gap_s) * (float(L) / total))
            s = t
            e = min(gap_e, t + span)
            if e < s:
                e = s
            aligned[k]["start"] = float(s)
            aligned[k]["end"] = float(e)
            aligned[k]["score"] = float(aligned[k].get("score") or 0.0)
            t = e

    return aligned



def align_lyrics_per_line(
    audio_path: str,
    lyrics_text: str,
    *,
    model_size: str = "medium",
    snap_onsets: Optional[List[float]] = None,
    snap_window_sec: float = 0.15,
) -> List[Dict[str, object]]:
    """
    Whisper(가능하면 word timestamps)로 ASR → 가사 '한 줄'마다 [start, end] 추정.
    개선점:
      - project.json의 music_meta.intro_sec 만큼 '최소 시작 시각'으로 사용하여 인트로를 후보에서 제외
      - 라인 순서 단조 증가(앞 라인 end 이후에서만 다음 라인 탐색)
    반환: [{'line': '...', 'start': 12.34, 'end': 15.67, 'score': 0.83, 'word_level': True/False}, ...]
    """
    from pathlib import Path
    import os, json
    src = Path(audio_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"audio not found: {src}")

    # 0) project.json에서 intro_sec 가져와 '최소 시작 시각'으로 사용
    proj_dir = src.parent
    min_start_sec = 0.0
    pj = proj_dir / "project.json"
    if pj.exists():
        try:
            meta = json.loads(pj.read_text(encoding="utf-8"))
            mm = meta.get("music_meta") or {}
            v = mm.get("intro_sec")
            if isinstance(v, (int, float)) and v > 0:
                # 인트로는 무가사로 가정 → 그 이전은 매칭 금지
                min_start_sec = float(v)
        except Exception:
            pass

    body_lines = _split_lyrics_body_lines(lyrics_text)
    if not body_lines:
        return []

    # ==== ASR 실행 (word-level 우선; CPU 강제) ====
    words: List[Tuple[str, float, float]] = []
    used_word_level = False
    try:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        from faster_whisper import WhisperModel  # type: ignore
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, _info = model.transcribe(str(src), vad_filter=False, word_timestamps=True)
        for seg in segments:
            for w in (seg.words or []):
                if w.word and (w.start is not None) and (w.end is not None):
                    words.append((w.word, float(w.start), float(w.end)))
        used_word_level = True
    except Exception:
        # fallback: 기존 sync 결과 세그먼트를 거대 단어로 취급
        sync = sync_lyrics_with_whisper(
            audio_path=str(src),
            lyrics_text=str(lyrics_text or ""),
            model_size=model_size,
            use_vocal_separation=False,
        )
        segs = sync.get("segments") or []
        for s in segs:
            txt = _normalize_korean_text(str(s.get("text") or ""))
            t0 = float(s.get("start") or 0.0)
            t1 = float(s.get("end") or 0.0)
            if txt:
                words.append((txt, t0, t1))
        used_word_level = False

    # 후보 단어에서 인트로 이전 구간 제거
    if min_start_sec > 0.0 and words:
        words = [(t, a, b) for (t, a, b) in words if b >= min_start_sec]

    aligned: List[Dict[str, object]] = []
    tokens_norm = [(_normalize_korean_text(w[0]), w[1], w[2]) for w in words]

    # 단조 증가(순서 고정) 매칭: 앞 라인의 종료시각 이후에서만 탐색
    cursor_time = float(min_start_sec)
    guard = 0.15  # 앞 라인 end 직후 보호구간

    def _find_start_index_from_time(ts: List[Tuple[str, float, float]], tmin: float) -> int:
        # 단어 시작시각이 tmin 이상인 첫 인덱스
        for i, (_t, a, _b) in enumerate(ts):
            if a >= tmin:
                return i
        return len(ts)

    si_hint = _find_start_index_from_time(tokens_norm, cursor_time)

    for raw in body_lines:
        line_norm = _normalize_korean_text(raw)
        # 윈도우를 '현재 커서 이후'로 제한
        slice_tokens = tokens_norm[si_hint:] if si_hint < len(tokens_norm) else []
        if slice_tokens:
            matched = _soft_match_line_in_words(line_norm, [(t, a, b) for t, a, b in slice_tokens], win=22)
        else:
            matched = None

        if matched is None:
            aligned.append({"line": raw, "start": None, "end": None, "score": 0.0, "word_level": bool(used_word_level)})
            # 다음 줄도 같은 힌트에서 다시 시도
            continue

        # slice 기준 인덱스를 전체 기준으로 변환
        si_local, ei_local, score = matched
        si = si_hint + si_local
        ei = si_hint + ei_local
        si = max(0, min(si, len(tokens_norm) - 1))
        ei = max(si + 1, min(ei, len(tokens_norm)))
        t0 = float(tokens_norm[si][1]); t1 = float(tokens_norm[ei - 1][2])

        # onsets 스냅(선택)
        if snap_onsets:
            def _snap(t: float) -> float:
                best = None
                for x in snap_onsets:
                    if abs(x - t) <= snap_window_sec:
                        if (best is None) or (abs(x - t) < abs(best - t)):
                            best = x
                return best if best is not None else t
            t0 = _snap(t0); t1 = _snap(t1)

        if t1 < t0:
            t1 = t0

        aligned.append({"line": raw, "start": t0, "end": t1, "score": float(score), "word_level": bool(used_word_level)})

        # 다음 라인은 이번 라인 종료 이후에서만 탐색
        cursor_time = t1 + guard
        si_hint = _find_start_index_from_time(tokens_norm, cursor_time)

    # 마지막 안전 보정(겹침 제거)
    last_end = float(min_start_sec)
    for it in aligned:
        s = it.get("start")
        e = it.get("end")
        if isinstance(s, (int, float)) and isinstance(e, (int, float)):
            if s < last_end:
                s = last_end
            if e < s:
                e = s
            it["start"], it["end"] = float(s), float(e)
            last_end = float(e)

    # ★ 빈 줄 보간으로 누락된 라인 채우기
    aligned = _interpolate_missing_lines(aligned, min_start_sec=float(min_start_sec))

    return aligned


def analyze_and_cut_project(
    project_dir: str,
    *,
    model_size: str = "medium",
    snap_window_sec: float = 0.15,
) -> dict:
    """
    1) project_dir 안의 vocal.wav, lyrics.txt(or project.json의 lyrics), music_analysis_preview.json(onsets)
    2) 줄별 정합(align_lyrics_per_line) → 마지막 줄 end 추출
    3) project.json의 music_meta.outro_sec (없으면 5~15 랜덤)로 컷 지점 = last_end + outro_sec
    4) Pydub 페이드아웃 적용하여 vocal_cut.wav 저장
    반환: 요약 딕셔너리
    """
    from pathlib import Path
    import json

    proj = Path(project_dir).resolve()
    audio_in = proj / "vocal.wav"
    if not audio_in.exists():
        raise FileNotFoundError(f"audio not found: {audio_in}")

    # 가사 가져오기: lyrics.txt > project.json
    lyrics_text = ""
    lyrics_txt = proj / "lyrics.txt"
    if lyrics_txt.exists():
        lyrics_text = lyrics_txt.read_text(encoding="utf-8", errors="ignore")
    else:
        pj = proj / "project.json"
        if pj.exists():
            try:
                meta = json.loads(pj.read_text(encoding="utf-8"))
                lyrics_text = str(meta.get("lyrics") or "")
            except Exception:
                lyrics_text = ""
    if not lyrics_text.strip():
        raise RuntimeError("lyrics not found in project")

    # onsets(선택)
    onsets = []
    preview_json = proj / "music_analysis_preview.json"
    if preview_json.exists():
        try:
            preview = json.loads(preview_json.read_text(encoding="utf-8"))
            onsets = preview.get("onsets") or []
        except Exception:
            onsets = []

    # 줄별 정합
    aligned = align_lyrics_per_line(
        audio_path=str(audio_in),
        lyrics_text=lyrics_text,
        model_size=model_size,
        snap_onsets=onsets,
        snap_window_sec=snap_window_sec,
    )

    # 마지막 줄 end
    last_end = 0.0
    for row in aligned:
        e = row.get("end")
        if isinstance(e, (int, float)) and e > last_end:
            last_end = float(e)

    # outro_sec 결정: project.json의 music_meta.outro_sec 우선
    outro_sec = None
    pj = proj / "project.json"
    if pj.exists():
        try:
            meta = json.loads(pj.read_text(encoding="utf-8"))
            mm = meta.get("music_meta") or {}
            v = mm.get("outro_sec")
            if isinstance(v, (int, float)) and v > 0:
                outro_sec = int(v)
        except Exception:
            pass

    # 컷 + 페이드아웃 실행 (내부에서 outro_sec 없으면 5~15 랜덤)
    cut_info = cut_audio_to_last_lyric_with_outro(
        audio_path=str(audio_in),
        lyrics_text=lyrics_text,
        project_dir=str(proj),
        outro_sec=outro_sec,
        model_size=model_size,
        fade_out=True,
        export_path=None,  # proj/vocal_cut.wav
    )

    # 정합 결과 저장(검증용)
    aligned_path = proj / "lyrics_aligned.json"
    try:
        aligned_path.write_text(json.dumps(aligned, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    return {
        "aligned_path": str(aligned_path),
        "aligned_count": len(aligned),
        "last_end": last_end,
        "cut": cut_info,
    }







