# -*- coding: utf-8 -*-
"""
audio_sync.py
- 오디오 길이와 (있으면) project.json의 가사를 이용해 타임라인 생성
- 결과는 segments.json / scene.json을 '항상' 오디오 파일 폴더에 저장(save=True일 때)
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast, Union
import tempfile
# ───────────────────────── utils 안전 import ─────────────────────────
try:
    from app.utils import audio_duration_sec, save_json, load_json, ensure_dir, sanitize_title
except (ImportError, ModuleNotFoundError):
    # 최소 동작 대체
    import json as _json
    from pathlib import Path as _Path

    def load_json(p: _Path, default: Optional[Any] = None) -> Union[Dict[str, Any], List[Any], None]:
        """
        [수정됨] JSON 파일 로드 (대체 구현). 객체(dict) 또는 배열(list) 반환 가능.
        """
        try:
            with open(p, "r", encoding="utf-8") as f:
                content: Any = _json.load(f)
                # dict 또는 list 타입만 유효한 JSON 내용으로 간주
                if isinstance(content, (dict, list)):
                    return content
                else:
                    return default # 그 외 타입은 default 반환
        except (FileNotFoundError, OSError, _json.JSONDecodeError):
            # 파일 없거나 JSON 오류 시 default 반환
            return default
        except Exception:
            # 예상치 못한 오류 발생 시에도 default 반환 (안정성)
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
    from app.video_build import build_image_json as _build_image_json, build_movie_json as _build_movie_json
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


from typing import Any, Dict, List

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


# ───────────────────────── 공개 API: 분석 ─────────────────────────


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
    import json
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
            from app.video_build import normalize_to_v11, validate_story_v11_dict
        except (ImportError, ModuleNotFoundError):
            from video_build import normalize_to_v11, validate_story_v11_dict  # type: ignore

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
import os, math

# ─────────────────────────────────────────────────────────────
# 0) 유틸: 오디오 길이 견고 획득(중앙값, 3배 튐 방지)
# ─────────────────────────────────────────────────────────────
def _probe_duration_ffprobe(path: str) -> float:
    import json
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







# ─────────────────────────────────────────────────────────────
# 5) 공식 가사 라인과 Whisper 단어 정렬
# ─────────────────────────────────────────────────────────────
def _norm_txt(s: str) -> str:
    s = re.sub(r"\s+", "", (s or ""))
    return s


def align_words_to_lyrics_lines(official_lyrics: str, words: List[Tuple[float, float, str]]) -> List[
    Tuple[Optional[float], Optional[float], str]]:
    """
    [개선된 버전] rapidfuzz를 이용한 편집 거리로 더 빠르고 정확하게 가사 라인과 Whisper 단어 시퀀스를 정렬합니다.
    - 슬라이딩 윈도우 방식으로 각 가사 라인에 가장 적합한 단어 시퀀스 구간을 탐색합니다.
    - 순차적 검색(word_cursor)으로 반복/유사 구절의 순서가 꼬이지 않도록 보장합니다.
    결과: (start, end, line_text) 리스트. 실패 라인은 (None, None, line_text)
    """
    import re
    from typing import List, Tuple, Optional

    # --- 내부 헬퍼: 텍스트 정규화 ---
    def _normalize_text_for_matching(text: str) -> str:
        s = (text or "").lower().strip()
        s = re.sub(r"[^a-z0-9가-힣\s]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    # --- rapidfuzz 라이브러리 로드 (없으면 폴백) ---
    try:
        from rapidfuzz.distance import Levenshtein
    except ImportError:
        print("[WARN] rapidfuzz 라이브러리가 없어 정렬 기능이 제한됩니다. 'pip install rapidfuzz'를 권장합니다.")
        lines_fallback = [ln.strip() for ln in (official_lyrics or "").splitlines() if ln.strip()]
        return [(None, None, ln) for ln in lines_fallback]

    # --- 메인 로직 ---
    lines_original = [ln.strip() for ln in (official_lyrics or "").splitlines() if ln.strip()]
    if not lines_original or not words:
        return [(None, None, ln) for ln in lines_original]

    # ASR 단어 준비
    asr_words_norm = [_normalize_text_for_matching(w[2]) for w in words]

    aligned: List[Tuple[Optional[float], Optional[float], str]] = []
    word_cursor = 0  # ASR 단어 시퀀스에서 검색 시작 위치

    for line_text in lines_original:
        line_norm = _normalize_text_for_matching(line_text)
        if not line_norm:
            aligned.append((None, None, line_text))
            continue

        line_words_count = len(line_norm.split())
        if line_words_count == 0:
            aligned.append((None, None, line_text))
            continue

        # 최적의 시작점과 윈도우 크기 탐색
        best_sim = -1.0
        best_match_info = {'start_idx': -1, 'win_size': -1}

        # 탐색 범위 제한: 너무 멀리 떨어진 단어는 보지 않음 (성능 및 정확도 향상)
        search_window_start = word_cursor
        search_window_end = min(len(asr_words_norm), word_cursor + 40)  # 현재 위치에서 40단어 앞까지 탐색

        for i in range(search_window_start, search_window_end):
            # 윈도우 크기를 가사 라인 단어 수에 약간의 여유를 줌 ( +/- 2 )
            min_win = max(1, line_words_count - 2)
            max_win = line_words_count + 3  # 여유를 조금 더 줌

            for win_size in range(min_win, max_win + 1):
                start_index = i
                end_index = start_index + win_size
                if end_index > len(asr_words_norm):
                    continue

                window_text = " ".join(asr_words_norm[start_index: end_index])
                sim = Levenshtein.normalized_similarity(line_norm, window_text)

                if sim > best_sim:
                    best_sim = sim
                    best_match_info['start_idx'] = start_index
                    best_match_info['win_size'] = win_size

        # 결과 처리
        if best_sim > 0.4:  # 유사도 40% 이상일 때만 채택
            start_word_idx = best_match_info['start_idx']
            end_word_idx = start_word_idx + best_match_info['win_size'] - 1

            if start_word_idx < len(words) and end_word_idx < len(words):
                start_time = words[start_word_idx][0]
                end_time = words[end_word_idx][1]
                aligned.append((start_time, end_time, line_text))
                word_cursor = end_word_idx + 1  # 다음 검색을 위해 커서 이동
            else:
                aligned.append((None, None, line_text))
        else:
            aligned.append((None, None, line_text))

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




# ------------------------------------------------------------
# 2) 가사 싱크: 보컬분리(옵션) → 전사 → 정렬 → 폴백 배분
# ------------------------------------------------------------
# audio_sync.py 파일에서 이 함수를 찾아 아래 내용으로 전체를 교체하세요.

# audio_sync.py









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
    return best[1], best[2], best[0]  # ← 중복 괄호 제거


from typing import List, Dict, Any  # 파일 상단에 이미 있다면 중복 무시


def _interpolate_missing_lines(
        aligned: List[Dict[str, Any]],
        *,
        min_start_sec: float = 0.0,
        guard: float = 0.12,
        min_line_sec: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    [개선된 버전] 앞/중간뿐만 아니라, 마지막에 남은 미할당 라인들도 보간(Extrapolate)합니다.
    - (New) 성공적으로 할당된 라인들의 평균 길이를 계산합니다.
    - (New) 마지막으로 할당된 라인 뒤에 남은 라인들을 이 평균 길이를 이용해 순차적으로 배치합니다.
    - 기존의 앞/중간 보간 기능은 그대로 유지합니다.
    """
    from typing import List, Any, Tuple

    def _is_num(x: Any) -> bool:
        return isinstance(x, (int, float))

    known: List[Tuple[int, float, float]] = []
    for i, item in enumerate(aligned):
        s_val, e_val = item.get("start"), item.get("end")
        if _is_num(s_val) and _is_num(e_val) and float(e_val) >= float(s_val):
            known.append((i, float(s_val), float(e_val)))

    if not known:
        # 모든 라인이 미할당 상태면, 최소 길이로 순차 배치
        t = float(min_start_sec)
        for item in aligned:
            item["start"], item["end"], item["score"] = t, t + min_line_sec, 0.0
            t += min_line_sec
        return aligned

    # --- 선두 구간 (첫 매칭 전) ---
    first_i, first_s, _ = known[0]
    if first_i > 0:
        lead_start = float(min_start_sec)
        lead_end = max(lead_start, first_s - guard)
        if lead_end > lead_start:
            lengths = [max(1, len(str(aligned[k].get("line", "")))) for k in range(first_i)]
            total_len = float(sum(lengths)) or 1.0
            t = lead_start
            for k, length_val in zip(range(first_i), lengths):
                span = max(min_line_sec, (lead_end - lead_start) * (length_val / total_len))
                aligned[k]["start"], aligned[k]["end"], aligned[k]["score"] = t, min(lead_end, t + span), 0.0
                t = aligned[k]["end"]

    # --- 중간 구간 (매칭된 라인 사이) ---
    for (a_idx, _, a_e), (b_idx, b_s, _) in zip(known, known[1:]):
        if b_idx - a_idx <= 1: continue
        gap_s = max(a_e + guard, a_e)
        gap_e = max(gap_s, b_s - guard)
        if gap_e > gap_s:
            mids = list(range(a_idx + 1, b_idx))
            lengths = [max(1, len(str(aligned[k].get("line", "")))) for k in mids]
            total_len = float(sum(lengths)) or 1.0
            t = gap_s
            for k, length_val in zip(mids, lengths):
                span = max(min_line_sec, (gap_e - gap_s) * (length_val / total_len))
                aligned[k]["start"], aligned[k]["end"], aligned[k]["score"] = t, min(gap_e, t + span), 0.0
                t = aligned[k]["end"]

    # --- [NEW] 후미 구간 (마지막 매칭 후) ---
    last_i, _, last_e = known[-1]
    if last_i < len(aligned) - 1:
        # 성공한 라인들의 평균 길이 계산
        durations = [e - s for _, s, e in known if e > s]
        avg_dur = float(np.mean(durations)) if durations else min_line_sec
        avg_dur_to_use = max(min_line_sec, avg_dur)

        trail_start = float(last_e + guard)
        t = trail_start
        for k in range(last_i + 1, len(aligned)):
            aligned[k]["start"], aligned[k]["end"], aligned[k]["score"] = t, t + avg_dur_to_use, 0.0
            t += avg_dur_to_use

    return aligned











# === [MERGED FROM music_gen.py] ===
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

from typing import Optional, List, Tuple, Iterable
from pathlib import Path
import random
import requests

from typing import Any

# settings 상수(대기/폴링 주기)

# ── settings / utils 유연 임포트 ───────────────────────────────────────────────
try:
    # 패키지 실행
    from app import settings as settings   # 소문자 별칭
    from app.settings import (
        BASE_DIR, COMFY_HOST, DEFAULT_HOST_CANDIDATES,
        ACE_STEP_PROMPT_JSON, FFMPEG_EXE, FINAL_OUT
    )
    from app.utils import load_json, save_json, sanitize_title, effective_title, save_to_user_library
except ImportError:
    # 단독 실행
    import settings as settings
    from settings import (  # type: ignore
        BASE_DIR, COMFY_HOST, DEFAULT_HOST_CANDIDATES,
        ACE_STEP_PROMPT_JSON, FFMPEG_EXE, FINAL_OUT
    )
    from utils import load_json, save_json, sanitize_title, effective_title, save_to_user_library  # type: ignore

S = settings  # noqa: N816
# --- ACE-Step 대기/폴링 기본값 & 헬퍼 (설정에 없으면 이 값 사용) ---
# --- ACE-Step 대기/폴링 기본값 & 헬퍼 ---
_DEFAULT_ACE_WAIT_TIMEOUT_SEC = 900.0   # 15분
_DEFAULT_ACE_POLL_INTERVAL_SEC = 2.0    # 2초

def _ace_wait_timeout_sec():
    try:
        from app import settings as _s
    except (ImportError, ModuleNotFoundError):
        _s = None
    return (
        (getattr(_s, "ACE_STEP_WAIT_TIMEOUT_SEC", None) if _s else None)
        or (getattr(_s, "ACE_WAIT_TIMEOUT_SEC", None) if _s else None)
        or _DEFAULT_ACE_WAIT_TIMEOUT_SEC
    )
def _ace_poll_interval_sec():
    try:
        from app import settings as _s
    except (ImportError, ModuleNotFoundError):
        _s = None
    return (
        (getattr(_s, "ACE_STEP_POLL_INTERVAL_SEC", None) if _s else None)
        or (getattr(_s, "ACE_POLL_INTERVAL_SEC", None) if _s else None)
        or _DEFAULT_ACE_POLL_INTERVAL_SEC
    )



# ─────────────────────────────  ────────────────────────────────────


# ───────────────────────────── 디버그 ────────────────────────────────────
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



# ───────────────────────────── HTTP 유틸 ────────────────────────────────────

def _http_get(base: str, path: str, timeout: int = 30, params: Optional[dict] = None) -> requests.Response:
    return requests.get(base.rstrip("/") + path, params=params or {}, timeout=timeout)

def _probe_server(base: str, timeout: int = 3) -> bool:
    for p in ("/view", "/history"):
        try:
            r = requests.get(base.rstrip("/") + p, timeout=timeout)
            if r.status_code == 200:
                return True
        except Exception:
            pass
    return False


# ───────────────────────────── 워크플로 로더 ──────────────────────────────────
def _load_workflow_graph(json_path: str | Path) -> dict:
    """
    ComfyUI용 워크플로 JSON을 로드해, /prompt에 바로 넣을 수 있는
    '노드 딕셔너리' 형태로 정규화해서 반환한다.

    추가 보정(기능 불변, 참조만 정리):
      - 문자열 형태 노드 참조 '#74' → '74'
      - 잘못된 3원소 참조 예: ["문자태그", "17", 0] → ["17", 0]
      - {"nodes":[...]} / {id→node} 두 포맷 모두 정규화
    """
    from pathlib import Path
    import json, re
    from typing import Any

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





# ───────────────────────────── 서버/노드 유틸 ──────────────────────────────────
def _choose_host() -> str:
    """settings의 COMFY_HOST 우선, 그 다음 DEFAULT_HOST_CANDIDATES에서 살아있는 서버를 선택."""
    cand: List[str] = []
    if COMFY_HOST:
        cand.append(COMFY_HOST)
    for c in (getattr(S, "DEFAULT_HOST_CANDIDATES", DEFAULT_HOST_CANDIDATES) or []):
        if c and c not in cand:
            cand.append(c)
    for host in cand:
        if host and _probe_server(host):
            return host
    return cand[0] if cand else "http://127.0.0.1:8188"




# ───────────────────────────── 태그/노드 유틸 ──────────────────────────────────
def _collect_effective_tags(meta: dict) -> List[str]:
    """
    project.json에서 실제 주입할 태그 리스트:
    - auto_tags == True : ace_tags + tags_in_use
    - auto_tags == False: manual_tags
    """
    if meta.get("auto_tags", True):
        tags = list(meta.get("ace_tags") or [])
        tags = list(dict.fromkeys(tags + (meta.get("tags_in_use") or [])))
        return tags
    else:
        return list(meta.get("manual_tags") or [])

def _graph(prompt) -> dict:
    """
    프롬프트 JSON이 다양한 스키마(dict / {'prompt':{}} / {'nodes':[...]})로 올 수 있으니
    항상 dict를 돌려주도록 방어적으로 정규화한다.
    """
    # 1) 없으면 빈 dict
    if prompt is None:
        return {}

    # 2) 최상위에 {"prompt": {...}} 형태면 내부만
    if isinstance(prompt, dict):
        inner = prompt.get("prompt")
        if isinstance(inner, dict):
            prompt = inner

    # 3) {"nodes":[{id:.., class_type:..}, ...]} → {id(str): node(dict)}
    if isinstance(prompt, dict) and isinstance(prompt.get("nodes"), list):
        g = {}
        for n in prompt["nodes"]:
            if isinstance(n, dict):
                nid = str(n.get("id") or "")
                if nid:
                    g[nid] = {k: v for k, v in n.items() if k != "id"}
        return g

    # 4) dict면 그대로, 아니면 빈 dict
    return prompt if isinstance(prompt, dict) else {}


def _find_nodes_by_class_names(graph: dict, class_names: Iterable[str]) -> List[Tuple[str, dict]]:
    names = set(class_names)
    res: List[Tuple[str, dict]] = []
    for nid, node in (graph or {}).items():
        if isinstance(node, dict) and node.get("class_type") in names:
            res.append((str(nid), node))
    return res

def _find_nodes_by_class_contains(graph: dict, needle: str) -> list[tuple[str, dict]]:
    needle = (needle or "").lower()
    out = []
    for nid, node in (graph or {}).items():
        ct = str(node.get("class_type", "")).lower()
        if needle and needle in ct:
            out.append((str(nid), node))
    return out

def rewrite_prompt_audio_format(json_path: Path, desired_fmt: str) -> Tuple[int, str]:
    """
    워크플로 파일 안의 SaveAudio* 노드를 desired_fmt('wav'|'mp3'|'opus')로 바꿔 저장.
    """
    desired_fmt = (desired_fmt or "mp3").lower().strip()
    if desired_fmt not in ("mp3", "wav", "opus"):
        desired_fmt = "mp3"
        # return 0
    try:
        data = load_json(json_path, {}) or {}
    except Exception as e:
        return 0, f"프롬프트 JSON 로드 실패: {e}"

    g = _graph(data)
    if not isinstance(g, dict) or not g:
        return 0, f"프롬프트 JSON 형식 오류: {json_path}"

    targets = _find_nodes_by_class_contains(g, "saveaudio")
    if not targets:
        return 0, "SaveAudio* 노드를 찾지 못했습니다."

    changed = 0
    for _nid, node in targets:
        ins = node.setdefault("inputs", {})
        if desired_fmt == "wav":
            node["class_type"] = "SaveAudioWAV"
            ins.pop("quality", None)
            changed += 1
        elif desired_fmt == "opus":
            node["class_type"] = "SaveAudioOpus"
            ins.pop("quality", None)
            changed += 1
        else:  # mp3
            node["class_type"] = "SaveAudioMP3"
            q = str(ins.get("quality", "")).strip().lower()
            if q not in ("v0", "128k", "320k"):
                ins["quality"] = "320k"
            changed += 1

    if changed:
        try:
            save_json(json_path, data)
            return changed, f"{json_path.name} 저장 노드 {changed}개를 '{desired_fmt}'로 갱신."
        except Exception as e:
            return 0, f"프롬프트 JSON 저장 실패: {e}"
    return 0, "변경 사항 없음."


def _rand_seed() -> int:
    return random.randint(1, 2_147_483_646)


# ────────────────────── 저장 노드 '적응' 로직(핵심) ────────────────────────────
# --- NEW: 서버에 설치된 노드 클래스 목록 가져오기 (없으면 빈 set) ---




# ───────────────────────────── 결과 파일 처리 ─────────────────────────────────
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



# ───────────────────────────── 제출/폴링 ──────────────────────────────────────


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
    import time
    import requests
    from typing import Any, Dict, cast

    # 안전한 기본값 결정
    timeout_val = float(timeout if timeout is not None else _ace_wait_timeout_sec())
    poll_val = float(poll if poll is not None else _ace_poll_interval_sec())

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








# ───────────────────────────── 메인 함수 ──────────────────────────────────────

from pathlib import Path

# ─────────────────────────────
# 필요한 유틸이 이 파일에 이미 있다고 가정:
# - _dlog, load_json, save_json, effective_title, sanitize_title
# - _load_workflow_graph(ACE_STEP_PROMPT_JSON), _choose_host
# - _apply_save_audio_node_adaptively, _find_nodes_by_class_names, _find_nodes_by_class_contains
# - _ensure_filename_prefix, _submit_and_wait, _rand_seed
# - save_to_user_library
# - settings as S (FFMPEG_EXE, FINAL_OUT, AUDIO_SAVE_FORMAT 등)
# ─────────────────────────────

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

##########################################################################
#######################음악생성 패치##########################################
##########################################################################

# audio_sync.py (ensure these imports are present at the top)
import re
import requests
import json
import uuid
import time
from pathlib import Path
from typing import Optional, List, Dict, Callable, Any # Tuple 제거됨
import subprocess
import numpy as np
import shutil # Added for _ensure_vocal_wav

# (Other imports and functions in audio_sync.py remain the same)

# --- Settings alias (ensure S is defined) ---
# (v11과 동일)
try: from app import settings as S
except ImportError:
    try: import settings as S # type: ignore[no-redef]
    except ImportError:
        class DummySettings: # Fallback settings
            ACE_STEP_PROMPT_JSON = "jsons/ace_step_1_t2m.json"; JSONS_DIR = "jsons"; FFMPEG_EXE = "ffmpeg"; BASE_DIR = "."; FINAL_OUT = "."; COMFY_HOST = "http://127.0.0.1:8188"; DEFAULT_HOST_CANDIDATES: List[str] = []; MASTER_TARGET_I = -12.0; MASTER_TARGET_TP = -1.0; MASTER_TARGET_LRA = 11.0
        S = DummySettings(); print("[WARN] Settings module not found, using dummy.")

# audio_sync.py 파일에서 이 함수를 찾아 아래 내용으로 전체를 교체하세요.
def generate_music_with_acestep(
        project_dir: str,
        *,
        on_progress: Optional[Callable[[dict], None]] = None,
        target_seconds: int | None = None,
) -> str:
    """
    ComfyUI(ACE-Step) 단일 트랙 음악 생성 — project.json 단일 소스 정책.
    - [수정 v12] API 결과 추출 노드 ID(75) 재확인 및 파싱 강화, 세미콜론 제거 완료:
        * 청크 변환 API 결과 추출 시, 로그에서 확인된 Node 75(PreviewAny)의 'outputs.text[0]'을 정확히 파싱 시도.
        * 폴링 로직 및 history 파싱 부분 안정성 강화.
        * 코드 내 모든 후행 세미콜론 제거 완료.
    - lyrics_lls_after 복원 등 이전 수정사항 유지.
    - 모든 코딩 규칙 준수.
    """
    # --- 내부 헬퍼 함수 ---
    def notify(stage: str, **kw: Any) -> None: # (v11과 동일)
        if on_progress:
            try: info: Dict[str, Any] = {"stage": stage}; info.update(kw); on_progress(info)
            except (TypeError, ValueError, RuntimeError): pass

    def _iter_nodes(graph: dict) -> Any: # (v11과 동일)
        if not isinstance(graph, dict): return
        nodes_list = graph.get("nodes")
        if isinstance(nodes_list, list):
            for nobj in nodes_list:
                if isinstance(nobj, dict): yield nobj
        else:
            for _k, nobj in graph.items():
                if isinstance(nobj, dict) and "class_type" in nobj: yield nobj
                elif isinstance(nobj, dict) and "prompt" in nobj and isinstance(nobj["prompt"], dict):
                     for _pk, p_nobj in nobj["prompt"].items():
                          if isinstance(p_nobj, dict) and "class_type" in p_nobj: yield p_nobj

    def _force_wav_save_for_this_graph(graph: dict, *, proj_dir: Path, filename_prefix: str) -> str: # (v11과 동일)
        for nobj in _iter_nodes(graph):
            ct_local = str(nobj.get("class_type", "")).lower()
            if ct_local.startswith("saveaudio"):
                inputs_map_local = nobj.setdefault("inputs", {})
                inputs_map_local["filename_prefix"] = filename_prefix
                if "output_path" in inputs_map_local: inputs_map_local["output_path"] = str(proj_dir)
                for base_key in ("basename", "base_filename", "filename"):
                    if base_key in inputs_map_local: inputs_map_local[base_key] = "vocal"
                for wav_key, wav_val in (("format", "wav"), ("container", "wav"), ("codec", "pcm_s16le")):
                     if wav_key in inputs_map_local: inputs_map_local[wav_key] = wav_val
                if "sample_rate" in inputs_map_local: inputs_map_local.setdefault("sample_rate", 44100)
                for bd_key in ("bit_depth", "bitdepth", "bits"):
                     if bd_key in inputs_map_local: inputs_map_local[bd_key] = 16
        return ".wav"

    # --- Comfy API 호출 로직 (generate_music_with_acestep 함수 내부에 통합) ---
    def _call_comfy_api_for_ko_chunk(text_chunk: str, host_address: str) -> str:
        """
        [수정됨 v12] 한글 덩어리를 전용 워크플로우로 변환 시도. 결과 추출 노드 ID=75 재확인 및 파싱 강화.
        """
        lyrics_node_id = "74"
        # ★★ 결과 추출 대상 노드 ID 재확인 및 명시 ★★
        result_node_id = "75" # PreviewAny 노드
        save_text_node_id = "79" # SaveText 노드 (워크플로우에는 존재)

        transformed_result = text_chunk
        api_workflow_path: Optional[Path] = None

        try: # 워크플로우 로드 (v11과 동일)
            jsons_dir_val = getattr(S, "JSONS_DIR", "jsons")
            api_workflow_path = Path(jsons_dir_val) / "ace_step_text_change.json"
            if not api_workflow_path.exists():
                api_workflow_path = Path("jsons") / "ace_step_text_change.json"
                if not api_workflow_path.exists(): raise FileNotFoundError("API workflow not found")
            with open(api_workflow_path, 'r', encoding='utf-8') as f_api_wf: api_workflow_graph = json.load(f_api_wf)
        except Exception as load_api_wf_err:
             error_path_str = str(api_workflow_path) if api_workflow_path else "configured/default path"
             _dlog(f"[ERROR] Comfy API(Chunk): Failed load/parse API WF '{error_path_str}'. Err: {load_api_wf_err}.")
             return transformed_result

        try:
            # Node 74 입력 수정 (v11과 동일)
            if lyrics_node_id in api_workflow_graph: api_workflow_graph[lyrics_node_id].setdefault("inputs", {})["lyrics"] = text_chunk
            else: _dlog(f"[WARN] Comfy API(Chunk): Node {lyrics_node_id} not found."); return transformed_result
            # Node 79 설정 (v11과 동일)
            if save_text_node_id in api_workflow_graph:
                save_inputs = api_workflow_graph[save_text_node_id].setdefault("inputs", {}); save_inputs["text"] = [lyrics_node_id, 0]; save_inputs["filename_prefix"] = f"comfy_api_transform_temp_{uuid.uuid4().hex[:8]}"; save_inputs["append"] = "overwrite"

            # API 제출 (v11과 동일)
            prompt_payload_api = {"prompt": api_workflow_graph, "client_id": str(uuid.uuid4())}
            api_url_post = f"{host_address.rstrip('/')}/prompt"
            response_post_api = requests.post(api_url_post, json=prompt_payload_api, timeout=10); response_post_api.raise_for_status()
            response_json_api = response_post_api.json(); prompt_id_api = response_json_api.get('prompt_id')
            if not prompt_id_api: _dlog("[WARN] Comfy API(Chunk): Prompt ID 없음."); return transformed_result

            # 결과 폴링 (타임아웃 15초, 파싱 로직 강화)
            start_time_poll_api = time.time(); api_url_get = f"{host_address.rstrip('/')}/history/{prompt_id_api}"
            history_data_api: Optional[Dict] = None
            found_result_text: Optional[str] = None # ★★ 결과 저장 변수 ★★

            while time.time() - start_time_poll_api < 15:
                try:
                    response_get_api = requests.get(api_url_get, timeout=5)
                    if response_get_api.ok:
                        history_data_api = response_get_api.json()
                        if prompt_id_api in history_data_api:
                            history_entry = history_data_api[prompt_id_api]
                            status_obj = history_entry.get("status", {}); status_str = status_obj.get("status_str", "")
                            if status_str == "error": _dlog(f"[ERROR] Comfy API(Chunk): Job failed. Err: {status_obj.get('exception_message')}"); return transformed_result # 실패 시 즉시 반환

                            outputs_api = history_entry.get('outputs')
                            # ★★ Node 75 결과 파싱 강화 ★★
                            if isinstance(outputs_api, dict):
                                result_node_output = outputs_api.get(result_node_id) # Node 75 시도
                                if isinstance(result_node_output, dict):
                                    # 로그 확인 결과: 'text' 키에 리스트 형태로 저장됨
                                    result_list = result_node_output.get('text')
                                    if isinstance(result_list, list) and result_list:
                                        potential_text = result_list[0]
                                        if isinstance(potential_text, str) and potential_text.strip():
                                            found_result_text = potential_text.strip() # 결과 찾음!
                                            break # 결과 찾으면 폴링 중단

                            # 작업 완료 시 (결과 못 찾았어도) 폴링 중단
                            if status_obj.get("completed", False):
                                 # Node 75에서 결과를 못 찾았다는 로그 남기기
                                 _dlog(f"[WARN] Comfy API(Chunk): Job completed but Node {result_node_id} output was missing or invalid.")
                                 _dlog("       History Entry:", json.dumps(history_entry, indent=2, ensure_ascii=False))
                                 break # 루프 종료

                except requests.exceptions.Timeout: pass
                except requests.exceptions.RequestException as poll_err: _dlog(f"[WARN] Comfy API(Chunk): History polling error: {poll_err}")
                time.sleep(0.5)

            # ★★ 최종 결과 처리 ★★
            if found_result_text is not None:
                transformed_result = found_result_text
                _dlog("[DEBUG] Comfy API(Chunk) 변환 성공:", text_chunk, "->", transformed_result)
                return transformed_result # 성공 (태그 포함 가능)
            else:
                # 폴링 시간 초과 또는 완료되었으나 결과 못 찾음
                _dlog("[WARN] Comfy API(Chunk): 폴링 시간 초과 또는 최종 결과 없음. 원본 반환:", text_chunk)
                if history_data_api: _dlog("       Last History Response:", json.dumps(history_data_api.get(prompt_id_api, {}), indent=2, ensure_ascii=False))
                return text_chunk # 실패 시 원본 반환 (태그 없음)

        except requests.exceptions.RequestException as req_err_api: _dlog(f"[ERROR] Comfy API(Chunk) 호출 실패: {req_err_api}.")
        except Exception as exy: _dlog(f"[ERROR] Comfy API(Chunk) 처리 오류: {exy}.")
        return text_chunk # 최종 실패 시 원본 반환

    # --- 메인 로직 시작 ---
    _dlog("ENTER", f"project_dir={project_dir}")
    proj = Path(project_dir); proj.mkdir(parents=True, exist_ok=True)
    pj = proj / "project.json"; meta: Dict[str, Any] = load_json(pj, {}) or {}
    title = effective_title(meta); lyrics_raw = (meta.get("lyrics") or "").strip()

    # LLS 사용 여부 판단 (v11과 동일)
    use_lls = meta.get("lls_enabled") is True
    _dlog("LLS_CHECK", f"lls_enabled is {meta.get('lls_enabled')}, use_lls set to {use_lls}")

    # 가사 처리 로직 시작 (v11과 동일 - API 호출 로직은 위에서 수정됨)
    try: comfy_host_addr_local = _choose_host()
    except Exception as host_err: _dlog(f"[ERROR] ComfyUI host selection failed: {host_err}."); return f"Music generation failed: Cannot connect to ComfyUI host ({getattr(S, 'COMFY_HOST', 'default')})."

    lyrics_eff: str
    if use_lls: # (v11과 동일)
        lyrics_eff = (meta.get("lyrics_lls") or "").strip(); _dlog("LYRICS_MODE", "Using LLS (from project.json)")
        if not lyrics_eff: _dlog("[ERROR] LLS enabled but 'lyrics_lls' is empty."); return "Music generation failed: LLS enabled but lyrics_lls is empty."
    else: # (v11과 동일 - 내부 API 호출 로직 변경됨)
        _dlog("LYRICS_MODE", "Using Raw Lyrics (Applying split/API transform with dedicated workflow)")
        processed_lines_new: List[str] = []; chunk_pattern = re.compile(r'[가-힣]+(?:[,\s]*[가-힣]+)*|[^가-힣]+(?:[,\s]*[^가-힣]+)*')
        original_lines = lyrics_raw.splitlines(); total_lines_to_process = len(original_lines); _dlog("LYRICS_PROCESSING_START", f"Total lines: {total_lines_to_process}")
        for line_index, line in enumerate(original_lines):
            stripped_line = line.strip()
            if (line_index + 1) % 5 == 0 or (line_index + 1) == total_lines_to_process: _dlog("LYRICS_PROCESSING_PROGRESS", f"Processing line {line_index + 1}/{total_lines_to_process}")
            if not stripped_line: processed_lines_new.append(""); continue
            if stripped_line.startswith('[') and stripped_line.endswith(']'): processed_lines_new.append(stripped_line.lower()); continue
            tagged_chunks: List[str] = []; found_chunks = chunk_pattern.findall(stripped_line)
            if not found_chunks and stripped_line: tagged_chunks.append(f"[en]{stripped_line}")
            else:
                 for chunk_text in found_chunks:
                     stripped_chunk = chunk_text.strip()
                     if not stripped_chunk: continue
                     if re.search(r'[가-힣]', stripped_chunk):
                         transformed_ko_chunk = _call_comfy_api_for_ko_chunk(stripped_chunk, comfy_host_addr_local)
                         if transformed_ko_chunk.startswith("[ko]"): tagged_chunks.append(transformed_ko_chunk)
                         else: tagged_chunks.append(f"[ko]{stripped_chunk}"); _dlog(f"[WARN] API result for '{stripped_chunk}' invalid. Used original.")
                     else: tagged_chunks.append(f"[en]{stripped_chunk}")
            processed_line_result = " ".join(tagged_chunks); processed_lines_new.append(processed_line_result)
        lyrics_eff = "\n".join(processed_lines_new)
        _dlog("LYRICS_PROCESSING_END", f"Finished processing {total_lines_to_process} lines.")
        _dlog("[DEBUG] Final lyrics_eff preview (New Logic, first 200 chars):", lyrics_eff[:200])

    # --- 이하 로직 (초 계산 ~ 음악 생성 완료)은 v11과 동일 ---
    if not lyrics_eff: _dlog("[ERROR] No effective lyrics."); return "Music generation failed: No effective lyrics."

    # 변수 초기화 (v11과 동일)
    seconds_val: int = 60; positive_tags: List[str] = []; negative_tags: List[str] = []
    main_wf_path: Union[str, Path] = getattr(S, "ACE_STEP_PROMPT_JSON", "jsons/ace_step_1_t2m.json")
    graph_loaded: Optional[Dict] = None; base_host: str = comfy_host_addr_local
    subfolder_path: str = ""; save_prefix: str = ""; output_ext: str = ".wav"
    history_result: Optional[Dict] = None; final_audio_path: Optional[Path] = None; library_saved_path: Optional[Path] = None; lls_after_result: str = ""

    try: # 전체 메인 로직 try 블록 (v11과 동일)
        # 초 계산 (v11과 동일)
        if target_seconds is not None: seconds_val = int(max(1, target_seconds))
        else:
            time_meta = meta.get("time"); ts_meta = meta.get("target_seconds")
            try: seconds_val = int(ts_meta) if ts_meta is not None else int(time_meta) # type: ignore
            except (ValueError, TypeError): seconds_val = 60
            seconds_val = int(max(1, seconds_val))
        meta["target_seconds"] = seconds_val; meta["time"] = seconds_val; meta["lyrics_lls_now"] = lyrics_eff
        try: save_json(pj, meta); _dlog("META_SAVE_PRE", f"Saved target_seconds={seconds_val} and lyrics_lls_now")
        except OSError as e: _dlog(f"[WARN] Failed save pre-gen meta: {e}")
        # 태그 수집 (v11과 동일)
        positive_tags = _collect_effective_tags(meta)
        neg_raw = meta.get("prompt_neg") or ""; negative_tags = [t.strip() for t in re.split(r'[,;\n]+', str(neg_raw)) if t.strip()]
        # 워크플로 로드 (v11과 동일)
        try: graph_loaded = _load_workflow_graph(main_wf_path)
        except Exception as e: _dlog(f"[ERROR] Failed load main WF: {e}"); raise
        # 서버 주소 및 저장 경로 (v11과 동일)
        _dlog("HOST", base_host, "| DESIRED_FMT wav")
        sanitized_title = sanitize_title(title); subfolder_path = f"shorts_make/{sanitized_title}"; save_prefix = f"{subfolder_path}/vocal_final"
        output_ext = _force_wav_save_for_this_graph(graph_loaded, proj_dir=proj, filename_prefix=save_prefix)
        # 워크플로 노드 주입 (v11과 동일)
        try:
            lyrics_id="74"; txt_pos_id="14"; sec_id="17"; sampler_id="52"
            if lyrics_id in graph_loaded: graph_loaded[lyrics_id].setdefault("inputs", {})["lyrics"]=lyrics_eff; _dlog("INJECT", f"Node {lyrics_id} (Lyrics)")
            if txt_pos_id in graph_loaded: graph_loaded[txt_pos_id].setdefault("inputs", {})["tags"]=", ".join(positive_tags); _dlog("INJECT", f"Node {txt_pos_id} (Pos Tags)")
            if sec_id in graph_loaded: graph_loaded[sec_id].setdefault("inputs", {})["seconds"]=seconds_val; _dlog("INJECT", f"Node {sec_id} (Seconds)")
            if sampler_id in graph_loaded: graph_loaded[sampler_id].setdefault("inputs", {})["seed"]=_rand_seed(); _dlog("INJECT", f"Node {sampler_id} (Seed)")
            if negative_tags: _dlog("INFO", f"Negative tags ({len(negative_tags)}) not injected.")
        except Exception as e: _dlog(f"[ERROR] Main WF injection failed: {e}"); raise
        # 작업 제출 및 대기 (v11과 동일)
        notify("submitting", host=base_host); progress_cb = on_progress if callable(on_progress) else (lambda i: _dlog("PROG", i))
        try: dbg_wf = proj / "_debug_workflow_sent.json"; save_json(dbg_wf, {"prompt": graph_loaded}); _dlog("DEBUG_WORKFLOW_SAVE", f"Saved final WF to {dbg_wf.name}")
        except Exception as e: _dlog(f"[WARN] Failed save debug WF: {e}")
        history_result = _submit_and_wait(base_host, graph_loaded, timeout=_ace_wait_timeout_sec(), poll=_ace_poll_interval_sec(), on_progress=progress_cb)

        # 결과 처리 (v11과 동일)
        saved_files_list: List[Path] = []; outputs_hist = history_result.get("outputs") if isinstance(history_result, dict) else None
        if isinstance(outputs_hist, dict):
            for _nid, node_out in outputs_hist.items():
                for key in ("audio", "audios", "files", "wav", "mp3", "output"):
                    arr = node_out.get(key)
                    if not isinstance(arr, list): continue
                    for item in arr:
                        if not isinstance(item, dict): continue
                        fn = (item.get("filename") or "").strip(); sf = (item.get("subfolder") or "").strip()
                        if not fn or fn.startswith("ComfyUI_temp_"): continue
                        sf_norm = sf.replace("\\", "/").lstrip("/")
                        if not sf_norm.startswith(subfolder_path): continue
                        try:
                            dl_file = _download_output_file(base_host, fn, sf_norm, out_dir=proj)
                            if isinstance(dl_file, Path) and dl_file.exists() and dl_file.stat().st_size > 0: saved_files_list.append(dl_file); _dlog("DOWNLOAD_SUCCESS", f"File: '{fn}' -> '{dl_file.name}'")
                        except Exception as e: _dlog("DOWNLOAD_ERROR", f"'{fn}': {type(e).__name__}")

        # 최종 파일 선택 및 처리 (v11과 동일)
        if saved_files_list:
            wavs=sorted([p for p in saved_files_list if p.suffix.lower()==".wav"],key=lambda p:p.stat().st_mtime,reverse=True); others=sorted([p for p in saved_files_list if p.suffix.lower()!=".wav"],key=lambda p:p.stat().st_mtime,reverse=True)
            candidates=wavs+others; source_audio:Optional[Path]=candidates[0] if candidates else None
            if source_audio and source_audio.exists():
                _dlog("SELECTED_SOURCE_AUDIO", f"Selected '{source_audio.name}'.")
                ffmpeg_p = getattr(S, "FFMPEG_EXE", "ffmpeg") or "ffmpeg"
                try:
                    final_audio_path = _ensure_vocal_wav(source_audio, proj, ffmpeg_exe=ffmpeg_p); _dlog("ENSURED_WAV", f"Final WAV: '{final_audio_path.name}'")
                    master_fn = globals().get("_master_wav_precise")
                    if callable(master_fn) and isinstance(final_audio_path, Path) and final_audio_path.exists():
                         _dlog("MASTERING_START", f"Applying mastering...")
                         try:
                             ti=float(getattr(S,"MASTER_TARGET_I",-12.0)); tp=float(getattr(S,"MASTER_TARGET_TP",-1.0)); tl=float(getattr(S,"MASTER_TARGET_LRA",11.0))
                             mastered_p = master_fn(final_audio_path, I=ti, TP=tp, LRA=tl, ffmpeg_exe=ffmpeg_p)
                             if isinstance(mastered_p, Path) and mastered_p.exists():
                                 if mastered_p.resolve() != final_audio_path.resolve():
                                     try: final_audio_path.unlink(); mastered_p.rename(final_audio_path); _dlog("MASTERING_SUCCESS_REPLACED", f"Mastered: '{final_audio_path.name}'")
                                     except OSError as e: _dlog(f"[WARN] Failed replace after mastering: {e}. Kept at '{mastered_p.name}'."); final_audio_path = mastered_p
                                 else: _dlog("MASTERING_SUCCESS_INPLACE", f"Mastered in-place: '{final_audio_path.name}'")
                         except Exception as e: _dlog("MASTERING_ERROR", f"Error: {type(e).__name__}")
                except Exception as e: _dlog(f"[ERROR] Ensuring WAV failed: {e}"); final_audio_path = None

        # lyrics_lls_after 저장 로직 (v11과 동일)
        if isinstance(outputs_hist, dict):
            buffer_lls_after: List[str] = []; main_lyrics_node_output = outputs_hist.get("74", {})
            if main_lyrics_node_output:
                for key_lls_res in ("text", "result", "string", "strings"):
                    value_lls_res = main_lyrics_node_output.get(key_lls_res)
                    if isinstance(value_lls_res, str) and value_lls_res.strip(): buffer_lls_after.append(value_lls_res.strip()); break
                    elif isinstance(value_lls_res, list):
                         found = False
                         for item in value_lls_res:
                             if isinstance(item, str) and item.strip(): buffer_lls_after.append(item.strip()); found = True
                         if found: break
            if not buffer_lls_after: # 폴백: Node 75 시도
                preview_node_output = outputs_hist.get("75", {})
                if isinstance(preview_node_output, dict):
                    for key_preview in ("text", "previews"):
                        value_preview = preview_node_output.get(key_preview)
                        if isinstance(value_preview, str) and value_preview.strip(): buffer_lls_after.append(value_preview.strip()); break
                        elif isinstance(value_preview, list):
                            found_preview = False
                            for item_preview in value_preview:
                                if isinstance(item_preview, str) and item_preview.strip(): buffer_lls_after.append(item_preview.strip()); found_preview = True
                            if found_preview: break
            if buffer_lls_after:
                 lls_after_result = "\n".join(buffer_lls_after).strip()
                 if lls_after_result: meta["lyrics_lls_after"] = lls_after_result; _dlog("LLS_AFTER_CAPTURE", f"Captured Node 74/75 output: {len(lls_after_result)} chars")
                 # else: _dlog("LLS_AFTER_CAPTURE", "Captured output empty.") # 로그 간소화
            # else: _dlog("LLS_AFTER_CAPTURE", "Node 74/75 output not found.") # 로그 간소화
        # else: _dlog("LLS_AFTER_CAPTURE", "History 'outputs' missing.") # 로그 간소화

        # 최종 메타 업데이트 (v11과 동일)
        if isinstance(final_audio_path, Path) and final_audio_path.exists() and final_audio_path.stat().st_size > 0: meta.setdefault("paths", {})["vocal"] = str(final_audio_path); meta["audio"] = str(final_audio_path)
        else: final_audio_path = None
        comfy_debug_section = meta.setdefault("comfy_debug", {})
        comfy_debug_section.update({"host": base_host, "prompt_json": str(main_wf_path), "prompt_seconds": seconds_val, "requested_format": "wav", "requested_ext": output_ext, "subfolder": subfolder_path})
        meta["tags_effective"] = {"positive": positive_tags, "negative": negative_tags}
        try: save_json(pj, meta); _dlog("META_SAVE_FINAL", f"Final project.json saved: '{pj.name}'")
        except Exception as e: _dlog(f"[ERROR] Failed save final pj: {e}")

        # 라이브러리 복사 (v11과 동일)
        if isinstance(final_audio_path, Path) and final_audio_path.exists():
            try:
                library_path = save_to_user_library("audio", title, final_audio_path, rename=True)
                if isinstance(library_path, Path) and library_path.exists(): library_saved_path = library_path
            except Exception as e: _dlog("LIBRARY_COPY_ERROR", f"Failed: {e}")

    except Exception as main_execution_err: # 메인 로직 오류 처리 (v11과 동일)
        _dlog(f"[FATAL] Music generation failed: {main_execution_err}")
        notify("error", error=f"음악 생성 실패: {main_execution_err}")
        summary_final = f"ACE-Step 실패 ❌\n- 오류: {main_execution_err}"; _dlog("LEAVE", summary_final.replace("\n", " | ")); return summary_final

    # --- 최종 요약 (v11과 동일) ---
    wf_name = Path(main_wf_path).name if main_wf_path else "Unknown WF"
    message_lines: List[str] = [f"ACE-Step 완료 ✅", f"- 프롬프트: {wf_name}", f"- 길이: {seconds_val}s", f"- 태그: +{len(positive_tags)} / -{len(negative_tags)}"]
    if isinstance(final_audio_path, Path) and final_audio_path.exists(): message_lines.append(f"- 저장:     '{final_audio_path.name}'")
    else: message_lines.append("- 저장:     (오류 또는 파일 없음)")
    if isinstance(library_saved_path, Path) and library_saved_path.exists(): message_lines.append(f"- 라이브러리: '{library_saved_path.name}'")
    summary_final = "\n".join(message_lines); _dlog("LEAVE", summary_final.replace("\n", " | ")); notify("finished", summary=summary_final)
    return summary_final

# (... 기존 audio_sync.py 파일의 나머지 부분 ...)


##########################################################################
##########################################################################
##########################################################################









######################분석 보완############################################
def preprocess_for_analysis(src: str) -> str:
    """
    분석 친화 전처리:
      - 48k/mono로 로드
      - 약한 노이즈 리덕션
      - 하모닉 우세화(보컬 강조)
      - -16 LUFS 근사 정규화
      - 16-bit WAV 임시 파일로 저장
    실패 또는 의존성 부재 시 원본 경로 반환.
    """
    try:
        from pathlib import Path
        import soundfile as sf
        import librosa
        import noisereduce as nr
        import pyloudnorm as pyln

        src_path = Path(src)
        if not src_path.exists():
            return src

        y, sr = librosa.load(str(src_path), sr=48000, mono=True)
        if y is None or y.size == 0:
            return src

        # 약한 노이즈 리덕션(stationary=False: 음악에 안전)
        y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.3, stationary=False)

        # 하모닉/퍼커시브 분리 후 보컬 쪽 비중 약간 높임
        harm, perc = librosa.effects.hpss(y)
        y = (harm * 0.9) + (perc * 0.1)

        # -16 LUFS 정규화(EBU R128)
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        y = pyln.normalize.loudness(y, loudness, -16.0)

        # 소프트 클립으로 과도 피크 방지
        y = np.clip(y, -0.98, 0.98)

        out_path = src_path.with_suffix(".pre.wav")
        sf.write(str(out_path), y, sr, subtype="PCM_16")
        return str(out_path)
    except ImportError:
        # 의존성 없으면 원본 사용
        return src
    except Exception:
        # 전처리 실패 시에도 원본 사용(분석 파이프는 계속 진행)
        return src
def demucs_vocals_drums(src: str) -> dict:
    """
    Demucs로 stems 분리 후 vocals/drums 경로 딕셔너리 반환.
    - 실패/미설치 시 빈 dict 반환
    - 호출부는 존재 여부만 검사해서 쓰도록 설계
    """
    try:
        import sys
        import subprocess
        from pathlib import Path

        src_path = Path(src)
        if not src_path.exists():
            return {}

        # 출력 폴더: 원본과 같은 폴더 아래 demucs_out/
        out_dir = src_path.parent / "demucs_out"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [sys.executable, "-m", "demucs", "-n", "htdemucs", "-o", str(out_dir), str(src_path)]
        # demucs 실행 (에러 시 예외 발생 → 빈 dict 폴백)
        subprocess.run(cmd, check=True)

        # 결과: out_dir/model/songname/*.wav
        # 여러 번 실행 가능하므로 가장 최근 파일을 고른다.
        vocals = None
        drums = None
        latest_time = -1.0
        for p in out_dir.rglob("*.wav"):
            name = p.name.lower()
            try:
                t = p.stat().st_mtime
            except OSError:
                t = 0.0
            if t >= latest_time:
                latest_time = t
                if "vocals" in name:
                    vocals = p
                if "drums" in name:
                    drums = p

        out = {}
        if vocals and vocals.exists():
            out["vocals"] = str(vocals)
        if drums and drums.exists():
            out["drums"] = str(drums)
        return out
    except ImportError:
        return {}
    except Exception:
        return {}

def detect_onsets_percussive_librosa(wav_path: str) -> list[float]:
    """
    퍼커시브(타격) 성분을 강조해 리듬 경계를 잡는 온셋 검출(리브로사만 사용).
    - HPSS로 퍼커시브 분리 후 onset_detect
    - backtrack 활성화(근접 트랜지언트로 스냅)
    - 중복 제거 임계 10ms

    설치 의존성: librosa, soundfile
    실패 시: 빈 리스트 반환(호출부는 기존 온셋으로 폴백)
    """
    from pathlib import Path
    try:
        import soundfile as sf
        import librosa
    except ImportError:
        return []

    p = Path(wav_path)
    if not p.exists():
        return []

    try:
        # sr=None: 원 샘플레이트 유지(정밀 타임스탬프에 유리)
        y, sr = librosa.load(str(p), sr=None, mono=True)
        if y is None or y.size == 0 or sr is None or sr <= 0:
            return []

        # 하모닉/퍼커시브 분리 → 리듬 경계는 퍼커시브가 민감
        harm, perc = librosa.effects.hpss(y)
        # 온셋 탐지 파라미터 튜닝:
        # - backtrack=True: 에너지 피크 이전 트랜지언트로 이동해 경계가 자연스러움
        # - pre/post_max/avg, delta: 트랙별 거짓 양성 줄이는 안정 세팅
        on_times = librosa.onset.onset_detect(
            y=perc,
            sr=sr,
            units="time",
            backtrack=True,
            pre_max=20,
            post_max=20,
            pre_avg=100,
            post_avg=100,
            delta=0.2,
            wait=0,
        )

        # 10ms 중복 제거 + 반올림
        out: list[float] = []
        last = -1.0
        for t in on_times:
            t = float(t)
            if last < 0.0 or (t - last) >= 0.010:
                out.append(round(t, 3))
                last = t
        return out
    except (ValueError, RuntimeError):
        return []

###################################################################
#######################테스트중##################################
###################################################################


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
    from pathlib import Path
    import json
    import math
    import os

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

    final_audio_duration_after_processing = get_audio_duration(str(slp_audio_path))
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



# 한글 자모 분해를 위한 전역 상수
CHOSUNG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSUNG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']


# audio_sync.py 파일에서 이 함수를 찾아 아래 내용으로 전체를 교체하세요.

def _create_final_segments_from_ready(
        seg_ready_payload: list,
        clean_lyrics_lines: list, # 헤더 포함된 원본 가사 (입력)
        lyrics_compare_lines: list, # 헤더 포함된 비교용 가사 (입력)
        project_dir: str
) -> list:
    """
    [헤더 제거 최종 수정 v5.1]
    - Boundary detection uses lyric lines *without* headers.
    - Word correction reference (`ref_all_words`) is now also built from lyric lines *without* headers.
    - Unnecessary trailing semicolons removed.
    - Adheres to all specified coding rules.
    - Debug logging remains.
    """
    import difflib
    from pathlib import Path
    from typing import List, Dict, Pattern, Optional, Any # Any 추가
    import re
    # numpy는 현재 사용되지 않음

    # --- Constants and local utils ---
    _MATCH_TYPE_KEPT = "KEPT"
    _MATCH_TYPE_AUTHENTICATED = "AUTHENTICATED"
    _MATCH_TYPE_CORRECTED = "CORRECTED"
    _FILTER_SHORT_WORDS = {'a', 'ah', 'oh', 'uh', 'hm', 'hmm', 'um', 'o', 'eh', '음', 'nan', 'neo'}

    def _safe_float(val: Any) -> float: # 타입 힌트 Any 추가
        try: return float(val)
        except (TypeError, ValueError): return 0.0

    def _norm_for_compare(text: str) -> str:
        s = str(text or "").lower().strip(); s = re.sub(r"[^a-z0-9가-힣\s']+", " ", s); s = re.sub(r"\s+", " ", s).strip(); return s

    def _match_score(a: str, b: str) -> float:
        if not a or not b: return 0.0
        # SequenceMatcher 사용 유지 (타입 문제 없음)
        return difflib.SequenceMatcher(None, a, b).ratio()

    # kroman 처리 (ImportError 시 None 할당 및 except 블록에서도 정의)
    kroman: Optional[Any] # 타입 힌트 추가
    try:
        import kroman
    except ImportError:
        kroman = None
        print("[WARN] kroman library not found. Romanization comparison will be limited.")
    # except 블록에서도 kroman = None 으로 정의됨

    _KOREAN_PATTERN: Pattern[str] = re.compile(r'[\uac00-\ud7a3]')

    def _is_korean(word: str) -> bool: return bool(_KOREAN_PATTERN.search(str(word or "")))

    def _romanize(text: str) -> str:
        text_norm = _norm_for_compare(text) # 세미콜론 제거됨
        if not text_norm: return ""
        korean_parts = _KOREAN_PATTERN.findall(text) # 세미콜론 제거됨
        if korean_parts and kroman:
            try:
                korean_text_only = "".join(korean_parts)
                # kroman.parse가 None을 반환할 수 있으므로 처리
                parsed: Optional[str] = kroman.parse(korean_text_only)
                # None이 아닐 때만 replace/lower 호출
                return parsed.replace("-", "").lower() if parsed else text_norm
            except Exception as e_kroman_parse: # 구체적인 예외 처리
                print(f"[WARN] kroman parsing failed for '{text}': {type(e_kroman_parse).__name__}")
                return text_norm # 실패 시 정규화된 원본 반환
        return text_norm

    # ---- 1. Prepare Reference Lines & Boundary Detection ----
    print("\n--- [BOUNDARY DETECTION] ---")
    correction_ref_lines_with_headers = lyrics_compare_lines if lyrics_compare_lines else clean_lyrics_lines
    if not correction_ref_lines_with_headers or not seg_ready_payload:
        print("[ERROR] Reference lyrics (with headers) or ready segments are empty.")
        return []

    # 헤더 제외한 가사 목록 생성
    boundary_ref_lines_no_headers: List[str] = []
    header_pattern = re.compile(r"^\s*\[[^]]+]\s*$")
    for line_with_header in correction_ref_lines_with_headers: # 변수명 변경 (가리기 방지)
        if not header_pattern.match(line_with_header):
            boundary_ref_lines_no_headers.append(line_with_header)

    if not boundary_ref_lines_no_headers:
        print("[ERROR] No actual lyric lines found after removing headers.")
        # 헤더만 있는 경우: v5의 폴백 로직 유지 (헤더 포함 목록 사용)
        boundary_ref_lines_no_headers = correction_ref_lines_with_headers
        if not boundary_ref_lines_no_headers: return []
        print("[WARN] Fallback: Using lines *with* headers for boundary detection.")

    # 경계 탐색용 첫/마지막 줄 (헤더 제외 목록 사용)
    first_line_raw = boundary_ref_lines_no_headers[0] if boundary_ref_lines_no_headers else ""
    last_line_raw = boundary_ref_lines_no_headers[-1] if boundary_ref_lines_no_headers else ""
    first_line_norm = _norm_for_compare(first_line_raw)
    last_line_norm = _norm_for_compare(last_line_raw)
    print(f"[DEBUG] Target First Line (Raw, No Header): '{first_line_raw}'") # ... (로그 유지)
    print(f"[DEBUG] Target First Line (Norm, No Header): '{first_line_norm}'")
    print(f"[DEBUG] Target Last Line (Raw, No Header): '{last_line_raw}'")
    print(f"[DEBUG] Target Last Line (Norm, No Header): '{last_line_norm}'")

    first_idx, last_idx = -1, -1
    score_threshold_boundary = 0.25

    # 첫 줄 탐색
    print("\n[DEBUG] Searching for First Line Match...") # ... (탐색 및 로그 로직 유지)
    if first_line_norm:
        for i, seg_item in enumerate(seg_ready_payload): # 변수명 변경 (가리기 방지)
            if isinstance(seg_item, dict) and "text" in seg_item:
                 seg_text_norm = _norm_for_compare(seg_item.get("text", ""))
                 score_val = _match_score(seg_text_norm, first_line_norm) # 변수명 변경 (가리기 방지)
                 print(f"  [DEBUG] Comparing seg[{i}] ('{seg_text_norm}') vs First ('{first_line_norm}') -> Score: {score_val:.4f}")
                 if score_val > score_threshold_boundary: first_idx = i; print(f"    => Match found! Setting first_idx = {i}"); break
    else: first_idx = 0; print("[WARN] Target first line empty, setting first_idx = 0.")

    # 마지막 줄 탐색
    print("\n[DEBUG] Searching for Last Line Match (reverse)...") # ... (탐색 및 로그 로직 유지)
    if last_line_norm:
        # range 길이를 안전하게 계산
        payload_len = len(seg_ready_payload)
        for i_rev in range(payload_len - 1, -1, -1): # 변수명 변경 (가리기 방지)
            current_seg_item = seg_ready_payload[i_rev] # 변수명 변경 (가리기 방지)
            if isinstance(current_seg_item, dict) and "text" in current_seg_item:
                 seg_text_norm_rev = _norm_for_compare(current_seg_item.get("text", "")) # 변수명 변경
                 score_val_rev = _match_score(seg_text_norm_rev, last_line_norm) # 변수명 변경
                 print(f"  [DEBUG] Comparing seg[{i_rev}] ('{seg_text_norm_rev}') vs Last ('{last_line_norm}') -> Score: {score_val_rev:.4f}")
                 if score_val_rev > score_threshold_boundary: last_idx = i_rev; print(f"    => Match found! Setting last_idx = {i_rev}"); break
    else: last_idx = len(seg_ready_payload) - 1 if seg_ready_payload else -1; print("[WARN] Target last line empty, setting last_idx = end.")

    print(f"\n[DEBUG] Final Boundary Indices: first_idx = {first_idx}, last_idx = {last_idx}")

    # seg_compare_payload 생성
    seg_compare_payload: list = [] # 타입 명시
    if first_idx != -1 and last_idx != -1 and last_idx >= first_idx: seg_compare_payload = seg_ready_payload[first_idx : last_idx + 1]; print(f"[INFO] Slicing seg_ready from {first_idx} to {last_idx + 1}.")
    elif first_idx != -1: seg_compare_payload = seg_ready_payload[first_idx:]; print(f"[WARN] Last line fail. Slicing from {first_idx} to end.")
    elif last_idx != -1: seg_compare_payload = seg_ready_payload[: last_idx + 1]; print(f"[WARN] First line fail. Slicing from start to {last_idx + 1}.")
    else: seg_compare_payload = seg_ready_payload; print("[WARN] Both boundary fails. Using all segments.")
    print(f"       => Resulting seg_compare_payload length: {len(seg_compare_payload)}")

    if not seg_compare_payload: print("[ERROR] No segments selected. Cannot proceed."); return []

    # seg_compare.json 저장
    try: # ... (저장 로직 유지)
        compare_path = Path(project_dir) / "seg_compare.json"
        # save_json 임포트 확인 (파일 상단 또는 try 블록 내부)
        try: from app.utils import save_json
        except ImportError: from utils import save_json # type: ignore[no-redef]
        save_json(compare_path, seg_compare_payload); print(f"[SYNC-PRO] seg_compare.json saved: {compare_path} ({len(seg_compare_payload)} segments)")
    except Exception as e_save_compare: # 구체적인 예외 처리
        print(f"[WARN] Failed to save seg_compare.json: {type(e_save_compare).__name__}")


    # ---- 2 & 3. Word Correction (Using header-less reference words) ----
    # [핵심 수정] ref_all_words 생성 시 헤더 제외 목록 사용
    ref_all_words_raw = " ".join(boundary_ref_lines_no_headers).replace(",", "").replace("'", "")
    ref_all_words = ref_all_words_raw.split()
    ref_all_words = [w for w in ref_all_words if w]
    if not ref_all_words: print("[ERROR] Reference words list (no headers) is empty."); return []
    print(f"[DEBUG] Using {len(ref_all_words)} reference words for correction (headers excluded).")
    ref_words_norm = [_norm_for_compare(w) for w in ref_all_words]
    ref_words_roman = [_romanize(w) for w in ref_all_words]

    intermediate_segments: List[Dict] = []
    print("\n--- [SEGMENT CORRECTION PROCESS] ---")
    for seg_idx, seg_corr in enumerate(seg_compare_payload): # 변수명 변경 (가리기 방지)
        if not isinstance(seg_corr, dict) or "text" not in seg_corr: continue
        whisper_text_corr = str(seg_corr.get("text", "")).strip() # 변수명 변경
        if not whisper_text_corr: continue
        segment_corrected_words: List[str] = []
        whisper_words_list = whisper_text_corr.split(); whisper_cursor = 0; leftover_whisper_part = "" # 변수명 변경
        while whisper_cursor < len(whisper_words_list):
            candidate_whisper_raw = leftover_whisper_part + whisper_words_list[whisper_cursor]; candidate_whisper = candidate_whisper_raw.strip()
            candidate_norm = _norm_for_compare(candidate_whisper)
            if not candidate_norm: whisper_cursor += 1; leftover_whisper_part = ""; continue
            best_match: Dict = {"score": -1.0, "word": candidate_whisper, "type": _MATCH_TYPE_KEPT, "ref_origin": ""}
            for i_ref, loop_ref_word in enumerate(ref_all_words): # 변수명 변경 (가리기 방지)
                # [수정] 후행 세미콜론 제거됨
                ref_norm_loop = ref_words_norm[i_ref] # 변수명 변경
                ref_roman_loop = ref_words_roman[i_ref] # 변수명 변경
                if not ref_norm_loop: continue
                auth_score = -1.0
                if candidate_norm and ref_norm_loop and candidate_norm in ref_norm_loop: auth_score = 0.85 + (len(candidate_norm) / len(ref_norm_loop)) * 0.15
                spell_score = _match_score(candidate_norm, ref_norm_loop); roman_score = 0.0; candidate_roman = _romanize(candidate_whisper)
                if candidate_roman and ref_roman_loop: roman_score = _match_score(candidate_roman, ref_roman_loop)
                elif not _is_korean(candidate_whisper) and ref_roman_loop: roman_score = _match_score(candidate_norm, ref_roman_loop) * 0.9
                correction_score = max(spell_score, roman_score); current_best_score = best_match["score"]; updated = False
                if auth_score > correction_score and auth_score > current_best_score: best_match.update({"score": auth_score, "word": candidate_whisper, "type": _MATCH_TYPE_AUTHENTICATED, "ref_origin": loop_ref_word}); updated = True
                elif correction_score >= auth_score and correction_score > current_best_score:
                    is_candidate_korean = _is_korean(candidate_whisper); is_ref_single_eng = len(loop_ref_word) == 1 and 'a' <= loop_ref_word.lower() <= 'z'
                    if not (is_candidate_korean and is_ref_single_eng): best_match.update({"score": correction_score, "word": loop_ref_word, "type": _MATCH_TYPE_CORRECTED, "ref_origin": loop_ref_word}); updated = True
            match_type = best_match["type"]; word_to_process = best_match["word"]; best_score = best_match["score"]; ref_origin = best_match["ref_origin"]; score_threshold = 0.50
            word_to_append: Optional[str] = None; leftover_whisper_part = ""
            if match_type == _MATCH_TYPE_KEPT:
                if candidate_whisper.lower() not in _FILTER_SHORT_WORDS: word_to_append = candidate_whisper
            else:
                if best_score >= score_threshold:
                    word_to_append = word_to_process
                    ref_matched_norm = _norm_for_compare(ref_origin); candidate_actual_norm = _norm_for_compare(candidate_whisper)
                    if len(candidate_actual_norm) > len(ref_matched_norm) and candidate_actual_norm.startswith(ref_matched_norm):
                        try:
                            match_obj = re.match(re.escape(ref_matched_norm), candidate_whisper, flags=re.IGNORECASE | re.UNICODE) # 변수명 변경
                            if match_obj: leftover_whisper_part = candidate_whisper[match_obj.end():].strip()
                        except re.error: leftover_whisper_part = ""
                else:
                    if candidate_whisper.lower() not in _FILTER_SHORT_WORDS: word_to_append = candidate_whisper
            if word_to_append: segment_corrected_words.append(word_to_append)
            whisper_cursor += 1
        corrected_text = " ".join(segment_corrected_words).strip()
        if corrected_text:
            intermediate_segments.append({"start": _safe_float(seg_corr.get("start")), "end": _safe_float(seg_corr.get("end")), "text": corrected_text, "line_ko": corrected_text})

    print("--- [SEGMENT CORRECTION PROCESS COMPLETED] ---")
    print(f"Total intermediate segments generated: {len(intermediate_segments)}")

    # ---- 4. Final Boundary Check & Return ----
    if not intermediate_segments: print("[ERROR] No segments survived correction."); return []

    final_start_idx, final_end_idx = -1, -1
    final_boundary_threshold = 0.45

    print("\n--- [FINAL BOUNDARY CHECK] ---") # ... (최종 경계 확인 로직 및 로그 유지)
    if first_line_norm:
        print(f"[DEBUG] Final check - Searching for First Line (Norm): '{first_line_norm}'")
        for i_final, seg_final in enumerate(intermediate_segments): # 변수명 변경 (가리기 방지)
            score_final = _match_score(_norm_for_compare(seg_final.get("text", "")), first_line_norm) # 변수명 변경
            if score_final > final_boundary_threshold: final_start_idx = i_final; print(f"    => Match found! final_start_idx = {i_final}"); break
    else: final_start_idx = 0; print("[WARN] Final check - Target first line empty, using index 0.")

    if last_line_norm:
        print(f"[DEBUG] Final check - Searching for Last Line (Norm, reverse): '{last_line_norm}'")
        # range 길이 안전 계산
        inter_len = len(intermediate_segments)
        for i_final_rev in range(inter_len - 1, -1, -1): # 변수명 변경
            seg_final_rev = intermediate_segments[i_final_rev] # 변수명 변경
            score_final_rev = _match_score(_norm_for_compare(seg_final_rev.get("text", "")), last_line_norm) # 변수명 변경
            if score_final_rev > final_boundary_threshold: final_end_idx = i_final_rev; print(f"    => Match found! final_end_idx = {i_final_rev}"); break
    else: final_end_idx = len(intermediate_segments) - 1 if intermediate_segments else -1; print("[WARN] Final check - Target last line empty, using last index.")

    print(f"[DEBUG] Final Boundary Indices after check: start={final_start_idx}, end={final_end_idx}")

    final_segments: List[Dict] = [] # ... (슬라이싱 및 로그 로직 유지)
    if final_start_idx != -1 and final_end_idx != -1 and final_end_idx >= final_start_idx: final_segments = intermediate_segments[final_start_idx : final_end_idx + 1]; print(f"[SYNC-PRO] Final check: Kept segments {final_start_idx} to {final_end_idx} (Total: {len(final_segments)}).")
    elif intermediate_segments: final_segments = intermediate_segments; print("[WARN] Final check failed, using all intermediate.")
    else: print("[ERROR] Final check failed, no intermediate segments."); return []


    # Time Adjustment
    final_segments.sort(key=lambda x: _safe_float(x.get("start", 0.0)))
    print("\n--- [TIME ADJUSTMENT] ---") # ... (시간 조정 로직 및 로그 유지)
    for i_adjust in range(1, len(final_segments)): # 변수명 변경
        prev_end_adjust = _safe_float(final_segments[i_adjust - 1].get("end", 0.0)) # 변수명 변경
        curr_start_adjust = _safe_float(final_segments[i_adjust].get("start", 0.0)) # 변수명 변경
        if curr_start_adjust < prev_end_adjust - 0.01:
            final_segments[i_adjust]["start"] = prev_end_adjust
            curr_end_adjust = _safe_float(final_segments[i_adjust].get("end", 0.0)) # 변수명 변경
            if curr_end_adjust < prev_end_adjust + 0.1: final_segments[i_adjust]["end"] = prev_end_adjust + 0.1

    # Final Return
    print(f"\n[SYNC-PRO] Returning {len(final_segments)} final segments.")
    print("------------------------------------\n")
    return final_segments


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
    from pathlib import Path
    from typing import Any, Dict, List, Tuple
    import sys

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





# (다른 함수 정의들...)

# ============================================================
# 최종 오디오 처리 및 시간 업데이트 함수 (모든 경고 해결 최종 버전)
# ============================================================
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
        try:
            from app import settings as app_settings_local
        except ImportError:
            app_settings_local = None

        try:
            import settings as root_settings_local
        except ImportError:
            root_settings_local = None

        if app_settings_local and hasattr(app_settings_local, "FFMPEG_EXE"):
            ffmpeg_exe = getattr(app_settings_local, "FFMPEG_EXE") or "ffmpeg"
        elif root_settings_local and hasattr(root_settings_local, "FFMPEG_EXE"):
            ffmpeg_exe = getattr(root_settings_local, "FFMPEG_EXE") or "ffmpeg"
    except Exception as e:
        print(f"[WARN] Error loading settings for FFMPEG_EXE: {e}. Using default.")

    # ---- 3. 오디오 처리 및 'time' 업데이트 ----
    last_lyric_end = _safe_float(organized_segments[-1].get("end", 0.0))
    target_duration = last_lyric_end + 5.0
    actual_duration = get_audio_duration(audio_path)
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


###################################################################
###################################################################
###################################################################









def estimate_global_shift(segments: list, onsets: list[float], *, max_abs_sec: float = 12.0) -> float:
    """
    세그먼트 start 시각과 가까운 퍼커시브 온셋을 매칭해 '전역 시프트(지연/앞당김)'를 추정.
    - shift = median( nearest_onset - seg.start )
    - 너무 큰 값은 max_abs_sec로 제한.
    - onsets/segments가 비어있으면 0.0
    """
    import math
    if not segments or not onsets:
        return 0.0
    starts = []
    for seg in segments:
        try:
            st = float(seg.get("start", 0.0))
            if math.isfinite(st):
                starts.append(st)
        except (TypeError, ValueError):
            continue
    if not starts:
        return 0.0
    on = sorted(float(t) for t in onsets if isinstance(t, (int, float)))
    if not on:
        return 0.0

    diffs = []
    j = 0
    n = len(on)
    for st in starts:
        while j + 1 < n and abs(on[j + 1] - st) <= abs(on[j] - st):
            j += 1
        diffs.append(on[j] - st)
    diffs.sort()
    mid = len(diffs) // 2
    if len(diffs) % 2 == 1:
        shift = diffs[mid]
    else:
        shift = 0.5 * (diffs[mid - 1] + diffs[mid])
    if abs(shift) > max_abs_sec:
        if shift > 0:
            shift = max_abs_sec
        else:
            shift = -max_abs_sec
    return float(round(shift, 3))

def apply_global_shift(res: dict, shift_sec: float) -> dict:
    """
    res(dict)에 포함된 segments/start_at을 shift_sec만큼 이동해 새로운 dict 반환.
    - shift_sec > 0: 지연(뒤로), shift_sec < 0: 당김(앞으로)
    - duration_sec은 변경하지 않음.
    - start/end/start_at은 '원래 타입'을 유지해 기록(문자열이면 문자열로).
    """
    import copy

    out = copy.deepcopy(res or {})
    if not isinstance(out, dict):
        return res

    # segments: 각 항목의 start/end 타입을 보존해 기록
    segs = out.get("segments") or []
    new_segs = []
    for seg in segs:
        if not isinstance(seg, dict):
            new_segs.append(seg)
            continue

        s2 = dict(seg)

        # start
        orig_start = seg.get("start", 0.0)
        try:
            start_num = float(orig_start) + float(shift_sec)
            start_num = max(0.0, start_num)
            if isinstance(orig_start, str):
                s2["start"] = f"{start_num:.3f}"
            else:
                s2["start"] = round(start_num, 3)
        except (TypeError, ValueError):
            # 원본이 비정상이라면 그대로 둠
            pass

        # end
        orig_end = seg.get("end", 0.0)
        try:
            end_num = float(orig_end) + float(shift_sec)
            end_num = max(0.0, end_num)
            if isinstance(orig_end, str):
                s2["end"] = f"{end_num:.3f}"
            else:
                s2["end"] = round(end_num, 3)
        except (TypeError, ValueError):
            pass

        new_segs.append(s2)

    out["segments"] = new_segs

    # start_at: 타입 보존
    orig_sa = out.get("start_at", 0.0)
    try:
        sa_num = float(orig_sa) + float(shift_sec)
        sa_num = max(0.0, sa_num)
        if isinstance(orig_sa, str):
            out["start_at"] = f"{sa_num:.3f}"
        else:
            out["start_at"] = round(sa_num, 3)
    except (TypeError, ValueError):
        # 원본 유지
        pass

    return out

def _rms_envelope(audio_path: str, *, sr_out: int = 16000, hop_length: int = 512) -> tuple[list[float], list[float]]:
    """
    오디오에서 RMS 엔벨로프 추출 → (times_sec, rms_list) 반환.
    - librosa.feature.rms 를 직접 임포트하여 타입 스텁 경고 회피
    - 실패 시 빈 리스트 반환
    """
    try:
        import librosa
        # 여기서 'librosa.feature.rms'를 모듈 속성으로 접근하지 않고 직접 임포트
        from librosa.feature import rms as librosa_rms
    except ImportError:
        return ([], [])



    try:
        y, sr = librosa.load(audio_path, sr=sr_out, mono=True)
        if y is None or y.size == 0:
            return ([], [])

        # 프레임 RMS (직접 임포트한 함수 사용)
        rms_vals = librosa_rms(
            y=y,
            frame_length=2048,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
        )[0]

        # 프레임 인덱스 → 시간축
        n_frames = int(rms_vals.shape[0])
        t = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

        # 정규화
        peak = float(np.max(rms_vals)) if n_frames > 0 else 0.0
        if peak > 0.0:
            rms_vals = (rms_vals / peak).astype(float)

        return ([float(x) for x in t], [float(x) for x in rms_vals])
    except (ValueError, RuntimeError, OSError):
        return ([], [])



def estimate_global_shift_signal(audio_path: str, segments: list, *, max_abs_sec: float = 12.0) -> float:
    """
    엔벨로프와 '줄 시작 시각'의 임펄스 열을 교차상관으로 정렬 → 전역 시프트 추정.
    +면 뒤로(지연), -면 앞으로(당김). 실패 시 0.0
    """


    times, env = _rms_envelope(audio_path)
    if not times or not env or not segments:
        return 0.0


    # 줄 시작 시각 벡터(임펄스)
    starts = []
    for s in segments:
        try:
            st = float(s.get("start", 0.0))
            if np.isfinite(st):
                starts.append(st)
        except (TypeError, ValueError):
            continue
    if not starts:
        return 0.0

    # env에 맞춰 임펄스 시퀀스 생성
    dt = float(times[1] - times[0]) if len(times) > 1 else 0.01
    n = len(times)
    imp = np.zeros(n, dtype=float)
    for st in starts:
        idx = int(round(st / dt))
        if 0 <= idx < n:
            imp[idx] = 1.0

    # 교차상관 (valid 범위 내에서 argmax)
    c = np.correlate(env, imp, mode="full")  # env를 기준
    lag = int(np.argmax(c)) - (n - 1)
    shift = -lag * dt  # env(t) ≈ imp(t+shift) → 시프트 부호 보정

    if abs(shift) > max_abs_sec:
        shift = max_abs_sec if shift > 0 else -max_abs_sec
    return float(round(shift, 3))


# audio_sync.py 파일에서 이 함수를 찾아 아래 내용으로 전체를 교체하세요.

def estimate_affine_drift(segments: list, onsets: list[float]) -> tuple[float, float]:
    """
    RANSAC 알고리즘을 사용하여 이상치(outlier)에 강건한 선형 드리프트를 추정한다.
    - seg.start와 가장 가까운 onsets를 짝지어 (t, t') 데이터 포인트를 생성.
    - 무작위 샘플링으로 여러 선형 모델(t' = a*t + b)을 만들고,
      가장 많은 데이터 포인트를 지지하는 (inliers) 모델을 선택.
    - 최종적으로 모든 inlier를 사용하여 모델을 다시 피팅하여 정확도 향상.
    - 실패 또는 데이터 부족 시 (1.0, 0.0) 반환.
    """
    import random

    if not segments or not onsets or len(segments) < 5:
        return (1.0, 0.0)

    # 1. 데이터 포인트 생성: (segment_start, nearest_onset)
    points = []
    on = sorted([float(o) for o in onsets if isinstance(o, (int, float))])
    j = 0
    n_onsets = len(on)
    for seg in segments:
        try:
            st = float(seg.get("start", 0.0))
        except (TypeError, ValueError):
            continue

        # 가장 가까운 온셋 탐색
        while j + 1 < n_onsets and abs(on[j + 1] - st) <= abs(on[j] - st):
            j += 1
        points.append((st, on[j]))

    if len(points) < 5:
        return (1.0, 0.0)

    # 2. RANSAC 파라미터
    n_points = len(points)
    iterations = 50
    sample_size = 2  # 최소 2개의 점으로 직선 정의
    inlier_threshold = 0.5  # FIX: 허용 오차를 0.25초에서 0.5초로 늘려 더 유연하게 추세 분석
    min_inliers = max(5, int(n_points * 0.3))  # 최소 30%의 데이터가 모델을 지지해야 함

    best_model = (1.0, 0.0)
    best_inliers_count = 0

    # numpy 배열로 변환
    data = np.array(points)
    x_coords = data[:, 0]
    y_coords = data[:, 1]

    # 3. RANSAC 반복
    for _ in range(iterations):
        # 무작위 샘플 추출
        sample_indices = random.sample(range(n_points), sample_size)
        sample = data[sample_indices]

        # 샘플로 모델 피팅 (t' = a*t + b)
        aaa = np.vstack([sample[:, 0], np.ones(len(sample))]).T
        try:
            a, b = np.linalg.lstsq(aaa, sample[:, 1], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        # 전체 데이터에 대한 오차 계산
        y_predicted = a * x_coords + b
        errors = np.abs(y_coords - y_predicted)

        # Inlier 개수 계산
        inlier_indices = np.where(errors < inlier_threshold)[0]
        current_inliers_count = len(inlier_indices)

        # 더 나은 모델이면 교체
        if current_inliers_count > best_inliers_count:
            best_inliers_count = current_inliers_count
            best_model = (a, b)
            # 수렴 조건: 충분한 inlier를 찾으면 조기 종료
            if best_inliers_count >= n_points * 0.7:
                break

    # 4. 최종 모델 피팅
    if best_inliers_count >= min_inliers:
        y_predicted_best = best_model[0] * x_coords + best_model[1]
        errors_best = np.abs(y_coords - y_predicted_best)
        final_inlier_indices = np.where(errors_best < inlier_threshold)[0]

        if len(final_inlier_indices) >= 2:
            inlier_points = data[final_inlier_indices]
            aa_final = np.vstack([inlier_points[:, 0], np.ones(len(inlier_points))]).T
            try:
                a_final, b_final = np.linalg.lstsq(aa_final, inlier_points[:, 1], rcond=None)[0]

                # 극단적인 값 방지
                if 0.85 <= a_final <= 1.15 and abs(b_final) < 15.0:
                    return (float(a_final), float(b_final))
            except np.linalg.LinAlgError:
                pass

    # RANSAC 실패 시 기본값 반환
    return (1.0, 0.0)


def apply_affine_time_map(res: dict, a: float, b: float) -> dict:
    """
    segments/start_at에 t' = a*t + b 선형 맵 적용. 타입 보존.
    """
    import copy
    out = copy.deepcopy(res or {})
    if not isinstance(out, dict):
        return res

    segs = out.get("segments") or []
    new_segs = []
    for seg in segs:
        if not isinstance(seg, dict):
            new_segs.append(seg)
            continue
        s2 = dict(seg)

        # start
        orig_start = seg.get("start", 0.0)
        try:
            start_num = (a * float(orig_start)) + b
            start_num = max(0.0, start_num)
            s2["start"] = f"{start_num:.3f}" if isinstance(orig_start, str) else round(start_num, 3)
        except (TypeError, ValueError):
            pass

        # end
        orig_end = seg.get("end", 0.0)
        try:
            end_num = (a * float(orig_end)) + b
            end_num = max(0.0, end_num)
            s2["end"] = f"{end_num:.3f}" if isinstance(orig_end, str) else round(end_num, 3)
        except (TypeError, ValueError):
            pass

        new_segs.append(s2)
    out["segments"] = new_segs

    orig_sa = out.get("start_at", 0.0)
    try:
        sa_num = (a * float(orig_sa)) + b
        sa_num = max(0.0, sa_num)
        out["start_at"] = f"{sa_num:.3f}" if isinstance(orig_sa, str) else round(sa_num, 3)
    except (TypeError, ValueError):
        pass

    return out

def _auto_anchor_from_energy(audio_path: str, *, sr_out: int = 16000, hop_length: int = 512,
                             floor_db: float = 35.0, min_run_sec: float = 0.35) -> float:
    """
    오디오의 RMS/무음구간을 이용해 '최초로 충분히 큰 에너지'가 등장하는 시각을 추출.
    - 반환: anchor_sec (없으면 0.0)
    """
    try:
        import librosa
    except ImportError:
        return 0.0

    try:
        y, sr = librosa.load(audio_path, sr=sr_out, mono=True)
        if y is None or y.size == 0:
            return 0.0

        # 무음 구간 split (top_db 낮을수록 민감)
        intervals = librosa.effects.split(y, top_db=floor_db, frame_length=2048, hop_length=hop_length)
        if intervals is None or len(intervals) == 0:
            return 0.0

        # 첫 유효 구간(길이가 충분한) 시작 시각
        min_run = int(round(min_run_sec * sr))
        for beg, end in intervals:
            if end - beg >= min_run:
                t = float(beg) / float(sr)
                return round(max(0.0, t), 3)
        # 짧은 구간만 있다면 첫 구간 시작
        t = float(intervals[0][0]) / float(sr)
        return round(max(0.0, t), 3)
    except (ValueError, RuntimeError, OSError):
        return 0.0

def enforce_monotonic_segments(res: dict, *, min_gap_sec: float = 0.05) -> dict:
    """
    segments의 start/end가 서로 겹치지 않도록 단조 증가로 정렬/보정한다.
    - 인접 세그먼트간 최소 간격(min_gap_sec) 보장
    - 각 세그먼트의 '길이'는 가능한 한 유지(앞 세그먼트 end를 기준으로 다음 start를 밀어냄)
    - duration_sec은 변경하지 않음
    - start/end/start_at의 '기존 타입(str/float)'을 보존
    """
    import copy

    out = copy.deepcopy(res or {})
    if not isinstance(out, dict):
        return res

    segs = out.get("segments") or []
    if not isinstance(segs, list) or not segs:
        return out

    # 헬퍼: 타입 보존하여 수치 쓰기
    def _write(v_old, v_num):
        v_num = max(0.0, float(v_num))
        return f"{v_num:.3f}" if isinstance(v_old, str) else round(v_num, 3)

    # 전체 정렬(있다면) → 순서가 이상할 때를 방지
    try:
        segs_sorted = sorted(
            [s for s in segs if isinstance(s, dict)],
            key=lambda d: float(d.get("start", 0.0))
        )
    except Exception:
        segs_sorted = [s for s in segs if isinstance(s, dict)]

    prev_end = None
    new_segs = []
    for i, seg in enumerate(segs_sorted):
        s2 = dict(seg)

        # 원래 값
        st_old = seg.get("start", 0.0)
        et_old = seg.get("end", 0.0)

        try:
            st = float(st_old)
        except (TypeError, ValueError):
            st = 0.0
        try:
            et = float(et_old)
        except (TypeError, ValueError):
            et = st

        # 역전 방지
        if et < st:
            et = st

        # 이전 끝 이후로 최소 간격 보장
        if prev_end is not None:
            min_start = float(prev_end) + float(min_gap_sec)
            if st < min_start:
                # 길이는 유지하려 노력: start 밀면 end도 같이 민다
                dur = max(0.0, et - st)
                st = min_start
                et = st + dur

        # 기록(타입 보존)
        s2["start"] = _write(st_old, st)
        s2["end"] = _write(et_old, et)

        new_segs.append(s2)
        prev_end = et

    out["segments"] = new_segs
    return out
#####################################################################################
##########################음악분석 개선###########################################
#####################################################################################
from typing import List, Tuple, Dict, Any

def sync_lyrics_with_audio_whisperx(
    audio_path: str,
    lyrics_text: str,
    *,
    whisperx_model: str = "large-v3",
    batch_size: int = 16,
    min_line_sec: float = 0.20,
    round_ndigits: int = 3,
    start_preroll: float = 0.30,
    end_bias_sec: float = 2.50
) -> List[Dict[str, Any]]:
    """
    WhisperX(CPU) 단어 타임스탬프 → 편집거리 매칭 → 온셋 스냅(+보컬 tail 보정)으로
    각 가사 라인의 [start, end]를 산출한다.
    - 외부 이름 가리기 없음 / 지역 소문자 / 광범위 예외 금지 / 세미콜론 없음 / 없는 함수 호출 없음
    """

    # ---------------- 내부 헬퍼들 ----------------
    import re

    def _h_norm_ko_simple(text_in: str) -> str:
        lowered_local = (text_in or "").lower()
        kept_local = re.sub(r"[^0-9a-z가-힣\s]", " ", lowered_local)
        return re.sub(r"\s+", " ", kept_local).strip()

    def _h_split_lyrics_lines(src_text: str) -> List[str]:
        out_lines: List[str] = []
        for ln_raw in (src_text or "").splitlines():
            ln_stripped = ln_raw.strip()
            if not ln_stripped:
                continue
            if re.match(r"^\s*\[[^]]+]\s*$", ln_stripped):
                continue
            out_lines.append(ln_stripped)
        return out_lines

    def _h_edit_distance_dp(a_list: List[str], b_list: List[str]) -> np.ndarray:
        na_local = len(a_list)
        nb_local = len(b_list)
        dp_mat = np.zeros((na_local + 1, nb_local + 1), dtype=np.int32)
        for i_local in range(1, na_local + 1):
            dp_mat[i_local, 0] = i_local
        for j_local in range(1, nb_local + 1):
            dp_mat[0, j_local] = j_local
        for i_local in range(1, na_local + 1):
            ai_local = a_list[i_local - 1]
            for j_local in range(1, nb_local + 1):
                bj_local = b_list[j_local - 1]
                cost_local = 0 if ai_local == bj_local else 1
                up_local = int(dp_mat[i_local - 1, j_local]) + 1
                left_local = int(dp_mat[i_local, j_local - 1]) + 1
                diag_local = int(dp_mat[i_local - 1, j_local - 1]) + cost_local
                dp_mat[i_local, j_local] = min(up_local, left_local, diag_local)
        return dp_mat

    def _h_backtrack_path(dp_mat_in: np.ndarray,
                          a_list: List[str],
                          b_list: List[str]) -> List[Tuple[int, int]]:
        i_local = int(dp_mat_in.shape[0] - 1)
        j_local = int(dp_mat_in.shape[1] - 1)
        path_pairs: List[Tuple[int, int]] = []
        while i_local > 0 or j_local > 0:
            cand_list: List[Tuple[int, int, int]] = []
            if i_local > 0:
                cand_list.append((int(dp_mat_in[i_local - 1, j_local]) + 1, i_local - 1, j_local))
            if j_local > 0:
                cand_list.append((int(dp_mat_in[i_local, j_local - 1]) + 1, i_local, j_local - 1))
            if i_local > 0 and j_local > 0:
                sub_local = 0 if a_list[i_local - 1] == b_list[j_local - 1] else 1
                cand_list.append((int(dp_mat_in[i_local - 1, j_local - 1]) + sub_local, i_local - 1, j_local - 1))
            cand_list.sort(key=lambda x: int(x[0]))
            _, ni_local, nj_local = cand_list[0]
            path_pairs.append((ni_local, nj_local))
            i_local, j_local = ni_local, nj_local
        path_pairs.reverse()
        return path_pairs

    def _h_distribute_lines_time(
        lyric_lines_in: List[str],
        b_words_in: List[Tuple[str, float, float]],
        path_pairs_in: List[Tuple[int, int]],
        *,
        min_line_sec_in: float,
        round_ndigits_in: int
    ) -> List[Dict[str, Any]]:
        spans_local: List[Tuple[int, int]] = []
        token_idx_local = 0
        for ln_once in lyric_lines_in:
            toks_local = _h_norm_ko_simple(ln_once).split()
            if toks_local:
                s_local = token_idx_local
                e_local = token_idx_local + len(toks_local) - 1
                spans_local.append((s_local, e_local))
                token_idx_local = e_local + 1
            else:
                spans_local.append((token_idx_local, token_idx_local))

        tok2word_local: Dict[int, List[int]] = {}
        for ai_local, bj_local in path_pairs_in:
            if ai_local >= 0 and bj_local >= 0:
                if ai_local not in tok2word_local:
                    tok2word_local[ai_local] = []
                tok2word_local[ai_local].append(bj_local)

        items_synced: List[Dict[str, Any]] = []
        for (s_local, e_local), ln_text in zip(spans_local, lyric_lines_in):
            idxs_local: List[int] = []
            for k_local in range(s_local, e_local + 1):
                if k_local in tok2word_local:
                    idxs_local.extend(tok2word_local[k_local])
            idxs_local = sorted(set([z_local for z_local in idxs_local if 0 <= z_local < len(b_words_in)]))
            if idxs_local:
                starts_local = [float(b_words_in[z_local][1]) for z_local in idxs_local]
                ends_local = [float(b_words_in[z_local][2]) for z_local in idxs_local]
                st_local = min(starts_local)
                ed_local = max(ends_local)
            else:
                st_local = items_synced[-1]["end"] if items_synced else 0.0
                ed_local = st_local + min_line_sec_in
            if ed_local - st_local < min_line_sec_in:
                ed_local = st_local + min_line_sec_in
            items_synced.append({
                "line": ln_text,
                "start": round(st_local, round_ndigits_in),
                "end": round(ed_local, round_ndigits_in),
            })

        for idx_local in range(1, len(items_synced)):
            prev_end_local = float(items_synced[idx_local - 1]["end"])
            if float(items_synced[idx_local]["start"]) < prev_end_local:
                items_synced[idx_local]["start"] = prev_end_local
                if float(items_synced[idx_local]["end"]) < float(items_synced[idx_local]["start"]):
                    items_synced[idx_local]["end"] = round(float(items_synced[idx_local]["start"]) + min_line_sec_in, round_ndigits_in)
        return items_synced

    from typing import List, Dict, Any

    def _h_onset_snap_with_tail(
            audio_path_in: str,
            items_in: List[Dict[str, Any]],
            *,
            last_word_end_in: float,
            start_preroll_in: float,
            end_bias_sec_in: float,
            outro_pad_sec_in: float,
            round_ndigits_in: int
    ) -> List[Dict[str, Any]]:
        """
        온셋 스냅 + 보컬 VAD(HPSS + 보컬대역 멜에너지 + RMS)로 마지막 라인 end를 보정한다.
        - 마지막 라인 end = max(온셋 스냅, last_word_end+pad, 마지막 보컬 종료, duration-0.02)
        - 지역변수 소문자, 광범위 예외 없음, 세미콜론 없음, 없는 함수 호출 없음
        """

        if not items_in:
            return []

        # 기본 폴백 결과(오디오 분석 실패 시에도 동작)
        base_results: List[Dict[str, Any]] = []
        for item_loop in items_in[:-1]:
            s0 = float(item_loop["start"])
            e0 = float(item_loop["end"])
            s_adj = max(0.0, s0 - start_preroll_in)
            e_adj = max(s_adj, e0 + end_bias_sec_in)
            base_results.append({
                "line": item_loop["line"],
                "start": round(s_adj, round_ndigits_in),
                "end": round(e_adj, round_ndigits_in),
            })
        last_item = dict(items_in[-1])
        last_s = float(last_item["start"])
        last_e = float(last_item["end"])
        last_e_base = max(last_e + end_bias_sec_in, last_word_end_in + outro_pad_sec_in)
        if last_e_base < last_s:
            last_e_base = last_s
        final_last_e = last_e_base  # librosa 성공 시 갱신

        # ── librosa 및 하위 함수 "명시 임포트"(속성 접근 금지) ──
        try:
            import librosa as lb  # 로딩만 사용
            from librosa.util.exceptions import ParameterError  # CamelCase 그대로
            from librosa.feature import melspectrogram as lb_melspectrogram, rms as lb_rms
            from librosa.onset import onset_strength as lb_onset_strength, onset_detect as lb_onset_detect
            from librosa.effects import hpss as lb_hpss
            from librosa import power_to_db as lb_power_to_db, frames_to_time as lb_frames_to_time
            from librosa.core import mel_frequencies as lb_mel_frequencies
        except ImportError:
            last_item["end"] = round(final_last_e, round_ndigits_in)
            base_results.append(last_item)
            return base_results

        # 오디오 로드
        try:
            y, sr = lb.load(audio_path_in, sr=16000, mono=True)
        except (FileNotFoundError, ParameterError, RuntimeError, ValueError, OSError):
            last_item["end"] = round(final_last_e, round_ndigits_in)
            base_results.append(last_item)
            return base_results

        duration = float(len(y) / sr) if sr else 0.0

        # 온셋 검출
        try:
            onset_env = lb_onset_strength(y=y, sr=sr)
            onset_times = lb_onset_detect(onset_envelope=onset_env, sr=sr, units="time")
            on_arr = np.asarray([]) if onset_times is None else np.asarray(onset_times)
        except (ValueError, RuntimeError):
            on_arr = np.asarray([])

        def _prev_onset(t: float) -> float:
            if on_arr.size == 0:
                return t
            idx = int(np.searchsorted(on_arr, t, side="right")) - 1
            if idx < 0:
                idx = 0
            return float(on_arr[idx])

        def _next_onset(t: float) -> float:
            if on_arr.size == 0:
                return t
            idx = int(np.searchsorted(on_arr, t, side="right"))
            if idx >= on_arr.size:
                idx = on_arr.size - 1
            return float(on_arr[idx])

        # 보컬 강조(VAD 입력)
        try:
            y_harm, _y_perc = lb_hpss(y)
            y_vad = y_harm
        except Exception:
            y_vad = y

        # 멜 스펙트로그램 + RMS (하위 함수 직접 사용: feature 속성 접근 안 함)
        hop = 512
        win = 2048
        mel = lb_melspectrogram(y=y_vad, sr=sr, n_fft=win, hop_length=hop, n_mels=64)
        mel_db = lb_power_to_db(np.maximum(mel, 1e-10))
        n_mels = int(mel.shape[0])
        freqs = lb_mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr / 2.0)
        band_mask = (freqs >= 150.0) & (freqs <= 5000.0)
        band_energy = np.mean(mel_db[band_mask, :], axis=0)

        rms = lb_rms(y=y_vad, frame_length=win, hop_length=hop, center=True)[0]
        eps = 1e-9
        rms_db = 20.0 * np.log10(np.maximum(rms, eps))

        score_raw = 0.6 * band_energy + 0.4 * rms_db
        if score_raw.size >= 5:
            kernel = np.ones(5) / 5.0
            score_smooth = np.convolve(score_raw, kernel, mode="same")
        else:
            score_smooth = score_raw

        med = float(np.median(score_smooth))
        mad = float(np.median(np.abs(score_smooth - med))) + 1e-6
        th_high = med + 0.6 * mad
        th_low = med + 0.3 * mad
        frame_times = lb_frames_to_time(np.arange(score_smooth.shape[0]), sr=sr, hop_length=hop)

        voiced = False
        last_vocal_end = 0.0
        hold_sec = 0.30
        hold_frames = int(hold_sec * sr / hop) if sr else 0
        off_cnt = 0

        for i, val in enumerate(score_smooth):
            if not voiced and val >= th_high:
                voiced = True
                off_cnt = 0
            elif voiced:
                if val <= th_low:
                    off_cnt += 1
                    if off_cnt >= max(1, hold_frames):
                        last_vocal_end = float(frame_times[i])
                        voiced = False
                        off_cnt = 0
                else:
                    off_cnt = 0
                    last_vocal_end = float(frame_times[i])

        # 라인 스냅 재계산
        refined_results: List[Dict[str, Any]] = []
        for item_loop in items_in[:-1]:
            s0 = float(item_loop["start"])
            e0 = float(item_loop["end"])
            s_adj = max(0.0, _prev_onset(s0) - start_preroll_in)
            e_adj = max(s_adj, _next_onset(e0) + end_bias_sec_in)
            refined_results.append({
                "line": item_loop["line"],
                "start": round(s_adj, round_ndigits_in),
                "end": round(e_adj, round_ndigits_in),
            })

        base_results = refined_results

        # 마지막 라인 종료 확정(최대값)
        snap_end = max(last_s, _next_onset(last_e) + end_bias_sec_in)
        final_last_e = max(
            snap_end,
            last_word_end_in + outro_pad_sec_in,
            last_vocal_end,
            duration - 0.02 if duration > 0 else last_e_base
        )
        if final_last_e < last_s:
            final_last_e = last_s

        last_item["end"] = round(final_last_e, round_ndigits_in)
        base_results.append(last_item)
        return base_results

    # ---------------- /내부 헬퍼들 ----------------

    lyric_lines = _h_split_lyrics_lines(lyrics_text)
    if not lyric_lines:
        return []

    # 1) WhisperX 전사 + 정렬 (항상 CPU)
    import whisperx
    audio_arr = whisperx.load_audio(audio_path)
    device_fixed = "cpu"
    model_obj = whisperx.load_model(whisperx_model, device=device_fixed)
    result_obj = model_obj.transcribe(audio_arr, batch_size=batch_size)

    lang_code_local = result_obj.get("language") or "ko"
    align_model_obj, meta_obj = whisperx.load_align_model(language_code=lang_code_local, device=device_fixed)
    aligned_obj = whisperx.align(result_obj["segments"], align_model_obj, meta_obj, audio_arr, device_fixed)

    word_triplets: List[Tuple[str, float, float]] = []
    for seg_obj in aligned_obj.get("segments", []):
        for w_obj in seg_obj.get("words", []) or []:
            w_norm = _h_norm_ko_simple(str(w_obj.get("word", "")))
            if not w_norm:
                continue
            st_f = float(w_obj.get("start", 0.0))
            ed_f = float(w_obj.get("end", st_f + 0.12))
            word_triplets.append((w_norm, st_f, ed_f))

    # 2) 편집거리 기반 라인-단어 매칭
    toks_a = _h_norm_ko_simple(" ".join(lyric_lines)).split()
    toks_b = [w0 for (w0, _, _) in word_triplets]
    if not toks_a or not toks_b:
        fallback_items: List[Dict[str, Any]] = []
        for i_idx, ln_val in enumerate(lyric_lines):
            st_f = i_idx * min_line_sec
            ed_f = st_f + min_line_sec
            fallback_items.append({"line": ln_val, "start": round(st_f, round_ndigits), "end": round(ed_f, round_ndigits)})
        return fallback_items

    dp_mat_main = _h_edit_distance_dp(toks_a, toks_b)
    pairs_bt = _h_backtrack_path(dp_mat_main, toks_a, toks_b)
    rough_items = _h_distribute_lines_time(
        lyric_lines_in=lyric_lines,
        b_words_in=word_triplets,
        path_pairs_in=pairs_bt,
        min_line_sec_in=min_line_sec,
        round_ndigits_in=round_ndigits
    )

    # 3) 온셋 스냅 + 마지막 라인 tail 보정
    last_word_end_val = float(word_triplets[-1][2]) if word_triplets else float(rough_items[-1]["end"])
    snapped_items = _h_onset_snap_with_tail(
        audio_path_in=audio_path,
        items_in=rough_items,
        last_word_end_in=last_word_end_val,
        start_preroll_in=start_preroll,
        end_bias_sec_in=end_bias_sec,
        outro_pad_sec_in=3.0,          # 곡에 따라 2.0~6.0로 조정 가능
        round_ndigits_in=round_ndigits
    )
    return snapped_items




def estimate_vocal_end_sec(audio_path_in: str) -> float:
    """
    파일 전체에서 '보컬이 끝나는 시점'을 추정해 초 단위로 반환.
    - HPSS로 하모닉(보컬 성분) 강조
    - 보컬대역(150~5000Hz) 멜 에너지 + RMS 결합
    - 이동평균 + 히스테리시스(상/하한)로 무가창(outro) 진입 검출
    - librosa 미설치/로드 실패 시 0.0 반환
    """
    try:
        import librosa as lb
        from librosa.util.exceptions import ParameterError
        from librosa.feature import melspectrogram as lb_melspectrogram, rms as lb_rms
        from librosa.effects import hpss as lb_hpss
        from librosa import power_to_db as lb_power_to_db, frames_to_time as lb_frames_to_time
        from librosa.core import mel_frequencies as lb_mel_frequencies
    except ImportError:
        return 0.0

    try:
        y, sr = lb.load(audio_path_in, sr=16000, mono=True)
    except (FileNotFoundError, ParameterError, RuntimeError, ValueError, OSError):
        return 0.0

    # 보컬 강조 신호
    try:
        y_harm, _y_perc = lb_hpss(y)
        y_vad = y_harm
    except (ValueError, RuntimeError):
        y_vad = y

    # 보컬 대역 멜 + RMS
    hop = 512
    win = 2048
    mel = lb_melspectrogram(y=y_vad, sr=sr, n_fft=win, hop_length=hop, n_mels=64)
    mel_db = lb_power_to_db(np.maximum(mel, 1e-10))
    n_mels = int(mel.shape[0])
    freqs = lb_mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr / 2.0)
    band_mask = (freqs >= 150.0) & (freqs <= 5000.0)
    band_energy = np.mean(mel_db[band_mask, :], axis=0)

    rms = lb_rms(y=y_vad, frame_length=win, hop_length=hop, center=True)[0]
    eps = 1e-9
    rms_db = 20.0 * np.log10(np.maximum(rms, eps))

    score_raw = 0.6 * band_energy + 0.4 * rms_db
    if score_raw.size >= 5:
        kernel = np.ones(5) / 5.0
        score_smooth = np.convolve(score_raw, kernel, mode="same")
    else:
        score_smooth = score_raw

    # 적응형 임계 + 홀드
    med = float(np.median(score_smooth))
    mad = float(np.median(np.abs(score_smooth - med))) + 1e-6
    th_high = med + 0.6 * mad
    th_low = med + 0.3 * mad
    frame_times = lb_frames_to_time(np.arange(score_smooth.shape[0]), sr=sr, hop_length=hop)

    voiced = False
    last_vocal_end = 0.0
    hold_sec = 0.30
    hold_frames = int(hold_sec * sr / hop) if sr else 0
    off_cnt = 0

    for i, val in enumerate(score_smooth):
        if not voiced and val >= th_high:
            voiced = True
            off_cnt = 0
        elif voiced:
            if val <= th_low:
                off_cnt += 1
                if off_cnt >= max(1, hold_frames):
                    last_vocal_end = float(frame_times[i])
                    voiced = False
                    off_cnt = 0
            else:
                off_cnt = 0
                last_vocal_end = float(frame_times[i])

    return float(last_vocal_end)

from typing import List, Dict, Any

def add_korean_lines_to_items(items_in: List[Dict[str, Any]], lyrics_text: str) -> List[Dict[str, Any]]:
    """
    segments(each: dict)에 'line_ko'를 가사 원문(한글)으로 주입한다.
    - [Intro], [Hook] 등 대괄호 메타 라인은 제거
    - 빈 줄 제거
    - items 길이와 가사 줄 수가 달라도 앞에서부터 순차 매칭 (초과분은 빈 문자열)
    - 함수 내 변수는 전부 소문자, 광범위 예외 미사용, 세미콜론 없음
    """
    if not isinstance(items_in, list):
        return []

    raw_lines: List[str] = []
    for raw in (lyrics_text or "").splitlines():
        line = (raw or "").strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            continue
        # 로마자/영문 변환을 하지 않고, 원문 그대로 유지
        raw_lines.append(line)

    out_list: List[Dict[str, Any]] = []
    line_idx = 0
    n_lines = len(raw_lines)

    for seg in items_in:
        seg_new = dict(seg)
        if line_idx < n_lines:
            seg_new["line_ko"] = str(raw_lines[line_idx])
            line_idx += 1
        else:
            seg_new["line_ko"] = ""
        out_list.append(seg_new)

    return out_list



from typing import List, Dict, Any

def adjust_last_end_with_vocal_end(
    items_in: List[Dict[str, Any]],
    vocal_end_sec: float,
    *,
    round_ndigits_in: int = 3
) -> List[Dict[str, Any]]:
    """
    기존 줄별 결과의 '마지막 라인 end'만 보컬 종료 시점으로 보정.
    - 구조/키 보존. vocal_end_sec가 현재 end보다 뒤이고 start보다 크면 반영.
    """
    if not items_in:
        return []
    result_items = [dict(x) for x in items_in]
    last = result_items[-1]
    try:
        st = float(last.get("start", 0.0))
        ed = float(last.get("end", 0.0))
        ve = float(vocal_end_sec or 0.0)
    except (TypeError, ValueError):
        return result_items
    if ve > st and ve > ed:
        last["end"] = round(ve, round_ndigits_in)
    return result_items

def estimate_vocal_end_sec_silero(audio_path_in: str,
                                  *,
                                  threshold_in: float = 0.5,
                                  min_speech_sec_in: float = 0.20) -> float:
    """
    Silero VAD로 audio_path_in에서 '사람 발성' 구간을 검출하고
    마지막 구간의 끝 시각(초)을 반환한다. 실패 시 0.0 반환.
    - CPU에서 매우 빠름, 정확도 높음 (공식 repo 참조)
    """
    try:
        import torch
        import torchaudio
    except ImportError:
        return 0.0

    # silero 모델 로드 (onnx/jit 둘 다 가능하나 기본 jit)
    try:
        # snakers4/silero-vad 권장 로드 방법
        model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad",
                                      model="silero_vad",
                                      force_reload=False,
                                      trust_repo=True)
        get_speech_timestamps = utils.get("get_speech_timestamps")
        read_audio = utils.get("read_audio")
        vad_sr = 16000
    except Exception:
        return 0.0

    try:
        wav = read_audio(audio_path_in, sampling_rate=vad_sr)
    except Exception:
        # torchaudio로 재시도
        try:
            waveform, sr = torchaudio.load(audio_path_in)  # type: ignore
            if sr != vad_sr:
                waveform = torchaudio.functional.resample(waveform, sr, vad_sr)  # type: ignore
            wav = waveform.squeeze().numpy()
        except Exception:
            return 0.0

    try:
        # VAD 실행
        speech_ts = get_speech_timestamps(wav, model,
                                          sampling_rate=vad_sr,
                                          threshold=threshold_in,
                                          min_speech_duration=int(min_speech_sec_in * 1000.0))
    except Exception:
        return 0.0

    if not speech_ts:
        return 0.0

    last = speech_ts[-1]
    try:
        last_end = float(last["end"]) / float(vad_sr)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return 0.0
    return last_end

def cap_last_line_end_with_vad(items_in: List[Dict[str, Any]],
                               last_speech_end_in: float,
                               *,
                               pad_sec_in: float = 0.30,
                               round_ndigits_in: int = 3) -> List[Dict[str, Any]]:
    """
    줄별 결과에서 '마지막 라인 end'를 VAD 기반 마지막 가창 종료로 상한(cap).
    - end = min(end, last_speech_end + pad)
    - last_speech_end가 start보다 이르면 보정하지 않음
    """
    if not items_in:
        return []
    out = [dict(x) for x in items_in]
    last = out[-1]
    try:
        st = float(last.get("start", 0.0))
        ed = float(last.get("end", 0.0))
        le = float(last_speech_end_in or 0.0) + float(pad_sec_in or 0.0)
    except (TypeError, ValueError):
        return out
    if le > st and ed > st:
        if ed > le:
            last["end"] = round(le, round_ndigits_in)
    return out

def estimate_vocal_end_sec_energy(audio_path_in: str,
                                  *,
                                  anchor_sec_in: float = 0.0,
                                  low_ratio_in: float = 0.22,
                                  hold_sec_in: float = 1.8) -> float:
    """
    노래(가창)에서 마지막 종료점을 '에너지 하강'으로 추정한다.
    - HPSS 하모닉 성분 + 보컬대역(150~5000Hz) 멜에너지 + RMS를 결합한 점수를 사용
    - anchor_sec_in 이후 구간에서 점수가 충분히 낮아지고(비율로) 일정 시간(hold_sec_in) 유지되는 첫 지점을 '가창 종료'로 판단
    - librosa 미설치/로드 실패 시 0.0 반환
    - 모든 지역 변수는 소문자, 세미콜론 없음, 광범위 예외 없음
    """
    try:
        import librosa as lb
        from librosa.util.exceptions import ParameterError
        from librosa.feature import melspectrogram as lb_melspectrogram, rms as lb_rms
        from librosa.effects import hpss as lb_hpss
        from librosa import power_to_db as lb_power_to_db, frames_to_time as lb_frames_to_time
        from librosa.core import mel_frequencies as lb_mel_frequencies
    except ImportError:
        return 0.0

    try:
        y, sr = lb.load(audio_path_in, sr=16000, mono=True)
    except (FileNotFoundError, ParameterError, RuntimeError, ValueError, OSError):
        return 0.0

    # 하모닉 강조(실패 시 원 신호)
    try:
        y_harm, _y_perc = lb_hpss(y)
        y_src = y_harm
    except (ValueError, RuntimeError):
        y_src = y

    hop = 512
    win = 2048

    mel = lb_melspectrogram(y=y_src, sr=sr, n_fft=win, hop_length=hop, n_mels=64)
    mel_db = lb_power_to_db(np.maximum(mel, 1e-10))
    n_mels = int(mel.shape[0])
    freqs = lb_mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr / 2.0)
    band_mask = (freqs >= 150.0) & (freqs <= 5000.0)
    band_energy = np.mean(mel_db[band_mask, :], axis=0)

    rms = lb_rms(y=y_src, frame_length=win, hop_length=hop, center=True)[0]
    eps = 1e-9
    rms_db = 20.0 * np.log10(np.maximum(rms, eps))

    score_raw = 0.6 * band_energy + 0.4 * rms_db
    if score_raw.size >= 5:
        kernel = np.ones(5) / 5.0
        score_smooth = np.convolve(score_raw, kernel, mode="same")
    else:
        score_smooth = score_raw

    ft = lb_frames_to_time(np.arange(score_smooth.shape[0]), sr=sr, hop_length=hop)

    # anchor 이후 구간만 평가
    if anchor_sec_in > 0.0:
        idx0 = int(np.searchsorted(ft, anchor_sec_in, side="left"))
    else:
        idx0 = 0

    # 기준 레벨: anchor~anchor+4초 구간의 중앙값(없으면 전체 중앙값)
    idx1 = int(np.searchsorted(ft, min(anchor_sec_in + 4.0, ft[-1] if ft.size else 0.0), side="left"))
    local_slice = score_smooth[idx0:max(idx0 + 1, idx1)]
    if local_slice.size == 0:
        local_slice = score_smooth
    ref = float(np.median(local_slice)) if local_slice.size else 0.0
    if ref <= 0.0:
        ref = float(np.median(score_smooth)) if score_smooth.size else 0.0

    if ref == 0.0 or score_smooth.size == 0 or ft.size == 0:
        return 0.0

    low_th = ref * float(low_ratio_in)
    hold_frames = int(float(hold_sec_in) * sr / hop) if sr else 0

    run = 0
    cand_end = 0.0
    for i in range(idx0, score_smooth.size):
        if score_smooth[i] <= low_th:
            run += 1
            if run >= max(1, hold_frames):
                cand_end = float(ft[i])
                break
        else:
            run = 0

    return float(cand_end if cand_end > 0.0 else 0.0)


# audio_sync.py 파일에서 이 함수를 찾아 아래 내용으로 전체를 교체하세요.

def estimate_vocal_end_from_vocal_stem(audio_path_in: str,
                                       *,
                                       hold_sec_in: float = 0.60,
                                       drop_db_in: float = 23.0) -> float:
    """
    demucs가 생성한 '보컬 스템(vocals*.wav)'만을 분석하여 마지막 가창 종료 시점을 반환한다.
    - 개선된 로직: RMS 에너지를 분석하여, 끝에서부터 역으로 탐색해
      마지막으로 유의미한 에너지가 나타나는 지점을 가창 종료로 판단한다.
      이는 긴 잔향이나 페이드아웃에 더 강건하다.
    - 실패 시 0.0 반환
    - 지역 변수 소문자, 광범위 예외 없음
    """
    import glob
    from pathlib import Path

    try:
        base_dir = Path(audio_path_in).resolve().parent
    except (OSError, RuntimeError, ValueError):
        return 0.0

    try:
        # demucs_out 폴더 내 모든 vocals.wav 파일을 재귀적으로 찾습니다.
        candidates = glob.glob(str(base_dir / "demucs_out" / "**" / "vocals*.wav"), recursive=True)
    except (OSError, RuntimeError, ValueError):
        candidates = []

    if not candidates:
        return 0.0

    # 가장 신뢰도 높은 스템(파일 크기 최대)을 선택
    try:
        stem_path = max((Path(p) for p in candidates if p.lower().endswith(".wav")),
                        key=lambda p: p.stat().st_size)
    except (ValueError, OSError):
        return 0.0

    try:
        import librosa as lb
        from librosa.feature import rms as lb_rms
        from librosa import frames_to_time as lb_frames_to_time
    except ImportError:
        return 0.0

    try:
        y_local, sr_local = lb.load(str(stem_path), sr=16000, mono=True)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return 0.0

    hop = 512
    win = 2048
    try:
        r_local = lb_rms(y=y_local, frame_length=win, hop_length=hop, center=True)[0]
    except (RuntimeError, ValueError):
        return 0.0

    eps = 1e-12
    r_db = 20.0 * np.log10(np.maximum(r_local, eps))

    if r_db.size == 0:
        return 0.0

    # 최고점에서 drop_db_in 만큼 낮은 값을 임계치로 설정
    peak_db = float(np.max(r_db))
    threshold_db = peak_db - float(drop_db_in)

    frame_times = lb_frames_to_time(np.arange(r_db.shape[0]), sr=sr_local, hop_length=hop)

    # 끝에서부터 역방향으로 탐색하여 임계치를 넘는 마지막 프레임을 찾습니다.
    last_vocal_frame_idx = -1
    for i in range(r_db.size - 1, -1, -1):
        if r_db[i] > threshold_db:
            last_vocal_frame_idx = i
            break

    if last_vocal_frame_idx == -1:
        return 0.0

    # 마지막 유의미한 보컬 지점의 시간에 약간의 패딩(hold_sec)을 더해 반환
    last_end_time = float(frame_times[last_vocal_frame_idx]) + float(hold_sec_in)

    # 오디오 전체 길이를 넘지 않도록 제한
    duration_total = float(len(y_local) / sr_local) if sr_local > 0 else 0.0
    if duration_total > 0 and last_end_time > duration_total:
        last_end_time = duration_total

    return float(last_end_time if last_end_time > 0.0 else 0.0)

def estimate_last_lyric_end_with_mfa(
    audio_path: str,
    lyrics_text: str,
    *,
    mfa_bin: str = "mfa",
    acoustic_model: str = "korean_mfa",      # MFA Korean acoustic model id
    dictionary: str = "korean_mfa",          # MFA Korean dictionary id
    tmp_prefix: str = "mfa_tmp_align"
) -> float:
    """
    MFA(Kaldi 기반 강제정렬)로 전체 가사를 오디오에 정렬하고,
    마지막 align된 토큰(단어/음소)의 종료시각(초)을 반환한다.
    - MFA 설치 및 한국어 모델/사전이 있어야 동작. 없으면 0.0 반환.
    - 반환값이 0.0이면 사용하지 않으면 된다(보완 소스).
    참고: https://montreal-forced-aligner.readthedocs.io/  (한국어 모델/사전 제공)
    """
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    # 0) 사전 조건 체크: mfa 실행 가능?
    try:
        proc = subprocess.run([mfa_bin, "--version"], capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return 0.0
    except (OSError, FileNotFoundError):
        return 0.0

    # 1) 임시 디렉터리 구성 (MFA는 폴더 단위 정렬을 권장)
    tmp_dir_obj = tempfile.TemporaryDirectory(prefix=tmp_prefix)
    tmp_dir = Path(tmp_dir_obj.name)
    data_dir = tmp_dir / "data"
    out_dir = tmp_dir / "out"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # MFA 입력: audio(.wav), lab(txt) 파일명은 동일 베이스여야 편함
    wav_src = Path(audio_path).resolve()
    if not wav_src.exists():
        tmp_dir_obj.cleanup()
        return 0.0
    wav_dst = data_dir / "utt.wav"
    shutil.copy2(str(wav_src), str(wav_dst))

    # 텍스트: 한 줄 가사 전체를 단일 발화로 두면 문맥이 길어도 강제정렬은 가능
    # (필요하면 줄 단위로 나누어 여러 발화로 만들 수도 있음)
    lab_path = data_dir / "utt.lab"
    try:
        # 메타 라인([Hook] 등) 제거
        lines = []
        for raw in (lyrics_text or "").splitlines():
            s = (raw or "").strip()
            if not s:
                continue
            if s.startswith("[") and s.endswith("]"):
                continue
            lines.append(s)
        lab_path.write_text(" ".join(lines), encoding="utf-8")
    except Exception:
        tmp_dir_obj.cleanup()
        return 0.0

    # 2) MFA 실행: align -> TextGrid 생성
    # 모델/사전이 로컬에 받아져 있어야 함(없으면 mfa model download로 사전 설치 필요)
    # - acoustic_model, dictionary 이름은 MFA 문서의 한국어 모델 id를 사용
    #   (예: 'Korean MFA acoustic model', 'Korean MFA dictionary')
    #   설치 가이드는 공식 문서 참고.
    try:
        cmd = [
            mfa_bin, "align",
            str(data_dir),
            dictionary,
            acoustic_model,
            str(out_dir),
            "--clean"
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            # MFA가 실패하면 0.0 반환(보조 신호로만 쓰므로 침묵)
            tmp_dir_obj.cleanup()
            return 0.0
    except (OSError, FileNotFoundError, subprocess.SubprocessError):
        tmp_dir_obj.cleanup()
        return 0.0

    # 3) TextGrid 파싱: 마지막 word tier 또는 phones tier의 마지막 end 시각
    # 생성 파일명은 utt.TextGrid일 가능성이 큼
    tg_path = out_dir / "utt.TextGrid"
    if not tg_path.exists():
        tmp_dir_obj.cleanup()
        return 0.0

    try:
        # TextGrid를 파싱(파서 없이도 간단히 정규식으로 마지막 'xmax' 뽑기)
        text = tg_path.read_text(encoding="utf-8", errors="ignore")
        # 가장 마지막 xmax(=segment end)를 찾는다
        import re
        xmax_vals: List[float] = []
        for m in re.finditer(r'\bxmax\s*=\s*([0-9]*\.?[0-9]+)', text):
            try:
                xmax_vals.append(float(m.group(1)))
            except ValueError:
                continue
        last = max(xmax_vals) if xmax_vals else 0.0
        tmp_dir_obj.cleanup()
        return float(last if last > 0 else 0.0)
    except Exception:
        tmp_dir_obj.cleanup()
        return 0.0


def merge_last_end_with_external(
    items_in: List[Dict[str, Any]],
    ext_last_end: float,
    *,
    minimum_delta: float = 0.10,
    round_ndigits: int = 3
) -> List[Dict[str, Any]]:
    """
    existing 줄별 결과(items_in)에 대해 '마지막 라인 end'만 외부 추정치(ext_last_end)로 보정.
    - ext_last_end가 현재 end보다 충분히 뒤(=minimum_delta 이상)이고 start보다 크면 반영
    - 구조/키는 보존
    """
    if not items_in:
        return []
    out = [dict(x) for x in items_in]
    last = out[-1]
    try:
        st = float(last.get("start", 0.0))
        ed = float(last.get("end", 0.0))
        ve = float(ext_last_end or 0.0)
    except (TypeError, ValueError):
        return out
    if ve > st and (ve - ed) >= float(minimum_delta):
        last["end"] = round(ve, round_ndigits)
    return out




