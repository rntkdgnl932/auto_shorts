# app/image_movie_docs.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from textwrap import dedent
import re
from pathlib import Path
import json
# 유연 import
try:
    # 패키지 실행
    from app import settings as settings   # 소문자 별칭 유지
    from app.utils import load_json, save_json
except ImportError:
    # 단독 실행
    import settings as settings            # 소문자 별칭 유지
    from utils import load_json, save_json  # type: ignore

S = settings  # noqa: N816  # (하위호환: 기존 코드가 S를 참조해도 동작)




_FACE_RULES = "front-facing, eyes looking at camera, full face visible; photorealistic; no profile, no back view"
_COMPOSITION = "vertical 9:16, face-centered, cinematic grading"
_VARIATIONS = [
    "subtle lighting change",
    "micro head tilt (still facing camera)",
    "background depth-of-field shift",
    "slight camera height change",
    "gentle lens breathing",
]

def _load_json(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return default

def _save_json(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)



def _read_total_seconds(project_dir: str, *, default_seconds: int = 60) -> int:
    """project.json에서 target_seconds > time*60 > default_seconds 순으로 총 길이(초)를 읽는다."""
    try:
        from app.utils import load_json  # 패키지 실행
    except ImportError:
        from utils import load_json      # 단독 실행

    pj = Path(project_dir) / "project.json"
    meta = load_json(pj, {}) or {}
    try:
        if "target_seconds" in meta:
            return int(max(1, int(meta["target_seconds"])))
    except (TypeError, ValueError):
        pass
    try:
        if "time" in meta:
            return int(max(1, int(meta["time"])) * 60)
    except (TypeError, ValueError):
        pass
    return int(default_seconds)


def _build_base_timeline_from_scenes(
    scenes: List[Dict[str, Any]],
    total_dur: float,
) -> List[Dict[str, Any]]:
    """scenes 배열로 기본 타임라인을 만든다(가중 분배→없으면 균등)."""
    total_dur = float(max(0.5, total_dur))
    n = max(1, len(scenes))
    try:
        raw_durs = [float(s.get("duration") or 0.0) for s in scenes]
    except (TypeError, ValueError):
        raw_durs = [0.0] * n

    sum_dur = float(sum(d for d in raw_durs if d > 0.0))
    if sum_dur <= 0.0:
        part = total_dur / float(n)
        alloc = [part] * n
    else:
        scale = total_dur / sum_dur
        alloc = [max(0.01, d * scale) for d in raw_durs]

    timeline: List[Dict[str, Any]] = []
    cur = 0.0
    for idx, sc in enumerate(scenes, 1):
        section = str(sc.get("section") or "").strip().lower() or "scene"
        sid = sc.get("id") or f"t_{idx:02d}"
        dur = float(alloc[idx - 1])
        start = cur
        end = min(total_dur, start + dur)
        timeline.append({
            "section": section,
            "label": section.capitalize(),
            "id": str(sid),
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(max(0.0, end - start), 3),
        })
        cur = end

    if timeline:
        timeline[-1]["end"] = round(total_dur, 3)
        timeline[-1]["duration"] = round(
            float(timeline[-1]["end"]) - float(timeline[-1]["start"]), 3
        )
    return timeline


def _with_intro_outro_ratio(
    timeline: List[Dict[str, Any]],
    total_dur: float,
    *,
    intro_ratio: float = 0.10,
    outro_ratio: float = 0.10,
    min_intro: float = 1.5,
    min_outro: float = 1.5,
    max_intro: Optional[float] = None,
    max_outro: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """타임라인에 인트로/아웃트로를 비율 기반으로 반영한다."""
    total_dur = float(max(0.5, total_dur))
    has_intro = any(str(b.get("section", "")).lower().startswith("intro") for b in timeline)
    has_outro = any(str(b.get("section", "")).lower().startswith("outro") for b in timeline)

    hi_intro = max_intro if max_intro is not None else max(0.5, total_dur * 0.25)
    hi_outro = max_outro if max_outro is not None else max(0.5, total_dur * 0.25)

    want_intro = float(max(0.0, intro_ratio)) * total_dur
    want_outro = float(max(0.0, outro_ratio)) * total_dur

    intro_len = float(max(min_intro, min(hi_intro, want_intro)))
    outro_len = float(max(min_outro, min(hi_outro, want_outro)))

    intro_len = float(max(0.0, min(intro_len, max(0.0, total_dur - 0.01))))

    new_timeline: List[Dict[str, Any]] = []
    if not has_intro and intro_len > 0.0:
        new_timeline.append({
            "section": "intro",
            "label": "Intro",
            "id": "intro",
            "start": 0.0,
            "end": round(intro_len, 3),
            "duration": round(intro_len, 3),
        })
        for blk in timeline:
            s = float(blk.get("start", 0.0)) + intro_len
            e = float(blk.get("end", 0.0)) + intro_len
            s = max(s, intro_len)
            e = min(e, total_dur)
            if e <= s:
                continue
            b2 = dict(blk)
            b2["start"] = round(s, 3)
            b2["end"] = round(e, 3)
            b2["duration"] = round(e - s, 3)
            new_timeline.append(b2)
    else:
        new_timeline = list(timeline)

    if not has_outro and outro_len > 0.0:
        start = max(0.0, total_dur - outro_len)
        end = total_dur
        if new_timeline:
            last = new_timeline[-1]
            last_start = float(last.get("start", 0.0))
            last_end = float(last.get("end", 0.0))
            if last_end > start:
                clipped = max(last_start, start)
                last["end"] = round(clipped, 3)
                last["duration"] = round(max(0.0, clipped - last_start), 3)
                if last["duration"] <= 1e-6:
                    new_timeline.pop()
        new_timeline.append({
            "section": "outro",
            "label": "Outro",
            "id": "outro",
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
        })

    if new_timeline:
        new_timeline[-1]["end"] = round(total_dur, 3)
        new_timeline[-1]["duration"] = round(
            float(new_timeline[-1]["end"]) - float(new_timeline[-1]["start"]), 3
        )
    return new_timeline


def apply_intro_outro_to_story_json(
    project_dir: str,
    *,
    intro_ratio: float = 0.10,
    outro_ratio: float = 0.10,
) -> str:
    """
    (기존과 동일한 시그니처/동작)
    project_dir/story.json 을 제자리 덮어쓰기. 내부적으로 파일지정형 함수를 사용.
    """
    proj = Path(project_dir)
    src = proj / "story.json"
    dest = proj / "story.json"
    return apply_intro_outro_to_story_file(
        src,
        dest,
        intro_ratio=intro_ratio,
        outro_ratio=outro_ratio,
    )

def apply_intro_outro_to_image_json(
    project_dir: str,
    *,
    intro_ratio: float = 0.10,
    outro_ratio: float = 0.10,
) -> str:
    """image.json → story.scenes 기반으로 timeline 추가/갱신."""
    try:
        from app.utils import load_json, save_json
    except ImportError:
        from utils import load_json, save_json

    proj = Path(project_dir)
    story = load_json(proj / "story.json", {}) or {}
    scenes = list(story.get("scenes") or [])
    total_sec = float(_read_total_seconds(project_dir))

    base = _build_base_timeline_from_scenes(scenes, total_sec)
    timeline = _with_intro_outro_ratio(base, total_sec, intro_ratio=intro_ratio, outro_ratio=outro_ratio)

    image_path = proj / "image.json"
    image_doc = load_json(image_path, {}) or {}
    image_doc["timeline"] = timeline
    save_json(image_path, image_doc)
    return str(image_path)


def apply_intro_outro_to_movie_json(
    project_dir: str,
    *,
    intro_ratio: float = 0.10,
    outro_ratio: float = 0.10,
) -> str:
    """movie.json → items.duration(있으면) 우선, 없으면 story.scenes 기반 timeline 추가/갱신."""
    try:
        from app.utils import load_json, save_json
    except ImportError:
        from utils import load_json, save_json

    proj = Path(project_dir)
    movie_path = proj / "movie.json"
    movie_doc = load_json(movie_path, {}) or {}
    items = list(movie_doc.get("items") or [])

    total_sec = float(_read_total_seconds(project_dir))
    if items and any(float(i.get("duration") or 0.0) > 0.0 for i in items):
        pseudo_scenes = []
        for idx, it in enumerate(items, 1):
            pseudo_scenes.append({
                "id": it.get("id") or f"t_{idx:02d}",
                "section": "scene",
                "duration": float(it.get("duration") or 0.0),
            })
        base = _build_base_timeline_from_scenes(pseudo_scenes, total_sec)
    else:
        story = load_json(proj / "story.json", {}) or {}
        scenes = list(story.get("scenes") or [])
        base = _build_base_timeline_from_scenes(scenes, total_sec)

    timeline = _with_intro_outro_ratio(base, total_sec, intro_ratio=intro_ratio, outro_ratio=outro_ratio)
    movie_doc["timeline"] = timeline
    save_json(movie_path, movie_doc)
    return str(movie_path)

# FILE: app/image_movie_docs.py
# WHAT: 아래 3개 함수만 "그대로 바꿔치기" 하세요. (기존 동일 함수 대체)
# WHY:
#  - UI 드롭다운 값(ui_prefs: W/H/FPS)을 실제 image.json / movie.json에 반영
#  - i2v 모션 힌트(효과 → 카메라 모션) 주입
#  - defaults 섹션 주입(컨슈머가 참조 가능)
#  - 안전 가드 및 경계 케이스 처리

def build_image_json(project_dir: str,
                     hair_map: dict | None = None,
                     _workflow_dir: Path | None = None):
    """
    story.json → image.json 빌드
    - ui_prefs(image_size, movie_fps 등) 일부 반영: image.defaults.width/height
    - 각 item에 t2i_prompt(캐릭터 일관성 가이드 포함) 기록
    """
    proj = Path(project_dir)
    story = _load_json(proj / "story.json", {}) or {}

    # ★ UI 드롭다운 값 로드
    try:
        prefs = load_ui_prefs_for_audio(story.get("audio", ""))
    except Exception:
        prefs = {"image_w": 832, "image_h": 1472, "movie_fps": 24, "overlap": 12}

    char_styles = dict(hair_map or story.get("character_styles") or {})
    scenes = story.get("scenes") or []

    out = {
        "title": story.get("title", ""),
        "audio": story.get("audio", ""),
        "characters": char_styles,
        # defaults를 넣어줘서 생성 파이프라인에서 참조 가능
        "defaults": {
            "image": {"width": int(prefs["image_w"]), "height": int(prefs["image_h"]), "negative": "@global"}
        },
        "items": []
    }

    for i, sc in enumerate(scenes, 1):
        sid = sc.get("id") or f"t_{i:02d}"
        chars = sc.get("characters") or []
        style_str = ", ".join(f"{cid}: {char_styles.get(cid,'')}"
                              for cid in chars if cid in char_styles)
        t2i_prompt = (
            f"{sc.get('scene','')}; {sc.get('prompt','')}. "
            f"[Face rules: {_FACE_RULES}]. {_COMPOSITION}. "
            f"Character consistency: {style_str}"
        )

        out["items"].append({
            "id": sid,
            "section": (sc.get("section") or "").lower() or "scene",
            "img_file": sc.get("img_file"),
            "characters": chars,
            "t2i_prompt": t2i_prompt
        })

    return _save_json(proj / "image.json", out)


def build_movie_json(project_dir: str,
                     hair_map: dict | None = None,
                     _workflow_dir: Path | None = None):
    """
    story.json → movie.json 빌드
    - ui_prefs(movie_fps/overlap) 반영: movie.defaults.target_fps/overlap_frames
    - 효과(effect)→i2v 모션 힌트 주입
    - 섹션 경계 전환 힌트/씬 전환 플래그(screen_transition) 병합
    """
    proj = Path(project_dir)
    story = _load_json(proj / "story.json", {}) or {}

    # ★ UI 드롭다운 값 로드
    try:
        prefs = load_ui_prefs_for_audio(story.get("audio", ""))
    except Exception:
        prefs = {"image_w": 832, "image_h": 1472, "movie_fps": 24, "overlap": 12}

    char_styles = dict(hair_map or story.get("character_styles") or {})
    scenes = story.get("scenes") or []

    # 우선순위: story.fps > ui_prefs.movie_fps
    fps = int(story.get("fps") or prefs["movie_fps"])

    out = {
        "title": story.get("title", ""),
        "audio": story.get("audio", ""),
        "fps": fps,
        "defaults": {
            "movie": {"target_fps": int(prefs["movie_fps"]), "overlap_frames": int(prefs["overlap"]), "negative": "@global"}
        },
        "items": []
    }

    prev_section = None
    for i, sc in enumerate(scenes, 1):
        sid = sc.get("id") or f"t_{i:02d}"
        section = (sc.get("section") or "").lower() or "scene"
        big_transition = prev_section is not None and section != prev_section
        prev_section = section

        # 전환 힌트 우선순위: 섹션 경계 > screen_transition 플래그
        if big_transition:
            transition_hint = "Impactful transition at boundary (whip-pan / flash cut / strong light-leak)."
        elif sc.get("screen_transition"):
            transition_hint = "Soft cross-dissolve or light-leak."
        else:
            transition_hint = ""

        # 효과 → 모션 힌트
        motion = _effect_to_motion(sc.get("effect") or [], bool(sc.get("screen_transition")))
        variety = _VARIATIONS[(i - 1) % len(_VARIATIONS)]
        style_str = ", ".join(
            f"{cid}: {char_styles.get(cid,'')}"
            for cid in (sc.get("characters") or [])
            if cid in char_styles
        )

        i2v_prompt = (
            f"{sc.get('scene','')}; {sc.get('prompt','')}. "
            f"[Face rules: {_FACE_RULES}]. {_COMPOSITION}. "
            f"Motion: {motion}. Variation: {variety}. {transition_hint} "
            f"Character consistency: {style_str}"
        )

        out["items"].append({
            "id": sid,
            "duration": float(sc.get("duration") or 0.75),
            "source_image": sc.get("img_file"),
            "i2v_prompt": i2v_prompt
        })

    return _save_json(proj / "movie.json", out)





def normalize_to_v11(story: dict) -> dict:
    """
    v1.1 스펙 정규화 (UI 설정 반영 + lyrics_sections + per-scene lyric 분배)
    - paths: img_name_pattern / clip_name_pattern 두 개만 유지(동의어 키 제거)
    - title 바로 다음에 전체 가사 필드 'lyrics' 추가
      * 'lyrics'는 순수 가사만: 개행 제거, [verse]/[intro]/[...] 태그 제거, 공백 정리
    - 씬 시간 재배치(최소 길이 보정 + 반올림, 'B안': project.json time 우선)
    - prompt 규칙 보강(배경/효과/동작) — GPT 비의존
    - characters: "id:index" 문자열 + 객체형 [{"id":..,"index":..}] 병기
    - lyrics_sections 생성 후, **가사를 씬별로 분배**해서 scene["lyric"]에 나눠 넣음
      * 분배 규칙: 가사 블록 텍스트를 줄 단위로 나눠, 그 블록 시간대에 속하는 씬들에 순서대로 균등 배정
      * 경계는 [start, end)로 처리 → 블록 end 시각에 딱 걸치는 씬은 **다음 블록**으로 간다
    - 입력 story에 timeline이 있으면 그대로 보존(out["timeline"]에 전달)
    """
    from pathlib import Path
    from typing import Any, Dict, List, Tuple
    import math
    import re

    # ---- utils ----
    try:
        from app.utils import load_json
    except Exception:
        from utils import load_json  # type: ignore

    # ---- UI 프리셋 로더(있으면 사용) + 기본값 ----
    def _ui_defaults() -> dict:
        return {
            "image_size": (832, 1472),
            "movie_fps": 24,
            "movie_overlap": 12,
            "min_scene_sec": 0.5,
            "round_sec": 3,
            "negative_bank": "손가락 왜곡, 눈 왜곡, 과도한 보정, 노이즈, 흐릿함, 텍스트 워터마크",
        }

    try:
        prefs_loader = globals().get("_load_ui_prefs_from_project")
        if prefs_loader is None:
            raise RuntimeError
        ui_prefs = prefs_loader(story.get("audio") or "")
        img_w, img_h = ui_prefs.get("image_size", (832, 1472))
        movie_fps = int(ui_prefs.get("movie_fps", 24))
        movie_overlap = int(ui_prefs.get("movie_overlap", 12))
        min_scene_sec = float(ui_prefs.get("min_scene_sec", 0.5))
        round_sec = int(ui_prefs.get("round_sec", 3))
        negative_bank = ui_prefs.get("negative_bank") or _ui_defaults()["negative_bank"]
    except Exception:
        ui_prefs = _ui_defaults()
        img_w, img_h = ui_prefs["image_size"]
        movie_fps = int(ui_prefs["movie_fps"])
        movie_overlap = int(ui_prefs["movie_overlap"])
        min_scene_sec = float(ui_prefs["min_scene_sec"])
        round_sec = int(ui_prefs["round_sec"])
        negative_bank = ui_prefs["negative_bank"]

    def _r(v: float) -> float:
        return round(float(v), round_sec)

    # ---- 원본 ----
    src = dict(story or {})
    audio_path = src.get("audio") or ""
    title = src.get("title") or "무제"
    total_sec = float(src.get("duration") or 0.0)
    fps_src = int(src.get("fps") or 0)
    lang = src.get("lang") or "ko"
    root = Path(audio_path).parent if audio_path else Path(".")

    # ---- project.json에서 가사/총길이 ----
    pj = root / "project.json"
    meta = load_json(pj, {}) if pj.exists() else {}
    lyrics_text_raw = (meta.get("lyrics") or "").strip()  # 전체 가사(원문)
    proj_time = float(meta.get("time") or 0.0)

    # ---- 전체 가사 정제(순수 가사) ----
    def _clean_full_lyrics(raw: str) -> str:
        t = (raw or "").replace("\r\n", "\n").replace("\r", "\n")
        # [verse]/[intro]/[anything] 태그 제거
        t = re.sub(r"\[[^\]]+\]", " ", t)
        # 개행 제거 → 공백으로 치환
        t = t.replace("\n", " ")
        # 연속 공백 축소
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    lyrics_text_clean = _clean_full_lyrics(lyrics_text_raw)

    # ───────────── 프롬프트 강제 규칙 ─────────────
    def _merge_global_bg_defaults(base_map: dict) -> dict:
        merged = dict(base_map)
        try:
            g = BG_DEFAULTS  # type: ignore[name-defined]
            if isinstance(g, dict):
                for k, v in g.items():
                    if isinstance(k, str) and isinstance(v, str):
                        merged[k.lower()] = v
        except NameError:
            try:
                g2 = bg_defaults  # type: ignore[name-defined]
                if isinstance(g2, dict):
                    for k, v in g2.items():
                        if isinstance(k, str) and isinstance(v, str):
                            merged[k.lower()] = v
            except Exception:
                pass
        return merged

    base_bg = {
        "intro": "황혼이 내려앉은 골목",
        "verse": "도시 야간 거리",
        "chorus": "네온이 번지는 광장",
        "bridge": "지하철 플랫폼",
        "outro": "비 내린 새벽 도로",
    }
    merged_bg = _merge_global_bg_defaults(base_bg)

    def _ensure_background(text: str, sec_name: str) -> str:
        txt = (text or "").strip()
        if "배경:" in txt:
            return txt
        key = (sec_name or "").lower().strip()
        bg_name = merged_bg.get(key, "도시 야간 거리")
        return f"배경: {bg_name}. {txt}".strip()

    def _ensure_effects_in_movie(text: str, effects: list[str]) -> str:
        txt = (text or "").strip()
        eff = [e.strip() for e in (effects or []) if e and e.strip()]
        if eff:
            if not txt.endswith("."):
                txt += "."
            txt += " " + ", ".join(eff)
        return txt.strip()

    def _ensure_motion_if_characters(text: str, has_chars: bool) -> str:
        txt = (text or "").strip()
        if has_chars and ("인물 동작:" not in txt):
            if not txt.endswith("."):
                txt += "."
            txt += " 인물 동작: 천천히 시선을 돌린다."
        return txt.strip()

    # ───────────── 타이밍 재배치 ─────────────
    scenes_in: List[Dict[str, Any]] = list(src.get("scenes") or [])
    spans: List[Tuple[float, float]] = []
    for sc in scenes_in:
        st = float(sc.get("start") or 0.0)
        ed = float(sc.get("end") or (st + float(sc.get("duration") or 0.0)))
        if ed <= st:
            ed = st + max(0.0, float(sc.get("duration") or 0.0))
        spans.append((st, ed))

    def _choose_total(orig_total: float, pj_time: float) -> float:
        # B안: project.json time(초)이 10초 이상이면 우선 사용
        return pj_time if pj_time >= 10.0 else orig_total

    def _retime(total_s: float, spans_in: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if total_s <= 0 or not spans_in:
            return spans_in
        durs = [max(0.0, ed - st) for st, ed in spans_in]
        sum_d = sum(durs)
        if sum_d <= 0:
            each = total_s / len(spans_in)
            t0 = 0.0
            out_spans: List[Tuple[float, float]] = []
            for _ in spans_in:
                out_spans.append((_r(t0), _r(min(total_s, t0 + each))))
                t0 += each
            return out_spans
        alloc = [(d / sum_d) * total_s for d in durs]
        alloc = [max(a, min_scene_sec) for a in alloc]
        sum_alloc = sum(alloc)
        if sum_alloc > total_s:
            scale = total_s / sum_alloc
            alloc = [a * scale for a in alloc]
        out_spans = []
        t0 = 0.0
        for a in alloc:
            s_val, e_val = t0, min(total_s, t0 + a)
            out_spans.append((_r(s_val), _r(e_val)))
            t0 = e_val
        if out_spans:
            out_spans[-1] = (out_spans[-1][0], _r(total_s))
        return out_spans

    max_end = max((e for (_, e) in spans), default=0.0)
    if (total_sec <= 3.0 or max_end <= 3.0):
        target_total = _choose_total(total_sec, proj_time)
        if target_total > 5.0 and max_end > 0.0:
            scale = target_total / max_end
            spans = [(st * scale, ed * scale) for (st, ed) in spans]
            total_sec = float(target_total)

    spans = _retime(total_sec, spans)

    # ───────────── characters: "id:index" + 객체형 ─────────────
    def _coerce_char_with_index(char_list: List[Any]) -> tuple[list[str], list[dict]]:
        """
        입력: ["female_01", "male_01"] 또는 ["female_01:1", ...]
        출력:
          - ["male_01:0","female_01:1"] 같은 문자열 리스트
          - [{"id":"male_01","index":0}, ...] 객체 리스트
        규칙:
          - index 미지정 시 자동 배정
          - 2인(남/여) 기본: 남성 0(오른쪽), 여성 1(왼쪽) — face swap 규칙(x 큰 쪽 0)
          - 1명: 0 / 3명+: 오른쪽→왼쪽 순 0,1,2...
        """
        parsed: list[tuple[str, int]] = []
        raw_ids: list[str] = []
        for item in (char_list or []):
            s_item = str(item or "").strip()
            if ":" in s_item:
                cid, idx_s = s_item.split(":", 1)
                try:
                    parsed.append((cid.strip(), int(idx_s.strip())))
                except Exception:
                    raw_ids.append(cid.strip())
            elif s_item:
                raw_ids.append(s_item)

        if parsed and not raw_ids:
            str_list = [f"{cid}:{idx}" for cid, idx in parsed]
            obj_list = [{"id": cid, "index": int(idx)} for cid, idx in parsed]
            return str_list, obj_list

        ids = list(raw_ids)
        if len(ids) == 2 and set(ids) == {"female_01", "male_01"}:
            auto = [("male_01", 0), ("female_01", 1)]
        else:
            auto = []
            for i, cid in enumerate(ids):
                auto.append((cid, i))

        str_list = [f"{cid}:{idx}" for cid, idx in auto]
        obj_list = [{"id": cid, "index": int(idx)} for cid, idx in auto]
        return str_list, obj_list

    # ───────────── 씬 재구성 ─────────────
    scenes_out: List[Dict[str, Any]] = []
    for i, sc in enumerate(scenes_in):
        sid = sc.get("id") or f"t_{i+1:03d}"
        section_name = str((sc.get("section") or sc.get("scene") or "scene")).lower().strip()
        s_val, e_val = spans[i] if i < len(spans) else (
            float(sc.get("start") or 0.0),
            float(sc.get("end") or 0.0),
        )
        d_val = max(0.0, e_val - s_val)

        base_prompt = (sc.get("prompt") or "").strip()
        prompt_img_in = (sc.get("prompt_img") or "").strip()
        prompt_movie_in = (sc.get("prompt_movie") or "").strip()
        effects_list = list(sc.get("effect") or [])

        # characters: "id:index" + 객체형
        char_str_list, char_obj_list = _coerce_char_with_index(list(sc.get("characters") or []))
        has_chars = bool(char_obj_list)

        # 프롬프트 보강
        prompt_img = prompt_img_in or base_prompt
        prompt_movie = prompt_movie_in or (prompt_img or base_prompt)
        prompt_img = _ensure_background(prompt_img, section_name)
        prompt_movie = _ensure_background(prompt_movie, section_name)
        prompt_movie = _ensure_effects_in_movie(prompt_movie, effects_list)
        prompt_movie = _ensure_motion_if_characters(prompt_movie, has_chars)

        prompt_negative = (sc.get("prompt_negative") or "@global")

        scn = {
            "id": sid,
            "section": section_name,
            "start": _r(s_val),
            "end": _r(e_val),
            "duration": _r(d_val),
            "scene": sc.get("scene") or section_name,
            "characters": char_str_list,
            "character_objs": char_obj_list,
            "effect": effects_list,
            "screen_transition": bool(sc.get("screen_transition")),
            "img_file": str(root / "imgs" / f"{sid}.png"),
            "clip_file": str(root / "clips" / f"{sid}.mp4"),
            "prompt_img": prompt_img,
            "prompt_movie": prompt_movie,
            "prompt_negative": prompt_negative,
            # "lyric": 는 아래에서 씬별 분배로 채움
        }
        scn.setdefault("prompt", scn["prompt_img"])  # 레거시 alias
        scenes_out.append(scn)

    # ───────────── 가사 섹션 생성 ─────────────
    lyrics_sections: List[Dict[str, Any]] = []
    if lyrics_text_raw:
        try:
            builder = globals().get("_build_lyrics_sections")
            if builder:
                lyrics_sections = builder(lyrics_text_raw, total_sec, scenes_out, round_sec=round_sec) or []
        except Exception:
            lyrics_sections = []

    # ───────────── 가사 씬별 분배 ─────────────
    def _split_block_lines(raw_text: str) -> List[str]:
        # 맨 윗줄이 [verse]/[chorus] 같은 태그면 제거
        t = (raw_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        lines = [ln.strip() for ln in t.split("\n")]
        if lines and re.fullmatch(r"\[[^\]]+\]", lines[0]):
            lines = lines[1:]
        # 빈 줄 제거
        return [ln for ln in lines if ln]

    # 경계 판정: [start, end) — 끝점 배제
    def _scene_in_block(scene_start: float, block_start: float, block_end: float) -> bool:
        return (block_start <= scene_start) and (scene_start < block_end)

    # 섹션ID 매핑(옵션)
    sec2id: Dict[str, str] = {}
    for idx, ls in enumerate(lyrics_sections, start=1):
        sec = str(ls.get("section") or "").lower().strip()
        if sec and sec not in sec2id:
            sec2id[sec] = f"L{idx:02d}"

    for sc in scenes_out:
        sec_id = sec2id.get(sc["section"])
        if sec_id:
            sc["section_id"] = sec_id

    # 블록 단위로 라인 → 씬에 분배
    for blk in lyrics_sections:
        try:
            b_st = float(blk.get("start", 0.0) or 0.0)
            b_ed = float(blk.get("end", 0.0) or 0.0)
        except Exception:
            continue
        lines = _split_block_lines(blk.get("text") or "")
        if not lines:
            lines = [(blk.get("text") or "").strip()] if blk.get("text") else []

        blk_scenes_idx = [i for i, sc in enumerate(scenes_out)
                          if _scene_in_block(float(sc.get("start") or 0.0), b_st, b_ed)]
        if not blk_scenes_idx:
            continue

        n_scenes = len(blk_scenes_idx)
        n_lines = len(lines)
        if n_lines <= 0:
            for i_sc in blk_scenes_idx:
                scenes_out[i_sc]["lyric"] = ""
            continue

        # 라인 분배: 씬 수에 맞춰 균등 청크로 나눔
        chunk = int(math.ceil(n_lines / n_scenes))
        for k, sc_idx in enumerate(blk_scenes_idx):
            a = k * chunk
            b = min((k + 1) * chunk, n_lines)
            piece = "\n".join(lines[a:b]).strip()
            scenes_out[sc_idx]["lyric"] = piece

    # 블록에 속하지 않아 lyric이 비어있는 씬은 안전하게 빈 문자열로 세팅
    for sc in scenes_out:
        if "lyric" not in sc:
            sc["lyric"] = ""

    # ───────────── 글로벌 컨텍스트 보강 ─────────────
    global_ctx = dict(src.get("global_context") or {})
    if not global_ctx.get("negative_bank"):
        global_ctx["negative_bank"] = negative_bank

    # ───────────── paths (동의어 키 제거: 표준 2개만 유지) ─────────────
    paths_obj = {
        "root": (str(root) if str(root).endswith(("\\", "/")) else str(root) + "\\"),
        "imgs_dir": "imgs",
        "clips_dir": "clips",
        "img_name_pattern": "{id}.png",
        "clip_name_pattern": "{id}.mp4",
    }

    # ───────────── 결과 구성 (title 바로 뒤에 lyrics 추가; '순수 가사') ─────────────
    out = {
        "version": "1.1",
        "audio": audio_path,
        "fps": int(fps_src or movie_fps),
        "duration": _r(total_sec),
        "offset": float(src.get("offset") or 0.0),
        "title": title,
        "lyrics": lyrics_text_clean,  # ★ 순수 가사(개행/태그 제거)
        "lang": lang or "ko",
        "paths": paths_obj,
        "characters": list(src.get("characters") or []),
        "character_styles": dict(src.get("character_styles") or {}),
        "global_context": global_ctx,
        "defaults": {
            "image": {"width": int(img_w), "height": int(img_h), "negative": "@global"},
            "movie": {"target_fps": int(movie_fps), "overlap_frames": int(movie_overlap), "negative": "@global"},
        },
        "lyrics_sections": lyrics_sections,
        "scenes": scenes_out,
        "audit": {"generated_by": "gpt-5 (no-gpt-prompts, rules only, per-scene-lyrics, clean-lyrics)"},
    }

    # 입력 story에 timeline 있었으면 보존
    if "timeline" in src:
        out["timeline"] = src["timeline"]

    return out










# 섹션/이펙트 → I2V 모션 힌트 매핑(필요시 자유 확장)
_I2V_MOTION_MAP = {
    "zoom-in-soft": "slow push-in, soft easing",
    "handheld-subtle": "subtle handheld micro-shake",
    "parallax": "layered parallax motion",
    "bloom": "lens bloom highlights, gentle light wrap",
    "flare": "anamorphic flare sweep",
    "soft-glow": "soft glow diffusion",
    "slow-dolly": "slow forward dolly",
    "rack-focus": "rack focus transition",
    "color-shift": "gradual color temperature shift",
    "light-streaks": "fast light streaks pass-by",
    "fade-out": "fade to black",
    "soft-vignette": "soft vignette emphasis",
    "bokeh": "shallow depth bokeh, slow breathing",
    "wide angle": "wide field with slight perspective drift",
    "pan-left-soft": "soft pan left",
    "zoom": "gentle zoom",
}
_TRANSITION_HINT = "screen transition between shots"

def _ensure_dict(d, default):
    return d if isinstance(d, dict) else dict(default)

def _safe_list(x):
    return list(x or []) if isinstance(x, (list, tuple)) else []


def _project_title(project_dir: Path) -> str:
    pj = project_dir / "project.json"
    meta = load_json(pj, {}) or {}
    t = str(meta.get("title") or project_dir.name).strip()
    return t or "untitled"

def _final_out_dir(title: str) -> Path:
    return Path(S.FINAL_OUT.replace("[title]", title))

def _story_path_candidates(project_dir: Path) -> List[Path]:
    """project_dir/stoty.json → FINAL_OUT/[title]/story.json 순서로 조회"""
    title = _project_title(project_dir)
    return [
        project_dir / "story.json",
        _final_out_dir(title) / "story.json",
    ]

def _read_story(project_dir: Path) -> Tuple[dict, Path]:
    story = {}
    spath = None
    for p in _story_path_candidates(project_dir):
        if p.exists():
            story = load_json(p, {}) or {}
            spath = p
            break
    if not story:
        raise FileNotFoundError("story.json을 찾지 못했습니다. (project_dir 또는 FINAL_OUT)")
    # legacy 'shots' → 'scenes' 호환은 UI 쪽에서 처리하지만 여기서도 방어
    if "shots" in story and "scenes" not in story:
        # 아주 단순 변환(필요 최소)
        scs = []
        for i, sh in enumerate(story.get("shots") or [], 1):
            scs.append({
                "id": sh.get("title") or f"t_{i:02d}",
                "section": sh.get("section", "verse"),
                "start": float(sh.get("start", 0.0)),
                "end": float(sh.get("end", 0.0)),
                "duration": float(sh.get("duration", 0.0)),
                "scene": sh.get("scene") or "",
                "characters": _safe_list(sh.get("characters") or ["female_01"]),
                "effect": _safe_list(sh.get("effect") or []),
                "screen_transition": bool(sh.get("screen_transition", False)),
                "img_file": sh.get("img_file") or "",
                "prompt": sh.get("prompt") or "",
            })
        story["scenes"] = scs
    if not story.get("scenes"):
        raise ValueError("story.json에 scenes가 없습니다.")
    return story, spath  # type: ignore



def _effect_to_motion(effect_list: List[str], screen_transition: bool) -> str:
    effs = [e for e in effect_list or []]
    parts = []
    for e in effs:
        m = _I2V_MOTION_MAP.get(str(e).lower())
        if m and m not in parts:
            parts.append(m)
    if screen_transition:
        parts.append(_TRANSITION_HINT)
    # 너무 비면 기본 카메라
    if not parts:
        parts = ["gentle camera movement for vertical video"]
    return ", ".join(parts)





# ---------- 메인 빌더들 ----------

def load_ui_prefs_for_audio(audio_path: str) -> dict:
    """
    audio 경로(…\[title]\vocal.mp3) 기준으로 project.json을 찾아 ui_prefs를 돌려준다.
    없으면 settings 기본값으로 채운다.
    """
    try:
        # noinspection PyPep8Naming
        from app import settings as S
    except ImportError:
        # noinspection PyPep8Naming
        import settings  # 1. 별칭 없이 모듈을 먼저 가져옵니다.
        S = settings  # 2. 가져온 모듈을 'S' 변수에 할당합니다.
    pj = Path(audio_path).parent / "project.json"
    meta = load_json(pj, {}) if pj.exists() else {}
    ui = meta.get("ui_prefs") or {}
    w, h = tuple(ui.get("image_size") or getattr(S, "DEFAULT_IMG_SIZE", (832, 1472)))
    fps = int(ui.get("movie_fps") or getattr(S, "DEFAULT_MOVIE_FPS", 24))
    overlap = int(ui.get("movie_overlap") or getattr(S, "DEFAULT_MOVIE_OVERLAP", 12))
    return {"image_w": int(w), "image_h": int(h), "movie_fps": int(fps), "overlap": int(overlap)}


# out["defaults"] = defaults  # ← 여러분의 story/json 작성 코드에 맞게 주입

# FAQ (주석으로 남깁니다)
#
# - DEFAULT_IMG_SIZE = (w, h)?  네, (가로, 세로)입니다. 드롭다운에서 고른 값이 ui_prefs.image_size로 저장되고,
#   이후 story/image/movie 생성 코드가 이 값을 읽어 사용합니다.
#
# - DEFAULT_MOVIE_FPS는 기본 24이고, 드롭다운으로 24/60 중 선택하도록 했습니다.
#
# - DEFAULT_MOVIE_OVERLAP는 i2v 청크 경계 프레임을 '겹치게' 만드는 값(프레임 수)입니다.
#   예: 60fps에서 12 → 0.2초 정도 앞뒤를 겹쳐서 경계 끊김을 줄입니다.
#
# - MIN_SCENE_SEC_DEFAULT = 0.2는 "0.2초 이하면 삭제"가 아니라
#   '보정 시 최소 길이로 승격'하는 기준입니다. (삭제 로직이 필요하면 알려주세요)
#
# - ROUND_SEC_DEFAULT = 3은 컷 시간값을 소수점 셋째 자리(1ms 단위)까지 반올림하는 기준입니다.

# FILE: tools/hotfix_image_movie_docs.py
# WHY THIS EXISTS (요약):
#   - 아래 전역 예시 코드는 "예시"일 뿐이며, 모듈 하단에 남아 있으면 안 됩니다.
#       prefs = load_ui_prefs_for_audio(story.get("audio", ""))
#       defaults = { ... }
#       # out["defaults"] = defaults
#   - import 시 story 변수가 없어서 NameError가 발생합니다.
#   - 이미 함수 내부에서 ui_prefs를 반영하므로, 전역 예시 블록은 삭제합니다.
#
# USAGE:
#   python tools/hotfix_image_movie_docs.py



TARGET = Path("app/image_movie_docs.py")

# 들여쓰기 제거 + 멀티라인/도트올 플래그 사용
_DEMO_BLOCK_RE = re.compile(
    dedent(r"""
        ^\s*prefs\s*=\s*load_ui_prefs_for_audio\([^)]*\)\s*
        .*?
        ^\s*#\s*out\["defaults"]\s*=\s*defaults.*?$
    """),
    re.MULTILINE | re.DOTALL,
)

# 여분의 ===== 같은 구분선 제거
_SEP_LINE_RE = re.compile(r"(?m)^\s*=+\s*$")

def verify_demo_block(path: str | Path = TARGET) -> dict:
    """전역 예시 블록 존재 여부 확인."""
    p = Path(path)
    if not p.exists():
        return {"file": str(p), "exists": False}
    src = p.read_text(encoding="utf-8")
    return {"file": str(p), "exists": bool(_DEMO_BLOCK_RE.search(src))}

def strip_demo_block(path: str | Path = TARGET) -> dict:
    """전역 예시 블록 삭제(.bak 백업 생성)."""
    p = Path(path)
    if not p.exists():
        return {"file": str(p), "changed": False, "backup": None}

    src = p.read_text(encoding="utf-8")
    new_src = _DEMO_BLOCK_RE.sub("", src)
    new_src = _SEP_LINE_RE.sub("", new_src)
    new_src = re.sub(r"\n{3,}", "\n\n", new_src)  # 공백 줄 정리

    if new_src == src:
        return {"file": str(p), "changed": False, "backup": None}

    bak = p.with_suffix(p.suffix + ".bak")
    bak.write_text(src, encoding="utf-8")
    p.write_text(new_src, encoding="utf-8")
    return {"file": str(p), "changed": True, "backup": str(bak)}

# FILE: app/image_movie_docs.py
# PLACE: 파일 하단(기존 normalize_to_v11 아래 아무 곳)

def validate_story_v11_dict(story: dict) -> list[str]:
    """필수 키만 빠르게 검증."""
    errs: list[str] = []
    if story.get("version") != "1.1":
        errs.append("version != '1.1'")
    for k in ("audio", "fps", "duration", "title", "paths", "defaults", "scenes"):
        if k not in story:
            errs.append(f"missing key: {k}")
    # paths/defaults 구조 체크(간단)
    paths = story.get("paths") or {}
    for k in ("root", "imgs_dir", "clips_dir", "img_name_pattern", "clip_name_pattern"):
        if k not in paths:
            errs.append(f"paths.{k} is missing")
    defaults = story.get("defaults") or {}
    if "image" not in defaults or "movie" not in defaults:
        errs.append("defaults.image/movie is missing")
    # scenes 최소 필수 키 체크
    for i, sc in enumerate(story.get("scenes") or []):
        for k in ("id", "section", "start", "end", "duration", "img_file", "clip_file"):
            if k not in sc:
                errs.append(f"scenes[{i}].{k} is missing")
        # prompt 필드 세트 확인
        for k in ("prompt_img", "prompt_movie", "prompt_negative"):
            if k not in sc:
                errs.append(f"scenes[{i}].{k} is missing")
    return errs


def validate_story_v11_file(path: str | Path) -> list[str]:
    """파일을 읽어 validate_story_v11_dict에 넘긴다."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return [f"JSON 로드 실패: {e}"]
    return validate_story_v11_dict(data)


# === 강제 규칙 기반 GPT 프롬프트 생성기 & 적용기 ==============================


def gpt_scene_prompt_writer(
    story: dict,
    gpt_fn: Optional[Callable[[str], dict]] = None,
    gpt_only: bool = True,
) -> Dict[str, Dict[str, str]]:
    """
    - prompt_img: 반드시 '배경' 포함
    - prompt_movie: 반드시 '배경 + 카메라 + (캐릭터 있으면 동작) + 효과' 포함
    - gpt_only=True이면 필수 항목 누락 시 예외
    """
    if gpt_only and not callable(gpt_fn):
        raise RuntimeError("gpt_only=True인데 gpt_fn이 없습니다. gpt_fn을 전달하세요.")

    prompts: Dict[str, Dict[str, str]] = {}
    gc = story.get("global_context") or {}
    section_moods = (gc.get("section_moods") or {})
    style_guide = (gc.get("style_guide") or "")
    global_palette = (gc.get("palette") or "")

    last_img = None
    last_mov = None

    def _S(x): return x.strip() if isinstance(x, str) else ""

    for idx, sc in enumerate(story.get("scenes") or []):
        sid = sc.get("id") or f"t_{idx+1:03d}"
        section = (sc.get("section") or "").lower().strip()
        eff_list = sc.get("effect") or []
        chars = sc.get("characters") or []
        lyric_hint = _S(sc.get("lyric_text") or "")

        sys = (
            "너는 뮤직비디오 씬 연출 프롬프트 생성기다. "
            "절대 가사 원문을 인용하지 말고, 연출/카메라/배경/조명/효과/동작을 구조화해라. JSON만 출력."
        )
        usr_payload = {
            "section": section,
            "section_mood": section_moods.get(section, ""),
            "style_guide": style_guide,
            "global_palette": global_palette,
            "lyric_concept": lyric_hint,
            "characters": chars,
            "effects_given": eff_list,
            "need": {
                "prompt_img":   "배경(필수) + 정지 이미지용 묘사(100~180자)",
                "prompt_movie": "배경(필수) + 카메라(필수) + (캐릭터 있으면 동작) + 효과(필수), 100~180자",
            },
            "return_schema": {
                "background": "짧고 명확(예: '비에 젖은 도심 보도')",
                "lighting": "간단 조명",
                "camera": "예: '슬로우 푸시인'",
                "effects_extra": "효과 추가(쉼표로, 선택)",
                "character_motion": "캐릭터 있으면 동작 1개(선택)",
                "img_phrase": "정지 이미지용 핵심 묘사",
                "movie_phrase": "영상용 핵심 묘사(카메라/리듬/공간감 우선)",
            }
        }
        prompt_text = f"[SYSTEM]\n{sys}\n[USER]\n{json.dumps(usr_payload, ensure_ascii=False)}"

        res = {}
        if callable(gpt_fn):
            try:
                res = gpt_fn(prompt_text) or {}
                if isinstance(res, str):
                    res = json.loads(res)
            except Exception:
                if gpt_only:
                    raise
                res = {}

        bg   = _S(res.get("background"))
        lit  = _S(res.get("lighting"))
        cam  = _S(res.get("camera"))
        imgp = _S(res.get("img_phrase") or res.get("prompt_img"))
        movp = _S(res.get("movie_phrase") or res.get("prompt_movie"))
        fx_x = _S(res.get("effects_extra"))
        mot  = _S(res.get("character_motion"))

        # (1) prompt_img: 배경 필수
        if not bg:
            if gpt_only: raise ValueError(f"[{sid}] background 누락")
            bg = "도시 야간 거리"
        img = f"배경: {bg}."
        if lit: img += f" 조명: {lit}."
        img += f" {imgp or '인물 정면, 일관된 헤어/의상, 고품질.'}".strip()

        # (2) prompt_movie: 배경 + 카메라 + 효과 + (캐릭터 있으면 동작)
        if not cam:
            if gpt_only: raise ValueError(f"[{sid}] camera 누락")
            cam = "슬로우 푸시인"
        mov = f"배경: {bg}. 카메라: {cam}."
        if movp: mov += f" {movp}"
        eff_all = [e for e in (eff_list or []) if e] + ([fx_x] if fx_x else [])
        eff_all = [e.strip() for e in eff_all if e and e.strip()]
        if eff_all:
            mov = mov.rstrip(".") + ", " + ", ".join(eff_all)
        if chars:
            mov = mov.rstrip(".") + f". 인물 동작: {mot or '천천히 걸어간다'}."

        if last_img == img.strip():
            img += " 구도 변주: 하프바디."
        if last_mov == mov.strip():
            mov += " 리듬 변주: 템포 살짝 느리게."
        last_img, last_mov = img.strip(), mov.strip()

        prompts[sid] = {"prompt_img": last_img, "prompt_movie": last_mov}
    return prompts


#================페이스 스왑 관련=============#


def parse_character_spec(item: Any) -> Dict[str, Any]:
    """
    'characters' 항목의 원소를 정규화한다.
    허용 입력:
      - "female_01"            → {"id":"female_01", "index": None}
      - "female_01:1"          → {"id":"female_01", "index": 1}
      - {"id":"female_01"}     → {"id":"female_01", "index": None}
      - {"id":"female_01","index":1,"pos":"left","desc":"..."} → 그대로 보존
    반환: {"id": str, "index": Optional[int], ...부가필드 유지}
    """
    if isinstance(item, dict):
        out = dict(item)
        cid = str(out.get("id") or "").strip()
        out["id"] = cid
        if "index" in out and out["index"] is not None:
            try:
                out["index"] = int(out["index"])
            except Exception:
                out["index"] = None
        else:
            out["index"] = None
        return out

    if isinstance(item, str):
        txt = item.strip()
        if ":" in txt:
            cid, _, idx = txt.partition(":")
            cid = cid.strip()
            try:
                idx_val = int(idx.strip())
            except Exception:
                idx_val = None
            return {"id": cid, "index": idx_val}
        return {"id": txt, "index": None}

    # 알 수 없는 형식은 무시하지 말고 안전 디폴트
    return {"id": str(item), "index": None}


def normalize_scene_characters(scene: Dict[str, Any]) -> Dict[str, Any]:
    """
    scene["characters"]를 표준화:
      - 리스트 각 원소를 parse_character_spec()으로 dict화
      - scene["layout"]["face_indices"] (있다면)로 index를 보완
      - scene["face_indices"] (flat dict)도 함께 만들어 소비 측이 편하게 사용
    반환: 수정된 scene (원본 shallow copy 후 필드 갱신)
    """
    sc = dict(scene or {})
    chars_in = sc.get("characters") or []
    norm: List[Dict[str, Any]] = [parse_character_spec(x) for x in chars_in]

    # layout.face_indices에 매핑이 있으면 index 보완
    layout = sc.get("layout") or {}
    fi_map = layout.get("face_indices") or {}
    if isinstance(fi_map, dict):
        for c in norm:
            cid = c.get("id")
            if cid in fi_map:
                try:
                    c["index"] = int(fi_map[cid])
                except Exception:
                    pass

    # flat map 생성 (소비 편의)
    face_indices: Dict[str, int] = {}
    for c in norm:
        cid = c.get("id")
        idx = c.get("index")
        if isinstance(cid, str) and idx is not None:
            face_indices[cid] = int(idx)

    sc["characters"] = norm
    if "layout" not in sc:
        sc["layout"] = {}
    sc["layout"]["face_indices"] = dict(face_indices)
    sc["face_indices"] = dict(face_indices)  # 사용처가 layout 안을 안 보는 경우 대비
    return sc


def sort_faces_right_to_left(face_boxes: List[Tuple[float, float, float, float]]) -> List[int]:
    """
    탐지된 얼굴 박스들을 '오른쪽→왼쪽' 순서로 정렬한 뒤,
    '탐지 인덱스' 리스트를 반환한다.
    - face_boxes: [(xmin, ymin, xmax, ymax), ...]  (탐지 순서 기준)
    반환 예: [2, 0, 1]  → 오른쪽이 탐지#2, 그다음 탐지#0, 그다음 탐지#1
    """
    if not face_boxes:
        return []
    centers = []
    for i, b in enumerate(face_boxes):
        try:
            xmin, ymin, xmax, ymax = map(float, b)
            cx = (xmin + xmax) * 0.5
        except Exception:
            cx = float(i)
        centers.append((i, cx))
    centers.sort(key=lambda t: t[1], reverse=True)  # x 큰 순 (오른쪽 먼저)
    return [i for i, _ in centers]





def _cleanup_punctuation(text: str) -> str:
    """불필요한 공백/중복 점('. .') 제거, 문장 끝 공백 정리."""
    t = (text or "").strip()
    # ". ." → "."
    t = re.sub(r"\s*\.\s*\.", ".", t)
    # "  " → " "
    t = re.sub(r"\s{2,}", " ", t)
    # " ." → "."
    t = re.sub(r"\s+\.", ".", t)
    return t.strip()

def _ensure_background(text: str, sec: str) -> str:
    """
    prompt_img / prompt_movie에 '배경:'이 반드시 들어가도록 보장.
    - 전역 BG_DEFAULTS/bg_defaults 존재 시 합쳐 사용(없으면 내부 기본값).
    - 내부 변수는 소문자.
    """
    txt = (text or "").strip()
    if "배경:" in txt:
        return _cleanup_punctuation(txt)

    base_map = {
        "intro":  "황혼이 내려앉은 골목",
        "verse":  "도시 야간 거리",
        "chorus": "네온이 번지는 광장",
        "bridge": "지하철 플랫폼",
        "outro":  "비 내린 새벽 도로",
    }

    global_defaults: Dict[str, str] = {}
    try:
        # 전역이 있으면 덮어쓰기
        global_defaults = BG_DEFAULTS  # type: ignore[name-defined]
    except NameError:
        try:
            global_defaults = bg_defaults  # type: ignore[name-defined]
        except Exception:
            global_defaults = {}

    if isinstance(global_defaults, dict):
        for k, v in global_defaults.items():
            if isinstance(k, str) and isinstance(v, str):
                base_map[k.lower()] = v

    key = (sec or "").lower().strip()
    bg_name = base_map.get(key, "도시 야간 거리")
    if txt and not txt.endswith("."):
        txt += "."
    txt = f"배경: {bg_name}. {txt}".strip()
    return _cleanup_punctuation(txt)

def _ensure_effects_in_movie(text: str, effects: List[str]) -> str:
    """
    movie 프롬프트에 효과들을 '한 번만' 추가.
    - 이미 포함된 효과 문자열은 중복 추가하지 않음(부분 문자열 매칭).
    - 마지막 마침표/공백 보정.
    """
    txt = (text or "").strip()
    eff = [e.strip() for e in (effects or []) if e and isinstance(e, str) and e.strip()]
    to_add: List[str] = []
    for e in eff:
        if e and e not in txt:
            to_add.append(e)
    if to_add:
        if txt and not txt.endswith("."):
            txt += "."
        txt += " " + ", ".join(to_add)
    return _cleanup_punctuation(txt)

def _ensure_motion_if_characters(text: str, has_chars: bool, debug_print: bool=False) -> str:
    """
    인물이 있으면 movie 프롬프트에 '인물 동작: ...'을 정확히 한 번만 포함.
    - 기존 '인물 동작:' 문구가 있으면 그대로 유지(중복 추가 금지).
    - 인물이 없으면 기존 '인물 동작:' 문구를 제거.
    - 대소문자/공백/마침표 변형까지 폭넓게 감지.
    """
    before = (text or "").strip()
    txt = _cleanup_punctuation(before)

    # '인물 동작:' 탐지 (마침표 이전의 내용을 한 문장으로 캡쳐)
    motion_pat = re.compile(r"(?:^|\s)인물\s*동작\s*:\s*([^\.]+)\.", re.IGNORECASE)

    found = bool(motion_pat.search(txt))
    if debug_print:
        print(f"[PROMPT][MOTION] has_chars={has_chars} | found_existing={found} | before='{before}'")

    if has_chars:
        if found:
            # 이미 있으면 아무것도 추가하지 않음. (중복 제거만 보장)
            # 중복 '인물 동작:'이 여러 번인 케이스 정리: 첫 번째만 남기고 제거
            all_motions = motion_pat.findall(txt)
            if len(all_motions) > 1:
                first = all_motions[0].strip()
                # 전체에서 전부 제거 후 한 번만 추가
                txt = motion_pat.sub("", txt)
                if txt and not txt.endswith("."):
                    txt += "."
                txt += f" 인물 동작: {first}."
            # 끝 정리
            txt = _cleanup_punctuation(txt)
            after = txt
            if debug_print:
                print(f"[PROMPT][MOTION] keep-existing | after='{after}'")
            return after
        else:
            # 없으면 기본 문구를 한 번만 추가
            if txt and not txt.endswith("."):
                txt += "."
            txt += " 인물 동작: 천천히 시선을 돌린다."
            after = _cleanup_punctuation(txt)
            if debug_print:
                print(f"[PROMPT][MOTION] add-default | after='{after}'")
            return after
    else:
        # 인물 없으면 기존 '인물 동작:' 제거
        txt = motion_pat.sub("", txt)
        after = _cleanup_punctuation(txt)
        if debug_print:
            print(f"[PROMPT][MOTION] remove-because-no-characters | after='{after}'")
        return after

def _ensure_face_front_in_img(text: str, has_chars: bool) -> str:
    """
    이미지 프롬프트에 인물이 있다면 반드시 '얼굴 정면' 포함(한 번만).
    movie에는 넣지 않음(A안 규칙).
    """
    txt = (text or "").strip()
    if not has_chars:
        return _cleanup_punctuation(txt)

    # 이미 있으면 중복 추가 금지(유사 표현 포함)
    if ("얼굴 정면" in txt) or ("정면 얼굴" in txt):
        return _cleanup_punctuation(txt)

    if txt and not txt.endswith("."):
        txt += "."
    txt += " 인물은 얼굴 정면."
    return _cleanup_punctuation(txt)



JsonPath = Union[str, Path]
def apply_intro_outro_to_story_file(
    src: JsonPath,
    dest: JsonPath,
    *,
    intro_ratio: float = 0.10,
    outro_ratio: float = 0.10,
) -> str:
    """
    src의 story.json을 읽어 timeline을 인/아웃트로 비율로 갱신하고 dest에 저장.
    반환값: dest 경로(str)

    요구 조건:
    - intro_ratio, outro_ratio는 0.0 이상 0.5 미만 권장
    - _read_total_seconds, _build_base_timeline_from_scenes, _with_intro_outro_ratio 는
      본 모듈 내 기존 구현을 사용
    """
    # 경계 검사 (과도한 예외 방지: 명확한 ValueError만 발생)
    if not (0.0 <= intro_ratio < 0.5):
        raise ValueError("intro_ratio must be >= 0.0 and < 0.5")
    if not (0.0 <= outro_ratio < 0.5):
        raise ValueError("outro_ratio must be >= 0.0 and < 0.5")

    try:
        # app 패키지 실행/단일 파일 실행 모두 대응
        from app.utils import load_json, save_json  # type: ignore
    except ImportError:
        from utils import load_json, save_json  # type: ignore

    src_path = Path(src)
    dest_path = Path(dest)

    story = load_json(src_path, {}) or {}
    scenes = list(story.get("scenes") or [])

    # 총 길이는 프로젝트 디렉토리 기준으로 추정
    project_dir = src_path.parent
    total_sec = float(_read_total_seconds(str(project_dir)))

    base = _build_base_timeline_from_scenes(scenes, total_sec)
    timeline = _with_intro_outro_ratio(
        base,
        total_sec,
        intro_ratio=intro_ratio,
        outro_ratio=outro_ratio,
    )

    story["timeline"] = timeline
    save_json(dest_path, story)
    return str(dest_path)





# ======================= /A안 GPT 적용기 끝 =======================

if __name__ == "__main__":
    info = verify_demo_block()
    print("BEFORE:", info)
    if info["exists"]:
        print(strip_demo_block())
    else:
        print("No demo block found.")
