# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Callable, Any, Optional, Tuple
import os
import re
import json
from copy import deepcopy

TraceFn = Callable[[str, str], None]

def _t(trace: TraceFn|None, tag: str, msg: str) -> None:
    if trace:
        try:
            trace(tag, msg)
        except Exception:
            pass

# --- 추가 헬퍼 ---.

def _equal_chunks_by_words(text: str, n: int) -> List[str]:
    text = re.sub(r'\s+', ' ', (text or '').strip())
    if n <= 0:
        return []
    if not text:
        return [''] * n

    tokens = [t for t in text.split(' ') if t]  # 공백 기준 단어 토큰
    if not tokens:
        return [''] * n

    step = max(1, len(tokens) // n)
    out: List[str] = []
    i = 0
    for k in range(n - 1):
        j = min(len(tokens), i + step)
        out.append(' '.join(tokens[i:j]).strip())
        i = j
    out.append(' '.join(tokens[i:]).strip())
    return out


def _merge_global_context(S: dict, g: dict) -> None:
    """AI가 준 전역 컨텍스트를 story에 병합한다."""
    if not isinstance(g, dict):
        g = {}

    # 기본값 준비
    S.setdefault("global_context", {})
    GC = S["global_context"]

    # 1) 요약
    if g.get("global_summary"):
        GC["global_summary"] = str(g["global_summary"]).strip()

    # 2) themes
    th = g.get("themes") or []
    if isinstance(th, str):
        th = [x.strip() for x in re.split(r"[,|/]+", th) if x.strip()]
    GC["themes"] = th

    # 3) palette / style_guide / negative_bank / section_moods / effect
    if g.get("palette"):
        GC["palette"] = str(g["palette"]).strip()

    # style_guide: 고정 문구 + AI 추가
    base_sg = "실사, 인물 정면, 일관된 헤어/의상/분위기, 자연스러운 조명"
    extra_sg = str(g.get("style_guide") or "").strip()
    GC["style_guide"] = base_sg if not extra_sg else f"{base_sg}, {extra_sg}"

    base_neg = "손가락 왜곡, 눈 왜곡, 과도한 보정, 노이즈, 흐릿함, 텍스트 워터마크"
    extra_neg = str(g.get("negative_bank") or "").strip()
    GC["negative_bank"] = base_neg if not extra_neg else f"{base_neg}, {extra_neg}"

    # section_moods: 기본 문구 + AI 추가
    def _sm(name: str, base: str) -> str:
        add = str((g.get("section_moods") or {}).get(name, "")).strip()
        return base if not add else f"{base}/{add}"

    GC["section_moods"] = {
        "intro":  _sm("intro",  "도입/암시/미니멀"),
        "verse":  _sm("verse",  "잔잔함/근접/친밀감"),
        "chorus": _sm("chorus", "개방감/광각/확장"),
        "bridge": _sm("bridge", "전환감/대비/변화"),
        "outro":  _sm("outro",  "여운/잔상/감쇠"),
    }

    # effect(전역)
    eff = g.get("effect") or []
    if isinstance(eff, str):
        eff = [x.strip() for x in re.split(r"[,|/]+", eff) if x.strip()]
    GC["effect"] = eff

    # 4) defaults.image (렌더 W/H 반영 + negative="@global")
    S.setdefault("defaults", {})
    S["defaults"].setdefault("image", {})
    di = S["defaults"]["image"]
    # 기존 값 우선, 없으면 g의 width/height 사용
    W = int(di.get("width") or g.get("image_width") or 832)
    H = int(di.get("height") or g.get("image_height") or 1472)
    di["width"] = W
    di["height"] = H
    di["negative"] = "@global"

def _fallback_scene_bg(section: str, themes: list[str]) -> str:
    """AI가 씬 배경을 안 줄 때 간단한 한글 배경 문장 생성."""
    sec = (section or "").lower()
    ths = "·".join(themes or [])
    if sec == "intro":
        base = "황혼의 고요한 골목"
    elif sec == "chorus":
        base = "네온이 번지는 광장"
    elif sec == "bridge":
        base = "밝고 어두움이 교차하는 전환 공간"
    elif sec == "outro":
        base = "불빛이 가라앉는 야간 거리"
    else:
        base = "도시의 야간 거리"
    return f"배경: {base}. 무드 키워드: {ths}"

def _build_korean_prompts2(scene: dict, styles: Dict[str, str], base_bg: str) -> tuple[str, str]:
    """
    base_bg(씬 배경 요약)를 중심으로 이미지/무빙 프롬프트 생성.
    - 캐릭터 스타일 문장 주입
    - 효과/전환/모션 규칙 반영
    """
    base_bg = (base_bg or "").strip()
    eff = ', '.join(scene.get('effect') or [])
    if eff:
        eff = f' | 효과: {eff}'

    # 캐릭터 라인
    chars = []
    for ch in scene.get('characters') or []:
        cid = ch.split(':', 1)[0] if isinstance(ch, str) else ch.get('id') or str(ch)
        desc = styles.get(cid, cid)
        chars.append(desc)
    char_line = f" | 캐릭터: {', '.join(chars)}" if chars else ''

    # 전환
    trans = ' | 화면 전환: 부드러운 페이드' if scene.get('screen_transition') else ''

    # 영화쪽은 '모션'을 기본 문장으로 넣어 준다(캐릭 있으면 동작 지시어)
    has_char = bool(chars)
    motion = " | 카메라: 슬로우 푸시인"  # 기본
    if (scene.get('section') or '').lower() == 'chorus':
        motion = " | 카메라: 와이드/트래킹으로 확장감"
    elif (scene.get('section') or '').lower() == 'intro':
        motion = " | 카메라: 느린 패닝으로 도입의 여백"

    if has_char:
        motion += " | 인물 동작: 호흡과 시선 변화를 섬세하게"

    prompt_img = f"{base_bg}{char_line}{eff}".strip()
    prompt_movie = f"{base_bg}{char_line}{eff}{trans}{motion}".strip()
    return prompt_img, prompt_movie


def _equal_chunks_by_chars(text: str, n: int) -> List[str]:
    text = re.sub(r'\s+', ' ', (text or '').strip())
    if n <= 0:
        return []
    if not text:
        return [''] * n
    L = len(text)
    step = max(1, L // n)
    out = []
    i = 0
    for k in range(n - 1):
        j = min(L, i + step)
        # 경계 보정: 공백/구두점에서 끊기 시도
        m = re.search(r'[\s,.!?…]+', text[j:j+12])
        if m:
            j = j + m.start() + 1
        out.append(text[i:j].strip())
        i = j
    out.append(text[i:].strip())
    return out

def _segment_lyrics_for_scenes(record: dict, audio_info: dict=None, ai=None, lang='ko'):
    """
    record: story.json dict
    audio_info = {"duration": float, "onsets": [초, ...]}  # onsets는 있어도 되고 없어도 됨
    결과:
      - record["lyrics_sections"] = [{id, start, end, text}...]
      - record["scenes"][i]["lyric"] = 해당 구간과 겹치는 첫 가사 또는 ""
    """
    rec = deepcopy(record)
    # 1) 한 줄 가사 확보
    lyrics_text = " ".join((rec.get("lyrics") or "").split())

    # 2) 의미 단위(하드코딩 문구 금지, AI 호출)
    units = split_lyrics_into_semantic_units(lyrics_text, ai=ai)
    if not units:
        rec["lyrics_sections"] = []
        return rec
    print(f"[debug] split_lyrics_into_semantic_units -> {units}")
    # 3) 시간 범위
    total_offset = float(rec.get("offset", 0.0))
    total_duration = float(rec.get("duration", 0.0))
    total_start = total_offset
    total_end = total_offset + total_duration

    # 4) 오디오 정보(있으면 교정)
    if audio_info:
        dur = float(audio_info.get("duration") or 0)
        if dur > 0 and abs(dur - total_duration) > 0.25:
            total_end = total_start + dur
            rec["duration"] = round(dur, 3)

    # 5) 가중치(한글 음절 수)로 시간 분배 (온셋 있으면 온셋 우선)
    def _syllable_w(t):
        ks = re.findall(r"[가-힣]", t or "")
        return len(ks) if ks else max(1, len((t or "").replace(" ", "")))

    weights = [_syllable_w(t) for t in units]
    s = sum(weights)
    if s <= 0: weights = [1]*len(units); s = len(units)
    ratios = [w/s for w in weights]

    # 온셋 보정(선택): audio_info.get("onsets")
    boundaries = [total_start]
    acc = total_start
    for r in ratios:
        acc += (total_end-total_start)*r
        boundaries.append(acc)
    boundaries[-1] = total_end
    # (온셋 기반 스냅은 필요하면 여기서 보정 가능)

    # 6) lyrics_sections 생성
    new_lyrics = []
    for i, text in enumerate(units):
        s0, s1 = boundaries[i], boundaries[i+1]
        new_lyrics.append({
            "id": f"L{i+1:02d}",
            "start": round(s0, 3),
            "end": round(s1, 3),
            "text": text
        })
    rec["lyrics_sections"] = new_lyrics

    # 7) 씬에 1:1 매핑(겹치는 첫 가사 텍스트를 lyric에 기입)
    scenes = rec.get("scenes", [])
    j = 0
    for sc in scenes:
        s0, s1 = float(sc.get("start", 0)), float(sc.get("end", 0))
        chosen = ""
        while j < len(new_lyrics) and new_lyrics[j]["end"] <= s0:
            j += 1
        if j < len(new_lyrics):
            l0, l1 = new_lyrics[j]["start"], new_lyrics[j]["end"]
            if not (l1 <= s0 or l0 >= s1):
                chosen = new_lyrics[j]["text"]
        sc["lyric"] = chosen
    rec["scenes"] = scenes
    return rec


def _enforce_character_style_rules(styles: Dict[str, str]) -> Dict[str, str]:
    """여성: 'huge breasts, slim legs' 강제 포함, 성별 한국어 명시."""
    out: Dict[str, str] = {}
    for cid, txt in (styles or {}).items():
        s = (txt or '').strip()
        if not s:
            s = cid
        # 성별 판정
        if re.search(r'female|여성', cid, re.I) or re.search(r'\bF\b', cid):
            if '여성' not in s:
                s = '여성, ' + s
            # 필수 문구 강제
            if 'huge breasts' not in s:
                s += ', huge breasts'
            if 'slim legs' not in s:
                s += ', slim legs'
        elif re.search(r'male|남성', cid, re.I) or re.search(r'\bM\b', cid):
            if '남성' not in s:
                s = '남성, ' + s
        out[cid] = s
    return out

def _inject_styles_into_prompt(text: str, styles: Dict[str, str]) -> str:
    t = text or ''
    for cid, desc in styles.items():
        # <female_01>, <female_01:0> 등 모두 치환
        t = re.sub(rf'<{re.escape(cid)}(?::\d+)?>', desc, t)
    return t

def _build_korean_prompts(scene: dict, styles: Dict[str, str]) -> tuple[str, str]:
    """씬의 기존 한글 'prompt' 필드와 효과/전환 정보를 묶어 img/movie 문장 생성."""
    base = (scene.get('prompt') or '').strip()
    eff = ', '.join(scene.get('effect') or [])
    if eff:
        eff = f' | 효과: {eff}'
    chars = []
    for ch in scene.get('characters') or []:
        cid = ch.split(':', 1)[0] if isinstance(ch, str) else ch.get('id') or str(ch)
        desc = styles.get(cid, cid)
        chars.append(desc)
    char_line = f" | 캐릭터: {', '.join(chars)}" if chars else ''
    trans = ''
    if scene.get('screen_transition') is True:
        trans = ' | 화면 전환: 부드러운 페이드'
    prompt_img = f"{base}{char_line}{eff}".strip()
    prompt_movie = f"{base}{char_line}{eff}{trans}".strip()
    return prompt_img, prompt_movie

def apply_gpt_to_story_v11(
    story: dict,
    *,
    ask: Callable[..., str],
    prefer: str | None = None,
    allow_fallback: bool | None = None,
    trace: TraceFn | None = None,
    temperature: float | None = None,   # 넘어오면 무시
    **kwargs,                            # 여분 키워드도 무시
) -> dict:
    if temperature is not None:
        _t(trace, "warn", f"ignored kw: temperature={temperature}")
    if kwargs:
        _t(trace, "warn", f"ignored extra kwargs: {list(kwargs.keys())}")

    import re, json
    from typing import List

    # ──────────────────────────────────────────────────────────────
    # 내부 유틸(이 함수 안에서만 사용)
    # ──────────────────────────────────────────────────────────────
    FORBIDDEN_PHRASES = [
        "황혼이 내려앉은 골목",
        "리듬 포인트 중심",
    ]
    # ID 패턴(female_01, male_02, female_01:0 등)
    RE_ID = re.compile(r"(?:fe)?male_\d+(?::\d+)?", re.IGNORECASE)
    # 선정성/문자삽입 등 프롬프트에서 제외하고 싶은 토큰 (스타일에는 있을 수 있으니 프롬프트만 정리)
    FORBIDDEN_TOKENS = [
        "huge breasts", "slim legs", "노출", "선정적",
        "text", "letters", "typography", "워터마크", "캡션", "자막", "closed captions",
    ]

    # 씬 다양화용 회전 토큰
    CAMERA_VARIATIONS = [
        "정면", "반정면", "아이레벨", "로우앵글", "하이앵글",
        "클로즈업", "미디엄 샷", "와이드 샷",
    ]
    LIGHT_VARIATIONS = [
        "소프트 라이트", "시네마틱 라이팅", "네온 리플렉션", "림 라이트", "워밍 톤", "쿨 톤",
    ]
    MOVE_VARIATIONS = [
        "slow push-in", "gentle camera pan", "tracking shot", "dolly in", "rack-focus",
    ]

    EFFECT_FALLBACKS = [
        "soft light", "bokeh", "film grain", "soft focus", "warm rim light", "color pop"
    ]

    def _clean_text(s: str) -> str:
        s = s.replace("\u200b", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _strip_forbidden_phrases(s: str) -> str:
        if not s:
            return s
        out = s
        for p in FORBIDDEN_PHRASES:
            out = out.replace(p, "")
        return _clean_text(out)

    def _remove_ids_to_common_noun(s: str) -> str:
        # female_01 → 여성 / male_02 → 남성 (콜론 인덱스 포함 케이스도 포함)
        def repl(m: re.Match) -> str:
            token = m.group(0).lower()
            return "여성" if token.startswith("fe") else ("남성" if token.startswith("male") else "")
        return RE_ID.sub(repl, s)

    def _remove_forbidden_tokens(s: str) -> str:
        out = s
        for t in FORBIDDEN_TOKENS:
            out = re.sub(re.escape(t), "", out, flags=re.IGNORECASE)
        return _clean_text(out)

    def _sanitize_prompt_text(s: str) -> str:
        # 순서: ID 제거 → 금칙어 토큰 삭제 → 고정문구 제거 → 공백 정리
        return _clean_text(_strip_forbidden_phrases(_remove_forbidden_tokens(_remove_ids_to_common_noun(s))))

    def _ensure_korean_labeling(s: str) -> str:
        # 실사용에서는 언어 감지/번역을 붙일 수 있지만, 여기서는 한글 토큰 강화로 대체(형식 보정)
        # 영문만 남는 극단 케이스 방지용. 필요하면 더 강하게 필터링 가능.
        return s

    def _ensure_effects(effs: List[str]) -> List[str]:
        # 공백/중복 제거 후 2~4개로 정규화. 부족하면 폴백에서 보충.
        uniq = []
        for x in (effs or []):
            x2 = _clean_text(str(x))
            if x2 and (x2 not in uniq):
                uniq.append(x2)
        i = 0
        while len(uniq) < 2 and i < len(EFFECT_FALLBACKS):
            if EFFECT_FALLBACKS[i] not in uniq:
                uniq.append(EFFECT_FALLBACKS[i])
            i += 1
        if len(uniq) > 4:
            uniq = uniq[:4]
        return uniq

    def _diversify_across_scenes(idx: int, base: str) -> str:
        """
        연속 씬의 프롬프트가 비슷하게 보일 때 카메라/조명/무빙 토큰을 살짝 섞어서
        반복 인상을 줄인다. (문장 의미 보존, 한국어 유지)
        """
        cam = CAMERA_VARIATIONS[idx % len(CAMERA_VARIATIONS)]
        lit = LIGHT_VARIATIONS[idx % len(LIGHT_VARIATIONS)]
        mov = MOVE_VARIATIONS[idx % len(MOVE_VARIATIONS)]
        # 이미 같은 단어가 있으면 중복 방지
        tag_bits = []
        if cam not in base:
            tag_bits.append(cam)
        if lit not in base:
            tag_bits.append(lit)
        if mov not in base:
            tag_bits.append(mov)
        if tag_bits:
            return _clean_text(f"{base}, {', '.join(tag_bits)}")
        return base

    def _must_fill_img_mov(sc: dict, styles: dict, bg: str, cur_img: str, cur_mov: str) -> tuple[str, str]:
        """prompt_img / prompt_movie 비었거나 너무 짧으면 보강."""
        p_img, p_mov = cur_img, cur_mov
        if not p_img or len(p_img) < 8 or not re.search(r"[가-힣]", p_img):
            _img2, _mov2 = _build_korean_prompts2(sc, styles, bg)
            p_img = _img2 or p_img or ""
        if not p_mov or len(p_mov) < 8 or not re.search(r"[가-힣]", p_mov):
            _img2, _mov2 = _build_korean_prompts2(sc, styles, bg)
            p_mov = _mov2 or p_mov or ""
        return p_img, p_mov

    # ──────────────────────────────────────────────────────────────

    # 안전 복사
    S: dict = json.loads(json.dumps(story, ensure_ascii=False))

    # ------- 페이로드 구성 -------
    title = S.get('title') or ''
    lyrics_all = (S.get('lyrics') or '').strip()
    scenes = S.get('scenes') or []
    characters = sorted(set([
        (c.split(':',1)[0] if isinstance(c, str) else c.get('id', ''))
        for sc in scenes for c in (sc.get('characters') or [])
    ]))

    payload_scenes: List[dict] = []
    for sc in scenes:
        payload_scenes.append({
            "id": sc.get("id"),
            "section": (sc.get("section") or "").lower(),
            "hint": (sc.get("prompt") or ""),
            "effect": sc.get("effect") or [],
            "screen_transition": bool(sc.get("screen_transition")),
            "characters": [
                (c.split(':',1)[0] if isinstance(c,str) else c.get('id'))
                for c in (sc.get('characters') or [])
            ],
        })

    # 렌더 W/H 힌트
    W = int(((S.get("defaults") or {}).get("image") or {}).get("width") or 832)
    H = int(((S.get("defaults") or {}).get("image") or {}).get("height") or 1472)

    payload = {
        "title": title,
        "lyrics_all": lyrics_all,
        "characters": characters,
        "scenes": payload_scenes,
        "need_korean": True,
        "render_hint": {"image_width": W, "image_height": H},
        "rules": {
            # 캐릭터
            "character_styles": "모두 한국어. 성별(여성/남성) 명시. 선정성 강조 금지.",
            # 프롬프트
            "prompts": "항상 한국어. <id> 자리표시 금지. 자연어에는 여성/남성만 사용.",
            "prompt": "항상 한국어. 자연어에는 여성/남성만 사용. ID 노출 금지.",
            # 씬별 가사
            "per_scene_lyrics": "intro 제외하고 가능한 씬에 가사를 균형 배분. 한 줄씩 간결하게.",
            # 전역 컨텍스트
            "global": "전체 가사를 요약해 global_summary를 만들고, themes/palette/style_guide/negative_bank/section_moods/effect를 작성. defaults.image.width/height는 render_hint 사용."
        }
    }

    # ------- 시스템/유저 프롬프트 -------
    system = (
        "너는 영상 기획 보조 도구다. 반드시 한글만 써라.\n"
        "하나의 JSON만 반환한다:\n"
        "{"
        "\"character_styles\": {id:text,...},"
        "\"per_scene_lyrics\":[{\"id\":\"...\",\"lyric\":\"...\"}],"
        "\"prompts\":[{"
        "  \"id\":\"...\"," 
        "  \"prompt\":\"...\"," 
        "  \"prompt_img\":\"...\"," 
        "  \"prompt_movie\":\"...\"," 
        "  \"effect\":[\"...\"]"
        "}],"
        "\"global\":{"
        "  \"global_summary\":\"...\"," 
        "  \"themes\":[\"...\"]," 
        "  \"palette\":\"...\"," 
        "  \"style_guide\":\"...\"," 
        "  \"negative_bank\":\"...\"," 
        "  \"section_moods\": {\"intro\":\"...\",\"verse\":\"...\",\"chorus\":\"...\",\"bridge\":\"...\",\"outro\":\"...\"},"
        "  \"effect\":[\"...\"],"
        "  \"image_width\":0,"
        "  \"image_height\":0"
        "}"
        "}\n"
        "# 엄격한 작성 규칙:\n"
        "- 자연어에는 캐릭터 ID(female_01, male_01 등) 절대 표기 금지(여성/남성으로 서술).\n"
        "- 씬에 캐릭터가 2명 이상이면 반드시 위치를 함께 설명한다(예: '화면 왼쪽에 여성, 오른쪽에 남성').\n"
        "- prompt_img는 얼굴이 잘 보이도록 정면/반정면 구도를 기본 포함한다.\n"
        "- 각 씬의 effect 배열은 2~4개로 채운다(빈 배열 금지)."
    )

    user = json.dumps(payload, ensure_ascii=False)

    # ------- 호출 ------
    _t(trace, "ai:prepare", f"prefer={prefer or '(auto)'}, allow_fallback={allow_fallback if allow_fallback is not None else '(default)'}")
    raw = ask(system, user, prefer=prefer, allow_fallback=allow_fallback, trace=trace)
    if not raw or not str(raw).strip():
        raise RuntimeError("AI 응답이 비었습니다.")

    from json import JSONDecodeError

    txt = str(raw).strip()
    i, j = txt.find("{"), txt.rfind("}")
    try:
        data = json.loads(txt[i:j + 1] if (i != -1 and j != -1 and j > i) else txt)
    except JSONDecodeError:
        data = {}

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except JSONDecodeError:
            data = {}

    # ⬇️ 추가: 문자열이 한 번 더 중첩된 경우 재파싱
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            data = {}

    # 이후 기존 로직 그대로…
    styles_in = (data.get("character_styles") or {})
    styles = _enforce_character_style_rules({str(k): str(v) for k, v in styles_in.items()})

    # ------- 전역 컨텍스트 병합 -------
    _merge_global_context(S, data.get("global") or {})
    themes = (S.get("global_context") or {}).get("themes") or []

    # ------- 씬별 가사(폴백 포함) -------
    sc_lyrics = {d["id"]: (d.get("lyric") or "").strip()
                 for d in (data.get("per_scene_lyrics") or []) if d.get("id")}

    # ---- 가사 재주입(시간 겹침 기반)
    if isinstance(S.get("lyrics_sections"), list) and S.get("lyrics_sections"):
        _rec = _assign_scene_lyrics_by_time(S)
        sc_lyrics = {d.get("id"): (d.get("lyric") or "") for d in (_rec.get("scenes") or []) if d.get("id")}
        _t(trace, "info", "scene lyrics re-assigned by time overlap")

    need_fallback = (not sc_lyrics) or (len([v for v in sc_lyrics.values() if v]) < max(1, len(scenes) // 4))
    if need_fallback:
        rec_tmp = _segment_lyrics_for_scenes(S, audio_info=None, ai=None, lang="ko")
        sc_lyrics = {sc.get("id"): (sc.get("lyric") or "").strip()
                     for sc in (rec_tmp.get("scenes") or []) if sc.get("id")}
        _t(trace, "warn", "AI per_scene_lyrics 부족 -> 내부 세그먼트 폴백 사용")

    # ------- 씬 프롬프트(배경/이미지/무빙) + 강제 정리 -------
    prompts_in = {d["id"]: d for d in (data.get("prompts") or []) if d.get("id")}

    last_bg = ""  # 씬 간 반복 줄이기 위해 직전 배경문장 기억

    for idx, sc in enumerate(scenes):
        sid = sc.get("id")
        section = (sc.get("section") or "").lower()

        # 1) 배경 프롬프트(bg)
        if sid in prompts_in and (prompts_in[sid].get("prompt") or "").strip():
            bg = str(prompts_in[sid]["prompt"]).strip()
        else:
            bg = _fallback_scene_bg(section, themes)

        # 2) 이미지/무빙 프롬프트(있는 값 우선, 부족분 보강)
        if sid in prompts_in:
            p_img_raw = (prompts_in[sid].get("prompt_img") or "").strip()
            p_mov_raw = (prompts_in[sid].get("prompt_movie") or "").strip()
        else:
            p_img_raw, p_mov_raw = "", ""

        # 부족하면 내부 빌더로 보강
        p_img_raw, p_mov_raw = _must_fill_img_mov(sc, styles, bg, p_img_raw, p_mov_raw)

        # 3) 위치/ID 주입(기존 헬퍼) → 이후 자연어 ID 제거로 후정리
        p_img_pos = _ensure_positions_with_ids(sc, p_img_raw)
        p_mov_pos = _ensure_positions_with_ids(sc, p_mov_raw)

        # 4) 금칙/ID 제거 + 한국어 라벨링 보정
        bg = _ensure_korean_labeling(_sanitize_prompt_text(bg))
        p_img = _ensure_korean_labeling(_sanitize_prompt_text(p_img_pos))
        p_mov = _ensure_korean_labeling(_sanitize_prompt_text(p_mov_pos))

        # 5) 씬 간 중복 완화(카메라/조명/무빙 토큰 살짝 섞기)
        if last_bg and (bg == last_bg):
            bg = _diversify_across_scenes(idx, bg)
        last_bg = bg

        # 6) effect (AI > global > 폴백) + 정규화
        eff_ai = []
        if sid in prompts_in:
            eff_ai = prompts_in[sid].get("effect") or []
        if not eff_ai:
            eff_ai = (data.get("global") or {}).get("effect") or []
        eff = _ensure_effects(eff_ai)

        # 7) 저장
        sc["prompt"] = bg
        sc["prompt_img"] = p_img
        sc["prompt_movie"] = p_mov

        if section in ("intro", "outro", "bridge"):
            sc["lyric"] = ""
        else:
            sc["lyric"] = sc_lyrics.get(sid, "")

        sc["effect"] = eff

    # 저장 필드
    S["character_styles"] = styles
    S["scenes"] = scenes
    S.setdefault("audit", {})["generated_by"] = "gpt-5 (ko-prompts, strict-sanitized, global-context, per-scene-lyrics)"

    _t(trace, "gpt", "apply_gpt_to_story_v11 완료 (strict post-process)")
    return S





###############################################################
################################################################
###############################################################
###################추가(아래))###########################
###############################################################
################################################################
###############################################################



def _assign_scene_lyrics_by_time(story: dict) -> dict:
    """
    lyrics_sections 과 scenes의 시간 겹침으로 씬별 lyric을 안전하게 주입한다.
    - 포인터 j는 '가사 끝 <= 씬 시작'일 때만 전진 → 첫 씬 건너뛰기 방지
    - 섹션명이 intro/outro/bridge 라도, 시간이 겹치면 가사 주입
    """
    rec = deepcopy(story)
    sections = rec.get("lyrics_sections") or []
    scenes = rec.get("scenes") or []

    j = 0
    for sc in scenes:
        s0 = float(sc.get("start", 0.0))
        s1 = float(sc.get("end", 0.0))
        lyric_text = ""

        # 씬 시작보다 앞에서 끝나는 가사들만 건너뜀
        while j < len(sections) and float(sections[j].get("end", 0.0)) <= s0:
            j += 1

        # 겹치면 주입
        if j < len(sections):
            l0 = float(sections[j].get("start", 0.0))
            l1 = float(sections[j].get("end", 0.0))
            if not (l1 <= s0 or l0 >= s1):
                lyric_text = str(sections[j].get("text") or "")

        sc["lyric"] = lyric_text

    rec["scenes"] = scenes
    return rec



def _ensure_positions_with_ids(scene: dict, text: str) -> str:
    """
    scene.characters: ["female_01:0","male_01:1", ...] 또는 ["female_01","male_01", ...]
    2명 이상일 때 배치 문장을 text 앞에 주입.
    """
    chars = scene.get("characters") or []
    pairs: list[tuple[str, int]] = []
    for s in chars:
        if isinstance(s, str):
            if ":" in s:
                cid, idx_str = s.split(":", 1)
                try:
                    idx = int(idx_str.strip())
                except ValueError:
                    idx = 0
                pairs.append((cid.strip(), idx))
            else:
                pairs.append((s.strip(), 0))
    if len(pairs) < 2:
        return text

    pos_map = {
        0: "왼쪽",
        1: "오른쪽",
        2: "중앙",
        3: "왼쪽 뒤",
        4: "오른쪽 뒤",
    }

    pairs.sort(key=lambda x: x[1])
    parts = []
    for cid, idx in pairs:
        pos = pos_map.get(idx, f"기타{idx}")
        parts.append(f"{pos}에 {cid}")
    head = "장면 배치: " + ", ".join(parts) + ". 얼굴이 잘 보이도록 정면/반정면 구도."
    return f"{head} {text}".strip()




def _parse_marked_lyrics(lyrics_text: str) -> List[Dict[str, Any]]:
    """
    [verse]/[chorus]/[bridge]/[outro] 헤더가 섞인 가사를 파싱해
    [{"section": str, "lines": [..]}] 형태로 돌려준다.
    헤더가 없다면 전체를 verse로 본다.
    """
    if not isinstance(lyrics_text, str):
        raise ValueError("lyrics_text must be a string")

    section = "verse"
    sections: List[Dict[str, Any]] = []
    buf: List[str] = []

    def flush(current_section: str, current_lines: List[str]) -> None:
        if current_lines:
            sections.append({"section": current_section, "lines": [ln.strip() for ln in current_lines if ln.strip()]})

    pattern = re.compile(r"^\s*\[(verse|chorus|bridge|outro)]\s*$", flags=re.IGNORECASE)
    for raw_line in lyrics_text.splitlines():
        line = raw_line.rstrip("\n")
        m = pattern.match(line)
        if m:
            flush(section, buf)
            section = m.group(1).lower()
            buf = []
        else:
            if line.strip():
                buf.append(line.strip())

    flush(section, buf)
    if not sections:
        return [{"section": "verse", "lines": [ln.strip() for ln in lyrics_text.splitlines() if ln.strip()]}]
    return sections


def _group_lines_semantically(lines: List[str], max_group_lines: int = 2) -> List[List[str]]:
    """
    AI 미사용 시의 안전한 휴리스틱:
    - 짧은/중간 길이는 2줄씩 묶기
    - 긴 줄은 단독 유지
    - 끝에 1줄 남으면 이전 1줄과 합치기(리듬 보정)
    """
    groups: List[List[str]] = []
    cur: List[str] = []
    for line in lines:
        line_len = len(line)
        if line_len >= 18:
            if cur:
                groups.append(cur)
                cur = []
            groups.append([line])
            continue
        cur.append(line)
        if len(cur) >= max_group_lines:
            groups.append(cur)
            cur = []
    if cur:
        if len(cur) == 1 and groups and len(groups[-1]) == 1 and len(groups[-1][0]) < 14:
            prev = groups.pop()
            groups.append(prev + cur)
        else:
            groups.append(cur)
    return groups


def split_lyrics_into_semantic_units(lyrics_text: str, ai: Optional[Any] = None) -> List[Dict[str, Any]]:
    """
    의미 단위 배열을 반환.
    각 원소: {"section": "verse|chorus|bridge|outro", "text": "...", "lines": [..]}

    ai 객체에 segment_lyrics(sections=[...])가 있으면 우선 사용.
    없으면 위 휴리스틱으로 1~2줄 단위로 묶는다.
    """
    sections = _parse_marked_lyrics(lyrics_text)

    if ai is not None and hasattr(ai, "segment_lyrics"):
        try:
            result = ai.segment_lyrics(sections=sections)  # 기대 반환: 위와 동일 스키마
            if isinstance(result, list) and result and all(isinstance(u, dict) for u in result):
                norm: List[Dict[str, Any]] = []
                for unit in result:
                    sec = str(unit.get("section", "verse")).lower()
                    ln_list = [str(x) for x in unit.get("lines", [])]
                    txt = unit.get("text") or " ".join(ln_list)
                    norm.append({"section": sec, "text": txt.strip(), "lines": ln_list})
                if norm:
                    return norm
        except (TypeError, ValueError, RuntimeError):
            pass  # 휴리스틱으로 폴백

    units: List[Dict[str, Any]] = []
    for sec in sections:
        groups = _group_lines_semantically(sec["lines"])
        for g in groups:
            units.append({"section": sec["section"], "text": " ".join(g).strip(), "lines": g})
    return units


def build_scenes_from_units(units: List[Dict[str, Any]],
                            boundaries: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
    """
    의미 단위 + 시간 경계로 최소 필드의 scenes를 만든다.
    (프롬프트/이펙트 등은 기존 후처리 단계에서 덧입히면 됨)
    """
    if len(units) != len(boundaries):
        raise ValueError("units and boundaries must have the same length")

    scenes: List[Dict[str, Any]] = []
    for idx, (unit, tpair) in enumerate(zip(units, boundaries), start=1):
        start_t, end_t = tpair
        duration = max(0.0, float(end_t) - float(start_t))
        scene_id = f"t_{idx:03d}"
        scenes.append({
            "id": scene_id,
            "section": unit.get("section", "verse"),
            "start": float(start_t),
            "end": float(end_t),
            "duration": duration,
            "scene": f"{unit.get('section','verse').capitalize()}_{idx:02d}",
            "characters": [],
            "character_objs": [],
            "effect": [],
            "screen_transition": False,
            "img_file": "",
            "clip_file": "",
            "prompt_img": "",
            "prompt_movie": "",
            "prompt_negative": "@global",
            "needs_character_asset": True,
            "prompt": "",
            "lyric": unit.get("text", "")
        })
    return scenes


def apply_semantic_units_to_story(record: dict, ai=None):
    """
    story dict 하나 받아서:
    - 가사 한 줄 추출
    - AI로 의미단위 분리
    - 오디오(vocal.mp3) 있으면 길이 보정
    - 시간 배분 & 씬에 매핑
    반환: 수정된 story dict
    """
    audio = record.get("audio")
    audio_info = {"duration": float(record.get("duration", 0.0))}
    # (librosa 온셋 분석 등은 audio_sync.py 참고해서 이어붙일 수 있음)
    fixed = _segment_lyrics_for_scenes(record, audio_info=audio_info, ai=ai, lang=record.get("lang","ko"))
    return fixed

###############################################################

# ─────────────────────────────────────────────────────────────
# 의미단위 분할/시간배분/섹션/스켈레톤/자산 보존 머지 유틸
# ─────────────────────────────────────────────────────────────

# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석
# -*- coding: utf-8 -*-
"""
story_enrich.py — 가사 의미단위 분할 유틸 (flow_all / breath8 / breath_auto / auto)
- flow_all: 원문 흐름/반복 그대로 유지, 길기만 하면 추가 분할(병합 없음, 1~2자만 앞에 부착)
- breath8: 짧은 곡(≈20s)용, 호흡 위주 8~10컷 느낌
- breath_auto: 곡 길이에 따라 컷 수 자동(초/컷 비율 기반), 호흡 규칙 적용
- auto: AI→규칙 보정의 일반 프리셋
"""


TraceFn = Callable[[str, str], None]


# ─────────────────────────────────────────────────────────────
# 공통 보조
# ─────────────────────────────────────────────────────────────
def _normalize_text_ko(s: str) -> str:
    s = (s or "").replace("\r", "").strip()
    s = re.sub(r"\s+", " ", s)
    # 쉼표 주변 공백 정리
    s = s.replace(" ,", ",").replace(" , ", ", ")
    return s


def _split_basic_chunks(text: str) -> List[str]:
    """쉼표/줄바꿈/구분자 기준 1차 분절 (반복 보존)."""
    parts = re.split(r"[,\n/·;]+", text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def _split_long_ko(u: str, *, max_chars_per_unit: int = 18) -> List[str]:
    """한국어 길이 제한 분할: 접속/보조어미 경계 우선, 그 다음 공백 기준."""
    u = (u or "").strip()
    if not u:
        return []
    if len(u) <= max_chars_per_unit:
        return [u]

    # 1) 접속/보조어미 경계 후보에서 우선 분할
    parts = re.split(
        r"\s*(?:(?:그리고|하지만|그러나)\s+|(?:하고|하며|면서|고)\s+|(?:도록|싶어)\b)\s*",
        u,
    )
    parts = [p.strip() for p in parts if p and p.strip()]

    out: List[str] = []
    if len(parts) >= 2:
        for p in parts:
            if len(p) <= max_chars_per_unit:
                out.append(p)
            else:
                # 2) 공백 단위로 패킹
                toks = re.split(r"\s+", p)
                buf = ""
                for t in toks:
                    t = t.strip()
                    if not t:
                        continue
                    if not buf:
                        buf = t
                    elif len(buf) + 1 + len(t) <= max_chars_per_unit:
                        buf += " " + t
                    else:
                        out.append(buf)
                        buf = t
                if buf:
                    out.append(buf)
        return out

    # 접속 경계가 적절치 않으면 공백 기준으로 균등 패킹
    toks = re.split(r"\s+", u)
    buf = ""
    for t in toks:
        t = t.strip()
        if not t:
            continue
        if not buf:
            buf = t
        elif len(buf) + 1 + len(t) <= max_chars_per_unit:
            buf += " " + t
        else:
            out.append(buf)
            buf = t
    if buf:
        out.append(buf)
    return out


def _post_refine_flow_all(units: List[str]) -> List[str]:
    """
    '전체 흐름 그대로(반복 포함)' 규칙:
    - 1~2자 초소형 조각만 앞 조각에 붙임
    - 그 외 병합 없음, 순서 보존
    """
    outs: List[str] = []
    for u in units:
        u = (u or "").strip()
        if not u:
            continue
        if len(u) <= 2 and outs:
            outs[-1] = (outs[-1] + " " + u).strip()
        else:
            outs.append(u)
    return outs


# ─────────────────────────────────────────────────────────────
# 호흡 기반 보정(8~10컷 느낌)
# ─────────────────────────────────────────────────────────────
def _post_refine_breath8(units: List[str]) -> List[str]:
    """
    호흡 위주 8~10컷 느낌:
    - '어둠 속에' + '빛이 되어' 병합
    - '함께 걸어가고 싶어' + '너와 나' 병합
    - 너무 긴 덩어리는 동사/보조어미/접속 경계에서 분할
    - 1~2자 초소형은 앞에 부착
    - 최종 컷수 8~10 범위로 완만 보정
    """
    outs: List[str] = []
    i = 0
    while i < len(units):
        u = (units[i] or "").strip()
        if not u:
            i += 1
            continue
        n = (units[i + 1].strip() if i + 1 < len(units) else "")

        # 병합 규칙 1: 어둠 속에 + 빛이 되어
        if re.fullmatch(r"어둠\s*속에", u) and re.fullmatch(r"빛이\s*되어", n):
            outs.append("어둠 속에 빛이 되어")
            i += 2
            continue

        # 병합 규칙 2: 함께 걸어가고 싶어 + 너와 나
        if re.fullmatch(r"함께\s*걸어가고\s*싶어", u) and re.fullmatch(r"너와\s*나", n):
            outs.append("함께 걸어가고 싶어 너와 나")
            i += 2
            continue

        # 너무 긴 덩어리는 자연 경계에서 분할
        if len(u) > 18:
            cand = re.split(
                r"\s*(?:(?:하며|면서|고)\s+|(?:도록|싶어)\b|\s+느껴\b|\s+알아가\b|\s+되어\b|\s+새겨둘게\b|\s+이름으로\b)\s*",
                u,
            )
            cand = [c.strip() for c in cand if c.strip()]
            if 2 <= len(cand) <= 4:
                outs.extend(cand)
            else:
                outs.append(u)
        else:
            outs.append(u)
        i += 1

    # 초소형 부착
    merged: List[str] = []
    for s in outs:
        s = s.strip()
        if not s:
            continue
        if len(s) <= 2 and merged:
            merged[-1] = (merged[-1] + " " + s).strip()
        else:
            merged.append(s)

    # 목표 컷수 8~10 보정
    def _split_mid(x: str) -> List[str]:
        toks = x.split()
        if len(toks) >= 4:
            mid = len(toks) // 2
            return [" ".join(toks[:mid]).strip(), " ".join(toks[mid:]).strip()]
        midc = max(2, len(x) // 2)
        return [x[:midc].strip(), x[midc:].strip()]

    while len(merged) < 8 and any(len(x) > 18 for x in merged):
        j = max(range(len(merged)), key=lambda k: len(merged[k]))
        tgt = merged.pop(j)
        pieces = [p for p in _split_mid(tgt) if p]
        for z in reversed(pieces):
            merged.insert(j, z)

    while len(merged) > 10 and len(merged) >= 2:
        a = merged.pop()
        merged[-1] = (merged[-1] + " " + a).strip()

    return merged


# ─────────────────────────────────────────────────────────────
# 길이 기반 컷 수 자동 산정 + 개수 맞춤
# ─────────────────────────────────────────────────────────────
def _target_units_from_duration(
    duration_sec: float | None,
    *,
    sec_per_unit: float | None = None,
) -> Tuple[int, int]:
    """
    곡 길이→목표 컷수 범위 산출.
    - 기본: 구간별 sec_per_unit 가이드
      * ≤30s : 2.4 s/unit
      * ≤90s : 3.2 s/unit
      * ≤240s: 4.2 s/unit
      * >240s: 5.0 s/unit
    - 15% 가변 범위
    """
    if not duration_sec or duration_sec <= 0:
        return (8, 10)

    s_per = sec_per_unit or (
        2.4 if duration_sec <= 30 else
        3.2 if duration_sec <= 90 else
        4.2 if duration_sec <= 240 else
        5.0
    )
    base = max(8, duration_sec / s_per)
    low = max(8, int(round(base * 0.85)))
    high = min(120, int(round(base * 1.15)))
    if low > high:
        low, high = high, low
    return (low, high)


def _fit_to_target_count(units: List[str], target_min: int, target_max: int) -> List[str]:
    """목표 개수 범위로 맞추기: 많으면 뒤에서 병합, 적으면 가장 긴 항목을 반분."""
    out = [u for u in units if (u or "").strip()]

    # 많을 때: 뒤에서부터 인접 병합
    while len(out) > target_max and len(out) >= 2:
        a = out.pop()
        out[-1] = (out[-1] + " " + a).strip()

    # 적을 때: 가장 긴 항목을 반으로
    def _split_mid(x: str) -> List[str]:
        toks = x.split()
        if len(toks) >= 4:
            mid = len(toks) // 2
            return [" ".join(toks[:mid]).strip(), " ".join(toks[mid:]).strip()]
        midc = max(2, len(x) // 2)
        return [x[:midc].strip(), x[midc:].strip()]

    guard = 0
    while len(out) < target_min and any(len(u) > 8 for u in out):
        guard += 1
        if guard > 100:  # 안전장치
            break
        idx = max(range(len(out)), key=lambda i: len(out[i]))
        tgt = out.pop(idx)
        parts = [p for p in _split_mid(tgt) if p]
        for p in reversed(parts):
            out.insert(idx, p)

    return [u for u in out if u]


# ─────────────────────────────────────────────────────────────
# 메인: 의미단위 분할
# ─────────────────────────────────────────────────────────────
def split_lyrics_into_semantic_units_ai(
    lyrics: str,
    *,
    ai: Any | None = None,
    lang: str = "ko",
    trace: Optional[TraceFn] = None,
    min_units: int = 6,
    max_units: int = 12,
    max_chars_per_unit: int = 18,
    preset: str = "auto",          # auto | breath8 | breath_auto | flow_all
    duration_sec: float | None = None,
    sec_per_unit: float | None = None,
) -> List[str]:
    """
    가사를 의미 단위로 분할하는 통합 엔트리.
    - preset="flow_all": 원문 흐름/반복 보존, 길면 추가 분할만
    - preset="breath8": 호흡 위주 8~10컷 느낌
    - preset="breath_auto": 곡 길이에 따라 목표 컷 수 자동
    - preset="auto": AI→규칙 보정(일반)
    """
    text = _normalize_text_ko(lyrics)
    if not text:
        return []

    # ── flow_all: 전체 흐름 그대로(반복 포함) ──
    if preset == "flow_all":
        seeds = _split_basic_chunks(text)
        refined: List[str] = []
        for s in seeds:
            refined.extend(_split_long_ko(s, max_chars_per_unit=max_chars_per_unit))
        return _post_refine_flow_all(refined)

    # ── breath8: 짧은 곡 8~10컷 느낌 ──
    if preset == "breath8":
        # 1차: 표면 단서로 쪼갬
        seeds = _split_basic_chunks(text)
        # 2차: 길이 제한 충족
        refined: List[str] = []
        for s in seeds:
            refined.extend(_split_long_ko(s, max_chars_per_unit=max_chars_per_unit))
        # 3차: 호흡 감각 보정
        return _post_refine_breath8(refined)

    # ── breath_auto: 길이에 따라 목표 컷 수 자동 ──
    if preset == "breath_auto":
        tmin, tmax = _target_units_from_duration(duration_sec, sec_per_unit=sec_per_unit)

        # 1) 우선 AI 시도
        units: List[str]
        used_ai = False
        if ai is not None:
            try:
                system = (
                    f"한국어 가사를 의미 단위로 분해. 전체 {tmin}~{tmax}개, "
                    f"각 요소는 짧고 자연스럽게(최대 {max_chars_per_unit}자). "
                    "접속/보조어미(고/며/면서/도록/싶어 등) 경계 고려. JSON 배열만."
                )
                user = "자연스러운 호흡 단위로 분리:\n\n" + text
                raw = ai.ask_smart(
                    system, user,
                    prefer=getattr(ai, "default_prefer", "openai"),
                    allow_fallback=True,
                    trace=trace,
                )
                arr = json.loads(str(raw).strip())
                units = [str(x).strip() for x in arr if str(x).strip()]
                used_ai = True
            except Exception:
                units = _split_basic_chunks(text)
        else:
            units = _split_basic_chunks(text)

        # 2) 호흡 규칙 보정 + 목표 개수 맞춤
        units = _post_refine_breath8(units)
        units = _fit_to_target_count(units, tmin, tmax)
        return units

    # ── auto: 일반 경로(AI→규칙 보정) ──
    units: List[str]
    if ai is not None:
        try:
            system = (
                f"한국어 가사를 의미 단위로 분리. 전체 {min_units}~{max_units}개, "
                f"각 요소 최대 {max_chars_per_unit}자. 접속/보조어미 경계 고려. JSON 배열만."
            )
            user = "가사를 자연스러운 호흡 단위로 분리:\n\n" + text
            raw = ai.ask_smart(
                system, user,
                prefer=getattr(ai, "default_prefer", "openai"),
                allow_fallback=True,
                trace=trace,
            )
            arr = json.loads(str(raw).strip())
            units = [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            units = _split_basic_chunks(text)
    else:
        units = _split_basic_chunks(text)

    # 규칙 보정으로 호흡감만 정리
    units = _post_refine_breath8(units)
    return units





# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석


def _korean_weight(s: str) -> int:
    """가중치: 한글 음절/문자 수 기반(최소 1)."""
    t = (s or "").strip()
    # 한글/공백/일반문자 제외 모두 포함하되, 0 방지
    return max(1, len(re.findall(r"[가-힣A-Za-z0-9]", t)))


def layout_time_by_weights(
    units: List[str],
    *,
    total_start: float,
    total_end: float,
    onsets: Optional[List[float]] = None,
) -> List[float]:
    """유닛 가중치 비율로 시간 경계를 배치."""
    start = float(total_start)
    end = float(total_end)
    if end <= start or not units:
        return [start, end]

    weights = [_korean_weight(u) for u in units]
    s = sum(weights)
    ratios = [w / s for w in weights]
    dur = end - start

    boundaries = [start]
    acc = start
    for r in ratios:
        acc += dur * r
        boundaries.append(round(acc, 3))

    # onsets가 있으면 경계 스냅(선택적, 간단히 가장 가까운 onset으로 보정)
    if onsets:
        snapped = [boundaries[0]]
        for b in boundaries[1:-1]:
            near = min(onsets, key=lambda x: abs(x - b))
            snapped.append(round(near, 3))
        snapped.append(boundaries[-1])
        boundaries = snapped

    # 단조 증가 보장
    for i in range(1, len(boundaries)):
        if boundaries[i] <= boundaries[i-1]:
            boundaries[i] = round(boundaries[i-1] + 0.01, 3)
    return boundaries


def build_lyrics_sections(units: List[str], boundaries: List[float]) -> List[Dict[str, Any]]:
    """씬 섹션 목록 생성. 첫 구간 누락 방지용 인덱싱 검증 포함."""
    n = len(units)
    if n == 0:
        return []
    if len(boundaries) != n + 1:
        # 길이가 다르면 간단히 앞에서 n+1개만 사용
        boundaries = (boundaries + [boundaries[-1]])[: n + 1]

    sections: List[Dict[str, Any]] = []
    for i in range(n):
        sections.append({
            "id": f"L{i+1:02d}",
            "start": round(float(boundaries[i]), 3),
            "end": round(float(boundaries[i+1]), 3),
            "text": str(units[i]).strip(),
        })
    return sections


def build_scene_skeleton_from_units(
    units: List[str],
    boundaries: List[float],
    *,
    old_scenes: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """
    스켈레톤 씬: 시간/가사/섹션 id만 채움. 자산 필드는 두지 않음(머지 단계에서 보존).
    """
    n = len(units)
    if n == 0:
        return []
    if len(boundaries) != n + 1:
        boundaries = (boundaries + [boundaries[-1]])[: n + 1]

    skeleton: List[Dict[str, Any]] = []
    for i in range(n):
        sk = {
            "id": f"S{i+1:02d}",
            "name": f"scene_{i+1:02d}",
            "start": round(float(boundaries[i]), 3),
            "end": round(float(boundaries[i+1]), 3),
            "lyric": str(units[i]).strip(),
            "section": f"L{i+1:02d}",
        }
        # old_scenes가 있고 길이가 맞으면 기존 id/name만 참고(자산은 넣지 않음)
        if old_scenes and i < len(old_scenes or []):
            prev = old_scenes[i] or {}
            if prev.get("id"):
                sk["id"] = str(prev["id"])
            if prev.get("name"):
                sk["name"] = str(prev["name"])
        skeleton.append(sk)
    return skeleton


def merge_scenes_preserve_assets_strict(
    *,
    old_scenes: List[Dict[str, Any]],
    new_skeleton: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    핵심: 시간/가사/섹션만 갱신, 자산 필드는 유지.
    - 보존 대상 예: character_objs, img_file, clip_file, characters, effect, transition 등
    """
    keep_keys = {
        "character_objs", "img_file", "clip_file", "characters",
        "effect", "transition", "xfade", "left", "right", "center",
        "camera", "tags", "keep_asset",
    }

    merged: List[Dict[str, Any]] = []
    count = min(len(old_scenes or []), len(new_skeleton or []))
    for i in range(count):
        old = deepcopy(old_scenes[i] or {})
        sk = new_skeleton[i] or {}

        # 시간/가사/섹션 갱신
        old["id"] = sk.get("id") or old.get("id") or f"S{i+1:02d}"
        old["name"] = sk.get("name") or old.get("name") or f"scene_{i+1:02d}"
        old["start"] = float(sk.get("start") if sk.get("start") is not None else old.get("start") or 0.0)
        old["end"] = float(sk.get("end") if sk.get("end") is not None else old.get("end") or old.get("start") or 0.0)
        old["lyric"] = str(sk.get("lyric") or old.get("lyric") or "").strip()
        old["section"] = sk.get("section") or old.get("section") or f"L{i+1:02d}"

        # 보존 필드 외 나머지 키는 유지하되, 보존 키는 절대 덮어쓰지 않음
        for k in list(old.keys()):
            if k in keep_keys:
                continue
        merged.append(old)

    # 스켈레톤이 더 길면 뒤는 스켈레톤로 채우되, 자산 필드 빈값 유지
    for i in range(count, len(new_skeleton or [])):
        sk = deepcopy(new_skeleton[i])
        for k in list(sk.keys()):
            if k in keep_keys:
                del sk[k]
        merged.append(sk)

    return merged


# ─────────────────────────────────────────────────────────────
# 프롬프트/배치 후처리: 중복 좌/우 문구 정리, characters 인덱스 일관화
# ─────────────────────────────────────────────────────────────

def _normalize_layout_phrases(text: str) -> str:
    """'왼쪽에 ... 왼쪽에 ...' 같은 중복 배치 문구를 간단히 정규화."""
    t = (text or "").strip()
    if not t:
        return t
    # '왼쪽에'/'오른쪽에'/'중앙에' 반복 제거
    t = re.sub(r"(왼쪽에\s+[^,|]+)(?:\s*,?\s*왼쪽에\s+[^,|]+)+", r"\1", t)
    t = re.sub(r"(오른쪽에\s+[^,|]+)(?:\s*,?\s*오른쪽에\s+[^,|]+)+", r"\1", t)
    t = re.sub(r"(중앙에\s+[^,|]+)(?:\s*,?\s*중앙에\s+[^,|]+)+", r"\1", t)
    # '왼쪽/오른쪽' 단어 바로 연속 중복 제거
    t = re.sub(r"(왼쪽|오른쪽|중앙)(?:\s*\1)+", r"\1", t)
    return t


def postprocess_story_layout(story: Dict[str, Any]) -> Dict[str, Any]:
    """
    사후 정리:
    - prompt_img/prompt_movie 좌/우/중앙 중복 제거
    - characters를 'id:index' 형태로 강제
    - character_objs가 있으면 id:index 동기화
    - 첫 씬 intro 오인 교정(lyrics 기반)
    - 씬별 duration = end - start 재계산
    - 자산 파일(img_file/clip_file) 존재 시 needs_character_asset 자동 false
    """


    data = deepcopy(story or {})
    scenes = list(data.get("scenes") or [])

    for i, sc in enumerate(scenes):
        # 1) 프롬프트 정규화
        for k in ("prompt_img", "prompt_movie"):
            if sc.get(k):
                sc[k] = _normalize_layout_phrases(str(sc[k]))

        # 2) characters id:index 강제
        chars = []
        for j, ch in enumerate(sc.get("characters") or []):
            if isinstance(ch, str):
                cid = ch.split(":", 1)[0]
            else:
                cid = (ch or {}).get("id") or f"char{j+1}"
            chars.append(f"{cid}:{j}")
        if chars:
            sc["characters"] = chars

        # 3) character_objs 동기화
        cobjs = sc.get("character_objs") or []
        if cobjs and len(cobjs) == len(chars):
            new_objs = []
            for j, obj in enumerate(cobjs):
                o = dict(obj or {})
                o["id"] = chars[j].split(":", 1)[0]
                o["index"] = j
                new_objs.append(o)
            sc["character_objs"] = new_objs

        # 4) 첫 씬 intro 오인 교정
        if i == 0:
            lyr = str(sc.get("lyric") or "").strip()
            if lyr:
                # scene/name에 intro 흔적이 있으면 교정
                if str(sc.get("scene") or "").lower().startswith("intro"):
                    sc["scene"] = "verse_01"
                if str(sc.get("name") or "").lower().startswith("intro"):
                    sc["name"] = "scene_01"
                if str(sc.get("section") or "").lower() in ("intro",):
                    sc["section"] = "L01"

        # 5) duration 재계산 = end - start
        try:
            start_v = float(sc.get("start") or 0.0)
            end_v = float(sc.get("end") or start_v)
            dur_v = max(0.0, round(end_v - start_v, 3))
            sc["duration"] = dur_v
        except (TypeError, ValueError):
            # 값 문제면 일단 제거
            if "duration" in sc:
                del sc["duration"]

        # 6) 자산 파일 존재 시 needs_character_asset 자동 false
        has_img = False
        has_clip = False
        img_p = str(sc.get("img_file") or "").strip()
        clip_p = str(sc.get("clip_file") or "").strip()
        if img_p:
            try:
                has_img = os.path.exists(img_p)
            except OSError:
                has_img = False
        if clip_p:
            try:
                has_clip = os.path.exists(clip_p)
            except OSError:
                has_clip = False
        if has_img or has_clip:
            if sc.get("needs_character_asset") is True:
                sc["needs_character_asset"] = False

        scenes[i] = sc

    data["scenes"] = scenes

    # 루트 duration도 마지막 씬에 맞춰 정리(있을 때만)
    try:
        if scenes:
            total_start = float(data.get("offset") or 0.0)
            last_end = float(scenes[-1].get("end") or total_start)
            new_total = max(0.0, round(last_end - total_start, 3))
            data["duration"] = new_total
    except (TypeError, ValueError):
        pass

    # 루트 characters vs 실제 사용 동기화 권장(자동 제거는 위험하므로 여기선 미변경)
    return data



# ─────────────────────────────────────────────────────────────
# 기존 스타일 치환 완화: <id> 토큰 있을 때만 치환
# (파일 상단의 기존 _inject_styles_into_prompt 정의를 아래로 대체해도 됨)
# ─────────────────────────────────────────────────────────────




def _fallback_units_from_text(
    lyrics: str,
    *,
    min_units: int = 4,
    max_units: int = 8
) -> List[str]:
    """
    AI가 의미단위를 1개만 반환하거나 품질이 낮을 때,
    가사를 최소 min_units 이상으로 강제 분할한다.
    - 1차: 구두점(, / · / ·· / 그리고 / 하지만 / 그리고도) 기준
    - 2차: 공백 기반 균등 분할(문자 길이 균등화)
    """
    text = (lyrics or "").strip()
    if not text:
        return []

    # 1) 구두점/접속사 스플릿
    parts = re.split(r"(?:,|，|·|·+|그리고|하지만|그러나|그리고도|또는)\s*", text)
    parts = [p.strip() for p in parts if p and p.strip()]
    if len(parts) >= min_units:
        return parts[:max_units]

    # 2) 공백 단어 기반 균등 분할
    tokens = text.split()
    if not tokens:
        return [text]

    target = max(min_units, min(max_units, max(2, len(tokens) // 4)))
    # 블록 길이(토큰 수) 계산
    block = max(1, len(tokens) // target)
    units: List[str] = []
    for i in range(0, len(tokens), block):
        chunk = " ".join(tokens[i:i+block]).strip()
        if chunk:
            units.append(chunk)
        if len(units) >= max_units:
            break

    # 너무 잘게 쪼개졌으면 병합
    if len(units) < min_units and units:
        units = [text]  # 마지막 안전망
    return units







def recompute_durations_and_labels(story: Dict[str, Any]) -> Dict[str, Any]:
    """
    - 모든 씬 duration = round(end - start, 3)로 재계산
    - scene 라벨을 verse_01 ~ verse_N 으로 일관화
    - name도 scene_01 ~ scene_N으로 정리(기존 name이 있으면 유지하되 Intro_*는 교정)
    - 루트 duration도 마지막 씬 기준으로 재산출
    """
    data = deepcopy(story or {})
    scenes = list(data.get("scenes") or [])
    if not scenes:
        return data

    # 1) duration 재계산 + verse 라벨 통일
    verse_idx = 1
    for i, sc in enumerate(scenes):
        start_v = float(sc.get("start") or 0.0)
        end_v = float(sc.get("end") or start_v)
        dur_v = max(0.0, round(end_v - start_v, 3))
        sc["duration"] = dur_v

        # scene 라벨: verse_XX 로 강제
        sc_label = str(sc.get("scene") or "").strip().lower()
        if not sc_label or sc_label.startswith("intro") or not sc_label.startswith("verse"):
            sc["scene"] = f"verse_{verse_idx:02d}"
        else:
            sc["scene"] = f"verse_{verse_idx:02d}"
        # name 교정: Intro_* 등은 scene_XX 로
        nm = str(sc.get("name") or "").strip().lower()
        if not nm or nm.startswith("intro"):
            sc["name"] = f"scene_{i+1:02d}"
        # 저장
        scenes[i] = sc
        verse_idx += 1

    data["scenes"] = scenes

    # 2) 루트 duration 재계산(마지막 end - offset)
    try:
        total_start = float(data.get("offset") or 0.0)
        last_end = float(scenes[-1].get("end") or total_start)
        data["duration"] = max(0.0, round(last_end - total_start, 3))
    except (TypeError, ValueError):
        pass

    return data




def finalize_story_coherence(story: Dict[str, Any]) -> Dict[str, Any]:
    """
    최종 일관성 패스(저장 직전 1회):
    1) duration/라벨 정리(recompute_durations_and_labels)  ← Intro_* → verse_01~ 로 통일(가사 해석 X, 순서 기반)
    2) needs_character_asset 필드 전체 제거
    3) 루트 characters ↔ 씬 등장 캐릭터 동기화
    4) prompt_img / prompt_movie / prompt 모두에 성별(여성/남성) 자동 반영(이미 있으면 미변경)
    5) prompt 계열 텍스트의 좌/우/중앙 중복 문구 간단 정규화(_normalize_layout_phrases 재사용)
    """
    def infer_gender(cid: str) -> str | None:
        t = (cid or "").lower()
        if t.startswith(("female", "girl", "woman")):
            return "여성"
        if t.startswith(("male", "boy", "man")):
            return "남성"
        return None

    def has_gender_phrase(txt: str) -> bool:
        return bool(re.search(r"(여성|남성)", (txt or "").strip()))

    def inject_gender(txt: str, genders: List[str]) -> str:
        t = (txt or "").strip()
        if not t or has_gender_phrase(t) or not genders:
            return t
        head = "여성 인물 중심, " if genders == ["여성"] else (
               "남성 인물 중심, " if genders == ["남성"] else
               "여성과 남성 인물 동시 등장, ")
        return head + t

    # 1) duration/라벨 정리 (Intro_* 제거, verse_01~로 통일)
    data = recompute_durations_and_labels(story)

    # 2) needs_character_asset 제거
    scenes = list(data.get("scenes") or [])
    for i, sc in enumerate(scenes):
        if "needs_character_asset" in sc:
            sc = dict(sc)
            del sc["needs_character_asset"]
        scenes[i] = sc
    data["scenes"] = scenes

    # 3) 루트 characters 동기화 + 4/5) 프롬프트 보강/정규화
    used_ids: List[str] = []
    per_scene_genders: List[List[str]] = []

    for sc in scenes:
        genders: List[str] = []
        for ch in (sc.get("characters") or []):
            cid = ch.split(":", 1)[0] if isinstance(ch, str) else (ch or {}).get("id")
            if cid and cid not in used_ids:
                used_ids.append(cid)
            g = infer_gender(cid or "")
            if g and g not in genders:
                genders.append(g)
        per_scene_genders.append(genders)

    if used_ids:
        data["characters"] = used_ids

    # prompt_img / prompt_movie / prompt 모두 처리
    for i, sc in enumerate(scenes):
        genders = per_scene_genders[i] if i < len(per_scene_genders) else []
        for key in ("prompt_img", "prompt_movie", "prompt"):
            if sc.get(key):
                txt = str(sc[key])
                txt = inject_gender(txt, genders)
                # 좌/우/중앙 중복 문구 정리(있을 때만)
                if "_normalize_layout_phrases" in globals():
                    txt = _normalize_layout_phrases(txt)
                sc[key] = txt
        scenes[i] = sc

    data["scenes"] = scenes
    data = label_scenes_by_kinds(data)
    return data



# ─────────────────────────────────────────────────────────────
# [verse]/[chorus]/[bridge]/[intro]/[outro] 태그 파싱
# ─────────────────────────────────────────────────────────────
def parse_tagged_lyrics(lyrics: str) -> List[Dict[str, str]]:
    """
    [verse], [chorus], [bridge], [intro], [outro] 섹션 헤더를 읽어
    줄 단위로 {kind, text}를 생성한다.
    - 공백/빈 줄은 무시
    - kind는 소문자 고정
    """
    text = (lyrics or "").strip()
    if not text:
        return []

    lines = text.splitlines()
    current = "verse"
    result: List[Dict[str, str]] = []
    for ln in lines:
        s = (ln or "").strip()
        if not s:
            continue
        m = re.match(r"^\[(intro|verse|chorus|bridge|outro)]\s*$", s, flags=re.I)
        if m:
            current = m.group(1).lower()
            continue
        result.append({"kind": current, "text": s})
    return result


def units_and_kinds_from_tagged(items: List[Dict[str, str]]) -> Tuple[List[str], List[str]]:
    """
    태그 파싱 결과에서 (units, kinds) 병렬 리스트 생성.
    """
    units: List[str] = []
    kinds: List[str] = []
    for it in items:
        t = (it.get("text") or "").strip()
        k = (it.get("kind") or "verse").strip().lower()
        if not t:
            continue
        units.append(t)
        kinds.append(k if k in ("intro", "verse", "chorus", "bridge", "outro") else "verse")
    return units, kinds


# ─────────────────────────────────────────────────────────────
# 섹션/스켈레톤에 kind 보존
# ─────────────────────────────────────────────────────────────
def build_lyrics_sections_with_kinds(
    units: List[str],
    boundaries: List[float],
    kinds: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    기존 build_lyrics_sections에 kind 필드를 추가한 버전.
    kinds 길이가 units와 다르면 무시.
    """
    n = len(units)
    if n == 0:
        return []
    if len(boundaries) != n + 1:
        boundaries = (boundaries + [boundaries[-1]])[: n + 1]

    have_kinds = bool(kinds and len(kinds or []) == n)
    sections: List[Dict[str, Any]] = []
    for i in range(n):
        sec = {
            "id": f"L{i+1:02d}",
            "start": round(float(boundaries[i]), 3),
            "end": round(float(boundaries[i+1]), 3),
            "text": str(units[i]).strip(),
        }
        if have_kinds:
            sec["kind"] = (kinds[i] or "verse").lower()
        sections.append(sec)
    return sections


def build_scene_skeleton_from_units_with_kinds(
    units: List[str],
    boundaries: List[float],
    *,
    kinds: Optional[List[str]] = None,
    old_scenes: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    스켈레톤에 kind를 함께 저장(자산은 비움 → 이후 머지에서 보존).
    """
    n = len(units)
    if n == 0:
        return []
    if len(boundaries) != n + 1:
        boundaries = (boundaries + [boundaries[-1]])[: n + 1]

    have_kinds = bool(kinds and len(kinds or []) == n)
    skeleton: List[Dict[str, Any]] = []
    for i in range(n):
        sk = {
            "id": f"S{i+1:02d}",
            "name": f"scene_{i+1:02d}",
            "start": round(float(boundaries[i]), 3),
            "end": round(float(boundaries[i+1]), 3),
            "lyric": str(units[i]).strip(),
            "section": f"L{i+1:02d}",
        }
        if have_kinds:
            sk["kind"] = (kinds[i] or "verse").lower()
        if old_scenes and i < len(old_scenes or []):
            prev = old_scenes[i] or {}
            if prev.get("id"):
                sk["id"] = str(prev["id"])
            if prev.get("name"):
                sk["name"] = str(prev["name"])
        skeleton.append(sk)
    return skeleton


# ─────────────────────────────────────────────────────────────
# kind 기반 라벨링: intro_01 / verse_01 / chorus_01 ...
# ─────────────────────────────────────────────────────────────
def label_scenes_by_kinds(story: Dict[str, Any]) -> Dict[str, Any]:
    """
    scenes[*].kind 를 읽어 scene 라벨을 kind_XX로 부여한다.
    kind 부재/비정상일 경우 verse로 간주.
    """
    data = deepcopy(story or {})
    scenes = list(data.get("scenes") or [])
    if not scenes:
        return data

    counters = {"intro": 0, "verse": 0, "chorus": 0, "bridge": 0, "outro": 0}
    for i, sc in enumerate(scenes):
        kind = str(sc.get("kind") or "verse").lower()
        if kind not in counters:
            kind = "verse"
        counters[kind] += 1
        sc["scene"] = f"{kind}_{counters[kind]:02d}"
        if not str(sc.get("name") or "").strip():
            sc["name"] = f"scene_{i+1:02d}"
        scenes[i] = sc

    data["scenes"] = scenes
    return data


def _merge_negative_texts(*parts: str, dedup: bool = True) -> str:
    """쉼표/줄바꿈로 구분된 네거티브들을 합치고 중복 제거"""
    bag = []
    def _push(s: str):
        for t in str(s or "").replace("\n", ",").split(","):
            t = t.strip()
            if not t:
                continue
            if dedup:
                if t.lower() not in [x.lower() for x in bag]:
                    bag.append(t)
            else:
                bag.append(t)
    for p in parts:
        _push(p)
    return ", ".join(bag)

def ensure_global_negative(story: dict) -> dict:
    """
    - settings.NEGATIVE_BANK_DEFAULT와 story.defaults.image.negative를 병합
    - @global 규칙 유지
    - 씬별 prompt_negative 누락 시 @global 부여
    - global_context.negative_bank를 항상 채워 에디터에서 보이게 함
    """
    # 0) 설정 기본값
    try:
        import settings as S
        default_bank = getattr(S, "NEGATIVE_BANK_DEFAULT", "")
    except Exception:
        default_bank = ""

    s = story or {}
    defaults = s.setdefault("defaults", {})
    image_def = defaults.setdefault("image", {})
    gc = s.setdefault("global_context", {})

    cur = str(image_def.get("negative") or "").strip()

    # 1) @global 사용 의사 결정
    #    - 비었거나 '@global'이면 '@global' 유지하고, 실제 문구는 global_context.negative_bank에 보관
    #    - '@global'이 포함되지 않은 경우엔 '현재값 + default_bank'를 합쳐 문자열로 둠
    if (not cur) or (cur.strip() == "@global"):
        image_def["negative"] = "@global"
        # 에디터에서 보이도록 글로벌 뱅크는 항상 채움(기존 것과 설정 기본을 합침)
        gc["negative_bank"] = _merge_negative_texts(gc.get("negative_bank", ""), default_bank)
    else:
        if "@global" in cur:
            # '@global, foo, bar' 같은 형태 → 기본뱅크는 global_context에 두고,
            # defaults.image.negative는 그대로 '@global, foo, bar' 유지(중복 정리)
            merged_lit = _merge_negative_texts(cur)  # 중복만 정리
            image_def["negative"] = merged_lit
            gc["negative_bank"] = _merge_negative_texts(gc.get("negative_bank", ""), default_bank)
        else:
            # 리터럴만 쓰겠다는 의미 → 현재값 + 설정기본 병합해서 리터럴로 저장
            merged = _merge_negative_texts(cur, default_bank)
            image_def["negative"] = merged
            # 그래도 에디터 편의를 위해 글로벌 뱅크도 채워둔다(겹치면 dedup됨)
            gc["negative_bank"] = _merge_negative_texts(gc.get("negative_bank", ""), default_bank)

    # 2) 씬별 누락 보강: 비었으면 '@global'
    for sc in s.get("scenes") or []:
        if not str(sc.get("prompt_negative") or "").strip():
            sc["prompt_negative"] = "@global"

    return s

# story_enrich.py
BAN_TEXT = [
  "가사 원문 인용 금지", "문자 텍스트 삽입 금지"
]

def normalize_prompts(story: dict) -> dict:
    # @global → 실제 문자열 치환
    global_neg = (story.get("defaults", {}).get("image", {}) or {}).get("negative") or ""
    for sc in story.get("scenes", []):
        # 1) prompt_img / prompt / prompt_movie가 문자열인지 보장
        for k in ("prompt_img", "prompt", "prompt_movie", "prompt_negative"):
            if not isinstance(sc.get(k), str):
                sc[k] = sc.get(k) or ""

        # 2) @global 해소
        if sc.get("prompt_negative", "") == "@global":
            sc["prompt_negative"] = global_neg

        # 3) 금칙어/텍스트 제거(문자/자막/워터마크 등)
        txts = ["prompt_img", "prompt", "prompt_movie"]
        for k in txts:
            t = sc[k]
            for bad in ("text", "letters", "typography", "korean letters", "hangul", "hangeul",
                        "handwriting", "caption", "subtitles", "closed captions",
                        "워터마크", "자막", "문자", "글자", "글씨", "표지판", "간판", "로고"):
                t = t.replace(bad, "")
            # 필요시 콤마 정리
            sc[k] = ", ".join([s.strip() for s in t.split(",") if s.strip()])

    return story



