# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Callable, Any, Optional, Tuple
import os
import re
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
    base_sg = "실사, 일관된 헤어/의상/분위기, 자연스러운 조명"
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
    ######
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


# story_enrich.py 파일의 _enforce_character_style_rules 함수 전체를 교체하세요.

def _enforce_character_style_rules(styles: Dict[str, str]) -> Dict[str, str]:
    """[수정됨] 여성: 'huge breasts, slim legs'를 UI 체크 시(환경변수) 강제 포함."""
    out: Dict[str, str] = {}

    # --- ▼▼▼ [신규] UI 체크박스 상태(환경 변수) 읽기 ▼▼▼ ---
    # (이 함수가 호출될 때쯤이면 shorts_ui.py의 job 함수가
    #  os.environ["FORCE_HUGE_BREASTS"] = "1" 또는 "0"을 설정했어야 함)
    force_huge = os.environ.get("FORCE_HUGE_BREASTS") == "1"
    # --- ▲▲▲ [신규] 로직 끝 ▲▲▲ ---

    for cid, txt in (styles or {}).items():
        s = (txt or '').strip()
        if not s:
            s = cid
        # 성별 판정
        if re.search(r'female|여성', cid, re.I) or re.search(r'\bF\b', cid):
            if '여성' not in s:
                s = '여성, ' + s

            # --- ▼▼▼ [수정] 강제 주입 로직 변경 ▼▼▼ ---
            if force_huge:  # UI에서 체크했을 때만 주입
                if 'huge breasts' not in s:
                    s += ', huge breasts'
                if 'slim legs' not in s:
                    s += ', slim legs'
            # --- ▲▲▲ [수정] 로직 끝 ▲▲▲ ---

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


# story_enrich.py 파일의 apply_gpt_to_story_v11 함수 전체를 교체하세요.
# (파일 상단의 import os, re, json, ... 등은 그대로 둡니다)

def apply_gpt_to_story_v11(
        story: dict,
        *,
        ask: Callable[..., str],
        prefer: str | None = None,
        allow_fallback: bool | None = None,
        trace: TraceFn | None = None,
        temperature: float | None = None,
        force_huge_breasts: bool = False,  # <-- [신규] 체크박스 상태 전달
        **kwargs,
) -> dict:
    """
    [수정됨 v7] AI가 장면별 핵심 영어 태그를 제안하고, 코드가 이를 조합/보강합니다.
    - [수정] 씬에 'direct_prompt'가 있으면, AI 요청 시 'lyrics_all' 대신 'direct_prompt'를 우선 사용합니다.
    - [수정] AI에게 'context_source' 플래그를 전달하여 이 동작을 제어합니다.
    - [기존] 씬별 characters 목록을 분석하여 :0, :1... 인덱스를 자동 부여합니다.
    - [기존] 2인 이상 씬의 경우 AI payload의 hint 필드에 "장면 배치: 왼쪽 0번..." 프롬프트를 주입합니다.
    - [기존] force_huge_breasts=True일 때만 AI 시스템 프롬프트에 'huge breasts' 규칙을 동적으로 추가합니다.
    """
    if temperature is not None: _t(trace, "warn", f"ignored kw: temperature={temperature}")
    if kwargs: _t(trace, "warn", f"ignored extra kwargs: {list(kwargs.keys())}")

    import re
    import json
    from typing import List, Dict, Any, Set

    # ──────────────────────────────────────────────────────────────
    # 내부 유틸리티 함수들 (기존과 동일)
    # ──────────────────────────────────────────────────────────────
    def _clean_and_split_tags(text_input: str) -> List[str]:
        if not isinstance(text_input, str): return []
        text_cleaned = text_input.replace("\u200b", " ")
        tags_raw = re.split(r'[,/\n\s]+', text_cleaned)
        return [tag.strip() for tag in tags_raw if tag.strip()]

    def _combine_unique_tags(*tag_groups: Any) -> str:
        seen_tags: Set[str] = set()
        ordered_tags: List[str] = []
        for group in tag_groups:
            tags_to_process: List[str] = []
            if isinstance(group, list):
                tags_to_process = [str(item) for item in group if isinstance(item, str)]
            elif isinstance(group, str):
                tags_to_process = _clean_and_split_tags(group)
            for tag in tags_to_process:
                tag_cleaned = tag.strip()
                if not tag_cleaned: continue
                tag_lower = tag_cleaned.lower()
                if tag_lower not in seen_tags:
                    seen_tags.add(tag_lower)
                    ordered_tags.append(tag_cleaned)
        final_str = ", ".join(ordered_tags)
        return re.sub(r'\s*,\s*', ', ', final_str).strip(', ')

    # [기존] 캐릭터 스타일 한국어 설명을 영어 태그로 변환 (간단 버전)
    def _convert_char_style_ko_to_en(style_ko: str) -> List[str]:
        """간단한 규칙과 키워드 매핑으로 한국어 설명을 영어 태그 리스트로 변환"""
        if not style_ko: return []
        tags: List[str] = []
        style_lower = style_ko.lower()

        # 성별
        if "여성" in style_ko or "female" in style_lower:
            tags.append("young woman")
        elif "남성" in style_ko or "male" in style_lower:
            tags.append("young man")

        # [기존] 필수 태그 (UI 체크 여부 확인)
        # (os.environ은 shorts_ui.py의 job 함수에서 설정됨)
        if "young woman" in tags and os.environ.get("FORCE_HUGE_BREASTS") == "1":
            tags.extend(["huge breasts", "slim legs"])

        # 헤어 스타일
        hair_map = {"긴": "long hair", "짧은": "short hair", "중간": "medium hair", "웨이브": "wavy hair", "펌": "permed hair",
                    "생머리": "straight hair"}
        for ko, en in hair_map.items():
            if ko in style_ko: tags.append(en)

        # 헤어 색상
        color_map = {"갈색": "brown hair", "검정": "black hair", "금발": "blonde hair", "밝은": "light hair"}
        for ko, en in color_map.items():
            if ko in style_ko: tags.append(en)

        # 의상 (간단 키워드)
        clothing_map = {"후드티": "hoodie", "진": "jeans", "조거": "jogger pants", "팬츠": "pants", "드레스": "dress",
                        "셔츠": "shirt", "자켓": "jacket"}
        for ko, en in clothing_map.items():
            if ko in style_ko: tags.append(en)

        # 기타 특징
        if "게이머" in style_ko: tags.append("gamer style")
        if "안경" in style_ko: tags.append("wearing glasses")
        if "모자" in style_ko: tags.append("wearing hat")
        if "피곤" in style_ko: tags.append("tired expression")
        if "집중" in style_ko: tags.append("focused expression")
        if "미소" in style_ko: tags.append("slight smile")

        return _clean_and_split_tags(_combine_unique_tags(tags))  # 중복 제거 및 정리

    # --------------------------------------------------------------
    # - 안전 복사 및 페이로드 구성 (rules 수정)
    # --------------------------------------------------------------
    story_data = json.loads(json.dumps(story, ensure_ascii=False))
    title = story_data.get('title') or ''
    lyrics_all = (story_data.get('lyrics') or '').strip()
    scenes = story_data.get('scenes') or []

    # --- ▼▼▼ [수정됨] AI 요청 페이로드 생성 시 direct_prompt 로직 추가 ▼▼▼ ---
    characters_in_scenes = sorted(
        set([(c.split(':', 1)[0] if isinstance(c, str) else (c.get('id', '') if isinstance(c, dict) else '')) for sc in
             scenes if isinstance(sc, dict) for c in (sc.get('characters') or [])]))
    payload_scenes: List[dict] = []

    # [기존] ReActor 탐지 순서(왼쪽->오른쪽)에 맞춘 위치 맵
    POSITION_MAP = {
        0: "왼쪽",
        1: "오른쪽",
        2: "가운데",
        3: "왼쪽 뒤",
        4: "오른쪽 뒤",
    }

    # [기존] 씬 데이터를 순회하며 AI에게 보낼 페이로드(payload_scenes) 가공
    indexed_characters_map: Dict[str, List[str]] = {}  # 최종 저장을 위해 인덱스 부여된 캐릭터 목록 저장

    for sc_item in scenes:
        if not isinstance(sc_item, dict): continue
        scene_id = sc_item.get("id")

        # 1. [기존] 원본 캐릭터 ID 목록 추출 (예: ["female_01", "male_01"])
        original_char_ids = [
            (c.split(':', 1)[0] if isinstance(c, str) else (c.get('id') if isinstance(c, dict) else '')) for c in
            (sc_item.get('characters') or [])]
        original_char_ids = [cid for cid in original_char_ids if cid]  # 빈 ID 제거

        # [수정] 변수명 변경 (가리기 방지)
        original_hint_from_scene = (sc_item.get("prompt") or "").strip()  # AI에게 전달할 힌트 (기존 한국어 힌트)
        indexed_chars_for_ai: List[str] = []  # AI에게 전달할 인덱스 포함 목록

        num_chars_in_scene = len(original_char_ids)
        pos_prompt_for_layout = ""  # <--- [신규] 위치 프롬프트 초기화

        if num_chars_in_scene == 1:
            # 1명: :0 부여
            indexed_chars_for_ai = [f"{original_char_ids[0]}:0"]

        elif num_chars_in_scene > 1:
            # 2명 이상: :0, :1, :2... 순차 부여 및 위치 프롬프트 생성
            pos_descs_list = []
            for i, char_id_loop in enumerate(original_char_ids):
                indexed_chars_for_ai.append(f"{char_id_loop}:{i}")
                pos_name_str = POSITION_MAP.get(i, f"{i}번 위치")
                pos_descs_list.append(f"{pos_name_str}에 {char_id_loop}")

            # [수정] 힌트에 바로 주입하지 않고, pos_prompt 변수에 저장
            pos_prompt_for_layout = f"장면 배치: {', '.join(pos_descs_list)}. 자연스러움."

        # [기존] 최종 저장용 맵에 기록
        indexed_characters_map[scene_id] = indexed_chars_for_ai

        # --- ▼▼▼ [사용자 요청] direct_prompt 확인 로직 ▼▼▼ ---
        direct_prompt_text_from_scene = (sc_item.get("direct_prompt") or "").strip()

        context_source_for_ai: str
        final_hint_for_ai: str

        if direct_prompt_text_from_scene:
            # [A] direct_prompt가 있으면:
            context_source_for_ai = "direct_prompt_hint"
            # direct_prompt에 위치 프롬프트(2인 이상시)를 결합
            final_hint_for_ai = f"{pos_prompt_for_layout} {direct_prompt_text_from_scene}".strip()
        else:
            # [B] direct_prompt가 없으면 (기존 로직):
            context_source_for_ai = "global_lyrics_and_scene_hint"
            # 기존 한국어 힌트에 위치 프롬프트를 결합
            final_hint_for_ai = f"{pos_prompt_for_layout} {original_hint_from_scene}".strip()
        # --- ▲▲▲ [사용자 요청] 로직 끝 ▲▲▲ ---

        # AI 페이로드 씬 목록에 추가
        payload_scenes.append({
            "id": scene_id,
            "section": (sc_item.get("section") or "").lower(),
            "hint": final_hint_for_ai,  # [수정됨]
            "context_source": context_source_for_ai,  # [신규] AI에게 보낼 플래그
            "effect": sc_item.get("effect") or [],
            "screen_transition": bool(sc_item.get("screen_transition")),
            "characters": indexed_chars_for_ai,  # [수정됨]
        })
    # --- ▲▲▲ [수정됨] 씬 페이로드 가공 끝 ▲▲▲ ---

    render_defaults = (story_data.get("defaults") or {}).get("image") or {}
    render_width = int(render_defaults.get("width") or 832)
    render_height = int(render_defaults.get("height") or 1472)
    payload = {
        "title": title, "lyrics_all": lyrics_all, "characters": characters_in_scenes,
        "scenes": payload_scenes,  # [수정됨] 가공된 씬 페이로드
        "need_korean": True,
        "render_hint": {"image_width": render_width, "image_height": render_height},
        "rules": {
            "character_styles": "모두 한국어. 성별(여성/남성) 명시.",
            "prompts": "각 장면에 대해 prompt(한국어 설명), prompt_img_base(간결한 영어 핵심 태그: 배경/인물/행동), motion_hint(간결한 영어 모션 태그: 카메라/인물 움직임) 생성.",
            "prompt": "한국어. 가사 시각화 (배경, 인물, 행동). '장면 배치' 힌트가 있으면 반영.",
            "context_source": "각 씬의 'context_source' 필드 확인: 'direct_prompt_hint'면 'hint' 필드의 내용을 최우선으로 사용 (global lyrics 무시). 'global_lyrics_and_scene_hint'면 'lyrics_all'과 'hint'를 모두 참고.",
            # <-- [신규]
            "prompt_img_base": "영어 태그. 배경, 인물, 행동/상황 관련 핵심 키워드 5-10개. 예: 'night street, young woman walking, looking down, neon lights'.",
            "motion_hint": "영어 태그. 카메라 움직임 또는 인물 미세 동작 관련 키워드 1-3개. 예: 'slow zoom in', 'subtle eye blink', 'camera pan left'. 없으면 빈 문자열 `\"\"`.",
            "per_scene_lyrics": "intro 제외 가사 배분.",
            "global": "전체 요약 + 다양한 themes/palette/style_guide/negative_bank/section_moods/effect 작성."
        }
    }

    # --------------------------------------------------------------
    # - 시스템/유저 프롬프트 및 AI 호출 (force_huge_breasts 동적 적용)
    # --------------------------------------------------------------
    system_prompt_base = (
        "너는 영상 기획 보조 도구다.\n"
        "하나의 JSON만 반환한다:\n"
        "{\"character_styles\":{id:text,...},\"per_scene_lyrics\":[{\"id\":\"...\",\"lyric\":\"...\"}],"
        "\"prompts\":[{\"id\":\"...\",\"prompt\":\"...\",\"prompt_img_base\":\"...\",\"motion_hint\":\"...\",\"effect\":[\"...\"]}],"
        "\"global\":{\"global_summary\":\"...\",\"themes\":[\"...\"],\"palette\":\"...\",\"style_guide\":\"...\",\"negative_bank\":\"...\", "
        "\"section_moods\": {\"intro\":\"...\",\"verse\":\"...\",\"chorus\":\"...\",\"bridge\":\"...\",\"outro\":\"...\"},\"effect\":[\"...\"],"
        "\"image_width\":0,\"image_height\":0}}\n"
        "# 엄격한 작성 규칙:\n"
        "- [중요] 각 씬의 'context_source' 필드를 확인:\n"  # <-- [신규]
        "  - 'direct_prompt_hint'면: 'hint' 필드(사용자 직접 지시)를 최우선으로 사용하여 프롬프트 생성. (이 경우 'lyrics_all' 무시)\n"  # <-- [신규]
        "  - 'global_lyrics_and_scene_hint'면: 'lyrics_all'(전체 가사)과 'hint' 필드(씬 힌트)를 모두 참고하여 생성.\n"  # <-- [신규]
        "- character_styles: 한국어 설명.\n"
        "- prompt (장면 설명): 한국어. 가사 시각화 (배경, 인물, 행동). '장면 배치' 힌트가 있으면 반영.\n"
        "- prompt_img_base (이미지 핵심 태그): **영어**. 배경/인물/행동 관련 **핵심 태그 5-10개**. 쉼표 구분.\n"
        "- motion_hint (모션 힌트): **영어**. 카메라/인물 미세 동작 태그 **1-3개**. 없으면 빈 문자열 `\"\"`. 쉼표 구분.\n"
        "- global (전역 컨셉): 전체 요약 + **다양한 분위기** 포함.\n"
        "- effect 배열: 각 씬 2~4개 필수 (영어)."
    )

    # --- ▼▼▼ [기존] 'huge breasts' 규칙 동적 주입 ▼▼▼ ---
    system_prompt_final = system_prompt_base
    if force_huge_breasts:
        # [기존] 규칙이 "character_styles"를 타겟하도록 명시
        rule_marker = "- character_styles: 한국어 설명."
        rule_replacement = (
            "- character_styles: 한국어 설명.\n"
            "- [중요 규칙] 'female' ID를 가진 character_styles 설명에 'huge breasts, slim legs'를 반드시 한국어로 포함."
        )
        system_prompt_final = system_prompt_final.replace(rule_marker, rule_replacement)
        _t(trace, "ai:rule", "Injecting 'huge breasts' rule for AI.")
    # --- ▲▲▲ [기존] 주입 끝 ▲▲▲ ---

    user_prompt = json.dumps(payload, ensure_ascii=False)

    _t(trace, "ai:prepare",
       f"prefer={prefer or '(auto)'}, allow_fallback={allow_fallback if allow_fallback is not None else '(default)'}")
    raw_response = ask(system_prompt_final, user_prompt, prefer=prefer, allow_fallback=allow_fallback,
                       trace=trace)  # [기존] system_prompt_final 사용
    if not raw_response or not str(raw_response).strip(): raise RuntimeError("AI 응답이 비었습니다.")

    # [기존] JSON 파싱
    ai_data = {}
    try:
        from json import JSONDecodeError
        text_response = str(raw_response).strip()
        json_start, json_end = text_response.find("{"), text_response.rfind("}")
        if 0 <= json_start < json_end:
            json_str = text_response[json_start: json_end + 1]
            try:
                ai_data = json.loads(json_str)
            except JSONDecodeError:
                ai_data = {}
            if isinstance(ai_data, str):
                try:
                    ai_data = json.loads(ai_data)
                except JSONDecodeError:
                    ai_data = {}
    except (ImportError, NameError):
        try:
            ai_data = json.loads(raw_response)
        except ValueError:
            ai_data = {}
    if not isinstance(ai_data, dict): ai_data = {}

    # --------------------------------------------------------------
    # - AI 응답 데이터 처리 및 최종 프롬프트 조합 (v6) - (기존 로직 동일)
    # --------------------------------------------------------------
    styles_from_ai = (ai_data.get("character_styles") or {})
    character_styles_en_tags: Dict[str, List[str]] = {}
    for char_id_str, style_ko_str in styles_from_ai.items():
        if isinstance(char_id_str, str) and isinstance(style_ko_str, str):
            character_styles_en_tags[char_id_str] = _convert_char_style_ko_to_en(style_ko_str)
    character_styles_ko = {str(k): str(v) for k, v in styles_from_ai.items()}

    _merge_global_context(story_data, ai_data.get("global") or {})
    scene_lyrics_map = {d["id"]: (d.get("lyric") or "").strip() for d in (ai_data.get("per_scene_lyrics") or []) if
                        isinstance(d, dict) and d.get("id")}
    if not scene_lyrics_map:
        rec_tmp = _segment_lyrics_for_scenes(story_data, audio_info=None, ai=None, lang="ko")
        scene_lyrics_map = {sc.get("id"): (sc.get("lyric") or "").strip() for sc in (rec_tmp.get("scenes") or []) if
                            isinstance(sc, dict) and sc.get("id")}
        _t(trace, "warn", "AI per_scene_lyrics 부족 -> 내부 세그먼트 폴백 사용")

    prompts_from_ai = {d["id"]: d for d in (ai_data.get("prompts") or []) if isinstance(d, dict) and d.get("id")}

    QUALITY_TAGS = "photorealistic, cinematic lighting, high detail, 8k, masterpiece"
    DEFAULT_NEGATIVE_TAGS = "lowres, bad anatomy, bad proportions, extra limbs, extra fingers, missing fingers, jpeg artifacts, signature, logo, nsfw, text, letters, typography, watermark"

    # 각 씬 순회하며 최종 프롬프트 조합
    for scene_obj in scenes:
        if not isinstance(scene_obj, dict): continue
        scene_id_for_loop = scene_obj.get("id")  # 변수명 변경 (가리기 방지)
        if not scene_id_for_loop: continue

        # 1. AI 제안 데이터 가져오기
        ai_prompt_data_item = prompts_from_ai.get(scene_id_for_loop, {})  # 변수명 변경
        prompt_ko_from_ai = (ai_prompt_data_item.get("prompt") or "").strip()  # 변수명 변경
        prompt_img_base_from_ai = (ai_prompt_data_item.get("prompt_img_base") or "").strip()  # 변수명 변경
        motion_hint_base_from_ai = (ai_prompt_data_item.get("motion_hint") or "").strip()  # 변수명 변경
        current_scene_effects_list = ai_prompt_data_item.get("effect") or story_data.get("global_context", {}).get(
            "effect", [])  # 변수명 변경
        if not isinstance(current_scene_effects_list, list) or not all(
                isinstance(e, str) for e in current_scene_effects_list): current_scene_effects_list = []

        # 2. [기존] 씬의 인덱스(:0, :1)가 포함된 캐릭터 태그(영어) 수집
        char_tags_final_list: List[str] = []  # 변수명 변경
        indexed_char_list_for_scene = indexed_characters_map.get(scene_id_for_loop, [])  # AI 요청 시 사용했던 인덱스 목록 # 변수명 변경

        for char_ref_str in indexed_char_list_for_scene:  # 예: "female_01:0" # 변수명 변경
            char_id_from_ref = char_ref_str.split(':', 1)[0]  # 변수명 변경
            if char_id_from_ref and char_id_from_ref in character_styles_en_tags:
                char_tags_final_list.extend(character_styles_en_tags[char_id_from_ref])

        # 3. 최종 prompt_img: AI 베이스 태그 + 캐릭터 태그 + 효과 + 품질
        final_prompt_img_str = _combine_unique_tags(  # 변수명 변경
            prompt_img_base_from_ai,  # AI 제안 핵심 태그
            char_tags_final_list,  # 캐릭터 태그 (영어)
            current_scene_effects_list,  # 효과 태그
            QUALITY_TAGS  # 품질 태그
        )

        # 4. 최종 prompt_movie: 이미지 프롬프트 + AI 제안 모션 힌트
        final_prompt_movie_str = _combine_unique_tags(  # 변수명 변경
            final_prompt_img_str,  # 완성된 이미지 프롬프트
            motion_hint_base_from_ai  # AI 제안 모션 힌트 (없으면 빈 문자열)
        )

        # 5. 최종 prompt_negative
        global_ctx_data = story_data.get("global_context", {})  # 변수명 변경
        final_prompt_negative_str = _combine_unique_tags(global_ctx_data.get("negative_bank", ""),
                                                         DEFAULT_NEGATIVE_TAGS)  # 변수명 변경

        # 6. scene 객체에 최종 결과 저장
        scene_obj["prompt"] = prompt_ko_from_ai or scene_obj.get("prompt", "")  # 한국어 설명
        scene_obj["prompt_img"] = final_prompt_img_str  # 최종 조합 영어 태그
        scene_obj["prompt_movie"] = final_prompt_movie_str  # 최종 조합 영어 태그 + 모션
        scene_obj["prompt_negative"] = final_prompt_negative_str  # 네거티브
        scene_obj["effect"] = _clean_and_split_tags(" ".join(current_scene_effects_list))  # 효과
        scene_obj["lyric"] = scene_lyrics_map.get(scene_id_for_loop, scene_obj.get("lyric", ""))  # 가사

        # --- ▼▼▼ [기존] 인덱스가 적용된 캐릭터 목록을 씬에 저장 ▼▼▼ ---
        scene_obj["characters"] = indexed_characters_map.get(scene_id_for_loop, [])
        # --- ▲▲▲ [기존] 저장 끝 ▲▲▲ ---

    # --------------------------------------------------------------
    # - 최종 story_data 반환 (character_styles 저장 방식 변경됨)
    # --------------------------------------------------------------
    story_data["character_styles"] = character_styles_ko
    story_data["scenes"] = scenes
    story_data.setdefault("audit", {})["generated_by"] = "gpt-5-v11-final-prompts-v7-direct-prompt"  # 버전명 업데이트

    _t(trace, "gpt", "apply_gpt_to_story_v11 완료 (v7: direct_prompt 우선 적용)")
    return story_data





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
    head = "장면 배치: " + ", ".join(parts) # + ". 얼굴이 잘 보이도록 정면/반정면 구도."
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
) -> tuple[int, int]:
    """
    곡 길이→목표 컷수 범위 산출.
    """
    if duration_sec is None or duration_sec <= 0.0:
        return (8, 10)

    s_per = (
        float(sec_per_unit)
        if sec_per_unit is not None
        else (
            2.4 if duration_sec <= 30.0
            else 3.2 if duration_sec <= 90.0
            else 4.2 if duration_sec <= 240.0
            else 5.0
        )
    )

    base = max(8.0, duration_sec / s_per)  # float로 통일
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






# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석# 가사분석


def _korean_weight(s: str) -> int:
    """가중치: 한글 음절/문자 수 기반(최소 1)."""
    t = (s or "").strip()
    # 한글/공백/일반문자 제외 모두 포함하되, 0 방지
    return max(1, len(re.findall(r"[가-힣A-Za-z0-9]", t)))





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




# story_enrich.py 파일에서 이 함수를 찾아 아래 내용으로 전체를 교체하세요.
def finalize_story_coherence(story: Dict[str, Any]) -> Dict[str, Any]:
    """
    [수정됨] 최종 일관성 패스(저장 직전 1회): 이름 가리기 경고 수정.
    1) duration/라벨 정리 (recompute_durations_and_labels)
    2) needs_character_asset 필드 전체 제거 (purge_legacy_scene_keys 호출로 대체 가능)
    3) 루트 characters ↔ 씬 등장 캐릭터 동기화
    4) prompt_img / prompt_movie / prompt 모두에 성별(여성/남성) 자동 반영
    5) prompt 계열 텍스트의 좌/우/중앙 중복 문구 간단 정규화
    """
    # --- Helper Functions (No changes needed here) ---
    def infer_gender(char_id_input: str) -> str | None: # 파라미터 이름 변경 (cid -> char_id_input)
        """Helper to infer gender from character ID."""
        normalized_id = (char_id_input or "").lower()
        if normalized_id.startswith(("female", "girl", "woman")):
            return "여성"
        if normalized_id.startswith(("male", "boy", "man")):
            return "남성"
        return None

    def has_gender_phrase(text_input: str) -> bool: # 파라미터 이름 변경 (txt -> text_input)
        """Helper to check if gender phrase exists in text."""
        return bool(re.search(r"(여성|남성)", (text_input or "").strip()))

    def inject_gender(text_input: str, gender_list: List[str]) -> str: # 파라미터 이름 변경 (txt -> text_input, genders -> gender_list)
        """Helper to inject gender phrase into text."""
        cleaned_text = (text_input or "").strip()
        # 이미 성별 구문이 있거나, 텍스트가 비어있거나, 성별 리스트가 비어있으면 원본 반환
        if not cleaned_text or has_gender_phrase(cleaned_text) or not gender_list:
            return cleaned_text
        # 성별에 따른 머리말 결정
        if gender_list == ["여성"]:
            header = "여성 인물 중심, "
        elif gender_list == ["남성"]:
            header = "남성 인물 중심, "
        elif "여성" in gender_list and "남성" in gender_list:
             header = "여성과 남성 인물 동시 등장, "
        else: # 성별 리스트는 있으나 여성/남성이 아닌 경우 (혹은 단일 성별만 있는 경우)
             header = f"{gender_list[0]} 인물 중심, " # 첫 번째 성별 사용

        return header + cleaned_text
    # --- End Helper Functions ---

    # 1) duration/라벨 정리 (기존 호출 유지)
    data = recompute_durations_and_labels(story)

    # 2) 레거시 키 제거 (purge_legacy_scene_keys 함수 사용 권장)
    # data = purge_legacy_scene_keys(data) # 이 줄을 활성화하면 아래 루프 제거 가능
    # --- 임시: purge 함수가 없다면 기존 루프 방식 유지 (경고 없이) ---
    scenes = list(data.get("scenes") or [])
    updated_scenes_for_legacy = []
    for scene_data in scenes: # 루프 변수 이름 변경 (sc -> scene_data)
        if isinstance(scene_data, dict):
            current_scene = dict(scene_data) # 복사해서 작업
            if "needs_character_asset" in current_scene:
                current_scene.pop("needs_character_asset", None)
            updated_scenes_for_legacy.append(current_scene)
        else:
            updated_scenes_for_legacy.append(scene_data) # dict 아니면 그대로 추가
    data["scenes"] = updated_scenes_for_legacy
    # --- 레거시 키 제거 완료 ---

    # 3) 루트 characters 동기화 + 4/5) 프롬프트 보강/정규화
    used_character_ids: List[str] = []
    genders_per_scene: List[List[str]] = [] # 루프 밖에서 정의

    current_scenes = list(data.get("scenes") or []) # scenes 변수를 새로 정의
    for scene_item in current_scenes: # 루프 변수 이름 변경 (sc -> scene_item)
        scene_genders: List[str] = [] # 현재 씬의 성별 리스트 (내부 변수)
        if not isinstance(scene_item, dict): # scene_item 타입 확인
            genders_per_scene.append(scene_genders) # 빈 리스트 추가
            continue

        for character_ref in (scene_item.get("characters") or []):
            # 루프 내 캐릭터 ID 변수 이름 변경 (cid -> current_char_id)
            current_char_id = character_ref.split(":", 1)[0] if isinstance(character_ref, str) else (character_ref or {}).get("id")
            if current_char_id and current_char_id not in used_character_ids:
                used_character_ids.append(current_char_id)

            # 성별 추론 (헬퍼 함수 사용)
            gender = infer_gender(current_char_id or "")
            if gender and gender not in scene_genders:
                scene_genders.append(gender)
        genders_per_scene.append(scene_genders) # 현재 씬의 성별 리스트 저장

    if used_character_ids:
        data["characters"] = used_character_ids # 루트 캐릭터 목록 업데이트

    # 프롬프트 처리 루프
    final_scenes = [] # 최종 씬 리스트
    scenes_to_process = list(data.get("scenes") or []) # 다시 scenes 목록 가져오기
    for scene_index, scene_content in enumerate(scenes_to_process): # 루프 변수 이름 변경 (i -> scene_index, sc -> scene_content)
        if not isinstance(scene_content, dict): # 타입 확인
            final_scenes.append(scene_content) # dict 아니면 그대로 추가
            continue

        current_scene_data = dict(scene_content) # 복사해서 작업
        # 해당 씬의 성별 정보 가져오기
        genders_for_current_scene = genders_per_scene[scene_index] if scene_index < len(genders_per_scene) else []

        for key in ("prompt_img", "prompt_movie", "prompt"):
            if current_scene_data.get(key):
                # 루프 내 텍스트 관련 변수 이름 변경 (txt -> original_prompt_text 등)
                original_prompt_text = str(current_scene_data[key])
                # 성별 주입 (헬퍼 함수 사용)
                gender_injected_text = inject_gender(original_prompt_text, genders_for_current_scene)
                # 레이아웃 정규화 (전역 함수 사용)
                if "_normalize_layout_phrases" in globals():
                    normalized_layout_text = _normalize_layout_phrases(gender_injected_text)
                else:
                    normalized_layout_text = gender_injected_text
                # 최종 결과 저장
                current_scene_data[key] = normalized_layout_text

        final_scenes.append(current_scene_data) # 처리된 씬 데이터 추가

    data["scenes"] = final_scenes # 업데이트된 씬 리스트로 교체
    data = label_scenes_by_kinds(data) # 라벨링 함수 호출 (기존 유지)
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



