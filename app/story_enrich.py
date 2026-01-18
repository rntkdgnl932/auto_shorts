# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Callable, Any, Optional, Tuple, Set
import os
from copy import deepcopy
import re
import json
from pathlib import Path
from app.utils import load_json, save_json
from app.settings import I2V_CHUNK_BASE_FRAMES, I2V_OVERLAP_FRAMES, I2V_PAD_TAIL_FRAMES

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

# real_use
def _merge_global_context(s: dict, g: dict) -> None:
    """AI가 준 전역 컨텍스트를 story에 병합한다."""
    if not isinstance(g, dict):
        g = {}

    # 기본값 준비
    s.setdefault("global_context", {})
    gc = s["global_context"]

    # 1) 요약
    if g.get("global_summary"):
        gc["global_summary"] = str(g["global_summary"]).strip()

    # 2) themes
    th = g.get("themes") or []
    if isinstance(th, str):
        th = [x.strip() for x in re.split(r"[,|/]+", th) if x.strip()]
    gc["themes"] = th

    # 3) palette / style_guide / negative_bank / section_moods / effect
    if g.get("palette"):
        gc["palette"] = str(g["palette"]).strip()

    # style_guide: 고정 문구 + AI 추가
    base_sg = "실사, 일관된 헤어/의상/분위기, 자연스러운 조명"
    extra_sg = str(g.get("style_guide") or "").strip()
    gc["style_guide"] = base_sg if not extra_sg else f"{base_sg}, {extra_sg}"

    base_neg = "손가락 왜곡, 눈 왜곡, 과도한 보정, 노이즈, 흐릿함, 텍스트 워터마크"
    extra_neg = str(g.get("negative_bank") or "").strip()
    gc["negative_bank"] = base_neg if not extra_neg else f"{base_neg}, {extra_neg}"

    # section_moods: 기본 문구 + AI 추가
    def _sm(name: str, base: str) -> str:
        add = str((g.get("section_moods") or {}).get(name, "")).strip()
        return base if not add else f"{base}/{add}"

    gc["section_moods"] = {
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
    gc["effect"] = eff

    # 4) defaults.image (렌더 W/H 반영 + negative="@global")
    s.setdefault("defaults", {})
    s["defaults"].setdefault("image", {})
    di = s["defaults"]["image"]
    # 기존 값 우선, 없으면 g의 width/height 사용
    width = int(di.get("width") or g.get("image_width") or 832)
    height = int(di.get("height") or g.get("image_height") or 1472)
    di["width"] = width
    di["height"] = height
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
    lan = len(text)
    step = max(1, lan // n)
    out = []
    i = 0
    for k in range(n - 1):
        j = min(lan, i + step)
        # 경계 보정: 공백/구두점에서 끊기 시도
        m = re.search(r'[\s,.!?…]+', text[j:j+12])
        if m:
            j = j + m.start() + 1
        out.append(text[i:j].strip())
        i = j
    out.append(text[i:].strip())
    return out



def _segment_lyrics_for_scenes(record: dict, audio_info: dict = None, ai=None, lang: str = "ko"):
    """
    record: story.json dict
    audio_info = {"duration": float, "onsets": [초, ...]}  # onsets는 있어도 되고 없어도 됨
    결과:
      - record["lyrics_sections"] = [{id, start, end, text}...]
      - record["scenes"][i]["lyric"] = 해당 구간과 겹치는 첫 가사 또는 ""
    """


    # 원본 보존용 복사
    record_local = deepcopy(record)

    # 원래 가사 줄바꿈 보존
    lyrics_text_local = (record_local.get("lyrics") or "").strip()

    # 의미 단위로 분리 (이 함수는 외부에 있다고 가정)
    units_data_local = split_lyrics_into_semantic_units(lyrics_text_local, ai=ai)
    if not units_data_local:
        record_local["lyrics_sections"] = []
        return record_local

    # 여기서 dict → str 리스트로 정규화해서 이후 로직이 항상 list[str]을 보게 한다
    units_text_list: list[str] = []
    if isinstance(units_data_local, list):
        # case 1: 새 포맷 - dict 안에 text가 있는 경우
        if all(isinstance(u_item, dict) for u_item in units_data_local):
            for unit_item in units_data_local:
                unit_text = str(unit_item.get("text", "") or "")
                units_text_list.append(unit_text)
        # case 2: 예전 포맷 - 애초에 문자열 리스트로 오는 경우
        elif all(isinstance(u_item, str) for u_item in units_data_local):
            units_text_list = [str(u_item) for u_item in units_data_local]
        else:
            # 이상한 포맷이면 빈 결과 반환(기존 동작과 동일한 방어)
            record_local["lyrics_sections"] = []
            return record_local
    else:
        record_local["lyrics_sections"] = []
        return record_local

    # 여기까지 오면 units_text_list는 무조건 list[str]
    # 3) 시간 범위
    total_offset_val = float(record_local.get("offset", 0.0))
    total_duration_val = float(record_local.get("duration", 0.0))
    total_start_val = total_offset_val
    total_end_val = total_offset_val + total_duration_val

    # 4) 오디오 정보 있으면 duration 보정
    if audio_info:
        audio_duration_val = float(audio_info.get("duration") or 0.0)
        if audio_duration_val > 0.0 and abs(audio_duration_val - total_duration_val) > 0.25:
            total_end_val = total_start_val + audio_duration_val
            record_local["duration"] = round(audio_duration_val, 3)

    # 5) 한글 음절 수 기반 가중치
    def _syllable_w(text_input_local: str) -> int:
        korean_chars = re.findall(r"[가-힣]", text_input_local or "")
        if korean_chars:
            return len(korean_chars)
        # 한글이 없을 때는 공백 제거한 길이로라도 1 이상 주기
        no_space_len = len((text_input_local or "").replace(" ", ""))
        return no_space_len if no_space_len > 0 else 1

    weight_list = [_syllable_w(text_item) for text_item in units_text_list]
    weight_sum = sum(weight_list)
    if weight_sum <= 0:
        weight_list = [1] * len(units_text_list)
        weight_sum = len(units_text_list)
    ratio_list = [weight_val / weight_sum for weight_val in weight_list]

    # 6) 구간 경계 만들기
    boundaries_list = [total_start_val]
    acc_val = total_start_val
    for ratio_val in ratio_list:
        acc_val += (total_end_val - total_start_val) * ratio_val
        boundaries_list.append(acc_val)
    # 마지막 값은 총 길이에 스냅
    boundaries_list[-1] = total_end_val

    # 7) lyrics_sections 생성
    new_lyrics_list: list[dict] = []
    for idx, text_item in enumerate(units_text_list):
        seg_start_val = boundaries_list[idx]
        seg_end_val = boundaries_list[idx + 1]
        new_lyrics_list.append(
            {
                "id": f"L{idx + 1:02d}",
                "start": round(seg_start_val, 3),
                "end": round(seg_end_val, 3),
                "text": text_item,
            }
        )
    record_local["lyrics_sections"] = new_lyrics_list

    # 8) 씬에 첫 번째로 겹치는 가사 꽂아주기
    scenes_list_local = record_local.get("scenes", [])
    lyric_index = 0
    for scene_item_local in scenes_list_local:
        scene_start_val = float(scene_item_local.get("start", 0.0))
        scene_end_val = float(scene_item_local.get("end", 0.0))
        chosen_text = ""
        # lyrics가 씬 시작 이전에 끝난 건 건너뛰기
        while lyric_index < len(new_lyrics_list) and new_lyrics_list[lyric_index]["end"] <= scene_start_val:
            lyric_index += 1
        if lyric_index < len(new_lyrics_list):
            lyr_start_val = new_lyrics_list[lyric_index]["start"]
            lyr_end_val = new_lyrics_list[lyric_index]["end"]
            # 겹치는지 확인
            if not (lyr_end_val <= scene_start_val or lyr_start_val >= scene_end_val):
                chosen_text = new_lyrics_list[lyric_index]["text"]
        scene_item_local["lyric"] = chosen_text

    record_local["scenes"] = scenes_list_local
    return record_local



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


# real_use
def apply_ai_to_story_v11(
        story: dict,
        *,
        ask: Callable[..., str],
        prefer: str | None = None,
        allow_fallback: bool | None = None,
        trace: TraceFn | None = None,
        temperature: float | None = None,
        force_huge_breasts: bool = False,
        **kwargs,
) -> dict:
    """
    프로젝트 분석 → video.json 생성용 메인 함수 (v11, 캐릭터 5슬롯/프롬프트 문장형 개선 버전)
    """
    if temperature is not None:
        _t(trace, "warn", f"ignored kw: temperature={temperature}")
    if kwargs:
        _t(trace, "warn", f"ignored extra kwargs: {list(kwargs.keys())}")

    # --- 내부 유틸 ---
    def _clean_and_split_tags(text_input: str) -> List[str]:
        if not isinstance(text_input, str):
            return []
        text_cleaned = text_input.replace("\u200b", " ")
        tags_raw = re.split(r'[,/\n\s]+', text_cleaned)
        return [t.strip() for t in tags_raw if t.strip()]

    def _combine_unique_tags(*tag_groups: Any) -> str:
        seen_tags: Set[str] = set()
        ordered_tags: List[str] = []
        for group in tag_groups:
            tags_to_process: List[str] = []
            if isinstance(group, list):
                tags_to_process = [str(item) for item in group if isinstance(item, str)]
            elif isinstance(group, str):
                tags_to_process = [group]
            for t_val in tags_to_process:
                t_cleaned = t_val.strip()
                if not t_cleaned:
                    continue
                if t_cleaned not in seen_tags:
                    seen_tags.add(t_cleaned)
                    ordered_tags.append(t_cleaned)
        final_str = ", ".join(ordered_tags)
        return re.sub(r'\s*,\s*', ', ', final_str).strip(', ')

    def _convert_char_style_ko_to_en(style_ko: str) -> List[str]:
        if not style_ko: return []
        out_tags: List[str] = []
        style_lower = style_ko.lower()
        if "여성" in style_ko or "female" in style_lower:
            out_tags.append("young woman")
        elif "남성" in style_ko or "male" in style_lower:
            out_tags.append("young man")

        # 강제 옵션
        if "young woman" in out_tags and (force_huge_breasts or os.environ.get("FORCE_HUGE_BREASTS") == "1"):
            out_tags.extend(["huge breasts", "slim legs"])

        mapping = {
            "긴": "long hair", "웨이브": "wavy hair", "단발": "bob cut", "묶은": "ponytail",
            "안경": "glasses", "선글라스": "sunglasses", "모자": "hat", "귀걸이": "earrings",
            "정장": "suit", "드레스": "dress", "티셔츠": "t-shirt", "청바지": "jeans",
            "치마": "skirt", "교복": "school uniform", "수영복": "swimsuit", "비키니": "bikini",
            "운동복": "gym clothes", "한복": "hanbok", "기모노": "kimono",
            "검은": "black", "흰": "white", "빨간": "red", "파란": "blue",
            "노란": "yellow", "초록": "green", "보라": "purple", "분홍": "pink"
        }
        for k, v in mapping.items():
            if k in style_ko: out_tags.append(v)
        return out_tags

    # --- 입력 story 복사 및 전처리 ---
    story_data = json.loads(json.dumps(story, ensure_ascii=False))
    title = story_data.get("title") or ""
    lyrics_all = (story_data.get("lyrics") or "").strip()
    scenes = story_data.get("scenes") or []

    characters_in_scenes = sorted(set([
        (c.split(":", 1)[0] if isinstance(c, str) else (c.get("id", "") if isinstance(c, dict) else ""))
        for sc in scenes if isinstance(sc, dict) for c in (sc.get("characters") or [])
    ]))

    # --- 씬별 payload 준비 ---
    payload_scenes: List[dict] = []
    position_map = {0: "왼쪽", 1: "오른쪽", 2: "가운데", 3: "왼쪽 뒤", 4: "오른쪽 뒤"}
    indexed_characters_map: Dict[str, List[str]] = {}

    for sc_item in scenes:
        if not isinstance(sc_item, dict): continue
        scene_id = sc_item.get("id")
        if not scene_id: continue

        original_char_ids = [
            (c.split(":", 1)[0] if isinstance(c, str) else (c.get("id") if isinstance(c, dict) else "")) for c in
            (sc_item.get("characters") or [])]
        original_char_ids = [cid for cid in original_char_ids if cid]

        original_hint_from_scene = (sc_item.get("prompt") or "").strip()
        indexed_chars_for_ai: List[str] = []
        pos_prompt_for_layout = ""

        if len(original_char_ids) == 1:
            indexed_chars_for_ai = [f"{original_char_ids[0]}:0"]
        elif len(original_char_ids) > 1:
            pos_descs_list: List[str] = []
            for i, char_id_loop in enumerate(original_char_ids[:5]):
                indexed_chars_for_ai.append(f"{char_id_loop}:{i}")
                pos_name_str = position_map.get(i, f"{i}번 위치")
                pos_descs_list.append(f"{pos_name_str}에 {char_id_loop}")
            pos_prompt_for_layout = f"장면 배치: {', '.join(pos_descs_list)}."

        indexed_characters_map[scene_id] = indexed_chars_for_ai

        direct_prompt_text = (sc_item.get("direct_prompt") or "").strip()
        if direct_prompt_text:
            context_source_for_ai = "direct_prompt_hint"
            final_hint_for_ai = f"{pos_prompt_for_layout} {direct_prompt_text}".strip()
        else:
            context_source_for_ai = "global_lyrics_and_scene_hint"
            final_hint_for_ai = f"{pos_prompt_for_layout} {original_hint_from_scene}".strip()

        payload_scenes.append({
            "id": scene_id,
            "section": (sc_item.get("section") or "").lower(),
            "hint": final_hint_for_ai,
            "context_source": context_source_for_ai,
            "effect": sc_item.get("effect") or [],
            "screen_transition": bool(sc_item.get("screen_transition")),
            "characters": indexed_chars_for_ai,  # "id:index" 형태 전달
        })

    render_defaults = (story_data.get("defaults") or {}).get("image") or {}

    # Payload 구성
    payload = {
        "title": title,
        "lyrics_all": lyrics_all,
        "characters": characters_in_scenes,
        "scenes": payload_scenes,
        "need_korean": True,
        "render_hint": {
            "image_width": int(render_defaults.get("width") or 832),
            "image_height": int(render_defaults.get("height") or 1472),
        },
        "rules": {
            "prompts": "prompt(한글), prompt_img_base(영어 문장), motion_hint(영어) 생성.",
            "prompt_img_base": "ENGLISH SENTENCES ONLY.",
            "global": "전체 요약 및 영문 style_guide 포함.",
        },
    }

    # --- [핵심 수정] 시스템 프롬프트: 태그 방식 -> 문장 방식 변경 & 인덱스 규칙 강화 ---
    system_prompt_base = (
        "You are a professional AI Video Director.\n"
        "Return ONLY one JSON object.\n"
        "{\"character_styles\":{id:text,...},"
        "\"per_scene_lyrics\":[{\"id\":\"...\",\"lyric\":\"...\"}],"
        "\"prompts\":[{\"id\":\"...\",\"prompt\":\"...\",\"prompt_img_base\":\"...\",\"motion_hint\":\"...\",\"effect\":[\"...\"]}],"
        "\"global\":{...}}\n"
        "\n"
        "# STRICT RULES:\n"
        "1. **prompt (Scene Description)**: Korean. Used for user UI.\n"
        "2. **prompt_img_base (Visual Description)**: **STRICTLY ENGLISH SENTENCES**.\n"
        "   - **DO NOT** use comma-separated tags. Write a **descriptive, natural English sentence**.\n"
        "   - **[Character Reference Rule]**: The 'characters' field provides 'id:index'.\n"
        "     * ':0' -> MUST be referred to as 'from image 1'\n"
        "     * ':1' -> MUST be referred to as 'from image 2'\n"
        "     * ':2' -> 'from image 3', etc.\n"
        "   - **[Content Generation]**:\n"
        "     * Infer the character's gender from their ID (e.g. 'female' -> woman, 'male' -> man).\n"
        "     * **Invent specific outfits and actions** based on the scene mood, lyrics, and hint.\n"
        "     * Combine multiple characters into a coherent sentence.\n"
        "   - **[Example Output]**:\n"
        "     * BAD: 'woman, beach, summer, bikini, walking'\n"
        "     * GOOD: 'A woman from image 1, wearing a colorful bikini, is walking along the sunny beach smiling.'\n"
        "     * GOOD: 'A man from image 2 in a black suit is dancing with a woman from image 1 in a red dress under the streetlights.'\n"
        "3. **motion_hint**: ENGLISH ONLY. Camera or action phrases.\n"
        "4. **effect**: ENGLISH ONLY tags.\n"
        "5. **character_styles**: Korean description including gender/age/style.\n"
    )

    system_prompt_final = system_prompt_base
    if force_huge_breasts:
        rule_replacement = (
            "5. **character_styles**: Korean description including gender/age/body/overall style.\n"
            "   - [RULE] For 'female' characters, the Korean description MUST clearly imply "
            "'huge breasts and slim legs'."
        )
        system_prompt_final = system_prompt_final.replace(
            "5. **character_styles**: Korean description including gender/age/style.\n",
            rule_replacement,
        )

    # --- LLM 호출 ---
    user_prompt = json.dumps(payload, ensure_ascii=False)
    _t(trace, "ai:ask", "Requesting analysis with English sentences (image 1/2 references)...")

    raw_response = ask(
        system_prompt_final,
        user_prompt,
        prefer=prefer,
        allow_fallback=allow_fallback,
        trace=trace,
    )

    ai_data: Dict[str, Any] = {}
    try:
        text_response = str(raw_response).strip()
        json_start, json_end = text_response.find("{"), text_response.rfind("}")
        if 0 <= json_start < json_end:
            ai_data = json.loads(text_response[json_start: json_end + 1])
    except Exception:
        ai_data = {}

    # --- 후처리 ---
    styles_from_ai = ai_data.get("character_styles") or {}
    character_styles_ko = {str(k): str(v) for k, v in styles_from_ai.items()}

    # 글로벌 컨텍스트 병합 (기존 함수가 없으므로 직접 병합 로직 구현 - 보통 utils나 내부에 있어야 하나 독립 실행 보장 위해 간단 처리)
    global_ctx_ai = ai_data.get("global") or {}
    if "global_context" not in story_data:
        story_data["global_context"] = {}
    if isinstance(global_ctx_ai, dict):
        story_data["global_context"].update(global_ctx_ai)

    scene_lyrics_map = {d["id"]: (d.get("lyric") or "").strip() for d in (ai_data.get("per_scene_lyrics") or []) if
                        isinstance(d, dict) and d.get("id")}
    prompts_from_ai = {d["id"]: d for d in (ai_data.get("prompts") or []) if isinstance(d, dict) and d.get("id")}

    quality_tags = "photorealistic, cinematic lighting, high detail, 8k, masterpiece"

    default_negative_tags = (
        "lowres, bad anatomy, bad proportions, extra limbs, extra fingers, "
        "missing fingers, jpeg artifacts, signature, logo, nsfw, text, letters, "
        "typography, watermark"
    )

    for scene_obj in scenes:
        if not isinstance(scene_obj, dict): continue
        sid = scene_obj.get("id")
        if not sid: continue

        ai_item = prompts_from_ai.get(sid, {})
        p_ko = (ai_item.get("prompt") or "").strip()

        # [중요] AI가 생성한 문장 그대로 가져오기
        p_img_base = (ai_item.get("prompt_img_base") or "").strip()
        p_motion = (ai_item.get("motion_hint") or "").strip()
        p_effect = ai_item.get("effect") or story_data.get("global_context", {}).get("effect", [])

        final_img = p_img_base
        if quality_tags not in final_img:
            final_img = f"{final_img}, {quality_tags}"

        final_movie = f"{final_img}, {p_motion}"

        # 네거티브 처리
        global_neg = story_data.get("global_context", {}).get("negative_bank", "")
        raw_neg = _combine_unique_tags(global_neg, default_negative_tags)
        clean_neg_list = [t.strip() for t in raw_neg.split(",") if t.strip() and not re.search(r"[가-힣]", t)]
        final_neg = ", ".join(clean_neg_list)

        # Scene 업데이트
        scene_obj["prompt"] = p_ko or scene_obj.get("prompt", "")
        scene_obj["prompt_img"] = final_img
        scene_obj["prompt_movie"] = final_movie
        scene_obj["prompt_negative"] = final_neg
        scene_obj["effect"] = _clean_and_split_tags(" ".join(p_effect) if isinstance(p_effect, list) else str(p_effect))
        scene_obj["lyric"] = scene_lyrics_map.get(sid, scene_obj.get("lyric", ""))
        scene_obj["characters"] = indexed_characters_map.get(sid, scene_obj.get("characters", []))

    story_data["character_styles"] = character_styles_ko
    story_data["scenes"] = scenes
    story_data.setdefault("audit", {})["generated_by"] = "gpt-5-v11-english-sentences-strict-slots"

    _t(trace, "gpt", "apply_gpt_to_story_v11 완료 (Sentence Mode)")
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

# real_use
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


def plan_segments_s_e(total_frames: int, base_chunk: int = 41) -> List[Tuple[int, int]]:
    """
    [Wan 전용] 단순 (start, end) 세그먼트 분할.
    - 오버랩 없음.
    - 각 세그먼트 길이는 최대 base_chunk 프레임.
    - 마지막 세그먼트는 남은 프레임만 사용.

    예)
      total_frames=33 → [(0, 33)]
      total_frames=41 → [(0, 41)]
      total_frames=60 → [(0, 41), (41, 60)]
      total_frames=65 → [(0, 41), (41, 65)]
    """
    out: List[Tuple[int, int]] = []

    if total_frames <= 0:
        return out
    if base_chunk <= 0:
        # base_chunk가 0이거나 음수면 전체를 한 번에
        out.append((0, total_frames))
        return out

    start = 0
    while start < total_frames:
        end = start + base_chunk
        if end > total_frames:
            end = total_frames
        out.append((start, end))
        start = end

    return out

def plan_i2v_frame_segments(
        total_frames: int,
        *,
        base_frames: int = I2V_CHUNK_BASE_FRAMES,
        overlap_frames: int = I2V_OVERLAP_FRAMES,
        pad_tail_frames: int = I2V_PAD_TAIL_FRAMES,
) -> List[Dict[str, int]]:
    """
    I2V 롱/쇼핑 기준 세그먼트 분할.

    - video.json에는 "유효 구간" 기준의 start_frame/end_frame 를 기록한다.
      (end_frame은 exclusive: [start, end) )
    - 실제 생성(Comfy)용으로는 overlap/pad를 고려한 gen_start_frame/gen_end_frame도 함께 제공한다.

    규칙:
      seg0:
        effective: 0 ~ base
        generate : 0 ~ base + pad
      segN (N>=1):
        effective: prev_end ~ prev_end + base (또는 남은 프레임)
        generate : (start - overlap) ~ (end + pad)
    """
    out: List[Dict[str, int]] = []
    if total_frames <= 0:
        return out

    base = max(1, int(base_frames))
    ov = max(0, int(overlap_frames))
    pad = max(0, int(pad_tail_frames))

    start = 0
    while start < total_frames:
        end = min(total_frames, start + base)

        if start == 0:
            gen_s = 0
        else:
            gen_s = max(0, start - ov)

        gen_e = min(total_frames, end + pad)

        out.append({
            "start_frame": int(start),
            "end_frame": int(end),
            "gen_start_frame": int(gen_s),
            "gen_end_frame": int(gen_e),
        })

        start = end

    return out


# shorts 탭 video.json 빌드
def fill_prompt_movie_with_ai(
        project_dir: "Path",
        ask: "Callable[[str, str], str]",
        *,
        log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """
    [통일 패치]
    - fill_prompt_movie_with_ai / fill_prompt_movie_with_ai_long 모두
      동일한 I2V 세그먼트 분할 기준을 사용한다:
        I2V_CHUNK_BASE_FRAMES, I2V_OVERLAP_FRAMES, I2V_PAD_TAIL_FRAMES
    - shorts 탭 video.json도 shopping/i2v롱과 같은 내부 세그 구조(frame_segments)를 갖도록 만든다.
    """
    import json

    def _log(msg: str) -> None:
        if callable(log_fn):
            try:
                log_fn(msg)
            except Exception:
                pass

    pdir = Path(project_dir).resolve()
    vpath = pdir / "video.json"

    vdoc: Dict[str, Any] = load_json(vpath, {}) or {}
    if not isinstance(vdoc, dict):
        _log("[fill_prompt_movie_with_ai] video.json 형식 오류")
        return

    # 원본 분위기 (project.json)
    pj_path = pdir / "project.json"
    original_vibe_prompt = ""
    if pj_path.exists():
        pj_doc = load_json(pj_path, {}) or {}
        if isinstance(pj_doc, dict):
            original_vibe_prompt = pj_doc.get("prompt_user") or pj_doc.get("prompt", "") or ""

    # FPS 확정
    defaults_map: Dict[str, Any] = vdoc.get("defaults") or {}
    movie_def: Dict[str, Any] = defaults_map.get("movie") or {}
    image_def: Dict[str, Any] = defaults_map.get("image") or {}

    fps_candidates = [movie_def.get("target_fps"), vdoc.get("fps"), image_def.get("fps"), movie_def.get("fps"), 24]
    fps = 24
    for cand in fps_candidates:
        if cand is None:
            continue
        try:
            fps = int(cand)
            break
        except Exception:
            continue

    vdoc.setdefault("fps", fps)
    vdoc.setdefault("defaults", {})
    vdoc["defaults"].setdefault("movie", {})
    vdoc["defaults"]["movie"]["target_fps"] = fps
    vdoc["defaults"]["movie"]["input_fps"] = fps
    vdoc["defaults"]["movie"]["fps"] = fps
    vdoc["defaults"].setdefault("image", {})["fps"] = fps

    # I2V 분할 기준(통일)
    base_frames = int(I2V_CHUNK_BASE_FRAMES)
    overlap_frames = int(I2V_OVERLAP_FRAMES)
    pad_tail_frames = int(I2V_PAD_TAIL_FRAMES)

    scenes = vdoc.get("scenes") or []
    if not isinstance(scenes, list):
        _log("[fill_prompt_movie_with_ai] scenes 없음")
        save_json(vpath, vdoc)
        return

    changed = False

    # 시스템 프롬프트(연속성 강제)
    system_msg = (
        "You are a Strict AI Cinematographer specializing in I2V continuity.\n"
        "Your goal is to generate segment prompts for ONE continuous shot.\n\n"
        "[ABSOLUTE PROHIBITIONS]\n"
        "❌ NO turning around (back view).\n"
        "❌ NO full rotation.\n"
        "❌ NO hiding important subjects.\n\n"
        "[MANDATORY RULES]\n"
        "1) Segment N must start from the end state of Segment N-1.\n"
        "2) Keep camera angle stable. Use micro-movements.\n"
        "3) Output MUST be ENGLISH.\n"
        "Return JSON only: {\"segment_prompts\": [\"...\", ...]}\n"
    )

    forced_negative = (
        "nsfw, watermark, text, ugly, distorted face, "
        "back view, turning around, extra fingers, mutated hands, "
        "blurry, signature, logo, subtitle, words, caption"
    )

    for i, sc in enumerate(scenes):
        if not isinstance(sc, dict):
            continue

        sid = str(sc.get("id") or f"scene_{i:05d}")

        # negative 강화
        current_neg = str(sc.get("prompt_negative") or "").strip()
        if not current_neg:
            sc["prompt_negative"] = forced_negative
            changed = True
        elif "back view" not in current_neg.lower():
            sc["prompt_negative"] = current_neg + ", " + forced_negative
            changed = True

        # duration -> total_frames
        try:
            dur = float(sc.get("duration") or 0.0)
        except Exception:
            dur = 0.0
        if dur <= 0:
            try:
                dur = float(sc.get("seconds") or 0.0)
            except Exception:
                dur = 0.0

        total_frames = int(round(dur * fps)) if dur > 0 else 0
        if total_frames <= 0:
            continue

        sc["total_frames"] = total_frames
        sc["fps"] = fps
        sc["overlap_frames"] = overlap_frames

        # ★ 통일된 frame_segments 생성
        segs = sc.get("frame_segments")
        if not isinstance(segs, list) or not segs:
            segs_out = plan_i2v_frame_segments(
                total_frames,
                base_frames=base_frames,
                overlap_frames=overlap_frames,
                pad_tail_frames=pad_tail_frames,
            )
            # prompt_movie 슬롯 추가
            for seg in segs_out:
                seg.setdefault("prompt_movie", "")
            sc["frame_segments"] = segs_out
            segs = segs_out
            changed = True

        # seg_count도 frame_segments 기반으로 통일
        sc["seg_count"] = int(len(segs))

        # 이미 다 채워졌으면 스킵
        if all(str(seg.get("prompt_movie") or "").strip() for seg in segs):
            _log(f"[{sid}] 세그먼트 프롬프트 이미 존재 (스킵)")
            continue

        # base_visual
        base_visual = ""
        for key in ("prompt_img_1", "prompt_img", "prompt"):
            v = sc.get(key)
            if isinstance(v, str) and v.strip():
                base_visual = v.strip()
                break

        scene_lyric = str(sc.get("lyric") or "").strip()
        if not base_visual and not scene_lyric:
            _log(f"[{sid}] 참조 텍스트 부족 (스킵)")
            continue

        next_scene_lyric = "(Scene End)"
        if i + 1 < len(scenes):
            nsc = scenes[i + 1]
            if isinstance(nsc, dict):
                next_scene_lyric = str(nsc.get("lyric") or "").strip() or "(Next scene has no lyric)"

        frame_ranges_info = [f"{s.get('start_frame')}-{s.get('end_frame')}f" for s in segs]

        user_payload = {
            "original_vibe": original_vibe_prompt,
            "scene_lyric": scene_lyric,
            "base_visual": base_visual,
            "characters": sc.get("characters", []),
            "time_structure": frame_ranges_info,
            "next_scene_lyric": next_scene_lyric,
            "instruction": "Generate chained prompts. Maintain continuity."
        }
        user_msg = json.dumps(user_payload, ensure_ascii=False)

        _log(f"[{sid}] AI 프롬프트 생성 요청 (segments={len(segs)})")

        try:
            ai_raw = ask(system_msg, user_msg)

            json_start = ai_raw.find("{")
            json_end = ai_raw.rfind("}") + 1
            if not (0 <= json_start < json_end):
                raise RuntimeError(f"AI JSON 응답 형식 오류: {ai_raw[:80]}")

            ai_json = json.loads(ai_raw[json_start:json_end])
            new_prompts = ai_json.get("segment_prompts", [])

            if not isinstance(new_prompts, list):
                raise RuntimeError("AI segment_prompts가 list가 아님")

            # 길이 불일치면 가능한 범위만 채우기
            filled = 0
            for k, seg in enumerate(segs):
                if k >= len(new_prompts):
                    break
                if str(seg.get("prompt_movie") or "").strip():
                    continue
                p_text = str(new_prompts[k] or "").strip()
                if p_text:
                    seg["prompt_movie"] = p_text
                    filled += 1

            if filled:
                sc["frame_segments"] = segs
                changed = True
                _log(f"[{sid}] 세그먼트 프롬프트 {filled}개 채움")

        except Exception as e:
            _log(f"[{sid}] AI 호출 실패: {e}")
            continue

    # 저장
    save_json(vpath, vdoc)
    if changed:
        _log("[fill_prompt_movie_with_ai] 업데이트 완료 (video.json 저장)")
    else:
        _log("[fill_prompt_movie_with_ai] 변경 없음 (video.json 저장)")


# shopping 탭 json
def fill_prompt_movie_with_ai_shopping(
        story_data: dict,
        ai_ask_func: Callable[[str, str], str],
        trace: TraceFn | None = None
) -> dict:
    """
    [Step 6] Long-Take Shopping 스타일:
    각 씬을 지정된 FPS/Chunk 단위로 쪼개고,
    AI에게 "시간 흐름에 따른 연속적 프롬프트(문장)" 생성을 요청하여 채워넣는다.
    """
    import math
    import json
    import re

    # 1. UI 설정값 로드 (없으면 기본값)
    ui_prefs = (story_data.get("defaults") or {}).get("ui_prefs") or {}
    try:
        fps = float(ui_prefs.get("movie_fps", 30))
    except:
        fps = 30.0

    # Long-Take 설정 (기본값: 쇼핑 스타일 81프레임)
    base_chunk = I2V_CHUNK_BASE_FRAMES  # e.g. 81 or 118
    overlap = I2V_OVERLAP_FRAMES  # e.g. 10 or 20
    pad_tail = I2V_PAD_TAIL_FRAMES  # e.g. 20

    scenes = story_data.get("scenes", [])
    if not scenes:
        return story_data

    _t(trace, "info", f"🚀 [AI Long-Take] 프롬프트 상세화 시작 (FPS: {fps}, chunk={base_chunk}, ov={overlap}, pad={pad_tail})")

    for sc in scenes:
        sid = sc.get("id")
        # 1) 프레임 계산
        try:
            duration = float(sc.get("duration") or 2.0)
            total_frames = int(duration * fps)
        except:
            total_frames = 60

        # 2) 세그먼트 개수 계산 (단순 나눗셈이 아니라 Overlap 고려)
        # 필요한 유효 길이 = total_frames
        # 첫 청크 = base_chunk
        # 이후 청크 추가분 = base_chunk - overlap
        # 식: base_chunk + (n-1)*(base_chunk - overlap) >= total_frames + pad_tail
        # (n-1) * step >= target - base
        step = base_chunk - overlap
        target = total_frames + pad_tail

        if target <= base_chunk:
            seg_count = 1
        else:
            needed = target - base_chunk
            additional = math.ceil(needed / step)
            seg_count = 1 + additional

        _t(trace, "info", f"   - Scene {sid}: {duration:.2f}s * {fps}fps = {total_frames}f -> segments={seg_count}")

        # 메타데이터 저장
        sc["frame_segments"] = {
            "fps": fps,
            "total_frames": total_frames,
            "segment_count": seg_count,
            "base_chunk": base_chunk,
            "overlap": overlap,
            "segments": []  # 여기에 채움
        }

        # AI 요청 준비
        base_prompt = sc.get("prompt_img") or sc.get("prompt") or "A cinematic shot"
        # 문장형인지 확인 (대소문자 구별 없이)
        is_sentence_mode = len(base_prompt.split()) > 6  # 대략 6단어 이상이면 문장으로 간주

        # 3) AI에게 시퀀스 프롬프트 요청 (1개면 굳이 요청 안하고 복사할 수도 있지만, 일관성 위해 요청 권장)
        #    단, 1개이고 내용이 짧으면 그냥 복사
        if seg_count == 1 and not is_sentence_mode:
            sc["frame_segments"]["segments"] = [base_prompt]
            continue

        # 시스템 프롬프트: 문장형 흐름을 요청
        sys_msg = (
            "You are an AI Video Sequencer.\n"
            "Break down the provided 'base_prompt' into a sequence of prompts for a continuous video shot.\n"
            f"Target segments: {seg_count}\n"
            "\n"
            "OUTPUT FORMAT (JSON ONLY):\n"
            "{\n"
            "  \"segment_prompts\": [\n"
            "    \"String: Prompt for segment 1 (start)\",\n"
            "    \"String: Prompt for segment 2 (middle action)...\",\n"
            "    ...\n"
            "  ]\n"
            "}\n"
            "\n"
            "RULES:\n"
            "1. The output list MUST have exactly the requested number of segments.\n"
            "2. Ensure continuous action flow. Do not change the character's clothing or core appearance.\n"
            "3. If 'base_prompt' is a sentence, maintain the sentence structure but advance the action slightly.\n"
            "4. Use ENGLISH sentences only.\n"
        )

        user_msg = json.dumps({
            "scene_id": sid,
            "base_prompt": base_prompt,
            "segment_count_needed": seg_count,
            "duration_sec": duration
        }, ensure_ascii=False)

        try:
            raw = ai_ask_func(sys_msg, user_msg)

            # [강화된 파싱 로직] 마크다운 제거 및 JSON 추출
            text_cleaned = raw.strip()
            # ```json ... ``` 제거
            if "```" in text_cleaned:
                text_cleaned = re.sub(r"```json|```", "", text_cleaned).strip()

            # 중괄호 찾기
            idx_start = text_cleaned.find("{")
            idx_end = text_cleaned.rfind("}")

            if idx_start != -1 and idx_end != -1:
                json_str = text_cleaned[idx_start: idx_end + 1]
                parsed = json.loads(json_str)

                prompts_list = parsed.get("segment_prompts", [])

                # 개수 부족하면 마지막꺼 복사, 넘치면 자름
                if len(prompts_list) < seg_count:
                    last_p = prompts_list[-1] if prompts_list else base_prompt
                    while len(prompts_list) < seg_count:
                        prompts_list.append(last_p)
                elif len(prompts_list) > seg_count:
                    prompts_list = prompts_list[:seg_count]

                # 결과 저장
                sc["frame_segments"]["segments"] = [str(p).strip() for p in prompts_list]

            else:
                raise ValueError("JSON braces not found")

        except Exception as e:
            _t(trace, "info", f"❌ Scene {sid} AI prompt failed: {e}")
            # 실패 시 폴백: 기본 프롬프트를 모든 세그먼트에 복사
            fallback_list = [base_prompt] * seg_count
            sc["frame_segments"]["segments"] = fallback_list

    _t(trace, "info", "✅ [AI Long-Take] 프롬프트 상세화 완료")
    return story_data


# shorts 탭 json
def fill_prompt_movie_with_ai_shorts(
        story_data: dict,
        ai_ask_func: Callable[[str, str], str],
        trace: TraceFn | None = None
) -> dict:
    """
    [Step 6] Long-Take Shopping 스타일:
    각 씬을 지정된 FPS/Chunk 단위로 쪼개고,
    AI에게 "시간 흐름에 따른 연속적 프롬프트(문장)" 생성을 요청하여 채워넣는다.

    [수정] 캐릭터/의상/배경 보존 강화:
    - 모든 세그먼트 프롬프트에 '의상'과 '배경' 묘사를 강제로 포함시킴.
    """
    import math
    import json
    import re

    # 1. UI 설정값 로드
    ui_prefs = (story_data.get("defaults") or {}).get("ui_prefs") or {}
    try:
        fps = float(ui_prefs.get("movie_fps", 30))
    except:
        fps = 30.0

    # Long-Take 설정 (기본값)
    base_chunk = I2V_CHUNK_BASE_FRAMES  # e.g. 81
    overlap = I2V_OVERLAP_FRAMES  # e.g. 10
    pad_tail = I2V_PAD_TAIL_FRAMES  # e.g. 20

    scenes = story_data.get("scenes", [])
    if not scenes:
        return story_data

    _t(trace, "info", f"🚀 [AI Long-Take] 프롬프트 상세화 시작 (FPS: {fps}, chunk={base_chunk})")

    for sc in scenes:
        sid = sc.get("id")

        # 1) 프레임/세그먼트 계산
        try:
            duration = float(sc.get("duration") or 2.0)
            total_frames = int(duration * fps)
        except:
            total_frames = 60

        step = base_chunk - overlap
        target = total_frames + pad_tail

        if target <= base_chunk:
            seg_count = 1
        else:
            needed = target - base_chunk
            additional = math.ceil(needed / step)
            seg_count = 1 + additional

        _t(trace, "info", f"   - Scene {sid}: {duration:.2f}s ({total_frames}f) -> segments={seg_count}")

        # 메타데이터 저장
        sc["frame_segments"] = {
            "fps": fps,
            "total_frames": total_frames,
            "segment_count": seg_count,
            "base_chunk": base_chunk,
            "overlap": overlap,
            "segments": []
        }

        # 베이스 프롬프트 (상세 묘사가 있는 prompt_img 우선)
        base_prompt = sc.get("prompt_img") or sc.get("prompt") or "A cinematic shot"

        # 2) 단순 복사 vs AI 요청 판단
        # 세그먼트가 1개이고 내용이 짧으면 그냥 복사 (API 절약)
        is_long_text = len(base_prompt.split()) > 10
        if seg_count == 1 and not is_long_text:
            sc["frame_segments"]["segments"] = [base_prompt]
            continue

        # ------------------------------------------------------------
        # [핵심 수정] 시스템 프롬프트 강화: 의상/배경 고정 명령
        # ------------------------------------------------------------
        sys_msg = (
            "You are an AI Video Sequencer specializing in consistent character animation.\n"
            "Your task is to break down the 'base_prompt' into a sequence of prompts for a long-take video.\n"
            f"Target segments: {seg_count}\n"
            "\n"
            "*** CRITICAL REQUIREMENT (Visual Consistency) ***\n"
            "Every single segment prompt MUST explicitly include:\n"
            "1. The BACKGROUND description (e.g., 'in a dark alley', 'sunny park').\n"
            "2. The Character's APPEARANCE & CLOTHING (e.g., 'wearing a red dress', 'in a suit').\n"
            "3. The specific ACTION for that segment.\n"
            "\n"
            "Do NOT assume the AI remembers the previous segment. You MUST repeat the clothing and background details in every line.\n"
            "\n"
            "OUTPUT FORMAT (JSON ONLY):\n"
            "{\n"
            "  \"segment_prompts\": [\n"
            "    \"[Background] [Character + Clothing] [Action for start]\",\n"
            "    \"[Background] [Character + Clothing] [Action for middle]\",\n"
            "    ...\n"
            "  ]\n"
            "}\n"
            "\n"
            "RULES:\n"
            "1. The output list MUST have exactly the requested number of segments.\n"
            "2. NEVER change the clothing or background. Consistency is key.\n"
            "3. Use ENGLISH sentences only.\n"
        )

        user_msg = json.dumps({
            "scene_id": sid,
            "base_prompt": base_prompt,
            "segment_count_needed": seg_count,
            "duration_sec": duration,
            "instruction": "Ensure the character's clothing and background are described in EVERY segment."
        }, ensure_ascii=False)

        try:
            raw = ai_ask_func(sys_msg, user_msg)

            # 파싱 로직
            text_cleaned = raw.strip()
            if "```" in text_cleaned:
                text_cleaned = re.sub(r"```json|```", "", text_cleaned).strip()

            idx_start = text_cleaned.find("{")
            idx_end = text_cleaned.rfind("}")

            if idx_start != -1 and idx_end != -1:
                json_str = text_cleaned[idx_start: idx_end + 1]
                parsed = json.loads(json_str)
                prompts_list = parsed.get("segment_prompts", [])

                # 개수 보정
                if len(prompts_list) < seg_count:
                    last_p = prompts_list[-1] if prompts_list else base_prompt
                    while len(prompts_list) < seg_count:
                        prompts_list.append(last_p)
                elif len(prompts_list) > seg_count:
                    prompts_list = prompts_list[:seg_count]

                sc["frame_segments"]["segments"] = [str(p).strip() for p in prompts_list]

            else:
                raise ValueError("JSON braces not found")

        except Exception as e:
            _t(trace, "info", f"❌ Scene {sid} AI prompt failed: {e}")
            # 실패 시 폴백
            sc["frame_segments"]["segments"] = [base_prompt] * seg_count

    _t(trace, "info", "✅ [AI Long-Take] 프롬프트 상세화 완료 (의상/배경 고정)")
    return story_data

