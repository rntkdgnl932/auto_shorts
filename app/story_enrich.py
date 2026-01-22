# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Callable, Any, Optional, Tuple, Set, SupportsFloat, SupportsIndex
import os
from copy import deepcopy
import re
import math
import json
from pathlib import Path
from app.utils import load_json, save_json
from app.settings import I2V_CHUNK_BASE_FRAMES, I2V_OVERLAP_FRAMES, I2V_PAD_TAIL_FRAMES

import math
import json
import re
from app.settings import I2V_CHUNK_BASE_FRAMES, I2V_OVERLAP_FRAMES, I2V_PAD_TAIL_FRAMES

TraceFn = Callable[[str, str], None]

def _t(trace: TraceFn|None, tag: str, msg: str) -> None:
    if trace:
        try:
            trace(tag, msg)
        except Exception:
            pass


# shopping 탭 json
def fill_prompt_movie_with_ai_shopping(
        story_data: dict,
        ai_ask_func: Callable[[str, str], str],
        trace: TraceFn | None = None
) -> dict:
    """
    [Step 6] Long-Take Shopping 스타일 (완전 동기화 버전):
    1. 무조건 한글 'prompt'를 원본으로 사용 (기존 prompt_img 무시).
    2. 인물 키워드가 없으면 캐릭터 정보(앵커)를 물리적으로 차단.
    3. [NEW] 생성된 클린 프롬프트(seg1)로 기존 'prompt_img'를 강제 덮어쓰기 -> 불일치 해결.
    """
    import math
    import json
    import re
    # 상수 임포트
    try:
        from app.settings import I2V_CHUNK_BASE_FRAMES, I2V_OVERLAP_FRAMES, I2V_PAD_TAIL_FRAMES
    except ImportError:
        I2V_CHUNK_BASE_FRAMES = 81
        I2V_OVERLAP_FRAMES = 10
        I2V_PAD_TAIL_FRAMES = 20

    # 1. FPS 로드
    defaults = story_data.get("defaults", {})
    movie_opts = defaults.get("movie", {})

    fps = 16.0
    if "fps" in movie_opts:
        try:
            fps = float(movie_opts["fps"])
        except:
            pass
    elif "ui_prefs" in defaults:
        try:
            fps = float(defaults["ui_prefs"].get("movie_fps", 30))
        except:
            pass

    # 원본 캐릭터 정보
    origin_char_info = story_data.get("meta", {}).get("character_prompt", "")
    if not origin_char_info:
        origin_char_info = story_data.get("global_context", {}).get("character_desc", "A professional model")

    # 인물 감지 키워드
    PERSON_KEYWORDS = ["앵커", "아나운서", "사람", "여자", "남자", "여성", "남성", "모델", "인물", "얼굴", "표정", "제스처", "바라보", "설명", "말하"]

    base_chunk = I2V_CHUNK_BASE_FRAMES
    overlap = I2V_OVERLAP_FRAMES
    pad_tail = I2V_PAD_TAIL_FRAMES

    scenes = story_data.get("scenes", [])
    if not scenes:
        return story_data

    _t(trace, "info", f"🚀 [AI Long-Take] 쇼핑 프롬프트 동기화 시작 (FPS: {fps})")

    for sc in scenes:
        sid = sc.get("id")

        # 1) 시간 및 프레임 계산
        try:
            duration = float(sc.get("duration") or 2.0)
            total_frames = int(duration * fps)
        except:
            total_frames = 60
            duration = 4.0

        # 2) 세그먼트 개수 계산
        step = base_chunk - overlap
        target = total_frames + pad_tail

        if target <= base_chunk:
            seg_count = 1
        else:
            needed = target - base_chunk
            additional = math.ceil(needed / step)
            seg_count = 1 + additional

        sc["total_frames"] = total_frames
        sc["seg_count"] = seg_count
        sc["movie_duration"] = duration

        # 3) 한글 prompt 원본 사용 (prompt_img 무시)
        korean_prompt = str(sc.get("prompt", "")).strip()
        if not korean_prompt:
            korean_prompt = "Product shot"

        # 4) 인물 키워드 검사 -> Mode 결정
        has_person = any(kw in korean_prompt for kw in PERSON_KEYWORDS)

        if has_person:
            current_char_info = origin_char_info
            subject_mode = "PERSON"
        else:
            current_char_info = ""  # 캐릭터 정보 삭제 (앵커 차단)
            subject_mode = "OBJECT"

        _t(trace, "info", f"   - [{sid}] '{korean_prompt[:10]}...' -> Mode: {subject_mode}")

        lyric_context = sc.get("lyric", "")

        # 시스템 프롬프트
        sys_msg = (
            "You are an Elite AI Video Director. Transform the Korean 'base_prompt' into a vivid English video description.\n"
            f"Target segments: {seg_count}\n"
            "\n"
            "RULES:\n"
            "1. **Source of Truth**: Use the Korean 'base_prompt' as the ONLY guide.\n"
            "2. **Subject Consistency**:\n"
            f"   - Detected Mode: **{subject_mode}**\n"
            "   - If OBJECT mode: DO NOT mention people/anchors. Show ONLY the object/place.\n"
            "   - If PERSON mode: Integrate 'Character Info'.\n"
            "3. **Format**: Single descriptive English sentence per segment.\n"
            "\n"
            "OUTPUT JSON ONLY:\n"
            "{\n"
            "  \"prompts_en\": [\"Seg1 English...\", \"Seg2 English...\"],\n"
            "  \"prompts_kor\": [\"Seg1 한글...\", \"Seg2 한글...\"]\n"
            "}"
        )

        user_msg = json.dumps({
            "scene_id": sid,
            "base_prompt": korean_prompt,
            "lyric_context": lyric_context,
            "character_info": current_char_info,
            "segment_count_needed": seg_count,
            "duration_sec": duration
        }, ensure_ascii=False)

        try:
            # AI 호출
            raw = ai_ask_func(sys_msg, user_msg)

            # 파싱
            text_cleaned = raw.strip()
            if "```" in text_cleaned:
                text_cleaned = re.sub(r"```json|```", "", text_cleaned).strip()

            idx_start = text_cleaned.find("{")
            idx_end = text_cleaned.rfind("}")

            if idx_start != -1 and idx_end != -1:
                parsed = json.loads(text_cleaned[idx_start: idx_end + 1])

                p_en = parsed.get("prompts_en", [])
                p_kor = parsed.get("prompts_kor", [])

                if isinstance(p_en, str): p_en = [p_en]
                if isinstance(p_kor, str): p_kor = [p_kor]

                # 부족하면 채우기
                while len(p_en) < seg_count:
                    p_en.append(p_en[-1] if p_en else "Cinematic shot")
                while len(p_kor) < seg_count:
                    p_kor.append(p_kor[-1] if p_kor else "영화 같은 샷")

                # 저장
                for i in range(seg_count):
                    sc[f"prompt_{i + 1}"] = str(p_en[i]).strip()
                    sc[f"prompt_{i + 1}_kor"] = str(p_kor[i]).strip()

                # [★핵심 수정] prompt_img도 첫 번째 세그먼트로 강제 동기화 (앵커 제거)
                if p_en:
                    sc["prompt_img"] = str(p_en[0]).strip()

            else:
                raise ValueError("JSON braces not found")

        except Exception as e:
            _t(trace, "warn", f"❌ [{sid}] AI Prompt Error: {e}")
            # Fallback
            fallback_txt = "Cinematic shot of " + korean_prompt
            for i in range(seg_count):
                sc[f"prompt_{i + 1}"] = fallback_txt
                sc[f"prompt_{i + 1}_kor"] = korean_prompt
            # Fallback 시에도 prompt_img 동기화
            sc["prompt_img"] = fallback_txt

    _t(trace, "info", "✅ [AI Long-Take] 상세화 및 prompt_img 동기화 완료")
    return story_data


# shorts 탭 json
def fill_prompt_movie_with_ai_shorts(
    video_data: dict,
    ask: Callable[[str, str], str],
    trace: TraceFn | None = None,
) -> dict:
    """
    Shorts 탭용 프롬프트/세그먼트 엔진.

    - 입력: video.json dict
    - 출력: 같은 dict에 아래 필드를 주입/갱신
        * 각 scene:
            - total_frames
            - seg_count
            - movie_duration
            - prompt_1 ~ prompt_N
            - prompt_1_kor ~ prompt_N_kor
    - 세그먼트 규칙:
        * BASE_CHUNK = 81 프레임
        * OVERLAP   = 10 프레임
        * fps = defaults.movie.fps (없으면 30)
    """

    defaults = video_data.get("defaults") or {}
    movie_opts = defaults.get("movie") or {}

    try:
        fps = float(movie_opts.get("fps") or 30.0)
    except Exception:
        fps = 30.0

    BASE_CHUNK = 81
    OVERLAP = 10

    scenes = video_data.get("scenes") or []

    _t(trace, "shorts", f"[Shorts] 세그먼트 프롬프트 생성 시작 (FPS:{fps})")

    for sc in scenes:
        sid = sc.get("id")
        if not sid:
            continue

        # ─────────────────────────────────────
        # 1. 시간 / 세그먼트 계산 (기존 규칙 유지)
        # ─────────────────────────────────────
        try:
            start_t = float(sc.get("start", 0.0))
            end_t = float(sc.get("end", 0.0))
            duration = max(0.1, end_t - start_t)
        except Exception:
            duration = 4.0

        total_frames = int(math.ceil(duration * fps))

        if total_frames <= BASE_CHUNK:
            seg_count = 1
        else:
            step = BASE_CHUNK - OVERLAP
            seg_count = 1 + math.ceil((total_frames - BASE_CHUNK) / step)

        sc["total_frames"] = total_frames
        sc["seg_count"] = seg_count
        sc["movie_duration"] = duration

        # ─────────────────────────────────────
        # 2. 베이스 컨텍스트 수집 (값만 사용)
        # ─────────────────────────────────────
        base_prompt_img = (sc.get("prompt_img") or "").strip()
        base_prompt_kor = (sc.get("prompt") or "").strip()
        lyric = (sc.get("lyric") or "").strip()
        characters = sc.get("characters") or []

        # [삭제됨] Gap 씬 강제 덮어쓰기 로직 제거 완료
        # (기존 코드가 무조건 특정 문장을 박아넣던 부분 삭제)

        # lyric가 있으면 반드시 캐릭터가 존재한다고 간주
        has_character = bool(characters) or bool(lyric)

        # ─────────────────────────────────────
        # 3. AI 시스템 프롬프트
        # ─────────────────────────────────────
        if has_character:
            sys_msg = (
                "You are a video director creating animated segments from a fixed base image.\n"
                "RULES:\n"
                "- The character already exists in the base image.\n"
                "- Do NOT describe appearance, clothes, background, lighting, or style.\n"
                "- ONLY describe character actions, facial expressions, body movement,\n"
                "  camera movement, and cinematic effects.\n"
                "- Refer to the character naturally (e.g. 'the woman', 'the man', 'the character').\n"
                "- Each prompt must be a complete sentence.\n"
                "OUTPUT JSON ONLY:\n"
                "{"
                "\"prompts_en\":[\"...\"],"
                "\"prompts_kor\":[\"...\"],"
                "\"last_state_en\":\"...\","
                "\"last_state_kor\":\"...\""
                "}\n"
                f"Create exactly {seg_count} prompts."
            )
        else:
            sys_msg = (
                "You are a video director creating animated segments from a fixed base image.\n"
                "RULES:\n"
                "- There is NO character or person in the scene.\n"
                "- Do NOT invent people or human actions.\n"
                "- ONLY describe environmental motion, atmosphere, light changes,\n"
                "  camera movement, and cinematic effects.\n"
                "- Each prompt must be a complete sentence.\n"
                "OUTPUT JSON ONLY:\n"
                "{"
                "\"prompts_en\":[\"...\"],"
                "\"prompts_kor\":[\"...\"],"
                "\"last_state_en\":\"...\","
                "\"last_state_kor\":\"...\""
                "}\n"
                f"Create exactly {seg_count} prompts."
            )

        user_msg = json.dumps(
            {
                "scene_id": sid,
                "base_image_prompt_en": base_prompt_img,
                "base_prompt_kor": base_prompt_kor,
                "lyric_context": lyric,
                "segment_count": seg_count,
            },
            ensure_ascii=False,
        )

        # ─────────────────────────────────────
        # 4. AI 호출
        # ─────────────────────────────────────
        try:
            raw = ask(sys_msg, user_msg)
        except Exception:
            _t(trace, "warn", f"{sid}: AI 호출 실패 → fallback 사용")
            for i in range(seg_count):
                if has_character:
                    sc[f"prompt_{i+1}"] = (
                        "The character makes a subtle movement as the camera gently shifts."
                    )
                    sc[f"prompt_{i+1}_kor"] = (
                        "인물이 미묘한 움직임을 보이며 카메라가 부드럽게 이동한다."
                    )
                else:
                    sc[f"prompt_{i+1}"] = (
                        "The camera slowly moves, adding subtle cinematic motion."
                    )
                    sc[f"prompt_{i+1}_kor"] = (
                        "카메라가 천천히 움직이며 미묘한 시네마틱 효과가 더해진다."
                    )
            continue

        # ─────────────────────────────────────
        # 5. 응답 파싱
        # ─────────────────────────────────────
        try:
            txt = (raw or "").strip()
            txt = txt.replace("```json", "").replace("```", "")
            s = txt.find("{")
            e = txt.rfind("}")
            if s == -1 or e == -1:
                raise ValueError("JSON block not found in response")
            parsed = json.loads(txt[s : e + 1])
        except Exception:
            _t(trace, "warn", f"{sid}: JSON 파싱 실패 → fallback 사용")
            for i in range(seg_count):
                if has_character:
                    sc[f"prompt_{i+1}"] = (
                        "The character makes a subtle movement as the camera gently shifts."
                    )
                    sc[f"prompt_{i+1}_kor"] = (
                        "인물이 미묘한 움직임을 보이며 카메라가 부드럽게 이동한다."
                    )
                else:
                    sc[f"prompt_{i+1}"] = (
                        "The camera slowly moves, adding subtle cinematic motion."
                    )
                    sc[f"prompt_{i+1}_kor"] = (
                        "카메라가 천천히 움직이며 미묘한 시네마틱 효과가 더해진다."
                    )
            continue

        p_en = parsed.get("prompts_en", [])
        p_kor = parsed.get("prompts_kor", [])

        if isinstance(p_en, str):
            p_en = [p_en]
        if isinstance(p_kor, str):
            p_kor = [p_kor]

        if not p_en:
            p_en = [
                (
                    "The camera slowly moves, adding subtle cinematic motion."
                    if not has_character
                    else "The character makes a subtle movement as the camera gently shifts."
                )
            ] * seg_count

        if not p_kor:
            p_kor = [
                (
                    "카메라가 천천히 움직이며 미묘한 시네마틱 효과가 더해진다."
                    if not has_character
                    else "인물이 미묘한 움직임을 보이며 카메라가 부드럽게 이동한다."
                )
            ] * seg_count

        while len(p_en) < seg_count:
            p_en.append(p_en[-1])
        while len(p_kor) < seg_count:
            p_kor.append(p_kor[-1])

        # ★ Shopping 스타일 필드명(prompt_1, prompt_2...)으로 저장
        for i in range(seg_count):
            sc[f"prompt_{i + 1}"] = p_en[i].strip()
            sc[f"prompt_{i + 1}_kor"] = p_kor[i].strip()

        _t(trace, "shorts", f"Scene {sid}: seg_count={seg_count}, total_frames={total_frames}")

    _t(trace, "shorts", "✅ 세그먼트 프롬프트 생성 완료.")
    return video_data


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

    # --- [수정됨] 시스템 프롬프트: 일치성(Consistency) 규칙 강화 ---
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
        "   - **[Consistency Rule]**: The content of 'prompt_img_base' (English) MUST strictly match the meaning of 'prompt' (Korean). **Do not** write a Korean prompt about 'Sky' and an English prompt about 'A woman'.\n"
        "   - **DO NOT** use comma-separated tags. Write a **descriptive, natural English sentence**.\n"
        "   - **[Character Reference Rule]**: The 'characters' field provides 'id:index'.\n"
        "     * **[EMPTY CHARACTERS]**: If the 'characters' list is empty, **DO NOT** mention any person or character. Describe ONLY the background/scenery.\n"
        "     * ':0' -> MUST be referred to as 'from image 1'\n"
        "     * ':1' -> MUST be referred to as 'from image 2'\n"
        "     * ':2' -> 'from image 3', etc.\n"
        "   - **[Content Generation]**:\n"
        "     * Infer the character's gender from their ID (e.g. 'female' -> woman, 'male' -> man).\n"
        "     * **Invent specific outfits and actions** based on the scene mood, lyrics, and hint.\n"
        "     * Combine multiple characters into a coherent sentence.\n"
        "     * If NO characters are listed, describe the atmosphere, lighting, and environment only.\n"
        "   - **[Example Output]**:\n"
        "     * BAD: 'woman, beach, summer, bikini, walking'\n"
        "     * GOOD: 'A woman from image 1, wearing a colorful bikini, is walking along the sunny beach smiling.'\n"
        "     * GOOD: 'A man from image 2 in a black suit is dancing with a woman from image 1 in a red dress under the streetlights.'\n"
        "     * GOOD (No Character): 'A quiet, sunlit park bench surrounded by falling autumn leaves, bathed in warm golden light.'\n"
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


def build_video_json_with_gap_policy(
    project_dir: str,
    *,
    small_gap_sec: float = 2.0,
    filler_section: str = "bridge",
    filler_label: str = "Gap",
    filler_scene: str = "bridge",
) -> str:
    """
    story.json을 읽어 공백 구간을 보정해 video.json을 생성한다.

    규칙:
    - gap < small_gap_sec: 이전 씬을 다음 씬 시작까지 '연장'(새 씬 추가 없음)
    - gap ≥ small_gap_sec: 공백을 갭 씬으로 '삽입'
    - 시작/끝 공백에도 동일 규칙 적용

    요구사항:
    - 갭 씬은 일반 씬과 동일 스키마를 갖되, 프롬프트는 비워서 AI가 전부 구성하도록 맡긴다.
    - 갭 씬 ID는 3자리로 통일: gap_###, 그리고 바로 앞 t_### 번호를 따른다(예: t_007 뒤 gap_007).
    """

    proj_path = Path(project_dir)
    story_path = proj_path / "story.json"
    if not story_path.exists():
        raise FileNotFoundError(f"story.json 파일을 찾을 수 없습니다: {story_path}")

    story_doc = load_json(story_path, {}) or {}
    if not isinstance(story_doc, dict):
        raise TypeError("story.json 형식 오류(dict 아님)")

    scenes_in = list(story_doc.get("scenes") or [])
    if not scenes_in:
        raise ValueError("story.json에 scenes가 없습니다.")

    defaults = story_doc.get("defaults") or {}
    default_img = defaults.get("image") or {}
    default_negative = str(default_img.get("negative") or "")

    imgs_dir = proj_path / "imgs"
    try:
        imgs_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    # ✅ 타입체커가 허용하는 입력 타입으로 정확히 명시
    def _as_float(v: str | bytes | bytearray | SupportsFloat | SupportsIndex | None) -> float:
        if v is None:
            return 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    # 시간 정렬 및 정규화
    tmp_scenes: list[dict] = []
    for it in scenes_in:
        if not isinstance(it, dict):
            continue
        s = _as_float(it.get("start"))
        e = _as_float(it.get("end"))
        if e < s:
            e = s
        item = dict(it)
        item["start"] = s
        item["end"] = e
        item["duration"] = round(max(0.0, e - s), 3)
        tmp_scenes.append(item)
    tmp_scenes.sort(key=lambda x: float(x.get("start", 0.0)))

    total_duration = _as_float(story_doc.get("duration"))
    if total_duration <= 0.0 and tmp_scenes:
        total_duration = float(tmp_scenes[-1].get("end", 0.0))
    offset_val = _as_float(story_doc.get("offset"))

    out_scenes: list[dict] = []

    # 직전 t_### 번호를 기억 → gap은 같은 번호 사용
    last_t_num = 0
    rx_tnum = re.compile(r"^t_(\d{3})$")

    def _copy_scene_for_video(src: dict) -> dict:
        nonlocal last_t_num
        sc_id = str(src.get("id") or "")
        m = rx_tnum.match(sc_id)
        if m:
            try:
                last_t_num = int(m.group(1))
            except ValueError:
                last_t_num = last_t_num
        return {
            "id": src.get("id"),
            "section": src.get("section"),
            "label": src.get("label"),
            "start": float(src.get("start", 0.0) or 0.0),
            "end": float(src.get("end", 0.0) or 0.0),
            "duration": round(max(0.0, float(src.get("end", 0.0) or 0.0) - float(src.get("start", 0.0) or 0.0)), 3),
            "scene": src.get("scene"),
            "characters": list(src.get("characters") or []),
            "effect": list(src.get("effect") or []),
            "screen_transition": bool(src.get("screen_transition", False)),
            "img_file": str(src.get("img_file") or ""),
            "prompt": str(src.get("prompt") or ""),
            "prompt_img": str(src.get("prompt_img") or ""),
            "prompt_movie": str(src.get("prompt_movie") or ""),
            "prompt_negative": str(src.get("prompt_negative") or default_negative),
            "lyric": str(src.get("lyric") or ""),
        }

    def _mk_gap(start: float, end: float) -> dict:
        sc_id = f"gap_{last_t_num:03d}"  # 직전 t_### 번호를 따른다. 시작부면 000
        return {
            "id": sc_id,
            "section": filler_section,
            "label": filler_label,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(max(0.0, end - start), 3),
            "scene": filler_scene,
            "characters": [],
            "effect": ["dissolve"],
            "screen_transition": True,
            "img_file": str((imgs_dir / f"{sc_id}.png").resolve()),
            "prompt": "",
            "prompt_img": "",
            "prompt_movie": "",
            "prompt_negative": "",
            "lyric": "",
            "origin": "gap-fill",
        }

    # 시작부 공백
    if tmp_scenes:
        first_id = str(tmp_scenes[0].get("id") or "")
        m0 = rx_tnum.match(first_id)
        if m0:
            try:
                last_t_num = int(m0.group(1)) - 1  # 첫 씬 앞 갭은 gap_(첫t-1) → gap_000부터 가능
            except ValueError:
                last_t_num = 0
        else:
            last_t_num = 0

        first_start = float(tmp_scenes[0].get("start", 0.0))
        head_gap = round(max(0.0, first_start - offset_val), 3)
        if head_gap > 0.0:
            if head_gap < small_gap_sec:
                sc0 = _copy_scene_for_video(tmp_scenes[0])
                sc0["start"] = offset_val
                sc0["duration"] = round(max(0.0, sc0["end"] - sc0["start"]), 3)
                out_scenes.append(sc0)
            else:
                out_scenes.append(_mk_gap(offset_val, first_start))
                out_scenes.append(_copy_scene_for_video(tmp_scenes[0]))
        else:
            out_scenes.append(_copy_scene_for_video(tmp_scenes[0]))

    # 본문 공백
    for i in range(len(tmp_scenes) - 1):
        cur = tmp_scenes[i]
        nxt = tmp_scenes[i + 1]
        cur_end = float(cur.get("end", 0.0))
        nxt_start = float(nxt.get("start", 0.0))
        gap = round(max(0.0, nxt_start - cur_end), 3)

        if gap <= 0.0:
            out_scenes.append(_copy_scene_for_video(nxt))
            continue

        if gap < small_gap_sec:
            last = out_scenes[-1]
            last["end"] = nxt_start
            last["duration"] = round(max(0.0, last["end"] - last["start"]), 3)
            out_scenes.append(_copy_scene_for_video(nxt))
        else:
            out_scenes.append(_mk_gap(cur_end, nxt_start))
            out_scenes.append(_copy_scene_for_video(nxt))

    # 끝부분 공백
    if tmp_scenes:
        last_end = float(tmp_scenes[-1].get("end", 0.0))
        if total_duration > 0.0:
            tail_gap = round(max(0.0, total_duration - last_end), 3)
            if tail_gap > 0.0:
                if tail_gap < small_gap_sec:
                    if out_scenes:
                        out_scenes[-1]["end"] = total_duration
                        out_scenes[-1]["duration"] = round(max(0.0, out_scenes[-1]["end"] - out_scenes[-1]["start"]), 3)
                else:
                    out_scenes.append(_mk_gap(last_end, total_duration))

    video_obj = dict(story_doc)
    video_obj["scenes"] = out_scenes
    video_obj.setdefault("audit", {})
    video_obj["audit"]["gap_policy"] = {
        "applied": True,
        "small_gap_sec": float(small_gap_sec),
        "source": str(story_path),
        "id_width": 3,
        "note": "gaps inserted empty so AI can fully author them; gap id follows previous t id",
    }

    video_path = story_path.parent / "video.json"
    save_json(video_path, video_obj)
    return str(video_path)




