# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime
import json
import re
import shutil
import os
import random
import requests
import subprocess
import time
import uuid
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from pydub import AudioSegment

from app.utils import (
    AI,
    load_json,
    save_json,
    ensure_dir,
    _submit_and_wait as submit_and_wait,
    get_duration,

)
from app import settings
from app.video_build import build_shots_with_i2v, concatenate_scene_clips_final_av
from app.video_build import build_step1_zimage_base, build_step2_qwen_composite
from app.story_enrich import fill_prompt_movie_with_ai_shopping
def _now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")



# -----------------------------------------------------------------------------
# 1. Zonos TTS 생성 함수
# -----------------------------------------------------------------------------
def _get_zonos_config(scene: Dict[str, Any], ai: AI = None) -> Dict[str, Any]:
    """
    씬 정보에서 Zonos용 설정(emotion, speed)을 가져오거나,
    기존 텍스트 포맷([톤] 내용 [효과음])일 경우 AI로 분석해 변환(Migration)한다.
    """
    # 1. 이미 설정값이 있으면 반환
    if "voice_config" in scene:
        return scene["voice_config"]

    # 2. 설정값이 없으면 내레이션 파싱 시도
    raw_narr = scene.get("narration", "").strip()
    if not raw_narr:
        return {"speed": 1.0, "emotion": {"neutral": 1.0}}

    # 정규식으로 [톤] 내용 [효과음] 분리
    tone_match = re.match(r"^\[(.*?)\]\s*(.*)", raw_narr)

    tone_text = "calm"
    clean_text = raw_narr
    sfx_text = ""

    if tone_match:
        tone_text = tone_match.group(1).strip()
        remain = tone_match.group(2).strip()

        # 뒤쪽 SFX 체크
        sfx_match = re.search(r"\s*\[(.*?)\]$", remain)
        if sfx_match:
            sfx_text = sfx_match.group(1).strip()
            clean_text = remain[:sfx_match.start()].strip()
        else:
            clean_text = remain

        # 따옴표 제거
        clean_text = clean_text.strip("'").strip('"')

    # AI에게 수치 변환 요청 (Migration)
    if ai:
        try:
            sys_p = (
                "You are a Voice Director. Analyze the 'Tone Description' and convert it into Zonos TTS parameters.\n"
                "Output JSON only: { \"speed\": float(0.8-1.5), \"emotion\": { neutral, happy, sad, disgust, fear, surprise, anger, other } (sum approx 1.0) }"
            )
            user_p = f"Tone Description: \"{tone_text}\""
            res = ai.ask_smart(sys_p, user_p, prefer="openai")

            # JSON 파싱
            if "```" in res: res = res.split("```")[1].replace("json", "")
            config = json.loads(res[res.find("{"):res.rfind("}") + 1])

            # 씬 데이터 업데이트 (영구 저장용)
            scene["narration"] = clean_text
            scene["voice_config"] = config
            if sfx_text:
                scene["sfx"] = sfx_text

            return config
        except Exception as e:
            print(f"⚠️ 톤 분석 실패: {e}")

    # 실패/AI 없으면 기본값
    return {"speed": 1.0, "emotion": {"neutral": 1.0}}


# [shopping_video_build.py]
# generate_tts_zonos 함수 전체를 아래 코드로 교체하세요.

def generate_tts_zonos(
        text: str,
        out_path: Path,
        ref_audio: Path,
        comfy_host: str = "http://127.0.0.1:8188",
        config: Dict[str, Any] = None
) -> bool:
    if not text:
        return False

    # 1. Zonos 초기 잡음 방지용 '..' 자동 추가
    # (앞에 점을 찍으면 호흡음이 줄어드는 효과가 있지만, 완벽하지 않아 트리밍도 병행합니다)
    tts_prompt_text = text
    if not tts_prompt_text.strip().startswith("."):
        tts_prompt_text = ".." + tts_prompt_text

    wf_path = Path(settings.JSONS_DIR) / "who_voice.json"
    if not wf_path.exists():
        wf_path = Path(r"C:\my_games\shorts_make\app\jsons\who_voice.json")

    if not wf_path.exists():
        print(f"❌ TTS 워크플로우 없음: {wf_path}")
        return False

    try:
        with open(wf_path, "r", encoding="utf-8") as f:
            graph = json.load(f)
    except Exception as e:
        print(f"❌ 워크플로우 로드 실패: {e}")
        return False

    if not ref_audio.exists():
        print(f"❌ 참조 오디오 없음: {ref_audio}")
        return False

    comfy_input_dir = Path(settings.COMFY_INPUT_DIR)
    comfy_input_dir.mkdir(parents=True, exist_ok=True)

    ref_copy_name = f"ref_{uuid.uuid4().hex[:8]}{ref_audio.suffix}"
    dst_ref = comfy_input_dir / ref_copy_name
    shutil.copy2(ref_audio, dst_ref)

    # 2. 노드 값 설정 (텍스트, 시드, 속도)
    found_gen = False
    for nid, node in graph.items():
        if "Zonos" in node.get("class_type", "") and "speech" in node.get("inputs", {}):
            node["inputs"]["speech"] = tts_prompt_text
            node["inputs"]["seed"] = random.randint(1, 2 ** 32)
            if config and "speed" in config:
                node["inputs"]["speed"] = config["speed"]
            found_gen = True
            break

    if not found_gen and "24" in graph:
        graph["24"]["inputs"]["speech"] = tts_prompt_text
        graph["24"]["inputs"]["seed"] = random.randint(1, 2 ** 32)
        if config and "speed" in config:
            graph["24"]["inputs"]["speed"] = config["speed"]

    # 감정 설정
    if config and "emotion" in config:
        emotions = config["emotion"]
        for nid, node in graph.items():
            if node.get("class_type") == "Zonos Emotion":
                for k, v in emotions.items():
                    if k in node["inputs"]:
                        node["inputs"][k] = v
                break

    # 참조 오디오 설정
    found_audio = False
    for nid, node in graph.items():
        if node.get("class_type") == "LoadAudio":
            node["inputs"]["audio"] = ref_copy_name
            found_audio = True
            break

    if not found_audio and "12" in graph:
        graph["12"]["inputs"]["audio"] = ref_copy_name

    try:
        res = submit_and_wait(comfy_host, graph, timeout=300)
        outputs = res.get("outputs", {})
        for nid, out_d in outputs.items():
            if "audio" in out_d:
                for item in out_d["audio"]:
                    fname = item["filename"]
                    params = {"filename": fname, "subfolder": item.get("subfolder", ""),
                              "type": item.get("type", "output")}
                    resp = requests.get(f"{comfy_host}/view", params=params)

                    if resp.status_code == 200:
                        ensure_dir(out_path.parent)
                        # 일단 원본 저장
                        with open(out_path, "wb") as f:
                            f.write(resp.content)

                        # [New] 앞부분 0.2초(200ms) 트리밍 로직
                        try:
                            audio = AudioSegment.from_file(str(out_path))
                            # 길이가 충분할 때만 자름
                            if len(audio) > 300:  # 최소 0.3초는 되어야 0.2초를 자름
                                trimmed = audio[350:]  # 200ms 부터 끝까지
                                trimmed.export(str(out_path), format="wav")
                                # print(f"✂️ Audio trimmed 0.2s: {out_path.name}")
                        except Exception as e:
                            print(f"⚠️ 오디오 트리밍 실패 (원본 유지): {e}")

                        return True
        return False
    except Exception as e:
        print(f"TTS 생성 실패: {e}")
        return False



# -----------------------------------------------------------------------------
# 1.5 BGM 생성 함수 (Ace-Step) - [New]
# -----------------------------------------------------------------------------
def generate_bgm_acestep(
        prompt: str,
        out_path: Path,
        duration_sec: float,
        comfy_host: str = "http://127.0.0.1:8188",
        negative_prompt: str = "vocal, vocals, singing, human voice, lyrics, rap, speech, chant, noise, distortion"
) -> bool:
    """
    [쇼핑 전용] Ace-Step 워크플로우를 사용하여 배경음(BGM) 생성
    - 가사 생성 방지(lyrics_strength=0) 및 Instrumental 강제
    """
    if not prompt:
        return False

    # 워크플로우 로드
    wf_path = Path(settings.JSONS_DIR) / "ace_step_1_t2mm.json"
    if not wf_path.exists():
        # 기본 경로 폴백
        wf_path = Path(r"C:\my_games\shorts_make\app\jsons\ace_step_1_t2mm.json")

    if not wf_path.exists():
        print(f"❌ Ace-Step 워크플로우 없음: {wf_path}")
        return False

    try:
        with open(wf_path, "r", encoding="utf-8") as f:
            graph = json.load(f)
    except Exception as e:
        print(f"❌ 워크플로우 로드 실패: {e}")
        return False

    # 노드 매핑 및 값 주입
    # 14: TextEncodeAceStepAudio (tags, lyrics_strength)
    # 80: CLIPTextEncode (Negative)
    # 17: EmptyAceStepLatentAudio (seconds)
    # 52: KSampler (seed)
    # 78: SaveAudioMP3 (filename)

    # 1. 긍정 프롬프트 & 가사 억제
    if "14" in graph:
        graph["14"]["inputs"]["tags"] = prompt
        graph["14"]["inputs"]["lyrics_strength"] = 0  # 가사 방지 핵심
        # lyrics 입력 끊기 (안전장치) -> 빈 문자열을 주는 노드가 있다면 연결, 아니면 텍스트 비우기
        # 여기서는 tags에만 의존하고 lyrics_strength=0으로 제어

    # 2. 부정 프롬프트 (보컬 방지)
    if "80" in graph:
        graph["80"]["inputs"]["text"] = negative_prompt

    # 3. 길이 설정 (여유분 3초 추가)
    target_sec = math.ceil(duration_sec + 3.0)
    if "17" in graph:
        graph["17"]["inputs"]["seconds"] = target_sec

    # 4. 시드 랜덤화
    if "52" in graph:
        graph["52"]["inputs"]["seed"] = random.randint(1, 2 ** 32)

    # 5. 저장 경로 설정
    # ComfyUI output 폴더 기준 상대 경로
    # 예: "bgm/product_name_bgm" (확장자는 SaveAudio 노드가 붙임)
    file_prefix = f"bgm/shopping_bgm_{uuid.uuid4().hex[:6]}"
    if "78" in graph:
        graph["78"]["inputs"]["filename_prefix"] = file_prefix

    # 실행
    try:
        res = submit_and_wait(comfy_host, graph, timeout=600)  # 오디오 생성은 좀 걸릴 수 있음

        # 결과 파일 다운로드 (move to out_path)
        outputs = res.get("outputs", {})
        for nid, out_d in outputs.items():
            if "audio" in out_d:
                for item in out_d["audio"]:
                    fname = item["filename"]
                    # ComfyUI output 폴더에서 가져오기
                    params = {"filename": fname, "subfolder": item.get("subfolder", ""),
                              "type": item.get("type", "output")}
                    resp = requests.get(f"{comfy_host}/view", params=params)
                    if resp.status_code == 200:
                        ensure_dir(out_path.parent)
                        with open(out_path, "wb") as f:
                            f.write(resp.content)
                        return True
        return False

    except Exception as e:
        print(f"❌ BGM 생성 실패: {e}")
        return False


# -----------------------------------------------------------------------------
# 2. 이미지 생성 함수 (2-Step: Z-Image -> Qwen)
# -----------------------------------------------------------------------------


def build_shopping_images_2step(
        video_json_path: str | object,
        *,
        source_json_path: str | object | None = None,
        product_image_path: str | object | None = None,
        ui_width: int | None = 720,
        ui_height: int | None = 1280,
        steps: int | None = 20,
        skip_if_exists: bool = True,
        on_progress: object = None,
) -> None:
    """
    쇼핑 2단계 이미지 생성 (정상 동작 버전)
    - 기준 JSON: video_shopping.json
    - Step1: prompt_img_1 기반 Z-Image → imgs/temp_{sid}.png
    - Step2: prompt_img_2 기반 QwenEdit 합성
        * slot1 (image1) = temp_{sid}.png (배경/인물)
        * slot2 (image2) = product_image_path (제품)
    """

    def _emit(msg: str) -> None:
        if not on_progress:
            return
        try:
            if callable(on_progress):
                on_progress({"stage": "debug", "msg": msg})
            elif isinstance(on_progress, dict) and callable(on_progress.get("callback")):
                on_progress["callback"]({"stage": "debug", "msg": msg})
        except Exception:
            pass

    if not video_json_path:
        _emit("[Image][ERR] video_json_path is None")
        return

    vpath = Path(str(video_json_path)).resolve()
    if not vpath.exists():
        _emit(f"[Image][ERR] video_shopping.json 없음: {vpath}")
        return

    if source_json_path is None:
        source_json_path = vpath

    proj_dir = vpath.parent
    imgs_dir = proj_dir / "imgs"

    doc = load_json(vpath, {}) or {}
    scenes = doc.get("scenes", []) or []

    prod_path = None
    if product_image_path:
        prod_path = Path(str(product_image_path)).resolve()
    _emit(f"[Image][DBG] product_image_path={str(prod_path) if prod_path else None} exists={prod_path.exists() if prod_path else False}")

    # -------------------------
    # Step1: Z-Image
    # -------------------------
    _emit(f"[Image] Step1(Z-Image) 시작 ({ui_width}x{ui_height}, steps={steps})")
    try:
        build_step1_zimage_base(
            video_json_path=vpath,
            source_json_path=source_json_path,
            ui_width=ui_width,
            ui_height=ui_height,
            steps=steps,
            skip_if_exists=skip_if_exists,
            on_progress=on_progress,
        )
    except Exception as e:
        _emit(f"[Image][WARN] Step1 실패(계속 진행): {e}")

    src_map = {}
    try:
        p_src = Path(str(source_json_path)).resolve() if source_json_path else None
        if p_src and p_src.exists():
            sdoc = load_json(p_src, {}) or {}
            for s in (sdoc.get("scenes", []) or []):
                sid = str(s.get("id", "")).strip()
                if sid:
                    src_map[sid] = s
    except Exception as e:
        _emit(f"[Image][WARN] source_json_path 매핑 실패: {e}")

    # -------------------------
    # Step2: Qwen Composite
    # -------------------------
    _emit("[Image] Step2(Qwen Composite) 시작")

    for sc in scenes:
        sid = str(sc.get("id", "")).strip()
        if not sid:
            continue

        src_sc = src_map.get(sid, sc)

        p_edit = ""
        for k in ["prompt_img_2", "prompt_edit", "prompt"]:
            val = src_sc.get(k)
            if isinstance(val, str) and val.strip():
                p_edit = val.strip()
                break

        # [수정] 사용자가 요청한 대로 slot1, slot2로 명확히 구분
        # slot1 = Z-Image 결과 (배경/인물) -> ComfyUI Load Image 1에 매핑됨 (리스트 인덱스 0)
        step1_img_path = imgs_dir / f"temp_{sid}.png"
        slot1 = str(step1_img_path) if step1_img_path.exists() else None

        # slot2 = 제품 이미지 -> ComfyUI Load Image 2에 매핑됨 (리스트 인덱스 1)
        slot2 = str(prod_path) if (prod_path and prod_path.exists()) else None

        _emit(f"[Image][DBG] sid={sid}")
        _emit(f"[Image][DBG]  slot1(z-image/image1)={slot1} exists={bool(slot1 and Path(slot1).exists())}")
        _emit(f"[Image][DBG]  slot2(product/image2)={slot2} exists={bool(slot2 and Path(slot2).exists())}")
        _emit(f"[Image][DBG]  prompt_img_2(actual)={p_edit}")

        try:
            build_step2_qwen_composite(
                video_json_path=vpath,
                source_json_path=source_json_path,
                workflow_path=None,
                ui_width=int(ui_width or 720),
                ui_height=int(ui_height or 1280),
                steps=int(steps or 20),
                edit_keys=["prompt_img_2", "prompt_edit", "prompt"],
                skip_if_exists=skip_if_exists,
                on_progress=on_progress,
                # [수정] slot1(Z-Image)을 첫 번째, slot2(제품)를 두 번째로 전달
                slot_images=[slot1, slot2],
                # [중요] 현재 처리 중인 씬 ID만 지정하여 해당 씬만 합성하도록 제한
                target_scene_ids=[sid]
            )
        except Exception as e:
            _emit(f"[Image][ERR] sid={sid} Step2 실패: {e}")
            continue

    _emit("[Image] 쇼핑 이미지 생성 완료")









# -----------------------------------------------------------------------------
# 3. 옵션 데이터 클래스
# -----------------------------------------------------------------------------
@dataclass
class BuildOptions:
    scene_count: int = 6
    style: str = "news_hook"
    hook_level: int = 3
    fps: int = 24
    allow_fallback_rule: bool = True


# -----------------------------------------------------------------------------
# 4. JSON 빌더
# -----------------------------------------------------------------------------

# ai 상세화
class ShoppingVideoJsonBuilder:
    def __init__(self, on_progress: Optional[Callable[[str], None]] = None):
        self.on_progress = on_progress or (lambda msg: None)
        self.ai = AI()

    def create_draft(self, product_dir: str | Path, product_data: Dict[str, Any], options: BuildOptions) -> Path:
        """
        [1단계] 기획 초안 생성
        - 초안 생성 단계에서 prompt_img_1/2 + 한글 버전(prompt_img_1_kor/2_kor)까지 같이 생성
        - ID는 t_001 포맷으로 고정
        """
        p_dir = Path(product_dir)
        vpath = p_dir / "video_shopping.json"

        product_name = product_data.get("product_name", "상품명 없음")
        desc = product_data.get("description") or product_data.get("summary_source") or ""

        self.on_progress(f"[Draft] AI 기획 시작 (상품: {product_name})...")

        # BGM 가이드
        bgm_guide = (
            "4. **Background Music (BGM)**: \n"
            "   - Design a prompt for audio generation.\n"
            "   - Format: 'instrumental, background music, [Mood], [Genre], [Instruments], [Tempo], [Energy]'.\n"
            "   - **NO Vocals**: Do not use words like song, singing, voice.\n"
            "   - **NO Model Name**: Do NOT include the word 'Ace-Step' in the output prompt.\n"
            "   - Example: 'instrumental, background music, bright, acoustic pop, guitar, piano, medium tempo, uplifting'."
        )

        # 시각적 제약사항
        visual_rules = (
            "5. **Visual Description Rules (Clean Prompt)**:\n"
            "   - Focus ONLY on the **Situation, Action, and Background**.\n"
            "   - **FORBIDDEN**: Do NOT describe specific product details like 'Logo', 'Text', 'QR Code', 'Specific Color', 'Label'.\n"
            "   - Bad: 'Product with a red logo spinning.' -> Good: 'The product spinning on the table.'\n"
            "   - Reason: The actual product image will be composited later, so text descriptions of details cause hallucinations."
        )

        # 이미지 프롬프트 생성 규칙 (초안 단계에서 바로 생성)
        img_prompt_rules = (
            "6. **Image Prompt Rules (Two-Language Output)**:\n"
            "   - You MUST output BOTH Korean and English versions.\n"
            "   - `prompt_img_1_kor`: 장면의 인물/배경/상황을 한국어로 간단히 묘사. (제품의 로고/텍스트/라벨/색상 같은 디테일 금지)\n"
            "   - `prompt_img_1`: 위 `prompt_img_1_kor`를 자연스러운 영어로 번역한 문장.\n"
            "   - `prompt_img_2_kor`: 합성(제품 끼워넣기) 단계 지시를 한국어로 간단히 작성.\n"
            "   - `prompt_img_2`: 반드시 아래 영어 고정 문장 구조를 지켜 작성:\n"
            "       \"[Subject] from image 1 [action] the object from image 2\"\n"
            "     예: \"The woman from image 1 holds the object from image 2 in her hand.\"\n"
            "   - `prompt_img_2`에서는 object를 절대 구체적으로 묘사하지 말 것(색/라벨/텍스트/로고 금지).\n"
        )

        system_prompt = (
            "당신은 AI 영상 생성(I2V)을 위한 숏폼 기획 전문가이자 음악 감독입니다. "
            "상품을 분석하여 기획안을 작성하세요.\n\n"
            "**[필수 시각화 규칙]**\n"
            "1. **One Scene = One Action**: 한 장면에는 오직 '하나의 동작'만 묘사.\n"
            "2. **No Split Screens**: 전체 화면 구성.\n"
            "3. **Focus on Impact**: 결정적 순간 포착.\n"
            f"{bgm_guide}\n"
            f"{visual_rules}\n"
            f"{img_prompt_rules}\n\n"
            "중요: 출력은 반드시 JSON만. 코드블록/설명문 금지."
        )

        user_prompt = f"""
        [상품 정보]
        - 상품명: {product_name}
        - 설명: {desc}

        [제작 가이드]
        1. 총 장면: {options.scene_count}개
        2. 스타일: {options.style}

        [출력 포맷 (JSON)]
        {{
            "meta": {{
                "title": "...",
                "voice_gender": "female",
                "character_prompt": "...",
                "bgm_prompt": "instrumental, background music, ... (English Only)"
            }},
            "scenes": [
                {{
                    "id": "t_001",
                    "banner": "...",
                    "prompt": "화면 묘사 (한글, 절대 시퀀스/단계 나열 금지, 단일 동작 위주, 로고/텍스트 묘사 금지)",
                    "narration": "실제 읽을 대사 (지시문 제외)",
                    "sfx": "효과음",
                    "voice_config": {{
                        "speed": 1.0,
                        "emotion": {{ "neutral": 1.0, "happy": 0.0, "sad": 0.0, "disgust": 0.0, "fear": 0.0, "surprise": 0.0, "anger": 0.0, "other": 0.0 }}
                    }},
                    "subtitle": "...",

                    "prompt_img_1_kor": "장면의 인물/배경/상황 (한글)",
                    "prompt_img_2_kor": "합성 단계 지시 (한글)",
                    "prompt_img_1": "English translation of prompt_img_1_kor",
                    "prompt_img_2": "MUST follow: \\"[Subject] from image 1 [action] the object from image 2\\"",

                    "prompt_movie": "Simple camera movement in English",
                    "prompt_negative": "negative prompt in English (short)"
                }},
                ...
            ]
        }}
        """

        try:
            resp_text = self.ai.ask_smart(system_prompt, user_prompt, prefer="openai")
            data = self._safe_json_parse(resp_text)
        except Exception as e:
            self.on_progress(f"❌ 초안 생성 실패: {e}")
            raise

        final_json = {
            "schema": "shopping_shorts_v2",
            "style": options.style,
            "product": product_data,
            "meta": data.get("meta", {}),
            "defaults": {"image": {"width": 720, "height": 1280}, "movie": {"fps": options.fps}},
            "audit": {"created_at": _now_str(), "step": "draft"},
            "scenes": []
        }

        imgs_dir = p_dir / "imgs"
        clips_dir = p_dir / "clips"
        voice_dir = p_dir / "voice"
        ensure_dir(imgs_dir)
        ensure_dir(clips_dir)
        ensure_dir(voice_dir)

        for idx, sc in enumerate(data.get("scenes", [])):
            sid = f"t_{idx + 1:03d}"

            p1_kor = (sc.get("prompt_img_1_kor") or "").strip()
            p2_kor = (sc.get("prompt_img_2_kor") or "").strip()
            p1_eng = (sc.get("prompt_img_1") or "").strip()
            p2_eng = (sc.get("prompt_img_2") or "").strip()

            new_scene = {
                "id": sid,
                "banner": sc.get("banner"),
                "prompt": sc.get("prompt", ""),
                "narration": sc.get("narration", ""),
                "sfx": sc.get("sfx", ""),
                "voice_config": sc.get("voice_config", {"speed": 1.0, "emotion": {"neutral": 1.0}}),
                "subtitle": sc.get("subtitle", ""),
                "seconds": 0,

                # --- 한글/영문 이미지 프롬프트 동시 저장 ---
                "prompt_img_1_kor": p1_kor,
                "prompt_img_2_kor": p2_kor,
                "prompt_img_1": p1_eng,
                "prompt_img_2": p2_eng,

                "prompt_movie": (sc.get("prompt_movie") or ""),
                "prompt_negative": (sc.get("prompt_negative") or ""),

                # 호환용: 기존 로직에서 prompt_img를 참조할 수도 있으니 유지
                "prompt_img": p1_eng,

                # 이미지/영상/보이스 경로
                "img_file": str(imgs_dir / f"{sid}.png"),
                "movie_file": str(clips_dir / f"{sid}.mp4"),
                "voice_file": str(voice_dir / f"{sid}.wav")
            }
            final_json["scenes"].append(new_scene)

        save_json(vpath, final_json)
        self.on_progress("[Draft] 초안 완료. (prompt_img_1/2 + _kor 포함)")
        return vpath

    def enrich_video_json(
            self,
            video_json_path: str | Path,
            product_data: Dict[str, Any],
            # [New] UI 설정값을 받을 인자 추가
            ui_width: int = 720,
            ui_height: int = 1280,
            ui_fps: int = 24,
            ui_steps: int = 20
    ) -> Path:
        """
        [2단계] 상세화 (음성 -> BGM -> 영어 프롬프트)
        - ID 매칭: t_001 포맷 지원
        - 프롬프트 2: 복잡한 묘사 제거하고 "Subject from image 1 ... object from image 2" 공식 강제
        - [Fix] 오디오 파일이 있어도 길이(duration)를 강제 재측정하여 0초 문제 해결
        - [Fix] UI 설정(해상도, FPS 등)을 defaults에 저장
        """
        vpath = Path(video_json_path)
        p_dir = vpath.parent
        voice_dir = ensure_dir(p_dir / "voice")
        bgm_path = p_dir / "bgm.mp3"

        data = load_json(vpath, {})
        scenes = data.get("scenes", [])
        meta = data.get("meta", {})

        # [Fix] UI 설정값 저장 (defaults 업데이트)
        data.setdefault("defaults", {})
        data["defaults"].update({
            "image": {"width": ui_width, "height": ui_height, "fps": ui_fps},
            "movie": {"fps": ui_fps, "target_fps": ui_fps},
            "generator": {"steps": ui_steps}
        })

        # ---------------------------------------------------------------------
        # 1. 음성 생성 (유지) 및 시간 측정
        # ---------------------------------------------------------------------
        gender = meta.get("voice_gender", "female").lower()
        if "male" == gender:
            ref_voice = Path(r"C:\my_games\shorts_make\voice\남자성우1.mp3")
        else:
            ref_voice = Path(r"C:\my_games\shorts_make\voice\꼬꼬 음성.m4a")

        self.on_progress(f"[Enrich] 1/3단계: 음성 생성 ({gender}) 및 정밀 측정...")
        comfy_host = getattr(settings, "COMFY_HOST", "http://127.0.0.1:8188")
        total_dur = 0.0

        for sc in scenes:
            sid = sc["id"]
            config = _get_zonos_config(sc, self.ai)
            narr = sc.get("narration", "").strip()
            v_path = Path(sc.get("voice_file") or str(voice_dir / f"{sid}.wav"))

            # 내레이션이 없으면 기본 3초
            if not narr:
                if sc.get("seconds", 0) == 0: sc["seconds"] = 3
                total_dur += sc["seconds"]
                continue

            # (A) 파일이 없으면 생성
            if not v_path.exists() or v_path.stat().st_size == 0:
                self.on_progress(f"   🎙️ Scene {sid} 음성 생성...")
                success = generate_tts_zonos(narr, v_path, ref_voice, comfy_host, config)
                if not success:
                    # 실패 시 임시 4초 (다음 단계에서 재시도 가능)
                    if sc.get("seconds", 0) <= 0: sc["seconds"] = 4.0

            # (B) [Fix] 파일이 존재하면(생성 직후든 원래 있었든) 무조건 길이 측정
            if v_path.exists() and v_path.stat().st_size > 0:
                final_dur = 0.0
                # 파일 쓰기 완료 대기 겸 재시도
                for _ in range(3):
                    try:
                        d = get_duration(str(v_path))
                        if d > 0:
                            final_dur = d
                            break
                    except:
                        pass
                    time.sleep(0.1)

                if final_dur > 0:
                    # [Fix] 오디오 길이 + 0.5초 여유
                    sc["seconds"] = round(final_dur + 0.5, 2)
                else:
                    # 측정 실패 시 안전장치
                    if sc.get("seconds", 0) <= 0: sc["seconds"] = 4.0

            total_dur += sc["seconds"]
            sc["voice_file"] = str(v_path)

        data.setdefault("meta", {})["total_duration"] = round(total_dur, 2)
        save_json(vpath, data)

        # ---------------------------------------------------------------------
        # 2. BGM 생성 (유지)
        # ---------------------------------------------------------------------
        bgm_prompt = meta.get("bgm_prompt", "")
        if not bgm_prompt:
            bgm_prompt = "instrumental, background music, calm, minimal, piano, soft, loopable"
            meta["bgm_prompt"] = bgm_prompt

        if bgm_path.exists() and bgm_path.stat().st_size > 1024:
            self.on_progress(f"[Enrich] 2/3단계: BGM 이미 존재 (스킵).")
        else:
            self.on_progress(f"[Enrich] 2/3단계: BGM 생성 중...")
            generate_bgm_acestep(
                prompt=bgm_prompt,
                out_path=bgm_path,
                duration_sec=total_dur,
                comfy_host=comfy_host,
            )

        # ---------------------------------------------------------------------
        # 3. 프롬프트 상세화 (심플하고 강력한 합성 공식 적용)
        # ---------------------------------------------------------------------
        self.on_progress("[Enrich] 3/3단계: 비주얼 프롬프트 고도화 (합성 공식 강제)...")

        char_prompt = meta.get("character_prompt", "Young Korean model")
        if "male" == gender:
            gender_kw = "male, man"
        else:
            gender_kw = "female, woman"

        scene_texts = []
        for sc in scenes:
            scene_texts.append(f"- Scene {sc['id']} (Action): {sc.get('prompt')}")

        # [수정] 복잡한 설명 다 빼고 '공식'만 지키라고 명령
        sys_p = (
            "You are a prompt engineer for AI Image Compositing.\n"
            "Your Goal: Generate simple English prompts for 2-step generation.\n\n"
            "** STRICT RULES for `prompt_img_2` (The Paint/Composite Step) **\n"
            "You MUST use this exact sentence structure:\n"
            "\"[Subject] from image 1 [action] the object from image 2\"\n\n"
            "**Examples (Follow these exactly):**\n"
            "- \"The woman from image 1 holds the object from image 2 in her hand.\"\n"
            "- \"The man from image 1 looks at the object from image 2.\"\n"
            "- \"The table from image 1 has the object from image 2 placed on it.\"\n\n"
            "Do NOT use adjectives for the object (e.g. don't say 'red bottle', just say 'object from image 2').\n"
            "Do NOT add complex lighting or background details in prompt_img_2."
        )

        user_p = f"""
        Context: Character is "{char_prompt}" ({gender_kw}).

        Analyze these scenes and generate prompts:

        1. `prompt_img_1`: Describe the character/background. Leave space for the product. (e.g., "Woman extending empty hand")
        2. `prompt_img_2`: Apply the STRICT FORMULA: "[Subject] from image 1 ... object from image 2".
        3. `prompt_movie`: Simple camera movement (e.g., "Slow zoom in").

        [Scenes]
        {chr(10).join(scene_texts)}

        [Output JSON Format]
        {{
            "scenes": {{
                "t_001": {{ "prompt_img_1": "...", "prompt_img_2": "...", "prompt_negative": "...", "prompt_movie": "..." }},
                ...
            }}
        }}
        """

        try:
            resp = self.ai.ask_smart(sys_p, user_p, prefer="openai")
            enriched = self._safe_json_parse(resp)
            en_map = enriched.get("scenes", {})

            if isinstance(en_map, list):
                en_map = {f"t_{i + 1:03d}": item for i, item in enumerate(en_map)}

            for sc in scenes:
                sid = str(sc["id"])

                # 매칭 후보 (t_001, 001 등)
                candidates = [sid, sid.replace("t_", ""), sid.replace("t_", "").lstrip("0"), f"t_{sid}"]
                tgt = None

                # 키로 찾기
                for key in candidates:
                    if key in en_map:
                        tgt = en_map[key]
                        break

                # 값으로 찾기
                if not tgt:
                    for val in en_map.values():
                        if isinstance(val, dict) and str(val.get("id", "")) in candidates:
                            tgt = val
                            break

                if tgt:
                    sc["prompt_img_1"] = tgt.get("prompt_img_1", "")
                    sc["prompt_img_2"] = tgt.get("prompt_img_2", "")
                    sc["prompt_negative"] = tgt.get("prompt_negative", "")
                    sc["prompt_movie"] = tgt.get("prompt_movie", "")
                    sc["prompt_img"] = sc["prompt_img_1"]

            data["audit"]["enriched_at"] = _now_str()
            save_json(vpath, data)
            self.on_progress(f"[Enrich] 상세화 완료 (합성 공식 적용됨).")

        except Exception as e:
            self.on_progress(f"❌ 상세화 실패: {e}")

        return vpath

    def _safe_json_parse(self, text: str) -> Dict:
        try:
            text = re.sub(r"```json", "", text, flags=re.I).replace("```", "")
            return json.loads(text)
        except:
            s, e = text.find("{"), text.rfind("}")
            if s != -1 and e != -1:
                return json.loads(text[s:e + 1])
            raise ValueError("Invalid JSON response")


# -----------------------------------------------------------------------------
# 5. 이미지 생성기
# -----------------------------------------------------------------------------
# class ShoppingImageGenerator:
#     def __init__(self, on_progress: Optional[Callable[[str], None]] = None):
#         self.on_progress = on_progress or (lambda msg: None)
#
#     def generate_images(self, video_json_path: str | Path, skip_if_exists: bool = True) -> None:
#
#
#         def _cb(d):
#             self.on_progress(d.get("msg", ""))
#
#         vpath = Path(video_json_path).resolve()
#         proj_dir = vpath.parent
#
#         # 1. 해상도 가져오기 (기본값 안전장치 포함)
#         img_size = settings.DEFAULT_IMG_SIZE
#         width_val = img_size[0]
#         height_val = img_size[1]
#
#         # 2. 스텝 수 가져오기
#         steps_val = settings.DEFAULT_T2I_STEPS
#
#         # 3. 제품 이미지 경로 계산 (shopping.py on_gen_images_clicked와 동일 로직)
#         prod_path: str | None = None
#         try:
#             doc = load_json(vpath, {}) or {}
#             product = doc.get("product") or {}
#             img_file = (product.get("image_file") or "").strip()
#             if img_file:
#                 cand = (proj_dir / img_file).resolve()
#                 if cand.exists():
#                     prod_path = str(cand)
#         except Exception:
#             prod_path = None
#
#         self.on_progress(f"[Image][DBG] product_image_path={prod_path}")
#
#         # 4. 2-Step 이미지 생성 (제품 이미지를 image2로 강제 주입)
#         try:
#             build_shopping_images_2step(
#                 video_json_path=vpath,
#                 source_json_path=vpath,
#                 product_image_path=prod_path,
#                 ui_width=width_val,
#                 ui_height=height_val,
#                 steps=steps_val,
#                 skip_if_exists=skip_if_exists,
#                 on_progress=_cb,
#             )
#         except Exception as e:
#             self.on_progress(f"❌ 이미지 생성 오류: {e}")
#             raise e


# -----------------------------------------------------------------------------
# 6. 영상 생성/병합기
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 6. 영상 생성/병합기 (pydub 오디오 믹싱 + 자막 합성 추가)
# -----------------------------------------------------------------------------
class ShoppingMovieGenerator:
    def __init__(self, on_progress: Optional[Callable[[str], None]] = None):
        self.on_progress = on_progress or (lambda msg: None)

    def generate_movies(self, video_json_path: str | Path, skip_if_exists: bool = True, fps: int = 24) -> None:
        vpath = Path(video_json_path)  # 이것은 video_shopping.json 입니다.
        project_dir = vpath.parent

        # [중요] I2V 엔진(build_shots_with_i2v)은 무조건 'video.json'을 찾습니다.
        # 따라서 video_shopping.json 내용을 복사한 '임시 파일'을 만들어야 합니다.
        temp_video_json = project_dir / "video.json"

        self.on_progress(f"[Movie] I2V 준비: {vpath.name}")

        # 1. video_shopping.json 내용을 읽음
        data = load_json(vpath, {})

        # 2. duration 안전장치 (0이면 기본값 부여)
        for sc in data.get("scenes", []):
            if float(sc.get("duration", 0)) <= 0:
                sc["duration"] = float(sc.get("seconds", 4.0))

        # 3. 임시 파일(video.json)로 저장 -> 엔진이 이걸 읽음
        save_json(temp_video_json, data)

        def _cb(d):
            self.on_progress(d.get("msg", ""))

        try:
            # 4. 엔진 실행 (엔진은 폴더 내의 video.json을 자동으로 찾음)
            build_shots_with_i2v(str(project_dir), total_frames=0, ui_fps=fps, on_progress=_cb)
            self.on_progress("[Movie] 생성 완료")
        finally:
            # 5. [필수] 작업이 끝나면 임시 파일(video.json)은 헷갈리지 않게 삭제
            if temp_video_json.exists():
                try:
                    os.remove(temp_video_json)
                except:
                    pass

    def merge_movies(self, video_json_path: str | Path):
        """
        [최종 병합 - 오버랩 없음]
        - clips/*.mp4를 실제 길이 기준으로 재타임라인 구성
        - 내레이션/자막을 각 씬(클립)의 중앙에 배치
        - 내레이션이 길면 필요한 만큼 atempo로 줄임(상한 1.30)
        - 자막은 pad가 아니라 fade-in/out(alpha)로 처리
        - 최종 out: final_shopping_video.mp4
        """
        vpath = Path(video_json_path)
        project_dir = vpath.parent
        clips_dir = project_dir / "clips"
        bgm_path = project_dir / "bgm.mp3"

        final_output_path = project_dir / "final_shopping_video.mp4"
        ffmpeg_exe = getattr(settings, "FFMPEG_EXE", "ffmpeg")

        self.on_progress("[Merge] video.json 로드...")
        data = load_json(vpath, {})
        scenes = data.get("scenes", [])
        if not isinstance(scenes, list) or not scenes:
            self.on_progress("❌ scenes가 비었습니다.")
            return

        defaults = data.get("defaults", {}) if isinstance(data.get("defaults", {}), dict) else {}
        defaults_sub = defaults.get("subtitle", {}) if isinstance(defaults.get("subtitle", {}), dict) else {}

        font_family = str(defaults_sub.get("font_family") or getattr(settings, "DEFAULT_FONT_FAMILY", "Malgun Gothic"))
        title_size = int(defaults_sub.get("title_size") or getattr(settings, "DEFAULT_TITLE_FONT_SIZE", 55))
        narr_size = int(defaults_sub.get("narr_size") or getattr(settings, "DEFAULT_NARRATION_FONT_SIZE", 25))

        title_text = str(data.get("title") or "").strip()

        # 자막 페이드(초) — 기본 0.25
        subtitle_fade_in_sec = float(defaults_sub.get("fade_in_sec") or 0.25)
        subtitle_fade_out_sec = float(defaults_sub.get("fade_out_sec") or 0.25)

        self.on_progress(
            f"[Merge] 적용값: font='{font_family}', title={title_size}, narr={narr_size}, "
            f"bgm={'YES' if bgm_path.exists() else 'NO'}, "
            f"subtitle_fade_in={subtitle_fade_in_sec:.2f}, subtitle_fade_out={subtitle_fade_out_sec:.2f}"
        )

        # 1) 클립 수집(씬 순서대로)
        clip_paths: List[Path] = []
        missing = False
        for sc in scenes:
            sid = str(sc.get("id") or "").strip()
            if not sid:
                continue
            cpath = clips_dir / f"{sid}.mp4"
            if cpath.exists() and cpath.stat().st_size > 0:
                clip_paths.append(cpath)
            else:
                self.on_progress(f"⚠️ 클립 누락됨: {cpath.name}")
                missing = True

        if not clip_paths:
            self.on_progress("❌ 병합할 클립이 없습니다.")
            return
        if missing:
            self.on_progress("⚠️ 일부 씬 클립이 누락되어, 존재하는 클립만으로 병합합니다.")

        # 2) 최종 병합(영상+오디오+자막을 한 번에)
        self.on_progress("[Merge] 최종 병합(실측 길이 기반, 오버랩 없음, 자막 페이드) 시작...")
        try:
            concatenate_scene_clips_final_av(
                clip_paths=clip_paths,
                out_path=final_output_path,
                ffmpeg_exe=ffmpeg_exe,
                scenes=scenes,
                bgm_path=(bgm_path if bgm_path.exists() else None),
                bgm_volume=0.1,
                narration_volume=1.0,

                # ✅ 최종 병합은 오버랩/usable 축소 없음
                pad_in_sec=0.0,
                pad_out_sec=0.0,

                # ✅ 자막 페이드는 별도 파라미터로
                subtitle_fade_in_sec=subtitle_fade_in_sec,
                subtitle_fade_out_sec=subtitle_fade_out_sec,

                subtitle_font=font_family,
                subtitle_fontsize=narr_size,
                subtitle_y="h-140",
                subtitle_box=True,
                subtitle_boxcolor="black@0.45",
                subtitle_boxborderw=18,
                title_text=title_text,
                title_fontsize=title_size,
                title_y="h*0.12",
                video_crf=18,
                video_preset="medium",
                audio_bitrate="192k",

                # ✅ 병합 중 상세 로그를 비동기창에 출력
                on_progress=self.on_progress,
            )

            try:
                save_json(vpath, data)
            except Exception:
                pass

            self.on_progress(f"✅ 최종 병합 완료: {final_output_path.name}")

        except Exception as e:
            self.on_progress(f"❌ 최종 병합 실패: {e}")

    def _finalize_with_ffmpeg(
            self,
            ffmpeg_exe: str,
            video_path: Path,
            audio_path: Path,
            srt_path: Path,
            out_path: Path,
            title_text: str = "",
            font_settings: dict = None
    ):
        """
        - drawtext(제목): fontfile 사용
        - subtitles(SRT): fontsdir + force_style + original_size로 크기 스케일 문제 방지
        """

        if font_settings is None:
            font_settings = {}

        font_family = str(font_settings.get("family") or getattr(settings, "DEFAULT_FONT_FAMILY", "Malgun Gothic"))
        title_size = int(font_settings.get("title_size") or getattr(settings, "DEFAULT_TITLE_FONT_SIZE", 55))
        narr_size = int(font_settings.get("narr_size") or getattr(settings, "DEFAULT_NARRATION_FONT_SIZE", 25))
        sub_original_size = str(font_settings.get("sub_original_size") or "").strip()

        # drawtext용 폰트 파일 매핑
        font_file = "C:/Windows/Fonts/malgun.ttf"
        fam_lower = font_family.lower()

        if "굴림" in fam_lower or "gulim" in fam_lower:
            font_file = "C:/Windows/Fonts/gulim.ttc"
        elif "바탕" in fam_lower or "batang" in fam_lower:
            font_file = "C:/Windows/Fonts/batang.ttc"
        elif "돋움" in fam_lower or "dotum" in fam_lower:
            font_file = "C:/Windows/Fonts/dotum.ttc"
        elif "궁서" in fam_lower or "gungsuh" in fam_lower:
            font_file = "C:/Windows/Fonts/gungsuh.ttc"

        font_path_ffmpeg = font_file.replace("\\", "/").replace(":", "\\:")

        # subtitles 필터용 경로/스타일
        srt_path_str = str(srt_path).replace("\\", "/").replace(":", "\\:")
        fonts_dir = "C:/Windows/Fonts".replace("\\", "/").replace(":", "\\:")

        sub_style_raw = (
            f"FontName={font_family},FontSize={narr_size},Bold=1,"
            f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            f"BorderStyle=1,Outline=2,Shadow=0,Alignment=2,MarginV=50"
        )

        # ✅ 매우 중요: force_style 내부 콤마는 \, 로 이스케이프 (필터 체인 콤마와 충돌 방지)
        sub_style = sub_style_raw.replace(",", r"\,")

        filters: List[str] = []

        # (1) 자막 필터: fontsdir + force_style + original_size
        # original_size를 주면 libass 스케일링이 “영상 기준”으로 잡혀서 25가 25처럼 보입니다.
        if sub_original_size:
            filters.append(
                f"subtitles='{srt_path_str}':fontsdir='{fonts_dir}':original_size={sub_original_size}:force_style='{sub_style}'"
            )
        else:
            filters.append(
                f"subtitles='{srt_path_str}':fontsdir='{fonts_dir}':force_style='{sub_style}'"
            )

        # (2) 제목 drawtext
        if title_text:
            safe_title = title_text.replace("'", r"\'").replace(":", r"\:")

            alpha_expr = "if(lt(t,1),0,if(lt(t,3),(t-1)/2,if(lt(t,4),1,if(lt(t,6),(6-t)/2,0))))"

            drawtext_filter = (
                f"drawtext=fontfile='{font_path_ffmpeg}':text='{safe_title}':"
                f"fontsize={title_size}:fontcolor=white:borderw=2:bordercolor=black:"
                f"x=(w-text_w)/2:y=h*0.15:"
                f"alpha='{alpha_expr}'"
            )
            filters.append(drawtext_filter)

        filter_complex = ",".join(filters)

        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-vf", filter_complex,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(out_path)
        ]

        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            startupinfo=startupinfo
        )

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg 렌더링 실패:\n{result.stderr}")

    # --- 내부 헬퍼 메서드 ---

    def _mix_audio_with_pydub(self, scenes: List[Dict], bgm_path: Path, voice_dir: Path, total_dur_sec: float,
                              out_path: Path):
        """pydub를 사용하여 BGM(20%볼륨, 페이드아웃)과 내레이션을 믹싱"""
        if AudioSegment is None:
            raise ImportError("pydub 라이브러리가 필요합니다.")

        # pydub는 밀리초 단위 사용
        total_ms = int(total_dur_sec * 1000)

        # 1. 베이스 트랙 생성 (무음)
        final_mix = AudioSegment.silent(duration=total_ms)

        # 2. BGM 처리
        if bgm_path.exists():
            bgm = AudioSegment.from_file(str(bgm_path))

            # 길이 맞추기 (짧으면 루프)
            if len(bgm) < total_ms:
                loops = int(total_ms / len(bgm)) + 1
                bgm = bgm * loops

            bgm = bgm[:total_ms]  # 길이 자르기

            # 볼륨 20%로 줄이기 (약 -14dB)
            # 20 * log10(0.2) ≈ -13.97 dB
            bgm = bgm - 14

            # 마지막 2초 페이드 아웃
            bgm = bgm.fade_out(2000)

            # 베이스에 BGM 합성
            final_mix = final_mix.overlay(bgm)

        # 3. 내레이션(Voice) 배치
        for sc in scenes:
            sid = sc.get("id")
            voice_file = sc.get("voice_file")

            # voice_file 경로가 절대경로가 아닐 경우 voice_dir 기준 탐색
            v_path = None
            if voice_file:
                if Path(voice_file).exists():
                    v_path = Path(voice_file)
                elif (voice_dir / Path(voice_file).name).exists():
                    v_path = voice_dir / Path(voice_file).name

            # 없으면 id 기반으로 다시 찾기
            if not v_path and (voice_dir / f"{sid}.wav").exists():
                v_path = voice_dir / f"{sid}.wav"

            if v_path and v_path.exists():
                voice_seg = AudioSegment.from_file(str(v_path))
                start_time = float(sc.get("start", 0))
                start_ms = int(start_time * 1000)

                # 믹싱 (position 인자로 위치 지정)
                final_mix = final_mix.overlay(voice_seg, position=start_ms)

        # 4. 저장
        final_mix.export(str(out_path), format="wav")

    def _create_srt_file(self, scenes: List[Dict], srt_path: Path):
        """video.json 정보를 바탕으로 자막(SRT) 파일 생성"""

        def sec_to_srt_fmt(seconds: float) -> str:
            """초 단위를 SRT 시간 포맷(HH:MM:SS,mmm)으로 변환"""
            millis = int((seconds % 1) * 1000)
            seconds = int(seconds)
            mins, secs = divmod(seconds, 60)
            hrs, mins = divmod(mins, 60)
            return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

        with open(srt_path, "w", encoding="utf-8") as f:
            idx = 1
            for sc in scenes:
                text = sc.get("lyric") or sc.get("narration") or ""
                text = text.strip()
                if not text:
                    continue

                start = float(sc.get("start", 0))
                end = float(sc.get("end", 0))

                # 자막이 너무 짧게 지나가는 것 방지
                if end - start < 0.5:
                    end = start + 2.0

                f.write(f"{idx}\n")
                f.write(f"{sec_to_srt_fmt(start)} --> {sec_to_srt_fmt(end)}\n")
                f.write(f"{text}\n\n")
                idx += 1




# -----------------------------------------------------------------------------
# 7. 파이프라인
# -----------------------------------------------------------------------------
# class ShoppingShortsPipeline:
#     def __init__(self, on_progress: Optional[Callable[[str], None]] = None):
#         self.on_progress = on_progress or (lambda msg: None)
#
#     def run_all(
#             self,
#             product_dir: str | Path,
#             product_data: Dict[str, Any],
#             options: Optional[BuildOptions] = None,
#             build_json: bool = True,
#             build_images: bool = True,
#             build_movies: bool = True,
#             merge: bool = True,
#             skip_if_exists: bool = True,
#     ) -> Path:
#         options = options or BuildOptions()
#         vpath = Path(product_dir) / "video_shopping.json"
#
#         builder = ShoppingVideoJsonBuilder(self.on_progress)
#
#         if build_json:
#             if not vpath.exists():
#                 vpath = builder.create_draft(product_dir, product_data, options)
#             builder.enrich_video_json(vpath, product_data)
#
#         if build_images:
#             img_gen = ShoppingImageGenerator(self.on_progress)
#             img_gen.generate_images(vpath, skip_if_exists)
#
#         if build_movies:
#             mov_gen = ShoppingMovieGenerator(self.on_progress)
#             mov_gen.generate_movies(vpath, skip_if_exists, fps=options.fps)
#
#         if merge:
#             mov_gen = ShoppingMovieGenerator(self.on_progress)
#             mov_gen.merge_movies(vpath)
#
#         return vpath

# video_shopping_build.json 에서 음성길이가 각 0.5초 추가된 것을 그대로 가져옴 (sc["seconds"] = round(final_dur + 0.5, 2)) 이 부분임.
def convert_shopping_to_video_json_with_ai(
        project_dir: str,
        ai_client: Any = None,
        fps: int = 30,
        width: int = 1080,
        height: int = 1920,
        steps: int = 20,
        # [추가] UI에서 전달받을 인자들
        font_path: str = "",
        title_fontsize: int = 60,
        sub_fontsize: int = 40,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None
) -> str:
    """
    [쇼핑->쇼츠 변환 최종판]
    - video_shopping.json -> video.json 구조 변환
    - [수정] lyric 필드는 'narration' 값을 우선 사용
    - [수정] UI 설정(글꼴, 크기)을 인자로 받아 video.json에 저장
    """
    import json
    import datetime
    from pathlib import Path
    from app.story_enrich import fill_prompt_movie_with_ai_shopping

    def _log(msg: str):
        if on_progress:
            on_progress({"msg": msg})
        print(f"[ShoppingConverter] {msg}")

    proj_path = Path(project_dir)
    src_json_path = proj_path / "video_shopping.json"
    dst_json_path = proj_path / "video.json"
    imgs_dir = proj_path / "imgs"

    if not src_json_path.exists():
        raise FileNotFoundError(f"video_shopping.json이 없습니다: {src_json_path}")

    try:
        with open(src_json_path, "r", encoding="utf-8") as f:
            src_data = json.load(f)
    except Exception as e:
        raise ValueError(f"데이터 로드 실패: {e}")

    _log(f"데이터 구조 변환 시작. (해상도: {width}x{height}, FPS: {fps})")

    prod = src_data.get("product", {})
    project_name = prod.get("product_name") or src_data.get("project_name", "Shopping Project")

    # 캐릭터 정보 추출
    meta_info = src_data.get("meta", {})

    src_scenes = src_data.get("scenes", [])
    if not src_scenes:
        src_scenes = src_data.get("groups", [])

    new_scenes: List[Dict[str, Any]] = []
    current_time = 0.0
    full_lyrics_parts: List[str] = []

    for idx, sc in enumerate(src_scenes):
        original_id = str(sc.get("id", "")).strip()
        if original_id:
            scene_id = original_id
        else:
            scene_id = f"t_{idx + 1:03d}"

        target_img_name = f"{scene_id}.png"

        voice_file = sc.get("voice_file") or sc.get("audio_path") or ""
        voice_path_obj = None
        if voice_file:
            if Path(voice_file).is_absolute():
                voice_path_obj = Path(voice_file)
            else:
                voice_path_obj = proj_path / voice_file
            if not voice_path_obj.exists():
                voice_path_obj = None

        try:
            dur = float(sc.get("duration") or sc.get("seconds") or 4.0)
        except Exception:
            dur = 4.0
        if dur <= 0:
            dur = 4.0

        narration = str(sc.get("narration") or sc.get("narration_text") or sc.get("lyric") or "").strip()

        start_t = current_time
        end_t = current_time + dur
        current_time = end_t

        if narration:
            full_lyrics_parts.append(narration)

        new_scene = {
            "id": scene_id,
            "section": "main",
            "start": round(start_t, 3),
            "end": round(end_t, 3),
            "duration": round(dur, 3),
            "img_file": str(imgs_dir / target_img_name),
            "voice_file": str(voice_path_obj) if voice_path_obj else "",
            "lyric": narration,
            "prompt": sc.get("prompt", ""),
            "prompt_movie": sc.get("prompt_movie", ""),
            "prompt_img": sc.get("prompt_img", ""),
            "prompt_negative": sc.get("prompt_negative", ""),
            "effect": [],
            "screen_transition": (idx == len(src_scenes) - 1)
        }
        new_scenes.append(new_scene)

    total_duration = current_time
    full_lyrics = "\n".join(full_lyrics_parts)

    # [중요] 4번 기능: UI에서 받은 값 저장
    final_ui_prefs = {
        "font_path": str(font_path).strip(),
        "title_fontsize": int(title_fontsize) if title_fontsize > 0 else 60,
        "sub_fontsize": int(sub_fontsize) if sub_fontsize > 0 else 40
    }

    # 기존 파일이 있다면 UI 설정 병합
    if dst_json_path.exists():
        try:
            with open(dst_json_path, "r", encoding="utf-8") as old_f:
                old_data = json.load(old_f)
                old_ui = old_data.get("defaults", {}).get("ui_prefs", {})
                if old_ui:
                    # 빈 값이면 기존 값 유지, 값이 있으면 덮어쓰기
                    if not final_ui_prefs["font_path"] and old_ui.get("font_path"):
                        final_ui_prefs["font_path"] = old_ui["font_path"]
                    # 폰트 크기는 UI 값 우선 (이미 위에서 int 변환됨)
                _log("기존 UI 설정을 병합했습니다.")
        except Exception:
            pass

    video_data = {
        "title": project_name,
        "duration": round(total_duration, 3),
        "fps": fps,
        "lyrics": full_lyrics,
        "meta": meta_info,
        "scenes": new_scenes,
        "defaults": {
            "movie": {"fps": fps, "target_fps": fps, "input_fps": fps},
            "image": {"width": width, "height": height, "fps": fps},
            "generator": {"steps": steps},
            "ui_prefs": final_ui_prefs  # 저장
        },
        "audit": {
            "source": "shopping_converter_v2",
            "converted_at": str(datetime.datetime.now())
        }
    }

    # 1차 저장
    with open(dst_json_path, "w", encoding="utf-8") as f:
        json.dump(video_data, f, indent=2, ensure_ascii=False)

    _log(f"video.json 기본 생성 완료 (총 {total_duration:.2f}초)")

    # AI 상세화
    if ai_client:
        _log("AI 상세화 (Long-Take Prompt) 진행...")
        try:
            def ask_wrapper(sys_msg, user_msg):
                return ai_client.ask_smart(sys_msg, user_msg, prefer="openai")

            def _trace_wrapper(tag, msg):
                _log(f"[{tag}] {msg}")

            video_data = fill_prompt_movie_with_ai_shopping(
                video_data,
                ask_wrapper,
                trace=_trace_wrapper
            )

            # AI 결과 반영 후 재저장
            with open(dst_json_path, "w", encoding="utf-8") as f:
                json.dump(video_data, f, indent=2, ensure_ascii=False)

            _log("✅ AI 상세화 완료.")
        except Exception as e:
            _log(f"❌ AI 상세화 실패: {e}")
            import traceback
            traceback.print_exc()

    return str(dst_json_path)




def add_shopping_texts_with_drawtext(
        *,
        video_in_path: Path,
        video_json_path: Path,
        out_path: Path,
        ffmpeg_exe: str,
        font_family: str,
        title_fontsize: int,
        narr_fontsize: int,
) -> str:
    """
    Shopping 탭 최종 렌더링(drawtext):
    - subtitles(libass) 사용 안 함
    - drawtext로 제목 + 내레이션을 직접 하드코딩
    - 제목: 1초 대기 -> 2초 fade in -> 1초 유지 -> 2초 fade out (상단)
    - 내레이션: scene start~end 동안 하단 표시

    [중요]
    - drawtext는 일부 폰트(TTC/비트맵/컬렉션)를 "1bpp"로 판정하여 거부할 수 있음.
    - 따라서 Windows에서는 안정적으로 동작하는 TTF(예: malgun.ttf)로 강제.
    """

    video_data = load_json(video_json_path, {}) or {}
    scenes = video_data.get("scenes", [])

    meta = video_data.get("meta", {}) if isinstance(video_data.get("meta", {}), dict) else {}
    title = (meta.get("title") or video_data.get("title") or "").strip()

    # ✅ drawtext 안정 폰트: malgun.ttf로 강제 (Shorts에서 검증된 방식)
    # 굴림/바탕/돋움 등 TTC 계열은 drawtext에서 1bpp 판정 이슈가 발생할 수 있어 사용하지 않음.
    font_file = "C:/Windows/Fonts/malgun.ttf"
    font_path_ffmpeg = font_file.replace(os.path.sep, "/").replace(":", "\\:")

    filters: list[str] = []

    def _esc_ffmpeg_text(s: str) -> str:
        return (
            s.replace("\\", "\\\\")
             .replace(":", "\\:")
             .replace("'", "'\\\\''")
        )

    # 1) 제목 (페이드)
    if title:
        title_escaped = _esc_ffmpeg_text(title)
        alpha_expr = "if(lt(t,1),0,if(lt(t,3),(t-1)/2,if(lt(t,4),1,if(lt(t,6),(6-t)/2,0))))"

        filters.append(
            "drawtext="
            f"fontfile='{font_path_ffmpeg}':"
            f"text='{title_escaped}':"
            f"fontsize={int(title_fontsize)}:"
            "fontcolor=white:"
            "box=1:boxcolor=black@0.5:boxborderw=6:"
            "x=(w-text_w)/2:y=h*0.12:"
            f"alpha='{alpha_expr}'"
        )

    # 2) 내레이션(씬별)
    for sc in scenes:
        text = (sc.get("narration") or sc.get("lyric") or "").strip()
        if not text:
            continue

        start = float(sc.get("start", 0.0) or 0.0)
        end = float(sc.get("end", 0.0) or 0.0)
        if end <= start:
            continue

        text_escaped = _esc_ffmpeg_text(text).replace("\n", "\\n")

        filters.append(
            "drawtext="
            f"fontfile='{font_path_ffmpeg}':"
            f"text='{text_escaped}':"
            f"fontsize={int(narr_fontsize)}:"
            "fontcolor=white:"
            "box=1:boxcolor=black@0.5:boxborderw=5:"
            "x=(w-text_w)/2:y=h*0.82:"
            f"enable='between(t,{start},{end})'"
        )

    if not filters:
        shutil.copy2(str(video_in_path), str(out_path))
        return str(out_path)

    filter_complex = ",".join(filters)

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(video_in_path),
        "-vf", filter_complex,
        "-c:a", "copy",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        str(out_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")

    if result.returncode != 0:
        raise RuntimeError(f"FFMPEG(drawtext) 텍스트 삽입 실패:\n{result.stderr}")

    return str(out_path)



#