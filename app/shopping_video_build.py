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
    ensure_dir
)
from app import settings
from app.video_build import build_shots_with_i2v, concatenate_scene_clips, fill_prompt_movie_with_ai
from app.audio_sync import get_audio_duration

def _now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# -----------------------------------------------------------------------------
# [New] ComfyUI 제출 및 대기 함수 (독립형)
# -----------------------------------------------------------------------------
def _submit_and_wait_local(
        base_url: str,
        graph: dict,
        timeout: int = 900,
        poll: float = 2.0,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None
) -> dict:
    """ComfyUI 워크플로우 제출 및 대기"""
    client_id = str(uuid.uuid4())
    payload = {"prompt": graph, "client_id": client_id}

    try:
        resp = requests.post(f"{base_url}/prompt", json=payload, timeout=30)
        resp.raise_for_status()
        prompt_id = resp.json().get("prompt_id")
    except Exception as e:
        raise RuntimeError(f"ComfyUI 제출 실패: {e}")

    start_t = time.time()
    while True:
        elapsed = time.time() - start_t
        if elapsed > timeout:
            raise TimeoutError(f"ComfyUI 시간 초과 ({elapsed:.1f}s)")

        try:
            h_resp = requests.get(f"{base_url}/history/{prompt_id}", timeout=10)
            if h_resp.status_code == 200:
                h_data = h_resp.json()
                if prompt_id in h_data:
                    return h_data[prompt_id]
        except Exception:
            pass
        time.sleep(poll)


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


def generate_tts_zonos(
        text: str,
        out_path: Path,
        ref_audio: Path,
        comfy_host: str = "http://127.0.0.1:8188",
        config: Dict[str, Any] = None
) -> bool:
    if not text:
        return False

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

    # 1. 텍스트/시드/속도 설정
    found_gen = False
    for nid, node in graph.items():
        # Zonos 노드 찾기 (class_type에 Zonos 포함 & inputs에 speech가 있는 노드)
        if "Zonos" in node.get("class_type", "") and "speech" in node.get("inputs", {}):
            node["inputs"]["speech"] = text
            node["inputs"]["seed"] = random.randint(1, 2 ** 32)
            if config and "speed" in config:
                node["inputs"]["speed"] = config["speed"]
            found_gen = True
            break

    # 만약 위 루프에서 못 찾았으면 ID 24번 시도 (fallback)
    if not found_gen and "24" in graph:
        graph["24"]["inputs"]["speech"] = text
        graph["24"]["inputs"]["seed"] = random.randint(1, 2 ** 32)
        if config and "speed" in config:
            graph["24"]["inputs"]["speed"] = config["speed"]

    # 2. 감정 설정 (Zonos Emotion 노드)
    if config and "emotion" in config:
        emotions = config["emotion"]
        for nid, node in graph.items():
            if node.get("class_type") == "Zonos Emotion":
                for k, v in emotions.items():
                    if k in node["inputs"]:
                        node["inputs"][k] = v
                break

    # 3. 참조 오디오 설정
    found_audio = False
    for nid, node in graph.items():
        if node.get("class_type") == "LoadAudio":
            node["inputs"]["audio"] = ref_copy_name
            found_audio = True
            break

    if not found_audio and "12" in graph:
        graph["12"]["inputs"]["audio"] = ref_copy_name

    try:
        res = _submit_and_wait_local(comfy_host, graph, timeout=300)
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
                        with open(out_path, "wb") as f:
                            f.write(resp.content)
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
        res = _submit_and_wait_local(comfy_host, graph, timeout=600)  # 오디오 생성은 좀 걸릴 수 있음

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
        video_json_path: str | Path,
        *,
        ui_width: int = 720,
        ui_height: int = 1280,
        steps: int = 28,
        skip_if_exists: bool = True,
        on_progress: Optional[Callable[[Dict], None]] = None
) -> None:
    """
    [최종 수정] 이미지 생성 함수
    - 원칙: 프롬프트는 무조건 'video_shopping.json'(원본)에서만 가져옵니다.
    - video.json은 생성된 이미지 경로를 저장하는 용도로만 사용합니다.
    """
    print(f"\n======== [Image Build Start (Source: video_shopping.json)] ========")
    print(f"Target: {video_json_path}")

    vpath = Path(video_json_path)
    product_dir = vpath.parent
    imgs_dir = ensure_dir(product_dir / "imgs")

    # 1. 제품 이미지 찾기
    product_json_path = product_dir / "product.json"
    product_img_file = None
    if product_json_path.exists():
        try:
            pj = json.loads(product_json_path.read_text(encoding="utf-8"))
            if pj.get("image_file"):
                pi = product_dir / pj["image_file"]
                if pi.exists():
                    product_img_file = pi
        except:
            pass

    if product_img_file:
        print(f"✅ Product Image Found: {product_img_file.name}")
    else:
        print(f"❌ Product Image NOT FOUND! Step 2 will be skipped.")

    # 2. [핵심] 원본 데이터(video_shopping.json) 로드 -> 프롬프트의 유일한 출처
    shopping_source_map = {}
    shop_json_path = product_dir / "video_shopping.json"

    if not shop_json_path.exists():
        print(f"❌ Critical Error: video_shopping.json not found!")
        if on_progress: on_progress({"msg": "❌ 원본 데이터(video_shopping.json)가 없습니다."})
        return

    try:
        shop_data = load_json(shop_json_path, {})
        shop_scenes = shop_data.get("scenes", [])

        # ID 매핑 (001, 1, t_001 등 다양한 포맷 대응)
        for ss in shop_scenes:
            raw_id = str(ss.get("id", ""))
            # 그대로 저장
            shopping_source_map[raw_id] = ss
            # 숫자만 추출해서 저장 (001 -> 1)
            if raw_id.isdigit():
                shopping_source_map[str(int(raw_id))] = ss
                shopping_source_map[f"t_{int(raw_id):03d}"] = ss
            # t_ 제거 버전 저장
            if raw_id.startswith("t_"):
                shopping_source_map[raw_id.replace("t_", "")] = ss

        print(f"✅ Source Data Loaded: {len(shop_scenes)} scenes from video_shopping.json")
    except Exception as e:
        print(f"❌ Failed to load video_shopping.json: {e}")
        return

    # 3. 타겟 데이터(video.json) 로드
    video_doc = load_json(vpath, {})
    scenes = video_doc.get("scenes", [])

    comfy_host = getattr(settings, "COMFY_HOST", "http://127.0.0.1:8188").rstrip("/")
    comfy_input_dir = Path(settings.COMFY_INPUT_DIR)
    comfy_input_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------
    # Step 1: Z-Image Batch (베이스 생성)
    # -----------------------------------------------------------
    if on_progress: on_progress({"msg": "=== [Step 1] 베이스 이미지 생성 (Source 참조) ==="})

    wf_z_path = Path(settings.JSONS_DIR) / "Z-Image-lora.json"
    if not wf_z_path.exists():
        wf_z_path = Path(r"C:\my_games\shorts_make\app\jsons\Z-Image-lora.json")

    if wf_z_path.exists():
        with open(wf_z_path, "r", encoding="utf-8") as f:
            graph_z_origin = json.load(f)

        for sc in scenes:
            sid = sc.get("id")
            temp_file = imgs_dir / f"temp_{sid}.png"

            # [핵심] 무조건 원본(video_shopping.json)에서 프롬프트 가져옴
            source_scene = shopping_source_map.get(sid)
            if not source_scene:
                # 매핑 실패 시 다른 키 시도
                if sid.startswith("t_"):
                    source_scene = shopping_source_map.get(sid.replace("t_", ""))
                    if not source_scene and sid.replace("t_", "").isdigit():
                        source_scene = shopping_source_map.get(str(int(sid.replace("t_", ""))))

            p1 = ""
            if source_scene:
                p1 = source_scene.get("prompt_img_1") or source_scene.get("prompt_img", "")

            if not p1:
                print(f"⚠️ [Step 1] No prompt in video_shopping.json for {sid} (Skipping)")
                continue

            # 파일 존재 시 스킵
            if skip_if_exists and temp_file.exists() and temp_file.stat().st_size > 0:
                print(f"[Step 1] Skip existing: {sid}")
                continue

            if on_progress: on_progress({"msg": f"[Step 1] 베이스 생성: {sid}..."})
            print(f"[Step 1] Generating {sid} using prompt from Source...")

            graph = json.loads(json.dumps(graph_z_origin))
            for nid, node in graph.items():
                ctype = node.get("class_type", "")
                inputs = node.get("inputs", {})

                if ctype == "CLIPTextEncode" and nid == "6":
                    inputs["text"] = p1
                if "LatentImage" in ctype:
                    inputs["width"] = ui_width
                    inputs["height"] = ui_height
                if ctype == "KSampler" and "seed" in inputs:
                    inputs["seed"] = random.randint(1, 10 ** 9)
                    if "steps" in inputs: inputs["steps"] = steps
                if ctype == "PreviewImage":
                    node["class_type"] = "SaveImage"
                    node.setdefault("inputs", {})["filename_prefix"] = "Z_Base"

            try:
                res = _submit_and_wait_local(comfy_host, graph, on_progress=on_progress)
                outputs = res.get("outputs", {})
                found = False
                for _, out_d in outputs.items():
                    for img in out_d.get("images", []):
                        fname = img["filename"]
                        resp = requests.get(f"{comfy_host}/view", params={"filename": fname, "type": img["type"]})
                        with open(temp_file, "wb") as f:
                            f.write(resp.content)
                        found = True
                        break
                    if found: break
                if found:
                    print(f"✅ [Step 1] Created: {temp_file.name}")
            except Exception as e:
                print(f"❌ [Step 1] Error {sid}: {e}")

    # -----------------------------------------------------------
    # Step 2: Qwen Batch (제품 합성)
    # -----------------------------------------------------------
    if on_progress: on_progress({"msg": "=== [Step 2] 제품 합성 (Source 참조) ==="})
    print("\n-------- Starting Step 2 (Qwen Edit) --------")

    wf_q_path = Path(settings.JSONS_DIR) / "QwenEdit2511-V1.json"
    if not wf_q_path.exists():
        wf_q_path = Path(r"C:\my_games\shorts_make\app\jsons\QwenEdit2511-V1.json")

    if not wf_q_path.exists() or not product_img_file:
        print("❌ Step 2 Aborted: Missing workflow or product image.")
        return

    with open(wf_q_path, "r", encoding="utf-8") as f:
        graph_q_origin = json.load(f)

    prod_input_name = f"prod_{uuid.uuid4().hex[:6]}.png"
    shutil.copy2(product_img_file, comfy_input_dir / prod_input_name)

    for sc in scenes:
        sid = sc.get("id")
        final_file = imgs_dir / f"{sid}.png"
        temp_file = imgs_dir / f"temp_{sid}.png"

        # 파일 존재 시 스킵
        if skip_if_exists and final_file.exists() and final_file.stat().st_size > 0:
            sc["img_file"] = str(final_file)
            print(f"[Step 2] Skip existing: {sid}")
            continue

        if not temp_file.exists():
            print(f"⚠️ [Step 2] Base image missing for {sid}. Skipping.")
            continue

        # [핵심] 무조건 원본(video_shopping.json)에서 프롬프트 가져옴
        source_scene = shopping_source_map.get(sid)
        if not source_scene:
            if sid.startswith("t_"):
                source_scene = shopping_source_map.get(sid.replace("t_", ""))
                if not source_scene and sid.replace("t_", "").isdigit():
                    source_scene = shopping_source_map.get(str(int(sid.replace("t_", ""))))

        raw_p2 = ""
        if source_scene:
            raw_p2 = source_scene.get("prompt_img_2") or ""

        if not raw_p2:
            print(f"  - No Prompt 2 in Source. Copying Step 1 image.")
            shutil.copy2(temp_file, final_file)
            sc["img_file"] = str(final_file)
            continue

        # [Auto-Fix] 선생님이 가르쳐주신 문법 적용 ('from image 1' 필수)
        p2_fixed = raw_p2
        if "from image 1" not in raw_p2.lower():
            pattern = re.compile(r"^(The|A|An)\s+([a-zA-Z0-9\s]+?)\s+(holding|has|is|with|placing|looking)",
                                 re.IGNORECASE)
            match = pattern.search(raw_p2)
            if match:
                p2_fixed = raw_p2.replace(match.group(2), f"{match.group(2)} from image 1", 1)
            else:
                p2_fixed = f"The subject from image 1 {raw_p2}"
            print(f"🔧 [Auto-Fix] {sid}: {p2_fixed}")
        else:
            print(f"👍 [Prompt OK] {sid} (Source)")

        base_input_name = f"base_{sid}_{uuid.uuid4().hex[:6]}.png"
        shutil.copy2(temp_file, comfy_input_dir / base_input_name)

        graph = json.loads(json.dumps(graph_q_origin))

        if "9" in graph: graph["9"]["inputs"]["image"] = base_input_name
        if "32" in graph: graph["32"]["inputs"]["image"] = prod_input_name
        if "88" in graph: graph["88"]["inputs"]["value"] = p2_fixed

        for nid, node in graph.items():
            if node.get("class_type") == "PreviewImage":
                node["class_type"] = "SaveImage"
                node.setdefault("inputs", {})["filename_prefix"] = "ShopFinal"

        if on_progress: on_progress({"msg": f"[Step 2] 합성 진행({sid})..."})
        try:
            res = _submit_and_wait_local(comfy_host, graph, on_progress=on_progress)
            outputs = res.get("outputs", {})
            found = False
            for _, out_d in outputs.items():
                for img in out_d.get("images", []):
                    fname = img["filename"]
                    resp = requests.get(f"{comfy_host}/view", params={"filename": fname, "type": img["type"]})
                    with open(final_file, "wb") as f:
                        f.write(resp.content)
                    sc["img_file"] = str(final_file)
                    found = True
                    break
                if found: break

            if found:
                print(f"✅ Scene {sid} Synthesis Done.")
            else:
                print(f"❌ Scene {sid} Failed (No output).")

        except Exception as e:
            print(f"❌ Scene {sid} Error: {e}")

    # 최종 결과 업데이트 (이미지 경로 등)
    try:
        with open(vpath, "w", encoding="utf-8") as f:
            json.dump(video_doc, f, indent=2, ensure_ascii=False)
    except:
        pass

    print("======== [Image Build End] ========\n")


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

class ShoppingVideoJsonBuilder:
    def __init__(self, on_progress: Optional[Callable[[str], None]] = None):
        self.on_progress = on_progress or (lambda msg: None)
        self.ai = AI()

    def create_draft(self, product_dir: str | Path, product_data: Dict[str, Any], options: BuildOptions) -> Path:
        """
        [1단계] 기획 초안 생성
        [수정] 시작부터 ID를 't_001' 포맷으로 고정하여 파이프라인 전체 통일성을 보장합니다.
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

        system_prompt = (
            "당신은 AI 영상 생성(I2V)을 위한 숏폼 기획 전문가이자 음악 감독입니다. "
            "상품을 분석하여 기획안을 작성하세요.\n\n"
            "**[필수 시각화 규칙]**\n"
            "1. **One Scene = One Action**: 한 장면에는 오직 '하나의 동작'만 묘사.\n"
            "2. **No Split Screens**: 전체 화면 구성.\n"
            "3. **Focus on Impact**: 결정적 순간 포착.\n"
            f"{bgm_guide}\n"
            f"{visual_rules}"
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
                    "id": "t_001",  <-- AI에게 예시도 t_포맷으로 제시
                    "banner": "...",
                    "prompt": "화면 묘사 (한글, 절대 시퀀스/단계 나열 금지, 단일 동작 위주, 로고/텍스트 묘사 금지)",
                    "narration": "실제 읽을 대사 (지시문 제외)",
                    "sfx": "효과음",
                    "voice_config": {{
                        "speed": 1.0, 
                        "emotion": {{ "neutral": 1.0, "happy": 0.0, "sad": 0.0, "disgust": 0.0, "fear": 0.0, "surprise": 0.0, "anger": 0.0, "other": 0.0 }}
                    }},
                    "subtitle": "..."
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
            # [핵심 수정] AI가 001로 주든 1로 주든, 무조건 t_001 포맷으로 강제 변환
            # 이제 video_shopping.json 단계부터 t_001로 저장됩니다.
            sid = f"t_{idx + 1:03d}"

            new_scene = {
                "id": sid,
                "banner": sc.get("banner"),
                "prompt": sc.get("prompt", ""),
                "narration": sc.get("narration", ""),
                "sfx": sc.get("sfx", ""),
                "voice_config": sc.get("voice_config", {"speed": 1.0, "emotion": {"neutral": 1.0}}),
                "subtitle": sc.get("subtitle", ""),
                "seconds": 0,
                "prompt_img_1": "",
                "prompt_img_2": "",
                "prompt_movie": "",
                "prompt_negative": "",
                # 이미지 경로도 t_001.png 로 통일
                "img_file": str(imgs_dir / f"{sid}.png"),
                "movie_file": str(clips_dir / f"{sid}.mp4"),
                "voice_file": str(voice_dir / f"{sid}.wav")
            }
            final_json["scenes"].append(new_scene)

        save_json(vpath, final_json)
        self.on_progress(f"[Draft] 초안 완료. (ID 포맷: t_001)")
        return vpath

    def enrich_video_json(self, video_json_path: str | Path, product_data: Dict[str, Any]) -> Path:
        """
        [2단계] 상세화 (음성 -> BGM -> 영어 프롬프트)
        - ID 매칭: t_001 포맷 지원
        - 프롬프트 2: 복잡한 묘사 제거하고 "Subject from image 1 ... object from image 2" 공식 강제
        """
        vpath = Path(video_json_path)
        p_dir = vpath.parent
        voice_dir = ensure_dir(p_dir / "voice")
        bgm_path = p_dir / "bgm.mp3"

        data = load_json(vpath, {})
        scenes = data.get("scenes", [])
        meta = data.get("meta", {})

        # ---------------------------------------------------------------------
        # 1. 음성 생성 (유지)
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

            if not narr:
                if sc.get("seconds", 0) == 0: sc["seconds"] = 3
                total_dur += sc["seconds"]
                continue

            if not v_path.exists() or v_path.stat().st_size == 0:
                self.on_progress(f"   🎙️ Scene {sid} 음성 생성...")
                success = generate_tts_zonos(narr, v_path, ref_voice, comfy_host, config)
                if not success:
                    sc["seconds"] = 4
                else:
                    final_dur = 0.0
                    for _ in range(5):
                        try:
                            d = get_audio_duration(str(v_path))
                            if d > 0:
                                final_dur = d
                                break
                        except:
                            pass
                        time.sleep(0.2)
                    sc["seconds"] = round(final_dur + 0.5, 2) if final_dur > 0 else 4

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
class ShoppingImageGenerator:
    def __init__(self, on_progress: Optional[Callable[[str], None]] = None):
        self.on_progress = on_progress or (lambda msg: None)

    def generate_images(self, video_json_path: str | Path, skip_if_exists: bool = True) -> None:
        def _cb(d):
            self.on_progress(d.get("msg", ""))

        # 1. 해상도 가져오기 (기본값 안전장치 포함)
        img_size = settings.DEFAULT_IMG_SIZE
        width_val = img_size[0]  # 가로
        height_val = img_size[1]  # 세로

        # 2. 스텝 수 가져오기
        steps_val = settings.DEFAULT_T2I_STEPS

        try:
            build_shopping_images_2step(
                video_json_path=video_json_path,
                ui_width=width_val,  # settings에서 가져온 가로값
                ui_height=height_val,  # settings에서 가져온 세로값
                steps=steps_val,  # settings에서 가져온 스텝 수
                on_progress=_cb
            )
        except Exception as e:
            self.on_progress(f"❌ 이미지 생성 오류: {e}")
            raise e

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
        [3단계 병합 프로세스]
        1. Visual: concatenate_scene_clips로 영상 트랙 병합 (temp_visual.mp4)
        2. Audio: pydub로 BGM(20%, fadeout) + Voice(타임스탬프) 믹싱 (mixed_audio.wav)
        3. Final: FFmpeg로 자막(.srt) 생성 후 영상+오디오+자막 합성 (final_shopping_video.mp4)
        """
        vpath = Path(video_json_path)
        project_dir = vpath.parent
        clips_dir = project_dir / "clips"
        voice_dir = project_dir / "voice"
        bgm_path = project_dir / "bgm.mp3"

        # 중간 산출물 경로
        temp_visual_path = project_dir / "temp_visual_merged.mp4"
        mixed_audio_path = project_dir / "mixed_audio.wav"
        srt_path = project_dir / "subtitles.srt"
        final_output_path = project_dir / "final_shopping_video.mp4"

        ffmpeg_exe = getattr(settings, "FFMPEG_EXE", "ffmpeg")

        self.on_progress("[Merge] 1/3단계: 영상 트랙 합치는 중...")
        data = load_json(vpath, {})
        scenes = data.get("scenes", [])

        # 1. 영상 트랙 병합 (Visual Only)
        clip_paths = []
        for sc in scenes:
            cpath = clips_dir / f"{sc['id']}.mp4"
            if cpath.exists():
                clip_paths.append(cpath)
            else:
                self.on_progress(f"⚠️ 클립 누락됨: {cpath.name}")

        if not clip_paths:
            self.on_progress("❌ 병합할 클립이 없습니다.")
            return

        try:
            # 기존 함수 활용 (영상만 빠르게 이어붙임)
            concatenate_scene_clips(clip_paths, temp_visual_path, ffmpeg_exe)
        except Exception as e:
            self.on_progress(f"❌ 영상 트랙 병합 실패: {e}")
            return

        # 2. 오디오 믹싱 (BGM + Voice)
        self.on_progress("[Merge] 2/3단계: BGM 및 내레이션 믹싱 중...")
        try:
            total_duration = float(data.get("duration", 0))
            if total_duration <= 0:
                # 메타데이터에 없으면 씬 합계로 계산
                total_duration = sum(float(s.get("end", 0)) - float(s.get("start", 0)) for s in scenes)

            self._mix_audio_with_pydub(scenes, bgm_path, voice_dir, total_duration, mixed_audio_path)
        except Exception as e:
            self.on_progress(f"❌ 오디오 믹싱 실패: {e}")
            return

        # 3. 자막 생성 및 최종 렌더링
        self.on_progress("[Merge] 3/3단계: 자막 생성 및 최종 렌더링...")
        try:
            # 3-1. SRT 파일 생성
            self._create_srt_file(scenes, srt_path)

            # 3-2. FFmpeg 최종 합성 (Visual + Mixed Audio + Subtitles)
            self._finalize_with_ffmpeg(
                ffmpeg_exe,
                temp_visual_path,
                mixed_audio_path,
                srt_path,
                final_output_path
            )

            self.on_progress(f"✅ 최종 병합 완료: {final_output_path.name}")

            # (선택) 임시 파일 정리
            # if temp_visual_path.exists(): os.remove(temp_visual_path)
            # if mixed_audio_path.exists(): os.remove(mixed_audio_path)
            # if srt_path.exists(): os.remove(srt_path)

        except Exception as e:
            self.on_progress(f"❌ 최종 렌더링 실패: {e}")

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

    def _finalize_with_ffmpeg(self, ffmpeg_exe: str, video_path: Path, audio_path: Path, srt_path: Path,
                              out_path: Path):
        """영상, 오디오, 자막을 최종 합성 (재인코딩 발생)"""

        # FFmpeg filter에서 윈도우 경로 역슬래시(\)는 이스케이프가 까다로우므로 슬래시(/)로 변환
        # 자막 경로 처리
        srt_path_str = str(srt_path).replace("\\", "/").replace(":", "\\:")

        # 명령어 구성
        # -shortest: 영상/오디오 중 짧은 쪽에 맞춰 끝냄
        # -c:v libx264: 자막 합성을 위해 비디오 재인코딩
        # -c:a aac: 오디오 인코딩
        # -vf subtitles=...: 자막 필터 (force_style로 폰트 크기/배경 등 스타일 지정 가능)

        # 기본 자막 스타일: 폰트크기 20, 굵게, 흰색 글씨, 검은 테두리, 하단 여백 20
        sub_style = "Fontname=Arial,FontSize=20,Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,MarginV=30"

        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-vf", f"subtitles='{srt_path_str}':force_style='{sub_style}'",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(out_path)
        ]

        # 윈도우에서 subprocess 실행 시 콘솔 창 숨기기
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
            raise RuntimeError(f"FFmpeg Final Render Failed: {result.stderr}")


# -----------------------------------------------------------------------------
# 7. 파이프라인
# -----------------------------------------------------------------------------
class ShoppingShortsPipeline:
    def __init__(self, on_progress: Optional[Callable[[str], None]] = None):
        self.on_progress = on_progress or (lambda msg: None)

    def run_all(
            self,
            product_dir: str | Path,
            product_data: Dict[str, Any],
            options: Optional[BuildOptions] = None,
            build_json: bool = True,
            build_images: bool = True,
            build_movies: bool = True,
            merge: bool = True,
            skip_if_exists: bool = True,
    ) -> Path:
        options = options or BuildOptions()
        vpath = Path(product_dir) / "video_shopping.json"

        builder = ShoppingVideoJsonBuilder(self.on_progress)

        if build_json:
            if not vpath.exists():
                vpath = builder.create_draft(product_dir, product_data, options)
            builder.enrich_video_json(vpath, product_data)

        if build_images:
            img_gen = ShoppingImageGenerator(self.on_progress)
            img_gen.generate_images(vpath, skip_if_exists)

        if build_movies:
            mov_gen = ShoppingMovieGenerator(self.on_progress)
            mov_gen.generate_movies(vpath, skip_if_exists, fps=options.fps)

        if merge:
            mov_gen = ShoppingMovieGenerator(self.on_progress)
            mov_gen.merge_movies(vpath)

        return vpath


def convert_shopping_to_video_json_with_ai(
        project_dir: str,
        ai_client: Any = None,
        fps: int = 30,
        width: int = 1080,
        height: int = 1920,
        steps: int = 20,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None
) -> str:
    """
    [쇼핑->쇼츠 변환 최종판]
    - 원본(video_shopping.json)의 ID가 't_001'이면 그대로 계승합니다.
    - 억지로 인덱스 기준으로 ID를 새로 만들지 않습니다. (삭제/순서변경 대응)
    """

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

    _log("데이터 구조 변환 시작 (ID 계승 모드)...")

    prod = src_data.get("product", {})
    project_name = prod.get("product_name") or src_data.get("project_name", "Shopping Project")
    src_scenes = src_data.get("scenes", [])
    if not src_scenes:
        src_scenes = src_data.get("groups", [])

    new_scenes = []
    current_time = 0.0
    full_lyrics_parts = []

    for idx, sc in enumerate(src_scenes):
        # [핵심 수정]
        # video_shopping.json에 있는 ID를 최우선으로 사용한다.
        # 초안 생성 단계에서 이미 't_001'로 만들어졌으므로 그대로 가져온다.
        original_id = str(sc.get("id", "")).strip()

        if original_id:
            scene_id = original_id
        else:
            # 만약 구버전 데이터라 ID가 없다면 안전장치로 생성
            scene_id = f"t_{idx + 1:03d}"

        # 이미지는 ID와 동일한 이름의 png 파일
        target_img_name = f"{scene_id}.png"

        dur = float(sc.get("seconds") or sc.get("duration") or 4.0)
        start_t = current_time
        end_t = current_time + dur
        current_time = end_t

        narration = str(sc.get("narration") or sc.get("narration_text") or "")
        full_lyrics_parts.append(narration)

        new_scene = {
            "id": scene_id,  # t_001 그대로 유지
            "section": "main",
            "start": round(start_t, 3),
            "end": round(end_t, 3),
            "duration": round(dur, 3),
            "img_file": str(imgs_dir / target_img_name),  # imgs/t_001.png
            "voice_file": sc.get("voice_file") or sc.get("audio_path") or "",
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

    video_data = {
        "title": project_name,
        "duration": round(total_duration, 3),
        "fps": fps,
        "lyrics": full_lyrics,
        "scenes": new_scenes,
        "defaults": {
            "movie": {"fps": fps, "target_fps": fps, "input_fps": fps},
            "image": {"width": width, "height": height, "fps": fps},
            "generator": {"steps": steps}
        },
        "audit": {
            "source": "shopping_converter_v2",
            "converted_at": str(datetime.datetime.now())
        }
    }

    with open(dst_json_path, "w", encoding="utf-8") as f:
        json.dump(video_data, f, indent=2, ensure_ascii=False)

    _log(f"video.json 저장 완료 (ID: {scene_id} 등)")
    _log("AI 상세화 진행...")

    if ai_client:
        try:
            def ask_wrapper(sys_msg, user_msg):
                return ai_client.ask_smart(sys_msg, user_msg, prefer="openai")

            # ID가 t_001 형식이므로 세그먼트 분할 AI가 정상적으로 작동합니다.
            fill_prompt_movie_with_ai(
                str(dst_json_path.parent),
                ask_wrapper,
                log_fn=_log
            )
            _log("✅ AI 상세화(Segments/Prompts) 완료.")
        except Exception as e:
            _log(f"❌ AI 상세화 실패: {e}")

    return str(dst_json_path)
#