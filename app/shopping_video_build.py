# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime
import json
import re
import shutil
import os
import random
import requests
import time
import uuid
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# [수정] 강력한 오디오 측정 함수(audio_duration_sec) 사용
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
        """[1단계] 기획 초안 - BGM 기획 추가 + [수정] 시각적 환각(로고/QR) 방지"""
        p_dir = Path(product_dir)
        vpath = p_dir / "video_shopping.json"

        product_name = product_data.get("product_name", "상품명 없음")
        desc = product_data.get("description") or product_data.get("summary_source") or ""

        self.on_progress(f"[Draft] AI 기획 시작 (상품: {product_name})...")

        # BGM 가이드 (기존 동일)
        bgm_guide = (
            "4. **Background Music (BGM)**: \n"
            "   - Design a prompt for audio generation.\n"
            "   - Format: 'instrumental, background music, [Mood], [Genre], [Instruments], [Tempo], [Energy]'.\n"
            "   - **NO Vocals**: Do not use words like song, singing, voice.\n"
            "   - **NO Model Name**: Do NOT include the word 'Ace-Step' in the output prompt.\n"
            "   - Example: 'instrumental, background music, bright, acoustic pop, guitar, piano, medium tempo, uplifting'."
        )

        # [수정] 시각적 제약사항(Visual Constraints) 강력 추가
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
                    "id": "001",
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
            sid = sc.get("id") or f"{idx + 1:03d}"
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
                "img_file": str(imgs_dir / f"{sid}.png"),
                "movie_file": str(clips_dir / f"{sid}.mp4"),
                "voice_file": str(voice_dir / f"{sid}.wav")
            }
            final_json["scenes"].append(new_scene)

        save_json(vpath, final_json)
        self.on_progress(f"[Draft] 초안 완료. BGM: {final_json['meta'].get('bgm_prompt')}")
        return vpath

    def enrich_video_json(self, video_json_path: str | Path, product_data: Dict[str, Any]) -> Path:
        """
        [3단계] 상세화 (음성 -> BGM -> 영어 프롬프트)
        [수정] Qwen 모델 전용 문법("Subject from image 1...") 강제 적용
        """
        vpath = Path(video_json_path)
        p_dir = vpath.parent
        voice_dir = ensure_dir(p_dir / "voice")
        bgm_path = p_dir / "bgm.mp3"

        data = load_json(vpath, {})
        scenes = data.get("scenes", [])
        meta = data.get("meta", {})

        # ---------------------------------------------------------------------
        # 1. 음성 생성 (기존 유지)
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
        # 2. BGM 생성 (기존 유지)
        # ---------------------------------------------------------------------
        bgm_prompt = meta.get("bgm_prompt", "")
        if not bgm_prompt:
            bgm_prompt = "instrumental, background music, calm, minimal, piano, soft, loopable"
            meta["bgm_prompt"] = bgm_prompt

        if bgm_path.exists() and bgm_path.stat().st_size > 1024:
            self.on_progress(f"[Enrich] 2/3단계: BGM 이미 존재 (스킵).")
        else:
            self.on_progress(f"[Enrich] 2/3단계: BGM 생성 중...")
            success_bgm = generate_bgm_acestep(
                prompt=bgm_prompt,
                out_path=bgm_path,
                duration_sec=total_dur,
                comfy_host=comfy_host,
            )
            if success_bgm:
                self.on_progress("   ✅ BGM 생성 완료!")
            else:
                self.on_progress("   ❌ BGM 생성 실패 (로그 확인).")

        # ---------------------------------------------------------------------
        # 3. 프롬프트 상세화 (문법 대수술)
        # ---------------------------------------------------------------------
        self.on_progress("[Enrich] 3/3단계: 비주얼 프롬프트 고도화 (Qwen 문법 적용)...")

        char_prompt = meta.get("character_prompt", "Young Korean model")
        if "male" == gender:
            gender_kw = "male, man, 1boy"
        else:
            gender_kw = "female, woman, 1girl"

        scene_texts = []
        for sc in scenes:
            sfx_info = f" (SFX: {sc.get('sfx')})" if sc.get("sfx") else ""
            scene_texts.append(f"- Scene {sc['id']} (지문): {sc.get('prompt')}{sfx_info}")

        # [핵심] Qwen 모델이 알아먹는 문법 강제
        sys_p = (
            "You are a ComfyUI Compositing Director using Qwen-Edit.\n"
            "Your task is to generate 2-step prompts. Step 2 requires a VERY SPECIFIC grammar to work.\n\n"
            "**[CRITICAL RULE FOR Step 2]**\n"
            "You MUST explicitly identify the subject in Image 1 and the object in Image 2.\n"
            "**Syntax:** \"[Subject] from image 1 [action] [object] from image 2\"\n"
            "**Examples:**\n"
            "- \"The woman from image 1 holds the object from image 2 in her hand.\"\n"
            "- \"The table from image 1 has the object from image 2 placed on it.\"\n"
            "- \"The man from image 1 is looking at the object from image 2.\"\n\n"
            "**Language:** English Only. No Korean."
        )

        user_p = f"""
        [Prompting Strategy]

        **Context**:
        - Character: "{char_prompt}" ({gender_kw})

        Analyze each scene and apply these rules:

        ### 1. `prompt_img_1` (Canvas)
        - Describe background, lighting, and character pose.
        - **IMPORTANT**: Leave space for the product (e.g., "empty hand", "empty table space").
        - **NO Product Details**: Do not describe the product itself.

        ### 2. `prompt_img_2` (Paint / Qwen)
        - **STRICT GRAMMAR**: You MUST use "from image 1" for the subject/background AND "from image 2" for the product.
        - **Format (Person)**: "The {gender_kw} from image 1 holds the object from image 2 in hand."
        - **Format (Background)**: "The surface from image 1 has the object from image 2 placed on it."
        - **Format (Action)**: "The {gender_kw} from image 1 throws the object from image 2."
        - **NO Adjectives for Product**: Do not say "red bottle". Just say "object from image 2".

        ### 3. `prompt_negative`
        - Standard: "text, watermark, logo, deformed hands, extra fingers, product details".

        ### 4. `prompt_movie`
        - Camera movement description (English).

        [Input Scenarios]
        {chr(10).join(scene_texts)}

        [Output Format (JSON)]
        {{
            "scenes": {{
                "001": {{ "prompt_img_1": "...", "prompt_img_2": "...", "prompt_negative": "...", "prompt_movie": "..." }},
                ...
            }}
        }}
        """

        try:
            resp = self.ai.ask_smart(sys_p, user_p, prefer="openai")
            enriched = self._safe_json_parse(resp)
            en_map = enriched.get("scenes", {})

            if isinstance(en_map, list):
                en_map = {item.get("id", f"t_{i + 1:03d}"): item for i, item in enumerate(en_map)}

            for sc in scenes:
                sid = sc["id"]
                tgt = None
                candidates = [sid, sid.lstrip("0"), f"Scene {sid}", f"t_{sid}", sid.replace("t_", "")]

                for key in candidates:
                    if key in en_map:
                        tgt = en_map[key]
                        break

                if not tgt:
                    for val in en_map.values():
                        if isinstance(val, dict) and str(val.get("id", "")) == sid:
                            tgt = val
                            break

                if tgt:
                    sc["prompt_img_1"] = tgt.get("prompt_img_1", "")
                    sc["prompt_img_2"] = tgt.get("prompt_img_2", "")
                    sc["prompt_negative"] = tgt.get("prompt_negative", "")
                    sc["prompt_movie"] = tgt.get("prompt_movie", "")
                    sc["prompt_img"] = sc["prompt_img_1"]

            data["audit"]["enriched_at"] = _now_str()
            data["audit"]["step"] = "enriched"

            save_json(vpath, data)
            self.on_progress(f"[Enrich] 상세화 완료 (Qwen 문법 적용됨).")

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
                    # self.on_progress("ℹ️ 임시 파일 정리 완료")
                except:
                    pass

    def merge_movies(self, video_json_path: str | Path):
        vpath = Path(video_json_path)
        project_dir = vpath.parent
        clips_dir = project_dir / "clips"

        self.on_progress("[Merge] 영상 합치기...")
        data = load_json(vpath, {})
        clip_paths = []
        for sc in data.get("scenes", []):
            cpath = clips_dir / f"{sc['id']}.mp4"
            if cpath.exists():
                clip_paths.append(cpath)

        if not clip_paths:
            self.on_progress("❌ 병합할 클립 없음")
            return

        out_path = project_dir / "final_shopping_video.mp4"
        ffmpeg_exe = getattr(settings, "FFMPEG_EXE", "ffmpeg")
        concatenate_scene_clips(clip_paths, out_path, ffmpeg_exe)
        self.on_progress(f"✅ 병합 완료: {out_path.name}")


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
    [쇼핑->쇼츠 변환]
    video_shopping.json을 읽어서 Shorts 탭과 호환되는 video.json을 생성합니다.
    UI에서 입력받은 FPS, 해상도, Steps 값을 video.json에 영구 저장합니다.
    """

    def _log(msg: str):
        if on_progress:
            on_progress({"msg": msg})
        print(f"[ShoppingConverter] {msg}")

    proj_path = Path(project_dir)
    src_json_path = proj_path / "video_shopping.json"
    dst_json_path = proj_path / "video.json"

    if not src_json_path.exists():
        raise FileNotFoundError(f"video_shopping.json이 없습니다: {src_json_path}")

    # 1. 쇼핑 데이터 로드
    try:
        with open(src_json_path, "r", encoding="utf-8") as f:
            src_data = json.load(f)
    except Exception as e:
        raise ValueError(f"데이터 로드 실패: {e}")

    _log("데이터 구조 변환 시작...")

    # 2. 기본 정보 매핑
    prod = src_data.get("product", {})
    project_name = prod.get("product_name") or src_data.get("project_name", "Shopping Project")
    src_scenes = src_data.get("scenes", [])
    if not src_scenes:
        src_scenes = src_data.get("groups", [])

    # 3. video.json 씬 리스트 조립
    new_scenes = []
    current_time = 0.0
    full_lyrics_parts = []

    for idx, sc in enumerate(src_scenes):
        # =========================================================================
        # [수정] ID 통일성 유지: 원본 ID가 있으면 그대로 사용, 없으면 t_001 생성
        # 이렇게 해야 이미지 파일명(001.png vs t_001.png) 불일치가 사라짐
        # =========================================================================
        original_id = str(sc.get("id", "")).strip()
        if original_id:
            scene_id = original_id
        else:
            scene_id = f"t_{idx + 1:03d}"

        dur = float(sc.get("seconds") or sc.get("duration") or 4.0)
        start_t = current_time
        end_t = current_time + dur
        current_time = end_t

        narration = str(sc.get("narration") or sc.get("narration_text") or "")
        full_lyrics_parts.append(narration)

        new_scene = {
            "id": scene_id,
            "section": "main",
            "start": round(start_t, 3),
            "end": round(end_t, 3),
            "duration": round(dur, 3),
            "img_file": sc.get("img_file") or sc.get("image_path") or "",
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

    # 4. video.json 뼈대 저장 (UI 설정값 반영)
    video_data = {
        "title": project_name,
        "duration": round(total_duration, 3),
        "fps": fps,  # [UI값] 최상위 FPS 저장
        "lyrics": full_lyrics,
        "scenes": new_scenes,
        "defaults": {
            "movie": {
                "fps": fps,
                "target_fps": fps,
                "input_fps": fps
            },
            "image": {
                "width": width,  # [UI값] 해상도 저장
                "height": height,  # [UI값] 해상도 저장
                "fps": fps
            },
            # [New] 생성 관련 파라미터 저장
            "generator": {
                "steps": steps
            }
        },
        "audit": {
            "source": "shopping_converter_v2",
            "converted_at": str(datetime.datetime.now())
        }
    }

    with open(dst_json_path, "w", encoding="utf-8") as f:
        json.dump(video_data, f, indent=2, ensure_ascii=False)

    _log(f"video.json 저장 완료 (FPS: {fps}, Size: {width}x{height}, Steps: {steps})")
    _log("AI 상세화 진행...")

    # 5. AI 상세화 (세그먼트 생성)
    if ai_client:
        try:
            def ask_wrapper(sys_msg, user_msg):
                return ai_client.ask_smart(sys_msg, user_msg, prefer="openai")

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