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
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# 유틸리티 임포트
from app.utils import (
    AI,
    load_json,
    save_json,
    ensure_dir,
    audio_duration_sec
)
from app import settings

# 영상 생성/병합 관련 함수
from app.video_build import (
    build_shots_with_i2v,
    concatenate_scene_clips,
)


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
def generate_tts_zonos(
        text: str,
        out_path: Path,
        ref_audio: Path,
        comfy_host: str = "http://127.0.0.1:8188"
) -> bool:
    """
    Zonos 워크플로우(who_voice.json)를 사용하여 TTS 생성
    """
    if not text:
        return False

    wf_path = Path(settings.JSONS_DIR) / "who_voice.json"
    if not wf_path.exists():
        # 폴백 경로
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

    # 참조 오디오 준비
    if not ref_audio.exists():
        print(f"❌ 참조 오디오 없음: {ref_audio}")
        return False

    comfy_input_dir = Path(settings.COMFY_INPUT_DIR)
    comfy_input_dir.mkdir(parents=True, exist_ok=True)

    # 중복 방지 파일명
    ref_copy_name = f"ref_{uuid.uuid4().hex[:8]}{ref_audio.suffix}"
    dst_ref = comfy_input_dir / ref_copy_name
    shutil.copy2(ref_audio, dst_ref)

    # 노드 값 주입 (who_voice.json 구조 가정)
    # Node 24: Zonos Generate, Node 12: Load Audio
    if "24" in graph:
        graph["24"]["inputs"]["speech"] = text
        graph["24"]["inputs"]["seed"] = random.randint(1, 2 ** 32)
    else:
        # ID가 다를 경우 class_type으로 찾기
        for nid, node in graph.items():
            if "Zonos" in node.get("class_type", ""):
                node["inputs"]["speech"] = text
                node["inputs"]["seed"] = random.randint(1, 2 ** 32)
                break

    if "12" in graph:
        graph["12"]["inputs"]["audio"] = ref_copy_name
    else:
        for nid, node in graph.items():
            if node.get("class_type") == "LoadAudio":
                node["inputs"]["audio"] = ref_copy_name
                break

    # 실행 및 다운로드
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
# 2. 이미지 생성 함수 (Z-Image)
# -----------------------------------------------------------------------------
def build_shopping_images_z_image(
        video_json_path: str | Path,
        *,
        ui_width: int = 720,
        ui_height: int = 1280,
        steps: int = 28,
        on_progress: Optional[Callable[[Dict], None]] = None
) -> List[Path]:
    vpath = Path(video_json_path)
    product_dir = vpath.parent
    imgs_dir = ensure_dir(product_dir / "imgs")

    video_doc = load_json(vpath, {})
    scenes = video_doc.get("scenes", [])

    wf_path = Path(settings.JSONS_DIR) / "Z-Image-lora.json"
    if not wf_path.exists():
        wf_path = Path(r"C:\my_games\shorts_make\app\jsons\Z-Image-lora.json")

    if not wf_path.exists():
        if on_progress: on_progress({"msg": f"❌ 워크플로우 파일 없음: {wf_path}"})
        return []

    with open(wf_path, "r", encoding="utf-8") as f:
        graph_origin = json.load(f)

    comfy_host = getattr(settings, "COMFY_HOST", "http://127.0.0.1:8188").rstrip("/")
    created = []

    for sc in scenes:
        sid = sc.get("id")
        target_file = imgs_dir / f"{sid}.png"

        # 파일 있으면 스킵
        if target_file.exists() and target_file.stat().st_size > 0:
            if on_progress: on_progress({"msg": f"[Img] 스킵(존재): {sid}"})
            sc["img_file"] = str(target_file)
            continue

        prompt = sc.get("prompt_img") or sc.get("prompt", "")
        neg = sc.get("prompt_negative", "")

        if not prompt:
            if on_progress: on_progress({"msg": f"[Img] 프롬프트 없음(스킵): {sid}"})
            continue

        if on_progress: on_progress({"msg": f"[Img] 생성 요청: {sid}..."})

        graph = json.loads(json.dumps(graph_origin))

        for nid, node in graph.items():
            ctype = node.get("class_type", "")
            inputs = node.get("inputs", {})
            title = str(node.get("_meta", {}).get("title", "")).lower()

            if ctype == "CLIPTextEncode":
                if nid == "6":
                    inputs["text"] = prompt
                elif nid == "92" or "negative" in title:
                    inputs["text"] = neg
                elif "positive" in title:
                    inputs["text"] = prompt

            if "LatentImage" in ctype:
                if "width" in inputs: inputs["width"] = ui_width
                if "height" in inputs: inputs["height"] = ui_height

            if ctype == "KSampler" and "seed" in inputs:
                inputs["seed"] = random.randint(1, 10 ** 9)
                if "steps" in inputs: inputs["steps"] = steps

            if ctype == "PreviewImage":
                node["class_type"] = "SaveImage"
                node.setdefault("inputs", {})["filename_prefix"] = "ShopImg"

        try:
            res = _submit_and_wait_local(comfy_host, graph, on_progress=on_progress)
            outputs = res.get("outputs", {})
            for _, out_d in outputs.items():
                for img in out_d.get("images", []):
                    fname = img["filename"]
                    resp = requests.get(f"{comfy_host}/view", params={"filename": fname, "type": img["type"]})
                    with open(target_file, "wb") as f:
                        f.write(resp.content)
                    sc["img_file"] = str(target_file)
                    created.append(target_file)
                    break
                break
        except Exception as e:
            if on_progress: on_progress({"msg": f"❌ 생성 에러({sid}): {e}"})

    save_json(vpath, video_doc)
    return created


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
# 4. JSON 빌더 (AI 기획 + 음성 생성 + 프롬프트 상세화)
# -----------------------------------------------------------------------------
class ShoppingVideoJsonBuilder:
    def __init__(self, on_progress: Optional[Callable[[str], None]] = None):
        self.on_progress = on_progress or (lambda msg: None)
        self.ai = AI()

    def create_draft(self, product_dir: str | Path, product_data: Dict[str, Any], options: BuildOptions) -> Path:
        """[1단계] 기획 초안 (성별 판단 및 4요소 시나리오)"""
        p_dir = Path(product_dir)
        vpath = p_dir / "video_shopping.json"

        product_name = product_data.get("product_name", "상품명 없음")
        desc = product_data.get("description") or product_data.get("summary_source") or ""

        self.on_progress(f"[Draft] AI 기획 시작 (상품: {product_name})...")

        system_prompt = (
            "당신은 숏폼 커머스 영상 기획 전문가입니다. "
            "상품 정보를 분석하여 **내레이션 성우의 성별(male/female)**을 결정하고, "
            "시각(화면), 청각(내레이션), 자막이 유기적으로 연결된 기획안을 작성하세요.\n"
            "결과는 오직 **JSON 포맷**으로만 출력해야 합니다."
        )

        user_prompt = f"""
        [상품 정보]
        - 상품명: {product_name}
        - 설명: {desc}

        [제작 가이드]
        1. 총 장면 수: {options.scene_count}개
        2. 스타일: {options.style}
        3. 후크 강도: {options.hook_level}/5

        [성별 판단 기준]
        - 여성(female): 뷰티, 패션, 육아, 주방, 감성, 부드러운 톤
        - 남성(male): IT, 자동차, 공구, 운동, 신뢰/뉴스 톤, 웅장함

        [출력 포맷 (JSON)]
        {{
            "meta": {{ 
                "title": "영상 제목", 
                "voice_gender": "male 또는 female", 
                "tone": "전체적인 톤앤매너" 
            }},
            "scenes": [
                {{
                    "id": "001",
                    "banner": "상단 배너 문구 (없으면 null)",
                    "prompt": "화면 묘사 (AI가 그림을 그릴 수 있게 구체적으로)",
                    "narration": "성우 내레이션 대본 (구어체)",
                    "subtitle": "화면 하단 핵심 자막"
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

        # JSON 구조 조립
        final_json = {
            "schema": "shopping_shorts_v2",
            "style": options.style,
            "product": product_data,
            "meta": data.get("meta", {}),
            "defaults": {"image": {"width": 720, "height": 1280}, "movie": {"fps": options.fps}},
            "audit": {"created_at": _now_str(), "step": "draft"},
            "scenes": []
        }

        # 폴더 준비
        imgs_dir = p_dir / "imgs"
        clips_dir = p_dir / "clips"
        voice_dir = p_dir / "voice"

        imgs_dir.mkdir(parents=True, exist_ok=True)
        clips_dir.mkdir(parents=True, exist_ok=True)
        voice_dir.mkdir(parents=True, exist_ok=True)

        for idx, sc in enumerate(data.get("scenes", [])):
            sid = sc.get("id") or f"{idx + 1:03d}"
            new_scene = {
                "id": sid,
                "banner": sc.get("banner"),
                "prompt": sc.get("prompt", ""),
                "narration": sc.get("narration", ""),
                "subtitle": sc.get("subtitle", ""),
                "seconds": 0,  # 상세화 단계에서 채움
                "prompt_img": "",
                "prompt_movie": "",
                "prompt_negative": "",
                "img_file": str(imgs_dir / f"{sid}.png"),
                "movie_file": str(clips_dir / f"{sid}.mp4"),
                "voice_file": str(voice_dir / f"{sid}.wav")
            }
            final_json["scenes"].append(new_scene)

        save_json(vpath, final_json)
        self.on_progress(f"[Draft] 초안 완료. 성별: {final_json['meta'].get('voice_gender', 'unknown')}")
        return vpath

    def enrich_video_json(self, video_json_path: str | Path, product_data: Dict[str, Any]) -> Path:
        """
        [3단계] 상세화 (Enrich)
        1. 내레이션 생성 (TTS) -> seconds 확정
        2. 총 시간 계산 -> meta['total_duration']
        3. 비주얼 프롬프트 및 BGM 태그 생성 (AI)
        """
        vpath = Path(video_json_path)
        p_dir = vpath.parent
        voice_dir = ensure_dir(p_dir / "voice")

        data = load_json(vpath, {})
        scenes = data.get("scenes", [])

        # 1. 성별에 따른 참조 음성 선택
        gender = data.get("meta", {}).get("voice_gender", "female").lower()
        if gender == "male":
            ref_voice = Path(r"C:\my_games\shorts_make\voice\남자성우1.mp3")
        else:
            ref_voice = Path(r"C:\my_games\shorts_make\voice\꼬꼬 음성.m4a")

        if not ref_voice.exists():
            self.on_progress(f"⚠️ 참조 음성 파일 없음: {ref_voice} (기본값 사용 주의)")

        self.on_progress(f"[Enrich] 1/2단계: 음성 생성 ({gender}) 및 시간 측정...")
        comfy_host = getattr(settings, "COMFY_HOST", "http://127.0.0.1:8188")

        total_dur = 0.0

        for sc in scenes:
            sid = sc["id"]
            narr = sc.get("narration", "").strip()
            v_path = Path(sc.get("voice_file") or str(voice_dir / f"{sid}.wav"))

            # 내레이션 없으면 기본 3초
            if not narr:
                sc["seconds"] = 3
                total_dur += 3
                continue

            # 파일이 없거나 0바이트면 생성 시도
            if not v_path.exists() or v_path.stat().st_size == 0:
                self.on_progress(f"   🎙️ Scene {sid} 음성 생성...")
                success = generate_tts_zonos(narr, v_path, ref_voice, comfy_host)
                if not success:
                    sc["seconds"] = 4
                    total_dur += 4
                    continue

            # 길이 측정
            dur = audio_duration_sec(v_path)
            if dur > 0:
                sc["seconds"] = round(dur + 0.5, 2)  # 0.5초 여유
            else:
                sc["seconds"] = 4

            sc["voice_file"] = str(v_path)
            total_dur += sc["seconds"]

        # 총 시간 저장
        data.setdefault("meta", {})["total_duration"] = round(total_dur, 2)
        save_json(vpath, data)

        # 2. 프롬프트 및 BGM 태그 상세화
        self.on_progress("[Enrich] 2/2단계: 비주얼 프롬프트 및 BGM 태그 작성...")

        scene_texts = []
        for sc in scenes:
            scene_texts.append(f"- Scene {sc['id']} ({sc['seconds']}초): {sc.get('prompt')}")

        sys_p = (
            "당신은 ComfyUI 영상 및 오디오 기획 전문가입니다. "
            "주어진 시나리오를 바탕으로 **비주얼 프롬프트**와 **Ace-Step BGM 태그**를 작성하세요.\n"
            "**시간(seconds)은 이미 확정되었으니 수정하지 마세요.**\n"
            "반드시 JSON 포맷으로 출력해야 합니다."
        )

        user_p = f"""
        [작업 요청]
        1. 각 장면의 `prompt_img` (실사, 8k, 구체적 묘사)와 `prompt_movie` (카메라 무빙)를 작성하세요.
        2. 전체 영상 분위기에 어울리는 **BGM 태그(bgm_tags)**를 3~5개 선정하여 `meta`에 추가하세요.
           (선택지: electronic, pop, rock, cinematic, emotional, fast tempo, slow tempo, piano, happy, dark 등)

        [입력 시나리오]
        {chr(10).join(scene_texts)}

        [출력 포맷 (JSON)]
        {{
            "meta": {{
                "bgm_tags": ["tag1", "tag2", "tag3"]
            }},
            "scenes": [
                {{
                    "id": "001",
                    "prompt_img": "...",
                    "prompt_movie": "...",
                    "prompt_negative": "..."
                }},
                ...
            ]
        }}
        """

        try:
            resp = self.ai.ask_smart(sys_p, user_p, prefer="openai")
            enriched = self._safe_json_parse(resp)
        except Exception as e:
            self.on_progress(f"❌ 상세화 AI 실패: {e}")
            raise

        # 결과 병합
        if "meta" in enriched and "bgm_tags" in enriched["meta"]:
            data["meta"]["bgm_tags"] = enriched["meta"]["bgm_tags"]

        en_map = {s["id"]: s for s in enriched.get("scenes", [])}
        for sc in scenes:
            if sc["id"] in en_map:
                tgt = en_map[sc["id"]]
                sc["prompt_img"] = tgt.get("prompt_img", "")
                sc["prompt_movie"] = tgt.get("prompt_movie", "")
                sc["prompt_negative"] = tgt.get("prompt_negative", "")
                # seconds는 덮어쓰지 않음 (오디오 기준 유지)

        data["audit"]["enriched_at"] = _now_str()
        data["audit"]["step"] = "enriched"

        save_json(vpath, data)
        self.on_progress(f"[Enrich] 상세화 완료. (총 길이: {total_dur:.1f}초)")
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

        try:
            build_shopping_images_z_image(
                video_json_path=video_json_path,
                ui_width=720,
                ui_height=1280,
                steps=28,
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
        vpath = Path(video_json_path)
        project_dir = vpath.parent
        temp_video_json = project_dir / "video.json"

        self.on_progress(f"[Movie] I2V 준비: {vpath.name} -> video.json 복사")

        data = load_json(vpath, {})
        # duration 보정 (없으면 seconds 사용)
        for sc in data.get("scenes", []):
            if float(sc.get("duration", 0)) <= 0:
                sc["duration"] = float(sc.get("seconds", 4.0))

        save_json(temp_video_json, data)

        def _cb(d):
            self.on_progress(d.get("msg", ""))

        try:
            build_shots_with_i2v(str(project_dir), total_frames=0, ui_fps=fps, on_progress=_cb)
            self.on_progress("[Movie] 생성 완료")
        finally:
            if temp_video_json.exists():
                try:
                    os.remove(temp_video_json)
                except:
                    pass

    def merge_movies(self, video_json_path: str | Path):
        vpath = Path(video_json_path)
        project_dir = vpath.parent
        clips_dir = project_dir / "clips"

        # ※ 오디오 병합(Muxing) 기능은 concatenate_scene_clips 함수 내부 혹은 별도 로직 필요.
        # 현재는 비디오만 병합하는 것으로 호출.
        # (추후 4단계 병합 로직 개선 시, 여기서 voice_file들을 함께 넘겨서 처리해야 함)

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
            # 1. 초안
            if not vpath.exists():
                vpath = builder.create_draft(product_dir, product_data, options)
            # 2. 상세화 (음성+프롬프트+BGM태그)
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