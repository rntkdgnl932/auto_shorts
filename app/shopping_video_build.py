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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# 유틸리티 임포트 (외부 의존성 최소화)
from app.utils import (
    AI,
    load_json,
    save_json,
    ensure_dir
)
from app import settings

# [중요] 영상 생성/병합 관련 함수는 video_build.py의 것을 사용
from app.video_build import (
    build_shots_with_i2v,
    concatenate_scene_clips,
)


def _now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _sanitize_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"[\\/:*?\"<>|]", "_", name).strip()
    if len(name) > 120:
        name = name[:120].rstrip("_")
    return name or "untitled"


def _ensure_dirs(product_dir: Path) -> Dict[str, Path]:
    imgs = product_dir / "imgs"
    clips = product_dir / "clips"
    imgs.mkdir(parents=True, exist_ok=True)
    clips.mkdir(parents=True, exist_ok=True)
    return {"imgs": imgs, "clips": clips}


# -----------------------------------------------------------------------------
# [New] ComfyUI 제출 및 대기 함수 (이 파일 전용 독립 구현)
# -----------------------------------------------------------------------------
def _submit_and_wait_local(
        base_url: str,
        graph: dict,
        timeout: int = 900,
        poll: float = 2.0,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None
) -> dict:
    """
    ComfyUI에 워크플로우를 제출하고 완료될 때까지 기다립니다.
    utils.py의 _dlog에 의존하지 않고 독립적으로 동작합니다.
    """
    client_id = str(uuid.uuid4())
    payload = {"prompt": graph, "client_id": client_id}

    # 1. 제출 (/prompt)
    try:
        resp = requests.post(f"{base_url}/prompt", json=payload, timeout=30)
        resp.raise_for_status()
        res_json = resp.json()
        prompt_id = res_json.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"프롬프트 ID를 받지 못했습니다: {res_json}")
    except Exception as e:
        raise RuntimeError(f"ComfyUI 제출 실패: {e}")

    # 2. 대기 (/history)
    start_t = time.time()

    while True:
        elapsed = time.time() - start_t
        if elapsed > timeout:
            raise TimeoutError(f"ComfyUI 시간 초과 ({elapsed:.1f}s)")

        # 히스토리 확인 (완료 여부)
        try:
            h_resp = requests.get(f"{base_url}/history/{prompt_id}", timeout=10)
            if h_resp.status_code == 200:
                h_data = h_resp.json()
                # history에 prompt_id가 키로 존재하면 완료된 것
                if prompt_id in h_data:
                    return h_data[prompt_id]
        except Exception:
            pass  # 일시적 네트워크 오류 등은 무시하고 계속 대기

        # 진행률 로깅 (선택)
        if on_progress and int(elapsed) % 5 == 0:
            # 너무 빈번한 호출 방지
            pass

        time.sleep(poll)


# -----------------------------------------------------------------------------
# 1. Shopping 전용 이미지 생성 함수 (Z-Image 워크플로우 독립 실행)
# -----------------------------------------------------------------------------
def build_shopping_images_z_image(
        video_json_path: str | Path,
        *,
        ui_width: int = 720,
        ui_height: int = 1280,
        steps: int = 28,
        timeout_sec: int = 900,
        poll_sec: float = 2.0,
        workflow_path: str | Path | None = None,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Path]:
    """
    shopping_video_build.py 내부에 정의된 쇼핑 전용 이미지 생성 함수.
    - video_shopping.json을 읽음
    - Z-Image-lora.json 워크플로우 사용
    - PreviewImage -> SaveImage 자동 변환 처리 포함
    - _submit_and_wait_local 사용으로 외부 의존성 제거
    """

    # 1. 경로 설정 및 JSON 로드
    vpath = Path(video_json_path).resolve()
    product_dir = vpath.parent
    imgs_dir = ensure_dir(product_dir / "imgs")

    if not vpath.exists():
        raise FileNotFoundError(f"JSON 파일이 없습니다: {vpath}")

    video_doc = load_json(vpath, {}) or {}
    scenes = video_doc.get("scenes", [])
    if not scenes:
        if on_progress:
            try:
                on_progress({"msg": "⚠ 생성할 씬(scenes)이 없습니다."})
            except:
                pass
        return []

    # 2. 워크플로우 로드
    if workflow_path:
        wf_path = Path(workflow_path)
    else:
        # 기본값: app/jsons/Z-Image-lora.json
        wf_path = Path(settings.JSONS_DIR) / "Z-Image-lora.json"
        if not wf_path.exists():
            # 폴백 경로 체크
            fallback = Path(r"C:\my_games\shorts_make\app\jsons\Z-Image-lora.json")
            if fallback.exists():
                wf_path = fallback
            else:
                raise FileNotFoundError(f"워크플로우 파일을 찾을 수 없습니다: {wf_path}")

    try:
        with open(wf_path, "r", encoding="utf-8") as f:
            graph_origin = json.load(f)
    except Exception as e:
        raise RuntimeError(f"워크플로우 로드 실패: {e}")

    # 3. ComfyUI 설정
    comfy_host = getattr(settings, "COMFY_HOST", "http://127.0.0.1:8188").rstrip("/")
    base_url = comfy_host

    # 진행률 알림 헬퍼
    def _notify(msg):
        if on_progress:
            try:
                on_progress({"msg": msg})
            except:
                pass

    _notify(f"[Img] 워크플로우: {wf_path.name}")
    _notify(f"[Img] 대상 씬: {len(scenes)}개")

    created_files = []

    # 4. 씬 루프
    for i, sc in enumerate(scenes):
        scene_id = sc.get("id", f"{i:03d}")

        # 목표 파일 경로
        target_file_name = f"{scene_id}.png"
        target_path = imgs_dir / target_file_name

        # 이미 존재하면 스킵 (video.json에 기록된 경로 혹은 파일 존재 여부 확인)
        existing_file = sc.get("img_file")
        if existing_file and Path(existing_file).exists():
            _notify(f"[Img] 스킵(이미 존재): {scene_id}")
            continue
        if target_path.exists() and target_path.stat().st_size > 0:
            _notify(f"[Img] 스킵(파일 있음): {target_path.name}")
            # JSON 업데이트
            sc["img_file"] = str(target_path)
            continue

        _notify(f"[Img] 생성 시작: {scene_id} ...")

        # 워크플로우 복제
        graph = json.loads(json.dumps(graph_origin))

        # (A) 프롬프트 주입
        # 쇼핑 데이터는 'prompt_img' 혹은 'prompt'를 사용
        prompt_text = sc.get("prompt_img") or sc.get("prompt") or ""
        neg_text = sc.get("prompt_negative") or ""

        # 노드 찾기 및 값 주입
        for nid, node in graph.items():
            ctype = node.get("class_type", "")
            inputs = node.get("inputs", {})
            meta_title = str(node.get("_meta", {}).get("title", "")).lower()

            # 텍스트 입력 노드 (CLIPTextEncode)
            if ctype == "CLIPTextEncode":
                # Z-Image-lora.json 구조상 6번이 Positive, 92번이 Negative일 가능성이 높음
                if nid == "6":
                    inputs["text"] = prompt_text
                elif nid == "92" or "negative" in meta_title:
                    inputs["text"] = neg_text
                elif "positive" in meta_title:
                    inputs["text"] = prompt_text

            # (B) 해상도 주입 (EmptySD3LatentImage or EmptyLatentImage)
            if "LatentImage" in ctype:
                if "width" in inputs: inputs["width"] = int(ui_width)
                if "height" in inputs: inputs["height"] = int(ui_height)

            # (C) 시드 주입 (KSampler)
            if ctype == "KSampler" and "seed" in inputs:
                inputs["seed"] = random.randint(1, 9999999999)
                if "steps" in inputs: inputs["steps"] = int(steps)

        # (D) 저장 노드 처리 (PreviewImage -> SaveImage 변환)
        save_node_found = False
        # 딕셔너리 크기가 변하지 않도록 list(items) 사용
        for nid, node in list(graph.items()):
            ctype = node.get("class_type", "")

            if ctype == "SaveImage":
                save_node_found = True
                node["inputs"]["filename_prefix"] = "ShopImg"

            elif ctype == "PreviewImage":
                # PreviewImage를 SaveImage로 변환
                node["class_type"] = "SaveImage"
                node.setdefault("inputs", {})["filename_prefix"] = "ShopImg"
                save_node_found = True

        if not save_node_found:
            _notify("⚠ 저장 노드(SaveImage)를 찾을 수 없어 생성 실패 가능성 있음.")

        # 5. ComfyUI 요청 및 대기 (로컬 함수 사용)
        try:
            # 여기서 _submit_and_wait_local 사용 -> _dlog 에러 방지
            res = _submit_and_wait_local(
                base_url,
                graph,
                timeout=timeout_sec,
                poll=poll_sec,
                on_progress=on_progress
            )

            # 6. 결과 다운로드
            outputs = res.get("outputs", {})
            file_saved = False

            for nid, out_data in outputs.items():
                images = out_data.get("images", [])
                for img_info in images:
                    fname = img_info.get("filename")
                    sfolder = img_info.get("subfolder", "")
                    itype = img_info.get("type", "output")

                    # 이미지 다운로드
                    params = {"filename": fname, "subfolder": sfolder, "type": itype}
                    resp = requests.get(f"{base_url}/view", params=params)

                    if resp.status_code == 200:
                        with open(target_path, "wb") as f:
                            f.write(resp.content)
                        file_saved = True
                        break  # 하나만 저장하면 됨
                if file_saved:
                    break

            if file_saved:
                _notify(f"[Img] 저장 완료: {target_file_name}")
                sc["img_file"] = str(target_path)
                created_files.append(target_path)
            else:
                _notify(f"[Img] ❌ 생성 실패 (결과물 못 찾음): {scene_id}")

        except Exception as e:
            _notify(f"[Img] ❌ 에러 발생: {e}")
            continue

    # 7. JSON 저장 (이미지 경로 업데이트)
    save_json(vpath, video_doc)
    _notify(f"[Img] 전체 완료. 생성된 이미지: {len(created_files)}장")
    return created_files


# -----------------------------------------------------------------------------
# 2. 클래스 정의
# -----------------------------------------------------------------------------

@dataclass
class BuildOptions:
    scene_count: int = 6
    style: str = "news_hook"
    hook_level: int = 3
    fps: int = 24
    allow_fallback_rule: bool = True


class ShoppingImageGenerator:
    """
    쇼핑 전용 이미지 생성기
    - build_shopping_images_z_image 함수를 호출합니다.
    """

    def __init__(self, on_progress: Optional[Callable[[str], None]] = None):
        self.on_progress = on_progress or (lambda msg: None)

    def generate_images(self, video_json_path: str | Path, skip_if_exists: bool = True) -> None:
        # 진행률 콜백 래퍼
        def _prog_cb(d: dict):
            msg = d.get("msg", "")
            if msg:
                self.on_progress(msg)

        try:
            build_shopping_images_z_image(
                video_json_path=video_json_path,
                ui_width=720,
                ui_height=1280,
                steps=28,
                on_progress=_prog_cb
            )
        except Exception as e:
            self.on_progress(f"❌ 이미지 생성 중 오류: {e}")
            raise e


class ShoppingMovieGenerator:
    """
    쇼핑 전용 영상(I2V) 생성 및 병합기
    - video_build.build_shots_with_i2v는 폴더 내 'video.json'을 자동으로 찾습니다.
    - 따라서 'video_shopping.json'을 'video.json'으로 잠시 복사해두고 작업을 수행합니다.
    - [중요] duration이 없거나 0인 경우 강제로 4.0초를 할당하여 스킵되지 않도록 함
    """

    def __init__(self, on_progress: Optional[Callable[[str], None]] = None):
        self.on_progress = on_progress or (lambda msg: None)

    def generate_movies(self, video_json_path: str | Path, skip_if_exists: bool = True, fps: int = 24) -> None:
        vpath = Path(video_json_path)
        project_dir = vpath.parent

        # 호환성을 위한 임시 파일 경로 (video.json)
        temp_video_json = project_dir / "video.json"

        self.on_progress(f"[Movie] 준비: {vpath.name} -> video.json 복사 (호환성 확보)")

        try:
            # 1. JSON 로드
            data = load_json(vpath, {})

            # 2. duration 보정 (없으면 4초로 강제 설정)
            scenes = data.get("scenes", [])
            for sc in scenes:
                dur = float(sc.get("duration", 0))
                if dur <= 0:
                    sc["duration"] = 4.0

            # 3. video.json으로 저장 (shutil.copy 대신 수정된 데이터 저장)
            save_json(temp_video_json, data)

            # 4. 진행률 콜백 래퍼
            def _prog_cb(d: dict):
                msg = d.get("msg", "")
                if msg:
                    self.on_progress(msg)

            # 5. I2V 실행 (video_build.py 함수 이용)
            self.on_progress(f"[Movie] 영상 생성(I2V) 시작 (FPS: {fps})...")

            build_shots_with_i2v(
                project_dir=str(project_dir),
                total_frames=0,
                ui_fps=fps,
                on_progress=_prog_cb
            )
            self.on_progress("[Movie] 영상 생성 완료.")

        except Exception as e:
            self.on_progress(f"❌ 영상 생성 중 오류: {e}")
            raise e
        finally:
            # 6. 임시 파일 정리
            if temp_video_json.exists():
                try:
                    os.remove(temp_video_json)
                    self.on_progress("[Movie] 임시 video.json 정리 완료.")
                except:
                    pass

    def merge_movies(self, video_json_path: str | Path):
        vpath = Path(video_json_path)
        project_dir = vpath.parent
        clips_dir = project_dir / "clips"

        try:
            data = load_json(vpath, {})
            scenes = data.get("scenes", [])
            if not scenes:
                self.on_progress("⚠ 합칠 씬이 없습니다 (JSON에 scenes 비어있음).")
                return

            clip_paths = []
            for sc in scenes:
                sid = sc.get("id")
                cpath = clips_dir / f"{sid}.mp4"
                if cpath.exists() and cpath.stat().st_size > 0:
                    clip_paths.append(cpath)
                else:
                    self.on_progress(f"⚠ 클립 누락(스킵): {sid}.mp4")

            if not clip_paths:
                self.on_progress("❌ 병합할 유효한 클립이 하나도 없습니다.")
                return

            out_path = project_dir / "final_shopping_video.mp4"
            self.on_progress(f"[Merge] {len(clip_paths)}개 클립 병합 시작 -> {out_path.name}")

            ffmpeg_exe = getattr(settings, "FFMPEG_EXE", "ffmpeg")

            concatenate_scene_clips(
                clip_paths=clip_paths,
                out_path=out_path,
                ffmpeg_exe=ffmpeg_exe
            )
            self.on_progress(f"✅ 병합 완료: {out_path}")

        except Exception as e:
            self.on_progress(f"❌ 병합 중 오류: {e}")
            raise e


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

        # ShoppingVideoJsonBuilder는 같은 파일 내에 있거나 순환 참조를 피해 임포트 필요
        from app.shopping_video_build import ShoppingVideoJsonBuilder

        vpath = Path(product_dir) / "video_shopping.json"

        if build_json:
            self.on_progress("[Pipeline] 1단계: JSON 시나리오 생성...")
            builder = ShoppingVideoJsonBuilder(on_progress=self.on_progress)
            vpath = builder.build(product_dir=product_dir, product_data=product_data, options=options)

        if build_images:
            self.on_progress("[Pipeline] 2단계: 이미지 생성...")
            img_gen = ShoppingImageGenerator(on_progress=self.on_progress)
            img_gen.generate_images(vpath, skip_if_exists=skip_if_exists)

        if build_movies:
            self.on_progress("[Pipeline] 3단계: 영상 생성 (I2V)...")
            mov_gen = ShoppingMovieGenerator(on_progress=self.on_progress)
            mov_gen.generate_movies(vpath, skip_if_exists=skip_if_exists, fps=int(options.fps))

        if merge:
            self.on_progress("[Pipeline] 4단계: 영상 합치기...")
            mov_gen = ShoppingMovieGenerator(on_progress=self.on_progress)
            mov_gen.merge_movies(vpath)

        self.on_progress("[Pipeline] 전체 작업 완료!")
        return vpath


# -----------------------------------------------------------------------------
# 3. JSON 빌더 관련 클래스 (기존 코드 유지용)
# -----------------------------------------------------------------------------
@dataclass
class VideoShoppingBuildInput:
    product_name: str
    product_price: str
    summary_text: str
    image_paths: List[str]


class ShoppingVideoJsonBuilder:
    def __init__(self, on_progress: Optional[Callable[[str], None]] = None):
        self.on_progress = on_progress or (lambda msg: None)

    def build(self, product_dir: str | Path, product_data: Dict[str, Any], options: BuildOptions) -> Path:
        self.on_progress("[JSON] (기존 로직 수행 가정) video_shopping.json 생성 중...")
        vpath = Path(product_dir) / "video_shopping.json"

        # [수정] 더미 데이터 생성 시 duration을 명시적으로 넣어줍니다.
        if not vpath.exists():
            dummy_data = {
                "scenes": [
                    {"id": "001", "prompt": "test scene 1", "duration": 5.0},
                    {"id": "002", "prompt": "test scene 2", "duration": 5.0},
                    {"id": "003", "prompt": "test scene 3", "duration": 5.0},
                    {"id": "004", "prompt": "test scene 4", "duration": 5.0},
                    {"id": "005", "prompt": "test scene 5", "duration": 5.0},
                    {"id": "006", "prompt": "test scene 6", "duration": 5.0}
                ]
            }
            save_json(vpath, dummy_data)
        return vpath