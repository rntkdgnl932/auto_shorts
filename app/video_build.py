# -*- coding: utf-8 -*-
# i2v 분할/실행/합치기(교차 페이드) + 누락 이미지 생성 (ComfyUI 연동)
from __future__ import annotations

from pathlib import Path as _Path
import os

# ── 유연 임포트 ─────────────────────────────────────────────────────────────
from app.utils import ensure_dir, load_json
from app.settings import BASE_DIR, I2V_WORKFLOW, FFMPEG_EXE, USE_HWACCEL, FINAL_OUT, COMFY_HOST
# music_gen에 있는 견고한 함수들을 우선 재사용 (가능할 때)
from app.audio_sync import _submit_and_wait as _submit_and_wait_comfy_func
try:
    from app.audio_sync import (
        _http_get as _http_get_audio,
        _load_workflow_graph as _load_workflow_graph_audio,
        _find_nodes_by_class_contains as _find_nodes_by_class_contains_audio,
    )
except Exception:  # 단독 실행/상대 경로일 수도 있으니 폴백 제공
    _http_get_audio = None
    _load_workflow_graph_audio = None
    _find_nodes_by_class_contains_audio = None
import math
import json as json_mod
import time
import random
from typing import Dict, List, Tuple, Optional, Any, Callable, Union, SupportsFloat, SupportsIndex
from textwrap import dedent
import re
import json

from app.utils import (
    load_json as _load_json_func,
    ensure_dir as _ensure_dir_func,
)

from app.settings import JSONS_DIR
from pathlib import Path


import json as _json

import json as _json_loader
import shutil
import requests
import random as _img_seed_random
from app import settings as settings_obj

from app.settings import CHARACTER_DIR, COMFY_INPUT_DIR, I2V_CHUNK_BASE_FRAMES, I2V_OVERLAP_FRAMES, I2V_PAD_TAIL_FRAMES

from app.utils import load_json, save_json


WF_T2I    = JSONS_DIR / "nunchaku_t2i.json"
WF_SWAP_1 = JSONS_DIR / "nunchaku-t2i_swap_1.json"
WF_SWAP_2 = JSONS_DIR / "nunchaku-t2i_swap_2.json"
WF_SWAP_3 = JSONS_DIR / "nunchaku-t2i_swap_3.json"  # 3명은 옵션
WF_LIPSYNC = JSONS_DIR / "wanvideo_I2V_InfiniteTalk_song.json"

CHUNK_BASE_FRAMES = I2V_CHUNK_BASE_FRAMES
OVERLAP_FRAMES = I2V_OVERLAP_FRAMES
PAD_TAIL_FRAMES = I2V_PAD_TAIL_FRAMES

# ── 공용 HTTP 유틸 ──────────────────────────────────────────────────────────
_KOR_KEEP = re.compile(r"[^가-힣ㄱ-ㅎㅏ-ㅣ .,!?~…·\-_/]+")

def _http_get(base: str, path: str, *, timeout: int = 30, params: Optional[dict] = None) -> requests.Response:
    """
    music_gen의 _http_get을 우선 재사용, 없으면 폴백으로 직접 호출
    """
    if _http_get_audio:
        return _http_get_audio(base, path, timeout=timeout, params=params)
    return requests.get(base.rstrip("/") + path, params=params or {}, timeout=timeout)

def _http_post_json(base: str, path: str, payload: dict, *, timeout: int = 30) -> requests.Response:
    return requests.post(base.rstrip("/") + path, json=payload, timeout=timeout)

def _probe_server(base: str, timeout: int = 3) -> bool:
    for p in ("/view", "/history"):
        try:
            r = requests.get(base.rstrip("/") + p, timeout=timeout)
            if r.status_code == 200:
                return True
        except Exception:
            pass
    return False

def _choose_host() -> str:
    base = str(COMFY_HOST).rstrip("/")
    return base

# ── 워크플로 로더(폴백) ─────────────────────────────────────────────────────
def _load_workflow_graph(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _find_nodes_by_class_contains(graph: dict, needle: str) -> List[Tuple[str, dict]]:
    if _find_nodes_by_class_contains_audio:
        return _find_nodes_by_class_contains_audio(graph, needle)
    needle = (needle or "").lower()
    out: List[Tuple[str, dict]] = []
    for nid, node in (graph or {}).items():
        ct = str(node.get("class_type", "")).lower()
        if needle and needle in ct:
            out.append((str(nid), node))
    return out



# [수정] 오디오 자르기 헬퍼 함수 (정확도 우선 모드 - Output Seeking)
def _slice_audio_segment(
        src_audio: Path,
        start_sec: float,
        end_sec: float,
        out_audio: Path,
        ffmpeg_exe: str
) -> bool:
    """
    [정확도 모드] ffmpeg Output Seeking 사용
    - -i (입력)를 먼저 두고 -ss (시작 시간)를 나중에 둡니다.
    - 속도는 아주 조금 느려지지만, 밀림 현상 없이 정확한 시간을 잘라냅니다.
    """
    duration = max(0.1, end_sec - start_sec)

    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(src_audio),  # [변경 1] 입력 파일을 먼저 부르고
        "-ss", f"{start_sec:.6f}",  # [변경 2] 그 다음에 시간을 찾습니다 (정밀 탐색)
        "-t", f"{duration:.6f}",
        "-c:a", "pcm_s16le",  # WAV 표준 코덱
        "-vn",
        str(out_audio)
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="ignore"
        )
        return out_audio.exists() and out_audio.stat().st_size > 0
    except Exception as e:
        print(f"[AudioSlice] 실패: {e}")
        return False
# ── 분할/겹침 ───────────────────────────────────────────────────────────────
def plan_segments(frame_length: int, base_chunk: int = 82, overlap: int = 6, pad_tail: int = 5) -> List[
    Tuple[int, int]]:
    """
    [수정됨] Wan 2.2 오버랩 생성 전략:
    - base_chunk: 유효 구간 길이 (예: 82)
    - overlap: 다음 청크 생성을 위해 뒤로 감는 프레임 수 (예: 6)
    - pad_tail: 생성 안전망 (예: 5) -> 실제 생성 요청 길이는 base + pad

    반환값: [(start_frame, end_frame), ...]
    1) 0 ~ 87 (82+5) -> 유효: 0~82
    2) 76 (82-6) ~ 163 (76+82+5) -> 유효: 82~158
    ...
    """
    total = int(frame_length)
    if total <= 0:
        return []

    base = int(base_chunk)
    ov = int(overlap)
    pad = int(pad_tail)

    segments: List[Tuple[int, int]] = []

    # '유효 구간'의 시작 커서
    effective_cursor = 0

    while effective_cursor < total:
        # 1. 실제 생성 시작점
        if effective_cursor == 0:
            start_frame = 0
        else:
            # 이전 유효 구간 끝에서 오버랩만큼 뒤로 이동
            start_frame = effective_cursor - ov

        # 2. 실제 생성 끝점 (시작점 + 기본길이 + 안전패딩)
        end_frame = start_frame + base + pad

        segments.append((start_frame, end_frame))

        # 3. 유효 커서 전진 (기본 길이만큼)
        effective_cursor += base

    return segments


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


def recalc_overlap(in_fps: int, out_fps: int, overlap: int) -> int:
    if in_fps == out_fps:
        return overlap
    return max(1, int(round(overlap * (out_fps / float(in_fps)))))

# ── 워크플로 입력값 덮어쓰기 ─────────────────────────────────────────────────
def _apply_overrides(graph: dict, overrides: Dict[str, Any]) -> None:
    """
    path 형식: "<node_id>.inputs.<key>" -> 값
    """
    for path, val in (overrides or {}).items():
        nid, _, key = str(path).partition(".inputs.")
        try:
            graph[str(nid)]["inputs"][key] = val
        except Exception:
            pass

# ── 영상 및 음악 합치기 ────────────────────────────────────────────────────────────

# video_build.py (파일 하단에 추가)

import subprocess


# real_use
def build_and_merge_full_video(project_dir: str,
                               on_progress: Callable[[Dict[str, Any]], None]) -> str:
    """
    영상 생성부터 병합, 오디오 mux까지 전체 파이프라인을 실행합니다.
    [사용자 요청 6단계 실행]
    """

    def _log(stage: int, msg: str):
        print(f"[BuildPipeline] {stage}/6: {msg}", flush=True)
        on_progress({"msg": f"[{stage}/6] {msg}"})

    p_dir = Path(project_dir)
    video_json_path = p_dir / "video.json"
    clips_dir = p_dir / "clips"

    # --- 1. video.json 읽기 ---
    _log(1, "video.json 로드...")
    if not video_json_path.exists():
        raise FileNotFoundError(f"video.json을 찾을 수 없습니다: {video_json_path}")

    video_data = load_json(video_json_path, {}) or {}
    scenes = video_data.get("scenes", [])
    if not scenes:
        raise ValueError("video.json에 'scenes' 목록이 없습니다.")

    scene_clip_paths = [clips_dir / f"{s['id']}.mp4" for s in scenes]

    # --- 2. 누락된 씬(mp4) 확인 및 생성 (i2v) ---
    _log(2, "누락된 씬(mp4) 확인...")
    missing_clips = [p for p in scene_clip_paths if not p.exists() or p.stat().st_size == 0]

    if missing_clips:
        _log(3, f"누락된 씬 {len(missing_clips)}개 생성 시작 (i2v)...")
        # build_shots_with_i2v는 이미 "skip if exists" 로직을 갖고 있음
        # UI에서 읽어올 값이 없으므로 기본값으로 실행
        build_shots_with_i2v(
            project_dir=project_dir,
            total_frames=0,  # (이 함수에서는 사용되지 않음)
            ui_width=None,  # (video.json의 defaults 값을 따름)
            ui_height=None,
            ui_fps=None,
            ui_steps=None,
            on_progress=on_progress  # 로그 콜백 전달
        )
    else:
        _log(3, "모든 씬(mp4)이 이미 존재합니다. (i2v 생략)")

    # --- 4. 씬 병합 (Video Only) ---
    _log(4, f"{len(scene_clip_paths)}개 씬 병합 중 (비디오만)...")
    video_only_output = p_dir / "music_vocal_ready.mp4"

    concatenate_scene_clips(
        clip_paths=scene_clip_paths,
        out_path=video_only_output,
        ffmpeg_exe=FFMPEG_EXE  #
    )

    # --- 5. 오디오 파일 찾기 및 병합 (Muxing) ---
    _log(5, "오디오 파일(vocal.wav/mp3) 탐색...")
    audio_file = p_dir / "vocal.wav"
    if not audio_file.exists():
        audio_file = p_dir / "vocal.mp3"
        if not audio_file.exists():
            # glob로 유연하게 탐색
            found = list(p_dir.glob("vocal.*"))
            if not found:
                raise FileNotFoundError(f"프로젝트 폴더에서 오디오 파일(vocal.wav/mp3)을 찾을 수 없습니다: {p_dir}")
            audio_file = found[0]

    _log(5, f"오디오 파일({audio_file.name})과 비디오 병합 중...")
    final_output = p_dir / "music_ready.mp4"

    mux_video_and_audio(
        video_in_path=video_only_output,
        audio_in_path=audio_file,
        out_path=final_output,
        ffmpeg_exe=FFMPEG_EXE  #
    )

    # --- 6. 완료 ---
    _log(6, f"최종 영상 생성 완료: {final_output.name}")
    return str(final_output)


def concatenate_scene_clips(clip_paths: List[Path], out_path: Path, ffmpeg_exe: str):
    """
    [신규] FFMPEG concat demuxer를 사용해 여러 비디오 클립을 순서대로 병합합니다.
    (가장 안정적인 방식)
    """
    work_dir = out_path.parent
    list_file = work_dir / "ffmpeg_concat_list.txt"

    try:
        # 1. ffmpeg에 전달할 목록 파일(list.txt) 생성
        with open(list_file, "w", encoding="utf-8") as f:
            for clip in clip_paths:
                if not clip.exists():
                    raise FileNotFoundError(f"병합할 클립을 찾을 수 없습니다: {clip.name}")
                # FFMPEG concat은 경로에 특수문자가 있으면 오류가 날 수 있으므로,
                # 'file' 키워드와 함께 절대 경로를 따옴표로 감쌉니다.
                f.write(f"file '{clip.as_posix()}'\n")

        # 2. FFMPEG concat 실행
        cmd = [
            ffmpeg_exe,
            "-y",  # 덮어쓰기
            "-f", "concat",  # concat demuxer 사용
            "-safe", "0",  # 절대 경로 허용
            "-i", str(list_file),
            "-c", "copy",  # 재인코딩 없이 스트림 복사 (매우 빠름)
            str(out_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        if result.returncode != 0:
            raise RuntimeError(f"FFMPEG 씬 병합 실패:\n{result.stderr}")

    finally:
        # 3. 임시 목록 파일 삭제
        if list_file.exists():
            list_file.unlink()


def mux_video_and_audio(video_in_path: Path, audio_in_path: Path, out_path: Path, ffmpeg_exe: str):
    """
    [신규] 비디오 파일과 오디오 파일을 하나로 합칩니다(Muxing).
    """
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(video_in_path),  # 입력 1 (비디오)
        "-i", str(audio_in_path),  # 입력 2 (오디오)
        "-c:v", "copy",  # 비디오는 재인코딩 없이 복사
        "-c:a", "aac",  # 오디오는 호환성을 위해 AAC로 인코딩 (wav -> aac)
        "-b:a", "192k",
        "-shortest",  # 비디오/오디오 중 짧은 쪽 길이에 맞춰 종료
        str(out_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if result.returncode != 0:
        raise RuntimeError(f"FFMPEG 오디오/비디오 병합 실패:\n{result.stderr}")

# real_use
def add_subtitles_with_ffmpeg(video_in_path: Path,
                             video_json_path: Path,
                             out_path: Path,
                             ffmpeg_exe: str,
                             font_name: str = "Malgun Gothic", # UI에서 이 값을 보내지만, 아래에서 무시합니다.
                             title_fontsize: int = 70,
                             lyric_fontsize: int = 48
                             ) -> str:
    """
    FFMPEG의 drawtext 필터를 사용하여 video.json의 자막을 비디오에 직접 하드코딩합니다.
    (MoviePy 불필요)
    [수정됨] 폰트 경로(fontfile) 대신 폰트 이름(font)을 사용합니다.
    """

    # 1. [수정] 사용할 폰트 경로를 하드코딩하고, 올바르게 이스케이프합니다.
    #    (UI에서 전달된 font_name은 무시됩니다.)
    font_path_hardcoded = "C:/Windows/Fonts/malgun.ttf"  # <-- 사용할 폰트 경로

    font_path_ffmpeg = font_path_hardcoded.replace(os.path.sep, "/")
    font_path_ffmpeg = font_path_ffmpeg.replace(":", "\\:")

    # 2. video.json 로드
    video_data = load_json(video_json_path, {}) or {}
    scenes = video_data.get("scenes", [])
    title = video_data.get("title", "제목 없음")

    filters = []

    # 3. 제목 필터 생성 (예: 0.5초 ~ 3.5초간 표시)
    if title:
        # FFMPEG 텍스트 필터는 특수문자(' : \) 이스케이프 처리가 필요합니다.
        title_escaped = title.replace("'", "'\\\\''").replace(":", "\\:").replace("\\", "\\\\")
        filters.append(
            f"drawtext=fontfile='{font_path_ffmpeg}':text='{title_escaped}':"
            f"fontsize={int(title_fontsize)}:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5:"
            f"x=(w-text_w)/2:y=h*0.2:"  # 상단 20% 위치
            f"enable='between(t,0.5,3.5)'"
        )

    # 4. 가사 필터 생성
    for scene in scenes:
        lyric = scene.get("lyric", "").strip()
        if not lyric:
            continue  # 가사가 없는 씬(gap_002 등) 건너뛰기

        start = scene.get("start", 0.0)
        end = scene.get("end", 0.0)
        if end <= start:
            continue

        # FFMPEG 텍스트 필터 이스케이프
        lyric_escaped = lyric.replace("'", "'\\\\''").replace(":", "\\:").replace("\\", "\\\\").replace("\n", "\\n")

        filters.append(
            f"drawtext=fontfile='{font_path_ffmpeg}':text='{lyric_escaped}':"
            f"fontsize={int(lyric_fontsize)}:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=4:"
            f"x=(w-text_w)/2:y=h*0.8:"  # 하단 80% 위치
            f"enable='between(t,{start},{end})'"
        )

    if not filters:
        # 자막이 없으면 그냥 복사하고 반환
        shutil.copy2(str(video_in_path), str(out_path))
        return str(out_path)

    # 5. FFMPEG 명령어 생성
    filter_complex_string = ",".join(filters)

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(video_in_path),
        "-vf", filter_complex_string,  # 비디오 필터로 자막 적용
        "-c:a", "copy",  # 오디오는 그대로 복사
        "-c:v", "libx264",  # 비디오는 재인코딩 (자막을 입혀야 하므로)
        "-preset", "fast",  # 빠른 인코딩
        "-crf", "22",  # 적절한 품질
        "-pix_fmt", "yuv420p",  # 호환성
        str(out_path)
    ]

    # 6. FFMPEG 실행
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if result.returncode != 0:
        # 오류 발생 시, FFMPEG 로그를 포함하여 예외 발생
        raise RuntimeError(f"FFMPEG 자막 삽입 실패:\n{result.stderr}")

    return str(out_path)



# ── i2v 샷 생성 ────────────────────────────────────────────────────────────

def _detect_reactor_face_slots_i2v(
    graph: Dict[str, Dict[str, Any]],
    notify,
) -> Dict[str, Tuple[str, str]]:
    """
    워크플로우(graph)에서 ReActorFaceSwap ↔ LoadImage ↔ 캐릭터ID를 자동으로 찾는다.
    반환: {reactor_node_id: (load_node_id, char_id)}
      - char_id 예: "female_01", "male_01", 그 외에는 파일명 stem을 그대로 씀
    """
    result: Dict[str, Tuple[str, str]] = {}

    for nid, node in graph.items():
        if node.get("class_type") != "ReActorFaceSwap":
            continue

        inputs = node.get("inputs") or {}
        src = inputs.get("source_image")

        # source_image: [load_node_id, 0] 형태라고 가정
        if not isinstance(src, list) or not src:
            continue

        load_nid = str(src[0])
        load_node = graph.get(load_nid)
        if not load_node or load_node.get("class_type") != "LoadImage":
            continue

        load_inputs = load_node.get("inputs") or {}
        img_name = str(load_inputs.get("image") or "")
        base = os.path.basename(img_name)
        stem, _ = os.path.splitext(base)

        char_id = None
        if "female_01" in stem:
            char_id = "female_01"
        elif "male_01" in stem:
            char_id = "male_01"
        elif stem:
            # 그 외 → 파일명 그대로 캐릭터 ID로 사용 (예: custom_hero_01)
            char_id = stem

        if not char_id:
            continue

        result[str(nid)] = (load_nid, char_id)

    if not result:
        notify("[I2V][FACE] ReActor ↔ LoadImage 매핑을 찾지 못함 (워크플로우 구조 확인 필요)")
    else:
        notify(f"[I2V][FACE] 자동 매핑 완료: {result!r}")

    return result

def _inject_scene_main_image_i2v(
    graph: Dict[str, Dict[str, Any]],
    scene_item: Dict[str, Any],
    comfy_input_dir: Path,
    notify,
) -> None:
    """
    - 해당 씬의 img_file을 ComfyUI input 폴더에 복사하고,
    - ReActor가 source_image로 쓰는 LoadImage 노드를 제외한
      나머지 LoadImage 노드에만 파일명을 주입한다.
    """
    scene_id = scene_item.get("id")
    img_path = Path(scene_item.get("img_file") or "")

    if not img_path.is_file():
        notify(f"[I2V][IMG] scene={scene_id} img_file 없음 → 주입 스킵: {img_path}")
        return

    comfy_input_dir.mkdir(parents=True, exist_ok=True)
    dst = comfy_input_dir / img_path.name

    if str(img_path).lower() != str(dst).lower():
        try:
            shutil.copy2(img_path, dst)
        except Exception as e:
            notify(f"[I2V][IMG] scene={scene_id} 메인 이미지 복사 실패: {img_path} → {dst} ({e!r})")
            return

    # 1) ReActor에서 source_image로 쓰는 LoadImage 노드 id 전부 수집
    reserved_load_ids: set[str] = set()
    for nid, node in graph.items():
        if node.get("class_type") != "ReActorFaceSwap":
            continue
        inputs = node.get("inputs") or {}
        src = inputs.get("source_image")
        if isinstance(src, list) and src:
            reserved_load_ids.add(str(src[0]))

    # 2) 구버전에서 쓰던 하드코딩 값도 안전하게 추가
    reserved_load_ids.update({"176", "177", "178", "479", "480", "481"})

    notify(
        f"[I2V][IMG] scene={scene_id} inject image='{dst.name}' "
        f"into load nodes (reserved={sorted(reserved_load_ids)!r})"
    )

    # 3) reserved 에 속하지 않는 LoadImage 노드에만 주입
    for nid, node in graph.items():
        if node.get("class_type") != "LoadImage":
            continue
        if str(nid) in reserved_load_ids:
            continue

        inputs = node.get("inputs") or {}
        old_img = inputs.get("image")
        inputs["image"] = dst.name
        node["inputs"] = inputs

        notify(
            f"[I2V][IMG] node={nid} image {old_img!r} → {dst.name!r}"
        )

# ───────────────────────── Helper: 캐릭터 이미지 경로 ─────────────────────────
# ───────────────────────── Helper: 캐릭터 이미지 경로 ─────────────────────────
def _resolve_character_image_path_i2v(
        scene_obj: Dict[str, Any],
        video_doc_obj: Dict[str, Any],
        character_id: str,
        base_dir: Path,
) -> Optional[Path]:
    """
    캐릭터 ID에 해당하는 이미지를 찾는다.
    우선순위:
      1) scene_obj["characters_ex"][character_id]["img_file"] 등 (있다면)
      2) CHARACTER_DIR/character_id.png/jpg/webp...
    """
    char_id_str = str(character_id).strip()
    if not char_id_str:
        return None

    # 1) scene_obj 또는 video_doc_obj에 캐릭터 메타가 있다면 활용
    for container in (scene_obj, video_doc_obj):
        chars_ex = container.get("characters_ex")
        if isinstance(chars_ex, dict):
            meta = chars_ex.get(char_id_str)
            if isinstance(meta, dict):
                img_candidate = meta.get("img_file") or meta.get("image_path")
                if img_candidate:
                    path_candidate = Path(str(img_candidate))
                    if path_candidate.is_file():
                        return path_candidate

    # 2) CHARACTER_DIR에서 이름으로 탐색
    exts = [".png", ".jpg", ".jpeg", ".webp"]
    for ext in exts:
        candidate = base_dir / f"{char_id_str}{ext}"
        if candidate.is_file():
            return candidate

    return None

# video_build.py 상단 헬퍼 함수 구역에 추가


# video_build.py 상단 import 아래에 추가

def _i2v_extract_still_frame_by_time(
    src_video: Path,
    target_time_sec: float,
    out_dir: Path,
    prefix: str = "ref",
    *,
    log_func: Optional[Callable[[str], None]] = None,
) -> Optional[Path]:
    """
    src_video 의 target_time_sec 위치에서 1프레임을 PNG로 추출해서
    out_dir 에 저장하고 Path 를 반환한다.
    - ffmpeg 실행 실패 시 None 반환
    - settings.FFMPEG_EXE 를 우선 사용하고, 없으면 "ffmpeg" 사용
    """
    # 로깅 헬퍼
    def _log(msg: str) -> None:
        if log_func:
            try:
                log_func(msg)
            except Exception:
                pass
        else:
            print(msg)

    # 시간 보정
    if target_time_sec < 0:
        target_time_sec = 0.0

    out_dir.mkdir(parents=True, exist_ok=True)

    # 출력 파일명: t_001_seg01_raw_seg1_4750ms.png 이런 식
    ms = int(round(target_time_sec * 1000.0))
    out_name = f"{src_video.stem}_{prefix}_{ms:05d}ms.png"
    out_path = out_dir / out_name

    # 이미 있으면 재사용
    if out_path.exists() and out_path.stat().st_size > 0:
        _log(f"[FRAME] 이미 존재하는 정지 프레임 재사용: {out_path.name}")
        return out_path

    # ffmpeg 실행 파일 결정
    try:
        ffmpeg_exe = getattr(settings_obj, "FFMPEG_EXE", "ffmpeg")
    except Exception:
        ffmpeg_exe = "ffmpeg"

    cmd = [
        ffmpeg_exe,
        "-y",
        "-ss",
        f"{target_time_sec:.6f}",
        "-i",
        str(src_video),
        "-frames:v",
        "1",
        "-f",
        "image2",
        str(out_path),
    ]

    _log(
        f"[FRAME] ffmpeg 스틸 추출 실행: time={target_time_sec:.3f}s → {out_path.name}"
    )

    try:
        proc = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        # 혹시 0바이트 생성되면 실패 처리
        if not out_path.exists() or out_path.stat().st_size == 0:
            _log("[FRAME][ERR] 출력 파일이 생성되지 않았습니다.")
            return None

        _log(f"[FRAME] 스틸 추출 성공: {out_path.name}")
        return out_path

    except subprocess.CalledProcessError as e:
        out_txt = (e.stdout or "").strip()
        _log("[FRAME][ERR] ffmpeg 실행 실패")
        if out_txt:
            for line in out_txt.splitlines():
                _log(f"[FRAME][ffmpeg] {line}")
        return None
    except Exception as e:
        _log(f"[FRAME][ERR] 예기치 못한 오류: {e}")
        return None

def _trim_by_duration(file_path: Path, target_duration_sec: float) -> None:
    tmp_path = file_path.with_suffix(".trim.mp4")
    try:
        subprocess.run(["ffmpeg", "-y", "-i", str(file_path), "-t", f"{target_duration_sec:.6f}", "-c", "copy", str(tmp_path)], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if tmp_path.exists() and tmp_path.stat().st_size > 0:
            shutil.move(str(tmp_path), str(file_path))
    except: pass
    if tmp_path.exists(): tmp_path.unlink()


def _toggle_upscale_node(graph: Dict[str, Any], enable: bool) -> None:
    """
    No.48 워크플로우 제어:
    - 166: GIMMVFI_interpolate (보간, 원본 해상도)
    - 164: SeedVR2VideoUpscaler (업스케일)
    - 172, 119: VHS_VideoCombine (저장 노드들)
    """
    # 저장 노드가 될 수 있는 후보들
    save_node_ids = ["172", "119"]

    # 164번(업스케일러) 자체가 있는지 확인
    upscaler = graph.get("164")

    for nid in save_node_ids:
        if nid not in graph: continue

        if enable and upscaler:
            # 업스케일 ON: 저장 노드 <- 업스케일러(164) 연결
            graph[nid]["inputs"]["images"] = ["164", 0]
        else:
            # 업스케일 OFF: 저장 노드 <- 보간 노드(166) 직접 연결 (Bypass)
            # 이렇게 해야 164번이 아예 작동을 안 해서 속도가 빨라짐
            graph[nid]["inputs"]["images"] = ["166", 0]



# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# [메인 실행 래퍼]
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# [메인 실행 래퍼] — 원클릭: RAW 전체 생성 → 전체 보간/업스케일
# ─────────────────────────────────────────────────────────────────────────────
# real_use
def build_shots_with_i2v(
    project_dir: str,
    total_frames: int,
    *,  # 키워드 전용
    ui_width: Optional[int] = None,
    ui_height: Optional[int] = None,
    ui_fps: Optional[int] = None,
    ui_steps: Optional[int] = None,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    """
    새 I2V 파이프라인 래퍼 (원클릭).

    1) raw_make_addup(...)
       - WAN I2V + ReActor + 정크 분할/병합까지 수행 (clips/{scene_id}_raw.mp4 생성)
    2) Interpolation_upscale(...)
       - RAW 씬 영상을 Interpolation_upscale.json 으로
         GIMMVFI + SeedVR2 업스케일/보간 (clips/{scene_id}.mp4 생성)

    기존 동작이 필요하면 build_shots_with_i2v_old(...) 를 직접 호출하면 된다.
    """

    # 1단계: 모든 씬 RAW 생성 및 병합
    raw_make_addup(
        project_dir,
        total_frames,
        ui_width=ui_width,
        ui_height=ui_height,
        ui_fps=ui_fps,
        ui_steps=ui_steps,
        on_progress=on_progress,
    )

    # 2단계: RAW가 만들어진 씬에 대해 업스케일/보간 일괄 수행
    Interpolation_upscale(
        project_dir,
        total_frames,
        ui_width=ui_width,
        ui_height=ui_height,
        ui_fps=ui_fps,
        ui_steps=ui_steps,
        on_progress=on_progress,
    )





# ─────────────────────────────────────────────────────────────────────────────
# [1단계] Raw 생성 및 병합 (기능 삭제 없음: 프롬프트, ReActor, Xfade 모두 포함)
# ─────────────────────────────────────────────────────────────────────────────
# real_use
def raw_make_addup(
    project_dir: str,
    total_frames: int,
    *,  # 키워드 전용
    ui_width: Optional[int] = None,
    ui_height: Optional[int] = None,
    ui_fps: Optional[int] = None,
    ui_steps: Optional[int] = None,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    """
    [1단계] RAW I2V 생성 전담 함수

    - No.48.WAN2.2-LightX2V-I2V.json 워크플로우를 사용해,
      각 씬을 정크(82+overlap+pad) 단위로 생성한다. :contentReference[oaicite:2]{index=2}
    - ComfyUI 출력 중 **Node 167 (VideoCombine)** 만 사용해
      clips/{scene_id}/{scene_id}_segXX_raw.mp4 를 만든다.
    - 모든 세그먼트를 cross-fade 병합해
      clips/{scene_id}_raw.mp4 를 만든다.
    - FPS는 “기본 fps”(16 또는 video.json / UI에서 온 값) 기준으로 정규화한다.
    - GIMMVFI/SeedVR2 출력(Node 172)은 이 단계에서는 **사용하지 않는다**.
    """


    # ───────────────────────── utils 안전 import ─────────────────────────


    # ───────────────────────── settings 안전 import ─────────────────────────


    # ───────────────────────── Comfy submit/wait 재사용 ─────────────────────────


    # ───────────────────────── Helper: on_progress ─────────────────────────
    start_ts = time.time()
    last_timer_ts = start_ts

    def _notify(msg: str, *, force: bool = False) -> None:
        nonlocal last_timer_ts

        now = time.time()
        elapsed = int(now - start_ts)
        prefix = f"[+{elapsed:4d}s] "
        line = prefix + msg

        print(line)
        if on_progress:
            try:
                on_progress({"msg": line})
            except Exception:
                pass

        if force or (now - last_timer_ts >= 10.0):
            last_timer_ts = now
            timer_msg = f"[I2V-RAW][TIMER] 현재까지 {elapsed}초 경과"
            print(prefix + timer_msg)
            if on_progress:
                try:
                    on_progress({"msg": prefix + timer_msg})
                except Exception:
                    pass

    # ───────────────────────── Helper: ffmpeg / ffprobe ─────────────────────────
    ffmpeg_exe_val = getattr(settings_obj, "FFMPEG_EXE", "ffmpeg")
    ffprobe_exe_val = getattr(settings_obj, "FFPROBE_EXE", "ffprobe")

    def _i2v_probe_nb_frames(path_obj: Path) -> Optional[int]:
        try:
            cmd_args = [
                ffprobe_exe_val,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(path_obj),
            ]
            proc = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                check=True,
            )
            txt = (proc.stdout or "").strip()
            if not txt:
                return None
            return int(txt)
        except Exception:
            return None

    def _i2v_trim_tail(
        path_in: Path, path_out: Path, target_frames: int, fps_val: int
    ) -> None:
        nb_frames = _i2v_probe_nb_frames(path_in)
        if nb_frames is None:
            _notify(f"[RAW][WARN] nb_frames 파악 실패, trim 스킵: {path_in.name}")
            if path_in != path_out:
                shutil.copy2(str(path_in), str(path_out))
            return

        if nb_frames <= target_frames:
            _notify(
                f"[RAW] trim 불필요 (nb_frames={nb_frames}, target={target_frames})"
            )
            if path_in != path_out:
                shutil.copy2(str(path_in), str(path_out))
            return

        sec = float(target_frames) / float(max(fps_val, 1))
        _notify(
            f"[RAW] tail trim: nb_frames={nb_frames} → target={target_frames} ({sec:.3f}s)"
        )

        if path_in == path_out:
            tmp_out = path_out.with_suffix(".trim_tmp.mp4")
        else:
            tmp_out = path_out

        if tmp_out.exists():
            try:
                tmp_out.unlink()
            except OSError:
                pass

        cmd_args = [
            ffmpeg_exe_val,
            "-y",
            "-i",
            str(path_in),
            "-t",
            f"{sec:.6f}",
            "-c",
            "copy",
            str(tmp_out),
        ]

        try:
            subprocess.run(
                cmd_args,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
        except subprocess.CalledProcessError as eexc:
            _notify(f"[RAW] tail trim ffmpeg 실패: {eexc}")
            if tmp_out.exists():
                try:
                    tmp_out.unlink()
                except OSError:
                    pass
            return

        if tmp_out is not path_out:
            try:
                if path_out.exists():
                    path_out.unlink()
            except OSError:
                pass
            tmp_out.rename(path_out)

    def _i2v_concat_ab(
        path_a: Path,
        path_b: Path,
        path_out: Path,
        crossfade_sec: float = 0.3,
    ) -> None:
        """
        RAW 단계에서는 해상도는 이미 동일(같은 워크플로우)이라고 가정하고,
        단순 crossfade + libx264 인코딩을 수행한다.
        (해상도 mismatch 케이스는 기존 build_shots_with_i2v_old 의 복잡한 버전을 참고)
        """

        def _duration(csrc_path: Path) -> float:
            try:
                ccmd_args = [
                    ffprobe_exe_val,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=duration",
                    "-of",
                    "default=nokey=1:noprint_wrappers=1",
                    str(csrc_path),
                ]
                procc = subprocess.run(
                    ccmd_args,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                txt = (procc.stdout or "").strip()
                if not txt:
                    return 0.0
                return float(txt)
            except Exception:
                return 0.0

        for xp in (path_a, path_b):
            if (not xp.exists()) or xp.stat().st_size <= 0:
                raise RuntimeError(f"[RAW] xfade 입력 영상이 비정상입니다: {xp}")

        dur_a = _duration(path_a)
        dur_b = _duration(path_b)
        _notify(
            f"[RAW][XF] concat '{path_a.name}'(dur={dur_a:.3f}s) + "
            f"'{path_b.name}'(dur={dur_b:.3f}s)"
        )

        if dur_a <= 0:
            _notify("[RAW][XF] dur_a<=0 → 단순 바이너리 concat 사용")
            with open(path_out, "wb") as cout_file:
                for src_path in (path_a, path_b):
                    with open(src_path, "rb") as in_file:
                        while True:
                            chunk = in_file.read(1024 * 1024)
                            if not chunk:
                                break
                            cout_file.write(chunk)
            return

        offset = max(dur_a - crossfade_sec, 0.0)

        cmd_args = [
            ffmpeg_exe_val,
            "-y",
            "-i",
            str(path_a),
            "-i",
            str(path_b),
            "-filter_complex",
            (
                f"[0:v]format=yuv420p[v0];"
                f"[1:v]format=yuv420p[v1];"
                f"[v0][v1]xfade=transition=fade:duration={crossfade_sec}:offset={offset}[v_out]"
            ),
            "-map",
            "[v_out]",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            str(path_out),
        ]

        try:
            subprocess.run(
                cmd_args,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
        except subprocess.CalledProcessError as exxx:
            out_txt = (exxx.stdout or "").strip()
            _notify("[RAW][XF][ERR] ffmpeg xfade 실패")
            if out_txt:
                for line in out_txt.splitlines():
                    _notify(f"[RAW][XF][ffmpeg] {line}")
            raise

    def _i2v_norm_fps_and_size(
        path_in: Path, path_out: Path, fps_val: int
    ) -> None:
        """
        RAW 씬 단위 결과에 대해 FPS 정규화만 수행.
        (해상도는 워크플로우에서 이미 맞추었다고 가정)
        """
        cmd_args = [
            ffmpeg_exe_val,
            "-y",
            "-i",
            str(path_in),
            "-vf",
            f"fps={fps_val}",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "slow",
            str(path_out),
        ]
        subprocess.run(cmd_args, check=True)

    # ───────────────────────── I2V 세그먼트 공통 상수 ─────────────────────────
    # CHUNK_BASE_FRAMES = 82
    # OVERLAP_FRAMES = 6
    # PAD_TAIL_FRAMES = 5
    #


    def _i2v_plan_segments(
        frame_len_total: int,
        base_chunk: int = 82,
        overlap_val: int = 6,
        pad_tail_val: int = 5,
    ) -> List[Tuple[int, int]]:
        total_val = int(frame_len_total)
        if total_val <= 0:
            return []

        segments: List[Tuple[int, int]] = []
        current_valid_end = 0

        while current_valid_end < total_val:
            if current_valid_end == 0:
                start_frame = 0
            else:
                start_frame = current_valid_end - overlap_val

            end_frame = start_frame + base_chunk + pad_tail_val
            segments.append((start_frame, end_frame))
            current_valid_end += base_chunk

        return segments

    # ───────────────────────── Main Logic ─────────────────────────
    project_dir_path = Path(project_dir).resolve()
    _ensure_dir_func(project_dir_path)

    video_json_path = project_dir_path / "video.json"
    video_doc = _load_json_func(video_json_path, {}) or {}

    # total_frames 보정
    tframes_corrected = int(total_frames)
    if tframes_corrected <= 0:
        _notify("[RAW][경고] total_frames <= 0 → video.json 기준 재계산")
        try:
            fps_val_for_calc = int(video_doc.get("fps", 16))
        except Exception:
            fps_val_for_calc = 16

        try:
            duration_sec = float(video_doc.get("duration", 0.0))
        except Exception:
            duration_sec = 0.0

        if duration_sec <= 0.0:
            scenes_for_calc = video_doc.get("scenes") or []
            total_d = 0.0
            for s in scenes_for_calc:
                if not isinstance(s, dict):
                    continue
                try:
                    d = float(s.get("duration", 0.0))
                    if d <= 0.0:
                        d = float(s.get("end", 0.0)) - float(s.get("start", 0.0))
                except Exception:
                    d = 0.0
                if d > 0:
                    total_d += d
            duration_sec = total_d

        if duration_sec > 0.0:
            tframes_corrected = int(
                round(duration_sec * float(max(fps_val_for_calc, 1)))
            )
            _notify(
                f"[RAW][INFO] video.json 기준 total_frames={tframes_corrected} "
                f"(duration={duration_sec:.3f}s, fps={fps_val_for_calc})"
            )
        else:
            tframes_corrected = max(tframes_corrected, 16 * 5)
            _notify(
                f"[RAW][WARN] duration 정보 부족 → 최소 {tframes_corrected} 프레임 가정"
            )

    total_frames_int = tframes_corrected
    if total_frames_int > 0:
        _notify(f"[RAW] raw_make_addup 호출 (total_frames={total_frames_int})")

    # 기본 fps/해상도/steps
    try:
        fps_from_video = int(video_doc.get("fps", 16))
    except (TypeError, ValueError):
        fps_from_video = 16

    if ui_fps is not None and ui_fps > 0:
        base_fps_in = int(ui_fps)
    else:
        base_fps_in = fps_from_video

    defaults_doc: Dict[str, Any] = video_doc.get("defaults") or {}
    defaults_movie: Dict[str, Any] = (defaults_doc.get("movie") or {}) if isinstance(
        defaults_doc, dict
    ) else {}

    try:
        default_w = int(defaults_movie.get("width", 720))
        default_h = int(defaults_movie.get("height", 1280))
    except Exception:
        default_w, default_h = 720, 1280

    try:
        default_steps = int(defaults_movie.get("steps", 28))
    except Exception:
        default_steps = 28

    if ui_width and ui_width > 0:
        target_w = int(ui_width)
    else:
        target_w = default_w

    if ui_height and ui_height > 0:
        target_h = int(ui_height)
    else:
        target_h = default_h

    if ui_steps and ui_steps > 0:
        target_steps = int(ui_steps)
    else:
        target_steps = default_steps

    _notify(
        f"[RAW][CFG] base_fps={base_fps_in}, size={target_w}x{target_h}, steps={target_steps}"
    )

    # ───────────────────────── 워크플로우 파일 선택 ─────────────────────────
    jsons_dir_conf = getattr(
        settings_obj, "JSONS_DIR", str(project_dir_path / "jsons")
    )
    i2v_workflow_conf = getattr(settings_obj, "I2V_WORKFLOW", None)
    comfy_host = getattr(settings_obj, "COMFY_HOST", "http://127.0.0.1:8188")
    comfy_input_dir = Path(
        str(getattr(settings_obj, "COMFY_INPUT_DIR", project_dir_path / "input"))
    )
    _ensure_dir_func(comfy_input_dir)

    base_jsons_path = Path(str(jsons_dir_conf))
    _ensure_dir_func(base_jsons_path)

    workflow_path: Optional[Path] = None

    if i2v_workflow_conf:
        c1 = Path(str(i2v_workflow_conf))
        if not c1.is_absolute():
            c1 = base_jsons_path / c1.name
        if c1.is_file():
            workflow_path = c1

    if workflow_path is None:
        defaults_all_doc = video_doc.get("defaults") or {}
        defaults_i2v_doc = (defaults_all_doc.get("i2v") or {}) if isinstance(
            defaults_all_doc, dict
        ) else {}
        workflow_from_video = str(defaults_i2v_doc.get("workflow") or "").strip()
        if workflow_from_video:
            c2 = Path(workflow_from_video)
            if not c2.is_absolute():
                c2 = base_jsons_path / c2.name
            if c2.is_file():
                workflow_path = c2

    if workflow_path is None:
        c3 = base_jsons_path / "No.48.WAN2.2-LightX2V-I2V.json"
        if c3.is_file():
            workflow_path = c3
        else:
            c_old = base_jsons_path / "guff_movie.json"
            if c_old.is_file():
                workflow_path = c_old
            else:
                raise FileNotFoundError(
                    "i2v 워크플로우 파일을 찾을 수 없습니다. "
                    "(settings.I2V_WORKFLOW, defaults.i2v.workflow, "
                    "No.48.WAN2.2-LightX2V-I2V.json, guff_movie.json)"
                )

    _notify(f"[RAW] 사용할 워크플로우: {workflow_path.name} (경로: {workflow_path})")

    with workflow_path.open("r", encoding="utf-8") as wf_f:
        try:
            graph_origin = json_mod.load(wf_f)
        except Exception as exc:
            raise RuntimeError(f"워크플로우 JSON 파싱 실패: {exc}")

    # convert_resolution / steps 노드 ID (네 워크플로우 기준)
    RES_NODE_IDS = ["21", "22", "23"]
    STEPS_NODE_IDS = ["15", "16"]

    # ───────────────────────── 보조: 캐릭터 스펙 파서 ─────────────────────────
    def _parse_character_spec_i2v(raw_obj: Any) -> Dict[str, Any]:
        if isinstance(raw_obj, dict):
            out = dict(raw_obj)
            cid_val = str(out.get("id") or out.get("name") or "").strip()
            out["id"] = cid_val
            if "index" in out and out["index"] is not None:
                try:
                    out["index"] = int(out["index"])
                except Exception:
                    out["index"] = None
            elif "face_index" in out and out["face_index"] is not None:
                try:
                    out["index"] = int(out["face_index"])
                except Exception:
                    out["index"] = None
            else:
                out["index"] = None
            return out

        if isinstance(raw_obj, str):
            txt_val = raw_obj.strip()
            if ":" in txt_val:
                left, right = txt_val.split(":", 1)
                try:
                    idx_val = int(right.strip())
                except Exception:
                    idx_val = None
                return {"id": left.strip(), "index": idx_val}
            return {"id": txt_val, "index": None}

        return {"id": "", "index": None}

    # ───────────────────────── 기본 경로 ─────────────────────────
    clips_dir = project_dir_path / "clips"
    _ensure_dir_func(clips_dir)

    character_dir_base = Path(
        str(getattr(settings_obj, "CHARACTER_DIR", project_dir_path / "characters"))
    )
    _ensure_dir_func(character_dir_base)

    global_ctx = dict(video_doc.get("global_context") or {})
    defaults_root = video_doc.get("defaults") or {}
    defaults_image = (defaults_root.get("image") or {}) if isinstance(
        defaults_root, dict
    ) else {}

    global_negative_bank = str(global_ctx.get("negative_bank") or "").strip()
    default_negative = str(defaults_image.get("negative") or "").strip()
    if default_negative == "@global":
        base_negative = global_negative_bank
    else:
        base_negative = ", ".join(
            x for x in [global_negative_bank, default_negative] if x
        ).strip()

    scenes = video_doc.get("scenes") or []
    if not isinstance(scenes, list):
        raise RuntimeError("video.json['scenes']가 리스트가 아닙니다.")

    total_scenes = len(scenes)
    if total_scenes == 0:
        _notify("[RAW] scenes 가 비어 있습니다. 작업 종료.")
        return

    # [추가] 립싱크 워크플로우 로드 (루프 밖에서 한 번만 시도)
    graph_lip = {}
    if WF_LIPSYNC.exists():
        try:
            with open(WF_LIPSYNC, "r", encoding="utf-8") as f:
                graph_lip = json_mod.load(f)
        except Exception as e:
            _notify(f"[WARN] 립싱크 워크플로우 로드 실패: {e}")

    # [추가] 오디오 소스 찾기 (립싱크용)
    audio_source = project_dir_path / "vocal.wav"
    if not audio_source.exists():
        audio_source = project_dir_path / "vocal.mp3"

    # ───────────────────────── 전체 씬 LOOP ─────────────────────────
    scene_index = 0
    for scene_item in scenes:
        scene_index += 1
        if not isinstance(scene_item, dict):
            continue

        scene_id = str(scene_item.get("id") or f"scene_{scene_index:03d}")
        scene_out_dir = clips_dir / scene_id
        _ensure_dir_func(scene_out_dir)

        # 기존 최종 RAW 씬 파일
        scene_raw_tmp = clips_dir / f"{scene_id}_raw_tmp.mp4"
        scene_raw_norm = clips_dir / f"{scene_id}_raw.mp4"

        try:
            scene_duration = float(scene_item.get("duration", 0.0))
            scene_start = float(scene_item.get("start", 0.0))
            scene_end = float(scene_item.get("end", 0.0))
        except Exception:
            scene_duration = 0.0
            scene_start = 0.0
            scene_end = 0.0

        if scene_duration <= 0:
            _notify(f"[RAW] 씬 {scene_index}/{total_scenes} ({scene_id}) duration<=0, 스킵")
            continue

        try:
            frame_length_val = max(1, int(round(scene_duration * float(base_fps_in))))
        except Exception:
            frame_length_val = base_fps_in

        # 이미 RAW 씬이 있으면 스킵
        if scene_raw_norm.exists() and scene_raw_norm.stat().st_size > 1024:
            _notify(f"[RAW] 씬 {scene_index}/{total_scenes} ({scene_id}) RAW 결과 존재 → 스킵")
            continue

        # =========================================================
        # [CASE 1] 립싱크 모드 (InfiniteTalk One-Pass)
        # =========================================================
        is_lipsync = scene_item.get("lync_bool", False)

        if is_lipsync and graph_lip and audio_source.exists():
            lync_prompt = scene_item.get('lync_prompt', 'sing a song')
            _notify(f"  -> [LipSync Mode] ON (Prompt: {lync_prompt})")

            # (1) 오디오 자르기 (vocal.wav -> clips/{id}/song.wav)
            scene_audio_path = scene_out_dir / "song.wav"
            # ▼▼▼ [수정된 로직] 파일이 이미 존재하면 자르기 스킵 ▼▼▼
            if scene_audio_path.exists() and scene_audio_path.stat().st_size > 0:
                _notify(f"  -> [LipSync] 기존 오디오 파일 사용 (스킵): {scene_audio_path.name}")
            else:
                # 파일이 없을 때만 자르기 수행
                if not _slice_audio_segment(audio_source, scene_start, scene_end, scene_audio_path, ffmpeg_exe_val):
                    _notify(f"  -> [ERR] 오디오 자르기 실패. 일반 모드로 전환합니다.")
                    is_lipsync = False  # 실패 시 아래 일반 모드로 진행
            # ▲▲▲ [수정 끝] ▲▲▲

            if is_lipsync:
                # (2) 워크플로우 복제 및 파라미터 주입
                graph = json_mod.loads(json_mod.dumps(graph_lip))

                # --- ▼▼▼ [수정] 오디오 파일을 ComfyUI input 폴더로 복사 및 경로 설정 ▼▼▼ ---
                # 절대 경로 대신, ComfyUI input 폴더에 복사 후 파일명만 전달해야 안전합니다.
                audio_filename = f"{scene_id}_song.wav"  # 씬 ID를 붙여 겹침 방지
                comfy_audio_dst = comfy_input_dir / audio_filename

                try:
                    shutil.copy2(str(scene_audio_path), str(comfy_audio_dst))

                    # Node 125 (Audio): 복사된 파일명만 전달
                    if "125" in graph:
                        graph["125"]["inputs"]["audio"] = audio_filename

                except Exception as e:
                    _notify(f"  -> [WARN] 오디오 파일 복사 실패(절대경로 시도): {e}")
                    # 복사 실패 시 기존처럼 절대 경로 시도 (폴백)
                    if "125" in graph:
                        graph["125"]["inputs"]["audio"] = str(scene_audio_path)
                # --- ▲▲▲ [수정 끝] ▲▲▲ ---

                # Node 284 (Image): 씬 대표 이미지
                img_name = f"{scene_id}.png"
                # ★ 수정됨: imgs_dir 변수 대신 project_dir_path를 직접 사용하여 경로 생성
                src_img = project_dir_path / "imgs" / img_name

                if src_img.exists():
                    shutil.copy2(str(src_img), comfy_input_dir / img_name)
                    if "284" in graph: graph["284"]["inputs"]["image"] = img_name
                else:
                    _notify(f"  -> [ERR] 대표 이미지({img_name}) 없음.")
                    continue

                # Node 245, 246 (Res): UI 설정값
                if "245" in graph: graph["245"]["inputs"]["value"] = target_w
                if "246" in graph: graph["246"]["inputs"]["value"] = target_h

                # Node 241 (Prompt): 립싱크 프롬프트
                if "241" in graph: graph["241"]["inputs"]["positive_prompt"] = lync_prompt

                # Node 270 (Max frames): 계산된 전체 프레임 수 주입 (One-Pass 핵심)
                if "270" in graph: graph["270"]["inputs"]["value"] = frame_length_val

                # Node 128 (Steps): UI 설정값
                if "128" in graph: graph["128"]["inputs"]["steps"] = target_steps

                # Node 131 (Save): 파일명 접두사
                if "131" in graph: graph["131"]["inputs"]["filename_prefix"] = f"lipsync/{scene_id}"

                # (3) 실행 (One-Pass)
                _notify(f"  -> ComfyUI 요청 (Duration: {scene_duration:.2f}s, Frames: {frame_length_val})...")
                try:
                    # 타임아웃 넉넉하게
                    res = _submit_and_wait_comfy_func(comfy_host, graph, timeout=1200, poll=2.0,
                                                      on_progress=lambda x: None)

                    # 결과 다운로드 (Node 131 출력)
                    out_node = res.get("outputs", {}).get("131", {})
                    vid_list = out_node.get("videos") or out_node.get("gifs") or []

                    if vid_list:
                        fname = vid_list[0]['filename']
                        subfolder = vid_list[0]['subfolder']
                        r = requests.get(f"{comfy_host}/view", params={"filename": fname, "subfolder": subfolder},
                                         timeout=60)

                        # 바로 최종 RAW 파일로 저장 (One-Pass이므로)
                        with open(scene_raw_norm, "wb") as f:
                            f.write(r.content)

                        if scene_raw_norm.exists():
                            _notify(f"  -> [LipSync] 길이 보정: {frame_length_val} 프레임으로 자르기...")
                            _i2v_trim_tail(
                                scene_raw_norm,  # 입력 파일
                                scene_raw_norm,  # 출력 파일 (덮어쓰기)
                                int(frame_length_val),  # 목표 프레임 수 (video.json 기준)
                                int(base_fps_in)  # FPS
                            )
                        _notify(f"  -> [LipSync] 생성 완료: {scene_raw_norm.name}")
                        continue  # ★ 성공 시 다음 씬으로 건너뜀 (일반 모드 실행 안 함)
                    else:
                        _notify(f"  -> [ERR] 결과물 없음. 일반 모드로 재시도합니다.")
                except Exception as e:
                    _notify(f"  -> [ERR] ComfyUI 실행 중 오류: {e}")
                    # 실패 시 일반 모드로 넘어감 (continue 안 함)



        # ── 세그먼트 / 프롬프트 구성 ──────────────────────────────
        segments_list: List[Tuple[int, int]] = []
        segment_prompts: List[str] = []

        base_scene_pos = str(
            scene_item.get("prompt_movie") or scene_item.get("prompt") or ""
        ).strip()

        # scene-level negative
        scene_negative = str(scene_item.get("prompt_negative") or "").strip()
        scene_negative_full = ", ".join(
            x for x in [base_negative, scene_negative] if x
        ).strip()

        frame_segs_doc = scene_item.get("frame_segments")

        if isinstance(frame_segs_doc, list) and frame_segs_doc:
            base_segments: List[Tuple[int, int]] = []
            base_prompts: List[str] = []

            for seg_obj in frame_segs_doc:
                if not isinstance(seg_obj, dict):
                    continue
                try:
                    seg_start = int(seg_obj.get("start_frame", 0))
                    seg_end = int(seg_obj.get("end_frame", 0))
                except Exception:
                    continue
                if seg_end <= seg_start:
                    continue

                base_segments.append((seg_start, seg_end))
                base_prompts.append(
                    str(seg_obj.get("prompt_movie") or "").strip()
                )

            if base_segments:
                sorted_pairs = sorted(
                    zip(base_segments, base_prompts),
                    key=lambda item_pair: int(item_pair[0][0]),
                )
                sorted_segments = [pair[0] for pair in sorted_pairs]
                sorted_prompts = [pair[1] for pair in sorted_pairs]

                overlap_frames = OVERLAP_FRAMES
                pad_tail_frames = PAD_TAIL_FRAMES

                idx = 0
                while idx < len(sorted_segments):
                    seg1 = sorted_segments[idx]
                    prompt1 = sorted_prompts[idx] or ""
                    if idx + 1 < len(sorted_segments):
                        seg2 = sorted_segments[idx + 1]
                        prompt2 = sorted_prompts[idx + 1] or ""
                    else:
                        seg2 = seg1
                        prompt2 = ""

                    start_valid = int(seg1[0])
                    end_valid = int(seg2[1])

                    if idx == 0:
                        gen_start = start_valid
                    else:
                        gen_start = max(0, start_valid - overlap_frames)

                    gen_end = end_valid + pad_tail_frames
                    segments_list.append((gen_start, gen_end))

                    if prompt1 or prompt2 or base_scene_pos:
                        chunk_prompt_parts: List[str] = []
                        if prompt1:
                            chunk_prompt_parts.append(prompt1)
                        if prompt2 and prompt2 != prompt1:
                            chunk_prompt_parts.append(prompt2)
                        if base_scene_pos:
                            chunk_prompt_parts.append(
                                f"({base_scene_pos}) style, background only, do not change main action"
                            )
                        segment_prompts.append(" ".join(chunk_prompt_parts))
                    else:
                        segment_prompts.append(base_scene_pos)

                    idx += 2

        if not segments_list:
            segments_list = _i2v_plan_segments(
                frame_length_val,
                base_chunk=CHUNK_BASE_FRAMES,
                overlap_val=OVERLAP_FRAMES,
                pad_tail_val=PAD_TAIL_FRAMES,
            )

        seg_count = len(segments_list)
        _notify(
            f"[RAW] scene={scene_id} duration={scene_duration:.3f}s "
            f"frames={frame_length_val} segments={seg_count}"
        )

        # 캐릭터 스펙
        scene_character_specs: Dict[str, int] = {}
        try:
            scene_chars_list = (
                scene_item.get("characters") or scene_item.get("character_objs")
            )
            if not scene_chars_list:
                scene_chars_list = (
                    video_doc.get("characters")
                    or video_doc.get("character_objs")
                    or []
                )
            for scene_char_raw in scene_chars_list:
                parsed = _parse_character_spec_i2v(scene_char_raw)
                parsed_id_val = parsed.get("id")
                parsed_idx_val = parsed.get("index")
                if parsed_id_val:
                    if parsed_idx_val is None:
                        scene_character_specs[parsed_id_val] = 0
                    else:
                        scene_character_specs[parsed_id_val] = int(parsed_idx_val)
        except Exception:
            pass

        _notify(
            f"[RAW][CHAR] scene={scene_id} specs="
            f"{ {k: scene_character_specs[k] for k in sorted(scene_character_specs.keys())} }"
        )

        # ───────────────────────── 세그먼트별 RAW 생성 ─────────────────────────
        chunk_paths: List[Path] = []
        for chunk_index, (start_f, end_f) in enumerate(segments_list):
            length_f = end_f - start_f
            target_chunk_duration = float(length_f) / float(max(base_fps_in, 1))

            raw_filename_fixed = f"{scene_id}_seg{chunk_index + 1:02d}_raw.mp4"
            raw_path = scene_out_dir / raw_filename_fixed
            chunk_paths.append(raw_path)

            if raw_path.exists() and raw_path.stat().st_size > 0:
                _notify(
                    f"[RAW] {scene_id} Seg{chunk_index + 1}/{seg_count} "
                    f"이미 존재 → 스킵"
                )
                continue

            _notify(
                f"[RAW] {scene_id} Seg{chunk_index + 1}/{seg_count} 생성 시작 "
                f"(frames={start_f}~{end_f}, len={length_f}, "
                f"{target_chunk_duration:.3f}s @ {base_fps_in}fps)"
            )

            graph_chunk = json_mod.loads(json_mod.dumps(graph_origin))
            fixed_seed = random.randint(1, 9999999999)
            for nid in ["116", "117", "391"]:
                if nid in graph_chunk:
                    graph_chunk[nid]["inputs"]["noise_seed"] = fixed_seed

            # 해상도/스텝 주입
            for nid in RES_NODE_IDS:
                if nid in graph_chunk:
                    graph_chunk[nid]["inputs"]["width"] = target_w
                    graph_chunk[nid]["inputs"]["height"] = target_h
            for nid in STEPS_NODE_IDS:
                if nid in graph_chunk:
                    graph_chunk[nid]["inputs"]["steps"] = target_steps

            # 프롬프트
            if segment_prompts and chunk_index < len(segment_prompts):
                pos_txt = segment_prompts[chunk_index]
            else:
                pos_txt = base_scene_pos
            neg_txt = scene_negative_full

            for nid_key, node in graph_chunk.items():
                if "inputs" not in node:
                    continue
                inp = node["inputs"]
                meta_title = str(node.get("_meta", {}).get("title", "")).lower()
                if "text" in inp:
                    if "neg" in meta_title:
                        inp["text"] = neg_txt
                    elif "pos" in meta_title or nid_key == "135":
                        inp["text"] = pos_txt

            # ReActor 설정 (축약 버전)
            reactor_setup = {
                "173": ("177", "female_01"),
                "174": ("178", "male_01"),
                "175": ("176", "other"),
            }
            scene_chars_map: Dict[str, int] = {}
            for c in (scene_item.get("characters") or []):
                if isinstance(c, str) and ":" in c:
                    cid, cidx = c.split(":", 1)
                    try:
                        scene_chars_map[cid.strip()] = int(cidx)
                    except Exception:
                        scene_chars_map[cid.strip()] = 0
                elif isinstance(c, dict):
                    cid = c.get("id")
                    if cid:
                        scene_chars_map[cid] = int(c.get("index", 0))
                else:
                    scene_chars_map[str(c)] = 0

            for rid, (lid, char_key) in reactor_setup.items():
                if rid not in graph_chunk or lid not in graph_chunk:
                    continue

                # 1. 캐릭터 이미지 파일 찾기 & 복사 (무조건 수행)
                char_img_path = None
                for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                    p = character_dir_base / f"{char_key}{ext}"
                    if p.exists():
                        char_img_path = p
                        break

                # 파일이 있으면 ComfyUI input으로 복사하고 노드에 경로 설정
                if char_img_path:
                    try:
                        shutil.copy2(str(char_img_path), comfy_input_dir / char_img_path.name)
                        graph_chunk[lid]["inputs"]["image"] = char_img_path.name
                    except Exception as e:
                        _notify(f"[WARN] 캐릭터 복사 실패({char_key}): {e}")

                # 2. 씬 등장 여부에 따라 ReActor 노드 켜기/끄기
                enabled = False
                if char_key in scene_chars_map:
                    if char_img_path:
                        face_idx = scene_chars_map[char_key]
                        graph_chunk[rid]["inputs"]["enabled"] = True
                        graph_chunk[rid]["inputs"]["input_faces_index"] = str(face_idx)
                        enabled = True
                    else:
                        _notify(f"[WARN] {char_key} 캐릭터가 씬에 필요하지만 이미지가 없습니다.")

                if not enabled:
                    graph_chunk[rid]["inputs"]["enabled"] = False

            # 첫 세그먼트는 대표 이미지, 이후는 이전 RAW 세그먼트에서 ref 이미지 추출
            if chunk_index == 0:
                _notify(
                    f"[RAW][IMG] Seg1 → 대표이미지 사용 "
                    f"scene_id={scene_id}, img_file={scene_item.get('img_file')}"
                )
                _inject_scene_main_image_i2v(
                    graph_chunk, scene_item, comfy_input_dir, _notify
                )
            else:
                prev_raw_filename = f"{scene_id}_seg{chunk_index:02d}_raw.mp4"
                prev_raw_path = scene_out_dir / prev_raw_filename
                prev_seg = segments_list[chunk_index - 1]
                prev_start = int(prev_seg[0])
                prev_end = int(prev_seg[1])
                prev_len = prev_end - prev_start
                base_chunk_len = max(1, prev_len - PAD_TAIL_FRAMES)
                local_frame = base_chunk_len - OVERLAP_FRAMES
                if local_frame < 0:
                    local_frame = 0
                if local_frame >= prev_len:
                    local_frame = max(0, prev_len - 1)
                target_time = float(local_frame) / float(max(base_fps_in, 1))
                _notify(
                    f"[RAW][IMG] Seg{chunk_index + 1}용 ref 추출 from="
                    f"{prev_raw_path.name} (local_frame={local_frame}, "
                    f"time={target_time:.3f}s)"
                )
                if prev_raw_path.exists():
                    ref_img_path = _i2v_extract_still_frame_by_time(
                        prev_raw_path,
                        target_time,
                        comfy_input_dir,
                        prefix=f"seg{chunk_index}",
                    )
                    if ref_img_path:
                        _inject_scene_main_image_i2v(
                            graph_chunk,
                            {"img_file": str(ref_img_path)},
                            comfy_input_dir,
                            _notify,
                        )
                    else:
                        _inject_scene_main_image_i2v(
                            graph_chunk, scene_item, comfy_input_dir, _notify
                        )
                else:
                    _inject_scene_main_image_i2v(
                        graph_chunk, scene_item, comfy_input_dir, _notify
                    )

            # RAW 저장: Node 167만 사용, Node 172는 save_output=False
            if "167" in graph_chunk:
                graph_chunk["167"]["inputs"]["filename_prefix"] = (
                    f"temp_raw/{scene_id}_seg{chunk_index + 1:02d}_raw"
                )
                graph_chunk["167"]["inputs"]["save_output"] = True
                graph_chunk["167"]["inputs"]["frame_rate"] = int(base_fps_in)

            if "172" in graph_chunk:
                graph_chunk["172"]["inputs"]["save_output"] = False

            _notify(
                f"[RAW] {scene_id} Seg{chunk_index + 1} → Comfy 실행 (RAW 전용)"
            )

            try:
                res = _submit_and_wait_comfy_func(
                    comfy_host,
                    graph_chunk,
                    timeout=10000,
                    poll=10.0,
                    on_progress=lambda _prog: None,  # 외부 d 이름 가리기 방지
                )

                out_raw_node = res.get("outputs", {}).get("167", {})
                vid_raw_list = (
                    out_raw_node.get("videos") or out_raw_node.get("gifs") or []
                )
                if vid_raw_list:
                    fname_raw = vid_raw_list[0]["filename"]
                    r_raw = requests.get(
                        f"{comfy_host}/view",
                        params={
                            "filename": fname_raw,
                            "subfolder": vid_raw_list[0]["subfolder"],
                        },
                        timeout=60,
                    )
                    with open(raw_path, "wb") as f_raw:
                        f_raw.write(r_raw.content)
                    _notify(f"[RAW] Raw 파일 저장 완료: {raw_path.name}")
                else:
                    raise RuntimeError(
                        "RAW 파일(Node 167) 출력을 찾을 수 없습니다."
                    )

            except Exception as e:
                _notify(f"[RAW][ERR] 세그먼트 생성 실패: {e}")
                continue

            if raw_path.exists():
                target_frames_val = int(
                    round(target_chunk_duration * float(base_fps_in))
                )
                _i2v_trim_tail(
                    raw_path,
                    raw_path,
                    target_frames_val,
                    int(base_fps_in),
                )

        valid_chunk_paths = [
            p for p in chunk_paths if p.exists() and p.stat().st_size > 0
        ]
        if not valid_chunk_paths:
            _notify(
                f"[RAW] 씬 {scene_id}의 유효한 RAW 세그먼트가 없어 병합 스킵"
            )
            continue

        if len(valid_chunk_paths) == 1:
            shutil.copy2(str(valid_chunk_paths[0]), str(scene_raw_tmp))
        else:
            current_concat = valid_chunk_paths[0]
            step_idx = 0
            for next_path in valid_chunk_paths[1:]:
                step_idx += 1
                temp_out = scene_out_dir / f"{scene_id}_raw_step_{step_idx:03d}.mp4"
                _i2v_concat_ab(current_concat, next_path, temp_out)
                current_concat = temp_out

            shutil.copy2(str(current_concat), str(scene_raw_tmp))

        # FPS 정규화 + tail trim
        _i2v_norm_fps_and_size(
            scene_raw_tmp,
            scene_raw_norm,
            int(base_fps_in),
        )

        target_duration_sec = float(frame_length_val) / float(
            max(base_fps_in, 1)
        )
        target_total_frames = int(
            round(target_duration_sec * float(max(base_fps_in, 1)))
        )

        _notify(
            f"[RAW][TRIM] scene={scene_id} duration={target_duration_sec:.3f}s "
            f"FPS={base_fps_in} target_frames={target_total_frames}"
        )

        _i2v_trim_tail(
            scene_raw_norm,
            scene_raw_norm,
            target_total_frames,
            int(base_fps_in),
        )

        try:
            if scene_raw_tmp.exists():
                scene_raw_tmp.unlink()
        except OSError:
            pass

        _notify(f"[RAW] 씬 {scene_id} RAW 처리 완료 → {scene_raw_norm}")




# ─────────────────────────────────────────────────────────────────────────────
# [2단계] 보간 및 업스케일 (별도 워크플로우 사용)
# ─────────────────────────────────────────────────────────────────────────────
# real_use
def Interpolation_upscale(
    project_dir: str,
    total_frames: int,
    *,  # total_frames는 현재 사용 안 하지만 시그니처 맞추기 용
    ui_width: Optional[int] = None,
    ui_height: Optional[int] = None,
    ui_fps: Optional[int] = None,
    ui_steps: Optional[int] = None,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    """
    [2단계] RAW 씬 영상을 Interpolation_upscale.json 으로
    GIMMVFI + SeedVR2 업스케일하는 함수.

    입력:
      - clips/{scene_id}_raw.mp4 또는 clips/{scene_id}/{scene_id}_raw.mp4
    출력:
      - clips/{scene_id}.mp4 (업스케일/보간 완료본)
    """






    ffmpeg_exe_val = getattr(settings_obj, "FFMPEG_EXE", "ffmpeg")
    ffprobe_exe_val = getattr(settings_obj, "FFPROBE_EXE", "ffprobe")
    comfy_host = getattr(settings_obj, "COMFY_HOST", "http://127.0.0.1:8188")
    comfy_input_dir = Path(
        str(getattr(settings_obj, "COMFY_INPUT_DIR", Path(project_dir) / "input"))
    )
    _ensure_dir_func(comfy_input_dir)

    def _notify(msg: str) -> None:
        print(msg)
        if on_progress:
            try:
                on_progress({"msg": msg})
            except Exception:
                pass

    def _probe_fps_and_size(path_obj: Path) -> Tuple[float, int, int]:
        fps_val = 0.0
        w = h = 0
        try:
            cmd = [
                ffprobe_exe_val,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate,width,height",
                "-of", "default=nokey=1:noprint_wrappers=1",
                str(path_obj),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]

            for ln in lines:
                # r_frame_rate e.g. "16/1"
                if "/" in ln and fps_val == 0.0:
                    num, den = ln.split("/")
                    num_f = float(num)
                    den_f = float(den) if float(den) != 0 else 1.0
                    fps_val = num_f / den_f
                elif "x" in ln:
                    try:
                        w, h = ln.split("x")
                        w = int(w)
                        h = int(h)
                    except Exception:
                        pass
        except Exception:
            pass

        if fps_val <= 0.0:
            fps_val = 16.0
        if w <= 0 or h <= 0:
            w, h = 720, 1280

        return fps_val, w, h

    # JSON 로드
    project_dir_path = Path(project_dir).resolve()
    jsons_dir_conf = getattr(settings_obj, "JSONS_DIR", str(project_dir_path / "jsons"))
    interp_json_path = Path(str(jsons_dir_conf)) / "Interpolation_upscale.json"
    if not interp_json_path.is_file():
        raise FileNotFoundError(f"Interpolation_upscale.json 없음: {interp_json_path}")

    with interp_json_path.open("r", encoding="utf-8") as f:
        interp_graph_origin = json_mod.load(f)

    video_json_path = project_dir_path / "video.json"
    video_doc = _load_json_func(video_json_path, {}) or {}
    scenes = video_doc.get("scenes") or []
    if not isinstance(scenes, list):
        raise RuntimeError("video.json['scenes']가 리스트가 아닙니다.")

    clips_dir = project_dir_path / "clips"
    _ensure_dir_func(clips_dir)

    # Node ID
    LOAD_NODE = "5"
    VFI_NODE = "1"
    UPSCALE_NODE = "3"
    COMBINE_NODE = "2"

    for idx, scene_item in enumerate(scenes, start=1):
        if not isinstance(scene_item, dict):
            continue

        scene_id = str(scene_item.get("id") or f"scene_{idx:03d}")

        # ------------------------------
        # RAW 파일을 두 위치 모두 확인
        # ------------------------------
        raw_path_sub = clips_dir / scene_id / f"{scene_id}_raw.mp4"  # 예: clips/t_001/t_001_raw.mp4
        raw_path_root = clips_dir / f"{scene_id}_raw.mp4"            # 예: clips/t_001_raw.mp4

        if raw_path_sub.exists() and raw_path_sub.stat().st_size > 0:
            raw_path = raw_path_sub
        elif raw_path_root.exists() and raw_path_root.stat().st_size > 0:
            raw_path = raw_path_root
        else:
            _notify(f"[UP] scene_id={scene_id} RAW 없음 → 스킵 ({raw_path_sub} / {raw_path_root})")
            continue

        # 최종 출력
        out_path = clips_dir / f"{scene_id}.mp4"
        if out_path.exists() and out_path.stat().st_size > 0:
            _notify(f"[UP] scene_id={scene_id} 업스케일 이미 존재 → 스킵")
            continue

        _notify(f"[UP] scene_id={scene_id} 업스케일 시작")

        # RAW → Comfy input 복사
        comfy_in_name = f"{scene_id}_raw_input.mp4"
        comfy_in_path = comfy_input_dir / comfy_in_name
        shutil.copy2(str(raw_path), str(comfy_in_path))

        # RAW fps/size
        raw_fps, raw_w, raw_h = _probe_fps_and_size(raw_path)

        # UI FPS 적용
        if ui_fps and ui_fps > 0:
            raw_fps = float(ui_fps)

        target_fps = raw_fps * 2  # 보간 2배

        # SeedVR2 resolution
        if ui_width and ui_height and ui_width > 0 and ui_height > 0:
            res_val = max(int(ui_width), int(ui_height))
        else:
            res_val = max(raw_w, raw_h, 720)

        # 그래프 클론
        graph = json_mod.loads(json_mod.dumps(interp_graph_origin))

        # LoadVideo
        if LOAD_NODE in graph:
            graph[LOAD_NODE]["inputs"]["video"] = comfy_in_name
            graph[LOAD_NODE]["inputs"]["force_rate"] = 0

        # SeedVR2 업스케일
        if UPSCALE_NODE in graph:
            graph[UPSCALE_NODE]["inputs"]["resolution"] = int(res_val)

        # VideoCombine
        if COMBINE_NODE in graph:
            graph[COMBINE_NODE]["inputs"]["frame_rate"] = int(round(target_fps))
            graph[COMBINE_NODE]["inputs"]["filename_prefix"] = f"up/{scene_id}"
            graph[COMBINE_NODE]["inputs"]["save_output"] = True

        _notify(f"[UP] scene={scene_id} raw_fps={raw_fps:.3f} → target_fps={target_fps:.3f}, res={res_val}")

        # 제출
        try:
            res = _submit_and_wait_comfy_func(
                comfy_host,
                graph,
                timeout=10000,
                poll=10.0,
                on_progress=lambda d: None,
            )

            out_node = res.get("outputs", {}).get(COMBINE_NODE, {})
            vids = out_node.get("videos") or out_node.get("gifs") or []
            if not vids:
                raise RuntimeError("VideoCombine 출력 없음")

            fname = vids[0]["filename"]
            r = requests.get(
                f"{comfy_host}/view",
                params={"filename": fname, "subfolder": vids[0]["subfolder"]},
                timeout=120,
            )
            with open(out_path, "wb") as f:
                f.write(r.content)

            _notify(f"[UP] scene={scene_id} 업스케일 완료 → {out_path}")

        except Exception as e:
            _notify(f"[UP][ERR] scene={scene_id} 실패: {e}")
            continue









def xfade_concat(
    clip_paths: List[Path],
    overlap_frames: int,
    fps: Optional[int] = None,
    *,
    audio_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    # 과거 호출 호환 인자 (기존 시그니처 유지)
    out_fps: Optional[int] = None,
    scale_w: Optional[int] = None,
    scale_h: Optional[int] = None,
    work_dir: Optional[Path] = None,
    # 신규 옵션: True=페이드, False=하드컷 병합
    xfade: bool = True,
) -> Path:
    """
    연속 MP4들을 연결합니다.

    - 최신: xfade_concat(clip_paths, overlap_frames, fps, *, audio_path=None, out_path=None, xfade=True)
    - 과거: xfade_concat(work_dir=..., clip_paths=..., out_fps=..., overlap_frames=..., out_path=..., scale_w=..., scale_h=..., audio_path=...)
    반환: 최종 mp4 경로

    [확장/보존]
      - xfade=True  → 기존과 동일하게 크로스페이드로 연결
      - xfade=False → 겹침 구간을 '뒤 청크'가 그대로 덮어쓰는 하드컷 병합
      - 기존 로깅/미러 보관/인자 호환성 100% 유지
    """
    import shutil
    import subprocess

    if not clip_paths:
        raise ValueError("clip_paths가 비어있습니다.")
    if overlap_frames is None or int(overlap_frames) < 0:
        raise ValueError("overlap_frames가 유효하지 않습니다.")

    target_fps = int(out_fps if out_fps is not None else (fps if fps is not None else 24))
    if target_fps <= 0:
        target_fps = 24

    first_clip_path = Path(clip_paths[0])
    base_dir = first_clip_path.parent

    if work_dir is None:
        if base_dir.name == "xfade_work":
            work_dir = base_dir
        else:
            work_dir = base_dir / "xfade_work"
    else:
        work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if out_path is None:
        out_path = base_dir / "final.mp4"
    out_path = Path(out_path)
    mirror_dir = out_path.parent / out_path.stem
    mirror_dir.mkdir(parents=True, exist_ok=True)
    print(f"[XC-MIRROR] dir={mirror_dir}")

    # 기본 출력 해상도
    out_w = int(scale_w) if scale_w else 1280
    out_h = int(scale_h) if scale_h else 720

    try:
        ffmpeg_bin = globals().get("FFMPEG_EXE", "ffmpeg")
    except Exception:
        ffmpeg_bin = "ffmpeg"
    try:
        ffprobe_bin = globals().get("FFPROBE_EXE", "ffprobe")
    except Exception:
        ffprobe_bin = "ffprobe"

    def _probe_video_info(path_for_probe: Path) -> Dict[str, Any]:
        probe_map: Dict[str, Any] = {"path": str(path_for_probe), "exists": path_for_probe.exists()}
        if not path_for_probe.exists():
            return probe_map
        cmd_probe = [
            ffprobe_bin,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path_for_probe),
        ]
        proc_probe = subprocess.run(
            cmd_probe,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        if proc_probe.returncode == 0:
            try:
                parsed = _json.loads(proc_probe.stdout)
            except _json.JSONDecodeError:
                probe_map["ffprobe_out"] = proc_probe.stdout
            else:
                vstreams = [s for s in parsed.get("streams", []) if s.get("codec_type") == "video"]
                if vstreams:
                    vs0 = vstreams[0]
                    probe_map.update({
                        "width": vs0.get("width"),
                        "height": vs0.get("height"),
                        "r_frame_rate": vs0.get("r_frame_rate"),
                        "avg_frame_rate": vs0.get("avg_frame_rate"),
                        "nb_frames": vs0.get("nb_frames"),
                        "time_base": vs0.get("time_base"),
                        "pix_fmt": vs0.get("pix_fmt"),
                    })
                fmt_obj = parsed.get("format") or {}
                dur_val = fmt_obj.get("duration")
                if dur_val is not None:
                    try:
                        probe_map["duration_sec"] = float(dur_val)
                    except (TypeError, ValueError):
                        pass
                bit_val = fmt_obj.get("bit_rate")
                if bit_val is not None:
                    probe_map["bit_rate"] = bit_val
        else:
            probe_map["ffprobe_out"] = proc_probe.stdout
        try:
            probe_map["size_bytes"] = path_for_probe.stat().st_size
        except OSError:
            pass
        return probe_map

    # 여기서만 보강: 호출자가 scale을 안 줬을 때만 첫 클립 크기를 따른다.
    if scale_w is None or scale_h is None:
        first_info = _probe_video_info(first_clip_path)
        first_w = first_info.get("width")
        first_h = first_info.get("height")
        if isinstance(first_w, int) and isinstance(first_h, int) and first_w > 0 and first_h > 0:
            out_w = first_w
            out_h = first_h

    def _normalize_one(src_path_norm: Path, dst_path_norm: Path) -> None:
        cmd_norm = [
            ffmpeg_bin,
            "-y",
            "-fflags",
            "+genpts",
            "-i",
            str(src_path_norm),
            "-an",
            "-vf",
            (
                f"fps={target_fps},"
                f"scale=w={out_w}:h={out_h}:force_original_aspect_ratio=decrease,"
                f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2,"
                "setsar=1,format=yuv420p,setpts=PTS-STARTPTS"
            ),
            "-r",
            f"{target_fps}",
            "-vsync",
            "cfr",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(dst_path_norm),
        ]
        proc_norm = subprocess.run(
            cmd_norm,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        if proc_norm.returncode != 0:
            raise RuntimeError(f"ffmpeg 실패(정규화):\n{proc_norm.stdout}")

    def _xfade_two(
        cur_path_in: Path,
        nxt_path_in: Path,
        out_path_pair: Path,
        overlap_frames_in: int,
    ) -> None:
        norm_a_path = work_dir / "_norm_a.mp4"
        norm_b_path = work_dir / "_norm_b.mp4"
        _normalize_one(cur_path_in, norm_a_path)
        _normalize_one(nxt_path_in, norm_b_path)

        info_a = _probe_video_info(norm_a_path)
        info_b = _probe_video_info(norm_b_path)

        def _frames_from_info(info_map: Dict[str, Any]) -> int:
            nb_val = info_map.get("nb_frames")
            if isinstance(nb_val, str) and nb_val.isdigit():
                try:
                    return int(nb_val)
                except ValueError:
                    pass
            dur_val_local = info_map.get("duration_sec")
            if isinstance(dur_val_local, (int, float)) and dur_val_local > 0:
                return max(0, int(dur_val_local * target_fps))
            return 0

        frames_a = _frames_from_info(info_a)
        frames_b = _frames_from_info(info_b)
        fade_frames = max(0, int(overlap_frames_in))
        if frames_a <= 0 or frames_b <= 0:
            raise RuntimeError("정규화본의 프레임 수 계산 실패")

        offset_sec = float(max(0, frames_a - fade_frames)) / float(target_fps)
        duration_sec = float(fade_frames) / float(target_fps)

        print(
            f"[XC-PLAN] fade={fade_frames}f ({duration_sec:.3f}s), "
            f"offset={offset_sec:.3f}s, expected≥ {(frames_a + frames_b - fade_frames)}f"
        )

        cmd_xf = [
            ffmpeg_bin,
            "-y",
            "-fflags",
            "+genpts",
            "-i",
            str(norm_a_path),
            "-i",
            str(norm_b_path),
            "-filter_complex",
            (
                "[0:v]format=yuv420p,setsar=1[v0];"
                "[1:v]format=yuv420p,setsar=1[v1];"
                f"[v0][v1]xfade=transition=fade:duration={duration_sec:.6f}:offset={offset_sec:.6f}[v]"
            ),
            "-map",
            "[v]",
            "-r",
            f"{target_fps}",
            "-fps_mode",
            "cfr",
            "-vsync",
            "cfr",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(out_path_pair),
        ]
        proc_xf = subprocess.run(
            cmd_xf,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        if proc_xf.returncode != 0:
            raise RuntimeError(f"ffmpeg 실패(_xfade_two):\n{proc_xf.stdout}")

        out_info = _probe_video_info(out_path_pair)
        print(
            f"[XC-OUT] {out_path_pair.name}: sec={out_info.get('duration_sec', '?')} "
            f"frames={out_info.get('nb_frames', '?')} @fps={target_fps}"
        )

    def _concat_cut_no_fade(
        ffmpeg_exe: str,
        ffprobe_exe: str,
        clip_paths_cut: List[Path],
        overlap_frames_cut: int,
        fps_cut: int,
        out_path_cut: Path,
        work_dir_cut: Path,
    ) -> bool:
        import subprocess as _sub
        import shutil as _sh

        norm_list: List[Path] = []
        for idx_norm, src_norm in enumerate(clip_paths_cut):
            dst_norm = work_dir_cut / f"_concat_norm_{idx_norm:03d}.mp4"
            cmd_norm2 = [
                ffmpeg_exe,
                "-y",
                "-fflags",
                "+genpts",
                "-i",
                str(src_norm),
                "-an",
                "-vf",
                (
                    f"fps={fps_cut},"
                    f"scale=w={out_w}:h={out_h}:force_original_aspect_ratio=decrease,"
                    f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2,"
                    "setsar=1,format=yuv420p,setpts=PTS-STARTPTS"
                ),
                "-r",
                f"{fps_cut}",
                "-vsync",
                "cfr",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(dst_norm),
            ]
            proc_n2 = _sub.run(
                cmd_norm2,
                stdout=_sub.PIPE,
                stderr=_sub.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=False,
            )
            if proc_n2.returncode != 0:
                raise RuntimeError(f"ffmpeg 실패(정규화, no-fade):\n{proc_n2.stdout}")
            norm_list.append(dst_norm)

        trimmed_list: List[Path] = []
        for idx_trim, src_trim in enumerate(norm_list):
            cmd_probe2 = [
                ffprobe_exe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(src_trim),
            ]
            proc_p2 = _sub.run(
                cmd_probe2,
                stdout=_sub.PIPE,
                stderr=_sub.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=False,
            )
            nb_frames_int = 0
            nb_frames_str = (proc_p2.stdout or "").strip()
            if nb_frames_str.isdigit():
                try:
                    nb_frames_int = int(nb_frames_str)
                except ValueError:
                    nb_frames_int = 0

            if idx_trim == 0 or overlap_frames_cut <= 0:
                trimmed_list.append(src_trim)
                continue

            keep_frames = max(0, nb_frames_int - overlap_frames_cut)
            if keep_frames <= 0:
                continue

            trim_out = work_dir_cut / f"_concat_trim_{idx_trim:03d}.mp4"
            cmd_trim = [
                ffmpeg_exe,
                "-y",
                "-i",
                str(src_trim),
                "-vf",
                f"select='gte(n\\,{overlap_frames_cut})',setpts=N/FRAME_RATE/TB",
                "-r",
                f"{fps_cut}",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(trim_out),
            ]
            proc_t = _sub.run(
                cmd_trim,
                stdout=_sub.PIPE,
                stderr=_sub.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=False,
            )
            if proc_t.returncode != 0:
                raise RuntimeError(f"ffmpeg 실패(트림, no-fade):\n{proc_t.stdout}")
            trimmed_list.append(trim_out)

        if not trimmed_list:
            return False

        concat_list_path = work_dir_cut / "_concat_list.txt"
        with open(concat_list_path, "w", encoding="utf-8", newline="\n") as f_list:
            for p_item in trimmed_list:
                p_str = str(p_item).replace("\\", "\\\\").replace("'", "\\'")
                f_list.write(f"file '{p_str}'\n")

        cmd_concat = [
            ffmpeg_exe,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
            "-c:v",
            "copy",
        ]
        if audio_path is None:
            cmd_concat.append(str(out_path_cut))
        else:
            cmd_concat.extend(["-c:a", "aac", "-shortest", str(out_path_cut)])

        proc_c = _sub.run(
            cmd_concat,
            stdout=_sub.PIPE,
            stderr=_sub.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        if proc_c.returncode != 0:
            raise RuntimeError(f"ffmpeg 실패(concat, no-fade):\n{proc_c.stdout}")

        try:
            dst_concat = mirror_dir / out_path_cut.name
            _sh.copyfile(out_path_cut, dst_concat)
            print(f"[XC-MIRROR-COPY] concat->{dst_concat.name}")
        except OSError:
            pass

        return True

    print(
        f"[XC-BEGIN] items={len(clip_paths)} fps={target_fps} "
        f"overlap_frames={int(overlap_frames)} out={out_w}x{out_h}"
    )
    for idx_in, clip_path_item in enumerate(clip_paths):
        clip_path_obj = Path(clip_path_item)
        print(f"[XC-IN-{idx_in:03d}] {_probe_video_info(clip_path_obj)}")
        try:
            if clip_path_obj.exists():
                dst_init = mirror_dir / clip_path_obj.name
                if not dst_init.exists():
                    shutil.copyfile(clip_path_obj, dst_init)
                    print(f"[XC-MIRROR-COPY] chunk->{dst_init.name}")
        except OSError:
            pass

    if not xfade:
        ok_merge = _concat_cut_no_fade(
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
            clip_paths_cut=clip_paths,
            overlap_frames_cut=int(overlap_frames),
            fps_cut=target_fps,
            out_path_cut=out_path,
            work_dir_cut=work_dir,
        )
        if not ok_merge:
            raise RuntimeError("하드컷 병합 실패")
        if audio_path and Path(audio_path).exists():
            cmd_mux_nf = [
                ffmpeg_bin,
                "-y",
                "-fflags",
                "+genpts",
                "-i",
                str(out_path),
                "-i",
                str(audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                str(out_path),
            ]
            print(f"[XC-CMD-MUX] {' '.join(cmd_mux_nf)}")
            proc_mux_nf = subprocess.run(
                cmd_mux_nf,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=False,
            )
            print(f"[XC-OUT-MUX] return={proc_mux_nf.returncode}")
            if proc_mux_nf.returncode != 0:
                raise RuntimeError(f"ffmpeg 실패(오디오 합성, no-fade):\n{proc_mux_nf.stdout}")

        fin_map_nf = _probe_video_info(out_path)
        print(f"[XC-FINAL] {fin_map_nf}")
        print(f"[XC-DONE] out_path={out_path}")
        return out_path

    cur_path = work_dir / "cur_000.mp4"
    cmd_first = [
        ffmpeg_bin,
        "-y",
        "-fflags",
        "+genpts",
        "-i",
        str(first_clip_path),
        "-an",
        "-vf",
        (
            f"fps={target_fps},"
            f"scale=w={out_w}:h={out_h}:force_original_aspect_ratio=decrease,"
            f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2,"
            "setsar=1,format=yuv420p,setpts=PTS-STARTPTS"
        ),
        "-r",
        f"{target_fps}",
        "-vsync",
        "cfr",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(cur_path),
    ]
    print(f"[XC-CMD-FIRST] {' '.join(cmd_first)}")
    proc0 = subprocess.run(
        cmd_first,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
    )
    print(f"[XC-OUT-FIRST] return={proc0.returncode}")
    if proc0.returncode != 0:
        raise RuntimeError(f"ffmpeg 실패(초기 cur_000 인코딩):\n{proc0.stdout}")
    print(f"[XC-CUR-000] {_probe_video_info(cur_path)}")

    try:
        dst0 = mirror_dir / cur_path.name
        shutil.copyfile(cur_path, dst0)
        print(f"[XC-MIRROR-COPY] cur_000->{dst0.name}")
    except OSError:
        pass

    for i in range(1, len(clip_paths)):
        next_path = Path(clip_paths[i])
        tmp_out = work_dir / f"cur_{i:03d}.mp4"
        print(f"[XC-STEP] pair={i-1}->{i} out={tmp_out.name}")

        _xfade_two(
            cur_path_in=cur_path,
            nxt_path_in=next_path,
            out_path_pair=tmp_out,
            overlap_frames_in=int(overlap_frames),
        )
        cur_path = tmp_out
        print(f"[XC-CUR-{i:03d}] {_probe_video_info(cur_path)}")

        # ← 여기 이름만 바꿨다
        try:
            norm_a_copy_path = work_dir / "_norm_a.mp4"
            norm_b_copy_path = work_dir / "_norm_b.mp4"
            if norm_a_copy_path.exists():
                dst_a = mirror_dir / f"_norm_a_{i:03d}.mp4"
                shutil.copyfile(norm_a_copy_path, dst_a)
                print(f"[XC-MIRROR-COPY] norm_a->{dst_a.name}")
            if norm_b_copy_path.exists():
                dst_b = mirror_dir / f"_norm_b_{i:03d}.mp4"
                shutil.copyfile(norm_b_copy_path, dst_b)
                print(f"[XC-MIRROR-COPY] norm_b->{dst_b.name}")
            dst_cur = mirror_dir / cur_path.name
            shutil.copyfile(cur_path, dst_cur)
            print(f"[XC-MIRROR-COPY] cur->{dst_cur.name}")
        except OSError:
            pass

    if audio_path and Path(audio_path).exists():
        cmd_mux = [
            ffmpeg_bin,
            "-y",
            "-fflags",
            "+genpts",
            "-i",
            str(cur_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(out_path),
        ]
        print(f"[XC-CMD-MUX] {' '.join(cmd_mux)}")
        proc_mux = subprocess.run(
            cmd_mux,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        print(f"[XC-OUT-MUX] return={proc_mux.returncode}")
        if proc_mux.returncode != 0:
            raise RuntimeError(f"ffmpeg 실패(오디오 합성):\n{proc_mux.stdout}")
    else:
        cmd_mv = [
            ffmpeg_bin,
            "-y",
            "-fflags",
            "+genpts",
            "-i",
            str(cur_path),
            "-c:v",
            "copy",
            "-an",
            str(out_path),
        ]
        print(f"[XC-CMD-MV] {' '.join(cmd_mv)}")
        proc_mv = subprocess.run(
            cmd_mv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        print(f"[XC-OUT-MV] return={proc_mv.returncode}")
        if proc_mv.returncode != 0:
            raise RuntimeError(f"ffmpeg 실패(최종 비디오 산출):\n{proc_mv.stdout}")

    fin_map = _probe_video_info(Path(out_path))
    last_map = _probe_video_info(cur_path)
    same_size = fin_map.get("size_bytes") == last_map.get("size_bytes")
    print(f"[XC-FINAL] {fin_map}")
    print(f"[XC-FINAL==CUR-LAST] same_size={bool(same_size)} out={out_path}")

    try:
        dst_final = mirror_dir / Path(out_path).name
        if Path(out_path).exists():
            shutil.copyfile(Path(out_path), dst_final)
            print(f"[XC-MIRROR-COPY] final->{dst_final.name}")
    except OSError:
        pass

    print(f"[XC-DONE] out_path={out_path}")
    return Path(out_path)
# real_use
def build_missing_images_from_story(
        story_path: str | _Path,
        *,
        ui_width: int,
        ui_height: int,
        steps: int = 28,
        timeout_sec: int = 900,
        poll_sec: float = 3,
        workflow_path: str | _Path | None = None,
        on_progress: Optional[Dict[str, Any] | Callable[[Dict[str, Any]], None]] = None,
) -> List[_Path]:
    """
    story.json(또는 video.json)을 읽어 누락된 씬 이미지를 ComfyUI로 생성한다.

    - SaveImage 노드의 output_path를 동적으로 주입해 /view 404 문제를 줄인다.
    - result_data.outputs의 images 목록에서 type="output" 이미지를 우선 선택한다.
    - /view 호출 시 filename / subfolder / type 파라미터를 모두 로그로 남겨 디버깅에 활용한다.
    """
    # --- [1. 모듈 임포트] ---


    # --- [2. 알림 콜백 함수 (중첩)] ---
    def _notify(stage: str, msg: str = "", **extra: Any) -> None:
        """진행률 콜백을 안전하게 호출하는 헬퍼"""
        if not on_progress:
            return
        data: Dict[str, Any] = {"stage": stage, "msg": msg}
        if extra:
            data.update(extra)
        try:
            if callable(on_progress):
                on_progress(data)
            elif isinstance(on_progress, dict):
                cb_func = on_progress.get("callback")
                if callable(cb_func):
                    cb_func(data)
        except (RuntimeError, ValueError, TypeError) as e_notify_callback:
            print(f"[WARN] build_missing_images_from_story._notify callback failed: {e_notify_callback}", flush=True)

    # --- [3. 경로 및 문서 로드] ---
    try:
        p_story = _Path(story_path).resolve()
        story_dir = p_story.parent if p_story.is_file() else _Path(story_path).resolve()
        if not story_dir.exists():
            raise FileNotFoundError(f"경로 없음: {story_dir}")

        p_video = story_dir / "video.json"
        video_doc = load_json(p_video, {}) or {}
        p_story_json = story_dir / "story.json"
        story_doc = load_json(p_story_json, {}) or {}

    except FileNotFoundError as e_story_path:
        raise e_story_path
    except (IOError, OSError) as e_story_io:
        raise IOError(f"스토리 경로 접근 오류: {e_story_io}") from e_story_io

    paths_v = video_doc.get("paths") or {}
    root_dir = _Path(paths_v.get("root") or story_doc.get("paths", {}).get("root") or story_dir)
    imgs_dir_name = str(paths_v.get("imgs_dir") or story_doc.get("paths", {}).get("imgs_dir") or "imgs")
    img_root = ensure_dir(root_dir / imgs_dir_name)

    try:

        base_jsons = _Path(JSONS_DIR)
    except (ImportError, AttributeError):
        base_jsons = _Path(r"C:\my_games\shorts_make\app\jsons")

    wf_path_resolved = _Path(workflow_path) if workflow_path else (base_jsons / "nunchaku_qwen_image_swap.json")
    if not wf_path_resolved.exists():
        raise FileNotFoundError(f"필수 워크플로 없음: {wf_path_resolved}")

    try:
        with open(wf_path_resolved, "r", encoding="utf-8") as f_graph:
            graph_origin: Dict[str, Any] = _json_loader.load(f_graph)
    except (IOError, OSError, _json_loader.JSONDecodeError) as e_load_wf:
        raise RuntimeError(f"워크플로 로드 실패: {wf_path_resolved}") from e_load_wf

    # --- [4. 워크플로/Comfy 헬퍼] ---
    def _find_nodes(gdict: Dict[str, Any], class_type_str: str) -> List[str]:
        hits: List[str] = []
        for nid, node_item in (gdict or {}).items():
            try:
                if str(node_item.get("class_type")) == class_type_str:
                    hits.append(str(nid))
            except (AttributeError, KeyError, TypeError):
                continue
        return hits

    def _set_input(gdict: Dict[str, Any], nid: str, key: str, val: Any) -> None:
        try:
            gdict[str(nid)].setdefault("inputs", {})[key] = val
        except (KeyError, TypeError, AttributeError):
            _notify("warn", f"[IMG] 워크플로 주입 실패: Node {nid}, Key {key}")

    # Latent / KSampler / Seed 관련
    latent_ids = _find_nodes(graph_origin, "EmptySD3LatentImage") + _find_nodes(graph_origin, "EmptyLatentImage")
    latent_ids = list(dict.fromkeys(latent_ids))
    ksampler_ids = _find_nodes(graph_origin, "KSampler")

    img_seed_from_video = None
    root_seed_val = video_doc.get("t2i_seed_for_workflow")
    if isinstance(root_seed_val, int) and root_seed_val > 0:
        img_seed_from_video = root_seed_val
    else:
        defaults_doc_img = (video_doc.get("defaults") or {}).get("image") or {}
        seed_in_defaults = defaults_doc_img.get("t2i_seed")
        if isinstance(seed_in_defaults, int) and seed_in_defaults > 0:
            img_seed_from_video = seed_in_defaults

    if img_seed_from_video is None:
        img_seed_from_video = _img_seed_random.randint(1, 2_147_483_646)

    for k_id in ksampler_ids:
        try:
            graph_origin[str(k_id)].setdefault("inputs", {})["seed"] = int(img_seed_from_video)
        except (KeyError, TypeError, ValueError, AttributeError):
            _notify("warn", f"[IMG] 시드 주입 실패: KSampler {k_id}")

    # SaveImage 노드 사전 수집
    save_image_ids = _find_nodes(graph_origin, "SaveImage")
    if not save_image_ids:
        _notify("warn", f"[IMG] 워크플로우({wf_path_resolved.name})에 'SaveImage' 노드가 없습니다. API 출력이 실패할 수 있습니다.")

    # --- [ReActor 관련 헬퍼 - 기존 호환 유지용] ---
    def _resolve_face_image_by_name(name: str) -> _Path | None:
        try:

            base_dir_char = _Path(CHARACTER_DIR)
        except (ImportError, AttributeError):
            base_dir_char = _Path(r"C:\my_games\shorts_make\character")
        exts = (".png", ".jpg", ".jpeg", ".webp")
        for ext_item in exts:
            p_path = base_dir_char / f"{name}{ext_item}"
            if p_path.exists():
                return p_path
        return None

    def _pick_scene_character_name(scene: Dict[str, Any]) -> str | None:
        return None

    def _inject_face_to_reactors(gdict: Dict[str, Any], file_name: str) -> List[str]:
        return []

    def _inject_face_image_loaders(_gdict: Dict[str, Any], _file_name: str) -> List[str]:
        return []

    def _parse_first_char_index(scene: Dict[str, Any]) -> int:
        return 0

    # --- [ComfyUI 제출/대기 헬퍼] ---


    def _wait_img(url: str, gdict: Dict[str, Any], *, timeout: int, poll: float,
                  progress_cb: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
        wait_i_counter = 0

        def _relay_progress(prog: Dict[str, Any]) -> None:
            nonlocal wait_i_counter
            raw_msg = str(prog.get("msg") or "")
            if raw_msg.startswith("[MUSIC]"):
                patched_msg = "[IMG]" + raw_msg[len("[MUSIC]"):]
            else:
                patched_msg = "[IMG] " + raw_msg if raw_msg else "[IMG]"
            wait_i_counter += 1
            if wait_i_counter % 50 != 0:
                return
            try:
                progress_cb({"stage": "wait", "msg": patched_msg})
            except (RuntimeError, ValueError, TypeError) as e_relay_callback:
                print(f"[WARN] _relay_progress callback failed: {e_relay_callback}", flush=True)

        return _submit_and_wait_comfy_func(url, gdict, timeout=timeout, poll=poll, on_progress=_relay_progress)

    # --- [5. 씬 루프] ---
    created: List[_Path] = []
    base_url = str(video_doc.get("comfy_host") or story_doc.get("comfy_host") or "http://127.0.0.1:8188").rstrip("/")

    scenes_list: List[Dict[str, Any]] = list(video_doc.get("scenes") or story_doc.get("scenes") or [])
    _notify("begin", f"[IMG] 대상 scenes={len(scenes_list)} wf={wf_path_resolved.name}")

    defaults_v = video_doc.get("defaults") or {}
    defaults_img = defaults_v.get("image") or (story_doc.get("defaults", {}).get("image") if story_doc else {}) or {}
    default_neg = str(defaults_img.get("negative") or "")

    req_count = 0

    for idx, sc in enumerate(scenes_list):
        try:
            sid = str(sc.get("id") or f"scene_{idx:05d}")
            img_path_target = img_root / f"{sid}.png"

            # 이미 있는 이미지 스킵
            scene_img_file = str(sc.get("img_file") or "")
            if scene_img_file:
                p_scene_img = _Path(scene_img_file)
                if not p_scene_img.is_absolute():
                    p_scene_img = img_root / p_scene_img.name
                if p_scene_img.exists():
                    _notify("skip", f"[IMG] {sid} → 존재 (img_file) {p_scene_img.name}")
                    continue
            if img_path_target.exists():
                _notify("skip", f"[IMG] {sid} → 존재 (id) {img_path_target.name}")
                continue

            # 워크플로 복제
            graph_clone = _json_loader.loads(_json_loader.dumps(graph_origin))

            # Latent 크기 주입
            for nid_latent in latent_ids:
                _set_input(graph_clone, nid_latent, "width", int(ui_width))
                _set_input(graph_clone, nid_latent, "height", int(ui_height))

            # 프롬프트 구성
            pos_text_raw = str(sc.get("prompt_movie") or "").strip()
            if not pos_text_raw:
                p_main = str(sc.get("prompt") or "").strip()
                p_img = str(sc.get("prompt_img") or "").strip()
                pos_text_raw = f"{p_main}\n{p_img}" if p_main and p_img else (p_main or p_img)
            neg_text = str(sc.get("prompt_negative") or "") or default_neg

            def _apply_prompts(gdict: Dict[str, Any], pos: str, neg: str) -> None:
                fields_list = ("text", "text_g", "text_l")
                for nid_prompt, node_prompt in gdict.items():
                    try:
                        if not isinstance(node_prompt, dict):
                            continue
                        node_inputs = node_prompt.get("inputs", {})
                        if not isinstance(node_inputs, dict):
                            node_inputs = node_prompt.setdefault("inputs", {})
                        if not any(f_key in node_inputs for f_key in fields_list):
                            continue
                        hint_str = str(node_prompt.get("label") or "") + " " + str(
                            node_prompt.get("_meta", {}).get("title") or ""
                        )
                        is_neg_prompt = (
                            "neg" in hint_str.lower()
                            or "negative" in hint_str.lower()
                            or "Negative" in hint_str
                        )
                        val_to_set = neg if is_neg_prompt else pos
                        for f_key in fields_list:
                            if f_key in node_inputs:
                                node_inputs[f_key] = val_to_set
                    except (AttributeError, KeyError, TypeError) as e_apply_prompt:
                        _notify("warn", f"[IMG] 프롬프트 적용 중 경고: {e_apply_prompt}")
                        continue

            _apply_prompts(graph_clone, pos_text_raw, neg_text)

            # KSampler steps 주입
            for nid_sampler in ksampler_ids:
                _set_input(graph_clone, nid_sampler, "steps", int(steps))

            # --- SaveImage output_path 주입 ---
            relative_output_path = f"t2i_output/{sid}"
            for save_node_id in save_image_ids:
                _set_input(graph_clone, save_node_id, "output_path", relative_output_path)
                _notify("debug", f"[IMG] SaveImage 노드({save_node_id})에 output_path='{relative_output_path}' 주입")

            # --- ReActor 얼굴 스왑 설정 ---
            WORKFLOW_REACTOR_MAP = {
                "28": ("25", "male_01"),
                "23": ("24", "female_01"),
                "22": ("19", "other_char"),
            }

            scene_char_specs: Dict[str, int] = {}
            try:
                chars_list_data = sc.get("characters") or sc.get("character_objs") or []
                for item_spec in chars_list_data:
                    parsed_id = None
                    parsed_idx = None
                    if isinstance(item_spec, dict):
                        parsed_id = str(item_spec.get("id") or item_spec.get("name") or "").strip()
                        if "index" in item_spec:
                            try:
                                parsed_idx = int(item_spec.get("index"))
                            except (TypeError, ValueError):
                                parsed_idx = None
                        elif "face_index" in item_spec:
                            try:
                                parsed_idx = int(item_spec.get("face_index"))
                            except (TypeError, ValueError):
                                parsed_idx = None
                    elif isinstance(item_spec, str):
                        raw_txt_val = item_spec.strip()
                        if ":" in raw_txt_val:
                            left_txt, right_txt = raw_txt_val.split(":", 1)
                            parsed_id = left_txt.strip()
                            try:
                                parsed_idx = int(right_txt.strip())
                            except ValueError:
                                parsed_idx = None
                        else:
                            parsed_id = raw_txt_val
                    if parsed_id:
                        scene_char_specs[parsed_id] = 0 if parsed_idx is None else int(parsed_idx)
            except (AttributeError, TypeError, ValueError, IndexError) as e_parse:
                _notify("warn", f"[IMG] {sid} 캐릭터 파싱 실패: {e_parse}")

            _notify("debug", f"[IMG] {sid} 씬 캐릭터 스펙: {scene_char_specs}")

            enabled_reactors_count = 0
            for reactor_id_str, (load_id_str, default_char_id_str) in WORKFLOW_REACTOR_MAP.items():
                char_id_for_node: Optional[str] = None
                face_index_for_node: int = 0
                if default_char_id_str in scene_char_specs:
                    char_id_for_node = default_char_id_str
                    face_index_for_node = scene_char_specs[default_char_id_str]

                if char_id_for_node:
                    face_path_resolved = _resolve_face_image_by_name(char_id_for_node)
                    face_name_in_input: Optional[str] = None

                    if face_path_resolved and face_path_resolved.exists():
                        try:
                              # type: ignore
                            comfy_in_dir = _Path(COMFY_INPUT_DIR)
                            comfy_in_dir.mkdir(parents=True, exist_ok=True)
                            face_name_in_input = face_path_resolved.name
                            shutil.copy2(str(face_path_resolved), str(comfy_in_dir / face_name_in_input))
                        except (ImportError, AttributeError) as e_import_settings:
                            _notify("warn", f"[IMG] {sid} settings.COMFY_INPUT_DIR 임포트 실패: {e_import_settings}")
                            face_name_in_input = None
                        except (IOError, OSError, shutil.Error) as e_copy:
                            _notify("warn", f"[IMG] {sid} 얼굴 복사 실패 ({face_name_in_input}): {e_copy}")
                            face_name_in_input = None

                    if face_name_in_input:
                        _set_input(graph_clone, reactor_id_str, "enabled", True)
                        _set_input(graph_clone, load_id_str, "image", face_name_in_input)
                        _set_input(graph_clone, reactor_id_str, "input_faces_index", str(face_index_for_node))
                        _set_input(graph_clone, reactor_id_str, "source_faces_index", "0")
                        _notify(
                            "info",
                            f"[IMG] {sid} ReActor 활성화: Node {reactor_id_str} "
                            f"(Char: {char_id_for_node}, FaceIdx: {face_index_for_node}, Img: {face_name_in_input})",
                        )
                        enabled_reactors_count += 1
                    else:
                        _set_input(graph_clone, reactor_id_str, "enabled", False)
                        _notify(
                            "warn",
                            f"[IMG] {sid} ReActor 비활성화: Node {reactor_id_str} ({char_id_for_node} 이미지 파일 없음)",
                        )
                else:
                    _set_input(graph_clone, reactor_id_str, "enabled", False)

            if enabled_reactors_count == 0:
                _notify("info", f"[IMG] {sid} 씬에 매칭되는 캐릭터 없음. 페이스 스왑 비활성화.")

            # --- ComfyUI 제출 ---
            req_count += 1
            _notify("submit", f"[IMG] /prompt {sid}")
            result_data = _wait_img(
                base_url,
                graph_clone,
                timeout=timeout_sec,
                poll=poll_sec,
                progress_cb=lambda d: _notify("wait", str(d.get("msg") or "")),
            )

            # --- 결과 파싱 + 디버그 ---
            files_list: List[Dict[str, Any]] = []

            outputs_dict = result_data.get("outputs") or {}
            try:
                out_keys = list(outputs_dict.keys())
                _notify("debug", f"[IMG] {sid} outputs keys={out_keys}")
            except (TypeError, AttributeError):
                pass

            for out_k, out_val in outputs_dict.items():
                if isinstance(out_val, dict):
                    imgs_in_output = out_val.get("images", [])
                    if isinstance(imgs_in_output, list):
                        files_list.extend(imgs_in_output)

            if not files_list:
                _notify("debug", f"[IMG] {sid} result images empty, raw outputs={outputs_dict}")
                raise RuntimeError(f"{sid}: 출력 이미지 없음")

            # type="output" 우선 선택
            img_output_list = [file_info for file_info in files_list if str(file_info.get("type") or "") == "output"]
            if img_output_list:
                last_file_info = img_output_list[-1]
            else:
                last_file_info = files_list[-1]

            subfolder_str = str(last_file_info.get("subfolder") or "")
            fname_str = str(last_file_info.get("filename") or "")
            img_type_str = str(last_file_info.get("type") or "output")

            _notify(
                "debug",
                f"[IMG] {sid} pick filename={fname_str}, subfolder={subfolder_str}, type={img_type_str}",
            )

            if not fname_str:
                raise RuntimeError(f"{sid}: 출력 파일명 없음")

            params = {"filename": fname_str, "subfolder": subfolder_str}
            if img_type_str:
                params["type"] = img_type_str

            # --- 이미지 다운로드 및 저장 ---
            try:
                r_image = requests.get(
                    base_url.rstrip("/") + "/view",
                    params=params,
                    timeout=30,
                )
                r_image.raise_for_status()
            except requests.RequestException as e_requests:
                _notify("debug", f"[IMG] {sid} /view 실패 params={params}")
                raise RuntimeError(f"{sid}: /view 실패: {e_requests}") from e_requests

            img_path_target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(img_path_target, "wb") as fpw:
                    fpw.write(r_image.content)
            except (IOError, OSError) as e_write_img:
                raise IOError(f"{sid}: 이미지 파일 저장 실패: {e_write_img}") from e_write_img

            # --- video.json / story.json 갱신 ---
            for scene_item in scenes_list:
                if str(scene_item.get("id")) == sid:
                    scene_item["img_file"] = str(img_path_target)
                    break
            try:
                save_json(p_video, video_doc)
                _notify("save", f"[IMG] video.json 갱신: {sid}")
            except (OSError, ValueError, TypeError) as e_save_video:
                _notify("warn", f"[IMG] {sid} video.json 저장 실패: {e_save_video}")

            if story_doc:
                try:
                    scs_list = story_doc.get("scenes") or []
                    for scene_item_story in scs_list:
                        if str(scene_item_story.get("id")) == sid:
                            scene_item_story["img_file"] = str(img_path_target)
                            break
                    save_json(p_story_json, story_doc)
                except (OSError, ValueError, TypeError) as e_save_story:
                    _notify("warn", f"[IMG] {sid} story.json 저장 실패: {e_save_story}")

            created.append(img_path_target)
            _notify("done", f"[IMG] 저장 {img_path_target.name}")

        except (OSError, ValueError, TypeError, KeyError, RuntimeError, requests.RequestException) as err_scene:
            _notify("scene-error", f"[IMG] {sc.get('id')}: {type(err_scene).__name__}: {err_scene}")
            continue
        except Exception as e_scene_unknown:
            _notify(
                "scene-error",
                f"[IMG] {sc.get('id')} 예상치 못한 오류: {type(e_scene_unknown).__name__}: {e_scene_unknown}",
            )
            continue

    _notify("summary", f"[IMG] 생성={len(created)} / 요청={req_count}")
    return created
















# 파일 상단 import 근처에 추가

def _set_face_on_reactors(
    graph: dict,
    face_filename_in_input: str,
    *,
    target_label_hint: str | None = None
) -> list[str]:
    """
    ReActor 노드(22/23/28)가 참조하는 LoadImage(19/24/25)의 inputs['image']를 face_filename_in_input으로 교체.
    그리고 '사용할' ReActor만 enabled=True, 나머지는 False로 둔다.
    target_label_hint에 "female"/"male" 등 힌트를 주면 그에 맞는 리액터를 우선 활성화.
    반환: 실제로 파일명을 바꾼 LoadImage 노드 ID 목록
    """
    applied_load_ids: list[str] = []

    # 1) ReActor -> source LoadImage 매핑 수집
    reactors_list: list[tuple[str, str, str]] = []
    for reactor_node_id, node in graph.items():
        if node.get("class_type") == "ReActorFaceSwap":
            ins = node.get("inputs", {}) or {}
            src = ins.get("source_image")
            if isinstance(src, list) and len(src) == 2:
                load_node_id = str(src[0])
                node_title = str(node.get("_meta", {}).get("title", ""))
                reactors_list.append((str(reactor_node_id), load_node_id, node_title))

    # 2) 우선순위 결정: 라벨 힌트가 있으면 그 라벨이 붙은 ReActor를 우선 활성화
    def choose_active(rlist: list[tuple[str, str, str]]) -> str | None:
        if not rlist:
            return None
        if target_label_hint:
            tl = target_label_hint.lower()
            for cand_id, _cand_load_id, cand_title in rlist:
                if tl in str(cand_title).lower():
                    return cand_id
        # 힌트 없으면 "마지막(가장 downstream)"을 활성화
        return rlist[-1][0]

    active_reactor = choose_active(reactors_list)

    # 3) LoadImage 파일명 교체 & enable 토글
    for reactor_id, load_node_id, _node_title in reactors_list:
        # LoadImage 파일명 교체
        if load_node_id in graph and "LoadImage" in str(graph[load_node_id].get("class_type")):
            graph[load_node_id].setdefault("inputs", {})["image"] = face_filename_in_input
            applied_load_ids.append(load_node_id)
        # enable/disable
        graph[reactor_id].setdefault("inputs", {})["enabled"] = (reactor_id == active_reactor)

    return applied_load_ids






def _resolve_character_image_path(scene: dict, story: dict) -> str | None:
    # characters: ["female_01:0"] 또는 character_objs: [{"id":"female_01","index":0}]
    specs = scene.get("characters") or scene.get("character_objs") or []
    if not specs:
        return None

    def _parse(spec):
        if isinstance(spec, str):
            parts = spec.split(":")
            return parts[0].strip(), int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        if isinstance(spec, dict):
            return spec.get("id"), int(spec.get("index", 0))
        return None, 0

    cid, _ = _parse(specs[0])
    if not cid:
        return None

    root = Path((story.get("paths") or {}).get("root") or ".")
    p1 = root / "character" / f"{cid}.png"
    if p1.exists():
        return str(p1)

    p2 = Path(CHARACTER_DIR) / f"{cid}.png"
    if p2.exists():
        return str(p2)

    # 확장자 유연 탐색
    for cand in (root / "character").glob(f"{cid}.*"):
        return str(cand)
    for cand in Path(CHARACTER_DIR).glob(f"{cid}.*"):
        return str(cand)
    return None

def _copy_to_comfy_input(src_path: str) -> str:
    """COMFY_INPUT_DIR로 복사하고 파일명만 반환(LoadImage는 input 기준으로 파일명만 필요)."""
    src = Path(src_path)
    dst = Path(COMFY_INPUT_DIR) / src.name
    if str(src.resolve()) != str(dst.resolve()):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return dst.name  # 파일명만 반환




def _inject_face_image_to_graph(graph: dict, face_filename_in_input: str) -> list[str]:
    """
    워크플로 내에서 ReActor의 source_image로 연결된 LoadImage 노드를 찾아
    그 노드의 파일명(inputs['image'])만 face_filename_in_input으로 교체.
    반환: 적용된 LoadImage 노드ID 목록
    """
    applied = []
    # 1) ReActor 노드 수집
    reactor_ids = [nid for nid, node in graph.items() if "ReActor" in str(node.get("label") or "")]
    if not reactor_ids:
        return applied

    # 2) LoadImage 노드들 중 라벨에 'swap', 'face', 'source' 포함 → 우선 타깃팅
    load_ids = []
    for nid, node in graph.items():
        ct = str(node.get("class_type") or "").lower()
        lbl = str(node.get("label") or "").lower()
        if "loadimage" in ct or ct == "load image":
            if any(key in lbl for key in ("swap", "face", "source")):
                load_ids.append(nid)
    # 없다면 모든 LoadImage 후보
    if not load_ids:
        load_ids = [nid for nid, node in graph.items()
                    if "loadimage" in str(node.get("class_type") or "").lower()
                    or str(node.get("class_type") or "").lower() == "load image"]

    # 3) 파일명 주입
    for nid in load_ids:
        ins = graph[nid].setdefault("inputs", {})
        if "image" in ins:
            ins["image"] = face_filename_in_input
            applied.append(nid)

    # 4) ReActor enable 토글
    for nid in reactor_ids:
        graph[nid].setdefault("inputs", {})["enabled"] = bool(applied)

    return applied







# app/image_movie_docs.py


S = settings_obj  # noqa: N816  # (하위호환: 기존 코드가 S를 참조해도 동작)




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

    - 타겟 해상도/프레임레이트/스텝은 video.json(story.json)의 값을 우선 사용:
        * width  = story.defaults.image.width   (없으면 1080)
        * height = story.defaults.image.height  (없으면 1920)
        * fps    = story.fps 또는 story.defaults.movie.target_fps (없으면 60)
        * steps  = story.defaults.movie.steps (없으면 24)
        * overlap_frames = story.defaults.movie.overlap_frames (없으면 12)
    - i2v 프레임 계획:
        * frames = round(duration * fps)
        * 두 번째 아이템부터 head_handle_frames = overlap_frames (첫 아이템은 0)
    - 기존 필드/구조 보존: items[].duration / source_image / i2v_prompt 등 기존 흐름은 그대로.
    """

    proj = Path(project_dir)
    story: Dict[str, Any] = _load_json(proj / "story.json", {}) or {}

    # hair_map이 전달되더라도 현재 단계에서는 결과를 바꾸지 않음(경고 방지용으로 접근만 수행)
    if isinstance(hair_map, dict):
        pass

    # ---- 안전한 기본값 추출 (video.json 기반) ----
    defaults_dict: Dict[str, Any] = story.get("defaults") or {}
    image_defaults: Dict[str, Any] = defaults_dict.get("image") or {}
    movie_defaults: Dict[str, Any] = defaults_dict.get("movie") or {}

    target_width: int = int(image_defaults.get("width") or 1080)
    target_height: int = int(image_defaults.get("height") or 1920)

    # fps 우선순위: story.fps → defaults.movie.target_fps → 60
    fps_val: int = int(story.get("fps") or movie_defaults.get("target_fps") or 60)

    # steps/overlap_frames 기본
    steps_val: int = int(movie_defaults.get("steps") or 24)
    overlap_frames_val: int = int(movie_defaults.get("overlap_frames") or 5)

    # ---- 씬 목록 ----
    scenes: List[Dict[str, Any]] = list(story.get("scenes") or [])

    # ---- 출력 스켈레톤 (기존 구조 보존) ----
    out: Dict[str, Any] = {
        "title": story.get("title", ""),
        "audio": story.get("audio", ""),
        "fps": fps_val,  # 상단에도 기록(기존 키 유지)
        "defaults": {
            "movie": {
                # i2v/렌더 타겟 설정(후속 단계에서 그대로 사용)
                "target_width": target_width,
                "target_height": target_height,
                "target_fps": fps_val,
                "steps": steps_val,
                "overlap_frames": overlap_frames_val,
                # 네거티브 프롬프트: story.defaults.image.negative 우선
                "negative": (image_defaults.get("negative") or "")
            }
        },
        "items": []
    }

    # ---- 아이템 구성 (기존 프롬프트 규칙 보존) ----
    for idx, sc in enumerate(scenes, 1):
        scene_id = sc.get("id") or f"t_{idx:02d}"
        duration_sec = float(sc.get("duration") or 0.75)

        # i2v 프롬프트 우선순위: prompt_movie → (prompt + prompt_img)
        prompt_movie = (sc.get("prompt_movie") or "").strip()
        if prompt_movie:
            i2v_prompt = prompt_movie
        else:
            prompt_base = (sc.get("prompt") or "").strip()
            prompt_img = (sc.get("prompt_img") or "").strip()
            if prompt_img:
                i2v_prompt = f"{prompt_base}\n{prompt_img}" if prompt_base else prompt_img
            else:
                i2v_prompt = prompt_base

        # 프레임 산출 (60fps 기본, story 기반으로 이미 fps_val 계산됨)
        base_frames = int(round(duration_sec * fps_val))
        head_handle_frames = overlap_frames_val if idx >= 2 else 0

        out["items"].append({
            "id": scene_id,
            "duration": duration_sec,                 # 기존 보존
            "source_image": sc.get("img_file"),       # 기존 보존 (이미지 생성 단계에서 채워진 값)
            "i2v_prompt": i2v_prompt,                 # 기존 보존 (prompt_movie 우선)
            # ↓↓↓ 추가 메타(후속 i2v 렌더에서 사용)
            "frames": base_frames,
            "head_handle_frames": head_handle_frames
        })

    # 저장 (기존 방식)
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


    # ---- utils ----

    # ---- UI 프리셋 로더(있으면 사용) + 기본값 ----
    def _ui_defaults_local() -> dict:
        return {
            "image_size": (832, 1472),
            "movie_fps": 24,
            "movie_overlap": 12,
            "min_scene_sec": 0.5,
            "round_sec": 3,
            "negative_bank": "손가락 왜곡, 눈 왜곡, 과도한 보정, 노이즈, 흐릿함, 텍스트 워터마크",
        }

    try:
        prefs_loader_func = globals().get("_load_ui_prefs_from_project")
        if prefs_loader_func is None:
            raise RuntimeError
        ui_prefs_map = prefs_loader_func(story.get("audio") or "")
        img_w_val, img_h_val = ui_prefs_map.get("image_size", (832, 1472))
        movie_fps_val = int(ui_prefs_map.get("movie_fps", 24))
        movie_overlap_val = int(ui_prefs_map.get("movie_overlap", 12))
        min_scene_sec_val = float(ui_prefs_map.get("min_scene_sec", 0.5))
        round_sec_val = int(ui_prefs_map.get("round_sec", 3))
        negative_bank_val = ui_prefs_map.get("negative_bank") or _ui_defaults_local()["negative_bank"]
    except Exception:
        ui_prefs_map = _ui_defaults_local()
        img_w_val, img_h_val = ui_prefs_map["image_size"]
        movie_fps_val = int(ui_prefs_map["movie_fps"])
        movie_overlap_val = int(ui_prefs_map["movie_overlap"])
        min_scene_sec_val = float(ui_prefs_map["min_scene_sec"])
        round_sec_val = int(ui_prefs_map["round_sec"])
        negative_bank_val = ui_prefs_map["negative_bank"]

    def _round_sec_local(v_in: float) -> float:
        return round(float(v_in), round_sec_val)

    # ---- 원본 ----
    src_story = dict(story or {})
    audio_path_str = src_story.get("audio") or ""
    title_str = src_story.get("title") or "무제"
    total_duration_sec = float(src_story.get("duration") or 0.0)
    fps_src_val = int(src_story.get("fps") or 0)
    lang_str = src_story.get("lang") or "ko"
    root_dir_path = Path(audio_path_str).parent if audio_path_str else Path(".")

    # ---- project.json에서 가사/총길이 ----
    pj_path = root_dir_path / "project.json"
    pj_meta = load_json(pj_path, {}) if pj_path.exists() else {}
    lyrics_text_raw = (pj_meta.get("lyrics") or "").strip()
    proj_time_sec = float(pj_meta.get("time") or 0.0)

    # ---- 전체 가사 정제(순수 가사) ----
    def _clean_full_lyrics_local(raw_text: str) -> str:
        t_text = (raw_text or "").replace("\r\n", "\n").replace("\r", "\n")
        t_text = re.sub(r"\[[^]]+]", " ", t_text)
        t_text = t_text.replace("\n", " ")
        t_text = re.sub(r"\s+", " ", t_text)
        return t_text.strip()

    lyrics_text_clean = _clean_full_lyrics_local(lyrics_text_raw)

    # ───────────── 프롬프트 강제 규칙 ─────────────
    def _merge_global_bg_defaults_local(base_bg_map: dict) -> dict:
        merged_map = dict(base_bg_map)
        try:
            global_bg_defaults = BG_DEFAULTS  # type: ignore[name-defined]
            if isinstance(global_bg_defaults, dict):
                for key_bg, val_bg in global_bg_defaults.items():
                    if isinstance(key_bg, str) and isinstance(val_bg, str):
                        merged_map[key_bg.lower()] = val_bg
        except NameError:
            try:
                global_bg_defaults2 = bg_defaults  # type: ignore[name-defined]
                if isinstance(global_bg_defaults2, dict):
                    for key_bg2, val_bg2 in global_bg_defaults2.items():
                        if isinstance(key_bg2, str) and isinstance(val_bg2, str):
                            merged_map[key_bg2.lower()] = val_bg2
            except Exception:
                pass
        return merged_map

    base_bg_presets = {
        "intro": "황혼이 내려앉은 골목",
        "verse": "도시 야간 거리",
        "chorus": "네온이 번지는 광장",
        "bridge": "지하철 플랫폼",
        "outro": "비 내린 새벽 도로",
    }
    merged_bg_presets = _merge_global_bg_defaults_local(base_bg_presets)

    def _ensure_background_local(prompt_text: str, sec_name: str) -> str:
        txt_prompt = (prompt_text or "").strip()
        if "배경:" in txt_prompt:
            return txt_prompt
        sec_key = (sec_name or "").lower().strip()
        bg_name_val = merged_bg_presets.get(sec_key, "도시 야간 거리")
        return f"배경: {bg_name_val}. {txt_prompt}".strip()

    def _ensure_effects_in_movie_local(movie_text: str, effects_list_in: list[str]) -> str:
        txt_movie = (movie_text or "").strip()
        eff_items = [e_item.strip() for e_item in (effects_list_in or []) if e_item and e_item.strip()]
        if eff_items:
            if not txt_movie.endswith("."):
                txt_movie += "."
            txt_movie += " " + ", ".join(eff_items)
        return txt_movie.strip()

    def _ensure_motion_if_characters_local(movie_text: str, has_characters_flag: bool) -> str:
        txt_movie2 = (movie_text or "").strip()
        if has_characters_flag and ("인물 동작:" not in txt_movie2):
            if not txt_movie2.endswith("."):
                txt_movie2 += "."
            txt_movie2 += " 인물 동작: 천천히 시선을 돌린다."
        return txt_movie2.strip()

    # ───────────── 타이밍 재배치 ─────────────
    scenes_input: List[Dict[str, Any]] = list(src_story.get("scenes") or [])
    span_list: List[Tuple[float, float]] = []
    for sc_item in scenes_input:
        span_start = float(sc_item.get("start") or 0.0)
        span_end = float(sc_item.get("end") or (span_start + float(sc_item.get("duration") or 0.0)))
        if span_end <= span_start:
            span_end = span_start + max(0.0, float(sc_item.get("duration") or 0.0))
        span_list.append((span_start, span_end))

    def _choose_total_local(orig_total_sec: float, pj_total_sec: float) -> float:
        return pj_total_sec if pj_total_sec >= 10.0 else orig_total_sec

    def _retime_local(total_seconds_in: float, spans_in_list: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if total_seconds_in <= 0 or not spans_in_list:
            return spans_in_list
        durations = [max(0.0, ed_i - st_i) for st_i, ed_i in spans_in_list]
        sum_durations = sum(durations)
        if sum_durations <= 0:
            each_len = total_seconds_in / len(spans_in_list)
            t_cursor = 0.0
            out_spans_list: List[Tuple[float, float]] = []
            for _ in spans_in_list:
                out_spans_list.append((_round_sec_local(t_cursor), _round_sec_local(min(total_seconds_in, t_cursor + each_len))))
                t_cursor += each_len
            return out_spans_list
        alloc_list = [(d_val / sum_durations) * total_seconds_in for d_val in durations]
        alloc_list = [max(a_val, min_scene_sec_val) for a_val in alloc_list]
        sum_alloc = sum(alloc_list)
        if sum_alloc > total_seconds_in:
            scale_factor = total_seconds_in / sum_alloc
            alloc_list = [a_val * scale_factor for a_val in alloc_list]
        out_spans2: List[Tuple[float, float]] = []
        t_cur2 = 0.0
        for alloc_len in alloc_list:
            s_val2, e_val2 = t_cur2, min(total_seconds_in, t_cur2 + alloc_len)
            out_spans2.append((_round_sec_local(s_val2), _round_sec_local(e_val2)))
            t_cur2 = e_val2
        if out_spans2:
            last_start, _ = out_spans2[-1]
            out_spans2[-1] = (last_start, _round_sec_local(total_seconds_in))
        return out_spans2

    max_end_val = max((end_v for (_, end_v) in span_list), default=0.0)
    if total_duration_sec <= 3.0 or max_end_val <= 3.0:
        desired_total = _choose_total_local(total_duration_sec, proj_time_sec)
        if desired_total > 5.0 and max_end_val > 0.0:
            scale_factor2 = desired_total / max_end_val
            span_list = [(st_v * scale_factor2, ed_v * scale_factor2) for (st_v, ed_v) in span_list]
            total_duration_sec = float(desired_total)

    span_list = _retime_local(total_duration_sec, span_list)

    # ───────────── characters: "id:index" + 객체형 ─────────────
    def _coerce_char_with_index_local(char_list_in: List[Any]) -> tuple[list[str], list[dict]]:
        parsed_pairs: list[tuple[str, int]] = []
        raw_ids_accum: list[str] = []
        for item_in in (char_list_in or []):
            s_item_in = str(item_in or "").strip()
            if ":" in s_item_in:
                cid_str, idx_str = s_item_in.split(":", 1)
                try:
                    parsed_pairs.append((cid_str.strip(), int(idx_str.strip())))
                except Exception:
                    raw_ids_accum.append(cid_str.strip())
            elif s_item_in:
                raw_ids_accum.append(s_item_in)

        if parsed_pairs and not raw_ids_accum:
            str_list_out = [f"{cid}:{idx}" for cid, idx in parsed_pairs]
            obj_list_out = [{"id": cid, "index": int(idx)} for cid, idx in parsed_pairs]
            return str_list_out, obj_list_out

        ids_seq = list(raw_ids_accum)
        if len(ids_seq) == 2 and set(ids_seq) == {"female_01", "male_01"}:
            auto_pairs = [("male_01", 0), ("female_01", 1)]
        else:
            auto_pairs = [(cid_auto, i_auto) for i_auto, cid_auto in enumerate(ids_seq)]

        str_list_out2 = [f"{cid}:{idx}" for cid, idx in auto_pairs]
        obj_list_out2 = [{"id": cid, "index": int(idx)} for cid, idx in auto_pairs]
        return str_list_out2, obj_list_out2

    # ───────────── 씬 재구성 ─────────────
    scenes_out_list: List[Dict[str, Any]] = []
    for scene_idx, sc_in in enumerate(scenes_input):
        sid_val = sc_in.get("id") or f"t_{scene_idx+1:03d}"
        section_name_val = str((sc_in.get("section") or sc_in.get("scene") or "scene")).lower().strip()
        if scene_idx < len(span_list):
            start_val, end_val = span_list[scene_idx]
        else:
            start_val = float(sc_in.get("start") or 0.0)
            end_val = float(sc_in.get("end") or 0.0)
        duration_val = max(0.0, end_val - start_val)

        base_prompt_text = (sc_in.get("prompt") or "").strip()
        prompt_img_in_text = (sc_in.get("prompt_img") or "").strip()
        prompt_movie_in_text = (sc_in.get("prompt_movie") or "").strip()
        effects_list_val = list(sc_in.get("effect") or [])

        char_strs, char_objs = _coerce_char_with_index_local(list(sc_in.get("characters") or []))
        has_chars_flag = bool(char_objs)

        effective_prompt_img = prompt_img_in_text or base_prompt_text
        effective_prompt_movie = prompt_movie_in_text or (effective_prompt_img or base_prompt_text)
        effective_prompt_img = _ensure_background_local(effective_prompt_img, section_name_val)
        effective_prompt_movie = _ensure_background_local(effective_prompt_movie, section_name_val)
        effective_prompt_movie = _ensure_effects_in_movie_local(effective_prompt_movie, effects_list_val)
        effective_prompt_movie = _ensure_motion_if_characters_local(effective_prompt_movie, has_chars_flag)

        prompt_negative_val = (sc_in.get("prompt_negative") or "@global")

        sc_out = {
            "id": sid_val,
            "section": section_name_val,
            "start": _round_sec_local(start_val),
            "end": _round_sec_local(end_val),
            "duration": _round_sec_local(duration_val),
            "scene": sc_in.get("scene") or section_name_val,
            "characters": char_strs,
            "character_objs": char_objs,
            "effect": effects_list_val,
            "screen_transition": bool(sc_in.get("screen_transition")),
            "img_file": str(root_dir_path / "imgs" / f"{sid_val}.png"),
            "clip_file": str(root_dir_path / "clips" / f"{sid_val}.mp4"),
            "prompt_img": effective_prompt_img,
            "prompt_movie": effective_prompt_movie,
            "prompt_negative": prompt_negative_val,
        }
        sc_out.setdefault("prompt", sc_out["prompt_img"])
        scenes_out_list.append(sc_out)

    # ───────────── 가사 섹션 생성 ─────────────
    lyrics_sections_list: List[Dict[str, Any]] = []
    if lyrics_text_raw:
        try:
            builder_func = globals().get("_build_lyrics_sections")
            if builder_func:
                lyrics_sections_list = builder_func(lyrics_text_raw, total_duration_sec, scenes_out_list, round_sec=round_sec_val) or []
        except Exception:
            lyrics_sections_list = []

    # ───────────── 가사 씬별 분배 ─────────────
    def _split_block_lines_local(raw_block_text: str) -> List[str]:
        t_block = (raw_block_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        line_list = [ln_i.strip() for ln_i in t_block.split("\n")]
        if line_list and re.fullmatch(r"\[[^]]+]", line_list[0]):
            line_list = line_list[1:]
        return [ln_j for ln_j in line_list if ln_j]

    def _scene_in_block_local(scene_start_val: float, block_start_val: float, block_end_val: float) -> bool:
        return (block_start_val <= scene_start_val) and (scene_start_val < block_end_val)

    section_to_id: Dict[str, str] = {}
    for sec_idx_num, ls_item in enumerate(lyrics_sections_list, start=1):
        sec_name_norm = str(ls_item.get("section") or "").lower().strip()
        if sec_name_norm and sec_name_norm not in section_to_id:
            section_to_id[sec_name_norm] = f"L{sec_idx_num:02d}"

    for sc_out_item in scenes_out_list:
        sec_id_val = section_to_id.get(sc_out_item["section"])
        if sec_id_val:
            sc_out_item["section_id"] = sec_id_val

    for block_item in lyrics_sections_list:
        try:
            block_start = float(block_item.get("start", 0.0) or 0.0)
            block_end = float(block_item.get("end", 0.0) or 0.0)
        except Exception:
            continue
        line_chunks = _split_block_lines_local(block_item.get("text") or "")
        if not line_chunks:
            line_chunks = [(block_item.get("text") or "").strip()] if block_item.get("text") else []

        block_scene_indices = [
            idx_scene for idx_scene, scene_obj in enumerate(scenes_out_list)
            if _scene_in_block_local(float(scene_obj.get("start") or 0.0), block_start, block_end)
        ]
        if not block_scene_indices:
            continue

        num_scenes_in_blk = len(block_scene_indices)
        num_lines_in_blk = len(line_chunks)
        if num_lines_in_blk <= 0:
            for scene_index_empty in block_scene_indices:
                scenes_out_list[scene_index_empty]["lyric"] = ""
            continue

        chunk_size = int(math.ceil(num_lines_in_blk / num_scenes_in_blk))
        for chunk_idx, scene_index in enumerate(block_scene_indices):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, num_lines_in_blk)
            piece_text = "\n".join(line_chunks[start_idx:end_idx]).strip()
            scenes_out_list[scene_index]["lyric"] = piece_text

    for scene_obj2 in scenes_out_list:
        if "lyric" not in scene_obj2:
            scene_obj2["lyric"] = ""

    # ───────────── 글로벌 컨텍스트 보강 ─────────────
    global_ctx_map = dict(src_story.get("global_context") or {})
    if not global_ctx_map.get("negative_bank"):
        global_ctx_map["negative_bank"] = negative_bank_val

    # ───────────── paths ─────────────
    paths_obj_out = {
        "root": (str(root_dir_path) if str(root_dir_path).endswith(("\\", "/")) else str(root_dir_path) + "\\"),
        "imgs_dir": "imgs",
        "clips_dir": "clips",
        "img_name_pattern": "{id}.png",
        "clip_name_pattern": "{id}.mp4",
    }

    out_story = {
        "version": "1.1",
        "audio": audio_path_str,
        "fps": int(fps_src_val or movie_fps_val),
        "duration": _round_sec_local(total_duration_sec),
        "offset": float(src_story.get("offset") or 0.0),
        "title": title_str,
        "lyrics": lyrics_text_clean,
        "lang": lang_str or "ko",
        "paths": paths_obj_out,
        "characters": list(src_story.get("characters") or []),
        "character_styles": dict(src_story.get("character_styles") or {}),
        "global_context": global_ctx_map,
        "defaults": {
            "image": {"width": int(img_w_val), "height": int(img_h_val), "negative": "@global"},
            "movie": {"target_fps": int(movie_fps_val), "overlap_frames": int(movie_overlap_val), "negative": "@global"},
        },
        "lyrics_sections": lyrics_sections_list,
        "scenes": scenes_out_list,
        "audit": {"generated_by": "gpt-5 (no-gpt-prompts, rules only, per-scene-lyrics, clean-lyrics)"},
    }

    if "timeline" in src_story:
        out_story["timeline"] = src_story["timeline"]

    return out_story












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

    pj = Path(audio_path).parent / "project.json"
    meta = load_json(pj, {}) if pj.exists() else {}
    ui = meta.get("ui_prefs") or {}

    img_size = ui.get("image_size") or getattr(settings_obj, "DEFAULT_IMG_SIZE", (832, 1472))
    w, h = tuple(img_size)

    fps = int(ui.get("movie_fps") or getattr(settings_obj, "DEFAULT_MOVIE_FPS", 24))
    overlap = int(ui.get("movie_overlap") or getattr(settings_obj, "DEFAULT_MOVIE_OVERLAP", 12))

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

    def _ssssss(x): return x.strip() if isinstance(x, str) else ""

    for idx, sc in enumerate(story.get("scenes") or []):
        sid = sc.get("id") or f"t_{idx+1:03d}"
        section = (sc.get("section") or "").lower().strip()
        eff_list = sc.get("effect") or []
        chars = sc.get("characters") or []
        lyric_hint = _ssssss(sc.get("lyric_text") or "")

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

        bg   = _ssssss(res.get("background"))
        lit  = _ssssss(res.get("lighting"))
        cam  = _ssssss(res.get("camera"))
        imgp = _ssssss(res.get("img_phrase") or res.get("prompt_img"))
        movp = _ssssss(res.get("movie_phrase") or res.get("prompt_movie"))
        fx_x = _ssssss(res.get("effects_extra"))
        mot  = _ssssss(res.get("character_motion"))

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


def _parse_character_spec_img(item: Any) -> Dict[str, Any]:
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
      - scene["layout"]["face_indices"] (있다면)로 index를 보완
      - scene["face_indices"] (flat dict)도 함께 만들어 소비 측이 편하게 사용
    반환: 수정된 scene (원본 shallow copy 후 필드 갱신)
    """
    sc = dict(scene or {})
    chars_in = sc.get("characters") or []
    norm: List[Dict[str, Any]] = [_parse_character_spec_img(x) for x in chars_in]

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
    motion_pat = re.compile(r"(?:^|\s)인물\s*동작\s*:\s*([^.]+)\.", re.IGNORECASE)

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




# video_build.py



# video_build.py의 fill_prompt_movie_with_ai 함수 전체를 이 코드로 교체하세요.

def fill_prompt_movie_with_ai(
        project_dir: "Path",
        ask: "Callable[[str, str], str]",
        *,
        log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """
    [수정됨]
    AI의 역할을 '뮤직비디오 감독'으로 명시하고, '가사'를 기반으로
    '캐릭터의 새로운 행동/포즈/표정' 및 '카메라 워크'가 포함된 '장면 묘사'를 생성하도록 수정.
    """
    # 안전 로드/세이브



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

    # --- ▼▼▼ [신규] 1. 원본 분위기 (project.json) 로드 ▼▼▼ ---
    pj_path = pdir / "project.json"
    original_vibe_prompt = ""
    if pj_path.exists():
        pj_doc = load_json(pj_path, {}) or {}
        if isinstance(pj_doc, dict):
            original_vibe_prompt = pj_doc.get("prompt_user") or pj_doc.get("prompt", "")
    # --- ▲▲▲ [신규] 끝 ▲▲▲ ---

    # ── 2) FPS 확정 (기존과 동일) ─────────────────────────────
    defaults_map: Dict[str, Any] = vdoc.get("defaults") or {}
    movie_def: Dict[str, Any] = defaults_map.get("movie") or {}
    image_def: Dict[str, Any] = defaults_map.get("image") or {}

    fps_candidates = [movie_def.get("target_fps"), vdoc.get("fps"), image_def.get("fps"), 30, ]
    fps = 30
    for cand in fps_candidates:
        if cand is not None:
            try:
                fps = int(cand); break
            except (TypeError, ValueError):
                continue

    # (FPS 동기화 로직 - 기존과 동일)
    vdoc.setdefault("fps", fps)
    vdoc.setdefault("defaults", {})
    vdoc["defaults"].setdefault("movie", {})["target_fps"] = fps
    vdoc["defaults"]["movie"]["input_fps"] = fps
    vdoc["defaults"]["movie"]["fps"] = fps
    vdoc["defaults"].setdefault("image", {})["fps"] = fps

    # (Chunk/Overlap 값 읽기 - 기존과 동일)
    try:
        base_chunk_val = int(movie_def.get("base_chunk", 41))
        overlap_val = int(movie_def.get("overlap", 0))
    except Exception:
        base_chunk_val, overlap_val = 41, 0

    # ── 3) 씬 루프 (구조 변경) ──────────────────────────────────
    scenes = vdoc.get("scenes") or []
    if not isinstance(scenes, list):
        _log("[fill_prompt_movie_with_ai] scenes 없음")
        save_json(vpath, vdoc)
        return

    changed = False

    # --- ▼▼▼ [수정된 AI 시스템 프롬프트 (뮤직비디오 감독)] ▼▼▼ ---
    # --- ▼▼▼ [수정된 AI 시스템 프롬프트 (생동감/동작 강화 버전)] ▼▼▼ ---
    system_msg = (
        "You are an expert Music Video Director specializing in AI video generation.\n"
        "Your goal is to break the static nature of Image-to-Image generation by writing **action-heavy prompts**.\n"
        "The generated video segments must show **clear physical movement**, not just camera movement.\n\n"
        "[Context Provided]\n"
        "1. `original_vibe`: The overall theme (e.g., 'sadness', 'joy').\n"
        "2. `scene_lyric`: The lyric for THIS scene (Korean).\n"
        "3. `base_visual`: The character and background setting.\n"
        "4. `time_structure`: The segments for THIS scene (e.g., [\"0-82f\", \"82-164f\"]).\n\n"
        "Your task is to return a JSON object ONLY: {\"segment_prompts\": [\"prompt 1\", \"prompt 2\", ...]}.\n"
        "The array length MUST exactly match the `time_structure` list length.\n\n"
        "[!! CRITICAL RULES !!]\n"
        "1. **LANGUAGE: MUST BE ENGLISH.** Translate any Korean input into English descriptions. (NO Korean output allowed).\n"
        "2. **FORBIDDEN:** Do NOT write 'static', 'standing still', or just 'cinematic lighting'.\n"
        "3. **MANDATORY ACTION:** Every prompt MUST contain a strong verb describing body movement.\n"
        "    - BAD: 'A woman standing in the rain.'\n"
        "    - GOOD: 'She **tilts her head back** to feel the rain, **raising her hand** slowly.'\n"
        "    - GOOD: 'She **turns around** quickly, hair whipping in the wind, expression changing to shock.'\n"
        "4. **FOCUS ON LIMBS & HEAD:** Explicitly describe what the hands, head, or eyes are doing (e.g., 'wiping tears', 'reaching out', 'looking over shoulder').\n"
        "5. **PROGRESSION:** Segment 1 should start an action, and Segment 2 must continue or finish it. Make a mini-story.\n"
        "6. **CAMERA:** Use dynamic angles (low angle, dutch angle) ONLY combined with character movement."
    )
    # --- ▲▲▲ [수정 끝] ▲▲▲ ---
    # --- ▲▲▲ [수정 끝] ▲▲▲ ---

    for i, sc in enumerate(scenes):
        if not isinstance(sc, dict):
            continue

        scene_id = sc.get("id", "unknown")

        # 1. 총 프레임 계산 (기존과 동일)
        try:
            dur = float(sc.get("duration") or 0.0)
        except (TypeError, ValueError):
            dur = 0.0
        total_frames = int(round(dur * fps)) if dur > 0 else 0
        if total_frames <= 0: continue

        # 2. frame_segments 생성 (기존과 동일)
        segs = sc.get("frame_segments")
        if not isinstance(segs, list) or not segs:
            pairs_tuples =  plan_segments_s_e(total_frames, base_chunk=base_chunk_val)
            segs_out: List[Dict[str, Any]] = []
            for s_f, e_f in pairs_tuples:
                segs_out.append({"start_frame": int(s_f), "end_frame": int(e_f), "prompt_movie": ""})
            sc["frame_segments"] = segs_out
            segs = segs_out
            changed = True  # 세그먼트가 생성되었으므로 'changed'

        # 3. 비어있는 프롬프트가 있는지 확인 (기존과 동일)
        prompts_list = [seg.get("prompt_movie", "") for seg in segs]
        if all(prompts_list):
            _log(f"[{scene_id}] 모든 세그먼트 프롬프트(묘사)가 이미 존재합니다. (AI 호출 스킵)")
            continue

        # 4. AI 호출을 위한 데이터 수집 (기존과 동일)
        base_visual = ""
        for key in ("direct_prompt", "prompt", "prompt_img", "prompt_movie"):
            val = sc.get(key)
            if isinstance(val, str) and val.strip():
                base_visual = val.strip()
                break

        scene_lyric = sc.get("lyric", "")

        if not base_visual and not scene_lyric:  # 둘 다 없으면 스킵
            _log(f"[{scene_id}] 참조 텍스트(prompt/lyric)가 없어 AI 호출을 건너뜁니다.")
            continue

        # --- ▼▼▼ [신규] 5. '다음 씬 가사' 찾기 ▼▼▼ ---
        next_scene_lyric = "(Scene End)"
        if i + 1 < len(scenes):
            next_sc = scenes[i + 1]
            if isinstance(next_sc, dict):
                next_scene_lyric = next_sc.get("lyric", "") or "(Next scene has no lyric)"
        # --- ▲▲▲ [신규] 끝 ▲▲▲ ---

        # 6. AI 호출 (씬당 1회)
        _log(f"[{scene_id}] {len(segs)}개 세그먼트 프롬프트(묘사) AI 요청 중...")

        frame_ranges_info = [f"{s.get('start_frame')}-{s.get('end_frame')}f" for s in segs]

        # --- ▼▼▼ [수정] AI에게 전달할 문맥 6가지로 확장 ▼▼▼ ---
        user_prompt_payload = {
            "original_vibe": original_vibe_prompt,
            "scene_lyric": scene_lyric,
            "base_visual": base_visual,
            "characters": sc.get("characters", []),
            "time_structure": frame_ranges_info,
            "next_scene_lyric": next_scene_lyric
        }
        user_msg = json.dumps(user_prompt_payload, ensure_ascii=False)
        # --- ▲▲▲ [수정] 끝 ▲▲▲ ---

        try:
            ai_raw_response = ask(system_msg, user_msg)

            # (AI 응답 파싱)
            json_start = ai_raw_response.find("{")
            json_end = ai_raw_response.rfind("}") + 1
            if not (0 <= json_start < json_end):
                raise RuntimeError(f"AI가 JSON을 반환하지 않았습니다: {ai_raw_response[:100]}")

            ai_json = json.loads(ai_raw_response[json_start:json_end])

            # --- ▼▼▼ [수정] 'segment_prompts' 키로 파싱 ▼▼▼ ---
            new_prompts = ai_json.get("segment_prompts", [])

            if not isinstance(new_prompts, list) or len(new_prompts) != len(segs):
                raise RuntimeError(f"AI가 요청된 세그먼트 개수({len(segs)})만큼 프롬프트를 반환하지 않았습니다. (반환: {len(new_prompts)}개)")

            # 'prompt_movie' 필드에 "행동 묘사"를 저장
            filled_count = 0
            for i_seg, seg in enumerate(segs):
                if not seg.get("prompt_movie", ""):
                    seg["prompt_movie"] = str(new_prompts[i_seg]).strip()
                    filled_count += 1

            if filled_count > 0:
                _log(f"[{scene_id}] AI로 {filled_count}개의 새 행동 묘사 주입 완료.")
                sc["frame_segments"] = segs
                changed = True
            else:
                _log(f"[{scene_id}] AI가 묘사를 반환했으나, 이미 모든 프롬프트가 채워져 있었습니다.")
            # --- ▲▲▲ [수정 끝] ▲▲▲ ---

        except Exception as e_ai_call:
            _log(f"[{scene_id}] AI 호출 또는 묘사 주입 실패: {e_ai_call}")
            continue

    if changed:
        vdoc["scenes"] = scenes
        save_json(vpath, vdoc)
        _log("[fill_prompt_movie_with_ai] FPS 동기화, frame_segments 생성/보강, AI 행동 묘사 채우기 완료.")
    else:
        save_json(vpath, vdoc)
        _log("[fill_prompt_movie_with_ai] 변경 없음 (또는 FPS 동기화만 수행)")


# video_build.py (파일 상단, import 구역 바로 뒤에 붙여넣으세요)

def _probe_nb_frames_ffprobe(ffprobe_exe_local: str, src_path_local: Path) -> int:
    """[신규-전역] ffprobe로 비디오의 총 프레임 수를 읽어옵니다."""

    cmd_probe = [
        ffprobe_exe_local,
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(src_path_local),
    ]
    proc_probe = subprocess.run(
        cmd_probe,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
    )
    if proc_probe.returncode != 0:
        return 0
    try:
        return int((proc_probe.stdout or "0").strip() or "0")
    except ValueError:
        return 0


def _trim_to_frames(
        ffmpeg_exe_local: str,
        ffprobe_exe_local: str,
        src_local: Path,
        dst_local: Path,
        target_frames: int,
        fps_val: int,
        on_progress: "Optional[Callable[[Dict[str, Any]], None]]" = None,
) -> bool:
    """[신규-전역] 비디오를 target_frames 길이로 정확히 잘라냅니다."""

    def _notify(notify_msg: str) -> None:
        if on_progress is None: return
        try:
            on_progress({"msg": "[TRIM] " + notify_msg})
        except Exception:
            pass

    if target_frames <= 0:
        return False
    tmp_out_local = dst_local.with_suffix(".tmp.mp4")
    if tmp_out_local.exists():
        try:
            tmp_out_local.unlink()
        except OSError:
            pass
    trim_cmd = [
        ffmpeg_exe_local, "-y",
        "-fflags", "+genpts",
        "-i", str(src_local),
        "-vf", "trim=end_frame={0},setpts=PTS-STARTPTS".format(int(target_frames)),
        "-r", str(int(max(1, fps_val))),
        "-fps_mode", "cfr",
        "-vsync", "cfr",
        "-map_metadata", "-1",
        "-metadata", "title=",
        "-sn",
        "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(tmp_out_local),
    ]
    trim_proc = subprocess.run(
        trim_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
    )
    if trim_proc.returncode != 0:
        _notify(f"ffmpeg 트림 실패: {trim_proc.stdout}")
        if tmp_out_local.exists():
            try:
                tmp_out_local.unlink()
            except OSError:
                pass
        return False
    got_frames = _probe_nb_frames_ffprobe(ffprobe_exe_local, tmp_out_local)
    if got_frames != int(target_frames):
        _notify("프레임 불일치: 기대 {0}f vs 실제 {1}f".format(target_frames, got_frames))
    try:
        if dst_local.exists():
            dst_local.unlink()
    except OSError:
        pass
    tmp_out_local.rename(dst_local)
    return True


# video_build.py 맨 아래에 추가

def retry_cut_audio_for_scene(project_dir: str, scene_id: str, offset: float) -> str:
    """
    [UI 요청] 특정 씬의 오디오를 오프셋(싱크 조절)을 적용하여 다시 자릅니다.
    - offset > 0 : 오디오 시작 지점을 뒤로 밈 (늦게 시작)
    - offset < 0 : 오디오 시작 지점을 앞으로 당김 (일찍 시작)
    """
    p_dir = Path(project_dir)

    # 1. 원본 오디오(vocal) 찾기
    src_audio = p_dir / "vocal.wav"
    if not src_audio.exists():
        src_audio = p_dir / "vocal.mp3"
        if not src_audio.exists():
            raise FileNotFoundError(f"프로젝트 폴더에 vocal.wav 또는 vocal.mp3가 없습니다: {p_dir}")

    # 2. video.json에서 씬 정보 읽기
    video_json_path = p_dir / "video.json"
    video_data = load_json(video_json_path, {}) or {}
    scenes = video_data.get("scenes", [])

    target_scene = next((s for s in scenes if s.get("id") == scene_id), None)
    if not target_scene:
        raise ValueError(f"video.json에서 씬 ID '{scene_id}'를 찾을 수 없습니다.")

    # 3. 시간 계산 (Start/End + Offset)
    orig_start = float(target_scene.get("start", 0.0))
    orig_end = float(target_scene.get("end", 0.0))

    # ★ 싱크 적용 로직
    new_start = max(0.0, orig_start + offset)  # 0초보다 작아질 수 없음
    new_end = max(new_start + 0.1, orig_end + offset)  # 최소 길이 보장

    # 4. 저장 경로 (clips/씬ID/song.wav)
    scene_dir = p_dir / "clips" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    out_audio = scene_dir / "song.wav"

    # 5. 자르기 실행 (_slice_audio_segment 재사용)
    # ffmpeg_exe는 전역 설정에서 가져옴
    success = _slice_audio_segment(
        src_audio=src_audio,
        start_sec=new_start,
        end_sec=new_end,
        out_audio=out_audio,
        ffmpeg_exe=FFMPEG_EXE
    )

    if not success:
        raise RuntimeError(f"FFmpeg 오디오 자르기 실패 (Scene: {scene_id})")

    return str(out_audio)




# ======================= /A안 GPT 적용기 끝 =======================

if __name__ == "__main__":
    info = verify_demo_block()
    print("BEFORE:", info)
    if info["exists"]:
        print(strip_demo_block())
    else:
        print("No demo block found.")
