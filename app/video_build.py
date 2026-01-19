# -*- coding: utf-8 -*-
# i2v 분할/실행/합치기(교차 페이드) + 누락 이미지 생성 (ComfyUI 연동)
from __future__ import annotations

from pathlib import Path as _Path
import os
from PIL import Image
import uuid
from app import settings
# ── 유연 임포트 ─────────────────────────────────────────────────────────────
from app.utils import ensure_dir, load_json, get_duration, _ffmpeg_escape_drawtext, SubtitleComposer, SubtitleStyle
from app.settings import BASE_DIR, I2V_WORKFLOW, FFMPEG_EXE, USE_HWACCEL, FINAL_OUT, COMFY_HOST
# music_gen에 있는 견고한 함수들을 우선 재사용 (가능할 때)
from app.utils import _submit_and_wait as _submit_and_wait_comfy_func
try:
    from app.audio_sync import (
        _http_get as _http_get_audio,
        _load_workflow_graph as _load_workflow_graph_audio,
    )
except Exception:  # 단독 실행/상대 경로일 수도 있으니 폴백 제공
    _http_get_audio = None
    _load_workflow_graph_audio = None
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
    _ffmpeg_escape_filter_path,
    split_subtitle_two_lines
)

from app.settings import JSONS_DIR
from pathlib import Path



import json as _json_loader

import json as _json
import random as _random
import time as _time
import urllib.parse as _urlparse
import uuid as _uuid
from pathlib import Path as _P
import requests
import shutil

import random as _img_seed_random
from app import settings as settings_obj

from app.settings import CHARACTER_DIR, COMFY_INPUT_DIR, I2V_CHUNK_BASE_FRAMES, I2V_OVERLAP_FRAMES, I2V_PAD_TAIL_FRAMES

from app.utils import load_json, save_json, resolve_windows_fontfile


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

# 영상 등 합치기 ; shorts 탭 음악 버젼
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


# 영상합치기 shopping 탭
def concatenate_scene_clips_final_av(
    *,
    clip_paths: List[Path],
    out_path: Path,
    ffmpeg_exe: str,
    scenes: List[Dict[str, Any]],
    bgm_path: Optional[Path] = None,
    bgm_volume: float = 0.0,
    narration_volume: float = 1.0,

    # (호환 유지: 최종 병합에서는 0으로 넘기는 것을 권장)
    pad_in_sec: float = 0.0,
    pad_out_sec: float = 0.0,

    subtitle_font: str = "Gulim",
    subtitle_fontsize: int = 36,
    subtitle_y: str = "h-140",
    subtitle_box: bool = True,
    subtitle_boxcolor: str = "black@0.45",
    subtitle_boxborderw: int = 18,

    # ✅ 자막 페이드 전용
    subtitle_fade_in_sec: float = 0.25,
    subtitle_fade_out_sec: float = 0.25,

    title_text: str = "",
    title_fontsize: int = 55,
    title_y: str = "h*0.12",
    video_crf: int = 18,
    video_preset: str = "medium",
    audio_bitrate: str = "192k",

    # ✅ 내레이션 가속 상한
    max_narration_speed: float = 1.30,

    # ✅ 진행 로그(비동기 창)
    on_progress=None,
) -> None:
    """
    [최종 합치기 - 오버랩 없음, 자막 페이드]
    - clip_paths를 concat demuxer로 이어붙여 temp_visual.mp4 생성(오디오는 제외)
    - 각 클립의 "실제 길이"를 get_duration()으로 측정하여 scene별 movie_duration에 기록
    - 각 voice_file의 "실제 길이"를 get_duration()으로 측정하여 narration_duration에 기록
    - 내레이션/자막은 해당 씬(클립)의 중앙에 배치
    - 내레이션이 너무 길면 필요한 만큼 atempo로 줄임(상한 max_narration_speed)
    - drawtext는 제목(선택) + 자막(씬별)에 alpha 기반 fade-in/out 적용
    - ✅ 자막 줄바꿈은 텍스트에 개행을 넣지 않고 drawtext 여러 개로 처리(Windows 안정)
    - ✅ 폰트 해결/줄나눔/줄간격/y올림은 SubtitleComposer가 전담(재사용 목적)
    """

    def _p(msg: str) -> None:
        if callable(on_progress):
            try:
                on_progress(msg)
            except Exception:
                pass

    if not clip_paths:
        raise ValueError("clip_paths가 비었습니다.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    work_dir = out_path.parent
    list_file = work_dir / "ffmpeg_concat_list_final.txt"
    temp_visual = work_dir / (out_path.stem + "_temp_visual.mp4")

    # 1) concat list 생성
    with open(list_file, "w", encoding="utf-8") as f:
        for clip in clip_paths:
            if not clip.exists():
                raise FileNotFoundError(f"병합할 클립을 찾을 수 없습니다: {clip}")
            f.write(f"file '{clip.as_posix()}'\n")

    # 2) 비디오만 합쳐서 temp_visual 생성
    cmd_concat = [
        ffmpeg_exe, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-an",
        "-c:v", "libx264",
        "-preset", video_preset,
        "-crf", str(video_crf),
        str(temp_visual),
    ]

    startupinfo = None
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    r = subprocess.run(
        cmd_concat,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        startupinfo=startupinfo,
    )
    if r.returncode != 0:
        raise RuntimeError(f"FFMPEG concat(temp_visual) 실패:\n{r.stderr}")

    visual_dur = float(get_duration(str(temp_visual)) or 0.0)
    if visual_dur <= 0:
        visual_dur = 1.0

    # 3) 각 씬별 실제 영상 길이 측정 → start/end 재구성
    clip_durs: List[float] = []
    for cp in clip_paths:
        d = float(get_duration(str(cp)) or 0.0)
        if d <= 0:
            d = max(0.01, visual_dur / max(len(clip_paths), 1))
        clip_durs.append(d)

    t_cursor = 0.0
    for i, sc in enumerate(scenes):
        if i >= len(clip_durs):
            break
        d = clip_durs[i]
        sc["movie_duration"] = float(d)
        sc["start"] = float(t_cursor)
        sc["end"] = float(t_cursor + d)
        t_cursor += d

    _p(f"[Merge][DBG] total_visual={visual_dur:.3f}s, scenes_used={min(len(scenes), len(clip_durs))}")

    # 4) 오디오 입력/필터 구성
    inputs: List[str] = [str(temp_visual)]
    audio_inputs_meta: List[Tuple[str, int]] = []  # (type, input_index)

    if bgm_path and Path(bgm_path).is_file():
        inputs.append(str(Path(bgm_path)))
        audio_inputs_meta.append(("bgm", len(inputs) - 1))

    narration_items: List[Dict[str, Any]] = []

    # 허용 오차(초) — duration 측정 오차 방지(예: 50ms)
    EPS_SEC = 0.05

    for i, sc in enumerate(scenes):
        vf = (sc.get("voice_file") or "").strip()
        if not vf:
            continue

        vp = Path(vf)
        if not vp.is_absolute():
            cand = (work_dir / vf).resolve()
            if cand.is_file():
                vp = cand
        if not vp.is_file():
            continue

        start_t = float(sc.get("start") or 0.0)
        end_t = float(sc.get("end") or start_t)
        clip_dur = float(sc.get("movie_duration") or (end_t - start_t) or 0.0)
        if clip_dur <= 0:
            continue

        nar_dur_raw = float(get_duration(str(vp)) or 0.0)
        sc["narration_duration"] = float(nar_dur_raw)

        speed = 1.0
        nar_dur_eff = float(nar_dur_raw)
        required = (nar_dur_raw / max(0.001, clip_dur)) if clip_dur > 0 else 1.0

        if nar_dur_raw > clip_dur + EPS_SEC:
            speed = min(float(max_narration_speed), max(1.0, round(required + 0.01, 2)))
            nar_dur_eff = nar_dur_raw / speed

            _p(
                f"[Merge][DBG] scene={sc.get('id')} clip={clip_dur:.3f}s nar={nar_dur_raw:.3f}s "
                f"required≈{required:.3f} -> speed={speed:.2f} eff={nar_dur_eff:.3f}s"
            )

            if nar_dur_eff > clip_dur + EPS_SEC:
                raise RuntimeError(
                    f"내레이션이 씬 길이를 초과합니다. scene_id={sc.get('id')} "
                    f"clip={clip_dur:.3f}s nar={nar_dur_raw:.3f}s "
                    f"speed={speed:.2f} eff={nar_dur_eff:.3f}s (max={max_narration_speed:.2f})"
                )
        else:
            _p(f"[Merge][DBG] scene={sc.get('id')} clip={clip_dur:.3f}s nar={nar_dur_raw:.3f}s -> OK(no speed)")

        center_offset = max(0.0, (clip_dur - nar_dur_eff) * 0.5)
        delay_sec = start_t + center_offset

        sc["narration_speed"] = float(speed)
        sc["narration_effective_duration"] = float(nar_dur_eff)

        narration_items.append({
            "scene_index": i,
            "delay_sec": float(delay_sec),
            "path": vp,
            "speed": float(speed),
            "effective_dur": float(nar_dur_eff),
        })

    for item in narration_items:
        inputs.append(str(item["path"]))
        audio_inputs_meta.append(("nar", len(inputs) - 1))

    fc_parts: List[str] = []
    mix_labels: List[str] = []

    # bgm
    bgm_label = None
    for typ, idx in audio_inputs_meta:
        if typ == "bgm":
            bgm_label = "abgm"
            fc_parts.append(f"[{idx}:a]volume={bgm_volume}[{bgm_label}]")
            mix_labels.append(f"[{bgm_label}]")
            break

    # narration (adelay + (optional) atempo)
    nar_count = 0
    nar_input_base = 1 + (1 if bgm_label else 0)

    for j, item in enumerate(narration_items):
        idx = nar_input_base + j
        ms = int(round(float(item["delay_sec"]) * 1000.0))
        lbl = f"anar{nar_count}"
        spd = float(item["speed"])

        if abs(spd - 1.0) < 1e-6:
            fc_parts.append(f"[{idx}:a]adelay={ms}|{ms},volume={narration_volume}[{lbl}]")
        else:
            fc_parts.append(f"[{idx}:a]atempo={spd:.6f},adelay={ms}|{ms},volume={narration_volume}[{lbl}]")

        mix_labels.append(f"[{lbl}]")
        nar_count += 1

    if not mix_labels:
        fc_parts.append(f"anullsrc=r=48000:cl=stereo,atrim=0:{visual_dur:.6f}[aout]")
    else:
        mix_in = "".join(mix_labels)
        fc_parts.append(f"{mix_in}amix=inputs={len(mix_labels)}:normalize=0:dropout_transition=0[aout]")

    # ============================================================
    # 5) 자막(drawtext) 필터: 중앙 구간 + alpha fade
    #    - SubtitleComposer가 폰트 resolve + 줄 나눔 + y 자동올림까지 전부 책임
    #    - 여기서는 "씬별로 show/hide/alpha만 계산해서 composer에 넘긴다"
    # ============================================================

    vf_parts: List[str] = []
    vf_in = "[0:v]"
    vf_out = "vout"

    # (1) Composer 생성: 여기 값만 바꾸면 "자막 정책"이 통째로 바뀜
    #     max_units=20.0  : 실제 화면 기준 한 줄 제한(기본 20)
    #     max_lines=2     : UX 망가짐 방지. 필요하면 3/4로 호출자가 변경 가능
    #     line_gap_px=None: None이면 fontsize/boxborderw 기반 자동 계산
    #     lift_ratio=0.62 : 줄 수가 늘어날 때 블록을 위로 끌어올리는 비율
    composer = SubtitleComposer(
        font_family=subtitle_font,
        fontsize=subtitle_fontsize,
        y=subtitle_y,
        box=subtitle_box,
        boxcolor=subtitle_boxcolor,
        boxborderw=subtitle_boxborderw,
        max_units=20.0,  # ✅ 여기 숫자만 바꾸면 됨
        max_lines=3,  # ✅ 기본값 2 유지 (호출 시 3으로도 가능)
        line_gap_px=None,
        lift_ratio=0.62,
    )

    # 디버그 로그: 폰트가 실제 파일로 떨어졌는지 확인 가능
    _p(f"[Merge][DBG] subtitle_font='{subtitle_font}' resolved_fontfile='{composer.fontfile}'")
    if callable(on_progress):
        on_progress(
            f"[Merge][DBG] drawtext fontfile={'OK' if composer.fontfile else 'NONE'} : {composer.fontfile or subtitle_font}")

    # (2) 제목 drawtext(기존 alpha 유지) — font_arg도 composer가 책임
    title_text = (title_text or "").strip()
    if title_text:
        title_alpha = (
            "if(lt(t,1),0,"
            " if(lt(t,3),(t-1)/2,"
            "  if(lt(t,4),1,"
            "   if(lt(t,6),(6-t)/2,0)"
            "  )"
            " )"
            ")"
        )
        title_dt = composer.build_title_drawtext(
            title_text=title_text,
            title_fontsize=int(title_fontsize),
            title_y=str(title_y),
            alpha_expr=title_alpha,
        )
        if title_dt:
            vf_parts.append(title_dt)

    fade_in = max(0.0, float(subtitle_fade_in_sec))
    fade_out = max(0.0, float(subtitle_fade_out_sec))

    # (3) 씬별 자막 처리: 여기 목표는 "3~6줄" 수준으로 유지
    for sc in scenes:
        start_t = float(sc.get("start") or 0.0)
        end_t = float(sc.get("end") or start_t)
        clip_dur = float(sc.get("movie_duration") or (end_t - start_t) or 0.0)
        if clip_dur <= 0:
            continue

        # 씬에서 표시할 텍스트 선택(우선순위 유지)
        txt = (sc.get("lyric") or sc.get("subtitle") or sc.get("narration") or "").strip()
        if not txt:
            continue

        # 표시 길이 계산(내레이션 유효길이 우선, 없으면 clip의 70%)
        nar_dur_eff = float(sc.get("narration_effective_duration") or sc.get("narration_duration") or 0.0)
        if nar_dur_eff <= 0:
            nar_dur_eff = clip_dur * 0.7
        if nar_dur_eff > clip_dur:
            nar_dur_eff = clip_dur

        show_t = start_t + max(0.0, (clip_dur - nar_dur_eff) * 0.5)
        hide_t = min(end_t, show_t + nar_dur_eff)
        if hide_t <= show_t:
            hide_t = min(end_t, show_t + 0.10)

        # alpha 페이드 표현식
        if fade_in <= 1e-6 and fade_out <= 1e-6:
            alpha_expr = "1"
        else:
            fi = max(1e-6, fade_in)
            fo = max(1e-6, fade_out)
            alpha_expr = (
                f"if(lt(t,{show_t:.3f}),0,"
                f" if(lt(t,{(show_t + fade_in):.3f}), (t-{show_t:.3f})/{fi:.6f},"
                f"  if(lt(t,{(hide_t - fade_out):.3f}), 1,"
                f"   if(lt(t,{hide_t:.3f}), ({hide_t:.3f}-t)/{fo:.6f}, 0)"
                f"  )"
                f" )"
                f")"
            )

        # ✅ 자막 drawtext 생성: (줄 나눔/균형/y 자동올림) 전부 composer가 처리
        vf_parts.extend(
            composer.build_subtitle_drawtexts(
                text=txt,
                show_t=show_t,
                hide_t=hide_t,
                alpha_expr=alpha_expr,
            )
        )

    vf_chain = ",".join(vf_parts) if vf_parts else "null"
    vf = f"{vf_in}{vf_chain}[{vf_out}]"

    # 6) 최종 ffmpeg 렌더
    cmd = [ffmpeg_exe, "-y"]
    for p in inputs:
        cmd += ["-i", p]

    filter_complex = ";".join(fc_parts + [vf])
    cmd += [
        "-filter_complex", filter_complex,
        "-map", f"[{vf_out}]",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-preset", video_preset,
        "-crf", str(video_crf),
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-movflags", "+faststart",
        "-shortest",
        str(out_path),
    ]

    rr = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        startupinfo=startupinfo,
    )
    if rr.returncode != 0:
        raise RuntimeError(f"FFMPEG 최종 렌더 실패:\n{rr.stderr}")

    # 7) 임시 파일 정리
    try:
        if list_file.exists():
            list_file.unlink()
    except Exception:
        pass
    try:
        if temp_visual.exists():
            temp_visual.unlink()
    except Exception:
        pass






# ===================[[[[[[[[여기까지 영상합치기]]]]]]]============

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

    # # 이미 있으면 재사용
    # if out_path.exists() and out_path.stat().st_size > 0:
    #     _log(f"[FRAME] 이미 존재하는 정지 프레임 재사용: {out_path.name}")
    #     return out_path

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


def build_shots_with_i2v_long(
        project_dir: str,
        total_frames: int = 0,  # video.json 값 사용
        ui_fps: int = 24,
        on_progress: Callable = print
):
    """
    [New] 영상 생성 메인 파이프라인 (Long Take 버전)
    1. 75번 워크플로우를 이용해 가변 프레임 Raw 영상 생성
    2. 생성된 Raw 영상 일괄 업스케일 (Interpolation_upscale)
    """
    p_dir = Path(project_dir)
    v_path = p_dir / "video.json"
    clips_dir = p_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    wf_path = Path(settings.JSONS_DIR) / "75.wan22_SVI_Pro.json"
    if not wf_path.exists():
        wf_path = Path(r"C:\my_games\shorts_make\app\jsons\75.wan22_SVI_Pro.json")

    with open(wf_path, "r", encoding="utf-8") as f:
        wf_template = json.load(f)

    comfy_url = getattr(settings, "COMFY_HOST", "http://127.0.0.1:8188")

    # video.json 로드
    data = json.loads(v_path.read_text(encoding="utf-8"))

    # UI 설정값 우선 (video.json에 저장된 defaults 값 사용)
    defaults = data.get("defaults", {})
    width = int(defaults.get("image", {}).get("width", 720))
    height = int(defaults.get("image", {}).get("height", 1280))

    scenes = data.get("scenes", [])

    # 1. Raw 영상 생성 루프
    on_progress({"msg": "🎬 [Step 1] RAW 영상 생성 시작 (Long Take Mode)..."})

    raw_created_count = 0

    for sc in scenes:
        sid = sc.get("id")
        raw_path = clips_dir / f"{sid}_raw.mp4"
        final_path = clips_dir / f"{sid}.mp4"

        # 최종본 있으면 스킵
        if final_path.exists() and final_path.stat().st_size > 1000:
            on_progress({"msg": f"   ⏭️ Skip {sid} (Final exists)"})
            continue

        # Raw 있으면 스킵 (업스케일 대기)
        if raw_path.exists() and raw_path.stat().st_size > 1000:
            on_progress({"msg": f"   ⏭️ Skip Gen {sid} (Raw exists)"})
            raw_created_count += 1
            continue

        # 생성
        on_progress({"msg": f"   🎥 Generating {sid}..."})
        success = raw_make_addup_long(
            scene_data=sc,
            workflow_template=wf_template,
            ui_width=width,
            ui_height=height,
            comfy_url=comfy_url,
            output_dir=clips_dir,
            log_fn=lambda m: on_progress({"msg": m})
        )
        if success:
            raw_created_count += 1
        else:
            on_progress({"msg": f"   ❌ Failed {sid}"})

    # 2. 일괄 업스케일 (기존 함수 재사용)
    if raw_created_count > 0:
        on_progress({"msg": "✨ [Step 2] 업스케일링 (Interpolation) 시작..."})
        try:

            Interpolation_upscale(
                str(p_dir),
                total_frames=0,
                ui_width=width,
                ui_height=height,
                ui_fps=ui_fps,
                on_progress=on_progress,
            )

            on_progress({"msg": "✅ 모든 영상 생성 및 업스케일 완료!"})
        except Exception as e:
            on_progress({"msg": f"❌ 업스케일 중 오류: {e}"})
    else:
        on_progress({"msg": "✅ 생성할 신규 영상이 없습니다."})



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
    [1단계] RAW I2V 생성 전담 함수 (Hard Cut 버전)
    - 400줄 줄어든 이유: 반복문 최적화 (기능 삭제 없음)
    - xfade 제거됨 -> Hard Cut (앞 영상 꼬리 자르고 붙이기) 적용
    """

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
                ffprobe_exe_val, "-v", "error", "-select_streams", "v:0",
                "-count_frames", "-show_entries", "stream=nb_read_frames",
                "-of", "default=nokey=1:noprint_wrappers=1", str(path_obj),
            ]
            proc = subprocess.run(
                cmd_args, capture_output=True, text=True, check=True,
            )
            txt = (proc.stdout or "").strip()
            if not txt:
                return None
            return int(txt)
        except Exception:
            return None

    def _duration(csrc_path: Path) -> float:
        try:
            ccmd_args = [
                ffprobe_exe_val, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=duration", "-of", "default=nokey=1:noprint_wrappers=1",
                str(csrc_path),
            ]
            procc = subprocess.run(
                ccmd_args, capture_output=True, text=True, check=True,
            )
            txt = (procc.stdout or "").strip()
            if not txt:
                return 0.0
            return float(txt)
        except Exception:
            return 0.0

    # 립싱크용 오디오 자르기 (누락 방지)
    def _slice_audio_segment(src_audio: Path, start_sec: float, end_sec: float, out_audio: Path,
                             ffmpeg_path: str) -> bool:
        if not src_audio.exists(): return False
        duration = end_sec - start_sec
        if duration <= 0: return False
        try:
            cmd = [
                ffmpeg_path, "-y", "-i", str(src_audio),
                "-ss", f"{start_sec:.3f}", "-t", f"{duration:.3f}",
                "-c:a", "pcm_s16le", str(out_audio)
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return out_audio.exists()
        except Exception:
            return False

    def _i2v_trim_tail(
            path_in: Path, path_out: Path, target_frames: int, fps_val: int
    ) -> None:
        nb_frames = _i2v_probe_nb_frames(path_in)
        if nb_frames is None:
            if path_in != path_out:
                shutil.copy2(str(path_in), str(path_out))
            return

        if nb_frames <= target_frames:
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
            ffmpeg_exe_val, "-y", "-i", str(path_in), "-t", f"{sec:.6f}",
            "-c", "copy", str(tmp_out),
        ]

        try:
            subprocess.run(
                cmd_args, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="ignore",
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

    # [수정] 하드 컷(Hard Cut) 적용된 병합 함수
    def _i2v_concat_ab(
            path_a: Path,
            path_b: Path,
            path_out: Path,
            crossfade_sec: float = 0.3,
    ) -> None:
        for xp in (path_a, path_b):
            if (not xp.exists()) or xp.stat().st_size <= 0:
                raise RuntimeError(f"[RAW] 입력 영상이 비정상입니다: {xp}")

        dur_a = _duration(path_a)

        # 오버랩 시간만큼 앞 영상의 끝을 잘라냄 (Hard Cut)
        trim_duration = max(0.0, dur_a - crossfade_sec)

        _notify(
            f"[RAW][MERGE] Hard Cut 적용: "
            f"앞 영상 {dur_a:.3f}s 중 {trim_duration:.3f}s 사용 + 뒷 영상"
        )

        cmd_args = [
            ffmpeg_exe_val, "-y", "-i", str(path_a), "-i", str(path_b),
            "-filter_complex",
            (
                f"[0:v]trim=duration={trim_duration:.6f},setpts=PTS-STARTPTS[v0];"
                f"[1:v]format=yuv420p,setpts=PTS-STARTPTS[v1];"
                f"[v0][v1]concat=n=2:v=1:a=0[v_out]"
            ),
            "-map", "[v_out]", "-an", "-c:v", "libx264", "-crf", "18",
            "-preset", "fast", "-pix_fmt", "yuv420p", str(path_out),
        ]

        try:
            subprocess.run(
                cmd_args, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="ignore",
            )
        except subprocess.CalledProcessError as exxx:
            out_txt = (exxx.stdout or "").strip()
            _notify("[RAW][MERGE] ffmpeg concat 실패")
            if out_txt:
                for line in out_txt.splitlines():
                    _notify(f"[RAW][ffmpeg] {line}")
            raise

    def _i2v_norm_fps_and_size(path_in: Path, path_out: Path, fps_val: int) -> None:
        cmd_args = [
            ffmpeg_exe_val, "-y", "-i", str(path_in), "-vf", f"fps={fps_val}",
            "-c:v", "libx264", "-crf", "18", "-preset", "slow", str(path_out),
        ]
        subprocess.run(cmd_args, check=True)

    def _i2v_plan_segments(
            frame_len_total: int, base_chunk: int = 82, overlap_val: int = 6, pad_tail_val: int = 5,
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
                    "i2v 워크플로우 파일을 찾을 수 없습니다."
                )

    _notify(f"[RAW] 사용할 워크플로우: {workflow_path.name} (경로: {workflow_path})")

    with workflow_path.open("r", encoding="utf-8") as wf_f:
        try:
            graph_origin = json_mod.load(wf_f)
        except Exception as exc:
            raise RuntimeError(f"워크플로우 JSON 파싱 실패: {exc}")

    RES_NODE_IDS = ["21", "22", "23"]
    STEPS_NODE_IDS = ["15", "16"]

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

    graph_lip = {}
    if WF_LIPSYNC.exists():
        try:
            with open(WF_LIPSYNC, "r", encoding="utf-8") as f:
                graph_lip = json_mod.load(f)
        except Exception as e:
            _notify(f"[WARN] 립싱크 워크플로우 로드 실패: {e}")

    audio_source = project_dir_path / "vocal.wav"
    if not audio_source.exists():
        audio_source = project_dir_path / "vocal.mp3"

    scene_index = 0
    for scene_item in scenes:
        scene_index += 1
        if not isinstance(scene_item, dict):
            continue

        scene_id = str(scene_item.get("id") or f"scene_{scene_index:03d}")
        scene_out_dir = clips_dir / scene_id
        _ensure_dir_func(scene_out_dir)

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

        if scene_raw_norm.exists() and scene_raw_norm.stat().st_size > 1024:
            _notify(f"[RAW] 씬 {scene_index}/{total_scenes} ({scene_id}) RAW 결과 존재 → 스킵")
            continue

        # =========================================================
        # [CASE 1] 립싱크 모드 (유지)
        # =========================================================
        is_lipsync = scene_item.get("lync_bool", False)

        if is_lipsync and graph_lip and audio_source.exists():
            lync_prompt = scene_item.get('lync_prompt', 'sing a song')
            _notify(f"  -> [LipSync Mode] ON (Prompt: {lync_prompt})")

            scene_audio_path = scene_out_dir / "song.wav"
            if scene_audio_path.exists() and scene_audio_path.stat().st_size > 0:
                _notify(f"  -> [LipSync] 기존 오디오 파일 사용 (스킵): {scene_audio_path.name}")
            else:
                if not _slice_audio_segment(audio_source, scene_start, scene_end, scene_audio_path, ffmpeg_exe_val):
                    _notify(f"  -> [ERR] 오디오 자르기 실패. 일반 모드로 전환합니다.")
                    is_lipsync = False

            if is_lipsync:
                graph = json_mod.loads(json_mod.dumps(graph_lip))
                audio_filename = f"{scene_id}_song.wav"
                comfy_audio_dst = comfy_input_dir / audio_filename

                try:
                    shutil.copy2(str(scene_audio_path), str(comfy_audio_dst))
                    if "125" in graph:
                        graph["125"]["inputs"]["audio"] = audio_filename
                except Exception as e:
                    _notify(f"  -> [WARN] 오디오 파일 복사 실패(절대경로 시도): {e}")
                    if "125" in graph:
                        graph["125"]["inputs"]["audio"] = str(scene_audio_path)

                img_name = f"{scene_id}.png"
                src_img = project_dir_path / "imgs" / img_name

                if src_img.exists():
                    shutil.copy2(str(src_img), comfy_input_dir / img_name)
                    if "284" in graph: graph["284"]["inputs"]["image"] = img_name
                else:
                    _notify(f"  -> [ERR] 대표 이미지({img_name}) 없음.")
                    continue

                if "245" in graph: graph["245"]["inputs"]["value"] = target_w
                if "246" in graph: graph["246"]["inputs"]["value"] = target_h
                if "241" in graph: graph["241"]["inputs"]["positive_prompt"] = lync_prompt
                if "270" in graph: graph["270"]["inputs"]["value"] = frame_length_val
                if "128" in graph: graph["128"]["inputs"]["steps"] = target_steps
                if "131" in graph: graph["131"]["inputs"]["filename_prefix"] = f"lipsync/{scene_id}"

                _notify(f"  -> ComfyUI 요청 (Duration: {scene_duration:.2f}s, Frames: {frame_length_val})...")
                try:
                    res = _submit_and_wait_comfy_func(comfy_host, graph, timeout=1200, poll=2.0,
                                                      on_progress=lambda x: None)

                    out_node = res.get("outputs", {}).get("131", {})
                    vid_list = out_node.get("videos") or out_node.get("gifs") or []

                    if vid_list:
                        fname = vid_list[0]['filename']
                        subfolder = vid_list[0]['subfolder']
                        r = requests.get(f"{comfy_host}/view", params={"filename": fname, "subfolder": subfolder},
                                         timeout=60)
                        with open(scene_raw_norm, "wb") as f:
                            f.write(r.content)

                        if scene_raw_norm.exists():
                            _notify(f"  -> [LipSync] 길이 보정: {frame_length_val} 프레임으로 자르기...")
                            _i2v_trim_tail(
                                scene_raw_norm,
                                scene_raw_norm,
                                int(frame_length_val),
                                int(base_fps_in)
                            )
                        _notify(f"  -> [LipSync] 생성 완료: {scene_raw_norm.name}")
                        continue
                    else:
                        _notify(f"  -> [ERR] 결과물 없음. 일반 모드로 재시도합니다.")
                except Exception as e:
                    _notify(f"  -> [ERR] ComfyUI 실행 중 오류: {e}")

        # ── 세그먼트 / 프롬프트 구성 ──────────────────────────────
        segments_list: List[Tuple[int, int]] = []
        segment_prompts: List[str] = []

        base_scene_pos = str(
            scene_item.get("prompt_movie") or scene_item.get("prompt") or ""
        ).strip()

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

                segments_list = sorted_segments
                segment_prompts = sorted_prompts

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

        PAD_TAIL_FRAMES_USER = 5
        OVERLAP_FRAMES_USER = 6

        chunk_paths: List[Path] = []

        for chunk_index, (start_f, end_f) in enumerate(segments_list):

            pure_len = end_f - start_f

            if chunk_index == 0:
                gen_len = pure_len + PAD_TAIL_FRAMES_USER
                target_save_len = pure_len
            else:
                gen_len = OVERLAP_FRAMES_USER + pure_len + PAD_TAIL_FRAMES_USER
                target_save_len = OVERLAP_FRAMES_USER + pure_len

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
                f"(TargetFrames={gen_len}, SaveFrames={target_save_len} | "
                f"Original: {start_f}~{end_f})"
            )

            graph_chunk = json_mod.loads(json_mod.dumps(graph_origin))
            fixed_seed = random.randint(1, 9999999999)
            for nid in ["116", "117", "391"]:
                if nid in graph_chunk:
                    graph_chunk[nid]["inputs"]["noise_seed"] = fixed_seed

            for nid in RES_NODE_IDS:
                if nid in graph_chunk:
                    graph_chunk[nid]["inputs"]["width"] = target_w
                    graph_chunk[nid]["inputs"]["height"] = target_h
            for nid in STEPS_NODE_IDS:
                if nid in graph_chunk:
                    graph_chunk[nid]["inputs"]["steps"] = target_steps

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

            # ReActor 설정
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

                char_img_path = None
                for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                    p = character_dir_base / f"{char_key}{ext}"
                    if p.exists():
                        char_img_path = p
                        break

                if char_img_path:
                    try:
                        shutil.copy2(str(char_img_path), comfy_input_dir / char_img_path.name)
                        graph_chunk[lid]["inputs"]["image"] = char_img_path.name
                    except Exception as e:
                        _notify(f"[WARN] 캐릭터 복사 실패({char_key}): {e}")

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

            # [이미지 주입 및 강제 추출]
            if chunk_index == 0:
                _notify(
                    f"[RAW][IMG] Seg1 → 대표이미지 사용 "
                    f"scene_id={scene_id}, img_file={scene_item.get('img_file')}"
                )
                _inject_scene_main_image_i2v(
                    graph_chunk, scene_item, comfy_input_dir, _notify
                )
            else:
                prev_raw_path = chunk_paths[chunk_index - 1]

                if prev_raw_path.exists():
                    prev_frames = _i2v_probe_nb_frames(prev_raw_path)
                    if prev_frames is None: prev_frames = 1

                    extract_target_frame = max(0, prev_frames - OVERLAP_FRAMES_USER)
                    extract_time = float(extract_target_frame) / float(base_fps_in)

                    _notify(
                        f"[RAW][IMG] Seg{chunk_index + 1}용 ref 추출 from="
                        f"{prev_raw_path.name} (local_frame={extract_target_frame}, "
                        f"time={extract_time:.3f}s)"
                    )

                    # [강제 추출]
                    ref_img_path = _i2v_extract_still_frame_by_time(
                        prev_raw_path,
                        extract_time,
                        comfy_input_dir,
                        prefix=f"seg{chunk_index}_start",
                    )

                    if ref_img_path:
                        _inject_scene_main_image_i2v(
                            graph_chunk,
                            {"img_file": str(ref_img_path)},
                            comfy_input_dir,
                            _notify,
                        )
                    else:
                        _notify("[RAW][ERR] 이미지 추출 실패. 대표 이미지 사용.")
                        _inject_scene_main_image_i2v(
                            graph_chunk, scene_item, comfy_input_dir, _notify
                        )
                else:
                    _notify("[RAW][ERR] 이전 세그먼트 없음. 대표 이미지 사용.")
                    _inject_scene_main_image_i2v(
                        graph_chunk, scene_item, comfy_input_dir, _notify
                    )

            wan_node_id = "18"
            if wan_node_id in graph_chunk and "inputs" in graph_chunk[wan_node_id]:
                graph_chunk[wan_node_id]["inputs"]["length"] = gen_len

            if "167" in graph_chunk:
                graph_chunk["167"]["inputs"]["filename_prefix"] = (
                    f"temp_raw/{scene_id}_seg{chunk_index + 1:02d}_raw"
                )
                graph_chunk["167"]["inputs"]["save_output"] = True
                graph_chunk["167"]["inputs"]["frame_rate"] = int(base_fps_in)

            if "172" in graph_chunk:
                graph_chunk["172"]["inputs"]["save_output"] = False

            try:
                res = _submit_and_wait_comfy_func(
                    comfy_host,
                    graph_chunk,
                    timeout=10000,
                    poll=10.0,
                    on_progress=lambda _prog: None,
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
                _notify(f"[RAW] 패딩 제거: {gen_len}f → {target_save_len}f 로 자름")
                _i2v_trim_tail(
                    raw_path,
                    raw_path,
                    target_save_len,
                    int(base_fps_in),
                )

        # ───────────────────────── 병합 (Hard Cut) ─────────────────────────
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
            for i, next_path in enumerate(valid_chunk_paths[1:]):
                step_idx = i + 1
                temp_out = scene_out_dir / f"{scene_id}_raw_step_{step_idx:03d}.mp4"

                # [Hard Cut] 오버랩 6프레임 시간만큼 자르고 붙이기
                overlap_sec = float(OVERLAP_FRAMES_USER) / float(base_fps_in)

                _i2v_concat_ab(current_concat, next_path, temp_out, crossfade_sec=overlap_sec)
                current_concat = temp_out

            shutil.copy2(str(current_concat), str(scene_raw_tmp))

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


def raw_make_addup_long(
        scene_data: Dict[str, Any],
        workflow_template: Dict[str, Any],
        ui_width: int,
        ui_height: int,
        comfy_url: str,
        output_dir: Path,
        log_fn: Callable = print
) -> bool:
    """
    [New] 75번 워크플로우(Long Take) 실행 함수 (Strict Output Version)
    - 오직 최종 병합된 영상(Node 204)만 기다리고 가져옵니다.
    - scene_data의 total_frames / seg_count / prompt_1~3 를 활용합니다.
    - 세그먼트 간 overlap(워크플로우의 ImageBatchExtendWithOverlap=5)을 고려하여
      최종 프레임 수가 total_frames가 되도록 세그 length를 계산합니다.
    """


    sid = scene_data.get("id", "unknown")

    # ---------------------------------------------------------
    # 0. total_frames / seg_count 결정
    # ---------------------------------------------------------
    duration = float(scene_data.get("duration", 0) or scene_data.get("seconds", 0) or 0.0)

    # video.json이 total_frames를 이미 갖는 구조(당신 설명)라면 그 값을 우선 신뢰
    if int(scene_data.get("total_frames", 0) or 0) > 0:
        total_frames = int(scene_data["total_frames"])
    else:
        # 혹시 fps가 제공되면 그걸 우선, 없으면 24
        fps = float(scene_data.get("fps", 0) or 24.0)
        total_frames = max(1, int(round(duration * fps)))

    # seg_count도 scene_data에 있으면 우선 신뢰, 없으면 규칙으로 산출
    if int(scene_data.get("seg_count", 0) or 0) in (1, 2, 3):
        seg_count = int(scene_data["seg_count"])
    else:
        if total_frames > 162:
            seg_count = 3
        elif total_frames > 81:
            seg_count = 2
        else:
            seg_count = 1

    # 워크플로우의 overlap 값(기본 템플릿이 5로 설정되어 있음) :contentReference[oaicite:3]{index=3}
    overlap = int(scene_data.get("overlap_frames", 0) or 5)

    # ---------------------------------------------------------
    # 0-b. 세그먼트 길이 계산 (최종 프레임이 total_frames가 되도록)
    #
    # 결합 후 최종 길이:
    # 2세그: L1 + L2 - overlap = total_frames  => L1 + L2 = total_frames + overlap
    # 3세그: L1 + L2 + L3 - 2*overlap = total_frames => 합 = total_frames + 2*overlap
    # ---------------------------------------------------------
    target_sum = total_frames + (seg_count - 1) * overlap

    # 균등 분배(정수) + 나머지 분배
    base = target_sum // seg_count
    rem = target_sum % seg_count
    lens = []
    for i in range(seg_count):
        lens.append(base + (1 if i < rem else 0))

    # 안전장치: 너무 짧으면(특히 overlap보다 짧으면) overlap 블렌딩이 의미 없어질 수 있음
    # 최소 1프레임은 보장, 가능하면 overlap+1 이상이 되도록 조정
    # (극단적으로 total_frames가 작을 때만 발동)
    for i in range(len(lens)):
        if lens[i] < 1:
            lens[i] = 1
    if seg_count >= 2:
        min_len = overlap + 1
        # lens 중 min_len 미만이 있으면, 가능한 범위에서 다른 세그에서 빼서 보정
        for i in range(seg_count):
            if lens[i] < min_len:
                need = min_len - lens[i]
                lens[i] = min_len
                # 다른 세그에서 need만큼 차감
                for j in range(seg_count):
                    if j == i:
                        continue
                    give = min(need, max(0, lens[j] - min_len))
                    if give > 0:
                        lens[j] -= give
                        need -= give
                    if need <= 0:
                        break
                # 그래도 need가 남으면(전체가 너무 짧은 케이스), 그냥 둔다.

    log_fn(f"   [Gen] Scene {sid}: total_frames={total_frames}f, seg_count={seg_count}, "
           f"overlap={overlap}, seg_lengths={lens} (sum={sum(lens)})")

    # ---------------------------------------------------------
    # 1. 이미지 복사 (필수)
    # ---------------------------------------------------------
    img_path_str = scene_data.get("img_file", "")
    target_image_name = ""

    if img_path_str:
        src_path = Path(img_path_str)
        if src_path.exists():
            comfy_input_dir = Path(settings.COMFY_INPUT_DIR)
            if not comfy_input_dir.is_absolute():
                comfy_input_dir = Path(settings.BASE_DIR) / settings.COMFY_INPUT_DIR

            comfy_input_dir.mkdir(parents=True, exist_ok=True)
            target_image_name = f"{sid}_{src_path.name}"
            try:
                shutil.copy2(src_path, comfy_input_dir / target_image_name)
            except Exception as e:
                log_fn(f"   ⚠️ Image copy failed: {e}")

    # ---------------------------------------------------------
    # 2. 워크플로우 수정 (중간 저장 끄기 & 입력 주입)
    # ---------------------------------------------------------
    graph = json.loads(json.dumps(workflow_template))

    final_node_id = "204"

    # [중요] 중간 저장 노드 비활성화 (204번 제외)
    for nid, node in graph.items():
        if nid != final_node_id and isinstance(node, dict) and "inputs" in node:
            inputs = node.get("inputs", {})
            if "save_output" in inputs:
                inputs["save_output"] = False
            if "filename_prefix" in inputs:
                inputs["filename_prefix"] = "TEMP_IGNORE_"

    # (A) 공통 입력 설정
    if "136" in graph and "inputs" in graph["136"]:
        graph["136"]["inputs"]["width"] = ui_width
        graph["136"]["inputs"]["height"] = ui_height

    if "97" in graph and target_image_name and "inputs" in graph["97"]:
        graph["97"]["inputs"]["image"] = target_image_name

    # Negative prompt 주입
    neg_text = scene_data.get("prompt_negative", "") or ""
    for nid in ("193:182", "181:182", "203:182"):
        if nid in graph and "inputs" in graph[nid]:
            graph[nid]["inputs"]["text"] = neg_text

    # overlap 값도 워크플로우에 주입(템플릿은 5) :contentReference[oaicite:4]{index=4}
    if seg_count >= 2 and "181:168" in graph and "inputs" in graph["181:168"]:
        graph["181:168"]["inputs"]["overlap"] = overlap
    if seg_count >= 3 and "203:168" in graph and "inputs" in graph["203:168"]:
        graph["203:168"]["inputs"]["overlap"] = overlap

    # (B) 세그먼트별 설정: prompt_n + length_n
    base_prompt = scene_data.get("prompt_1") or scene_data.get("prompt") or ""
    if "193:160" in graph and "inputs" in graph["193:160"]:
        graph["193:160"]["inputs"]["length"] = int(lens[0])
    if "193:152" in graph and "inputs" in graph["193:152"]:
        graph["193:152"]["inputs"]["text"] = base_prompt
    if "189" in graph and "inputs" in graph["189"]:
        graph["189"]["inputs"]["noise_seed"] = random.randint(1, 10 ** 14)

    if seg_count >= 2:
        p2 = scene_data.get("prompt_2") or base_prompt
        if "181:160" in graph and "inputs" in graph["181:160"]:
            graph["181:160"]["inputs"]["length"] = int(lens[1])
        if "181:152" in graph and "inputs" in graph["181:152"]:
            graph["181:152"]["inputs"]["text"] = p2
        if "182" in graph and "inputs" in graph["182"]:
            graph["182"]["inputs"]["noise_seed"] = random.randint(1, 10 ** 14)

    if seg_count >= 3:
        p3 = scene_data.get("prompt_3") or base_prompt
        if "203:160" in graph and "inputs" in graph["203:160"]:
            graph["203:160"]["inputs"]["length"] = int(lens[2])
        if "203:152" in graph and "inputs" in graph["203:152"]:
            graph["203:152"]["inputs"]["text"] = p3
        if "199" in graph and "inputs" in graph["199"]:
            graph["199"]["inputs"]["noise_seed"] = random.randint(1, 10 ** 14)

    # (C) 최종 연결 (Rewiring)
    # 템플릿에서 최종 204는 "203:168"의 2번 출력 포트를 입력으로 사용 :contentReference[oaicite:5]{index=5}
    # 2세그일 때는 "181:168"의 '배치 출력' 포트를 넣어야 seg2 경로가 pruning되지 않음.
    if final_node_id in graph and "inputs" in graph[final_node_id]:
        if seg_count == 1:
            graph[final_node_id]["inputs"]["images"] = ["193:162", 0]
        elif seg_count == 2:
            # 핵심 수정: 포트 인덱스를 2로 맞춤(템플릿 계열)
            graph[final_node_id]["inputs"]["images"] = ["181:168", 2]
        else:
            # 3세그는 템플릿 기본(203:168, 2) 그대로 사용
            graph[final_node_id]["inputs"]["images"] = ["203:168", 2]

        graph[final_node_id]["inputs"]["filename_prefix"] = f"LONG_{sid}"
        # 저장은 최종만 True로(템플릿상 save_output False로 되어있을 수 있어 강제)
        if "save_output" in graph[final_node_id]["inputs"]:
            graph[final_node_id]["inputs"]["save_output"] = True

    # ---------------------------------------------------------
    # 3. 제출 및 최종 결과물만 회수
    # ---------------------------------------------------------
    try:
        res = _submit_and_wait_comfy_func(
            comfy_url,
            graph,
            timeout=10000,
            poll=5.0,
            on_progress=lambda x: None
        )

        outputs = res.get("outputs", {}) if isinstance(res, dict) else {}
        target_out = outputs.get(final_node_id)

        if not target_out:
            # 혹시 ID가 바뀌었을 경우: 첫 VHS_VideoCombine 노드 찾기
            combine_node_id = None
            for nid, node in graph.items():
                if isinstance(node, dict) and node.get("class_type") == "VHS_VideoCombine":
                    combine_node_id = nid
                    break
            if combine_node_id:
                target_out = outputs.get(combine_node_id)

        if not target_out:
            log_fn("   ❌ 최종 병합 노드(VHS_VideoCombine)의 결과를 찾을 수 없습니다.")
            log_fn(f"      outputs keys: {list(outputs.keys())}")
            return False

        files = target_out.get("gifs", []) + target_out.get("videos", [])
        if not files:
            log_fn("   ❌ 최종 노드는 찾았으나 출력 파일 목록이 비었습니다.")
            return False

        item = files[0]
        fname = item.get("filename")
        subfolder = item.get("subfolder", "")
        ftype = item.get("type", "output")

        if not fname:
            log_fn("   ❌ 출력 파일 항목에 filename이 없습니다.")
            return False

        params = {"filename": fname, "subfolder": subfolder, "type": ftype}
        resp = requests.get(f"{comfy_url}/view", params=params, timeout=120)

        if resp.status_code != 200:
            log_fn(f"   ❌ Download Failed: {resp.status_code}")
            return False

        output_dir.mkdir(parents=True, exist_ok=True)
        target_path = output_dir / f"{sid}_raw.mp4"
        with open(target_path, "wb") as f:
            f.write(resp.content)

        log_fn(f"   ✅ Saved Final: {target_path.name} (frames={total_frames}, segs={seg_count}, lens={lens}, overlap={overlap})")
        return True

    except Exception as e:
        log_fn(f"❌ Generation Failed {sid}: {e}")
        return False


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








# 이건 남겨두자
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
        steps: int = 6,
        timeout_sec: int = 1800,
        poll_sec: float = 3,
        workflow_path: str | _Path | None = None,
        on_progress: Optional[Dict[str, Any] | Callable[[Dict[str, Any]], None]] = None,
) -> List[_Path]:
    """
    story.json(또는 video.json)을 읽어 누락된 씬 이미지를 ComfyUI로 생성한다.

    - 기본 워크플로우: QwenEdit2511-V1 (Step2용)
    - SaveImage 노드의 output_path를 동적으로 주입해 /view 404 문제를 줄인다.
    - result_data.outputs의 images 목록에서 type="output" 이미지를 우선 선택한다.
    - /view 호출 시 filename / subfolder / type 파라미터를 모두 로그로 남겨 디버깅에 활용한다.
    - scene.characters 의 "female_01:0" 같은 값을 기반으로 캐릭터 이미지를
      Load Image1~5 슬롯에 자동 매핑한다. (index+1 == 이미지 슬롯 번호)
    """
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

    # --- [4. 워크플로우 로드 (Qwen Step2 기본)] ---
    try:
        base_jsons = _Path(JSONS_DIR)
    except (ImportError, AttributeError):
        base_jsons = _Path(r"C:\my_games\shorts_make\app\jsons")

    # 기본값을 QwenEdit Step2 워크플로우로 변경
    wf_path_resolved = _Path(workflow_path) if workflow_path else (base_jsons / "QwenEdit2511-V1.json")
    if not wf_path_resolved.exists():
        raise FileNotFoundError(f"필수 워크플로 없음: {wf_path_resolved}")

    try:
        with open(wf_path_resolved, "r", encoding="utf-8") as f_graph:
            graph_origin: Dict[str, Any] = _json_loader.load(f_graph)
    except (IOError, OSError, _json_loader.JSONDecodeError) as e_load_wf:
        raise RuntimeError(f"워크플로 로드 실패: {wf_path_resolved}") from e_load_wf

    # --- [4-1. 워크플로/Comfy 헬퍼] ---
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

    # Latent / KSampler / Seed 관련 (Qwen 워크플로에 없어도 안전하게 동작)
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

    # SaveImage 노드 수집
    save_image_ids = _find_nodes(graph_origin, "SaveImage")
    if not save_image_ids:
        _notify("warn", f"[IMG] 워크플로우({wf_path_resolved.name})에 'SaveImage' 노드가 없습니다. API 출력이 실패할 수 있습니다.")

    # --- [4-2. Qwen Step2용 LoadImage 슬롯 수집] ---
    #  - "_meta.title" 에 "Load Image1", "Load Image2" ... 로 들어 있다고 가정
    load_image_slots: Dict[int, str] = {}
    for nid, node in (graph_origin or {}).items():
        try:
            if str(node.get("class_type")) != "LoadImage":
                continue
            title = str(node.get("_meta", {}).get("title") or node.get("label") or "")
            m = re.search(r"Load\s*Image\s*(\d+)", title, re.IGNORECASE)
            if not m:
                continue
            slot_idx = int(m.group(1))  # 1,2,3...
            load_image_slots[slot_idx] = str(nid)
        except Exception:
            continue

    # --- [캐릭터 이미지 경로 해석] ---
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

            # Latent 크기 주입 (Qwen 워크플로에 Latent가 없어도 안전)
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
                """
                - CLIPTextEncode / DualClipTextEncode 의 text/text_g/text_l 갱신
                - PrimitiveString(멀티라인) 프롬프트 노드(value) 갱신 (Qwen에서 사용)
                - QwenImageIntegratedKSampler 의 negative_prompt 갱신
                """
                for nid_prompt, node_prompt in gdict.items():
                    try:
                        if not isinstance(node_prompt, dict):
                            continue
                        ctype = str(node_prompt.get("class_type") or "")
                        node_inputs = node_prompt.get("inputs", {})
                        if not isinstance(node_inputs, dict):
                            node_inputs = node_prompt.setdefault("inputs", {})

                        label = str(node_prompt.get("label") or "")
                        title = str(node_prompt.get("_meta", {}).get("title") or "") + " " + label
                        lower_title = title.lower()

                        # 1) CLIP 기반 프롬프트 (예전 워크플로 호환)
                        fields_list = ("text", "text_g", "text_l")
                        if any(f_key in node_inputs for f_key in fields_list):
                            is_neg_prompt = (
                                "neg" in lower_title
                                or "negative" in lower_title
                            )
                            val_to_set = neg if is_neg_prompt else pos
                            for f_key in fields_list:
                                if f_key in node_inputs:
                                    node_inputs[f_key] = val_to_set

                        # 2) PrimitiveString / PrimitiveStringMultiline 기반 (Qwen용)
                        if "PrimitiveString" in ctype and "prompt" in lower_title:
                            # 이 노드는 양수 프롬프트용으로만 사용 (neg는 Sampler에 직접 주입)
                            node_inputs["value"] = pos

                        # 3) Qwen Sampler의 negative_prompt 직접 주입
                        if ctype == "QwenImageIntegratedKSampler":
                            node_inputs["negative_prompt"] = neg or ""
                    except (AttributeError, KeyError, TypeError) as e_apply_prompt:
                        _notify("warn", f"[IMG] 프롬프트 적용 중 경고: {e_apply_prompt}")
                        continue

            _apply_prompts(graph_clone, pos_text_raw, neg_text)

            # KSampler steps 주입 (Qwen 워크플로에 없으면 무시)
            for nid_sampler in ksampler_ids:
                _set_input(graph_clone, nid_sampler, "steps", int(steps))

            # --- 캐릭터 → Load Image 슬롯 매핑 (Step2 방식) ---
            try:
                # scene.characters 정규화 ("female_01:0" → {"id":"female_01","index":0} ...)
                sc_norm = normalize_scene_characters(sc)
                sc.update(sc_norm)
                chars_norm = sc_norm.get("characters") or []

                slot_to_path: Dict[int, _Path] = {}
                for c in chars_norm:
                    cid = str(c.get("id") or "").strip()
                    if not cid:
                        continue
                    idx_val = c.get("index")
                    if not isinstance(idx_val, int) or idx_val < 0:
                        continue
                    slot_idx = idx_val + 1  # 0 → image1, 1 → image2 ...
                    if slot_idx not in load_image_slots:
                        continue
                    if slot_idx in slot_to_path:
                        continue
                    p_face = _resolve_face_image_by_name(cid)
                    if not p_face:
                        continue
                    slot_to_path[slot_idx] = p_face

                if slot_to_path:
                    comfy_dir = _Path(COMFY_INPUT_DIR)
                    comfy_dir.mkdir(parents=True, exist_ok=True)
                    for slot_idx, src_path in slot_to_path.items():
                        node_id = load_image_slots.get(slot_idx)
                        if not node_id:
                            continue
                        # 씬ID + 슬롯번호를 붙여서 파일명 충돌 최소화
                        dest_name = f"{sid}_slot{slot_idx}_{src_path.name}"
                        dest_path = comfy_dir / dest_name
                        try:
                            shutil.copy2(src_path, dest_path)
                        except Exception as e_copy:
                            _notify(
                                "warn",
                                f"[IMG] {sid} 캐릭터 이미지 복사 실패: slot={slot_idx}, src={src_path.name}, err={e_copy}",
                            )
                            continue
                        _set_input(graph_clone, node_id, "image", dest_name)
                        _notify("debug", f"[IMG] {sid} slot{slot_idx} → {dest_name} (node {node_id})")
                else:
                    # 캐릭터 매핑이 없어도 워크플로 기본값(9.jpg 등)으로 동작
                    _notify("debug", f"[IMG] {sid} 캐릭터 이미지 매핑 없음 → 워크플로 기본 이미지 사용")
            except Exception as e_char:
                _notify("warn", f"[IMG] {sid} 캐릭터 슬롯 매핑 중 오류: {e_char}")

            # --- SaveImage output_path 주입 (있는 경우에만) ---
            relative_output_path = f"t2i_output/{sid}"
            for save_node_id in save_image_ids:
                _set_input(graph_clone, save_node_id, "output_path", relative_output_path)
                _notify("debug", f"[IMG] SaveImage 노드({save_node_id})에 output_path='{relative_output_path}' 주입")

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


# ─────────────────────────────────────────────────────────────────────────────
# 1. Step 1: Z-Image (베이스 생성)
# ─────────────────────────────────────────────────────────────────────────────
def build_step1_zimage_base(
        *,
        video_json_path: str | Path,
        source_json_path: str | Path,
        workflow_path: str | Path | None = None,
        ui_width: int = 720,
        ui_height: int = 1280,
        steps: int = 28,
        skip_if_exists: bool = True,
        timeout_sec: int = 1800,
        poll_sec: float = 1.0,
        pos_keys: List[str] | None = None,
        neg_keys: List[str] | None = None,
        reactor_disable_all_by_default: bool = True,
        reactor_enable_node_ids: List[str] | None = None,
        fallback_positive_clip_node_id: str | None = "6",
        fallback_negative_clip_node_id: str | None = "217",
        out_prefix: str = "temp_",
        out_ext: str = ".png",
        on_progress: Optional[Dict[str, Any] | Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Path]:
    # 1. 경로 및 설정 로드
    p_video = Path(str(video_json_path)).resolve()
    if not p_video.exists(): raise FileNotFoundError(f"video.json 없음: {p_video}")

    video_doc = load_json(p_video, {}) or {}
    scenes = video_doc.get("scenes", []) or []

    p_src = Path(str(source_json_path)).resolve()
    if not p_src.exists():
        src_doc = {}
        src_scenes = []
    else:
        src_doc = load_json(p_src, {}) or {}
        src_scenes = src_doc.get("scenes", []) or []

    # 소스 매핑
    src_map = {}
    for s in src_scenes:
        rid = str(s.get("id", "")).strip()
        if rid:
            src_map[rid] = s
            if rid.startswith("t_"):
                src_map[rid.replace("t_", "")] = s
                x = rid.replace("t_", "")
                if x.isdigit(): src_map[str(int(x))] = s
            if rid.isdigit():
                src_map[str(int(rid))] = s
                src_map[f"t_{int(rid):03d}"] = s

    # 이미지 저장 경로
    paths_v = video_doc.get("paths") or {}
    root_dir = Path(str(paths_v.get("root") or p_video.parent))
    imgs_dir = ensure_dir(root_dir / str(paths_v.get("imgs_dir") or "imgs"))

    # 워크플로우 로드
    wf = Path(str(workflow_path)) if workflow_path else (Path(JSONS_DIR) / "Z-Image-lora.json")
    if not wf.exists(): raise FileNotFoundError(f"Step1 WF 없음: {wf}")

    # ★ 로그 출력
    if on_progress:
        msg = f"▶ [Step1] 워크플로우: {wf.name}"
        if callable(on_progress):
            on_progress({"stage": "debug", "msg": msg})
        elif isinstance(on_progress, dict) and callable(on_progress.get("callback")):
            on_progress["callback"]({"stage": "debug", "msg": msg})

    graph_origin = json.loads(wf.read_text(encoding="utf-8"))

    # ComfyUI URL 처리
    base_url = str(COMFY_HOST).strip()
    if not base_url.startswith("http"): base_url = f"http://{base_url}"
    base_url = base_url.rstrip("/")

    if pos_keys is None: pos_keys = ["prompt_img_1", "prompt_img", "prompt"]
    if neg_keys is None: neg_keys = ["prompt_negative", "prompt_img_neg", "prompt_neg"]

    created = {}
    enable_set = set(reactor_enable_node_ids or [])

    # 씬 처리 루프
    for sc in scenes:
        sid = str(sc.get("id", "")).strip()
        if not sid: continue

        # 데이터 병합
        src_sc = src_map.get(sid)
        if not src_sc and sid.startswith("t_"):
            src_sc = src_map.get(sid.replace("t_", ""))
            if not src_sc and sid.replace("t_", "").isdigit():
                src_sc = src_map.get(str(int(sid.replace("t_", ""))))

        final_sc = sc.copy()
        if src_sc: final_sc.update(src_sc)

        # 프롬프트 추출
        pos = ""
        for k in pos_keys:
            v = final_sc.get(k)
            if isinstance(v, str) and v.strip():
                pos = v.strip()
                break

        neg = ""
        for k in neg_keys:
            v = final_sc.get(k)
            if isinstance(v, str) and v.strip():
                neg = v.strip()
                break

        if not pos: continue

        out_path = imgs_dir / f"{out_prefix}{sid}{out_ext}"
        if skip_if_exists and out_path.exists() and out_path.stat().st_size > 0:
            created[sid] = out_path
            continue

        if on_progress:
            msg = f"[Step1] {sid} 생성 중... (Prompt: {pos[:30]}...)"
            if callable(on_progress):
                on_progress({"stage": "step1_scene", "msg": msg})
            elif isinstance(on_progress, dict) and callable(on_progress.get("callback")):
                on_progress["callback"]({"stage": "step1_scene", "msg": msg})

        # 그래프 복사 및 값 주입
        graph = json.loads(json.dumps(graph_origin))

        for nid, node in graph.items():
            inputs = node.get("inputs", {})
            cls = str(node.get("class_type", ""))

            # 해상도
            if "width" in inputs and "height" in inputs:
                inputs["width"] = int(ui_width)
                inputs["height"] = int(ui_height)

            # 스텝
            if cls == "KSampler" and "steps" in inputs:
                inputs["steps"] = int(steps)

            # 파일명 Prefix
            if cls == "PreviewImage":
                node["class_type"] = "SaveImage"
                node.setdefault("inputs", {})["filename_prefix"] = f"Z_Base_{sid}"

            # Seed 랜덤화
            if cls == "KSampler" and "seed" in inputs:
                inputs["seed"] = random.randint(1, 2 ** 31)

            # ReActor 설정
            if cls == "ReActorFaceSwap":
                inputs["enabled"] = (str(nid) in enable_set)

        # 프롬프트 주입 (Inline Logic)
        # KSampler 찾아서 연결된 CLIPTextEncode 추적
        ok_pos = False
        ok_neg = False

        ksamplers = [node for nid, node in graph.items() if node.get("class_type") == "KSampler"]
        for ks in ksamplers:
            inp = ks.get("inputs", {})

            # Positive Link
            pos_link = inp.get("positive")
            if isinstance(pos_link, list) and len(pos_link) > 0:
                pos_nid = str(pos_link[0])
                if pos_nid in graph and graph[pos_nid].get("class_type") == "CLIPTextEncode":
                    graph[pos_nid]["inputs"]["text"] = pos
                    ok_pos = True

            # Negative Link
            neg_link = inp.get("negative")
            if isinstance(neg_link, list) and len(neg_link) > 0:
                neg_nid = str(neg_link[0])
                if neg_nid in graph and graph[neg_nid].get("class_type") == "CLIPTextEncode":
                    graph[neg_nid]["inputs"]["text"] = neg
                    ok_neg = True

        # Fallback (못 찾았으면 지정된 ID 사용)
        if not ok_pos and fallback_positive_clip_node_id in graph:
            graph[fallback_positive_clip_node_id]["inputs"]["text"] = pos
        if not ok_neg and fallback_negative_clip_node_id in graph:
            graph[fallback_negative_clip_node_id]["inputs"]["text"] = neg

        # 실행
        try:
            res = _submit_and_wait_comfy_func(base_url, graph, timeout=timeout_sec, poll=poll_sec)

            # 이미지 다운로드 (Inline Logic)
            outputs = res.get("outputs", {})
            downloaded = False
            for _, out_d in outputs.items():
                if "images" in out_d:
                    for img in out_d["images"]:
                        fname = img.get("filename")
                        if fname:
                            p = {"filename": fname, "type": img.get("type", "output"),
                                 "subfolder": img.get("subfolder", "")}
                            r = requests.get(f"{base_url}/view", params=p, timeout=60)
                            if r.status_code == 200:
                                out_path.parent.mkdir(parents=True, exist_ok=True)
                                with open(out_path, "wb") as f: f.write(r.content)
                                downloaded = True
                                break
                if downloaded: break

            if downloaded:
                created[sid] = out_path
            else:
                if on_progress:
                    # 간단 알림
                    pass

        except Exception as e:
            if on_progress:
                msg = f"[{sid}] 생성 실패: {e}"
                if callable(on_progress):
                    on_progress({"stage": "err", "msg": msg})
                elif isinstance(on_progress, dict) and callable(on_progress.get("callback")):
                    on_progress["callback"]({"stage": "err", "msg": msg})
            continue

    return created


# ─────────────────────────────────────────────────────────────────────────────
# 2. Step 2: Qwen Composite (합성)
# ─────────────────────────────────────────────────────────────────────────────
def build_step2_qwen_composite(
        *,
        # 기본(레거시) 모드: video.json 전체를 순회하면서 합성
        video_json_path: str | Path | None = None,
        source_json_path: str | Path | None = None,
        workflow_path: str | Path | None = None,
        base_prefix: str = "temp_",
        base_ext: str = ".png",
        base_from_key: str | None = None,
        product_image_path: str | Path | None = None,
        node_id_base_image: str = "9",
        node_id_product_image: str = "32",
        node_id_prompt_value: str = "88",
        node_id_sampler: str = "107",
        sampler_steps_key: str = "steps",
        sampler_seed_key: str = "seed",
        edit_keys: list | None = None,
        ui_width: int = 720,
        ui_height: int = 1280,
        steps: int = 28,
        skip_if_exists: bool = True,
        timeout_sec: int = 1800,
        poll_sec: float = 1.0,
        on_progress: object = None,
        # 신규 모드: shorts / shopping UI에서 단일 씬 기준으로 호출
        project_dir_or_file: str | Path | None = None,
        scene: Optional[Dict[str, Any]] = None,
        slot_images: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
        target_scene_ids: Optional[List[str]] = None,
) -> Optional[Path]:
    """Qwen Composite(이미지 합성) 공통 실행 함수.

    사용 패턴:
      - video_json_path만 넘기면 video.json의 모든 씬을 순회하며 합성.
      - project_dir_or_file + scene + slot_images를 넘기면 해당 씬 1개만 처리.
        slot_images는 [image1, image2, image3, image4, image5] 형식.

    정책:
      - 최종 해상도는 video.json의 defaults.image.width/height 를 최우선으로 사용
      - 워크플로우에서 리사이즈 처리한다면, Python에서 결과 PNG 리사이즈는 하지 않는다.
    """

    # ------------------------------------------------------------------
    # 0) 옵션 병합 (UI에서 넘어온 옵션 우선 적용) - 단, video.json defaults가 있으면 덮어씀
    # ------------------------------------------------------------------
    if options and isinstance(options, dict):
        try:
            ui_width = int(options.get("width", ui_width))
        except Exception:
            pass
        try:
            ui_height = int(options.get("height", ui_height))
        except Exception:
            pass
        try:
            steps = int(options.get("steps", steps))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 1) video.json 경로 확정
    # ------------------------------------------------------------------
    p_video: Path | None = None

    if project_dir_or_file is not None:
        p = Path(str(project_dir_or_file)).resolve()
        if p.is_dir():
            p_video = p / "video.json"
        else:
            p_video = p
    elif video_json_path is not None:
        p_video = Path(str(video_json_path)).resolve()

    if p_video is None or not p_video.exists():
        raise FileNotFoundError(f"video.json 없음: {p_video}")

    # ------------------------------------------------------------------
    # 2) video.json / 소스 JSON 로드
    # ------------------------------------------------------------------
    video_doc = load_json(p_video, {}) or {}
    scenes = video_doc.get("scenes", []) or []

    # ✅ video.json.defaults.image 기준으로 해상도 가져오기 (최우선)
    defaults_img = (video_doc.get("defaults") or {}).get("image") or {}
    try:
        w_val = defaults_img.get("width")
        h_val = defaults_img.get("height")
        if w_val is not None:
            w_int = int(w_val)
            if w_int > 0:
                ui_width = w_int
        if h_val is not None:
            h_int = int(h_val)
            if h_int > 0:
                ui_height = h_int
    except Exception:
        pass

    ui_width = int(ui_width)
    ui_height = int(ui_height)

    # 처리 대상 scene id 집합 계산
    target_ids: Optional[set[str]] = None
    if scene and scene.get("id"):
        target_ids = {str(scene.get("id")).strip()}
    elif target_scene_ids:
        target_ids = {str(sid).strip() for sid in target_scene_ids if str(sid).strip()}

    # 소스 데이터(프롬프트 등) 로드
    src_map: Dict[str, Dict[str, Any]] = {}
    if source_json_path:
        p_src = Path(str(source_json_path)).resolve()
        if p_src.exists():
            src_doc = load_json(p_src, {}) or {}
            for s in src_doc.get("scenes", []) or []:
                rid = str(s.get("id", "")).strip()
                if rid:
                    src_map[rid] = s

    # ------------------------------------------------------------------
    # 3) 워크플로우 / Comfy 설정
    # ------------------------------------------------------------------
    wf = Path(str(workflow_path)) if workflow_path else (Path(JSONS_DIR) / "QwenEdit2511-V1.json")
    if not wf.exists():
        raise FileNotFoundError(f"Step2 WF 없음: {wf}")

    if on_progress:
        msg = f"[Step2] 워크플로우: {wf.name} / size={ui_width}x{ui_height} steps={steps}"
        try:
            if callable(on_progress):
                on_progress({"stage": "debug", "msg": msg})
            elif isinstance(on_progress, dict) and callable(on_progress.get("callback")):
                on_progress["callback"]({"stage": "debug", "msg": msg})
        except Exception:
            pass

    graph_origin = json.loads(wf.read_text(encoding="utf-8"))

    base_url = str(COMFY_HOST).strip()
    if not base_url.startswith("http"):
        base_url = f"http://{base_url}"
    base_url = base_url.rstrip("/")

    comfy_input_dir = ensure_dir(Path(COMFY_INPUT_DIR))

    # blank 이미지 준비 (비활성 슬롯용)
    blank_path = comfy_input_dir / "blank.png"
    if not blank_path.exists():
        try:
            Image.new("RGB", (512, 512), (0, 0, 0)).save(blank_path)
        except Exception:
            pass

    # 제품 이미지(레거시 모드 기본값)
    prod_input_name_global: Optional[str] = None
    if product_image_path:
        p_prod = Path(str(product_image_path)).resolve()
        if p_prod.exists():
            prod_input_name_global = f"prod_{uuid.uuid4().hex[:6]}.png"
            shutil.copy2(str(p_prod), str(comfy_input_dir / prod_input_name_global))

    if edit_keys is None:
        edit_keys = ["prompt_img_2", "prompt_edit", "prompt"]

    # image1~5 에 해당하는 LoadImage 슬롯 탐색 (base/product 제외한 나머지)
    char_loader_ids: List[int] = []
    reactor_ids: List[int] = []
    ipadapter_ids: List[int] = []

    for nid, node in graph_origin.items():
        cls = str(node.get("class_type", ""))

        if "LoadImage" in cls:
            if str(nid) not in [str(node_id_base_image), str(node_id_product_image)]:
                char_loader_ids.append(int(nid))

        if "ReActor" in cls or "FaceSwap" in cls:
            reactor_ids.append(int(nid))

        if "IPAdapter" in cls:
            ipadapter_ids.append(int(nid))

    char_loader_ids.sort()
    reactor_ids.sort()
    ipadapter_ids.sort()

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

    def _disable_ref_nodes(graph: Dict[str, Any]) -> None:
        """
        참조 슬롯(image3~ 등)이 비어있을 때:
        - IPAdapter/ReActor/FaceSwap 계열을 비활성화 + weight 0
        """
        for nid in ipadapter_ids:
            sid_str = str(nid)
            node = graph.get(sid_str)
            if not node:
                continue
            inputs = node.setdefault("inputs", {})
            if "enabled" in inputs:
                inputs["enabled"] = False
            if "weight" in inputs:
                try:
                    inputs["weight"] = 0.0
                except Exception:
                    pass

        for nid in reactor_ids:
            sid_str = str(nid)
            node = graph.get(sid_str)
            if not node:
                continue
            inputs = node.setdefault("inputs", {})
            if "enabled" in inputs:
                inputs["enabled"] = False

    def _apply_size_to_inputs(inputs: Dict[str, Any], w: int, h: int) -> None:
        """
        워크플로우마다 키가 다른 경우가 있어서 폭넓게 커버한다.
        - (width,height) 쌍
        - (image_width,image_height) 쌍
        - (target_width,target_height) 쌍
        - (resize_width,resize_height) 쌍
        """
        pairs = [
            ("width", "height"),
            ("image_width", "image_height"),
            ("target_width", "target_height"),
            ("resize_width", "resize_height"),
        ]
        for kw, kh in pairs:
            if kw in inputs and kh in inputs:
                try:
                    inputs[kw] = int(w)
                    inputs[kh] = int(h)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # 4) scene 루프
    # ------------------------------------------------------------------
    proj_dir = p_video.parent
    paths_v = video_doc.get("paths") or {}
    root_dir = Path(str(paths_v.get("root") or proj_dir))
    imgs_dir = ensure_dir(root_dir / str(paths_v.get("imgs_dir") or "imgs"))

    created_single: Optional[Path] = None

    for sc in scenes:
        sid = str(sc.get("id", "")).strip()
        if not sid:
            continue
        if target_ids is not None and sid not in target_ids:
            continue

        # 소스 병합
        src_sc = src_map.get(sid, sc)

        # 1) 프롬프트 결정
        p_edit = ""
        for k in edit_keys:
            val = src_sc.get(k)
            if isinstance(val, str) and val.strip():
                p_edit = val.strip()
                break
        if not p_edit:
            p_edit = src_sc.get("prompt_img") or src_sc.get("prompt") or ""

        # 2) 슬롯 이미지 결정 (UI에서 직접 넘겨준 경우)
        current_slots: Optional[List[str]] = None
        if target_ids is not None and slot_images is not None:
            current_slots = list(slot_images)

        # 기본값: 모두 blank
        base_input_name = "blank.png"
        prod_input_name = prod_input_name_global

        # --- (A) slot_images 기반 모드 (shorts / shopping 통합 규칙 - base/product 우선 세팅) ---
        if current_slots:
            # slot 0 → base
            if len(current_slots) > 0 and current_slots[0]:
                p0 = Path(str(current_slots[0])).resolve()
                if p0.exists():
                    n0 = f"slot0_{sid}_{uuid.uuid4().hex[:6]}{p0.suffix.lower()}"
                    shutil.copy2(str(p0), str(comfy_input_dir / n0))
                    base_input_name = n0

            # slot 1 → product (※ characters 기반 브랜치에서 다시 덮어쓸 수 있음)
            if len(current_slots) > 1 and current_slots[1]:
                p1 = Path(str(current_slots[1])).resolve()
                if p1.exists():
                    n1 = f"slot1_{sid}_{uuid.uuid4().hex[:6]}{p1.suffix.lower()}"
                    shutil.copy2(str(p1), str(comfy_input_dir / n1))
                    prod_input_name = n1
            elif prod_input_name is None:
                prod_input_name = "blank.png"

            extra_slots = current_slots[2:]

        # --- (B) 레거시/자동 모드 ---
        else:
            extra_slots = []

            # 베이스 이미지: base_from_key 우선, 없으면 temp_{sid}.png 사용
            base_file: Optional[Path] = None
            if base_from_key:
                v = src_sc.get(base_from_key)
                if v:
                    cand = (proj_dir / str(v)).resolve()
                    if cand.exists():
                        base_file = cand

            if base_file is None:
                base_file = imgs_dir / f"{base_prefix}{sid}{base_ext}"

            if base_file.exists():
                base_input_name = f"base_{sid}_{uuid.uuid4().hex[:6]}{base_file.suffix.lower()}"
                shutil.copy2(str(base_file), str(comfy_input_dir / base_input_name))

            # 제품 이미지는 전역 prod_input_name_global 그대로 사용
            prod_input_name = prod_input_name_global or "blank.png"

        # 결과 파일 경로
        final_file = imgs_dir / f"{sid}.png"
        if skip_if_exists and final_file.exists() and final_file.stat().st_size > 0:
            sc["img_file"] = str(final_file)
            if target_ids is not None:
                created_single = final_file
            continue

        if on_progress:
            msg = f"[Step2] {sid} 합성 중... (Prompt: {p_edit[:60]}...)"
            try:
                if callable(on_progress):
                    on_progress({"stage": "step2_scene", "msg": msg})
                elif isinstance(on_progress, dict) and callable(on_progress.get("callback")):
                    on_progress["callback"]({"stage": "step2_scene", "msg": msg})
            except Exception:
                pass

        # ------------------------------------------------------------------
        # 5) 그래프 복사 및 노드 주입
        # ------------------------------------------------------------------
        graph = json.loads(json.dumps(graph_origin))

        # image1, image2 (Base / Product)
        if node_id_base_image in graph:
            graph[node_id_base_image].setdefault("inputs", {})["image"] = base_input_name

        if node_id_product_image in graph:
            val = prod_input_name if prod_input_name else "blank.png"
            graph[node_id_product_image].setdefault("inputs", {})["image"] = val

        # slot 3~ (char_loader_ids) + characters 기반 이미지 매핑
        scene_chars = src_sc.get("characters", []) or []

        # 기본: 모든 char_loader 입력은 우선 blank 로 초기화
        for lid in char_loader_ids:
            lid_str = str(lid)
            if lid_str in graph:
                graph[lid_str].setdefault("inputs", {})["image"] = "blank.png"

        # --------------------------------------------------------------
        # 4-1) characters 와 slot_images 가 모두 있는 경우
        #      → slot_images 를 "캐릭터 이미지"로 보고 characters 개수만큼만 사용
        # --------------------------------------------------------------
        if scene_chars and current_slots:
            resolved_char_files: List[str] = []

            # (A) slot_images 기반으로 캐릭터 수만큼만 복사
            for idx, _ in enumerate(scene_chars):
                if idx >= len(current_slots):
                    break
                src_path = Path(str(current_slots[idx])).resolve()
                if not src_path.exists():
                    continue
                name = f"char{idx}_{sid}_{uuid.uuid4().hex[:6]}{src_path.suffix.lower()}"
                shutil.copy2(str(src_path), str(comfy_input_dir / name))
                resolved_char_files.append(name)

            # (B) 부족한 캐릭터 이미지는 CHARACTER_DIR 기준으로 보충
            if len(resolved_char_files) < len(scene_chars):
                char_base_dir = Path(CHARACTER_DIR)
                for idx in range(len(resolved_char_files), len(scene_chars)):
                    entry = scene_chars[idx]
                    if isinstance(entry, dict):
                        cid = str(entry.get("id", "")).split(":")[0].strip()
                    else:
                        cid = str(entry).split(":")[0].strip()
                    if not cid:
                        continue
                    for ext in (".png", ".jpg", ".webp"):
                        cpath = char_base_dir / f"{cid}{ext}"
                        if cpath.exists():
                            name = f"char{idx}_{sid}_{uuid.uuid4().hex[:6]}{ext}"
                            shutil.copy2(str(cpath), str(comfy_input_dir / name))
                            resolved_char_files.append(name)
                            break

            if resolved_char_files:
                # 첫 번째 캐릭터 이미지는 base 이미지로 사용
                base_input_name = resolved_char_files[0]

                # product 이미지는 slot_images 가 아니라 product_image_path 에서만 받도록 복원
                prod_input_name = prod_input_name_global or "blank.png"

                if len(resolved_char_files) > 1 and char_loader_ids:
                    any_extra_active = False
                    for idx, loader_id in enumerate(char_loader_ids):
                        extra_idx = idx + 1  # 두 번째 캐릭터부터 char_loader 에 매핑
                        if extra_idx >= len(resolved_char_files):
                            break
                        img_val = resolved_char_files[extra_idx]
                        lid_str = str(loader_id)
                        if lid_str in graph:
                            graph[lid_str].setdefault("inputs", {})["image"] = img_val
                            any_extra_active = True

                    if not any_extra_active:
                        _emit(f"[Step2][DBG] {sid}: characters는 있으나 추가 참조 슬롯 없음 → IPAdapter/ReActor/FaceSwap 비활성화")
                        _disable_ref_nodes(graph)
                    else:
                        _emit(f"[Step2][DBG] {sid}: characters 기반 추가 참조 슬롯 활성 → 참조 노드 유지")
                else:
                    # 캐릭터가 1명뿐이면 참조 노드는 의미가 없으므로 비활성화
                    _emit(f"[Step2][DBG] {sid}: characters=1 → IPAdapter/ReActor/FaceSwap 비활성화")
                    _disable_ref_nodes(graph)
            else:
                _emit(f"[Step2][DBG] {sid}: characters는 있지만 이미지 로드 실패 → 참조 노드 비활성화")
                _disable_ref_nodes(graph)

        # --------------------------------------------------------------
        # 4-2) characters 만 있고 slot_images 는 없는 경우 (레거시 모드)
        #      → CHARACTER_DIR 기반 자동 세팅
        # --------------------------------------------------------------
        elif scene_chars:
            active_slots = set()
            slot_files: Dict[int, str] = {}

            char_base_dir = Path(CHARACTER_DIR)

            for idx, char_entry in enumerate(scene_chars):
                if idx >= len(char_loader_ids):
                    break
                if isinstance(char_entry, dict):
                    cid = str(char_entry.get("id", "")).split(":")[0].strip()
                else:
                    cid = str(char_entry).split(":")[0].strip()
                if not cid:
                    continue
                for ext in [".png", ".jpg", ".webp"]:
                    cpath = char_base_dir / f"{cid}{ext}"
                    if cpath.exists():
                        name = f"char{idx}_{sid}_{uuid.uuid4().hex[:6]}{ext}"
                        shutil.copy2(str(cpath), str(comfy_input_dir / name))
                        active_slots.add(idx)
                        slot_files[idx] = name
                        break

            if not active_slots:
                for nid, node in graph.items():
                    cls = node.get("class_type", "")
                    if "IPAdapter" in cls or "ReActor" in cls or "FaceSwap" in cls:
                        inputs = node.setdefault("inputs", {})
                        inputs["enabled"] = False
                        if "weight" in inputs:
                            inputs["weight"] = 0.0
                for lid in char_loader_ids:
                    lid_str = str(lid)
                    if lid_str in graph:
                        graph[lid_str].setdefault("inputs", {})["image"] = "blank.png"
            else:
                for i, lid in enumerate(char_loader_ids):
                    lid_str = str(lid)
                    if lid_str not in graph:
                        continue
                    img_val = slot_files.get(i, "blank.png") if i in active_slots else "blank.png"
                    graph[lid_str].setdefault("inputs", {})["image"] = img_val

        # --------------------------------------------------------------
        # 4-3) characters 는 없고 slot_images 만 있는 경우
        #      → 예전 규칙(0: base, 1: product, 2~: 참조 이미지) 유지
        # --------------------------------------------------------------
        elif current_slots:
            any_extra_active = False

            for idx, loader_id in enumerate(char_loader_ids):
                img_val = "blank.png"
                slot_idx = idx + 2  # slot_images 기준 index
                if slot_idx < len(current_slots) and current_slots[slot_idx]:
                    p_img = Path(str(current_slots[slot_idx])).resolve()
                    if p_img.exists():
                        name = f"slot{slot_idx}_{sid}_{uuid.uuid4().hex[:6]}.png"
                        shutil.copy2(str(p_img), str(comfy_input_dir / name))
                        img_val = name
                        any_extra_active = True

                lid_str = str(loader_id)
                if lid_str in graph:
                    graph[lid_str].setdefault("inputs", {})["image"] = img_val

            if not any_extra_active:
                _emit(f"[Step2][DBG] {sid}: slot2~5 비어있음 → IPAdapter/ReActor/FaceSwap 비활성화")
                _disable_ref_nodes(graph)
            else:
                _emit(f"[Step2][DBG] {sid}: slot2~5 활성 이미지 있음 → 참조 노드 유지(활성)")

        # --------------------------------------------------------------
        # 4-4) characters 도 없고 slot_images 도 없는 경우
        #      → 모든 참조 노드를 끄고 blank만 사용
        # --------------------------------------------------------------
        else:
            _emit(f"[Step2][DBG] {sid}: characters/slot_images 모두 없음 → 참조 노드 비활성화")
            _disable_ref_nodes(graph)

        # prompt/value
        if node_id_prompt_value in graph:
            graph[node_id_prompt_value].setdefault("inputs", {})["value"] = p_edit

        # sampler/size 설정 + PreviewImage -> SaveImage
        for nid, node in graph.items():
            inputs = node.get("inputs", {})
            if isinstance(inputs, dict):
                # ✅ 워크플로우에서 리사이즈 처리할 수 있게 폭넓게 size 주입
                _apply_size_to_inputs(inputs, ui_width, ui_height)

            if str(nid) == node_id_sampler and isinstance(inputs, dict) and sampler_steps_key in inputs:
                try:
                    inputs[sampler_steps_key] = int(steps)
                except Exception:
                    pass
                try:
                    inputs[sampler_seed_key] = random.randint(1, 2 ** 31)
                except Exception:
                    pass

            if node.get("class_type") == "PreviewImage":
                node["class_type"] = "SaveImage"
                node.setdefault("inputs", {})["filename_prefix"] = f"Step2_{sid}"

        # ------------------------------------------------------------------
        # 6) Comfy 실행 및 결과 다운로드
        # ------------------------------------------------------------------
        try:
            res = _submit_and_wait_comfy_func(base_url, graph, timeout=timeout_sec, poll=poll_sec)

            outputs = res.get("outputs", {})
            downloaded = False

            for _, out_d in outputs.items():
                if "images" in out_d:
                    for img in out_d["images"]:
                        fname = img.get("filename")
                        if not fname:
                            continue
                        params = {
                            "filename": fname,
                            "type": img.get("type", "output"),
                            "subfolder": img.get("subfolder", ""),
                        }
                        r = requests.get(f"{base_url}/view", params=params, timeout=60)
                        if r.status_code == 200:
                            final_file.parent.mkdir(parents=True, exist_ok=True)
                            with open(final_file, "wb") as f:
                                f.write(r.content)
                            downloaded = True
                            break
                if downloaded:
                    break

            if not downloaded:
                raise RuntimeError("Comfy에서 이미지를 다운로드하지 못했습니다.")

            # ✅ 워크플로우에서 리사이즈가 끝난 결과를 그대로 저장한다 (Python 리사이즈 금지)
            sc["img_file"] = str(final_file)
            if target_ids is not None:
                created_single = final_file

        except Exception as e:
            if on_progress:
                msg = f"[{sid}] 합성 실패: {e}"
                try:
                    if callable(on_progress):
                        on_progress({"stage": "err", "msg": msg})
                    elif isinstance(on_progress, dict) and callable(on_progress.get("callback")):
                        on_progress["callback"]({"stage": "err", "msg": msg})
                except Exception:
                    pass
            continue

    save_json(p_video, video_doc)
    return created_single







#================페이스 스왑 관련=============##================페이스 스왑 관련=============##================페이스 스왑 관련=============#
#================페이스 스왑 관련=============##================페이스 스왑 관련=============##================페이스 스왑 관련=============#
#================페이스 스왑 관련=============##================페이스 스왑 관련=============##================페이스 스왑 관련=============#

# 일딴 보류
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

# 일딴 보류
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

# 일딴 보류
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


#================페이스 스왑 관련=============##================페이스 스왑 관련=============##================페이스 스왑 관련=============#
#================페이스 스왑 관련=============##================페이스 스왑 관련=============##================페이스 스왑 관련=============#
#================페이스 스왑 관련=============##================페이스 스왑 관련=============##================페이스 스왑 관련=============#


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


