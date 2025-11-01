﻿# -*- coding: utf-8 -*-
# i2v 분할/실행/합치기(교차 페이드) + 누락 이미지 생성 (ComfyUI 연동)
from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any
import re
from pathlib import Path
from pathlib import Path as _Path
import json as _json
import time, requests, shutil

try:
    from app.utils import load_json, save_json, ensure_dir
    from app.settings import COMFY_HOST, JSONS_DIR
except Exception:
    from utils import load_json, save_json, ensure_dir  # type: ignore
    from settings import COMFY_HOST, JSONS_DIR          # type: ignore

# ── 유연 임포트 ─────────────────────────────────────────────────────────────
try:
    from app.utils import ensure_dir, load_json
    from app.settings import BASE_DIR, I2V_WORKFLOW, FFMPEG_EXE, USE_HWACCEL, FINAL_OUT, COMFY_HOST, DEFAULT_HOST_CANDIDATES
except ImportError:
    from utils import ensure_dir, load_json                    # type: ignore
    from settings import BASE_DIR, I2V_WORKFLOW, FFMPEG_EXE, USE_HWACCEL, FINAL_OUT, COMFY_HOST, DEFAULT_HOST_CANDIDATES  # type: ignore

# music_gen에 있는 견고한 함수들을 우선 재사용 (가능할 때)
try:
    from app.audio_sync import (
        _submit_and_wait as _submit_and_wait_comfy,
        _http_get as _http_get_audio,
        _load_workflow_graph as _load_workflow_graph_audio,
        _find_nodes_by_class_contains as _find_nodes_by_class_contains_audio,
    )
except Exception:  # 단독 실행/상대 경로일 수도 있으니 폴백 제공
    _submit_and_wait_comfy = None
    _http_get_audio = None
    _load_workflow_graph_audio = None
    _find_nodes_by_class_contains_audio = None

# === JSON 템플릿 폴더 & 워크플로 경로 ===
try:
    from app.settings import JSONS_DIR as _JSONS_DIR  # 있으면 사용
except ImportError:
    try:
        from settings import JSONS_DIR as _JSONS_DIR
    except ImportError:
        from pathlib import Path  # CamelCase는 CamelCase로!
        _JSONS_DIR = Path(__file__).resolve().parent / "jsons"  # 폴백

JSONS_DIR = Path(_JSONS_DIR)

WF_T2I    = JSONS_DIR / "nunchaku_t2i.json"
WF_SWAP_1 = JSONS_DIR / "nunchaku-t2i_swap_1.json"
WF_SWAP_2 = JSONS_DIR / "nunchaku-t2i_swap_2.json"
WF_SWAP_3 = JSONS_DIR / "nunchaku-t2i_swap_3.json"  # 3명은 옵션

# 고정 예시 대신 폴백으로만 사용
DEFAULT_CHAR_HAIR_STYLE = {
    "female_01": "긴 웨이브 헤어, 자연 갈색, 일관성 유지",
    "male_01":   "단정한 숏컷, 진한 갈색, 일관성 유지",
}



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

# ── Comfy 제출/폴링(폴백) ────────────────────────────────────────────────────
def _submit_and_wait(base_url: str, graph: Dict[str, Any], *, timeout: float, poll: float, on_progress=None) -> Dict[str, Any]:
    # 기존 모듈에 동일 함수가 있으면 그걸 사용하세요.
    # 없을 때만 이 간이 구현을 사용합니다.
    import urllib.request
    import urllib.error

    def _log(stage: str, msg: str) -> None:
        if on_progress:
            on_progress({"stage": stage, "msg": msg})

    data_bytes = json.dumps({"prompt": graph}).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/prompt", data=data_bytes, headers={"Content-Type": "application/json"})
    _log("post", "POST /prompt")
    try:
        with urllib.request.urlopen(req) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Comfy POST 실패: {e}") from e

    # 파일 생성 폴링은 호출부에서 실제 png 경로를 감시하므로 여기선 대기만
    elapsed = 0.0
    while elapsed < float(timeout):
        time.sleep(float(poll))
        elapsed += float(poll)
        _log("wait", f"elapsed={int(elapsed)}s")
    return {}  # history placeholder

# ── 분할/겹침 ───────────────────────────────────────────────────────────────
def plan_segments(frame_length: int, base_chunk: int = 60, overlap: int = 12, pad_tail: int = 5) -> list[tuple[int, int]]:
    """
    세그먼트 분할(앞에서 세기 규칙 + 항상 +5 프레임 여유).
    - 총 프레임 ≤ base_chunk 인 경우에도 (0, frame_length + pad_tail) 로 1청크 생성
    - 총 프레임 > base_chunk:
        * 시작 인덱스:
          s0 = 0
          s1 = s0 + (base_chunk - overlap + 1)  # 예: 0 + (60 - 12 + 1) = 49
          s(k) = s(k-1) + base_chunk            # 이후 +60
        * 종료 인덱스(배타 end):
          e0 = base_chunk + pad_tail            # 60 + 5 = 65
          e(k) = e(k-1) + base_chunk            # 이후 +60
          마지막 청크만 e_last = frame_length + pad_tail 로 캡
    - 반환 튜플은 (inclusive_start, exclusive_end).
    예)
      120 → [(0, 65), (49, 125)]
      174 → [(0, 65), (49, 125), (109, 179)]
      245 → [(0, 65), (49, 125), (109, 185), (169, 245), (229, 250)]
      55  → [(0, 60)] 가 아니라 [(0, 60)]이 아님. (항상 +5) → [(0, 60)]? 55+5=60 → [(0, 60)]
    """
    total = 0 if frame_length is None else int(frame_length)
    if total <= 0:
        return []

    base = 60 if base_chunk is None else int(base_chunk)
    if base <= 0:
        base = 60

    ov = 0 if overlap is None else int(overlap)
    if ov < 0:
        ov = 0
    if ov > base:
        ov = base

    pad = 5 if pad_tail is None else int(pad_tail)
    if pad < 0:
        pad = 0

    # 총 프레임이 base 이하여도 항상 +pad 적용
    if total <= base:
        end_ex = total + pad
        return [(0, end_ex)]

    first_step = base - ov + 1
    if first_step < 1:
        first_step = 1

    segments: list[tuple[int, int]] = []
    ps_start = 0
    ps_end = base + pad
    if ps_end >= total + pad:
        ps_end = total + pad
    segments.append((ps_start, ps_end))

    next_start = ps_start + first_step
    next_end = ps_end + base

    while next_start < total:
        end_capped = next_end if next_end < (total + pad) else (total + pad)
        if end_capped > next_start:
            segments.append((next_start, end_capped))
        else:
            break
        next_start = next_start + base
        next_end = next_end + base

    return segments

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

# ── i2v 샷 생성 ────────────────────────────────────────────────────────────


def build_shots_with_i2v(
    project_dir: str,
    total_frames: int,
    *,  # 키워드 전용
    ui_width: Optional[int] = None,
    ui_height: Optional[int] = None,
    ui_fps: Optional[int] = None,
    ui_steps: Optional[int] = None,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None
) -> None:
    """
    각 씬을 i2v로 생성합니다. UI 설정을 워크플로우에 적용하고, 파일 접두사에 현재 날짜를 사용합니다.
    - 씬 레벨 시드 1회 생성 → 모든 청크 공통 사용
    - 청크 2+ : 이전 청크의 '마지막 overlap 프레임 시퀀스(예: 12f)'를 conditioning 입력으로 제공(가능 시)
      * 시퀀스 미지원 워크플로우면 기존 1프레임 PNG 폴백 유지
    - 기본 청크 크기 60f(메모리 안전)
    - 임시 청크 mp4는 항상 clips/xfade_work 에만 저장
    (기존 기능 100% 보존)
    """
    # ── 표준/타입 ─────────────────────────────────────────────
    import json as json_mod
    import shutil
    import time
    import datetime
    import random
    import subprocess
    from pathlib import Path
    from typing import Any, Dict, List, Optional, Callable, Tuple, cast

    # 네트워크 통신 모듈
    try:
        import requests as requests_mod  # type: ignore
    except ImportError:
        requests_mod = None  # type: ignore[assignment]
    if requests_mod is None:
        raise ImportError("requests 모듈이 필요합니다. 설치 후 다시 실행하세요.")

    assert isinstance(total_frames, int)

    print(f"[I2V][TOTAL] total_frames={int(total_frames)}")

    def _notify(msg: str) -> None:
        if on_progress is None:
            return
        try:
            text = str(msg or "").strip()
            if text:
                on_progress({"msg": "[I2V] " + text})
        except (TypeError, ValueError):
            print("[경고] on_progress 콜백 처리 실패")

    # ── ffprobe: 프레임 수 확인 ───────────────────────────────
    def _probe_nb_frames_ffprobe_local(ffprobe_bin_param: str, src_path: Path) -> int:
        cmd_probe = [
            ffprobe_bin_param,
            "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1",
            str(src_path),
        ]
        proc = subprocess.run(
            cmd_probe,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        if proc.returncode != 0:
            return 0
        try:
            return int((proc.stdout or "0").strip() or "0")
        except ValueError:
            return 0

    # ── 최종 트림(원래 총프레임으로) ──────────────────────────
    def _trim_to_frames(ffmpeg_bin_param: str, ffprobe_bin_param: str, src: Path, dst: Path, target_frames: int, fps_val: int) -> bool:
        if target_frames <= 0:
            return False
        try:
            tmp_out = dst.with_suffix(".tmp.mp4")
            if tmp_out.exists():
                try:
                    tmp_out.unlink()
                except OSError:
                    pass

            cmd = [
                ffmpeg_bin_param, "-y",
                "-fflags", "+genpts",
                "-i", str(src),
                "-vf", f"trim=end_frame={int(target_frames)},setpts=PTS-STARTPTS",
                "-r", f"{int(max(1, fps_val))}",
                "-fps_mode", "cfr",
                "-vsync", "cfr",
                "-c:v", "libx264", "-crf", "18",
                "-pix_fmt", "yuv420p",
                str(tmp_out),
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=False,
            )
            if proc.returncode != 0:
                _notify(f"[TRIM][오류] ffmpeg 트림 실패\n{(proc.stdout or '')[:4000]}")
                if tmp_out.exists():
                    try:
                        tmp_out.unlink()
                    except OSError:
                        pass
                return False

            got_frames = _probe_nb_frames_ffprobe_local(ffprobe_bin_param, tmp_out)
            if got_frames != int(target_frames):
                _notify(f"[TRIM][경고] 프레임 불일치: 기대 {target_frames}f vs 실제 {got_frames}f")

            try:
                if dst.exists():
                    dst.unlink()
            except OSError:
                pass
            tmp_out.rename(dst)
            _notify(f"[TRIM] 완료: {dst.name} → {target_frames}f")
            return True
        except Exception as e_trim_any:
            _notify(f"[TRIM][오류] 트림 처리 예외: {e_trim_any}")
            return False

    # ── 컷 병합(덮어쓰기) ──────────────────────────────────────
    def _concat_cut_no_fade(
        ffmpeg_bin_arg: str,
        ffprobe_bin_arg: str,
        clip_paths_arg: List[Path],
        overlap_frames_arg: int,
        fps_arg: int,
        out_path_arg: Path,
        work_dir_arg: Path
    ) -> bool:
        """
        fade 없이 덮어쓰기:
          - A, B, C... 를 순차 병합.
          - 매 단계에서 current(A')의 뒤 overlap만큼을 잘라내고 B 전체를 이어붙임.
          - 해상도/비율 통일: 1280x720, SAR=1, CFR=fps.
        """
        if not clip_paths_arg:
            return False

        def _norm(src: Path, dst: Path) -> Tuple[int, bool]:
            cmd = [
                ffmpeg_bin_arg, "-y", "-fflags", "+genpts",
                "-i", str(src),
                "-vf",
                (
                    f"fps={int(max(1, fps_arg))},"
                    "scale=w=1280:h=720:force_original_aspect_ratio=decrease,"
                    "pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p"
                ),
                "-r", f"{int(max(1, fps_arg))}",
                "-vsync", "cfr",
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                str(dst),
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=False,
            )
            if proc.returncode != 0:
                return (0, False)
            frames = _probe_nb_frames_ffprobe_local(ffprobe_bin_arg, dst)
            return (frames, frames > 0)

        def _trim_tail(src: Path, dst: Path, keep_frames: int) -> bool:
            cmd = [
                ffmpeg_bin_arg, "-y",
                "-fflags", "+genpts",
                "-i", str(src),
                "-vf", f"trim=end_frame={int(max(0, keep_frames))},setpts=PTS-STARTPTS",
                "-r", f"{int(max(1, fps_arg))}",
                "-fps_mode", "cfr",
                "-vsync", "cfr",
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                str(dst),
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=False,
            )
            return proc.returncode == 0

        def _concat_ab(a: Path, b: Path, dst: Path) -> bool:
            cmd = [
                ffmpeg_bin_arg, "-y",
                "-fflags", "+genpts",
                "-i", str(a), "-i", str(b),
                "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
                "-map", "[v]",
                "-r", f"{int(max(1, fps_arg))}",
                "-fps_mode", "cfr",
                "-vsync", "cfr",
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                str(dst),
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=False,
            )
            return proc.returncode == 0

        print(f"[XC-BEGIN] items={len(clip_paths_arg)} fps={fps_arg} overlap_frames={overlap_frames_arg} out=1280x720")

        # 첫 번째 정규화
        cur = work_dir_arg / "cur_000.mp4"
        n0 = work_dir_arg / "_norm_a.mp4"
        f0, ok0 = _norm(clip_paths_arg[0], n0)
        if not ok0:
            print("[XC-ERR] 첫 청크 정규화 실패")
            return False
        print(f"[XC-IN-000] {{'path': '{clip_paths_arg[0]}', 'frames': {f0}, 'fps': {fps_arg}}}")
        shutil.copyfile(str(n0), str(cur))

        # 나머지 반복
        step_idx = 1
        for idx in range(1, len(clip_paths_arg)):
            nb = work_dir_arg / "_norm_b.mp4"
            fb, okb = _norm(clip_paths_arg[idx], nb)
            if not okb:
                print(f"[XC-ERR] 청크 정규화 실패 idx={idx}")
                return False

            fa = _probe_nb_frames_ffprobe_local(ffprobe_bin_arg, cur)
            print(f"[XC-CUR] A(frames)={fa}, B(frames)={fb}, overlap={overlap_frames_arg}")

            keep_a = max(0, fa - max(0, overlap_frames_arg))
            a_cut = work_dir_arg / f"_a_cut_{step_idx:03d}.mp4"
            ok_cut = _trim_tail(cur, a_cut, keep_a)
            if not ok_cut:
                print(f"[XC-ERR] A 컷 실패 keep={keep_a} step={step_idx}")
                return False

            cur_next = work_dir_arg / f"cur_{step_idx:03d}.mp4"
            ok_cat = _concat_ab(a_cut, nb, cur_next)
            if not ok_cat:
                print(f"[XC-ERR] concat 실패 step={step_idx}")
                return False

            try:
                cur.unlink(missing_ok=True)
            except OSError:
                pass
            cur = cur_next
            step_idx += 1

        # 최종 산출 이동
        try:
            if out_path_arg.exists():
                out_path_arg.unlink()
        except OSError:
            pass
        shutil.copyfile(str(cur), str(out_path_arg))

        f_out = _probe_nb_frames_ffprobe_local(ffprobe_bin_arg, out_path_arg)
        print(f"[XC-FINAL] {{'path': '{out_path_arg}', 'frames': {f_out}, 'fps': {fps_arg}}}")
        print(f"[XC-DONE] out_path={out_path_arg}")
        return True

    _notify(f"영상 생성 시작: project_dir='{project_dir}'")
    _notify(f"UI 설정: W={ui_width}, H={ui_height}, FPS={ui_fps}, Steps={ui_steps}")

    # ── 경로/파일 ─────────────────────────────────────────────
    try:
        pdir = Path(project_dir).resolve(strict=True)
        if not pdir.is_dir():
            _notify(f"[오류] 프로젝트 경로가 디렉토리가 아닙니다: {pdir}")
            return
    except FileNotFoundError:
        _notify(f"[오류] 프로젝트 디렉토리를 찾을 수 없습니다: {project_dir}")
        return
    except OSError as e_os:
        _notify(f"[오류] 프로젝트 경로 접근 오류: {e_os}")
        return

    p_video = pdir / "video.json"

    # ── 유틸 ─────────────────────────────────────────────
    try:
        from app.utils import load_json as load_json_fn, ensure_dir as ensure_dir_fn  # type: ignore
    except (ImportError, ModuleNotFoundError):
        try:
            from utils import load_json as load_json_fn, ensure_dir as ensure_dir_fn  # type: ignore
        except (ImportError, ModuleNotFoundError) as e_utils:
            raise ImportError(f"필수 유틸리티(load_json, ensure_dir) 로드 실패: {e_utils}") from e_utils

    # ── 설정 ─────────────────────────────────────────────
    try:
        import settings as settings_mod  # type: ignore
        s_mod = settings_mod
    except ImportError:
        try:
            from app import settings as settings_mod_app  # type: ignore
            s_mod = settings_mod_app
        except ImportError:
            class _SettingsFallback:
                JSONS_DIR = r"C:\my_games\shorts_make\app\jsons"
                I2V_WORKFLOW = None
                COMFY_INPUT_DIR = r"C:\my_games\shorts_make\app\comfy_inputs"
                COMFY_HOST = "http://127.0.0.1:8188"
                FFMPEG_EXE = "ffmpeg"
                FFPROBE_EXE = "ffprobe"
            s_mod = _SettingsFallback()

    jsons_dir_conf = getattr(s_mod, "JSONS_DIR", r"C:\my_games\shorts_make\app\jsons")
    i2v_workflow_conf = getattr(s_mod, "I2V_WORKFLOW", None)
    comfy_input_conf = getattr(s_mod, "COMFY_INPUT_DIR", r"C:\my_games\shorts_make\app\comfy_inputs")
    comfy_host_conf = getattr(s_mod, "COMFY_HOST", "http://127.0.0.1:8188")
    ffmpeg_exe = getattr(s_mod, "FFMPEG_EXE", "ffmpeg")
    ffprobe_exe = getattr(s_mod, "FFPROBE_EXE", "ffprobe")

    base_jsons = Path(str(jsons_dir_conf))
    comfy_input_dir = Path(str(comfy_input_conf))
    comfy_input_dir.mkdir(parents=True, exist_ok=True)

    base_url = "http://127.0.0.1:8188"
    try:
        base_url = str((load_json_fn(p_video, {}) or {}).get("comfy_host") or comfy_host_conf).rstrip("/")
    except Exception:
        base_url = str(comfy_host_conf).rstrip("/")

    # ── video.json ─────────────────────────────────────────────
    try:
        video = load_json_fn(p_video, {}) or {}
        if not isinstance(video, dict):
            _notify(f"[오류] video.json 형식 오류: {p_video}")
            return
    except (OSError, ValueError) as e_load_video:
        _notify(f"[오류] video.json 로드 실패: {e_load_video}")
        return

    scenes = list(video.get("scenes") or [])
    if not scenes:
        _notify("video.json에 생성할 씬('scenes') 없음.")
        return

    clips_dir = ensure_dir_fn(pdir / "clips")
    work_dir = clips_dir / "xfade_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(work_dir.glob("_chunk_*.mp4")) + list(work_dir.glob("cur_*.mp4")) + list(work_dir.glob("tmp_*.mp4")):
        try:
            stale.unlink()
        except OSError:
            pass

    # ── 워크플로 경로 결정 ──────────────────────────────────────
    wf_path: Optional[Path] = None
    if i2v_workflow_conf:
        c1 = Path(str(i2v_workflow_conf))
        if not c1.is_absolute():
            c1 = base_jsons / c1.name
        if c1.is_file():
            wf_path = c1
    if wf_path is None:
        defaults_all = video.get("defaults") or {}
        defaults_i2v = defaults_all.get("i2v") or {}
        wf_from_video_str = str(defaults_i2v.get("workflow") or "").strip()
        if wf_from_video_str:
            c2 = Path(wf_from_video_str)
            if not c2.is_absolute():
                c2 = base_jsons / c2.name
            if c2.is_file():
                wf_path = c2
    if wf_path is None:
        c3 = base_jsons / "guff_movie.json"
        if c3.is_file():
            wf_path = c3
        else:
            raise FileNotFoundError(f"워크플로 파일을 찾을 수 없습니다: {c3}")

    _notify(f"사용할 워크플로우: {wf_path.name} (경로: {wf_path})")

    # ── 워크플로 로드 ───────────────────────────────────────────
    try:
        with open(wf_path, "r", encoding="utf-8") as f_graph:
            graph_origin: Dict[str, Any] = json_mod.load(f_graph)
    except (OSError, json_mod.JSONDecodeError) as e_load_wf:
        raise RuntimeError(f"워크플로우 로드 실패: {wf_path} ({e_load_wf})") from e_load_wf

    # ── 제출/대기 코어 ─────────────────────────────────────────
    submit_and_wait_fn: Optional[Callable[..., Dict[str, Any]]] = None
    try:
        from app.audio_sync import _submit_and_wait as submit_and_wait_fn  # type: ignore
    except Exception:
        try:
            from audio_sync import _submit_and_wait as submit_and_wait_fn  # type: ignore
        except Exception as e_wait:
            raise RuntimeError(f"_submit_and_wait 함수 로드 실패: {e_wait}") from e_wait

    # ── 그래프 입력 주입 ───────────────────────────────────────
    def _set_input_safe(graph_dict: Dict[str, Any], node_id: str, input_key: str, value: Any) -> bool:
        node_id_s = str(node_id or "")
        input_key_s = str(input_key or "")
        if not node_id_s or not input_key_s:
            _notify(f"[경고] 노드 ID({node_id_s}) 또는 키({input_key_s}) 유효하지 않음.")
            return False

        target_node = graph_dict.get(node_id_s)
        if not isinstance(target_node, dict):
            _notify(f"[경고] 노드 ID '{node_id_s}' 없음 또는 형식 오류.")
            return False

        node_inputs_map: Dict[str, Any] = target_node.setdefault("inputs", {})
        if not isinstance(node_inputs_map, dict):
            _notify(f"[경고] 노드 {node_id_s} 'inputs' 필드가 dict 아님.")
            return False

        current_value = node_inputs_map.get(input_key_s)
        must_int = isinstance(current_value, int) or input_key_s in {
            "steps", "frame_rate", "width", "height", "length", "seed", "loop_count", "crf", "noise_seed"
        }
        must_float = isinstance(current_value, float) or input_key_s in {"cfg", "strength_model"}

        try:
            if must_int:
                node_inputs_map[input_key_s] = int(value)
            elif must_float:
                node_inputs_map[input_key_s] = float(value)
            else:
                node_inputs_map[input_key_s] = value
        except (TypeError, ValueError):
            _notify(f"[경고] 노드 {node_id_s}.{input_key_s} 값({value}) 변환 실패.")
            node_inputs_map[input_key_s] = value
        return True

    # ── 진행중 메시지 릴레이 ───────────────────────────────────
    def _relay(prog: Dict[str, Any]) -> None:
        msg_val = str(prog.get("msg") or "")
        if msg_val and on_progress is not None:
            on_progress({"msg": msg_val})

    try:
        date_prefix = datetime.date.today().strftime("%Y-%m-%d")
    except Exception:
        date_prefix = "YYYY-MM-DD"

    defaults_all = video.get("defaults") or {}
    defaults_i2v = defaults_all.get("i2v") or {}
    try:
        max_frames_per_chunk = int(defaults_i2v.get("max_frames_per_chunk", 60))
        overlap_frames_for_chunk = int(defaults_i2v.get("overlap_frames_chunk", 12))
    except (TypeError, ValueError):
        max_frames_per_chunk = 60
        overlap_frames_for_chunk = 12
    _notify(f"청크 설정: 최대 {max_frames_per_chunk}f, 겹침 {overlap_frames_for_chunk}f")

    # ── 씬 루프 ─────────────────────────────────────────────────
    for s_idx, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            _notify(f"[경고] scenes[{s_idx}] 항목 형식 오류. 건너뜁니다.")
            continue

        sid = (str(scene.get("id") or f"scene_{s_idx:05d}").strip() or f"scene_{s_idx:05d}")
        clip_mp4 = clips_dir / f"{sid}.mp4"

        try:
            if clip_mp4.is_file() and clip_mp4.stat().st_size > 1024:
                size_kb = clip_mp4.stat().st_size / 1024.0
                _notify(f"skip scene={sid} → exists {clip_mp4.name} ({size_kb:.1f} KB)")
                continue
        except OSError as e_stat_clip:
            _notify(f"[경고] 스킵 확인 중 오류 (씬 {sid}): {e_stat_clip}. 생성 시도.")

        # 입력 이미지
        img_path: Optional[Path] = None
        server_image_name: str = ""
        img_declared = str(scene.get("img_file") or "").strip()
        if img_declared:
            p_img = Path(img_declared)
            if not p_img.is_absolute():
                paths_info = video.get("paths") or {}
                root_dir_str = str(paths_info.get("root") or pdir)
                imgs_dir_name = str(paths_info.get("imgs_dir") or "imgs")
                p_img = Path(root_dir_str) / imgs_dir_name / p_img.name
            try:
                if p_img.is_file() and p_img.stat().st_size > 0:
                    img_path = p_img
            except OSError:
                pass
        if img_path is None:
            paths_info2 = video.get("paths") or {}
            root_dir_str2 = str(paths_info2.get("root") or pdir)
            imgs_dir_name2 = str(paths_info2.get("imgs_dir") or "imgs")
            cand_png = Path(root_dir_str2) / imgs_dir_name2 / f"{sid}.png"
            try:
                if cand_png.is_file() and cand_png.stat().st_size > 0:
                    img_path = cand_png
            except OSError:
                pass
        if img_path is None:
            _notify(f"scene={sid} 입력 이미지를 찾을 수 없음 → 건너뜁니다.")
            continue

        try:
            server_image_name = img_path.name
            dest = comfy_input_dir / server_image_name
            do_copy = True
            if dest.exists():
                try:
                    if dest.stat().st_mtime >= img_path.stat().st_mtime:
                        do_copy = False
                except OSError:
                    do_copy = True
            if do_copy:
                shutil.copy2(str(img_path), str(dest))
                _notify(f"이미지 복사 완료: {server_image_name} -> {comfy_input_dir.name}")
        except OSError as e_copy:
            _notify(f"[경고] 이미지 준비 실패({img_path.name}): {e_copy}")
        if not server_image_name:
            _notify(f"[오류] 이미지 이름 설정 불가 (씬 {sid}). 다음 씬으로.")
            continue

        try:
            scene_seed_value = random.randint(0, 999_999_999_999_999)
        except ValueError:
            scene_seed_value = int(time.time() * 1_000_000) % 999_999_999_999_999
        _notify(f"씬 {sid}에 사용할 마스터 시드: {scene_seed_value}")

        combine_node_id = "21"
        i2v_node_id = "25"
        target_fps_val: int = 16

        try:
            scene_duration = float(scene.get("duration", 1.0))
        except (TypeError, ValueError):
            scene_duration = 1.0

        if ui_fps is not None and ui_fps > 0:
            target_fps_val = int(ui_fps)
        else:
            fps_wf = None
            try:
                fps_wf = graph_origin.get(combine_node_id, {}).get("inputs", {}).get("frame_rate")
            except AttributeError:
                fps_wf = None
            if fps_wf is not None:
                try:
                    fps_int = int(fps_wf)
                    if fps_int > 0:
                        target_fps_val = fps_int
                except (TypeError, ValueError):
                    pass

        try:
            frame_length = max(1, int(round(scene_duration * target_fps_val)))
        except (TypeError, ValueError):
            frame_length = 16
            _notify(f"[경고] 프레임 길이 계산 오류 (씬 {sid}). 기본값 16f 사용.")

        print(f"[I2V][SCENE] id={sid} fps={target_fps_val} frame_length={frame_length}")

        # ── 분할(항상 plan_segments 사용, +5 패딩 규칙 내장) ──
        segments_list: List[Tuple[int, int]] = plan_segments(  # type: ignore[name-defined]
            frame_length,
            base_chunk=max_frames_per_chunk,
            overlap=overlap_frames_for_chunk
        )

        print(f"[I2V][SCENE] id={sid} segments={segments_list} overlap={overlap_frames_for_chunk}f")

        chunk_paths: List[Path] = []
        scene_failed = False
        current_input_image_name: str = server_image_name
        temp_frames: List[Path] = []

        # 시퀀스 conditioning 지원 여부 탐지
        has_sequence_loader = False
        sequence_loader_node_id = ""
        for node_key, node_val in graph_origin.items():
            try:
                if not isinstance(node_val, dict):
                    continue
                ctype = str(node_val.get("class_type") or "")
                inputs_dict = node_val.get("inputs") or {}
                if not isinstance(inputs_dict, dict):
                    continue
                if ctype.lower().find("load") >= 0 and (
                    "directory" in inputs_dict or "frame_rate" in inputs_dict or "pattern" in inputs_dict
                ):
                    sequence_loader_node_id = str(node_key)
                    has_sequence_loader = True
                    break
            except (AttributeError, TypeError):
                continue

        # ── 각 청크 처리 ────────────────────────────────────────
        for c_idx, (start_f, end_f) in enumerate(segments_list):
            if scene_failed:
                break

            chunk_len = max(0, int(end_f) - int(start_f))
            if chunk_len <= 0:
                _notify(f"[경고] 유효하지 않은 청크 길이 (씬 {sid}, idx={c_idx}): {start_f}~{end_f}")
                continue

            chunk_label = f"씬 {sid} (청크 {c_idx + 1}/{len(segments_list)})"
            _notify(f"{chunk_label}: 프레임 {start_f}-{end_f} (길이 {chunk_len}f) 처리 시작")
            print(f"[I2V][CHUNK] {chunk_label} range={start_f}~{end_f - 1} len={chunk_len}f")

            try:
                graph = json_mod.loads(json_mod.dumps(graph_origin))
            except (TypeError, ValueError) as e_clone:
                _notify(f"[오류] 워크플로우 복제 실패 ({chunk_label}): {e_clone}")
                scene_failed = True
                continue

            # 입력 주입
            load_nodes: List[Tuple[str, Dict[str, Any]]] = []
            for node_key2, node_val2 in graph.items():
                try:
                    if isinstance(node_val2, dict) and str(node_val2.get("class_type") or "") == "LoadImage":
                        load_nodes.append((str(node_key2), node_val2))
                except (AttributeError, TypeError):
                    continue
            for load_node_id, _node in load_nodes:
                _set_input_safe(graph, load_node_id, "image", current_input_image_name)
                _notify(f"LoadImage({load_node_id}) 이미지: {current_input_image_name} ({chunk_label})")

            for sampler_id in ("13", "14"):
                _set_input_safe(graph, sampler_id, "noise_seed", scene_seed_value)
            _notify(f"Sampler(13, 14) Seed: {scene_seed_value} ({chunk_label})")

            resize_node_id = "24"
            if ui_width and ui_width > 0:
                _set_input_safe(graph, resize_node_id, "width", ui_width)
            if ui_height and ui_height > 0:
                _set_input_safe(graph, resize_node_id, "height", ui_height)
            if ui_width or ui_height:
                _notify(f"Resize({resize_node_id}) 크기: {ui_width if ui_width else '기본값'}x{ui_height if ui_height else '기본값'} ({chunk_label})")

            if ui_fps and ui_fps > 0:
                _set_input_safe(graph, combine_node_id, "frame_rate", ui_fps)
                _notify(f"Combine({combine_node_id}) FPS: {ui_fps} ({chunk_label})")

            if ui_steps and ui_steps > 0:
                steps_set = 0
                for sampler_id in ("13", "14"):
                    if _set_input_safe(graph, sampler_id, "steps", ui_steps):
                        steps_set += 1
                if steps_set:
                    _notify(f"Sampler(13, 14) Steps: {ui_steps} ({chunk_label})")

            _set_input_safe(graph, i2v_node_id, "length", chunk_len)
            _notify(f"I2V({i2v_node_id}) Length: {chunk_len}f ({chunk_label})")

            pos_txt = str(scene.get("prompt_movie") or scene.get("prompt") or "")
            neg_txt = str(scene.get("prompt_negative") or "")
            if pos_txt:
                _set_input_safe(graph, "22", "text", pos_txt)
            if neg_txt:
                _set_input_safe(graph, "23", "text", neg_txt)
            if pos_txt or neg_txt:
                _notify(f"프롬프트(22/23) 설정 완료. ({chunk_label})")

            try:
                raw_prefix = str(graph.get(combine_node_id, {}).get("inputs", {}).get("filename_prefix", "wan22_"))
            except AttributeError:
                raw_prefix = "wan22_"
            suffix = raw_prefix.split("/", 1)[-1] if "/" in raw_prefix else raw_prefix
            if not suffix or suffix.isdigit() or suffix.replace("-", "").isdigit() or (suffix.endswith("_") and suffix[:-1].isdigit()):
                suffix = "wan22_"
            new_prefix = f"{datetime.date.today().strftime('%Y-%m-%d')}/{suffix}"
            _set_input_safe(graph, combine_node_id, "filename_prefix", new_prefix)
            _notify(f"Combine({combine_node_id}) Prefix: {new_prefix} ({chunk_label})")

            # conditioning(가능 시)
            use_sequence_conditioning = False
            seq_dir_for_condition: Optional[Path] = None
            seq_pattern_text = ""
            if has_sequence_loader and c_idx > 0 and overlap_frames_for_chunk > 0:
                prev_chunk_path = work_dir / f"_chunk_{sid}_{c_idx - 1:05d}.mp4"
                if prev_chunk_path.exists():
                    nb_prev_frames = _probe_nb_frames_ffprobe_local(ffprobe_exe, prev_chunk_path)
                    if nb_prev_frames > 0:
                        start_tail = max(0, nb_prev_frames - overlap_frames_for_chunk)
                        end_tail = nb_prev_frames
                        seq_dir_for_condition = comfy_input_dir / f"{sid}_cond_{c_idx - 1:05d}"
                        try:
                            seq_dir_for_condition.mkdir(parents=True, exist_ok=True)
                        except OSError:
                            seq_dir_for_condition = None
                        if seq_dir_for_condition is not None:
                            seq_template_png = seq_dir_for_condition / "cond_%03d.png"
                            cmd_extract_seq = [
                                ffmpeg_exe, "-y",
                                "-i", str(prev_chunk_path),
                                "-vf", f"trim=start_frame={start_tail}:end_frame={end_tail},setpts=PTS-STARTPTS,fps={target_fps_val}",
                                "-frames:v", str(overlap_frames_for_chunk),
                                str(seq_template_png)
                            ]
                            proc_seq = subprocess.run(
                                cmd_extract_seq,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                encoding="utf-8",
                                errors="ignore",
                                check=False,
                            )
                            if proc_seq.returncode == 0:
                                uploaded_one = False
                                for k in range(1, overlap_frames_for_chunk + 1):
                                    png_path_try = seq_dir_for_condition / f"cond_{k:03d}.png"
                                    if not png_path_try.exists():
                                        continue
                                    try:
                                        with open(png_path_try, "rb") as fbin:
                                            up = requests_mod.post(
                                                base_url + "/upload/image",
                                                files={"image": (png_path_try.name, fbin, "image/png")},
                                                data={"overwrite": "true"},
                                                timeout=60,
                                            )
                                        if 200 <= up.status_code < 300:
                                            uploaded_one = True
                                            break
                                    except requests_mod.RequestException:
                                        uploaded_one = False
                                        break
                                if uploaded_one:
                                    seq_pattern_text = "cond_*.png"
                                    use_sequence_conditioning = True

            if use_sequence_conditioning and seq_dir_for_condition is not None and sequence_loader_node_id:
                _set_input_safe(graph, sequence_loader_node_id, "directory", str(seq_dir_for_condition))
                if seq_pattern_text:
                    _set_input_safe(graph, sequence_loader_node_id, "pattern", seq_pattern_text)
                _set_input_safe(graph, sequence_loader_node_id, "frame_rate", target_fps_val)
                _notify(
                    f"conditioning 시퀀스 사용: node={sequence_loader_node_id}, dir='{seq_dir_for_condition.name}', "
                    f"pattern='{seq_pattern_text or 'cond_*.png'}', fps={target_fps_val} ({chunk_label})"
                )
            else:
                if c_idx > 0:
                    _notify(f"conditioning 시퀀스 미사용 → 1프레임 PNG 폴백 ({chunk_label})")

            # 제출/대기
            _notify(f"워크플로우 제출 시작 ({chunk_label})")
            try:
                result = submit_and_wait_fn(
                    base_url, graph,
                    timeout=int(defaults_i2v.get("timeout_sec") or 3600),
                    poll=float(defaults_i2v.get("poll_sec") or 1.5),
                    on_progress=_relay
                )
            except TimeoutError as e_to:
                _notify(f"[오류] 워크플로우 시간 초과 ({chunk_label}): {e_to}")
                scene_failed = True
                continue
            except (RuntimeError, ValueError) as e_submit:
                _notify(f"[오류] 워크플로우 제출/대기 실패 ({chunk_label}): {e_submit}")
                scene_failed = True
                continue

            outputs_dict = cast(dict, result.get("outputs") or {})
            target_info: Optional[Dict[str, Any]] = None
            valid_mp4_infos: List[Dict[str, Any]] = []
            any_outputs: List[Dict[str, Any]] = []

            for node_out_id, out_val in outputs_dict.items():
                if not isinstance(out_val, dict):
                    continue
                vids = out_val.get("videos") or out_val.get("gifs")
                if isinstance(vids, list):
                    any_outputs.extend(vids)
                    for vinfo in vids:
                        if isinstance(vinfo, dict):
                            fn_lower = str(vinfo.get("filename") or "").lower()
                            t_lower = str(vinfo.get("type") or "").lower()
                            if fn_lower.endswith(".mp4") and t_lower != "preview":
                                valid_mp4_infos.append(vinfo)
                                _notify(f"MP4 후보 발견 (노드 {node_out_id}): {fn_lower} (type: {t_lower})")
                imgs = out_val.get("images")
                if isinstance(imgs, list):
                    any_outputs.extend(imgs)

            if valid_mp4_infos:
                target_info = valid_mp4_infos[-1]
                _notify(f"최종 MP4 파일 선택: {target_info.get('filename')}")
            elif any_outputs:
                target_info = any_outputs[-1]
                _notify(f"MP4 없음. 폴백 파일 선택: {target_info.get('filename')}")
            else:
                _notify("처리할 출력 파일 정보 없음")
                scene_failed = True
                continue

            subfolder = str(target_info.get("subfolder") or "")
            filename_pick = str(target_info.get("filename") or "")
            filetype_pick = str(target_info.get("type") or "output")
            if not filename_pick:
                _notify("출력 파일명 없음")
                scene_failed = True
                continue

            _notify(f"결과 파일 선택: filename='{filename_pick}', type='{filetype_pick}'")

            # 청크 파일 저장
            chunk_name = f"_chunk_{sid}_{c_idx:05d}.mp4"
            chunk_path = work_dir / chunk_name

            try:
                resp = requests_mod.get(
                    base_url + "/view",
                    params={"filename": filename_pick, "subfolder": subfolder, "type": filetype_pick},
                    timeout=300
                )
                resp.raise_for_status()
            except requests_mod.RequestException as e_http:
                _notify(f"[오류] 결과 파일 요청 실패: {e_http}")
                scene_failed = True
                continue

            if not filename_pick.lower().endswith(".mp4"):
                _notify(f"[오류] 결과 파일이 .mp4가 아닙니다! (실제: {filename_pick}).")
                scene_failed = True
                continue

            try:
                chunk_path.parent.mkdir(parents=True, exist_ok=True)
                with open(chunk_path, "wb") as f_out_file:
                    f_out_file.write(resp.content)
                size_bytes = chunk_path.stat().st_size
                _notify(f"청크 파일 저장 완료: {chunk_path.name} ({size_bytes / 1024:.1f} KB)")
                chunk_paths.append(chunk_path)
            except OSError as e_write:
                _notify(f"[오류] 청크 파일 저장 실패: {e_write}")
                scene_failed = True
                continue

            # 다음 청크 입력용 폴백 PNG 준비
            if c_idx < len(segments_list) - 1:
                next_frame_name = f"{sid}_temp_frame_{c_idx:05d}.png"
                next_frame_path = comfy_input_dir / next_frame_name
                temp_frames.append(next_frame_path)

                try:
                    overlap_sec = float(overlap_frames_for_chunk) / float(max(1, target_fps_val))
                except (TypeError, ValueError, ZeroDivisionError):
                    overlap_sec = 1.0 / float(max(1, target_fps_val))
                if overlap_sec < (1.0 / float(max(1, target_fps_val))):
                    overlap_sec = 1.0 / float(max(1, target_fps_val))

                cond_start = max(0, end_f - overlap_frames_for_chunk)
                cond_end = max(0, end_f - 1)
                print(f"[I2V][COND] {chunk_label} next_cond_range={cond_start}~{cond_end} ({overlap_frames_for_chunk}f, ≈{overlap_sec:.3f}s)")

                try:
                    subprocess.run(
                        [ffmpeg_exe, "-y", "-sseof", f"-{overlap_sec:.6f}", "-i", str(chunk_path), "-frames:v", "1", str(next_frame_path)],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE
                    )
                except subprocess.SubprocessError as e_ff:
                    _notify(f"[오류] 마지막-겹침 프레임 추출 실패: {e_ff}")
                    current_input_image_name = server_image_name
                else:
                    ready = False
                    t0 = time.monotonic()
                    while time.monotonic() - t0 < 5.0:
                        try:
                            if next_frame_path.exists() and next_frame_path.stat().st_size > 0:
                                ready = True
                                break
                        except OSError:
                            pass
                        time.sleep(0.1)
                    if not ready:
                        current_input_image_name = server_image_name
                    else:
                        try:
                            with open(next_frame_path, "rb") as fbin:
                                up = requests_mod.post(
                                    base_url + "/upload/image",
                                    files={"image": (next_frame_name, fbin, "image/png")},
                                    data={"overwrite": "true"},
                                    timeout=60,
                                )
                            if not (200 <= up.status_code < 300):
                                current_input_image_name = server_image_name
                            else:
                                current_input_image_name = next_frame_name
                        except requests_mod.RequestException:
                            current_input_image_name = server_image_name

        # 임시 PNG 정리
        for tmp_png in temp_frames:
            try:
                tmp_png.unlink(missing_ok=True)
            except OSError:
                pass

        if scene_failed:
            _notify(f"[오류] 씬 {sid} 처리 중 실패가 있어 최종 파일 생성을 건너뜁니다.")
            for bad_chunk in chunk_paths:
                try:
                    bad_chunk.unlink(missing_ok=True)
                except OSError:
                    pass
            continue

        if not chunk_paths:
            _notify(f"[경고] 씬 {sid}에 대해 생성된 청크 파일이 없습니다. 건너뜁니다.")
            continue

        # ── 산출: 단일 or 병합 ─────────────────────────────────
        if len(chunk_paths) == 1:
            _notify(f"씬 {sid} 단일 청크 완료. 최종 파일로 이동합니다.")
            try:
                shutil.move(str(chunk_paths[0]), str(clip_mp4))
                _notify(f"파일 이동 완료: {clip_mp4.name}")
            except OSError as e_mv:
                _notify(f"[오류] 단일 청크 파일 이동 실패 ({sid}): {e_mv}")
        else:
            print(f"[I2V][MERGE] id={sid} chunks={len(chunk_paths)} overlap={overlap_frames_for_chunk}f fps={target_fps_val}")
            # 페이드(xfade) 대신 덮어쓰기(컷) 병합
            ok_merge = _concat_cut_no_fade(
                ffmpeg_bin_arg=ffmpeg_exe,
                ffprobe_bin_arg=ffprobe_exe,
                clip_paths_arg=chunk_paths,
                overlap_frames_arg=overlap_frames_for_chunk,
                fps_arg=target_fps_val,
                out_path_arg=clip_mp4,
                work_dir_arg=work_dir
            )
            if ok_merge:
                _notify(f"청크 병합 완료(컷): {clip_mp4.name}")
            else:
                _notify(f"[오류] 씬 {sid} 청크 병합(컷) 실패")

        # 임시 mp4 정리
        for tmp_mp4 in chunk_paths:
            try:
                tmp_mp4.unlink(missing_ok=True)
            except OSError:
                pass

        # ── 최종 트림: 처음 계산한 frame_length로 정확히 자르기 ──
        _trim_to_frames(ffmpeg_exe, ffprobe_exe, clip_mp4, clip_mp4, frame_length, target_fps_val)

    _notify("모든 씬 처리 완료.")























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

    # 작업 폴더(기존 규칙 유지)
    if work_dir is None:
        work_dir = base_dir if base_dir.name == "xfade_work" else (base_dir / "xfade_work")
    else:
        work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 출력 경로 기본값 및 정크 미러 폴더 결정
    if out_path is None:
        out_path = base_dir / "final.mp4"
    out_path = Path(out_path)
    mirror_dir = out_path.parent / out_path.stem
    mirror_dir.mkdir(parents=True, exist_ok=True)
    print(f"[XC-MIRROR] dir={mirror_dir}")

    # 출력 해상도(없으면 1280x720)
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
            "-v", "error",
            "-print_format", "json",
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
                fmt = parsed.get("format") or {}
                dur_val = fmt.get("duration")
                if dur_val is not None:
                    try:
                        probe_map["duration_sec"] = float(dur_val)
                    except (TypeError, ValueError):
                        pass
                bit_val = fmt.get("bit_rate")
                if bit_val is not None:
                    probe_map["bit_rate"] = bit_val
        else:
            probe_map["ffprobe_out"] = proc_probe.stdout
        try:
            probe_map["size_bytes"] = path_for_probe.stat().st_size
        except OSError:
            pass
        return probe_map

    def _normalize_one(src_path_norm: Path, dst_path_norm: Path) -> None:
        cmd_norm = [
            ffmpeg_bin, "-y",
            "-fflags", "+genpts",
            "-i", str(src_path_norm),
            "-an",
            "-vf",
            (
                f"fps={target_fps},"
                f"scale=w={out_w}:h={out_h}:force_original_aspect_ratio=decrease,"
                f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1,format=yuv420p,setpts=PTS-STARTPTS"
            ),
            "-r", f"{target_fps}",
            "-vsync", "cfr",
            "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p",
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
        # 입력 두 영상을 표준화본으로 변환
        norm_a_path = work_dir / "_norm_a.mp4"
        norm_b_path = work_dir / "_norm_b.mp4"
        _normalize_one(cur_path_in, norm_a_path)
        _normalize_one(nxt_path_in, norm_b_path)

        # 길이/오프셋 계산
        info_a = _probe_video_info(norm_a_path)
        info_b = _probe_video_info(norm_b_path)

        # nb_frames가 없을 수도 있으니 duration_sec 기반으로 보강
        def _frames_from_info(info_map: Dict[str, Any]) -> int:
            nb = info_map.get("nb_frames")
            if isinstance(nb, str) and nb.isdigit():
                try:
                    return int(nb)
                except ValueError:
                    pass
            dur = info_map.get("duration_sec")
            if isinstance(dur, (int, float)) and dur > 0:
                # 반올림보다 바닥으로 깎아 정확도를 확보
                return max(0, int(dur * target_fps))
            return 0

        frames_a = _frames_from_info(info_a)
        frames_b = _frames_from_info(info_b)
        fade_frames = max(0, int(overlap_frames_in))
        if frames_a <= 0 or frames_b <= 0:
            raise RuntimeError("정규화본의 프레임 수 계산 실패")

        # 페이드 오프셋: A의 끝에서 fade_frames만큼 앞
        # (A가 N프레임이면, 오프셋초는 (N - fade)/fps)
        offset_sec = float(max(0, frames_a - fade_frames)) / float(target_fps)
        duration_sec = float(fade_frames) / float(target_fps)

        print(f"[XC-PLAN] fade={fade_frames}f ({duration_sec:.3f}s), offset={offset_sec:.3f}s, expected≥ {(frames_a + frames_b - fade_frames)}f")

        cmd_xf = [
            ffmpeg_bin, "-y",
            "-fflags", "+genpts",
            "-i", str(norm_a_path),
            "-i", str(norm_b_path),
            "-filter_complex",
            (
                "[0:v]format=yuv420p,setsar=1[v0];"
                "[1:v]format=yuv420p,setsar=1[v1];"
                f"[v0][v1]xfade=transition=fade:duration={duration_sec:.6f}:offset={offset_sec:.6f}[v]"
            ),
            "-map", "[v]",
            "-r", f"{target_fps}",
            "-fps_mode", "cfr",
            "-vsync", "cfr",
            "-c:v", "libx264",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
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
        print(f"[XC-OUT] {out_path_pair.name}: sec={out_info.get('duration_sec','?')} frames={out_info.get('nb_frames','?')} @fps={target_fps}")

    def _concat_cut_no_fade(
        ffmpeg_exe: str,
        ffprobe_exe: str,
        clip_paths_cut: List[Path],
        overlap_frames_cut: int,
        fps_cut: int,
        out_path_cut: Path,
        work_dir_cut: Path,
    ) -> bool:
        """
        하드컷 병합:
        - 각 청크를 동일 fps/해상도로 정규화
        - 첫 청크는 전체 사용
        - 이후 청크는 overlap_frames_cut 만큼 '앞에서 잘라서' 이어붙임(뒤 청크가 겹침 구간을 덮어씀)
        """
        import subprocess as _sub
        import shutil as _sh

        # 정규화본 만들기
        norm_list: List[Path] = []
        for idx_norm, src_norm in enumerate(clip_paths_cut):
            dst_norm = work_dir_cut / f"_concat_norm_{idx_norm:03d}.mp4"
            cmd_norm2 = [
                ffmpeg_exe, "-y",
                "-fflags", "+genpts",
                "-i", str(src_norm),
                "-an",
                "-vf",
                (
                    f"fps={fps_cut},"
                    f"scale=w={out_w}:h={out_h}:force_original_aspect_ratio=decrease,"
                    f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2,"
                    f"setsar=1,format=yuv420p,setpts=PTS-STARTPTS"
                ),
                "-r", f"{fps_cut}",
                "-vsync", "cfr",
                "-c:v", "libx264", "-crf", "18",
                "-pix_fmt", "yuv420p",
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

        # 트리밍 규칙: 첫 청크는 그대로, 두 번째부터는 앞에서 overlap_frames_cut 만큼 제거
        trimmed_list: List[Path] = []
        for idx_trim, src_trim in enumerate(norm_list):
            # 프레임 수 파악
            cmd_probe2 = [
                ffprobe_exe,
                "-v", "error",
                "-select_streams", "v:0",
                "-count_frames",
                "-show_entries", "stream=nb_read_frames",
                "-of", "default=nokey=1:noprint_wrappers=1",
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
            try:
                nb_frames_str = (proc_p2.stdout or "0").strip()
                nb_frames_int = int(nb_frames_str) if nb_frames_str.isdigit() else 0
            except ValueError:
                nb_frames_int = 0

            if idx_trim == 0 or overlap_frames_cut <= 0:
                # 첫 조각은 그대로 사용
                trimmed_list.append(src_trim)
                continue

            keep_frames = max(0, nb_frames_int - overlap_frames_cut)
            if keep_frames <= 0:
                # 유지 프레임이 없다면 빈 조각, 스킵
                continue

            trim_out = work_dir_cut / f"_concat_trim_{idx_trim:03d}.mp4"
            # 앞에서 overlap_frames_cut 만큼 제거: select='gte(n,overlap)'을 사용
            cmd_trim = [
                ffmpeg_exe, "-y",
                "-i", str(src_trim),
                "-vf", f"select='gte(n\\,{overlap_frames_cut})',setpts=N/FRAME_RATE/TB",
                "-r", f"{fps_cut}",
                "-an",
                "-c:v", "libx264", "-crf", "18",
                "-pix_fmt", "yuv420p",
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

        # concat
        concat_list_path = work_dir_cut / "_concat_list.txt"
        try:
            with open(concat_list_path, "w", encoding="utf-8", newline="\n") as f_list:
                for p_item in trimmed_list:
                    p_str = str(p_item).replace("\\", "\\\\").replace("'", "\\'")
                    f_list.write("file '{}'\n".format(p_str))
        except OSError as e_osw:
            raise RuntimeError(f"concat list 작성 실패: {e_osw}") from e_osw

        cmd_concat = [
            ffmpeg_exe, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list_path),
            "-c:v", "copy",
            "-an" if audio_path is None else "-c:a",
        ]
        if audio_path is None:
            cmd_concat.append(str(out_path_cut))
        else:
            # 오디오가 있으면 비디오 copy + 오디오는 별도 멀티플렉스 단계에서 처리되므로 여기선 비디오만 만듦
            cmd_concat[-1] = "aac"
            cmd_concat.append("-shortest")
            cmd_concat.append(str(out_path_cut))

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

        # 미러 보관
        try:
            dst_concat = mirror_dir / out_path_cut.name
            _sh.copyfile(out_path_cut, dst_concat)
            print(f"[XC-MIRROR-COPY] concat->{dst_concat.name}")
        except OSError:
            pass

        return True

    # 입력 요약 및 원본 청크를 미러 폴더에 복사
    print(f"[XC-BEGIN] items={len(clip_paths)} fps={target_fps} overlap_frames={int(overlap_frames)} out={out_w}x{out_h}")
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

    # 페이드 off면 하드컷 병합 경로로 실행
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
        # 필요 시 오디오 멀티플렉스
        if audio_path and Path(audio_path).exists():
            cmd_mux_nf = [
                ffmpeg_bin, "-y",
                "-fflags", "+genpts",
                "-i", str(out_path),
                "-i", str(audio_path),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
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

    # ===== 여기부터는 기존 '크로스페이드' 경로 (기존 동작 보존) =====

    # 첫 클립 표준화: cur_000.mp4
    cur_path = work_dir / "cur_000.mp4"
    cmd_first = [
        ffmpeg_bin, "-y",
        "-fflags", "+genpts",
        "-i", str(first_clip_path),
        "-an",
        "-vf",
        (
            f"fps={target_fps},"
            f"scale=w={out_w}:h={out_h}:force_original_aspect_ratio=decrease,"
            f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1,format=yuv420p,setpts=PTS-STARTPTS"
        ),
        "-r", f"{target_fps}",
        "-vsync", "cfr",
        "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p",
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

    # 초기 표준화본 미러 복사
    try:
        dst0 = mirror_dir / cur_path.name
        shutil.copyfile(cur_path, dst0)
        print(f"[XC-MIRROR-COPY] cur_000->{dst0.name}")
    except OSError:
        pass

    # 순차 xfade(기존 흐름 유지)
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

        # _xfade_two가 생성한 정규화본과 현재 결과를 미러 복사
        try:
            norm_a = work_dir / "_norm_a.mp4"
            norm_b = work_dir / "_norm_b.mp4"
            if norm_a.exists():
                dst_a = mirror_dir / f"_norm_a_{i:03d}.mp4"
                shutil.copyfile(norm_a, dst_a)
                print(f"[XC-MIRROR-COPY] norm_a->{dst_a.name}")
            if norm_b.exists():
                dst_b = mirror_dir / f"_norm_b_{i:03d}.mp4"
                shutil.copyfile(norm_b, dst_b)
                print(f"[XC-MIRROR-COPY] norm_b->{dst_b.name}")
            dst_cur = mirror_dir / cur_path.name
            shutil.copyfile(cur_path, dst_cur)
            print(f"[XC-MIRROR-COPY] cur->{dst_cur.name}")
        except OSError:
            pass

    # 오디오 멀티플렉스(선택) - 기존 동작 보존
    if audio_path and Path(audio_path).exists():
        cmd_mux = [
            ffmpeg_bin, "-y",
            "-fflags", "+genpts",
            "-i", str(cur_path),
            "-i", str(audio_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
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
            ffmpeg_bin, "-y",
            "-fflags", "+genpts",
            "-i", str(cur_path),
            "-c:v", "copy",
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
    same_size = (fin_map.get("size_bytes") == last_map.get("size_bytes"))
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




















def build_missing_images_from_story(
    story_path: str | _Path,
    *,
    ui_width: int,
    ui_height: int,
    steps: int = 28,
    timeout_sec: int = 300,
    poll_sec: float = 1.5,
    workflow_path: str | _Path | None = None,
    on_progress: Optional[Dict[str, Any] | Callable[[Dict[str, Any]], None]] = None,
) -> List[_Path]:
    """
    누락 이미지 생성:
    - story_path와 같은 폴더의 video.json 우선, 없으면 story.json 사용
    - scenes[*].img_file 없으면 ComfyUI로 생성하고 파일 경로를 video.json(그리고 story.json)에 반영
    - 워크플로: settings.JSONS_DIR / nunchaku_qwen_image_swap.json (또는 인자 workflow_path)
    - 프롬프트: scene.prompt_movie > (scene.prompt + '\\n' + scene.prompt_img)
    - 네거티브: scene.prompt_negative or defaults.image.negative
    """

    def _notify(stage: str, msg: str = "", **extra: Any) -> None:
        if not on_progress:
            return
        data: Dict[str, Any] = {"stage": stage, "msg": msg}
        if extra:
            data.update(extra)
        try:
            if callable(on_progress):
                on_progress(data)
            elif isinstance(on_progress, dict):
                cb = on_progress.get("callback")
                if callable(cb):
                    cb(data)
        except (RuntimeError, ValueError, TypeError):
            pass

    p_story = _Path(story_path).resolve()
    story_dir = p_story.parent if p_story.is_file() else _Path(story_path).resolve()
    if not story_dir.exists():
        raise FileNotFoundError(f"경로 없음: {story_dir}")

    p_video = story_dir / "video.json"
    video = load_json(p_video, {}) or {}
    p_story_json = story_dir / "story.json"
    story = load_json(p_story_json, {}) or {}

    paths_v = video.get("paths") or {}
    root_dir = _Path(paths_v.get("root") or story.get("paths", {}).get("root") or story_dir)
    imgs_dir_name = str(paths_v.get("imgs_dir") or story.get("paths", {}).get("imgs_dir") or "imgs")
    img_root = ensure_dir(root_dir / imgs_dir_name)

    try:
        from settings import JSONS_DIR  # type: ignore
        base_jsons = _Path(JSONS_DIR)
    except (ImportError, AttributeError):
        base_jsons = _Path(r"C:\my_games\shorts_make\app\jsons")

    wf_path = _Path(workflow_path) if workflow_path else (base_jsons / "nunchaku_qwen_image_swap.json")
    if not wf_path.exists():
        raise FileNotFoundError(f"필수 워크플로 없음: {wf_path}")

    with open(wf_path, "r", encoding="utf-8") as f:
        graph_origin: Dict[str, Any] = _json.load(f)

    def _find_nodes(gdict: Dict[str, Any], class_type: str) -> List[str]:
        hits: List[str] = []
        for nid, node in (gdict or {}).items():
            try:
                if str(node.get("class_type")) == class_type:
                    hits.append(str(nid))
            except (AttributeError, KeyError, TypeError):
                continue
        return hits

    def _set_input(gdict: Dict[str, Any], nid: str, key: str, val: Any) -> None:
        try:
            gdict[str(nid)].setdefault("inputs", {})[key] = val
        except (KeyError, TypeError):
            pass

    latent_ids = _find_nodes(graph_origin, "EmptySD3LatentImage") + _find_nodes(graph_origin, "EmptyLatentImage")
    latent_ids = list(dict.fromkeys(latent_ids))
    ksampler_ids = _find_nodes(graph_origin, "KSampler")
    reactor_ids = [nid for nid, node in graph_origin.items() if str(node.get("class_type")) == "ReActorFaceSwap"]

    def _resolve_face_image_by_name(name: str) -> _Path | None:
        try:
            from settings import CHARACTER_DIR  # type: ignore
            base_dir = _Path(CHARACTER_DIR)
        except (ImportError, AttributeError):
            base_dir = _Path(r"C:\my_games\shorts_make\character")
        exts = (".png", ".jpg", ".jpeg", ".webp")
        for ext in exts:
            p = base_dir / f"{name}{ext}"
            if p.exists():
                return p
        return None

    def _pick_scene_character_name(scene: Dict[str, Any]) -> str | None:
        chars = scene.get("characters") or []
        if isinstance(chars, list) and chars:
            cc0 = chars[0]
            if isinstance(cc0, str):
                name = cc0.strip()
                return name or None
            if isinstance(cc0, dict):
                cand = (str(cc0.get("id") or "") or str(cc0.get("name") or "")).strip()
                return cand or None
        cobjs = scene.get("character_objs") or []
        if isinstance(cobjs, list) and cobjs:
            o0 = cobjs[0]
            if isinstance(o0, dict):
                cand = (str(o0.get("id") or "") or str(o0.get("name") or "")).strip()
                return cand or None
        return None

    def _inject_face_to_reactors(gdict: Dict[str, Any], file_name: str) -> List[str]:
        applied_ids: List[str] = []
        for ridt in reactor_ids:
            try:
                ins = gdict[ridt].setdefault("inputs", {})
                if "source_image" in ins:
                    ins["source_image"] = file_name
                    applied_ids.append(ridt)
                elif "input_image" in ins:
                    ins["input_image"] = file_name
                    applied_ids.append(ridt)
            except (KeyError, TypeError):
                continue
        return applied_ids

    def _inject_face_image_loaders(_gdict: Dict[str, Any], _file_name: str) -> List[str]:
        return []

    def _parse_first_char_index(scene: Dict[str, Any]) -> int:
        specs = scene.get("characters") or scene.get("character_objs") or []
        if not specs:
            return 0
        if isinstance(specs[0], str) and ":" in specs[0]:
            try:
                return int(specs[0].split(":")[1])
            except (ValueError, IndexError):
                return 0
        if isinstance(specs[0], dict):
            try:
                return int(specs[0].get("index", 0))
            except (ValueError, TypeError):
                return 0
        return 0

    try:
        from audio_sync import _submit_and_wait as _wait_core  # type: ignore
    except (ImportError, AttributeError):
        from app.audio_sync import _submit_and_wait as _wait_core  # type: ignore

    # ====== 여기만 변경(대기 로그 50번에 1번씩만 전달) ======
    def _wait_img(url: str, gdict: Dict[str, Any], *, timeout: int, poll: float, progress_cb: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
        wait_i = 0  # throttling counter inside one request

        def _relay(prog: Dict[str, Any]) -> None:
            nonlocal wait_i
            raw = str(prog.get("msg") or "")
            if raw.startswith("[MUSIC]"):
                patched = "[IMG]" + raw[len("[MUSIC]"):]
            else:
                patched = "[IMG] " + raw if raw else "[IMG]"

            # throttle: forward only every 50th wait message
            wait_i += 1
            if wait_i % 50 != 0:
                return

            try:
                progress_cb({"stage": "wait", "msg": patched})
            except (RuntimeError, ValueError, TypeError):
                pass

        return _wait_core(url, gdict, timeout=timeout, poll=poll, on_progress=_relay)
    # ================================================

    created: List[_Path] = []
    base_url = str(video.get("comfy_host") or story.get("comfy_host") or "http://127.0.0.1:8188").rstrip("/")
    _notify("begin", f"[IMG] 대상 scenes={(len(video.get('scenes') or story.get('scenes') or []))} wf={wf_path.name}")

    defaults_v = video.get("defaults") or {}
    defaults_img = defaults_v.get("image") or (story.get("defaults", {}).get("image") if story else {}) or {}
    default_neg = str(defaults_img.get("negative") or "")

    scenes: List[Dict[str, Any]] = list(video.get("scenes") or story.get("scenes") or [])
    req_count = 0

    for idx, sc in enumerate(scenes):
        try:
            # ★ 변경: 기본 SID 패딩 3→5 (기능 동일, 명명 규칙만 조정)
            sid = str(sc.get("id") or f"scene_{idx:05d}")
            img_path = img_root / f"{sid}.png"

            scene_img = str(sc.get("img_file") or "")
            if scene_img:
                p_scene_img = _Path(scene_img)
                if not p_scene_img.is_absolute():
                    p_scene_img = img_root / p_scene_img.name
                if p_scene_img.exists():
                    _notify("skip", f"[IMG] {sid} → 존재 {p_scene_img.name}")
                    continue
            if img_path.exists():
                _notify("skip", f"[IMG] {sid} → 존재 {img_path.name}")
                continue

            graph = _json.loads(_json.dumps(graph_origin))

            for nid in latent_ids:
                _set_input(graph, nid, "width", int(ui_width))
                _set_input(graph, nid, "height", int(ui_height))

            pos_text = str(sc.get("prompt_movie") or "").strip()
            if not pos_text:
                p_main = str(sc.get("prompt") or "").strip()
                p_img = str(sc.get("prompt_img") or "").strip()
                pos_text = f"{p_main}\n{p_img}" if p_main and p_img else (p_main or p_img)
            neg_text = str(sc.get("prompt_negative") or "") or default_neg

            def _apply_prompts(gdict: Dict[str, Any], pos: str, neg: str) -> None:
                fields = ("text", "text_g", "text_l")
                for nid2, node2 in gdict.items():
                    try:
                        if not isinstance(node2, dict):
                            continue
                        if not any(k in node2.get("inputs", {}) for k in fields):
                            continue
                        hint = str(node2.get("label") or "") + " " + str(node2.get("_meta", {}).get("title") or "")
                        is_neg = ("neg" in hint.lower()) or ("negative" in hint.lower()) or ("Negative" in hint)
                        val = neg if is_neg else pos
                        inp = node2.setdefault("inputs", {})
                        for fk in fields:
                            inp[fk] = val
                    except (AttributeError, KeyError, TypeError):
                        continue

            _apply_prompts(graph, pos_text, neg_text)

            face_path: _Path | None = None
            char_name = _pick_scene_character_name(sc)
            if isinstance(char_name, str):
                face_path = _resolve_face_image_by_name(char_name)

            if face_path is None:
                char_objs = sc.get("character_objs") or []
                if isinstance(char_objs, list) and char_objs:
                    obj0 = char_objs[0]
                    if isinstance(obj0, dict):
                        fi = obj0.get("img_file") or obj0.get("image") or ""
                        if fi:
                            fp = _Path(fi)
                            face_path = fp if fp.exists() else None

            if face_path is None:
                chars2 = sc.get("characters") or []
                if isinstance(chars2, list) and chars2:
                    c0 = chars2[0]
                    if isinstance(c0, dict):
                        fi2 = c0.get("img_file") or c0.get("image") or ""
                        if fi2:
                            fp2 = _Path(fi2)
                            face_path = fp2 if fp2.exists() else None

            face_name: str | None = None
            if face_path and face_path.exists():
                try:
                    from settings import COMFY_INPUT_DIR  # type: ignore
                    comfy_in = _Path(COMFY_INPUT_DIR)
                except (ImportError, AttributeError):
                    comfy_in = _Path(r"C:\my_games\shorts_make\app\comfy_inputs")
                comfy_in.mkdir(parents=True, exist_ok=True)
                face_name = face_path.name
                try:
                    shutil.copy2(str(face_path), str(comfy_in / face_name))
                except (OSError, shutil.Error):
                    pass

            applied_reactors: List[str] = []
            if face_name:
                target_rid = reactor_ids[0] if reactor_ids else None
                for rnid in reactor_ids:
                    try:
                        graph[rnid].setdefault("inputs", {})["enabled"] = (rnid == target_rid)
                    except (KeyError, TypeError):
                        continue
                if target_rid:
                    try:
                        rinp = graph[target_rid].setdefault("inputs", {})
                        link = rinp.get("source_image")
                        if isinstance(link, list) and len(link) == 2:
                            link_id = str(link[0])
                            if link_id in graph and str(graph[link_id].get("class_type")) == "LoadImage":
                                graph[link_id].setdefault("inputs", {})["image"] = face_name
                                applied_reactors.append(target_rid)
                        else:
                            if "source_image" in rinp:
                                rinp["source_image"] = face_name
                                applied_reactors.append(target_rid)
                            elif "input_image" in rinp:
                                rinp["input_image"] = face_name
                                applied_reactors.append(target_rid)
                    except (KeyError, TypeError):
                        pass

                face_idx = _parse_first_char_index(sc)
                for rnid in reactor_ids:
                    try:
                        inp_map = graph[rnid].setdefault("inputs", {})
                        if "source_faces_index" in inp_map:
                            inp_map["source_faces_index"] = int(face_idx)
                        if "input_faces_index" in inp_map:
                            inp_map["input_faces_index"] = int(face_idx)
                    except (KeyError, TypeError, ValueError):
                        continue
            else:
                for rnid in reactor_ids:
                    try:
                        graph[rnid].setdefault("inputs", {})["enabled"] = False
                    except (KeyError, TypeError):
                        continue

            try:
                load_ids = _find_nodes(graph, "LoadImage")
                for lid in load_ids:
                    try:
                        li = graph[lid].get("inputs", {})  # type: ignore
                        _notify("debug-loadimage", f"[IMG][DBG] LoadImage nid={lid} image={str(li.get('image') or '')}")
                    except (AttributeError, KeyError, TypeError):
                        continue

                for rid in reactor_ids:
                    try:
                        rin = graph[rid].get("inputs", {})  # type: ignore
                        _notify(
                            "debug-reactor",
                            "[IMG][DBG] ReActor nid="
                            + str(rid)
                            + " enabled="
                            + str(rin.get("enabled"))
                            + " src_img="
                            + str(rin.get("source_image") or rin.get("input_image") or "")
                            + " src_idx="
                            + str(rin.get("source_faces_index"))
                            + " in_idx="
                            + str(rin.get("input_faces_index"))
                        )
                    except (AttributeError, KeyError, TypeError):
                        continue

                if face_name and not applied_reactors:
                    _notify("debug-warn", "[IMG][DBG] face image 있음에도 ReActor에 연결된 입력이 감지되지 않음")
            except (RuntimeError, ValueError, TypeError):
                pass

            for nid in ksampler_ids:
                _set_input(graph, nid, "steps", int(steps))

            req_count += 1
            _notify("submit", f"[IMG] /prompt {sid}")
            result = _wait_img(
                base_url,
                graph,
                timeout=timeout_sec,
                poll=poll_sec,
                progress_cb=lambda d: _notify("wait", str(d.get("msg") or "")),
            )

            files: List[Dict[str, Any]] = []
            for _, out in (result.get("outputs") or {}).items():
                if isinstance(out, dict):
                    files.extend(out.get("images", []) or [])
            if not files:
                raise RuntimeError("출력 이미지 없음")

            last = files[-1]
            subfolder = str(last.get("subfolder") or "")
            fname = str(last.get("filename") or "")
            if not fname:
                raise RuntimeError("출력 파일명 없음")

            rimg = requests.get(
                base_url.rstrip("/") + "/view",
                params={"filename": fname, "subfolder": subfolder},
                timeout=30,
            )
            if not rimg.ok:
                raise RuntimeError(f"/view 실패: {rimg.status_code}")

            img_path.parent.mkdir(parents=True, exist_ok=True)
            with open(img_path, "wb") as fpw:
                fpw.write(rimg.content)

            for s in scenes:
                if str(s.get("id")) == sid:
                    s["img_file"] = str(img_path)
                    break
            try:
                save_json(p_video, video)
                _notify("save", f"[IMG] video.json 갱신: {sid}")
            except (OSError, ValueError, TypeError):
                pass

            if story:
                try:
                    scs = story.get("scenes") or []
                    for s in scs:
                        if str(s.get("id")) == sid:
                            s["img_file"] = str(img_path)
                            break
                    save_json(p_story_json, story)
                except (OSError, ValueError, TypeError):
                    pass

            created.append(img_path)
            _notify("done", f"[IMG] 저장 {img_path.name}")

        except (OSError, ValueError, TypeError, KeyError, RuntimeError, requests.RequestException) as err:
            _notify("scene-error", f"[IMG] {sc.get('id')}: {err}")
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




from settings import CHARACTER_DIR, COMFY_INPUT_DIR

def _resolve_character_image_path(scene: dict, story: dict) -> str | None:
    from pathlib import Path
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
    from pathlib import Path
    import shutil
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







# === [MERGED FROM image_movie_docs.py] ===
# app/image_movie_docs.py
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from textwrap import dedent
import re
from pathlib import Path
import json
# 유연 import
try:
    # 패키지 실행
    from app import settings as settings   # 소문자 별칭 유지
    from app.utils import load_json, save_json
except ImportError:
    # 단독 실행
    import settings as settings            # 소문자 별칭 유지
    from utils import load_json, save_json  # type: ignore

S = settings  # noqa: N816  # (하위호환: 기존 코드가 S를 참조해도 동작)




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
    try:
        from app.utils import load_json  # 패키지 실행
    except ImportError:
        from utils import load_json      # 단독 실행

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
    try:
        from app.utils import load_json, save_json
    except ImportError:
        from utils import load_json, save_json

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
    try:
        from app.utils import load_json, save_json
    except ImportError:
        from utils import load_json, save_json

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
    from pathlib import Path
    from typing import Any, Dict, List, Tuple
    import math
    import re

    # ---- utils ----
    try:
        from app.utils import load_json  # type: ignore
    except Exception:
        from utils import load_json  # type: ignore

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
    from pathlib import Path

    try:
        from app import settings as _s_cfg  # 로컬 별칭(소문자)로 가져와 외부 'S' 가리기 방지
    except ImportError:
        import settings as _s_cfg

    pj = Path(audio_path).parent / "project.json"
    meta = load_json(pj, {}) if pj.exists() else {}
    ui = meta.get("ui_prefs") or {}

    img_size = ui.get("image_size") or getattr(_s_cfg, "DEFAULT_IMG_SIZE", (832, 1472))
    w, h = tuple(img_size)

    fps = int(ui.get("movie_fps") or getattr(_s_cfg, "DEFAULT_MOVIE_FPS", 24))
    overlap = int(ui.get("movie_overlap") or getattr(_s_cfg, "DEFAULT_MOVIE_OVERLAP", 12))

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


def parse_character_spec(item: Any) -> Dict[str, Any]:
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
      - 리스트 각 원소를 parse_character_spec()으로 dict화
      - scene["layout"]["face_indices"] (있다면)로 index를 보완
      - scene["face_indices"] (flat dict)도 함께 만들어 소비 측이 편하게 사용
    반환: 수정된 scene (원본 shallow copy 후 필드 갱신)
    """
    sc = dict(scene or {})
    chars_in = sc.get("characters") or []
    norm: List[Dict[str, Any]] = [parse_character_spec(x) for x in chars_in]

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

    try:
        # app 패키지 실행/단일 파일 실행 모두 대응
        from app.utils import load_json, save_json  # type: ignore
    except ImportError:
        from utils import load_json, save_json  # type: ignore

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
    from pathlib import Path
    import re
    from typing import SupportsFloat, SupportsIndex

    try:
        from app.utils import load_json, save_json  # type: ignore
    except ImportError:
        from utils import load_json, save_json  # type: ignore

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









# ======================= /A안 GPT 적용기 끝 =======================

if __name__ == "__main__":
    info = verify_demo_block()
    print("BEFORE:", info)
    if info["exists"]:
        print(strip_demo_block())
    else:
        print("No demo block found.")
