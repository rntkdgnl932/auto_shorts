# -*- coding: utf-8 -*-
# i2v 분할/실행/합치기(교차 페이드) + 누락 이미지 생성 (ComfyUI 연동)
from __future__ import annotations

import subprocess
from typing import List, Tuple, Optional, Dict, Any, Callable
from pathlib import Path as _Path
import shutil
import re
import os
from pathlib import Path
import json, time, requests, shutil

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
    from app.music_gen import (
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
def plan_segments(total_frames: int, base_chunk: int = 300, overlap: int = 12) -> List[Tuple[int, int]]:
    segs, s = [], 0
    while s < total_frames:
        e = min(s + base_chunk, total_frames)
        segs.append((s, e))
        if e >= total_frames:
            break
        s = max(0, e - overlap)
    return segs

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
def build_shots_with_i2v(project_dir: str, total_frames: int) -> List[Path]:
    import copy

    proj = Path(project_dir)
    meta = load_json(proj / "project.json") or {}
    if not meta:
        raise RuntimeError("project.json 없음")

    vdir = ensure_dir(Path(meta["paths"]["video_dir"]))
    in_fps = int(meta.get("i2v_plan", {}).get("input_fps", 60))
    out_fps = int(meta.get("i2v_plan", {}).get("target_fps", in_fps))
    base = int(meta.get("i2v_plan", {}).get("base_chunk", 300))
    ov = int(meta.get("i2v_plan", {}).get("overlap", 12))

    # ▶ 입력/출력 FPS 차이를 반영해 겹침 프레임 보정
    ov_eff = recalc_overlap(in_fps, out_fps, ov)
    segs = plan_segments(total_frames, base, ov_eff)

    # 기본 그래프는 한 번만 로드하고, 매 세그먼트마다 deepcopy
    base_graph = _load_workflow_graph(I2V_WORKFLOW)

    base_url = _choose_host()
    out_paths: List[Path] = []

    for idx, (a, b) in enumerate(segs):
        out_mp4 = vdir / f"clip_{idx:03d}.mp4"
        if out_mp4.exists():
            out_paths.append(out_mp4)
            continue

        # ▶ 세그먼트별 그래프 복제(노드 입력 덮어쓰기의 부작용 방지)
        graph = copy.deepcopy(base_graph)

        # TODO: 실제 노드 ID/키는 환경에 맞게 조정
        overrides = {
            "20.inputs.start_frame": a,
            "20.inputs.end_frame": b,
            # "33.inputs.filename_prefix": f"clip_{idx:03d}",
            # "33.inputs.subfolder": str(vdir).replace("\\", "/"),
        }
        _apply_overrides(graph, overrides)

        # ▶ SaveVideo/VideoWrite 계열 노드에서 fps 입력키가 있으면 out_fps 주입
        for _nid, node in _find_nodes_by_class_contains(graph, "savevideo"):
            ins = node.setdefault("inputs", {})
            for k in ("fps", "frame_rate", "framerate", "frame_rate_num"):
                if k in ins:
                    try:
                        ins[k] = int(out_fps)
                    except Exception:
                        pass
        for _nid, node in _find_nodes_by_class_contains(graph, "videowrite"):
            ins = node.setdefault("inputs", {})
            for k in ("fps", "frame_rate", "framerate", "frame_rate_num"):
                if k in ins:
                    try:
                        ins[k] = int(out_fps)
                    except Exception:
                        pass

        # ▶ 제출 및 응답 활용(간단 로깅으로 hist '미사용' 경고 해소)
        def _wait_i2v(url, graph, *, timeout: int, poll: float):
            def relay(info: dict):
                msg = str(info.get("msg") or "")
                if msg.startswith("[MUSIC]"):
                    info = dict(info)
                    info["msg"] = "[I2V]" + msg[len("[MUSIC]"):]
                print(info.get("msg", ""), flush=True)

            return _submit_and_wait(url, graph, timeout=timeout, poll=poll)
        # hist = _submit_and_wait(base_url, graph, timeout=1200, poll=1.5)
        hist = _wait_i2v(base_url, graph, timeout=1200, poll=1.5)
        try:
            outputs = hist.get("outputs") or {}
            nouts = 0
            for _nid, out in outputs.items():
                if isinstance(out, dict):
                    nouts += len(out.get("images", []) or [])
                    nouts += len(out.get("files", []) or [])
                    nouts += len(out.get("video", []) or [])
            print(f"[I2V] seg {idx} {a}-{b} frames -> outputs: {nouts}", flush=True)
        except Exception:
            pass

        # ▶ 결과 파일 보정(파일명이 임의일 때 최근 파일을 clip_xxx.mp4로 정규화)
        if not out_mp4.exists():
            latest = max(vdir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, default=None)
            if latest:
                latest.rename(out_mp4)
        if not out_mp4.exists():
            raise RuntimeError(f"세그먼트 결과 없음: {idx}")

        out_paths.append(out_mp4)

    return out_paths


# ── 합치기(교차 페이드) ────────────────────────────────────────────────────
def xfade_concat(clip_paths: List[Path], overlap_frames: int, fps: int,
                 *, audio_path: Optional[Path] = None, out_path: Optional[Path] = None) -> Path:
    if not clip_paths:
        raise ValueError("clip_paths가 비어있습니다.")

    # 단일 클립이면 오디오만 입히거나 그대로 반환
    if len(clip_paths) == 1:
        single = clip_paths[0]
        if audio_path and audio_path.exists() and out_path:
            cmd = [FFMPEG_EXE, "-y", "-i", str(single), "-i", str(audio_path),
                   "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(out_path)]
            subprocess.run(cmd, check=True)
            return out_path
        return single

    work = clip_paths[0].parent / "xfade_work"
    work.mkdir(exist_ok=True)
    cur = work / "cur_000.mp4"
    shutil.copy2(clip_paths[0], cur)

    def _xfade_two(a: Path, b: Path, outp: Path, frames: int):
        dur = frames / float(fps)
        accel = ["-hwaccel", "cuda"] if USE_HWACCEL else []
        cmdline = [
            FFMPEG_EXE, *accel, "-y",
            "-i", str(a), "-i", str(b),
            "-filter_complex",
            f"[0:v][1:v]xfade=transition=fade:duration={dur}:offset=PTS-STARTPTS[v];"
            f"[0:a][1:a]acrossfade=d={dur}[a]",
            "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", str(outp)
        ]
        subprocess.run(cmdline, check=True)

    for i in range(1, len(clip_paths)):
        tmp_out = work / f"cur_{i:03d}.mp4"
        _xfade_two(cur, clip_paths[i], tmp_out, overlap_frames)
        cur = tmp_out

    if out_path is None:
        # 기본 산출 위치: 상위 폴더 + FINAL_OUT 파일명(혹은 디렉토리 규칙에 맞게)
        out_path = clip_paths[0].parent / (Path(FINAL_OUT).name if FINAL_OUT else "final.mp4")

    if audio_path and audio_path.exists():
        cmd = [FFMPEG_EXE, "-y", "-i", str(cur), "-i", str(audio_path),
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(out_path)]
        subprocess.run(cmd, check=True)
        return out_path
    else:
        shutil.copy2(cur, out_path)
        return out_path



# === ADD: video_build.py ===


def build_missing_images_from_story(
    story_path: str | Path,
    *,
    ui_width: int,
    ui_height: int,
    steps: int = 28,
    timeout_sec: int = 300,
    poll_sec: float = 1.5,
    workflow_path: str | Path | None = None,
    on_progress: Optional[Dict[str, Any] | Callable[[Dict[str, Any]], None]] = None,
) -> List[Path]:
    """
    story.json을 읽고, 각 scene의 img_file이 없으면 ComfyUI로 생성한다.
    - 워크플로: settings.JSONS_DIR / nunchaku_qwen_image_swap.json (강제)
    - 프롬프트: scene.prompt + scene.prompt_img (합침)
    - 네거티브: scene.prompt_negative or defaults.image.negative
    """


    def notify(stage: str, msg: str = "", **extra: Any) -> None:
        if on_progress:
            info: Dict[str, Any] = {"stage": stage, "msg": msg}
            info.update(extra)
            try:
                on_progress(info)
            except Exception:
                pass

    p_story = _Path(story_path).resolve()
    if not p_story.exists():
        raise FileNotFoundError(f"story.json 없음: {p_story}")

    story = load_json(p_story, {}) or {}

    # 레거시 키 정리(예: needs_character_asset)
    try:
        from utils import purge_legacy_scene_keys
        story = purge_legacy_scene_keys(story)
    except Exception:
        pass

    scenes = story.get("scenes") or []
    if not isinstance(scenes, list) or not scenes:
        raise RuntimeError("story.scenes 비어 있음")

    paths = story.get("paths") or {}
    root_dir = _Path(paths.get("root") or p_story.parent)
    imgs_dir = _Path(paths.get("imgs_dir") or "imgs")
    img_root = ensure_dir(root_dir / imgs_dir)

    # 워크플로 로드 (강제)
    try:
        from settings import JSONS_DIR  # type: ignore
        base_jsons = _Path(JSONS_DIR)
    except Exception:
        base_jsons = _Path(r"C:\my_games\shorts_make\app\jsons")

    wf_path = _Path(workflow_path) if workflow_path else (base_jsons / "nunchaku_qwen_image_swap.json")
    if not wf_path.exists():
        raise FileNotFoundError(f"필수 워크플로 없음: {wf_path}")

    try:
        graph = _load_workflow_graph(wf_path)
    except Exception as e:
        raise RuntimeError(f"워크플로 로드 실패: {e}")

    # Comfy 호스트
    try:
        base_url = _choose_host()
    except Exception:
        try:
            from settings import COMFY_HOST  # type: ignore
            base_url = str(COMFY_HOST)
        except Exception as e:
            raise RuntimeError(f"Comfy 호스트 결정 실패: {e}")

    # ── 그래프 탐색 유틸(노드 ID 달라도 동작) ─────────────
    def _find_nodes(g: dict, *, class_type: str = "", label_contains: str = "") -> list[str]:
        hits: list[str] = []
        for nid, node in g.items():
            try:
                ct = str(node.get("class_type") or "")
                lbl = str(node.get("label") or "")
                if class_type and ct != class_type:
                    continue
                if label_contains and (label_contains not in lbl):
                    continue
                hits.append(str(nid))
            except Exception:
                continue
        return hits

    def _set_input(g: dict, nid: str, key: str, val: Any) -> None:
        try:
            g[str(nid)].setdefault("inputs", {})[key] = val
        except Exception:
            pass

    # 후보 노드 (한 번만 스캔)
    latent_ids = _find_nodes(graph, class_type="EmptySD3LatentImage")
    latent_ids += _find_nodes(graph, class_type="EmptyLatentImage")
    latent_ids = list(dict.fromkeys(latent_ids))

    ksampler_ids = _find_nodes(graph, class_type="KSampler")
    ckpt_ids = _find_nodes(graph, class_type="CheckpointLoaderSimple")
    reactor_ids = [nid for nid, node in graph.items() if "ReActor" in str(node.get("label") or "")]

    # 스왑 소스 존재 여부(있으면 enable)
    def _swap_sources_available() -> bool:
        try:
            cand = list((root_dir / "swap_src").glob("*.*"))
        except OSError:
            return False
        return any(x.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} for x in cand)

    swap_ok = _swap_sources_available()

    # 전역 텍스트 필드 덮어쓰기 헬퍼
    def _override_text_inputs(g: dict, *, pos: str, neg: str) -> list[tuple[str, str, int]]:
        changed: list[tuple[str, str, int]] = []
        for nid, node in g.items():
            try:
                lbl = str(node.get("label") or "")
                ins = node.setdefault("inputs", {})
                fields = []
                if "text" in ins:   fields.append("text")
                if "text_g" in ins: fields.append("text_g")
                if "text_l" in ins: fields.append("text_l")
                if not fields:
                    continue
                is_neg = ("neg" in lbl.lower()) or ("negative" in lbl.lower())
                val = neg if is_neg else pos
                for f in fields:
                    ins[f] = val
                    changed.append((str(nid), f, len(val)))
            except Exception:
                continue
        return changed

    # 음악 대기/큐 모니터를 재사용하되, 로그 프리픽스를 [IMG]로 전환
    try:
        from music_gen import _submit_and_wait as _wait_core  # 재사용
    except Exception:
        from app.music_gen import _submit_and_wait as _wait_core  # type: ignore

    def _wait_img(url: str, g: dict, *, timeout: int, poll: float, on_progress: Callable[[Dict[str, Any]], None]) -> dict:
        def relay(info: Dict[str, Any]) -> None:
            if not callable(on_progress):
                return
            msg = str(info.get("msg") or "")
            if msg.startswith("[MUSIC]"):
                info = dict(info)
                info["msg"] = "[IMG]" + msg[len("[MUSIC]"):]
            on_progress(info)
        return _wait_core(url, g, timeout=timeout, poll=poll, on_progress=relay)

    created: List[Path] = []
    notify("begin", f"총 {len(scenes)}개 씬 검사")

    for idx, sc in enumerate(scenes):
        try:
            img_win = sc.get("img_file") or ""
            img_path = _Path(img_win) if img_win else (img_root / f"{sc.get('id','scene')}.png")
            if img_path.exists():
                continue  # 이미 있음

            # 프롬프트 결합(이미지 전용)
            base_prompt = str(sc.get("prompt") or "").strip()
            img_prompt = str(sc.get("prompt_img") or "").strip()
            parts = [p for p in (base_prompt, img_prompt) if p]
            pos_text = " ".join(parts) if parts else "photo portrait, realistic, high quality"
            neg_text = str(
                sc.get("prompt_negative")
                or (story.get("defaults", {}).get("image", {}) or {}).get("negative")
                or "low quality, artifacts, text watermark"
            ).strip()

            # 1) 해상도/스텝 일괄 주입(모든 latent/ksampler)
            for nid in latent_ids:
                _set_input(graph, nid, "width", int(ui_width))
                _set_input(graph, nid, "height", int(ui_height))
            for nid in ksampler_ids:
                _set_input(graph, nid, "steps", int(steps))

            # 2) 스왑 enable 토글
            for nid in reactor_ids:
                _set_input(graph, nid, "enabled", bool(swap_ok))

            # 3) 프롬프트 전역 덮어쓰기
            applied = _override_text_inputs(graph, pos=pos_text, neg=neg_text)

            # ---- 캐릭터 얼굴 주입 ----
            face_path = _resolve_character_image_path(sc, story)
            applied_face_nodes = []
            if face_path:
                face_fname = _copy_to_comfy_input(face_path)  # C:\comfy310\ComfyUI\input\ 에 복사 → 파일명만
                # female_01 / male_01에 따라 힌트 줄 수도 있음 (선택)
                hint = None
                chars = sc.get("characters") or []
                if chars and isinstance(chars[0], str):
                    if "female" in chars[0].lower():
                        hint = "female"
                    elif "male" in chars[0].lower():
                        hint = "male"
                applied_face_nodes = _set_face_on_reactors(graph, face_fname, target_label_hint=hint)
                notify("cfg", f"[IMG] face source set → {face_fname} | LoadImage nodes={applied_face_nodes}")
            else:
                notify("cfg", "[IMG] face-swap skipped (no character image)")

            def _parse_first_char_index(scene) -> int:
                specs = scene.get("characters") or scene.get("character_objs") or []
                if not specs: return 0
                if isinstance(specs[0], str) and ":" in specs[0]:
                    try:
                        return int(specs[0].split(":")[1])
                    except:
                        return 0
                if isinstance(specs[0], dict):
                    try:
                        return int(specs[0].get("index", 0))
                    except:
                        return 0
                return 0

            # … 얼굴 주입 직후:
            face_idx = _parse_first_char_index(sc)
            for rnid, node in graph.items():
                if node.get("class_type") == "ReActorFaceSwap":
                    ins = node.setdefault("inputs", {})
                    for k in ("source_faces_index", "input_faces_index"):
                        if k in ins:
                            ins[k] = str(face_idx)

            # 두 변수 미리 선언해두면 깔끔
            face_fname = None
            applied_face_nodes = []

            if face_path:
                face_fname = _copy_to_comfy_input(face_path)  # 또는 네가 선택한 방식
                applied_face_nodes = _inject_face_image_to_graph(graph, face_fname)
                notify("cfg", f"[IMG] face-swap source={face_fname} -> LoadImage nodes={applied_face_nodes or ['?']}")
            else:
                notify("cfg", "[IMG] face-swap skipped (no character image)")

            # ✅ 여기서 불리언으로 정리해 ReActor enable 토글에 사용
            applied_face = bool(applied_face_nodes)  # ← 이 한 줄 때문에 NameError 해결

            # 스왑 노드 enable
            for nid, node in graph.items():
                if "ReActor" in str(node.get("label") or ""):
                    graph[nid].setdefault("inputs", {})["enabled"] = bool(face_fname and applied_face)

            # 4) 체크포인트 이름 로깅
            ckpts: list[str] = []
            for nid in ckpt_ids:
                try:
                    name = str(graph[nid].get("inputs", {}).get("ckpt_name") or "")
                    if name:
                        ckpts.append(name)
                except Exception:
                    pass

            # 주입 결과 로그
            notify(
                "cfg",
                f"[IMG] size={ui_width}x{ui_height}, steps={steps}, pos_len={len(pos_text)}, "
                f"ckpt={ckpts[:1] or ['?']}, latents={len(latent_ids)}, ksamplers={len(ksampler_ids)}, reactors={len(reactor_ids)}"
            )
            if applied:
                head = ", ".join(f"{nid}:{field}:{ln}" for nid, field, ln in applied[:6])
                more = f" (+{len(applied)-6} more)" if len(applied) > 6 else ""
                notify("cfg", f"[IMG] prompt-applied → {head}{more}")

            notify("submit", f"scene[{idx}] 제출: {pos_text[:60]}…")

            # 제출 & 대기
            hist = _wait_img(
                base_url, graph,
                timeout=timeout_sec,
                poll=poll_sec,
                on_progress=(on_progress or (lambda _info: None)),
            )

            # 출력 파싱(outputs/output 모두 대응)
            files: List[Dict[str, Any]] = []
            outputs = hist.get("outputs") or hist.get("output") or {}
            for out in outputs.values():
                if isinstance(out, dict):
                    files.extend(out.get("images", []) or [])

            if not files:
                raise RuntimeError("출력 이미지 없음")

            last = files[-1]
            subfolder = str(last.get("subfolder") or "")
            fname = str(last.get("filename") or "")
            if not fname:
                raise RuntimeError("출력 파일명 없음")

            r = requests.get(
                base_url.rstrip("/") + "/view",
                params={"filename": fname, "subfolder": subfolder},
                timeout=30,
            )
            if not r.ok:
                raise RuntimeError(f"/view 실패: {r.status_code}")

            img_path.parent.mkdir(parents=True, exist_ok=True)
            with open(img_path, "wb") as f:
                f.write(r.content)

            # story.json 즉시 반영
            if not sc.get("img_file"):
                sc["img_file"] = str(img_path)
                try:
                    save_json(p_story, story)
                    notify("save", f"story.json 갱신: scenes[{idx}].img_file")
                except Exception as e:
                    notify("warn", f"story 저장 실패: {type(e).__name__}: {e}")

            created.append(img_path)
            notify("scene-done", f"scene[{idx}] → {img_path.name}")

        except requests.RequestException as e:
            notify("scene-error", f"scene[{idx}] 네트워크 오류: {e.__class__.__name__}: {e}")
        except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as e:
            notify("scene-error", f"scene[{idx}] 실패: {e.__class__.__name__}: {e}")

    # 요청 개수 리포트
    req_total = 0
    for s in scenes:
        img_win = s.get("img_file") or ""
        p = _Path(img_win) if img_win else (img_root / f"{str(s.get('id', 'scene'))}.png")
        if not p.exists():
            req_total += 1

    notify("end", f"생성 {len(created)}개 / 요청 {req_total}개")
    return created

# 파일 상단 import 근처에 추가

def _set_face_on_reactors(graph: dict, face_filename_in_input: str, *, target_label_hint: str | None = None) -> list[str]:
    """
    ReActor 노드(22/23/28)가 참조하는 LoadImage(19/24/25)의 inputs['image']를 face_filename_in_input으로 교체.
    그리고 '사용할' ReActor만 enabled=True, 나머지는 False로 둔다.
    target_label_hint에 "female"/"male" 등 힌트를 주면 그에 맞는 리액터를 우선 활성화.
    반환: 실제로 파일명을 바꾼 LoadImage 노드 ID 목록
    """
    applied_load_ids = []
    # 1) ReActor -> source LoadImage 매핑 수집
    reactors = []
    for nid, node in graph.items():
        if node.get("class_type") == "ReActorFaceSwap":
            ins = node.get("inputs", {}) or {}
            src = ins.get("source_image")
            if isinstance(src, list) and len(src) == 2:
                load_id = str(src[0])
                reactors.append((str(nid), load_id, str(node.get("_meta", {}).get("title", ""))))

    # 2) 우선순위 결정: 라벨 힌트가 있으면 그 라벨이 붙은 ReActor를 우선 활성화
    def choose_active(reactors):
        if not reactors:
            return None
        if target_label_hint:
            tl = target_label_hint.lower()
            for rnid, load_id, title in reactors:
                if tl in title.lower():
                    return rnid
        # 힌트 없으면 "마지막(가장 downstream)"을 활성화
        return reactors[-1][0]

    active_reactor = choose_active(reactors)

    # 3) LoadImage 파일명 교체 & enable 토글
    for rnid, load_id, _title in reactors:
        # LoadImage 파일명 교체
        if load_id in graph and "LoadImage" in str(graph[load_id].get("class_type")):
            graph[load_id].setdefault("inputs", {})["image"] = face_filename_in_input
            applied_load_ids.append(load_id)
        # enable/disable
        graph[rnid].setdefault("inputs", {})["enabled"] = (rnid == active_reactor)

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





