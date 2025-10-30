# -*- coding: utf-8 -*-
# i2v 분할/실행/합치기(교차 페이드) + 누락 이미지 생성 (ComfyUI 연동)
from __future__ import annotations

import subprocess
from typing import List, Tuple, Optional, Dict, Any, Callable
import re
import typing
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
# video_build.py 파일의 build_shots_with_i2v 함수 전체를 아래 코드로 교체하세요.
def build_shots_with_i2v(
        project_dir: str,
        total_frames: int,  # 이 인자는 현재 사용되지 않으나, 호환성을 위해 유지합니다.
        *,  # 키워드 전용 인자 시작
        ui_width: Optional[int] = None,
        ui_height: Optional[int] = None,
        ui_fps: Optional[int] = None,
        ui_steps: Optional[int] = None,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None
) -> None:
    """
    각 씬을 i2v로 생성합니다. UI 설정을 워크플로우에 적용하고, 파일 접두사에 현재 날짜를 사용합니다.
    - [수정됨 v2] ComfyUI 출력(outputs)의 'videos' 또는 'gifs' 리스트에서 '마지막' .mp4 파일을 선택합니다.
    - [수정됨] KSampler 노드의 noise_seed를 매번 랜덤값으로 변경하여 ComfyUI 캐시 방지.
    - [수정됨] VHS_VideoCombine 노드의 filename_prefix에 현재 날짜(YYYY-MM-DD)를 적용합니다.
    - [수정됨] 대기 중 로그 메시지를 50번에 1번만 전달하도록 조절합니다.
    - [수정됨] 대기 시간을 3600초(1시간)로 설정합니다.
    """
    # 필요 모듈 임포트
    import json as json_mod
    import shutil
    import time
    import traceback
    import datetime  # 날짜 처리
    import random  # <-- 시드 생성을 위해 random 임포트
    from pathlib import Path
    from typing import Any, Dict, List, Optional, Callable, Tuple

    # 써드파티 라이브러리 (requests)
    try:
        import requests
    except ImportError as e_req_import_vfinal:  # 변수명 변경
        # requests 없으면 진행 불가
        raise ImportError(
            "requests 라이브러리가 필요합니다. 'pip install requests'"
        ) from e_req_import_vfinal

    # --- 로그 콜백 ---
    def _notify(msg: str) -> None:
        """진행률 콜백을 호출하는 내부 함수. None 방지, 접두사 관리 포함."""
        if on_progress:
            try:
                prefix = "[I2V] "
                msg_str = str(msg or "").strip()  # None 방지 및 양끝 공백 제거
                if not msg_str:
                    return  # 빈 메시지 무시

                msg_to_send: str  # 타입 명시
                if not msg_str.startswith("[I2V]"):
                    msg_to_send = prefix + msg_str
                else:
                    # 이미 접두사 있으면 내용만 추출하여 재조합 (중복 공백 방지)
                    content_part = msg_str[msg_str.find("]") + 1:].strip()
                    if not content_part:
                        return  # 내용 없는 메시지 무시
                    msg_to_send = prefix.strip() + " " + content_part

                on_progress({"msg": msg_to_send})
            except Exception as e_notify_vfinal:  # 구체적인 예외 대신 Exception 사용 (콜백 오류는 로깅만)
                # 콜백 자체 오류 로깅 (작업 중단 방지)
                print(f"[경고] on_progress 콜백 오류: {type(e_notify_vfinal).__name__}: {e_notify_vfinal}")

    # --- 입력값 검증 및 로깅 ---
    _notify(f"영상 생성 시작: project_dir='{project_dir}'")
    _notify(f"UI 설정: W={ui_width}, H={ui_height}, FPS={ui_fps}, Steps={ui_steps}")

    # 프로젝트 경로 객체화 및 유효성 검증
    try:
        pdir = Path(project_dir).resolve(strict=True)
        if not pdir.is_dir():
            raise NotADirectoryError(f"프로젝트 경로가 디렉토리가 아닙니다: {pdir}")
    except FileNotFoundError:
        _notify(f"[오류] 프로젝트 디렉토리를 찾을 수 없습니다: {project_dir}")
        return
    except NotADirectoryError as e_not_dir_vfinal:  # 구체적인 예외 타입
        _notify(f"[오류] {e_not_dir_vfinal}")
        return
    except OSError as e_os_proj_vfinal:  # 구체적인 예외 타입
        _notify(f"[오류] 프로젝트 경로 접근 오류: {e_os_proj_vfinal}")
        return
    except Exception as e_proj_path_vfinal:  # 그 외 경로 관련 오류
        _notify(f"[오류] 프로젝트 경로 처리 중 예상치 못한 오류: {e_proj_path_vfinal}")
        return

    p_video = pdir / "video.json"

    # --- 유틸 안전 바인딩 ---
    load_json_fn: Callable[[Path, Any], Any]
    ensure_dir_fn: Callable[[Path], Path]
    try:
        from app.utils import load_json as load_json_fn, ensure_dir as ensure_dir_fn  # type: ignore
    except (ImportError, ModuleNotFoundError):  # 구체적인 예외 타입
        try:
            from utils import load_json as load_json_fn, ensure_dir as ensure_dir_fn  # type: ignore
        except (ImportError, ModuleNotFoundError) as e_util_import_vfinal:  # 구체적인 예외 타입
            # 필수 유틸 로드 실패 시 에러
            raise ImportError(f"필수 유틸리티(load_json, ensure_dir) 로드 실패: {e_util_import_vfinal}") from e_util_import_vfinal

    # --- 안전하게 노드 입력값 설정 ---
    def _set_input_safe(graph_dict: Dict, node_id: str, input_key: str, value: Any) -> bool:
        """지정된 노드의 입력값을 안전하게 설정하고 성공 여부 반환."""
        node_id_str = str(node_id or "")
        input_key_str = str(input_key or "")
        if not node_id_str or not input_key_str:
            _notify(f"[경고] 노드 ID({node_id_str}) 또는 키({input_key_str}) 유효하지 않음.")
            return False
        try:
            target_node = graph_dict.get(node_id_str)  # get으로 안전하게 접근
            if isinstance(target_node, dict):
                inputs_dict = target_node.setdefault("inputs", {})
                if not isinstance(inputs_dict, dict):
                    _notify(f"[경고] 노드 {node_id_str} 'inputs' 필드 오류.")
                    return False

                current_value = inputs_dict.get(input_key_str)
                value_to_set: Any
                # 타입 추정 및 변환 (int 우선)
                target_type_int = isinstance(current_value, int) or input_key_str in (
                    "steps", "frame_rate", "width", "height", "length", "seed", "loop_count", "crf", "noise_seed"
                # noise_seed 추가
                )
                target_type_float = isinstance(current_value, float) or input_key_str in ("cfg",
                                                                                          "strength_model")  # 추가적인 실수 키

                try:
                    if target_type_int:
                        value_to_set = int(value)
                    elif target_type_float:
                        value_to_set = float(value)
                    else:
                        value_to_set = value  # 다른 타입 (str, bool 등)
                except (ValueError, TypeError):
                    _notify(f"[경고] 노드 {node_id_str}.{input_key_str} 값({value}) 숫자 변환 실패.")
                    value_to_set = value  # 변환 실패 시 원본 유지

                inputs_dict[input_key_str] = value_to_set
                return True
            else:
                _notify(f"[경고] 노드 ID '{node_id_str}' 없거나 dict 아님.")
                return False
        except Exception as e_set_input_vfinal:  # 그 외 예외
            _notify(f"[오류] 노드 {node_id_str}.{input_key_str} 설정 중 오류: {e_set_input_vfinal}")
            print(traceback.format_exc())  # 상세 로그
            return False

    # --- video.json 로드 ---
    try:
        video = load_json_fn(p_video, {}) or {}
        if not isinstance(video, dict):
            _notify(f"[오류] video.json 형식 오류: {p_video}")
            return
    except Exception as e_load_vjson_vfinal:  # 그 외 예외
        _notify(f"[오류] video.json 로드 실패: {e_load_vjson_vfinal}")
        return

    scenes = list(video.get("scenes") or [])
    if not scenes:
        _notify("video.json에 생성할 씬('scenes') 없음.")
        return

    clips_dir = ensure_dir_fn(pdir / "clips")

    # --- 설정 모듈 로드 및 값 안전하게 가져오기 ---
    s_mod_vfinal: Any  # 변수명 변경 및 타입 명시
    try:
        import settings as settings_mod  # type: ignore
        s_mod_vfinal = settings_mod
    except ImportError:
        try:
            from app import settings as settings_mod_app  # type: ignore
            s_mod_vfinal = settings_mod_app
        except ImportError:
            print("[경고] 설정 모듈 로드 실패. 기본값 사용.")

            # 폴백 객체 정의 (필요한 속성만 포함)
            class SettingsFallback:
                JSONS_DIR = r"C:\my_games\shorts_make\app\jsons"
                I2V_WORKFLOW = None
                COMFY_INPUT_DIR = r"C:\my_games\shorts_make\app\comfy_inputs"
                COMFY_HOST = "http://127.0.0.1:8188"

            s_mod_vfinal = SettingsFallback()

    # getattr으로 안전하게 속성 접근
    _jsons_dir_setting_vfinal = getattr(s_mod_vfinal, "JSONS_DIR", r"C:\my_games\shorts_make\app\jsons")
    _i2v_wf_setting_vfinal = getattr(s_mod_vfinal, "I2V_WORKFLOW", None)
    _comfy_in_setting_vfinal = getattr(s_mod_vfinal, "COMFY_INPUT_DIR", r"C:\my_games\shorts_make\app\comfy_inputs")
    _comfy_host_setting_vfinal = getattr(s_mod_vfinal, "COMFY_HOST", "http://127.0.0.1:8188")

    base_jsons = Path(str(_jsons_dir_setting_vfinal))
    comfy_input_dir = Path(str(_comfy_in_setting_vfinal))
    comfy_input_dir.mkdir(parents=True, exist_ok=True)

    base_url = str(video.get("comfy_host") or _comfy_host_setting_vfinal).rstrip("/")

    defaults_v = video.get("defaults") or {}
    defaults_i2v = defaults_v.get("i2v") or {}

    # --- 워크플로우 경로 결정 ---
    wf_path: Optional[Path] = None
    if _i2v_wf_setting_vfinal:
        cand1_vfinal = Path(str(_i2v_wf_setting_vfinal))
        if not cand1_vfinal.is_absolute():
            cand1_vfinal = base_jsons / cand1_vfinal.name
        if cand1_vfinal.is_file():
            wf_path = cand1_vfinal
    if wf_path is None:
        wf_from_video_vfinal = str(defaults_i2v.get("workflow") or "").strip()
        if wf_from_video_vfinal:
            cand2_vfinal = Path(wf_from_video_vfinal)
            if not cand2_vfinal.is_absolute():
                cand2_vfinal = base_jsons / cand2_vfinal.name
            if cand2_vfinal.is_file():
                wf_path = cand2_vfinal
    if wf_path is None:
        wf_path_fallback_vfinal = base_jsons / "guff_movie.json"
        if wf_path_fallback_vfinal.is_file():
            wf_path = wf_path_fallback_vfinal
        else:
            tried_paths_vfinal = [
                str(_i2v_wf_setting_vfinal or 'N/A'),
                str(defaults_i2v.get('workflow') or 'N/A'),
                str(wf_path_fallback_vfinal)
            ]
            raise FileNotFoundError(f"워크플로 파일을 찾을 수 없습니다: {tried_paths_vfinal}")

    _notify(f"사용할 워크플로우: {wf_path.name} (경로: {wf_path})")

    # --- 워크플로우 원본 로드 ---
    try:
        with open(wf_path, "r", encoding="utf-8") as f_graph_vfinal:
            graph_origin: Dict[str, Any] = json_mod.load(f_graph_vfinal)
    except (OSError, json_mod.JSONDecodeError) as e_load_wf_vfinal:
        raise RuntimeError(f"워크플로우 로드 실패: {wf_path} ({e_load_wf_vfinal})") from e_load_wf_vfinal
    except Exception as e_load_wf_unknown_vfinal:
        raise RuntimeError(
            f"워크플로우 로드 중 예상치 못한 오류: {wf_path} ({e_load_wf_unknown_vfinal})") from e_load_wf_unknown_vfinal

    # --- 제출/대기 코어 로드 ---
    submit_and_wait_fn: Optional[Callable] = None
    load_err_msg_vfinal: Optional[str] = None
    try:
        from app.audio_sync import _submit_and_wait as submit_and_wait_fn  # type: ignore
    except Exception:
        try:
            from audio_sync import _submit_and_wait as submit_and_wait_fn  # type: ignore
        except Exception as e_load_wait_main:
            load_err_msg_vfinal = str(e_load_wait_main)

    if submit_and_wait_fn is None:
        raise RuntimeError(f"_submit_and_wait 함수 로드 실패: {load_err_msg_vfinal or 'unknown'}")

    # --- 현재 날짜 문자열 생성 ---
    try:
        current_date_str = datetime.date.today().strftime("%Y-%m-%d")
    except Exception as e_date_vfinal:
        _notify(f"[경고] 날짜 가져오기 실패: {e_date_vfinal}. 기본 접두사 사용.")
        current_date_str = "YYYY-MM-DD"

    # --- 씬 순회 및 생성 ---
    for idx, sc in enumerate(scenes):
        if not isinstance(sc, dict):
            _notify(f"[경고] scenes[{idx}] 항목 오류. 건너<0xEB><0x9B><0x84>니다.")
            continue

        sid = str(sc.get("id") or f"scene_{idx:03d}").strip() or f"scene_{idx:03d}"

        # --- 씬별 스킵 확인 ---
        clip_mp4 = clips_dir / f"{sid}.mp4"
        try:
            if clip_mp4.is_file() and clip_mp4.stat().st_size > 1024:  # 1KB 이상
                _notify(f"skip scene={sid} → exists {clip_mp4.name} ({clip_mp4.stat().st_size / 1024:.1f} KB)")
                continue
        except Exception as e_skip_check_local:
            _notify(f"[경고] 스킵 확인 중 오류 (씬 {sid}): {e_skip_check_local}. 생성 시도.")

        # --- 그래프 복제 ---
        try:
            graph = json_mod.loads(json_mod.dumps(graph_origin))
        except Exception as e_clone_main:
            _notify(f"[오류] 워크플로우 복제 실패 (씬 {sid}): {e_clone_main}")
            continue

        # --- 입력 이미지 찾기 및 준비 ---
        img_path: Optional[Path] = None
        server_image_name: Optional[str] = None
        img_from_scene_local = str(sc.get("img_file") or "").strip()
        if img_from_scene_local:
            p_img_local = Path(img_from_scene_local)
            if not p_img_local.is_absolute():
                paths_v_local_img = video.get("paths") or {}
                root_dir_str_local_img = str(paths_v_local_img.get("root") or pdir)
                root_dir_local_img = Path(root_dir_str_local_img)
                imgs_dir_name_local_img = str(paths_v_local_img.get("imgs_dir") or "imgs")
                p_img_local = root_dir_local_img / imgs_dir_name_local_img / p_img_local.name
            try:
                if p_img_local.is_file() and p_img_local.stat().st_size > 0:
                    img_path = p_img_local
            except Exception as e_img1_check_local:
                _notify(f"[경고] 이미지({p_img_local.name}) 상태 확인 오류: {e_img1_check_local}")

        if img_path is None:
            paths_v2_local_img = video.get("paths") or {}
            root_dir2_str_local_img = str(paths_v2_local_img.get("root") or pdir)
            root_dir2_local_img = Path(root_dir2_str_local_img)
            imgs_dir_name2_local_img = str(paths_v2_local_img.get("imgs_dir") or "imgs")
            cand_png_local_img = root_dir2_local_img / imgs_dir_name2_local_img / f"{sid}.png"
            try:
                if cand_png_local_img.is_file() and cand_png_local_img.stat().st_size > 0:
                    img_path = cand_png_local_img
            except Exception as e_img2_check_local:
                _notify(f"[경고] 기본 이미지({cand_png_local_img.name}) 상태 확인 오류: {e_img2_check_local}")

        if img_path is None:
            _notify(f"scene={sid} 입력 이미지를 찾을 수 없음 → 건너<0xEB><0x9B><0x84>니다.")
            continue

        try:
            server_image_name = img_path.name
            dest_path_vfinal = comfy_input_dir / server_image_name
            should_copy_vfinal = True
            if dest_path_vfinal.exists():
                try:
                    if dest_path_vfinal.stat().st_mtime >= img_path.stat().st_mtime:
                        should_copy_vfinal = False
                except FileNotFoundError:
                    should_copy_vfinal = True
                except OSError as e_stat_comp_vfinal:
                    _notify(f"[경고] 파일 시간 비교 오류({dest_path_vfinal.name}): {e_stat_comp_vfinal}.")

            if should_copy_vfinal:
                shutil.copy2(str(img_path), str(dest_path_vfinal))
                _notify(f"이미지 복사 완료: {server_image_name} -> {comfy_input_dir.name}")

        except Exception as e_copy_upload_local:
            _notify(f"[경고] 이미지 준비 실패 ({img_path.name}): {e_copy_upload_local}, 업로드/다음 단계 시도.")

        if not server_image_name:
            _notify(f"[오류] 이미지 이름 설정 불가 (씬 {sid}). 다음 씬으로.")
            continue

        # --- 워크플로우 노드 값 수정 ---
        nodes_updated_count = 0

        # 1. LoadImage 주입
        loadimage_nodes_found: List[Tuple[str, Dict]] = []
        for nid_lnf, node_lnf in graph.items():
            try:
                if isinstance(node_lnf, dict) and str(node_lnf.get("class_type", "")).strip() == "LoadImage":
                    loadimage_nodes_found.append((str(nid_lnf), node_lnf))
            except Exception:
                continue  # 노드 구조 오류 무시
        if not loadimage_nodes_found:
            _notify(f"[경고] LoadImage 노드 없음 (씬 {sid}).")
        else:
            for nid_lns, _ in loadimage_nodes_found:
                if _set_input_safe(graph, nid_lns, "image", server_image_name):
                    nodes_updated_count += 1
                    _notify(f"LoadImage({nid_lns}) 이미지: {server_image_name}")

        # 2. 새 난수 시드 생성 (캐시 방지)
        try:
            new_seed_value = random.randint(0, 999999999999999)
        except Exception:
            new_seed_value = int(time.time() * 1000000) % 999999999999999

        sampler_node_ids_for_seed = ["13", "14"]  # KSamplerAdvanced 노드 ID
        seed_update_count = 0
        for nid_seed_set in sampler_node_ids_for_seed:
            if _set_input_safe(graph, nid_seed_set, "noise_seed", new_seed_value):
                seed_update_count += 1
        if seed_update_count > 0:
            nodes_updated_count += seed_update_count
            _notify(f"Sampler({', '.join(sampler_node_ids_for_seed)}) Seed: {new_seed_value}")

        # 3. UI 설정값 주입 (W, H, FPS, Steps, Length)
        resize_node_id = "24"
        combine_node_id = "21"
        sampler_node_ids = ["13", "14"]  # 스텝 설정용
        i2v_node_id = "25"  # WanImageToVideo

        resize_update_count = 0
        if ui_width is not None and ui_width > 0 and _set_input_safe(graph, resize_node_id, "width", ui_width):
            resize_update_count += 1
        if ui_height is not None and ui_height > 0 and _set_input_safe(graph, resize_node_id, "height", ui_height):
            resize_update_count += 1
        if resize_update_count > 0:
            nodes_updated_count += resize_update_count
            w_log_vfinal = ui_width if ui_width is not None and ui_width > 0 else '기본값'
            h_log_vfinal = ui_height if ui_height is not None and ui_height > 0 else '기본값'
            _notify(f"Resize({resize_node_id}) 크기: {w_log_vfinal}x{h_log_vfinal}")

        if ui_fps is not None and ui_fps > 0 and _set_input_safe(graph, combine_node_id, "frame_rate", ui_fps):
            nodes_updated_count += 1
            _notify(f"Combine({combine_node_id}) FPS: {ui_fps}")

        steps_update_count = 0
        if ui_steps is not None and ui_steps > 0:
            for nid_sus in sampler_node_ids:
                if _set_input_safe(graph, nid_sus, "steps", ui_steps):
                    steps_update_count += 1
        if steps_update_count > 0:
            nodes_updated_count += steps_update_count
            _notify(f"Sampler({', '.join(sampler_node_ids)}) Steps: {ui_steps}")

        try:
            scene_duration = float(sc.get("duration", 1.0))
            target_fps_val: int = 16
            if ui_fps is not None and ui_fps > 0:
                target_fps_val = ui_fps
            else:
                try:
                    fps_wf = graph.get(combine_node_id, {}).get("inputs", {}).get("frame_rate")
                    if fps_wf is not None:
                        fps_int_wf = int(fps_wf)
                        if fps_int_wf > 0:
                            target_fps_val = fps_int_wf
                except Exception:
                    pass  # 오류 시 기본값 16 유지
            frame_length = max(1, int(round(scene_duration * target_fps_val)))
            if _set_input_safe(graph, i2v_node_id, "length", frame_length):
                nodes_updated_count += 1
                _notify(f"I2V({i2v_node_id}) Length: {frame_length}f ({scene_duration:.2f}s @ {target_fps_val}fps)")
        except Exception as e_len_main:
            _notify(f"[경고] 프레임 길이 설정 오류 (씬 {sid}): {e_len_main}")

        # 4. 프롬프트 주입
        positive_prompt = str(sc.get("prompt_movie") or sc.get("prompt") or "")
        negative_prompt = str(sc.get("prompt_negative") or "")
        pos_node_id = "22"
        neg_node_id = "23"
        prompt_update_count = 0
        if positive_prompt and pos_node_id in graph and _set_input_safe(graph, pos_node_id, "text", positive_prompt):
            prompt_update_count += 1
        if negative_prompt and neg_node_id in graph and _set_input_safe(graph, neg_node_id, "text", negative_prompt):
            prompt_update_count += 1
        if prompt_update_count > 0:
            nodes_updated_count += prompt_update_count
            _notify(f"프롬프트({pos_node_id}/{neg_node_id}) 설정 완료.")

        # 5. 파일 접두사(filename_prefix)에 현재 날짜 적용
        combine_node_id_prefix = "21"
        if combine_node_id_prefix in graph:
            original_prefix_vfinal = str(
                graph.get(combine_node_id_prefix, {}).get("inputs", {}).get("filename_prefix", "unknown_prefix_"))
            suffix_part_vfinal = original_prefix_vfinal.split("/", 1)[
                -1] if "/" in original_prefix_vfinal else original_prefix_vfinal
            if not suffix_part_vfinal or suffix_part_vfinal.isdigit() or suffix_part_vfinal.replace('-',
                                                                                                    '').isdigit() or (
                    suffix_part_vfinal.endswith('_') and suffix_part_vfinal[:-1].isdigit()):
                suffix_part_vfinal = "wan22_"
            new_prefix_vfinal = f"{current_date_str}/{suffix_part_vfinal}"
            if _set_input_safe(graph, combine_node_id_prefix, "filename_prefix", new_prefix_vfinal):
                nodes_updated_count += 1
                _notify(f"Combine({combine_node_id_prefix}) Prefix: {new_prefix_vfinal}")
            else:
                _notify(f"[경고] Prefix 설정 실패.")

        # --- 로그 조절 콜백 래퍼 ---
        _wait_counter = 0

        def _throttled_relay(prog: Dict[str, Any]) -> None:
            nonlocal _wait_counter
            _wait_counter += 1
            if _wait_counter % 50 == 0:
                msg_val = str(prog.get("msg") or "")
                if msg_val:
                    _notify(msg_val)

        # --- 워크플로우 실행 ---
        _notify(f"워크플로우 제출 시작 (씬 {sid}, {nodes_updated_count}개 노드 업데이트)")
        result: Dict[str, Any] = {}
        try:
            result = submit_and_wait_fn(
                base_url, graph,
                timeout=int(defaults_i2v.get("timeout_sec") or 3600),  # 1시간
                poll=float(defaults_i2v.get("poll_sec") or 1.5),
                on_progress=_throttled_relay
            )
        except TimeoutError as e_timeout_vfinal:
            _notify(f"[오류] 워크플로우 시간 초과 (씬 {sid}): {e_timeout_vfinal}")
            _notify(" -> ComfyUI 콘솔에서 OutOfMemory 등의 오류가 있는지 확인하세요.")
            continue
        except Exception as e_submit_vfinal:
            _notify(f"[오류] 워크플로우 제출/대기 실패 (씬 {sid}): {e_submit_vfinal}")
            print(traceback.format_exc())
            continue

        # --- [수정] 결과 딕셔너리 구조 상세 디버깅 ---
        _notify(f"[DEBUG] ComfyUI 작업 완료. API 응답(result) 수신.")
        outputs_dict_debug = result.get("outputs") or {}
        if not outputs_dict_debug:
            _notify(f"[DEBUG] 'outputs' 딕셔너리가 비어있거나 존재하지 않습니다.")
        elif not isinstance(outputs_dict_debug, dict):
            _notify(f"[DEBUG] 'outputs'가 딕셔너리가 아님 (타입: {type(outputs_dict_debug)})")
        else:
            _notify(f"[DEBUG] 'outputs' 딕셔너리 키 (노드 ID): {list(outputs_dict_debug.keys())}")
            for node_id_debug, node_output_debug in outputs_dict_debug.items():
                if isinstance(node_output_debug, dict):
                    # 노드 출력의 키만 로깅 (예: 'videos', 'images', 'ui', 'files' 등)
                    _notify(f"[DEBUG] -> 노드 {node_id_debug}의 출력 키: {list(node_output_debug.keys())}")
                    # VHS_VideoCombine('21') 노드의 출력을 더 자세히 로깅
                    if node_id_debug == "21":
                        # node_output_debug이 너무 클 수 있으므로 str() 변환하여 일부만 로깅
                        output_str_debug = str(node_output_debug)
                        _notify(f"[DEBUG] --> 노드 21 (VHS_VideoCombine) 상세 (일부): {output_str_debug[:500]}...")
                else:
                    _notify(f"[DEBUG] -> 노드 {node_id_debug}의 출력이 dict가 아님 (타입: {type(node_output_debug)})")
        # --- [수정] 디버깅 로그 끝 ---

        # --- ▼▼▼ [수정 v2] 결과 처리 (마지막 .mp4 파일 선택) ▼▼▼ ---
        valid_mp4_files: List[Dict[str, Any]] = []
        all_outputs: List[Dict[str, Any]] = []

        outputs_dict = result.get("outputs") or {}
        if isinstance(outputs_dict, dict):
            for node_output_id, out_value_vfinal in outputs_dict.items():
                if isinstance(out_value_vfinal, dict):

                    # 'gifs' 또는 'videos' 키에서 MP4 파일 목록 수집
                    vids_list_vfinal = out_value_vfinal.get("videos") or out_value_vfinal.get("gifs")

                    if isinstance(vids_list_vfinal, list):
                        all_outputs.extend(vids_list_vfinal)
                        for vid_info in vids_list_vfinal:
                            if not isinstance(vid_info, dict):
                                continue
                            fname_vid_check = str(vid_info.get("filename") or "").lower()
                            type_vid_check = str(vid_info.get("type") or "").lower()

                            # 'temp' 타입도 허용 ('preview'만 제외)
                            if fname_vid_check.endswith(".mp4") and type_vid_check != "preview":
                                # [수정] 발견 시 'break' 대신 리스트에 '추가'
                                valid_mp4_files.append(vid_info)
                                _notify(f"MP4 후보 발견 (노드 {node_output_id}): {fname_vid_check} (type: {type_vid_check})")
                        # (참고) 안쪽/바깥쪽 break 문이 여기서 제거되었습니다.

                    # 'images' 키 확인 (폴백용)
                    imgs_list_vfinal = out_value_vfinal.get("images")
                    if isinstance(imgs_list_vfinal, list):
                        all_outputs.extend(imgs_list_vfinal)

                    # 'ui' 키 확인 (폴백용)
                    ui_dict_vfinal = out_value_vfinal.get("ui")
                    if isinstance(ui_dict_vfinal, dict):
                        for ui_key, ui_value in ui_dict_vfinal.items():
                            if isinstance(ui_value, list):
                                all_outputs.extend(ui_value)

                    # 'files' 키 확인 (폴백용)
                    files_list_vfinal = out_value_vfinal.get("files")
                    if isinstance(files_list_vfinal, list):
                        all_outputs.extend(files_list_vfinal)

        # [수정] mp4 후보가 있으면 '마지막' 파일을 선택
        target_file_info: Optional[Dict[str, Any]] = None
        if valid_mp4_files:
            target_file_info = valid_mp4_files[-1]  # <-- 마지막 파일 선택
            _notify(f"최종 MP4 파일 선택: {target_file_info.get('filename')}")
        elif all_outputs:
            target_file_info = all_outputs[-1]  # (기존 폴백)
            _notify(f"MP4 없음. 폴백 파일 선택: {target_file_info.get('filename')}")
        else:
            target_file_info = None  # (기존 폴백)

        # --- ▲▲▲ [수V2] 결과 처리 수정 완료 ▲▲▲ ---

        if target_file_info is None:
            _notify(f"처리할 출력 파일 정보 없음 (씬 {sid})")
            continue

        subfolder = str(target_file_info.get("subfolder") or "")
        fname = str(target_file_info.get("filename") or "")
        file_type = str(target_file_info.get("type") or "output")  # 기본값 output

        if not fname:
            _notify(f"출력 파일명 없음 (씬 {sid})")
            continue

        _notify(f"결과 파일 선택: filename='{fname}', type='{file_type}' (씬 {sid})")

        # --- 파일 다운로드 및 저장 ---
        save_path: Optional[Path] = None
        try:
            resp_view = requests.get(
                base_url + "/view",
                params={"filename": fname, "subfolder": subfolder, "type": file_type},
                timeout=300
            )
            resp_view.raise_for_status()

            save_path = clip_mp4  # 목표 경로는 항상 {id}.mp4

            # .mp4 파일이 아닐 경우 저장 거부
            if not fname.lower().endswith(".mp4"):
                _notify(f"[오류] 결과 파일이 .mp4가 아닙니다! (실제: {fname}).")
                _notify(" -> ComfyUI 워크플로우의 마지막 노드가 .mp4를 출력하는지 확인하세요.")
                continue  # .png 등을 .mp4로 저장하지 않고 다음 씬으로

            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f_out_vfinal:
                f_out_vfinal.write(resp_view.content)

            try:
                file_size_bytes = save_path.stat().st_size
                _notify(f"파일 저장 완료: {save_path.name} ({file_size_bytes / 1024:.1f} KB) (씬 {sid})")
            except OSError as e_stat_final:
                _notify(f"[경고] 저장된 파일 크기 확인 실패 (씬 {sid}): {e_stat_final}")
                _notify(f"파일 저장 시도 완료: {save_path.name} (씬 {sid})")

        except requests.RequestException as e_req_final:
            _notify(f"[오류] 결과 파일 다운로드 실패 (씬 {sid}): {e_req_final}")
            continue
        except OSError as e_save_final:
            save_path_str = str(save_path) if save_path else "알 수 없음"
            _notify(f"[오류] 결과 파일 저장 실패 (씬 {sid}, 경로 {save_path_str}): {e_save_final}")
            continue
        except Exception as e_proc_final:
            _notify(f"[오류] 결과 처리 중 예상치 못한 오류 (씬 {sid}): {e_proc_final}")
            print(traceback.format_exc())
            continue

    # --- 최종 완료 로그 ---
    _notify("모든 씬 처리 완료.")


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
            sid = str(sc.get("id") or f"scene_{idx:03d}")
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
    overlap_frames_val: int = int(movie_defaults.get("overlap_frames") or 12)

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
