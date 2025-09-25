# -*- coding: utf-8 -*-
"""
ACE-Step(ComfyUI) 음악 생성 — 태그/길이 병합·주입 + 포맷 보장(노드 적응 + 후처리 트랜스코딩)

기능 요약
- project.json의 가사/태그/길이를 주입하여 ComfyUI 워크플로 실행
- LyricsLangSwitch(가사), TextEncodeAceStepAudio(tags), EmptyAceStepLatentAudio(seconds) 주입
- KSampler seed 랜덤화로 변주
- SaveAudio 계열 filename_prefix 고정(프로젝트별 하위폴더: shorts_make/<title>/vocal_final*)
- /history 스키마 A/B 모두 지원(+ node_errors 감지)
- 서버의 지원 노드(/object_info) 확인 → 가능한 저장 노드로 '적응'해서 class_type 변경
  * 없으면 MP3 저장으로 폴백(quality='320k')
- /view로 결과 다운로드 → settings.AUDIO_SAVE_FORMAT으로 최종 보장(필요 시 ffmpeg 트랜스코딩)
- 결과를 프로젝트 폴더에 'vocal.<fmt>'로 저장하고,
  설정(FINAL_OUT)에 지정한 사용자 폴더([title] 치환)에 자동 복사
- project.json 갱신(tags_effective, comfy_debug)
- 디버그: _prompt_sent.json, _history_raw.json 저장
"""

from __future__ import annotations
from typing import Optional, Callable, List, Tuple, Iterable
from pathlib import Path
import json
import random
import shutil
import subprocess
import requests

from typing import Any

# settings 상수(대기/폴링 주기)
from settings import ACE_WAIT_TIMEOUT_SEC, ACE_POLL_INTERVAL_SEC

# ── settings / utils 유연 임포트 ───────────────────────────────────────────────
try:
    # 패키지 실행
    from app import settings as settings   # 소문자 별칭
    from app.settings import (
        BASE_DIR, COMFY_HOST, DEFAULT_HOST_CANDIDATES,
        ACE_STEP_PROMPT_JSON, FFMPEG_EXE, FINAL_OUT
    )
    from app.utils import load_json, save_json, sanitize_title, effective_title, save_to_user_library
except ImportError:
    # 단독 실행
    import settings as settings
    from settings import (  # type: ignore
        BASE_DIR, COMFY_HOST, DEFAULT_HOST_CANDIDATES,
        ACE_STEP_PROMPT_JSON, FFMPEG_EXE, FINAL_OUT
    )
    from utils import load_json, save_json, sanitize_title, effective_title, save_to_user_library  # type: ignore

S = settings  # noqa: N816
# --- ACE-Step 대기/폴링 기본값 & 헬퍼 (설정에 없으면 이 값 사용) ---
# --- ACE-Step 대기/폴링 기본값 & 헬퍼 ---
_DEFAULT_ACE_WAIT_TIMEOUT_SEC = 900.0   # 15분
_DEFAULT_ACE_POLL_INTERVAL_SEC = 2.0    # 2초

def _ace_wait_timeout_sec():
    try:
        from app import settings as _S
    except Exception:
        _S = None
    return (
        (getattr(_S, "ACE_STEP_WAIT_TIMEOUT_SEC", None) if _S else None)
        or (getattr(_S, "ACE_WAIT_TIMEOUT_SEC", None) if _S else None)
        or _DEFAULT_ACE_WAIT_TIMEOUT_SEC
    )

def _ace_poll_interval_sec():
    try:
        from app import settings as _S
    except Exception:
        _S = None
    return (
        (getattr(_S, "ACE_STEP_POLL_INTERVAL_SEC", None) if _S else None)
        or (getattr(_S, "ACE_POLL_INTERVAL_SEC", None) if _S else None)
        or _DEFAULT_ACE_POLL_INTERVAL_SEC
    )


# ─────────────────────────────  ────────────────────────────────────

def _ensure_vocal_mp3(src: Path, proj_dir: Path, ffmpeg_exe: str = "ffmpeg") -> Path:
    """
    src(확장자 무관)를 프로젝트 폴더의 vocal.mp3로 강제 정착.
    src가 mp3면 move, 아니면 ffmpeg로 mp3 변환. 성공 시 vocal.mp3 Path 반환.
    """
    dst = proj_dir / "vocal.mp3"
    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.suffix.lower() == ".mp3":
        if src.resolve() != dst.resolve():
            try:
                dst.unlink()
            except FileNotFoundError:
                pass
            shutil.move(str(src), str(dst))
        return dst

    cp = subprocess.run(
        [ffmpeg_exe, "-y", "-i", str(src), "-vn", "-c:a", "libmp3lame", "-q:a", "2", str(dst)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if cp.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(f"mp3 변환 실패: {src}")

    try:
        src.unlink()  # 원본 정리(보관 원하면 주석 처리)
    except Exception:
        pass
    return dst

# ───────────────────────────── 디버그 ────────────────────────────────────
_LOG_PATH = Path(BASE_DIR) / "music_gen.log"  # 언제나 여기로도 기록

def _dlog(*args):
    msg = " ".join(str(a) for a in args)
    line = f"[MUSIC] {msg}"
    try:
        # 1) stdout (flush)
        print(line, flush=True)
    except Exception:
        pass
    try:
        # 2) stderr (일부 런처에서 stdout이 안 보일 때 대비)
        import sys
        print(line, file=sys.stderr, flush=True)
    except Exception:
        pass
    try:
        # 3) 파일 로그 (GUI에서도 100% 확인 가능)
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass



# ───────────────────────────── HTTP 유틸 ────────────────────────────────────
def _http_post_json(base: str, path: str, payload: dict, timeout: int = 30) -> requests.Response:
    return requests.post(base.rstrip("/") + path, json=payload, timeout=timeout)

def _http_get(base: str, path: str, timeout: int = 30, params: Optional[dict] = None) -> requests.Response:
    return requests.get(base.rstrip("/") + path, params=params or {}, timeout=timeout)

def _probe_server(base: str, timeout: int = 3) -> bool:
    for p in ("/view", "/history"):
        try:
            r = requests.get(base.rstrip("/") + p, timeout=timeout)
            if r.status_code == 200:
                return True
        except Exception:
            pass
    return False


# ───────────────────────────── 워크플로 로더 ──────────────────────────────────
def _load_workflow_graph(json_path: str | Path) -> dict:
    """
    ComfyUI용 워크플로 JSON을 로드해, /prompt에 바로 넣을 수 있는
    '노드 딕셔너리' 형태로 정규화해서 반환한다.
      - 파일 최상위에 {"prompt": {...}} 가 있을 수도/없을 수도 → 모두 처리
      - {"nodes":[{id:.., class_type:..}, ...]} 형식도 → dict로 변환
      - 각 노드에 class_type이 반드시 있도록 검증
    """
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"워크플로 JSON 없음: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1) 'prompt' 키로 한 번 감싸져 있으면 내부만 꺼냄
    g = data.get("prompt") if isinstance(data, dict) else None
    if not isinstance(g, (dict, list)):
        # prompt 키가 없으면 전체가 그래프라고 가정
        g = data

    # 2) list 형식(nodes 배열) → dict 형식으로 변환
    if isinstance(g, dict) and "nodes" in g and isinstance(g["nodes"], list):
        nodes_list = g["nodes"]
        g = {}
        for n in nodes_list:
            nid = str(n.get("id"))
            if not nid:
                raise ValueError("워크플로 형식 오류: nodes[*].id 가 없음")
            node = {k: v for k, v in n.items() if k != "id"}
            g[nid] = node

    # 3) 최종 형태는 dict 여야 함
    if not isinstance(g, dict):
        raise ValueError("워크플로 형식 오류: prompt 그래프가 dict 가 아님")

    # 4) 각 노드 검증
    bad = []
    for nid, node in g.items():
        if not isinstance(node, dict) or "class_type" not in node:
            bad.append(nid)
    if bad:
        raise ValueError(f"워크플로 노드에 class_type 누락: {', '.join(bad[:10])}" + ("..." if len(bad) > 10 else ""))

    return g


# ───────────────────────────── 서버/노드 유틸 ──────────────────────────────────
def _choose_host() -> str:
    """settings의 COMFY_HOST 우선, 그 다음 DEFAULT_HOST_CANDIDATES에서 살아있는 서버를 선택."""
    cand: List[str] = []
    if COMFY_HOST:
        cand.append(COMFY_HOST)
    for c in (getattr(S, "DEFAULT_HOST_CANDIDATES", DEFAULT_HOST_CANDIDATES) or []):
        if c and c not in cand:
            cand.append(c)
    for host in cand:
        if host and _probe_server(host):
            return host
    return cand[0] if cand else "http://127.0.0.1:8188"




# ───────────────────────────── 태그/노드 유틸 ──────────────────────────────────
def _collect_effective_tags(meta: dict) -> List[str]:
    """
    project.json에서 실제 주입할 태그 리스트:
    - auto_tags == True : ace_tags + tags_in_use
    - auto_tags == False: manual_tags
    """
    if meta.get("auto_tags", True):
        tags = list(meta.get("ace_tags") or [])
        tags = list(dict.fromkeys(tags + (meta.get("tags_in_use") or [])))
        return tags
    else:
        return list(meta.get("manual_tags") or [])

def _graph(prompt) -> dict:
    """
    프롬프트 JSON이 다양한 스키마(dict / {'prompt':{}} / {'nodes':[...]})로 올 수 있으니
    항상 dict를 돌려주도록 방어적으로 정규화한다.
    """
    # 1) 없으면 빈 dict
    if prompt is None:
        return {}

    # 2) 최상위에 {"prompt": {...}} 형태면 내부만
    if isinstance(prompt, dict):
        inner = prompt.get("prompt")
        if isinstance(inner, dict):
            prompt = inner

    # 3) {"nodes":[{id:.., class_type:..}, ...]} → {id(str): node(dict)}
    if isinstance(prompt, dict) and isinstance(prompt.get("nodes"), list):
        g = {}
        for n in prompt["nodes"]:
            if isinstance(n, dict):
                nid = str(n.get("id") or "")
                if nid:
                    g[nid] = {k: v for k, v in n.items() if k != "id"}
        return g

    # 4) dict면 그대로, 아니면 빈 dict
    return prompt if isinstance(prompt, dict) else {}


def _find_nodes_by_class_names(graph: dict, class_names: Iterable[str]) -> List[Tuple[str, dict]]:
    names = set(class_names)
    res: List[Tuple[str, dict]] = []
    for nid, node in (graph or {}).items():
        if isinstance(node, dict) and node.get("class_type") in names:
            res.append((str(nid), node))
    return res

def _find_nodes_by_class_contains(graph: dict, needle: str) -> list[tuple[str, dict]]:
    needle = (needle or "").lower()
    out = []
    for nid, node in (graph or {}).items():
        ct = str(node.get("class_type", "")).lower()
        if needle and needle in ct:
            out.append((str(nid), node))
    return out

def rewrite_prompt_audio_format(json_path: Path, desired_fmt: str) -> Tuple[int, str]:
    """
    워크플로 파일 안의 SaveAudio* 노드를 desired_fmt('wav'|'mp3'|'opus')로 바꿔 저장.
    """
    desired_fmt = (desired_fmt or "mp3").lower().strip()
    if desired_fmt not in ("mp3", "wav", "opus"):
        desired_fmt = "mp3"
        # return 0
    try:
        data = load_json(json_path, {}) or {}
    except Exception as e:
        return 0, f"프롬프트 JSON 로드 실패: {e}"

    g = _graph(data)
    if not isinstance(g, dict) or not g:
        return 0, f"프롬프트 JSON 형식 오류: {json_path}"

    targets = _find_nodes_by_class_contains(g, "saveaudio")
    if not targets:
        return 0, "SaveAudio* 노드를 찾지 못했습니다."

    changed = 0
    for _nid, node in targets:
        ins = node.setdefault("inputs", {})
        if desired_fmt == "wav":
            node["class_type"] = "SaveAudioWAV"
            ins.pop("quality", None)
            changed += 1
        elif desired_fmt == "opus":
            node["class_type"] = "SaveAudioOpus"
            ins.pop("quality", None)
            changed += 1
        else:  # mp3
            node["class_type"] = "SaveAudioMP3"
            q = str(ins.get("quality", "")).strip().lower()
            if q not in ("v0", "128k", "320k"):
                ins["quality"] = "320k"
            changed += 1

    if changed:
        try:
            save_json(json_path, data)
            return changed, f"{json_path.name} 저장 노드 {changed}개를 '{desired_fmt}'로 갱신."
        except Exception as e:
            return 0, f"프롬프트 JSON 저장 실패: {e}"
    return 0, "변경 사항 없음."

def _ensure_filename_prefix(graph: dict, prefix: str = "vocal_final") -> None:
    """모든 SaveAudio* 노드의 filename_prefix를 동일하게 강제."""
    for _nid, node in _find_nodes_by_class_contains(graph, "saveaudio"):
        node.setdefault("inputs", {})["filename_prefix"] = prefix

def _rand_seed() -> int:
    return random.randint(1, 2_147_483_646)


# ────────────────────── 저장 노드 '적응' 로직(핵심) ────────────────────────────
# --- NEW: 서버에 설치된 노드 클래스 목록 가져오기 (없으면 빈 set) ---
def _get_server_node_classes(base: str) -> set[str]:
    """
    ComfyUI 서버의 /object_info 를 조회해 설치된 노드 class_type 목록을 set로 반환.
    실패하면 빈 set를 반환하여 안전하게 폴백.
    """
    try:
        import requests  # ComfyUI를 돌리는 환경이면 보통 있음
        url = base.rstrip("/") + "/object_info"
        resp = requests.get(url, timeout=3)
        resp.raise_for_status()
        data = resp.json() or {}
        classes: set[str] = set()
        # 형식이 환경마다 약간 다를 수 있어 광범위하게 수집
        # 1) {"nodes": {"ClassName": {...}, ...}}
        nodes = data.get("nodes")
        if isinstance(nodes, dict):
            classes.update(nodes.keys())
        # 2) {"Categories": {"Audio": {"nodes": [{"class_type": "..."}]}}}
        cats = data.get("Categories") or data.get("categories")
        if isinstance(cats, dict):
            for _k, v in cats.items():
                lst = v.get("nodes") if isinstance(v, dict) else None
                if isinstance(lst, list):
                    for item in lst:
                        ct = (item.get("class_type") or item.get("name") or "").strip()
                        if ct:
                            classes.add(ct)
        return classes
    except Exception:
        return set()

# --- REPLACE: 저장 노드 자동 선택(외부 util 불필요) ---
def _apply_save_audio_node_adaptively(
    base: str,
    graph: dict,
    desired_fmt: str = "wav"
) -> str:
    """
    서버에 실제로 존재하는 '오디오 저장' 노드를 감지해 가능한 클래스로 교체한다.
    - desired_fmt: 'wav'|'mp3'|'opus' (기본 wav)
    - 반환: 실제 적용된 확장자('.wav' | '.mp3' | '.opus')
    """
    desired_fmt = (desired_fmt or "wav").lower().strip()
    if desired_fmt not in ("wav", "mp3", "opus"):
        desired_fmt = "wav"

    server_nodes = _get_server_node_classes(base)  # 실패 시 빈 set

    wav_candidates  = ("SaveAudioWAV", "SaveWAV", "SaveAudioWav")
    mp3_candidates  = ("SaveAudioMP3", "SaveMP3", "SaveAudioMp3")
    opus_candidates = ("SaveAudioOPUS", "SaveOPUS", "SaveAudioOpus")

    # 서버에서 지원하는지 먼저 확인, 실패하면 그래프 문자열에서라도 존재여부 체크
    import json as _json
    graph_txt = _json.dumps(graph, ensure_ascii=False)

    def _has(cls: str) -> bool:
        if server_nodes:
            return cls in server_nodes
        return cls in graph_txt  # 서버 조회 실패 시 대충이라도 판단

    fmt_groups = {
        "wav":  (wav_candidates,  (mp3_candidates, opus_candidates)),
        "mp3":  (mp3_candidates,  (wav_candidates, opus_candidates)),
        "opus": (opus_candidates, (wav_candidates, mp3_candidates)),
    }
    primary, fallbacks = fmt_groups[desired_fmt]

    chosen_fmt = None
    chosen_cls = None

    # 1차: 원하는 포맷 후보들 중 사용 가능한 클래스 고르기
    for cls in primary:
        if _has(cls):
            chosen_cls = cls
            chosen_fmt = "wav" if cls in wav_candidates else "mp3" if cls in mp3_candidates else "opus"
            break

    # 2차: 폴백 포맷 순회
    if not chosen_cls:
        for group in fallbacks:
            for cls in group:
                if _has(cls):
                    chosen_cls = cls
                    chosen_fmt = "wav" if cls in wav_candidates else "mp3" if cls in mp3_candidates else "opus"
                    break
            if chosen_cls:
                break

    # 3) 그래프 내 저장 노드 클래스명 치환
    if chosen_cls and chosen_fmt:
        nodes = graph.get("nodes") or []
        for node in nodes:
            if isinstance(node, dict):
                ct = str(node.get("class_type", ""))
                if ct.lower().startswith("saveaudio"):
                    node["class_type"] = chosen_cls
        return f".{chosen_fmt}"

    # 최후의 수단: 아무것도 못찾으면 그냥 wav로 가정
    return ".wav"




# ───────────────────────────── 결과 파일 처리 ─────────────────────────────────
def _download_output_file(base: str, filename: str, subfolder: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    _dlog("TRY-DOWNLOAD", f"filename={filename}", f"subfolder={subfolder}", f"out_dir={out_dir}")

    combos = [
        ("audio", subfolder or ""),
        ("output", subfolder or ""),
        ("audio", ""),   # subfolder 비움
        ("output", ""),  # subfolder 비움
    ]

    last_err = None
    for t, sf in combos:
        try:
            r = _http_get(base, "/view", params={
                "filename": filename,
                "subfolder": sf,
                "type": t
            }, timeout=20)
            _dlog("VIEW-RESP", f"type={t}", f"sf='{sf}'", f"status={r.status_code}", f"bytes={len(r.content) if r.ok else 0}")
            if r.ok and r.content:
                target = out_dir / filename
                with open(target, "wb") as f:
                    f.write(r.content)
                _dlog("DOWNLOADED", str(target), f"size={target.stat().st_size}")
                return target
        except Exception as e:
            last_err = e
            _dlog("VIEW-TRY-FAIL", f"type={t}", f"sf='{sf}'", f"{type(e).__name__}: {e}")

    if last_err:
        raise last_err
    raise RuntimeError(f"다운로드 실패: filename={filename}, subfolder='{subfolder}'")




def _transcode_if_needed(src_path: Path, desired_fmt: str, ffmpeg_exe: str = "ffmpeg") -> Path:
    """
    받은 파일을 최종 포맷(desired_fmt)으로 보정. 동일 포맷이면 파일명만 규격화.
    - wav : pcm_s16le 44.1kHz 2ch
    - mp3 : libmp3lame 320k
    - opus: libopus
    실패 시 원본 반환
    """
    desired_fmt = (desired_fmt or "mp3").lower()
    proj_dir = src_path.parent
    dst = proj_dir / f"vocal.{desired_fmt}"
    src_ext = src_path.suffix.lower().lstrip(".")

    if src_ext == desired_fmt:
        try:
            shutil.copyfile(str(src_path), str(dst))
            return dst
        except Exception:
            return src_path

    if desired_fmt == "wav":
        cmd = [ffmpeg_exe, "-y", "-i", str(src_path), "-vn",
               "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", str(dst)]
    elif desired_fmt == "mp3":
        cmd = [ffmpeg_exe, "-y", "-i", str(src_path), "-vn",
               "-acodec", "libmp3lame", "-b:a", "320k", str(dst)]
    elif desired_fmt == "opus":
        cmd = [ffmpeg_exe, "-y", "-i", str(src_path), "-vn",
               "-acodec", "libopus", str(dst)]
    else:
        return src_path

    try:
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if cp.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
            return dst
    except Exception:
        pass
    return src_path


# ───────────────────────────── 제출/폴링 ──────────────────────────────────────
def _ping_comfy(base: str, timeout: float = 3.0) -> None:
    try:
        r = requests.get(f"{base}/system_stats", timeout=timeout)
        if r.status_code != 200:
            raise RuntimeError(f"ComfyUI 응답 코드 {r.status_code}")
    except Exception as e:
        raise ConnectionError(f"ComfyUI에 연결 실패: {base} ({e})")

def _queue_status(base: str) -> dict[str, Any]:
    try:
        r = requests.get(f"{base}/queue", timeout=5.0)
        if r.ok:
            js = r.json() or {}
            return js if isinstance(js, dict) else {}
    except Exception:
        pass
    return {}



def _submit_and_wait(
    base: str,
    wf_graph: dict,
    timeout: int | float | None = None,
    poll: float | None = None,
    on_progress=None
) -> dict[str, Any]:
    """
    ComfyUI /prompt 제출 후 /history/<id>를 폴링하여 완료까지 대기.
    반환: history의 해당 entry(dict[str, Any]) — outputs/status 포함
    """
    import time
    import requests
    from typing import Any, Dict, cast

    # 안전한 기본값 결정
    timeout_val = float(timeout if timeout is not None else _ace_wait_timeout_sec())
    poll_val = float(poll if poll is not None else _ace_poll_interval_sec())

    _dlog("WAIT-START", f"timeout={timeout_val}", f"poll={poll_val}")

    # 0) 서버 핑
    try:
        url_stats = f"{base.rstrip('/')}/system_stats"
        _dlog("PING", url_stats)
        r = requests.get(url_stats, timeout=5.0)
        _dlog("PING-RESP", f"status={r.status_code}")
        if r.status_code != 200:
            raise RuntimeError(f"ComfyUI 응답 코드 {r.status_code}")
    except Exception as e:
        _dlog("PING-FAIL", f"{type(e).__name__}: {e}")
        raise ConnectionError(f"ComfyUI에 연결 실패: {base} ({e})")

    # 1) 제출
    try:
        url_prompt = f"{base.rstrip('/')}/prompt"
        _dlog("POST-/prompt", f"url={url_prompt}", f"nodes={len(wf_graph)}")
        r = requests.post(url_prompt, json={"prompt": wf_graph}, timeout=(5.0, 25.0))
        _dlog("POST-RESP", f"ok={r.ok}", f"status={r.status_code}")
    except Exception as e:
        _dlog("POST-FAIL", f"{type(e).__name__}: {e}")
        raise

    if not r.ok:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        _dlog("POST-NOK-BODY", str(detail)[:400])
        raise RuntimeError(f"/prompt 제출 실패: {r.status_code} {detail}")

    try:
        resp: Dict[str, Any] = r.json() or {}
    except Exception as e:
        _dlog("POST-JSON-FAIL", f"{type(e).__name__}: {e}", f"text={r.text[:400]}")
        raise

    prompt_id = resp.get("prompt_id") or resp.get("promptId") or resp.get("id")
    if not prompt_id:
        raise RuntimeError("ComfyUI가 prompt_id를 반환하지 않았습니다.")
    prompt_key = str(prompt_id)
    _dlog("PROMPT-ID", prompt_key)

    t0 = time.time()
    last_outputs = 0
    last_hist_status = None
    tick = 0

    last_queue_pending = None
    last_queue_running = None
    idle_ticks_after_output = 0

    while True:
        tick += 1
        elapsed = time.time() - t0
        if elapsed > timeout_val:
            _dlog("TIMEOUT", f"elapsed={elapsed:.1f}s")
            raise TimeoutError(
                "ACE-Step 대기 시간 초과\n"
                "- ComfyUI 실행/COMFY_HOST 확인\n"
                "- SaveAudio 노드 타입/옵션 확인 (mp3/wav/opus)\n"
                "- 큐가 길면 대기 또는 서버 재시작\n"
                "- 첫 실행 시 모델 다운로드 완료까지 대기\n"
            )

        # 큐 상태
        try:
            q = requests.get(f"{base.rstrip('/')}/queue", timeout=5.0)
            if q.ok:
                qj: Dict[str, Any] = cast(Dict[str, Any], q.json() or {})
                last_queue_pending = len(cast(list, qj.get("queue_pending", []) or []))
                last_queue_running = len(cast(list, qj.get("queue_running", []) or []))
                if on_progress:
                    on_progress({
                        "stage": "queue",
                        "pending": last_queue_pending,
                        "running": last_queue_running,
                        "elapsed": elapsed,
                    })
                if tick % 5 == 0:
                    _dlog("QUEUE", f"pending={last_queue_pending}", f"running={last_queue_running}")
        except Exception as e:
            if tick % 5 == 0:
                _dlog("QUEUE-READ-FAIL", f"{type(e).__name__}: {e}")

        # 히스토리
        try:
            h = requests.get(f"{base.rstrip('/')}/history/{prompt_key}", timeout=10.0)
            if not h.ok:
                if tick % 5 == 0:
                    _dlog("HIST-HTTP", f"status={h.status_code}")
                time.sleep(poll_val)
                continue
            hist_raw: Any = h.json()
            hist: Dict[str, Any] = cast(Dict[str, Any], hist_raw or {})
        except Exception as e:
            if tick % 5 == 0:
                _dlog("HIST-READ-FAIL", f"{type(e).__name__}: {e}")
            time.sleep(poll_val)
            continue

        entry_obj = hist.get(prompt_key)
        if not isinstance(entry_obj, dict):
            if tick % 5 == 0:
                _dlog("HIST-NO-ENTRY-YET")
            time.sleep(poll_val)
            continue
        entry_dict: Dict[str, Any] = cast(Dict[str, Any], entry_obj)

        # 상태/키 로깅(1회성)
        st_obj = (entry_dict.get("status") or {})
        st = st_obj.get("status") or st_obj.get("status_str")
        if st != last_hist_status:
            _dlog("HIST-STATUS", st, "| keys:", list(entry_dict.keys()))
            if st_obj:
                _dlog("HIST-STATUS-OBJ", {k: st_obj.get(k) for k in list(st_obj.keys())[:6]})
            last_hist_status = st

        # 에러
        if (st or "").lower() == "error":
            err = st_obj.get("error") or {}
            node_errors = err.get("node_errors") or {}
            details = []
            for nid, ne in node_errors.items():
                details.append(f"node {nid}: {ne.get('message') or ne}")
            msg = err.get("message") or "ComfyUI 내부 에러"
            _dlog("HIST-ERROR", msg, "|", " / ".join(details))
            raise RuntimeError(f"ComfyUI 에러: {msg}\n" + ("\n".join(details) if details else ""))

        # 출력 수 변화
        outs = entry_dict.get("outputs") or {}
        n_outs = sum(len(v or []) for v in (outs.values() if isinstance(outs, dict) else []))
        if n_outs != last_outputs:
            last_outputs = n_outs
            _dlog("HIST-OUTPUTS", f"count={n_outs}")
            if on_progress:
                on_progress({"stage": "running", "outputs": n_outs, "elapsed": elapsed})

        # 정상 완료
        exec_info = (st_obj.get("exec_info") or {})
        queue_info = (exec_info.get("queue") or "")
        if (st in ("success", "completed", "ok")) or ("completed" in str(queue_info).lower()):
            _dlog("HIST-DONE", f"elapsed={elapsed:.1f}s", f"outputs={n_outs}")
            if on_progress:
                on_progress({"stage": "completed", "outputs": n_outs, "elapsed": elapsed})
            return entry_dict  # ✅

        # 휴리스틱 완료: 출력이 있고 큐가 비었으면 몇 틱 후 완료 간주
        if n_outs > 0 and last_queue_pending == 0 and last_queue_running == 0:
            idle_ticks_after_output += 1
            if idle_ticks_after_output == 1:
                _dlog("HIST-HEURISTIC-ARMED", "outputs>0 & queue empty -> waiting few ticks")
            if idle_ticks_after_output >= 3:
                _dlog("HIST-HEURISTIC-DONE", f"elapsed={elapsed:.1f}s", f"outputs={n_outs}")
                if on_progress:
                    on_progress({"stage": "completed(heuristic)", "outputs": n_outs, "elapsed": elapsed})
                return entry_dict
        else:
            idle_ticks_after_output = 0

        time.sleep(poll_val)







# ───────────────────────────── 메인 함수 ──────────────────────────────────────
def _resolve_audio_dir_from_template(root_or_tpl: str, title: str) -> Path:
    """FINAL_OUT(예: C:\\my_games\\shorts_make\\maked_title\\[title])에서 [title] 치환."""
    tpl = (root_or_tpl or "").strip()
    if not tpl:
        return Path()  # 빈 Path → 사용 안 함
    safe_title = sanitize_title(title or "untitled")
    return Path(tpl.replace("[title]", safe_title)).resolve()

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

# ─────────────────────────────
# 필요한 유틸이 이 파일에 이미 있다고 가정:
# - _dlog, load_json, save_json, effective_title, sanitize_title
# - _load_workflow_graph(ACE_STEP_PROMPT_JSON), _choose_host
# - _apply_save_audio_node_adaptively, _find_nodes_by_class_names, _find_nodes_by_class_contains
# - _ensure_filename_prefix, _submit_and_wait, _rand_seed
# - save_to_user_library
# - settings as S (FFMPEG_EXE, FINAL_OUT, AUDIO_SAVE_FORMAT 등)
# ─────────────────────────────

def _ensure_vocal_wav(src_path: Path, proj_dir: Path, ffmpeg_exe: str = "ffmpeg") -> Path:
    """
    입력이 wav가 아니면 wav(16-bit PCM, 44.1kHz, 2ch)로 트랜스코딩하여
    proj_dir/vocal.wav 로 저장. 입력이 이미 wav면 파일 이동/복사로 통일.
    """
    proj_dir.mkdir(parents=True, exist_ok=True)
    out_wav = proj_dir / "vocal.wav"

    if src_path.suffix.lower() == ".wav":
        if src_path.resolve() != out_wav.resolve():
            try:
                if out_wav.exists():
                    out_wav.unlink()
            except Exception:
                pass
            src_path.replace(out_wav)
        return out_wav

    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(src_path),
        "-ac", "2",
        "-ar", "44100",
        "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return out_wav


def _normalize_saveaudio_nodes(graph: dict, *, out_path: str, prefix: str, out_dir: str) -> None:
    for node in _iter_nodes(graph):
        ct = str(node.get("class_type", "")).lower()
        if not ct.startswith("saveaudio"):
            continue
        ins = node.setdefault("inputs", {})
        ins["filename"] = out_path
        ins["filename_prefix"] = prefix
        ins["output_path"] = out_dir
        ins["basename"] = "vocal"
        ins["base_filename"] = "vocal"

def _iter_nodes(graph: dict):
    """
    ComfyUI 그래프가 { "nodes": [ {...}, ... ] } 또는
    { "59": {...}, "74": {...} } 두 형태 모두 올 수 있으니
    노드 dict들을 yield 해준다.
    """
    if not isinstance(graph, dict):
        return
    if "nodes" in graph and isinstance(graph["nodes"], list):
        for node in graph["nodes"]:
            if isinstance(node, dict):
                yield node
    else:
        for _k, node in graph.items():
            if isinstance(node, dict) and "class_type" in node:
                yield node

def _sanitize_saveaudio_options(graph: dict) -> None:
    """
    SaveAudio* 노드 옵션을 서버 스키마에 맞게 보정.
    - MP3: quality ∈ {'V0','128k','320k'} ; '320' → '320k' 등
    - WAV: sample_rate/bit_depth 기본값 채우기(있을 때만)
    - OPUS: bitrate '128k' 형태로 정리
    """
    for node in _iter_nodes(graph):
        ct = str(node.get("class_type", "")).lower()
        if not ct.startswith("saveaudio"):
            continue
        ins = node.setdefault("inputs", {}) or {}

        if "mp3" in ct:
            q = str(ins.get("quality", "")).strip().lower()
            alias = {
                "320": "320k", "320kbps": "320k",
                "128": "128k", "128kbps": "128k",
                "v0": "V0", "vo": "V0"
            }
            q = alias.get(q, q)
            if q not in ("V0", "v0", "128k", "320k"):
                q = "320k"
            ins["quality"] = "V0" if q.lower() == "v0" else q
            ins.pop("bitrate", None)  # 혼동 키 제거

        elif "wav" in ct:
            if "sample_rate" in ins:
                ins.setdefault("sample_rate", 44100)
            if "bit_depth" in ins:
                ins.setdefault("bit_depth", 16)

        elif "opus" in ct:
            br = ins.get("bitrate")
            if isinstance(br, (int, float)):
                ins["bitrate"] = f"{int(br)}k"
            elif isinstance(br, str):
                s = br.strip().lower()
                if s.isdigit():
                    ins["bitrate"] = f"{int(s)}k"
                elif not s.endswith("k"):
                    ins["bitrate"] = "128k"
def _force_wav_save_for_this_graph(graph: dict, *, proj_dir: Path, prefix: str) -> str:
    """
    이 워크플로(노드 78: SaveAudio)를 WAV 저장으로 강제.
    - filename_prefix/out_dir/베이스네임 보정
    - WAV 관련 옵션 키가 있으면 합리적 기본값 주입
    반환: ".wav"
    """
    # 공통: prefix 보정 (예: "shorts_make/<제목>/vocal_final")
    for node in _iter_nodes(graph):
        ct = str(node.get("class_type", "")).lower()
        if ct in ("saveaudio", "saveaudiowav", "saveaudiomp3", "pysssss_saveaudio"):
            ins = node.setdefault("inputs", {}) or {}
            # 경로 강제
            ins["filename_prefix"] = prefix
            # 일부 구현은 output_path/filename을 지원
            if "output_path" in ins:
                ins["output_path"] = str(proj_dir)
            if "basename" in ins:
                ins["basename"] = "vocal"
            if "base_filename" in ins:
                ins["base_filename"] = "vocal"
            # WAV로 유도 가능한 후보 키들(있을 때만 셋)
            # 구현별로 'format', 'container', 'codec' 등을 지원하기도 함
            for k, v in (
                ("format", "wav"),
                ("container", "wav"),
                ("codec", "pcm_s16le"),
            ):
                if k in ins:
                    ins[k] = v
            # 샘플레이트/비트뎁스 키가 있으면 기본값
            if "sample_rate" in ins:
                ins.setdefault("sample_rate", 44100)
            if "bit_depth" in ins:
                # 일부 노드는 "bit_depth" 대신 "bitdepth"/"bits"일 수도 있음
                ins["bit_depth"] = 16
            for bd_key in ("bitdepth", "bits"):
                if bd_key in ins:
                    ins[bd_key] = 16
    return ".wav"


def _retarget_lls_text_saver(graph: dict, *, subfolder: str, proj_dir: str) -> bool:
    """
    SaveText|pysssss 노드를 찾아 ComfyUI/output/<subfolder>/_lls_after.txt 로 리타겟팅.
    없으면 새 노드를 만들어 LyricsLangSwitch 출력(0)에 연결한다.
    return: 1개 이상 패치/생성되면 True, 아니면 False
    """
    def _iter_nodes(g: dict):
        if isinstance(g, dict) and "nodes" in g and isinstance(g["nodes"], list):
            for n in g["nodes"]:
                if isinstance(n, dict):
                    yield n
        elif isinstance(g, dict):
            for _k, n in g.items():
                if isinstance(n, dict):
                    yield n

    changed = False
    out_rel = f"{subfolder}/_lls_after.txt".replace("\\", "/")

    # 1) 기존 SaveText|pysssss를 패치
    for node in _iter_nodes(graph):
        if str(node.get("class_type") or "") != "SaveText|pysssss":
            continue
        ins = node.setdefault("inputs", {}) or {}
        ins["root_dir"] = "output"
        ins["file"] = out_rel
        ins["append"] = "overwrite"
        ins["insert"] = False
        changed = True

    if changed:
        return True

    # 2) 없으면 새로 만든다: LLS 노드(첫 번째)의 출력 0에 연결
    lls_node = None
    for node in _iter_nodes(graph):
        if str(node.get("class_type") or "") == "LyricsLangSwitch":
            lls_node = node
            break
    if lls_node is None:
        return False

    # 그래프 포맷이 dict(id->node)인 경우에만 안전 생성
    if "nodes" in graph and isinstance(graph["nodes"], list):
        # list 포맷이면 생성은 보류(워크플로 포맷마다 다름)
        return False

    # 새 노드 id 찾기
    used_ids = {str(k) for k in graph.keys() if isinstance(k, str)}
    new_id = "save_lls_text"
    i = 1
    while new_id in used_ids:
        new_id = f"save_lls_text_{i}"; i += 1

    graph[new_id] = {
        "inputs": {
            "root_dir": "output",
            "file": out_rel,
            "append": "overwrite",
            "insert": False,
            "text": [str(lls_node.get("id") or "74"), 0]
        },
        "class_type": "SaveText|pysssss",
        "_meta": {"title": "Save Text 🐍 (auto)"}
    }
    return True




def generate_music_with_acestep(
    project_dir: str,
    *,
    on_progress: Optional[Callable[[dict], None]] = None,
    target_seconds: int | None = None,
) -> str:
    """
    ComfyUI(ACE-Step)로 음악 생성:
      - SaveAudio 계열을 WAV 저장으로 강제 → <project_dir>/vocal.wav
      - LyricsLangSwitch(LLS) 입력/결과 텍스트 저장
      - 사용자 라이브러리(FINAL_OUT) 복사
      - 메타(project.json)에 lyrics/lyrics_lls 갱신(원본 lyrics는 기본 유지)
      - 길이 산정: 본문 줄수×5초 + intro/outro(각 5~15초 랜덤) + margin(2~5초)
    """
    from pathlib import Path
    import random as _random
    from time import time as _time

    def notify(stage: str, **kw):
        if on_progress:
            try:
                info = {"stage": stage}
                info.update(kw)
                on_progress(info)
            except Exception:
                pass

    # on_progress 중계: payload 안의 'stage' 키는 제거하여 중복 인자 오류 방지
    def _progress_forward(payload: dict):
        if not on_progress:
            return
        try:
            info2 = dict(payload or {})
            info2.pop("stage", None)
            notify("progress", **info2)
        except Exception:
            pass

    _dlog("ENTER", f"project_dir={project_dir}")
    proj = Path(project_dir); proj.mkdir(parents=True, exist_ok=True)
    pj = proj / "project.json"
    meta = load_json(pj, {}) or {}

    title = effective_title(meta)
    lyrics = (meta.get("lyrics") or "").strip()
    if not lyrics:
        raise RuntimeError("project.json에 가사가 없습니다. 먼저 저장/생성해 주세요.")

    # 원본 가사 보존
    try:
        (proj / "lyrics.txt").write_text(lyrics, encoding="utf-8", errors="ignore")
    except OSError:
        pass
    meta["lyrics"] = lyrics
    save_json(pj, meta)

    # === 목표 길이 산정 ===
    body_lines: list[str] = []
    for line in lyrics.splitlines():
        s = (line or "").strip()
        if not s:
            continue
        if s.startswith("[") and s.endswith("]"):
            continue
        body_lines.append(s)
    body_count = max(1, len(body_lines))

    avg_line_sec = 5
    core_sec = body_count * avg_line_sec
    intro_sec = _random.randint(5, 15)
    outro_sec = _random.randint(5, 15)
    margin_sec = _random.randint(2, 5)

    # === 목표 길이 산정 ===
    # ... (위쪽 동일)

    base_target = core_sec + intro_sec + outro_sec + margin_sec

    if target_seconds is not None:
        try:
            tsec = int(target_seconds)
            if tsec > base_target:
                base_target = tsec
        except ValueError:
            pass

    # ★ 스냅 로직: 기준 이상으로 올림
    import math  # ← 함수 상단 import 목록에 없으면 추가
    snap_bins = [30, 60, 120, 180]
    seconds = None
    for b in snap_bins:
        if base_target <= b:
            seconds = b
            break
    if seconds is None:
        # 180을 넘으면 30초 단위로 올림
        seconds = int(math.ceil(base_target / 30.0) * 30)

    # 이하 동일
    music_meta = {
        "computed_core_sec": int(core_sec),
        "intro_sec": int(intro_sec),
        "outro_sec": int(outro_sec),
        "margin_sec": int(margin_sec),
        "avg_line_sec": int(avg_line_sec),
        "body_lines": int(body_count),
        "snap_seconds": int(seconds),
        "calc_epoch": int(_time()),
    }
    meta["music_meta"] = music_meta
    meta["target_seconds"] = int(seconds)
    meta["time"] = int(seconds)
    save_json(pj, meta)

    effective_tags = _collect_effective_tags(meta)

    # ───────────────────── 워크플로 로드/보정 ─────────────────────
    g = _load_workflow_graph(ACE_STEP_PROMPT_JSON)

    base = _choose_host()
    _dlog("HOST", base, "| DESIRED_FMT wav")

    subfolder = f"shorts_make/{sanitize_title(title)}"
    prefix = f"{subfolder}/vocal_final"
    out_path = str(proj / "vocal.wav")

    # WAV 저장 강제
    ext = _force_wav_save_for_this_graph(g, proj_dir=proj, prefix=prefix)
    _dlog("SAVE-NODE-EXT", ext)

    # LLS 텍스트 saver 리타겟팅 — (필요 시) 서브폴더 지정
    _retarget_lls_text_saver(g, subfolder=subfolder, proj_dir=str(proj))

    # 가사/태그/길이 반영 (인트로/아웃트로 고려 + 중복 코러스 방지 전처리)
    _apply_lyrics_tags_seconds_to_graph(g, lyrics=lyrics, seconds=seconds, effective_tags=effective_tags, meta=meta)

    # ★ 기존 중간 산출물 정리(누적 방지)
    _cleanup_prev_outputs(proj)

    # ← 여기 추가: 실제 패치가 먹었는지 로컬 덤프
    _dump_graph_after_patch(g, proj)

    # LLS 입력 디버그 덤프
    _dump_lls_debug(g, proj)

    # 제출/대기
    notify("submit", seconds=int(seconds))
    hist_entry = _submit_and_wait(
        base, g,
        timeout=int(_ace_wait_timeout_sec()),
        poll=float(_ace_poll_interval_sec()),
        on_progress=_progress_forward,
    )

    # === 결과 다운로드/정리/메타 갱신 ===
    saved_files: list[Path] = []
    text_payloads: list[str] = []  # LLS 문자열 출력 누적
    outputs = hist_entry.get("outputs") if isinstance(hist_entry, dict) else None

    if isinstance(outputs, dict):
        for nid, out in outputs.items():
            # 1) 파일형 출력들(오디오/파일/이미지/텍스트 파일 등) 다운로드
            file_like_keys = ("audio", "files", "file", "images", "text")
            file_items: list[dict] = []
            for k in file_like_keys:
                arr = out.get(k) or []
                if isinstance(arr, list):
                    for item in arr:
                        if isinstance(item, dict) and item.get("filename"):
                            file_items.append(item)
                elif isinstance(arr, dict) and arr.get("filename"):
                    file_items.append(arr)

            for item in file_items:
                fn = (item.get("filename") or "").strip()
                if not fn:
                    continue
                sf = (item.get("subfolder") or "").strip()
                if not sf or fn.startswith("ComfyUI_temp_"):
                    _dlog("SKIP-PREVIEW", f"nid={nid}", f"fn={fn!r}", f"sf={sf!r}")
                    continue
                sf_norm = sf.replace("\\", "/").lstrip("/")
                out_file = _download_output_file(base, fn, sf_norm, out_dir=proj)
                if out_file:
                    saved_files.append(out_file)

            # 2) 문자열형 텍스트 출력들 수집
            for k in ("text", "txt", "string"):
                val = out.get(k)
                if isinstance(val, list):
                    for s in val:
                        if isinstance(s, str) and s.strip():
                            text_payloads.append(s.strip())
                elif isinstance(val, str) and val.strip():
                    text_payloads.append(val.strip())

    _dlog("SAVED-FILES", [str(p) for p in saved_files])

    # === 결과 오디오 정리 (최종 vocal.wav 로 고정) ===
    final_path = None
    try:
        # 1) 후보 고르기: saved_files + 프로젝트 폴더 스캔
        cand = _pick_best_audio_candidate(proj, saved_files)
        if cand is None:
            _dlog("FINALIZE", "no audio candidate found")
        else:
            ff = getattr(S, "FFMPEG_EXE", "ffmpeg")
            final_path = _write_stable_output_wav(cand, proj, ffmpeg_exe=ff)
            _dlog("FINAL", f"stable out -> {final_path}")
            # 2) 중간 산출물 정리(누적 방지)
            _cleanup_prev_outputs(proj)
    except OSError as e:
        _dlog("FINALIZE-FAIL", type(e).__name__, str(e))
    except Exception as e:
        _dlog("FINALIZE-UNEXPECTED", type(e).__name__, str(e))

    # 명시적 파일 다운로드 시도: SaveText를 output/<subfolder>/_lls_after.txt 로 리타겟했으므로
    try:
        _ = _download_output_file(base, "_lls_after.txt", subfolder, out_dir=proj)
    except Exception:
        pass

    # LLS 결과 텍스트 저장(파일형 또는 문자열형 모두 커버)
    try:
        lls_txt_path = proj / "_lls_after.txt"
        if text_payloads and not lls_txt_path.exists():
            # 문자열형 출력이 있으면 통합 저장
            buf = "\n\n" + ("-" * 60) + "\n\n"
            lls_txt_path.write_text(buf.join(text_payloads), encoding="utf-8", errors="ignore")
            _dlog("TEXT-SAVED(STRING)", str(lls_txt_path))

        # 파일형 텍스트를 이미 다운로드했다면 saved_files에 포함
        # (예: SaveText 노드가 파일을 남기는 경우)
        # lls_txt_path가 없고, saved_files 중 텍스트 파일이 있으면 첫 번째를 백업 이름으로 복사해도 됨
        if not lls_txt_path.exists():
            for p in saved_files:
                if p.suffix.lower() in (".txt", ".log", ".json"):
                    try:
                        lls_txt_path.write_text(p.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8", errors="ignore")
                        _dlog("TEXT-SAVED(FILECOPIED)", str(lls_txt_path))
                        break
                    except OSError:
                        pass

        # lyrics_lls.txt / meta 반영 (원본은 기본 유지)
        if lls_txt_path.exists():
            lls_after = lls_txt_path.read_text(encoding="utf-8", errors="ignore").strip()
            if lls_after:
                (proj / "lyrics_lls.txt").write_text(lls_after, encoding="utf-8", errors="ignore")
                meta["lyrics_lls"] = lls_after
                # 설정이 True일 때만 원본 덮어쓰기
                try:
                    from settings import OVERWRITE_LYRICS_WITH_LLS as _ovw  # type: ignore
                except Exception:
                    _ovw = False
                if _ovw:
                    meta["lyrics"] = lls_after
                save_json(pj, meta)
    except OSError as e:
        _dlog("LLS-LYRICS-SAVE-FAIL", type(e).__name__, str(e))

    # 사용자 라이브러리 복사
    try:
        save_root = getattr(S, "FINAL_OUT", "") or ""
        if final_path and final_path.exists() and save_root:
            _ = save_to_user_library("audio", title, final_path)
    except OSError as e:
        _dlog("COPY-FINAL-FAIL", type(e).__name__, str(e))

    notify("done", seconds=int(seconds), out=str(final_path) if final_path else "")
    return str(final_path) if final_path else str(proj / "vocal.wav")













import subprocess
from pathlib import Path

def _master_wav_fast(src: Path, out: Path | None = None) -> Path:
    """
    고속 1패스 마스터링:
    - 저역 하이패스, 소프트 다이나믹노멀라이즈, 라우드니스 표준화, 트루피크 리미팅
    - 출력: 48kHz / 24-bit PCM
    """
    out = out or (src.parent / (src.stem + "_master.wav"))
    af = ",".join([
        "highpass=f=25",                   # 초저역 정리
        "dynaudnorm=f=150:g=10",           # 소프트 볼륨 평탄화
        "loudnorm=I=-12:TP=-1.0:LRA=11",   # 라우드니스 정규화(스트리밍 표준 근처)
        "alimiter=limit=-1.0",             # 트루피크 -1.0 dB
    ])
    cmd = [
        "ffmpeg","-y","-i",str(src),
        "-ac","2","-ar","48000",
        "-af",af,"-c:a","pcm_s24le",
        str(out)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return out

import json, tempfile, subprocess
from pathlib import Path

def _master_wav_precise(
    src: Path,
    out: Path | None = None,
    *,
    I: float = -12.0,
    TP: float = -1.0,
    LRA: float = 11.0,
    ffmpeg_exe: str = "ffmpeg",
) -> Path:
    """
    2패스 loudnorm 기반 정밀 마스터링
    - 1패스: 측정(highpass + dynaudnorm + loudnorm 측정)
    - 2패스: 보정값 반영 + 트루피크 리미팅
    - 출력: 48kHz / 24-bit PCM WAV
    """
    out = out or (src.parent / (src.stem + "_master.wav"))

    # 1) 측정
    af1 = (
        "highpass=f=25,"
        "dynaudnorm=f=150:g=10,"
        f"loudnorm=I={I}:TP={TP}:LRA={LRA}:print_format=json"
    )
    cmd1 = [ffmpeg_exe, "-y", "-i", str(src), "-af", af1, "-f", "null", "-"]
    proc = subprocess.run(cmd1, capture_output=True, text=True)
    text = proc.stderr or proc.stdout
    s = text.rfind("{"); e = text.rfind("}")
    if s == -1 or e == -1:
        raise RuntimeError("loudnorm 측정 실패(파싱 오류)")
    meas = json.loads(text[s:e+1])

    # 2) 보정 적용 + 리미터
    af2 = (
        "highpass=f=25,"
        "dynaudnorm=f=150:g=10,"
        "loudnorm="
        f"I={I}:TP={TP}:LRA={LRA}:"
        f"measured_I={meas['input_i']}:"
        f"measured_TP={meas['input_tp']}:"
        f"measured_LRA={meas['input_lra']}:"
        f"measured_thresh={meas['input_thresh']}:"
        f"offset={meas['target_offset']},"
        "alimiter=limit=-1.0"
    )
    cmd2 = [
        ffmpeg_exe, "-y", "-i", str(src),
        "-ac", "2", "-ar", "48000",
        "-af", af2, "-c:a", "pcm_s24le",
        str(out),
    ]
    subprocess.run(cmd2, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return out


def _preprocess_lyrics_for_ace(lyrics: str) -> str:
    """
    ACE-Step에 전달할 가사 전처리.
    - 기본: [intro]/[verse]/[chorus]/[bridge]/[outro] 섹션 태그 라인 제거
    - settings.KEEP_SECTION_TAGS_FOR_LLS=True면 섹션 태그를 그대로 유지
    - 연속 중복 라인은 제거(코러스 중복 완화)
    - 공백/빈 줄 정리
    """
    import re
    try:
        from settings import KEEP_SECTION_TAGS_FOR_LLS  # type: ignore
    except Exception:
        KEEP_SECTION_TAGS_FOR_LLS = False

    lines = [re.sub(r"\s+", " ", (ln or "")).strip() for ln in str(lyrics or "").splitlines()]
    out: list[str] = []
    prev = ""
    sec_pat = re.compile(r"^\s*\[(intro|verse|chorus|bridge|outro)\]\s*$", re.IGNORECASE)
    for ln in lines:
        if not ln:
            continue
        if not KEEP_SECTION_TAGS_FOR_LLS and sec_pat.match(ln):
            prev = ln
            continue
        if ln == prev:
            continue
        out.append(ln)
        prev = ln
    return "\n".join(out).strip()



def _apply_lyrics_tags_seconds_to_graph(
    graph: dict,
    *,
    lyrics: str,
    seconds: int,
    effective_tags: list[str],
    meta: dict,
) -> None:
    """
    - LLS: 섹션 태그 보존된 가사(오타 교정 포함) + seconds_sing + offset/head_silence/lead_in
    - TextEncodeAceStepAudio: 구조 유도/인트로/아웃트로 태그 주입
    - inputs.seconds 보유 모든 노드에 '전체 seconds' 강제
    """
    def _iter_nodes(g: dict):
        if isinstance(g, dict) and "nodes" in g and isinstance(g["nodes"], list):
            for n in g["nodes"]:
                if isinstance(n, dict):
                    yield n
        elif isinstance(g, dict):
            for _k, n in g.items():
                if isinstance(n, dict):
                    yield n

    # --- 길이/인트로/아웃트로 계산 ---
    music_meta = (meta.get("music_meta") or {}) if isinstance(meta, dict) else {}
    intro_sec = int(music_meta.get("intro_sec") or 0)
    outro_sec = int(music_meta.get("outro_sec") or 0)
    seconds_total = int(max(1, seconds))
    seconds_sing = max(10, seconds_total - max(0, intro_sec) - max(0, outro_sec))

    # --- LLS에 보낼 가사: 섹션 태그 보존, 헤더 교정 ---
    lyrics_for_lls = _preprocess_lyrics_for_ace(lyrics)

    # --- LLS 입력 주입 ---
    for node in _iter_nodes(graph):
        if str(node.get("class_type") or "") == "LyricsLangSwitch":
            ins = node.setdefault("inputs", {}) or {}
            ins["lyrics"] = lyrics_for_lls
            ins["language"] = "Korean"
            ins.setdefault("threshold", 0.85)
            ins["seconds"] = int(seconds_sing)
            ins["offset"] = int(max(0, intro_sec))
            ins["head_silence"] = int(max(0, intro_sec))
            ins["lead_in"] = int(max(0, intro_sec))

    # --- 구조 태그 + 인트로/아웃트로 지시어 ---
    struct = _collect_structure_tags_from_lyrics(lyrics_for_lls) if ' _collect_structure_tags_from_lyrics' in globals() else []
    lead_tags: list[str] = []
    if intro_sec > 0:
        lead_tags += [f"instrumental intro {intro_sec}s", f"no vocals until {intro_sec}s", "soft fade-in before vocals"]
    if outro_sec > 0:
        lead_tags += [f"outro {outro_sec}s", "fade-out ending"]
    merged_tags = list(effective_tags) + struct + lead_tags

    for node in _iter_nodes(graph):
        if str(node.get("class_type") or "") == "TextEncodeAceStepAudio":
            ins = node.setdefault("inputs", {}) or {}
            prev = ins.get("tags", "")
            prev_s = ", ".join(prev) if isinstance(prev, list) else str(prev)
            extra = ", ".join(merged_tags)
            ins["tags"] = (prev_s.strip() + ", " + extra) if prev_s.strip() else extra
            ins.setdefault("lyrics_strength", 1.0)

    # --- seconds 보유 노드에 전체 길이 강제 ---
    for node in _iter_nodes(graph):
        ins = node.setdefault("inputs", {}) or {}
        if "seconds" in ins:
            try:
                ins["seconds"] = int(seconds_total)
            except Exception:
                pass




def _dump_graph_after_patch(graph: dict, proj_dir) -> None:
    """
    우리가 건든 핵심 노드들의 inputs를 덤프: LLS, TextEncodeAceStepAudio, seconds 보유 노드
    파일:
      - _graph_after_patch.json (핵심 노드 전체)
      - _tags_input.txt (TextEncodeAceStepAudio tags만)
    """
    from pathlib import Path
    import json
    proj = Path(proj_dir)
    proj.mkdir(parents=True, exist_ok=True)

    def _iter_nodes(g: dict):
        if isinstance(g, dict) and "nodes" in g and isinstance(g["nodes"], list):
            for n in g["nodes"]:
                if isinstance(n, dict):
                    yield n
        elif isinstance(g, dict):
            for _k, n in g.items():
                if isinstance(n, dict):
                    yield n

    picks = []
    tags_only = []
    for n in _iter_nodes(graph):
        ct = str(n.get("class_type") or "")
        ins = n.get("inputs") or {}
        if ct in ("LyricsLangSwitch", "TextEncodeAceStepAudio") or ("seconds" in ins):
            picks.append({
                "id": n.get("id"),
                "class_type": ct,
                "inputs": ins
            })
        if ct == "TextEncodeAceStepAudio":
            tags_only.append(str((ins or {}).get("tags", "")).strip())

    try:
        (proj / "_graph_after_patch.json").write_text(
            json.dumps(picks, ensure_ascii=False, indent=2),
            encoding="utf-8", errors="ignore"
        )
    except Exception:
        pass

    try:
        (proj / "_tags_input.txt").write_text(
            "\n\n" + ("-" * 60) + "\n\n".join(tags_only),
            encoding="utf-8", errors="ignore"
        )
    except Exception:
        pass


def _dump_lls_debug(graph: dict, proj_dir) -> None:
    """
    LLS(LyricsLangSwitch) 노드 입력 덤프:
    - _lls_input.json: LLS 노드별 inputs 전체
    - _lls_input_lyrics.txt: LLS에 보낸 가사(여러 노드면 구분선)
    - _lls_input_raw.txt: 원본 가사(정제 전)
    - _lls_input_clean.txt: 전처리 후 가사(정제 후)
    """
    from pathlib import Path
    import json
    proj = Path(proj_dir)
    proj.mkdir(parents=True, exist_ok=True)

    def _iter_nodes(g: dict):
        if isinstance(g, dict) and "nodes" in g and isinstance(g["nodes"], list):
            for n in g["nodes"]:
                if isinstance(n, dict):
                    yield n
        elif isinstance(g, dict):
            for _k, n in g.items():
                if isinstance(n, dict):
                    yield n

    nodes = [n for n in _iter_nodes(graph) if str(n.get("class_type") or "") == "LyricsLangSwitch"]
    if not nodes:
        return

    # 1) inputs 전체 저장
    dump = [{"id": n.get("id"), "class_type": n.get("class_type"), "inputs": (n.get("inputs") or {})} for n in nodes]
    try:
        (proj / "_lls_input.json").write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8", errors="ignore")
    except Exception:
        pass

    # 2) LLS에 들어간 lyrics/seconds/offset 요약 저장
    try:
        parts = []
        for idx, n in enumerate(nodes):
            ins = n.get("inputs") or {}
            lyr = str(ins.get("lyrics") or "").strip()
            sec = ins.get("seconds")
            off = ins.get("offset") if "offset" in ins else (ins.get("head_silence") or ins.get("lead_in"))
            header = [f"[LLS #{idx}]", f"seconds={sec}", f"offset/head_silence/lead_in={off}"]
            parts.append("\n".join(header) + "\n" + lyr)
        if parts:
            (proj / "_lls_input_lyrics.txt").write_text(
                ("\n\n" + ("-" * 60) + "\n\n").join(parts), encoding="utf-8", errors="ignore"
            )
    except Exception:
        pass

    # 3) 원본/정제본도 별도 저장
    try:
        # 원본은 project.json의 lyrics
        pj = proj / "project.json"
        raw_txt = ""
        try:
            from utils import load_json  # type: ignore
        except Exception:
            from app.utils import load_json  # type: ignore
        meta = load_json(pj, {}) or {}
        raw_txt = str((meta.get("lyrics") or "")).strip()

        clean_txt = _preprocess_lyrics_for_ace(raw_txt)

        (proj / "_lls_input_raw.txt").write_text(raw_txt, encoding="utf-8", errors="ignore")
        (proj / "_lls_input_clean.txt").write_text(clean_txt, encoding="utf-8", errors="ignore")
    except Exception:
        pass


def _write_stable_output_mp3(src_path: "Path", dst_dir: "Path", *, ffmpeg_exe: str = "ffmpeg") -> "Path":
    """
    최종 산출물을 항상 <dst_dir>/vocal.mp3 로 저장(덮어쓰기 보장).
    - 기존 vocal.mp3 / vocal.wav 있으면 먼저 삭제
    - src가 mp3가 아니면 ffmpeg 변환, 실패 시 pydub 폴백
    """
    import shutil
    from subprocess import run, PIPE

    dst = dst_dir / "vocal.mp3"
    # 선삭제(덮어쓰기 보장)
    for old in (dst, dst_dir / "vocal.wav"):
        try:
            if old.exists():
                old.unlink()
        except OSError:
            pass

    if src_path.suffix.lower() == ".mp3":
        try:
            shutil.copy2(str(src_path), str(dst))
            return dst
        except OSError:
            pass

    try:
        cmd = [ffmpeg_exe, "-y", "-i", str(src_path), "-vn", "-c:a", "libmp3lame", "-q:a", "2", str(dst)]
        proc = run(cmd, stdout=PIPE, stderr=PIPE)
        if dst.exists() and dst.stat().st_size > 0:
            return dst
    except Exception:
        pass

    try:
        from pydub import AudioSegment  # type: ignore
        seg = AudioSegment.from_file(str(src_path))
        seg.export(str(dst), format="mp3")
        return dst
    except Exception as e:
        raise RuntimeError(f"final mp3 write failed: {type(e).__name__}: {e}")



def _collect_structure_tags_from_lyrics(lyrics: str) -> list[str]:
    """
    가사의 섹션 헤더를 훑어 편성 힌트 태그 생성.
    예: ["structure: intro-verse-chorus-chorus-outro", "intro present", "double chorus", ...]
    """
    import re
    sec_seq: list[str] = []
    counts = {"intro": 0, "verse": 0, "chorus": 0, "bridge": 0, "outro": 0}
    for ln in str(lyrics or "").splitlines():
        m = re.match(r"^\s*\[(intro|verse|chorus|bridge|outro)\]\s*$", ln.strip(), re.IGNORECASE)
        if m:
            s = m.group(1).lower()
            sec_seq.append(s)
            if s in counts:
                counts[s] += 1
    tags: list[str] = []
    if sec_seq:
        tags.append("structure: " + "-".join(sec_seq))
    for k, v in counts.items():
        if v >= 1:
            tags.append(f"{k} present")
        if v >= 2:
            tags.append(f"double {k}")
    return tags

def _cleanup_prev_outputs(proj_dir: "Path", prefix: str = "vocal_final_") -> None:
    """
    Comfy에서 내려받은 중간 산출물(예: vocal_final_0000x_*.flac/.wav 등) 정리.
    - 프로젝트 폴더 내 prefix로 시작하는 flac/wav/mp3 파일 삭제
    - 최종 vocal.wav 는 보존
    """
    try:
        for p in proj_dir.glob(f"{prefix}*"):
            try:
                if p.name.lower() == "vocal.wav":
                    continue
                if p.suffix.lower() in (".flac", ".wav", ".mp3", ".ogg", ".m4a", ".aac"):
                    p.unlink()
            except OSError:
                pass
    except OSError:
        pass


def _pick_best_audio_candidate(proj_dir: "Path", saved_files: list["Path"]) -> "Path | None":
    """
    1) saved_files(다운로드 목록)에서 mtime 최신 파일을 후보로
    2) 없으면 프로젝트 폴더를 스캔해 확장자(.wav/.flac/.mp3/.ogg/.m4a/.aac) 중 mtime 최신 파일을 후보로
    """
    import itertools
    exts = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac")
    cand = None
    if saved_files:
        try:
            cand = max(saved_files, key=lambda p: (p.exists(), getattr(p.stat(), "st_mtime", 0)))
        except Exception:
            cand = saved_files[-1]
    if cand is None or not (hasattr(cand, "exists") and cand.exists()):
        try:
            pool = list(itertools.chain.from_iterable(proj_dir.glob(f"*{e}") for e in exts))
            if pool:
                cand = max(pool, key=lambda p: getattr(p.stat(), "st_mtime", 0))
        except Exception:
            pass
    return cand

def _write_stable_output_wav(src_path: "Path", dst_dir: "Path", *, ffmpeg_exe: str = "ffmpeg") -> "Path":
    """
    최종 산출물을 항상 <dst_dir>/vocal.wav 로 저장(덮어쓰기 보장).
    - 기존 vocal.wav / vocal.mp3 선삭제
    - src가 WAV가 아니면 ffmpeg 변환(실패 시 pydub 폴백)
    """
    import shutil
    from subprocess import run, PIPE

    dst = dst_dir / "vocal.wav"
    # 선삭제(덮어쓰기 보장)
    for old in (dst, dst_dir / "vocal.mp3"):
        try:
            if old.exists():
                old.unlink()
        except OSError:
            pass

    if src_path.suffix.lower() == ".wav":
        try:
            shutil.copy2(str(src_path), str(dst))
            return dst
        except OSError:
            pass

    # ffmpeg 변환 (PCM 16-bit, 샘플레이트는 원본 유지)
    try:
        cmd = [ffmpeg_exe, "-y", "-i", str(src_path), "-vn", "-acodec", "pcm_s16le", str(dst)]
        proc = run(cmd, stdout=PIPE, stderr=PIPE)
        if dst.exists() and dst.stat().st_size > 0:
            return dst
    except Exception:
        pass

    # pydub 폴백
    try:
        from pydub import AudioSegment  # type: ignore
        seg = AudioSegment.from_file(str(src_path))
        seg.export(str(dst), format="wav")
        return dst
    except Exception as e:
        raise RuntimeError(f"final wav write failed: {type(e).__name__}: {e}")

def _normalize_section_headers(lyrics: str) -> str:
    """
    섹션 헤더 오타/표기 정규화:
      - [corus] -> [chorus]
      - 대소문자/공백/양끝 공백 정리
    """
    import re

    map_norm = {
        "intro": "intro",
        "verse": "verse",
        "chorus": "chorus",
        "corus": "chorus",     # 오타 교정
        "bridge": "bridge",
        "outro": "outro",
    }

    out_lines: list[str] = []
    for raw in str(lyrics or "").splitlines():
        s = (raw or "").strip()
        m = re.match(r"^\s*\[\s*([A-Za-z]+)\s*\]\s*$", s)
        if m:
            key = m.group(1).lower()
            if key in map_norm:
                out_lines.append(f"[{map_norm[key]}]")
                continue
        out_lines.append(s)
    return "\n".join(out_lines)


def _preprocess_lyrics_for_ace(lyrics: str) -> str:
    """
    ACE-Step에 전달할 가사 전처리.
    요청에 따라 섹션 태그를 '그대로' 보존하며, 빈 줄/연속 중복만 정리.
    - [corus] 같은 오타는 [chorus]로 자동 교정
    - 연속 같은 줄은 1회만 남김
    """
    import re

    # 1) 섹션 헤더 정규화
    s = _normalize_section_headers(lyrics)

    # 2) 공백 정리 + 연속 중복 제거
    out: list[str] = []
    prev = None
    for ln in s.splitlines():
        t = re.sub(r"\s+", " ", (ln or "")).strip()
        # 공백 라인은 그대로 허용(섹션 사이 구분용), 단 연속 빈 줄은 1줄로
        if t == "":
            if prev == "":
                continue
            out.append("")
            prev = ""
            continue
        if t == prev:
            continue
        out.append(t)
        prev = t

    return "\n".join(out).strip()








