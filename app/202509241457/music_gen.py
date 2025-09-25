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
from typing import Optional, List, Tuple, Iterable
from pathlib import Path
import random
import requests

from typing import Any

# settings 상수(대기/폴링 주기)

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

    추가 보정(기능 불변, 참조만 정리):
      - 문자열 형태 노드 참조 '#74' → '74'
      - 잘못된 3원소 참조 예: ["문자태그", "17", 0] → ["17", 0]
      - {"nodes":[...]} / {id→node} 두 포맷 모두 정규화
    """
    from pathlib import Path
    import json, re
    from typing import Any

    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"워크플로 JSON 없음: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1) 'prompt' 키로 감싸져 있으면 내부만 꺼냄
    g: Any = data.get("prompt") if isinstance(data, dict) else None
    if not isinstance(g, (dict, list)):
        g = data  # prompt 키가 없으면 전체가 그래프라고 가정

    # 2) {"nodes":[{id:.., class_type:..}, ...]} → {id(str): node(dict)}
    if isinstance(g, dict) and "nodes" in g and isinstance(g["nodes"], list):
        nodes_list = g["nodes"]
        tmp: dict[str, dict] = {}
        for n in nodes_list:
            if not isinstance(n, dict):
                continue
            nid = str(n.get("id") or "").strip()
            if not nid:
                raise ValueError("워크플로 형식 오류: nodes[*].id 가 없음")
            node = {k: v for k, v in n.items() if k != "id"}
            tmp[nid] = node
        g = tmp

    # 3) 최종은 dict 여야 함
    if not isinstance(g, dict):
        raise ValueError("워크플로 형식 오류: prompt 그래프가 dict 가 아님")

    # 4) 각 노드 검증
    bad = [nid for nid, node in g.items() if not (isinstance(node, dict) and "class_type" in node)]
    if bad:
        raise ValueError(
            f"워크플로 노드에 class_type 누락: {', '.join(bad[:10])}" + ("..." if len(bad) > 10 else "")
        )

    # ──────────────── ★ 참조 정규화 블록 (여기서만 수행) ────────────────
    rx_hash_id = re.compile(r"^#(\d+)$")

    def _unhash(x: Any) -> Any:
        if isinstance(x, str):
            m = rx_hash_id.match(x)
            if m:
                return m.group(1)
        return x

    def _fix_seq(seq: list[Any]) -> list[Any]:
        out = [_unhash(v) for v in seq]
        # ["문자", "17", 0] → ["17", 0]
        if len(out) == 3 and isinstance(out[0], str) and not out[0].isdigit() and isinstance(out[1], (str, int)) and out[2] == 0:
            return [str(out[1]), 0]
        # ["#17", 0] → ["17", 0]
        if len(out) == 2 and isinstance(out[0], str):
            return [str(_unhash(out[0])), out[1]]
        return out

    def _normalize_inputs(ins: dict[str, Any]) -> None:
        for k, v in list(ins.items()):
            if isinstance(v, str):
                ins[k] = _unhash(v)
            elif isinstance(v, list):
                ins[k] = _fix_seq(v)
            elif isinstance(v, dict):
                # 1단계 깊이만 정리(워크플로 입력 스키마 상 충분)
                for kk, vv in list(v.items()):
                    if isinstance(vv, str):
                        v[kk] = _unhash(vv)
                    elif isinstance(vv, list):
                        v[kk] = _fix_seq(vv)

    # dict(id→node) 전체 순회하며 inputs 정규화
    for _nid, node in g.items():
        if not isinstance(node, dict):
            continue
        ins = node.get("inputs")
        if isinstance(ins, dict):
            _normalize_inputs(ins)
            # TextEncodeAceStepAudio의 lyrics가 '#숫자'면 ["숫자", 0]로 보정
            if node.get("class_type") == "TextEncodeAceStepAudio":
                val = ins.get("lyrics")
                if isinstance(val, list):
                    ins["lyrics"] = _fix_seq(val)
                elif isinstance(val, str):
                    m = rx_hash_id.match(val)
                    if m:
                        ins["lyrics"] = [m.group(1), 0]
    # ────────────────────────────────────────────────────────────────

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


def _rand_seed() -> int:
    return random.randint(1, 2_147_483_646)


# ────────────────────── 저장 노드 '적응' 로직(핵심) ────────────────────────────
# --- NEW: 서버에 설치된 노드 클래스 목록 가져오기 (없으면 빈 set) ---




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



# ───────────────────────────── 제출/폴링 ──────────────────────────────────────


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

import subprocess
from pathlib import Path
from typing import List, Optional, Callable

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







def generate_music_with_acestep(
    project_dir: str,
    *,
    on_progress: Optional[Callable[[dict], None]] = None,
    target_seconds: int | None = None,
) -> str:
    """
    ComfyUI(ACE-Step)로 단일 트랙 음악 생성.
    - SaveAudio는 WAV로 강제 → <project_dir>/vocal.wav
    - LLS(가사 변환)가 이미 수행되어 있다면: 변환된 가사(우선순위)에 기반해 생성
      우선순위: _lls_after.txt > lyrics_lls.txt > meta['lyrics_lls'] > meta['lyrics']
    - 생성 후 LLS 텍스트가 outputs로 떨어지면 _lls_after.txt/lyrics_lls.txt/meta 갱신(기존 로직 유지)
    - 길이(초): 사용자가 입력한 값을 그대로 사용 (스냅/분 환산 제거)
    """
    from pathlib import Path

    def notify(stage: str, **kw):
        if on_progress:
            try:
                info = {"stage": stage}
                info.update(kw)
                on_progress(info)
            except Exception:
                pass

    # ───────────────────────── 내부 헬퍼 ─────────────────────────
    def _iter_nodes(graph: dict):
        """{nodes:[...]}/ {id:{...}} 두 형식 모두 지원."""
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

    def _force_wav_save_for_this_graph(graph: dict, *, proj_dir: Path, prefix: str) -> str:
        """
        워크플로의 SaveAudio 계열 노드를 WAV 저장으로 강제.
        - filename_prefix/output_path/basename/base_filename 등 경로 키 보정
        - WAV 관련 키가 존재하면 합리적 기본값 지정
        반환: ".wav"
        """
        for node in _iter_nodes(graph):
            ct = str(node.get("class_type", "")).lower()
            if ct in ("saveaudio", "saveaudiowav", "saveaudiomp3", "pysssss_saveaudio"):
                ins = node.setdefault("inputs", {}) or {}
                ins["filename_prefix"] = prefix
                if "output_path" in ins:
                    ins["output_path"] = str(proj_dir)
                if "basename" in ins:
                    ins["basename"] = "vocal"
                if "base_filename" in ins:
                    ins["base_filename"] = "vocal"
                for k, v in (("format", "wav"), ("container", "wav"), ("codec", "pcm_s16le")):
                    if k in ins:
                        ins[k] = v
                if "sample_rate" in ins:
                    ins.setdefault("sample_rate", 44100)
                for bd_key in ("bit_depth", "bitdepth", "bits"):
                    if bd_key in ins:
                        ins[bd_key] = 16
        return ".wav"

    def _find_lls_ref(graph: dict) -> Optional[list]:
        """LyricsLangSwitch 노드의 출력 핸들을 찾아서 ['<nid>', 0] 형태로 반환."""
        for nid, _node in _find_nodes_by_class_names(graph, ("LyricsLangSwitch",)):
            return [str(nid), 0]
        return None

    def _retarget_lls_text_saver(graph: dict, *, proj_dir: Path) -> None:
        """
        LLS(LyricsLangSwitch) 텍스트 저장 노드를 프로젝트 폴더로 리타겟팅 → _lls_after.txt
        - SaveText/SaveString/pysssss text saver 류를 안전하게 덮어씀
        - text 입력이 비어 있으면 LLS 출력으로 연결
        """
        lls_ref = _find_lls_ref(graph)
        for node in _iter_nodes(graph):
            ct = str(node.get("class_type", "")).lower()
            if "savetext" in ct or ("pysssss" in ct and "text" in ct) or "savestring" in ct:
                ins = node.setdefault("inputs", {}) or {}
                if not ins.get("text") and lls_ref:
                    ins["text"] = lls_ref
                ins["root_dir"] = str(proj_dir)
                ins["file"] = "_lls_after.txt"
                for k in ("filename", "output_path", "filename_prefix", "basename", "base_filename"):
                    if k == "filename":
                        ins[k] = str(proj_dir / "_lls_after.txt")
                    elif k == "output_path":
                        ins[k] = str(proj_dir)
                    elif k == "filename_prefix":
                        ins[k] = "_lls_after"
                    elif k in ("basename", "base_filename"):
                        ins[k] = "_lls_after"

    def _copy_noclobber(src: Path, dst: Path) -> Path:
        """덮어쓰기 금지: 존재하면 _v2,_v3…로 복사."""
        import shutil
        cand = Path(dst)
        i = 2
        while cand.exists():
            cand = cand.with_name(f"{dst.stem}_v{i}{dst.suffix}")
            i += 1
        shutil.copy2(src, cand)
        return cand

    def _as_list(v):
        if v is None:
            return []
        return v if isinstance(v, list) else [v]

    # ───────────────────────── 기본 준비 ─────────────────────────
    _dlog("ENTER", f"project_dir={project_dir}")
    proj = Path(project_dir)
    proj.mkdir(parents=True, exist_ok=True)
    pj = proj / "project.json"
    meta = load_json(pj, {}) or {}

    title = effective_title(meta)

    # 원본/변환 가사 확보
    lyrics_raw = (meta.get("lyrics") or "").strip()
    if not lyrics_raw:
        raise RuntimeError("project.json에 가사가 없습니다. 먼저 저장/생성해 주세요.")

    # 변환 가사 우선 사용(생성에는 사용하지만, 원본은 보존)
    lls_after_path = proj / "_lls_after.txt"
    lyrics_lls_path = proj / "lyrics_lls.txt"
    lyrics_eff = lyrics_raw
    try:
        if lls_after_path.exists():
            lyrics_eff = lls_after_path.read_text(encoding="utf-8", errors="ignore").strip() or lyrics_eff
        elif lyrics_lls_path.exists():
            lyrics_eff = lyrics_lls_path.read_text(encoding="utf-8", errors="ignore").strip() or lyrics_eff
        elif (meta.get("lyrics_lls") or "").strip():
            lyrics_eff = str(meta.get("lyrics_lls")).strip()
    except Exception:
        pass

    # 실행 시작 시 원본 가사 백업(원본 유지)
    try:
        (proj / "lyrics.txt").write_text(lyrics_raw, encoding="utf-8", errors="ignore")
    except Exception:
        pass
    meta["lyrics"] = lyrics_raw
    save_json(pj, meta)

    # ⛳ 길이(초): 입력값 그대로 사용
    if target_seconds is not None:
        seconds = int(max(1, target_seconds))
    else:
        seconds = int(max(1, int(meta.get("target_seconds") or meta.get("time") or 60)))

    # 메타(초) 통일 저장
    meta["target_seconds"] = int(seconds)
    meta["time"] = int(seconds)
    save_json(pj, meta)

    effective_tags = _collect_effective_tags(meta)

    # ───────────────────── 워크플로 로드/보정 ─────────────────────
    g = _load_workflow_graph(ACE_STEP_PROMPT_JSON)

    base = _choose_host()
    _dlog("HOST", base, "| DESIRED_FMT wav")

    # ComfyUI 출력 폴더용 prefix + 로컬 최종 파일명
    subfolder = f"shorts_make/{sanitize_title(title)}"
    prefix = f"{subfolder}/vocal_final"
    out_path = str(proj / "vocal.wav")

    # 저장 노드 WAV 강제(이 워크플로에 맞춤)
    ext = _force_wav_save_for_this_graph(g, proj_dir=proj, prefix=prefix)  # '.wav'
    _dlog("SAVE-NODE-EXT", ext)

    # LLS 텍스트 저장 경로 리타겟팅(있으면)
    _retarget_lls_text_saver(g, proj_dir=proj)

    # 가사/태그/길이 반영 (변환 가사 주입, 초 단위 그대로)
    for _nid, node in _find_nodes_by_class_names(g, ("LyricsLangSwitch",)):
        ins = node.setdefault("inputs", {})
        ins["lyrics"] = lyrics_eff
        ins["language"] = "Korean"
        ins.setdefault("threshold", 0.85)
        ins["seconds"] = int(seconds)

    for _nid, node in _find_nodes_by_class_names(g, ("TextEncodeAceStepAudio",)):
        ins = node.setdefault("inputs", {})
        ins["tags"] = ", ".join(effective_tags)
        ins.setdefault("lyrics_strength", 1.0)

    targets = []
    targets += _find_nodes_by_class_names(g, ("EmptyAceStepLatentAudio", "EmptyLatentAudio", "EmptyAudio", "NoiseLatentAudio"))
    if not targets:
        for nid, node in _find_nodes_by_class_contains(g, "audio"):
            if "latent" in str(node.get("class_type", "")).lower():
                targets.append((nid, node))
    for _nid, node in targets:
        node.setdefault("inputs", {})["seconds"] = int(max(1, seconds))

    # 시드 랜덤
    try:
        for nid, node in list(g.items()):
            if str(node.get("class_type", "")).lower() == "ksampler":
                node.setdefault("inputs", {})["seed"] = _rand_seed()
    except Exception:
        pass

    # 프롬프트 저장(전/후 LLS 비교용)
    debug_path = proj / "_prompt_sent.json"
    save_json(debug_path, {"prompt": g})
    _dlog("PROMPT-SAVED", str(debug_path))

    # ───────────────────── 제출 & 대기 ─────────────────────
    notify("submitting", host=base)
    _dlog("SUBMIT")
    hist = _submit_and_wait(
        base, g,
        timeout=(globals().get("ACE_STEP_WAIT_TIMEOUT_SEC") or globals().get("ACE_WAIT_TIMEOUT_SEC", 1800.0)),
        poll=(globals().get("ACE_STEP_POLL_INTERVAL_SEC") or globals().get("ACE_POLL_INTERVAL_SEC", 2.0)),
        on_progress=(on_progress or (lambda info: _dlog("PROG", info))),
    )

    # 히스토리 저장
    hist_path = proj / "_history_raw.json"
    save_json(hist_path, hist)
    _dlog("HISTORY-SAVED", str(hist_path))

    # ───────────────────── 결과 다운로드(LLS 텍스트 포함) ──────────────────
    saved_files: List[Path] = []
    outputs = hist.get("outputs") or {}
    _dlog("HISTORY-OUTPUT-KEYS", list(outputs.keys()) if isinstance(outputs, dict) else type(outputs))

    # 1차: 표준 파일 다운로드
    if isinstance(outputs, dict):
        for nid, node_out in outputs.items():
            _dlog("NODE-OUTPUT", f"nid={nid}", f"keys={list(node_out.keys())}")
            for key in ("audio", "audios", "files"):
                arr = _as_list(node_out.get(key))
                _dlog("NODE-OUTPUT-LIST", f"key={key}", f"count={len(arr)}")
                for item in arr:
                    if not isinstance(item, dict):
                        continue
                    fn = (item.get("filename") or item.get("name") or "").strip()
                    sf = (item.get("subfolder") or "").strip()
                    if not sf or fn.startswith("ComfyUI_temp_"):
                        _dlog("SKIP-PREVIEW", f"nid={nid}", f"fn={fn!r}", f"sf={sf!r}")
                        continue
                    sf_norm = sf.replace("\\", "/").lstrip("/")
                    out_file = _download_output_file(base, fn, sf_norm, out_dir=proj)
                    if out_file:
                        saved_files.append(out_file)

    # 2차(백업): LLS 텍스트가 파일로 안 떨어졌다면, outputs의 text/txt를 수집해 _lls_after.txt로 저장
    lls_txt_path = proj / "_lls_after.txt"
    if not lls_txt_path.exists():
        lls_text_buf = []
        if isinstance(outputs, dict):
            for _nid, node_out in outputs.items():
                for key in ("text", "txt"):
                    arr = _as_list(node_out.get(key))
                    for item in arr:
                        if isinstance(item, str) and item.strip():
                            lls_text_buf.append(item.strip())
        if lls_text_buf:
            try:
                lls_txt_path.write_text("\n".join(lls_text_buf), encoding="utf-8", errors="ignore")
                _dlog("TEXT-SAVED(BACKUP)", str(lls_txt_path))
            except Exception as e:
                _dlog("TEXT-SAVE-FAIL", type(e).__name__, str(e))

    _dlog("SAVED-FILES", [str(p) for p in saved_files])

    # ───────────────── WAV 통일 + 마스터링 ─────────────────
    final_path: Optional[Path] = None
    if saved_files:
        audio_candidates = [p for p in saved_files if p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a")]
        src = max(audio_candidates or saved_files, key=lambda p: p.stat().st_mtime)
        _dlog("PICKED-SRC", str(src))
        ff = getattr(S, "FFMPEG_EXE", "ffmpeg")

        # 1) 포맷 통일: vocal.wav
        final_path = _ensure_vocal_wav(src, proj, ffmpeg_exe=ff)

        # 2) 정밀 마스터(옵션) — 존재하면 호출, 없으면 스킵
        master_fn = globals().get("_master_wav_precise")
        if callable(master_fn):
            try:
                final_path = master_fn(
                    final_path,
                    I=getattr(S, "MASTER_TARGET_I", -12.0),
                    TP=getattr(S, "MASTER_TARGET_TP", -1.0),
                    LRA=getattr(S, "MASTER_TARGET_LRA", 11.0),
                    ffmpeg_exe=ff,
                )
            except TypeError:
                try:
                    final_path = master_fn(
                        final_path,
                        I=getattr(S, "MASTER_TARGET_I", -12.0),
                        TP=getattr(S, "MASTER_TARGET_TP", -1.0),
                        LRA=getattr(S, "MASTER_TARGET_LRA", 11.0),
                    )
                except Exception as _e:
                    _dlog("MASTER-FAIL", type(_e).__name__, str(_e))
            except Exception as _e:
                _dlog("MASTER-FAIL", type(_e).__name__, str(_e))
        else:
            _dlog("MASTER-SKIP", "no _master_wav_precise")

        _dlog("FINAL-PATH", str(final_path), f"exists={final_path.exists()}", f"size={final_path.stat().st_size}")
    else:
        _dlog("NO-SAVED-FILES", "Check _history_raw.json for SaveAudio/SaveText outputs")

    # ─── LLS 결과가 있으면 최종 가사로 반영 (원본 가사는 보존) ───
    try:
        if lls_txt_path.exists():
            lls_after = lls_txt_path.read_text(encoding="utf-8", errors="ignore").strip()
            if lls_after:
                (proj / "lyrics_lls.txt").write_text(lls_after, encoding="utf-8", errors="ignore")
                meta["lyrics_lls"] = lls_after
                # ❗ 원본 가사(meta["lyrics"])는 변경하지 않음
                save_json(pj, meta)
    except Exception as _e:
        _dlog("LLS-LYRICS-SAVE-FAIL", type(_e).__name__, str(_e))

    # ─────────── 사용자 라이브러리 복사(덮어쓰기 방지) ───────────
    try:
        save_root = getattr(S, "FINAL_OUT", "") or ""
        if final_path and final_path.exists() and save_root:
            dst_hint = Path(save_to_user_library("audio", title, final_path))
            try:
                if Path(dst_hint).resolve().samefile(final_path.resolve()):
                    _dlog("COPY-SKIPPED", "samefile(src==dst)")
                else:
                    dst_versioned = _copy_noclobber(final_path, Path(dst_hint))
                    meta.setdefault("paths", {})["vocal_user_dir"] = str(dst_versioned)
                    _dlog("COPIED", f"{final_path} -> {dst_versioned}")
            except Exception:
                if str(Path(dst_hint).resolve()).lower() == str(final_path.resolve()).lower():
                    _dlog("COPY-SKIPPED", "samefile(str-compare)")
                else:
                    dst_versioned = _copy_noclobber(final_path, Path(dst_hint))
                    meta.setdefault("paths", {})["vocal_user_dir"] = str(dst_versioned)
                    _dlog("COPIED", f"{final_path} -> {dst_versioned}")
        else:
            _dlog("COPY-SKIPPED", f"final_path_ok={bool(final_path and final_path.exists())}", f"FINAL_OUT={bool(save_root)}")
    except Exception as e:
        _dlog("COPY-FAIL", f"{type(e).__name__}: {e}")

    # 메타/디버그 업데이트
    meta.setdefault("comfy_debug", {})
    meta["comfy_debug"].update({
        "host": base,
        "prompt_json": str(ACE_STEP_PROMPT_JSON),
        "prompt_seconds": seconds,
        "requested_format": "wav",
        "requested_ext": ext,
        "subfolder": subfolder,
    })
    meta["tags_effective"] = effective_tags
    if final_path:
        meta["audio"] = str(final_path)
        meta.setdefault("paths", {})["vocal"] = str(final_path)
    save_json(pj, meta)

    # LLS 전/후 비교 문자열 병기(원본 유지)
    try:
        dbg = load_json(debug_path) or {"prompt": g}
        dbg["lyrics_before_lls"] = (proj / "lyrics.txt").read_text(encoding="utf-8", errors="ignore")[:10000] if (proj / "lyrics.txt").exists() else lyrics_raw
        if lls_txt_path.exists():
            dbg["lyrics_after_lls"] = lls_txt_path.read_text(encoding="utf-8", errors="ignore")[:10000]
        save_json(debug_path, dbg)
    except Exception as _e:
        _dlog("LLS-DUMP-MERGE-FAIL", type(_e).__name__, str(_e))

    msg = [
        "ACE-Step 완료 ✅",
        f"- 프롬프트: {ACE_STEP_PROMPT_JSON}",
        f"- 길이:     {seconds}s",
        f"- 태그 수:  {len(effective_tags)}",
    ]
    if final_path:
        msg.append(f"- 저장:     {final_path}")
    else:
        msg.append("- 저장:     (다운로드/변환 실패 — _history_raw.json 확인)")
    summary = "\n".join(msg)
    _dlog("LEAVE", summary.replace("\n", " | "))
    return summary

