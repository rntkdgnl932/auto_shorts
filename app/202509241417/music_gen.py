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
    SaveText|pysssss를 'output/<subfolder>/_lls_after.txt'로 리타겟팅.
    없으면 dict(id->node) 포맷에서만 LLS(실제 id) 출력(0)에 연결하는 새 노드 생성.
    ※ 더미 id(예: "74" 하드코드) 절대 금지.
    ※ 무엇을 어떻게 연결했는지 상세 로그 출력.
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

    # 1) 기존 SaveText|pysssss 수정
    for node in _iter_nodes(graph):
        if str(node.get("class_type") or "") != "SaveText|pysssss":
            continue
        ins = node.setdefault("inputs", {}) or {}
        before = dict(ins)
        ins["root_dir"] = "output"
        ins["file"] = out_rel
        ins["append"] = "overwrite"
        ins["insert"] = False
        _dlog("[LLS-SAVE] retarget existing SaveText|pysssss:",
              "before=", {k: before.get(k) for k in ("root_dir", "file", "append", "insert", "text")},
              "after=", {k: ins.get(k) for k in ("root_dir", "file", "append", "insert", "text")})
        changed = True

    if changed:
        return True

    # 2) 없으면 새로 만들기 — dict(id->node)에서만
    if "nodes" in graph and isinstance(graph["nodes"], list):
        _dlog("[LLS-SAVE] graph is list-format; skip creating new SaveText.")
        return False

    # LLS 실제 id 찾기(키가 곧 id)
    lls_id = None
    for k, n in graph.items():
        if isinstance(n, dict) and str(n.get("class_type") or "") == "LyricsLangSwitch":
            lls_id = str(k)
            break

    if not lls_id:
        _dlog("[LLS-SAVE] LyricsLangSwitch not found; skip creating new SaveText.")
        return False

    # 유니크 id 생성
    used_ids = {str(k) for k in graph.keys() if isinstance(k, str)}
    new_id = "save_lls_text"; i = 1
    while new_id in used_ids:
        new_id = f"save_lls_text_{i}"; i += 1

    graph[new_id] = {
        "inputs": {
            "root_dir": "output",
            "file": out_rel,
            "append": "overwrite",
            "insert": False,
            "text": [lls_id, 0],
        },
        "class_type": "SaveText|pysssss",
        "_meta": {"title": "Save Text 🐍 (auto)"}
    }
    _dlog(f"[LLS-SAVE] created SaveText|pysssss id={new_id}  text=[{lls_id}, 0]  file={out_rel}")
    return True


def _iter_nodes_generic(graph: dict):
    """
    ComfyUI 그래프를 노드 dict generator로 통일해서 돌려준다.
    - {"nodes":[{...}, ...]} 또는 {"59":{...}, "74":{...}} 모두 처리
    """
    if not isinstance(graph, dict):
        return
    if "nodes" in graph and isinstance(graph["nodes"], list):
        for n in graph["nodes"]:
            if isinstance(n, dict) and "class_type" in n:
                yield n
    else:
        for _k, n in graph.items():
            if isinstance(n, dict) and "class_type" in n:
                yield n


def _debug_dump_connections(graph: dict, label: str = "") -> None:
    """
    노드 연결 레퍼런스(예: ["74", 0] 또는 ["#74", 0])를 전수 조사해서 프린트한다.
    - '#숫자' 같이 해시가 섞인 레퍼런스도 경고 출력
    - 존재하지 않는 id를 가리키는지 여부도 간단 체크
    """
    try:
        print(f"[WFDBG] ===== {label or 'DUMP'} =====", flush=True)
        # id 수집 (list 스키마엔 id 키가 없을 수 있어 None 허용)
        node_ids = set()
        for n in _iter_nodes_generic(graph):
            nid = n.get("id")
            if nid is not None:
                node_ids.add(str(nid))

        # 출력
        for n in _iter_nodes_generic(graph):
            nid = str(n.get("id"))
            ctype = str(n.get("class_type", ""))
            print(f"[WFDBG] node id={nid} type={ctype}", flush=True)

            ins = n.get("inputs") or {}
            for k, v in list(ins.items()):
                # 연결 형태만 추려서 표기: ["74", 0] 또는 [74, 0]
                if isinstance(v, list) and len(v) == 2:
                    rid_raw, out_idx = v[0], v[1]
                    rid = str(rid_raw)
                    hash_warn = rid.startswith("#")
                    rid_norm = rid.lstrip("#")

                    missing = (rid_norm not in node_ids) if node_ids else False
                    print(
                        f"[WFDBG]   input.{k} -> ref id={rid} (out={out_idx})"
                        f"{'  [WARN: leading #]' if hash_warn else ''}"
                        f"{'  [WARN: missing id]' if missing else ''}",
                        flush=True
                    )
                # 입력에 심볼릭 연결이 dict로 들어오는 경우(거의 없음)도 대비
                elif isinstance(v, dict) and "node_id" in v and "output" in v:
                    rid = str(v.get("node_id"))
                    hash_warn = rid.startswith("#")
                    rid_norm = rid.lstrip("#")
                    missing = (rid_norm not in node_ids) if node_ids else False
                    print(
                        f"[WFDBG]   input.{k} -> ref node_id={rid} (out={v.get('output')})"
                        f"{'  [WARN: leading #]' if hash_warn else ''}"
                        f"{'  [WARN: missing id]' if missing else ''}",
                        flush=True
                    )
        print(f"[WFDBG] ===== /{label or 'DUMP'} =====", flush=True)
    except Exception as e:
        # 디버그 함수는 절대 죽지 않게 — 경고만 찍고 무시
        print(f"[WFDBG] dump failed: {type(e).__name__}: {e}", flush=True)


def _strip_hash_from_refs(graph: dict) -> int:
    """
    (선택) 해시가 붙은 레퍼런스 ["#74", 0] → ["74", 0] 로 보정.
    실제 submit 전에 호출하면 '#숫자'로 인한 오류를 피할 수 있다.
    반환: 수정된 레퍼런스 개수
    """
    fixed = 0
    try:
        for n in _iter_nodes_generic(graph):
            ins = n.get("inputs") or {}
            for k, v in list(ins.items()):
                if isinstance(v, list) and len(v) == 2:
                    rid_raw = v[0]
                    if isinstance(rid_raw, str) and rid_raw.startswith("#"):
                        ins[k] = [rid_raw.lstrip("#"), v[1]]
                        fixed += 1
                elif isinstance(v, dict) and "node_id" in v:
                    rid_raw = v.get("node_id")
                    if isinstance(rid_raw, str) and rid_raw.startswith("#"):
                        v["node_id"] = rid_raw.lstrip("#")
                        fixed += 1
    except Exception:
        # 안전을 위해 조용히 무시
        return fixed
    return fixed






def generate_music_with_acestep(
    project_dir: str,
    *,
    on_progress: Optional[Callable[[dict], None]] = None,
    target_seconds: int | None = None,
) -> str:
    """
    ComfyUI(ACE-Step)로 음악 생성:
      - SaveAudio 계열을 WAV 저장으로 강제 → <project_dir>/vocal.wav
      - LyricsLangSwitch(LLS) 결과 텍스트를 <project_dir>/_lls_after.txt 로 '항상' 저장
        · 1차: 워크플로 내 SaveText/SaveString 노드를 프로젝트 폴더로 리타겟팅
        · 2차: 히스토리(outputs[text/txt])에서라도 주워 담아 저장(백업 플로우)
      - 정밀 마스터링(옵션) 적용
      - 사용자 라이브러리(FINAL_OUT) 복사(덮어쓰기 방지 _v2,_v3…)
      - 메타(project.json)에 lyrics/lyrics_lls 갱신
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
                # 포맷 지시(키가 있을 때만)
                for k, v in (("format", "wav"), ("container", "wav"), ("codec", "pcm_s16le")):
                    if k in ins:
                        ins[k] = v
                # 샘플레이트/비트뎁스 기본값
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
                    ins["text"] = lls_ref  # LLS 출력에 연결
                # 파일 경로를 프로젝트 폴더로 고정
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
    proj = Path(project_dir); proj.mkdir(parents=True, exist_ok=True)
    pj = proj / "project.json"
    meta = load_json(pj, {}) or {}

    title = effective_title(meta)
    lyrics = (meta.get("lyrics") or "").strip()
    if not lyrics:
        raise RuntimeError("project.json에 가사가 없습니다. 먼저 저장/생성해 주세요.")

    # ★ 실행 시작 시 원본 가사를 파일/메타에 보존
    try:
        (proj / "lyrics.txt").write_text(lyrics, encoding="utf-8", errors="ignore")
    except Exception:
        pass
    meta["lyrics"] = lyrics
    save_json(pj, meta)

    # 길이(초) 결정 + 구버전(1/2/3=분) 보정 + 스냅(20→30 정책)
    if target_seconds is not None:
        seconds = int(max(1, target_seconds))
    else:
        seconds = int(meta.get("target_seconds") or meta.get("time") or 60)
    if seconds in (1, 2, 3):
        seconds *= 60
    seconds = min({30, 60, 120, 180}, key=lambda x: abs(x - int(seconds)))

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

    # 가사/태그/길이 반영
    for _nid, node in _find_nodes_by_class_names(g, ("LyricsLangSwitch",)):
        ins = node.setdefault("inputs", {})
        ins["lyrics"] = lyrics
        ins["language"] = "Korean"
        ins.setdefault("threshold", 0.85)
        ins["seconds"] = int(max(10, seconds))

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
    for nid, node in list(g.items()):
        if str(node.get("class_type", "")).lower() == "ksampler":
            node.setdefault("inputs", {})["seed"] = _rand_seed()

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

    # 1차: 표준 파일 다운로드 (오디오/텍스트 saver가 파일로 남긴 것)
    if isinstance(outputs, dict):
        for nid, out in outputs.items():
            _dlog("NODE-OUTPUT", f"nid={nid}", f"keys={list(out.keys())}")
            for key in ("audio", "audios", "files"):
                arr = _as_list(out.get(key))
                _dlog("NODE-OUTPUT-LIST", f"key={key}", f"count={len(arr)}")
                for item in arr:
                    if not isinstance(item, dict):
                        continue
                    fn = (item.get("filename") or item.get("name") or "").strip()
                    sf = (item.get("subfolder") or "").strip()
                    # 미리보기 임시파일/빈 서브폴더 스킵
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
            for _nid, out in outputs.items():
                for key in ("text", "txt"):
                    arr = _as_list(out.get(key))
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
        # 오디오 파일만 추려 최신으로
        audio_candidates = [p for p in saved_files if p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a")]
        src = max(audio_candidates or saved_files, key=lambda p: p.stat().st_mtime)
        _dlog("PICKED-SRC", str(src))
        ff = getattr(S, "FFMPEG_EXE", "ffmpeg")
        # 1) 포맷 통일: vocal.wav
        final_path = _ensure_vocal_wav(src, proj, ffmpeg_exe=ff)
        # 2) 정밀 마스터(옵션) — 두 시그니처 모두 시도
        try:
            final_path = _master_wav_precise(
                final_path,
                I=getattr(S, "MASTER_TARGET_I", -12.0),
                TP=getattr(S, "MASTER_TARGET_TP", -1.0),
                LRA=getattr(S, "MASTER_TARGET_LRA", 11.0),
                ffmpeg_exe=ff,
            )
        except TypeError:
            try:
                final_path = _master_wav_precise(
                    final_path,
                    I=getattr(S, "MASTER_TARGET_I", -12.0),
                    TP=getattr(S, "MASTER_TARGET_TP", -1.0),
                    LRA=getattr(S, "MASTER_TARGET_LRA", 11.0),
                )
            except Exception as _e:
                _dlog("MASTER-FAIL", type(_e).__name__, str(_e))
        except Exception as _e:
            _dlog("MASTER-FAIL", type(_e).__name__, str(_e))
        _dlog("FINAL-PATH", str(final_path), f"exists={final_path.exists()}", f"size={final_path.stat().st_size}")
    else:
        _dlog("NO-SAVED-FILES", "Check _history_raw.json for SaveAudio/SaveText outputs")

    # ─── LLS 결과가 있으면 최종 가사로 반영 (파일/메타 저장) ───
    try:
        if lls_txt_path.exists():
            lls_after = lls_txt_path.read_text(encoding="utf-8", errors="ignore").strip()
            if lls_after:
                (proj / "lyrics_lls.txt").write_text(lls_after, encoding="utf-8", errors="ignore")
                meta["lyrics_lls"] = lls_after
                # 정책: 기본 lyrics도 LLS로 갱신(원문은 lyrics.txt에 이미 백업됨)
                meta["lyrics"] = lls_after
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
        "requested_ext": ext,             # '.wav'
        "subfolder": subfolder,
    })
    meta["tags_effective"] = effective_tags
    if final_path:
        meta["audio"] = str(final_path)
        meta.setdefault("paths", {})["vocal"] = str(final_path)
    save_json(pj, meta)

    # LLS 전/후 비교 문자열을 프롬프트 디버그에 병기
    try:
        dbg = load_json(debug_path) or {"prompt": g}
        dbg["lyrics_before_lls"] = (proj / "lyrics.txt").read_text(encoding="utf-8", errors="ignore")[:10000] if (proj / "lyrics.txt").exists() else lyrics
        if lls_txt_path.exists():
            dbg["lyrics_after_lls"] = lls_txt_path.read_text(encoding="utf-8", errors="ignore")[:10000]
        save_json(debug_path, dbg)
    except Exception as _e:
        _dlog("LLS-DUMP-MERGE-FAIL", type(_e).__name__, str(_e))

    # 요약
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


























def _apply_lyrics_tags_seconds_to_graph(
    graph: dict,
    *,
    lyrics: str,
    seconds: int,
    effective_tags: list[str],
    meta: dict,
) -> None:
    """
    - LLS에 lyrics/seconds_sing/offset류 주입
    - TextEncodeAceStepAudio에 tags 병합
    - seconds 보유 노드에 전체 길이 강제
    (※ 기능 변경 없음, 단지 무엇을 꽂았는지 로그만 강화)
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

    music_meta = (meta.get("music_meta") or {}) if isinstance(meta, dict) else {}
    intro_sec = int(music_meta.get("intro_sec") or 0)
    outro_sec = int(music_meta.get("outro_sec") or 0)
    seconds_total = int(max(1, seconds))
    seconds_sing = max(10, seconds_total - max(0, intro_sec) - max(0, outro_sec))

    lyrics_for_lls = _preprocess_lyrics_for_ace(lyrics)

    # LLS 주입 + 로그
    for n in _iter_nodes(graph):
        if str(n.get("class_type") or "") == "LyricsLangSwitch":
            ins = n.setdefault("inputs", {}) or {}
            before = dict(ins)
            ins["lyrics"] = lyrics_for_lls
            ins["language"] = "Korean"
            ins.setdefault("threshold", 0.85)
            ins["seconds"] = int(seconds_sing)
            for k in ("offset", "head_silence", "lead_in"):
                if k in ins:
                    ins[k] = int(max(0, intro_sec))
            _dlog("[WIRE-LLS] LyricsLangSwitch inputs:",
                  "seconds_sing=", seconds_sing,
                  "intro_sec=", intro_sec,
                  "changed_keys=", [k for k in ins.keys()],
                  "had_keys_before=", [k for k in before.keys()])

    # 구조/인트로/아웃트로 태그 merge (동작 동일)
    try:
        struct = _collect_structure_tags_from_lyrics(lyrics_for_lls)
    except Exception:
        struct = []
    lead_tags = []
    if intro_sec > 0:
        lead_tags += [f"instrumental intro {intro_sec}s", f"no vocals until {intro_sec}s", "soft fade-in before vocals"]
    if outro_sec > 0:
        lead_tags += [f"outro {outro_sec}s", "fade-out ending"]
    merged_tags = list(effective_tags) + struct + lead_tags

    for n in _iter_nodes(graph):
        if str(n.get("class_type") or "") == "TextEncodeAceStepAudio":
            ins = n.setdefault("inputs", {}) or {}
            prev = ins.get("tags", "")
            if isinstance(prev, list):
                prev_s = ", ".join(prev)
            else:
                prev_s = str(prev)
            extra = ", ".join(merged_tags)
            ins["tags"] = (prev_s.strip() + ", " + extra) if prev_s.strip() else extra
            ins.setdefault("lyrics_strength", 1.0)
            _dlog("[WIRE-TEXTENC] tags merged length=",
                  len(ins.get("tags", "")), "lyrics_strength=", ins.get("lyrics_strength"))

    # seconds 보유 노드 보정 + 로그
    for n in _iter_nodes(graph):
        ins = n.setdefault("inputs", {}) or {}
        if "seconds" in ins:
            try:
                before = ins["seconds"]
                ins["seconds"] = int(seconds_total)
                _dlog(f"[WIRE-SEC] {n.get('class_type')} seconds: {before} -> {ins['seconds']}")
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
    - 섹션 태그는 보존
    - [corus] 같은 오타는 [chorus]로 교정
    - 공백 정리 및 연속 중복 라인 제거(같은 줄이 연이어 나오면 1회만 남김)
    """
    import re

    # 섹션 헤더 오타/표기 정규화
    def _normalize_section_headers(text: str) -> str:
        map_norm = {
            "intro": "intro",
            "verse": "verse",
            "chorus": "chorus",
            "corus": "chorus",   # 오타 교정
            "bridge": "bridge",
            "outro": "outro",
        }
        out_lines: list[str] = []
        for raw in str(text or "").splitlines():
            s = (raw or "").strip()
            m = re.match(r"^\s*\[\s*([A-Za-z]+)\s*\]\s*$", s)
            if m:
                key = m.group(1).lower()
                if key in map_norm:
                    out_lines.append(f"[{map_norm[key]}]")
                    continue
            out_lines.append(s)
        return "\n".join(out_lines)

    s = _normalize_section_headers(lyrics)

    # 공백 정리 + 연속 중복 제거
    out: list[str] = []
    prev = None
    for ln in s.splitlines():
        t = re.sub(r"\s+", " ", (ln or "")).strip()
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




def _add_intro_branch_and_dual_output(
    graph: dict,
    *,
    intro_sec: int,
    subfolder: str,
    intro_style: str = "matching",      # 기존 그대로 유지
    intro_tags: list[str] | None = None # ★ 새로 추가: 외부에서 만든 인트로 태그를 우선 사용
) -> dict:
    """
    그래프에 '인트로(무가사) 분지'를 추가한다.
    - EmptyAceStepLatentAudio(seconds=intro_sec)
    - TextEncodeAceStepAudio(가사X, lyrics_strength=0)  → positive (intro_tags 있으면 그대로 사용)
    - TextEncodeAceStepAudio(보컬 금지 네거티브)        → negative
    - KSampler(본곡과 동일 파라미터/seed)
    - VAEDecodeAudio (기존 VAE 공유)
    - SaveAudio (filename_prefix: f"{subfolder}/intro_part")
    """
    if not isinstance(graph, dict):
        return graph

    def _nodes_iter(g: dict):
        for k, v in g.items():
            if isinstance(v, dict) and "class_type" in v:
                yield str(k), v

    used_ids = []
    for k, _ in _nodes_iter(graph):
        try:
            used_ids.append(int(k))
        except ValueError:
            continue
    base_id = max(used_ids) + 1 if used_ids else 100

    def _new_id() -> str:
        nonlocal base_id
        nid = str(base_id)
        base_id += 1
        return nid

    node_enc_song = None
    node_zero = None
    node_ksampler_song = None
    for _id, n in _nodes_iter(graph):
        ct = str(n.get("class_type") or "")
        if ct == "TextEncodeAceStepAudio":
            node_enc_song = n
        elif ct == "ConditioningZeroOut":
            node_zero = n
        elif ct == "KSampler":
            node_ksampler_song = n

    if not (node_enc_song and node_ksampler_song):
        return graph

    enc_inp_song = node_enc_song.setdefault("inputs", {}) or {}
    tags_song = str(enc_inp_song.get("tags", "")).strip()
    clip_ref = enc_inp_song.get("clip")

    # intro_tags 외부 주입이 있으면 그것을 우선 사용
    if intro_tags:
        tags_pos = ", ".join(_dedup_compact_tags(list(intro_tags)))
    else:
        if intro_style == "ambient":
            extra = "instrumental intro, no vocals, soft pads, minimal drums, light percussion, build-up"
        else:
            extra = "instrumental intro, no vocals, same groove as main, same tempo, same key, light percussion, no kick until vocals, cohesive transition"
        tags_pos = (tags_song + ", " + extra).strip(", ")

    # 인트로 Positive 인코더
    enc_intro_pos_id = _new_id()
    graph[enc_intro_pos_id] = {
        "inputs": {
            "tags": tags_pos,
            "lyrics": "",
            "lyrics_strength": 0.0,
            "clip": clip_ref,
        },
        "class_type": "TextEncodeAceStepAudio",
        "_meta": {"title": "TextEncodeAceStepAudio (intro/pos)"},
    }

    # 인트로 Negative 인코더(보컬 억제)
    neg_tags = (
        "no vocals, no voice, no singing, no choir, no humming, "
        "instrumental only, remove vocalization, mute voice, zero vocals"
    )
    enc_intro_neg_id = _new_id()
    graph[enc_intro_neg_id] = {
        "inputs": {
            "tags": neg_tags,
            "lyrics": "",
            "lyrics_strength": 0.0,
            "clip": clip_ref,
        },
        "class_type": "TextEncodeAceStepAudio",
        "_meta": {"title": "TextEncodeAceStepAudio (intro/neg)"},
    }

    # 인트로 latent
    empty_intro_id = _new_id()
    graph[empty_intro_id] = {
        "inputs": {"seconds": int(max(1, intro_sec)), "batch_size": 1},
        "class_type": "EmptyAceStepLatentAudio",
        "_meta": {"title": "EmptyAceStepLatentAudio (intro)"},
    }

    # 본곡 KSampler 파라미터 복사
    ks_song_in = node_ksampler_song.get("inputs", {}) or {}
    ks_intro_id = _new_id()
    neg_ref = [enc_intro_neg_id, 0] if enc_intro_neg_id in graph else (["44", 0] if node_zero else None)
    graph[ks_intro_id] = {
        "inputs": {
            "seed": ks_song_in.get("seed", 123456789012345),
            "steps": ks_song_in.get("steps", 80),
            "cfg": ks_song_in.get("cfg", 5),
            "sampler_name": ks_song_in.get("sampler_name", "euler"),
            "scheduler": ks_song_in.get("scheduler", "simple"),
            "denoise": ks_song_in.get("denoise", 1),
            "model": ks_song_in.get("model", ["49", 0]),
            "positive": [enc_intro_pos_id, 0],
            "negative": neg_ref,
            "latent_image": [empty_intro_id, 0],
        },
        "class_type": "KSampler",
        "_meta": {"title": "KSampler (intro)"},
    }

    # VAE 참조
    vae_ref = ["40", 2]
    for _id, n in _nodes_iter(graph):
        if n.get("class_type") == "VAEDecodeAudio":
            vi = n.get("inputs") or {}
            if isinstance(vi.get("vae"), list):
                vae_ref = vi.get("vae")
            break

    # 디코드 + 저장
    decode_intro_id = _new_id()
    graph[decode_intro_id] = {
        "inputs": {"samples": [ks_intro_id, 0], "vae": vae_ref},
        "class_type": "VAEDecodeAudio",
        "_meta": {"title": "오디오 VAE 디코드 (intro)"},
    }
    save_intro_id = _new_id()
    graph[save_intro_id] = {
        "inputs": {
            "filename_prefix": f"{subfolder}/intro_part",
            "audioUI": "",
            "audio": [decode_intro_id, 0],
        },
        "class_type": "SaveAudio",
        "_meta": {"title": "오디오 저장 (intro)"},
    }

    return graph





def _enforce_seconds_with_intro(graph: dict, *, seconds_total: int, intro_sec: int) -> None:
    """
    seconds 입력을 가진 노드들에:
    - 기본: 전체 길이(seconds_total) 덮어쓰기
    - 인트로 EmptyLatent: intro_sec
    - 본곡 EmptyLatent: seconds_total - intro_sec
    """
    if not isinstance(graph, dict):
        return
    song_sec = max(1, int(seconds_total) - int(intro_sec))
    intro_sec = max(1, int(intro_sec))

    def _nodes_iter(g: dict):
        for k, v in g.items():
            if isinstance(v, dict) and "class_type" in v:
                yield str(k), v

    # 전체 덮어쓰기
    for _id, n in _nodes_iter(graph):
        ins = n.setdefault("inputs", {}) or {}
        if "seconds" in ins:
            try:
                ins["seconds"] = int(seconds_total)
            except Exception:
                pass

    # 인트로/본곡 개별 지정
    for _id, n in _nodes_iter(graph):
        ct = str(n.get("class_type") or "")
        ins = n.get("inputs") or {}
        title = str(n.get("_meta", {}).get("title", ""))
        if ct == "EmptyAceStepLatentAudio":
            if "intro" in title.lower():
                ins["seconds"] = int(intro_sec)
            else:
                ins["seconds"] = int(song_sec)


def _concat_intro_and_song_to_wav(
    proj_dir: "Path",
    *,
    ffmpeg_exe: str = "ffmpeg",
    intro_prefix: str = "intro_part",
    song_hint_prefix: str = "vocal_final_",
    out_name: str = "vocal.wav",
    crossfade_ms: int = 600,     # 이전 200 → 600ms로 기본 상향
    normalize: bool = True,      # RMS 라우드니스 맞춤
) -> "Path":
    """
    인트로/본곡 산출물을 찾아 순서대로 이어붙여 최종 WAV 저장.
    - normalize=True: 두 파트를 RMS 기준으로 맞춘 뒤 크로스페이드
    - crossfade_ms: 400~800ms 권장(이질감 최소화)
    """
    from pathlib import Path
    import itertools
    from subprocess import run, PIPE

    proj = Path(proj_dir)
    out = proj / out_name
    try:
        if out.exists():
            out.unlink()
    except OSError:
        pass

    exts = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac")
    intro_cands = [p for p in proj.glob(f"{intro_prefix}*") if p.suffix.lower() in exts]
    song_cands = [p for p in itertools.chain(proj.glob(f"{song_hint_prefix}*"), proj.glob("*"))
                  if p.suffix.lower() in exts and intro_prefix not in p.name]
    if not intro_cands or not song_cands:
        raise FileNotFoundError("intro/song audio candidates not found")

    intro = max(intro_cands, key=lambda x: getattr(x.stat(), "st_mtime", 0))
    song = max(song_cands, key=lambda x: getattr(x.stat(), "st_mtime", 0))

    from pydub import AudioSegment, effects  # type: ignore
    a = AudioSegment.from_file(str(intro))
    b = AudioSegment.from_file(str(song))

    # 라우드니스 정합(RMS normalize)
    if normalize:
        try:
            a = effects.normalize(a)
            b = effects.normalize(b)
        except Exception:
            pass

    if crossfade_ms and crossfade_ms > 0:
        c = a.append(b, crossfade=int(crossfade_ms))
    else:
        c = a + b

    c.export(str(out), format="wav")
    return out

def _dedup_compact_tags(tags: list[str]) -> list[str]:
    """공백/중복 정리 + 길이 가드(과도한 문장형 방지)."""
    out: list[str] = []
    seen: set[str] = set()
    for t in tags:
        s = str(t or "").strip()
        if not s:
            continue
        s = s[:64]  # 너무 긴 문구는 자르기
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
    # 과도하게 길어지는 거 방지: 6~14개 이내 권장
    if len(out) > 14:
        out = out[:14]
    return out


def suggest_instruments_for_lyrics(
    lyrics: str,
    *,
    bpm_hint: int | None = None,
    key_hint: str | None = None,
    max_instruments: int = 4,
) -> dict:
    """
    가사 텍스트를 간단 키워드로 분석해 장르/무드를 추정하고,
    내장된 악기 풀에서 인트로/본곡/아웃트로에 넣을 태그를 생성한다.
    외부 라이브러리 없이 동작(간단 규칙 기반).
    반환: {
      "genre": str,
      "mood": str,
      "song_tags": list[str],
      "intro_tags": list[str],
      "outro_tags": list[str],
    }
    """
    s = " " + (lyrics or "").lower() + " "
    def has(*words: str) -> bool:
        return any((" " + w.lower() + " ") in s for w in words)

    # 장르 추정
    if has("jazz", "브러시", "스윙", "칵테일", "라운지"):
        genre = "jazz"
    elif has("lofi", "로파이", "공부", "chill", "치유"):
        genre = "lofi hip hop"
    elif has("포크", "folk", "캠프", "어쿠스틱", "산책", "바람", "노을"):
        genre = "acoustic folk"
    elif has("오케스트라", "현악", "스트링", "영화", "서정", "드라마틱"):
        genre = "cinematic"
    elif has("edm", "클럽", "댄스", "일렉트로", "강한 비트"):
        genre = "edm"
    else:
        genre = "pop"

    # 무드 추정
    if has("사랑", "그리움", "로맨스", "연인", "달빛", "kiss", "romance"):
        mood = "romantic"
    elif has("슬픔", "눈물", "외로움", "그리워", "쓸쓸", "sad"):
        mood = "melancholic"
    elif has("추억", "레트로", "nostalgia", "그때", "돌아가"):
        mood = "nostalgic"
    elif has("희망", "빛", "설렘", "봄", "미소", "행복", "smile", "hope"):
        mood = "uplifting"
    elif has("밤", "비", "고요", "잔잔", "몽환", "dream", "calm", "ambient"):
        mood = "calm"
    else:
        mood = "neutral"

    # 장르별 악기 풀
    pool_by_genre = {
        "jazz": [
            "piano", "upright bass", "brush drums", "ride cymbal",
            "saxophone", "muted trumpet"
        ],
        "lofi hip hop": [
            "electric piano", "soft drums", "vinyl crackle", "sub bass",
            "lofi synth pad"
        ],
        "acoustic folk": [
            "acoustic guitar", "light percussion", "shaker", "upright bass",
            "subtle pad"
        ],
        "cinematic": [
            "string ensemble", "violin", "cello", "piano", "french horn",
            "subtle percussion"
        ],
        "edm": [
            "synth lead", "side-chained pad", "saw chords", "kick", "snare", "sub bass"
        ],
        "pop": [
            "piano", "electric bass", "soft electric drums", "acoustic guitar",
            "synth pad"
        ],
    }

    # 무드별 믹스/스타일 보정 태그
    mood_mix = {
        "romantic": ["warm", "soft reverb", "intimate"],
        "melancholic": ["dark", "long reverb", "intimate"],
        "nostalgic": ["tape saturation", "warm", "vintage"],
        "uplifting": ["bright", "wide", "energetic"],
        "calm": ["soft reverb", "gentle", "airy"],
        "neutral": ["balanced mixing"],
    }

    bpm_tag = f"{int(bpm_hint)} bpm" if isinstance(bpm_hint, int) and bpm_hint > 0 else None
    key_tag = (key_hint or "").strip() or None

    base = list(pool_by_genre.get(genre, pool_by_genre["pop"]))
    core = base[:max(1, max_instruments)]

    song_tags: list[str] = []
    song_tags += [genre]
    song_tags += core
    if bpm_tag:
        song_tags.append(bpm_tag)
    if key_tag:
        song_tags.append(key_tag)
    song_tags += mood_mix.get(mood, [])
    song_tags = _dedup_compact_tags(song_tags)

    # Intro: 동일 그루브/템포/키 유지 + 보컬 억제
    intro_tags = song_tags + [
        "instrumental intro", "instrumental only",
        "no vocals", "no singing", "no humming",
        "same groove as main", "same tempo", "same key",
        "light percussion", "no kick until vocals",
    ]
    intro_tags = _dedup_compact_tags(intro_tags)

    # Outro: 자연스러운 마무리
    outro_tags = song_tags + [
        "instrumental outro", "no vocals", "gentle fade out"
    ]
    outro_tags = _dedup_compact_tags(outro_tags)

    return {
        "genre": genre,
        "mood": mood,
        "song_tags": song_tags,
        "intro_tags": intro_tags,
        "outro_tags": outro_tags,
    }


def _dedup_compact_tags(tags: list[str]) -> list[str]:
    """공백/중복 정리 + 길이 가드(과도한 문장형 방지)."""
    out: list[str] = []
    seen: set[str] = set()
    for t in tags:
        s = str(t or "").strip()
        if not s:
            continue
        s = s[:64]  # 너무 긴 문구는 자르기
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
    # 과도하게 길어지는 거 방지: 6~14개 이내 권장
    if len(out) > 14:
        out = out[:14]
    return out



def _parse_bpm_from_tags(tags: list[str]) -> int | None:
    """태그에서 '### bpm' 숫자를 찾아 리턴. 없으면 None."""
    import re
    for t in tags:
        m = re.search(r"(\d{2,3})\s*bpm", str(t), flags=re.I)
        if m:
            try:
                v = int(m.group(1))
                if 40 <= v <= 220:
                    return v
            except ValueError:
                pass
    return None


def _tempo_hint_from_tags(tags: list[str]) -> str | None:
    """태그에서 tempo 힌트 추출: 'fast', 'slow', 'mid' 중 하나 또는 None."""
    s = " " + " ".join([str(t).lower() for t in tags]) + " "
    if " fast " in s or " uptempo " in s or " up-tempo " in s or " high energy " in s:
        return "fast"
    if " slow " in s or " ballad " in s or " downbeat " in s:
        return "slow"
    if " mid " in s or " mid-tempo " in s or " medium " in s:
        return "mid"
    return None


def _dedup_list_str(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for it in items:
        s = str(it or "").strip()
        if not s:
            continue
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def _bias_textencode_with_tags(
    graph: dict,
    *,
    base_tags: list[str],
    default_lyrics_strength: float = 0.7,
) -> None:
    """
    - TextEncodeAceStepAudio 노드에:
      1) lyrics_strength를 낮춰(기본 0.7) 태그 영향력 ↑
      2) BPM/Tempo 힌트를 명시적으로 병합 주입
    """
    if not isinstance(graph, dict):
        return

    # 태그에서 BPM/Tempo 추출
    bpm = _parse_bpm_from_tags(base_tags)
    tempo = _tempo_hint_from_tags(base_tags)

    merged = list(base_tags)
    if bpm:
        merged.append(f"{bpm} bpm")
    else:
        # tempo만 있고 bpm 없으면 대략값 보정
        if tempo == "fast":
            merged.append("140 bpm")
        elif tempo == "slow":
            merged.append("70 bpm")
        elif tempo == "mid":
            merged.append("110 bpm")

    # 템포 힌트도 명시적 키워드 추가
    if tempo == "fast":
        merged.append("fast tempo")
        merged.append("energetic")
    elif tempo == "slow":
        merged.append("slow tempo")
        merged.append("relaxed")
    elif tempo == "mid":
        merged.append("mid-tempo")
        merged.append("balanced")

    merged = _dedup_list_str(merged)

    # 그래프에 주입
    for _nid, node in list(graph.items()):
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type") or "") != "TextEncodeAceStepAudio":
            continue
        inp = node.setdefault("inputs", {}) or {}
        # tags 병합(기존이 문자열이면 컴마 스플릿)
        old = inp.get("tags")
        old_list: list[str] = []
        if isinstance(old, str):
            old_list = [x.strip() for x in old.split(",") if x.strip()]
        elif isinstance(old, list):
            old_list = [str(x).strip() for x in old if str(x).strip()]

        new_tags = _dedup_list_str(old_list + merged)
        inp["tags"] = ", ".join(new_tags)

        # lyrics_strength 낮추기(너무 높은 값은 태그 무시로 이어짐)
        try:
            ls = float(inp.get("lyrics_strength") if "lyrics_strength" in inp else default_lyrics_strength)
        except Exception:
            ls = default_lyrics_strength
        # 이미 매우 낮게 세팅되어 있으면 존중, 아니면 내려줌
        if ls > default_lyrics_strength:
            inp["lyrics_strength"] = float(default_lyrics_strength)




def _keep_single_saveaudio(graph: dict) -> None:
    """SaveAudio 계열 노드가 여러 개면 첫 번째만 남기고 나머지는 무력화."""
    def _iter_nodes(g):
        if isinstance(g, dict) and "nodes" in g and isinstance(g["nodes"], list):
            for node in g["nodes"]:
                if isinstance(node, dict):
                    yield node
        elif isinstance(g, dict):
            for node in g.values():
                if isinstance(node, dict):
                    yield node

    save_nodes = []
    for node in _iter_nodes(graph):
        class_type = str(node.get("class_type") or "")
        if class_type.startswith("SaveAudio"):
            save_nodes.append(node)

    if len(save_nodes) <= 1:
        return

    for node in save_nodes[1:]:
        node["class_type"] = "Note"
        node["inputs"] = {"text": "disabled by _keep_single_saveaudio"}


def _keep_single_text_encoder(graph: dict) -> None:
    """TextEncodeAceStepAudio가 여러 개면 첫 번째만 남기고 나머지는 무력화."""
    def _iter_nodes(g):
        if isinstance(g, dict) and "nodes" in g and isinstance(g["nodes"], list):
            for node in g["nodes"]:
                if isinstance(node, dict):
                    yield node
        elif isinstance(g, dict):
            for node in g.values():
                if isinstance(node, dict):
                    yield node

    enc_nodes = []
    for node in _iter_nodes(graph):
        if str(node.get("class_type") or "") == "TextEncodeAceStepAudio":
            enc_nodes.append(node)

    if len(enc_nodes) <= 1:
        return

    for node in enc_nodes[1:]:
        node["class_type"] = "Note"
        node["inputs"] = {"text": "disabled by _keep_single_text_encoder"}
