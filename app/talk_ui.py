# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Any
import os
import shutil
import time
import json
import subprocess

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QPixmap, QImage

# ───────────────── 설정 가져오기 ─────────────────
try:
    from app.settings import (
        COMFY_HOST,
        JSONS_DIR,
        COMFY_INPUT_DIR,
        COMFY_OUTPUT_DIR,
        COMFY_RESULT_ROOT,  # ← 공통으로 가져오도록
        FFMPEG_EXE,
        AUDIO_SAVE_FORMAT,
        COMFY_LOG_FILE,
    )

except Exception:
    from settings import (  # type: ignore
        COMFY_HOST,
        JSONS_DIR,
        COMFY_INPUT_DIR,
        COMFY_OUTPUT_DIR,
        COMFY_RESULT_ROOT,  # ← 이 줄 추가!!!
        FFMPEG_EXE,
        AUDIO_SAVE_FORMAT,
        COMFY_LOG_FILE,
    )



# ───────────────── 헬퍼들 ─────────────────
def _sanitize_title(text: str) -> str:
    text = (text or "").strip()
    if not text:
        text = "talk_object"
    bad = '<>:"/\\|?*'
    for ch in bad:
        text = text.replace(ch, "_")
    return text[:80]


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _copy_to_comfy_input(src: Path) -> str:
    """
    업로드한 이미지를 COMFY_INPUT_DIR 로 복사하고,
    LoadImage 노드에서 쓸 파일명(파일명만)을 리턴.
    """
    in_dir = Path(COMFY_INPUT_DIR)
    in_dir.mkdir(parents=True, exist_ok=True)

    stem = src.stem
    suffix = src.suffix or ".png"
    base = f"talk_{stem}{suffix}"
    dst = in_dir / base

    idx = 1
    while dst.exists():
        dst = in_dir / f"talk_{stem}_{idx}{suffix}"
        idx += 1

    shutil.copy2(src, dst)
    return dst.name


def _ffmpeg_extract_and_merge(
    video_path: Path,
    out_dir: Path,
    safe_title: str,
    on_progress,
    raw_audio: Optional[Path] = None,  # ← 이 인자 추가!
) -> tuple[Path, Path]:
    """
    - raw_audio(flac)가 있으면 → 그걸 변환해서 사용
    - 없으면 → 기존처럼 영상에서 오디오를 뽑으려고 시도
    - 어떤 경우에도 실패하면 원본 mp4를 final_talk.mp4로 복사해서 사용
    """
    audio_fmt = (AUDIO_SAVE_FORMAT or "mp3").lower()
    if audio_fmt not in ("wav", "mp3", "opus"):
        audio_fmt = "mp3"

    audio_path = out_dir / f"{safe_title}_audio.{audio_fmt}"
    final_path = out_dir / "final_talk.mp4"

    # ── 1) 오디오 소스 선택 ──
    extracted_ok = False

    if raw_audio is not None and raw_audio.exists():
        # 1-A) ComfyResult/audio 의 flac 를 변환
        on_progress(
            {
                "stage": "ffmpeg",
                "msg": f"원본 오디오 발견({raw_audio.name}) → {audio_fmt}로 변환 중…",
            }
        )
        cmd_audio = [FFMPEG_EXE, "-y", "-i", str(raw_audio)]

        if audio_fmt == "wav":
            cmd_audio += ["-acodec", "pcm_s16le", str(audio_path)]
        elif audio_fmt == "opus":
            cmd_audio += ["-acodec", "libopus", "-b:a", "128k", str(audio_path)]
        else:  # mp3
            cmd_audio += ["-acodec", "libmp3lame", "-b:a", "192k", str(audio_path)]

        try:
            proc = subprocess.run(
                cmd_audio,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',  # ★ 핵심: UTF-8로 강제 지정
                errors='replace'  # ★ 핵심: 그래도 깨지는 글자는 ?로 바꿔서 에러 방지
            )
            if proc.returncode == 0 and audio_path.exists():
                extracted_ok = True
            else:
                msg = proc.stderr or "(ffmpeg stderr 비어 있음)"
                on_progress(
                    {
                        "stage": "ffmpeg",
                        "msg": f"flac 변환 실패(code={proc.returncode}) → 영상에서 직접 추출 시도\n{msg}",
                    }
                )
        except Exception as e:
            on_progress(
                {
                    "stage": "ffmpeg",
                    "msg": f"flac 변환 명령 실행 실패 → 영상에서 직접 추출 시도\n{e}",
                }
            )

    if not extracted_ok:
        # 1-B) 영상에서 오디오 직접 추출
        on_progress(
            {
                "stage": "ffmpeg",
                "msg": f"영상에서 오디오 직접 추출 중… ({audio_fmt})",
            }
        )
        cmd_audio = [FFMPEG_EXE, "-y", "-i", str(video_path), "-vn"]

        if audio_fmt == "wav":
            cmd_audio += ["-acodec", "pcm_s16le", str(audio_path)]
        elif audio_fmt == "opus":
            cmd_audio += ["-acodec", "libopus", "-b:a", "128k", str(audio_path)]
        else:  # mp3
            cmd_audio += ["-acodec", "libmp3lame", "-b:a", "192k", str(audio_path)]

        try:
            proc2 = subprocess.run(
                cmd_audio,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc2.returncode == 0 and audio_path.exists():
                extracted_ok = True
            else:
                msg2 = proc2.stderr or "(ffmpeg stderr 비어 있음)"
                on_progress(
                    {
                        "stage": "ffmpeg",
                        "msg": f"영상에서 오디오 추출 실패(code={proc2.returncode})\n{msg2}",
                    }
                )
        except Exception as e:
            on_progress(
                {
                    "stage": "ffmpeg",
                    "msg": f"영상에서 오디오 추출 명령 실행 실패\n{e}",
                }
            )

    # ── 2) 최종 파일 생성 ──
    if extracted_ok and audio_path.exists():
        # 오디오가 준비된 경우 → mux
        on_progress(
            {
                "stage": "ffmpeg",
                "msg": "오디오+영상 병합(final_talk.mp4) 중…",
            }
        )
        cmd_mux = [
            FFMPEG_EXE,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            str(final_path),
        ]
        try:
            proc3 = subprocess.run(
                cmd_mux,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc3.returncode != 0:
                msg3 = proc3.stderr or "(ffmpeg stderr 비어 있음)"
                on_progress(
                    {
                        "stage": "ffmpeg",
                        "msg": f"병합 실패(code={proc3.returncode}) → 원본 mp4를 final_talk.mp4 로 복사\n{msg3}",
                    }
                )
                shutil.copy2(video_path, final_path)
            else:
                on_progress(
                    {
                        "stage": "ffmpeg",
                        "msg": "병합 완료 → final_talk.mp4 생성됨",
                    }
                )
        except Exception as e:
            on_progress(
                {
                    "stage": "ffmpeg",
                    "msg": f"병합 실행 중 예외 → 원본 mp4를 final_talk.mp4 로 복사\n{e}",
                }
            )
            shutil.copy2(video_path, final_path)
    else:
        # 오디오를 끝내 확보하지 못한 경우
        on_progress(
            {
                "stage": "ffmpeg",
                "msg": "오디오를 확보하지 못해, 원본 mp4를 final_talk.mp4 로 그대로 사용합니다.",
            }
        )
        shutil.copy2(video_path, final_path)
        audio_path = video_path  # 타입 맞추기용

    return audio_path, final_path




def _run_talk_job(
    *,
    comfy_host: str,
    workflow_path: Path,
    title: str,
    text: str,
    action_prompt: str,
    image_name: str,
    width: int,
    height: int,
    out_dir: Path,
    on_progress,
) -> dict:
    """
    1단계: who_voice.json으로 음성 생성 + now_frames/오디오 확보
    2단계: wanvideo_I2V_InfiniteTalk.json으로 영상 생성 (행동 프롬프트 + 오디오 + 프레임 수)
    3단계: ffmpeg로 오디오/영상 병합
    """
    import re
    import requests

    base_url = comfy_host.rstrip("/")
    url_prompt = f"{base_url}/prompt"

    safe_title = _sanitize_title(title)
    _ensure_dir(out_dir)

    # ---------------- 공통 헬퍼 ----------------
    def _build_audio_roots() -> list[Path]:
        roots: list[Path] = []
        roots.append(Path(COMFY_OUTPUT_DIR) / "audio")
        try:
            if COMFY_RESULT_ROOT:
                root = Path(COMFY_RESULT_ROOT)
                roots.append(root / "audio")
                roots.append(root.parent / "audio")
        except NameError:
            pass
        uniq: list[Path] = []
        seen: set[Path] = set()
        for r in roots:
            r = r.resolve()
            if r in seen:
                continue
            seen.add(r)
            uniq.append(r)
        return uniq

    def _collect_audio_files(roots: list[Path]) -> set[Path]:
        out: set[Path] = set()
        for root in roots:
            if not root.exists():
                continue
            for pattern in ("*.flac", "*.wav", "*.mp3", "*.opus"):
                for p in root.rglob(pattern):
                    out.add(p)
        return out

    def _build_video_roots() -> list[Path]:
        roots: list[Path] = [Path(COMFY_OUTPUT_DIR)]
        try:
            if COMFY_RESULT_ROOT:
                root = Path(COMFY_RESULT_ROOT)
                roots.append(root)
                roots.append(root.parent)
        except NameError:
            pass
        uniq: list[Path] = []
        seen: set[Path] = set()
        for r in roots:
            r = r.resolve()
            if r in seen:
                continue
            seen.add(r)
            uniq.append(r)
        return uniq

    def _collect_videos(roots: list[Path]) -> set[Path]:
        out: set[Path] = set()
        for root in roots:
            if not root.exists():
                continue
            for p in root.rglob("*.mp4"):
                if p.name.startswith("talk_"):
                    out.add(p)
        return out

    def _probe_audio_duration_sec(audio_path: Path) -> float:
        """ffmpeg 출력의 Duration: hh:mm:ss.xx 를 파싱해 길이를 구한다. (재시도 로직 추가)"""
        import time

        out = ""
        # 파일이 생성되자마자 읽으면 0바이트거나 잠겨있을 수 있음 -> 최대 3번 재시도
        for attempt in range(3):
            try:
                time.sleep(1.0)  # 1초 대기 (파일 쓰기 완료 대기)

                proc = subprocess.run(
                    [FFMPEG_EXE, "-i", str(audio_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    errors='replace'  # 한글 경로 등 인코딩 깨짐 방지
                )
                out = proc.stdout or ""

                # Duration: 00:00:04.40 패턴 찾기
                m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", out)
                if m:
                    h = int(m.group(1))
                    m_ = int(m.group(2))
                    s = float(m.group(3))
                    duration = h * 3600.0 + m_ * 60.0 + s
                    on_progress({
                        "stage": "who_voice",
                        "msg": f"오디오 길이 추정 성공: {duration:.2f}초",
                    })
                    return duration

            except Exception as e:
                on_progress({"stage": "who_voice", "msg": f"ffmpeg 실행 에러(시도 {attempt + 1}): {e}"})

        # 3번 다 실패했을 때, ffmpeg가 뱉은 에러 메시지를 로그에 보여줌 (원인 파악용)
        on_progress({
            "stage": "who_voice",
            "msg": f"오디오 길이 파악 실패. ffmpeg 출력:\n{out[:500]}",
        })
        return 0.0

    def _fetch_now_frames_from_history(prompt_id: str) -> int | None:
        """
        ComfyUI /history/{prompt_id}에서 PreviewAny(now_frames)의 값을 읽어온다.
        - 바깥 스코프의 이름을 가리지 않기 위해 node_title, frame_count 등의 이름을 사용한다.
        """
        if not prompt_id:
            return None

        url_hist = f"{base_url}/history/{prompt_id}"
        try:
            resp = requests.get(url_hist, timeout=30)
            resp.raise_for_status()
            hist = resp.json()
        except Exception as err:
            on_progress(
                {
                    "stage": "who_voice",
                    "msg": f"history 조회 실패: {err}",
                }
            )
            return None

        outputs = hist.get("outputs") or {}

        # 1) 노드 id 35(PreviewAny, title=now_frames)를 우선 시도
        candidates: list[tuple[str, dict]] = []
        for node_id, node_out in outputs.items():
            node_info = node_out.get("node") or {}
            meta = node_info.get("_meta") or {}
            node_title = meta.get("title", "")
            class_type = node_info.get("class_type", "")

            if node_id == "35":
                candidates.append((node_id, node_out))
                continue

            if node_title == "now_frames":
                candidates.append((node_id, node_out))
                continue

            if class_type == "PreviewAny" and node_title:
                candidates.append((node_id, node_out))

        for node_id, node_out in candidates:
            raw_data = node_out.get("data") or node_out.get("value") or []
            value_obj: Any | None = None

            if isinstance(raw_data, list) and raw_data:
                item = raw_data[0]
                if isinstance(item, dict) and "value" in item:
                    value_obj = item["value"]
                elif isinstance(item, (int, float, str)):
                    value_obj = item
            elif isinstance(raw_data, (int, float, str)):
                value_obj = raw_data

            if value_obj is None:
                continue

            try:
                frame_count = int(float(value_obj))
            except Exception:
                continue

            if frame_count <= 0:
                continue

            on_progress(
                {
                    "stage": "who_voice",
                    "msg": f"history(now_frames)에서 프레임 수={frame_count} 확인 (node {node_id})",
                }
            )
            return frame_count

        on_progress(
            {
                "stage": "who_voice",
                "msg": "history에서 now_frames 값을 찾지 못했습니다.",
            }
        )
        return None

    # ---------------- 1단계: who_voice (음성 생성) ----------------
    who_voice_path = Path(JSONS_DIR) / "who_voice.json"
    if not who_voice_path.exists():
        raise FileNotFoundError(f"who_voice 워크플로우를 찾을 수 없습니다: {who_voice_path}")

    on_progress(
        {
            "stage": "who_voice",
            "msg": f"who_voice 워크플로 로드 중… ({who_voice_path.name})",
        }
    )
    with who_voice_path.open("r", encoding="utf-8") as f:
        who_prompt = json.load(f)

    # ZonosGenerate(24) speech 입력값 주입
    import random  # (상단에 없다면 여기서 import)
    try:
        raw_text = text or ""

        # ★ 핵심 1: 사용자가 입력한 글자 앞에 "... "을 자동으로 붙여줌 (발음 씹힘 방지)
        modified_text = "... " + raw_text

        who_prompt["24"]["inputs"]["speech"] = modified_text

        # ★ 핵심 2: 매번 새로운 목소리를 만들기 위해 시드값 무작위 변경 (0.01초 종료 방지)
        who_prompt["24"]["inputs"]["seed"] = random.randint(1, 9999999999)

    except Exception:
        pass

    audio_roots = _build_audio_roots()
    before_audios = _collect_audio_files(audio_roots)

    on_progress(
        {
            "stage": "who_voice",
            "msg": f"/prompt 전송 중… ({url_prompt})",
        }
    )
    resp1 = requests.post(url_prompt, json={"prompt": who_prompt}, timeout=60)
    resp1.raise_for_status()
    data1 = resp1.json()
    prompt_id1 = data1.get("prompt_id") or data1.get("id") or ""

    if prompt_id1:
        on_progress(
            {
                "stage": "who_voice",
                "msg": f"prompt_id={prompt_id1} 오디오 생성 대기 시작",
            }
        )

    t0 = time.time()
    timeout_voice = 900.0
    poll_sec = 5.0
    last_log = 0.0
    raw_audio: Path | None = None

    while True:
        elapsed = time.time() - t0
        if elapsed > timeout_voice:
            raise TimeoutError("who_voice 오디오 생성 대기 시간 초과")

        after_audios = _collect_audio_files(audio_roots)
        new_audios = sorted(
            after_audios - before_audios,
            key=lambda p: p.stat().st_mtime,
        )
        if new_audios:
            raw_audio = new_audios[-1]
            on_progress(
                {
                    "stage": "who_voice",
                    "msg": f"오디오 파일 발견: {raw_audio.name}",
                }
            )
            break

        if elapsed - last_log >= 15.0:
            roots_str = ", ".join(str(r) for r in audio_roots)
            on_progress(
                {
                    "stage": "who_voice",
                    "msg": f"오디오 생성 대기 중… 경과 {int(elapsed)}초 / roots=[{roots_str}]",
                }
            )
            last_log = elapsed

        time.sleep(poll_sec)

    if raw_audio is None:
        raise RuntimeError("who_voice 단계에서 생성된 오디오를 찾지 못했습니다.")

    # now_frames 우선 시도 → 실패 시 ffmpeg 로 duration 계산
    now_frames = _fetch_now_frames_from_history(prompt_id1)
    duration_sec = 0.0
    if now_frames is None:
        duration_sec = _probe_audio_duration_sec(raw_audio)
        if duration_sec <= 0.0:
            raise RuntimeError("오디오 길이를 파악하지 못해 영상 생성을 진행할 수 없습니다.")

    # Comfy LoadAudio 에서 사용할 수 있도록 input 폴더로 복사
    audio_name_in_comfy = _copy_to_comfy_input(raw_audio)

    # ---------------- 2단계: InfiniteTalk I2V (영상 생성) ----------------
    on_progress(
        {
            "stage": "comfy",
            "msg": f"talk I2V 워크플로 로드 중… ({workflow_path.name})",
        }
    )
    with workflow_path.open("r", encoding="utf-8") as f:
        wf = json.load(f)

    # 해상도 주입 (INTConstant Width/Height: 245, 246)
    try:
        if "245" in wf and "inputs" in wf["245"]:
            wf["245"]["inputs"]["value"] = int(width)
        if "246" in wf and "inputs" in wf["246"]:
            wf["246"]["inputs"]["value"] = int(height)
    except Exception:
        pass

    # 이미지 파일명 주입 (LoadImage 284)
    try:
        wf["284"]["inputs"]["image"] = image_name
    except Exception:
        pass

    # 행동 프롬프트 → WanVideoTextEncodeCached(241)
    try:
        node241 = wf["241"]["inputs"]
        base_positive = node241.get("positive_prompt") or ""
        extra = (action_prompt or "").strip()
        if extra:
            if base_positive:
                node241["positive_prompt"] = f"{base_positive}, {extra}"
            else:
                node241["positive_prompt"] = extra
    except Exception:
        pass

    # LoadAudio(328)에 1단계 오디오 파일 연결
    try:
        node313 = wf["328"]["inputs"]  # <--- 328로 수정
        node313["audio"] = audio_name_in_comfy
        node313["audioUI"] = ""
    except Exception:
        pass

    # MultiTalkWav2VecEmbeds(194)가 LoadAudio(328)를 사용하도록 변경
    fps_val = 16.0
    try:
        node194 = wf["194"]["inputs"]
        node194["audio_1"] = ["328", 0]  # <--- 328로 수정
        try:
            fps_val = float(node194.get("fps", 16))
        except Exception:
            fps_val = 16.0
    except Exception:
        pass

    # Max frames(INTConstant 270)
    max_frames = 500
    try:
        max_frames = int(str(wf["270"]["inputs"].get("value", 500)))
    except Exception:
        max_frames = 500

    # 최종 사용할 프레임 수 결정
    if now_frames is not None:
        frames = int(now_frames)
        if frames < 1:
            frames = 1
        if max_frames > 0:
            frames = min(frames, max_frames)
        try:
            wf["270"]["inputs"]["value"] = frames
        except Exception:
            pass
        on_progress(
            {
                "stage": "comfy",
                "msg": f"now_frames 기반 프레임 수 설정: {frames}",
            }
        )
    else:
        # duration_sec 기반 계산 (위에서 이미 duration_sec > 0 확인됨)
        frames = int(duration_sec * fps_val)
        if frames < 1:
            frames = 1
        if max_frames > 0:
            frames = min(frames, max_frames)
        try:
            wf["270"]["inputs"]["value"] = frames
        except Exception:
            pass
        on_progress(
            {
                "stage": "comfy",
                "msg": f"오디오 길이 기반 프레임 수 설정: {frames} (fps={fps_val})",
            }
        )

    # VHS VideoCombine(131)에 prefix 설정
    try:
        node131 = wf["131"]["inputs"]
        node131["filename_prefix"] = f"talk_{safe_title}"
        if "save_output" in node131:
            node131["save_output"] = True
    except Exception:
        pass

    video_roots = _build_video_roots()
    before_videos = _collect_videos(video_roots)

    on_progress(
        {
            "stage": "comfy",
            "msg": f"/prompt 전송 중… ({url_prompt})",
        }
    )
    resp2 = requests.post(url_prompt, json={"prompt": wf}, timeout=60)
    resp2.raise_for_status()
    data2 = resp2.json()
    prompt_id2 = data2.get("prompt_id") or data2.get("id") or ""
    if prompt_id2:
        on_progress(
            {
                "stage": "comfy",
                "msg": f"prompt_id={prompt_id2} 렌더링 대기 시작",
            }
        )

    t1 = time.time()
    timeout_video = 3600.0
    last_log = 0.0
    found_video: Path | None = None

    while True:
        elapsed = time.time() - t1
        if elapsed > timeout_video:
            raise TimeoutError("Comfy 출력 mp4를 찾지 못했습니다.")

        after_videos = _collect_videos(video_roots)
        new_videos = sorted(
            after_videos - before_videos,
            key=lambda p: p.stat().st_mtime,
        )
        if new_videos:
            found_video = new_videos[-1]
            on_progress(
                {
                    "stage": "post",
                    "msg": f"영상 파일 발견: {found_video.name}",
                }
            )
            break

        if elapsed - last_log >= 15.0:
            roots_str = ", ".join(str(r) for r in video_roots)
            on_progress(
                {
                    "stage": "comfy",
                    "msg": f"렌더링 대기 중… 경과 {int(elapsed)}초 / roots=[{roots_str}]",
                }
            )
            last_log = elapsed

        time.sleep(poll_sec)

    if found_video is None:
        raise RuntimeError("Comfy 출력 mp4를 찾지 못했습니다.")

    # ---------------- 3단계: ffmpeg 병합 ----------------
    on_progress(
        {
            "stage": "ffmpeg",
            "msg": "오디오/영상 병합(ffmpeg) 시작",
        }
    )

    merged_audio, final_video = _ffmpeg_extract_and_merge(
        found_video,
        out_dir,
        safe_title,
        on_progress,
        raw_audio=raw_audio,
    )

    on_progress(
        {
            "stage": "done",
            "msg": f"완료: {found_video.name}, {merged_audio.name if merged_audio else '오디오 없음'}, {final_video.name}",
        }
    )

    return {
        "video": str(found_video),
        "audio": str(merged_audio) if merged_audio else "",
        "final": str(final_video),
    }







# ───────────────── TALK 탭 UI ─────────────────
class TalkWidget(QtWidgets.QWidget):
    """
    TALK 탭 UI
    - 제목 (기본: talk_object)
    - 프롬프트 (Zonos speech)
    - 이미지 업로드 + 드래그&드롭
    - 이미지 미리보기 (클릭 시 원본 팝업)
    - 이미지 원본 크기로 W/H 자동 세팅
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_name: Optional[str] = None
        self._image_src_path: Optional[Path] = None
        self.setAcceptDrops(True)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # 제목
        row_title = QtWidgets.QHBoxLayout()
        lbl_title = QtWidgets.QLabel("제목:", self)
        self.edit_title = QtWidgets.QLineEdit(self)
        self.edit_title.setPlaceholderText("talk_object")
        self.edit_title.setText("talk_object")
        row_title.addWidget(lbl_title)
        row_title.addWidget(self.edit_title)
        layout.addLayout(row_title)

        # 말 프롬프트 (Zonos speech)
        lbl_prompt = QtWidgets.QLabel("프롬프트 (Zonos speech):", self)
        self.edit_prompt = QtWidgets.QPlainTextEdit(self)
        self.edit_prompt.setPlaceholderText("여기에 말할 내용을 입력하세요…")
        self.edit_prompt.setMinimumHeight(120)
        layout.addWidget(lbl_prompt)
        layout.addWidget(self.edit_prompt)

        # 행동 프롬프트 (WanVideo 동작 / 제스처)
        lbl_action = QtWidgets.QLabel("행동 프롬프트 (동작/제스처):", self)
        self.edit_action_prompt = QtWidgets.QPlainTextEdit(self)
        self.edit_action_prompt.setPlaceholderText("예: 여성이 펜을 들고 활짝 웃으며 카메라를 향해 손을 흔든다…")
        self.edit_action_prompt.setMinimumHeight(80)
        layout.addWidget(lbl_action)
        layout.addWidget(self.edit_action_prompt)

        # 이미지 업로드
        row_img = QtWidgets.QHBoxLayout()
        lbl_img = QtWidgets.QLabel("이미지:", self)
        self.edit_img_path = QtWidgets.QLineEdit(self)
        self.edit_img_path.setReadOnly(True)
        btn_img = QtWidgets.QPushButton("이미지 업로드", self)
        btn_img.clicked.connect(self.on_click_browse_image)
        row_img.addWidget(lbl_img)
        row_img.addWidget(self.edit_img_path)
        row_img.addWidget(btn_img)
        layout.addLayout(row_img)

        # 사이즈
        row_size = QtWidgets.QHBoxLayout()
        lbl_w = QtWidgets.QLabel("W:", self)
        self.spin_w = QtWidgets.QSpinBox(self)
        self.spin_w.setRange(256, 1920)
        self.spin_w.setValue(405)
        lbl_h = QtWidgets.QLabel("H:", self)
        self.spin_h = QtWidgets.QSpinBox(self)
        self.spin_h.setRange(256, 1920)
        self.spin_h.setValue(720)
        row_size.addWidget(lbl_w)
        row_size.addWidget(self.spin_w)
        row_size.addSpacing(20)
        row_size.addWidget(lbl_h)
        row_size.addWidget(self.spin_h)
        row_size.addStretch(1)
        layout.addLayout(row_size)

        # 미리보기
        layout.addWidget(QtWidgets.QLabel("미리보기:", self))
        self.preview_label = QtWidgets.QLabel(self)
        self.preview_label.setFixedSize(250, 250)
        self.preview_label.setStyleSheet("border: 1px solid #888; background-color: #222;")
        self.preview_label.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self.preview_label.setText("이미지 없음")
        self.preview_label.installEventFilter(self)
        layout.addWidget(self.preview_label)

        # 원본 이미지 크기 표시 라벨
        self.label_img_info = QtWidgets.QLabel("원본 크기: - x -", self)
        font = self.label_img_info.font()
        font.setPointSize(max(font.pointSize() - 1, 8))
        self.label_img_info.setFont(font)
        self.label_img_info.setStyleSheet("color: #bbbbbb;")
        layout.addWidget(self.label_img_info)

        # 버튼
        row_btn = QtWidgets.QHBoxLayout()
        row_btn.addStretch(1)
        self.btn_talk = QtWidgets.QPushButton("talk 생성", self)
        self.btn_talk.clicked.connect(self.on_click_talk)
        row_btn.addWidget(self.btn_talk)
        layout.addLayout(row_btn)

        layout.addStretch(1)

    # ── 이미지 처리 공통 ──
    def _handle_image_selected(self, src: Path) -> None:
        if not src.exists():
            QtWidgets.QMessageBox.warning(self, "오류", "선택한 파일이 존재하지 않습니다.")
            return

        self._image_src_path = src

        # 미리보기 + W/H 자동 세팅
        img = QImage(str(src))
        if img.isNull():
            QtWidgets.QMessageBox.warning(self, "오류", "이미지를 불러오지 못했습니다.")
        else:
            pix = QPixmap.fromImage(img)
            scaled = pix.scaled(
                self.preview_label.width(),
                self.preview_label.height(),
                Qt.KeepAspectRatio,  # type: ignore[attr-defined]
                Qt.SmoothTransformation,  # type: ignore[attr-defined]
            )
            self.preview_label.setPixmap(scaled)
            self.preview_label.setText("")

            # W/H 값은 건드리지 않고, 원본 크기만 표시
            if hasattr(self, "label_img_info") and self.label_img_info is not None:
                self.label_img_info.setText(
                    f"원본 크기: {img.width()} x {img.height()}"
                )

        # Comfy input 복사
        try:
            name_in_comfy = _copy_to_comfy_input(src)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "오류", f"이미지 복사 실패:\n{e}")
            return

        self._image_name = name_in_comfy
        self.edit_img_path.setText(str(src))

    # 파일 선택
    def on_click_browse_image(self) -> None:
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "이미지 선택",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All Files (*)",
        )
        if not path_str:
            return
        self._handle_image_selected(Path(path_str))

    # talk 생성
    def on_click_talk(self) -> None:
        try:
            from app.utils import run_job_with_progress_async
        except Exception:
            from utils import run_job_with_progress_async  # type: ignore

        if not self._image_name:
            QtWidgets.QMessageBox.warning(self, "이미지 필요", "먼저 이미지를 선택하거나 드래그 앤 드롭해 주세요.")
            return

        title = (self.edit_title.text() or "").strip() or "talk_object"
        text = (self.edit_prompt.toPlainText() or "").strip()
        if not text:
            QtWidgets.QMessageBox.warning(self, "프롬프트 없음", "프롬프트(말할 내용)를 입력해 주세요.")
            return

        # 행동 프롬프트는 비어 있어도 OK
        action_prompt = (self.edit_action_prompt.toPlainText() or "").strip()

        w = int(self.spin_w.value())
        h = int(self.spin_h.value())

        safe_title = _sanitize_title(title)
        talk_root = Path(r"C:\my_games\shorts_make\maked_talk")
        out_dir = _ensure_dir(talk_root / safe_title)

        workflow_path = Path(JSONS_DIR) / "wanvideo_I2V_InfiniteTalk.json"

        def job(on_progress_cb):
            return _run_talk_job(
                comfy_host=COMFY_HOST,
                workflow_path=workflow_path,
                title=safe_title,
                text=text,
                action_prompt=action_prompt,  # ← 추가
                image_name=self._image_name or "",
                width=w,
                height=h,
                out_dir=out_dir,
                on_progress=on_progress_cb,
            )

        tail_file = COMFY_LOG_FILE if COMFY_LOG_FILE and os.path.exists(str(COMFY_LOG_FILE)) else None

        run_job_with_progress_async(
            self,
            title=f"Talk 생성: {safe_title}",
            job=job,
            tail_file=tail_file,
        )

    # 미리보기 클릭 → 팝업
    def eventFilter(self, obj, event: QEvent) -> bool:  # type: ignore[override]
        if obj is self.preview_label and event.type() == QEvent.MouseButtonPress:  # type: ignore[attr-defined]
            self._show_image_popup()
            return True
        return super().eventFilter(obj, event)

    def _show_image_popup(self) -> None:
        if not self._image_src_path or not self._image_src_path.exists():
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("이미지 미리보기 (원본)")
        dlg.resize(800, 800)

        vbox = QtWidgets.QVBoxLayout(dlg)
        scroll = QtWidgets.QScrollArea(dlg)
        scroll.setWidgetResizable(True)
        vbox.addWidget(scroll)

        lbl = QtWidgets.QLabel(scroll)
        pix = QPixmap(str(self._image_src_path))
        if pix.isNull():
            lbl.setText("이미지를 불러오지 못했습니다.")
        else:
            lbl.setPixmap(pix)
        lbl.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        scroll.setWidget(lbl)

        btn_close = QtWidgets.QPushButton("닫기", dlg)
        btn_close.clicked.connect(dlg.accept)
        vbox.addWidget(btn_close, alignment=Qt.AlignRight)  # type: ignore[attr-defined]

        dlg.exec_()

    # 드래그&드롭
    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                p = Path(url.toLocalFile())
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event) -> None:  # type: ignore[override]
        urls = event.mimeData().urls()
        if not urls:
            event.ignore()
            return
        for u in urls:
            p = Path(u.toLocalFile())
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                self._handle_image_selected(p)
                event.acceptProposedAction()
                return
        event.ignore()


def create_talk_widget(parent=None) -> TalkWidget:
    return TalkWidget(parent)
