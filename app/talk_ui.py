# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional
import os
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QPixmap, QImage
from app.utils import run_job_with_progress_async
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
import re
import requests
import time
import random
import json
import shutil
import subprocess
from pathlib import Path


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
        title: str,
        prompt1_text: str,
        prompt2_text: str,
        audio1_path: Optional[Path],
        audio2_path: Optional[Path],
        ref_audio1: Optional[Path],
        ref_audio2: Optional[Path],
        action_prompt: str,
        image_name: str,
        width: int,
        height: int,
        out_dir: Path,
        on_progress,
) -> dict:


    # ─────────────────────────────────────────────────────────────
    # [0] 기본 설정 및 경로 초기화
    # ─────────────────────────────────────────────────────────────
    base_url = comfy_host.rstrip("/")
    url_prompt = f"{base_url}/prompt"
    url_history = f"{base_url}/history"

    safe_title = _sanitize_title(title)
    _ensure_dir(out_dir)

    # 기본 참조 음성 경로
    default_ref1 = Path(r"C:\my_games\shorts_make\voice\꼬꼬 음성.m4a")
    default_ref2 = Path(r"C:\my_games\shorts_make\voice\남자성우1.mp3")

    # ─────────────────────────────────────────────────────────────
    # [A] 내부 헬퍼 함수 정의 (완전 복구)
    # ─────────────────────────────────────────────────────────────
    def _get_search_roots() -> list[Path]:
        """
        검색할 모든 후보 폴더를 수집합니다. (사용자 지정 경로 최우선)
        """
        roots = []
        # 1. 사용자가 명시한 결과 경로 (최우선)
        roots.append(Path(r"C:\comfyResult"))
        roots.append(Path(r"C:\comfyResult\audio"))

        # 2. settings.py에 설정된 경로들
        try:
            if COMFY_RESULT_ROOT:
                p = Path(COMFY_RESULT_ROOT)
                roots.append(p)
                roots.append(p.parent)
                roots.append(p / "audio")
        except NameError:
            pass

        # 3. ComfyUI 기본 output 경로
        roots.append(Path(COMFY_OUTPUT_DIR))
        roots.append(Path(COMFY_OUTPUT_DIR) / "audio")

        # 중복 제거 및 존재하는 폴더만 남김
        uniq = []
        seen = set()
        for r in roots:
            try:
                p = r.resolve()
                if p not in seen and p.exists():
                    seen.add(p)
                    uniq.append(p)
            except Exception:
                pass
        return uniq

    def _find_latest_file(extensions: tuple, time_threshold: float) -> Path | None:
        """
        지정된 시간(time_threshold) 이후에 생성된 파일 중 가장 최신 파일을 찾습니다.
        """
        roots = _get_search_roots()
        candidates = []
        for r in roots:
            for ext in extensions:
                for p in r.rglob(ext):
                    try:
                        if p.stat().st_mtime >= time_threshold:
                            candidates.append(p)
                    except Exception:
                        pass

        if not candidates:
            return None
        # 가장 최근 수정된 순서로 정렬
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return candidates[0]

    def _find_file_exact(target_name: str, subfolder: str = "") -> Path | None:
        """정확한 파일명을 가진 파일을 모든 루트에서 검색"""
        roots = _get_search_roots()

        # 1. subfolder 포함 검색
        if subfolder:
            rel = Path(subfolder) / target_name
            for r in roots:
                cand = r / rel
                if cand.exists(): return cand

        # 2. 파일명 직접 검색
        for r in roots:
            cand = r / target_name
            if cand.exists(): return cand

        # 3. 재귀 검색 (깊이 제한 없이)
        for r in roots:
            try:
                for f in r.rglob(target_name):
                    return f
            except Exception:
                pass
        return None

    def _copy_to_input(src: Path) -> str:
        """파일을 ComfyUI input 폴더로 복사하고 파일명 반환"""
        if not src.exists(): return ""
        dst_dir = Path(COMFY_INPUT_DIR)
        dst_dir.mkdir(parents=True, exist_ok=True)
        base = f"talk_{src.stem}{src.suffix}"
        dst = dst_dir / base
        idx = 1
        while dst.exists():
            dst = dst_dir / f"talk_{src.stem}_{idx}{src.suffix}"
            idx += 1
        shutil.copy2(src, dst)
        return dst.name

    def _probe_duration(p: Path) -> float:
        """ffmpeg로 오디오 길이 측정 (재시도 로직 포함)"""
        for _ in range(3):
            try:
                cmd = [FFMPEG_EXE, "-i", str(p)]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                      encoding="utf-8", errors="replace")
                m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stdout or "")
                if m:
                    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
            except Exception:
                time.sleep(0.5)
        return 0.0

    # ─────────────────────────────────────────────────────────────
    # [B] 오디오 준비 로직 (Zonos 생성 / 파일 업로드 처리)
    # ─────────────────────────────────────────────────────────────
    def _prepare_audio(label: str, path_obj: Optional[Path], txt: str, ref_obj: Optional[Path], def_ref: Path):
        """
        오디오 파일 경로 또는 텍스트를 받아 처리된 오디오(ComfyInput용 이름, 길이, 원본경로)를 반환
        """
        # 1. 사용자가 업로드한 파일이 있는 경우 (최우선)
        if path_obj and path_obj.exists():
            on_progress({"stage": "prep", "msg": f"[{label}] 업로드된 파일 사용: {path_obj.name}"})
            return _copy_to_input(path_obj), _probe_duration(path_obj), path_obj

        # 2. 텍스트가 있는 경우 Zonos 생성 시도
        if txt:
            who_json = Path(JSONS_DIR) / "who_voice.json"
            if not who_json.exists():
                raise FileNotFoundError("who_voice.json 워크플로우 파일이 없습니다.")

            with who_json.open("r", encoding="utf-8") as f_who:
                wp = json.load(f_who)

            # 텍스트 및 시드 주입
            try:
                wp["24"]["inputs"]["speech"] = "... " + txt
                wp["24"]["inputs"]["seed"] = random.randint(1, 2147483647)
            except Exception:
                pass

            # 참조 음성 설정
            final_ref = ref_obj if (ref_obj and ref_obj.exists()) else def_ref
            if final_ref.exists():
                try:
                    wp["12"]["inputs"]["audio"] = _copy_to_input(final_ref)
                except Exception:
                    pass

            on_progress({"stage": "zonos", "msg": f"[{label}] 음성 생성 요청 중..."})

            # 생성 요청 전 시간 기록 (파일 찾기 기준점)
            start_time = time.time()

            try:
                requests.post(url_prompt, json={"prompt": wp}, timeout=60)
            except Exception as zzzzzzzzze:
                raise RuntimeError(f"[{label}] ComfyUI 요청 실패: {zzzzzzzzze}")

            # 파일 감시 루프 (15분 대기)
            found_file = None
            wait_limit = 900

            while time.time() - start_time < wait_limit:
                time.sleep(2)
                # start_time 이후에 생성된 오디오 파일 찾기
                found_file = _find_latest_file(("*.flac", "*.wav", "*.mp3"), start_time)
                if found_file:
                    # 파일이 막 생성되어 0바이트일 수 있으므로 잠시 대기 및 확인
                    if found_file.stat().st_size > 0:
                        break
                    found_file = None

            if not found_file:
                raise TimeoutError(f"[{label}] 음성 생성 실패: 파일을 찾을 수 없습니다.")

            on_progress({"stage": "zonos", "msg": f"[{label}] 생성 완료: {found_file.name}"})
            return _copy_to_input(found_file), _probe_duration(found_file), found_file

        return "", 0.0, None

    # ─────────────────────────────────────────────────────────────
    # [C] 메인 실행 흐름 시작
    # ─────────────────────────────────────────────────────────────

    # 1. 음성 슬롯 1 준비
    file1, dur1, raw1 = _prepare_audio("음성1", audio1_path, prompt1_text, ref_audio1, default_ref1)

    # 2. 음성 슬롯 2 준비 (동시 실행 방지 대기)
    if prompt2_text and not audio2_path:
        time.sleep(2)
    file2, dur2, raw2 = _prepare_audio("음성2", audio2_path, prompt2_text, ref_audio2, default_ref2)

    # 3. 워크플로우 선택 및 로드
    use_two = bool(file2)
    json_name = "wanvideo_I2V_InfiniteTalk_two.json" if use_two else "wanvideo_I2V_InfiniteTalk_one.json"
    wf_path = Path(JSONS_DIR) / json_name

    if not wf_path.exists():
        raise FileNotFoundError(f"I2V 워크플로우 파일 없음: {json_name}")

    on_progress({"stage": "comfy", "msg": f"I2V 워크플로우 로드: {json_name}"})
    with wf_path.open("r", encoding="utf-8") as f_wf:
        wf = json.load(f_wf)

    # 4. 노드 설정 주입
    total_dur = dur1 + dur2
    try:
        # 텍스트 노드 찾기 (타입 기반 검색)
        text_node_id = None
        for nid, ninfo in wf.items():
            if ninfo.get("class_type") in ("WanVideoTextEncode", "WanVideoTextEncodeCached"):
                text_node_id = nid
                break

        # 행동 프롬프트 추가
        if action_prompt and text_node_id:
            p_node = wf[text_node_id]["inputs"]
            base_p = p_node.get("positive_prompt", "")
            p_node["positive_prompt"] = f"{base_p}, {action_prompt}"

        # 해상도 설정
        if "245" in wf: wf["245"]["inputs"]["value"] = int(width)
        if "246" in wf: wf["246"]["inputs"]["value"] = int(height)

        # 이미지 설정
        if "284" in wf: wf["284"]["inputs"]["image"] = image_name

        # 프레임 설정 (FPS 16)
        fps = 16
        frames = max(16, int(total_dur * fps))
        if "270" in wf: wf["270"]["inputs"]["value"] = frames

        # 파일명 프리픽스
        prefix = f"talk_{safe_title}"
        if "131" in wf: wf["131"]["inputs"]["filename_prefix"] = prefix

        # 오디오 연결
        if "313" in wf and file1: wf["313"]["inputs"]["audio"] = file1
        if use_two and "351" in wf and file2: wf["351"]["inputs"]["audio"] = file2

    except Exception as e:
        on_progress({"stage": "warn", "msg": f"노드 설정 중 경고: {e}"})

    # 5. 영상 생성 요청
    on_progress({"stage": "comfy", "msg": f"영상 생성 요청 중... (예상 {total_dur:.1f}초)"})
    start_time_vid = time.time()
    res_main = requests.post(url_prompt, json={"prompt": wf}, timeout=60)
    pid_main = res_main.json().get("prompt_id")

    on_progress({"stage": "comfy", "msg": f"렌더링 시작 (ID: {pid_main})..."})

    # 6. 결과 대기 (API 폴링 + 파일 감지 하이브리드)
    final_video_path = None
    final_audio_path = None

    # 1시간 대기
    while time.time() - start_time_vid < 3600:
        time.sleep(4)

        # (1) API로 상태 확인 및 파일명 추출 시도
        try:
            hres = requests.get(f"{url_history}/{pid_main}", timeout=10)
            if hres.status_code == 200:
                hdata = hres.json()
                if pid_main in hdata:
                    # 완료됨, 출력 정보 확인
                    outputs = hdata[pid_main].get("outputs", {})

                    # 영상 찾기
                    if "131" in outputs:
                        items = outputs["131"].get("images", []) + outputs["131"].get("gifs", [])
                        for it in items:
                            if it.get("filename", "").endswith(".mp4"):
                                final_video_path = _find_file_exact(it["filename"], it.get("subfolder", ""))

                    # 오디오 찾기
                    if "327" in outputs:
                        for it in outputs["327"].get("audio", []):
                            final_audio_path = _find_file_exact(it["filename"], it.get("subfolder", ""))
        except Exception:
            pass

        # (2) API가 늦거나 실패할 경우를 대비해 파일 시스템 직접 감시 (백업)
        if not final_video_path:
            latest_vid = _find_latest_file(("talk_*.mp4",), start_time_vid)
            if latest_vid:
                final_video_path = latest_vid

        # 영상 찾았으면 루프 종료
        if final_video_path:
            on_progress({"stage": "post", "msg": f"영상 발견: {final_video_path.name}"})
            break

    if not final_video_path:
        raise TimeoutError(f"작업은 끝났으나 영상을 찾을 수 없습니다. (검색 경로: {_get_search_roots()})")

    # ─────────────────────────────────────────────────────────────
    # [D] 최종 병합 (무음 영상 대비 오디오 병합)
    # ─────────────────────────────────────────────────────────────
    on_progress({"stage": "ffmpeg", "msg": "오디오/영상 최종 병합 준비..."})

    # 사용할 오디오 소스 결정
    # 1순위: ComfyUI가 뱉어낸 결과 오디오 (싱크가 가장 정확함)
    target_audio = final_audio_path

    # 2순위: 없다면 원본 소스들을 직접 병합하여 사용
    if not target_audio:
        on_progress({"stage": "warn", "msg": "결과 오디오를 못 찾음. 원본 소스를 병합하여 사용합니다."})
        try:
            if use_two and raw1 and raw2:
                # 2개 파일 병합
                merged_wav = out_dir / f"{safe_title}_merged_source.wav"
                cmd_concat = [
                    FFMPEG_EXE, "-y",
                    "-i", str(raw1),
                    "-i", str(raw2),
                    "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[outa]",
                    "-map", "[outa]",
                    str(merged_wav)
                ]
                subprocess.run(cmd_concat, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                if merged_wav.exists():
                    target_audio = merged_wav
            elif raw1:
                # 1개 파일
                target_audio = raw1
        except Exception as e:
            on_progress({"stage": "warn", "msg": f"오디오 병합 실패: {e}"})

    # 병합 실행
    merged_audio_final = out_dir / f"{safe_title}_audio.mp3"
    final_mp4_path = out_dir / "final_talk.mp4"

    # 1. 오디오 포맷 변환 (mp3)
    if target_audio and target_audio.exists():
        subprocess.run([FFMPEG_EXE, "-y", "-i", str(target_audio), "-acodec", "libmp3lame", "-b:a", "192k",
                        str(merged_audio_final)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    # 2. Muxing
    if merged_audio_final.exists():
        subprocess.run([
            FFMPEG_EXE, "-y",
            "-i", str(final_video_path),
            "-i", str(merged_audio_final),
            "-c:v", "copy", "-c:a", "copy",
            str(final_mp4_path)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        on_progress({"stage": "done", "msg": f"최종 완료: {final_mp4_path.name}"})
    else:
        # 오디오가 정말 없으면 영상만 복사
        shutil.copy2(final_video_path, final_mp4_path)
        on_progress({"stage": "done", "msg": f"오디오 없이 영상만 완료: {final_mp4_path.name}"})

    return {
        "video": str(final_video_path),
        "audio": str(merged_audio_final) if merged_audio_final.exists() else "",
        "final": str(final_mp4_path)
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

        # 프롬프트별 참조 음성
        self._ref_audio1_src_path: Optional[Path] = None  # 프롬프트1용
        self._ref_audio2_src_path: Optional[Path] = None  # 프롬프트2용

        # 이미지 옆 InfiniteTalk용 오디오(기존 audio_1, audio_2 역할)
        self._audio1_src_path: Optional[Path] = None
        self._audio2_src_path: Optional[Path] = None


        self.setAcceptDrops(True)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # ───────── 제목 ─────────
        row_title = QtWidgets.QHBoxLayout()
        lbl_title = QtWidgets.QLabel("제목:", self)
        self.edit_title = QtWidgets.QLineEdit(self)
        self.edit_title.setPlaceholderText("talk_object")
        self.edit_title.setText("talk_object")
        row_title.addWidget(lbl_title)
        row_title.addWidget(self.edit_title)
        layout.addLayout(row_title)

        # ───────── 프롬프트1 / 프롬프트2 (Zonos speech) ─────────
        prompts_layout = QtWidgets.QHBoxLayout()

        # 왼쪽: 프롬프트1
        col_p1 = QtWidgets.QVBoxLayout()
        lbl_p1 = QtWidgets.QLabel("프롬프트1 (Zonos speech):", self)
        self.edit_prompt1 = QtWidgets.QPlainTextEdit(self)
        self.edit_prompt1.setPlaceholderText("프롬프트1에 말할 내용을 입력하세요…")
        self.edit_prompt1.setMinimumHeight(120)
        col_p1.addWidget(lbl_p1)
        col_p1.addWidget(self.edit_prompt1)
        prompts_layout.addLayout(col_p1)

        # 오른쪽: 프롬프트2
        col_p2 = QtWidgets.QVBoxLayout()
        lbl_p2 = QtWidgets.QLabel("프롬프트2 (Zonos speech):", self)
        self.edit_prompt2 = QtWidgets.QPlainTextEdit(self)
        self.edit_prompt2.setPlaceholderText("프롬프트2에 말할 내용을 입력하세요… (선택)")
        self.edit_prompt2.setMinimumHeight(120)
        col_p2.addWidget(lbl_p2)
        col_p2.addWidget(self.edit_prompt2)
        prompts_layout.addLayout(col_p2)

        layout.addLayout(prompts_layout)

        # ───────── 프롬프트별 참조 음성 ─────────
        layout.addWidget(QtWidgets.QLabel("참조 음성 (프롬프트별):", self))

        ref_audio_layout = QtWidgets.QHBoxLayout()

        # 프롬프트1 참조 음성
        col_ra1 = QtWidgets.QVBoxLayout()
        row_ra1 = QtWidgets.QHBoxLayout()
        lbl_ra1 = QtWidgets.QLabel("프롬프트1 참조 음성:", self)
        self.edit_ref_audio1_path = QtWidgets.QLineEdit(self)
        self.edit_ref_audio1_path.setReadOnly(True)
        btn_ra1 = QtWidgets.QPushButton("오디오 업로드", self)
        btn_ra1.clicked.connect(self.on_click_browse_ref_audio1)
        row_ra1.addWidget(lbl_ra1)
        row_ra1.addWidget(self.edit_ref_audio1_path)
        row_ra1.addWidget(btn_ra1)
        col_ra1.addLayout(row_ra1)
        ref_audio_layout.addLayout(col_ra1)

        # 프롬프트2 참조 음성
        col_ra2 = QtWidgets.QVBoxLayout()
        row_ra2 = QtWidgets.QHBoxLayout()
        lbl_ra2 = QtWidgets.QLabel("프롬프트2 참조 음성:", self)
        self.edit_ref_audio2_path = QtWidgets.QLineEdit(self)
        self.edit_ref_audio2_path.setReadOnly(True)
        btn_ra2 = QtWidgets.QPushButton("오디오 업로드", self)
        btn_ra2.clicked.connect(self.on_click_browse_ref_audio2)
        row_ra2.addWidget(lbl_ra2)
        row_ra2.addWidget(self.edit_ref_audio2_path)
        row_ra2.addWidget(btn_ra2)
        col_ra2.addLayout(row_ra2)
        ref_audio_layout.addLayout(col_ra2)

        layout.addLayout(ref_audio_layout)

        # ───────── 행동 프롬프트 ─────────
        lbl_action = QtWidgets.QLabel("행동 프롬프트 (동작/제스처):", self)
        self.edit_action_prompt = QtWidgets.QPlainTextEdit(self)
        self.edit_action_prompt.setPlaceholderText(
            "예: 여성이 펜을 들고 활짝 웃으며 카메라를 향해 손을 흔든다…"
        )
        self.edit_action_prompt.setMinimumHeight(80)
        layout.addWidget(lbl_action)
        layout.addWidget(self.edit_action_prompt)

        # ───────── 아래쪽: 이미지 + (이미지 옆 오디오 2개) ─────────
        bottom_layout = QtWidgets.QHBoxLayout()
        left_col = QtWidgets.QVBoxLayout()  # 이미지 관련
        right_col = QtWidgets.QVBoxLayout()  # 이미지 옆 오디오 2개

        # ===== 왼쪽: 이미지 업로드 + 사이즈 + 미리보기 =====
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
        left_col.addLayout(row_img)

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
        left_col.addLayout(row_size)

        # 미리보기
        left_col.addWidget(QtWidgets.QLabel("미리보기:", self))
        self.preview_label = QtWidgets.QLabel(self)
        self.preview_label.setFixedSize(250, 250)
        self.preview_label.setStyleSheet(
            "border: 1px solid #888; background-color: #222;"
        )
        self.preview_label.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self.preview_label.setText("이미지 없음")
        self.preview_label.installEventFilter(self)
        left_col.addWidget(self.preview_label)

        # 원본 이미지 크기 표시 라벨
        self.label_img_info = QtWidgets.QLabel("원본 크기: - x -", self)
        font = self.label_img_info.font()
        font.setPointSize(max(font.pointSize() - 1, 8))
        self.label_img_info.setFont(font)
        self.label_img_info.setStyleSheet("color: #bbbbbb;")
        left_col.addWidget(self.label_img_info)

        left_col.addStretch(1)

        # ===== 오른쪽: InfiniteTalk용 오디오 2개 (이미지 옆) =====
        right_col.addWidget(QtWidgets.QLabel("InfiniteTalk 오디오:", self))

        # 오디오 1
        row_audio1 = QtWidgets.QHBoxLayout()
        lbl_audio1 = QtWidgets.QLabel("오디오 1:", self)
        self.edit_audio1_path = QtWidgets.QLineEdit(self)
        self.edit_audio1_path.setReadOnly(True)
        btn_audio1 = QtWidgets.QPushButton("오디오 1 업로드", self)
        btn_audio1.clicked.connect(self.on_click_browse_audio1)
        row_audio1.addWidget(lbl_audio1)
        row_audio1.addWidget(self.edit_audio1_path)
        row_audio1.addWidget(btn_audio1)
        right_col.addLayout(row_audio1)

        # 오디오 2
        row_audio2 = QtWidgets.QHBoxLayout()
        lbl_audio2 = QtWidgets.QLabel("오디오 2:", self)
        self.edit_audio2_path = QtWidgets.QLineEdit(self)
        self.edit_audio2_path.setReadOnly(True)
        btn_audio2 = QtWidgets.QPushButton("오디오 2 업로드", self)
        btn_audio2.clicked.connect(self.on_click_browse_audio2)
        row_audio2.addWidget(lbl_audio2)
        row_audio2.addWidget(self.edit_audio2_path)
        row_audio2.addWidget(btn_audio2)
        right_col.addLayout(row_audio2)

        # 안내 텍스트
        lbl_audio_hint = QtWidgets.QLabel(
            "지원 형식: wav / mp3 / flac / opus", self
        )
        font2 = lbl_audio_hint.font()
        font2.setPointSize(max(font2.pointSize() - 1, 8))
        lbl_audio_hint.setFont(font2)
        lbl_audio_hint.setStyleSheet("color: #bbbbbb;")
        right_col.addWidget(lbl_audio_hint)

        right_col.addStretch(1)

        # 좌우 합치기
        bottom_layout.addLayout(left_col, 1)
        bottom_layout.addSpacing(20)
        bottom_layout.addLayout(right_col, 1)
        layout.addLayout(bottom_layout)

        # ───────── 버튼 ─────────
        row_btn = QtWidgets.QHBoxLayout()
        row_btn.addStretch(1)
        self.btn_talk = QtWidgets.QPushButton("talk 생성", self)
        self.btn_talk.clicked.connect(self.on_click_talk)
        row_btn.addWidget(self.btn_talk)
        layout.addLayout(row_btn)

        layout.addStretch(1)

        # 기본 참조 음성 경로 세팅
        self._set_default_ref_audios()

        layout.addStretch(1)

    def _set_default_ref_audios(self) -> None:
        """
        프롬프트별 참조 음성 기본값 세팅.
        - 프롬프트1: C:\my_games\shorts_make\voice\꼬꼬 음성.m4a
        - 프롬프트2: C:\my_games\shorts_make\voice\남자성우1.mp3
        파일이 없으면 조용히 무시.
        """
        base = Path(r"C:\my_games\shorts_make\voice")

        # 프롬프트1 기본 참조 음성
        default1 = base / "꼬꼬 음성.m4a"
        if default1.exists():
            self._ref_audio1_src_path = default1
            if hasattr(self, "edit_ref_audio1_path"):
                self.edit_ref_audio1_path.setText(str(default1))

        # 프롬프트2 기본 참조 음성
        default2 = base / "남자성우1.mp3"
        if default2.exists():
            self._ref_audio2_src_path = default2
            if hasattr(self, "edit_ref_audio2_path"):
                self.edit_ref_audio2_path.setText(str(default2))


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

    # ── 프롬프트별 참조 음성 업로드 ──
    def on_click_browse_ref_audio1(self) -> None:
        """프롬프트1용 참조 음성 선택."""
        self._browse_ref_audio_internal(slot=1)

    def on_click_browse_ref_audio2(self) -> None:
        """프롬프트2용 참조 음성 선택."""
        self._browse_ref_audio_internal(slot=2)

    def _browse_ref_audio_internal(self, slot: int) -> None:
        """공통 참조 음성 선택 로직."""
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            f"프롬프트{slot} 참조 음성 선택",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.opus *.m4a);;All Files (*)",
        )
        if not path_str:
            return

        p = Path(path_str)
        if not p.exists():
            QtWidgets.QMessageBox.warning(self, "오류", "선택한 오디오 파일이 존재하지 않습니다.")
            return

        if slot == 1:
            self._ref_audio1_src_path = p
            if hasattr(self, "edit_ref_audio1_path"):
                self.edit_ref_audio1_path.setText(str(p))
        else:
            self._ref_audio2_src_path = p
            if hasattr(self, "edit_ref_audio2_path"):
                self.edit_ref_audio2_path.setText(str(p))

    # ── 이미지 옆 InfiniteTalk 오디오 업로드 ──
    def on_click_browse_audio1(self) -> None:
        """InfiniteTalk용 오디오 1 선택."""
        self._browse_audio_internal(slot=1)

    def on_click_browse_audio2(self) -> None:
        """InfiniteTalk용 오디오 2 선택."""
        self._browse_audio_internal(slot=2)

    def _browse_audio_internal(self, slot: int) -> None:
        """공통 InfiniteTalk 오디오 선택 로직."""
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            f"오디오 {slot} 파일 선택",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.opus *.m4a);;All Files (*)",
        )
        if not path_str:
            return

        p = Path(path_str)
        if not p.exists():
            QtWidgets.QMessageBox.warning(self, "오류", "선택한 오디오 파일이 존재하지 않습니다.")
            return

        if slot == 1:
            self._audio1_src_path = p
            if hasattr(self, "edit_audio1_path"):
                self.edit_audio1_path.setText(str(p))
        else:
            self._audio2_src_path = p
            if hasattr(self, "edit_audio2_path"):
                self.edit_audio2_path.setText(str(p))

    # talk 생성
    def on_click_talk(self) -> None:

        # 1. 이미지 필수 체크
        if not self._image_name:
            QtWidgets.QMessageBox.warning(
                self,
                "이미지 필요",
                "먼저 이미지를 선택하거나 드래그 앤 드롭해 주세요.",
            )
            return

        title = (self.edit_title.text() or "").strip() or "talk_object"

        # 입력값 확보
        p1 = (self.edit_prompt1.toPlainText() or "").strip()
        p2 = (self.edit_prompt2.toPlainText() or "").strip()
        action_prompt = (self.edit_action_prompt.toPlainText() or "").strip()

        # 오디오 파일 존재 여부 확인
        has_audio1 = (self._audio1_src_path is not None and self._audio1_src_path.exists())
        # has_audio2 = (self._audio2_src_path is not None and self._audio2_src_path.exists())

        # 2. 필수 조건 체크: "프롬프트1 또는 오디오1" 중 하나는 반드시 있어야 함
        if not p1 and not has_audio1:
            QtWidgets.QMessageBox.warning(
                self,
                "입력 부족",
                "최소한 '프롬프트1'에 내용을 입력하거나 '오디오 1' 파일을 업로드해야 합니다.",
            )
            return

        w = int(self.spin_w.value())
        h = int(self.spin_h.value())
        safe_title = _sanitize_title(title)
        talk_root = Path(r"C:\my_games\shorts_make\maked_talk")
        out_dir = _ensure_dir(talk_root / safe_title)

        def job(on_progress_cb):
            return _run_talk_job(
                comfy_host=COMFY_HOST,
                title=safe_title,
                prompt1_text=p1,
                prompt2_text=p2,
                audio1_path=self._audio1_src_path,
                audio2_path=self._audio2_src_path,
                ref_audio1=self._ref_audio1_src_path,
                ref_audio2=self._ref_audio2_src_path,
                action_prompt=action_prompt,
                image_name=self._image_name or "",
                width=w,
                height=h,
                out_dir=out_dir,
                on_progress=on_progress_cb,
            )

        tail_file = (
            COMFY_LOG_FILE
            if COMFY_LOG_FILE and os.path.exists(str(COMFY_LOG_FILE))
            else None
        )

        run_job_with_progress_async(
            self,
            title=f"Talk 생성: {safe_title}",
            job=job,
            tail_file=tail_file,
        )

    def _generate_tts_for_slot1(self) -> Optional[Path]:
        """
        P1 또는 (P1없고 P2만 있는 특별 케이스는 P2 기반)으로 audio_1 TTS 생성.
        참조음성: self._ref_audio1_src_path
        """
        text = (self.edit_prompt1.toPlainText() or "").strip()
        if not text:
            # P1이 없으면 특별 케이스로 P2 기반 TTS
            text = (self.edit_prompt2.toPlainText() or "").strip()
            if not text:
                return None

        ref_audio = self._ref_audio1_src_path
        if not ref_audio or not ref_audio.exists():
            return None

        # 실제 TTS 생성 로직 (임시 placeholder)
        out_file = Path(r"C:\my_games\shorts_make\maked_talk\tts_audio1.wav")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_bytes(b"FAKE-WAV-DATA-1")

        return out_file

    def _generate_tts_for_slot2(self) -> Optional[Path]:
        """
        P2 기반 audio_2 TTS 생성.
        참조음성: self._ref_audio2_src_path
        """
        text = (self.edit_prompt2.toPlainText() or "").strip()
        if not text:
            return None

        ref_audio = self._ref_audio2_src_path
        if not ref_audio or not ref_audio.exists():
            return None

        # 실제 TTS 로직 (임시 placeholder)
        out_file = Path(r"C:\my_games\shorts_make\maked_talk\tts_audio2.wav")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_bytes(b"FAKE-WAV-DATA-2")

        return out_file




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
