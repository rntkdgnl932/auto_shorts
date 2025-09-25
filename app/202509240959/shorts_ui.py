from __future__ import annotations
# ↓ 최상단, 어떤 import보다 먼저 들어가야 함!

# CPU 강제(선택)
# ★ 다른 import 전에 최상단에 추가 (가장 중요)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""    # GPU 완전 차단
os.environ["CT2_FORCE_CPU"] = "1"          # faster-whisper(CTranslate2) CPU
os.environ["CT2_USE_GPU"] = "0"
os.environ["ONNXRUNTIME_FORCE_CPU"] = "1"  # onnxruntime CPU

print("GPU_LOCK:", os.getenv("CUDA_VISIBLE_DEVICES"), os.getenv("CT2_FORCE_CPU"), os.getenv("ONNXRUNTIME_FORCE_CPU"))

"""
쇼츠 자동화 메인 UI
- 가사 생성(제목 자동/수동, 길이 선택)
- 프로젝트 저장/불러오기
- 음악 생성(ACE-Step) : QThread + 실시간 진행 + 로그 tail
- 음악 분석 → segments.json / scene.json
- 설정 탭: settings_local.json 오버라이드 저장/즉시 적용
"""

import subprocess
import json
from typing import Optional
import requests
import shutil
import sys
import os
import faulthandler
import traceback
import datetime
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QPlainTextEdit, QTextEdit
try:
    from app.image_movie_docs import normalize_to_v11  # type: ignore
except Exception:
    try:
        from image_movie_docs import normalize_to_v11  # type: ignore
    except Exception:
        def normalize_to_v11(story: dict) -> dict:  # type: ignore
            return story
# shorts_ui.py (상단 import 구역 추가)
try:
    from app.audio_sync import analyze_and_cut_project
except ImportError:
    from audio_sync import analyze_and_cut_project

# (이미 있다면 중복 추가 금지)
from app.audio_sync import build_story_json  # ← story/analyze 사용
# shorts_ui.py (상단 import 구역에 추가)
try:
    from app.image_movie_docs import apply_intro_outro_to_story_json
except ImportError:
    from image_movie_docs import apply_intro_outro_to_story_json
# shorts_ui.py (상단 import 구역에 추가)
try:
    from app.utils import save_story_overwrite_with_prompts
except ImportError:
    from utils import save_story_overwrite_with_prompts
import re
_CANON_RE = re.compile(r"[^a-z0-9]+")
try:
    from ai import AI
except ImportError:
    from app.ai import AI  # type: ignore

# ── 유연한 임포트 ──────────────────────────────────────────────────────────────
try:
    # 모듈 통째로도 가져오되, 하위 함수/상수는 직접 임포트(둘 다 지원)
    # noinspection PyPep8Naming
    from app import settings as S
    from app.settings import (
        BASE_DIR, COMFY_HOST, DEFAULT_CHUNK, DEFAULT_OVERLAP, DEFAULT_INPUT_FPS, DEFAULT_TARGET_FPS,
        JSONS_DIR, ACE_STEP_PROMPT_JSON, I2V_WORKFLOW, AUDIO_SAVE_FORMAT, FFMPEG_EXE, USE_HWACCEL, FINAL_OUT,
        DEFAULT_HOST_CANDIDATES, save_overrides
    )
    from app.utils import sanitize_title, audio_duration_sec, load_json, save_json
    from app.lyrics_gen import generate_title_lyrics_tags, create_project_files
    from app.video_build import build_shots_with_i2v, xfade_concat, recalc_overlap
    from app.music_gen import generate_music_with_acestep, rewrite_prompt_audio_format
    from app.tag_norm import normalize_tags_to_english
    from app.audio_sync import analyze_project
except ImportError:
    # noinspection PyPep8Naming
    import settings as S  # type: ignore
    from settings import (  # type: ignore
        BASE_DIR, COMFY_HOST, DEFAULT_CHUNK, DEFAULT_OVERLAP, DEFAULT_INPUT_FPS, DEFAULT_TARGET_FPS,
        JSONS_DIR, ACE_STEP_PROMPT_JSON, I2V_WORKFLOW, AUDIO_SAVE_FORMAT, FFMPEG_EXE, USE_HWACCEL, FINAL_OUT,
        DEFAULT_HOST_CANDIDATES, save_overrides
    )
    from utils import sanitize_title, audio_duration_sec, load_json, save_json            # type: ignore
    from lyrics_gen import generate_title_lyrics_tags, create_project_files               # type: ignore
    from video_build import build_shots_with_i2v, xfade_concat, recalc_overlap                             # type: ignore
    from music_gen import generate_music_with_acestep, rewrite_prompt_audio_format                         # type: ignore
    from tag_norm import normalize_tags_to_english                                                          # type: ignore
    from audio_sync import analyze_project                                                                  # type: ignore                                                   # type: ignore

# ==== CRASH LOGGER (붙여넣기) ====
# from pathlib import Path as _Path # Path 중복 import
try:
    import settings as _settings_for_log
except ImportError:
    from app import settings as _settings_for_log  # type: ignore

try:
    from app import settings
    from app.progress import run_job_with_progress_async
except ImportError:
    import settings  # type: ignore
    from progress import run_job_with_progress_async  # type: ignore
# ============================================================
# PROMPT PIPE (self-contained) — paste into shorts_ui.py
# - 외부 모듈/패키지 불필요 (표준 라이브러리만 사용)
# - 최종 저장 대상: story.json (네가 넘기는 경로)
# - female_* 캐릭터에는 'huge breasts' 자동 포함
# ============================================================
import re
from typing import List, Any
# 공용헬퍼
from pathlib import Path

# =====================================
from app.utils import sanitize_title, audio_duration_sec, save_story_overwrite_with_prompts

# ---------- 실시간 로그 ----------



# ---------- 최종 저장 한방 함수 ----------


# ---------- 사용 예시 (네 코드 흐름에서 이 한 줄만 호출) ----------
# ============================================================
















# _LOG_DIR = _Path(getattr(_settings_for_log, "BASE_DIR", ".")) / "_debug"
_LOG_DIR = Path(getattr(_settings_for_log, "BASE_DIR", ".")) / "_debug"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_CRASH_LOG = _LOG_DIR / "qt_crash.log"

faulthandler.enable()  # SIGSEGV 같은 C레벨 크래시도 덤프

def _write_crash(text: str):
    try:
        with open(_CRASH_LOG, "a", encoding="utf-8") as f:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] {text}\n")
    except Exception:
        pass

def _excepthook(exc_type, exc, tb):
    msg = "".join(traceback.format_exception(exc_type, exc, tb))
    print("[FATAL] Uncaught:", msg, flush=True)
    _write_crash("[PY EXC]\n" + msg)

sys.excepthook = _excepthook

# PyQt 메시지 캡처(경고/오류 포함)
try:
    from PyQt5 import QtCore
    def _qt_msg_handler(mode, ctx, msg):
        s = f"[QT] {mode} {ctx.file}:{ctx.line} {msg}"
        print(s, flush=True)
        _write_crash(s)
    QtCore.qInstallMessageHandler(_qt_msg_handler)
except Exception:
    pass
# ==== /CRASH LOGGER ====


AUDIO_EXTS = (".mp3", ".wav", ".opus", ".flac", ".m4a", ".aac", ".ogg", ".wma")

def _sanitize_title_for_path(title: str) -> str:
    t = (title or "").strip()
    t = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", t)

    t = re.sub(r"\s+", " ", t).strip()
    return t or "untitled"

def _resolve_audio_dir_from_template(template_path: str, title: str) -> Path:
    safe_title = _sanitize_title_for_path(title)
    p = str(template_path).replace("[title]", safe_title)
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path

# ───────────────────────── 공통 Worker (가사 생성/일반 함수 호출) ────────────────
def _analyze_project_compat(*, project_dir: str, audio_path: str,
                            ai=None, on_progress=None):
    """
    audio_sync.analyze_project 버전별 파라미터 이름을 자동 매핑해서 호출한다.
    - audio_path <-> audio_file
    - on_progress / progress_cb / progress / callback / on_update / cb
    - ai 파라미터가 없으면 자동으로 생략
    """
    import inspect
    from audio_sync import analyze_project as _analyze_project

    sig = inspect.signature(_analyze_project)
    params = sig.parameters

    kwargs = {}

    # project_dir
    if 'project_dir' in params:
        kwargs['project_dir'] = project_dir

    # 오디오 인자 이름 자동 매핑
    if 'audio_path' in params:
        kwargs['audio_path'] = audio_path
    elif 'audio_file' in params:
        kwargs['audio_file'] = audio_path
    else:
        # 혹시 이름이 다르면 마지막 수단: 그대로 두고(포기),
        # positional이 필요할 수 있으므로 아래에서 args로 넣을 수도 있음.
        pass

    # ai 지원 여부
    if 'ai' in params:
        kwargs['ai'] = ai

    # 진행 콜백 이름 자동 매핑
    if on_progress is not None:
        for name in ('on_progress', 'progress_cb', 'progress',
                     'callback', 'on_update', 'cb'):
            if name in params:
                kwargs[name] = on_progress
                break
        # 어느 것도 없으면 조용히 콜백 미전달(해당 버전은 콜백 미지원)

    # 남은 필수 positional-only 파라미터가 있으면 안 건드려도 대개 없음
    return _analyze_project(**kwargs)

class Worker(QtCore.QObject):
    done = QtCore.pyqtSignal(object, object)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.is_open_dialog = False
        self.is_generating = False


    @QtCore.pyqtSlot()
    def run(self):
        """
        스레드 시작 시 호출될 공개 메서드(public slot).
        내부의 _run 메서드를 실행합니다.
        """
        self._run()

    # @QtCore.pyqtSlot() # _run은 직접 connect되지 않으므로 slot 데코레이터가 필수 아님
    def _run(self):
        try:
            print("[UI] Worker 시작:", getattr(self._fn, "__name__", str(self._fn)), flush=True)
            res = self._fn(*self._args, **self._kwargs)
            print("[UI] Worker 완료", flush=True)
            self.done.emit(res, None)
        except Exception as e:
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            print("[UI] Worker 예외:", e, flush=True)
            print(tb, flush=True)
            self.done.emit(None, f"{e}\n{tb}")



# ───────────────────────── 음악 전용 Worker (QThread) ─────────────────────────
from pathlib import Path

# utils / music_gen 임포트 (프로젝트 구조에 따라 app. 접두사 유무 대응)
try:
    from app.utils import load_json, save_json
    from app.music_gen import generate_music_with_acestep
except Exception:
    from utils import load_json, save_json
    from music_gen import generate_music_with_acestep


class MusicWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(bool, str)

    def __init__(self, project_dir: str, project_json: str,
                 target_seconds: int, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self.project_dir = str(project_dir)
        self.project_json = str(project_json)
        self.target_seconds = int(max(1, target_seconds))

    def _emit(self, msg: str) -> None:
        try:
            self.progress.emit(str(msg))
        except Exception as exc:
            print(f"[MusicWorker] progress emit fail: {exc}", flush=True)

    def _handle_progress_dict(self, info: dict) -> None:
        """
        dict 형태의 진행 상태를 받아 UI에 표시할 str으로 변환 후 emit합니다.
        """
        if not isinstance(info, dict):
            return

        stage = info.get("stage", "running")
        msg = info.get("msg", "")
        elapsed = info.get("elapsed")
        outputs = info.get("outputs")

        parts = [f"Stage: {stage}"]
        if elapsed is not None:
            parts.append(f"Elapsed: {float(elapsed):.1f}s")
        if outputs is not None:
            parts.append(f"Outputs: {outputs}")
        if msg:
            parts.append(f"Details: {msg}")

        self._emit(" | ".join(parts))

    # 클래스 MusicWorker 안 run()의 저장 파트만 교체
    def run(self) -> None:
        try:
            meta = load_json(Path(self.project_json), {}) or {}
            meta["target_seconds"] = int(self.target_seconds)
            # ★ B안: time은 ‘항상 초’로 저장
            meta["time"] = int(self.target_seconds)
            save_json(Path(self.project_json), meta)

            self._emit(f"음악 생성 시작: {self.target_seconds}초")
            try:
                generate_music_with_acestep(
                    self.project_dir,
                    on_progress=self._handle_progress_dict,
                    target_seconds=self.target_seconds
                )
            except TypeError:
                generate_music_with_acestep(
                    self.project_dir,
                    on_progress=self._handle_progress_dict
                )

            self._emit("음악 생성 완료")
            self.finished.emit(True, "ok")
        except Exception as exc:
            error_msg = f"음악 생성 실패: {exc}\n{traceback.format_exc()}"
            self._emit(error_msg)
            self.finished.emit(False, str(exc))


# ───────────────────────── 진행창(로그 포함) ─────────────────────────
class ProgressLogDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("음악 생성 진행")
        self.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self.setMinimumWidth(460)
        self._is_completed = False

        self.lbl = QtWidgets.QLabel("ACE-Step 준비 중…")
        self.bar = QtWidgets.QProgressBar(); self.bar.setRange(0, 0)
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(180)
        self.btn = QtWidgets.QPushButton("숨기기")
        self.btn.clicked.connect(self._on_btn)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.lbl); lay.addWidget(self.bar); lay.addWidget(self.log)
        row = QtWidgets.QHBoxLayout(); row.addStretch(1); row.addWidget(self.btn); lay.addLayout(row)

    def _on_btn(self):
        if self._is_completed: self.accept()
        else: self.hide()

    def set_status(self, text: str):
        self.lbl.setText(text)

    def set_completed(self, text: str):
        self._is_completed = True
        self.lbl.setText(text)
        self.setWindowTitle("음악 생성 — 완료")
        self.bar.setRange(0, 1); self.bar.setValue(1)
        self.btn.setText("닫기")

    def append_log(self, text: str):
        if not text: return
        self.log.appendPlainText(text.rstrip("\n"))
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ProgressLogDialog 클래스 안에 아래 3개 메서드 추가
    def enter_determinate(self, total: int):
        """결정형(0..total) 진행바 모드로 전환"""
        self._is_completed = False
        self.bar.setRange(0, max(0, int(total)))
        self.bar.setValue(0)
        self.setWindowTitle("진행 중")
        self.lbl.setText(f"0 / {int(total)}")

    def step(self, msg: str | None = None):
        """1단계 진행 + 로그 한 줄(있으면)"""
        if msg:
            self.append_log(msg)
        v = min(self.bar.value() + 1, self.bar.maximum())
        self.bar.setValue(v)
        self.lbl.setText(f"{v} / {self.bar.maximum()}")

    def set_title(self, text: str):
        self.setWindowTitle(text)


# ──────────────────────────────── Main UI ─────────────────────────────────────
class MainWindow(QtWidgets.QMainWindow):



    def __init__(self):
        super().__init__()
        # ai
        self._ai = AI()

        # 중복 방지
        # --- reentry guard flags (추가) ---
        self._analysis_running = False  # 음악 분석 중복 방지
        self._story_build_running = False  # story.json 빌드 중복 방지
        self._docs_build_running = False # ← 문서(image/movie) 생성 중복 방지 (추가)

        self._th: Optional[QtCore.QThread] = None
        self._worker: Optional[Worker] = None

        # 상태
        self._music_completed = False
        self._tail_completed = False
        self._analysis_running = False
        self._analysis_thread: Optional[QtCore.QThread] = None

        # 로그 tail
        self._log_timer: Optional[QtCore.QTimer] = None
        self._log_fp = None
        self._log_tail_pos = 0
        self._log_path: Optional[Path] = None

        # 기타
        self._forced_project_dir: Path | None = None
        self._last_tags: List[str] = []  # 자동 태그(정규화된 상태)를 보관

        self._dlg: ProgressLogDialog | None = None
        self._music_thread: Optional[QtCore.QThread] = None
        self._music_worker: Optional[MusicWorker] = None

        self.setWindowTitle("쇼츠 자동화 — 허브")
        self.resize(1100, 800)
        self._signals_wired = False
        self._build_ui()
        self._wire()
        self.status = self.statusBar()
        self.status.showMessage(f"기본 폴더: {BASE_DIR}")

        # 시작 시 project.json의 time 값을 라디오에 반영
        self._apply_time_from_project_json()

        self._bind_length_radios()


        # ---- ensure alias: self.txt_prompt ----
        if not hasattr(self, "txt_prompt"):
            for cand in ("prompt_edit", "te_prompt", "input_prompt", "plainTextEdit_prompt", "textEdit_prompt"):
                w = getattr(self, cand, None)
                if isinstance(w, (QPlainTextEdit, QTextEdit)):
                    self.txt_prompt = w  # 별칭으로 고정
                    break
        # 마지막 안전장치: 이름을 모를 때 모든 텍스트 에디트 중에서 첫 번째를 사용
        if not hasattr(self, "txt_prompt"):
            for w in self.findChildren((QPlainTextEdit, QTextEdit)):
                self.txt_prompt = w
                break

        if isinstance(getattr(self, "_current_project_dir", None), (str, Path)):
            delattr(self, "_current_project_dir")

        self._start_analysis_watchdog()

        self._actions_bound = False  # ← 1회 바인딩 가드
        QtCore.QTimer.singleShot(0, self._bind_actions)

    def _bind_actions(self) -> None:
        if getattr(self, "_actions_bound", False):
            return
        self._actions_bound = True

        from PyQt5 import QtWidgets

        def _btn(owner, name: str):
            obj = getattr(owner, name, None)
            return obj if isinstance(obj, QtWidgets.QAbstractButton) else None

        # ui가 없을 수도 있으니 self와 self.ui 둘 다 탐색
        ui = getattr(self, "ui", None)
        # shorts_ui.py (MainWindow._bind_actions 안)
        # shorts_ui.py :: MainWindow._bind_actions 안 pairs 목록에 한 줄 추가
        pairs = [
            (_btn(ui, "btn_generate_lyrics") or _btn(self, "btn_gen"), self.on_generate_lyrics_with_log),
            (_btn(ui, "btn_generate_music") or _btn(self, "btn_music"), self.on_click_generate_music),
            (_btn(ui, "btn_test1_story") or _btn(self, "btn_test1_story"), self.on_click_test1_analyze),
            (_btn(ui, "btn_test2_1_img") or _btn(self, "btn_test2_1_img"),
             self.on_click_test2_1_generate_missing_images_with_log),
            (_btn(ui, "btn_analyze") or _btn(self, "btn_analyze"), self.on_click_analyze_music),  # ← 추가
        ]

        for btn, handler in pairs:
            if not btn or not callable(handler):
                continue
            try:
                btn.clicked.disconnect()
            except TypeError:
                pass
            btn.clicked.connect(handler)

    # ==== PATCH: shorts_ui.py :: on_generate_lyrics_with_log ====
    def on_generate_lyrics_with_log(self) -> None:
        from app.progress import run_job_with_progress_async
        try:
            from app.lyrics_gen import generate_title_lyrics_tags
        except Exception:
            from lyrics_gen import generate_title_lyrics_tags

        # 버튼 잠금(있을 때만)
        btn = getattr(self, "btn_gen", None) or getattr(getattr(self, "ui", None), "btn_generate_lyrics", None)
        if btn:
            btn.setEnabled(False)

        def job(progress):
            # ★ 최초 진입 로그(항상 한 줄 보장)
            progress({"msg": "[ui] 가사생성 작업 시작"})

            # 입력값 수집
            title_in = getattr(self, "le_title", None).text().strip() if getattr(self, "le_title", None) else ""
            kw = ""
            for nm in ("te_prompt", "txt_prompt", "prompt_edit"):
                w = getattr(self, nm, None) or getattr(getattr(self, "ui", None), nm, None)
                if w and hasattr(w, "toPlainText"):
                    kw = w.toPlainText().strip()
                    break
            secs = self._current_seconds() if hasattr(self, "_current_seconds") else 60

            # 수동 태그 후보(있으면)
            allowed = []
            getter = getattr(self, "_manual_option_set", None)
            if callable(getter):
                try:
                    vals = getter() if callable(getter) else getter
                    allowed = sorted(vals)

                except Exception as e:
                    progress({"msg": f"[ui] manual tags 불러오기 실패: {e!r}"})

            # AI 선택
            prefer = "gemini" if (getattr(self, "btn_ai_toggle", None) and self.btn_ai_toggle.isChecked()) else "openai"
            allow_fb = False if prefer == "gemini" else True

            def trace(ev: str, msg: str):
                # 모델/파이프 단계 실시간 노출
                head = ev.split(":", 1)[0]
                progress({"msg": f"[{head}] {ev} | {msg}"})

            progress({"msg": f"[ai] prefer={prefer}, secs={secs}"})
            data = generate_title_lyrics_tags(
                prompt=kw,
                duration_min=max(1, min(3, int(round(secs / 60)) or 1)),
                duration_sec=secs,
                title_in=title_in,
                allowed_tags=allowed,
                trace=trace,
                prefer=prefer,
                allow_fallback=allow_fb,
            )
            return {"data": data, "title": title_in, "prompt": kw}

        def done(ok: bool, payload, err):
            if btn:
                btn.setEnabled(True)
            if not ok:
                from PyQt5 import QtWidgets
                QtWidgets.QMessageBox.critical(self, "가사 생성 실패", str(err))
                return
            pack = payload or {}
            data = pack.get("data", {})
            # 결과 반영 (기존 함수 사용)
            if hasattr(self, "_apply_lyrics_result"):
                self._apply_lyrics_result(data, pack.get("title", ""), pack.get("prompt", ""))

        run_job_with_progress_async(self, "가사 생성", job, tail_file=None, on_done=done)

    # ==== PATCH END ====

    # --- 누락 이미지 생성: 비동기 + 진행창 로그 (no _guess_project_dir) ---
    def on_click_test2_1_generate_missing_images_with_log(self) -> None:
        """story.json을 찾아 누락된 씬 이미지를 ComfyUI로 생성한다(진행창/실시간 로그)."""
        from PyQt5 import QtWidgets
        from pathlib import Path
        from typing import Optional

        # ----- 0) 버튼 비활성화(있을 때만) -----
        btn = getattr(self, "btn_test2_1_img", None) or getattr(getattr(self, "ui", None), "btn_test2_1_img", None)
        if isinstance(btn, QtWidgets.QAbstractButton):
            btn.setEnabled(False)

        # ----- 1) story.json 경로 결정 -----

        def _pick_latest_story(base_path: Path, title_hint: Optional[str] = None) -> Optional[Path]:
            """base_path 하위(깊이 최대 2단계)에서 최신 story.json 하나를 고른다.
            title_hint가 있으면 그 경로를 우선 확인한다.
            """
            try:
                if title_hint:
                    cand_path = (base_path / title_hint / "story.json").resolve()
                    if cand_path.exists():
                        return cand_path
            except OSError:
                pass

            best_ts: Optional[float] = None
            best_path: Optional[Path] = None

            # 깊이 1단계
            try:
                for path_ in base_path.glob("*/story.json"):
                    try:
                        ts = path_.stat().st_mtime
                    except OSError:
                        continue
                    if best_ts is None or ts > best_ts:
                        best_ts, best_path = ts, path_

                # 깊이 2단계
                for path_ in base_path.glob("*/*/story.json"):
                    try:
                        ts = path_.stat().st_mtime
                    except OSError:
                        continue
                    if best_ts is None or ts > best_ts:
                        best_ts, best_path = ts, path_
            except OSError:
                pass

            return best_path.resolve() if best_path else None

        story_path: Optional[Path] = None

        # (A) UI 텍스트박스 우선: 파일/폴더 모두 허용
        sp = getattr(self, "txt_story_path", None)
        if sp and hasattr(sp, "text"):
            txt = (sp.text() or "").strip()
            if txt:
                p = Path(txt).resolve()
                if p.suffix.lower() == ".json":
                    if p.exists():
                        story_path = p
                else:
                    cand = (p / "story.json").resolve()
                    if cand.exists():
                        story_path = cand

        # (B) settings.BASE_DIR 하위 검색 (제목 힌트 사용)
        if story_path is None:
            try:
                from settings import BASE_DIR  # type: ignore
                base_dir = Path(BASE_DIR).resolve()
                title_hint = ""
                t = getattr(self, "le_title", None)
                if t and hasattr(t, "text"):
                    title_hint = (t.text() or "").strip()
                story_path = _pick_latest_story(base_dir, title_hint or None)
            except Exception:
                story_path = None

        # (C) 마지막 폴백: 현재 폴더
        if story_path is None:
            cand = (Path.cwd() / "story.json").resolve()
            story_path = cand if cand.exists() else None

        if story_path is None or not story_path.exists():
            QtWidgets.QMessageBox.critical(self, "누락 이미지 생성", "story.json을 찾을 수 없습니다.")
            if isinstance(btn, QtWidgets.QAbstractButton):
                btn.setEnabled(True)
            return

        # ----- 2) UI에서 W/H/스텝 읽기 -----
        def _get_combo_int(combo_name: str, default_val: int) -> int:
            combo = getattr(self, combo_name, None)
            try:
                v = combo.currentData()
                return int(v if v is not None else default_val)
            except Exception:
                return int(default_val)

        ui_w = _get_combo_int("cmb_img_w", 1080)
        ui_h = _get_combo_int("cmb_img_h", 1920)

        spn = getattr(self, "spn_t2i_steps", None)
        try:
            steps = int(spn.value()) if spn is not None else 24
        except Exception:
            steps = 28

        # ----- 3) 진행창 + 비동기 작업 실행 -----
        from progress import run_job_with_progress_async
        try:
            from video_build import build_missing_images_from_story  # 로컬
        except Exception:
            from app.video_build import build_missing_images_from_story  # type: ignore

        try:
            from settings import COMFY_LOG_FILE
        except Exception:
            COMFY_LOG_FILE = None  # 로그 테일이 없어도 진행창은 동작

        def job(on_progress):
            # video_build 쪽에서 prompt_img + prompt 결합, qwen swap 워크플로, img_file 저장 반영까지 수행
            created = build_missing_images_from_story(
                story_path,
                ui_width=ui_w,
                ui_height=ui_h,
                steps=steps,
                timeout_sec=300,
                poll_sec=1.5,
                workflow_path=None,  # 강제: nunchaku_qwen_image_swap.json 사용
                on_progress=on_progress,
            )
            return {"created": [str(p) for p in created]}

        def done(ok: bool, payload, err):
            # 버튼 복구
            if isinstance(btn, QtWidgets.QAbstractButton):
                btn.setEnabled(True)

            if not ok:
                QtWidgets.QMessageBox.critical(self, "누락 이미지 생성 실패", str(err))
                return

            created = (payload or {}).get("created") or []
            QtWidgets.QMessageBox.information(self, "누락 이미지 생성 완료", f"생성 {len(created)}개 완료")

        run_job_with_progress_async(self, "누락 이미지 생성", job, tail_file=COMFY_LOG_FILE, on_done=done)

    def on_click_generate_music(self) -> None:
        from PyQt5 import QtWidgets
        try:
            from app.progress import run_job_with_progress_async
        except Exception:
            from progress import run_job_with_progress_async  # type: ignore

        btn = getattr(self, "btn_music", None)
        if btn:
            btn.setEnabled(False)

        def job(progress):
            from pathlib import Path
            try:
                from app.utils import load_json, save_json
            except Exception:
                from utils import load_json, save_json  # type: ignore
            try:
                from app.music_gen import generate_music_with_acestep
            except Exception:
                from music_gen import generate_music_with_acestep  # type: ignore

            # ▶▶ 활성 프로젝트만 사용 (원래 흐름 그대로)
            project_dir = self._get_active_project_dir()
            if not project_dir:
                raise RuntimeError("프로젝트 폴더가 없습니다. 먼저 가사를 생성하거나 프로젝트를 불러오세요.")

            # 길이(초) 즉시 반영 (원래 동작)
            secs = int(self._current_seconds()) if hasattr(self, "_current_seconds") else 60
            pj = Path(project_dir) / "project.json"
            meta = load_json(pj, {}) or {}
            meta["time"] = secs
            meta["target_seconds"] = secs
            save_json(pj, meta)

            # ─────────────────────────────────────────────────────────
            # ⬇️ 단 하나의 추가 동작: 변환 ON이면 story.json의 lyrics만 '임시로' 변환본으로 교체
            # OFF이면 아무 것도 건드리지 않음. 작업 끝나면 원복.
            # ─────────────────────────────────────────────────────────
            story_path = Path(project_dir) / "story.json"
            story_data = load_json(story_path, {}) or {}

            original_lyrics = None
            applied_conversion = False

            is_convert_on = False
            btn_convert = getattr(self, "btn_convert_toggle", None)
            if isinstance(btn_convert, QtWidgets.QPushButton):
                is_convert_on = btn_convert.isChecked()

            def _extract_lyrics(d):
                if isinstance(d, dict):
                    if "lyrics" in d and isinstance(d["lyrics"], str):
                        return d["lyrics"]
                    if "story" in d and isinstance(d["story"], dict) and isinstance(d["story"].get("lyrics"), str):
                        return d["story"]["lyrics"]
                return None

            if is_convert_on:
                src_lyrics = _extract_lyrics(story_data)
                if not src_lyrics:
                    te = getattr(self, "te_lyrics", None)
                    if hasattr(te, "toPlainText"):
                        src_lyrics = te.toPlainText()

                def _convert_for_api(text: str) -> str:
                    # 1순위: 사용자에게 보이는 우측 변환 칸 내용(있다면 그대로 사용)
                    dst = getattr(self, "te_lyrics_converted", None)
                    if isinstance(dst, QtWidgets.QTextEdit):
                        txt = dst.toPlainText().strip()
                        if txt:
                            return txt
                    # 2순위: kroman으로 라인별 변환 + [ko] 접두 + 하이픈 제거
                    try:
                        import kroman  # type: ignore
                    except ImportError:
                        return text or ""
                    out_lines = []
                    for raw in (text or "").splitlines():
                        s = raw
                        st = s.strip()
                        # 섹션 헤더/빈 줄 유지
                        if (st.startswith("[") and st.endswith("]")) or not st:
                            out_lines.append(s)
                            continue
                        rom = kroman.parse(s).strip().replace("-", "")
                        out_lines.append("[ko]" + rom)
                    return "\n".join(out_lines)

                if src_lyrics:
                    converted = _convert_for_api(src_lyrics)
                    if converted and converted != src_lyrics:
                        original_lyrics = src_lyrics
                        story_data["lyrics"] = converted
                        save_json(story_path, story_data)
                        applied_conversion = True

            # 진행 로그 콜백 (원래 동작)
            def forward(info: dict) -> None:
                st = str(info.get("stage", "")).upper()
                extra = {k: v for k, v in info.items() if k != "stage"}
                progress({"msg": f"[{st}] {extra}"})

            try:
                out = generate_music_with_acestep(
                    project_dir,
                    on_progress=forward,
                    target_seconds=secs,
                )
                progress({"msg": f"[done] 음악 저장: {out}"})
                return out
            finally:
                # 변환 적용했으면 원본 가사로 복원
                if applied_conversion and original_lyrics is not None:
                    story_data["lyrics"] = original_lyrics
                    save_json(story_path, story_data)

        def on_done(ok: bool, _payload, err):
            if btn:
                btn.setEnabled(True)
            if not ok and err is not None:
                QtWidgets.QMessageBox.warning(self, "음악 생성 오류", str(err))

        run_job_with_progress_async(self, "음악 생성 (ACE-Step)", job, on_done=on_done)

    def on_click_analyze_music(self) -> None:
        """
        '음악분석' 버튼(Whisper 우선 싱크 + 폴백 배분):
        - project.json → story.json에서 가사 로드
        - (선택) 보컬 분리 후 Whisper 단어 단위 전사
        - 공식 가사 라인과 정렬 → 성공 라인은 Whisper 시간 사용
        - 실패 라인은 온셋-가중치 배분 폴백
        - 콘솔/다이얼로그 출력 + preview JSON 저장 (story.json 미수정)
        - ★ 추가: 줄별 정합 기반 '마지막 줄 end + outro_sec'로 컷 & 페이드아웃 수행 → vocal_cut.wav, lyrics_aligned.json 저장
        """
        from pathlib import Path
        from typing import Optional, List
        from PyQt5 import QtWidgets
        import json

        # 설정값(기존 유지)
        USE_VOCAL_SEPARATION = False
        MODEL_SIZE = "medium"
        MIN_LEN = 0.5
        END_BIAS = 2.5
        AVG_MIN_SEC_PER_UNIT = 2.0
        START_PREROLL = 0.30

        # utils
        try:
            from app.utils import load_json  # type: ignore
        except Exception:
            from utils import load_json  # type: ignore

        # audio_sync 통합 모듈
        AS = None
        try:
            import app.audio_sync as AS  # type: ignore
        except Exception:
            try:
                import audio_sync as AS  # type: ignore
            except Exception:
                AS = None

        if AS is None:
            raise RuntimeError("audio_sync 모듈을 불러오지 못했습니다.")

        # 필수 함수 확인(기존 체크 유지)
        reqs = ["get_audio_duration", "detect_onsets_seconds", "layout_time_by_weights",
                "prepare_pure_lyrics_lines", "sync_lyrics_with_whisper"]
        for r in reqs:
            if not callable(getattr(AS, r, None)):
                raise RuntimeError(f"audio_sync.{r} 가 없습니다. audio_sync.py에 함수를 추가하세요.")

        def job(log):
            def p(tag_or_msg: str, msg: Optional[str] = None) -> None:
                log(tag_or_msg if msg is None else f"[{tag_or_msg}] {msg}")

            btn = getattr(self, "btn_analyze_music", None) or getattr(getattr(self, "ui", None), "btn_analyze_music",
                                                                      None)
            if isinstance(btn, QtWidgets.QAbstractButton):
                btn.setEnabled(False)

            try:
                p("ui", "프로젝트/보컬 탐색")
                proj_dir = self._current_project_dir()
                if not proj_dir:
                    raise RuntimeError("프로젝트 폴더가 없습니다. 먼저 프로젝트를 생성/불러오세요.")

                vocal_path = self._find_latest_vocal()
                if not (vocal_path and Path(vocal_path).exists()):
                    raise RuntimeError("보컬 오디오(vocal.*)를 찾지 못했습니다.")
                p("음악분석", f"보컬 파일: {Path(vocal_path).name}")

                # 가사 로드(project → story)
                lyrics_text = ""
                pj = Path(proj_dir) / "project.json"
                if pj.exists():
                    any_proj = load_json(pj, {}) or {}
                    if isinstance(any_proj, dict):
                        lyrics_text = str(any_proj.get("lyrics") or "").strip()
                if not lyrics_text:
                    sj = Path(proj_dir) / "story.json"
                    if sj.exists():
                        any_story = load_json(sj, {}) or {}
                        if isinstance(any_story, dict):
                            lyrics_text = str(any_story.get("lyrics") or "").strip()
                if not lyrics_text:
                    raise RuntimeError("가사를 찾을 수 없습니다. project.json/story.json에 lyrics가 없습니다.")
                # 변환 ON이면 변환본으로 교체(OFF면 원본 유지)
                lyrics_text = self._maybe_convert_lyrics_for_api(lyrics_text)

                # Whisper 우선 싱크 (기존 동작)
                res = AS.sync_lyrics_with_whisper(
                    str(vocal_path),
                    lyrics_text,
                    model_size=MODEL_SIZE,
                    use_vocal_separation=USE_VOCAL_SEPARATION,
                    min_len=MIN_LEN,
                    end_bias_sec=END_BIAS,
                    avg_min_sec_per_unit=AVG_MIN_SEC_PER_UNIT,
                    start_preroll=START_PREROLL,
                )

                segments = res.get("segments", [])
                onsets = res.get("onsets", [])
                duration = float(res.get("duration_sec", 0.0))
                start_at = float(res.get("start_at", 0.0))

                # 라인 표시용(줄바꿈 기준)
                lines = AS.prepare_pure_lyrics_lines(lyrics_text, drop_section_tags=True)

                # JSON 미리보기 저장 (기존)
                preview_path = Path(proj_dir) / "music_analysis_preview.json"
                with open(preview_path, "w", encoding="utf-8") as f:
                    json.dump(res, f, ensure_ascii=False, indent=2)

                # ── 출력 조립 시작: out을 여기서 '먼저' 만든다 ──
                out: List[str] = []
                out.append("===== [가사 원문] (줄바꿈 기준 라인) =====")
                for i, u in enumerate(lines):
                    out.append(f"  #{i:02d}: {u}")

                out.append("\n===== [시간 매핑 결과] =====")
                for seg in segments:
                    a = float(seg["start"])
                    b = float(seg["end"])
                    t = str(seg["text"])
                    out.append(f"{a:6.2f} ~ {b:6.2f}  {t}")

                out.append(f"\n(저장) {preview_path}")

                # ★ 추가: 줄별 정합 + 컷 & 페이드아웃(자동)
                from pathlib import Path as _Path  # 로컬 별칭(경고 회피)
                from audio_sync import analyze_and_cut_project  # 안전하게 지역 import

                cut_res = analyze_and_cut_project(
                    project_dir=str(_Path(proj_dir)),
                    model_size="medium",
                    snap_window_sec=0.15,
                )

                cut = cut_res.get("cut") or {}
                out.append("\n===== [컷 & 페이드아웃 요약] =====")
                out.append(f"last_end: {float(cut_res.get('last_end') or 0.0):.2f}s")
                out.append(f"outro_sec: {int(cut.get('outro_sec') or 0)}s")
                out.append(f"output: {str(cut.get('out') or '')}")
                out.append(f"duration_out: {float(cut.get('duration_out') or 0.0):.2f}s")
                out.append(f"aligned(json): {str(cut_res.get('aligned_path') or '')}")

                # 줄별 정합 결과를 함께 표시
                aligned_rows = []
                aligned_path_s = str(cut_res.get("aligned_path") or "")
                try:
                    if aligned_path_s:
                        ap = _Path(aligned_path_s)
                        if ap.exists():
                            import json as _json
                            aligned_rows = _json.loads(ap.read_text(encoding="utf-8"))
                except Exception:
                    aligned_rows = []

                out.append("\n===== [줄별 정합 결과] =====")
                if aligned_rows:
                    for idx, row in enumerate(aligned_rows):
                        line = str(row.get("line") or "")
                        s = row.get("start")
                        e = row.get("end")
                        sc = row.get("score")
                        if isinstance(s, (int, float)) and isinstance(e, (int, float)):
                            out.append(
                                f"#{idx:02d}: {float(s):6.2f} ~ {float(e):6.2f}  (score={float(sc or 0.0):.2f})  {line}")
                        else:
                            out.append(f"#{idx:02d}: (미매칭)  (score={float(sc or 0.0):.2f})  {line}")
                else:
                    out.append("(줄별 정합 결과 없음)")

                p("분석", "완료")
                return {"text": "\n".join(out)}

            except Exception as e:
                p("error", f"{type(e).__name__}: {e}")
                raise
            finally:
                if isinstance(btn, QtWidgets.QAbstractButton):
                    btn.setEnabled(True)

        def done(ok: bool, payload, err):
            from PyQt5 import QtWidgets
            if not ok:
                QtWidgets.QMessageBox.critical(self, "음악분석 실패", str(err))
                return
            text = (payload or {}).get("text", "")
            print("\n[음악분석 결과]\n" + text, flush=True)

            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("음악분석 결과 — Whisper 정렬 + 폴백 배분 + 컷/페이드아웃")
            dlg.resize(900, 720)
            v = QtWidgets.QVBoxLayout(dlg)
            ed = QtWidgets.QPlainTextEdit()
            ed.setReadOnly(True)
            ed.setPlainText(text)
            v.addWidget(ed)
            row = QtWidgets.QHBoxLayout()
            v.addLayout(row)
            row.addStretch(1)
            btn_close = QtWidgets.QPushButton("닫기")
            row.addWidget(btn_close)
            btn_close.clicked.connect(dlg.accept)
            dlg.exec_()

        try:
            from app.progress import run_job_with_progress_async  # type: ignore
        except Exception:
            from progress import run_job_with_progress_async  # type: ignore

        run_job_with_progress_async(self, "음악분석", job, tail_file=None, on_done=done)

    # class MainWindow(...) 내부 아무 유틸 메서드 구역에 추가


    def _find_latest_vocal(self) -> Path | None:
        """
        현재 프로젝트/FINAL_OUT 기준에서 가장 최신 vocal.* 또는 vocal_final_*.* 파일 하나를 찾는다.
        """
        cand: list[Path] = []

        # 1) 현재 프로젝트 폴더
        proj = self._current_project_dir()
        if proj:
            proj_dir = Path(proj)
            for ext in (".mp3", ".wav", ".m4a", ".flac", ".opus"):
                cand += list(proj_dir.glob(f"vocal{ext}"))
            cand += list(proj_dir.glob("vocal_final_*.*"))

        # 2) FINAL_OUT\[title]
        try:
            title = (self.le_title.text() or "").strip()
        except Exception:
            title = ""
        if title:
            out_dir = self._final_out_for_title(title)
            for ext in (".mp3", ".wav", ".m4a", ".flac", ".opus"):
                cand += list(out_dir.glob(f"vocal{ext}"))
            cand += list(out_dir.glob("vocal_final_*.*"))

        # 중복 제거 + 존재하는 파일만
        uniq: list[Path] = []
        seen: set[str] = set()
        for cand_path in cand:  # ← 변수명 변경
            try:
                rp = str(cand_path.resolve())
                if cand_path.exists() and rp not in seen:
                    seen.add(rp)
                    uniq.append(cand_path)
            except Exception:
                continue

        if not uniq:
            return None

        # 이름 가림 방지: 내부 함수 파라미터는 p, 위 for 변수는 cand_path
        def _mtime_safe(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except Exception:
                return -1.0

        return max(uniq, key=_mtime_safe)



    # shorts_ui.py (class MainWindow 내부)
    def on_click_test1_analyze(self) -> None:
        from pathlib import Path
        from progress import run_job_with_progress_async
        from utils import load_json, save_json

        TEST_PHASE = 0  # 0=전체, 1=가사만, 2=가사+오디오 매칭

        btn = getattr(self, "btn_test1_story", None) or getattr(getattr(self, "ui", None), "btn_test1_story", None)
        if btn:
            btn.setEnabled(False)

        def job(log):
            # 래퍼: log는 1인자만 받으므로 여기서 문자열로 합칩니다.
            def p(tag_or_msg: str, msg: str | None = None) -> None:
                if msg is None:
                    log(str(tag_or_msg))
                else:
                    log(f"[{tag_or_msg}] {msg}")

            try:
                # 2) 프로젝트/보컬 파악
                p("ui", "프로젝트/보컬 탐색")
                proj_dir_str = self._current_project_dir()
                if not proj_dir_str:
                    raise RuntimeError("프로젝트 폴더를 선택/생성해 주세요.")
                proj_dir = Path(proj_dir_str)

                meta_path = proj_dir / "project.json"
                if not meta_path.exists():
                    raise FileNotFoundError(str(meta_path))

                meta = load_json(str(meta_path), default={}) or {}
                duration_hint = float(meta.get("duration") or 0.0) or None

                audio_from_meta = str(meta.get("audio") or "").strip()
                if audio_from_meta:
                    audio_file = (proj_dir / audio_from_meta) if not Path(audio_from_meta).is_absolute() else Path(
                        audio_from_meta)
                else:
                    found = self._find_latest_vocal()
                    if not found:
                        raise FileNotFoundError("vocal.* 파일을 찾지 못했습니다.")
                    audio_file = Path(found)

                if not audio_file.exists():
                    raise FileNotFoundError(f"오디오 없음: {audio_file}")

                old_story = load_json(str(proj_dir / "story.json"), default={}) or {}
                old_scenes = list(old_story.get("scenes") or [])

                # 5) 가사 의미단위 (AI만)
                from story_enrich import split_lyrics_into_semantic_units_ai
                lyrics = str(meta.get("lyrics") or "").strip()
                if not lyrics:
                    raise RuntimeError("project.json 내 lyrics가 비어 있습니다.")

                p("ai", "AI 의미단위 분할 시작")

                def _trace(ev: str, msg: str) -> None:
                    p(ev, msg)  # trace도 1인자 log 규약 준수

                units = split_lyrics_into_semantic_units_ai(
                    lyrics,
                    ai=getattr(self, "_ai", None),
                    lang="ko",
                    preset="flow_all",  # ★ 전체 흐름 그대로(반복 포함)
                    max_chars_per_unit=18  # 필요시 16~22 사이로 미세조정
                )

                p("ai", f"의미단위 {len(units)}개")

                if TEST_PHASE == 1:
                    save_json(str(proj_dir / "story_ready.json"), {
                        "title": meta.get("title"),
                        "lyrics": lyrics,
                        "units_ai": units
                    })
                    p("done", "TEST_PHASE=1 완료 (story_ready.json 저장)")
                    return

                # 4) 오디오 분석
                from audio_sync import analyze_project
                p("audio", "오디오 분석(analyze_project)")
                analysis = analyze_project(
                    project_dir=str(proj_dir),
                    audio_path=str(audio_file),
                    save=True,
                    use_audio_segmentation=True,
                    force_total_sec=duration_hint,
                    ai=None,
                )
                effective_dur = float(analysis.get("effective_duration") or analysis.get("duration") or 0.0)
                if effective_dur <= 0:
                    raise RuntimeError("유효 길이를 확인할 수 없습니다.")

                # 6) 시간배치/섹션/스켈레톤/자산머지
                from story_enrich import (
                    layout_time_by_weights,
                    build_lyrics_sections,
                    build_scene_skeleton_from_units_with_kinds,
                    merge_scenes_preserve_assets_strict,
                    postprocess_story_layout,
                    label_scenes_by_kinds,
                )
                p("build", "시간배치/섹션/스켈레톤 구성")

                boundaries = layout_time_by_weights(
                    units,
                    total_start=0.0,
                    total_end=effective_dur,
                    onsets=None,
                )
                sections = build_lyrics_sections(units, boundaries)

                skeleton = build_scene_skeleton_from_units_with_kinds(
                    units,
                    boundaries,
                    kinds=None,
                    old_scenes=old_scenes if isinstance(old_scenes, list) else [],
                )

                story_tmp = {
                    "title": meta.get("title"),
                    "lyrics": lyrics,
                    "duration": round(effective_dur, 3),
                    "sections": sections,
                    "scenes": skeleton,
                }
                story_tmp = label_scenes_by_kinds(story_tmp)
                story_tmp = postprocess_story_layout(story_tmp)

                merged_scenes = merge_scenes_preserve_assets_strict(
                    old_scenes=list(old_scenes if isinstance(old_scenes, list) else []),
                    new_skeleton=list(story_tmp.get("scenes") or []),
                )
                story_tmp["scenes"] = merged_scenes

                if TEST_PHASE == 2:
                    save_json(str(proj_dir / "story.json"), story_tmp)
                    p("done", "TEST_PHASE=2 완료 (story.json 부분 저장)")
                    return

                # 7) GPT v11 강화
                p("gpt", "apply_gpt_to_story_v11")
                from story_enrich import apply_gpt_to_story_v11, finalize_story_coherence, ensure_global_negative, \
                    normalize_prompts
                def _ask(system, user, **kw):
                    return self._ai.ask_smart(system, user, **kw)

                # apply_gpt_to_story_v11 trace도 1인자 log 래퍼 사용
                story_tmp = apply_gpt_to_story_v11(
                    story_tmp,
                    ask=_ask,
                    prefer=getattr(self._ai, "default_prefer", "openai"),
                    allow_fallback=(getattr(self._ai, "default_prefer", "openai") == "openai"),
                    trace=_trace,
                )

                # 8) 최종 정합/정규화
                story_tmp = finalize_story_coherence(story_tmp)
                story_tmp = ensure_global_negative(story_tmp)
                story_tmp = normalize_prompts(story_tmp)

                save_json(str(proj_dir / "story.json"), story_tmp)
                p("done", "완료: story.json 저장")

            except Exception as e:
                p("error", f"{type(e).__name__}: {e}")
                raise

        try:
            run_job_with_progress_async(self, "프로젝트분석", job)  # 진행창 문구
        finally:
            if btn:
                btn.setEnabled(True)

    # ────────────── 공용 UI 헬퍼 ──────────────
    # Busy guard helpers (MainWindow 메서드로 추가)



    # ====================== GPT PROMPT ADAPTER (paste as-is) ======================
    # 모델 설정 (원하면 바꾸세요)
    _GPT_MODEL = os.getenv("PROMPT_MODEL", "gpt-4o-mini")

    # 캐릭터 규칙: female_*는 반드시 'adult woman' + 'huge breasts' 포함


    # 시스템 프롬프트 (JSON만 반환하도록 강하게 제약)
    _GPT_SYSTEM = """You are a prompt rewriter for image/video generation.
    Strictly return a single JSON object with two fields:
    - "prompt_img": string
    - "prompt_movie": string

    Rules:
    1) Convert any Korean hints to clean, model-friendly English tags.
    2) Always include character traits for each referenced character.
       - For any character id that starts with "female", you MUST include "adult woman" and "huge breasts".
       - For any character id that starts with "male", include "adult man".
    3) Merge global themes, palette, and section mood into concise tags (English).
    4) Keep quality tags concise (e.g., "photorealistic, cinematic lighting, high detail, 8k, masterpiece").
    5) Append negative tags at the end as: "--neg <comma-separated negatives>".
       Ensure "nsfw" stays inside negatives.
    6) For movie prompt, reuse the image prompt and append a motion hint if provided, like:
       '..., motion: <the action>'
    7) Do NOT add extra keys. Do NOT wrap with Markdown. JSON object ONLY.
    """

    _GPT_USER_TEMPLATE = """TITLE: {title}
    SECTION: {section}
    LYRIC (this scene): {lyric}
    GLOBAL SUMMARY: {global_summary}
    THEMES: {themes}
    PALETTE: {palette}
    STYLE GUIDE: {style_guide}
    SECTION MOOD: {section_mood}
    NEGATIVE: {negative}

    CHARACTERS (id:index → traits):
    {char_lines}

    IMAGE PROMPT (raw, may be Korean): {prompt_img_in}
    MOVIE PROMPT (raw, may be Korean): {prompt_movie_in}
    EFFECTS: {effects}

    Output JSON schema:
    {{
      "prompt_img": "<string>",
      "prompt_movie": "<string>"
    }}
    """

    # Basic Vocal(자동 포함 세트)
    BASIC_VOCAL_TAGS = [
        "clean vocals", "natural articulation", "warm emotional tone",
        "studio reverb light", "clear diction", "breath control", "balanced mixing",
    ]

    # === GPT 호출 래퍼 (추가) ===
    def _gpt_call(self, system_prompt=None, user_prompt: str = "", json_mode: bool = False, timeout: int = 30):
        """
        self._ai 백엔드의 가용 메서드(complete/chat/ask/__call__/scene_prompt)를 순차 시도.
        - system_prompt/user_prompt/json_mode/timeout을 최대한 전달
        - dict를 기대하는 경우(json_mode=True)면 dict만 반환
        - 실패 시 예외를 올려 상위에서 템플릿 폴백을 쓰도록 함
        """
        ai = getattr(self, "_ai", None)
        if ai is None:
            raise RuntimeError("AI 백엔드(self._ai)가 초기화되지 않았습니다.")

        # 선호 순서대로 호출 가능 함수 탐색
        for name in ("complete", "chat", "ask", "__call__", "scene_prompt", "scene_prompt_kor"):
            fn = getattr(ai, name, None)
            if not callable(fn):
                continue
            try:
                # 가장 풍부한 시그니처 시도
                return fn(system_prompt=system_prompt, user_prompt=user_prompt, json_mode=json_mode, timeout=timeout)
            except TypeError:
                try:
                    # 보통 형태
                    return fn(user_prompt or "")
                except TypeError:
                    # 최후방: 메시지 하나만
                    return fn(user_prompt or "")

        raise RuntimeError("사용 가능한 GPT 호출 메서드(complete/chat/ask/__call__/scene_prompt)가 없습니다.")


    def on_clear_inputs(self) -> None:
        """
        제목/가사/프롬프트 입력을 모두 비운다.
        - 없는 위젯 이름은 그냥 건너뛰도록 방어.
        """

        def _clr(name: str):
            w = getattr(self, name, None)
            if w is not None and hasattr(w, "clear"):
                w.clear()

        # 제목 / 가사 / 프롬프트 후보들
        _clr("le_title")  # QLineEdit(제목)
        _clr("te_lyrics")  # QTextEdit(가사)
        _clr("te_prompt")  # 통합 프롬프트 필드가 있다면
        _clr("te_prompt_img")  # 이미지 프롬프트 필드가 분리돼 있다면
        _clr("te_prompt_movie")  # 무비 프롬프트 필드가 분리돼 있다면

        # 상태바 안내(있을 때만)
        if hasattr(self, "status"):
            self.status.showMessage("제목/가사/프롬프트를 초기화했습니다.", 4000)

    def _add_clear_button_next_to_generate(self, parent_layout) -> None:
        """
        '가사 생성' 버튼 옆에 '초기화' 버튼을 추가한다.
        - parent_layout: self.btn_gen을 addWidget 한 바로 그 레이아웃을 넘겨줄 것.
        """
        from PyQt5 import QtWidgets  # or PySide2/PySide6, 프로젝트에 맞춰 이미 쓰는 모듈 그대로
        self.btn_clear_inputs = QtWidgets.QPushButton("초기화")
        self.btn_clear_inputs.setObjectName("btn_clear_inputs")
        self.btn_clear_inputs.setToolTip("제목/가사/프롬프트를 모두 비웁니다 (Ctrl+K)")
        try:
            self.btn_clear_inputs.setShortcut("Ctrl+K")
        except Exception:
            pass
        self.btn_clear_inputs.clicked.connect(self.on_clear_inputs)
        parent_layout.addWidget(self.btn_clear_inputs)

    def _task_busy(self, name: str) -> bool:
        """
        name 작업이 실제로 돌고 있는지 확인.
        - _{name}_thread 가 살아 있으면 True
        - 스레드가 끝났거나 없는데 플래그만 True면 스테일 → 정리 후 False
        """
        th = getattr(self, f"_{name}_thread", None)
        flag = bool(getattr(self, f"_{name}_running", False))

        if isinstance(th, QtCore.QThread):
            if th.isRunning():
                return True
            # 끝났으면 상태 정리
            setattr(self, f"_{name}_thread", None)
            setattr(self, f"_{name}_running", False)
            return False

        if flag:
            # 스레드 없는데 플래그만 True → 스테일 플래그 정리
            setattr(self, f"_{name}_running", False)
        return False

    # --- analyze_project 호환 래퍼 ---------------------------------------------


    # --------------------------------------------------------------------------

    def _try_begin(self, name: str, title: str = "안내",
                   msg: str = "작업이 진행 중입니다. 완료 후 다시 시도하세요.") -> bool:
        """
        작업 시작 가드: 이미 돌고 있으면 안내 후 False, 아니면 플래그 세팅 후 True.
        ※ 시작한 쪽에서 반드시 _task_done(name) 호출.
        """
        # 의도적 사용으로 린트 경고 방지(다른 훅에서 활용될 수 있음)
        _unused = (title, msg)

        if self._task_busy(name):
            # 가능하면 사용자 메시지 전달, 시그니처가 다르면 name만 전달
            try:
                self._guard_alert(name, title=title, msg=msg)  # type: ignore[call-arg]
            except TypeError:
                self._guard_alert(name)
            return False
        setattr(self, f"_{name}_running", True)
        return True

    def _task_done(self, name: str):
        """작업 종료 시 상태 정리(플래그/스레드 핸들)."""
        setattr(self, f"_{name}_running", False)
        setattr(self, f"_{name}_thread", None)

    # MainWindow 안 기존 _set_busy_ui 교체
    def _set_busy_ui(self, name: str, busy: bool):
        # 분석 중에도 테스트 버튼은 항상 활성 유지
        if name == "analysis":
            btn = getattr(self, "btn_analyze", None)  # 수동 분석 버튼만 잠그고 싶으면 이 줄 유지
            if btn is not None:
                btn.setEnabled(not busy)
        # 음악 가드 등 다른 용도는 그대로 두고 싶으면 여기서 분기 추가

    @staticmethod
    def _default_hair_map() -> dict:
        return {
            "female_01": "긴 웨이브 머리, 밝은 갈색, 앞머리 가볍게, 일관성 유지",
            "male_01": "블랙 투블럭, 풍성한 머리, 살짝 볼륨, 일관성 유지",
        }

    def _start_build_image_movie_docs(self, proj: Path, hair_map: dict | None = None) -> None:
        if self._docs_build_running:
            QtWidgets.QMessageBox.information(self, "안내", "image.json / movie.json 생성이 이미 진행 중입니다.")
            return
        self._docs_build_running = True

        hm = hair_map or self._default_hair_map()

        jsons_dir_val = getattr(S, "JSONS_DIR", None)
        wfdir = Path(jsons_dir_val) if jsons_dir_val else (Path(__file__).parent / "jsons")

        self._dbg(f"[DOCS] start build | proj={proj} | wfdir={wfdir} exists={wfdir.exists()} | hair_map={hm}")

        def job():
            import inspect

            # 빌더 + 타임라인 적용 함수 임포트 (ImportError만)
            try:
                from app.image_movie_docs import (
                    build_image_json, build_movie_json,
                    apply_intro_outro_to_story_json,
                    apply_intro_outro_to_image_json,
                    apply_intro_outro_to_movie_json,
                )
            except ImportError:
                from image_movie_docs import (
                    build_image_json, build_movie_json,
                    apply_intro_outro_to_story_json,
                    apply_intro_outro_to_image_json,
                    apply_intro_outro_to_movie_json,
                )

            def _call_compatible(func, **kwargs):
                """대상 함수 시그니처에 존재하는 파라미터만 추려 호출."""
                sig = inspect.signature(func)
                filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
                return func(**filtered)

            # image.json 생성
            self._dbg("[DOCS] calling build_image_json()")
            image_json_path = _call_compatible(
                build_image_json,
                project_dir=str(proj),
                hair_map=hm,
                workflow_dir=wfdir,  # 없으면 자동 필터
            )
            self._dbg(f"[DOCS] build_image_json -> {image_json_path}")

            # movie.json 생성
            self._dbg("[DOCS] calling build_movie_json()")
            movie_json_path = _call_compatible(
                build_movie_json,
                project_dir=str(proj),
                hair_map=hm,
                workflow_dir=wfdir,  # 없으면 자동 필터
            )
            self._dbg(f"[DOCS] build_movie_json -> {movie_json_path}")

            # 인트로/아웃트로 타임라인 10% 반영
            self._dbg("[DOCS] applying intro/outro timelines (10%)")
            story_json_path = apply_intro_outro_to_story_json(str(proj), intro_ratio=0.10, outro_ratio=0.10)
            image_json_path2 = apply_intro_outro_to_image_json(str(proj), intro_ratio=0.10, outro_ratio=0.10)
            movie_json_path2 = apply_intro_outro_to_movie_json(str(proj), intro_ratio=0.10, outro_ratio=0.10)

            return {
                "image": str(image_json_path or image_json_path2),
                "movie": str(movie_json_path or movie_json_path2),
                "story": str(story_json_path),
            }

        def on_done(res, err):
            try:
                if err:
                    self._dbg(f"[DOCS] ERROR: {err}")
                    QtWidgets.QMessageBox.critical(self, "문서 생성 실패", str(err))
                    self.status.showMessage("image.json / movie.json 생성 실패")
                    return
                imgp = res.get("image")
                movp = res.get("movie")
                self._dbg(f"[DOCS] DONE: image={imgp} | movie={movp}")
                QtWidgets.QMessageBox.information(self, "완료", f"image.json: {imgp}\nmovie.json: {movp}")
                self.status.showMessage(f"문서 생성 완료: {imgp} / {movp}")
            finally:
                self._docs_build_running = False

        th = QtCore.QThread(self)
        wk = Worker(job)
        wk.moveToThread(th)

        # ✅ protected 멤버 직접접근 회피: 공개 'run' 우선, 없으면 문자열 기반 getattr로 '_run'
        run_slot = getattr(wk, "run", None)
        if not callable(run_slot):
            run_slot = getattr(wk, "_run", None)  # linter의 protected-access 회피
        if callable(run_slot):
            th.started.connect(run_slot)  # type: ignore[arg-type]

        wk.done.connect(on_done)
        wk.done.connect(th.quit)
        wk.done.connect(wk.deleteLater)
        th.finished.connect(th.deleteLater)
        th.start()

    # 워크플로우 이용
    def _comfy_run_workflow(self, wf_path: Path, prompt_text: str, out_png: Path,
                            char_ref: Path | None = None) -> bool:
        """
        wf_path: nunchaku_t2i.json 또는 nunchaku-t2i_swap.json
        prompt_text: 장면 프롬프트(한국어 OK, 내부에서 Deep Translator 노드가 처리한다고 가정)
        out_png: 결과 저장 대상 (파일명 기준, 폴더/프리픽스/파일명 모두 세팅 시도)
        char_ref: i2i_swap일 때 캐릭터 참조 이미지 경로(없으면 t2i로 사용)
        """
        # ---- 단일 임포트 + 로그 ----
        try:
            import uuid, time, shutil
            from settings import COMFY_HOST, JSONS_DIR
        except Exception:
            import uuid, time, shutil
            from app.settings import COMFY_HOST, JSONS_DIR  # type: ignore

        print(f"[COMFY][T2_1] wf='{Path(wf_path).name}', out='{out_png.name}', "
              f"char_ref='{char_ref if char_ref else '-'}'", flush=True)
        print(f"[COMFY][T2_1] prompt_len={len(prompt_text)} | preview={prompt_text[:120]!r}", flush=True)

        # ---- 워크플로 JSON 찾기 ----
        wf_path = Path(wf_path)
        if not wf_path.exists():
            wf_path = Path(JSONS_DIR) / wf_path.name
            print(f"[COMFY][T2_1] fallback JSONS_DIR → {wf_path}", flush=True)
            if not wf_path.exists():
                QtWidgets.QMessageBox.warning(self, "Comfy", f"워크플로 JSON 없음:\n{wf_path}")
                return False

        with open(wf_path, "r", encoding="utf-8") as f:
            wf = json.load(f)

        g = wf.get("prompt") or wf  # 일부 템플릿은 루트가 곧 그래프
        if not isinstance(g, dict):
            print("[COMFY][T2_1] invalid workflow JSON (prompt dict 아님)", flush=True)
            QtWidgets.QMessageBox.warning(self, "Comfy", "워크플로 JSON 형식이 prompt(dict)가 아닙니다.")
            return False

        # ---- 1) 텍스트 프롬프트 주입 ----
        for node in g.values():
            if not isinstance(node, dict):
                continue
            ins = node.setdefault("inputs", {})
            for k in ("text", "prompt"):
                if k in ins and isinstance(ins[k], str) and len(ins[k]) < 4000:
                    ins[k] = prompt_text

        # ---- 2) i2i_swap 참조 이미지 ----
        if char_ref:
            for node in g.values():
                if not isinstance(node, dict):
                    continue
                ins = node.setdefault("inputs", {})
                for k in ("image", "ref_image", "input_image", "image_path"):
                    if k in ins and isinstance(ins[k], str):
                        ins[k] = str(char_ref)

        # ---- 3) 저장 노드(출력 경로/파일명) ----
        out_dir = out_png.parent
        out_stem = out_png.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        for node in g.values():
            if not isinstance(node, dict):
                continue
            ct = str(node.get("class_type", ""))
            if "SaveImage" in ct:  # SaveImage / SaveImagePIL 등 호환
                ins = node.setdefault("inputs", {})
                if "filename" in ins:
                    ins["filename"] = str(out_png)  # 절대 경로 지원 템플릿
                if "filename_prefix" in ins:
                    ins["filename_prefix"] = str(out_dir / out_stem)
                if "output_path" in ins:
                    ins["output_path"] = str(out_dir)

        # ---- 4) 요청 전송 ----
        payload = {"prompt": g, "client_id": str(uuid.uuid4())}
        try:
            url = COMFY_HOST.rstrip("/") + "/prompt"
            print(f"[COMFY][T2_1] POST {url}", flush=True)
            r = requests.post(url, json=payload, timeout=30)
            r.raise_for_status()
            print(f"[COMFY][T2_1] POST OK ({r.status_code}) → wait for '{out_png.name}'", flush=True)
        except Exception as e:
            print(f"[COMFY][T2_1] POST FAIL: {e}", flush=True)
            return False

        # ---- 5) 파일 생성 대기(간단 폴링) ----
        for _ in range(40):  # 최대 ~20초
            if out_png.exists() and out_png.stat().st_size > 0:
                print(f"[COMFY][T2_1] saved: {out_png} ({out_png.stat().st_size} bytes)", flush=True)
                return True
            time.sleep(0.5)

        # prefix로 생성된 최신 파일을 표준 이름으로 복사 시도
        cand = sorted(out_dir.glob(out_stem + "*.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cand and cand[0].stat().st_size > 0:
            try:
                shutil.copyfile(str(cand[0]), str(out_png))
                print(f"[COMFY][T2_1] copied from '{cand[0].name}' → '{out_png.name}'", flush=True)
                return True
            except Exception as ce:
                print(f"[COMFY][T2_1] copy-fallback FAIL: {ce}", flush=True)

        print(f"[COMFY][T2_1] FAIL to produce '{out_png.name}'", flush=True)
        return False



    def _find_character_asset_path(self, char_id: str) -> Path | None:
        """프로젝트→글로벌 순으로 캐릭터 자산 하나 찾기."""
        proj = self._latest_project()
        if proj:
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                p = Path(proj) / "character" / f"{char_id}{ext}"
                if p.exists(): return p
        # 글로벌 풀
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = Path(r"C:\my_games\shorts_make\character") / f"{char_id}{ext}"
            if p.exists(): return p
        return None



    # ==== [SCHEMA HELPERS] story.json 강제 scenes 스키마 ====
    @staticmethod
    def _assert_scenes_story(story: dict) -> None:
        need_scene = [
            "id", "section", "start", "end", "duration", "scene", "characters",
            "effect", "screen_transition", "img_file", "prompt",
        ]
        # 선택 필드의 안전 기본값(필요시)
        optional_defaults = {
            "prompt_img": "",
            "prompt_movie": "",
            "prompt_negative": "",
            "character_objs": [],
            "clip_file": "",
        }

        for idx, sc in enumerate(story.get("scenes") or [], 1):
            for k, v in optional_defaults.items():
                if k not in sc:
                    sc[k] = v
            miss2 = [k for k in need_scene if k not in sc]
            if miss2:
                raise ValueError(f"scenes[{idx}] missing keys: {miss2}")

    def _convert_shots_file_to_scenes(self, story_path: Path) -> bool:
        """story.json에 shots가 있으면 scenes 스키마로 변환 후 덮어쓴다. 변환했으면 True."""
        data = load_json(story_path, {}) or {}
        if "shots" not in data:
            return False

        # 제목/이미지 폴더 추정
        title = (self.le_title.text().strip()
                 or str(data.get("title") or story_path.parent.name)
                 or "untitled")
        img_dir = self._img_dir_for_title(title)
        img_dir.mkdir(parents=True, exist_ok=True)

        scenes = []
        shots = data.get("shots") or []
        for sh in shots:
            sid = (sh.get("title")
                   or f"t_{int(sh.get('idx', 0)):02d}"
                   or f"t_{len(scenes)+1:02d}")
            section = str(sh.get("section", "verse")).lower().strip()
            scenes.append({
                "id": sid,
                "section": section,
                "start": float(sh.get("start", 0.0) or 0.0),
                "end": float(sh.get("end", 0.0) or 0.0),
                "duration": float(sh.get("duration", 0.0) or 0.0),
                "scene": sh.get("scene") or "",
                "characters": ["female_01", "male_01"] if section == "chorus" else ["female_01"],
                "effect": [sh.get("effect")] if sh.get("effect") else [],
                "screen_transition": bool(sh.get("screen_transition", False)),
                "img_file": (img_dir / f"{sid}.png").as_posix(),
                "prompt": (sh.get("prompt") or "연출/카메라/분위기 중심; 가사 금지").strip(),
                "needs_character_asset": False,
            })

        new_story = {
            "title": title,
            "audio": (Path(data.get("audio")) if data.get("audio") else (story_path.parent / "vocal.mp3")).as_posix(),
            "duration": float(data.get("duration", 0.0) or 0.0),
            "characters": ["female_01", "male_01"],
            "scenes": scenes,
        }
        save_json(story_path, new_story)
        print(f"[SCHEMA] converted 'shots' → 'scenes' and overwrote: {story_path}", flush=True)
        return True

    def _normalize_story_file_to_scenes(self, story_path: Path) -> Path:
        """파일을 읽어 shots면 scenes로 변환, 아니면 스키마 검증."""
        data = load_json(story_path, {}) or {}
        if "shots" in data:
            self._convert_shots_file_to_scenes(story_path)
            data = load_json(story_path, {}) or {}
        # 최종 검증(에러 나면 어디서 틀렸는지 바로 알림)
        self._assert_scenes_story(data)
        return story_path

    @staticmethod
    def _ffmpeg_exe() -> str:
        try:
            import settings as _settings
        except ImportError:
            from app import settings as _settings  # type: ignore
        return _settings.FFMPEG_EXE or "ffmpeg"

    def _build_clip_from_image(self, img: Path, out_mp4: Path, duration: float, fps: int = 24,
                               width: int = 1080, height: int = 1920) -> bool:
        """
        단일 이미지로 duration 길이의 mp4를 만든다(세로 1080x1920 기본).
        - 이미지 비율 유지(scale + pad)
        - 코덱: h264, yuv420p
        """
        import subprocess, shlex
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        ff = self._ffmpeg_exe()
        vf = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        cmd = [
            ff, "-y",
            "-loop", "1",
            "-t", f"{max(0.1, float(duration))}",
            "-i", str(img),
            "-r", str(int(max(1, fps))),
            "-vf", vf,
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            str(out_mp4),
        ]
        print("[TEST2] ffmpeg:", " ".join(shlex.quote(c) for c in cmd), flush=True)
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        ok = (cp.returncode == 0 and out_mp4.exists() and out_mp4.stat().st_size > 0)
        if not ok:
            print("[TEST2][ffmpeg][stderr]", cp.stderr[:2000], flush=True)
        return ok















    def _get_prompt_text(self) -> str:
        # 기본 경로: 별칭이 있으면 바로 사용
        w = getattr(self, "txt_prompt", None)
        if isinstance(w, (QPlainTextEdit, QTextEdit)):
            try:
                return w.toPlainText().strip()
            except Exception:
                pass
        # 혹시 모를 대비: 후보 이름들 재탐색
        for name in ("txt_prompt", "prompt_edit", "te_prompt", "input_prompt", "plainTextEdit_prompt",
                     "textEdit_prompt"):
            w = getattr(self, name, None)
            if isinstance(w, (QPlainTextEdit, QTextEdit)):
                try:
                    return w.toPlainText().strip()
                except Exception:
                    pass
        # 최후: 모든 QTextEdit/ QPlainTextEdit 중 첫 번째
        for w in self.findChildren((QPlainTextEdit, QTextEdit)):
            try:
                return w.toPlainText().strip()
            except Exception:
                pass
        return ""

    @staticmethod
    def _spin(lo: int, hi: int, v: int, suffix: str = "") -> QtWidgets.QSpinBox:
        sb = QtWidgets.QSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(v)
        if suffix:
            sb.setSuffix(suffix)
        return sb

    @staticmethod
    def _group(title: str, widget: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox(title)
        l = QtWidgets.QVBoxLayout(g)
        l.addWidget(widget)
        return g

    @staticmethod
    def _build_checks_grid(names: List[str], columns: int = 4):
        cont = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(cont)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setHorizontalSpacing(14)
        cbs: List[QtWidgets.QCheckBox] = []
        for i, name in enumerate(names):
            cb = QtWidgets.QCheckBox(name)
            r, c = divmod(i, columns)
            grid.addWidget(cb, r, c)
            cbs.append(cb)
        return cont, cbs

    # ────────────── 최신 프로젝트/현재 프로젝트 ──────────────
    def _latest_project(self) -> Optional[Path]:
        if self._forced_project_dir and Path(self._forced_project_dir).exists():
            return Path(self._forced_project_dir)
        root = Path(BASE_DIR)
        if not root.exists():
            return None
        subs = [p for p in root.glob("*") if p.is_dir()]
        return max(subs, key=lambda p: p.stat().st_mtime) if subs else None

    def _current_project_dir(self) -> Optional[Path]:
        return self._forced_project_dir or self._latest_project()

    # ────────────── UI 구축 ──────────────
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        # 제목/가사

        self.le_title = QtWidgets.QLineEdit()
        self.le_title.setPlaceholderText("노래 제목 (비우면 자동 생성)")
        self.te_lyrics = QtWidgets.QTextEdit()
        self.te_lyrics.setPlaceholderText("여기에 가사가 표시됩니다")


        # 길이 선택
        self.grp_len = QtWidgets.QGroupBox("곡 길이")

        self.rb_20s = QtWidgets.QRadioButton("30초(테스트)")  # ★ 추가
        self.rb_1m = QtWidgets.QRadioButton("1분")
        self.rb_2m = QtWidgets.QRadioButton("2분")
        self.rb_3m = QtWidgets.QRadioButton("3분")
        self.rb_2m.setChecked(True)
        g = QtWidgets.QButtonGroup(self)
        g.setExclusive(True)
        for b in (self.rb_20s, self.rb_1m, self.rb_2m, self.rb_3m):
            g.addButton(b)
        l_lay = QtWidgets.QHBoxLayout(self.grp_len)
        l_lay.addWidget(self.rb_20s)
        l_lay.addWidget(self.rb_1m)
        l_lay.addWidget(self.rb_2m)
        l_lay.addWidget(self.rb_3m)
        self._add_direct_len_controls_to_grp_len()
        l_lay.addStretch(1)



        # 프롬프트
        self.te_prompt = QtWidgets.QTextEdit()
        self.te_prompt.setPlaceholderText("무드/키워드 입력")
        prompt_grp = self._group("프롬프트", self.te_prompt)

        # 자동 태그 토글
        self.cb_auto_tags = QtWidgets.QCheckBox("태그 자동(가사 분위기 기반 추천)")
        self.cb_auto_tags.setChecked(True)
        self.cb_auto_tags.toggled.connect(self._on_auto_toggle)

        # ai select 토글
        self.btn_ai_toggle = QtWidgets.QToolButton()
        self.btn_ai_toggle.setCheckable(True)
        self.btn_ai_toggle.setText("모드: GPT 우선")
        self.btn_ai_toggle.setToolTip("클릭: Gemini만 사용 / 다시 클릭: GPT 우선(부족 시 Gemini 폴백)")
        self.btn_ai_toggle.toggled.connect(self.on_ai_toggle)

        # Vocal 선택
        self.grp_vocal = QtWidgets.QGroupBox("Vocal(배타 선택)")
        self.rb_vocal_female = QtWidgets.QRadioButton("soft female voice")
        self.rb_vocal_male   = QtWidgets.QRadioButton("soft male voice")
        self.rb_vocal_mixed  = QtWidgets.QRadioButton("mixed vocals")
        self.rb_vocal_female.setChecked(True)
        self.bg_vocal = QtWidgets.QButtonGroup(self)
        self.bg_vocal.setExclusive(True)
        for w in (self.rb_vocal_female, self.rb_vocal_male, self.rb_vocal_mixed):
            self.bg_vocal.addButton(w)
        vgrid = QtWidgets.QGridLayout(self.grp_vocal)
        vgrid.addWidget(self.rb_vocal_female, 0, 0)
        vgrid.addWidget(self.rb_vocal_male,   0, 1)
        vgrid.addWidget(self.rb_vocal_mixed,  0, 2)

        # Basic Vocal 묶음
        self.grp_basic_vocal = QtWidgets.QGroupBox("Basic Vocal")
        basic_cont, basic_cbs = self._build_checks_grid(self.BASIC_VOCAL_TAGS, columns=4)
        self.cb_basic_vocal_list = basic_cbs
        blay = QtWidgets.QVBoxLayout(self.grp_basic_vocal)
        blay.addWidget(basic_cont)

        # 수동 태그(Style/Scene/Instrument/Tempo)
        style_list = ["electronic","rock","pop","funk","soul","cyberpunk","acid jazz","edm","soft electric drums","melodic"]

        scene_list = ["background music for parties","radio broadcasts","workout playlists"]
        instr_list = ["saxophone","jazz","piano","violin","acoustic guitar","electric bass"]
        tempo_list = ["110 bpm","fast tempo","slow tempo","loops","fills"]

        self.grp_manual_tags = QtWidgets.QGroupBox("수동 태그(체크)")
        cat_wrap = QtWidgets.QGridLayout(self.grp_manual_tags)

        def _make_cat_box(title: str, names: List[str]):
            box = QtWidgets.QGroupBox(title)
            cont, cbs = self._build_checks_grid(names, columns=4)
            lay = QtWidgets.QVBoxLayout(box)
            lay.addWidget(cont)
            return box, cbs

        box_style, self.cb_style_checks = _make_cat_box("Style", style_list)
        box_scene, self.cb_scene_checks = _make_cat_box("Scene", scene_list)
        box_instr, self.cb_instr_checks = _make_cat_box("Instrument", instr_list)
        box_tempo, self.cb_tempo_checks = _make_cat_box("Tempo/Pro", tempo_list)
        cat_wrap.addWidget(box_style, 0, 0)
        cat_wrap.addWidget(box_scene, 0, 1)
        cat_wrap.addWidget(box_instr, 1, 0)
        cat_wrap.addWidget(box_tempo, 1, 1)

        # 상단 배치(제목/가사/태그)
        top = QtWidgets.QVBoxLayout()
        top.addWidget(self._group("제목", self.le_title))
        top.addWidget(self._build_lyrics_group_three_columns(), 1)  # ← 가사 그룹 교체



        # 옵션(영상 관련)
        self.sb_total   = self._spin(60, 120000, 1200,              " frames")
        self.sb_chunk   = self._spin(60,   5000,   DEFAULT_CHUNK,   " /chunk")
        self.sb_overlap = self._spin(1,     120,   DEFAULT_OVERLAP, " overlap")
        self.sb_infps   = self._spin(1,     240,   DEFAULT_INPUT_FPS,  " inFPS")
        self.sb_outfps  = self._spin(1,     240,   DEFAULT_TARGET_FPS, " outFPS")
        opts = QtWidgets.QHBoxLayout()
        for w in (self.sb_total, self.sb_chunk, self.sb_overlap, self.sb_infps, self.sb_outfps):
            opts.addWidget(w)
        opts.addStretch(1)

        # 버튼들
        self.btn_gen = QtWidgets.QPushButton("가사생성")

        self.btn_save = QtWidgets.QPushButton("프로젝트 저장")
        self.btn_load_proj = QtWidgets.QPushButton("프로젝트 불러오기")
        self.btn_music = QtWidgets.QPushButton("음악생성(ACE-Step)")
        self.btn_show_progress = QtWidgets.QPushButton("진행상황 보기")
        self.btn_video = QtWidgets.QPushButton("영상생성(i2v)")
        self.btn_analyze = QtWidgets.QPushButton("음악분석")
        # self.btn_analyze.hide()  # 음악생성 완료 시 자동 분석으로 대체



        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.btn_gen)  # ① 가사생성 추가
        self._add_clear_button_next_to_generate(row)  # ② ← 여기로 이동/교체
        for b in (self.btn_save, self.btn_load_proj, self.btn_music, # ③ 나머지 버튼들
                  self.btn_show_progress, self.btn_video, self.btn_analyze):
            row.addWidget(b)

        # ==== ★ 테스트 버튼 3개 추가 ====
        self.btn_test1_story = QtWidgets.QPushButton("프로젝트분석")
        self.btn_test2_render = QtWidgets.QPushButton("테스트2: story→샷 렌더")
        self.btn_test3_concat = QtWidgets.QPushButton("테스트3: 샷 합치기")
        self.btn_test2_1_img = QtWidgets.QPushButton("테스트2_1: 누락 이미지 생성")

        row_test = QtWidgets.QHBoxLayout()
        for b in (self.btn_test1_story, self.btn_test2_render, self.btn_test3_concat, self.btn_test2_1_img):
            row_test.addWidget(b)

        # 메인 탭
        main_tab = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(main_tab)
        main_layout.addWidget(self.grp_len)
        main_layout.addWidget(prompt_grp)
        main_layout.addWidget(self.cb_auto_tags)
        main_layout.addWidget(self.grp_vocal)
        main_layout.addWidget(self.grp_basic_vocal)
        main_layout.addWidget(self.grp_manual_tags)
        main_layout.addLayout(opts)
        main_layout.addWidget(self.btn_ai_toggle)
        self._add_render_prefs_controls(main_layout)
        main_layout.addLayout(row)



        main_layout.addLayout(row_test)  # ★ 테스트 줄 추가

        # 설정 탭
        settings_tab = self._build_settings_tab()

        # 탭 구성
        tabs = QtWidgets.QTabWidget(self)
        tabs.addTab(main_tab, "메인")
        tabs.addTab(settings_tab, "설정")

        # 루트 레이아웃
        root = QtWidgets.QVBoxLayout(central)
        root.addLayout(top, 5)
        root.addWidget(tabs, 5)

        # 초기 상태: 자동 ON이면 수동 영역 비활성
        self._on_auto_toggle(self.cb_auto_tags.isChecked())

        self._ensure_wire_lyrics_to_direct_seconds()

        # ★ 변환 버튼 액션 연결(오른쪽 칸에 주입)
        self._wire_convert_toggle_action()

    # ai 토글
    def on_ai_toggle(self, checked: bool) -> None:
        if checked:
            self.btn_ai_toggle.setText("모드: Gemini만")
            try:
                self._ai.default_prefer = "gemini"
            except AttributeError:
                pass
        else:
            self.btn_ai_toggle.setText("모드: GPT 우선")
            try:
                self._ai.default_prefer = "openai"
            except AttributeError:
                pass

    def _estimate_seconds_from_lyrics(self, text: str) -> int:
        """
        가사 길이 추정:
          - [chorus] 본문은 2배 가중
          - 공백/빈 줄/섹션 헤더는 제외
          - 기본 초/글자(self.seconds_per_char 없으면 0.72)
          - 최종값에 1.2배 버퍼를 곱해(여유 시간) 정수 초로 반환
          - 1~3600초로 클램프
        """
        import re, math

        section = "default"
        pieces: list[tuple[str, str]] = []

        for raw in (text or "").splitlines():
            s = (raw or "").strip()
            if not s:
                continue
            m = re.fullmatch(r"\[([^\[\]\n]+)]", s, flags=re.I)
            if m:
                section = m.group(1).strip().lower()
                continue
            pieces.append((section, s))

        total_chars = 0
        for sec, content in pieces:
            body = re.sub(r"\s+", "", content)
            if not body:
                continue
            weight = 2 if sec == "chorus" else 1
            total_chars += weight * len(body)

        if total_chars <= 0:
            return 0

        spc = getattr(self, "seconds_per_char", 0.72)  # 초/글자
        seconds_raw = total_chars * float(spc)
        seconds_buf = math.ceil(seconds_raw * 1.20)  # ← 1.2배 버퍼 적용(올림)

        if seconds_buf < 1:
            return 1
        if seconds_buf > 3600:
            return 3600
        return int(seconds_buf)

    def _wire_convert_toggle_action(self) -> None:
        """
        '변환' 버튼(self.btn_convert_toggle)을 kroman 기반 로마자 변환에 연결한다.
        - ON : 왼쪽 가사(self.te_lyrics)의 각 본문 라인을 kroman.parse()로 변환해
                라인 앞에 [ko]를 붙여 오른쪽(self.te_lyrics_converted)에 주입
                (추가: 변환 결과에서 하이픈('-')만 제거)
        - OFF: 오른쪽 칸 비움
        - 섹션 헤더([intro]/[verse]/[chorus]/[outro] 등)와 빈 줄은 그대로 유지한다.
        - 다른 기능/레이아웃은 변경하지 않는다.
        """
        from PyQt5 import QtWidgets

        btn = getattr(self, "btn_convert_toggle", None)
        src = getattr(self, "te_lyrics", None)
        dst = getattr(self, "te_lyrics_converted", None)
        if not isinstance(btn, QtWidgets.QPushButton):
            return
        if not isinstance(src, QtWidgets.QTextEdit):
            return
        if not isinstance(dst, QtWidgets.QTextEdit):
            return
        if getattr(self, "_convert_toggle_wired", False):
            return

        try:
            import kroman  # type: ignore
        except ImportError:
            return

        def _is_section_header(line: str) -> bool:
            s = (line or "").strip()
            return s.startswith("[") and s.endswith("]") and len(s) >= 2

        def _convert_text_with_kroman(text: str) -> str:
            out_lines = []
            for raw in (text or "").splitlines():
                s = (raw or "")
                if _is_section_header(s) or not s.strip():
                    out_lines.append(s)
                    continue
                body = s
                rom = kroman.parse(body).strip()
                rom = rom.replace("-", "")  # ← 하이픈만 제거
                out_lines.append("[ko]" + rom)
            return "\n".join(out_lines)

        def _on_toggle(checked: bool) -> None:
            if checked:
                text = src.toPlainText()
                converted = _convert_text_with_kroman(text)
                dst.setPlainText(converted)
            else:
                dst.clear()

        btn.toggled.connect(_on_toggle)
        self._convert_toggle_wired = True

    def _build_lyrics_group_three_columns(self) -> QtWidgets.QGroupBox:
        """
        '가사' 그룹박스를 새로 만들어 4:2:4(가사/변환토글/변환결과)로 배치해 반환한다.
        - 기존 self.te_lyrics 그대로 사용(새 부모로 붙이기만 함)
        - 가운데 칼럼은 '변환' 토글 버튼(self.btn_convert_toggle)
        - 오른쪽 칼럼은 읽기전용 QTextEdit(self.te_lyrics_converted)
        기능 연결 없이 UI만 구성한다.
        """
        from PyQt5 import QtWidgets

        # 1) 그룹박스 + 가로 레이아웃
        grp = QtWidgets.QGroupBox("가사")
        row = QtWidgets.QWidget(grp)
        lay_grp = QtWidgets.QVBoxLayout(grp)
        lay_grp.setContentsMargins(6, 6, 6, 6)
        lay_grp.setSpacing(6)
        lay_row = QtWidgets.QHBoxLayout(row)
        lay_row.setContentsMargins(0, 0, 0, 0)
        lay_row.setSpacing(8)
        lay_grp.addWidget(row)

        # 2) 왼쪽(4): 기존 가사 입력 위젯
        te = getattr(self, "te_lyrics", None)
        if not isinstance(te, QtWidgets.QTextEdit):
            te = QtWidgets.QTextEdit()  # 안전 폴백
            self.te_lyrics = te
            self.te_lyrics.setPlaceholderText("여기에 가사가 표시됩니다")
        te.setParent(row)
        te.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        lay_row.addWidget(te, 4)

        # 3) 가운데(2): 변환 on/off 버튼(수평·수직 중앙 정렬: 스트레치로)
        col2 = QtWidgets.QWidget(row)
        col2_v = QtWidgets.QVBoxLayout(col2)
        col2_v.setContentsMargins(0, 0, 0, 0)
        col2_v.setSpacing(6)

        hwrap = QtWidgets.QWidget(col2)
        hwrap_h = QtWidgets.QHBoxLayout(hwrap)
        hwrap_h.setContentsMargins(0, 0, 0, 0)
        hwrap_h.setSpacing(0)

        btn = QtWidgets.QPushButton("변환")
        btn.setObjectName("btn_convert_toggle")
        btn.setCheckable(True)
        btn.setMinimumHeight(28)
        btn.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

        hwrap_h.addStretch(1)
        hwrap_h.addWidget(btn)
        hwrap_h.addStretch(1)

        col2_v.addStretch(1)
        col2_v.addWidget(hwrap)
        col2_v.addStretch(1)

        lay_row.addWidget(col2, 2)

        # 4) 오른쪽(4): 변환 결과 표시 칸
        te_conv = QtWidgets.QTextEdit(row)
        te_conv.setObjectName("te_lyrics_converted")
        te_conv.setReadOnly(True)
        te_conv.setPlaceholderText("변환된 가사가 여기 표시됩니다")
        te_conv.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        lay_row.addWidget(te_conv, 4)

        # 참조 보관
        self.btn_convert_toggle = btn
        self.te_lyrics_converted = te_conv

        return grp

    def _ensure_wire_lyrics_to_direct_seconds(self) -> None:
        """
        가사 입력(self.te_lyrics)의 textChanged → '직접' 옆 칸(self.le_len_seconds)에
        실시간 초 추정치를 채워넣는다.
        - 두 위젯이 아직 없으면 QTimer로 '나중에' 다시 시도한다(기존 기능 불변).
        - 중복 연결 방지 플래그 사용.
        """
        from PyQt5 import QtCore, QtWidgets

        if getattr(self, "_lyrics_len_wired", False):
            return

        te = getattr(self, "te_lyrics", None)
        le = getattr(self, "le_len_seconds", None)

        te_ok = isinstance(te, (QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit))
        le_ok = isinstance(le, QtWidgets.QLineEdit)
        if not (te_ok and le_ok):
            QtCore.QTimer.singleShot(100, self._ensure_wire_lyrics_to_direct_seconds)
            return

        # 기본 글자당 시간: 0.72초 (필요시 외부에서 self.seconds_per_char로 조정 가능)
        if not hasattr(self, "seconds_per_char"):
            self.seconds_per_char = 0.72

        def _on_text_changed() -> None:
            text = te.toPlainText()
            sec = self._estimate_seconds_from_lyrics(text)
            le.setText(str(int(sec)))

        te.textChanged.connect(_on_text_changed)
        self._lyrics_len_wired = True
        _on_text_changed()



    def _add_direct_len_controls_to_grp_len(self) -> None:
        """
        '곡 길이' 그룹(self.grp_len)의 기존 프리셋(30초/1분/2분/3분) '바로 뒤'에
        같은 QButtonGroup 소속으로 '직접' 라디오와 정수 전용 입력칸(초)을
        같은 줄(HBox)에 정확히 추가한다.
        - 기존 레이아웃/기능은 변경하지 않는다.
        """
        grp = getattr(self, "grp_len", None)
        if grp is None:
            return

        layout = grp.layout()
        if not isinstance(layout, QtWidgets.QHBoxLayout):
            return

        # 이미 추가돼 있으면 중복 생성 방지
        if isinstance(getattr(self, "rb_len_direct", None), QtWidgets.QRadioButton) and \
                isinstance(getattr(self, "le_len_seconds", None), QtWidgets.QLineEdit):
            return

        # '직접' 라디오
        rb = QtWidgets.QRadioButton("직접", grp)
        rb.setObjectName("rb_len_direct")

        # 숫자 전용 입력칸 (초)
        le = QtWidgets.QLineEdit(grp)
        le.setObjectName("le_len_seconds")
        le.setPlaceholderText("초")
        from PyQt5 import QtGui
        le.setValidator(QtGui.QIntValidator(1, 3600, le))  # 1~3600초 허용
        le.setFixedWidth(80)
        le.setClearButtonEnabled(True)
        le.setEnabled(False)  # '직접' 선택 시 활성화

        def _on_direct_toggled(checked: bool) -> None:
            le.setEnabled(checked)

        rb.toggled.connect(_on_direct_toggled)

        # 같은 QButtonGroup에 합류(동등한 위치 보장)
        group = None
        for name in ("rb_20s", "rb_1m", "rb_2m", "rb_3m"):
            btn = getattr(self, name, None)
            if isinstance(btn, QtWidgets.QRadioButton):
                group = btn.group()
                if isinstance(group, QtWidgets.QButtonGroup):
                    break
        if isinstance(group, QtWidgets.QButtonGroup):
            group.addButton(rb)

        # ---------- 핵심: 프리셋 '바로 뒤'에 끼워 넣기 ----------
        # 1) rb_3m 항목의 인덱스를 찾는다.
        insert_index = layout.count()  # 기본값: 맨 끝
        target = getattr(self, "rb_3m", None)
        if isinstance(target, QtWidgets.QRadioButton):
            for i in range(layout.count()):
                item = layout.itemAt(i)
                w = item.widget()
                if isinstance(w, QtWidgets.QRadioButton) and w is target:
                    insert_index = i + 1  # '3분' 바로 뒤
                    break

        # 2) 만약 스트레치/스페이서가 프리셋 뒤에 있다면, 그 앞에 삽입하도록 보정
        #    (레이아웃 중간의 stretch가 '사이를 벌려' 보이게 하는 문제 대응)
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item is not None and item.spacerItem() is not None:
                if i < insert_index:
                    # spacer가 더 앞에 있으면, 그 다음으로 삽입
                    insert_index = i + 1
                else:
                    # spacer가 뒤쪽이면, spacer 직전으로 삽입
                    insert_index = i
                break

        # 3) 최종 삽입: 프리셋 뒤에 '직접'과 '입력'을 순서대로 넣는다.
        #    필요시 간격도 함께 삽입(시각 간격 최소화)
        layout.insertSpacing(insert_index, 12)
        layout.insertWidget(insert_index + 1, rb)
        layout.insertWidget(insert_index + 2, le)

        # 참조 보관
        self.rb_len_direct = rb
        self.le_len_seconds = le

    @staticmethod
    def _find_vocal_in_project(proj: Path) -> Path | None:
        """프로젝트 폴더에서 분석에 쓸 오디오를 찾는다. wav 우선, 그다음 mp3, 그 외 vocal.*"""
        cand = [proj / "vocal.wav", proj / "vocal.mp3"]
        for p in cand:
            if p.exists() and p.stat().st_size > 0:
                return p
        for p in proj.glob("vocal.*"):
            if p.is_file() and p.stat().st_size > 0:
                return p
        return None



    def _maybe_convert_lyrics_for_api(self, lyrics_text: str) -> str:
        """
        음악 생성 시 넘길 가사를 최종 결정한다.
        - '변환' 버튼이 ON이면 오른쪽 칸(self.te_lyrics_converted)의 내용을 우선 사용.
          (비어 있으면 kroman으로 즉석 변환해 [ko] 접두 + 하이픈 제거)
        - OFF면 원본 lyrics_text 그대로 반환.
        - 기존 기능 보존: 다른 로직/파일 저장 방식은 건드리지 않는다.
        """
        from PyQt5 import QtWidgets

        btn = getattr(self, "btn_convert_toggle", None)
        if isinstance(btn, QtWidgets.QPushButton) and btn.isChecked():
            # 1순위: 사용자가 화면에서 확인한 변환 결과가 있으면 그대로 사용
            dst = getattr(self, "te_lyrics_converted", None)
            if hasattr(dst, "toPlainText"):
                txt = dst.toPlainText().strip()
                if txt:
                    return txt

            # 2순위: 변환 결과가 비어 있으면 kroman으로 라인별 변환
            try:
                import kroman  # type: ignore
            except ImportError:
                # kroman이 없으면 원본 그대로 (기능 보존)
                return lyrics_text

            lines_out = []
            for raw in (lyrics_text or "").splitlines():
                s = raw
                st = s.strip()
                # 섹션 헤더/빈 줄은 그대로 유지
                if (st.startswith("[") and st.endswith("]")) or not st:
                    lines_out.append(s)
                    continue
                rom = kroman.parse(s).strip().replace("-", "")  # 하이픈 제거
                lines_out.append("[ko]" + rom)
            return "\n".join(lines_out)

        # 변환 OFF → 원본 사용
        return lyrics_text

    def _build_settings_tab(self) -> QtWidgets.QWidget:
        # 항상 같은 모듈(alias S)만 쓰도록 통일!
        # import settings as S

        tab = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout()

        # BASE_DIR
        self.le_base_dir = QtWidgets.QLineEdit(str(S.BASE_DIR))
        btn_pick_base = QtWidgets.QPushButton("폴더 선택")

        def _pick_base():
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "BASE_DIR 선택", str(S.BASE_DIR))
            if d: self.le_base_dir.setText(d)

        btn_pick_base.clicked.connect(_pick_base)
        base_wrap = QtWidgets.QHBoxLayout()
        base_wrap.addWidget(self.le_base_dir)
        base_wrap.addWidget(btn_pick_base)
        base_widget = QtWidgets.QWidget()
        base_widget.setLayout(base_wrap)

        # COMFY_HOST & 후보
        self.le_comfy = QtWidgets.QLineEdit(S.COMFY_HOST)
        self.te_candidates = QtWidgets.QPlainTextEdit("\n".join(S.DEFAULT_HOST_CANDIDATES))

        # ffmpeg / hwaccel / 출력파일 / 오디오 포맷
        self.le_ffmpeg = QtWidgets.QLineEdit(S.FFMPEG_EXE)
        self.cb_hwaccel = QtWidgets.QCheckBox("USE_HWACCEL")
        self.cb_hwaccel.setChecked(bool(S.USE_HWACCEL))
        self.le_final = QtWidgets.QLineEdit(S.FINAL_OUT)

        self.cb_audio_fmt = QtWidgets.QComboBox()
        self.cb_audio_fmt.addItems(["mp3", "wav", "opus"])
        # overrides 적용 고려
        ov = S.load_overrides() or {}
        cur_fmt = str(ov.get("AUDIO_SAVE_FORMAT", S.AUDIO_SAVE_FORMAT)).lower()
        idx_fmt = max(0, self.cb_audio_fmt.findText(cur_fmt))
        self.cb_audio_fmt.setCurrentIndex(idx_fmt)

        # 기본 프레임/분할
        self.sb_d_chunk = self._spin(60, 5000, int(S.DEFAULT_CHUNK), " /chunk")
        self.sb_d_overlap = self._spin(1, 120, int(S.DEFAULT_OVERLAP), " overlap")
        self.sb_d_infps = self._spin(1, 240, int(S.DEFAULT_INPUT_FPS), " inFPS")
        self.sb_d_outfps = self._spin(1, 240, int(S.DEFAULT_TARGET_FPS), " outFPS")

        # 프롬프트/워크플로 파일 경로
        self.le_prompt_json = QtWidgets.QLineEdit(str(S.ACE_STEP_PROMPT_JSON))
        btn_pick_prompt = QtWidgets.QPushButton("파일 선택")

        def _pick_prompt():
            f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "ACE_STEP_PROMPT_JSON", str(S.JSONS_DIR),
                                                         "JSON (*.json)")
            if f: self.le_prompt_json.setText(f)

        btn_pick_prompt.clicked.connect(_pick_prompt)
        pj_wrap = QtWidgets.QHBoxLayout()
        pj_wrap.addWidget(self.le_prompt_json)
        pj_wrap.addWidget(btn_pick_prompt)
        pj_widget = QtWidgets.QWidget()
        pj_widget.setLayout(pj_wrap)

        self.le_i2v = QtWidgets.QLineEdit(str(S.I2V_WORKFLOW))
        btn_pick_i2v = QtWidgets.QPushButton("파일 선택")

        def _pick_i2v():
            f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "I2V_WORKFLOW", str(S.JSONS_DIR), "JSON (*.json)")
            if f: self.le_i2v.setText(f)

        btn_pick_i2v.clicked.connect(_pick_i2v)
        i2v_wrap = QtWidgets.QHBoxLayout()
        i2v_wrap.addWidget(self.le_i2v)
        i2v_wrap.addWidget(btn_pick_i2v)
        i2v_widget = QtWidgets.QWidget()
        i2v_widget.setLayout(i2v_wrap)

        # 폼 배치
        form.addRow("BASE_DIR", base_widget)
        form.addRow("COMFY_HOST", self.le_comfy)
        form.addRow("HOST 후보(줄바꿈 구분)", self.te_candidates)
        form.addRow("FFMPEG_EXE", self.le_ffmpeg)
        form.addRow("USE_HWACCEL", self.cb_hwaccel)
        form.addRow("FINAL_OUT", self.le_final)
        form.addRow("AUDIO_SAVE_FORMAT", self.cb_audio_fmt)
        form.addRow("DEFAULT_CHUNK", self.sb_d_chunk)
        form.addRow("DEFAULT_OVERLAP", self.sb_d_overlap)
        form.addRow("DEFAULT_INPUT_FPS", self.sb_d_infps)
        form.addRow("DEFAULT_TARGET_FPS", self.sb_d_outfps)
        form.addRow("ACE_STEP_PROMPT_JSON", pj_widget)
        form.addRow("I2V_WORKFLOW", i2v_widget)

        # 저장/적용 버튼
        btn_save = QtWidgets.QPushButton("저장(오버라이드)")
        btn_apply = QtWidgets.QPushButton("저장 후 즉시 적용")

        def _collect_overrides():
            cand = [s.strip() for s in self.te_candidates.toPlainText().splitlines() if s.strip()]
            return dict(
                BASE_DIR=self.le_base_dir.text().strip(),
                COMFY_HOST=self.le_comfy.text().strip(),
                DEFAULT_HOST_CANDIDATES=cand,
                FFMPEG_EXE=self.le_ffmpeg.text().strip(),
                USE_HWACCEL=bool(self.cb_hwaccel.isChecked()),
                FINAL_OUT=self.le_final.text().strip(),
                AUDIO_SAVE_FORMAT=self.cb_audio_fmt.currentText().strip().lower(),
                DEFAULT_CHUNK=int(self.sb_d_chunk.value()),
                DEFAULT_OVERLAP=int(self.sb_d_overlap.value()),
                DEFAULT_INPUT_FPS=int(self.sb_d_infps.value()),
                DEFAULT_TARGET_FPS=int(self.sb_d_outfps.value()),
                ACE_STEP_PROMPT_JSON=self.le_prompt_json.text().strip(),
                I2V_WORKFLOW=self.le_i2v.text().strip(),
            )

        def _do_save():
            path = S.save_overrides(**_collect_overrides())
            QtWidgets.QMessageBox.information(self, "저장 완료", f"settings_local.json 저장됨\n\n{path}")

        def _do_apply():
            """
            설정을 settings_local.json에 저장하고, 런타임 settings 모듈 값도 즉시 반영.
            그리고 현재 포맷/제목/경로에 맞게 ComfyUI 워크플로의 SaveAudio 노드들을
            프로젝트 오디오 폴더(템플릿 [title] 치환)에 저장하도록 패치한다.
            """
            # 1) 파일로 저장 (settings_local.json)
            overrides = _collect_overrides()
            save_overrides(**overrides)

            # 2) 런타임(모듈 settings)에도 즉시 반영
            try:
                import settings as _S
            except Exception:
                from app import settings as _S  # type: ignore

            _S.BASE_DIR = overrides.get("BASE_DIR", _S.BASE_DIR)
            _S.COMFY_HOST = overrides.get("COMFY_HOST", _S.COMFY_HOST)
            _S.DEFAULT_HOST_CANDIDATES = overrides.get("DEFAULT_HOST_CANDIDATES",
                                                       getattr(_S, "DEFAULT_HOST_CANDIDATES", []))
            _S.FFMPEG_EXE = overrides.get("FFMPEG_EXE", _S.FFMPEG_EXE)
            _S.USE_HWACCEL = bool(overrides.get("USE_HWACCEL", getattr(_S, "USE_HWACCEL", False)))
            _S.FINAL_OUT = overrides.get("FINAL_OUT", _S.FINAL_OUT)
            _S.AUDIO_SAVE_FORMAT = overrides.get("AUDIO_SAVE_FORMAT", getattr(_S, "AUDIO_SAVE_FORMAT", "mp3")).lower()
            _S.DEFAULT_CHUNK = int(overrides.get("DEFAULT_CHUNK", getattr(_S, "DEFAULT_CHUNK", 600)))
            _S.DEFAULT_OVERLAP = int(overrides.get("DEFAULT_OVERLAP", getattr(_S, "DEFAULT_OVERLAP", 12)))
            _S.DEFAULT_INPUT_FPS = int(overrides.get("DEFAULT_INPUT_FPS", getattr(_S, "DEFAULT_INPUT_FPS", 24)))
            _S.DEFAULT_TARGET_FPS = int(overrides.get("DEFAULT_TARGET_FPS", getattr(_S, "DEFAULT_TARGET_FPS", 24)))
            _S.ACE_STEP_PROMPT_JSON = overrides.get("ACE_STEP_PROMPT_JSON", _S.ACE_STEP_PROMPT_JSON)
            _S.I2V_WORKFLOW = overrides.get("I2V_WORKFLOW", _S.I2V_WORKFLOW)

            # 3) 워크플로 JSON 저장 노드 포맷/경로 패치
            try:
                # 현재 UI 기준 값
                fmt = self.cb_audio_fmt.currentText().strip().lower()
                json_path = Path(self.le_prompt_json.text().strip())

                # 현재 제목 확보(없으면 로드된 프로젝트의 project.json에서 시도)
                title = (self.le_title.text() or "").strip()
                if not title:
                    pdir = self._latest_project()
                    if pdir and (pdir / "project.json").exists():
                        try:
                            from utils import load_json
                            meta = load_json(pdir / "project.json", {}) or {}
                            title = (meta.get("title") or "").strip()
                        except Exception:
                            title = ""

                # 오디오 저장 폴더(템플릿 [title] 치환) 계산
                # 예: FINAL_OUT = r"C:\my_games\shorts_make\maked_title\[title]"
                save_root = _S.FINAL_OUT or str(_S.BASE_DIR)
                proj_audio_dir = _resolve_audio_dir_from_template(save_root, title or "untitled")

                # 워크플로 저장 노드 패치 실행
                changed_fmt = rewrite_prompt_audio_format(json_path=json_path, desired_fmt=fmt)
                changed = bool(changed_fmt)

                QtWidgets.QMessageBox.information(
                    self,
                    "워크플로 수정",
                    f"{'변경 적용됨' if changed else '변경 없음'}:\n{json_path}\n→ {proj_audio_dir}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "워크플로 수정 실패", str(e))

            # 4) 완료 안내
            QtWidgets.QMessageBox.information(self, "적용 완료", "저장하고 런타임 값에도 적용했습니다.")

        btn_save.clicked.connect(_do_save)
        btn_apply.clicked.connect(_do_apply)

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(btn_save)
        btns.addWidget(btn_apply)

        wrap = QtWidgets.QVBoxLayout()
        wrap.addLayout(form)
        wrap.addSpacing(8)
        wrap.addLayout(btns)
        tab.setLayout(wrap)
        return tab



    # ────────────── 이벤트/핸들러 묶기 ──────────────
    def _wire(self):
        if getattr(self, "_signals_wired", False):
            return
        self._signals_wired = True
        # ↓↓↓ 기존 connect 들 그대로 ↓↓↓
        self.btn_save.clicked.connect(self.on_save_project)
        self.btn_load_proj.clicked.connect(self.on_load_project)
        self.btn_show_progress.clicked.connect(self.on_show_progress)
        self.btn_video.clicked.connect(self.on_video)


        # self.btn_test2_1_img.clicked.connect(self.on_generate_missing_images)



        # 라디오 ↔ JSON 즉시 동기화 (toggled True일 때만 저장)
        self.rb_20s.toggled.connect(lambda on: on and self._on_seconds_changed(20))
        self.rb_1m.toggled.connect(lambda on: on and self._on_seconds_changed(60))
        self.rb_2m.toggled.connect(lambda on: on and self._on_seconds_changed(120))
        self.rb_3m.toggled.connect(lambda on: on and self._on_seconds_changed(180))

        # 테스트
        self.btn_test2_render.clicked.connect(self.on_test2_render_story)
        self.btn_test3_concat.clicked.connect(self.on_test3_concat_segments)

    # ────────────── 토글/태그 유틸 ──────────────
    def _toggle_manual_tag_widgets(self, enabled: bool):
        # Basic_Vocal은 자동 ON이면 항상 포함 → 체크만 유지, 수정은 자동 OFF 때만 허용
        for cb in self.cb_basic_vocal_list:
            cb.setEnabled(enabled)
        # 수동 카테고리
        for lst in (self.cb_style_checks, self.cb_scene_checks, self.cb_instr_checks, self.cb_tempo_checks):
            for cb in lst:
                cb.setEnabled(enabled)

    def _apply_auto_tags_to_ui(self, tags: List[str]):
        """
        자동 추천 태그를 UI에 반영하되,
        이미 사용자가 체크해 둔 수동 선택은 절대 해제하지 않는다(OR 방식).
        """
        tagset = set((t or "").lower() for t in tags or [])

        # Basic Vocal: 추천이면 체크만 추가, 이미 체크된 건 유지, 추천에 없다고 해제하지 않음
        for cb in self.cb_basic_vocal_list:
            if cb.text().lower() in tagset:
                cb.setChecked(True)  # OR

        # 수동 카테고리들도 동일하게 OR 적용
        for lst in (self.cb_style_checks, self.cb_scene_checks, self.cb_instr_checks, self.cb_tempo_checks):
            for cb in lst:
                if cb.text().lower() in tagset:
                    cb.setChecked(True)

        # 성별 라디오는 ‘추천이 있으면 세팅’, 추천이 없다고 해서 해제하지 않음
        if "soft female voice" in tagset:
            self.rb_vocal_female.setChecked(True)
        elif "soft male voice" in tagset:
            self.rb_vocal_male.setChecked(True)
        elif "mixed vocals" in tagset:
            self.rb_vocal_mixed.setChecked(True)

    def _collect_manual_checked_tags(self) -> List[str]:
        picked: List[str] = []
        for cb in self.cb_basic_vocal_list:
            if cb.isChecked():
                picked.append(cb.text())
        for lst in (self.cb_style_checks, self.cb_scene_checks, self.cb_instr_checks, self.cb_tempo_checks):
            for cb in lst:
                if cb.isChecked():
                    picked.append(cb.text())
        # 성별은 배타
        if self.rb_vocal_female.isChecked(): picked.append("soft female voice")
        elif self.rb_vocal_male.isChecked(): picked.append("soft male voice")
        elif self.rb_vocal_mixed.isChecked(): picked.append("mixed vocals")
        return picked

    def _collect_gender(self) -> str:
        if self.rb_vocal_female.isChecked(): return "female"
        if self.rb_vocal_male.isChecked():   return "male"
        if self.rb_vocal_mixed.isChecked():  return "mixed"
        return "female"

    def _on_auto_toggle(self, auto_on: bool):
        # 자동이면 Basic_Vocal은 항상 체크 상태로(보장)
        for cb in self.cb_basic_vocal_list:
            cb.setChecked(True)
        # 수동 UI enable/disable
        self._toggle_manual_tag_widgets(not auto_on)

    # --- class MainWindow(...) 내부 ---

    def _current_seconds(self) -> int:
        """
        현재 UI 상태에서 '생성 길이(초)'를 결정한다.
        우선순위:
          1) 변환 토글이 ON이면 → '직접' 입력칸(self.le_len_seconds)의 숫자
          2) '직접' 라디오가 체크되면 → self.le_len_seconds 숫자
          3) 30초/1분/2분/3분 라디오
          4) 그 외 60초
        """
        # 1) 변환 토글 ON → 입력칸 우선
        btn_conv = getattr(self, "btn_convert_toggle", None)
        if btn_conv is not None and getattr(btn_conv, "isChecked", lambda: False)():
            le = getattr(self, "le_len_seconds", None)
            if le is not None and hasattr(le, "text"):
                try:
                    txt = (le.text() or "").strip()
                    if txt:
                        secs = int("".join(ch for ch in txt if ch.isdigit()))
                        if secs > 0:
                            return secs
                except (ValueError, TypeError):
                    pass  # 폴백 아래 진행

        # 2) '직접' 라디오 선택 시 → 입력칸
        rb_direct = getattr(self, "rb_len_direct", None)
        if rb_direct is not None and getattr(rb_direct, "isChecked", lambda: False)():
            le = getattr(self, "le_len_seconds", None)
            if le is not None and hasattr(le, "text"):
                try:
                    txt = (le.text() or "").strip()
                    if txt:
                        secs = int("".join(ch for ch in txt if ch.isdigit()))
                        if secs > 0:
                            return secs
                except (ValueError, TypeError):
                    return 60
            return 60

        # 3) 프리셋 라디오(30/60/120/180)
        rb_30 = getattr(self, "rb_30s", None) or getattr(self, "rb_20s", None) or getattr(self, "rb_time_20s", None)
        if rb_30 is not None and getattr(rb_30, "isChecked", lambda: False)():
            return 30

        rb_1m = getattr(self, "rb_1m", None)
        if rb_1m is not None and getattr(rb_1m, "isChecked", lambda: False)():
            return 60

        rb_2m = getattr(self, "rb_2m", None)
        if rb_2m is not None and getattr(rb_2m, "isChecked", lambda: False)():
            return 120

        rb_3m = getattr(self, "rb_3m", None)
        if rb_3m is not None and getattr(rb_3m, "isChecked", lambda: False)():
            return 180

        # 4) 최후 폴백
        return 60

    def _gather_tag_state(self) -> dict:



        auto_on = bool(self.cb_auto_tags.isChecked())
        if self.rb_vocal_female.isChecked():   gender = "female"
        elif self.rb_vocal_male.isChecked():   gender = "male"
        else:                                   gender = "mixed"

        def _collect_manual_checked() -> List[str]:
            picked: List[str] = []
            for cb in self.cb_basic_vocal_list:
                if cb.isChecked(): picked.append(cb.text())
            for group in (self.cb_style_checks, self.cb_scene_checks, self.cb_instr_checks, self.cb_tempo_checks):
                for cb in group:
                    if cb.isChecked(): picked.append(cb.text())
            return picked

        state: dict[str, Any] = {
            "auto_tags": auto_on,
            "vocal_gender": gender,
            "time": int(self._current_seconds()),
        }

        if auto_on:
            state["ace_tags"] = list(self._last_tags or [])
            state["tags_in_use"] = _collect_manual_checked()
        else:
            state["manual_tags"] = _collect_manual_checked()
        return state

    def _apply_time_from_project_json(self) -> None:
        from pathlib import Path
        from utils import load_json

        proj = getattr(self, "project_dir", "") or ""
        if not proj:
            return

        pj = Path(proj) / "project.json"
        meta = load_json(pj, {}) or {}
        secs = int(meta.get("target_seconds") or meta.get("time") or 60)

        # 30초로 저장된 케이스
        if secs <= 40:  # 30초 근방이면 30초 라디오
            if hasattr(self, "rb_30s"):
                self.rb_30s.setChecked(True)
            elif hasattr(self, "rb_20s"):
                self.rb_20s.setChecked(True)  # 과거 UI 호환
            return

        if secs < 90 and hasattr(self, "rb_1m"):
            self.rb_1m.setChecked(True)
        elif secs < 150 and hasattr(self, "rb_2m"):
            self.rb_2m.setChecked(True)
        elif hasattr(self, "rb_3m"):
            self.rb_3m.setChecked(True)

    # shorts_ui.py

    def _on_seconds_changed(self, *_args) -> None:
        """
        라디오/체크 변경 시 project.json의 time/target_seconds(초)를 즉시 갱신.
        신호 인자(bool 등)를 받아도 무시하도록 *_args로 받는다.
        """
        from pathlib import Path
        try:
            from utils import load_json, save_json
        except Exception:
            from app.utils import load_json, save_json

        proj = getattr(self, "project_dir", "") or ""
        if not proj:
            return

        pj = Path(proj) / "project.json"
        meta = load_json(pj, {}) or {}

        secs = int(self._current_seconds())  # ← 라디오 상태에서 현재 초값을 계산
        meta["time"] = secs
        meta["target_seconds"] = secs

        save_json(pj, meta)
        print("[TIME] radio changed ->", secs, "saved to", pj, flush=True)
        if hasattr(self, "status"):
            self.status.showMessage(f"길이 설정: {secs}초")

    def _bind_length_radios(self) -> None:
        # 30초(테스트/구 20초 라벨 포함)
        if hasattr(self, "rb_30s"):
            self.rb_30s.clicked.connect(self._on_seconds_changed)
        if hasattr(self, "rb_20s"):
            self.rb_20s.clicked.connect(self._on_seconds_changed)
        if hasattr(self, "rb_time_20s"):
            self.rb_time_20s.clicked.connect(self._on_seconds_changed)

        # 1/2/3분
        if hasattr(self, "rb_1m"):
            self.rb_1m.clicked.connect(self._on_seconds_changed)
        if hasattr(self, "rb_2m"):
            self.rb_2m.clicked.connect(self._on_seconds_changed)
        if hasattr(self, "rb_3m"):
            self.rb_3m.clicked.connect(self._on_seconds_changed)

        # ✅ '직접' 라디오 및 입력칸도 저장 로직에 연결
        rb_direct = getattr(self, "rb_len_direct", None)
        if rb_direct is not None and hasattr(rb_direct, "toggled"):
            rb_direct.toggled.connect(self._on_seconds_changed)

        le = getattr(self, "le_len_seconds", None)
        if le is not None and hasattr(le, "textChanged"):
            def _on_le_changed(_text: str) -> None:
                # '직접' 라디오가 켜져 있거나 변환 토글이 ON일 때만 저장
                use = False
                r = getattr(self, "rb_len_direct", None)
                if r is not None and getattr(r, "isChecked", lambda: False)():
                    use = True
                btn = getattr(self, "btn_convert_toggle", None)
                if btn is not None and getattr(btn, "isChecked", lambda: False)():
                    use = True
                if use:
                    self._on_seconds_changed()

            le.textChanged.connect(_on_le_changed)

    def _save_project_snapshot(self, title: str, lyrics: str, prompt: str) -> str:
        """
        제목/가사/프롬프트로 새 프로젝트 폴더를 만들고 project.json을 저장한 뒤
        그 폴더를 '활성' 프로젝트로 전환한다. 반환: 생성한 프로젝트 폴더 경로
        """
        from pathlib import Path
        import time

        # utils
        try:
            from app.utils import save_json, load_json, sanitize_title
        except Exception:
            from utils import save_json, load_json, sanitize_title  # type: ignore

        title = (title or "").strip() or "untitled"
        lyrics = (lyrics or "").strip()
        prompt = (prompt or "").strip()

        # 1) 폴더 생성
        try:
            # 프로젝트 생성 유틸이 있으면 사용
            proj_dir = Path(create_project_files(title, lyrics, prompt))
        except Exception:
            # 없으면 수동 생성
            root = Path(getattr(self, "projects_root", "C:/my_games/shorts_make/maked_title"))
            root.mkdir(parents=True, exist_ok=True)
            safe = sanitize_title(title) if 'sanitize_title' in globals() else title
            proj_dir = root / safe
            proj_dir.mkdir(parents=True, exist_ok=True)
            # 최소 산출물
            with open(proj_dir / "lyrics.txt", "w", encoding="utf-8") as f:
                f.write(lyrics)

        # 2) project.json 병합 저장
        pj = proj_dir / "project.json"
        meta = load_json(pj, {}) or {}
        meta["title"] = title
        meta["lyrics"] = lyrics
        meta["prompt_user"] = prompt
        # 길이(초) → 현재 라디오에서 읽어서 저장
        secs = int(self._current_seconds()) if hasattr(self, "_current_seconds") else 60
        meta["time"] = secs
        meta["target_seconds"] = secs
        meta.setdefault("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
        save_json(pj, meta)

        # 3) 활성화 전환 + UI 갱신
        self._set_active_project_dir(str(proj_dir))
        try:
            # 제목/가사/태그/라디오까지 UI 복원
            self._apply_project_meta(str(proj_dir))
        except Exception:
            pass

        print("[SNAPSHOT] saved:", str(pj), flush=True)
        return str(proj_dir)

    # ────────────── 가사 생성 ──────────────
    # app/shorts_ui.py

    def _duration_minutes(self) -> float:
        """
        가사 생성용 '분' 단위 힌트.
        30초(테스트)는 0.5분으로 매핑.
        """
        try:
            # 30초(테스트)
            if hasattr(self, "rb_20s") and self.rb_20s.isChecked():
                return 0.5  # ✅ 0.33 -> 0.5
            if hasattr(self, "rb_time_20s") and self.rb_time_20s.isChecked():
                return 0.5  # ✅ 0.33 -> 0.5

            # 1/2/3분
            if hasattr(self, "rb_1m") and self.rb_1m.isChecked():
                return 1.0
            if hasattr(self, "rb_2m") and self.rb_2m.isChecked():
                return 2.0
            if hasattr(self, "rb_3m") and self.rb_3m.isChecked():
                return 3.0
        except Exception:
            pass
        return 2.0

    # ─────────────────────────────────────────────────────────

        # shorts_ui.py 파일의 on_generate_lyrics 함수를 아래 코드로 덮어쓰세요.







    def _manual_option_set(self) -> set[str]:
        """수동 체크박스에 표시되는 '모든 후보 텍스트'(소문자)를 집합으로 반환"""
        s: set[str] = set()
        for cb in self.cb_basic_vocal_list:
            s.add(cb.text().lower().strip())
        for group in (self.cb_style_checks, self.cb_scene_checks, self.cb_instr_checks, self.cb_tempo_checks):
            for cb in group:
                s.add(cb.text().lower().strip())
        return s


    def canon_key(s: str) -> str:
        return _CANON_RE.sub("", (s or "").lower())

    def _manual_option_map(self) -> dict[str, str]:
        """수동 체크박스의 '정식 라벨'을 캐논키로 매핑."""
        d: dict[str, str] = {}
        for cb in self.cb_basic_vocal_list:
            d[self._canon_key(cb.text())] = cb.text()
        for group in (self.cb_style_checks, self.cb_scene_checks, self.cb_instr_checks, self.cb_tempo_checks):
            for cb in group:
                d[self._canon_key(cb.text())] = cb.text()
        return d

    def _pick_allowed_tags(self, auto_tags: list[str]) -> list[str]:
        """
        auto_tags → 수동 후보에 존재하는 값만(원 라벨로) 선택.
        - 1차: 캐논키 완전일치
        - 2차: 부분일치(양방향 substring)
        - 3차: 간단 동의어 매핑
        """
        opts = self._manual_option_map()  # canon -> label
        picked: list[str] = []
        seen: set[str] = set()

        # 간단 동의어(필요시 추가)
        syn: dict[str, str] = {
            "electronicdancemusic": "edm",
            "edm": "edm",
            "partybackground": "background music for parties",
            "workout": "workout playlists",
            "gymmusic": "workout playlists",
            "electronicdrums": "soft electric drums",
            "softelectronicdrums": "soft electric drums",
            "soulful": "soul",
            "funky": "funk",
        }
        # 동의어를 캐논키 기반으로 역매핑(label→canon)
        syn_canon: dict[str, str] = {}
        for k, v in syn.items():
            syn_canon[self._canon_key(k)] = self._canon_key(v)

        # 1차 & 2차 & 동의어
        for t in (auto_tags or []):
            k = self._canon_key(t)
            label: str | None = None

            # 동의어 치환
            if k in syn_canon:
                k = syn_canon[k]

            # 완전일치
            if k in opts:
                label = opts[k]
            else:
                # 부분일치(양방향)
                for ok, lbl in opts.items():
                    if k and (k in ok or ok in k):
                        label = lbl
                        break

            if label and label not in seen:
                picked.append(label)
                seen.add(label)

        return picked

    # 테스트
    # ==== [도우미] 공통 경로/FFmpeg/파일 유틸 ==================================


    # shorts_ui.py 내부, 클래스(MainWindow 등) 메서드로 추가
    def _apply_lyrics_result(self, data: dict, manual_title: str, prompt: str) -> None:
        """
        가사 생성 결과를 UI에 반영 + 프로젝트 저장(project.json 생성/갱신) + 디버그 로그.
        - 제목/가사 UI 채움
        - 자동/수동 태그 반영(OR 체크)
        - 현재 선택된 길이(초)를 project.json 에 저장: time, target_seconds 모두 '초'로 통일
        - 생성 폴더를 활성 프로젝트로 설정(시그니처 차이를 안전 호출)
        """
        from pathlib import Path

        # 유틸 import (app 패키지/루트 양쪽 지원)
        try:
            from app.utils import load_json, save_json  # type: ignore
        except Exception:
            from utils import load_json, save_json  # type: ignore
        try:
            from app.tag_norm import normalize_tags_to_english  # type: ignore
        except Exception:
            from tag_norm import normalize_tags_to_english  # type: ignore
        try:
            from app.lyrics_gen import create_project_files  # type: ignore
        except Exception:
            from lyrics_gen import create_project_files  # type: ignore

        # 1) 결과 꺼내기
        title = (data.get("title") or manual_title or "untitled").strip()
        lyrics = (data.get("lyrics") or data.get("lyrics_ko") or "").strip()

        # tags(list/str 모두 지원 → list[str]로 정규화)
        raw = data.get("tags", [])
        if isinstance(raw, list):
            tags = [t for t in raw if str(t).strip()]
        else:
            tags = [s.strip() for s in str(raw).split(",") if s.strip()]

        # 2) UI 채우기 (제목/가사)
        if hasattr(self, "le_title"):
            self.le_title.setText(title)
        if hasattr(self, "te_lyrics"):
            self.te_lyrics.setPlainText(lyrics)

        # 3) 자동 태그(영문 정규화) + 수동 후보 안에서 picks 선택
        try:
            auto_en = normalize_tags_to_english(tags)  # 전체 후보(영문)
        except Exception:
            auto_en = [str(t) for t in tags]

        picks_from_ai = data.get("tags_pick") or []

        # _manual_option_set 이 함수이거나, 이미 Iterable일 수도 있으므로 모두 지원
        allowed = []
        getter = getattr(self, "_manual_option_set", None)
        if getter is not None:
            try:
                value = getter() if callable(getter) else getter
                from collections.abc import Iterable
                if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                    allowed = sorted({str(x).strip() for x in value if str(x).strip()})
                else:
                    allowed = []
            except Exception:
                allowed = []

        if allowed:
            allow_set = {t.lower() for t in allowed}
            picks = [t for t in (picks_from_ai or auto_en) if t.lower() in allow_set][:10]
        else:
            picks = (picks_from_ai or auto_en)[:10]

        print("[TAGDBG] auto_en:", auto_en, flush=True)
        print("[TAGDBG] manual options:", allowed, flush=True)
        print("[TAGDBG] picks(final):", picks, flush=True)

        try:
            if hasattr(self, "_apply_auto_tags_to_ui"):
                self._apply_auto_tags_to_ui(picks)
        except Exception as e:
            print("[TAGDBG] apply checks fail:", type(e).__name__, str(e), flush=True)

        if hasattr(self, "cb_auto_tags"):
            if self.cb_auto_tags.isChecked():
                if hasattr(self, "_toggle_manual_tag_widgets"):
                    self._toggle_manual_tag_widgets(False)
            else:
                if hasattr(self, "_toggle_manual_tag_widgets"):
                    self._toggle_manual_tag_widgets(True)

        # 4) 프로젝트 생성 + project.json 갱신
        seconds = 60
        try:
            if hasattr(self, "_current_seconds"):
                seconds = int(self._current_seconds())
        except Exception:
            seconds = 60

        pdir_str = create_project_files(title, lyrics, prompt)
        pdir = Path(pdir_str)
        pj = pdir / "project.json"
        meta = load_json(pj, {}) or {}

        meta["target_seconds"] = int(seconds)
        meta["time"] = int(seconds)
        meta["title"] = title
        meta["lyrics"] = lyrics
        meta["prompt_user"] = str(prompt or "")

        if hasattr(self, "cb_auto_tags") and self.cb_auto_tags.isChecked():
            meta["auto_tags"] = True
            meta["ace_tags"] = list(auto_en)
            meta["tags_in_use"] = list(picks)
            meta.pop("manual_tags", None)
        else:
            meta["auto_tags"] = False
            manual_checked = []
            try:
                if hasattr(self, "_collect_manual_checked_tags"):
                    manual_checked = list(self._collect_manual_checked_tags())
            except Exception:
                manual_checked = []
            meta["manual_tags"] = manual_checked
            meta.pop("tags_in_use", None)
            meta.pop("ace_tags", None)

        save_json(pj, meta)

        print("[TAGDBG] saved auto_tags:", meta.get("auto_tags"), flush=True)
        print("[TAGDBG] saved ace_tags len:", len(meta.get("ace_tags", [])), flush=True)
        print("[TAGDBG] saved tags_in_use len:", len(meta.get("tags_in_use", [])), flush=True)
        print("[TAGDBG] saved manual_tags len:", len(meta.get("manual_tags", [])), flush=True)
        print(
            "[TAGDBG] saved seconds:", meta.get("time"),
            "(target_seconds:", meta.get("target_seconds"), ")",
            flush=True
        )

        # 5) 활성 프로젝트로 전환 — 시그니처 차이에 안전하게 대응
        if hasattr(self, "_set_active_project_dir") and callable(self._set_active_project_dir):
            import inspect
            fn = self._set_active_project_dir
            try:
                sig = inspect.signature(fn)
                # bound method이면 self는 이미 바인딩됨 → '필수 인자' 개수로 판단
                required = [
                    p for p in sig.parameters.values()
                    if p.default is inspect._empty
                       and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                ]
                if len(required) == 0:
                    fn()
                else:
                    fn(str(pdir))
            except TypeError:
                # 어떤 형태든 실패하면 최후 보루
                self.project_dir = str(pdir)
        else:
            self.project_dir = str(pdir)

        # 6) 길이 라디오/상태바 갱신
        try:
            if hasattr(self, "_apply_time_from_project_json"):
                self._apply_time_from_project_json()
        except Exception:
            pass

        if hasattr(self, "status"):
            try:
                self.status.showMessage("가사 생성 완료")
            except Exception:
                pass

    from pathlib import Path
    from typing import Union

    def _apply_project_meta(self, project: Union[str, Path, dict], *, project_dir: str | Path | None = None) -> None:
        """
        project.json 메타를 UI에 복원.
        - project: '프로젝트 폴더 경로' 또는 'meta dict'
        - project_dir: project가 dict일 때 project.json이 있는 폴더를 알려주면 정확함
        """
        meta = None
        pdir: Path | None = None

        # 1) 입력 해석
        if isinstance(project, (str, Path)):
            pdir = Path(project)
            pj = pdir / "project.json"
            if not pj.is_file():
                print("[LOADDBG] project.json not found:", str(pj), flush=True)
                return
            try:
                meta = json.loads(pj.read_text(encoding="utf-8")) or {}
            except Exception as e:
                print("[LOADDBG] project.json read fail:", type(e).__name__, str(e), flush=True)
                return
        elif isinstance(project, dict):
            meta = project
            if project_dir is not None:
                pdir = Path(project_dir)
            else:
                # 힌트가 없으면 추정 시도 (최신 프로젝트/내부 경로 등)
                try:
                    if hasattr(self, "_latest_project") and self._latest_project():
                        pdir = Path(str(self._latest_project()))
                except Exception:
                    pdir = None
        else:
            print("[LOADDBG] invalid project arg type:", type(project).__name__, flush=True)
            return

        if meta is None:
            print("[LOADDBG] no meta resolved", flush=True)
            return

        # 2) 메타 필드 추출
        title = (meta.get("title") or "").strip()
        lyrics = (meta.get("lyrics") or "").strip()
        auto_tags = bool(meta.get("auto_tags", True))
        tags_pick = meta.get("tags_in_use") or meta.get("tags_pick") or meta.get("tags") or []
        manual = meta.get("manual_tags") or []

        print("[LOADDBG] === APPLY PROJECT META ===", flush=True)
        if pdir:
            print("[LOADDBG] project_dir:", str(pdir), flush=True)
        print("[LOADDBG] title:", title, flush=True)
        print("[LOADDBG] lyrics_len:", len(lyrics), flush=True)
        print("[LOADDBG] auto_tags:", auto_tags, flush=True)
        print("[LOADDBG] tags_pick len:", len(tags_pick), flush=True)
        print("[LOADDBG] manual_tags len:", len(manual), flush=True)

        # 3) 텍스트 필드 복원
        if hasattr(self, "le_title"):
            self.le_title.setText(title if title else "")
        if hasattr(self, "te_lyrics"):
            if hasattr(self.te_lyrics, "setPlainText"):
                self.te_lyrics.setPlainText(lyrics if lyrics else "")
            else:
                self.te_lyrics.setText(lyrics if lyrics else "")

        # 4) 자동/수동 토글(먼저)
        if hasattr(self, "cb_auto_tags"):
            self.cb_auto_tags.setChecked(auto_tags)

        # 5) 체크박스 반영
        try:
            if auto_tags:
                self._apply_auto_tags_to_ui(list(tags_pick))
            else:
                self._apply_auto_tags_to_ui(list(manual))
        except Exception as e:
            print("[LOADDBG] set checks fail:", type(e).__name__, str(e), flush=True)

        # 6) 수동 패널 활성/비활성
        try:
            self._toggle_manual_tag_widgets(not auto_tags)
        except Exception:
            pass

        # 7) 길이/초 라디오/슬라이더 복원(있으면)
        try:
            self._apply_time_from_project_json()
        except Exception:
            pass

        if hasattr(self, "status"):
            self.status.showMessage("프로젝트 불러오기 완료")

    # ────────────── 저장/불러오기 ──────────────
    def on_save_project(self):
        title = sanitize_title(self.le_title.text())
        lyrics = self.te_lyrics.toPlainText().strip()
        if not title or not lyrics:
            QtWidgets.QMessageBox.warning(self, "안내", "제목/가사를 확인하세요.")
            return

        # 이미 내부에서 ensure_project_dir(title)을 호출함 → 중복 방지됨
        proj_path_str = create_project_files(title, lyrics, self.te_prompt.toPlainText())
        pdir = Path(proj_path_str)

        meta = load_json(pdir / "project.json", {}) or {}
        meta["title"] = title
        meta["lyrics"] = lyrics
        meta.setdefault("paths", {})["project_dir"] = str(pdir)

        meta["time"] = int(self._current_seconds())
        meta["auto_tags"] = bool(self.cb_auto_tags.isChecked())
        meta["vocal_gender"] = self._collect_gender()

        if meta["auto_tags"]:
            meta["ace_tags"] = list(self._last_tags or [])
            meta["tags_in_use"] = self._collect_manual_checked_tags()
        else:
            meta["manual_tags"] = self._collect_manual_checked_tags()

        save_json(pdir / "project.json", meta)
        QtWidgets.QMessageBox.information(self, "완료", f"저장: {pdir}")
        self.status.showMessage(f"저장: {pdir}")

    def _normalize_saveaudio_nodes(graph: dict, *, out_path: str, prefix: str, out_dir: str) -> None:
        """
        SaveAudio* 노드들의 입력을 현재 실행 컨텍스트에 맞게 통일:
          - filename: out_path (절대경로, 예: C:/.../vocal.wav)
          - filename_prefix: prefix (예: shorts_make/<title>/vocal_final)
          - output_path: out_dir (폴더)
          - basename/base_filename 등은 필요 시 'vocal'로 고정
        기존 템플릿에 남아있던 다른 프로젝트 경로를 안전하게 덮어쓴다.
        """
        nodes = graph.get("nodes") or []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            ct = str(node.get("class_type") or "")
            if not ct.lower().startswith("saveaudio"):
                continue
            ins = node.setdefault("inputs", {})
            ins["filename"] = out_path
            ins["filename_prefix"] = prefix
            ins["output_path"] = out_dir
            ins["basename"] = "vocal"
            ins["base_filename"] = "vocal"

    def _set_active_project_dir(self, path: str) -> None:
        """
        현재 작업 대상 프로젝트 폴더를 지정하고 상태바/타이틀을 갱신한다.
        """
        from pathlib import Path
        p = str(path or "").strip()
        if not p:
            return
        self.project_dir = p
        # 최근/강제 선택 힌트도 같이 갱신
        try:
            self._forced_project_dir = p
        except Exception:
            pass
        try:
            if hasattr(self, "statusbar"):
                self.statusbar.showMessage(f"프로젝트 활성화: {Path(p).name}")
            if hasattr(self, "setWindowTitle"):
                self.setWindowTitle(f"쇼츠 자동화 — {Path(p).name}")
        except Exception:
            pass

    def _get_active_project_dir(self) -> str:
        """
        음악생성 등에서 사용할 확정된 프로젝트 폴더를 돌려준다.
        1) self.project_dir → 2) self._forced_project_dir → 3) 최근 프로젝트
        """
        p = getattr(self, "project_dir", "") or getattr(self, "_forced_project_dir", "") or ""
        if p:
            return p
        try:
            lp = self._latest_project()
            if lp:
                return str(lp)
        except Exception:
            pass
        return ""

    def on_load_project(self) -> None:
        from PyQt5 import QtWidgets
        from pathlib import Path

        try:
            from app.utils import load_json
        except Exception:
            from utils import load_json  # type: ignore

        # 1) project.json 선택
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "project.json 선택",
            str(getattr(self, "project_dir", "") or ""),
            "JSON (*.json)"
        )
        if not path_str:
            return

        pj = Path(path_str)
        pdir = pj.parent

        # 2) 선택 즉시 '활성 프로젝트'로 전환
        self._set_active_project_dir(str(pdir))

        # 3) UI 복원(제목/가사/태그/길이 라디오까지) — 이 함수가 다 해줍니다
        try:
            self._apply_project_meta(str(pdir))
        except Exception:
            pass

        # 4) 상태 표시
        print("[LOAD-PROJ] activated:", str(pdir), flush=True)
        if hasattr(self, "statusbar"):
            self.statusbar.showMessage(f"불러옴: {pj}")

    # ────────────── 진행창/로그 ──────────────


    def on_show_progress(self):
        if self._dlg:
            self._dlg.show(); self._dlg.raise_(); self._dlg.activateWindow()

    # ────────────── 음악 생성 ──────────────



    def on_generate_missing_images(self):
        """
        누락 이미지 생성 (검사 기준: FINAL_OUT\[title]\imgs)
        - story.json: FINAL_OUT\[title]\story.json
        - imgs 폴더: FINAL_OUT\[title]\imgs
        - 장면 프롬프트: scene.prompt (없으면 _gpt_scene_prompt 로 생성)
        - 캐릭터 1명일 때만 swap 시도 (참조 이미지가 있을 때)
        """
        try:
            title = (self.le_title.text().strip()
                     or (load_json((self._latest_project() / "project.json"), {}) or {}).get("title", "")
                     or "untitled")

            # 기준 경로(반드시 FINAL_OUT)
            imgs_dir = self._img_dir_for_title(title)
            imgs_dir.mkdir(parents=True, exist_ok=True)
            story = self._read_story(title)  # FINAL_OUT\[title]\story.json 을 찾아서 scenes 스키마로 정규화
            scenes = story.get("scenes") or []
            if not scenes:
                QtWidgets.QMessageBox.warning(self, "안내", "story.json에 scenes가 없습니다.")
                return

            # 누락 스캔 (이미지 경로 우선순위: scene.img_file → FINAL_OUT\[title]\imgs\[id].png)
            def _resolve_img_path(scene_id: str, img_file_str: str | None) -> Path:
                if img_file_str:
                    # 'p' -> 'path_obj'로 변경하여 외부 변수와의 충돌 방지
                    path_obj = Path(img_file_str)
                    if path_obj.exists() and path_obj.stat().st_size > 0:
                        return path_obj
                # 'sid' -> 'scene_id'로 변경하여 외부 변수와의 충돌 방지
                return imgs_dir / f"{scene_id}.png"

            missing: list[tuple[str, dict, Path]] = []
            for sc in scenes:
                sid = sc.get("id") or sc.get("title") or f"t_{int(sc.get('idx', 0) or 0):02d}"
                p = _resolve_img_path(sid, sc.get("img_file"))
                if not (p.exists() and p.stat().st_size > 0):
                    missing.append((sid, sc, p))

            if not missing:
                QtWidgets.QMessageBox.information(
                    self, "누락 없음",
                    f"누락된 이미지가 없습니다.\n검사 폴더: {imgs_dir}"
                )
                return

            # 생성 진행
            prog = QtWidgets.QProgressDialog("누락 이미지 생성 중…", "중지", 0, len(missing), self)
            self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
            prog.setMinimumDuration(0)

            ok_cnt, fail_cnt = 0, 0
            for i, (sid, sc, out_png) in enumerate(missing, 1):
                if prog.wasCanceled():
                    break
                # 프롬프트
                prompt = (sc.get("prompt") or "").strip()
                if not prompt:
                    try:
                        prompt = self._gpt_scene_prompt(sc, story)
                    except Exception:
                        prompt = "감정/연출 중심, 카메라/분위기 묘사; 가사 금지"

                # 캐릭터 참조(단일 캐릭터일 때만 swap)
                char_ref = None
                chars = sc.get("characters") or []
                if isinstance(chars, list) and len(chars) == 1:
                    char_ref = self._find_character_asset_path(str(chars[0]))

                # 워크플로 선택: 참조 있으면 swap, 없으면 t2i
                wf = Path("nunchaku-t2i_swap.json") if char_ref else Path("nunchaku_t2i.json")

                prog.setLabelText(f"{sid} → 생성 중…")
                QtCore.QCoreApplication.processEvents()

                ok = self._comfy_run_workflow(wf_path=wf, prompt_text=prompt, out_png=out_png, char_ref=char_ref)
                if ok:
                    ok_cnt += 1
                    prog.setLabelText(f"{sid} → 완료")
                else:
                    fail_cnt += 1
                    prog.setLabelText(f"{sid} → 실패")

                prog.setValue(i)
                QtCore.QCoreApplication.processEvents()

            QtWidgets.QMessageBox.information(
                self, "완료",
                f"누락 이미지 생성 결과\n\n생성: {ok_cnt}  |  실패: {fail_cnt}\n폴더: {imgs_dir}"
            )
            self.status.showMessage(f"누락 이미지 생성 완료 — 생성 {ok_cnt}, 실패 {fail_cnt}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "오류", str(e))
            self.status.showMessage(f"오류: {e}")



    @staticmethod
    def _move_latest_vocal_to_project(title: str) -> Path | None:
        """
        C:\comfyResult\shorts_make\[title]\ 에서 가장 최신 vocal_final_*.mp3 파일을
        C:\my_games\shorts_make\maked_title\[title]\vocal.mp3 로 이동한다.
        """
        import settings as _S

        safe_title = _sanitize_title_for_path(title)

        # 원본 폴더
        src_dir = Path(getattr(_S, "COMFY_RESULT_ROOT", r"C:\comfyResult\shorts_make")) / safe_title
        if not src_dir.exists():
            return None

        # 목적지 폴더
        dst_dir = _resolve_audio_dir_from_template(_S.FINAL_OUT, title)
        dst_dir.mkdir(parents=True, exist_ok=True)

        # 최신 파일 찾기 (vocal_final_*.*)
        candidates = list(src_dir.glob("vocal_final_*.*"))
        if not candidates:
            return None
        latest = max(candidates, key=lambda p: p.stat().st_mtime)

        # 목적지 파일명 통일
        dst_file = dst_dir / f"vocal{latest.suffix.lower()}"

        try:
            shutil.move(str(latest), str(dst_file))
        except Exception:
            shutil.copyfile(str(latest), str(dst_file))
        return dst_file





    @QtCore.pyqtSlot(bool, str)
    def _on_music_done(self, success: bool, msg: str):
        # (기존 내용은 동일)
        if self._dlg:
            self._dlg.set_status("ACE-Step 완료 — 파일 정리 중…")
            self._dlg.append_log(f"ACE-Step 완료 (성공: {success}) ✅")
            if msg:
                self._dlg.append_log(msg)

        # 로그 테일 정리만 하고 창은 유지
        try:
            if self._log_timer:
                self._log_timer.stop()
                self._log_timer.deleteLater()
                self._log_timer = None
            if self._log_fp:
                self._log_fp.close()
                self._log_fp = None
        except Exception:
            pass

        # 결과 이동 및 요약
        title = self.le_title.text().strip()
        if not title:
            proj = self._latest_project()
            if proj and (proj / "project.json").exists():
                try:
                    meta = load_json(proj / "project.json", {}) or {}
                    title = meta.get("title", "")
                except Exception:
                    pass

        moved_paths: list[Path] = []
        try:
            if title:
                moved_paths.extend(self._move_generated_audio_to_target(title) or [])
                one = self._move_latest_vocal_to_project(title)
                if one: moved_paths.append(one)
        except Exception as ex:
            self.status.showMessage(f"파일 이동 실패: {ex}")
            if self._dlg:
                self._dlg.append_log(f"[MOVE] 실패: {ex}")

        if moved_paths:
            last = moved_paths[-1]
            self.status.showMessage(f"음악 생성 완료 — {last.name}")
            if self._dlg:
                self._dlg.append_log(f"[MOVE] 완료: {last}")
            try:
                self._set_total_frames_from_audio(last)
            except Exception:
                pass
        else:
            self.status.showMessage("음악 생성 완료 — 이동된 파일 없음")
            if self._dlg:
                self._dlg.append_log("[MOVE] 이동된 파일 없음")

        self._task_done("music")  # 음악 가드 해제

        # story.json 빌드 준비: project.json과 분석 대상 오디오 경로 선택
        audio_path_for_analysis: Path | None = None
        try:
            proj = self._latest_project()
            meta = load_json((proj / "project.json") if proj else "", {}) or {}
            vud = (meta.get("paths", {}) or {}).get("vocal_user_dir", "")
            if vud:
                audio_path_for_analysis = self._resolve_audio_for_analysis(Path(vud))
        except Exception:
            pass

        if not audio_path_for_analysis:
            try:
                proj = self._latest_project()
                audio_path_for_analysis = self._resolve_audio_for_analysis(
                    self._find_vocal_in_project(proj) if proj else None
                )
            except Exception:
                pass

        if (not audio_path_for_analysis) and moved_paths:
            audio_path_for_analysis = self._resolve_audio_for_analysis(moved_paths[-1])

        # 바로 story 빌드 (팝업 없음)
        proj_dir = audio_path_for_analysis.parent if audio_path_for_analysis else None  # ✅ 현재 폴더 고정
        pj = (proj_dir / "project.json") if proj_dir else None

        if audio_path_for_analysis and audio_path_for_analysis.exists() and pj and pj.exists():
            if self._dlg:
                self._dlg.set_status("음악 완료 → story.json 생성 시작…")
                self._dlg.append_log(f"[AUTO-STORY] 시작: {audio_path_for_analysis}")
            self._start_build_story_from_analysis(audio_path_for_analysis, pj)
        else:
            print("[AUTO-STORY] skip: audio or project.json missing", flush=True)

        # ======================================================================

    @QtCore.pyqtSlot()
    def _on_analysis_thread_finished(self):
        """분석 QThread가 어떤 이유로든 끝나면 버튼/플래그를 반드시 복구."""
        try:
            # 공통 가드/플래그 정리
            if hasattr(self, "_task_done"):
                self._task_done("analysis")
            else:
                # 구버전 호환
                self._analysis_running = False
                self._analysis_thread = None
            # UI 잠금 해제
            self._set_busy_ui("analysis", False)
        except Exception:
            pass

    def _start_analysis_watchdog(self):
        """2초마다 analysis 스레드 상태를 보고 스테일이면 UI 잠금 해제."""
        if getattr(self, "_analysis_watchdog", None):
            return
        t = QtCore.QTimer(self)
        t.setInterval(2000)
        t.timeout.connect(self._analysis_watchdog_tick)
        t.start()
        self._analysis_watchdog = t

    def _force_unlock_all_buttons(self):
        for name in (
                "btn_test1_story", "btn_test2_render", "btn_test3_concat",
                "btn_analyze", "btn_test1",
                "btn_gen", "btn_save"
        ):
            b = getattr(self, name, None)
            if b is not None:
                b.setEnabled(True)

    def _analysis_watchdog_tick(self):
        th = getattr(self, "_analysis_thread", None)
        running = bool(th and isinstance(th, QtCore.QThread) and th.isRunning())
        if not running:
            self._set_busy_ui("analysis", False)
            setattr(self, "_analysis_running", False)
            # ✅ 혹시 남아있으면 강제 개방
            try:
                self._force_unlock_all_buttons()
            except Exception:
                pass

    # MainWindow 안에 추가
    def _guard_alert(self, name: str):
        """이미 같은 작업이 실행 중일 때 사용자에게 팝업으로 알림(과도한 중복 팝업은 3초 스로틀)."""
        import time
        if not hasattr(self, "_last_alert_at"):
            self._last_alert_at = {}

        now = time.monotonic()
        last = float(self._last_alert_at.get(name, 0.0) or 0.0)
        if now - last < 3.0:
            # 3초 내 재클릭이면 팝업은 생략하고 상태바만 갱신
            try:
                self.status.showMessage("이미 같은 작업이 진행 중입니다.")
            except Exception:
                pass
            return
        self._last_alert_at[name] = now

        # 작업별 안내 문구
        if name == "music":
            msg = "음악 생성이 이미 진행 중입니다. 동시에 두 번 실행할 수 없어요.\n작업이 끝나면 자동으로 해제됩니다."
        elif name == "analysis":
            msg = "음악 분석이 이미 진행 중입니다. 동시에 두 번 실행할 수 없어요.\n작업이 끝나면 자동으로 해제됩니다."
        else:
            msg = "작업이 진행 중입니다. 완료 후 다시 시도하세요."

        # 음악 생성은 진행창이 있으니 열어줄지 물어봄, 그 외엔 정보 팝업만
        if name == "music" and getattr(self, "_dlg", None) is not None:
            ret = QtWidgets.QMessageBox.question(
                self, "안내", msg + "\n\n지금 '진행상황 보기' 창을 열까요?",
                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No
            )
            if ret == QtWidgets.QMessageBox.Yes:
                try:
                    self.on_show_progress()
                except Exception:
                    pass
        else:
            QtWidgets.QMessageBox.information(self, "안내", msg)

    # 분석 연결 함수
    def _cleanup_analysis_state(self):
        # 플래그/스레드 핸들 정리
        self._task_done("analysis")
        # ★ 반드시 UI 잠금 해제까지
        self._set_busy_ui("analysis", False)









    def _start_build_story_from_analysis(self, vocal_path: Path, project_json: Path):
        if self._story_build_running:
            self._dbg("skip: story build already running")
            # 팝업 대신 상태표시줄/로그만
            self.status.showMessage("story.json 생성이 이미 진행 중입니다.")
            if self._dlg:
                self._dlg.append_log("[STORY] 이미 진행 중입니다.")
            return
        self._story_build_running = True

        fps = int(self.sb_outfps.value())
        self._dbg(f"start_build_story: vpath={vocal_path} (exists={vocal_path.exists()}) "
                  f"| project_json={project_json} (exists={project_json.exists()}) | fps={fps}")

        # 진행 콜백 → 진행창/콘솔로 브릿지
        def _onp(msg: str):
            m = str(msg).rstrip()
            print(f"[STORY] {m}", flush=True)
            if self._dlg:
                try:
                    self._dlg.append_log(m)
                except Exception:
                    pass

        def job():
            import time
            t0 = time.time()
            _onp("빌드 시작")

            # 기존 story.json 생성
            # (이미 파일 상단에서 build_story_json이 임포트되어 있을 겁니다)
            story_result = build_story_json(
                audio_file=str(vocal_path),
                project_json=str(project_json),
                fps=fps,
                max_shot_sec=0,  # 분할 없음
                enable_transitions=True,
                use_gpt=False,
                prompt_provider=lambda **kw: self._ai.scene_prompt_kor(**kw),
                also_build_docs=True,
                ai=self._ai,
                workflow_dir=getattr(S, "JSONS_DIR", None),
                on_progress=_onp,
            )

            # 인트로/아웃트로 타임라인 10% 반영
            try:
                from app.image_movie_docs import (
                    apply_intro_outro_to_story_json,
                    apply_intro_outro_to_image_json,
                    apply_intro_outro_to_movie_json,
                )
            except ImportError:
                from image_movie_docs import (
                    apply_intro_outro_to_story_json,
                    apply_intro_outro_to_image_json,
                    apply_intro_outro_to_movie_json,
                )

            proj_dir_str = str(vocal_path.parent)
            sp = Path(proj_dir_str) / "story.json"
            sd = load_json(sp, {}) or {}
            _dur = float(sd.get("duration") or 0.0)
            if _dur >= 60.0:
                apply_intro_outro_to_story_json(proj_dir_str, intro_ratio=0.10, outro_ratio=0.10)
                apply_intro_outro_to_image_json(proj_dir_str, intro_ratio=0.10, outro_ratio=0.10)
                apply_intro_outro_to_movie_json(proj_dir_str, intro_ratio=0.10, outro_ratio=0.10)

            _onp(f"빌드 종료 (elapsed {time.time() - t0:0.2f}s)")
            return story_result

        # QThread + Worker (강한 참조 유지)
        self._story_thread = QtCore.QThread(self)
        self._story_worker = Worker(job)
        self._story_worker.moveToThread(self._story_thread)

        self._story_thread.started.connect(lambda: self._dbg("QThread started (story build)"))
        self._story_thread.started.connect(self._story_worker.run)

        def on_done(res, err):
            try:
                if err:
                    self._dbg(f"[STORY] 실패: {err}")
                    self.status.showMessage("story.json 생성 실패")
                    if self._dlg:
                        self._dlg.append_log("[STORY] 실패")
                        self._dlg.append_log(err)
                    return

                # 반환 파싱
                story_p = img_p = mov_p = None
                if isinstance(res, dict):
                    story_p = Path(res.get("story") or "")
                    img_p = res.get("image")
                    mov_p = res.get("movie")
                elif res:
                    story_p = Path(res)

                if story_p and story_p.exists():
                    self._dbg(f"story ready: {story_p} size={story_p.stat().st_size}")
                    msg = f"story.json 생성됨: {story_p}"
                    if img_p: msg += f" | image.json: {img_p}"
                    if mov_p: msg += f" | movie.json: {mov_p}"
                    print("[STORY]", msg, flush=True)
                    self.status.showMessage("story.json 생성 완료")
                    try:
                        print("[PROMPTS] ===== GPT rewrite start =====", flush=True)
                        save_story_overwrite_with_prompts(story_p)
                        print("[PROMPTS] ===== GPT rewrite done  =====", flush=True)
                    except Exception as e:
                        print(f"[PROMPTS] ERROR: {e}", flush=True)

                    if self._dlg:
                        self._dlg.append_log(msg)
                        self._dlg.set_completed("음악 생성 + story.json 생성 완료 ✅")
                else:
                    self._dbg("on_done: invalid path returned or file missing")
                    self.status.showMessage("story.json 경로 확인 실패")
                    if self._dlg:
                        self._dlg.append_log("[STORY] 결과 파일을 확인하지 못했습니다.")
            finally:
                self._story_build_running = False
                try:
                    self._story_thread.quit()
                except Exception:
                    pass
                self._story_thread = None
                self._story_worker = None

        self._story_worker.done.connect(on_done)
        self._story_worker.done.connect(self._story_worker.deleteLater)
        self._story_thread.finished.connect(lambda: self._dbg("QThread finished (story build)"))
        self._story_thread.finished.connect(self._story_thread.deleteLater)

        # 진행창 제목/상태 업데이트
        if self._dlg:
            try:
                self._dlg.set_title("음악 생성 → story 빌드")
                self._dlg.set_status("story.json 생성 시작…")
            except Exception:
                pass

        self._story_thread.start()


    # --- class MainWindow(...) 내부에 메서드로 추가 ---

    from pathlib import Path

    @staticmethod
    def _move_generated_audio_to_target(title: str) -> list[Path]:
        """
        ComfyUI 결과 오디오를 최종 목적지로 이동하고 mp3로 변환한다.
        반환: 최종 저장된 파일 경로 리스트(최대 1개, vocal.mp3)
        """
        # 함수 내 import 제거, 파일 상단의 S를 사용합니다.
        safe_title = _sanitize_title_for_path(title)

        # 1) 원본 폴더(ComfyUI가 저장한 곳)
        src_root = Path(getattr(S, "COMFY_RESULT_ROOT", r"C:\comfyResult\shorts_make"))
        src_dir = src_root / safe_title
        if not src_dir.exists():
            # 제목 폴더가 없으면 최신 폴더로 추정
            cand = [p for p in src_root.glob("*") if p.is_dir()]
            src_dir = max(cand, key=lambda p: p.stat().st_mtime) if cand else src_dir

        # 2) 목적지 폴더(FINAL_OUT 템플릿 치환)
        dst_dir = _resolve_audio_dir_from_template(getattr(S, "FINAL_OUT", str(S.BASE_DIR)), title)
        dst_dir.mkdir(parents=True, exist_ok=True)

        # 허용 오디오 확장자
        audio_exts = set(getattr(S, "AUDIO_EXTS", {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}))

        # 3) 오디오 파일 수집 및 최신 파일 선택
        files = [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in audio_exts]
        if not files:
            return []

        files.sort(key=lambda p: p.stat().st_mtime)
        latest = files[-1]  # 가장 최근 파일만 사용

        dst = dst_dir / "vocal.mp3"
        try:
            if latest.suffix.lower() == ".mp3":
                # mp3면 이름만 변경하여 이동
                if dst.exists():
                    dst.unlink()
                shutil.move(str(latest), str(dst))
            else:
                # mp3가 아니면 ffmpeg로 변환
                ff = getattr(S, "FFMPEG_EXE", "") or "ffmpeg"
                cp = subprocess.run(
                    [ff, "-y", "-i", str(latest), "-vn", "-c:a", "libmp3lame", "-q:a", "2", str(dst)],
                    capture_output=True, text=True, check=False
                )
                # 변환 실패하면 원본명으로라도 이동 (유실 방지)
                if cp.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
                    fallback = dst_dir / latest.name
                    shutil.move(str(latest), str(fallback))
                    return [fallback]
        except Exception:
            return []

        return [dst]


    # ────────────── 영상 빌드(선택) ──────────────
    def on_video(self):
        proj = self._latest_project()
        if not proj:
            QtWidgets.QMessageBox.warning(self, "안내", "프로젝트가 없습니다.")
            return
        total = self.sb_total.value()
        ov_fixed = recalc_overlap(self.sb_infps.value(), self.sb_outfps.value(), self.sb_overlap.value())
        try:
            self.status.showMessage("i2v 세그먼트 생성…")
            clips = build_shots_with_i2v(str(proj), total)
            self.status.showMessage("합치기…")
            final = xfade_concat(
                clips, ov_fixed, self.sb_outfps.value(),
                audio_path=(Path(proj) / "vocal.mp3"),
                out_path=(Path(proj) / "final.mp4"),
            )
            QtWidgets.QMessageBox.information(self, "완료", f"영상 완성: {final}")
            self.status.showMessage(f"완료: {final}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "오류", str(e))
            self.status.showMessage(f"오류: {e}")

    # ────────────── 기타 ──────────────
    def _set_total_frames_from_audio(self, audio_path: Path) -> None:
        dur = audio_duration_sec(audio_path)
        if dur <= 0:
            QtWidgets.QMessageBox.warning(self, "안내", "오디오 길이를 읽지 못했습니다.")
            return
        fps = self.sb_outfps.value()
        total = int(round(dur * fps))
        self.sb_total.setValue(max(total, 1))
        self.status.showMessage(f"오디오 길이 {dur:.2f}s × {fps}fps → 총 프레임 {total}")

    @staticmethod
    def _final_out_for_title(title: str) -> Path:
        # 함수 내 import 제거, 파일 상단에 이미 정의된 S를 사용합니다.
        # 또한, _resolve_audio_dir_from_template도 전역 함수이므로 self 없이 호출합니다.
        return _resolve_audio_dir_from_template(getattr(S, "FINAL_OUT", str(S.BASE_DIR)), title)

    def _img_dir_for_title(self, title: str) -> Path:
        # C:\my_games\shorts_make\maked_title\[title]\imgs
        return self._final_out_for_title(title) / "imgs"

    def _seg_dir_for_title(self, title: str) -> Path:
        # C:\my_games\shorts_make\maked_title\[title]\segmentation
        d = self._final_out_for_title(title) / "segmentation"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _ff() -> str:
        # 함수 내 import 제거, 파일 상단의 S를 사용합니다.
        return getattr(S, "FFMPEG_EXE", "") or "ffmpeg"

    def _find_story_json(self, title: str) -> Path | None:
        """우선 오디오 폴더(story.json), 없으면 FINAL_OUT\[title]\story.json."""
        # 1) project.json 위치와 vocal.* 위치 기준
        proj = self._latest_project()
        if proj:
            pj = proj / "project.json"
            if pj.exists():
                meta = load_json(pj, {}) or {}
                vp = meta.get("paths", {}).get("vocal_user_dir")
                if vp and Path(vp).exists():
                    cand = Path(vp).parent / "story.json"
                    if cand.exists():
                        return cand
        # 2) FINAL_OUT\[title]\story.json
        cand2 = self._final_out_for_title(title) / "story.json"
        return cand2 if cand2.exists() else None

    def _read_story(self, title: str) -> dict:
        p = self._find_story_json(title)
        if not p:
            raise FileNotFoundError("story.json 을 찾지 못했습니다.")
        p = self._normalize_story_file_to_scenes(p)  # ✅ shots → scenes 정규화
        return load_json(p, {}) or {}

    # MainWindow 내부에 추가
    @staticmethod
    def _dbg(msg: str):
        s = f"[TEST1DBG] {msg}"
        print(s, flush=True)
        try:
            _write_crash(s)  # 이미 모듈 상단에 있는 crash logger 재사용
        except Exception:
            pass




    # ==== [테스트1] 분석→story.json (항상 scenes로 저장) =======================
    def _gpt_scene_prompt(self, scene_ctx: dict, global_ctx: dict) -> str:
        """story.json의 scene(dict) + 전역 컨텍스트에서 AI.scene_prompt_kor 인자 구성"""
        section = str(scene_ctx.get("section") or "").lower()
        scene_hint = str(scene_ctx.get("scene") or "")
        characters = [str(c) for c in (scene_ctx.get("characters") or [])]

        # 전역 태그는 너무 길지 않게 6개 정도만
        tags = [str(t) for t in (global_ctx.get("tags") or [])][:6]

        # effect: list/str 모두 허용 → 1개만 고름
        eff = scene_ctx.get("effect")
        if isinstance(eff, list) and eff:
            effect = str(eff[0])
        elif isinstance(eff, str):
            effect = eff
        else:
            effect = None

        # 화면 전환 플래그를 motion 힌트로 전달(선택)
        motion = "transition" if scene_ctx.get("screen_transition") else None

        return self._ai.scene_prompt_kor(
            section=section,
            scene_hint=scene_hint,
            characters=characters,
            tags=tags,
            effect=effect,
            motion=motion,
        )

    # ======= usage test =======


    # -------------------- 파이프라인 엔트리 --------------------



    # ======= /end =======

    # ==== [테스트2] story.json → 씬 렌더(이미지 누락 검사, 중복 pass) ==========
    def on_test2_render_story(self):
        """
        story.json 의 scenes 를 순서대로 렌더
        - 이미지 경로: scene.img_file (없으면 FINAL_OUT\[title]\imgs\[id].png 추정)
        - 출력:       FINAL_OUT\[title]\segmentation\[id].mp4
        - 이미 있으면 pass / 누락 이미지 있으면 경고 후 중단
        - 누락이 있을 때, '캐릭터 2명 이상' 필요한 scene도 같이 경고에 표시
        """
        try:
            title = self.le_title.text().strip() or "untitled"
            story = self._read_story(title)
            scenes = story.get("scenes") or []
            if not scenes:
                QtWidgets.QMessageBox.warning(self, "안내", "story.json에 scenes가 없습니다.")
                return

            imgs_dir = self._img_dir_for_title(title)
            seg_dir = self._seg_dir_for_title(title)

            # ▼▼▼ _resolve_img 함수의 매개변수명을 수정합니다 ▼▼▼
            def _resolve_img(scene_id: str, img_file: str | None) -> Path:
                if img_file:
                    p = Path(img_file)
                    if p.exists() and p.stat().st_size > 0:
                        return p
                for ext in (".png", ".jpg", ".jpeg", ".webp"):
                    p = imgs_dir / f"{scene_id}{ext}"  # 'sid' -> 'scene_id'
                    if p.exists() and p.stat().st_size > 0:
                        return p
                return imgs_dir / f"{scene_id}.png"  # 'sid' -> 'scene_id'

            missing, multi_char = [], []
            for sc in scenes:
                sid = sc.get("id") or sc.get("title") or f"t_{int(sc.get('idx', 0) or 0):02d}"
                # 함수 호출은 그대로 유지됩니다.
                img = _resolve_img(sid, sc.get("img_file"))
                if not (img.exists() and img.stat().st_size > 0):
                    missing.append(f"{sid} → {img}")
                ch = sc.get("characters") or []
                if isinstance(ch, list) and len(ch) >= 2:
                    multi_char.append(sid)

            if missing:
                lines = ["누락 이미지가 있어 중단합니다.", ""]
                lines.append("[누락 이미지]")
                lines += missing[:30]
                if len(missing) > 30:
                    lines.append(f"... (총 {len(missing)}개 중 30개만 표시)")
                if multi_char:
                    lines += ["", "[2명 이상 캐릭터가 필요한 장면(참고)]", ", ".join(multi_char)]
                QtWidgets.QMessageBox.warning(self, "이미지 누락", "\n".join(lines))
                return

            fps = int(self.sb_outfps.value())
            made, skipped, failed = 0, 0, 0

            prog = QtWidgets.QProgressDialog("샷 렌더링 중…", "중지", 0, len(scenes), self)
            self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
            prog.setMinimumDuration(0)

            for i, sc in enumerate(scenes, 1):
                if prog.wasCanceled():
                    break
                sid = sc.get("id") or f"t_{i:02d}"
                dur = float(sc.get("duration") or (float(sc.get("end", 0)) - float(sc.get("start", 0))) or 0.0)
                if dur <= 0:
                    failed += 1
                    prog.setLabelText(f"{sid}: 유효하지 않은 duration")
                    prog.setValue(i)
                    continue

                img = _resolve_img(sid, sc.get("img_file"))
                out_mp4 = seg_dir / f"{sid}.mp4"

                if out_mp4.exists() and out_mp4.stat().st_size > 0:
                    skipped += 1
                else:
                    ok = self._build_clip_from_image(img, out_mp4, duration=dur, fps=fps)
                    if ok:
                        made += 1
                    else:
                        failed += 1

                prog.setLabelText(f"{sid}: 완료({made}) / 건너뜀({skipped}) / 실패({failed})")
                prog.setValue(i)

            prog.close()
            QtWidgets.QMessageBox.information(
                self, "테스트2",
                f"컷 렌더 완료\n\n생성: {made} | 건너뜀: {skipped} | 실패: {failed}\n폴더: {seg_dir}"
            )
            self.status.showMessage(f"컷 렌더 완료 — 생성 {made}, skip {skipped}, 실패 {failed}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "오류", str(e))
            self.status.showMessage(f"오류: {e}")

    # ==== [테스트3] 샷 합치기 (이미 있으면 pass) ================================
    def on_test3_concat_segments(self):
        """
        FINAL_OUT\[title]\segmentation 의 t_xx.mp4 들을 scenes 순서대로 합쳐 final_story.mp4 생성
        - 이미 최종 파일 있으면 pass
        """
        try:
            title = self.le_title.text().strip() or "untitled"
            story = self._read_story(title)
            scenes = story.get("scenes") or []
            if not scenes:
                QtWidgets.QMessageBox.warning(self, "테스트3", "story.json 에 scenes 가 없습니다.")
                return

            seg_dir = self._seg_dir_for_title(title)
            out_dir = self._final_out_for_title(title)
            final_path = out_dir / "final_story.mp4"
            if final_path.exists() and final_path.stat().st_size > 0:
                QtWidgets.QMessageBox.information(self, "테스트3", f"이미 최종 파일이 있습니다.\n{final_path}")
                return

            clips = [seg_dir / f"{sc.get('id')}.mp4" for sc in scenes]
            clips = [p for p in clips if p.exists() and p.stat().st_size > 0]
            if not clips:
                QtWidgets.QMessageBox.warning(self, "테스트3", "합칠 mp4를 찾지 못했습니다.")
                return

            filelist = seg_dir / "_concat.txt"
            with open(filelist, "w", encoding="utf-8") as fp:
                for p in clips:
                    safe = str(p).replace("\\", "\\\\").replace("'", "\\'")
                    fp.write(f"file '{safe}'\n")

            import subprocess, shlex
            ff = self._ff()
            cmd = [ff, "-y", "-f", "concat", "-safe", "0", "-i", str(filelist),
                   "-c", "copy", str(final_path)]
            print("[TEST3] ffmpeg:", " ".join(shlex.quote(x) for x in cmd), flush=True)
            cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if cp.returncode != 0 or not final_path.exists():
                QtWidgets.QMessageBox.critical(self, "테스트3", f"합치기 실패\n{cp.stderr[:1200]}")
                return

            QtWidgets.QMessageBox.information(self, "테스트3", f"완료: {final_path}")
            self.status.showMessage(f"완료: {final_path}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "오류", str(e))
            self.status.showMessage(f"오류: {e}")


# ───────── 워크플로 저장 노드(class_type) 영구 수정 도우미 ─────────


# ---- Add: rewrite SaveAudio nodes in a ComfyUI workflow JSON ----------------




# ==================================================

def _inject_render_prefs_methods():
    """MainWindow에 _add_render_prefs_controls / _save_ui_prefs_to_project 가 없으면 주입."""
    # Qt import (이미 상단에서 되어 있어도 안전)
    from PyQt5 import QtWidgets

    # 안전 import (app 패키지/단독 실행 모두 고려)
    try:
        from app import settings as S
        from app.utils import load_json, save_json, sanitize_title
    except Exception:
        import settings as S  # type: ignore
        from utils import load_json, save_json  # type: ignore
        def sanitize_title(x: str) -> str:
            return "".join(ch for ch in x if ch.isalnum() or ch in " _-").strip()

    # 이미 있으면 건너뜀
    if hasattr(MainWindow, "_add_render_prefs_controls") and hasattr(MainWindow, "_save_ui_prefs_to_project"):
        return

    def _guess_project_dir(self) -> Path:
        """현재 프로젝트 폴더 추정: _current_project_dir() → FINAL_OUT → BASE_DIR/title"""
        # 1) UI에서 제공하는 메서드가 있으면 사용
        if hasattr(self, "_current_project_dir"):
            try:
                d = self._current_project_dir()
                if d:
                    return Path(d)
            except Exception:
                pass
        # 2) 제목 기반
        try:
            title = sanitize_title(self.le_title.text().strip())
        except Exception:
            title = ""
        if not title:
            title = "무제"
        # 2-1) FINAL_OUT 템플릿이 있으면 우선
        fin = getattr(S, "FINAL_OUT", "")
        if fin and "[title]" in fin:
            return Path(fin.replace("[title]", title))
        # 2-2) BASE_DIR/[title]
        base = getattr(S, "BASE_DIR", ".")
        return Path(base) / title

    # ==== 메서드 정의: 드롭다운 UI 추가 ====
    def _add_render_prefs_controls(self, parent_layout: QtWidgets.QBoxLayout) -> None:
        grp = QtWidgets.QGroupBox("렌더 설정")
        row = QtWidgets.QHBoxLayout(grp)

        # 기본 컨트롤
        self.cmb_img_w = QtWidgets.QComboBox()
        self.cmb_img_h = QtWidgets.QComboBox()
        self.cmb_movie_fps = QtWidgets.QComboBox()

        self.cmb_img_w.setToolTip("이미지 가로 (width)")
        self.cmb_img_h.setToolTip("이미지 세로 (height)")
        self.cmb_movie_fps.setToolTip("타깃 FPS (i2v/렌더)")

        size_choices = getattr(S, "IMAGE_SIZE_CHOICES", [480, 520, 720, 960, 1080, 1280, 1440])
        for w in size_choices:
            self.cmb_img_w.addItem(str(int(w)), int(w))

        default_w, default_h = getattr(S, "DEFAULT_IMG_SIZE", (1080, 1920))
        h_candidates = {int(round(w * 16 / 9)) for w in size_choices}
        h_candidates.update({default_h})
        for hh in sorted(h_candidates):
            self.cmb_img_h.addItem(str(int(hh)), int(hh))

        for f in getattr(S, "MOVIE_FPS_CHOICES", [24, 60]):
            self.cmb_movie_fps.addItem(str(int(f)), int(f))

        # ── 해상도 프리셋 + 스텝(샘플링 단계) 컨트롤 ─────────────────────
        self.cmb_res_preset = QtWidgets.QComboBox()
        self.cmb_res_preset.setToolTip("해상도 프리셋(선택 시 W/H 자동 설정)")

        presets = [
            ("Shorts 9:16 · 720×1280", 720, 1280, "shorts_720x1280"),
            ("Shorts 9:16 · 832×1472", 832, 1472, "shorts_832x1472"),
            ("Shorts 9:16 · 1080×1920", 1080, 1920, "shorts_1080x1920"),
            ("Landscape 16:9 · 1280×720", 1280, 720, "land_1280x720"),
            ("Landscape 16:9 · 1920×1080", 1920, 1080, "land_1920x1080"),
            ("Landscape 16:9 · 2560×1440", 2560, 1440, "land_2560x1440"),
            ("Square 1:1 · 512×512", 512, 512, "square_512"),
            ("Square 1:1 · 1024×1024", 1024, 1024, "square_1024"),
            ("맞춤(커스텀)", -1, -1, "custom"),
        ]
        for label, wv, hv, key in presets:
            self.cmb_res_preset.addItem(label, (wv, hv, key))

        self.spn_t2i_steps = QtWidgets.QSpinBox()
        self.spn_t2i_steps.setRange(1, 200)
        self.spn_t2i_steps.setValue(24)
        self.spn_t2i_steps.setToolTip("샘플링 스텝 수(확산 단계 수)")

        # project.json 초기값 반영
        proj_dir = _guess_project_dir(self)
        pj = proj_dir / "project.json"
        meta = load_json(pj, {}) if pj.exists() else {}
        ui = meta.get("ui_prefs") or {}

        def _set_combo(combo: QtWidgets.QComboBox, val: int, fallback: int):
            idx = combo.findData(int(val))
            if idx < 0:
                idx = combo.findData(int(fallback))
            combo.setCurrentIndex(idx if idx >= 0 else 0)

        _set_combo(self.cmb_img_w, int((ui.get("image_size") or [default_w, default_h])[0]), int(default_w))
        _set_combo(self.cmb_img_h, int((ui.get("image_size") or [default_w, default_h])[1]), int(default_h))
        _set_combo(self.cmb_movie_fps, int(ui.get("movie_fps") or getattr(S, "DEFAULT_MOVIE_FPS", 24)),
                   int(getattr(S, "DEFAULT_MOVIE_FPS", 24)))

        preset_key0 = str((ui.get("resolution_preset") or "custom"))
        steps0 = int(ui.get("t2i_steps") or 24)
        self.spn_t2i_steps.setValue(steps0)

        # 프리셋 선택 시 W/H 적용 + 잠금/해제
        def _lock_wh(lock: bool) -> None:
            self.cmb_img_w.setEnabled(not lock)
            self.cmb_img_h.setEnabled(not lock)
            tip = "프리셋을 '맞춤(커스텀)'으로 바꾸면 해상도를 수정할 수 있습니다." if lock else "W/H를 직접 선택하세요."
            self.cmb_img_w.setToolTip(tip)
            self.cmb_img_h.setToolTip(tip)

        def _apply_preset_to_wh() -> None:
            wv, hv, key = self.cmb_res_preset.currentData()
            if key == "custom":
                _lock_wh(False)
                return
            i = self.cmb_img_w.findData(int(wv))
            if i >= 0:
                self.cmb_img_w.setCurrentIndex(i)
            j = self.cmb_img_h.findData(int(hv))
            if j >= 0:
                self.cmb_img_h.setCurrentIndex(j)
            _lock_wh(True)

        # 프리셋 초기값 선택
        kidx = 0
        for idx in range(self.cmb_res_preset.count()):
            _, _, key = self.cmb_res_preset.itemData(idx)
            if key == preset_key0:
                kidx = idx
                break
        self.cmb_res_preset.setCurrentIndex(kidx)
        self.cmb_res_preset.currentIndexChanged.connect(_apply_preset_to_wh)
        _apply_preset_to_wh()

        # 레이아웃
        row.addWidget(QtWidgets.QLabel("W"))
        row.addWidget(self.cmb_img_w)
        row.addWidget(QtWidgets.QLabel("H"))
        row.addWidget(self.cmb_img_h)
        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel("FPS"))
        row.addWidget(self.cmb_movie_fps)
        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel("프리셋"))
        row.addWidget(self.cmb_res_preset)
        row.addWidget(QtWidgets.QLabel("스텝"))
        row.addWidget(self.spn_t2i_steps)
        row.addStretch(1)
        parent_layout.addWidget(grp)

        # 변경 시 저장(기존 저장 메서드 사용)
        self.cmb_img_w.currentIndexChanged.connect(self._save_ui_prefs_to_project)
        self.cmb_img_h.currentIndexChanged.connect(self._save_ui_prefs_to_project)
        self.cmb_movie_fps.currentIndexChanged.connect(self._save_ui_prefs_to_project)
        self.cmb_res_preset.currentIndexChanged.connect(self._save_ui_prefs_to_project)
        self.spn_t2i_steps.valueChanged.connect(self._save_ui_prefs_to_project)

    # ==== 메서드 정의: project.json 저장 ====
    def _save_ui_prefs_to_project(self) -> None:
        proj_dir = _guess_project_dir(self)
        pj = proj_dir / "project.json"
        meta = load_json(pj, {}) if pj.exists() else {}
        ui = meta.get("ui_prefs") or {}

        w = int(self.cmb_img_w.currentData())
        h = int(self.cmb_img_h.currentData())
        fps = int(self.cmb_movie_fps.currentData())

        # 현재 선택된 해상도 프리셋 키 안전 추출
        preset_key = "custom"
        data = self.cmb_res_preset.currentData()
        if isinstance(data, tuple) and len(data) == 3:
            _, _, preset_key = data  # (w, h, key)

        ui["image_size"] = [w, h]
        ui["movie_fps"] = fps
        ui["resolution_preset"] = str(preset_key)
        ui["t2i_steps"] = int(self.spn_t2i_steps.value())

        meta["ui_prefs"] = ui
        save_json(pj, meta)

    # === ADD OR REPLACE: inside ShortsMainWindow (or your main widget) ===
    def on_click_test2_1_generate_missing_images(self) -> None:
        from pathlib import Path
        from progress import run_job_with_progress_async
        from settings import COMFY_LOG_FILE  # ComfyUI 로그 tail 경로
        # 로컬 임포트(앱/단일 실행 모두 지원)
        try:
            from app.video_build import build_missing_images_from_story
        except Exception:
            from video_build import build_missing_images_from_story  # type: ignore

        story_path = Path(self.txt_story_path.text()).resolve()

        # UI 값
        ui_w = int(self.cmb_img_w.currentData())
        ui_h = int(self.cmb_img_h.currentData())
        steps = int(self.spn_t2i_steps.value())

        def job(on_progress):
            return build_missing_images_from_story(
                story_path,
                ui_width=ui_w,
                ui_height=ui_h,
                steps=steps,
                timeout_sec=300,
                poll_sec=1.5,
                workflow_path=None,  # JSONS_DIR/nunchaku_qwen_image_swap.json 자동
                on_progress=on_progress,
            )

        def done(ok, payload, err):
            from PyQt5 import QtWidgets
            if not ok:
                QtWidgets.QMessageBox.critical(self, "이미지 생성 실패", str(err))
                return
            cnt = len(payload or [])
            QtWidgets.QMessageBox.information(self, "완료", f"새 이미지 {cnt}개 생성")

        run_job_with_progress_async(self, "테스트2_1: 누락 이미지 생성", job, tail_file=COMFY_LOG_FILE, on_done=done)

    # 클래스에 바인딩
    setattr(MainWindow, "_add_render_prefs_controls", _add_render_prefs_controls)
    setattr(MainWindow, "_save_ui_prefs_to_project", _save_ui_prefs_to_project)

# === 주입을 즉시 실행 (MainWindow 인스턴스 생성 전에!) ===
_inject_render_prefs_methods()

# ───────────────────────────── 실행 진입점 ─────────────────────────────
def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)  # type: ignore[attr-defined]
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("쇼츠 자동화")
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
