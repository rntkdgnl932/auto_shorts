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
from typing import Optional, Dict, Set
import requests
import math
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
        from video_build import normalize_to_v11  # type: ignore
    except Exception:
        def normalize_to_v11(story: dict) -> dict:  # type: ignore
            return story
# shorts_ui.py (상단 import 구역 추가)

# (이미 있다면 중복 추가 금지)
from app.audio_sync import build_story_json  # ← story/analyze 사용
# shorts_ui.py (상단 import 구역에 추가)
try:
    from app.image_movie_docs import apply_intro_outro_to_story_json
except ImportError:
    from video_build import apply_intro_outro_to_story_json
# shorts_ui.py (상단 import 구역에 추가)
try:
    from app.utils import save_story_overwrite_with_prompts
except ImportError:
    from utils import save_story_overwrite_with_prompts
import re
_CANON_RE = re.compile(r"[^a-z0-9]+")
try:
    from utils import AI
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
    from app.audio_sync import generate_music_with_acestep, rewrite_prompt_audio_format
    from app.tag_norm import normalize_tags_to_english
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
    from audio_sync import generate_music_with_acestep, rewrite_prompt_audio_format                         # type: ignore
    from utils import normalize_tags_to_english                                                          # type: ignore                                                       # type: ignore                                                   # type: ignore

# ==== CRASH LOGGER (붙여넣기) ====
# from pathlib import Path as _Path # Path 중복 import
try:
    import settings as _settings_for_log
except ImportError:
    from app import settings as _settings_for_log  # type: ignore

try:
    from app import settings
    from app.utils import run_job_with_progress_async
except ImportError:
    import settings  # type: ignore
    from utils import run_job_with_progress_async  # type: ignore
# ============================================================
# PROMPT PIPE (self-contained) — paste into shorts_ui.py
# - 외부 모듈/패키지 불필요 (표준 라이브러리만 사용)
# - 최종 저장 대상: story.json (네가 넘기는 경로)
# - female_* 캐릭터에는 'huge breasts' 자동 포함
# ============================================================
import re
from typing import List, Any, Callable
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
    from app.audio_sync import generate_music_with_acestep
except Exception:
    from utils import load_json, save_json
    from audio_sync import generate_music_with_acestep


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
    _tag_sync_booted: bool = False
    _tag_sync_inited: bool = False
    _tag_boxes: Dict[str, QtWidgets.QCheckBox] = {}
    _checked_tags: Set[str] = set()
    _tag_save_timer: Optional[QtCore.QTimer] = None
    _tag_watch_timer: Optional[QtCore.QTimer] = None
    _tag_watch_last_path: Optional[str] = None
    _tag_watch_last_mtime: Optional[float] = None

    def __init__(self):
        super().__init__()
        # ai
        self._ai = AI()

        # 중복 방지
        self._music_inflight = False  # <-- 이 줄을 추가하여 경고를 해결합니다.
        # --- reentry guard flags (추가) ---
        self._analysis_running = False  # 음악 분석 중복 방지
        self._story_build_running = False  # story.json 빌드 중복 방지
        self._docs_build_running = False # ← 문서(image/movie) 생성 중복 방지 (추가)
        self._seg_story_busy = False  # <<< 여기 속성 초기화 추가

        self._th: Optional[QtCore.QThread] = None
        self._worker: Optional[Worker] = None

        # 상태
        self._music_completed = False
        self._tail_completed = False
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

        # 이전 버전과의 호환성 또는 실수로 인한 속성 존재 시 제거
        if isinstance(getattr(self, "_current_project_dir", None), (str, Path)):  #
            try:  # 안전하게 제거 시도
                delattr(self, "_current_project_dir")  #
            except AttributeError:
                pass  # 속성이 없으면 무시

        self._start_analysis_watchdog()

        self._actions_bound = False  # ← 1회 바인딩 가드
        QtCore.QTimer.singleShot(0, self._bind_actions)

    ######### ######### ######### #########
    ######### 태그 실시간  반영  #########
    ######### ######### ######### #########
    @staticmethod
    def _safe_connect(signal_obj, slot) -> None:
        """
        PyQt 신호(.connect) 안전 연결 헬퍼.
        - signal_obj: QTimer.timeout, QCheckBox.stateChanged 등
        - slot: 호출 가능한 콜러블
        정적 메서드로 두어 '메서드가 static일 수 있습니다' 경고 제거.
        """
        try:
            # 일부 정적 분석기가 pyqtSignal에 대해 connect 속성 해석을 못해 경고를 냄.
            # getattr로 동적 접근하여 경고를 피하고, 런타임 안전성도 확보한다.
            conn = getattr(signal_obj, "connect", None)
            if callable(conn) and callable(slot):
                conn(slot)
        except Exception:
            # 연결 실패해도 앱이 죽지 않도록 방어
            pass

    def showEvent(self, event):
        """창이 표시될 때 태그 동기화를 부트스트랩한다. 기존 __init__/핸들러는 수정하지 않는다."""
        super().showEvent(event)
        if getattr(self, "_tag_sync_booted", False):
            return
        self._init_tag_sync()
        self._sync_tags_from_project_json()
        self._start_tag_watch()
        self._tag_sync_booted = True

    def _init_tag_sync(self) -> None:
        """태그 체크박스 수집 및 시그널 연결(1회)."""
        if getattr(self, "_tag_sync_inited", False):
            return

        self._tag_boxes: Dict[str, QtWidgets.QCheckBox] = self._collect_tag_checkboxes()
        self._checked_tags: Set[str] = set()

        # 저장 디바운스 타이머
        self._tag_save_timer = QtCore.QTimer(self)
        self._tag_save_timer.setSingleShot(True)
        self._safe_connect(self._tag_save_timer.timeout, self._persist_checked_tags_now)

        # 체크 이벤트 연결(정적 경고 없이 안전 연결)
        for label, cb in self._tag_boxes.items():
            self._safe_connect(getattr(cb, "stateChanged", None),
                               lambda _s, lab=label: self._on_tag_state_changed(lab))

        # ★ 자동태그 체크박스 통일(alias): 둘 중 있는 위젯을 찾아 둘 다 같은 객체로 맞춤
        auto_any = (
                getattr(self, "chk_auto_tags", None)
                or getattr(self, "cb_auto_tags", None)
                or getattr(getattr(self, "ui", None), "chk_auto_tags", None)
                or getattr(getattr(self, "ui", None), "cb_auto_tags", None)
        )
        if auto_any is not None:
            # 둘 다 같은 객체를 가리키게 통일
            setattr(self, "chk_auto_tags", auto_any)
            setattr(self, "cb_auto_tags", auto_any)
            # 토글 → 수동 태그 활성/비활성 즉시 반영
            self._safe_connect(getattr(auto_any, "stateChanged", None), self._on_auto_tags_toggled)

        self._tag_sync_inited = True

    def _on_auto_tags_toggled(self, _state: int) -> None:
        """
        자동태그 체크박스 토글 시, 수동 태그 체크박스의 활성/비활성만 즉시 반영한다.
        - 체크 상태나 저장 로직은 변경하지 않는다.
        - _init_tag_sync에서 chk_auto_tags/cb_auto_tags를 동일 객체로 통일함.
        """
        auto_on = False
        auto_chk = getattr(self, "chk_auto_tags", None)  # ← 통일된 핸들 사용
        if auto_chk is not None and hasattr(auto_chk, "isChecked"):
            try:
                auto_on = bool(auto_chk.isChecked())
            except Exception:
                auto_on = False

        tag_boxes = getattr(self, "_tag_boxes", None)
        if isinstance(tag_boxes, dict):
            for _label, tag_box in tag_boxes.items():
                try:
                    tag_box.setEnabled(not auto_on)
                except Exception:
                    pass

    def _collect_tag_checkboxes(self) -> Dict[str, QtWidgets.QCheckBox]:
        """
        UI의 QCheckBox를 텍스트로 매칭한다(objectName 불필요). 비교는 소문자+trim 기준.
        기존 UI 구조/텍스트를 변경하지 않는다.
        """
        allowed: Set[str] = {
            # Vocal
            "soft female voice", "soft male voice", "mixed vocals",
            # Basic Vocal
            "clean vocals", "natural articulation", "warm emotional tone",
            "studio reverb light", "clear diction", "breath control", "balanced mixing",
            # Style
            "electronic", "rock", "pop", "funk", "soul", "cyberpunk",
            "acid jazz", "edm", "soft electric drums", "melodic",
            # Scene
            "background music for parties", "radio broadcasts", "workout playlists",
            # Instrument
            "saxophone", "jazz", "piano", "violin", "acoustic guitar", "electric bass",
            # Tempo/Pro
            "110 bpm", "fast tempo", "slow tempo", "loops", "fills",
        }

        mapping: Dict[str, QtWidgets.QCheckBox] = {}
        for cb in self.findChildren(QtWidgets.QCheckBox):
            label_raw = (cb.text() or "").strip()
            if not label_raw:
                continue
            if label_raw.lower() in allowed and label_raw not in mapping:
                mapping[label_raw] = cb
        return mapping

    def _on_tag_state_changed(self, label: str) -> None:
        cb = self._tag_boxes.get(label)
        if cb is None:
            return
        if cb.isChecked():
            self._checked_tags.add(label)
        else:
            self._checked_tags.discard(label)
        self._persist_checked_tags_debounced()

    def _sync_tags_from_project_json(self) -> None:
        """
        태그 UI 동기화.

        [원칙]
        1) 콜드 스타트(앱 첫 실행, 이 함수의 첫 호출):
           - project.json을 절대 읽지 않는다.
           - 자동태그를 켠 상태로 만들고(체크박스 체크),
           - Basic Vocal 7개를 모두 체크하며, UI는 비활성화로 보이도록 설정한다.
           - 저장하지 않는다.
        2) 그 이후(프로젝트 열기/가사생성/음악생성 등으로 project.json이 준비된 뒤):
           - project.json 의 checked_tags > tags_effective 순으로 반영.
           - 저장값이 없으면 Basic Vocal 7개를 체크하고 저장.
        """
        import os
        from pathlib import Path
        from PyQt5 import QtCore

        tag_boxes = getattr(self, "_tag_boxes", None)
        if not isinstance(tag_boxes, dict) or not tag_boxes:
            return

        basic_defaults_all = {
            "clean vocals",
            "clear diction",
            "natural articulation",
            "breath control",
            "warm emotional tone",
            "balanced mixing",
            "studio reverb light",
        }
        basic_defaults = [label for label in tag_boxes.keys() if label in basic_defaults_all]

        def _apply_defaults(auto_enabled: bool = True) -> None:
            auto_checkbox_local = getattr(self, "chk_auto_tags", None)
            if auto_checkbox_local is not None:
                try:
                    auto_checkbox_local.blockSignals(True)
                    auto_checkbox_local.setChecked(bool(auto_enabled))
                    auto_checkbox_local.blockSignals(False)
                except Exception:
                    pass

            self._checked_tags = set()
            for tag_label_local, tag_box_local in tag_boxes.items():
                should_mark_local = (tag_label_local in basic_defaults)
                try:
                    tag_box_local.blockSignals(True)
                    tag_box_local.setChecked(should_mark_local)
                    tag_box_local.setEnabled(False if auto_enabled else True)
                    tag_box_local.blockSignals(False)
                except Exception:
                    pass
                if should_mark_local:
                    self._checked_tags.add(tag_label_local)

        # 1) 콜드 스타트: 세션 내 첫 호출이면 기본값 적용 후 종료
        if not getattr(self, "_tags_synced_once", False):
            _apply_defaults(auto_enabled=True)
            try:
                QtCore.QTimer.singleShot(0, lambda: _apply_defaults(auto_enabled=True))
            except Exception:
                pass
            setattr(self, "_tags_synced_once", True)
            return

        # 2) 이후: project.json 준비됐을 때만 디스크 반영
        if not bool(getattr(self, "_project_context_ready", False)):
            return

        try:
            from app.utils import load_json  # type: ignore
        except Exception:
            try:
                from utils import load_json  # type: ignore
            except Exception:
                return

        # --- proj_dir 안전 획득(정적 타입 체커 만족) ---
        proj_dir_obj = getattr(self, "_current_project_dir", None)
        if callable(proj_dir_obj):
            try:
                proj_dir_obj = proj_dir_obj()
            except Exception:
                proj_dir_obj = None
        if not isinstance(proj_dir_obj, (str, bytes, os.PathLike)):
            proj_dir_obj = getattr(self, "project_dir", None) or getattr(self, "_forced_project_dir", None)
        if not isinstance(proj_dir_obj, (str, bytes, os.PathLike)):
            return

        proj_dir = os.fspath(proj_dir_obj)
        meta_path = Path(proj_dir) / "project.json"
        if not meta_path.exists():
            _apply_defaults(auto_enabled=True)
            return

        meta = load_json(meta_path, {}) or {}

        if isinstance(meta.get("checked_tags"), list):
            selected = [str(x) for x in meta["checked_tags"]]
            should_persist = False
        elif isinstance(meta.get("tags_effective"), list):
            selected = [str(x) for x in meta["tags_effective"]]
            should_persist = True
        else:
            selected = list(basic_defaults)
            should_persist = True

        auto_on_state = False
        auto_checkbox = getattr(self, "chk_auto_tags", None)
        if auto_checkbox is not None:
            try:
                auto_on_state = bool(auto_checkbox.isChecked())
            except Exception:
                auto_on_state = False

        self._checked_tags = set()
        for tag_label, tag_box in tag_boxes.items():
            should_mark = (tag_label in selected)
            try:
                tag_box.blockSignals(True)
                tag_box.setChecked(should_mark)
                tag_box.setEnabled(False if auto_on_state else True)
                tag_box.blockSignals(False)
            except Exception:
                pass
            if should_mark:
                self._checked_tags.add(tag_label)

        if should_persist and "checked_tags" not in meta:
            try:
                meta["checked_tags"] = sorted(self._checked_tags)
                if hasattr(self, "_persist_checked_tags_now") and callable(self._persist_checked_tags_now):
                    self._persist_checked_tags_now(meta_override=meta)
            except Exception:
                pass

    def _persist_checked_tags_debounced(self, msec: int = 250) -> None:
        if hasattr(self, "_tag_save_timer"):
            self._tag_save_timer.start(msec)
        else:
            self._persist_checked_tags_now()

    def _persist_checked_tags_now(self, *, meta_override: dict | None = None) -> None:
        import os
        from pathlib import Path
        try:
            from app.utils import load_json, save_json  # type: ignore
        except Exception:
            try:
                from utils import load_json, save_json  # type: ignore
            except Exception:
                return

        proj_dir_obj = getattr(self, "_current_project_dir", None)
        if callable(proj_dir_obj):
            try:
                proj_dir_obj = proj_dir_obj()
            except Exception:
                proj_dir_obj = None
        if not isinstance(proj_dir_obj, (str, bytes, os.PathLike)):
            proj_dir_obj = getattr(self, "project_dir", None) or getattr(self, "_forced_project_dir", None)
        if not isinstance(proj_dir_obj, (str, bytes, os.PathLike)):
            return

        proj_dir = os.fspath(proj_dir_obj)
        meta_path = Path(proj_dir) / "project.json"
        meta = meta_override if meta_override is not None else (load_json(meta_path, {}) or {})
        meta["checked_tags"] = sorted(self._checked_tags)
        try:
            save_json(meta_path, meta)
        except Exception:
            pass

    def get_checked_tags(self) -> List[str]:
        """외부에서 현재 선택 태그를 읽을 때 사용."""
        return sorted(self._checked_tags)

    def _start_tag_watch(self) -> None:
        """
        태그 파일(project.json) 변경 감시 시작.
        - 콜드 스타트에서는 사용자 액션(프로젝트 열기/가사생성/음악생성) 전까지 감시를 '대기' 상태로 둔다.
        """
        from PyQt5 import QtCore

        # 기존 타이머 가져오거나 새로 생성
        timer = getattr(self, "_tag_watch_timer", None)
        if not isinstance(timer, QtCore.QTimer):
            timer = QtCore.QTimer(self)
            timer.setInterval(800)  # 0.8s 주기

            # 정적 분석 경고 방지: 시그널 객체 존재와 connect 가용성 점검 후 연결
            timeout_sig = getattr(timer, "timeout", None)
            if hasattr(timeout_sig, "connect"):
                timeout_sig.connect(self._tick_tag_watch)

        # 인스턴스에 보관
        self._tag_watch_timer = timer

        # 프로젝트 준비 플래그: 반드시 False로 시작(콜드 스타트 차단)
        setattr(self, "_project_context_ready", False)
        setattr(self, "_tag_watch_last_mtime", None)

        # 타이머 시작 (중복 시작 방지)
        if not timer.isActive():
            timer.start()

    def _tick_tag_watch(self) -> None:
        """
        태그 파일 감시 틱.
        - 프로젝트 컨텍스트가 준비된 이후에만 디스크(project.json)를 읽어 UI에 반영
        - mtime 변화가 있을 때만 반영
        """
        import os
        from pathlib import Path

        if not bool(getattr(self, "_project_context_ready", False)):
            return

        proj_dir_obj = getattr(self, "_current_project_dir", None)
        if callable(proj_dir_obj):
            try:
                proj_dir_obj = proj_dir_obj()
            except Exception:
                proj_dir_obj = None
        if not isinstance(proj_dir_obj, (str, bytes, os.PathLike)):
            proj_dir_obj = getattr(self, "project_dir", None) or getattr(self, "_forced_project_dir", None)
        if not isinstance(proj_dir_obj, (str, bytes, os.PathLike)):
            return

        proj_dir = os.fspath(proj_dir_obj)
        meta_path = Path(proj_dir) / "project.json"
        if not meta_path.exists():
            return

        try:
            mtime = os.path.getmtime(meta_path)
        except Exception:
            return

        last_mtime = getattr(self, "_tag_watch_last_mtime", None)
        if last_mtime is not None and last_mtime == mtime:
            return  # 변화 없음

        setattr(self, "_tag_watch_last_mtime", mtime)
        try:
            self._sync_tags_from_project_json()
        except Exception:
            pass

    ######### ######### ######### #########
    ######### ######### ######### #########
    ######### ######### ######### #########
    def _bind_actions(self) -> None:
        if getattr(self, "_actions_bound", False):
            return
        self._actions_bound = True

        from PyQt5 import QtWidgets

        def _btn(owner, name: str):
            obj = getattr(owner, name, None)
            return obj if isinstance(obj, QtWidgets.QAbstractButton) else None

        ui = getattr(self, "ui", None)

        pairs = [
            (_btn(ui, "btn_generate_lyrics") or _btn(self, "btn_gen"), self.on_generate_lyrics_with_log),
            (_btn(ui, "btn_generate_music") or _btn(self, "btn_music"), self.on_click_generate_music),

            # ▶ 프로젝트분석 버튼: seg → story → AI (비동기)
            (_btn(ui, "btn_test1_story") or _btn(self, "btn_test1_story"), self.on_click_build_story_from_seg_async),

            (_btn(ui, "btn_test2_1_img") or _btn(self, "btn_test2_1_img"),
             self.on_click_test2_1_generate_missing_images_with_log),
            (_btn(ui, "btn_analyze") or _btn(self, "btn_analyze"), self.on_click_analyze_music),
            (_btn(ui, "btn_convert_toggle") or _btn(self, "btn_convert_toggle"), self.on_convert_toggle),
        ]

        for btn, handler in pairs:
            if not btn or not callable(handler):
                continue
            try:
                btn.clicked.disconnect()
            except TypeError:
                pass
            btn.clicked.connect(handler)

    def on_toggle_convert(self, checked: bool) -> None:
        """
        변환(LLS) 토글 핸들러.
        - ON  : 현재 가사를 영어 발음(로마자) 표기로 변환하여 lyrics_lls에 저장, lls_enabled=True
        - OFF : lyrics_lls 비우고 lls_enabled=False
        - 우측 변환 에디터/패널은 항상 보이되 enable 상태만 토글
        """
        from pathlib import Path
        import os
        import json
        import re
        from json import JSONDecodeError

        # normalize_sections (있으면 사용)
        try:
            from app.lyrics_gen import normalize_sections  # type: ignore
        except ImportError:
            try:
                from lyrics_gen import normalize_sections  # type: ignore
            except ImportError:
                def normalize_sections(text_in: str) -> str:  # type: ignore
                    return text_in

        # 활성 프로젝트 경로 정규화
        def _active_project_dir() -> str:
            attr_list = [
                "_get_active_project_dir", "_current_project_dir", "current_project_dir",
                "_active_project_dir", "project_dir", "_forced_project_dir",
            ]
            for attr_name in attr_list:
                val = getattr(self, attr_name, None)
                if val is None:
                    continue
                if callable(val):
                    try:
                        val = val()
                    except (TypeError, AttributeError):
                        val = None
                if callable(val):
                    continue
                if isinstance(val, (str, os.PathLike)):
                    try:
                        return os.fspath(val)
                    except TypeError:
                        continue
                try:
                    as_str = str(val)
                    if as_str:
                        return as_str
                except (TypeError, ValueError):
                    continue
            return ""

        proj_dir = _active_project_dir()
        if not proj_dir:
            return

        meta_path = Path(proj_dir) / "project.json"

        # project.json 로드/세이브
        def _load_meta(p: Path) -> dict:
            try:
                return json.loads(p.read_text(encoding="utf-8")) or {}
            except (FileNotFoundError, JSONDecodeError, UnicodeDecodeError, OSError, TypeError, ValueError):
                return {}

        def _save_meta(p: Path, data: dict) -> None:
            try:
                p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError:
                return

        meta = _load_meta(meta_path)

        # 우측 패널/에디터 핸들
        right_editor = getattr(self, "te_lyrics_converted", None) or getattr(self, "txt_lyrics_converted", None)
        panel_widget = None
        for panel_name in ("grp_convert", "box_convert", "frame_convert", "gb_convert", "w_convert"):
            w = getattr(self, panel_name, None)
            if w is not None:
                panel_widget = w
                break

        # 패널은 항상 보이기, enable 상태만 토글
        try:
            if panel_widget is not None and hasattr(panel_widget, "setVisible"):
                panel_widget.setVisible(True)
            if right_editor is not None and hasattr(right_editor, "setVisible"):
                right_editor.setVisible(True)
            if panel_widget is not None and hasattr(panel_widget, "setEnabled"):
                panel_widget.setEnabled(bool(checked))
            if right_editor is not None and hasattr(right_editor, "setEnabled"):
                right_editor.setEnabled(bool(checked))
        except (AttributeError, RuntimeError):
            pass

        # OFF: lyrics_lls 비우기
        if not checked:
            meta["lls_enabled"] = False
            if "lyrics_lls" in meta:
                meta["lyrics_lls"] = ""
            _save_meta(meta_path, meta)
            if right_editor is not None and hasattr(right_editor, "clear"):
                try:
                    right_editor.clear()
                except (AttributeError, RuntimeError):
                    pass
            return

        # ===== ON: 변환 수행 =====

        # 1) 원문 가사(왼쪽 에디터 우선, 없으면 project.json: lyrics)
        src_text = ""
        left_editor = getattr(self, "te_lyrics", None) or getattr(self, "txt_lyrics", None)
        if left_editor is not None and hasattr(left_editor, "toPlainText"):
            try:
                src_text = (left_editor.toPlainText() or "").strip()
            except (AttributeError, TypeError, ValueError):
                src_text = ""
        if not src_text:
            src_text = str(meta.get("lyrics") or "").strip()

        # 2) 섹션 정규화
        norm_text = normalize_sections(src_text)

        # 3) 변환기: kroman → 실패 시 간단 로마자화 fallback
        def _convert(text_in: str) -> str:
            # kroman 사용 시
            try:
                import kroman  # type: ignore
                out_lines_k: list[str] = []
                for line_text in (text_in or "").splitlines():
                    s_line = (line_text or "").strip()
                    if not s_line:
                        continue
                    # 섹션 헤더는 그대로
                    if re.match(r"^\s*\[(?:verse|bridge)(?:\s+\d+)?]\s*$", s_line, flags=re.IGNORECASE):
                        out_lines_k.append(s_line.lower())
                        continue
                    # 기존 [xx] 태그 제거 후 본문
                    m_head = re.match(r"^\s*\[([a-z]{2})]\s*", s_line, flags=re.IGNORECASE)
                    tail_text = s_line[m_head.end():].lstrip() if m_head else s_line
                    rom = kroman.parse(tail_text).replace("-", "").strip()
                    out_lines_k.append("[ko]" + rom if rom else "[ko]")
                return "\n".join(out_lines_k).strip()
            except ImportError:
                # 아래 fallback 사용
                pass

            # fallback: 간단 로마자화(한글 부분만 분해→매핑)
            cho_map = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p",
                       "h"]
            jung_map = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe",
                        "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i"]
            jong_map = ["", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "p", "t", "l", "m", "p", "l", "m", "p",
                        "t", "t", "ng", "t", "t", "k", "t", "p", "t"]

            def _romanize_korean(seg_text: str) -> str:
                buf_chars: list[str] = []
                for ch in seg_text:
                    code = ord(ch)
                    if 0xAC00 <= code <= 0xD7A3:
                        si = code - 0xAC00
                        cho = si // 588
                        jung = (si % 588) // 28
                        jong = si % 28
                        buf_chars.append(cho_map[cho] + jung_map[jung] + jong_map[jong])
                    else:
                        buf_chars.append(ch)
                merged = "".join(buf_chars)
                merged = re.sub(r"\s{2,}", " ", merged).strip()
                return merged.lower()

            out_lines_f: list[str] = []
            for line_text in (text_in or "").splitlines():
                s_line = (line_text or "").strip()
                if not s_line:
                    continue
                if re.match(r"^\s*\[(?:verse|bridge)(?:\s+\d+)?]\s*$", s_line, flags=re.IGNORECASE):
                    out_lines_f.append(s_line.lower())
                    continue
                m_head = re.match(r"^\s*\[([a-z]{2})]\s*", s_line, flags=re.IGNORECASE)
                tail_text = s_line[m_head.end():].lstrip() if m_head else s_line
                has_ko = bool(re.search(r"[가-힣]", tail_text))
                if has_ko:
                    out_lines_f.append("[ko]" + _romanize_korean(tail_text))
                else:
                    # 영문/숫자만 있으면 en로 태깅
                    out_lines_f.append("[en]" + tail_text)
            return "\n".join(out_lines_f).strip()

        lls_text = _convert(norm_text)

        # 4) 에디터/메타 반영
        if right_editor is not None and hasattr(right_editor, "setPlainText"):
            try:
                right_editor.setPlainText(lls_text)
            except (AttributeError, RuntimeError):
                pass

        meta["lls_enabled"] = True
        meta["lyrics_lls"] = lls_text
        _save_meta(meta_path, meta)

        # 5) 로그(lyrics_gen.log) 요약
        log_path = Path(proj_dir) / "lyrics_gen.log"
        try:
            preview = "\n".join(lls_text.splitlines()[:10])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as fp:
                fp.write("\n===== CONVERT (LLS) =====\n")
                fp.write(preview + "\n")
        except OSError:
            pass

    def _apply_convert_ui_from_meta(self, proj_dir: str) -> None:
        """
        project.json의 상태를 오른쪽 UI에 반영:
        - lyrics_lls 존재 or lls_enabled=True → 변환 패널/에디터 보이기 + 내용 표시
        - 아니면 숨김
        버튼 상태는 사용자의 현재 상태를 해치지 않도록 시그널 차단 후 체크만 반영
        """
        from pathlib import Path
        import json

        pj = Path(proj_dir) / "project.json"
        try:
            meta = json.loads(pj.read_text(encoding="utf-8")) if pj.exists() else {}
        except Exception:
            meta = {}

        te_c = getattr(self, "te_lyrics_converted", None)
        panel = None
        for name in ("grp_convert", "box_convert", "frame_convert", "gb_convert", "w_convert"):
            w = getattr(self, name, None)
            if w is not None:
                panel = w
                break
        btn = getattr(self, "btn_convert_toggle", None)

        lls = (meta.get("lyrics_lls") or "").strip()
        on = bool(meta.get("lls_enabled")) or bool(lls)

        # 우측 에디터/패널 표시/텍스트 반영
        try:
            if te_c is not None and hasattr(te_c, "setVisible"):
                te_c.setVisible(on)
            if panel is not None and hasattr(panel, "setVisible"):
                panel.setVisible(on)
            if on and lls and hasattr(te_c, "setPlainText"):
                te_c.setPlainText(lls)
        except Exception:
            pass

        # 토글 버튼 체크 상태만 맞춤(시그널 차단해서 불필요한 핸들러 재실행 방지)
        try:
            if btn is not None and hasattr(btn, "setChecked"):
                if hasattr(btn, "blockSignals"):
                    btn.blockSignals(True)
                btn.setChecked(on)
                if hasattr(btn, "blockSignals"):
                    btn.blockSignals(False)
        except Exception:
            pass

    def on_convert_toggle(self, checked: bool) -> None:
        """
        변환(LLS) 토글:
        - ON  : meta['lls_enabled']=True, 필요 시 lyrics_lls 생성. 오른쪽 패널/에디터 '표시+활성화'.
        - OFF : meta['lls_enabled']=False, lyrics_lls 제거. 오른쪽 패널/에디터 '표시+비활성화(내용은 빈칸)'.
        """
        from pathlib import Path
        import json
        from json import JSONDecodeError

        # 파일 IO 유틸
        try:
            from app.utils import load_json as _lj, save_json as _sj  # type: ignore
            load_json_fn, save_json_fn = _lj, _sj
        except Exception:
            def load_json_fn(p: Path, default=None):
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except (FileNotFoundError, JSONDecodeError, UnicodeDecodeError, OSError):
                    return default

            def save_json_fn(p: Path, data: dict) -> None:
                try:
                    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass

        proj_dir = getattr(self, "_active_project_dir", None) or getattr(self, "project_dir", None)
        if not proj_dir:
            return

        pj = Path(proj_dir) / "project.json"
        meta = load_json_fn(pj, {}) or {}
        if not isinstance(meta, dict):
            meta = {}

        # 우측 변환 UI 핸들(이름이 다를 수 있어 안전하게 탐색)
        te_c = getattr(self, "te_lyrics_converted", None)
        panel = None
        for name in ("grp_convert", "box_convert", "frame_convert", "gb_convert", "w_convert"):
            w = getattr(self, name, None)
            if w is not None:
                panel = w
                break

        # 좌측 원문 에디터(최초 ON 시 변환할 때만 사용)
        te_l = getattr(self, "te_lyrics", None)

        def _convert(txt: str) -> str:
            try:
                import kroman  # type: ignore
                out_lines = []
                for line in (txt or "").splitlines():
                    s = line.strip()
                    if not s or (s.startswith("[") and s.endswith("]")):
                        out_lines.append(line)
                    else:
                        rom = kroman.parse(line).strip().replace("-", "")
                        out_lines.append("[ko]" + rom)
                return "\n".join(out_lines)
            except Exception:
                return txt or ""

        if checked:
            # ON: 정책 유지(필요 시 1회 생성), UI는 '표시+활성화'
            meta["lls_enabled"] = True
            lls_now = (meta.get("lyrics_lls") or "").strip()
            if not lls_now and hasattr(te_l, "toPlainText"):
                raw_txt = te_l.toPlainText()
                conv_txt = _convert(raw_txt)
                meta["lyrics_lls"] = conv_txt
                if hasattr(te_c, "setPlainText"):
                    te_c.setPlainText(conv_txt)

            save_json_fn(pj, meta)

            # 표시 + 활성화
            if panel is not None and hasattr(panel, "setVisible"):
                panel.setVisible(True)
            if te_c is not None:
                if hasattr(te_c, "setVisible"):
                    te_c.setVisible(True)
                if hasattr(te_c, "setEnabled"):
                    te_c.setEnabled(True)

        else:
            # OFF: 파일에서는 lyrics_lls 제거. UI는 '표시+비활성화', 내용은 빈칸.
            meta["lls_enabled"] = False
            if "lyrics_lls" in meta:
                meta.pop("lyrics_lls", None)
            save_json_fn(pj, meta)

            if panel is not None and hasattr(panel, "setVisible"):
                panel.setVisible(True)  # 숨기지 않음
            if te_c is not None:
                if hasattr(te_c, "clear"):
                    te_c.clear()
                if hasattr(te_c, "setVisible"):
                    te_c.setVisible(True)  # 항상 보임
                if hasattr(te_c, "setEnabled"):
                    te_c.setEnabled(False)  # 비활성화

    def on_generate_lyrics_with_log(self) -> None:
        """
        가사 생성 버튼 핸들러.
        - 진행창 로그를 파일과 UI에 동시에 남김
        - generate_title_lyrics_tags 호출
        - 생성 직후 프로젝트 폴더 추정/고정(음악생성 이어지도록)
        - 자동태그 UI enable/disable 동기화
        """
        from PyQt5 import QtWidgets
        from pathlib import Path
        import os

        try:
            from app.utils import run_job_with_progress_async  # type: ignore
        except ImportError:
            from utils import run_job_with_progress_async  # type: ignore
        try:
            from app.lyrics_gen import generate_title_lyrics_tags  # type: ignore
        except ImportError:
            from lyrics_gen import generate_title_lyrics_tags  # type: ignore

        btn = getattr(self, "btn_gen", None) or getattr(getattr(self, "ui", None), "btn_generate_lyrics", None)
        if btn:
            try:
                btn.setEnabled(False)
            except Exception:
                pass

        def _get_base_dir() -> Path:
            """설정의 BASE_DIR을 안전하게 Path로 반환."""
            try:
                from app import settings as settings_mod  # type: ignore
            except ImportError:
                import settings as settings_mod  # type: ignore
            val = getattr(settings_mod, "BASE_DIR", ".")
            try:
                return Path(val)
            except Exception:
                return Path(".")

        def _get_proj_dir_str() -> str:
            """현재 활성 프로젝트 경로를 문자열로 반환(Callable 처리 포함)."""
            cur = getattr(self, "_current_project_dir", None)
            if callable(cur):
                try:
                    cur = cur()
                except Exception:
                    cur = None
            if isinstance(cur, (str, bytes, os.PathLike)):
                try:
                    return os.fspath(cur)
                except TypeError:
                    return ""
            alt = getattr(self, "project_dir", None) or getattr(self, "_forced_project_dir", None)
            if isinstance(alt, (str, bytes, os.PathLike)):
                try:
                    return os.fspath(alt)
                except TypeError:
                    return ""
            return ""

        proj_dir = _get_proj_dir_str()
        if proj_dir:
            log_path = str(Path(proj_dir) / "lyrics_gen.log")
        else:
            # 프로젝트가 아직 없다면 임시 위치에 로그 남김
            import tempfile
            log_path = str(Path(tempfile.gettempdir()) / "lyrics_gen.log")

        def job(progress):
            # 파일/진행창 동시 로깅
            def _emit(line: str) -> None:
                try:
                    p = Path(log_path)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with p.open("a", encoding="utf-8") as fp:
                        fp.write((line or "").rstrip("\r\n") + "\n")
                except OSError:
                    pass
                try:
                    progress({"msg": line})
                except Exception:
                    try:
                        progress(line)
                    except Exception:
                        pass

            _emit("[ui] 가사 생성 시작")

            # 입력 수집
            title_in = ""
            le = getattr(self, "le_title", None)
            if le and hasattr(le, "text"):
                try:
                    title_in = (le.text() or "").strip()
                except Exception:
                    title_in = ""

            prompt_text = ""
            for nm in ("te_prompt", "txt_prompt", "prompt_edit"):
                w = getattr(self, nm, None) or getattr(getattr(self, "ui", None), nm, None)
                if w and hasattr(w, "toPlainText"):
                    try:
                        prompt_text = (w.toPlainText() or "").strip()
                    except Exception:
                        prompt_text = ""
                    break

            secs = 60
            if hasattr(self, "_current_seconds") and callable(self._current_seconds):
                try:
                    secs = int(self._current_seconds())
                except Exception:
                    secs = 60

            allowed = []
            getter = getattr(self, "_manual_option_set", None)
            if callable(getter):
                try:
                    vals = getter()
                    if isinstance(vals, (list, set, tuple)):
                        allowed = sorted(str(x) for x in vals)
                except Exception:
                    allowed = []

            prefer = "gemini" if (getattr(self, "btn_ai_toggle", None) and getattr(self,
                                                                                   "btn_ai_toggle").isChecked()) else "openai"
            allow_fb = prefer != "gemini"

            def trace(ev: str, msg: str):
                head = (ev or "").split(":", 1)[0]
                _emit(f"[{head}] {msg}")

            _emit(f"[ai] prefer={prefer}, secs={secs}")

            data = generate_title_lyrics_tags(
                prompt=prompt_text,
                duration_min=max(1, min(3, int(round(secs / 60)) or 1)),
                duration_sec=secs,
                title_in=title_in,
                allowed_tags=allowed,
                trace=trace,
                prefer=prefer,
                allow_fallback=allow_fb,
            )

            # 이후 단계에서 폴더 유실 방지용 후보도 함께 반환
            return {"data": data, "title": title_in, "prompt": prompt_text, "proj_dir": proj_dir}

        def done(ok: bool, payload, err):
            if btn:
                try:
                    btn.setEnabled(True)
                except Exception:
                    pass

            if not ok:
                QtWidgets.QMessageBox.critical(self, "가사 생성 실패", str(err))
                return

            pack = payload or {}
            data = pack.get("data", {}) or {}

            # ── 프로젝트 폴더 확정: data → 추정 → 최근 project.json → 최종 폴백
            final_dir = ""

            def _pick_dir_from_data(obj) -> str:
                if not isinstance(obj, dict):
                    return ""
                # 1) paths.project_dir 최우선
                paths = obj.get("paths")
                if isinstance(paths, dict):
                    v = paths.get("project_dir")
                    if isinstance(v, (str, bytes, os.PathLike)):
                        v2 = os.fspath(v)
                        if Path(v2).exists():
                            return v2
                # 2) title 기반 추정: BASE_DIR/maked_title/<title>
                title_guess = (obj.get("title") or pack.get("title") or "").strip()
                if title_guess:
                    root_path = _get_base_dir()  # ← base_dir 섀도잉 방지(이름 변경)
                    guess_path = root_path / "maked_title" / title_guess
                    if guess_path.exists():
                        return str(guess_path)
                return ""

            # 1) data에서 직접
            final_dir = _pick_dir_from_data(data)

            # 2) 없으면, job 시작 시점의 proj_dir 후보 사용
            if not final_dir:
                pd = pack.get("proj_dir")
                if isinstance(pd, (str, bytes, os.PathLike)):
                    pd2 = os.fspath(pd)
                    if Path(pd2).exists():
                        final_dir = pd2

            # 3) 그래도 없으면, 현재 UI 제목으로 추정
            if not final_dir:
                try:
                    cur_title = ""
                    le2 = getattr(self, "le_title", None)
                    if le2 and hasattr(le2, "text"):
                        cur_title = (le2.text() or "").strip()
                    if cur_title:
                        base_dir = _get_base_dir()
                        guess2 = base_dir / "maked_title" / cur_title
                        if guess2.exists():
                            final_dir = str(guess2)
                except Exception:
                    pass

            # 4) 가장 최근 project.json에서 경로 복구
            if not final_dir:
                base_dir = _get_base_dir()
                try:
                    pj_list = list((base_dir / "maked_title").glob("*/project.json"))
                    pj_list.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    if pj_list:
                        import json
                        meta = json.loads(pj_list[0].read_text(encoding="utf-8")) or {}
                        paths_obj = meta.get("paths") if isinstance(meta, dict) else None
                        inner = str(paths_obj.get("project_dir") or "") if isinstance(paths_obj, dict) else ""
                        if inner and Path(inner).exists():
                            final_dir = inner
                except Exception:
                    pass

            # 5) 최후 폴백 훅
            if not final_dir:
                last = getattr(self, "_latest_project", None)
                if callable(last):
                    try:
                        lv = last()
                        if isinstance(lv, (str, bytes, os.PathLike)):
                            lv2 = os.fspath(lv)
                            if Path(lv2).exists():
                                final_dir = lv2
                    except Exception:
                        pass

            # 확정되면 UI 활성 프로젝트로 세팅
            setter = getattr(self, "_set_active_project_dir", None)
            if final_dir and callable(setter):
                try:
                    setter(final_dir)
                except Exception:
                    pass

            # 기존 apply 로직 유지
            if hasattr(self, "_apply_lyrics_result"):
                try:
                    self._apply_lyrics_result(data, pack.get("title", ""), pack.get("prompt", ""))
                except Exception:
                    pass

            # project.json 연동 허용
            try:
                setattr(self, "_project_context_ready", True)
            except Exception:
                pass

            # 자동태그 상태에 맞춰 수동 태그 박스 enable/disable 유지
            auto_on = False
            auto_chk = getattr(self, "chk_auto_tags", None)
            if auto_chk is not None and hasattr(auto_chk, "isChecked"):
                try:
                    auto_on = bool(auto_chk.isChecked())
                except Exception:
                    auto_on = False
            tag_boxes = getattr(self, "_tag_boxes", None)
            if isinstance(tag_boxes, dict):
                for _label, box in tag_boxes.items():
                    try:
                        box.setEnabled(not auto_on)
                    except Exception:
                        pass

        # 진행창에 tail_file 연결(실시간 로그 표시)
        run_job_with_progress_async(self, "가사 생성", job, tail_file=log_path, on_done=done)

    # --- 누락 이미지 생성: 비동기 + 진행창 로그 (no _guess_project_dir) ---
    def on_click_test2_1_generate_missing_images_with_log(self) -> None:
        """video.json만 대상으로 누락된 씬 이미지를 생성한다(진행창/실시간 로그)."""
        from PyQt5 import QtWidgets
        from pathlib import Path
        from typing import Optional

        # ----- 0) 버튼 비활성화(있을 때만) -----
        btn = getattr(self, "btn_test2_1_img", None) or getattr(getattr(self, "ui", None), "btn_test2_1_img", None)
        if isinstance(btn, QtWidgets.QAbstractButton):
            btn.setEnabled(False)

        # ----- 1) 대상 JSON 경로: 반드시 video.json만 -----
        video_path: Optional[Path] = None

        # (A) UI 텍스트박스 우선: 파일/폴더 모두 허용
        tb = getattr(self, "txt_story_path", None)
        if tb and hasattr(tb, "text"):
            txt = (tb.text() or "").strip()
            if txt:
                p = Path(txt).resolve()
                if p.suffix.lower() == ".json":
                    if p.name.lower() == "video.json" and p.exists():
                        video_path = p
                else:
                    v = (p / "video.json").resolve()
                    if v.exists():
                        video_path = v

        # (B) 프로젝트 디렉터리에서 video.json
        if video_path is None:
            proj_dir = ""
            getter = getattr(self, "_current_project_dir", None)
            if callable(getter):
                try:
                    proj_dir = getter() or ""
                except (RuntimeError, OSError, TypeError, ValueError):
                    proj_dir = ""
            if not proj_dir:
                proj_dir = getattr(self, "project_dir", "") or ""
            if proj_dir:
                v = (Path(proj_dir).resolve() / "video.json")
                if v.exists():
                    video_path = v

        # (C) 마지막 폴백: 현재 작업 폴더의 video.json
        if video_path is None:
            v = (Path.cwd() / "video.json").resolve()
            if v.exists():
                video_path = v

        if video_path is None or not video_path.exists():
            QtWidgets.QMessageBox.critical(self, "누락 이미지 생성", "video.json을 찾을 수 없습니다.")
            if isinstance(btn, QtWidgets.QAbstractButton):
                btn.setEnabled(True)
            return

        # ----- 2) 실행 전 정합성 보정: img_file 비어있는 씬(gap 포함)에 기본 경로 자동 주입 -----
        # - 기존 생성 함수가 'img_file 비어있으면 스킵'일 수 있으므로, 실행 전에만 보정
        try:
            try:
                from app.utils import load_json, save_json  # type: ignore
            except ImportError:
                from utils import load_json, save_json  # type: ignore

            doc = load_json(video_path, None)
            if isinstance(doc, dict):
                scenes = doc.get("scenes")
                if isinstance(scenes, list):
                    imgs_dir = video_path.parent / "imgs"
                    try:
                        imgs_dir.mkdir(parents=True, exist_ok=True)
                    except OSError:
                        imgs_dir = video_path.parent / "imgs"  # 디렉터리 실패 시에도 경로 구성은 유지

                    changed = False
                    for sc in scenes:
                        if not isinstance(sc, dict):
                            continue
                        sid = str(sc.get("id") or "").strip()
                        if not sid:
                            continue
                        img_path_str = str(sc.get("img_file") or "").strip()
                        if not img_path_str:
                            sc["img_file"] = str((imgs_dir / f"{sid}.png").resolve())
                            changed = True

                    if changed:
                        save_json(video_path, doc)
        except (FileNotFoundError, OSError, TypeError, ValueError) as e_prep:
            print(f"[prep] img_file 보정 경고: {e_prep}")

        # ----- 3) UI에서 W/H/스텝 읽기 -----
        def _get_combo_int(combo_name: str, default_val: int) -> int:
            combo = getattr(self, combo_name, None)
            try:
                voji = combo.currentData()
                return int(voji if voji is not None else default_val)
            except (TypeError, ValueError, AttributeError):
                return int(default_val)

        ui_w = _get_combo_int("cmb_img_w", 720)
        ui_h = _get_combo_int("cmb_img_h", 1080)

        spn = getattr(self, "spn_t2i_steps", None)
        try:
            steps = int(spn.value()) if spn is not None else 24
        except (TypeError, ValueError, AttributeError):
            steps = 28

        # ----- 4) 진행창 + 비동기 작업 실행 -----
        try:
            from utils import run_job_with_progress_async
        except ImportError:
            from app.utils import run_job_with_progress_async  # type: ignore

        try:
            from video_build import build_missing_images_from_story  # 로컬
        except ImportError:
            from app.video_build import build_missing_images_from_story  # type: ignore

        try:
            import settings as _settings  # type: ignore
            comfy_log_file = getattr(_settings, "COMFY_LOG_FILE", None)
        except (ImportError, AttributeError):
            comfy_log_file = None

        def job(on_progress):
            created = build_missing_images_from_story(
                str(video_path),  # ✅ video.json만 대상으로 고정
                ui_width=ui_w,
                ui_height=ui_h,
                steps=steps,
                timeout_sec=300,
                poll_sec=1.5,
                workflow_path=None,
                on_progress=on_progress,
            )
            return {"created": [str(img_path) for img_path in created]}

        def done(ok: bool, payload, err):
            if isinstance(btn, QtWidgets.QAbstractButton):
                btn.setEnabled(True)

            if not ok:
                QtWidgets.QMessageBox.critical(self, "누락 이미지 생성 실패", str(err))
                return

            created = (payload or {}).get("created") or []
            QtWidgets.QMessageBox.information(self, "누락 이미지 생성 완료", f"생성 {len(created)}개 완료")

        run_job_with_progress_async(self, "누락 이미지 생성", job, tail_file=comfy_log_file, on_done=done)

    def on_click_generate_music(self) -> None:
        """음악 생성 버튼 핸들러 — 중복 실행 가드 추가 및 부정적 프롬프트 저장."""
        from PyQt5 import QtWidgets
        try:
            from app.utils import run_job_with_progress_async
        except ImportError:
            from utils import run_job_with_progress_async  # type: ignore

        if getattr(self, "_music_inflight", False):
            print("[UI] music already running -> ignore", flush=True)
            return
        self._music_inflight = True

        btn = getattr(self, "btn_music", None)
        if btn:
            try:
                btn.setEnabled(False)
            except (AttributeError, RuntimeError):
                pass

        def job(progress):
            from pathlib import Path
            try:
                from app.utils import load_json, save_json  # type: ignore
            except ImportError:
                from utils import load_json, save_json  # type: ignore

            try:
                progress({"msg": "[ui] 음악 생성 시작"})
            except (AttributeError, RuntimeError, TypeError):
                pass

            project_dir = None
            if hasattr(self, "_current_project_dir"):
                try:
                    project_dir = self._current_project_dir()
                except (AttributeError, RuntimeError, TypeError):
                    project_dir = None
            if not project_dir:
                raise RuntimeError("프로젝트 폴더를 찾을 수 없습니다.")

            secs = int(self._current_seconds()) if hasattr(self, "_current_seconds") else 60
            pj = Path(project_dir) / "project.json"
            meta = load_json(pj, {}) or {}

            # ▼▼▼ 부정적 프롬프트 읽기 및 저장 추가 ▼▼▼
            neg_prompt = ""
            if hasattr(self, "te_prompt_neg") and hasattr(self.te_prompt_neg, "toPlainText"):
                try:
                    neg_prompt = self.te_prompt_neg.toPlainText().strip()
                except (AttributeError, RuntimeError):
                    neg_prompt = ""
            meta['prompt_neg'] = neg_prompt
            # ▲▲▲ 부정적 프롬프트 읽기 및 저장 완료 ▲▲▲

            auto_on = False
            cb_auto = getattr(self, "cb_auto_tags", None) or getattr(self, "chk_auto_tags", None)
            if cb_auto is not None and hasattr(cb_auto, "isChecked"):
                try:
                    auto_on = bool(cb_auto.isChecked())
                except (AttributeError, ValueError):
                    auto_on = False

            picked_manual = []
            if hasattr(self, "_collect_manual_checked_tags") and callable(self._collect_manual_checked_tags):
                try:
                    picked_manual = list(self._collect_manual_checked_tags() or [])
                except (TypeError, ValueError):
                    picked_manual = []
            else:
                for lst_name in ("cb_basic_vocal_list", "cb_style_checks", "cb_scene_checks", "cb_instr_checks",
                                 "cb_tempo_checks"):
                    lst = getattr(self, lst_name, None)
                    if lst:
                        try:
                            for cb in lst:
                                if getattr(cb, "isChecked", lambda: False)():
                                    label = getattr(cb, "text", lambda: "")()
                                    if label:
                                        picked_manual.append(label)
                        except (AttributeError, TypeError, ValueError):
                            pass
                try:
                    if getattr(self, "rb_vocal_female", None) and self.rb_vocal_female.isChecked():
                        picked_manual.append("soft female voice")
                    elif getattr(self, "rb_vocal_male", None) and self.rb_vocal_male.isChecked():
                        picked_manual.append("soft male voice")
                    elif getattr(self, "rb_vocal_mixed", None) and self.rb_vocal_mixed.isChecked():
                        picked_manual.append("mixed vocals")
                except (AttributeError, ValueError):
                    pass

            last_ai_tags = list(getattr(self, "_last_tags", []) or [])

            meta["time"] = int(secs)
            meta["target_seconds"] = int(secs)

            if auto_on:
                meta["auto_tags"] = True
                meta["ace_tags"] = last_ai_tags if last_ai_tags else list(meta.get("ace_tags", []) or [])
                meta["tags_in_use"] = list(dict.fromkeys(picked_manual))
            else:
                meta["auto_tags"] = False
                meta["manual_tags"] = list(dict.fromkeys(picked_manual))

            save_json(pj, meta)

            def forward(info) -> None:
                try:
                    if isinstance(info, dict):
                        st = str(info.get("stage", "")).upper()
                        extra = {k: v for k, v in info.items() if k != "stage"}
                        msg = f"[{st}] {extra}" if extra else f"[{st}]"
                        progress({"msg": msg})
                    else:
                        progress({"msg": str(info)})
                except (AttributeError, RuntimeError, TypeError):
                    pass

            try:
                from app.audio_sync import generate_music_with_acestep
            except ImportError:
                from audio_sync import generate_music_with_acestep  # type: ignore

            import os as _os
            out = generate_music_with_acestep(
                _os.fspath(project_dir),
                on_progress=forward,
                target_seconds=secs,
            )
            progress({"msg": f"[done] 음악 저장: {out}"})
            return out

        def on_done(ok: bool, _payload, err):
            if btn:
                try:
                    btn.setEnabled(True)
                except (AttributeError, RuntimeError):
                    pass
            if not ok and err is not None:
                QtWidgets.QMessageBox.warning(self, "음악 생성 오류", str(err))
            else:
                try:
                    setattr(self, "_project_context_ready", True)
                except (AttributeError, RuntimeError):
                    pass

                auto_on = False
                auto_chk = getattr(self, "chk_auto_tags", None)
                if auto_chk is not None and hasattr(auto_chk, "isChecked"):
                    try:
                        auto_on = bool(auto_chk.isChecked())
                    except (AttributeError, RuntimeError, ValueError):
                        auto_on = False
                tag_boxes = getattr(self, "_tag_boxes", None)
                if isinstance(tag_boxes, dict):
                    for _label, box in tag_boxes.items():
                        try:
                            box.setEnabled(not auto_on)
                        except (AttributeError, RuntimeError):
                            pass

            self._music_inflight = False

        try:
            run_job_with_progress_async(self, "음악 생성 (ACE-Step)", job, on_done=on_done)
        except Exception as e:
            if btn:
                try:
                    btn.setEnabled(True)
                except (AttributeError, RuntimeError):
                    pass
            self._music_inflight = False
            QtWidgets.QMessageBox.warning(self, "음악 생성 오류", str(e))

    def on_click_analyze_music(self) -> None:
        """
        (단순화) 음악분석: UI에서 필요한 정보를 수집하여 audio_sync.py의 핵심 분석 함수를 호출하고
        결과를 다이얼로그로 표시합니다. 모든 복잡한 로직은 audio_sync 모듈로 이전되었습니다.
        """
        from pathlib import Path
        from PyQt5 import QtWidgets

        print("\n--- DEBUG: on_click_analyze_music 함수가 성공적으로 호출되었습니다. ---")

        # --- 1. UI 요소 및 경로 준비 ---
        btn = getattr(self, "btn_analyze", None)
        if isinstance(btn, QtWidgets.QAbstractButton):
            btn.setEnabled(False)

        try:
            from app.utils import load_json, run_job_with_progress_async
        except Exception:
            from utils import load_json, run_job_with_progress_async  # type: ignore

        try:
            proj_dir = self._current_project_dir()
        except Exception:
            proj_dir = None

        if not proj_dir:
            if isinstance(btn, QtWidgets.QAbstractButton):
                btn.setEnabled(True)
            QtWidgets.QMessageBox.critical(self, "음악분석 실패", "프로젝트 폴더가 없습니다.")
            return

        vocal_path_obj = self._find_latest_vocal()
        if not vocal_path_obj or not vocal_path_obj.exists():
            if isinstance(btn, QtWidgets.QAbstractButton):
                btn.setEnabled(True)
            QtWidgets.QMessageBox.critical(self, "음악분석 실패", "보컬 오디오 파일을 찾을 수 없습니다.")
            return

        pj_path = Path(proj_dir) / "project.json"
        lyrics_raw_text = ""
        if pj_path.exists():
            meta = load_json(pj_path, {}) or {}
            lyrics_raw_text = str(meta.get("lyrics") or "").strip()

        if not lyrics_raw_text:
            if isinstance(btn, QtWidgets.QAbstractButton):
                btn.setEnabled(True)
            QtWidgets.QMessageBox.critical(self, "음악분석 실패", "가사를 찾지 못했습니다.")
            return

        print(f"--- CHECKPOINT: 분석 시작 ({vocal_path_obj.name}) ---")

        # --- 2. 백그라운드 작업(job) 정의 ---
        def job(log_callback):
            """백그라운드 스레드에서 실행될 작업. 핵심 분석 함수를 호출합니다."""
            log_callback({"msg": "[FLOW] audio_sync.sync_lyrics_with_whisper_pro 호출 시작"})
            try:
                from app.audio_sync import sync_lyrics_with_whisper_pro
            except ImportError:
                from audio_sync import sync_lyrics_with_whisper_pro

            # 새로워진 단일 분석 함수 호출
            result_dict = sync_lyrics_with_whisper_pro(
                audio_path=str(vocal_path_obj),
                lyrics_text=lyrics_raw_text
            )
            log_callback({"msg": "[FLOW] 분석 완료. 결과 정리 중..."})
            return result_dict

        # --- 3. 작업 완료 후(done) 콜백 정의 ---
        def done(ok: bool, payload: dict, err):
            try:
                if not ok:
                    QtWidgets.QMessageBox.critical(self, "음악분석 실패", str(err))
                    return

                summary = (payload or {}).get("summary_text", "분석 결과 요약을 가져오지 못했습니다.")
                print("\n[음악분석 결과]\n" + summary, flush=True)

                # 결과 표시 다이얼로그
                dlg = QtWidgets.QDialog(self)
                dlg.setWindowTitle("음악분석 결과")
                dlg.resize(1000, 760)
                vbox = QtWidgets.QVBoxLayout(dlg)
                ed = QtWidgets.QPlainTextEdit()
                ed.setReadOnly(True)
                ed.setPlainText(summary)
                vbox.addWidget(ed)
                btn_close = QtWidgets.QPushButton("닫기")
                btn_close.clicked.connect(dlg.accept)
                row = QtWidgets.QHBoxLayout()
                row.addStretch(1)
                row.addWidget(btn_close)
                vbox.addLayout(row)
                dlg.exec_()
            finally:
                if isinstance(btn, QtWidgets.QAbstractButton):
                    btn.setEnabled(True)

        # --- 4. 비동기 작업 실행 ---
        run_job_with_progress_async(self, "음악분석", job, on_done=done)



    @staticmethod
    def _persist_lyric_sections(*, proj_dir: str, sections: list, last_end: float) -> None:
        """
        가사 섹션을 story.json, scene.json, project.json에 안전하게 반영한다.
        - story.json: lyric_sections[], last_lyric_end_sec, effective_duration(없으면), updated_at
        - scene.json: segments[] (start/end/label/text), duration(없으면), updated_at
        - project.json: custom_seconds(없으면만) 기록
        """
        from pathlib import Path
        import time

        try:
            try:
                from app.utils import load_json, save_json  # type: ignore
            except Exception:
                from utils import load_json, save_json  # type: ignore
        except Exception as exc:
            raise RuntimeError("utils.load_json/save_json 로드 실패") from exc

        base = Path(proj_dir)
        now_ts = int(time.time())

        # ---- story.json ----
        story_path = base / "story.json"
        story = load_json(story_path, {}) or {}
        if not isinstance(story, dict):
            story = {}

        safe_sections = []
        for s in sections or []:
            if not isinstance(s, dict):
                continue
            try:
                st = float(s.get("start", 0.0))
                et = float(s.get("end", 0.0))
            except (TypeError, ValueError):
                continue
            tx = str(s.get("text") or "")
            lb = str(s.get("label") or "")
            if et < st:
                et = st
            safe_sections.append({"start": max(0.0, st), "end": max(0.0, et), "text": tx, "label": lb})

        story["lyric_sections"] = safe_sections
        story["last_lyric_end_sec"] = float(max(0.0, float(last_end or 0.0)))
        if "effective_duration" not in story:
            story["effective_duration"] = float(max(story.get("last_lyric_end_sec", 0.0), 0.0))
        story["updated_at"] = now_ts
        save_json(story_path, story)

        # ---- scene.json ----
        scene_path = base / "scene.json"
        scene = load_json(scene_path, {}) or {}
        if not isinstance(scene, dict):
            scene = {}

        scene_segments = []
        for i, s in enumerate(story.get("lyric_sections", [])):
            scene_segments.append(
                {
                    "id": f"line_{i:03d}",
                    "start": float(s.get("start", 0.0)),
                    "end": float(s.get("end", 0.0)),
                    "label": str(s.get("label") or ""),
                    "text": str(s.get("text") or ""),
                }
            )
        scene["segments"] = scene_segments
        if "duration" not in scene:
            scene["duration"] = float(max(story.get("effective_duration", 0.0), story.get("last_lyric_end_sec", 0.0)))
        scene["updated_at"] = now_ts
        save_json(scene_path, scene)

        # ---- project.json (보수적 반영) ----
        proj_path = base / "project.json"
        proj = load_json(proj_path, {}) or {}
        if not isinstance(proj, dict):
            proj = {}
        if "custom_seconds" not in proj:
            try:
                eff = float(story.get("effective_duration", story.get("last_lyric_end_sec", 0.0)) or 0.0)
            except (TypeError, ValueError):
                eff = float(story.get("last_lyric_end_sec", 0.0) or 0.0)
            proj["custom_seconds"] = int(round(max(0.0, eff)))
            save_json(proj_path, proj)

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

    # shorts_ui.py 파일에서 이 함수를 찾아 아래 내용으로 전체를 교체하세요.
    def on_click_build_story_from_seg_async(self) -> None:
        """
        seg.json → story.json 스켈레톤 → video.json(갭 포함) 생성 → video.json만 AI 강화 → 프롬프트 주입 → 가사 재주입
        - [수정됨] _ask_wrapper_internal 에서 len() 타입 오류 수정 (str() 추가).
        - [수정됨] AI 클래스 임포트 시 ai_class 대신 AI 직접 사용.
        - [수정됨] 내부 _ask_wrapper_internal 에서 사용되지 않는 trace_to_progress_internal 함수 제거.
        - [수정됨] 예외 처리 구체화 (FileNotFoundError, TypeError, ValueError 등).
        - [수정됨] 변수명 명확화 및 가독성 개선 (예: path -> proj_dir_path).
        - story.json 생성 로직/포맷 100% 보존
        - 갭 씬은 프롬프트 비우고, 가사도 항상 빈 문자열 유지(반주/인터루드)
        - gap ID는 3자리(gap_###)이며, 직전 t_### 번호를 따름(예: t_007 → gap_007)
        - 로그와 파일의 provider가 실제 사용 제공자와 일치(gemini/openai)
        """
        from pathlib import Path
        from typing import Any, Dict, List, Optional, Callable
        from PyQt5 import QtWidgets
        import json
        import re
        import inspect  # _ask_wrapper_internal 시그니처 검사용
        import traceback  # done 콜백 에러 로깅용

        # 진행기
        try:
            from app.utils import run_job_with_progress_async
        except ImportError:
            try:
                from utils import run_job_with_progress_async  # type: ignore
            except ImportError as import_err:
                QtWidgets.QMessageBox.critical(self, "오류", f"진행률 표시기 로드 실패: {import_err}")
                return

        # 재진입 방지
        # _seg_story_busy는 __init__에서 False로 초기화되어야 합니다.
        if getattr(self, "_seg_story_busy", False):
            QtWidgets.QMessageBox.information(self, "알림", "프로젝트 분석 작업이 이미 진행 중입니다.")
            return
        self._seg_story_busy = True

        # 버튼 비활성화
        btn_build_story = getattr(self, "btn_test1_story", None) or getattr(getattr(self, "ui", None),
                                                                            "btn_test1_story", None)
        if btn_build_story is not None:
            try:
                btn_build_story.setEnabled(False)
            except RuntimeError:  # 위젯이 삭제된 경우 등
                pass

        # 유틸/AI/강화기 로드 (구체적인 예외 처리 추가)
        try:
            try:
                from app.utils import load_json, save_json, save_story_overwrite_with_prompts
            except ImportError:
                from utils import load_json, save_json, save_story_overwrite_with_prompts  # type: ignore

            try:
                from app.story_enrich import apply_gpt_to_story_v11
            except ImportError:
                from story_enrich import apply_gpt_to_story_v11  # type: ignore

            try:
                from app.video_build import build_video_json_with_gap_policy
            except ImportError:
                from video_build import build_video_json_with_gap_policy  # type: ignore

            try:
                # AI 클래스 직접 사용 (ai_class 별칭 대신)
                from app.ai import AI  #
            except ImportError:
                try:
                    from ai import AI  # type: ignore #
                except ImportError:
                    # utils.AI가 최종 폴백인지 확인 필요, 없다면 NameError 발생 가능
                    from utils import AI  # type: ignore

        except ImportError as import_err_libs:
            QtWidgets.QMessageBox.critical(self, "오류", f"필수 라이브러리 로드 실패: {import_err_libs}")
            self._seg_story_busy = False
            if btn_build_story is not None:
                try:
                    btn_build_story.setEnabled(True)
                except RuntimeError:
                    pass
            return
        except NameError as name_err:  # utils.AI 폴백 실패 시
            QtWidgets.QMessageBox.critical(self, "오류", f"AI 클래스 로드 실패: {name_err}")
            self._seg_story_busy = False
            if btn_build_story is not None:
                try:
                    btn_build_story.setEnabled(True)
                except RuntimeError:
                    pass
            return

        # 프로젝트 경로 확인 (더 명확한 변수명 사용)
        proj_dir_str = ""
        try:
            # _current_project_dir 메서드가 Path 객체 또는 str을 반환한다고 가정
            proj_dir_callable = getattr(self, "_current_project_dir", None)  #
            if callable(proj_dir_callable):
                proj_dir_val = proj_dir_callable()
                # Path 객체면 문자열로 변환
                proj_dir_str = str(proj_dir_val) if isinstance(proj_dir_val, Path) else proj_dir_val or ""
            else:
                # project_dir 속성이 Path 또는 str이라고 가정
                proj_dir_attr = proj_dir_callable or getattr(self, "project_dir", None)  #
                proj_dir_str = str(proj_dir_attr) if isinstance(proj_dir_attr, Path) else proj_dir_attr or ""

            if not proj_dir_str:
                raise ValueError("프로젝트 경로를 가져올 수 없습니다.")  # 명시적 오류 타입

        except AttributeError:  # _current_project_dir, project_dir 둘 다 없는 경우
            proj_dir_str = ""  # 기본값 설정
        except (ValueError, TypeError, OSError) as e_proj_dir:  # 경로 관련 일반 오류 처리
            print(f"[WARN] 프로젝트 경로 확인 중 오류: {type(e_proj_dir).__name__}: {e_proj_dir}")
            proj_dir_str = ""  # 기본값 설정
        except Exception as e_unknown:  # 예상치 못한 오류
            print(f"[ERROR] 프로젝트 경로 확인 중 예상치 못한 오류: {type(e_unknown).__name__}: {e_unknown}")
            proj_dir_str = ""  # 기본값 설정

        if not proj_dir_str:
            QtWidgets.QMessageBox.warning(self, "오류", "프로젝트 폴더를 선택하거나 생성해 주세요.")
            self._seg_story_busy = False
            if btn_build_story is not None:
                try:
                    btn_build_story.setEnabled(True)
                except RuntimeError:
                    pass
            return

        # 경로 객체 생성 및 seg.json 확인 (구체적인 예외 처리)
        try:
            proj_dir_path = Path(proj_dir_str).resolve()  # 절대 경로로 변환
            seg_json_path = proj_dir_path / "seg.json"  #
            if not seg_json_path.is_file():  # exists() 대신 is_file()로 더 명확하게 확인
                raise FileNotFoundError(f"seg.json 파일을 찾을 수 없습니다:\n{seg_json_path}")
        except FileNotFoundError as e_fnf:
            QtWidgets.QMessageBox.warning(self, "오류", str(e_fnf))
            self._seg_story_busy = False
            if btn_build_story is not None:
                try:
                    btn_build_story.setEnabled(True)
                except RuntimeError:
                    pass
            return
        except OSError as e_os:  # 경로 관련 OS 오류 (권한 등)
            QtWidgets.QMessageBox.critical(self, "오류", f"경로 접근 오류: {e_os}")
            self._seg_story_busy = False
            if btn_build_story is not None:
                try:
                    btn_build_story.setEnabled(True)
                except RuntimeError:
                    pass
            return

        # seg 원문 가사 재주입 함수 (내부 함수로 정의)
        def _reinject_lyrics_from_seg(target_json_path: Path, seg_file_path: Path,
                                      log_fn: Callable[[str], None]) -> None:  #
            try:
                target_doc = load_json(target_json_path, None)  #
                if not isinstance(target_doc, dict):
                    log_fn("[WARN] 가사 재주입 실패: 대상 JSON 형식이 dict가 아닙니다.")
                    return

                seg_data_raw = load_json(seg_file_path, None)  #
                seg_data: List[Dict[str, Any]]
                # seg.json이 리스트가 아닐 경우 처리
                if not isinstance(seg_data_raw, list):
                    log_fn("[WARN] 가사 재주입 실패: seg.json 형식이 리스트가 아닙니다.")
                    seg_data = []  # 빈 리스트로 초기화
                else:
                    seg_data = seg_data_raw  # 타입 검증 후 할당

                scenes_raw = target_doc.get("scenes")  #
                if not isinstance(scenes_raw, list):
                    log_fn("[WARN] 가사 재주입 실패: 대상 JSON에 'scenes' 리스트가 없습니다.")
                    return
                scenes: List[Dict[str, Any]] = scenes_raw

                seg_map: Dict[str, str] = {}
                for seg_item in seg_data:
                    # seg_item이 dict인지, start 키를 포함하는지 확인
                    if isinstance(seg_item, dict) and "start" in seg_item:
                        try:
                            # start 값을 float으로 변환 시도
                            start_val = float(seg_item.get("start", 0.0))  #
                        except (TypeError, ValueError):
                            start_val = 0.0  # 변환 실패 시 0.0으로 설정
                        start_key = f"{start_val:.3f}"  # 소수점 3자리 문자열 키 생성
                        text_val = str(seg_item.get("text", "")).strip()  # text 값 가져오기 #
                        if text_val:  # 빈 문자열이 아니면 맵에 추가
                            seg_map[start_key] = text_val
                    else:
                        log_fn("[WARN] seg.json 내 항목 형식이 잘못되었습니다 (dict 아니거나 'start' 없음).")

                reinjected_count = 0
                cleared_gap_count = 0

                for sc_item in scenes:
                    if not isinstance(sc_item, dict):
                        continue
                    sc_id_str = str(sc_item.get("id") or "")  #
                    sc_origin_str = str(sc_item.get("origin") or "")  #
                    is_gap_scene = sc_id_str.startswith("gap_") or sc_origin_str == "gap-fill"  #

                    if is_gap_scene:
                        # 갭 씬이면 가사를 빈 문자열로 강제
                        if sc_item.get("lyric"):  # 기존 가사가 있을 경우에만 #
                            sc_item["lyric"] = ""  #
                            cleared_gap_count += 1
                        continue  # 다음 씬으로

                    # 갭 씬이 아닐 경우 가사 재주입 시도
                    try:
                        # start 값을 float으로 변환 시도
                        sc_start_val = float(sc_item.get("start", 0.0))  #
                    except (TypeError, ValueError):
                        sc_start_val = 0.0  # 실패 시 0.0
                    sc_key = f"{sc_start_val:.3f}"  # 검색용 키 생성

                    # seg_map에 해당 키가 있고, 가사가 다르다면 업데이트
                    if sc_key in seg_map:
                        original_text = seg_map[sc_key]
                        if original_text and sc_item.get("lyric", "") != original_text:  #
                            sc_item["lyric"] = original_text  #
                            reinjected_count += 1

                # 변경 사항이 있을 경우 파일 저장
                if reinjected_count > 0 or cleared_gap_count > 0:
                    target_doc["scenes"] = scenes  # 업데이트된 scenes 리스트 반영
                    try:
                        save_json(target_json_path, target_doc)  # 파일 저장 #
                        log_fn(f"[INFO] 가사 재주입 완료: 정상 씬 복원 {reinjected_count}개, 갭 씬 정리 {cleared_gap_count}개")
                    except (OSError, TypeError) as e_save_reinject:  # 파일 저장 오류 처리
                        log_fn(f"[ERROR] 가사 재주입 저장 오류: {type(e_save_reinject).__name__}: {e_save_reinject}")
                else:
                    log_fn("[INFO] 가사 재주입: 변경 없음")  # 변경 없을 경우 로그

            # 파일/JSON 관련 오류 처리 추가
            except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError) as e_reinject:
                log_fn(f"[ERROR] 가사 재주입 오류: {type(e_reinject).__name__}: {e_reinject}")
            except Exception as e_unknown_reinject:  # 예상치 못한 오류
                log_fn(f"[ERROR] 가사 재주입 중 예상치 못한 오류: {type(e_unknown_reinject).__name__}: {e_unknown_reinject}")

        # ----- job 함수 정의 -----
        # job 함수 정의 (가독성 개선 및 오류 처리 강화)
        def job(on_progress_callback: Callable[[dict], None]) -> Dict[str, str]:  #
            from typing import List, Dict, Any  # job 함수 내에서만 사용

            def _log_progress(message: str) -> None:
                """진행률 콜백을 호출하는 내부 헬퍼 함수."""
                try:
                    on_progress_callback({"msg": message})  #
                except Exception as e_log:
                    # 로그 콜백 자체에서 오류가 나도 작업은 계속 진행
                    print(f"[ERROR] Progress callback failed: {e_log}")

            # UI에서 prefer 감지 함수 (오류 처리 추가)
            def _detect_prefer_from_ui() -> str:  #
                default_prefer = "gemini"  # 기본값 설정
                widget_name_list = [
                    "ai_mode", "mode", "cmb_ai", "combo_ai", "cmb_model", "combo_model",
                    "model", "provider", "combo_provider"
                ]
                for widget_name in widget_name_list:
                    try:
                        # self 또는 self.ui에서 위젯 찾기 시도
                        widget_obj = getattr(self, widget_name, None) or \
                                     getattr(getattr(self, "ui", None), widget_name, None)
                        current_text_value = ""
                        # 위젯 타입에 따라 텍스트 가져오기
                        if hasattr(widget_obj, "currentText") and callable(widget_obj.currentText):  #
                            current_text_value = str(widget_obj.currentText())
                        elif isinstance(widget_obj, str):  # 위젯이 아닌 문자열 속성일 경우
                            current_text_value = widget_obj

                        # 텍스트 값으로 provider 결정
                        if current_text_value:
                            text_lower = current_text_value.lower()
                            if "gemini" in text_lower: return "gemini"
                            if "openai" in text_lower or "gpt" in text_lower: return "openai"
                    except AttributeError:  # 위젯 속성이 없는 경우
                        pass
                    except Exception as e_detect:  # 기타 예상치 못한 오류
                        _log_progress(f"[WARN] AI 선호도 감지 중 오류 ({widget_name}): {e_detect}")

                # AI 클래스 인스턴스의 기본값 사용 시도
                try:
                    # AI 클래스 직접 사용 (ai_class 대신)
                    ai_instance = AI()  #
                    ai_default_prefer = getattr(ai_instance, "default_prefer", None)  #
                    if isinstance(ai_default_prefer, str) and ai_default_prefer:
                        return ai_default_prefer.lower()
                except Exception as e_ai_init:  # AI 클래스 초기화 실패 등
                    _log_progress(f"[WARN] AI 기본 선호도 가져오기 실패: {e_ai_init}")

                return default_prefer  # 최종적으로 기본값 반환

            # 1) story.json 스켈레톤 생성 함수 (오류 처리 강화)
            def _build_story_skeleton_internal(seg_doc_list: List[Dict[str, Any]], project_dir: Path) -> Path:  #
                _log_progress("[1/5] story.json 스켈레톤 생성 시작...")
                # seg.json 내용 검증
                if not isinstance(seg_doc_list, list) or not seg_doc_list:
                    raise ValueError("seg.json 내용이 비어있거나 리스트 형식이 아닙니다.")

                # project.json 로드 (오류 처리 포함)
                meta_path = project_dir / "project.json"  #
                meta_info: Dict[str, Any] = {}
                try:
                    meta_info_raw = load_json(meta_path, None)  #
                    if isinstance(meta_info_raw, dict):
                        meta_info = meta_info_raw
                    else:
                        _log_progress("[WARN] project.json 형식이 잘못되었습니다 (빈 메타 사용)")
                except FileNotFoundError:
                    _log_progress("[WARN] project.json 파일을 찾을 수 없습니다 (빈 메타 사용)")
                except (json.JSONDecodeError, OSError) as e_load_meta:
                    _log_progress(f"[WARN] project.json 로드 실패 ({type(e_load_meta).__name__}, 빈 메타 사용)")

                # 이미지 폴더 생성 (오류 처리 포함)
                imgs_dir = project_dir / "imgs"  #
                try:
                    imgs_dir.mkdir(parents=True, exist_ok=True)  #
                except OSError as e_mkdir:
                    _log_progress(f"[WARN] 이미지 폴더 생성 실패: {e_mkdir}")
                    # 폴더 생성 실패해도 계속 진행 시도

                scenes_list: List[Dict[str, Any]] = []
                cursor_time = 0.0
                first_start_time = 0.0

                # 첫 세그먼트 시작 시간 확인 (타입 검증 강화)
                if seg_doc_list and isinstance(seg_doc_list[0], dict):
                    try:
                        first_start_time = float(seg_doc_list[0].get("start", 0.0))  #
                    except (ValueError, TypeError):
                        _log_progress("[WARN] 첫 세그먼트 시작 시간 변환 실패, 0.0 사용")
                        first_start_time = 0.0
                else:
                    _log_progress("[WARN] seg.json 첫 항목이 dict가 아니거나 비어있음")

                # 인트로 씬 추가 로직 (조건: 첫 시작 시간이 1초 이상)
                if first_start_time >= 1.0:  #
                    intro_end_time = round(first_start_time, 3)
                    intro_scene_id = "t_000"  # 인트로 씬 ID 고정
                    scenes_list.append({
                        "id": intro_scene_id, "section": "intro", "label": "Intro",  #
                        "start": 0.0, "end": intro_end_time, "duration": intro_end_time,  #
                        "scene": "intro", "characters": [],  #
                        "effect": ["soft light", "film grain", "gentle camera pan"],  #
                        "screen_transition": True,  #
                        "img_file": str(imgs_dir / f"{intro_scene_id}.png"),  # 경로 생성 #
                        "prompt": "인트로: 부드러운 전환과 분위기 암시",  # 기본 프롬프트 #
                        # 기본값 필드 추가 (v1.1 스키마 호환)
                        "prompt_img": "", "prompt_movie": "", "prompt_negative": "",  #
                        "lyric": "",  #
                    })
                    cursor_time = first_start_time  # 커서 시간 업데이트

                # 메인 씬 생성 루프 (타입 검증 및 오류 처리 강화)
                for i_seg, row_data in enumerate(seg_doc_list, start=1):
                    if not isinstance(row_data, dict):
                        _log_progress(f"[WARN] seg.json {i_seg}번째 항목이 dict가 아님. 건너뜀")
                        continue

                    try:
                        start_time_val = float(row_data.get("start", 0.0))  #
                    except (ValueError, TypeError):
                        start_time_val = 0.0
                    try:
                        end_time_val = float(row_data.get("end", start_time_val))  #
                    except (ValueError, TypeError):
                        end_time_val = start_time_val

                    # 시간 역전 및 커서 위치 보정
                    if end_time_val < start_time_val: end_time_val = start_time_val  #
                    if start_time_val < cursor_time: start_time_val = cursor_time  #

                    scene_id_str = f"t_{i_seg:03d}"  #
                    duration_val = round(max(0.0, end_time_val - start_time_val), 3)  #
                    lyric_text_val = str(row_data.get("text", "")).strip()  #

                    # 기본 캐릭터 설정 (예: female_01)
                    default_character = ["female_01"]  #
                    # project.json에 characters 정보가 있으면 사용
                    project_characters = meta_info.get("characters")  #
                    characters_to_use = project_characters if isinstance(project_characters,
                                                                         list) and project_characters else default_character

                    scenes_list.append({
                        "id": scene_id_str, "section": "verse", "label": "Verse",  # 기본값 verse #
                        "start": round(start_time_val, 3), "end": round(end_time_val, 3), "duration": duration_val,  #
                        "scene": "verse",  # scene 필드 추가 (기본값 verse) #
                        "characters": characters_to_use,  # 캐릭터 정보 사용 #
                        "effect": ["bokeh", "slow push-in"],  # 기본 효과 #
                        "screen_transition": False,  # 기본값 False #
                        "img_file": str(imgs_dir / f"{scene_id_str}.png"),  #
                        "prompt": "장면 묘사 필요",  # 기본 프롬프트 #
                        # 기본값 필드 추가
                        "prompt_img": "", "prompt_movie": "", "prompt_negative": "",  #
                        "lyric": lyric_text_val,  # 가사 정보 추가 #
                    })
                    cursor_time = end_time_val  # 커서 업데이트

                last_end_time = cursor_time  # 마지막 씬 종료 시간

                # 아웃트로 씬 추가 로직 (고정 길이 5초)
                outro_start_time = last_end_time
                outro_duration = 5.0  # 아웃트로 길이 고정
                outro_end_time = outro_start_time + outro_duration
                # 아웃트로 씬 ID 결정 (마지막 씬 번호 + 1)
                outro_scene_index = len(seg_doc_list) + (1 if first_start_time >= 1.0 else 0) + 1
                outro_id_str = f"t_{outro_scene_index:03d}"  #

                scenes_list.append({
                    "id": outro_id_str, "section": "outro", "label": "Outro",  #
                    "start": round(outro_start_time, 3), "end": round(outro_end_time, 3),  #
                    "duration": round(outro_duration, 3),  #
                    "scene": "outro", "characters": [],  # 아웃트로는 캐릭터 없음 #
                    "effect": ["fade-out", "soft glow"],  # 아웃트로 효과 #
                    "screen_transition": True,  # 전환 효과 #
                    "img_file": str(imgs_dir / f"{outro_id_str}.png"),  #
                    "prompt": "아웃트로: 여운과 마무리",  #
                    "prompt_img": "", "prompt_movie": "", "prompt_negative": "",  #
                    "lyric": "",  # 아웃트로는 가사 없음 #
                })

                # story.json 최종 구조 생성
                story_record: Dict[str, Any] = {
                    # title: project.json > 폴더명 순서
                    "title": str(meta_info.get("title") or project_dir.name),  #
                    # duration: 아웃트로 종료 시간 기준
                    "duration": round(outro_end_time, 3),  #
                    "offset": 0.0,  # 기본 오프셋
                    # lyrics: project.json > 빈 문자열
                    "lyrics": str(meta_info.get("lyrics") or ""),  #
                    # characters: project.json > 기본값
                    "characters": meta_info.get("characters") or ["female_01"],  #
                    # character_styles: project.json > 빈 dict
                    "character_styles": meta_info.get("character_styles") or {},  #
                    # global_context: project.json > 기본값 (모든 키 포함)
                    "global_context": meta_info.get("global_context") or {  #
                        "themes": [], "palette": "", "style_guide": "", "negative_bank": "",  #
                        "section_moods": {"intro": "", "verse": "", "chorus": "", "bridge": "", "outro": ""},  #
                        "effect": [],  #
                    },
                    # defaults: project.json > 기본값 (image 키 포함)
                    "defaults": meta_info.get("defaults") or {  #
                        "image": {"width": 720, "height": 1080, "negative": "@global"}  #
                    },
                    "scenes": scenes_list,  # 생성된 씬 리스트
                    # audit 정보 추가
                    "audit": {"built_from": "seg.json", "ai_used": False},  #
                }

                # story.json 파일 저장 (오류 처리 포함)
                story_file_path = project_dir / "story.json"
                try:
                    save_json(story_file_path, story_record)  #
                    _log_progress(f"[1/5] story.json 스켈레톤 생성 완료: {story_file_path.name}")
                except (OSError, TypeError) as e_save_story:
                    _log_progress(f"[ERROR] story.json 저장 실패: {e_save_story}")
                    # 저장 실패 시에도 경로 객체는 반환 (후속 단계에서 사용 시도 가능)
                return story_file_path  # 저장된 경로 반환

            # --- job 함수 메인 로직 시작 ---
            on_progress_callback({"stage": "start", "msg": f"프로젝트 분석 시작: {proj_dir_path.name}"})  #

            # seg.json 로드 (오류 처리 포함)
            seg_content_raw: Optional[list] = None
            try:
                seg_content_raw = load_json(seg_json_path, None)  #
                if not isinstance(seg_content_raw, list):
                    raise ValueError(f"seg.json 파일 내용이 리스트 형식이 아닙니다: {seg_json_path}")
            except FileNotFoundError:
                raise FileNotFoundError(f"seg.json 파일을 찾을 수 없습니다: {seg_json_path}")
            except (json.JSONDecodeError, ValueError) as e_load_seg:
                raise ValueError(f"seg.json 로드 또는 파싱 실패: {e_load_seg}")
            except OSError as e_os_seg:
                raise OSError(f"seg.json 접근 오류: {e_os_seg}")

            # seg_content 타입 검증 후 할당
            seg_content: List[Dict[str, Any]] = seg_content_raw if seg_content_raw is not None else []

            # 1) story.json 스켈레톤 생성 (내부 오류는 함수 안에서 처리)
            try:
                story_path = _build_story_skeleton_internal(seg_content, proj_dir_path)  #
            except Exception as e_build_skel:  # 스켈레톤 생성 실패 시 작업 중단
                raise RuntimeError(f"story.json 스켈레톤 생성 실패: {e_build_skel}") from e_build_skel

            # 2) video.json(갭 포함) 생성 (오류 처리 강화)
            _log_progress("[2/5] video.json(갭 포함) 생성...")  #
            video_path_str: Optional[str] = None  # Optional[str]로 타입 명시
            try:
                # build_video_json_with_gap_policy가 str을 반환한다고 가정
                video_path_result = build_video_json_with_gap_policy(str(proj_dir_path), small_gap_sec=2.0)  #
                if isinstance(video_path_result, str) and Path(video_path_result).is_file():
                    video_path_str = video_path_result
                    on_progress_callback(
                        {"stage": "video_json_built", "msg": f"video.json 생성 완료: {Path(video_path_str).name}"})  #
                else:
                    _log_progress("[WARN] video.json 생성 함수가 유효한 경로를 반환하지 않았습니다.")
            except FileNotFoundError:  # build_video_json_with_gap_policy 내부에서 story.json 못 찾는 경우
                _log_progress("[WARN] video.json 생성 건너뜀 (story.json 없음)")
            except (TypeError, ValueError) as e_v:  # 함수 인자 오류 또는 내부 로직 오류
                _log_progress(f"[WARN] video.json 생성 중 오류 발생: {type(e_v).__name__}: {e_v}")
            except Exception as e_v_unknown:  # 예상치 못한 오류
                _log_progress(f"[ERROR] video.json 생성 중 예상치 못한 오류: {type(e_v_unknown).__name__}: {e_v_unknown}")

            # 3) AI 강화 ( video.json이 성공적으로 생성되었을 경우에만 시도 )
            ai_client_instance: Optional[AI] = None  # Optional 타입 명시 (AI 클래스 직접 사용)
            try:
                ai_client_instance = AI()  # AI 클라이언트 초기화 #
            except Exception as e_ai_init_job:  # AI 클래스 초기화 실패
                _log_progress(f"[WARN] AI 클라이언트 초기화 실패 ({type(e_ai_init_job).__name__}), AI 강화 생략.")

            prefer_detected_provider = _detect_prefer_from_ui()  # AI 선호도 감지 #

            # _ask_wrapper_internal 내부 함수 (AI 호출 담당)
            def _ask_wrapper_internal(
                    system: str,
                    user: str,
                    *,
                    prefer: Optional[str] = None,
                    allow_fallback: Optional[bool] = None,
                    trace: Optional[Callable[[str, str], None]] = None,
            ) -> str:
                """
                내부 AI 호출 래퍼.
                - UI에서 trace= 를 넘겨줘도 안전하게 수용.
                - ai_client_instance.ask_smart 의 시그니처를 점검한 뒤, 지원하는 키워드만 전달.
                - 기존 동작(선호 provider, 폴백 허용) 100% 유지.
                """
                # 필수 전제: 바깥 스코프에 ai_client_instance, prefer_detected_provider, inspect, Any, Dict, Optional, Callable 존재
                if ai_client_instance is None:
                    raise RuntimeError("AI 클라이언트가 초기화되지 않아 호출할 수 없습니다.")

                # prefer/allow_fallback 해석
                prefer_final = prefer or prefer_detected_provider or getattr(ai_client_instance, "default_prefer",
                                                                             "openai")
                allow_fallback_final = True if allow_fallback is None else bool(allow_fallback)

                # ask_smart 존재/시그니처 확인
                ask_smart = getattr(ai_client_instance, "ask_smart", None)
                if not callable(ask_smart):
                    raise AttributeError("ask_smart method not found")

                sig = inspect.signature(ask_smart)
                call_kwargs: Dict[str, Any] = {}
                if "prefer" in sig.parameters:
                    call_kwargs["prefer"] = prefer_final
                if "allow_fallback" in sig.parameters:
                    call_kwargs["allow_fallback"] = allow_fallback_final
                # ask_smart 가 trace 파라미터를 지원하면 그대로 전달 (선택적)
                if "trace" in sig.parameters and trace is not None:
                    call_kwargs["trace"] = trace

                # trace 인자가 여기서 사용되지 않는다는 경고 방지 및 간단한 진단 로그 전달(선택적)
                if trace:
                    try:
                        # 길이 정보만 간단히 넘김 (민감한 프롬프트 내용 노출 방지)
                        trace("ai.ask", f"prefer={prefer_final} allow_fallback={allow_fallback_final} "
                                        f"len={len(str(system))}/{len(str(user))}")
                    except Exception:
                        pass

                # 실제 호출
                result_text = ask_smart(str(system), str(user), **call_kwargs)
                return str(result_text or "")

            # video.json AI 강화 실행 (video_path_str 유효성 검증 후)
            if video_path_str and Path(video_path_str).is_file():
                _log_progress("[3/5] video.json AI 강화 시작...")  #
                try:
                    video_doc_raw = load_json(Path(video_path_str), None)  #
                    if isinstance(video_doc_raw, dict):
                        # apply_gpt_to_story_v11 호출 (ask 래퍼 사용)
                        video_ai_enhanced = apply_gpt_to_story_v11(  #
                            video_doc_raw, ask=_ask_wrapper_internal,  # 수정된 래퍼 사용
                            prefer=prefer_detected_provider,  # 감지된 provider 전달
                            allow_fallback=None,  # 기본값 사용
                            trace=None  # trace 인자 제거
                        )
                        # 감사 정보 업데이트
                        video_ai_enhanced.setdefault("audit", {})  #
                        video_ai_enhanced["audit"]["ai_used"] = True  #
                        # provider는 _ask_wrapper_internal 에서 로그로 남기고 여기서도 기록
                        video_ai_enhanced["audit"]["provider"] = prefer_detected_provider  #
                        # 파일 저장
                        save_json(Path(video_path_str), video_ai_enhanced)  #
                        on_progress_callback({"stage": "ai_done_video",
                                              "msg": f"video.json AI 강화 완료 (provider: {prefer_detected_provider})"})  #
                    else:
                        _log_progress("[WARN] video.json 형식이 dict가 아니라서 AI 강화 생략")
                # AI 강화 중 발생 가능한 특정 예외들 처리
                except (RuntimeError, TypeError, ValueError) as e_ai_vid:
                    _log_progress(f"[WARN] video.json AI 강화 중 오류(진행 계속): {type(e_ai_vid).__name__}: {e_ai_vid}")
                    on_progress_callback({"stage": "ai_error_video", "msg": f"video.json AI 강화 실패: {e_ai_vid}"})  #
                # 예상치 못한 오류 처리
                except Exception as e_ai_unknown:
                    _log_progress(
                        f"[ERROR] video.json AI 강화 중 예상치 못한 오류: {type(e_ai_unknown).__name__}: {e_ai_unknown}")
                    on_progress_callback({"stage": "ai_error_video", "msg": f"video.json AI 강화 실패: {e_ai_unknown}"})
            elif video_path_str:  # 경로 문자열은 있으나 파일이 아닌 경우
                _log_progress("[WARN] video.json 경로가 유효한 파일이 아니라 AI 강화 생략")
            else:  # 경로 문자열 자체가 없는 경우
                _log_progress("[WARN] video.json 경로가 없어 AI 강화 생략")

            # 4) 프롬프트 주입/정규화 + 네거티브 쉼표 정리 (video_path_str 유효성 검증 후)
            if video_path_str and Path(video_path_str).is_file():
                _log_progress("[4/5] video.json 프롬프트 주입/정규화...")  #
                try:
                    # 프롬프트 덮어쓰기 함수 호출
                    save_story_overwrite_with_prompts(Path(video_path_str))  #

                    # 네거티브 프롬프트 정리 로직 (오류 처리 강화)
                    try:
                        doc_norm = load_json(Path(video_path_str), None)  #
                        if isinstance(doc_norm, dict):
                            scenes_norm_raw = doc_norm.get("scenes")  #
                            if isinstance(scenes_norm_raw, list):
                                scenes_norm: List[Dict[str, Any]] = scenes_norm_raw  # 타입 검증 후 할당
                                neg_changed_count = 0  # 변경 카운트 초기화
                                for sc_norm in scenes_norm:
                                    if not isinstance(sc_norm, dict): continue  # dict 아니면 건너뜀

                                    neg_raw = str(sc_norm.get("prompt_negative", "")).strip()  #
                                    if neg_raw:
                                        # 전체 폭 쉼표(，)를 반각 쉼표(,)로 변환
                                        neg_clean = neg_raw.replace("，", ",")  #
                                        # 연속 공백 및 쉼표 주변 공백 정리
                                        neg_clean = re.sub(r"\s*,\s*", ",", neg_clean).strip(',')  #
                                        # 쉼표 기준으로 분리 후 빈 문자열 제거 및 공백 제거
                                        parts_list = [p.strip() for p in neg_clean.split(",") if p.strip()]  #
                                        # 중복 제거 (순서 유지)
                                        seen_set: set[str] = set()  # 타입 명시
                                        uniq_list: list[str] = []  # 타입 명시
                                        for ptxt in parts_list:
                                            # 소문자 기준으로 중복 체크
                                            ptxt_lower = ptxt.lower()
                                            if ptxt_lower not in seen_set:
                                                seen_set.add(ptxt_lower)
                                                uniq_list.append(ptxt)  # 원본 대소문자 유지
                                        # 최종 문자열 생성 및 업데이트
                                        final_neg = ", ".join(uniq_list)  #
                                        if sc_norm.get("prompt_negative") != final_neg:  #
                                            sc_norm["prompt_negative"] = final_neg  #
                                            neg_changed_count += 1  # 변경 시 카운트 증가

                                # 변경 사항이 있을 경우에만 파일 저장
                                if neg_changed_count > 0:
                                    save_json(Path(video_path_str), doc_norm)  #
                                    _log_progress(f"[INFO] 네거티브 프롬프트 정리 완료 ({neg_changed_count}개 씬 수정됨)")
                                # else:
                                #      _log_progress("[INFO] 네거티브 프롬프트 정리: 변경 없음") # 로그 간소화

                            else:  # scenes가 리스트가 아닌 경우
                                _log_progress("[WARN] 네거티브 정리 실패: 'scenes'가 리스트가 아님")
                        else:  # doc_norm이 dict가 아닌 경우
                            _log_progress("[WARN] 네거티브 정리 실패: video.json 형식이 dict가 아님")
                    # 네거티브 정리 중 발생 가능한 오류 처리
                    except (OSError, ValueError, TypeError, json.JSONDecodeError) as e_norm:
                        _log_progress(f"[WARN] 네거티브 정리 중 경고: {type(e_norm).__name__}: {e_norm}")
                    except Exception as e_norm_unknown:  # 예상치 못한 오류
                        _log_progress(f"[ERROR] 네거티브 정리 중 예상치 못한 오류: {type(e_norm_unknown).__name__}: {e_norm_unknown}")

                    on_progress_callback({"stage": "prompt_inject_done_video", "msg": "video.json 프롬프트 주입 완료"})  #
                # 프롬프트 주입 함수 자체에서 발생 가능한 오류 처리
                except (RuntimeError, ValueError, TypeError) as e_prompt_vid:
                    _log_progress(f"[WARN] video.json 프롬프트 주입 오류: {type(e_prompt_vid).__name__}: {e_prompt_vid}")
                    on_progress_callback(
                        {"stage": "prompt_inject_error_video", "msg": f"video.json 프롬프트 주입 실패: {e_prompt_vid}"})  #
                # 예상치 못한 오류 처리
                except Exception as e_prompt_unknown:
                    _log_progress(
                        f"[ERROR] video.json 프롬프트 주입 중 예상치 못한 오류: {type(e_prompt_unknown).__name__}: {e_prompt_unknown}")
                    on_progress_callback(
                        {"stage": "prompt_inject_error_video", "msg": f"video.json 프롬프트 주입 실패: {e_prompt_unknown}"})
            # video_path_str이 유효하지 않은 경우 로그
            elif video_path_str:
                _log_progress("[WARN] video.json 경로가 유효한 파일이 아니라 프롬프트 주입 생략")
            else:
                _log_progress("[WARN] video.json 경로가 없어 프롬프트 주입 생략")

            # 5) video.json 가사 재주입(갭 씬 정리 포함) (video_path_str 유효성 검증 후)
            if video_path_str and Path(video_path_str).is_file():
                _log_progress("[5/5] video.json 가사 재주입...")  #
                try:
                    # 내부 함수 _reinject_lyrics_from_seg 호출
                    _reinject_lyrics_from_seg(Path(video_path_str), seg_json_path, _log_progress)  #
                    # 성공 로그는 _reinject_lyrics_from_seg 내부에서 호출됨
                    on_progress_callback({"stage": "lyric_reinject_done_video", "msg": "video.json 가사 재주입 시도 완료"})  #
                # 가사 재주입 중 발생 가능한 오류 처리 (함수 내부 오류 포함)
                except (FileNotFoundError, OSError, TypeError, ValueError) as e_reinject_call:
                    _log_progress(f"[ERROR] video.json 가사 재주입 오류: {type(e_reinject_call).__name__}: {e_reinject_call}")
                    on_progress_callback(
                        {"stage": "lyric_reinject_error_video", "msg": f"video.json 가사 재주입 오류: {e_reinject_call}"})  #
                # 예상치 못한 오류 처리
                except Exception as e_reinject_unknown:
                    _log_progress(
                        f"[ERROR] video.json 가사 재주입 중 예상치 못한 오류: {type(e_reinject_unknown).__name__}: {e_reinject_unknown}")
                    on_progress_callback(
                        {"stage": "lyric_reinject_error_video", "msg": f"video.json 가사 재주입 오류: {e_reinject_unknown}"})
            # video_path_str이 유효하지 않은 경우 로그
            elif video_path_str:
                _log_progress("[WARN] video.json 경로가 유효한 파일이 아니라 가사 재주입 생략")
            else:
                _log_progress("[WARN] video.json 경로가 없어 가사 재주입 생략")

            # (최종) provider 고정 (video_path_str 유효성 검증 후)
            if video_path_str and Path(video_path_str).is_file():
                try:
                    vid_doc_final = load_json(Path(video_path_str), None)  #
                    if isinstance(vid_doc_final, dict):
                        # audit 딕셔너리 가져오기 (없으면 생성)
                        aud_final = vid_doc_final.setdefault("audit", {})  #
                        # provider 키 업데이트
                        aud_final["provider"] = prefer_detected_provider  #
                        # 파일 저장
                        save_json(Path(video_path_str), vid_doc_final)  #
                    else:  # dict 아니면 저장 안 함
                        _log_progress("[WARN] provider 고정 실패: video.json 형식이 dict가 아님")
                # 파일 I/O 또는 JSON 관련 오류 처리
                except (OSError, TypeError, ValueError, json.JSONDecodeError) as e_final_save:
                    _log_progress(f"[WARN] 최종 provider 저장 실패: {type(e_final_save).__name__}: {e_final_save}")
                except Exception as e_final_unknown:  # 예상치 못한 오류
                    _log_progress(
                        f"[ERROR] 최종 provider 저장 중 예상치 못한 오류: {type(e_final_unknown).__name__}: {e_final_unknown}")

            # 최종 결과 반환 (story_path는 Path 객체이므로 str 변환)
            # video_path_str이 None일 수 있으므로 빈 문자열로 처리
            return {"story_path": str(story_path), "video_path": video_path_str or ""}  #

        # ----- done 콜백 함수 정의 -----
        def done(ok: bool, payload: Optional[dict], err: Optional[Exception]) -> None:  #
            """작업 완료 후 UI 업데이트 및 상태 정리."""
            # 재진입 방지 플래그 해제 및 버튼 활성화 (finally 블록으로 이동하여 항상 실행되도록 함)
            try:
                self._seg_story_busy = False  # 플래그 해제
                if btn_build_story is not None:
                    try:
                        btn_build_story.setEnabled(True)  # 버튼 활성화 #
                    except RuntimeError:  # 위젯 소멸 등 예외 처리
                        pass

                # 성공/실패에 따른 UI 업데이트
                if ok and payload:
                    story_path_out = payload.get("story_path") or ""  #
                    video_path_out = payload.get("video_path") or ""  #
                    msg_parts = ["프로젝트 분석 완료"]
                    if story_path_out: msg_parts.append(f"story: {Path(story_path_out).name}")
                    if video_path_out: msg_parts.append(f"video: {Path(video_path_out).name}")
                    final_msg = " | ".join(msg_parts)

                    print(f"[UI] {final_msg}")  # 콘솔 로그
                    # 상태바 업데이트 (오류 처리 포함)
                    try:
                        status_bar_obj = getattr(self, "status", None) or getattr(self, "statusBar", None)  #
                        status_bar = status_bar_obj() if callable(status_bar_obj) else status_bar_obj
                        if status_bar and hasattr(status_bar, "showMessage"):  #
                            status_bar.showMessage(final_msg, 5000)  # 5초간 표시
                    except AttributeError:  # 상태바 관련 속성/메서드 부재
                        pass
                    except Exception as e_status:  # 기타 상태바 오류
                        print(f"[WARN] 상태바 업데이트 실패: {e_status}")

                elif err is not None:  # 작업 실패 시
                    # err 객체를 문자열로 변환 (Traceback 포함 가능)
                    err_msg_detail = traceback.format_exception(type(err), err, err.__traceback__)
                    err_msg_short = f"프로젝트 분석 실패: {type(err).__name__}: {err}"

                    print(f"[UI][ERROR] {err_msg_short}\n{''.join(err_msg_detail)}")  # 콘솔 에러 로그 (상세 정보 포함)
                    QtWidgets.QMessageBox.critical(self, "오류", err_msg_short)  # 에러 팝업 (간략 정보) #
                    # 상태바 업데이트 (오류 처리 포함)
                    try:
                        status_bar_obj = getattr(self, "status", None) or getattr(self, "statusBar", None)  #
                        status_bar = status_bar_obj() if callable(status_bar_obj) else status_bar_obj
                        if status_bar and hasattr(status_bar, "showMessage"):  #
                            status_bar.showMessage(err_msg_short, 5000)
                    except AttributeError:
                        pass
                    except Exception as e_status_err:
                        print(f"[WARN] 상태바 에러 메시지 업데이트 실패: {e_status_err}")
                else:  # ok=False인데 err=None인 경우 (이론상 발생 어려움)
                    print("[UI][ERROR] 프로젝트 분석 실패 (알 수 없는 오류)")
                    QtWidgets.QMessageBox.critical(self, "오류", "알 수 없는 오류로 프로젝트 분석에 실패했습니다.")  #

            except Exception as e_done:  # done 콜백 자체의 예외 처리
                print(f"[ERROR] done 콜백 실행 중 오류: {type(e_done).__name__}: {e_done}")
                # 필요한 경우 추가적인 오류 처리 (예: 사용자 알림)

        # 비동기 작업 실행
        run_job_with_progress_async(  #
            owner=self,
            title="프로젝트 분석 (seg → story 스켈레톤 → video AI 강화)",  #
            job=job,
            on_done=done
        )

    def on_click_test1_analyze(self) -> None:
        """
        프로젝트분석: 음악 분석 통합 파이프라인을 호출합니다.
        - audio_sync.sync_lyrics_with_whisper_pro 함수 하나가 seg_ready.json 생성부터
          가사 매칭을 통한 최종 seg.json 정제까지 모든 과정을 한 번에 처리합니다.
        """
        from pathlib import Path
        from PyQt5 import QtWidgets

        print("\n--- CHECKPOINT 1: on_click_test1_analyze 시작 ---")

        # 버튼 비활성
        btn = getattr(self, "btn_test1_story", None)  # 버튼 이름 확인 및 수정
        if isinstance(btn, QtWidgets.QAbstractButton):
            btn.setEnabled(False)

        # 유틸 로드
        try:
            from app.utils import load_json, run_job_with_progress_async
        except ImportError:
            from utils import load_json, run_job_with_progress_async

        # audio_sync 모듈 및 핵심 함수 로드
        try:
            from app.audio_sync import sync_lyrics_with_whisper_pro
        except ImportError:
            from audio_sync import sync_lyrics_with_whisper_pro

        # 프로젝트 폴더 확인
        try:
            proj_dir = self._current_project_dir()
            if not proj_dir: raise FileNotFoundError("프로젝트 폴더를 찾을 수 없습니다.")
        except Exception as e:
            if isinstance(btn, QtWidgets.QAbstractButton): btn.setEnabled(True)
            QtWidgets.QMessageBox.critical(self, "프로젝트분석 실패", str(e))
            return
        print(f"--- CHECKPOINT 2: 프로젝트 폴더 확인 ({proj_dir}) ---")

        # 보컬 파일 확인
        vocal_path_obj = self._find_latest_vocal()
        if not vocal_path_obj or not vocal_path_obj.exists():
            if isinstance(btn, QtWidgets.QAbstractButton): btn.setEnabled(True)
            QtWidgets.QMessageBox.critical(self, "프로젝트분석 실패", "보컬 오디오 파일을 찾을 수 없습니다.")
            return
        print(f"--- CHECKPOINT 3: 보컬 파일 확인 ({vocal_path_obj}) ---")

        # 가사 확인
        pj = Path(proj_dir) / "project.json"
        meta = load_json(pj, {}) or {}
        lyrics_raw = (meta.get("lyrics") or "").strip()
        if not lyrics_raw:
            if isinstance(btn, QtWidgets.QAbstractButton): btn.setEnabled(True)
            QtWidgets.QMessageBox.critical(self, "프로젝트분석 실패", "project.json에서 가사를 찾지 못했습니다.")
            return
        print(f"--- CHECKPOINT 4: 가사 확인 (길이: {len(lyrics_raw)}) ---")

        # === 분석 잡 정의 ===
        def job(log_callback):
            """백그라운드 스레드에서 실행될 통합 분석 작업."""
            log_callback({"msg": "[FLOW] 통합 음악 분석 파이프라인 시작"})

            # 단일 함수 호출로 모든 분석 및 정제 수행
            result_dict = sync_lyrics_with_whisper_pro(
                audio_path=str(vocal_path_obj),
                lyrics_text=lyrics_raw
            )

            log_callback({"msg": "[FLOW] 통합 분석 완료. 결과 반환 중..."})
            return result_dict  # UI 스레드로 결과 전체를 전달

        # === 작업 완료 후 콜백 정의 ===
        def done(ok: bool, payload: dict, err):
            try:
                if not ok:
                    QtWidgets.QMessageBox.critical(self, "프로젝트분석 실패", str(err))
                    return

                # 성공 시, 결과 요약을 팝업으로 표시
                summary = (payload or {}).get("summary_text", "분석은 완료되었으나 요약 정보를 가져오지 못했습니다.")
                print("\n[프로젝트분석 최종 결과]\n" + summary, flush=True)

                dlg = QtWidgets.QDialog(self)
                dlg.setWindowTitle("프로젝트 분석 완료")
                dlg.resize(800, 600)
                vbox = QtWidgets.QVBoxLayout(dlg)
                ed = QtWidgets.QPlainTextEdit()
                ed.setReadOnly(True)
                ed.setPlainText(summary)
                vbox.addWidget(ed)
                btn_close = QtWidgets.QPushButton("닫기")
                btn_close.clicked.connect(dlg.accept)
                row = QtWidgets.QHBoxLayout()
                row.addStretch(1)
                row.addWidget(btn_close)
                vbox.addLayout(row)
                dlg.exec_()
            finally:
                if isinstance(btn, QtWidgets.QAbstractButton):
                    btn.setEnabled(True)

        # === 비동기 작업 실행 ===
        print("--- CHECKPOINT 5: 비동기 작업(run_job_with_progress_async) 호출 ---")
        run_job_with_progress_async(self, "프로젝트 분석", job, on_done=done)

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
        입력/상태 초기화:
        - 어떤 프로젝트도 로딩되지 않은 '콜드' 상태로 되돌린다.
        - 변환 패널(LLS)과 가사 에디터, 태그 UI, 내부 컨텍스트 플래그를 모두 초기화.
        - 디스크의 파일(project.json 등)은 건드리지 않는다(기존 기능 보존).
        """

        # 1) 가사/제목/태그 입력 에디터류 초기화 (가능한 위젯 이름들을 모두 시도)
        text_candidates = [
            "te_title", "txt_title", "lineEdit_title", "le_title",
            "te_lyrics", "txt_lyrics", "plainTextEdit_lyrics", "textEdit_lyrics",
            "te_lyrics_converted", "txt_lyrics_converted", "plainTextEdit_lyrics_converted",
            "txt_prompt", "prompt_edit", "te_prompt", "input_prompt", "plainTextEdit_prompt", "textEdit_prompt"
        ]
        for name in text_candidates:
            w = getattr(self, name, None) or getattr(getattr(self, "ui", None), name, None)
            if w is None:
                continue
            # setText / setPlainText / clear 순서로 방어적 호출
            try:
                if hasattr(w, "setText"):
                    w.setText("")
                    continue
            except Exception:
                pass
            try:
                if hasattr(w, "setPlainText"):
                    w.setPlainText("")
                    continue
            except Exception:
                pass
            try:
                if hasattr(w, "clear"):
                    w.clear()
            except Exception:
                pass

        # 2) 변환(LLS) 패널은 '항상 보이기' 유지 + 비활성화(ON/OFF 토글 상태를 초기화 느낌으로 OFF)
        panel = None
        for name in ("grp_convert", "box_convert", "frame_convert", "gb_convert", "w_convert"):
            w = getattr(self, name, None) or getattr(getattr(self, "ui", None), name, None)
            if w is not None:
                panel = w
                break
        try:
            if panel is not None and hasattr(panel, "setVisible"):
                panel.setVisible(True)
            if panel is not None and hasattr(panel, "setEnabled"):
                panel.setEnabled(False)
        except Exception:
            pass

        # 3) 자동태그/수동태그 UI를 '콜드 스타트' 기본값처럼 리셋
        try:
            # 자동태그 체크를 켠 상태로(수동 비활성) — 콜드 스타트 기본과 동일
            auto_chk = getattr(self, "chk_auto_tags", None) or getattr(self, "cb_auto_tags", None)
            if auto_chk is not None and hasattr(auto_chk, "setChecked"):
                auto_chk.blockSignals(True)
                auto_chk.setChecked(True)
                auto_chk.blockSignals(False)
        except Exception:
            pass

        # 태그 체크박스들 초기화
        tag_boxes = getattr(self, "_tag_boxes", None)
        if isinstance(tag_boxes, dict) and tag_boxes:
            try:
                basic_defaults_all = {
                    "clean vocals", "clear diction", "natural articulation",
                    "breath control", "warm emotional tone", "balanced mixing", "studio reverb light",
                }
                self._checked_tags = set()
                for label, cb in tag_boxes.items():
                    on_basic = bool(label in basic_defaults_all)
                    try:
                        cb.blockSignals(True)
                        cb.setChecked(on_basic)
                        cb.setEnabled(False)  # 자동태그 ON이므로 비활성
                        cb.blockSignals(False)
                    except Exception:
                        pass
                    if on_basic:
                        self._checked_tags.add(label)
            except Exception:
                pass

        # 4) 내부 컨텍스트(프로젝트 관련) 플래그 리셋
        #    - ⚠️ 메서드 이름(_current_project_dir)은 절대 건드리지 않는다(섀도잉 방지).
        try:
            if hasattr(self, "_project_context_ready"):
                setattr(self, "_project_context_ready", False)
            if hasattr(self, "_tag_watch_last_path"):
                setattr(self, "_tag_watch_last_path", None)
            if hasattr(self, "_tag_watch_last_mtime"):
                setattr(self, "_tag_watch_last_mtime", None)
            if hasattr(self, "_music_inflight"):
                setattr(self, "_music_inflight", False)
            if hasattr(self, "_analysis_running"):
                setattr(self, "_analysis_running", False)
            if hasattr(self, "_story_build_running"):
                setattr(self, "_story_build_running", False)
            if hasattr(self, "_docs_build_running"):
                setattr(self, "_docs_build_running", False)

            # 프로젝트 경로 상태만 안전하게 초기화
            # - project_dir(문자열/PathLike 저장소)만 비우고,
            # - _active_project_dir, _forced_project_dir도 비운다.
            for cand in ("project_dir", "_active_project_dir", "_forced_project_dir"):
                if hasattr(self, cand):
                    try:
                        setattr(self, cand, None)
                    except Exception:
                        pass

            # ⚠️ _current_project_dir 는 "메서드"이므로 절대로 delattr/setattr 하지 않음
            #    (이전 버전에서 None으로 섀도잉되어 TypeError가 발생했음)
        except Exception:
            pass

        # 5) 상태바/라벨류 안내(있을 때만)
        try:
            sb = getattr(self, "statusBar", None)
            bar = sb() if callable(sb) else None
            if bar is not None and hasattr(bar, "showMessage"):
                bar.showMessage("초기화 완료 — 현재 열려있는 프로젝트 없음")
        except Exception:
            pass

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
                from video_build import (
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

    def _build_clip_from_image(self, img: Path, out_mp4: Path, duration: float, fps: int = 16,
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
        """
        최근 프로젝트 폴더 추정.
        - [수정됨] settings.BASE_DIR 직접 참조로 상수 임포트 경고 수정.
        우선순위:
          1) self._forced_project_dir 이 유효하면 그것
          2) BASE_DIR/maked_title/* 중 project.json 존재 폴더의 최신 mtime
          3) BASE_DIR/* (과거 레이아웃 호환) 중 최신 mtime
        """
        from pathlib import Path
        import os  # stat().st_mtime 사용 위해 필요

        # 1) 강제 지정이 있으면 최우선
        forced_dir_attr = getattr(self, "_forced_project_dir", None)  #
        # str, Path 객체 모두 처리
        if isinstance(forced_dir_attr, (str, Path)):
            try:
                # os.fspath()로 Path 객체도 문자열로 변환
                forced_path_obj = Path(os.fspath(forced_dir_attr))
                if forced_path_obj.is_dir():  # is_dir()로 폴더인지 확인
                    return forced_path_obj
            except (TypeError, ValueError, OSError):
                # 경로 변환/접근 오류 시 무시하고 다음 단계로
                pass

        # BASE_DIR 확보 (settings 모듈 직접 사용)
        base_dir_path_obj: Optional[Path] = None
        settings_module = None
        try:
            # app 패키지 우선 시도
            from app import settings as settings_module  #
            base_dir_val = getattr(settings_module, "BASE_DIR", ".")  #
            base_dir_path_obj = Path(base_dir_val).resolve()  # Path 객체로 변환 및 절대경로화
        except ImportError:
            try:
                # 루트 settings 시도
                import settings as settings_module  # type: ignore [no-redef] #
                base_dir_val = getattr(settings_module, "BASE_DIR", ".")  #
                base_dir_path_obj = Path(base_dir_val).resolve()
            except ImportError:
                # 둘 다 실패 시 현재 작업 디렉토리 기준
                base_dir_path_obj = Path(".").resolve()
            except Exception as e_settings:
                # 설정 로드 중 다른 오류 발생 시
                print(f"[WARN] 설정 로드 중 오류 발생: {e_settings}")
                base_dir_path_obj = Path(".").resolve()
        except Exception as e_path_resolve:
            # Path 객체 생성 또는 resolve 실패 시
            print(f"[WARN] 기본 경로 처리 중 오류 발생: {e_path_resolve}")
            base_dir_path_obj = Path(".").resolve()  # 안전한 폴백

        if base_dir_path_obj is None or not base_dir_path_obj.is_dir():
            print("[WARN] 유효한 BASE_DIR를 찾을 수 없습니다.")
            return None  # 유효한 기본 경로 없으면 None 반환

        # 2) 신 레이아웃: BASE_DIR/maked_title/*/project.json 기준으로 최신
        try:
            maked_title_path = base_dir_path_obj / "maked_title"  #
            if maked_title_path.is_dir():  # 폴더 존재 확인
                # project.json 파일을 기준으로 최신 수정 시간 정렬
                candidates_list: list[tuple[float, Path]] = []  # 타입 명시
                # glob으로 project.json 파일 검색
                for project_json_file in maked_title_path.glob("*/project.json"):  #
                    try:
                        # 파일의 수정 시간과 부모 폴더 경로 저장
                        mod_time = project_json_file.stat().st_mtime  #
                        parent_dir = project_json_file.parent
                        if parent_dir.is_dir():  # 부모가 실제 폴더인지 확인
                            candidates_list.append((mod_time, parent_dir))
                    except (OSError, FileNotFoundError) as e_stat:
                        # 파일 상태 정보 읽기 실패 시 경고 출력 후 계속
                        print(f"[WARN] 파일 상태 읽기 실패 ({project_json_file.name}): {e_stat}")
                        continue
                if candidates_list:
                    # 수정 시간 기준 내림차순 정렬
                    candidates_list.sort(key=lambda item: item[0], reverse=True)  #
                    # 가장 최근 폴더 반환
                    return candidates_list[0][1]
        except OSError as e_os_new:  # maked_title 폴더 접근 오류 등
            print(f"[WARN] 신규 레이아웃 검색 중 오류 발생: {e_os_new}")
        except Exception as e_glob_new:  # glob 패턴 오류 등 예외 처리
            print(f"[WARN] 신규 레이아웃 검색 중 예외 발생: {e_glob_new}")

        # 3) 구 레이아웃 호환: BASE_DIR/* 디렉토리 중 최신
        try:
            if base_dir_path_obj.is_dir():  # 기본 경로가 폴더인지 확인
                # 모든 하위 항목 중 폴더만 필터링
                sub_dirs = [p for p in base_dir_path_obj.glob("*") if p.is_dir()]  #
                if sub_dirs:
                    valid_sub_dirs: list[tuple[float, Path]] = []  # 타입 명시
                    # 각 폴더의 수정 시간 확인 (오류 처리 포함)
                    for sub_dir_path in sub_dirs:
                        try:
                            mod_time = sub_dir_path.stat().st_mtime  #
                            valid_sub_dirs.append((mod_time, sub_dir_path))
                        except (OSError, FileNotFoundError) as e_stat_old:
                            print(f"[WARN] 폴더 상태 읽기 실패 ({sub_dir_path.name}): {e_stat_old}")
                            continue
                    if valid_sub_dirs:
                        # 수정 시간 기준 내림차순 정렬
                        valid_sub_dirs.sort(key=lambda item: item[0], reverse=True)  #
                        # 가장 최근 폴더 반환
                        return valid_sub_dirs[0][1]
        except OSError as e_os_old:  # 기본 폴더 접근 오류 등
            print(f"[WARN] 구 레이아웃 검색 중 오류 발생: {e_os_old}")
        except Exception as e_glob_old:  # glob 패턴 오류 등 예외 처리
            print(f"[WARN] 구 레이아웃 검색 중 예외 발생: {e_glob_old}")

        # 모든 방법 실패 시 None 반환
        return None

    def _current_project_dir(self) -> Optional[Path]:
        """
        현재 활성 프로젝트 폴더.
        우선순위:
          1) self.project_dir (가사 생성 직후 _apply_lyrics_result에서 설정됨)
          2) self._forced_project_dir
          3) self._latest_project()
        """
        from pathlib import Path
        import os

        # 1) 명시적으로 잡힌 project_dir 우선
        p = getattr(self, "project_dir", None)
        if isinstance(p, (str, bytes, os.PathLike)):
            try:
                pp = Path(os.fspath(p))
                if pp.exists():
                    return pp
            except Exception:
                pass

        # 2) 강제 지정 폴더
        f = getattr(self, "_forced_project_dir", None)
        if isinstance(f, (str, bytes, os.PathLike)):
            try:
                pf = Path(os.fspath(f))
                if pf.exists():
                    return pf
            except Exception:
                pass

        # 3) 최근 프로젝트 폴더
        return self._latest_project()

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

        # ▼▼ 추가: 형제 그룹 2개(긍정/부정)
        self.te_prompt_pos = QtWidgets.QTextEdit()
        self.te_prompt_pos.setPlainText("ace-step tag 추천해줘 : \n")
        prompt_grp_pos = self._group("긍정 프롬프트(+)", self.te_prompt_pos)

        self.te_prompt_neg = QtWidgets.QTextEdit()
        prompt_grp_neg = self._group("부정 프롬프트(-)", self.te_prompt_neg)

        # 자동 태그 토글
        self.cb_auto_tags = QtWidgets.QCheckBox("태그 자동(가사 분위기 기반 추천)")
        self.cb_auto_tags.setChecked(True)
        self.cb_auto_tags.toggled.connect(self._on_auto_toggle)

        # ai select 토글
        self.btn_ai_toggle = QtWidgets.QToolButton(self)
        self.btn_ai_toggle.setCheckable(True)
        self.btn_ai_toggle.toggled.connect(self.on_ai_toggle)

        # 기본 상태를 'Gemini만 사용'으로 맞춘다
        self.btn_ai_toggle.setChecked(True)

        # 표시/툴팁을 기본 상태에 맞춰 즉시 동기화
        try:
            # on_ai_toggle(bool)이 버튼 텍스트/툴팁을 갱신한다면, 현재 상태로 한 번 호출
            self.on_ai_toggle(self.btn_ai_toggle.isChecked())
        except Exception:
            # on_ai_toggle이 표시를 안 바꾼다면 안전하게 기본 표기를 직접 지정
            self.btn_ai_toggle.setText("모드: Gemini만")
            self.btn_ai_toggle.setToolTip("클릭: GPT 우선(부족 시 Gemini 폴백) / 다시 클릭: Gemini만 사용")

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

        # (교체 전) main_layout.addWidget(prompt_grp)

        # ▼▼ 교체: 3분할 가로 배치
        row_prompts = QtWidgets.QHBoxLayout()
        row_prompts.addWidget(prompt_grp)
        row_prompts.addWidget(prompt_grp_pos)
        row_prompts.addWidget(prompt_grp_neg)
        main_layout.addLayout(row_prompts)

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
        [수정 v2] 가사 길이 추정 (발음 기반):
          - seconds_per_unit 값을 대폭 축소 (0.25 수준으로 조정).
          - kroman 없으면 기존 글자 수 기반 로직으로 폴백.
          - 1~3600초로 클램프.
        """
        kroman_mod = None
        try:
            import kroman as kroman_mod
        except ImportError:
            print("[WARN] kroman library not found. Falling back to character count estimation.")
            # kroman 없을 때 폴백 로직 (v1과 동일)
            section_fb = "default"
            pieces_fb: list[tuple[str, str]] = []
            for raw_fb in (text or "").splitlines():
                s_fb = (raw_fb or "").strip()
                if not s_fb: continue
                m_fb = re.fullmatch(r"\[([^\[\]\n]+)]", s_fb, flags=re.I)
                if m_fb: section_fb = m_fb.group(1).strip().lower(); continue
                pieces_fb.append((section_fb, s_fb))
            total_chars_fb = 0.0  # float으로 변경
            for sec_fb, content_fb in pieces_fb:
                body_fb = re.sub(r"\s+", "", content_fb)
                if not body_fb: continue
                weight_fb = 1.5 if sec_fb == "chorus" else 1.0
                total_chars_fb += weight_fb * len(body_fb)
            if total_chars_fb <= 0: return 0
            spc_fb = getattr(self, "seconds_per_char", 0.72)
            seconds_raw_fb = total_chars_fb * float(spc_fb)
            seconds_buf_fb = math.ceil(seconds_raw_fb * 1.20)
            return int(max(1, min(3600, seconds_buf_fb)))

        # --- kroman 사용 가능 시: 발음 기반 로직 ---
        section = "default"
        pronunciation_based_length = 0.0
        for raw_line in (text or "").splitlines():  # (v1과 동일 로직)
            line_strip = (raw_line or "").strip()
            if not line_strip: continue
            match_header = re.fullmatch(r"\[([^\[\]\n]+)]", line_strip, flags=re.I)
            if match_header: section = match_header.group(1).strip().lower(); continue
            try:
                romanized_line: Optional[str] = kroman_mod.parse(line_strip)  # type: ignore[attr-defined]
                if romanized_line is not None:
                    processed_line = romanized_line.replace("-", "").lower()
                    line_len = len(re.sub(r"\s+", "", processed_line))
                else:
                    line_len = len(re.sub(r"\s+", "", line_strip))
            except Exception:
                line_len = len(re.sub(r"\s+", "", line_strip))
            weight = 1.5 if section == "chorus" else 1.0
            pronunciation_based_length += weight * line_len

        if pronunciation_based_length <= 0: return 0

        # ★★ 글자(발음 단위)당 시간 대폭 축소 (0.72 * 0.35 ≈ 0.25) ★★
        #    이 값을 조정하여 전체 길이를 튜닝하세요.
        adjustment_factor = 0.35
        seconds_per_unit = getattr(self, "seconds_per_char", 0.72) * adjustment_factor

        seconds_raw = pronunciation_based_length * float(seconds_per_unit)
        seconds_buffered = math.ceil(seconds_raw * 1.20)  # 1.2배 버퍼 유지

        final_seconds = int(max(1, min(3600, seconds_buffered)))
        return final_seconds

    def _wire_convert_toggle_action(self) -> None:
        """
        '변환' 토글(btn_convert_toggle)을 kroman 변환과 project.json 저장에 연결한다.
        - ON : 오른쪽 변환칸이 비어 있으면 즉석 변환 후 주입하고, meta['lyrics_lls']=변환텍스트 저장
        - OFF: meta['lyrics_lls']=""
        - 버튼 상태/활성화 여부는 절대 변경하지 않는다(배선만 함).
        """
        from pathlib import Path
        from json import loads, dumps, JSONDecodeError
        from PyQt5 import QtWidgets

        btn = getattr(self, "btn_convert_toggle", None)
        if not isinstance(btn, (QtWidgets.QPushButton, QtWidgets.QToolButton)):
            return
        if not btn.isCheckable():
            # UI는 건드리지 않지만, 토글형이 아니면 연결 불가
            return

        # 중복 연결 방지
        if getattr(btn, "_lls_wired", False):
            return
        btn._lls_wired = True

        te_src = getattr(self, "te_lyrics", None)
        te_dst = getattr(self, "te_lyrics_converted", None)
        if not hasattr(te_src, "toPlainText") or not hasattr(te_dst, "setPlainText"):
            return

        def _proj_dir() -> Path | None:
            p = getattr(self, "_active_project_dir", None)
            try:
                return Path(p) if p else None
            except TypeError:
                return None

        def _load_json(p: Path) -> dict:
            try:
                return loads(p.read_text(encoding="utf-8"))
            except (FileNotFoundError, JSONDecodeError, UnicodeDecodeError):
                return {}
            except OSError:
                return {}

        def _save_json(p: Path, data: dict) -> None:
            try:
                p.write_text(dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError:
                pass

        def _convert_text(text: str) -> str:
            # 섹션 헤더/빈 줄 보존, 본문 라인만 [ko]+kroman, 하이픈 제거
            try:
                import kroman  # type: ignore
            except ImportError:
                return text
            out = []
            for raw in (text or "").splitlines():
                s = raw
                st = s.strip()
                if (st.startswith("[") and st.endswith("]")) or not st:
                    out.append(s)
                else:
                    rom = kroman.parse(s).strip().replace("-", "")
                    out.append("[ko]" + rom)
            return "\n".join(out)

        def _on_toggled(checked: bool) -> None:
            proj = _proj_dir()
            if not proj:
                return
            pj = proj / "project.json"
            meta = _load_json(pj)

            if checked:
                # 변환 결과가 비어 있으면 즉석 변환
                current = ""
                try:
                    current = te_dst.toPlainText().strip()
                except Exception:
                    current = ""
                if not current:
                    src = ""
                    try:
                        src = te_src.toPlainText()
                    except Exception:
                        src = ""
                    current = _convert_text(src)
                    try:
                        te_dst.setPlainText(current)
                    except Exception:
                        pass
                meta["lyrics_lls"] = current
            else:
                meta["lyrics_lls"] = ""

            _save_json(pj, meta)

        # ── 여기만 변경: 정적 분석기 경고 방지용 안전 연결 ─────────────
        sig = getattr(btn, "toggled", None)
        try:
            # pyqtSignal 인스턴스면 connect가 존재함
            if hasattr(sig, "connect"):
                sig.connect(_on_toggled)  # type: ignore[attr-defined]
            else:
                # 혹시라도 타입 추론 실패 시 런타임 연결 시도
                btn.toggled.connect(_on_toggled)  # type: ignore[attr-defined]
        except Exception:
            pass

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
        # 항상 같은 모듈(alias s_mod)만 쓰도록 통일!
        try:
            import settings as s_mod
        except Exception:
            from app import settings as s_mod  # type: ignore

        tab = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout()

        # BASE_DIR
        self.le_base_dir = QtWidgets.QLineEdit(str(s_mod.BASE_DIR))
        btn_pick_base = QtWidgets.QPushButton("폴더 선택")

        def _pick_base():
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "BASE_DIR 선택", str(s_mod.BASE_DIR))
            if d:
                self.le_base_dir.setText(d)

        btn_pick_base.clicked.connect(_pick_base)
        base_wrap = QtWidgets.QHBoxLayout()
        base_wrap.addWidget(self.le_base_dir)
        base_wrap.addWidget(btn_pick_base)
        base_widget = QtWidgets.QWidget()
        base_widget.setLayout(base_wrap)

        # COMFY_HOST & 후보
        self.le_comfy = QtWidgets.QLineEdit(s_mod.COMFY_HOST)
        self.te_candidates = QtWidgets.QPlainTextEdit("\n".join(s_mod.DEFAULT_HOST_CANDIDATES))

        # ffmpeg / hwaccel / 출력파일 / 오디오 포맷
        self.le_ffmpeg = QtWidgets.QLineEdit(s_mod.FFMPEG_EXE)
        self.cb_hwaccel = QtWidgets.QCheckBox("USE_HWACCEL")
        self.cb_hwaccel.setChecked(bool(s_mod.USE_HWACCEL))
        self.le_final = QtWidgets.QLineEdit(s_mod.FINAL_OUT)

        self.cb_audio_fmt = QtWidgets.QComboBox()
        self.cb_audio_fmt.addItems(["mp3", "wav", "opus"])
        # overrides 적용 고려
        ov = s_mod.load_overrides() or {}
        cur_fmt = str(ov.get("AUDIO_SAVE_FORMAT", s_mod.AUDIO_SAVE_FORMAT)).lower()
        idx_fmt = max(0, self.cb_audio_fmt.findText(cur_fmt))
        self.cb_audio_fmt.setCurrentIndex(idx_fmt)

        # 기본 프레임/분할
        self.sb_d_chunk = self._spin(60, 5000, int(s_mod.DEFAULT_CHUNK), " /chunk")
        self.sb_d_overlap = self._spin(1, 120, int(s_mod.DEFAULT_OVERLAP), " overlap")
        self.sb_d_infps = self._spin(1, 240, int(s_mod.DEFAULT_INPUT_FPS), " inFPS")
        self.sb_d_outfps = self._spin(1, 240, int(s_mod.DEFAULT_TARGET_FPS), " outFPS")

        # 프롬프트/워크플로 파일 경로
        self.le_prompt_json = QtWidgets.QLineEdit(str(s_mod.ACE_STEP_PROMPT_JSON))
        btn_pick_prompt = QtWidgets.QPushButton("파일 선택")

        def _pick_prompt():
            f, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "ACE_STEP_PROMPT_JSON", str(s_mod.JSONS_DIR), "JSON (*.json)"
            )
            if f:
                self.le_prompt_json.setText(f)

        btn_pick_prompt.clicked.connect(_pick_prompt)
        pj_wrap = QtWidgets.QHBoxLayout()
        pj_wrap.addWidget(self.le_prompt_json)
        pj_wrap.addWidget(btn_pick_prompt)
        pj_widget = QtWidgets.QWidget()
        pj_widget.setLayout(pj_wrap)

        self.le_i2v = QtWidgets.QLineEdit(str(s_mod.I2V_WORKFLOW))
        btn_pick_i2v = QtWidgets.QPushButton("파일 선택")

        def _pick_i2v():
            f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "I2V_WORKFLOW", str(s_mod.JSONS_DIR), "JSON (*.json)")
            if f:
                self.le_i2v.setText(f)

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
            path = s_mod.save_overrides(**_collect_overrides())
            QtWidgets.QMessageBox.information(self, "저장 완료", f"settings_local.json 저장됨\n\n{path}")

        def _do_apply():
            """
            설정을 settings_local.json에 저장하고, 런타임 settings 모듈 값도 즉시 반영.
            그리고 현재 포맷/제목/경로에 맞게 ComfyUI 워크플로의 SaveAudio 노드들을
            프로젝트 오디오 폴더(템플릿 [title] 치환)에 저장하도록 패치한다.
            """
            # 1) 파일로 저장 (settings_local.json)
            overrides = _collect_overrides()
            s_mod.save_overrides(**overrides)

            # 2) 런타임(모듈 settings)에도 즉시 반영
            try:
                import settings as _s
            except Exception:
                from app import settings as _s  # type: ignore

            _s.BASE_DIR = overrides.get("BASE_DIR", _s.BASE_DIR)
            _s.COMFY_HOST = overrides.get("COMFY_HOST", _s.COMFY_HOST)
            _s.DEFAULT_HOST_CANDIDATES = overrides.get("DEFAULT_HOST_CANDIDATES",
                                                       getattr(_s, "DEFAULT_HOST_CANDIDATES", []))
            _s.FFMPEG_EXE = overrides.get("FFMPEG_EXE", _s.FFMPEG_EXE)
            _s.USE_HWACCEL = bool(overrides.get("USE_HWACCEL", getattr(_s, "USE_HWACCEL", False)))
            _s.FINAL_OUT = overrides.get("FINAL_OUT", _s.FINAL_OUT)
            _s.AUDIO_SAVE_FORMAT = overrides.get("AUDIO_SAVE_FORMAT", getattr(_s, "AUDIO_SAVE_FORMAT", "mp3")).lower()
            _s.DEFAULT_CHUNK = int(overrides.get("DEFAULT_CHUNK", getattr(_s, "DEFAULT_CHUNK", 600)))
            _s.DEFAULT_OVERLAP = int(overrides.get("DEFAULT_OVERLAP", getattr(_s, "DEFAULT_OVERLAP", 12)))
            _s.DEFAULT_INPUT_FPS = int(overrides.get("DEFAULT_INPUT_FPS", getattr(_s, "DEFAULT_INPUT_FPS", 24)))
            _s.DEFAULT_TARGET_FPS = int(overrides.get("DEFAULT_TARGET_FPS", getattr(_s, "DEFAULT_TARGET_FPS", 24)))
            _s.ACE_STEP_PROMPT_JSON = overrides.get("ACE_STEP_PROMPT_JSON", _s.ACE_STEP_PROMPT_JSON)
            _s.I2V_WORKFLOW = overrides.get("I2V_WORKFLOW", _s.I2V_WORKFLOW)

            # 3) 워크플로 JSON 저장 노드 포맷/경로 패치
            try:
                fmt = self.cb_audio_fmt.currentText().strip().lower()
                json_path = Path(self.le_prompt_json.text().strip())

                title = (self.le_title.text() or "").strip()
                if not title:
                    pdir = self._latest_project()
                    if pdir and (pdir / "project.json").exists():
                        try:
                            from utils import load_json as _load_json
                            meta = _load_json(pdir / "project.json", {}) or {}
                            title = (meta.get("title") or "").strip()
                        except Exception:
                            title = ""

                save_root = _s.FINAL_OUT or str(_s.BASE_DIR)
                proj_audio_dir = _resolve_audio_dir_from_template(save_root, title or "untitled")

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
    def _on_tags_changed(self, *_args) -> None:
        """
        태그(자동/수동, 보컬 성별, 개별 체크박스) 변경 시 즉시 project.json에 반영.
        - project.json이 없으면 아무 것도 하지 않음(가사 생성 단계에서 저장됨)
        - 자동 모드: ace_tags(최근 제안)와 tags_in_use(보조선택) 반영
        - 수동 모드: manual_tags만 반영
        - seconds/target_seconds 등 다른 필드는 건드리지 않음
        """
        from pathlib import Path
        try:
            from utils import load_json, save_json  # 우선 경로
        except ImportError:
            from app.utils import load_json, save_json  # 대체 경로

        proj_dir = getattr(self, "project_dir", "") or ""
        if not proj_dir:
            return  # 프로젝트 파일이 아직 없음 → 가사 생성 시점에서 저장

        pj = Path(proj_dir) / "project.json"
        if not pj.exists():
            return  # 아직 생성 전이면 건너뜀(요청한 정책)

        meta = load_json(pj, {}) or {}

        # 현재 UI 상태 수집(기존 유틸 사용)
        # _gather_tag_state는 auto_tags/성별/수동체크 리스트를 포함해야 함
        try:
            state = self._gather_tag_state()  # type: ignore[attr-defined]
        except Exception:
            state = {}

        auto_on = bool(state.get("auto_tags"))
        manual_checked = state.get("manual_checked") or []

        # 규칙: 자동 ON => ace_tags/tags_in_use, 자동 OFF => manual_tags
        meta["auto_tags"] = auto_on

        if auto_on:
            # 최근 AI 제안 태그는 self._last_tags 보존 규칙에 따름(없으면 기존 값 유지)
            last = getattr(self, "_last_tags", None)
            if isinstance(last, (list, tuple)):
                meta["ace_tags"] = list(last)
            # 수동 보조 선택은 tags_in_use로 기록
            meta["tags_in_use"] = list(manual_checked)
            # 수동 목록은 혼동 방지 위해 비우거나 보존 중 하나를 선택해야 함
            # 현재 로직 보존: manual_tags는 건드리지 않음(필요 시 이후 토글에서 사용)
        else:
            # 수동 모드에서는 manual_tags만 기록
            meta["manual_tags"] = list(manual_checked)

        save_json(pj, meta)
        # 상태바가 있다면 사용자에게 저장 알림(선택적, 존재할 때만)
        sb = getattr(self, "status", None)
        if sb is not None and hasattr(sb, "showMessage"):
            try:
                sb.showMessage("태그 상태 저장됨", 2000)
            except Exception:
                pass

    def _toggle_manual_tag_widgets(self, enabled: bool) -> None:
        """
        '수동 태그' 위젯들의 활성/비활성 토글.
        - 기존 동작(위젯 enable/disable)은 그대로 유지
        - 토글 직후 현재 태그 상태를 project.json에 '즉시' 반영
        - 태그 관련 체크박스/라디오에 변경 이벤트를 1회만 바인딩(중복 연결 방지)
        """
        # 1) 기존 UI 토글 동작 유지
        try:
            for group in (
                    getattr(self, "cb_basic_vocal_list", []),
                    getattr(self, "cb_style_checks", []),
                    getattr(self, "cb_scene_checks", []),
                    getattr(self, "cb_instr_checks", []),
                    getattr(self, "cb_tempo_checks", []),
            ):
                for cb in group:
                    if hasattr(cb, "setEnabled"):
                        cb.setEnabled(bool(enabled))
        except Exception:
            # 위젯이 아직 일부만 초기화된 초기 단계에서도 안전하게 넘어가도록 최소 범위 예외
            return

        # 2) 변경 이벤트 → 저장 핸들러 바인딩(최초 1회만)
        #    - 없는 함수 호출 금지: 아래에서 정의하는 self._on_tags_changed를 사용
        #    - 이미 연결되었는지 플래그로 확인
        if not getattr(self, "_tags_signal_bound", False):
            def _bind_cb_list(cbs):
                for cbb in cbs:
                    # QCheckBox / QRadioButton 호환: stateChanged 또는 toggled 사용 가능
                    if hasattr(cbb, "stateChanged"):
                        try:
                            cbb.stateChanged.connect(self._on_tags_changed)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    if hasattr(cbb, "toggled"):
                        try:
                            cbb.toggled.connect(self._on_tags_changed)  # type: ignore[attr-defined]
                        except Exception:
                            pass

            _bind_cb_list(getattr(self, "cb_basic_vocal_list", []))
            _bind_cb_list(getattr(self, "cb_style_checks", []))
            _bind_cb_list(getattr(self, "cb_scene_checks", []))
            _bind_cb_list(getattr(self, "cb_instr_checks", []))
            _bind_cb_list(getattr(self, "cb_tempo_checks", []))

            # 자동 태그 토글 체크박스 및 보컬 성별 라디오도 저장에 영향 → 같이 바인딩
            cb_auto = getattr(self, "cb_auto_tags", None)
            if cb_auto is not None:
                if hasattr(cb_auto, "toggled"):
                    try:
                        cb_auto.toggled.connect(self._on_tags_changed)  # type: ignore[attr-defined]
                    except Exception:
                        pass

            for rb_name in ("rb_vocal_female", "rb_vocal_male", "rb_vocal_mixed"):
                rb = getattr(self, rb_name, None)
                if rb is not None and hasattr(rb, "toggled"):
                    try:
                        rb.toggled.connect(self._on_tags_changed)  # type: ignore[attr-defined]
                    except Exception:
                        pass

            self._tags_signal_bound = True  # 중복 연결 방지

        # 3) 토글 직후 즉시 저장(실시간 반영)
        #    - 기존 규칙: auto ON → ace_tags/tags_in_use, auto OFF → manual_tags
        try:
            self._on_tags_changed()
        except Exception:
            # 저장 실패가 토글 동작 자체를 막지 않도록
            pass

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

    def _save_project_snapshot(self) -> None:
        """
        현재 UI 상태를 project.json에 저장.
        - 제목/가사/길이
        - 변환(LLS) 상태
        - 자동/수동 태그
        - 긍정 프롬프트: prompt_user(=UI 값), prompt(호환 필드)도 함께 기록
        """
        from pathlib import Path
        import json
        from json import JSONDecodeError
        from collections.abc import Iterable

        try:
            from app.utils import load_json as _lj, save_json as _sj  # type: ignore
            load_json_fn, save_json_fn = _lj, _sj
        except Exception:
            def load_json_fn(p: Path, default=None):
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except (FileNotFoundError, JSONDecodeError, UnicodeDecodeError, OSError):
                    return default

            def save_json_fn(p: Path, data: dict) -> None:
                try:
                    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass

        proj_dir = getattr(self, "_active_project_dir", None) or getattr(self, "project_dir", None)
        if not proj_dir:
            return
        pj = Path(proj_dir) / "project.json"
        meta = load_json_fn(pj, {}) or {}
        if not isinstance(meta, dict):
            meta = {}

        # 제목
        title = ""
        if hasattr(self, "le_title"):
            try:
                title = self.le_title.text().strip()
            except Exception:
                title = (meta.get("title") or "").strip()
        meta["title"] = title

        # 가사
        lyrics_left = ""
        if hasattr(self, "te_lyrics"):
            try:
                lyrics_left = self.te_lyrics.toPlainText().strip()
            except Exception:
                lyrics_left = (meta.get("lyrics") or "").strip()
        meta["lyrics"] = lyrics_left

        # 프롬프트(긍정)
        prompt_widget = (
                getattr(self, "le_prompt", None)
                or getattr(self, "te_prompt", None)
                or getattr(self, "te_prompt_pos", None)
        )
        prompt_text = ""
        try:
            if prompt_widget is not None:
                if hasattr(prompt_widget, "text"):
                    prompt_text = prompt_widget.text().strip()
                elif hasattr(prompt_widget, "toPlainText"):
                    prompt_text = prompt_widget.toPlainText().strip()
        except Exception:
            prompt_text = (meta.get("prompt_user") or meta.get("prompt") or "").strip()
        meta["prompt_user"] = prompt_text
        meta["prompt"] = prompt_text  # 레거시 호환

        # 길이
        try:
            seconds = self._read_seconds_from_ui()
        except Exception:
            seconds = meta.get("time") or meta.get("target_seconds") or 0
        try:
            seconds = int(seconds)
        except Exception:
            seconds = 0
        if seconds > 0:
            meta["time"] = seconds
            meta["target_seconds"] = seconds

        # 변환(LLS)
        cb_convert = (
                getattr(self, "btn_convert_toggle", None)
                or getattr(self, "cb_convert", None)
                or getattr(self, "toggle_convert", None)
        )
        lls_on = False
        if cb_convert is not None and hasattr(cb_convert, "isChecked"):
            try:
                lls_on = bool(cb_convert.isChecked())
            except Exception:
                lls_on = False
        te_conv = getattr(self, "te_lyrics_converted", None)
        if lls_on:
            lls_text = ""
            if te_conv is not None and hasattr(te_conv, "toPlainText"):
                try:
                    lls_text = te_conv.toPlainText().strip()
                except Exception:
                    lls_text = (meta.get("lyrics_lls") or "").strip()
            else:
                lls_text = (meta.get("lyrics_lls") or "").strip()
            meta["lyrics_lls"] = lls_text
            meta["lyrics_lls_backup"] = lls_text
        else:
            lls_now = (meta.get("lyrics_lls") or "").strip()
            if lls_now:
                meta["lyrics_lls_backup"] = lls_now
            if "lyrics_lls" in meta:
                try:
                    del meta["lyrics_lls"]
                except KeyError:
                    meta["lyrics_lls"] = ""

        # 자동/수동 태그
        cb_auto = getattr(self, "cb_auto_tags", None)
        auto_on = None
        if cb_auto is not None and hasattr(cb_auto, "isChecked"):
            try:
                auto_on = bool(cb_auto.isChecked())
            except Exception:
                auto_on = None
        if auto_on is None:
            auto_on = bool(meta.get("auto_tags", True))
        meta["auto_tags"] = bool(auto_on)

        # 현재 체크된 태그 수집 (Iterable 검증으로 타입 경고 제거)
        checked = []
        try:
            gather = getattr(self, "_gather_checked_tags", None) or getattr(self, "_collect_checked_tags", None)
            if callable(gather):
                gathered = gather()
                if isinstance(gathered, Iterable) and not isinstance(gathered, (str, bytes)):
                    checked = [str(x) for x in gathered if str(x).strip()]
                else:
                    checked = []
            else:
                prev = meta.get("checked_tags") or []
                checked = [str(x) for x in prev] if isinstance(prev, list) else []
        except (AttributeError, TypeError, ValueError):
            prev = meta.get("checked_tags") or []
            checked = [str(x) for x in prev] if isinstance(prev, list) else []

        if not bool(auto_on):
            meta["manual_tags"] = list(checked)
            meta["checked_tags"] = list(checked)
            meta["tags_effective"] = list(checked)
            meta.pop("tags_in_use", None)
            meta.pop("ace_tags", None)
        else:
            meta["checked_tags"] = list(checked)

        save_json_fn(pj, meta)

        try:
            self._set_active_project_dir(str(Path(proj_dir)))
        except Exception:
            pass

    # ────────────── 가사 생성 ──────────────
    # app/shorts_ui.py
    class YourClass:
        @staticmethod
        def set_lls_enabled_for_project(proj_dir: str, enabled: bool) -> None:
            """
            변환(LLS) 토글을 켰/껐을 때 프로젝트의 LLS 캐시를 관리한다.
            - ON(enabled=True): 캐시 유지. 필요시 파일 내용으로 meta['lyrics_lls'] 보강
            - OFF(enabled=False): lyrics_lls 비우고 캐시 파일 삭제
            """
            from pathlib import Path
            import json

            p = Path(proj_dir)
            pj = p / "project.json"

            def _load_json(path, default=None):
                try:
                    s = Path(path).read_text(encoding="utf-8")
                    return json.loads(s)
                except FileNotFoundError:
                    return default
                except Exception:
                    return default

            def _save_json(path, data):
                try:
                    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass

            def _safe_unlink(path_obj: Path) -> None:
                try:
                    path_obj.unlink(missing_ok=True)
                except TypeError:
                    if path_obj.exists():
                        try:
                            path_obj.unlink()
                        except Exception:
                            pass
                except Exception:
                    pass

            meta = _load_json(pj, {}) or {}
            meta["lls_enabled"] = bool(enabled)

            if enabled:
                if not meta.get("lyrics_lls"):
                    try:
                        txt = (p / "_lls_after.txt").read_text(encoding="utf-8", errors="ignore").strip()
                    except FileNotFoundError:
                        txt = ""
                    except Exception:
                        txt = ""
                    if not txt:
                        try:
                            txt = (p / "lyrics_lls.txt").read_text(encoding="utf-8", errors="ignore").strip()
                        except Exception:
                            txt = ""
                    if txt:
                        meta["lyrics_lls"] = txt
            else:
                if "lyrics_lls" in meta:
                    try:
                        del meta["lyrics_lls"]
                    except Exception:
                        meta["lyrics_lls"] = ""
                _safe_unlink(p / "_lls_after.txt")
                _safe_unlink(p / "lyrics_lls.txt")

            _save_json(pj, meta)
            print(f"[LLS] toggle -> enabled={bool(enabled)}", flush=True)

    @staticmethod
    def save_lls_text_for_project(proj_dir: str, lls_text: str) -> None:
        """
        변환(LLS) 결과 텍스트를 프로젝트에 저장한다.
        - 파일: _lls_after.txt, lyrics_lls.txt
        - 메타: project.json의 lyrics_lls
        - 원본 meta['lyrics']는 절대 변경하지 않음
        - 저장과 동시에 lls_enabled=True (저장/스냅샷 시 캐시가 지워지는 것 방지)
        """
        from pathlib import Path
        import json

        p = Path(proj_dir)
        pj = p / "project.json"

        # 안전 로더/세이버
        def _load_json(path, default=None):
            try:
                s = Path(path).read_text(encoding="utf-8")
                return json.loads(s)
            except FileNotFoundError:
                return default
            except Exception:
                return default

        def _save_json(path, data):
            try:
                Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

        meta = _load_json(pj, {}) or {}
        text = (lls_text or "").strip()

        # 파일 저장
        try:
            (p / "_lls_after.txt").write_text(text, encoding="utf-8", errors="ignore")
        except Exception:
            pass
        try:
            (p / "lyrics_lls.txt").write_text(text, encoding="utf-8", errors="ignore")
        except Exception:
            pass

        # 메타 저장 — 원본 가사는 유지
        meta["lyrics_lls"] = text
        meta["lls_enabled"] = True  # 변환 저장 시 자동 ON
        _save_json(pj, meta)

        print("[LLS] saved -> lls_enabled=True", flush=True)

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

    @staticmethod
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
        - 생성 폴더를 활성 프로젝트로 설정(시그니처 차이에 안전 호출)
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
            from utils import normalize_tags_to_english  # type: ignore
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
        try:
            if hasattr(self, "le_title") and self.le_title is not None and hasattr(self.le_title, "setText"):
                self.le_title.setText(title)
        except Exception:
            pass
        try:
            te_l = getattr(self, "te_lyrics", None) or getattr(getattr(self, "ui", None), "txt_lyrics", None)
            if te_l is not None and hasattr(te_l, "setPlainText"):
                te_l.setPlainText(lyrics)
        except Exception:
            pass

        # 3) 자동 태그(영문 정규화) + 수동 후보 안에서 picks 선택
        try:
            auto_en = normalize_tags_to_english(tags)  # 전체 후보(영문)
        except Exception:
            auto_en = [str(t) for t in tags]

        picks_from_ai = data.get("tags_pick") or []

        # _manual_option_set 이 함수이거나, Iterable일 수도 있으므로 모두 지원
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

        # ✅ 자동태그가 '켜져 있을 때만' UI에 반영
        try:
            if hasattr(self, "cb_auto_tags") and getattr(self.cb_auto_tags, "isChecked", lambda: False)():
                if hasattr(self, "_apply_auto_tags_to_ui"):
                    self._apply_auto_tags_to_ui(picks)
        except Exception as e:
            print("[TAGDBG] apply checks fail:", type(e).__name__, str(e), flush=True)

        # 수동 UI 활성/비활성은 기존 규칙 유지
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
            # 자동 태그 ON: ace_tags/tags_in_use/checked_tags/tags_effective 모두 채움
            meta["auto_tags"] = True
            meta["ace_tags"] = list(auto_en)
            meta["tags_in_use"] = list(picks)
            meta["checked_tags"] = list(picks)  # UI 반영/동기화를 위해
            meta["tags_effective"] = list(picks)  # 후속 파이프라인에서 바로 사용
            meta.pop("manual_tags", None)
        else:
            # 자동 태그 OFF: 현재 수동 체크들을 '단일 진실원천'으로 저장
            meta["auto_tags"] = False
            manual_checked = []
            try:
                if hasattr(self, "_collect_manual_checked_tags"):
                    manual_checked = list(self._collect_manual_checked_tags())
            except Exception:
                manual_checked = []
            meta["manual_tags"] = manual_checked
            # ✅ 핵심: 수동 선택을 효과 태그/체크 태그로도 동일 저장
            meta["checked_tags"] = list(manual_checked)
            meta["tags_effective"] = list(manual_checked)
            # 자동 관련 키는 제거
            meta.pop("tags_in_use", None)
            meta.pop("ace_tags", None)

        save_json(pj, meta)

        print("[TAGDBG] saved auto_tags:", meta.get("auto_tags"), flush=True)
        print("[TAGDBG] saved ace_tags len:", len(meta.get("ace_tags", [])), flush=True)
        print("[TAGDBG] saved tags_in_use len:", len(meta.get("tags_in_use", [])), flush=True)
        print("[TAGDBG] saved manual_tags len:", len(meta.get("manual_tags", [])), flush=True)
        print("[TAGDBG] saved checked_tags len:", len(meta.get("checked_tags", [])), flush=True)
        print("[TAGDBG] saved tags_effective len:", len(meta.get("tags_effective", [])), flush=True)
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
                required = [
                    p for p in sig.parameters.values()
                    if p.default is inspect._empty
                       and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                ]
                if len(required) == 0:
                    fn()
                elif len(required) == 1:
                    fn(str(pdir))
                else:
                    try:
                        fn(str(pdir))
                    except TypeError:
                        self.project_dir = str(pdir)
            except Exception:
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

    def _apply_project_meta(self, proj_dir: str) -> None:
        """
        project.json을 읽어 제목/가사/길이/변환/자동태그/프롬프트 UI에 반영.
        - 프롬프트: prompt_user(우선), 없으면 prompt 사용
        """
        from pathlib import Path
        try:
            from app.utils import load_json  # type: ignore
        except Exception:
            from utils import load_json  # type: ignore

        pj = Path(proj_dir) / "project.json"
        meta = load_json(pj, {}) or {}

        # 제목/가사
        title = (meta.get("title") or "").strip()
        if hasattr(self, "le_title"):
            self.le_title.setText(title)

        lyrics = (meta.get("lyrics") or "").strip()
        if hasattr(self, "te_lyrics"):
            self.te_lyrics.setPlainText(lyrics)

        # 길이
        try:
            self._apply_time_from_project_json()
        except Exception:
            pass

        # 변환(LLS) UI 동기화
        self._sync_convert_ui_from_meta(meta)

        # 자동태그 체크박스 동기화
        try:
            self._sync_auto_tags_checkbox_from_meta(meta)
        except Exception:
            pass

        # ✅ 프롬프트(긍정) UI 반영: prompt_user 우선, 없으면 prompt
        prompt_val = (meta.get("prompt_user") or meta.get("prompt") or "").strip()
        # 다양한 위젯 이름에 방어적으로 대응
        prompt_widget = (
                getattr(self, "le_prompt", None)
                or getattr(self, "te_prompt", None)
                or getattr(self, "te_prompt_pos", None)
        )
        try:
            if prompt_widget is not None:
                if hasattr(prompt_widget, "setText"):
                    prompt_widget.setText(prompt_val)
                elif hasattr(prompt_widget, "setPlainText"):
                    prompt_widget.setPlainText(prompt_val)
        except Exception:
            pass

    def _sync_convert_ui_from_meta(self, meta: dict) -> None:
        """
        meta(=project.json 내용)에서 lyrics_lls/lls_enabled를 읽어
        - 토글 버튼 체크 상태
        - 오른쪽 변환 에디터 표시/활성 상태
        - 에디터 텍스트
        를 동기화한다.

        정책: 패널은 항상 보이기(빈칸이라도), OFF일 땐 비활성화만.
        """
        btn = getattr(self, "btn_convert_toggle", None)
        te_conv = getattr(self, "te_lyrics_converted", None)

        # 상태 값
        lls_text = (meta.get("lyrics_lls") or "").strip()
        enabled = bool(meta.get("lls_enabled"))  # 사용자가 토글로 켠 상태
        on = enabled or bool(lls_text)  # 텍스트가 있으면 자연스럽게 ON으로 간주

        # 토글 버튼 체크만 반영
        try:
            if btn and hasattr(btn, "setChecked"):
                if hasattr(btn, "blockSignals"):
                    btn.blockSignals(True)
                btn.setChecked(on)
                if hasattr(btn, "blockSignals"):
                    btn.blockSignals(False)
        except Exception:
            pass

        # 에디터 표시/텍스트/활성 상태
        if te_conv:
            try:
                if hasattr(te_conv, "setVisible"):
                    te_conv.setVisible(True)  # 항상 보이기
            except Exception:
                pass
            try:
                if hasattr(te_conv, "setPlainText"):
                    te_conv.setPlainText(lls_text)
            except Exception:
                pass
            try:
                if hasattr(te_conv, "setEnabled"):
                    te_conv.setEnabled(on)  # OFF면 비활성화만
            except Exception:
                pass

        # 컨테이너(그룹박스/프레임)가 있으면 동일 정책 적용
        panel = None
        for name in ("grp_convert", "box_convert", "frame_convert", "gb_convert", "w_convert"):
            w = getattr(self, name, None)
            if w is not None:
                panel = w
                break
        if panel is not None:
            try:
                if hasattr(panel, "setVisible"):
                    panel.setVisible(True)  # 항상 보이기
            except Exception:
                pass
            try:
                if hasattr(panel, "setEnabled"):
                    panel.setEnabled(on)  # OFF면 비활성화
            except Exception:
                pass

    def _sync_auto_tags_checkbox_from_meta(self, meta: dict) -> None:
        """
        project.json의 auto_tags 값을 자동태그 체크박스(cb_auto_tags)에 반영한다.
        - auto_tags가 명시되어 있지 않으면 기존 UI 상태를 유지(불필요한 덮어쓰기 방지).
        - 값이 있다면 True/False 그대로 setChecked.
        """
        try:
            cb = getattr(self, "cb_auto_tags", None)
            if cb is None or not hasattr(cb, "setChecked"):
                return
            if "auto_tags" in meta:
                cb.setChecked(bool(meta.get("auto_tags")))
        except Exception:
            # UI 요소가 없거나 일시적 오류가 있어도 앱이 죽지 않도록 방어
            pass

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

    @staticmethod
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

        # ✅ 프로젝트 컨텍스트 준비 완료 플래그 (이후부터 태그 워치가 project.json과 동기화)
        try:
            setattr(self, "_project_context_ready", True)
        except Exception:
            pass


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
        from pathlib import Path
        import shutil
        import settings as _settings  # 소문자 별칭으로 사용

        safe_title = _sanitize_title_for_path(title)

        # 원본 폴더
        src_dir = Path(getattr(_settings, "COMFY_RESULT_ROOT", r"C:\comfyResult\shorts_make")) / safe_title
        if not src_dir.exists():
            return None

        # 목적지 폴더
        dst_dir = _resolve_audio_dir_from_template(getattr(_settings, "FINAL_OUT", ""), title)
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
                from video_build import (
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
    def on_video(self) -> None:
        """
        영상 생성:
          - UI에서 W/H/FPS/스텝 값을 읽어 build_shots_with_i2v로 전달.
          - run_job_with_progress_async를 사용하여 비동기 실행 및 실시간 로그 창 표시.
          - video_build.build_shots_with_i2v 기존 동작 100% 보존.
        """
        # 필요한 모듈 임포트
        from pathlib import Path
        from typing import Any, Dict, Optional, Callable
        from PyQt5 import QtWidgets
        import inspect # inspect 모듈은 시그니처 확인에 필요
        import traceback



        # --- 버튼 상태 관리 ---
        btn_video_widget: Optional[QtWidgets.QAbstractButton] = None
        for btn_name in ("btn_video", "btn_build_video"):
             widget_candidate = getattr(self, btn_name, None) or \
                                getattr(getattr(self, "ui", None), btn_name, None)
             if isinstance(widget_candidate, QtWidgets.QAbstractButton):
                  btn_video_widget = widget_candidate
                  break

        if btn_video_widget:
            btn_video_widget.setEnabled(False)

        try:
            # --- UI 값 읽기 ---
            ui_w: Optional[int] = None
            ui_h: Optional[int] = None
            ui_fps: Optional[int] = None
            ui_steps: Optional[int] = None

            def _get_ui_int_value(widget_name: str, data_attr: str = "currentData", default_val: Optional[int] = None) -> Optional[int]:
                """UI 위젯에서 정수 값을 안전하게 읽어옴."""
                widget = getattr(self, widget_name, None)
                if widget is not None:
                    try:
                        value_method = getattr(widget, data_attr, None)
                        if callable(value_method):
                            raw_value = value_method()
                            if callable(raw_value):
                                 print(f"[경고] UI 값 읽기 오류 ({widget_name}): 함수 반환.")
                                 return default_val
                            if raw_value is not None:
                                try:
                                    return int(raw_value) # type: ignore
                                except (ValueError, TypeError) as e_int_conv:
                                    print(f"[경고] UI 값 정수 변환 실패 ({widget_name}, 값: '{raw_value}'): {e_int_conv}")
                                    return default_val
                            else:
                                return default_val
                    except (AttributeError) as e_attr:
                         print(f"[경고] UI 위젯 속성 접근 실패 ({widget_name}.{data_attr}): {e_attr}")
                    except Exception as e_get_val_unexpected:
                         print(f"[경고] UI 값 읽기 중 예상치 못한 오류 ({widget_name}): {e_get_val_unexpected}")
                return default_val

            ui_w = _get_ui_int_value("cmb_img_w", "currentData")
            ui_h = _get_ui_int_value("cmb_img_h", "currentData")
            ui_fps = _get_ui_int_value("cmb_movie_fps", "currentData")
            ui_steps = _get_ui_int_value("spn_t2i_steps", "value")

            # --- 프로젝트 경로 확인 ---
            proj_dir_val: Optional[str] = None
            proj_dir_getter = getattr(self, "_current_project_dir", None)
            if callable(proj_dir_getter):
                try:
                    proj_dir_obj = proj_dir_getter()
                    if isinstance(proj_dir_obj, (str, Path)):
                        proj_dir_val = str(proj_dir_obj)
                except Exception as e_get_proj_inner:
                    print(f"[경고] 프로젝트 경로 얻기 실패 (_current_project_dir): {e_get_proj_inner}")
            if not proj_dir_val:
                 proj_dir_attr = getattr(self, "project_dir", None)
                 if isinstance(proj_dir_attr, (str, Path)):
                     proj_dir_val = str(proj_dir_attr)

            if not proj_dir_val:
                QtWidgets.QMessageBox.warning(self, "오류", "프로젝트 폴더가 선택되지 않았습니다.")
                return # finally 블록에서 버튼 활성화
            pdir = Path(proj_dir_val)

            # --- 백그라운드 작업 함수 정의 (콜백 인자 추가) ---
            def _job(progress_callback: Callable[[Dict[str, Any]], None]) -> None: # <-- 콜백 인자 명시
                """영상 생성을 수행하는 백그라운드 작업 함수."""
                build_func_local: Optional[Callable] = None # <-- 변수명 변경
                try:
                    from app.video_build import build_shots_with_i2v as build_func_imp # type: ignore
                    build_func_local = build_func_imp
                except (ImportError, ModuleNotFoundError, AttributeError):
                    try:
                         from video_build import build_shots_with_i2v as build_func_imp2 # type: ignore
                         build_func_local = build_func_imp2
                    except (ImportError, ModuleNotFoundError, AttributeError) as e_import_vb_inner:
                         raise ImportError(f"video_build.build_shots_with_i2v 로드 실패: {e_import_vb_inner}") from e_import_vb_inner

                tframes = 0
                sb_total_widget = getattr(self, "sb_total", None)
                if sb_total_widget is not None and hasattr(sb_total_widget, "value"):
                    try:
                        tframes = int(sb_total_widget.value())
                    except (TypeError, ValueError):
                        tframes = 0
                if tframes <= 0:
                     progress_callback({"msg": "[경고] total_frames가 0 이하입니다."})

                try:
                    sig_build_inner = inspect.signature(build_func_local) # <-- 변수명 변경
                    build_kwargs_inner: Dict[str, Any] = {
                        "project_dir": str(pdir),
                        "total_frames": tframes,
                        "on_progress": progress_callback # <-- 전달받은 콜백 명시적으로 사용
                    }
                    # UI 값 인자 추가 (시그니처 확인 후, None이 아닐 때만)
                    if "ui_width" in sig_build_inner.parameters and ui_w is not None: build_kwargs_inner["ui_width"] = ui_w
                    if "ui_height" in sig_build_inner.parameters and ui_h is not None: build_kwargs_inner["ui_height"] = ui_h
                    if "ui_fps" in sig_build_inner.parameters and ui_fps is not None: build_kwargs_inner["ui_fps"] = ui_fps
                    if "ui_steps" in sig_build_inner.parameters and ui_steps is not None: build_kwargs_inner["ui_steps"] = ui_steps

                    build_func_local(**build_kwargs_inner) # <-- 변수명 변경

                except TypeError as e_type_build_inner:
                    progress_callback({"msg": f"[경고] build_shots_with_i2v 호출 시그니처 불일치 ({e_type_build_inner}), UI 값 없이 호출 시도."})
                    try:
                        build_func_local(str(pdir), tframes, on_progress=progress_callback) # <-- 변수명 변경
                    except TypeError:
                         progress_callback({"msg": "[경고] on_progress 인자도 실패, 인자 없이 호출 시도."})
                         build_func_local(str(pdir), tframes) # type: ignore[call-arg] # <-- 변수명 변경
                    except Exception as e_fallback_call_inner:
                         raise RuntimeError(f"build_shots_with_i2v 최종 호출 실패: {e_fallback_call_inner}") from e_fallback_call_inner
                except Exception as e_build_other_inner:
                    raise RuntimeError(f"build_shots_with_i2v 실행 오류: {e_build_other_inner}") from e_build_other_inner

            # --- 작업 완료 콜백 ---
            def _done(ok: bool, _payload: Any, err: Optional[Exception]) -> None: # <-- _payload 사용 안 함 명시
                 """작업 완료 후 UI 업데이트 및 메시지 표시."""
                 if not ok and err:
                      err_type_name = type(err).__name__
                      err_message = str(err)
                      print(f"[오류] 영상 생성 작업 실패: {err_type_name}: {err_message}")
                      print(traceback.format_exc())
                      QtWidgets.QMessageBox.critical(self, "영상 생성 오류", f"오류 발생:\n{err_type_name}: {err_message}\n\n상세 내용은 콘솔 로그를 확인하세요.")
                 elif ok:
                      print("[정보] 영상 생성 작업 완료.")
                      QtWidgets.QMessageBox.information(self, "완료", "영상 생성 작업이 완료되었습니다.")

            # --- 진행창 유틸 로드 ---
            run_async_local: Optional[Callable] = None # <-- 변수명 변경
            try:
                from app.utils import run_job_with_progress_async as run_async_imp # type: ignore
                run_async_local = run_async_imp
            except (ImportError, ModuleNotFoundError, AttributeError):
                try:
                    from utils import run_job_with_progress_async as run_async_imp2 # type: ignore
                    run_async_local = run_async_imp2
                except (ImportError, ModuleNotFoundError, AttributeError):
                    run_async_local = None

            if run_async_local is None:
                # 유틸 로드 실패 시 동기 실행
                print("[경고] run_job_with_progress_async 로드 실패, 동기 실행합니다.")
                def _notify_sync(data: Dict[str, Any]) -> None:
                     print(f"[I2V][Sync] {data.get('msg', '')}")
                try:
                    _job(_notify_sync)
                    _done(True, None, None)
                except Exception as e_sync_job_inner:
                    _done(False, None, e_sync_job_inner)
                return # 동기 실행 후 종료

            # --- run_async 호출 준비 ---
            # run_job_with_progress_async 함수 시그니처 확인 (utils.py 기준)
            # def run_job_with_progress_async(owner, title, job, *, tail_file=None, on_done=None)
            kw_run_async: Dict[str, Any] = {}
            try:
                 sig_run_async_check = inspect.signature(run_async_local)
                 if "tail_file" in sig_run_async_check.parameters:
                      kw_run_async["tail_file"] = None # 영상 생성은 tail 불필요
                 if "on_done" in sig_run_async_check.parameters:
                      kw_run_async["on_done"] = _done
            except (TypeError, ValueError) as e_sig_check:
                 print(f"[경고] run_async 시그니처 분석 실패 (호출은 시도): {e_sig_check}")
                 # 기본 키워드 인자 설정 (on_done은 중요하므로 포함 시도)
                 kw_run_async = {"tail_file": None, "on_done": _done}

            # --- run_async 실행 (정확한 인자 전달) ---
            try:
                 # utils.py 시그니처에 맞춰 owner, title, job을 위치 인자로 전달
                 run_async_local(self, "영상 생성", _job, **kw_run_async) # <-- 호출 방식 수정
            except Exception as e_run_call_final:
                 # 호출 실패 시 오류 로깅 및 동기 실행
                 print(f"[오류] run_job_with_progress_async 호출 실패: {e_run_call_final}")
                 print("[경고] 동기 실행으로 전환합니다.")
                 def _notify_sync_fallback(data: Dict[str, Any]) -> None:
                      print(f"[I2V][SyncFallback] {data.get('msg', '')}")
                 try:
                      _job(_notify_sync_fallback)
                      _done(True, None, None)
                 except Exception as e_sync_job_fallback_inner:
                      _done(False, None, e_sync_job_fallback_inner)

        except Exception as e_outer_inner:
             # on_video 함수 자체의 최상위 예외 처리
             print(f"[오류] on_video 실행 중 오류 발생: {type(e_outer_inner).__name__}: {e_outer_inner}")
             print(traceback.format_exc())
             QtWidgets.QMessageBox.critical(self, "오류", f"영상 생성 시작 중 오류 발생:\n{e_outer_inner}")

        finally:
            # 버튼 활성화 (항상 실행)
            if btn_video_widget:
                 try:
                      btn_video_widget.setEnabled(True)
                 except RuntimeError: # 위젯 소멸 등 예외
                      pass
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
                lines = [
                    "누락 이미지가 있어 중단합니다.",
                    "",
                    "[누락 이미지]",
                    *missing[:30],
                    *([f"... (총 {len(missing)}개 중 30개만 표시)"] if len(missing) > 30 else []),
                    *(["", "[2명 이상 캐릭터가 필요한 장면(참고)]", ", ".join(multi_char)] if multi_char else []),
                ]
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
    """MainWindow에 _add_render_prefs_controls / _save_ui_prefs_to_project (및 테스트 버튼) 없으면 주입."""
    from pathlib import Path
    from PyQt5 import QtWidgets

    # 안전 import (app 패키지/단독 실행 모두 고려)
    try:
        from app import settings as settings_mod
        from app.utils import load_json, save_json, sanitize_title as sanitize_title_fn
    except Exception:
        import settings as settings_mod  # type: ignore
        from utils import load_json, save_json  # type: ignore

        def sanitize_title_fn(x: str) -> str:
            return "".join(ch for ch in x if ch.isalnum() or ch in " _-").strip()

    # 이미 있으면 건너뜀
    if (
        hasattr(MainWindow, "_add_render_prefs_controls")
        and hasattr(MainWindow, "_save_ui_prefs_to_project")
    ):
        return

    def _guess_project_dir(self) -> Path:
        """현재 프로젝트 폴더 추정: _current_project_dir() → FINAL_OUT → BASE_DIR/title"""
        # 1) UI 제공 메서드
        if hasattr(self, "_current_project_dir"):
            try:
                d = self._current_project_dir()
                if d:
                    return Path(d)
            except Exception:
                pass
        # 2) 제목 기반
        try:
            title_val = sanitize_title_fn(self.le_title.text().strip())
        except Exception:
            title_val = ""
        if not title_val:
            title_val = "무제"
        # 2-1) FINAL_OUT 템플릿 우선
        final_tpl = getattr(settings_mod, "FINAL_OUT", "")
        if final_tpl and "[title]" in final_tpl:
            return Path(final_tpl.replace("[title]", title_val))
        # 2-2) BASE_DIR/[title]
        base_dir_val = getattr(settings_mod, "BASE_DIR", ".")
        return Path(base_dir_val) / title_val

    # ==== 메서드 정의: 드롭다운 UI 추가 ====
    def _add_render_prefs_controls(self, parent_layout: QtWidgets.QBoxLayout) -> None:
        """렌더 설정(W, H, FPS, 프리셋, 스텝) UI를 생성하고 부모 레이아웃에 추가합니다."""
        # settings 모듈을 안전하게 참조 (s_mod_prefs 별칭 사용)
        try:
            from app import settings as s_mod_prefs # type: ignore
        except ImportError:
            try:
                import settings as s_mod_prefs # type: ignore
            except ImportError:
                print("[경고] _add_render_prefs_controls: settings 모듈 로드 실패, 기본값 사용")
                class SettingsFallbackPrefs:
                    IMAGE_SIZE_CHOICES = [480, 512, 720, 832, 960, 1024, 1080, 1280, 1440, 1920, 2560]
                    DEFAULT_IMG_SIZE = (1080, 1920)
                    MOVIE_FPS_CHOICES = [24, 30, 60]
                    DEFAULT_MOVIE_FPS = 30
                s_mod_prefs = SettingsFallbackPrefs()

        #--- 내부 유틸 함수 (load_json) ---
        _load_json_local_prefs: Callable # 타입 명시
        try:
            from app.utils import load_json as _load_json_local_prefs # type: ignore
        except ImportError:
            try:
                from utils import load_json as _load_json_local_prefs # type: ignore
            except ImportError:
                print("[오류] _add_render_prefs_controls: load_json 함수 로드 실패")
                def _load_json_fb_prefs(p, default=None):
                    try: return json.loads(Path(p).read_text(encoding="utf-8"))
                    except Exception: return default
                _load_json_local_prefs = _load_json_fb_prefs

        #--- 내부 유틸 함수 (_guess_project_dir) ---
        _guess_project_dir_local: Callable[[], Path] # 타입 명시
        try:
            _guess_project_dir_local = getattr(self, "_guess_project_dir")
            if not callable(_guess_project_dir_local):
                 raise AttributeError("_guess_project_dir 메서드를 찾을 수 없습니다.")
        except AttributeError as e_guess_dir_local: # 변수명 변경
             print(f"[오류] _add_render_prefs_controls: {e_guess_dir_local}")
             def _guess_project_dir_fb() -> Path: return Path(".")
             _guess_project_dir_local = _guess_project_dir_fb


        grp = QtWidgets.QGroupBox("렌더 설정")
        row = QtWidgets.QHBoxLayout(grp)

        # 기본 컨트롤
        self.cmb_img_w = QtWidgets.QComboBox()
        self.cmb_img_h = QtWidgets.QComboBox()
        self.cmb_movie_fps = QtWidgets.QComboBox()

        self.cmb_img_w.setToolTip("이미지 가로 (width)")
        self.cmb_img_h.setToolTip("이미지 세로 (height)")
        self.cmb_movie_fps.setToolTip("타깃 FPS (i2v/렌더)")

        # --- ▼▼▼ 프리셋 값들(832, 1024 등)이 포함되도록 수정 ▼▼▼ ---
        default_w_val, default_h_val = getattr(s_mod_prefs, "DEFAULT_IMG_SIZE", (1080, 1920))
        # 프리셋에 사용된 모든 W 값을 size_choices_val에 포함시킴
        preset_widths = {720, 832, 1080, 1280, 1920, 512, 1024}
        preset_heights = {1280, 1472, 1920, 720, 1080, 512, 1024}

        size_choices_conf = getattr(s_mod_prefs, "IMAGE_SIZE_CHOICES", [480, 520, 720, 960, 1080, 1280, 1440])
        size_choices_set = set(int(w) for w in size_choices_conf if str(w).isdigit())
        size_choices_set.update(preset_widths) # 프리셋 W 값 추가
        size_choices_set.add(int(default_w_val)) # 기본 W 값 추가
        size_choices_val = sorted(list(size_choices_set)) # 정렬된 리스트

        # H 값 목록 생성
        h_candidates_set = {int(round(w * 16 / 9)) for w in size_choices_val} # 16:9 비율
        h_candidates_set.update({int(round(w * 9 / 16)) for w in size_choices_val}) # 9:16 비율
        h_candidates_set.update(preset_heights) # 프리셋 H 값 추가
        h_candidates_set.add(int(default_h_val)) # 기본 H 값 추가
        h_candidates_val = sorted(list(h_candidates_set)) # 정렬된 리스트
        # --- ▲▲▲ 목록 수정 끝 ▲▲▲ ---

        fps_choices_val = getattr(s_mod_prefs, "MOVIE_FPS_CHOICES", [24, 30, 60])
        default_fps_val = int(getattr(s_mod_prefs, "DEFAULT_MOVIE_FPS", 30))

        # W/H 콤보박스 채우기
        try:
            for w_val_item in size_choices_val:
                self.cmb_img_w.addItem(str(w_val_item), int(w_val_item))
            for h_val_item in h_candidates_val:
                self.cmb_img_h.addItem(str(h_val_item), int(h_val_item))
        except (ValueError, TypeError) as e_size_fill_local: # 변수명 변경
             print(f"[경고] W/H 콤보박스 채우기 오류: {e_size_fill_local}")
             if self.cmb_img_w.count() == 0: self.cmb_img_w.addItem(str(default_w_val), int(default_w_val))
             if self.cmb_img_h.count() == 0: self.cmb_img_h.addItem(str(default_h_val), int(default_h_val))

        # FPS 콤보박스 채우기
        try:
            valid_fps_choices = [int(f) for f in fps_choices_val if str(f).isdigit()]
            if default_fps_val not in valid_fps_choices: # 기본값이 목록에 없으면 추가
                 valid_fps_choices.append(default_fps_val)
                 valid_fps_choices.sort()
            for fps_val_item in valid_fps_choices:
                self.cmb_movie_fps.addItem(str(fps_val_item), int(fps_val_item))
        except (ValueError, TypeError) as e_fps_fill_local: # 변수명 변경
             print(f"[경고] FPS 콤보박스 채우기 오류: {e_fps_fill_local}")
             if self.cmb_movie_fps.count() == 0: self.cmb_movie_fps.addItem(str(default_fps_val), int(default_fps_val))


        # 해상도 프리셋 + 스텝
        self.cmb_res_preset = QtWidgets.QComboBox()
        self.cmb_res_preset.setToolTip("해상도 프리셋(선택 시 W/H 자동 설정)")

        presets_data = [
            ("Shorts 9:16 · 720×1280", 720, 1280, "shorts_720x1280"),
            ("Shorts 9:16 · 832×1472", 832, 1472, "shorts_832x1472"),
            ("Shorts 9:16 · 1080×1920", 1080, 1920, "shorts_1080x1920"),
            ("Landscape 16:9 · 1280×720", 1280, 720, "land_1280x720"),
            ("Landscape 16:9 · 1920×1080", 1920, 1080, "land_1920x1080"),
            ("Square 1:1 · 512×512", 512, 512, "square_512"),
            ("Square 1:1 · 1024×1024", 1024, 1024, "square_1024"),
            ("맞춤(커스텀)", -1, -1, "custom"), # '맞춤'이 항상 마지막
        ]
        for label_text, w_preset, h_preset, key_preset in presets_data:
            self.cmb_res_preset.addItem(label_text, (w_preset, h_preset, key_preset))

        self.spn_t2i_steps = QtWidgets.QSpinBox()
        self.spn_t2i_steps.setRange(1, 200)
        self.spn_t2i_steps.setValue(12) # 기본값 12
        self.spn_t2i_steps.setToolTip("샘플링 스텝 수(확산 단계 수)")

        # project.json 초기값 반영
        proj_dir_current = _guess_project_dir_local()
        pj_current = proj_dir_current / "project.json"
        meta_current = _load_json_local_prefs(pj_current, {}) if pj_current.exists() else {}
        ui_prefs_current = meta_current.get("ui_prefs") or {}

        def _set_combo_safe(combo: QtWidgets.QComboBox, val_to_set: int, fallback_val: int):
            """콤보박스 값을 안전하게 설정 (데이터 값 기준)."""
            try:
                val_int = int(val_to_set)
                idx_found = combo.findData(val_int)
                if idx_found < 0:
                    idx_found = combo.findData(int(fallback_val))
                combo.setCurrentIndex(idx_found if idx_found >= 0 else 0)
            except (ValueError, TypeError):
                combo.setCurrentIndex(0)

        ui_w_val = (ui_prefs_current.get("image_size") or [default_w_val, default_h_val])[0]
        ui_h_val = (ui_prefs_current.get("image_size") or [default_w_val, default_h_val])[1]
        ui_fps_val = ui_prefs_current.get("movie_fps") or default_fps_val

        _set_combo_safe(self.cmb_img_w, ui_w_val, default_w_val)
        _set_combo_safe(self.cmb_img_h, ui_h_val, default_h_val)
        _set_combo_safe(self.cmb_movie_fps, ui_fps_val, default_fps_val)

        preset_key_from_json = str((ui_prefs_current.get("resolution_preset") or "custom"))
        steps_from_json = int(ui_prefs_current.get("t2i_steps") or 12)
        self.spn_t2i_steps.setValue(steps_from_json)

        def _lock_wh_inputs(lock: bool) -> None:
            """W/H 콤보박스 활성화/비활성화 및 툴팁 설정."""
            self.cmb_img_w.setEnabled(not lock)
            self.cmb_img_h.setEnabled(not lock)
            tip_text = (
                "프리셋을 '맞춤(커스텀)'으로 바꾸면 해상도를 수정할 수 있습니다."
                if lock else "W/H를 직접 선택하세요."
            )
            self.cmb_img_w.setToolTip(tip_text)
            self.cmb_img_h.setToolTip(tip_text)

        def _apply_preset_to_wh_inputs() -> None:
            """프리셋 콤보박스 변경 시 W/H 콤보박스 값 업데이트 및 잠금 처리."""
            current_data = self.cmb_res_preset.currentData()
            if not (isinstance(current_data, tuple) and len(current_data) == 3):
                 return

            w_sel_preset, h_sel_preset, preset_key_sel = current_data
            if preset_key_sel == "custom":
                _lock_wh_inputs(False)
                return

            idx_w_preset = self.cmb_img_w.findData(int(w_sel_preset))
            if idx_w_preset >= 0:
                self.cmb_img_w.setCurrentIndex(idx_w_preset)
            else:
                # 목록에 없으면 경고 (하지만 목록 생성 로직이 수정되어 이 경우는 드물 것)
                print(f"[경고] 프리셋 W 값({w_sel_preset})이 콤보박스 목록에 없습니다.")

            idx_h_preset = self.cmb_img_h.findData(int(h_sel_preset))
            if idx_h_preset >= 0:
                self.cmb_img_h.setCurrentIndex(idx_h_preset)
            else:
                print(f"[경고] 프리셋 H 값({h_sel_preset})이 콤보박스 목록에 없습니다.")
            _lock_wh_inputs(True)

        # 프리셋 초기값 설정
        kidx_preset_init = 0
        for idx_preset_loop in range(self.cmb_res_preset.count()):
            item_data = self.cmb_res_preset.itemData(idx_preset_loop)
            if isinstance(item_data, tuple) and len(item_data) == 3:
                _, _, key_val_loop = item_data
                if key_val_loop == preset_key_from_json:
                    kidx_preset_init = idx_preset_loop
                    break
        self.cmb_res_preset.setCurrentIndex(kidx_preset_init)
        # 시그널 연결 (프리셋 변경 -> W/H 업데이트)
        self.cmb_res_preset.currentIndexChanged.connect(_apply_preset_to_wh_inputs)
        # 초기 잠금 상태 적용
        _apply_preset_to_wh_inputs()

        # 레이아웃
        row.addWidget(QtWidgets.QLabel("W")); row.addWidget(self.cmb_img_w)
        row.addWidget(QtWidgets.QLabel("H")); row.addWidget(self.cmb_img_h)
        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel("FPS")); row.addWidget(self.cmb_movie_fps)
        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel("프리셋")); row.addWidget(self.cmb_res_preset)
        row.addWidget(QtWidgets.QLabel("스텝")); row.addWidget(self.spn_t2i_steps)
        row.addStretch(1)
        parent_layout.addWidget(grp)

        # 변경 시 저장 시그널 연결
        self.cmb_img_w.currentIndexChanged.connect(self._save_ui_prefs_to_project)
        self.cmb_img_h.currentIndexChanged.connect(self._save_ui_prefs_to_project)
        self.cmb_movie_fps.currentIndexChanged.connect(self._save_ui_prefs_to_project)
        # self.cmb_res_preset.currentIndexChanged.connect(self._save_ui_prefs_to_project) # <-- 이 라인 제거됨 (유지)
        self.spn_t2i_steps.valueChanged.connect(self._save_ui_prefs_to_project)

    # ==== 메서드 정의: project.json 저장 ====
    def _save_ui_prefs_to_project(self) -> None:
        proj_dir = _guess_project_dir(self)
        pj = proj_dir / "project.json"
        meta = load_json(pj, {}) if pj.exists() else {}
        ui = meta.get("ui_prefs") or {}

        w_sel = int(self.cmb_img_w.currentData())
        h_sel = int(self.cmb_img_h.currentData())
        fps_sel = int(self.cmb_movie_fps.currentData())

        preset_key_sel = "custom"
        data_val = self.cmb_res_preset.currentData()
        if isinstance(data_val, tuple) and len(data_val) == 3:
            _, _, preset_key_sel = data_val  # (w, h, key)

        ui["image_size"] = [w_sel, h_sel]
        ui["movie_fps"] = fps_sel
        ui["resolution_preset"] = str(preset_key_sel)
        ui["t2i_steps"] = int(self.spn_t2i_steps.value())

        meta["ui_prefs"] = ui
        save_json(pj, meta)

    # === 옵션: 누락 이미지 생성 버튼 핸들러도 주입 ===
    def on_click_test2_1_generate_missing_images(self) -> None:
        from pathlib import Path
        from utils import run_job_with_progress_async
        try:
            from app.video_build import build_missing_images_from_story
        except Exception:
            from video_build import build_missing_images_from_story  # type: ignore

        # 상수 직접 import 대신 getattr로 안전 접근
        try:
            import settings as _settings_local  # type: ignore
        except Exception:
            from app import settings as _settings_local  # type: ignore
        comfy_log_file = getattr(_settings_local, "COMFY_LOG_FILE", None)

        story_path = Path(self.txt_story_path.text()).resolve()
        ui_w = int(self.cmb_img_w.currentData())
        ui_h = int(self.cmb_img_h.currentData())
        steps_val = int(self.spn_t2i_steps.value())

        def _job(on_progress):
            return build_missing_images_from_story(
                story_path,
                ui_width=ui_w,
                ui_height=ui_h,
                steps=steps_val,
                timeout_sec=300,
                poll_sec=1.5,
                workflow_path=None,  # JSONS_DIR/nunchaku_qwen_image_swap.json 자동
                on_progress=on_progress,
            )

        def _done(ok, payload, err):
            from PyQt5 import QtWidgets
            if not ok:
                QtWidgets.QMessageBox.critical(self, "이미지 생성 실패", str(err))
                return
            cnt = len(payload or [])
            QtWidgets.QMessageBox.information(self, "완료", f"새 이미지 {cnt}개 생성")

        run_job_with_progress_async(self, "테스트2_1: 누락 이미지 생성", _job, tail_file=comfy_log_file, on_done=_done)

    # 클래스에 바인딩
    setattr(MainWindow, "_add_render_prefs_controls", _add_render_prefs_controls)
    setattr(MainWindow, "_save_ui_prefs_to_project", _save_ui_prefs_to_project)
    setattr(MainWindow, "on_click_test2_1_generate_missing_images", on_click_test2_1_generate_missing_images)


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
