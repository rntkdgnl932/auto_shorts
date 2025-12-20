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
from PyQt5.QtGui import QFont
from typing import Any, Dict, List, Optional, Callable, Callable as _CallableType, Set, Tuple
import math
import sys
import faulthandler
import datetime
from app.utils import sanitize_title as _sanitize_title_fn, normalize_tags_to_english, audio_duration_sec, AI, save_story_overwrite_with_prompts, _normalize_maked_title_root, sanitize_title
from app.utils import load_json as _lj, save_json as _sj
from app.utils import load_json as _load_json
from app.utils import sanitize_title as _sanitize
from app.utils import run_job_with_progress_async as run_async_imp
from app.utils import run_job_with_progress_async
from app.utils import load_json, save_json, sanitize_title as sanitize_title_fn
from app.video_build import build_and_merge_full_video, add_subtitles_with_ffmpeg, xfade_concat, concatenate_scene_clips
from app.video_build import build_shots_with_i2v as build_func_imp
from app.video_build import build_video_json_with_gap_policy, retry_cut_audio_for_scene, fill_prompt_movie_with_ai, build_missing_images_from_story

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPlainTextEdit, QTextEdit, QFontComboBox

from app.audio_sync import generate_music_with_acestep, sync_lyrics_with_whisper_pro, build_story_json
from app.audio_sync import _submit_and_wait as _submit_and_wait_comfy
import re
_CANON_RE = re.compile(r"[^a-z0-9]+")


import app.settings as _settings

from app.settings import FFPROBE_EXE, BASE_DIR, CHARACTER_DIR, COMFY_HOST, JSONS_DIR, COMFY_INPUT_DIR, FFMPEG_EXE
from video_build import (
                    build_image_json, build_movie_json,
                    apply_intro_outro_to_story_json,
                    apply_intro_outro_to_image_json,
                    apply_intro_outro_to_movie_json,
                )
from app.lyrics_gen import create_project_files, normalize_sections, generate_title_lyrics_tags
from json import loads, dumps
from app.story_enrich import apply_gpt_to_story_v11
import subprocess
import requests
import functools
from PyQt5 import QtGui

import random  # ★ 랜덤 시드
from collections.abc import Iterable

import os
from PyQt5 import QtCore
from json import JSONDecodeError
import json
import re
import uuid, time, shutil
from pathlib import Path
from pathlib import Path as _p
from PyQt5 import QtWidgets
import inspect
import traceback
from app import settings as settings_mod





_LOG_DIR = Path(getattr(settings_mod, "BASE_DIR", ".")) / "_debug"
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
# ───────────────────────── 제이슨 수정 ─────────────────────────

# ───────────────────────── JSON 편집 다이얼로그 (신규 추가) ─────────────────────────


try:
    _Cursor_PointingHand = Qt.PointingHandCursor
    _Button_Left = Qt.LeftButton
except (NameError, AttributeError):
    print("[FATAL] ScenePromptEditDialog: Qt 속성을 로드할 수 없습니다.")
    _Cursor_PointingHand = 13  # Qt.PointingHandCursor의 기본값
    _Button_Left = 1  # Qt.LeftButton의 기본값


class ClickableLabel(QtWidgets.QLabel):
    """클릭 시 'clicked' 시그널을 방출하는 QLabel"""
    clicked = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setCursor(_Cursor_PointingHand)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == _Button_Left:
            self.clicked.emit()
        super().mousePressEvent(event)


class ScenePromptEditDialog(QtWidgets.QDialog):
    """
    [수정됨 v36] (사용자 요청 사항이 모두 통합된 최종본)
    - 캐릭터 스타일 편집기 (추가/삭제, 한 줄 QLineEdit, 레이아웃 압축)
    - 씬 라벨 한 줄 표시 (가사 '...' 제거, 캐릭터 표시)
    - direct_prompt 높이 축소
    - "업데이트" 시 character_styles 및 루트 characters 갱신
    - "삭제" 버튼 충돌 해결 (deleteLater 제거)
    - 이미지 미리보기, AI 요청, 이미지/영상 삭제 기능 유지
    """

    _AI_QUALITY_TAGS = "photorealistic, cinematic lighting, high detail, 8k, masterpiece"
    _AI_DEFAULT_NEGATIVE_TAGS = "lowres, bad anatomy, bad proportions, extra limbs, extra fingers, missing fingers, jpeg artifacts, signature, logo, nsfw, text, letters, typography, watermark"

    def __init__(self, json_path: Path, ai_instance: AI, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.json_path = json_path
        self.ai_instance = ai_instance
        self.full_video_data: dict = {}
        self.scenes_data: list = []
        self.widget_map: List[Tuple[str, QtWidgets.QTextEdit]] = []
        self.style_widget_map: Dict[str, Tuple[QtWidgets.QLabel, QtWidgets.QWidget]] = {}
        self.character_styles_group: QtWidgets.QWidget | None = None

        self.current_page = 0
        self.total_pages = 0
        self.PAGE_SIZE = 4

        # === [v41 수정] 150px 썸네일 크기로 복귀 ===
        self.THUMBNAIL_SIZE = 150
        # ======================================

        self.scene_char_widgets: Dict[str, QtWidgets.QListWidget] = {}

        self.setWindowTitle(f"Scene 프롬프트 편집: {self.json_path.name}")
        self.setMinimumSize(900, 750)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        # 1. 메인 레이아웃 (v26 수정: 간격 2px)
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(2)

        # 2. 씬 목록을 담을 QStackedWidget (페이지 전환용)
        self.stacked_widget = QtWidgets.QStackedWidget()
        # (이하 __init__의 나머지 코드는 v36과 동일합니다)

        # 3. 페이지네이션 컨트롤
        pagination_layout = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton("< 이전")
        self.page_label = QtWidgets.QLabel("Page 0 / 0")
        self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.btn_next = QtWidgets.QPushButton("다음 >")
        pagination_layout.addWidget(self.btn_prev)
        pagination_layout.addStretch(1)
        pagination_layout.addWidget(self.page_label)
        pagination_layout.addStretch(1)
        pagination_layout.addWidget(self.btn_next)

        # 4. 하단 버튼 레이아웃
        button_layout = QtWidgets.QHBoxLayout()
        self.btn_ai_request = QtWidgets.QPushButton("AI 요청")
        self.btn_update = QtWidgets.QPushButton("업데이트")
        self.btn_cancel = QtWidgets.QPushButton("닫기")
        self.btn_ai_request.setToolTip(
            "현재 페이지의 'direct_prompt' 내용을 기반으로\nAI에게 'prompt'(한국어), 'prompt_img', 'prompt_movie' 3개 필드를 새로 요청합니다.")
        button_layout.addStretch(1)
        button_layout.addWidget(self.btn_cancel)
        button_layout.addWidget(self.btn_ai_request)
        button_layout.addWidget(self.btn_update)

        # 5. 메인 레이아웃에 위젯 추가
        main_layout.addWidget(self.stacked_widget)
        main_layout.addLayout(pagination_layout)
        main_layout.addLayout(button_layout)

        # 6. 시그널 연결
        self.btn_update.clicked.connect(self.on_update_and_close)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_prev.clicked.connect(self.on_prev_page)
        self.btn_next.clicked.connect(self.on_next_page)
        self.btn_ai_request.clicked.connect(self.on_ai_request)

        # 7. JSON 데이터 로드 및 UI 빌드 (이 함수가 5~7초 소요)
        self.load_and_build_ui()

    @staticmethod
    def _reindex_char_list(list_widget: QtWidgets.QListWidget) -> None:
        """
        [신규] QListWidget의 항목을 읽어 인덱스를 0부터 순차적으로 다시 매깁니다.
        (예: ["f:0", "m:2"] -> ["f:0", "m:1"])
        """
        current_items = []
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if item:
                # "female_01:2" -> "female_01"
                char_id = item.text().split(":")[0]
                current_items.append(char_id)

        # 리스트를 비우고, 새 인덱스로 다시 채웁니다.
        list_widget.clear()
        for new_index, char_id in enumerate(current_items):
            list_widget.addItem(f"{char_id}:{new_index}")

    def on_add_char_to_scene(self, list_widget: QtWidgets.QListWidget):
        """[신규] 씬에 캐릭터 "추가" 버튼 핸들러"""

        # 1. 마스터 캐릭터 목록 (상단 편집기에서 ID만 가져옴)
        if not hasattr(self, "style_widget_map") or not self.style_widget_map:
            QtWidgets.QMessageBox.warning(self, "오류", "전체 캐릭터 스타일 목록이 없습니다.")
            return

        master_char_ids = list(self.style_widget_map.keys())  # ["female_01", "female_02"]

        # 2. 현재 씬에 이미 추가된 캐릭터 ID 목록
        currently_in_scene = set()
        for i in range(list_widget.count()):
            item_text = list_widget.item(i).text()
            currently_in_scene.add(item_text.split(":")[0])  # "female_01"

        # 3. 마스터 목록에서 이미 추가된 캐릭터를 제외
        available_to_add = [cid for cid in master_char_ids if cid not in currently_in_scene]

        if not available_to_add:
            QtWidgets.QMessageBox.information(self, "알림", "모든 마스터 캐릭터가 이미 이 씬에 추가되었습니다.")
            return

        # 4. 사용자에게 선택 요청
        char_id_to_add, ok = QtWidgets.QInputDialog.getItem(
            self,
            "씬에 캐릭터 추가",
            "추가할 캐릭터를 선택하세요:",
            available_to_add,
            0,
            False  # 편집 불가능
        )

        if ok and char_id_to_add:
            # 5. 새 인덱스 계산 및 추가
            next_index = list_widget.count()
            list_widget.addItem(f"{char_id_to_add}:{next_index}")
            # (인덱스 재정렬은 삭제 시에만 수행)

    def on_del_char_from_scene(self, list_widget: QtWidgets.QListWidget):
        """[신규] 씬에서 캐릭터 "삭제" 버튼 핸들러 (인덱스 재정렬 포함)"""

        selected_item = list_widget.currentItem()
        if not selected_item:
            QtWidgets.QMessageBox.warning(self, "오류", "삭제할 캐릭터를 씬 캐릭터 목록에서 선택하세요.")
            return

        # 1. UI에서 항목 제거
        row = list_widget.row(selected_item)
        list_widget.takeItem(row)

        # 2. [핵심] 인덱스 재정렬
        self._reindex_char_list(list_widget)


    def show_large_image(self, path_str: str):
        """(v17 수정 유지) 썸네일 클릭 시 800x800 박스에 맞게 이미지 표시"""
        if not path_str:
            QtWidgets.QMessageBox.information(self, "미리보기", "이 씬에는 이미지 경로가 지정되지 않았습니다.")
            return

        pixmap = QtGui.QPixmap(path_str)
        if pixmap.isNull():
            QtWidgets.QMessageBox.warning(self, "미리보기 오류", f"이미지를 불러올 수 없습니다:\n{path_str}")
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"이미지 미리보기: {Path(path_str).name}")
        label = QtWidgets.QLabel()
        max_preview_w, max_preview_h = 800, 800
        img_w, img_h = pixmap.width(), pixmap.height()

        if img_w > max_preview_w or img_h > max_preview_h:
            scaled_pixmap = pixmap.scaled(
                max_preview_w, max_preview_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
            dialog.resize(scaled_pixmap.width() + 20, scaled_pixmap.height() + 20)
        else:
            label.setPixmap(pixmap)
            dialog.resize(img_w + 20, img_h + 20)

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(label)
        dialog.exec_()

    def _add_character_style_row(self, char_id: str, style_text: str):
        """(v19 추가) 캐릭터 스타일 폼에 한 줄(라벨, 입력창, 삭제 버튼)을 추가"""
        if not hasattr(self, "character_styles_form_layout"):
            return

        label = QtWidgets.QLabel(f"<b>{char_id}</b>")
        label.setToolTip(f"캐릭터 ID: {char_id}")

        edit = QtWidgets.QLineEdit(style_text)
        font = QtGui.QFont()
        font.setFamily("Courier" if "Courier" in QtGui.QFontDatabase().families() else "Monospace")
        font.setPointSize(10)
        edit.setFont(font)
        edit.setToolTip(f"ID: {char_id}\n스타일: {style_text}")
        edit.setMinimumWidth(500)

        delete_button = QtWidgets.QPushButton("삭제")

        field_container = QtWidgets.QWidget()
        field_layout = QtWidgets.QHBoxLayout(field_container)
        field_layout.setContentsMargins(0, 0, 0, 0)
        field_layout.addWidget(edit)
        field_layout.addWidget(delete_button)

        self.character_styles_form_layout.addRow(label, field_container)
        self.style_widget_map[char_id] = (label, field_container)
        delete_button.clicked.connect(lambda: self.on_delete_character_style(char_id))

    def on_add_character_style(self):
        """(v19 추가) "추가" 버튼 핸들러. CHARACTER_DIR에서 캐릭터 파일을 선택."""
        start_dir = r"C:\my_games\shorts_make\character"
        try:

            start_dir = getattr(settings_mod, "CHARACTER_DIR", start_dir)
        except (ImportError, AttributeError):
            pass

        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "캐릭터 추가 (이미지 파일 선택)", start_dir, "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if not file_path: return

        char_id = Path(file_path).stem
        if char_id in self.style_widget_map:
            QtWidgets.QMessageBox.warning(self, "오류", f"캐릭터 ID '{char_id}'가 이미 목록에 있습니다.")
            return

        self._add_character_style_row(char_id, f"{char_id}의 기본 스타일 (설명 입력)")

    def on_delete_character_style(self, char_id: str):
        """(v20 수정) "삭제" 버튼 핸들러 (deleteLater 제거로 충돌 해결)"""
        if char_id not in self.style_widget_map:
            return

        reply = QtWidgets.QMessageBox.question(
            self, "캐릭터 삭제 확인",
            f"'{char_id}' 캐릭터를 스타일 목록에서 삭제하시겠습니까?\n\n"
            f"(참고: '업데이트' 버튼을 눌러야 video.json의\n"
            f"루트 'characters' 목록에서도 최종 제거됩니다.)",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        label_widget, field_container_widget = self.style_widget_map[char_id]

        # 1. removeRow가 위젯을 자동 삭제
        self.character_styles_form_layout.removeRow(label_widget)

        # 2. (제거됨) deleteLater()

        # 3. 맵에서 제거
        del self.style_widget_map[char_id]

    def _build_character_styles_group(self) -> QtWidgets.QGroupBox:
        """(v29 수정) 캐릭터 스타일 그룹 (높이 100px 제한, 여백 0)"""
        group = QtWidgets.QGroupBox("전체 등장 캐릭터 스타일")
        main_v_layout = QtWidgets.QVBoxLayout(group)
        main_v_layout.setContentsMargins(2, 0, 2, 0)
        main_v_layout.setSpacing(0)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(100)  # [v29] 100px
        scroll_area.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                  QtWidgets.QSizePolicy.Policy.Preferred)  # [v27]

        scroll_content_widget = QtWidgets.QWidget()
        scroll_content_widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                            QtWidgets.QSizePolicy.Policy.Preferred)  # [v29]

        self.character_styles_form_layout = QtWidgets.QFormLayout(scroll_content_widget)
        self.character_styles_form_layout.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapAllRows)
        self.character_styles_form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.character_styles_form_layout.setContentsMargins(0, 0, 0, 0)
        self.character_styles_form_layout.setVerticalSpacing(0)  # [v26]
        self.character_styles_form_layout.setHorizontalSpacing(4)

        scroll_area.setWidget(scroll_content_widget)
        main_v_layout.addWidget(scroll_area)  # 스크롤 영역을 먼저 추가

        # --- ▼▼▼ [신규] "캐릭터 추가" 버튼 레이아웃 ▼▼▼ ---

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setContentsMargins(0, 4, 0, 0)  # 상단 여백
        add_button = QtWidgets.QPushButton("캐릭터 추가")
        add_button.setToolTip(
            f"폴더[{getattr(settings_mod, 'CHARACTER_DIR', 'C:/.../character')}]에서 캐릭터 파일을 선택해 목록에 추가합니다."
        )
        add_button.clicked.connect(self.on_add_character_style)
        button_layout.addStretch(1)  # 버튼을 오른쪽으로 밀기
        button_layout.addWidget(add_button)

        main_v_layout.addLayout(button_layout)  # 스크롤 영역 *아래*에 버튼 레이아웃 추가
        # --- ▲▲▲ [신규] 코드 끝 ▲▲▲ ---

        self.style_widget_map: Dict[str, Tuple[QtWidgets.QLabel, QtWidgets.QWidget]] = {}
        styles_data = self.full_video_data.get("character_styles", {})

        if not styles_data:
            self.character_styles_form_layout.addRow(QtWidgets.QLabel("프로젝트에 character_styles 정보가 없습니다."))
        else:
            for char_id, style_text in styles_data.items():
                self._add_character_style_row(char_id, style_text)
        return group

    def old_on_ai_request(self):
        """(v17 수정) 'AI 요청' 버튼 핸들러 (기능 동일)"""
        scenes_to_process: List[Tuple[Dict[str, Any], str]] = []
        scenes_map = {scene.get("id"): scene for scene in self.scenes_data if isinstance(scene, dict) and "id" in scene}

        for scene_id, text_edit_widget in self.widget_map:
            if scene_id in scenes_map:
                direct_prompt_text = text_edit_widget.toPlainText().strip()
                if direct_prompt_text:
                    scenes_to_process.append((scenes_map[scene_id], direct_prompt_text))

        if not scenes_to_process:
            QtWidgets.QMessageBox.information(self, "알림",
                                              "AI로 요청할 'direct_prompt' 내용이 없습니다.\n먼저 'direct_prompt' 텍스트창에 내용을 입력하세요.")
            return

        self.btn_ai_request.setEnabled(False)
        self.btn_update.setEnabled(False)
        self.btn_cancel.setEnabled(False)

        def job(progress_callback: Callable[[dict], None]):
            _log = lambda msg: progress_callback({"msg": msg})
            _log(f"총 {len(scenes_to_process)}개의 씬에 대해 3개 필드(prompt, prompt_img, prompt_movie) 생성을 요청합니다...")

            system_prompt = (
                "You are an AI assistant for video prompt generation. "
                "The user will provide core keywords. "
                "Return a JSON object with 3 keys: "
                "1. `prompt_ko`: A single, vivid scene description in Korean based on the keywords. "
                "2. `prompt_img_base`: A comma-separated list of English keywords (5-10 tags) for the scene's background, mood, and subject. "
                "3. `motion_hint`: A single, short English motion tag (e.g., 'slow zoom in', 'camera pan left', 'subtle eye blink'). If no motion, return an empty string \"\". "
                "Respond ONLY with the JSON object."
            )

            updated_count = 0
            quality_tags = self._AI_QUALITY_TAGS
            default_negative_tags = self._AI_DEFAULT_NEGATIVE_TAGS

            for scene_dict, dp_text in scenes_to_process:
                current_scene_id = scene_dict.get("id", "Unknown")
                _log(f"[{current_scene_id}] 요청 중... (Keywords: {dp_text[:30]}...)")
                user_prompt = f"Keywords: {dp_text}\n\nJSON Response:"

                try:
                    ai_response_str = self.ai_instance.ask_smart(system_prompt, user_prompt, prefer="gemini",
                                                                 allow_fallback=True)
                    json_start = ai_response_str.find("{")
                    json_end = ai_response_str.rfind("}") + 1

                    if 0 <= json_start < json_end:
                        json_str = ai_response_str[json_start:json_end]
                        try:
                            ai_json_data = json.loads(json_str)
                            if not isinstance(ai_json_data, dict): raise json.JSONDecodeError("Not a dict", json_str, 0)
                        except json.JSONDecodeError as e_json:
                            _log(f"[{current_scene_id}] AI 응답 JSON 파싱 실패: {e_json}\n응답: {ai_response_str[:50]}...")
                            continue
                    else:
                        _log(f"[{current_scene_id}] AI가 JSON 응답을 반환하지 않았습니다.")
                        continue

                    prompt_ko = (ai_json_data.get("prompt_ko") or "").strip().replace('"', '').replace("'", "")
                    prompt_img_base = (ai_json_data.get("prompt_img_base") or "").strip()
                    motion_hint = (ai_json_data.get("motion_hint") or "").strip()

                    if prompt_ko and prompt_img_base:
                        scene_dict["prompt"] = prompt_ko
                        scene_dict["prompt_img"] = f"{prompt_img_base}, {quality_tags}"
                        scene_dict[
                            "prompt_movie"] = f"{prompt_img_base}, {quality_tags}, motion: {motion_hint}" if motion_hint else \
                        scene_dict["prompt_img"]
                        scene_dict["prompt_negative"] = default_negative_tags
                        updated_count += 1
                        _log(f"[{current_scene_id}] 3개 필드 완료: {prompt_ko[:30]}...")
                    else:
                        _log(f"[{current_scene_id}] AI가 필수 필드(prompt_ko, prompt_img_base)를 반환하지 않았습니다.")

                except Exception as e_ai_call:
                    _log(f"[{current_scene_id}] AI 요청 실패 ({type(e_ai_call).__name__}): {e_ai_call}")

            if updated_count > 0:
                _log(f"총 {updated_count}개 씬의 프롬프트를 갱신했습니다. video.json 파일에 저장합니다...")
                self.full_video_data["scenes"] = self.scenes_data
                try:
                    save_json(self.json_path, self.full_video_data)
                except (IOError, OSError, Exception) as save_e:
                    _log(f"[ERROR] video.json 저장 실패: {save_e}")

            return {"updated_count": updated_count}

        def done(ok: bool, payload: Optional[dict], err: Optional[Exception]):
            self.btn_ai_request.setEnabled(True)
            self.btn_update.setEnabled(True)
            self.btn_cancel.setEnabled(True)

            if not ok:
                QtWidgets.QMessageBox.critical(self, "AI 요청 실패", f"작업 중 오류가 발생했습니다:\n{err}")
                return
            if payload:
                count = payload.get("updated_count", 0)
                if count > 0:
                    QtWidgets.QMessageBox.information(self, "AI 요청 완료",
                                                      f"총 {count}개 씬의 'prompt', 'prompt_img', 'prompt_movie' 필드를 갱신하고 저장했습니다.\n\n"
                                                      f"(이제 '업데이트' 버튼을 눌러 direct_prompt도 저장하거나, 창을 닫으세요.)")
                else:
                    QtWidgets.QMessageBox.warning(self, "AI 요청", "AI가 갱신한 내용이 없거나 저장에 실패했습니다.")

        run_job_with_progress_async(owner=self, title=f"AI 프롬프트 생성 중 ({self.json_path.name})", job=job, on_done=done)

    # shorts_ui.py의 ScenePromptEditDialog 클래스 내부
    # on_ai_request 함수를 이 코드로 덮어쓰세요.

    def on_ai_request(self):
        """
        제이슨수정 창에서 'AI 요청' 버튼 핸들러
        [수정됨] 6가지 문맥(원본분위기, 현재가사, direct_prompt, 캐릭터, 시간, 다음가사)을 AI에 전달
        """


        # --- [필요한 유틸리티 임포트] ---
        # --- [임포트 끝] ---

        # --- [FPS 동기화 로직 (기존과 동일)] ---
        try:
            main_window = self.parent()
            if main_window and hasattr(main_window, "cmb_movie_fps"):
                cmb_fps = getattr(main_window, "cmb_movie_fps")
                ui_fps = int(cmb_fps.currentData())

                self.full_video_data.setdefault("defaults", {})
                self.full_video_data["defaults"].setdefault("movie", {})
                self.full_video_data["defaults"]["movie"]["target_fps"] = ui_fps
                self.full_video_data["defaults"]["movie"]["input_fps"] = ui_fps
                self.full_video_data["defaults"]["movie"]["fps"] = ui_fps
                self.full_video_data["defaults"].setdefault("image", {})["fps"] = ui_fps
                self.full_video_data["fps"] = int(ui_fps)

                print(f"[JSON Edit] AI 요청 시 UI FPS ({ui_fps})를 video.json 데이터에 동기화했습니다.")

        except Exception as e_fps_sync:
            print(f"[JSON Edit] AI 요청 중 FPS 동기화 실패: {e_fps_sync}")
        # --- [FPS 동기화 끝] ---

        # --- ▼▼▼ [신규] 1. 원본 분위기 (project.json) 로드 ▼▼▼ ---
        original_vibe_prompt = ""
        try:
            pj_path = self.json_path.parent / "project.json"
            if pj_path.exists():
                pj_doc = load_json(pj_path, {}) or {}
                if isinstance(pj_doc, dict):
                    original_vibe_prompt = pj_doc.get("prompt_user") or pj_doc.get("prompt", "")
        except Exception as e_load_pj:
            print(f"[JSON Edit] project.json 로드 실패: {e_load_pj}")
        # --- ▲▲▲ [신규] 끝 ▲▲▲ ---

        # 2) 화면에 있는 direct_prompt 들고오기 (기존과 동일)
        scenes_to_process: list[tuple[dict, str]] = []
        scenes_map = {scene.get("id"): scene for scene in self.scenes_data if
                      isinstance(scene, dict) and "id" in scene}
        for scene_id, text_edit_widget in self.widget_map:
            if scene_id in scenes_map:
                direct_prompt_text = text_edit_widget.toPlainText().strip()
                if direct_prompt_text:
                    scenes_to_process.append((scenes_map[scene_id], direct_prompt_text))

        if not scenes_to_process:
            QtWidgets.QMessageBox.information(self, "알림",
                                              "AI로 요청할 'direct_prompt' 내용이 없습니다.\n(UI의 FPS 설정값은 video.json에 저장됩니다.)", )
            # [FPS 저장 로직 (기존과 동일)]
            try:
                save_json(self.json_path, self.full_video_data)
                print(f"[JSON Edit] 프롬프트는 비어있으나, FPS 값({self.full_video_data.get('fps')})을 저장했습니다.")
            except Exception as e_save_fps_only:
                print(f"[JSON Edit] FPS만 저장하는 데 실패: {e_save_fps_only}")
            return

            # 버튼 잠그기 (기존과 동일)
        self.btn_ai_request.setEnabled(False)
        self.btn_update.setEnabled(False)
        self.btn_cancel.setEnabled(False)

        def job(progress_callback):
            _log = lambda msg: progress_callback({"msg": msg})
            _log(f"총 {len(scenes_to_process)}개 씬에 대해 프롬프트를 AI로 갱신합니다...")

            quality_tags = self._AI_QUALITY_TAGS
            default_negative_tags = self._AI_DEFAULT_NEGATIVE_TAGS
            updated_count = 0

            # --- ▼▼▼ [수정된 AI 시스템 프롬프트 (뮤직비디오 감독)] ▼▼▼ ---
            base_system_prompt = (
                "You are a creative Music Video Director.\n"
                "Your most important goal is to create **dynamic character action** that matches the **lyrics**. Avoid static, mannequin-like images.\n\n"
                "[Context Provided]\n"
                "1. `original_vibe`: The overall theme of the entire song.\n"
                "2. `scene_lyric`: The lyric for THIS scene (THIS IS THE MOST IMPORTANT).\n"
                "3. `base_visual` (direct_prompt): The user's core visual idea for THIS scene (use this for the SETTING only, but you can change it creatively).\n"
                "4. `characters`: The characters in THIS scene (e.g., 'female_01').\n"
                "5. `time_structure`: The frame segments for THIS scene (e.g., [\"0-65f\", \"49-125f\"]).\n"
                "6. `next_scene_lyric`: The lyric for the *next* scene (for transition context).\n\n"
                "[Your Task (Return JSON ONLY)]\n"
                "1. \"prompt_ko\": Korean description of the whole scene (based on all context).\n"
                "2. \"prompt_img_base\": English, comma-separated visual tags for the whole scene (5-12 words).\n"
                "3. \"motion_hint\": short English motion/camera hint (e.g. \"slow zoom in\"). Can be \"\".\n"
                "4. \"segment_prompts\": an array of English **scene descriptions**.\n"
                "   The array length MUST exactly match the `time_structure` list length.\n\n"
                "[!! CRITICAL RULES !!]\n"
                "1.  **New Action (Most Important):** Based on the `scene_lyric` and `original_vibe`, you MUST describe a specific, creative **new pose and action** for the `characters` in *each* `segment_prompts` description.\n"
                "    (Examples: \"female_01 starts walking down the autumn path\", \"female_01 stops and picks up an autumn leaf, smiling\", \"female_01 spins around, throwing leaves in the air\").\n"
                "2.  **Background:** Use `base_visual` as the background, but change it creatively (e.g., \"The red autumn background becomes darker\").\n"
                "3.  **Progression:** Design the actions to be continuous and logical, telling a small story that matches the lyric's emotion.\n"
                "4.  **Camera:** Include dynamic camera work in each description (e.g., \"close up on her face\", \"camera rotates around her\", \"from a top-down view\").\n"
                "5.  **Emotion:** Describe the character's expression (e.g., \"smiling happily\", \"peaceful expression\").\n"
                "6.  **Prohibition:** NO \"mannequins\". Every prompt must describe a **change in the character's action or pose**."
            )
            # --- ▲▲▲ [수정 끝] ▲▲▲ ---

            for scene_dict, dp_text in scenes_to_process:
                # 'scene_id'를 'current_scene_id'로 변경하여 외부 변수와의 충돌을 피합니다.
                current_scene_id = scene_dict.get("id", "scene")

                # --- [6가지 문맥 수집 (기존과 동일)] ---
                frame_segments = scene_dict.get("frame_segments") or []
                seg_count = len(frame_segments)
                frame_ranges_info = [f"{s.get('start_frame')}-{s.get('end_frame')}f" for s in frame_segments]

                scene_lyric = scene_dict.get("lyric", "")
                characters = scene_dict.get("characters", [])

                next_scene_lyric = "(Scene End)"
                current_index = -1
                for idx, s in enumerate(self.scenes_data):  # self.scenes_data는 전체 씬 목록
                    # 변경된 'current_scene_id'를 사용합니다.
                    if isinstance(s, dict) and s.get("id") == current_scene_id:
                        current_index = idx
                        break

                if current_index != -1 and current_index + 1 < len(self.scenes_data):
                    next_sc = self.scenes_data[current_index + 1]
                    if isinstance(next_sc, dict):
                        next_scene_lyric = next_sc.get("lyric", "") or "(Next scene has no lyric)"
                # --- [수집 끝] ---

                # 변경된 'current_scene_id'를 사용합니다.
                _log(f"[{current_scene_id}] AI 요청 중... (segments={seg_count})")

                # --- [user 프롬프트 6가지 문맥 전달 (기존과 동일)] ---
                user_prompt_payload = {
                    "original_vibe": original_vibe_prompt,
                    "scene_lyric": scene_lyric,
                    "base_visual": dp_text,  # direct_prompt를 base_visual로 전달
                    "characters": characters,
                    "time_structure": frame_ranges_info,
                    "next_scene_lyric": next_scene_lyric
                }
                user_prompt = json.dumps(user_prompt_payload, ensure_ascii=False)

                try:
                    ai_raw = self.ai_instance.ask_smart(
                        base_system_prompt,
                        user_prompt,
                        prefer="gemini",
                        allow_fallback=True,
                    )
                except Exception as e_ai:
                    # 변경된 'current_scene_id'를 사용합니다.
                    _log(f"[{current_scene_id}] AI 호출 실패: {e_ai}")
                    continue

                # (AI 응답 파싱)
                json_start = ai_raw.find("{")
                json_end = ai_raw.rfind("}") + 1
                if not (0 <= json_start < json_end):
                    # 변경된 'current_scene_id'를 사용합니다.
                    _log(f"[{current_scene_id}] AI가 JSON을 반환하지 않았습니다.")
                    continue
                try:
                    ai_json = json.loads(ai_raw[json_start:json_end])
                except Exception as e_json:
                    # 변경된 'current_scene_id'를 사용합니다.
                    _log(f"[{current_scene_id}] JSON 파싱 실패: {e_json}")
                    continue

                prompt_ko = (ai_json.get("prompt_ko") or "").strip()
                prompt_img_base = (ai_json.get("prompt_img_base") or "").strip()
                motion_hint = (ai_json.get("motion_hint") or "").strip()

                # --- ▼▼▼ [수정] 'segment_prompts' 키로 파싱 ▼▼▼ ---
                seg_prompts = ai_json.get("segment_prompts", [])  # 'segment_instructions' -> 'segment_prompts'

                if prompt_ko and prompt_img_base:
                    scene_dict["prompt"] = prompt_ko
                    scene_dict["prompt_img"] = f"{prompt_img_base}, {quality_tags}"
                    if motion_hint:
                        scene_dict["prompt_movie"] = f"{prompt_img_base}, {quality_tags}, motion: {motion_hint}"
                    else:
                        scene_dict["prompt_movie"] = scene_dict["prompt_img"]
                    scene_dict["prompt_negative"] = default_negative_tags
                    updated_count += 1
                    # 변경된 'current_scene_id'를 사용합니다.
                    _log(f"[{current_scene_id}] 기본 프롬프트 갱신 완료")

                if seg_count > 0:
                    filled = 0
                    if isinstance(seg_prompts, list) and len(seg_prompts) >= seg_count:
                        for i in range(seg_count):
                            seg_item = frame_segments[i]
                            prompt_text = seg_prompts[i]  # 행동 묘사

                            if isinstance(prompt_text, dict):
                                prompt_text = (prompt_text.get("prompt_movie") or prompt_text.get("text") or "")

                            prompt_text = str(prompt_text).strip()
                            if prompt_text:
                                # 'prompt_movie' 필드에 "행동 묘사"를 저장
                                seg_item["prompt_movie"] = prompt_text
                                filled += 1
                        # 변경된 'current_scene_id'를 사용합니다.
                        _log(f"[{current_scene_id}] 세그먼트 묘사 {filled}/{seg_count}개 AI로 채움")

                    elif filled == 0:
                        # (폴백: AI가 묘사를 안 줬다면, 차라리 direct_prompt를 그대로 쓴다)
                        base_cmd = dp_text
                        for seg_item in frame_segments:
                            seg_item["prompt_movie"] = base_cmd
                        # 변경된 'current_scene_id'를 사용합니다.
                        _log(f"[{current_scene_id}] AI 세그먼트 묘사 응답 없음 → Direct Prompt로 일괄 채움")

                    scene_dict["frame_segments"] = frame_segments
                # --- ▲▲▲ [수정 끝] ▲▲▲ ---

            # 저장 (기존과 동일)
            if updated_count > 0:
                _log("변경 내용을 video.json 에 저장합니다...")
                self.full_video_data["scenes"] = self.scenes_data
                try:
                    save_json(self.json_path, self.full_video_data)
                except Exception as e_save:
                    _log(f"video.json 저장 실패: {e_save}")

            return {"updated_count": updated_count}

        def done(ok, payload, err):
            # (done 콜백 함수는 기존과 동일)
            self.btn_ai_request.setEnabled(True)
            self.btn_update.setEnabled(True)
            self.btn_cancel.setEnabled(True)
            if not ok:
                QtWidgets.QMessageBox.critical(self, "AI 요청 실패", f"작업 중 오류가 발생했습니다:\n{err}")
                return
            count = (payload or {}).get("updated_count", 0)
            if count > 0:
                QtWidgets.QMessageBox.information(
                    self,
                    "AI 요청 완료",
                    f"총 {count}개 씬의 프롬프트를 갱신했습니다.\n세그먼트 프롬프트(행동 묘사)도 함께 저장되었습니다.",
                )
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    "AI 요청",
                    "AI가 갱신한 내용이 없거나 저장에 실패했습니다.",
                )

        run_job_with_progress_async(
            owner=self,
            title=f"AI 프롬프트 생성 중 ({self.json_path.name})",
            job=job,
            on_done=done,
        )

    def on_upload_image(self, scene_id: str, preview_label: QtWidgets.QLabel, upload_button: QtWidgets.QPushButton,
                        delete_image_button: QtWidgets.QPushButton, scene_data: Dict[str, Any]):
        """(v57 최종) 파일 리사이즈 로직 완전 삭제 / 썸네일 에러 제거"""


        imgs_dir = self.json_path.parent / "imgs"
        try:
            imgs_dir.mkdir(parents=True, exist_ok=True)
        except:
            pass

        target_path = imgs_dir / f"{scene_id}.png"

        src_path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"'{scene_id}' 씬 이미지 선택", str(imgs_dir), "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if not src_path_str: return

        # 메인 UI에서 W/H 값만 읽어서 워크플로우에 전달 (파이썬 처리 X)
        main_win = self.parent()
        while main_win and not hasattr(main_win, "cmb_img_w"):
            main_win = main_win.parent()

        target_w, target_h = 720, 1280
        if main_win:
            try:
                target_w = int(main_win.cmb_img_w.currentData())
                target_h = int(main_win.cmb_img_h.currentData())
            except:
                pass

        def job(progress_callback):
            _log = lambda msg: progress_callback({"msg": msg})

            wf_path = Path(JSONS_DIR) / "only_faceswap.json"
            if not wf_path.exists(): raise FileNotFoundError("워크플로우 파일 없음")

            workflow = load_json(wf_path)

            # A. 원본 파일 그대로 업로드 (리사이즈 안 함)
            comfy_in = Path(COMFY_INPUT_DIR)
            src_path = Path(src_path_str)
            temp_input_name = f"raw_{scene_id}_{src_path.name}"
            shutil.copy2(str(src_path), comfy_in / temp_input_name)

            # B. 노드 설정
            if "2" in workflow: workflow["2"]["inputs"]["image"] = temp_input_name
            # Node 11 (Resize): 워크플로우가 알아서 함
            if "11" in workflow:
                workflow["11"]["inputs"]["width"] = target_w
                workflow["11"]["inputs"]["height"] = target_h

            # C. 캐릭터 매핑
            chars = scene_data.get("characters", [])
            char_map = {}
            for c in chars:
                cid, cidx = "", 0
                if isinstance(c, dict):
                    cid, cidx = c.get("id", ""), int(c.get("index", 0) or 0)
                elif isinstance(c, str):
                    if ":" in c:
                        p = c.split(":")
                        cid, cidx = p[0].strip(), int(p[1].strip())
                    else:
                        cid, cidx = c.strip(), 0
                if cid: char_map[cidx] = cid

            reactor_setup = {0: {"r": "8", "l": "3"}, 1: {"r": "7", "l": "10"}, 2: {"r": "6", "l": "9"}}
            char_base = Path(CHARACTER_DIR)
            for idx, nodes in reactor_setup.items():
                rid, lid = nodes["r"], nodes["l"]
                if idx in char_map:
                    char_id = char_map[idx]
                    c_img_path = None
                    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                        p = char_base / f"{char_id}{ext}"
                        if p.exists(): c_img_path = p; break
                    if c_img_path:
                        c_dst_name = f"char_{char_id}{c_img_path.suffix}"
                        shutil.copy2(str(c_img_path), comfy_in / c_dst_name)
                        if rid in workflow: workflow[rid]["inputs"]["enabled"] = True
                        if lid in workflow: workflow[lid]["inputs"]["image"] = c_dst_name
                    else:
                        if rid in workflow: workflow[rid]["inputs"]["enabled"] = False
                else:
                    if rid in workflow: workflow[rid]["inputs"]["enabled"] = False

            # D. 실행
            if "5" in workflow: workflow["5"]["inputs"]["filename_prefix"] = f"upload_proc/{scene_id}"

            _log("ComfyUI 실행 (Resize Node 11)...")
            result = _submit_and_wait_comfy(COMFY_HOST, workflow, timeout=60, poll=0.5)

            outputs = result.get("outputs", {}).get("5", {}).get("images", [])
            if not outputs: raise RuntimeError("결과 없음")

            info = outputs[0]
            resp = requests.get(f"{COMFY_HOST}/view", params={
                "filename": info.get("filename"), "subfolder": info.get("subfolder"), "type": "output"
            })
            resp.raise_for_status()

            # 결과 저장 (워크플로우가 처리한 파일)
            with open(target_path, "wb") as f:
                f.write(resp.content)
            return str(target_path)

        def done(ok, payload, err):
            if not ok:
                QtWidgets.QMessageBox.critical(self, "오류", f"실패:\n{err}")
                return

            final_path = str(payload)
            scene_data["img_file"] = final_path

            # [수정] 썸네일 표시용: 복잡한 옵션 다 빼고 기본값 사용 (에러 원천 차단)
            pixmap = QtGui.QPixmap(final_path)
            if not pixmap.isNull():
                # 옵션 없이 크기만 지정 -> 기본값(FastTransformation, IgnoreAspectRatio)으로 동작하여 에러 안 남
                pixmap_scaled = pixmap.scaled(self.THUMBNAIL_SIZE, self.THUMBNAIL_SIZE)

                preview_label.setPixmap(pixmap_scaled)
                preview_label.setText("")
                preview_label.setToolTip(f"경로: {final_path}\n(자동 처리됨)")

                upload_button.setText("이미지 변경")
                delete_image_button.setEnabled(True)

                try:
                    preview_label.clicked.disconnect()
                except:
                    pass
                preview_label.clicked.connect(lambda: self.show_large_image(final_path))

                try:
                    delete_image_button.clicked.disconnect()
                except:
                    pass
                delete_image_button.clicked.connect(
                    functools.partial(self.on_delete_image, Path(final_path), preview_label, upload_button,
                                      delete_image_button, scene_data)
                )

            QtWidgets.QMessageBox.information(self, "완료", "처리 완료")

        run_job_with_progress_async(self, f"이미지 처리 ({scene_id})", job, on_done=done)

    def on_delete_video(self, video_path: Path, button_widget: QtWidgets.QPushButton):
        """(v17 수정) '영상삭제' 버튼 핸들러 (기능 동일)"""
        if not video_path.exists():
            QtWidgets.QMessageBox.warning(self, "오류", "파일이 이미 존재하지 않습니다.")
            button_widget.setEnabled(False)
            return

        reply = QtWidgets.QMessageBox.question(
            self, "영상 삭제 확인", f"정말로 이 씬의 비디오 파일을 삭제하시겠습니까?\n\n{video_path.name}",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No
        )
        if reply != QtWidgets.QMessageBox.Yes: return

        try:
            video_path.unlink()
            QtWidgets.QMessageBox.information(self, "삭제 완료", f"파일을 삭제했습니다:\n{video_path.name}")
            button_widget.setEnabled(False)
        except FileNotFoundError:
            QtWidgets.QMessageBox.warning(self, "오류", "파일이 이미 존재하지 않았습니다.")
            button_widget.setEnabled(False)
        except (OSError, Exception) as e_delete:
            QtWidgets.QMessageBox.critical(self, "삭제 실패", f"파일 삭제 중 오류가 발생했습니다:\n{e_delete}")

    def on_delete_image(self, image_path: Path, preview_label: QtWidgets.QLabel, upload_button: QtWidgets.QPushButton,
                        delete_image_button: QtWidgets.QPushButton, scene_data: Dict[str, Any]):
        """(v17 수정) '이미지 삭제' 버튼 핸들러 (기능 동일)"""
        if not image_path.exists():
            QtWidgets.QMessageBox.warning(self, "오류", "파일이 이미 존재하지 않습니다.")

        reply = QtWidgets.QMessageBox.question(
            self, "이미지 삭제 확인", f"정말로 이 씬의 이미지 파일을 삭제하시겠습니까?\n\n{image_path.name}",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No
        )
        if reply != QtWidgets.QMessageBox.Yes: return

        try:
            if image_path.exists(): image_path.unlink()
            QtWidgets.QMessageBox.information(self, "삭제 완료", f"파일을 삭제했습니다:\n{image_path.name}")
            scene_data["img_file"] = ""
            preview_label.setPixmap(QtGui.QPixmap())
            preview_label.setText("[이미지\n없음]")
            preview_label.setToolTip(f"파일이 삭제되었습니다. (경로: {image_path})")
            preview_label.setStyleSheet("border: 1px dashed gray; color: gray;")
            try:
                preview_label.clicked.disconnect()
            except (TypeError, RuntimeError):
                pass
            upload_button.setText("업로드")
            delete_image_button.setEnabled(False)
        except FileNotFoundError:
            QtWidgets.QMessageBox.warning(self, "오류", "파일이 이미 존재하지 않았습니다.")
            delete_image_button.setEnabled(False)
        except (OSError, Exception) as e_delete:
            QtWidgets.QMessageBox.critical(self, "삭제 실패", f"파일 삭제 중 오류가 발생했습니다:\n{e_delete}")

    # shorts_ui.py의 ScenePromptEditDialog 클래스 내부
    # load_and_build_ui 함수를 이 코드로 덮어쓰세요.

    def load_and_build_ui(self):
        """
        [수정됨 v42] (사장님 요청 반영)
        - "세그먼트 수정" 버튼 추가 및 `SegmentEditDialog` 열기 기능 연결.
        """
        try:
            # --- (v37) settings 임포트 (버그 수정) ---

            # --- (v37) 수정 끝 ---

            data = load_json(self.json_path, None)
            if not isinstance(data, dict):
                raise ValueError("video.json 파일의 형식이 올바르지 않습니다.")

            self.full_video_data = data
            self.scenes_data = self.full_video_data.get("scenes", [])
            if not isinstance(self.scenes_data, list):
                raise ValueError("video.json에 'scenes' 키가 없거나 리스트가 아닙니다.")

            # --- [v29] 캐릭터 스타일 그룹 + 추가 버튼 가로 배치 ---
            # (기존 캐릭터 스타일 그룹 로직은 동일)
            top_char_layout = QtWidgets.QHBoxLayout()
            self.character_styles_group = self._build_character_styles_group()
            top_char_layout.addWidget(self.character_styles_group, 1)
            # (캐릭터 추가 버튼 로직은 _build_character_styles_group 안으로 이동했음)

            main_layout = self.layout()
            if main_layout and hasattr(main_layout, "insertLayout"):
                # [수정] 0번 인덱스(맨 위)에 캐릭터 스타일 그룹 삽입
                main_layout.insertLayout(0, top_char_layout)
                # --- [v29] 배치 끝 ---

            font = QtGui.QFont()
            font.setFamily("Courier" if "Courier" in QtGui.QFontDatabase().families() else "Monospace")
            font.setPointSize(10)

            scene_chunks = [self.scenes_data[i:i + self.PAGE_SIZE] for i in
                            range(0, len(self.scenes_data), self.PAGE_SIZE)]
            self.total_pages = len(scene_chunks) or 1
            if not scene_chunks: scene_chunks = [[]]

            self.scene_char_widgets.clear()

            for chunk in scene_chunks:
                page_scroll_area = QtWidgets.QScrollArea()
                page_scroll_area.setWidgetResizable(True)
                scroll_content_widget = QtWidgets.QWidget()
                form_layout_page = QtWidgets.QFormLayout(scroll_content_widget)
                form_layout_page.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapAllRows)
                form_layout_page.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
                form_layout_page.setVerticalSpacing(8)
                page_scroll_area.setWidget(scroll_content_widget)

                if not chunk:
                    form_layout_page.addRow(QtWidgets.QLabel("이 페이지에 씬이 없습니다."))

                for scene in chunk:
                    if not isinstance(scene, dict): continue

                    # (씬 라벨 정보 ... 기존과 동일)
                    scene_id = scene.get("id", "ID-없음")
                    lyric = (scene.get("lyric") or "").strip()
                    start_f = scene.get("start", 0.0)
                    end_f = scene.get("end", 0.0)
                    duration_f = scene.get("duration", 0.0)
                    if duration_f == 0.0 and end_f > start_f: duration_f = end_f - start_f
                    chars_list_label = scene.get("characters", [])
                    chars_str = ", ".join(chars_list_label) if chars_list_label else "없음"
                    label_text = (
                        f"<b>{scene_id}</b> [{lyric or '가사 없음'}] | "
                        f"<b>캐릭터:</b> [{chars_str}] | "
                        f"<b>시간:</b> [{start_f:.2f} ~ {end_f:.2f}, ({duration_f:.2f}s)]"
                    )
                    label = QtWidgets.QLabel(label_text)
                    label.setWordWrap(False)

                    # --- [v41 수정] 가로 3단 레이아웃 ---
                    row_container = QtWidgets.QWidget()
                    row_layout = QtWidgets.QHBoxLayout(row_container)
                    row_layout.setContentsMargins(0, 0, 0, 0)
                    row_layout.setSpacing(8)

                    # --- 1. (왼쪽) 이미지 + 버튼 ---
                    left_vbox_widget = QtWidgets.QWidget()
                    left_vbox = QtWidgets.QVBoxLayout(left_vbox_widget)
                    left_vbox.setContentsMargins(0, 0, 0, 0)
                    left_vbox.setSpacing(4)

                    img_preview_label = ClickableLabel()
                    img_preview_label.setFixedSize(self.THUMBNAIL_SIZE, self.THUMBNAIL_SIZE)
                    img_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

                    upload_button = QtWidgets.QPushButton("업로드")
                    upload_button.setFixedSize(self.THUMBNAIL_SIZE, 28)
                    delete_image_button = QtWidgets.QPushButton("이미지 삭제")
                    delete_image_button.setFixedSize(self.THUMBNAIL_SIZE, 28)
                    delete_video_button = QtWidgets.QPushButton("영상삭제")
                    delete_video_button.setFixedSize(self.THUMBNAIL_SIZE, 28)

                    left_vbox.addWidget(img_preview_label)
                    left_vbox.addWidget(upload_button)
                    left_vbox.addWidget(delete_image_button)
                    left_vbox.addWidget(delete_video_button)
                    left_vbox.addStretch(1)
                    row_layout.addWidget(left_vbox_widget, 0)

                    # --- 2. (중간) Direct Prompt 텍스트 편집기 ---
                    text_edit = QtWidgets.QTextEdit()
                    text_edit.setPlainText(scene.get("direct_prompt", ""))
                    text_edit.setFont(font)
                    text_edit.setMinimumHeight(150)
                    text_edit.setToolTip(f"Scene ID: {scene_id}\n이 씬의 direct_prompt를 입력하세요.")
                    text_edit.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                            QtWidgets.QSizePolicy.Policy.Expanding)
                    row_layout.addWidget(text_edit, 1)

                    # --- 3. (오른쪽) 씬별 캐릭터 관리 UI ---
                    right_vbox_widget = QtWidgets.QWidget()
                    right_vbox = QtWidgets.QVBoxLayout(right_vbox_widget)
                    right_vbox.setContentsMargins(0, 0, 0, 0)
                    right_vbox.setSpacing(4)

                    right_vbox.addWidget(QtWidgets.QLabel("<b>씬 캐릭터 목록 (인덱스)</b>"))
                    char_list_widget = QtWidgets.QListWidget()
                    char_list_widget.setToolTip("이 씬에만 등장하는 캐릭터와 순서(인덱스)입니다.")
                    char_list_widget.setMaximumHeight(100)

                    current_scene_chars = scene.get("characters", [])
                    if current_scene_chars:
                        char_list_widget.addItems(current_scene_chars)

                    self.scene_char_widgets[scene_id] = char_list_widget
                    right_vbox.addWidget(char_list_widget)

                    char_btn_layout = QtWidgets.QHBoxLayout()
                    btn_add_char = QtWidgets.QPushButton("추가")
                    btn_del_char = QtWidgets.QPushButton("삭제")
                    char_btn_layout.addWidget(btn_add_char)
                    char_btn_layout.addWidget(btn_del_char)
                    right_vbox.addLayout(char_btn_layout)

                    btn_edit_segments = QtWidgets.QPushButton("세그먼트 수정")
                    btn_edit_segments.setToolTip(f"[{scene_id}] 씬의 세그먼트(키프레임) 이미지와 프롬프트를 수정합니다.")
                    right_vbox.addWidget(btn_edit_segments)  # 캐릭터 목록 아래에 추가

                    # --- ▼▼▼ [신규] "립싱크 모드" 토글 버튼 (가사가 있는 경우만) ▼▼▼ ---
                    if lyric:
                        # -------------------------
                        # 1. 립싱크 토글 버튼 생성
                        # -------------------------
                        btn_lipsync = QtWidgets.QPushButton("립싱크 모드: OFF")
                        btn_lipsync.setCheckable(True)
                        btn_lipsync.setToolTip("ON: lync_bool=true (립싱크 적용)\nOFF: lync_bool=false (기본)")

                        # 초기 상태 로드
                        is_lipsync_on = scene.get("lync_bool", False)
                        btn_lipsync.setChecked(is_lipsync_on)

                        # UI 업데이트 함수
                        def update_lipsync_ui(checked, btn=btn_lipsync):
                            if checked:
                                btn.setText("립싱크 모드: ON")
                                btn.setStyleSheet("background-color: #d1e7dd; color: #0f5132; font-weight: bold;")
                            else:
                                btn.setText("립싱크 모드: OFF")
                                btn.setStyleSheet("")

                        update_lipsync_ui(is_lipsync_on)

                        # -------------------------
                        # 2. 립싱크 프롬프트 입력창 생성
                        # -------------------------
                        txt_lync_prompt = QtWidgets.QLineEdit()
                        txt_lync_prompt.setPlaceholderText("립싱크 프롬프트 (예: sing a song)")

                        # 초기 값 로드 (없으면 빈 문자열)
                        current_prompt = scene.get("lync_prompt", "")
                        txt_lync_prompt.setText(current_prompt)

                        # 텍스트 변경 시 메모리(scene dict)에 즉시 반영
                        def on_prompt_text_changed(text, s=scene):
                            s["lync_prompt"] = text.strip()

                        txt_lync_prompt.textChanged.connect(on_prompt_text_changed)

                        # -------------------------
                        # 3. 토글 핸들러 (로직 적용 및 저장)
                        # -------------------------
                        def on_lipsync_toggled(checked, s=scene, b=btn_lipsync, t=txt_lync_prompt):
                            # (1) lync_bool 상태 업데이트
                            s["lync_bool"] = checked

                            # (2) [핵심 로직] ON 할 때, 프롬프트가 비어있으면 기본값 주입
                            if checked:
                                current_text = t.text().strip()
                                if not current_text:
                                    default_val = "sing a song"
                                    t.setText(default_val)
                                    s["lync_prompt"] = default_val
                            # OFF 할 때는 텍스트를 건드리지 않음 (유지)

                            # (3) UI 스타일 갱신
                            update_lipsync_ui(checked, b)

                            # (4) 실시간 파일 저장
                            try:

                                save_json(self.json_path, self.full_video_data)
                                print(
                                    f"[JSON Edit] {s.get('id')} lync_bool={checked}, prompt='{s.get('lync_prompt')}' (저장됨)")
                            except Exception as e:
                                print(f"[JSON Edit] 저장 실패: {e}")
                                QtWidgets.QMessageBox.warning(self, "저장 실패", f"설정 저장 중 오류가 발생했습니다.\n{e}")

                        btn_lipsync.toggled.connect(on_lipsync_toggled)

                        # -------------------------
                        # 4. 레이아웃 배치
                        # -------------------------
                        right_vbox.addWidget(btn_lipsync)
                        right_vbox.addWidget(txt_lync_prompt)
                    # --- ▲▲▲ [신규] 끝 ▲▲▲ ---

                    # --- ▼▼▼ [신규] 오디오 오프셋 입력 & 음악 자르기 버튼 추가 ▼▼▼ ---
                    if lyric:  # 가사가 있는 씬일 경우에만 표시
                        layout_trim = QtWidgets.QHBoxLayout()
                        layout_trim.setContentsMargins(0, 5, 0, 0)  # 위쪽 여백 살짝 줌
                        layout_trim.setSpacing(5)

                        # 1. 라벨
                        lbl_offset = QtWidgets.QLabel("싱크(초):")
                        lbl_offset.setFixedWidth(50)

                        # 2. 오프셋 입력창 (DoubleSpinBox 사용: +/- 입력 용이)
                        spn_audio_offset = QtWidgets.QDoubleSpinBox()
                        spn_audio_offset.setRange(-5.0, 5.0)  # -5초 ~ +5초 범위
                        spn_audio_offset.setSingleStep(0.01)  # 0.01초 단위 미세 조정
                        spn_audio_offset.setDecimals(3)  # 소수점 3자리까지 표시
                        spn_audio_offset.setFixedWidth(70)  # 작게 설정
                        spn_audio_offset.setToolTip("양수(+): 시작을 늦춤 / 음수(-): 시작을 당김")

                        # 데이터 로드 및 저장 연결
                        current_offset = float(scene.get("audio_offset", 0.0))
                        spn_audio_offset.setValue(current_offset)

                        def on_offset_changed(val, s=scene):
                            s["audio_offset"] = float(val)
                            # (선택 사항) 즉시 저장하려면 아래 주석 해제
                            # save_json(self.json_path, self.full_video_data)

                        spn_audio_offset.valueChanged.connect(on_offset_changed)

                        # 3. 음악 자르기 버튼
                        btn_trim_music = QtWidgets.QPushButton("음악 자르기")
                        btn_trim_music.setFixedWidth(80)
                        btn_trim_music.setToolTip("설정된 오프셋을 적용하여 이 씬의 오디오 파일(wav)을 다시 생성합니다.")

                        # 버튼 핸들러 (기능 연결용)
                        def on_click_trim_music(s_id=scene_id, s_data=scene, off_widget=spn_audio_offset):
                            try:
                                # ... (내부 로직은 그대로 유지) ...
                                # 1. 현재 입력된 오프셋 값 가져오기
                                offset_val = off_widget.value()

                                # 2. UI 데이터(video.json)에 오프셋 값 메모리 저장
                                s_data["audio_offset"] = float(offset_val)

                                # 3. 백엔드 함수 호출 (동적 임포트)


                                # self.json_path는 video.json의 경로 (Path 객체)
                                project_root_path = self.json_path.parent

                                # Path 객체를 문자열(str)로 변환하여 전달
                                out_path = retry_cut_audio_for_scene(str(project_root_path), s_id, offset_val)

                                # 4. 성공 메시지

                                file_name = Path(out_path).name

                                print(f"[UI] 오디오 재성성 완료: {out_path} (Offset: {offset_val}s)")
                                QtWidgets.QMessageBox.information(
                                    self, "완료",
                                    f"오디오 파일을 새로 만들었습니다.\n\n"
                                    f"경로: {file_name}\n"
                                    f"적용된 싱크: {offset_val:+.3f}초"
                                )

                            except Exception as e:
                                print(f"[UI] 오디오 자르기 실패: {e}")
                                import traceback
                                traceback.print_exc()
                                QtWidgets.QMessageBox.critical(self, "실패", f"오디오 자르기 중 오류가 발생했습니다:\n{e}")

                        # ★ [수정] lambda를 사용하여 클릭 시그널(False) 무시
                        btn_trim_music.clicked.connect(lambda checked: on_click_trim_music())

                        # 레이아웃 배치
                        layout_trim.addWidget(lbl_offset)
                        layout_trim.addWidget(spn_audio_offset)
                        layout_trim.addWidget(btn_trim_music)
                        layout_trim.addStretch(1)  # 왼쪽 정렬

                        right_vbox.addLayout(layout_trim)
                    # --- ▲▲▲ [신규] 끝 ▲▲▲ ---

                    right_vbox.addStretch(1)
                    row_layout.addWidget(right_vbox_widget, 0)
                    # --- [v41] 레이아웃 수정 끝 ---

                    # (이미지 상태 설정 및 버튼 연결 코드는 기존과 동일)
                    img_file_str = scene.get("img_file", "")
                    img_path = Path(img_file_str) if img_file_str else None
                    has_image = img_path and img_path.exists()
                    # (이미지 표시 로직은 기존과 동일)
                    if has_image:
                        pixmap = QtGui.QPixmap(str(img_path))
                        if not pixmap.isNull():
                            pixmap_scaled = pixmap.scaled(self.THUMBNAIL_SIZE, self.THUMBNAIL_SIZE,
                                                          Qt.AspectRatioMode.KeepAspectRatio,
                                                          Qt.TransformationMode.SmoothTransformation)
                            img_preview_label.setPixmap(pixmap_scaled)
                            img_preview_label.setToolTip(f"경로: {img_file_str}\n(클릭해서 크게 보기)")
                            img_preview_label.clicked.connect(
                                functools.partial(self.show_large_image, img_file_str))
                            upload_button.setText("이미지 변경")
                            delete_image_button.setEnabled(True)
                        else:
                            img_preview_label.setText("[파일\n오류]")
                            img_preview_label.setStyleSheet("border: 1px solid red; color: red;")
                            upload_button.setText("다시 업로드")
                            delete_image_button.setEnabled(True)
                    else:
                        img_preview_label.setText("[이미지\n없음]")
                        img_preview_label.setToolTip(f"경로: {img_file_str}\n(파일이 존재하지 않습니다)")
                        img_preview_label.setStyleSheet("border: 1px dashed gray; color: gray;")
                        upload_button.setText("업로드")
                        delete_image_button.setEnabled(False)

                    # (기존 버튼 연결)
                    upload_button.clicked.connect(
                        functools.partial(self.on_upload_image, scene_id, img_preview_label, upload_button,
                                          delete_image_button, scene))
                    delete_image_button.clicked.connect(
                        functools.partial(self.on_delete_image, img_path if img_path else Path(), img_preview_label,
                                          upload_button, delete_image_button, scene))
                    clips_dir = self.json_path.parent / "clips"
                    video_file_path = clips_dir / f"{scene_id}.mp4"
                    video_exists = video_file_path.exists()
                    delete_video_button.setEnabled(video_exists)
                    delete_video_button.clicked.connect(
                        functools.partial(self.on_delete_video, video_file_path, delete_video_button))

                    # (캐릭터 추가/삭제 연결)
                    btn_add_char.clicked.connect(functools.partial(self.on_add_char_to_scene, char_list_widget))
                    btn_del_char.clicked.connect(functools.partial(self.on_del_char_from_scene, char_list_widget))

                    btn_edit_segments.clicked.connect(
                        functools.partial(self.on_open_segment_editor, scene_id, scene)
                    )

                    form_layout_page.addRow(label, row_container)
                    self.widget_map.append((scene_id, text_edit))

                self.stacked_widget.addWidget(page_scroll_area)

            self.current_page = 0
            self.update_page_ui()

        except Exception as e_load_ui:
            # (예외 처리 기존과 동일)
            error_label = QtWidgets.QLabel(f"파일 로드 또는 UI 빌드 중 오류 발생:\n{e_load_ui}\n\n{traceback.format_exc()}")
            error_label.setWordWrap(True)
            error_page = QtWidgets.QWidget()
            error_layout = QtWidgets.QVBoxLayout(error_page)
            error_layout.addWidget(error_label)
            self.stacked_widget.addWidget(error_page)
            self.btn_update.setEnabled(False)
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)

    def on_open_segment_editor(self, scene_id: str, scene_data: Dict[str, Any]):
            """[신규] '세그먼트 수정' 버튼 핸들러. 새 다이얼로그를 엽니다."""
            try:
                # SegmentEditDialog를 생성하고 실행합니다.
                dialog = SegmentEditDialog(
                    scene_id=scene_id,
                    scene_data=scene_data,
                    full_video_data=self.full_video_data,
                    json_path=self.json_path,
                    ai_instance=self.ai_instance,
                    parent=self
                )
                dialog.exec_()

                # (참고) SegmentEditDialog가 '업데이트' 버튼을 누를 때
                # self.full_video_data 객체를 직접 수정하고 파일에 저장합니다.
                # 이 창(ScenePromptEditDialog)은 닫힐 때 최신 데이터를 다시 로드하지 않으므로,
                # '업데이트' 버튼을 눌러야 이 창의 변경사항도 저장됩니다.

            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "세그먼트 편집기 오류", f"세그먼트 편집기를 여는 중 오류가 발생했습니다:\n{e}")

    def update_page_ui(self):
        """(v17 추가) 페이지 전환 및 버튼 상태 업데이트"""
        self.stacked_widget.setCurrentIndex(self.current_page)
        self.page_label.setText(f"페이지 {self.current_page + 1} / {self.total_pages}")
        self.btn_prev.setEnabled(self.current_page > 0)
        self.btn_next.setEnabled(self.current_page < self.total_pages - 1)

    def on_prev_page(self):
        """(v17 추가) 이전 페이지 버튼"""
        if self.current_page > 0:
            self.current_page -= 1
            self.update_page_ui()

    def on_next_page(self):
        """(v17 추가) 다음 페이지 버튼"""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.update_page_ui()

    def on_update_and_close(self):
        """
        [수정됨 v38] (사용자 요청 반영)
        1. 'direct_prompt' (씬별) 저장
        2. 'character_styles' (전체) 저장 (UI 맵 기준)
        3. 루트 'characters' 목록 갱신 (UI 맵 기준)
        4. [신규] 씬별 'characters' 목록 갱신 (씬별 QListWidget 기준)
        """
        try:
            # --- ▼▼▼ [여기에 코드 추가] ▼▼▼ ---
            # 5. [신규] UI의 현재 FPS를 video.json에 동기화
            updated_fps = 0
            try:
                main_window = self.parent()  # self.parent()가 MainWindow 인스턴스
                if main_window and hasattr(main_window, "cmb_movie_fps"):
                    cmb_fps = getattr(main_window, "cmb_movie_fps")
                    ui_fps = int(cmb_fps.currentData())
                    updated_fps = ui_fps

                    # self.full_video_data (video.json의 메모리 내 복사본)에 FPS 값 주입
                    self.full_video_data.setdefault("defaults", {})
                    self.full_video_data["defaults"].setdefault("movie", {})
                    self.full_video_data["defaults"]["movie"]["target_fps"] = ui_fps
                    self.full_video_data["defaults"]["movie"]["input_fps"] = ui_fps
                    self.full_video_data["defaults"]["movie"]["fps"] = ui_fps
                    self.full_video_data["defaults"].setdefault("image", {})["fps"] = ui_fps
                    self.full_video_data["fps"] = int(ui_fps)

                    print(f"[JSON Edit] '업데이트' 시 UI FPS ({ui_fps})를 video.json 데이터에 동기화했습니다.")

            except Exception as e_fps_sync:
                print(f"[JSON Edit] '업데이트' 중 FPS 동기화 실패: {e_fps_sync}")
            # --- ▲▲▲ [추가 끝] ▲▲▲ ---
            # 1. [v19] 캐릭터 스타일 업데이트 (self.style_widget_map 기준)
            updated_styles = {}
            updated_char_id_list = []

            if hasattr(self, "style_widget_map"):
                for char_id, (label, field_container) in self.style_widget_map.items():
                    edit_widget = field_container.findChild(QtWidgets.QLineEdit)
                    if edit_widget:
                        updated_styles[char_id] = edit_widget.text().strip()
                        updated_char_id_list.append(char_id)

            self.full_video_data["character_styles"] = updated_styles
            self.full_video_data["characters"] = updated_char_id_list  # 루트 "characters" 갱신
            updated_styles_count = len(updated_styles)

            # 2. [v19] 씬별 'direct_prompt' 업데이트
            scene_map = {scene.get("id"): scene for scene in self.scenes_data if
                         isinstance(scene, dict) and "id" in scene}

            updated_prompts_count = 0
            for scene_id, text_edit in self.widget_map:
                if scene_id in scene_map:
                    new_prompt = text_edit.toPlainText().strip()
                    scene = scene_map[scene_id]
                    if scene.get("direct_prompt", "") != new_prompt:
                        scene["direct_prompt"] = new_prompt
                        updated_prompts_count += 1

            # 3. [신규] 씬별 'characters' 목록 업데이트 (self.scene_char_widgets 기준)
            updated_scene_chars_count = 0
            if hasattr(self, "scene_char_widgets"):
                for scene_id, list_widget in self.scene_char_widgets.items():
                    if scene_id in scene_map:
                        # QListWidget에서 현재 항목들을 읽어옵니다.
                        new_scene_chars = []
                        for i in range(list_widget.count()):
                            item = list_widget.item(i)
                            if item:
                                new_scene_chars.append(item.text())

                        # video.json 데이터 객체(scene_map[scene_id])에 반영
                        scene = scene_map[scene_id]
                        if scene.get("characters") != new_scene_chars:
                            scene["characters"] = new_scene_chars
                            updated_scene_chars_count += 1

            # 4. [기존] 파일 저장 (수정된 full_video_data 통째로)
            self.full_video_data["scenes"] = self.scenes_data
            save_json(self.json_path, self.full_video_data)

            # 5. [수정] 완료 메시지
            QtWidgets.QMessageBox.information(self, "업데이트 완료",
                                              f"파일에 저장되었습니다: {self.json_path.name}\n\n"
                                              f"- 캐릭터 스타일 {updated_styles_count}개 항목 업데이트됨\n"
                                              f"- (루트 'characters' 목록도 {len(updated_char_id_list)}개로 갱신됨)\n"
                                              f"- 씬 'direct_prompt' {updated_prompts_count}개 항목 업데이트됨\n"
                                              f"- 씬별 'characters' {updated_scene_chars_count}개 항목 업데이트됨")

        except Exception as e_update:
            QtWidgets.QMessageBox.critical(self, "저장 오류",
                                           f"파일을 저장하는 중 오류가 발생했습니다:\n{e_update}")


# (이하 MainWindow의 나머지 코드는 생략)


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
        self.project_dir: str = ""
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


        tag_boxes = getattr(self, "_tag_boxes", None)
        if not isinstance(tag_boxes, dict) or not tag_boxes:
            return

        # 이 함수 안에서만 쓰는 기본 7개 태그 집합
        basic_defaults_all = {
            "clean vocals",
            "clear diction",
            "natural articulation",
            "breath control",
            "warm emotional tone",
            "balanced mixing",
            "studio reverb light",
        }
        # 실제 UI에 있는 것만 걸러낸 리스트
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


        # --- proj_dir 안전 획득 ---
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

        # 저장된 태그 우선순위: checked_tags > tags_effective > 기본 7개
        if isinstance(meta.get("checked_tags"), list):
            selected = [str(x) for x in meta["checked_tags"]]
            should_persist = False
        elif isinstance(meta.get("tags_effective"), list):
            selected = [str(x) for x in meta["tags_effective"]]
            should_persist = True
        else:
            selected = list(basic_defaults)
            should_persist = True

        # 🔴 여기서 예전에는 BASIC_VOCAL_TAGS 를 썼는데,
        # 이 함수 안에서 만든 basic_defaults 를 써야 참조 오류가 안 난다.
        for basic_tag_name in basic_defaults:
            if basic_tag_name not in selected:
                selected.append(basic_tag_name)

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

    def _generate_and_save_ai_tags(self, project_path: Path, meta: dict, progress_callback: Callable) -> List[str]:
        """
        '긍정 프롬프트 (+)' 박스 내용을 AI로 보내 태그로 변환하고 project.json에 저장합니다.
        (수정됨: progress_callback을 항상 dict 형태로 변환하여 호출하도록 수정)
        """

        # 헬퍼 함수: progress_callback을 항상 dict 형태로 호출
        def _log_progress(message: str) -> None:
            """progress_callback을 항상 dict 형태로 호출하는 래퍼."""
            try:
                # 상위 run_job_with_progress_async의 progress 콜백은 dict를 받습니다.
                progress_callback({"msg": message, "stage": "AI Tags"})
            except Exception as e:
                # 만약 상위 콜백이 문자열을 인자로 받는 형태의 Fallback이라면:
                try:
                    progress_callback(f"[AI Tags] {message}")
                except Exception:
                    # 최후의 수단: 아무것도 하지 않음
                    pass


        prompt_text_src = ""
        try:
            if hasattr(self, "te_prompt_pos") and hasattr(self.te_prompt_pos, "toPlainText"):
                prompt_text_src = self.te_prompt_pos.toPlainText().strip()

            # 사용자가 수동으로 제거할 기본 텍스트 (안전장치)
            if prompt_text_src == "ace-step tag 추천해줘 :":
                prompt_text_src = ""
        except Exception as read_exc:
            # 콜백은 문자열만
            _log_progress(f"긍정 프롬프트 UI 읽기 실패: {read_exc}")
            prompt_text_src = ""

        ai_tags_result: List[str] = []

        if not prompt_text_src:
            _log_progress("긍정 프롬프트가 비어있어 AI 태그 생성을 건너뜁니다.")
        else:
            _log_progress(f"긍정 프롬프트 AI 태그 변환 시작... (내용: {prompt_text_src[:30]}...)")

            system_prompt = (
                "You are an expert tag generator for music AI. "
                "Based on the user's text, extract key themes, moods, genres, instruments, and styles. "
                "Return *only* a simple, comma-separated list of 5-10 relevant English tags. "
                "Do not add any other text, explanation, or markdown. Just the tags."
                "Example: lyrical, calm, pop, piano, female vocal, night, urban"
            )
            user_prompt = f'Text: "{prompt_text_src}"\n\nTags (comma-separated):'

            try:
                if not hasattr(self, "_ai"):
                    raise RuntimeError("AI instance (self._ai) not found.")

                # gemini 우선
                raw_response = self._ai.ask_smart(
                    system_prompt,
                    user_prompt,
                    prefer="gemini",
                    allow_fallback=True,
                )

                # dict / list 등 올 수 있으니 문자열로 정규화
                import json

                if isinstance(raw_response, dict):
                    # 모델이 줄 법한 키 우선
                    text_body = (
                            raw_response.get("tags")
                            or raw_response.get("text")
                            or raw_response.get("content")
                            or raw_response.get("output")
                            or json.dumps(raw_response, ensure_ascii=False)
                    )
                elif isinstance(raw_response, (list, tuple)):
                    # ["pop", "jazz"] 이런 식이면 콤마로 합치기
                    text_body = ", ".join(str(item) for item in raw_response)
                else:
                    text_body = str(raw_response or "")

                text_body = text_body.strip()

                # 기본: 콤마로 나눈 태그
                ai_tags_result = [item.strip().lower() for item in text_body.split(",") if item.strip()]

                # "tags: ..." 형식이면 콜론 뒤만 다시 파싱
                if not ai_tags_result and ":" in text_body:
                    tail_text = text_body.split(":", 1)[-1]
                    ai_tags_result = [item.strip().lower() for item in tail_text.split(",") if item.strip()]

                _log_progress(f"AI 태그 생성 완료: {ai_tags_result}")
            except Exception as ai_exc:
                _log_progress(f"AI 태그 생성 실패: {ai_exc}. 빈 목록을 사용합니다.")
                ai_tags_result = []

        # project.json에 'prompt_user_ai_tags' 키로 저장
        try:
            meta["prompt_user_ai_tags"] = ai_tags_result
            save_json(project_path, meta)
            _log_progress(f"{len(ai_tags_result)}개의 AI 태그를 project.json에 저장했습니다.")
        except Exception as save_exc:
            _log_progress(f"project.json 저장 실패: {save_exc}")

        return ai_tags_result

    def _bind_actions(self) -> None:
        if getattr(self, "_actions_bound", False):
            return
        self._actions_bound = True


        def _btn(owner, name: str):
            obj = getattr(owner, name, None)
            return obj if isinstance(obj, QtWidgets.QAbstractButton) else None

        ui = getattr(self, "ui", None)

        pairs = [
            (_btn(ui, "btn_generate_lyrics") or _btn(self, "btn_gen"), self.on_generate_lyrics_with_log),
            (_btn(ui, "btn_generate_music") or _btn(self, "btn_music"), self.on_click_generate_music),

            # ▶ 프로젝트분석 버튼: seg → story → AI (비동기)
            (_btn(ui, "btn_test1_story") or _btn(self, "btn_test1_story"), self.on_click_build_story_from_seg_async),

            (_btn(ui, "btn_missing_img") or _btn(self, "btn_missing_img"),
             self.on_click_generate_missing_images_with_log),
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

        # 파일 IO 유틸
        try:

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
        - [수정] 긍정 프롬프트(+) AI 태그 변환을 먼저 수행하고 저장합니다.
        - [수정] AI가 생성한 태그를 수동 태그로 필터링하여 덮어씁니다.
        """

        # [수정] 필요한 유틸리티 함수들을 여기서 import


        btn = getattr(self, "btn_gen", None) or getattr(getattr(self, "ui", None), "btn_generate_lyrics", None)
        if btn:
            try:
                btn.setEnabled(False)
            except Exception:
                pass

        def _get_base_dir() -> Path:
            val = getattr(settings_mod, "BASE_DIR", ".")
            try:
                return Path(val)
            except Exception:
                return Path(".")

        def _get_proj_dir_str() -> str:
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
            import tempfile
            log_path = str(Path(tempfile.gettempdir()) / "lyrics_gen.log")

        def job(progress):
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

            # --- [신규] AI 태그 변환을 위한 project.json 경로 선-결정 ---
            title_in_pre = ""
            le = getattr(self, "le_title", None)
            if le and hasattr(le, "text"):
                try:
                    title_in_pre = (le.text() or "").strip()
                except Exception:
                    title_in_pre = ""

            prompt_text_pre = ""
            for nm in ("te_prompt", "txt_prompt", "prompt_edit"):
                w = getattr(self, nm, None) or getattr(getattr(self, "ui", None), nm, None)
                if w and hasattr(w, "toPlainText"):
                    try:
                        prompt_text_pre = (w.toPlainText() or "").strip(); break
                    except Exception:
                        prompt_text_pre = ""

            base_dir_path = _get_base_dir()

            temp_title = (title_in_pre or "").strip()
            if not temp_title:
                temp_title = "untitled"  # 제목 없으면 그냥 untitled 로

            temp_safe_title = sanitize_title(temp_title)

            # _normalize_maked_title_root 함수 필요 (utils.py에서 제공되어야 함)
            pj_path = _normalize_maked_title_root(base_dir_path) / temp_safe_title / "project.json"
            pj_path.parent.mkdir(parents=True, exist_ok=True)

            meta = load_json(pj_path, {}) or {}

            # --- [신규] 긍정 프롬프트 AI 태그 변환 호출 ---
            try:
                # 이 함수는 1번 단계에서 추가한 헬퍼 함수입니다.
                ai_tags_list = self._generate_and_save_ai_tags(pj_path, meta, _emit)
            except Exception as e_ai_tag:
                _emit(f"[AI Tags] 태그 생성 중 예외 발생: {e_ai_tag}")
                ai_tags_list = []
            # --- [신규] 끝 ---

            title_in = title_in_pre
            prompt_text = prompt_text_pre
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
                    if isinstance(vals, (list, set, tuple)): allowed = sorted(str(x) for x in vals)
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

            # --- [신규] 긍정 프롬프트 AI 태그로 덮어쓰기 ---
            final_tags_list = []
            if ai_tags_list:
                # _manual_option_set이 반환하는 '원본 라벨' 목록
                allowed_labels_from_ui = [str(t) for t in allowed if str(t).strip()]

                if allowed_labels_from_ui:  # 수동 태그가 하나라도 체크되어 있다면 필터링
                    # { "canon_key": "Original Label" } 맵
                    manual_map = {}
                    if hasattr(self, "_manual_option_map") and callable(self._manual_option_map):
                        try:
                            manual_map = self._manual_option_map()
                        except Exception:
                            manual_map = {}

                    # { "Original Label": "canon_key" } 역맵
                    label_to_canon = {v: k for k, v in manual_map.items()}
                    # UI에 있는 라벨의 canon_key 집합
                    allowed_canon_set = {label_to_canon.get(label, self.canon_key(label)) for label in
                                         allowed_labels_from_ui}

                    final_tags_list = []
                    seen_labels = set()

                    # 동의어 맵
                    syn_canon: dict[str, str] = {
                        "electronicdancemusic": "edm", "partybackground": "background music for parties",
                        "workout": "workout playlists", "electronicdrums": "soft electric drums",
                        "soulful": "soul", "funky": "funk",
                    }

                    for ai_tag in ai_tags_list:
                        ai_tag_canon = self.canon_key(ai_tag)
                        ai_tag_canon = syn_canon.get(ai_tag_canon, ai_tag_canon)  # 동의어 변환

                        if ai_tag_canon in allowed_canon_set:
                            official_label = manual_map.get(ai_tag_canon, ai_tag)  # 체크박스 원본 라벨
                            if official_label not in seen_labels:
                                final_tags_list.append(official_label)
                                seen_labels.add(official_label)
                    _emit(f"[AI Tags] 수동 태그({len(allowed_canon_set)}개)로 필터링 -> {final_tags_list}")

                else:  # 수동 태그가 없거나 '자동 태그' 모드이면 AI 태그 모두 사용
                    final_tags_list = ai_tags_list

            data["tags"] = final_tags_list  # (-> ace_tags)
            data["tags_pick"] = final_tags_list  # (-> tags_in_use)
            # --- [신규] 덮어쓰기 끝 ---

            data["_project_dir_path"] = str(pj_path.parent)

            return {"data": data, "title": title_in, "prompt": prompt_text, "proj_dir": str(pj_path.parent)}

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

            # ── 프로젝트 폴더 확정 (기존 로직 유지) ... ──
            final_dir = ""

            def _pick_dir_from_data(obj) -> str:
                if not isinstance(obj, dict): return ""
                paths = obj.get("paths")
                if isinstance(paths, dict):
                    v = paths.get("project_dir")
                    if isinstance(v, (str, bytes, os.PathLike)):
                        v2 = os.fspath(v)
                        if Path(v2).exists(): return v2
                title_guess = (obj.get("title") or pack.get("title") or "").strip()
                if title_guess:
                    root_path = _get_base_dir()
                    guess_path = _normalize_maked_title_root(root_path) / sanitize_title(title_guess)
                    if guess_path.exists(): return str(guess_path)
                return ""

            final_dir = _pick_dir_from_data(data)
            if not final_dir:
                pd = pack.get("proj_dir")
                if isinstance(pd, (str, bytes, os.PathLike)):
                    pd2 = os.fspath(pd)
                    if Path(pd2).exists(): final_dir = pd2

            if not final_dir:
                try:
                    cur_title = ""
                    le2 = getattr(self, "le_title", None)
                    if le2 and hasattr(le2, "text"): cur_title = (le2.text() or "").strip()
                    if cur_title:
                        base_dir = _get_base_dir()
                        guess2 = _normalize_maked_title_root(base_dir) / sanitize_title(cur_title)
                        if guess2.exists(): final_dir = str(guess2)
                except Exception:
                    pass

            if not final_dir:
                base_dir = _get_base_dir()
                try:
                    pj_list = list(_normalize_maked_title_root(base_dir).glob("*/project.json"))
                    pj_list.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    if pj_list:
                        import json
                        meta = json.loads(pj_list[0].read_text(encoding="utf-8")) or {}
                        paths_obj = meta.get("paths") if isinstance(meta, dict) else None
                        inner = str(paths_obj.get("project_dir") or "") if isinstance(paths_obj, dict) else ""
                        if inner and Path(inner).exists(): final_dir = inner
                except Exception:
                    pass

            if not final_dir:
                last = getattr(self, "_latest_project", None)
                if callable(last):
                    try:
                        lv = last()
                        if isinstance(lv, (str, bytes, os.PathLike)):
                            lv2 = os.fspath(lv)
                            if Path(lv2).exists(): final_dir = lv2
                    except Exception:
                        pass

            # -----------------------------------------------

            setter = getattr(self, "_set_active_project_dir", None)
            if final_dir and callable(setter):
                try:
                    setter(final_dir)
                except Exception:
                    pass

            if hasattr(self, "_apply_lyrics_result"):
                try:
                    # _apply_lyrics_result가 수정된 data (AI 태그 포함)를 사용합니다.
                    self._apply_lyrics_result(data, pack.get("title", ""), pack.get("prompt", ""))
                except Exception as e:
                    print(f"[ERROR] _apply_lyrics_result failed: {e}")

            try:
                setattr(self, "_project_context_ready", True)
            except Exception:
                pass

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

        run_job_with_progress_async(self, "가사 생성", job, tail_file=log_path, on_done=done)

    # --- 누락 이미지 생성: 비동기 + 진행창 로그 (no _guess_project_dir) ---
    def on_click_generate_missing_images_with_log(self) -> None:
        """video.json만 대상으로 누락된 씬 이미지를 생성한다(진행창/실시간 로그)."""


        # ----- 0) 버튼 비활성화(있을 때만) -----
        btn = getattr(self, "btn_missing_img", None) or getattr(getattr(self, "ui", None), "btn_missing_img", None)
        if isinstance(btn, QtWidgets.QAbstractButton):
            btn.setEnabled(False)

        # ----- 1) 대상 JSON 경로: 반드시 video.json만 -----
        video_path: Optional[Path] = None

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

        if video_path is None:
            proj_dir = ""
            getter = getattr(self, "_current_project_dir", None)
            if callable(getter):
                try:
                    proj_dir = getter() or ""
                except Exception:
                    proj_dir = ""
            if not proj_dir:
                proj_dir = getattr(self, "project_dir", "") or ""
            if proj_dir:
                v = (Path(proj_dir).resolve() / "video.json")
                if v.exists():
                    video_path = v

        if video_path is None:
            v = (Path.cwd() / "video.json").resolve()
            if v.exists():
                video_path = v

        if video_path is None or not video_path.exists():
            QtWidgets.QMessageBox.critical(self, "누락 이미지 생성", "video.json을 찾을 수 없습니다.")
            if isinstance(btn, QtWidgets.QAbstractButton):
                btn.setEnabled(True)
            return

        # ----- 2) 실행 전 정합성 보정: img_file 비어있는 씬에 기본 경로 -----
        try:

            doc = load_json(video_path, None)
            if isinstance(doc, dict):
                scenes = doc.get("scenes")
                if isinstance(scenes, list):
                    imgs_dir = video_path.parent / "imgs"
                    try:
                        imgs_dir.mkdir(parents=True, exist_ok=True)
                    except OSError:
                        imgs_dir = video_path.parent / "imgs"

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

                    # ★ 여기서 랜덤 시드도 같이 주입해버리자
                    #    Comfy 워크플로우에서 t2i_seed_for_workflow 같은 걸 읽도록 돼 있다면 여기 값이 쓰일 거야
                    new_seed_val = random.randint(1, 2_147_483_646)
                    doc.setdefault("defaults", {})
                    doc["defaults"].setdefault("image", {})
                    doc["defaults"]["image"]["t2i_seed"] = new_seed_val
                    doc["t2i_seed_for_workflow"] = new_seed_val

                    if changed or True:  # 시드 넣었으니까 저장
                        save_json(video_path, doc)

        except Exception as e_prep:
            print(f"[prep] img_file/seed 보정 경고: {e_prep}")

        # ----- 3) UI에서 W/H/스텝 읽기 -----
        def _get_combo_int(combo_name: str, default_val: int) -> int:
            combo = getattr(self, combo_name, None)
            try:
                val = combo.currentData()
                return int(val if val is not None else default_val)
            except Exception:
                return int(default_val)

        ui_w = _get_combo_int("cmb_img_w", 720)
        ui_h = _get_combo_int("cmb_img_h", 1080)

        spn = getattr(self, "spn_t2i_steps", None)
        try:
            steps = int(spn.value()) if spn is not None else 16
        except Exception:
            steps = 28

        # ----- 4) 비동기 작업 실행 -----

        try:
            comfy_log_file = getattr(_settings, "COMFY_LOG_FILE", None)
        except (ImportError, AttributeError):
            comfy_log_file = None

        def job(on_progress):
            created = build_missing_images_from_story(
                str(video_path),
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

    # shorts_ui.py의 MainWindow 클래스 내부에 이 함수를 통째로 덮어쓰세요.

    def on_click_segments_missing_images_with_log(self):
        """
        [2단계] 세그먼트 이미지 생성 (Qwen I2I) - 주입 검증 강화판

        수정 사항:
        1. [Fix] ReActor 매핑 오류 수정 (23번 ReActor의 소스는 20번이 아니라 18번).
           -> 20번(메인 이미지)이 오염되지 않도록 보호.
        2. [Debug] 주입되는 이미지 경로와 프롬프트를 로그에 출력.
        """



        btn = getattr(self, "btn_segments_img", None)
        if btn: btn.setEnabled(False)

        try:
            proj_dir = self._current_project_dir()
            if not proj_dir or not proj_dir.exists():
                raise FileNotFoundError("프로젝트 디렉터리를 찾을 수 없습니다.")

            video_json_path = proj_dir / "video.json"
            if not video_json_path.exists():
                raise FileNotFoundError(f"video.json을 찾을 수 없습니다.\n{video_json_path}")

            qwen_wf_path = Path(JSONS_DIR) / "qwen2509_i2i.json"
            if not qwen_wf_path.exists():
                raise FileNotFoundError(f"Qwen I2I 워크플로우를 찾을 수 없습니다.\n{qwen_wf_path}")

            qwen_workflow_template = load_json(qwen_wf_path)
            if not qwen_workflow_template:
                raise ValueError("qwen2509_i2i.json 파일을 읽을 수 없습니다.")

            comfy_input_path = Path(COMFY_INPUT_DIR)
            character_base_dir = Path(CHARACTER_DIR)

        except Exception as e_prep:
            QtWidgets.QMessageBox.critical(self, "준비 오류", str(e_prep))
            if btn: btn.setEnabled(True)
            return

        def job(progress_callback):
            _log = lambda msg: progress_callback({"msg": msg})
            _log(f"qwen2509_i2i.json 워크플로우(ReActor 포함)를 사용하여 키프레임 생성을 시작합니다.")

            # --- 헬퍼 함수 ---
            def _parse_character_spec(raw_obj: Any) -> Dict[str, Any]:
                if isinstance(raw_obj, str):
                    parts = raw_obj.split(":", 1)
                    cid = parts[0].strip()
                    idxx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                    return {"id": cid, "index": idxx}
                if isinstance(raw_obj, dict):
                    cid = str(raw_obj.get("id", "")).strip()
                    idxx = raw_obj.get("index")
                    try:
                        idx_int = int(idxx) if idxx is not None else 0
                    except:
                        idx_int = 0
                    return {"id": cid, "index": idx_int}
                return {"id": "", "index": 0}

            def _resolve_character_image_path(char_id: str) -> Optional[Path]:
                if not char_id: return None
                for ext in (".png", ".jpg", ".jpeg", ".webp"):
                    p = character_base_dir / f"{char_id}{ext}"
                    if p.exists(): return p
                return None

            def _copy_to_input(src_path: Path) -> str:
                try:
                    dst = comfy_input_path / src_path.name
                    shutil.copy2(str(src_path), str(dst))
                    return src_path.name
                except Exception as e:
                    _log(f"[WARN] 파일 복사 실패 ({src_path.name}): {e}")
                    return ""

            vdoc = load_json(video_json_path)
            scenes = vdoc.get("scenes", [])
            total_segments = sum(len(s.get("frame_segments", [])) for s in scenes)
            _log(f"총 {len(scenes)}개 씬, {total_segments}개 세그먼트(키프레임) 처리 예정.")

            processed_count = 0

            # ★ [수정됨] ReActor ID 매핑 (정확한 Source Loader ID로 수정)
            # Node 20: 메인 입력 이미지 (건드리면 안 됨, Source로 쓰면 안 됨)
            # Node 18: ReActor_2 (23번)용 Source Loader
            # Node 17: ReActor_1 (19번)용 Source Loader
            # Node 21: ReActor_0 (22번)용 Source Loader
            QWEN_REACTOR_MAP = {
                "19": ("17", "female_01"),  # Slot 1
                "22": ("21", "male_01"),  # Slot 2
                "23": ("18", "other_char"),  # Slot 3 (20 -> 18로 수정!)
            }

            SAVE_NODE_ID = "24"

            for scene_index, scene in enumerate(scenes):
                scene_id = scene.get("id")
                img_file_str = scene.get("img_file")
                frame_segments = scene.get("frame_segments", [])

                if not scene_id or not img_file_str or not frame_segments:
                    continue

                base_image_path = Path(img_file_str)
                if not base_image_path.exists():
                    _log(f"[{scene_id}] 스킵: 기준 이미지 없음.")
                    continue

                # 메인 이미지 복사
                main_img_name = _copy_to_input(base_image_path)

                keyframe_dir = base_image_path.parent / scene_id
                keyframe_dir.mkdir(parents=True, exist_ok=True)

                # 원본 보존 (kf_0)
                kf_0_path = keyframe_dir / "kf_0.png"
                if not kf_0_path.exists():
                    shutil.copy2(str(base_image_path), str(kf_0_path))

                # --- ReActor 설정 준비 ---
                scene_char_specs: Dict[str, int] = {}
                for char_raw in scene.get("characters", []):
                    spec = _parse_character_spec(char_raw)
                    if spec["id"]: scene_char_specs[spec["id"]] = spec["index"]

                face_files_to_inject: Dict[str, str] = {}
                reactors_to_enable: Dict[str, str] = {}

                for reactor_id, (load_id, default_char) in QWEN_REACTOR_MAP.items():
                    if default_char in scene_char_specs:
                        idx = scene_char_specs[default_char]
                        fpath = _resolve_character_image_path(default_char)
                        if fpath:
                            fname = _copy_to_input(fpath)
                            if fname:
                                face_files_to_inject[load_id] = fname
                                reactors_to_enable[reactor_id] = str(idx)

                # --- 세그먼트 루프 ---
                for seg_index, segment in enumerate(frame_segments):
                    seg_prompt = segment.get("prompt_movie", "").strip()
                    if not seg_prompt: continue

                    keyframe_name = f"kf_{seg_index + 1}.png"
                    keyframe_path = keyframe_dir / keyframe_name

                    if keyframe_path.exists():
                        processed_count += 1
                        continue

                    _log(f"[{scene_id}] 세그먼트 {seg_index + 1} 생성 중...")

                    try:
                        wf = json.loads(json.dumps(qwen_workflow_template))

                        # 1. ★ 메인 이미지 주입 (Node 20) ★
                        if "20" in wf:
                            wf["20"]["inputs"]["image"] = main_img_name
                            _log(f"   -> [Input] Image: {main_img_name}")
                        else:
                            _log("   -> [ERR] Node 20 (LoadImage) not found!")

                        # 2. ★ 프롬프트 주입 (Node 7) ★
                        if "7" in wf:
                            wf["7"]["inputs"]["prompt"] = seg_prompt
                            _log(f"   -> [Input] Prompt: {seg_prompt[:50]}...")

                        # 3. 시드
                        if "4" in wf:
                            wf["4"]["inputs"]["seed"] = (scene_index * 1000) + seg_index + 12345

                        # 4. ReActor 설정
                        for r_id, (l_id, def_char) in QWEN_REACTOR_MAP.items():
                            if r_id in wf:
                                if r_id in reactors_to_enable:
                                    wf[r_id]["inputs"]["enabled"] = True
                                    wf[r_id]["inputs"]["input_faces_index"] = reactors_to_enable[r_id]
                                    wf[r_id]["inputs"]["source_faces_index"] = 0
                                    if l_id in wf and l_id in face_files_to_inject:
                                        wf[l_id]["inputs"]["image"] = face_files_to_inject[l_id]
                                        _log(f"   -> [ReActor] {def_char} ON (Node {r_id})")
                                else:
                                    wf[r_id]["inputs"]["enabled"] = False

                        # 5. 저장 파일명 설정
                        if SAVE_NODE_ID in wf:
                            wf[SAVE_NODE_ID]["inputs"]["filename_prefix"] = f"qwen_kf/{scene_id}/{seg_index + 1}"

                        # 실행
                        result = _submit_and_wait_comfy(COMFY_HOST, wf, timeout=900, poll=3.0)

                        # 결과 파싱
                        outputs_dict = result.get("outputs", {})
                        target_out = outputs_dict.get(SAVE_NODE_ID, {})
                        images = target_out.get("images", [])

                        # Fallback
                        if not images:
                            for nid, out_val in outputs_dict.items():
                                if "images" in out_val:
                                    images = out_val["images"]
                                    break

                        if not images:
                            raise RuntimeError("이미지 생성 실패 (출력 없음)")

                        img_info = images[0]
                        filename = img_info.get("filename")
                        subfolder = img_info.get("subfolder")
                        file_type = img_info.get("type", "output")

                        _log(f"   -> [Download] {filename}")

                        resp = requests.get(
                            f"{COMFY_HOST}/view",
                            params={"filename": filename, "subfolder": subfolder, "type": file_type},
                            timeout=60
                        )
                        resp.raise_for_status()

                        with open(keyframe_path, "wb") as f:
                            f.write(resp.content)

                        processed_count += 1

                    except Exception as e_seg:
                        _log(f"[{scene_id}] 세그먼트 {seg_index + 1} 실패: {e_seg}")

            return f"완료. 생성된 키프레임: {processed_count}개"

        def done(ok, payload, err):
            if btn: btn.setEnabled(True)
            if not ok:
                QtWidgets.QMessageBox.critical(self, "실패", str(err))
            else:
                QtWidgets.QMessageBox.information(self, "완료", str(payload))

        run_job_with_progress_async(self, "세그먼트 이미지 생성 (Qwen)", job, on_done=done)




    def on_click_generate_music(self) -> None:
        """
        음악 생성 버튼 핸들러
        - [수정] 긍정 프롬프트(+) AI 태그 변환을 먼저 수행하고 저장합니다.
        - [수정] prompt_user 저장 로직을 제거하고, AI 태그를 ace_tags로 사용합니다.
        """

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

            progress({"msg": "[ui] 음악 생성 시작"})

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

            # --- ▼▼▼ [신규] 긍정 프롬프트 AI 태그 변환 ▼▼▼ ---
            try:
                # 이 함수는 1번 단계에서 추가한 헬퍼 함수입니다.
                self._generate_and_save_ai_tags(pj, meta, lambda d: progress({"msg": d.get("msg", "")}))
                # 저장이 완료된 meta를 다시 로드
                meta = load_json(pj, {}) or {}
            except Exception as e_ai_tag:
                progress({"msg": f"[AI Tags] 태그 생성 중 예외 발생: {e_ai_tag}"})
            # --- ▲▲▲ [신규] 끝 ▲▲▲ ---

            # --- ▼▼▼ [제거] 긍정 프롬프트 저장 로직 제거 ▼▼▼ ---
            # (pos_prompt, meta['prompt_user'] 등 관련 로직 삭제)
            # --- ▲▲▲ [제거] 끝 ▲▲▲ ---

            # --- [유지] 부정 프롬프트 저장 로직 ---
            neg_prompt = ""
            if hasattr(self, "te_prompt_neg") and hasattr(self.te_prompt_neg, "toPlainText"):
                try:
                    neg_prompt = self.te_prompt_neg.toPlainText().strip()
                except (AttributeError, RuntimeError):
                    neg_prompt = ""
            meta['prompt_neg'] = neg_prompt
            progress({"msg": f"[UI] 부정 프롬프트(-) 저장: {neg_prompt[:50]}..."})

            # --- [유지] UI 태그 및 시간 저장 로직 ---
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
                                    if label: picked_manual.append(label)
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
                # [수정] ace_tags의 기본값을 AI 태그(prompt_user_ai_tags)로 변경
                meta["ace_tags"] = meta.get("prompt_user_ai_tags", last_ai_tags)
                meta["tags_in_use"] = list(dict.fromkeys(picked_manual))
            else:
                meta["auto_tags"] = False
                meta["manual_tags"] = list(dict.fromkeys(picked_manual))

            save_json(pj, meta)  # 수정된 meta 저장


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

    def on_click_analyze_music(self, *, on_done_override: Optional[Callable] = None) -> None: # <-- 시그니처 수정
        """
        (단순화) 음악분석: UI에서 필요한 정보를 수집하여 audio_sync.py의 핵심 분석 함수를 호출하고
        ...
        """

        print("\n--- DEBUG: on_click_analyze_music 함수가 성공적으로 호출되었습니다. ---")

        # --- 1. UI 요소 및 경로 준비 ---
        btn = getattr(self, "btn_analyze", None)
        if isinstance(btn, QtWidgets.QAbstractButton):
            btn.setEnabled(False)


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
                if on_done_override:
                    on_done_override(ok, payload, err)  # 매크로 콜백이 있으면 그것만 호출
                elif not ok:
                    QtWidgets.QMessageBox.critical(self, "음악분석 실패", str(err))
                    return
                else:
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

    # shos_ui.py 파일의 on_click_build_story_from_seg_async 함수 전체를 교체하세요.
    def fps_to_partion(self, video_json_path: Path) -> int:
        """
        UI에서 렌더 FPS 값을 읽어 video.json의 defaults.movie.target_fps 및
        defaults.image.fps, 그리고 루트 fps에 저장하고, 해당 FPS 값을 반환합니다.
        (함수 이름은 'fps_to_partion'이지만 실제로는 FPS 값 저장/반환 역할)
        """

        # 1) UI에서 FPS 읽기
        ui_fps_widget = getattr(self, "cmb_movie_fps", None)
        try:
            target_fps_val = int(ui_fps_widget.currentData()) if ui_fps_widget and hasattr(ui_fps_widget,
                                                                                           "currentData") else 30
        except Exception:
            target_fps_val = 30
            print("[WARN] UI에서 FPS 값 읽기 실패, 기본값 30 사용.")

        # 2) video.json 로드
        video_doc = load_json(video_json_path, {}) or {}
        if not isinstance(video_doc, dict):
            video_doc = {}

        # 3) defaults 섹션에 FPS 주입
        video_doc.setdefault("defaults", {})
        movie_defaults = video_doc["defaults"].setdefault("movie", {})
        image_defaults = video_doc["defaults"].setdefault("image", {})

        movie_defaults["target_fps"] = target_fps_val
        movie_defaults["input_fps"] = target_fps_val  # 레거시/호환
        movie_defaults["fps"] = target_fps_val  # 레거시/호환
        image_defaults["fps"] = target_fps_val

        # ✅ 루트 fps도 같이 써줘야, (doc.get("fps"))를 읽는 코드가 제대로 작동함
        video_doc["fps"] = target_fps_val

        # 4) 저장
        try:
            save_json(video_json_path, video_doc)
            print(f"[FPS-PARTITION] {target_fps_val} FPS 저장 완료: {video_json_path.name}")
        except Exception as e_save_fps:
            print(f"[ERROR] video.json에 FPS 값 저장 실패: {e_save_fps}")

        return target_fps_val

    # shorts_ui.py의 on_click_build_story_from_seg_async 함수를
    # (다시) 아래의 전체 코드로 덮어쓰십시오.

    def on_click_build_story_from_seg_async(self) -> None:
        """
        seg.json → story.json 스켈레톤 → video.json(갭 포함) 생성 → video.json만 AI 강화 → 프롬프트 주입 → 가사 재주입
        - [수정됨] 'huge breasts' 체크박스 상태를 읽어 환경 변수(FORCE_HUGE_BREASTS)에 설정.
        - [수정됨] AI 강화(apply_gpt_to_story_v11) 호출 시 force_huge_breasts 플래그 전달.
        - [수정됨] AI 클래스 임포트 시 ai_class 대신 AI 직접 사용.
        - [수정됨] 예외 처리 구체화 및 변수명 명확화.
        - [수정됨] 6단계 추가: UI FPS를 video.json에 저장하고 fill_prompt_movie_with_ai를 호출.
        - 갭 씬은 프롬프트 비우고, 가사도 항상 빈 문자열 유지(반주/인터루드)
        - gap ID는 3자리(gap_###)이며, 직전 t_### 번호를 따름(예: t_007 → gap_007)
        """


        # 재진입 방지
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



        # 프로젝트 경로 확인
        proj_dir_str = ""
        try:
            proj_dir_callable = getattr(self, "_current_project_dir", None)
            if callable(proj_dir_callable):
                proj_dir_val = proj_dir_callable()
                proj_dir_str = str(proj_dir_val) if isinstance(proj_dir_val, Path) else proj_dir_val or ""
            else:
                proj_dir_attr = proj_dir_callable or getattr(self, "project_dir", None)
                proj_dir_str = str(proj_dir_attr) if isinstance(proj_dir_attr, Path) else proj_dir_attr or ""

            if not proj_dir_str:
                raise ValueError("프로젝트 경로를 가져올 수 없습니다.")

        except AttributeError:
            proj_dir_str = ""
        except (ValueError, TypeError, OSError) as e_proj_dir:
            print(f"[WARN] 프로젝트 경로 확인 중 오류: {type(e_proj_dir).__name__}: {e_proj_dir}")
            proj_dir_str = ""
        except Exception as e_unknown:
            print(f"[ERROR] 프로젝트 경로 확인 중 예상치 못한 오류: {type(e_unknown).__name__}: {e_unknown}")
            proj_dir_str = ""

        if not proj_dir_str:
            QtWidgets.QMessageBox.warning(self, "오류", "프로젝트 폴더를 선택하거나 생성해 주세요.")
            self._seg_story_busy = False
            if btn_build_story is not None:
                try:
                    btn_build_story.setEnabled(True)
                except RuntimeError:
                    pass
            return

        # 경로 객체 생성 및 seg.json 확인
        try:
            proj_dir_path = Path(proj_dir_str).resolve()
            seg_json_path = proj_dir_path / "seg.json"
            if not seg_json_path.is_file():
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
        except OSError as e_os:
            QtWidgets.QMessageBox.critical(self, "오류", f"경로 접근 오류: {e_os}")
            self._seg_story_busy = False
            if btn_build_story is not None:
                try:
                    btn_build_story.setEnabled(True)
                except RuntimeError:
                    pass
            return

        # seg 원문 가사 재주입 함수 (내부 함수로 정의)
        def _reinject_lyrics_from_seg(target_json_path: Path, source_story_path: Path,
                                      log_fn: Callable[[str], None]) -> None:
            """
            [수정됨]
            - 원본 seg.json 대신, 1단계에서 생성된 story.json (source_story_path)을 읽는다.
            - 'start' 시간 대신 'id'를 키로 매칭하여 가사를 재주입한다.
            """
            try:
                target_doc = load_json(target_json_path, None)
                if not isinstance(target_doc, dict):
                    log_fn("[WARN] 가사 재주입 실패: 대상 JSON(video.json) 형식이 dict가 아닙니다.")
                    return

                source_doc = load_json(source_story_path, None)
                if not isinstance(source_doc, dict):
                    log_fn(f"[WARN] 가사 재주입 실패: 원본 JSON(story.json) 형식이 dict가 아닙니다. 경로: {source_story_path}")
                    return

                scenes_raw = target_doc.get("scenes")
                if not isinstance(scenes_raw, list):
                    log_fn("[WARN] 가사 재주입 실패: 대상 JSON(video.json)에 'scenes' 리스트가 없습니다.")
                    return
                scenes: List[Dict[str, Any]] = scenes_raw

                # --- ▼▼▼ [핵심 수정] ▼▼▼ ---
                # 'start' 시간 대신 'id'를 키로 하는 가사 맵 생성 (story.json 기준)
                source_scenes = source_doc.get("scenes", [])
                if not isinstance(source_scenes, list):
                    log_fn(f"[WARN] 원본 JSON(story.json)에 'scenes' 리스트가 없습니다.")
                    return

                lyric_map: Dict[str, str] = {}
                for src_scene in source_scenes:
                    if isinstance(src_scene, dict):
                        scene_id_key = str(src_scene.get("id", "")).strip()
                        # [주의] story.json의 lyric을 사용 (사용자 의도)
                        text_val = str(src_scene.get("lyric", "")).strip()
                        if scene_id_key and text_val:
                            lyric_map[scene_id_key] = text_val
                # --- ▲▲▲ [핵심 수정] ▲▲▲ ---

                reinjected_count = 0
                cleared_gap_count = 0

                for sc_item in scenes:
                    if not isinstance(sc_item, dict):
                        continue
                    sc_id_str = str(sc_item.get("id") or "")
                    sc_origin_str = str(sc_item.get("origin") or "")
                    is_gap_scene = sc_id_str.startswith("gap_") or sc_origin_str == "gap-fill"

                    if is_gap_scene:
                        if sc_item.get("lyric"):
                            sc_item["lyric"] = ""
                            cleared_gap_count += 1
                        continue

                    # --- ▼▼▼ [핵심 수정] ▼▼▼ ---
                    # 'start' 시간(sc_key) 대신 'id'(sc_id_str)로 맵에서 검색
                    if sc_id_str in lyric_map:
                        original_text = lyric_map[sc_id_str]
                        if original_text and sc_item.get("lyric", "") != original_text:
                            sc_item["lyric"] = original_text
                            reinjected_count += 1
                    # --- ▲▲▲ [핵심 수정] ▲▲▲ ---

                if reinjected_count > 0 or cleared_gap_count > 0:
                    target_doc["scenes"] = scenes
                    try:
                        save_json(target_json_path, target_doc)
                        log_fn(
                            f"[INFO] 가사 재주입 완료 (ID 매칭): 정상 씬 복원 {reinjected_count}개, 갭 씬 정리 {cleared_gap_count}개")
                    except (OSError, TypeError) as e_save_reinject:
                        log_fn(f"[ERROR] 가사 재주입 저장 오류: {type(e_save_reinject).__name__}: {e_save_reinject}")
                else:
                    log_fn("[INFO] 가사 재주입 (ID 매칭): 변경 없음")

            except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError) as e_reinject:
                log_fn(f"[ERROR] 가사 재주입 오류: {type(e_reinject).__name__}: {e_reinject}")
            except Exception as e_unknown_reinject:
                log_fn(f"[ERROR] 가사 재주입 중 예상치 못한 오류: {type(e_unknown_reinject).__name__}: {e_unknown_reinject}")

        # --- ▼▼▼ [수정] video_json_path 변수를 job 함수 상단에서 미리 정의 ▼▼▼ ---
        video_json_path = proj_dir_path / "video.json"

        # --- ▼▼▼ [수정] FPS 읽기 도우미 함수 추가 ▼▼▼ ---
        def _read_ui_fps() -> int:
            """(on_click_test1_analyze에서 가져옴)"""
            try:
                cmb = getattr(self, "cmb_movie_fps", None)
                return int(cmb.currentData()) if cmb and hasattr(cmb, "currentData") else 30
            except Exception:
                return 30

        # --- ▼▼▼ [수정] AI 호출 래퍼 추가 ▼▼▼ ---
        def _ask_en(system_msg: str, user_msg: str) -> str:
            """(on_click_test1_analyze에서 가져옴)"""
            try:
                # self._ai 인스턴스가 AI 클래스라고 가정
                return str(self._ai.ask_smart(system_msg, user_msg, prefer="gemini", allow_fallback=True) or "")
            except Exception:
                return ""

        # (참고: self.fps_to_partion 함수는 UI 스레드에서 이미 호출되었으므로 여기서는 호출 X)

        # ----- job 함수 정의 -----
        def job(on_progress_callback: Callable[[dict], None]) -> Dict[str, str]:


            # --- ▼▼▼ [수정됨] _log_progress 정의를 최상단으로 이동 ▼▼▼ ---
            def _log_progress(message: str) -> None:
                """진행률 콜백을 호출하는 내부 헬퍼 함수."""
                try:
                    on_progress_callback({"msg": message})
                except Exception as e_log:
                    print(f"[ERROR] Progress callback failed: {e_log}")

            # --- ▲▲▲ [수정됨] 이동 끝 ▲▲▲ ---

            # --- ▼▼▼ [신규] 체크박스 상태 읽기 ▼▼▼ ---
            try:
                checkbox = getattr(self, "chk_huge_breasts", None)
                if checkbox and callable(getattr(checkbox, "isChecked", None)):
                    force_huge = bool(checkbox.isChecked())
                else:
                    force_huge = False
            except Exception:
                force_huge = False

            os.environ["FORCE_HUGE_BREASTS"] = "1" if force_huge else "0"
            _log_progress(f"[INFO] 'huge breasts' 강제 주입: {force_huge}")

            # --- ▲▲▲ [신규] 체크박스 상태 읽기 끝 ▲▲▲ ---

            def _detect_prefer_from_ui() -> str:
                default_prefer = "gemini"
                widget_name_list = [
                    "ai_mode", "mode", "cmb_ai", "combo_ai", "cmb_model", "combo_model",
                    "model", "provider", "combo_provider", "btn_ai_toggle"
                ]
                for widget_name in widget_name_list:
                    try:
                        widget_obj = getattr(self, widget_name, None) or \
                                     getattr(getattr(self, "ui", None), widget_name, None)

                        current_text_value = ""

                        if widget_name == "btn_ai_toggle" and hasattr(widget_obj, "isChecked") and callable(
                                widget_obj.isChecked):
                            current_text_value = "gemini" if widget_obj.isChecked() else "openai"

                        elif hasattr(widget_obj, "currentText") and callable(widget_obj.currentText):
                            current_text_value = str(widget_obj.currentText())
                        elif isinstance(widget_obj, str):
                            current_text_value = widget_obj

                        if current_text_value:
                            text_lower = current_text_value.lower()
                            if "gemini" in text_lower: return "gemini"
                            if "openai" in text_lower or "gpt" in text_lower: return "openai"
                    except AttributeError:
                        pass
                    except Exception as e_detect:
                        _log_progress(f"[WARN] AI 선호도 감지 중 오류 ({widget_name}): {e_detect}")

                try:
                    ai_instance = AI()
                    ai_default_prefer = getattr(ai_instance, "default_prefer", None)
                    if isinstance(ai_default_prefer, str) and ai_default_prefer:
                        return ai_default_prefer.lower()
                except Exception as e_ai_init:
                    _log_progress(f"[WARN] AI 기본 선호도 가져오기 실패: {e_ai_init}")

                return default_prefer

            # 1) story.json 스켈레톤 생성 함수
            def _build_story_skeleton_internal(seg_doc_list: List[Dict[str, Any]], project_dir: Path) -> Path:
                _log_progress("[1/5] story.json 스켈레톤 생성 시작...")
                if not isinstance(seg_doc_list, list) or not seg_doc_list:
                    raise ValueError("seg.json 내용이 비어있거나 리스트 형식이 아닙니다.")

                meta_path = project_dir / "project.json"
                meta_info: Dict[str, Any] = {}
                try:
                    meta_info_raw = load_json(meta_path, None)
                    if isinstance(meta_info_raw, dict):
                        meta_info = meta_info_raw
                    else:
                        _log_progress("[WARN] project.json 형식이 잘못되었습니다 (빈 메타 사용)")
                except FileNotFoundError:
                    _log_progress("[WARN] project.json 파일을 찾을 수 없습니다 (빈 메타 사용)")
                except (json.JSONDecodeError, OSError) as e_load_meta:
                    _log_progress(f"[WARN] project.json 로드 실패 ({type(e_load_meta).__name__}, 빈 메타 사용)")

                imgs_dir = project_dir / "imgs"
                try:
                    imgs_dir.mkdir(parents=True, exist_ok=True)
                except OSError as e_mkdir:
                    _log_progress(f"[WARN] 이미지 폴더 생성 실패: {e_mkdir}")

                scenes_list: List[Dict[str, Any]] = []
                cursor_time = 0.0
                first_start_time = 0.0

                if seg_doc_list and isinstance(seg_doc_list[0], dict):
                    try:
                        first_start_time = float(seg_doc_list[0].get("start", 0.0))
                    except (ValueError, TypeError):
                        _log_progress("[WARN] 첫 세그먼트 시작 시간 변환 실패, 0.0 사용")
                        first_start_time = 0.0
                else:
                    _log_progress("[WARN] seg.json 첫 항목이 dict가 아니거나 비어있음")

                if first_start_time >= 1.0:
                    intro_end_time = round(first_start_time, 3)
                    intro_scene_id = "t_000"
                    scenes_list.append({
                        "id": intro_scene_id, "section": "intro", "label": "Intro",
                        "start": 0.0, "end": intro_end_time, "duration": intro_end_time,
                        "scene": "intro", "characters": [],
                        "effect": ["soft light", "film grain", "gentle camera pan"],
                        "screen_transition": True,
                        "img_file": str(imgs_dir / f"{intro_scene_id}.png"),
                        "prompt": "인트로: 부드러운 전환과 분위기 암시",
                        "prompt_img": "", "prompt_movie": "", "prompt_negative": "",
                        "lyric": "",
                    })
                    cursor_time = first_start_time

                for i_seg, row_data in enumerate(seg_doc_list, start=1):
                    if not isinstance(row_data, dict):
                        _log_progress(f"[WARN] seg.json {i_seg}번째 항목이 dict가 아님. 건너뜀")
                        continue

                    try:
                        start_time_val = float(row_data.get("start", 0.0))
                    except (ValueError, TypeError):
                        start_time_val = 0.0
                    try:
                        end_time_val = float(row_data.get("end", start_time_val))
                    except (ValueError, TypeError):
                        end_time_val = start_time_val

                    if end_time_val < start_time_val: end_time_val = start_time_val
                    if start_time_val < cursor_time: start_time_val = cursor_time

                    scene_id_str = f"t_{i_seg:03d}"
                    duration_val = round(max(0.0, end_time_val - start_time_val), 3)
                    lyric_text_val = str(row_data.get("text", "")).strip()

                    default_character = ["female_01"]
                    project_characters = meta_info.get("characters")
                    characters_to_use = project_characters if isinstance(project_characters,
                                                                         list) and project_characters else default_character

                    scenes_list.append({
                        "id": scene_id_str, "section": "verse", "label": "Verse",
                        "start": round(start_time_val, 3), "end": round(end_time_val, 3), "duration": duration_val,
                        "scene": "verse",
                        "characters": characters_to_use,
                        "effect": ["bokeh", "slow push-in"],
                        "screen_transition": False,
                        "img_file": str(imgs_dir / f"{scene_id_str}.png"),
                        "prompt": "장면 묘사 필요",
                        "prompt_img": "", "prompt_movie": "", "prompt_negative": "",
                        "lyric": lyric_text_val,
                    })
                    cursor_time = end_time_val

                last_end_time = cursor_time
                outro_start_time = last_end_time
                outro_duration = 5.0
                outro_end_time = outro_start_time + outro_duration
                outro_scene_index = len(seg_doc_list) + (1 if first_start_time >= 1.0 else 0) + 1
                outro_id_str = f"t_{outro_scene_index:03d}"

                scenes_list.append({
                    "id": outro_id_str, "section": "outro", "label": "Outro",
                    "start": round(outro_start_time, 3), "end": round(outro_end_time, 3),
                    "duration": round(outro_duration, 3),
                    "scene": "outro", "characters": [],
                    "effect": ["fade-out", "soft glow"],
                    "screen_transition": True,
                    "img_file": str(imgs_dir / f"{outro_id_str}.png"),
                    "prompt": "아웃트로: 여운과 마무리",
                    "prompt_img": "", "prompt_movie": "", "prompt_negative": "",
                    "lyric": "",
                })

                story_record: Dict[str, Any] = {
                    "title": str(meta_info.get("title") or project_dir.name),
                    "duration": round(outro_end_time, 3),
                    "offset": 0.0,
                    "lyrics": str(meta_info.get("lyrics") or ""),
                    "characters": meta_info.get("characters") or ["female_01"],
                    "character_styles": meta_info.get("character_styles") or {},
                    "global_context": meta_info.get("global_context") or {
                        "themes": [], "palette": "", "style_guide": "", "negative_bank": "",
                        "section_moods": {"intro": "", "verse": "", "chorus": "", "bridge": "", "outro": ""},
                        "effect": [],
                    },
                    "defaults": meta_info.get("defaults") or {
                        "image": {"width": 720, "height": 1080, "negative": "@global"}
                    },
                    "scenes": scenes_list,
                    "audit": {"built_from": "seg.json", "ai_used": False},
                }

                story_file_path = project_dir / "story.json"
                try:
                    save_json(story_file_path, story_record)
                    _log_progress(f"[1/5] story.json 스켈레톤 생성 완료: {story_file_path.name}")
                except (OSError, TypeError) as e_save_story:
                    _log_progress(f"[ERROR] story.json 저장 실패: {e_save_story}")
                return story_file_path

            # --- job 함수 메인 로직 시작 ---
            on_progress_callback({"stage": "start", "msg": f"프로젝트 분석 시작: {proj_dir_path.name}"})

            seg_content_raw: Optional[list] = None
            try:
                seg_content_raw = load_json(seg_json_path, None)
                if not isinstance(seg_content_raw, list):
                    raise ValueError(f"seg.json 파일 내용이 리스트 형식이 아닙니다: {seg_json_path}")
            except FileNotFoundError:
                raise FileNotFoundError(f"seg.json 파일을 찾을 수 없습니다: {seg_json_path}")
            except (json.JSONDecodeError, ValueError) as e_load_seg:
                raise ValueError(f"seg.json 로드 또는 파싱 실패: {e_load_seg}")
            except OSError as e_os_seg:
                raise OSError(f"seg.json 접근 오류: {e_os_seg}")

            seg_content: List[Dict[str, Any]] = seg_content_raw if seg_content_raw is not None else []

            try:
                story_path = _build_story_skeleton_internal(seg_content, proj_dir_path)
            except Exception as e_build_skel:
                raise RuntimeError(f"story.json 스켈레톤 생성 실패: {e_build_skel}") from e_build_skel

            _log_progress("[2/5] video.json(갭 포함) 생성...")
            video_path_str: Optional[str] = None
            try:
                video_path_result = build_video_json_with_gap_policy(str(proj_dir_path), small_gap_sec=2.0)
                if isinstance(video_path_result, str) and Path(video_path_result).is_file():
                    video_path_str = video_path_result
                    on_progress_callback(
                        {"stage": "video_json_built", "msg": f"video.json 생성 완료: {Path(video_path_str).name}"})
                else:
                    _log_progress("[WARN] video.json 생성 함수가 유효한 경로를 반환하지 않았습니다.")
            except FileNotFoundError:
                _log_progress("[WARN] video.json 생성 건너뜀 (story.json 없음)")
            except (TypeError, ValueError) as e_v:
                _log_progress(f"[WARN] video.json 생성 중 오류 발생: {type(e_v).__name__}: {e_v}")
            except Exception as e_v_unknown:
                _log_progress(f"[ERROR] video.json 생성 중 예상치 못한 오류: {type(e_v_unknown).__name__}: {e_v_unknown}")

            ai_client_instance: Optional[AI] = None
            try:
                ai_client_instance = AI()
            except Exception as e_ai_init_job:
                _log_progress(f"[WARN] AI 클라이언트 초기화 실패 ({type(e_ai_init_job).__name__}), AI 강화 생략.")

            prefer_detected_provider = _detect_prefer_from_ui()

            # _ask_wrapper_internal 내부 함수 (AI 호출 담당)
            def _ask_wrapper_internal(
                    system: str,
                    user: str,
                    *,
                    prefer: Optional[str] = None,
                    allow_fallback: Optional[bool] = None,
                    trace: Optional[Callable[[str, str], None]] = None,
            ) -> str:
                if ai_client_instance is None:
                    raise RuntimeError("AI 클라이언트가 초기화되지 않아 호출할 수 없습니다.")

                prefer_final = prefer or prefer_detected_provider or getattr(ai_client_instance, "default_prefer",
                                                                             "openai")
                allow_fallback_final = False if allow_fallback is None else bool(allow_fallback)

                ask_smart = getattr(ai_client_instance, "ask_smart", None)
                if not callable(ask_smart):
                    raise AttributeError("ask_smart method not found")

                sig = inspect.signature(ask_smart)
                call_kwargs: Dict[str, Any] = {}
                if "prefer" in sig.parameters:
                    call_kwargs["prefer"] = prefer_final
                if "allow_fallback" in sig.parameters:
                    call_kwargs["allow_fallback"] = allow_fallback_final
                if "trace" in sig.parameters and trace is not None:
                    call_kwargs["trace"] = trace

                # [수정됨] job 스코프의 force_huge 변수를 읽어 시스템 프롬프트 수정
                system_prompt_final = str(system)
                if force_huge:
                    rule_2_marker = "2) Always include character traits for each referenced character."
                    if rule_2_marker in system_prompt_final:
                        rule_2_replacement = (
                            "2) Always include character traits for each referenced character.\n"
                            "       - For any character id that starts with \"female\", you MUST include \"adult woman\" AND \"huge breasts\"."
                        )
                        system_prompt_final = system_prompt_final.replace(rule_2_marker, rule_2_replacement)
                    elif "# 엄격한 작성 규칙:" in system_prompt_final:  # story_enrich v6+
                        rule_marker = "- character_styles: 한국어 설명."
                        rule_replacement = (
                            "- character_styles: 한국어 설명.\n"
                            "- [중요 규칙] 'female' ID를 가진 character_styles 설명에 'huge breasts, slim legs'를 반드시 한국어로 포함."
                        )
                        system_prompt_final = system_prompt_final.replace(rule_marker, rule_replacement)
                    else:
                        system_prompt_final += "\n\nRULE: For 'female' characters, you MUST include 'huge breasts'."

                if trace:
                    try:
                        trace("ai.ask", f"prefer={prefer_final} allow_fallback={allow_fallback_final} "
                                        f"len={len(str(system_prompt_final))}/{len(str(user))}")
                    except Exception:
                        pass

                result_text = ask_smart(system_prompt_final, str(user),
                                        **call_kwargs)  # [수정] system_prompt_final 사용
                return str(result_text or "")

            if video_path_str and Path(video_path_str).is_file():
                _log_progress("[3/5] video.json AI 강화 시작...")
                try:
                    video_doc_raw = load_json(Path(video_path_str), None)
                    if isinstance(video_doc_raw, dict):

                        # --- ▼▼▼ [신규] apply_gpt_to_story_v11 호출 시 force_huge_breasts 인자 전달 ▼▼▼ ---
                        video_ai_enhanced = apply_gpt_to_story_v11(
                            video_doc_raw,
                            ask=_ask_wrapper_internal,
                            prefer=prefer_detected_provider,
                            allow_fallback=None,
                            trace=None,
                            force_huge_breasts=force_huge  # <-- [신규] 플래그 전달
                        )
                        # --- ▲▲▲ [신규] 호출 끝 ▲▲▲ ---

                        video_ai_enhanced.setdefault("audit", {})
                        video_ai_enhanced["audit"]["ai_used"] = True
                        video_ai_enhanced["audit"]["provider"] = prefer_detected_provider
                        save_json(Path(video_path_str), video_ai_enhanced)
                        on_progress_callback({"stage": "ai_done_video",
                                              "msg": f"video.json AI 강화 완료 (provider: {prefer_detected_provider})"})
                    else:
                        _log_progress("[WARN] video.json 형식이 dict가 아니라서 AI 강화 생략")
                except (RuntimeError, TypeError, ValueError) as e_ai_vid:
                    _log_progress(f"[WARN] video.json AI 강화 중 오류(진행 계속): {type(e_ai_vid).__name__}: {e_ai_vid}")
                    on_progress_callback({"stage": "ai_error_video", "msg": f"video.json AI 강화 실패: {e_ai_vid}"})
                except Exception as e_ai_unknown:
                    _log_progress(
                        f"[ERROR] video.json AI 강화 중 예상치 못한 오류: {type(e_ai_unknown).__name__}: {e_ai_unknown}")
                    on_progress_callback({"stage": "ai_error_video", "msg": f"video.json AI 강화 실패: {e_ai_unknown}"})
            elif video_path_str:
                _log_progress("[WARN] video.json 경로가 유효한 파일이 아니라 AI 강화 생략")
            else:
                _log_progress("[WARN] video.json 경로가 없어 AI 강화 생략")

            if video_path_str and Path(video_path_str).is_file():
                _log_progress("[4/5] video.json 프롬프트 주입/정규화...")
                try:
                    save_story_overwrite_with_prompts(Path(video_path_str))

                    try:
                        doc_norm = load_json(Path(video_path_str), None)
                        if isinstance(doc_norm, dict):
                            scenes_norm_raw = doc_norm.get("scenes")
                            if isinstance(scenes_norm_raw, list):
                                scenes_norm: List[Dict[str, Any]] = scenes_norm_raw
                                neg_changed_count = 0
                                for sc_norm in scenes_norm:
                                    if not isinstance(sc_norm, dict): continue

                                    neg_raw = str(sc_norm.get("prompt_negative", "")).strip()
                                    if neg_raw:
                                        neg_clean = neg_raw.replace("，", ",")
                                        neg_clean = re.sub(r"\s*,\s*", ",", neg_clean).strip(',')
                                        parts_list = [p.strip() for p in neg_clean.split(",") if p.strip()]
                                        seen_set: set[str] = set()
                                        uniq_list: list[str] = []
                                        for ptxt in parts_list:
                                            ptxt_lower = ptxt.lower()
                                            if ptxt_lower not in seen_set:
                                                seen_set.add(ptxt_lower)
                                                uniq_list.append(ptxt)
                                        final_neg = ", ".join(uniq_list)
                                        if sc_norm.get("prompt_negative") != final_neg:
                                            sc_norm["prompt_negative"] = final_neg
                                            neg_changed_count += 1

                                if neg_changed_count > 0:
                                    save_json(Path(video_path_str), doc_norm)
                                    _log_progress(f"[INFO] 네거티브 프롬프트 정리 완료 ({neg_changed_count}개 씬 수정됨)")

                            else:
                                _log_progress("[WARN] 네거티브 정리 실패: 'scenes'가 리스트가 아님")
                        else:
                            _log_progress("[WARN] 네거티브 정리 실패: video.json 형식이 dict가 아님")
                    except (OSError, ValueError, TypeError, json.JSONDecodeError) as e_norm:
                        _log_progress(f"[WARN] 네거티브 정리 중 경고: {type(e_norm).__name__}: {e_norm}")
                    except Exception as e_norm_unknown:
                        _log_progress(
                            f"[ERROR] 네거티브 정리 중 예상치 못한 오류: {type(e_norm_unknown).__name__}: {e_norm_unknown}")

                    on_progress_callback({"stage": "prompt_inject_done_video", "msg": "video.json 프롬프트 주입 완료"})
                except (RuntimeError, ValueError, TypeError) as e_prompt_vid:
                    _log_progress(f"[WARN] video.json 프롬프트 주입 오류: {type(e_prompt_vid).__name__}: {e_prompt_vid}")
                    on_progress_callback(
                        {"stage": "prompt_inject_error_video", "msg": f"video.json 프롬프트 주입 실패: {e_prompt_vid}"})
                except Exception as e_prompt_unknown:
                    _log_progress(
                        f"[ERROR] video.json 프롬프트 주입 중 예상치 못한 오류: {type(e_prompt_unknown).__name__}: {e_prompt_unknown}")
                    on_progress_callback(
                        {"stage": "prompt_inject_error_video", "msg": f"video.json 프롬프트 주입 실패: {e_prompt_unknown}"})
            elif video_path_str:
                _log_progress("[WARN] video.json 경로가 유효한 파일이 아니라 프롬프트 주입 생략")
            else:
                _log_progress("[WARN] video.json 경로가 없어 프롬프트 주입 생략")

            if video_path_str and Path(video_path_str).is_file():
                _log_progress("[5/5] video.json 가사 재주입 (ID 매칭)...")
                try:
                    # [수정] seg_json_path 대신 story_path (뼈대)를 원본으로 사용
                    _reinject_lyrics_from_seg(Path(video_path_str), story_path, _log_progress)
                    on_progress_callback(
                        {"stage": "lyric_reinject_done_video", "msg": "video.json 가사 재주입 시도 완료 (ID 매칭)"})
                except (FileNotFoundError, OSError, TypeError, ValueError) as e_reinject_call:
                    _log_progress(
                        f"[ERROR] video.json 가사 재주입 오류: {type(e_reinject_call).__name__}: {e_reinject_call}")
                    on_progress_callback(
                        {"stage": "lyric_reinject_error_video", "msg": f"video.json 가사 재주입 오류: {e_reinject_call}"})
                except Exception as e_reinject_unknown:
                    _log_progress(
                        f"[ERROR] video.json 가사 재주입 중 예상치D 못한 오류: {type(e_reinject_unknown).__name__}: {e_reinject_unknown}")
                    on_progress_callback(
                        {"stage": "lyric_reinject_error_video",
                         "msg": f"video.json 가사 재주입 오류: {e_reinject_unknown}"})
            elif video_path_str:
                _log_progress("[WARN] video.json 경로가 유효한 파일이 아니라 가사 재주입 생략")
            else:
                _log_progress("[WARN] video.json 경로가 없어 가사 재주입 생략")

            # --- ▼▼▼ [수정] 6단계: UI FPS 동기화 및 세그먼트 프롬프트 생성 ▼▼▼ ---
            # (on_click_test1_analyze에서 가져온 로직)
            _log_progress("[6/6] FPS 동기화 및 세그먼트 프롬프트 생성 시작...")
            try:
                # video_json_path (Path 객체)와 _ask_en (Callable)은
                # 이 job 함수 외부 스코프(on_click_build_story_from_seg_async)에 정의되어 있음
                # fill_prompt_movie_with_ai 함수는 파일 상단에서 임포트됨

                # --- [핵심 수정] UI FPS를 video.json에 먼저 저장 ---
                vdoc_fps = load_json(video_json_path, {}) or {}
                if not isinstance(vdoc_fps, dict): vdoc_fps = {}

                ui_fps_val = _read_ui_fps()  # helper 함수 호출
                vdoc_fps.setdefault("defaults", {})
                vdoc_fps["defaults"].setdefault("movie", {})["target_fps"] = ui_fps_val
                vdoc_fps["defaults"]["movie"]["input_fps"] = ui_fps_val
                vdoc_fps["defaults"]["movie"]["fps"] = ui_fps_val
                vdoc_fps["defaults"].setdefault("image", {})["fps"] = ui_fps_val
                vdoc_fps["fps"] = int(ui_fps_val)

                save_json(video_json_path, vdoc_fps)
                _log_progress(f"[6/6] UI FPS ({ui_fps_val}) 저장 완료.")
                # --- [수정] FPS 저장 끝 ---

                fill_prompt_movie_with_ai(
                    proj_dir_path,  # 1. Path 객체 (job 시작 시 정의됨)
                    _ask_en,  # 2. AI 호출 함수 (외부 스코프에서 가져옴)
                    log_fn=_log_progress  # 3. 로그 출력 함수
                )
                _log_progress("[6/6] 세그먼트 프롬프트 생성 완료 (UI FPS 기준).")

            except ImportError as e_import_fill:
                _log_progress(f"[ERROR] 6/6 단계: fill_prompt_movie_with_ai 임포트 실패: {e_import_fill}")
            except Exception as e_fill:
                _log_progress(f"[ERROR] 6/6 단계(세그먼트 프롬프트) 실행 실패: {e_fill}")
            # --- ▲▲▲ [수정] 6단계 끝 ▲▲▲ ---

            if video_path_str and Path(video_path_str).is_file():
                try:
                    vid_doc_final = load_json(Path(video_path_str), None)
                    if isinstance(vid_doc_final, dict):
                        aud_final = vid_doc_final.setdefault("audit", {})
                        aud_final["provider"] = prefer_detected_provider
                        save_json(Path(video_path_str), vid_doc_final)
                    else:
                        _log_progress("[WARN] provider 고정 실패: video.json 형식이 dict가 아님")
                except (OSError, TypeError, ValueError, json.JSONDecodeError) as e_final_save:
                    _log_progress(f"[WARN] 최종 provider 저장 실패: {type(e_final_save).__name__}: {e_final_save}")
                except Exception as e_final_unknown:
                    _log_progress(
                        f"[ERROR] 최종 provider 저장 중 예상치 못한 오류: {type(e_final_unknown).__name__}: {e_final_unknown}")

            # --- ▼▼▼ [수정] 반환값에 UI FPS 포함 ▼▼▼ ---
            final_fps_val = _read_ui_fps()  # 최종 저장된 UI FPS 값
            return {"story_path": str(story_path), "video_path": video_path_str or "", "fps": final_fps_val}
            # --- ▲▲▲ [수정] 끝 ▲▲▲ ---

        # ----- done 콜백 함수 정의 -----
        def done(ok: bool, payload: Optional[dict], err: Optional[Exception]) -> None:
            """작업 완료 후 UI 업데이트 및 상태 정리."""
            try:
                self._seg_story_busy = False
                if btn_build_story is not None:
                    try:
                        btn_build_story.setEnabled(True)
                    except RuntimeError:
                        pass

                if ok and payload:
                    story_path_out = payload.get("story_path") or ""
                    video_path_out = payload.get("video_path") or ""

                    # --- ▼▼▼ [수정] 완료 메시지에 FPS 추가 ▼▼▼ ---
                    fps_val_done = payload.get("fps")
                    msg_parts = ["프로젝트 분석 완료"]
                    if story_path_out: msg_parts.append(f"story: {Path(story_path_out).name}")
                    if video_path_out: msg_parts.append(f"video: {Path(video_path_out).name}")
                    if fps_val_done: msg_parts.append(f"FPS: {fps_val_done}")
                    final_msg = " | ".join(msg_parts)
                    # --- ▲▲▲ [수정] 끝 ▲▲▲ ---

                    print(f"[UI] {final_msg}")
                    try:
                        status_bar_obj = getattr(self, "status", None) or getattr(self, "statusBar", None)
                        status_bar = status_bar_obj() if callable(status_bar_obj) else status_bar_obj
                        if status_bar and hasattr(status_bar, "showMessage"):
                            status_bar.showMessage(final_msg, 5000)
                    except AttributeError:
                        pass
                    except Exception as e_status:
                        print(f"[WARN] 상태바 업데이트 실패: {e_status}")

                elif err is not None:
                    err_msg_detail = traceback.format_exception(type(err), err, err.__traceback__)
                    err_msg_short = f"프로젝트 분석 실패: {type(err).__name__}: {err}"

                    print(f"[UI][ERROR] {err_msg_short}\n{''.join(err_msg_detail)}")
                    QtWidgets.QMessageBox.critical(self, "오류", err_msg_short)
                    try:
                        status_bar_obj = getattr(self, "status", None) or getattr(self, "statusBar", None)
                        status_bar = status_bar_obj() if callable(status_bar_obj) else status_bar_obj
                        if status_bar and hasattr(status_bar, "showMessage"):
                            status_bar.showMessage(err_msg_short, 5000)
                    except AttributeError:
                        pass
                    except Exception as e_status_err:
                        print(f"[WARN] 상태바 에러 메시지 업데이트 실패: {e_status_err}")
                else:
                    print("[UI][ERROR] 프로젝트 분석 실패 (알 수 없는 오류)")
                    QtWidgets.QMessageBox.critical(self, "오류", "알 수 없는 오류로 프로젝트 분석에 실패했습니다.")

            except Exception as e_done:
                print(f"[ERROR] done 콜백 실행 중 오류: {type(e_done).__name__}: {e_done}")

        # 비동기 작업 실행
        run_job_with_progress_async(
            owner=self,
            title="프로젝트 분석 (seg → story 스켈레톤 → video AI 강화)",
            job=job,
            on_done=done
        )

    def _inject_ui_fps_to_video(self, video_json_path):
        """
        video.json에 UI에서 선택한 FPS를 주입한다.
        - root "fps"
        - defaults.movie.{target_fps,input_fps,fps}
        - defaults.image.fps
        를 모두 일관되게 세팅.
        """

        p = Path(video_json_path)
        doc = load_json(p, {}) or {}
        if not isinstance(doc, dict):
            doc = {}

        # 1) UI에서 FPS 읽기
        ui_fps = 30
        cmb = getattr(self, "cmb_movie_fps", None)
        try:
            if cmb and hasattr(cmb, "currentData"):
                val = cmb.currentData()
                ui_fps = int(val if val is not None else 30)
        except Exception:
            ui_fps = 30

        # 2) 문서에 주입
        doc.setdefault("defaults", {})
        movie = doc["defaults"].setdefault("movie", {})
        image = doc["defaults"].setdefault("image", {})

        doc["fps"] = ui_fps
        movie["target_fps"] = ui_fps
        movie["input_fps"] = ui_fps
        movie["fps"] = ui_fps  # 레거시 호환
        image["fps"] = ui_fps

        try:
            save_json(p, doc)
            print(f"[FPS] UI {ui_fps} FPS → {p.name} 반영 완료")
        except Exception as e:
            print(f"[FPS][WARN] 저장 실패: {e}")

        return ui_fps

    # all_ui.py (또는 해당 클래스 안)

    def on_click_test1_analyze(self) -> None:
        """
        [프로젝트 분석]
        - 기존 기능 100% 유지 (1/5 ~ 5/5)
        - video.json '최종 저장 직후'에:
            1) UI FPS를 root/ defaults.movie / defaults.image 에 동기화
            2) fill_prompt_movie_with_ai(Path(project_dir), ask_en) 1회 호출
               → duration×fps 기반 frame_segments 생성 + 세그먼트 프롬프트 보강
        """


        def _resolve_project_dir() -> Path:
            proj_dir = ""
            getdir = getattr(self, "_current_project_dir", None)
            if callable(getdir):
                try:
                    proj_dir = getdir() or ""
                except Exception:
                    proj_dir = ""
            if not proj_dir:
                proj_dir = getattr(self, "project_dir", "") or ""
            return Path(proj_dir).resolve()

        def _log_to_progress(cb, text: str) -> None:
            if callable(cb):
                try:
                    cb({"msg": text})
                except Exception:
                    pass

        def _ask_en(system_msg: str, user_msg: str) -> str:
            try:
                # self._ai 인스턴스가 AI 클래스라고 가정
                return str(self._ai.ask_smart(system_msg, user_msg, prefer="gemini", allow_fallback=True) or "")
            except Exception:
                return ""

        def _read_ui_fps() -> int:
            try:
                cmb = getattr(self, "cmb_movie_fps", None)
                return int(cmb.currentData()) if cmb and hasattr(cmb, "currentData") else 30
            except Exception:
                return 30

        def job(progress_cb):
            pdir = _resolve_project_dir()
            if not pdir.exists():
                raise FileNotFoundError("프로젝트 디렉터리를 찾을 수 없습니다.")

            _log_to_progress(progress_cb, f"[start] 프로젝트 분석 시작: {pdir.name}")

            # 1) story.json 스켈레톤 (기존 유지)
            spath = pdir / "story.json"
            sdoc = load_json(spath, {}) or {}
            if not isinstance(sdoc, dict):
                sdoc = {}
            save_json(spath, sdoc)

            # 2) video.json 생성/보강 (기존 유지; 갭 포함 생성 직후 저장)
            vpath = pdir / "video.json"
            vdoc = load_json(vpath, {}) or {}
            if not isinstance(vdoc, dict):
                vdoc = {}
            # paths 보강
            vpaths = vdoc.get("paths") or {}
            vpaths.setdefault("root", str(pdir))
            vpaths.setdefault("imgs_dir", "imgs")
            vdoc["paths"] = vpaths
            save_json(vpath, vdoc)  # 1차 저장(기존 흐름과 동일)

            # (선택) 음악/가사 분석
            if callable(sync_lyrics_with_whisper_pro):
                try:
                    pj = load_json(pdir / "project.json", {}) or {}
                    audio_path = str(pj.get("audio") or pj.get("music") or "")
                    lyrics_text = str(pj.get("lyrics_raw") or pj.get("lyrics") or "")
                    if audio_path and lyrics_text:
                        _log_to_progress(progress_cb, "[info] 음악/가사 싱크 분석(Whisper Pro) 시도...")
                        sync_lyrics_with_whisper_pro(audio_path, lyrics_text)
                        _log_to_progress(progress_cb, "[info] 음악/가사 싱크 분석 완료.")
                except Exception as e_sync:
                    _log_to_progress(progress_cb, f"[WARN] 음악/가사 싱크 분석 실패 (스킵): {e_sync}")
                    pass  # 실패해도 계속 진행

            # 3~5단계: (주석 처리된 기존 AI 강화/주입/가사 재주입 흐름)
            #  ... (이 부분은 사용자의 기존 코드대로 주석 처리된 상태를 유지합니다)
            _log_to_progress(progress_cb, "[info] 3~5단계 (AI 강화/주입/가사 재주입)는 현재 건너뜁니다.")

            # ── ★ FPS 동기화: 최종 저장 '직전' 보강 ─────────────────
            _log_to_progress(progress_cb, "[info] UI FPS 값을 video.json에 동기화합니다...")
            vdoc = load_json(vpath, {}) or {}
            if not isinstance(vdoc, dict):
                vdoc = {}

            ui_fps = _read_ui_fps()
            vdoc.setdefault("defaults", {})
            vdoc["defaults"].setdefault("movie", {})
            vdoc["defaults"]["movie"]["target_fps"] = ui_fps
            vdoc["defaults"]["movie"]["input_fps"] = ui_fps
            vdoc["defaults"]["movie"]["fps"] = ui_fps
            vdoc["defaults"].setdefault("image", {})
            vdoc["defaults"]["image"]["fps"] = ui_fps
            vdoc["fps"] = int(ui_fps)

            save_json(vpath, vdoc)  # ★ FPS 값이 포함된 저장
            _log_to_progress(progress_cb, f"[info] FPS ({ui_fps}) 동기화 완료.")

            # ── ★ 정크 분할 + 세그먼트 프롬프트 자동 보강 ───────────
            _log_to_progress(progress_cb, "[info] 씬 세그먼트 생성 및 AI 프롬프트 채우기를 시작합니다...")
            try:
                fill_prompt_movie_with_ai(
                    Path(pdir),  # 1. 프로젝트 경로
                    _ask_en,  # 2. AI 호출 함수
                    log_fn=lambda m: _log_to_progress(progress_cb, m),  # 3. 로그 함수
                )
                _log_to_progress(progress_cb, "[info] 세그먼트 프롬프트 작업 완료.")

            except Exception as e_fill:
                _log_to_progress(progress_cb, f"[WARN] fill_prompt_movie_with_ai 실패(스킵): {e_fill}")

            _log_to_progress(progress_cb, "[done] 완료")
            # vdoc이 최신 상태를 반영하도록 fps 값을 다시 읽어옴
            final_fps = vdoc.get("fps", ui_fps)
            return {"project_dir": str(pdir), "fps": final_fps}

        def done(ok: bool, payload, err):
            if not ok:
                QtWidgets.QMessageBox.critical(self, "분석 실패", str(err))
                return
            info = payload or {}
            QtWidgets.QMessageBox.information(
                self,
                "분석 완료",
                f"프로젝트 분석이 완료되었습니다.\n\n"
                f"- 프로젝트: {info.get('project_dir')}\n"
                f"- FPS: {info.get('fps')}\n"
                f"- (FPS 동기화 및 세그먼트 프롬프트 자동 채움이 수행되었습니다.)"
            )

        run_job_with_progress_async(
            owner=self,
            title="프로젝트 분석 중",
            job=job,
            on_done=done,
        )

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
       - For any character id that starts with "female", you MUST include "adult woman".
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

        jsons_dir_val = getattr(settings_mod, "JSONS_DIR", None)
        wfdir = Path(jsons_dir_val) if jsons_dir_val else (Path(__file__).parent / "jsons")

        self._dbg(f"[DOCS] start build | proj={proj} | wfdir={wfdir} exists={wfdir.exists()} | hair_map={hm}")

        def job():
            import inspect

            # 빌더 + 타임라인 적용 함수 임포트 (ImportError만)


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
    def _guess_project_dir(self) -> Path:
        """현재 프로젝트 폴더 추정: _current_project_dir() → FINAL_OUT → BASE_DIR/title"""
        # 안전 import (app 패키지/단독 실행 모두 고려)

        # 1) UI 제공 메서드 우선
        if hasattr(self, "_current_project_dir"):
            try:
                d = self._current_project_dir()
                if d:
                    return Path(d)
            except Exception:
                pass

        # 2) 제목 기반 추정
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
        return _settings.FFMPEG_EXE or "ffmpeg"


    @staticmethod
    def _build_clip_from_image(
            self,
            img: "str | Path",
            out_mp4: "str | Path",
            duration: float,
            fps: int = 16,
            width: int = 1080,
            height: int = 1920,
    ) -> bool:
        """
        단일 이미지로 MP4 클립을 만든다.
        - 시그니처 유지(호출부 호환).
        - duration(초) 동안 -loop 1 정지영상 → 고정 FPS로 인코딩.
        - yuv420p, SAR 1:1, CFR 보장(후속 xfade 등 필터 호환).
        """


        try:
            img_path = Path(img)
            out_path = Path(out_mp4)

            if not img_path.exists():
                return False

            out_path.parent.mkdir(parents=True, exist_ok=True)

            # ffmpeg 경로 확보
            try:

                ffmpeg_bin = getattr(settings_mod, "FFMPEG_EXE", "") or "ffmpeg"
            except Exception:
                ffmpeg_bin = "ffmpeg"

            dur_safe = max(0.01, float(duration))
            fps_safe = int(fps)
            w_safe = int(width)
            h_safe = int(height)

            cmd = [
                ffmpeg_bin, "-y",
                "-loop", "1",
                "-i", str(img_path),
                "-t", f"{dur_safe}",
                "-vf", f"scale={w_safe}:{h_safe},fps={fps_safe},setsar=1,format=yuv420p,setpts=PTS-STARTPTS",
                "-r", f"{fps_safe}",
                "-pix_fmt", "yuv420p",
                "-c:v", "libx264",
                "-movflags", "+faststart",
                str(out_path),
            ]

            # Windows 콘솔 인코딩 이슈를 피하려면 text=True로 캡처만 수행
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                return False

            return out_path.exists() and out_path.stat().st_size > 0

        except (OSError, ValueError):
            return False

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
        # self.te_prompt_pos.setPlainText("ace-step tag 추천해줘 : \n")
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




        # --- ▼▼▼ [신규] "huge" 체크박스 생성 (그룹박스 제거) ▼▼▼ ---
        self.chk_huge_breasts = QtWidgets.QCheckBox("huge breasts")
        self.chk_huge_breasts.setChecked(False)  # 기본값 False (체크 해제)
        self.chk_huge_breasts.setToolTip("체크 시 여성 캐릭터 프롬프트에 'huge breasts'를 강제로 추가합니다.")
        # --- ▲▲▲ [신규] 생성 끝 ▲▲▲ ---

        opts = QtWidgets.QHBoxLayout()
        opts.addStretch(1)
        opts.addWidget(self.chk_huge_breasts)
        opts.addStretch(1)

        # 버튼들
        self.btn_gen = QtWidgets.QPushButton("가사생성")
        self.btn_save = QtWidgets.QPushButton("프로젝트 저장")
        self.btn_load_proj = QtWidgets.QPushButton("프로젝트 불러오기")
        self.btn_music = QtWidgets.QPushButton("음악생성(ACE-Step)")
        self.btn_show_progress = QtWidgets.QPushButton("테스트")  # (생성하지만 레이아웃에 추가 안 함)
        self.btn_video = QtWidgets.QPushButton("영상생성(i2v)")
        self.btn_analyze = QtWidgets.QPushButton("음악분석")

        self.btn_test1_story = QtWidgets.QPushButton("프로젝트분석")
        self.btn_json_edit = QtWidgets.QPushButton("제이슨수정")  # <-- [신규] 버튼 생성
        self.btn_merging_videos = QtWidgets.QPushButton("영상합치기")
        self.btn_lyrics_in = QtWidgets.QPushButton("가사넣기")
        self.btn_missing_img = QtWidgets.QPushButton("누락 이미지 생성")

        # --- ▼▼▼ [이 코드 추가] ▼▼▼ ---
        self.btn_segments_img = QtWidgets.QPushButton("세그먼트 이미지 생성")
        self.btn_segments_img.setToolTip("2단계: video.json의 frame_segments 프롬프트로 Qwen I2I를 실행해 키프레임 이미지들을 생성합니다.")
        # --- ▲▲▲ [추가 끝] ▲▲▲ ---

        # --- ▼▼▼ 신규 매크로 버튼 생성 ▼▼▼ ---
        self.btn_macro_analyze = QtWidgets.QPushButton("분석")
        self.btn_macro_analyze.setToolTip("음악분석 -> 프로젝트분석을 순차적으로 실행합니다.")
        self.btn_macro_build_video = QtWidgets.QPushButton("영상만들기")
        self.btn_macro_build_video.setToolTip("영상생성(i2v) -> 영상합치기 -> 가사넣기를 순차적으로 실행합니다.")
        # --- ▲▲▲ 신규 매크로 버튼 생성 ▲▲▲ ---

        # --- [수정됨] 버튼 레이아웃 재배치 ---

        # 첫째 줄: 가사생성, 초기화, 프로젝트저장, 프로젝트불러오기, 음악생성, 분석(매크로), 누락 이미지 생성, 영상만들기(매크로)
        row = QtWidgets.QHBoxLayout()
        # --- ▼▼▼ [이 부분을 수정합니다] ▼▼▼ ---

        row.addWidget(self.btn_gen)
        self._add_clear_button_next_to_generate(row)  # "초기화" 버튼 추가
        row.addWidget(self.btn_save)
        row.addWidget(self.btn_load_proj)
        row.addWidget(self.btn_music)
        row.addSpacing(15)  # 구분선
        row.addWidget(self.btn_macro_analyze)  # <-- "분석" 매크로 추가
        row.addWidget(self.btn_missing_img)  # <-- "누락 이미지 생성" 이동

        row.addWidget(self.btn_segments_img)
        row.addWidget(self.btn_macro_build_video)  # <-- "영상만들기" 매크로 추가
        row.addStretch(1)  # 버튼을 왼쪽으로 정렬

        # 둘째 줄: 음악분석, 프로젝트분석, 영상생성, 영상합치기, 가사넣기
        row_test = QtWidgets.QHBoxLayout()
        row_test.addWidget(self.btn_analyze)
        row_test.addWidget(self.btn_test1_story)
        row_test.addWidget(self.btn_json_edit)  # <-- [신규] 버튼 레이아웃에 추가
        row_test.addWidget(self.btn_video)
        row_test.addWidget(self.btn_merging_videos)
        row_test.addWidget(self.btn_lyrics_in)
        row_test.addStretch(1)  # 버튼을 왼쪽으로 정렬

        # --- [수정 끝] ---

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
        #
        # --- [신규] 렌더/이미지/기타 설정 재배치 ---

        # 1. 모든 렌더/이미지 관련 위젯을 생성 (self.cmb_img_w, self.cmb_render_w 등을 생성)
        self._create_render_widgets()  # [수정] _add_render_prefs_controls -> _create_render_widgets

        # 2. 새로운 상단 가로줄 (이미지 설정 + 기타 설정)
        top_settings_row = QtWidgets.QHBoxLayout()

        # 2a. (왼쪽) "이미지 설정" 그룹 [요청 사항]
        grp_image = QtWidgets.QGroupBox("이미지 설정")
        layout_image = QtWidgets.QHBoxLayout(grp_image)
        layout_image.addWidget(QtWidgets.QLabel("W"))
        layout_image.addWidget(self.cmb_img_w)  # "이미지"용 W
        layout_image.addWidget(QtWidgets.QLabel("H"))
        layout_image.addWidget(self.cmb_img_h)  # "이미지"용 H
        layout_image.addSpacing(12)
        layout_image.addWidget(QtWidgets.QLabel("프리셋"))
        layout_image.addWidget(self.cmb_res_preset)  # "이미지"용 프리셋
        layout_image.addSpacing(12)
        layout_image.addWidget(QtWidgets.QLabel("스텝"))
        layout_image.addWidget(self.spn_t2i_steps)  # "이미지"용 스텝
        layout_image.addStretch(1)
        top_settings_row.addWidget(grp_image, 1)  # 1 stretch

        # 2b. (오른쪽) "기타 설정" 그룹 [요청 사항]
        grp_other = QtWidgets.QGroupBox("기타 설정")
        layout_other = QtWidgets.QHBoxLayout(grp_other)
        layout_other.addWidget(self.btn_ai_toggle)  # (기존 위치에서 이동)
        layout_other.addWidget(self.chk_huge_breasts)  # (기존 위치(opts)에서 이동)
        layout_other.addStretch(1)
        top_settings_row.addWidget(grp_other, 1)  # 1 stretch

        main_layout.addLayout(top_settings_row)  # [신규] 상단 줄 추가

        # 3. (아래쪽) "렌더 설정" 그룹 (기존 위젯 + 신규 복제 위젯)
        grp_render = QtWidgets.QGroupBox("렌더 설정")
        layout_render = QtWidgets.QHBoxLayout(grp_render)
        layout_render.addWidget(QtWidgets.QLabel("W"))
        layout_render.addWidget(self.cmb_render_w)  # "렌더"용 W
        layout_render.addWidget(QtWidgets.QLabel("H"))
        layout_render.addWidget(self.cmb_render_h)  # "렌더"용 H
        layout_render.addSpacing(12)
        layout_render.addWidget(QtWidgets.QLabel("FPS"))
        layout_render.addWidget(self.cmb_movie_fps)  # "렌더"용 FPS (공통)
        layout_render.addSpacing(12)
        layout_render.addWidget(QtWidgets.QLabel("프리셋"))
        layout_render.addWidget(self.cmb_render_preset)  # "렌더"용 프리셋
        layout_render.addSpacing(12)
        layout_render.addWidget(QtWidgets.QLabel("스텝"))
        layout_render.addWidget(self.spn_render_steps)  # "렌더"용 스텝
        layout_render.addSpacing(12)
        layout_render.addWidget(self.cmb_font)  # "렌더"용 폰트 (공통)
        layout_render.addSpacing(10)
        layout_render.addWidget(QtWidgets.QLabel("제목크기:"))
        layout_render.addWidget(self.spn_title_font_size)  # (공통)
        layout_render.addWidget(QtWidgets.QLabel("가사크기:"))
        layout_render.addWidget(self.spn_lyric_font_size)  # (공통)
        layout_render.addStretch(1)

        main_layout.addWidget(grp_render)  # [신규] 렌더 설정 그룹 추가
        # --- [신규] 재배치 끝 ---
        #
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

        tab = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout()

        # BASE_DIR
        self.le_base_dir = QtWidgets.QLineEdit(str(settings_mod.BASE_DIR))
        btn_pick_base = QtWidgets.QPushButton("폴더 선택")

        def _pick_base():
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "BASE_DIR 선택", str(settings_mod.BASE_DIR))
            if d:
                self.le_base_dir.setText(d)

        btn_pick_base.clicked.connect(_pick_base)
        base_wrap = QtWidgets.QHBoxLayout()
        base_wrap.addWidget(self.le_base_dir)
        base_wrap.addWidget(btn_pick_base)
        base_widget = QtWidgets.QWidget()
        base_widget.setLayout(base_wrap)

        # COMFY_HOST & 후보
        self.le_comfy = QtWidgets.QLineEdit(settings_mod.COMFY_HOST)
        self.te_candidates = QtWidgets.QPlainTextEdit("\n".join(settings_mod.DEFAULT_HOST_CANDIDATES))

        # ffmpeg / hwaccel / 출력파일 / 오디오 포맷
        self.le_ffmpeg = QtWidgets.QLineEdit(settings_mod.FFMPEG_EXE)
        self.cb_hwaccel = QtWidgets.QCheckBox("USE_HWACCEL")
        self.cb_hwaccel.setChecked(bool(settings_mod.USE_HWACCEL))
        self.le_final = QtWidgets.QLineEdit(settings_mod.FINAL_OUT)

        self.cb_audio_fmt = QtWidgets.QComboBox()
        self.cb_audio_fmt.addItems(["mp3", "wav", "opus"])
        ov = settings_mod.load_overrides() or {}
        cur_fmt = str(ov.get("AUDIO_SAVE_FORMAT", settings_mod.AUDIO_SAVE_FORMAT)).lower()
        idx_fmt = max(0, self.cb_audio_fmt.findText(cur_fmt))
        self.cb_audio_fmt.setCurrentIndex(idx_fmt)

        # 프롬프트/워크플로 파일 경로
        self.le_prompt_json = QtWidgets.QLineEdit(str(settings_mod.ACE_STEP_PROMPT_JSON))
        btn_pick_prompt = QtWidgets.QPushButton("파일 선택")

        def _pick_prompt():
            f, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "ACE_STEP_PROMPT_JSON", str(settings_mod.JSONS_DIR), "JSON (*.json)"
            )
            if f:
                self.le_prompt_json.setText(f)

        btn_pick_prompt.clicked.connect(_pick_prompt)
        pj_wrap = QtWidgets.QHBoxLayout()
        pj_wrap.addWidget(self.le_prompt_json)
        pj_wrap.addWidget(btn_pick_prompt)
        pj_widget = QtWidgets.QWidget()
        pj_widget.setLayout(pj_wrap)

        self.le_i2v = QtWidgets.QLineEdit(str(settings_mod.I2V_WORKFLOW))
        btn_pick_i2v = QtWidgets.QPushButton("파일 선택")

        def _pick_i2v():
            f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "I2V_WORKFLOW", str(settings_mod.JSONS_DIR), "JSON (*.json)")
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
        # ↓ 여기 4줄(DEFAULT_...)은 삭제
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
                ACE_STEP_PROMPT_JSON=self.le_prompt_json.text().strip(),
                I2V_WORKFLOW=self.le_i2v.text().strip(),
            )

        def _do_save():
            path = settings_mod.save_overrides(**_collect_overrides())
            QtWidgets.QMessageBox.information(self, "저장 완료", f"settings_local.json 저장됨\n\n{path}")

        def _do_apply():
            overrides = _collect_overrides()
            settings_mod.save_overrides(**overrides)


            settings_mod.BASE_DIR = overrides.get("BASE_DIR", settings_mod.BASE_DIR)
            settings_mod.COMFY_HOST = overrides.get("COMFY_HOST", settings_mod.COMFY_HOST)
            settings_mod.DEFAULT_HOST_CANDIDATES = overrides.get(
                "DEFAULT_HOST_CANDIDATES", getattr(settings_mod, "DEFAULT_HOST_CANDIDATES", [])
            )
            settings_mod.FFMPEG_EXE = overrides.get("FFMPEG_EXE", settings_mod.FFMPEG_EXE)
            settings_mod.USE_HWACCEL = bool(overrides.get("USE_HWACCEL", getattr(settings_mod, "USE_HWACCEL", False)))
            settings_mod.FINAL_OUT = overrides.get("FINAL_OUT", settings_mod.FINAL_OUT)
            settings_mod.AUDIO_SAVE_FORMAT = overrides.get("AUDIO_SAVE_FORMAT", getattr(settings_mod, "AUDIO_SAVE_FORMAT", "mp3")).lower()
            settings_mod.ACE_STEP_PROMPT_JSON = overrides.get("ACE_STEP_PROMPT_JSON", settings_mod.ACE_STEP_PROMPT_JSON)
            settings_mod.I2V_WORKFLOW = overrides.get("I2V_WORKFLOW", settings_mod.I2V_WORKFLOW)

            # 나머지 워크플로 패치 부분은 그대로 두면 됨
            # (필요 없으면 여기서도 지워도 되고)

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
        # --- ▼▼▼ [이 부분을 수정합니다] ▼▼▼ ---
        self.btn_video.clicked.connect(self.on_video) # <-- 기존 연결 주석 처리
        # self.btn_video.clicked.connect(self.on_video_wan)  # <-- [신규] 3단계 함수로 교체

        # [신규] 2단계 버튼 연결
        if hasattr(self, "btn_segments_img"):  # 버튼이 생성되었는지 확인
            self.btn_segments_img.clicked.connect(self.on_click_segments_missing_images_with_log)
        # --- ▲▲▲ [수정 끝] ▲▲▲ ---


        # 라디오 ↔ JSON 즉시 동기화 (toggled True일 때만 저장)
        self.rb_20s.toggled.connect(lambda on: on and self._on_seconds_changed(20))
        self.rb_1m.toggled.connect(lambda on: on and self._on_seconds_changed(60))
        self.rb_2m.toggled.connect(lambda on: on and self._on_seconds_changed(120))
        self.rb_3m.toggled.connect(lambda on: on and self._on_seconds_changed(180))

        # 테스트
        self.btn_json_edit.clicked.connect(self.on_click_edit_json)  # <-- [신규] 시그널 연결
        self.btn_merging_videos.clicked.connect(self.merging_videos_start)
        self.btn_lyrics_in.clicked.connect(self.lyrics_in_start)

        # --- ▼▼▼ 신규 매크로 버튼 연결 ▼▼▼ ---
        self.btn_macro_analyze.clicked.connect(self.on_click_macro_analyze)
        self.btn_macro_build_video.clicked.connect(self.on_click_macro_build_video)
        # --- ▲▲▲ 신규 매크로 버튼 연결 ▲▲▲ ---

    # ────────────── 토글/태그 유틸 ──────────────
    def _on_tags_changed(self, *_args) -> None:
        """
        태그(자동/수동, 보컬 성별, 개별 체크박스) 변경 시 즉시 project.json에 반영.
        - project.json이 없으면 아무 것도 하지 않음(가사 생성 단계에서 저장됨)
        - 자동 모드: ace_tags(최근 제안)와 tags_in_use(보조선택) 반영
        - 수동 모드: manual_tags만 반영
        - seconds/target_seconds 등 다른 필드는 건드리지 않음
        """
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

        try:

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


    def _apply_project_meta(self, proj_dir: str) -> None:
        """
        project.json과 video.json을 읽어 UI에 반영.
        - [수정됨] video.json의 FPS 값을 우선적으로 UI에 반영.
        - [수정됨] ui_prefs 복원 시 findData(int)를 사용하여 정확한 인덱스 복원.
        """



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

        # 프롬프트(긍정) UI 반영
        prompt_val = (meta.get("prompt_user") or meta.get("prompt") or "").strip()
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

        # --- ▼▼▼ [수정된 FPS 로드 로직] ▼▼▼ ---

        # 1. video.json 로드 시도
        vj_path = Path(proj_dir) / "video.json"
        vmeta = load_json(vj_path, {}) if vj_path.exists() else {}
        if not isinstance(vmeta, dict): vmeta = {}

        # 2. video.json에서 FPS 값 탐색 (우선순위)
        vj_fps_val = None
        try:
            defaults = vmeta.get("defaults", {})
            movie_defaults = defaults.get("movie", {})
            image_defaults = defaults.get("image", {})

            candidates = [
                movie_defaults.get("target_fps"),
                vmeta.get("fps"),
                movie_defaults.get("fps"),
                image_defaults.get("fps")
            ]
            for cand in candidates:
                if cand is not None:
                    vj_fps_val = int(cand)
                    break
        except Exception:
            vj_fps_val = None

        # 3. project.json (ui_prefs)에서 FPS 값 탐색 (폴백)
        ui_prefs = meta.get("ui_prefs") or {}
        pj_fps_val = None
        if ui_prefs:
            try:
                pj_fps_val = int(ui_prefs.get("movie_fps", 30))
            except Exception:
                pj_fps_val = 30

        # 4. 최종 FPS 결정 (video.json 우선)
        final_fps_to_set = vj_fps_val if vj_fps_val is not None else (pj_fps_val if pj_fps_val is not None else 30)

        # 5. ComboBox 복원 헬퍼 (데이터 값(int)으로 인덱스 찾기)
        def _set_combo_by_data(combo, data_val):
            # 콤보박스가 로드되기 전일 수 있으므로 방어
            if not hasattr(self, "cmb_movie_fps"):
                return
            idx = combo.findData(int(data_val))
            if idx >= 0:
                combo.setCurrentIndex(idx)

        # 6. UI에 최종 FPS 값 설정
        try:
            _set_combo_by_data(self.cmb_movie_fps, final_fps_to_set)
        except Exception as e_set_fps:
            print(f"[UI] FPS 콤보박스 설정 실패: {e_set_fps}")

        # --- ▲▲▲ [수정된 FPS 로직] 끝 ▲▲▲ ---

        # --- ▼▼▼ [기존 로직] 나머지 ui_prefs (W, H, Steps 등)는 project.json 값 그대로 사용 ▼▼▼ ---
        if ui_prefs:
            try:
                # 1. 이미지 설정 값 로드 (int로 강제)
                img_w, img_h = ui_prefs.get("image_size", [0, 0])
                img_w_val, img_h_val = int(img_w), int(img_h)
                img_steps = int(ui_prefs.get("image_steps", 28))
                img_preset_key = ui_prefs.get("image_preset", "custom")

                # 2. 렌더 설정 값 로드 (int로 강제)
                render_w, render_h = ui_prefs.get("render_size", [0, 0])
                render_w_val, render_h_val = int(render_w), int(render_h)
                # render_fps = int(ui_prefs.get("movie_fps", 30)) # <-- 이 줄은 위(수정된 로직)에서 처리됨
                render_steps = int(ui_prefs.get("render_steps", 28))
                render_preset_key = ui_prefs.get("render_preset", "custom")

                # ComboBox 복원 헬퍼 (데이터 값(int)으로 인덱스 찾기)
                # (위에서 이미 정의했지만, 만약을 위해 여기서도 정의)
                def _set_combo_by_data(combo, data_val):
                    if not hasattr(combo, "findData"): return
                    idx = combo.findData(int(data_val))
                    if idx >= 0:
                        combo.setCurrentIndex(idx)

                # 콤보박스 값 복원 (FPS 제외)
                _set_combo_by_data(self.cmb_img_w, img_w_val)
                _set_combo_by_data(self.cmb_img_h, img_h_val)
                self.spn_t2i_steps.setValue(img_steps)

                _set_combo_by_data(self.cmb_render_w, render_w_val)
                _set_combo_by_data(self.cmb_render_h, render_h_val)
                # _set_combo_by_data(self.cmb_movie_fps, render_fps) # <-- 이 줄은 위(수정된 로직)에서 처리됨
                self.spn_render_steps.setValue(render_steps)

                # 프리셋 복원 (핸들러 호출을 위한 emit)
                def _set_preset_by_key(combo, key):
                    if not hasattr(combo, "findData"): return
                    index = combo.findData(key)
                    if index < 0:
                        for i in range(combo.count()):
                            data = combo.itemData(i)
                            if isinstance(data, tuple) and len(data) == 3 and data[2] == key:
                                index = i
                                break
                    if index >= 0:
                        combo.setCurrentIndex(index)
                        # 프리셋 핸들러를 호출하여 W/H 콤보박스의 잠금 상태를 업데이트
                        try:
                            handler = getattr(combo, "currentIndexChanged")
                            if handler and hasattr(handler, "emit"): handler.emit(index)
                        except Exception:
                            pass

                _set_preset_by_key(self.cmb_res_preset, img_preset_key)
                _set_preset_by_key(self.cmb_render_preset, render_preset_key)

                # 폰트
                font_family = ui_prefs.get("font_family", "굴림")
                self.cmb_font.setCurrentFont(QFont(font_family))

                # 폰트 크기
                title_size = ui_prefs.get("title_font_size", 55)
                lyric_size = ui_prefs.get("lyric_font_size", 25)
                self.spn_title_font_size.setValue(int(title_size))
                self.spn_lyric_font_size.setValue(int(lyric_size))

            except Exception as e:
                print(f"[UI] 렌더 설정 불러오기 실패: {e}")
        # --- ▲▲▲ [기존 로직] 끝 ▲▲▲ ---

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

        # --- ▼▼▼ [신규] 활성 프로젝트 경로 갱신 ▼▼▼ ---
        self._set_active_project_dir(str(pdir))
        # --- ▲▲▲ [신규] 추가 끝 ▲▲▲ ---

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
        """
        프로젝트 불러오기(최소 수정):
          - 초기 열기 경로: BASE_DIR/maked_title (없으면 생성)
          - 기존 동작(활성 프로젝트 전환, UI 복원, 상태표시, 플래그) 그대로 유지
        """

        # ── BASE_DIR 로딩 및 시작 폴더 계산 ─────────────────────────
        # BASE_DIR 우선순위: app.settings → settings → cwd

        _base_dir_text = str(getattr(settings_mod, "BASE_DIR", Path.cwd()))


        _base_dir_path = Path(_base_dir_text)
        # maked_title 중복 방지: BASE_DIR가 이미 maked_title이면 그대로, 아니면 하위에 생성
        _start_dir_path = _base_dir_path if _base_dir_path.name.lower() == "maked_title" else (
                    _base_dir_path / "maked_title")
        try:
            _start_dir_path.mkdir(parents=True, exist_ok=True)
        except OSError:
            # 폴더 생성 실패해도 파일 대화상자는 동작 가능하므로 무시
            pass

        # 사용자가 이전에 열었던 프로젝트 폴더가 있으면 그 폴더를 우선 시작 경로로 사용
        try:
            _prev_proj_dir = str(getattr(self, "project_dir", "") or "").strip()
        except Exception:
            _prev_proj_dir = ""
        if _prev_proj_dir:
            _initial_dir_text = _prev_proj_dir
        else:
            _initial_dir_text = str(_start_dir_path)

        # (참고) 원래 코드에 있던 load_json 임포트는 기능상 사용되지 않아 생략해도 동작 동일

        # 1) project.json 선택 (초기 경로만 바뀜)
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "project.json 선택",
            _initial_dir_text,
            "JSON (*.json)"
        )
        if not path_str:
            return

        pj = Path(path_str)
        pdir = pj.parent

        # 2) 선택 즉시 '활성 프로젝트'로 전환 (원래 흐름 유지)
        try:
            self._set_active_project_dir(str(pdir))
        except Exception:
            # 기존 코드가 예외 전파하지 않았으므로 동일하게 조용히 무시
            pass

        # 3) UI 복원(제목/가사/태그/길이 라디오까지) — 기존 함수 호출 유지
        try:
            self._apply_project_meta(str(pdir))
        except Exception:
            pass

        # 4) 상태 표시 (원래 출력 유지)
        print("[LOAD-PROJ] activated:", str(pdir), flush=True)
        try:
            if hasattr(self, "statusbar"):
                self.statusbar.showMessage(f"불러옴: {pj}")
        except Exception:
            pass

        # ✅ 프로젝트 컨텍스트 준비 완료 플래그 (원래 플래그 유지)
        try:
            setattr(self, "_project_context_ready", True)
        except Exception:
            pass


    # --- [신규] JSON 편집 버튼 핸들러 ---
    def on_click_edit_json(self):
        """
        '제이슨수정' 버튼 클릭 핸들러.
        [수정 v33] 5~7초 딜레이 문제를 해결하기 위해,
        QApplication.processEvents()를 '즉시' 호출하여 버튼 비활성화를 강제 실행.
        """

        btn_to_disable = getattr(self, "btn_json_edit", None)
        if not btn_to_disable:
            print("[ERROR] btn_json_edit 위젯을 찾을 수 없습니다.")
            return

        # --- [신규] 1. 버튼 비활성화 '요청' ---
        btn_to_disable.setEnabled(False)

        # --- [신규] 2. GUI 이벤트 루프 강제 실행 ---
        # (비활성화된 버튼을 '즉시' 그리도록 강제)
        QtWidgets.QApplication.processEvents()

        proj_dir = None
        try:
            # --- (여기서 1~2초 딜레이 발생 가능) ---
            proj_dir = self._current_project_dir()
        except Exception as e:
            print(f"[UI] JSON 편집: 프로젝트 디렉터리를 가져오는 중 오류: {e}")

        if not proj_dir:
            QtWidgets.QMessageBox.warning(self, "오류", "먼저 프로젝트를 불러오거나 생성해주세요.")
            btn_to_disable.setEnabled(True)  # [수정] 오류 시 버튼 복구
            return

        video_json_path = Path(proj_dir) / "video.json"

        if not video_json_path.exists():
            QtWidgets.QMessageBox.warning(self, "오류",
                                          f"video.json 파일을 찾을 수 없습니다.\n\n"
                                          f"경로: {video_json_path}")
            btn_to_disable.setEnabled(True)  # [수정] 오류 시 버튼 복구
            return

        try:
            # --- 3. (여기서 5~7초 멈춤) ---
            # 이제 버튼이 비활성화된 상태로 멈춥니다.
            dialog = ScenePromptEditDialog(video_json_path, self._ai, self)

            # --- 4. 다이얼로그 실행 ---
            dialog.exec_()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "편집기 오류", f"JSON 편집기를 여는 중 오류가 발생했습니다:\n{e}")

        finally:
            # --- 5. 다이얼로그가 닫히면(finally) 버튼을 다시 활성화 ---
            if btn_to_disable:
                btn_to_disable.setEnabled(True)
    # ────────────── 진행창/로그 ──────────────

    def on_show_progress(self) -> None:
        """
        [테스트] xfade 병합(5개 청크 고정) – 업로드된 _chunk_gap_002_0000{0..4}.mp4 메타를 읽어
        공통 fps/해상도(가장 많이 등장하는 해상도)를 선택해 정규화 후 병합한다.
        결과: clips/test_merge.mp4
        """


        # 프로젝트/폴더 경로
        try:
            proj_dir_str = self._current_project_dir()
        except Exception:
            proj_dir_str = None
        if not proj_dir_str:
            QtWidgets.QMessageBox.warning(self, "테스트", "프로젝트 폴더를 찾을 수 없습니다.")
            return

        proj_dir = Path(proj_dir_str)
        clips_dir = proj_dir / "clips"
        work_dir = clips_dir / "xfade_work"
        work_dir.mkdir(parents=True, exist_ok=True)

        # 업로드된 5개 파일(고정 이름) – xfade_work에 있어야 함
        file_names = [
            "_chunk_t_001_00000.mp4",
            "_chunk_t_001_00001.mp4",
            "_chunk_t_001_00002.mp4",
        ]
        clip_paths: List[Path] = [work_dir / name for name in file_names]
        missing_paths = [p for p in clip_paths if not p.exists()]
        if missing_paths:
            QtWidgets.QMessageBox.warning(
                self,
                "테스트",
                "다음 파일이 없습니다:\n" + "\n".join(str(p) for p in missing_paths),
            )
            return

        # ffprobe 헬퍼: fps/해상도 등 추출
        def _ffprobe_meta(video_path: Path) -> Dict[str, object]:
            meta: Dict[str, object] = {
                "path": str(video_path),
                "width": None,
                "height": None,
                "r_frame_rate": None,
                "avg_frame_rate": None,
                "nb_frames": None,
                "duration_sec": None,
            }
            cmd = [
                (globals().get("FFPROBE_EXE", "ffprobe")),
                "-v", "error",
                "-print_format", "json",
                "-show_streams",
                "-show_format",
                str(video_path),
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
                meta["ffprobe_out"] = proc.stdout
                return meta
            try:
                parsed = json.loads(proc.stdout)
            except json.JSONDecodeError:
                meta["ffprobe_out"] = proc.stdout
                return meta

            vstreams = [s for s in parsed.get("streams", []) if s.get("codec_type") == "video"]
            if vstreams:
                vs = vstreams[0]
                meta["width"] = vs.get("width")
                meta["height"] = vs.get("height")
                meta["r_frame_rate"] = vs.get("r_frame_rate")
                meta["avg_frame_rate"] = vs.get("avg_frame_rate")
                meta["nb_frames"] = vs.get("nb_frames")
                tb = parsed.get("format", {}).get("duration")
                if isinstance(tb, str):
                    try:
                        meta["duration_sec"] = float(tb)
                    except ValueError:
                        pass
            return meta

        def _parse_rate(rate_str: Optional[str]) -> float:
            if not rate_str:
                return 0.0
            s = str(rate_str)
            if "/" in s:
                num, den = s.split("/", 1)
                try:
                    n = float(num)
                    d = float(den) if float(den) != 0 else 1.0
                    return n / d
                except ValueError:
                    return 0.0
            try:
                return float(s)
            except ValueError:
                return 0.0

        # 5개 파일 메타 수집
        metas: List[Dict[str, object]] = []
        for vpath in clip_paths:
            m = _ffprobe_meta(vpath)
            print("[TEST] META:", m)
            metas.append(m)

        # 공통 fps: 다수결(동률이면 첫 파일 기준), 소수점은 반올림 정수
        fps_candidates: List[int] = []
        for m in metas:
            fps_val = 0.0
            if m.get("avg_frame_rate"):
                fps_val = _parse_rate(m.get("avg_frame_rate"))  # type: ignore[arg-type]
            if not fps_val and m.get("r_frame_rate"):
                fps_val = _parse_rate(m.get("r_frame_rate"))  # type: ignore[arg-type]
            fps_int = int(round(fps_val)) if fps_val > 0 else 0
            fps_candidates.append(max(fps_int, 0))
        # 모드 선택
        fps_mode = None
        if fps_candidates:
            counts: Dict[int, int] = {}
            for f in fps_candidates:
                counts[f] = counts.get(f, 0) + 1
            fps_mode = max(counts.items(), key=lambda kv: kv[1])[0]
        if not fps_mode or fps_mode <= 0:
            # 합의 실패 시 첫 파일 또는 30
            first_f = fps_candidates[0] if fps_candidates and fps_candidates[0] > 0 else 30
            fps_mode = first_f if first_f > 0 else 30

        # 공통 해상도: (width,height) 다수결, 동률이면 첫 파일 해상도
        wh_list: List[Optional[tuple]] = []
        for m in metas:
            w = m.get("width")
            h = m.get("height")
            if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
                wh_list.append((w, h))
            else:
                wh_list.append(None)
        wh_counts: Dict[tuple, int] = {}
        for wh in wh_list:
            if wh is None:
                continue
            wh_counts[wh] = wh_counts.get(wh, 0) + 1
        if wh_counts:
            out_w, out_h = max(wh_counts.items(), key=lambda kv: kv[1])[0]
        else:
            # 메타가 비정상일 때 기본값(세로 예시: 720x1080)
            out_w, out_h = 720, 1080

        print(f"[TEST] SELECTED: fps={fps_mode}, size={out_w}x{out_h}")

        # 병합 실행
        out_path = clips_dir / "test_merge.mp4"
        try:
            xfade_concat(
                clip_paths=clip_paths,
                overlap_frames=12,  # 업로드 분석 시 테스트했던 기본값 그대로
                fps=fps_mode,
                audio_path=None,
                out_path=out_path,
                out_fps=None,
                scale_w=out_w,
                scale_h=out_h,
                work_dir=work_dir,
            )
            print("[TEST] DONE:", out_path)
            QtWidgets.QMessageBox.information(self, "테스트 완료", f"출력: {out_path}")
        except (OSError, ValueError, RuntimeError) as e_merge:
            print("[TEST][ERROR]", e_merge)
            QtWidgets.QMessageBox.warning(self, "테스트 실패", str(e_merge))

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
    def _on_music_done(self, success: bool, msg: str) -> None:
        """
        ACE-Step 완료 이후:
          1) 진행 로그/타이머 정리
          2) 생성된 오디오를 프로젝트 폴더로 이동(최신본 1개)
          3) 이동 결과 상태 표시 및 총 프레임 갱신
          4) 바로 story.json 생성(가능할 때만)
        (기존 기능 보존, 파라미터/경로 안전성만 보강)
        """
        # 진행창 메시지
        if getattr(self, "_dlg", None):
            self._dlg.set_status("ACE-Step 완료 — 파일 정리 중…")
            self._dlg.append_log(f"ACE-Step 완료 (성공: {bool(success)}) ✅")
            if msg:
                self._dlg.append_log(str(msg))

        # 로그 테일 정리 (창은 유지)
        try:
            if getattr(self, "_log_timer", None):
                self._log_timer.stop()
                self._log_timer.deleteLater()
                self._log_timer = None
            if getattr(self, "_log_fp", None):
                self._log_fp.close()
                self._log_fp = None
        except (AttributeError, OSError):
            pass

        # ── 이동 대상 결정: title 또는 프로젝트 폴더(Path) ──

        title_text = ""
        try:
            if hasattr(self, "le_title") and self.le_title is not None:
                title_text = (self.le_title.text() or "").strip()
        except Exception:
            title_text = ""

        proj_dir: Path | None = None
        if not title_text:
            try:
                proj_dir = self._latest_project()
            except Exception:
                proj_dir = None
            if proj_dir and (proj_dir / "project.json").exists():
                try:
                    meta = _load_json(proj_dir / "project.json", {}) or {}
                    title_text = str(meta.get("title", "")).strip()
                except Exception:
                    title_text = ""

        # ── 실제 이동 시도 ──
        moved_paths: list[Path] = []
        try:
            # 제목이 있으면 제목으로, 없으면 프로젝트 폴더(Path)로 이동 함수 호출
            target_hint: str | Path = title_text if title_text else (proj_dir or Path(""))
            if str(target_hint):
                moved_paths.extend(self._move_generated_audio_to_target(target_hint) or [])
            # vocal.wav 최신본 별도 보강(기존 함수 호출 보존)
            try:
                one_more = self._move_latest_vocal_to_project(title_text or (proj_dir.name if proj_dir else ""))
                if one_more:
                    moved_paths.append(one_more)
            except Exception:
                # 선택적 경로이므로 조용히 패스
                pass
        except Exception as ex_move:
            try:
                self.status.showMessage(f"파일 이동 실패: {ex_move}")
            except Exception:
                pass
            if getattr(self, "_dlg", None):
                self._dlg.append_log(f"[MOVE] 실패: {ex_move}")

        # ── 상태 표시 및 총 프레임 산출 ──
        if moved_paths:
            last_dst = moved_paths[-1]
            try:
                self.status.showMessage(f"음악 생성 완료 — {last_dst.name}")
            except Exception:
                pass
            if getattr(self, "_dlg", None):
                self._dlg.append_log(f"[MOVE] 완료: {last_dst}")
            try:
                self._set_total_frames_from_audio(last_dst)
            except Exception:
                pass
        else:
            try:
                self.status.showMessage("음악 생성 완료 — 이동된 파일 없음")
            except Exception:
                pass
            if getattr(self, "_dlg", None):
                self._dlg.append_log("[MOVE] 이동된 파일 없음")

        # 음악 가드 해제
        try:
            self._task_done("music")
        except Exception:
            pass

        # ── story.json 빌드용 분석 오디오 선택 ──
        audio_for_analysis: Path | None = None
        try:
            proj2 = self._latest_project()
        except Exception:
            proj2 = None

        try:
            meta2 = _load_json((proj2 / "project.json") if proj2 else "", {}) or {}
            vud = (meta2.get("paths", {}) or {}).get("vocal_user_dir", "")
            if vud:
                audio_for_analysis = self._resolve_audio_for_analysis(Path(vud))
        except Exception:
            audio_for_analysis = None

        if not audio_for_analysis:
            try:
                audio_for_analysis = self._resolve_audio_for_analysis(
                    self._find_vocal_in_project(proj2) if proj2 else None)
            except Exception:
                audio_for_analysis = None

        if (not audio_for_analysis) and moved_paths:
            audio_for_analysis = self._resolve_audio_for_analysis(moved_paths[-1])

        # ── 바로 story 빌드 (팝업 없음) ──
        proj_dir_final = audio_for_analysis.parent if audio_for_analysis else None
        pj_path = (proj_dir_final / "project.json") if proj_dir_final else None

        if audio_for_analysis and audio_for_analysis.exists() and pj_path and pj_path.exists():
            if getattr(self, "_dlg", None):
                self._dlg.set_status("음악 완료 → story.json 생성 시작…")
                self._dlg.append_log(f"[AUTO-STORY] 시작: {audio_for_analysis}")
            try:
                self._start_build_story_from_analysis(audio_for_analysis, pj_path)
            except Exception:
                pass
        else:
            print("[AUTO-STORY] skip: audio or project.json missing", flush=True)

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
                "btn_test1_story", "btn_merging_videos", "btn_lyrics_in",
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
                workflow_dir=getattr(settings_mod, "JSONS_DIR", None),
                on_progress=_onp,
            )

            # 인트로/아웃트로 타임라인 10% 반영

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



    @staticmethod
    def _move_generated_audio_to_target(self, title_or_project: "str | Path | None" = None) -> "list[Path]":
        """
        ComfyUI가 만든 최신 오디오 결과물(mp3/wav)을 프로젝트 폴더로 이동.
        - 입력: 프로젝트 제목(str), 프로젝트 폴더(Path), 또는 None(자동 추정)
        - 출력: 실제로 이동된 파일들의 경로 리스트
        - shutil 예외 참조 경고 방지 위해 함수 상단에서 import.
        - 기존 기능 보존: 제목 기반 경로/FINAL_OUT 템플릿/BASE_DIR 폴백 유지, 최신 1개만 이동.
        """

        # ---- 프로젝트 폴더 결정 ----
        proj_dir: Path | None = None

        # 0) 호출 인자가 Path
        if isinstance(title_or_project, Path):
            proj_dir = title_or_project

        # 1) 호출 인자가 str(제목)
        if proj_dir is None and isinstance(title_or_project, str) and title_or_project.strip():
            title_clean = _sanitize(title_or_project.strip())

            final_tpl = getattr(settings_mod, "FINAL_OUT", "")
            if final_tpl and "[title]" in final_tpl:
                proj_dir = Path(final_tpl.replace("[title]", title_clean))
            else:
                base_dir_val = getattr(settings_mod, "BASE_DIR", ".")
                proj_dir = Path(base_dir_val) / title_clean

        # 2) 인자가 None이거나 빈 문자열 → UI/메타에서 추정
        if proj_dir is None:
            # 2-1) 최신 프로젝트 폴더 시도
            try:
                latest_dir = self._latest_project()
                if latest_dir:
                    proj_dir = Path(latest_dir)
            except Exception:
                proj_dir = None

        if proj_dir is None:
            # 2-2) project.json의 title 기반 경로 시도
            try:
                latest_dir2 = self._latest_project()
                pj = (Path(latest_dir2) / "project.json") if latest_dir2 else None
                meta = _load_json(pj, {}) if pj else {}
                title_from_meta = str((meta.get("title", "") or "")).strip()
                if title_from_meta:
                    title_clean2 = _sanitize(title_from_meta)
                    final_tpl = getattr(settings_mod, "FINAL_OUT", "")
                    if final_tpl and "[title]" in final_tpl:
                        proj_dir = Path(final_tpl.replace("[title]", title_clean2))
                    else:
                        base_dir_val = getattr(settings_mod, "BASE_DIR", ".")
                        proj_dir = Path(base_dir_val) / title_clean2
            except Exception:
                proj_dir = None

        if proj_dir is None:
            # 2-3) UI의 제목 위젯
            title_txt = ""
            try:
                if hasattr(self, "le_title") and self.le_title is not None:
                    title_txt = (self.le_title.text() or "").strip()
            except Exception:
                title_txt = ""
            if title_txt:
                title_clean3 = _sanitize(title_txt)
                final_tpl = getattr(settings_mod, "FINAL_OUT", "")
                if final_tpl and "[title]" in final_tpl:
                    proj_dir = Path(final_tpl.replace("[title]", title_clean3))
                else:
                    base_dir_val = getattr(settings_mod, "BASE_DIR", ".")
                    proj_dir = Path(base_dir_val) / title_clean3

        if proj_dir is None:
            # 2-4) 최종 폴백: BASE_DIR/무제
            base_dir_val = getattr(settings_mod, "BASE_DIR", ".")
            proj_dir = Path(base_dir_val) / "무제"

        proj_dir = Path(proj_dir)
        proj_dir.mkdir(parents=True, exist_ok=True)

        moved: list[Path] = []

        # ---- 소스 후보 폴더 구성 (안전한 범용 탐색) ----
        guess_roots: list[Path] = []
        for key in ("COMFY_RESULT_ROOT", "AUDIO_OUT_DIR", "RESULT_DIR", "TEMP_DIR"):
            val = getattr(settings_mod, key, "")
            if isinstance(val, str) and val:
                guess_roots.append(Path(val))

        # 프로젝트 폴더 및 인접 폴더도 스캔
        guess_roots.extend([proj_dir, proj_dir.parent])

        # 중복 제거 & 존재하는 폴더만
        uniq_roots: list[Path] = []
        seen = set()
        for r in guess_roots:
            try:
                rp = r.resolve()
            except Exception:
                continue
            if not rp.exists():
                continue
            if rp in seen:
                continue
            seen.add(rp)
            uniq_roots.append(rp)

        # ---- 최신 오디오 파일 선택 ----
        candidates: list[Path] = []
        patterns = ("*.wav", "*.mp3", "*vocal*.wav", "*vocal*.mp3")
        for root in uniq_roots:
            try:
                for pat in patterns:
                    for p in root.rglob(pat):
                        try:
                            # 24시간 이내 파일 우선
                            if time.time() - p.stat().st_mtime <= 60 * 60 * 24:
                                candidates.append(p)
                        except Exception:
                            continue
            except Exception:
                # 권한/경로 에러는 무시하고 다음 root
                pass

        # 수정시간 최신 순
        try:
            candidates.sort(key=lambda pp: pp.stat().st_mtime if pp.exists() else 0, reverse=True)
        except Exception:
            pass

        if not candidates:
            return moved  # 이동할 후보 없음

        # ---- 목적지 경로 결정 (wav/mp3 모두 대응) ----
        target_wav = proj_dir / "vocal.wav"
        target_mp3 = proj_dir / "vocal.mp3"

        for src in candidates:
            try:
                suffix = src.suffix.lower()
                if suffix == ".wav":
                    dst = target_wav
                elif suffix == ".mp3":
                    dst = target_mp3
                else:
                    continue

                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    try:
                        dst.unlink()
                    except Exception:
                        # 지우지 못하면 새 이름으로 보존
                        dst = dst.with_name(dst.stem + "_new" + dst.suffix)

                shutil.move(str(src), str(dst))
                moved.append(dst)
                # 최신 1개만 이동(기존 의도 유지). 모두 이동하려면 break 제거.
                break
            except (OSError, shutil.Error):
                # 다음 후보 시도
                continue
            except Exception:
                # 예기치 못한 케이스도 다음 후보 시도
                continue

        return moved

    # ────────────── 영상 빌드(선택) ──────────────
    # shorts_ui.py 파일의 on_video 함수 전체를 이 코드로 교체하세요. (약 7076 라인)

    def on_video(self, *, on_done_override: Optional[Callable] = None) -> None:
        try:
            self._save_ui_prefs_to_project()
        except Exception as e_save_prefs:
            print(f"[UI] on_video: UI 설정 저장 실패: {e_save_prefs}")
        """
        [수정됨] 영상 생성:
          - [요청 수정] UI에서 '렌더 설정' 그룹의 W/H/FPS/스텝 값을 읽어 build_shots_with_i2v로 전달.
          - run_job_with_progress_async를 사용하여 비동기 실행 및 실시간 로그 창 표시.
          - video_build.build_shots_with_i2v 기존 동작 100% 보존.
        """


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

            def _get_ui_int_value(widget_name: str, data_attr: str = "currentData",
                                  default_val: Optional[int] = None) -> Optional[int]:
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
                                    return int(raw_value)  # type: ignore
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

            # --- ▼▼▼ [핵심 수정] "렌더 설정" 그룹의 위젯 이름으로 변경 ▼▼▼ ---
            ui_w = _get_ui_int_value("cmb_render_w", "currentData", 540) # 기본값 540
            ui_h = _get_ui_int_value("cmb_render_h", "currentData", 960) # 기본값 960
            ui_fps = _get_ui_int_value("cmb_movie_fps", "currentData", 30)  # (이름 동일, 렌더 그룹 소속)
            ui_steps = _get_ui_int_value("spn_render_steps", "value", 6)  # spn_t2i_steps -> spn_render_steps
            # --- ▲▲▲ [핵심 수정] 끝 ▲▲▲ ---

            print(f"[UI-DEBUG] 영상생성 요청값: W={ui_w}, H={ui_h}, FPS={ui_fps}, Steps={ui_steps}")

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
                return  # finally 블록에서 버튼 활성화
            pdir = Path(proj_dir_val)

            # --- 백그라운드 작업 함수 정의 (콜백 인자 추가) ---
            def _job(progress_callback: Callable[[Dict[str, Any]], None]) -> None:
                """영상 생성을 수행하는 백그라운드 작업 함수."""
                build_func_local: Optional[Callable] = None

                # 0) project_dir 확보
                pdir = Path(getattr(self, "project_dir", "") or "")
                if not pdir.is_dir():
                    raise RuntimeError("project_dir가 설정되어 있지 않습니다.")

                # 1) build_shots_with_i2v import
                build_func_local = build_func_imp


                # 2) UI에서 total_frames 읽기
                tframes = 0
                sb_total_widget = getattr(self, "sb_total", None)
                if sb_total_widget is not None and hasattr(sb_total_widget, "value"):
                    try:
                        tframes = int(sb_total_widget.value())
                    except (TypeError, ValueError):
                        tframes = 0

                # 3) 0 이하이면 video.json 기준으로 다시 계산 시도
                if tframes <= 0:
                    progress_callback({"msg": "[경고] total_frames가 0 이하입니다. video.json 기준으로 재계산합니다..."})

                    # utils.load_json 안전 가져오기

                    video_path = pdir / "video.json"
                    video_doc = _load_json(video_path, {}) or {}

                    # fps
                    try:
                        fps_val = int(video_doc.get("fps", 16))
                    except Exception:
                        fps_val = 16

                    # duration
                    duration_sec = 0.0
                    try:
                        duration_sec = float(video_doc.get("duration", 0.0))
                    except Exception:
                        duration_sec = 0.0

                    # duration 없으면 scenes duration 합산
                    if duration_sec <= 0.0:
                        scenes = video_doc.get("scenes") or []
                        total_d = 0.0
                        for s in scenes:
                            if not isinstance(s, dict):
                                continue
                            try:
                                d = float(s.get("duration", 0.0))
                            except Exception:
                                d = 0.0
                            if d > 0:
                                total_d += d
                        duration_sec = total_d

                    if duration_sec > 0.0:
                        tframes = int(round(duration_sec * max(fps_val, 1)))
                        progress_callback({
                            "msg": f"[INFO] video.json 기준으로 total_frames={tframes} 으로 보정했습니다. "
                                   f"(duration={duration_sec:.3f}s, fps={fps_val})"
                        })
                        # UI에도 반영
                        if sb_total_widget is not None and hasattr(sb_total_widget, "setValue"):
                            try:
                                sb_total_widget.setValue(tframes)
                            except Exception:
                                pass
                    else:
                        # 진짜로 계산 불가 → 여기서 명확히 에러로 종료
                        raise RuntimeError(
                            "total_frames를 계산할 수 없습니다. '프로젝트 분석'을 먼저 실행해 주세요."
                        )

                # 여기까지 오면 tframes > 0.
                # ★★★ [수정] 여기서 위에서 읽은 ui_w, ui_h 변수를 사용합니다. ★★★
                # 기존 코드는 여기서 다시 getattr을 호출했지만,
                # 위에서 이미 렌더 설정값으로 ui_w, ui_h를 읽어두었으므로 그것을 그대로 씁니다.

                try:
                    sig_build_inner = inspect.signature(build_func_local)
                    build_kwargs_inner: Dict[str, Any] = {
                        "project_dir": str(pdir),
                        "total_frames": tframes,
                        "on_progress": progress_callback,
                    }

                    if "ui_width" in sig_build_inner.parameters and ui_w is not None:
                        build_kwargs_inner["ui_width"] = int(ui_w)
                    if "ui_height" in sig_build_inner.parameters and ui_h is not None:
                        build_kwargs_inner["ui_height"] = int(ui_h)
                    if "ui_fps" in sig_build_inner.parameters and ui_fps is not None:
                        build_kwargs_inner["ui_fps"] = int(ui_fps)
                    if "ui_steps" in sig_build_inner.parameters and ui_steps is not None:
                        build_kwargs_inner["ui_steps"] = int(ui_steps)

                    build_func_local(**build_kwargs_inner)

                except TypeError as e_type_build_inner:
                    progress_callback({
                        "msg": f"[경고] build_shots_with_i2v 호출 시그니처 불일치 ({e_type_build_inner}), UI 값 없이 호출 시도."
                    })
                    try:
                        build_func_local(str(pdir), tframes, on_progress=progress_callback)
                    except TypeError:
                        progress_callback({"msg": "[경고] on_progress 인자도 실패, 인자 없이 호출 시도."})
                        build_func_local(str(pdir), tframes)  # type: ignore[call-arg]
                    except Exception as e_fallback_call_inner:
                        raise RuntimeError(
                            f"build_shots_with_i2v 최종 호출 실패: {e_fallback_call_inner}"
                        ) from e_fallback_call_inner
                except Exception as e_build_other_inner:
                    raise RuntimeError(
                        f"build_shots_with_i2v 실행 오류: {e_build_other_inner}"
                    ) from e_build_other_inner

            # --- 작업 완료 콜백 ---
            def _done(ok: bool, payload: Any, err: Optional[Exception]) -> None:  # <-- payload 타입 Any로
                """작업 완료 후 UI 업데이트 및 메시지 표시."""

                # --- ▼▼▼ 수정된 부분 (on_done_override 호출) ▼▼▼ ---
                if on_done_override:
                    try:
                        # 매크로 콜백이 있으면 팝업을 띄우지 않고, 결과를 매크로로 전달
                        on_done_override(ok, payload, err)
                    except Exception as e_override:
                        print(f"[오류] on_video의 on_done_override 호출 실패: {e_override}")
                        # 폴백: 매크로 콜백 실패 시 직접 팝업 표시
                        QtWidgets.QMessageBox.critical(self, "매크로 오류", f"영상 생성 후 콜백 실패:\n{e_override}")
                    return  # 매크로가 호출되었으므로 여기서 종료
                # --- ▲▲▲ 수정된 부분 끝 ▲▲▲ ---

                # (단독 실행 시 기존 로직)
                if not ok and err:
                    err_type_name = type(err).__name__
                    err_message = str(err)
                    print(f"[오류] 영상 생성 작업 실패: {err_type_name}: {err_message}")
                    print(traceback.format_exc())
                    QtWidgets.QMessageBox.critical(self, "영상 생성 오류",
                                                   f"오류 발생:\n{err_type_name}: {err_message}\n\n상세 내용은 콘솔 로그를 확인하세요.")
                elif ok:
                    print("[정보] 영상 생성 작업 완료.")
                    QtWidgets.QMessageBox.information(self, "완료", "영상 생성 작업이 완료되었습니다.")

            # --- 진행창 유틸 로드 ---
            run_async_local: Optional[Callable] = None  # <-- 변수명 변경
            run_async_local = run_async_imp


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
                return  # 동기 실행 후 종료

            # --- run_async 호출 준비 ---
            kw_run_async: Dict[str, Any] = {}
            try:
                sig_run_async_check = inspect.signature(run_async_local)
                if "tail_file" in sig_run_async_check.parameters:
                    kw_run_async["tail_file"] = None
                if "on_done" in sig_run_async_check.parameters:
                    kw_run_async["on_done"] = _done
            except (TypeError, ValueError) as e_sig_check:
                print(f"[경고] run_async 시그니처 분석 실패 (호출은 시도): {e_sig_check}")
                kw_run_async = {"tail_file": None, "on_done": _done}

            # --- run_async 실행 (정확한 인자 전달) ---
            try:
                run_async_local(self, "영상 생성", _job, **kw_run_async)
            except Exception as e_run_call_final:
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
            print(f"[오류] on_video 실행 중 오류 발생: {type(e_outer_inner).__name__}: {e_outer_inner}")
            print(traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, "오류", f"영상 생성 시작 중 오류 발생:\n{e_outer_inner}")

        finally:
            if on_done_override is None:
                if btn_video_widget:
                    try:
                        btn_video_widget.setEnabled(True)
                    except RuntimeError:
                        pass

    def on_video_wan(self, *, on_done_override: Optional[Callable] = None) -> None:
        """
        [3단계] 영상 생성 (Wan 2.2 Start-End Interpolation 모드)

        수정 사항:
        1. [워크플로우] 'wan2.2movie.json' 전용 ID 매핑 적용 (Start: 386, End: 387, Len: 405)
        2. [로직] 키프레임(kf_i, kf_i+1)을 Start/End로 주입하여 구간 생성.
        3. [구조] 41프레임 단위 고정 분할 (오버랩 없음).
        4. [ReActor] 제거됨 (워크플로우에 없음).
        """

        # --- 버튼 제어 ---
        btn_video_widget: Optional[QtWidgets.QAbstractButton] = None
        for btn_name in ("btn_video", "btn_build_video"):
            widget_candidate = getattr(self, btn_name, None)
            if widget_candidate is not None:
                btn_video_widget = widget_candidate
                break
        if btn_video_widget:
            try:
                btn_video_widget.setEnabled(False)
            except RuntimeError:
                pass

        try:


            # --- 경로 준비 ---
            proj_dir_val = None
            if hasattr(self, "_current_project_dir") and callable(self._current_project_dir):
                proj_dir_val = str(self._current_project_dir())
            if not proj_dir_val:
                proj_dir_val = str(getattr(self, "project_dir", ""))

            if not proj_dir_val:
                QtWidgets.QMessageBox.warning(self, "오류", "프로젝트 폴더가 선택되지 않았습니다.")
                if btn_video_widget: btn_video_widget.setEnabled(True)
                return

            proj_dir = Path(proj_dir_val)
            video_json_path = proj_dir / "video.json"
            if not video_json_path.exists():
                QtWidgets.QMessageBox.critical(self, "오류", f"video.json 없음: {video_json_path}")
                if btn_video_widget: btn_video_widget.setEnabled(True)
                return

            # ★ [수정] wan2.2movie.json 로드
            wan_wf_path = Path(JSONS_DIR) / "wan2.2movie.json"
            if not wan_wf_path.exists():
                QtWidgets.QMessageBox.critical(self, "오류", f"워크플로우 파일 없음: {wan_wf_path.name}")
                if btn_video_widget: btn_video_widget.setEnabled(True)
                return

            wan_workflow_template = load_json(wan_wf_path)

            comfy_input_path = Path(COMFY_INPUT_DIR)

            clips_dir = proj_dir / "clips"
            clips_dir.mkdir(parents=True, exist_ok=True)

            imgs_base_dir = proj_dir / "imgs"
            imgs_base_dir.mkdir(parents=True, exist_ok=True)

            # --- 헬퍼 ---
            def _is_valid_video(path: Path) -> bool:
                if not path.exists() or path.stat().st_size < 1024: return False
                try:
                    cmd = [FFPROBE_EXE, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                           "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
                    return True
                except:
                    return False

            def _submit_and_wait_custom(base_url: str, graph: Dict, timeout: int, poll: float,
                                        log_func: Callable) -> Dict:
                prompt_url = f"{base_url.rstrip('/')}/prompt"
                history_url = f"{base_url.rstrip('/')}/history"
                try:
                    payload = {"prompt": graph, "client_id": "shorts_maker_wan_se"}
                    resp = requests.post(prompt_url, json=payload, timeout=30)
                    if resp.status_code != 200: raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
                    prompt_id = resp.json().get("prompt_id")
                    if not prompt_id: raise RuntimeError("No prompt_id")
                except Exception as e:
                    raise RuntimeError(f"제출 실패: {e}")

                start_time = time.time()
                while True:
                    if time.time() - start_time > timeout: raise TimeoutError("시간 초과")
                    try:
                        h_resp = requests.get(f"{history_url}/{prompt_id}", timeout=10)
                        if h_resp.status_code == 200 and prompt_id in h_resp.json():
                            return h_resp.json()[prompt_id]
                    except:
                        pass
                    time.sleep(poll)

            # --- Job ---
            def _job(progress_callback: _CallableType[[Dict[str, Any]], None]) -> str:
                _log = lambda msg: progress_callback({"msg": msg})
                _log(f"워크플로우({wan_wf_path.name}) - Start-End Interpolation 모드")

                vdoc = load_json(video_json_path) or {}
                fps_val = int(vdoc.get("fps", 16))
                scenes = vdoc.get("scenes") or []
                if not scenes: return "생성할 씬이 없습니다."

                SEG_LEN = 41  # 고정 세그먼트 길이

                for scene in scenes:
                    scene_id = scene.get("id")
                    if not scene_id: continue

                    # 1. 목표 프레임 계산
                    try:
                        duration_val = float(scene.get("duration", 0.0))
                    except:
                        duration_val = 0.0
                    scene_total_frames = int(round(duration_val * fps_val))
                    if scene_total_frames <= 0: continue

                    _log(f"[{scene_id}] 처리 시작 (목표 {scene_total_frames}f)")

                    # 2. 구간 분할 (41프레임 단위)
                    # 예: 100f -> 0~41, 41~82, 82~100 (3구간)
                    segments = []
                    cursor = 0
                    while cursor < scene_total_frames:
                        end = min(cursor + SEG_LEN, scene_total_frames)
                        segments.append((cursor, end))
                        cursor = end

                    # 3. 폴더 준비
                    scene_chunk_dir = clips_dir / scene_id
                    if scene_chunk_dir.exists(): shutil.rmtree(scene_chunk_dir)  # 기존 잔여물 삭제 (필수)
                    scene_chunk_dir.mkdir(parents=True, exist_ok=True)

                    # 4. 이 씬의 캐릭터:인덱스 맵 (예: ["female_01:0", "male_01:1"])
                    # 4. 이 씬의 캐릭터:인덱스 맵 (예: ["female_01:0", "male_01:1"])
                    char_index_map: Dict[str, int] = {}
                    chars_list = (
                            scene.get("characters")
                            or scene.get("character_objs")
                            or vdoc.get("characters")
                            or vdoc.get("character_objs")
                            or []
                    )

                    for c in chars_list:
                        cid = ""
                        cidx = 0
                        if isinstance(c, dict):
                            cid = str(c.get("id") or c.get("name") or "").strip()
                            try:
                                cidx = int(c.get("index") or c.get("face_index") or 0)
                            except Exception:
                                cidx = 0
                        elif isinstance(c, str):
                            text_c = c.strip()
                            if ":" in text_c:
                                left, right = text_c.split(":", 1)
                                cid = left.strip()
                                try:
                                    cidx = int(right.strip())
                                except Exception:
                                    cidx = 0
                            else:
                                cid = text_c
                                cidx = 0
                        if cid:
                            # 같은 캐릭터가 여러 번 나오면 마지막 값으로 덮어씀
                            char_index_map[cid] = cidx

                    # wan2.2movie.json 기준 ReActor ↔ LoadImage ↔ 캐릭터ID 매핑
                    # 475 ↔ 480(female_01), 477 ↔ 479(male_01), 478 ↔ 481(other_char/기타)
                    reactor_map = {
                        "475": ("480", "female_01"),
                        "477": ("479", "male_01"),
                        "478": ("481", "other_char"),
                    }


                    video_chunks: list[Path] = []
                    scene_failed = False

                    # 4. 구간별 생성
                    for idx, (s_f, e_f) in enumerate(segments):
                        chunk_name = f"seg_{idx:03d}.mp4"
                        chunk_path = scene_chunk_dir / chunk_name
                        video_chunks.append(chunk_path)

                        # 실제 생성해야 할 길이 (끝-시작)
                        # 주의: Interpolation 모델은 보통 '양 끝 포함' 프레임 수를 요구함.
                        # 예: 0~41 -> 41프레임 생성 (0번, 41번 이미지가 있고 그 사이+양끝 포함 42장? 아니면 간격이 41?)
                        # Wan VACE 노드는 'num_frames'가 생성될 총 프레임 수(양 끝 포함)를 의미할 가능성이 높음.
                        current_seg_len = e_f - s_f
                        if current_seg_len <= 1: continue  # 너무 짧으면 스킵

                        # 키프레임 이미지 준비 (Start & End)
                        # 1단계(이미지 생성)에서 kf_0, kf_1... 로 저장되어 있다고 가정
                        # idx번째 구간의 시작은 kf_{idx}, 끝은 kf_{idx+1}
                        # 마지막 구간의 끝 이미지는? -> kf_{idx+1}이 없으면 곤란함.
                        # 사용자 로직: "세그먼트별 프롬프트 이용" -> video.json에 frame_segments가 있고 그 개수만큼 이미지가 있어야 함.

                        kf_start_name = f"kf_{idx}.png"
                        kf_end_name = f"kf_{idx + 1}.png"

                        kf_start_path = imgs_base_dir / scene_id / kf_start_name
                        kf_end_path = imgs_base_dir / scene_id / kf_end_name

                        # 예외 처리: 마지막 구간인데 끝 이미지가 없다면?
                        # 보통 마지막 이미지는 t_00x.png (씬 대표 이미지)일 수도 있고, kf_마지막 번호일 수도 있음.
                        # 여기선 일단 파일이 있어야 진행. 없으면 에러.
                        if not kf_start_path.exists():
                            # 첫 시작이면 씬 대표 이미지 사용 시도
                            if idx == 0 and Path(scene.get("img_file", "")).exists():
                                kf_start_path = Path(scene.get("img_file"))
                            else:
                                _log(f"[ERR] 시작 이미지 {kf_start_name} 없음.")
                                scene_failed = True
                                break

                        if not kf_end_path.exists():
                            # 마지막 구간이면 시작 이미지를 복사해서 끝 이미지로 쓸 수도 없음 (움직임이 없으므로).
                            # 일단 에러 처리
                            _log(f"[ERR] 끝 이미지 {kf_end_name} 없음.")
                            scene_failed = True
                            break

                        _log(
                            f"[{scene_id}] 구간 {idx + 1}/{len(segments)} 생성 ({current_seg_len}f, {kf_start_name}~{kf_end_name})...")

                        try:
                            wf = json.loads(json.dumps(wan_workflow_template))

                            # 이미지 복사 & 주입
                            shutil.copy2(str(kf_start_path), comfy_input_path / kf_start_path.name)
                            shutil.copy2(str(kf_end_path), comfy_input_path / kf_end_path.name)

                            # ★ ID 매핑 (wan2.2movie.json 기준)
                            if "386" in wf: wf["386"]["inputs"]["image"] = kf_start_path.name  # Start
                            if "387" in wf: wf["387"]["inputs"]["image"] = kf_end_path.name  # End
                            if "405" in wf: wf["405"]["inputs"]["value"] = int(current_seg_len)  # Frames

                            # 프롬프트 (해당 구간의 프롬프트 찾기)
                            # video.json의 frame_segments 리스트에서 idx번째 항목 사용
                            seg_prompt = ""
                            frame_segs = scene.get("frame_segments", [])
                            if idx < len(frame_segs):
                                seg_prompt = frame_segs[idx].get("prompt_movie", "")
                            if not seg_prompt: seg_prompt = scene.get("prompt_movie", "")

                            # if "106" in wf: wf["106"]["inputs"]["positive_prompt"] = seg_prompt

                            # 시드
                            seed = random.randint(1, 9999999999)
                            if "391" in wf: wf["391"]["inputs"]["seed"] = seed

                            # --- 페이스스왑 주입 (캐릭터:인덱스 기반) ---

                            # 1) GIMMVFI 입력을 ReActor 체인 출력(477)으로 변경
                            #    28(디코드) → 478 → 475 → 477 → 455(GIMMVFI) → 450(업스케일) → 439(저장)
                            if "455" in wf:
                                wf["455"].setdefault("inputs", {})["images"] = ["477", 0]

                            # 2) 캐릭터ID 별로 ReActor/LoadImage 세팅
                            char_base = Path(CHARACTER_DIR)
                            for reactor_id, (load_node_id, char_id) in reactor_map.items():
                                node_re = wf.get(reactor_id)
                                node_load = wf.get(load_node_id)
                                if not isinstance(node_re, dict) or not isinstance(node_load, dict):
                                    continue

                                inputs_re = node_re.setdefault("inputs", {})
                                face_idx = char_index_map.get(char_id)

                                # 이 씬에 해당 캐릭터가 없으면 비활성화
                                if face_idx is None:
                                    inputs_re["enabled"] = False
                                    continue

                                # 캐릭터 이미지 찾기 (CHARACTER_DIR/female_01.png 같은 것)
                                char_img_path = None
                                for ext in (".png", ".jpg", ".jpeg", ".webp"):
                                    p = char_base / f"{char_id}{ext}"
                                    if p.exists():
                                        char_img_path = p
                                        break

                                if char_img_path is None:
                                    # 이미지 없으면 이 ReActor만 끔
                                    inputs_re["enabled"] = False
                                    continue

                                dst_name = f"{char_id}{char_img_path.suffix}"
                                try:
                                    shutil.copy2(str(char_img_path), comfy_input_path / dst_name)
                                except Exception:
                                    # 복사 실패해도 일단 계속 진행
                                    pass

                                # LoadImage 노드에 소스 얼굴 이미지 주입
                                node_load.setdefault("inputs", {})["image"] = dst_name

                                # ReActor 활성화 + 인덱스 설정
                                inputs_re["enabled"] = True
                                inputs_re["input_faces_index"] = str(int(face_idx))
                                # source_faces_index는 기본값(0)을 유지, 없으면 0으로
                                if "source_faces_index" not in inputs_re:
                                    inputs_re["source_faces_index"] = "0"


                            # 저장 (439번이 Upscaled 최종본)
                            target_save = "439"
                            if target_save in wf:
                                prefix = f"wan_se/{scene_id}/{idx}"
                                wf[target_save]["inputs"]["filename_prefix"] = prefix
                            else:
                                raise RuntimeError("저장 노드(439)를 찾을 수 없습니다.")

                            # 실행
                            res = _submit_and_wait_custom(COMFY_HOST, wf, timeout=1800, poll=5.0, log_func=_log)

                            # 결과 처리
                            outs = res.get("outputs", {})
                            tgt_out = outs.get(target_save, {})
                            vids = tgt_out.get("videos") or tgt_out.get("gifs")

                            if not vids: raise RuntimeError(f"결과 없음 (Output: {list(outs.keys())})")

                            dl_info = vids[0]
                            ftype = dl_info.get("type", "output")
                            resp = requests.get(
                                f"{COMFY_HOST}/view",
                                params={"filename": dl_info['filename'], "subfolder": dl_info['subfolder'],
                                        "type": ftype},
                                timeout=120
                            )
                            if len(resp.content) < 1024: raise RuntimeError("0바이트 파일")

                            with open(chunk_path, "wb") as f:
                                f.write(resp.content)

                            if not _is_valid_video(chunk_path): raise RuntimeError("손상된 파일")

                        except Exception as e:
                            _log(f"[{scene_id}] 생성 실패: {e}")
                            scene_failed = True
                            try:
                                chunk_path.unlink()
                            except:
                                pass
                            break

                    # 5. 단순 병합 (Concat)
                    if not scene_failed and video_chunks:
                        final_path = clips_dir / f"{scene_id}.mp4"
                        try:
                            concatenate_scene_clips(video_chunks, final_path, FFMPEG_EXE)
                            _log(f"[{scene_id}] 병합 완료.")

                            # 최종 길이 보정 (선택 사항)
                            # _trim_to_frames(FFMPEG_EXE, FFPROBE_EXE, final_path, final_path, scene_total_frames, fps_val)

                        except Exception as e:
                            _log(f"[{scene_id}] 병합 실패: {e}")

                return "작업 완료"

            def _done(ok, payload, err):
                if on_done_override:
                    on_done_override(ok, payload, err)
                elif not ok:
                    QtWidgets.QMessageBox.critical(self, "실패", str(err))
                else:
                    QtWidgets.QMessageBox.information(self, "완료", str(payload))
                if btn_video_widget: btn_video_widget.setEnabled(True)

            run_job_with_progress_async(owner=self, title="Wan 2.2 (Start-End)", job=_job, on_done=_done)

        except Exception as e:
            if btn_video_widget: btn_video_widget.setEnabled(True)
            QtWidgets.QMessageBox.critical(self, "오류", str(e))

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
        return _resolve_audio_dir_from_template(getattr(settings_mod, "FINAL_OUT", str(settings_mod.BASE_DIR)), title)

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
        return getattr(settings_mod, "FFMPEG_EXE", "") or "ffmpeg"

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

    # ==== 완성된 영상 합치기 ==========
    def merging_videos_start(self, *, on_done_override: Optional[Callable] = None) -> None: # <-- 시그니처 수정
        """
        '영상 합치기' 버튼 핸들러:
        1. 누락된 씬(i2v) 생성
        2. 씬 비디오 병합 (music_vocal_ready.mp4)
        3. 오디오(vocal.wav) 합치기 (music_ready.mp4)
        """
        proj_dir = self._current_project_dir()
        if not proj_dir:
            QtWidgets.QMessageBox.warning(self, "오류", "프로젝트가 열려있지 않습니다.")
            return

        # --- 백그라운드 작업 정의 ---
        def job(progress_callback):
            # 1. 메인 파이프라인 함수 호출
            final_path = build_and_merge_full_video(
                project_dir=str(proj_dir),
                on_progress=progress_callback
            )
            return final_path # 성공 시 최종 파일 경로 반환

        # --- 완료 콜백 정의 ---
        def done(ok: bool, payload, err):
            # --- ▼▼▼ 수정된 콜백 로직 ▼▼▼ ---
            if on_done_override:
                on_done_override(ok, payload, err) # 매크로 콜백이 있으면 그것만 호출
            elif not ok:
                QtWidgets.QMessageBox.critical(self, "병합 실패", str(err))
            else:
                QtWidgets.QMessageBox.information(self, "병합 완료", f"최종 영상 생성 완료:\n{payload}")
            # --- ▲▲▲ 수정된 콜백 로직 ▲▲▲ ---

        # --- 비동기 실행 ---
        run_job_with_progress_async(
            owner=self,
            title="최종 영상 병합",
            job=job,
            on_done=done
        )

    # ==== 가사넣기 ================================
    def lyrics_in_start(self, *, on_done_override: Optional[Callable] = None): # <-- 시그니처 수정
        """
        [수정됨] music_ready.mp4 파일에 video.json의 제목과 가사를 주입하고
        최종본인 music.mp4 (또는 final_with_subs.mp4)를 생성합니다.
        (MoviePy 대신 FFMPEG drawtext 사용)
        """
        proj_dir_str = self._current_project_dir()
        if not proj_dir_str:
            QtWidgets.QMessageBox.warning(self, "오류", "프로젝트가 열려있지 않습니다.")
            return

        proj_dir = Path(proj_dir_str)
        video_ready_path = proj_dir / "music_ready.mp4"
        video_json_path = proj_dir / "video.json"
        final_output_path = proj_dir / "final_with_subs.mp4"  # 최종 파일 이름

        if not video_ready_path.exists():
            QtWidgets.QMessageBox.warning(self, "오류", f"원본 영상({video_ready_path.name})이 없습니다.\n'영상 합치기'를 먼저 실행하세요.")
            return

        if not video_json_path.exists():
            QtWidgets.QMessageBox.warning(self, "오류", f"자막 정보({video_json_path.name})가 없습니다.\n'프로젝트분석'을 먼저 실행하세요.")
            return


        selected_font_name = self.cmb_font.currentFont().family()
        title_size = self.spn_title_font_size.value()  # <-- 1줄 추가
        lyric_size = self.spn_lyric_font_size.value()  # <-- 1줄 추가

        # --- 백그라운드 작업 정의 ---
        def job(progress_callback):
            progress_callback({"msg": "FFMPEG으로 자막/제목 삽입 시작..."})

            final_path = add_subtitles_with_ffmpeg(
                video_in_path=video_ready_path,
                video_json_path=video_json_path,
                out_path=final_output_path,
                ffmpeg_exe=FFMPEG_EXE,  #
                font_name=selected_font_name,
                title_fontsize=title_size,  # 이제 'title_size' 변수가 존재함
                lyric_fontsize=lyric_size  # 이제 'lyric_size' 변수가 존재함
            )
            return final_path

        # --- 완료 콜백 정의 ---
        def done(ok: bool, payload, err):
            # --- ▼▼▼ 수정된 콜백 로직 ▼▼▼ ---
            if on_done_override:
                on_done_override(ok, payload, err) # 매크로 콜백이 있으면 그것만 호출
            elif not ok:
                QtWidgets.QMessageBox.critical(self, "자막 삽입 실패", str(err))
            else:
                QtWidgets.QMessageBox.information(self, "자막 삽입 완료", f"최종 영상 생성 완료:\n{payload}")
            # --- ▲▲▲ 수정된 콜백 로직 ▲▲▲ ---

        # --- 비동기 실행 ---
        run_job_with_progress_async(
            owner=self,
            title="자막 및 제목 삽입 중",
            job=job,
            on_done=done
        )

        # ────────────── 신규 매크로 핸들러 ──────────────

    def on_click_macro_analyze(self) -> None:
        """매크로: 1. 음악분석 -> 2. 프로젝트분석"""

        def _on_analysis_done(ok: bool, payload: Any, err: Optional[Exception]):
            """1단계(음악분석) 완료 콜백"""
            if not ok:
                print(f"[MACRO-ANALYZE] 1단계 (음악분석) 실패: {err}")
                QtWidgets.QMessageBox.critical(self, "분석 매크로 1단계 실패", f"음악 분석 중 오류가 발생했습니다:\n{err}")
                return

            print("[MACRO-ANALYZE] 1단계 (음악분석) 완료. 2단계 (프로젝트분석) 시작...")

            # 2단계(프로젝트분석) 호출
            # on_click_build_story_from_seg_async는 자체적으로 팝업을 띄우므로 on_done_override 불필요
            try:
                self.on_click_build_story_from_seg_async()
            except Exception as erroe:
                QtWidgets.QMessageBox.critical(self, "분석 매크로 2단계 오류", f"프로젝트 분석 시작 중 오류:\n{erroe}")

        # 1단계(음악분석) 호출
        print("[MACRO-ANALYZE] 1단계 (음악분석) 시작...")
        try:
            # on_done_override를 전달하여 1단계 완료 시 _on_analysis_done이 호출되도록 함
            self.on_click_analyze_music(on_done_override=_on_analysis_done)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "분석 매크로 1단계 오류", f"음악 분석 시작 중 오류:\n{e}")

        # shorts_ui.py

    def on_click_macro_build_video(self) -> None:
        """매크로: 1. 영상생성(i2v) -> 2. 영상합치기 -> 3. 가사넣기"""
        # (Callback 시그니처를 위해 typing 임포트)

        # --- 3단계 (가사넣기) ---
        def _on_merge_done(ok: bool, payload: Any, err: Optional[Exception]):
            """2단계(영상합치기) 완료 콜백"""
            if not ok:
                print(f"[MACRO-BUILD] 2단계 (영상합치기) 실패: {err}")
                QtWidgets.QMessageBox.critical(self, "영상만들기 2단계 실패", f"영상 병합 중 오류:\n{err}")
                return

            print("[MACRO-BUILD] 2단계 (영상합치기) 완료. 3단계 (가사넣기) 시작...")
            try:
                # 3단계(가사넣기) 호출. 이게 마지막이므로 override 불필요.
                # (lyrics_in_start는 작업 완료 시 자체적으로 팝업을 띄웁니다)
                self.lyrics_in_start()
            except Exception as e_lyrics_start:  # 'ee' -> 'e_lyrics_start'
                QtWidgets.QMessageBox.critical(self, "영상만들기 3단계 오류", f"가사 삽입 시작 중 오류:\n{e_lyrics_start}")

        # --- 2단계 (영상합치기) ---
        def _on_video_gen_done(ok: bool, payload: Any, err: Optional[Exception]):
            """1단계(영상생성) 완료 콜백"""

            # --- ▼▼▼ [수정] 1단계(on_video)가 끝났으므로 "영상생성" 버튼을 여기서 복구 ▼▼▼ ---
            try:
                btn_video_widget: Optional[QtWidgets.QAbstractButton] = None
                # "영상생성" 버튼(btn_video)을 찾습니다.
                for btn_name in ("btn_video", "btn_build_video"):
                    widget_candidate = getattr(self, btn_name, None) or \
                                       getattr(getattr(self, "ui", None), btn_name, None)
                    if isinstance(widget_candidate, QtWidgets.QAbstractButton):
                        btn_video_widget = widget_candidate
                        break
                if btn_video_widget:
                    btn_video_widget.setEnabled(True)  # 버튼을 다시 활성화합니다.
            except (AttributeError, RuntimeError) as e_btn_enable:
                print(f"[WARN] 매크로: 영상생성 버튼 복구 실패: {e_btn_enable}")
            # --- ▲▲▲ [수정] 끝 ▲▲▲ ---

            if not ok:
                print(f"[MACRO-BUILD] 1단계 (영상생성) 실패: {err}")
                QtWidgets.QMessageBox.critical(self, "영상만들기 1단계 실패", f"영상 생성(i2v) 중 오류:\n{err}")
                return  # (실패 시 2단계 진입 안 함)

            print("[MACRO-BUILD] 1단계 (영상생성) 완료. 2단계 (영상합치기) 시작...")
            try:
                # 2단계(영상합치기) 호출, 3단계를 콜백으로 전달
                self.merging_videos_start(on_done_override=_on_merge_done)
            except Exception as e_merge_start:  # 'ae' -> 'e_merge_start'
                QtWidgets.QMessageBox.critical(self, "영상만들기 2단계 오류", f"영상 합치기 시작 중 오류:\n{e_merge_start}")

        # --- 1단계 (영상생성) ---
        print("[MACRO-BUILD] 1단계 (영상생성) 시작...")
        try:
            # 1단계(영상생성) 호출, 2단계를 콜백으로 전달
            # 이 on_video 함수는 UI의 W/H/FPS/스텝 설정을 읽어옵니다.
            self.on_video(on_done_override=_on_video_gen_done)
        except Exception as e_video_start:  # 'e' -> 'e_video_start'
            QtWidgets.QMessageBox.critical(self, "영상만들기 1단계 오류", f"영상 생성 시작 중 오류:\n{e_video_start}")


# ───────── 워크플로 저장 노드(class_type) 영구 수정 도우미 ─────────


# ---- Add: rewrite SaveAudio nodes in a ComfyUI workflow JSON ----------------




# ==================================================
def _inject_render_prefs_methods():
    """
    [수정됨 v20] MainWindow에 _create_render_widgets / _save_ui_prefs_to_project (및 테스트 버튼) 없으면 주입.
    - [버그 수정] "이 코드에 도달할 수 없습니다" (unreachable code) 버그 수정.
      'custom' 프리셋에서 return 하던 로직을 'if/else' 구조로 변경하여,
      다른 프리셋 선택 시 lock_handler(True)가 올바르게 호출되도록 수정.
    - [기존] '이미지 설정'과 '렌더 설정' 위젯을 별도로 생성합니다.
    """


    # 이미 있으면 건너뜀
    if (
            hasattr(MainWindow, "_create_render_widgets")
            and hasattr(MainWindow, "_save_ui_prefs_to_project")
    ):
        return

    # MainWindow 클래스 내부 메서드로 추가
    def _guess_project_dir(self) -> Path:
        """
        현재 프로젝트 폴더 추정:
          1) self._current_project_dir()가 있으면 우선 사용
          2) settings.FINAL_OUT 템플릿의 [title] 치환
          3) settings.BASE_DIR / <정제된 제목>
        """


        # 1) UI 제공 메서드
        if hasattr(self, "_current_project_dir"):
            try:
                d = self._current_project_dir()
                if d:
                    return _p(d)
            except Exception:
                pass

        # 2) 제목 기반
        try:
            le = getattr(self, "le_title", None) or getattr(getattr(self, "ui", None), "le_title", None)
            title_val = _sanitize_title_fn(le.text().strip()) if le is not None else ""
        except Exception:
            title_val = ""
        if not title_val:
            title_val = "무제"

        # 2-1) FINAL_OUT 템플릿 우선
        final_tpl = getattr(settings_mod, "FINAL_OUT", "")
        if final_tpl and "[title]" in final_tpl:
            return _p(final_tpl.replace("[title]", title_val))

        # 2-2) BASE_DIR/[title]
        base_dir_val = getattr(settings_mod, "BASE_DIR", ".")
        return _p(base_dir_val) / title_val

    # ==== 메서드 정의: 드롭다운 UI 추가 ====
    def _create_render_widgets(self) -> None:
        """
        [수정됨] '이미지 설정'과 '렌더 설정' 그룹에 필요한 모든 위젯을 생성하고 시그널을 연결합니다.
        """
        # settings 모듈을 안전하게 참조 (s_mod_prefs 별칭 사용)


        # --- 내부 유틸 함수 (load_json) ---
        _load_json_local_prefs: Callable  # 타입 명시
        # --- 내부 유틸 함수 (_guess_project_dir) ---
        _guess_project_dir_local: Callable[[], Path]  # 타입 명시
        try:
            _guess_project_dir_local = getattr(self, "_guess_project_dir")
            if not callable(_guess_project_dir_local):
                raise AttributeError("_guess_project_dir 메서드를 찾을 수 없습니다.")
        except AttributeError as e_guess_dir_local:  # 변수명 변경
            print(f"[오류] _create_render_widgets: {e_guess_dir_local}")

            def _guess_project_dir_fb() -> Path:
                return Path(".")

            _guess_project_dir_local = _guess_project_dir_fb

        # --- [신규] 헬퍼 함수: Linter 경고를 피하기 위한 안전 연결 ---
        def _safe_connect(signal_object, slot_function):
            """Linter 경고를 피하기 위해 getattr로 안전하게 connect 호출"""
            try:
                connect_method = getattr(signal_object, "connect", None)
                if callable(connect_method) and callable(slot_function):
                    connect_method(slot_function)
            except Exception as e_connect:
                print(f"[경고] _create_render_widgets: 시그널 연결 실패: {e_connect}")

        # --- [수정] 위젯 생성 (두 그룹용으로 분리) ---

        # [신규] 1. "이미지 설정" 그룹용 위젯 생성
        self.cmb_img_w = QtWidgets.QComboBox()
        self.cmb_img_h = QtWidgets.QComboBox()
        self.cmb_res_preset = QtWidgets.QComboBox()
        self.spn_t2i_steps = QtWidgets.QSpinBox()

        self.cmb_img_w.setToolTip("이미지 가로 (width)")
        self.cmb_img_h.setToolTip("이미지 세로 (height)")
        self.cmb_res_preset.setToolTip("이미지 해상도 프리셋")
        self.spn_t2i_steps.setToolTip("이미지 생성 샘플링 스텝 수")

        # [신규] 2. "렌더 설정" 그룹용 위젯 생성
        self.cmb_render_w = QtWidgets.QComboBox()
        self.cmb_render_h = QtWidgets.QComboBox()
        self.cmb_render_preset = QtWidgets.QComboBox()
        self.spn_render_steps = QtWidgets.QSpinBox()
        self.cmb_movie_fps = QtWidgets.QComboBox()

        self.cmb_render_w.setToolTip("최종 렌더링 가로 (width)")
        self.cmb_render_h.setToolTip("최종 렌더링 세로 (height)")
        self.cmb_render_preset.setToolTip("렌더링 해상도 프리셋")
        self.spn_render_steps.setToolTip("렌더링 샘플링 스텝 수 (i2v 등)")
        self.cmb_movie_fps.setToolTip("타깃 FPS (i2v/렌더)")

        # 3. 공통: 폰트 관련 위젯
        self.cmb_font = QFontComboBox()
        self.cmb_font.setToolTip("자막에 사용할 폰트를 선택합니다.")
        self.cmb_font.setMinimumWidth(150)
        self.spn_title_font_size = self._spin(10, 200, 55, " px")
        self.spn_lyric_font_size = self._spin(10, 200, 25, " px")
        self.spn_title_font_size.setToolTip("제목 폰트 크기 (기본값 55)")
        self.spn_lyric_font_size.setToolTip("가사 폰트 크기 (기본값 25)")

        # --- [수정] 데이터 목록 준비 ---
        default_w_val, default_h_val = getattr(settings_mod, "DEFAULT_IMG_SIZE", (720, 1080))
        preset_widths = {304, 405, 540, 720, 832, 1080, 1280, 1920, 512, 1024}
        preset_heights = {540, 960, 1280, 1472, 1920, 720, 1080, 512, 1024}
        size_choices_conf = getattr(settings_mod, "IMAGE_SIZE_CHOICES", [240, 304, 480, 520, 540, 720, 960, 1080, 1280, 1440])
        size_choices_set = set(int(w) for w in size_choices_conf if str(w).isdigit())
        size_choices_set.update(preset_widths)
        size_choices_set.add(int(default_w_val))
        size_choices_val = sorted(list(size_choices_set))
        h_candidates_set = {int(round(w * 16 / 9)) for w in size_choices_val}
        h_candidates_set.update({int(round(w * 9 / 16)) for w in size_choices_val})
        h_candidates_set.update(preset_heights)
        h_candidates_set.add(int(default_h_val))
        h_candidates_val = sorted(list(h_candidates_set))

        fps_choices_val = getattr(settings_mod, "MOVIE_FPS_CHOICES", [16, 24, 30, 60])
        default_fps_val = int(getattr(settings_mod, "DEFAULT_MOVIE_FPS", 30))
        default_steps_val = int(getattr(settings_mod, "DEFAULT_T2I_STEPS", 6))

        presets_data = [
            ("Shorts 9:16 · 405×720", 405, 720, "405×720"),
            ("Shorts 9:16 · 540×960", 540, 960, "540×960"),
            ("Shorts 9:16 · 720×1280", 720, 1280, "shorts_720x1280"),
            ("Shorts 9:16 · 832×1472", 832, 1472, "shorts_832x1472"),
            ("Shorts 9:16 · 1080×1920", 1080, 1920, "shorts_1080x1920"),
            ("Landscape 16:9 · 1280×720", 1280, 720, "land_1280x720"),
            ("Landscape 16:9 · 1920×1080", 1920, 1080, "land_1920x1080"),
            ("Square 1:1 · 512×512", 512, 512, "square_512"),
            ("Square 1:1 · 1024×1024", 1024, 1024, "square_1024"),
            ("맞춤(커스텀)", -1, -1, "custom"),
        ]

        # --- [수정] 콤보박스 채우기 (두 그룹 모두) ---
        for combo_w, combo_h, combo_preset, spin_step in [
            (self.cmb_img_w, self.cmb_img_h, self.cmb_res_preset, self.spn_t2i_steps),
            (self.cmb_render_w, self.cmb_render_h, self.cmb_render_preset, self.spn_render_steps)
        ]:
            try:
                for w_val_item in size_choices_val:
                    combo_w.addItem(str(w_val_item), int(w_val_item))
                for h_val_item in h_candidates_val:
                    combo_h.addItem(str(h_val_item), int(h_val_item))
                for label_text, w_preset, h_preset, key_preset in presets_data:
                    combo_preset.addItem(label_text, (w_preset, h_preset, key_preset))
                spin_step.setRange(1, 200)
                spin_step.setValue(default_steps_val)
            except (ValueError, TypeError) as e_fill:
                print(f"[경고] W/H/Preset 콤보박스 채우기 오류: {e_fill}")
                if combo_w.count() == 0: combo_w.addItem(str(default_w_val), int(default_w_val))
                if combo_h.count() == 0: combo_h.addItem(str(default_h_val), int(default_h_val))

        # FPS 콤보박스 채우기 (렌더 설정에만)
        try:
            valid_fps_choices = [int(f) for f in fps_choices_val if str(f).isdigit()]
            if default_fps_val not in valid_fps_choices:
                valid_fps_choices.append(default_fps_val)
                valid_fps_choices.sort()
            for fps_val_item in valid_fps_choices:
                self.cmb_movie_fps.addItem(str(fps_val_item), int(fps_val_item))
        except (ValueError, TypeError) as e_fps_fill_local:
            print(f"[경고] FPS 콤보박스 채우기 오류: {e_fps_fill_local}")
            if self.cmb_movie_fps.count() == 0: self.cmb_movie_fps.addItem(str(default_fps_val), int(default_fps_val))

        # --- [수정] project.json 초기값 반영 (두 그룹 모두) ---
        proj_dir_current = _guess_project_dir_local()
        pj_current = proj_dir_current / "project.json"
        meta_current = _load_json_local_prefs(pj_current, {}) if pj_current.exists() else {}
        ui_prefs_current = meta_current.get("ui_prefs") or {}

        def _set_combo_safe(combo: QtWidgets.QComboBox, val_to_set: int, fallback_val: int):
            try:
                val_int = int(val_to_set)
                idx_found = combo.findData(val_int)
                if idx_found < 0:
                    idx_found = combo.findData(int(fallback_val))
                combo.setCurrentIndex(idx_found if idx_found >= 0 else 0)
            except (ValueError, TypeError):
                combo.setCurrentIndex(0)

        # [신규] "이미지 설정" 값 로드
        img_size = ui_prefs_current.get("image_size", [default_w_val, default_h_val])
        img_preset_key = str(ui_prefs_current.get("image_preset", "custom"))
        img_steps = int(ui_prefs_current.get("image_steps", default_steps_val))

        _set_combo_safe(self.cmb_img_w, img_size[0], default_w_val)
        _set_combo_safe(self.cmb_img_h, img_size[1], default_h_val)
        self.spn_t2i_steps.setValue(img_steps)

        # [신규] "렌더 설정" 값 로드 (기존 키 사용)
        render_size = ui_prefs_current.get("render_size",
                                           ui_prefs_current.get("image_size", [default_w_val, default_h_val]))  # 호환성
        render_preset_key = str(
            ui_prefs_current.get("render_preset", ui_prefs_current.get("resolution_preset", "custom")))  # 호환성
        render_steps = int(
            ui_prefs_current.get("render_steps", ui_prefs_current.get("t2i_steps", default_steps_val)))  # 호환성
        ui_fps_val = ui_prefs_current.get("movie_fps", default_fps_val)

        _set_combo_safe(self.cmb_render_w, render_size[0], default_w_val)
        _set_combo_safe(self.cmb_render_h, render_size[1], default_h_val)
        _set_combo_safe(self.cmb_movie_fps, ui_fps_val, default_fps_val)
        self.spn_render_steps.setValue(render_steps)

        # 폰트 설정 불러오기
        try:
            font_family = ui_prefs_current.get("font_family", "굴림")
            self.cmb_font.setCurrentFont(QFont(font_family))
            title_size = ui_prefs_current.get("title_font_size", 55)
            lyric_size = ui_prefs_current.get("lyric_font_size", 25)
            self.spn_title_font_size.setValue(int(title_size))
            self.spn_lyric_font_size.setValue(int(lyric_size))
        except Exception as e_font_load:
            print(f"[경고] 폰트 설정 불러오기 실패: {e_font_load}")

        # --- [신규] 시그널 연결 (두 그룹 모두) ---

        def _create_wh_lock_handler(cmb_w: QtWidgets.QComboBox, cmb_h: QtWidgets.QComboBox):
            """W/H 콤보박스 잠금/해제 핸들러 생성"""

            def _lock_wh(lock: bool):
                cmb_w.setEnabled(not lock)
                cmb_h.setEnabled(not lock)
                tip = "프리셋을 '맞춤(커스텀)'으로 바꾸면 수정 가능" if lock else "W/H를 직접 선택"
                cmb_w.setToolTip(tip)
                cmb_h.setToolTip(tip)

            return _lock_wh

        def _create_preset_apply_handler(cmb_preset, cmb_w, cmb_h, lock_handler):
            """
            프리셋 변경 시 W/H 값 적용 핸들러.
            - 'custom'이면 잠금 해제
            - 그 외 프리셋이면 W/H 강제 세팅 후 잠금
            """

            def _apply_preset():
                data = cmb_preset.currentData()
                if not (isinstance(data, tuple) and len(data) == 3):
                    return

                # 바깥 for문의 w_preset/h_preset이랑 이름 안 겹치게 바꿨다
                w_val, h_val, key = data
                is_custom = (key == "custom")

                if not is_custom:
                    cmb_w.blockSignals(True)
                    cmb_h.blockSignals(True)
                    try:
                        idx_w = cmb_w.findData(int(w_val))
                        if idx_w >= 0:
                            cmb_w.setCurrentIndex(idx_w)

                        idx_h = cmb_h.findData(int(h_val))
                        if idx_h >= 0:
                            cmb_h.setCurrentIndex(idx_h)
                    finally:
                        cmb_w.blockSignals(False)
                        cmb_h.blockSignals(False)

                # custom → 잠금 해제 / 프리셋 → 잠금
                lock_handler(not is_custom)

            return _apply_preset

        def _create_wh_changed_handler(cmb_preset: QtWidgets.QComboBox, lock_handler: Callable):
            """W/H 변경 시 프리셋을 '맞춤'으로 변경하는 핸들러 생성"""

            def _on_changed():
                custom_idx = -1
                for i in range(cmb_preset.count()):
                    data = cmb_preset.itemData(i)
                    if isinstance(data, tuple) and len(data) == 3 and data[2] == "custom":
                        custom_idx = i
                        break

                # 현재 프리셋이 '맞춤'이 아닌 경우에만 '맞춤'으로 변경
                if custom_idx >= 0 and cmb_preset.currentIndex() != custom_idx:
                    cmb_preset.blockSignals(True)
                    cmb_preset.setCurrentIndex(custom_idx)
                    cmb_preset.blockSignals(False)
                    lock_handler(False)

            return _on_changed

        # 1. "이미지 설정" 그룹 시그널 연결
        img_lock_handler = _create_wh_lock_handler(self.cmb_img_w, self.cmb_img_h)
        img_preset_handler = _create_preset_apply_handler(self.cmb_res_preset, self.cmb_img_w, self.cmb_img_h,
                                                          img_lock_handler)
        img_wh_changed_handler = _create_wh_changed_handler(self.cmb_res_preset, img_lock_handler)

        _safe_connect(self.cmb_res_preset.currentIndexChanged, img_preset_handler)
        _safe_connect(self.cmb_img_w.currentIndexChanged, img_wh_changed_handler)
        _safe_connect(self.cmb_img_h.currentIndexChanged, img_wh_changed_handler)

        # 2. "렌더 설정" 그룹 시그널 연결
        render_lock_handler = _create_wh_lock_handler(self.cmb_render_w, self.cmb_render_h)
        render_preset_handler = _create_preset_apply_handler(self.cmb_render_preset, self.cmb_render_w,
                                                             self.cmb_render_h, render_lock_handler)
        render_wh_changed_handler = _create_wh_changed_handler(self.cmb_render_preset, render_lock_handler)

        _safe_connect(self.cmb_render_preset.currentIndexChanged, render_preset_handler)
        _safe_connect(self.cmb_render_w.currentIndexChanged, render_wh_changed_handler)
        _safe_connect(self.cmb_render_h.currentIndexChanged, render_wh_changed_handler)

        # 3. 프리셋 초기값 설정 (두 그룹 모두)
        def _set_initial_preset(cmb_preset: QtWidgets.QComboBox, key: str, apply_handler: Callable):
            init_idx = 0
            for i in range(cmb_preset.count()):
                data = cmb_preset.itemData(i)
                if isinstance(data, tuple) and len(data) == 3 and data[2] == key:
                    init_idx = i
                    break
            cmb_preset.setCurrentIndex(init_idx)
            apply_handler()  # 초기 잠금 상태 적용

        _set_initial_preset(self.cmb_res_preset, img_preset_key, img_preset_handler)
        _set_initial_preset(self.cmb_render_preset, render_preset_key, render_preset_handler)

        # 4. 저장 시그널 연결 (모든 위젯)
        all_widgets_to_save = [
            self.cmb_img_w, self.cmb_img_h, self.cmb_res_preset, self.spn_t2i_steps,
            self.cmb_render_w, self.cmb_render_h, self.cmb_render_preset, self.spn_render_steps,
            self.cmb_movie_fps, self.cmb_font, self.spn_title_font_size, self.spn_lyric_font_size
        ]

        for widget in all_widgets_to_save:
            if hasattr(widget, "currentIndexChanged"):  # QComboBox
                _safe_connect(widget.currentIndexChanged, self._save_ui_prefs_to_project)
            elif hasattr(widget, "valueChanged"):  # QSpinBox
                _safe_connect(widget.valueChanged, self._save_ui_prefs_to_project)
            elif hasattr(widget, "currentFontChanged"):  # QFontComboBox
                _safe_connect(widget.currentFontChanged, self._save_ui_prefs_to_project)

    # ==== 메서드 정의: project.json 저장 ====
    # 파일: shorts_ui.py (MainWindow 클래스 내부)

    def _save_ui_prefs_to_project(self) -> None:
        """[수정됨] '이미지 설정'과 '렌더 설정' 값을 별도 키로 저장하고 project.json에만 저장합니다."""


        # 현재 활성 프로젝트 경로를 가져옵니다.
        proj_dir_func = getattr(self, "_current_project_dir", None)
        proj_dir = proj_dir_func() if callable(proj_dir_func) else getattr(self, "project_dir", None)

        if proj_dir is None:
            proj_dir = self._latest_project()
            if proj_dir is None: return

        pj = Path(proj_dir) / "project.json"

        meta = load_json(pj, {}) if pj.exists() else {}
        if not isinstance(meta, dict): meta = {}

        ui = meta.get("ui_prefs") or {}

        # 1. "이미지 설정" 그룹 저장 (image_size/image_steps)
        img_w_sel = int(self.cmb_img_w.currentData())
        img_h_sel = int(self.cmb_img_h.currentData())
        img_preset_key = "custom"
        img_data_val = self.cmb_res_preset.currentData()
        if isinstance(img_data_val, tuple) and len(img_data_val) == 3:
            _, _, img_preset_key = img_data_val

        ui["image_size"] = [img_w_sel, img_h_sel]
        ui["image_preset"] = str(img_preset_key)
        ui["image_steps"] = int(self.spn_t2i_steps.value())

        # 2. "렌더 설정" 그룹 저장 (render_size/render_steps/movie_fps)
        render_w_sel = int(self.cmb_render_w.currentData())
        render_h_sel = int(self.cmb_render_h.currentData())
        render_preset_key = "custom"
        render_data_val = self.cmb_render_preset.currentData()
        if isinstance(render_data_val, tuple) and len(render_data_val) == 3:
            _, _, render_preset_key = render_data_val

        ui["render_size"] = [render_w_sel, render_h_sel]
        ui["render_preset"] = str(render_preset_key)
        ui["render_steps"] = int(self.spn_render_steps.value())

        # 3. 공통 설정 (FPS, 폰트) 저장
        ui["movie_fps"] = int(self.cmb_movie_fps.currentData())
        try:
            ui["font_family"] = self.cmb_font.currentFont().family()
            ui["title_font_size"] = self.spn_title_font_size.value()
            ui["lyric_font_size"] = self.spn_lyric_font_size.value()
        except Exception as e:
            print(f"[UI] 폰트/크기 저장 실패: {e}")

        # 4. 호환성 키 저장 (기존 로직 유지)
        ui["t2i_steps"] = int(self.spn_t2i_steps.value())
        ui["resolution_preset"] = str(render_preset_key)
        ui["image_size"] = [img_w_sel, img_h_sel]

        meta["ui_prefs"] = ui
        save_json(pj, meta)

    # === 옵션: 누락 이미지 생성 버튼 핸들러도 주입 ===
    def on_click_test2_1_generate_missing_images(self) -> None:
        """[수정됨] '이미지 설정' 그룹의 W/H/Step 값을 사용합니다."""

        comfy_log_file = getattr(settings_mod, "COMFY_LOG_FILE", None)

        story_path = Path(self.txt_story_path.text()).resolve()

        # [수정] '이미지 설정' (self.cmb_img_w) 그룹의 값을 읽음
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
                workflow_path=None,
                on_progress=on_progress,
            )

        def _done(ok, payload, err):
            if not ok:
                QtWidgets.QMessageBox.critical(self, "이미지 생성 실패", str(err))
                return
            cnt = len(payload or [])
            QtWidgets.QMessageBox.information(self, "완료", f"새 이미지 {cnt}개 생성")

        run_job_with_progress_async(self, "테스트2_1: 누락 이미지 생성", _job, tail_file=comfy_log_file, on_done=_done)

    # 클래스에 바인딩
    setattr(MainWindow, "_guess_project_dir", _guess_project_dir)
    setattr(MainWindow, "_create_render_widgets", _create_render_widgets)
    setattr(MainWindow, "_save_ui_prefs_to_project", _save_ui_prefs_to_project)
    setattr(MainWindow, "on_click_test2_1_generate_missing_images", on_click_test2_1_generate_missing_images)


# === 주입을 즉시 실행 (MainWindow 인스턴스 생성 전에!) ===
_inject_render_prefs_methods()


# shorts_ui.py 파일 맨 아래 (main 함수 직전)에 이 클래스 전체를 붙여넣으세요.

class SegmentEditDialog(QtWidgets.QDialog):
    """
    [신규] 씬(Scene) 내부의 세그먼트(Frame Segments)를 편집하는 전용 다이얼로그.
    - 1단계(qwen)로 생성된 키프레임(kf_1.png...)을 직접 보고 수정합니다.
    - 'prompt_movie' (AI가 생성한 행동 묘사)를 직접 수정합니다.
    - 'direct_prompt' (AI 요청용)를 입력받습니다.
    - [요청사항] "영상 삭제" 버튼은 UI만 있고 기능은 비활성화됩니다.
    """
    _AI_QUALITY_TAGS = "photorealistic, cinematic lighting, high detail, 8k, masterpiece"
    _AI_DEFAULT_NEGATIVE_TAGS = "lowres, bad anatomy, bad proportions, extra limbs, extra fingers, missing fingers, jpeg artifacts, signature, logo, nsfw, text, letters, typography, watermark"

    def __init__(self,
                 scene_id: str,
                 scene_data: Dict[str, Any],
                 full_video_data: Dict[str, Any],
                 json_path: Path,
                 ai_instance: AI,
                 parent: Optional[QtWidgets.QWidget] = None):

        super().__init__(parent)

        self.scene_id = scene_id
        self.scene_data = scene_data  # t_001 씬 객체
        self.full_video_data = full_video_data  # video.json 전체 (저장용)
        self.json_path = json_path  # video.json 경로 (저장용)
        self.ai_instance = ai_instance

        # 이 씬의 세그먼트 목록 (이것이 실제 데이터 소스)
        self.frame_segments_data: List[Dict[str, Any]] = self.scene_data.get("frame_segments", [])

        # 위젯 매핑용
        self.widget_map_prompt: List[Tuple[int, QtWidgets.QTextEdit]] = []  # (seg_index, prompt_movie_widget)
        self.widget_map_direct: List[Tuple[int, QtWidgets.QTextEdit]] = []  # (seg_index, direct_prompt_widget)

        self.THUMBNAIL_SIZE = 150

        self.setWindowTitle(f"세그먼트 수정: [{self.scene_id}]")
        self.setMinimumSize(900, 700)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        # --- 1. 메인 레이아웃 ---
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(2)

        # --- 2. 스크롤 영역 (세그먼트 목록) ---
        page_scroll_area = QtWidgets.QScrollArea()
        page_scroll_area.setWidgetResizable(True)
        scroll_content_widget = QtWidgets.QWidget()
        form_layout_page = QtWidgets.QFormLayout(scroll_content_widget)
        form_layout_page.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapAllRows)
        form_layout_page.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form_layout_page.setVerticalSpacing(8)
        page_scroll_area.setWidget(scroll_content_widget)

        main_layout.addWidget(page_scroll_area)

        # --- 3. 하단 버튼 레이아웃 ---
        button_layout = QtWidgets.QHBoxLayout()
        self.btn_ai_request = QtWidgets.QPushButton("AI 요청 (Direct Prompt)")
        self.btn_update = QtWidgets.QPushButton("업데이트")
        self.btn_cancel = QtWidgets.QPushButton("닫기")
        self.btn_ai_request.setToolTip(
            "오른쪽 'Direct Prompt' 내용을 기반으로\nAI에게 이 씬의 모든 'prompt_movie' 행동 묘사를 새로 요청합니다."
        )
        button_layout.addStretch(1)
        button_layout.addWidget(self.btn_cancel)
        button_layout.addWidget(self.btn_ai_request)
        button_layout.addWidget(self.btn_update)
        main_layout.addLayout(button_layout)

        # --- 4. 시그널 연결 ---
        self.btn_update.clicked.connect(self.on_update_and_close)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ai_request.clicked.connect(self.on_ai_request_segment)

        # --- 5. UI 빌드 ---
        self.load_and_build_segments_ui(form_layout_page)

    def load_and_build_segments_ui(self, form_layout: QtWidgets.QFormLayout):
        """세그먼트 목록을 UI에 그립니다."""
        try:
            base_imgs_dir = self.json_path.parent / "imgs"

            # 폰트
            font = QtGui.QFont()
            font.setFamily("Courier" if "Courier" in QtGui.QFontDatabase().families() else "Monospace")
            font.setPointSize(10)

            if not self.frame_segments_data:
                form_layout.addRow(QtWidgets.QLabel(f"[{self.scene_id}] 씬에 'frame_segments' 데이터가 없습니다."))
                return

            # --- 세그먼트 루프 ---
            for seg_index, segment_data in enumerate(self.frame_segments_data):
                if not isinstance(segment_data, dict): continue

                # --- 라벨 ---
                start_f = segment_data.get("start_frame", 0)
                end_f = segment_data.get("end_frame", 0)
                keyframe_id = f"kf_{seg_index + 1}"  # kf_0은 원본이므로 kf_1부터 시작

                label_text = (
                    f"<b>{self.scene_id} / 세그먼트 {seg_index + 1}</b> (키프레임: {keyframe_id}.png) | "
                    f"<b>프레임:</b> [{start_f} ~ {end_f}]"
                )
                label = QtWidgets.QLabel(label_text)
                label.setWordWrap(False)

                # --- 3단 레이아웃 컨테이너 ---
                row_container = QtWidgets.QWidget()
                row_layout = QtWidgets.QHBoxLayout(row_container)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(8)

                # --- 1. (왼쪽) 이미지 + 버튼 3개 ---
                left_vbox_widget = QtWidgets.QWidget()
                left_vbox = QtWidgets.QVBoxLayout(left_vbox_widget)
                left_vbox.setContentsMargins(0, 0, 0, 0)
                left_vbox.setSpacing(4)

                img_preview_label = ClickableLabel()
                img_preview_label.setFixedSize(self.THUMBNAIL_SIZE, self.THUMBNAIL_SIZE)
                img_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

                upload_button = QtWidgets.QPushButton("이미지 변경")
                upload_button.setFixedSize(self.THUMBNAIL_SIZE, 28)
                delete_image_button = QtWidgets.QPushButton("이미지 삭제")
                delete_image_button.setFixedSize(self.THUMBNAIL_SIZE, 28)

                # [요청사항] 영상삭제 버튼은 UI만 만들고 기능 비활성화
                delete_video_button = QtWidgets.QPushButton("청크 영상삭제 (UI)")
                delete_video_button.setFixedSize(self.THUMBNAIL_SIZE, 28)
                delete_video_button.setEnabled(False)
                delete_video_button.setToolTip("이 버튼은 현재 비활성화되어 있습니다.")

                left_vbox.addWidget(img_preview_label)
                left_vbox.addWidget(upload_button)
                left_vbox.addWidget(delete_image_button)
                left_vbox.addWidget(delete_video_button)
                left_vbox.addStretch(1)
                row_layout.addWidget(left_vbox_widget, 0)

                # --- 2. (중간) prompt_movie 텍스트 편집기 ---
                prompt_movie_edit = QtWidgets.QTextEdit()
                prompt_movie_edit.setPlainText(segment_data.get("prompt_movie", ""))
                prompt_movie_edit.setFont(font)
                prompt_movie_edit.setMinimumHeight(150)
                prompt_movie_edit.setToolTip(f"[{keyframe_id}] 행동 묘사 프롬프트 (prompt_movie)")
                prompt_movie_edit.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                                QtWidgets.QSizePolicy.Policy.Expanding)
                row_layout.addWidget(prompt_movie_edit, 1)

                # --- 3. (오른쪽) direct_prompt (AI 요청용) 텍스트 편집기 ---
                direct_prompt_edit = QtWidgets.QTextEdit()
                direct_prompt_edit.setPlainText(segment_data.get("direct_prompt", ""))  # 세그먼트에도 direct_prompt가 있을 수 있음
                direct_prompt_edit.setFont(font)
                direct_prompt_edit.setMinimumHeight(150)
                direct_prompt_edit.setToolTip(f"[{keyframe_id}] AI 요청 시 사용할 Direct Prompt")
                direct_prompt_edit.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                                 QtWidgets.QSizePolicy.Policy.Expanding)
                row_layout.addWidget(direct_prompt_edit, 1)

                # --- 위젯 매핑 ---
                self.widget_map_prompt.append((seg_index, prompt_movie_edit))
                self.widget_map_direct.append((seg_index, direct_prompt_edit))

                # --- 이미지 상태 설정 및 버튼 연결 ---
                keyframe_path = base_imgs_dir / self.scene_id / f"{keyframe_id}.png"
                img_file_str = str(keyframe_path)
                has_image = keyframe_path.exists()

                if has_image:
                    pixmap = QtGui.QPixmap(img_file_str)
                    if not pixmap.isNull():
                        pixmap_scaled = pixmap.scaled(self.THUMBNAIL_SIZE, self.THUMBNAIL_SIZE,
                                                      Qt.AspectRatioMode.KeepAspectRatio,
                                                      Qt.TransformationMode.SmoothTransformation)
                        img_preview_label.setPixmap(pixmap_scaled)
                        img_preview_label.setToolTip(f"경로: {img_file_str}\n(클릭해서 크게 보기)")
                        img_preview_label.clicked.connect(functools.partial(self.show_large_image, img_file_str))
                        upload_button.setText("이미지 변경")
                        delete_image_button.setEnabled(True)
                    else:
                        img_preview_label.setText("[파일\n오류]")
                        img_preview_label.setStyleSheet("border: 1px solid red; color: red;")
                        upload_button.setText("다시 업로드")
                        delete_image_button.setEnabled(True)
                else:
                    img_preview_label.setText("[이미지\n없음]")
                    img_preview_label.setToolTip(f"경로: {img_file_str}\n(파일이 존재하지 않습니다)")
                    img_preview_label.setStyleSheet("border: 1px dashed gray; color: gray;")
                    upload_button.setText("업로드")
                    delete_image_button.setEnabled(False)

                # [기능 연결]
                upload_button.clicked.connect(
                    functools.partial(self.on_upload_segment_image, seg_index, keyframe_path, img_preview_label,
                                      upload_button, delete_image_button)
                )
                delete_image_button.clicked.connect(
                    functools.partial(self.on_delete_segment_image, seg_index, keyframe_path, img_preview_label,
                                      upload_button, delete_image_button)
                )

                form_layout.addRow(label, row_container)

        except Exception as e_load_ui:
            form_layout.addRow(QtWidgets.QLabel(f"UI 빌드 중 오류 발생:\n{e_load_ui}\n\n{traceback.format_exc()}"))
            self.btn_update.setEnabled(False)
            self.btn_ai_request.setEnabled(False)

    def on_upload_segment_image(self, seg_index: int, target_path: Path,
                                preview_label: ClickableLabel, upload_button: QtWidgets.QPushButton,
                                delete_button: QtWidgets.QPushButton):
        """(v57 최종) 파일 리사이즈 로직 완전 삭제 / 썸네일 에러 제거"""


        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
        except:
            pass

        src_path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"'{self.scene_id}' 세그먼트 {seg_index + 1} 이미지", str(target_path.parent),
            "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if not src_path_str: return

        main_win = self.parent()
        while main_win and not hasattr(main_win, "cmb_img_w"):
            main_win = main_win.parent()

        target_w, target_h = 720, 1280
        if main_win:
            try:
                target_w = int(main_win.cmb_img_w.currentData())
                target_h = int(main_win.cmb_img_h.currentData())
            except:
                pass

        def job(progress_callback):
            _log = lambda msg: progress_callback({"msg": msg})

            wf_path = Path(JSONS_DIR) / "only_faceswap.json"
            if not wf_path.exists(): raise FileNotFoundError("워크플로우 없음")

            workflow = load_json(wf_path)

            comfy_in = Path(COMFY_INPUT_DIR)
            src_path = Path(src_path_str)
            tmp_name = f"raw_seg_{self.scene_id}_{seg_index}_{src_path.name}"
            shutil.copy2(str(src_path), comfy_in / tmp_name)

            if "2" in workflow: workflow["2"]["inputs"]["image"] = tmp_name
            if "11" in workflow:
                workflow["11"]["inputs"]["width"] = target_w
                workflow["11"]["inputs"]["height"] = target_h

            chars = self.scene_data.get("characters", [])
            char_map = {}
            for c in chars:
                cid, cidx = "", 0
                if isinstance(c, dict):
                    cid, cidx = c.get("id", ""), int(c.get("index", 0) or 0)
                elif isinstance(c, str):
                    if ":" in c:
                        p = c.split(":")
                        cid, cidx = p[0].strip(), int(p[1].strip())
                    else:
                        cid, cidx = c.strip(), 0
                if cid: char_map[cidx] = cid

            reactor_setup = {0: {"r": "8", "l": "3"}, 1: {"r": "7", "l": "10"}, 2: {"r": "6", "l": "9"}}
            char_base = Path(CHARACTER_DIR)
            for idx, nodes in reactor_setup.items():
                rid, lid = nodes["r"], nodes["l"]
                if idx in char_map:
                    char_id = char_map[idx]
                    c_img_path = None
                    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                        p = char_base / f"{char_id}{ext}"
                        if p.exists(): c_img_path = p; break
                    if c_img_path:
                        c_dst_name = f"char_{char_id}{c_img_path.suffix}"
                        shutil.copy2(str(c_img_path), comfy_in / c_dst_name)
                        if rid in workflow: workflow[rid]["inputs"]["enabled"] = True
                        if lid in workflow: workflow[lid]["inputs"]["image"] = c_dst_name
                    else:
                        if rid in workflow: workflow[rid]["inputs"]["enabled"] = False
                else:
                    if rid in workflow: workflow[rid]["inputs"]["enabled"] = False

            if "5" in workflow: workflow["5"]["inputs"]["filename_prefix"] = f"seg_proc/{self.scene_id}_{seg_index}"

            _log("ComfyUI 실행...")
            result = _submit_and_wait_comfy(COMFY_HOST, workflow, timeout=60, poll=0.5)

            outputs = result.get("outputs", {}).get("5", {}).get("images", [])
            if not outputs: raise RuntimeError("결과 없음")

            info = outputs[0]
            resp = requests.get(f"{COMFY_HOST}/view", params={
                "filename": info.get("filename"), "subfolder": info.get("subfolder"), "type": "output"
            })
            resp.raise_for_status()

            with open(target_path, "wb") as f:
                f.write(resp.content)
            return str(target_path)

        def done(ok, payload, err):
            if not ok:
                QtWidgets.QMessageBox.critical(self, "오류", f"실패:\n{err}")
                return

            final_path = str(payload)
            pixmap = QtGui.QPixmap(final_path)
            if not pixmap.isNull():
                # [수정] 옵션 없이 크기만 지정 (에러 방지 최우선)
                pixmap_scaled = pixmap.scaled(self.THUMBNAIL_SIZE, self.THUMBNAIL_SIZE)
                preview_label.setPixmap(pixmap_scaled)
                preview_label.setText("")
                preview_label.setToolTip(f"경로: {final_path}\n(자동 처리됨)")
                upload_button.setText("이미지 변경")
                delete_button.setEnabled(True)

                try:
                    preview_label.clicked.disconnect()
                except:
                    pass
                preview_label.clicked.connect(lambda: self.show_large_image(final_path))

            QtWidgets.QMessageBox.information(self, "완료", "처리 완료")

        run_job_with_progress_async(self, f"세그먼트 처리 ({seg_index + 1})", job, on_done=done)

    def on_delete_segment_image(self, seg_index: int, image_path: Path,
                                preview_label: ClickableLabel, upload_button: QtWidgets.QPushButton,
                                delete_button: QtWidgets.QPushButton):
        """[신규] 세그먼트 키프레임 이미지 삭제"""
        if not image_path.exists():
            QtWidgets.QMessageBox.warning(self, "오류", "파일이 이미 존재하지 않습니다.")

        reply = QtWidgets.QMessageBox.question(
            self, "이미지 삭제 확인", f"정말로 이 세그먼트의 키프레임 이미지를 삭제하시겠습니까?\n\n{image_path.name}",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No
        )
        if reply != QtWidgets.QMessageBox.Yes: return

        try:
            if image_path.exists(): image_path.unlink()
            QtWidgets.QMessageBox.information(self, "삭제 완료", f"파일을 삭제했습니다:\n{image_path.name}")

            preview_label.setPixmap(QtGui.QPixmap())
            preview_label.setText("[이미지\n없음]")
            preview_label.setToolTip(f"파일이 삭제되었습니다. (경로: {image_path})")
            preview_label.setStyleSheet("border: 1px dashed gray; color: gray;")
            try:
                preview_label.clicked.disconnect()
            except (TypeError, RuntimeError):
                pass
            upload_button.setText("업로드")
            delete_button.setEnabled(False)
        except FileNotFoundError:
            QtWidgets.QMessageBox.warning(self, "오류", "파일이 이미 존재하지 않았습니다.")
            delete_button.setEnabled(False)
        except (OSError, Exception) as e_delete:
            QtWidgets.QMessageBox.critical(self, "삭제 실패", f"파일 삭제 중 오류가 발생했습니다:\n{e_delete}")


    def on_update_and_close(self):
        """
        [수정됨] '업데이트' 버튼. 수정한 prompt_movie와 direct_prompt를 video.json에 저장
        (창을 닫지 않습니다.)
        """
        try:
            # 1. prompt_movie 업데이트
            updated_prompts = 0
            for seg_index, text_edit in self.widget_map_prompt:
                if seg_index < len(self.frame_segments_data):
                    new_prompt = text_edit.toPlainText().strip()
                    segment = self.frame_segments_data[seg_index]
                    if segment.get("prompt_movie", "") != new_prompt:
                        segment["prompt_movie"] = new_prompt
                        updated_prompts += 1

            # 2. direct_prompt 업데이트
            updated_direct = 0
            for seg_index, text_edit in self.widget_map_direct:
                if seg_index < len(self.frame_segments_data):
                    new_prompt = text_edit.toPlainText().strip()
                    segment = self.frame_segments_data[seg_index]
                    if segment.get("direct_prompt", "") != new_prompt:
                        segment["direct_prompt"] = new_prompt
                        updated_direct += 1

            # 3. 씬 데이터에 세그먼트 목록 다시 쓰기
            self.scene_data["frame_segments"] = self.frame_segments_data

            # 4. full_video_data에서 씬 찾아서 교체
            scene_list = self.full_video_data.get("scenes", [])
            for i, scene in enumerate(scene_list):
                if isinstance(scene, dict) and scene.get("id") == self.scene_id:
                    scene_list[i] = self.scene_data  # 수정된 씬 객체로 교체
                    break
            self.full_video_data["scenes"] = scene_list

            # 5. 파일 저장
            save_json(self.json_path, self.full_video_data)

            QtWidgets.QMessageBox.information(self, "업데이트 완료",
                                              f"[{self.scene_id}] 씬의 세그먼트가 저장되었습니다:\n\n"
                                              f"- 행동 묘사(prompt_movie) {updated_prompts}개 업데이트됨\n"
                                              f"- Direct Prompt {updated_direct}개 업데이트됨\n\n"
                                              f"(창은 닫히지 않았습니다. '닫기' 버튼으로 종료하세요.)")

            # self.accept() # <-- [핵심 수정] 이 줄을 삭제/주석 처리하여 창이 닫히지 않게 함

        except Exception as e_update:
            QtWidgets.QMessageBox.critical(self, "저장 오류", f"파일을 저장하는 중 오류가 발생했습니다:\n{e_update}")

    # shorts_ui.py의 SegmentEditDialog 클래스 내부
    # on_ai_request_segment 함수를 이 코드로 덮어쓰세요.

    def on_ai_request_segment(self):
            """[신규] 세그먼트 수정 창의 'AI 요청' 버튼"""


            # 1. Direct Prompt 읽기 (하나라도 있어야 함)
            direct_prompts_to_process: List[Tuple[int, str]] = []
            for seg_index, text_edit_widget in self.widget_map_direct:
                direct_prompt_text = text_edit_widget.toPlainText().strip()
                if direct_prompt_text:
                    direct_prompts_to_process.append((seg_index, direct_prompt_text))

            if not direct_prompts_to_process:
                QtWidgets.QMessageBox.information(self, "알림", "AI로 요청할 'Direct Prompt' 내용이 없습니다.")
                return

            # (FPS 수정은 ScenePromptEditDialog에서만 하므로 여기서는 제외)

            # 2. 원본 분위기 (project.json) 로드
            original_vibe_prompt = ""
            try:
                pj_path = self.json_path.parent / "project.json"
                if pj_path.exists():
                    pj_doc = load_json(pj_path, {}) or {}
                    if isinstance(pj_doc, dict):
                        original_vibe_prompt = pj_doc.get("prompt_user") or pj_doc.get("prompt", "")
            except Exception as e_load_pj:
                print(f"[JSON Edit] project.json 로드 실패: {e_load_pj}")

            # 3. 다음 씬 가사 찾기
            next_scene_lyric = "(Scene End)"
            scene_list = self.full_video_data.get("scenes", [])
            current_index = -1
            for idx, s in enumerate(scene_list):
                if isinstance(s, dict) and s.get("id") == self.scene_id:
                    current_index = idx
                    break
            if current_index != -1 and current_index + 1 < len(scene_list):
                next_sc = scene_list[current_index + 1]
                if isinstance(next_sc, dict):
                    next_scene_lyric = next_sc.get("lyric", "") or "(Next scene has no lyric)"

            # 4. 버튼 잠그기
            self.btn_ai_request.setEnabled(False)
            self.btn_update.setEnabled(False)
            self.btn_cancel.setEnabled(False)

            # 5. Job 정의
            def job(progress_callback):
                _log = lambda msg: progress_callback({"msg": msg})
                _log(f"[{self.scene_id}] 씬의 {len(direct_prompts_to_process)}개 세그먼트에 대해 AI 요청...")

                # [수정된 AI 시스템 프롬프트 (뮤직비디오 감독)]
                # --- ▼▼▼ [수정된 AI 시스템 프롬프트 (뮤직비디오 감독)] ▼▼▼ ---
                base_system_prompt = (
                    "You are a creative Music Video Director.\n"
                    "Your most important goal is to create dynamic, cinematic prompts for an AI video model that match the **lyrics**.\n"
                    "Avoid static, mannequin-like images.\n\n"
                    "[Context Provided]\n"
                    "1. `original_vibe`: The overall theme of the entire song.\n"
                    "2. `scene_lyric`: The lyric for THIS scene (THIS IS THE MOST IMPORTANT).\n"
                    "3. `base_visual` (direct_prompt): The user's concept text for THIS scene.\n"
                    "   - It may contain explicit POSE / ACTION instructions (e.g. \"both arms up\", \"jumping\", \"hugging\").\n"
                    "4. `characters`: The characters in THIS scene (e.g., 'female_01').\n"
                    "5. `time_structure`: The frame segments for THIS scene (e.g., [\"0-65f\", \"49-125f\"]).\n"
                    "6. `next_scene_lyric`: The lyric for the *next* scene (for transition context).\n\n"
                    "[Your Task (Return JSON ONLY)]\n"
                    "1. \"prompt_ko\": Korean description of the whole scene (based on all context).\n"
                    "2. \"prompt_img_base\": English, comma-separated visual tags for the whole scene (5-12 words).\n"
                    "3. \"motion_hint\": short English motion/camera hint (e.g. \"slow zoom in\"). Can be \"\".\n"
                    "4. \"segment_prompts\": an array of English scene descriptions.\n"
                    "   The array length MUST exactly match the `time_structure` list length.\n\n"
                    "[!! CRITICAL RULES !!]\n"
                    "1.  **Hard Rule for base_visual (POSE/ACTION):**\n"
                    "    - If `base_visual` contains any explicit pose/action from the user\n"
                    "      (e.g. \"raise both arms up\", \"arms wide open\", \"kneeling down\"),\n"
                    "      you MUST preserve that pose/action in ALL `segment_prompts` descriptions\n"
                    "      unless it clearly contradicts the scene_lyric.\n"
                    "    - NEVER ignore or replace a user-specified pose/action from `base_visual`.\n"
                    "    - You may ADD extra motion around that pose (camera moves, small variations),\n"
                    "      but the core pose (e.g. \"both arms up\") must remain.\n"
                    "2.  **New Action:** Based on the lyrics, design specific actions and motion\n"
                    "    for the `characters` in each `segment_prompts` description.\n"
                    "    (Examples: \"female_01 walks forward through falling leaves\",\n"
                    "              \"female_01 spins around, throwing leaves in the air\").\n"
                    "    These actions must still respect any explicit pose from `base_visual`.\n"
                    "3.  **Background:** Use `base_visual` as the base for the background/setting\n"
                    "    (location, time of day, mood). You may evolve it gradually\n"
                    "    (e.g., \"The red autumn background becomes darker\"),\n"
                    "    but do not contradict explicit user instructions.\n"
                    "4.  **Progression:** Design the actions to be continuous and logical\n"
                    "    across segments, telling a small story that matches the lyric's emotion.\n"
                    "5.  **Camera:** Include dynamic camera work in the segment prompts when possible\n"
                    "    (e.g. \"slow zoom towards her face\", \"camera rotates around her\",\n"
                    "          \"from a top-down view\").\n"
                    "6.  **Emotion:** Describe the character's expression\n"
                    "    (e.g., \"smiling happily\", \"peaceful expression\").\n"
                    "7.  **Prohibition:** NO mannequins. Every segment description must indicate\n"
                    "    some change in the character's action, pose, camera, or emotion.\n"
                )
                # --- ▲▲▲ [수정 끝] ▲▲▲ ---

                updated_count = 0

                # 6. [수정] 세그먼트 루프 (Direct Prompt가 있는 것만)
                for seg_index, dp_text in direct_prompts_to_process:
                    if seg_index >= len(self.frame_segments_data): continue

                    segment_data = self.frame_segments_data[seg_index]
                    scene_id = self.scene_id
                    seg_label = f"{scene_id} (Seg {seg_index + 1})"

                    frame_range_info = f"{segment_data.get('start_frame')}-{segment_data.get('end_frame')}f"

                    _log(f"[{seg_label}] AI 요청 중... (Direct Prompt: {dp_text[:30]}...)")

                    user_prompt_payload = {
                        "original_vibe": original_vibe_prompt,
                        "scene_lyric": self.scene_data.get("lyric", ""),  # 씬 전체 가사
                        "base_visual": dp_text,  # 세그먼트의 Direct Prompt
                        "characters": self.scene_data.get("characters", []),
                        "time_structure": frame_range_info,  # 세그먼트 1개의 시간
                        "next_scene_lyric": next_scene_lyric
                    }
                    user_prompt = json.dumps(user_prompt_payload, ensure_ascii=False)

                    try:
                        ai_raw = self.ai_instance.ask_smart(
                            base_system_prompt, user_prompt,
                            prefer="gemini", allow_fallback=True,
                        )
                        json_start = ai_raw.find("{")
                        json_end = ai_raw.rfind("}") + 1
                        if not (0 <= json_start < json_end):
                            _log(f"[{seg_label}] AI가 JSON을 반환하지 않았습니다.")
                            continue

                        ai_json = json.loads(ai_raw[json_start:json_end])

                        # [수정] 여러 형태의 응답을 허용하는 segment 프롬프트 추출 로직
                        new_prompt_movie = ""

                        # 1) 권장 키: "segment_prompt"
                        val = ai_json.get("segment_prompt")
                        if isinstance(val, str):
                            new_prompt_movie = val.strip()

                        # 2) 배열 형태 호환: "segment_prompts": ["...", ...] → 첫 번째 사용
                        if not new_prompt_movie:
                            seg_list = ai_json.get("segment_prompts")
                            if isinstance(seg_list, list) and seg_list:
                                first = seg_list[0]
                                if isinstance(first, str):
                                    new_prompt_movie = first.strip()

                        # 3) 기타 키 호환: "prompt_movie", "prompt"
                        if not new_prompt_movie:
                            for alt_key in ("prompt_movie", "prompt"):
                                alt_val = ai_json.get(alt_key)
                                if isinstance(alt_val, str) and alt_val.strip():
                                    new_prompt_movie = alt_val.strip()
                                    break

                        # 4) dict에 필드가 하나뿐이고 그 값이 문자열이면 그것도 허용
                        if not new_prompt_movie and isinstance(ai_json, dict) and len(ai_json) == 1:
                            only_val = next(iter(ai_json.values()))
                            if isinstance(only_val, str):
                                new_prompt_movie = only_val.strip()

                        if new_prompt_movie:
                            # [수정] 메모리(self.frame_segments_data)에 즉시 반영
                            segment_data["prompt_movie"] = new_prompt_movie
                            updated_count += 1
                            _log(f"[{seg_label}] AI 행동 묘사 갱신 완료.")
                        else:
                            _log(f"[{seg_label}] AI가 사용 가능한 행동 묘사를 반환하지 않았습니다.")


                    except Exception as e_ai:
                        _log(f"[{seg_label}] AI 호출 실패: {e_ai}")
                        continue

                # --- ▼▼▼ [핵심 수정] AI 작업 완료 후 즉시 저장 로직 추가 ▼▼▼ ---
                if updated_count > 0:
                    _log(f"AI가 갱신한 {updated_count}개 내용을 video.json에 저장합니다...")
                    try:
                        # 1. 씬 데이터에 세그먼트 목록 다시 쓰기
                        self.scene_data["frame_segments"] = self.frame_segments_data

                        # 2. full_video_data에서 씬 찾아서 교체
                        scene_list_local = self.full_video_data.get("scenes", [])
                        for i, scene_local in enumerate(scene_list_local):
                            if isinstance(scene_local, dict) and scene_local.get("id") == self.scene_id:
                                scene_list_local[i] = self.scene_data  # 수정된 씬 객체로 교체
                                break
                        self.full_video_data["scenes"] = scene_list_local

                        # 3. 파일 저장
                        save_json(self.json_path, self.full_video_data)
                        _log("저장 완료.")

                    except Exception as e_save:
                        _log(f"[ERROR] AI 요청 후 저장 실패: {e_save}")
                # --- ▲▲▲ [수정 끝] ▲▲▲ ---

                return {"updated_count": updated_count}

            # 7. Done 콜백
            def done(ok, payload, err):
                self.btn_ai_request.setEnabled(True)
                self.btn_update.setEnabled(True)
                self.btn_cancel.setEnabled(True)

                if not ok:
                    QtWidgets.QMessageBox.critical(self, "AI 요청 실패", f"작업 중 오류가 발생했습니다:\n{err}")
                    return

                count = (payload or {}).get("updated_count", 0)
                if count > 0:
                    # AI가 갱신한 데이터를 UI 텍스트 상자에 다시 로드
                    self.reload_prompts_from_data()
                    QtWidgets.QMessageBox.information(self, "AI 요청 완료",
                                                      f"총 {count}개 세그먼트의 'prompt_movie'를 갱신하고 파일에 즉시 저장했습니다.")
                else:
                    QtWidgets.QMessageBox.warning(self, "AI 요청", "AI가 갱신한 내용이 없습니다.")

            run_job_with_progress_async(
                owner=self,
                title=f"AI 세그먼트 프롬프트 생성 중 ({self.scene_id})",
                job=job,
                on_done=done,
            )

    def reload_prompts_from_data(self):
        """[신규] AI 작업 후, self.frame_segments_data의 최신 'prompt_movie' 값을 UI에 다시 로드"""
        for seg_index, text_edit in self.widget_map_prompt:
            if seg_index < len(self.frame_segments_data):
                new_prompt = self.frame_segments_data[seg_index].get("prompt_movie", "")
                text_edit.setPlainText(new_prompt)

    # (show_large_image는 ScenePromptEditDialog의 것을 그대로 사용)
    def show_large_image(self, path_str: str):
        """썸네일 클릭 시 800x800 박스에 맞게 이미지 표시"""
        if not path_str:
            QtWidgets.QMessageBox.information(self, "미리보기", "이 씬에는 이미지 경로가 지정되지 않았습니다.")
            return

        pixmap = QtGui.QPixmap(path_str)
        if pixmap.isNull():
            QtWidgets.QMessageBox.warning(self, "미리보기 오류", f"이미지를 불러올 수 없습니다:\n{path_str}")
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"이미지 미리보기: {Path(path_str).name}")
        label = QtWidgets.QLabel()
        max_preview_w, max_preview_h = 800, 800
        img_w, img_h = pixmap.width(), pixmap.height()

        if img_w > max_preview_w or img_h > max_preview_h:
            scaled_pixmap = pixmap.scaled(
                max_preview_w, max_preview_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
            dialog.resize(scaled_pixmap.width() + 20, scaled_pixmap.height() + 20)
        else:
            label.setPixmap(pixmap)
            dialog.resize(img_w + 20, img_h + 20)

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(label)
        dialog.exec_()

# ───────────────────────────── 실행 진입점 ─────────────────────────────


def create_shorts_widget(parent=None):
    """
    all_ui.py에서 탭으로 불러 쓸 때 사용하는 진입점.
    QMainWindow를 그냥 위젯처럼 돌려준다.
    """
    win = MainWindow()
    if parent is not None:
        win.setParent(parent)
    return win




# (기존 main() 함수 시작)

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
