from __future__ import annotations
import os
import sys
import json
import shutil
import functools
import requests
import traceback
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt

# --- App 모듈 Import ---
from app.utils import load_json, save_json, run_job_with_progress_async, AI
from app.utils import _submit_and_wait as _submit_and_wait_comfy
from app.video_build import retry_cut_audio_for_scene
import app.settings as settings_mod

# --- 상수 편의 참조 ---
JSONS_DIR = settings_mod.JSONS_DIR
COMFY_INPUT_DIR = settings_mod.COMFY_INPUT_DIR
COMFY_HOST = settings_mod.COMFY_HOST
CHARACTER_DIR = getattr(settings_mod, "CHARACTER_DIR", r"C:\my_games\shorts_make\character")



# ─────────────────────────────────────────────────────────────────────────────
# [Helper Class] ClickableLabel
# ─────────────────────────────────────────────────────────────────────────────
class ClickableLabel(QtWidgets.QLabel):
    """클릭 시 'clicked' 시그널을 방출하는 QLabel"""
    clicked = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# [Dialog] SegmentEditDialog (세그먼트/키프레임 수정)
# ─────────────────────────────────────────────────────────────────────────────
class SegmentEditDialog(QtWidgets.QDialog):
    """
    [UPDATED]
    - 레거시: scene_data["frame_segments"]가 있으면 기존 방식 그대로 편집
    - 최신: frame_segments가 없고 prompt_1..N 구조(seg_count/prompt_1_kor 등)면
            쇼핑탭 '최종안 수정'처럼 (한글/영어 2칸) 편집 + 한글->영어 번역(AI 요청) 지원
    """
    _AI_DEFAULT_NEGATIVE_TAGS = (
        "lowres, bad anatomy, bad proportions, extra limbs, extra fingers, missing fingers, "
        "jpeg artifacts, signature, logo, nsfw, text, letters, typography, watermark"
    )

    def __init__(
        self,
        scene_id: str,
        scene_data: Dict[str, Any],
        full_video_data: Dict[str, Any],
        json_path: Path,
        ai_instance: AI,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)

        self.scene_id = scene_id
        self.scene_data = scene_data
        self.full_video_data = full_video_data
        self.json_path = json_path
        self.ai_instance = ai_instance

        # -------------------------
        # 모드 결정
        # -------------------------
        self.frame_segments_data: List[Dict[str, Any]] = self.scene_data.get("frame_segments", []) or []

        self._mode = "frame_segments" if self.frame_segments_data else "prompt_keys"

        # prompt_1..N 기반 편집용
        self._seg_count = self._infer_seg_count_from_scene(self.scene_data)
        self._seg_indices = list(range(1, self._seg_count + 1))

        # 위젯 맵
        self.widget_map_prompt: List[Tuple[int, QtWidgets.QTextEdit]] = []  # 레거시: prompt_movie
        self.widget_map_direct: List[Tuple[int, QtWidgets.QTextEdit]] = []  # 레거시: direct_prompt

        self.widget_map_kor: List[Tuple[int, QtWidgets.QTextEdit]] = []  # 최신: prompt_i_kor
        self.widget_map_en: List[Tuple[int, QtWidgets.QTextEdit]] = []   # 최신: prompt_i (EN)

        self.THUMBNAIL_SIZE = 150

        self.setWindowTitle(f"세그먼트 수정: [{self.scene_id}]")
        self.setMinimumSize(900, 700)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(2)

        page_scroll_area = QtWidgets.QScrollArea()
        page_scroll_area.setWidgetResizable(True)
        scroll_content_widget = QtWidgets.QWidget()
        self.form_layout_page = QtWidgets.QFormLayout(scroll_content_widget)
        self.form_layout_page.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapAllRows)
        self.form_layout_page.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.form_layout_page.setVerticalSpacing(8)
        page_scroll_area.setWidget(scroll_content_widget)

        main_layout.addWidget(page_scroll_area)

        button_layout = QtWidgets.QHBoxLayout()
        self.btn_ai_request = QtWidgets.QPushButton("AI 요청 (번역/동기화)")
        self.btn_update = QtWidgets.QPushButton("업데이트")
        self.btn_cancel = QtWidgets.QPushButton("닫기")

        if self._mode == "frame_segments":
            self.btn_ai_request.setToolTip(
                "레거시(frame_segments): 오른쪽 Direct Prompt를 기반으로\n"
                "AI에게 해당 세그먼트의 prompt_movie(행동 묘사)를 생성 요청합니다."
            )
        else:
            self.btn_ai_request.setToolTip(
                "최신(prompt_1..N): 한글(prompt_i_kor) 내용을 영어(prompt_i)로 번역합니다.\n"
                "(쇼핑탭 '최종안 수정'과 같은 목적)"
            )

        button_layout.addStretch(1)
        button_layout.addWidget(self.btn_cancel)
        button_layout.addWidget(self.btn_ai_request)
        button_layout.addWidget(self.btn_update)
        main_layout.addLayout(button_layout)

        self.btn_update.clicked.connect(self.on_update_and_close)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ai_request.clicked.connect(self.on_ai_request_segment)

        self.load_and_build_segments_ui(self.form_layout_page)

    # ------------------------------------------------------------------
    # 최신 구조(seg_count/prompt_1..N) 감지
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_seg_count_from_scene(sc: Dict[str, Any]) -> int:
        # 1) seg_count 우선
        try:
            v = sc.get("seg_count")
            if v is not None:
                n = int(v)
                if n > 0:
                    return n
        except Exception:
            pass

        # 2) prompt_1.. 키를 스캔
        max_i = 0
        try:
            for k in list(sc.keys()):
                m = re.match(r"^prompt_(\d+)$", str(k))
                if m:
                    max_i = max(max_i, int(m.group(1)))
        except Exception:
            pass
        return max_i

    # ------------------------------------------------------------------
    # 쇼핑탭 '최종안 수정'과 동일한 목적의 번역기
    # ------------------------------------------------------------------
    @staticmethod
    def _translate_kor_to_en(ai: AI, kor_text: str) -> str:
        kor_text = (kor_text or "").strip()
        if not kor_text:
            return ""

        if ai is None:
            return kor_text

        sys_msg = (
            "You are a professional translator for AI image/video generation prompts.\n"
            "Translate Korean to natural, concise English.\n"
            "Do NOT add new details. Keep meaning faithful.\n"
            "Output English ONLY (no explanations)."
        )
        user_msg = f'Korean:\n"{kor_text}"\n\nEnglish:'
        try:
            out = ai.ask_smart(sys_msg, user_msg, prefer="openai", allow_fallback=True)
            out = (out or "").strip()
            out = out.replace("```", "").strip()
            return out
        except Exception:
            return kor_text

    # ------------------------------------------------------------------
    # UI 빌드
    # ------------------------------------------------------------------
    def load_and_build_segments_ui(self, form_layout: QtWidgets.QFormLayout):
        try:
            base_imgs_dir = self.json_path.parent / "imgs"
            font = QtGui.QFont()
            font.setFamily("Courier" if "Courier" in QtGui.QFontDatabase().families() else "Monospace")
            font.setPointSize(10)

            # ─────────────────────────
            # [A] 레거시: frame_segments
            # ─────────────────────────
            if self._mode == "frame_segments":
                if not self.frame_segments_data:
                    form_layout.addRow(QtWidgets.QLabel(f"[{self.scene_id}] 씬에 'frame_segments' 데이터가 없습니다."))
                    return

                for seg_index, segment_data in enumerate(self.frame_segments_data):
                    if not isinstance(segment_data, dict):
                        continue

                    start_f = segment_data.get("start_frame", 0)
                    end_f = segment_data.get("end_frame", 0)
                    keyframe_id = f"kf_{seg_index + 1}"

                    label_text = (
                        f"<b>{self.scene_id} / 세그먼트 {seg_index + 1}</b> (키프레임: {keyframe_id}.png) | "
                        f"<b>프레임:</b> [{start_f} ~ {end_f}]"
                    )
                    label = QtWidgets.QLabel(label_text)
                    label.setWordWrap(False)

                    row_container = QtWidgets.QWidget()
                    row_layout = QtWidgets.QHBoxLayout(row_container)
                    row_layout.setContentsMargins(0, 0, 0, 0)
                    row_layout.setSpacing(8)

                    # 좌측: 이미지
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

                    prompt_movie_edit = QtWidgets.QTextEdit()
                    prompt_movie_edit.setPlainText(segment_data.get("prompt_movie", ""))
                    prompt_movie_edit.setFont(font)
                    prompt_movie_edit.setMinimumHeight(150)
                    prompt_movie_edit.setToolTip(f"[{keyframe_id}] 행동 묘사 프롬프트 (prompt_movie)")
                    prompt_movie_edit.setSizePolicy(
                        QtWidgets.QSizePolicy.Policy.Expanding,
                        QtWidgets.QSizePolicy.Policy.Expanding,
                    )
                    row_layout.addWidget(prompt_movie_edit, 1)

                    direct_prompt_edit = QtWidgets.QTextEdit()
                    direct_prompt_edit.setPlainText(segment_data.get("direct_prompt", ""))
                    direct_prompt_edit.setFont(font)
                    direct_prompt_edit.setMinimumHeight(150)
                    direct_prompt_edit.setToolTip(f"[{keyframe_id}] AI 요청 시 사용할 Direct Prompt")
                    direct_prompt_edit.setSizePolicy(
                        QtWidgets.QSizePolicy.Policy.Expanding,
                        QtWidgets.QSizePolicy.Policy.Expanding,
                    )
                    row_layout.addWidget(direct_prompt_edit, 1)

                    self.widget_map_prompt.append((seg_index, prompt_movie_edit))
                    self.widget_map_direct.append((seg_index, direct_prompt_edit))

                    keyframe_path = base_imgs_dir / self.scene_id / f"{keyframe_id}.png"
                    img_file_str = str(keyframe_path)
                    has_image = keyframe_path.exists()

                    if has_image:
                        pixmap = QtGui.QPixmap(img_file_str)
                        if not pixmap.isNull():
                            pixmap_scaled = pixmap.scaled(
                                self.THUMBNAIL_SIZE,
                                self.THUMBNAIL_SIZE,
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation,
                            )
                            img_preview_label.setPixmap(pixmap_scaled)
                            img_preview_label.setToolTip(f"경로: {img_file_str}\n(클릭해서 크게 보기)")
                            img_preview_label.clicked.connect(
                                functools.partial(self.show_large_image, img_file_str)
                            )
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

                    upload_button.clicked.connect(
                        functools.partial(
                            self.on_upload_segment_image,
                            seg_index,
                            keyframe_path,
                            img_preview_label,
                            upload_button,
                            delete_image_button,
                        )
                    )
                    delete_image_button.clicked.connect(
                        functools.partial(
                            self.on_delete_segment_image,
                            seg_index,
                            keyframe_path,
                            img_preview_label,
                            upload_button,
                            delete_image_button,
                        )
                    )

                    form_layout.addRow(label, row_container)

                return  # 레거시 UI 끝

            # ─────────────────────────
            # [B] 최신: prompt_1..N
            # ─────────────────────────
            if self._seg_count <= 0:
                form_layout.addRow(
                    QtWidgets.QLabel(
                        f"[{self.scene_id}] 씬에 세그먼트 데이터가 없습니다.\n"
                        f"- frame_segments 없음\n"
                        f"- seg_count/prompt_1.. 없음\n\n"
                        f"먼저 프로젝트분석(세그먼트 생성) 단계에서 prompt_1..N 또는 seg_count를 생성해야 합니다."
                    )
                )
                return

            # 안내 라벨
            info = QtWidgets.QLabel(
                "✅ 최신 세그먼트 편집 모드(prompt_1..N)\n"
                "- 왼쪽: 영어(prompt_i)\n"
                "- 오른쪽: 한글(prompt_i_kor)\n"
                "- 'AI 요청(번역/동기화)'를 누르면 한글을 영어로 자동 번역해 채웁니다."
            )
            info.setWordWrap(True)
            form_layout.addRow(info)

            for i in self._seg_indices:
                en_key = f"prompt_{i}"
                ko_key = f"{en_key}_kor"

                label = QtWidgets.QLabel(f"<b>{self.scene_id} / 세그먼트 {i}</b>")
                label.setWordWrap(False)

                row_container = QtWidgets.QWidget()
                row_layout = QtWidgets.QHBoxLayout(row_container)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(8)

                # 영어(왼쪽)
                en_edit = QtWidgets.QTextEdit()
                en_edit.setFont(font)
                en_edit.setMinimumHeight(140)
                en_edit.setPlainText(str(self.scene_data.get(en_key, "") or ""))
                en_edit.setToolTip(f"{en_key} (English)")
                row_layout.addWidget(en_edit, 1)

                # 한글(오른쪽)
                ko_edit = QtWidgets.QTextEdit()
                ko_edit.setFont(font)
                ko_edit.setMinimumHeight(140)
                ko_edit.setPlainText(str(self.scene_data.get(ko_key, "") or ""))
                ko_edit.setToolTip(f"{ko_key} (Korean)")
                row_layout.addWidget(ko_edit, 1)

                self.widget_map_en.append((i, en_edit))
                self.widget_map_kor.append((i, ko_edit))

                form_layout.addRow(label, row_container)

        except Exception as e_load_ui:
            form_layout.addRow(QtWidgets.QLabel(f"UI 빌드 중 오류 발생:\n{e_load_ui}\n\n{traceback.format_exc()}"))
            self.btn_update.setEnabled(False)
            self.btn_ai_request.setEnabled(False)

    # ------------------------------------------------------------------
    # 레거시 이미지 업로드/삭제/미리보기는 기존 코드 그대로 사용
    # (이 아래 3개 함수는 원본 SegmentEditDialog의 것을 그대로 둬도 됨)
    # ------------------------------------------------------------------
    def on_upload_segment_image(self, seg_index: int, target_path: Path,
                                preview_label: ClickableLabel, upload_button: QtWidgets.QPushButton,
                                delete_button: QtWidgets.QPushButton):
        # ✅ 원본 코드 그대로 (shorts_json_edit.py에 이미 있음)
        # (이 함수는 frame_segments 모드에서만 사용)
        return super().on_upload_segment_image(seg_index, target_path, preview_label, upload_button, delete_button)  # type: ignore

    def on_delete_segment_image(self, seg_index: int, image_path: Path,
                                preview_label: ClickableLabel, upload_button: QtWidgets.QPushButton,
                                delete_button: QtWidgets.QPushButton):
        return super().on_delete_segment_image(seg_index, image_path, preview_label, upload_button, delete_button)  # type: ignore

    def show_large_image(self, path_str: str):
        return super().show_large_image(path_str)  # type: ignore

    # ------------------------------------------------------------------
    # 저장(업데이트)
    # ------------------------------------------------------------------
    def on_update_and_close(self):
        try:
            # 레거시: frame_segments
            if self._mode == "frame_segments":
                updated_prompts = 0
                for seg_index, text_edit in self.widget_map_prompt:
                    if seg_index < len(self.frame_segments_data):
                        new_prompt = text_edit.toPlainText().strip()
                        segment = self.frame_segments_data[seg_index]
                        if segment.get("prompt_movie", "") != new_prompt:
                            segment["prompt_movie"] = new_prompt
                            updated_prompts += 1

                updated_direct = 0
                for seg_index, text_edit in self.widget_map_direct:
                    if seg_index < len(self.frame_segments_data):
                        new_prompt = text_edit.toPlainText().strip()
                        segment = self.frame_segments_data[seg_index]
                        if segment.get("direct_prompt", "") != new_prompt:
                            segment["direct_prompt"] = new_prompt
                            updated_direct += 1

                self.scene_data["frame_segments"] = self.frame_segments_data
                self._save_scene_back()

                QtWidgets.QMessageBox.information(
                    self,
                    "업데이트 완료",
                    f"[{self.scene_id}] 저장 완료\n"
                    f"- prompt_movie {updated_prompts}개\n"
                    f"- direct_prompt {updated_direct}개\n\n"
                    f"(창은 닫히지 않았습니다. '닫기'로 종료하세요.)"
                )
                return

            # 최신: prompt_1..N
            updated_en = 0
            updated_ko = 0

            for i, en_edit in self.widget_map_en:
                key = f"prompt_{i}"
                new_val = en_edit.toPlainText().strip()
                if str(self.scene_data.get(key, "") or "") != new_val:
                    self.scene_data[key] = new_val
                    updated_en += 1

            for i, ko_edit in self.widget_map_kor:
                key = f"prompt_{i}_kor"
                new_val = ko_edit.toPlainText().strip()
                if str(self.scene_data.get(key, "") or "") != new_val:
                    self.scene_data[key] = new_val
                    updated_ko += 1

            # seg_count도 보정
            self.scene_data["seg_count"] = self._seg_count

            self._save_scene_back()

            QtWidgets.QMessageBox.information(
                self,
                "업데이트 완료",
                f"[{self.scene_id}] 저장 완료\n"
                f"- 영어(prompt_i) {updated_en}개 변경\n"
                f"- 한글(prompt_i_kor) {updated_ko}개 변경\n\n"
                f"(창은 닫히지 않았습니다. '닫기'로 종료하세요.)"
            )

        except Exception as e_update:
            QtWidgets.QMessageBox.critical(self, "저장 오류", f"파일을 저장하는 중 오류가 발생했습니다:\n{e_update}")

    def _save_scene_back(self) -> None:
        scene_list = self.full_video_data.get("scenes", []) or []
        for idx, s in enumerate(scene_list):
            if isinstance(s, dict) and str(s.get("id", "")).strip() == self.scene_id:
                scene_list[idx] = self.scene_data
                break
        self.full_video_data["scenes"] = scene_list
        save_json(self.json_path, self.full_video_data)

    # ------------------------------------------------------------------
    # AI 요청 버튼 동작
    # ------------------------------------------------------------------
    def on_ai_request_segment(self):
        # 레거시(frame_segments): 기존 AI 행동묘사 생성 로직 그대로 사용 (원본 함수 유지)
        if self._mode == "frame_segments":
            return super().on_ai_request_segment()  # type: ignore

        # 최신(prompt_1..N): 쇼핑탭 최종안 수정처럼 "한글 -> 영어 번역"만 수행
        if self._seg_count <= 0:
            QtWidgets.QMessageBox.information(self, "알림", "번역할 세그먼트가 없습니다.")
            return

        # 현재 UI의 한글 텍스트 수집
        kor_items: List[Tuple[int, str]] = []
        for i, ko_edit in self.widget_map_kor:
            t = ko_edit.toPlainText().strip()
            if t:
                kor_items.append((i, t))

        if not kor_items:
            QtWidgets.QMessageBox.information(self, "알림", "번역할 한글(prompt_i_kor) 내용이 없습니다.")
            return

        self.btn_ai_request.setEnabled(False)
        self.btn_update.setEnabled(False)
        self.btn_cancel.setEnabled(False)

        def job(progress_callback):
            def _log(msg: str):
                try:
                    progress_callback({"msg": msg})
                except Exception:
                    pass

            _log(f"[{self.scene_id}] 한글→영어 번역 시작: {len(kor_items)}개")

            updated = 0
            for i, kor_text in kor_items:
                en = self._translate_kor_to_en(self.ai_instance, kor_text)
                en = (en or "").strip()
                if not en:
                    continue
                key = f"prompt_{i}"
                if str(self.scene_data.get(key, "") or "") != en:
                    self.scene_data[key] = en
                    updated += 1
                _log(f" - Seg {i}: translated")

            self.scene_data["seg_count"] = self._seg_count
            self._save_scene_back()

            return {"updated": updated}

        def done(ok, payload, err):
            self.btn_ai_request.setEnabled(True)
            self.btn_update.setEnabled(True)
            self.btn_cancel.setEnabled(True)

            if not ok:
                QtWidgets.QMessageBox.critical(self, "AI 요청 실패", f"오류:\n{err}")
                return

            updated = (payload or {}).get("updated", 0)
            self.reload_prompts_from_data()

            QtWidgets.QMessageBox.information(
                self,
                "완료",
                f"[{self.scene_id}] 번역 완료: 영어(prompt_i) {updated}개 갱신\n"
                f"(video.json에 즉시 저장됨)"
            )

        run_job_with_progress_async(
            owner=self,
            title=f"한글→영어 번역 중 ({self.scene_id})",
            job=job,
            on_done=done,
        )

    def reload_prompts_from_data(self):
        if self._mode == "frame_segments":
            # 레거시
            for seg_index, text_edit in self.widget_map_prompt:
                if seg_index < len(self.frame_segments_data):
                    new_prompt = self.frame_segments_data[seg_index].get("prompt_movie", "")
                    text_edit.setPlainText(new_prompt)
            return

        # 최신: prompt_1..N
        for i, en_edit in self.widget_map_en:
            en_edit.setPlainText(str(self.scene_data.get(f"prompt_{i}", "") or ""))
        for i, ko_edit in self.widget_map_kor:
            ko_edit.setPlainText(str(self.scene_data.get(f"prompt_{i}_kor", "") or ""))



# ─────────────────────────────────────────────────────────────────────────────
# [Dialog] ScenePromptEditDialog (메인 제이슨 수정 다이얼로그)
# ─────────────────────────────────────────────────────────────────────────────
class ScenePromptEditDialog(QtWidgets.QDialog):
    """
    video.json 편집기 (씬 단위). 캐릭터 스타일, Direct Prompt, 이미지 관리 등.
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
        self.THUMBNAIL_SIZE = 150
        self.scene_char_widgets: Dict[str, QtWidgets.QListWidget] = {}

        self.setWindowTitle(f"Scene 프롬프트 편집: {self.json_path.name}")
        self.setMinimumSize(900, 750)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(2)

        self.stacked_widget = QtWidgets.QStackedWidget()

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

        main_layout.addWidget(self.stacked_widget)
        main_layout.addLayout(pagination_layout)
        main_layout.addLayout(button_layout)

        self.btn_update.clicked.connect(self.on_update_and_close)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_prev.clicked.connect(self.on_prev_page)
        self.btn_next.clicked.connect(self.on_next_page)
        self.btn_ai_request.clicked.connect(self.on_ai_request)

        self.load_and_build_ui()

    @staticmethod
    def _reindex_char_list(list_widget: QtWidgets.QListWidget) -> None:
        current_items = []
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if item:
                char_id = item.text().split(":")[0]
                current_items.append(char_id)
        list_widget.clear()
        for new_index, char_id in enumerate(current_items):
            list_widget.addItem(f"{char_id}:{new_index}")

    def on_add_char_to_scene(self, list_widget: QtWidgets.QListWidget):
        if not hasattr(self, "style_widget_map") or not self.style_widget_map:
            QtWidgets.QMessageBox.warning(self, "오류", "전체 캐릭터 스타일 목록이 없습니다.")
            return

        master_char_ids = list(self.style_widget_map.keys())
        currently_in_scene = set()
        for i in range(list_widget.count()):
            item_text = list_widget.item(i).text()
            currently_in_scene.add(item_text.split(":")[0])

        available_to_add = [cid for cid in master_char_ids if cid not in currently_in_scene]
        if not available_to_add:
            QtWidgets.QMessageBox.information(self, "알림", "모든 마스터 캐릭터가 이미 이 씬에 추가되었습니다.")
            return

        char_id_to_add, ok = QtWidgets.QInputDialog.getItem(
            self, "씬에 캐릭터 추가", "추가할 캐릭터를 선택하세요:", available_to_add, 0, False
        )

        if ok and char_id_to_add:
            next_index = list_widget.count()
            list_widget.addItem(f"{char_id_to_add}:{next_index}")

    def on_del_char_from_scene(self, list_widget: QtWidgets.QListWidget):
        selected_item = list_widget.currentItem()
        if not selected_item:
            QtWidgets.QMessageBox.warning(self, "오류", "삭제할 캐릭터를 씬 캐릭터 목록에서 선택하세요.")
            return
        row = list_widget.row(selected_item)
        list_widget.takeItem(row)
        self._reindex_char_list(list_widget)

    def show_large_image(self, path_str: str):
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
                max_preview_w, max_preview_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
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
        if not hasattr(self, "character_styles_form_layout"): return
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
        start_dir = CHARACTER_DIR
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
        if char_id not in self.style_widget_map: return
        reply = QtWidgets.QMessageBox.question(
            self, "캐릭터 삭제 확인",
            f"'{char_id}' 캐릭터를 스타일 목록에서 삭제하시겠습니까?\n\n(참고: '업데이트' 버튼을 눌러야 저장됩니다)",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No
        )
        if reply != QtWidgets.QMessageBox.Yes: return
        label_widget, field_container_widget = self.style_widget_map[char_id]
        self.character_styles_form_layout.removeRow(label_widget)
        del self.style_widget_map[char_id]

    def _build_character_styles_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("전체 등장 캐릭터 스타일")
        main_v_layout = QtWidgets.QVBoxLayout(group)
        main_v_layout.setContentsMargins(2, 0, 2, 0)
        main_v_layout.setSpacing(0)
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(100)
        scroll_area.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        scroll_content_widget = QtWidgets.QWidget()
        scroll_content_widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self.character_styles_form_layout = QtWidgets.QFormLayout(scroll_content_widget)
        self.character_styles_form_layout.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapAllRows)
        self.character_styles_form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.character_styles_form_layout.setContentsMargins(0, 0, 0, 0)
        self.character_styles_form_layout.setVerticalSpacing(0)
        self.character_styles_form_layout.setHorizontalSpacing(4)
        scroll_area.setWidget(scroll_content_widget)
        main_v_layout.addWidget(scroll_area)
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setContentsMargins(0, 4, 0, 0)
        add_button = QtWidgets.QPushButton("캐릭터 추가")
        add_button.setToolTip(f"폴더[{CHARACTER_DIR}]에서 캐릭터 파일을 선택해 목록에 추가합니다.")
        add_button.clicked.connect(self.on_add_character_style)
        button_layout.addStretch(1)
        button_layout.addWidget(add_button)
        main_v_layout.addLayout(button_layout)
        self.style_widget_map: Dict[str, Tuple[QtWidgets.QLabel, QtWidgets.QWidget]] = {}
        styles_data = self.full_video_data.get("character_styles", {})
        if not styles_data:
            self.character_styles_form_layout.addRow(QtWidgets.QLabel("프로젝트에 character_styles 정보가 없습니다."))
        else:
            for char_id, style_text in styles_data.items():
                self._add_character_style_row(char_id, style_text)
        return group

    def on_ai_request(self):
        # 1) UI FPS → video.json 동기화 (기존 유지)
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

        # 2) original_vibe (project.json에서 가져오되, 실패해도 진행)
        original_vibe_prompt = ""
        try:
            pj_path = self.json_path.parent / "project.json"
            if pj_path.exists():
                pj_doc = load_json(pj_path, {}) or {}
                if isinstance(pj_doc, dict):
                    original_vibe_prompt = pj_doc.get("prompt_user") or pj_doc.get("prompt", "")
        except Exception as e_load_pj:
            print(f"[JSON Edit] project.json 로드 실패: {e_load_pj}")

        # 3) AI 요청 대상 씬 수집 (direct_prompt 입력된 씬만)
        scenes_to_process: list[tuple[dict, str]] = []
        scenes_map = {scene.get("id"): scene for scene in self.scenes_data if isinstance(scene, dict) and "id" in scene}
        for scene_id, text_edit_widget in self.widget_map:
            if scene_id in scenes_map:
                direct_prompt_text = text_edit_widget.toPlainText().strip()
                if direct_prompt_text:
                    scenes_to_process.append((scenes_map[scene_id], direct_prompt_text))

        if not scenes_to_process:
            QtWidgets.QMessageBox.information(
                self, "알림",
                "AI로 요청할 'direct_prompt' 내용이 없습니다.\n(UI의 FPS 설정값은 video.json에 저장됩니다.)"
            )
            try:
                save_json(self.json_path, self.full_video_data)
                print(f"[JSON Edit] 프롬프트는 비어있으나, FPS 값({self.full_video_data.get('fps')})을 저장했습니다.")
            except Exception as e_save_fps_only:
                print(f"[JSON Edit] FPS만 저장하는 데 실패: {e_save_fps_only}")
            return

        self.btn_ai_request.setEnabled(False)
        self.btn_update.setEnabled(False)
        self.btn_cancel.setEnabled(False)

        def _guess_gender_word(char_id: str) -> str:
            cid = (char_id or "").lower()
            if "female" in cid or "woman" in cid or "girl" in cid:
                return "woman"
            if "male" in cid or "man" in cid or "boy" in cid:
                return "man"
            return "person"

        def _build_slot_mapping_text(scene_chars, char_styles: dict) -> tuple[str, list[tuple[int, str, str]]]:
            """
            return:
              mapping_text: system prompt에 넣을 텍스트
              slot_items: [(image_num, char_id, gender_word), ...]
            """
            slot_map: dict[int, str] = {}

            for c in (scene_chars or []):
                cid = ""
                slot_idx = None

                if isinstance(c, str):
                    parts = c.split(":")
                    cid = parts[0].strip()
                    if len(parts) > 1:
                        try:
                            slot_idx = int(parts[1].strip())
                        except Exception:
                            slot_idx = 0
                    else:
                        slot_idx = 0

                elif isinstance(c, dict):
                    cid = str(c.get("id", "")).strip()
                    if "slot" in c:
                        try:
                            slot_idx = int(c.get("slot"))
                        except Exception:
                            slot_idx = 0
                    elif "index" in c:
                        try:
                            slot_idx = int(c.get("index", 0) or 0)
                        except Exception:
                            slot_idx = 0
                    else:
                        slot_idx = 0

                if cid and slot_idx is not None and slot_idx not in slot_map:
                    slot_map[slot_idx] = cid

            if not slot_map:
                return "", []

            lines = []
            slot_items = []
            for slot_idx in sorted(slot_map.keys()):
                cid = slot_map[slot_idx]
                image_num = slot_idx + 1
                style = char_styles.get(cid, "Unknown style")
                gender_word = _guess_gender_word(cid)
                slot_items.append((image_num, cid, gender_word))

                lines.append(f"   - **Image {image_num} Source**: '{cid}' ({style})")
                lines.append(
                    f"     -> RULE: When describing this character in English, you MUST refer to them as "
                    f"**'the {gender_word} from image {image_num}'** (NOT just '{gender_word}' or '{cid}')."
                )

            mapping_text = "\n".join(lines)
            return mapping_text, slot_items

        def _ensure_from_image(prompt: str, slot_items: list[tuple[int, str, str]]) -> str:
            """
            AI가 'from image X'를 빼먹어도 최소한 1줄에서 강제로 보정.
            - 이미 'from image'가 들어가 있으면 그대로 둠
            - 없으면 "the {gender} from image {X}"를 앞에 프리픽스로 삽입
            """
            p = (prompt or "").strip()
            if not p:
                return p
            low = p.lower()
            if "from image" in low:
                return p

            if not slot_items:
                return p

            # 여러 캐릭이면 "the ... from image 1 and the ... from image 2" 형태로 앞에 붙임
            refs = []
            for image_num, _cid, gender_word in slot_items[:3]:
                refs.append(f"the {gender_word} from image {image_num}")

            if len(refs) == 1:
                prefix = refs[0]
            else:
                prefix = " and ".join(refs)

            # 기존 프롬프트가 "woman on rooftop, ..." 같은 태그형이어도 앞에 강제 삽입
            return f"{prefix}, {p}"

        def job(progress_callback):
            _log = lambda msg: progress_callback({"msg": msg})
            _log(f"총 {len(scenes_to_process)}개 씬에 대해 프롬프트를 AI로 갱신합니다.")

            quality_tags = self._AI_QUALITY_TAGS
            default_negative_tags = self._AI_DEFAULT_NEGATIVE_TAGS
            updated_count = 0

            # 🔥 핵심: 이제 on_ai_request는 '태그 5~12'가 아니라
            #         'prompt_img_core' / 'prompt_movie_core'를 "문장형"으로 받는다.
            base_system_prompt_template = (
                "You are a creative Music Video Director.\n"
                "Your most important goal is to create dynamic, cinematic prompts for an AI image/video model that match the **lyrics**.\n"
                "Avoid static, mannequin-like images.\n\n"
                "[Context Provided]\n"
                "1. `original_vibe`: The overall theme of the entire song.\n"
                "2. `scene_lyric`: The lyric for THIS scene (THIS IS THE MOST IMPORTANT).\n"
                "3. `base_visual` (direct_prompt): The user's concept text for THIS scene.\n"
                "   - Treat it as background/setting + anchor action if it contains explicit pose/action.\n"
                "4. `characters`: The characters in THIS scene (e.g., 'female_01:0').\n"
                "5. `time_structure`: The frame segments for THIS scene (e.g., [\"0-65f\", \"49-125f\"]).\n"
                "6. `next_scene_lyric`: The lyric for the *next* scene (for transition context).\n\n"
                "[Your Task (Return JSON ONLY)]\n"
                "1. \"prompt_ko\": Korean description of the whole scene.\n"
                "2. \"prompt_img_core\": English full image prompt sentence(s). MUST include character references using IMAGE SLOT MAPPING.\n"
                "3. \"motion_hint\": short English motion/camera hint (e.g. \"slow zoom in\"). Can be \"\".\n"
                "4. \"segment_prompts\": an array of English scene descriptions for each segment.\n"
                "   The array length MUST exactly match the `time_structure` list length.\n\n"
                "[!! CRITICAL RULES !!]\n"
                "1. If `base_visual` contains explicit pose/action from the user, you MUST preserve that pose/action in ALL segment_prompts unless it contradicts the lyric.\n"
                "2. For each segment_prompts item: describe a clear change of action/pose and camera (close-up, orbit, dolly, etc.).\n"
                "3. Every segment prompt must be action-driven. NO mannequin/static.\n"
                "4. You MUST follow IMAGE SLOT MAPPING when referring to characters in English.\n"
            )

            for scene_dict, dp_text in scenes_to_process:
                current_scene_id = scene_dict.get("id", "scene")
                frame_segments = scene_dict.get("frame_segments") or []
                seg_count = len(frame_segments)
                frame_ranges_info = [f"{s.get('start_frame')}-{s.get('end_frame')}f" for s in frame_segments]

                scene_lyric = scene_dict.get("lyric", "")
                characters = scene_dict.get("characters", []) or []

                next_scene_lyric = "(Scene End)"
                current_index = -1
                for idx, s in enumerate(self.scenes_data):
                    if isinstance(s, dict) and s.get("id") == current_scene_id:
                        current_index = idx
                        break
                if current_index != -1 and current_index + 1 < len(self.scenes_data):
                    next_sc = self.scenes_data[current_index + 1]
                    if isinstance(next_sc, dict):
                        next_scene_lyric = next_sc.get("lyric", "") or "(Next scene has no lyric)"

                # ✅ video.json에서 character_styles 읽어와 매핑 설명에 사용
                char_styles = self.full_video_data.get("character_styles", {}) or {}
                mapping_text, slot_items = _build_slot_mapping_text(characters, char_styles)

                system_prompt = base_system_prompt_template
                if mapping_text:
                    system_prompt += (
                        "\n[IMPORTANT: IMAGE SLOT MAPPING]\n"
                        "This workflow uses specific image slots for character images. You MUST follow these rules:\n"
                        f"{mapping_text}\n"
                        "- In English prompts (prompt_img_core, segment_prompts), DO NOT say just 'woman'/'man'.\n"
                        "- Always say: 'the woman/man from image X'.\n"
                    )

                _log(f"[{current_scene_id}] AI 요청 중... (segments={seg_count})")

                user_prompt_payload = {
                    "original_vibe": original_vibe_prompt,
                    "scene_lyric": scene_lyric,
                    "base_visual": dp_text,
                    "characters": characters,
                    "time_structure": frame_ranges_info,
                    "next_scene_lyric": next_scene_lyric,
                }
                user_prompt = json.dumps(user_prompt_payload, ensure_ascii=False)

                try:
                    ai_raw = self.ai_instance.ask_smart(
                        system_prompt,
                        user_prompt,
                        prefer="gemini",
                        allow_fallback=True,
                    )
                except Exception as e_ai:
                    _log(f"[{current_scene_id}] AI 호출 실패: {e_ai}")
                    continue

                json_start = ai_raw.find("{")
                json_end = ai_raw.rfind("}") + 1
                if not (0 <= json_start < json_end):
                    _log(f"[{current_scene_id}] AI가 JSON을 반환하지 않았습니다.")
                    continue

                try:
                    ai_json = json.loads(ai_raw[json_start:json_end])
                except Exception as e_json:
                    _log(f"[{current_scene_id}] JSON 파싱 실패: {e_json}")
                    continue

                prompt_ko = (ai_json.get("prompt_ko") or "").strip()
                prompt_img_core = (ai_json.get("prompt_img_core") or "").strip()
                motion_hint = (ai_json.get("motion_hint") or "").strip()
                seg_prompts = ai_json.get("segment_prompts", [])

                # 🔒 from image 누락 시 강제 보정
                prompt_img_core = _ensure_from_image(prompt_img_core, slot_items)

                # prompt / prompt_img / prompt_movie / negative 세팅
                if prompt_ko and prompt_img_core:
                    scene_dict["prompt"] = prompt_ko

                    # prompt_img / prompt_movie는 "문장형 + 퀄리티 태그"로 고정
                    scene_dict["prompt_img"] = f"{prompt_img_core}, {quality_tags}".strip().strip(",")
                    if motion_hint:
                        scene_dict[
                            "prompt_movie"] = f"{prompt_img_core}, {quality_tags}, motion: {motion_hint}".strip().strip(
                            ",")
                    else:
                        scene_dict["prompt_movie"] = scene_dict["prompt_img"]

                    scene_dict["prompt_negative"] = default_negative_tags
                    updated_count += 1
                    _log(f"[{current_scene_id}] 기본 프롬프트 갱신 완료")

                # 세그먼트별 prompt_movie 세팅
                if seg_count > 0:
                    filled = 0
                    if isinstance(seg_prompts, list) and len(seg_prompts) >= seg_count:
                        for i in range(seg_count):
                            seg_item = frame_segments[i]
                            prompt_text = seg_prompts[i]
                            if isinstance(prompt_text, dict):
                                prompt_text = prompt_text.get("prompt_movie") or prompt_text.get("text") or ""
                            prompt_text = str(prompt_text).strip()
                            if prompt_text:
                                # 세그먼트도 from image 보정
                                seg_item["prompt_movie"] = _ensure_from_image(prompt_text, slot_items)
                                filled += 1
                        _log(f"[{current_scene_id}] 세그먼트 묘사 {filled}/{seg_count}개 AI로 채움")
                    else:
                        # 응답이 부족하면 base로 채우되 from image 보정 유지
                        base_cmd = _ensure_from_image(dp_text, slot_items)
                        for seg_item in frame_segments:
                            seg_item["prompt_movie"] = base_cmd
                        _log(f"[{current_scene_id}] AI 세그먼트 묘사 응답 부족 → Direct Prompt로 일괄 채움")

                    scene_dict["frame_segments"] = frame_segments

            # 저장
            if updated_count > 0:
                _log("변경 내용을 video.json 에 저장합니다...")
                self.full_video_data["scenes"] = self.scenes_data
                try:
                    save_json(self.json_path, self.full_video_data)
                except Exception as e_save:
                    _log(f"video.json 저장 실패: {e_save}")

            return {"updated_count": updated_count}

        def done(ok, payload, err):
            self.btn_ai_request.setEnabled(True)
            self.btn_update.setEnabled(True)
            self.btn_cancel.setEnabled(True)
            if not ok:
                QtWidgets.QMessageBox.critical(self, "AI 요청 실패", f"작업 중 오류가 발생했습니다:\n{err}")
                return
            count = (payload or {}).get("updated_count", 0)
            if count > 0:
                QtWidgets.QMessageBox.information(
                    self, "AI 요청 완료",
                    f"총 {count}개 씬의 프롬프트를 갱신했습니다.\n(캐릭터는 'from image N' 규칙이 강제 적용됩니다.)"
                )
            else:
                QtWidgets.QMessageBox.warning(self, "AI 요청", "AI가 갱신한 내용이 없거나 저장에 실패했습니다.")

        run_job_with_progress_async(
            owner=self,
            title=f"AI 프롬프트 생성 중 ({self.json_path.name})",
            job=job,
            on_done=done,
        )

    def on_upload_image(self, scene_id: str, preview_label: QtWidgets.QLabel, upload_button: QtWidgets.QPushButton,
                        delete_image_button: QtWidgets.QPushButton, scene_data: Dict[str, Any]):
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
            comfy_in = Path(COMFY_INPUT_DIR)
            src_path = Path(src_path_str)
            temp_input_name = f"raw_{scene_id}_{src_path.name}"
            shutil.copy2(str(src_path), comfy_in / temp_input_name)

            if "2" in workflow: workflow["2"]["inputs"]["image"] = temp_input_name
            if "11" in workflow:
                workflow["11"]["inputs"]["width"] = target_w
                workflow["11"]["inputs"]["height"] = target_h

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

            with open(target_path, "wb") as f:
                f.write(resp.content)
            return str(target_path)

        def done(ok, payload, err):
            if not ok:
                QtWidgets.QMessageBox.critical(self, "오류", f"실패:\n{err}")
                return
            final_path = str(payload)
            scene_data["img_file"] = final_path
            pixmap = QtGui.QPixmap(final_path)
            if not pixmap.isNull():
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

    def on_open_segment_editor(self, scene_id: str, scene_data: Dict[str, Any]):
        try:
            dialog = SegmentEditDialog(
                scene_id=scene_id,
                scene_data=scene_data,
                full_video_data=self.full_video_data,
                json_path=self.json_path,
                ai_instance=self.ai_instance,
                parent=self
            )
            dialog.exec_()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "세그먼트 편집기 오류", f"세그먼트 편집기를 여는 중 오류가 발생했습니다:\n{e}")

    def load_and_build_ui(self):
        try:
            data = load_json(self.json_path, None)
            if not isinstance(data, dict):
                raise ValueError("video.json 파일의 형식이 올바르지 않습니다.")

            self.full_video_data = data
            self.scenes_data = self.full_video_data.get("scenes", [])
            if not isinstance(self.scenes_data, list):
                raise ValueError("video.json에 'scenes' 키가 없거나 리스트가 아닙니다.")

            top_char_layout = QtWidgets.QHBoxLayout()
            self.character_styles_group = self._build_character_styles_group()
            top_char_layout.addWidget(self.character_styles_group, 1)

            main_layout = self.layout()
            if main_layout and hasattr(main_layout, "insertLayout"):
                main_layout.insertLayout(0, top_char_layout)

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

                    scene_id = scene.get("id", "ID-없음")
                    lyric = (scene.get("lyric") or "").strip()
                    start_f = scene.get("start", 0.0)
                    end_f = scene.get("end", 0.0)
                    duration_f = scene.get("duration", 0.0)
                    if duration_f == 0.0 and end_f > start_f: duration_f = end_f - start_f
                    chars_list_label = scene.get("characters", [])
                    # chars_str = ", ".join(chars_list_label) if chars_list_label else "없음"
                    # label_text = (
                    #     f"<b>{scene_id}</b> [{lyric or '가사 없음'}] | "
                    #     f"<b>캐릭터:</b> [{chars_str}] | "
                    #     f"<b>시간:</b> [{start_f:.2f} ~ {end_f:.2f}, ({duration_f:.2f}s)]"
                    # )
                    # label = QtWidgets.QLabel(label_text)
                    # label.setWordWrap(False)
                    chars_str = ", ".join(chars_list_label) if chars_list_label else "없음"

                    # 기존의 단순 QLabel(label) 대신, 가로로 배치된 컨테이너 위젯(label_widget)을 만듭니다.
                    label_widget = QtWidgets.QWidget()
                    lw_layout = QtWidgets.QHBoxLayout(label_widget)
                    lw_layout.setContentsMargins(0, 0, 0, 0)
                    lw_layout.setSpacing(5)

                    # 1. ID
                    lw_layout.addWidget(QtWidgets.QLabel(f"<b>{scene_id}</b>"))

                    # 2. 가사 입력창
                    txt_lyric = QtWidgets.QLineEdit(scene.get("lyric", ""))
                    txt_lyric.setPlaceholderText("가사 없음")
                    # 스타일: 배경을 살짝 밝게 해서 입력창임을 강조
                    txt_lyric.setStyleSheet("background-color: #fff; color: #000; border: 1px solid #ccc;")
                    txt_lyric.setMinimumWidth(400)
                    lw_layout.addWidget(txt_lyric, 1)  # 1 = 늘어나게 설정

                    # 3. 수정 버튼
                    btn_lyric_save = QtWidgets.QPushButton("가사수정")
                    btn_lyric_save.setCursor(Qt.PointingHandCursor)
                    btn_lyric_save.setFixedWidth(100)
                    btn_lyric_save.setStyleSheet("background-color: #e1f5fe; color: #0277bd; font-weight: bold;")

                    # 클릭 시 동작
                    def _update_lyric_click(checked, s=scene, t=txt_lyric):
                        new_text = t.text().strip()
                        s["lyric"] = new_text
                        print(f"[{s.get('id')}] 가사 메모리 업데이트: {new_text}")
                        # ★ 버튼 누를 때 즉시 파일 저장을 원하면 아래 주석을 푸세요
                        # save_json(self.json_path, self.full_video_data)

                    btn_lyric_save.clicked.connect(_update_lyric_click)
                    lw_layout.addWidget(btn_lyric_save)

                    # 4. 기타 정보 (캐릭터, 시간)
                    info_text = f"| 캐릭터: [{chars_str}] | 시간: {start_f:.2f}~{end_f:.2f}"
                    lw_layout.addWidget(QtWidgets.QLabel(info_text))

                    # 5. 변수명 매핑 (아래 코드에서 label 변수를 쓰므로, 위젯을 label에 할당)
                    label = label_widget

                    row_container = QtWidgets.QWidget()
                    row_layout = QtWidgets.QHBoxLayout(row_container)
                    row_layout.setContentsMargins(0, 0, 0, 0)
                    row_layout.setSpacing(8)

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

                    text_edit = QtWidgets.QTextEdit()
                    text_edit.setPlainText(scene.get("direct_prompt", ""))
                    text_edit.setFont(font)
                    text_edit.setMinimumHeight(150)
                    text_edit.setToolTip(f"Scene ID: {scene_id}\n이 씬의 direct_prompt를 입력하세요.")
                    text_edit.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
                    row_layout.addWidget(text_edit, 1)

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
                    right_vbox.addWidget(btn_edit_segments)

                    if lyric:
                        btn_lipsync = QtWidgets.QPushButton("립싱크 모드: OFF")
                        btn_lipsync.setCheckable(True)
                        btn_lipsync.setToolTip("ON: lync_bool=true (립싱크 적용)\nOFF: lync_bool=false (기본)")
                        is_lipsync_on = scene.get("lync_bool", False)
                        btn_lipsync.setChecked(is_lipsync_on)
                        def update_lipsync_ui(checked, btn=btn_lipsync):
                            if checked:
                                btn.setText("립싱크 모드: ON")
                                btn.setStyleSheet("background-color: #d1e7dd; color: #0f5132; font-weight: bold;")
                            else:
                                btn.setText("립싱크 모드: OFF")
                                btn.setStyleSheet("")
                        update_lipsync_ui(is_lipsync_on)
                        txt_lync_prompt = QtWidgets.QLineEdit()
                        txt_lync_prompt.setPlaceholderText("립싱크 프롬프트 (예: sing a song)")
                        current_prompt = scene.get("lync_prompt", "")
                        txt_lync_prompt.setText(current_prompt)
                        def on_prompt_text_changed(text, s=scene):
                            s["lync_prompt"] = text.strip()
                        txt_lync_prompt.textChanged.connect(on_prompt_text_changed)
                        def on_lipsync_toggled(checked, s=scene, b=btn_lipsync, t=txt_lync_prompt):
                            s["lync_bool"] = checked
                            if checked:
                                current_text = t.text().strip()
                                if not current_text:
                                    default_val = "sing a song"
                                    t.setText(default_val)
                                    s["lync_prompt"] = default_val
                            update_lipsync_ui(checked, b)
                            try:
                                save_json(self.json_path, self.full_video_data)
                                print(f"[JSON Edit] {s.get('id')} lync_bool={checked}, prompt='{s.get('lync_prompt')}' (저장됨)")
                            except Exception as e:
                                print(f"[JSON Edit] 저장 실패: {e}")
                        btn_lipsync.toggled.connect(on_lipsync_toggled)
                        right_vbox.addWidget(btn_lipsync)
                        right_vbox.addWidget(txt_lync_prompt)

                        layout_trim = QtWidgets.QHBoxLayout()
                        layout_trim.setContentsMargins(0, 5, 0, 0)
                        layout_trim.setSpacing(5)
                        lbl_offset = QtWidgets.QLabel("싱크(초):")
                        lbl_offset.setFixedWidth(50)
                        spn_audio_offset = QtWidgets.QDoubleSpinBox()
                        spn_audio_offset.setRange(-5.0, 5.0)
                        spn_audio_offset.setSingleStep(0.01)
                        spn_audio_offset.setDecimals(3)
                        spn_audio_offset.setFixedWidth(70)
                        spn_audio_offset.setToolTip("양수(+): 시작을 늦춤 / 음수(-): 시작을 당김")
                        current_offset = float(scene.get("audio_offset", 0.0))
                        spn_audio_offset.setValue(current_offset)
                        def on_offset_changed(val, s=scene):
                            s["audio_offset"] = float(val)
                        spn_audio_offset.valueChanged.connect(on_offset_changed)
                        btn_trim_music = QtWidgets.QPushButton("음악 자르기")
                        btn_trim_music.setFixedWidth(120)
                        btn_trim_music.setToolTip("설정된 오프셋을 적용하여 이 씬의 오디오 파일(wav)을 다시 생성합니다.")
                        def on_click_trim_music(s_id=scene_id, s_data=scene, off_widget=spn_audio_offset):
                            try:
                                offset_val = off_widget.value()
                                s_data["audio_offset"] = float(offset_val)
                                project_root_path = self.json_path.parent
                                out_path = retry_cut_audio_for_scene(str(project_root_path), s_id, offset_val)
                                file_name = Path(out_path).name
                                print(f"[UI] 오디오 재성성 완료: {out_path} (Offset: {offset_val}s)")
                                QtWidgets.QMessageBox.information(
                                    self, "완료",
                                    f"오디오 파일을 새로 만들었습니다.\n\n경로: {file_name}\n적용된 싱크: {offset_val:+.3f}초"
                                )
                            except Exception as e:
                                print(f"[UI] 오디오 자르기 실패: {e}")
                                traceback.print_exc()
                                QtWidgets.QMessageBox.critical(self, "실패", f"오디오 자르기 중 오류가 발생했습니다:\n{e}")
                        btn_trim_music.clicked.connect(lambda checked: on_click_trim_music())
                        layout_trim.addWidget(lbl_offset)
                        layout_trim.addWidget(spn_audio_offset)
                        layout_trim.addWidget(btn_trim_music)
                        layout_trim.addStretch(1)
                        right_vbox.addLayout(layout_trim)

                    right_vbox.addStretch(1)
                    row_layout.addWidget(right_vbox_widget, 0)

                    img_file_str = scene.get("img_file", "")
                    img_path = Path(img_file_str) if img_file_str else None
                    has_image = img_path and img_path.exists()
                    if has_image:
                        pixmap = QtGui.QPixmap(str(img_path))
                        if not pixmap.isNull():
                            pixmap_scaled = pixmap.scaled(self.THUMBNAIL_SIZE, self.THUMBNAIL_SIZE, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
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

                    upload_button.clicked.connect(
                        functools.partial(self.on_upload_image, scene_id, img_preview_label, upload_button, delete_image_button, scene)
                    )
                    delete_image_button.clicked.connect(
                        functools.partial(self.on_delete_image, img_path if img_path else Path(), img_preview_label, upload_button, delete_image_button, scene)
                    )
                    clips_dir = self.json_path.parent / "clips"
                    video_file_path = clips_dir / f"{scene_id}.mp4"
                    video_exists = video_file_path.exists()
                    delete_video_button.setEnabled(video_exists)
                    delete_video_button.clicked.connect(functools.partial(self.on_delete_video, video_file_path, delete_video_button))
                    btn_add_char.clicked.connect(functools.partial(self.on_add_char_to_scene, char_list_widget))
                    btn_del_char.clicked.connect(functools.partial(self.on_del_char_from_scene, char_list_widget))
                    btn_edit_segments.clicked.connect(functools.partial(self.on_open_segment_editor, scene_id, scene))
                    form_layout_page.addRow(label, row_container)
                    self.widget_map.append((scene_id, text_edit))

                self.stacked_widget.addWidget(page_scroll_area)

            self.current_page = 0
            self.update_page_ui()

        except Exception as e_load_ui:
            error_label = QtWidgets.QLabel(f"파일 로드 또는 UI 빌드 중 오류 발생:\n{e_load_ui}\n\n{traceback.format_exc()}")
            error_label.setWordWrap(True)
            error_page = QtWidgets.QWidget()
            error_layout = QtWidgets.QVBoxLayout(error_page)
            error_layout.addWidget(error_label)
            self.stacked_widget.addWidget(error_page)
            self.btn_update.setEnabled(False)
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)

    def update_page_ui(self):
        self.stacked_widget.setCurrentIndex(self.current_page)
        self.page_label.setText(f"페이지 {self.current_page + 1} / {self.total_pages}")
        self.btn_prev.setEnabled(self.current_page > 0)
        self.btn_next.setEnabled(self.current_page < self.total_pages - 1)

    def on_prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.update_page_ui()

    def on_next_page(self):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.update_page_ui()

    def on_update_and_close(self):
        try:
            updated_fps = 0
            try:
                main_window = self.parent()
                if main_window and hasattr(main_window, "cmb_movie_fps"):
                    cmb_fps = getattr(main_window, "cmb_movie_fps")
                    ui_fps = int(cmb_fps.currentData())
                    updated_fps = ui_fps
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

            updated_styles = {}
            updated_char_id_list = []
            if hasattr(self, "style_widget_map"):
                for char_id, (label, field_container) in self.style_widget_map.items():
                    edit_widget = field_container.findChild(QtWidgets.QLineEdit)
                    if edit_widget:
                        updated_styles[char_id] = edit_widget.text().strip()
                        updated_char_id_list.append(char_id)

            self.full_video_data["character_styles"] = updated_styles
            self.full_video_data["characters"] = updated_char_id_list
            updated_styles_count = len(updated_styles)

            scene_map = {scene.get("id"): scene for scene in self.scenes_data if isinstance(scene, dict) and "id" in scene}
            updated_prompts_count = 0
            for scene_id, text_edit in self.widget_map:
                if scene_id in scene_map:
                    new_prompt = text_edit.toPlainText().strip()
                    scene = scene_map[scene_id]
                    if scene.get("direct_prompt", "") != new_prompt:
                        scene["direct_prompt"] = new_prompt
                        updated_prompts_count += 1

            updated_scene_chars_count = 0
            if hasattr(self, "scene_char_widgets"):
                for scene_id, list_widget in self.scene_char_widgets.items():
                    if scene_id in scene_map:
                        new_scene_chars = []
                        for i in range(list_widget.count()):
                            item = list_widget.item(i)
                            if item:
                                new_scene_chars.append(item.text())
                        scene = scene_map[scene_id]
                        if scene.get("characters") != new_scene_chars:
                            scene["characters"] = new_scene_chars
                            updated_scene_chars_count += 1

            self.full_video_data["scenes"] = self.scenes_data
            save_json(self.json_path, self.full_video_data)

            QtWidgets.QMessageBox.information(self, "업데이트 완료",
                                              f"파일에 저장되었습니다: {self.json_path.name}\n\n"
                                              f"- 캐릭터 스타일 {updated_styles_count}개 항목 업데이트됨\n"
                                              f"- (루트 'characters' 목록도 {len(updated_char_id_list)}개로 갱신됨)\n"
                                              f"- 씬 'direct_prompt' {updated_prompts_count}개 항목 업데이트됨\n"
                                              f"- 씬별 'characters' {updated_scene_chars_count}개 항목 업데이트됨")

        except Exception as e_update:
            QtWidgets.QMessageBox.critical(self, "저장 오류", f"파일을 저장하는 중 오류가 발생했습니다:\n{e_update}")