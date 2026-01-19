# -*- coding: utf-8 -*-
from __future__ import annotations

import shutil
import datetime
import time
import json
import os
import re
from pathlib import Path
from PyQt5 import QtWidgets, QtCore, QtGui
from typing import List
from app import settings

from app.utils import (
    run_job_with_progress_async,
    sanitize_title,
    load_json,
    save_json,
    get_duration
)

from app.issue_list_builder import (
    save_issue_list_for_shopping_all,
    save_issue_list_for_shopping_ai_b_from_a,
)

from app.shopping_video_build import build_shopping_images_2step
from app.shopping_video_build import convert_shopping_to_video_json_with_ai
from app.video_build import build_shots_with_i2v_long
from app.shopping_video_build import (
    ShoppingVideoJsonBuilder,
    ShoppingImageGenerator,
    ShoppingMovieGenerator,
    ShoppingShortsPipeline,
    BuildOptions,
    generate_tts_zonos as _tts_generate_zonos,
    _get_zonos_config as _zonos_get_config,
)



class SceneEditDialog(QtWidgets.QDialog):
    """
    씬별 기획 + 캐릭터/성별 + 배경음(BGM) 수정
    + prompt_img_1_kor / prompt_img_2_kor 표시 및 저장
    + 'AI 수정 요청' 버튼 클릭 시:
        - prompt_img_1_kor, prompt_img_2_kor 는 그대로 저장
        - prompt_img_1, prompt_img_2 는 각각 *_kor 를 영어 번역한 결과로 덮어써서 저장
        - ai_edit_request 에 요청 텍스트 저장

    (추가)
    + '내레이션 생성' 버튼:
        - 클릭한 해당 씬만 narration 기반으로 TTS 생성
        - voice_file / seconds 를 즉시 갱신 저장
    """

    def __init__(self, json_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("시나리오 초안 및 상세 설정 수정")
        self.resize(950, 980)
        self.json_path = Path(json_path)

        self.data = load_json(self.json_path, {})
        self.meta = self.data.get("meta", {})
        self.scenes = self.data.get("scenes", [])

        # 번역용 AI (prompt_img_*_kor -> EN)
        try:
            from app.utils import AI
            self._ai = AI()
        except Exception:
            self._ai = None

        layout = QtWidgets.QVBoxLayout(self)

        # ─── 상단: 기획 설정 (성별, 캐릭터, BGM) ───
        char_group = QtWidgets.QGroupBox("👤 기획 설정 (캐릭터/성별/BGM)", self)
        char_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 2px solid #0078d7; margin-top: 10px; background-color: #f0f8ff; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #0078d7; }"
        )
        char_layout = QtWidgets.QFormLayout(char_group)

        self.cb_gender = QtWidgets.QComboBox()
        self.cb_gender.addItems(["Female", "Male"])
        current_gender = str(self.meta.get("voice_gender", "female")).capitalize()
        if "Male" in current_gender:
            self.cb_gender.setCurrentText("Male")
        else:
            self.cb_gender.setCurrentText("Female")
        char_layout.addRow("성별 (Voice & Visual):", self.cb_gender)

        char_val = self.meta.get("character_prompt", "")
        self.le_char = QtWidgets.QLineEdit(char_val)
        self.le_char.setPlaceholderText("예: Stylish 30s urban news anchor")
        self.le_char.setStyleSheet("background-color: white; padding: 5px; font-weight: bold;")
        char_layout.addRow("캐릭터 컨셉 (EN):", self.le_char)

        bgm_val = self.meta.get("bgm_prompt", "")
        self.te_bgm = QtWidgets.QPlainTextEdit(bgm_val)
        self.te_bgm.setPlaceholderText("예: instrumental, background music, calm, piano, loopable...")
        self.te_bgm.setMaximumHeight(60)
        self.te_bgm.setStyleSheet("background-color: white; padding: 5px;")
        char_layout.addRow("배경음(BGM) 설정:", self.te_bgm)

        char_layout.addRow(QtWidgets.QLabel("* AI가 제안한 설정이 마음에 들지 않으면 여기서 수정하세요."))
        layout.addWidget(char_group)

        lbl_info = QtWidgets.QLabel(
            "아래에서 각 장면별(Scene) 세부 내용과 프롬프트를 수정하세요.\n"
            "AI 수정 요청 버튼을 누르면 KR 프롬프트는 저장하고, EN 프롬프트는 KR을 영어로 번역해 덮어씁니다.\n"
            "내레이션 생성 버튼은 클릭한 씬만 음성을 생성하고 seconds/voice_file을 즉시 갱신합니다."
        )
        lbl_info.setStyleSheet(
            "color: #333; font-weight: bold; margin: 10px 0; background: #f0f0f0; padding: 10px; border-radius: 5px;"
        )
        layout.addWidget(lbl_info)

        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, 1)

        container = QtWidgets.QWidget()
        self.form_layout = QtWidgets.QVBoxLayout(container)
        self.form_layout.setSpacing(20)
        scroll.setWidget(container)

        self.editors = []

        def _translate_kor_to_en(kor_text: str) -> str:
            kor_text = (kor_text or "").strip()
            if not kor_text:
                return ""

            if self._ai is None:
                return kor_text

            sys_msg = (
                "You are a professional translator for AI image generation prompts.\n"
                "Translate Korean to natural, concise English.\n"
                "Do NOT add new details. Keep meaning faithful.\n"
                "Output English ONLY."
            )
            user_msg = f'Korean:\n"{kor_text}"\n\nEnglish:'
            try:
                out = self._ai.ask_smart(sys_msg, user_msg, prefer="openai")
                out = (out or "").strip()
                out = out.replace("```", "").strip()
                return out
            except Exception:
                return kor_text

        def _pick_ref_audio_by_gender(voice_gender: str) -> Path:
            """
            [고정 규칙] 초안생성(Enrich)과 동일한 레퍼런스 보이스 선택
            - male  : C:\\my_games\\shorts_make\\voice\\남자성우1.mp3
            - female: C:\\my_games\\shorts_make\\voice\\꼬꼬 음성.m4a
            """
            g = (voice_gender or "female").strip().lower()
            if g == "male":
                return Path(r"C:\my_games\shorts_make\voice\남자성우1.mp3")
            return Path(r"C:\my_games\shorts_make\voice\꼬꼬 음성.m4a")

        for idx, sc in enumerate(self.scenes):
            sid = sc.get("id", f"{idx + 1:03d}")

            group = QtWidgets.QGroupBox(f"🎬 Scene {sid}", container)
            group.setStyleSheet(
                "QGroupBox { font-weight: bold; border: 1px solid #ccc; margin-top: 10px; background-color: #ffffff; } "
                "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; background-color: #ffffff; }"
            )
            g_layout = QtWidgets.QFormLayout(group)
            g_layout.setLabelAlignment(QtCore.Qt.AlignRight)
            g_layout.setContentsMargins(15, 15, 15, 15)

            le_banner = QtWidgets.QLineEdit(str(sc.get("banner") or ""))
            te_prompt = QtWidgets.QPlainTextEdit(str(sc.get("prompt") or ""))
            te_prompt.setMinimumHeight(60)

            te_narr = QtWidgets.QPlainTextEdit(str(sc.get("narration") or ""))
            te_narr.setMinimumHeight(60)
            te_narr.setStyleSheet("background-color: #fdfae8;")

            le_sub = QtWidgets.QLineEdit(str(sc.get("subtitle") or ""))

            # seconds 표시(간단 라벨)
            lbl_seconds = QtWidgets.QLabel(f"⏱ seconds: {sc.get('seconds', 0)}")
            lbl_seconds.setStyleSheet("color:#444; font-weight:bold;")

            g_layout.addRow("🚩 배너:", le_banner)
            g_layout.addRow("🖼️ 화면설명(KR):", te_prompt)
            g_layout.addRow("🎙️ 내레이션:", te_narr)
            g_layout.addRow("💬 자막:", le_sub)
            g_layout.addRow(" ", lbl_seconds)

            line = QtWidgets.QFrame()
            line.setFrameShape(QtWidgets.QFrame.HLine)
            line.setFrameShadow(QtWidgets.QFrame.Sunken)
            g_layout.addRow(line)

            # KR 프롬프트
            te_p1_kor = QtWidgets.QPlainTextEdit(str(sc.get("prompt_img_1_kor") or ""))
            te_p1_kor.setPlaceholderText("Step 1 (KR): 배경/모델 묘사 (한국어)")
            te_p1_kor.setMinimumHeight(60)

            te_p2_kor = QtWidgets.QPlainTextEdit(str(sc.get("prompt_img_2_kor") or ""))
            te_p2_kor.setPlaceholderText("Step 2 (KR): 제품 합성 지시 (한국어)")
            te_p2_kor.setMinimumHeight(60)

            # EN 프롬프트
            te_p1 = QtWidgets.QPlainTextEdit(str(sc.get("prompt_img_1") or sc.get("prompt_img") or ""))
            te_p1.setPlaceholderText("Step 1 (EN): Base Prompt (Background/Model)")
            te_p1.setMinimumHeight(60)

            te_p2 = QtWidgets.QPlainTextEdit(str(sc.get("prompt_img_2") or ""))
            te_p2.setPlaceholderText("Step 2 (EN): Merge Prompt (Product Synthesis)")
            te_p2.setMinimumHeight(60)

            # AI 수정 요청 버튼
            btn_ai_req = QtWidgets.QPushButton("🤖 AI 수정 요청")
            btn_ai_req.setToolTip(
                "클릭하면: (1) 요청 텍스트를 ai_edit_request에 저장\n"
                "(2) prompt_img_1_kor/2_kor는 저장\n"
                "(3) prompt_img_1/2는 각각 *_kor를 영어로 번역해 덮어쓰기\n"
                "(4) 즉시 파일 저장까지 수행"
            )
            btn_ai_req.setMinimumHeight(34)

            def _on_ai_req_clicked(
                    checked: bool = False,
                    _sid=sid,
                    _sc=sc,
                    _te_p1_kor=te_p1_kor,
                    _te_p2_kor=te_p2_kor,
                    _te_p1=te_p1,
                    _te_p2=te_p2,
            ):
                txt, ok = QtWidgets.QInputDialog.getMultiLineText(
                    self,
                    "AI 수정 요청",
                    f"Scene {_sid}에 대해 AI에게 요청할 수정 사항을 입력하세요.\n"
                    "확인을 누르면 KR은 저장하고, EN은 KR->EN 번역으로 덮어써서 저장합니다.",
                    str(_sc.get("ai_edit_request") or "")
                )
                if not ok:
                    return

                _sc["ai_edit_request"] = (txt or "").strip()

                p1_kor = _te_p1_kor.toPlainText().strip()
                p2_kor = _te_p2_kor.toPlainText().strip()
                _sc["prompt_img_1_kor"] = p1_kor
                _sc["prompt_img_2_kor"] = p2_kor

                p1_en = _translate_kor_to_en(p1_kor)
                p2_en = _translate_kor_to_en(p2_kor)

                _sc["prompt_img_1"] = p1_en
                _sc["prompt_img_2"] = p2_en
                _sc["prompt_img"] = p1_en  # 호환성

                _te_p1.setPlainText(p1_en)
                _te_p2.setPlainText(p2_en)

                try:
                    save_json(self.json_path, self.data)
                    QtWidgets.QMessageBox.information(
                        self,
                        "저장 완료",
                        "KR 저장 + KR→EN 번역 덮어쓰기 + 파일 저장까지 완료했습니다."
                    )
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, "저장 실패", f"파일 저장 중 오류:\n{e}")

            btn_ai_req.clicked.connect(_on_ai_req_clicked)

            # ✅ (추가) 내레이션 생성 버튼
            btn_make_narr = QtWidgets.QPushButton("🎙️ 내레이션 생성")
            btn_make_narr.setToolTip(
                "클릭한 씬만 내레이션 음성을 생성합니다.\n"
                "- narration 텍스트로 TTS 생성\n"
                "- voice_file / seconds(실측 +0.5s) 갱신 후 즉시 저장"
            )
            btn_make_narr.setMinimumHeight(34)

            def _on_make_narr_clicked(
                    checked: bool = False,
                    _sid=sid,
                    _sc=sc,
                    _te_narr=te_narr,
                    _lbl_seconds=lbl_seconds,
            ):
                narration_text = (_te_narr.toPlainText() or "").strip()
                if not narration_text:
                    QtWidgets.QMessageBox.warning(self, "알림", "내레이션 텍스트가 비어있습니다.")
                    return

                def job(on_progress=None):
                    # 1) 최신 json 다시 로드 후 scene 재매칭
                    data = load_json(self.json_path, {})
                    meta = data.get("meta", {}) if isinstance(data, dict) else {}
                    scenes = data.get("scenes", []) if isinstance(data, dict) else []
                    if not isinstance(scenes, list):
                        raise RuntimeError("video_shopping.json scenes가 리스트가 아닙니다.")

                    target = None
                    for s in scenes:
                        if str(s.get("id")) == str(_sid):
                            target = s
                            break
                    if target is None:
                        raise RuntimeError(f"scene id를 찾을 수 없습니다: {_sid}")

                    # 2) narration 저장
                    target["narration"] = narration_text

                    # 3) voice_file 경로 결정
                    prod_dir = self.json_path.parent
                    voice_dir = prod_dir / "voice"
                    voice_dir.mkdir(parents=True, exist_ok=True)

                    vf = (target.get("voice_file") or "").strip()
                    if vf:
                        out_wav = Path(vf)
                        if not out_wav.is_absolute():
                            out_wav = (prod_dir / out_wav).resolve()
                    else:
                        out_wav = (voice_dir / f"{_sid}.wav").resolve()
                        target["voice_file"] = str(out_wav.relative_to(prod_dir).as_posix())

                    # 4) ref voice 선택 (초안생성과 동일)
                    gender = str(meta.get("voice_gender", "female")).lower()
                    ref_voice = _pick_ref_audio_by_gender(gender)
                    if not ref_voice.exists() or ref_voice.stat().st_size == 0:
                        raise RuntimeError(f"레퍼런스 보이스 파일이 없습니다: {ref_voice}")

                    # 5) zonos config
                    cfg = _zonos_get_config(target, self._ai)

                    # 6) TTS 생성 (초안생성과 동일 시그니처)
                    comfy_host = getattr(settings, "COMFY_HOST", "http://127.0.0.1:8188")

                    # (A) 파일이 없으면 생성
                    if not out_wav.exists() or out_wav.stat().st_size == 0:
                        ok = _tts_generate_zonos(narration_text, out_wav, ref_voice, comfy_host, cfg)
                        if not ok:
                            # 실패 시 seconds 안전값
                            if float(target.get("seconds") or 0) <= 0:
                                target["seconds"] = 4.0
                            save_json(self.json_path, data)
                            raise RuntimeError("내레이션 음성 생성 실패")

                    # (B) 파일이 존재하면 무조건 길이 측정 (3회 재시도)
                    final_dur = 0.0
                    for _ in range(3):
                        try:
                            d = float(get_duration(str(out_wav)) or 0.0)
                            if d > 0:
                                final_dur = d
                                break
                        except Exception:
                            pass
                        time.sleep(0.1)

                    if final_dur > 0:
                        target["seconds"] = round(final_dur + 0.5, 2)
                    else:
                        if float(target.get("seconds") or 0) <= 0:
                            target["seconds"] = 4.0

                    # voice_file는 상대경로 유지(이미 위에서 세팅됨)
                    save_json(self.json_path, data)

                    return {
                        "scene_id": _sid,
                        "voice_file": str(out_wav),
                        "duration": final_dur,
                        "seconds": target["seconds"],
                    }

                def done(ok: bool, result=None):
                    if not ok:
                        QtWidgets.QMessageBox.critical(self, "실패", f"내레이션 생성 실패:\n{result}")
                        return

                    sec = result.get("seconds", 0) if isinstance(result, dict) else 0
                    _lbl_seconds.setText(f"⏱ seconds: {sec}")

                    QtWidgets.QMessageBox.information(
                        self,
                        "완료",
                        f"Scene {_sid} 내레이션 생성 완료\n"
                        f"- seconds: {sec}\n"
                        f"- voice_file: {result.get('voice_file', '')}"
                    )

                run_job_with_progress_async(
                    owner=self,
                    title=f"[Scene {_sid}] 내레이션 생성",
                    job=job,
                    on_done=done,
                )

            btn_make_narr.clicked.connect(_on_make_narr_clicked)

            # 배치
            g_layout.addRow("🇰🇷 Base Prompt:", te_p1_kor)
            g_layout.addRow("🇰🇷 Merge Prompt:", te_p2_kor)
            g_layout.addRow("✨ Base Prompt:", te_p1)
            g_layout.addRow("🔗 Merge Prompt:", te_p2)
            g_layout.addRow(" ", btn_ai_req)
            g_layout.addRow(" ", btn_make_narr)  # ✅ AI 버튼 아래 추가

            self.form_layout.addWidget(group)

            self.editors.append({
                "sc": sc,
                "banner": le_banner,
                "prompt": te_prompt,
                "narration": te_narr,
                "subtitle": le_sub,
                "p1_kor": te_p1_kor,
                "p2_kor": te_p2_kor,
                "p1": te_p1,
                "p2": te_p2,
            })

        btn_box = QtWidgets.QHBoxLayout()
        btn_save = QtWidgets.QPushButton("💾 저장 및 닫기", self)
        btn_save.setMinimumHeight(45)
        btn_save.setStyleSheet("font-weight: bold; font-size: 14px; color: white; background-color: #0078d7; border-radius: 6px;")

        btn_cancel = QtWidgets.QPushButton("취소", self)
        btn_cancel.setMinimumHeight(45)

        btn_save.clicked.connect(self.on_save)
        btn_cancel.clicked.connect(self.reject)

        btn_box.addStretch(1)
        btn_box.addWidget(btn_save, 2)
        btn_box.addWidget(btn_cancel, 1)
        layout.addLayout(btn_box)

    def on_save(self):
        # 1) 메인 설정 저장
        self.meta["voice_gender"] = self.cb_gender.currentText().lower()

        new_char = self.le_char.text().strip()
        if new_char:
            self.meta["character_prompt"] = new_char

        new_bgm = self.te_bgm.toPlainText().strip()
        if new_bgm:
            self.meta["bgm_prompt"] = new_bgm

        # 2) 씬별 저장
        changed_count = 0
        for item in self.editors:
            sc = item["sc"]

            new_banner = item["banner"].text().strip()
            new_prompt = item["prompt"].toPlainText().strip()
            new_narr = item["narration"].toPlainText().strip()
            new_sub = item["subtitle"].text().strip()

            new_p1_kor = item["p1_kor"].toPlainText().strip()
            new_p2_kor = item["p2_kor"].toPlainText().strip()

            new_p1 = item["p1"].toPlainText().strip()
            new_p2 = item["p2"].toPlainText().strip()

            if (str(sc.get("banner") or "") != new_banner or
                    str(sc.get("prompt") or "") != new_prompt or
                    str(sc.get("narration") or "") != new_narr or
                    str(sc.get("subtitle") or "") != new_sub or
                    str(sc.get("prompt_img_1_kor") or "") != new_p1_kor or
                    str(sc.get("prompt_img_2_kor") or "") != new_p2_kor or
                    str(sc.get("prompt_img_1") or "") != new_p1 or
                    str(sc.get("prompt_img_2") or "") != new_p2):

                sc["banner"] = new_banner
                sc["prompt"] = new_prompt
                sc["narration"] = new_narr
                sc["subtitle"] = new_sub

                sc["prompt_img_1_kor"] = new_p1_kor
                sc["prompt_img_2_kor"] = new_p2_kor

                sc["prompt_img_1"] = new_p1
                sc["prompt_img_2"] = new_p2

                # 호환성 동기화
                sc["prompt_img"] = new_p1

                changed_count += 1

        try:
            save_json(self.json_path, self.data)
            QtWidgets.QMessageBox.information(self, "저장 완료", f"기획 설정 및 {changed_count}개 장면이 수정되었습니다.")
            self.accept()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "저장 실패", f"파일 저장 중 오류가 발생했습니다:\n{e}")







class MediaEditDialog(QtWidgets.QDialog):
    """
    [수정됨] 3.5단계 이미지/영상 수정 및 검수 다이얼로그 (2-Step 프롬프트 적용)
    - 좌측: 씬 정보, 이미지 미리보기, 변경/삭제, 영상 삭제 기능
    - 우측: Base Prompt, Merge Prompt, Movie Prompt, Negative Prompt 수정
    """

    def __init__(self, json_path: str, product_dir: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("이미지 및 영상 수정/검수 (2-Step)")
        self.resize(1200, 800)
        self.json_path = Path(json_path)
        self.product_dir = Path(product_dir)
        self.imgs_dir = self.product_dir / "imgs"
        self.clips_dir = self.product_dir / "clips"

        # JSON 로드
        self.data = load_json(self.json_path, {})
        self.scenes = self.data.get("scenes", [])

        # 메인 레이아웃
        layout = QtWidgets.QVBoxLayout(self)

        # 안내 문구
        lbl_info = QtWidgets.QLabel(
            "생성된 이미지와 프롬프트를 검수하는 공간입니다.\n"
            "이미지 생성은 [Step 1: 베이스] -> [Step 2: 합성]으로 진행되므로 프롬프트가 2개입니다.\n"
            "이미지를 수동으로 변경하면 '{id}.png'로 덮어씌워집니다."
        )
        lbl_info.setStyleSheet("background: #f0f0f0; padding: 10px; font-weight: bold; border-radius: 5px;")
        layout.addWidget(lbl_info)

        # 스크롤 영역
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, 1)

        container = QtWidgets.QWidget()
        self.container_layout = QtWidgets.QVBoxLayout(container)
        self.container_layout.setSpacing(20)
        scroll.setWidget(container)

        self.editors = []  # 저장 시 참조할 위젯들 보관

        # 씬별 UI 생성
        for idx, sc in enumerate(self.scenes):
            self._add_scene_widget(idx, sc, container)

        # 하단 버튼 (전체 저장)
        btn_box = QtWidgets.QHBoxLayout()
        btn_save = QtWidgets.QPushButton("💾 변경사항(프롬프트) 저장 및 닫기", self)
        btn_save.setMinimumHeight(50)
        btn_save.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #0078d7; color: white;")

        btn_close = QtWidgets.QPushButton("닫기", self)
        btn_close.setMinimumHeight(50)

        btn_save.clicked.connect(self.on_save)
        btn_close.clicked.connect(self.reject)

        btn_box.addStretch(1)
        btn_box.addWidget(btn_save, 2)
        btn_box.addWidget(btn_close, 1)
        layout.addLayout(btn_box)

    def _add_scene_widget(self, idx, sc, parent_widget):
        sid = sc.get("id", f"{idx + 1:03d}")

        # 그룹박스 프레임
        group = QtWidgets.QGroupBox(f"Scene {sid}", parent_widget)
        group.setStyleSheet("QGroupBox { border: 2px solid #ccc; margin-top: 10px; background: white; } "
                            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; font-weight: bold; }")
        row_layout = QtWidgets.QHBoxLayout(group)

        # ─── [좌측] 미디어 관리 (이미지/영상/정보) ───
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_widget.setFixedWidth(320)  # 좌측 고정 너비

        # 1. 정보 표시
        narr_short = (sc.get("narration") or "")[:30] + "..." if len(sc.get("narration") or "") > 30 else sc.get(
            "narration")
        info_txt = f"⏱ 시간: {sc.get('seconds', 0)}초\n🎙 내레이션: {narr_short}"
        lbl_info = QtWidgets.QLabel(info_txt)
        lbl_info.setStyleSheet("color: #555; margin-bottom: 5px;")
        left_layout.addWidget(lbl_info)

        # 2. 이미지 미리보기 (수정됨: 클릭 이벤트 추가)
        lbl_img = QtWidgets.QLabel()
        lbl_img.setFixedSize(300, 300)
        lbl_img.setStyleSheet("background-color: #eee; border: 1px solid #aaa;")
        lbl_img.setAlignment(QtCore.Qt.AlignCenter)
        lbl_img.setCursor(QtCore.Qt.PointingHandCursor)  # 마우스 올리면 손가락 모양
        lbl_img.setToolTip("클릭하면 원본 크기로 봅니다.")

        # [핵심] 클릭 이벤트 바인딩 (람다로 sid 캡처)
        # QLabel은 clicked 시그널이 없으므로 mousePressEvent를 오버라이딩
        def on_img_click(event):
            if event.button() == QtCore.Qt.LeftButton:
                self.on_view_full_image(sid)

        lbl_img.mousePressEvent = on_img_click

        self._refresh_preview(sid, lbl_img)  # 이미지 로드
        left_layout.addWidget(lbl_img)

        # 3. 버튼들
        btn_change_img = QtWidgets.QPushButton("📂 이미지 변경 (불러오기)")
        btn_del_img = QtWidgets.QPushButton("🗑 이미지 삭제")
        btn_del_mov = QtWidgets.QPushButton("❌ 영상 삭제 (재생성용)")

        btn_change_img.clicked.connect(lambda: self.on_change_image(sid, lbl_img))
        btn_del_img.clicked.connect(lambda: self.on_delete_image(sid, lbl_img))
        btn_del_mov.clicked.connect(lambda: self.on_delete_movie(sid))

        left_layout.addWidget(btn_change_img)
        left_layout.addWidget(btn_del_img)
        left_layout.addWidget(btn_del_mov)
        left_layout.addStretch(1)

        row_layout.addWidget(left_widget)

        # ─── [우측] 프롬프트 수정 (2-Step 적용) ───
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QFormLayout(right_widget)
        right_layout.setLabelAlignment(QtCore.Qt.AlignRight)

        # 데이터 가져오기
        p1 = str(sc.get("prompt_img_1") or sc.get("prompt_img") or "")  # Prompt 1 (Base)
        p2 = str(sc.get("prompt_img_2") or "")  # Prompt 2 (Merge)
        p_mov = str(sc.get("prompt_movie") or "")
        p_neg = str(sc.get("prompt_negative") or "")

        # 에디터 생성
        te_p1 = QtWidgets.QPlainTextEdit(p1)
        te_p1.setPlaceholderText("✨ Step 1: Base Prompt (Background/Model)")
        te_p1.setMinimumHeight(60)

        te_p2 = QtWidgets.QPlainTextEdit(p2)
        te_p2.setPlaceholderText("🔗 Step 2: Merge Prompt (Product Synthesis)")
        te_p2.setMinimumHeight(60)

        te_p_mov = QtWidgets.QPlainTextEdit(p_mov)
        te_p_mov.setPlaceholderText("🎥 Movie Prompt (Camera movement)")
        te_p_mov.setMinimumHeight(50)

        te_p_neg = QtWidgets.QPlainTextEdit(p_neg)
        te_p_neg.setPlaceholderText("🚫 Negative Prompt")
        te_p_neg.setMinimumHeight(50)

        right_layout.addRow("✨ Base Prompt:", te_p1)
        right_layout.addRow("🔗 Merge Prompt:", te_p2)
        right_layout.addRow("🎥 Mov Prompt:", te_p_mov)
        right_layout.addRow("🚫 Negative:", te_p_neg)

        row_layout.addWidget(right_widget, 1)  # 우측은 늘어남

        self.container_layout.addWidget(group)

        # 저장 목록에 추가
        self.editors.append({
            "sc": sc,
            "p1": te_p1, "p2": te_p2,
            "p_mov": te_p_mov, "p_neg": te_p_neg
        })

    def _refresh_preview(self, sid: str, label: QtWidgets.QLabel):
        """
        이미지 미리보기 갱신 (ID 호환성 강화)
        - 기획서 ID(001)로 파일이 없으면,
        - 영상제작용 ID(t_001)로 자동으로 찾아서 보여줍니다.
        """
        # 1차 시도: 001.png
        img_path = self.imgs_dir / f"{sid}.png"

        # 2차 시도: t_001.png (영상 엔진용 포맷)
        if not img_path.exists():
            # sid가 숫자형태면 t_ 붙임
            if sid.isdigit():
                img_path = self.imgs_dir / f"t_{sid}.png"
            # 만약 sid가 이미 t_001이면 여기서 걸림

        # 3차 시도: temp_ 파일 (Step 1만 있는 경우) 확인
        if not img_path.exists():
            img_path = self.imgs_dir / f"temp_{sid}.png"
            if not img_path.exists() and sid.isdigit():
                img_path = self.imgs_dir / f"temp_t_{sid}.png"

        if img_path.exists():
            pix = QtGui.QPixmap(str(img_path))
            if not pix.isNull():
                label.setPixmap(pix.scaled(label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
                label.setText("")
                # [중요] 뷰어에서 원본 볼 때 쓸 경로를 라벨에 저장해둠 (Full View 연동용)
                label.setProperty("full_path", str(img_path))
            else:
                label.setText("이미지 로드 실패")
        else:
            label.clear()
            label.setText("이미지 없음")

    def on_change_image(self, sid: str, label: QtWidgets.QLabel):
        """1. 이미지 변경: 외부 파일 선택 -> {id}.png로 복사/덮어쓰기"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "변경할 이미지 선택", "", "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if not path:
            return

        try:
            target_path = self.imgs_dir / f"{sid}.png"
            shutil.copy2(path, target_path)
            self._refresh_preview(sid, label)
            QtWidgets.QMessageBox.information(self, "완료", f"이미지가 변경되었습니다.\n{target_path.name}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "오류", f"이미지 변경 실패: {e}")

    def on_delete_image(self, sid: str, label: QtWidgets.QLabel):
        """
        [완벽 삭제 버전]
        ID가 '001'이든 't_001'이든 상관없이 관련된 모든 이미지 파일(원본/Temp)을 찾아 삭제합니다.
        """
        # 삭제 대상 후보군 (구버전 001, 신버전 t_001 모두 포함)
        candidates = [
            self.imgs_dir / f"{sid}.png",  # 001.png
            self.imgs_dir / f"t_{sid}.png",  # t_001.png (현재 생성되는 파일)
            self.imgs_dir / f"temp_{sid}.png",  # temp_001.png
            self.imgs_dir / f"temp_t_{sid}.png"  # temp_t_001.png
        ]

        # 실제로 존재하는 파일이 하나라도 있는지 확인
        exists_files = [p for p in candidates if p.exists()]

        if not exists_files:
            QtWidgets.QMessageBox.information(self, "알림", "삭제할 이미지가 없습니다.")
            return

        reply = QtWidgets.QMessageBox.question(
            self, "이미지 완전 삭제",
            f"Scene {sid} 관련 이미지를 모두 삭제하시겠습니까?\n"
            f"(총 {len(exists_files)}개 파일 감지됨)\n\n"
            "⚠️ [Step 1: 베이스]와 [Step 2: 완성본]이 모두 삭제되어 초기화됩니다.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            try:
                deleted_count = 0
                for p in exists_files:
                    try:
                        os.remove(p)
                        deleted_count += 1
                    except Exception:
                        pass  # 혹시 파일이 사용 중이라 안 지워져도 계속 진행

                # 화면 갱신
                self._refresh_preview(sid, label)
                QtWidgets.QMessageBox.information(self, "삭제 완료", f"총 {deleted_count}개의 이미지 파일이 삭제되었습니다.")

            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "오류", f"삭제 중 오류 발생: {e}")

    def on_delete_movie(self, sid: str):
        """3. 영상 삭제: clips/{id}.mp4 삭제"""
        target_path = self.clips_dir / f"{sid}.mp4"
        if not target_path.exists():
            QtWidgets.QMessageBox.information(self, "알림", "삭제할 영상 파일이 없습니다.")
            return

        reply = QtWidgets.QMessageBox.question(
            self, "영상 삭제",
            f"Scene {sid}의 영상을 삭제하시겠습니까?\n삭제 후 '영상 생성'을 다시 실행하면 이 컷만 다시 생성됩니다.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            try:
                os.remove(target_path)
                QtWidgets.QMessageBox.information(self, "완료", "영상이 삭제되었습니다.")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "오류", f"삭제 실패: {e}")

    def on_save(self):
        """4. 프롬프트 수정 사항 JSON 저장"""
        changed_count = 0
        for item in self.editors:
            sc = item["sc"]
            new_p1 = item["p1"].toPlainText().strip()
            new_p2 = item["p2"].toPlainText().strip()
            new_p_mov = item["p_mov"].toPlainText().strip()
            new_p_neg = item["p_neg"].toPlainText().strip()

            if (sc.get("prompt_img_1") != new_p1 or
                    sc.get("prompt_img_2") != new_p2 or
                    sc.get("prompt_movie") != new_p_mov or
                    sc.get("prompt_negative") != new_p_neg):
                sc["prompt_img_1"] = new_p1
                sc["prompt_img_2"] = new_p2
                sc["prompt_img"] = new_p1  # 호환성 동기화
                sc["prompt_movie"] = new_p_mov
                sc["prompt_negative"] = new_p_neg
                changed_count += 1

        if changed_count > 0:
            try:
                save_json(self.json_path, self.data)
                QtWidgets.QMessageBox.information(self, "저장 완료", f"{changed_count}개 장면의 프롬프트가 수정되었습니다.")
                self.accept()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "저장 실패", f"JSON 저장 중 오류: {e}")
        else:
            self.accept()

    def on_view_full_image(self, sid: str):
        """원본 이미지 팝업 열기 (경로 자동 인식)"""
        # _refresh_preview에서 찾은 경로가 있다면 그걸 최우선으로 사용
        # (UI 라벨에서 찾아서 넘겨주는 방식이 가장 정확함)

        # 현재 해당 씬의 라벨 위젯을 찾기가 번거로울 수 있으므로 재탐색 로직 사용
        # (위 _refresh_preview와 동일한 탐색 순서)

        candidates = [
            self.imgs_dir / f"{sid}.png",
            self.imgs_dir / f"t_{sid}.png",
            self.imgs_dir / f"temp_{sid}.png",
            self.imgs_dir / f"temp_t_{sid}.png"
        ]

        target_path = None
        for p in candidates:
            if p.exists():
                target_path = p
                break

        if target_path:
            viewer = ImageViewerDialog(str(target_path), parent=self)
            viewer.exec_()
        else:
            QtWidgets.QMessageBox.warning(self, "알림", "확인할 이미지가 없습니다.")


class FinalEditDialog(QtWidgets.QDialog):
    """
    video.json 전용 최종 프롬프트 편집기
    - prompt_1_kor / prompt_2_kor / prompt_3_kor 편집 가능
    - prompt_1 / prompt_2 / prompt_3 (영어)도 같이 표시/편집 가능
    - 각 씬에 "AI 수정 요청(한글→영어 반영)" 버튼 제공:
        클릭한 '해당 씬'만 수정됨.
        kor 필드는 보존, en 필드는 kor를 기반으로 재생성/번역하여 덮어씀.
    """

    def __init__(self, video_json_path: str, project_dir: str, parent=None):
        super().__init__(parent)
        self.video_json_path = str(video_json_path)
        self.project_dir = str(project_dir)

        self.setWindowTitle("최종안 수정 (video.json)")
        self.resize(980, 820)

        self._ai = None  # lazy

        root = QtWidgets.QVBoxLayout(self)

        top_info = QtWidgets.QLabel(
            "각 씬의 prompt_1_kor / prompt_2_kor / prompt_3_kor 를 편집한 뒤,\n"
            "필요하면 'AI 수정 요청(한글→영어 반영)'을 눌러 영어(prompt_1/2/3)를 자동 반영하세요.\n"
            "주의: AI 버튼은 클릭한 '해당 씬'만 수정됩니다."
        )
        root.addWidget(top_info)

        self.scroll = QtWidgets.QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        root.addWidget(self.scroll, 1)

        self.inner = QtWidgets.QWidget()
        self.scroll.setWidget(self.inner)
        self.vbox = QtWidgets.QVBoxLayout(self.inner)
        self.vbox.setContentsMargins(8, 8, 8, 8)
        self.vbox.setSpacing(10)

        btn_row = QtWidgets.QHBoxLayout()
        root.addLayout(btn_row)

        self.btn_save = QtWidgets.QPushButton("저장", self)
        self.btn_close = QtWidgets.QPushButton("닫기", self)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_close)

        self.btn_close.clicked.connect(self.reject)
        self.btn_save.clicked.connect(self._save)

        self._scene_widgets = []  # [{sid, seg_count, te_p1_k, te_p1, ...}]

        self._load()

    def _lazy_ai(self):
        if self._ai is None:
            from app.utils import AI
            self._ai = AI()
        return self._ai

    def _read_json(self):
        import json
        from pathlib import Path
        p = Path(self.video_json_path)
        return json.loads(p.read_text(encoding="utf-8"))

    def _write_json(self, data):
        import json
        from pathlib import Path
        p = Path(self.video_json_path)
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load(self):
        import json

        try:
            data = self._read_json()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "오류", f"video.json 로드 실패:\n{e}")
            self.reject()
            return

        scenes = data.get("scenes", [])
        if not isinstance(scenes, list):
            QtWidgets.QMessageBox.critical(self, "오류", "video.json의 scenes가 리스트가 아닙니다.")
            self.reject()
            return

        # 기존 위젯 제거
        while self.vbox.count():
            item = self.vbox.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        self._scene_widgets.clear()

        for sc in scenes:
            if not isinstance(sc, dict):
                continue

            sid = str(sc.get("id", ""))
            seg_count = int(sc.get("seg_count") or 1)
            if seg_count < 1:
                seg_count = 1
            if seg_count > 3:
                seg_count = 3

            card = QtWidgets.QGroupBox(f"Scene: {sid}  (seg_count={seg_count})", self.inner)
            form = QtWidgets.QVBoxLayout(card)

            # 참고 정보(시작 이미지/자막) — 읽기 전용
            ref_row = QtWidgets.QHBoxLayout()
            form.addLayout(ref_row)

            te_ref = QtWidgets.QPlainTextEdit(card)
            te_ref.setReadOnly(True)
            te_ref.setFixedHeight(90)
            ref_txt = []
            ref_txt.append(f"prompt_img: {sc.get('prompt_img','')}")
            ref_txt.append(f"subtitle/lyric: {sc.get('subtitle') or sc.get('lyric') or ''}")
            te_ref.setPlainText("\n".join(ref_txt))
            ref_row.addWidget(te_ref, 1)

            btn_ai = QtWidgets.QPushButton("AI 수정 요청 (한글→영어 반영)", card)
            btn_ai.setMinimumWidth(220)
            ref_row.addWidget(btn_ai)

            # 편집 영역
            grid = QtWidgets.QGridLayout()
            form.addLayout(grid)

            def _mk_pair(row_idx: int, title: str, kor_val: str, en_val: str):
                lbl = QtWidgets.QLabel(title, card)
                grid.addWidget(lbl, row_idx, 0)

                te_k = QtWidgets.QPlainTextEdit(card)
                te_k.setPlaceholderText(f"{title}_kor (한글)")
                te_k.setPlainText(kor_val or "")
                te_k.setFixedHeight(70)

                te_en = QtWidgets.QPlainTextEdit(card)
                te_en.setPlaceholderText(f"{title} (English)")
                te_en.setPlainText(en_val or "")
                te_en.setFixedHeight(70)

                grid.addWidget(te_k, row_idx, 1)
                grid.addWidget(te_en, row_idx, 2)
                return te_k, te_en

            te_p1_k, te_p1 = _mk_pair(
                0, "prompt_1",
                sc.get("prompt_1_kor", ""),
                sc.get("prompt_1", "")
            )
            te_p2_k = te_p2 = None
            te_p3_k = te_p3 = None

            if seg_count >= 2:
                te_p2_k, te_p2 = _mk_pair(
                    1, "prompt_2",
                    sc.get("prompt_2_kor", ""),
                    sc.get("prompt_2", "")
                )

            if seg_count >= 3:
                te_p3_k, te_p3 = _mk_pair(
                    2, "prompt_3",
                    sc.get("prompt_3_kor", ""),
                    sc.get("prompt_3", "")
                )

            self.vbox.addWidget(card)

            bundle = {
                "sid": sid,
                "seg_count": seg_count,
                "btn_ai": btn_ai,
                "te_p1_k": te_p1_k, "te_p1": te_p1,
                "te_p2_k": te_p2_k, "te_p2": te_p2,
                "te_p3_k": te_p3_k, "te_p3": te_p3,
                "ref_visual": sc.get("prompt_img", ""),
                "ref_story": sc.get("subtitle") or sc.get("lyric") or "",
            }
            self._scene_widgets.append(bundle)

            # "해당 씬만" AI 반영
            btn_ai.clicked.connect(lambda _=False, b=bundle: self._apply_ai_for_one_scene(b))

        self.vbox.addStretch(1)

    def _apply_ai_for_one_scene(self, bundle: dict):
        """
        클릭한 해당 씬만:
        - kor 필드는 유지/보존
        - en(prompt_1/2/3)은 kor를 기반으로 AI 번역/연출로 덮어씀
        """
        seg_count = int(bundle.get("seg_count", 1))
        sid = bundle.get("sid", "")

        p1_k = (bundle["te_p1_k"].toPlainText() or "").strip()
        p2_k = (bundle["te_p2_k"].toPlainText() or "").strip() if seg_count >= 2 and bundle.get("te_p2_k") else ""
        p3_k = (bundle["te_p3_k"].toPlainText() or "").strip() if seg_count >= 3 and bundle.get("te_p3_k") else ""

        # kor가 비어있으면 의미가 없으니 중단
        if not p1_k and (seg_count == 1):
            QtWidgets.QMessageBox.warning(self, "알림", f"{sid}: prompt_1_kor가 비어있습니다.")
            return
        if seg_count >= 2 and not p2_k:
            QtWidgets.QMessageBox.warning(self, "알림", f"{sid}: prompt_2_kor가 비어있습니다.")
            return
        if seg_count >= 3 and not p3_k:
            QtWidgets.QMessageBox.warning(self, "알림", f"{sid}: prompt_3_kor가 비어있습니다.")
            return

        visual = bundle.get("ref_visual", "")
        story = bundle.get("ref_story", "")

        sys_msg = (
            "You are a senior I2V prompt translator/director.\n"
            "Task: translate the given Korean segment prompts into English for I2V generation.\n\n"
            "Hard constraints:\n"
            "1) English only in output.\n"
            "2) Product reference MUST be ONLY: 'the object'.\n"
            "3) Do NOT introduce new background/location/props/characters.\n"
            "4) Keep camera moves subtle; avoid cuts/time jumps.\n"
            "Return JSON only."
        )

        # 출력 키 구성
        if seg_count == 1:
            out_fmt = '{ "prompt_1": "..." }'
        elif seg_count == 2:
            out_fmt = '{ "prompt_1": "...", "prompt_2": "..." }'
        else:
            out_fmt = '{ "prompt_1": "...", "prompt_2": "...", "prompt_3": "..." }'

        user_msg = f"""
[Start Image Description 참고]
{visual}

[Story Context 참고]
{story}

[Korean Segment Prompts]
prompt_1_kor: {p1_k}
prompt_2_kor: {p2_k}
prompt_3_kor: {p3_k}

[Rules]
- Translate faithfully.
- Replace any product noun into ONLY 'the object'.
- Keep continuity; each segment begins from previous end.
- Minimal background description.

[Output JSON]
{out_fmt}
"""

        try:
            ai = self._lazy_ai()
            resp = ai.ask_smart(sys_msg, user_msg, prefer="openai")

            import json, re
            txt = re.sub(r"```json|```", "", resp).strip()
            s = txt.find("{")
            e = txt.rfind("}")
            if s == -1 or e == -1 or e <= s:
                raise ValueError("AI 응답에서 JSON 객체를 찾지 못했습니다.")
            parsed = json.loads(txt[s:e + 1])

            # 영어 필드만 덮어씀 (kor는 그대로 둠)
            bundle["te_p1"].setPlainText(parsed.get("prompt_1", "") or bundle["te_p1"].toPlainText())
            if seg_count >= 2 and bundle.get("te_p2"):
                bundle["te_p2"].setPlainText(parsed.get("prompt_2", "") or bundle["te_p2"].toPlainText())
            if seg_count >= 3 and bundle.get("te_p3"):
                bundle["te_p3"].setPlainText(parsed.get("prompt_3", "") or bundle["te_p3"].toPlainText())

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "오류", f"{sid}: AI 반영 실패\n{e}")

    def _save(self):
        import json

        try:
            data = self._read_json()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "오류", f"video.json 로드 실패:\n{e}")
            return

        scenes = data.get("scenes", [])
        if not isinstance(scenes, list):
            QtWidgets.QMessageBox.critical(self, "오류", "video.json의 scenes가 리스트가 아닙니다.")
            return

        # sid -> sc 매핑
        sc_map = {}
        for sc in scenes:
            if isinstance(sc, dict):
                sc_map[str(sc.get("id", ""))] = sc

        # 위젯 -> JSON 반영
        for b in self._scene_widgets:
            sid = b.get("sid", "")
            seg_count = int(b.get("seg_count", 1))
            target = sc_map.get(sid)
            if not target:
                continue

            # kor 저장
            target["prompt_1_kor"] = (b["te_p1_k"].toPlainText() or "").strip()
            if seg_count >= 2 and b.get("te_p2_k"):
                target["prompt_2_kor"] = (b["te_p2_k"].toPlainText() or "").strip()
            if seg_count >= 3 and b.get("te_p3_k"):
                target["prompt_3_kor"] = (b["te_p3_k"].toPlainText() or "").strip()

            # en 저장
            target["prompt_1"] = (b["te_p1"].toPlainText() or "").strip()
            if seg_count >= 2 and b.get("te_p2"):
                target["prompt_2"] = (b["te_p2"].toPlainText() or "").strip()
            if seg_count >= 3 and b.get("te_p3"):
                target["prompt_3"] = (b["te_p3"].toPlainText() or "").strip()

        try:
            self._write_json(data)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "오류", f"video.json 저장 실패:\n{e}")
            return

        QtWidgets.QMessageBox.information(self, "완료", "저장되었습니다.")


class ImageViewerDialog(QtWidgets.QDialog):
    """
    [New] 이미지를 스크롤 가능한 원본 크기로 보여주는 다이얼로그
    """
    def __init__(self, img_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"원본 이미지 보기: {Path(img_path).name}")
        self.resize(1000, 900)  # 기본 창 크기

        layout = QtWidgets.QVBoxLayout(self)

        # 스크롤 영역 (이미지가 화면보다 클 수 있으므로)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        # 이미지 라벨
        self.lbl_image = QtWidgets.QLabel()
        self.lbl_image.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_image.setStyleSheet("background-color: #333;") # 배경을 어둡게 해서 이미지 집중

        # 이미지 로드
        pix = QtGui.QPixmap(img_path)
        if not pix.isNull():
            self.lbl_image.setPixmap(pix)
            # 이미지가 너무 작으면 적당히 확대할 수도 있지만, 원본 확인용이라 그대로 둠
        else:
            self.lbl_image.setText("이미지를 불러올 수 없습니다.")
            self.lbl_image.setStyleSheet("color: white;")

        scroll.setWidget(self.lbl_image)

        # 닫기 버튼
        btn_close = QtWidgets.QPushButton("닫기", self)
        btn_close.setMinimumHeight(40)
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

class VideoBuildDialog(QtWidgets.QDialog):
    """
    [수정됨] 영상 제작 설정 창
    - settings.py의 기본값(FPS 16, 405x720, Step 6)을 UI 초기값으로 반영
    - video.json이 있다면 그 값을 우선 사용
    """

    def __init__(self, product_dir: str, product_data: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"영상 제작 - {product_data.get('product_name', '')}")
        self.resize(1000, 850)

        self.product_dir = str(product_dir)
        self.product_data = product_data or {}
        # video_shopping.json (기획 데이터)
        self.video_json_path = str(Path(self.product_dir) / "video_shopping.json")
        # video.json (최종 쇼츠 데이터)
        self.target_video_json = str(Path(self.product_dir) / "video.json")

        root = QtWidgets.QVBoxLayout(self)

        # ── [New] 상단 공통 설정 영역 (settings.py 연동) ──
        setting_group = QtWidgets.QGroupBox("영상 제작 공통 설정 (Settings)", self)
        setting_group.setStyleSheet("QGroupBox { font-weight: bold; background-color: #f9f9f9; }")
        setting_layout = QtWidgets.QHBoxLayout(setting_group)
        setting_layout.setSpacing(20)

        # 1. 해상도
        setting_layout.addWidget(QtWidgets.QLabel("해상도:"))
        self.combo_res = QtWidgets.QComboBox()
        # settings에 정의된 해상도 목록 사용
        res_choices = getattr(settings, "IMG_SIZE_CHOICES", ["405x720", "540x960", "720x1280", "1080x1920"])

        # (중요) 목록에 기본값(405x720)이 없으면 강제로 추가하기 위해 확인
        default_wh = getattr(settings, "DEFAULT_IMG_SIZE", (405, 720))
        if isinstance(default_wh, (list, tuple)):
            def_str = f"{default_wh[0]}x{default_wh[1]}"
            # 리스트에 문자열 형태나 튜플 형태로 존재하는지 체크 (단순화를 위해 문자열 변환 비교)
            str_choices = [f"{x[0]}x{x[1]}" if isinstance(x, (list, tuple)) else str(x) for x in res_choices]
            if def_str not in str_choices:
                res_choices.insert(0, default_wh)  # 맨 앞에 추가

        for rc in res_choices:
            if isinstance(rc, (list, tuple)) and len(rc) == 2:
                self.combo_res.addItem(f"{rc[0]}x{rc[1]}", tuple(rc))
            elif isinstance(rc, str):
                parts = rc.lower().split("x")
                if len(parts) == 2:
                    self.combo_res.addItem(rc, (int(parts[0]), int(parts[1])))
                else:
                    self.combo_res.addItem(rc, (1080, 1920))  # fallback
        setting_layout.addWidget(self.combo_res)

        # 2. FPS
        setting_layout.addWidget(QtWidgets.QLabel("FPS:"))
        self.combo_fps = QtWidgets.QComboBox()
        fps_choices = getattr(settings, "MOVIE_FPS_CHOICES", [16, 24, 30, 60])
        for fps in fps_choices:
            self.combo_fps.addItem(str(fps), int(fps))
        setting_layout.addWidget(self.combo_fps)

        # 3. Steps
        setting_layout.addWidget(QtWidgets.QLabel("Steps:"))
        self.spin_steps = QtWidgets.QSpinBox()
        self.spin_steps.setRange(1, 100)
        setting_layout.addWidget(self.spin_steps)

        # ── [New] 4. 폰트 설정 (Font Family) ──
        setting_layout.addWidget(QtWidgets.QLabel("폰트:"))
        self.combo_font = QtWidgets.QFontComboBox()  # 시스템 폰트 자동 로드
        self.combo_font.setEditable(False)  # 직접 입력 방지

        # 폰트 필터: 모든 폰트 보여주기 (필요시 ScalableFonts 등으로 제한 가능)
        self.combo_font.setFontFilters(QtWidgets.QFontComboBox.AllFonts)

        # 기본값 선택 ("맑은 고딕" 등 settings 값 찾아서 선택)
        def_font = getattr(settings, "DEFAULT_FONT_FAMILY", "Malgun Gothic")
        self.combo_font.setCurrentFont(QtGui.QFont(def_font))

        # 너비 제한 (너무 길어지지 않게)
        self.combo_font.setMaximumWidth(150)
        setting_layout.addWidget(self.combo_font)

        # ── [New] 5. 제목 크기 (Title Size) ──
        setting_layout.addWidget(QtWidgets.QLabel("제목:"))
        self.spin_title_size = QtWidgets.QSpinBox()
        self.spin_title_size.setRange(10, 300)
        # settings 값 적용
        def_title = getattr(settings, "DEFAULT_TITLE_FONT_SIZE", 55)
        self.spin_title_size.setValue(def_title)
        self.spin_title_size.setSuffix(" px")  # 단위 표시
        setting_layout.addWidget(self.spin_title_size)

        # ── [New] 6. 내레이션 크기 (Narration Size) ──
        setting_layout.addWidget(QtWidgets.QLabel("자막:"))
        self.spin_narr_size = QtWidgets.QSpinBox()
        self.spin_narr_size.setRange(10, 200)
        # settings 값 적용
        def_narr = getattr(settings, "DEFAULT_NARRATION_FONT_SIZE", 25)
        self.spin_narr_size.setValue(def_narr)
        self.spin_narr_size.setSuffix(" px")
        setting_layout.addWidget(self.spin_narr_size)

        setting_layout.addStretch(1)
        root.addWidget(setting_group)

        # ── 탭 위젯 ──
        self.tabs = QtWidgets.QTabWidget(self)
        root.addWidget(self.tabs, 1)

        # ── 탭1: 시나리오 (JSON) ─────────────────────────
        tab1 = QtWidgets.QWidget(self)
        t1 = QtWidgets.QVBoxLayout(tab1)

        # 옵션 행
        opt_row = QtWidgets.QHBoxLayout()
        t1.addLayout(opt_row)

        opt_row.addWidget(QtWidgets.QLabel("Scene 개수:", tab1))
        self.sp_scene_count = QtWidgets.QSpinBox(tab1)
        self.sp_scene_count.setRange(3, 12)
        self.sp_scene_count.setValue(6)
        opt_row.addWidget(self.sp_scene_count)

        opt_row.addSpacing(10)
        opt_row.addWidget(QtWidgets.QLabel("후크 강도(1~5):", tab1))
        self.sp_hook = QtWidgets.QSpinBox(tab1)
        self.sp_hook.setRange(1, 5)
        self.sp_hook.setValue(3)
        opt_row.addWidget(self.sp_hook)

        opt_row.addSpacing(10)
        opt_row.addWidget(QtWidgets.QLabel("스타일:", tab1))
        self.cb_style = QtWidgets.QComboBox(tab1)
        self.cb_style.addItems(["news_hook", "daily", "meme"])
        opt_row.addWidget(self.cb_style)

        opt_row.addStretch(1)

        # JSON 관련 3단계 버튼 그룹
        btn_group = QtWidgets.QGroupBox("1단계: 시나리오 작업 순서", tab1)
        btn_layout = QtWidgets.QHBoxLayout(btn_group)

        self.btn_draft = QtWidgets.QPushButton("1. 초안 생성 (Draft)", tab1)
        self.btn_edit_json = QtWidgets.QPushButton("2. 초안 수정 (편집)", tab1)
        self.btn_enrich = QtWidgets.QPushButton("3. AI 상세화 (최종 변환)", tab1)
        self.btn_enrich.setStyleSheet("color: blue; font-weight: bold;")

        # [NEW] 4. 비디오 JSON 생성 (쇼츠 호환용)
        self.btn_make_video_json = QtWidgets.QPushButton("4. 비디오 JSON 생성 (AI)", tab1)
        self.btn_make_video_json.setStyleSheet("color: darkgreen; font-weight: bold;")
        self.btn_make_video_json.setToolTip(
            "상단의 설정(FPS, 해상도, Steps)을 적용하여 video.json을 생성하고,\n"
            "AI로 장면 연출(프롬프트)을 채웁니다."
        )

        self.btn_final_edit = QtWidgets.QPushButton("5. 최종안 수정", self)
        self.btn_final_edit.setToolTip("video.json의 prompt_1_kor/prompt_2_kor... 를 편집하고, 한글→영어 반영(AI) 가능")

        self.btn_make_video_json.clicked.connect(self.on_click_make_video_json)
        self.btn_final_edit.clicked.connect(self.on_final_edit_clicked)

        btn_layout.addWidget(self.btn_draft)
        btn_layout.addWidget(self.btn_edit_json)
        btn_layout.addWidget(self.btn_enrich)
        btn_layout.addWidget(self.btn_make_video_json)
        btn_layout.addWidget(self.btn_final_edit)

        t1.addWidget(btn_group)

        # 유틸 버튼
        util_row = QtWidgets.QHBoxLayout()
        self.btn_open_product_dir = QtWidgets.QPushButton("상품 폴더 열기", tab1)
        self.btn_refresh_view = QtWidgets.QPushButton("화면 새로고침", tab1)

        util_row.addStretch(1)
        util_row.addWidget(self.btn_refresh_view)
        util_row.addWidget(self.btn_open_product_dir)
        t1.addLayout(util_row)

        self.te_preview = QtWidgets.QPlainTextEdit(tab1)
        self.te_preview.setReadOnly(True)
        self.te_preview.setPlaceholderText("여기에 JSON 내용이 표시됩니다.")
        t1.addWidget(self.te_preview, 1)

        self.tabs.addTab(tab1, "1. 시나리오(JSON)")

        # ── 탭2: 제작 패널 (이미지/영상/합치기) ─────────────────────────
        tab2 = QtWidgets.QWidget(self)
        t2 = QtWidgets.QVBoxLayout(tab2)

        # 설정 행
        chk_row = QtWidgets.QHBoxLayout()
        self.cb_img_skip = QtWidgets.QCheckBox("이미지: 존재하면 스킵", tab2)
        self.cb_img_skip.setChecked(True)
        self.cb_mov_skip = QtWidgets.QCheckBox("영상: 존재하면 스킵", tab2)
        self.cb_mov_skip.setChecked(True)

        chk_row.addWidget(self.cb_img_skip)
        chk_row.addSpacing(15)
        chk_row.addWidget(self.cb_mov_skip)
        chk_row.addStretch(1)
        t2.addLayout(chk_row)

        t2.addSpacing(20)

        # 개별 단계 실행 그룹
        grp_actions = QtWidgets.QGroupBox("개별 단계 실행 (테스트용)", tab2)
        lay_actions = QtWidgets.QHBoxLayout(grp_actions)
        lay_actions.setSpacing(10)

        self.btn_gen_images = QtWidgets.QPushButton("2. 이미지 생성", tab2)
        self.btn_gen_images.setMinimumHeight(40)

        self.btn_gen_movies = QtWidgets.QPushButton("3. 영상 생성 (I2V)", tab2)
        self.btn_gen_movies.setMinimumHeight(40)

        self.btn_media_edit = QtWidgets.QPushButton("3.5 이미지/영상 수정 (검수)", tab2)
        self.btn_media_edit.setMinimumHeight(40)
        self.btn_media_edit.setStyleSheet("color: #d32f2f; font-weight: bold;")

        self.btn_merge = QtWidgets.QPushButton("4. 영상 합치기 (Merge)", tab2)
        self.btn_merge.setMinimumHeight(40)

        lay_actions.addWidget(self.btn_gen_images)
        lay_actions.addWidget(self.btn_gen_movies)
        lay_actions.addWidget(self.btn_media_edit)
        lay_actions.addWidget(self.btn_merge)

        t2.addWidget(grp_actions)

        # 폴더 열기 편의 버튼
        t2.addSpacing(10)
        row_folders = QtWidgets.QHBoxLayout()
        self.btn_open_imgs = QtWidgets.QPushButton("imgs 폴더 열기", tab2)
        self.btn_open_clips = QtWidgets.QPushButton("clips 폴더 열기", tab2)
        row_folders.addWidget(self.btn_open_imgs)
        row_folders.addWidget(self.btn_open_clips)
        row_folders.addStretch(1)
        t2.addLayout(row_folders)

        t2.addStretch(1)
        self.tabs.addTab(tab2, "2. 제작(이미지/영상)")

        # ── 하단: 전체 실행 + 로그 ─────────────────────────
        bottom_row = QtWidgets.QHBoxLayout()
        root.addLayout(bottom_row)

        self.btn_run_all = QtWidgets.QPushButton("🚀 전체 실행 (1~4 단계 일괄 수행)", self)
        self.btn_run_all.setStyleSheet("background-color: #d4f7d4; font-weight: bold; font-size: 14px; padding: 8px;")
        self.btn_run_all.setMinimumHeight(50)

        self.btn_close = QtWidgets.QPushButton("닫기", self)
        self.btn_close.setMinimumHeight(50)

        bottom_row.addWidget(self.btn_run_all, 1)
        bottom_row.addWidget(self.btn_close)

        self.log = QtWidgets.QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        root.addWidget(self.log, 0)

        # ── 시그널 연결 ─────────────────────────
        self.btn_close.clicked.connect(self.close)

        self.btn_draft.clicked.connect(self.on_draft_clicked)
        self.btn_edit_json.clicked.connect(self.on_edit_json_clicked)
        self.btn_enrich.clicked.connect(self.on_enrich_clicked)

        self.btn_refresh_view.clicked.connect(self._refresh_preview_if_exists)

        self.btn_open_product_dir.clicked.connect(lambda: self._open_path(self.product_dir))
        self.btn_open_imgs.clicked.connect(lambda: self._open_path(str(Path(self.product_dir) / "imgs")))
        self.btn_open_clips.clicked.connect(lambda: self._open_path(str(Path(self.product_dir) / "clips")))

        self.btn_gen_images.clicked.connect(self.on_gen_images_clicked)
        self.btn_gen_movies.clicked.connect(self.on_gen_movies_clicked)
        self.btn_media_edit.clicked.connect(self.on_media_edit_clicked)
        self.btn_merge.clicked.connect(self.on_merge_clicked)
        self.btn_run_all.clicked.connect(self.on_run_all_clicked)

        # [핵심] 초기값 로드 실행
        self._load_initial_settings()
        self._refresh_preview_if_exists()

    def _load_initial_settings(self):
        """
        video.json이 있으면 거기서 값을 읽고, 없으면 settings.py 기본값 사용
        """
        # 1. settings.py 기본값 확보
        # IMG_SIZE가 튜플 (405, 720) 형태로 온다고 가정
        def_res = getattr(settings, "DEFAULT_IMG_SIZE", (405, 720))
        def_fps = int(getattr(settings, "DEFAULT_MOVIE_FPS", 16))
        def_steps = int(getattr(settings, "DEFAULT_T2I_STEPS", 6))
        def_font_family = getattr(settings, "DEFAULT_FONT_FAMILY", "Malgun Gothic")
        def_title_size = int(getattr(settings, "DEFAULT_TITLE_FONT_SIZE", 55))
        def_narr_size = int(getattr(settings, "DEFAULT_NARRATION_FONT_SIZE", 25))

        # 2. video.json 파일 확인 (덮어쓰기)
        vpath = Path(self.target_video_json)
        if vpath.exists():
            try:
                with open(vpath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # FPS 읽기
                file_fps = data.get("fps") or data.get("defaults", {}).get("movie", {}).get("fps")
                if file_fps:
                    def_fps = int(file_fps)

                # 해상도 읽기
                img_def = data.get("defaults", {}).get("image", {})
                w = img_def.get("width")
                h = img_def.get("height")
                if w and h:
                    def_res = (int(w), int(h))

                # Steps 읽기
                gen_def = data.get("defaults", {}).get("generator", {})
                st = gen_def.get("steps")
                if st:
                    def_steps = int(st)


                sub_def = data.get("defaults", {}).get("subtitle", {})
                # 폰트 읽기
                if sub_def.get("font_family"):
                    def_font_family = sub_def["font_family"]
                # 제목 크기 읽기
                if sub_def.get("title_size"):
                    def_title_size = int(sub_def["title_size"])
                # 내레이션 크기 읽기
                if sub_def.get("narr_size"):
                    def_narr_size = int(sub_def["narr_size"])

            except Exception as e:
                self._append_log(f"⚠ video.json 설정 로드 실패 (기본값 사용): {e}")

        # 3. UI에 값 반영 (여기가 중요: 콤보박스 선택)

        # (1) 해상도 선택
        # combo_res의 data(UserRole)는 튜플 (w, h) 형태임
        found_res = False
        target_res_val = tuple(def_res) if isinstance(def_res, list) else def_res

        for i in range(self.combo_res.count()):
            item_data = self.combo_res.itemData(i)
            # 튜플 비교 (405, 720) == (405, 720)
            if item_data == target_res_val:
                self.combo_res.setCurrentIndex(i)
                found_res = True
                break

        if not found_res:
            # 목록에 없으면 추가 후 선택
            label = f"{target_res_val[0]}x{target_res_val[1]}"
            self.combo_res.addItem(label, target_res_val)
            self.combo_res.setCurrentIndex(self.combo_res.count() - 1)

        # (2) FPS 선택
        found_fps = False
        for i in range(self.combo_fps.count()):
            if self.combo_fps.itemData(i) == def_fps:
                self.combo_fps.setCurrentIndex(i)
                found_fps = True
                break

        if not found_fps:
            self.combo_fps.addItem(str(def_fps), def_fps)
            self.combo_fps.setCurrentIndex(self.combo_fps.count() - 1)

        # (3) Steps 설정
        self.spin_steps.setValue(def_steps)

        # (4) 폰트 설정
        self.combo_font.setCurrentFont(QtGui.QFont(def_font_family))
        # (5) 제목크기 설정
        self.spin_title_size.setValue(def_title_size)
        # (6) 내레이션 크기 설정
        self.spin_narr_size.setValue(def_narr_size)

        # 로그 확인용
        # self._append_log(f"[Init] UI 설정 적용: {target_res_val}, {def_fps}fps, {def_steps}steps")

    def _get_current_settings(self):
        """현재 UI에 설정된 (width, height, fps, steps) 반환"""
        w, h = self.combo_res.currentData()
        fps = self.combo_fps.currentData()
        steps = self.spin_steps.value()
        font_family = self.combo_font.currentFont().family()  # 폰트 이름 문자열
        title_size = self.spin_title_size.value()
        narr_size = self.spin_narr_size.value()
        return w, h, fps, steps, font_family, title_size, narr_size

    # ── 헬퍼 함수들 ─────────────────────────
    def _append_log(self, msg: str):
        self.log.appendPlainText(msg)

    def _open_path(self, path: str):
        p = Path(path)
        if not p.exists():
            self._append_log(f"⚠ 경로 없음: {p}")
            return
        try:
            os.startfile(str(p))
        except Exception as e:
            self._append_log(f"⚠ 열기 실패: {e}")

    def _options(self) -> BuildOptions:
        # [수정] 반환값 7개 언패킹 (fps 사용)
        _, _, fps, _, _, _, _ = self._get_current_settings()
        return BuildOptions(
            scene_count=int(self.sp_scene_count.value()),
            style=str(self.cb_style.currentText()),
            hook_level=int(self.sp_hook.value()),
            fps=fps,
        )

    def _refresh_preview_if_exists(self):
        p = Path(self.target_video_json)
        if not p.exists():
            p = Path(self.video_json_path)

        if p.exists():
            try:
                txt = p.read_text(encoding="utf-8")
                self.te_preview.setPlainText(f"[{p.name}]\n{txt}")
            except Exception as e:
                self.te_preview.setPlainText(f"읽기 실패: {e}")
        else:
            self.te_preview.setPlainText("(아직 파일이 없습니다. '초안 생성'을 먼저 진행하세요)")

    # ── 핸들러들 ─────────────────────────

    def on_draft_clicked(self):
        """1단계: 초안(Draft) JSON 생성"""

        def job(progress):
            progress("[Draft] 초안 생성 시작...")
            builder = ShoppingVideoJsonBuilder(on_progress=progress)
            out_path = builder.create_draft(
                product_dir=self.product_dir,
                product_data=self.product_data,
                options=self._options(),
            )
            return str(out_path)

        def done(ok, res, err):
            if ok:
                self._append_log(f"✅ 초안 생성 완료: {res}")
                self._refresh_preview_if_exists()
            else:
                self._append_log(f"❌ 초안 생성 실패: {err}")

        run_job_with_progress_async(self, "초안 생성", job, on_done=done)

    def on_edit_json_clicked(self):
        """1단계: 초안 수정 (SceneEditDialog 호출)"""
        # video_shopping.json이 대상
        p = Path(self.video_json_path)
        if not p.exists():
            self._append_log("⚠ 수정할 파일(video_shopping.json)이 없습니다. 먼저 '초안 생성'을 하세요.")
            return

        dlg = SceneEditDialog(self.video_json_path, parent=self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self._append_log("📝 초안 수정이 완료되었습니다.")
            self._refresh_preview_if_exists()

    def on_enrich_clicked(self):
        """1단계: AI 상세화 (Draft -> Final)"""
        if not Path(self.video_json_path).exists():
            self._append_log("⚠ 수정할 JSON 파일이 없습니다. '초안 생성'을 먼저 하세요.")
            return

        # [Fix] UI 설정값 읽기 (Width, Height, FPS, Steps)
        w, h, fps, steps, _, _, _ = self._get_current_settings()

        def job(progress):
            progress(f"[Enrich] AI 상세화 작업 시작 (FPS:{fps}, Size:{w}x{h})...")
            builder = ShoppingVideoJsonBuilder(on_progress=progress)

            # [Fix] 설정값 전달
            out_path = builder.enrich_video_json(
                video_json_path=self.video_json_path,
                product_data=self.product_data,
                ui_width=w,
                ui_height=h,
                ui_fps=fps,
                ui_steps=steps
            )
            return str(out_path)

        def done(ok, res, err):
            if ok:
                self._append_log(f"✅ AI 상세화 완료! (오디오 시간 측정됨)")
                self._refresh_preview_if_exists()
            else:
                self._append_log(f"❌ AI 상세화 실패: {err}")

        run_job_with_progress_async(self, "AI 상세화", job, on_done=done)

    def on_click_make_video_json(self):
        """
        [시나리오 탭] video.json(쇼츠 호환) 생성 및 AI 상세화 버튼
        """
        if not self.product_dir or not Path(self.product_dir).exists():
            QtWidgets.QMessageBox.warning(self, "오류", "프로젝트 경로가 유효하지 않습니다.")
            return

        src_path = Path(self.product_dir) / "video_shopping.json"
        if not src_path.exists():
            QtWidgets.QMessageBox.warning(self, "알림", "video_shopping.json이 없습니다.\n먼저 '쇼핑데이터생성' 3단계까지 진행해주세요.")
            return

        # UI 설정값 가져오기
        w, h, fps, steps, font, t_size, n_size = self._get_current_settings()

        self._append_log(f"⚙ 설정: {w}x{h}, {fps}fps, {steps}steps, 폰트={font}(T:{t_size}, N:{n_size})")

        self.btn_make_video_json.setEnabled(False)



        def job(progress_callback):
            # AI 객체 확보
            ai_client = getattr(self, "ai", None)
            if not ai_client:
                from app.utils import AI
                ai_client = AI()

            result = convert_shopping_to_video_json_with_ai(
                project_dir=self.product_dir,
                ai_client=ai_client,
                fps=fps,
                width=w,
                height=h,
                steps=steps,  # [New] steps 전달
                on_progress=progress_callback
            )
            return result

        def done(ok, payload, err):
            self.btn_make_video_json.setEnabled(True)
            if ok:
                QtWidgets.QMessageBox.information(self, "완료", f"video.json 생성 및 AI 연출 완료!\n경로: {payload}")
                self._refresh_preview_if_exists()
            else:
                QtWidgets.QMessageBox.critical(self, "실패", f"작업 실패:\n{err}")

        from app.utils import run_job_with_progress_async
        run_job_with_progress_async(
            owner=self,
            title="video.json 생성 및 AI 상세화",
            job=job,
            on_done=done
        )

    def on_final_edit_clicked(self):
        """
        5단계: 최종안 수정 (video.json의 prompt_1_kor/prompt_2_kor... 편집 + 한글→영어 반영)
        """
        from pathlib import Path
        from PyQt5 import QtWidgets

        if not Path(self.target_video_json).exists():
            QtWidgets.QMessageBox.warning(self, "알림", "video.json이 없습니다.\n'4. 비디오 JSON 생성'을 먼저 진행해주세요.")
            return

        dlg = FinalEditDialog(self.target_video_json, self.product_dir, parent=self)
        dlg.exec_()
        self._append_log("🔧 최종안 수정 창을 닫았습니다.")
        self._refresh_preview_if_exists()

    def on_gen_images_clicked(self):
        """
        쇼핑탭 이미지 생성은 video_shopping.json 기준으로 수행해야 함.
        - Step1: prompt_img_1 (Z-image)
        - Step2: prompt_img_2 (Qwen composite, image1=Step1결과, image2=제품 image.png)
        """
        if not Path(self.video_json_path).exists():
            QtWidgets.QMessageBox.warning(self, "알림", "video_shopping.json이 없습니다.\n'1~3단계'를 먼저 진행해주세요.")
            return

        # UI 설정값
        w, h, _, steps, _, _, _ = self._get_current_settings()

        def job(progress):
            progress(f"[Image] (SHOP) 2-Step 이미지 생성 시작 ({w}x{h}, steps={steps})")
            progress(f"[Image][DBG] video_shopping_json={self.video_json_path}")

            from app.utils import load_json
            from pathlib import Path

            # ✅ 제품 이미지 경로 계산: product_dir / product.image_file
            prod_path = None
            try:
                doc = load_json(self.video_json_path, {}) or {}
                product = doc.get("product") or {}
                img_file = (product.get("image_file") or "").strip()  # 보통 image.png
                if img_file:
                    cand = (Path(self.product_dir) / img_file).resolve()
                    prod_path = str(cand)
            except Exception:
                prod_path = None

            progress(f"[Image][DBG] product_image_path={prod_path}")

            from app.shopping_video_build import build_shopping_images_2step

            build_shopping_images_2step(
                video_json_path=self.video_json_path,  # ✅ video_shopping.json 사용
                source_json_path=self.video_json_path,  # ✅ prompt_img_1/2도 여기서 읽음
                product_image_path=prod_path,  # ✅ Step2 image2(제품)로 들어갈 경로
                ui_width=w,
                ui_height=h,
                steps=steps,
                on_progress=lambda d: progress(d.get("msg", ""))
            )
            return "OK"

        def done(ok, res, err):
            if ok:
                self._append_log("✅ (SHOP) 이미지 생성 완료")
            else:
                self._append_log(f"❌ (SHOP) 이미지 생성 실패: {err}")

        run_job_with_progress_async(self, "이미지 생성", job, on_done=done)

    def on_media_edit_clicked(self):
        if not Path(self.video_json_path).exists():
            self._append_log("⚠ video_shopping.json이 없습니다.")
            return

        dlg = MediaEditDialog(self.video_json_path, self.product_dir, parent=self)
        dlg.exec_()
        self._append_log("🔧 미디어 수정/검수 창을 닫았습니다.")

    def on_gen_movies_clicked(self):
        if not Path(self.target_video_json).exists():
            QtWidgets.QMessageBox.warning(self, "알림", "video.json이 없습니다.\n'4. 비디오 JSON 생성'을 먼저 진행해주세요.")
            return

        # UI 설정값 읽기 (w, h, fps, steps, ...)
        _, _, fps, _, _, _, _ = self._get_current_settings()

        def job(progress):
            progress(f"[Movie] I2V 영상 생성 시작 (Long Take, {fps} fps)...")
            # [변경] Long 버전 빌드 함수 호출

            build_shots_with_i2v_long(
                project_dir=self.product_dir,
                total_frames=0, # video.json 값 사용
                ui_fps=fps,
                on_progress=lambda d: progress(d.get("msg", ""))
            )
            return "OK"

        def done(ok, res, err):
            if ok:
                self._append_log("✅ 영상 생성 및 업스케일 완료")
            else:
                self._append_log(f"❌ 영상 생성 실패: {err}")

        run_job_with_progress_async(self, "영상 생성", job, on_done=done)

    def on_merge_clicked(self):
        if not Path(self.target_video_json).exists():
            self._append_log("⚠ video.json이 없습니다.")
            return

        # merge 직전 UI 설정을 json에 강제로 반영
        try:
            w, h, fps, steps, font_family, title_size, narr_size = self._get_current_settings()

            p = Path(self.target_video_json)
            data = {}
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
            except Exception:
                data = {}

            defaults = data.get("defaults")
            if not isinstance(defaults, dict):
                defaults = {}
                data["defaults"] = defaults

            sub = defaults.get("subtitle")
            if not isinstance(sub, dict):
                sub = {}
                defaults["subtitle"] = sub

            # UI에서 읽은 값 저장 (merge_movies가 이 값을 읽도록 강제)
            sub["font_family"] = str(font_family)
            sub["title_size"] = int(title_size)
            sub["narr_size"] = int(narr_size)

            # (참고) 다른 설정도 같이 기록해두면 추후 일관성 유지에 도움
            img = defaults.get("image")
            if not isinstance(img, dict):
                img = {}
                defaults["image"] = img
            img["width"] = int(w)
            img["height"] = int(h)

            mov = defaults.get("movie")
            if not isinstance(mov, dict):
                mov = {}
                defaults["movie"] = mov
            mov["fps"] = int(fps)

            gen = defaults.get("generator")
            if not isinstance(gen, dict):
                gen = {}
                defaults["generator"] = gen
            gen["steps"] = int(steps)

            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self._append_log(
                f"🧷 merge 직전 UI 설정 저장 완료: font='{font_family}', title={title_size}, narr={narr_size}"
            )

        except Exception as e:
            self._append_log(f"⚠ UI 설정을 video.json에 반영 실패(그래도 merge는 시도): {e}")

        def job(progress):
            progress("[Merge] 영상 합치기 시작.")
            gen = ShoppingMovieGenerator(on_progress=progress)
            gen.merge_movies(self.target_video_json)
            return "OK"

        def done(ok, res, err):
            if ok:
                self._append_log("✅ 영상 합치기 완료")
            else:
                self._append_log(f"❌ 영상 합치기 실패: {err}")

        run_job_with_progress_async(self, "영상 합치기", job, on_done=done)

    def on_run_all_clicked(self):
        # 전체 실행 로직... (순차 실행 필요, 일단 메시지 처리)
        self.on_click_make_video_json()
        self._append_log("ℹ '전체 실행'은 현재 순차 실행 로직 보완이 필요합니다. 각 버튼을 순서대로 눌러주세요.")


class ShoppingWidget(QtWidgets.QWidget):
    """
    쇼핑/쿠팡/쇼츠 자동화용 메인 UI.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.products_dir = Path(r"C:\my_games\shorts_make\products")
        self.products_dir.mkdir(parents=True, exist_ok=True)
        self._uploaded_image_path = ""  # 사용자가 업로드한 이미지 원본 경로
        self._build_ui()
        self._wire_signals()

    # ────────────────────────────────────────
    # UI 구성
    # ────────────────────────────────────────
    def _build_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # 1) 상단 검색 영역
        search_box = QtWidgets.QGroupBox("상품 검색 / 추천", self)
        search_layout = QtWidgets.QGridLayout(search_box)
        search_layout.setContentsMargins(8, 8, 8, 8)
        search_layout.setHorizontalSpacing(6)
        search_layout.setVerticalSpacing(4)

        lbl_keyword = QtWidgets.QLabel("키워드:", search_box)
        self.le_keyword = QtWidgets.QLineEdit(search_box)
        self.le_keyword.setPlaceholderText("예: 강아지 옷, 블루투스 이어폰, 게이밍 키보드 ...")

        lbl_category = QtWidgets.QLabel("카테고리:", search_box)
        self.combo_category = QtWidgets.QComboBox(search_box)
        self.combo_category.addItems([
            "전체",
            "강아지/반려동물",
            "패션(남성)",
            "패션(여성)",
            "디지털/가전",
            "생활/주방",
            "뷰티/코스메틱",
            "식품",
        ])

        self.btn_search_popular = QtWidgets.QPushButton("인기템 불러오기", search_box)
        self.btn_search_keyword = QtWidgets.QPushButton("키워드 검색", search_box)

        search_layout.addWidget(lbl_keyword, 0, 0)
        search_layout.addWidget(self.le_keyword, 0, 1, 1, 3)
        search_layout.addWidget(lbl_category, 1, 0)
        search_layout.addWidget(self.combo_category, 1, 1)
        search_layout.addWidget(self.btn_search_popular, 1, 2)
        search_layout.addWidget(self.btn_search_keyword, 1, 3)

        # 2) 중앙 영역: 좌(트리 리스트) / 우(상세 미리보기)
        center_splitter = QtWidgets.QSplitter(self)
        center_splitter.setOrientation(QtCore.Qt.Horizontal)

        # 2-1) 좌측: 선택한 리스트 (TreeWidget)
        left_widget = QtWidgets.QWidget(center_splitter)
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        lbl_list = QtWidgets.QLabel("선택한 리스트 (Del:삭제 / F2:수정)", left_widget)

        self.tree_selected = QtWidgets.QTreeWidget(left_widget)
        self.tree_selected.setHeaderLabels(["상품명 / 제목"])  # 컬럼 단순화 (상세정보는 우측에서)
        self.tree_selected.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tree_selected.setAlternatingRowColors(True)
        self.tree_selected.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked | QtWidgets.QAbstractItemView.EditKeyPressed)

        # 컨텍스트 메뉴
        self.tree_selected.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree_selected.customContextMenuRequested.connect(self._show_context_menu)

        left_layout.addWidget(lbl_list)
        left_layout.addWidget(self.tree_selected)

        # 2-2) 상품 상세/프리뷰 (우측)
        right_widget = QtWidgets.QWidget(center_splitter)
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(4)

        lbl_detail = QtWidgets.QLabel("선택 상품 상세", right_widget)

        # 썸네일 + 기본 정보
        thumb_group = QtWidgets.QGroupBox("이미지 / 기본 정보", right_widget)
        thumb_layout = QtWidgets.QHBoxLayout(thumb_group)
        thumb_layout.setContentsMargins(8, 8, 8, 8)
        thumb_layout.setSpacing(8)

        self.lbl_thumbnail = QtWidgets.QLabel(thumb_group)
        self.lbl_thumbnail.setFixedSize(150, 150)
        self.lbl_thumbnail.setFrameShape(QtWidgets.QFrame.Box)
        self.lbl_thumbnail.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_thumbnail.setText("이미지\n(ComfyUI 예정)")

        info_form = QtWidgets.QFormLayout()
        info_form.setLabelAlignment(QtCore.Qt.AlignRight)
        info_form.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        self.le_title = QtWidgets.QLineEdit(thumb_group)
        self.le_title.setReadOnly(True)
        self.le_title.setPlaceholderText("상품명 또는 제목")

        self.le_price = QtWidgets.QLineEdit(thumb_group)
        self.le_price.setPlaceholderText("가격 (참고용)")

        info_form.addRow("상품명:", self.le_title)
        info_form.addRow("가격:", self.le_price)

        thumb_layout.addWidget(self.lbl_thumbnail)
        thumb_layout.addLayout(info_form)

        # [추가] 이미지 업로드 버튼
        self.btn_upload_image = QtWidgets.QPushButton("이미지 업로드", thumb_group)
        self.btn_upload_image.setMinimumWidth(150)
        thumb_layout.addWidget(self.btn_upload_image)

        # 링크 및 설명
        link_group = QtWidgets.QGroupBox("링크 / 설명", right_widget)
        link_layout = QtWidgets.QFormLayout(link_group)
        link_layout.setLabelAlignment(QtCore.Qt.AlignRight)

        self.le_product_url = QtWidgets.QLineEdit(link_group)
        self.le_product_url.setPlaceholderText("https://...")

        self.le_affiliate_url = QtWidgets.QLineEdit(link_group)
        self.le_affiliate_url.setPlaceholderText("쿠팡 파트너스 딥링크 (자동 생성 예정)")

        self.te_description = QtWidgets.QPlainTextEdit(link_group)
        self.te_description.setPlaceholderText("상품 소개/쇼츠 시나리오용 요약 문구 (AI가 자동 생성 예정)...")

        self.te_description.setFixedHeight(80)

        link_layout.addRow("상품 원본 링크:", self.le_product_url)
        link_layout.addRow("파트너스 링크:", self.le_affiliate_url)
        link_layout.addRow("요약/설명:", self.te_description)

        # 우측 버튼들
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(6)
        btn_row.addStretch(1)

        self.btn_test = QtWidgets.QPushButton("테스트", right_widget)
        self.btn_load_list = QtWidgets.QPushButton("1단계: 리스트가져오기", right_widget)
        self.btn_load_b = QtWidgets.QPushButton("리스트불러오기", right_widget)
        self.btn_make_script = QtWidgets.QPushButton("쇼츠 스크립트 만들기 (준비중)", right_widget)
        self.btn_make_infok = QtWidgets.QPushButton("인포크링크 텍스트 생성 (준비중)", right_widget)

        btn_row.addWidget(self.btn_test)
        btn_row.addWidget(self.btn_load_list)
        btn_row.addWidget(self.btn_load_b)
        btn_row.addWidget(self.btn_make_script)
        btn_row.addWidget(self.btn_make_infok)

        right_layout.addWidget(lbl_detail)
        right_layout.addWidget(thumb_group)
        right_layout.addWidget(link_group)

        # [추가] 확정 버튼 (설명 아래에 위치)
        confirm_row = QtWidgets.QHBoxLayout()
        confirm_row.addStretch(1)
        self.btn_confirm = QtWidgets.QPushButton("확정", right_widget)
        self.btn_video_build = QtWidgets.QPushButton("영상제작", right_widget)

        self.btn_confirm.setMinimumWidth(120)
        confirm_row.addWidget(self.btn_confirm)
        confirm_row.addWidget(self.btn_video_build)  # ✅ 추가
        right_layout.addLayout(confirm_row)


        right_layout.addLayout(btn_row)

        # [추가] 등록상품/새상품 관리 버튼
        manage_row = QtWidgets.QHBoxLayout()
        manage_row.setSpacing(6)
        manage_row.addStretch(1)

        self.btn_new_product = QtWidgets.QPushButton("새상품", right_widget)
        self.btn_load_registered = QtWidgets.QPushButton("등록된 상품 불러오기", right_widget)
        self.btn_delete_registered = QtWidgets.QPushButton("등록 상품 삭제하기", right_widget)

        self.btn_new_product.setMinimumWidth(90)
        self.btn_load_registered.setMinimumWidth(180)
        self.btn_delete_registered.setMinimumWidth(160)

        manage_row.addWidget(self.btn_new_product)
        manage_row.addWidget(self.btn_load_registered)
        manage_row.addWidget(self.btn_delete_registered)

        right_layout.addLayout(manage_row)

        center_splitter.addWidget(left_widget)
        center_splitter.addWidget(right_widget)
        center_splitter.setStretchFactor(0, 3)
        center_splitter.setStretchFactor(1, 4)

        # 3) 하단 로그 영역
        log_group = QtWidgets.QGroupBox("로그", self)
        log_layout = QtWidgets.QVBoxLayout(log_group)
        log_layout.setContentsMargins(8, 4, 8, 8)

        self.log = QtWidgets.QPlainTextEdit(log_group)
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setPlaceholderText("쇼핑/쿠팡/이슈/인포크링크 자동화 관련 로그가 여기에 표시됩니다.")
        log_layout.addWidget(self.log)

        main_layout.addWidget(search_box)
        main_layout.addWidget(center_splitter, 1)
        main_layout.addWidget(log_group, 0)

    # ────────────────────────────────────────
    # 시그널 연결
    # ────────────────────────────────────────
    def _wire_signals(self):
        self.btn_search_keyword.clicked.connect(self.on_search_keyword)
        self.btn_search_popular.clicked.connect(self.on_search_popular)

        # 트리 선택 변경 시
        self.tree_selected.currentItemChanged.connect(self.on_tree_item_selected)

        # 기존 버튼들
        self.btn_load_list.clicked.connect(
            lambda: self._run_stage1_collect_issue_list("1단계버튼")
        )
        self.btn_make_script.clicked.connect(self.on_make_script_clicked)
        self.btn_make_infok.clicked.connect(self.on_make_infok_clicked)
        self.btn_load_b.clicked.connect(self.on_load_b_clicked)

        # 테스트 버튼은 지금 단계에서는 API 미사용: 그대로 둬도 되고, 로그만 찍게 해도 됨
        self.btn_test.clicked.connect(self.on_test_clicked)

        # [추가] 이미지 업로드 / 확정
        self.btn_upload_image.clicked.connect(self.on_upload_image_clicked)
        self.btn_confirm.clicked.connect(self.on_confirm_clicked)
        self.btn_video_build.clicked.connect(self.on_video_build_clicked)


        # [추가] 등록상품/새상품 관리
        self.btn_new_product.clicked.connect(self.on_new_product_clicked)
        self.btn_load_registered.clicked.connect(self.on_load_registered_clicked)
        self.btn_delete_registered.clicked.connect(self.on_delete_registered_clicked)

    # ────────────────────────────────────────
    # 공통 로그 함수
    # ────────────────────────────────────────
    def append_log(self, msg: str):
        self.log.appendPlainText(msg)

    def _norm_text(self, s: str) -> str:
        """트리 표시용 이모지/접두어 제거 후 비교용 텍스트로 정규화"""
        s = (s or "").strip()
        # 예: "📂 가정용 디자인 소화기" -> "가정용 디자인 소화기"
        for prefix in ("📂", "🗂", "✅", "☑", "■", "□"):
            if s.startswith(prefix):
                s = s[len(prefix):].strip()
        return s

    def _find_top_item_by_title(self, title: str):
        """TopLevel(폴더/이슈)에서 동일 타이틀을 찾아 반환(없으면 None)"""
        want = self._norm_text(title)
        for i in range(self.tree_selected.topLevelItemCount()):
            it = self.tree_selected.topLevelItem(i)
            if self._norm_text(it.text(0)) == want:
                return it
        return None

    def _child_name_set(self, parent_item) -> set:
        """부모 폴더 아래 자식 상품명 set(정규화 비교)"""
        out = set()
        if not parent_item:
            return out
        for i in range(parent_item.childCount()):
            out.add(self._norm_text(parent_item.child(i).text(0)))
        return out

    def _sort_tree(self):
        """폴더 오름차순 + 폴더 내부 상품 오름차순"""
        # 1) top-level 정렬
        self.tree_selected.sortItems(0, QtCore.Qt.AscendingOrder)
        # 2) 각 폴더 내부 정렬
        for i in range(self.tree_selected.topLevelItemCount()):
            top = self.tree_selected.topLevelItem(i)
            if top.childCount() > 0:
                top.sortChildren(0, QtCore.Qt.AscendingOrder)

    def _products_base_dir(self) -> Path:
        return Path(r"C:\my_games\shorts_make\products")

    def _safe_product_folder_name(self, name: str) -> str:
        safe = sanitize_title(name or "")
        safe = re.sub(r"[\\/:*?\"<>|]", "_", safe).strip()
        return safe[:120] if safe else "untitled"

    def _strip_confirm_prefix(self, text: str) -> str:
        # 트리 표시용 "✅ " 프리픽스 제거
        return re.sub(r"^\s*✅\s*", "", (text or "")).strip()

    def _set_tree_confirmed_text(self, item: QtWidgets.QTreeWidgetItem, raw_name: str):
        raw_name = (raw_name or "").strip()
        if not raw_name:
            return
        cur = (item.text(0) or "")
        if not re.match(r"^\s*✅\s*", cur):
            item.setText(0, f"✅ {raw_name}")
        else:
            # 이미 ✅가 있어도 뒤 텍스트가 바뀌었으면 교정
            item.setText(0, f"✅ {raw_name}")

    def _normalize_product_display_name(self, display: str) -> str:
        """
        트리 표시용 텍스트에서 실제 상품명을 복원.
        예) "✅ 아이폰 케이스" -> "아이폰 케이스"
        """
        s = (display or "").strip()
        s = re.sub(r"^[✅\s]+", "", s).strip()
        return s

    def _get_product_name_from_item(self, item: QtWidgets.QTreeWidgetItem) -> str:
        """
        상품 아이템에서 '실제 상품명'을 우선적으로 가져온다.
        - UserRole의 product_name 우선
        - 없으면 text(0)에서 ✅ 같은 표시 제거
        """
        if not item:
            return ""
        data = item.data(0, QtCore.Qt.UserRole) or {}
        if isinstance(data, dict):
            pn = (data.get("product_name") or "").strip()
            if pn:
                return pn
        return self._normalize_product_display_name(item.text(0))

    def _try_load_saved_product(self, product_name: str) -> dict:
        """
        products/{상품명}/product.json 이 존재하면 읽어서 dict로 반환.
        없으면 {} 반환.
        """
        # ✅ 표시 제거 후 파일 탐색
        product_name = self._strip_confirm_prefix(product_name)

        base = self._products_base_dir()
        folder = base / self._safe_product_folder_name(product_name)
        json_path = folder / "product.json"
        if not json_path.exists():
            return {}

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {}
            data.setdefault("product_dir", str(folder))
            return data
        except Exception as e:
            self.append_log(f"⚠ 저장된 product.json 로드 실패: {json_path} ({e})")
            return {}

    def _hydrate_product_item_from_disk(self, product_item: QtWidgets.QTreeWidgetItem) -> dict:
        """
        현재 트리의 '상품(자식)' 아이템에 대해,
        디스크에 저장된 product.json이 있으면 UserRole에 merge + ✅ 표시 반영.
        반환: merge된 최종 dict
        """
        display_name = (product_item.text(0) or "").strip()
        raw_name = self._strip_confirm_prefix(display_name)
        if not raw_name:
            return product_item.data(0, QtCore.Qt.UserRole) or {}

        saved = self._try_load_saved_product(raw_name)
        if not saved:
            return product_item.data(0, QtCore.Qt.UserRole) or {}

        item_data = product_item.data(0, QtCore.Qt.UserRole) or {}
        if not isinstance(item_data, dict):
            item_data = {"type": "product"}

        # saved(product.json)의 키들을 우선 반영
        item_data.update(saved)

        # 호환성: 기존 코드가 url 키를 보는 경우가 있어서 동기화
        if item_data.get("product_url") and not item_data.get("url"):
            item_data["url"] = item_data["product_url"]

        # ✅ 확정 표시용 마킹(원하면 사용)
        item_data["type"] = "product"
        item_data["is_confirmed"] = True

        # UserRole 업데이트
        product_item.setData(0, QtCore.Qt.UserRole, item_data)

        # ✅ 트리 텍스트도 즉시 갱신 (리스트 불러오기 직후에도 체크가 보이게)
        self._set_tree_confirmed_text(product_item, raw_name)

        return item_data

    # ────────────────────────────────────────
    # 트리 위젯 관리
    # ────────────────────────────────────────
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Delete:
            self._delete_selected_tree_item()
        else:
            super().keyPressEvent(event)

    def _show_context_menu(self, pos):
        item = self.tree_selected.itemAt(pos)
        menu = QtWidgets.QMenu(self)

        action_add_prod = menu.addAction("상품 추가")
        action_del = menu.addAction("삭제")

        action = menu.exec_(self.tree_selected.mapToGlobal(pos))

        if action == action_del:
            self._delete_selected_tree_item()
        elif action == action_add_prod:
            target = item if item else None
            if target:
                parent = target if not target.parent() else target.parent()
                self._add_dummy_product(parent)

    def _delete_selected_tree_item(self):
        items = self.tree_selected.selectedItems()
        if not items:
            return

        for item in items:
            parent = item.parent()
            if parent:
                parent.removeChild(item)
            else:
                # top-level 삭제: '내 상품' 같은 루트는 실수로 지우지 않게 막고 싶으면 여기에서 차단
                if self._norm_text(item.text(0)) == "내 상품":
                    continue
                idx = self.tree_selected.indexOfTopLevelItem(item)
                self.tree_selected.takeTopLevelItem(idx)

        self._sort_tree()

    def _add_dummy_product(self, parent_item):
        child = QtWidgets.QTreeWidgetItem(parent_item)
        child.setText(0, "새 상품")
        child.setFlags(child.flags() | QtCore.Qt.ItemIsEditable)
        parent_item.setExpanded(True)

    def on_tree_item_selected(self, current, previous):
        """
        트리 아이템 선택 시 우측 상세창 채우기
        - 상품(자식) 선택 시: 저장된 product.json이 있으면 자동 복구하여 UI에 반영
        - 이슈(부모) 선택 시: reason 표시
        """
        # 1) UI 초기화 (할인/평점은 프로젝트 상태에 따라 있을 수도/없을 수도 있으므로 안전하게 처리)
        self.le_title.clear()
        self.le_price.clear()

        if hasattr(self, "le_discount"):
            self.le_discount.clear()
        if hasattr(self, "le_rating"):
            self.le_rating.clear()

        self.le_product_url.clear()
        self.le_affiliate_url.clear()
        self.te_description.clear()

        self.lbl_thumbnail.setText("이미지")
        self.lbl_thumbnail.setPixmap(QtGui.QPixmap())

        if not current:
            return

        parent = current.parent()
        item_data = current.data(0, QtCore.Qt.UserRole) or {}

        self.le_title.setText(current.text(0))

        if parent:
            # === 상품(자식) ===
            # ✅ 클릭 시에도 디스크의 product.json을 읽어 merge (있으면 복구)
            item_data = self._hydrate_product_item_from_disk(current)

            # price
            price = item_data.get("price", "")
            self.le_price.setText(str(price))

            # url (호환: url/product_url 둘 다 인정)
            product_url = item_data.get("product_url") or item_data.get("url") or ""
            self.le_product_url.setText(str(product_url))

            # affiliate url
            affiliate_url = item_data.get("affiliate_url") or ""
            self.le_affiliate_url.setText(str(affiliate_url))

            # description (호환: description 없으면 reason fallback)
            desc = (item_data.get("description") or "").strip()
            if not desc:
                my_reason = (item_data.get("reason") or "").strip()
                if not my_reason:
                    parent_data = parent.data(0, QtCore.Qt.UserRole) or {}
                    my_reason = (parent_data.get("reason") or "").strip()
                desc = my_reason
            self.te_description.setPlainText(desc)

            # thumbnail
            product_dir = item_data.get("product_dir") or ""
            image_file = item_data.get("image_file") or ""
            img_path = ""
            if product_dir and image_file:
                p = Path(product_dir) / image_file
                if p.exists():
                    img_path = str(p)

            if img_path:
                pix = QtGui.QPixmap(img_path)
                if not pix.isNull():
                    pix = pix.scaled(self.lbl_thumbnail.size(), QtCore.Qt.KeepAspectRatio,
                                     QtCore.Qt.SmoothTransformation)
                    self.lbl_thumbnail.setPixmap(pix)
                else:
                    self.lbl_thumbnail.setText("이미지 로드 실패")
            else:
                self.lbl_thumbnail.setText("상품 이미지(없음/미등록)")

        else:
            # === 이슈(부모) ===
            self.lbl_thumbnail.setText("카테고리/이슈")
            reason = (item_data.get("reason") or "").strip()
            self.te_description.setPlainText(reason)

    # ────────────────────────────────────────
    # 슬롯들
    # ────────────────────────────────────────
    def on_search_keyword(self):
        keyword = self.le_keyword.text().strip()
        self.append_log(f"🔍 키워드 검색 (구현예정): {keyword}")

    def on_search_popular(self):
        self.append_log("🔥 인기템 리스트 (구현예정)")

    def on_test_clicked(self):
        self.append_log("🧪 [테스트] 현재는 API 미사용 단계입니다. '확정'으로 수동 입력값 저장을 테스트하세요.")

    def on_upload_image_clicked(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "상품 이미지 선택",
            "",
            "Image Files (*.png *.jpg *.jpeg *.webp);;All Files (*.*)"
        )
        if not file_path:
            return

        self._uploaded_image_path = file_path

        pix = QtGui.QPixmap(file_path)
        if pix.isNull():
            self.lbl_thumbnail.setText("이미지 로드 실패")
            self.lbl_thumbnail.setPixmap(QtGui.QPixmap())
            return

        pix = pix.scaled(
            self.lbl_thumbnail.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
        self.lbl_thumbnail.setPixmap(pix)

        self.append_log(f"🖼 이미지 선택: {file_path}")

    def on_confirm_clicked(self):
        current = self.tree_selected.currentItem()
        if not current or not current.parent():
            QtWidgets.QMessageBox.warning(self, "확정 불가", "좌측 트리에서 '상품(자식)' 항목을 선택하세요.")
            return

        # ✅ 표시 제거한 원본명으로 저장/폴더명 결정
        product_name = self._strip_confirm_prefix((current.text(0) or "").strip())
        if not product_name:
            QtWidgets.QMessageBox.warning(self, "확정 불가", "상품명이 비어 있습니다.")
            return

        base_dir = Path(r"C:\my_games\shorts_make\products")
        base_dir.mkdir(parents=True, exist_ok=True)

        safe_name = sanitize_title(product_name)
        safe_name = re.sub(r"[\\/:*?\"<>|]", "_", safe_name).strip()
        if not safe_name:
            safe_name = "untitled"

        product_dir = base_dir / safe_name
        product_dir.mkdir(parents=True, exist_ok=True)

        # 이미지 복사(선택한 경우)
        image_file = ""
        if self._uploaded_image_path and os.path.exists(self._uploaded_image_path):
            ext = os.path.splitext(self._uploaded_image_path)[1].lower()
            if ext not in (".jpg", ".jpeg", ".png", ".webp"):
                ext = ".jpg"
            dst = product_dir / f"image{ext}"
            try:
                shutil.copy2(self._uploaded_image_path, dst)
                image_file = dst.name
            except Exception as e:
                self.append_log(f"⚠ 이미지 저장 실패: {e}")

        payload = {
            "product_name": product_name,
            "price": (self.le_price.text() or "").strip(),
            "product_url": (self.le_product_url.text() or "").strip(),
            "affiliate_url": (self.le_affiliate_url.text() or "").strip(),
            "description": (self.te_description.toPlainText() or "").strip(),
            "image_file": image_file,
            "product_dir": str(product_dir),
            "confirmed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "manual",
        }

        json_path = product_dir / "product.json"
        try:
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "저장 실패", f"product.json 저장 중 오류:\n{e}")
            return

        # 트리 아이템 UserRole에도 저장값 반영
        item_data = current.data(0, QtCore.Qt.UserRole) or {}
        if not isinstance(item_data, dict):
            item_data = {"type": "product"}
        item_data.update(payload)
        item_data["type"] = "product"
        item_data["is_confirmed"] = True
        current.setData(0, QtCore.Qt.UserRole, item_data)

        # ✅ 저장 즉시 트리에도 체크 표시
        self._set_tree_confirmed_text(current, product_name)

        self.append_log(f"✅ 저장 완료: {json_path}")

    def on_video_build_clicked(self):
        """
        [영상제작]
        - 현재 선택된 상품(자식) 기준으로 product.json 존재 확인
        - 존재하면 VideoBuildDialog를 열어 shopping_video_build.py와 연동
        """
        current = self.tree_selected.currentItem()
        if not current or not current.parent():
            QtWidgets.QMessageBox.warning(self, "영상제작 불가", "좌측 트리에서 '상품(자식)' 항목을 선택하세요.")
            return

        # product_dir 추정: 확정 시 payload에 product_dir 저장해두는 구조를 이미 쓰고 있음
        item_data = current.data(0, QtCore.Qt.UserRole) or {}
        if not isinstance(item_data, dict):
            item_data = {}

        product_dir = (item_data.get("product_dir") or "").strip()
        product_name = (item_data.get("product_name") or current.text(0) or "").strip()

        # product_dir이 없으면 표준 경로로 추정
        if not product_dir:
            base_dir = Path(r"C:\my_games\shorts_make\products")
            # sanitize_title을 이미 프로젝트에서 쓰고 있으니 있으면 사용, 없으면 안전치환
            try:
                safe = sanitize_title(product_name)
            except Exception:
                safe = re.sub(r"[\\/:*?\"<>|]", "_", product_name).strip() or "untitled"
            product_dir = str(base_dir / safe)

        pdir = Path(product_dir)
        pjson = pdir / "product.json"

        if not pjson.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "영상제작 불가",
                "해당 상품은 아직 확정 저장(product.json)이 없습니다.\n먼저 우측의 '확정'을 눌러 저장한 뒤 진행하세요.",
            )
            return

        try:
            product_data = json.loads(pjson.read_text(encoding="utf-8"))
            if not isinstance(product_data, dict):
                raise ValueError("product.json이 dict가 아닙니다.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "product.json 읽기 실패", f"product.json 로드 중 오류:\n{e}")
            return

        dlg = VideoBuildDialog(product_dir=str(pdir), product_data=product_data, parent=self)
        dlg.exec_()

    def _ensure_root_for_my_products(self) -> QtWidgets.QTreeWidgetItem:
        """
        새상품을 넣을 루트가 없으면 생성: "📦 내 상품"
        """
        for i in range(self.tree_selected.topLevelItemCount()):
            it = self.tree_selected.topLevelItem(i)
            if (it.text(0) or "").strip() == "📦 내 상품":
                return it

        root = QtWidgets.QTreeWidgetItem(self.tree_selected)
        root.setText(0, "📦 내 상품")
        root.setFlags(root.flags() | QtCore.Qt.ItemIsEditable)
        root.setData(0, QtCore.Qt.UserRole, {"type": "issue", "reason": "내가 직접 등록한 상품"})
        root.setExpanded(True)
        return root

    def _ensure_my_products_root(self):
        root = self._find_top_item_by_title("내 상품")
        if root is None:
            root = QtWidgets.QTreeWidgetItem(self.tree_selected)
            root.setText(0, "📂 내 상품")
            root.setFlags(root.flags() | QtCore.Qt.ItemIsEditable)
            root.setData(0, QtCore.Qt.UserRole, {"type": "my_products"})
            root.setExpanded(True)
            self._sort_tree()
        return root

    def on_new_product_clicked(self):
        """
        새상품:
        - '미확정 새 상품'은 최대 3개까지만 생성
        - (확정하면 '새 상품'이 실제 상품명으로 바뀌므로) 항상 최대 3개 유지 UX 가능
        """
        root = self._ensure_my_products_root()

        # 미확정 새상품 개수 카운트: type='draft' 이거나 이름이 '새 상품'인 항목
        draft_count = 0
        for i in range(root.childCount()):
            ch = root.child(i)
            data = ch.data(0, QtCore.Qt.UserRole) or {}
            if isinstance(data, dict) and data.get("type") == "draft":
                draft_count += 1
            elif self._norm_text(ch.text(0)) in ("새 상품", "새상품"):
                draft_count += 1

        if draft_count >= 3:
            self.append_log("ℹ 새 상품은 미확정 상태로 최대 3개까지만 생성됩니다. (먼저 확정하거나 삭제하세요)")
            return

        child = QtWidgets.QTreeWidgetItem(root)
        child.setText(0, "새 상품")
        child.setFlags(child.flags() | QtCore.Qt.ItemIsEditable)
        child.setData(0, QtCore.Qt.UserRole, {"type": "draft"})  # ✅ 미확정 표시
        root.setExpanded(True)

        self._sort_tree()
        self.append_log("➕ 새 상품을 추가했습니다. (이름/내용 입력 후 우측에서 확정)")

    def _scan_registered_products(self) -> list[dict]:
        """
        C:\\my_games\\shorts_make\\products\\*/product.json 스캔
        반환: [{"product_name":..., "product_dir":..., "json_path":...}, ...]
        """
        base = self._products_base_dir()
        if not base.exists():
            return []

        result = []
        for folder in base.iterdir():
            if not folder.is_dir():
                continue
            json_path = folder / "product.json"
            if not json_path.exists():
                continue
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                pn = (data.get("product_name") or folder.name).strip()
                result.append({
                    "product_name": pn,
                    "product_dir": str(folder),
                    "json_path": str(json_path),
                    "data": data,
                })
            except Exception:
                continue

        # 이름 기준 정렬
        result.sort(key=lambda x: x.get("product_name", ""))
        return result

    def on_load_registered_clicked(self):
        """기존 상품 불러오기 버튼 클릭 시 (TreeWidget 버전 수정)"""
        if not self.products_dir.exists():
            QtWidgets.QMessageBox.warning(self, "오류", f"폴더가 없습니다: {self.products_dir}")
            return

        # LoadProductDialog 사용
        dlg = LoadProductDialog(self.products_dir, self)

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            paths = dlg.get_selected_paths()
            added_count = 0

            # 1. '내 상품' 루트 폴더 확보 (없으면 자동 생성)
            root = self._ensure_my_products_root()

            # 2. 이미 있는 상품인지 확인하기 위해 이름 목록 수집
            existing_names = set()
            for i in range(root.childCount()):
                existing_names.add(root.child(i).text(0))

            last_item = None

            for p in paths:
                product_name = p.name

                # 중복이면 추가하지 않음
                if product_name in existing_names:
                    continue

                # 3. 트리에 자식 아이템으로 추가
                item = QtWidgets.QTreeWidgetItem(root)
                item.setText(0, product_name)

                # 데이터 설정 (product.json 자동 로드를 위한 초기값)
                # product_dir 경로를 명시적으로 심어줌
                init_data = {"product_dir": str(p), "type": "product"}
                item.setData(0, QtCore.Qt.UserRole, init_data)

                # 4. 디스크 정보(이미지, 가격 등) 즉시 로드 및 '✅' 표시 반영
                self._hydrate_product_item_from_disk(item)

                added_count += 1
                last_item = item

            if added_count > 0:
                root.setExpanded(True)
                self.append_log(f"상품 {added_count}개를 '내 상품' 목록에 추가했습니다.")

                # [핵심] 마지막으로 추가된 항목 자동 선택
                if last_item:
                    self.tree_selected.setCurrentItem(last_item)
                    # 트리 선택 시 상세정보 로드 함수는 시그널로 자동 연결되어 있으므로 setCurrentItem만 하면 됨
            else:
                self.append_log("추가된 상품이 없습니다 (이미 목록에 있거나 선택 안 함).")

    def on_delete_registered_clicked(self):
        """
        등록 상품 삭제:
        - 좌측 트리에서 '상품(자식)'이 선택되어 있을 때만 동작
        - product_dir(폴더) 삭제 후 트리에서도 제거
        """
        cur = self.tree_selected.currentItem()
        if not cur or not cur.parent():
            QtWidgets.QMessageBox.warning(self, "삭제 불가", "좌측 트리에서 삭제할 '상품(자식)'을 선택하세요.")
            return

        data = cur.data(0, QtCore.Qt.UserRole) or {}
        if not isinstance(data, dict):
            data = {}

        pname = self._get_product_name_from_item(cur)
        product_dir = (data.get("product_dir") or "").strip()

        if not product_dir:
            # product_dir가 없다면 이름으로 추정
            product_dir = str(self._products_base_dir() / self._safe_product_folder_name(pname))

        if not os.path.isdir(product_dir):
            QtWidgets.QMessageBox.warning(self, "삭제 불가", f"상품 폴더를 찾지 못했습니다:\n{product_dir}")
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "등록상품 삭제",
            f"아래 등록상품을 완전히 삭제합니다.\n\n- 상품명: {pname}\n- 폴더: {product_dir}\n\n계속할까요?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        try:
            shutil.rmtree(product_dir)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "삭제 실패", f"폴더 삭제 중 오류:\n{e}")
            return

        # 트리에서 제거
        parent = cur.parent()
        parent.removeChild(cur)

        # UI 초기화
        self.le_title.clear()
        self.le_price.clear()
        self.le_product_url.clear()
        self.le_affiliate_url.clear()
        self.te_description.clear()
        self.lbl_thumbnail.setText("이미지")
        self.lbl_thumbnail.setPixmap(QtGui.QPixmap())

        self.append_log(f"🗑 등록상품 삭제 완료: {pname} ({product_dir})")

    def _run_stage1_collect_issue_list(self, trigger_label: str):
        self.append_log(f"🧩 [1단계] ({trigger_label}) 쇼핑 이슈 + AI 상품 후보 전체 파이프라인 시작...")

        def job(progress):
            progress({"msg": f"[1단계] 쇼핑 이슈 수집 중... (_a.json)"})
            path_a = save_issue_list_for_shopping_all(on_progress=progress)

            progress({"msg": f"[1단계] AI로 상품 후보 분석 중... (_a -> _b)"})
            path_b = save_issue_list_for_shopping_ai_b_from_a(
                on_progress=progress,
                a_path=str(path_a),
            )
            return {"b_path": str(path_b) if path_b else ""}

        def done(ok, payload, err):
            if (not ok) or (err is not None):
                self.append_log(f"❌ 실패: {err}")
                return

            b_path = payload.get("b_path", "")
            if not b_path:
                self.append_log(f"⚠ _b 경로 없음.")
                return

            self.append_log(f"✅ 완료. 리스트 팝업을 엽니다.")
            self._open_list_dialog(b_path)

        run_job_with_progress_async(
            owner=self,
            title=f"쇼핑 이슈+AI 후보 (1단계)",
            job=job,
            on_done=done,
        )

    def on_make_script_clicked(self):
        title = self.le_title.text().strip()
        if not title:
            self.append_log("⚠ 먼저 상품을 선택해 주세요.")
            return
        self.append_log(f"✏ 쇼츠 스크립트 생성 시도: {title}")

    def on_make_infok_clicked(self):
        title = self.le_title.text().strip()
        if not title:
            self.append_log("⚠ 상품을 먼저 선택해 주세요.")
            return
        self.append_log(f"🧱 인포크링크 생성 시도: {title}")

    def on_load_b_clicked(self):
        default_dir = str(Path(r"C:\my_games\shorts_make\issue_list"))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "리스트 불러오기 (_b.json / _selected.json)", default_dir, "JSON Files (*.json);;All Files (*.*)"
        )
        if path:
            self._open_list_dialog(path)

    def _open_list_dialog(self, path: str):
        import json
        try:
            path_obj = Path(path)
            txt = path_obj.read_text(encoding="utf-8")
            items = json.loads(txt)
            if not isinstance(items, list):
                raise ValueError("JSON 구조가 리스트가 아닙니다.")

            # 🔥 [수정됨] 자동 매칭 로드 로직
            # "_b.json"을 열었을 때, "_b_selected.json"이 있으면 미리 읽어서 '나의 선택'에 넣어줌
            pre_selected = []
            if "_selected" not in path_obj.stem:
                sel_name = f"{path_obj.stem}_selected.json"
                sel_path = path_obj.parent / sel_name
                if sel_path.exists():
                    try:
                        sel_txt = sel_path.read_text(encoding="utf-8")
                        pre_selected = json.loads(sel_txt)
                        self.append_log(f"ℹ 기존 선택 내역 발견: {len(pre_selected)}개 항목을 우측 리스트에 복구합니다.")
                    except Exception as e:
                        self.append_log(f"⚠ 선택 내역 로드 실패: {e}")

            # Dialog 생성 시 pre_selected_items 전달
            dlg = IssueListViewerDialog(
                items,
                parent=self,
                title=f"리스트 보기: {path_obj.name}",
                original_path=path,
                pre_selected_items=pre_selected
            )

            if dlg.exec_() == QtWidgets.QDialog.Accepted:
                selected_data = dlg.get_selected_data_full()
                if selected_data:
                    self._add_data_to_main_tree(selected_data)
                    self.append_log(f"📥 {len(selected_data)}개의 카테고리 세트가 로드되었습니다.")
        except Exception as e:
            self.append_log(f"⚠ 리스트 열기 실패: {e}")

    def _add_data_to_main_tree(self, data_list: list[dict]):
        """
        데이터를 메인 트리에 추가(중복 방지 + 폴더 병합 + 정렬).
        data_list 구조: [{"title": "...", "extra": {"reason": "...", "products": [...]}}, ...]
        """
        if not data_list:
            return

        for item in data_list:
            title = (item.get("title") or "제목 없음").strip()
            extra = item.get("extra", {}) or {}
            reason = (extra.get("reason") or "").strip()
            products = extra.get("products", [])

            # ✅ 기존 동일 폴더(이슈 제목)가 있으면 재사용, 없으면 생성
            root = self._find_top_item_by_title(title)
            if root is None:
                root = QtWidgets.QTreeWidgetItem(self.tree_selected)
                root.setText(0, f"📂 {title}")
                root.setFlags(root.flags() | QtCore.Qt.ItemIsEditable)
                root.setData(0, QtCore.Qt.UserRole, {"reason": reason, "type": "issue"})
            else:
                # reason 업데이트(기존 reason이 비어있으면 채움, 또는 최신으로 덮어씌우고 싶으면 항상 덮어쓰기)
                root_data = root.data(0, QtCore.Qt.UserRole) or {}
                if isinstance(root_data, dict):
                    if reason and (not root_data.get("reason")):
                        root_data["reason"] = reason
                    root_data.setdefault("type", "issue")
                    root.setData(0, QtCore.Qt.UserRole, root_data)

            # ✅ 자식(상품) 중복 방지
            existing = self._child_name_set(root)

            def _add_child_product(p_name: str, p_data: dict):
                nm = (p_name or "").strip()
                if not nm:
                    return
                if self._norm_text(nm) in existing:
                    return  # 중복이면 스킵
                child = QtWidgets.QTreeWidgetItem(root)
                child.setText(0, nm)
                child.setFlags(child.flags() | QtCore.Qt.ItemIsEditable)
                child.setData(0, QtCore.Qt.UserRole, p_data)
                self._hydrate_product_item_from_disk(child)  # ✅ 저장된 상품이면 즉시 체크표시 반영

                existing.add(self._norm_text(nm))

            if isinstance(products, list):
                for p_val in products:
                    p_name = ""
                    p_data = {"type": "product"}

                    if isinstance(p_val, str):
                        p_name = p_val
                    elif isinstance(p_val, dict):
                        p_name = p_val.get("name") or p_val.get("title") or "상품명 없음"
                        p_data.update(p_val)

                    _add_child_product(p_name, p_data)

            elif isinstance(products, str) and products.strip():
                for p in products.split(","):
                    _add_child_product(p.strip(), {"type": "product"})

            root.setExpanded(True)

        # ✅ 폴더/상품 오름차순 정렬
        self._sort_tree()

    def _sort_selected_tree(self) -> None:
        """
        선택한 리스트 트리 정렬:
        - 최상위(폴더/이슈) 오름차순
        - 각 폴더 하위(상품) 오름차순
        - 텍스트 앞의 아이콘/체크표시(📂, ✅ 등)는 정렬 키에서 제외
        """
        def _norm_key(text: str) -> str:
            t = (text or "").strip()

            # 앞쪽 아이콘/기호/체크문자 제거 (📂, ✅, ■ 등)
            # '내 상품', '새 상품' 같은 일반 텍스트는 보존
            t = re.sub(r"^[^\w가-힣]+", "", t).strip()

            # '📂 '처럼 아이콘 + 공백 케이스 한 번 더 방어
            t = t.lstrip("📂✅☑✔■●▶▷•·- ").strip()

            return t.casefold()

        tree = self.tree_selected
        if tree is None:
            return

        # 1) 최상위 아이템 정렬
        top_items = []
        for i in range(tree.topLevelItemCount()):
            it = tree.topLevelItem(i)
            if it is not None:
                top_items.append(it)

        # takeTopLevelItem으로 모두 분리
        for _ in range(tree.topLevelItemCount() - 1, -1, -1):
            tree.takeTopLevelItem(_)

        # 정렬 후 재삽입
        top_items.sort(key=lambda it: _norm_key(it.text(0)))
        for it in top_items:
            tree.addTopLevelItem(it)

            # 2) 각 최상위 아이템의 자식 정렬
            child_items = []
            for c in range(it.childCount()):
                ch = it.child(c)
                if ch is not None:
                    child_items.append(ch)

            # 기존 자식 제거 후 정렬 재삽입
            for _c in range(it.childCount() - 1, -1, -1):
                it.takeChild(_c)

            child_items.sort(key=lambda ch: _norm_key(ch.text(0)))
            for ch in child_items:
                it.addChild(ch)


# [shopping.py] ShoppingWidget 클래스 정의 바로 위에 추가

class LoadProductDialog(QtWidgets.QDialog):
    """
    [New] 기존 탐색기 대신, 상품 폴더만 리스트로 보여주고
    '더블 클릭'으로 바로 추가할 수 있게 만든 커스텀 창
    """

    def __init__(self, products_dir: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("상품 불러오기")
        self.resize(400, 500)
        self.products_dir = products_dir
        self._init_ui()
        self._load_list()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # 안내
        lbl = QtWidgets.QLabel(f"<b>폴더 위치:</b> {self.products_dir}<br>불러올 상품을 더블 클릭하세요.")
        lbl.setTextFormat(QtCore.Qt.RichText)
        layout.addWidget(lbl)

        # 리스트 위젯
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        # [핵심] 아이템 더블 클릭 시 -> 바로 accept(확인) 실행
        self.list_widget.itemDoubleClicked.connect(self.accept)

        layout.addWidget(self.list_widget)

        # 버튼
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("불러오기")
        self.btn_load.clicked.connect(self.accept)

        self.btn_cancel = QtWidgets.QPushButton("취소")
        self.btn_cancel.clicked.connect(self.reject)

        btn_layout.addStretch(1)
        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_cancel)

        layout.addLayout(btn_layout)

    def _load_list(self):
        """products 폴더 안의 하위 폴더만 깔끔하게 보여줌"""
        self.list_widget.clear()
        if not self.products_dir.exists():
            return

        try:
            subdirs = [p for p in self.products_dir.iterdir() if p.is_dir()]
            subdirs.sort(key=lambda x: x.name)  # 이름순 정렬

            for p in subdirs:
                item = QtWidgets.QListWidgetItem(p.name)
                item.setData(QtCore.Qt.UserRole, str(p))
                self.list_widget.addItem(item)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "오류", f"목록 로드 실패: {e}")

    def get_selected_paths(self) -> List[Path]:
        selected_items = self.list_widget.selectedItems()
        paths = []
        for item in selected_items:
            path_str = item.data(QtCore.Qt.UserRole)
            if path_str:
                paths.append(Path(path_str))
        return paths

# ────────────────────────────────────────
# 팝업 다이얼로그 (수정됨)
# ────────────────────────────────────────
class IssueListViewerDialog(QtWidgets.QDialog):
    def __init__(self, items, parent=None, title="리스트 보기", original_path=None, pre_selected_items=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1100, 650)
        self._items = items or []
        self._pre_selected = pre_selected_items or []  # 기존 선택 내역
        self._original_path = original_path  # 저장 경로 생성을 위해 보관

        root = QtWidgets.QVBoxLayout(self)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        root.addWidget(splitter, 1)

        # 1. LEFT: Title List
        left_wrap = QtWidgets.QWidget(self)
        left_lay = QtWidgets.QVBoxLayout(left_wrap)
        left_lay.setContentsMargins(8, 8, 8, 8)
        left_lay.addWidget(QtWidgets.QLabel("1. 제목(Title) - 더블클릭 시 선택", left_wrap))

        self.list_titles = QtWidgets.QListWidget(left_wrap)
        self.list_titles.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        left_lay.addWidget(self.list_titles, 1)
        splitter.addWidget(left_wrap)

        # 2. MIDDLE: Reason + Products
        mid_wrap = QtWidgets.QWidget(self)
        mid_lay = QtWidgets.QVBoxLayout(mid_wrap)
        mid_lay.setContentsMargins(8, 8, 8, 8)

        mid_lay.addWidget(QtWidgets.QLabel("사유(Reason)", mid_wrap))
        self.te_reason = QtWidgets.QTextEdit(mid_wrap)
        self.te_reason.setReadOnly(True)
        self.te_reason.setMinimumHeight(100)
        mid_lay.addWidget(self.te_reason, 1)

        mid_lay.addWidget(QtWidgets.QLabel("상품(Products) - 참고용", mid_wrap))
        self.list_products = QtWidgets.QListWidget(mid_wrap)
        mid_lay.addWidget(self.list_products, 2)
        splitter.addWidget(mid_wrap)

        # 3. RIGHT: My Selection
        right_wrap = QtWidgets.QWidget(self)
        right_lay = QtWidgets.QVBoxLayout(right_wrap)
        right_lay.setContentsMargins(8, 8, 8, 8)

        lbl_info = QtWidgets.QLabel("3. 나의 선택(My Selection)", right_wrap)
        lbl_info.setStyleSheet("color: blue; font-weight: bold;")
        right_lay.addWidget(lbl_info)

        self.list_my_selection = QtWidgets.QListWidget(right_wrap)
        self.list_my_selection.setToolTip("더블클릭하면 목록에서 제거됩니다.")
        right_lay.addWidget(self.list_my_selection, 1)
        splitter.addWidget(right_wrap)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 3)

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)

        # 버튼 기능 변경: "저장 후 추가"
        self.btn_add_to_main = QtWidgets.QPushButton("선택한 리스트 저장 및 추가", self)
        self.btn_close = QtWidgets.QPushButton("닫기", self)

        btns.addWidget(self.btn_add_to_main)
        btns.addWidget(self.btn_close)
        root.addLayout(btns)

        self.btn_close.clicked.connect(self.close)
        # accept 대신 자체 저장 함수 연결
        self.btn_add_to_main.clicked.connect(self._save_and_accept)

        self.list_titles.currentItemChanged.connect(self._on_item_changed)
        self.list_titles.itemDoubleClicked.connect(self._add_title_to_selection)
        self.list_my_selection.itemDoubleClicked.connect(self._remove_from_selection)

        self._populate()

        # 🔥 [추가됨] 기존 선택 내역이 있다면 우측 리스트에 채워넣기
        if self._pre_selected:
            self._populate_selection()

        if self.list_titles.count() > 0:
            self.list_titles.setCurrentRow(0)

    def _populate(self):
        self.list_titles.clear()
        for it in self._items:
            if not isinstance(it, dict): continue
            title = (it.get("title") or "").strip()
            if not title: continue

            # 데이터 로드 시 구조가 평탄화된 경우와 계층형인 경우 모두 대응
            extra = it.get("extra")
            if not isinstance(extra, dict):
                # 호환성: extra가 없으면 top-level에서 찾기 시도
                extra = {
                    "reason": it.get("reason", ""),
                    "products": it.get("products", [])
                }

            data = {
                "title": title,
                "extra": extra,  # 원본 구조 유지
                "source": it.get("source", "")
            }

            item = QtWidgets.QListWidgetItem(f"{title}")
            item.setData(QtCore.Qt.UserRole, data)
            self.list_titles.addItem(item)

    def _populate_selection(self):
        """기존 저장된 선택 내역(_selected.json)을 우측 리스트에 복구"""
        for it in self._pre_selected:
            if not isinstance(it, dict): continue
            title = it.get("title")
            if not title: continue

            # 저장된 데이터 구조 그대로 UserRole에 심기
            # 이미 {"title":..., "extra":...} 구조로 저장되어 있다고 가정
            item = QtWidgets.QListWidgetItem(title)
            item.setData(QtCore.Qt.UserRole, it)
            self.list_my_selection.addItem(item)

    def _on_item_changed(self, current, previous):
        self.te_reason.clear()
        self.list_products.clear()
        if not current: return

        data = current.data(QtCore.Qt.UserRole) or {}
        extra = data.get("extra", {})

        reason = (extra.get("reason") or "").strip()
        self.te_reason.setPlainText(reason if reason else "사유 정보 없음")

        products = extra.get("products")
        if isinstance(products, list):
            for p in products:
                # p가 dict일 수도 string일 수도 있음
                if isinstance(p, dict):
                    self.list_products.addItem(str(p.get("name") or p.get("title") or str(p)))
                else:
                    self.list_products.addItem(str(p))
        elif isinstance(products, str) and products.strip():
            for p in products.split(","):
                self.list_products.addItem(p.strip())

    def _add_title_to_selection(self, item):
        data = item.data(QtCore.Qt.UserRole) or {}
        title = data.get("title")
        if not title: return

        for i in range(self.list_my_selection.count()):
            if self.list_my_selection.item(i).text() == title:
                return

        new_item = QtWidgets.QListWidgetItem(title)
        new_item.setData(QtCore.Qt.UserRole, data)
        self.list_my_selection.addItem(new_item)
        self.list_my_selection.scrollToBottom()

    def _remove_from_selection(self, item):
        row = self.list_my_selection.row(item)
        self.list_my_selection.takeItem(row)

    def get_selected_data_full(self) -> list[dict]:
        result = []
        for i in range(self.list_my_selection.count()):
            item = self.list_my_selection.item(i)
            data = item.data(QtCore.Qt.UserRole)
            if data:
                # 저장 형식: [{"title":..., "extra":{...}}, ...] 형태로 반환
                # _populate에서 이미 이 구조로 data를 만들어두었음
                result.append(data)
        return result

    def _save_and_accept(self):
        """선택한 항목을 _selected.json 파일로 저장 후 창 닫기"""
        selected_data = self.get_selected_data_full()
        if not selected_data:
            QtWidgets.QMessageBox.warning(self, "경고", "선택된 항목이 없습니다.")
            return

        # 저장 경로 생성
        if self._original_path:
            p = Path(self._original_path)
            # ex: 2024-12-25_123456_b.json -> 2024-12-25_123456_b_selected.json
            stem = p.stem
            # 이미 _selected가 붙어있으면 또 붙이지 않음 (불러오기 후 다시 저장 시)
            if stem.endswith("_selected"):
                new_name = p.name
            else:
                new_name = f"{stem}_selected.json"

            save_path = p.parent / new_name

            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(selected_data, f, ensure_ascii=False, indent=2)
                print(f"[Shopping] Selection saved to: {save_path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "저장 실패", f"파일 저장 중 오류 발생:\n{e}")
                return

        self.accept()


def create_shopping_widget(parent=None) -> QtWidgets.QWidget:
    return ShoppingWidget(parent)