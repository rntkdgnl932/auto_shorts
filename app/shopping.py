# -*- coding: utf-8 -*-
from __future__ import annotations

import shutil
import datetime

import json
import os
import re
from pathlib import Path
from PyQt5 import QtWidgets, QtCore, QtGui

from app.utils import run_job_with_progress_async, sanitize_title

from app.issue_list_builder import (
    save_issue_list_for_shopping_all,
    save_issue_list_for_shopping_ai_b_from_a,
)

from app.utils import sanitize_title


from app.shopping_video_build import (
    ShoppingVideoJsonBuilder,
    ShoppingImageGenerator,
    ShoppingMovieGenerator,
    ShoppingShortsPipeline,
    BuildOptions,
)

from app.shopping_video_build import ShoppingVideoJsonBuilder, VideoShoppingBuildInput


class VideoBuildDialog(QtWidgets.QDialog):
    """
    '영상제작' 버튼 클릭 시 열리는 창.
    - 탭1: video_shopping.json 생성 및 확인
    - 탭2: 이미지/영상 생성 및 합치기 (개별 테스트 버튼 + 전체 실행)
    """

    def __init__(self, product_dir: str, product_data: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"영상 제작 - {product_data.get('product_name', '')}")
        self.resize(1000, 750)

        self.product_dir = str(product_dir)
        self.product_data = product_data or {}
        self.video_json_path = str(Path(self.product_dir) / "video_shopping.json")

        root = QtWidgets.QVBoxLayout(self)

        # 탭 위젯
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

        # JSON 관련 버튼
        btn_row = QtWidgets.QHBoxLayout()
        t1.addLayout(btn_row)

        self.btn_build_json = QtWidgets.QPushButton("1. video_shopping.json 생성/갱신", tab1)
        self.btn_build_json.setStyleSheet("color: blue; font-weight: bold;")

        self.btn_open_product_dir = QtWidgets.QPushButton("상품 폴더 열기", tab1)
        self.btn_open_video_json = QtWidgets.QPushButton("JSON 파일 열기", tab1)

        btn_row.addWidget(self.btn_build_json)
        btn_row.addWidget(self.btn_open_product_dir)
        btn_row.addWidget(self.btn_open_video_json)
        btn_row.addStretch(1)

        self.te_preview = QtWidgets.QPlainTextEdit(tab1)
        self.te_preview.setReadOnly(True)
        self.te_preview.setPlaceholderText("여기에 생성된 JSON 내용이 표시됩니다.")
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
        chk_row.addSpacing(15)
        chk_row.addWidget(QtWidgets.QLabel("FPS:", tab2))
        self.sp_fps = QtWidgets.QSpinBox(tab2)
        self.sp_fps.setRange(8, 60)
        self.sp_fps.setValue(24)
        chk_row.addWidget(self.sp_fps)
        chk_row.addStretch(1)
        t2.addLayout(chk_row)

        t2.addSpacing(20)

        # ★ [개별 단계 실행 그룹] ★
        grp_actions = QtWidgets.QGroupBox("개별 단계 실행 (테스트용)", tab2)
        lay_actions = QtWidgets.QHBoxLayout(grp_actions)
        lay_actions.setSpacing(10)

        self.btn_gen_images = QtWidgets.QPushButton("2. 누락이미지 생성 (Z-Image)", tab2)
        self.btn_gen_images.setMinimumHeight(40)
        self.btn_gen_images.setToolTip("video_shopping.json을 읽어 imgs 폴더에 없는 이미지를 생성합니다.")

        self.btn_gen_movies = QtWidgets.QPushButton("3. 영상 생성 (I2V)", tab2)
        self.btn_gen_movies.setMinimumHeight(40)
        self.btn_gen_movies.setToolTip("imgs의 이미지를 이용해 clips 폴더에 영상을 생성합니다.")

        self.btn_merge = QtWidgets.QPushButton("4. 영상 합치기 (Merge)", tab2)
        self.btn_merge.setMinimumHeight(40)
        self.btn_merge.setToolTip("clips 폴더의 영상을 모아 하나로 합칩니다.")

        lay_actions.addWidget(self.btn_gen_images)
        lay_actions.addWidget(self.btn_gen_movies)
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
        self.btn_open_product_dir.clicked.connect(lambda: self._open_path(self.product_dir))
        self.btn_open_imgs.clicked.connect(lambda: self._open_path(str(Path(self.product_dir) / "imgs")))
        self.btn_open_clips.clicked.connect(lambda: self._open_path(str(Path(self.product_dir) / "clips")))
        self.btn_open_video_json.clicked.connect(lambda: self._open_path(self.video_json_path))

        # 기능 버튼 (비동기 처리 함수 연결)
        self.btn_build_json.clicked.connect(self.on_build_json_clicked)
        self.btn_gen_images.clicked.connect(self.on_gen_images_clicked)
        self.btn_gen_movies.clicked.connect(self.on_gen_movies_clicked)
        self.btn_merge.clicked.connect(self.on_merge_clicked)
        self.btn_run_all.clicked.connect(self.on_run_all_clicked)

        # 초기 로드
        self._refresh_preview_if_exists()

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
        return BuildOptions(
            scene_count=int(self.sp_scene_count.value()),
            style=str(self.cb_style.currentText()),
            hook_level=int(self.sp_hook.value()),
            fps=int(self.sp_fps.value()),
        )

    def _refresh_preview_if_exists(self):
        p = Path(self.video_json_path)
        if p.exists():
            try:
                txt = p.read_text(encoding="utf-8")
                self.te_preview.setPlainText(txt)
            except Exception as e:
                self.te_preview.setPlainText(f"읽기 실패: {e}")

    # ── 비동기 작업 핸들러 (run_job_with_progress_async 사용) ─────────────────────────

    def on_build_json_clicked(self):
        """1단계: JSON 생성"""

        def job(progress):
            progress("[JSON] 생성 시작...")
            builder = ShoppingVideoJsonBuilder(on_progress=progress)
            out_path = builder.build(
                product_dir=self.product_dir,
                product_data=self.product_data,
                options=self._options(),
            )
            progress(f"[JSON] 완료: {out_path}")
            return str(out_path)

        def done(ok, res, err):
            if ok:
                self._append_log(f"✅ JSON 생성 성공")
                self._refresh_preview_if_exists()
            else:
                self._append_log(f"❌ JSON 생성 실패: {err}")

        run_job_with_progress_async(self, "JSON 생성", job, on_done=done)

    def on_gen_images_clicked(self):
        """2단계: 누락 이미지 생성 (비동기)"""
        if not Path(self.video_json_path).exists():
            self._append_log("⚠ video_shopping.json이 없습니다. 1단계를 먼저 실행하세요.")
            return

        def job(progress):
            progress("[Image] 이미지 생성 시작...")
            gen = ShoppingImageGenerator(on_progress=progress)
            gen.generate_images(self.video_json_path, skip_if_exists=self.cb_img_skip.isChecked())
            return "OK"

        def done(ok, res, err):
            if ok:
                self._append_log("✅ 이미지 생성 완료")
            else:
                self._append_log(f"❌ 이미지 생성 실패: {err}")

        run_job_with_progress_async(self, "이미지 생성", job, on_done=done)

    def on_gen_movies_clicked(self):
        """3단계: 영상 생성 (비동기)"""
        if not Path(self.video_json_path).exists():
            self._append_log("⚠ video_shopping.json이 없습니다.")
            return

        def job(progress):
            progress("[Movie] I2V 영상 생성 시작...")
            gen = ShoppingMovieGenerator(on_progress=progress)
            gen.generate_movies(
                self.video_json_path,
                skip_if_exists=self.cb_mov_skip.isChecked(),
                fps=int(self.sp_fps.value())
            )
            return "OK"

        def done(ok, res, err):
            if ok:
                self._append_log("✅ 영상 생성 완료")
            else:
                self._append_log(f"❌ 영상 생성 실패: {err}")

        run_job_with_progress_async(self, "영상 생성", job, on_done=done)

    def on_merge_clicked(self):
        """4단계: 영상 합치기 (비동기)"""
        if not Path(self.video_json_path).exists():
            self._append_log("⚠ video_shopping.json이 없습니다.")
            return

        def job(progress):
            progress("[Merge] 영상 합치기 시작...")
            gen = ShoppingMovieGenerator(on_progress=progress)
            gen.merge_movies(self.video_json_path)
            return "OK"

        def done(ok, res, err):
            if ok:
                self._append_log("✅ 영상 합치기 완료")
            else:
                self._append_log(f"❌ 영상 합치기 실패: {err}")

        run_job_with_progress_async(self, "영상 합치기", job, on_done=done)

    def on_run_all_clicked(self):
        """전체 파이프라인 일괄 실행 (비동기)"""

        def job(progress):
            progress("[All] 전체 파이프라인 시작...")
            pipe = ShoppingShortsPipeline(on_progress=progress)
            pipe.run_all(
                product_dir=self.product_dir,
                product_data=self.product_data,
                options=self._options(),
                build_json=True,
                build_images=True,
                build_movies=True,
                merge=True,
                skip_if_exists=True,
            )
            return "OK"

        def done(ok, res, err):
            if ok:
                self._append_log("✅ 전체 실행 완료!")
                self._refresh_preview_if_exists()
            else:
                self._append_log(f"❌ 전체 실행 실패: {err}")

        run_job_with_progress_async(self, "전체 실행", job, on_done=done)


class ShoppingWidget(QtWidgets.QWidget):
    """
    쇼핑/쿠팡/쇼츠 자동화용 메인 UI.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
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
        """
        등록된 상품 불러오기:
        - 등록상품을 스캔해서 선택(멀티)
        - 좌측 트리에 "📦 등록상품" 루트 아래에 추가
        """
        products = self._scan_registered_products()
        if not products:
            QtWidgets.QMessageBox.information(self, "등록상품 없음", "등록된 상품(product.json)을 찾지 못했습니다.")
            return

        # 선택 다이얼로그
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("등록된 상품 불러오기")
        dlg.resize(650, 500)
        lay = QtWidgets.QVBoxLayout(dlg)

        info = QtWidgets.QLabel("불러올 상품을 선택하세요. (Ctrl/Shift 멀티 선택 가능)")
        lay.addWidget(info)

        lw = QtWidgets.QListWidget(dlg)
        lw.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        lay.addWidget(lw, 1)

        for p in products:
            item = QtWidgets.QListWidgetItem(p["product_name"])
            item.setData(QtCore.Qt.UserRole, p)
            lw.addItem(item)

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        btn_ok = QtWidgets.QPushButton("추가", dlg)
        btn_cancel = QtWidgets.QPushButton("취소", dlg)
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        lay.addLayout(btns)

        btn_cancel.clicked.connect(dlg.reject)
        btn_ok.clicked.connect(dlg.accept)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        selected_items = lw.selectedItems()
        if not selected_items:
            return

        # "📦 등록상품" 루트 찾거나 생성
        root = None
        for i in range(self.tree_selected.topLevelItemCount()):
            it = self.tree_selected.topLevelItem(i)
            if (it.text(0) or "").strip() == "📦 등록상품":
                root = it
                break
        if root is None:
            root = QtWidgets.QTreeWidgetItem(self.tree_selected)
            root.setText(0, "📦 등록상품")
            root.setFlags(root.flags() | QtCore.Qt.ItemIsEditable)
            root.setData(0, QtCore.Qt.UserRole, {"type": "issue", "reason": "확정으로 저장된 등록상품"})
            root.setExpanded(True)

        added = 0
        existing_names = set()
        for i in range(root.childCount()):
            existing_names.add(self._get_product_name_from_item(root.child(i)))

        for it in selected_items:
            p = it.data(QtCore.Qt.UserRole) or {}
            pname = (p.get("product_name") or "").strip()
            if not pname:
                continue
            if pname in existing_names:
                continue

            child = QtWidgets.QTreeWidgetItem(root)
            child.setText(0, f"✅ {pname}")
            child.setFlags(child.flags() | QtCore.Qt.ItemIsEditable)

            data = p.get("data") or {}
            if not isinstance(data, dict):
                data = {"type": "product"}
            data.setdefault("type", "product")
            data["product_name"] = pname
            data["product_dir"] = p.get("product_dir", "")
            data["source"] = data.get("source") or "registered"
            child.setData(0, QtCore.Qt.UserRole, data)

            # 디스크 내용으로 최종 hydrate
            self._hydrate_product_item_from_disk(child)

            existing_names.add(pname)
            added += 1

        root.setExpanded(True)
        self.append_log(f"📥 등록상품 {added}개를 좌측 트리에 추가했습니다.")

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