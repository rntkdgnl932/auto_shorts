# -*- coding: utf-8 -*-
from __future__ import annotations

from PyQt5 import QtWidgets, QtCore, QtGui
from app.utils import run_job_with_progress_async

from app.issue_list_builder import (
    save_issue_list_for_shopping_all,
    save_issue_list_for_shopping_ai_b_from_a,
)


class ShoppingWidget(QtWidgets.QWidget):
    """
    쇼핑/쿠팡/쇼츠 자동화용 메인 UI.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
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

        # 2-1) [수정됨] 좌측: 선택한 리스트 (TreeWidget)
        left_widget = QtWidgets.QWidget(center_splitter)
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        lbl_list = QtWidgets.QLabel("선택한 리스트 (Del:삭제 / F2:수정)", left_widget)

        # ✅ 메인 화면은 트리 구조 (제목 └ 상품)
        self.tree_selected = QtWidgets.QTreeWidget(left_widget)
        self.tree_selected.setHeaderLabels(["상품명 / 제목", "가격", "할인", "평점"])
        self.tree_selected.setColumnWidth(0, 250)
        self.tree_selected.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tree_selected.setAlternatingRowColors(True)
        # 수정 가능
        self.tree_selected.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked | QtWidgets.QAbstractItemView.EditKeyPressed)

        # 컨텍스트 메뉴 (우클릭)
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
        self.lbl_thumbnail.setFixedSize(180, 180)
        self.lbl_thumbnail.setFrameShape(QtWidgets.QFrame.Box)
        self.lbl_thumbnail.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_thumbnail.setText("이미지 없음")

        info_form = QtWidgets.QFormLayout()
        info_form.setLabelAlignment(QtCore.Qt.AlignRight)
        info_form.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        self.le_title = QtWidgets.QLineEdit(thumb_group)
        self.le_title.setReadOnly(True)  # 기본 정보는 읽기 전용 (트리에서 수정 권장)

        self.le_price = QtWidgets.QLineEdit(thumb_group)
        self.le_price.setReadOnly(True)

        self.le_discount = QtWidgets.QLineEdit(thumb_group)
        self.le_discount.setReadOnly(True)

        self.le_rating = QtWidgets.QLineEdit(thumb_group)
        self.le_rating.setReadOnly(True)

        info_form.addRow("상품명:", self.le_title)
        info_form.addRow("가격:", self.le_price)
        info_form.addRow("할인:", self.le_discount)
        info_form.addRow("평점:", self.le_rating)

        thumb_layout.addWidget(self.lbl_thumbnail)
        thumb_layout.addLayout(info_form)

        # 링크 및 설명
        link_group = QtWidgets.QGroupBox("링크 / 설명", right_widget)
        link_layout = QtWidgets.QFormLayout(link_group)
        link_layout.setLabelAlignment(QtCore.Qt.AlignRight)

        self.le_product_url = QtWidgets.QLineEdit(link_group)
        self.le_affiliate_url = QtWidgets.QLineEdit(link_group)
        self.le_affiliate_url.setPlaceholderText("쿠팡 파트너스 딥링크 (자동 생성 예정)")

        self.te_description = QtWidgets.QPlainTextEdit(link_group)
        self.te_description.setPlaceholderText("상품 소개/쇼츠 시나리오용 요약 문구 (AI가 자동 생성 예정)...")

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
        right_layout.addLayout(btn_row)

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

        # [수정] 트리 선택 변경 시
        self.tree_selected.currentItemChanged.connect(self.on_tree_item_selected)

        self.btn_test.clicked.connect(self.on_test_clicked)
        self.btn_load_list.clicked.connect(
            lambda: self._run_stage1_collect_issue_list("1단계버튼")
        )
        self.btn_make_script.clicked.connect(self.on_make_script_clicked)
        self.btn_make_infok.clicked.connect(self.on_make_infok_clicked)
        self.btn_load_b.clicked.connect(self.on_load_b_clicked)

    # ────────────────────────────────────────
    # 공통 로그 함수
    # ────────────────────────────────────────
    def append_log(self, msg: str):
        self.log.appendPlainText(msg)

    # ────────────────────────────────────────
    # 트리 위젯 관리 (수정/삭제/메뉴)
    # ────────────────────────────────────────
    def keyPressEvent(self, event):
        """Del 키 누르면 삭제 기능 구현"""
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
                # 선택된게 있으면 그 아래(혹은 부모 아래)에 추가
                parent = target if not target.parent() else target.parent()
                self._add_dummy_product(parent)
            else:
                # 선택 없으면 루트 추가? (여기서는 스킵)
                pass

    def _delete_selected_tree_item(self):
        items = self.tree_selected.selectedItems()
        if not items: return
        for item in items:
            parent = item.parent()
            if parent:
                parent.removeChild(item)
            else:
                idx = self.tree_selected.indexOfTopLevelItem(item)
                self.tree_selected.takeTopLevelItem(idx)

    def _add_dummy_product(self, parent_item):
        """임의 상품 추가 (우클릭 메뉴용)"""
        child = QtWidgets.QTreeWidgetItem(parent_item)
        child.setText(0, "새 상품")
        child.setFlags(child.flags() | QtCore.Qt.ItemIsEditable)
        parent_item.setExpanded(True)

    def on_tree_item_selected(self, current, previous):
        """트리 아이템 선택 시 우측 상세창 채우기"""
        if not current: return

        # 부모가 있으면 '상품', 없으면 '제목(카테고리)'
        parent = current.parent()

        self.le_title.setText(current.text(0))
        self.le_price.setText(current.text(1))
        self.le_discount.setText(current.text(2))
        self.le_rating.setText(current.text(3))

        if parent:
            self.lbl_thumbnail.setText("상품")
            # TODO: 상품 데이터(URL 등)가 있으면 가져와서 채우기
        else:
            self.lbl_thumbnail.setText("카테고리")
            self.le_product_url.clear()
            self.le_affiliate_url.clear()

    # ────────────────────────────────────────
    # 슬롯들
    # ────────────────────────────────────────
    def on_search_keyword(self):
        keyword = self.le_keyword.text().strip()
        category = self.combo_category.currentText().strip()
        if not keyword:
            self.append_log("⚠ 키워드를 입력해 주세요.")
            return
        self.append_log(f"🔍 키워드 검색 준비중... (키워드: {keyword}, 카테고리: {category})")
        # TODO: 실제 쿠팡 상품 목록 가져오는 함수 연결
        # 트리 구조에서는 더미 데이터 채우는 방식이 다르므로 일단 로그만
        self.append_log("ℹ 트리 구조에서는 검색 결과 연동 로직 수정 필요")

    def on_search_popular(self):
        category = self.combo_category.currentText().strip()
        self.append_log(f"🔥 인기템 리스트 불러오기 준비중... (카테고리: {category})")

    def on_test_clicked(self):
        self.append_log("🧪 [테스트] 버튼 클릭.")

    def _run_stage1_collect_issue_list(self, trigger_label: str):
        self.append_log(f"🧩 [1단계] ({trigger_label}) 쇼핑 이슈 + AI 상품 후보 전체 파이프라인 시작...")

        def job(progress):
            progress({"msg": f"[1단계/{trigger_label}] 쇼핑 이슈 수집 중... (_a.json)"})
            path_a = save_issue_list_for_shopping_all(on_progress=progress)

            progress({"msg": f"[1단계/{trigger_label}] AI로 상품 후보 분석 중... (_a -> _b)"})
            path_b = save_issue_list_for_shopping_ai_b_from_a(
                on_progress=progress,
                a_path=str(path_a),
            )
            return {"a_path": str(path_a) if path_a else "", "b_path": str(path_b) if path_b else ""}

        def done(ok, payload, err):
            if (not ok) or (err is not None):
                self.append_log(f"❌ [1단계/{trigger_label}] 파이프라인 실패: {err}")
                return

            b_path = payload.get("b_path", "") if isinstance(payload, dict) else ""
            if not b_path:
                self.append_log(f"⚠ [1단계/{trigger_label}] _b 경로 없음.")
                return

            self.append_log(f"✅ [1단계/{trigger_label}] 완료. 리스트 팝업을 엽니다.")
            self._open_list_dialog(b_path)

        run_job_with_progress_async(
            owner=self,
            title=f"쇼핑 이슈+AI 후보 전체 파이프라인 (1단계/{trigger_label})",
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
        aff_link = self.le_affiliate_url.text().strip()
        if not title:
            self.append_log("⚠ 상품을 먼저 선택해 주세요.")
            return
        self.append_log(f"🧱 인포크링크 생성 시도: {title} / {aff_link}")

    def on_load_b_clicked(self):
        import json
        from pathlib import Path
        default_dir = str(Path(r"C:\my_games\shorts_make\issue_list"))

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "리스트 불러오기 (_b.json)", default_dir, "JSON Files (*.json);;All Files (*.*)"
        )
        if path:
            self._open_list_dialog(path)
            return

        # 파일 선택 안 하면 자동 로드 시도
        root = Path(default_dir)
        latest = None
        if root.exists():
            try:
                date_dirs = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
                for d in date_dirs:
                    cand = sorted(d.glob("*_b.json"), key=lambda x: x.name, reverse=True)
                    if cand:
                        latest = cand[0]
                        break
            except Exception:
                pass

        if latest:
            self._open_list_dialog(str(latest))
        else:
            QtWidgets.QMessageBox.information(self, "안내", "불러올 파일이 없습니다.")

    def _open_list_dialog(self, path: str):
        import json
        from pathlib import Path
        try:
            txt = Path(path).read_text(encoding="utf-8")
            items = json.loads(txt)
            if not isinstance(items, list):
                raise ValueError("JSON 구조가 리스트가 아닙니다.")

            dlg = IssueListViewerDialog(items, parent=self, title=f"리스트 보기: {Path(path).name}")
            if dlg.exec_() == QtWidgets.QDialog.Accepted:
                # [수정됨] 팝업에서 선택한 데이터를 가져와서 메인 트리에 추가
                selected_data = dlg.get_selected_data_full()
                if selected_data:
                    self._add_data_to_main_tree(selected_data)
                    self.append_log(f"📥 {len(selected_data)}개의 카테고리 세트가 추가되었습니다.")
        except Exception as e:
            self.append_log(f"⚠ 리스트 열기 실패: {e}")

    def _add_data_to_main_tree(self, data_list: list[dict]):
        """
        팝업에서 가져온 [{title:..., products:[]}, ...] 데이터를
        메인 화면의 TreeWidget에 추가.
        """
        for item in data_list:
            title = item.get("title", "제목 없음")
            products = item.get("products", [])

            # 1. 루트(제목) 생성
            root = QtWidgets.QTreeWidgetItem(self.tree_selected)
            root.setText(0, f"📂 {title}")
            root.setFlags(root.flags() | QtCore.Qt.ItemIsEditable)

            # 2. 자식(상품) 생성
            if isinstance(products, list):
                for p_name in products:
                    child = QtWidgets.QTreeWidgetItem(root)
                    child.setText(0, str(p_name))
                    child.setFlags(child.flags() | QtCore.Qt.ItemIsEditable)
            elif isinstance(products, str) and products.strip():
                # 콤마 구분 스트링일 경우
                for p in products.split(","):
                    if not p.strip(): continue
                    child = QtWidgets.QTreeWidgetItem(root)
                    child.setText(0, p.strip())
                    child.setFlags(child.flags() | QtCore.Qt.ItemIsEditable)

            # 펼치기
            root.setExpanded(True)


# all_ui.py에서 사용하기 위한 팩토리 함수
def create_shopping_widget(parent=None) -> QtWidgets.QWidget:
    return ShoppingWidget(parent)


class IssueListViewerDialog(QtWidgets.QDialog):
    def __init__(self, items, parent=None, title="리스트 보기"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1100, 650)

        self._items = items or []

        root = QtWidgets.QVBoxLayout(self)

        # ✅ 3분할 스플리터 (좌: 제목 | 중: 사유+상품 | 우: 나의 선택)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        root.addWidget(splitter, 1)

        # ---------------------------
        # 1. LEFT: Title List (더블클릭 -> 우측에 추가)
        # ---------------------------
        left_wrap = QtWidgets.QWidget(self)
        left_lay = QtWidgets.QVBoxLayout(left_wrap)
        left_lay.setContentsMargins(8, 8, 8, 8)
        left_lay.addWidget(QtWidgets.QLabel("1. 제목(Title) - 더블클릭 시 선택", left_wrap))

        self.list_titles = QtWidgets.QListWidget(left_wrap)
        self.list_titles.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        left_lay.addWidget(self.list_titles, 1)

        splitter.addWidget(left_wrap)

        # ---------------------------
        # 2. MIDDLE: Reason + Products
        # ---------------------------
        mid_wrap = QtWidgets.QWidget(self)
        mid_lay = QtWidgets.QVBoxLayout(mid_wrap)
        mid_lay.setContentsMargins(8, 8, 8, 8)

        # Reason
        mid_lay.addWidget(QtWidgets.QLabel("사유(Reason)", mid_wrap))
        self.te_reason = QtWidgets.QTextEdit(mid_wrap)
        self.te_reason.setReadOnly(True)
        self.te_reason.setMinimumHeight(100)
        mid_lay.addWidget(self.te_reason, 1)

        # Products
        mid_lay.addWidget(QtWidgets.QLabel("상품(Products) - 참고용", mid_wrap))
        self.list_products = QtWidgets.QListWidget(mid_wrap)
        self.list_products.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        mid_lay.addWidget(self.list_products, 2)

        splitter.addWidget(mid_wrap)

        # ---------------------------
        # 3. RIGHT: My Selection (나의 선택)
        # ---------------------------
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

        # 비율 조절
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 3)

        # 하단 버튼
        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)

        self.btn_add_to_main = QtWidgets.QPushButton("선택한 리스트에 추가", self)
        self.btn_close = QtWidgets.QPushButton("닫기", self)

        btns.addWidget(self.btn_add_to_main)
        btns.addWidget(self.btn_close)
        root.addLayout(btns)

        # ────────────────────────────────────────
        # 이벤트 연결
        # ────────────────────────────────────────
        self.btn_close.clicked.connect(self.close)
        self.btn_add_to_main.clicked.connect(self.accept)

        # 왼쪽 목록 선택 시 -> 중앙 정보(사유, 상품) 갱신
        self.list_titles.currentItemChanged.connect(self._on_item_changed)

        # 1. 왼쪽(제목) 더블클릭 -> 우측(나의 선택)에 추가
        self.list_titles.itemDoubleClicked.connect(self._add_title_to_selection)

        # 2. 우측(나의 선택) 더블클릭 -> 목록에서 제거
        self.list_my_selection.itemDoubleClicked.connect(self._remove_from_selection)

        # 초기 데이터 로드
        self._populate()
        if self.list_titles.count() > 0:
            self.list_titles.setCurrentRow(0)

    # ────────────────────────────────────────
    # 로직 구현
    # ────────────────────────────────────────

    def _populate(self):
        """JSON 데이터 파싱하여 왼쪽 리스트 채우기"""
        self.list_titles.clear()
        for it in self._items:
            if not isinstance(it, dict): continue
            title = (it.get("title") or "").strip()
            if not title: continue

            rank = it.get("rank", "")
            src = (it.get("source") or "").strip()
            extra = it.get("extra") if isinstance(it.get("extra"), dict) else {}
            products = extra.get("products") or extra.get("related_products")

            data = {
                "title": title,
                "reason": extra.get("reason") or "",
                "products": products
            }

            item = QtWidgets.QListWidgetItem(f"[{rank}] {title}  ({src})")
            item.setData(QtCore.Qt.UserRole, data)
            self.list_titles.addItem(item)

    def _on_item_changed(self, current, previous):
        """왼쪽 리스트 선택 변경 시 중앙 패널 갱신"""
        self.te_reason.clear()
        self.list_products.clear()
        if not current: return

        data = current.data(QtCore.Qt.UserRole) or {}

        # 사유 표시
        reason = (data.get("reason") or "").strip()
        self.te_reason.setPlainText(reason if reason else "사유 정보 없음")

        # 상품 리스트 표시
        products = data.get("products")
        prod_list = []
        if isinstance(products, list):
            prod_list = [str(x).strip() for x in products if str(x).strip()]
        elif isinstance(products, str) and products.strip():
            prod_list = [p.strip() for p in products.split(",") if p.strip()]

        if not prod_list:
            self.list_products.addItem("(상품 정보 없음)")
        else:
            for p in prod_list:
                self.list_products.addItem(p)

    def _add_title_to_selection(self, item):
        """왼쪽 제목 더블클릭 -> 우측 나의 선택에 추가"""
        data = item.data(QtCore.Qt.UserRole) or {}
        title = data.get("title")
        if not title: return

        # 중복 방지
        for i in range(self.list_my_selection.count()):
            if self.list_my_selection.item(i).text() == title:
                return

        new_item = QtWidgets.QListWidgetItem(title)
        # 중요: 데이터를 그대로 복사해서 넣어둠 (나중에 메인으로 넘기기 위해)
        new_item.setData(QtCore.Qt.UserRole, data)
        self.list_my_selection.addItem(new_item)
        self.list_my_selection.scrollToBottom()

    def _remove_from_selection(self, item):
        """나의 선택 목록에서 제거"""
        row = self.list_my_selection.row(item)
        self.list_my_selection.takeItem(row)

    # ────────────────────────────────────────
    # 데이터 반환
    # ────────────────────────────────────────
    def get_selected_data_full(self) -> list[dict]:
        """
        나의 선택 리스트에 있는 모든 아이템의 데이터(title, products 등)를 리스트로 반환.
        """
        result = []
        for i in range(self.list_my_selection.count()):
            item = self.list_my_selection.item(i)
            data = item.data(QtCore.Qt.UserRole)
            if data:
                result.append(data)
        return result