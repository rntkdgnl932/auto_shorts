# -*- coding: utf-8 -*-
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
    쇼핑/쿠팡/쇼츠 자동화용 기본 UI 골격.

    1단계(현재 구현 상태):
      - 상단: 키워드 + 카테고리 + 검색 버튼
      - 중앙: 좌측 상품 리스트 / 우측 선택 상품 상세
      - 우측 하단 버튼:
          [테스트] → 네이버 기반 이슈 리스트 JSON 저장 테스트
          [1단계: 리스트가져오기] → 나중에 전체 1단계 파이프라인(네이버+기존 이슈)을 실행
          [쇼츠 스크립트 만들기 (준비중)]
          [인포크링크 텍스트 생성 (준비중)]
      - 하단: 로그창
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

        search_layout.addWidget(lbl_keyword,         0, 0)
        search_layout.addWidget(self.le_keyword,     0, 1, 1, 3)
        search_layout.addWidget(lbl_category,        1, 0)
        search_layout.addWidget(self.combo_category, 1, 1)
        search_layout.addWidget(self.btn_search_popular, 1, 2)
        search_layout.addWidget(self.btn_search_keyword, 1, 3)

        # 2) 중앙 영역: 좌(상품 리스트) / 우(상세 미리보기)
        center_splitter = QtWidgets.QSplitter(self)
        center_splitter.setOrientation(QtCore.Qt.Horizontal)

        # 2-1) 상품 리스트 테이블 (좌측)
        left_widget = QtWidgets.QWidget(center_splitter)
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        lbl_list = QtWidgets.QLabel("상품 리스트", left_widget)
        self.table_items = QtWidgets.QTableWidget(left_widget)
        self.table_items.setColumnCount(5)
        self.table_items.setHorizontalHeaderLabels([
            "이미지", "상품명", "가격", "할인", "평점"
        ])
        self.table_items.horizontalHeader().setStretchLastSection(True)
        self.table_items.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_items.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table_items.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_items.verticalHeader().setVisible(False)
        self.table_items.setIconSize(QtCore.QSize(48, 48))

        left_layout.addWidget(lbl_list)
        left_layout.addWidget(self.table_items)

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
        self.le_title.setReadOnly(True)

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

        # 🔹 공용 테스트 버튼: 1단계(이슈 수집) + 나중에 2단계(쇼핑 키워드)까지 같이 실행할 예정
        self.btn_test = QtWidgets.QPushButton("테스트", right_widget)

        # ✅ 1단계 버튼: 나중에 전체 파이프라인(네이버+기존 이슈)로 확장 예정
        self.btn_load_list = QtWidgets.QPushButton("1단계: 리스트가져오기", right_widget)
        self.btn_load_b = QtWidgets.QPushButton("리스트불러오기", right_widget)
        # 기존 버튼들
        self.btn_make_script = QtWidgets.QPushButton("쇼츠 스크립트 만들기 (준비중)", right_widget)
        self.btn_make_infok = QtWidgets.QPushButton("인포크링크 텍스트 생성 (준비중)", right_widget)

        # 순서: [테스트] → [1단계] → [쇼츠 스크립트] → [인포크 텍스트]
        btn_row.addWidget(self.btn_test)
        btn_row.addWidget(self.btn_load_list)
        btn_row.addWidget(self.btn_load_b)
        btn_row.addWidget(self.btn_make_script)
        btn_row.addWidget(self.btn_make_infok)

        right_layout.addWidget(lbl_detail)
        right_layout.addWidget(thumb_group)
        right_layout.addWidget(link_group)
        right_layout.addLayout(btn_row)

        # splitter에 위젯 장착
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

        # 전체 레이아웃 조립
        main_layout.addWidget(search_box)
        main_layout.addWidget(center_splitter, 1)
        main_layout.addWidget(log_group, 0)

    # ────────────────────────────────────────
    # 시그널 연결
    # ────────────────────────────────────────
    def _wire_signals(self):
        self.btn_search_keyword.clicked.connect(self.on_search_keyword)
        self.btn_search_popular.clicked.connect(self.on_search_popular)
        self.table_items.itemSelectionChanged.connect(self.on_item_selected)

        # 새 테스트 버튼
        self.btn_test.clicked.connect(self.on_test_clicked)

        # 1단계 버튼
        self.btn_load_list.clicked.connect(self.on_load_list_clicked)

        self.btn_make_script.clicked.connect(self.on_make_script_clicked)
        self.btn_make_infok.clicked.connect(self.on_make_infok_clicked)

        self.btn_load_b.clicked.connect(self.on_load_b_clicked)

    # ────────────────────────────────────────
    # 공통 로그 함수
    # ────────────────────────────────────────
    def append_log(self, msg: str):
        self.log.appendPlainText(msg)

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
        self.append_log("   → 나중에 쿠팡 상품 검색/네이버 쇼핑 BEST 연동 함수에 연결.\n")

        # TODO: 여기서 실제 쿠팡 상품 목록 가져오는 함수 연결
        self._populate_dummy_items()

    def on_search_popular(self):
        category = self.combo_category.currentText().strip()
        self.append_log(f"🔥 인기템 리스트 불러오기 준비중... (카테고리: {category})")
        self.append_log("   → 추후 '쿠팡 BEST/급상승', 네이버 쇼핑 BEST, 다나와 등과 연결.\n")
        self._populate_dummy_items()

    def on_item_selected(self):
        rows = self.table_items.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        title_item = self.table_items.item(row, 1)
        price_item = self.table_items.item(row, 2)
        disc_item = self.table_items.item(row, 3)
        rate_item = self.table_items.item(row, 4)

        title = title_item.text() if title_item else ""
        price = price_item.text() if price_item else ""
        disc = disc_item.text() if disc_item else ""
        rate = rate_item.text() if rate_item else ""

        self.le_title.setText(title)
        self.le_price.setText(price)
        self.le_discount.setText(disc)
        self.le_rating.setText(rate)

        # TODO: 실제 상품/이미지/링크 정보와 연결할 때 채우기
        self.le_product_url.setText("https://www.coupang.com/...")
        self.le_affiliate_url.setText("https://link.coupang.com/partners/...")
        self.te_description.setPlainText(f"{title} 에 대한 쇼츠 시나리오/상품 소개를 여기에 채울 예정.")

        self.append_log(f"✅ 상품 선택: {title}")

    def on_test_clicked(self):
        """
        테스트 버튼:
        - 최신 _a.json의 title들을 AI에게 한 번에 전달
        - 판매 가능 상품 후보를 뽑아
        - issue_list/YYYYMMDD/HHMMSS_b.json 저장
        """
        self.append_log("🧪 [테스트] AI로 판매 가능 상품 후보(_b) 생성 시작...")

        def job(progress):
            progress({"msg": "[테스트] AI 분석 중... (_a -> _b)"})
            path = save_issue_list_for_shopping_ai_b_from_a(on_progress=progress)
            return {"path": str(path)}

        def done(ok, payload, err):
            if (not ok) or (err is not None):
                self.append_log(f"❌ [테스트] AI 후보 생성 실패: {err}")
                return

            path_str = ""
            if isinstance(payload, dict):
                path_str = (payload.get("path") or "").strip()

            if path_str:
                self.append_log(f"✅ [테스트] _b 저장 완료: {path_str}")
            else:
                self.append_log("⚠ [테스트] 완료됐지만 결과 경로를 확인하지 못했습니다.")

        run_job_with_progress_async(
            owner=self,
            title="AI 상품 후보 생성 (테스트: _a -> _b)",
            job=job,
            on_done=done,
        )

    # ✅ 1단계 버튼 (지금은 계획 설명만, 나중에 전체 파이프라인 연결 예정)
    def on_load_list_clicked(self):
        """
        1단계: 이슈/트렌드 + 쇼핑 데이터 수집 → issue_list/{날짜}/{시분초}.json 생성
        지금은 아직 설계 단계라, 나중에 통합 파이프라인을 여기에 붙일 예정.
        """
        self.append_log("🧩 [1단계] 리스트 가져오기 트리거 도착.")
        self.append_log("    → 앞으로 여기에서:")
        self.append_log("       1) blog_trend_search_page.collect_all_topics() + 네이버/쿠팡/다나와/뉴스 수집")
        self.append_log("       2) AI로 쿠팡에 올릴 만한 후보 아이템 50개 선정")
        self.append_log("       3) C:\\my_games\\shorts_make\\issue_list\\{날짜}\\{시분초}.json 저장")
        self.append_log("    까지를 한 번에 수행하게 만들 예정.\n")
        self._run_stage1_collect_issue_list(trigger_label="1단계버튼")

        # 추후 상세내용 파악해서 추출하기.

    def _run_stage1_collect_issue_list(self, trigger_label: str):
        """
        1단계(실행): 쇼핑 이슈 전체 수집 → issue_list/날짜/시분초.json 저장
        - save_issue_list_for_shopping_all()가 실제 수집/저장 수행
        - 성공 시 payload = {"path": "<저장경로>"} 형태로 done()에 전달
        """
        self.append_log(f"🧩 [1단계] ({trigger_label}) 쇼핑 이슈 전체 수집 시작...")

        def job(progress):
            progress({"msg": f"[1단계/{trigger_label}] 쇼핑 이슈 수집 중..."})
            path = save_issue_list_for_shopping_all(on_progress=progress)
            return {"path": str(path) if path is not None else ""}

        def done(ok, payload, err):
            if (not ok) or (err is not None):
                self.append_log(f"❌ [1단계/{trigger_label}] 쇼핑 이슈 수집 실패: {err}")
                return

            path_str = ""
            if isinstance(payload, dict):
                path_str = (payload.get("path") or "").strip()

            if path_str:
                self.append_log(f"✅ [1단계/{trigger_label}] 쇼핑 이슈 리스트 저장 완료: {path_str}")
            else:
                self.append_log(f"⚠ [1단계/{trigger_label}] 수집은 끝났지만, 결과 경로를 알 수 없습니다.")

        run_job_with_progress_async(
            owner=self,
            title=f"쇼핑 이슈 수집 (1단계/{trigger_label})",
            job=job,
            on_done=done,
        )

    def on_make_script_clicked(self):
        title = self.le_title.text().strip()
        if not title:
            self.append_log("⚠ 먼저 상품을 선택해 주세요.")
            return
        self.append_log(f"✏ 쇼츠/영상용 스크립트 생성 준비중... (상품: {title})")
        self.append_log("   → 이후 GPT + ComfyUI 파이프라인에 연결해서 시나리오/이미지 생성.\n")

    def on_make_infok_clicked(self):
        title = self.le_title.text().strip()
        aff_link = self.le_affiliate_url.text().strip()
        if not title or not aff_link:
            self.append_log("⚠ 상품과 파트너스 링크를 먼저 확인해 주세요.")
            return
        self.append_log(f"🧱 인포크링크 블록 텍스트 생성 준비중... (상품: {title})")
        self.append_log(f"   링크: {aff_link}")
        self.append_log("   → 나중에 '클립보드 복사' or '브라우저 자동 등록' 모듈에 연결.\n")

    # ────────────────────────────────────────
    # 임시 더미 데이터 (초기 UI 테스트용)
    # ────────────────────────────────────────
    def _populate_dummy_items(self):
        """
        실제 쿠팡/네이버/다나와 연동 전까지는 더미 데이터로 UI 확인용.
        나중에 쇼핑 이슈 기반 후보 50개를 여기에 채워 넣으면 됨.
        """
        dummy = [
            ("", "강아지 겨울 패딩 점퍼", "19,900원", "30%↓", "⭐ 4.7"),
            ("", "기모 맨투맨 후드티 (남녀공용)", "24,900원", "15%↓", "⭐ 4.5"),
            ("", "USB C 고속 충전 케이블 3종 세트", "9,900원", "50%↓", "⭐ 4.8"),
        ]
        self.table_items.setRowCount(len(dummy))
        for row, (img, title, price, disc, rate) in enumerate(dummy):
            # 이미지 셀은 나중에 QIcon 세팅
            icon_item = QtWidgets.QTableWidgetItem()
            icon_item.setFlags(icon_item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.table_items.setItem(row, 0, icon_item)

            self.table_items.setItem(row, 1, QtWidgets.QTableWidgetItem(title))
            self.table_items.setItem(row, 2, QtWidgets.QTableWidgetItem(price))
            self.table_items.setItem(row, 3, QtWidgets.QTableWidgetItem(disc))
            self.table_items.setItem(row, 4, QtWidgets.QTableWidgetItem(rate))

        self.append_log(f"ℹ 더미 상품 {len(dummy)}개 로드 (실제 연동 전까지 테스트용).")

    def on_load_b_clicked(self):
        """
        리스트불러오기:
        - _b.json을 선택해서 불러오거나(파일 선택)
        - 취소 시 최신 _b.json 자동 로드 시도
        - 새 창에서 클릭 가능한 리스트로 표시
        """
        import json
        from pathlib import Path

        default_dir = str(Path(r"C:\my_games\shorts_make\issue_list"))

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "리스트 불러오기 (_b.json)",
            default_dir,
            "JSON Files (*.json);;All Files (*.*)"
        )

        items = None

        def _load_json(p: str):
            try:
                txt = Path(p).read_text(encoding="utf-8")
                obj = json.loads(txt)
                return obj if isinstance(obj, list) else None
            except Exception:
                return None

        if path:
            items = _load_json(path)
            if items is None:
                QtWidgets.QMessageBox.warning(self, "실패", "JSON 로드 실패 또는 리스트 구조가 아닙니다.")
                return
            dlg = IssueListViewerDialog(items, parent=self, title=f"리스트 보기: {Path(path).name}")
            dlg.exec_()
            return

        # 파일 선택 취소 → 최신 _b 자동 시도 (간단 로직)
        root = Path(default_dir)
        latest = None
        try:
            if root.exists():
                date_dirs = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
                for d in date_dirs:
                    cand = sorted(d.glob("*_b.json"), key=lambda x: x.name, reverse=True)
                    if cand:
                        latest = cand[0]
                        break
        except Exception:
            latest = None

        if not latest:
            QtWidgets.QMessageBox.information(self, "안내", "불러올 _b.json이 없습니다. 먼저 테스트 버튼으로 _b를 생성하세요.")
            return

        items = _load_json(str(latest))
        if items is None:
            QtWidgets.QMessageBox.warning(self, "실패", f"최신 _b.json 로드 실패: {latest}")
            return

        dlg = IssueListViewerDialog(items, parent=self, title=f"리스트 보기: {latest.name}")
        dlg.exec_()

# all_ui.py에서 사용하기 위한 팩토리 함수
def create_shopping_widget(parent=None) -> QtWidgets.QWidget:
    return ShoppingWidget(parent)




class IssueListViewerDialog(QtWidgets.QDialog):
    def __init__(self, items, parent=None, title="리스트 보기"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(980, 600)

        self._items = items or []

        root = QtWidgets.QVBoxLayout(self)

        # ✅ 3분할 스플리터(좌: 제목, 중: reason, 우: products)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        root.addWidget(splitter, 1)

        # ---------------------------
        # LEFT: title list
        # ---------------------------
        left_wrap = QtWidgets.QWidget(self)
        left_lay = QtWidgets.QVBoxLayout(left_wrap)
        left_lay.setContentsMargins(8, 8, 8, 8)

        left_title = QtWidgets.QLabel("제목(Title)", left_wrap)
        left_lay.addWidget(left_title)

        self.listw = QtWidgets.QListWidget(left_wrap)
        self.listw.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        left_lay.addWidget(self.listw, 1)

        splitter.addWidget(left_wrap)

        # ---------------------------
        # MIDDLE: reason
        # ---------------------------
        mid_wrap = QtWidgets.QWidget(self)
        mid_lay = QtWidgets.QVBoxLayout(mid_wrap)
        mid_lay.setContentsMargins(8, 8, 8, 8)

        mid_title = QtWidgets.QLabel("사유(Reason)", mid_wrap)
        mid_lay.addWidget(mid_title)

        self.te_reason = QtWidgets.QTextEdit(mid_wrap)
        self.te_reason.setReadOnly(True)
        self.te_reason.setPlaceholderText("왼쪽에서 항목을 선택하면 reason이 표시됩니다.")
        mid_lay.addWidget(self.te_reason, 1)

        splitter.addWidget(mid_wrap)

        # ---------------------------
        # RIGHT: products list
        # ---------------------------
        right_wrap = QtWidgets.QWidget(self)
        right_lay = QtWidgets.QVBoxLayout(right_wrap)
        right_lay.setContentsMargins(8, 8, 8, 8)

        right_title = QtWidgets.QLabel("상품(Products)", right_wrap)
        right_lay.addWidget(right_title)

        self.list_products = QtWidgets.QListWidget(right_wrap)
        self.list_products.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        right_lay.addWidget(self.list_products, 1)

        splitter.addWidget(right_wrap)

        # 초기 비율(원하면 조절)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 3)

        # 하단 힌트/버튼
        hint = QtWidgets.QLabel("더블클릭: URL 있으면 열기 / 없으면 제목 복사", self)
        root.addWidget(hint)

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        self.btn_copy = QtWidgets.QPushButton("선택 제목 복사", self)
        self.btn_close = QtWidgets.QPushButton("닫기", self)
        btns.addWidget(self.btn_copy)
        btns.addWidget(self.btn_close)
        root.addLayout(btns)

        self.btn_close.clicked.connect(self.close)
        self.btn_copy.clicked.connect(self._copy_selected)

        # ✅ 선택 변화 시 중앙/우측 갱신
        self.listw.currentItemChanged.connect(self._on_item_changed)
        self.listw.itemDoubleClicked.connect(self._open_or_copy)

        self._populate()

        # 첫 항목 자동 선택(있으면)
        if self.listw.count() > 0:
            self.listw.setCurrentRow(0)

    def _populate(self):
        self.listw.clear()
        for it in self._items:
            if not isinstance(it, dict):
                continue

            title = (it.get("title") or "").strip()
            if not title:
                continue

            rank = it.get("rank", "")
            src = (it.get("source") or "").strip()
            url = it.get("url")

            extra = it.get("extra") if isinstance(it.get("extra"), dict) else {}
            # products 키는 케이스 다양성을 조금 허용
            products = extra.get("products")
            if products is None:
                products = extra.get("related_products")

            data = {
                "title": title,
                "url": url,
                "extra": extra,
                "reason": extra.get("reason") or "",
                "products": products,
            }

            text = f"[{rank}] {title}  ({src})"
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.UserRole, data)
            self.listw.addItem(item)

    def _on_item_changed(self, current, previous):
        self.te_reason.clear()
        self.list_products.clear()

        if not current:
            return

        data = current.data(QtCore.Qt.UserRole) or {}
        reason = (data.get("reason") or "").strip()

        if not reason:
            # extra 자체를 reason 대신 보여주고 싶으면 여기서 확장 가능
            reason = "reason 정보가 없습니다."
        self.te_reason.setPlainText(reason)

        products = data.get("products")

        # products 형태 정규화
        prod_list = []
        if isinstance(products, list):
            prod_list = [str(x).strip() for x in products if str(x).strip()]
        elif isinstance(products, str) and products.strip():
            # "A, B, C" 같은 문자열도 지원
            if "," in products:
                prod_list = [p.strip() for p in products.split(",") if p.strip()]
            else:
                prod_list = [products.strip()]
        elif isinstance(products, dict):
            # dict면 key/value 펼쳐서라도 보여줌
            for k, v in products.items():
                s = f"{k}: {v}"
                if s.strip():
                    prod_list.append(s)

        if not prod_list:
            self.list_products.addItem("(products 없음)")
        else:
            for p in prod_list:
                self.list_products.addItem(p)

    def _copy_selected(self):
        item = self.listw.currentItem()
        if not item:
            return
        data = item.data(QtCore.Qt.UserRole) or {}
        title = data.get("title") or ""
        if title:
            QtWidgets.QApplication.clipboard().setText(title)

    def _open_or_copy(self, item):
        data = item.data(QtCore.Qt.UserRole) or {}
        url = data.get("url")
        title = data.get("title") or ""

        if url:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(str(url)))
        else:
            if title:
                QtWidgets.QApplication.clipboard().setText(title)


