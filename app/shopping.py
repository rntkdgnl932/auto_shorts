# -*- coding: utf-8 -*-
from __future__ import annotations

from PyQt5 import QtWidgets, QtCore, QtGui


class ShoppingWidget(QtWidgets.QWidget):
    """
    쇼핑/쿠팡/쇼츠 자동화용 기본 UI 골격.

    1단계(현재 구현 상태):
      - 상단: 키워드 + 카테고리 + 검색 버튼
      - 중앙: 좌측 상품 리스트 / 우측 선택 상품 상세
      - 우측 하단 버튼:
          [1단계: 리스트가져오기] → 이후 "이슈+트렌드 기반 상품 50개 리스트" 가져오는 트리거
          [쇼츠 스크립트 만들기 (준비중)]
          [인포크링크 텍스트 생성 (준비중)]
      - 하단: 로그창

    실제 로직:
      - 지금은 더미 데이터/로그만 넣어둔 상태.
      - 나중에:
        1) issue_list/{날짜}/{시분초}.json 생성
        2) AI가 쿠팡 아이템 50개 선정
        3) 쇼츠/쇼핑 탭과 연동
      쪽으로 확장 예정.
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
        # noinspection PyUnresolvedReferences
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
        # noinspection PyUnresolvedReferences
        self.lbl_thumbnail.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_thumbnail.setText("이미지 없음")

        info_form = QtWidgets.QFormLayout()
        # noinspection PyUnresolvedReferences
        info_form.setLabelAlignment(QtCore.Qt.AlignRight)
        # noinspection PyUnresolvedReferences
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
        # noinspection PyUnresolvedReferences
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

        # ✅ 새 버튼: 1단계 리스트 가져오기
        self.btn_load_list = QtWidgets.QPushButton("1단계: 리스트가져오기", right_widget)

        # 기존 버튼들
        self.btn_make_script = QtWidgets.QPushButton("쇼츠 스크립트 만들기 (준비중)", right_widget)
        self.btn_make_infok = QtWidgets.QPushButton("인포크링크 텍스트 생성 (준비중)", right_widget)

        # 순서: [1단계 버튼] → [쇼츠 스크립트] → [인포크 텍스트]
        btn_row.addWidget(self.btn_load_list)
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

        # 새 1단계 버튼
        self.btn_load_list.clicked.connect(self.on_load_list_clicked)

        self.btn_make_script.clicked.connect(self.on_make_script_clicked)
        self.btn_make_infok.clicked.connect(self.on_make_infok_clicked)

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

    # ✅ 새 슬롯: 1단계 리스트 가져오기
    def on_load_list_clicked(self):
        """
        1단계: 이슈/트렌드 + 쇼핑 데이터 수집 → issue_list/{날짜}/{시분초}.json 생성
        지금은 아직 설계 단계라, 로그만 남겨둠.
        """
        self.append_log("🧩 [1단계] 리스트 가져오기 트리거 도착.")
        self.append_log("    → 앞으로 여기에서:")
        self.append_log("       1) blog_trend_search_page.collect_all_topics() + 네이버/쿠팡/다나와/뉴스 수집")
        self.append_log("       2) AI로 쿠팡에 올릴 만한 후보 아이템 50개 선정")
        self.append_log("       3) C:\\my_games\\shorts_make\\issue_list\\{날짜}\\{시분초}.json 저장")
        self.append_log("    까지를 한 번에 수행하게 만들 예정.\n")

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
            # noinspection PyUnresolvedReferences
            icon_item.setFlags(icon_item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.table_items.setItem(row, 0, icon_item)

            self.table_items.setItem(row, 1, QtWidgets.QTableWidgetItem(title))
            self.table_items.setItem(row, 2, QtWidgets.QTableWidgetItem(price))
            self.table_items.setItem(row, 3, QtWidgets.QTableWidgetItem(disc))
            self.table_items.setItem(row, 4, QtWidgets.QTableWidgetItem(rate))

        self.append_log(f"ℹ 더미 상품 {len(dummy)}개 로드 (실제 연동 전까지 테스트용).")


# all_ui.py에서 사용하기 위한 팩토리 함수
def create_shopping_widget(parent=None) -> QtWidgets.QWidget:
    return ShoppingWidget(parent)
