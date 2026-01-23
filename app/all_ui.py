# all_ui.py
from PyQt5 import QtWidgets
import sys

# 1) 기존 쇼츠 UI 위젯
try:
    from app.shorts_ui import create_shorts_widget
except Exception as e:
    import traceback
    print("[IMPORT_FAIL] app.shorts_ui import failed:", repr(e))
    traceback.print_exc()

    def create_shorts_widget(parent=None):
        w = QtWidgets.QWidget(parent)
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel("Shorts UI (Not found?)"))
        return w


# 2) blog UI 위젯
try:
    from app.blog_ui import create_blog_widget
except ImportError:
    def create_blog_widget(parent=None):
        w = QtWidgets.QWidget(parent)
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel("블로그 자동화 UI (todo)"))
        return w

# 3) talk UI 위젯
try:
    from app.talk_ui import create_talk_widget
except ImportError:
    def create_talk_widget(parent=None):
        w = QtWidgets.QWidget(parent)
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel("Talk / 음성·입모양 자동화 UI (todo)"))
        return w

# 4) youtube UI 위젯 (새로 추가됨)
try:
    from app.youtube_ui import create_youtube_widget
except ImportError:
    def create_youtube_widget(parent=None):
        w = QtWidgets.QWidget(parent)
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel("유튜브 분석 UI (파일 없음)"))
        return w

# 5) shopping UI 위젯 (쇼핑 자동화)
try:
    from app.shopping import create_shopping_widget
except ImportError:
    def create_shopping_widget(parent=None):
        w = QtWidgets.QWidget(parent)
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel("쇼핑 자동화 UI (shopping.py 없음)"))
        return w

# 6) [새로 추가] Update UI 위젯
try:
    from app.update_ui import create_update_widget
except ImportError:
    def create_update_widget(parent=None):
        w = QtWidgets.QWidget(parent)
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel("업데이트 UI (update_ui.py 없음)"))
        return w

class AllMain(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("자동화 허브")
        self.resize(1200, 800)
        self._build_ui()

    def _build_ui(self):
        tabs = QtWidgets.QTabWidget(self)
        self.setCentralWidget(tabs)

        # shorts 탭
        shorts_page = create_shorts_widget(self)
        tabs.addTab(shorts_page, "Shorts")

        # blog 탭
        blog_page = create_blog_widget(self)
        tabs.addTab(blog_page, "Blog")

        # talk 탭
        talk_page = create_talk_widget(self)
        tabs.addTab(talk_page, "Talk")

        # youtube 탭
        youtube_page = create_youtube_widget(self)
        tabs.addTab(youtube_page, "Youtube")

        # shopping 탭 (새로 추가)
        shopping_page = create_shopping_widget(self)
        tabs.addTab(shopping_page, "Shopping")

        # [새로 추가] Update 탭
        update_page = create_update_widget(self)
        tabs.addTab(update_page, "Update")


def main():
    app = QtWidgets.QApplication(sys.argv)

    # ✅ 다크 테마 + 체크박스/트리 UI 개선
    app.setStyleSheet("""
    /* 기본 위젯 공통 */
    QWidget {
        background-color: #121212;
        color: #E0E0E0;
        font-size: 13px;
    }

    QTabWidget::pane {
        border: 1px solid #2c2c2c;
    }

    QTabBar::tab {
        background: #1e1e1e;
        color: #bdbdbd;
        padding: 8px 16px;
        border: 1px solid #2c2c2c;
        border-bottom: none;
        min-width: 90px;
    }

    QTabBar::tab:selected {
        background: #2b2b2b;
        color: #ffffff;
    }

    QTabBar::tab:hover {
        background: #333333;
    }

    QPushButton {
        background-color: #2d2d2d;
        border: 1px solid #3c3c3c;
        padding: 6px 12px;
        border-radius: 4px;
    }

    QPushButton:hover {
        background-color: #3a3a3a;
    }

    QPushButton:pressed {
        background-color: #1f1f1f;
    }

    QLineEdit, QTextEdit, QPlainTextEdit {
        background-color: #1c1c1c;
        border: 1px solid #3c3c3c;
        padding: 4px;
        selection-background-color: #007acc;
    }

    QLabel {
        background: transparent;
    }

    /* ─────────────────────────
       1) 체크박스(Shorts 탭) 가시성 강화
       ───────────────────────── */
    QCheckBox {
        spacing: 6px;
        color: #E0E0E0;
    }

    QCheckBox:checked {
        font-weight: 600;
        color: #ffffff;
    }

    QCheckBox::indicator {
        width: 16px;
        height: 16px;
        border-radius: 3px;
        border: 1px solid #888888;
        background: #1f1f1f;
    }

    QCheckBox::indicator:unchecked {
        background: #1f1f1f;
        border: 1px solid #555555;
    }

    QCheckBox::indicator:checked {
        background: #00c853;          /* 선명한 초록 체크박스 */
        border: 1px solid #00e676;
    }

    QCheckBox::indicator:disabled {
        background: #333333;
        border: 1px solid #555555;
    }

    /* ─────────────────────────
       2) 트리/리스트 (Shopping 탭 리스트)
       ───────────────────────── */
    QTreeWidget, QTreeView, QListWidget {
        background-color: #101010;
        alternate-background-color: #181818;
        color: #E0E0E0;
        border: 1px solid #333333;
    }

    QTreeWidget::item, QTreeView::item, QListWidget::item {
        padding: 2px 6px;
    }

    /* 선택됐을 때(체크된 항목 포함) 색 강조 */
    QTreeWidget::item:selected,
    QTreeView::item:selected,
    QListWidget::item:selected {
        background-color: #2962ff;     /* 진한 파란색 선택 배경 */
        color: #ffffff;
    }

    QTreeWidget::item:selected:inactive,
    QTreeView::item:selected:inactive,
    QListWidget::item:selected:inactive {
        background-color: #29434e;
    }

    /* 헤더 (혹시 사용 중일 경우) */
    QHeaderView::section {
        background-color: #1f1f1f;
        color: #E0E0E0;
        padding: 4px;
        border: 1px solid #303030;
    }

    /* ─────────────────────────
       3) GroupBox 테두리/타이틀 가독성
       ───────────────────────── */
    QGroupBox {
        border: 1px solid #333333;
        margin-top: 12px;
        padding-top: 16px;
        border-radius: 4px;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
        color: #b0bec5;
        background-color: #121212;
    }
    """)

    win = AllMain()
    win.show()
    sys.exit(app.exec_())




if __name__ == "__main__":
    main()