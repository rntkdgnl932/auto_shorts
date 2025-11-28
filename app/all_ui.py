# all_ui.py
from PyQt5 import QtWidgets
import sys

# 1) 기존 쇼츠 UI 위젯
from shorts_ui import create_shorts_widget  # 기존 그대로 사용

# 2) blog UI 위젯
try:
    from blog_ui import create_blog_widget
except ImportError:
    def create_blog_widget(parent=None):
        w = QtWidgets.QWidget(parent)
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel("블로그 자동화 UI (todo)"))
        return w

# 3) talk UI 위젯 (새 탭)
try:
    from talk_ui import create_talk_widget
except ImportError:
    def create_talk_widget(parent=None):
        """talk_ui가 아직 없을 때 임시로 쓰는 플레이스홀더"""
        w = QtWidgets.QWidget(parent)
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel("Talk / 음성·입모양 자동화 UI (todo)"))
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
        tabs.addTab(shorts_page, "shorts")

        # blog 탭
        blog_page = create_blog_widget(self)
        tabs.addTab(blog_page, "blog")

        # talk 탭 (새로운 탭)
        talk_page = create_talk_widget(self)
        tabs.addTab(talk_page, "talk")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = AllMain()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
