# all_ui.py
from PyQt5 import QtWidgets
import sys

# 1) 기존 쇼츠 UI 위젯
try:
    from shorts_ui import create_shorts_widget
except ImportError:
    # shorts_ui가 없을 경우 대비 (혹은 기존 파일 그대로 사용)
    def create_shorts_widget(parent=None):
        w = QtWidgets.QWidget(parent)
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel("Shorts UI (Not found)"))
        return w

# 2) blog UI 위젯
try:
    from blog_ui import create_blog_widget
except ImportError:
    def create_blog_widget(parent=None):
        w = QtWidgets.QWidget(parent)
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel("블로그 자동화 UI (todo)"))
        return w

# 3) talk UI 위젯
try:
    from talk_ui import create_talk_widget
except ImportError:
    def create_talk_widget(parent=None):
        w = QtWidgets.QWidget(parent)
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel("Talk / 음성·입모양 자동화 UI (todo)"))
        return w

# 4) youtube UI 위젯 (새로 추가됨)
try:
    from youtube_ui import create_youtube_widget
except ImportError:
    def create_youtube_widget(parent=None):
        w = QtWidgets.QWidget(parent)
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel("유튜브 분석 UI (파일 없음)"))
        return w

# 5) shopping UI 위젯 (쇼핑 자동화)
try:
    from shopping import create_shopping_widget
except ImportError:
    def create_shopping_widget(parent=None):
        w = QtWidgets.QWidget(parent)
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel("쇼핑 자동화 UI (shopping.py 없음)"))
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


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = AllMain()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()