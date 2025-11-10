# all_ui.py
from PyQt5 import QtWidgets
import sys

# 1) 기존 쇼츠 UI를 위젯으로 만들어 주는 함수/클래스를 가져온다
from shorts_ui import create_shorts_widget  # 이건 아래에서 우리가 추가할 거야
# 2) 아직 없으니 임시 blog_ui
try:
    from blog_ui import create_blog_widget
except ImportError:
    def create_blog_widget(parent=None):
        w = QtWidgets.QWidget(parent)
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel("블로그 자동화 UI (todo)"))
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


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = AllMain()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
