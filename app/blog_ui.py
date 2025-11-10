# blog_ui.py
from PyQt5 import QtWidgets

def create_blog_widget(parent=None):
    w = QtWidgets.QWidget(parent)
    lay = QtWidgets.QVBoxLayout(w)
    lay.addWidget(QtWidgets.QLabel("블로그 자동화 설정/생성 UI"))
    # 여기다 나중에 wordpress, 카테고리, 프롬프트, 미리보기 등 넣으면 됨
    return w
