# blog_ui.py
from PyQt5 import QtWidgets

# 너가 말한 대로 여기서 가져옴
from blog_start import suggest_life_tip_topic, issue_start

# utils 경로는 프로젝트 구조에 맞게
try:
    from app.utils import run_job_with_progress_async
except ImportError:
    from utils import run_job_with_progress_async


class BlogPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._wire()

    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)

        # 1) 생활정보 주제 버튼
        self.btn_suggest = QtWidgets.QPushButton("자동 주제 뽑아서 바로 작성")
        lay.addWidget(self.btn_suggest)

        # 2) 이슈 주제 버튼
        self.btn_issue = QtWidgets.QPushButton("이슈 주제 뽑아서 바로 작성")
        lay.addWidget(self.btn_issue)

        # 3) 로그창
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        lay.addWidget(self.log)

    def _wire(self):
        self.btn_suggest.clicked.connect(self.on_suggest)
        self.btn_issue.clicked.connect(self.on_issue)

    # ── 생활정보 자동 주제
    def on_suggest(self):
        self.btn_suggest.setEnabled(False)

        def job(progress):
            progress({"msg": "생활정보 주제 추천 + 작성 중..."})
            ok = suggest_life_tip_topic()
            progress({"msg": f"끝: {ok}"})
            return ok

        def done(ok, payload, err):
            if ok:
                self.log.appendPlainText(f"✅ 생활정보 글 생성 완료: {payload}")
            else:
                self.log.appendPlainText(f"❌ 생활정보 글 생성 실패: {err}")
            self.btn_suggest.setEnabled(True)

        run_job_with_progress_async(
            owner=self,
            title="생활정보 글 자동 생성",
            job=job,
            on_done=done,
        )

    # ── 이슈 자동 주제
    def on_issue(self):
        self.btn_issue.setEnabled(False)

        def job(progress):
            progress({"msg": "이슈 주제 수집/필터링 중..."})
            # 여기서 blog_start.issue_start()가 blog_trend_search_page를 요구함
            ok = issue_start()
            progress({"msg": f"끝: {ok}"})
            return ok

        def done(ok, payload, err):
            if ok:
                self.log.appendPlainText(f"✅ 이슈 글 생성 완료: {payload}")
            else:
                self.log.appendPlainText(f"❌ 이슈 글 생성 실패: {err}")
            self.btn_issue.setEnabled(True)

        run_job_with_progress_async(
            owner=self,
            title="이슈 글 자동 생성",
            job=job,
            on_done=done,
        )


def create_blog_widget(parent=None):
    return BlogPage(parent)
