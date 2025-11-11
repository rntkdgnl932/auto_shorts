# blog_ui.py
from PyQt5 import QtWidgets
import os

# .env 읽기 (있으면)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from blog_start import suggest_life_tip_topic, issue_start

try:
    from app.utils import run_job_with_progress_async
except ImportError:
    from utils import run_job_with_progress_async

try:
    import settings as s_mod
except Exception:
    from app import settings as s_mod  # type: ignore


# 공통: 카테고리 파일 경로
CATEGORY_FILE = r"C:\my_games\shorts_make\app\blog_setting\category_list.txt"


def load_category_lines() -> list[str]:
    """카테고리 txt를 읽어서 줄 단위 리스트로 반환"""
    if not os.path.exists(CATEGORY_FILE):
        return []
    with open(CATEGORY_FILE, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
    return lines


def save_category_lines(lines: list[str]) -> None:
    os.makedirs(os.path.dirname(CATEGORY_FILE), exist_ok=True)
    with open(CATEGORY_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─────────────────────────────
# 메인 탭
# ─────────────────────────────
class BlogMainWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._wire()

    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)

        # (1) 카테고리 선택 콤보박스
        self.combo_category = QtWidgets.QComboBox()
        cats = load_category_lines()
        if cats:
            self.combo_category.addItems(cats)
        else:
            self.combo_category.addItem("카테고리 없음 (category_list.txt 확인)")

        lay.addWidget(QtWidgets.QLabel("블로그 카테고리 선택"))
        lay.addWidget(self.combo_category)

        # (2) 버튼들
        self.btn_suggest = QtWidgets.QPushButton("자동 주제 뽑아서 바로 작성")
        lay.addWidget(self.btn_suggest)

        self.btn_issue = QtWidgets.QPushButton("이슈 주제 뽑아서 바로 작성")
        lay.addWidget(self.btn_issue)

        # (3) 로그창
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        lay.addWidget(self.log)

    def _wire(self):
        self.btn_suggest.clicked.connect(self.on_suggest)
        self.btn_issue.clicked.connect(self.on_issue)

    # ── 생활정보 자동 주제
    def on_suggest(self):
        self.btn_suggest.setEnabled(False)
        selected_cat = self.combo_category.currentText().strip()

        def job(progress):
            progress({"msg": f"[{selected_cat}] 생활정보 주제 추천 + 작성 중..."})
            # 현재 blog_start.suggest_life_tip_topic()이 카테고리 인자를 안 받으니까
            # 여기서는 선택한 카테고리를 로그로만 남겨둔다.
            ok = suggest_life_tip_topic()
            progress({"msg": f"끝: {ok}"})
            return {"ok": ok, "category": selected_cat}

        def done(ok, payload, err):
            if ok:
                self.log.appendPlainText(
                    f"✅ 생활정보 글 생성 완료 (카테고리: {payload.get('category')})"
                )
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
        selected_cat = self.combo_category.currentText().strip()

        def job(progress):
            progress({"msg": f"[{selected_cat}] 이슈 주제 수집/필터링 중..."})
            ret = issue_start()  # 우리가 수정한 버전이면 dict가 내려올 거고, 아니면 True/False
            progress({"msg": f"끝: {ret}"})
            return {"result": ret, "category": selected_cat}

        def done(ok, payload, err):
            if ok:
                self.log.appendPlainText(
                    f"✅ 이슈 글 생성 완료 (카테고리: {payload.get('category')}) → {payload.get('result')}"
                )
            else:
                self.log.appendPlainText(f"❌ 이슈 글 생성 실패: {err}")
            self.btn_issue.setEnabled(True)

        run_job_with_progress_async(
            owner=self,
            title="이슈 글 자동 생성",
            job=job,
            on_done=done,
        )


# ─────────────────────────────
# 설정 탭
# ─────────────────────────────
class BlogSettingsWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        form = QtWidgets.QFormLayout(self)

        # COMFY_HOST
        comfy_default = os.getenv("COMFY_HOST", getattr(s_mod, "COMFY_HOST", "http://127.0.0.1:8188"))
        self.le_comfy = QtWidgets.QLineEdit(comfy_default)
        form.addRow("COMFY_HOST", self.le_comfy)

        # WP 정보
        self.le_wp_domain = QtWidgets.QLineEdit(
            os.getenv("WP_DOMAIN", getattr(s_mod, "WP_DOMAIN", ""))
        )
        self.le_wp_id = QtWidgets.QLineEdit(
            os.getenv("WP_ID", getattr(s_mod, "WP_ID", ""))
        )
        self.le_wp_pw = QtWidgets.QLineEdit(
            os.getenv("WP_PW", getattr(s_mod, "WP_PW", ""))
        )
        self.le_wp_pw.setEchoMode(QtWidgets.QLineEdit.Password)
        form.addRow("WP_DOMAIN", self.le_wp_domain)
        form.addRow("WP_ID", self.le_wp_id)
        form.addRow("WP_PW", self.le_wp_pw)

        # fallback 이미지 경로
        self.le_fb_thumb = QtWidgets.QLineEdit(
            os.getenv("FALLBACK_THUMB_PATH", getattr(s_mod, "FALLBACK_THUMB_PATH", ""))
        )
        self.le_fb_scene = QtWidgets.QLineEdit(
            os.getenv("FALLBACK_SCENE_PATH", getattr(s_mod, "FALLBACK_SCENE_PATH", ""))
        )
        form.addRow("FALLBACK_THUMB_PATH", self.le_fb_thumb)
        form.addRow("FALLBACK_SCENE_PATH", self.le_fb_scene)

        # (중요) 카테고리 리스트 편집창
        self.te_categories = QtWidgets.QPlainTextEdit()
        # 파일에서 바로 읽어와서 한 줄씩 보여준다.
        existing = load_category_lines()
        self.te_categories.setPlainText("\n".join(existing))
        form.addRow("카테고리 목록 (한 줄에 하나)", self.te_categories)

        self.btn_save = QtWidgets.QPushButton("저장(오버라이드 + 카테고리 txt)")
        form.addRow(self.btn_save)

        self.btn_save.clicked.connect(self._save)

    def _save(self):
        try:
            # 1) settings_local.json 에 저장
            path = s_mod.save_overrides(
                COMFY_HOST=self.le_comfy.text().strip(),
                WP_DOMAIN=self.le_wp_domain.text().strip(),
                WP_ID=self.le_wp_id.text().strip(),
                WP_PW=self.le_wp_pw.text().strip(),
                FALLBACK_THUMB_PATH=self.le_fb_thumb.text().strip(),
                FALLBACK_SCENE_PATH=self.le_fb_scene.text().strip(),
            )

            # 2) 카테고리 파일에도 저장
            lines = [ln.strip() for ln in self.te_categories.toPlainText().splitlines() if ln.strip()]
            save_category_lines(lines)

            QtWidgets.QMessageBox.information(
                self,
                "저장 완료",
                f"settings_local.json 저장됨\n{path}\n카테고리도 저장됨",
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "저장 실패", str(e))


# ─────────────────────────────
# 최종 블로그 페이지
# ─────────────────────────────
class BlogPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(BlogMainWidget(self), "메인")
        tabs.addTab(BlogSettingsWidget(self), "설정")

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(tabs)
        self.setLayout(lay)


def create_blog_widget(parent=None):
    return BlogPage(parent)
