# blog_ui.py
from PyQt5 import QtWidgets, QtCore
import os

# .env 읽기 (있으면)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from app.blog_start import suggest_life_tip_topic, issue_start, life_tips_keyword


from app.utils import run_job_with_progress_async

from app import settings as s_mod

import variable as v_  # ← 여기서 우리가 말한 파일 경로들 전부 가져옴


ENV_PATH = r"C:\my_games\shorts_make\app\.env"
v_.category_list = r"C:\my_games\shorts_make\app\blog_setting\category_list.txt"


def load_category_lines() -> list[str]:
    if not os.path.exists(v_.category_list):
        return []
    with open(v_.category_list, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f.read().splitlines() if ln.strip()]


def save_env_blog_category(new_value: str) -> None:
    """ .env 파일에서 BLOG_CATEGORY= 줄만 교체/추가 """
    lines = []
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

    found = False
    new_lines = []
    for ln in lines:
        if ln.startswith("BLOG_CATEGORY="):
            new_lines.append(f"BLOG_CATEGORY={new_value}")
            found = True
        else:
            new_lines.append(ln)

    if not found:
        new_lines.append(f"BLOG_CATEGORY={new_value}")

    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")

    # 런타임에서도 바로 반영
    os.environ["BLOG_CATEGORY"] = new_value


def save_category_lines(lines: list[str]) -> None:
    os.makedirs(os.path.dirname(v_.category_list), exist_ok=True)
    with open(v_.category_list, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─────────────────────────────
# 메인 탭
# ─────────────────────────────
class BlogMainWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._wire()

        # 반복 실행용 타이머
        self.timer_suggest = QtCore.QTimer(self)
        self.timer_suggest.timeout.connect(self._run_suggest_once)

        self.timer_issue = QtCore.QTimer(self)
        self.timer_issue.timeout.connect(self._run_issue_once)

        # 10분마다 상태 찍는 하트비트 타이머
        self.timer_heartbeat = QtCore.QTimer(self)
        self.timer_heartbeat.timeout.connect(self._heartbeat)
        self.timer_heartbeat.start(10 * 60 * 1000)  # 10분

        # 실행 중인지 체크
        self._suggest_running = False
        self._issue_running = False

        # 처음 켰을 때 선택된 카테고리 기준으로 프롬프트도 불러온다
        self._load_prompt_for_current_category()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)

        # ── 위쪽: [카테고리]  [반복(분)]  [버튼들]
        top = QtWidgets.QHBoxLayout()

        # 1) 왼쪽: 카테고리
        left_box = QtWidgets.QVBoxLayout()
        lbl_cat = QtWidgets.QLabel("블로그 카테고리 선택")
        self.combo_category = QtWidgets.QComboBox()
        self.combo_category.setMaximumWidth(350)
        cats = load_category_lines()
        if cats:
            self.combo_category.addItems(cats)
        else:
            self.combo_category.addItem("카테고리 없음 (category_list.txt 확인)")
        left_box.addWidget(lbl_cat)
        left_box.addWidget(self.combo_category)
        left_box.addStretch(1)

        # 2) 가운데: 반복(분)
        mid_box = QtWidgets.QVBoxLayout()
        lbl_repeat = QtWidgets.QLabel("반복(분)")
        self.sb_repeat = QtWidgets.QSpinBox()
        self.sb_repeat.setRange(1, 10080)  # 1분 ~ 7일
        self.sb_repeat.setValue(60)        # 디폴트 60분
        mid_box.addWidget(lbl_repeat)
        mid_box.addWidget(self.sb_repeat)
        mid_box.addStretch(1)

        # 3) 오른쪽: 작성 버튼들
        right_box = QtWidgets.QVBoxLayout()
        self.btn_suggest = QtWidgets.QPushButton("자동 주제 뽑아서 바로 작성")
        self.btn_issue = QtWidgets.QPushButton("이슈 주제 뽑아서 바로 작성")
        right_box.addWidget(self.btn_suggest)
        right_box.addWidget(self.btn_issue)
        right_box.addStretch(1)

        top.addLayout(left_box, 1)
        top.addLayout(mid_box, 0)
        top.addLayout(right_box, 0)

        root.addLayout(top)

        # ── 키워드 직접 입력 후 1회 생성 영역
        keyword_row = QtWidgets.QHBoxLayout()
        lbl_keyword = QtWidgets.QLabel("키워드 입력")
        self.le_keyword = QtWidgets.QLineEdit()
        self.le_keyword.setPlaceholderText("예: 청년 전세자금 대출 정리")

        self.btn_keyword_blog = QtWidgets.QPushButton("키워드 블로그 생성")

        keyword_row.addWidget(lbl_keyword)
        keyword_row.addWidget(self.le_keyword, 1)
        keyword_row.addWidget(self.btn_keyword_blog)

        root.addLayout(keyword_row)

        # ── 중간: 프롬프트 편집 영역 (카테고리 따라 my_topic / my_issue 세트 바뀜)
        gb = QtWidgets.QGroupBox("프롬프트 / 시스템 / 사용자 조건")
        form = QtWidgets.QFormLayout(gb)

        self.te_prompt = QtWidgets.QPlainTextEdit()
        self.te_system = QtWidgets.QPlainTextEdit()
        self.te_user = QtWidgets.QPlainTextEdit()

        # 보기 좋게 높이만 좀 제한
        for te in (self.te_prompt, self.te_system, self.te_user):
            te.setFixedHeight(80)

        form.addRow("본문 프롬프트", self.te_prompt)
        form.addRow("시스템 프롬프트", self.te_system)
        form.addRow("사용자 조건", self.te_user)

        self.btn_save_prompt = QtWidgets.QPushButton("프롬프트 저장")
        form.addRow(self.btn_save_prompt)

        root.addWidget(gb)

        # ── 아래: 로그
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        root.addWidget(self.log, 1)

    def _wire(self):
        self.combo_category.currentTextChanged.connect(self.on_category_changed)
        self.btn_suggest.clicked.connect(self.on_suggest)
        self.btn_issue.clicked.connect(self.on_issue)
        self.btn_save_prompt.clicked.connect(self._save_current_prompt)
        self.btn_keyword_blog.clicked.connect(self.on_keyword_blog)

    # ─────────────────────
    # 프롬프트 로드/저장 관련
    # ─────────────────────
    def _load_prompt_for_current_category(self):
        """현재 콤보에 선택된 카테고리에 맞는 txt 3개를 읽어와서 에디터에 채워 넣는다."""
        cat = self.combo_category.currentText().strip()
        # 규칙: 이름에 '이슈'가 들어가면 이슈용 파일 세트
        is_issue = "이슈" in cat

        if is_issue:
            path_main = v_.file_path_issue
            path_sys = v_.file_path_issue_system
            path_user = v_.file_path_issue_user
        else:
            path_main = v_.file_path_topic
            path_sys = v_.file_path_topic_system
            path_user = v_.file_path_topic_user

        def _read(p):
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8-sig") as f:
                    return f.read()
            return ""

        self._current_paths = {
            "main": path_main,
            "system": path_sys,
            "user": path_user,
        }

        self.te_prompt.setPlainText(_read(path_main))
        self.te_system.setPlainText(_read(path_sys))
        self.te_user.setPlainText(_read(path_user))

        self.log.appendPlainText(f"📄 '{cat}' 프롬프트 로드 완료")

    def _save_current_prompt(self):
        """현재 보이는 3개 텍스트를 각자 파일에 덮어쓴다."""
        if not hasattr(self, "_current_paths"):
            return
        paths = self._current_paths

        def _write(p, txt):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8-sig") as f:
                f.write(txt)

        _write(paths["main"], self.te_prompt.toPlainText())
        _write(paths["system"], self.te_system.toPlainText())
        _write(paths["user"], self.te_user.toPlainText())

        self.log.appendPlainText("💾 프롬프트 3종 저장 완료")

    # ─────────────────────
    # 카테고리 변경
    # ─────────────────────
    def on_category_changed(self, text: str):
        text = text.strip()
        if not text or "카테고리 없음" in text:
            return
        # .env 수정
        save_env_blog_category(text)
        # variable.py 의 my_category도 갱신
        try:
            v_.my_category = text
        except Exception:
            pass

        # 카테고리 바뀌면 대응되는 프롬프트도 다시 읽어온다
        self._load_prompt_for_current_category()

        self.log.appendPlainText(f"🔁 BLOG_CATEGORY 갱신: {text}")

    # ─────────────────────
    # 생활정보 자동 주제
    # ─────────────────────
    def on_suggest(self):
        minutes = int(self.sb_repeat.value())
        interval_ms = minutes * 60 * 1000

        self.timer_suggest.stop()
        self.timer_suggest.start(interval_ms)
        self.log.appendPlainText(f"🟢 생활정보 자동 실행 타이머 시작: {minutes}분 간격")

        self.btn_suggest.setEnabled(False)
        self._run_suggest_once()

    def _run_suggest_once(self):
        selected_cat = self.combo_category.currentText().strip()
        self.log.appendPlainText(f"⏱ 생활정보 작성 트리거 도착 (카테고리: {selected_cat})")

        if getattr(self, "_suggest_running", False):
            self.log.appendPlainText("↪ 이미 생활정보 작성이 실행 중이라 이번엔 건너뜀")
            return

        self._suggest_running = True

        def job(progress):
            progress({"msg": f"[{selected_cat}] 생활정보 주제 추천 + 작성 중..."})
            ok = suggest_life_tip_topic()
            progress({"msg": f"끝: {ok}"})
            return {"ok": ok, "category": selected_cat}

        def done(ok, payload, err):
            cat = selected_cat
            result_text = None
            if isinstance(payload, dict):
                cat = payload.get("category", selected_cat)
                result_text = payload.get("ok")
            else:
                result_text = payload

            if ok:
                self.log.appendPlainText(
                    f"✅ 생활정보 글 생성 완료 (카테고리: {cat}) → {result_text}"
                )
            else:
                self.log.appendPlainText(f"❌ 생활정보 글 생성 실패: {err}")

            self._suggest_running = False
            self.btn_suggest.setEnabled(True)

        run_job_with_progress_async(
            owner=self,
            title="생활정보 글 자동 생성",
            job=job,
            on_done=done,
        )

    # ─────────────────────
    # 이슈 자동 주제
    # ─────────────────────
    def on_issue(self):
        minutes = int(self.sb_repeat.value())
        interval_ms = minutes * 60 * 1000

        self.timer_issue.stop()
        self.timer_issue.start(interval_ms)
        self.log.appendPlainText(f"🟢 이슈 자동 실행 타이머 시작: {minutes}분 간격")

        self.btn_issue.setEnabled(False)
        self._run_issue_once()

    def _run_issue_once(self):
        selected_cat = self.combo_category.currentText().strip()
        self.log.appendPlainText(f"⏱ 이슈 작성 트리거 도착 (카테고리: {selected_cat})")

        if getattr(self, "_issue_running", False):
            self.log.appendPlainText("↪ 이미 이슈 작성이 실행 중이라 이번엔 건너뜀")
            return

        self._issue_running = True

        def job(progress):
            progress({"msg": f"[{selected_cat}] 이슈 주제 수집/필터링 중..."})
            ret = issue_start()
            progress({"msg": f"끝: {ret}"})
            return {"result": ret, "category": selected_cat}

        def done(ok, payload, err):
            cat = selected_cat
            result_text = None
            if isinstance(payload, dict):
                cat = payload.get("category", selected_cat)
                result_text = payload.get("result")
            else:
                result_text = payload

            if ok:
                self.log.appendPlainText(
                    f"✅ 이슈 글 생성 완료 (카테고리: {cat}) → {result_text}"
                )
            else:
                self.log.appendPlainText(f"❌ 이슈 글 생성 실패: {err}")

            self._issue_running = False
            self.btn_issue.setEnabled(True)

        run_job_with_progress_async(
            owner=self,
            title="이슈 글 자동 생성",
            job=job,
            on_done=done,
        )

    # ─────────────────────
    # 키워드 1회 블로그 생성
    # ─────────────────────
    def on_keyword_blog(self):
        keyword = self.le_keyword.text().strip()
        if not keyword:
            self.log.appendPlainText("⚠ 키워드를 입력해 주세요.")
            return

        if getattr(self, "_keyword_running", False):
            self.log.appendPlainText("↪ 키워드 글 생성이 이미 실행 중입니다.")
            return

        self._keyword_running = True

        selected_cat = self.combo_category.currentText().strip()
        self.log.appendPlainText(
            f"⏱ 키워드 글 생성 트리거 도착 (카테고리: {selected_cat}, 키워드: {keyword})"
        )

        def job(progress):
            progress({"msg": f"[{selected_cat}] 키워드 '{keyword}'로 블로그 글 생성 중..."})
            ret = life_tips_keyword(keyword)
            progress({"msg": f"끝: {ret}"})
            return {"result": ret, "keyword": keyword, "category": selected_cat}

        def done(ok, payload, err):
            kw = keyword
            cat = selected_cat
            result_text = None

            if isinstance(payload, dict):
                kw = payload.get("keyword", keyword)
                cat = payload.get("category", selected_cat)
                result_text = payload.get("result")
            else:
                result_text = payload

            if ok:
                self.log.appendPlainText(
                    f"✅ 키워드 블로그 생성 완료 (카테고리: {cat}, 키워드: {kw}) → {result_text}"
                )
            else:
                self.log.appendPlainText(
                    f"❌ 키워드 블로그 생성 실패 (키워드: {kw}): {err}"
                )

            self._keyword_running = False

        run_job_with_progress_async(
            owner=self,
            title="키워드 블로그 생성",
            job=job,
            on_done=done,
        )


    def _heartbeat(self):
        msg_parts = []
        if self.timer_suggest.isActive():
            msg_parts.append("생활정보 반복 대기중")
        if self.timer_issue.isActive():
            msg_parts.append("이슈 반복 대기중")

        if not msg_parts:
            msg = "⏸ 반복 타이머 꺼짐"
        else:
            msg = " / ".join(msg_parts)

        if self._suggest_running or self._issue_running:
            msg += " (현재 작업 실행 중...)"

        self.log.appendPlainText(f"[대기] {msg}")


# ─────────────────────────────
# 설정 탭 (기존 그대로)
# ─────────────────────────────
class BlogSettingsWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        form = QtWidgets.QFormLayout(self)

        comfy_default = os.getenv("COMFY_HOST", getattr(s_mod, "COMFY_HOST", "http://127.0.0.1:8188"))
        self.le_comfy = QtWidgets.QLineEdit(comfy_default)
        form.addRow("COMFY_HOST", self.le_comfy)

        self.le_wp_domain = QtWidgets.QLineEdit(
            os.getenv("WP_DOMAIN", getattr(s_mod, "WP_DOMAIN", "")))
        self.le_wp_id = QtWidgets.QLineEdit(
            os.getenv("WP_ID", getattr(s_mod, "WP_ID", "")))
        self.le_wp_pw = QtWidgets.QLineEdit(
            os.getenv("WP_PW", getattr(s_mod, "WP_PW", "")))
        self.le_wp_pw.setEchoMode(QtWidgets.QLineEdit.Password)
        form.addRow("WP_DOMAIN", self.le_wp_domain)
        form.addRow("WP_ID", self.le_wp_id)
        form.addRow("WP_PW", self.le_wp_pw)

        self.le_fb_thumb = QtWidgets.QLineEdit(
            os.getenv("FALLBACK_THUMB_PATH", getattr(s_mod, "FALLBACK_THUMB_PATH", "")))
        self.le_fb_scene = QtWidgets.QLineEdit(
            os.getenv("FALLBACK_SCENE_PATH", getattr(s_mod, "FALLBACK_SCENE_PATH", "")))
        form.addRow("FALLBACK_THUMB_PATH", self.le_fb_thumb)
        form.addRow("FALLBACK_SCENE_PATH", self.le_fb_scene)

        self.te_categories = QtWidgets.QPlainTextEdit()
        existing = load_category_lines()
        self.te_categories.setPlainText("\n".join(existing))
        form.addRow("카테고리 목록 (한 줄에 하나)", self.te_categories)

        self.btn_save = QtWidgets.QPushButton("저장(오버라이드 + 카테고리 txt)")
        form.addRow(self.btn_save)

        self.btn_save.clicked.connect(self._save)

    def _save(self):
        try:
            path = s_mod.save_overrides(
                COMFY_HOST=self.le_comfy.text().strip(),
                WP_DOMAIN=self.le_wp_domain.text().strip(),
                WP_ID=self.le_wp_id.text().strip(),
                WP_PW=self.le_wp_pw.text().strip(),
                FALLBACK_THUMB_PATH=self.le_fb_thumb.text().strip(),
                FALLBACK_SCENE_PATH=self.le_fb_scene.text().strip(),
            )

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
