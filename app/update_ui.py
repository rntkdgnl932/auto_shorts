# update_ui.py
import os
import subprocess
from PyQt5 import QtWidgets, QtCore


class GitUpdateThread(QtCore.QThread):
    """Git 명령어를 백그라운드에서 실행하는 스레드"""
    log_signal = QtCore.pyqtSignal(str)
    finished_signal = QtCore.pyqtSignal()

    def run(self):
        try:
            self.log_signal.emit("=== 업데이트 시작 (Git Pull) ===")

            # 1. 작업 디렉토리 확인 (현재 프로젝트 루트)
            cwd = os.getcwd()
            self.log_signal.emit(f"작업 경로: {cwd}")

            # 2. git 명령어 실행 함수
            def run_git_cmd(args):
                # 윈도우 콘솔 인코딩 대응 (cp949)
                process = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=cwd,
                    shell=True
                )
                stdout, stderr = process.communicate()

                try:
                    out_str = stdout.decode('cp949', errors='ignore')
                except:
                    out_str = stdout.decode('utf-8', errors='ignore')

                try:
                    err_str = stderr.decode('cp949', errors='ignore')
                except:
                    err_str = stderr.decode('utf-8', errors='ignore')

                return out_str, err_str

            # 3. git pull 실행
            # (혹시 충돌 방지를 위해 강제 리셋이 필요하면 아래 주석을 해제해서 사용하세요)
            # run_git_cmd(["git", "fetch", "--all"])
            # run_git_cmd(["git", "reset", "--hard", "origin/main"])

            self.log_signal.emit("> git pull origin main...")
            out, err = run_git_cmd(["git", "pull", "origin", "main"])

            if out:
                self.log_signal.emit(f"[Output]\n{out.strip()}")
            if err:
                self.log_signal.emit(f"[Error]\n{err.strip()}")

            if "Already up to date" in out:
                self.log_signal.emit("\n>> 이미 최신 버전입니다.")
            elif "Updating" in out:
                self.log_signal.emit("\n>> 업데이트가 완료되었습니다. 프로그램을 재시작해주세요.")
            else:
                self.log_signal.emit("\n>> 작업이 완료되었습니다.")

        except Exception as e:
            self.log_signal.emit(f"\n[치명적 오류 발생]: {str(e)}")
        finally:
            self.log_signal.emit("=== 종료 ===")
            self.finished_signal.emit()


class UpdateWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # [수정] 여기에 변수를 미리 선언해줍니다. (초기값은 None)
        self.thread = None

        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # 설명 라벨
        lbl_info = QtWidgets.QLabel("Git 저장소에서 최신 소스 코드를 받아옵니다.")
        lbl_info.setStyleSheet("font-weight: bold; color: #555;")
        layout.addWidget(lbl_info)

        # 업데이트 버튼
        self.btn_update = QtWidgets.QPushButton("최신 버전으로 업데이트 (Git Pull)")
        self.btn_update.setMinimumHeight(50)
        self.btn_update.setStyleSheet("font-size: 14px; font-weight: bold; background-color: #E6F0FF;")
        self.btn_update.clicked.connect(self.start_update)
        layout.addWidget(self.btn_update)

        # 로그창
        self.txt_log = QtWidgets.QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet("background-color: #222; color: #0F0; font-family: Consolas;")
        layout.addWidget(self.txt_log)

    def start_update(self):
        self.btn_update.setEnabled(False)
        self.txt_log.clear()

        # 여기서 self.thread에 실제 객체를 할당합니다.
        self.thread = GitUpdateThread()
        self.thread.log_signal.connect(self.append_log)
        self.thread.finished_signal.connect(self.on_finished)
        self.thread.start()

    def append_log(self, text):
        self.txt_log.append(text)
        # 스크롤 최하단으로 이동
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def on_finished(self):
        self.btn_update.setEnabled(True)
        QtWidgets.QMessageBox.information(self, "알림", "업데이트 작업이 종료되었습니다.\n로그를 확인해주세요.")


def create_update_widget(parent=None):
    return UpdateWidget(parent)