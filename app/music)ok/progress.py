# app/progress.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import io, traceback
import os
from typing import Callable, Optional, Any, Dict, Union
from os import PathLike
from PyQt5 import QtWidgets, QtCore
import threading

def _qt_get(*names):
    """
    예: _qt_get("Qt","NonModal")  # PyQt5
        _qt_get("Qt","WindowModality","NonModal")  # PyQt6
    존재하면 그 값을, 없으면 None을 반환
    """
    obj = QtCore
    for n in names:
        obj = getattr(obj, n, None)
        if obj is None:
            return None
    return obj

flags = getattr(QtCore.QEventLoop, "AllEvents", None)
if flags is None:
    pef = getattr(QtCore.QEventLoop, "ProcessEventsFlag", None)
    flags = getattr(pef, "AllEvents", None) if pef else None

# 호환 상수
WINDOW_NONMODAL = _qt_get("Qt", "NonModal") or _qt_get("Qt", "WindowModality", "NonModal")
PEF_ALL_EVENTS = _qt_get("QEventLoop", "AllEvents") or _qt_get("QEventLoop", "ProcessEventsFlag", "AllEvents")
DOCK_BOTTOM = _qt_get("Qt", "BottomDockWidgetArea") or _qt_get("Qt", "DockWidgetArea", "BottomDockWidgetArea")

TailT = Optional[Union[str, PathLike[str]]]

class ProgressLogDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, title="작업 진행"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self.setMinimumWidth(520)
        self._completed = False

        self.lbl = QtWidgets.QLabel("준비 중…")
        self.bar = QtWidgets.QProgressBar(); self.bar.setRange(0, 0)
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(200)
        self.btn = QtWidgets.QPushButton("숨기기")
        self.btn.clicked.connect(self._on_btn)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.lbl); lay.addWidget(self.bar); lay.addWidget(self.log)
        row = QtWidgets.QHBoxLayout(); row.addStretch(1); row.addWidget(self.btn); lay.addLayout(row)

    def _on_btn(self):
        if self._completed: self.accept()
        else: self.hide()

    def set_status(self, text: str):
        self.lbl.setText(text or "")

    def append_log(self, text: str):
        if not text: return
        self.log.appendPlainText(text.rstrip("\n"))
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def enter_determinate(self, total: int):
        self._completed = False
        total = max(0, int(total))
        self.bar.setRange(0, total); self.bar.setValue(0)
        self.setWindowTitle("진행 중")
        self.set_status(f"0 / {total}")

    def step(self, msg: str | None = None):
        if msg: self.append_log(msg)
        v = min(self.bar.value() + 1, self.bar.maximum())
        self.bar.setValue(v)
        if self.bar.maximum() > 0:
            self.set_status(f"{v} / {self.bar.maximum()}")

    def set_completed(self, text: str = "완료"):
        self._completed = True
        self.bar.setRange(0, 1); self.bar.setValue(1)
        self.setWindowTitle("완료")
        self.set_status(text)
        self.btn.setText("닫기")

    def finish(self, ok: bool, err_text: str | None) -> None:
        # 진행바 완료 상태
        try:
            self.bar.setRange(0, 1)
            self.bar.setValue(1)
        except Exception:
            pass

        # 타이틀/로그
        try:
            base = self.windowTitle()
            self.setWindowTitle(f"{base} — {'완료' if ok else '오류'}")
        except Exception:
            pass

        try:
            self.append_log("[done] 완료" if ok else f"[error] {err_text or '오류'}")
        except Exception:
            pass

        # '닫기' 버튼 확보 (기존 버튼 있으면 재활용)
        btn = None
        for name in ("btn_close", "btnCancel", "btn_cancel", "btnClose", "btn_ok", "closeButton"):
            b = getattr(self, name, None)
            if isinstance(b, QtWidgets.QPushButton):
                btn = b
                break

        # 버튼이 없으면 생성해서 우측 정렬로 레이아웃에 추가
        if btn is None:
            btn = QtWidgets.QPushButton("닫기", self)
            lay = self.layout()
            if lay is not None:
                hb = QtWidgets.QHBoxLayout()
                hb.addStretch(1)  # 왼쪽 공간으로 밀어 우측 정렬 효과
                hb.addWidget(btn)

                if isinstance(lay, (QtWidgets.QVBoxLayout, QtWidgets.QHBoxLayout, QtWidgets.QFormLayout)):
                    lay.addLayout(hb)
                elif isinstance(lay, QtWidgets.QGridLayout):
                    row = lay.rowCount()
                    cols = max(1, lay.columnCount())
                    lay.addLayout(hb, row, 0, 1, cols)
                else:
                    # 레이아웃 타입을 모를 때 안전판
                    cont = QtWidgets.QWidget(self)
                    cont.setLayout(hb)
                    try:
                        lay.addWidget(cont)
                    except Exception:
                        v = QtWidgets.QVBoxLayout(self)
                        v.addWidget(cont)
                        self.setLayout(v)

        # 버튼을 '닫기' 동작으로 전환
        try:
            btn.clicked.disconnect()
        except Exception:
            pass
        btn.setText("닫기")
        btn.setEnabled(True)
        btn.clicked.connect(self.accept)

        # ESC로도 닫히게(보강)
        try:
            self.setResult(0)
        except Exception:
            pass


class JobWorker(QtCore.QObject):
    """백그라운드에서 실제 작업을 실행하고, 진행 상황을 신호로 넘김"""
    progress = QtCore.pyqtSignal(dict)
    finished = QtCore.pyqtSignal(object, object)  # (result, err)

    def __init__(self, job_callable):
        super().__init__()
        self._job = job_callable

    @QtCore.pyqtSlot()
    def run(self):
        try:
            def progress_cb(info: dict):
                # 항상 dict 복사 후 emit (예상치 못한 뮤터블 공유 방지)
                try:
                    self.progress.emit(dict(info or {}))
                except Exception as abc:
                    print("[progress] emit error:", abc, flush=True)

            res = self._job(progress_cb)
            self.finished.emit(res, None)
        except Exception as e:
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            self.finished.emit(None, f"{e}\n{tb}")


class _Tailer(QtCore.QObject):
    """선택: 텍스트 로그 파일 tail. UI 스레드에서 QTimer로만 동작."""
    tick = QtCore.pyqtSignal(str)

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self._path = Path(path) if path else None
        self._fp = None
        self._pos = 0
        self._tm = QtCore.QTimer(self)
        self._tm.setInterval(500)
        self._tm.timeout.connect(self._poll)

    def start(self):
        if not (self._path and self._path.exists()):
            return
        try:
            self._fp = open(self._path, "r", encoding="utf-8", errors="ignore")
            self._fp.seek(0, io.SEEK_END)
            self._pos = self._fp.tell()
            self._tm.start()
            self.tick.emit(f"[tail] {self._path}")
        except Exception:
            pass

    def stop(self):
        try:
            self._tm.stop()
        except Exception:
            pass
        try:
            if self._fp:
                self._fp.close()
                self._fp = None
        except Exception:
            pass

    def _poll(self):
        try:
            if not self._fp:
                return
            self._fp.seek(self._pos)
            chunk = self._fp.read()
            self._pos = self._fp.tell()
            if chunk:
                self.tick.emit(chunk)
        except Exception:
            pass

# (여기에 ProgressLogDialog 클래스가 먼저 옴)

def _mk_progress(owner: QtWidgets.QWidget, title: str, tail_file: TailT = None):
    tail_str = os.fspath(tail_file) if tail_file is not None else None

    dlg = ProgressLogDialog(parent=owner, title=title)
    try:
        flag = getattr(QtCore.Qt, "WindowStaysOnTopHint", None)
        if flag is None:
            flag = getattr(getattr(QtCore.Qt, "WindowType"), "WindowStaysOnTopHint")
        dlg.setWindowFlag(flag, False)
    except Exception:
        pass

    dlg.resize(780, 420)
    dlg.show()
    QtWidgets.QApplication.processEvents()
    try:
        dlg.raise_()
        dlg.activateWindow()
    except Exception:
        pass

    def _post_ui(fn) -> None:
        QtCore.QTimer.singleShot(0, fn)

    def on_progress(info: Dict[str, Any]) -> None:
        stage = str(info.get("stage") or "info")
        msg = str(info.get("msg") or "")
        step = info.get("step")

        if msg:
            _post_ui(lambda: dlg.append_log(f"[{stage}] {msg}"))
            if hasattr(owner, "log_line") and callable(getattr(owner, "log_line")):
                _post_ui(lambda: owner.log_line(stage, msg))

        if isinstance(step, int) and step > 0:
            def _advance():
                if dlg.bar.maximum() == 0:
                    dlg.enter_determinate(step)
                dlg.step(msg or "")
            _post_ui(_advance)

    def finalize(ok: bool, _payload: Any = None, err: Optional[BaseException] = None) -> None:
        _post_ui(lambda: dlg.finish(ok, (str(err) if err else None)))

    # (선택) 실시간 로그 테일링 — 이미 외부에서 tail_file을 주면 활성화
    if tail_str and os.path.exists(tail_str):
        try:
            fp = open(tail_str, "r", encoding="utf-8", errors="ignore")
            fp.seek(0, os.SEEK_END)
            timer = QtCore.QTimer(dlg)
            timer.setInterval(500)

            def _poll():
                try:
                    line = fp.readline()
                    flushed = False
                    while line:
                        dlg.append_log(line.rstrip("\n"))
                        flushed = True
                        line = fp.readline()
                    if flushed:
                        dlg.ensure_log_visible()
                except Exception:
                    pass

            timer.timeout.connect(_poll)
            timer.start()

            def _stop():
                try:
                    timer.stop()
                except Exception:
                    pass
                try:
                    fp.close()
                except Exception:
                    pass

            try:
                dlg.finished.connect(lambda _=None: _stop())
            except Exception:
                pass
        except Exception:
            pass

    return on_progress, finalize, dlg



def run_job_with_progress(
    owner: QtWidgets.QWidget,
    title: str,
    job: Callable[[Callable[[Dict[str, Any]], None]], Any],
    *,
    tail_file: TailT = None,
    on_done: Optional[Callable[[bool, Any, Optional[BaseException]], None]] = None,
) -> None:
    """동기 실행(간단 작업용)"""
    on_progress, finalize, _dlg = _mk_progress(owner, title, tail_file=tail_file)
    ok = True
    payload: Any = None
    err: Optional[BaseException] = None
    try:
        payload = job(on_progress)
    except BaseException as exc:
        ok = False
        err = exc
    finally:
        finalize(ok, payload, err)
        if on_done:
            try:
                on_done(ok, payload, err)
            except Exception:
                pass

def connect_worker_with_progress(
    owner: QtWidgets.QWidget,
    title: str,
    *,
    worker: QtCore.QObject,
    signal_name: str = "progress",  # 워커의 진행 시그널 이름(인자: dict)
    finished_signal: str = "finished",  # 종료 시그널 이름
    error_signal: Optional[str] = None,  # 에러 시그널 이름(인자: Exception 또는 str)
    tail_file: Optional[str] = None,
    on_done: Optional[Callable[[bool, Any, Optional[BaseException]], None]] = None,
) -> None:
    """
    스레드/워커 기반 작업을 진행창에 붙이고 싶은 경우 사용.
    - worker.progress(dict) → 진행 로그
    - worker.finished() → ok=True 종료 처리
    - worker.error(Exception|str) → ok=False 종료 처리
    """
    on_progress, finalize, _dlg = _mk_progress(owner, title, tail_file=tail_file)

    # progress(dict) 연결
    try:
        getattr(worker, signal_name).connect(lambda info: on_progress(info if isinstance(info, dict) else {"msg": str(info)}))  # type: ignore
    except Exception:
        pass

    def _done(ok: bool, payload: Any = None, exc: Optional[BaseException] = None):
        finalize(ok, payload, exc)
        if on_done:
            try:
                on_done(ok, payload, exc)
            except Exception:
                pass

    # finished()
    try:
        getattr(worker, finished_signal).connect(lambda: _done(True, None, None))  # type: ignore
    except Exception:
        pass

    # error(...)
    if error_signal:
        try:
            def _on_error(e: Any):
                _done(False, None, e if isinstance(e, BaseException) else RuntimeError(str(e)))
            getattr(worker, error_signal).connect(_on_error)  # type: ignore
        except Exception:
            pass

# ───────── 비동기(스레드) 실행기 추가 ─────────
class _JobWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(dict)
    finished = QtCore.pyqtSignal(object, object)  # payload, err
    def __init__(self, job):
        super().__init__()
        self._job = job
    @QtCore.pyqtSlot()
    def run(self):
        try:
            payload = self._job(self._emit_progress)
            self.finished.emit(payload, None)
        except BaseException as exc:
            self.finished.emit(None, exc)
    def _emit_progress(self, info: Dict[str, Any]) -> None:
        if not isinstance(info, dict):
            info = {"msg": str(info)}
        self.progress.emit(info)

# progress.py (하단에 추가 또는 기존 함수 교체)

# progress.py  ── 전체 교체
def run_job_with_progress_async(
    owner: QtWidgets.QWidget,
    title: str,
    job,  # Callable[[Callable[[dict], None]], Any]
    *,
    tail_file=None,
    on_done=None,  # Callable[[bool, Any, Optional[BaseException]], None]
) -> None:
    # 진행창
    dlg = ProgressLogDialog(parent=owner)
    try:
        dlg.set_title(title)
    except Exception:
        dlg.setWindowTitle(title)
    dlg.show()
    QtWidgets.QApplication.processEvents()

    # 항상 한 줄은 보장
    try:
        dlg.append_log("[ui] 작업 시작 준비")
    except Exception:
        pass

    class _JobObj(QtCore.QObject):
        progress = QtCore.pyqtSignal(dict)
        finished = QtCore.pyqtSignal(object, object)  # (payload, err)

        @QtCore.pyqtSlot()
        def run(self):
            payload = None
            err = None
            try:
                def on_progress(info: dict):
                    if not isinstance(info, dict):
                        info = {"msg": str(info)}
                    self.progress.emit(info)
                payload = job(on_progress)
            except BaseException as e:
                err = e
            finally:
                self.finished.emit(payload, err)

    obj = _JobObj()
    th = QtCore.QThread(owner)
    obj.moveToThread(th)

    def _on_progress(info: dict):
        msg = str(info.get("msg") or "")
        step = info.get("step")
        if msg:
            try:
                dlg.append_log(msg)
            except Exception:
                pass
        if isinstance(step, int) and step > 0:
            try:
                if dlg.bar.maximum() == 0:
                    dlg.enter_determinate(step)
                dlg.step(msg or "")
            except Exception:
                pass

    def _on_finished(payload, err):
        ok = (err is None)
        try:
            dlg.finish(ok, str(err) if err else None)
        except Exception:
            pass
        if callable(on_done):
            try:
                on_done(ok, payload, err)
            except Exception:
                pass
        # 스레드 정리
        th.quit()
        th.wait(100)
        # owner 보관 목록에서 제거
        try:
            bag = getattr(owner, "_progress_jobs", [])
            if th in bag:
                bag.remove(th)
            setattr(owner, "_progress_jobs", bag)
        except Exception:
            pass

    obj.progress.connect(_on_progress)
    obj.finished.connect(_on_finished)
    th.started.connect(obj.run)

    # ★★★ 핵심: 스레드/워커를 owner에 붙여서 GC 방지
    bag = getattr(owner, "_progress_jobs", None)
    if not isinstance(bag, list):
        bag = []
    bag.append(th)
    setattr(owner, "_progress_jobs", bag)
    setattr(th, "_worker_ref", obj)  # 스레드 객체에도 워커 참조를 잡아둠

    try:
        th.start()
        try:
            dlg.append_log("[ui] 백그라운드 스레드 시작")
        except Exception:
            pass
    except BaseException as e:
        try:
            dlg.append_log(f"[error] thread start failed: {e}")
            dlg.finish(False, str(e))
        except Exception:
            pass
        if callable(on_done):
            on_done(False, None, e)







# ───────── 백업 ProgressLogDialog (없을 때만) ─────────
class _FallbackProgressDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, title: str = "진행", tail_file: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(720, 360)
        self.text = QtWidgets.QPlainTextEdit(self)
        self.text.setReadOnly(True)
        self.bar = QtWidgets.QProgressBar(self)
        self.bar.setRange(0, 0)  # indeterminate
        btn = QtWidgets.QPushButton("닫기", self)
        btn.clicked.connect(self.accept)
        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(self.text)
        v.addWidget(self.bar)
        v.addWidget(btn)
        self._tail_file = tail_file

    def append_log(self, line: str) -> None:
        self.text.appendPlainText(line)
        sb = self.text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def enter_determinate(self, steps: int) -> None:
        self.bar.setRange(0, max(steps, 1))
        self.bar.setValue(0)

    def step(self, _label: str = "") -> None:
        self.bar.setValue(self.bar.value() + 1)

    def finish(self, ok: bool, err_text: str | None) -> None:
        self.append_log("[done] 완료" if ok else f"[error] {err_text or '오류'}")
        try:
            self.bar.setRange(0, 1)
            self.bar.setValue(1)
        except Exception:
            pass
        # 버튼이 '닫기'로 보이도록(이미 닫기면 스킵)
        try:
            self.findChild(QtWidgets.QPushButton).setText("닫기")
        except Exception:
            pass
        self.close()  # 바로 닫고 싶다면 유지, 남겨두려면 이 줄을 제거

