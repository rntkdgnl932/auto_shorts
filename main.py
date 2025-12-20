# main.py
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path


def _runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def main():
    root = _runtime_root()

    # 1) import 기준을 안정화
    sys.path.insert(0, str(root))

    # ✅ (추가) app 폴더도 import path에 추가
    app_dir = root / "app"
    if app_dir.exists():
        sys.path.insert(0, str(app_dir))

    # 2) CWD를 app으로 고정 (all_ui.py 직접 실행과 동일 조건)
    if app_dir.exists():
        os.chdir(str(app_dir))
    else:
        os.chdir(str(root))

    # 3) 실행
    from app import all_ui
    all_ui.main()


if __name__ == "__main__":
    main()
