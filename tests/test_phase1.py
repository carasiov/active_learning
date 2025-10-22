from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "use_cases" / "dashboard" / "app.py"
README_PATH = PROJECT_ROOT / "use_cases" / "dashboard" / "README.md"


def test_dashboard_files_exist() -> None:
    assert APP_PATH.exists(), f"Expected dashboard app at {APP_PATH}"
    assert README_PATH.exists(), f"Expected dashboard README at {README_PATH}"


@pytest.mark.slow
def test_dashboard_module_imports() -> None:
    spec = importlib.util.spec_from_file_location("dashboard_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    assert hasattr(module, "app")


@pytest.mark.slow
def test_dashboard_process_starts_and_runs_briefly() -> None:
    env = os.environ.copy()
    process = subprocess.Popen(
        [sys.executable, str(APP_PATH)],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(10)
        assert process.poll() is None, "Dashboard process exited prematurely"
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
