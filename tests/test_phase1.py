from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path
from queue import Queue

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "use_cases" / "dashboard" / "app.py"
README_PATH = PROJECT_ROOT / "use_cases" / "dashboard" / "README.md"
CALLBACK_PATH = PROJECT_ROOT / "use_cases" / "dashboard" / "dashboard_callback.py"


def test_dashboard_files_exist() -> None:
    assert APP_PATH.exists(), f"Expected dashboard app at {APP_PATH}"
    assert README_PATH.exists(), f"Expected dashboard README at {README_PATH}"


@pytest.fixture(scope="session")
def dashboard_module():
    spec = importlib.util.spec_from_file_location("dashboard_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _collect_ids(component) -> set[str]:
    collected: set[str] = set()
    if component is None:
        return collected
    if isinstance(component, (list, tuple)):
        for item in component:
            collected |= _collect_ids(item)
        return collected
    identifier = getattr(component, "id", None)
    if identifier is not None:
        collected.add(str(identifier))
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            collected |= _collect_ids(child)
    elif children is not None:
        collected |= _collect_ids(children)
    return collected


@pytest.mark.slow
def test_dashboard_module_imports(dashboard_module) -> None:
    module = dashboard_module
    assert hasattr(module, "app")


def test_training_controls_present(dashboard_module) -> None:
    ids = _collect_ids(dashboard_module.app.layout)
    expected_ids = {
        "start-training-button",
        "recon-weight-slider",
        "kl-weight-slider",
        "learning-rate-slider",
        "num-epochs-input",
        "training-status",
        "training-poll",
    }
    for identifier in expected_ids:
        assert identifier in ids, f"Expected component id '{identifier}' in dashboard layout"


def test_dashboard_metrics_callback_emits_payload() -> None:
    spec = importlib.util.spec_from_file_location("dashboard_callback", CALLBACK_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    DashboardMetricsCallback = getattr(module, "DashboardMetricsCallback")
    queue: Queue = Queue()
    callback = DashboardMetricsCallback(queue, target_epochs=5)
    metrics = {
        "train": {
            "loss": 1.23,
            "reconstruction_loss": 0.45,
            "kl_loss": 0.12,
            "classification_loss": 0.34,
        },
        "val": {
            "loss": 1.56,
            "reconstruction_loss": 0.52,
            "kl_loss": 0.18,
            "classification_loss": 0.40,
        },
    }
    callback.on_epoch_end(0, metrics, history={}, trainer=None)
    payload = queue.get_nowait()
    assert payload["type"] == "epoch_complete"
    assert payload["epoch"] == 1
    assert payload["target_epochs"] == 5
    assert pytest.approx(payload["train_loss"], rel=1e-6) == 1.23
    assert pytest.approx(payload["val_loss"], rel=1e-6) == 1.56
    callback.on_train_end(history={}, trainer=None)
    completion = queue.get_nowait()
    assert completion["type"] == "training_complete"


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
