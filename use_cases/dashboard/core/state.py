"""State management layer for the dashboard."""

from __future__ import annotations

import sys
from pathlib import Path
import os
from typing import Optional

import pandas as pd
from matplotlib import colormaps

# Ensure repository imports work when running without installation.
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
# Local package imports rely on namespace packages; keep repo root on path.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from use_cases.dashboard.core.state_manager import AppStateManager  # noqa: E402
from use_cases.dashboard.core.state_models import AppState  # noqa: E402

# Old global paths removed - now model-specific via ModelManager
COOLWARM_CMAP = colormaps["coolwarm"]

# Phase 3: Application state manager (replaces all global state)
state_manager = AppStateManager()


# Helper functions that delegate to state_manager
def _append_status_message(message: str) -> None:
    """Append status message to active model."""
    state_manager.append_status_message(message)


def _update_history_with_epoch(payload: dict[str, float]) -> None:
    """Update training history with epoch metrics."""
    state_manager.update_history_with_epoch(payload)


def _clear_metrics_queue() -> None:
    """Clear all pending metrics from queue."""
    state_manager.clear_metrics_queue()


def initialize_app_state() -> None:
    """Initialize app state with model registry."""
    state_manager.initialize()


def initialize_model_and_data() -> None:
    """DEPRECATED: Use load_model(model_id) instead."""
    state_manager.initialize()


def load_model(model_id: str) -> None:
    """Load a specific model as active."""
    state_manager.load_model(model_id)


def _load_labels_dataframe() -> pd.DataFrame:
    """Load labels CSV for ACTIVE model.

    NOTE: This is deprecated - use LabelingService instead.
    """
    app_state = state_manager.state
    if app_state is None or app_state.active_model is None:
        empty = pd.DataFrame(columns=["label"])
        empty.index = pd.Index([], name="Serial", dtype=int)
        empty["label"] = pd.Series(dtype="Int64")
        return empty

    from use_cases.dashboard.core.model_manager import ModelManager
    labels_path = ModelManager.labels_path(app_state.active_model.model_id)

    columns = ["Serial", "label"]
    if labels_path.exists():
        df = pd.read_csv(labels_path, usecols=columns)
    else:
        df = pd.DataFrame(columns=columns)

    if df.empty:
        empty = pd.DataFrame(columns=["label"])
        empty.index = pd.Index([], name="Serial", dtype=int)
        empty["label"] = pd.Series(dtype="Int64")
        return empty

    df["Serial"] = pd.to_numeric(df["Serial"], errors="coerce")
    df = df.dropna(subset=["Serial"])
    df["Serial"] = df["Serial"].astype(int)
    df["label"] = pd.to_numeric(df.get("label"), errors="coerce").astype("Int64")
    df = df.set_index("Serial")
    df.index.name = "Serial"
    return df


def _persist_labels_dataframe(df: pd.DataFrame) -> None:
    """Save labels CSV for ACTIVE model.

    NOTE: This is deprecated - use LabelingService instead.
    """
    app_state = state_manager.state
    if app_state is None or app_state.active_model is None:
        return

    from use_cases.dashboard.core.model_manager import ModelManager
    labels_path = ModelManager.labels_path(app_state.active_model.model_id)

    persisted = df.copy()
    if not persisted.empty:
        persisted.index = persisted.index.astype(int)
        persisted["label"] = persisted["label"].astype("Int64")
    persisted.index.name = "Serial"
    persisted.to_csv(labels_path)
