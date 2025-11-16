"""State management layer for the dashboard."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
import os
from queue import Empty, Queue
from typing import Dict, Optional, Tuple
from dataclasses import replace

import numpy as np
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

from rcmvae.application.runtime.interactive import InteractiveTrainer  # noqa: E402
from rcmvae.application.model_api import SSVAE  # noqa: E402
from rcmvae.domain.config import SSVAEConfig  # noqa: E402
from data.mnist import load_train_images_for_ssvae, load_mnist_splits  # noqa: E402

from use_cases.dashboard.utils.visualization import (  # noqa: E402
    _build_hover_metadata,
    _format_hover_metadata_entry,
)
from use_cases.dashboard.core.state_models import (  # noqa: E402
    AppState,
    ModelState,
    ModelMetadata,
    DataState,
    TrainingStatus,
    TrainingState,
    UIState,
    TrainingHistory,
)
from use_cases.dashboard.core.commands import CommandDispatcher  # noqa: E402
from use_cases.dashboard.services import ServiceContainer  # noqa: E402


# Old global paths removed - now model-specific via ModelManager
COOLWARM_CMAP = colormaps["coolwarm"]
MAX_STATUS_MESSAGES = 10
PREVIEW_SAMPLE_LIMIT = 2048  # Limit dataset size when loading models for dashboard
FAST_DASHBOARD_MODE = os.environ.get("DASHBOARD_FAST_MODE", "1").lower() not in {"0", "false", "no"}


# Global state lock - use RLock to allow re-entrant locking
# (needed when commands call helper functions that also acquire the lock)
state_lock = threading.RLock()
_init_lock = threading.Lock()  # Separate lock for initialization
metrics_queue: Queue[Dict[str, float]] = Queue()

app_state: Optional[AppState] = None

# Phase 2: Initialize service container with all domain services
services = ServiceContainer.create_default(metrics_queue)

# Global command dispatcher (initialized with state_lock and services)
dispatcher = CommandDispatcher(state_lock, services)


def _append_status_message_locked(message: str) -> None:
    """Append status message (must be called with state_lock held)."""
    global app_state
    if app_state.active_model is None:
        return
    updated_model = replace(
        app_state.active_model,
        training=app_state.active_model.training.with_message(message, max_messages=MAX_STATUS_MESSAGES)
    )
    app_state = app_state.with_active_model(updated_model)


def _append_status_message(message: str) -> None:
    with state_lock:
        _append_status_message_locked(message)


def _update_history_with_epoch(payload: Dict[str, float]) -> None:
    """Update training history with epoch metrics."""
    with state_lock:
        global app_state
        if app_state.active_model is None:
            return
        current_history = app_state.active_model.history
        next_epoch = (current_history.epochs[-1] + 1) if current_history.epochs else 1
        metrics = dict(payload)
        metrics["epoch_absolute"] = float(next_epoch)
        metrics["epoch_in_run"] = float(payload.get("epoch", next_epoch))
        new_history = current_history.with_epoch(next_epoch, metrics)
        updated_model = app_state.active_model.with_history(new_history)
        app_state = app_state.with_active_model(updated_model)


def _clear_metrics_queue() -> None:
    while True:
        try:
            metrics_queue.get_nowait()
        except Empty:
            break


def initialize_app_state() -> None:
    """Initialize app state with model registry."""
    global app_state
    
    if app_state is not None:
        return
    
    with _init_lock:
        if app_state is not None:
            return
        
        from use_cases.dashboard.core.model_manager import ModelManager
        
        # Load all model metadata
        models = ModelManager.list_all_models()
        
        # Create empty registry (no active model)
        with state_lock:
            app_state = AppState(
                models=models,
                active_model=None,
                cache={}
            )


def initialize_model_and_data() -> None:
    """DEPRECATED: Use load_model(model_id) instead. Kept for backward compat."""
    initialize_app_state()


def load_model(model_id: str) -> None:
    """Load a specific model as active."""
    global app_state
    from use_cases.dashboard.core.model_manager import ModelManager
    from use_cases.dashboard.core.model_runs import load_run_records
    
    # Ensure app state initialized
    initialize_app_state()
    
    with state_lock:
        # Don't reload if already active
        if app_state.active_model and app_state.active_model.model_id == model_id:
            return
        
        # Load metadata
        metadata = ModelManager.load_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not found")
        
        # Load persisted config if available
        config = ModelManager.load_config(model_id) or SSVAEConfig()
        
        # Load model
        model = SSVAE(input_dim=(28, 28), config=config)
        checkpoint_path = ModelManager.checkpoint_path(model_id)
        if checkpoint_path.exists():
            model.load_model_weights(str(checkpoint_path))
            model.weights_path = str(checkpoint_path)
        
        trainer = InteractiveTrainer(model)
        
        # Load data
        if FAST_DASHBOARD_MODE:
            preview_n = min(PREVIEW_SAMPLE_LIMIT, 256)
            rng = np.random.default_rng(0)
            x_train = rng.random((preview_n, 28, 28), dtype=np.float32)
            true_labels = np.zeros(preview_n, dtype=np.int32)
        else:
            x_train = load_train_images_for_ssvae(dtype=np.float32)
            (_, true_labels), _ = load_mnist_splits(normalize=True, reshape=False, dtype=np.float32)
            true_labels = np.asarray(true_labels, dtype=np.int32)

            preview_n = min(PREVIEW_SAMPLE_LIMIT, x_train.shape[0])
            x_train = x_train[:preview_n]
            true_labels = true_labels[:preview_n]
        
        # Load history and run manifest
        history = ModelManager.load_history(model_id)
        run_records = load_run_records(model_id)
        
        # Load labels
        labels_array = np.full(shape=(x_train.shape[0],), fill_value=np.nan, dtype=float)
        labels_path = ModelManager.labels_path(model_id)
        if labels_path.exists():
            stored_labels = pd.read_csv(labels_path)
            if not stored_labels.empty and "Serial" in stored_labels.columns:
                stored_labels["Serial"] = pd.to_numeric(stored_labels["Serial"], errors="coerce")
                stored_labels = stored_labels.dropna(subset=["Serial"])
                stored_labels["Serial"] = stored_labels["Serial"].astype(int)
                stored_labels["label"] = pd.to_numeric(stored_labels.get("label"), errors="coerce").astype("Int64")
                serials = stored_labels["Serial"].to_numpy()
                label_values = stored_labels["label"].astype(int).to_numpy()
                valid_mask = (serials >= 0) & (serials < x_train.shape[0])
                labels_array[serials[valid_mask]] = label_values[valid_mask].astype(float)
        
        # Get predictions
        if FAST_DASHBOARD_MODE:
            latent = np.zeros((preview_n, model.config.latent_dim), dtype=np.float32)
            recon = np.zeros_like(x_train)
            pred_classes = np.zeros(preview_n, dtype=np.int32)
            pred_certainty = np.zeros(preview_n, dtype=np.float32)
        else:
            latent, recon, pred_classes, pred_certainty = model.predict(x_train)

        hover_metadata = _build_hover_metadata(pred_classes, pred_certainty, labels_array, true_labels)
        
        # Build ModelState
        data_state = DataState(
            x_train=x_train,
            labels=labels_array,
            true_labels=true_labels,
            latent=latent,
            reconstructed=recon,
            pred_classes=pred_classes,
            pred_certainty=pred_certainty,
            hover_metadata=hover_metadata,
            version=0
        )
        
        training_status = TrainingStatus(
            state=TrainingState.IDLE,
            target_epochs=0,
            status_messages=[],
            thread=None
        )
        
        ui_state = UIState(
            selected_sample=0,
            color_mode="user_labels"
        )
        
        model_state = ModelState(
            model_id=model_id,
            metadata=metadata,
            model=model,
            trainer=trainer,
            config=model.config,
            data=data_state,
            training=training_status,
            ui=ui_state,
            history=history,
            runs=tuple(run_records),
        )
        
        # Update app state
        app_state = app_state.with_active_model(model_state)


def _load_labels_dataframe() -> pd.DataFrame:
    """Load labels CSV for ACTIVE model."""
    if app_state.active_model is None:
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
    """Save labels CSV for ACTIVE model."""
    if app_state.active_model is None:
        return
    
    from use_cases.dashboard.core.model_manager import ModelManager
    labels_path = ModelManager.labels_path(app_state.active_model.model_id)
    
    persisted = df.copy()
    if not persisted.empty:
        persisted.index = persisted.index.astype(int)
        persisted["label"] = persisted["label"].astype("Int64")
    persisted.index.name = "Serial"
    persisted.to_csv(labels_path)
