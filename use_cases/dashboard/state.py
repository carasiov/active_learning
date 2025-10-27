"""State management layer for the dashboard."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
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

from ssvae import SSVAE, SSVAEConfig  # noqa: E402
from training.interactive_trainer import InteractiveTrainer  # noqa: E402
from data.mnist import load_train_images_for_ssvae, load_mnist_splits  # noqa: E402

from use_cases.dashboard.utils import (  # noqa: E402
    _build_hover_metadata,
    _format_hover_metadata_entry,
)
from use_cases.dashboard.state_models import (  # noqa: E402
    AppState,
    DataState,
    TrainingStatus,
    TrainingState,
    UIState,
    TrainingHistory,
)


CHECKPOINT_PATH = ROOT_DIR / "artifacts" / "checkpoints" / "ssvae.ckpt"
LABELS_PATH = ROOT_DIR / "data" / "mnist" / "labels.csv"
COOLWARM_CMAP = colormaps["coolwarm"]
MAX_STATUS_MESSAGES = 10


state_lock = threading.Lock()
_init_lock = threading.Lock()  # Separate lock for initialization
metrics_queue: Queue[Dict[str, float]] = Queue()

app_state: Optional[AppState] = None


def _append_status_message_locked(message: str) -> None:
    """Append status message (must be called with state_lock held)."""
    global app_state
    app_state = replace(
        app_state,
        training=app_state.training.with_message(message, max_messages=MAX_STATUS_MESSAGES)
    )


def _append_status_message(message: str) -> None:
    with state_lock:
        _append_status_message_locked(message)


def _update_history_with_epoch(payload: Dict[str, float]) -> None:
    """Update training history with epoch metrics."""
    with state_lock:
        global app_state
        epoch = int(payload["epoch"])
        new_history = app_state.history.with_epoch(epoch, payload)
        app_state = app_state.with_history(new_history)


def _clear_metrics_queue() -> None:
    while True:
        try:
            metrics_queue.get_nowait()
        except Empty:
            break


def initialize_model_and_data() -> None:
    """Load model, dataset, labels, and derived predictions into memory.
    
    Uses a separate initialization lock to ensure only one thread initializes,
    while other threads wait for completion.
    """
    global app_state
    
    # Quick check without lock - if already initialized, return immediately
    if app_state is not None:
        return
    
    # Use initialization lock to serialize initialization attempts
    with _init_lock:
        # Double-check after acquiring init lock
        if app_state is not None:
            return
        
        # We're the initializing thread - do the work
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

        config = SSVAEConfig()
        model = SSVAE(input_dim=(28, 28), config=config)

        if CHECKPOINT_PATH.exists():
            model.load_model_weights(str(CHECKPOINT_PATH))
            model.weights_path = str(CHECKPOINT_PATH)

        trainer = InteractiveTrainer(model)

        x_train = load_train_images_for_ssvae(dtype=np.float32)
        (_, true_labels), _ = load_mnist_splits(normalize=True, reshape=False, dtype=np.float32)
        true_labels = np.asarray(true_labels, dtype=np.int32)

        latent, recon, pred_classes, pred_certainty = model.predict(x_train)

        labels_array = np.full(shape=(x_train.shape[0],), fill_value=np.nan, dtype=float)
        stored_labels = _load_labels_dataframe()
        if not stored_labels.empty:
            serials = stored_labels.index.to_numpy()
            label_values = stored_labels["label"].astype(int).to_numpy()
            valid_mask = (serials >= 0) & (serials < x_train.shape[0])
            labels_array[serials[valid_mask]] = label_values[valid_mask].astype(float)

        hover_metadata = _build_hover_metadata(pred_classes, pred_certainty, labels_array, true_labels)

        # Build immutable state tree
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
        
        history = TrainingHistory.empty()
        
        cache: Dict[str, object] = {"base_figures": {}, "colors": {}}
        
        # Create AppState FIRST
        new_state = AppState(
            model=model,
            trainer=trainer,
            config=config,
            data=data_state,
            training=training_status,
            ui=ui_state,
            cache=cache,
            history=history
        )
        
        # Then atomically assign it under state_lock
        with state_lock:
            app_state = new_state


def _load_labels_dataframe() -> pd.DataFrame:
    """Load the labels CSV (index=Serial) or create an empty frame with Int64 labels."""
    columns = ["Serial", "label"]
    if LABELS_PATH.exists():
        df = pd.read_csv(LABELS_PATH, usecols=columns)
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
    persisted = df.copy()
    if not persisted.empty:
        persisted.index = persisted.index.astype(int)
        persisted["label"] = persisted["label"].astype("Int64")
    persisted.index.name = "Serial"
    persisted.to_csv(LABELS_PATH)


def _update_label(sample_idx: int, new_label: float | None) -> Tuple[dict, str]:
    """Update label state and CSV, returning store payload and status message."""
    with state_lock:
        global app_state
        
        # Copy labels array and update
        labels_array = app_state.data.labels.copy()
        if new_label is None:
            labels_array[sample_idx] = np.nan
        else:
            labels_array[sample_idx] = float(new_label)
        
        # Update CSV persistence
        df = _load_labels_dataframe()
        if new_label is None:
            if sample_idx in df.index:
                df = df.drop(sample_idx)
        else:
            df.loc[sample_idx, "label"] = int(new_label)
        _persist_labels_dataframe(df)
        
        # Update hover metadata
        hover_metadata = list(app_state.data.hover_metadata)
        true_label_value = int(app_state.data.true_labels[sample_idx])
        hover_metadata[sample_idx] = _format_hover_metadata_entry(
            sample_idx,
            int(app_state.data.pred_classes[sample_idx]),
            float(app_state.data.pred_certainty[sample_idx]),
            float(labels_array[sample_idx]),
            true_label_value,
        )
        
        # Atomic state update
        app_state = app_state.with_label_update(labels_array, hover_metadata)
        version_payload = {"version": app_state.data.version}
    
    if new_label is None:
        message = f"Removed label for sample {sample_idx}"
    else:
        message = f"Labeled sample {sample_idx} as {int(new_label)}"
    return version_payload, message
