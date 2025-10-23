"""State management layer for the dashboard."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, Tuple

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

from use_cases.dashboard.utils import _build_hover_text  # noqa: E402


CHECKPOINT_PATH = ROOT_DIR / "artifacts" / "checkpoints" / "ssvae.ckpt"
LABELS_PATH = ROOT_DIR / "data" / "mnist" / "labels.csv"
COOLWARM_CMAP = colormaps["coolwarm"]
MAX_STATUS_MESSAGES = 10


state_lock = threading.Lock()
metrics_queue: Queue[Dict[str, float]] = Queue()

app_state: Dict[str, object] = {
    "model": None,
    "trainer": None,
    "config": None,
    "data": {
        "x_train": None,
        "labels": None,
        "true_labels": None,
        "latent": None,
        "reconstructed": None,
        "pred_classes": None,
        "pred_certainty": None,
        "hover_text": None,
    },
    "training": {
        "active": False,
        "thread": None,
        "target_epochs": 0,
        "status_messages": [],
    },
    "ui": {
        "selected_sample": 0,
        "color_mode": "user_labels",
        "labels_version": 0,
        "latent_version": 0,
    },
    "history": {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_reconstruction_loss": [],
        "val_reconstruction_loss": [],
        "train_kl_loss": [],
        "val_kl_loss": [],
        "train_classification_loss": [],
        "val_classification_loss": [],
    },
}


def _append_status_message_locked(message: str) -> None:
    messages = app_state["training"].setdefault("status_messages", [])
    messages.append(message)
    if len(messages) > MAX_STATUS_MESSAGES:
        messages = messages[-MAX_STATUS_MESSAGES:]
    app_state["training"]["status_messages"] = messages


def _append_status_message(message: str) -> None:
    with state_lock:
        _append_status_message_locked(message)


def _update_history_with_epoch(payload: Dict[str, float]) -> None:
    with state_lock:
        history = app_state["history"]
        history["epochs"].append(int(payload["epoch"]))
        for key in (
            "train_loss",
            "val_loss",
            "train_reconstruction_loss",
            "val_reconstruction_loss",
            "train_kl_loss",
            "val_kl_loss",
            "train_classification_loss",
            "val_classification_loss",
        ):
            value = payload.get(key)
            if value is not None:
                history.setdefault(key, []).append(float(value))


def _clear_metrics_queue() -> None:
    while True:
        try:
            metrics_queue.get_nowait()
        except Empty:
            break


def initialize_model_and_data() -> None:
    """Load model, dataset, labels, and derived predictions into memory."""
    with state_lock:
        if app_state["model"] is not None:
            return
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

    hover_text = _build_hover_text(pred_classes, pred_certainty, labels_array, true_labels)

    with state_lock:
        app_state["model"] = model
        app_state["trainer"] = trainer
        app_state["config"] = config
        app_state["data"]["x_train"] = x_train
        app_state["data"]["labels"] = labels_array
        app_state["data"]["true_labels"] = true_labels
        app_state["data"]["latent"] = latent
        app_state["data"]["reconstructed"] = recon
        app_state["data"]["pred_classes"] = pred_classes
        app_state["data"]["pred_certainty"] = pred_certainty
        app_state["data"]["hover_text"] = hover_text
        app_state["ui"]["selected_sample"] = int(app_state["ui"]["selected_sample"])
        app_state["training"]["active"] = False
        app_state["training"]["thread"] = None
        app_state["training"]["target_epochs"] = 0
        app_state["training"]["status_messages"] = []
        app_state["ui"]["latent_version"] = 0


def _format_hover_entry(
    idx: int,
    pred_class: int,
    pred_certainty: float,
    user_label: float,
    true_label: int | None,
) -> str:
    label_text = "Unlabeled" if np.isnan(user_label) else f"{int(user_label)}"
    true_label_text = "?" if true_label is None else f"{true_label}"
    return (
        f"Index: {idx}<br>Prediction: {pred_class}"
        f"<br>Confidence: {pred_certainty * 100:.1f}%"
        f"<br>User Label: {label_text}"
        f"<br>True Label: {true_label_text}"
    )


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
        labels_array: np.ndarray = app_state["data"]["labels"]
        if new_label is None:
            labels_array[sample_idx] = np.nan
        else:
            labels_array[sample_idx] = float(new_label)

        df = _load_labels_dataframe()
        if new_label is None:
            if sample_idx in df.index:
                df = df.drop(sample_idx)
        else:
            df.loc[sample_idx, "label"] = int(new_label)
        _persist_labels_dataframe(df)

        app_state["data"]["labels"] = labels_array
        pred_classes = np.array(app_state["data"]["pred_classes"], dtype=np.int32)
        pred_certainty = np.array(app_state["data"]["pred_certainty"], dtype=np.float64)
        true_labels = app_state["data"]["true_labels"]
        hover_text = list(app_state["data"]["hover_text"])
        true_label_value = int(true_labels[sample_idx]) if true_labels is not None else None
        hover_text[sample_idx] = _format_hover_entry(
            sample_idx,
            int(pred_classes[sample_idx]),
            float(pred_certainty[sample_idx]),
            labels_array[sample_idx],
            true_label_value,
        )
        app_state["data"]["hover_text"] = hover_text
        app_state["ui"]["labels_version"] = int(app_state["ui"]["labels_version"]) + 1
        version_payload = {"version": app_state["ui"]["labels_version"]}

    if new_label is None:
        message = f"Removed label for sample {sample_idx}"
    else:
        message = f"Labeled sample {sample_idx} as {int(new_label)}"
    return version_payload, message
