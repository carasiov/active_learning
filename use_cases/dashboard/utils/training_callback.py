"""
Training callback used by the dashboard to stream metrics from the background
training thread to the Dash UI.

The callback packages scalar metrics into plain Python floats so they can be
serialized safely across threads and enqueued for the polling callback.
"""

from __future__ import annotations

from queue import Queue
from typing import Dict, Optional

import numpy as np

from rcmvae.application.callbacks import TrainingCallback


# Custom exception for user-initiated stop
class TrainingStoppedException(Exception):
    """Raised when user requests training to stop."""
    pass


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        arr = np.asarray(value)
        if arr.size == 0:
            return None
        return float(arr.reshape(()))


class DashboardMetricsCallback(TrainingCallback):
    """Callback that forwards epoch-level metrics to a thread-safe queue.
    
    Also checks for user-initiated stop requests between epochs.
    """

    def __init__(self, queue: Queue, target_epochs: int):
        self._queue = queue
        self._target_epochs = int(target_epochs)
        self._stop_checked = False

    def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, Dict[str, object]],
        history: Dict[str, list[float]],
        trainer,
    ) -> None:
        train_metrics = metrics.get("train", {})
        val_metrics = metrics.get("val", {})

        payload: Dict[str, float | int] = {
            "type": "epoch_complete",
            "epoch": int(epoch + 1),
            "target_epochs": self._target_epochs,
        }

        metric_map = {
            "train_loss": train_metrics.get("loss"),
            "val_loss": val_metrics.get("loss"),
            "train_reconstruction_loss": train_metrics.get("reconstruction_loss"),
            "val_reconstruction_loss": val_metrics.get("reconstruction_loss"),
            "train_kl_loss": train_metrics.get("kl_loss"),
            "val_kl_loss": val_metrics.get("kl_loss"),
            "train_classification_loss": train_metrics.get("classification_loss"),
            "val_classification_loss": val_metrics.get("classification_loss"),
        }

        for key, raw_value in metric_map.items():
            value = _as_float(raw_value)
            if value is not None:
                payload[key] = value

        self._queue.put(payload)
        
        # Check if user requested stop
        if self._check_stop_request():
            self._queue.put({"type": "training_stopped", "message": "Training stopped by user"})
            # Raise exception to break training loop gracefully
            raise TrainingStoppedException("User requested training stop")
    
    def _check_stop_request(self) -> bool:
        """Check if user requested training to stop."""
        try:
            from use_cases.dashboard import state as dashboard_state

            with dashboard_state.state_manager.state_lock:
                current_state = dashboard_state.state_manager.state
                if current_state and current_state.active_model:
                    return current_state.active_model.training.stop_requested
        except Exception:
            # Don't crash training if check fails
            pass
        return False

    def on_train_end(self, history, trainer) -> None:
        self._queue.put({"type": "training_complete"})
