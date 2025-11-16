"""Application state manager - encapsulates all state and provides thread-safe access."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
import os
from queue import Empty, Queue
from typing import Dict, Optional
from dataclasses import replace

import numpy as np
import pandas as pd

# Ensure repository imports work when running without installation.
ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
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
)
from use_cases.dashboard.core.state_models import (  # noqa: E402
    AppState,
    ModelState,
    DataState,
    TrainingStatus,
    TrainingState,
    UIState,
)
from use_cases.dashboard.core.commands import CommandDispatcher  # noqa: E402
from use_cases.dashboard.services import ServiceContainer  # noqa: E402
from use_cases.dashboard.core.logging_config import get_logger  # noqa: E402


logger = get_logger('state_manager')
MAX_STATUS_MESSAGES = 10
PREVIEW_SAMPLE_LIMIT = 2048  # Limit dataset size when loading models for dashboard
FAST_DASHBOARD_MODE = os.environ.get("DASHBOARD_FAST_MODE", "1").lower() not in {"0", "false", "no"}


class AppStateManager:
    """Manages application state and provides thread-safe access.

    This class encapsulates all global state and provides a clean interface
    for state operations. All state modifications are thread-safe.
    """

    def __init__(self):
        """Initialize state manager with empty state."""
        self._state_lock = threading.RLock()
        self._init_lock = threading.Lock()
        self._metrics_queue: Queue[Dict[str, float]] = Queue()
        self._state: Optional[AppState] = None
        self._services = ServiceContainer.create_default(self._metrics_queue)
        self._dispatcher = CommandDispatcher(self._state_lock, self._services)

    @property
    def state(self) -> Optional[AppState]:
        """Get current app state (read-only access).

        Note: Returns a snapshot of state. For modifications, use update_state().
        """
        with self._state_lock:
            return self._state

    @property
    def services(self) -> ServiceContainer:
        """Get service container."""
        return self._services

    @property
    def dispatcher(self) -> CommandDispatcher:
        """Get command dispatcher."""
        return self._dispatcher

    @property
    def metrics_queue(self) -> Queue[Dict[str, float]]:
        """Get metrics queue."""
        return self._metrics_queue

    @property
    def state_lock(self) -> threading.RLock:
        """Get state lock for external synchronization if needed."""
        return self._state_lock

    def update_state(self, new_state: AppState) -> None:
        """Update app state (thread-safe).

        Args:
            new_state: New application state to set
        """
        with self._state_lock:
            # Log training state transitions for debugging
            if self._state and self._state.active_model and new_state.active_model:
                old_training_state = self._state.active_model.training.state
                new_training_state = new_state.active_model.training.state
                if old_training_state != new_training_state:
                    model_id = new_state.active_model.model_id
                    logger.info(
                        f"Training state transition | model={model_id} | "
                        f"{old_training_state.name} → {new_training_state.name}"
                    )

            self._state = new_state

    def initialize(self) -> None:
        """Initialize app state with model registry.

        This is idempotent - safe to call multiple times.
        """
        if self._state is not None:
            return

        with self._init_lock:
            if self._state is not None:
                return

            from use_cases.dashboard.core.model_manager import ModelManager

            # Load all model metadata
            models = ModelManager.list_all_models()

            # Create empty registry (no active model)
            with self._state_lock:
                self._state = AppState(
                    models=models,
                    active_model=None,
                    cache={}
                )

    def load_model(self, model_id: str) -> None:
        """Load a specific model as active.

        Args:
            model_id: ID of model to load

        Raises:
            ValueError: If model not found
        """
        from use_cases.dashboard.core.model_manager import ModelManager
        from use_cases.dashboard.core.model_runs import load_run_records

        # Ensure app state initialized
        self.initialize()

        with self._state_lock:
            # Don't reload if already active
            if self._state.active_model and self._state.active_model.model_id == model_id:
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
            self._state = self._state.with_active_model(model_state)

    def append_status_message(self, message: str) -> None:
        """Append status message to active model.

        Args:
            message: Status message to append
        """
        with self._state_lock:
            if self._state is None or self._state.active_model is None:
                return
            updated_model = replace(
                self._state.active_model,
                training=self._state.active_model.training.with_message(
                    message, max_messages=MAX_STATUS_MESSAGES
                )
            )
            self._state = self._state.with_active_model(updated_model)

    def update_history_with_epoch(self, payload: Dict[str, float]) -> None:
        """Update training history with epoch metrics.

        Args:
            payload: Dictionary of metrics for the epoch
        """
        with self._state_lock:
            if self._state is None or self._state.active_model is None:
                return
            current_history = self._state.active_model.history
            next_epoch = (current_history.epochs[-1] + 1) if current_history.epochs else 1
            metrics = dict(payload)
            metrics["epoch_absolute"] = float(next_epoch)
            metrics["epoch_in_run"] = float(payload.get("epoch", next_epoch))
            new_history = current_history.with_epoch(next_epoch, metrics)
            updated_model = self._state.active_model.with_history(new_history)
            self._state = self._state.with_active_model(updated_model)

    def clear_metrics_queue(self) -> None:
        """Clear all pending metrics from queue."""
        while True:
            try:
                self._metrics_queue.get_nowait()
            except Empty:
                break
