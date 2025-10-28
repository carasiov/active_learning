"""Immutable state models for dashboard state management."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Dict, List, Optional

import numpy as np

from ssvae import SSVAE, SSVAEConfig
from training.interactive_trainer import InteractiveTrainer


class TrainingState(Enum):
    """Training lifecycle states."""
    IDLE = auto()
    QUEUED = auto()
    RUNNING = auto()
    COMPLETE = auto()
    ERROR = auto()
    
    def is_active(self) -> bool:
        """Check if training is currently active."""
        return self in {TrainingState.QUEUED, TrainingState.RUNNING}
    
    def can_start(self) -> bool:
        """Check if training can be started from this state."""
        return self in {TrainingState.IDLE, TrainingState.COMPLETE, TrainingState.ERROR}


@dataclass(frozen=True)
class DataState:
    """Immutable container for all dataset and model predictions.
    
    Version increments whenever data changes (labels, latent, predictions).
    This single version replaces the separate labels_version and latent_version.
    """
    x_train: np.ndarray
    labels: np.ndarray  # Float array with NaN for unlabeled
    true_labels: np.ndarray
    latent: np.ndarray
    reconstructed: np.ndarray
    pred_classes: np.ndarray
    pred_certainty: np.ndarray
    hover_metadata: List[List[object]]
    version: int
    
    def with_updated_labels(
        self, 
        new_labels: np.ndarray, 
        new_hover_metadata: List[List[object]]
    ) -> DataState:
        """Create new state with updated labels and hover metadata."""
        return replace(
            self,
            labels=new_labels,
            hover_metadata=new_hover_metadata,
            version=self.version + 1
        )
    
    def with_updated_predictions(
        self,
        latent: np.ndarray,
        reconstructed: np.ndarray,
        pred_classes: np.ndarray,
        pred_certainty: np.ndarray,
        hover_metadata: List[List[object]]
    ) -> DataState:
        """Create new state after training completes."""
        return replace(
            self,
            latent=latent,
            reconstructed=reconstructed,
            pred_classes=pred_classes,
            pred_certainty=pred_certainty,
            hover_metadata=hover_metadata,
            version=self.version + 1
        )


@dataclass(frozen=True)
class TrainingStatus:
    """Immutable training progress state."""
    state: TrainingState
    target_epochs: int
    status_messages: List[str]  # Immutable list
    thread: Optional[object]  # Thread reference (not truly immutable, but handled carefully)
    stop_requested: bool = False  # Flag to signal training should stop
    
    def is_active(self) -> bool:
        """Convenience method - delegates to state enum."""
        return self.state.is_active()
    
    def can_start(self) -> bool:
        """Convenience method - delegates to state enum."""
        return self.state.can_start()
    
    def with_message(self, message: str, max_messages: int = 10) -> TrainingStatus:
        """Add status message, keeping only last N messages."""
        new_messages = list(self.status_messages) + [message]
        if len(new_messages) > max_messages:
            new_messages = new_messages[-max_messages:]
        return replace(self, status_messages=new_messages)
    
    def with_queued(self, target_epochs: int) -> TrainingStatus:
        """Transition to QUEUED state."""
        return replace(
            self,
            state=TrainingState.QUEUED,
            target_epochs=target_epochs,
            stop_requested=False
        ).with_message(f"Queued training for {target_epochs} epoch(s).")
    
    def with_running(self, thread: object) -> TrainingStatus:
        """Transition to RUNNING state."""
        return replace(
            self,
            state=TrainingState.RUNNING,
            thread=thread
        )
    
    def with_complete(self) -> TrainingStatus:
        """Transition to COMPLETE state."""
        return replace(
            self,
            state=TrainingState.COMPLETE,
            thread=None,
            stop_requested=False
        ).with_message("Training complete.")
    
    def with_error(self, error_message: str) -> TrainingStatus:
        """Transition to ERROR state."""
        return replace(
            self,
            state=TrainingState.ERROR,
            thread=None,
            stop_requested=False
        ).with_message(f"Error: {error_message}")
    
    def with_idle(self) -> TrainingStatus:
        """Transition to IDLE state."""
        return replace(
            self,
            state=TrainingState.IDLE,
            thread=None,
            target_epochs=0,
            stop_requested=False
        )
    
    def with_stop_requested(self) -> TrainingStatus:
        """Request training to stop."""
        return replace(self, stop_requested=True).with_message("Stop requested - will halt after current epoch.")
    
    def with_cleared_messages(self) -> TrainingStatus:
        """Clear all status messages."""
        return replace(self, status_messages=[])


@dataclass(frozen=True)
class UIState:
    """Immutable UI state (selected sample, color mode, etc.)."""
    selected_sample: int
    color_mode: str
    
    def with_selection(self, sample_idx: int) -> UIState:
        """Update selected sample."""
        return replace(self, selected_sample=sample_idx)
    
    def with_color_mode(self, mode: str) -> UIState:
        """Update color mode."""
        return replace(self, color_mode=mode)


@dataclass(frozen=True)
class TrainingHistory:
    """Immutable training history accumulated across epochs."""
    epochs: List[int]
    train_loss: List[float]
    val_loss: List[float]
    train_reconstruction_loss: List[float]
    val_reconstruction_loss: List[float]
    train_kl_loss: List[float]
    val_kl_loss: List[float]
    train_classification_loss: List[float]
    val_classification_loss: List[float]
    
    @classmethod
    def empty(cls) -> TrainingHistory:
        """Create empty history."""
        return cls(
            epochs=[],
            train_loss=[],
            val_loss=[],
            train_reconstruction_loss=[],
            val_reconstruction_loss=[],
            train_kl_loss=[],
            val_kl_loss=[],
            train_classification_loss=[],
            val_classification_loss=[]
        )
    
    def with_epoch(self, epoch: int, metrics: Dict[str, float]) -> TrainingHistory:
        """Add epoch data, returning new history."""
        return replace(
            self,
            epochs=self.epochs + [epoch],
            train_loss=self.train_loss + [metrics.get("train_loss", 0.0)],
            val_loss=self.val_loss + [metrics.get("val_loss", 0.0)],
            train_reconstruction_loss=self.train_reconstruction_loss + [metrics.get("train_reconstruction_loss", 0.0)],
            val_reconstruction_loss=self.val_reconstruction_loss + [metrics.get("val_reconstruction_loss", 0.0)],
            train_kl_loss=self.train_kl_loss + [metrics.get("train_kl_loss", 0.0)],
            val_kl_loss=self.val_kl_loss + [metrics.get("val_kl_loss", 0.0)],
            train_classification_loss=self.train_classification_loss + [metrics.get("train_classification_loss", 0.0)],
            val_classification_loss=self.val_classification_loss + [metrics.get("val_classification_loss", 0.0)]
        )


@dataclass(frozen=True)
class AppState:
    """Root immutable state container for the entire dashboard.
    
    Thread-safety: The entire AppState instance is replaced atomically under state_lock.
    Never mutate fields directly - always use with_* methods to create new instances.
    """
    model: SSVAE
    trainer: InteractiveTrainer
    config: SSVAEConfig
    data: DataState
    training: TrainingStatus
    ui: UIState
    cache: Dict[str, object]  # Keep as dict for now (can refine later)
    history: TrainingHistory
    
    def _clear_cache(self) -> AppState:
        """Clear visualization caches - use when data changes."""
        return replace(self, cache={"base_figures": {}, "colors": {}})
    
    # Atomic update helpers
    
    def with_data(self, **changes) -> AppState:
        """Immutable update of data state."""
        return replace(self, data=replace(self.data, **changes))
    
    def with_training(self, **changes) -> AppState:
        """Immutable update of training state."""
        return replace(self, training=replace(self.training, **changes))
    
    def with_ui(self, **changes) -> AppState:
        """Immutable update of UI state."""
        return replace(self, ui=replace(self.ui, **changes))
    
    def with_history(self, new_history: TrainingHistory) -> AppState:
        """Replace entire history."""
        return replace(self, history=new_history)
    
    # Domain-specific updates
    
    def with_label_update(
        self,
        new_labels: np.ndarray,
        new_hover_metadata: List[List[object]]
    ) -> AppState:
        """Update labels and increment data version."""
        new_data = self.data.with_updated_labels(new_labels, new_hover_metadata)
        # Clear cache since hover metadata changed
        return replace(self, data=new_data)._clear_cache()
    
    def with_training_complete(
        self,
        latent: np.ndarray,
        reconstructed: np.ndarray,
        pred_classes: np.ndarray,
        pred_certainty: np.ndarray,
        hover_metadata: List[List[object]]
    ) -> AppState:
        """Update after training completes - new predictions and training state."""
        new_data = self.data.with_updated_predictions(
            latent, reconstructed, pred_classes, pred_certainty, hover_metadata
        )
        new_training = self.training.with_complete()
        # Clear cache since latent space changed completely
        return replace(self, data=new_data, training=new_training)._clear_cache()
