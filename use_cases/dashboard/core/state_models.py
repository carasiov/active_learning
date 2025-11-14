"""Immutable state models for dashboard state management."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Dict, List, Optional

import numpy as np

from model.ssvae import SSVAE, SSVAEConfig
from model.training.interactive_trainer import InteractiveTrainer


@dataclass(frozen=True)
class ModelMetadata:
    """Lightweight model info for home page and registry."""
    model_id: str
    name: str
    created_at: str  # ISO format
    last_modified: str  # ISO format
    dataset: str
    total_epochs: int
    labeled_count: int
    latest_loss: Optional[float]
    
    @classmethod
    def from_dict(cls, data: dict) -> ModelMetadata:
        """Load from metadata.json - handles old and new field names."""
        # Handle old field name 'labeled_samples' -> new name 'labeled_count'
        data = dict(data)  # Copy to avoid modifying original
        if "labeled_samples" in data and "labeled_count" not in data:
            data["labeled_count"] = data.pop("labeled_samples")
        
        # Handle old field name 'id' -> new name 'model_id'
        if "id" in data and "model_id" not in data:
            data["model_id"] = data.pop("id")
        
        # Remove any fields that don't exist in current ModelMetadata
        valid_fields = {'model_id', 'name', 'created_at', 'last_modified', 'dataset', 
                       'total_epochs', 'labeled_count', 'latest_loss'}
        data = {k: v for k, v in data.items() if k in valid_fields}
        
        # Ensure required fields have defaults for old metadata
        data.setdefault('model_id', 'unknown')
        data.setdefault('name', data.get('model_id', 'Unnamed Model'))
        data.setdefault('dataset', 'mnist')
        data.setdefault('created_at', '2025-01-01T00:00:00')
        data.setdefault('last_modified', data.get('created_at', '2025-01-01T00:00:00'))
        data.setdefault('total_epochs', 0)
        data.setdefault('labeled_count', 0)
        data.setdefault('latest_loss', None)
        
        return cls(**data)
    
    def to_dict(self) -> dict:
        """Serialize to metadata.json"""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "dataset": self.dataset,
            "total_epochs": self.total_epochs,
            "labeled_count": self.labeled_count,
            "latest_loss": self.latest_loss,
        }


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
class ModelState:
    """Full state for one model - only loaded when active."""
    model_id: str
    metadata: ModelMetadata  # Embed metadata here too
    model: SSVAE
    trainer: InteractiveTrainer
    config: SSVAEConfig
    data: DataState
    training: TrainingStatus
    ui: UIState
    history: TrainingHistory
    
    def with_updated_metadata(self, **kwargs) -> ModelState:
        """Helper to update metadata fields."""
        new_metadata = replace(self.metadata, **kwargs)
        return replace(self, metadata=new_metadata)
    
    def _clear_cache_in_app_state(self, app_state: AppState) -> AppState:
        """Helper to clear cache when this model's state changes."""
        return replace(app_state, cache={"base_figures": {}, "colors": {}})
    
    # Atomic update helpers (updated for ModelState)
    
    def with_data(self, **changes) -> ModelState:
        """Immutable update of data state."""
        return replace(self, data=replace(self.data, **changes))
    
    def with_training(self, **changes) -> ModelState:
        """Immutable update of training state."""
        return replace(self, training=replace(self.training, **changes))
    
    def with_ui(self, **changes) -> ModelState:
        """Immutable update of UI state."""
        return replace(self, ui=replace(self.ui, **changes))
    
    def with_history(self, new_history: TrainingHistory) -> ModelState:
        """Replace entire history."""
        return replace(self, history=new_history)
    
    def with_config(self, config: SSVAEConfig) -> ModelState:
        """Replace configuration snapshot."""
        return replace(self, config=config)
    
    # Domain-specific updates
    
    def with_label_update(
        self,
        new_labels: np.ndarray,
        new_hover_metadata: List[List[object]]
    ) -> ModelState:
        """Update labels and increment data version."""
        new_data = self.data.with_updated_labels(new_labels, new_hover_metadata)
        return replace(self, data=new_data)
    
    def with_training_complete(
        self,
        latent: np.ndarray,
        reconstructed: np.ndarray,
        pred_classes: np.ndarray,
        pred_certainty: np.ndarray,
        hover_metadata: List[List[object]]
    ) -> ModelState:
        """Update after training completes - new predictions and training state."""
        new_data = self.data.with_updated_predictions(
            latent, reconstructed, pred_classes, pred_certainty, hover_metadata
        )
        new_training = self.training.with_complete()
        return replace(self, data=new_data, training=new_training)


@dataclass(frozen=True)
class AppState:
    """Root state - manages model registry.
    
    Thread-safety: The entire AppState instance is replaced atomically under state_lock.
    """
    models: Dict[str, ModelMetadata]  # All models (lightweight)
    active_model: Optional[ModelState]  # Only one loaded
    cache: Dict[str, object]  # Shared cache
    
    def with_active_model(self, model_state: ModelState) -> AppState:
        """Load a model as active."""
        # Update registry with latest metadata
        updated_models = dict(self.models)
        updated_models[model_state.model_id] = model_state.metadata
        return replace(
            self,
            models=updated_models,
            active_model=model_state,
            cache={"base_figures": {}, "colors": {}}  # Clear cache
        )
    
    def with_unloaded_model(self) -> AppState:
        """Unload active model."""
        return replace(self, active_model=None, cache={})
    
    def with_model_metadata(self, metadata: ModelMetadata) -> AppState:
        """Update registry entry."""
        updated_models = dict(self.models)
        updated_models[metadata.model_id] = metadata
        return replace(self, models=updated_models)
