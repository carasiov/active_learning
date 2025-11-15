"""Command pattern infrastructure for state-modifying actions."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import threading
import os

from use_cases.dashboard.core.state_models import AppState
from use_cases.dashboard.core.logging_config import get_logger

logger = get_logger('commands')
PREVIEW_SAMPLE_LIMIT = 2048
FAST_DASHBOARD_MODE = os.environ.get("DASHBOARD_FAST_MODE", "1").lower() not in {"0", "false", "no"}


class Command(ABC):
    """Base class for all state-modifying commands.
    
    Commands encapsulate both validation and execution logic.
    This makes actions testable, auditable, and explicit.
    """
    
    @abstractmethod
    def validate(self, state: AppState) -> Optional[str]:
        """Validate if command can execute on current state.
        
        Args:
            state: Current application state (read-only)
        
        Returns:
            Error message if invalid, None if valid
        """
        pass
    
    @abstractmethod
    def execute(self, state: AppState) -> Tuple[AppState, str]:
        """Execute command, producing new state.
        
        Args:
            state: Current application state (read-only)
        
        Returns:
            (new_state, status_message)
        
        Raises:
            Exception: If execution fails unexpectedly
        """
        pass


@dataclass
class CommandHistoryEntry:
    """Record of a command execution."""
    command: Command
    timestamp: float
    success: bool
    message: str


class CommandDispatcher:
    """Central dispatcher for all state-modifying commands.
    
    Provides:
    - Atomic execution under lock
    - Validation before execution
    - Command history for debugging
    - Thread-safe access to state
    """
    
    def __init__(self, state_lock: threading.Lock | threading.RLock):
        """Initialize dispatcher with state lock.
        
        Args:
            state_lock: Lock or RLock protecting app_state access
        """
        self._state_lock = state_lock
        self._command_history: List[CommandHistoryEntry] = []
        self._history_lock = threading.Lock()
    
    def execute(self, command: Command) -> Tuple[bool, str]:
        """Execute command atomically.
        
        Args:
            command: Command to execute
        
        Returns:
            (success, message) tuple
        """
        from use_cases.dashboard.core import state as dashboard_state
        
        with self._state_lock:
            # Check state is initialized
            if dashboard_state.app_state is None:
                error_msg = "Application state not initialized"
                self._log_command(command, success=False, message=error_msg)
                return False, error_msg
            
            # Validate
            error = command.validate(dashboard_state.app_state)
            if error:
                self._log_command(command, success=False, message=error)
                return False, error
            
            # Execute
            try:
                new_state, message = command.execute(dashboard_state.app_state)
                dashboard_state.app_state = new_state
                self._log_command(command, success=True, message=message)
                return True, message
            
            except Exception as e:
                error_msg = f"Command execution failed: {e}"
                self._log_command(command, success=False, message=error_msg)
                return False, error_msg
    
    def _log_command(self, command: Command, success: bool, message: str) -> None:
        """Log command to history (thread-safe)."""
        entry = CommandHistoryEntry(
            command=command,
            timestamp=time.time(),
            success=success,
            message=message
        )
        
        with self._history_lock:
            self._command_history.append(entry)
            # Keep last 1000 commands
            if len(self._command_history) > 1000:
                self._command_history = self._command_history[-500:]
    
    def get_history(self, limit: int = 100) -> List[CommandHistoryEntry]:
        """Get recent command history.
        
        Args:
            limit: Maximum number of entries to return
        
        Returns:
            List of recent command history entries
        """
        with self._history_lock:
            return list(self._command_history[-limit:])
    
    def clear_history(self) -> None:
        """Clear command history (useful for testing)."""
        with self._history_lock:
            self._command_history.clear()


# ============================================================================
# Concrete Command Implementations
# ============================================================================

import numpy as np
import pandas as pd


@dataclass
class LabelSampleCommand(Command):
    """Command to assign a label to a sample.
    
    Handles:
    - Label assignment
    - CSV persistence  
    - Hover metadata update
    - Data version increment
    """
    sample_idx: int
    label: Optional[int]  # None = delete label
    
    def validate(self, state: AppState) -> Optional[str]:
        """Validate sample index and label value."""
        if state.active_model is None:
            return "No model loaded"
        
        # Check sample index bounds
        if self.sample_idx < 0 or self.sample_idx >= len(state.active_model.data.labels):
            return f"Invalid sample index: {self.sample_idx}"
        
        # Check label value (if not deletion)
        if self.label is not None:
            if not isinstance(self.label, int):
                return f"Label must be an integer, got {type(self.label)}"
            if not (0 <= self.label <= 9):
                return f"Label must be 0-9, got {self.label}"
        
        return None  # Valid
    
    def execute(self, state: AppState) -> Tuple[AppState, str]:
        """Execute label update on ACTIVE model."""
        if state.active_model is None:
            return state, "No model loaded"
        
        from use_cases.dashboard.core.state import _load_labels_dataframe, _persist_labels_dataframe
        from use_cases.dashboard.core.model_manager import ModelManager
        from datetime import datetime
        
        # Copy labels array and update
        labels_array = state.active_model.data.labels.copy()
        if self.label is None:
            labels_array[self.sample_idx] = np.nan
        else:
            labels_array[self.sample_idx] = float(self.label)
        
        # Update CSV persistence
        df = _load_labels_dataframe()
        if self.label is None:
            if self.sample_idx in df.index:
                df = df.drop(self.sample_idx)
        else:
            df.loc[self.sample_idx, "label"] = int(self.label)
        _persist_labels_dataframe(df)
        
        # Update hover metadata
        from use_cases.dashboard.utils.visualization import _format_hover_metadata_entry
        hover_metadata = list(state.active_model.data.hover_metadata)
        true_label_value = int(state.active_model.data.true_labels[self.sample_idx])
        hover_metadata[self.sample_idx] = _format_hover_metadata_entry(
            self.sample_idx,
            int(state.active_model.data.pred_classes[self.sample_idx]),
            float(state.active_model.data.pred_certainty[self.sample_idx]),
            float(labels_array[self.sample_idx]),
            true_label_value,
        )
        
        # Update model with label changes
        updated_model = state.active_model.with_label_update(labels_array, hover_metadata)
        
        # Update metadata
        labeled_count = int(np.sum(~np.isnan(labels_array)))
        updated_model = updated_model.with_updated_metadata(
            labeled_count=labeled_count,
            last_modified=datetime.utcnow().isoformat()
        )
        
        # Save metadata
        ModelManager.save_metadata(updated_model.metadata)
        
        # Update app state
        new_state = state.with_active_model(updated_model)
        
        # Generate status message
        if self.label is None:
            message = f"Removed label for sample {self.sample_idx}"
        else:
            message = f"Labeled sample {self.sample_idx} as {self.label}"
        
        return new_state, message


@dataclass
class StartTrainingCommand(Command):
    """Command to start a training run.
    
    Validates:
    - Training not already active
    - Valid epoch count
    - Has labeled samples
    - Valid hyperparameters
    """
    num_epochs: int
    recon_weight: float
    kl_weight: float
    learning_rate: float
    
    def validate(self, state: AppState) -> Optional[str]:
        """Validate training can start."""
        if state.active_model is None:
            return "No model loaded"
        
        # Check not already training
        if state.active_model.training.is_active():
            return "Training already in progress"
        
        # Validate epochs
        if self.num_epochs < 1 or self.num_epochs > 200:
            return f"Epochs must be between 1 and 200, got {self.num_epochs}"
        
        # Validate hyperparameters
        if self.recon_weight < 0:
            return "Reconstruction weight must be non-negative"
        if self.kl_weight < 0 or self.kl_weight > 10:
            return "KL weight must be between 0 and 10"
        if self.learning_rate <= 0 or self.learning_rate > 0.1:
            return "Learning rate must be between 0.00001 and 0.1"
        
        return None  # Valid
    
    def execute(self, state: AppState) -> Tuple[AppState, str]:
        """Queue training with updated config."""
        if state.active_model is None:
            return state, "No model loaded"
        
        labeled_count = int(np.sum(~np.isnan(state.active_model.data.labels)))
        if labeled_count == 0:
            logger.warning(
                "Starting training with zero labeled samples for model %s",
                state.active_model.model_id,
            )
        
        # Update config (mutable for now - configs remain mutable)
        state.active_model.config.recon_weight = float(self.recon_weight)
        state.active_model.config.kl_weight = float(self.kl_weight)
        state.active_model.config.learning_rate = float(self.learning_rate)
        
        # Update model config
        state.active_model.model.config.recon_weight = float(self.recon_weight)
        state.active_model.model.config.kl_weight = float(self.kl_weight)
        state.active_model.model.config.learning_rate = float(self.learning_rate)
        
        # Update trainer config
        state.active_model.trainer.config.recon_weight = float(self.recon_weight)
        state.active_model.trainer.config.kl_weight = float(self.kl_weight)
        state.active_model.trainer.config.learning_rate = float(self.learning_rate)
        
        # Transition to QUEUED state
        from use_cases.dashboard.core.state_models import TrainingState
        updated_model = state.active_model.with_training(
            state=TrainingState.QUEUED,
            target_epochs=self.num_epochs,
            stop_requested=False
        )
        
        # Update app state
        new_state = state.with_active_model(updated_model)
        
        if labeled_count == 0:
            message = (
                f"Queued training for {self.num_epochs} epoch(s). "
                "Warning: 0 labeled samples; supervised loss will be skipped."
            )
        else:
            message = (
                f"Queued training for {self.num_epochs} epoch(s) "
                f"with {labeled_count:,} labeled sample(s)."
            )
        return new_state, message


@dataclass  
class CompleteTrainingCommand(Command):
    """Command to mark training as complete with new predictions.
    
    This is called by the background training worker when training finishes.
    """
    latent: np.ndarray
    reconstructed: np.ndarray
    pred_classes: np.ndarray
    pred_certainty: np.ndarray
    hover_metadata: list
    
    def validate(self, state: AppState) -> Optional[str]:
        """Validate arrays have correct shape."""
        if state.active_model is None:
            return "No model loaded"
        
        expected_samples = len(state.active_model.data.x_train)
        
        if len(self.latent) != expected_samples:
            return f"Latent shape mismatch: expected {expected_samples}, got {len(self.latent)}"
        if len(self.reconstructed) != expected_samples:
            return f"Reconstruction shape mismatch"
        if len(self.pred_classes) != expected_samples:
            return f"Predictions shape mismatch"
        
        return None
    
    def execute(self, state: AppState) -> Tuple[AppState, str]:
        """Update state with training results."""
        if state.active_model is None:
            return state, "No model loaded"
        
        from use_cases.dashboard.core.model_manager import ModelManager
        from datetime import datetime
        
        # Update model with predictions
        updated_model = state.active_model.with_training_complete(
            latent=self.latent,
            reconstructed=self.reconstructed,
            pred_classes=self.pred_classes,
            pred_certainty=self.pred_certainty,
            hover_metadata=self.hover_metadata
        )
        
        # Get latest loss from history
        if updated_model.history.val_loss:
            latest_loss = float(updated_model.history.val_loss[-1])
        else:
            latest_loss = None
        
        # Update metadata
        total_epochs = len(updated_model.history.epochs)
        updated_model = updated_model.with_updated_metadata(
            total_epochs=total_epochs,
            latest_loss=latest_loss,
            last_modified=datetime.utcnow().isoformat()
        )
        
        # Persist
        ModelManager.save_metadata(updated_model.metadata)
        ModelManager.save_history(updated_model.model_id, updated_model.history)
        
        # Update app state
        new_state = state.with_active_model(updated_model)
        
        return new_state, "Training complete"


@dataclass
class SelectSampleCommand(Command):
    """Command to select a sample in the UI."""
    sample_idx: int
    
    def validate(self, state: AppState) -> Optional[str]:
        """Validate sample index."""
        if state.active_model is None:
            return "No model loaded"
        if self.sample_idx < 0 or self.sample_idx >= len(state.active_model.data.x_train):
            return f"Invalid sample index: {self.sample_idx}"
        return None
    
    def execute(self, state: AppState) -> Tuple[AppState, str]:
        """Update selected sample."""
        if state.active_model is None:
            return state, "No model loaded"
        
        updated_model = state.active_model.with_ui(selected_sample=self.sample_idx)
        new_state = state.with_active_model(updated_model)
        return new_state, f"Selected sample {self.sample_idx}"


@dataclass
class ChangeColorModeCommand(Command):
    """Command to change visualization color mode."""
    color_mode: str
    
    def validate(self, state: AppState) -> Optional[str]:
        """Validate color mode."""
        if state.active_model is None:
            return "No model loaded"
        valid_modes = {"user_labels", "pred_class", "true_class", "certainty"}
        if self.color_mode not in valid_modes:
            return f"Invalid color mode: {self.color_mode}"
        return None
    
    def execute(self, state: AppState) -> Tuple[AppState, str]:
        """Update color mode."""
        if state.active_model is None:
            return state, "No model loaded"
        
        updated_model = state.active_model.with_ui(color_mode=self.color_mode)
        new_state = state.with_active_model(updated_model)
        return new_state, f"Color mode changed to {self.color_mode}"


@dataclass
class StopTrainingCommand(Command):
    """Command to stop ongoing training.
    
    Sets a flag that the training worker checks between epochs.
    Training will complete the current epoch and then halt gracefully.
    """
    
    def validate(self, state: AppState) -> Optional[str]:
        """Validate training is actually running."""
        if state.active_model is None:
            return "No model loaded"
        if not state.active_model.training.is_active():
            return "No training in progress to stop"
        if state.active_model.training.stop_requested:
            return "Stop already requested"
        return None
    
    def execute(self, state: AppState) -> Tuple[AppState, str]:
        """Set stop_requested flag."""
        if state.active_model is None:
            return state, "No model loaded"
        
        updated_model = state.active_model.with_training(
            stop_requested=True
        )
        new_state = state.with_active_model(updated_model)
        return new_state, "Training stop requested"


@dataclass
class UpdateModelConfigCommand(Command):
    """Update model configuration and persist changes to disk."""
    batch_size: Optional[int]
    max_epochs: Optional[int]
    patience: Optional[int]
    learning_rate: Optional[float]
    encoder_type: Optional[str]
    decoder_type: Optional[str]
    latent_dim: Optional[int]
    hidden_dims: Optional[str]
    recon_weight: Optional[float]
    kl_weight: Optional[float]
    label_weight: Optional[float]
    weight_decay: Optional[float]
    dropout_rate: Optional[float]
    grad_clip_norm: Optional[float]
    monitor_metric: Optional[str]
    use_contrastive: List[object]
    contrastive_weight: Optional[float]
    
    _normalized: Dict[str, object] = field(init=False, default_factory=dict)
    
    def validate(self, state: AppState) -> Optional[str]:
        """Validate configuration parameters."""
        if state.active_model is None:
            return "No model loaded"
        
        errors: List[str] = []
        
        def _int_in_range(value, name: str, minimum: int, maximum: int) -> Optional[int]:
            try:
                converted = int(value)
            except (TypeError, ValueError):
                errors.append(f"{name} must be between {minimum} and {maximum}")
                return None
            if converted < minimum or converted > maximum:
                errors.append(f"{name} must be between {minimum} and {maximum}")
                return None
            return converted
        
        def _float_in_range(value, name: str, minimum: float, maximum: float | None = None) -> Optional[float]:
            try:
                converted = float(value)
            except (TypeError, ValueError):
                errors.append(f"{name} must be a number")
                return None
            if converted < minimum or (maximum is not None and converted > maximum):
                max_clause = f" and at most {maximum}" if maximum is not None else ""
                errors.append(f"{name} must be at least {minimum}{max_clause}")
                return None
            return converted
        
        batch_size = _int_in_range(self.batch_size, "Batch size", 32, 2048)
        max_epochs = _int_in_range(self.max_epochs, "Max epochs", 1, 500)
        patience = _int_in_range(self.patience, "Patience", 1, 100)
        latent_dim = _int_in_range(self.latent_dim, "Latent dimension", 2, 512)
        
        learning_rate = _float_in_range(self.learning_rate, "Learning rate", 1e-5, 1e-1)
        recon_weight = _float_in_range(self.recon_weight, "Reconstruction weight", 0.0)
        kl_weight = _float_in_range(self.kl_weight, "KL weight", 0.0, 10.0)
        label_weight = _float_in_range(self.label_weight, "Label weight", 0.0)
        weight_decay = _float_in_range(self.weight_decay, "Weight decay", 0.0)
        dropout_rate = _float_in_range(self.dropout_rate, "Dropout rate", 0.0, 0.5)
        contrastive_weight = _float_in_range(self.contrastive_weight, "Contrastive weight", 0.0)
        
        if self.encoder_type not in {"dense", "conv"}:
            errors.append("Encoder type must be 'dense' or 'conv'")
        if self.decoder_type not in {"dense", "conv"}:
            errors.append("Decoder type must be 'dense' or 'conv'")
        if self.monitor_metric not in {"loss", "classification_loss"}:
            errors.append("Monitor metric must be 'loss' or 'classification_loss'")
        
        hidden_dims_tuple: Optional[Tuple[int, ...]] = None
        hidden_dims_str = self.hidden_dims or ""
        if not hidden_dims_str.strip():
            errors.append("Hidden dimensions must be specified")
        else:
            try:
                dims = [int(part.strip()) for part in hidden_dims_str.split(",") if part.strip()]
            except ValueError:
                dims = []
            if not dims or any(dim <= 0 for dim in dims):
                errors.append("Hidden dimensions must be positive integers (e.g., '256,128,64')")
            else:
                hidden_dims_tuple = tuple(dims)
        
        grad_clip_norm_value: Optional[float]
        if self.grad_clip_norm is None:
            grad_clip_norm_value = None
        else:
            norm_value = _float_in_range(self.grad_clip_norm, "Gradient clip norm", 0.0)
            if norm_value is None:
                grad_clip_norm_value = None
            elif norm_value == 0.0:
                grad_clip_norm_value = None
            else:
                grad_clip_norm_value = norm_value
        
        use_contrastive_bool = bool(self.use_contrastive)
        
        if errors:
            return "; ".join(errors)
        
        self._normalized = {
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "learning_rate": learning_rate,
            "encoder_type": self.encoder_type,
            "decoder_type": self.decoder_type,
            "latent_dim": latent_dim,
            "hidden_dims": hidden_dims_tuple,
            "recon_weight": recon_weight,
            "kl_weight": kl_weight,
            "label_weight": label_weight,
            "weight_decay": weight_decay,
            "dropout_rate": dropout_rate,
            "grad_clip_norm": grad_clip_norm_value,
            "monitor_metric": self.monitor_metric,
            "use_contrastive": use_contrastive_bool,
            "contrastive_weight": contrastive_weight,
        }
        return None
    
    def execute(self, state: AppState) -> Tuple[AppState, str]:
        """Apply configuration update and persist it."""
        if state.active_model is None:
            return state, "No model loaded"
        if not self._normalized:
            # Should not happen if dispatcher calls validate first.
            error = self.validate(state)
            if error:
                return state, error
        
        from dataclasses import replace as dc_replace
        from datetime import datetime
        from use_cases.dashboard.core.model_manager import ModelManager
        
        normalized = self._normalized
        current_model = state.active_model
        current_config = current_model.config
        
        new_config = dc_replace(
            current_config,
            batch_size=normalized["batch_size"],
            max_epochs=normalized["max_epochs"],
            patience=normalized["patience"],
            learning_rate=normalized["learning_rate"],
            encoder_type=normalized["encoder_type"],
            decoder_type=normalized["decoder_type"],
            latent_dim=normalized["latent_dim"],
            hidden_dims=normalized["hidden_dims"],
            recon_weight=normalized["recon_weight"],
            kl_weight=normalized["kl_weight"],
            label_weight=normalized["label_weight"],
            weight_decay=normalized["weight_decay"],
            dropout_rate=normalized["dropout_rate"],
            grad_clip_norm=normalized["grad_clip_norm"],
            monitor_metric=normalized["monitor_metric"],
            use_contrastive=normalized["use_contrastive"],
            contrastive_weight=normalized["contrastive_weight"],
        )
        
        architecture_changed = any([
            current_config.encoder_type != new_config.encoder_type,
            current_config.decoder_type != new_config.decoder_type,
            current_config.latent_dim != new_config.latent_dim,
            current_config.hidden_dims != new_config.hidden_dims,
        ])
        
        # Update live objects
        model = current_model.model
        trainer = current_model.trainer
        model.config = new_config
        trainer.config = new_config
        if hasattr(trainer, "_trainer"):
            trainer._trainer.config = new_config  # type: ignore[attr-defined]
        
        updated_model = current_model.with_config(new_config)
        updated_model = updated_model.with_updated_metadata(
            last_modified=datetime.utcnow().isoformat()
        )
        
        # Persist changes
        ModelManager.save_config(current_model.model_id, new_config)
        ModelManager.save_metadata(updated_model.metadata)
        
        new_state = state.with_active_model(updated_model)
        if architecture_changed:
            message = (
                "Configuration saved. Architecture changes require restarting the dashboard."
            )
        else:
            message = "Configuration updated successfully."
        return new_state, message


# ============================================================================
# Model Management Commands (Multi-Model Support)
# ============================================================================

import pandas as pd


@dataclass
class CreateModelCommand(Command):
    """Create a new model with fresh state."""
    name: Optional[str] = None  # User-friendly name (optional)
    config_preset: str = "default"  # "default", "high_recon", "classification"
    
    def validate(self, state: AppState) -> Optional[str]:
        """Validate preset exists."""
        valid_presets = {"default", "high_recon", "classification"}
        if self.config_preset not in valid_presets:
            return f"Invalid preset: {self.config_preset}. Must be one of {valid_presets}"
        return None
    
    def execute(self, state: AppState) -> Tuple[AppState, str]:
        """Create new model directory and metadata."""
        from use_cases.dashboard.core.model_manager import ModelManager
        from use_cases.dashboard.core.state_models import ModelMetadata
        from datetime import datetime
        from use_cases.dashboard.core import state as dashboard_state
        from rcmvae.application.model_api import SSVAE
        from rcmvae.domain.config import SSVAEConfig
        import re
        
        # Use the provided name as model_id (sanitized)
        # If no name provided, generate a default one
        if self.name and self.name.strip():
            # Sanitize name: lowercase, replace spaces/special chars with underscores
            sanitized = re.sub(r'[^a-z0-9_-]', '_', self.name.strip().lower())
            # Remove consecutive underscores
            sanitized = re.sub(r'_+', '_', sanitized)
            # Remove leading/trailing underscores
            model_id = sanitized.strip('_')
            
            # Ensure unique (append number if exists)
            base_id = model_id
            counter = 1
            while (ModelManager.model_dir(model_id)).exists():
                model_id = f"{base_id}_{counter}"
                counter += 1
        else:
            # Generate sequential ID
            model_id = ModelManager.generate_model_id()
        
        display_name = self.name.strip() if self.name and self.name.strip() else model_id
        
        # Create directory
        ModelManager.create_model_directory(model_id)
        
        # Create config based on preset
        config = SSVAEConfig()
        if self.config_preset == "high_recon":
            config.recon_weight = 5000.0
            config.kl_weight = 0.01
        elif self.config_preset == "classification":
            config.label_weight = 10.0
            config.recon_weight = 500.0
        
        # Create metadata
        now = datetime.utcnow().isoformat()
        metadata = ModelMetadata(
            model_id=model_id,
            name=display_name,
            created_at=now,
            last_modified=now,
            dataset="mnist",
            total_epochs=0,
            labeled_count=0,
            latest_loss=None
        )
        
        # Save files
        ModelManager.save_metadata(metadata)
        from use_cases.dashboard.core.state_models import TrainingHistory
        ModelManager.save_history(model_id, TrainingHistory.empty())
        ModelManager.save_config(model_id, config)
        
        # Note: Don't save model weights yet - they'll be saved after first training
        # The model directory and metadata are created, that's enough for now
        
        # Create empty labels.csv
        labels_path = ModelManager.labels_path(model_id)
        pd.DataFrame(columns=["Serial", "label"]).to_csv(labels_path, index=False)
        
        # Update registry
        new_state = state.with_model_metadata(metadata)
        
        # Note: Callback should call LoadModelCommand after this succeeds
        # to actually load the model as active
        
        return new_state, model_id  # Return model_id so callback can load it


@dataclass
class LoadModelCommand(Command):
    """Load a model as active."""
    model_id: str
    
    def validate(self, state: AppState) -> Optional[str]:
        """Check model exists."""
        if self.model_id not in state.models:
            return f"Model not found: {self.model_id}"
        
        # Allow reloading same model (no-op is fine)
        return None
    
    def execute(self, state: AppState) -> Tuple[AppState, str]:
        """Load model into active state."""
        # Check if already loaded
        if state.active_model and state.active_model.model_id == self.model_id:
            return state, f"Model {self.model_id} already active"
        
        # Load model components (we're already inside state_lock from dispatcher)
        from use_cases.dashboard.core.model_manager import ModelManager
        from use_cases.dashboard.core.state_models import (
            ModelState,
            DataState,
            TrainingStatus,
            TrainingState,
            UIState,
        )
        from use_cases.dashboard.utils.visualization import _build_hover_metadata
        from data.mnist.mnist import load_train_images_for_ssvae, load_mnist_splits
        from rcmvae.application.model_api import SSVAE
        from rcmvae.domain.config import SSVAEConfig
        from rcmvae.application.runtime.interactive import InteractiveTrainer
        import pandas as pd
        import numpy as np
        
        # Load metadata
        metadata = ModelManager.load_metadata(self.model_id)
        if not metadata:
            raise ValueError(f"Model {self.model_id} not found")
        
        # Load config
        config = ModelManager.load_config(self.model_id) or SSVAEConfig()
        
        # Load model
        model = SSVAE(input_dim=(28, 28), config=config)
        checkpoint_path = ModelManager.checkpoint_path(self.model_id)
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
        
        # Load history
        history = ModelManager.load_history(self.model_id)
        
        # Load labels
        labels_array = np.full(shape=(x_train.shape[0],), fill_value=np.nan, dtype=float)
        labels_path = ModelManager.labels_path(self.model_id)
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
            model_id=self.model_id,
            metadata=metadata,
            model=model,
            trainer=trainer,
            config=model.config,
            data=data_state,
            training=training_status,
            ui=ui_state,
            history=history
        )
        
        # Update state
        new_state = state.with_active_model(model_state)
        return new_state, f"Loaded model: {self.model_id}"


@dataclass
class DeleteModelCommand(Command):
    """Delete a model permanently."""
    model_id: str
    
    def validate(self, state: AppState) -> Optional[str]:
        """Check model exists and not active."""
        if self.model_id not in state.models:
            return f"Model not found: {self.model_id}"
        
        if state.active_model and state.active_model.model_id == self.model_id:
            return "Cannot delete active model. Switch to another model first."
        
        return None
    
    def execute(self, state: AppState) -> Tuple[AppState, str]:
        """Delete model files and remove from registry."""
        from use_cases.dashboard.core.model_manager import ModelManager
        from dataclasses import replace
        from use_cases.dashboard.core import state as dashboard_state

        logger.info(f"DELETING MODEL: model_id={self.model_id}")
        logger.debug(f"Available models before delete: {list(state.models.keys())}")

        # Double-check active model guard in case state changed between validate+execute
        active = dashboard_state.app_state.active_model if dashboard_state.app_state else None
        if active and active.model_id == self.model_id:
            raise ValueError("Cannot delete active model. Switch to another model first.")

        # Get name for message
        model_name = state.models[self.model_id].name
        
        # Delete files
        logger.info(f"Deleting files for model: {self.model_id}")
        ModelManager.delete_model(self.model_id)
        logger.info(f"Files deleted successfully for: {self.model_id}")
        
        # Remove from registry
        updated_models = dict(state.models)
        del updated_models[self.model_id]
        new_state = replace(state, models=updated_models)
        
        logger.info(f"Model {self.model_id} removed from registry")
        logger.debug(f"Available models after delete: {list(updated_models.keys())}")
        
        return new_state, f"Deleted model: {model_name}"
