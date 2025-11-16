"""Command pattern infrastructure for state-modifying actions."""

from __future__ import annotations

import time
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import threading
import os

import numpy as np

from rcmvae.domain.config import SSVAEConfig

from use_cases.dashboard.core.state_models import AppState
from use_cases.dashboard.core.logging_config import get_logger
from use_cases.dashboard.core.run_generation import generate_dashboard_run
from use_cases.dashboard.core.model_runs import append_run_record, load_run_records

logger = get_logger('commands')
PREVIEW_SAMPLE_LIMIT = 2048
FAST_DASHBOARD_MODE = os.environ.get("DASHBOARD_FAST_MODE", "1").lower() not in {"0", "false", "no"}


class Command(ABC):
    """Base class for all state-modifying commands.

    Commands encapsulate both validation and execution logic.
    This makes actions testable, auditable, and explicit.

    Phase 2 Update: Commands now receive services for domain operations.
    """

    @abstractmethod
    def validate(self, state: AppState, services: Optional[Any] = None) -> Optional[str]:
        """Validate if command can execute on current state.

        Args:
            state: Current application state (read-only)
            services: Optional service container for domain operations

        Returns:
            Error message if invalid, None if valid
        """
        pass

    @abstractmethod
    def execute(self, state: AppState, services: Optional[Any] = None) -> Tuple[AppState, str]:
        """Execute command, producing new state.

        Args:
            state: Current application state (read-only)
            services: Optional service container for domain operations

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
    - Service injection for domain operations (Phase 2)
    """

    def __init__(self, state_lock: threading.Lock | threading.RLock, services: Optional[Any] = None):
        """Initialize dispatcher with state lock and optional services.

        Args:
            state_lock: Lock or RLock protecting app_state access
            services: Optional service container for dependency injection (Phase 2)
        """
        self._state_lock = state_lock
        self._services = services
        self._command_history: List[CommandHistoryEntry] = []
        self._history_lock = threading.Lock()
    
    def execute(self, command: Command) -> Tuple[bool, str]:
        """Execute command atomically with service injection.

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

            # Validate (with services)
            error = command.validate(dashboard_state.app_state, self._services)
            if error:
                self._log_command(command, success=False, message=error)
                return False, error

            # Execute (with services)
            try:
                new_state, message = command.execute(dashboard_state.app_state, self._services)
                dashboard_state.app_state = new_state
                self._log_command(command, success=True, message=message)
                return True, message

            except Exception as e:
                error_msg = f"Command execution failed: {e}"
                self._log_command(command, success=False, message=error_msg)
                logger.error("Command execution failed", exc_info=True)
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
    
    def validate(self, state: AppState, services: Optional[Any] = None) -> Optional[str]:
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
    
    def execute(self, state: AppState, services: Optional[Any] = None) -> Tuple[AppState, str]:
        """Execute label update on ACTIVE model.

        Phase 2: Uses LabelingService when available for persistence.
        """
        if state.active_model is None:
            return state, "No model loaded"

        from use_cases.dashboard.core.model_manager import ModelManager

        # Update labels array (in-memory)
        labels_array = state.active_model.data.labels.copy()
        if self.label is None:
            labels_array[self.sample_idx] = np.nan
        else:
            labels_array[self.sample_idx] = float(self.label)

        # Persist label via service (if available) or fallback to direct CSV
        if services is not None and hasattr(services, 'labeling'):
            try:
                services.labeling.set_label(
                    model_id=state.active_model.model_id,
                    sample_idx=self.sample_idx,
                    label=self.label,
                )
            except ValueError as e:
                return state, f"Label update failed: {e}"
        else:
            # Fallback: direct CSV update (backward compatibility)
            from use_cases.dashboard.core.state import _load_labels_dataframe, _persist_labels_dataframe
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

        # Get labeled count via service (if available) or calculate
        if services is not None and hasattr(services, 'labeling'):
            labeled_count = services.labeling.get_labeled_count(state.active_model.model_id)
        else:
            labeled_count = int(np.sum(~np.isnan(labels_array)))

        # Update metadata
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
    
    def validate(self, state: AppState, services: Optional[Any] = None) -> Optional[str]:
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
    
    def execute(self, state: AppState, services: Optional[Any] = None) -> Tuple[AppState, str]:
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
    """Command to mark training as complete with new predictions."""

    latent: np.ndarray
    reconstructed: np.ndarray | Tuple[np.ndarray, np.ndarray]
    pred_classes: np.ndarray
    pred_certainty: np.ndarray
    hover_metadata: list
    responsibilities: Optional[np.ndarray] = None
    pi_values: Optional[np.ndarray] = None
    train_time: Optional[float] = None
    epoch_offset: int = 0
    epochs_completed: int = 0

    def validate(self, state: AppState, services: Optional[Any] = None) -> Optional[str]:
        """Validate arrays have correct shape."""
        if state.active_model is None:
            return "No model loaded"

        expected_samples = len(state.active_model.data.x_train)

        if len(self.latent) != expected_samples:
            return f"Latent shape mismatch: expected {expected_samples}, got {len(self.latent)}"
        if isinstance(self.reconstructed, tuple):
            if len(self.reconstructed[0]) != expected_samples:
                return "Reconstruction shape mismatch"
        elif len(self.reconstructed) != expected_samples:
            return "Reconstruction shape mismatch"
        if len(self.pred_classes) != expected_samples:
            return "Predictions shape mismatch"
        if self.responsibilities is not None and len(self.responsibilities) != expected_samples:
            return "Responsibilities shape mismatch"
        if self.epoch_offset < 0 or self.epochs_completed < 0:
            return "Epoch counters must be non-negative"

        return None

    def execute(self, state: AppState, services: Optional[Any] = None) -> Tuple[AppState, str]:
        """Update state with training results."""
        if state.active_model is None:
            return state, "No model loaded"

        from use_cases.dashboard.core.model_manager import ModelManager
        from datetime import datetime

        label_version = int(state.active_model.data.version)
        start_epoch = self.epoch_offset + 1 if self.epochs_completed > 0 else self.epoch_offset
        end_epoch = self.epoch_offset + self.epochs_completed

        # Update model with predictions
        updated_model = state.active_model.with_training_complete(
            latent=self.latent,
            reconstructed=self.reconstructed,
            pred_classes=self.pred_classes,
            pred_certainty=self.pred_certainty,
            hover_metadata=self.hover_metadata,
        )

        total_epochs = len(updated_model.history.epochs)
        latest_val_loss = float(updated_model.history.val_loss[-1]) if updated_model.history.val_loss else None
        latest_train_loss = float(updated_model.history.train_loss[-1]) if updated_model.history.train_loss else None
        latest_loss = latest_val_loss if latest_val_loss is not None else latest_train_loss
        updated_metadata_fields = {
            "total_epochs": total_epochs,
            "last_modified": datetime.utcnow().isoformat(),
        }
        if latest_loss is not None:
            updated_metadata_fields["latest_loss"] = latest_loss
        updated_model = updated_model.with_updated_metadata(**updated_metadata_fields)

        # Persist core model artifacts
        ModelManager.save_metadata(updated_model.metadata)
        ModelManager.save_history(updated_model.model_id, updated_model.history)

        # Update app state immediately
        new_state = state.with_active_model(updated_model)

        # Persist experiment run bundle (best-effort)
        artifact_note = ""
        try:
            record = generate_dashboard_run(
                model_state=updated_model,
                latent=self.latent,
                reconstructed=self.reconstructed,
                pred_classes=self.pred_classes,
                pred_certainty=self.pred_certainty,
                responsibilities=self.responsibilities,
                pi_values=self.pi_values,
                train_time=self.train_time,
                label_version=label_version,
                epoch_offset=self.epoch_offset,
                epochs_completed=self.epochs_completed,
            )
            if record and record.get("run_id"):
                record.setdefault("start_epoch", start_epoch)
                record.setdefault("end_epoch", end_epoch)
                record.setdefault("epochs_completed", self.epochs_completed)
                runs = append_run_record(updated_model.model_id, record)
                active_model = new_state.active_model
                if active_model is not None:
                    new_state = new_state.with_active_model(active_model.with_runs(runs))
                if self.epochs_completed > 0:
                    range_desc = f"{start_epoch}→{end_epoch}"
                else:
                    range_desc = "no epochs"
                artifact_note = f"; run captured as {record['run_id']} ({range_desc})"
        except Exception as exc:  # pragma: no cover - diagnostics only
            logger.exception("Failed to generate dashboard run artifacts", exc_info=exc)
            artifact_note = f"; artifacts not captured ({exc})"

        return new_state, f"Training complete{artifact_note}"


@dataclass
class SelectSampleCommand(Command):
    """Command to select a sample in the UI."""
    sample_idx: int
    
    def validate(self, state: AppState, services: Optional[Any] = None) -> Optional[str]:
        """Validate sample index."""
        if state.active_model is None:
            return "No model loaded"
        if self.sample_idx < 0 or self.sample_idx >= len(state.active_model.data.x_train):
            return f"Invalid sample index: {self.sample_idx}"
        return None
    
    def execute(self, state: AppState, services: Optional[Any] = None) -> Tuple[AppState, str]:
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
    
    def validate(self, state: AppState, services: Optional[Any] = None) -> Optional[str]:
        """Validate color mode."""
        if state.active_model is None:
            return "No model loaded"
        valid_modes = {"user_labels", "pred_class", "true_class", "certainty"}
        if self.color_mode not in valid_modes:
            return f"Invalid color mode: {self.color_mode}"
        return None
    
    def execute(self, state: AppState, services: Optional[Any] = None) -> Tuple[AppState, str]:
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
    
    def validate(self, state: AppState, services: Optional[Any] = None) -> Optional[str]:
        """Validate training is actually running."""
        if state.active_model is None:
            return "No model loaded"
        if not state.active_model.training.is_active():
            return "No training in progress to stop"
        if state.active_model.training.stop_requested:
            return "Stop already requested"
        return None
    
    def execute(self, state: AppState, services: Optional[Any] = None) -> Tuple[AppState, str]:
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

    updates: Dict[str, Any]
    _new_config: Optional[SSVAEConfig] = field(init=False, default=None)

    def validate(self, state: AppState, services: Optional[Any] = None) -> Optional[str]:
        if state.active_model is None:
            return "No model loaded"
        if not self.updates:
            return "No configuration changes detected."

        current_config = state.active_model.config
        config_data = {
            name: getattr(current_config, name)
            for name in current_config.__dataclass_fields__.keys()
        }
        config_data.update(self.updates)

        try:
            self._new_config = SSVAEConfig(**config_data)
        except Exception as exc:  # pragma: no cover - relying on config validation
            return str(exc)

        return None

    def execute(self, state: AppState, services: Optional[Any] = None) -> Tuple[AppState, str]:
        if state.active_model is None:
            return state, "No model loaded"
        if not self.updates:
            return state, "No configuration changes detected."
        if self._new_config is None:
            error = self.validate(state)
            if error:
                return state, error

        assert self._new_config is not None  # for type checkers

        from datetime import datetime
        from use_cases.dashboard.core.model_manager import ModelManager

        current_model = state.active_model
        current_config = current_model.config
        new_config = self._new_config

        architecture_fields = {
            "encoder_type",
            "decoder_type",
            "latent_dim",
            "hidden_dims",
            "prior_type",
            "num_components",
            "component_embedding_dim",
            "use_component_aware_decoder",
            "use_heteroscedastic_decoder",
            "reconstruction_loss",
        }
        architecture_changed = any(
            getattr(current_config, field) != getattr(new_config, field)
            for field in architecture_fields
            if hasattr(current_config, field)
        )

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

        ModelManager.save_config(current_model.model_id, new_config)
        ModelManager.save_metadata(updated_model.metadata)

        new_state = state.with_active_model(updated_model)
        if architecture_changed:
            message = (
                "Configuration saved. Structural changes require restarting the dashboard."
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
    num_samples: int = 1024
    num_labeled: int = 128
    seed: Optional[int] = None
    
    def validate(self, state: AppState, services: Optional[Any] = None) -> Optional[str]:
        """Validate dataset sizing inputs."""
        errors: List[str] = []
        total = self.num_samples
        labeled = self.num_labeled
        if total <= 0:
            errors.append("Total samples must be greater than zero")
        if labeled < 0:
            errors.append("Labeled samples must be non-negative")
        if total > 70000:
            errors.append("Total samples must be at most 70000 for MNIST")
        if labeled > total:
            errors.append("Labeled samples cannot exceed total samples")
        if errors:
            return "; ".join(errors)
        return None
    
    def execute(self, state: AppState, services: Optional[Any] = None) -> Tuple[AppState, str]:
        """Create new model directory, dataset snapshot, and metadata."""
        from use_cases.dashboard.core.model_manager import ModelManager
        from use_cases.dashboard.core.state_models import ModelMetadata, TrainingHistory
        from datetime import datetime
        from rcmvae.domain.config import SSVAEConfig
        from use_cases.experiments.data.mnist.mnist import load_mnist_scaled
        import numpy as np
        import pandas as pd
        import re
        
        # Use the provided name as model_id (sanitized)
        if self.name and self.name.strip():
            sanitized = re.sub(r'[^a-z0-9_-]', '_', self.name.strip().lower())
            sanitized = re.sub(r'_+', '_', sanitized)
            model_id = sanitized.strip('_') or "model"
            base_id = model_id
            counter = 1
            while ModelManager.model_dir(model_id).exists():
                model_id = f"{base_id}_{counter}"
                counter += 1
        else:
            model_id = ModelManager.generate_model_id()
        
        display_name = self.name.strip() if self.name and self.name.strip() else model_id
        ModelManager.create_model_directory(model_id)
        
        # Base configuration (default SSVAE config for dashboard usage)
        config = SSVAEConfig()
        
        # Snapshot dataset configuration
        total_samples = int(self.num_samples)
        labeled_samples = int(self.num_labeled)
        rng_seed = int(self.seed if self.seed is not None else 0)
        rng = np.random.default_rng(rng_seed)
        
        x_full, y_full, _, _, source = load_mnist_scaled(
            reshape=True,
            hw=(28, 28),
            dtype=np.float32,
        )
        max_available = x_full.shape[0]
        if total_samples > max_available:
            total_samples = max_available
        selected_indices = rng.choice(max_available, size=total_samples, replace=False)
        selected_indices = selected_indices.tolist()
        labeled_count = min(labeled_samples, total_samples)
        if labeled_count > 0:
            labeled_positions = sorted(rng.choice(total_samples, size=labeled_count, replace=False).tolist())
        else:
            labeled_positions = []
        
        # Persist dataset config for reproducible loading
        dataset_config = {
            "dataset": "mnist",
            "source": source,
            "indices": selected_indices,
            "labeled_positions": labeled_positions,
            "seed": rng_seed,
            "total_samples": total_samples,
            "labeled_samples": labeled_count,
        }
        ModelManager.save_dataset_config(model_id, dataset_config)
        
        # Pre-populate labels.csv with labeled subset
        labels_path = ModelManager.labels_path(model_id)
        if labeled_positions:
            selected_labels = y_full[selected_indices]
            rows = [
                {"Serial": int(pos), "label": int(selected_labels[pos])}
                for pos in labeled_positions
            ]
            df = pd.DataFrame(rows, columns=["Serial", "label"])
        else:
            df = pd.DataFrame(columns=["Serial", "label"])
        df.to_csv(labels_path, index=False)
        
        # Save metadata and auxiliary files
        now = datetime.utcnow().isoformat()
        metadata = ModelMetadata(
            model_id=model_id,
            name=display_name,
            created_at=now,
            last_modified=now,
            dataset="mnist",
            total_epochs=0,
            labeled_count=labeled_count,
            latest_loss=None,
            dataset_total_samples=total_samples,
            dataset_seed=rng_seed,
        )
        ModelManager.save_metadata(metadata)
        ModelManager.save_history(model_id, TrainingHistory.empty())
        ModelManager.save_config(model_id, config)
        
        # Update registry with new metadata
        new_state = state.with_model_metadata(metadata)
        
        return new_state, model_id


@dataclass
class LoadModelCommand(Command):
    """Load a model as active."""
    model_id: str
    
    def validate(self, state: AppState, services: Optional[Any] = None) -> Optional[str]:
        """Check model exists."""
        if self.model_id not in state.models:
            return f"Model not found: {self.model_id}"
        
        # Prevent switching while current model is still training
        if state.active_model and state.active_model.training.is_active():
            active_id = state.active_model.model_id
            return (
                "Training is currently running for "
                f"model '{active_id}'. Please wait for it to finish or stop the run "
                "before switching models."
            )

        # Allow reloading same model (no-op is fine)
        return None
    
    def execute(self, state: AppState, services: Optional[Any] = None) -> Tuple[AppState, str]:
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
        from use_cases.experiments.data.mnist.mnist import load_mnist_scaled
        from rcmvae.application.model_api import SSVAE
        from rcmvae.domain.config import SSVAEConfig
        from rcmvae.application.runtime.interactive import InteractiveTrainer
        import pandas as pd
        import numpy as np
        from dataclasses import replace as dc_replace
        from use_cases.dashboard.core import state as dashboard_state
        
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
        
        dataset_config = ModelManager.load_dataset_config(self.model_id)

        if dataset_config:
            indices = np.asarray(dataset_config.get("indices", []), dtype=np.int64)
            if indices.size == 0:
                raise ValueError("Dataset configuration has no indices")
            x_full, y_full, _, _, _ = load_mnist_scaled(
                reshape=True,
                hw=(28, 28),
                dtype=np.float32,
            )
            max_available = x_full.shape[0]
            if np.any(indices >= max_available):
                raise ValueError("Dataset indices exceed available MNIST samples")
            x_train = x_full[indices]
            true_labels = y_full[indices].astype(np.int32)
        else:
            # Backward compatibility: fall back to preview dataset
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
        
        # Load history and prior runs
        history = ModelManager.load_history(self.model_id)
        run_records = load_run_records(self.model_id)
        
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

        labeled_count = int(np.sum(~np.isnan(labels_array)))
        metadata = dc_replace(
            metadata,
            labeled_count=labeled_count,
            dataset_total_samples=x_train.shape[0],
        )
        
        # Get predictions
        if dataset_config:
            latent, recon, pred_classes, pred_certainty = model.predict(x_train)
        elif FAST_DASHBOARD_MODE:
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
            history=history,
            runs=tuple(run_records)
        )
        
        # Reset any leftover training metrics from previous model
        dashboard_state._clear_metrics_queue()

        # Update state
        new_state = state.with_active_model(model_state)
        return new_state, f"Loaded model: {self.model_id}"


@dataclass
class DeleteModelCommand(Command):
    """Delete a model permanently."""
    model_id: str
    
    def validate(self, state: AppState, services: Optional[Any] = None) -> Optional[str]:
        """Check model exists and not active."""
        if self.model_id not in state.models:
            return f"Model not found: {self.model_id}"
        
        if state.active_model and state.active_model.model_id == self.model_id:
            return "Cannot delete active model. Switch to another model first."
        
        return None
    
    def execute(self, state: AppState, services: Optional[Any] = None) -> Tuple[AppState, str]:
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
