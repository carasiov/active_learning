"""Command pattern infrastructure for state-modifying actions."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import threading

from use_cases.dashboard.state_models import AppState


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
    
    def __init__(self, state_lock: threading.Lock):
        """Initialize dispatcher with state lock.
        
        Args:
            state_lock: Lock protecting app_state access
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
        from use_cases.dashboard import state as dashboard_state
        
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
        # Check sample index bounds
        if self.sample_idx < 0 or self.sample_idx >= len(state.data.labels):
            return f"Invalid sample index: {self.sample_idx}"
        
        # Check label value (if not deletion)
        if self.label is not None:
            if not isinstance(self.label, int):
                return f"Label must be an integer, got {type(self.label)}"
            if not (0 <= self.label <= 9):
                return f"Label must be 0-9, got {self.label}"
        
        return None  # Valid
    
    def execute(self, state: AppState) -> Tuple[AppState, str]:
        """Execute label update."""
        # Import here to avoid circular dependency
        from use_cases.dashboard.state import _load_labels_dataframe, _persist_labels_dataframe
        
        # Copy labels array and update
        labels_array = state.data.labels.copy()
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
        from use_cases.dashboard.utils import _format_hover_metadata_entry
        hover_metadata = list(state.data.hover_metadata)
        true_label_value = int(state.data.true_labels[self.sample_idx])
        hover_metadata[self.sample_idx] = _format_hover_metadata_entry(
            self.sample_idx,
            int(state.data.pred_classes[self.sample_idx]),
            float(state.data.pred_certainty[self.sample_idx]),
            float(labels_array[self.sample_idx]),
            true_label_value,
        )
        
        # Create new state with updates
        new_state = state.with_label_update(labels_array, hover_metadata)
        
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
        # Check not already training
        if state.training.is_active():
            return "Training already in progress"
        
        # Validate epochs
        if self.num_epochs < 1 or self.num_epochs > 200:
            return f"Epochs must be between 1 and 200, got {self.num_epochs}"
        
        # Check has labeled samples
        labeled_count = int(np.sum(~np.isnan(state.data.labels)))
        if labeled_count == 0:
            return "No labeled samples available for training"
        
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
        # Update config (mutable for now - configs remain mutable)
        state.config.recon_weight = float(self.recon_weight)
        state.config.kl_weight = float(self.kl_weight)
        state.config.learning_rate = float(self.learning_rate)
        
        # Update model config
        state.model.config.recon_weight = float(self.recon_weight)
        state.model.config.kl_weight = float(self.kl_weight)
        state.model.config.learning_rate = float(self.learning_rate)
        
        # Update trainer config
        state.trainer.config.recon_weight = float(self.recon_weight)
        state.trainer.config.kl_weight = float(self.kl_weight)
        state.trainer.config.learning_rate = float(self.learning_rate)
        
        # Transition to QUEUED state
        from dataclasses import replace
        new_state = replace(
            state,
            training=state.training.with_queued(self.num_epochs)
        )
        
        message = f"Queued training for {self.num_epochs} epoch(s)"
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
        expected_samples = len(state.data.x_train)
        
        if len(self.latent) != expected_samples:
            return f"Latent shape mismatch: expected {expected_samples}, got {len(self.latent)}"
        if len(self.reconstructed) != expected_samples:
            return f"Reconstruction shape mismatch"
        if len(self.pred_classes) != expected_samples:
            return f"Predictions shape mismatch"
        
        return None
    
    def execute(self, state: AppState) -> Tuple[AppState, str]:
        """Update state with training results."""
        new_state = state.with_training_complete(
            latent=self.latent,
            reconstructed=self.reconstructed,
            pred_classes=self.pred_classes,
            pred_certainty=self.pred_certainty,
            hover_metadata=self.hover_metadata
        )
        
        return new_state, "Training complete"


@dataclass
class SelectSampleCommand(Command):
    """Command to select a sample in the UI."""
    sample_idx: int
    
    def validate(self, state: AppState) -> Optional[str]:
        """Validate sample index."""
        if self.sample_idx < 0 or self.sample_idx >= len(state.data.x_train):
            return f"Invalid sample index: {self.sample_idx}"
        return None
    
    def execute(self, state: AppState) -> Tuple[AppState, str]:
        """Update selected sample."""
        new_state = state.with_ui(selected_sample=self.sample_idx)
        return new_state, f"Selected sample {self.sample_idx}"


@dataclass
class ChangeColorModeCommand(Command):
    """Command to change visualization color mode."""
    color_mode: str
    
    def validate(self, state: AppState) -> Optional[str]:
        """Validate color mode."""
        valid_modes = {"user_labels", "pred_class", "true_class", "certainty"}
        if self.color_mode not in valid_modes:
            return f"Invalid color mode: {self.color_mode}"
        return None
    
    def execute(self, state: AppState) -> Tuple[AppState, str]:
        """Update color mode."""
        new_state = state.with_ui(color_mode=self.color_mode)
        return new_state, f"Color mode changed to {self.color_mode}"


@dataclass
class StopTrainingCommand(Command):
    """Command to stop ongoing training.
    
    Sets a flag that the training worker checks between epochs.
    Training will complete the current epoch and then halt gracefully.
    """
    
    def validate(self, state: AppState) -> Optional[str]:
        """Validate training is actually running."""
        if not state.training.is_active():
            return "No training in progress to stop"
        if state.training.stop_requested:
            return "Stop already requested"
        return None
    
    def execute(self, state: AppState) -> Tuple[AppState, str]:
        """Set stop_requested flag."""
        from dataclasses import replace
        new_state = replace(
            state,
            training=state.training.with_stop_requested()
        )
        return new_state, "Training stop requested"
