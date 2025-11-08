"""Callback for tracking mixture prior dynamics during training."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict

import jax.numpy as jnp
import numpy as np

from callbacks.base_callback import TrainingCallback

if TYPE_CHECKING:
    from training.trainer import Trainer

    HistoryDict = Dict[str, list]
    MetricsDict = Dict[str, float]


class MixtureHistoryTracker(TrainingCallback):
    """Tracks π (mixture weights) and component usage evolution during training.

    This callback periodically saves:
    - π values (mixture weights) at each tracked epoch
    - Component usage (mean responsibilities) at each tracked epoch

    Args:
        output_dir: Directory to save history files
        log_every: Track every N epochs (default: 1 = every epoch)
        enabled: Whether tracking is enabled (default: True)
    """

    def __init__(
        self,
        output_dir: str | Path,
        log_every: int = 1,
        enabled: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.log_every = log_every
        self.enabled = enabled

        # History storage
        self.pi_history: list[np.ndarray] = []
        self.usage_history: list[np.ndarray] = []
        self.tracked_epochs: list[int] = []

    def on_train_start(self, trainer: "Trainer") -> None:
        """Called once before training begins."""
        if not self.enabled:
            return

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Reset history
        self.pi_history = []
        self.usage_history = []
        self.tracked_epochs = []

    def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, MetricsDict],
        history: HistoryDict,
        trainer: "Trainer",
    ) -> None:
        """Track π and usage at specified intervals."""
        if not self.enabled:
            return

        # Check if we should log this epoch
        if (epoch % self.log_every) != 0:
            return

        # Extract data splits
        splits = trainer.latest_splits
        if splits is None or splits.x_train.size == 0:
            return

        # Get model state
        state = getattr(trainer, '_state', None)
        if state is None:
            return

        # Use a small subset for efficiency (max 5000 samples)
        x_train = splits.x_train
        max_samples = 5000
        if x_train.shape[0] > max_samples:
            indices = np.random.choice(x_train.shape[0], size=max_samples, replace=False)
            x_train = x_train[indices]

        # Forward pass to get responsibilities
        try:
            forward_output = state.apply_fn(state.params, x_train, training=False)
            component_logits, _, _, _, _, _, extras = forward_output

            # Extract responsibilities and π
            responsibilities = extras.get("responsibilities") if hasattr(extras, "get") else None
            pi = extras.get("pi") if hasattr(extras, "get") else None

            if responsibilities is not None:
                # Compute mean usage across samples
                usage = np.asarray(responsibilities).mean(axis=0)
                self.usage_history.append(usage)

            if pi is not None:
                pi_np = np.asarray(pi)
                self.pi_history.append(pi_np)

            self.tracked_epochs.append(epoch)

        except Exception as e:
            print(f"Warning: Failed to track mixture history at epoch {epoch}: {e}")

    def on_train_end(self, history: HistoryDict, trainer: "Trainer") -> None:
        """Save tracked histories to disk."""
        if not self.enabled or not self.tracked_epochs:
            return

        try:
            # Save π history
            if self.pi_history:
                pi_array = np.stack(self.pi_history, axis=0)  # Shape: (n_tracked_epochs, K)
                np.save(self.output_dir / "pi_history.npy", pi_array)

            # Save usage history
            if self.usage_history:
                usage_array = np.stack(self.usage_history, axis=0)  # Shape: (n_tracked_epochs, K)
                np.save(self.output_dir / "usage_history.npy", usage_array)

            # Save epochs array
            epochs_array = np.array(self.tracked_epochs, dtype=np.int32)
            np.save(self.output_dir / "tracked_epochs.npy", epochs_array)

            print(f"Mixture history saved to {self.output_dir} ({len(self.tracked_epochs)} epochs tracked)")

        except Exception as e:
            print(f"Warning: Failed to save mixture history: {e}")
