"""Callback for tracking mixture prior dynamics during training."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict

import jax.numpy as jnp
import numpy as np

from .base import TrainingCallback

if TYPE_CHECKING:  # pragma: no cover
    from rcmvae.application.services.training_service import Trainer

    HistoryDict = Dict[str, list]
    MetricsDict = Dict[str, float]


class MixtureHistoryTracker(TrainingCallback):
    """Tracks π and component usage evolution during training."""

    def __init__(
        self,
        output_dir: str | Path,
        log_every: int = 1,
        enabled: bool = True,
        max_samples: int = 5000,
        eval_batch_size: int = 256,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.log_every = log_every
        self.enabled = enabled
        self.max_samples = max_samples
        self.eval_batch_size = eval_batch_size

        self.pi_history: list[np.ndarray] = []
        self.usage_history: list[np.ndarray] = []
        self.tracked_epochs: list[int] = []

    def on_train_start(self, trainer: "Trainer") -> None:
        if not self.enabled:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pi_history.clear()
        self.usage_history.clear()
        self.tracked_epochs.clear()

    def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, MetricsDict],
        history: HistoryDict,
        trainer: "Trainer",
    ) -> None:
        if not self.enabled or (epoch % self.log_every) != 0:
            return

        splits = trainer.latest_splits
        if splits is None or splits.x_train.size == 0:
            return

        state = getattr(trainer, "_current_state", None)
        if state is None:
            return

        x_train = splits.x_train
        if x_train.shape[0] > self.max_samples:
            indices = np.random.choice(x_train.shape[0], size=self.max_samples, replace=False)
            x_train = x_train[indices]

        batch_size = int(max(1, min(self.eval_batch_size, x_train.shape[0])))
        usage_sum = None
        total_samples = 0
        pi_value = None

        try:
            for start in range(0, x_train.shape[0], batch_size):
                end = min(start + batch_size, x_train.shape[0])
                batch = jnp.asarray(x_train[start:end])
                forward_output = state.apply_fn({"params": state.params}, batch, training=False)
                extras = getattr(forward_output, "extras", None)
                if extras is None:
                    continue

                responsibilities = extras.get("responsibilities") if hasattr(extras, "get") else None
                pi = extras.get("pi") if hasattr(extras, "get") else None

                if responsibilities is not None:
                    resp_np = np.asarray(responsibilities)
                    batch_sum = resp_np.sum(axis=0)
                    usage_sum = batch_sum if usage_sum is None else usage_sum + batch_sum
                    total_samples += resp_np.shape[0]

                if pi is not None:
                    pi_value = np.asarray(pi)

            if usage_sum is not None and total_samples > 0:
                usage = usage_sum / float(total_samples)
                self.usage_history.append(usage)

            if pi_value is not None:
                self.pi_history.append(pi_value)

            if usage_sum is not None or pi_value is not None:
                self.tracked_epochs.append(epoch)
        except Exception as exc:  # pragma: no cover - observational
            print(f"Warning: Failed to track mixture history at epoch {epoch}: {exc}")

    def on_train_end(self, history: HistoryDict, trainer: "Trainer") -> None:
        if not self.enabled or not self.tracked_epochs:
            return

        try:
            if self.pi_history:
                pi_array = np.stack(self.pi_history, axis=0)
                np.save(self.output_dir / "pi_history.npy", pi_array)

            if self.usage_history:
                usage_array = np.stack(self.usage_history, axis=0)
                np.save(self.output_dir / "usage_history.npy", usage_array)

            np.save(self.output_dir / "tracked_epochs.npy", np.array(self.tracked_epochs, dtype=np.int32))

            print(
                f"Mixture history saved to {self.output_dir} ({len(self.tracked_epochs)} epochs tracked)"
            )

        except Exception as exc:  # pragma: no cover - observational
            print(f"Warning: Failed to save mixture history: {exc}")
