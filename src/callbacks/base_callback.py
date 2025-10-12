"""Base callback interface."""

from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from training.trainer import Trainer

MetricsDict = Dict[str, jnp.ndarray]
HistoryDict = Dict[str, List[float]]


class TrainingCallback:
    """Base callback interface for training observability."""

    def on_train_start(self, trainer: "Trainer") -> None:
        """Called once before training begins."""
        return None

    def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, MetricsDict],
        history: HistoryDict,
        trainer: "Trainer",
    ) -> None:
        """Called after each epoch completes."""
        return None

    def on_train_end(self, history: HistoryDict, trainer: "Trainer") -> None:
        """Called once after training completes."""
        return None
