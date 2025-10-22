from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ssvae.config import SSVAEConfig
from callbacks import TrainingCallback
from training.trainer import Trainer

HistoryDict = Dict[str, List[float]]


class InteractiveTrainer:
    """Stateful trainer for incremental/interactive SSVAE sessions."""

    def __init__(
        self,
        model,
        *,
        export_history: bool = False,
        callbacks: Sequence[TrainingCallback] | None = None,
    ):
        self.model = model
        self.config: SSVAEConfig = model.config
        self._trainer = Trainer(self.config)
        self._export_history = export_history
        self._callbacks = list(callbacks) if callbacks is not None else None
        self._state = model.state
        self._shuffle_rng = model._shuffle_rng
        self._weights_path: Optional[str] = model.weights_path

    def train_epochs(
        self,
        num_epochs: int,
        data: np.ndarray,
        labels: np.ndarray,
        *,
        weights_path: Optional[str] = None,
        patience: Optional[int] = None,
    ) -> HistoryDict:
        """Train for a fixed number of epochs while preserving optimizer state."""
        if num_epochs <= 0:
            return self._trainer._init_history()

        if weights_path is not None:
            self._weights_path = weights_path
        elif self.model.weights_path is not None:
            self._weights_path = self.model.weights_path

        if self._callbacks is not None:
            callbacks = self._callbacks
        else:
            callbacks = self.model._build_callbacks(
                weights_path=self._weights_path,
                export_history=self._export_history,
            )

        self._state, self._shuffle_rng, history = self._trainer.train(
            self._state,
            data=data,
            labels=labels,
            weights_path=self._weights_path,
            shuffle_rng=self._shuffle_rng,
            train_step_fn=self.model._train_step,
            eval_metrics_fn=self.model._eval_metrics,
            save_fn=self.model._save_weights,
            callbacks=callbacks,
            num_epochs=num_epochs,
            patience=patience,
        )

        self.model.state = self._state
        self.model._shuffle_rng = self._shuffle_rng
        self.model._rng = self._state.rng
        self.model.weights_path = self._weights_path
        return history

    def get_latent_space(self, data: np.ndarray) -> np.ndarray:
        """Return deterministic latent coordinates for visualization."""
        x = jnp.array(data, dtype=jnp.float32)
        z_mean, _, _, _, _ = self.model._apply_fn(self._state.params, x, training=False)
        return np.array(z_mean)

    def predict(
        self,
        data: np.ndarray,
        *,
        sample: bool = False,
        num_samples: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Mirror SSVAE.predict() while reusing incremental trainer state."""
        return self.model.predict(data, sample=sample, num_samples=num_samples)

    def save_checkpoint(self, path: str) -> None:
        """Persist current model parameters and optimizer state."""
        self.model._save_weights(self._state, path)
        self._weights_path = path

    def load_checkpoint(self, path: str) -> None:
        """Load model parameters and optimizer state, resuming training."""
        self.model.load_model_weights(path)
        self._state = self.model.state
        self._shuffle_rng = self.model._shuffle_rng
        self._weights_path = path
