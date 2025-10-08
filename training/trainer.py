from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from configs.base import SSVAEConfig
from training.train_state import SSVAETrainState

MetricsDict = Dict[str, jnp.ndarray]
HistoryDict = Dict[str, list[float]]
TrainStepFn = Callable[[SSVAETrainState, jnp.ndarray, jnp.ndarray, jax.Array], Tuple[SSVAETrainState, MetricsDict]]
EvalMetricsFn = Callable[[Dict[str, Dict[str, jnp.ndarray]], jnp.ndarray, jnp.ndarray], MetricsDict]
LogFn = Callable[[int, MetricsDict, MetricsDict, HistoryDict], None]
SaveFn = Callable[[str], None]
ExportHistoryFn = Callable[[HistoryDict], None]


class Trainer:
    """JAX training loop for the SSVAE model with early stopping and checkpointing."""

    def __init__(self, config: SSVAEConfig):
        self.config = config

    @staticmethod
    def _init_history() -> HistoryDict:
        return {
            "loss": [],
            "reconstruction_loss": [],
            "kl_loss": [],
            "classification_loss": [],
            "contrastive_loss": [],
            "val_loss": [],
            "val_reconstruction_loss": [],
            "val_kl_loss": [],
            "val_classification_loss": [],
            "val_contrastive_loss": [],
        }

    def train(
        self,
        state: SSVAETrainState,
        *,
        data: np.ndarray,
        labels: np.ndarray,
        weights_path: str | None,
        shuffle_rng: jax.Array,
        train_step_fn: TrainStepFn,
        eval_metrics_fn: EvalMetricsFn,
        log_fn: LogFn,
        save_fn: SaveFn,
        export_history_fn: ExportHistoryFn,
        num_epochs: int | None = None,
        patience: int | None = None,
    ) -> Tuple[SSVAETrainState, jax.Array, HistoryDict]:
        x_np = np.asarray(data, dtype=np.float32)
        y_np = np.asarray(labels, dtype=np.float32).reshape((-1,))

        total_samples = x_np.shape[0]
        if total_samples <= 1:
            val_size = 0
        else:
            val_size = max(1, min(int(self.config.val_split * total_samples), total_samples - 1))
        train_size = total_samples - val_size

        state_rng = state.rng
        if total_samples > 0:
            state_rng, dataset_key = jax.random.split(state_rng)
            perm = np.asarray(jax.random.permutation(dataset_key, total_samples))
            x_np = x_np[perm]
            y_np = y_np[perm]

        x_train_np, y_train_np = x_np[:train_size], y_np[:train_size]
        x_val_np, y_val_np = x_np[train_size:], y_np[train_size:]

        x_train = jnp.array(x_train_np)
        y_train = jnp.array(y_train_np)
        x_val = jnp.array(x_val_np)
        y_val = jnp.array(y_val_np)

        history = self._init_history()
        best_val = np.inf
        wait = 0
        batch_size = self.config.batch_size
        max_epochs = num_epochs if num_epochs is not None else self.config.max_epochs
        used_patience = patience if patience is not None else self.config.patience

        if max_epochs <= 0:
            state = state.replace(rng=state_rng)
            export_history_fn(history)
            return state, shuffle_rng, history

        for epoch in range(max_epochs):
            if train_size > 0:
                shuffle_rng, epoch_key = jax.random.split(shuffle_rng)
                perm = jax.random.permutation(epoch_key, train_size)
                x_train = jnp.take(x_train, perm, axis=0)
                y_train = jnp.take(y_train, perm, axis=0)

                full_span = (train_size // batch_size) * batch_size
                if full_span == 0:
                    state_rng, raw_key = jax.random.split(state_rng)
                    batch_key = jax.random.fold_in(raw_key, int(state.step))
                    state, _ = train_step_fn(state, x_train, y_train, batch_key)
                else:
                    for start in range(0, full_span, batch_size):
                        end = start + batch_size
                        bx = jnp.asarray(x_train[start:end])
                        by = jnp.asarray(y_train[start:end])
                        state_rng, raw_key = jax.random.split(state_rng)
                        batch_key = jax.random.fold_in(raw_key, int(state.step))
                        state, _ = train_step_fn(state, bx, by, batch_key)
                    if full_span < train_size:
                        bx = jnp.asarray(x_train[full_span:train_size])
                        by = jnp.asarray(y_train[full_span:train_size])
                        state_rng, raw_key = jax.random.split(state_rng)
                        batch_key = jax.random.fold_in(raw_key, int(state.step))
                        state, _ = train_step_fn(state, bx, by, batch_key)

            if train_size > 0:
                train_metrics = eval_metrics_fn(state.params, x_train, y_train)
            else:
                train_metrics = eval_metrics_fn(state.params, x_val, y_val)

            val_metrics = eval_metrics_fn(state.params, x_val, y_val) if val_size > 0 else train_metrics

            for key in ("loss", "reconstruction_loss", "kl_loss", "classification_loss", "contrastive_loss"):
                history[key].append(float(train_metrics[key]))
                history["val_" + key].append(float(val_metrics[key]))

            log_fn(epoch, train_metrics, val_metrics, history)

            current_val = float(val_metrics["loss"])
            if current_val < best_val:
                best_val = current_val
                wait = 0
                if weights_path is not None:
                    Path(weights_path).parent.mkdir(parents=True, exist_ok=True)
                    save_fn(weights_path)
            else:
                wait += 1
                if wait >= used_patience:
                    break

        state = state.replace(rng=state_rng)
        export_history_fn(history)
        return state, shuffle_rng, history

