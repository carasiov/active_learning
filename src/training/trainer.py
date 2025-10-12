from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ssvae.config import SSVAEConfig
from callbacks import TrainingCallback
from training.train_state import SSVAETrainState

MetricsDict = Dict[str, jnp.ndarray]
HistoryDict = Dict[str, list[float]]
TrainStepFn = Callable[[SSVAETrainState, jnp.ndarray, jnp.ndarray, jax.Array], Tuple[SSVAETrainState, MetricsDict]]
EvalMetricsFn = Callable[[Dict[str, Dict[str, jnp.ndarray]], jnp.ndarray, jnp.ndarray], MetricsDict]
SaveFn = Callable[[str], None]


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

    def _log_session_hyperparameters(self, *, max_epochs: int, patience: int) -> None:
        summary = self.config.get_informative_hyperparameters()
        summary["max_epochs"] = max_epochs
        summary["patience"] = patience
        lines = ["Starting training session with hyperparameters:"]
        padding = max(len(name) for name in summary)
        for key, value in summary.items():
            lines.append(f"  - {key.ljust(padding)} : {value}")
        print("\n".join(lines), flush=True)

    @staticmethod
    def _count_labeled_samples(labels: np.ndarray) -> int:
        """Count the number of non-NaN labels."""
        return int(np.sum(~np.isnan(labels.reshape(-1))))

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
        save_fn: SaveFn,
        callbacks: Sequence[TrainingCallback] | None = None,
        num_epochs: int | None = None,
        patience: int | None = None,
    ) -> Tuple[SSVAETrainState, jax.Array, HistoryDict]:  # returns (state, updated_shuffle_rng, history)
        x_np = np.asarray(data, dtype=np.float32)
        y_np = np.asarray(labels, dtype=np.float32).reshape((-1,))

        total_samples = x_np.shape[0]
        if total_samples <= 1:
            val_size = 0
        else:
            val_size = max(1, min(int(self.config.val_split * total_samples), total_samples - 1))
        train_size = total_samples - val_size

        labeled_count = self._count_labeled_samples(y_np)
        has_labels = labeled_count > 0
        if self.config.monitor_metric == "auto":
            monitor_metric = "classification_loss" if has_labels else "loss"
        elif self.config.monitor_metric in {"loss", "classification_loss"}:
            monitor_metric = self.config.monitor_metric
        else:
            raise ValueError(f"Unknown monitor_metric: {self.config.monitor_metric}")

        print(f"Monitoring validation {monitor_metric} for early stopping.", flush=True)
        if has_labels:
            print(f"Detected {labeled_count} labeled samples.", flush=True)

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
        self._log_session_hyperparameters(max_epochs=max_epochs, patience=used_patience)

        eval_batch_size = min(batch_size, 1024)
        callback_list = list(callbacks) if callbacks is not None else []

        def _run_eval(params, ex, ey):
            """Evaluate metrics in smaller chunks to avoid large temporary buffers on GPU."""
            total = ex.shape[0]
            if total == 0:
                return eval_metrics_fn(params, ex, ey)

            metrics_sum: Dict[str, jnp.ndarray] | None = None
            processed = 0
            for start in range(0, total, eval_batch_size):
                end = min(start + eval_batch_size, total)
                bx = jnp.asarray(ex[start:end])
                by = jnp.asarray(ey[start:end])
                batch_metrics = eval_metrics_fn(params, bx, by)
                weight = end - start
                if metrics_sum is None:
                    metrics_sum = {k: batch_metrics[k] * weight for k in batch_metrics}
                else:
                    for key in metrics_sum:
                        metrics_sum[key] = metrics_sum[key] + batch_metrics[key] * weight
                processed += weight

            return {k: metrics_sum[k] / processed for k in metrics_sum}

        for callback in callback_list:
            callback.on_train_start(self)

        if max_epochs <= 0:
            state = state.replace(rng=state_rng)
            for callback in callback_list:
                callback.on_train_end(history, self)
            return state, shuffle_rng, history

        halted_early = False
        checkpoint_saved = False
        epochs_ran = 0

        for epoch in range(max_epochs):
            epochs_ran = epoch + 1
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
                train_metrics = _run_eval(state.params, x_train, y_train)
            else:
                train_metrics = _run_eval(state.params, x_val, y_val)

            val_metrics = _run_eval(state.params, x_val, y_val) if val_size > 0 else train_metrics

            for key in ("loss", "reconstruction_loss", "kl_loss", "classification_loss", "contrastive_loss"):
                history[key].append(float(train_metrics[key]))
                history["val_" + key].append(float(val_metrics[key]))

            metrics_bundle = {"train": train_metrics, "val": val_metrics}
            for callback in callback_list:
                callback.on_epoch_end(epoch, metrics_bundle, history, self)

            current_val = float(val_metrics[monitor_metric])
            if current_val < best_val:
                best_val = current_val
                wait = 0
                if weights_path is not None:
                    Path(weights_path).parent.mkdir(parents=True, exist_ok=True)
                    save_fn(weights_path)
                    checkpoint_saved = True
            else:
                wait += 1
                if wait >= used_patience:
                    print(
                        f"Early stopping triggered: {monitor_metric} stalled for {used_patience} epochs "
                        f"(best {best_val:.4f})."
                    )
                    halted_early = True
                    break

        if checkpoint_saved:
            status = "Early stopping" if halted_early else "Training complete"
            print(
                f"{status} after {epochs_ran} epochs. Best {monitor_metric} = {best_val:.4f} (checkpoint saved to {weights_path})."
            )

        state = state.replace(rng=state_rng)
        for callback in callback_list:
            callback.on_train_end(history, self)
        return state, shuffle_rng, history
