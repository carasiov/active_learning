from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from callbacks import TrainingCallback
from ssvae.config import SSVAEConfig
from training.train_state import SSVAETrainState
from ssvae.components.tau_classifier import accumulate_soft_counts, update_tau_in_params

MetricsDict = Dict[str, jnp.ndarray]
HistoryDict = Dict[str, list[float]]
TrainStepFn = Callable[[SSVAETrainState, jnp.ndarray, jnp.ndarray, jax.Array, float], Tuple[SSVAETrainState, MetricsDict]]
EvalMetricsFn = Callable[[Dict[str, Dict[str, jnp.ndarray]], jnp.ndarray, jnp.ndarray], MetricsDict]
SaveFn = Callable[[SSVAETrainState, str], None]


@dataclass(frozen=True)
class DataSplits:
    x_train: jnp.ndarray
    y_train: jnp.ndarray
    x_val: jnp.ndarray
    y_val: jnp.ndarray
    train_size: int
    val_size: int
    total_samples: int
    labeled_count: int

    @property
    def has_labels(self) -> bool:
        return self.labeled_count > 0

    def with_train(self, x_train: jnp.ndarray, y_train: jnp.ndarray) -> DataSplits:
        return DataSplits(
            x_train=x_train,
            y_train=y_train,
            x_val=self.x_val,
            y_val=self.y_val,
            train_size=self.train_size,
            val_size=self.val_size,
            total_samples=self.total_samples,
            labeled_count=self.labeled_count,
        )


@dataclass(frozen=True)
class TrainingSetup:
    batch_size: int
    eval_batch_size: int
    max_epochs: int
    patience: int
    monitor_metric: str


@dataclass
class EarlyStoppingTracker:
    monitor_metric: str
    patience: int
    best_val: float = np.inf
    wait: int = 0
    checkpoint_saved: bool = False
    halted_early: bool = False

    def update(
        self,
        current_val: float,
        *,
        state: SSVAETrainState,
        weights_path: str | None,
        save_fn: SaveFn,
    ) -> bool:
        if current_val < self.best_val:
            self.best_val = current_val
            self.wait = 0
            if weights_path is not None:
                Path(weights_path).parent.mkdir(parents=True, exist_ok=True)
                save_fn(state, weights_path)
                self.checkpoint_saved = True
            return False

        self.wait += 1
        if self.wait >= self.patience:
            print(
                f"Early stopping triggered: {self.monitor_metric} stalled for {self.patience} epochs "
                f"(best {self.best_val:.4f})."
            )
            self.halted_early = True
            return True
        return False


class Trainer:
    """JAX training loop for the SSVAE model with early stopping and checkpointing."""

    def __init__(self, config: SSVAEConfig):
        self.config = config
        self._latest_splits: DataSplits | None = None
        self._current_state: SSVAETrainState | None = None  # For callback access during training
        self._soft_counts: np.ndarray | None = None  # For τ-classifier soft count accumulation

    @staticmethod
    def _init_history() -> HistoryDict:
        return {
            "loss": [],
            "loss_no_global_priors": [],
            "reconstruction_loss": [],
            "kl_loss": [],
            "kl_z": [],
            "kl_c": [],
            "dirichlet_penalty": [],
            "component_diversity": [],
            "classification_loss": [],
            "contrastive_loss": [],
            "component_entropy": [],
            "pi_entropy": [],
            "val_loss": [],
            "val_loss_no_global_priors": [],
            "val_reconstruction_loss": [],
            "val_kl_loss": [],
            "val_kl_z": [],
            "val_kl_c": [],
            "val_dirichlet_penalty": [],
            "val_component_diversity": [],
            "val_classification_loss": [],
            "val_contrastive_loss": [],
            "val_component_entropy": [],
            "val_pi_entropy": [],
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
    ) -> Tuple[SSVAETrainState, jax.Array, HistoryDict]:
        state_rng = state.rng
        splits, state_rng = self._prepare_data(state_rng=state_rng, data=data, labels=labels)
        self._latest_splits = splits
        setup = self._configure_training(splits=splits, num_epochs=num_epochs, patience=patience)

        history = self._init_history()
        tracker = EarlyStoppingTracker(monitor_metric=setup.monitor_metric, patience=setup.patience)
        callback_list = list(callbacks or [])

        # Initialize τ-classifier soft counts
        if self.config.use_tau_classifier:
            self._soft_counts = np.zeros(
                (self.config.num_components, self.config.num_classes),
                dtype=np.float32,
            )

        self._run_callbacks(callback_list, "on_train_start", self)

        if setup.max_epochs <= 0:
            state = state.replace(rng=state_rng)
            self._run_callbacks(callback_list, "on_train_end", history, self)
            return state, shuffle_rng, history

        epochs_ran = 0
        for epoch in range(setup.max_epochs):
            epochs_ran = epoch + 1
            if self.config.kl_c_anneal_epochs > 0:
                kl_c_scale = min(1.0, (epoch + 1) / float(self.config.kl_c_anneal_epochs))
            else:
                kl_c_scale = 1.0
            state, state_rng, shuffle_rng, splits = self._train_one_epoch(
                state,
                splits=splits,
                state_rng=state_rng,
                shuffle_rng=shuffle_rng,
                train_step_fn=train_step_fn,
                batch_size=setup.batch_size,
                kl_c_scale=kl_c_scale,
            )
            self._latest_splits = splits

            # τ-classifier: accumulate soft counts and update τ matrix
            if self.config.use_tau_classifier:
                self._accumulate_tau_counts(state, splits, setup.eval_batch_size)
                state = self._update_tau_parameters(state)

            train_metrics, val_metrics = self._evaluate_both_splits(
                state.params, splits, eval_metrics_fn=eval_metrics_fn, eval_batch_size=setup.eval_batch_size
            )
            self._update_history(history, train_metrics, val_metrics)

            # Store state for callback access
            self._current_state = state

            metrics_bundle = {"train": train_metrics, "val": val_metrics}
            self._run_callbacks(callback_list, "on_epoch_end", epoch, metrics_bundle, history, self)

            current_val = float(val_metrics[setup.monitor_metric])
            should_stop = tracker.update(
                current_val,
                state=state,
                weights_path=weights_path,
                save_fn=save_fn,
            )
            if should_stop:
                break

        if tracker.checkpoint_saved:
            status = "Early stopping" if tracker.halted_early else "Training complete"
            print(
                f"{status} after {epochs_ran} epochs. Best {tracker.monitor_metric} = {tracker.best_val:.4f} "
                f"(checkpoint saved to {weights_path})."
            )

        state = state.replace(rng=state_rng)
        self._run_callbacks(callback_list, "on_train_end", history, self)

        # Clear state reference after training
        self._current_state = None

        # Clean up τ-classifier soft counts
        if self.config.use_tau_classifier:
            total_counts = np.sum(self._soft_counts) if self._soft_counts is not None else 0
            print(f"τ-classifier: Accumulated {total_counts:.0f} total soft counts over training.")
            self._soft_counts = None

        return state, shuffle_rng, history

    @property
    def latest_splits(self) -> DataSplits | None:
        return self._latest_splits

    def _prepare_data(self, *, state_rng: jax.Array, data: np.ndarray, labels: np.ndarray) -> Tuple[DataSplits, jax.Array]:
        x_np = np.asarray(data, dtype=np.float32)
        y_np = np.asarray(labels, dtype=np.float32).reshape((-1,))

        total_samples = x_np.shape[0]
        if total_samples <= 1:
            val_size = 0
        else:
            val_size = max(1, min(int(self.config.val_split * total_samples), total_samples - 1))
        train_size = total_samples - val_size

        labeled_count = self._count_labeled_samples(y_np)

        if total_samples > 0:
            state_rng, dataset_key = jax.random.split(state_rng)
            perm = np.asarray(jax.random.permutation(dataset_key, total_samples))
            x_np = x_np[perm]
            y_np = y_np[perm]

        x_train_np, y_train_np = x_np[:train_size], y_np[:train_size]
        x_val_np, y_val_np = x_np[train_size:], y_np[train_size:]

        splits = DataSplits(
            x_train=jnp.array(x_train_np),
            y_train=jnp.array(y_train_np),
            x_val=jnp.array(x_val_np),
            y_val=jnp.array(y_val_np),
            train_size=train_size,
            val_size=val_size,
            total_samples=total_samples,
            labeled_count=labeled_count,
        )
        return splits, state_rng

    def _configure_training(
        self,
        *,
        splits: DataSplits,
        num_epochs: int | None,
        patience: int | None,
    ) -> TrainingSetup:
        if self.config.monitor_metric == "auto":
            monitor_metric = "classification_loss" if splits.has_labels else "loss"
        elif self.config.monitor_metric in {"loss", "classification_loss"}:
            monitor_metric = self.config.monitor_metric
        else:
            raise ValueError(f"Unknown monitor_metric: {self.config.monitor_metric}")

        print(f"Monitoring validation {monitor_metric} for early stopping.", flush=True)
        if splits.has_labels:
            print(f"Detected {splits.labeled_count} labeled samples.", flush=True)

        batch_size = self.config.batch_size
        max_epochs = num_epochs if num_epochs is not None else self.config.max_epochs
        used_patience = patience if patience is not None else self.config.patience

        self._log_session_hyperparameters(max_epochs=max_epochs, patience=used_patience)

        eval_batch_size = min(batch_size, 1024)
        return TrainingSetup(
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            max_epochs=max_epochs,
            patience=used_patience,
            monitor_metric=monitor_metric,
        )

    def _train_one_epoch(
        self,
        state: SSVAETrainState,
        *,
        splits: DataSplits,
        state_rng: jax.Array,
        shuffle_rng: jax.Array,
        train_step_fn: TrainStepFn,
        batch_size: int,
        kl_c_scale: float,
    ) -> Tuple[SSVAETrainState, jax.Array, jax.Array, DataSplits]:
        if splits.train_size == 0:
            return state, state_rng, shuffle_rng, splits

        shuffle_rng, epoch_key = jax.random.split(shuffle_rng)
        perm = jax.random.permutation(epoch_key, splits.train_size)
        x_train = jnp.take(splits.x_train, perm, axis=0)
        y_train = jnp.take(splits.y_train, perm, axis=0)

        for batch_x, batch_y in self._batch_iterator(x_train, y_train, batch_size):
            state, state_rng = self._train_one_batch(
                state,
                batch_x,
                batch_y,
                state_rng,
                train_step_fn,
                kl_c_scale,
            )

        updated_splits = splits.with_train(x_train=x_train, y_train=y_train)
        return state, state_rng, shuffle_rng, updated_splits

    def _batch_iterator(
        self,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        batch_size: int,
    ) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        train_size = x_train.shape[0]
        if train_size == 0:
            return

        full_span = (train_size // batch_size) * batch_size
        if full_span == 0:
            yield jnp.asarray(x_train), jnp.asarray(y_train)
            return

        for start in range(0, full_span, batch_size):
            end = start + batch_size
            yield jnp.asarray(x_train[start:end]), jnp.asarray(y_train[start:end])

        if full_span < train_size:
            yield jnp.asarray(x_train[full_span:train_size]), jnp.asarray(y_train[full_span:train_size])

    def _train_one_batch(
        self,
        state: SSVAETrainState,
        batch_x: jnp.ndarray,
        batch_y: jnp.ndarray,
        state_rng: jax.Array,
        train_step_fn: TrainStepFn,
        kl_c_scale: float,
    ) -> Tuple[SSVAETrainState, jax.Array]:
        state_rng, raw_key = jax.random.split(state_rng)
        batch_key = jax.random.fold_in(raw_key, int(state.step))
        state, _ = train_step_fn(state, batch_x, batch_y, batch_key, kl_c_scale)
        return state, state_rng

    def _accumulate_tau_counts(
        self,
        state: SSVAETrainState,
        splits: DataSplits,
        eval_batch_size: int,
    ) -> None:
        """Accumulate soft counts for τ-classifier from labeled training samples.

        This method runs a forward pass on all labeled training samples to extract
        responsibilities and accumulate soft counts for τ matrix updates.

        Args:
            state: Current training state
            splits: Data splits containing training/validation data
            eval_batch_size: Batch size for evaluation passes
        """
        if not self.config.use_tau_classifier or not splits.has_labels:
            return

        # Initialize soft counts on first call
        if self._soft_counts is None:
            self._soft_counts = np.zeros(
                (self.config.num_components, self.config.num_classes),
                dtype=np.float32,
            )

        # Process training data in chunks
        x_data = splits.x_train
        y_data = splits.y_train

        # Filter to labeled samples only
        labeled_mask = ~np.isnan(np.array(y_data))
        if not np.any(labeled_mask):
            return

        x_labeled = x_data[labeled_mask]
        y_labeled = y_data[labeled_mask]

        num_samples = x_labeled.shape[0]

        # Forward pass to get responsibilities
        for start_idx in range(0, num_samples, eval_batch_size):
            end_idx = min(start_idx + eval_batch_size, num_samples)
            batch_x = x_labeled[start_idx:end_idx]
            batch_y = y_labeled[start_idx:end_idx]

            # Run forward pass (no gradient computation)
            output = state.apply_fn(
                {"params": state.params},
                batch_x,
                training=False,
            )

            # Extract responsibilities if available
            if hasattr(output, 'extras') and output.extras and 'responsibilities' in output.extras:
                responsibilities = output.extras['responsibilities']

                # Accumulate soft counts
                batch_counts = accumulate_soft_counts(
                    responsibilities,
                    batch_y,
                    self.config.num_classes,
                    mask=None,  # Already filtered to labeled samples
                )
                self._soft_counts += np.array(batch_counts)

    def _update_tau_parameters(self, state: SSVAETrainState) -> SSVAETrainState:
        """Update τ matrix in model parameters from accumulated soft counts.

        Args:
            state: Current training state

        Returns:
            Updated training state with new τ matrix and reinitialized optimizer state

        Note:
            Since τ parameters are updated outside gradient descent, we must reinitialize
            the optimizer state to match the new parameter tree structure. This happens
            once per epoch, so the impact on training is minimal.
        """
        if not self.config.use_tau_classifier or self._soft_counts is None:
            return state

        # Update τ from accumulated counts
        new_params = update_tau_in_params(
            state.params,
            jnp.array(self._soft_counts),
            alpha_0=self.config.tau_alpha_0,
        )

        # Reinitialize optimizer state to match new parameter tree
        # This is necessary because τ was updated outside gradient descent
        new_opt_state = state.tx.init(new_params)

        return state.replace(params=new_params, opt_state=new_opt_state)

    def _evaluate_both_splits(
        self,
        params: Dict[str, Dict[str, jnp.ndarray]],
        splits: DataSplits,
        *,
        eval_metrics_fn: EvalMetricsFn,
        eval_batch_size: int,
    ) -> Tuple[MetricsDict, MetricsDict]:
        if splits.train_size > 0:
            train_metrics = self._evaluate_in_chunks(
                params,
                splits.x_train,
                splits.y_train,
                eval_metrics_fn=eval_metrics_fn,
                eval_batch_size=eval_batch_size,
            )
        else:
            train_metrics = self._evaluate_in_chunks(
                params,
                splits.x_val,
                splits.y_val,
                eval_metrics_fn=eval_metrics_fn,
                eval_batch_size=eval_batch_size,
            )

        if splits.val_size > 0:
            val_metrics = self._evaluate_in_chunks(
                params,
                splits.x_val,
                splits.y_val,
                eval_metrics_fn=eval_metrics_fn,
                eval_batch_size=eval_batch_size,
            )
        else:
            val_metrics = train_metrics

        return train_metrics, val_metrics

    def _evaluate_in_chunks(
        self,
        params: Dict[str, Dict[str, jnp.ndarray]],
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        *,
        eval_metrics_fn: EvalMetricsFn,
        eval_batch_size: int,
    ) -> MetricsDict:
        total = inputs.shape[0]
        if total == 0:
            return eval_metrics_fn(params, inputs, targets)

        metrics_sum: Dict[str, jnp.ndarray] | None = None
        processed = 0
        for start in range(0, total, eval_batch_size):
            end = min(start + eval_batch_size, total)
            batch_inputs = jnp.asarray(inputs[start:end])
            batch_targets = jnp.asarray(targets[start:end])
            batch_metrics = eval_metrics_fn(params, batch_inputs, batch_targets)
            weight = end - start
            if metrics_sum is None:
                metrics_sum = {k: batch_metrics[k] * weight for k in batch_metrics}
            else:
                for key in metrics_sum:
                    metrics_sum[key] = metrics_sum[key] + batch_metrics[key] * weight
            processed += weight

        assert metrics_sum is not None  # for mypy; total > 0 ensures this
        return {k: metrics_sum[k] / processed for k in metrics_sum}

    def _update_history(
        self,
        history: HistoryDict,
        train_metrics: MetricsDict,
        val_metrics: MetricsDict,
    ) -> None:
        tracked_keys = (
            "loss",
            "loss_no_global_priors",
            "reconstruction_loss",
            "kl_loss",
            "kl_z",
            "kl_c",
            "dirichlet_penalty",
            "component_diversity",
            "classification_loss",
            "contrastive_loss",
            "component_entropy",
            "pi_entropy",
        )
        for key in tracked_keys:
            history[key].append(float(train_metrics[key]))
            history[f"val_{key}"].append(float(val_metrics[key]))

    @staticmethod
    def _run_callbacks(
        callbacks: Sequence[TrainingCallback],
        method: str,
        *args,
    ) -> None:
        for callback in callbacks:
            getattr(callback, method)(*args)
