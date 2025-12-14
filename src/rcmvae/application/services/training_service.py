from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from rcmvae.application.callbacks import TrainingCallback
from rcmvae.application.runtime.context import ModelRuntime
from rcmvae.application.runtime.state import SSVAETrainState
from rcmvae.application.runtime.types import EvalMetricsFn, MetricsDict, TrainStepFn
from rcmvae.domain.config import SSVAEConfig

HistoryDict = Dict[str, list[float]]
SaveFn = Callable[[SSVAETrainState, str], None]

# Curriculum snapshot callback type: (params, epoch, k_active, event_type) -> None
CurriculumSnapshotFn = Callable[[Dict, int, int, str], None]


@dataclass
class TrainerLoopHooks:
    """Optional extension points for the training loop."""

    batch_context_fn: Callable[[SSVAETrainState, jnp.ndarray, jnp.ndarray], Dict[str, jnp.ndarray] | None] | None = None
    post_batch_fn: Callable[[SSVAETrainState, jnp.ndarray, jnp.ndarray, MetricsDict], None] | None = None
    eval_context_fn: Callable[[], Dict[str, jnp.ndarray] | None] | None = None


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
        # Trigger mode state: tracks current k_active across epochs
        self._trigger_k_active: int | None = None
        # Track epochs since last unlock for migration window in trigger mode
        self._trigger_epochs_since_unlock: int = 0

    @staticmethod
    def _init_history() -> HistoryDict:
        return {
            "loss": [],
            "loss_no_global_priors": [],
            "reconstruction_loss": [],
            "kl_loss": [],
            "kl_z": [],
            "kl_c": [],
            "kl_c_logit_mog": [],
            "dirichlet_penalty": [],
            "usage_sparsity_loss": [],
            "component_diversity": [],
            "classification_loss": [],
            "contrastive_loss": [],
            "component_entropy": [],
            "pi_entropy": [],
            "k_active": [],  # Curriculum: number of active channels
            "in_migration_window": [],  # Curriculum: whether in migration window
            # Trigger-based unlock metrics (Stage 4)
            "normality_score": [],  # Latent normality proxy
            "plateau_detected": [],  # Whether plateau condition was met
            "plateau_improvement": [],  # Relative improvement over plateau window
            "unlock_triggered": [],  # Whether unlock was triggered this epoch
            "val_loss": [],
            "val_loss_no_global_priors": [],
            "val_reconstruction_loss": [],
            "val_kl_loss": [],
            "val_kl_z": [],
            "val_kl_c": [],
            "val_kl_c_logit_mog": [],
            "val_dirichlet_penalty": [],
            "val_usage_sparsity_loss": [],
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
        runtime: ModelRuntime,
        *,
        data: np.ndarray,
        labels: np.ndarray,
        weights_path: str | None,
        save_fn: SaveFn,
        callbacks: Sequence[TrainingCallback] | None = None,
        num_epochs: int | None = None,
        patience: int | None = None,
        loop_hooks: TrainerLoopHooks | None = None,
        curriculum_snapshot_fn: CurriculumSnapshotFn | None = None,
    ) -> Tuple[ModelRuntime, HistoryDict]:
        state = runtime.state
        state_rng = state.rng
        shuffle_rng = runtime.shuffle_rng
        train_step_fn = runtime.train_step_fn
        eval_metrics_fn = runtime.eval_metrics_fn
        splits, state_rng = self._prepare_data(state_rng=state_rng, data=data, labels=labels)
        self._latest_splits = splits
        setup = self._configure_training(splits=splits, num_epochs=num_epochs, patience=patience)

        history = self._init_history()
        tracker = EarlyStoppingTracker(monitor_metric=setup.monitor_metric, patience=setup.patience)
        callback_list = list(callbacks or [])

        self._run_callbacks(callback_list, "on_train_start", self)

        if setup.max_epochs <= 0:
            state = state.replace(rng=state_rng)
            updated_runtime = runtime.replace(state=state, shuffle_rng=shuffle_rng)
            self._run_callbacks(callback_list, "on_train_end", history, self)
            self._current_state = None
            return updated_runtime, history

        # Initialize trigger mode state
        use_trigger_mode = (
            self.config.curriculum_enabled
            and self.config.curriculum_unlock_mode == "trigger"
        )
        if use_trigger_mode:
            self._trigger_k_active = self.config.curriculum_start_k_active
            self._trigger_epochs_since_unlock = 0

        epochs_ran = 0
        prev_k_active = self.config.curriculum_start_k_active if self.config.curriculum_enabled else self.config.num_components
        for epoch in range(setup.max_epochs):
            epochs_ran = epoch + 1
            if self.config.kl_c_anneal_epochs > 0:
                kl_c_scale = min(1.0, (epoch + 1) / float(self.config.kl_c_anneal_epochs))
            else:
                kl_c_scale = 1.0

            # Calculate curriculum k_active based on mode
            if use_trigger_mode:
                k_active = self._trigger_k_active
            else:
                k_active = self.config.get_k_active(epoch)

            # Detect epoch-based unlock (k_active increased from previous epoch)
            epoch_unlock_triggered = (not use_trigger_mode) and (k_active > prev_k_active)

            # Calculate Gumbel temperature (with migration window boost if applicable)
            # For trigger mode, use epochs_since_unlock for migration window
            if use_trigger_mode:
                in_migration = (
                    self.config.curriculum_migration_epochs > 0
                    and self._trigger_epochs_since_unlock < self.config.curriculum_migration_epochs
                )
                if in_migration:
                    gumbel_temp = self.config.gumbel_temperature * self.config.curriculum_temp_boost_during_migration
                    effective_logit_mog_weight = self.config.c_logit_prior_weight * self.config.curriculum_logit_mog_scale_during_migration
                    use_straight_through = not self.config.curriculum_soft_routing_during_migration
                else:
                    gumbel_temp = self.config.gumbel_temperature
                    effective_logit_mog_weight = self.config.c_logit_prior_weight
                    use_straight_through = self.config.use_straight_through_gumbel
            else:
                gumbel_temp = self.config.get_effective_gumbel_temperature(epoch)
                use_straight_through = self.config.use_straight_through_for_epoch(epoch)
                effective_logit_mog_weight = self.config.get_effective_logit_mog_weight(epoch)
                in_migration = self.config.is_in_migration_window(epoch)

            state, state_rng, shuffle_rng, splits = self._train_one_epoch(
                state,
                splits=splits,
                state_rng=state_rng,
                shuffle_rng=shuffle_rng,
                train_step_fn=train_step_fn,
                batch_size=setup.batch_size,
                kl_c_scale=kl_c_scale,
                gumbel_temperature=gumbel_temp,
                k_active=k_active,
                use_straight_through=use_straight_through,
                effective_logit_mog_weight=effective_logit_mog_weight,
                loop_hooks=loop_hooks,
            )
            self._latest_splits = splits

            eval_context = loop_hooks.eval_context_fn() if loop_hooks and loop_hooks.eval_context_fn else None
            train_metrics, val_metrics = self._evaluate_both_splits(
                state.params,
                splits,
                eval_metrics_fn=eval_metrics_fn,
                eval_batch_size=setup.eval_batch_size,
                eval_context=eval_context,
            )
            self._update_history(history, train_metrics, val_metrics)

            # Record curriculum k_active and migration window status
            history["k_active"].append(k_active)
            history["in_migration_window"].append(in_migration)

            # Trigger-based unlock logic (Stage 4)
            unlock_triggered = False
            normality_score = 0.0
            plateau_detected = False
            plateau_improvement = 0.0

            if use_trigger_mode and self.config.curriculum_enabled:
                # Compute normality score from validation latents
                normality_score = self._compute_normality_score_for_trigger(
                    state.params, splits, runtime, k_active
                )

                # Check trigger conditions
                from rcmvae.application.services.curriculum_trigger import (
                    check_plateau,
                    should_unlock_trigger,
                )

                plateau_detected, plateau_improvement = check_plateau(
                    history,
                    self.config.curriculum_plateau_metric,
                    self.config.curriculum_plateau_window_epochs,
                    self.config.curriculum_plateau_min_improvement,
                )

                should_unlock, _ = should_unlock_trigger(
                    history, normality_score, self.config, k_active
                )

                # Enforce minimum epochs per channel before allowing unlock
                min_epochs = self.config.curriculum_min_epochs_per_channel
                if min_epochs > 0 and self._trigger_epochs_since_unlock < min_epochs:
                    should_unlock = False

                if should_unlock:
                    unlock_triggered = True
                    self._trigger_k_active = min(
                        self._trigger_k_active + 1,
                        self.config.curriculum_max_k_active,
                    )
                    self._trigger_epochs_since_unlock = 0
                else:
                    self._trigger_epochs_since_unlock += 1

            # Record trigger metrics
            history["normality_score"].append(normality_score)
            history["plateau_detected"].append(plateau_detected)
            history["plateau_improvement"].append(plateau_improvement)
            history["unlock_triggered"].append(unlock_triggered)

            # Save curriculum snapshots at key events
            if curriculum_snapshot_fn is not None and self.config.curriculum_enabled:
                # Save snapshot at unlock events (both trigger and epoch mode)
                any_unlock = unlock_triggered or epoch_unlock_triggered
                if any_unlock:
                    print(f"[Curriculum] Saving unlock snapshot at epoch {epoch}, k_active={k_active}")
                    try:
                        result = curriculum_snapshot_fn(
                            state.params, epoch, k_active, "unlock"
                        )
                        print(f"[Curriculum] Snapshot result: {result}")
                    except Exception as e:
                        import traceback
                        print(f"Warning: Failed to save unlock snapshot: {e}")
                        traceback.print_exc()

                # Save snapshot at end of migration window (trigger mode)
                if use_trigger_mode:
                    migration_epochs = self.config.curriculum_migration_epochs
                    if (
                        migration_epochs > 0
                        and self._trigger_epochs_since_unlock == migration_epochs
                    ):
                        print(f"[Curriculum] Saving migration_end snapshot at epoch {epoch}")
                        try:
                            result = curriculum_snapshot_fn(
                                state.params, epoch, k_active, "migration_end"
                            )
                            print(f"[Curriculum] Snapshot result: {result}")
                        except Exception as e:
                            import traceback
                            print(f"Warning: Failed to save migration_end snapshot: {e}")
                            traceback.print_exc()
                # Save snapshot at end of migration window (epoch mode)
                elif self.config.curriculum_migration_epochs > 0:
                    epochs_since = self.config.get_epochs_since_last_unlock(epoch)
                    if epochs_since == self.config.curriculum_migration_epochs:
                        print(f"[Curriculum] Saving migration_end snapshot at epoch {epoch}")
                        try:
                            result = curriculum_snapshot_fn(
                                state.params, epoch, k_active, "migration_end"
                            )
                            print(f"[Curriculum] Snapshot result: {result}")
                        except Exception as e:
                            import traceback
                            print(f"Warning: Failed to save migration_end snapshot: {e}")
                            traceback.print_exc()
            elif self.config.curriculum_enabled and curriculum_snapshot_fn is None:
                # Only log once at first epoch
                if epoch == 0:
                    print("[Curriculum] Warning: curriculum_snapshot_fn is None, snapshots will not be saved")

            # Update previous k_active for next epoch's unlock detection
            prev_k_active = k_active

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
        updated_runtime = runtime.replace(state=state, shuffle_rng=shuffle_rng)
        self._run_callbacks(callback_list, "on_train_end", history, self)

        # Clear state reference after training
        self._current_state = None

        return updated_runtime, history

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

        # Phase 6 (Terminal Cleanup): Silenced verbose output
        # Monitoring metric and labeled samples shown in experiment header
        # print(f"Monitoring validation {monitor_metric} for early stopping.", flush=True)
        # if splits.has_labels:
        #     print(f"Detected {splits.labeled_count} labeled samples.", flush=True)

        batch_size = self.config.batch_size
        max_epochs = num_epochs if num_epochs is not None else self.config.max_epochs
        used_patience = patience if patience is not None else self.config.patience

        # Phase 6 (Terminal Cleanup): Silenced hyperparameter dump
        # Full config available in experiment.log and config.yaml
        # self._log_session_hyperparameters(max_epochs=max_epochs, patience=used_patience)

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
        gumbel_temperature: float,
        k_active: int | None = None,
        use_straight_through: bool | None = None,
        effective_logit_mog_weight: float | None = None,
        loop_hooks: TrainerLoopHooks | None = None,
    ) -> Tuple[SSVAETrainState, jax.Array, jax.Array, DataSplits]:
        if splits.train_size == 0:
            return state, state_rng, shuffle_rng, splits

        shuffle_rng, epoch_key = jax.random.split(shuffle_rng)
        perm = jax.random.permutation(epoch_key, splits.train_size)
        x_train = jnp.take(splits.x_train, perm, axis=0)
        y_train = jnp.take(splits.y_train, perm, axis=0)

        for batch_x, batch_y in self._batch_iterator(x_train, y_train, batch_size):
            state, state_rng, _ = self._train_one_batch(
                state,
                batch_x,
                batch_y,
                state_rng,
                train_step_fn,
                kl_c_scale,
                gumbel_temperature,
                k_active=k_active,
                use_straight_through=use_straight_through,
                effective_logit_mog_weight=effective_logit_mog_weight,
                loop_hooks=loop_hooks,
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
        gumbel_temperature: float,
        k_active: int | None = None,
        use_straight_through: bool | None = None,
        effective_logit_mog_weight: float | None = None,
        loop_hooks: TrainerLoopHooks | None = None,
    ) -> Tuple[SSVAETrainState, jax.Array, MetricsDict]:
        batch_kwargs: Dict[str, jnp.ndarray] = {}
        if loop_hooks and loop_hooks.batch_context_fn is not None:
            context = loop_hooks.batch_context_fn(state, batch_x, batch_y) or {}
            batch_kwargs.update(context)

        state_rng, raw_key = jax.random.split(state_rng)
        batch_key = jax.random.fold_in(raw_key, int(state.step))
        state, batch_metrics = train_step_fn(state, batch_x, batch_y, batch_key, kl_c_scale, gumbel_temperature=gumbel_temperature, k_active=k_active, use_straight_through=use_straight_through, effective_logit_mog_weight=effective_logit_mog_weight, **batch_kwargs)

        if loop_hooks and loop_hooks.post_batch_fn is not None:
            loop_hooks.post_batch_fn(state, batch_x, batch_y, batch_metrics)

        return state, state_rng, batch_metrics

    def _evaluate_both_splits(
        self,
        params: Dict[str, Dict[str, jnp.ndarray]],
        splits: DataSplits,
        *,
        eval_metrics_fn: EvalMetricsFn,
        eval_batch_size: int,
        eval_context: Dict[str, jnp.ndarray] | None = None,
    ) -> Tuple[MetricsDict, MetricsDict]:
        if splits.train_size > 0:
            train_metrics = self._evaluate_in_chunks(
                params,
                splits.x_train,
                splits.y_train,
                eval_metrics_fn=eval_metrics_fn,
                eval_batch_size=eval_batch_size,
                eval_context=eval_context,
            )
        else:
            train_metrics = self._evaluate_in_chunks(
                params,
                splits.x_val,
                splits.y_val,
                eval_metrics_fn=eval_metrics_fn,
                eval_batch_size=eval_batch_size,
                eval_context=eval_context,
            )

        if splits.val_size > 0:
            val_metrics = self._evaluate_in_chunks(
                params,
                splits.x_val,
                splits.y_val,
                eval_metrics_fn=eval_metrics_fn,
                eval_batch_size=eval_batch_size,
                eval_context=eval_context,
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
        eval_context: Dict[str, jnp.ndarray] | None = None,
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
            extra_kwargs = eval_context or {}
            batch_metrics = eval_metrics_fn(params, batch_inputs, batch_targets, **extra_kwargs)
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
            "kl_c_logit_mog",
            "dirichlet_penalty",
            "usage_sparsity_loss",
            "component_diversity",
            "classification_loss",
            "contrastive_loss",
            "component_entropy",
            "pi_entropy",
        )
        for key in tracked_keys:
            train_value = float(train_metrics.get(key, 0.0))
            val_value = float(val_metrics.get(key, 0.0))
            history[key].append(train_value)
            history[f"val_{key}"].append(val_value)

    def _compute_normality_score_for_trigger(
        self,
        params: Dict[str, Dict[str, jnp.ndarray]],
        splits: DataSplits,
        runtime: ModelRuntime,
        k_active: int,
    ) -> float:
        """Compute latent normality score for trigger-based unlock.

        Uses a sample of validation data to compute how close the latent
        posterior q(z|x) is to N(0,I) over active channels.

        Args:
            params: Model parameters.
            splits: Data splits.
            runtime: Model runtime with forward function.
            k_active: Current number of active channels.

        Returns:
            Normality score (lower = closer to N(0,I)).
        """
        from rcmvae.application.services.curriculum_trigger import compute_normality_score

        # Use a sample of validation data (up to 256 samples for efficiency)
        sample_size = min(256, splits.val_size if splits.val_size > 0 else splits.train_size)
        if sample_size == 0:
            return 0.0

        if splits.val_size > 0:
            sample_x = splits.x_val[:sample_size]
        else:
            sample_x = splits.x_train[:sample_size]

        # Forward pass to get latent statistics
        # Use the model's apply function without training noise
        try:
            forward_output = runtime.model.apply(
                {"params": params},
                sample_x,
                training=False,
                k_active=k_active,
            )
            _, z_mean, z_log_var, _, _, _, extras = forward_output

            # Check for per-component latents (decentralized layout)
            z_mean_pc = extras.get("z_mean_per_component")
            z_log_pc = extras.get("z_log_var_per_component")

            if z_mean_pc is not None and z_log_pc is not None:
                # Decentralized layout
                return compute_normality_score(
                    z_mean_pc, z_log_pc, k_active, latent_layout="decentralized"
                )
            else:
                # Shared layout
                return compute_normality_score(
                    z_mean, z_log_var, k_active, latent_layout="shared"
                )
        except Exception:
            # If forward pass fails, return 0 (don't block training)
            return 0.0

    @staticmethod
    def _run_callbacks(
        callbacks: Sequence[TrainingCallback],
        method: str,
        *args,
    ) -> None:
        for callback in callbacks:
            getattr(callback, method)(*args)
