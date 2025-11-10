"""
Refactored SSVAE - Cleaner architecture using focused components.

This is the new implementation that delegates to:
- SSVAEFactory: Model creation
- CheckpointManager: Save/load
- DiagnosticsCollector: Diagnostics generation
- Trainer: Training loop

Once validated, this will replace the original models.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from jax import random
from jax.nn import softmax
from utils import configure_jax_device, print_device_banner

configure_jax_device()

import jax.numpy as jnp

from callbacks import CSVExporter, ConsoleLogger, LossCurvePlotter, MixtureHistoryTracker, TrainingCallback
from ssvae.checkpoint import CheckpointManager
from ssvae.config import SSVAEConfig
from ssvae.diagnostics import DiagnosticsCollector
from ssvae.factory import SSVAEFactory
from training.trainer import Trainer


class SSVAE:
    """
    Modular JAX SSVAE with clean separation of concerns.

    **Public API** (backward compatible):
    - `fit(data, labels, weights_path)` → Train and save model
    - `predict(data, **kwargs)` → Inference
    - `load_model_weights(path)` → Load checkpoint

    **Architecture**:
    - Factory creates model components
    - CheckpointManager handles persistence
    - DiagnosticsCollector generates mixture stats
    - Trainer orchestrates training loop
    """

    _DEVICE_BANNER_PRINTED = False

    def __init__(self, input_dim: Tuple[int, int], config: SSVAEConfig | None = None):
        """Initialize SSVAE model.

        Args:
            input_dim: Input image dimensions (height, width)
            config: Model configuration (uses defaults if None)
        """
        self.input_dim = input_dim
        self.config = config or SSVAEConfig()
        self.latent_dim = self.config.latent_dim
        self.weights_path: str | None = None

        # Print device banner once
        if not SSVAE._DEVICE_BANNER_PRINTED:
            print_device_banner()
            SSVAE._DEVICE_BANNER_PRINTED = True

        # Create components using factory
        factory = SSVAEFactory()
        (
            self.model,
            self.state,
            self._train_step,
            self._eval_metrics,
            self._shuffle_rng,
            self.prior,
        ) = factory.create_model(input_dim, self.config)

        # Initialize managers
        self._checkpoint_mgr = CheckpointManager()
        self._diagnostics = DiagnosticsCollector(self.config)
        self._trainer = Trainer(self.config)
        self._mixture_metrics: Dict = {}  # Store mixture diagnostics metrics

        # Initialize τ-classifier for mixture prior with latent-only classification
        self._tau_classifier = None
        if self.config.prior_type == "mixture" and self.config.use_tau_classifier:
            from ssvae.components.tau_classifier import TauClassifier
            self._tau_classifier = TauClassifier(
                num_components=self.config.num_components,
                num_classes=self.config.num_classes,
                alpha_0=self.config.tau_smoothing_alpha,
            )

        # Build apply function for predictions
        model_apply = self.state.apply_fn

        def apply_fn(params, *args, **kwargs):
            return model_apply({"params": params}, *args, **kwargs)

        self._apply_fn = apply_fn
        self._rng = self.state.rng

    def prepare_data_for_keras_model(self, data: np.ndarray) -> np.ndarray:
        """Legacy method for data preprocessing."""
        return np.where(data == 0, 0.0, 1.0)

    def load_model_weights(self, weights_path: str):
        """Load parameters and optimizer state from checkpoint.

        Args:
            weights_path: Path to checkpoint file
        """
        self.weights_path = str(weights_path)
        self.state = self._checkpoint_mgr.load(self.state, self.weights_path)

    def fit(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        weights_path: str,
        *,
        export_history: bool = True,
    ) -> dict:
        """Train model with semi-supervised labels (NaN = unlabeled).

        Args:
            data: Input images [N, H, W]
            labels: Labels [N] (use np.nan for unlabeled samples)
            weights_path: Path to save best checkpoint
            export_history: Whether to export history CSV and plots

        Returns:
            Dictionary with training history metrics
        """
        self.weights_path = str(weights_path)

        # Build callbacks
        callbacks = self._build_callbacks(
            weights_path=self.weights_path,
            export_history=export_history,
        )

        # Use custom training loop for τ-classifier
        if self._tau_classifier is not None:
            history = self._fit_with_tau_classifier(
                data=data,
                labels=labels,
                weights_path=weights_path,
                callbacks=callbacks,
            )
        else:
            # Standard training path
            self.state, self._shuffle_rng, history = self._trainer.train(
                self.state,
                data=data,
                labels=labels,
                weights_path=self.weights_path,
                shuffle_rng=self._shuffle_rng,
                train_step_fn=self._train_step,
                eval_metrics_fn=self._eval_metrics,
                save_fn=self._checkpoint_mgr.save,
                callbacks=callbacks,
            )

            # Update RNG from state
            self._rng = self.state.rng

        # Generate diagnostics if mixture prior
        if self.config.prior_type == "mixture" and self._trainer.latest_splits is not None:
            self._save_mixture_diagnostics(self._trainer.latest_splits)

        return history

    def _fit_with_tau_classifier(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        weights_path: str,
        callbacks: List[TrainingCallback],
    ) -> dict:
        """Custom training loop with τ-classifier count updates.

        This method wraps the standard Trainer but adds τ-classifier updates:
        1. After each batch, update τ counts from responsibilities
        2. Get current τ matrix
        3. Pass τ to train/eval functions

        Args:
            data: Input images [N, H, W]
            labels: Labels [N] (NaN for unlabeled)
            weights_path: Path to save checkpoint
            callbacks: Training callbacks

        Returns:
            Training history dictionary
        """
        from training.trainer import DataSplits

        # Prepare data splits (replicating Trainer logic)
        state_rng = self.state.rng
        x_np = np.asarray(data, dtype=np.float32)
        y_np = np.asarray(labels, dtype=np.float32).reshape((-1,))

        total_samples = x_np.shape[0]
        if total_samples <= 1:
            val_size = 0
        else:
            val_size = max(1, min(int(self.config.val_split * total_samples), total_samples - 1))
        train_size = total_samples - val_size

        # Shuffle data
        if total_samples > 0:
            state_rng, dataset_key = random.split(state_rng)
            perm = np.asarray(random.permutation(dataset_key, total_samples))
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
            labeled_count=int(np.sum(~np.isnan(y_np))),
        )

        # Store splits for diagnostics
        self._trainer._latest_splits = splits

        # Initialize history
        history = self._trainer._init_history()

        # Initialize early stopping
        from training.trainer import EarlyStoppingTracker
        monitor_metric = self.config.monitor_metric
        if monitor_metric == "auto":
            monitor_metric = "classification_loss" if splits.labeled_count > 0 else "loss"

        tracker = EarlyStoppingTracker(
            monitor_metric=monitor_metric,
            patience=self.config.patience,
        )

        # Log hyperparameters
        print(f"Training with τ-classifier (τ-based latent-only classification)", flush=True)
        print(f"Monitoring validation {monitor_metric} for early stopping.", flush=True)
        if splits.labeled_count > 0:
            print(f"Detected {splits.labeled_count} labeled samples.", flush=True)

        # Run callbacks
        self._trainer._run_callbacks(list(callbacks), "on_train_start", self._trainer)

        # Training loop
        batch_size = self.config.batch_size
        eval_batch_size = min(batch_size, 1024)

        for epoch in range(self.config.max_epochs):
            # KL annealing
            if self.config.kl_c_anneal_epochs > 0:
                kl_c_scale = min(1.0, (epoch + 1) / float(self.config.kl_c_anneal_epochs))
            else:
                kl_c_scale = 1.0

            # Shuffle training data
            self._shuffle_rng, epoch_key = random.split(self._shuffle_rng)
            perm = random.permutation(epoch_key, train_size)
            x_train = jnp.take(splits.x_train, perm, axis=0)
            y_train = jnp.take(splits.y_train, perm, axis=0)

            # Train one epoch with τ updates
            for start in range(0, train_size, batch_size):
                end = min(start + batch_size, train_size)
                batch_x = jnp.asarray(x_train[start:end])
                batch_y = jnp.asarray(y_train[start:end])

                # Get current τ matrix
                current_tau = self._tau_classifier.get_tau() if self._tau_classifier else None

                # Train step
                state_rng, raw_key = random.split(state_rng)
                batch_key = random.fold_in(raw_key, int(self.state.step))
                self.state, batch_metrics = self._train_step(
                    self.state, batch_x, batch_y, batch_key, kl_c_scale, current_tau
                )

                # Update τ counts from responsibilities
                if self._tau_classifier is not None:
                    # Forward pass to get responsibilities (deterministic)
                    forward_output = self._apply_fn(
                        self.state.params, batch_x, training=False
                    )
                    component_logits, z_mean, z_log, z, recon, class_logits, extras = forward_output

                    if hasattr(extras, "get") and extras.get("responsibilities") is not None:
                        responsibilities = extras.get("responsibilities")
                        labeled_mask = jnp.logical_not(jnp.isnan(batch_y))
                        self._tau_classifier.update_counts(
                            responsibilities, batch_y, labeled_mask
                        )

            # Update splits with shuffled data
            splits = splits.with_train(x_train=x_train, y_train=y_train)

            # Evaluate on both splits with current τ
            current_tau = self._tau_classifier.get_tau() if self._tau_classifier else None

            # Evaluate training split
            train_metrics = self._evaluate_with_tau(
                self.state.params, splits.x_train, splits.y_train,
                current_tau, eval_batch_size
            )

            # Evaluate validation split
            if splits.val_size > 0:
                val_metrics = self._evaluate_with_tau(
                    self.state.params, splits.x_val, splits.y_val,
                    current_tau, eval_batch_size
                )
            else:
                val_metrics = train_metrics

            # Update history
            self._trainer._update_history(history, train_metrics, val_metrics)

            # Store state for callback access
            self._trainer._current_state = self.state

            # Run callbacks
            metrics_bundle = {"train": train_metrics, "val": val_metrics}
            self._trainer._run_callbacks(
                list(callbacks), "on_epoch_end", epoch, metrics_bundle, history, self._trainer
            )

            # Early stopping check
            current_val = float(val_metrics[monitor_metric])
            should_stop = tracker.update(
                current_val,
                state=self.state,
                weights_path=weights_path,
                save_fn=self._checkpoint_mgr.save,
            )
            if should_stop:
                break

        # Training complete
        if tracker.checkpoint_saved:
            status = "Early stopping" if tracker.halted_early else "Training complete"
            print(
                f"{status}. Best {tracker.monitor_metric} = {tracker.best_val:.4f} "
                f"(checkpoint saved to {weights_path})."
            )

        # Update RNG
        self.state = self.state.replace(rng=state_rng)
        self._rng = self.state.rng

        # Clear state reference
        self._trainer._current_state = None

        # Run end callbacks
        self._trainer._run_callbacks(list(callbacks), "on_train_end", history, self._trainer)

        return history

    def _evaluate_with_tau(
        self,
        params,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        tau: jnp.ndarray | None,
        batch_size: int,
    ):
        """Evaluate in chunks with τ matrix."""
        total = inputs.shape[0]
        if total == 0:
            return self._eval_metrics(params, inputs, targets, tau)

        metrics_sum = None
        processed = 0
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_inputs = jnp.asarray(inputs[start:end])
            batch_targets = jnp.asarray(targets[start:end])
            batch_metrics = self._eval_metrics(params, batch_inputs, batch_targets, tau)
            weight = end - start
            if metrics_sum is None:
                metrics_sum = {k: batch_metrics[k] * weight for k in batch_metrics}
            else:
                for key in metrics_sum:
                    metrics_sum[key] = metrics_sum[key] + batch_metrics[key] * weight
            processed += weight

        return {k: metrics_sum[k] / processed for k in metrics_sum}

    def predict(
        self,
        data: np.ndarray,
        *,
        sample: bool = False,
        num_samples: int = 1,
        return_mixture: bool = False,
    ) -> Tuple:
        """Perform inference on data.

        Args:
            data: Input images [N, H, W]
            sample: Whether to sample from latent distribution
            num_samples: Number of samples to draw (if sample=True)
            return_mixture: Return mixture-specific outputs (responsibilities, π)

        Returns:
            Standard: (latent, reconstruction, class_predictions, certainty)
            With mixture: (latent, reconstruction, class_predictions, certainty, q_c, π)
        """
        x = jnp.array(data, dtype=jnp.float32)
        mixture_active = self.config.prior_type == "mixture"

        if return_mixture and not mixture_active:
            raise ValueError("return_mixture=True only supported for mixture priors.")

        if sample:
            return self._predict_with_sampling(x, num_samples, return_mixture)
        else:
            return self._predict_deterministic(x, return_mixture)

    def _predict_deterministic(self, x: jnp.ndarray, return_mixture: bool) -> Tuple:
        """Deterministic prediction (use mean of latent distribution)."""
        forward = self._apply_fn(self.state.params, x, training=False)
        component_logits, z_mean, _, _, recon, logits, extras = forward

        # Use τ-classifier predictions if available
        if self._tau_classifier is not None and hasattr(extras, "get"):
            responsibilities = extras.get("responsibilities")
            if responsibilities is not None:
                # τ-based latent-only classification
                pred_class, probs = self._tau_classifier.predict(responsibilities)
                pred_certainty = self._tau_classifier.get_certainty(responsibilities)
            else:
                # Fallback to standard classifier
                probs = softmax(logits, axis=1)
                pred_class = jnp.argmax(probs, axis=1)
                pred_certainty = jnp.max(probs, axis=1)
        else:
            # Standard classifier head
            probs = softmax(logits, axis=1)
            pred_class = jnp.argmax(probs, axis=1)
            pred_certainty = jnp.max(probs, axis=1)

        result = (
            np.array(z_mean),
            np.array(recon),
            np.array(pred_class, dtype=np.int32),
            np.array(pred_certainty),
        )

        if return_mixture:
            responsibilities = extras.get("responsibilities") if hasattr(extras, "get") else None
            pi_val = extras.get("pi") if hasattr(extras, "get") else None
            if responsibilities is None or pi_val is None:
                raise ValueError("Mixture responsibilities unavailable.")
            result += (np.array(responsibilities), np.array(pi_val))

        return result

    def _predict_with_sampling(
        self, x: jnp.ndarray, num_samples: int, return_mixture: bool
    ) -> Tuple:
        """Prediction with sampling from latent distribution."""
        num_samples = max(1, int(num_samples))

        latent_samples = []
        recon_samples = []
        logits_samples = []
        resp_samples = []
        pi_value = None

        for _ in range(num_samples):
            self._rng, subkey = random.split(self._rng)
            forward = self._apply_fn(
                self.state.params,
                x,
                training=False,
                rngs={"reparam": subkey},
            )
            component_logits, z_mean, _, z, recon, logits, extras = forward

            latent_samples.append(z)
            recon_samples.append(recon)
            logits_samples.append(logits)

            if return_mixture:
                responsibilities = extras.get("responsibilities") if hasattr(extras, "get") else None
                if responsibilities is None:
                    raise ValueError("Mixture responsibilities unavailable.")
                resp_samples.append(responsibilities)
                if pi_value is None:
                    pi_val = extras.get("pi") if hasattr(extras, "get") else None
                    if pi_val is not None:
                        pi_value = pi_val

        # Stack samples
        latent_stack = jnp.stack(latent_samples) if num_samples > 1 else latent_samples[0]
        recon_stack = jnp.stack(recon_samples) if num_samples > 1 else recon_samples[0]
        logits_stack = jnp.stack(logits_samples) if num_samples > 1 else logits_samples[0]

        # Use τ-classifier if available (using last sample's responsibilities)
        if self._tau_classifier is not None and resp_samples:
            # For sampling, use the last sample's responsibilities for prediction
            last_resp = resp_samples[-1] if resp_samples else None
            if last_resp is not None:
                pred_class, probs = self._tau_classifier.predict(last_resp)
                pred_certainty = self._tau_classifier.get_certainty(last_resp)
            else:
                # Fallback to standard classifier
                probs = softmax(logits_stack, axis=-1)
                if num_samples > 1:
                    pred_class = jnp.argmax(probs, axis=-1)
                    pred_certainty = jnp.max(probs, axis=-1)
                else:
                    pred_class = jnp.argmax(probs, axis=1)
                    pred_certainty = jnp.max(probs, axis=1)
        else:
            # Standard classifier head
            probs = softmax(logits_stack, axis=-1)
            if num_samples > 1:
                pred_class = jnp.argmax(probs, axis=-1)
                pred_certainty = jnp.max(probs, axis=-1)
            else:
                pred_class = jnp.argmax(probs, axis=1)
                pred_certainty = jnp.max(probs, axis=1)

        result = (
            np.array(latent_stack),
            np.array(recon_stack),
            np.array(pred_class, dtype=np.int32),
            np.array(pred_certainty),
        )

        if return_mixture:
            q_stack = jnp.stack(resp_samples) if num_samples > 1 else resp_samples[0]
            pi_np = np.array(pi_value) if pi_value is not None else None
            result += (np.array(q_stack), pi_np)

        return result

    def _build_callbacks(
        self, *, weights_path: str | None, export_history: bool
    ) -> List[TrainingCallback]:
        """Build callbacks for training loop."""
        callbacks: List[TrainingCallback] = [ConsoleLogger()]

        if not export_history:
            return callbacks

        base_path = Path(weights_path) if weights_path else Path("experiments/runs/checkpoints/ssvae.ckpt")
        history_path = base_path.with_name(f"{base_path.stem}_history.csv")
        plot_path = base_path.with_name(f"{base_path.stem}_loss.png")

        callbacks.append(CSVExporter(history_path))
        callbacks.append(LossCurvePlotter(plot_path))

        # Add mixture history tracker for mixture priors
        if self.config.prior_type == "mixture":
            mixture_hist_dir = base_path.parent / "diagnostics" / base_path.stem
            callbacks.append(
                MixtureHistoryTracker(
                    output_dir=mixture_hist_dir,
                    log_every=self.config.mixture_history_log_every,
                    enabled=True,
                )
            )

        return callbacks

    def _save_mixture_diagnostics(self, splits: Trainer.DataSplits) -> None:
        """Generate and save mixture prior diagnostics."""
        if self.config.prior_type != "mixture" or splits is None:
            return

        # Use validation split if available, otherwise training split
        val_x = np.asarray(splits.x_val)
        val_y = np.asarray(splits.y_val)
        if val_x.size == 0:
            val_x = np.asarray(splits.x_train)
            val_y = np.asarray(splits.y_train)
        if val_x.size == 0:
            return

        # Determine output directory
        if self.weights_path:
            base_path = Path(self.weights_path)
            diag_dir = base_path.parent / "diagnostics" / base_path.stem
        else:
            diag_dir = Path("artifacts/diagnostics/ssvae")

        # Collect diagnostics using DiagnosticsCollector
        self._mixture_metrics = self._diagnostics.collect_mixture_stats(
            apply_fn=self._apply_fn,
            params=self.state.params,
            data=val_x,
            labels=val_y,
            output_dir=diag_dir,
            batch_size=min(self.config.batch_size, 1024),
        )

    @property
    def last_diagnostics_dir(self) -> Path | None:
        """Get directory where diagnostics were last saved."""
        return self._diagnostics.last_output_dir

    @property
    def mixture_metrics(self) -> Dict:
        """Get mixture diagnostics metrics (K_eff, etc.)."""
        return self._mixture_metrics


# Repo root and default paths
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "runs" / "checkpoints" / "ssvae.ckpt"
