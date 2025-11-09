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

        # Train using Trainer
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

        # For τ-classifier, logits are already log probabilities
        # For standard classifier, logits need softmax
        if self.config.use_tau_classifier:
            # logits = log(p(y|x)) from τ-classifier
            probs = jnp.exp(logits)  # Convert log probs back to probs
        else:
            # Standard classifier: apply softmax to get probabilities
            probs = softmax(logits, axis=1)

        pred_class = jnp.argmax(probs, axis=1)
        pred_certainty = jnp.max(probs, axis=1)

        # For τ-classifier with mixture prior, use component-aware certainty
        if self.config.use_tau_classifier and self.config.prior_type == "mixture":
            from ssvae.components.tau_classifier import get_certainty, extract_tau_from_params

            responsibilities = extras.get("responsibilities") if hasattr(extras, "get") else None
            if responsibilities is not None:
                try:
                    tau = extract_tau_from_params(self.state.params)
                    # Use τ-based certainty: max_c (r_c · max_y τ_{c,y})
                    pred_certainty = get_certainty(responsibilities, tau)
                except (ValueError, KeyError):
                    # Fall back to max probability if τ not available
                    pass

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

        # Handle τ-classifier vs standard classifier
        if self.config.use_tau_classifier:
            # logits are already log probabilities
            probs = jnp.exp(logits_stack)
        else:
            # Standard classifier: apply softmax
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
