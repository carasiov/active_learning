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
from typing import Dict, List, Tuple

import numpy as np
import warnings
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
from training.trainer import MetricsDict, Trainer, TrainerLoopHooks
from training.train_state import SSVAETrainState


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
        data = np.asarray(data, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32).reshape((-1,))

        self.weights_path = str(weights_path)

        if self._tau_classifier is not None:
            self._log_tau_data_requirements(labels)

        # Build callbacks
        callbacks = self._build_callbacks(
            weights_path=self.weights_path,
            export_history=export_history,
        )

        loop_hooks = self._build_tau_loop_hooks() if self._tau_classifier else None
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
            loop_hooks=loop_hooks,
        )

        # Update RNG from state
        self._rng = self.state.rng

        # Generate diagnostics if mixture prior
        if self.config.prior_type == "mixture" and self._trainer.latest_splits is not None:
            self._save_mixture_diagnostics(self._trainer.latest_splits)

        return history


    @staticmethod
    def _summarize_labeled_data(labels: np.ndarray) -> Tuple[int, int]:
        """Return (num_labeled_samples, num_classes_observed)."""
        if labels.size == 0:
            return 0, 0

        labeled_mask = ~np.isnan(labels)
        labeled_count = int(np.sum(labeled_mask))
        if labeled_count == 0:
            return 0, 0

        observed_classes = int(np.unique(labels[labeled_mask]).size)
        return labeled_count, observed_classes

    @staticmethod
    def _describe_label_regime(labeled_count: int) -> str:
        """Map label count to a qualitative training regime."""
        if labeled_count <= 0:
            return "zero-label regime"
        if labeled_count < 25:
            return "few-shot learning mode"
        if labeled_count < 200:
            return "low-data regime"
        return "standard regime"

    def _log_tau_data_requirements(self, labels: np.ndarray) -> None:
        """Log guidance and emit warnings based on labeled data availability."""
        labeled_count, observed_classes = self._summarize_labeled_data(labels)
        regime = self._describe_label_regime(labeled_count)
        num_components = self.config.num_components
        total_classes = self.config.num_classes

        message_lines = [
            f"INFO: τ-classifier training with {labeled_count} labeled samples "
            f"across {observed_classes}/{total_classes} classes.",
            "      Component specialization will develop gradually - more labels will "
            f"improve separation. Current regime: {regime}.",
            f"      Configured for {num_components} components; ensure labeled data remains proportional.",
            "",
            "Note: Baseline testing uses ~500 labeled samples. Performance improves "
            "with more labels, but the model is designed for label-efficient learning.",
        ]
        print("\n".join(message_lines), flush=True)

        if labeled_count == 0:
            warnings.warn(
                "No labeled samples detected. τ-classifier counts will remain at the prior; "
                "provide labeled data or disable use_tau_classifier.",
                UserWarning,
            )
            return

        if labeled_count < 10:
            warnings.warn(
                "τ-classifier received fewer than 10 labeled samples; expect limited component specialization.",
                UserWarning,
            )
        elif observed_classes < total_classes:
            warnings.warn(
                "Some classes have no labeled samples. Expect delayed separation until each class is represented.",
                UserWarning,
            )

        if labeled_count < num_components:
            print(
                "      Tip: Labeled samples are fewer than configured components — "
                "consider reducing num_components or collecting more labels.",
                flush=True,
            )

    @staticmethod
    def _extract_extras_from_forward(forward_output):
        """Return the extras object from a network forward pass."""
        if hasattr(forward_output, "extras"):
            return forward_output.extras
        if isinstance(forward_output, (tuple, list)) and forward_output:
            return forward_output[-1]
        return None

    @staticmethod
    def _get_responsibilities_from_extras(extras):
        if extras is None or not hasattr(extras, "get"):
            return None
        return extras.get("responsibilities")

    def _resolve_predictions(
        self,
        logits: jnp.ndarray,
        responsibilities: jnp.ndarray | None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return (predicted_class, certainty) for deterministic or sampled logits."""
        if self._tau_classifier is not None and responsibilities is not None:
            pred_class, _ = self._tau_classifier.predict(responsibilities)
            certainty = self._tau_classifier.get_certainty(responsibilities)
            return pred_class, certainty

        probs = softmax(logits, axis=-1)
        pred_class = jnp.argmax(probs, axis=-1)
        certainty = jnp.max(probs, axis=-1)
        return pred_class, certainty

    def _build_tau_loop_hooks(self) -> TrainerLoopHooks | None:
        """Construct Trainer hooks that keep τ-classifier updates in sync with the loop."""
        if self._tau_classifier is None:
            return None

        def batch_context_fn(
            state: SSVAETrainState,
            batch_x: jnp.ndarray,
            batch_y: jnp.ndarray,
        ) -> Dict[str, jnp.ndarray]:
            del state, batch_x, batch_y  # Unused in context construction
            return {"tau": self._tau_classifier.get_tau()}

        def post_batch_fn(
            state: SSVAETrainState,
            batch_x: jnp.ndarray,
            batch_y: jnp.ndarray,
            batch_metrics: MetricsDict,
        ) -> None:
            del batch_metrics  # Not needed for τ updates
            forward_output = self._apply_fn(state.params, batch_x, training=False)
            extras = self._extract_extras_from_forward(forward_output)
            responsibilities = self._get_responsibilities_from_extras(extras)

            if responsibilities is None:
                return

            labeled_mask = jnp.logical_not(jnp.isnan(batch_y))
            self._tau_classifier.update_counts(responsibilities, batch_y, labeled_mask)

        def eval_context_fn() -> Dict[str, jnp.ndarray]:
            return {"tau": self._tau_classifier.get_tau()}

        return TrainerLoopHooks(
            batch_context_fn=batch_context_fn,
            post_batch_fn=post_batch_fn,
            eval_context_fn=eval_context_fn,
        )

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
        extras = self._extract_extras_from_forward(forward)
        _, z_mean, _, _, recon, logits, _ = forward

        responsibilities = self._get_responsibilities_from_extras(extras)
        pred_class, pred_certainty = self._resolve_predictions(logits, responsibilities)

        result = (
            np.array(z_mean),
            np.array(recon),
            np.array(pred_class, dtype=np.int32),
            np.array(pred_certainty),
        )

        if return_mixture:
            responsibilities = self._get_responsibilities_from_extras(extras)
            pi_val = extras.get("pi") if extras is not None and hasattr(extras, "get") else None
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
        last_responsibilities = None

        for _ in range(num_samples):
            self._rng, subkey = random.split(self._rng)
            forward = self._apply_fn(
                self.state.params,
                x,
                training=False,
                rngs={"reparam": subkey},
            )
            extras = self._extract_extras_from_forward(forward)
            _, z_mean, _, z, recon, logits, _ = forward

            latent_samples.append(z)
            recon_samples.append(recon)
            logits_samples.append(logits)

            responsibilities = self._get_responsibilities_from_extras(extras)
            if return_mixture:
                if responsibilities is None:
                    raise ValueError("Mixture responsibilities unavailable.")
                resp_samples.append(responsibilities)
                if pi_value is None:
                    pi_val = extras.get("pi") if extras is not None and hasattr(extras, "get") else None
                    if pi_val is not None:
                        pi_value = pi_val
            if responsibilities is not None:
                last_responsibilities = responsibilities

        # Stack samples
        latent_stack = jnp.stack(latent_samples) if num_samples > 1 else latent_samples[0]
        recon_stack = jnp.stack(recon_samples) if num_samples > 1 else recon_samples[0]
        logits_stack = jnp.stack(logits_samples) if num_samples > 1 else logits_samples[0]

        pred_class, pred_certainty = self._resolve_predictions(logits_stack, last_responsibilities)

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

        default_ckpt = Path("use_cases/experiments/results/checkpoints/ssvae.ckpt")
        base_path = Path(weights_path) if weights_path else default_ckpt
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
DEFAULT_CHECKPOINT_PATH = (
    REPO_ROOT / "use_cases" / "experiments" / "results" / "checkpoints" / "ssvae.ckpt"
)
