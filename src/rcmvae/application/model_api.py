"""SSVAE public API and orchestration layer.

This module exposes the user-facing `SSVAE` class and delegates core
responsibilities to focused services:
- ``ModelFactoryService`` – model creation and initialization
- ``CheckpointManager`` – saving and loading training state
- ``DiagnosticsCollector`` – mixture and latent-space diagnostics
- ``Trainer`` – JAX training loop with early stopping and callbacks
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import warnings
from jax import random
from jax.nn import softmax
from rcmvae.utils import configure_jax_device, print_device_banner

configure_jax_device()

import jax.numpy as jnp

from rcmvae.application.callbacks import (
    CSVExporter,
    ConsoleLogger,
    LossCurvePlotter,
    MixtureHistoryTracker,
    TrainingCallback,
)
from rcmvae.application.runtime.context import ModelRuntime
from rcmvae.application.runtime.state import SSVAETrainState
from rcmvae.application.runtime.types import EvalMetricsFn, MetricsDict, TrainStepFn
from rcmvae.application.services.checkpoint_service import CheckpointManager
from rcmvae.application.services.diagnostics_service import DiagnosticsCollector
from rcmvae.application.services.factory_service import ModelFactoryService
from rcmvae.application.services.training_service import Trainer, TrainerLoopHooks
from rcmvae.domain.config import SSVAEConfig


COMPONENT_PRIORS = {"mixture", "geometric_mog", "vamp"}


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

    def __init__(self, input_dim: Tuple[int, int], config: SSVAEConfig | None = None, init_data: np.ndarray | None = None):
        """Initialize SSVAE model.

        Args:
            input_dim: Input image dimensions (height, width)
            config: Model configuration (uses defaults if None)
            init_data: Optional training data for VampPrior pseudo-input initialization.
                      If provided and prior_type="vamp", initializes pseudo-inputs
                      before creating the optimizer. This avoids pytree mismatch errors.
        """
        self.input_dim = input_dim
        self.config = config or SSVAEConfig()
        self.latent_dim = self.config.latent_dim
        self.weights_path: str | None = None

        # Device banner and model configuration shown in experiment header
        # Commented out to avoid duplicate output
        # print(f"\n{'=' * 60}")
        # print("Model Configuration")
        # print(f"{'=' * 60}")
        # print(f"Prior type: {self.config.prior_type}")
        # print(f"Latent dim: {self.config.latent_dim}")
        # if self.config.prior_type in COMPONENT_PRIORS:
        #     print(f"Components (K): {self.config.num_components}")
        #     if self.config.prior_type in ["mixture", "geometric_mog"]:
        #         print(f"Learnable π: {self.config.learnable_pi}")
        #         if self.config.learnable_pi and self.config.dirichlet_alpha is not None:
        #             print(f"Dirichlet α: {self.config.dirichlet_alpha}")
        #             print(f"Dirichlet weight: {self.config.dirichlet_weight}")
        #     print(f"Component-aware decoder: {self.config.use_component_aware_decoder}")
        #     if self.config.use_tau_classifier:
        #         print(f"τ-classifier enabled: True")
        # print(f"Batch size: {self.config.batch_size}")
        # print(f"Learning rate: {self.config.learning_rate}")
        # print(f"{'=' * 60}\n")

        # Create components using factory
        factory = ModelFactoryService()
        self._runtime = factory.build_runtime(input_dim, self.config, init_data=init_data)
        self.prior = self._runtime.prior

        # Initialize managers
        self._checkpoint_mgr = CheckpointManager()
        self._diagnostics = DiagnosticsCollector(self.config)
        self._trainer = Trainer(self.config)
        self._mixture_metrics: Dict = {}  # Store mixture diagnostics metrics

        # Initialize τ-classifier for mixture prior with latent-only classification
        self._tau_classifier = None
        if self.config.is_mixture_based_prior() and self.config.use_tau_classifier:
            from rcmvae.domain.components.tau_classifier import TauClassifier
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

    @property
    def runtime(self) -> ModelRuntime:
        return self._runtime

    def _set_runtime(self, runtime: ModelRuntime) -> None:
        self._runtime = runtime

    @property
    def state(self) -> SSVAETrainState:
        return self._runtime.state

    @state.setter
    def state(self, new_state: SSVAETrainState) -> None:
        self._runtime = self._runtime.replace(state=new_state)

    @property
    def _train_step(self) -> TrainStepFn:
        """Expose current train_step for tests and advanced integrations."""
        return self._runtime.train_step_fn

    @_train_step.setter
    def _train_step(self, fn: TrainStepFn) -> None:
        self._runtime = self._runtime.replace(train_step_fn=fn)

    @property
    def _eval_metrics(self) -> EvalMetricsFn:
        """Expose current eval metrics fn for tests and adapters."""
        return self._runtime.eval_metrics_fn

    @_eval_metrics.setter
    def _eval_metrics(self, fn: EvalMetricsFn) -> None:
        self._runtime = self._runtime.replace(eval_metrics_fn=fn)

    def initialize_pseudo_inputs(
        self,
        data: np.ndarray,
        method: str | None = None,
    ) -> None:
        """Initialize VampPrior pseudo-inputs from data.

        Must be called after model creation and before training for VampPrior.
        Uses config.vamp_pseudo_init_method if method not specified.

        Args:
            data: Training images [N, H, W]
            method: Initialization method ("random" or "kmeans"). 
                   If None, uses config.vamp_pseudo_init_method

        Raises:
            ValueError: If not using VampPrior or method is invalid
        """
        from rcmvae.domain.priors.vamp import VampPrior

        if not isinstance(self.prior, VampPrior):
            raise ValueError(
                "initialize_pseudo_inputs() only applies to VampPrior. "
                f"Current prior type: {self.config.prior_type}"
            )

        # Use config default if method not specified
        if method is None:
            method = self.config.vamp_pseudo_init_method

        valid_methods = {"random", "kmeans"}
        if method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got '{method}'"
            )

        K = self.config.num_components
        H, W = self.input_dim
        data_np = np.asarray(data, dtype=np.float32)

        if data_np.shape[0] < K:
            raise ValueError(
                f"Not enough data samples ({data_np.shape[0]}) to initialize "
                f"{K} pseudo-inputs. Need at least {K} samples."
            )

        print(f"Initializing VampPrior pseudo-inputs using {method} method...")

        if method == "random":
            # Sample random images from data
            self._rng, subkey = random.split(self._rng)
            indices = random.choice(
                subkey,
                data_np.shape[0],
                shape=(K,),
                replace=False,
            )
            pseudo_inputs = data_np[indices]  # [K, H, W]
            print(f"  Selected {K} random samples from {data_np.shape[0]} images")

        elif method == "kmeans":
            # Run k-means clustering on flattened data
            try:
                from sklearn.cluster import KMeans
            except ImportError:
                raise ImportError(
                    "scikit-learn required for kmeans initialization. "
                    "Install with: pip install scikit-learn"
                )

            # Flatten data for k-means
            X_flat = data_np.reshape(data_np.shape[0], -1)  # [N, H*W]

            # Run k-means
            kmeans = KMeans(
                n_clusters=K,
                random_state=self.config.random_seed,
                n_init=10,
                max_iter=300,
                verbose=0,
            )
            kmeans.fit(X_flat)

            # Reshape cluster centers back to image shape
            pseudo_inputs = kmeans.cluster_centers_.reshape(K, H, W)  # [K, H, W]
            print(
                f"  K-means converged in {kmeans.n_iter_} iterations "
                f"(inertia={kmeans.inertia_:.2f})"
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        # Convert to JAX array
        pseudo_inputs_jax = jnp.array(pseudo_inputs, dtype=jnp.float32)

        # Update model parameters - just replace the values in-place
        # The optimizer will handle the updated params on first gradient step
        from flax.core import freeze, unfreeze
        
        # Unfreeze, update, and refreeze to preserve FrozenDict structure
        params_dict = unfreeze(self.state.params)
        if "prior" not in params_dict:
            params_dict["prior"] = {}
        params_dict["prior"]["pseudo_inputs"] = pseudo_inputs_jax
        new_params = freeze(params_dict)
        
        # Update only the params, not optimizer state
        # The optimizer state tree structure will be fine - it just tracks the params
        self.state = self.state.replace(params=new_params)

        print(f"  ✓ Pseudo-inputs initialized with shape {pseudo_inputs_jax.shape}")

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

        # Build curriculum snapshot function if curriculum is enabled
        curriculum_snapshot_fn = None
        if self.config.curriculum_enabled:
            output_dir = Path(self.weights_path).parent
            curriculum_snapshot_fn = self._build_curriculum_snapshot_fn(
                data=data,
                labels=labels,
                output_dir=output_dir,
            )

        loop_hooks = self._build_tau_loop_hooks() if self._tau_classifier else None
        self._runtime, history = self._trainer.train(
            self._runtime,
            data=data,
            labels=labels,
            weights_path=self.weights_path,
            save_fn=self._checkpoint_mgr.save,
            callbacks=callbacks,
            loop_hooks=loop_hooks,
            curriculum_snapshot_fn=curriculum_snapshot_fn,
        )

        # Update RNG from state
        self._rng = self.state.rng

        # Generate diagnostics for any component-based prior
        if (
            self.config.prior_type in COMPONENT_PRIORS
            and self._trainer.latest_splits is not None
        ):
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

        # τ-classifier info shown in experiment header
        # Warnings below are important for misconfiguration detection
        # message_lines = [
        #     f"INFO: τ-classifier training with {labeled_count} labeled samples "
        #     f"across {observed_classes}/{total_classes} classes.",
        #     "      Component specialization will develop gradually - more labels will "
        #     f"improve separation. Current regime: {regime}.",
        #     f"      Configured for {num_components} components; ensure labeled data remains proportional.",
        #     "",
        #     "Note: Baseline testing uses ~500 labeled samples. Performance improves "
        #     "with more labels, but the model is designed for label-efficient learning.",
        # ]
        # print("\n".join(message_lines), flush=True)

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
            With component priors: (latent, reconstruction, class_predictions, certainty, q_c, π)
        """
        x = jnp.array(data, dtype=jnp.float32)
        mixture_active = self.config.prior_type in COMPONENT_PRIORS

        if return_mixture and not mixture_active:
            raise ValueError("return_mixture=True only supported for component-based priors.")

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

        # Handle heteroscedastic decoder (returns tuple)
        if isinstance(recon, tuple):
            recon_np = (np.array(recon[0]), np.array(recon[1]))
        else:
            recon_np = np.array(recon)

        result = (
            np.array(z_mean),
            recon_np,
            np.array(pred_class, dtype=np.int32),
            np.array(pred_certainty),
        )

        if return_mixture:
            responsibilities = self._get_responsibilities_from_extras(extras)
            if responsibilities is None:
                raise ValueError("Mixture responsibilities unavailable.")
            pi_val = extras.get("pi") if extras is not None and hasattr(extras, "get") else None
            if pi_val is None:
                pi_val = jnp.ones((self.config.num_components,)) / self.config.num_components
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
        logits_stack = jnp.stack(logits_samples) if num_samples > 1 else logits_samples[0]

        # Handle heteroscedastic decoder (recon_samples contains tuples)
        if isinstance(recon_samples[0], tuple):
            # Stack means and sigmas separately
            means = [r[0] for r in recon_samples]
            sigmas = [r[1] for r in recon_samples]
            mean_stack = jnp.stack(means) if num_samples > 1 else means[0]
            sigma_stack = jnp.stack(sigmas) if num_samples > 1 else sigmas[0]
            recon_stack = (mean_stack, sigma_stack)
        else:
            recon_stack = jnp.stack(recon_samples) if num_samples > 1 else recon_samples[0]

        pred_class, pred_certainty = self._resolve_predictions(logits_stack, last_responsibilities)

        # Convert recon_stack to numpy
        if isinstance(recon_stack, tuple):
            recon_np = (np.array(recon_stack[0]), np.array(recon_stack[1]))
        else:
            recon_np = np.array(recon_stack)

        result = (
            np.array(latent_stack),
            recon_np,
            np.array(pred_class, dtype=np.int32),
            np.array(pred_certainty),
        )

        if return_mixture:
            q_stack = jnp.stack(resp_samples) if num_samples > 1 else resp_samples[0]
            pi_np = np.array(pi_value) if pi_value is not None else None
            result += (np.array(q_stack), pi_np)

        return result

    def _default_predict_batch_size(self) -> int:
        """Heuristic batch size that stays GPU-friendly for heavy decoders."""

        # Start from training batch size when available, else fall back.
        target = self.config.batch_size or 128

        # Convolutional + heteroscedastic decoders allocate large workspaces.
        if self.config.decoder_type == "conv" or self.config.use_heteroscedastic_decoder:
            target = min(target, 64)

        # Keep a reasonable floor to avoid excessive looping on tiny configs.
        return max(32, target)

    def predict_batched(
        self,
        data: np.ndarray,
        *,
        batch_size: int | None = None,
        sample: bool = False,
        num_samples: int = 1,
        return_mixture: bool = False,
    ) -> Tuple:
        """Perform batched prediction to avoid OOM with large datasets and conv architectures.
        
        This method splits large datasets into smaller batches to prevent out-of-memory errors
        that can occur with convolutional architectures due to large intermediate tensors.
        
        Args:
            data: Input images [N, H, W]
            batch_size: Batch size for prediction (defaults to safe heuristic based on config)
            sample: Whether to sample from latent distribution
            num_samples: Number of samples to draw (if sample=True)
            return_mixture: Return mixture-specific outputs (responsibilities, π)
            
        Returns:
            Standard: (latent, reconstruction, class_predictions, certainty)
            With mixture: (latent, reconstruction, class_predictions, certainty, q_c, π)
        """
        inferred_batch_size = batch_size or self._default_predict_batch_size()
        total = data.shape[0]
        if total == 0 or total <= inferred_batch_size:
            return self.predict(
                data,
                sample=sample,
                num_samples=num_samples,
                return_mixture=return_mixture,
            )
        
        # Collect results from each batch
        latent_batches = []
        recon_batches = []
        recon_mean_batches = []
        recon_sigma_batches = []
        pred_batches = []
        cert_batches = []
        resp_batches = [] if return_mixture else None
        pi_value = None
        heteroscedastic_mode: bool | None = None
        
        for start in range(0, total, inferred_batch_size):
            end = min(start + inferred_batch_size, total)
            batch_data = data[start:end]
            
            if return_mixture:
                latent, recon, preds, cert, resp, pi = self.predict(
                    batch_data,
                    sample=sample,
                    num_samples=num_samples,
                    return_mixture=True,
                )
                resp_batches.append(resp)
                if pi_value is None:
                    pi_value = pi  # π is constant across batches
            else:
                latent, recon, preds, cert = self.predict(
                    batch_data, sample=sample, num_samples=num_samples
                )
            
            latent_batches.append(latent)

            if isinstance(recon, tuple):
                if heteroscedastic_mode is False:
                    raise ValueError("Mixed heteroscedastic and standard reconstructions in batched prediction.")
                heteroscedastic_mode = True
                recon_mean_batches.append(recon[0])
                recon_sigma_batches.append(recon[1])
            else:
                if heteroscedastic_mode is True:
                    raise ValueError("Mixed heteroscedastic and standard reconstructions in batched prediction.")
                heteroscedastic_mode = False
                recon_batches.append(recon)

            pred_batches.append(preds)
            cert_batches.append(cert)
        
        # Concatenate batches
        latent_all = np.concatenate(latent_batches, axis=0)
        if heteroscedastic_mode:
            mean_all = np.concatenate(recon_mean_batches, axis=0)
            sigma_all = np.concatenate(recon_sigma_batches, axis=0)
            recon_all: np.ndarray | Tuple[np.ndarray, np.ndarray] = (mean_all, sigma_all)
        else:
            recon_all = np.concatenate(recon_batches, axis=0)
        pred_all = np.concatenate(pred_batches, axis=0)
        cert_all = np.concatenate(cert_batches, axis=0)
        
        if return_mixture:
            resp_all = np.concatenate(resp_batches, axis=0)
            if pi_value is None:
                pi_value = jnp.ones((self.config.num_components,))
            return (
                latent_all,
                recon_all,
                pred_all,
                cert_all,
                resp_all,
                np.array(pi_value),
            )
        else:
            return latent_all, recon_all, pred_all, cert_all

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

        # Add mixture history tracker for component-based priors
        if self.config.prior_type in COMPONENT_PRIORS:
            mixture_hist_dir = base_path.parent / "diagnostics" / base_path.stem
            callbacks.append(
                MixtureHistoryTracker(
                    output_dir=mixture_hist_dir,
                    log_every=self.config.mixture_history_log_every,
                    enabled=True,
                )
            )

        return callbacks

    def _build_curriculum_snapshot_fn(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        output_dir: Path,
    ):
        """Build curriculum snapshot function for training loop.

        Returns a closure that captures the necessary context (data, labels,
        output_dir, apply_fn, diagnostics) and can be called during training
        to save latent snapshots at curriculum events.

        Args:
            data: Full training data
            labels: Labels for training data
            output_dir: Base directory for saving snapshots

        Returns:
            Callable that takes (params, epoch, k_active, event_type)
        """
        # Capture references to instance methods/objects
        diagnostics = self._diagnostics
        apply_fn = self._apply_fn

        def snapshot_fn(params: Dict, epoch: int, k_active: int, event_type: str):
            """Save curriculum snapshot at a training event."""
            diagnostics.save_curriculum_snapshot(
                apply_fn=apply_fn,
                params=params,
                data=data,
                labels=labels,
                output_dir=output_dir,
                epoch=epoch,
                k_active=k_active,
                event_type=event_type,
            )

        return snapshot_fn

    def _save_weights(self, state: SSVAETrainState, path: str) -> None:
        """Helper for adapters that expect a bound save function."""
        self._checkpoint_mgr.save(state, path)

    def _save_mixture_diagnostics(self, splits: Trainer.DataSplits) -> None:
        """Generate and save component-prior diagnostics (mixture/vamp/geometric)."""
        if self.config.prior_type not in COMPONENT_PRIORS or splits is None:
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
        """Get component-prior diagnostics metrics (K_eff, etc.)."""
        return self._mixture_metrics


# Repo root and default paths
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = (
    REPO_ROOT / "use_cases" / "experiments" / "results" / "checkpoints" / "ssvae.ckpt"
)
