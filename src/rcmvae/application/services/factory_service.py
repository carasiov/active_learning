"""
ModelFactoryService - Pure factory for creating model components.

Separates model initialization from the SSVAE class to make testing
and reuse of components easier.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import traverse_util
from flax.core import FrozenDict, freeze
from jax import random

from rcmvae.application.runtime.context import ModelRuntime
from rcmvae.application.runtime.state import SSVAETrainState
from rcmvae.application.runtime.types import EvalMetricsFn, TrainStepFn
from rcmvae.application.services.loss_pipeline import compute_loss_and_metrics_v2
from rcmvae.domain.components.factory import (
    build_classifier,
    build_decoder,
    build_encoder,
    get_architecture_dims,
)
from rcmvae.domain.config import SSVAEConfig
from rcmvae.domain.network import SSVAENetwork, _make_weight_decay_mask
from rcmvae.domain.priors import get_prior
from rcmvae.domain.priors.base import PriorMode


class ModelFactoryService:
    """Pure factory for creating SSVAE components without side effects."""

    @staticmethod
    def build_runtime(
        input_dim: Tuple[int, int],
        config: SSVAEConfig,
        random_seed: int | None = None,
        init_data: jnp.ndarray | None = None,
    ) -> ModelRuntime:
        """Create complete SSVAE model with train/eval functions.

        Args:
            input_dim: Input image dimensions (height, width)
            config: SSVAE configuration
            random_seed: Random seed for reproducibility (uses config.random_seed if None)
            init_data: Optional training data for VampPrior pseudo-input initialization.
                      If provided and prior_type="vamp", initializes pseudo-inputs
                      BEFORE creating the optimizer to avoid pytree mismatch.

        Returns:
            ModelRuntime tying together network, state, compiled functions, and prior.
        """
        seed = random_seed if random_seed is not None else config.random_seed
        out_hw = input_dim
        if config.input_hw is None:
            config.input_hw = out_hw

        # Build architecture
        enc_dims, dec_dims, clf_dims = get_architecture_dims(config, input_hw=out_hw)
        network = SSVAENetwork(
            config=config,
            input_hw=out_hw,
            encoder_hidden_dims=enc_dims,
            decoder_hidden_dims=dec_dims,
            classifier_hidden_dims=clf_dims,
            classifier_dropout_rate=config.dropout_rate,
            latent_dim=config.latent_dim,
            output_hw=out_hw,
            encoder_type=config.encoder_type,
            decoder_type=config.decoder_type,
            classifier_type=config.classifier_type,
        )

        # Initialize parameters
        rng = random.PRNGKey(seed)
        params_key, sample_key, dropout_key, rng = random.split(rng, 4)
        dummy_input = jnp.zeros((1, *out_hw), dtype=jnp.float32)
        variables = network.init(
            {"params": params_key, "reparam": sample_key, "dropout": dropout_key},
            dummy_input,
            training=True,
        )

        # Initialize VampPrior pseudo-inputs BEFORE creating optimizer if data provided
        if config.prior_type == "vamp" and init_data is not None:
            variables = ModelFactoryService._initialize_vamp_pseudo_inputs(
                variables=variables,
                data=init_data,
                config=config,
                input_shape=out_hw,
                rng=rng,
            )

        # Create optimizer with weight decay masking
        decay_mask = _make_weight_decay_mask(variables["params"])
        opt_core = (
            optax.adamw(
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                mask=decay_mask,
            )
            if config.weight_decay > 0.0
            else optax.adam(config.learning_rate)
        )

        tx_steps = []
        if config.grad_clip_norm is not None:
            tx_steps.append(optax.clip_by_global_norm(config.grad_clip_norm))
        tx_steps.append(opt_core)
        tx = optax.chain(*tx_steps) if len(tx_steps) > 1 else tx_steps[0]

        # Create initial training state
        state = SSVAETrainState.create(
            apply_fn=network.apply,
            params=variables["params"],
            tx=tx,
            rng=rng,
        )

        shuffle_rng = random.PRNGKey(seed + 1)

        # Create prior instance with appropriate parameters
        prior = ModelFactoryService._create_prior(config, input_dim)

        # Build train and eval functions (using protocol-based losses)
        train_step_fn = ModelFactoryService._build_train_step_v2(network, config, state.apply_fn, prior)
        eval_metrics_fn = ModelFactoryService._build_eval_metrics_v2(network, config, state.apply_fn, prior)

        return ModelRuntime(
            network=network,
            state=state,
            train_step_fn=train_step_fn,
            eval_metrics_fn=eval_metrics_fn,
            prior=prior,
            shuffle_rng=shuffle_rng,
        )

    @staticmethod
    def _initialize_vamp_pseudo_inputs(
        variables: Dict,
        data: jnp.ndarray,
        config: SSVAEConfig,
        input_shape: Tuple[int, int],
        rng: random.PRNGKey,
    ) -> Dict:
        """Initialize VampPrior pseudo-inputs before optimizer creation.

        Args:
            variables: Initial model variables from network.init()
            data: Training data [N, H, W]
            config: SSVAE configuration
            input_shape: Input image shape (H, W)
            rng: Random key for sampling

        Returns:
            Updated variables dict with initialized pseudo-inputs
        """
        import numpy as np
        from flax.core import freeze, unfreeze

        method = config.vamp_pseudo_init_method
        K = config.num_components
        H, W = input_shape

        # Convert to numpy for k-means
        data_np = np.asarray(data, dtype=np.float32)
        if data_np.ndim == 2:
            # Already flattened [N, H*W] - reshape to [N, H, W]
            data_np = data_np.reshape(-1, H, W)
        elif data_np.ndim == 3:
            # Expected shape [N, H, W]
            pass
        else:
            raise ValueError(f"Expected data shape [N, H, W] or [N, H*W], got {data_np.shape}")

        if data_np.shape[0] < K:
            raise ValueError(
                f"Not enough data samples ({data_np.shape[0]}) to initialize "
                f"{K} pseudo-inputs. Need at least {K} samples."
            )

        print(f"\nInitializing VampPrior pseudo-inputs using {method} method...")

        if method == "random":
            # Sample random images from data
            rng, subkey = random.split(rng)
            indices = random.choice(
                subkey,
                data_np.shape[0],
                shape=(K,),
                replace=False,
            )
            pseudo_inputs = data_np[np.array(indices)]  # [K, H, W]
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
                random_state=config.random_seed,
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
            raise ValueError(f"Unknown vamp_pseudo_init_method: {method}")

        # Convert to JAX array
        pseudo_inputs_jax = jnp.array(pseudo_inputs, dtype=jnp.float32)

        # Update variables dict (preserving FrozenDict structure)
        params_dict = unfreeze(variables["params"])
        if "prior" not in params_dict:
            params_dict["prior"] = {}
        params_dict["prior"]["pseudo_inputs"] = pseudo_inputs_jax
        variables = {**variables, "params": freeze(params_dict)}

        print(f"  ✓ Pseudo-inputs initialized with shape {pseudo_inputs_jax.shape}\n")

        return variables

    @staticmethod
    def _create_prior(config: SSVAEConfig, input_shape: Tuple[int, int]) -> PriorMode:
        """Create prior instance based on configuration.

        Args:
            config: SSVAE configuration
            input_shape: Input image shape (for VampPrior pseudo-inputs)

        Returns:
            Prior instance configured according to config.prior_type
        """
        if config.prior_type == "standard":
            return get_prior("standard")
        
        elif config.prior_type == "mixture":
            return get_prior("mixture")
        
        elif config.prior_type == "vamp":
            return get_prior(
                "vamp",
                num_components=config.num_components,
                latent_dim=config.latent_dim,
                input_shape=input_shape,
                uniform_weights=True,  # Learnable weights not yet implemented
                num_samples_kl=max(1, config.vamp_num_samples_kl),
            )
        
        elif config.prior_type == "geometric_mog":
            return get_prior(
                "geometric_mog",
                num_components=config.num_components,
                latent_dim=config.latent_dim,
                arrangement=config.geometric_arrangement,
                radius=config.geometric_radius,
            )
        
        else:
            raise ValueError(
                f"Unknown prior_type: {config.prior_type}. "
                f"Valid options: standard, mixture, vamp, geometric_mog"
            )

    @staticmethod
    def _build_train_step_v2(
        network: SSVAENetwork,
        config: SSVAEConfig,
        apply_fn: Callable,
        prior: PriorMode,
    ) -> Callable:
        """Build JIT-compiled training step function using losses_v2."""

        def _apply_fn_wrapper(params, *args, **kwargs):
            return apply_fn({"params": params}, *args, **kwargs)

        def _model_forward(
            params: Dict[str, Dict[str, jnp.ndarray]],
            batch_x: jnp.ndarray,
            *,
            training: bool,
            key: jax.Array | None,
            gumbel_temperature: float | None = None,
            k_active: int | None = None,
        ):
            if key is None:
                return _apply_fn_wrapper(params, batch_x, training=training, gumbel_temperature=gumbel_temperature, k_active=k_active)
            reparam_key, dropout_key, gumbel_key = random.split(key, 3)
            return _apply_fn_wrapper(
                params,
                batch_x,
                training=training,
                rngs={"reparam": reparam_key, "dropout": dropout_key, "gumbel": gumbel_key},
                gumbel_temperature=gumbel_temperature,
                k_active=k_active,
            )

        def _loss_and_metrics(
            params: Dict[str, Dict[str, jnp.ndarray]],
            batch_x: jnp.ndarray,
            batch_y: jnp.ndarray,
            key: jax.Array | None,
            training: bool,
            kl_c_scale: float,
            tau: jnp.ndarray | None = None,
            gumbel_temperature: float | None = None,
            k_active: int | None = None,
        ):
            rng = key if training else None
            return compute_loss_and_metrics_v2(
                params,
                batch_x,
                batch_y,
                _model_forward,
                config,
                prior,
                rng,
                training=training,
                kl_c_scale=kl_c_scale,
                tau=tau,
                gumbel_temperature=gumbel_temperature,
                k_active=k_active,
            )

        train_loss_and_grad = jax.value_and_grad(
            lambda p, x, y, k, scale, t, temp, k_act: _loss_and_metrics(p, x, y, k, True, scale, t, temp, k_act),
            argnums=0,
            has_aux=True,
        )

        @jax.jit
        def train_step(
            state: SSVAETrainState,
            batch_x: jnp.ndarray,
            batch_y: jnp.ndarray,
            key: jax.Array,
            kl_c_scale: float,
            tau: jnp.ndarray | None = None,
            gumbel_temperature: float | None = None,
            k_active: int | None = None,
        ):
            (loss, metrics), grads = train_loss_and_grad(state.params, batch_x, batch_y, key, kl_c_scale, tau, gumbel_temperature, k_active)
            if config.prior_type == "vamp":
                grads = _scale_vamp_pseudo_gradients(grads, config.vamp_pseudo_lr_scale)
            new_state = state.apply_gradients(grads=grads)
            return new_state, metrics

        return train_step

    @staticmethod
    def _build_eval_metrics_v2(
        network: SSVAENetwork,
        config: SSVAEConfig,
        apply_fn: Callable,
        prior: PriorMode,
    ) -> Callable:
        """Build JIT-compiled evaluation metrics function using losses_v2."""

        def _apply_fn_wrapper(params, *args, **kwargs):
            return apply_fn({"params": params}, *args, **kwargs)

        def _model_forward(
            params: Dict[str, Dict[str, jnp.ndarray]],
            batch_x: jnp.ndarray,
            *,
            training: bool,
            key: jax.Array | None,
            gumbel_temperature: float | None = None,
            k_active: int | None = None,
        ):
            if key is None:
                return _apply_fn_wrapper(params, batch_x, training=training, gumbel_temperature=gumbel_temperature, k_active=k_active)
            reparam_key, dropout_key, gumbel_key = random.split(key, 3)
            return _apply_fn_wrapper(
                params,
                batch_x,
                training=training,
                rngs={"reparam": reparam_key, "dropout": dropout_key, "gumbel": gumbel_key},
                gumbel_temperature=gumbel_temperature,
                k_active=k_active,
            )

        def _eval_metrics(
            params: Dict[str, Dict[str, jnp.ndarray]],
            batch_x: jnp.ndarray,
            batch_y: jnp.ndarray,
            tau: jnp.ndarray | None = None,
            k_active: int | None = None,
        ):
            _, metrics = compute_loss_and_metrics_v2(
                params,
                batch_x,
                batch_y,
                _model_forward,
                config,
                prior,
                None,
                training=False,
                kl_c_scale=1.0,
                tau=tau,
                k_active=k_active,
            )
            return metrics

        return jax.jit(_eval_metrics)


def _scale_vamp_pseudo_gradients(
    grads: Dict[str, Dict[str, jnp.ndarray]] | FrozenDict,
    scale: float,
):
    """Scale pseudo-input gradients to honor VampPrior-specific LR adjustments."""
    scale = float(np.clip(scale, 1e-6, 1.0))
    if abs(scale - 1.0) < 1e-9:
        return grads

    is_frozen = isinstance(grads, FrozenDict)
    grads_dict = grads.unfreeze() if is_frozen else dict(grads)

    prior_grads = grads_dict.get("prior")
    if prior_grads is None:
        return grads

    prior_is_frozen = isinstance(prior_grads, FrozenDict)
    prior_dict = prior_grads.unfreeze() if prior_is_frozen else dict(prior_grads)

    pseudo_grad = prior_dict.get("pseudo_inputs")
    if pseudo_grad is None:
        return grads

    prior_dict["pseudo_inputs"] = pseudo_grad * scale

    grads_dict["prior"] = freeze(prior_dict) if prior_is_frozen else prior_dict
    return freeze(grads_dict) if is_frozen else grads_dict
