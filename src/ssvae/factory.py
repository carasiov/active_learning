"""
SSVAEFactory - Pure factory for creating model components.

This module separates model initialization from the SSVAE class,
following the Factory Pattern to centralize component creation,
validation, and wiring.

Purpose:
    - Create compatible encoder/decoder/classifier components
    - Initialize parameters with correct shapes
    - Validate configuration consistency
    - Build train/eval functions with proper JIT compilation
    - Apply weight decay masking selectively

Benefits:
    - Single source of truth for component creation
    - Easy to test in isolation
    - Reusable across different model types
    - Early validation catches configuration errors

Usage:
    Typically called by SSVAE.__init__, but can be used standalone:

    >>> from ssvae.factory import SSVAEFactory
    >>> from ssvae.config import SSVAEConfig
    >>>
    >>> config = SSVAEConfig(latent_dim=2, prior_type="mixture")
    >>> network, state, train_fn, eval_fn, rng, prior = SSVAEFactory.create_model(
    ...     input_dim=(28, 28),
    ...     config=config
    ... )
    >>>
    >>> # Now you have everything needed for training
    >>> state, rng, history = trainer.train(
    ...     state, data, labels, train_step_fn=train_fn, ...
    ... )

Component Creation:
    The factory handles these creation tasks:

    1. Build encoder based on config.encoder_type
    2. Build decoder based on config.decoder_type
    3. Build classifier
    4. Create prior instance (StandardGaussianPrior or MixtureGaussianPrior)
    5. Initialize parameters with dummy input
    6. Create optimizer with weight decay masking
    7. Build JIT-compiled train/eval functions
    8. Return complete bundle ready for training

Weight Decay Masking:
    The factory automatically excludes certain parameters from weight decay:
    - Bias terms (name contains "bias")
    - Scale terms (name contains "scale")
    - Prior parameters (pi_logits, component_embeddings)

    See _make_weight_decay_mask() for implementation.

Extension:
    To add a new component type:
    1. Create the component class (e.g., in components/)
    2. Add case to factory methods (e.g., in create_encoder)
    3. Add config parameter (e.g., encoder_type="my_encoder")
    4. Factory handles the rest automatically

    See CONTRIBUTING.md > Adding Components for guide.

See Also:
    - Component implementations: src/ssvae/components/
    - Prior implementations: src/ssvae/priors/
    - Configuration: src/ssvae/config.py
    - Architecture: docs/development/ARCHITECTURE.md
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn, traverse_util
from flax.core import freeze, FrozenDict
from jax import random

from ssvae.config import SSVAEConfig
from ssvae.components.factory import build_classifier, build_decoder, build_encoder, get_architecture_dims
from ssvae.network import SSVAENetwork, _make_weight_decay_mask
from ssvae.priors import get_prior
from ssvae.priors.base import PriorMode
from training.losses import compute_loss_and_metrics_v2
from training.train_state import SSVAETrainState


class SSVAEFactory:
    """Pure factory for creating SSVAE components without side effects."""

    @staticmethod
    def create_model(
        input_dim: Tuple[int, int],
        config: SSVAEConfig,
        random_seed: int | None = None,
    ) -> Tuple[SSVAENetwork, SSVAETrainState, Callable, Callable, random.PRNGKey, PriorMode]:
        """Create complete SSVAE model with train/eval functions.

        Args:
            input_dim: Input image dimensions (height, width)
            config: SSVAE configuration
            random_seed: Random seed for reproducibility (uses config.random_seed if None)

        Returns:
            Tuple of:
                - network: SSVAENetwork Flax module
                - state: Initial training state with parameters
                - train_step_fn: JIT-compiled training step function
                - eval_metrics_fn: JIT-compiled evaluation function
                - shuffle_rng: RNG key for data shuffling
                - prior: PriorMode instance

        Example:
            >>> factory = SSVAEFactory()
            >>> network, state, train_fn, eval_fn, shuffle_rng, prior = factory.create_model(
            ...     input_dim=(28, 28),
            ...     config=SSVAEConfig(latent_dim=2)
            ... )
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

        # Create prior instance
        prior = get_prior(config.prior_type)

        # Build train and eval functions (using protocol-based losses)
        train_step_fn = SSVAEFactory._build_train_step_v2(network, config, state.apply_fn, prior)
        eval_metrics_fn = SSVAEFactory._build_eval_metrics_v2(network, config, state.apply_fn, prior)

        return network, state, train_step_fn, eval_metrics_fn, shuffle_rng, prior

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
        ):
            if key is None:
                return _apply_fn_wrapper(params, batch_x, training=training)
            reparam_key, dropout_key = random.split(key)
            return _apply_fn_wrapper(
                params,
                batch_x,
                training=training,
                rngs={"reparam": reparam_key, "dropout": dropout_key},
            )

        def _loss_and_metrics(
            params: Dict[str, Dict[str, jnp.ndarray]],
            batch_x: jnp.ndarray,
            batch_y: jnp.ndarray,
            key: jax.Array | None,
            training: bool,
            kl_c_scale: float,
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
            )

        train_loss_and_grad = jax.value_and_grad(
            lambda p, x, y, k, scale: _loss_and_metrics(p, x, y, k, True, scale),
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
        ):
            (loss, metrics), grads = train_loss_and_grad(state.params, batch_x, batch_y, key, kl_c_scale)
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
        ):
            if key is None:
                return _apply_fn_wrapper(params, batch_x, training=training)
            reparam_key, dropout_key = random.split(key)
            return _apply_fn_wrapper(
                params,
                batch_x,
                training=training,
                rngs={"reparam": reparam_key, "dropout": dropout_key},
            )

        def _eval_metrics(
            params: Dict[str, Dict[str, jnp.ndarray]],
            batch_x: jnp.ndarray,
            batch_y: jnp.ndarray,
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
            )
            return metrics

        return jax.jit(_eval_metrics)
