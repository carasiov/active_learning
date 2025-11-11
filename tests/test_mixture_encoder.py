"""Unit tests for mixture encoder implementation."""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from callbacks.mixture_tracking import MixtureHistoryTracker
from ssvae.components.encoders import DenseEncoder, MixtureConvEncoder, MixtureDenseEncoder


def test_mixture_encoder_output_shapes():
    """Verify MixtureDenseEncoder returns correct output shapes."""
    batch_size = 32
    K = 5
    latent_dim = 2
    hidden_dims = (128, 64)
    
    encoder = MixtureDenseEncoder(
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        num_components=K,
    )
    
    rng = random.PRNGKey(0)
    params_key, reparam_key = random.split(rng)
    dummy_input = jnp.zeros((batch_size, 28, 28), dtype=jnp.float32)
    
    variables = encoder.init(
        {"params": params_key, "reparam": reparam_key},
        dummy_input,
        training=True,
    )
    
    component_logits, z_mean, z_log, z = encoder.apply(
        variables,
        dummy_input,
        training=True,
        rngs={"reparam": reparam_key},
    )
    
    assert component_logits.shape == (batch_size, K)
    assert z_mean.shape == (batch_size, latent_dim)
    assert z_log.shape == (batch_size, latent_dim)
    assert z.shape == (batch_size, latent_dim)


def test_mixture_encoder_responsibilities_sum_to_one():
    """Verify responsibilities from component_logits sum to 1.0."""
    batch_size = 16
    K = 10
    
    encoder = MixtureDenseEncoder(
        hidden_dims=(64,),
        latent_dim=2,
        num_components=K,
    )
    
    rng = random.PRNGKey(42)
    params_key, reparam_key = random.split(rng)
    dummy_input = jnp.zeros((batch_size, 28, 28), dtype=jnp.float32)
    
    variables = encoder.init(
        {"params": params_key, "reparam": reparam_key},
        dummy_input,
        training=True,
    )
    
    component_logits, _, _, _ = encoder.apply(
        variables,
        dummy_input,
        training=False,
    )
    
    responsibilities = jax.nn.softmax(component_logits, axis=-1)
    resp_sums = jnp.sum(responsibilities, axis=-1)
    
    assert jnp.allclose(resp_sums, 1.0, atol=1e-6)


def test_standard_encoder_unchanged():
    """Verify DenseEncoder still returns 3-tuple format."""
    batch_size = 8
    latent_dim = 2
    
    encoder = DenseEncoder(
        hidden_dims=(128, 64),
        latent_dim=latent_dim,
    )
    
    rng = random.PRNGKey(0)
    params_key, reparam_key = random.split(rng)
    dummy_input = jnp.zeros((batch_size, 28, 28), dtype=jnp.float32)
    
    variables = encoder.init(
        {"params": params_key, "reparam": reparam_key},
        dummy_input,
        training=True,
    )
    
    output = encoder.apply(
        variables,
        dummy_input,
        training=True,
        rngs={"reparam": reparam_key},
    )
    
    assert len(output) == 3
    z_mean, z_log, z = output
    assert z_mean.shape == (batch_size, latent_dim)
    assert z_log.shape == (batch_size, latent_dim)
    assert z.shape == (batch_size, latent_dim)


def test_mixture_encoder_without_reparam_rng():
    """Verify mixture encoder works without reparam RNG (returns z_mean)."""
    batch_size = 4
    
    encoder = MixtureDenseEncoder(
        hidden_dims=(64,),
        latent_dim=2,
        num_components=3,
    )
    
    rng = random.PRNGKey(0)
    dummy_input = jnp.zeros((batch_size, 28, 28), dtype=jnp.float32)
    
    variables = encoder.init(
        {"params": rng},
        dummy_input,
        training=False,
    )
    
    component_logits, z_mean, z_log, z = encoder.apply(
        variables,
        dummy_input,
        training=False,
    )
    
    # Without reparam RNG, z should equal z_mean
    assert jnp.allclose(z, z_mean, atol=1e-6)


def test_mixture_conv_encoder_output_shapes():
    """MixtureConvEncoder should emit logits and latent stats with correct shapes."""
    batch_size = 16
    num_components = 4
    latent_dim = 3

    encoder = MixtureConvEncoder(latent_dim=latent_dim, num_components=num_components)

    rng = random.PRNGKey(7)
    params_key, reparam_key = random.split(rng)
    dummy_input = jnp.zeros((batch_size, 28, 28), dtype=jnp.float32)

    variables = encoder.init(
        {"params": params_key, "reparam": reparam_key},
        dummy_input,
        training=True,
    )

    component_logits, z_mean, z_log, z = encoder.apply(
        variables,
        dummy_input,
        training=True,
        rngs={"reparam": reparam_key},
    )

    assert component_logits.shape == (batch_size, num_components)
    assert z_mean.shape == (batch_size, latent_dim)
    assert z_log.shape == (batch_size, latent_dim)
    assert z.shape == (batch_size, latent_dim)


def test_mixture_conv_encoder_softmax_normalized():
    """Responsibilities derived from conv logits should sum to 1."""
    batch_size = 5
    num_components = 6
    encoder = MixtureConvEncoder(latent_dim=2, num_components=num_components)

    rng = random.PRNGKey(123)
    dummy_input = jnp.zeros((batch_size, 28, 28), dtype=jnp.float32)
    variables = encoder.init({"params": rng}, dummy_input, training=False)

    component_logits, _, _, _ = encoder.apply(
        variables,
        dummy_input,
        training=False,
    )

    responsibilities = jax.nn.softmax(component_logits, axis=-1)
    sums = responsibilities.sum(axis=-1)

    assert jnp.allclose(sums, 1.0, atol=1e-6)


class _DummyState:
    """Minimal stand-in for trainer state to record apply_fn batch sizes."""

    def __init__(self):
        self.params = {}
        self.batch_sizes: list[int] = []

    def apply_fn(self, params, batch, training):
        del params, training
        self.batch_sizes.append(batch.shape[0])
        extras = {
            "responsibilities": jnp.ones((batch.shape[0], 2), dtype=jnp.float32) / 2.0,
            "pi": jnp.array([0.6, 0.4], dtype=jnp.float32),
        }
        output = SimpleNamespace(extras=extras)
        return output


def test_mixture_history_tracker_batches_inputs(tmp_path):
    """MixtureHistoryTracker should chunk inputs to avoid oversized GPU batches."""
    tracker = MixtureHistoryTracker(
        tmp_path / "diag",
        log_every=1,
        max_samples=8,
        eval_batch_size=3,
    )
    tracker.on_train_start(trainer=None)

    x_train = np.random.rand(10, 28, 28).astype(np.float32)
    splits = SimpleNamespace(
        x_train=x_train,
        y_train=np.zeros(10, dtype=np.float32),
    )
    trainer = SimpleNamespace(
        latest_splits=splits,
        _current_state=_DummyState(),
    )

    tracker.on_epoch_end(epoch=0, metrics={}, history={}, trainer=trainer)

    assert trainer._current_state.batch_sizes == [3, 3, 2]
    assert len(tracker.tracked_epochs) == 1
    assert len(tracker.usage_history) == 1
    assert len(tracker.pi_history) == 1
