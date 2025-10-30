"""Unit tests for mixture encoder implementation."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import random

from ssvae.components.encoders import DenseEncoder, MixtureDenseEncoder


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
