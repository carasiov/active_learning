"""Unit tests for mixture KL divergence loss function."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from training.losses import kl_divergence, kl_divergence_mixture


def test_kl_mixture_matches_manual_calculation():
    """Verify kl_divergence_mixture matches manual implementation."""
    batch_size = 4
    K = 3
    latent_dim = 2
    weight = 0.1
    
    # Create mock data
    component_logits = jnp.array([
        [1.0, 0.5, 0.2],
        [0.3, 2.0, 0.1],
        [0.5, 0.5, 1.5],
        [1.5, 0.3, 0.8],
    ])
    z_mean = jnp.ones((batch_size, latent_dim)) * 0.5
    z_log = jnp.ones((batch_size, latent_dim)) * (-0.5)
    
    # Computed via function
    kl_computed = kl_divergence_mixture(component_logits, z_mean, z_log, weight)
    
    # Manual calculation
    import jax
    responsibilities = jax.nn.softmax(component_logits, axis=-1)
    
    # KL per component
    kl_per_comp = -0.5 * (1.0 + z_log - jnp.square(z_mean) - jnp.exp(z_log))
    kl_per_comp_sum = jnp.sum(kl_per_comp, axis=-1)
    
    # Component entropy
    log_resp = jnp.log(responsibilities + 1e-10)
    comp_entropy = -jnp.sum(responsibilities * log_resp, axis=-1)
    
    # KL(q(c) || Uniform)
    kl_comp = jnp.log(float(K)) - comp_entropy
    
    # Total
    mixture_kl = kl_per_comp_sum + kl_comp
    kl_manual = weight * jnp.mean(mixture_kl)
    
    assert jnp.allclose(kl_computed, kl_manual, atol=1e-5)


def test_kl_mixture_reduces_to_standard_when_K_equals_1():
    """Verify mixture KL approximates standard KL when K=1."""
    batch_size = 8
    latent_dim = 2
    weight = 0.1
    
    z_mean = jnp.array(np.random.randn(batch_size, latent_dim), dtype=jnp.float32)
    z_log = jnp.array(np.random.randn(batch_size, latent_dim) * 0.5, dtype=jnp.float32)
    
    # K=1 case (deterministic component)
    component_logits = jnp.zeros((batch_size, 1))
    
    kl_mix = kl_divergence_mixture(component_logits, z_mean, z_log, weight)
    kl_std = kl_divergence(z_mean, z_log, weight)
    
    # Should be very close (difference only from log(1)=0 term)
    assert jnp.allclose(kl_mix, kl_std, atol=1e-4)


def test_component_entropy_metric_range():
    """Verify component entropy is in [0, log(K)] range."""
    batch_size = 16
    K = 10
    
    # Uniform responsibilities → max entropy = log(K)
    component_logits_uniform = jnp.zeros((batch_size, K))
    
    import jax
    responsibilities = jax.nn.softmax(component_logits_uniform, axis=-1)
    log_resp = jnp.log(responsibilities + 1e-10)
    entropy = -jnp.mean(jnp.sum(responsibilities * log_resp, axis=-1))
    
    expected_max_entropy = jnp.log(float(K))
    assert jnp.allclose(entropy, expected_max_entropy, atol=1e-3)
    
    # Deterministic responsibilities → min entropy = 0
    component_logits_deterministic = jnp.array([[10.0] + [0.0] * (K-1)] * batch_size)
    responsibilities_det = jax.nn.softmax(component_logits_deterministic, axis=-1)
    log_resp_det = jnp.log(responsibilities_det + 1e-10)
    entropy_det = -jnp.mean(jnp.sum(responsibilities_det * log_resp_det, axis=-1))
    
    assert entropy_det < 0.1  # Should be close to 0


def test_kl_mixture_finite_values():
    """Verify mixture KL returns finite values for valid inputs."""
    batch_size = 32
    K = 5
    latent_dim = 2
    
    component_logits = jnp.array(np.random.randn(batch_size, K), dtype=jnp.float32)
    z_mean = jnp.array(np.random.randn(batch_size, latent_dim), dtype=jnp.float32)
    z_log = jnp.array(np.random.randn(batch_size, latent_dim) * 0.5, dtype=jnp.float32)
    
    kl = kl_divergence_mixture(component_logits, z_mean, z_log, weight=0.1)
    
    assert jnp.isfinite(kl)
    assert kl >= 0.0  # KL divergence is non-negative
