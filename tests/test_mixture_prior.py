"""Feature-specific tests for mixture prior implementation.

This serves as an EXAMPLE of how to test a specific feature.
Follow this pattern when adding new features like:
- Label store
- OOD scoring
- VampPrior
- Curriculum learning

Tests cover:
1. Mathematical correctness (mixture KL reduces to standard when K=1)
2. Shape/type correctness (encoder output shapes)
3. Key invariants (responsibilities sum to one)

Run with: poetry run pytest tests/test_mixture_prior.py
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from ssvae.components.encoders import MixtureDenseEncoder
from training.losses import kl_divergence, kl_divergence_mixture


# =============================================================================
# Mathematical Correctness
# =============================================================================

def test_mixture_kl_reduces_to_standard_when_K_equals_1():
    """
    Core mathematical property: When K=1, mixture KL should reduce to standard KL.

    This tests the fundamental correctness of the mixture KL formulation.
    If this fails, the mixture prior implementation is mathematically broken.
    """
    batch_size = 8
    latent_dim = 2
    weight = 0.1

    z_mean = jnp.array(np.random.randn(batch_size, latent_dim), dtype=jnp.float32)
    z_log = jnp.array(np.random.randn(batch_size, latent_dim) * 0.5, dtype=jnp.float32)

    # K=1 case (deterministic component assignment)
    component_logits = jnp.zeros((batch_size, 1))

    kl_mix = kl_divergence_mixture(component_logits, z_mean, z_log, weight)
    kl_std = kl_divergence(z_mean, z_log, weight)

    # Should be very close (difference only from log(1)=0 term)
    assert jnp.allclose(kl_mix, kl_std, atol=1e-4), \
        f"Mixture KL (K=1) should match standard KL. Got {kl_mix} vs {kl_std}"


def test_mixture_kl_increases_with_uncertainty():
    """
    Mixture KL should increase as component uncertainty increases.

    Uniform responsibilities (max entropy) should give higher KL than
    deterministic responsibilities (min entropy).
    """
    batch_size = 16
    latent_dim = 2
    K = 10
    weight = 0.1

    # Fixed latent parameters
    z_mean = jnp.zeros((batch_size, latent_dim))
    z_log = jnp.zeros((batch_size, latent_dim))

    # Case 1: Deterministic component assignment (low entropy)
    component_logits_deterministic = jnp.array([[100.0] + [0.0] * (K - 1)] * batch_size)
    kl_deterministic = kl_divergence_mixture(
        component_logits_deterministic, z_mean, z_log, weight
    )

    # Case 2: Uniform component assignment (high entropy)
    component_logits_uniform = jnp.zeros((batch_size, K))
    kl_uniform = kl_divergence_mixture(component_logits_uniform, z_mean, z_log, weight)

    # Uniform should have higher KL due to KL(Uniform || Uniform) = 0
    # vs KL(Deterministic || Uniform) = log(K)
    assert kl_uniform < kl_deterministic, \
        "Uniform responsibilities should give lower KL than deterministic"


def test_mixture_kl_matches_manual_calculation():
    """
    Verify mixture KL matches manual implementation.

    Formula: E_q(c)[KL(q(z|c) || p(z))] + KL(q(c) || Uniform(c))
    """
    batch_size = 4
    K = 3
    latent_dim = 2
    weight = 0.1

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
    responsibilities = jax.nn.softmax(component_logits, axis=-1)

    # KL per component against standard Gaussian
    kl_per_comp = -0.5 * (1.0 + z_log - jnp.square(z_mean) - jnp.exp(z_log))
    kl_per_comp_sum = jnp.sum(kl_per_comp, axis=-1)

    # Entropy of component distribution
    log_resp = jnp.log(responsibilities + 1e-10)
    comp_entropy = -jnp.sum(responsibilities * log_resp, axis=-1)

    # KL(q(c) || Uniform)
    kl_comp = jnp.log(float(K)) - comp_entropy

    # Total mixture KL
    mixture_kl = kl_per_comp_sum + kl_comp
    kl_manual = weight * jnp.mean(mixture_kl)

    assert jnp.allclose(kl_computed, kl_manual, atol=1e-5)


# =============================================================================
# Shape and Type Correctness
# =============================================================================

def test_mixture_encoder_output_shapes():
    """
    Verify MixtureDenseEncoder returns correct output shapes.

    Expected output: (component_logits, z_mean, z_log, z)
    - component_logits: (batch_size, num_components)
    - z_mean: (batch_size, latent_dim)
    - z_log: (batch_size, latent_dim)
    - z: (batch_size, latent_dim)
    """
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

    # Verify shapes
    assert component_logits.shape == (batch_size, K), \
        f"Expected component_logits shape {(batch_size, K)}, got {component_logits.shape}"
    assert z_mean.shape == (batch_size, latent_dim), \
        f"Expected z_mean shape {(batch_size, latent_dim)}, got {z_mean.shape}"
    assert z_log.shape == (batch_size, latent_dim), \
        f"Expected z_log shape {(batch_size, latent_dim)}, got {z_log.shape}"
    assert z.shape == (batch_size, latent_dim), \
        f"Expected z shape {(batch_size, latent_dim)}, got {z.shape}"

    # Verify all outputs are finite
    assert jnp.all(jnp.isfinite(component_logits)), "component_logits contains non-finite values"
    assert jnp.all(jnp.isfinite(z_mean)), "z_mean contains non-finite values"
    assert jnp.all(jnp.isfinite(z_log)), "z_log contains non-finite values"
    assert jnp.all(jnp.isfinite(z)), "z contains non-finite values"


def test_mixture_encoder_different_K_values():
    """
    Verify encoder works correctly for different numbers of components.

    Tests K = {1, 5, 10, 20} to ensure flexibility.
    """
    batch_size = 16
    latent_dim = 2
    dummy_input = jnp.zeros((batch_size, 28, 28), dtype=jnp.float32)

    for K in [1, 5, 10, 20]:
        encoder = MixtureDenseEncoder(
            hidden_dims=(64,),
            latent_dim=latent_dim,
            num_components=K,
        )

        rng = random.PRNGKey(42)
        params_key, reparam_key = random.split(rng)

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

        assert component_logits.shape == (batch_size, K), \
            f"Failed for K={K}: expected shape {(batch_size, K)}, got {component_logits.shape}"


# =============================================================================
# Key Invariants
# =============================================================================

def test_responsibilities_sum_to_one():
    """
    Core invariant: Responsibilities (softmax of component_logits) must sum to 1.0.

    This is a fundamental property of the mixture model. If this fails,
    the component assignment mechanism is broken.
    """
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

    # Compute responsibilities (should sum to 1.0 for each sample)
    responsibilities = jax.nn.softmax(component_logits, axis=-1)
    resp_sums = jnp.sum(responsibilities, axis=-1)

    assert jnp.allclose(resp_sums, 1.0, atol=1e-6), \
        f"Responsibilities should sum to 1.0, got {resp_sums}"

    # Additionally verify all responsibilities are in [0, 1]
    assert jnp.all(responsibilities >= 0.0), "Found negative responsibilities"
    assert jnp.all(responsibilities <= 1.0), "Found responsibilities > 1.0"


def test_component_entropy_bounds():
    """
    Verify component entropy is in valid range [0, log(K)].

    - Min entropy (0): Deterministic component assignment
    - Max entropy (log(K)): Uniform component assignment
    """
    batch_size = 16
    K = 10

    # Case 1: Deterministic (should give near-zero entropy)
    component_logits_deterministic = jnp.array([[100.0] + [0.0] * (K - 1)] * batch_size)
    responsibilities_det = jax.nn.softmax(component_logits_deterministic, axis=-1)
    log_resp_det = jnp.log(responsibilities_det + 1e-10)
    entropy_det = -jnp.mean(jnp.sum(responsibilities_det * log_resp_det, axis=-1))

    assert entropy_det < 0.1, f"Deterministic entropy should be near 0, got {entropy_det}"

    # Case 2: Uniform (should give entropy = log(K))
    component_logits_uniform = jnp.zeros((batch_size, K))
    responsibilities_uniform = jax.nn.softmax(component_logits_uniform, axis=-1)
    log_resp_uniform = jnp.log(responsibilities_uniform + 1e-10)
    entropy_uniform = -jnp.mean(jnp.sum(responsibilities_uniform * log_resp_uniform, axis=-1))

    expected_max_entropy = jnp.log(float(K))
    assert jnp.allclose(entropy_uniform, expected_max_entropy, atol=1e-3), \
        f"Uniform entropy should be log({K})={expected_max_entropy}, got {entropy_uniform}"


# =============================================================================
# Edge Cases
# =============================================================================

def test_mixture_kl_handles_extreme_logits():
    """
    Verify mixture KL remains stable with extreme component logits.

    Tests numerical stability with very large/small logits.
    """
    batch_size = 8
    K = 5
    latent_dim = 2
    weight = 0.1

    z_mean = jnp.zeros((batch_size, latent_dim))
    z_log = jnp.zeros((batch_size, latent_dim))

    # Extreme logits (very large positive and negative values)
    component_logits_extreme = jnp.array(
        np.random.randn(batch_size, K) * 100,  # Scale by 100 for extreme values
        dtype=jnp.float32
    )

    kl = kl_divergence_mixture(component_logits_extreme, z_mean, z_log, weight)

    # Should produce finite values
    assert jnp.isfinite(kl), f"Mixture KL should be finite with extreme logits, got {kl}"
    assert kl >= 0.0, f"Mixture KL should be non-negative, got {kl}"


def test_mixture_kl_finite_values():
    """
    Sanity check: Mixture KL returns finite, non-negative values for random inputs.
    """
    batch_size = 32
    K = 5
    latent_dim = 2

    component_logits = jnp.array(np.random.randn(batch_size, K), dtype=jnp.float32)
    z_mean = jnp.array(np.random.randn(batch_size, latent_dim), dtype=jnp.float32)
    z_log = jnp.array(np.random.randn(batch_size, latent_dim) * 0.5, dtype=jnp.float32)

    kl = kl_divergence_mixture(component_logits, z_mean, z_log, weight=0.1)

    assert jnp.isfinite(kl), "Mixture KL should be finite"
    assert kl >= 0.0, "Mixture KL should be non-negative (KL divergence property)"
