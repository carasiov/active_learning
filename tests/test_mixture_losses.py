"""Unit tests for Priors v1 loss helpers."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from training.losses import (
    categorical_kl,
    dirichlet_map_penalty,
    reconstruction_loss,
    usage_sparsity_penalty,
    weighted_reconstruction_loss,
)


def test_categorical_kl_zero_when_q_matches_pi():
    batch_size = 8
    num_components = 5
    responsibilities = jnp.full((batch_size, num_components), 1.0 / num_components)
    pi = jnp.full((num_components,), 1.0 / num_components)

    kl = categorical_kl(responsibilities, pi, weight=1.0)

    assert jnp.allclose(kl, 0.0, atol=1e-6)


def test_categorical_kl_positive_when_q_differs():
    responsibilities = jnp.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.1, 0.8],
        ],
        dtype=jnp.float32,
    )
    pi = jnp.array([1 / 3, 1 / 3, 1 / 3], dtype=jnp.float32)

    kl = categorical_kl(responsibilities, pi, weight=1.0)

    assert kl > 0.0


def test_dirichlet_map_penalty_finite_for_common_alphas():
    pi = jnp.array([0.2, 0.3, 0.5], dtype=jnp.float32)

    for alpha in (0.5, 1.0, 2.0):
        penalty = dirichlet_map_penalty(pi, alpha, weight=1.0)
        assert jnp.isfinite(penalty)


def test_weighted_reconstruction_matches_standard_when_single_component():
    batch_size = 4
    pixels = 6
    x = jnp.array(np.random.rand(batch_size, pixels), dtype=jnp.float32)
    recon = jnp.array(np.random.rand(batch_size, pixels), dtype=jnp.float32)
    responsibilities = jnp.ones((batch_size, 1), dtype=jnp.float32)

    weighted_mse = weighted_reconstruction_loss(
        x,
        recon[:, None, :],
        responsibilities,
        weight=0.5,
        loss_type="mse",
    )
    standard_mse = reconstruction_loss(x, recon, weight=0.5, loss_type="mse")

    assert jnp.allclose(weighted_mse, standard_mse, atol=1e-6)


def test_weighted_reconstruction_bce_finite():
    batch_size = 3
    pixels = 10
    x = jnp.array(np.random.rand(batch_size, pixels), dtype=jnp.float32)
    logits = jnp.array(np.random.randn(batch_size, 2, pixels), dtype=jnp.float32)
    responsibilities = jnp.array(
        [
            [0.6, 0.4],
            [0.3, 0.7],
            [0.8, 0.2],
        ],
        dtype=jnp.float32,
    )

    loss = weighted_reconstruction_loss(
        x,
        logits,
        responsibilities,
        weight=1.0,
        loss_type="bce",
    )

    assert jnp.isfinite(loss)
    assert loss >= 0.0


def test_usage_sparsity_penalty_prefers_concentration():
    responsibilities = jnp.array(
        [
            [0.9, 0.05, 0.05],
            [0.8, 0.1, 0.1],
        ],
        dtype=jnp.float32,
    )

    penalty = usage_sparsity_penalty(responsibilities, weight=0.2)

    assert jnp.isfinite(penalty)
    assert penalty <= 0.0


def test_dirichlet_penalty_zero_when_alpha_none():
    pi = jnp.array([0.4, 0.6], dtype=jnp.float32)
    penalty = dirichlet_map_penalty(pi, alpha=None, weight=3.0)
    assert penalty == 0.0
