from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


class StandardHead(nn.Module):
    """Mean-only reconstruction head for decoder outputs.

    Outputs probabilities in [0, 1] via sigmoid activation.
    Use with MSE reconstruction loss.
    """

    output_hw: Tuple[int, int]

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        h, w = self.output_hw
        if features.ndim == 4:
            mean = nn.Conv(
                features=1,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                name="recon",
            )(features)
            mean = nn.sigmoid(mean)
            mean = mean.squeeze(axis=-1)
            return mean

        mean = nn.Dense(h * w, name="projection")(features)
        mean = nn.sigmoid(mean)
        return mean.reshape((-1, h, w))


class LogitsHead(nn.Module):
    """Logits-only reconstruction head for decoder outputs.

    Outputs raw logits (unbounded) without sigmoid activation.
    Use with BCE reconstruction loss for numerical stability.
    The BCE loss function applies sigmoid internally.
    """

    output_hw: Tuple[int, int]

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        h, w = self.output_hw
        if features.ndim == 4:
            logits = nn.Conv(
                features=1,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                name="recon",
            )(features)
            logits = logits.squeeze(axis=-1)
            return logits

        logits = nn.Dense(h * w, name="projection")(features)
        return logits.reshape((-1, h, w))


class HeteroscedasticHead(nn.Module):
    """Mean + variance head with clamped σ for stability."""

    output_hw: Tuple[int, int]
    sigma_min: float = 0.05
    sigma_max: float = 0.5

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        h, w = self.output_hw

        if features.ndim == 4:
            mean = nn.Conv(
                features=1,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                name="mean_recon",
            )(features)
            mean = mean.squeeze(axis=-1)

            sigma_features = nn.Conv(
                features=32,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                name="sigma_conv",
            )(features)
            sigma_features = nn.leaky_relu(sigma_features, negative_slope=0.2)
            sigma_pooled = jnp.mean(sigma_features, axis=(1, 2))
            sigma_raw = nn.Dense(1, name="sigma_head")(sigma_pooled)
            sigma_raw = sigma_raw.squeeze(-1)
        else:
            mean = nn.Dense(h * w, name="mean_head")(features)
            mean = mean.reshape((-1, h, w))
            sigma_raw = nn.Dense(1, name="sigma_head")(features)
            sigma_raw = sigma_raw.squeeze(-1)

        sigma = self.sigma_min + jax.nn.softplus(sigma_raw)
        sigma = jnp.clip(sigma, self.sigma_min, self.sigma_max)
        return mean, sigma
