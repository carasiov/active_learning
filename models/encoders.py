from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn
from jax import random


class DenseEncoder(nn.Module):
    """Dense encoder mapping images to latent statistics."""

    hidden_dims: Tuple[int, ...]
    latent_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, training: bool) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x = x.reshape((x.shape[0], -1))
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f"hidden_{i}")(x)
            x = nn.leaky_relu(x)

        z_mean = nn.Dense(self.latent_dim, name="z_mean")(x)
        z_log = nn.Dense(self.latent_dim, name="z_log")(x)
        if self.has_rng("reparam"):
            eps = random.normal(self.make_rng("reparam"), z_mean.shape)
            z = z_mean + jnp.exp(0.5 * z_log) * eps
        else:
            z = z_mean
        return z_mean, z_log, z

