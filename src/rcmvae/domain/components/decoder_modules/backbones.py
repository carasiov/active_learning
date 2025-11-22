from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn


class ConvBackbone(nn.Module):
    """Convolutional decoder backbone (latent → spatial feature map)."""

    latent_dim: int
    output_hw: Tuple[int, int]

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        if self.output_hw != (28, 28):
            raise ValueError(f"ConvBackbone expects output_hw of (28, 28) but received {self.output_hw!r}")

        x = nn.Dense(7 * 7 * 128, name="projection")(z)
        x = x.reshape((-1, 7, 7, 128))
        x = nn.ConvTranspose(
            features=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            name="deconv_0",
        )(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.ConvTranspose(
            features=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            name="deconv_1",
        )(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        return x


class DenseBackbone(nn.Module):
    """Dense decoder backbone (latent → hidden features)."""

    hidden_dims: Tuple[int, ...]

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        x = z
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f"hidden_{i}")(x)
            x = nn.leaky_relu(x)
        return x
