from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn


class DenseDecoder(nn.Module):
    """Dense decoder projecting latent vectors back to image space."""

    hidden_dims: Tuple[int, ...]
    output_hw: Tuple[int, int]

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        x = z
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f"hidden_{i}")(x)
            x = nn.leaky_relu(x)
        h, w = self.output_hw
        x = nn.Dense(h * w, name="projection")(x)
        return x.reshape((-1, h, w))

