from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn


class Classifier(nn.Module):
    """Dense classifier operating on latent vectors."""

    hidden_dims: Tuple[int, ...]
    num_classes: int = 10

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        x = z
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f"hidden_{i}")(x)
            x = nn.leaky_relu(x)
        # Return raw logits; downstream code applies softmax where probabilities are needed.
        return nn.Dense(self.num_classes, name="logits")(x)
