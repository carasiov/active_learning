from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn


class Classifier(nn.Module):
    """Dense classifier operating on latent vectors."""

    hidden_dims: Tuple[int, ...]
    num_classes: int = 10
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, z: jnp.ndarray, *, training: bool = False) -> jnp.ndarray:
        x = z
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f"hidden_{i}")(x)
            x = nn.leaky_relu(x)
            if self.dropout_rate > 0.0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        # Return raw logits; downstream code applies softmax where probabilities are needed.
        return nn.Dense(self.num_classes, name="logits")(x)
