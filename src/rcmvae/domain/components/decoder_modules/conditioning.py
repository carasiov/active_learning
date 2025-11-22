from __future__ import annotations

from flax import linen as nn
import jax.numpy as jnp


class FiLMLayer(nn.Module):
    """Generates FiLM parameters from a component embedding and modulates features."""

    component_embedding_dim: int

    @nn.compact
    def __call__(self, features: jnp.ndarray, component_embedding: jnp.ndarray) -> jnp.ndarray:
        if component_embedding is None:
            raise ValueError("component_embedding is required for FiLMLayer.")
        if component_embedding.shape[-1] != self.component_embedding_dim:
            raise ValueError(
                f"Expected component_embedding dimension {self.component_embedding_dim}, "
                f"got {component_embedding.shape[-1]}"
            )

        feature_dim = features.shape[-1]
        gamma_beta = nn.Dense(2 * feature_dim, name="film_dense")(component_embedding)
        gamma, beta = jnp.split(gamma_beta, 2, axis=-1)

        # Broadcast FiLM parameters over all non-channel feature dimensions.
        broadcast_shape = (gamma.shape[0],) + (1,) * (features.ndim - 2) + (gamma.shape[-1],)
        gamma = gamma.reshape(broadcast_shape)
        beta = beta.reshape(broadcast_shape)
        return features * gamma + beta


class ConcatConditioner(nn.Module):
    """Concatenates decoder features with projected component embeddings."""

    @nn.compact
    def __call__(self, features: jnp.ndarray, component_embedding: jnp.ndarray) -> jnp.ndarray:
        if component_embedding is None:
            raise ValueError("component_embedding is required for ConcatConditioner.")

        feature_dim = features.shape[-1]
        projected = nn.Dense(feature_dim, name="component_projection")(component_embedding)

        if features.ndim == 4:
            # Broadcast projected embedding across spatial dimensions.
            projected = projected[:, None, None, :]
            projected = jnp.broadcast_to(projected, features.shape)

        return jnp.concatenate([features, projected], axis=-1)


class NoopConditioner(nn.Module):
    """Pass-through conditioner for decoders without component context."""

    @nn.compact
    def __call__(self, features: jnp.ndarray, component_embedding: jnp.ndarray | None = None) -> jnp.ndarray:
        return features
