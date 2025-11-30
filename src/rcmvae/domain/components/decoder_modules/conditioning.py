from __future__ import annotations

from typing import Protocol, runtime_checkable

from flax import linen as nn
import jax.numpy as jnp


# Numerical stability constant for normalization
EPS = 1e-5


@runtime_checkable
class Conditioner(Protocol):
    """Protocol for decoder conditioning modules.

    Conditioners modulate decoder features based on component embeddings,
    enabling component-specific reconstruction in mixture-of-VAEs.

    See docs/development/architecture.md for valid configurations.
    """

    def __call__(
        self, features: jnp.ndarray, component_embedding: jnp.ndarray | None
    ) -> jnp.ndarray:
        ...


class ConditionalInstanceNorm(nn.Module):
    """Conditional Instance Normalization (Dumoulin et al., 2017).

    Normalizes features, then applies learned γ/β from component embedding:
        X̂ = γ_c · ((X - μ) / σ) + β_c

    Enables each component to control "rendering style" of reconstructions.
    """

    component_embedding_dim: int
    epsilon: float = EPS

    @nn.compact
    def __call__(self, features: jnp.ndarray, component_embedding: jnp.ndarray) -> jnp.ndarray:
        if component_embedding is None:
            raise ValueError("component_embedding is required for ConditionalInstanceNorm.")
        if component_embedding.shape[-1] != self.component_embedding_dim:
            raise ValueError(
                f"Expected component_embedding dimension {self.component_embedding_dim}, "
                f"got {component_embedding.shape[-1]}"
            )

        feature_dim = features.shape[-1]

        # Generate γ and β from component embedding
        # Initialize to identity transform (γ=1, β=0)
        gamma_beta = nn.Dense(
            2 * feature_dim,
            name="cin_affine",
            kernel_init=nn.initializers.zeros,
        )(component_embedding)
        gamma_raw, beta = jnp.split(gamma_beta, 2, axis=-1)
        gamma = 1.0 + gamma_raw

        # Compute instance statistics and normalize
        if features.ndim == 4:
            # Convolutional: normalize over (H, W)
            axes = (1, 2)
            mu = jnp.mean(features, axis=axes, keepdims=True)
            var = jnp.var(features, axis=axes, keepdims=True)
            features_norm = (features - mu) / jnp.sqrt(var + self.epsilon)
            gamma = gamma[:, None, None, :]
            beta = beta[:, None, None, :]
        elif features.ndim == 2:
            # Dense: normalize over features
            axes = (1,)
            mu = jnp.mean(features, axis=axes, keepdims=True)
            var = jnp.var(features, axis=axes, keepdims=True)
            features_norm = (features - mu) / jnp.sqrt(var + self.epsilon)
        else:
            raise ValueError(
                f"ConditionalInstanceNorm expects 2D or 4D features, got shape {features.shape}"
            )

        return gamma * features_norm + beta


class FiLMLayer(nn.Module):
    """FiLM: Feature-wise Linear Modulation (Perez et al., 2018).

    Applies learned γ/β from component embedding without normalization:
        X̂ = γ_c · X + β_c
    """

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

        broadcast_shape = (gamma.shape[0],) + (1,) * (features.ndim - 2) + (gamma.shape[-1],)
        gamma = gamma.reshape(broadcast_shape)
        beta = beta.reshape(broadcast_shape)
        return features * gamma + beta


class ConcatConditioner(nn.Module):
    """Concatenates projected component embedding with decoder features.

    Note: Doubles the feature dimension for downstream layers.
    """

    component_embedding_dim: int | None = None

    @nn.compact
    def __call__(self, features: jnp.ndarray, component_embedding: jnp.ndarray) -> jnp.ndarray:
        if component_embedding is None:
            raise ValueError("component_embedding is required for ConcatConditioner.")
        if self.component_embedding_dim is not None:
            if component_embedding.shape[-1] != self.component_embedding_dim:
                raise ValueError(
                    f"Expected component_embedding dimension {self.component_embedding_dim}, "
                    f"got {component_embedding.shape[-1]}"
                )

        feature_dim = features.shape[-1]
        projected = nn.Dense(feature_dim, name="component_projection")(component_embedding)

        if features.ndim == 4:
            projected = projected[:, None, None, :]
            projected = jnp.broadcast_to(projected, features.shape)

        return jnp.concatenate([features, projected], axis=-1)


class NoopConditioner(nn.Module):
    """Pass-through conditioner for standard/vamp priors."""

    @nn.compact
    def __call__(self, features: jnp.ndarray, component_embedding: jnp.ndarray | None = None) -> jnp.ndarray:
        return features
