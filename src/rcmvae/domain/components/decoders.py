from __future__ import annotations

from typing import Tuple, TypeVar, Callable
import functools
import warnings

import jax
import jax.numpy as jnp
from flax import linen as nn

from rcmvae.domain.components.decoder_modules import (
    ConcatConditioner,
    ConvBackbone,
    DenseBackbone,
    FiLMLayer,
    HeteroscedasticHead,
    NoopConditioner,
    StandardHead,
)

T = TypeVar("T")


def deprecated(message: str) -> Callable[[T], T]:
    """Decorator to emit deprecation warnings on module call."""
    def decorator(cls: T) -> T:
        original_call = getattr(cls, "__call__", None)

        if original_call is None:
            return cls

        @functools.wraps(original_call)
        def wrapped(self, *args, **kwargs):
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return original_call(self, *args, **kwargs)

        setattr(cls, "__call__", wrapped)
        return cls

    return decorator


# ============================================================================
# Modular Decoders (Composable conditioning + backbone + output head)
# ============================================================================


class ModularConvDecoder(nn.Module):
    """Composable convolutional decoder for mixture-of-VAEs."""

    conditioner: nn.Module
    backbone: nn.Module
    output_head: nn.Module

    @nn.compact
    def __call__(self, z: jnp.ndarray, component_embedding: jnp.ndarray | None = None):
        features = self.backbone(z)
        conditioned = self.conditioner(features, component_embedding)
        return self.output_head(conditioned)


class ModularDenseDecoder(nn.Module):
    """Composable dense decoder for mixture-of-VAEs."""

    conditioner: nn.Module
    backbone: nn.Module
    output_head: nn.Module

    @nn.compact
    def __call__(self, z: jnp.ndarray, component_embedding: jnp.ndarray | None = None):
        features = self.backbone(z)
        conditioned = self.conditioner(features, component_embedding)
        return self.output_head(conditioned)


# ============================================================================
# Component-Aware Decoders (Separate processing for z and component embeddings)
# ============================================================================

@deprecated("Deprecated: use ModularDenseDecoder with ConcatConditioner and StandardHead.")
class ComponentAwareDenseDecoder(nn.Module):
    """Deprecated. Use ModularDenseDecoder with ConcatConditioner + StandardHead.

    This decoder enables functional specialization per component by:
    1. Processing z through its own pathway
    2. Processing component embedding through its own pathway
    3. Combining them before final projection to image space

    Architecture:
        z → Dense(hidden) → LeakyReLU → z_processed
        e_c → Dense(hidden) → LeakyReLU → e_processed
        [z_processed; e_processed] → Dense layers → output

    Args:
        hidden_dims: Sizes of hidden layers (same as standard decoder)
        output_hw: Output image dimensions (height, width)
        component_embedding_dim: Dimensionality of component embeddings
        latent_dim: Dimensionality of latent vector z
    """

    hidden_dims: Tuple[int, ...]
    output_hw: Tuple[int, int]
    component_embedding_dim: int
    latent_dim: int

    @nn.compact
    def __call__(self, z: jnp.ndarray, component_embedding: jnp.ndarray) -> jnp.ndarray:
        """Decode with component-aware processing.

        Args:
            z: Latent vectors [batch, latent_dim]
            component_embedding: Component embeddings [batch, component_embedding_dim]

        Returns:
            Reconstructed images [batch, height, width]
        """
        # Separate processing pathways
        z_processed = nn.Dense(self.hidden_dims[0] // 2, name="z_pathway")(z)
        z_processed = nn.leaky_relu(z_processed)

        e_processed = nn.Dense(self.hidden_dims[0] // 2, name="component_pathway")(component_embedding)
        e_processed = nn.leaky_relu(e_processed)

        # Combine processed representations
        x = jnp.concatenate([z_processed, e_processed], axis=-1)

        # Continue through remaining hidden layers
        for i, dim in enumerate(self.hidden_dims[1:], start=1):
            x = nn.Dense(dim, name=f"hidden_{i}")(x)
            x = nn.leaky_relu(x)

        # Final projection to image space
        h, w = self.output_hw
        x = nn.Dense(h * w, name="projection")(x)
        return x.reshape((-1, h, w))


@deprecated("Deprecated: use ModularConvDecoder with ConcatConditioner and StandardHead.")
class ComponentAwareConvDecoder(nn.Module):
    """Deprecated. Use ModularConvDecoder with ConcatConditioner + StandardHead.

    This decoder processes z and component embeddings separately before
    combining them in the spatial feature maps.

    Architecture:
        z → Dense(latent_features) → z_features
        e_c → Dense(component_features) → e_features
        [z_features; e_features] → reshape → [7, 7, channels]
        → ConvTranspose → [14, 14, 64]
        → ConvTranspose → [28, 28, 32]
        → Conv → [28, 28, 1]

    Args:
        latent_dim: Dimensionality of latent vector z
        output_hw: Output image dimensions (must be (28, 28))
        component_embedding_dim: Dimensionality of component embeddings
    """

    latent_dim: int
    output_hw: Tuple[int, int]
    component_embedding_dim: int

    @nn.compact
    def __call__(self, z: jnp.ndarray, component_embedding: jnp.ndarray) -> jnp.ndarray:
        """Decode with component-aware processing.

        Args:
            z: Latent vectors [batch, latent_dim]
            component_embedding: Component embeddings [batch, component_embedding_dim]

        Returns:
            Reconstructed images [batch, 28, 28]
        """
        if self.output_hw != (28, 28):
            raise ValueError(
                f"ComponentAwareConvDecoder expects output_hw of (28, 28) "
                f"but received {self.output_hw!r}"
            )

        # Separate pathways for z and component embedding (symmetric capacity per §3.3)
        z_features = nn.Dense(7 * 7 * 64, name="z_pathway")(z)  # 64 channels for z
        e_features = nn.Dense(7 * 7 * 64, name="component_pathway")(component_embedding)  # 64 for component

        # Combine and reshape to spatial
        combined = jnp.concatenate([z_features, e_features], axis=-1)
        x = combined.reshape((-1, 7, 7, 128))  # Total: 64 + 64 = 128 channels

        # Standard conv transpose layers
        x = nn.ConvTranspose(
            features=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            name="deconv_0"
        )(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.ConvTranspose(
            features=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            name="deconv_1"
        )(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.Conv(
            features=1,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            name="recon"
        )(x)
        x = x.squeeze(axis=-1)
        return x


# ============================================================================
# FiLM-Conditioned Decoders (dense)
# ============================================================================

@deprecated("Deprecated: use ModularDenseDecoder with FiLMLayer and StandardHead.")
class FiLMDenseDecoder(nn.Module):
    """Deprecated. Use ModularDenseDecoder with FiLMLayer + StandardHead."""

    hidden_dims: Tuple[int, ...]
    output_hw: Tuple[int, int]
    component_embedding_dim: int
    latent_dim: int

    @nn.compact
    def __call__(self, z: jnp.ndarray, component_embedding: jnp.ndarray) -> jnp.ndarray:
        """Decode latent vectors with FiLM conditioning."""
        x = z
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f"hidden_{i}")(x)

            # Generate FiLM parameters from component embedding
            gamma_beta = nn.Dense(2 * dim, name=f"film_{i}")(component_embedding)
            gamma, beta = jnp.split(gamma_beta, 2, axis=-1)
            x = x * gamma + beta
            x = nn.leaky_relu(x)

        h, w = self.output_hw
        x = nn.Dense(h * w, name="projection")(x)
        return x.reshape((-1, h, w))


@deprecated("Deprecated: use ModularConvDecoder with FiLMLayer and StandardHead (heteroscedastic now supported).")
class FiLMConvDecoder(nn.Module):
    """Deprecated. Use ModularConvDecoder with FiLMLayer + StandardHead (or HeteroscedasticHead)."""

    latent_dim: int
    output_hw: Tuple[int, int]
    component_embedding_dim: int

    def _apply_film(self, x: jnp.ndarray, component_embedding: jnp.ndarray, features: int, name: str) -> jnp.ndarray:
        """Applies FiLM modulation to the feature map."""
        gamma_beta = nn.Dense(2 * features, name=name)(component_embedding)
        gamma, beta = jnp.split(gamma_beta, 2, axis=-1)
        # Reshape for broadcasting: [B, 1, 1, C]
        gamma = gamma[:, None, None, :]
        beta = beta[:, None, None, :]
        return x * gamma + beta

    @nn.compact
    def __call__(self, z: jnp.ndarray, component_embedding: jnp.ndarray) -> jnp.ndarray:
        """Decode latent vectors with FiLM modulation on the initial feature map."""
        if self.output_hw != (28, 28):
            raise ValueError(
                f"FiLMConvDecoder expects output_hw of (28, 28) but received {self.output_hw!r}"
            )

        # Base projection from z
        x = nn.Dense(7 * 7 * 128, name="projection")(z)
        x = x.reshape((-1, 7, 7, 128))

        # FiLM on initial projection
        x = self._apply_film(x, component_embedding, 128, "film_proj")

        x = nn.ConvTranspose(
            features=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            name="deconv_0",
        )(x)
        # FiLM after first deconv
        x = self._apply_film(x, component_embedding, 64, "film_0")
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.ConvTranspose(
            features=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            name="deconv_1",
        )(x)
        # FiLM after second deconv
        x = self._apply_film(x, component_embedding, 32, "film_1")
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.Conv(
            features=1,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            name="recon",
        )(x)
        x = x.squeeze(axis=-1)
        return x


# ============================================================================
# Standard Decoders (Backward compatibility)
# ============================================================================

@deprecated("Deprecated: use ModularDenseDecoder with NoopConditioner and StandardHead.")
class DenseDecoder(nn.Module):
    """Deprecated. Use ModularDenseDecoder with NoopConditioner + StandardHead."""

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


@deprecated("Deprecated: use ModularConvDecoder with NoopConditioner and StandardHead.")
class ConvDecoder(nn.Module):
    """Deprecated. Use ModularConvDecoder with NoopConditioner + StandardHead."""

    latent_dim: int
    output_hw: Tuple[int, int]

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        if self.output_hw != (28, 28):
            raise ValueError(f"ConvDecoder expects output_hw of (28, 28) but received {self.output_hw!r}")

        x = nn.Dense(7 * 7 * 128, name="projection")(z)
        x = x.reshape((-1, 7, 7, 128))
        x = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME", name="deconv_0")(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME", name="deconv_1")(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Conv(features=1, kernel_size=(3, 3), strides=(1, 1), padding="SAME", name="recon")(x)
        x = x.squeeze(axis=-1)
        return x


# ============================================================================
# Heteroscedastic Decoders (Learned per-input variance for aleatoric uncertainty)
# ============================================================================

@deprecated("Deprecated: use ModularDenseDecoder with NoopConditioner and HeteroscedasticHead.")
class HeteroscedasticDenseDecoder(nn.Module):
    """Deprecated. Use ModularDenseDecoder with NoopConditioner + HeteroscedasticHead.

    Outputs both mean reconstruction and variance σ²(x) to model observation noise.
    This enables proper probabilistic modeling: p(x|z) = N(x; μ(z), σ²(z)I).

    The variance is parameterized as:
        σ(x) = σ_min + softplus(s_θ(x))
        with hard clamping: σ ∈ [σ_min, σ_max]

    This provides:
    - Aleatoric (observation) uncertainty quantification
    - Adaptive noise modeling (high σ for ambiguous inputs)
    - Improved reconstruction likelihood via heteroscedastic loss

    Args:
        hidden_dims: Sizes of hidden layers (same as standard decoder)
        output_hw: Output image dimensions (height, width)
        sigma_min: Minimum allowed standard deviation (default: 0.05)
        sigma_max: Maximum allowed standard deviation (default: 0.5)

    Returns:
        Tuple of (mean, sigma):
            - mean: Reconstructed images [batch, height, width]
            - sigma: Per-image standard deviations [batch,]
    """

    hidden_dims: Tuple[int, ...]
    output_hw: Tuple[int, int]
    sigma_min: float = 0.05
    sigma_max: float = 0.5

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Decode latent to mean and variance.

        Args:
            z: Latent vectors [batch, latent_dim]

        Returns:
            mean: Reconstructed images [batch, height, width]
            sigma: Per-image standard deviations [batch,]
        """
        # Shared trunk processing
        x = z
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f"hidden_{i}")(x)
            x = nn.leaky_relu(x)

        # Mean head (standard reconstruction)
        h, w = self.output_hw
        mean = nn.Dense(h * w, name="mean_head")(x)
        mean = mean.reshape((-1, h, w))

        # Variance head (per-image scalar)
        # Use softplus to ensure positivity, then shift and clamp
        sigma_raw = nn.Dense(1, name="sigma_head")(x)  # [batch, 1]
        sigma_raw = sigma_raw.squeeze(-1)  # [batch,]
        sigma = self.sigma_min + jax.nn.softplus(sigma_raw)
        sigma = jnp.clip(sigma, self.sigma_min, self.sigma_max)

        return mean, sigma


@deprecated("Deprecated: use ModularConvDecoder with NoopConditioner and HeteroscedasticHead.")
class HeteroscedasticConvDecoder(nn.Module):
    """Deprecated. Use ModularConvDecoder with NoopConditioner + HeteroscedasticHead.

    Similar to HeteroscedasticDenseDecoder but uses convolutional architecture
    for spatial feature processing. The variance head uses global pooling to
    produce a single scalar σ per image.

    Architecture:
        z → Dense(7×7×128) → reshape → ConvTranspose layers → mean [28, 28]
        z → Dense(7×7×128) → reshape → ConvTranspose layers → global pool → σ [scalar]

    Args:
        latent_dim: Dimensionality of latent vector z
        output_hw: Output image dimensions (must be (28, 28))
        sigma_min: Minimum allowed standard deviation (default: 0.05)
        sigma_max: Maximum allowed standard deviation (default: 0.5)

    Returns:
        Tuple of (mean, sigma):
            - mean: Reconstructed images [batch, 28, 28]
            - sigma: Per-image standard deviations [batch,]
    """

    latent_dim: int
    output_hw: Tuple[int, int]
    sigma_min: float = 0.05
    sigma_max: float = 0.5

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Decode latent to mean and variance.

        Args:
            z: Latent vectors [batch, latent_dim]

        Returns:
            mean: Reconstructed images [batch, 28, 28]
            sigma: Per-image standard deviations [batch,]
        """
        if self.output_hw != (28, 28):
            raise ValueError(
                f"HeteroscedasticConvDecoder expects output_hw of (28, 28) "
                f"but received {self.output_hw!r}"
            )

        # Shared trunk: project and reshape to spatial
        x = nn.Dense(7 * 7 * 128, name="projection")(z)
        x = x.reshape((-1, 7, 7, 128))

        # Mean head: standard conv decoder pathway
        mean = nn.ConvTranspose(
            features=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            name="mean_deconv_0"
        )(x)
        mean = nn.leaky_relu(mean, negative_slope=0.2)
        mean = nn.ConvTranspose(
            features=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            name="mean_deconv_1"
        )(mean)
        mean = nn.leaky_relu(mean, negative_slope=0.2)
        mean = nn.Conv(
            features=1,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            name="mean_recon"
        )(mean)
        mean = mean.squeeze(axis=-1)  # [batch, 28, 28]

        # Variance head: conv processing + global pooling
        sigma_features = nn.ConvTranspose(
            features=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            name="sigma_deconv_0"
        )(x)
        sigma_features = nn.leaky_relu(sigma_features, negative_slope=0.2)
        sigma_features = nn.ConvTranspose(
            features=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            name="sigma_deconv_1"
        )(sigma_features)
        sigma_features = nn.leaky_relu(sigma_features, negative_slope=0.2)

        # Global average pooling to get per-image features
        sigma_pooled = jnp.mean(sigma_features, axis=(1, 2))  # [batch, 32]

        # Project to scalar variance
        sigma_raw = nn.Dense(1, name="sigma_head")(sigma_pooled)  # [batch, 1]
        sigma_raw = sigma_raw.squeeze(-1)  # [batch,]
        sigma = self.sigma_min + jax.nn.softplus(sigma_raw)
        sigma = jnp.clip(sigma, self.sigma_min, self.sigma_max)

        return mean, sigma


@deprecated("Deprecated: use ModularDenseDecoder with ConcatConditioner and HeteroscedasticHead.")
class ComponentAwareHeteroscedasticDenseDecoder(nn.Module):
    """Deprecated. Use ModularDenseDecoder with ConcatConditioner + HeteroscedasticHead.

    Combines component-aware decoding (separate z and embedding pathways) with
    heteroscedastic variance modeling. Each component can learn its own uncertainty
    characteristics.

    Architecture:
        z → Dense(hidden/2) → z_processed
        e_c → Dense(hidden/2) → e_processed
        [z_processed; e_processed] → Dense layers → {mean head, σ head}

    Args:
        hidden_dims: Sizes of hidden layers
        output_hw: Output image dimensions (height, width)
        component_embedding_dim: Dimensionality of component embeddings
        latent_dim: Dimensionality of latent vector z
        sigma_min: Minimum allowed standard deviation (default: 0.05)
        sigma_max: Maximum allowed standard deviation (default: 0.5)

    Returns:
        Tuple of (mean, sigma):
            - mean: Reconstructed images [batch, height, width]
            - sigma: Per-image standard deviations [batch,]
    """

    hidden_dims: Tuple[int, ...]
    output_hw: Tuple[int, int]
    component_embedding_dim: int
    latent_dim: int
    sigma_min: float = 0.05
    sigma_max: float = 0.5

    @nn.compact
    def __call__(
        self,
        z: jnp.ndarray,
        component_embedding: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Decode with component-aware processing and variance prediction.

        Args:
            z: Latent vectors [batch, latent_dim]
            component_embedding: Component embeddings [batch, component_embedding_dim]

        Returns:
            mean: Reconstructed images [batch, height, width]
            sigma: Per-image standard deviations [batch,]
        """
        # Separate processing pathways
        z_processed = nn.Dense(self.hidden_dims[0] // 2, name="z_pathway")(z)
        z_processed = nn.leaky_relu(z_processed)

        e_processed = nn.Dense(
            self.hidden_dims[0] // 2,
            name="component_pathway"
        )(component_embedding)
        e_processed = nn.leaky_relu(e_processed)

        # Combine processed representations
        x = jnp.concatenate([z_processed, e_processed], axis=-1)

        # Continue through remaining hidden layers
        for i, dim in enumerate(self.hidden_dims[1:], start=1):
            x = nn.Dense(dim, name=f"hidden_{i}")(x)
            x = nn.leaky_relu(x)

        # Mean head
        h, w = self.output_hw
        mean = nn.Dense(h * w, name="mean_head")(x)
        mean = mean.reshape((-1, h, w))

        # Variance head
        sigma_raw = nn.Dense(1, name="sigma_head")(x)  # [batch, 1]
        sigma_raw = sigma_raw.squeeze(-1)  # [batch,]
        sigma = self.sigma_min + jax.nn.softplus(sigma_raw)
        sigma = jnp.clip(sigma, self.sigma_min, self.sigma_max)

        return mean, sigma


@deprecated("Deprecated: use ModularConvDecoder with ConcatConditioner and HeteroscedasticHead.")
class ComponentAwareHeteroscedasticConvDecoder(nn.Module):
    """Deprecated. Use ModularConvDecoder with ConcatConditioner + HeteroscedasticHead.

    Combines component-aware decoding with heteroscedastic variance modeling
    in a convolutional architecture. Separate pathways for z and component
    embeddings are combined in the spatial domain.

    Architecture:
        z → Dense(7×7×64) → z_features
        e_c → Dense(7×7×64) → e_features
        [z_features; e_features] → reshape → ConvTranspose layers → {mean, σ}

    Args:
        latent_dim: Dimensionality of latent vector z
        output_hw: Output image dimensions (must be (28, 28))
        component_embedding_dim: Dimensionality of component embeddings
        sigma_min: Minimum allowed standard deviation (default: 0.05)
        sigma_max: Maximum allowed standard deviation (default: 0.5)

    Returns:
        Tuple of (mean, sigma):
            - mean: Reconstructed images [batch, 28, 28]
            - sigma: Per-image standard deviations [batch,]
    """

    latent_dim: int
    output_hw: Tuple[int, int]
    component_embedding_dim: int
    sigma_min: float = 0.05
    sigma_max: float = 0.5

    @nn.compact
    def __call__(
        self,
        z: jnp.ndarray,
        component_embedding: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Decode with component-aware processing and variance prediction.

        Args:
            z: Latent vectors [batch, latent_dim]
            component_embedding: Component embeddings [batch, component_embedding_dim]

        Returns:
            mean: Reconstructed images [batch, 28, 28]
            sigma: Per-image standard deviations [batch,]
        """
        if self.output_hw != (28, 28):
            raise ValueError(
                f"ComponentAwareHeteroscedasticConvDecoder expects output_hw of (28, 28) "
                f"but received {self.output_hw!r}"
            )

        # Separate pathways for z and component embedding (symmetric capacity)
        z_features = nn.Dense(7 * 7 * 64, name="z_pathway")(z)
        e_features = nn.Dense(
            7 * 7 * 64,
            name="component_pathway"
        )(component_embedding)

        # Combine and reshape to spatial
        combined = jnp.concatenate([z_features, e_features], axis=-1)
        x = combined.reshape((-1, 7, 7, 128))  # Total: 64 + 64 = 128 channels

        # Mean head: standard conv decoder pathway
        mean = nn.ConvTranspose(
            features=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            name="mean_deconv_0"
        )(x)
        mean = nn.leaky_relu(mean, negative_slope=0.2)
        mean = nn.ConvTranspose(
            features=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            name="mean_deconv_1"
        )(mean)
        mean = nn.leaky_relu(mean, negative_slope=0.2)
        mean = nn.Conv(
            features=1,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            name="mean_recon"
        )(mean)
        mean = mean.squeeze(axis=-1)  # [batch, 28, 28]

        # Variance head: conv processing + global pooling
        sigma_features = nn.ConvTranspose(
            features=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            name="sigma_deconv_0"
        )(x)
        sigma_features = nn.leaky_relu(sigma_features, negative_slope=0.2)
        sigma_features = nn.ConvTranspose(
            features=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            name="sigma_deconv_1"
        )(sigma_features)
        sigma_features = nn.leaky_relu(sigma_features, negative_slope=0.2)

        # Global average pooling
        sigma_pooled = jnp.mean(sigma_features, axis=(1, 2))  # [batch, 32]

        # Project to scalar variance
        sigma_raw = nn.Dense(1, name="sigma_head")(sigma_pooled)  # [batch, 1]
        sigma_raw = sigma_raw.squeeze(-1)  # [batch,]
        sigma = self.sigma_min + jax.nn.softplus(sigma_raw)
        sigma = jnp.clip(sigma, self.sigma_min, self.sigma_max)

        return mean, sigma
