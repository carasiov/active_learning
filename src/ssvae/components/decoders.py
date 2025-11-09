from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn


# ============================================================================
# Component-Aware Decoders (Separate processing for z and component embeddings)
# ============================================================================

class ComponentAwareDenseDecoder(nn.Module):
    """Component-aware dense decoder that processes z and component embeddings separately.

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


class ComponentAwareConvDecoder(nn.Module):
    """Component-aware convolutional decoder for MNIST (28x28).

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

        # Separate pathways for z and component embedding
        z_features = nn.Dense(7 * 7 * 96, name="z_pathway")(z)  # 96 channels for z
        e_features = nn.Dense(7 * 7 * 32, name="component_pathway")(component_embedding)  # 32 for component

        # Combine and reshape to spatial
        combined = jnp.concatenate([z_features, e_features], axis=-1)
        x = combined.reshape((-1, 7, 7, 128))  # Total: 96 + 32 = 128 channels

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
# Standard Decoders (Backward compatibility)
# ============================================================================

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


class ConvDecoder(nn.Module):
    """Convolutional decoder reconstructing images from latent vectors."""

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
