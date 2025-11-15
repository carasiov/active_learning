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


class MixtureDenseEncoder(nn.Module):
    """Dense encoder with mixture-of-Gaussians prior producing component assignments."""

    hidden_dims: Tuple[int, ...]
    latent_dim: int
    num_components: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, training: bool) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x = x.reshape((x.shape[0], -1))
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f"hidden_{i}")(x)
            x = nn.leaky_relu(x)

        component_logits = nn.Dense(self.num_components, name="component_logits")(x)
        z_mean = nn.Dense(self.latent_dim, name="z_mean")(x)
        z_log = nn.Dense(self.latent_dim, name="z_log")(x)
        
        if self.has_rng("reparam"):
            eps = random.normal(self.make_rng("reparam"), z_mean.shape)
            z = z_mean + jnp.exp(0.5 * z_log) * eps
        else:
            z = z_mean
        
        return component_logits, z_mean, z_log, z


class ConvEncoder(nn.Module):
    """Convolutional encoder for image inputs producing latent statistics."""

    latent_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, training: bool) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        del training  # Conv blocks are deterministic; kept for API consistency.
        x = x[..., None]
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME", name="conv_0")(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME", name="conv_1")(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME", name="conv_2")(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = x.reshape((x.shape[0], -1))

        z_mean = nn.Dense(self.latent_dim, name="z_mean")(x)
        z_log = nn.Dense(self.latent_dim, name="z_log")(x)
        if self.has_rng("reparam"):
            eps = random.normal(self.make_rng("reparam"), z_mean.shape)
            z = z_mean + jnp.exp(0.5 * z_log) * eps
        else:
            z = z_mean
        return z_mean, z_log, z


class MixtureConvEncoder(nn.Module):
    """Convolutional encoder that also predicts mixture component assignments."""

    latent_dim: int
    num_components: int

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        training: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        del training  # Conv blocks are deterministic; kept for API consistency.
        x = x[..., None]
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME", name="conv_0")(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME", name="conv_1")(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME", name="conv_2")(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = x.reshape((x.shape[0], -1))

        component_logits = nn.Dense(self.num_components, name="component_logits")(x)
        z_mean = nn.Dense(self.latent_dim, name="z_mean")(x)
        z_log = nn.Dense(self.latent_dim, name="z_log")(x)
        if self.has_rng("reparam"):
            eps = random.normal(self.make_rng("reparam"), z_mean.shape)
            z = z_mean + jnp.exp(0.5 * z_log) * eps
        else:
            z = z_mean
        return component_logits, z_mean, z_log, z
