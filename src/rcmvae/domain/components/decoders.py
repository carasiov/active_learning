from __future__ import annotations

from typing import Tuple

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


class ModularConvDecoder(nn.Module):
    """Composable convolutional decoder for mixture-of-VAEs.

    Decoder = Conditioner + Backbone + OutputHead
    Conditioning priority (handled by factory): FiLM → Concat → Noop.
    Output head: heteroscedastic if enabled, else standard mean-only.
    """

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

