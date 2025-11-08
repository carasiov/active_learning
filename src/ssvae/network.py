from __future__ import annotations
"""SSVAE models module (JAX)."""

from pathlib import Path

from typing import Dict, List, NamedTuple, Optional, Tuple

from utils import configure_jax_device, print_device_banner

configure_jax_device()

import numpy as np

from ssvae.config import SSVAEConfig
from ssvae.components.factory import build_classifier, build_decoder, build_encoder, get_architecture_dims
from callbacks import CSVExporter, ConsoleLogger, LossCurvePlotter, TrainingCallback
from training.train_state import SSVAETrainState
from training.trainer import Trainer

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    from jax.nn import softmax
    import optax
    from flax import linen as nn
    from flax import traverse_util
    from flax.core import freeze, FrozenDict
    from flax.serialization import from_bytes, to_bytes
except Exception as e:  # pragma: no cover
    raise ImportError(
        "ssvae requires JAX, Flax, and Optax. Please install jax/jaxlib, flax, and optax."
    ) from e


class ForwardOutput(NamedTuple):
    component_logits: Optional[jnp.ndarray]
    z_mean: jnp.ndarray
    z_log: jnp.ndarray
    z: jnp.ndarray
    recon: jnp.ndarray
    class_logits: jnp.ndarray
    extras: Dict[str, jnp.ndarray]


class MixturePriorParameters(nn.Module):
    """Container for learnable mixture prior parameters."""

    num_components: int
    embed_dim: int

    @nn.compact
    def __call__(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        embeddings = self.param(
            "component_embeddings",
            nn.initializers.normal(stddev=0.02),
            (self.num_components, self.embed_dim),
        )
        pi_logits = self.param(
            "pi_logits",
            nn.initializers.zeros,
            (self.num_components,),
        )
        return embeddings, pi_logits


def _make_weight_decay_mask(params: Dict[str, Dict[str, jnp.ndarray]]):
    """Create a mask matching ``params`` tree type for weight decay.

    Optax's masked wrappers require mask and params to share the same
    pytree structure and node types (e.g., dict vs FrozenDict). Build the
    mask and wrap it to match the container type of ``params``.
    """
    flat_params = traverse_util.flatten_dict(params)
    mask = {}
    for key in flat_params:
        param_name = key[-1]
        apply_decay = param_name not in ("bias", "scale")
        if "prior" in key and param_name in ("pi_logits", "component_embeddings"):
            apply_decay = False
        mask[key] = apply_decay
    unflat = traverse_util.unflatten_dict(mask)
    if isinstance(params, FrozenDict):
        return freeze(unflat)
    return unflat


class SSVAENetwork(nn.Module):
    config: SSVAEConfig
    input_hw: Tuple[int, int]
    encoder_hidden_dims: Tuple[int, ...]
    decoder_hidden_dims: Tuple[int, ...]
    classifier_hidden_dims: Tuple[int, ...]
    classifier_dropout_rate: float
    latent_dim: int
    output_hw: Tuple[int, int]
    encoder_type: str
    decoder_type: str
    classifier_type: str

    def setup(self):
        self.encoder = build_encoder(self.config, input_hw=self.input_hw)
        self.decoder = build_decoder(self.config, input_hw=self.input_hw)
        self.classifier = build_classifier(self.config, input_hw=self.input_hw)
        self.prior_module: MixturePriorParameters | None = None
        if self.config.prior_type == "mixture":
            self.prior_module = MixturePriorParameters(
                name="prior",
                num_components=self.config.num_components,
                embed_dim=self.latent_dim,
            )

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        training: bool,
    ) -> ForwardOutput:
        """Forward pass returning latent statistics, reconstructions, and classifier logits."""
        encoder_output = self.encoder(x, training=training)
        extras: Dict[str, jnp.ndarray] = {}

        if self.config.prior_type == "mixture":
            component_logits, z_mean, z_log, z = encoder_output
            if self.prior_module is None:
                raise ValueError("Mixture prior selected but prior_module was not initialized.")
            embeddings, pi_logits = self.prior_module()
            responsibilities = softmax(component_logits, axis=-1)
            pi = softmax(pi_logits, axis=-1)

            batch_size = z.shape[0]
            num_components = self.config.num_components

            z_tiled = jnp.broadcast_to(z[:, None, :], (batch_size, num_components, self.latent_dim))
            embed_tiled = jnp.broadcast_to(embeddings[None, :, :], (batch_size, num_components, embeddings.shape[-1]))
            decoder_inputs = jnp.concatenate([z_tiled, embed_tiled], axis=-1)
            decoder_inputs_flat = decoder_inputs.reshape((batch_size * num_components, -1))
            recon_per_component_flat = self.decoder(decoder_inputs_flat)
            recon_per_component = recon_per_component_flat.reshape(
                (batch_size, num_components, *self.output_hw)
            )
            expected_recon = jnp.sum(
                responsibilities[..., None, None] * recon_per_component,
                axis=1,
            )
            recon = expected_recon
            extras = {
                "recon_per_component": recon_per_component,
                "responsibilities": responsibilities,
                "pi_logits": pi_logits,
                "pi": pi,
                "component_embeddings": embeddings,
            }
        else:
            z_mean, z_log, z = encoder_output
            component_logits = None
            recon = self.decoder(z)

        logits = self.classifier(z, training=training)
        return ForwardOutput(component_logits, z_mean, z_log, z, recon, logits, extras)



