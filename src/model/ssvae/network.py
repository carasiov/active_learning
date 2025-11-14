from __future__ import annotations
"""SSVAE models module (JAX)."""

from pathlib import Path

from typing import Dict, List, NamedTuple, Optional, Tuple

from model.utils import configure_jax_device, print_device_banner

configure_jax_device()

import numpy as np

from model.ssvae.config import SSVAEConfig
from model.ssvae.components.factory import build_classifier, build_decoder, build_encoder, get_architecture_dims
from model.callbacks import CSVExporter, ConsoleLogger, LossCurvePlotter, TrainingCallback
from model.training.train_state import SSVAETrainState
from model.training.trainer import Trainer

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
    recon: jnp.ndarray | Tuple[jnp.ndarray, jnp.ndarray]  # Array or (mean, sigma) tuple
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


class VampPriorParameters(nn.Module):
    """Container for VampPrior pseudo-inputs."""

    num_components: int
    input_shape: Tuple[int, int]

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        """Return pseudo-inputs [K, H, W].
        
        Note: Initialized randomly, should be set from data via
        SSVAE.initialize_pseudo_inputs() for better results.
        """
        pseudo_inputs = self.param(
            "pseudo_inputs",
            nn.initializers.uniform(scale=1.0),  # Random init in [0, 1]
            (self.num_components, *self.input_shape),
        )
        return pseudo_inputs


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
        apply_decay = param_name not in ("bias", "scale", "pseudo_inputs")  # Don't decay pseudo-inputs
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
        self.prior_module: MixturePriorParameters | VampPriorParameters | None = None
        
        if self.config.prior_type == "mixture":
            # Use component_embedding_dim for embeddings (defaults to latent_dim if not specified)
            embed_dim = self.config.component_embedding_dim
            self.prior_module = MixturePriorParameters(
                name="prior",
                num_components=self.config.num_components,
                embed_dim=embed_dim,
            )
        
        elif self.config.prior_type == "vamp":
            # VampPrior: Initialize pseudo-inputs as trainable parameters
            self.prior_module = VampPriorParameters(
                name="prior",
                num_components=self.config.num_components,
                input_shape=self.input_hw,
            )
        
        elif self.config.prior_type == "geometric_mog":
            # Geometric MoG: Uses fixed centers (no learnable prior params)
            # But still needs component embeddings for component-aware decoder
            embed_dim = self.config.component_embedding_dim
            self.prior_module = MixturePriorParameters(
                name="prior",
                num_components=self.config.num_components,
                embed_dim=embed_dim,
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

        # Handle mixture-based priors (mixture, vamp, geometric_mog)
        if self.config.prior_type in ["mixture", "vamp", "geometric_mog"]:
            component_logits, z_mean, z_log, z = encoder_output
            if self.prior_module is None:
                raise ValueError(f"{self.config.prior_type} prior selected but prior_module was not initialized.")
            
            responsibilities = softmax(component_logits, axis=-1)
            batch_size = z.shape[0]
            num_components = self.config.num_components

            # Get prior-specific parameters
            if self.config.prior_type == "vamp":
                # VampPrior: Get pseudo-inputs
                pseudo_inputs = self.prior_module()
                extras["pseudo_inputs"] = pseudo_inputs

                # Encode pseudo-inputs once to expose q(z|u_k) statistics
                pseudo_encoder_output = self.encoder(pseudo_inputs, training=False)
                _, pseudo_z_mean, pseudo_z_log_var, _ = pseudo_encoder_output
                extras["pseudo_z_mean"] = pseudo_z_mean
                extras["pseudo_z_log_var"] = pseudo_z_log_var

                # VampPrior keeps π uniform today; expose it so diagnostics stay consistent.
                extras["pi"] = jnp.ones((num_components,)) / num_components

                # VampPrior doesn't use component embeddings, so decode directly from z.
                recon = self.decoder(z)
                extras["responsibilities"] = responsibilities

            else:
                # Mixture or Geometric MoG: Get embeddings and pi
                embeddings, pi_logits = self.prior_module()
                pi = softmax(pi_logits, axis=-1)
                
                # Stop gradients through π if not learnable
                if not self.config.learnable_pi:
                    pi = jax.lax.stop_gradient(pi)

                # Tile z and embeddings for all components
                z_tiled = jnp.broadcast_to(z[:, None, :], (batch_size, num_components, self.latent_dim))
                embed_tiled = jnp.broadcast_to(embeddings[None, :, :], (batch_size, num_components, embeddings.shape[-1]))

                # Check if decoder is component-aware
                is_component_aware = self.config.use_component_aware_decoder

                if is_component_aware:
                    # Component-aware decoder: pass z and component_embedding separately
                    z_flat = z_tiled.reshape((batch_size * num_components, -1))
                    embed_flat = embed_tiled.reshape((batch_size * num_components, -1))
                    decoder_output_flat = self.decoder(z_flat, embed_flat)
                else:
                    # Standard decoder: concatenate [z, e_c]
                    decoder_inputs = jnp.concatenate([z_tiled, embed_tiled], axis=-1)
                    decoder_inputs_flat = decoder_inputs.reshape((batch_size * num_components, -1))
                    decoder_output_flat = self.decoder(decoder_inputs_flat)

                # Handle heteroscedastic decoder (returns tuple)
                if isinstance(decoder_output_flat, tuple):
                    mean_flat, sigma_flat = decoder_output_flat
                    mean_per_component = mean_flat.reshape((batch_size, num_components, *self.output_hw))
                    sigma_per_component = sigma_flat.reshape((batch_size, num_components))

                    # Take expectation over components
                    expected_mean = jnp.sum(
                        responsibilities[..., None, None] * mean_per_component,
                        axis=1,
                    )
                    expected_sigma = jnp.sum(
                        responsibilities * sigma_per_component,
                        axis=1,
                    )
                    recon = (expected_mean, expected_sigma)
                    extras = {
                        "recon_per_component": (mean_per_component, sigma_per_component),
                        "responsibilities": responsibilities,
                        "pi_logits": pi_logits,
                        "pi": pi,
                        "component_embeddings": embeddings,
                    }
                else:
                    # Standard decoder
                    recon_per_component = decoder_output_flat.reshape(
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
            # Standard prior
            z_mean, z_log, z = encoder_output
            component_logits = None
            recon = self.decoder(z)

        logits = self.classifier(z, training=training)
        return ForwardOutput(component_logits, z_mean, z_log, z, recon, logits, extras)

