from __future__ import annotations
"""SSVAE models module (JAX)."""

from pathlib import Path

from typing import Dict, List, NamedTuple, Optional, Tuple

from rcmvae.utils import configure_jax_device, print_device_banner

configure_jax_device()

import numpy as np

from rcmvae.domain.config import SSVAEConfig
from rcmvae.domain.components.factory import build_classifier, build_decoder, build_encoder, get_architecture_dims
from rcmvae.application.callbacks import (
    CSVExporter,
    ConsoleLogger,
    LossCurvePlotter,
    TrainingCallback,
)
from rcmvae.application.runtime.state import SSVAETrainState

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    from jax.nn import one_hot, softmax
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


def apply_top_m_gating(
    weights: jnp.ndarray,
    top_m: jnp.ndarray | int,
    k_active: jnp.ndarray | int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply Top-M gating to component weights, keeping only top-M non-zero.

    This function selects the top-M components by weight within the active set,
    zeros out the rest, and renormalizes so the weights sum to 1.

    This implementation is fully JAX-traceable and works with traced values
    during JIT compilation.

    Args:
        weights: Component weights [batch, K], should already have inactive channels
                 zeroed via curriculum masking.
        top_m: Number of top components to keep (0 = no filtering, use all).
               Can be a Python int or JAX array (traced).
        k_active: Number of active channels (from curriculum).
                  Can be a Python int or JAX array (traced).

    Returns:
        Tuple of:
            - Filtered and renormalized weights [batch, K]
            - effective_m: Actual M used as a JAX array (min(top_m, k_active) or k_active if top_m=0)

    Behavior:
        - If top_m = 0: return weights unchanged (use all active channels)
        - If top_m >= k_active: return weights unchanged (use all active channels)
        - If top_m < k_active: keep only top_m highest weights, zero others, renormalize
    """
    K = weights.shape[-1]

    # Convert to JAX arrays for consistent tracing
    top_m_arr = jnp.asarray(top_m)
    k_active_arr = jnp.asarray(k_active)

    # effective_m = min(top_m, k_active), but top_m=0 means "use all active"
    effective_m = jnp.where(
        top_m_arr == 0,
        k_active_arr,
        jnp.minimum(top_m_arr, k_active_arr),
    )

    # Check if filtering is needed: top_m > 0 and effective_m < k_active
    needs_filtering = (top_m_arr > 0) & (effective_m < k_active_arr)

    # Sort indices by weight descending
    sorted_indices = jnp.argsort(-weights, axis=-1)  # [batch, K]

    # Create position-based mask: position p is kept if p < effective_m
    # This avoids dynamic slicing which doesn't work with traced shapes
    positions = jnp.arange(K)  # [K]
    position_mask = (positions < effective_m).astype(jnp.float32)  # [K]

    # Map position mask back to original index space using one-hot encoding
    # For each batch: one_hot(sorted_indices, K) gives [K, K] where row p is one-hot at original index
    # Weight by position_mask and sum to get mask in original index space
    one_hot_indices = jax.nn.one_hot(sorted_indices, K)  # [batch, K, K]
    mask = jnp.einsum("p,bpk->bk", position_mask, one_hot_indices)  # [batch, K]

    # Apply mask and renormalize
    filtered_weights = weights * mask
    weight_sum = jnp.sum(filtered_weights, axis=-1, keepdims=True)
    weight_sum = jnp.maximum(weight_sum, 1e-10)  # Avoid division by zero
    normalized_weights = filtered_weights / weight_sum

    # Use jnp.where to select between filtered and original based on needs_filtering
    result = jnp.where(needs_filtering, normalized_weights, weights)

    return result, effective_m


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
        gumbel_temperature: float | None = None,
        k_active: int | None = None,
        use_straight_through: bool | None = None,
        top_m_gating: int | None = None,
    ) -> ForwardOutput:
        """Forward pass returning latent statistics, reconstructions, and classifier logits.

        Args:
            x: Input images [batch, H, W]
            training: Whether in training mode
            gumbel_temperature: Optional temperature override for Gumbel-Softmax
            k_active: Number of active channels for curriculum (None = all channels active).
                      When specified, channels k >= k_active are masked out with -inf logits.
            use_straight_through: Override for straight-through Gumbel (None = use config value).
                                  Set to False during migration window for soft routing.
            top_m_gating: Number of top components to keep for reconstruction (None = use config).
                          0 = use all active channels, >0 = keep only top-M by weight.
        """
        encoder_output = self.encoder(x, training=training)
        extras: Dict[str, jnp.ndarray] = {}

        # Handle mixture-based priors (mixture, vamp, geometric_mog)
        if self.config.prior_type in ["mixture", "vamp", "geometric_mog"]:
            component_logits, z_mean_raw, z_log_raw, z_raw = encoder_output
            if self.prior_module is None:
                raise ValueError(f"{self.config.prior_type} prior selected but prior_module was not initialized.")

            # Apply curriculum active-set mask to component_logits BEFORE softmax/Gumbel
            # This ensures inactive channels get exactly zero probability
            component_logits_raw = component_logits  # Keep raw logits for diagnostics
            num_components = self.config.num_components

            # Curriculum masking: channels >= k_active get -inf logits
            # k_active is always an integer (caller ensures this); when k_active == num_components,
            # the mask is all-True (no-op). This keeps the code JIT-compatible.
            # Handle None at Python level (before tracing) by defaulting to num_components
            _k_active = num_components if k_active is None else k_active
            # Mask: set logits for inactive channels (k >= k_active) to -inf
            channel_indices = jnp.arange(num_components)  # [K]
            mask = channel_indices < _k_active  # [K] bool - JIT traces this with _k_active as int
            component_logits = jnp.where(mask, component_logits, -jnp.inf)

            responsibilities = softmax(component_logits, axis=-1)
            responsibilities = jnp.nan_to_num(responsibilities, nan=0.0, posinf=0.0, neginf=0.0)
            batch_size = z_raw.shape[0]
            latent_layout = getattr(self.config, "latent_layout", "shared")
            use_decentralized = latent_layout == "decentralized" and self.config.prior_type != "vamp"

            def _component_selection():
            # 1. Get component probabilities from encoder
            # q_c_logits: [B, K]
            
            # Use provided temperature override or fall back to config
                if gumbel_temperature is not None:
                    temp = gumbel_temperature
                else:
                    temp = self.config.gumbel_temperature

                if self.config.use_gumbel_softmax:
                    # Gumbel-Softmax sampling
                    # If hard=True, returns one-hot, but gradients flow through soft sample
                    # We'll use the straight-through estimator manually if needed, or just soft
                    
                    # Sample Gumbel noise only during training
                    if training and self.has_rng("gumbel"):
                        gumbel_key = self.make_rng("gumbel")
                        u = jax.random.uniform(gumbel_key, shape=component_logits.shape)
                        g = -jnp.log(-jnp.log(u + 1e-20) + 1e-20)
                    else:
                        # No noise during evaluation (or if no RNG provided)
                        g = 0.0
                    
                    # Softmax with temperature
                    # y = softmax((logits + g) / temp)
                    y_soft = nn.softmax((component_logits + g) / temp)

                    # Compute straight-through version unconditionally
                    # ST: forward is one-hot, backward is soft
                    index = jnp.argmax(y_soft, axis=-1)
                    y_hard = jax.nn.one_hot(index, self.config.num_components)
                    y_st = y_hard - jax.lax.stop_gradient(y_soft) + y_soft

                    # Determine whether to use straight-through
                    # Handle None at Python level (before tracing), use config default
                    if use_straight_through is None:
                        # Not specified: use config value (not traced)
                        if self.config.use_straight_through_gumbel:
                            return y_st, True
                        else:
                            return y_soft, False
                    else:
                        # Specified: use jax.lax.cond for traced boolean
                        # Convert to JAX array if needed for lax.cond
                        _use_st = jnp.asarray(use_straight_through)
                        y = jax.lax.cond(
                            _use_st,
                            lambda: y_st,
                            lambda: y_soft,
                        )
                        return y, _use_st
                else:
                # Standard categorical sampling (non-differentiable for gradients to encoder)
                # This path is likely not what we want for "decentralized" learning unless using REINFORCE
                # For now, just return softmax probabilities or hard sample without gradients?
                # Let's stick to Gumbel-Softmax as the primary path for this architecture.
                # Fallback: just return nn.softmax(component_logits) (soft assignment)
                    return nn.softmax(component_logits), False

            component_selection, selection_is_hard = _component_selection()

            # Apply Top-M gating for reconstruction weighting
            # Note: component_selection is kept unmodified for KL/responsibilities
            # recon_weights has Top-M filtering applied (if configured)
            _top_m = top_m_gating if top_m_gating is not None else self.config.top_m_gating
            recon_weights, effective_m = apply_top_m_gating(
                component_selection, _top_m, _k_active
            )
            extras["effective_m"] = jnp.array(effective_m)
            extras["top_m_gating"] = jnp.array(_top_m)

            if use_decentralized and z_raw.ndim == 3:
                z_mean_per_component = z_mean_raw
                z_log_per_component = z_log_raw
                z_per_component = z_raw
                z_mean = jnp.sum(component_selection[..., None] * z_mean_per_component, axis=1)
                z_log = jnp.sum(component_selection[..., None] * z_log_per_component, axis=1)
                z = jnp.sum(component_selection[..., None] * z_per_component, axis=1)
            else:
                z_mean_per_component = None
                z_log_per_component = None
                z_per_component = None
                z_mean = z_mean_raw
                z_log = z_log_raw
                z = z_raw

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

                # VampPrior: single latent z decoded (not per-component), so Top-M doesn't affect recon
                # but we still track the values for consistency
                recon = self.decoder(z)
                extras["responsibilities"] = responsibilities
                extras["component_selection"] = component_selection
                extras["recon_weights"] = recon_weights  # Top-M filtered weights
                extras["component_logits_raw"] = component_logits_raw
                extras["k_active"] = jnp.array(k_active if k_active is not None else num_components)
                extras["effective_m"] = jnp.array(effective_m)
                extras["top_m_gating"] = jnp.array(_top_m)
            else:
                # Mixture or Geometric MoG: Get embeddings and pi
                embeddings, pi_logits = self.prior_module()
                pi = softmax(pi_logits, axis=-1)

                # Stop gradients through π if not learnable
                if not self.config.learnable_pi:
                    pi = jax.lax.stop_gradient(pi)

                # Tile z and embeddings for all components
                if use_decentralized and z_per_component is not None and z_per_component.ndim == 3:
                    z_components = z_per_component
                else:
                    z_components = jnp.broadcast_to(z[:, None, :], (batch_size, num_components, self.latent_dim))
                embed_tiled = jnp.broadcast_to(embeddings[None, :, :], (batch_size, num_components, embeddings.shape[-1]))

                # Decoder handles conditioning internally (FiLM / concat / noop)
                z_flat = z_components.reshape((batch_size * num_components, -1))
                embed_flat = embed_tiled.reshape((batch_size * num_components, -1))
                decoder_output_flat = self.decoder(z_flat, embed_flat)

                # Handle heteroscedastic decoder (returns tuple)
                if isinstance(decoder_output_flat, tuple):
                    mean_flat, sigma_flat = decoder_output_flat
                    mean_per_component = mean_flat.reshape((batch_size, num_components, *self.output_hw))
                    sigma_per_component = sigma_flat.reshape((batch_size, num_components))

                    # Take expectation over components using Top-M filtered weights
                    expected_mean = jnp.sum(
                        recon_weights[..., None, None] * mean_per_component,
                        axis=1,
                    )
                    expected_sigma = jnp.sum(
                        recon_weights * sigma_per_component,
                        axis=1,
                    )
                    recon = (expected_mean, expected_sigma)
                    extras = {
                        "recon_per_component": (mean_per_component, sigma_per_component),
                        "responsibilities": responsibilities,
                        "component_selection": component_selection,  # Original selection (for KL)
                        "recon_weights": recon_weights,  # Top-M filtered weights (for reconstruction)
                        "selection_is_hard": jnp.array(selection_is_hard),
                        "pi_logits": pi_logits,
                        "pi": pi,
                        "component_embeddings": embeddings,
                        "component_logits_raw": component_logits_raw,  # Pre-mask logits for regularizer
                        "k_active": jnp.array(k_active if k_active is not None else num_components),
                        "effective_m": jnp.array(effective_m),  # Actual M used for Top-M gating
                        "top_m_gating": jnp.array(_top_m),  # Configured Top-M value
                    }
                else:
                    # Standard decoder
                    recon_per_component = decoder_output_flat.reshape(
                        (batch_size, num_components, *self.output_hw)
                    )
                    # Use Top-M filtered weights for reconstruction
                    expected_recon = jnp.sum(
                        recon_weights[..., None, None] * recon_per_component,
                        axis=1,
                    )
                    recon = expected_recon
                    extras = {
                        "recon_per_component": recon_per_component,
                        "responsibilities": responsibilities,
                        "component_selection": component_selection,  # Original selection (for KL)
                        "recon_weights": recon_weights,  # Top-M filtered weights (for reconstruction)
                        "selection_is_hard": jnp.array(selection_is_hard),
                        "pi_logits": pi_logits,
                        "pi": pi,
                        "component_embeddings": embeddings,
                        "component_logits_raw": component_logits_raw,  # Pre-mask logits for regularizer
                        "k_active": jnp.array(k_active if k_active is not None else num_components),
                        "effective_m": jnp.array(effective_m),  # Actual M used for Top-M gating
                        "top_m_gating": jnp.array(_top_m),  # Configured Top-M value
                    }

                if z_mean_per_component is not None:
                    extras["z_mean_per_component"] = z_mean_per_component
                if z_log_per_component is not None:
                    extras["z_log_var_per_component"] = z_log_per_component
                if z_per_component is not None:
                    extras["z_samples_per_component"] = z_per_component
        else:
            # Standard prior
            z_mean, z_log, z = encoder_output
            component_logits = None
            recon = self.decoder(z)

        logits = self.classifier(z, training=training)
        return ForwardOutput(component_logits, z_mean, z_log, z, recon, logits, extras)
