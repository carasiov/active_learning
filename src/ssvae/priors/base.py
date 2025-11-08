"""
Base protocol for prior modes.

Defines the interface that all prior implementations must follow.
"""
from __future__ import annotations

from typing import Dict, NamedTuple, Protocol

import jax.numpy as jnp


class EncoderOutput(NamedTuple):
    """Standardized encoder output for any prior type.

    Attributes:
        z_mean: Mean of latent distribution [batch, latent_dim]
        z_log_var: Log variance of latent distribution [batch, latent_dim]
        z: Sampled latent code [batch, latent_dim]
        component_logits: Component logits for mixture priors [batch, num_components] (None for standard)
        extras: Additional outputs (responsibilities, π, embeddings, etc.)
    """

    z_mean: jnp.ndarray
    z_log_var: jnp.ndarray
    z: jnp.ndarray
    component_logits: jnp.ndarray | None = None
    extras: Dict[str, jnp.ndarray] | None = None


class PriorMode(Protocol):
    """Protocol defining the interface for all prior modes.

    All prior implementations must provide these methods to be compatible
    with the SSVAE training pipeline.
    """

    def compute_kl_terms(
        self,
        encoder_output: EncoderOutput,
        config,
    ) -> Dict[str, jnp.ndarray]:
        """Compute all KL divergence terms for this prior.

        Args:
            encoder_output: Output from encoder
            config: Model configuration

        Returns:
            Dictionary of KL terms, e.g.:
                - Standard prior: {"kl_z": scalar}
                - Mixture prior: {"kl_z": scalar, "kl_c": scalar}

        Example:
            >>> kl_terms = prior.compute_kl_terms(encoder_output, config)
            >>> total_kl = sum(kl_terms.values())
        """
        ...

    def compute_reconstruction_loss(
        self,
        x_true: jnp.ndarray,
        x_recon: jnp.ndarray,
        encoder_output: EncoderOutput,
        config,
    ) -> jnp.ndarray:
        """Compute reconstruction loss for this prior.

        Args:
            x_true: Ground truth images [batch, H, W]
            x_recon: Reconstructed images (or per-component reconstructions)
            encoder_output: Encoder output (may contain responsibilities for weighting)
            config: Model configuration

        Returns:
            Scalar reconstruction loss

        Note:
            - Standard prior: simple MSE/BCE between x_true and x_recon
            - Mixture prior: weighted reconstruction over components
        """
        ...

    def get_prior_type(self) -> str:
        """Return the prior type identifier.

        Returns:
            String identifier ("standard", "mixture", etc.)
        """
        ...

    def requires_component_embeddings(self) -> bool:
        """Whether this prior needs component embeddings in the decoder.

        Returns:
            True if decoder should receive component embeddings
        """
        ...
