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
    """Protocol for implementing custom priors.

    All prior implementations must provide these methods to integrate
    with the SSVAE training pipeline. This protocol enables pluggable
    prior distributions without modifying core model code.

    Implementation Guide:
        To add a custom prior:

        1. Create new file in src/ssvae/priors/ implementing this protocol
        2. Register in priors/__init__.py: PRIOR_REGISTRY["my_prior"] = MyPrior
        3. Add config parameters to SSVAEConfig (if needed)
        4. Write tests in tests/test_my_prior.py

    Required Methods:
        - compute_kl_terms(): Return dict of KL divergences
        - compute_reconstruction_loss(): Compute reconstruction loss
        - get_prior_type(): Return string identifier
        - requires_component_embeddings(): Whether decoder needs embeddings

    Example Implementation:
        >>> class MyPrior:
        ...     def compute_kl_terms(self, encoder_output, config):
        ...         kl_z = kl_divergence(
        ...             encoder_output.z_mean,
        ...             encoder_output.z_log_var,
        ...             weight=config.kl_weight
        ...         )
        ...         return {"kl_z": kl_z}
        ...
        ...     def compute_reconstruction_loss(self, x_true, x_recon, encoder_output, config):
        ...         if config.reconstruction_loss == "mse":
        ...             return reconstruction_loss_mse(x_true, x_recon, config.recon_weight)
        ...         elif config.reconstruction_loss == "bce":
        ...             return reconstruction_loss_bce(x_true, x_recon, config.recon_weight)
        ...
        ...     def get_prior_type(self) -> str:
        ...         return "my_prior"
        ...
        ...     def requires_component_embeddings(self) -> bool:
        ...         return False

    Reference Implementations:
        - StandardGaussianPrior (src/ssvae/priors/standard.py):
          Simple N(0,I) prior - good starting point
        - MixtureGaussianPrior (src/ssvae/priors/mixture.py):
          Complete mixture prior with regularization - advanced example

    Testing:
        See tests/test_mixture_prior_regression.py for test patterns.

    See Also:
        - CONTRIBUTING.md > Adding a Prior
        - docs/theory/mathematical_specification.md for theory
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
