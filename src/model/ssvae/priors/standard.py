"""
Standard Gaussian prior: p(z) = N(0, I)

This is the classic VAE prior with a simple diagonal Gaussian.
"""
from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from model.ssvae.priors.base import EncoderOutput
from model.training.losses import (
    kl_divergence,
    reconstruction_loss_bce,
    reconstruction_loss_mse,
    heteroscedastic_reconstruction_loss,
)


class StandardGaussianPrior:
    """Standard N(0, I) prior for VAE.

    This is the simplest prior mode:
    - Encoder outputs (z_mean, z_log_var, z)
    - KL divergence to N(0, I)
    - Simple reconstruction loss (MSE or BCE)

    Example:
        >>> prior = StandardGaussianPrior()
        >>> kl_terms = prior.compute_kl_terms(encoder_output, config)
        >>> kl_terms.keys()
        dict_keys(['kl_z', 'dirichlet_penalty'])
    """

    def compute_kl_terms(
        self,
        encoder_output: EncoderOutput,
        config,
    ) -> Dict[str, jnp.ndarray]:
        """Compute KL(q(z|x) || N(0,I)).

        Args:
            encoder_output: Contains z_mean and z_log_var
            config: Configuration with kl_weight

        Returns:
            Dictionary with keys:
                - kl_z: KL(q(z|x) || N(0, I))
                - dirichlet_penalty: Always zero for standard prior
        """
        kl_z = kl_divergence(
            encoder_output.z_mean,
            encoder_output.z_log_var,
            weight=config.kl_weight,
        )
        return {
            "kl_z": kl_z,
            "dirichlet_penalty": jnp.zeros_like(kl_z),
        }

    def compute_reconstruction_loss(
        self,
        x_true: jnp.ndarray,
        x_recon: jnp.ndarray | tuple,
        encoder_output: EncoderOutput,
        config,
    ) -> jnp.ndarray:
        """Compute reconstruction loss (standard or heteroscedastic).

        Args:
            x_true: Ground truth images [batch, H, W]
            x_recon: Reconstructed images or (mean, sigma) tuple
                - Standard decoder: [batch, H, W]
                - Heteroscedastic decoder: ([batch, H, W], [batch,])
            encoder_output: Not used for standard prior
            config: Configuration with reconstruction_loss type and weight

        Returns:
            Weighted reconstruction loss (MSE, BCE, or heteroscedastic NLL)
        """
        # Check if heteroscedastic (tuple output)
        if isinstance(x_recon, tuple):
            mean, sigma = x_recon
            return heteroscedastic_reconstruction_loss(
                x_true, mean, sigma, config.recon_weight
            )

        # Standard reconstruction (backward compatible)
        if config.reconstruction_loss == "mse":
            return reconstruction_loss_mse(x_true, x_recon, config.recon_weight)
        elif config.reconstruction_loss == "bce":
            return reconstruction_loss_bce(x_true, x_recon, config.recon_weight)
        else:
            raise ValueError(
                f"Unknown reconstruction_loss: {config.reconstruction_loss}"
            )

    def get_prior_type(self) -> str:
        """Return prior type identifier."""
        return "standard"

    def requires_component_embeddings(self) -> bool:
        """Standard prior does not use component embeddings."""
        return False
