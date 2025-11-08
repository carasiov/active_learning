"""
Standard Gaussian prior: p(z) = N(0, I)

This is the classic VAE prior with a simple diagonal Gaussian.
"""
from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from ssvae.priors.base import EncoderOutput
from training.losses import kl_divergence, reconstruction_loss_bce, reconstruction_loss_mse


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
        dict_keys(['kl_z'])
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
            Dictionary with single key 'kl_z'
        """
        kl_z = kl_divergence(
            encoder_output.z_mean,
            encoder_output.z_log_var,
            weight=config.kl_weight,
        )
        return {"kl_z": kl_z}

    def compute_reconstruction_loss(
        self,
        x_true: jnp.ndarray,
        x_recon: jnp.ndarray,
        encoder_output: EncoderOutput,
        config,
    ) -> jnp.ndarray:
        """Compute simple reconstruction loss.

        Args:
            x_true: Ground truth images
            x_recon: Reconstructed images
            encoder_output: Not used for standard prior
            config: Configuration with reconstruction_loss type and weight

        Returns:
            Weighted reconstruction loss (MSE or BCE)
        """
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
