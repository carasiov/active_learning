"""
Mixture of Gaussians prior.

This prior models the latent space as a mixture:
    p(c) = Cat(π)
    p(z|c) = N(0, I)  (same for all components)
    p(z) = Σ_c π_c N(0, I)

The decoder is component-aware: p(x|z,c) uses component embeddings.
"""
from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from ssvae.priors.base import EncoderOutput
from training.losses import (
    categorical_kl,
    dirichlet_map_penalty,
    kl_divergence,
    usage_sparsity_penalty,
    weighted_reconstruction_loss_bce,
    weighted_reconstruction_loss_mse,
)


class MixtureGaussianPrior:
    """Mixture of Gaussians prior with component-aware decoder.

    Key features:
    - Encoder outputs component logits q(c|x) in addition to z
    - KL divergence has two terms: KL_z and KL_c
    - Reconstruction is weighted expectation over components
    - Supports optional regularizers (Dirichlet, usage sparsity)

    Example:
        >>> prior = MixtureGaussianPrior()
        >>> kl_terms = prior.compute_kl_terms(encoder_output, config)
        >>> kl_terms.keys()
        dict_keys(['kl_z', 'kl_c', 'dirichlet_penalty', 'usage_sparsity'])
    """

    def compute_kl_terms(
        self,
        encoder_output: EncoderOutput,
        config,
    ) -> Dict[str, jnp.ndarray]:
        """Compute all KL and regularization terms for mixture prior.

        Args:
            encoder_output: Contains z_mean, z_log_var, and extras with:
                - responsibilities: q(c|x)
                - pi: learned mixture weights
            config: Configuration with weights for each term

        Returns:
            Dictionary with keys:
                - kl_z: KL(q(z|x,c) || N(0,I))
                - kl_c: KL(q(c|x) || π)
                - dirichlet_penalty: (optional) Dirichlet MAP on π
                - usage_sparsity: (optional) entropy penalty on component usage
                - component_entropy: (diagnostic) H[q(c|x)]
                - pi_entropy: (diagnostic) H[π]
        """
        if encoder_output.extras is None:
            raise ValueError("Mixture prior requires extras with responsibilities and pi")

        responsibilities = encoder_output.extras.get("responsibilities")
        pi = encoder_output.extras.get("pi")

        if responsibilities is None or pi is None:
            raise ValueError("Mixture prior requires responsibilities and pi in extras")

        # KL divergence terms
        kl_z = kl_divergence(
            encoder_output.z_mean,
            encoder_output.z_log_var,
            weight=config.kl_weight,
        )

        kl_c = categorical_kl(
            responsibilities,
            pi,
            weight=config.kl_c_weight,
        )

        # Optional regularizers
        dirichlet_penalty = dirichlet_map_penalty(
            pi,
            alpha=config.dirichlet_alpha,
            weight=config.dirichlet_weight,
        )

        usage_penalty = usage_sparsity_penalty(
            responsibilities,
            weight=config.usage_sparsity_weight,
        )

        # Diagnostic metrics (entropy calculations)
        eps = 1e-8
        resp_safe = jnp.clip(responsibilities, eps, 1.0)
        component_entropy = -jnp.mean(jnp.sum(resp_safe * jnp.log(resp_safe), axis=-1))

        pi_safe = jnp.clip(pi, eps, 1.0)
        pi_entropy = -jnp.sum(pi_safe * jnp.log(pi_safe))

        return {
            "kl_z": kl_z,
            "kl_c": kl_c,
            "dirichlet_penalty": dirichlet_penalty,
            "usage_sparsity": usage_penalty,
            "component_entropy": component_entropy,
            "pi_entropy": pi_entropy,
        }

    def compute_reconstruction_loss(
        self,
        x_true: jnp.ndarray,
        x_recon: jnp.ndarray,
        encoder_output: EncoderOutput,
        config,
    ) -> jnp.ndarray:
        """Compute weighted reconstruction loss over components.

        The reconstruction is computed as:
            L_recon = E_{q(c|x)} [L(x, recon_c)]

        Args:
            x_true: Ground truth images [batch, H, W]
            x_recon: Per-component reconstructions [batch, num_components, H, W]
            encoder_output: Contains extras with responsibilities
            config: Configuration

        Returns:
            Weighted reconstruction loss
        """
        if encoder_output.extras is None:
            raise ValueError("Mixture prior requires extras with responsibilities")

        responsibilities = encoder_output.extras.get("responsibilities")
        if responsibilities is None:
            raise ValueError("Mixture prior requires responsibilities in extras")

        # x_recon should be per-component: [batch, K, H, W]
        # We compute the weighted expectation over components

        if config.reconstruction_loss == "mse":
            return weighted_reconstruction_loss_mse(
                x_true,
                x_recon,
                responsibilities,
                config.recon_weight,
            )
        elif config.reconstruction_loss == "bce":
            return weighted_reconstruction_loss_bce(
                x_true,
                x_recon,
                responsibilities,
                config.recon_weight,
            )
        else:
            raise ValueError(
                f"Unknown reconstruction_loss: {config.reconstruction_loss}"
            )

    def get_prior_type(self) -> str:
        """Return prior type identifier."""
        return "mixture"

    def requires_component_embeddings(self) -> bool:
        """Mixture prior uses component embeddings in decoder."""
        return True
