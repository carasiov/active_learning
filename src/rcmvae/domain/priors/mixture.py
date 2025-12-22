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

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from rcmvae.domain.priors.base import EncoderOutput
from rcmvae.application.services.loss_pipeline import (
    categorical_kl,
    dirichlet_map_penalty,
    kl_divergence,
    usage_sparsity_penalty,
    weighted_reconstruction_loss_bce,
    weighted_reconstruction_loss_mse,
    weighted_heteroscedastic_reconstruction_loss,
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
        dict_keys(['kl_z', 'kl_c', 'dirichlet_penalty', 'component_diversity'])
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
                - component_diversity: (optional) entropy-based diversity regularization
                - component_entropy: (diagnostic) H[q(c|x)]
                - pi_entropy: (diagnostic) H[π]
        """
        if encoder_output.extras is None:
            raise ValueError("Mixture prior requires extras with responsibilities and pi")

        responsibilities = encoder_output.extras.get("responsibilities")
        pi = encoder_output.extras.get("pi")
        z_mean_per_component = encoder_output.extras.get("z_mean_per_component")
        z_log_per_component = encoder_output.extras.get("z_log_var_per_component")

        if responsibilities is None or pi is None:
            raise ValueError("Mixture prior requires responsibilities and pi in extras")

        # KL divergence terms
        if z_mean_per_component is not None and z_log_per_component is not None:
            kl_z = kl_divergence(
                z_mean_per_component,
                z_log_per_component,
                weight=config.kl_weight,
            )
        else:
            kl_z = kl_divergence(
                encoder_output.z_mean,
                encoder_output.z_log_var,
                weight=config.kl_weight,
            )

        # Per-sample c prior selection
        c_regularizer = getattr(config, "c_regularizer", "categorical")

        # Standard categorical KL
        if c_regularizer in {"categorical", "both"}:
            kl_c = categorical_kl(
                responsibilities,
                pi,
                weight=config.kl_c_weight,
            )
        else:
            kl_c = jnp.array(0.0, dtype=responsibilities.dtype)

        # Logistic-normal mixture regularizer on logits
        # IMPORTANT: Uses raw (finite) logits, NOT masked routing_logits
        # Under curriculum, restricts mixture sum to active components only
        if c_regularizer in {"logit_mog", "both"}:
            # Prefer raw_logits (curriculum-safe, always finite)
            # Fall back to component_logits for backward compatibility
            raw_logits = encoder_output.extras.get("raw_logits") if encoder_output.extras else None
            if raw_logits is None:
                raw_logits = encoder_output.component_logits

            if raw_logits is None:
                kl_c_logit_mog = jnp.array(0.0, dtype=kl_z.dtype)
            else:
                k_max = raw_logits.shape[-1]

                # Get active_mask from extras if available (curriculum)
                active_mask = encoder_output.extras.get("active_mask") if encoder_output.extras else None

                if active_mask is not None:
                    # Curriculum mode: only sum over active components
                    active_mask = jnp.asarray(active_mask, dtype=jnp.bool_)
                    k_active = jnp.sum(active_mask.astype(jnp.float32))
                    # Create active indices for computing log probabilities
                    # We'll compute all K log probs but mask out inactive ones
                else:
                    # No curriculum: all components active
                    active_mask = jnp.ones(k_max, dtype=jnp.bool_)
                    k_active = jnp.array(k_max, dtype=jnp.float32)

                # Gate logit-MoG when k_active <= 1 to prevent early lock-in
                # When only one channel is active, logit-MoG would just pull toward that channel
                # which bakes in "channel 0 forever" before other channels have a chance
                def _compute_logit_mog():
                    # Means at +M*e_k for each component k
                    means = jnp.eye(k_max) * config.c_logit_prior_mean  # [K_max, K_max]
                    sigma_sq = config.c_logit_prior_sigma ** 2

                    # Log prob under each Gaussian component
                    # raw_logits: [B, K_max], means: [K_max, K_max]
                    # centered: [B, K_max, K_max] (for each sample, each component, distance in each dim)
                    centered = raw_logits[:, None, :] - means[None, :, :]  # [B, K_max, K_max]
                    quad = jnp.sum(jnp.square(centered) / sigma_sq, axis=-1)  # [B, K_max]
                    log_norm = -0.5 * (k_max * jnp.log(2 * jnp.pi * sigma_sq))
                    log_prob = log_norm - 0.5 * quad  # [B, K_max]

                    # Mask inactive components by setting their log_prob to -inf
                    # This ensures they contribute 0 to the mixture sum
                    log_prob_masked = jnp.where(active_mask[None, :], log_prob, -jnp.inf)

                    # Mixture probability: logsumexp over active components with uniform mixture weight 1/|A_t|
                    # log p_mix(y) = logsumexp_k∈A [log p_k(y)] + log(1/|A_t|)
                    #              = logsumexp_k∈A [log p_k(y)] - log(|A_t|)
                    log_mix = logsumexp(log_prob_masked, axis=1) - jnp.log(k_active)  # [B]

                    # Handle edge case where k_active could be 0 (though shouldn't happen)
                    log_mix = jnp.where(jnp.isfinite(log_mix), log_mix, jnp.array(-100.0))

                    return -jnp.mean(log_mix)

                def _skip_logit_mog():
                    return jnp.array(0.0, dtype=kl_z.dtype)

                # Only compute logit-MoG when more than 1 channel is active
                # Use jax.lax.cond for traced-safe conditional (k_active is a JAX array)
                nll = jax.lax.cond(
                    k_active > 1.0,
                    _compute_logit_mog,
                    _skip_logit_mog,
                )
                kl_c_logit_mog = config.c_logit_prior_weight * nll
        else:
            kl_c_logit_mog = jnp.array(0.0, dtype=kl_z.dtype)

        # Optional regularizers
        dirichlet_penalty = dirichlet_map_penalty(
            pi,
            alpha=config.dirichlet_alpha,
            weight=config.dirichlet_weight,
        )

        diversity_penalty = usage_sparsity_penalty(
            responsibilities,
            weight=config.component_diversity_weight,
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
            "component_diversity": diversity_penalty,
            "component_entropy": component_entropy,
            "pi_entropy": pi_entropy,
            "kl_c_logit_mog": kl_c_logit_mog,
        }

    def compute_reconstruction_loss(
        self,
        x_true: jnp.ndarray,
        x_recon: jnp.ndarray | tuple,
        encoder_output: EncoderOutput,
        config,
    ) -> jnp.ndarray:
        """Compute weighted reconstruction loss over components (standard or heteroscedastic).

        The reconstruction is computed as:
            L_recon = E_{q(c|x)} [L(x, recon_c)]

        For heteroscedastic decoders:
            L_recon = E_{q(c|x)} [ ||x - mean_c||²/(2σ_c²) + log σ_c ]

        Args:
            x_true: Ground truth images [batch, H, W]
            x_recon: Per-component reconstructions or (mean, sigma) tuple
                - Standard decoder: [batch, K, H, W]
                - Heteroscedastic decoder: ([batch, K, H, W], [batch, K])
            encoder_output: Contains extras with responsibilities
            config: Configuration

        Returns:
            Weighted reconstruction loss
        """
        if encoder_output.extras is None:
            raise ValueError("Mixture prior requires extras with responsibilities")

        responsibilities = encoder_output.extras.get("responsibilities")
        component_weights = encoder_output.extras.get("component_selection", responsibilities)
        if responsibilities is None:
            raise ValueError("Mixture prior requires responsibilities in extras")

        # Check if heteroscedastic (tuple output)
        if isinstance(x_recon, tuple):
            mean_components, sigma_components = x_recon
            return weighted_heteroscedastic_reconstruction_loss(
                x_true,
                mean_components,
                sigma_components,
                component_weights,
                config.recon_weight,
            )

        # Standard reconstruction (backward compatible)
        # x_recon should be per-component: [batch, K, H, W]
        if config.reconstruction_loss == "mse":
            return weighted_reconstruction_loss_mse(
                x_true,
                x_recon,
                component_weights,
                config.recon_weight,
            )
        elif config.reconstruction_loss == "bce":
            return weighted_reconstruction_loss_bce(
                x_true,
                x_recon,
                component_weights,
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
