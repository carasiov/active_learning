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
        effective_logit_mog_weight: float | None = None,
    ) -> Dict[str, jnp.ndarray]:
        """Compute all KL and regularization terms for mixture prior.

        Args:
            encoder_output: Contains z_mean, z_log_var, and extras with:
                - responsibilities: q(c|x)
                - pi: learned mixture weights
            config: Configuration with weights for each term
            effective_logit_mog_weight: Optional override for logit-mog weight.
                                        Used during migration window to reduce peakiness pressure.

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
        # In curriculum mode, only active channels contribute to the mixture
        if c_regularizer in {"logit_mog", "both"}:
            # Use raw (pre-mask) logits for the regularizer
            component_logits_raw = encoder_output.extras.get("component_logits_raw")
            if component_logits_raw is None:
                component_logits_raw = encoder_output.component_logits
            if component_logits_raw is None:
                kl_c_logit_mog = jnp.array(0.0, dtype=kl_z.dtype)
            else:
                k_total = component_logits_raw.shape[-1]
                # Get k_active from extras (curriculum); defaults to all if not present
                # Keep as JAX array for JIT compatibility (avoid Python int conversion)
                k_active_arr = encoder_output.extras.get("k_active")
                k_active = k_active_arr if k_active_arr is not None else jnp.array(k_total)

                # Build means: [K_total, K_total] with μ_k = M * e_k
                means = jnp.eye(k_total) * config.c_logit_prior_mean
                sigma_sq = config.c_logit_prior_sigma ** 2

                # Log prob under each Gaussian component: N(y; μ_k, σ²I)
                # centered: [B, K_total, K_total]
                centered = component_logits_raw[:, None, :] - means[None, :, :]
                quad = jnp.sum(jnp.square(centered) / sigma_sq, axis=-1)  # [B, K_total]
                log_norm = -0.5 * (k_total * jnp.log(2 * jnp.pi * sigma_sq))
                log_prob = log_norm - 0.5 * quad  # [B, K_total]

                # Mask: only active channels (k < k_active) contribute to mixture
                active_mask = jnp.arange(k_total) < k_active  # [K_total] bool
                # Set log_prob for inactive channels to -inf so they don't contribute
                log_prob_masked = jnp.where(active_mask, log_prob, -jnp.inf)

                # Uniform mixture over active channels: p_mix = (1/k_active) * Σ_{k∈A} N(y; μ_k)
                # log p_mix = logsumexp(log_prob_masked) - log(k_active)
                log_mix = logsumexp(log_prob_masked, axis=1) - jnp.log(k_active)
                nll = -jnp.mean(log_mix)
                # Use effective weight if provided (migration window), else config value
                _logit_mog_weight = effective_logit_mog_weight if effective_logit_mog_weight is not None else config.c_logit_prior_weight
                kl_c_logit_mog = _logit_mog_weight * nll
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
