"""
Variational Mixture of Posteriors (VampPrior) prior.

VampPrior learns K pseudo-inputs {u_1, ..., u_K} and defines the prior as:
    p(z) = Σ_k π_k q(z|u_k)

where q(z|u_k) is the encoder posterior evaluated at pseudo-input u_k.

This creates spatial separation in latent space via learned prototypes,
providing an alternative to component-aware decoders for inducing component
specialization.

Key features:
- Learned pseudo-inputs (trainable parameters)
- Monte Carlo KL estimation: KL(q(z|x) || Σ_k π_k q(z|u_k))
- Optional prior shaping via MMD/MC-KL (not yet implemented)
- Does NOT use component embeddings (spatial separation is sufficient)

Reference:
    Tomczak & Welling (2018). VAE with a VampPrior.
    https://arxiv.org/abs/1705.07120
"""
from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp

from rcmvae.domain.priors.base import EncoderOutput
from rcmvae.application.services.loss_pipeline import (
    categorical_kl,
    usage_sparsity_penalty,
    weighted_reconstruction_loss_bce,
    weighted_reconstruction_loss_mse,
    weighted_heteroscedastic_reconstruction_loss,
)


class VampPrior:
    """Variational Mixture of Posteriors prior.

    Prior is defined as:
        p(c) = Cat(π), π_c = 1/K (uniform, can be made learnable)
        p(z|c) = q_φ(z|u_c), where u_c are learned pseudo-inputs
        p(z) = Σ_c π_c q_φ(z|u_c)

    The pseudo-inputs are trainable parameters stored in the network params
    under 'prior/pseudo_inputs'. They are initialized randomly from data or
    via k-means clustering.

    KL divergence is estimated via Monte Carlo:
        KL(q(z|x) || p(z)) = E_{z~q(z|x)}[log q(z|x) - log p(z)]
                            = E_{z~q(z|x)}[log q(z|x) - log Σ_k π_k q(z|u_k)]

    Example:
        >>> prior = VampPrior(
        ...     num_components=50,
        ...     latent_dim=16,
        ...     input_shape=(28, 28),
        ...     uniform_weights=True,
        ...     num_samples_kl=1,
        ... )
        >>>
        >>> # During the forward pass the network stores pseudo-input statistics
        >>> # in EncoderOutput.extras so compute_kl_terms can consume them.
        >>> kl_terms = prior.compute_kl_terms(encoder_output, config)
    """

    def __init__(
        self,
        num_components: int = 50,
        latent_dim: int = 16,
        input_shape: tuple = (28, 28),
        uniform_weights: bool = True,
        num_samples_kl: int = 1,
    ):
        """Initialize VampPrior.

        Args:
            num_components: Number of pseudo-inputs (K)
            latent_dim: Dimensionality of latent space
            input_shape: Shape of pseudo-inputs (e.g., (28, 28) for MNIST)
            uniform_weights: If True, use uniform π; else learnable (future)
            num_samples_kl: Number of Monte Carlo samples for KL estimation

        Note:
            Pseudo-inputs are NOT initialized here. They must be initialized
            via SSVAE.initialize_pseudo_inputs(data) after model creation.
        """
        self.num_components = num_components
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.uniform_weights = uniform_weights
        self.num_samples_kl = num_samples_kl
        self._legacy_encoder = None

        # Mixture weights (uniform only; learnable weights not yet implemented)
        if uniform_weights:
            self.pi = jnp.ones(num_components) / num_components
        else:
            raise NotImplementedError(
                "Learnable mixture weights for VampPrior deferred to Phase 3."
            )

    def _log_gaussian_prob(
        self,
        z: jnp.ndarray,
        mean: jnp.ndarray,
        log_var: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute log probability of z under Gaussian(mean, exp(log_var)).

        Args:
            z: Latent samples [batch, latent_dim] or [batch, num_samples, latent_dim]
            mean: Gaussian mean [batch, latent_dim]
            log_var: Gaussian log variance [batch, latent_dim]

        Returns:
            Log probability [batch,] or [batch, num_samples]
        """
        # Handle both [batch, d] and [batch, num_samples, d]
        if z.ndim == 3:
            # Expand mean and log_var: [batch, 1, d]
            mean = mean[:, None, :]
            log_var = log_var[:, None, :]

        # Compute log N(z; mean, var)
        var = jnp.exp(log_var)
        log_prob = -0.5 * jnp.sum(
            jnp.log(2 * jnp.pi * var) + jnp.square(z - mean) / var,
            axis=-1,  # Sum over latent dimensions
        )
        return log_prob

    def _compute_log_prior_prob(
        self,
        z: jnp.ndarray,
        pseudo_z_mean: jnp.ndarray,
        pseudo_z_log_var: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute log p(z) = log Σ_k π_k q(z|u_k) from cached pseudo-input stats.

        Args:
            z: Latent samples [batch, latent_dim] or [batch, num_samples, latent_dim]
            pseudo_z_mean: Encoder means for pseudo-inputs [K, latent_dim]
            pseudo_z_log_var: Encoder log variances for pseudo-inputs [K, latent_dim]

        Returns:
            Log probability [batch,] or [batch, num_samples]
        """
        u_means = pseudo_z_mean
        u_log_vars = pseudo_z_log_var

        if z.ndim == 2:
            z_expanded = z[:, None, :]
            u_means_expanded = u_means[None, :, :]
            u_log_vars_expanded = u_log_vars[None, :, :]

            var_k = jnp.exp(u_log_vars_expanded)
            log_q_z_uk = -0.5 * jnp.sum(
                jnp.log(2 * jnp.pi * var_k)
                + jnp.square(z_expanded - u_means_expanded) / var_k,
                axis=-1,
            )

        elif z.ndim == 3:
            z_expanded = z[:, :, None, :]
            u_means_expanded = u_means[None, None, :, :]
            u_log_vars_expanded = u_log_vars[None, None, :, :]

            var_k = jnp.exp(u_log_vars_expanded)
            log_q_z_uk = -0.5 * jnp.sum(
                jnp.log(2 * jnp.pi * var_k)
                + jnp.square(z_expanded - u_means_expanded) / var_k,
                axis=-1,
            )
        else:
            raise ValueError(f"Unexpected z shape: {z.shape}")

        log_pi = jnp.log(self.pi)

        if z.ndim == 2:
            log_p_z = jax.scipy.special.logsumexp(
                log_pi[None, :] + log_q_z_uk,
                axis=-1,
            )
        else:
            log_p_z = jax.scipy.special.logsumexp(
                log_pi[None, None, :] + log_q_z_uk,
                axis=-1,
            )

        return log_p_z

    # Backward compatibility API (legacy tests expect this hook)
    def set_encoder(self, encoder_fn, encoder_params) -> None:
        """Legacy no-op retained for backward compatibility."""
        self._legacy_encoder = (encoder_fn, encoder_params)

    def compute_kl_terms(
        self,
        encoder_output: EncoderOutput,
        config,
    ) -> Dict[str, jnp.ndarray]:
        """Compute KL divergence via Monte Carlo estimation.

        KL(q(z|x) || p(z)) = E_{z~q(z|x)}[log q(z|x) - log p(z)]

        where:
            log q(z|x) = log N(z; μ_enc, Σ_enc)
            log p(z) = log Σ_k π_k q(z|u_k)

        Args:
            encoder_output: Contains z_mean, z_log_var, z (sampled), and extras
                - extras must contain 'pseudo_inputs' [K, H, W]
                - extras may contain 'responsibilities' for diversity regularization
                - extras may contain 'rng_key' for MC sampling (if num_samples_kl > 1)
            config: Configuration with kl_weight, etc.

        Returns:
            Dictionary with keys:
                - kl_z: Monte Carlo estimate of KL divergence
                - kl_c: KL(q(c|x) || π) if responsibilities provided, else 0
                - component_diversity: Usage entropy regularization
                - component_entropy: H[q(c|x)] (diagnostic)
                - pi_entropy: H[π] (diagnostic)
        """
        if encoder_output.extras is None:
            raise ValueError(
                "VampPrior requires extras with pseudo_z_mean and pseudo_z_log_var."
            )

        pseudo_z_mean = encoder_output.extras.get("pseudo_z_mean")
        pseudo_z_log_var = encoder_output.extras.get("pseudo_z_log_var")

        if (pseudo_z_mean is None or pseudo_z_log_var is None) and self._legacy_encoder:
            encoder_fn, encoder_params = self._legacy_encoder
            pseudo_inputs = encoder_output.extras.get("pseudo_inputs")
            if pseudo_inputs is not None:
                stats = encoder_fn(encoder_params, pseudo_inputs, training=False)
                pseudo_z_mean = stats.z_mean
                pseudo_z_log_var = stats.z_log_var

        if pseudo_z_mean is None or pseudo_z_log_var is None:
            raise ValueError(
                "VampPrior requires pseudo_z_mean and pseudo_z_log_var in extras"
            )

        # Sample z from q(z|x) - use multiple samples if configured
        # encoder_output.z is already sampled once
        if self.num_samples_kl == 1:
            z_samples = encoder_output.z  # [batch, latent_dim]
        else:
            # Sample multiple times from q(z|x)
            # Try to get RNG key from extras, otherwise use deterministic key
            rng_key = encoder_output.extras.get("rng_key")
            if rng_key is None:
                # Fallback to deterministic (not ideal but maintains backward compatibility)
                import warnings
                warnings.warn(
                    "VampPrior: num_samples_kl > 1 but no rng_key in extras. "
                    "Using deterministic sampling. For proper stochastic sampling, "
                    "pass rng_key through extras.",
                    UserWarning
                )
                rng_key = jax.random.PRNGKey(0)
            
            eps = jax.random.normal(
                rng_key,
                shape=(
                    encoder_output.z_mean.shape[0],
                    self.num_samples_kl,
                    self.latent_dim,
                ),
            )
            # Reparameterization: z = μ + σ * ε
            std = jnp.exp(0.5 * encoder_output.z_log_var)
            z_samples = (
                encoder_output.z_mean[:, None, :] + std[:, None, :] * eps
            )  # [batch, num_samples, latent_dim]

        # Compute log q(z|x)
        log_q_z_x = self._log_gaussian_prob(
            z_samples,
            encoder_output.z_mean,
            encoder_output.z_log_var,
        )  # [batch,] or [batch, num_samples]

        # Compute log p(z) = log Σ_k π_k q(z|u_k) from cached statistics
        log_p_z = self._compute_log_prior_prob(
            z_samples,
            pseudo_z_mean,
            pseudo_z_log_var,
        )

        # KL divergence: E[log q(z|x) - log p(z)]
        kl_per_sample = log_q_z_x - log_p_z

        if self.num_samples_kl > 1:
            # Average over MC samples, then over batch
            kl_z = jnp.mean(kl_per_sample)
        else:
            # Average over batch
            kl_z = jnp.mean(kl_per_sample)

        kl_z = config.kl_weight * kl_z

        # Optional: KL(q(c|x) || π) if responsibilities provided
        # This is useful if encoder outputs component logits
        responsibilities = encoder_output.extras.get("responsibilities")
        if responsibilities is not None:
            kl_c = categorical_kl(
                responsibilities,
                self.pi,
                weight=config.kl_c_weight,
            )
            
            # Diversity regularization
            diversity_penalty = usage_sparsity_penalty(
                responsibilities,
                weight=config.component_diversity_weight,
            )

            # Diagnostics
            eps = 1e-8
            resp_safe = jnp.clip(responsibilities, eps, 1.0)
            component_entropy = -jnp.mean(
                jnp.sum(resp_safe * jnp.log(resp_safe), axis=-1)
            )
        else:
            # No component assignments (standard VampPrior)
            kl_c = jnp.array(0.0)
            diversity_penalty = jnp.array(0.0)
            component_entropy = jnp.array(0.0)

        # π entropy - compute from actual pi (even though it's uniform)
        eps = 1e-8
        pi_safe = jnp.clip(self.pi, eps, 1.0)
        pi_entropy = -jnp.sum(pi_safe * jnp.log(pi_safe))

        # Dirichlet penalty (always 0 for VampPrior since π is uniform and not learnable)
        dirichlet_penalty = jnp.array(0.0)

        return {
            "kl_z": kl_z,
            "kl_c": kl_c,
            "dirichlet_penalty": dirichlet_penalty,
            "component_diversity": diversity_penalty,
            "component_entropy": component_entropy,
            "pi_entropy": pi_entropy,
        }

    def compute_reconstruction_loss(
        self,
        x_true: jnp.ndarray,
        x_recon: jnp.ndarray | tuple,
        encoder_output: EncoderOutput,
        config,
    ) -> jnp.ndarray:
        """Compute reconstruction loss.

        VampPrior does NOT use component-aware decoder by design (spatial
        separation is sufficient). Therefore, expects standard reconstructions
        even when using a mixture encoder.

        Args:
            x_true: Ground truth images [batch, H, W]
            x_recon: Reconstructed images or (mean, sigma) tuple
                - Standard decoder: [batch, H, W]
                - Heteroscedastic decoder: ([batch, H, W], [batch,])
            encoder_output: Not used for VampPrior reconstruction
            config: Configuration

        Returns:
            Reconstruction loss

        """
        responsibilities = None
        num_components = None
        if encoder_output.extras is not None:
            responsibilities = encoder_output.extras.get("responsibilities")
            if responsibilities is not None:
                num_components = responsibilities.shape[-1]

        from rcmvae.application.services.loss_pipeline import (
            reconstruction_loss_mse,
            reconstruction_loss_bce,
            heteroscedastic_reconstruction_loss,
        )

        if responsibilities is not None:
            assert num_components is not None

            def _ensure_component_axis(
                array: jnp.ndarray,
                *,
                base_ndim: int | None,
            ) -> jnp.ndarray:
                """Ensure array has [batch, num_components, ...] layout."""
                if base_ndim is not None:
                    if array.ndim == base_ndim + 1 and array.shape[1] == num_components:
                        return array
                else:
                    if array.ndim >= 2 and array.shape[1] == num_components:
                        return array
                expanded = jnp.expand_dims(array, axis=1)
                target_shape = (array.shape[0], num_components, *array.shape[1:])
                return jnp.broadcast_to(expanded, target_shape)

            def _prepare_sigma(array: jnp.ndarray) -> jnp.ndarray:
                """σ outputs are either [batch] or [batch, num_components]."""
                if array.ndim == 2 and array.shape[1] == num_components:
                    return array
                if array.ndim == 1:
                    return jnp.broadcast_to(array[:, None], (array.shape[0], num_components))
                raise ValueError(
                    "Heteroscedastic decoder with VampPrior expects sigma shaped "
                    "[batch] or [batch, num_components]. "
                    f"Got {array.shape}."
                )

            if isinstance(x_recon, tuple):
                mean, sigma = x_recon
                mean_components = _ensure_component_axis(mean, base_ndim=x_true.ndim)
                sigma_components = _prepare_sigma(sigma)
                return weighted_heteroscedastic_reconstruction_loss(
                    x_true,
                    mean_components,
                    sigma_components,
                    responsibilities,
                    config.recon_weight,
                )
            if config.reconstruction_loss == "mse":
                recon_components = _ensure_component_axis(x_recon, base_ndim=x_true.ndim)
                return weighted_reconstruction_loss_mse(
                    x_true,
                    recon_components,
                    responsibilities,
                    config.recon_weight,
                )
            if config.reconstruction_loss == "bce":
                recon_components = _ensure_component_axis(x_recon, base_ndim=x_true.ndim)
                return weighted_reconstruction_loss_bce(
                    x_true,
                    recon_components,
                    responsibilities,
                    config.recon_weight,
                )
            raise ValueError(
                f"Unknown reconstruction_loss: {config.reconstruction_loss}"
            )

        if isinstance(x_recon, tuple):
            mean, sigma = x_recon
            return heteroscedastic_reconstruction_loss(
                x_true, mean, sigma, config.recon_weight
            )

        if config.reconstruction_loss == "mse":
            return reconstruction_loss_mse(x_true, x_recon, config.recon_weight)
        if config.reconstruction_loss == "bce":
            return reconstruction_loss_bce(x_true, x_recon, config.recon_weight)
        raise ValueError(
            f"Unknown reconstruction_loss: {config.reconstruction_loss}"
        )

    def get_prior_type(self) -> str:
        """Return prior type identifier."""
        return "vamp"

    def requires_component_embeddings(self) -> bool:
        """VampPrior does NOT use component embeddings.

        Spatial separation in latent space is achieved via learned pseudo-inputs,
        so decoder does not need component-specific pathways.
        """
        return False

    def sample(
        self,
        key: jax.Array,
        num_samples: int = 1,
    ) -> jnp.ndarray:
        """Sample from VampPrior p(z) = Σ_k π_k q(z|u_k).

        Strategy:
        1. Sample component k ~ Cat(π)
        2. Sample z ~ q(z|u_k)

        Args:
            key: JAX random key
            num_samples: Number of samples to generate

        Returns:
            Samples [num_samples, latent_dim]

        Note:
            Requires pseudo_inputs to be available. In practice, this should
            be called during generation with pseudo_inputs from model params.
        """
        # This is a placeholder - full implementation requires pseudo_inputs
        # In practice, this would be called with pseudo_inputs from params
        raise NotImplementedError(
            "VampPrior.sample() requires pseudo_inputs from model params. "
            "Use model-level sampling instead."
        )
