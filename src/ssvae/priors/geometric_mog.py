"""
Fixed Geometric Mixture of Gaussians prior.

This prior models the latent space as a mixture with FIXED centers arranged
geometrically (circle or grid). This is primarily a diagnostic/curriculum tool.

WARNING: Induces artificial topology on latent space. Use cautiously.

Key features:
- Fixed geometric arrangement of component centers (circle)
- Analytical KL divergence: E_{q(c|x)} [KL(q(z|x,c) || N(μ_c, I))]
- Uniform mixture weights π (fixed, not learned)
- Compatible with component-aware decoder for functional specialization

Example:
    Circle arrangement in 2D latent space with K=10 components:
    μ_1 = [r*cos(0), r*sin(0)]
    μ_2 = [r*cos(2π/10), r*sin(2π/10)]
    ...
    
    For higher-dimensional latent spaces, remaining dimensions are zero-padded.
"""
from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from ssvae.priors.base import EncoderOutput
from training.losses import (
    categorical_kl,
    dirichlet_map_penalty,
    usage_sparsity_penalty,
    weighted_reconstruction_loss_bce,
    weighted_reconstruction_loss_mse,
    weighted_heteroscedastic_reconstruction_loss,
)


class GeometricMixtureOfGaussiansPrior:
    """Fixed geometric mixture prior for diagnostic and curriculum purposes.

    The prior is:
        p(c) = Cat(π), π_c = 1/K (uniform)
        p(z|c) = N(μ_c, I), μ_c arranged geometrically
        p(z) = Σ_c π_c N(μ_c, I)

    Supports:
    - Circle arrangement: Centers on circle of radius r in first 2 dimensions
    - Grid arrangement (future): Centers on 2D grid (requires K = perfect square)

    WARNING: This induces artificial topology on latent space. Use for:
    - Quick visualization of component separation
    - Curriculum learning (start with geometric, anneal to learned)
    - Debugging component-aware decoder
    DO NOT use for production models - prefer VampPrior or standard mixture.

    Example:
        >>> prior = GeometricMixtureOfGaussiansPrior(
        ...     num_components=10,
        ...     latent_dim=16,
        ...     arrangement="circle",
        ...     radius=2.0
        ... )
        >>> kl_terms = prior.compute_kl_terms(encoder_output, config)
    """

    def __init__(
        self,
        num_components: int = 10,
        latent_dim: int = 2,
        arrangement: str = "circle",
        radius: float = 2.0,
    ):
        """Initialize geometric mixture prior.

        Args:
            num_components: Number of mixture components (K)
            latent_dim: Dimensionality of latent space
            arrangement: Geometric arrangement ("circle" or "grid")
            radius: Radius for circle arrangement (ignored for grid)

        Raises:
            ValueError: If arrangement is invalid or grid requested with non-square K
        """
        self.num_components = num_components
        self.latent_dim = latent_dim
        self.arrangement = arrangement
        self.radius = radius

        # Validate arrangement
        if arrangement not in ["circle", "grid"]:
            raise ValueError(
                f"arrangement must be 'circle' or 'grid', got '{arrangement}'"
            )

        if arrangement == "grid":
            # Check if K is perfect square
            sqrt_k = int(jnp.sqrt(num_components))
            if sqrt_k * sqrt_k != num_components:
                raise ValueError(
                    f"Grid arrangement requires num_components to be perfect square, "
                    f"got {num_components}"
                )

        # Generate fixed centers
        self.centers = self._generate_centers()

        # Uniform mixture weights (default, can be overridden by params if learnable)
        self._default_pi = jnp.ones(num_components) / num_components

    def _generate_centers(self) -> jnp.ndarray:
        """Generate fixed geometric centers.

        Returns:
            Centers array of shape [K, latent_dim]
        """
        if self.arrangement == "circle":
            return self._generate_circle_centers()
        elif self.arrangement == "grid":
            return self._generate_grid_centers()
        else:
            raise ValueError(f"Unknown arrangement: {self.arrangement}")

    def _generate_circle_centers(self) -> jnp.ndarray:
        """Generate centers arranged on a circle in first 2 dimensions.

        Returns:
            Centers [K, latent_dim] where first 2 dims form circle, rest are zeros
        """
        angles = jnp.linspace(0, 2 * jnp.pi, self.num_components, endpoint=False)
        
        # Create centers in 2D
        x = self.radius * jnp.cos(angles)
        y = self.radius * jnp.sin(angles)
        
        # Stack and pad with zeros for higher dimensions
        if self.latent_dim == 2:
            centers = jnp.stack([x, y], axis=1)  # [K, 2]
        else:
            # Pad with zeros for remaining dimensions
            xy = jnp.stack([x, y], axis=1)  # [K, 2]
            zeros = jnp.zeros((self.num_components, self.latent_dim - 2))
            centers = jnp.concatenate([xy, zeros], axis=1)  # [K, latent_dim]
        
        return centers

    def _generate_grid_centers(self) -> jnp.ndarray:
        """Generate centers arranged on a 2D grid.

        Returns:
            Centers [K, latent_dim] arranged on grid in first 2 dims
        """
        sqrt_k = int(jnp.sqrt(self.num_components))
        
        # Create grid in [-radius, radius] x [-radius, radius]
        x = jnp.linspace(-self.radius, self.radius, sqrt_k)
        y = jnp.linspace(-self.radius, self.radius, sqrt_k)
        xx, yy = jnp.meshgrid(x, y)
        
        # Flatten to [K, 2]
        xy = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
        
        # Pad with zeros for higher dimensions
        if self.latent_dim == 2:
            centers = xy
        else:
            zeros = jnp.zeros((self.num_components, self.latent_dim - 2))
            centers = jnp.concatenate([xy, zeros], axis=1)
        
        return centers

    def compute_kl_terms(
        self,
        encoder_output: EncoderOutput,
        config,
    ) -> Dict[str, jnp.ndarray]:
        """Compute KL divergence and regularization terms.

        KL divergence for geometric mixture:
            KL = E_{q(c|x)} [KL(q(z|x,c) || N(μ_c, I))]

        where KL(N(μ_enc, Σ_enc) || N(μ_c, I)) has closed form:
            KL = 0.5 * (tr(Σ_enc) + ||μ_enc - μ_c||² - d - log|Σ_enc|)
            
        For diagonal Σ_enc = diag(σ²):
            KL = 0.5 * Σ_i (σ²_i + (μ_enc_i - μ_c_i)² - 1 - log σ²_i)

        Args:
            encoder_output: Contains z_mean, z_log_var, and extras with responsibilities
            config: Configuration with weights for each term

        Returns:
            Dictionary with keys:
                - kl_z: Weighted KL divergence to geometric centers
                - kl_c: KL(q(c|x) || π) (always zero since π uniform and not learned)
                - dirichlet_penalty: Dirichlet MAP penalty on π (zero unless enabled)
                - component_diversity: Usage entropy regularization
                - component_entropy: H[q(c|x)] (diagnostic)
                - pi_entropy: H[π] (diagnostic, always log(K) for uniform)
        """
        if encoder_output.extras is None:
            raise ValueError(
                "Geometric MoG prior requires extras with responsibilities"
            )

        responsibilities = encoder_output.extras.get("responsibilities")
        if responsibilities is None:
            raise ValueError(
                "Geometric MoG prior requires responsibilities in extras"
            )

        # Use π from extras if available (learnable π), else default uniform
        pi = encoder_output.extras.get("pi", self._default_pi)

        # Compute KL(q(z|x,c) || N(μ_c, I)) for each component
        # Shape: z_mean, z_log_var are [batch, latent_dim]
        # centers is [K, latent_dim]
        # Need to compute KL for each batch-component pair
        
        batch_size = encoder_output.z_mean.shape[0]
        
        # Expand for broadcasting: [batch, 1, latent_dim]
        z_mean_expanded = encoder_output.z_mean[:, None, :]
        z_log_var_expanded = encoder_output.z_log_var[:, None, :]
        
        # Expand centers: [1, K, latent_dim]
        centers_expanded = self.centers[None, :, :]
        
        # Compute KL for each (batch, component) pair
        # KL = 0.5 * Σ_d (σ²_d + (μ_d - μ_c_d)² - 1 - log σ²_d)
        var = jnp.exp(z_log_var_expanded)  # [batch, 1, latent_dim]
        mean_diff_sq = jnp.square(z_mean_expanded - centers_expanded)  # [batch, K, latent_dim]
        
        kl_per_component = 0.5 * jnp.sum(
            var + mean_diff_sq - 1.0 - z_log_var_expanded,
            axis=-1  # Sum over latent dimensions
        )  # [batch, K]
        
        # Weight by responsibilities and average over batch
        kl_z = jnp.mean(jnp.sum(responsibilities * kl_per_component, axis=-1))
        kl_z = config.kl_weight * kl_z

        # KL(q(c|x) || π): Since π is uniform by default, this is just negative entropy
        # But when π is learnable, this provides gradients to update mixture weights
        kl_c = categorical_kl(
            responsibilities,
            pi,
            weight=config.kl_c_weight,
        )

        dirichlet_penalty = dirichlet_map_penalty(
            pi,
            alpha=config.dirichlet_alpha,
            weight=config.dirichlet_weight,
        )

        # Diversity regularization (same as mixture prior)
        diversity_penalty = usage_sparsity_penalty(
            responsibilities,
            weight=config.component_diversity_weight,
        )

        # Diagnostic metrics
        eps = 1e-8
        resp_safe = jnp.clip(responsibilities, eps, 1.0)
        component_entropy = -jnp.mean(jnp.sum(resp_safe * jnp.log(resp_safe), axis=-1))

        # π entropy (diagnostic)
        pi_safe = jnp.clip(pi, eps, 1.0)
        pi_entropy = -jnp.sum(pi_safe * jnp.log(pi_safe))

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
        """Compute weighted reconstruction loss over components.

        Same as mixture prior - expects per-component reconstructions weighted
        by responsibilities.

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
            raise ValueError(
                "Geometric MoG prior requires extras with responsibilities"
            )

        responsibilities = encoder_output.extras.get("responsibilities")
        if responsibilities is None:
            raise ValueError(
                "Geometric MoG prior requires responsibilities in extras"
            )

        # Check if heteroscedastic (tuple output)
        if isinstance(x_recon, tuple):
            mean_components, sigma_components = x_recon
            return weighted_heteroscedastic_reconstruction_loss(
                x_true,
                mean_components,
                sigma_components,
                responsibilities,
                config.recon_weight,
            )

        # Standard reconstruction
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
        return "geometric_mog"

    def requires_component_embeddings(self) -> bool:
        """Geometric MoG is compatible with component-aware decoder.
        
        Even though centers are fixed geometrically, the decoder can still
        learn functional specialization per component. This is useful for
        diagnostic purposes.
        """
        return True

    def get_centers(self) -> jnp.ndarray:
        """Get fixed geometric centers.
        
        Returns:
            Centers array of shape [K, latent_dim]
            
        Useful for visualization and debugging.
        """
        return self.centers
