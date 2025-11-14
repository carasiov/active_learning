"""
Prior modes for SSVAE.

This module provides a pluggable interface for different prior distributions.
Each prior mode defines how to:
- Encode inputs to latent space
- Decode latent codes to reconstructions
- Compute KL divergence terms

Available priors:
- StandardGaussianPrior: Simple N(0,I) prior
- MixtureGaussianPrior: Mixture of Gaussians with learnable π and component embeddings
"""
from __future__ import annotations

from model.ssvae.priors.base import EncoderOutput, PriorMode
from model.ssvae.priors.standard import StandardGaussianPrior
from model.ssvae.priors.mixture import MixtureGaussianPrior
from model.ssvae.priors.vamp import VampPrior
from model.ssvae.priors.geometric_mog import GeometricMixtureOfGaussiansPrior

# Prior registry for factory pattern
PRIOR_REGISTRY = {
    "standard": StandardGaussianPrior,
    "mixture": MixtureGaussianPrior,
    "vamp": VampPrior,
    "geometric_mog": GeometricMixtureOfGaussiansPrior,
}


def get_prior(prior_type: str, **kwargs) -> PriorMode:
    """Factory function to create prior instances.

    Args:
        prior_type: Type of prior ("standard", "mixture", "vamp", or "geometric_mog")
        **kwargs: Additional arguments for prior initialization
            For VampPrior: num_components, latent_dim, input_shape, uniform_weights, num_samples_kl
            For GeometricMoG: num_components, latent_dim, arrangement, radius

    Returns:
        Prior instance

    Example:
        >>> prior = get_prior("standard")
        >>> prior = get_prior("mixture", num_components=10, latent_dim=16)
        >>> prior = get_prior("vamp", num_components=50, latent_dim=16, input_shape=(28, 28))
        >>> prior = get_prior("geometric_mog", num_components=10, latent_dim=2, radius=2.0)
    """
    if prior_type not in PRIOR_REGISTRY:
        available = ", ".join(PRIOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown prior type '{prior_type}'. Available: {available}"
        )

    prior_class = PRIOR_REGISTRY[prior_type]
    return prior_class(**kwargs)


__all__ = [
    "PriorMode",
    "EncoderOutput",
    "StandardGaussianPrior",
    "MixtureGaussianPrior",
    "VampPrior",
    "GeometricMixtureOfGaussiansPrior",
    "get_prior",
    "PRIOR_REGISTRY",
]
