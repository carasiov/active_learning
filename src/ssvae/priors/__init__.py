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

from ssvae.priors.base import EncoderOutput, PriorMode
from ssvae.priors.standard import StandardGaussianPrior
from ssvae.priors.mixture import MixtureGaussianPrior

# Prior registry for factory pattern
PRIOR_REGISTRY = {
    "standard": StandardGaussianPrior,
    "mixture": MixtureGaussianPrior,
}


def get_prior(prior_type: str, **kwargs) -> PriorMode:
    """Factory function to create prior instances.

    Args:
        prior_type: Type of prior ("standard" or "mixture")
        **kwargs: Additional arguments for prior initialization

    Returns:
        Prior instance

    Example:
        >>> prior = get_prior("standard")
        >>> prior = get_prior("mixture", num_components=10, latent_dim=16)
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
    "get_prior",
    "PRIOR_REGISTRY",
]
