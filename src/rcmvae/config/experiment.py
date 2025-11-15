"""Top-level experiment configuration orchestrating all components.

ExperimentConfig composes network, training, loss, decoder features, and prior
configurations into a single cohesive experiment specification.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from .base import DecoderFeatures, LossConfig, NetworkConfig, TrainingConfig
from .priors import (
    GeometricMoGPriorConfig,
    MixturePriorConfig,
    PriorConfig,
    StandardPriorConfig,
    VampPriorConfig,
)


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    Composes all sub-configurations (network, training, loss, decoder, prior)
    into a single experiment specification. This replaces SSVAEConfig with a
    modular, composable design.

    Attributes:
        network: Network architecture configuration.
        training: Training loop hyperparameters.
        loss: Loss function weights and settings.
        decoder: Decoder feature flags and parameters.
        prior: Prior distribution configuration (typed subclass of PriorConfig).
    """

    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    decoder: DecoderFeatures = field(default_factory=DecoderFeatures)
    prior: PriorConfig = field(default_factory=StandardPriorConfig)

    def __post_init__(self):
        """Validate cross-cutting concerns and inter-config constraints."""
        # Set component_embedding_dim default based on latent_dim if not specified
        if self.decoder.component_embedding_dim is None:
            self.decoder.component_embedding_dim = self.network.latent_dim

        # Validate top_m_gating constraint for mixture priors
        if isinstance(self.prior, (MixturePriorConfig, VampPriorConfig, GeometricMoGPriorConfig)):
            if self.decoder.top_m_gating > self.prior.num_components:
                raise ValueError(
                    f"top_m_gating ({self.decoder.top_m_gating}) cannot exceed "
                    f"num_components ({self.prior.num_components})"
                )

        # Warn if component-aware decoder used with non-mixture prior
        mixture_based_priors = (MixturePriorConfig, VampPriorConfig, GeometricMoGPriorConfig)
        if self.decoder.use_component_aware_decoder and not isinstance(self.prior, mixture_based_priors):
            warnings.warn(
                f"use_component_aware_decoder: true only applies to mixture-based priors. "
                f"Got prior_type: '{self.prior.get_prior_type()}'. Falling back to standard decoder.",
                UserWarning,
            )

        # τ-classifier validation
        if isinstance(self.prior, (MixturePriorConfig, VampPriorConfig, GeometricMoGPriorConfig)):
            if hasattr(self.prior, "use_tau_classifier") and self.prior.use_tau_classifier:
                if isinstance(self.prior, (MixturePriorConfig, VampPriorConfig)):
                    if self.prior.num_components < self.network.num_classes:
                        raise ValueError(
                            "num_components must be >= num_classes when use_tau_classifier=True. "
                            f"Got num_components={self.prior.num_components}, num_classes={self.network.num_classes}."
                        )
                elif isinstance(self.prior, GeometricMoGPriorConfig):
                    if self.prior.num_components < self.network.num_classes:
                        warnings.warn(
                            "Geometric MoG τ-classifier typically benefits from num_components >= num_classes. "
                            f"Got num_components={self.prior.num_components}, num_classes={self.network.num_classes}.",
                            RuntimeWarning,
                        )
        elif hasattr(self.prior, "use_tau_classifier") and self.prior.use_tau_classifier:
            warnings.warn(
                f"use_tau_classifier: true only applies to mixture-based priors. "
                f"Got prior_type: '{self.prior.get_prior_type()}'.",
                RuntimeWarning,
            )

    def is_mixture_based_prior(self) -> bool:
        """Check if prior type uses mixture encoder (outputs component logits).

        Returns:
            True if prior is mixture, vamp, or geometric_mog
        """
        return isinstance(
            self.prior,
            (MixturePriorConfig, VampPriorConfig, GeometricMoGPriorConfig),
        )

    def get_prior_type(self) -> str:
        """Get the prior type string identifier.

        Returns:
            Prior type string ("standard", "mixture", "vamp", "geometric_mog")
        """
        return self.prior.get_prior_type()
