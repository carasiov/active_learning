"""Modular configuration system for SSVAE experiments.

This package provides a decomposed, composable configuration system that replaces
the monolithic SSVAEConfig. Each module focuses on a single concern:

- base: Core network, training, loss, and decoder feature configs
- priors: Typed prior configurations (Standard, Mixture, Vamp, GeometricMoG)
- experiment: Top-level ExperimentConfig orchestrating all components

Usage:
    from rcmvae.config import (
        ExperimentConfig,
        NetworkConfig,
        TrainingConfig,
        LossConfig,
        DecoderFeatures,
        StandardPriorConfig,
        MixturePriorConfig,
    )

    config = ExperimentConfig(
        network=NetworkConfig(latent_dim=2, encoder_type="conv"),
        training=TrainingConfig(batch_size=128, max_epochs=50),
        loss=LossConfig(recon_weight=500.0),
        prior=MixturePriorConfig(num_components=10),
    )
"""

from .base import (
    DecoderFeatures,
    LossConfig,
    NetworkConfig,
    TrainingConfig,
)
from .converters import (
    experiment_config_from_dict,
    experiment_config_to_ssvae_config,
    ssvae_config_to_experiment_config,
)
from .experiment import ExperimentConfig
from .priors import (
    GeometricMoGPriorConfig,
    MixturePriorConfig,
    PriorConfig,
    StandardPriorConfig,
    VampPriorConfig,
)

__all__ = [
    # Base configs
    "NetworkConfig",
    "TrainingConfig",
    "LossConfig",
    "DecoderFeatures",
    # Prior configs
    "PriorConfig",
    "StandardPriorConfig",
    "MixturePriorConfig",
    "VampPriorConfig",
    "GeometricMoGPriorConfig",
    # Top-level config
    "ExperimentConfig",
    # Converters
    "experiment_config_from_dict",
    "experiment_config_to_ssvae_config",
    "ssvae_config_to_experiment_config",
]
