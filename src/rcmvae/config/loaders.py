"""YAML and dict loaders for ExperimentConfig.

Loads ExperimentConfig from dictionaries (typically from YAML files).
"""

from __future__ import annotations

from typing import Any, Dict

from .base import DecoderFeatures, LossConfig, NetworkConfig, TrainingConfig
from .experiment import ExperimentConfig
from .priors import (
    GeometricMoGPriorConfig,
    MixturePriorConfig,
    PriorConfig,
    StandardPriorConfig,
    VampPriorConfig,
)


def experiment_config_from_dict(config_dict: Dict[str, Any]) -> ExperimentConfig:
    """Load ExperimentConfig from a dictionary (e.g., from YAML).

    Args:
        config_dict: Dictionary containing configuration parameters.
            Expected structure:
                {
                    "network": {...},
                    "training": {...},
                    "loss": {...},
                    "decoder": {...},
                    "prior": {"type": "...", ...},
                }
            Or flat structure (legacy) with all params at top level.

    Returns:
        Populated ExperimentConfig instance.

    Raises:
        ValueError: If prior type is unknown.
    """
    # Check if dict has modular structure or flat structure
    has_modular_structure = any(
        key in config_dict for key in ["network", "training", "loss", "decoder", "prior"]
    )

    if has_modular_structure:
        return _load_from_modular_dict(config_dict)
    else:
        return _load_from_flat_dict(config_dict)


def _load_from_modular_dict(config_dict: Dict[str, Any]) -> ExperimentConfig:
    """Load from modular dictionary structure."""
    network_dict = config_dict.get("network", {})
    training_dict = config_dict.get("training", {})
    loss_dict = config_dict.get("loss", {})
    decoder_dict = config_dict.get("decoder", {})
    prior_dict = config_dict.get("prior", {})

    # Convert tuples from lists (YAML doesn't have tuples)
    if "hidden_dims" in network_dict and isinstance(network_dict["hidden_dims"], list):
        network_dict["hidden_dims"] = tuple(network_dict["hidden_dims"])
    if "input_hw" in network_dict and isinstance(network_dict["input_hw"], list):
        network_dict["input_hw"] = tuple(network_dict["input_hw"])

    network = NetworkConfig(**network_dict)
    training = TrainingConfig(**training_dict)
    loss = LossConfig(**loss_dict)
    decoder = DecoderFeatures(**decoder_dict)
    prior = _load_prior_from_dict(prior_dict)

    return ExperimentConfig(
        network=network,
        training=training,
        loss=loss,
        decoder=decoder,
        prior=prior,
    )


def _load_from_flat_dict(config_dict: Dict[str, Any]) -> ExperimentConfig:
    """Load from flat dictionary structure (legacy SSVAEConfig-like)."""
    # Extract prior type to determine which prior config to use
    prior_type = config_dict.get("prior_type", "standard")

    # Convert tuples from lists
    config_dict = dict(config_dict)  # Make a copy
    if "hidden_dims" in config_dict and isinstance(config_dict["hidden_dims"], list):
        config_dict["hidden_dims"] = tuple(config_dict["hidden_dims"])
    if "input_hw" in config_dict and isinstance(config_dict["input_hw"], list):
        config_dict["input_hw"] = tuple(config_dict["input_hw"])

    # Build network config
    network = NetworkConfig(
        num_classes=config_dict.get("num_classes", 10),
        latent_dim=config_dict.get("latent_dim", 2),
        hidden_dims=config_dict.get("hidden_dims", (256, 128, 64)),
        encoder_type=config_dict.get("encoder_type", "dense"),
        decoder_type=config_dict.get("decoder_type", "dense"),
        classifier_type=config_dict.get("classifier_type", "dense"),
        input_hw=config_dict.get("input_hw"),
        dropout_rate=config_dict.get("dropout_rate", 0.2),
    )

    # Build training config
    training = TrainingConfig(
        learning_rate=config_dict.get("learning_rate", 1e-3),
        batch_size=config_dict.get("batch_size", 128),
        max_epochs=config_dict.get("max_epochs", 300),
        patience=config_dict.get("patience", 50),
        val_split=config_dict.get("val_split", 0.1),
        random_seed=config_dict.get("random_seed", 42),
        grad_clip_norm=config_dict.get("grad_clip_norm", 1.0),
        weight_decay=config_dict.get("weight_decay", 1e-4),
        monitor_metric=config_dict.get("monitor_metric", "classification_loss"),
    )

    # Build loss config
    loss = LossConfig(
        reconstruction_loss=config_dict.get("reconstruction_loss", "mse"),
        recon_weight=config_dict.get("recon_weight", 500.0),
        kl_weight=config_dict.get("kl_weight", 5.0),
        label_weight=config_dict.get("label_weight", 0.0),
        use_contrastive=config_dict.get("use_contrastive", False),
        contrastive_weight=config_dict.get("contrastive_weight", 0.0),
    )

    # Build decoder features
    decoder = DecoderFeatures(
        use_heteroscedastic_decoder=config_dict.get("use_heteroscedastic_decoder", False),
        sigma_min=config_dict.get("sigma_min", 0.05),
        sigma_max=config_dict.get("sigma_max", 0.5),
        use_component_aware_decoder=config_dict.get("use_component_aware_decoder", True),
        component_embedding_dim=config_dict.get("component_embedding_dim"),
        top_m_gating=config_dict.get("top_m_gating", 0),
        soft_embedding_warmup_epochs=config_dict.get("soft_embedding_warmup_epochs", 0),
    )

    # Build prior config based on type
    prior = _load_prior_from_flat_dict(prior_type, config_dict)

    return ExperimentConfig(
        network=network,
        training=training,
        loss=loss,
        decoder=decoder,
        prior=prior,
    )


def _load_prior_from_dict(prior_dict: Dict[str, Any]) -> PriorConfig:
    """Load prior config from modular dictionary."""
    prior_type = prior_dict.get("type", "standard")
    return _load_prior_from_flat_dict(prior_type, prior_dict)


def _load_prior_from_flat_dict(prior_type: str, config_dict: Dict[str, Any]) -> PriorConfig:
    """Load prior config from flat dictionary with given prior_type."""
    if prior_type == "standard":
        return StandardPriorConfig()

    elif prior_type == "mixture":
        return MixturePriorConfig(
            num_components=config_dict.get("num_components", 10),
            kl_c_weight=config_dict.get("kl_c_weight", 1.0),
            kl_c_anneal_epochs=config_dict.get("kl_c_anneal_epochs", 0),
            learnable_pi=config_dict.get("learnable_pi", False),
            dirichlet_alpha=config_dict.get("dirichlet_alpha"),
            dirichlet_weight=config_dict.get("dirichlet_weight", 1.0),
            component_diversity_weight=config_dict.get("component_diversity_weight", 0.0),
            mixture_history_log_every=config_dict.get("mixture_history_log_every", 1),
            use_tau_classifier=config_dict.get("use_tau_classifier", False),
            tau_smoothing_alpha=config_dict.get("tau_smoothing_alpha", 1.0),
        )

    elif prior_type == "vamp":
        return VampPriorConfig(
            num_components=config_dict.get("num_components", 10),
            kl_c_weight=config_dict.get("kl_c_weight", 1.0),
            kl_c_anneal_epochs=config_dict.get("kl_c_anneal_epochs", 0),
            learnable_pi=config_dict.get("learnable_pi", False),
            dirichlet_alpha=config_dict.get("dirichlet_alpha"),
            dirichlet_weight=config_dict.get("dirichlet_weight", 1.0),
            component_diversity_weight=config_dict.get("component_diversity_weight", 0.0),
            mixture_history_log_every=config_dict.get("mixture_history_log_every", 1),
            use_tau_classifier=config_dict.get("use_tau_classifier", False),
            tau_smoothing_alpha=config_dict.get("tau_smoothing_alpha", 1.0),
            vamp_num_samples_kl=config_dict.get("vamp_num_samples_kl", 1),
            vamp_pseudo_lr_scale=config_dict.get("vamp_pseudo_lr_scale", 0.1),
            vamp_pseudo_init_method=config_dict.get("vamp_pseudo_init_method", "random"),
        )

    elif prior_type == "geometric_mog":
        return GeometricMoGPriorConfig(
            num_components=config_dict.get("num_components", 10),
            kl_c_weight=config_dict.get("kl_c_weight", 1.0),
            kl_c_anneal_epochs=config_dict.get("kl_c_anneal_epochs", 0),
            learnable_pi=config_dict.get("learnable_pi", False),
            dirichlet_alpha=config_dict.get("dirichlet_alpha"),
            dirichlet_weight=config_dict.get("dirichlet_weight", 1.0),
            component_diversity_weight=config_dict.get("component_diversity_weight", 0.0),
            mixture_history_log_every=config_dict.get("mixture_history_log_every", 1),
            use_tau_classifier=config_dict.get("use_tau_classifier", False),
            tau_smoothing_alpha=config_dict.get("tau_smoothing_alpha", 1.0),
            geometric_arrangement=config_dict.get("geometric_arrangement", "circle"),
            geometric_radius=config_dict.get("geometric_radius", 2.0),
        )

    else:
        raise ValueError(
            f"Unknown prior_type: '{prior_type}'. "
            f"Valid options: standard, mixture, vamp, geometric_mog"
        )
