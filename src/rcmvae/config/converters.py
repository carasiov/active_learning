"""Converters between ExperimentConfig and legacy SSVAEConfig.

These utilities provide bidirectional conversion for gradual migration:
- YAML dict → ExperimentConfig (new path)
- ExperimentConfig → SSVAEConfig (backward compat)
- SSVAEConfig → ExperimentConfig (migration helper)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from rcmvae.domain.config import SSVAEConfig

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


def experiment_config_to_ssvae_config(exp_config: ExperimentConfig) -> SSVAEConfig:
    """Convert ExperimentConfig to legacy SSVAEConfig.

    This enables backward compatibility during gradual migration.

    Args:
        exp_config: The new modular experiment configuration.

    Returns:
        Equivalent SSVAEConfig instance.
    """
    # Build the flat dict with all parameters
    config_dict = {
        # Network
        "num_classes": exp_config.network.num_classes,
        "latent_dim": exp_config.network.latent_dim,
        "hidden_dims": exp_config.network.hidden_dims,
        "encoder_type": exp_config.network.encoder_type,
        "decoder_type": exp_config.network.decoder_type,
        "classifier_type": exp_config.network.classifier_type,
        "input_hw": exp_config.network.input_hw,
        "dropout_rate": exp_config.network.dropout_rate,
        # Training
        "learning_rate": exp_config.training.learning_rate,
        "batch_size": exp_config.training.batch_size,
        "max_epochs": exp_config.training.max_epochs,
        "patience": exp_config.training.patience,
        "val_split": exp_config.training.val_split,
        "random_seed": exp_config.training.random_seed,
        "grad_clip_norm": exp_config.training.grad_clip_norm,
        "weight_decay": exp_config.training.weight_decay,
        "monitor_metric": exp_config.training.monitor_metric,
        # Loss
        "reconstruction_loss": exp_config.loss.reconstruction_loss,
        "recon_weight": exp_config.loss.recon_weight,
        "kl_weight": exp_config.loss.kl_weight,
        "label_weight": exp_config.loss.label_weight,
        "use_contrastive": exp_config.loss.use_contrastive,
        "contrastive_weight": exp_config.loss.contrastive_weight,
        # Decoder features
        "use_heteroscedastic_decoder": exp_config.decoder.use_heteroscedastic_decoder,
        "sigma_min": exp_config.decoder.sigma_min,
        "sigma_max": exp_config.decoder.sigma_max,
        "use_component_aware_decoder": exp_config.decoder.use_component_aware_decoder,
        "component_embedding_dim": exp_config.decoder.component_embedding_dim,
        "top_m_gating": exp_config.decoder.top_m_gating,
        "soft_embedding_warmup_epochs": exp_config.decoder.soft_embedding_warmup_epochs,
        # Prior type
        "prior_type": exp_config.prior.get_prior_type(),
    }

    # Add prior-specific parameters
    prior = exp_config.prior
    if isinstance(prior, (MixturePriorConfig, VampPriorConfig, GeometricMoGPriorConfig)):
        config_dict.update({
            "num_components": prior.num_components,
            "kl_c_weight": prior.kl_c_weight,
            "kl_c_anneal_epochs": prior.kl_c_anneal_epochs,
            "learnable_pi": prior.learnable_pi,
            "dirichlet_alpha": prior.dirichlet_alpha,
            "dirichlet_weight": prior.dirichlet_weight,
            "component_diversity_weight": prior.component_diversity_weight,
            "mixture_history_log_every": prior.mixture_history_log_every,
            "use_tau_classifier": prior.use_tau_classifier,
            "tau_smoothing_alpha": prior.tau_smoothing_alpha,
        })

    if isinstance(prior, VampPriorConfig):
        config_dict.update({
            "vamp_num_samples_kl": prior.vamp_num_samples_kl,
            "vamp_pseudo_lr_scale": prior.vamp_pseudo_lr_scale,
            "vamp_pseudo_init_method": prior.vamp_pseudo_init_method,
        })

    if isinstance(prior, GeometricMoGPriorConfig):
        config_dict.update({
            "geometric_arrangement": prior.geometric_arrangement,
            "geometric_radius": prior.geometric_radius,
        })

    return SSVAEConfig(**config_dict)


def ssvae_config_to_experiment_config(ssvae_config: SSVAEConfig) -> ExperimentConfig:
    """Convert legacy SSVAEConfig to ExperimentConfig.

    This enables migration of existing code to the new config system.

    Args:
        ssvae_config: Legacy monolithic configuration.

    Returns:
        Equivalent ExperimentConfig instance.
    """
    # Build network config
    network = NetworkConfig(
        num_classes=ssvae_config.num_classes,
        latent_dim=ssvae_config.latent_dim,
        hidden_dims=ssvae_config.hidden_dims,
        encoder_type=ssvae_config.encoder_type,
        decoder_type=ssvae_config.decoder_type,
        classifier_type=ssvae_config.classifier_type,
        input_hw=ssvae_config.input_hw,
        dropout_rate=ssvae_config.dropout_rate,
    )

    # Build training config
    training = TrainingConfig(
        learning_rate=ssvae_config.learning_rate,
        batch_size=ssvae_config.batch_size,
        max_epochs=ssvae_config.max_epochs,
        patience=ssvae_config.patience,
        val_split=ssvae_config.val_split,
        random_seed=ssvae_config.random_seed,
        grad_clip_norm=ssvae_config.grad_clip_norm,
        weight_decay=ssvae_config.weight_decay,
        monitor_metric=ssvae_config.monitor_metric,
    )

    # Build loss config
    loss = LossConfig(
        reconstruction_loss=ssvae_config.reconstruction_loss,
        recon_weight=ssvae_config.recon_weight,
        kl_weight=ssvae_config.kl_weight,
        label_weight=ssvae_config.label_weight,
        use_contrastive=ssvae_config.use_contrastive,
        contrastive_weight=ssvae_config.contrastive_weight,
    )

    # Build decoder features
    decoder = DecoderFeatures(
        use_heteroscedastic_decoder=ssvae_config.use_heteroscedastic_decoder,
        sigma_min=ssvae_config.sigma_min,
        sigma_max=ssvae_config.sigma_max,
        use_component_aware_decoder=ssvae_config.use_component_aware_decoder,
        component_embedding_dim=ssvae_config.component_embedding_dim,
        top_m_gating=ssvae_config.top_m_gating,
        soft_embedding_warmup_epochs=ssvae_config.soft_embedding_warmup_epochs,
    )

    # Build prior config based on type
    prior_type = ssvae_config.prior_type

    if prior_type == "standard":
        prior: PriorConfig = StandardPriorConfig()

    elif prior_type == "mixture":
        prior = MixturePriorConfig(
            num_components=ssvae_config.num_components,
            kl_c_weight=ssvae_config.kl_c_weight,
            kl_c_anneal_epochs=ssvae_config.kl_c_anneal_epochs,
            learnable_pi=ssvae_config.learnable_pi,
            dirichlet_alpha=ssvae_config.dirichlet_alpha,
            dirichlet_weight=ssvae_config.dirichlet_weight,
            component_diversity_weight=ssvae_config.component_diversity_weight,
            mixture_history_log_every=ssvae_config.mixture_history_log_every,
            use_tau_classifier=ssvae_config.use_tau_classifier,
            tau_smoothing_alpha=ssvae_config.tau_smoothing_alpha,
        )

    elif prior_type == "vamp":
        prior = VampPriorConfig(
            num_components=ssvae_config.num_components,
            kl_c_weight=ssvae_config.kl_c_weight,
            kl_c_anneal_epochs=ssvae_config.kl_c_anneal_epochs,
            learnable_pi=ssvae_config.learnable_pi,
            dirichlet_alpha=ssvae_config.dirichlet_alpha,
            dirichlet_weight=ssvae_config.dirichlet_weight,
            component_diversity_weight=ssvae_config.component_diversity_weight,
            mixture_history_log_every=ssvae_config.mixture_history_log_every,
            use_tau_classifier=ssvae_config.use_tau_classifier,
            tau_smoothing_alpha=ssvae_config.tau_smoothing_alpha,
            vamp_num_samples_kl=ssvae_config.vamp_num_samples_kl,
            vamp_pseudo_lr_scale=ssvae_config.vamp_pseudo_lr_scale,
            vamp_pseudo_init_method=ssvae_config.vamp_pseudo_init_method,
        )

    elif prior_type == "geometric_mog":
        prior = GeometricMoGPriorConfig(
            num_components=ssvae_config.num_components,
            kl_c_weight=ssvae_config.kl_c_weight,
            kl_c_anneal_epochs=ssvae_config.kl_c_anneal_epochs,
            learnable_pi=ssvae_config.learnable_pi,
            dirichlet_alpha=ssvae_config.dirichlet_alpha,
            dirichlet_weight=ssvae_config.dirichlet_weight,
            component_diversity_weight=ssvae_config.component_diversity_weight,
            mixture_history_log_every=ssvae_config.mixture_history_log_every,
            use_tau_classifier=ssvae_config.use_tau_classifier,
            tau_smoothing_alpha=ssvae_config.tau_smoothing_alpha,
            geometric_arrangement=ssvae_config.geometric_arrangement,
            geometric_radius=ssvae_config.geometric_radius,
        )

    else:
        raise ValueError(f"Unknown prior_type: {prior_type}")

    return ExperimentConfig(
        network=network,
        training=training,
        loss=loss,
        decoder=decoder,
        prior=prior,
    )
