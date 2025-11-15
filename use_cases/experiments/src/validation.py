"""Configuration validation for experiment management.

This module enforces architectural constraints and validates config before
training starts. Following AGENTS.md principle: "fail fast" - validate at
config load time, not after 20 minutes of training.

Validation rules mirror the naming constraints:
1. τ-classifier requires mixture-based prior (ExperimentConfig warns + auto-disables)
2. Component-aware decoder requires mixture-based prior (ExperimentConfig warns, factory handles fallback)
3. VampPrior requires initialization method (validated here + ExperimentConfig)
4. Geometric MoG requires arrangement and validates grid constraints (validated here)
5. Other architectural invariants

Note: Some validations (τ-classifier, component-aware decoder) issue warnings and
continue execution, while others (VampPrior invalid init, non-square grid) are
hard errors. This reflects whether the system can gracefully handle the misconfiguration.

Usage:
    from use_cases.experiments.src.validation import validate_config

    try:
        validate_config(config)
    except ValueError as e:
        print(f"Invalid config: {e}")
        sys.exit(1)
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from rcmvae.config import ExperimentConfig


class ConfigValidationError(ValueError):
    """Raised when configuration violates architectural constraints.

    This is a ValueError subtype for backward compatibility, but provides
    a distinct type for catching experiment-specific validation errors.
    """
    pass


def validate_config(config: ExperimentConfig) -> None:
    """Validate experiment configuration against architectural constraints.

    This performs additional validation beyond ExperimentConfig.__post_init__(),
    specifically for experiment management requirements.

    Args:
        config: Experiment configuration to validate

    Raises:
        ConfigValidationError: If configuration violates constraints

    Example:
        >>> from rcmvae.config import ExperimentConfig, StandardPriorConfig, MixturePriorConfig
        >>> config = ExperimentConfig(
        ...     prior=MixturePriorConfig(use_tau_classifier=True, num_components=5),
        ...     network=NetworkConfig(num_classes=10)
        ... )
        >>> validate_config(config)  # Raises ConfigValidationError
    """
    # Run all validation checks
    _validate_tau_classifier(config)
    _validate_component_aware_decoder(config)
    _validate_vamp_prior(config)
    _validate_geometric_mog(config)
    _validate_heteroscedastic_decoder(config)


def _validate_tau_classifier(config: ExperimentConfig) -> None:
    """Validate τ-classifier configuration.

    Rules:
    1. Requires mixture-based prior (mixture, vamp, or geometric_mog)
    2. Requires num_components >= num_classes

    Note: Rule 2 is already enforced by ExperimentConfig.__post_init__(),
    but we include it here for completeness.
    """
    from rcmvae.config import MixturePriorConfig, VampPriorConfig, GeometricMoGPriorConfig

    prior = config.prior
    if not isinstance(prior, (MixturePriorConfig, VampPriorConfig, GeometricMoGPriorConfig)):
        return  # Not mixture-based, skip

    if not prior.use_tau_classifier:
        return  # Not enabled, skip validation

    prior_type = config.get_prior_type()
    mixture_based_priors = {"mixture", "vamp", "geometric_mog"}

    if prior_type not in mixture_based_priors:
        raise ConfigValidationError(
            f"τ-classifier (use_tau_classifier=true) requires mixture-based prior. "
            f"Got prior_type='{prior_type}'. "
            f"Valid priors: {mixture_based_priors}"
        )

    # This is already checked in ExperimentConfig.__post_init__, but include for clarity
    if prior.num_components < config.network.num_classes:
        warnings.warn(
            f"τ-classifier typically requires num_components >= num_classes "
            f"(got {prior.num_components} vs {config.network.num_classes}). "
            "Proceeding, but expect degraded performance.",
            RuntimeWarning,
        )


def _validate_component_aware_decoder(config: ExperimentConfig) -> None:
    """Validate component-aware decoder configuration.

    NOTE: This validation was moved to ExperimentConfig.__post_init__() as a warning
    for consistency with τ-classifier and learnable_pi behavior. The factory
    handles the fallback gracefully, so this is a warning rather than an error.

    Rule: Requires mixture-based prior (standard prior has no components).

    This function is no longer called but kept for reference.
    """
    if not config.decoder.use_component_aware_decoder:
        return  # Not enabled, skip validation

    mixture_based_priors = {"mixture", "vamp", "geometric_mog"}
    prior_type = config.get_prior_type()

    if prior_type not in mixture_based_priors:
        raise ConfigValidationError(
            f"Component-aware decoder (use_component_aware_decoder=true) "
            f"requires mixture-based prior. "
            f"Got prior_type='{prior_type}'. "
            f"Valid priors: {mixture_based_priors}"
        )


def _validate_vamp_prior(config: ExperimentConfig) -> None:
    """Validate VampPrior configuration.

    Rules:
    1. Must specify initialization method ("kmeans" or "random")
    2. num_samples_kl should be reasonable (1-10)
    3. pseudo_lr_scale should be smaller than main learning rate
    """
    from rcmvae.config import VampPriorConfig

    if not isinstance(config.prior, VampPriorConfig):
        return  # Not VampPrior, skip validation

    prior = config.prior

    # Rule 1: Initialization method
    valid_init_methods = {"kmeans", "random"}
    if prior.vamp_pseudo_init_method not in valid_init_methods:
        raise ConfigValidationError(
            f"VampPrior requires vamp_pseudo_init_method in {valid_init_methods}. "
            f"Got '{prior.vamp_pseudo_init_method}'"
        )

    # Rule 2: Reasonable number of MC samples
    if prior.vamp_num_samples_kl < 1:
        raise ConfigValidationError(
            f"vamp_num_samples_kl must be >= 1, got {prior.vamp_num_samples_kl}"
        )
    if prior.vamp_num_samples_kl > 10:
        # Warning, not error (might be intentional for low-variance estimation)
        warnings.warn(
            f"vamp_num_samples_kl={prior.vamp_num_samples_kl} is high. "
            "This will slow training significantly. Typical range: 1-10."
        )

    # Rule 3: Pseudo-input LR should be smaller than main LR
    if prior.vamp_pseudo_lr_scale >= 1.0:
        warnings.warn(
            f"vamp_pseudo_lr_scale={prior.vamp_pseudo_lr_scale} is >= 1.0. "
            "Pseudo-inputs typically learn slower than network parameters. "
            "Recommended range: 0.05-0.2"
        )


def _validate_geometric_mog(config: ExperimentConfig) -> None:
    """Validate Geometric MoG configuration.

    Rules:
    1. Must specify arrangement ("circle" or "grid")
    2. Grid arrangement requires perfect square num_components
    3. Radius should be reasonable (not too small or too large)

    Note: Topology warning is issued by GeometricMoGPriorConfig.__post_init__().
    """
    from rcmvae.config import GeometricMoGPriorConfig

    if not isinstance(config.prior, GeometricMoGPriorConfig):
        return  # Not geometric MoG, skip validation

    prior = config.prior

    warnings.warn(
        "Geometric MoG prior induces fixed topology in latent space. "
        "Use for diagnostics only.",
        RuntimeWarning,
    )

    # Rule 1: Arrangement specification
    valid_arrangements = {"circle", "grid"}
    if prior.geometric_arrangement not in valid_arrangements:
        raise ConfigValidationError(
            f"Geometric MoG requires geometric_arrangement in {valid_arrangements}. "
            f"Got '{prior.geometric_arrangement}'"
        )

    # Rule 2: Grid requires perfect square K
    if prior.geometric_arrangement == "grid":
        k = prior.num_components
        sqrt_k = int(k ** 0.5)
        if sqrt_k * sqrt_k != k:
            raise ConfigValidationError(
                f"Geometric grid arrangement requires perfect square num_components. "
                f"Got num_components={k}, which is not a perfect square. "
                f"Try: 4, 9, 16, 25, 36, 49, 64, 81, 100"
            )

    # Rule 3: Reasonable radius
    radius = prior.geometric_radius
    if radius < 0.5:
        warnings.warn(
            f"geometric_radius={radius} is very small. "
            "Components may overlap excessively. Recommended range: 1.5-3.0"
        )
    if radius > 5.0:
        warnings.warn(
            f"geometric_radius={radius} is very large. "
            "Latent space may be poorly utilized. Recommended range: 1.5-3.0"
        )

    # Note: Topology warning is issued by GeometricMoGPriorConfig.__post_init__() to avoid duplication


def _validate_heteroscedastic_decoder(config: ExperimentConfig) -> None:
    """Validate heteroscedastic decoder configuration.

    Rules:
    1. sigma_min must be positive
    2. sigma_max must be > sigma_min
    3. Range should be reasonable (not too narrow or too wide)

    Note: Rules 1-2 are already enforced by DecoderFeatures.__post_init__(),
    but we include them here with better error messages.
    """
    if not config.decoder.use_heteroscedastic_decoder:
        return  # Not enabled, skip validation

    # Rules 1-2: Already checked in __post_init__, but provide context
    if config.decoder.sigma_min <= 0:
        raise ConfigValidationError(
            f"sigma_min must be positive, got {config.decoder.sigma_min}"
        )

    if config.decoder.sigma_max <= config.decoder.sigma_min:
        raise ConfigValidationError(
            f"sigma_max must be > sigma_min. "
            f"Got sigma_min={config.decoder.sigma_min}, sigma_max={config.decoder.sigma_max}"
        )

    # Rule 3: Reasonable range
    ratio = config.decoder.sigma_max / config.decoder.sigma_min
    if ratio < 2.0:
        warnings.warn(
            f"Heteroscedastic range is narrow: "
            f"sigma_min={config.decoder.sigma_min}, sigma_max={config.decoder.sigma_max} "
            f"(ratio={ratio:.1f}). "
            "Consider wider range for better uncertainty estimation. "
            "Typical: sigma_min=0.05, sigma_max=0.5 (ratio=10)"
        )
    if ratio > 100:
        warnings.warn(
            f"Heteroscedastic range is very wide: "
            f"sigma_min={config.decoder.sigma_min}, sigma_max={config.decoder.sigma_max} "
            f"(ratio={ratio:.0f}). "
            "This may cause numerical instability."
        )


# Optional: Validation for common configuration mistakes
def validate_hyperparameters(config: ExperimentConfig) -> None:
    """Validate hyperparameter choices (warnings only, not errors).

    This checks for common mistakes or unusual settings that might indicate
    configuration errors, but doesn't enforce hard constraints.

    Args:
        config: Experiment configuration to check
    """
    from rcmvae.config import MixturePriorConfig, VampPriorConfig, GeometricMoGPriorConfig

    # KL weight checks
    if config.loss.kl_weight > 10:
        warnings.warn(
            f"kl_weight={config.loss.kl_weight} is unusually high. "
            "This may cause posterior collapse. Typical range: 0.1-1.0"
        )
    if config.loss.kl_weight < 0.01:
        warnings.warn(
            f"kl_weight={config.loss.kl_weight} is unusually low. "
            "The latent space may not be regularized. Typical range: 0.1-1.0"
        )

    # Component KL weight (mixture only)
    prior = config.prior
    if isinstance(prior, (MixturePriorConfig, VampPriorConfig, GeometricMoGPriorConfig)):
        if prior.kl_c_weight > 1.0:
            warnings.warn(
                f"kl_c_weight={prior.kl_c_weight} is high. "
                "This may cause component collapse. Typical range: 0.0001-0.001"
            )

        # Diversity weight sign check
        if prior.component_diversity_weight > 0:
            warnings.warn(
                f"component_diversity_weight={prior.component_diversity_weight} is POSITIVE. "
                "This DISCOURAGES diversity (may cause collapse). "
                "To ENCOURAGE diversity, use NEGATIVE values (e.g., -0.05 to -0.15)"
            )

    # Learning rate checks
    if config.training.learning_rate > 0.01:
        warnings.warn(
            f"learning_rate={config.training.learning_rate} is high. "
            "Training may be unstable. Typical range: 0.0001-0.001"
        )
    if config.training.learning_rate < 1e-5:
        warnings.warn(
            f"learning_rate={config.training.learning_rate} is very low. "
            "Training may be slow. Typical range: 0.0001-0.001"
        )

    # Batch size checks
    if config.training.batch_size < 32:
        warnings.warn(
            f"batch_size={config.training.batch_size} is small. "
            "Gradient estimates may be noisy. Typical range: 64-256"
        )
    if config.training.batch_size > 1024:
        warnings.warn(
            f"batch_size={config.training.batch_size} is large. "
            "May require learning rate adjustment. Typical range: 64-256"
        )
