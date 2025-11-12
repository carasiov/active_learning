"""Configuration validation for experiment management.

This module enforces architectural constraints and validates config before
training starts. Following AGENTS.md principle: "fail fast" - validate at
config load time, not after 20 minutes of training.

Validation rules mirror the naming constraints:
1. τ-classifier requires mixture-based prior (SSVAEConfig warns + auto-disables)
2. Component-aware decoder requires mixture-based prior (SSVAEConfig warns, factory handles fallback)
3. VampPrior requires initialization method (validated here + SSVAEConfig)
4. Geometric MoG requires arrangement and validates grid constraints (validated here)
5. Other architectural invariants

Note: Some validations (τ-classifier, component-aware decoder) issue warnings and
continue execution, while others (VampPrior invalid init, non-square grid) are
hard errors. This reflects whether the system can gracefully handle the misconfiguration.

Usage:
    from src.core.validation import validate_config

    try:
        validate_config(config)
    except ValueError as e:
        print(f"Invalid config: {e}")
        sys.exit(1)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ssvae.config import SSVAEConfig


class ConfigValidationError(ValueError):
    """Raised when configuration violates architectural constraints.

    This is a ValueError subtype for backward compatibility, but provides
    a distinct type for catching experiment-specific validation errors.
    """
    pass


def validate_config(config: SSVAEConfig) -> None:
    """Validate experiment configuration against architectural constraints.

    This performs additional validation beyond SSVAEConfig.__post_init__(),
    specifically for experiment management requirements.

    Args:
        config: SSVAE configuration to validate

    Raises:
        ConfigValidationError: If configuration violates constraints

    Example:
        >>> config = SSVAEConfig(
        ...     prior_type="standard",
        ...     use_tau_classifier=True  # Invalid combination
        ... )
        >>> validate_config(config)  # Raises ConfigValidationError
    """
    # Run all validation checks
    _validate_tau_classifier(config)
    # _validate_component_aware_decoder(config)  # Moved to SSVAEConfig (warning, not error)
    _validate_vamp_prior(config)
    _validate_geometric_mog(config)
    _validate_heteroscedastic_decoder(config)


def _validate_tau_classifier(config: SSVAEConfig) -> None:
    """Validate τ-classifier configuration.

    Rules:
    1. Requires mixture-based prior (mixture, vamp, or geometric_mog)
    2. Requires num_components >= num_classes

    Note: Rule 2 is already enforced by SSVAEConfig.__post_init__(),
    but we include it here for completeness.
    """
    if not config.use_tau_classifier:
        return  # Not enabled, skip validation

    mixture_based_priors = {"mixture", "vamp", "geometric_mog"}

    if config.prior_type not in mixture_based_priors:
        raise ConfigValidationError(
            f"τ-classifier (use_tau_classifier=true) requires mixture-based prior. "
            f"Got prior_type='{config.prior_type}'. "
            f"Valid priors: {mixture_based_priors}"
        )

    # This is already checked in SSVAEConfig.__post_init__, but include for clarity
    if config.num_components < config.num_classes:
        raise ConfigValidationError(
            f"τ-classifier requires num_components >= num_classes. "
            f"Got num_components={config.num_components}, "
            f"num_classes={config.num_classes}"
        )


def _validate_component_aware_decoder(config: SSVAEConfig) -> None:
    """Validate component-aware decoder configuration.

    DEPRECATED: This validation was moved to SSVAEConfig.__post_init__() as a warning
    (not error) for consistency with τ-classifier and learnable_pi behavior.

    The factory (src/ssvae/components/factory.py) handles the fallback gracefully,
    so this should be a warning that allows execution to continue, not a hard error.

    Rule: Requires mixture-based prior (standard prior has no components).
    """
    # This function is no longer called (commented out in validate_config)
    # Kept for reference/documentation purposes
    if not config.use_component_aware_decoder:
        return  # Not enabled, skip validation

    mixture_based_priors = {"mixture", "vamp", "geometric_mog"}

    if config.prior_type not in mixture_based_priors:
        raise ConfigValidationError(
            f"Component-aware decoder (use_component_aware_decoder=true) "
            f"requires mixture-based prior. "
            f"Got prior_type='{config.prior_type}'. "
            f"Valid priors: {mixture_based_priors}"
        )


def _validate_vamp_prior(config: SSVAEConfig) -> None:
    """Validate VampPrior configuration.

    Rules:
    1. Must specify initialization method ("kmeans" or "random")
    2. num_samples_kl should be reasonable (1-10)
    3. pseudo_lr_scale should be smaller than main learning rate
    """
    if config.prior_type != "vamp":
        return  # Not VampPrior, skip validation

    # Rule 1: Initialization method
    valid_init_methods = {"kmeans", "random"}
    if config.vamp_pseudo_init_method not in valid_init_methods:
        raise ConfigValidationError(
            f"VampPrior requires vamp_pseudo_init_method in {valid_init_methods}. "
            f"Got '{config.vamp_pseudo_init_method}'"
        )

    # Rule 2: Reasonable number of MC samples
    if config.vamp_num_samples_kl < 1:
        raise ConfigValidationError(
            f"vamp_num_samples_kl must be >= 1, got {config.vamp_num_samples_kl}"
        )
    if config.vamp_num_samples_kl > 10:
        # Warning, not error (might be intentional for low-variance estimation)
        import warnings
        warnings.warn(
            f"vamp_num_samples_kl={config.vamp_num_samples_kl} is high. "
            "This will slow training significantly. Typical range: 1-10."
        )

    # Rule 3: Pseudo-input LR should be smaller than main LR
    if config.vamp_pseudo_lr_scale >= 1.0:
        import warnings
        warnings.warn(
            f"vamp_pseudo_lr_scale={config.vamp_pseudo_lr_scale} is >= 1.0. "
            "Pseudo-inputs typically learn slower than network parameters. "
            "Recommended range: 0.05-0.2"
        )


def _validate_geometric_mog(config: SSVAEConfig) -> None:
    """Validate Geometric MoG configuration.

    Rules:
    1. Must specify arrangement ("circle" or "grid")
    2. Grid arrangement requires perfect square num_components
    3. Radius should be reasonable (not too small or too large)
    4. Issue warning about induced topology
    """
    if config.prior_type != "geometric_mog":
        return  # Not geometric MoG, skip validation

    # Rule 1: Arrangement specification
    valid_arrangements = {"circle", "grid"}
    if config.geometric_arrangement not in valid_arrangements:
        raise ConfigValidationError(
            f"Geometric MoG requires geometric_arrangement in {valid_arrangements}. "
            f"Got '{config.geometric_arrangement}'"
        )

    # Rule 2: Grid requires perfect square K
    if config.geometric_arrangement == "grid":
        k = config.num_components
        sqrt_k = int(k ** 0.5)
        if sqrt_k * sqrt_k != k:
            raise ConfigValidationError(
                f"Geometric grid arrangement requires perfect square num_components. "
                f"Got num_components={k}, which is not a perfect square. "
                f"Try: 4, 9, 16, 25, 36, 49, 64, 81, 100"
            )

    # Rule 3: Reasonable radius
    radius = config.geometric_radius
    if radius < 0.5:
        import warnings
        warnings.warn(
            f"geometric_radius={radius} is very small. "
            "Components may overlap excessively. Recommended range: 1.5-3.0"
        )
    if radius > 5.0:
        import warnings
        warnings.warn(
            f"geometric_radius={radius} is very large. "
            "Latent space may be poorly utilized. Recommended range: 1.5-3.0"
        )

    # Rule 4: Topology warning
    import warnings
    warnings.warn(
        "Geometric MoG prior induces topology in latent space. "
        "This is a diagnostic tool only. For production use, prefer "
        "mixture or VampPrior."
    )


def _validate_heteroscedastic_decoder(config: SSVAEConfig) -> None:
    """Validate heteroscedastic decoder configuration.

    Rules:
    1. sigma_min must be positive
    2. sigma_max must be > sigma_min
    3. Range should be reasonable (not too narrow or too wide)

    Note: Rules 1-2 are already enforced by SSVAEConfig.__post_init__(),
    but we include them here with better error messages.
    """
    if not config.use_heteroscedastic_decoder:
        return  # Not enabled, skip validation

    # Rules 1-2: Already checked in __post_init__, but provide context
    if config.sigma_min <= 0:
        raise ConfigValidationError(
            f"sigma_min must be positive, got {config.sigma_min}"
        )

    if config.sigma_max <= config.sigma_min:
        raise ConfigValidationError(
            f"sigma_max must be > sigma_min. "
            f"Got sigma_min={config.sigma_min}, sigma_max={config.sigma_max}"
        )

    # Rule 3: Reasonable range
    ratio = config.sigma_max / config.sigma_min
    if ratio < 2.0:
        import warnings
        warnings.warn(
            f"Heteroscedastic range is narrow: "
            f"sigma_min={config.sigma_min}, sigma_max={config.sigma_max} "
            f"(ratio={ratio:.1f}). "
            "Consider wider range for better uncertainty estimation. "
            "Typical: sigma_min=0.05, sigma_max=0.5 (ratio=10)"
        )
    if ratio > 100:
        import warnings
        warnings.warn(
            f"Heteroscedastic range is very wide: "
            f"sigma_min={config.sigma_min}, sigma_max={config.sigma_max} "
            f"(ratio={ratio:.0f}). "
            "This may cause numerical instability."
        )


# Optional: Validation for common configuration mistakes
def validate_hyperparameters(config: SSVAEConfig) -> None:
    """Validate hyperparameter choices (warnings only, not errors).

    This checks for common mistakes or unusual settings that might indicate
    configuration errors, but doesn't enforce hard constraints.

    Args:
        config: SSVAE configuration to check
    """
    import warnings

    # KL weight checks
    if config.kl_weight > 10:
        warnings.warn(
            f"kl_weight={config.kl_weight} is unusually high. "
            "This may cause posterior collapse. Typical range: 0.1-1.0"
        )
    if config.kl_weight < 0.01:
        warnings.warn(
            f"kl_weight={config.kl_weight} is unusually low. "
            "The latent space may not be regularized. Typical range: 0.1-1.0"
        )

    # Component KL weight (mixture only)
    if config.is_mixture_based_prior() and config.kl_c_weight > 1.0:
        warnings.warn(
            f"kl_c_weight={config.kl_c_weight} is high. "
            "This may cause component collapse. Typical range: 0.0001-0.001"
        )

    # Diversity weight sign check
    if config.is_mixture_based_prior() and config.component_diversity_weight > 0:
        warnings.warn(
            f"component_diversity_weight={config.component_diversity_weight} is POSITIVE. "
            "This DISCOURAGES diversity (may cause collapse). "
            "To ENCOURAGE diversity, use NEGATIVE values (e.g., -0.05 to -0.15)"
        )

    # Learning rate checks
    if config.learning_rate > 0.01:
        warnings.warn(
            f"learning_rate={config.learning_rate} is high. "
            "Training may be unstable. Typical range: 0.0001-0.001"
        )
    if config.learning_rate < 1e-5:
        warnings.warn(
            f"learning_rate={config.learning_rate} is very low. "
            "Training may be slow. Typical range: 0.0001-0.001"
        )

    # Batch size checks
    if config.batch_size < 32:
        warnings.warn(
            f"batch_size={config.batch_size} is small. "
            "Gradient estimates may be noisy. Typical range: 64-256"
        )
    if config.batch_size > 1024:
        warnings.warn(
            f"batch_size={config.batch_size} is large. "
            "May require learning rate adjustment. Typical range: 64-256"
        )
