"""Clean, professional formatting for experiment console output.

Designed for robustness and extensibility:
- Handles all prior types (standard, mixture, vamp, geometric_mog)
- Handles all classifier types (τ-classifier, standard)
- Handles all decoder configurations
- Easy to extend for new features
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def format_experiment_header(
    config: Dict[str, Any],
    run_id: str,
    architecture_code: str,
    output_path: Path,
    data_info: Dict[str, Any] | None = None,
    device_info: tuple[str, int] | None = None,
) -> str:
    """Format complete experiment header with all configuration sections.

    Args:
        config: Full experiment configuration dict
        run_id: Experiment run ID
        architecture_code: Generated architecture code
        output_path: Output directory path
        data_info: Optional data loading info (dataset, sizes, splits)
        device_info: Optional (device_type, device_count) tuple

    Returns:
        Formatted header string ready for printing

    Example:
        >>> header = format_experiment_header(
        ...     config=config,
        ...     run_id="baseline__mix10-dir__20241112_182725",
        ...     architecture_code="mix10-dir_tau_ca-het",
        ...     output_path=Path("results/baseline__mix10-dir__20241112_182725"),
        ...     data_info={"dataset": "MNIST", "total": 10000, "labeled": 100,
        ...               "train_size": 9000, "val_size": 1000},
        ...     device_info=("GPU", 2)
        ... )
        >>> print(header)
    """
    exp_meta = config.get("experiment", {})
    data_config = config.get("data", {})
    model_config = config.get("model", {})

    lines = []
    sep = "=" * 80

    # Header
    lines.append(sep)
    exp_name = exp_meta.get("name", "Experiment")
    lines.append(f"Experiment: {exp_name}")
    lines.append(sep)

    # Experiment identity
    lines.append(f"Architecture:  {architecture_code}")
    lines.append(f"Run ID:        {run_id}")
    # Show relative path for cleaner output
    rel_path = output_path.relative_to(Path.cwd()) if output_path.is_absolute() else output_path
    lines.append(f"Output:        {rel_path}")
    lines.append("")

    # Data configuration
    lines.append(_format_data_config(data_config, data_info))
    lines.append("")

    # Model configuration
    lines.append(_format_model_config(model_config))
    lines.append("")

    # Training configuration
    lines.append(_format_training_config(model_config, device_info))

    return "\n".join(lines)


def format_training_section_header() -> str:
    """Format header for training progress section."""
    sep = "=" * 80
    lines = [
        "",
        sep,
        "Training Progress",
        sep,
    ]
    return "\n".join(lines)


def _format_data_config(
    data_config: Dict[str, Any],
    data_info: Dict[str, Any] | None = None,
) -> str:
    """Format data configuration section."""
    lines = ["Data Configuration:"]

    # Use data_info if provided (from actual data loading), otherwise use config
    if data_info:
        dataset = data_info.get("dataset", data_config.get("dataset", "unknown"))
        total = data_info.get("total", data_config.get("num_samples", 0))
        labeled = data_info.get("labeled", data_config.get("num_labeled", 0))
        train_size = data_info.get("train_size", 0)
        val_size = data_info.get("val_size", 0)
    else:
        dataset = data_config.get("dataset", "unknown")
        total = data_config.get("num_samples", 0)
        labeled = data_config.get("num_labeled", 0)
        val_split = data_config.get("val_split", 0.1)
        train_size = int(total * (1 - val_split))
        val_size = total - train_size

    # Format with proper pluralization
    samples_str = f"{total:,} sample" + ("s" if total != 1 else "")
    labeled_str = f"{labeled:,} labeled"
    lines.append(f"  Dataset:     {dataset.upper()} ({samples_str}, {labeled_str})")

    if val_size > 0:
        lines.append(f"  Split:       {train_size:,} train / {val_size:,} validation")
    else:
        lines.append(f"  Split:       {train_size:,} train (no validation)")

    return "\n".join(lines)


def _format_model_config(model_config: Dict[str, Any]) -> str:
    """Format model configuration section (robust to all config types)."""
    lines = ["Model Configuration:"]

    # Prior configuration (extensible)
    prior_desc = _format_prior_type(model_config)
    lines.append(f"  Prior:       {prior_desc}")

    # Encoder
    encoder_type = model_config.get("encoder_type", "dense")
    latent_dim = model_config.get("latent_dim", 2)
    lines.append(f"  Encoder:     {encoder_type.capitalize()} (latent_dim={latent_dim})")

    # Decoder (can be multiline if complex)
    decoder_desc = _format_decoder_type(model_config)
    lines.append(f"  Decoder:     {decoder_desc}")

    # Classifier
    classifier_desc = _format_classifier_type(model_config)
    lines.append(f"  Classifier:  {classifier_desc}")

    return "\n".join(lines)


def _format_prior_type(model_config: Dict[str, Any]) -> str:
    """Format prior type description (handles all prior types).

    Extensible: Add new prior types here.
    """
    prior_type = model_config.get("prior_type", "standard")

    if prior_type == "standard":
        return "Standard Gaussian N(0, I)"

    elif prior_type == "mixture":
        K = model_config.get("num_components", 10)
        learnable_pi = model_config.get("learnable_pi", False)
        pi_str = "learnable π" if learnable_pi else "fixed π"

        desc = f"Mixture of Gaussians (K={K}, {pi_str})"

        # Add regularization info if present
        reg_parts = []
        dirichlet_alpha = model_config.get("dirichlet_alpha")
        if dirichlet_alpha is not None and dirichlet_alpha > 0:
            reg_parts.append(f"Dirichlet(α={dirichlet_alpha})")

        diversity_weight = model_config.get("component_diversity_weight")
        if diversity_weight is not None and diversity_weight != 0:
            reg_parts.append(f"diversity={diversity_weight}")

        if reg_parts:
            desc += f"\n               Regularization: {', '.join(reg_parts)}"

        return desc

    elif prior_type == "vamp":
        K = model_config.get("num_components", 20)
        init_method = model_config.get("vamp_pseudo_init_method", "kmeans")
        kl_samples = model_config.get("vamp_num_samples_kl", 1)
        lr_scale = model_config.get("vamp_pseudo_lr_scale", 0.1)

        desc = f"VampPrior (K={K}, pseudo-inputs via {init_method})"
        desc += f"\n               KL samples: {kl_samples}, LR scale: {lr_scale}"
        return desc

    elif prior_type == "geometric_mog":
        K = model_config.get("num_components", 9)
        arrangement = model_config.get("geometric_arrangement", "grid")
        radius = model_config.get("geometric_radius", 2.0)

        desc = f"Geometric MoG (K={K}, {arrangement} arrangement)"
        desc += f"\n               Radius: {radius} (fixed positions)"
        return desc

    else:
        # Unknown prior type - graceful fallback
        return f"{prior_type.capitalize()} (custom)"


def _format_classifier_type(model_config: Dict[str, Any]) -> str:
    """Format classifier type description."""
    use_tau = model_config.get("use_tau_classifier", False)
    prior_type = model_config.get("prior_type", "standard")

    # τ-classifier only works with mixture-based priors
    component_priors = {"mixture", "vamp", "geometric_mog"}
    if use_tau and prior_type in component_priors:
        K = model_config.get("num_components", 10)
        num_classes = model_config.get("num_classes", 10)
        alpha = model_config.get("tau_smoothing_alpha", 1.0)

        desc = f"τ-classifier (latent-only, {K} components → {num_classes} classes)"
        if alpha != 1.0:
            desc += f"\n               Smoothing: α={alpha}"
        return desc
    else:
        # Standard classifier
        classifier_type = model_config.get("classifier_type", "dense")
        dropout = model_config.get("dropout_rate", 0.0)

        desc = f"Standard ({classifier_type})"
        if dropout > 0:
            desc += f", dropout={dropout}"
        return desc


def _format_decoder_type(model_config: Dict[str, Any]) -> str:
    """Format decoder type description."""
    decoder_type = model_config.get("decoder_type", "dense")
    base_desc = decoder_type.capitalize()

    features = []

    # Component-aware decoder
    if model_config.get("use_component_aware_decoder", False):
        embed_dim = model_config.get("component_embedding_dim", 8)
        features.append(f"component-aware (embed={embed_dim})")

    # Heteroscedastic decoder
    if model_config.get("use_heteroscedastic_decoder", False):
        sigma_min = model_config.get("sigma_min", 0.05)
        sigma_max = model_config.get("sigma_max", 0.5)
        features.append(f"heteroscedastic (σ ∈ [{sigma_min}, {sigma_max}])")

    if features:
        return f"{base_desc}, {', '.join(features)}"
    else:
        return base_desc


def _format_training_config(
    model_config: Dict[str, Any],
    device_info: tuple[str, int] | None = None,
) -> str:
    """Format training configuration section."""
    lines = ["Training Configuration:"]

    # Device info
    if device_info:
        device_type, device_count = device_info
        plural = "s" if device_count != 1 else ""
        lines.append(f"  Device:      {device_type.upper()} ({device_count} device{plural})")
    else:
        lines.append(f"  Device:      (detecting...)")

    # Optimizer
    lr = model_config.get("learning_rate", 0.001)
    wd = model_config.get("weight_decay", 0.0)
    lines.append(f"  Optimizer:   Adam (lr={lr}, weight_decay={wd})")

    # Training setup
    batch_size = model_config.get("batch_size", 128)
    max_epochs = model_config.get("max_epochs", 100)
    patience = model_config.get("patience", 20)
    monitor_metric = model_config.get("monitor_metric", "loss")

    lines.append(f"  Batch size:  {batch_size}")
    lines.append(f"  Epochs:      {max_epochs} (patience={patience})")
    lines.append(f"  Monitoring:  validation {monitor_metric}")

    return "\n".join(lines)
