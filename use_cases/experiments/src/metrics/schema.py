"""Canonical naming schema for all metrics and losses.

This module defines the single source of truth for metric naming across:
- Training logs (stdout)
- Summary JSON files
- CSV histories
- Plots and reports

Design principles (from AGENTS.md):
- Single source of truth: All metric names defined here
- Stability: These are stable API contracts (change carefully)
- Grep-friendly: Use dotted notation for log messages
- Nested for JSON: Convert to nested dicts for summary.json

Adding a new metric:
1. Add constant to appropriate class below
2. Use constant in metric provider (autocomplete prevents typos)
3. Update REPORT template if metric should be displayed
4. Document in experiment guide (use_cases/experiments/README.md)

Example:
    from .schema import LossKeys, MixtureKeys

    @register_metric
    def compute_mixture_metrics(context):
        return ComponentResult.success(data={
            MixtureKeys.K_EFF: 7.3,
            MixtureKeys.ACTIVE_COMPONENTS: 8,
        })
"""
from __future__ import annotations

from typing import Any, Dict


class LossKeys:
    """Core loss components from training loop.

    These correspond to terms in the ELBO objective (see docs/theory/mathematical_specification.md §4).
    """
    # Total objective
    TOTAL = "loss.total"

    # Reconstruction term: E_q[log p(x|z,c)]
    RECON = "loss.recon"

    # KL divergences
    KL_Z = "loss.kl_z"  # KL[q(z|x) || p(z|c)]
    KL_C = "loss.kl_c"  # KL[q(c|x) || p(c)]

    # Classification loss (when using standard head)
    CLASSIFIER = "loss.classifier"

    # Mixture regularization
    DIRICHLET = "loss.dirichlet"  # Dirichlet prior on π
    DIVERSITY = "loss.diversity"  # Entropy reward/penalty on component usage

    # Optional losses
    CONTRASTIVE = "loss.contrastive"  # Contrastive learning term


class MetricKeys:
    """Classification accuracy and performance metrics."""
    # Accuracy splits
    ACC_TRAIN = "metric.acc.train"
    ACC_VAL = "metric.acc.val"
    ACC_TEST = "metric.acc.test"

    # Loss values on different splits
    LOSS_TRAIN = "metric.loss.train"
    LOSS_VAL = "metric.loss.val"
    LOSS_TEST = "metric.loss.test"


class MixtureKeys:
    """Mixture-specific metrics (only for mixture-based priors).

    See docs/theory/conceptual_model.md §How-We-Mix for design rationale.
    """
    # Effective number of components (renormalized entropy of π)
    K_EFF = "mixture.K_eff"

    # Number of components with >1% usage
    ACTIVE_COMPONENTS = "mixture.active_components"

    # Mean of max_c q(c|x) across samples (confidence in assignments)
    RESPONSIBILITY_CONFIDENCE = "mixture.responsibility_confidence_mean"

    # Entropy of mixture weights π
    PI_ENTROPY = "mixture.pi_entropy"

    # Component entropy: -Σ_c q(c|x) log q(c|x)
    COMPONENT_ENTROPY = "mixture.component_entropy"

    # Mixture weight statistics
    PI_MAX = "mixture.pi_max"
    PI_MIN = "mixture.pi_min"
    PI_ARGMAX = "mixture.pi_argmax"


class TauKeys:
    """τ-classifier metrics (latent-only classification).

    See docs/theory/conceptual_model.md §How-We-Classify for τ-classifier design.
    """
    # How many labels have at least one component assigned
    LABEL_COVERAGE = "tau.label_coverage"

    # Average components per label (from τ > 0.15 threshold)
    COMPONENTS_PER_LABEL_MEAN = "tau.components_per_label_mean"

    # Mean prediction certainty: Σ_c q(c|x) max_y τ_{c,y}
    CERTAINTY_MEAN = "tau.certainty_mean"
    CERTAINTY_STD = "tau.certainty_std"
    CERTAINTY_MIN = "tau.certainty_min"
    CERTAINTY_MAX = "tau.certainty_max"

    # OOD score: max_c q(c|x) × (1 - max_y τ_{c,y})
    OOD_SCORE_MEAN = "tau.ood_score_mean"
    OOD_SCORE_STD = "tau.ood_score_std"

    # Free channels: components with low usage AND low τ confidence
    NUM_FREE_CHANNELS = "tau.num_free_channels"

    # Matrix properties
    TAU_SPARSITY = "tau.matrix_sparsity"  # Fraction of τ_{c,y} < 0.05


class ClusteringKeys:
    """Clustering alignment metrics (only for 2D latent space).

    These compare component assignments to true labels as a diagnostic.
    Only computed when latent_dim=2 for visualization purposes.
    """
    # Normalized Mutual Information
    NMI = "clustering.nmi"

    # Adjusted Rand Index
    ARI = "clustering.ari"


class TrainingKeys:
    """Training metadata and timing."""
    # Wall-clock time (seconds)
    TIME_SEC = "training.time_sec"

    # Number of epochs completed
    EPOCHS_COMPLETED = "training.epochs_completed"

    # Final loss value (last epoch)
    FINAL_LOSS = "training.final_loss"

    # Early stopping info
    BEST_EPOCH = "training.best_epoch"
    BEST_LOSS = "training.best_loss"


class UncertaintyKeys:
    """Heteroscedastic decoder uncertainty metrics."""
    # Mean predicted variance
    VARIANCE_MEAN = "uncertainty.variance_mean"
    VARIANCE_STD = "uncertainty.variance_std"

    # Calibration metrics (if ground truth available)
    CALIBRATION_ERROR = "uncertainty.calibration_error"


def to_nested_dict(flat_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert flat dotted keys to nested dict structure.

    This is used when writing summary.json to create clean nested structure
    instead of flat dotted keys.

    Args:
        flat_metrics: Dict with dotted keys like {"loss.recon": 0.5}

    Returns:
        Nested dict like {"loss": {"recon": 0.5}}

    Example:
        >>> to_nested_dict({
        ...     "loss.recon": 0.5,
        ...     "loss.kl_z": 1.2,
        ...     "mixture.K_eff": 7.3
        ... })
        {
            "loss": {"recon": 0.5, "kl_z": 1.2},
            "mixture": {"K_eff": 7.3}
        }
    """
    result: Dict[str, Any] = {}

    for key, value in flat_metrics.items():
        parts = key.split(".")
        current = result

        # Navigate/create nested structure
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set leaf value
        current[parts[-1]] = value

    return result


def flatten_dict(nested: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Convert nested dict to flat dotted keys.

    Inverse of to_nested_dict. Useful for reading existing summary.json files.

    Args:
        nested: Nested dict structure
        prefix: Current prefix (used in recursion)

    Returns:
        Flat dict with dotted keys

    Example:
        >>> flatten_dict({
        ...     "loss": {"recon": 0.5, "kl_z": 1.2},
        ...     "mixture": {"K_eff": 7.3}
        ... })
        {"loss.recon": 0.5, "loss.kl_z": 1.2, "mixture.K_eff": 7.3}
    """
    result: Dict[str, Any] = {}

    for key, value in nested.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            result.update(flatten_dict(value, prefix=full_key))
        else:
            result[full_key] = value

    return result


# Legacy key mappings for backward compatibility
# Maps old unstructured keys to new canonical keys
LEGACY_KEY_MAP = {
    "final_loss": LossKeys.TOTAL,
    "final_recon_loss": LossKeys.RECON,
    "final_kl_z": LossKeys.KL_Z,
    "final_kl_c": LossKeys.KL_C,
    "final_accuracy": MetricKeys.ACC_TRAIN,
    "training_time_sec": TrainingKeys.TIME_SEC,
    "epochs_completed": TrainingKeys.EPOCHS_COMPLETED,
}
