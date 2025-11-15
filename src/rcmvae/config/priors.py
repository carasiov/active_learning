"""Typed prior configuration classes.

Replaces the string-based prior_type field with a type hierarchy that makes
prior-specific parameters explicit and type-safe.
"""

from __future__ import annotations

import warnings
from abc import ABC
from dataclasses import dataclass


@dataclass
class PriorConfig(ABC):
    """Base class for all prior configurations.

    Subclasses specify prior-type-specific parameters and provide validation.
    This design replaces the string-based prior_type with explicit types.
    """

    def get_prior_type(self) -> str:
        """Return the string identifier for this prior type.

        This method allows polymorphic access to the prior type for legacy code
        and factory functions that need the string identifier.
        """
        raise NotImplementedError


@dataclass
class StandardPriorConfig(PriorConfig):
    """Standard Gaussian prior N(0, I).

    This is the simplest VAE prior with no mixture components or learned parameters.
    """

    def get_prior_type(self) -> str:
        return "standard"


@dataclass
class MixturePriorConfig(PriorConfig):
    """Mixture of Gaussians prior with component-aware decoder.

    Attributes:
        num_components: Number of mixture components (K).
        kl_c_weight: Scaling factor for KL(q(c|x) || π) regularization.
        kl_c_anneal_epochs: If >0, linearly ramp kl_c_weight from 0 to final value.
        learnable_pi: If True, learn mixture weights π during training.
        dirichlet_alpha: Scalar prior strength for Dirichlet-MAP regularization on π.
        dirichlet_weight: Scaling applied to Dirichlet-MAP penalty.
        component_diversity_weight: Component usage diversity regularization.
            - NEGATIVE (e.g., -0.05): Encourage diversity - RECOMMENDED
            - POSITIVE: Discourage diversity (causes mode collapse)
        mixture_history_log_every: Track π and usage every N epochs.
        use_tau_classifier: If True, use τ-based latent-only classification.
        tau_smoothing_alpha: Laplace smoothing prior (α_0) for τ-classifier.
    """

    num_components: int = 10
    kl_c_weight: float = 1.0
    kl_c_anneal_epochs: int = 0
    learnable_pi: bool = False
    dirichlet_alpha: float | None = None
    dirichlet_weight: float = 1.0
    component_diversity_weight: float = 0.0
    mixture_history_log_every: int = 1
    use_tau_classifier: bool = False
    tau_smoothing_alpha: float = 1.0

    def __post_init__(self):
        """Validate mixture prior configuration."""
        if self.num_components <= 0:
            raise ValueError(f"num_components must be positive, got {self.num_components}")
        if self.kl_c_weight < 0:
            raise ValueError(f"kl_c_weight must be non-negative, got {self.kl_c_weight}")
        if self.kl_c_anneal_epochs < 0:
            raise ValueError(f"kl_c_anneal_epochs must be non-negative, got {self.kl_c_anneal_epochs}")
        if self.dirichlet_weight < 0:
            raise ValueError(f"dirichlet_weight must be non-negative, got {self.dirichlet_weight}")
        if self.mixture_history_log_every <= 0:
            raise ValueError(f"mixture_history_log_every must be positive, got {self.mixture_history_log_every}")
        if self.tau_smoothing_alpha <= 0:
            raise ValueError(f"tau_smoothing_alpha must be positive, got {self.tau_smoothing_alpha}")

        if self.dirichlet_alpha is not None and self.dirichlet_alpha <= 0.0:
            warnings.warn(
                "dirichlet_alpha <= 0 disables the Dirichlet prior. "
                "If you intended to turn it off, leave it as None.",
                RuntimeWarning,
            )
            self.dirichlet_alpha = None

    def get_prior_type(self) -> str:
        return "mixture"


@dataclass
class VampPriorConfig(PriorConfig):
    """Variational Mixture of Posteriors (VampPrior).

    Uses learned pseudo-inputs to create a flexible mixture prior with
    spatial separation in latent space.

    Attributes:
        num_components: Number of pseudo-inputs (K).
        kl_c_weight: Scaling factor for KL(q(c|x) || π) regularization.
        kl_c_anneal_epochs: If >0, linearly ramp kl_c_weight from 0 to final value.
        learnable_pi: If True, learn mixture weights π during training.
        dirichlet_alpha: Scalar prior strength for Dirichlet-MAP regularization on π.
        dirichlet_weight: Scaling applied to Dirichlet-MAP penalty.
        component_diversity_weight: Component usage diversity regularization.
        mixture_history_log_every: Track π and usage every N epochs.
        use_tau_classifier: If True, use τ-based latent-only classification.
        tau_smoothing_alpha: Laplace smoothing prior (α_0) for τ-classifier.
        vamp_num_samples_kl: Monte Carlo samples for KL estimation.
        vamp_pseudo_lr_scale: Learning rate scale for pseudo-inputs (relative to model LR).
        vamp_pseudo_init_method: Pseudo-input initialization method ("random" or "kmeans").
    """

    num_components: int = 10
    kl_c_weight: float = 1.0
    kl_c_anneal_epochs: int = 0
    learnable_pi: bool = False
    dirichlet_alpha: float | None = None
    dirichlet_weight: float = 1.0
    component_diversity_weight: float = 0.0
    mixture_history_log_every: int = 1
    use_tau_classifier: bool = False
    tau_smoothing_alpha: float = 1.0
    vamp_num_samples_kl: int = 1
    vamp_pseudo_lr_scale: float = 0.1
    vamp_pseudo_init_method: str = "random"

    def __post_init__(self):
        """Validate VampPrior configuration."""
        if self.num_components <= 0:
            raise ValueError(f"num_components must be positive, got {self.num_components}")
        if self.kl_c_weight < 0:
            raise ValueError(f"kl_c_weight must be non-negative, got {self.kl_c_weight}")
        if self.kl_c_anneal_epochs < 0:
            raise ValueError(f"kl_c_anneal_epochs must be non-negative, got {self.kl_c_anneal_epochs}")
        if self.vamp_num_samples_kl < 1:
            raise ValueError(f"vamp_num_samples_kl must be >= 1, got {self.vamp_num_samples_kl}")
        if self.vamp_pseudo_lr_scale <= 0 or self.vamp_pseudo_lr_scale > 1.0:
            raise ValueError(
                f"vamp_pseudo_lr_scale must be in (0, 1], got {self.vamp_pseudo_lr_scale}"
            )
        if self.tau_smoothing_alpha <= 0:
            raise ValueError(f"tau_smoothing_alpha must be positive, got {self.tau_smoothing_alpha}")

        valid_vamp_init_methods = {"random", "kmeans"}
        if self.vamp_pseudo_init_method not in valid_vamp_init_methods:
            warnings.warn(
                f"vamp_pseudo_init_method should be one of {valid_vamp_init_methods}. "
                f"Got '{self.vamp_pseudo_init_method}'. Downstream code may raise.",
                RuntimeWarning,
            )

        if self.dirichlet_alpha is not None and self.dirichlet_alpha <= 0.0:
            warnings.warn(
                "dirichlet_alpha <= 0 disables the Dirichlet prior. "
                "If you intended to turn it off, leave it as None.",
                RuntimeWarning,
            )
            self.dirichlet_alpha = None

    def get_prior_type(self) -> str:
        return "vamp"


@dataclass
class GeometricMoGPriorConfig(PriorConfig):
    """Geometric Mixture of Gaussians with fixed spatial arrangement.

    WARNING: This prior induces artificial topology on the latent space.
    Use only for diagnostic/curriculum purposes, not production models.

    Attributes:
        num_components: Number of mixture components (K).
        kl_c_weight: Scaling factor for KL(q(c|x) || π) regularization.
        kl_c_anneal_epochs: If >0, linearly ramp kl_c_weight from 0 to final value.
        learnable_pi: If True, learn mixture weights π during training.
        dirichlet_alpha: Scalar prior strength for Dirichlet-MAP regularization on π.
        dirichlet_weight: Scaling applied to Dirichlet-MAP penalty.
        component_diversity_weight: Component usage diversity regularization.
        mixture_history_log_every: Track π and usage every N epochs.
        use_tau_classifier: If True, use τ-based latent-only classification.
        tau_smoothing_alpha: Laplace smoothing prior (α_0) for τ-classifier.
        geometric_arrangement: Geometric arrangement pattern ("circle" or "grid").
        geometric_radius: Radius for circle arrangement (distance from origin).
    """

    num_components: int = 10
    kl_c_weight: float = 1.0
    kl_c_anneal_epochs: int = 0
    learnable_pi: bool = False
    dirichlet_alpha: float | None = None
    dirichlet_weight: float = 1.0
    component_diversity_weight: float = 0.0
    mixture_history_log_every: int = 1
    use_tau_classifier: bool = False
    tau_smoothing_alpha: float = 1.0
    geometric_arrangement: str = "circle"
    geometric_radius: float = 2.0

    def __post_init__(self):
        """Validate Geometric MoG configuration."""
        if self.num_components <= 0:
            raise ValueError(f"num_components must be positive, got {self.num_components}")
        if self.kl_c_weight < 0:
            raise ValueError(f"kl_c_weight must be non-negative, got {self.kl_c_weight}")
        if self.kl_c_anneal_epochs < 0:
            raise ValueError(f"kl_c_anneal_epochs must be non-negative, got {self.kl_c_anneal_epochs}")
        if self.geometric_radius <= 0:
            raise ValueError(f"geometric_radius must be positive, got {self.geometric_radius}")
        if self.tau_smoothing_alpha <= 0:
            raise ValueError(f"tau_smoothing_alpha must be positive, got {self.tau_smoothing_alpha}")

        valid_arrangements = {"circle", "grid"}
        if self.geometric_arrangement not in valid_arrangements:
            raise ValueError(
                f"geometric_arrangement must be one of {valid_arrangements}, got '{self.geometric_arrangement}'"
            )

        if self.dirichlet_alpha is not None and self.dirichlet_alpha <= 0.0:
            warnings.warn(
                "dirichlet_alpha <= 0 disables the Dirichlet prior. "
                "If you intended to turn it off, leave it as None.",
                RuntimeWarning,
            )
            self.dirichlet_alpha = None

        warnings.warn(
            "geometric_mog prior induces artificial topology on latent space. "
            "Use only for diagnostic/curriculum purposes, not production models.",
            UserWarning,
        )

    def get_prior_type(self) -> str:
        return "geometric_mog"
