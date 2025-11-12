from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

"""Base configuration for the SSVAE models.

Architecture options
- encoder_type: "dense" or "conv"
- decoder_type: "dense" or "conv"
- classifier_type: "dense" (classifier operates on the latent and is architecture-agnostic)

Notes
- Dense path uses `hidden_dims` to define encoder MLP sizes; decoder mirrors them in reverse.
- Conv path is hardcoded for MNIST 28x28 with a small, standard Conv → Conv and mirrored ConvTranspose → ConvTranspose stack.
- `input_hw` can be provided to override output shape; for the conv decoder it must be (28, 28).

Quick recipes
- Dense baseline (default): SSVAEConfig(encoder_type="dense", decoder_type="dense")
- Conv MNIST: SSVAEConfig(encoder_type="conv", decoder_type="conv")

CLI usage (use_cases/scripts/train.py)
- Train conv:  `python use_cases/scripts/train.py --encoder-type conv --decoder-type conv --latent-dim 2 --batch-size 512 --max-epochs 50`
- Train dense: `python use_cases/scripts/train.py --encoder-type dense --decoder-type dense`

Programmatic usage
- from ssvae import SSVAE
  config = SSVAEConfig(encoder_type="conv", decoder_type="conv", latent_dim=2)
  vae = SSVAE(input_dim=(28, 28), config=config)
  vae.fit(x, labels, weights_path)
"""

INFORMATIVE_HPARAMETERS = (
    "encoder_type",
    "decoder_type",
    "classifier_type",
    "num_classes",
    "latent_dim",
    "hidden_dims",
    "learning_rate",
    "batch_size",
    "recon_weight",
    "reconstruction_loss",
    "kl_weight",
    "label_weight",
    "kl_c_weight",
    "dirichlet_alpha",
    "dirichlet_weight",
    "learnable_pi",
    "component_diversity_weight",
    "weight_decay",
    "dropout_rate",
    "monitor_metric",
    "use_contrastive",
    "contrastive_weight",
    "prior_type",
    "num_components",
    "kl_c_anneal_epochs",
    "component_embedding_dim",
    "use_component_aware_decoder",
    "top_m_gating",
    "soft_embedding_warmup_epochs",
    "use_tau_classifier",
    "tau_smoothing_alpha",
    "use_heteroscedastic_decoder",
    "sigma_min",
    "sigma_max",
    "vamp_num_samples_kl",
    "vamp_pseudo_lr_scale",
    "vamp_pseudo_init_method",
    "geometric_arrangement",
    "geometric_radius",
)

@dataclass
class SSVAEConfig:
    """Hyperparameters controlling the SSVAE architecture and training loop.

    Attributes:
        num_classes: Number of output classes for the classifier head.
        latent_dim: Dimensionality of the latent representation.
        hidden_dims: Dense layer sizes for the encoder; decoder mirrors in reverse (dense only).
        reconstruction_loss: Loss function for reconstruction term.
            - "mse": Mean squared error, treats pixels as continuous Gaussian.
                     Appropriate for natural images. Default weight: 500.
            - "bce": Binary cross-entropy with logits, treats pixels as Bernoulli.
                     Appropriate for binary/binarized images (e.g., MNIST).
                     Recommended weight: 1.0 (BCE is already pixel-wise summed).
        recon_weight: Weight applied to the reconstruction term. 
            Typical values: 500 for MSE, 1.0 for BCE (due to different scales).
        kl_weight: Scaling factor for the KL divergence regularizer.
        learning_rate: Optimizer learning rate.
        batch_size: Number of samples per training batch.
        max_epochs: Maximum number of training epochs.
        patience: Early stopping patience measured in epochs without validation improvement.
        val_split: Fraction of the dataset reserved for validation.
        random_seed: Base random seed used for parameter initialization and shuffling.
        grad_clip_norm: Global norm threshold for gradient clipping; disabled when ``None``.
        weight_decay: L2-style weight decay applied through the optimizer.
        dropout_rate: Dropout applied inside the classifier network.
        label_weight: (Unused today) scaling factor for the classification loss term.
        input_hw: Optional (height, width) tuple for decoder output; defaults to the model input.
        encoder_type: Identifier for the encoder family ("dense" or "conv").
        decoder_type: Identifier for the decoder family ("dense" or "conv").
        classifier_type: Identifier for the classifier family ("dense").
        monitor_metric: Validation metric name used for early stopping.
        use_contrastive: Whether to include the contrastive loss term.
        contrastive_weight: Scaling factor for the contrastive loss when enabled.
        prior_type: Type of prior distribution ("standard" | "mixture" | "vamp" | "geometric_mog").
            - "standard": Simple N(0,I) Gaussian prior
            - "mixture": Mixture of identical Gaussians with component-aware decoder
            - "vamp": Variational Mixture of Posteriors (learned pseudo-inputs, spatial separation)
            - "geometric_mog": Fixed geometric mixture (diagnostic tool, WARNING: induces topology)
        num_components: Number of mixture components when prior_type="mixture".
        kl_c_weight: Scaling factor applied to KL(q(c|x) || π) when mixture prior is active.
        dirichlet_alpha: Optional scalar prior strength for Dirichlet-MAP regularization on π.
        dirichlet_weight: Scaling applied to the Dirichlet-MAP penalty (no effect when alpha is None).
        component_diversity_weight: Component usage diversity regularization. Loss: λ × (-H[p̂_c])
            - NEGATIVE (e.g., -0.05): Encourage diversity - RECOMMENDED
            - POSITIVE: Discourage diversity (causes mode collapse)
        kl_c_anneal_epochs: If >0, linearly ramp kl_c_weight from 0 to its configured value across this many epochs.
        component_embedding_dim: Dimensionality of component embeddings (default: same as latent_dim).
            Small values (4-16) recommended to avoid overwhelming latent information.
        use_component_aware_decoder: If True, use component-aware decoder architecture that processes
            z and component embeddings separately (recommended for mixture prior).
        top_m_gating: If >0, compute reconstruction using only top-M components by responsibility.
            Reduces computation for large K. Default 0 means use all components.
        soft_embedding_warmup_epochs: If >0, use soft-weighted component embeddings for this many
            initial epochs before switching to hard sampling. Helps early training stability.
        use_tau_classifier: If True, use τ-based latent-only classification instead of separate
            classifier head (only applies to mixture prior).
        tau_smoothing_alpha: Laplace smoothing prior (α_0) for τ-classifier soft counts.
            Prevents zero probabilities for unseen component-label pairs.
        use_heteroscedastic_decoder: If True, use heteroscedastic decoder that learns per-image
            variance σ(x) for aleatoric uncertainty quantification. Decoder outputs (mean, sigma)
            tuple instead of just mean. Loss becomes: ||x - mean||²/(2σ²) + log σ.
        sigma_min: Minimum allowed standard deviation for heteroscedastic decoder (default: 0.05).
            Prevents variance collapse and ensures numerical stability.
        sigma_max: Maximum allowed standard deviation for heteroscedastic decoder (default: 0.5).
            Prevents variance explosion and keeps uncertainty estimates reasonable.
    """

    num_classes: int = 10
    latent_dim: int = 2
    hidden_dims: Tuple[int, ...] = (256, 128, 64)
    reconstruction_loss: str = "mse"
    recon_weight: float = 500.0
    kl_weight: float = 5
    learning_rate: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 300
    patience: int = 50
    val_split: float = 0.1
    random_seed: int = 42
    grad_clip_norm: float | None = 1.0
    weight_decay: float = 1e-4
    dropout_rate: float = 0.2
    label_weight: float = 0.0
    xla_flags: str | None = None
    input_hw: Tuple[int, int] | None = None
    encoder_type: str = "dense"
    decoder_type: str = "dense"
    classifier_type: str = "dense"
    monitor_metric: str = "classification_loss"
    use_contrastive: bool = False
    contrastive_weight: float = 0.0
    prior_type: str = "standard"
    num_components: int = 10
    kl_c_weight: float = 1.0
    dirichlet_alpha: float | None = None
    dirichlet_weight: float = 1.0
    learnable_pi: bool = False  # Learn mixture weights (mixture/geometric_mog only)
    component_diversity_weight: float = 0.0
    kl_c_anneal_epochs: int = 0
    mixture_history_log_every: int = 1  # Track π and usage every N epochs
    component_embedding_dim: int | None = None  # Defaults to latent_dim if None
    use_component_aware_decoder: bool = True  # Enable by default for mixture prior
    top_m_gating: int = 0  # 0 means use all components; >0 uses top-M
    soft_embedding_warmup_epochs: int = 0  # 0 means no warmup
    use_tau_classifier: bool = True  # Use τ-based classification (mixture prior only)
    tau_smoothing_alpha: float = 1.0  # Laplace smoothing prior (α_0)
    use_heteroscedastic_decoder: bool = False  # Learn per-image variance σ(x)
    sigma_min: float = 0.05  # Minimum allowed σ (prevents collapse)
    sigma_max: float = 0.5  # Maximum allowed σ (prevents explosion)

    # ═════════════════════════════════════════════════════════════════════════
    # VampPrior Configuration (prior_type="vamp")
    # ═════════════════════════════════════════════════════════════════════════
    vamp_num_samples_kl: int = 1  # Monte Carlo samples for KL estimation
    vamp_pseudo_lr_scale: float = 0.1  # Learning rate scale for pseudo-inputs (smaller than model LR)
    vamp_pseudo_init_method: str = "random"  # Pseudo-input initialization: "random" or "kmeans"
    
    # ═════════════════════════════════════════════════════════════════════════
    # Geometric MoG Configuration (prior_type="geometric_mog")
    # ═════════════════════════════════════════════════════════════════════════
    geometric_arrangement: str = "circle"  # Geometric arrangement: "circle" or "grid"
    geometric_radius: float = 2.0  # Radius for circle arrangement (distance from origin)

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_losses = {"mse", "bce"}
        if self.reconstruction_loss not in valid_losses:
            raise ValueError(
                f"reconstruction_loss must be one of {valid_losses}, "
                f"got '{self.reconstruction_loss}'"
            )
        if self.kl_c_anneal_epochs < 0:
            raise ValueError("kl_c_anneal_epochs must be >= 0")
        if self.dirichlet_alpha is not None and self.dirichlet_alpha <= 0.0:
            raise ValueError("dirichlet_alpha must be positive when provided")

        # Component-aware decoder defaults and validation
        if self.component_embedding_dim is None:
            self.component_embedding_dim = self.latent_dim
        if self.component_embedding_dim <= 0:
            raise ValueError("component_embedding_dim must be positive")
        if self.top_m_gating < 0:
            raise ValueError("top_m_gating must be >= 0")
        if self.top_m_gating > self.num_components:
            raise ValueError(f"top_m_gating ({self.top_m_gating}) cannot exceed num_components ({self.num_components})")
        if self.soft_embedding_warmup_epochs < 0:
            raise ValueError("soft_embedding_warmup_epochs must be >= 0")

        # τ-classifier validation
        # All mixture-based priors (mixture, vamp, geometric_mog) support τ-classifier
        mixture_based_priors = {"mixture", "vamp", "geometric_mog"}
        if self.use_tau_classifier and self.prior_type not in mixture_based_priors:
            import warnings
            warnings.warn(
                f"use_tau_classifier: true only applies to mixture-based priors {mixture_based_priors}. "
                "Falling back to standard classifier."
            )
            self.use_tau_classifier = False
        if self.use_tau_classifier and self.num_components < self.num_classes:
            raise ValueError(
                "num_components must be >= num_classes when use_tau_classifier=True "
                f"(got num_components={self.num_components}, num_classes={self.num_classes})"
            )
        if self.tau_smoothing_alpha <= 0:
            raise ValueError("tau_smoothing_alpha must be positive")

        # Heteroscedastic decoder validation
        if self.sigma_min <= 0:
            raise ValueError("sigma_min must be positive")
        if self.sigma_max <= self.sigma_min:
            raise ValueError(
                f"sigma_max ({self.sigma_max}) must be greater than "
                f"sigma_min ({self.sigma_min})"
            )

        # Learnable π validation
        if self.learnable_pi and self.prior_type not in ["mixture", "geometric_mog"]:
            import warnings
            warnings.warn(
                f"learnable_pi: true only applies to mixture and geometric_mog priors. "
                f"This setting will be ignored for prior_type: '{self.prior_type}'.",
                UserWarning
            )

        # Component-aware decoder validation
        mixture_based_priors = {"mixture", "vamp", "geometric_mog"}
        if self.use_component_aware_decoder and self.prior_type not in mixture_based_priors:
            import warnings
            warnings.warn(
                f"use_component_aware_decoder: true only applies to mixture-based priors {mixture_based_priors}. "
                f"Got prior_type: '{self.prior_type}'. Falling back to standard decoder.",
                UserWarning
            )
            # Note: Factory will handle fallback to standard decoder gracefully

        # VampPrior validation
        if self.vamp_num_samples_kl < 1:
            raise ValueError("vamp_num_samples_kl must be >= 1")
        if self.vamp_pseudo_lr_scale <= 0 or self.vamp_pseudo_lr_scale > 1.0:
            raise ValueError(
                f"vamp_pseudo_lr_scale must be in (0, 1], got {self.vamp_pseudo_lr_scale}"
            )
        valid_vamp_init_methods = {"random", "kmeans"}
        if self.vamp_pseudo_init_method not in valid_vamp_init_methods:
            raise ValueError(
                f"vamp_pseudo_init_method must be one of {valid_vamp_init_methods}, "
                f"got '{self.vamp_pseudo_init_method}'"
            )

        # Geometric MoG validation
        valid_arrangements = {"circle", "grid"}
        if self.geometric_arrangement not in valid_arrangements:
            raise ValueError(
                f"geometric_arrangement must be one of {valid_arrangements}, "
                f"got '{self.geometric_arrangement}'"
            )
        if self.geometric_radius <= 0:
            raise ValueError("geometric_radius must be positive")
        
        # Warn if using geometric_mog (topology concerns)
        if self.prior_type == "geometric_mog":
            import warnings
            warnings.warn(
                "geometric_mog prior induces artificial topology on latent space. "
                "Use only for diagnostic/curriculum purposes, not production models.",
                UserWarning
            )

    def get_informative_hyperparameters(self) -> Dict[str, object]:
        return {name: getattr(self, name) for name in INFORMATIVE_HPARAMETERS}

    def is_mixture_based_prior(self) -> bool:
        """Check if prior type uses mixture encoder (outputs component logits).
        
        Returns:
            True if prior_type is mixture, vamp, or geometric_mog
        """
        return self.prior_type in {"mixture", "vamp", "geometric_mog"}


def get_architecture_defaults(encoder_type: str) -> dict:
    try:
        return ARCHITECTURE_DEFAULTS[encoder_type]
    except KeyError as exc:
        raise ValueError(f"No defaults registered for encoder_type '{encoder_type}'") from exc
