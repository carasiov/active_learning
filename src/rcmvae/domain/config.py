from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import warnings

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
- from rcmvae.domain import SSVAE
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
    "latent_layout",
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
    "l1_weight",
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
    "decoder_conditioning",
    "top_m_gating",
    "soft_embedding_warmup_epochs",
    "c_regularizer",
    "c_logit_prior_weight",
    "c_logit_prior_mean",
    "c_logit_prior_sigma",
    "use_tau_classifier",
    "tau_smoothing_alpha",
    "use_heteroscedastic_decoder",
    "sigma_min",
    "sigma_max",
    "use_gumbel_softmax",
    "gumbel_temperature",
    "use_straight_through_gumbel",
    "vamp_num_samples_kl",
    "vamp_pseudo_lr_scale",
    "vamp_pseudo_init_method",
    "geometric_arrangement",
    "geometric_radius",
    "curriculum_enabled",
    "curriculum_start_k_active",
    "curriculum_unlock_every_epochs",
    "curriculum_max_k_active",
    "curriculum_unlock_mode",
    "curriculum_plateau_window_epochs",
    "curriculum_plateau_min_improvement",
    "curriculum_normality_threshold",
    "curriculum_min_epochs_per_channel",
)

@dataclass
class SSVAEConfig:
    """Hyperparameters controlling the SSVAE architecture and training loop.

    Attributes:
        num_classes: Number of output classes for the classifier head.
        latent_dim: Dimensionality of the latent representation.
        latent_layout: Arrangement of latents when using mixture-style priors.
            - "shared": single latent vector shared by all components (legacy behavior)
            - "decentralized": one latent vector per component (Mixture of VAEs)
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
        l1_weight: L1 regularization strength applied via loss (masked like weight_decay).
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
        decoder_conditioning: Conditioning method for component-aware decoders.
            Options: "cin" (Conditional Instance Norm), "film", "concat", "none".
            Requires mixture or geometric_mog prior. See architecture.md for details.
        use_component_aware_decoder: DEPRECATED. Has no effect. Use decoder_conditioning instead.
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
        use_gumbel_softmax: Sample discrete components with Gumbel-Softmax (mixture/geometric priors).
        gumbel_temperature: Temperature for Gumbel-Softmax routing.
        use_straight_through_gumbel: Use straight-through one-hot for decoder selection.
    """

    num_classes: int = 10
    latent_dim: int = 2
    latent_layout: str = "shared"
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
    l1_weight: float = 0.0
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
    use_component_aware_decoder: bool = False  # DEPRECATED: use decoder_conditioning instead
    decoder_conditioning: str = "none"  # Conditioning method: "cin" (Conditional Instance Norm), "film", "concat", "none"
    top_m_gating: int = 0  # 0 means use all components; >0 uses top-M
    soft_embedding_warmup_epochs: int = 0  # 0 means no warmup
    c_regularizer: str = "categorical"  # {"categorical", "logit_mog", "both"} — per-sample prior on q(c|x)
    c_logit_prior_weight: float = 0.0  # Strength of logistic-normal mixture regularizer on component logits
    c_logit_prior_mean: float = 5.0  # Mean magnitude M for the per-axis Gaussian components in logit space
    c_logit_prior_sigma: float = 1.0  # Isotropic sigma for the Gaussian components in logit space
    use_tau_classifier: bool = False  # Opt-in τ-classifier for mixture-based priors
    tau_smoothing_alpha: float = 1.0  # Laplace smoothing prior (α_0)
    use_heteroscedastic_decoder: bool = False  # Learn per-image variance σ(x)
    sigma_min: float = 0.05  # Minimum allowed σ (prevents collapse)
    sigma_max: float = 0.5  # Maximum allowed σ (prevents explosion)
    use_gumbel_softmax: bool = False  # Sample c via Gumbel-Softmax when decentralized latents are active
    gumbel_temperature: float = 1.0  # Initial temperature for Gumbel-Softmax
    gumbel_temperature_min: float = 0.5  # Minimum temperature after annealing
    gumbel_temperature_decay: float = 0.0  # Decay rate per epoch (exponential) or linear step (if > 1, maybe epochs?)
    # Let's use a simple linear annealing over N epochs for consistency with kl_c_anneal_epochs
    gumbel_temperature_anneal_epochs: int = 0  # If >0, anneal from initial to min over this many epochs
    use_straight_through_gumbel: bool = True  # Use straight-through one-hot for decoder selection

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

    # ═════════════════════════════════════════════════════════════════════════
    # Channel Unlocking Curriculum (mixture/geometric_mog/vamp priors)
    # ═════════════════════════════════════════════════════════════════════════
    curriculum_enabled: bool = False  # Enable channel unlocking curriculum
    curriculum_start_k_active: int = 1  # Number of active channels at start (must be >= 1)
    curriculum_unlock_every_epochs: int = 5  # Unlock one additional channel every N epochs
    curriculum_max_k_active: int | None = None  # Max active channels (None = num_components)

    # Migration window (post-unlock softening for example migration)
    curriculum_migration_epochs: int = 0  # Number of epochs after unlock to soften routing (0 = disabled)
    curriculum_soft_routing_during_migration: bool = True  # Disable straight-through during migration
    curriculum_temp_boost_during_migration: float = 1.0  # Multiply gumbel_temperature by this during migration (>1 softens)
    curriculum_logit_mog_scale_during_migration: float = 1.0  # Scale logit_mog weight by this during migration (<1 reduces peakiness pressure)

    # Trigger-based unlock (alternative to epoch-based)
    curriculum_unlock_mode: str = "epoch"  # "epoch" = unlock every N epochs; "trigger" = unlock when plateau + normality
    curriculum_plateau_window_epochs: int = 5  # Window size for plateau detection (recon loss must stagnate for this many epochs)
    curriculum_plateau_min_improvement: float = 0.01  # Minimum relative improvement required to NOT be a plateau
    curriculum_plateau_metric: str = "reconstruction_loss"  # Metric for plateau detection ("reconstruction_loss" or "loss")
    curriculum_normality_threshold: float = 1.0  # Max normality score to allow unlock (lower = stricter)
    curriculum_min_epochs_per_channel: int = 0  # Minimum epochs at each k_active before unlock (0 = no constraint)

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
            warnings.warn(
                "dirichlet_alpha <= 0 disables the Dirichlet prior. "
                "If you intended to turn it off, leave it as None.",
                RuntimeWarning,
            )
            self.dirichlet_alpha = None

        # Deprecation warning for use_component_aware_decoder
        if self.use_component_aware_decoder:
            warnings.warn(
                "use_component_aware_decoder is deprecated and has no effect. "
                "Use decoder_conditioning instead (options: 'cin', 'film', 'concat', 'none').",
                DeprecationWarning,
                stacklevel=2,
            )

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

        # Component regularizer mode and parameters
        valid_c_regularizers = {"categorical", "logit_mog", "both"}
        if self.c_regularizer not in valid_c_regularizers:
            raise ValueError(
                f"c_regularizer must be one of {valid_c_regularizers}, got '{self.c_regularizer}'."
            )
        if self.c_logit_prior_weight < 0:
            raise ValueError("c_logit_prior_weight must be non-negative")
        if self.c_logit_prior_mean <= 0:
            raise ValueError("c_logit_prior_mean must be positive")
        if self.c_logit_prior_sigma <= 0:
            raise ValueError("c_logit_prior_sigma must be positive")

        # Decoder conditioning validation
        valid_conditioning = {"cin", "film", "concat", "none"}
        if self.decoder_conditioning not in valid_conditioning:
            raise ValueError(
                f"decoder_conditioning must be one of {valid_conditioning}, "
                f"got '{self.decoder_conditioning}'"
            )
        mixture_condition_priors = {"mixture", "geometric_mog"}
        uses_conditioning = self.decoder_conditioning in {"cin", "film", "concat"}
        if uses_conditioning and self.prior_type not in mixture_condition_priors:
            if self.prior_type == "vamp":
                warnings.warn(
                    f"VampPrior does not supply component embeddings; "
                    f"decoder_conditioning='{self.decoder_conditioning}' will be ignored (using 'none').",
                    UserWarning,
                )
            else:
                raise ValueError(
                    f"decoder_conditioning='{self.decoder_conditioning}' requires mixture-like priors "
                    f"{mixture_condition_priors}; got prior_type='{self.prior_type}'."
                )

        # τ-classifier validation
        mixture_based_priors = {"mixture", "vamp", "geometric_mog"}
        if self.use_tau_classifier and self.prior_type not in mixture_based_priors:
            warnings.warn(
                f"use_tau_classifier: true only applies to mixture-based priors {mixture_based_priors}. ",
                RuntimeWarning,
            )
        if self.use_tau_classifier:
            if self.prior_type not in mixture_based_priors:
                raise ValueError("τ-classifier requires mixture-based priors that emit component responsibilities.")
            if self.num_components < self.num_classes:
                raise ValueError(
                    "num_components must be >= num_classes when use_tau_classifier=True for mixture-based priors. "
                    f"Got num_components={self.num_components}, num_classes={self.num_classes}."
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
            warnings.warn(
                f"learnable_pi is not supported for prior_type='{self.prior_type}'; disabling learnable_pi.",
                UserWarning,
            )
            self.learnable_pi = False

        # VampPrior validation
        if self.vamp_num_samples_kl < 1:
            raise ValueError("vamp_num_samples_kl must be >= 1.")
        if self.vamp_pseudo_lr_scale <= 0 or self.vamp_pseudo_lr_scale > 1.0:
            raise ValueError(
                "vamp_pseudo_lr_scale must be in (0, 1]. "
                f"Got {self.vamp_pseudo_lr_scale}."
            )
        valid_vamp_init_methods = {"random", "kmeans"}
        if self.vamp_pseudo_init_method not in valid_vamp_init_methods:
            warnings.warn(
                f"vamp_pseudo_init_method should be one of {valid_vamp_init_methods}. "
                f"Got '{self.vamp_pseudo_init_method}'. Downstream code may raise.",
                RuntimeWarning,
            )

        # Geometric MoG validation
        if self.prior_type == "geometric_mog":
            valid_arrangements = {"circle", "grid"}
            if self.geometric_arrangement not in valid_arrangements:
                raise ValueError(
                    f"Geometric MoG requires geometric_arrangement in {valid_arrangements}. "
                    f"Got '{self.geometric_arrangement}'."
                )
        if self.geometric_radius <= 0:
            raise ValueError("geometric_radius must be positive")
        
        # Warn if using geometric_mog (topology concerns)
        if self.prior_type == "geometric_mog":
            warnings.warn(
                "geometric_mog prior induces artificial topology on latent space. "
                "Use only for diagnostic/curriculum purposes, not production models.",
                UserWarning
            )
        if self.latent_layout not in {"shared", "decentralized"}:
            raise ValueError(
                f"latent_layout must be 'shared' or 'decentralized', got '{self.latent_layout}'."
            )
        if self.latent_layout == "decentralized" and self.prior_type not in {"mixture", "geometric_mog", "vamp"}:
            raise ValueError("latent_layout='decentralized' requires a mixture/geometric/vamp prior.")
        if self.latent_layout == "decentralized" and self.num_components <= 1:
            raise ValueError(
                f"latent_layout='decentralized' requires num_components > 1, got {self.num_components}."
            )
        if self.gumbel_temperature <= 0:
            raise ValueError("gumbel_temperature must be positive")
        if self.use_gumbel_softmax and self.prior_type not in {"mixture", "geometric_mog", "vamp"}:
            raise ValueError("use_gumbel_softmax requires a mixture/geometric/vamp prior.")
        if self.use_gumbel_softmax and self.gumbel_temperature < self.gumbel_temperature_min:
            raise ValueError(
                f"gumbel_temperature ({self.gumbel_temperature}) must be >= gumbel_temperature_min ({self.gumbel_temperature_min})."
            )

        # Channel Unlocking Curriculum validation
        if self.curriculum_max_k_active is None:
            self.curriculum_max_k_active = self.num_components
        if self.curriculum_enabled:
            mixture_based_priors = {"mixture", "vamp", "geometric_mog"}
            if self.prior_type not in mixture_based_priors:
                raise ValueError(
                    f"curriculum_enabled requires mixture-based priors {mixture_based_priors}. "
                    f"Got prior_type='{self.prior_type}'."
                )
            if self.curriculum_start_k_active < 1:
                raise ValueError("curriculum_start_k_active must be >= 1")
            if self.curriculum_unlock_every_epochs < 1:
                raise ValueError("curriculum_unlock_every_epochs must be >= 1")
            if self.curriculum_max_k_active > self.num_components:
                raise ValueError(
                    f"curriculum_max_k_active ({self.curriculum_max_k_active}) cannot exceed "
                    f"num_components ({self.num_components})."
                )
            if self.curriculum_start_k_active > self.curriculum_max_k_active:
                raise ValueError(
                    f"curriculum_start_k_active ({self.curriculum_start_k_active}) cannot exceed "
                    f"curriculum_max_k_active ({self.curriculum_max_k_active})."
                )
            if self.curriculum_migration_epochs < 0:
                raise ValueError("curriculum_migration_epochs must be >= 0")
            if self.curriculum_temp_boost_during_migration <= 0:
                raise ValueError("curriculum_temp_boost_during_migration must be positive")
            if self.curriculum_logit_mog_scale_during_migration < 0:
                raise ValueError("curriculum_logit_mog_scale_during_migration must be >= 0")
            # Trigger-based unlock validation
            valid_unlock_modes = {"epoch", "trigger"}
            if self.curriculum_unlock_mode not in valid_unlock_modes:
                raise ValueError(
                    f"curriculum_unlock_mode must be one of {valid_unlock_modes}, "
                    f"got '{self.curriculum_unlock_mode}'."
                )
            if self.curriculum_plateau_window_epochs < 1:
                raise ValueError("curriculum_plateau_window_epochs must be >= 1")
            if self.curriculum_plateau_min_improvement < 0:
                raise ValueError("curriculum_plateau_min_improvement must be >= 0")
            valid_plateau_metrics = {"reconstruction_loss", "loss"}
            if self.curriculum_plateau_metric not in valid_plateau_metrics:
                raise ValueError(
                    f"curriculum_plateau_metric must be one of {valid_plateau_metrics}, "
                    f"got '{self.curriculum_plateau_metric}'."
                )
            if self.curriculum_normality_threshold <= 0:
                raise ValueError("curriculum_normality_threshold must be positive")
            if self.curriculum_min_epochs_per_channel < 0:
                raise ValueError("curriculum_min_epochs_per_channel must be >= 0")

    def get_informative_hyperparameters(self) -> Dict[str, object]:
        return {name: getattr(self, name) for name in INFORMATIVE_HPARAMETERS}

    def is_mixture_based_prior(self) -> bool:
        """Check if prior type uses mixture encoder (outputs component logits).

        Returns:
            True if prior_type is mixture, vamp, or geometric_mog
        """
        return self.prior_type in {"mixture", "vamp", "geometric_mog"}

    def get_k_active(self, epoch: int) -> int:
        """Compute number of active channels for curriculum at given epoch.

        Args:
            epoch: Current epoch (0-indexed)

        Returns:
            Number of active channels. When curriculum is disabled, returns num_components.
        """
        if not self.curriculum_enabled:
            return self.num_components
        unlocks = epoch // self.curriculum_unlock_every_epochs
        k_active = self.curriculum_start_k_active + unlocks
        return min(k_active, self.curriculum_max_k_active)

    def get_epochs_since_last_unlock(self, epoch: int) -> int:
        """Get number of epochs since the most recent channel unlock.

        Args:
            epoch: Current epoch (0-indexed)

        Returns:
            Number of epochs since last unlock event. Returns epoch if curriculum disabled.
        """
        if not self.curriculum_enabled:
            return epoch
        return epoch % self.curriculum_unlock_every_epochs

    def is_in_migration_window(self, epoch: int) -> bool:
        """Check if current epoch is within the migration window after an unlock.

        The migration window spans from an unlock event (epoch where k_active increases)
        through the next (curriculum_migration_epochs - 1) epochs.

        Args:
            epoch: Current epoch (0-indexed)

        Returns:
            True if within migration window, False otherwise.
        """
        if not self.curriculum_enabled or self.curriculum_migration_epochs == 0:
            return False
        # Check if k_active has reached its max (no more unlocks possible)
        if self.get_k_active(epoch) >= self.curriculum_max_k_active:
            # If we're already at max and this isn't the epoch we just reached it,
            # no migration window applies
            prev_k = self.get_k_active(max(0, epoch - 1)) if epoch > 0 else self.curriculum_start_k_active
            if prev_k >= self.curriculum_max_k_active:
                return False
        epochs_since_unlock = self.get_epochs_since_last_unlock(epoch)
        return epochs_since_unlock < self.curriculum_migration_epochs

    def get_effective_gumbel_temperature(self, epoch: int) -> float:
        """Get Gumbel temperature for current epoch, with migration window boost.

        Args:
            epoch: Current epoch (0-indexed)

        Returns:
            Gumbel temperature, potentially boosted during migration window.
        """
        base_temp = self.gumbel_temperature
        # Apply annealing if configured
        if self.gumbel_temperature_anneal_epochs > 0 and epoch > 0:
            progress = min(1.0, epoch / self.gumbel_temperature_anneal_epochs)
            base_temp = self.gumbel_temperature - progress * (
                self.gumbel_temperature - self.gumbel_temperature_min
            )
        # Apply migration boost if in window
        if self.is_in_migration_window(epoch):
            return base_temp * self.curriculum_temp_boost_during_migration
        return base_temp

    def get_effective_logit_mog_weight(self, epoch: int) -> float:
        """Get logit-MoG weight for current epoch, with migration window scaling.

        Args:
            epoch: Current epoch (0-indexed)

        Returns:
            Logit-MoG weight, potentially scaled down during migration window.
        """
        if self.is_in_migration_window(epoch):
            return self.c_logit_prior_weight * self.curriculum_logit_mog_scale_during_migration
        return self.c_logit_prior_weight

    def use_straight_through_for_epoch(self, epoch: int) -> bool:
        """Determine if straight-through should be used for current epoch.

        During migration window, straight-through can be disabled for soft routing.

        Args:
            epoch: Current epoch (0-indexed)

        Returns:
            True if straight-through should be used, False for soft routing.
        """
        if not self.use_straight_through_gumbel:
            return False
        # Disable ST during migration window if configured
        if self.is_in_migration_window(epoch) and self.curriculum_soft_routing_during_migration:
            return False
        return True


def get_architecture_defaults(encoder_type: str) -> dict:
    try:
        return ARCHITECTURE_DEFAULTS[encoder_type]
    except KeyError as exc:
        raise ValueError(f"No defaults registered for encoder_type '{encoder_type}'") from exc
