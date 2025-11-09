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
    "component_diversity_weight",  # Renamed from usage_sparsity_weight
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
        prior_type: Type of prior distribution ("standard" | "mixture").
        num_components: Number of mixture components when prior_type="mixture".
        kl_c_weight: Scaling factor applied to KL(q(c|x) || π) when mixture prior is active.
        dirichlet_alpha: Optional scalar prior strength for Dirichlet-MAP regularization on π.
        dirichlet_weight: Scaling applied to the Dirichlet-MAP penalty (no effect when alpha is None).
        component_diversity_weight: Component usage diversity regularization. Loss: λ × (-H[p̂_c])
            - NEGATIVE (e.g., -0.05): Encourage diversity - RECOMMENDED
            - POSITIVE: Discourage diversity (causes mode collapse)
        kl_c_anneal_epochs: If >0, linearly ramp kl_c_weight from 0 to its configured value across this many epochs.
        component_kl_weight: Deprecated alias for kl_c_weight kept for backward compatibility.
        usage_sparsity_weight: Deprecated alias for component_diversity_weight.
        component_embedding_dim: Dimensionality of component embeddings (default: same as latent_dim).
            Small values (4-16) recommended to avoid overwhelming latent information.
        use_component_aware_decoder: If True, use component-aware decoder architecture that processes
            z and component embeddings separately (recommended for mixture prior).
        top_m_gating: If >0, compute reconstruction using only top-M components by responsibility.
            Reduces computation for large K. Default 0 means use all components.
        soft_embedding_warmup_epochs: If >0, use soft-weighted component embeddings for this many
            initial epochs before switching to hard sampling. Helps early training stability.
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
    component_kl_weight: float | None = None
    kl_c_weight: float = 1.0
    dirichlet_alpha: float | None = None
    dirichlet_weight: float = 1.0
    component_diversity_weight: float = 0.0  # Primary name (negative = diversity reward)
    usage_sparsity_weight: float | None = None  # Deprecated alias
    kl_c_anneal_epochs: int = 0
    mixture_history_log_every: int = 1  # Track π and usage every N epochs
    component_embedding_dim: int | None = None  # Defaults to latent_dim if None
    use_component_aware_decoder: bool = True  # Enable by default for mixture prior
    top_m_gating: int = 0  # 0 means use all components; >0 uses top-M
    soft_embedding_warmup_epochs: int = 0  # 0 means no warmup

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_losses = {"mse", "bce"}
        if self.reconstruction_loss not in valid_losses:
            raise ValueError(
                f"reconstruction_loss must be one of {valid_losses}, "
                f"got '{self.reconstruction_loss}'"
            )
        # Backward compatibility: if legacy component_kl_weight is provided and
        # kl_c_weight wasn't explicitly set, adopt the legacy value.
        if self.component_kl_weight is not None:
            default_kl_c = SSVAEConfig.__dataclass_fields__["kl_c_weight"].default
            if self.kl_c_weight == default_kl_c:
                self.kl_c_weight = float(self.component_kl_weight)
        # Mirror into legacy field for any downstream code still reading it.
        self.component_kl_weight = float(self.kl_c_weight)

        # Backward compatibility: usage_sparsity_weight → component_diversity_weight
        if self.usage_sparsity_weight is not None:
            default_diversity = SSVAEConfig.__dataclass_fields__["component_diversity_weight"].default
            if self.component_diversity_weight == default_diversity:
                self.component_diversity_weight = float(self.usage_sparsity_weight)
                import warnings
                warnings.warn(
                    "Parameter 'usage_sparsity_weight' is deprecated. Use 'component_diversity_weight'.",
                    DeprecationWarning,
                    stacklevel=2
                )
        self.usage_sparsity_weight = float(self.component_diversity_weight)

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

    def get_informative_hyperparameters(self) -> Dict[str, object]:
        return {name: getattr(self, name) for name in INFORMATIVE_HPARAMETERS}


def get_architecture_defaults(encoder_type: str) -> dict:
    try:
        return ARCHITECTURE_DEFAULTS[encoder_type]
    except KeyError as exc:
        raise ValueError(f"No defaults registered for encoder_type '{encoder_type}'") from exc
