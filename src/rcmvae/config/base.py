"""Core configuration dataclasses for network, training, loss, and decoder features.

These configs replace the scattered parameters from SSVAEConfig, grouping related
settings by their domain concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class NetworkConfig:
    """Network architecture configuration.

    Attributes:
        num_classes: Number of output classes for the classifier head.
        latent_dim: Dimensionality of the latent representation.
        hidden_dims: Dense layer sizes for encoder (decoder mirrors in reverse for dense).
        encoder_type: Encoder family identifier ("dense" or "conv").
        decoder_type: Decoder family identifier ("dense" or "conv").
        classifier_type: Classifier family identifier (always "dense").
        input_hw: Optional (height, width) tuple for decoder output; defaults to model input.
        dropout_rate: Dropout applied inside the classifier network.
    """

    num_classes: int = 10
    latent_dim: int = 2
    hidden_dims: Tuple[int, ...] = (256, 128, 64)
    encoder_type: str = "dense"
    decoder_type: str = "dense"
    classifier_type: str = "dense"
    input_hw: Tuple[int, int] | None = None
    dropout_rate: float = 0.2

    def __post_init__(self):
        """Validate network configuration."""
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")
        if not self.hidden_dims:
            raise ValueError("hidden_dims cannot be empty")
        if any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("All hidden_dims must be positive")
        if self.dropout_rate < 0 or self.dropout_rate > 1:
            raise ValueError(f"dropout_rate must be in [0, 1], got {self.dropout_rate}")

        valid_encoder_types = {"dense", "conv"}
        if self.encoder_type not in valid_encoder_types:
            raise ValueError(
                f"encoder_type must be one of {valid_encoder_types}, got '{self.encoder_type}'"
            )

        valid_decoder_types = {"dense", "conv"}
        if self.decoder_type not in valid_decoder_types:
            raise ValueError(
                f"decoder_type must be one of {valid_decoder_types}, got '{self.decoder_type}'"
            )


@dataclass
class TrainingConfig:
    """Training loop hyperparameters.

    Attributes:
        learning_rate: Optimizer learning rate.
        batch_size: Number of samples per training batch.
        max_epochs: Maximum number of training epochs.
        patience: Early stopping patience measured in epochs without validation improvement.
        val_split: Fraction of dataset reserved for validation.
        random_seed: Base random seed for parameter initialization and shuffling.
        grad_clip_norm: Global norm threshold for gradient clipping; disabled when None.
        weight_decay: L2-style weight decay applied through the optimizer.
        monitor_metric: Validation metric name used for early stopping.
    """

    learning_rate: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 300
    patience: int = 50
    val_split: float = 0.1
    random_seed: int = 42
    grad_clip_norm: float | None = 1.0
    weight_decay: float = 1e-4
    monitor_metric: str = "classification_loss"

    def __post_init__(self):
        """Validate training configuration."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {self.max_epochs}")
        if self.patience < 0:
            raise ValueError(f"patience must be non-negative, got {self.patience}")
        if self.val_split < 0 or self.val_split >= 1:
            raise ValueError(f"val_split must be in [0, 1), got {self.val_split}")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError(f"grad_clip_norm must be positive or None, got {self.grad_clip_norm}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")


@dataclass
class LossConfig:
    """Loss function weights and configuration.

    Attributes:
        reconstruction_loss: Loss function for reconstruction term.
            - "mse": Mean squared error (Gaussian pixel model). Default weight: 500.
            - "bce": Binary cross-entropy (Bernoulli pixel model). Default weight: 1.0.
        recon_weight: Weight applied to the reconstruction term.
        kl_weight: Scaling factor for KL divergence regularizer.
        label_weight: Scaling factor for classification loss (currently unused).
        use_contrastive: Whether to include contrastive loss term.
        contrastive_weight: Scaling factor for contrastive loss when enabled.
    """

    reconstruction_loss: str = "mse"
    recon_weight: float = 500.0
    kl_weight: float = 5.0
    label_weight: float = 0.0
    use_contrastive: bool = False
    contrastive_weight: float = 0.0

    def __post_init__(self):
        """Validate loss configuration."""
        valid_losses = {"mse", "bce"}
        if self.reconstruction_loss not in valid_losses:
            raise ValueError(
                f"reconstruction_loss must be one of {valid_losses}, got '{self.reconstruction_loss}'"
            )
        if self.recon_weight < 0:
            raise ValueError(f"recon_weight must be non-negative, got {self.recon_weight}")
        if self.kl_weight < 0:
            raise ValueError(f"kl_weight must be non-negative, got {self.kl_weight}")
        if self.contrastive_weight < 0:
            raise ValueError(f"contrastive_weight must be non-negative, got {self.contrastive_weight}")


@dataclass
class DecoderFeatures:
    """Decoder feature flags and parameters.

    Attributes:
        use_heteroscedastic_decoder: If True, decoder learns per-image variance σ(x)
            for aleatoric uncertainty quantification.
        sigma_min: Minimum allowed standard deviation for heteroscedastic decoder.
        sigma_max: Maximum allowed standard deviation for heteroscedastic decoder.
        use_component_aware_decoder: If True, use component-aware decoder that processes
            z and component embeddings separately (applies to mixture-based priors).
        component_embedding_dim: Dimensionality of component embeddings (defaults to latent_dim).
        top_m_gating: If >0, use only top-M components by responsibility for reconstruction.
        soft_embedding_warmup_epochs: If >0, use soft-weighted embeddings before hard sampling.
    """

    use_heteroscedastic_decoder: bool = False
    sigma_min: float = 0.05
    sigma_max: float = 0.5
    use_component_aware_decoder: bool = True
    component_embedding_dim: int | None = None
    top_m_gating: int = 0
    soft_embedding_warmup_epochs: int = 0

    def __post_init__(self):
        """Validate decoder features."""
        if self.sigma_min <= 0:
            raise ValueError(f"sigma_min must be positive, got {self.sigma_min}")
        if self.sigma_max <= self.sigma_min:
            raise ValueError(
                f"sigma_max ({self.sigma_max}) must be greater than sigma_min ({self.sigma_min})"
            )
        if self.component_embedding_dim is not None and self.component_embedding_dim <= 0:
            raise ValueError(f"component_embedding_dim must be positive, got {self.component_embedding_dim}")
        if self.top_m_gating < 0:
            raise ValueError(f"top_m_gating must be non-negative, got {self.top_m_gating}")
        if self.soft_embedding_warmup_epochs < 0:
            raise ValueError(f"soft_embedding_warmup_epochs must be non-negative, got {self.soft_embedding_warmup_epochs}")
