from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

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

CLI usage (scripts/train.py)
- Train conv:  `python scripts/train.py --encoder-type conv --decoder-type conv --latent-dim 2 --batch-size 512 --max-epochs 50`
- Train dense: `python scripts/train.py --encoder-type dense --decoder-type dense`

Programmatic usage
- from ssvae import SSVAE
  config = SSVAEConfig(encoder_type="conv", decoder_type="conv", latent_dim=2)
  vae = SSVAE(input_dim=(28, 28), config=config)
  vae.fit(x, labels, weights_path)
"""


@dataclass
class SSVAEConfig:
    """Hyperparameters controlling the SSVAE architecture and training loop.

    Attributes:
        latent_dim: Dimensionality of the latent representation.
        hidden_dims: Dense layer sizes for the encoder; decoder mirrors in reverse (dense only).
        recon_weight: Weight applied to the reconstruction MSE term.
        kl_weight: Scaling factor for the KL divergence regularizer.
        learning_rate: Optimizer learning rate.
        batch_size: Number of samples per training batch.
        max_epochs: Maximum number of training epochs.
        patience: Early stopping patience measured in epochs without validation improvement.
        val_split: Fraction of the dataset reserved for validation.
        random_seed: Base random seed used for parameter initialization and shuffling.
        grad_clip_norm: Global norm threshold for gradient clipping; disabled when ``None``.
        weight_decay: L2-style weight decay applied through the optimizer.
        label_weight: (Unused today) scaling factor for the classification loss term.
        input_hw: Optional (height, width) tuple for decoder output; defaults to the model input.
        encoder_type: Identifier for the encoder family ("dense" or "conv").
        decoder_type: Identifier for the decoder family ("dense" or "conv").
        classifier_type: Identifier for the classifier family ("dense").
        use_contrastive: Whether to include the contrastive loss term.
        contrastive_weight: Scaling factor for the contrastive loss when enabled.
    """

    latent_dim: int = 2
    hidden_dims: Tuple[int, ...] = (128, 64, 32, 16)
    recon_weight: float = 1000.0
    kl_weight: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 4 * 1024
    max_epochs: int = 200
    patience: int = 20
    val_split: float = 0.1
    random_seed: int = 42
    grad_clip_norm: float | None = 1.0
    weight_decay: float = 0.0
    label_weight: float = 1000.0
    input_hw: Tuple[int, int] | None = None
    encoder_type: str = "dense"
    decoder_type: str = "dense"
    classifier_type: str = "dense"
    use_contrastive: bool = False
    contrastive_weight: float = 0.0
