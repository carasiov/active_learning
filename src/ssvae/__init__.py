"""
SSVAE: Semi-Supervised Variational Autoencoder

Public API:
    SSVAE: Main interface (fit/predict/load_model_weights)
    SSVAEConfig: Configuration dataclass

Quick Start:
    >>> config = SSVAEConfig(latent_dim=2, max_epochs=50)
    >>> vae = SSVAE(input_dim=(28, 28), config=config)
    >>> history = vae.fit(data, labels, "model.ckpt")
    >>> z, recon, pred, cert = vae.predict(test_data)

See docs/IMPLEMENTATION.md for architecture details and extension guides.
"""

from .config import SSVAEConfig
from .models_refactored import SSVAE

__all__ = ["SSVAE", "SSVAEConfig"]
