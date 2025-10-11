#!/usr/bin/env python3
"""Quick test that convolutional encoder/decoder work.

For local CPU-only runs, force JAX to use the CPU backend even if
CUDA plugins are installed via Poetry extras.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler

os.environ.setdefault("JAX_PLATFORMS", "cpu")
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from configs.base import SSVAEConfig  # noqa: E402
from ssvae import SSVAE  # noqa: E402


def main() -> None:
    print("Loading MNIST subset...", flush=True)
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X[:5000].astype(np.float32)
    y = y[:5000].astype(np.int32)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X).reshape(-1, 28, 28)
    X_binary = np.where(X_scaled > 0.5, 1.0, 0.0).astype(np.float32)

    labels = np.full(X_binary.shape[0], np.nan, dtype=np.float32)
    rng = np.random.default_rng(42)
    for digit in range(10):
        digit_idx = np.where(y == digit)[0]
        if len(digit_idx) >= 5:
            chosen = rng.choice(digit_idx, size=5, replace=False)
            labels[chosen] = y[chosen].astype(np.float32)

    print(f"Labeled samples: {np.sum(~np.isnan(labels))}", flush=True)

    print("\nCreating Conv SSVAE...", flush=True)
    config = SSVAEConfig(
        encoder_type="conv",
        decoder_type="conv",
        latent_dim=2,
        max_epochs=5,
        batch_size=512,
        patience=3,
    )
    vae = SSVAE(input_dim=(28, 28), config=config)
    print("✓ Conv SSVAE created successfully", flush=True)

    print("\nTraining for 5 epochs...", flush=True)
    artifact_dir = BASE_DIR / "artifacts" / "conv_test"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = str(artifact_dir / "test_conv.ckpt")

    history = vae.fit(X_binary, labels, weights_path=weights_path)
    print("✓ Training completed", flush=True)
    print(f"  Final loss: {history['loss'][-1]:.4f}")
    print(f"  Final reconstruction loss: {history['reconstruction_loss'][-1]:.4f}")

    print("\nRunning inference...", flush=True)
    latent, recon, pred_class, pred_certainty = vae.predict(X_binary)
    print("✓ Inference completed", flush=True)
    print(f"  Latent shape: {latent.shape}")
    print(f"  Reconstruction shape: {recon.shape}")
    print(f"  Predictions shape: {pred_class.shape}")

    assert latent.shape == (5000, 2), f"Expected latent shape (5000, 2), got {latent.shape}"
    assert recon.shape == (5000, 28, 28), f"Expected recon shape (5000, 28, 28), got {recon.shape}"
    assert pred_class.shape == (5000,), f"Expected pred_class shape (5000,), got {pred_class.shape}"

    final_recon_loss = history["reconstruction_loss"][-1]
    assert final_recon_loss < 1000, f"Reconstruction loss too high: {final_recon_loss}"

    print("\n" + "=" * 50)
    print("SUCCESS: Conv encoder/decoder working correctly!")
    print("=" * 50)
    print(f"\nArtifacts saved to: {artifact_dir.resolve()}")


if __name__ == "__main__":
    main()
