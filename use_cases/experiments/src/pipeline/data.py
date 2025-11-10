"""Dataset loading utilities for experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from mnist.mnist import load_mnist_scaled


def prepare_data(data_config: Dict[str, int | float], *, rng_seed: int | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and subset MNIST according to the experiment configuration."""
    num_samples = int(data_config.get("num_samples", 5000))
    num_labeled = int(data_config.get("num_labeled", 50))
    seed = int(data_config.get("seed", rng_seed or 42))

    print(f"Loading MNIST: {num_samples} samples, {num_labeled} labeled...")
    X_train, y_train, X_test, y_test = load_mnist_scaled(reshape=True, hw=(28, 28))

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(X_train), size=num_samples, replace=False)
    X_subset = X_train[indices]
    y_subset = y_train[indices]

    # Semi-supervised labels
    y_semi = np.full(num_samples, np.nan, dtype=np.float32)
    labeled_indices = rng.choice(num_samples, size=num_labeled, replace=False)
    y_semi[labeled_indices] = y_subset[labeled_indices]

    print(f"  Train: {len(X_subset)} ({num_labeled} labeled)")
    return X_subset, y_semi, y_subset
