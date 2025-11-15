"""Dataset loading utilities for experiments."""
from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np

from mnist.mnist import load_mnist_scaled


def prepare_data(
    data_config: Dict[str, int | float],
    *,
    rng_seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load and subset MNIST according to the experiment configuration.

    Silent loading; dataset info displayed in experiment header.
    """
    num_samples = int(data_config.get("num_samples", 5000))
    num_labeled = int(data_config.get("num_labeled", 50))
    seed = int(data_config.get("seed", rng_seed or 42))
    dataset_variant = str(data_config.get("dataset_variant", "mnist")).lower()
    prefer_digits = dataset_variant in {"digits", "mnist_digits", "fallback"}

    # Load MNIST silently (progress shown in experiment header)
    X_train, y_train, X_test, y_test, dataset_label = load_mnist_scaled(
        reshape=True,
        hw=(28, 28),
        prefer_digits_fallback=prefer_digits,
    )

    total_available = len(X_train)
    replace = False
    if num_samples > total_available:
        warnings.warn(
            f"Requested num_samples={num_samples} but only {total_available} images available; "
            "sampling with replacement.",
            RuntimeWarning,
        )
        replace = True

    rng = np.random.RandomState(seed)
    indices = rng.choice(total_available, size=num_samples, replace=replace)
    X_subset = X_train[indices]
    y_subset = y_train[indices]

    # Semi-supervised labels
    y_semi = np.full(num_samples, np.nan, dtype=np.float32)
    labeled_pool = min(num_samples, num_labeled)
    if labeled_pool < num_labeled:
        warnings.warn(
            f"Requested num_labeled={num_labeled} exceeds subset size {num_samples}; "
            f"using {labeled_pool} labeled samples instead.",
            RuntimeWarning,
        )
    labeled_indices = rng.choice(num_samples, size=labeled_pool, replace=False)
    y_semi[labeled_indices] = y_subset[labeled_indices]

    return X_subset, y_semi, y_subset, dataset_label
