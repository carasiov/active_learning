from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler


def _ensure_dtype(x: np.ndarray, dtype=np.float32) -> np.ndarray:
    return x.astype(dtype, copy=False)


def load_mnist_all(*, normalize: bool = True, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
    """Load the full MNIST dataset (70k samples, flattened 784 features)."""
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = _ensure_dtype(X, dtype)
    if normalize:
        X = X / 255.0
    y = y.astype(np.int32)
    return X, y


def load_mnist_splits(
    *, normalize: bool = True, reshape: bool = False, hw: Tuple[int, int] = (28, 28), dtype=np.float32
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Return MNIST train/test splits."""
    X, y = load_mnist_all(normalize=normalize, dtype=dtype)
    x_train, y_train = X[:60000], y[:60000]
    x_test, y_test = X[60000:], y[60000:]
    if reshape:
        x_train = x_train.reshape(-1, *hw)
        x_test = x_test.reshape(-1, *hw)
    return (x_train, y_train), (x_test, y_test)


def load_mnist_scaled(
    *, reshape: bool = True, hw: Tuple[int, int] = (28, 28), dtype=np.float32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST, MinMax scale features, and reshape if requested."""
    (x_train, y_train), (x_test, y_test) = load_mnist_splits(normalize=True, reshape=False, dtype=dtype)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    if reshape:
        x_train = x_train.reshape(-1, *hw)
        x_test = x_test.reshape(-1, *hw)
    return x_train, y_train, x_test, y_test


def load_train_images_for_ssvae(*, dtype=np.float32) -> np.ndarray:
    """Return preprocessed train images for SSVAE training."""
    x_train, _, _, _ = load_mnist_scaled(reshape=True, hw=(28, 28), dtype=dtype)
    return x_train

