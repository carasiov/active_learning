from __future__ import annotations

from typing import Tuple

import numpy as np
import os
import warnings
from sklearn.datasets import fetch_openml, load_digits
from sklearn.preprocessing import MinMaxScaler


def _ensure_dtype(x: np.ndarray, dtype=np.float32) -> np.ndarray:
    return x.astype(dtype, copy=False)


def _load_digits_fallback(dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback using sklearn's builtin digits dataset (8x8)."""
    digits = load_digits()
    X = digits.data.astype(np.float32)
    # Normalize to [0,1]
    X /= X.max(initial=1.0)
    y = digits.target.astype(np.int32)

    # Pad into 28x28 canvas centered to roughly match MNIST shape
    upscaled = np.zeros((X.shape[0], 28, 28), dtype=np.float32)
    patch = X.reshape(-1, 8, 8)
    upscaled[:, 10:18, 10:18] = patch
    upscaled = upscaled.reshape(-1, 28 * 28)
    return upscaled.astype(dtype, copy=False), y


def load_mnist_all(
    *,
    normalize: bool = True,
    dtype=np.float32,
    prefer_digits_fallback: bool = False,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load MNIST (default) with automatic fallback to sklearn digits.

    Returns both the data and a label describing which source was used so that
    downstream summaries (reports/plots) can reflect the actual dataset.
    """

    if not prefer_digits_fallback:
        try:
            X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
            X = _ensure_dtype(X, dtype)
            if normalize:
                X = X / 255.0
            y = y.astype(np.int32)
            return X, y, "MNIST (70k)"
        except Exception as exc:  # pragma: no cover - exercised in offline CI
            warnings.warn(
                f"fetch_openml('mnist_784') failed ({exc}). Falling back to sklearn.load_digits(). \
Digits dataset is smaller (8x8) but enables offline tests.",
                RuntimeWarning,
            )

    X, y = _load_digits_fallback(dtype=dtype)
    return X, y, "Digits (8x8 fallback)"


def load_mnist_splits(
    *,
    normalize: bool = True,
    reshape: bool = False,
    hw: Tuple[int, int] = (28, 28),
    dtype=np.float32,
    prefer_digits_fallback: bool = False,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], str]:
    """Return MNIST train/test splits along with the source label."""
    X, y, source = load_mnist_all(
        normalize=normalize,
        dtype=dtype,
        prefer_digits_fallback=prefer_digits_fallback,
    )
    total = X.shape[0]
    if total < 2:
        raise ValueError("MNIST loader requires at least two samples.")

    if total >= 70000:
        split_idx = 60000
    else:
        split_idx = int(total * 0.8)
        split_idx = max(1, min(split_idx, total - 1))

    x_train, y_train = X[:split_idx], y[:split_idx]
    x_test, y_test = X[split_idx:], y[split_idx:]
    if reshape:
        x_train = x_train.reshape(-1, *hw)
        x_test = x_test.reshape(-1, *hw)
    return (x_train, y_train), (x_test, y_test), source


def load_mnist_scaled(
    *,
    reshape: bool = True,
    hw: Tuple[int, int] = (28, 28),
    dtype=np.float32,
    prefer_digits_fallback: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Load MNIST, MinMax scale features, and reshape if requested."""
    (x_train, y_train), (x_test, y_test), source = load_mnist_splits(
        normalize=True,
        reshape=False,
        dtype=dtype,
        prefer_digits_fallback=prefer_digits_fallback,
    )
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    if reshape:
        x_train = x_train.reshape(-1, *hw)
        x_test = x_test.reshape(-1, *hw)
    return x_train, y_train, x_test, y_test, source


def load_train_images_for_ssvae(*, dtype=np.float32, prefer_digits_fallback: bool = False) -> np.ndarray:
    """Return preprocessed train images for SSVAE training."""
    x_train, _, _, _, _ = load_mnist_scaled(
        reshape=True,
        hw=(28, 28),
        dtype=dtype,
        prefer_digits_fallback=prefer_digits_fallback,
    )
    return x_train
