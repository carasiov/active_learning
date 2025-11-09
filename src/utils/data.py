"""Data loading utilities for SSVAE experiments."""

import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist(standardize=True, return_X_y=True):
    """
    Load MNIST dataset using scikit-learn.

    Args:
        standardize: If True, normalize pixel values to [0, 1]
        return_X_y: If True, return (X, y) tuple; else return dict

    Returns:
        If return_X_y=True:
            (x_train, y_train), (x_test, y_test)
            where x_train, x_test: (n_samples, 28, 28) arrays of pixel values
                  y_train, y_test: (n_samples,) arrays of integer labels 0-9

        If return_X_y=False:
            dict with keys: 'x_train', 'y_train', 'x_test', 'y_test'
    """
    # Fetch MNIST from OpenML
    mnist = fetch_openml('mnist_784', version=1, parser='auto')

    # Get data and labels (convert from DataFrame to numpy if needed)
    if hasattr(mnist.data, 'values'):
        X = mnist.data.values.astype(np.float32)
    else:
        X = np.array(mnist.data, dtype=np.float32)

    if hasattr(mnist.target, 'values'):
        y = mnist.target.values.astype(np.int32)
    else:
        y = np.array(mnist.target, dtype=np.int32)

    # Reshape to (n_samples, 28, 28)
    X = X.reshape(-1, 28, 28)

    # Normalize to [0, 1] if requested
    if standardize:
        X = X / 255.0

    # Standard MNIST train/test split: first 60K train, last 10K test
    x_train = X[:60000]
    y_train = y[:60000]
    x_test = X[60000:]
    y_test = y[60000:]

    if return_X_y:
        return (x_train, y_train), (x_test, y_test)
    else:
        return {
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test,
        }


def create_semisupervised_split(x, y, num_labeled, seed=42):
    """
    Create a semi-supervised split with limited labeled samples.

    Args:
        x: Data array (n_samples, ...)
        y: Label array (n_samples,)
        num_labeled: Number of labeled samples to keep
        seed: Random seed for reproducibility

    Returns:
        x_labeled: Labeled data (num_labeled, ...)
        y_labeled: Labeled targets (num_labeled,)
        x_unlabeled: Unlabeled data (remaining, ...)
    """
    rng = np.random.RandomState(seed)
    n_samples = len(x)

    # Random permutation
    indices = rng.permutation(n_samples)

    # Split
    labeled_idx = indices[:num_labeled]
    unlabeled_idx = indices[num_labeled:]

    x_labeled = x[labeled_idx]
    y_labeled = y[labeled_idx]
    x_unlabeled = x[unlabeled_idx]

    return x_labeled, y_labeled, x_unlabeled
