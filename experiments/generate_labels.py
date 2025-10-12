#!/usr/bin/env python3
"""Generate balanced MNIST label files for semi-supervised experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LABEL_DIR = BASE_DIR / "experiments" / "label_sets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a balanced label CSV for the MNIST train split."
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        required=True,
        help="Total number of labeled examples to sample (distributed as evenly as possible across digits).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination CSV path (defaults to experiments/label_sets/labels_<num>.csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the NumPy random number generator.",
    )
    return parser.parse_args()


def load_mnist_labels() -> np.ndarray:
    _, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    return y[:60000].astype(np.int32)


def sample_balanced_indices(labels: np.ndarray, num_labels: int, seed: int) -> np.ndarray:
    if num_labels <= 0:
        raise ValueError("--num-labels must be a positive integer.")

    base_per_digit = num_labels // 10
    remainder = num_labels % 10
    rng = np.random.default_rng(seed)

    selections = []
    for digit in range(10):
        digit_indices = np.where(labels == digit)[0]
        quota = base_per_digit + (1 if digit < remainder else 0)
        if quota == 0:
            continue
        if digit_indices.size < quota:
            raise RuntimeError(f"Not enough samples available for digit {digit}.")
        chosen = rng.choice(digit_indices, size=quota, replace=False)
        selections.append(chosen)
    return np.sort(np.concatenate(selections))


def write_labels(indices: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
    df = pd.DataFrame({"label": labels[indices]}, index=indices)
    df.index.name = "Serial"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)


def main() -> None:
    args = parse_args()
    output_path = (
        Path(args.output)
        if args.output is not None
        else DEFAULT_LABEL_DIR / f"labels_{args.num_labels}.csv"
    )

    y_train = load_mnist_labels()
    selected_indices = sample_balanced_indices(y_train, args.num_labels, args.seed)
    write_labels(selected_indices, y_train, output_path)
    print(f"Wrote {selected_indices.size} labels to {output_path}")


if __name__ == "__main__":
    main()
