#!/usr/bin/env python3
"""Aggregate label efficiency experiment results."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIR = BASE_DIR / "experiments" / "artifacts" / "label_efficiency"

HISTORY_SUFFIX = "_history.csv"
FILENAME_PATTERN = re.compile(r"labels-(\d+)_history$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect classification metrics from experiment histories.")
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=str(DEFAULT_EXPERIMENT_DIR),
        help="Directory containing *_history.csv files.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="Optional override for the summary CSV location.",
    )
    return parser.parse_args()


def extract_num_labels(path: Path) -> int:
    match = FILENAME_PATTERN.search(path.stem)
    if not match:
        raise ValueError(f"Could not infer label count from filename '{path.name}'.")
    return int(match.group(1))


def collect_records(history_paths: List[Path]) -> pd.DataFrame:
    records = []
    for history_path in history_paths:
        num_labels = extract_num_labels(history_path)
        df = pd.read_csv(history_path)
        if df.empty:
            raise ValueError(f"History file '{history_path}' is empty.")
        required_cols = {"classification_loss", "val_classification_loss"}
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"History file '{history_path}' is missing columns: {sorted(missing)}")
        last_row = df.iloc[-1]
        records.append(
            {
                "run_tag": history_path.parent.name,
                "num_labels": num_labels,
                "train_cls_loss": float(last_row["classification_loss"]),
                "val_cls_loss": float(last_row["val_classification_loss"]),
            }
        )
    summary = pd.DataFrame(records)
    if not summary.empty:
        summary = summary.sort_values(["run_tag", "num_labels"]).reset_index(drop=True)
    return summary


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory '{experiment_dir}' does not exist.")

    history_paths = sorted(p for p in experiment_dir.rglob(f"*{HISTORY_SUFFIX}") if p.is_file())
    if not history_paths:
        raise FileNotFoundError(f"No history files matching '*{HISTORY_SUFFIX}' found in '{experiment_dir}'.")

    summary = collect_records(history_paths)
    summary_path = Path(args.summary_path) if args.summary_path else experiment_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    float_fmt = lambda x: f"{x:.6f}"  # noqa: E731  # small inline helper for printing
    print(summary.to_string(index=False, float_format=float_fmt))
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
