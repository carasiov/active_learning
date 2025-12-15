#!/usr/bin/env python3
"""Unified CLI for running and validating experiments."""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from rcmvae.utils import get_device_info  # noqa: E402
from rcmvae.domain.config import SSVAEConfig  # noqa: E402
from use_cases.experiments.src.config import (  # noqa: E402
    add_repo_paths,
    augment_config_metadata,
    load_experiment_config,
)
from use_cases.experiments.src.data import prepare_data  # noqa: E402
from use_cases.experiments.src.formatters import (  # noqa: E402
    format_experiment_header,
    format_training_section_header,
)
from use_cases.experiments.src.naming import generate_architecture_code  # noqa: E402
from use_cases.experiments.src.reporting import (  # noqa: E402
    write_config_copy,
    write_report,
    write_summary,
)
from use_cases.experiments.src.run import run_training_pipeline  # noqa: E402
from use_cases.experiments.src.structure import create_run_paths  # noqa: E402
from use_cases.experiments.src.validation import (  # noqa: E402
    ConfigValidationError,
    validate_config,
)


EXPERIMENTS_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
DEFAULT_CONFIG = CONFIGS_DIR / "default.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or validate an experiment configuration.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path or name of the experiment config (relative names are resolved in configs/).",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configs and exit.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the configuration (and print warnings) without training.",
    )
    return parser


def list_available_configs() -> list[Path]:
    if not CONFIGS_DIR.exists():
        return []
    return sorted(CONFIGS_DIR.glob("*.yaml"))


def print_available_configs() -> None:
    configs = list_available_configs()
    if not configs:
        print("No configs found under", CONFIGS_DIR)
        return
    print("Available configs:")
    for cfg in configs:
        print(f"  - {cfg.name}")


def resolve_config_path(raw_value: str) -> Path:
    candidate = Path(raw_value).expanduser()
    if candidate.is_file():
        return candidate

    if candidate.exists() and candidate.is_dir():
        raise SystemExit(f"Config path refers to a directory: {candidate}")

    for suffix in ("", ".yaml"):
        guess = CONFIGS_DIR / f"{raw_value}{suffix}"
        if guess.is_file():
            return guess
    raise FileNotFoundError(
        f"Could not find config '{raw_value}'. "
        f"Checked absolute path and {CONFIGS_DIR}."
    )


def _normalize_model_config(model_config: dict) -> dict:
    cfg = dict(model_config)
    hidden = cfg.get("hidden_dims")
    if isinstance(hidden, list):
        cfg["hidden_dims"] = tuple(hidden)
    return cfg


def _render_warning_block(captured: Sequence[warnings.WarningMessage]) -> None:
    if not captured:
        return
    print("\n" + "=" * 80)
    print("Configuration Warnings")
    print("=" * 80)
    for warning in captured:
        print(f"  ⚠  {warning.message}")
    print("=" * 80 + "\n")


def _print_validation_success(config_path: Path) -> None:
    print("\n" + "=" * 80)
    print(f"Configuration '{config_path.name}' is valid.")
    print("=" * 80 + "\n")


def run_with_config(config_path: Path, *, validate_only: bool) -> int:
    add_repo_paths(EXPERIMENTS_DIR)

    try:
        experiment_config = load_experiment_config(config_path)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    exp_meta = experiment_config.get("experiment", {})
    data_config = experiment_config.get("data", {})
    model_config = _normalize_model_config(experiment_config.get("model", {}))

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", UserWarning)
        ssvae_config = SSVAEConfig(**model_config)
        architecture_code = generate_architecture_code(ssvae_config)

    try:
        validate_config(ssvae_config)
    except ConfigValidationError as exc:
        print("\n" + "=" * 80)
        print("Configuration Error")
        print("=" * 80)
        print(f"  ✗  {exc}")
        print("=" * 80 + "\n")
        return 2

    _render_warning_block(caught_warnings)

    if validate_only:
        _print_validation_success(config_path)
        return 0

    run_id, timestamp, run_paths = create_run_paths(
        exp_meta.get("name"),
        architecture_code,
    )
    experiment_config = augment_config_metadata(
        experiment_config,
        run_id,
        architecture_code,
        timestamp,
    )
    write_config_copy(experiment_config, run_paths)

    warnings.filterwarnings("ignore", category=UserWarning)
    x_train, y_semi, y_true, dataset_label = prepare_data(data_config)

    val_split = model_config.get("val_split", 0.1)
    train_size = int(len(x_train) * (1 - val_split))
    val_size = len(x_train) - train_size
    labeled_count = int((~np.isnan(y_semi)).sum())
    data_info = {
        "dataset": dataset_label,
        "total": len(x_train),
        "labeled": labeled_count,
        "train_size": train_size,
        "val_size": val_size,
    }

    device_type, device_count = get_device_info()
    device_info = (device_type, device_count) if device_type else None

    header = format_experiment_header(
        config=experiment_config,
        run_id=run_id,
        architecture_code=architecture_code,
        output_path=run_paths.root,
        data_info=data_info,
        device_info=device_info,
    )
    print(header)
    print(format_training_section_header())

    # Extract curriculum config (NOT forwarded to SSVAEConfig)
    curriculum_config = experiment_config.get("curriculum")

    model, history, summary, viz_meta = run_training_pipeline(
        model_config,
        x_train,
        y_semi,
        y_true,
        run_paths,
        curriculum_config=curriculum_config,
    )

    summary["metadata"] = {
        "run_id": run_id,
        "architecture_code": architecture_code,
        "timestamp": timestamp,
        "experiment_name": exp_meta.get("name", "experiment"),
    }

    write_summary(summary, run_paths)
    recon_paths = viz_meta.get("reconstructions") if isinstance(viz_meta, dict) else None
    plot_status = viz_meta.get("_plot_status") if isinstance(viz_meta, dict) else None
    write_report(summary, history, experiment_config, run_paths, recon_paths, plot_status)

    print("\n" + "=" * 80)
    print(f"Experiment complete! Results: {run_paths.root}")
    print("=" * 80 + "\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_configs:
        print_available_configs()
        return 0

    try:
        config_path = resolve_config_path(args.config)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    return run_with_config(config_path, validate_only=args.validate_only)


if __name__ == "__main__":
    raise SystemExit(main())
