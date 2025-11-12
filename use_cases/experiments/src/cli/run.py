"""CLI entrypoint for running a single experiment.

Phase 6: Enhanced to generate architecture codes and augment configs with metadata.
Phase 6 (Terminal Cleanup): Professional, organized console output.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from ssvae import SSVAEConfig
from utils import get_device_info

# Phase 6 (Terminal Cleanup): Show each warning only once to prevent duplicates
warnings.simplefilter('once', UserWarning)

from ..core.naming import generate_architecture_code
from ..io import create_run_paths, write_config_copy, write_report, write_summary
from ..pipeline import (
    add_repo_paths,
    augment_config_metadata,
    load_experiment_config,
    prepare_data,
    run_training_pipeline,
)
from ..utils import format_experiment_header, format_training_section_header

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    default_config = EXPERIMENTS_DIR / "configs" / "default.yaml"
    parser = argparse.ArgumentParser(description="Run SSVAE experiment from config file.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help=f"Path to experiment YAML config (default: {default_config})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_repo_paths(EXPERIMENTS_DIR)

    experiment_config = load_experiment_config(args.config)
    exp_meta = experiment_config.get("experiment", {})
    data_config = experiment_config.get("data", {})
    model_config = experiment_config.get("model", {})

    # Phase 6: Generate architecture code for directory naming
    # Create SSVAEConfig early for validation and code generation
    _model_config = {**model_config}
    if isinstance(_model_config.get("hidden_dims"), list):
        _model_config["hidden_dims"] = tuple(_model_config["hidden_dims"])

    ssvae_config = SSVAEConfig(**_model_config)
    architecture_code = generate_architecture_code(ssvae_config)

    # Phase 6: Create run paths with architecture code
    run_id, timestamp, run_paths = create_run_paths(
        exp_meta.get("name"),
        architecture_code,
    )

    # Phase 6: Augment config with metadata
    experiment_config = augment_config_metadata(
        experiment_config,
        run_id,
        architecture_code,
        timestamp,
    )
    write_config_copy(experiment_config, run_paths)

    # Load data
    X_train, y_semi, y_true = prepare_data(data_config)

    # Create data info for header
    val_split = model_config.get("val_split", 0.1)
    train_size = int(len(X_train) * (1 - val_split))
    val_size = len(X_train) - train_size
    labeled_count = int((~np.isnan(y_semi)).sum())

    data_info = {
        "dataset": data_config.get("dataset", "MNIST"),
        "total": len(X_train),
        "labeled": labeled_count,
        "train_size": train_size,
        "val_size": val_size,
    }

    # Get device info
    device_type, device_count = get_device_info()
    device_info = (device_type, device_count) if device_type else None

    # Phase 6 (Terminal Cleanup): Print clean experiment header
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

    model, history, summary, viz_meta = run_training_pipeline(
        model_config, X_train, y_semi, y_true, run_paths
    )

    # Phase 6: Enhance summary with metadata for self-documenting experiments
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

    print("\n" + "=" * 60)
    print(f"Experiment complete! Results: {run_paths.root}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
