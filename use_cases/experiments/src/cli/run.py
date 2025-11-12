"""CLI entrypoint for running a single experiment.

Phase 6: Enhanced to generate architecture codes and augment configs with metadata.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ssvae import SSVAEConfig

from ..core.naming import generate_architecture_code
from ..io import create_run_paths, write_config_copy, write_report, write_summary
from ..pipeline import (
    add_repo_paths,
    augment_config_metadata,
    load_experiment_config,
    prepare_data,
    run_training_pipeline,
)

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
    print(f"Architecture: {architecture_code}")
    print(f"Output directory: {run_paths.root}")

    # Phase 6: Augment config with metadata
    experiment_config = augment_config_metadata(
        experiment_config,
        run_id,
        architecture_code,
        timestamp,
    )
    write_config_copy(experiment_config, run_paths)

    X_train, y_semi, y_true = prepare_data(data_config)

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
