"""Experiment-specific run directory creation.

IMPORTANT: The RunPaths schema has been moved to src/io/ at the repository root
to enable reuse across projects. This module retains experiment-specific logic
for creating runs in the experiments/results/ directory.

For the RunPaths dataclass itself, import from:
    from io import RunPaths, sanitize_name
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Tuple

# Import schema from new location
from io import RunPaths, sanitize_name

# Experiment-specific constants
EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = EXPERIMENTS_ROOT / "results"


def create_run_paths(
    experiment_name: str | None,
    architecture_code: str | None = None,
) -> Tuple[str, str, RunPaths]:
    """Create organized run directory structure with architecture code.

    Creates a timestamped directory with organized subdirectories for
    artifacts, figures, and logs, including architecture code in directory
    name for easy experiment identification.

    Args:
        experiment_name: User-provided experiment name (will be sanitized)
        architecture_code: Optional architecture code (e.g., "mix10-dir_tau_ca-het")

    Returns:
        Tuple of (run_id, timestamp, RunPaths object)
        - run_id: Full directory name (e.g., "baseline__mix10-dir_tau_ca-het__20241112_143027")
        - timestamp: ISO format timestamp string
        - RunPaths: Structured paths for all experiment outputs

    Directory structure:
        {experiment_name}__{architecture_code}__{timestamp}/
        ├── config.yaml
        ├── summary.json
        ├── REPORT.md
        ├── artifacts/
        │   ├── checkpoints/       # Model checkpoints
        │   ├── diagnostics/       # Latent dumps, histories
        │   ├── tau/              # τ-classifier artifacts
        │   ├── ood/              # OOD scoring data
        │   └── uncertainty/      # Heteroscedastic outputs
        ├── figures/
        │   ├── core/             # Loss curves, latent spaces
        │   ├── mixture/          # Mixture evolution plots
        │   ├── tau/              # τ-classifier visualizations
        │   ├── uncertainty/      # Variance maps
        │   └── ood/              # OOD distributions
        └── logs/
            ├── experiment.log    # Full detailed log
            ├── training.log      # Training progress only
            └── errors.log        # Errors and warnings

    Note:
        Double underscore (__) separators used for easy parsing of directory names.
        If architecture_code is None, falls back to {name}_{timestamp} format
        for backward compatibility.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = sanitize_name(experiment_name)

    # Include architecture code in directory name if provided
    if architecture_code:
        run_id = f"{slug}__{architecture_code}__{timestamp}"
    else:
        # Backward compatibility: no architecture code
        run_id = f"{slug}_{timestamp}"

    run_root = RESULTS_DIR / run_id

    # Top-level paths
    artifacts = run_root / "artifacts"
    figures = run_root / "figures"
    logs = run_root / "logs"

    paths = RunPaths(
        root=run_root,
        config=run_root / "config.yaml",
        summary=run_root / "summary.json",
        report=run_root / "REPORT.md",
        artifacts=artifacts,
        figures=figures,
        logs=logs,
        # Artifact subdirectories
        artifacts_checkpoints=artifacts / "checkpoints",
        artifacts_diagnostics=artifacts / "diagnostics",
        artifacts_tau=artifacts / "tau",
        artifacts_ood=artifacts / "ood",
        artifacts_uncertainty=artifacts / "uncertainty",
        # Figure subdirectories
        figures_core=figures / "core",
        figures_mixture=figures / "mixture",
        figures_tau=figures / "tau",
        figures_uncertainty=figures / "uncertainty",
        figures_ood=figures / "ood",
    )
    paths.ensure()
    return run_id, timestamp, paths
