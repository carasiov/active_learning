"""Run directory organization helpers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = EXPERIMENTS_ROOT / "results"


def sanitize_name(name: str | None) -> str:
    if not name:
        return "experiment"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_")
    return slug or "experiment"


@dataclass(slots=True)
class RunPaths:
    root: Path
    config: Path
    summary: Path
    report: Path
    artifacts: Path
    figures: Path
    logs: Path

    # Artifact subdirectories
    artifacts_checkpoints: Path
    artifacts_diagnostics: Path
    artifacts_tau: Path
    artifacts_ood: Path
    artifacts_uncertainty: Path

    # Figure subdirectories
    figures_core: Path
    figures_mixture: Path
    figures_tau: Path
    figures_uncertainty: Path
    figures_ood: Path

    def ensure(self) -> None:
        """Create all directories in the run structure."""
        # Top-level directories
        for path in (self.root, self.artifacts, self.figures, self.logs):
            path.mkdir(parents=True, exist_ok=True)

        # Artifact subdirectories
        for path in (
            self.artifacts_checkpoints,
            self.artifacts_diagnostics,
            self.artifacts_tau,
            self.artifacts_ood,
            self.artifacts_uncertainty,
        ):
            path.mkdir(parents=True, exist_ok=True)

        # Figure subdirectories
        for path in (
            self.figures_core,
            self.figures_mixture,
            self.figures_tau,
            self.figures_uncertainty,
            self.figures_ood,
        ):
            path.mkdir(parents=True, exist_ok=True)


def create_run_paths(experiment_name: str | None) -> Tuple[str, RunPaths]:
    """Create organized run directory structure.

    Creates a timestamped directory with organized subdirectories for
    artifacts, figures, and logs. Following Phase 1 design for component-based
    organization.

    Args:
        experiment_name: User-provided experiment name (will be sanitized)

    Returns:
        Tuple of (timestamp, RunPaths object)

    Directory structure:
        {experiment_name}_{timestamp}/
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
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = sanitize_name(experiment_name)
    run_root = RESULTS_DIR / f"{slug}_{timestamp}"

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
    return timestamp, paths
