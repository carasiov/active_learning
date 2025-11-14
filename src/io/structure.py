"""Run directory organization schema.

This module defines the RunPaths dataclass, which provides a standardized
schema for organizing experiment outputs. This schema is reusable across
different tools and projects that need to work with experiment results.

The actual creation logic (create_run_paths) remains experiment-specific
in use_cases/experiments/src/io/ since it depends on EXPERIMENTS_ROOT.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


def sanitize_name(name: str | None) -> str:
    """Convert experiment name to filesystem-safe slug.

    Args:
        name: User-provided experiment name

    Returns:
        Sanitized string safe for directory names

    Examples:
        >>> sanitize_name("My Experiment!")
        'my_experiment'
        >>> sanitize_name(None)
        'experiment'
    """
    if not name:
        return "experiment"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_")
    return slug or "experiment"


@dataclass(slots=True)
class RunPaths:
    """Structured paths for experiment outputs.

    Provides a canonical schema for organizing training results, checkpoints,
    figures, and logs. This schema is used by:
    - Training pipelines to save outputs
    - Dashboard to locate and load experiment results
    - Analysis tools to process saved experiments

    Directory structure:
        {run_root}/
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

    Attributes:
        root: Base directory for this experiment run
        config: Path to config.yaml
        summary: Path to summary.json
        report: Path to REPORT.md
        artifacts: Base artifacts directory
        figures: Base figures directory
        logs: Base logs directory
        artifacts_*: Subdirectories within artifacts/
        figures_*: Subdirectories within figures/
    """
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
        """Create all directories in the run structure.

        Creates all top-level and subdirectories if they don't exist.
        Safe to call multiple times (idempotent).
        """
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
