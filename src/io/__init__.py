"""Filesystem helpers for experiment inputs and outputs.

This module provides the RunPaths schema for organizing experiment results
in a standardized way. The schema can be used by:
- Training pipelines to save outputs
- Dashboard to locate and load experiments
- Analysis tools to process results

Usage:
    from io import RunPaths, sanitize_name

    # Create paths manually
    paths = RunPaths(
        root=Path("results/experiment_001"),
        config=Path("results/experiment_001/config.yaml"),
        ...
    )
    paths.ensure()  # Create all directories

    # Or use experiment-specific create_run_paths() from use_cases/experiments
"""

from .structure import (
    RunPaths,
    sanitize_name,
)

__all__ = [
    "RunPaths",
    "sanitize_name",
]
