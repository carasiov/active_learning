"""Configuration helpers for experiments CLI.

Includes metadata augmentation for self-documenting experiments.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict


def augment_config_metadata(
    config: Dict[str, Any],
    run_id: str,
    architecture_code: str,
    timestamp: str,
) -> Dict[str, Any]:
    """Augment config with metadata for self-documenting experiments.

    Adds run_id, architecture_code, and timestamp to config for
    complete provenance tracking.

    Args:
        config: Experiment configuration dict
        run_id: Full run directory name (e.g., "baseline__mix10-dir__20241112_143027")
        architecture_code: Architecture code (e.g., "mix10-dir_tau_ca-het")
        timestamp: ISO format timestamp string

    Returns:
        Augmented config with metadata added to top level

    Example:
        >>> config = {"experiment": {...}, "data": {...}, "model": {...}}
        >>> config = augment_config_metadata(
        ...     config,
        ...     "baseline__mix10-dir_tau_ca-het__20241112_143027",
        ...     "mix10-dir_tau_ca-het",
        ...     "20241112_143027"
        ... )
        >>> config["run_id"]
        'baseline__mix10-dir_tau_ca-het__20241112_143027'
        >>> config["architecture_code"]
        'mix10-dir_tau_ca-het'
        >>> config["timestamp"]
        '20241112_143027'
    """
    # Add metadata at top level
    config["run_id"] = run_id
    config["architecture_code"] = architecture_code
    config["timestamp"] = timestamp

    return config


def load_experiment_config(config_path: str | Path) -> Dict[str, Any]:
    """Load experiment configuration from a YAML file with helpful errors."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "PyYAML is required to load experiment configs. Install with `poetry add pyyaml`."
        ) from exc

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data: Dict[str, Any] = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:  # pragma: no cover
        raise SystemExit(f"Failed to parse config {config_path}: {exc}") from exc

    return data


def add_repo_paths(experiments_dir: Path) -> None:
    """Ensure src/ and experiments/data/ are importable."""
    root_dir = experiments_dir.parents[1]
    src_dir = root_dir / "src"
    data_dir = experiments_dir / "data"

    for path in (src_dir, data_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
