"""Configuration helpers for experiments CLI."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict


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
