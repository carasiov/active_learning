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

    def ensure(self) -> None:
        for path in (self.root, self.artifacts, self.figures, self.logs):
            path.mkdir(parents=True, exist_ok=True)


def create_run_paths(experiment_name: str | None) -> Tuple[str, RunPaths]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = sanitize_name(experiment_name)
    run_root = RESULTS_DIR / f"{slug}_{timestamp}"

    paths = RunPaths(
        root=run_root,
        config=run_root / "config.yaml",
        summary=run_root / "summary.json",
        report=run_root / "REPORT.md",
        artifacts=run_root / "artifacts",
        figures=run_root / "figures",
        logs=run_root / "logs",
    )
    paths.ensure()
    return timestamp, paths
