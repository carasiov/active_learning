#!/usr/bin/env python3
"""Backward-compatible script for running experiments."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from use_cases.experiments.src.cli import main


if __name__ == "__main__":
    main()
