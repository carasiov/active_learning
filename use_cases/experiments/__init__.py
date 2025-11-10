"""Experiments package bootstrap."""
from __future__ import annotations

import sys
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXPERIMENTS_DIR.parents[1]
_SRC_DIR = _REPO_ROOT / "src"
_DATA_DIR = _EXPERIMENTS_DIR / "data"

for path in (_SRC_DIR, _DATA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
