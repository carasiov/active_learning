"""Logging infrastructure for experiment management.

This package provides structured logging with dual output:
- Clean stdout for monitoring (INFO level, no timestamps)
- Detailed file logs for debugging (DEBUG level with timestamps)
- Separate error log for quick diagnosis

Following AGENTS.md principle: Important messages go to both stdout and persistent log.
"""
from __future__ import annotations

from .setup import setup_experiment_logging

__all__ = ["setup_experiment_logging"]
