"""Logging infrastructure for experiment runs.

This module provides structured logging with multiple output handlers for
clean console output and detailed file logging.

Public API:
    setup_experiment_logging - Configure logging handlers
    log_section_header - Log section dividers
    log_config_summary - Log configuration details
    log_model_initialization - Log model architecture
    log_training_epoch - Log training progress
"""
from __future__ import annotations

from .setup import (
    setup_experiment_logging,
    log_section_header,
    log_config_summary,
    log_model_initialization,
    log_training_epoch,
)

__all__ = [
    "setup_experiment_logging",
    "log_section_header",
    "log_config_summary",
    "log_model_initialization",
    "log_training_epoch",
]
