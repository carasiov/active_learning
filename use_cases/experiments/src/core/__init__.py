"""Core experiment management infrastructure.

This package provides foundational experiment management capabilities:
- naming: Architecture code generation from config
- validation: Config validation rules and precondition checks

These modules follow the AGENTS.md principle of "fail fast" - validation
happens at config load time, not after training starts.
"""
from __future__ import annotations

from .naming import generate_architecture_code, generate_naming_legend
from .validation import validate_config

__all__ = [
    "generate_architecture_code",
    "generate_naming_legend",
    "validate_config",
]
