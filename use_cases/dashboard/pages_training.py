"""Compatibility shim exposing training page helpers."""
from use_cases.dashboard.pages.training import (
    build_training_config_page,
    register_config_page_callbacks,
)

__all__ = ["build_training_config_page", "register_config_page_callbacks"]
