"""Compatibility shim exposing dashboard ModelManager."""
import sys

from use_cases.dashboard.core import model_manager as _core_model_manager

sys.modules[__name__] = _core_model_manager
