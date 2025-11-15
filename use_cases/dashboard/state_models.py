"""Compatibility shim exposing dashboard state models."""
import sys

from use_cases.dashboard.core import state_models as _core_state_models

sys.modules[__name__] = _core_state_models
