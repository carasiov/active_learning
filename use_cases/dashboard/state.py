"""Compatibility shim exposing dashboard state module."""
import sys

from use_cases.dashboard.core import state as _core_state

sys.modules[__name__] = _core_state
