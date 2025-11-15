"""Compatibility shim exposing dashboard command module."""
import sys

from use_cases.dashboard.core import commands as _core_commands

sys.modules[__name__] = _core_commands
