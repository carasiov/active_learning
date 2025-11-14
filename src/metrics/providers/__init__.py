"""Default metric providers for common experiment metrics.

This module contains the default metric providers that register themselves
when imported. Experiments can import this module to auto-register all
default metrics.

Usage:
    from metrics.providers import defaults  # Auto-registers all default metrics
"""
from __future__ import annotations

from . import defaults

__all__ = ["defaults"]
