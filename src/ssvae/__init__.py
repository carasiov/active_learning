"""Canonical package for the JAX SSVAE implementation."""

from .config import SSVAEConfig
from .models import SSVAE

__all__ = ["SSVAE", "SSVAEConfig"]
