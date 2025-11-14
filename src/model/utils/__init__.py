"""Model utilities.

Provides JAX device configuration and detection.
"""
from .device import configure_jax_device, get_device_info, print_device_banner

__all__ = [
    "configure_jax_device",
    "get_device_info",
    "print_device_banner",
]
