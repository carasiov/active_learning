"""Minimal JAX device detection with graceful fallback."""
import logging
import os

logger = logging.getLogger(__name__)

_DEVICE_TYPE = None
_DEVICE_COUNT = 0


def configure_jax_device():
    """Initialize JAX with graceful CPU fallback. Call once at startup."""
    global _DEVICE_TYPE, _DEVICE_COUNT

    if _DEVICE_TYPE is not None:
        return _DEVICE_TYPE

    import jax

    try:
        devices = jax.devices()
        _DEVICE_TYPE = devices[0].platform
        _DEVICE_COUNT = len(devices)

        logger.info(f"JAX initialized: {_DEVICE_TYPE} ({_DEVICE_COUNT} device(s))")
        return _DEVICE_TYPE

    except RuntimeError as e:
        # GPU initialization failed, try CPU
        if "CUDA_ERROR_NO_DEVICE" in str(e) or "No device found" in str(e):
            logger.warning(f"GPU initialization failed: {e}")
            logger.info("Falling back to CPU")

            os.environ["JAX_PLATFORMS"] = "cpu"
            jax.clear_backends()

            devices = jax.devices()
            _DEVICE_TYPE = "cpu"
            _DEVICE_COUNT = len(devices)

            logger.info(f"JAX initialized: cpu ({_DEVICE_COUNT} core(s))")
            return "cpu"
        else:
            logger.error(f"JAX initialization failed: {e}")
            raise


def get_device_info():
    """Return (device_type, device_count) or None if not initialized."""
    if _DEVICE_TYPE is None:
        configure_jax_device()
    return _DEVICE_TYPE, _DEVICE_COUNT


def print_device_banner():
    """Print a simple device status banner."""
    device_type, device_count = get_device_info()

    if device_type is None:
        return

    plural = "s" if device_count != 1 else ""
    forced = " [JAX_PLATFORMS override]" if os.environ.get("JAX_PLATFORMS") else ""

    print("=" * 60)
    print(f" JAX Device: {device_type.upper()} ({device_count} device{plural}){forced}")

    if device_type == "cpu" and not os.environ.get("JAX_PLATFORMS"):
        print(" Note: Running on CPU (GPU unavailable)")
        print(" Tip: Check 'nvidia-smi' and Docker '--gpus' flag")

    print("=" * 60)
    print(flush=True)
