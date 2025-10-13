from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, Tuple

DeviceType = Literal["cpu", "gpu", "tpu"]


@dataclass(frozen=True)
class DeviceInfo:
    device_type: DeviceType
    platform: str
    device_count: int
    device_names: Tuple[str, ...]
    forced: bool
    message: str | None = None


_DEVICE_INFO: DeviceInfo | None = None


def configure_jax_device(force_refresh: bool = False) -> DeviceType:
    """Detect the active JAX backend and expose a stable device summary."""
    global _DEVICE_INFO
    if _DEVICE_INFO is not None and not force_refresh:
        return _DEVICE_INFO.device_type

    logger = logging.getLogger(__name__)
    env_value = os.environ.get("JAX_PLATFORMS")
    forced = bool(env_value)
    reason: str | None = None

    if forced:
        target_env = _extract_first_token(env_value)
        backend_arg = _platform_env_to_backend_arg(target_env)
        if "," in env_value:
            backend_arg = None
    else:
        target_env, reason = _detect_preferred_env(logger)
        os.environ["JAX_PLATFORMS"] = target_env
        backend_arg = _platform_env_to_backend_arg(target_env)

    jax = importlib.import_module("jax")

    try:
        devices = _fetch_devices(jax, backend_arg)
    except Exception as err:  # pragma: no cover - defensive
        if not forced and backend_arg == "gpu" and _is_cuda_unavailable(err):
            logger.warning(
                "CUDA backend initialization failed after detection (%s). Falling back to CPU.",
                err,
            )
            os.environ["JAX_PLATFORMS"] = "cpu"
            jax = importlib.reload(jax)
            backend_arg = "cpu"
            devices = _fetch_devices(jax, backend_arg)
            reason = "CUDA devices unavailable; running on CPU"
        else:
            raise

    if not devices:
        raise RuntimeError("No JAX devices available; check your installation.")

    device_type = _device_platform_to_type(devices[0].platform)
    platform_name = devices[0].platform
    device_names = _collect_device_names(devices)

    _DEVICE_INFO = DeviceInfo(
        device_type=device_type,
        platform=platform_name,
        device_count=len(devices),
        device_names=device_names,
        forced=forced,
        message=reason,
    )

    note = " (forced by JAX_PLATFORMS)" if forced else ""
    logger.info(
        "JAX backend: %s with %d device(s)%s",
        platform_name.upper(),
        len(devices),
        note,
    )
    if reason:
        logger.info("Device note: %s", reason)

    return _DEVICE_INFO.device_type


def get_configured_device_info() -> DeviceInfo | None:
    """Return cached device information, detecting it if necessary."""
    if _DEVICE_INFO is None:
        try:
            configure_jax_device()
        except Exception:  # pragma: no cover - detection failure should propagate elsewhere
            return None
    return _DEVICE_INFO


def print_device_banner(info: DeviceInfo | None = None) -> None:
    """Print a human-readable banner describing the active JAX device."""
    device_info = info or get_configured_device_info()
    if device_info is None:
        return

    label = device_info.device_type.upper()
    plural = "s" if device_info.device_count != 1 else ""
    forced_note = " [JAX_PLATFORMS override]" if device_info.forced else ""

    lines = [
        "=" * 60,
        f" JAX Device: {label} ({device_info.device_count} device{plural}){forced_note}",
    ]
    if device_info.device_names:
        lines.append(f" Devices: {', '.join(device_info.device_names)}")
    if device_info.message:
        lines.append(f" Note: {device_info.message}")
    lines.append("=" * 60)
    print("\n".join(lines), flush=True)


def _extract_first_token(raw: str) -> str:
    return raw.split(",")[0].strip().lower()


def _detect_preferred_env(logger: logging.Logger) -> Tuple[str, str | None]:
    try:
        from jaxlib import xla_extension
    except ImportError:
        return "cpu", "jaxlib lacks GPU runtime; running on CPU"

    try:
        backend = xla_extension.get_backend("gpu")
        platform = getattr(backend, "platform", "gpu")
        if platform == "gpu":
            return "cuda", None
    except RuntimeError as err:
        message = str(err)
        if _is_cuda_unavailable(err):
            logger.debug("CUDA backend unavailable: %s", message)
            gpu_reason = "CUDA devices unavailable"
        elif "backend 'gpu' not found" in message.lower():
            logger.debug("GPU backend not built in jaxlib: %s", message)
            gpu_reason = "JAX installed without GPU support"
        else:
            logger.warning("Unexpected GPU backend error (%s); using CPU.", message)
            gpu_reason = message
        try:
            backend = xla_extension.get_backend("tpu")
            platform = getattr(backend, "platform", "tpu")
            if platform == "tpu":
                return "tpu", None
        except RuntimeError:
            return "cpu", gpu_reason
        return "cpu", gpu_reason
    return "cuda", None


def _fetch_devices(jax_module: Any, backend_arg: str | None) -> Sequence[Any]:
    if backend_arg:
        return jax_module.devices(backend_arg)
    return jax_module.devices()


def _platform_env_to_backend_arg(platform_env: str) -> str | None:
    mapping = {
        "cuda": "gpu",
        "gpu": "gpu",
        "cpu": "cpu",
        "tpu": "tpu",
    }
    return mapping.get(platform_env, None)


def _device_platform_to_type(platform: str) -> DeviceType:
    platform_lower = platform.lower()
    if platform_lower == "gpu":
        return "gpu"
    if platform_lower == "tpu":
        return "tpu"
    return "cpu"


def _collect_device_names(devices: Sequence[Any]) -> Tuple[str, ...]:
    names = []
    for device in devices:
        platform = getattr(device, "platform", "unknown").upper()
        device_kind = getattr(device, "device_kind", "").strip()
        device_id = getattr(device, "id", None)
        if device_kind:
            descriptor = device_kind
        else:
            descriptor = platform
        if device_id is not None:
            names.append(f"{platform}:{device_id} ({descriptor})")
        else:
            names.append(f"{platform} ({descriptor})")
    return tuple(names)


def _is_cuda_unavailable(err: Exception) -> bool:
    message = str(err).lower()
    return "cuda_error_no_device" in message or "could not initialize cuda" in message
