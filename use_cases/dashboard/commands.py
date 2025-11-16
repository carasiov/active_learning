"""Compatibility shim exposing dashboard command module."""

from use_cases.dashboard.core.commands import *  # noqa: F401,F403
from use_cases.dashboard.core.commands import CreateModelCommand as _CoreCreateModelCommand


class CreateModelCommand(_CoreCreateModelCommand):
	"""Backward-compatible signature accepting legacy keyword arguments."""

	def __init__(
		self,
		name: str | None = None,
		*,
		config_preset: str | None = "default",
		num_samples: int = 1024,
		num_labeled: int = 128,
		seed: int | None = None,
	) -> None:
		# config_preset is retained for compatibility but currently unused
		super().__init__(name=name, num_samples=num_samples, num_labeled=num_labeled, seed=seed)


__all__ = [name for name in globals() if not name.startswith("_")]  # type: ignore[var-annotated]
