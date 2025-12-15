"""Curriculum controller for channel unlocking."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class UnlockEvent:
    """Record of a single unlock event."""

    epoch: int
    k_active_before: int
    k_active_after: int
    trigger_metric: str
    trigger_value: float


@dataclass
class CurriculumConfig:
    """Configuration for the curriculum controller.

    Attributes:
        enabled: Whether curriculum is active
        k_active_init: Initial number of active channels (default 1)
        k_active_max: Maximum channels to unlock (None = use model's num_components)

        Unlock policy:
        unlock_policy: "plateau" (v1 default; "normality" planned for v2)
        unlock_monitor: Metric to monitor for plateau detection (e.g., "val_loss")
        unlock_patience_epochs: Epochs of no improvement before unlocking
        unlock_min_delta: Minimum improvement to reset patience counter
        unlock_cooldown_epochs: Epochs to wait after unlock before allowing another

        Kick policy:
        kick_enabled: Whether to apply kick after unlock
        kick_epochs: Number of epochs for kick window
        kick_gumbel_temperature: Temperature override during kick
    """

    enabled: bool = False
    k_active_init: int = 1
    k_active_max: Optional[int] = None  # None = use config.num_components

    # Unlock policy
    unlock_policy: str = "plateau"
    unlock_monitor: str = "val_loss"
    unlock_patience_epochs: int = 10
    unlock_min_delta: float = 0.001
    unlock_cooldown_epochs: int = 3

    # Kick policy
    kick_enabled: bool = True
    kick_epochs: int = 5
    kick_gumbel_temperature: float = 5.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CurriculumConfig:
        """Create config from a dictionary (YAML-friendly)."""
        if not d:
            return cls()

        unlock = d.get("unlock", {})
        kick = d.get("kick", {})

        return cls(
            enabled=d.get("enabled", False),
            k_active_init=d.get("k_active_init", 1),
            k_active_max=d.get("k_active_max"),
            unlock_policy=unlock.get("policy", "plateau"),
            unlock_monitor=unlock.get("monitor", "val_loss"),
            unlock_patience_epochs=unlock.get("patience_epochs", 10),
            unlock_min_delta=unlock.get("min_delta", 0.001),
            unlock_cooldown_epochs=unlock.get("cooldown_epochs", 3),
            kick_enabled=kick.get("enabled", True),
            kick_epochs=kick.get("epochs", 5),
            kick_gumbel_temperature=kick.get("gumbel_temperature", 5.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "k_active_init": self.k_active_init,
            "k_active_max": self.k_active_max,
            "unlock": {
                "policy": self.unlock_policy,
                "monitor": self.unlock_monitor,
                "patience_epochs": self.unlock_patience_epochs,
                "min_delta": self.unlock_min_delta,
                "cooldown_epochs": self.unlock_cooldown_epochs,
            },
            "kick": {
                "enabled": self.kick_enabled,
                "epochs": self.kick_epochs,
                "gumbel_temperature": self.kick_gumbel_temperature,
            },
        }


@dataclass
class CurriculumState:
    """Mutable state for curriculum tracking."""

    k_active: int
    k_max: int
    kick_remaining: int = 0
    cooldown_remaining: int = 0

    # Plateau tracking
    best_metric: float = float("inf")
    patience_counter: int = 0

    # History
    unlock_events: List[UnlockEvent] = field(default_factory=list)

    @property
    def active_mask(self) -> np.ndarray:
        """Return boolean mask [K_max] with True for active channels."""
        mask = np.zeros(self.k_max, dtype=bool)
        mask[:self.k_active] = True
        return mask

    @property
    def is_in_kick(self) -> bool:
        """Return True if currently in a kick window."""
        return self.kick_remaining > 0

    @property
    def can_unlock(self) -> bool:
        """Return True if an unlock is allowed (cooldown expired, not at max)."""
        return self.cooldown_remaining <= 0 and self.k_active < self.k_max


class CurriculumController:
    """Controller for channel curriculum ("pots") learning.

    Tracks active channels and implements plateau-based unlock policy.
    Produces hooks for injecting curriculum state into training.

    Example:
        >>> config = CurriculumConfig(enabled=True, k_active_init=1)
        >>> controller = CurriculumController(config, k_max=10)
        >>> mask = controller.get_active_mask()  # [True, False, ..., False]
        >>> controller.on_epoch_end(epoch=5, metrics={"val_loss": 0.5})
    """

    def __init__(self, config: CurriculumConfig, k_max: int):
        """Initialize curriculum controller.

        Args:
            config: Curriculum configuration
            k_max: Maximum number of channels (from model config)
        """
        self.config = config
        self._k_max = k_max

        # Resolve k_active_max
        effective_max = config.k_active_max if config.k_active_max is not None else k_max
        effective_max = min(effective_max, k_max)

        self._state = CurriculumState(
            k_active=min(config.k_active_init, effective_max),
            k_max=effective_max,
        )

        # Epoch history for reporting
        self._epoch_history: List[Dict[str, Any]] = []

    @property
    def k_active(self) -> int:
        """Current number of active channels."""
        return self._state.k_active

    @property
    def k_max(self) -> int:
        """Maximum number of channels that can be unlocked."""
        return self._state.k_max

    def get_active_mask(self) -> np.ndarray:
        """Return boolean mask indicating active channels [K_max]."""
        # Full mask over architectural K_max (not just unlockable subset)
        mask = np.zeros(self._k_max, dtype=bool)
        mask[:self._state.k_active] = True
        return mask

    def get_gumbel_temperature_override(self) -> Optional[float]:
        """Return temperature override during kick, or None if not in kick."""
        if not self.config.kick_enabled:
            return None
        if self._state.is_in_kick:
            return self.config.kick_gumbel_temperature
        return None

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Process end of epoch: check for unlock, update state.

        Args:
            epoch: Current epoch number (0-indexed)
            metrics: Dictionary of metrics from training/validation

        Returns:
            Dictionary with curriculum state for logging:
                - k_active: int
                - unlocked: bool (1 if unlock happened this epoch)
                - kick_active: bool (1 if in kick window)
        """
        if not self.config.enabled:
            return {"k_active": self._k_max, "unlocked": False, "kick_active": False}

        unlocked = False
        monitor_value = metrics.get(self.config.unlock_monitor, float("inf"))

        # Decrement counters
        if self._state.kick_remaining > 0:
            self._state.kick_remaining -= 1
        if self._state.cooldown_remaining > 0:
            self._state.cooldown_remaining -= 1

        # Check plateau trigger (only when not in cooldown)
        if self._state.can_unlock:
            improved = (self._state.best_metric - monitor_value) > self.config.unlock_min_delta

            if improved:
                self._state.best_metric = monitor_value
                self._state.patience_counter = 0
            else:
                self._state.patience_counter += 1

                # Trigger unlock if patience exceeded
                if self._state.patience_counter >= self.config.unlock_patience_epochs:
                    unlocked = self._do_unlock(epoch, monitor_value)

        # Record epoch state
        epoch_record = {
            "epoch": epoch,
            "k_active": self._state.k_active,
            "unlocked": unlocked,
            "kick_active": self._state.is_in_kick,
            "monitor_value": monitor_value,
            "patience_counter": self._state.patience_counter,
        }
        self._epoch_history.append(epoch_record)

        return {
            "k_active": self._state.k_active,
            "unlocked": unlocked,
            "kick_active": self._state.is_in_kick,
        }

    def _do_unlock(self, epoch: int, trigger_value: float) -> bool:
        """Perform an unlock: increment k_active, start kick, record event."""
        if self._state.k_active >= self._state.k_max:
            return False

        k_before = self._state.k_active
        self._state.k_active += 1
        k_after = self._state.k_active

        # Reset patience for next plateau detection
        self._state.patience_counter = 0
        self._state.best_metric = float("inf")

        # Start kick window and cooldown
        if self.config.kick_enabled:
            self._state.kick_remaining = self.config.kick_epochs
        self._state.cooldown_remaining = self.config.unlock_cooldown_epochs

        # Record event
        event = UnlockEvent(
            epoch=epoch,
            k_active_before=k_before,
            k_active_after=k_after,
            trigger_metric=self.config.unlock_monitor,
            trigger_value=trigger_value,
        )
        self._state.unlock_events.append(event)

        print(
            f"[Curriculum] Epoch {epoch}: Unlocked channel {k_after} "
            f"(plateau trigger: {self.config.unlock_monitor}={trigger_value:.4f})"
        )

        return True

    def get_unlock_events(self) -> List[UnlockEvent]:
        """Return list of all unlock events."""
        return list(self._state.unlock_events)

    def get_epoch_history(self) -> List[Dict[str, Any]]:
        """Return per-epoch curriculum state history."""
        return list(self._epoch_history)

    def get_summary(self) -> Dict[str, Any]:
        """Return summary statistics for reporting."""
        return {
            "final_k_active": self._state.k_active,
            "k_max": self._state.k_max,
            "unlock_count": len(self._state.unlock_events),
            "unlock_epochs": [e.epoch for e in self._state.unlock_events],
        }

    def reset(self) -> None:
        """Reset controller to initial state (for re-training)."""
        effective_max = (
            self.config.k_active_max
            if self.config.k_active_max is not None
            else self._k_max
        )
        effective_max = min(effective_max, self._k_max)

        self._state = CurriculumState(
            k_active=min(self.config.k_active_init, effective_max),
            k_max=effective_max,
        )
        self._epoch_history = []
