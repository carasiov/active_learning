"""Infrastructure layer for experiment tooling.

This module provides reusable components for:
- Logging: Structured logging with multiple handlers
- Metrics: Metric collection and registry
- Visualization: Plot generation and registry
- RunPaths: Standardized directory structures

All components use ComponentResult for explicit status tracking.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class ComponentStatus(Enum):
    """Status of a metric component computation.

    States follow the fail-fast principle:
    - SUCCESS: Computation completed normally
    - DISABLED: Component not enabled in config (expected)
    - SKIPPED: Enabled but preconditions not met (warning-worthy)
    - FAILED: Enabled and attempted but computation failed (error)
    """
    SUCCESS = "success"
    DISABLED = "disabled"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ComponentResult:
    """Result from a metric computation with explicit status.

    Attributes:
        status: Whether computation succeeded or why it didn't
        data: The computed metrics (only for SUCCESS status)
        reason: Human-readable explanation (for DISABLED/SKIPPED/FAILED)
        error: Original exception (only for FAILED status)

    Design notes:
    - Immutable (frozen=True) to prevent accidental modification
    - Uses slots for memory efficiency (experiment runs collect many results)
    - Factory methods enforce correct attribute combinations

    Examples:
        >>> ComponentResult.success(data={"K_eff": 7.3})
        ComponentResult(status=SUCCESS, data={"K_eff": 7.3}, ...)

        >>> ComponentResult.disabled(reason="Prior is not mixture-based")
        ComponentResult(status=DISABLED, data=None, reason=..., ...)
    """
    status: ComponentStatus
    data: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    error: Optional[Exception] = None

    @classmethod
    def success(cls, data: Dict[str, Any]) -> ComponentResult:
        """Create a successful result with computed metrics."""
        return cls(status=ComponentStatus.SUCCESS, data=data)

    @classmethod
    def disabled(cls, reason: str) -> ComponentResult:
        """Create a disabled result for components not enabled in config."""
        return cls(status=ComponentStatus.DISABLED, reason=reason)

    @classmethod
    def skipped(cls, reason: str) -> ComponentResult:
        """Create a skipped result for enabled components with unmet preconditions."""
        return cls(status=ComponentStatus.SKIPPED, reason=reason)

    @classmethod
    def failed(cls, reason: str, error: Exception) -> ComponentResult:
        """Create a failed result for computation errors."""
        return cls(status=ComponentStatus.FAILED, reason=reason, error=error)

    @property
    def is_success(self) -> bool:
        """True if computation succeeded."""
        return self.status == ComponentStatus.SUCCESS

    @property
    def is_disabled(self) -> bool:
        """True if component is disabled."""
        return self.status == ComponentStatus.DISABLED

    @property
    def is_skipped(self) -> bool:
        """True if component was skipped."""
        return self.status == ComponentStatus.SKIPPED

    @property
    def is_failed(self) -> bool:
        """True if computation failed."""
        return self.status == ComponentStatus.FAILED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dict suitable for summary.json."""
        result = {"status": self.status.value}

        # Add data for success (merge into result dict)
        if self.is_success and self.data:
            result.update(self.data)

        # Add reason for non-success statuses
        if self.reason:
            result["reason"] = self.reason

        return result


__all__ = [
    "ComponentStatus",
    "ComponentResult",
]
