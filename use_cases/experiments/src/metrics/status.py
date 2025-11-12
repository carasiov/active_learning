"""Status objects for robust metric computation.

This module provides explicit status tracking for metric providers, replacing
the previous pattern of returning None (silent failure) with structured
status objects that communicate why a metric was not computed.

Design principle (from AGENTS.md):
- Components must NEVER return None without explanation
- Fail fast with clear error messages
- Enable vs. disabled vs. failed states are explicit

Usage:
    from .status import ComponentResult

    @register_metric
    def my_metric(context: MetricContext) -> ComponentResult:
        if not is_enabled(context.config):
            return ComponentResult.disabled(
                reason="Requires mixture-based prior"
            )

        try:
            data = compute_metric(context)
            return ComponentResult.success(data=data)
        except Exception as e:
            return ComponentResult.failed(
                reason="Computation failed",
                error=e
            )
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
        """Create a successful result with computed metrics.

        Args:
            data: Nested dict of metric keys and values

        Returns:
            ComponentResult with SUCCESS status

        Example:
            >>> ComponentResult.success(data={
            ...     "K_eff": 7.3,
            ...     "active_components": 8
            ... })
        """
        return cls(status=ComponentStatus.SUCCESS, data=data)

    @classmethod
    def disabled(cls, reason: str) -> ComponentResult:
        """Create a disabled result for components not enabled in config.

        Args:
            reason: Why the component is disabled (e.g., "use_tau_classifier=false")

        Returns:
            ComponentResult with DISABLED status

        Example:
            >>> ComponentResult.disabled(
            ...     reason="τ-classifier requires mixture-based prior"
            ... )
        """
        return cls(status=ComponentStatus.DISABLED, reason=reason)

    @classmethod
    def skipped(cls, reason: str) -> ComponentResult:
        """Create a skipped result for enabled components with unmet preconditions.

        Use this when a component is enabled but cannot run due to missing
        data or incompatible settings discovered at runtime.

        Args:
            reason: Why the component was skipped

        Returns:
            ComponentResult with SKIPPED status

        Example:
            >>> ComponentResult.skipped(
            ...     reason="Clustering metrics only computed for latent_dim=2"
            ... )
        """
        return cls(status=ComponentStatus.SKIPPED, reason=reason)

    @classmethod
    def failed(cls, reason: str, error: Exception) -> ComponentResult:
        """Create a failed result for computation errors.

        Args:
            reason: High-level description of the failure
            error: The original exception for debugging

        Returns:
            ComponentResult with FAILED status

        Example:
            >>> try:
            ...     result = compute_metric()
            ... except ValueError as e:
            ...     return ComponentResult.failed(
            ...         reason="Invalid configuration for metric",
            ...         error=e
            ...     )
        """
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
        """Convert to a dict suitable for summary.json.

        Returns:
            For SUCCESS: Returns the data dict directly
            For other statuses: Returns status info dict

        Examples:
            >>> ComponentResult.success({"K_eff": 7.3}).to_dict()
            {"K_eff": 7.3}

            >>> ComponentResult.disabled("not enabled").to_dict()
            {"status": "disabled", "reason": "not enabled"}
        """
        if self.is_success:
            return self.data or {}

        result = {"status": self.status.value}
        if self.reason:
            result["reason"] = self.reason
        return result
