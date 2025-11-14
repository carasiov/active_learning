"""Status objects for robust metric computation.

This module provides explicit status tracking for metric providers, replacing
the previous pattern of returning None (silent failure) with structured
status objects that communicate why a metric was not computed.

IMPORTANT: This module is now a backward compatibility wrapper. The actual
implementation has been moved to common.status for reuse across the codebase
(including visualization, metrics, and future infrastructure components).

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

# Re-export from common.status for backward compatibility
from common.status import ComponentResult, ComponentStatus

__all__ = ["ComponentResult", "ComponentStatus"]
