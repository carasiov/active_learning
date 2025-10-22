"""
Placeholder for Phase 2 dashboard training callback integration.

The real implementation will define ``DashboardMetricsCallback`` that
extends :class:`callbacks.TrainingCallback` and pushes metrics into the
dashboard queue. It is intentionally left minimal during Phase 1 so that
imports established in automated checks resolve successfully.
"""

from __future__ import annotations

from callbacks import TrainingCallback


class DashboardMetricsCallback(TrainingCallback):
    """Stub callback for dashboard metrics (implemented in Phase 2)."""

    def on_epoch_end(self, epoch, metrics, history, trainer):  # pragma: no cover
        raise NotImplementedError(
            "DashboardMetricsCallback will be implemented in Phase 2 of the dashboard."
        )
