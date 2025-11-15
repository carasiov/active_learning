"""Compatibility shim for the old callback path."""
from use_cases.dashboard.utils.training_callback import DashboardMetricsCallback, TrainingStoppedException

__all__ = ["DashboardMetricsCallback", "TrainingStoppedException"]
