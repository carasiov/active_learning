"""Backward-compatible shim for the experiment runner adapter."""
from rcmvae.adapters.experiments.runner import run_training_pipeline

__all__ = ["run_training_pipeline"]
