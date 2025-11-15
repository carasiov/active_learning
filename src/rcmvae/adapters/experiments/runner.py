"""Adapter that wires ExperimentService outputs into metrics + visualization."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from infrastructure.metrics import MetricContext, collect_metrics
from infrastructure.visualization import VisualizationContext, render_all_plots
from infrastructure.metrics.providers import defaults as _  # noqa: F401
from rcmvae.application.services.experiment_service import ExperimentService
from rcmvae.application.model_api import SSVAE
from rcmvae.domain.config import SSVAEConfig
from use_cases.experiments.src.structure import RunPaths


def _maybe_tuple_hidden_dims(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure hidden_dims uses tuples (JAX-friendly) for component builders."""
    if isinstance(model_config.get("hidden_dims"), list):
        model_config = {**model_config, "hidden_dims": tuple(model_config["hidden_dims"])}
    return model_config


def run_training_pipeline(
    model_config: Dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    y_true: np.ndarray,
    run_paths: RunPaths,
) -> Tuple[SSVAE, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Train the model, collect experiment metrics, and render plots."""
    model_config = _maybe_tuple_hidden_dims(model_config)
    config = SSVAEConfig(**model_config)

    service = ExperimentService(input_dim=(28, 28))
    artifacts = service.run(
        config=config,
        x_train=x_train,
        y_train=y_train,
        weights_path=run_paths.artifacts / "checkpoint.ckpt",
    )
    model = artifacts.model

    metric_context = MetricContext(
        model=model,
        config=config,
        history=artifacts.history,
        x_train=x_train,
        y_true=y_true,
        y_semi=y_train,
        latent=artifacts.latent,
        reconstructions=artifacts.reconstructions,
        predictions=artifacts.predictions,
        certainty=artifacts.certainty,
        responsibilities=artifacts.responsibilities,
        pi_values=artifacts.pi_values,
        train_time=artifacts.train_time,
        diagnostics_dir=Path(artifacts.diagnostics_dir) if artifacts.diagnostics_dir else None,
    )
    summary = collect_metrics(metric_context)

    viz_context = VisualizationContext(
        model=model,
        config=config,
        history=artifacts.history,
        x_train=x_train,
        y_true=y_true,
        figures_dir=run_paths.figures,
    )
    visualization_meta = render_all_plots(viz_context)

    return model, artifacts.history, summary, visualization_meta
