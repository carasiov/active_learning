"""Training orchestration that feeds metrics and visualization registries."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from ssvae import SSVAE, SSVAEConfig

from ..io import RunPaths
from ..metrics import MetricContext, collect_metrics
from visualization import VisualizationContext, render_all_plots


def _maybe_tuple_hidden_dims(model_config: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(model_config.get("hidden_dims"), list):
        model_config = {**model_config, "hidden_dims": tuple(model_config["hidden_dims"])}
    return model_config


def run_training_pipeline(
    model_config: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_true: np.ndarray,
    run_paths: RunPaths,
) -> Tuple[SSVAE, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Train the model, collect metrics, and render visualizations."""
    model_config = _maybe_tuple_hidden_dims(model_config)
    config = SSVAEConfig(**model_config)
    model = SSVAE(input_dim=(28, 28), config=config)

    weights_path = run_paths.artifacts / "checkpoint.ckpt"
    history = {}

    # Train model (header shown in CLI)
    start = time.time()
    history = model.fit(X_train, y_train, weights_path=str(weights_path))
    train_time = time.time() - start
    print(f"Training complete in {train_time:.1f}s")

    # Use batched prediction to avoid OOM with convolutional architectures
    latent, recon, predictions, certainty = model.predict_batched(X_train)
    responsibilities = None
    pi_values = None
    if getattr(config, "prior_type", "standard") == "mixture":
        try:
            (
                _,
                _,
                _,
                _,
                responsibilities,
                pi_values,
            ) = model.predict_batched(X_train, return_mixture=True)
        except TypeError:
            responsibilities = None
            pi_values = None

    diagnostics_dir = Path(model.last_diagnostics_dir) if model.last_diagnostics_dir else None

    metric_context = MetricContext(
        model=model,
        config=config,
        history=history,
        x_train=X_train,
        y_true=y_true,
        y_semi=y_train,
        latent=latent,
        reconstructions=recon,
        predictions=predictions,
        certainty=certainty,
        responsibilities=responsibilities,
        pi_values=pi_values,
        train_time=train_time,
        diagnostics_dir=diagnostics_dir,
    )
    summary = collect_metrics(metric_context)

    viz_context = VisualizationContext(
        model=model,
        config=config,
        history=history,
        x_train=X_train,
        y_true=y_true,
        figures_dir=run_paths.figures,
    )
    visualization_meta = render_all_plots(viz_context)

    return model, history, summary, visualization_meta
