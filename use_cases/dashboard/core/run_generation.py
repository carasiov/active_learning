from __future__ import annotations

"""Utilities to persist dashboard training runs in the experiment catalog."""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from infrastructure.metrics import MetricContext, collect_metrics
from infrastructure.visualization import VisualizationContext, render_all_plots
from use_cases.experiments.src.naming import generate_architecture_code
from use_cases.experiments.src.reporting import write_config_copy, write_report, write_summary
from use_cases.experiments.src.structure import create_run_paths

from use_cases.dashboard.core.state_models import ModelState, TrainingHistory


def generate_dashboard_run(
    *,
    model_state: ModelState,
    latent: np.ndarray,
    reconstructed: np.ndarray | Tuple[np.ndarray, np.ndarray],
    pred_classes: np.ndarray,
    pred_certainty: np.ndarray,
    responsibilities: Optional[np.ndarray],
    pi_values: Optional[np.ndarray],
    train_time: Optional[float],
    label_version: int,
    epoch_offset: int,
    epochs_completed: int,
) -> Dict[str, Any]:
    """Create an experiment run artifact bundle for a dashboard training session."""

    config = model_state.config
    model = model_state.model
    metadata = model_state.metadata
    data_state = model_state.data
    history_state = model_state.history

    architecture_code = generate_architecture_code(config)
    run_name = metadata.name or model_state.model_id
    run_id, timestamp, run_paths = create_run_paths(run_name, architecture_code)

    config_snapshot = asdict(config)
    if isinstance(config_snapshot.get("hidden_dims"), tuple):
        config_snapshot["hidden_dims"] = list(config_snapshot["hidden_dims"])

    labeled_mask = ~np.isnan(data_state.labels)
    labeled_count = int(np.sum(labeled_mask))
    total_samples = int(data_state.x_train.shape[0])

    experiment_config: Dict[str, Any] = {
        "experiment": {
            "name": run_name,
            "description": "Dashboard retraining session",
            "tags": ["dashboard", metadata.dataset],
            "model_id": model_state.model_id,
        },
        "data": {
            "dataset": metadata.dataset,
            "total_samples": total_samples,
            "labeled_samples": labeled_count,
            "label_version": label_version,
            "dataset_seed": metadata.dataset_seed,
        },
        "model": config_snapshot,
        "timestamp": timestamp,
    }

    write_config_copy(experiment_config, run_paths)

    history_dict = _history_for_reporting(history_state)
    train_time_value = float(train_time) if train_time is not None else 0.0
    start_epoch = epoch_offset + 1 if epochs_completed > 0 else epoch_offset
    end_epoch = epoch_offset + epochs_completed

    x_train = np.asarray(data_state.x_train)
    y_true = np.asarray(data_state.true_labels)
    y_semi = np.asarray(data_state.labels)
    latent_arr = np.asarray(latent)
    recon_arr = _normalize_reconstruction(reconstructed)
    preds = np.asarray(pred_classes)
    certainty = np.asarray(pred_certainty)
    responsibilities_arr = np.asarray(responsibilities) if responsibilities is not None else None
    pi_arr = np.asarray(pi_values) if pi_values is not None else None

    diagnostics_dir = _collect_diagnostics_if_needed(
        model=model,
        config=config,
        x_train=x_train,
        labels=y_semi,
        output_dir=run_paths.artifacts_diagnostics,
    )

    metric_context = MetricContext(
        model=model,
        config=config,
        history=history_dict,
        x_train=x_train,
        y_true=y_true,
        y_semi=y_semi,
        latent=latent_arr,
        reconstructions=recon_arr,
        predictions=preds,
        certainty=certainty,
        responsibilities=responsibilities_arr,
        pi_values=pi_arr,
        train_time=train_time_value,
        diagnostics_dir=diagnostics_dir,
    )
    summary = collect_metrics(metric_context)

    summary.setdefault("metadata", {})
    summary["metadata"].update(
        {
            "run_id": run_id,
            "architecture_code": architecture_code,
            "timestamp": timestamp,
            "experiment_name": run_name,
            "model_id": model_state.model_id,
            "source": "dashboard",
            "dataset": metadata.dataset,
            "label_version": label_version,
            "train_time_sec": train_time_value,
            "labeled_samples": labeled_count,
            "total_samples": total_samples,
            "status": "complete",
            "start_epoch": start_epoch,
            "end_epoch": end_epoch,
            "epochs_completed": epochs_completed,
        }
    )

    summary_path = write_summary(summary, run_paths)

    viz_context = VisualizationContext(
        model=model,
        config=config,
        history=history_dict,
        x_train=x_train,
        y_true=y_true,
        figures_dir=run_paths.figures,
    )
    visualization_meta = render_all_plots(viz_context)

    recon_paths = visualization_meta.get("reconstructions") if isinstance(visualization_meta, dict) else None
    plot_status = visualization_meta.get("_plot_status") if isinstance(visualization_meta, dict) else None

    write_report(
        summary,
        history_dict,
        experiment_config,
        run_paths,
        recon_paths=recon_paths,
        plot_status=plot_status,
    )

    metrics_snapshot: Dict[str, float] = {}
    for key in (
        "loss",
        "val_loss",
        "reconstruction_loss",
        "val_reconstruction_loss",
        "kl_loss",
        "val_kl_loss",
        "classification_loss",
        "val_classification_loss",
    ):
        series = history_dict.get(key)
        if series:
            metrics_snapshot[key] = float(series[-1])

    record = {
        "run_id": run_id,
        "timestamp": timestamp,
        "architecture_code": architecture_code,
        "model_id": model_state.model_id,
        "model_name": metadata.name,
        "results_root": str(run_paths.root),
        "summary_path": str(summary_path),
        "report_path": str(run_paths.report),
        "config_path": str(run_paths.config),
        "label_version": label_version,
        "labeled_samples": labeled_count,
        "total_samples": total_samples,
        "train_time_sec": train_time_value,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "epochs_completed": epochs_completed,
        "metrics": metrics_snapshot,
    }

    return record


def _normalize_reconstruction(data: np.ndarray | Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    if isinstance(data, tuple):
        return np.asarray(data[0])
    return np.asarray(data)


def _history_for_reporting(history: TrainingHistory) -> Dict[str, Any]:
    return {
        "epochs": list(history.epochs),
        "loss": list(history.train_loss),
        "val_loss": list(history.val_loss),
        "reconstruction_loss": list(history.train_reconstruction_loss),
        "val_reconstruction_loss": list(history.val_reconstruction_loss),
        "kl_loss": list(history.train_kl_loss),
        "val_kl_loss": list(history.val_kl_loss),
        "classification_loss": list(history.train_classification_loss),
        "val_classification_loss": list(history.val_classification_loss),
        "component_entropy": [],
        "pi_entropy": [],
    }


def _collect_diagnostics_if_needed(
    *,
    model,
    config,
    x_train: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
) -> Optional[Path]:
    if not hasattr(config, "is_mixture_based_prior"):
        return None
    try:
        is_mixture = bool(config.is_mixture_based_prior())
    except Exception:
        is_mixture = False
    if not is_mixture:
        return None

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        collector = getattr(model, "_diagnostics", None)
        apply_fn = getattr(model, "_apply_fn", None)
        params = getattr(model.state, "params", None)
        if collector is None or apply_fn is None or params is None:
            return None

        batch_hint = getattr(config, "batch_size", 512) or 512
        try:
            batch_size = int(batch_hint)
        except (TypeError, ValueError):
            batch_size = 512
        metrics = collector.collect_mixture_stats(
            apply_fn=apply_fn,
            params=params,
            data=x_train,
            labels=labels,
            output_dir=output_dir,
            batch_size=max(1, min(batch_size, 1024)),
        )
        if metrics:
            setattr(model, "_mixture_metrics", metrics)
        return output_dir
    except Exception:
        return None