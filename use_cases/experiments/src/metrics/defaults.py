"""Default metric providers mirroring the legacy single-script behavior."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ssvae.diagnostics import DiagnosticsCollector

from .registry import MetricContext, register_metric


def _final(history: Dict[str, list[float]], key: str, default: float = 0.0) -> float:
    values = history.get(key, [])
    return float(values[-1]) if values else float(default)


@register_metric
def training_metrics(context: MetricContext):
    history = context.history
    summary = {
        "training": {
            "final_loss": _final(history, "loss"),
            "final_recon_loss": _final(history, "reconstruction_loss"),
            "final_kl_z": _final(history, "kl_z"),
            "final_kl_c": _final(history, "kl_c"),
            "training_time_sec": float(context.train_time),
            "epochs_completed": len(history.get("loss", [])),
        }
    }
    return summary


@register_metric
def classification_metrics(context: MetricContext):
    accuracy = DiagnosticsCollector.compute_accuracy(
        context.predictions, context.y_true
    )
    return {
        "classification": {
            "final_accuracy": float(accuracy),
            "final_classification_loss": _final(
                context.history, "classification_loss"
            ),
        }
    }


@register_metric
def mixture_metrics(context: MetricContext):
    config = context.config
    if getattr(config, "prior_type", "standard") != "mixture":
        return None

    history = context.history
    mixture_summary = {
        "K": getattr(config, "num_components", 0),
        "final_component_entropy": _final(history, "component_entropy"),
        "final_pi_entropy": _final(history, "pi_entropy"),
    }

    mixture_stats = getattr(context.model, "mixture_metrics", None) or {}
    for key in [
        "K_eff",
        "active_components",
        "responsibility_confidence_mean",
        "component_majority_labels",
        "component_majority_confidence",
    ]:
        if key in mixture_stats:
            value = mixture_stats[key]
            mixture_summary[key] = (
                value.tolist() if isinstance(value, np.ndarray) else value
            )

    diag_dir: Optional[Path] = context.diagnostics_dir
    result = {"mixture": mixture_summary}
    if diag_dir and diag_dir.exists():
        result["diagnostics_path"] = str(diag_dir)
        pi_path = diag_dir / "pi.npy"
        usage_path = diag_dir / "component_usage.npy"
        if pi_path.exists():
            pi = np.load(pi_path)
            mixture_summary.update(
                {
                    "pi_max": float(np.max(pi)),
                    "pi_min": float(np.min(pi)),
                    "pi_argmax": int(np.argmax(pi)),
                    "pi_values": pi.tolist(),
                }
            )
        if usage_path.exists():
            usage = np.load(usage_path)
            mixture_summary["component_usage"] = usage.tolist()

    return result


@register_metric
def tau_classifier_metrics(context: MetricContext):
    tau_classifier = getattr(context.model, "_tau_classifier", None)
    if tau_classifier is None:
        return None

    tau = tau_classifier.get_tau()
    tau_metrics = {
        "tau_matrix_shape": list(tau.shape),
        "tau_sparsity": float(np.sum(tau < 0.05) / tau.size),
    }

    components_per_label = np.sum(tau > 0.15, axis=0)
    tau_metrics["components_per_label"] = components_per_label.tolist()
    tau_metrics["avg_components_per_label"] = float(np.mean(components_per_label))
    tau_metrics["label_coverage"] = int(np.sum(components_per_label > 0))
    tau_metrics["component_dominant_labels"] = np.argmax(tau, axis=1).tolist()

    certainty = context.certainty
    tau_metrics.update(
        {
            "certainty_mean": float(np.mean(certainty)),
            "certainty_std": float(np.std(certainty)),
            "certainty_min": float(np.min(certainty)),
            "certainty_max": float(np.max(certainty)),
        }
    )

    if context.responsibilities is not None:
        ood_scores = tau_classifier.get_ood_score(context.responsibilities)
        tau_metrics["ood_score_mean"] = float(np.mean(ood_scores))
        tau_metrics["ood_score_std"] = float(np.std(ood_scores))

    free_channels = tau_classifier.get_free_channels()
    tau_metrics["num_free_channels"] = int(len(free_channels))
    tau_metrics["free_channels"] = free_channels.tolist()

    return {"tau_classifier": tau_metrics}


@register_metric
def clustering_metrics(context: MetricContext):
    config = context.config
    if getattr(config, "latent_dim", 0) != 2:
        return None
    if getattr(config, "prior_type", "standard") != "mixture":
        return None

    diag_dir = context.diagnostics_dir
    if not diag_dir:
        return None

    latent_data = context.model._diagnostics.load_latent_data(diag_dir)  # type: ignore[attr-defined]
    if not latent_data:
        return None

    responsibilities = latent_data.get("q_c")
    labels = latent_data.get("labels")
    if responsibilities is None or labels is None:
        return None

    component_assignments = np.asarray(responsibilities).argmax(axis=1)
    clustering = DiagnosticsCollector.compute_clustering_metrics(
        component_assignments, labels
    )
    if clustering:
        return {"clustering": clustering}
    return None
