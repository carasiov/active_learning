"""Default metric providers mirroring the legacy single-script behavior.

Returns ComponentResult for explicit status tracking.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import jax.numpy as jnp  # type: ignore
from jax import random  # type: ignore

from rcmvae.application.services.diagnostics_service import DiagnosticsCollector

# Import infrastructure from metrics registry
from infrastructure.metrics import MetricContext, register_metric
# Import status directly from infrastructure
from infrastructure import ComponentResult


def _final(history: Dict[str, list[float]], key: str, default: float = 0.0) -> float:
    values = history.get(key, [])
    return float(values[-1]) if values else float(default)


@register_metric
def training_metrics(context: MetricContext) -> ComponentResult:
    """Compute core training metrics (always succeeds).

    Returns:
        ComponentResult with training metrics
    """
    try:
        history = context.history
        summary = {
            "training": {
                "final_loss": _final(history, "loss"),
                "final_recon_loss": _final(history, "reconstruction_loss"),
                "final_kl_z": _final(history, "kl_z"),
                "final_kl_c": _final(history, "kl_c"),
                "final_kl_c_logit_mog": _final(history, "kl_c_logit_mog"),
                "training_time_sec": float(context.train_time),
                "epochs_completed": len(history.get("loss", [])),
            }
        }
        return ComponentResult.success(data=summary)

    except Exception as e:
        return ComponentResult.failed(
            reason="Failed to compute training metrics",
            error=e,
        )


@register_metric
def classification_metrics(context: MetricContext) -> ComponentResult:
    """Compute classification metrics (always succeeds).

    Returns:
        ComponentResult with classification accuracy
    """
    try:
        accuracy = DiagnosticsCollector.compute_accuracy(
            context.predictions, context.y_true
        )
        return ComponentResult.success(data={
            "classification": {
                "final_accuracy": float(accuracy),
                "final_classification_loss": _final(
                    context.history, "classification_loss"
                ),
            }
        })

    except Exception as e:
        return ComponentResult.failed(
            reason="Failed to compute classification metrics",
            error=e,
        )


@register_metric
def mixture_metrics(context: MetricContext) -> ComponentResult:
    """Compute mixture-specific metrics.

    Returns:
        ComponentResult.disabled if prior is not mixture-based
        ComponentResult.success with mixture metrics otherwise
    """
    config = context.config
    prior_type = getattr(config, "prior_type", "standard")

    # Check if mixture-based prior is enabled
    is_mixture_based = hasattr(config, "is_mixture_based_prior") and config.is_mixture_based_prior()
    if not is_mixture_based and prior_type != "mixture":
        return ComponentResult.disabled(
            reason=f"Prior type is '{prior_type}' (requires 'mixture', 'vamp', or 'geometric_mog')"
        )

    try:
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

        return ComponentResult.success(data=result)

    except Exception as e:
        return ComponentResult.failed(
            reason="Failed to compute mixture metrics",
            error=e,
        )


@register_metric
def tau_classifier_metrics(context: MetricContext) -> ComponentResult:
    """Compute τ-classifier metrics.

    Returns:
        ComponentResult.disabled if τ-classifier is not enabled
        ComponentResult.success with τ metrics otherwise
    """
    tau_classifier = getattr(context.model, "_tau_classifier", None)

    if tau_classifier is None:
        config = context.config
        use_tau = getattr(config, "use_tau_classifier", False)
        if use_tau:
            return ComponentResult.skipped(
                reason="τ-classifier configured but not initialized (check prior type)"
            )
        else:
            return ComponentResult.disabled(
                reason="use_tau_classifier=false"
            )

    try:
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

        return ComponentResult.success(data={"tau_classifier": tau_metrics})

    except Exception as e:
        return ComponentResult.failed(
            reason="Failed to compute τ-classifier metrics",
            error=e,
        )


@register_metric
def routing_and_specialization_metrics(context: MetricContext) -> ComponentResult:
    """Routing hardness, ownership, and per-component KL summaries for mixture models."""
    config = context.config
    if not getattr(config, "is_mixture_based_prior", lambda: False)():
        return ComponentResult.disabled(reason="Requires mixture-based prior")

    try:
        data = np.asarray(context.x_train)
        if data.size == 0:
            return ComponentResult.skipped(reason="No training data available")
        max_points = min(1024, data.shape[0])
        data = data[:max_points]

        summary: Dict[str, Dict[str, float]] = {}

        # Soft responsibilities
        resp_soft_np = None
        try:
            _, _, _, _, resp_soft, _ = context.model.predict_batched(data, return_mixture=True)
            resp_soft_np = np.asarray(resp_soft) if resp_soft is not None else None
        except Exception:
            resp_soft_np = None

        hardness: Dict[str, float] = {}
        if resp_soft_np is not None and resp_soft_np.size > 0:
            hardness["mean_max_soft"] = float(np.mean(resp_soft_np.max(axis=1)))
            hardness["active_components_1pct"] = int(np.sum(resp_soft_np.mean(axis=0) > 0.01))

        # Gumbel-sampled routing hardness (single pass)
        hardness_gumbel = None
        try:
            forward = context.model._apply_fn(
                context.model.state.params,
                jnp.asarray(data),
                training=False,
                rngs={"gumbel": random.PRNGKey(0)},
            )
            extras = forward.extras if hasattr(forward, "extras") else forward[6]
            comp_sel = extras.get("component_selection") if hasattr(extras, "get") else None
            if comp_sel is not None:
                hardness_gumbel = float(np.mean(np.asarray(comp_sel).max(axis=1)))
        except Exception:
            hardness_gumbel = None

        if hardness_gumbel is not None:
            hardness["mean_max_gumbel"] = hardness_gumbel

        # Ownership diagonal strength (mean responsibility for each component's majority label)
        ownership_diag = None
        if resp_soft_np is not None and context.y_true.size > 0:
            labels = np.asarray(context.y_true)[:resp_soft_np.shape[0]]
            valid = ~np.isnan(labels)
            labels = labels[valid].astype(int)
            resp_f = resp_soft_np[valid]
            if labels.size > 0:
                num_components = resp_f.shape[1]
                num_classes = int(getattr(config, "num_classes", labels.max() + 1))
                ownership = np.zeros((num_components, num_classes), dtype=np.float32)
                for c in range(num_classes):
                    mask = labels == c
                    if mask.any():
                        ownership[:, c] = resp_f[mask].mean(axis=0)
                majority = ownership.argmax(axis=1)
                diag_vals = ownership[np.arange(num_components), majority]
                ownership_diag = float(diag_vals.mean())

        component_kl: Dict[str, float] = {}
        # Per-component KL (if per-component stats are available)
        try:
            kl_sums = None
            count = 0
            batch_size = 256
            for start in range(0, data.shape[0], batch_size):
                batch = data[start:start + batch_size]
                if batch.size == 0:
                    continue
                forward = context.model._apply_fn(
                    context.model.state.params,
                    jnp.asarray(batch),
                    training=False,
                    rngs={"gumbel": random.PRNGKey(start)},
                )
                extras = forward.extras if hasattr(forward, "extras") else forward[6]
                if not hasattr(extras, "get"):
                    continue
                z_mean = extras.get("z_mean_per_component")
                z_log_var = extras.get("z_log_var_per_component")
                if z_mean is None or z_log_var is None:
                    continue
                z_mean = np.asarray(z_mean)
                z_log_var = np.asarray(z_log_var)
                kl = -0.5 * (1.0 + z_log_var - np.square(z_mean) - np.exp(z_log_var))
                kl = kl.sum(axis=2)  # [B, K]
                if kl_sums is None:
                    kl_sums = np.zeros(kl.shape[1], dtype=np.float64)
                kl_sums += kl.sum(axis=0)
                count += kl.shape[0]
            if count > 0 and kl_sums is not None:
                kl_mean = kl_sums / float(count)
                component_kl = {
                    "component_kl_mean": kl_mean.tolist(),
                    "component_kl_max": float(np.max(kl_mean)),
                    "component_kl_min": float(np.min(kl_mean)),
                }
        except Exception:
            component_kl = {}

        if hardness:
            summary["routing"] = hardness
        if ownership_diag is not None:
            summary.setdefault("routing", {})["ownership_diagonal_mean"] = ownership_diag
        if component_kl:
            summary["component_kl"] = component_kl

        if not summary:
            return ComponentResult.skipped(reason="Routing metrics unavailable (missing responsibilities or per-component stats)")

        return ComponentResult.success(data={"routing_metrics": summary})

    except Exception as e:
        return ComponentResult.failed(
            reason="Failed to compute routing/specialization metrics",
            error=e,
        )


@register_metric
def clustering_metrics(context: MetricContext) -> ComponentResult:
    """Compute clustering alignment metrics (NMI, ARI).

    Only computed for 2D latent space with mixture-based prior.

    Returns:
        ComponentResult.disabled if latent_dim != 2 or not mixture prior
        ComponentResult.skipped if preconditions not met (missing data)
        ComponentResult.success with clustering metrics otherwise
    """
    config = context.config
    latent_dim = getattr(config, "latent_dim", 0)
    prior_type = getattr(config, "prior_type", "standard")

    # Check if clustering metrics are applicable
    if latent_dim != 2:
        return ComponentResult.disabled(
            reason=f"Clustering metrics only computed for latent_dim=2, got {latent_dim}"
        )

    is_mixture_based = hasattr(config, "is_mixture_based_prior") and config.is_mixture_based_prior()
    if not is_mixture_based and prior_type != "mixture":
        return ComponentResult.disabled(
            reason=f"Clustering metrics require mixture-based prior, got '{prior_type}'"
        )

    # Check if diagnostics directory exists
    diag_dir = context.diagnostics_dir
    if not diag_dir or not diag_dir.exists():
        return ComponentResult.skipped(
            reason="Diagnostics directory not available"
        )

    try:
        # Load latent data
        latent_data = context.model._diagnostics.load_latent_data(diag_dir)  # type: ignore[attr-defined]
        if not latent_data:
            return ComponentResult.skipped(
                reason="Latent data not found in diagnostics directory"
            )

        responsibilities = latent_data.get("q_c")
        labels = latent_data.get("labels")

        if responsibilities is None or labels is None:
            return ComponentResult.skipped(
                reason="Missing responsibilities or labels in latent data"
            )

        # Compute clustering metrics
        component_assignments = np.asarray(responsibilities).argmax(axis=1)
        clustering = DiagnosticsCollector.compute_clustering_metrics(
            component_assignments, labels
        )

        if not clustering:
            return ComponentResult.skipped(
                reason="Clustering computation returned no results"
            )

        return ComponentResult.success(data={"clustering": clustering})

    except Exception as e:
        return ComponentResult.failed(
            reason="Failed to compute clustering metrics",
            error=e,
        )
