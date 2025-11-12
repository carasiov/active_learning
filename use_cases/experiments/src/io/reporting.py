"""Reporting helpers for experiment runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional

from .structure import RunPaths


def write_config_copy(config: Mapping, run_paths: RunPaths) -> Path:
    """Persist the resolved configuration for provenance."""
    try:
        import yaml  # type: ignore
    except ImportError:  # pragma: no cover
        print("Warning: PyYAML missing; skipping config snapshot.")
        return run_paths.config

    with run_paths.config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    print(f"Config saved: {run_paths.config}")
    return run_paths.config


def write_summary(summary: Mapping, run_paths: RunPaths) -> Path:
    with run_paths.summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"  Saved summary: {run_paths.summary}")
    return run_paths.summary


def write_report(
    summary: Mapping,
    history: Mapping,
    experiment_config: Mapping,
    run_paths: RunPaths,
    recon_paths: Optional[Dict[str, str]] = None,
    plot_status: Optional[Mapping] = None,
) -> Path:
    """Render a single-run markdown report referencing figures/ subdir.

    Args:
        summary: Experiment summary metrics
        history: Training history
        experiment_config: Experiment configuration
        run_paths: Run directory paths
        recon_paths: Optional reconstruction file paths
        plot_status: Optional plot generation status (from Phase 4)
    """
    report_path = run_paths.report
    figures_rel = Path("figures")

    exp_meta = experiment_config.get("experiment", {})
    data_config = experiment_config.get("data", {})
    model_config = experiment_config.get("model", {})

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Experiment Report\n\n")
        if exp_meta.get("name"):
            handle.write(f"**Experiment:** {exp_meta['name']}\n\n")
        if exp_meta.get("description"):
            handle.write(f"**Description:** {exp_meta['description']}\n\n")
        if exp_meta.get("tags"):
            tags = ", ".join(exp_meta["tags"])
            handle.write(f"**Tags:** {tags}\n\n")

        timestamp = experiment_config.get("timestamp", "N/A")
        handle.write(f"**Generated:** {timestamp}\n\n")

        handle.write("## Configuration\n\n### Data\n\n")
        for key, value in data_config.items():
            handle.write(f"- {key}: {value}\n")

        handle.write("\n### Model\n\n")
        handle.write(f"- Prior: {model_config.get('prior_type', 'standard')}\n")
        handle.write(f"- Latent dim: {model_config.get('latent_dim', 2)}\n")
        handle.write(f"- Hidden dims: {model_config.get('hidden_dims', [256, 128, 64])}\n")
        if model_config.get("prior_type") == "mixture":
            handle.write(f"- Components (K): {model_config.get('num_components', 10)}\n")
        handle.write(f"- Reconstruction loss: {model_config.get('reconstruction_loss', 'mse')}\n")
        handle.write(f"- Learning rate: {model_config.get('learning_rate', 0.001)}\n")
        handle.write(f"- Batch size: {model_config.get('batch_size', 128)}\n")
        handle.write(f"- Max epochs: {model_config.get('max_epochs', 300)}\n")

        handle.write("\n## Results\n\n### Summary Metrics\n\n")
        handle.write("| Category | Metric | Value |\n")
        handle.write("|----------|--------|-------|\n")

        def _write_metric_rows(category: str, metrics: Mapping):
            for key, value in metrics.items():
                if key in {"pi_values", "component_usage", "components_per_label", "component_dominant_labels", "free_channels"}:
                    continue
                metric = key.replace("_", " ").replace("final ", "").title()
                if isinstance(value, float):
                    handle.write(f"| {category} | {metric} | {value:.4f} |\n")
                else:
                    handle.write(f"| {category} | {metric} | {value} |\n")

        _write_metric_rows("Training", summary.get("training", {}))
        _write_metric_rows("Classification", summary.get("classification", {}))
        if "mixture" in summary:
            _write_metric_rows("Mixture", summary["mixture"])
        if "clustering" in summary:
            _write_metric_rows("Clustering", summary["clustering"])
        if "tau_classifier" in summary:
            _write_metric_rows("τ-Classifier", summary["tau_classifier"])

        # Add plot status section (Phase 4)
        if plot_status:
            handle.write("\n### Visualization Status\n\n")
            handle.write("| Plot | Status | Details |\n")
            handle.write("|------|--------|----------|\n")

            for plot_name, status_info in plot_status.items():
                if isinstance(status_info, dict):
                    status = status_info.get("status", "unknown")
                    reason = status_info.get("reason", "")
                    legacy = status_info.get("legacy", False)

                    # Format status with indicator
                    if status == "success":
                        status_str = "✓ Success"
                    elif status == "disabled":
                        status_str = "○ Disabled"
                    elif status == "skipped":
                        status_str = "⊘ Skipped"
                    elif status == "failed":
                        status_str = "✗ Failed"
                    else:
                        status_str = status

                    if legacy:
                        status_str += " (legacy)"

                    # Format plot name
                    plot_display = plot_name.replace("_", " ").title()

                    handle.write(f"| {plot_display} | {status_str} | {reason} |\n")

        handle.write("\n## Visualizations\n\n")

        def _embed_if_exists(label: str, filename: str) -> None:
            file_path = run_paths.figures / filename
            if file_path.exists():
                rel = figures_rel / filename
                handle.write(f"### {label}\n\n")
                handle.write(f"![{label}]({rel.as_posix()})\n\n")

        _embed_if_exists("Loss Comparison", "loss_comparison.png")
        handle.write("### Latent Space\n\n")
        if (run_paths.figures / "latent_spaces.png").exists():
            handle.write("**By Class Label:**\n\n")
            handle.write("![Latent Spaces](figures/latent_spaces.png)\n\n")
        if (run_paths.figures / "latent_by_component.png").exists():
            handle.write("**By Component Assignment:**\n\n")
            handle.write("![Latent by Component](figures/latent_by_component.png)\n\n")

        if (run_paths.figures / "responsibility_histogram.png").exists():
            handle.write("### Responsibility Confidence\n\n")
            handle.write("Distribution of max_c q(c|x):\n\n")
            handle.write("![Responsibility Histogram](figures/responsibility_histogram.png)\n\n")

        if recon_paths:
            handle.write("### Reconstructions\n\n")
            for model_name, filename in recon_paths.items():
                rel = figures_rel / filename
                handle.write(f"**{model_name}**\n\n")
                handle.write(f"![Reconstructions]({rel.as_posix()})\n\n")

        mixture_dir = run_paths.figures / "mixture"
        if mixture_dir.exists():
            evolution_plots = sorted(mixture_dir.glob("*_evolution.png"))
            if evolution_plots:
                handle.write("### Mixture Evolution\n\n")
                for plot_path in evolution_plots:
                    rel = figures_rel / "mixture" / plot_path.name
                    handle.write(f"![Mixture Evolution]({rel.as_posix()})\n\n")

        if (run_paths.figures / "component_embedding_divergence.png").exists():
            handle.write("### Component Embedding Divergence\n\n")
            handle.write("![Component Embedding Divergence](figures/component_embedding_divergence.png)\n\n")

        recon_by_component = sorted(run_paths.figures.glob("*_reconstruction_by_component.png"))
        if recon_by_component:
            handle.write("### Reconstruction by Component\n\n")
            for plot_path in recon_by_component:
                rel = figures_rel / plot_path.name
                handle.write(f"![Reconstruction by Component]({rel.as_posix()})\n\n")

        if (run_paths.figures / "tau_matrix_heatmap.png").exists():
            handle.write("### τ Matrix (Component → Label Mapping)\n\n")
            handle.write("![τ Matrix Heatmap](figures/tau_matrix_heatmap.png)\n\n")

        if (run_paths.figures / "tau_per_class_accuracy.png").exists():
            handle.write("### Per-Class Accuracy\n\n")
            handle.write("![Per-Class Accuracy](figures/tau_per_class_accuracy.png)\n\n")

        if (run_paths.figures / "tau_certainty_analysis.png").exists():
            handle.write("### Certainty Calibration\n\n")
            handle.write("![Certainty Analysis](figures/tau_certainty_analysis.png)\n\n")

    print(f"  Saved report: {report_path}")
    return report_path
