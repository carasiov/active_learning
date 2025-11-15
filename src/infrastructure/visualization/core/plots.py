"""Core visualization functions for basic VAE diagnostics.

This module contains fundamental plotting functions used across all
VAE experiments: loss curves, latent space visualization, and reconstructions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from ..utils import (
    _sanitize_model_name,
    _prep_image,
    safe_save_plot,
    style_axes,
    LATENT_POINT_SIZE,
)


def plot_loss_comparison(
    histories: Dict[str, Dict[str, List[float]]],
    output_dir: Path,
    metrics: List[tuple] = None
):
    """Generate multi-panel loss comparison plots with train and validation curves.

    Creates a grid of subplots showing training dynamics for different loss
    components. Includes both training and validation curves to identify overfitting.

    Args:
        histories: Dictionary mapping model_name -> history_dict
        output_dir: Base directory for saving figures
        metrics: List of (metric_key, display_title) tuples. Defaults to standard VAE losses.

    Output:
        Saves figure to: output_dir/core/loss_comparison.png
    """
    if metrics is None:
        metrics = [
            ('loss', 'Total Loss'),
            ('reconstruction_loss', 'Reconstruction Loss'),
            ('kl_loss', 'KL Divergence'),
            ('classification_loss', 'Classification Loss'),
        ]

    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]

    for (metric, title), ax in zip(metrics, axes):
        for model_name, history in histories.items():
            # Plot training curve
            if metric in history and len(history[metric]) > 0:
                epochs = range(1, len(history[metric]) + 1)
                ax.plot(epochs, history[metric],
                       label=f'{model_name} (Train)',
                       linewidth=2, alpha=0.8, linestyle='-')

            # Plot validation curve
            val_metric = f'val_{metric}'
            if val_metric in history and len(history[val_metric]) > 0:
                epochs = range(1, len(history[val_metric]) + 1)
                ax.plot(epochs, history[val_metric],
                       label=f'{model_name} (Val)',
                       linewidth=2, alpha=0.8, linestyle='--')

        style_axes(ax, title=title, xlabel='Epoch', ylabel=title)
        ax.legend(loc='best', fontsize=9)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # Save to appropriate subdirectory
    core_dir = output_dir / 'core'
    core_dir.mkdir(parents=True, exist_ok=True)
    output_path = core_dir / 'loss_comparison.png'

    safe_save_plot(fig, output_path)


def plot_latent_spaces(
    models: Dict[str, object],
    X_data: np.ndarray,
    y_true: np.ndarray,
    output_dir: Path
):
    """Generate latent space scatter plots for all models.

    Visualizes the 2D latent space learned by each VAE, colored by class labels.
    Helps assess whether the model learns meaningful semantic structure.

    Args:
        models: Dictionary mapping model_name -> model object
        X_data: Input data to encode (N, ...)
        y_true: True class labels (N,)
        output_dir: Base directory for saving figures

    Output:
        Saves figure to: output_dir/core/latent_spaces.png

    Note:
        Only visualizes first 2 latent dimensions. For higher-dimensional latents,
        consider using PCA or t-SNE projections.
    """
    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_models > 1 else [axes]

    for idx, (model_name, model) in enumerate(models.items()):
        ax = axes[idx]

        # Get latent representations (batched to avoid OOM with conv architectures)
        latent, _, _, _ = model.predict_batched(X_data)

        # Plot each digit class
        for digit in range(10):
            mask = y_true == digit
            if mask.sum() > 0:
                ax.scatter(
                    latent[mask, 0],
                    latent[mask, 1],
                    label=str(digit),
                    alpha=0.75,
                    s=LATENT_POINT_SIZE,
                )

        style_axes(
            ax,
            title=f'{model_name}',
            xlabel='Latent Dim 1',
            ylabel='Latent Dim 2',
        )
        ax.legend(title='Digit', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # Save to appropriate subdirectory
    core_dir = output_dir / 'core'
    core_dir.mkdir(parents=True, exist_ok=True)
    output_path = core_dir / 'latent_spaces.png'

    safe_save_plot(fig, output_path)


def plot_reconstructions(
    models: Dict[str, object],
    X_data: np.ndarray,
    output_dir: Path,
    *,
    num_samples: int = 8,
    seed: int = 0,
) -> Dict[str, str]:
    """Generate reconstruction grids (original vs. recon) for each model.

    Shows how well each model can reconstruct random samples from the dataset.
    Useful for assessing reconstruction quality and identifying failure modes.

    Args:
        models: Dictionary mapping model_name -> model object
        X_data: Input data to reconstruct (N, ...)
        output_dir: Base directory for saving figures
        num_samples: Number of examples to visualize per model
        seed: Random seed for sample selection

    Returns:
        Dictionary mapping model_name -> relative_path_to_figure

    Output:
        Saves figures to: output_dir/core/{model_name}_reconstructions.png
    """
    if X_data.size == 0 or not models:
        return {}

    rng = np.random.RandomState(seed)
    num_samples = max(1, min(num_samples, X_data.shape[0]))
    if X_data.shape[0] <= num_samples:
        indices = np.arange(X_data.shape[0])
    else:
        indices = np.sort(rng.choice(X_data.shape[0], size=num_samples, replace=False))

    samples = X_data[indices]
    saved = {}

    for model_name, model in models.items():
        try:
            _, recon, _, _ = model.predict_batched(samples)
        except TypeError:
            # Fall back if predict signature differs; try without unpacking extras.
            prediction = model.predict_batched(samples)
            if isinstance(prediction, tuple) and len(prediction) >= 2:
                recon = prediction[1]
            else:
                raise

        # Handle heteroscedastic decoder (recon is tuple of (mean, sigma))
        if isinstance(recon, tuple):
            recon = recon[0]  # Use mean for visualization

        # For BCE, decoder outputs logits; map to probabilities for visualization.
        try:
            use_sigmoid = getattr(getattr(model, "config", None), "reconstruction_loss", "mse") == "bce"
        except Exception:
            use_sigmoid = False

        if use_sigmoid:
            # numerically stable-ish sigmoid for display
            recon = 1.0 / (1.0 + np.exp(-recon))

        fig, axes = plt.subplots(2, num_samples, figsize=(1.6 * num_samples, 3.2))
        if num_samples == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        for idx in range(num_samples):
            original_ax = axes[0, idx]
            recon_ax = axes[1, idx]
            original_ax.imshow(_prep_image(samples[idx]), cmap="gray")
            style_axes(original_ax, title="Original" if idx == 0 else "", grid=False)
            original_ax.set_xticks([])
            original_ax.set_yticks([])

            recon_ax.imshow(_prep_image(recon[idx]), cmap="gray")
            style_axes(recon_ax, title="Reconstruction" if idx == 0 else "", grid=False)
            recon_ax.set_xticks([])
            recon_ax.set_yticks([])

        plt.tight_layout()

        # Save to appropriate subdirectory
        core_dir = output_dir / 'core'
        core_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{_sanitize_model_name(model_name)}_reconstructions.png"
        output_path = core_dir / filename

        if safe_save_plot(fig, output_path):
            saved[model_name] = str(Path('core') / filename)  # Return relative path

    return saved


def generate_report(
    summaries: Dict[str, Dict],
    histories: Dict[str, Dict],
    config_info: Dict,
    output_dir: Path,
    recon_paths: Optional[Dict[str, str]] = None,
):
    """Generate comprehensive comparison report in Markdown format.

    Creates a human-readable summary of all experiment results including
    configurations, visualizations, and performance metrics.

    Args:
        summaries: Dictionary mapping model_name -> summary metrics
        histories: Dictionary mapping model_name -> training history
        config_info: Experiment configuration metadata
        output_dir: Base directory for saving report
        recon_paths: Optional mapping of model_name -> reconstruction figure paths

    Output:
        Saves report to: output_dir/COMPARISON_REPORT.md
    """
    report_path = output_dir / 'COMPARISON_REPORT.md'

    with open(report_path, 'w') as f:
        f.write("# Model Comparison Report\n\n")
        f.write(f"**Generated:** {config_info.get('timestamp', 'N/A')}\n\n")

        f.write("## Configuration\n\n")
        for key, value in config_info.items():
            if key not in ['models', 'timestamp']:
                f.write(f"- {key.replace('_', ' ').title()}: {value}\n")

        f.write("\n## Models Compared\n\n")
        for model_name, model_config in config_info.get('models', {}).items():
            f.write(f"### {model_name}\n\n")
            for key, value in model_config.items():
                f.write(f"- {key}: {value}\n")
            f.write("\n")

        f.write("## Results\n\n")
        f.write("### Loss Curves\n\n")
        f.write("![Loss Comparison](loss_comparison.png)\n\n")

        f.write("### Latent Spaces\n\n")
        f.write("**By Class Label:**\n\n")
        f.write("![Latent Spaces](latent_spaces.png)\n\n")

        # Check if latent_by_component.png exists
        if (output_dir / 'latent_by_component.png').exists():
            f.write("**By Component Assignment:**\n\n")
            f.write("![Latent by Component](latent_by_component.png)\n\n")

        # Check if responsibility histogram exists
        if (output_dir / 'responsibility_histogram.png').exists():
            f.write("### Responsibility Confidence\n\n")
            f.write("Distribution of max_c q(c|x) showing how confidently the encoder assigns components:\n\n")
            f.write("![Responsibility Histogram](responsibility_histogram.png)\n\n")

        if recon_paths:
            f.write("### Reconstructions\n\n")
            for model_name, filename in recon_paths.items():
                f.write(f"**{model_name}**\n\n")
                f.write(f"![{model_name} Reconstructions]({filename})\n\n")

        # Add mixture evolution plots section
        mixture_dir = output_dir / 'mixture'
        if mixture_dir.exists():
            evolution_plots = list(mixture_dir.glob('*_evolution.png'))
            if evolution_plots:
                f.write("### Mixture Evolution\n\n")
                f.write("π (mixture weights) and component usage evolution over training:\n\n")
                for plot_path in sorted(evolution_plots):
                    rel_path = plot_path.relative_to(output_dir)
                    model_name = plot_path.stem.replace('_evolution', '').replace('_', ' ').title()
                    f.write(f"**{model_name}:**\n\n")
                    f.write(f"![{model_name} Evolution]({rel_path})\n\n")

        mixture_models = {name: summaries[name] for name in summaries if 'mixture' in summaries[name]}
        if mixture_models:
            f.write("### Mixture Diagnostics Summary\n\n")
            for model_name, summary in mixture_models.items():
                mixture = summary.get('mixture', {})
                f.write(f"**{model_name}**\n")
                if 'K' in mixture:
                    f.write(f"- K (total components): {mixture['K']}\n")
                if 'K_eff' in mixture:
                    f.write(f"- K_eff (effective components): {mixture['K_eff']:.2f}\n")
                if 'active_components' in mixture:
                    f.write(f"- Active components (>1% usage): {mixture['active_components']}\n")
                if 'responsibility_confidence_mean' in mixture:
                    f.write(f"- Mean responsibility confidence: {mixture['responsibility_confidence_mean']:.3f}\n")
                if 'final_component_entropy' in mixture:
                    f.write(f"- Final component entropy: {mixture['final_component_entropy']:.4f}\n")
                if 'final_pi_entropy' in mixture:
                    f.write(f"- Final π entropy: {mixture['final_pi_entropy']:.4f}\n")
                if 'pi_max' in mixture and 'pi_min' in mixture:
                    f.write(f"- π range: [{mixture['pi_min']:.4f}, {mixture['pi_max']:.4f}] (argmax={mixture.get('pi_argmax', '-')})\n")
                if 'diagnostics_path' in summary:
                    try:
                        rel_path = Path(summary['diagnostics_path']).relative_to(output_dir)
                        f.write(f"- Diagnostics folder: `{rel_path}`\n")
                    except ValueError:
                        pass
                f.write("\n")

        # Check if any model has component entropy
        has_mixture = any('component_entropy' in h and len(h['component_entropy']) > 0 for h in histories.values())
        if has_mixture:
            f.write("### Component Metrics\n\n")
            for model_name, history in histories.items():
                if 'component_entropy' in history and len(history['component_entropy']) > 0:
                    final_entropy = history['component_entropy'][-1]
                    f.write(f"**{model_name}:** Final component entropy = {final_entropy:.4f}\n\n")

        f.write("## Metrics Summary\n\n")

        # Flatten nested dictionaries for table display
        def flatten_metrics(summary_dict, prefix=''):
            """Flatten nested dictionary into dot-separated keys."""
            flat = {}
            for key, value in summary_dict.items():
                new_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    flat.update(flatten_metrics(value, new_key))
                elif not isinstance(value, (list, tuple)):  # Skip lists/tuples
                    flat[new_key] = value
            return flat

        flattened_summaries = {
            name: flatten_metrics(summary)
            for name, summary in summaries.items()
        }

        # Get all metrics
        all_metrics = set()
        for flat_summary in flattened_summaries.values():
            all_metrics.update(flat_summary.keys())

        # Group metrics by category
        metric_groups = {
            'Training': [m for m in all_metrics if m.startswith('training.')],
            'Classification': [m for m in all_metrics if m.startswith('classification.')],
            'Mixture': [m for m in all_metrics if m.startswith('mixture.')],
            'Clustering': [m for m in all_metrics if m.startswith('clustering.')],
        }

        for group_name, metrics in metric_groups.items():
            if not metrics:
                continue

            f.write(f"### {group_name} Metrics\n\n")

            # Generate comparison table
            f.write("| Metric |")
            for model_name in summaries.keys():
                f.write(f" {model_name} |")
            f.write("\n")

            f.write("|--------|")
            for _ in summaries.keys():
                f.write("----------|")
            f.write("\n")

            for metric in sorted(metrics):
                # Remove prefix for display
                metric_display = metric.split('.', 1)[1].replace('_', ' ').title()
                f.write(f"| {metric_display} |")

                for model_name in summaries.keys():
                    value = flattened_summaries[model_name].get(metric, '-')
                    if isinstance(value, (int, float)):
                        f.write(f" {value:.4f} |")
                    else:
                        f.write(f" {value} |")
                f.write("\n")

            f.write("\n")

        f.write("\n## Conclusion\n\n")
        f.write(f"Compared {len(summaries)} model configurations. ")
        f.write("All models trained successfully and results are documented above.\n")

    print(f"  Saved: {report_path}")
