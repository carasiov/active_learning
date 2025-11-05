"""Reusable utilities for model comparison and visualization."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")


def plot_loss_comparison(
    histories: Dict[str, Dict[str, List[float]]],
    output_dir: Path,
    metrics: List[tuple] = None
):
    """Generate multi-panel loss comparison plots."""
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
            if metric in history and len(history[metric]) > 0:
                epochs = range(1, len(history[metric]) + 1)
                ax.plot(epochs, history[metric], label=model_name, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = output_dir / 'loss_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_latent_spaces(
    models: Dict[str, object],
    X_data: np.ndarray,
    y_true: np.ndarray,
    output_dir: Path
):
    """Generate latent space scatter plots for all models."""
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
        
        # Get latent representations
        latent, _, _, _ = model.predict(X_data)
        
        # Plot each digit class
        for digit in range(10):
            mask = y_true == digit
            if mask.sum() > 0:
                ax.scatter(latent[mask, 0], latent[mask, 1], 
                          label=str(digit), alpha=0.6, s=20)
        
        ax.set_xlabel('Latent Dim 1')
        ax.set_ylabel('Latent Dim 2')
        ax.set_title(f'{model_name}')
        ax.legend(title='Digit', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = output_dir / 'latent_spaces.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def _sanitize_model_name(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_") or "model"


def plot_reconstructions(
    models: Dict[str, object],
    X_data: np.ndarray,
    output_dir: Path,
    *,
    num_samples: int = 8,
    seed: int = 0,
) -> Dict[str, str]:
    """Generate reconstruction grids (original vs. recon) for each model.

    Returns a mapping of model name to the relative image filename saved under ``output_dir``.
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

    def _prep_image(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[-1] == 1:
            return img[..., 0]
        return img

    for model_name, model in models.items():
        try:
            _, recon, _, _ = model.predict(samples)
        except TypeError:
            # Fall back if predict signature differs; try without unpacking extras.
            prediction = model.predict(samples)
            if isinstance(prediction, tuple) and len(prediction) >= 2:
                recon = prediction[1]
            else:
                raise

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
            original_ax.set_title("Original" if idx == 0 else "")
            original_ax.axis("off")

            recon_ax.imshow(_prep_image(recon[idx]), cmap="gray")
            recon_ax.set_title("Reconstruction" if idx == 0 else "")
            recon_ax.axis("off")

        plt.tight_layout()
        filename = f"{_sanitize_model_name(model_name)}_reconstructions.png"
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved[model_name] = filename
        print(f"  Saved: {output_path}")

    return saved


def generate_report(
    summaries: Dict[str, Dict],
    histories: Dict[str, Dict],
    config_info: Dict,
    output_dir: Path,
    recon_paths: Optional[Dict[str, str]] = None,
):
    """Generate comprehensive comparison report."""
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
        f.write("![Latent Spaces](latent_spaces.png)\n\n")

        if recon_paths:
            f.write("### Reconstructions\n\n")
            for model_name, filename in recon_paths.items():
                f.write(f"**{model_name}**\n\n")
                f.write(f"![{model_name} Reconstructions]({filename})\n\n")

        mixture_models = {name: summaries[name] for name in summaries if 'pi_values' in summaries[name] or 'component_usage' in summaries[name]}
        if mixture_models:
            f.write("### Mixture Diagnostics\n\n")
            for model_name, summary in mixture_models.items():
                f.write(f"**{model_name}**\n")
                if 'final_component_entropy' in summary:
                    f.write(f"- Final component entropy: {summary['final_component_entropy']:.4f}\n")
                if 'final_pi_entropy' in summary:
                    f.write(f"- Final π entropy: {summary['final_pi_entropy']:.4f}\n")
                if 'pi_max' in summary and 'pi_min' in summary:
                    f.write(f"- π max/min: {summary['pi_max']:.4f} / {summary['pi_min']:.4f} (argmax={summary.get('pi_argmax', '-')})\n")
                if 'pi_values' in summary:
                    pi_str = ", ".join(f"{v:.3f}" for v in summary['pi_values'])
                    f.write(f"- π values: [{pi_str}]\n")
                if 'component_usage' in summary:
                    usage = summary['component_usage']
                    usage_str = ", ".join(f"{v:.3f}" for v in usage)
                    f.write(f"- Component usage: [{usage_str}]\n")
                if 'diagnostics_path' in summary:
                    rel_path = Path(summary['diagnostics_path']).relative_to(output_dir)
                    f.write(f"- Diagnostics folder: `{rel_path}`\n")
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
        
        # Generate comparison table
        f.write("| Metric |")
        for model_name in summaries.keys():
            f.write(f" {model_name} |")
        f.write("\n")
        
        f.write("|--------|")
        for _ in summaries.keys():
            f.write("----------|")
        f.write("\n")
        
        # Get all metrics
        all_metrics = set()
        for summary in summaries.values():
            all_metrics.update(summary.keys())
        
        for metric in sorted(all_metrics):
            metric_display = metric.replace('_', ' ').title()
            f.write(f"| {metric_display} |")
            
            for model_name in summaries.keys():
                value = summaries[model_name].get(metric, '-')
                if isinstance(value, (int, float)):
                    f.write(f" {value:.4f} |")
                else:
                    f.write(f" {value} |")
            f.write("\n")
        
        f.write("\n## Conclusion\n\n")
        f.write(f"Compared {len(summaries)} model configurations. ")
        f.write("All models trained successfully and results are documented above.\n")
    
    print(f"  Saved: {report_path}")
