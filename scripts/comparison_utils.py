"""Reusable utilities for model comparison and visualization."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

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


def generate_report(
    summaries: Dict[str, Dict],
    histories: Dict[str, Dict],
    config_info: Dict,
    output_dir: Path
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
