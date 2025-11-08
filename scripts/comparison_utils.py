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


def plot_latent_by_component(
    models: Dict[str, object],
    X_data: np.ndarray,
    y_true: np.ndarray,
    output_dir: Path
):
    """Generate latent space scatter plots colored by component assignment.

    Only applicable for mixture priors with 2D latents.

    Args:
        models: Dictionary of model_name -> model
        X_data: Input data
        y_true: True labels
        output_dir: Directory to save plots
    """
    mixture_models = {
        name: model for name, model in models.items()
        if hasattr(model.config, 'prior_type') and model.config.prior_type == 'mixture'
        and model.config.latent_dim == 2
    }

    if not mixture_models:
        return

    n_models = len(mixture_models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_models > 1 else [axes]

    for idx, (model_name, model) in enumerate(mixture_models.items()):
        ax = axes[idx]

        # Load latent data with responsibilities
        diag_dir = model.last_diagnostics_dir
        if not diag_dir:
            continue

        try:
            latent_data = model._diagnostics.load_latent_data(diag_dir)
            if latent_data is None:
                continue

            z_mean = latent_data['z_mean']
            responsibilities = latent_data['q_c']

            # Component assignments
            component_assignments = responsibilities.argmax(axis=1)
            n_components = responsibilities.shape[1]

            # Use a colormap with enough distinct colors
            cmap = plt.cm.get_cmap('tab20' if n_components <= 20 else 'hsv')

            for c in range(n_components):
                mask = component_assignments == c
                if mask.sum() > 0:
                    color = cmap(c / n_components)
                    ax.scatter(z_mean[mask, 0], z_mean[mask, 1],
                              label=f'C{c}', alpha=0.6, s=20, color=color)

            ax.set_xlabel('Latent Dim 1')
            ax.set_ylabel('Latent Dim 2')
            ax.set_title(f'{model_name} (by Component)')
            ax.legend(title='Component', bbox_to_anchor=(1.05, 1), loc='upper left',
                     fontsize=8, ncol=1 if n_components <= 10 else 2)
            ax.grid(True, alpha=0.3)

        except Exception as e:
            print(f"Warning: Could not plot latent by component for {model_name}: {e}")

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    output_path = output_dir / 'latent_by_component.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_responsibility_histogram(
    models: Dict[str, object],
    output_dir: Path
):
    """Generate histogram of responsibility confidence (max_c q(c|x)).

    Args:
        models: Dictionary of model_name -> model
        output_dir: Directory to save plots
    """
    mixture_models = {
        name: model for name, model in models.items()
        if hasattr(model.config, 'prior_type') and model.config.prior_type == 'mixture'
    }

    if not mixture_models:
        return

    n_models = len(mixture_models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_models > 1 else [axes]

    for idx, (model_name, model) in enumerate(mixture_models.items()):
        ax = axes[idx]

        # Load latent data with responsibilities
        diag_dir = model.last_diagnostics_dir
        if not diag_dir:
            continue

        try:
            latent_data = model._diagnostics.load_latent_data(diag_dir)
            if latent_data is None:
                continue

            responsibilities = latent_data['q_c']
            max_responsibilities = responsibilities.max(axis=1)

            # Plot histogram
            ax.hist(max_responsibilities, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(max_responsibilities.mean(), color='red', linestyle='--',
                      linewidth=2, label=f'Mean: {max_responsibilities.mean():.3f}')

            ax.set_xlabel('max_c q(c|x)')
            ax.set_ylabel('Count')
            ax.set_title(f'{model_name}\nResponsibility Confidence')
            ax.legend()
            ax.grid(True, alpha=0.3)

        except Exception as e:
            print(f"Warning: Could not plot responsibility histogram for {model_name}: {e}")

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    output_path = output_dir / 'responsibility_histogram.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_mixture_evolution(
    models: Dict[str, object],
    output_dir: Path
):
    """Generate π and component usage evolution plots.

    Args:
        models: Dictionary of model_name -> model
        output_dir: Directory to save plots
    """
    mixture_models = {
        name: model for name, model in models.items()
        if hasattr(model.config, 'prior_type') and model.config.prior_type == 'mixture'
    }

    if not mixture_models:
        return

    # Create subdirectory for mixture evolution plots
    mixture_dir = output_dir / 'visualizations' / 'mixture'
    mixture_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in mixture_models.items():
        diag_dir = model.last_diagnostics_dir
        if not diag_dir:
            continue

        try:
            # Load history data
            pi_hist_path = Path(diag_dir) / "pi_history.npy"
            usage_hist_path = Path(diag_dir) / "usage_history.npy"
            epochs_path = Path(diag_dir) / "tracked_epochs.npy"

            if not pi_hist_path.exists() or not usage_hist_path.exists() or not epochs_path.exists():
                continue

            pi_history = np.load(pi_hist_path)  # Shape: (n_epochs, K)
            usage_history = np.load(usage_hist_path)  # Shape: (n_epochs, K)
            tracked_epochs = np.load(epochs_path)  # Shape: (n_epochs,)

            # Create figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            K = pi_history.shape[1]

            # π evolution
            for c in range(K):
                ax1.plot(tracked_epochs, pi_history[:, c], label=f'π_{c}', alpha=0.7, linewidth=1.5)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('π (Mixture Weight)')
            ax1.set_title(f'{model_name}: π Evolution')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2 if K > 10 else 1)
            ax1.grid(True, alpha=0.3)

            # Component usage evolution
            for c in range(K):
                ax2.plot(tracked_epochs, usage_history[:, c], label=f'C_{c}', alpha=0.7, linewidth=1.5)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Component Usage')
            ax2.set_title(f'{model_name}: Component Usage Evolution')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2 if K > 10 else 1)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save with sanitized model name
            safe_name = "".join(c.lower() if c.isalnum() else "_" for c in model_name).strip("_")
            output_path = mixture_dir / f'{safe_name}_evolution.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {output_path}")
            plt.close()

        except Exception as e:
            print(f"Warning: Could not plot mixture evolution for {model_name}: {e}")


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
        mixture_dir = output_dir / 'visualizations' / 'mixture'
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
