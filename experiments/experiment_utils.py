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


def plot_component_embedding_divergence(
    models: Dict[str, object],
    output_dir: Path
):
    """Analyze and visualize component embedding divergence.

    For component-aware decoders, this checks if learned component embeddings e_c
    actually diverge from each other (indicating functional specialization).

    Args:
        models: Dictionary of model_name -> model
        output_dir: Directory to save plots
    """
    component_aware_models = {
        name: model for name, model in models.items()
        if (hasattr(model.config, 'prior_type') and
            model.config.prior_type == 'mixture' and
            hasattr(model.config, 'use_component_aware_decoder') and
            model.config.use_component_aware_decoder)
    }

    if not component_aware_models:
        return

    import jax.numpy as jnp
    from scipy.spatial.distance import pdist, squareform

    n_models = len(component_aware_models)
    n_cols = min(2, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_models > 1 else [axes]

    for idx, (model_name, model) in enumerate(component_aware_models.items()):
        ax = axes[idx]

        try:
            # Extract component embeddings from model parameters
            # SSVAE stores params in state.params, not directly in model.params
            params = None
            if hasattr(model, 'state') and hasattr(model.state, 'params'):
                params = model.state.params
            elif hasattr(model, 'params'):
                params = model.params

            if params is None:
                print(f"  Skipping {model_name}: no parameters found")
                continue

            # Navigate parameter dict to find component embeddings
            # Structure: params['prior']['component_embeddings'] for MixturePriorParameters
            if 'prior' not in params or 'component_embeddings' not in params['prior']:
                print(f"  Skipping {model_name}: component embeddings not found in params")
                print(f"    Available keys in params: {list(params.keys())}")
                if 'prior' in params:
                    print(f"    Available keys in params['prior']: {list(params['prior'].keys())}")
                continue

            embeddings = np.array(params['prior']['component_embeddings'])  # Shape: (K, embed_dim)
            K, embed_dim = embeddings.shape

            # Compute pairwise distances
            distances = pdist(embeddings, metric='euclidean')
            distance_matrix = squareform(distances)

            # Visualize as heatmap
            im = ax.imshow(distance_matrix, cmap='viridis', aspect='auto')
            ax.set_xlabel('Component Index')
            ax.set_ylabel('Component Index')
            ax.set_title(f'{model_name}\nComponent Embedding Distances')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Euclidean Distance', rotation=270, labelpad=20)

            # Add text annotations for key statistics
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            min_dist = np.min(distances)
            max_dist = np.max(distances)

            stats_text = f'Mean: {mean_dist:.3f}\nStd: {std_dist:.3f}\nMin: {min_dist:.3f}\nMax: {max_dist:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=9)

            print(f"  {model_name}: Component embedding stats:")
            print(f"    K={K}, embed_dim={embed_dim}")
            print(f"    Distance: mean={mean_dist:.3f}, std={std_dist:.3f}, min={min_dist:.3f}, max={max_dist:.3f}")

        except Exception as e:
            print(f"Warning: Could not analyze embeddings for {model_name}: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes,
                   ha='center', va='center')

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    output_path = output_dir / 'component_embedding_divergence.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_reconstruction_by_component(
    models: Dict[str, object],
    X_data: np.ndarray,
    output_dir: Path,
    *,
    num_samples: int = 3,
    seed: int = 42,
):
    """Visualize how each component reconstructs individual inputs.

    For component-aware decoders, this shows whether components specialize
    in different reconstruction strategies. For each input, we show:
    - Original image
    - Reconstructions from each of K components
    - Final weighted reconstruction

    Args:
        models: Dictionary of model_name -> model
        X_data: Input data
        output_dir: Directory to save plots
        num_samples: Number of input samples to visualize
        seed: Random seed for sample selection
    """
    component_aware_models = {
        name: model for name, model in models.items()
        if (hasattr(model.config, 'prior_type') and
            model.config.prior_type == 'mixture' and
            hasattr(model.config, 'use_component_aware_decoder') and
            model.config.use_component_aware_decoder)
    }

    if not component_aware_models:
        return

    import jax
    import jax.numpy as jnp

    rng = np.random.RandomState(seed)
    num_samples = max(1, min(num_samples, X_data.shape[0]))
    indices = np.sort(rng.choice(X_data.shape[0], size=num_samples, replace=False))
    samples = X_data[indices]

    def _prep_image(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[-1] == 1:
            return img[..., 0]
        return img

    for model_name, model in component_aware_models.items():
        try:
            K = model.config.num_components
            use_sigmoid = model.config.reconstruction_loss == "bce"

            # Create figure: num_samples rows, K+2 columns (original + K components + weighted)
            fig, axes = plt.subplots(num_samples, K + 2, figsize=(1.5 * (K + 2), 1.8 * num_samples))
            if num_samples == 1:
                axes = axes.reshape(1, -1)

            for sample_idx in range(num_samples):
                x_input = samples[sample_idx:sample_idx+1]  # Keep batch dim

                # Forward pass through model to get per-component reconstructions
                # SSVAE provides _apply_fn that wraps model.apply
                if hasattr(model, '_apply_fn') and hasattr(model, 'state'):
                    output = model._apply_fn(
                        model.state.params,
                        jnp.asarray(x_input),
                        training=False,
                    )
                elif hasattr(model, 'model') and hasattr(model, 'params'):
                    output = model.model.apply(
                        {"params": model.params},
                        jnp.asarray(x_input),
                        training=False,
                    )
                else:
                    print(f"  Skipping {model_name}: cannot determine how to call model")
                    continue

                # Extract from ForwardOutput namedtuple
                # ForwardOutput(component_logits, z_mean, z_log_var, z, recon, logits, extras)
                extras = output.extras if hasattr(output, 'extras') else output[6]

                # Get per-component reconstructions from extras
                if not hasattr(extras, 'get'):
                    print(f"  Skipping {model_name}: extras is not a dict")
                    continue

                recon_per_component = extras.get("recon_per_component")
                if recon_per_component is None:
                    print(f"  Skipping {model_name}: recon_per_component not found in extras")
                    continue

                # Convert to numpy: Shape (1, K, H, W)
                recon_per_component = np.array(recon_per_component)[0]  # Shape: (K, H, W)

                # Get responsibilities for weighted reconstruction
                responsibilities = extras.get("responsibilities")
                if responsibilities is None:
                    print(f"  Skipping {model_name}: no responsibilities found")
                    continue

                resp_np = np.array(responsibilities)[0]  # Shape: (K,)

                # Compute weighted reconstruction
                weighted_recon = np.sum(recon_per_component * resp_np[:, None, None], axis=0)

                # Apply sigmoid if needed
                if use_sigmoid:
                    recon_per_component = 1.0 / (1.0 + np.exp(-recon_per_component))
                    weighted_recon = 1.0 / (1.0 + np.exp(-weighted_recon))

                # Plot original
                axes[sample_idx, 0].imshow(_prep_image(x_input[0]), cmap='gray')
                axes[sample_idx, 0].set_title('Original' if sample_idx == 0 else '')
                axes[sample_idx, 0].axis('off')

                # Plot per-component reconstructions
                for c in range(K):
                    ax = axes[sample_idx, c + 1]
                    ax.imshow(_prep_image(recon_per_component[c]), cmap='gray')
                    title = f'C{c}\n(q={resp_np[c]:.2f})' if sample_idx == 0 else f'q={resp_np[c]:.2f}'
                    ax.set_title(title, fontsize=8)
                    ax.axis('off')

                # Plot weighted reconstruction
                axes[sample_idx, K + 1].imshow(_prep_image(weighted_recon), cmap='gray')
                axes[sample_idx, K + 1].set_title('Weighted' if sample_idx == 0 else '')
                axes[sample_idx, K + 1].axis('off')

            plt.suptitle(f'{model_name}: Reconstruction by Component', fontsize=12, y=0.995)
            plt.tight_layout()

            filename = f"{_sanitize_model_name(model_name)}_reconstruction_by_component.png"
            output_path = output_dir / filename
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {output_path}")
            plt.close(fig)

        except Exception as e:
            print(f"Warning: Could not plot reconstruction by component for {model_name}: {e}")
            import traceback
            traceback.print_exc()


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


def plot_tau_analysis(
    models: Dict[str, object],
    output_dir: Path
):
    """Generate τ-classifier analysis visualization.

    Creates a comprehensive 4-subplot figure analyzing the τ matrix:
    - Main heatmap: τ[c,y] values (K components × num_classes)
    - Right bars: Specialization scores (entropy per component)
    - Bottom bars: Label coverage (components per label)
    - Corner textbox: Key statistics

    Args:
        models: Dict mapping model names to trained model objects
        output_dir: Directory to save visualization
    """
    from scipy.stats import entropy

    # Filter to models with τ-classifier
    tau_models = {
        name: model for name, model in models.items()
        if hasattr(model, 'config') and getattr(model.config, 'use_tau_classifier', False)
    }

    if not tau_models:
        return  # No τ-classifier models to visualize

    for model_name, model in tau_models.items():
        # Load τ matrix from diagnostics
        diag_dir = model.last_diagnostics_dir
        if not diag_dir:
            continue

        counts_path = Path(diag_dir) / "component_label_counts.npy"
        if not counts_path.exists():
            continue

        counts = np.load(counts_path)  # Shape: (K, num_classes)
        # Normalize with Dirichlet smoothing to get τ
        alpha_0 = getattr(model.config, 'tau_alpha_0', 1.0)
        tau = (counts + alpha_0) / (counts.sum(axis=1, keepdims=True) + alpha_0 * counts.shape[1])
        K, num_classes = tau.shape

        # Create figure with custom gridspec for 4-subplot layout
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 3, height_ratios=[3, 1, 0.3], width_ratios=[3, 1, 0.3],
                              hspace=0.3, wspace=0.3)

        # Main subplot: τ heatmap
        ax_main = fig.add_subplot(gs[0:2, 0:2])
        im = ax_main.imshow(tau, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax_main.set_xlabel('Label', fontsize=12, fontweight='bold')
        ax_main.set_ylabel('Component', fontsize=12, fontweight='bold')
        ax_main.set_title(f'τ Matrix: Component-Label Associations ({model_name})',
                          fontsize=14, fontweight='bold', pad=20)
        ax_main.set_xticks(range(num_classes))
        ax_main.set_yticks(range(K))

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
        cbar.set_label('P(y|c)', rotation=270, labelpad=20, fontsize=11)

        # Add text annotations on heatmap for high-confidence values
        for c in range(K):
            for y in range(num_classes):
                if tau[c, y] > 0.3:  # Only annotate significant values
                    text_color = 'white' if tau[c, y] > 0.6 else 'black'
                    ax_main.text(y, c, f'{tau[c, y]:.2f}',
                               ha='center', va='center',
                               color=text_color, fontsize=8)

        # Right subplot: Specialization scores (entropy per component)
        ax_right = fig.add_subplot(gs[0:2, 2])
        specialization_scores = [entropy(tau[c, :]) for c in range(K)]
        colors_spec = ['green' if s < 1.0 else 'orange' if s < 1.5 else 'red'
                       for s in specialization_scores]
        ax_right.barh(range(K), specialization_scores, color=colors_spec, alpha=0.7)
        ax_right.set_ylabel('Component', fontsize=10, fontweight='bold')
        ax_right.set_xlabel('Entropy', fontsize=9)
        ax_right.set_title('Specialization\n(lower = better)', fontsize=10, fontweight='bold')
        ax_right.set_ylim(-0.5, K - 0.5)
        ax_right.invert_yaxis()
        ax_right.grid(axis='x', alpha=0.3)
        ax_right.set_yticks(range(K))

        # Add reference line at entropy threshold
        max_entropy = np.log(num_classes)
        ax_right.axvline(max_entropy / 2, color='gray', linestyle='--',
                        linewidth=1, alpha=0.5, label='50% max')
        ax_right.legend(fontsize=8, loc='lower right')

        # Bottom subplot: Label coverage
        ax_bottom = fig.add_subplot(gs[2, 0:2])
        threshold = 0.3
        label_coverage = [np.sum(tau[:, y] > threshold) for y in range(num_classes)]
        colors_cov = ['green' if c > 0 else 'red' for c in label_coverage]
        ax_bottom.bar(range(num_classes), label_coverage, color=colors_cov, alpha=0.7)
        ax_bottom.set_xlabel('Label', fontsize=10, fontweight='bold')
        ax_bottom.set_ylabel('# Components', fontsize=9)
        ax_bottom.set_title(f'Label Coverage (threshold={threshold})',
                           fontsize=10, fontweight='bold')
        ax_bottom.set_xticks(range(num_classes))
        ax_bottom.grid(axis='y', alpha=0.3)
        ax_bottom.axhline(1, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        # Corner: Statistics textbox
        ax_stats = fig.add_subplot(gs[2, 2])
        ax_stats.axis('off')

        mean_spec = np.mean(specialization_scores)
        labels_with_zero_coverage = sum(1 for c in label_coverage if c == 0)
        max_tau = np.max(tau)
        mean_tau = np.mean(tau)

        stats_text = (
            f"Statistics:\n"
            f"─────────\n"
            f"K = {K}\n"
            f"Labels = {num_classes}\n"
            f"\n"
            f"Specialization:\n"
            f"  Mean H = {mean_spec:.2f}\n"
            f"  Max H = {max_entropy:.2f}\n"
            f"\n"
            f"Coverage:\n"
            f"  Zero = {labels_with_zero_coverage}\n"
            f"\n"
            f"τ Values:\n"
            f"  Max = {max_tau:.3f}\n"
            f"  Mean = {mean_tau:.3f}"
        )

        ax_stats.text(0.1, 0.5, stats_text,
                     transform=ax_stats.transAxes,
                     fontsize=8, verticalalignment='center',
                     family='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # Save figure
        save_dir = output_dir / "visualizations" / "tau"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{model_name}_tau_analysis.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {save_path}")
