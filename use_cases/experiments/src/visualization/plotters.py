"""Reusable utilities for model comparison and visualization."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .registry import VisualizationContext, register_plotter

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
        
        # Get latent representations (batched to avoid OOM with conv architectures)
        latent, _, _, _ = model.predict_batched(X_data)
        
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
    mixture_dir = output_dir / 'mixture'
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


def plot_tau_matrix_heatmap(
    models: Dict[str, object],
    output_dir: Path
):
    """Visualize τ matrix (components → labels mapping) for τ-classifier models.

    Shows the learned probability distribution τ_{c,y} indicating which components
    are associated with which labels.

    Args:
        models: Dictionary of model_name -> model
        output_dir: Directory to save plots
    """
    tau_models = {
        name: model for name, model in models.items()
        if hasattr(model, '_tau_classifier') and model._tau_classifier is not None
    }

    if not tau_models:
        return

    n_models = len(tau_models)
    n_cols = min(2, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_models > 1 else [axes]

    for idx, (model_name, model) in enumerate(tau_models.items()):
        ax = axes[idx]

        try:
            # Get τ matrix
            tau = model._tau_classifier.get_tau()  # Shape: (K, num_classes)
            K, num_classes = tau.shape

            # Plot heatmap
            im = ax.imshow(tau, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('P(label | component)', rotation=270, labelpad=20)

            # Labels
            ax.set_xlabel('Class Label')
            ax.set_ylabel('Component Index')
            ax.set_title(f'{model_name}: τ Matrix (Components → Labels)')

            # Add ticks
            ax.set_xticks(range(num_classes))
            ax.set_yticks(range(K))

            # Add text annotations for values > 0.15
            for c in range(K):
                for y in range(num_classes):
                    if tau[c, y] > 0.15:
                        text_color = 'white' if tau[c, y] > 0.5 else 'black'
                        ax.text(y, c, f'{tau[c, y]:.2f}',
                               ha='center', va='center',
                               color=text_color, fontsize=8)

            # Add dominant label markers
            dominant_labels = np.argmax(tau, axis=1)
            for c in range(K):
                y_dom = dominant_labels[c]
                ax.plot(y_dom, c, 'b*', markersize=10, markeredgecolor='blue',
                       markerfacecolor='none', markeredgewidth=2)

            # Compute and display metrics
            sparsity = np.sum(tau < 0.05) / tau.size
            components_per_label = np.sum(tau > 0.15, axis=0)
            avg_components = np.mean(components_per_label)

            stats_text = f'Sparsity: {sparsity:.2f}\nAvg comp/label: {avg_components:.1f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round',
                   facecolor='white', alpha=0.8), fontsize=9)

            print(f"  {model_name}: τ matrix shape={tau.shape}, sparsity={sparsity:.3f}, avg_comp/label={avg_components:.2f}")

        except Exception as e:
            print(f"Warning: Could not plot τ matrix for {model_name}: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes,
                   ha='center', va='center')

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    output_path = output_dir / 'tau_matrix_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_tau_per_class_accuracy(
    models: Dict[str, object],
    X_data: np.ndarray,
    y_true: np.ndarray,
    output_dir: Path
):
    """Plot per-class accuracy comparison for τ-classifier models.

    Args:
        models: Dictionary of model_name -> model
        X_data: Input data
        y_true: True labels
        output_dir: Directory to save plots
    """
    tau_models = {
        name: model for name, model in models.items()
        if hasattr(model, '_tau_classifier') and model._tau_classifier is not None
    }

    if not tau_models:
        return

    n_models = len(tau_models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for idx, (model_name, model) in enumerate(tau_models.items()):
        ax = axes[idx]

        try:
            # Get predictions
            _, _, predictions, _ = model.predict_batched(X_data)

            # Compute per-class accuracy
            num_classes = model.config.num_classes
            accuracies = []
            sample_counts = []

            for class_id in range(num_classes):
                mask = y_true == class_id
                if mask.sum() > 0:
                    class_acc = np.mean(predictions[mask] == y_true[mask])
                    accuracies.append(class_acc * 100)
                    sample_counts.append(mask.sum())
                else:
                    accuracies.append(0)
                    sample_counts.append(0)

            # Bar plot
            x = np.arange(num_classes)
            bars = ax.bar(x, accuracies, alpha=0.7, color='steelblue', edgecolor='black')

            # Color bars by accuracy
            for i, (bar, acc) in enumerate(zip(bars, accuracies)):
                if acc >= 80:
                    bar.set_color('green')
                    bar.set_alpha(0.7)
                elif acc >= 50:
                    bar.set_color('orange')
                    bar.set_alpha(0.7)
                elif acc > 0:
                    bar.set_color('red')
                    bar.set_alpha(0.7)
                else:
                    bar.set_color('gray')
                    bar.set_alpha(0.3)

            # Add sample counts as text
            for i, (acc, count) in enumerate(zip(accuracies, sample_counts)):
                if count > 0:
                    ax.text(i, acc + 3, f'n={count}', ha='center', fontsize=8)

            ax.set_xlabel('Class Label')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'{model_name}: Per-Class Accuracy')
            ax.set_xticks(x)
            ax.set_ylim([0, 105])
            ax.axhline(y=100 * np.mean(predictions == y_true), color='red',
                      linestyle='--', linewidth=2, label=f'Overall: {np.mean(predictions == y_true)*100:.1f}%')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            print(f"  {model_name}: Overall accuracy = {np.mean(predictions == y_true)*100:.1f}%")

        except Exception as e:
            print(f"Warning: Could not plot per-class accuracy for {model_name}: {e}")

    plt.tight_layout()
    output_path = output_dir / 'tau_per_class_accuracy.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_tau_certainty_analysis(
    models: Dict[str, object],
    X_data: np.ndarray,
    y_true: np.ndarray,
    output_dir: Path
):
    """Analyze certainty vs accuracy for τ-classifier models.

    Args:
        models: Dictionary of model_name -> model
        X_data: Input data
        y_true: True labels
        output_dir: Directory to save plots
    """
    tau_models = {
        name: model for name, model in models.items()
        if hasattr(model, '_tau_classifier') and model._tau_classifier is not None
    }

    if not tau_models:
        return

    n_models = len(tau_models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for idx, (model_name, model) in enumerate(tau_models.items()):
        ax = axes[idx]

        try:
            # Get predictions and certainty
            _, _, predictions, certainty = model.predict_batched(X_data)
            correct = (predictions == y_true).astype(float)

            # Scatter plot with binning
            n_bins = 20
            certainty_bins = np.linspace(certainty.min(), certainty.max(), n_bins + 1)
            bin_accuracies = []
            bin_centers = []
            bin_counts = []

            for i in range(n_bins):
                mask = (certainty >= certainty_bins[i]) & (certainty < certainty_bins[i+1])
                if mask.sum() > 0:
                    bin_accuracies.append(np.mean(correct[mask]))
                    bin_centers.append((certainty_bins[i] + certainty_bins[i+1]) / 2)
                    bin_counts.append(mask.sum())

            # Plot binned accuracy
            ax.scatter(bin_centers, bin_accuracies, s=[c*5 for c in bin_counts],
                      alpha=0.6, color='steelblue', edgecolor='black', linewidth=1.5)

            # Ideal line (perfect calibration)
            ideal_x = np.linspace(0, 1, 100)
            ax.plot(ideal_x, ideal_x, 'r--', linewidth=2, label='Perfect calibration', alpha=0.7)

            ax.set_xlabel('Certainty (max_c r_c × τ_c,y)')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{model_name}: Certainty vs Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

            # Add correlation coefficient
            if len(bin_centers) > 1:
                corr = np.corrcoef(bin_centers, bin_accuracies)[0, 1]
                ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round',
                       facecolor='white', alpha=0.8), fontsize=10)

            print(f"  {model_name}: Certainty range=[{certainty.min():.3f}, {certainty.max():.3f}], mean={certainty.mean():.3f}")

        except Exception as e:
            print(f"Warning: Could not plot certainty analysis for {model_name}: {e}")

    plt.tight_layout()
    output_path = output_dir / 'tau_certainty_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


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


# ---------------------------------------------------------------------------
# Registry bindings
# ---------------------------------------------------------------------------

def _single_model_dict(model):
    return {"Model": model}


def _single_history_dict(history):
    return {"Model": history}


@register_plotter
def loss_curves_plotter(context: VisualizationContext):
    plot_loss_comparison(_single_history_dict(context.history), context.figures_dir)


@register_plotter
def latent_space_plotter(context: VisualizationContext):
    plot_latent_spaces(_single_model_dict(context.model), context.x_train, context.y_true, context.figures_dir)


@register_plotter
def latent_by_component_plotter(context: VisualizationContext):
    if getattr(context.config, "prior_type", "standard") != "mixture":
        return None
    plot_latent_by_component(_single_model_dict(context.model), context.x_train, context.y_true, context.figures_dir)
    return None


@register_plotter
def reconstructions_plotter(context: VisualizationContext):
    paths = plot_reconstructions(_single_model_dict(context.model), context.x_train, context.figures_dir)
    return {"reconstructions": paths}


@register_plotter
def responsibility_histogram_plotter(context: VisualizationContext):
    if getattr(context.config, "prior_type", "standard") != "mixture":
        return None
    plot_responsibility_histogram(_single_model_dict(context.model), context.figures_dir)
    return None


@register_plotter
def mixture_evolution_plotter(context: VisualizationContext):
    if getattr(context.config, "prior_type", "standard") != "mixture":
        return None
    plot_mixture_evolution(_single_model_dict(context.model), context.figures_dir)
    return None


@register_plotter
def component_embedding_plotter(context: VisualizationContext):
    if getattr(context.config, "prior_type", "standard") != "mixture":
        return None
    if not getattr(context.config, "use_component_aware_decoder", False):
        return None
    plot_component_embedding_divergence(_single_model_dict(context.model), context.figures_dir)
    plot_reconstruction_by_component(_single_model_dict(context.model), context.x_train, context.figures_dir)
    return None


@register_plotter
def tau_matrix_plotter(context: VisualizationContext):
    if getattr(context.model, "_tau_classifier", None) is None:
        return None
    plot_tau_matrix_heatmap(_single_model_dict(context.model), context.figures_dir)
    plot_tau_per_class_accuracy(_single_model_dict(context.model), context.x_train, context.y_true, context.figures_dir)
    plot_tau_certainty_analysis(_single_model_dict(context.model), context.x_train, context.y_true, context.figures_dir)
    return None
