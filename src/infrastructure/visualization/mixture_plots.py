"""Mixture-specific visualization functions.

This module contains plotting functions specific to mixture prior VAEs,
including component analysis, responsibility visualization, and mixture dynamics.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns

from .plot_utils import (
    _sanitize_model_name,
    _build_label_palette,
    _downsample_points,
    _compute_limits,
    _extract_component_recon,
    _prep_image,
    safe_save_plot,
)

sns.set_style("whitegrid")


def plot_latent_by_component(
    models: Dict[str, object],
    X_data: np.ndarray,
    y_true: np.ndarray,
    output_dir: Path
):
    """Generate latent space scatter plots colored by component assignment.

    For mixture prior VAEs, shows how the encoder assigns data points to
    different mixture components in latent space. Useful for understanding
    component specialization and usage patterns.

    Args:
        models: Dictionary mapping model_name -> model object
        X_data: Input data (N, ...)
        y_true: True labels (N,)
        output_dir: Base directory for saving figures

    Output:
        Saves figure to: output_dir/mixture/latent_by_component.png

    Note:
        Only applicable for mixture priors with 2D latents.
    """
    mixture_models = {
        name: model for name, model in models.items()
        if model.config.is_mixture_based_prior()
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

        try:
            # Compute on full dataset instead of loading validation subset
            # This matches plot_latent_spaces and ensures all points are shown
            latent, _, _, _, responsibilities, _ = model.predict_batched(
                X_data, return_mixture=True
            )

            if responsibilities is None:
                print(f"Warning: Model {model_name} did not return responsibilities")
                continue

            # Component assignments
            component_assignments = responsibilities.argmax(axis=1)
            n_components = responsibilities.shape[1]

            # Use a colormap with enough distinct colors
            cmap = plt.cm.get_cmap('tab20' if n_components <= 20 else 'hsv')

            for c in range(n_components):
                mask = component_assignments == c
                if mask.sum() > 0:
                    color = cmap(c / n_components)
                    ax.scatter(latent[mask, 0], latent[mask, 1],
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

    # Save to appropriate subdirectory
    mixture_dir = output_dir / 'mixture'
    mixture_dir.mkdir(parents=True, exist_ok=True)
    output_path = mixture_dir / 'latent_by_component.png'

    safe_save_plot(fig, output_path)


def plot_channel_latent_responsibility(
    models: Dict[str, object],
    X_data: np.ndarray,
    y_true: np.ndarray,
    output_dir: Path,
    *,
    max_points: int = 20000,
    seed: int = 0,
) -> Dict[str, Dict[str, List[str]]]:
    """Render per-channel latent plots (color=label, alpha=responsibility).

    Creates detailed visualizations showing both class labels (via color) and
    component responsibilities (via alpha transparency) in latent space. Generates
    both a grid view of all components and individual per-component figures.

    Args:
        models: Dictionary mapping model_name -> model object
        X_data: Input data (N, ...)
        y_true: True labels (N,)
        output_dir: Base directory for saving figures
        max_points: Maximum points to plot (downsamples if exceeded)
        seed: Random seed for downsampling

    Returns:
        Dictionary mapping model_name -> {"grid": path, "channels": [paths]}

    Output:
        Saves figures to: output_dir/mixture/channel_latents/
    """
    channel_viz_models = {
        name: model
        for name, model in models.items()
        if hasattr(model, "config") and model.config.is_mixture_based_prior()
        and getattr(getattr(model, "config", None), "latent_dim", 2) == 2
    }

    if not channel_viz_models:
        return {}

    saved_paths: Dict[str, Dict[str, List[str]]] = {}
    y_array = np.asarray(y_true).astype(int)

    for model_name, model in channel_viz_models.items():
        try:
            latent, _, _, _, responsibilities, _ = model.predict_batched(
                X_data,
                return_mixture=True,
            )
        except Exception as exc:
            print(f"Warning: Could not retrieve mixture outputs for {model_name}: {exc}")
            continue

        if latent.shape[1] < 2 or responsibilities is None:
            print(f"Warning: Skipping channel latent plot for {model_name}: latent_dim < 2 or missing q(c|x)")
            continue

        latent_2d = np.asarray(latent)[:, :2]
        resp = np.asarray(responsibilities)
        if resp.shape[0] != latent_2d.shape[0]:
            print(f"Warning: Responsibilities size mismatch for {model_name}")
            continue

        latent_2d, resp, labels = _downsample_points(
            latent_2d,
            resp,
            y_array,
            max_points=max_points,
            seed=seed,
        )

        if labels.size == 0:
            print(f"Warning: No labels available for channel latent plot ({model_name})")
            continue

        max_label = labels.max()
        num_classes = int(getattr(model.config, "num_classes", max(max_label + 1, 1)))
        palette = _build_label_palette(num_classes)
        label_colors = palette[np.clip(labels, 0, num_classes - 1)].copy()
        invalid_mask = (labels < 0) | (labels >= num_classes)
        if invalid_mask.any():
            unknown_color = np.array([0.5, 0.5, 0.5, 1.0])
            label_colors[invalid_mask] = unknown_color

        channel_count = resp.shape[1]
        if channel_count == 0:
            continue

        safe_name = _sanitize_model_name(model_name)
        mixture_dir = output_dir / "mixture"
        channel_dir = mixture_dir / "channel_latents"
        if len(channel_viz_models) > 1:
            channel_dir = channel_dir / safe_name
        channel_dir.mkdir(parents=True, exist_ok=True)

        (x_lim, y_lim) = _compute_limits(latent_2d)
        legend_handles = [
            Patch(facecolor=palette[i], edgecolor="none", label=str(i))
            for i in range(num_classes)
        ]
        if invalid_mask.any():
            legend_handles.append(Patch(facecolor=[0.5, 0.5, 0.5, 1.0], edgecolor="none", label="Unknown"))

        def _draw_channel(ax: plt.Axes, channel_idx: int) -> None:
            rgba = label_colors.copy()
            rgba[:, 3] = np.clip(resp[:, channel_idx], 0.0, 1.0)
            ax.scatter(
                latent_2d[:, 0],
                latent_2d[:, 1],
                c=rgba,
                s=10,
                linewidths=0,
                edgecolors="none",
            )
            ax.set_title(f"Channel {channel_idx}", fontsize=9)
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("#f5f5f5")

        # Grid figure
        n_cols = min(5, max(1, channel_count))
        n_rows = math.ceil(channel_count / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if isinstance(axes, np.ndarray):
            axes_list = axes.flatten()
        else:
            axes_list = [axes]

        for idx in range(channel_count):
            _draw_channel(axes_list[idx], idx)

        for idx in range(channel_count, len(axes_list)):
            axes_list[idx].axis("off")

        legend_present = bool(legend_handles)
        if legend_present:
            fig.legend(
                handles=legend_handles,
                loc="lower center",
                ncol=min(len(legend_handles), 6),
                frameon=False,
            )

        fig.suptitle("Per-channel latent space with label colors and responsibility alpha", fontsize=14, y=0.995)
        grid_filename = "channel_latents_grid.png" if len(channel_viz_models) == 1 else f"{safe_name}_channel_latents_grid.png"
        grid_path = channel_dir / grid_filename
        bottom_margin = 0.18 if legend_present else 0.06
        fig.tight_layout(rect=(0, bottom_margin, 1, 0.94))
        safe_save_plot(fig, grid_path)

        per_channel_paths: List[str] = []
        for channel_idx in range(channel_count):
            fig_single, ax_single = plt.subplots(figsize=(4, 4))
            _draw_channel(ax_single, channel_idx)
            fig_single.suptitle("Latent space with label colors and responsibility alpha", fontsize=12, y=0.98)
            single_path = channel_dir / f"channel_{channel_idx:02d}.png"
            fig_single.tight_layout()
            safe_save_plot(fig_single, single_path, verbose=False)
            try:
                per_channel_paths.append(str(single_path.relative_to(output_dir)))
            except ValueError:
                per_channel_paths.append(str(single_path))

        # Record relative paths for reporting
        try:
            relative_grid = str(grid_path.relative_to(output_dir))
        except ValueError:
            relative_grid = str(grid_path)

        saved_paths[model_name] = {
            "grid": relative_grid,
            "channels": per_channel_paths,
        }

    return saved_paths


def plot_responsibility_histogram(
    models: Dict[str, object],
    output_dir: Path
):
    """Generate histogram of responsibility confidence (max_c q(c|x)).

    Shows the distribution of how confidently the encoder assigns components.
    High confidence (values near 1.0) indicates clear component specialization.

    Args:
        models: Dictionary mapping model_name -> model object
        output_dir: Base directory for saving figures

    Output:
        Saves figure to: output_dir/mixture/responsibility_histogram.png
    """
    mixture_models = {
        name: model for name, model in models.items()
        if model.config.is_mixture_based_prior()
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

    # Save to appropriate subdirectory
    mixture_dir = output_dir / 'mixture'
    mixture_dir.mkdir(parents=True, exist_ok=True)
    output_path = mixture_dir / 'responsibility_histogram.png'

    safe_save_plot(fig, output_path)


def plot_mixture_evolution(
    models: Dict[str, object],
    output_dir: Path
):
    """Generate π and component usage evolution plots.

    Visualizes how mixture weights (π) and component usage patterns change
    during training. Helps identify component collapse or specialization dynamics.

    Args:
        models: Dictionary mapping model_name -> model object
        output_dir: Base directory for saving figures

    Output:
        Saves figures to: output_dir/mixture/{model_name}_evolution.png
    """
    mixture_models = {
        name: model for name, model in models.items()
        if model.config.is_mixture_based_prior()
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
            safe_name = _sanitize_model_name(model_name)
            output_path = mixture_dir / f'{safe_name}_evolution.png'
            safe_save_plot(fig, output_path)

        except Exception as e:
            print(f"Warning: Could not plot mixture evolution for {model_name}: {e}")


def plot_component_embedding_divergence(
    models: Dict[str, object],
    output_dir: Path
):
    """Analyze and visualize component embedding divergence.

    For component-aware decoders, checks if learned component embeddings e_c
    actually diverge from each other (indicating functional specialization).
    Shows pairwise distances between component embeddings as a heatmap.

    Args:
        models: Dictionary mapping model_name -> model object
        output_dir: Base directory for saving figures

    Output:
        Saves figure to: output_dir/mixture/component_embedding_divergence.png
    """
    component_aware_models = {
        name: model for name, model in models.items()
        if (
            model.config.is_mixture_based_prior() and
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

    # Save to appropriate subdirectory
    mixture_dir = output_dir / 'mixture'
    mixture_dir.mkdir(parents=True, exist_ok=True)
    output_path = mixture_dir / 'component_embedding_divergence.png'

    safe_save_plot(fig, output_path)


def plot_reconstruction_by_component(
    models: Dict[str, object],
    X_data: np.ndarray,
    output_dir: Path,
    *,
    num_samples: int = 3,
    seed: int = 42,
):
    """Visualize how each component reconstructs individual inputs.

    For component-aware decoders, shows whether components specialize
    in different reconstruction strategies. For each input, displays:
    - Original image
    - Reconstructions from each of K components
    - Final weighted reconstruction

    Args:
        models: Dictionary mapping model_name -> model object
        X_data: Input data (N, ...)
        output_dir: Base directory for saving figures
        num_samples: Number of input samples to visualize
        seed: Random seed for sample selection

    Output:
        Saves figures to: output_dir/mixture/{model_name}_reconstruction_by_component.png
    """
    component_aware_models = {
        name: model for name, model in models.items()
        if (
            model.config.is_mixture_based_prior() and
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

                recon_per_component, sigma_per_component, error_msg = _extract_component_recon(extras)
                if error_msg:
                    print(f"  Skipping {model_name}: {error_msg}")
                    continue

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
                    if sigma_per_component is not None:
                        title = f'C{c}\n(q={resp_np[c]:.2f}, σ={sigma_per_component[c]:.2f})' if sample_idx == 0 else f'q={resp_np[c]:.2f}, σ={sigma_per_component[c]:.2f}'
                    else:
                        title = f'C{c}\n(q={resp_np[c]:.2f})' if sample_idx == 0 else f'q={resp_np[c]:.2f}'
                    ax.set_title(title, fontsize=8)
                    ax.axis('off')

                # Plot weighted reconstruction
                axes[sample_idx, K + 1].imshow(_prep_image(weighted_recon), cmap='gray')
                axes[sample_idx, K + 1].set_title('Weighted' if sample_idx == 0 else '')
                axes[sample_idx, K + 1].axis('off')

            plt.suptitle(f'{model_name}: Reconstruction by Component', fontsize=12, y=0.995)
            plt.tight_layout()

            # Save to appropriate subdirectory
            mixture_dir = output_dir / 'mixture'
            mixture_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{_sanitize_model_name(model_name)}_reconstruction_by_component.png"
            output_path = mixture_dir / filename

            safe_save_plot(fig, output_path)

        except Exception as e:
            print(f"Warning: Could not plot reconstruction by component for {model_name}: {e}")
            import traceback
            traceback.print_exc()
