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
import jax
import jax.numpy as jnp

from ..utils import (
    _sanitize_model_name,
    _build_label_palette,
    _downsample_points,
    _compute_limits,
    _extract_component_recon,
    _prep_image,
    safe_save_plot,
    style_axes,
    LATENT_POINT_SIZE,
    CHANNEL_POINT_SIZE,
)


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

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

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

            latent_2d = np.asarray(latent)[:, :2]
            component_assignments = responsibilities.argmax(axis=1)
            n_components = responsibilities.shape[1]

            (x_lim, y_lim) = _compute_limits(latent_2d)
            cmap = plt.cm.get_cmap("tab20" if n_components <= 20 else "hsv")

            for c in range(n_components):
                mask = component_assignments == c
                if mask.sum() == 0:
                    continue
                color = cmap(c / n_components)
                ax.scatter(
                    latent_2d[mask, 0],
                    latent_2d[mask, 1],
                    label=f"C{c}",
                    alpha=0.65,
                    s=LATENT_POINT_SIZE,
                    color=color,
                    linewidths=0,
                )

            style_axes(
                ax,
                title=f"{model_name} · Components",
                xlabel="z₁",
                ylabel="z₂",
            )
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            ax.legend(
                title="Component",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                fontsize=8,
                ncol=1 if n_components <= 10 else 2,
                borderpad=0.2,
            )

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
    y_array = np.asarray(y_true)

    for model_name, model in channel_viz_models.items():
        latent_2d = None
        channel_latents = None  # shape: [N, K, 2] when available
        resp = None
        labels = None

        # Prefer diagnostics payload (captures per-component latents if available)
        latent_data = None
        try:
            diag_dir = getattr(model, "last_diagnostics_dir", None)
            if diag_dir:
                latent_data = model._diagnostics.load_latent_data(diag_dir)
        except Exception:
            latent_data = None

        if latent_data and isinstance(latent_data, dict):
            resp = np.asarray(latent_data.get("q_c"))
            labels = np.asarray(latent_data.get("labels"))
            z_mean_per_component = latent_data.get("z_mean_per_component")
            if z_mean_per_component is not None:
                channel_latents = np.asarray(z_mean_per_component)

            # Verify size match with X_data (if provided via y_true length proxy)
            # We use y_true as a proxy for X_data length because X_data might be passed as different type/shape
            if y_true is not None and resp.shape[0] != len(y_true):
                print(f"Info: Discarding cached diagnostics for {model_name} due to size mismatch "
                      f"(cached={resp.shape[0]}, current={len(y_true)}). Re-running prediction.")
                latent_data = None
                resp = None
                channel_latents = None
                labels = None

        if channel_latents is None:
            try:
                latent, _, _, _, responsibilities, _ = model.predict_batched(
                    X_data,
                    return_mixture=True,
                )
                latent_2d = np.asarray(latent)[:, :2]
                resp = np.asarray(responsibilities)
                labels = y_array
            except Exception as exc:
                print(f"Warning: Could not retrieve mixture outputs for {model_name}: {exc}")
                continue

        if resp is None or resp.size == 0:
            print(f"Warning: Missing responsibilities for {model_name}")
            continue

        if channel_latents is not None:
            if channel_latents.shape[-1] < 2:
                print(f"Warning: Skipping channel latent plot for {model_name}: latent_dim < 2")
                continue
            if resp.shape[0] != channel_latents.shape[0]:
                print(f"Warning: Responsibilities size mismatch for {model_name}")
                continue

            total = resp.shape[0]
            if total > max_points:
                rng = np.random.default_rng(seed)
                idx = np.sort(rng.choice(total, size=max_points, replace=False))
            else:
                idx = slice(None)

            channel_latents = channel_latents[idx]
            resp = resp[idx]
            # Prefer y_array (true labels) if available and matching size, otherwise fallback to cached labels
            if y_array is not None and y_array.shape[0] >= resp.shape[0]:
                 labels = y_array[idx]
            else:
                 labels = labels[idx] if labels is not None else y_array[: resp.shape[0]]
            flat_latent = channel_latents.reshape(-1, channel_latents.shape[-1])
            (x_lim, y_lim) = _compute_limits(flat_latent[:, :2])
        else:
            if latent_2d is None or latent_2d.shape[1] < 2:
                print(f"Warning: Skipping channel latent plot for {model_name}: latent_dim < 2 or missing q(c|x)")
                continue
            if resp.shape[0] != latent_2d.shape[0]:
                print(f"Warning: Responsibilities size mismatch for {model_name}")
                continue

            latent_2d, resp, labels = _downsample_points(
                latent_2d,
                resp,
                np.asarray(labels).astype(int),
                max_points=max_points,
                seed=seed,
            )
            (x_lim, y_lim) = _compute_limits(latent_2d)

        if labels.size == 0:
            print(f"Warning: No labels available for channel latent plot ({model_name})")
            continue

        if labels.dtype.kind == "f":
            labels = np.where(np.isnan(labels), -1, labels).astype(int)
        else:
            labels = labels.astype(int)

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

        legend_handles = [
            Patch(facecolor=palette[i], edgecolor="none", label=str(i))
            for i in range(num_classes)
        ]
        if invalid_mask.any():
            legend_handles.append(Patch(facecolor=[0.5, 0.5, 0.5, 1.0], edgecolor="none", label="Unknown"))

        def _draw_channel(ax: plt.Axes, channel_idx: int) -> None:
            rgba = label_colors.copy()
            rgba[:, 3] = np.clip(resp[:, channel_idx], 0.0, 1.0)
            points = channel_latents[:, channel_idx, :2] if channel_latents is not None else latent_2d
            ax.scatter(
                points[:, 0],
                points[:, 1],
                c=rgba,
                s=CHANNEL_POINT_SIZE,
                linewidths=0,
                edgecolors="none",
            )
            style_axes(
                ax,
                title=f"Channel {channel_idx}",
                grid=False,
            )
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            ax.set_xticks([])
            ax.set_yticks([])

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


def plot_selected_vs_weighted_reconstruction(
    models: Dict[str, object],
    X_data: np.ndarray,
    output_dir: Path,
    *,
    num_samples: int = 6,
    seed: int = 0,
):
    """Compare selected-channel (hard) vs weighted reconstructions.

    Caption: Shows original, Gumbel/argmax-selected reconstruction, weighted reconstruction, and alt (2nd-best) per sample.
    """
    mixture_models = {
        name: model for name, model in models.items()
        if model.config.is_mixture_based_prior()
    }
    if not mixture_models:
        return

    rng = np.random.default_rng(seed)
    total = X_data.shape[0]
    idx = rng.choice(total, size=min(num_samples, total), replace=False)
    samples = np.asarray(X_data[idx])

    for model_name, model in mixture_models.items():
        try:
            # Forward with gumbel key to fetch component_selection if available
            try:
                import jax
                jax_key = jax.random.PRNGKey(seed)
                forward = model._apply_fn(
                    model.state.params,
                    jnp.asarray(samples),
                    training=False,
                    rngs={"gumbel": jax_key},
                )
            except Exception:
                forward = model._apply_fn(
                    model.state.params,
                    jnp.asarray(samples),
                    training=False,
                )

            extras = forward.extras if hasattr(forward, "extras") else forward[6]
            if not hasattr(extras, "get"):
                continue

            recon_per_component, sigma_per_component, err = _extract_component_recon(extras)
            if err:
                continue

            responsibilities = extras.get("responsibilities")
            component_selection = extras.get("component_selection", responsibilities)
            if component_selection is None:
                continue
            component_selection = np.asarray(component_selection)
            responsibilities = np.asarray(responsibilities)
            recon_per_component = np.asarray(recon_per_component)

            # Derive selections
            selected_idx = component_selection.argmax(axis=1)
            alt_idx = responsibilities.argsort(axis=1)[:, -2] if responsibilities.shape[1] > 1 else selected_idx

            def _gather_components(indices: np.ndarray) -> np.ndarray:
                gathered = []
                for i, c in enumerate(indices):
                    gathered.append(recon_per_component[i, c])
                return np.stack(gathered, axis=0)

            recon_selected = _gather_components(selected_idx)
            recon_alt = _gather_components(alt_idx)
            recon_weighted = np.sum(
                responsibilities[..., None, None] * recon_per_component,
                axis=1,
            )

            # Plot grid
            cols = 4
            rows = recon_selected.shape[0]
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
            axes = np.atleast_2d(axes)
            for r in range(rows):
                axes[r, 0].imshow(_prep_image(samples[r]), cmap="gray")
                axes[r, 0].set_title("Original" if r == 0 else "")
                axes[r, 1].imshow(_prep_image(recon_selected[r]), cmap="gray")
                axes[r, 1].set_title("Selected" if r == 0 else "")
                axes[r, 2].imshow(_prep_image(recon_weighted[r]), cmap="gray")
                axes[r, 2].set_title("Weighted" if r == 0 else "")
                axes[r, 3].imshow(_prep_image(recon_alt[r]), cmap="gray")
                axes[r, 3].set_title("Alt (2nd best)" if r == 0 else "")
                for c in range(cols):
                    axes[r, c].axis("off")

            fig.suptitle("Selected vs weighted reconstructions (checks hard routing)", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.96))

            mixture_dir = output_dir / "mixture"
            mixture_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{_sanitize_model_name(model_name)}_recon_selected.png"
            safe_save_plot(fig, mixture_dir / fname)

        except Exception as exc:  # pragma: no cover
            print(f"Warning: Failed selected vs weighted recon plot for {model_name}: {exc}")


def plot_channel_ownership_heatmap(
    models: Dict[str, object],
    X_data: np.ndarray,
    y_true: np.ndarray,
    output_dir: Path,
):
    """Heatmap of channel vs label ownership using mean responsibilities.

    Caption: Shows how responsibility mass distributes over (channel, label); bright diagonals imply specialization.
    """
    mixture_models = {
        name: model for name, model in models.items()
        if model.config.is_mixture_based_prior()
    }
    if not mixture_models:
        return

    for model_name, model in mixture_models.items():
        try:
            latent, _, _, _, responsibilities, _ = model.predict_batched(
                X_data, return_mixture=True
            )
        except Exception as exc:
            print(f"Warning: Skipping ownership heatmap for {model_name}: {exc}")
            continue

        if responsibilities is None or y_true.size == 0:
            continue

        resp = np.asarray(responsibilities)
        labels = np.asarray(y_true)
        valid = ~np.isnan(labels)
        if not valid.any():
            continue
        labels = labels[valid].astype(int)
        resp = resp[valid]

        num_components = resp.shape[1]
        num_classes = int(getattr(model.config, "num_classes", labels.max() + 1))
        # ownership[k, y] = mean responsibility for class y on component k
        ownership = np.zeros((num_components, num_classes), dtype=np.float32)
        for c in range(num_classes):
            mask = labels == c
            if mask.any():
                ownership[:, c] = resp[mask].mean(axis=0)

        plt.close("all")
        fig, ax = plt.subplots(figsize=(max(6, num_classes * 0.6), max(4, num_components * 0.4)))
        im = ax.imshow(ownership, aspect="auto", cmap="magma")
        ax.set_xlabel("Label")
        ax.set_ylabel("Component")
        ax.set_title("Channel vs label ownership (mean q(c|x) per class)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_components))

        mixture_dir = output_dir / "mixture"
        mixture_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{_sanitize_model_name(model_name)}_channel_ownership.png"
        safe_save_plot(fig, mixture_dir / fname)


def plot_component_kl_heatmap(
    models: Dict[str, object],
    X_data: np.ndarray,
    output_dir: Path,
    *,
    max_points: int = 2048,
    batch_size: int = 256,
):
    """Per-component KL heatmap to spot dead or over-regularized channels.

    Caption: KL(q(z_k|x) || N(0,I)) summed over dims and averaged over data.
    """
    mixture_models = {
        name: model for name, model in models.items()
        if model.config.is_mixture_based_prior()
    }
    if not mixture_models:
        return

    for model_name, model in mixture_models.items():
        try:
            import jax.numpy as jnp  # noqa: WPS433
            from jax import random  # noqa: WPS433
        except Exception:
            print(f"Warning: JAX unavailable for KL heatmap ({model_name})")
            continue

        data = np.asarray(X_data)
        if data.shape[0] > max_points:
            data = data[:max_points]

        num_components = getattr(model.config, "num_components", 0)
        kl_sums = np.zeros((num_components,), dtype=np.float64)
        count = 0

        for start in range(0, data.shape[0], batch_size):
            batch = data[start:start + batch_size]
            if batch.size == 0:
                continue
            key = random.PRNGKey(start)
            forward = model._apply_fn(
                model.state.params,
                jnp.asarray(batch),
                training=False,
                rngs={"gumbel": key},
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
            kl = kl.sum(axis=2)  # sum over latent dims -> [B, K]
            kl_sums += kl.sum(axis=0)
            count += kl.shape[0]

        if count == 0:
            continue
        kl_mean = kl_sums / float(count)

        plt.close("all")
        fig, ax = plt.subplots(figsize=(max(6, num_components * 0.4), 2.5))
        im = ax.imshow(kl_mean[None, :], aspect="auto", cmap="viridis")
        ax.set_yticks([])
        ax.set_xticks(np.arange(num_components))
        ax.set_xlabel("Component")
        ax.set_title("Per-component KL (mean over data)")
        fig.colorbar(im, ax=ax, fraction=0.046)

        mixture_dir = output_dir / "mixture"
        mixture_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{_sanitize_model_name(model_name)}_component_kl.png"
        safe_save_plot(fig, mixture_dir / fname)


def plot_routing_hardness(
    models: Dict[str, object],
    X_data: np.ndarray,
    output_dir: Path,
    *,
    sample_size: int = 1024,
):
    """Compare soft vs Gumbel-sampled routing hardness.

    Caption: Bars show mean max q(c|x) with/without Gumbel; gap indicates impact of hard routing.
    """
    mixture_models = {
        name: model for name, model in models.items()
        if model.config.is_mixture_based_prior()
    }
    if not mixture_models:
        return

    for model_name, model in mixture_models.items():
        data = np.asarray(X_data)
        if data.shape[0] > sample_size:
            data = data[:sample_size]

        # Soft responsibilities
        try:
            _, _, _, _, resp_soft, _ = model.predict_batched(data, return_mixture=True)
        except Exception as exc:
            print(f"Warning: Skipping routing hardness for {model_name}: {exc}")
            continue
        if resp_soft is None:
            continue
        resp_soft = np.asarray(resp_soft)
        hardness_soft = float(np.mean(resp_soft.max(axis=1)))

        # Gumbel sampled
        try:
            import jax
            forward = model._apply_fn(
                model.state.params,
                jnp.asarray(data),
                training=False,
                rngs={"gumbel": jax.random.PRNGKey(0)},
            )
            extras = forward.extras if hasattr(forward, "extras") else forward[6]
            component_selection = extras.get("component_selection")
            if component_selection is None:
                hardness_gumbel = None
            else:
                hardness_gumbel = float(np.mean(np.asarray(component_selection).max(axis=1)))
        except Exception:
            hardness_gumbel = None

        labels = ["soft"]
        values = [hardness_soft]
        if hardness_gumbel is not None:
            labels.append("gumbel")
            values.append(hardness_gumbel)

        plt.close("all")
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(labels, values, color=["#0B84A5", "#EC5B56"][:len(values)])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Mean max q(c|x)")
        ax.set_title("Routing hardness (soft vs gumbel sample)")

        mixture_dir = output_dir / "mixture"
        mixture_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{_sanitize_model_name(model_name)}_routing_hardness.png"
        safe_save_plot(fig, mixture_dir / fname)

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

            ax.hist(
                max_responsibilities,
                bins=40,
                alpha=0.85,
                color="#0B84A5",
                edgecolor="#F7F8FA",
            )
            mean_val = max_responsibilities.mean()
            ax.axvline(
                mean_val,
                color="#EC5B56",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {mean_val:.3f}",
            )

            style_axes(
                ax,
                title=f"{model_name}\nResponsibility Confidence",
                xlabel="max₍c₎ q(c|x)",
                ylabel="Count",
            )
            ax.legend()

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
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.5))

            K = pi_history.shape[1]
            palette = _build_label_palette(max(K, 1))

            # π evolution
            for c in range(K):
                ax1.plot(
                    tracked_epochs,
                    pi_history[:, c],
                    label=f"π_{c}",
                    alpha=0.8,
                    linewidth=1.6,
                    color=palette[c % len(palette)],
                )
            style_axes(
                ax1,
                title=f"{model_name}: π Evolution",
                xlabel="Epoch",
                ylabel="π (mixture weight)",
            )
            ax1.legend(
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
                fontsize=8,
                ncol=2 if K > 10 else 1,
            )

            # Component usage evolution
            for c in range(K):
                ax2.plot(
                    tracked_epochs,
                    usage_history[:, c],
                    label=f"C_{c}",
                    alpha=0.8,
                    linewidth=1.6,
                    color=palette[c % len(palette)],
                )
            style_axes(
                ax2,
                title=f"{model_name}: Component Usage Evolution",
                xlabel="Epoch",
                ylabel="Component usage",
            )
            ax2.legend(
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
                fontsize=8,
                ncol=2 if K > 10 else 1,
            )

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
            style_axes(
                ax,
                title=f"{model_name}\nComponent Embedding Distances",
                xlabel="Component index",
                ylabel="Component index",
                grid=False,
            )
            ax.set_xticks(range(K))
            ax.set_yticks(range(K))

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
