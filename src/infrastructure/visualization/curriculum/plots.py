"""Curriculum learning visualization plots."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from infrastructure import ComponentResult
from infrastructure.visualization.registry import VisualizationContext, register_plotter


@register_plotter
def curriculum_k_active_plotter(context: VisualizationContext) -> ComponentResult:
    """Plot k_active over time with unlock markers and kick windows.

    Output: figures/curriculum/k_active_over_time.png
    """
    curriculum_history = context.curriculum_history
    if curriculum_history is None or len(curriculum_history) == 0:
        return ComponentResult.disabled(reason="Curriculum not enabled")

    # Extract time series
    epochs = [h["epoch"] for h in curriculum_history]
    k_active = [h["k_active"] for h in curriculum_history]
    unlocked = [h.get("unlocked", False) for h in curriculum_history]
    kick_active = [h.get("kick_active", False) for h in curriculum_history]

    # Create output directory
    curriculum_dir = context.figures_dir / "curriculum"
    curriculum_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot k_active line
    ax.plot(epochs, k_active, "b-", linewidth=2, label="k_active")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Active Channels")
    ax.set_title("Channel Curriculum: k_active Over Time")

    # Mark unlock events
    unlock_epochs = [e for e, u in zip(epochs, unlocked) if u]
    unlock_k = [k for k, u in zip(k_active, unlocked) if u]
    if unlock_epochs:
        ax.scatter(unlock_epochs, unlock_k, c="red", s=100, marker="^",
                   zorder=5, label="Unlock event")

    # Shade kick windows
    in_kick = False
    kick_start = None
    for i, (e, k) in enumerate(zip(epochs, kick_active)):
        if k and not in_kick:
            kick_start = e
            in_kick = True
        elif not k and in_kick:
            ax.axvspan(kick_start, epochs[i-1] if i > 0 else e,
                      alpha=0.2, color="orange", label="Kick window" if kick_start == unlock_epochs[0] if unlock_epochs else True else "")
            in_kick = False
    # Close final kick if still active
    if in_kick and kick_start is not None:
        ax.axvspan(kick_start, epochs[-1], alpha=0.2, color="orange")

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(k_active) + 1)

    out_path = curriculum_dir / "k_active_over_time.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return ComponentResult.success(data={"curriculum_k_active": str(out_path)})


@register_plotter
def curriculum_usage_plotter(context: VisualizationContext) -> ComponentResult:
    """Plot active channel usage over time.

    Uses component_usage from training history if available.
    Output: figures/curriculum/active_usage_over_time.png
    """
    curriculum_history = context.curriculum_history
    if curriculum_history is None or len(curriculum_history) == 0:
        return ComponentResult.disabled(reason="Curriculum not enabled")

    # Check if we have component usage data in history
    history = context.history
    if "component_usage" not in history:
        return ComponentResult.skipped(reason="No component_usage in history")

    usage_history = history.get("component_usage", [])
    if not usage_history or len(usage_history) == 0:
        return ComponentResult.skipped(reason="component_usage history is empty")

    # Create output directory
    curriculum_dir = context.figures_dir / "curriculum"
    curriculum_dir.mkdir(parents=True, exist_ok=True)

    # Get k_active per epoch from curriculum history
    k_active_by_epoch = {h["epoch"]: h["k_active"] for h in curriculum_history}

    # Plot usage for active channels
    fig, ax = plt.subplots(figsize=(10, 5))

    usage_array = np.array(usage_history)  # [epochs, K]
    num_epochs, num_components = usage_array.shape
    epochs = np.arange(num_epochs)

    # Create stacked area plot for active channels
    colors = plt.cm.tab10(np.linspace(0, 1, num_components))

    for k in range(num_components):
        # Only show usage when channel is active
        usage_k = usage_array[:, k].copy()
        for e in range(num_epochs):
            k_act = k_active_by_epoch.get(e, num_components)
            if k >= k_act:
                usage_k[e] = 0

        if np.any(usage_k > 0):
            ax.fill_between(epochs, 0, usage_k, alpha=0.5, color=colors[k],
                           label=f"Channel {k}")
            ax.plot(epochs, usage_k, color=colors[k], linewidth=1)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Usage (E[r_k(x)])")
    ax.set_title("Active Channel Usage Over Time")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, num_epochs - 1)
    ax.set_ylim(0, 1)

    out_path = curriculum_dir / "active_usage_over_time.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return ComponentResult.success(data={"curriculum_usage": str(out_path)})


@register_plotter
def curriculum_peakiness_plotter(context: VisualizationContext) -> ComponentResult:
    """Plot peakiness metrics over time: entropy and max responsibility.

    Output: figures/curriculum/peakiness_over_time.png
    """
    curriculum_history = context.curriculum_history
    if curriculum_history is None or len(curriculum_history) == 0:
        return ComponentResult.disabled(reason="Curriculum not enabled")

    history = context.history

    # Check for required metrics
    has_entropy = "component_entropy" in history or "val_component_entropy" in history
    has_max_resp = "max_responsibility" in history or "val_max_responsibility" in history

    if not (has_entropy or has_max_resp):
        return ComponentResult.skipped(
            reason="No peakiness metrics (component_entropy, max_responsibility) in history"
        )

    # Create output directory
    curriculum_dir = context.figures_dir / "curriculum"
    curriculum_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Get epochs from curriculum history
    epochs = [h["epoch"] for h in curriculum_history]
    max_epoch = max(epochs) if epochs else len(history.get("loss", []))

    # Plot entropy
    ax_entropy = axes[0]
    entropy_key = "val_component_entropy" if "val_component_entropy" in history else "component_entropy"
    if entropy_key in history:
        entropy_vals = history[entropy_key]
        plot_epochs = np.arange(len(entropy_vals))
        ax_entropy.plot(plot_epochs, entropy_vals, "b-", linewidth=2, label="E[H(r(x))]")
        ax_entropy.set_ylabel("Entropy")
        ax_entropy.set_title("Routing Distribution Entropy")
        ax_entropy.legend(loc="best")
        ax_entropy.grid(True, alpha=0.3)

    # Plot max responsibility
    ax_max_resp = axes[1]
    max_resp_key = "val_max_responsibility" if "val_max_responsibility" in history else "max_responsibility"
    if max_resp_key in history:
        max_resp_vals = history[max_resp_key]
        plot_epochs = np.arange(len(max_resp_vals))
        ax_max_resp.plot(plot_epochs, max_resp_vals, "g-", linewidth=2, label="E[max_k r_k(x)]")
        ax_max_resp.set_ylabel("Max Responsibility")
        ax_max_resp.set_title("Mean Maximum Responsibility (Peakiness)")
        ax_max_resp.legend(loc="best")
        ax_max_resp.grid(True, alpha=0.3)

    ax_max_resp.set_xlabel("Epoch")

    # Mark unlock epochs on both plots
    unlock_epochs = [h["epoch"] for h in curriculum_history if h.get("unlocked", False)]
    for ax in axes:
        for ue in unlock_epochs:
            ax.axvline(x=ue, color="red", linestyle="--", alpha=0.5, linewidth=1)

    out_path = curriculum_dir / "peakiness_over_time.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return ComponentResult.success(data={"curriculum_peakiness": str(out_path)})


@register_plotter
def curriculum_latent_grid_plotter(context: VisualizationContext) -> ComponentResult:
    """Plot 2D latent scatter per active channel, colored by label.

    Shows channel specialization at end of training.
    Output: figures/snapshots/final_latent_grid.png
    """
    curriculum_history = context.curriculum_history
    if curriculum_history is None or len(curriculum_history) == 0:
        return ComponentResult.disabled(reason="Curriculum not enabled")

    model = context.model
    config = context.config

    # Check that latent_dim is 2
    if config.latent_dim != 2:
        return ComponentResult.skipped(
            reason=f"Latent grid requires latent_dim=2, got {config.latent_dim}"
        )

    # Check for mixture-based prior
    if not config.is_mixture_based_prior():
        return ComponentResult.skipped(reason="Latent grid requires mixture-based prior")

    # Get final k_active
    final_k_active = curriculum_history[-1].get("k_active", config.num_components)

    # Run inference to get per-component latents and responsibilities
    x_train = context.x_train
    y_true = context.y_true

    # Sample a subset for visualization
    n_samples = min(2000, len(x_train))
    rng = np.random.default_rng(42)
    indices = rng.choice(len(x_train), size=n_samples, replace=False)
    x_subset = x_train[indices]
    y_subset = y_true[indices]

    try:
        # Get per-component latents and responsibilities
        result = model.predict_batched(x_subset, return_mixture=True)
        if len(result) >= 6:
            latent, _, _, _, responsibilities, _ = result[:6]
        else:
            return ComponentResult.skipped(reason="Model does not return responsibilities")

        # Get per-component latents if available
        # For decentralized layout, latent should be [N, K, z_dim]
        if latent.ndim == 2:
            # Shared layout - duplicate latent for all components
            latent_per_k = np.broadcast_to(
                latent[:, np.newaxis, :],
                (n_samples, final_k_active, config.latent_dim)
            )
        elif latent.ndim == 3:
            latent_per_k = latent[:, :final_k_active, :]
        else:
            return ComponentResult.skipped(
                reason=f"Unexpected latent shape: {latent.shape}"
            )

    except Exception as e:
        return ComponentResult.failed(reason=f"Failed to get latents: {e}")

    # Create output directory
    snapshots_dir = context.figures_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    # Create grid of plots
    n_cols = min(4, final_k_active)
    n_rows = (final_k_active + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if final_k_active == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    # Color map for labels
    num_classes = config.num_classes
    cmap = plt.cm.get_cmap("tab10", num_classes)

    for k in range(final_k_active):
        row, col = k // n_cols, k % n_cols
        ax = axes[row, col]

        # Get latent for channel k
        z_k = latent_per_k[:, k, :]  # [N, 2]
        r_k = responsibilities[:, k]  # [N]

        # Calculate usage and purity
        usage = np.mean(r_k)

        # Purity: max_y p(y|k) where p(y|k) = sum_i r_ki * 1[y_i=y] / sum_i r_ki
        purity = 0.0
        if np.sum(r_k) > 1e-6:
            label_weights = np.zeros(num_classes)
            for y in range(num_classes):
                mask = (y_subset == y)
                label_weights[y] = np.sum(r_k[mask])
            label_weights /= np.sum(label_weights) + 1e-10
            purity = np.max(label_weights)

        # Scatter plot with alpha proportional to responsibility
        for y in range(num_classes):
            mask = (y_subset == y)
            if np.sum(mask) == 0:
                continue
            alpha = np.clip(r_k[mask] * 2, 0.1, 1.0)  # Scale for visibility
            ax.scatter(
                z_k[mask, 0], z_k[mask, 1],
                c=[cmap(y)] * np.sum(mask),
                alpha=alpha,
                s=10,
                label=f"{y}" if k == 0 else None,
            )

        ax.set_title(f"Channel {k}\nusage={usage:.2f}, purity={purity:.2f}", fontsize=10)
        ax.set_xlabel("z₀")
        ax.set_ylabel("z₁")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for k in range(final_k_active, n_rows * n_cols):
        row, col = k // n_cols, k % n_cols
        axes[row, col].axis("off")

    # Add legend to first plot
    if final_k_active > 0:
        axes[0, 0].legend(
            title="Label", loc="upper left",
            fontsize=8, markerscale=1.5,
            bbox_to_anchor=(0, 1)
        )

    fig.suptitle(f"Channel Specialization (k_active={final_k_active})", fontsize=12)
    out_path = snapshots_dir / "final_latent_grid.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return ComponentResult.success(data={"curriculum_latent_grid": str(out_path)})


@register_plotter
def curriculum_channel_label_heatmap_plotter(context: VisualizationContext) -> ComponentResult:
    """Plot channel-label heatmap showing specialization.

    Rows: active channels, Cols: labels 0-9
    Values: responsibility-weighted counts normalized per channel
    Output: figures/snapshots/final_channel_label_heatmap.png
    """
    curriculum_history = context.curriculum_history
    if curriculum_history is None or len(curriculum_history) == 0:
        return ComponentResult.disabled(reason="Curriculum not enabled")

    model = context.model
    config = context.config

    # Check for mixture-based prior
    if not config.is_mixture_based_prior():
        return ComponentResult.skipped(reason="Heatmap requires mixture-based prior")

    # Get final k_active
    final_k_active = curriculum_history[-1].get("k_active", config.num_components)

    # Run inference to get responsibilities
    x_train = context.x_train
    y_true = context.y_true

    try:
        result = model.predict_batched(x_train, return_mixture=True)
        if len(result) >= 6:
            _, _, _, _, responsibilities, _ = result[:6]
        else:
            return ComponentResult.skipped(reason="Model does not return responsibilities")
    except Exception as e:
        return ComponentResult.failed(reason=f"Failed to get responsibilities: {e}")

    # Compute responsibility-weighted label distribution per channel
    num_classes = config.num_classes
    heatmap = np.zeros((final_k_active, num_classes))

    for k in range(final_k_active):
        r_k = responsibilities[:, k]
        for y in range(num_classes):
            mask = (y_true == y)
            heatmap[k, y] = np.sum(r_k[mask])

        # Normalize per channel
        row_sum = np.sum(heatmap[k, :])
        if row_sum > 1e-6:
            heatmap[k, :] /= row_sum

    # Create output directory
    snapshots_dir = context.figures_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, max(4, final_k_active * 0.5)))

    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("Label")
    ax.set_ylabel("Channel")
    ax.set_title(f"Channel-Label Distribution (k_active={final_k_active})")

    # Set ticks
    ax.set_xticks(np.arange(num_classes))
    ax.set_xticklabels([str(i) for i in range(num_classes)])
    ax.set_yticks(np.arange(final_k_active))
    ax.set_yticklabels([f"Ch {i}" for i in range(final_k_active)])

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("P(label|channel)")

    # Add text annotations
    for k in range(final_k_active):
        for y in range(num_classes):
            val = heatmap[k, y]
            text_color = "white" if val > 0.5 else "black"
            ax.text(y, k, f"{val:.2f}", ha="center", va="center",
                   color=text_color, fontsize=8)

    out_path = snapshots_dir / "final_channel_label_heatmap.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return ComponentResult.success(data={"curriculum_heatmap": str(out_path)})
