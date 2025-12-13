"""Curriculum learning visualization functions.

This module provides visualizations for channel unlocking curriculum:
- k_active evolution over epochs
- Migration window indicators
- Trigger-based unlock diagnostics (normality score, plateau detection)
- Curriculum-aware channel latent space visualization
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from ..utils import (
    _build_label_palette,
    _compute_limits,
    _sanitize_model_name,
    safe_save_plot,
    style_axes,
    CHANNEL_POINT_SIZE,
)


def plot_curriculum_metrics(
    histories: Dict[str, Dict[str, List[float]]],
    output_dir: Path,
) -> bool:
    """Generate curriculum learning diagnostic plots.

    Creates a multi-panel figure showing:
    - k_active evolution (number of active channels over time)
    - Migration window indicator (when soft routing is active)
    - Normality score (latent proximity to N(0,I))
    - Plateau detection diagnostics (improvement and trigger status)

    Args:
        histories: Dictionary mapping model_name -> history_dict.
            Expected keys: k_active, in_migration_window, normality_score,
            plateau_detected, plateau_improvement, unlock_triggered.
        output_dir: Base directory for saving figures.

    Returns:
        True if plot was saved successfully, False otherwise.

    Output:
        Saves figure to: output_dir/curriculum/curriculum_metrics.png
    """
    # Check if any history has curriculum data
    has_curriculum_data = False
    for history in histories.values():
        if "k_active" in history and len(history["k_active"]) > 0:
            has_curriculum_data = True
            break

    if not has_curriculum_data:
        return False

    # Determine layout based on available metrics
    # Check what metrics we have
    has_trigger_metrics = any(
        "normality_score" in h and len(h.get("normality_score", [])) > 0
        for h in histories.values()
    )

    if has_trigger_metrics:
        # Full 3x2 layout for trigger mode
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        axes = axes.flatten()
    else:
        # Simpler 2x1 layout for epoch mode
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        axes = [axes[0], axes[1]]

    # Panel 1: k_active evolution
    ax_k_active = axes[0]
    for model_name, history in histories.items():
        if "k_active" in history and len(history["k_active"]) > 0:
            epochs = range(1, len(history["k_active"]) + 1)
            ax_k_active.plot(
                epochs,
                history["k_active"],
                label=model_name,
                linewidth=2,
                marker='o',
                markersize=3,
                alpha=0.8,
            )
            # Mark unlock events (where k_active increases)
            k_active = np.array(history["k_active"])
            unlock_epochs = np.where(np.diff(k_active) > 0)[0] + 1  # +1 for 1-indexed
            if len(unlock_epochs) > 0:
                ax_k_active.scatter(
                    unlock_epochs + 1,  # +1 because diff shifts by 1
                    k_active[unlock_epochs],
                    marker='^',
                    s=100,
                    c='red',
                    zorder=5,
                    label='_nolegend_',  # Don't add to legend
                )

    style_axes(
        ax_k_active,
        title='Active Channels (k_active)',
        xlabel='Epoch',
        ylabel='k_active',
    )
    ax_k_active.legend(loc='best', fontsize=9)
    ax_k_active.set_ylim(bottom=0)

    # Panel 2: Migration window indicator
    ax_migration = axes[1]
    for model_name, history in histories.items():
        if "in_migration_window" in history and len(history["in_migration_window"]) > 0:
            epochs = range(1, len(history["in_migration_window"]) + 1)
            # Convert to binary for step plot
            migration = np.array(history["in_migration_window"], dtype=float)
            ax_migration.fill_between(
                epochs,
                0,
                migration,
                alpha=0.3,
                step='mid',
                label=f'{model_name} (migration)',
            )
            ax_migration.step(
                epochs,
                migration,
                where='mid',
                linewidth=1.5,
                alpha=0.8,
            )

    style_axes(
        ax_migration,
        title='Migration Window Active',
        xlabel='Epoch',
        ylabel='In Migration (0/1)',
    )
    ax_migration.set_ylim(-0.1, 1.1)
    ax_migration.set_yticks([0, 1])
    ax_migration.set_yticklabels(['No', 'Yes'])
    if any("in_migration_window" in h and len(h.get("in_migration_window", [])) > 0 for h in histories.values()):
        ax_migration.legend(loc='best', fontsize=9)

    if has_trigger_metrics:
        # Panel 3: Normality score evolution
        ax_normality = axes[2]
        for model_name, history in histories.items():
            if "normality_score" in history and len(history["normality_score"]) > 0:
                epochs = range(1, len(history["normality_score"]) + 1)
                ax_normality.plot(
                    epochs,
                    history["normality_score"],
                    label=model_name,
                    linewidth=2,
                    alpha=0.8,
                )

        style_axes(
            ax_normality,
            title='Latent Normality Score (lower = closer to N(0,I))',
            xlabel='Epoch',
            ylabel='Normality Score',
        )
        ax_normality.legend(loc='best', fontsize=9)
        ax_normality.set_ylim(bottom=0)

        # Panel 4: Plateau improvement
        ax_improvement = axes[3]
        for model_name, history in histories.items():
            if "plateau_improvement" in history and len(history["plateau_improvement"]) > 0:
                epochs = range(1, len(history["plateau_improvement"]) + 1)
                ax_improvement.plot(
                    epochs,
                    history["plateau_improvement"],
                    label=model_name,
                    linewidth=2,
                    alpha=0.8,
                )

        style_axes(
            ax_improvement,
            title='Plateau Detection: Relative Improvement',
            xlabel='Epoch',
            ylabel='Improvement (old-new)/|old|',
        )
        ax_improvement.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax_improvement.legend(loc='best', fontsize=9)

        # Panel 5: Plateau detected indicator
        ax_plateau = axes[4]
        for model_name, history in histories.items():
            if "plateau_detected" in history and len(history["plateau_detected"]) > 0:
                epochs = range(1, len(history["plateau_detected"]) + 1)
                plateau = np.array(history["plateau_detected"], dtype=float)
                ax_plateau.fill_between(
                    epochs,
                    0,
                    plateau,
                    alpha=0.3,
                    step='mid',
                    label=f'{model_name}',
                )
                ax_plateau.step(
                    epochs,
                    plateau,
                    where='mid',
                    linewidth=1.5,
                    alpha=0.8,
                )

        style_axes(
            ax_plateau,
            title='Plateau Detected',
            xlabel='Epoch',
            ylabel='Plateau (0/1)',
        )
        ax_plateau.set_ylim(-0.1, 1.1)
        ax_plateau.set_yticks([0, 1])
        ax_plateau.set_yticklabels(['No', 'Yes'])
        ax_plateau.legend(loc='best', fontsize=9)

        # Panel 6: Unlock triggered indicator
        ax_unlock = axes[5]
        for model_name, history in histories.items():
            if "unlock_triggered" in history and len(history["unlock_triggered"]) > 0:
                epochs = range(1, len(history["unlock_triggered"]) + 1)
                unlock = np.array(history["unlock_triggered"], dtype=float)
                # Mark trigger events with vertical lines
                trigger_epochs = np.where(unlock > 0)[0] + 1
                for te in trigger_epochs:
                    ax_unlock.axvline(x=te, color='green', alpha=0.7, linewidth=2)
                ax_unlock.fill_between(
                    epochs,
                    0,
                    unlock,
                    alpha=0.3,
                    step='mid',
                    color='green',
                    label=f'{model_name}',
                )

        style_axes(
            ax_unlock,
            title='Unlock Triggered (trigger mode)',
            xlabel='Epoch',
            ylabel='Triggered (0/1)',
        )
        ax_unlock.set_ylim(-0.1, 1.1)
        ax_unlock.set_yticks([0, 1])
        ax_unlock.set_yticklabels(['No', 'Yes'])
        ax_unlock.legend(loc='best', fontsize=9)

    plt.tight_layout()

    # Save to curriculum subdirectory
    curriculum_dir = output_dir / 'curriculum'
    curriculum_dir.mkdir(parents=True, exist_ok=True)
    output_path = curriculum_dir / 'curriculum_metrics.png'

    return safe_save_plot(fig, output_path)


def compute_unlock_epochs(k_active_history: List[float]) -> Dict[int, int]:
    """Compute which epoch each channel was unlocked.

    Args:
        k_active_history: List of k_active values per epoch.

    Returns:
        Dictionary mapping channel_index -> unlock_epoch (1-indexed).
        Channel 0 is always unlocked at epoch 1.
    """
    if not k_active_history:
        return {}

    unlock_epochs = {0: 1}  # Channel 0 always starts active
    k_active = np.array(k_active_history)

    for epoch_idx in range(1, len(k_active)):
        if k_active[epoch_idx] > k_active[epoch_idx - 1]:
            # New channel(s) unlocked at this epoch
            prev_k = int(k_active[epoch_idx - 1])
            curr_k = int(k_active[epoch_idx])
            for channel in range(prev_k, curr_k):
                unlock_epochs[channel] = epoch_idx + 1  # 1-indexed

    return unlock_epochs


def plot_curriculum_channel_progression(
    model: Any,
    X_data: np.ndarray,
    y_true: np.ndarray,
    history: Dict[str, List[float]],
    output_dir: Path,
    *,
    max_points: int = 20000,
    seed: int = 0,
) -> Optional[str]:
    """Render curriculum-aware channel latent visualization showing unlock progression.

    Creates a visualization that shows channels in unlock order, with annotations
    indicating when each channel was unlocked and its usage statistics. Only shows
    channels that were actually unlocked based on the training history.

    Args:
        model: Trained model with predict_batched method.
        X_data: Input data (N, ...).
        y_true: True labels (N,).
        history: Training history containing 'k_active' per epoch.
        output_dir: Base directory for saving figures.
        max_points: Maximum points to plot (downsamples if exceeded).
        seed: Random seed for downsampling.

    Returns:
        Relative path to saved figure, or None if not applicable.

    Output:
        Saves figure to: output_dir/curriculum/channel_progression.png
    """
    # Check prerequisites
    if not hasattr(model, "config") or not model.config.is_mixture_based_prior():
        return None

    if not history.get("k_active"):
        return None

    k_active_history = history["k_active"]
    final_k_active = int(k_active_history[-1]) if k_active_history else 0
    if final_k_active == 0:
        return None

    # Compute unlock epochs
    unlock_epochs = compute_unlock_epochs(k_active_history)
    total_epochs = len(k_active_history)

    # Get latent representations and responsibilities
    try:
        latent, _, _, _, responsibilities, _ = model.predict_batched(
            X_data, return_mixture=True
        )
    except Exception as e:
        print(f"Warning: Could not get latent data for curriculum progression: {e}")
        return None

    if responsibilities is None:
        return None

    resp = np.asarray(responsibilities)
    y_array = np.asarray(y_true)

    # Check for per-component latents (decentralized layout)
    channel_latents = None
    try:
        diag_dir = getattr(model, "last_diagnostics_dir", None)
        if diag_dir:
            latent_data = model._diagnostics.load_latent_data(diag_dir)
            if latent_data and isinstance(latent_data, dict):
                z_mean_per_component = latent_data.get("z_mean_per_component")
                if z_mean_per_component is not None:
                    cached_latents = np.asarray(z_mean_per_component)
                    # Only use cached latents if size matches current data
                    if cached_latents.shape[0] == resp.shape[0]:
                        channel_latents = cached_latents
                    else:
                        print(f"Info: Discarding cached diagnostics for curriculum progression "
                              f"due to size mismatch (cached={cached_latents.shape[0]}, current={resp.shape[0]})")
    except Exception:
        pass

    # Downsample if needed
    total = resp.shape[0]
    if total > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(total, size=max_points, replace=False))
    else:
        idx = slice(None)

    # Always compute latent_2d as fallback
    latent_2d = np.asarray(latent)[idx, :2]
    x_lim, y_lim = _compute_limits(latent_2d)

    if channel_latents is not None:
        channel_latents = channel_latents[idx]
        # Use per-channel latents for limits if available
        flat_latent = channel_latents.reshape(-1, channel_latents.shape[-1])
        x_lim, y_lim = _compute_limits(flat_latent[:, :2])

    resp = resp[idx]
    labels = y_array[idx] if y_array is not None else np.zeros(resp.shape[0])

    if labels.dtype.kind == "f":
        labels = np.where(np.isnan(labels), -1, labels).astype(int)
    else:
        labels = labels.astype(int)

    # Build color palette
    max_label = labels.max()
    num_classes = int(getattr(model.config, "num_classes", max(max_label + 1, 1)))
    palette = _build_label_palette(num_classes)
    label_colors = palette[np.clip(labels, 0, num_classes - 1)].copy()
    invalid_mask = (labels < 0) | (labels >= num_classes)
    if invalid_mask.any():
        label_colors[invalid_mask] = np.array([0.5, 0.5, 0.5, 1.0])

    # Only show channels that were unlocked
    channels_to_show = sorted(unlock_epochs.keys())
    if not channels_to_show:
        return None

    n_channels = len(channels_to_show)
    n_cols = min(5, n_channels)
    n_rows = math.ceil(n_channels / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.5 * n_rows))
    if isinstance(axes, np.ndarray):
        axes_list = axes.flatten()
    else:
        axes_list = [axes]

    legend_handles = [
        Patch(facecolor=palette[i], edgecolor="none", label=str(i))
        for i in range(num_classes)
    ]

    for plot_idx, channel_idx in enumerate(channels_to_show):
        ax = axes_list[plot_idx]

        # Get points for this channel
        if channel_latents is not None and channel_idx < channel_latents.shape[1]:
            points = channel_latents[:, channel_idx, :2]
        else:
            points = latent_2d

        # Color with responsibility alpha
        rgba = label_colors.copy()
        if channel_idx < resp.shape[1]:
            rgba[:, 3] = np.clip(resp[:, channel_idx], 0.0, 1.0)
        else:
            rgba[:, 3] = 0.1  # Fallback: very transparent if channel not in resp

        ax.scatter(
            points[:, 0],
            points[:, 1],
            c=rgba,
            s=CHANNEL_POINT_SIZE,
            linewidths=0,
            edgecolors="none",
        )

        # Compute channel statistics
        if channel_idx < resp.shape[1]:
            mean_resp = resp[:, channel_idx].mean()
            max_resp_count = (resp.argmax(axis=1) == channel_idx).sum()
            usage_pct = 100.0 * max_resp_count / len(resp)
        else:
            mean_resp = 0.0
            usage_pct = 0.0

        unlock_epoch = unlock_epochs.get(channel_idx, "?")
        epochs_active = total_epochs - unlock_epoch + 1 if isinstance(unlock_epoch, int) else "?"

        # Title with unlock info
        title = f"Channel {channel_idx}\n"
        title += f"Unlocked: epoch {unlock_epoch}"
        if isinstance(epochs_active, int):
            title += f" ({epochs_active} epochs)"

        ax.set_title(title, fontsize=10, fontweight='bold' if channel_idx == 0 else 'normal')

        # Add usage stats as text box
        stats = f"Usage: {usage_pct:.1f}%\nMean q(c|x): {mean_resp:.3f}"
        ax.text(
            0.02, 0.98, stats,
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_xticks([])
        ax.set_yticks([])
        style_axes(ax, grid=False)

    # Hide unused subplots
    for idx in range(n_channels, len(axes_list)):
        axes_list[idx].axis("off")

    # Add legend
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 10),
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        f"Curriculum Channel Progression (unlocked over {total_epochs} epochs)",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))

    # Save
    curriculum_dir = output_dir / "curriculum"
    curriculum_dir.mkdir(parents=True, exist_ok=True)
    output_path = curriculum_dir / "channel_progression.png"

    if safe_save_plot(fig, output_path):
        try:
            return str(output_path.relative_to(output_dir))
        except ValueError:
            return str(output_path)
    return None


def plot_curriculum_evolution(
    snapshots: List[Dict],
    output_dir: Path,
    *,
    channels_to_show: Optional[List[int]] = None,
    num_classes: int = 10,
) -> Optional[str]:
    """Render curriculum evolution grid: channels (rows) × time (columns).

    Shows how each channel's latent space evolved from unlock through
    migration end to the final state.

    Args:
        snapshots: List of snapshot dicts from DiagnosticsCollector.load_curriculum_snapshots.
            Each dict should have: epoch, k_active, event_type, z_mean/z_mean_per_component,
            responsibilities, labels.
        output_dir: Base directory for saving figures.
        channels_to_show: Optional list of channel indices to show. If None, shows all
            channels that appear in any snapshot.
        num_classes: Number of label classes for coloring.

    Returns:
        Relative path to saved figure, or None if not applicable.
    """
    if not snapshots:
        return None

    # Organize snapshots by channel
    # For each channel, we want: (unlock snapshot, migration_end snapshot, final snapshot if different)
    unlock_snapshots = [s for s in snapshots if s["event_type"] == "unlock"]
    migration_end_snapshots = [s for s in snapshots if s["event_type"] == "migration_end"]

    if not unlock_snapshots:
        return None

    # Build list of channels that were unlocked
    unlocked_channels = set()
    for snap in unlock_snapshots:
        # Channel unlocked is k_active - 1 (the newly unlocked channel)
        # But on first unlock, k_active=1 means channel 0 was already active
        # We need to figure out which channel was unlocked at each snapshot
        k_active = snap["k_active"]
        # At unlock time, k_active is the NEW count, so the newly unlocked channel is k_active - 1
        # But for epoch 0, channel 0 starts active without an "unlock" event
        # Let's use k_active to determine what channels are active in each snapshot
        for ch in range(k_active):
            unlocked_channels.add(ch)

    if channels_to_show is None:
        channels_to_show = sorted(unlocked_channels)

    if not channels_to_show:
        return None

    # For each channel, find its unlock snapshot and migration_end snapshot (if any)
    channel_snapshots: Dict[int, Dict[str, Dict]] = {}
    for ch in channels_to_show:
        channel_snapshots[ch] = {"unlock": None, "migration_end": None}

        # Find unlock snapshot where this channel became active
        for snap in unlock_snapshots:
            if snap["k_active"] > ch:  # Channel is active after this unlock
                # Check if this is the first snapshot where ch is active
                if channel_snapshots[ch]["unlock"] is None:
                    channel_snapshots[ch]["unlock"] = snap
                elif snap["epoch"] < channel_snapshots[ch]["unlock"]["epoch"]:
                    channel_snapshots[ch]["unlock"] = snap

        # Find migration_end snapshot closest to unlock
        for snap in migration_end_snapshots:
            if snap["k_active"] > ch:  # Channel is active
                unlock_snap = channel_snapshots[ch]["unlock"]
                if unlock_snap is not None:
                    # Find migration_end that follows the unlock
                    if snap["epoch"] > unlock_snap["epoch"]:
                        if channel_snapshots[ch]["migration_end"] is None:
                            channel_snapshots[ch]["migration_end"] = snap
                        elif snap["epoch"] < channel_snapshots[ch]["migration_end"]["epoch"]:
                            channel_snapshots[ch]["migration_end"] = snap

    # Also add the final snapshot for comparison
    final_snapshot = snapshots[-1] if snapshots else None

    # Build color palette
    palette = _build_label_palette(num_classes)

    # Determine grid layout
    n_rows = len(channels_to_show)
    n_cols = 3  # unlock, migration_end, final

    if n_rows == 0:
        return None

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Compute global limits from all snapshots
    all_points = []
    for snap in snapshots:
        if "z_mean_per_component" in snap:
            z_pc = snap["z_mean_per_component"]  # [N, K, D]
            for ch in channels_to_show:
                if ch < z_pc.shape[1]:
                    all_points.append(z_pc[:, ch, :2])
        else:
            all_points.append(snap["z_mean"][:, :2])

    if all_points:
        combined = np.concatenate(all_points, axis=0)
        x_lim, y_lim = _compute_limits(combined)
    else:
        x_lim, y_lim = (-3, 3), (-3, 3)

    col_titles = ["At Unlock", "After Migration", "Latest"]

    for row_idx, ch in enumerate(channels_to_show):
        snap_dict = channel_snapshots[ch]

        for col_idx, (key, col_title) in enumerate(zip(
            ["unlock", "migration_end", None], col_titles
        )):
            ax = axes[row_idx, col_idx]

            if col_idx == 2:
                snap = final_snapshot
            else:
                snap = snap_dict.get(key)

            if snap is None:
                ax.text(
                    0.5, 0.5, "N/A",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=12, color="gray",
                )
                ax.set_xlim(*x_lim)
                ax.set_ylim(*y_lim)
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(col_title, fontsize=11)
                if col_idx == 0:
                    ax.set_ylabel(f"Ch {ch}", fontsize=10, fontweight="bold")
                style_axes(ax, grid=False)
                continue

            # Get data for this snapshot
            labels = snap["labels"]
            resp = snap["responsibilities"]

            if "z_mean_per_component" in snap:
                z_pc = snap["z_mean_per_component"]  # [N, K, D]
                if ch < z_pc.shape[1]:
                    points = z_pc[:, ch, :2]
                else:
                    points = snap["z_mean"][:, :2]
            else:
                points = snap["z_mean"][:, :2]

            # Build colors from labels
            label_int = labels.astype(int)
            label_int = np.clip(label_int, 0, num_classes - 1)
            label_colors = palette[label_int].copy()

            # Apply responsibility as alpha
            if ch < resp.shape[1]:
                label_colors[:, 3] = np.clip(resp[:, ch], 0.0, 1.0)
            else:
                label_colors[:, 3] = 0.1

            ax.scatter(
                points[:, 0],
                points[:, 1],
                c=label_colors,
                s=CHANNEL_POINT_SIZE * 0.8,
                linewidths=0,
                edgecolors="none",
            )

            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            ax.set_xticks([])
            ax.set_yticks([])
            style_axes(ax, grid=False)

            # Add column title (first row only)
            if row_idx == 0:
                epoch_info = f" (e{snap['epoch']})" if snap else ""
                ax.set_title(f"{col_title}{epoch_info}", fontsize=11)

            # Add row label (first column only)
            if col_idx == 0:
                ax.set_ylabel(f"Ch {ch}", fontsize=10, fontweight="bold")

            # Add epoch annotation in corner
            if snap:
                ax.text(
                    0.02, 0.98,
                    f"e{snap['epoch']}",
                    transform=ax.transAxes,
                    va="top", ha="left",
                    fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                )

    # Add legend
    legend_handles = [
        Patch(facecolor=palette[i], edgecolor="none", label=str(i))
        for i in range(num_classes)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 10),
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        "Curriculum Evolution: Channels × Time",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    # Save
    curriculum_dir = output_dir / "curriculum"
    curriculum_dir.mkdir(parents=True, exist_ok=True)
    output_path = curriculum_dir / "channel_evolution.png"

    if safe_save_plot(fig, output_path):
        try:
            return str(output_path.relative_to(output_dir))
        except ValueError:
            return str(output_path)
    return None
