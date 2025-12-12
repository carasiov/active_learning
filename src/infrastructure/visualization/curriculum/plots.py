"""Curriculum learning visualization functions.

This module provides visualizations for channel unlocking curriculum:
- k_active evolution over epochs
- Migration window indicators
- Trigger-based unlock diagnostics (normality score, plateau detection)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from ..utils import safe_save_plot, style_axes


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
