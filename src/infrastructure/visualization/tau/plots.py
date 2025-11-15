"""τ-classifier specific visualization functions.

This module contains plotting functions for semi-supervised VAE models with
τ-classifier components, including label-component mappings and prediction diagnostics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from ..utils import safe_save_plot, style_axes


def plot_tau_matrix_heatmap(
    models: Dict[str, object],
    output_dir: Path
):
    """Visualize τ matrix (components → labels mapping) for τ-classifier models.

    Shows the learned probability distribution τ_{c,y} indicating which components
    are associated with which labels. Annotates high-probability entries and marks
    dominant label per component with stars.

    Args:
        models: Dictionary mapping model_name -> model object
        output_dir: Base directory for saving figures

    Output:
        Saves figure to: output_dir/tau/tau_matrix_heatmap.png
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

            style_axes(
                ax,
                title=f'{model_name}: τ Matrix (Components → Labels)',
                xlabel='Class label',
                ylabel='Component index',
                grid=False,
            )

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

    # Save to appropriate subdirectory
    tau_dir = output_dir / 'tau'
    tau_dir.mkdir(parents=True, exist_ok=True)
    output_path = tau_dir / 'tau_matrix_heatmap.png'

    safe_save_plot(fig, output_path)


def plot_tau_per_class_accuracy(
    models: Dict[str, object],
    X_data: np.ndarray,
    y_true: np.ndarray,
    output_dir: Path
):
    """Plot per-class accuracy comparison for τ-classifier models.

    Shows classification accuracy broken down by individual class, helping
    identify which classes are well-predicted vs poorly-predicted.
    Color-codes bars by performance level.

    Args:
        models: Dictionary mapping model_name -> model object
        X_data: Input data (N, ...)
        y_true: True labels (N,)
        output_dir: Base directory for saving figures

    Output:
        Saves figure to: output_dir/tau/tau_per_class_accuracy.png
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

            style_axes(
                ax,
                title=f'{model_name}: Per-Class Accuracy',
                xlabel='Class label',
                ylabel='Accuracy (%)',
            )
            ax.set_xticks(x)
            ax.set_ylim([0, 105])
            ax.axhline(y=100 * np.mean(predictions == y_true), color='red',
                      linestyle='--', linewidth=2, label=f'Overall: {np.mean(predictions == y_true)*100:.1f}%')
            ax.legend()

            print(f"  {model_name}: Overall accuracy = {np.mean(predictions == y_true)*100:.1f}%")

        except Exception as e:
            print(f"Warning: Could not plot per-class accuracy for {model_name}: {e}")

    plt.tight_layout()

    # Save to appropriate subdirectory
    tau_dir = output_dir / 'tau'
    tau_dir.mkdir(parents=True, exist_ok=True)
    output_path = tau_dir / 'tau_per_class_accuracy.png'

    safe_save_plot(fig, output_path)


def plot_tau_certainty_analysis(
    models: Dict[str, object],
    X_data: np.ndarray,
    y_true: np.ndarray,
    output_dir: Path
):
    """Analyze certainty vs accuracy for τ-classifier models.

    Investigates the relationship between prediction certainty and actual
    correctness. Well-calibrated models should show high correlation:
    high certainty predictions should be correct more often.

    Args:
        models: Dictionary mapping model_name -> model object
        X_data: Input data (N, ...)
        y_true: True labels (N,)
        output_dir: Base directory for saving figures

    Output:
        Saves figure to: output_dir/tau/tau_certainty_analysis.png

    Note:
        The ideal line (y=x) represents perfect calibration where certainty
        matches accuracy.
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

            style_axes(
                ax,
                title=f'{model_name}: Certainty vs Accuracy',
                xlabel='Certainty (max_c r_c × τ_c,y)',
                ylabel='Accuracy',
            )
            ax.legend()
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

    # Save to appropriate subdirectory
    tau_dir = output_dir / 'tau'
    tau_dir.mkdir(parents=True, exist_ok=True)
    output_path = tau_dir / 'tau_certainty_analysis.png'

    safe_save_plot(fig, output_path)
