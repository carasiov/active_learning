#!/usr/bin/env python3
"""Run a single SSVAE experiment from configuration file."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import numpy as np

# Add paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from mnist.mnist import load_mnist_scaled
from ssvae import SSVAE, SSVAEConfig
from ssvae.diagnostics import DiagnosticsCollector
from comparison_utils import (
    plot_loss_comparison,
    plot_latent_spaces,
    plot_latent_by_component,
    plot_reconstructions,
    plot_responsibility_histogram,
    plot_mixture_evolution,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SSVAE experiment from config file")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to experiment YAML config (default: configs/default.yaml)")
    return parser.parse_args()


def load_experiment_config(config_path: str) -> dict:
    """Load experiment configuration from YAML."""
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except ImportError:
        print("Error: PyYAML not installed. Install with: poetry add pyyaml")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)


def prepare_data(data_config: dict):
    """Load and prepare MNIST data from config."""
    num_samples = data_config.get('num_samples', 5000)
    num_labeled = data_config.get('num_labeled', 50)
    seed = data_config.get('seed', 42)

    print(f"Loading MNIST: {num_samples} samples, {num_labeled} labeled...")
    X_train, y_train, X_test, y_test = load_mnist_scaled(reshape=True, hw=(28, 28))

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(X_train), size=num_samples, replace=False)
    X_subset = X_train[indices]
    y_subset = y_train[indices]

    # Semi-supervised labels
    y_semi = np.full(num_samples, np.nan, dtype=np.float32)
    labeled_indices = rng.choice(num_samples, size=num_labeled, replace=False)
    y_semi[labeled_indices] = y_subset[labeled_indices]

    print(f"  Train: {len(X_subset)} ({num_labeled} labeled)")
    return X_subset, y_semi, y_subset


def train_model(model_config: dict, X_train, y_train, y_true, output_dir: Path):
    """Train model from configuration.

    Args:
        model_config: Model configuration dictionary
        X_train: Training data
        y_train: Semi-supervised labels (may contain NaN)
        y_true: True labels (for evaluation)
        output_dir: Directory to save outputs

    Returns:
        Tuple of (model, history, summary_dict)
    """
    print(f"\n{'='*60}\nTraining Model\n{'='*60}")

    # Convert list to tuple for hidden_dims if needed
    if 'hidden_dims' in model_config and isinstance(model_config['hidden_dims'], list):
        model_config['hidden_dims'] = tuple(model_config['hidden_dims'])

    config = SSVAEConfig(**model_config)

    model = SSVAE(input_dim=(28, 28), config=config)
    weights_path = str(output_dir / "checkpoint.ckpt")

    start_time = time.time()
    history = model.fit(X_train, y_train, weights_path=weights_path)
    train_time = time.time() - start_time

    print(f"Training complete in {train_time:.1f}s")

    # Compute predictions for accuracy
    latent, recon, predictions, certainty = model.predict(X_train)
    accuracy = DiagnosticsCollector.compute_accuracy(predictions, y_true)

    # Build structured summary
    summary = {
        'training': {
            'final_loss': float(history['loss'][-1]),
            'final_recon_loss': float(history['reconstruction_loss'][-1]),
            'final_kl_z': float(history.get('kl_z', [0])[-1]) if 'kl_z' in history and history['kl_z'] else 0.0,
            'final_kl_c': float(history.get('kl_c', [0])[-1]) if 'kl_c' in history and history['kl_c'] else 0.0,
            'training_time_sec': float(train_time),
            'epochs_completed': len(history['loss']),
        },
        'classification': {
            'final_accuracy': float(accuracy),
            'final_classification_loss': float(history['classification_loss'][-1]),
        }
    }

    # Add mixture-specific metrics
    if config.prior_type == 'mixture':
        mixture_metrics = model.mixture_metrics

        mixture_summary = {
            'K': config.num_components,
            'final_component_entropy': float(history.get('component_entropy', [0])[-1]) if 'component_entropy' in history and history['component_entropy'] else 0.0,
            'final_pi_entropy': float(history.get('pi_entropy', [0])[-1]) if 'pi_entropy' in history and history['pi_entropy'] else 0.0,
        }

        # Add metrics from diagnostics collector
        if mixture_metrics:
            mixture_summary['K_eff'] = mixture_metrics.get('K_eff', 0.0)
            mixture_summary['active_components'] = mixture_metrics.get('active_components', 0)
            mixture_summary['responsibility_confidence_mean'] = mixture_metrics.get('responsibility_confidence_mean', 0.0)
            if "component_majority_labels" in mixture_metrics:
                mixture_summary['component_majority_labels'] = mixture_metrics["component_majority_labels"]
            if "component_majority_confidence" in mixture_metrics:
                mixture_summary['component_majority_confidence'] = mixture_metrics["component_majority_confidence"]

        # Add π statistics if available
        diag_dir = model.last_diagnostics_dir
        if diag_dir:
            diag_path = Path(diag_dir)
            summary['diagnostics_path'] = str(diag_path)

            pi_path = diag_path / "pi.npy"
            if pi_path.exists():
                pi = np.load(pi_path)
                mixture_summary['pi_max'] = float(np.max(pi))
                mixture_summary['pi_min'] = float(np.min(pi))
                mixture_summary['pi_argmax'] = int(np.argmax(pi))
                mixture_summary['pi_values'] = pi.tolist()

            usage_path = diag_path / "component_usage.npy"
            if usage_path.exists():
                usage = np.load(usage_path)
                mixture_summary['component_usage'] = usage.tolist()

        summary['mixture'] = mixture_summary

    # Add clustering metrics for 2D latents
    if config.latent_dim == 2 and config.prior_type == 'mixture':
        diag_dir = model.last_diagnostics_dir
        if diag_dir:
            latent_data = model._diagnostics.load_latent_data(diag_dir)
            if latent_data is not None:
                responsibilities = latent_data['q_c']
                component_assignments = responsibilities.argmax(axis=1)
                labels = latent_data['labels']

                clustering_metrics = DiagnosticsCollector.compute_clustering_metrics(
                    component_assignments, labels
                )
                if clustering_metrics:
                    summary['clustering'] = clustering_metrics

    return model, history, summary


def generate_visualizations(model, history, X_train, y_true, output_dir: Path):
    """Generate all visualizations for single model."""
    print("\nGenerating visualizations...")

    # Wrap in dict for compatibility with plotting functions
    models = {'Model': model}
    histories = {'Model': history}

    plot_loss_comparison(histories, output_dir)
    plot_latent_spaces(models, X_train, y_true, output_dir)
    plot_latent_by_component(models, X_train, y_true, output_dir)
    plot_responsibility_histogram(models, output_dir)
    plot_mixture_evolution(models, output_dir)
    recon_paths = plot_reconstructions(models, X_train, output_dir)

    return recon_paths


def generate_report(summary: dict, history: dict, experiment_config: dict, output_dir: Path, recon_paths: dict):
    """Generate concise single-model experiment report."""
    report_path = output_dir / 'REPORT.md'

    exp_meta = experiment_config.get('experiment', {})
    data_config = experiment_config.get('data', {})
    model_config = experiment_config.get('model', {})

    with open(report_path, 'w') as f:
        # Header
        f.write("# Experiment Report\n\n")

        # Experiment metadata
        if exp_meta.get('name'):
            f.write(f"**Experiment:** {exp_meta['name']}\n\n")
        if exp_meta.get('description'):
            f.write(f"**Description:** {exp_meta['description']}\n\n")
        if exp_meta.get('tags'):
            tags = ', '.join(exp_meta['tags'])
            f.write(f"**Tags:** {tags}\n\n")

        timestamp = experiment_config.get('timestamp', 'N/A')
        f.write(f"**Generated:** {timestamp}\n\n")

        # Configuration
        f.write("## Configuration\n\n")
        f.write("### Data\n\n")
        for key, value in data_config.items():
            f.write(f"- {key}: {value}\n")

        f.write("\n### Model\n\n")
        f.write(f"- Prior: {model_config.get('prior_type', 'standard')}\n")
        f.write(f"- Latent dim: {model_config.get('latent_dim', 2)}\n")
        f.write(f"- Hidden dims: {model_config.get('hidden_dims', [256, 128, 64])}\n")
        if model_config.get('prior_type') == 'mixture':
            f.write(f"- Components (K): {model_config.get('num_components', 10)}\n")
        f.write(f"- Reconstruction loss: {model_config.get('reconstruction_loss', 'mse')}\n")
        f.write(f"- Learning rate: {model_config.get('learning_rate', 0.001)}\n")
        f.write(f"- Batch size: {model_config.get('batch_size', 128)}\n")
        f.write(f"- Max epochs: {model_config.get('max_epochs', 300)}\n")

        # Results
        f.write("\n## Results\n\n")

        # Key metrics table
        f.write("### Summary Metrics\n\n")
        f.write("| Category | Metric | Value |\n")
        f.write("|----------|--------|-------|\n")

        # Training metrics
        training = summary.get('training', {})
        for key, value in training.items():
            metric = key.replace('_', ' ').replace('final ', '').title()
            if isinstance(value, float):
                f.write(f"| Training | {metric} | {value:.4f} |\n")
            else:
                f.write(f"| Training | {metric} | {value} |\n")

        # Classification metrics
        classification = summary.get('classification', {})
        for key, value in classification.items():
            metric = key.replace('_', ' ').replace('final ', '').title()
            if isinstance(value, float):
                f.write(f"| Classification | {metric} | {value:.4f} |\n")
            else:
                f.write(f"| Classification | {metric} | {value} |\n")

        # Mixture metrics
        if 'mixture' in summary:
            mixture = summary['mixture']
            for key, value in mixture.items():
                if key in ['pi_values', 'component_usage']:  # Skip arrays
                    continue
                metric = key.replace('_', ' ').replace('final ', '').title()
                if isinstance(value, float):
                    f.write(f"| Mixture | {metric} | {value:.4f} |\n")
                else:
                    f.write(f"| Mixture | {metric} | {value} |\n")

        # Clustering metrics
        if 'clustering' in summary:
            clustering = summary['clustering']
            for key, value in clustering.items():
                metric = key.upper()
                f.write(f"| Clustering | {metric} | {value:.4f} |\n")

        # Visualizations
        f.write("\n## Visualizations\n\n")

        f.write("### Loss Curves\n\n")
        f.write("![Loss Comparison](loss_comparison.png)\n\n")

        f.write("### Latent Space\n\n")
        f.write("**By Class Label:**\n\n")
        f.write("![Latent Spaces](latent_spaces.png)\n\n")

        if (output_dir / 'latent_by_component.png').exists():
            f.write("**By Component Assignment:**\n\n")
            f.write("![Latent by Component](latent_by_component.png)\n\n")

        if (output_dir / 'responsibility_histogram.png').exists():
            f.write("### Responsibility Confidence\n\n")
            f.write("Distribution of max_c q(c|x):\n\n")
            f.write("![Responsibility Histogram](responsibility_histogram.png)\n\n")

        if recon_paths:
            f.write("### Reconstructions\n\n")
            for model_name, filename in recon_paths.items():
                f.write(f"![Reconstructions]({filename})\n\n")

        # Mixture evolution
        mixture_dir = output_dir / 'visualizations' / 'mixture'
        if mixture_dir.exists():
            evolution_plots = list(mixture_dir.glob('*_evolution.png'))
            if evolution_plots:
                f.write("### Mixture Evolution\n\n")
                for plot_path in sorted(evolution_plots):
                    rel_path = plot_path.relative_to(output_dir)
                    f.write(f"![Mixture Evolution]({rel_path})\n\n")

    print(f"  Saved: {report_path}")


def main():
    args = parse_args()

    # Load experiment configuration
    experiment_config = load_experiment_config(args.config)

    # Extract sections
    exp_meta = experiment_config.get('experiment', {})
    data_config = experiment_config.get('data', {})
    model_config = experiment_config.get('model', {})

    # Create output directory with experiment name if provided
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if exp_meta.get('name'):
        exp_name = exp_meta['name'].replace(' ', '_').lower()
        output_dir = ROOT_DIR / 'artifacts' / 'experiments' / f"{exp_name}_{timestamp}"
    else:
        output_dir = ROOT_DIR / 'artifacts' / 'experiments' / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Save config copy
    config_copy_path = output_dir / 'config.yaml'
    try:
        import yaml
        with open(config_copy_path, 'w') as f:
            yaml.dump(experiment_config, f, default_flow_style=False, sort_keys=False)
        print(f"Config saved: {config_copy_path}")
    except:
        pass

    # Prepare data
    X_train, y_semi, y_true = prepare_data(data_config)

    # Train model
    model, history, summary = train_model(model_config, X_train, y_semi, y_true, output_dir)

    # Generate visualizations
    recon_paths = generate_visualizations(model, history, X_train, y_true, output_dir)

    # Save summary
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    # Generate report
    experiment_config['timestamp'] = timestamp
    generate_report(summary, history, experiment_config, output_dir, recon_paths)

    print(f"\n{'='*60}\nExperiment complete! Results: {output_dir}\n{'='*60}\n")


if __name__ == '__main__':
    main()
