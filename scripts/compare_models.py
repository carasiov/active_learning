#!/usr/bin/env python3
"""Flexible model comparison tool - compare any SSVAE configurations."""
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
from comparison_utils import (
    plot_loss_comparison,
    plot_latent_spaces,
    plot_reconstructions,
    generate_report,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare SSVAE model configurations")
    parser.add_argument("--config", type=str, help="Path to comparison YAML config")
    parser.add_argument("--num-samples", type=int, default=5000, help="Training samples")
    parser.add_argument("--num-labeled", type=int, default=50, help="Labeled samples")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--models", nargs='+', help="Model names to compare (e.g., standard mixture_k10)")
    return parser.parse_args()


def load_comparison_config(config_path: str) -> dict:
    """Load comparison configuration from YAML."""
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except ImportError:
        print("Warning: PyYAML not installed. Install with: poetry add pyyaml")
        return None


def prepare_data(num_samples: int, num_labeled: int, seed: int):
    """Load and prepare MNIST data."""
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


def train_model(name: str, config: SSVAEConfig, X_train, y_train, output_dir: Path):
    """Train a single model configuration."""
    print(f"\n{'='*60}\nTraining: {name}\n{'='*60}")
    
    model = SSVAE(input_dim=(28, 28), config=config)
    weights_path = str(output_dir / f"{name.replace(' ', '_').lower()}_checkpoint.ckpt")
    
    start_time = time.time()
    history = model.fit(X_train, y_train, weights_path=weights_path)
    train_time = time.time() - start_time
    
    print(f"{name} complete in {train_time:.1f}s")
    
    # Compute summary
    summary = {
        'final_loss': float(history['loss'][-1]),
        'final_recon_loss': float(history['reconstruction_loss'][-1]),
        'final_kl_loss': float(history['kl_loss'][-1]),
        'final_class_loss': float(history['classification_loss'][-1]),
        'training_time_sec': float(train_time),
    }

    if 'component_entropy' in history and len(history['component_entropy']) > 0:
        summary['final_component_entropy'] = float(history['component_entropy'][-1])
    if 'pi_entropy' in history and len(history['pi_entropy']) > 0:
        summary['final_pi_entropy'] = float(history['pi_entropy'][-1])

    diag_dir = getattr(model, "_last_diagnostics_dir", None)
    if diag_dir:
        diag_path = Path(diag_dir)
        summary['diagnostics_path'] = str(diag_path)
        usage_path = diag_path / "component_usage.npy"
        pi_path = diag_path / "pi.npy"
        if usage_path.exists():
            usage = np.load(usage_path)
            summary['component_usage'] = usage.tolist()
        if pi_path.exists():
            pi = np.load(pi_path)
            summary['pi_values'] = pi.tolist()
            summary['pi_max'] = float(np.max(pi))
            summary['pi_min'] = float(np.min(pi))
            summary['pi_argmax'] = int(np.argmax(pi))

    return model, history, summary


def get_predefined_models():
    """Return predefined model configurations."""
    return {
        'standard': {
            'name': 'Standard',
            'config': {'prior_type': 'standard'}
        },
        'mixture_k5': {
            'name': 'Mixture (K=5)',
            'config': {'prior_type': 'mixture', 'num_components': 5}
        },
        'mixture_k10': {
            'name': 'Mixture (K=10)',
            'config': {'prior_type': 'mixture', 'num_components': 10}
        },
        'mixture_k20': {
            'name': 'Mixture (K=20)',
            'config': {'prior_type': 'mixture', 'num_components': 20}
        },
    }


def main():
    args = parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = ROOT_DIR / 'artifacts' / 'comparisons' / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")
    
    # Determine which models to compare
    if args.config:
        comparison_config = load_comparison_config(args.config)
        if comparison_config is None:
            print("Error: Could not load config file")
            sys.exit(1)
        models_to_compare = comparison_config['models']
        data_config = comparison_config.get('data', {})
        num_samples = data_config.get('num_samples', args.num_samples)
        num_labeled = data_config.get('num_labeled', args.num_labeled)
        epochs = data_config.get('epochs', args.epochs)
        seed = data_config.get('seed', args.seed)
    elif args.models:
        # Use predefined models specified on command line
        predefined = get_predefined_models()
        models_to_compare = {}
        for model_key in args.models:
            if model_key in predefined:
                model_def = predefined[model_key]
                models_to_compare[model_def['name']] = model_def['config']
            else:
                print(f"Warning: Unknown model '{model_key}', skipping")
        
        if not models_to_compare:
            print("Error: No valid models specified")
            sys.exit(1)
        
        num_samples = args.num_samples
        num_labeled = args.num_labeled
        epochs = args.epochs
        seed = args.seed
    else:
        # Default: compare standard vs mixture_k10
        models_to_compare = {
            'Standard': {'prior_type': 'standard'},
            'Mixture (K=10)': {'prior_type': 'mixture', 'num_components': 10},
        }
        num_samples = args.num_samples
        num_labeled = args.num_labeled
        epochs = args.epochs
        seed = args.seed
    
    print(f"\nComparing {len(models_to_compare)} models: {', '.join(models_to_compare.keys())}")
    
    # Prepare data
    X_train, y_semi, y_true = prepare_data(num_samples, num_labeled, seed)
    
    # Train all models
    trained_models = {}
    histories = {}
    summaries = {}
    
    for name, model_config in models_to_compare.items():
        # Convert hidden_dims list to tuple if present
        if 'hidden_dims' in model_config and isinstance(model_config['hidden_dims'], list):
            model_config['hidden_dims'] = tuple(model_config['hidden_dims'])
        
            # Only set global training settings if not specified in model config
            if 'max_epochs' not in model_config:
                model_config['max_epochs'] = epochs
            if 'random_seed' not in model_config:
                model_config['random_seed'] = seed
            if 'patience' not in model_config:
                model_config['patience'] = epochs  # No early stopping
        
        config = SSVAEConfig(**model_config)
        
        model, history, summary = train_model(name, config, X_train, y_semi, output_dir)
        
        trained_models[name] = model
        histories[name] = history
        summaries[name] = summary
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_loss_comparison(histories, output_dir)
    plot_latent_spaces(trained_models, X_train, y_true, output_dir)
    recon_paths = plot_reconstructions(trained_models, X_train, output_dir)
    
    # Save summaries
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f"  Saved: {summary_path}")
    
    # Generate report
    config_info = {
        'timestamp': timestamp,
        'num_samples': num_samples,
        'num_labeled': num_labeled,
        'epochs': epochs,
        'seed': seed,
        'models': models_to_compare,
    }
    generate_report(summaries, histories, config_info, output_dir, recon_paths=recon_paths)
    
    print(f"\n{'='*60}\nComparison complete! Results: {output_dir}\n{'='*60}\n")


if __name__ == '__main__':
    main()
