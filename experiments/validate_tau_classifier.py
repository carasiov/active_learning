"""
Validation experiment comparing z-based vs τ-based classifiers.

This script trains two models on the same dataset:
1. Mixture prior with standard z-based classifier
2. Mixture prior with τ-based classifier

Both use identical hyperparameters except for the classifier type,
allowing direct comparison of classification performance.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from ssvae import SSVAE, SSVAEConfig
from utils import load_mnist


def prepare_semi_supervised_data(x_train, y_train, labeled_per_class=100):
    """Create semi-supervised dataset with limited labels.

    Args:
        x_train: Training images
        y_train: Training labels
        labeled_per_class: Number of labeled samples per class

    Returns:
        x_train, y_train_semi (with NaN for unlabeled)
    """
    y_semi = np.full_like(y_train, np.nan, dtype=np.float32)

    for label in range(10):
        label_indices = np.where(y_train == label)[0]
        selected = np.random.choice(label_indices, size=labeled_per_class, replace=False)
        y_semi[selected] = y_train[selected]

    labeled_count = np.sum(~np.isnan(y_semi))
    print(f"Created semi-supervised dataset: {labeled_count}/{len(y_train)} labeled samples")

    return x_train, y_semi


def train_and_evaluate(config_name, config, x_train, y_train, x_test, y_test, output_dir):
    """Train model and evaluate performance.

    Args:
        config_name: Name for this configuration
        config: SSVAEConfig instance
        x_train, y_train: Training data
        x_test, y_test: Test data
        output_dir: Directory to save results

    Returns:
        Dictionary with metrics
    """
    print("\n" + "="*80)
    print(f"Training: {config_name}")
    print("="*80)

    # Create model
    model = SSVAE(input_dim=(28, 28), config=config)

    # Train
    weights_path = str(output_dir / f"{config_name.replace(' ', '_')}.pkl")
    history = model.fit(
        data=x_train,
        labels=y_train,
        weights_path=weights_path,
        export_history=True,
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    latent, recon, predictions, certainty = model.predict(x_test)

    # Compute metrics
    accuracy = np.mean(predictions == y_test)
    mean_certainty = np.mean(certainty)

    # Classification metrics per class
    per_class_acc = []
    for label in range(10):
        mask = y_test == label
        if np.any(mask):
            class_acc = np.mean(predictions[mask] == label)
            per_class_acc.append(class_acc)

    min_class_acc = np.min(per_class_acc)
    max_class_acc = np.max(per_class_acc)

    # Training metrics
    final_train_loss = history['loss'][-1]
    final_val_loss = history['val_loss'][-1]
    final_train_cls_loss = history['classification_loss'][-1]
    final_val_cls_loss = history['val_classification_loss'][-1]

    results = {
        'config_name': config_name,
        'test_accuracy': accuracy,
        'mean_certainty': mean_certainty,
        'min_class_accuracy': min_class_acc,
        'max_class_accuracy': max_class_acc,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'final_train_cls_loss': final_train_cls_loss,
        'final_val_cls_loss': final_val_cls_loss,
        'epochs_trained': len(history['loss']),
    }

    print("\n" + "-"*80)
    print("Results:")
    print("-"*80)
    print(f"Test Accuracy: {accuracy:.2%}")
    print(f"Mean Certainty: {mean_certainty:.4f}")
    print(f"Per-class Accuracy: [{min_class_acc:.2%}, {max_class_acc:.2%}]")
    print(f"Final Train Loss: {final_train_loss:.2f}")
    print(f"Final Val Loss: {final_val_loss:.2f}")
    print(f"Final Train Cls Loss: {final_train_cls_loss:.4f}")
    print(f"Final Val Cls Loss: {final_val_cls_loss:.4f}")
    print("-"*80)

    return results


def main():
    """Run validation experiment."""
    # Set seed for reproducibility
    np.random.seed(42)

    # Create output directory
    output_dir = Path("experiments/results/tau_classifier_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # Prepare semi-supervised data (100 labels per class = 1000 total)
    x_train_semi, y_train_semi = prepare_semi_supervised_data(
        x_train, y_train, labeled_per_class=100
    )

    # Base configuration (shared by both models)
    base_config = {
        'prior_type': 'mixture',
        'num_components': 20,
        'latent_dim': 16,
        'use_component_aware_decoder': True,
        'component_embedding_dim': 8,
        'component_diversity_weight': -0.05,  # Encourage diversity
        'kl_c_weight': 0.0,  # No KL_c by default
        'kl_c_anneal_epochs': 0,
        'reconstruction_loss': 'bce',
        'recon_weight': 1.0,
        'kl_weight': 1.0,
        'label_weight': 100.0,
        'learning_rate': 1e-3,
        'batch_size': 128,
        'max_epochs': 50,
        'patience': 15,
        'encoder_type': 'dense',
        'decoder_type': 'dense',
        'random_seed': 42,
    }

    # Configuration 1: Standard z-based classifier
    config_z = SSVAEConfig(
        **base_config,
        use_tau_classifier=False,
    )

    # Configuration 2: τ-based classifier
    config_tau = SSVAEConfig(
        **base_config,
        use_tau_classifier=True,
        tau_alpha_0=1.0,
    )

    # Train both models
    results = []

    print("\n" + "="*80)
    print("VALIDATION EXPERIMENT: z-based vs τ-based Classifier")
    print("="*80)
    print(f"Dataset: MNIST semi-supervised (1000 labeled / 60000 total)")
    print(f"Architecture: Mixture VAE with {base_config['num_components']} components")
    print(f"Component-aware decoder: {base_config['use_component_aware_decoder']}")
    print(f"Training epochs: {base_config['max_epochs']}")
    print("="*80)

    # Experiment 1: z-based classifier
    results_z = train_and_evaluate(
        config_name="Mixture + Z-based Classifier",
        config=config_z,
        x_train=x_train_semi,
        y_train=y_train_semi,
        x_test=x_test,
        y_test=y_test,
        output_dir=output_dir,
    )
    results.append(results_z)

    # Experiment 2: τ-based classifier
    results_tau = train_and_evaluate(
        config_name="Mixture + Tau-based Classifier",
        config=config_tau,
        x_train=x_train_semi,
        y_train=y_train_semi,
        x_test=x_test,
        y_test=y_test,
        output_dir=output_dir,
    )
    results.append(results_tau)

    # Summary comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Metric':<30} {'Z-based':<15} {'τ-based':<15} {'Δ':<10}")
    print("-"*80)

    metrics_to_compare = [
        ('Test Accuracy', 'test_accuracy', '.2%'),
        ('Mean Certainty', 'mean_certainty', '.4f'),
        ('Min Class Accuracy', 'min_class_accuracy', '.2%'),
        ('Max Class Accuracy', 'max_class_accuracy', '.2%'),
        ('Final Val Cls Loss', 'final_val_cls_loss', '.4f'),
    ]

    for metric_name, key, fmt in metrics_to_compare:
        z_val = results_z[key]
        tau_val = results_tau[key]
        delta = tau_val - z_val

        # Format values
        if fmt == '.2%':
            z_str = f"{z_val:.2%}"
            tau_str = f"{tau_val:.2%}"
            delta_str = f"{delta:+.2%}"
        else:
            z_str = f"{z_val:{fmt}}"
            tau_str = f"{tau_val:{fmt}}"
            delta_str = f"{delta:+{fmt}}"

        print(f"{metric_name:<30} {z_str:<15} {tau_str:<15} {delta_str:<10}")

    print("="*80)

    # Save results to file
    results_file = output_dir / "comparison_results.txt"
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("τ-CLASSIFIER VALIDATION RESULTS\n")
        f.write("="*80 + "\n\n")

        for result in results:
            f.write(f"\nConfiguration: {result['config_name']}\n")
            f.write("-"*80 + "\n")
            for key, value in result.items():
                if key != 'config_name':
                    f.write(f"{key}: {value}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Test Accuracy Improvement: {results_tau['test_accuracy'] - results_z['test_accuracy']:+.2%}\n")
        f.write(f"Certainty Improvement: {results_tau['mean_certainty'] - results_z['mean_certainty']:+.4f}\n")

    print(f"\nResults saved to: {results_file}")

    # Determine outcome
    acc_improvement = results_tau['test_accuracy'] - results_z['test_accuracy']
    if acc_improvement > 0.01:
        print("\n✅ SUCCESS: τ-classifier shows significant improvement!")
    elif acc_improvement > -0.01:
        print("\n✓ PASS: τ-classifier performs comparably to z-based classifier")
    else:
        print("\n⚠️ WARNING: τ-classifier underperforms. Check training logs.")

    return results


if __name__ == "__main__":
    main()
