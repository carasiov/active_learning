"""Quick test of TauClassifier functionality."""
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from src.ssvae.components.tau_classifier import (
    TauClassifier,
    accumulate_soft_counts,
    compute_tau_from_counts,
    tau_supervised_loss,
    predict_from_tau,
)

def test_tau_classifier():
    """Test basic TauClassifier functionality."""
    print("Testing TauClassifier...")

    # Setup
    num_components = 5
    num_classes = 3
    batch_size = 10
    alpha_0 = 1.0

    # Create classifier
    classifier = TauClassifier(
        num_components=num_components,
        num_classes=num_classes,
        alpha_0=alpha_0,
    )

    # Initialize
    rng = random.PRNGKey(42)
    dummy_responsibilities = jnp.ones((batch_size, num_components)) / num_components
    variables = classifier.init(rng, dummy_responsibilities, training=True)
    params = variables['params']

    print(f"✓ Initialized TauClassifier")
    print(f"  τ shape: {params['tau'].shape}")
    print(f"  Expected: ({num_components}, {num_classes})")

    # Test forward pass
    responsibilities = random.uniform(rng, (batch_size, num_components))
    responsibilities = responsibilities / jnp.sum(responsibilities, axis=1, keepdims=True)

    logits = classifier.apply(variables, responsibilities, training=False)
    print(f"\n✓ Forward pass successful")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, {num_classes})")

    # Test soft count accumulation
    labels = jnp.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=jnp.int32)
    counts = accumulate_soft_counts(responsibilities, labels, num_classes)
    print(f"\n✓ Soft count accumulation successful")
    print(f"  Counts shape: {counts.shape}")
    print(f"  Expected: ({num_components}, {num_classes})")
    print(f"  Counts sum: {jnp.sum(counts):.2f} (should be ~{batch_size})")

    # Test τ computation
    tau = compute_tau_from_counts(counts, alpha_0)
    print(f"\n✓ τ computation successful")
    print(f"  τ shape: {tau.shape}")
    print(f"  Row sums (should all be 1.0): {jnp.sum(tau, axis=1)}")

    # Test supervised loss
    loss = tau_supervised_loss(responsibilities, labels, tau, weight=1.0)
    print(f"\n✓ Supervised loss computation successful")
    print(f"  Loss: {loss:.4f}")

    # Test prediction
    predictions, probs = predict_from_tau(responsibilities, tau)
    print(f"\n✓ Prediction successful")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Probabilities shape: {probs.shape}")
    print(f"  Sample predictions: {predictions[:5]}")
    print(f"  Sample labels:      {labels[:5]}")

    # Test accuracy
    accuracy = jnp.mean(predictions == labels)
    print(f"  Random accuracy: {accuracy:.2%}")

    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)

if __name__ == "__main__":
    test_tau_classifier()
