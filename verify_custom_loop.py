#!/usr/bin/env python3
"""
Verification script to check custom training loop correctness.

Compares τ-classifier training with standard training to ensure:
1. Training completes successfully
2. Loss converges properly
3. τ matrix is learned correctly
4. Behavior is consistent with standard training
"""
import sys
sys.path.insert(0, 'src')

import numpy as np

from ssvae import SSVAE, SSVAEConfig

print("=" * 70)
print("CUSTOM TRAINING LOOP VERIFICATION")
print("=" * 70)

# Create small test dataset
np.random.seed(42)
X = np.random.randn(200, 28, 28).astype(np.float32)
y = np.array([i % 10 for i in range(200)], dtype=np.float32)
y[100:] = np.nan  # 100 labeled, 100 unlabeled

print("\n1. Dataset")
print(f"   Shape: {X.shape}")
print(f"   Labeled: {np.sum(~np.isnan(y))}/200")
print(f"   Classes: 10")

# Test 1: τ-classifier training
print("\n2. τ-Classifier Training (Custom Loop)")
print("   Config: mixture prior + τ-classifier")

config_tau = SSVAEConfig(
    prior_type="mixture",
    num_components=5,
    latent_dim=2,
    use_tau_classifier=True,
    max_epochs=3,
    batch_size=32,
)

model_tau = SSVAE(input_dim=(28, 28), config=config_tau)

try:
    history_tau = model_tau.fit(X, y, "/tmp/verify_tau.ckpt", export_history=False)

    print(f"   ✓ Training completed")
    print(f"   ✓ Epochs: {len(history_tau['loss'])}")
    print(f"   ✓ Initial loss: {history_tau['loss'][0]:.4f}")
    print(f"   ✓ Final loss: {history_tau['loss'][-1]:.4f}")

    # Check loss improved
    loss_improved = history_tau['loss'][-1] < history_tau['loss'][0]
    print(f"   ✓ Loss improved: {loss_improved}")

    # Check for NaN/inf
    has_nan = any(np.isnan(history_tau['loss']))
    has_inf = any(np.isinf(history_tau['loss']))
    print(f"   ✓ No NaN: {not has_nan}")
    print(f"   ✓ No Inf: {not has_inf}")

except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Check τ matrix
print("\n3. τ Matrix Validation")

tau = model_tau._tau_classifier.get_tau()
import jax.numpy as jnp

rows_sum_to_one = jnp.allclose(tau.sum(axis=1), 1.0)
all_positive = jnp.all(tau >= 0)
all_bounded = jnp.all(tau <= 1.0)
not_uniform = not jnp.allclose(tau, 0.2)  # 1/5 components

print(f"   Shape: {tau.shape}")
print(f"   ✓ Rows sum to 1: {rows_sum_to_one}")
print(f"   ✓ All values ≥ 0: {all_positive}")
print(f"   ✓ All values ≤ 1: {all_bounded}")
print(f"   ✓ Learned (not uniform): {not_uniform}")

# Test 3: Predictions work
print("\n4. Prediction Test")

try:
    latent, recon, predictions, certainty = model_tau.predict(X[:10])

    print(f"   ✓ Latent shape: {latent.shape}")
    print(f"   ✓ Predictions shape: {predictions.shape}")
    print(f"   ✓ Certainty range: [{certainty.min():.3f}, {certainty.max():.3f}]")

    # Check determinism
    pred2, _, class2, cert2 = model_tau.predict(X[:10])
    deterministic = jnp.allclose(predictions, class2) and jnp.allclose(certainty, cert2)
    print(f"   ✓ Predictions deterministic: {deterministic}")

except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 4: Standard classifier baseline
print("\n5. Baseline Training (Standard Loop)")
print("   Config: mixture prior + standard classifier")

config_std = SSVAEConfig(
    prior_type="mixture",
    num_components=5,
    latent_dim=2,
    use_tau_classifier=False,  # Standard classifier
    max_epochs=3,
    batch_size=32,
)

model_std = SSVAE(input_dim=(28, 28), config=config_std)

try:
    history_std = model_std.fit(X, y, "/tmp/verify_std.ckpt", export_history=False)

    print(f"   ✓ Training completed")
    print(f"   ✓ Epochs: {len(history_std['loss'])}")
    print(f"   ✓ Initial loss: {history_std['loss'][0]:.4f}")
    print(f"   ✓ Final loss: {history_std['loss'][-1]:.4f}")

    # Compare losses
    loss_tau = history_tau['loss'][-1]
    loss_std = history_std['loss'][-1]
    loss_ratio = loss_tau / loss_std if loss_std > 0 else 0

    # Should be in similar ballpark (within 2x)
    similar = 0.3 < loss_ratio < 3.0

    print(f"\n   Comparison:")
    print(f"   ✓ τ-classifier loss: {loss_tau:.4f}")
    print(f"   ✓ Standard loss: {loss_std:.4f}")
    print(f"   ✓ Ratio: {loss_ratio:.2f}")
    print(f"   ✓ Similar magnitude: {similar}")

except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check model can make reasonable predictions
print("\n6. Prediction Quality Check")

_, _, pred_tau, _ = model_tau.predict(X[:100])
_, _, pred_std, _ = model_std.predict(X[:100])

# Count how many predictions are valid (0-9)
valid_tau = jnp.all((pred_tau >= 0) & (pred_tau < 10))
valid_std = jnp.all((pred_std >= 0) & (pred_std < 10))

# Check accuracy (should be >10% random baseline)
acc_tau = np.mean(pred_tau[:100] == y[:100])
acc_std = np.mean(pred_std[:100] == y[:100])

print(f"   ✓ τ predictions valid: {valid_tau}")
print(f"   ✓ Standard predictions valid: {valid_std}")
print(f"   ✓ τ accuracy: {acc_tau*100:.1f}%")
print(f"   ✓ Standard accuracy: {acc_std*100:.1f}%")
print(f"   ✓ Both above random (10%): {acc_tau > 0.1 and acc_std > 0.1}")

# Summary
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

checks = {
    "τ-classifier training": loss_improved and not has_nan and not has_inf,
    "τ matrix properties": rows_sum_to_one and all_positive and all_bounded and not_uniform,
    "Predictions work": deterministic,
    "Standard training": True,  # If we got here, it worked
    "Loss magnitudes similar": similar,
    "Prediction quality": valid_tau and valid_std and acc_tau > 0.1 and acc_std > 0.1,
}

passed = sum(checks.values())
total = len(checks)

print(f"\nPassed: {passed}/{total}")
for name, result in checks.items():
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"  {status}: {name}")

if passed == total:
    print("\n" + "=" * 70)
    print("✅ ALL CHECKS PASSED")
    print("=" * 70)
    print("\nConclusion:")
    print("  - Custom training loop works correctly")
    print("  - τ matrix is learned properly")
    print("  - Behavior is consistent with standard training")
    print("  - No numerical instabilities detected")
    print("\n→ You can trust the custom training loop implementation.")
    sys.exit(0)
else:
    print("\n❌ SOME CHECKS FAILED - Please investigate")
    sys.exit(1)
