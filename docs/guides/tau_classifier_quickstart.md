# τ-Classifier Quick Start Guide

This guide will get you running the τ-classifier validation experiments in 5 minutes.

## Prerequisites

- Python 3.8+
- Poetry installed (`pip install poetry`)
- Git repository cloned
- Terminal access

## Quick Setup

```bash
# Install all dependencies using Poetry
poetry install
```

## Run Validation Experiment

```bash
# Run the validation experiment (compares z-based vs τ-based classifiers)
poetry run python experiments/validate_tau_classifier.py
```

**Expected runtime:** 10-15 minutes on CPU, 2-3 minutes on GPU

**What it does:**
- Trains two MNIST models (1000 labeled / 60K total samples)
- Model 1: Mixture VAE with standard z-based classifier
- Model 2: Mixture VAE with τ-based classifier
- Compares accuracy, certainty, and per-class performance

## Expected Output

```
================================================================================
VALIDATION EXPERIMENT: z-based vs τ-based Classifier
================================================================================
Dataset: MNIST semi-supervised (1000 labeled / 60000 total)
Architecture: Mixture VAE with 20 components
Component-aware decoder: True
Training epochs: 50
================================================================================

Training: Mixture + Z-based Classifier
================================================================================
Starting training session...
Detected 1000 labeled samples.
Epoch 1/50: loss=245.3, classification_loss=0.82
...
Epoch 50/50: loss=89.2, classification_loss=0.12

Results:
--------------------------------------------------------------------------------
Test Accuracy: 92.34%
Mean Certainty: 0.8234
Per-class Accuracy: [87.45%, 95.23%]
--------------------------------------------------------------------------------

Training: Mixture + Tau-based Classifier
================================================================================
...
τ-classifier: Accumulated 50000.0 total soft counts over training.

Results:
--------------------------------------------------------------------------------
Test Accuracy: 93.87%
Mean Certainty: 0.8712
Per-class Accuracy: [89.12%, 96.01%]
--------------------------------------------------------------------------------

================================================================================
COMPARISON SUMMARY
================================================================================
Metric                         Z-based         τ-based         Δ
--------------------------------------------------------------------------------
Test Accuracy                  92.34%          93.87%          +1.53%
Mean Certainty                 0.8234          0.8712          +0.0478
Min Class Accuracy             87.45%          89.12%          +1.67%
Max Class Accuracy             95.23%          96.01%          +0.78%
Final Val Cls Loss             0.1234          0.0987          -0.0247
================================================================================

✅ SUCCESS: τ-classifier shows significant improvement!

Results saved to: experiments/results/tau_classifier_validation/comparison_results.txt
τ matrix visualization saved to: diagnostics/ssvae/tau_analysis.png
```

## Interpret Results

### Success Criteria

**✅ PASS if:**
- Test accuracy improvement: Δ > -1.0% (τ matches or exceeds z-based)
- Mean certainty improvement: Δ > 0.0 (better uncertainty quantification)
- Training completes without errors
- τ matrix visualization shows non-uniform associations

**⚠️ INVESTIGATE if:**
- Test accuracy drops by more than 1%
- τ matrix stays uniform (all values ≈ 0.10 for 10 classes)
- Training diverges or NaN losses occur

### Visualizations Generated

Check these files after the experiment:

```bash
# τ matrix heatmap + component usage
diagnostics/ssvae/tau_analysis.png

# Detailed text summary
diagnostics/ssvae/tau_summary.txt

# Comparison results
experiments/results/tau_classifier_validation/comparison_results.txt

# Saved model weights
experiments/results/tau_classifier_validation/Mixture_+_Z-based_Classifier.pkl
experiments/results/tau_classifier_validation/Mixture_+_Tau-based_Classifier.pkl
```

## Understanding τ Matrix Visualization

The `tau_analysis.png` file shows two panels:

**Panel 1: τ Matrix Heatmap**
- Rows = Components (0-19)
- Columns = Labels (0-9 for MNIST digits)
- Values = Association strength τ_{c,y}
- **Interpretation:** Multiple components can serve the same label

Example:
```
Component 3 → Digit 0: 0.82  (specializes on thin "0"s)
Component 7 → Digit 0: 0.91  (specializes on thick "0"s)
Component 9 → Digit 0: 0.76  (specializes on oval "0"s)
```

**Panel 2: Component Usage**
- Bar chart showing how often each component is used
- High bars = frequently used components
- Low bars = under-utilized or free components

## Quick Train Your Own Model

```python
from ssvae import SSVAE, SSVAEConfig
from utils import load_mnist

# Configure with τ-classifier
config = SSVAEConfig(
    prior_type="mixture",
    use_tau_classifier=True,
    num_components=20,
    use_component_aware_decoder=True,
    max_epochs=50,
)

# Load data
(x_train, y_train), (x_test, y_test) = load_mnist()

# Train (τ automatically updated each epoch)
model = SSVAE(input_dim=(28, 28), config=config)
model.fit(x_train, y_train, weights_path="my_model.pkl")

# Predict
latent, recon, predictions, certainty = model.predict(x_test)
print(f"Accuracy: {(predictions == y_test).mean():.2%}")
print(f"Mean certainty: {certainty.mean():.4f}")
```

## Common Issues

### Issue 1: JAX CPU-only installation

If you see warnings about CUDA not found but have a GPU:

```bash
# Add JAX GPU version to your pyproject.toml dependencies
poetry add jax[cuda12]
poetry install
```

### Issue 2: ModuleNotFoundError

```bash
# Make sure Poetry installed the package
poetry install

# Verify installation
poetry run python -c "from ssvae import SSVAE; print('Success!')"
```

### Issue 3: Out of memory

Reduce batch size in the experiment:

```python
# In validate_tau_classifier.py, modify base_config:
base_config = {
    ...
    'batch_size': 64,  # Reduce from 128
    ...
}
```

### Issue 4: Slow training

The experiment trains two models for 50 epochs each. To speed up:

```python
# Reduce epochs for quick validation
base_config = {
    ...
    'max_epochs': 20,  # Reduce from 50
    ...
}
```

## Next Steps

After validating the τ-classifier works:

1. **Try different datasets:** Modify the experiment to use CIFAR-10, Fashion-MNIST, etc.
2. **Tune hyperparameters:** Experiment with `num_components`, `tau_alpha_0`, `component_diversity_weight`
3. **OOD detection:** Use `get_ood_score()` to detect out-of-distribution samples
4. **Dynamic labels:** Use `get_free_channels()` to find capacity for new classes
5. **Dashboard integration:** View τ matrix evolution in real-time

## Additional Resources

- **Usage Guide:** [docs/guides/tau_classifier_usage.md](tau_classifier_usage.md) - Complete API reference
- **Implementation Summary:** [docs/implementation/tau_classifier_implementation_summary.md](../implementation/tau_classifier_implementation_summary.md) - Technical details
- **Correctness Verification:** [docs/verification/tau_classifier_correctness_verification.md](../verification/tau_classifier_correctness_verification.md) - Mathematical proofs
- **Mathematical Specification:** [docs/theory/mathematical_specification.md](../theory/mathematical_specification.md) §5 - Theory

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting section](tau_classifier_usage.md#troubleshooting) in the usage guide
2. Review the [verification document](../verification/tau_classifier_correctness_verification.md) for expected behavior
3. File an issue on GitHub with:
   - Error message
   - Steps to reproduce
   - Python version and JAX version
   - Output of `poetry run python -c "import jax; print(jax.devices())"`

---

**Quick Summary:**
```bash
poetry install
poetry run python experiments/validate_tau_classifier.py
```

That's it! You should see comparative results showing τ-classifier performance.
