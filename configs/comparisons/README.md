# Model Comparison Tool

Flexible framework for comparing different SSVAE configurations with automated visualization and reporting.

> **Note:** This is a legacy tool for multi-model comparisons. For the current primary experimentation workflow using single-model configs, see the **[Experiment Guide](../../EXPERIMENT_GUIDE.md)**.

## Quick Start

```bash
# Default comparison (Standard vs Mixture K=10)
python scripts/compare_models.py

# Quick test with fewer epochs
python scripts/compare_models.py --epochs 10 --num-samples 1000

# Compare specific models
python scripts/compare_models.py --models standard mixture_k5 mixture_k10

# Use a config file
python scripts/compare_models.py --config configs/comparisons/mixture_vs_standard.yaml
```

## Command Line Options

- `--config PATH`: Path to YAML config file
- `--num-samples N`: Number of training samples (default: 5000)
- `--num-labeled N`: Number of labeled samples (default: 50)
- `--epochs N`: Training epochs (default: 30)
- `--seed N`: Random seed (default: 42)
- `--models [...]`: Predefined models to compare

## Predefined Models

Use with `--models` flag:

- `standard`: Standard Gaussian prior
- `mixture_k5`: Mixture prior with K=5 components
- `mixture_k10`: Mixture prior with K=10 components
- `mixture_k20`: Mixture prior with K=20 components

Example:
```bash
python scripts/compare_models.py --models standard mixture_k5 mixture_k10 mixture_k20
```

## Configuration Files

Create YAML configs in `configs/comparisons/`:

```yaml
description: "Your comparison description"

data:
  num_samples: 5000
  num_labeled: 50
  epochs: 30
  seed: 42

models:
  ModelName1:
    prior_type: standard
    latent_dim: 2
  
  ModelName2:
    prior_type: mixture
    num_components: 10
    latent_dim: 2
```

Any `SSVAEConfig` parameter can be specified per model.

## Output

Results are saved to `artifacts/comparisons/{timestamp}/`:

- `loss_comparison.png`: Multi-panel loss curves
- `latent_spaces.png`: Latent space visualizations
- `<model>_reconstructions.png`: Original vs reconstruction grids
- `summary.json`: Final metrics for each model
- `COMPARISON_REPORT.md`: Comprehensive markdown report

## Examples

### Compare Mixture Components

```bash
python scripts/compare_models.py --models standard mixture_k5 mixture_k10 mixture_k20 --epochs 20
```

### Custom Config

```yaml
# configs/comparisons/my_experiment.yaml
data:
  num_samples: 3000
  num_labeled: 30
  epochs: 15

models:
  Baseline:
    prior_type: standard
    hidden_dims: [128, 64]
  
  Mixture_Small:
    prior_type: mixture
    num_components: 5
    hidden_dims: [128, 64]
  
  Mixture_Large:
    prior_type: mixture
    num_components: 15
    hidden_dims: [256, 128]
```

Run with:
```bash
python scripts/compare_models.py --config configs/comparisons/my_experiment.yaml
```

## Extending

### Add New Predefined Models

Edit `get_predefined_models()` in `scripts/compare_models.py`:

```python
'my_model': {
    'name': 'My Model',
    'config': {'prior_type': 'mixture', 'num_components': 7}
}
```

### Custom Visualizations

Add functions to `scripts/comparison_utils.py` and call from `compare_models.py`.

## Requirements

- Python 3.11+
- JAX, Flax, Optax (SSVAE dependencies)
- matplotlib, seaborn (visualization)
- pyyaml (optional, for YAML configs)

Install YAML support:
```bash
poetry add pyyaml
```

## How Comparisons Work (Under the Hood)

### Data Preparation & Reproducibility

**All models in a comparison use identical data:**

1. **MNIST is loaded once** - Full training set (60,000 samples) is loaded
2. **Fixed random seed** - Using the specified seed (default: 42), samples are selected
3. **Subsampling** - `num_samples` are randomly drawn from the 60k (e.g., 5000 samples)
4. **Label masking** - Of those samples, only `num_labeled` have their labels revealed (e.g., 50 labeled)
5. **Train/val split** - The same 80/20 split is used by all models

**This ensures fair comparison:**
- Same data distribution
- Same labeled/unlabeled split
- Same validation set
- Same random seed for reproducibility

### Configuration Constraints

**Number of labeled samples (`num_labeled`):**
- ✓ Can be any value ≤ `num_samples`
- ✓ Must be ≤ 60,000 (MNIST training set size)
- ⚠️ Very small values (< 10) may cause unstable training
- 💡 Typical range: 10-500 for semi-supervised experiments

**Number of samples (`num_samples`):**
- ✓ Can be any value ≤ 60,000
- ⚠️ Smaller datasets train faster but may not show mixture component benefits
- 💡 Recommended: 1000-10000 for quick experiments, 30000+ for production

**Why subsampling matters:**
```python
# Example: Your config specifies
num_samples: 5000
num_labeled: 50

# What happens internally:
# 1. Load MNIST: 60,000 samples available
# 2. Random seed selects indices [42317, 7821, ...] (5000 total)
# 3. Create labels: y_semi = [NaN, NaN, 7, NaN, ..., 3, NaN] 
#    └─ Only 50 positions have real labels (e.g., indices [42, 199, ...])
# 4. All models train on THIS EXACT data
```

### Ensuring Fair Comparisons

**Shared base configuration:**
All models inherit these defaults (unless overridden):
- `latent_dim: 2`
- `hidden_dims: (256, 128, 64)`
- `batch_size: 128`
- `learning_rate: 0.001`
- `max_epochs: 30`
- `random_seed: 42`

**Model-specific overrides:**
Only parameters you specify in the config differ between models:
```yaml
models:
  Standard:
    prior_type: standard  # Only this differs
  
  Mixture:
    prior_type: mixture   # And this
    num_components: 10
```

**What gets compared:**
- Same architecture (unless you change `hidden_dims` per model)
- Same optimizer & hyperparameters (unless overridden)
- Same data & splits
- **Only the prior distribution changes**

### Mixture Diagnostics

For `prior_type: mixture`, the training run also saves:

- `diagnostics/component_usage.npy`: empirical usage (mean `q(c|x)` over validation)
- `diagnostics/component_entropy.npy`: mean responsibility entropy
- `diagnostics/pi.npy`: learned π (softmax of `pi_logits`)
- `diagnostics/latent.npz`: if `latent_dim==2`, includes `z_mean`, labels and `q_c`

The report summarizes π and usage; use these to verify Dirichlet/usage penalties are having the intended effect.

### Training Mechanics

**For each model:**
1. Config is created: `base_config + model_specific_overrides`
2. SSVAE is instantiated with that config
3. Model trains on the shared dataset
4. Checkpoint saved to `{model_name}_checkpoint.ckpt`
5. History & summary extracted

**Sequential training:**
- Models train one after another (not parallel)
- Each gets fresh initialization (no weight sharing)
- JAX compilation happens once per model type

### Memory & Performance Notes

**Batch size considerations:**
- Default: 128 (works on most GPUs)
- Smaller batches → slower training, more stable gradients
- Larger batches → faster training, requires more GPU memory

**Component count impact:**
- More components (K=20 vs K=5) → slightly slower training
- More parameters in the encoder head
- Minimal difference for K ≤ 20

**Dataset size impact:**
```
1,000 samples × 10 epochs → ~1 minute (CPU), ~20 seconds (GPU)
5,000 samples × 30 epochs → ~5 minutes (CPU), ~1 minute (GPU)
30,000 samples × 50 epochs → ~30 minutes (CPU), ~5 minutes (GPU)
```

## Troubleshooting

### GPU Compilation Errors

If you encounter JAX/XLA GPU compilation errors (`ptxas exited with non-zero error code`), you can:

1. **Run on CPU** (slower but stable):
   ```bash
   JAX_PLATFORMS=cpu python scripts/compare_models.py --epochs 10
   ```

2. **Clear JAX cache** and retry:
   ```bash
   rm -rf ~/.cache/jax* /tmp/jax*
   python scripts/compare_models.py
   ```

3. **Reduce batch size** if GPU memory is limited:
   ```yaml
   # In your config file
   models:
     YourModel:
       batch_size: 64  # Default is 128
   ```

### Validation Errors

**"num_labeled cannot exceed num_samples":**
- Fix: Ensure `num_labeled ≤ num_samples` in your config

**"num_samples exceeds dataset size (60000)":**
- Fix: MNIST training set has 60,000 samples max
- Use `num_samples ≤ 60000`

**"Empty validation set":**
- Fix: Increase `num_samples` (validation is 20% of data)
- Minimum recommended: `num_samples ≥ 100`
