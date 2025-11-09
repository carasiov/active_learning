# Experiment Guide

**Primary tool for training and evaluating SSVAE models.**

This guide covers the practical workflow: edit config → run experiment → interpret results.

---

## Quick Start

Run your first experiment in 3 commands:

```bash
# 1. Install dependencies (one-time)
poetry install

# 2. Run a quick test (7 seconds)
JAX_PLATFORMS=cpu poetry run python scripts/run_experiment.py --config configs/quick.yaml

# 3. View results
cat artifacts/experiments/baseline_quick_*/REPORT.md
```

**Output location:** `artifacts/experiments/baseline_quick_<timestamp>/`

**Next:** Try a full experiment with `configs/default.yaml` or `configs/mixture_example.yaml`

---

## Configuration

### Example Configs

| Config | Purpose | Runtime | Notes |
|--------|---------|---------|-------|
| `configs/quick.yaml` | Sanity checks | ~7s | 1K samples, 10 epochs |
| `configs/default.yaml` | Baseline runs | ~2min | Standard prior, 10K samples |
| `configs/mixture_example.yaml` | Mixture features | ~5min | K=10, history tracking enabled |
| `configs/mixture_quick.yaml` | Debug mixture | ~11s | Fast mixture test |

### Config Structure

```yaml
experiment:
  name: "my_experiment"           # Short identifier
  description: "Testing..."       # What you're investigating
  tags: ["baseline", "debug"]     # For organizing results

data:
  dataset: "mnist"
  num_samples: 10000              # Total dataset size
  num_labeled: 50                 # Labeled samples for classification

model:
  # Architecture
  latent_dim: 2
  hidden_dims: [256, 128, 64]

  # Prior type
  prior_type: "standard"          # or "mixture"
  num_components: 10              # Only for mixture prior

  # Training
  max_epochs: 100
  batch_size: 128
  learning_rate: 0.001
  patience: 30                    # Early stopping patience

  # Loss weights
  recon_weight: 1.0
  kl_weight: 0.5
  kl_c_weight: 0.0005             # Component KL (mixture only)

  # Mixture-specific
  mixture_history_log_every: 1    # Track π/usage every N epochs
```

**For detailed parameter descriptions**, see annotated configs in `configs/`.

**For model architecture theory**, see `/docs/theory/`.

---

## Understanding Output

### Directory Structure

```
artifacts/experiments/<name>_<timestamp>/
├── config.yaml                              # Snapshot of config used
├── checkpoint.ckpt                          # Trained model weights
├── summary.json                             # All metrics (structured)
├── REPORT.md                                # Human-readable report
│
├── diagnostics/checkpoint/                  # Model artifacts
│   ├── latent.npz                           # Latent embeddings
│   ├── pi.npy                               # Final π weights (mixture only)
│   ├── component_usage.npy                  # Final usage (mixture only)
│   ├── pi_history.npy                       # π evolution (mixture only)
│   ├── usage_history.npy                    # Usage evolution (mixture only)
│   └── tracked_epochs.npy                   # Epochs tracked (mixture only)
│
└── visualizations/
    ├── loss_comparison.png                  # Training curves
    ├── latent_spaces.png                    # Latent space by class
    ├── latent_by_component.png              # Latent space by component (mixture)
    ├── responsibility_histogram.png         # Responsibility confidence (mixture)
    ├── model_reconstructions.png            # Input/output samples
    └── mixture/
        └── model_evolution.png              # π and usage over time (mixture)
```

### Key Files

**`REPORT.md`** - Start here! Human-readable summary with embedded visualizations.

**`summary.json`** - Structured metrics for programmatic analysis:
```json
{
  "training": {
    "final_loss": 198.27,
    "training_time_sec": 10.8,
    "epochs_completed": 10
  },
  "classification": {
    "final_accuracy": 0.09
  },
  "mixture": {              // Only for mixture prior
    "K_eff": 1.45,
    "active_components": 3,
    "responsibility_confidence_mean": 0.91
  },
  "clustering": {           // Only if latent_dim=2
    "nmi": 0.0,
    "ari": 0.0
  }
}
```

**`checkpoint.ckpt`** - Model weights for inference or resuming training.

---

## Common Workflows

### 1. Quick Sanity Check

```bash
JAX_PLATFORMS=cpu poetry run python scripts/run_experiment.py --config configs/quick.yaml
```

**Expect:**
- Runtime: ~7 seconds
- Accuracy: Random (~10% for MNIST)
- Loss: Decreasing trend

**Use for:** Verifying code changes didn't break anything.

---

### 2. Train a Baseline Model

```bash
poetry run python scripts/run_experiment.py --config configs/default.yaml
```

**Expect:**
- Runtime: ~2 minutes
- Accuracy: Low (semi-supervised with minimal labels)
- `latent_spaces.png`: Some cluster structure visible

**Use for:** Reference point before experimenting with mixture priors.

---

### 3. Train a Mixture Model

```bash
poetry run python scripts/run_experiment.py --config configs/mixture_example.yaml
```

**Expect:**
- Runtime: ~5 minutes
- K_eff: Ideally > 5 (check for component collapse if < 2)
- Evolution plots: Shows π and usage dynamics
- `latent_by_component.png`: Color-coded by component assignment

**Use for:** Full mixture features with evolution tracking.

---

### 4. Create Custom Experiment

```bash
# 1. Copy a base config
cp configs/default.yaml configs/my_experiment.yaml

# 2. Edit config
# - Update experiment.name and experiment.description
# - Adjust hyperparameters (e.g., kl_weight, num_components)

# 3. Run
poetry run python scripts/run_experiment.py --config configs/my_experiment.yaml

# 4. Compare results
# Check artifacts/experiments/ for timestamped outputs
```

---

## Interpreting Metrics

### Core Metrics

| Metric | Meaning | Good Range | Bad Signs |
|--------|---------|------------|-----------|
| **final_loss** | Total training objective | Decreasing | Increasing or NaN |
| **final_recon_loss** | Reconstruction quality | < 200 (MNIST) | > 300 |
| **final_kl_z** | Latent regularization | 1-10 | > 50 (posterior collapse) |
| **final_accuracy** | Classification accuracy | N/A (few labels) | N/A |
| **training_time_sec** | Wall-clock time | - | - |

### Mixture-Specific Metrics

| Metric | Meaning | Good Range | Bad Signs |
|--------|---------|------------|-----------|
| **K_eff** | Effective components used | 5-9 (if K=10) | < 2 (collapse) |
| **active_components** | Components with >1% usage | 5-10 | 1-2 |
| **responsibility_confidence_mean** | Avg max_c q(c\|x) | 0.5-0.8 | > 0.95 (over-confident) |
| **final_pi_entropy** | Diversity of π weights | 1.5-2.3 | < 0.5 (uniform prior collapsed) |

### Clustering Metrics (latent_dim=2 only)

| Metric | Meaning | Good Range |
|--------|---------|------------|
| **nmi** | Normalized Mutual Information | 0.3-0.8 (higher = better cluster-class alignment) |
| **ari** | Adjusted Rand Index | 0.2-0.6 (higher = better cluster-class alignment) |

**Note:** Clustering metrics compare latent clusters to true class labels. Only computed when `latent_dim=2` for visualization purposes.

---

## Regression Indicators

Watch for these warning signs:

### Component Collapse (Mixture Models)
**Symptoms:**
- K_eff ≈ 1
- active_components = 1
- responsibility_confidence_mean > 0.95

**Fixes:**
- Reduce `kl_c_weight` (try 0.0001 instead of 0.0005)
- Increase `dirichlet_alpha` (try 2.0 or 5.0)
- Use NEGATIVE `component_diversity_weight` (try -0.05) to encourage diversity

---

### Posterior Collapse
**Symptoms:**
- final_kl_z < 0.1
- Latent space shows no structure
- Reconstructions poor despite low reconstruction loss

**Fixes:**
- Reduce `kl_weight` (try 0.1 instead of 0.5)
- Add KL annealing: `kl_c_anneal_epochs: 10`

---

### Training Instability
**Symptoms:**
- Loss oscillates wildly
- NaN losses
- Training crashes

**Fixes:**
- Reduce `learning_rate` (try 0.0001)
- Reduce `batch_size` (try 64)
- Add weight decay: `weight_decay: 0.0001`

**For comprehensive regression testing**, see `VERIFICATION_CHECKLIST.md`.

---

## Troubleshooting

### JAX Device Errors

**Error:** `No GPU/TPU found`

**Solution:** Use CPU mode:
```bash
JAX_PLATFORMS=cpu poetry run python scripts/run_experiment.py --config <config>
```

---

### Out of Memory (OOM)

**Error:** `CUDA out of memory` or similar

**Solutions:**
1. Reduce batch size: `batch_size: 64` → `batch_size: 32`
2. Reduce samples: `num_samples: 10000` → `num_samples: 5000`
3. Use CPU mode: `JAX_PLATFORMS=cpu`

---

### Evolution Plots Missing

**Expected files not created:**
- `pi_history.npy`
- `usage_history.npy`
- `visualizations/mixture/model_evolution.png`

**Solution:** Verify mixture config has:
```yaml
model:
  prior_type: "mixture"
  mixture_history_log_every: 1
```

---

### Slow Training

**Solutions:**
1. Use GPU (remove `JAX_PLATFORMS=cpu`)
2. Increase batch size: `batch_size: 256`
3. Reduce dataset: `num_samples: 5000`
4. Use quick config for testing

---

## Advanced Topics

### Experiment Metadata

Use tags for organizing experiments:

```yaml
experiment:
  name: "ablation_kl_weight_01"
  description: "Testing reduced KL weight impact on posterior collapse"
  tags: ["ablation", "kl-tuning", "2025-11"]
```

Organize results by grepping tags:
```bash
grep -r '"ablation"' artifacts/experiments/*/config.yaml
```

---

### History Tracking Configuration

Control evolution plot granularity:

```yaml
model:
  mixture_history_log_every: 5  # Track every 5 epochs (faster, less storage)
```

**Trade-offs:**
- `log_every: 1` - Detailed evolution plots, more I/O overhead
- `log_every: 10` - Coarse evolution plots, minimal overhead

---

### Hyperparameter Tuning Guidelines

**Start with these ranges:**

| Parameter | Conservative | Aggressive | Notes |
|-----------|--------------|------------|-------|
| `kl_weight` | 0.1 | 1.0 | Higher = stronger regularization |
| `kl_c_weight` | 0.0001 | 0.001 | Too high causes component collapse |
| `learning_rate` | 0.0001 | 0.001 | JAX/Flax Adam default: 0.001 |
| `num_components` | 5 | 20 | More components = harder to utilize all |

**Tuning strategy:**
1. Start with `configs/default.yaml`
2. Change ONE parameter at a time
3. Run 3 times (check stability)
4. Compare `summary.json` metrics

---

## Related Documentation

- **Theory & Architecture:** `/docs/theory/` - Mathematical specifications and design vision
- **Regression Testing:** `VERIFICATION_CHECKLIST.md` - Comprehensive test procedures
- **Code Internals:** `/docs/development/` - Module-by-module implementation guide
- **Issues & Bugs:** GitHub Issues - Report problems or request features
