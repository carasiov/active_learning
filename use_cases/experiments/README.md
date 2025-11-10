# Experiment Guide

**Primary tool for training and evaluating SSVAE models.**

> **New location:** The experimentation toolkit now lives under `use_cases/experiments/` alongside the dashboard so all user-facing workflows share a common root.
>
> **Implementation vs. results:** All code resides in `src/` (CLI, pipeline, metrics, visualization, IO) while generated artifacts land in `results/` to keep the workspace tidy.

This guide covers the practical workflow: edit config → run experiment → interpret results.

---

## Quick Start

Run your first experiment in 3 commands:

```bash
# 1. Install dependencies (one-time)
poetry install

# 2. Run a quick test (7 seconds)
JAX_PLATFORMS=cpu poetry run python use_cases/experiments/run_experiment.py --config use_cases/experiments/configs/quick.yaml

# 3. View results
cat use_cases/experiments/results/baseline_quick_*/REPORT.md
```

**Output location:** `use_cases/experiments/results/baseline_quick_<timestamp>/`

**Next:** Try a full experiment with `use_cases/experiments/configs/mixture_example.yaml`

---

## Run Directory Layout

Each run now writes to a standardized structure so downstream tools know where to look:

```
use_cases/experiments/results/<name>_<timestamp>/
├── config.yaml                # Snapshot of experiment config
├── summary.json               # Structured metrics
├── REPORT.md                  # Human-readable report (links into figures/)
├── artifacts/                 # Checkpoints + diagnostics
├── figures/                   # All plots (loss curves, latent spaces, τ heatmaps…)
│   └── mixture/               # Mixture evolution panels
└── logs/                      # CSV histories or future log streams
```

The layout lives in `src/io/structure.py`, so any new tooling can reuse the same helper.

---

## Configuration

### Example Configs

| Config | Purpose | Runtime | Notes |
|--------|---------|---------|-------|
| `use_cases/experiments/configs/quick.yaml` | Sanity checks | ~7s | 1K samples, 10 epochs |
| `use_cases/experiments/configs/mixture_example.yaml` | Mixture features | ~5min | K=10, history tracking enabled |

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
  use_tau_classifier: true        # Latent-only classifier (mixture only; requires num_components >= num_classes)

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

> **Note:** When `use_tau_classifier: true`, the config validator enforces `num_components >= num_classes`. During training you'll also see an INFO block summarizing the labeled-data regime (zero / few-shot / low-data / standard) plus warnings if there are too few or no labeled samples—these guardrails help interpret τ-classifier results.

---

## Understanding Output

### Directory Structure

```
use_cases/experiments/results/<name>_<timestamp>/
├── config.yaml                              # Snapshot of config used
├── summary.json                             # All metrics (structured)
├── REPORT.md                                # Human-readable report
├── artifacts/
│   ├── checkpoint.ckpt                      # Trained model weights
│   └── diagnostics/checkpoint/              # Model artifacts
│       ├── latent.npz
│       ├── pi.npy
│       ├── component_usage.npy
│       ├── pi_history.npy
│       ├── usage_history.npy
│       └── tracked_epochs.npy
├── figures/
│   ├── loss_comparison.png                  # Training curves
│   ├── latent_spaces.png                    # Latent space by class
│   ├── latent_by_component.png              # Latent space by component (mixture)
│   ├── responsibility_histogram.png         # Responsibility confidence (mixture)
│   ├── model_reconstructions.png            # Input/output samples
│   ├── component_embedding_divergence.png
│   ├── tau_matrix_heatmap.png
│   └── mixture/
│        └── model_evolution.png             # π and usage over time (mixture)
└── logs/                                    # CSV histories (future use)
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

## Module Layout & Extension Points

The experimentation toolkit is now split into small packages to keep code paths obvious:

| Module | Purpose | Typical changes |
|--------|---------|-----------------|
| `src/cli/` | Entry point + argument parsing | Add new CLI flags or composite workflows |
| `src/pipeline/` | Config loading, data prep, training orchestration | Swap trainers, add pre/post hooks |
| `src/metrics/` | Metric registry + default providers | Register new providers with `@register_metric` |
| `src/visualization/` | Plot registry + helper functions | Register new plotters with `@register_plotter` |
| `src/io/` | Results-directory schema + reporting helpers | Customize layout or report template |

### Adding a Metric

1. Implement a function that accepts `src.metrics.registry.MetricContext`.
2. Decorate it with `@register_metric`.
3. Return a nested dict fragment; it will be deep-merged into `summary.json`.

`MetricContext` already exposes the trained model, histories, latent samples, π/responsibilities, and diagnostics directory so you can compute arbitrary aggregates.

### Adding a Plot

1. Implement a function that accepts `src.visualization.registry.VisualizationContext`.
2. Decorate it with `@register_plotter`.
3. Save artifacts under `context.figures_dir`; return any metadata (e.g., reconstruction filenames) that the report should mention.

Plotters run sequentially after training, so new visualizations can be dropped in without touching the CLI or pipeline.

---

## Common Workflows

### 1. Quick Sanity Check

```bash
JAX_PLATFORMS=cpu poetry run python use_cases/experiments/run_experiment.py --config use_cases/experiments/configs/quick.yaml
```

**Expect:**
- Runtime: ~7 seconds
- Accuracy: Random (~10% for MNIST)
- Loss: Decreasing trend

**Use for:** Verifying code changes didn't break anything.

---

### 2. Train a Baseline Model

```bash
poetry run python use_cases/experiments/run_experiment.py --config use_cases/experiments/configs/mixture_example.yaml
```

**Expect:**
- Runtime: ~2 minutes
- Accuracy: Low (semi-supervised with minimal labels)
- `latent_spaces.png`: Some cluster structure visible

**Use for:** Reference point before experimenting with mixture priors.

---

### 3. Train a Mixture Model

```bash
poetry run python use_cases/experiments/run_experiment.py --config use_cases/experiments/configs/mixture_example.yaml
```

**Expect:**
- Runtime: ~5 minutes
- K_eff: Ideally > 5 (check for component collapse if < 2)
- Evolution plots: Shows π and usage dynamics
- `latent_by_component.png`: Color-coded by component assignment

**Use for:** Full mixture features with evolution tracking.

---

### 4. Run τ-Classifier Validation

```bash
poetry run python use_cases/experiments/run_experiment.py \
  --config use_cases/experiments/configs/tau_classifier_validation.yaml
```

**Expect:**
- Same runtime envelope as the mixture example (≈5 min with default settings).
- Console logs describing the labeled-data regime and any warnings about sparse labels.
- `REPORT.md` includes τ-specific metrics (label coverage, components/label, certainty, OOD score) and visualizations (τ heatmap, per-class accuracy).

**Use for:** Evaluating the latent-only classifier, monitoring component specialization, and benchmarking the new vectorized count updates (sub-millisecond per batch even at K=50+).

---

### 5. Create Custom Experiment

```bash
# 1. Copy a base config
cp use_cases/experiments/configs/mixture_example.yaml use_cases/experiments/configs/my_experiment.yaml

# 2. Edit config
# - Update experiment.name and experiment.description
# - Adjust hyperparameters (e.g., kl_weight, num_components)

# 3. Run
poetry run python use_cases/experiments/run_experiment.py --config use_cases/experiments/configs/my_experiment.yaml

# 4. Compare results
# Check use_cases/experiments/results/ for timestamped outputs
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
grep -r '"ablation"' use_cases/experiments/results/*/config.yaml
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
