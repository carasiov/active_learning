# Usage Guide

Comprehensive guide to all ways of working with the SSVAE model.

---

## Overview

There are multiple ways to use the SSVAE:

1. **[Experiment Tool](../../EXPERIMENT_GUIDE.md)** - Single-model workflow with config-driven experiments (current primary workflow)
2. **[Interactive Dashboard](#interactive-dashboard)** - Web-based interface for active learning (future primary interface)
3. **[Python API](#python-api)** - Programmatic access for custom integration
4. **[Legacy Tools](#legacy-tools)** - Older comparison and CLI tools (see appendix)

---

## Experiment Tool (Primary Workflow)

**Purpose:** Configuration-driven single-model training and evaluation with comprehensive reporting.

**Status:** Current primary workflow for experimentation and development.

### Quick Start

```bash
# Quick test (7 seconds)
JAX_PLATFORMS=cpu poetry run python experiments/run_experiment.py --config experiments/configs/quick.yaml

# Full baseline
poetry run python experiments/run_experiment.py --config experiments/configs/mixture_example.yaml

# Mixture model with evolution tracking
poetry run python scripts/run_experiment.py --config configs/mixture_example.yaml
```

### Output Structure

Each experiment generates a timestamped directory:

```
artifacts/experiments/<name>_<timestamp>/
├── REPORT.md                    # Human-readable summary
├── config.yaml                  # Configuration snapshot
├── checkpoint.ckpt              # Trained model weights
├── summary.json                 # Structured metrics
├── visualizations/              # Loss curves, latent spaces, reconstructions
└── diagnostics/checkpoint/      # Latent embeddings, mixture evolution
```

### When to Use

✅ **Use experiment tool for:**
- Training and evaluating individual models
- Testing configurations and hyperparameters
- Generating comprehensive analysis reports
- Tracking mixture prior evolution
- Standard experimentation workflow

**For detailed workflow, configuration options, and interpretation guide, see [Experiment Guide](../../EXPERIMENT_GUIDE.md).**

---

### Basic Usage

```bash
# Quick comparison of predefined models
poetry run python scripts/compare_models.py --models standard mixture_k10

# Custom parameters
poetry run python scripts/compare_models.py \
  --num-samples 10000 \
  --num-labeled 100 \
  --epochs 50 \
  --seed 42

# YAML-based configuration (recommended for complex experiments)
poetry run python scripts/compare_models.py --config configs/comparisons/my_experiment.yaml
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--models` | Space-separated predefined model names | `standard mixture_k10` |
| `--config` | Path to YAML configuration file | None |
| `--num-samples` | Total training samples | 5000 |
| `--num-labeled` | Number of labeled samples | 50 |
| `--epochs` | Training epochs | 30 |
| `--seed` | Random seed for reproducibility | 42 |

### Predefined Models

Use with `--models` flag:

- `standard` - Standard Gaussian prior
- `mixture_k5` - Mixture of 5 Gaussians
- `mixture_k10` - Mixture of 10 Gaussians
- `mixture_k20` - Mixture of 20 Gaussians

**Example:**
```bash
poetry run python scripts/compare_models.py --models standard mixture_k5 mixture_k10 mixture_k20
```

### YAML Configuration

For complex experiments, create YAML configs in `configs/comparisons/`:

```yaml
description: "Comparing latent dimensions"

data:
  num_samples: 5000
  num_labeled: 50
  epochs: 30
  seed: 42

models:
  Latent2D:
    latent_dim: 2
    prior_type: mixture
    num_components: 10
  
  Latent10D:
    latent_dim: 10
    prior_type: mixture
    num_components: 10
  
  Latent50D:
    latent_dim: 50
    prior_type: mixture
    num_components: 10
```

Any `SSVAEConfig` parameter can be specified. Run with:

```bash
poetry run python scripts/compare_models.py --config configs/comparisons/latent_dimensions.yaml
```

### Output Structure

Each comparison generates a timestamped directory:

```
artifacts/comparisons/20241031_143022/
├── loss_comparison.png           # Multi-panel loss curves
├── latent_spaces.png             # 2D visualizations (if latent_dim=2)
├── <model>_reconstructions.png   # Original vs reconstruction grids
├── summary.json                  # Final metrics for each model
├── COMPARISON_REPORT.md          # Comprehensive markdown analysis
├── standard_checkpoint.ckpt      # Model weights (standard)
└── mixture_k10_checkpoint.ckpt   # Model weights (mixture)
```


### When to Use

✅ **Use comparison tool for:**
- Testing new architectures or loss functions
- Hyperparameter tuning and ablation studies
- Validating model behavior before dashboard integration
- Generating publication-quality figures
- Batch experimentation (multiple configs)

❌ **Don't use for:**
- Interactive labeling sessions (use dashboard)
- Real-time exploration (use dashboard)
- Non-technical users (use dashboard)

**Learn more:** See [`configs/comparisons/README.md`](../configs/comparisons/README.md) for detailed configuration syntax, troubleshooting, and examples.

---

## Interactive Dashboard

**Purpose:** Web-based interface for interactive active learning workflows.

**Status:** Feature-complete, will become primary interface once core model features stabilize. Currently secondary while experimentation via comparison tool is prioritized.

### Quick Start

```bash
poetry run python use_cases/dashboard/app.py
```

Open http://localhost:8050 in your browser.

### Key Features

**Multi-Model Management:**
- Create, switch, and delete models
- Each model has isolated state (labels, checkpoints, history)
- Compare multiple experiments side-by-side

**Interactive Labeling:**
- 60k-point WebGL scatter plot of latent space
- Click samples to label them
- Keyboard shortcuts (0-9 for digit labels)
- Real-time uncertainty visualization

**Background Training:**
- Train without blocking the UI
- Live progress updates (epoch, loss, metrics)
- Graceful stop/resume
- Checkpoint management

**Configuration:**
- 17+ hyperparameters adjustable via UI
- Preset configurations (standard, mixture)
- Real-time validation
- History tracking

### Typical Workflow

1. **Create a new model** with initial configuration
2. **Load data** (MNIST auto-loaded, or upload custom)
3. **Label a few samples** (click in latent space)
4. **Train** the model with background worker
5. **Evaluate** results via loss curves and metrics
6. **Identify uncertain samples** (low certainty scores)
7. **Label more samples** in uncertain regions
8. **Retrain** and repeat

### Routes

- `/` - Model list and management
- `/model/{id}` - Main dashboard (visualization + labeling)
- `/model/{id}/training-hub` - Training controls and monitoring
- `/model/{id}/configure-training` - Hyperparameter configuration

### When to Use

✅ **Use dashboard for:**
- Interactive active learning sessions
- Demonstrations and presentations
- Real-time exploration of latent space
- Rapid prototyping with visual feedback
- Teaching and education

❌ **Don't use for:**
- Batch experimentation (use comparison tool)
- Automated hyperparameter search (use comparison tool)
- Production model training (use Python API)

**Learn more:**
- [Dashboard Overview](../../use_cases/dashboard/README.md) - Features, architecture, routing
- [Dashboard Developer Guide](../../use_cases/dashboard/docs/DEVELOPER_GUIDE.md) - Internal architecture
- [Dashboard Agent Guide](../../use_cases/dashboard/docs/AGENT_GUIDE.md) - Extension patterns
