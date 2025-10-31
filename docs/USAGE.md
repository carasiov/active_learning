# Usage Guide

Comprehensive guide to all ways of working with the SSVAE model.

---

## Overview

There are three primary ways to use the SSVAE:

1. **[Comparison Tool](#comparison-tool)** - Command-line tool for experimenting with model configurations (current primary workflow)
2. **[Interactive Dashboard](#interactive-dashboard)** - Web-based interface for active learning (future primary interface)
3. **[Python API](#python-api)** - Programmatic access for custom integration

---

## Comparison Tool

**Purpose:** Rapidly compare different model configurations, validate new features, and generate analysis artifacts.

**Status:** Current primary workflow for development and experimentation.

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
├── summary.json                  # Final metrics for each model
├── COMPARISON_REPORT.md          # Comprehensive markdown analysis
├── standard_checkpoint.ckpt      # Model weights (standard)
└── mixture_k10_checkpoint.ckpt   # Model weights (mixture)
```

### Common Use Cases

**1. Hyperparameter tuning:**
```bash
poetry run python scripts/compare_models.py \
  --config configs/comparisons/hyperparameter_search.yaml
```

**2. Architecture comparison:**
```yaml
models:
  Dense:
    encoder_type: dense
    hidden_dims: [256, 128, 64]
  
  Conv:
    encoder_type: conv
    hidden_dims: [32, 64, 128]
```

**3. Few-shot learning experiments:**
```bash
# Extreme few-shot (10 labels)
poetry run python scripts/compare_models.py --num-labeled 10 --epochs 50

# Standard semi-supervised (100 labels)
poetry run python scripts/compare_models.py --num-labeled 100 --epochs 30
```

**4. Loss function comparison:**
```yaml
models:
  MSE:
    reconstruction_loss: mse
    recon_weight: 500.0
  
  BCE:
    reconstruction_loss: bce
    recon_weight: 1.0
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
- [Dashboard Overview](../use_cases/dashboard/README.md) - Features, architecture, routing
- [Dashboard Developer Guide](../use_cases/dashboard/docs/DEVELOPER_GUIDE.md) - Internal architecture
- [Dashboard Agent Guide](../use_cases/dashboard/docs/AGENT_GUIDE.md) - Extension patterns

---

## Python API

**Purpose:** Programmatic access for custom workflows, notebooks, and integration into other tools.

### Basic Example

```python
import numpy as np
from ssvae import SSVAE, SSVAEConfig

# Configure model
config = SSVAEConfig(
    latent_dim=2,
    prior_type="mixture",
    num_components=10,
    max_epochs=50,
    learning_rate=1e-3
)

# Initialize model
vae = SSVAE(input_dim=(28, 28), config=config)

# Prepare semi-supervised labels (NaN = unlabeled)
labels = np.full(1000, np.nan)
labels[:10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Only 10 labeled

# Train
history = vae.fit(X_train, labels, "model.ckpt")

# Inference
z, recon, preds, cert = vae.predict(X_test)
```

### Key APIs

**`SSVAE` class:**
- `__init__(input_dim, config)` - Initialize model
- `fit(data, labels, weights_path)` - Train and save checkpoint
- `predict(data, sample=False, num_samples=1)` - Run inference
- `load_model_weights(weights_path)` - Load from checkpoint

**`SSVAEConfig` dataclass:**
- Architecture: `latent_dim`, `hidden_dims`, `encoder_type`, `decoder_type`, `prior_type`
- Loss weights: `recon_weight`, `kl_weight`, `label_weight`, `reconstruction_loss`
- Training: `batch_size`, `learning_rate`, `max_epochs`, `patience`

**Returns:**
- `fit()` → `dict` with keys: `loss`, `val_loss`, `reconstruction_loss`, `kl_loss`, `classification_loss`
- `predict()` → 4-tuple: `(latent, reconstruction, predictions, certainty)`

### Common Patterns

**Pattern 1: Active Learning Loop**

```python
from ssvae import SSVAE, SSVAEConfig

config = SSVAEConfig(latent_dim=2, max_epochs=30)
vae = SSVAE(input_dim=(28, 28), config=config)

# Start with minimal labels
labels = np.full(len(X_pool), np.nan)
labels[:10] = initial_labels

for round in range(5):
    # Train
    vae.fit(X_pool, labels, "model.ckpt")
    
    # Find uncertain samples
    z, recon, preds, cert = vae.predict(X_pool)
    uncertain_idx = cert.argsort()[:10]  # 10 most uncertain
    
    # Label them (oracle or human)
    labels[uncertain_idx] = oracle(X_pool[uncertain_idx])
    
    print(f"Round {round}: {np.sum(~np.isnan(labels))} labels")
```

**Pattern 2: Configuration Search**

```python
configs = [
    SSVAEConfig(prior_type="standard", latent_dim=2),
    SSVAEConfig(prior_type="mixture", num_components=5, latent_dim=2),
    SSVAEConfig(prior_type="mixture", num_components=10, latent_dim=2),
]

results = []
for i, config in enumerate(configs):
    vae = SSVAE(input_dim=(28, 28), config=config)
    history = vae.fit(X_train, labels, f"model_{i}.ckpt")
    results.append({
        'config': config,
        'final_loss': history['loss'][-1],
        'final_accuracy': compute_accuracy(vae, X_test, y_test)
    })

best = min(results, key=lambda x: x['final_loss'])
```

**Pattern 3: Custom Evaluation**

```python
vae = SSVAE(input_dim=(28, 28), config=config)
vae.load_model_weights("trained_model.ckpt")

# Latent space analysis
z, _, _, _ = vae.predict(X_test)
from sklearn.cluster import KMeans
clusters = KMeans(n_clusters=10).fit_predict(z)

# Reconstruction quality
_, recon, _, _ = vae.predict(X_test)
mse = np.mean((X_test - recon) ** 2)
```

### When to Use

✅ **Use Python API for:**
- Jupyter notebooks and research workflows
- Custom active learning strategies
- Integration into larger pipelines
- Automated experimentation
- Production deployment

**Learn more:** See [`docs/IMPLEMENTATION.md`](IMPLEMENTATION.md) for complete API reference, architecture details, and extension patterns.

---

## Legacy Tools

Single-model CLI scripts predating the comparison tool. Still functional but less feature-rich.

### Train Script

```bash
poetry run python use_cases/scripts/train.py \
  --labels data/mnist/labels.csv \
  --weights artifacts/checkpoints/ssvae.ckpt
```

Trains a single model using labels from CSV file (format: `Serial,label`).

### Inference Script

```bash
poetry run python use_cases/scripts/infer.py \
  --weights artifacts/checkpoints/ssvae.ckpt \
  --output data/output_latent.npz \
  --split train
```

Runs inference and saves latent representations to NumPy archive.

### Legacy Viewer

```bash
poetry run python use_cases/scripts/view_latent.py
```

Simple visualization of latent space (predates dashboard).

### Migration Note

For new experiments, prefer:
- **Comparison tool** for experimentation
- **Dashboard** for interactive exploration
- **Python API** for custom workflows

Legacy scripts remain for backward compatibility and simple single-model use cases.

---

## Common Workflows

### Workflow 1: Quick Experiment

```bash
# Test if mixture prior helps
poetry run python scripts/compare_models.py --models standard mixture_k10 --epochs 20
```

### Workflow 2: Hyperparameter Search

1. Create YAML config with parameter grid
2. Run comparison tool with config
3. Analyze `summary.json` for best configuration
4. Use best config in production

### Workflow 3: Active Learning Session

1. Start dashboard: `poetry run python use_cases/dashboard/app.py`
2. Create model with initial configuration
3. Label 10-20 samples manually
4. Train model
5. Sort by uncertainty, label more samples
6. Retrain and repeat until satisfied

### Workflow 4: Custom Integration

1. Use Python API to load data
2. Configure model with `SSVAEConfig`
3. Implement custom training loop or evaluation
4. Export results in desired format

---

## Troubleshooting

### Comparison Tool Issues

**"Comparison hangs or is very slow"**
- Use CPU mode: `JAX_PLATFORMS=cpu python scripts/compare_models.py`
- Reduce samples: `--num-samples 1000`
- Reduce epochs: `--epochs 10`

**"Out of memory"**
- Reduce batch size in YAML config: `batch_size: 64`
- Use fewer samples: `--num-samples 3000`
- Close other GPU applications

### Dashboard Issues

**"Dashboard won't start"**
- Check port 8050 is free: `lsof -i :8050`
- Check logs: `tail -f /tmp/ssvae_dashboard.log`
- Try different port: Edit `app.py` to change port

**"Training stuck or crashes"**
- Check logs for error messages
- Try simpler configuration (standard prior, fewer epochs)
- Restart dashboard

### API Issues

**"Model not converging"**
- Check label format (NaN for unlabeled)
- Increase `max_epochs` or adjust `patience`
- Verify `recon_weight` appropriate for loss type (BCE ~1.0, MSE ~500)

**"Checkpoint errors"**
- Ensure parent directory exists
- Check write permissions
- Use absolute paths for `weights_path`

For more help, see specialized guides in the [Documentation Map](../README.md#documentation-map).
