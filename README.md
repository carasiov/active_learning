# Active Learning – Semi-Supervised VAE (JAX/Flax)

> Modular semi-supervised variational autoencoder for learning from predominantly unlabeled data.  
> JAX/Flax implementation.

---

## What is this?

### Current Implementation Status

The repository provides:
- **Semi-supervised VAE** (JAX/Flax) with standard and Priors v1 mixture prior
  - Learnable mixture weights π (via `prior/pi_logits`, no weight decay)
  - Conditional decoder via concatenation `[z; e_c]` and exact expected reconstruction over components
- **Losses (Priors v1)**
  - `KL_z(q(z|x)||N(0,I))` and `KL_c(q(c|x)||π)` with separate weights
  - Optional Dirichlet MAP on π (`dirichlet_alpha`, `dirichlet_weight`)
  - Optional usage sparsity on empirical component usage
  - Reported auxiliary metric `loss_no_global_priors` (recon + KL only)
- **Training infrastructure** for incremental/interactive runs with curriculum support
- **Experiment scripts** to compare configurations and generate reports
  - Reports include loss curves, latent plots, reconstruction grids, and mixture diagnostics (π, usage, entropies)
- **Dashboard scaffold** (`use_cases/dashboard/`) for interactive labeling interface

## Project Structure

```
active_learning_showcase/
│
├── src/ssvae/                   # 🧠 Core Model (JAX/Flax)
│   ├── models.py                #    SSVAE class (public API)
│   ├── config.py                #    SSVAEConfig (25+ hyperparameters)
│   └── components/              #    Encoder, decoder, classifier (factory pattern)
│
├── src/training/                # 🔄 Training Infrastructure
│   ├── trainer.py               #    Training loop with early stopping
│   ├── losses.py                #    Loss functions (reconstruction, KL, classification)
│   └── interactive_trainer.py  #    Incremental training for active learning
│
├── src/callbacks/               # 📊 Training Observability
│   ├── logging.py               #    Console & CSV logging
│   └── plotting.py              #    Loss curve visualization
│
├── scripts/                     # 🔬 Experimentation Tools (Current Focus)
│   ├── run_experiment.py        #    Primary experimentation script
│   ├── compare_models.py        #    Legacy multi-model comparison tool
│   └── comparison_utils.py      #    Visualization & reporting utilities
│
├── use_cases/dashboard/         # 🎛️ Interactive Interface (Future Focus)
│   ├── app.py                   #    Web-based active learning interface
│   ├── core/                    #    State management & commands
│   ├── pages/                   #    Dashboard UI pages
│   └── docs/                    #    Dashboard-specific documentation
│
├── configs/                     # ⚙️ Experiment Configurations
│   ├── default.yaml             #    Standard baseline config
│   ├── quick.yaml               #    Fast sanity checks
│   ├── mixture_example.yaml     #    Full mixture features
│   └── comparisons/             #    Legacy multi-model configs
│       └── *.yaml
│
├── data/mnist/                  # 📦 Dataset
│   └── labels.csv               #    Shared label format (Serial, label)
│
├── artifacts/                   # 💾 Outputs
│   ├── experiments/             #    Experiment results (timestamped)
│   ├── comparisons/             #    Legacy multi-model comparisons
│   ├── checkpoints/             #    Standalone model weights
│   └── models/                  #    Dashboard model state
│
└── docs/                        # 📖 Documentation
    └──...
```

### Component Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                      SSVAE Model Core                       │
│  (src/ssvae/ + src/training/ + src/callbacks/)              │
│                                                              │
│  • Configuration-driven architecture                         │
│  • Factory pattern for components                            │
│  • Pure functional training loop                             │
│  • Callback-based observability                              │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               │                              │
     ┌─────────▼────────┐          ┌─────────▼──────────┐
     │ Experiment Tool  │          │     Dashboard      │
     │  (scripts/)      │          │  (use_cases/)      │
     │                  │          │                    │
     │  Current primary │          │  Future primary    │
     │  workflow for    │          │  interface once    │
     │  experimentation │          │  features stable   │
     └──────────────────┘          └────────────────────┘
```

**Current Reality:** Experimentation happens via `scripts/run_experiment.py` for rapid iteration and validation.

**Target State:** Dashboard becomes the primary interface for interactive active learning once model features stabilize.

---


## Quick Start

Get your first results in under 10 seconds:

```bash
# 1. Install dependencies (one-time setup)
poetry install

# 2. Run a quick experiment
JAX_PLATFORMS=cpu poetry run python scripts/run_experiment.py --config configs/quick.yaml
```

**Output:** `artifacts/experiments/baseline_quick_<timestamp>/` with visualizations, metrics, and a human-readable report.

**Next steps:** See [Experiment Guide](#experiment-guide) for detailed workflows and configuration options.

---

## Documentation

### 🚀 Experiment Guide

**Primary workflow for training and evaluation:**
- **[Experiment Guide](EXPERIMENT_GUIDE.md)** - Complete workflow guide: configuration → execution → interpretation

**Quick reference:**
```bash
# Run quick test
JAX_PLATFORMS=cpu poetry run python scripts/run_experiment.py --config configs/quick.yaml

# Full baseline
poetry run python scripts/run_experiment.py --config configs/default.yaml

# Mixture model with evolution tracking
poetry run python scripts/run_experiment.py --config configs/mixture_example.yaml
```

---

### 📖 Understanding the Project

**Theoretical Foundation:**
- **[Conceptual Model](docs/theory/conceptual_model.md)** - High-level vision and mental model for the RCM-VAE architecture
- **[Mathematical Specification](docs/theory/mathematical_specification.md)** - Precise mathematical formulations, objectives, and training protocols
- **[Implementation Roadmap](docs/theory/implementation_roadmap.md)** - Bridge between current implementation and full RCM-VAE system

### 🚀 Getting Started & Usage

**User Guides:**
- **[Getting Started](docs/guides/getting_started.md)** - Installation, setup, and first successful run
- **[Usage Guide](docs/guides/usage.md)** - Dashboard and Python API usage

**Python API example:**

```python
from ssvae import SSVAE, SSVAEConfig

config = SSVAEConfig(latent_dim=2, prior_type="mixture", num_components=10)
model = SSVAE(input_dim=(28, 28), config=config)
history = model.fit(X_train, y_train, "model.ckpt")
z, recon, preds, cert = model.predict(X_test)
```

### 🏗️ Architecture & Development

**For Developers:**
- **[System Architecture](docs/development/architecture.md)** - Design patterns, component structure, and architectural decisions
- **[Implementation Guide](docs/development/implementation.md)** - Module-by-module reference for working with the codebase
- **[Extending the System](docs/development/extending.md)** - Step-by-step tutorials for adding new features (VampPrior, component-aware decoder, etc.)

### 🔧 Specialized Guides

**Tool-Specific Documentation:**
- **[Experiment Guide](EXPERIMENT_GUIDE.md)** - Primary experimentation workflow (config → run → interpret)
- **[Verification Checklist](VERIFICATION_CHECKLIST.md)** - Comprehensive regression testing guide
- **[Dashboard Overview](use_cases/dashboard/README.md)** - Interactive interface features and workflows
- **[Dashboard Development](use_cases/dashboard/docs/DEVELOPER_GUIDE.md)** - Internal architecture and debugging
- **[Dashboard Extensions](use_cases/dashboard/docs/AGENT_GUIDE.md)** - Adding custom commands and UI components

**Infrastructure:**
- **[GPU Setup & Troubleshooting](.devcontainer/README.md)** - Devcontainer, CUDA, device selection

---

## Usage

**Experiment Tool** (current primary workflow):
```bash
# Quick test
JAX_PLATFORMS=cpu poetry run python scripts/run_experiment.py --config configs/quick.yaml

# Full experiment
poetry run python scripts/run_experiment.py --config configs/default.yaml
```

**Interactive Dashboard:**
```bash
poetry run python use_cases/dashboard/app.py
# Open http://localhost:8050
```

**Python API:**
```python
from ssvae import SSVAE, SSVAEConfig

config = SSVAEConfig(latent_dim=2, prior_type="mixture", num_components=10)
model = SSVAE(input_dim=(28, 28), config=config)
history = model.fit(X_train, y_train, "model.ckpt")
z, recon, preds, cert = model.predict(X_test)
```

See the [Experiment Guide](EXPERIMENT_GUIDE.md) for detailed workflows and configuration options.

---

