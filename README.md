# Active Learning – Semi-Supervised VAE (JAX/Flax)

> Modular semi-supervised variational autoencoder for learning from predominantly unlabeled data.  
> JAX/Flax implementation.

---

## What is this?

This repository contains a **deep learning model** that learns useful representations from datasets with very few labeled examples. 

**The model is the core.** Everything else (comparison tools, dashboard, scripts) exists to experiment with, validate, and interact with the SSVAE model.

**Why does this exist?** While MNIST serves as the proof of concept, the ultimate goal is interactive active learning for high-dimensional text embeddings (device notifications, error messages). See [**Context & Motivation**](docs/CONTEXT.md) for the full story.


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
│   ├── compare_models.py        #    Compare model configurations
│   └── comparison_utils.py      #    Visualization & reporting utilities
│
├── use_cases/dashboard/         # 🎛️ Interactive Interface (Future Focus)
│   ├── app.py                   #    Web-based active learning interface
│   ├── core/                    #    State management & commands
│   ├── pages/                   #    Dashboard UI pages
│   └── docs/                    #    Dashboard-specific documentation
│
├── configs/comparisons/         # ⚙️ Experiment Configurations
│   └── *.yaml                   #    YAML configs for model comparisons
│
├── data/mnist/                  # 📦 Dataset
│   └── labels.csv               #    Shared label format (Serial, label)
│
├── artifacts/                   # 💾 Outputs
│   ├── comparisons/             #    Experiment results (plots, metrics, reports)
│   ├── checkpoints/             #    Model weights
│   └── models/                  #    Dashboard model state
│
└── docs/                        # 📖 Documentation
    ├── IMPLEMENTATION.md        #    Core model architecture & APIreference                       
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
     │  Comparison Tool │          │     Dashboard      │
     │  (scripts/)      │          │  (use_cases/)      │
     │                  │          │                    │
     │  Current primary │          │  Future primary    │
     │  workflow for    │          │  interface once    │
     │  experimentation │          │  features stable   │
     └──────────────────┘          └────────────────────┘
```

**Current Reality:** Experimentation happens via `scripts/compare_models.py` for rapid iteration and validation.

**Target State:** Dashboard becomes the primary interface for interactive active learning once model features stabilize.

---

---

## Quick Start

Get your first results in under 3 minutes:

```bash
# 1. Install dependencies (one-time setup)
poetry install

# 2. Run a model comparison
poetry run python scripts/compare_models.py --models standard mixture_k10 --epochs 10
```

**Output:** `artifacts/comparisons/20241031_143022/` with loss curves, latent visualizations, metrics, and checkpoints.

**Next steps:** [Getting Started Guide](docs/GETTING_STARTED.md) for detailed setup and verification.

---
---

## Usage

All usage patterns are documented in the [**Usage Guide**](docs/USAGE.md):

- **[Comparison Tool](docs/USAGE.md#comparison-tool)** - Command-line experimentation (current primary workflow)
- **[Interactive Dashboard](docs/USAGE.md#interactive-dashboard)** - Web interface for active learning  
- **[Python API](docs/USAGE.md#python-api)** - Programmatic access for custom integration
- **[Legacy Tools](docs/USAGE.md#legacy-tools)** - Single-model CLI scripts

**Quick example:**

```bash
# Compare two models
poetry run python scripts/compare_models.py --models standard mixture_k10 --epochs 10
```

See the [Usage Guide](docs/USAGE.md) for detailed examples, workflows, and troubleshooting.

---

---

---

## Documentation Map

**Start here based on your goal:**

| Get Started | Use Tools | Understand Model | Extend System |
|-------------|-----------|------------------|---------------|
| [**Getting Started**](docs/GETTING_STARTED.md)<br>• Installation<br>• Requirements<br>• Quick start<br>• Verification | [**Usage Guide**](docs/USAGE.md)<br>• Comparison tool<br>• Dashboard<br>• Python API<br>• Workflows | [**Implementation**](docs/IMPLEMENTATION.md)<br>• Architecture<br>• API reference<br>• Components<br>• Patterns | [**Contributing**](docs/CONTRIBUTING.md)<br>• Dev workflow<br>• Adding features<br>• Testing<br>• Code style |

**Context & Philosophy:**

- 📖 [**Context & Motivation**](docs/CONTEXT.md) - Why this architecture? What's the target application? Why MNIST first?

**Specialized Guides:**

- 🐳 [GPU Setup & Troubleshooting](.devcontainer/README.md) - Devcontainer, CUDA, device selection
- 🔬 [Comparison Tool Details](configs/comparisons/README.md) - YAML configs, options, troubleshooting  
- 🎛️ [Dashboard Overview](use_cases/dashboard/README.md) - Features, routing, development
- 🤖 [Dashboard Extensions](use_cases/dashboard/docs/AGENT_GUIDE.md) - Adding commands, UI, callbacks
- 🔧 [Dashboard Internals](use_cases/dashboard/docs/DEVELOPER_GUIDE.md) - Architecture, debugging, state


