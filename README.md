# Active Learning – Semi-Supervised VAE (JAX/Flax)

A JAX/Flax implementation of a semi-supervised variational autoencoder with mixture priors, component-aware decoding, and active learning capabilities.

---

## Getting Started

Choose your path:

### 🔬 Run an Experiment (5 minutes)

```bash
# Install dependencies
poetry install

# Run quick sanity check (~7 seconds)
JAX_PLATFORMS=cpu poetry run python use_cases/experiments/run_experiment.py \
  --config use_cases/experiments/configs/quick.yaml

# View results
cat use_cases/experiments/runs/baseline_quick_*/REPORT.md
```

**→ [Full Experiment Guide](use_cases/experiments/README.md)** for configuration, workflows, and interpreting results

### 🎛️ Launch Interactive Dashboard

```bash
poetry run python use_cases/dashboard/app.py
# Open http://localhost:8050
```

**→ [Dashboard Guide](use_cases/dashboard/README.md)** for features and usage

### 📖 Understand the Theory

**→ [Conceptual Model](docs/theory/conceptual_model.md)** - Mental model and core invariants

**→ [Mathematical Specification](docs/theory/mathematical_specification.md)** - Precise formulations

### 💻 Extend the Core Model

**→ [Development Overview](docs/development/OVERVIEW.md)** - Quick intro to `/src/` codebase

**→ [Architecture](docs/development/architecture.md)** - Design patterns and philosophy

---

# Project Structure

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
├── experiments/                 # 🔬 Experimentation Workflow
│   ├── run_experiment.py        #    Primary experimentation script
│   ├── experiment_utils.py      #    Visualization & reporting utilities
│   ├── configs/                 #    Experiment configurations
│   ├── data/                    #    Dataset loaders (MNIST)
│   └── runs/                    #    Experiment outputs (timestamped)
│
├── use_cases/dashboard/         # 🎛️ Interactive Interface (Future Focus)
│   ├── app.py                   #    Web-based active learning interface
│   ├── core/                    #    State management & commands
│   ├── pages/                   #    Dashboard UI pages
│   └── docs/                    #    Dashboard-specific documentation
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
     │ (experiments/)   │          │  (use_cases/)      │
     │                  │          │                    │
     │  Current primary │          │  Future primary    │
     │  workflow for    │          │  interface once    │
     │  experimentation │          │  features stable   │
     └──────────────────┘          └────────────────────┘
```

**Current Reality:** Experimentation happens via `experiments/run_experiment.py` for rapid iteration and validation.

**Target State:** Dashboard becomes the primary interface for interactive active learning once model features stabilize.

## Documentation Map

**Find the right documentation for your role:**

### 👤 Researchers (Theory Focus)

**Understand the approach:**
- [Conceptual Model](docs/theory/conceptual_model.md) - Mental model and core invariants
- [Mathematical Specification](docs/theory/mathematical_specification.md) - Precise formulations
- [Vision Gap](docs/theory/vision_gap.md) - Current implementation vs. full vision

### 💻 Developers (Extending Core Model)

**Work with `/src/` codebase:**
- [Development Overview](docs/development/OVERVIEW.md) - Quick intro to codebase structure
- [Architecture](docs/development/architecture.md) - Design patterns and philosophy
- [API Reference](docs/development/api_reference.md) - Module-by-module guide
- [Status](docs/development/STATUS.md) - Current implementation status
- [Decisions](docs/development/DECISIONS.md) - Why we chose specific approaches
- [Extending](docs/development/extending.md) - Step-by-step tutorials for adding features

### 🔬 Users (Running Experiments)

**Use the model:**
- [Experiment Guide](use_cases/experiments/README.md) - Batch experimentation workflow
- [Dashboard Guide](use_cases/dashboard/README.md) - Interactive active learning interface

### 🎓 New to the Project?

**Quick paths:**
- **Run first experiment** → [Getting Started](#getting-started) (above)
- **Understand theory** → [Conceptual Model](docs/theory/conceptual_model.md)
- **Navigate code** → [Development Overview](docs/development/OVERVIEW.md)


### 🔧 Dashboard Guides

- **[Dashboard Overview](use_cases/dashboard/README.md)** - Interactive interface features and workflows
- **[Dashboard Development](use_cases/dashboard/docs/DEVELOPER_GUIDE.md)** - Internal architecture and debugging
- **[Dashboard Extensions](use_cases/dashboard/docs/AGENT_GUIDE.md)** - Adding custom commands and UI components
