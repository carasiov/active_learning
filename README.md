# Active Learning – Semi-Supervised VAE (JAX/Flax)

> **Documentation Overview**: This README provides the narrative overview and entry points for this project's comprehensive documentation network. For how to work effectively with this codebase and navigate the documentation graph, see [AGENTS.md](AGENTS.md).

---

# Project Structure
```
active_learning_showcase/
│
├── src/ssvae/                   # 🧠 Core Model (JAX/Flax)
│   ├── models.py                #    SSVAE class (public API)
│   ├── config.py                #    SSVAEConfig (50+ hyperparameters)
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
├── use_cases/
│   ├── experiments/             # 🔬 Experimentation Workflow
│   │   ├── src/                 #    Implementation (CLI, pipeline, metrics, viz, IO)
│   │   ├── configs/             #    Experiment configurations
│   │   ├── data/                #    Dataset loaders (MNIST)
│   │   ├── results/             #    Experiment outputs (timestamped)
│   │   └── run_experiment.py    #    Backward-compatible CLI entry
│   │
│   └── dashboard/               # 🎛️ Interactive Interface (Future Focus)
│       ├── app.py               #    Web-based active learning interface
│       ├── core/                #    State management & commands
│       ├── pages/               #    Dashboard UI pages
│       └── docs/                #    Dashboard-specific documentation
│
└── docs/                        # 📖 Documentation
    ├── theory/                  #    Conceptual foundations & math
    └── development/             #    Architecture & implementation guides
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
     │ (use_cases/      │          │  (use_cases/       │
     │  experiments/)   │          │   dashboard/)      │
     │                  │          │                    │
     │  Current primary │          │  Future primary    │
     │  workflow for    │          │  interface once    │
     │  experimentation │          │  features stable   │
     └────────────────────┘          └────────────────────┘
```

**Current Reality:** Experimentation happens via [`use_cases/experiments/run_experiment.py`](use_cases/experiments/run_experiment.py) for rapid iteration and validation.

**Target State:** Dashboard becomes the primary interface for interactive active learning once model features stabilize.

---

## Documentation Network

This project has a layered documentation structure (see [AGENTS.md](AGENTS.md) for how to navigate effectively):

**Theory Layer** (Stable Foundations):
- [Conceptual Model](docs/theory/conceptual_model.md) - Design vision and mental model
- [Mathematical Specification](docs/theory/mathematical_specification.md) - Precise formulations
- [Implementation Roadmap](docs/theory/implementation_roadmap.md) - Current status vs full vision

**Implementation Layer** (Current Patterns):
- [System Architecture](docs/development/architecture.md) - Design patterns and component structure
- [Implementation Guide](docs/development/implementation.md) - Module-by-module reference
- [Extending the System](docs/development/extending.md) - Step-by-step tutorials for adding features

**Usage Layer** (Workflows):
- [Experiment Guide](use_cases/experiments/README.md) - Primary workflow (configuration → execution → interpretation) with modular CLI/pipeline/registry structure
- [Dashboard Guide](use_cases/dashboard/README.md) - Interactive interface (future primary)

---

### 🔧 Dashboard Guides

- **[Dashboard Overview](use_cases/dashboard/README.md)** - Interactive interface features and workflows
- **[Dashboard Development](use_cases/dashboard/docs/DEVELOPER_GUIDE.md)** - Internal architecture and debugging
- **[Dashboard Extensions](use_cases/dashboard/docs/AGENT_GUIDE.md)** - Adding custom commands and UI components

---

## Working with This Codebase

For how to navigate the documentation network effectively, understand what to trust when information conflicts, and learn implicit knowledge not obvious from linear reading, see **[AGENTS.md](AGENTS.md)**.
