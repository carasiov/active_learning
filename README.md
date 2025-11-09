# Active Learning – Semi-Supervised VAE (JAX/Flax)


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

**👤 I'm a researcher interested in the theory:**
- Start → [Conceptual Model](docs/theory/conceptual_model.md) - High-level vision and mental model
- Then → [Mathematical Specification](docs/theory/mathematical_specification.md) - Precise mathematical formulations
- Status → [Implementation Roadmap](docs/theory/implementation_roadmap.md) - Current implementation vs. full vision

**💻 I'm a developer extending the codebase:**
- Start → [System Architecture](docs/development/architecture.md) - Design patterns and component structure
- Then → [Implementation Guide](docs/development/implementation.md) - Module-by-module reference
- How-to → [Extending the System](docs/development/extending.md) - Step-by-step tutorials for adding features

**🔬 I'm running experiments:**
- Start → [Experiment Guide](experiments/README.md) - Primary workflow (configuration → execution → interpretation)
- Dashboard → [Interactive Interface](use_cases/dashboard/README.md) - Web-based active learning (future primary)

**🎓 I'm new to the project:**
- Quick Start → See [Experiment Guide](experiments/README.md) for installation and first run
- Theory → [Conceptual Model](docs/theory/conceptual_model.md) for understanding the approach
- Code → [System Architecture](docs/development/architecture.md) for navigating the codebase


### 🔧 Dashboard Guides

- **[Dashboard Overview](use_cases/dashboard/README.md)** - Interactive interface features and workflows
- **[Dashboard Development](use_cases/dashboard/docs/DEVELOPER_GUIDE.md)** - Internal architecture and debugging
- **[Dashboard Extensions](use_cases/dashboard/docs/AGENT_GUIDE.md)** - Adding custom commands and UI components
