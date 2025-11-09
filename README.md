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
- Start → [Experiment Guide](EXPERIMENT_GUIDE.md) - Primary workflow (configuration → execution → interpretation)
- Or → [Usage Guide](docs/guides/usage.md) - All available tools (comparison, dashboard, Python API)

**🎓 I'm new to the project:**
- Start → [Getting Started](docs/guides/getting_started.md) - Installation, setup, first run
- Then → Pick a path above based on your goals


### 🔧 Dashboard Guides

- **[Dashboard Overview](use_cases/dashboard/README.md)** - Interactive interface features and workflows
- **[Dashboard Development](use_cases/dashboard/docs/DEVELOPER_GUIDE.md)** - Internal architecture and debugging
- **[Dashboard Extensions](use_cases/dashboard/docs/AGENT_GUIDE.md)** - Adding custom commands and UI components
