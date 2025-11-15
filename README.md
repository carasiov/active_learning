# Active Learning – Semi-Supervised VAE (JAX/Flax)

> **Documentation Overview**: This README provides the narrative overview and entry points for this project's comprehensive documentation network. For how to work effectively with this codebase and navigate the documentation graph, see [AGENTS.md](AGENTS.md).

## Purpose & Vision

This project is a research-oriented sandbox for **mixture-structured latent spaces with component specialization**: different components (channels) should learn distinct regions, modes, or labels in latent space. We use [VAE-style generative models](src/rcmvae/application/model_api.py) with [mixture/component-based priors](src/rcmvae/domain/priors) so that individual channels can specialize on MNIST digits.

The long-term goal is a [web application](use_cases/dashboard/app.py) that connects three things tightly:
- model training and configuration,
- visualization and diagnostics,
- human-in-the-loop corrections (labeling, curriculum changes, active learning).

### Primary goals

- Keep the conceptual model explicit and simple so architectural decisions stay obvious and extendable.
- Make experimentation low-friction: swap priors, curricula, or architectures via [configuration](src/rcmvae/domain/config.py), not code edits.
- Expose latent behavior directly through [visualizations](src/infrastructure/visualization) and experiment-management tooling.

### End-user experience (target)

- Multiple dashboard pages on top of the experiment results:
  - latent “microscopes” (2D projections, uncertainty overlays; see `core` plots in [visualization](src/infrastructure/visualization/core/plots.py)),
  - component specialization views (per-channel lenses; see [mixture plots](src/infrastructure/visualization/mixture/plots.py)),
  - τ-matrix summaries (component → label mapping; see [τ plots](src/infrastructure/visualization/tau/plots.py)),
  - curriculum controls,
  - interactive labeling and active learning tools.
- Today: `run_experiment.py` and generated reports/figures are the main interface.
- Tomorrow: the dashboard becomes the primary way to explore runs and launch new ones.

## Ways to Use It

There are two main workflows:

1. **Experiment CLI (current primary workflow)**  
   - Configure an experiment via YAML under [`use_cases/experiments/configs/`](use_cases/experiments/configs).
   - Run [`use_cases/experiments/run_experiment.py`](use_cases/experiments/run_experiment.py) to train and evaluate.
   - Inspect the generated `REPORT.md` and the timestamped run directory:
     - latent plots,
     - mixture/component diagnostics,
     - τ-matrix visualizations,
     - metrics and logs.

2. **Dashboard App (planned primary interface)**  
   - Web UI on top of the same run/result structure.
   - Browse, filter, and search runs; open a run to see its latent/component/τ views and metrics (see [Dashboard Overview](use_cases/dashboard/README.md)).
   - Use interactive labeling and curriculum controls in the browser to trigger new experiments.

For more details on the underlying concepts (mixture-structured latents, responsibilities, τ-classifier), see:

- `docs/theory/conceptual_model.md`
- `docs/theory/mathematical_specification.md`

## Active Learning Loop

The intended human-in-the-loop workflow is:

1. **Diagnose**  
   Use latent/component/τ plots plus uncertainty and reconstruction metrics to find ambiguous or interesting regions.

2. **Intervene**  
   Relabel points, flag OOD samples, or adjust curriculum parameters (weights, annealing schedules, sampling strategy).

3. **Retrain**  
   Run new experiments or fine-tune checkpoints with the updated supervision and curriculum.

4. **Re-visualize**  
   Inspect how component specialization, uncertainties, and cluster structure changed; repeat as needed.

This loop is the core connection between model, curriculum, and UI.

The roadmap in `docs/theory/implementation_roadmap.md` tracks which parts of this vision are implemented and which are still planned.


# Project Structure
```
active_learning_showcase/
│
├── src/rcmvae/                  # 🧠 Core Model Layer
│   ├── domain/                  #    Configs, components, priors, network math
│   ├── application/             #    api/, runtime/, services/ subpackages (factory/trainer/diagnostics)
│   ├── utils/                   #    Device helpers (JAX runtime setup)
│   └── adapters/                #    Bridges into CLI/dashboard tooling
│
├── src/infrastructure/          # ♻️ Shared Infrastructure (dashboard + experiments)
│   ├── logging/                 #    Structured logging setup
│   ├── metrics/                 #    Registry + default metric providers
│   ├── visualization/           #    Plotting registry & implementations
│   └── runpaths/                #    Experiment run directory schema helpers
│
├── use_cases/                   # Product-facing workflows
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
└── docs/                        # 📖 Documentation network (theory → implementation → usage)
    ├── theory/                  #    Conceptual foundations & math
    └── development/             #    Architecture & implementation guides
```

### Component Relationships
```
┌─────────────────────────────────────────────────────────────┐
│                      SSVAE Model Core                       │
│     (src/rcmvae/domain + src/rcmvae/application + utils)    │
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

### Dataset Defaults
- Experiment configs now load the full 70k MNIST dataset by default (downloaded via OpenML and cached locally).
- To force the lighter sklearn digits fallback (needed only for fully offline CI), set `data.dataset_variant: "digits"` in your config.
- The experiment runner records which dataset source was used in the run header so reports remain self-describing.
