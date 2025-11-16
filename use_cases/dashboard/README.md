# SSVAE Dashboard - Multi-Model Architecture

Interactive dashboard for semi-supervised learning with multiple independent model experiments.

## Quick Start
```bash
cd /workspaces/active_learning
poetry run python use_cases/dashboard/app.py
```
Open http://localhost:8050

## Key Features

- **Multi-Model Management:** Create, switch, and delete models with isolated state
- **Interactive Labeling:** 60k-point WebGL visualization with click-to-label
- **Background Training:** Live progress updates with graceful stop
- **Configuration:** 17+ hyperparameters with presets

## Project Structure
```
use_cases/dashboard/
├── app.py                 # Entry point
├── core/                  # Infrastructure (state, commands, I/O)
├── pages/                 # Page layouts
├── callbacks/             # Event handlers
├── utils/                 # Helpers (visualization, logging)
├── assets/                # Static files
└── docs/                  # Documentation

artifacts/models/{model_id}/
├── checkpoint.ckpt    # Weights
├── labels.csv         # Labels
├── history.json       # Loss curves
└── metadata.json      # Stats
```

## Routes
- `/` - Model list
- `/model/{id}` - Dashboard
- `/model/{id}/training-hub` - Training
- `/model/{id}/configure-training` - Config

## Development
```bash
# Verbose logging (optional)
export DASHBOARD_LOG_LEVEL=DEBUG
poetry run python use_cases/dashboard/app.py

# Runtime logs (training workers emit structured events here)
tail -f /tmp/ssvae_dashboard.log

# Tests
poetry run python tests/run_dashboard_tests.py
```

### Checkpoint Compatibility

- Legacy checkpoints without the latest optimizer slots are automatically merged with the current template.
- The merge preserves the original pytree structure, so Optax never sees mismatched update/state chains.
- If a checkpoint is too old to merge, the dashboard falls back to a fresh optimizer state so training can continue safely.

---

## 📚 Documentation Map

| Doc | When to read | Highlights |
| --- | --- | --- |
| [Developer Guide](docs/DEVELOPER_GUIDE.md) | Understanding architecture or backend touchpoints | State flow, command pipeline, backend integration notes, debugging toolkit |
| [Agent Guide](docs/AGENT_GUIDE.md) | Implementing or extending features | Command templates, UI recipes, testing checklist |
| [Autonomous Agent Spec](docs/autonomous_agent_spec.md) | Delegating work to an autonomous agent | Operating contract, prioritized backlog, acceptance criteria |
| [Collaboration & Status](docs/collaboration_notes.md) | Picking up ongoing work | Working agreements, debugging playbook, active priorities |
| [Dashboard State Plan](docs/dashboard_state_plan.md) | Roadmap / big-picture context | Current capabilities, gaps, and strategic next steps |

The broader backend design remains documented in `docs/development/architecture.md`.
