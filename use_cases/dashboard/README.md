# SSVAE Dashboard

One interface for the complete active learning cycle: explore, label, train, evaluate.
The dashboard replaces scattered scripts and manual CSV editing with a unified UI where you can see your latent space, label uncertain samples, train your model, and immediately see results—all without touching the terminal.

## Quick Start
```bash
poetry run python use_cases/dashboard/app.py
```

Open http://localhost:8050

## Features

- Interactive 60k-point latent space visualization
- Instant color-by toggles (user labels, predicted, true, certainty) powered by client-side palette caching
- Click-to-label workflow (updates CSV immediately)
- Configure training parameters and launch background runs with live status updates
- Real-time loss curve visualization
- Dataset statistics panel with label counts
- Keyboard shortcuts (0-9) for rapid labeling

## Product Vision

- Audience: single-user ML researchers on localhost.
- Goal: one interface to label, train, and evaluate without touching the terminal.
- Full label–train–evaluate cycles happen inside the dashboard; 60k‑point scatter feels smooth.

## Design Principles

- Localhost only: no auth, no separate backend services.
- UI layer only: model, training loops, and callbacks in `src/` remain.
- Compatibility first: share `labels.csv` and checkpoints with the CLI without migrations.
- Fast interactions: cache visuals and patch updates to keep the app responsive.

## Non‑Goals

- Multi‑user deployments, authentication, or remote hosting.
- Changing data formats or checkpoint schema.

## Workflow

1. Browse latent space, click uncertain points
2. Label them (0-9 via keyboard or buttons)
3. Adjust training parameters if desired
4. Click "Start Training"
5. Watch status updates as training progresses
6. Repeat

## Data & Persistence

- Labels live in `data/mnist/labels.csv` with columns `Serial,label`.
- Label updates persist immediately; unlabeled entries are stored as NaN or absent.
- Model checkpoints are read/written under `artifacts/checkpoints/ssvae.ckpt` (Flax format).

## Architecture

- Single-user localhost deployment
- Background threading for training
- State preserved across training sessions
- Integrates with existing CLI tools (train.py, infer.py still work)
- Modular layout:
  - `app.py` orchestrates initialization, layout, and callback registration
  - `state.py` owns shared model/data state, locks, and labeling helpers
  - `layouts.py` builds the Dash component tree, including hidden stores (`color-mode-store`, `color-palette-store`) for instantaneous coloring
  - `callbacks/` groups training, visualization, and labeling callbacks
  - `utils.py` hosts colorization and image encoding helpers

### How It Works (short)

- State stores the full dataset (images, latent, recon, predictions, labels) in memory.
- Training runs in a background thread; progress/events flow through a `Queue`.
- Callbacks poll for messages, update history, and refresh the scatter when latent changes.
- Scatter uses Plotly WebGL with cached figures and `Patch()` updates for fast color changes.

## Performance Notes

- Scatter figure is cached per latent version; color-mode swaps never hit the server thanks to the client-side palette store.
- Polling callback returns `no_update` for unchanged outputs.
- Polling auto-disables when idle (no training, empty queue) to reduce CPU use.

## Constraints & Scale

- MNIST scale (~60k points) for the latent scatter via WebGL.
- Single background training thread; queue-based metrics to the UI.

## Compatibility

- The dashboard and CLI share the labels CSV and checkpoint format.
- You can switch between CLI and dashboard without migration steps.
