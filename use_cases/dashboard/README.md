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
- Instant color-by toggles (user labels, predicted, true, certainty)
- Click-to-label workflow (updates CSV immediately)
- Advanced training configuration page with 17+ hyperparameters
- Launch background training runs with live status updates
- Real-time loss curve visualization with optional smoothing
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
3. Adjust training parameters via "⚙️ Advanced Configuration" if desired
4. Click "Start Training"
5. Watch status updates and loss curves as training progresses
6. Repeat

## Data & Persistence

- Labels live in `data/mnist/labels.csv` with columns `Serial,label`.
- Label updates persist immediately; unlabeled entries are stored as NaN or absent.
- Model checkpoints are read/written under `artifacts/checkpoints/ssvae.ckpt` (Flax format).

## Architecture

- Single-user localhost deployment
- Background threading for training
- State preserved across training sessions
- Multi-page app: main dashboard (`/`) and config page (`/configure-training`)
- Integrates with existing CLI tools (train.py, infer.py still work)
- Modular layout:
  - `app.py` orchestrates initialization, layout, and callback registration
  - `state.py` owns shared model/data state, locks, and labeling helpers
  - `layouts.py` builds the main dashboard layout
  - `pages_training.py` builds the advanced configuration page
  - `callbacks/` groups training, visualization, labeling, and config callbacks
  - `utils.py` hosts colorization and image encoding helpers

### How It Works

- State stores the full dataset (images, latent, recon, predictions, labels) in memory.
- Training runs in a background thread; progress/events flow through a `Queue`.
- Callbacks poll for messages, update history, and refresh the scatter when latent changes.
- Scatter uses Plotly WebGL with smart caching for fast rendering.

## Performance

### Scatter Plot Optimization
The dashboard uses intelligent figure caching to minimize render time:

| Scenario | Performance | Notes |
|----------|-------------|-------|
| **First load** | 2-4s | Initial figure build with 60k points |
| **Navigation** | 50-100ms | Cached figure reused |
| **Color mode change** | 50-100ms | Cached colors reused |
| **Point selection** | <10ms | Only highlight marker updates |

**Why it's fast:**
- Figures cached in `app_state["cache"]` persist across navigation
- Color computations cached separately (60k color strings)
- Cache limits (20 figures, 50 color sets) prevent memory growth
- Simplified hover template reduces JSON payload size

**First-load optimization tips:**
- First render must build the figure (unavoidable ~2-4s)
- Subsequent interactions are near-instant due to caching
- Browser refresh clears cache, requiring rebuild

## Troubleshooting

### Scatter plot not showing points
**Symptom:** Empty plot after navigation or reload.

**Cause:** Cache not initialized properly in `app_state`.

**Fix:** Ensure `state.py` initializes cache in `initialize_model_and_data()`:
```python
if "cache" not in app_state:
    app_state["cache"] = {"base_figures": {}, "colors": {}}
```

### Slow scatter updates
**Symptom:** 2-4 second delay on every interaction.

**Cause:** Figure cache not being reused (rebuilding on every callback).

**Check:** `visualization_callbacks.py` should have fast path:
```python
cached_figure = base_figure_cache.get(figure_cache_key)
if cached_figure is not None:
    return cached_figure  # Fast path
```

## Tech Stack

**Framework:** Dash + Plotly (Python)

**Why Dash?**
- Pure Python (no context switching)
- Excellent ML integration (JAX/NumPy)
- WebGL support for 60k points
- Perfect for single-user localhost dashboards

**Considered alternatives:**
- React + FastAPI: More flexible but higher engineering overhead
- Streamlit: Simpler but less control, full page reruns
- Jupyter Widgets: Prototyping only, poor large-data performance

**Verdict:** Dash is appropriate for this use case.

## Constraints & Scale

- MNIST scale (~60k points) for the latent scatter via WebGL.
- Single background training thread; queue-based metrics to the UI.
- Memory: ~500MB for full dataset + predictions + cache.

## Compatibility

- The dashboard and CLI share the labels CSV and checkpoint format.
- You can switch between CLI and dashboard without migration steps.
- Training config changes requiring architecture restart (encoder/decoder type, latent dim) show a warning.
