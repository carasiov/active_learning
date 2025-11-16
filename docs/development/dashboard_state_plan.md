# Dashboard State & Next Steps (November 2025)

This note captures the current status of the interactive dashboard and its supporting backend, plus the roadmap for the next round of work. It is intended to be self-contained so engineers can come up to speed without cross-referencing prior threads.

---

## 1. Current Capabilities

### 1.1 UI Surface
- **Home page** (`use_cases/dashboard/pages/home.py`)
  - Lists all models tracked by `ModelManager`, sorted by `last_modified`.
  - Provides model creation modal with dataset controls (total samples, labeled samples, sampling seed).
  - Displays per-model summary (dataset size, labeled count, epochs, latest loss) and navigation controls.
- **Model dashboard** (`use_cases/dashboard/pages/layouts.py`)
  - Three-pane layout (data preview, latent visualizations, controls) for the active model.
  - Training controls expose reconstruction/kl weights, epoch target, and launch/stop buttons.
  - Status area renders recent training messages and selected sample metadata.
- **Training configuration page** (`use_cases/dashboard/pages/training.py`)
  - Read/write form for core hyperparameters (batch size, epochs, architecture choices, loss weights, regularization, advanced flags).
  - Save action persists the config via `UpdateModelConfigCommand` and warns when architecture changes require restart.
- **Training hub page** (`use_cases/dashboard/pages/training_hub.py`)
  - Aggregates training status, metrics, and dataset stats for quick monitoring (unchanged from earlier iteration).

### 1.2 Global Behaviors
- **Routing** is handled in `app.py` with a single callback that switches between home, model dashboard, training hub, and config page based on the URL.
- **Callback registration** is centralised in `app.py`. Callbacks assume the presence of components defined in the main dashboard layout; navigating to other pages triggers benign Dash warnings because those IDs fall out of the rendered tree (see §4.1).
- **Styling** relies on the custom CSS block in `app.py` plus Bootstrap defaults.

---

## 2. Backend & Data Flow

### 2.1 State Management
- **Global state** (`use_cases/dashboard/core/state.py`)
  - `AppState` holds the model registry, the active model, and a page-level cache.
  - Synchronisation uses a module-level `state_lock` (`threading.RLock`).
  - `CommandDispatcher` wraps all mutations, validates inputs, and records history.
- **Model metadata** (`use_cases/dashboard/core/state_models.py`)
  - `ModelMetadata` now stores `dataset_total_samples` and `dataset_seed` alongside existing fields.
  - `ModelState` snapshots the SSVAE instance, config, dataset preview, training status, and history.

### 2.2 Persistence Layer (`use_cases/dashboard/core/model_manager.py`)
- Maintains a directory per model under `use_cases/artifacts/models/`.
- Persists:
  - `metadata.json` (lightweight registry data),
  - `config.json` (serialised `SSVAEConfig`),
  - `history.json` (loss curves),
  - `labels.csv` (user-labeled points),
  - `dataset.json` (deterministic sampling manifest: indices, labeled positions, seed).
- Provides helpers to load/save the above, delete models, and list all registered models.

### 2.3 Commands of Note (`use_cases/dashboard/core/commands.py`)
- **CreateModelCommand**
  - Sanitises model ID; samples MNIST deterministically based on provided seed/size via `use_cases.experiments.data.mnist.mnist` helpers.
  - Writes dataset manifest, pre-populates `labels.csv`, and registers metadata.
- **LoadModelCommand**
  - Loads or instantiates SSVAE with persisted config.
  - Replays dataset manifest to reconstruct the preview subset and labels.
  - Updates metadata with current labeled count and total samples.
- **UpdateModelConfigCommand / StartTrainingCommand / CompleteTrainingCommand**
  - Continue to manage hyperparameter edits and queue training runs, but they do **not** yet trigger report/visualisation regeneration.

### 2.4 Training & Metrics
- **Interactive trainer** (`rcmvae.application.runtime.interactive`) still powers dashboard retraining, now instrumented to capture responsibilities/π when mixture priors are active and to measure wall-clock training time.
- **Dashboard run pipeline** (`use_cases/dashboard/core/run_generation.py`) reuses the experiment infrastructure to render figures, compute metrics, and emit Markdown reports whenever `CompleteTrainingCommand` fires. Outputs land in the canonical `use_cases/experiments/results/{run_id}` directories and are catalogued per model.
- **Run manifest** (`use_cases/dashboard/core/model_runs.py`) persists a lightweight history (`runs.json`) under each model directory, recording timestamps, config snapshots, label version, and artifact paths for traceability.

---

## 3. Known Gaps

1. **Callback validation warnings**: Dash emits "nonexistent object" warnings on pages that do not include dashboard-only components (`training-status`, `training-poll`, etc.). Functionality is unaffected but UX is noisy.
2. **Configuration coverage**: The UI exposes only a subset of the newer components/decoders/priors/loss settings recently added under `src/rcmvae/domain`.
3. **Run comparison tooling**: The experiment browser surfaces single runs, but there is no side-by-side or aggregate comparison UI yet (latent overlays, metric deltas, etc.).
4. **Incremental label context**: Label additions are captured in `labels.csv`, but runs are not yet visualised alongside the label snapshot they used, making it harder to audit how labels evolved over time.

---

## 4. Roadmap (Proposed)

### 4.1 Platform Hardening
- Define `app.validation_layout` with hidden placeholders for all callback IDs (e.g., stores, buttons, status divs) to eliminate warnings without reintroducing duplicate elements.
- Optional follow-up: partition callback registration per page or migrate high-traffic callbacks to pattern-matching IDs for better isolation.

### 4.2 Configuration Experience Upgrade
- Build an adapter that surfaces available components/priors/decoders/losses from the registry layer (`src/rcmvae/domain`). Use it to populate dropdowns and tooltips so new building blocks appear automatically.
- Restructure configuration UI into logical sections or tabs (Data, Architecture, Priors, Losses, Advanced) and persist presets so common setups can be reloaded quickly.
- Extend validation feedback inside the form using existing command errors.

### 4.3 Experiment Browser Page *(Completed — Nov 2025)*
- `/experiments` route renders a run catalog sourced from `use_cases/experiments/results/`, including inline figure previews, metrics tables, and config snapshots.
- Refresh action rescans the filesystem; detail view falls back gracefully if artifacts are missing.
- **Follow-ups**: add comparisons/filters (by tag, model, prior) and expose download actions for large artifacts that exceed inline limits.

### 4.4 Cohesive Training History *(Completed — Nov 2025)*
- `CompleteTrainingCommand` now funnels every dashboard retraining through the shared metrics/visualisation stack and appends a manifest row per model.
- Run metadata captures label version, training time, and architecture code so experiment provenance is auditable from the dashboard or CLI.
- **Follow-ups**: surface per-model run history within the main dashboard, including quick links into the experiment browser.

### 4.5 Label Versioning & Provenance
- Version `labels.csv` or maintain a lightweight changelog so each run references the label state it trained on.
- Surface label statistics (count by class, last modified) on both the home page and experiment browser.

### 4.6 Test & Documentation Support
- Expand integration tests to cover model creation with dataset manifests, config updates across new fields, and experiment browser listing.
- Document the run manifest format and the bridging logic between dashboard training and experiment outputs (update `use_cases/dashboard/README.md` after implementation).

---

## 5. Recommended Next Steps (Short Horizon)

1. **(Done)** Implement `validation_layout` so cross-page callbacks no longer emit warnings.
2. **(Done)** Build registry-driven metadata adapters and refactor the configuration UI to render from metadata.
3. Extend the experiment browser with comparison/filter tooling (select multiple runs, tag filters, export helpers).
4. Surface model-specific run history inside the dashboard layout (e.g., collapsible timeline or thumbnail gallery).
5. Wire label provenance into the UI (show label version counts, diff from previous run) and document the manifest format.

These steps create a foundation for the larger goal: an interactive dashboard that can orchestrate experiments, manage evolving label sets, and surface all metrics/visualisations generated by the SSVAE pipeline.
