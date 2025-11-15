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
- **Interactive trainer** (`rcmvae.application.runtime.interactive`) is used for on-demand training from the dashboard, but only the live preview is accessible.
- **Experiment pipeline** (`use_cases/experiments/run_experiment.py`) remains the only workflow that produces the full suite of visualisations, metrics, and markdown reports. Outputs are stored in timestamped run folders under `use_cases/experiments/results/`.
- There is no automated bridge between dashboard-triggered training runs and the experiment visualisation pipeline yet.

---

## 3. Known Gaps

1. **Callback validation warnings**: Dash emits "nonexistent object" warnings on pages that do not include dashboard-only components (`training-status`, `training-poll`, etc.). Functionality is unaffected but UX is noisy.
2. **Configuration coverage**: The UI exposes only a subset of the newer components/decoders/priors/loss settings recently added under `src/rcmvae/domain`.
3. **Run history cohesion**: Dashboard-triggered trainings update `history.json` but do not generate the richer artifacts (embeddings, τ plots, reconstructions) that the CLI pipeline produces.
4. **Experiment browsing**: No UI surface to inspect historical experiment outputs; users must browse the filesystem and open reports manually.
5. **Incremental label context**: Label additions are captured in `labels.csv`, but runs are not versioned against the label set that produced them, making it hard to audit how labels evolved over time.

---

## 4. Roadmap (Proposed)

### 4.1 Platform Hardening
- Define `app.validation_layout` with hidden placeholders for all callback IDs (e.g., stores, buttons, status divs) to eliminate warnings without reintroducing duplicate elements.
- Optional follow-up: partition callback registration per page or migrate high-traffic callbacks to pattern-matching IDs for better isolation.

### 4.2 Configuration Experience Upgrade
- Build an adapter that surfaces available components/priors/decoders/losses from the registry layer (`src/rcmvae/domain`). Use it to populate dropdowns and tooltips so new building blocks appear automatically.
- Restructure configuration UI into logical sections or tabs (Data, Architecture, Priors, Losses, Advanced) and persist presets so common setups can be reloaded quickly.
- Extend validation feedback inside the form using existing command errors.

### 4.3 Experiment Browser Page
- Introduce a `/runs` page that lists experiment runs by reading manifests under `use_cases/experiments/results/` (leveraging `src/infrastructure/runpaths`).
- Provide per-run detail view: embed generated plots, show metrics tables, link to reports, surface config summary and tags.
- Optional comparison mode to overlay metrics or latent plots from multiple runs.

### 4.4 Cohesive Training History
- When a dashboard training run finishes, invoke the shared reporting pipeline to regenerate visualisations/metrics for the active model (or a lightweight subset stored alongside the model).
- Record a manifest entry per run (timestamp, config snapshot, label version, output paths) so the model’s history reflects its evolution.
- Update UI to present run history (loss curves, metrics, artifact gallery) per model, with clear linkage between label changes and retraining results.

### 4.5 Label Versioning & Provenance
- Version `labels.csv` or maintain a lightweight changelog so each run references the label state it trained on.
- Surface label statistics (count by class, last modified) on both the home page and experiment browser.

### 4.6 Test & Documentation Support
- Expand integration tests to cover model creation with dataset manifests, config updates across new fields, and experiment browser listing.
- Document the run manifest format and the bridging logic between dashboard training and experiment outputs (update `use_cases/dashboard/README.md` after implementation).

---

## 5. Recommended Next Steps (Short Horizon)

1. Implement `validation_layout` to silence cross-page warnings.
2. Build registry-driven metadata adapters and refactor the configuration modal/pages to use them.
3. Design the experiment browser layout and service layer for reading run outputs.
4. Add run-manifest persistence to `CompleteTrainingCommand`, ensuring new runs regenerate visualisations.

These steps create a foundation for the larger goal: an interactive dashboard that can orchestrate experiments, manage evolving label sets, and surface all metrics/visualisations generated by the SSVAE pipeline.
