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
  - Aggregates training status, metrics, and dataset stats for quick monitoring.
- **Experiment browser** (`use_cases/dashboard/pages/experiments.py`)
  - Lists generated runs, metrics summaries, and links into experiment artifacts.

### 1.2 Global Behaviors
- **Routing** is handled in `app.py` with a single callback that switches between home, model dashboard, training hub, config page, and experiments based on the URL.
- **Callback registration** is centralised in `app.py`. Ensure `validation_layout` contains hidden placeholders for callback IDs referenced on other pages to avoid Dash warnings.
- **Styling** relies on the custom CSS block in `app.py` plus Bootstrap defaults.

---

## 2. Backend & Data Flow

### 2.1 State Management
- **Global state** (`use_cases/dashboard/core/state.py`)
  - `AppState` holds the model registry, the active model, and a page-level cache.
  - Synchronisation uses a module-level `state_lock` (`threading.RLock`).
  - `CommandDispatcher` wraps all mutations, validates inputs, and records history.
- **Model metadata** (`use_cases/dashboard/core/state_models.py`)
  - `ModelMetadata` stores dataset size, labeled count, total epochs, latest loss, etc.
  - `ModelState` snapshots the SSVAE instance, config, dataset preview, training status, and history.

### 2.2 Persistence Layer (`use_cases/dashboard/core/model_manager.py`)
- Maintains a directory per model under `use_cases/artifacts/models/`.
- Persists metadata, configs, history, labels, dataset manifests, and run logs.

### 2.3 Commands of Note (`use_cases/dashboard/core/commands.py`)
- **CreateModelCommand**: Samples data deterministically and seeds artifacts.
- **LoadModelCommand**: Loads configs, checkpoints, dataset manifest, and labels into memory.
- **UpdateModelConfigCommand** / **StartTrainingCommand** / **CompleteTrainingCommand**: Manage configuration changes and orchestrate training, run generation, and history updates.

### 2.4 Training & Metrics
- **Interactive trainer** (`src/rcmvae/application/runtime/interactive.py`) powers retraining and preserves optimizer state.
- **Legacy checkpoints** merge safely via `src/rcmvae/application/services/checkpoint_service.py`.
- **Run generation** (`use_cases/dashboard/core/run_generation.py`) leverages experiment infrastructure to produce reports/plots on completion.
- **Run manifest** (`use_cases/dashboard/core/model_runs.py`) tracks runs per model.

---

## 3. Known Gaps

1. ~~**Callback validation warnings**~~: ✅ **Completed November 2025** - Added all missing placeholders to `validation_layout`.
2. ~~**Configuration coverage**~~: ✅ **Completed November 2025** - Full metadata-driven config UI in `pages/training.py` and Training Hub now exposes all backend options via `core/config_metadata.py`.
3. **Run comparison tooling**: Experiment browser lacks side-by-side comparisons, tagging, filtering.
4. **Label provenance**: Runs capture label version, but UI doesn't visualise label history or diffs.

---

## 4. Roadmap (Proposed)

### 4.1 Platform Hardening ✅ **Completed November 2025**
- ✅ Defined `validation_layout` placeholders for all callback IDs.
- Future: Consider page-specific callback modules or pattern-matching IDs for better isolation.

### 4.2 Configuration Experience Upgrade ✅ **Completed November 2025**
- ✅ Generated form controls from `core/config_metadata.py` (registry-driven metadata).
- ✅ Organized UI into sections/tabs (Data, Architecture, Priors, Losses, Advanced).
- ✅ Refactored Training Hub quick controls to use metadata (consistent with config page).
- Future: Persist presets for quick setup.

### 4.3 Experiment Browser Enhancements ✅ **Completed November 2025**
- ✅ Added model and tag filter dropdowns with clear filters button.
- ✅ Backend helpers (`get_available_models()`, `get_available_tags()`) in `core/experiment_catalog.py`.
- ✅ Callbacks support URL-based and dropdown-based filtering.
- ✅ Dynamic filter indicator shows active filters and result counts.
- Future: Add date range filtering and run comparison views.
- Future: Provide artifact downloads for large items.

### 4.4 Cohesive Training History ✅ **Completed November 2025**
- ✅ Surface per-model run history in main dashboard (timeline/cards).
- ✅ Link runs to experiment browser entries and label versions.

### 4.5 Label Versioning & Provenance
- Version `labels.csv` or track diffs for each run.
- Display label statistics on home and experiment pages.

### 4.6 Test & Documentation Support
- Extend integration tests for model creation, config updates, experiment browser.
- Keep `README.md`, `docs/AGENT_GUIDE.md`, and this plan updated as features land.

---

## 5. Recommended Next Steps (Short Horizon)

1. ✅ ~~Implement `validation_layout` placeholders (warning cleanup).~~
2. ✅ ~~Build metadata-driven configuration UI.~~
3. ✅ ~~Enhance experiment browser with comparisons/filters.~~
4. ✅ ~~Surface run history within dashboard layout.~~
5. **Expose label provenance and statistics.** ← Next priority

These steps move the dashboard toward being the primary interface for interactive active learning while keeping the CLI experiment pipeline aligned.
