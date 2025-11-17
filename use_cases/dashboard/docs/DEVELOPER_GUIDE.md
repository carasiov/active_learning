# Dashboard Developer Guide

> **Purpose:** Spell out how the dashboard is wired together, where it touches the backend, and how to reason about changes safely.

## 1. Architecture at a Glance

- **Entry point** – `use_cases/dashboard/app.py`
  - Configures logging (`DASHBOARD_LOG_LEVEL`), initializes AppStateManager, registers pages, and wires callbacks.
  - Hosts the router callback that renders `/`, `/model/{id}`, `/model/{id}/training-hub`, `/model/{id}/configure-training`, and `/experiments`.
- **State management** – `use_cases/dashboard/core/state_manager.py`
  - `AppStateManager` encapsulates `AppState` with internal `RLock` for thread safety.
  - `CommandDispatcher` orchestrates all mutations atomically.
  - Replaces old global `app_state` and module-level `state_lock` (Phase 3 refactoring, November 2025).
- **Service layer** – `use_cases/dashboard/services/`
  - `ModelService`, `TrainingService`, `LabelingService` handle domain logic.
  - Commands receive services via dependency injection, delegate business logic to services.
- **Command layer** – `use_cases/dashboard/core/commands.py`
  - Every user action is a dataclass with `validate(state, services)` + `execute(state, services)`; commands return a new `AppState` and message.
- **Persistence services** – `core/model_manager.py`, `core/model_runs.py`, `core/run_generation.py`
  - Manage model directories, configs, dataset manifests, run histories, and generated artifacts.
- **UI & callbacks** – `use_cases/dashboard/pages/*`, `use_cases/dashboard/callbacks/*`
  - Layouts describe structure; callbacks translate UI events into commands/background work.

## 2. Backend Integration Points

The dashboard layers on top of the SSVAE backend (`src/rcmvae`). Keep these touchpoints in mind when backend code changes:

| Concern | Dashboard touchpoint | Backend module(s) | Notes |
| --- | --- | --- | --- |
| Model lifecycle | `services/ModelService` → `LoadModelCommand` | `rcmvae/application/model_api.py` | ModelService instantiates `SSVAE`, restores params + optimizer state. |
| Configuration | `core/config_metadata.py`, `state_models.py` | `rcmvae/domain/config.py::SSVAEConfig` | UI fields map directly to dataclass attributes; validation happens both in commands and config docstrings. |
| Training loop | `services/TrainingService`, `callbacks/training_callbacks.py` | `rcmvae/application/services/training_service.py`, `application/runtime/interactive.py` | TrainingService validates and starts training; worker calls `InteractiveTrainer.train`. |
| Checkpoints | `services/ModelService`, `core/model_manager.py` | `rcmvae/application/services/checkpoint_service.py` | Tuple-preserving merge shim keeps legacy checkpoints usable; fallback initializes fresh optimizer state. |
| Labeling | `services/LabelingService` | `core/model_manager.py` (labels.csv) | LabelingService handles label persistence and updates. |
| Losses & metrics | `core/run_generation.py` | `rcmvae/application/services/loss_pipeline.py`, `use_cases/experiments` | Training completion triggers the experiment pipeline to regenerate plots/reports. |
| Data sampling | `CreateModelCommand` → `ModelService` | `use_cases/experiments/data/mnist/mnist.py` | Dataset manifests ensure CLI and dashboard sample the same data. |

Whenever a backend change affects these modules (new priors, additional metrics, altered state), update dashboard config metadata, persistence, and run generation accordingly.

## 3. Dashboard Internals

### 3.1 State & Commands (Refactored November 2025)
- **AppStateManager** (`core/state_manager.py`) manages all state operations:
  - `AppState` tracks model registry, `active_model`, run history cache, and global settings.
  - Internal `RLock` ensures thread safety.
  - `update_state(new_state)` is the **only** way to update state (never assign directly).
- **Service Layer** (`services/`) handles business logic:
  - `ModelService`: Model CRUD, loading, predictions
  - `TrainingService`: Training execution, validation
  - `LabelingService`: Label persistence
- **Commands** (`core/commands.py`) orchestrate state transitions:
  - Receive `state` and `services` parameters via dependency injection.
  - Should be pure: compute the result, build copies of dataclasses, return new state.
  - Delegate domain logic to services, only orchestrate state transitions.
- Register commands through callbacks (e.g., `callbacks/home_callbacks.py`, `callbacks/training_callbacks.py`).

### 3.2 Persistence & Runs
- `ModelManager` serializes every model asset (`metadata.json`, `config.json`, `history.json`, `labels.csv`, dataset manifest).
- `model_runs.py` appends run metadata to `runs.json`; UI reads from this manifest instead of scanning the filesystem.
- `run_generation.py` reuses experiment infrastructure, keeping CLI and dashboard artifacts aligned.

### 3.3 Background Training Worker
- Spawns via `threading.Thread` in `callbacks/training_callbacks.py`.
- Logs structured messages (start/stop/errors) and pushes metrics into `metrics_queue` for polling callbacks.
- Uses try/except to handle `TrainingStoppedException` versus genuine failures; in failure scenarios it prints stack traces to `/tmp/ssvae_dashboard.log` and surfaces user-friendly messages.

### 3.4 Validation Layout
- `app.validation_layout` must include hidden placeholders for any callback outputs absent on certain pages (training status divs, stores, etc.).
- Missing placeholders trigger Dash "nonexistent ID" warnings even though callbacks run.

## 4. Debugging Toolkit

1. `export DASHBOARD_LOG_LEVEL=DEBUG`
2. Launch via `poetry run python use_cases/dashboard/app.py`
3. Reproduce the issue and capture terminal output plus:
   ```bash
   tail -n 200 /tmp/ssvae_dashboard.log
   ```
4. Run focused tests:
   ```bash
   poetry run pytest tests/test_dashboard_config.py tests/test_dashboard_integration.py
   ```
5. For backend mismatches (optimizer state, priors, losses), cross-reference unit tests under `tests/test_*` in the backend.

## 5. Extension Workflow

Use this checklist whenever you implement or review dashboard changes:

1. **Clarify the flow** – what user action, on which page, affects which state field?
2. **Design the command** – add/extend a command in `core/commands.py`; adjust `state_models.py` if new data is stored.
3. **Persist artifacts** – update `model_manager.py`, `model_runs.py`, or add helpers as needed.
4. **Expose configuration** – prefer metadata-driven forms via `core/config_metadata.py` over hard-coded inputs.
5. **Wire UI + callbacks** – update layouts and handlers; ensure `validation_layout` covers new component IDs.
6. **Log + test** – add meaningful logging, run dashboard tests, and capture `/tmp/ssvae_dashboard.log` for verification.
7. **Document updates** – note material changes in `docs/collaboration_notes.md` / `docs/dashboard_state_plan.md`.

## 6. Related Docs & Tests

- Orientation: `../README.md`
- Recipes: `docs/AGENT_GUIDE.md`
- Collaboration status: `docs/collaboration_notes.md`, `docs/dashboard_state_plan.md`
- Backend architecture: `../../../docs/development/architecture.md`
- Core tests: `tests/test_dashboard_config.py`, `tests/test_dashboard_integration.py`

Keeping these integration points in sync ensures the dashboard evolves in lockstep with the SSVAE backend.
