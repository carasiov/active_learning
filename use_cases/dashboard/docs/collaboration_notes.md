# Dashboard Collaboration Notes

> **Purpose:** Snapshot of the current dashboard architecture, debugging aids, and working agreements so future sessions can resume quickly.

## System Snapshot (November 2025)

The fuller history, open issues, and roadmap live in [`dashboard_state_plan.md`](dashboard_state_plan.md). This section distills the highlights most relevant when jumping back into day-to-day work:

- **Entry point:** `use_cases/dashboard/app.py`
  - Configures logging via `DASHBOARD_LOG_LEVEL` (default INFO) and boots Dash routes.
  - State management handled by `AppStateManager` in `use_cases/dashboard/core/state_manager.py`
  - Service container provides `TrainingService`, `ModelService`, `LabelingService` instances
- **Architecture (Phase 3 Refactoring - November 2025)**:
  - `AppStateManager` encapsulates all state with internal lock, replaces global `app_state`
  - Service layer (`services/`) handles domain logic (training, model I/O, labeling)
  - Commands use dependency injection, receive `services` parameter
  - Commands call `state_manager.update_state()` exclusively (never direct assignment)
- **State mutations:** Commands in `use_cases/dashboard/core/commands.py` remain the sole way to mutate state.
  - `LoadModelCommand` uses `ModelService.load_model()` and refuses switches during active training
  - `StartTrainingCommand` validates hyperparameters via `TrainingService` before dispatching worker
  - `CompleteTrainingCommand` threads through mixture artifacts and generates experiment outputs
- **Training worker:** `use_cases/dashboard/callbacks/training_callbacks.py::train_worker`
  - Background thread that logs structured start/stop/failure events and publishes metrics into `metrics_queue`.
  - Failures now emit full stack traces (see `/tmp/ssvae_dashboard.log`) and surface user-friendly status messages.
- **Checkpoint compatibility:** `src/rcmvae/application/services/checkpoint_service.py`
  - Legacy checkpoints merge with the current template without corrupting tuple-based optimizer state.
  - When merging fails, the optimizer falls back to the freshly initialised template so Optax never sees mismatched update/state chains.
- **Runtime observability & tests:**
  - File log: `/tmp/ssvae_dashboard.log`; console level set via `DASHBOARD_LOG_LEVEL`.
  - Regression coverage: `tests/test_dashboard_config.py`, `tests/test_dashboard_integration.py`.

## Debugging Playbook

1. **Enable verbose logs** (only when investigating):
   ```bash
   export DASHBOARD_LOG_LEVEL=DEBUG
   poetry run python use_cases/dashboard/app.py
   ```
2. **Capture symptoms:** note the exact UI steps and copy the terminal output.
3. **Collect evidence:**
   ```bash
   tail -n 200 /tmp/ssvae_dashboard.log
   ```
   Attach the tail alongside any screenshots.
4. **Run focused tests** if a change touches core flows:
   ```bash
   poetry run pytest tests/test_dashboard_config.py tests/test_dashboard_integration.py
   ```

## Incremental Collaboration Loop

When resuming work:

1. **Sync plan:** Describe the user journey or feature slice you want to tackle. Mention where it sits (page, command, model, etc.).
2. **Agent prep:** I review the relevant docs (`use_cases/dashboard/README.md`, architecture/development notes, config docstrings) and inspect code before editing.
3. **Code changes:** I implement scoped modifications, keep logging/tests up to date, and report back with:
   - Files touched + rationale
   - Commands/tests run
   - Next validation steps for you (usually UI interaction)
4. **Human validation:** You exercise the UI, capture logs/screens if anything misbehaves, and we iterate.

This keeps the loop tight and avoids regressions.

## Useful Reference Points

**Core Architecture**:
- **State manager:** `use_cases/dashboard/core/state_manager.py` (AppStateManager class)
- **State models:** `use_cases/dashboard/core/state_models.py` (immutable dataclasses)
- **Service layer:** `use_cases/dashboard/services/` (TrainingService, ModelService, LabelingService)
- **Commands:** `use_cases/dashboard/core/commands.py` (all state mutations)

**Infrastructure**:
- **Model persistence:** `use_cases/dashboard/core/model_manager.py`
- **Run generation:** `use_cases/dashboard/core/run_generation.py` (experiment outputs)
- **Checkpoint service:** `src/rcmvae/application/services/checkpoint_service.py`

**UI Layer**:
- **Training callbacks:** `use_cases/dashboard/callbacks/training_callbacks.py` (main dashboard)
- **Training hub callbacks:** `use_cases/dashboard/callbacks/training_hub_callbacks.py` (training hub page)
- **Pages:** `use_cases/dashboard/pages/` (layouts and UI components)
- **Utilities:** `use_cases/dashboard/utils/` (visualization, logging)

**Documentation**:
- **Extended roadmap:** [`dashboard_state_plan.md`](dashboard_state_plan.md)
- **Integration plan:** [`../ROADMAP.md`](../ROADMAP.md) (Phase 0-4 vision)
- **Autonomous agent contract:** [`autonomous_agent_spec.md`](autonomous_agent_spec.md)

## Future Checklist Ideas

- **Validation layout cleanup (Completed November 2025):** Added missing component IDs (`home-delete-feedback`, `config-feedback`) to `core/validation.py` to eliminate Dash "nonexistent ID" warnings. All callback outputs now have corresponding placeholders in the validation layout.
- **Metadata-driven training configuration (Completed November 2025):** Refactored Training Hub quick controls to consume `core/config_metadata.py`, ensuring consistent validation, min/max values, labels, and descriptions across all training interfaces. Configuration page (`pages/training.py`) and Training Hub (`pages/training_hub.py`) now share a single source of truth for training parameters.
- **Experiment browser with model-centric organization (Completed November 2025):** Refactored experiment browser to use three-panel layout (Models | Runs | Detail). Left panel shows list of models with run counts and latest timestamps. Middle panel shows training runs filtered by selected model. Right panel shows detailed run information including metrics, figures, and configuration. Tag filtering available as secondary filter. Backend support in `core/experiment_catalog.py` extracts available models. Callbacks handle URL-based navigation with `/experiments?model=<model_id>&run=<run_id>` pattern.
- **Run history in dashboard (Completed November 2025):** Surfaced per-model run history in main dashboard layout via `_build_run_history_section()` in `pages/layouts.py`. The `refresh_run_history` callback in `callbacks/training_callbacks.py` loads recent runs from `core/model_runs.py` and renders compact cards showing run ID, timestamp, epoch range, label version, metrics (train/val loss), and training duration. Displays latest 8 runs with link to full experiment browser. Run records are persisted in `runs.json` and automatically updated when training completes via `CompleteTrainingCommand`.
- Expand dashboard docs folder with targeted tutorials (e.g., adding a training metric).
- Automate end-to-end smoke tests for the dashboard (start training → view history).
- Consider surfacing a UI toggle for `DASHBOARD_LOG_LEVEL` presets to ease debugging.
- Work from the short-horizon priorities in [`dashboard_state_plan.md`](dashboard_state_plan.md) (label provenance and statistics).

Keep this file handy when starting a fresh session so we can pick up momentum immediately.
