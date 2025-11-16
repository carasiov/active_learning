# Dashboard Collaboration Notes

> **Purpose:** Snapshot of the current dashboard architecture, debugging aids, and working agreements so future sessions can resume quickly.

## System Snapshot (November 2025)

The fuller history, open issues, and roadmap live in [`dashboard_state_plan.md`](dashboard_state_plan.md). This section distills the highlights most relevant when jumping back into day-to-day work:

- **Entry point:** `use_cases/dashboard/app.py`
  - Configures logging via `DASHBOARD_LOG_LEVEL` (default INFO) and boots Dash routes.
  - Global state resides in `use_cases.dashboard.core.state` (`app_state`, `metrics_queue`, `CommandDispatcher`).
- **State mutations:** Commands in `use_cases/dashboard/core/commands.py` remain the sole way to mutate state.
  - `LoadModelCommand` refuses model switches during active training and clears the metrics queue on activation.
  - `StartTrainingCommand` validates hyperparameters before dispatching the background worker; `CompleteTrainingCommand` threads through mixture artefacts.
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

- **Global state:** `use_cases/dashboard/core/state.py`
- **Training status polling:** `use_cases/dashboard/callbacks/training_callbacks.py::poll_training_status`
- **Model registry helpers:** `use_cases/dashboard/core/model_manager.py`
- **Experiment utilities:** `use_cases/dashboard/pages/*`, `use_cases/dashboard/utils/`
- **Checkpoint service:** `src/rcmvae/application/services/checkpoint_service.py`
- **Extended roadmap/context:** [`dashboard_state_plan.md`](dashboard_state_plan.md)
- **Autonomous agent contract:** [`autonomous_agent_spec.md`](autonomous_agent_spec.md)

## Future Checklist Ideas

- **Validation layout cleanup (Completed November 2025):** Added missing component IDs (`home-delete-feedback`, `config-feedback`) to `core/validation.py` to eliminate Dash "nonexistent ID" warnings. All callback outputs now have corresponding placeholders in the validation layout.
- Expand dashboard docs folder with targeted tutorials (e.g., adding a training metric).
- Automate end-to-end smoke tests for the dashboard (start training → view history).
- Consider surfacing a UI toggle for `DASHBOARD_LOG_LEVEL` presets to ease debugging.
- Work from the short-horizon priorities in [`dashboard_state_plan.md`](dashboard_state_plan.md) (config upgrade, experiment compare tools, run history surfacing, label provenance).

Keep this file handy when starting a fresh session so we can pick up momentum immediately.
