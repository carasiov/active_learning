# Dashboard Agent Guide

> **Purpose:** Provide quick recipes and guardrails for AI assistants (and humans) when extending the dashboard.

## Working Principles

1. **Read before writing:** Consult `use_cases/dashboard/README.md` for high-level context and `docs/collaboration_notes.md` for the current snapshot.
2. **Use commands for mutations:** Never mutate `AppState` directly—always add/extend command classes in `core/commands.py`.
3. **Keep UI/data coupled via metadata:** Prefer schema-driven configuration (`core/config_metadata.py`) to hard-coded forms.
4. **Log everything:** Background workers and callbacks should log meaningful context; use `DASHBOARD_LOG_LEVEL=DEBUG` during development.
5. **Respect tests:** Update or add tests under `tests/test_dashboard_*` and run them before handing work back.

## Common Tasks

### 1. Add a Command
- Extend `core/commands.py` with a new `@dataclass` command.
- Implement `validate()` (return error message string or `None`).
- Implement `execute()` returning `(new_state, message)`.
- Wire the command into `core/dispatcher` callers (callbacks, pages).
- Update tests (`tests/test_dashboard_config.py` or similar).

### 2. Add UI Controls
- Update the relevant layout (`pages/home.py`, `pages/layouts.py`, `pages/training.py`, etc.).
- Add callback handlers or extend existing ones under `callbacks/`.
- Ensure `app.validation_layout` includes any new component IDs.

### 3. Extend Training Workflow
- Update `callbacks/training_callbacks.py` for additional metrics or behavior.
- If new artifacts are produced, integrate them into `core/run_generation.py` and `core/model_runs.py`.
- Surface new data on the UI via `pages/training.py` or experiment browser.

### 4. Add Configuration Options
- Update `core/config_metadata.py` with new field metadata.
- Ensure `core/state_models.py` and `core/model_manager.py` persist the new settings.
- Modify `pages/training.py` to render the controls using metadata.

### 5. Update Experiment Browser
- Modify `pages/experiments.py` for layout changes.
- Use `core/experiment_catalog.py` for data access.
- Keep large artifacts optional; reference the run manifest (`core/model_runs.py`).

## Safety Checklist

- ✅ Run `poetry run pytest tests/test_dashboard_config.py tests/test_dashboard_integration.py`
- ✅ Capture `/tmp/ssvae_dashboard.log` after new training flows.
- ✅ Confirm commands fail gracefully with meaningful messages.
- ✅ Ensure navigation across pages does not trigger callback errors.

## Quick Links

- Collaboration notes: `docs/collaboration_notes.md`
- Developer guide: `docs/DEVELOPER_GUIDE.md`
- Architecture (global): `../../../docs/development/architecture.md`

## Future Improvements

- Add more task-specific recipes (e.g., “adding a new metric card”).
- Provide pattern examples for multi-model management features.
- Document known pitfalls (Dash callback gotchas, long-running threads).
