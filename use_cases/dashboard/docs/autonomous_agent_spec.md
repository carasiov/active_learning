# Dashboard Autonomous Agent Spec

> **Purpose:** Provide a self-contained roadmap and operating contract for an autonomous coding agent extending the dashboard without human supervision.

## 1. Context Snapshot (November 2025)

- **Target system:** `use_cases/dashboard` – Dash-based UI for orchestrating SSVAE experiments.
- **Backend interface:** `src/rcmvae` (model API, training services, checkpoint manager).
- **Documentation index:**
  - Orientation – `use_cases/dashboard/README.md`
  - Architecture – `use_cases/dashboard/docs/DEVELOPER_GUIDE.md`
  - Implementation recipes – `use_cases/dashboard/docs/AGENT_GUIDE.md`
  - Collaboration notes – `use_cases/dashboard/docs/collaboration_notes.md`
  - Roadmap detail – `use_cases/dashboard/docs/dashboard_state_plan.md`

## 2. Operating Principles

1. **Read before writing.** Confirm understanding of the Developer Guide (architecture) and Agent Guide (recipes) before touching code.
2. **Command-first mutations.** All state changes flow through `core/commands.py`; never mutate `AppState` directly.
3. **Preserve compatibility.** Changes must not break CLI experiments or existing checkpoint loading.
4. **Log + test.** Every task must end with relevant logs captured and dashboard tests passing.
5. **Document deltas.** Update documentation when behavior or workflow changes.

## 3. Baseline Verification Steps

Before submitting any work, perform:

```bash
# Optional: enable verbose dashboard logs during manual checks
export DASHBOARD_LOG_LEVEL=DEBUG

# Core dashboard tests
poetry run pytest tests/test_dashboard_config.py tests/test_dashboard_integration.py
```

If backend behavior is modified (training, checkpoints), also run the focused suite:

```bash
poetry run pytest tests/test_mixture_encoder.py tests/test_mixture_losses.py
```

## 4. Active Roadmap (Prioritized)

### 4.1 Validation Layout Cleanup (High Priority)
- **Goal:** Remove Dash "nonexistent ID" warnings by adding hidden placeholders in `app.validation_layout`.
- **Deliverables:**
  - Updated `app.py` validation layout.
  - Regression tests still passing.
  - Note added to `docs/collaboration_notes.md` under "Future Checklist".
- **Acceptance:** Launching the dashboard produces zero callback ID warnings in the browser console.

### 4.2 Metadata-Driven Training Form (High Priority)
- **Goal:** Render training/config controls from `core/config_metadata.py` and keep backend config symmetry.
- **Deliverables:**
  - Extended metadata schema covering existing fields.
  - `pages/training.py` refactored to consume metadata.
  - Persistence (`state_models.py`, `model_manager.py`) verified.
  - Documentation updates summarising the change.
- **Acceptance:** UI reflects metadata-driven form, no regression in config save/load, tests green.

### 4.3 Experiment Browser Enhancements (Medium Priority)
- **Goal:** Allow filtering by model/run tag and provide quick comparison summaries.
- **Deliverables:**
  - `core/experiment_catalog.py` supports basic filters.
  - `pages/experiments.py` UI exposing filters + detail view.
  - Update collaboration notes with feature summary.
- **Acceptance:** Users can select a model filter; the list updates accordingly, tests green.

### 4.4 Run History in Model Dashboard (Medium Priority)
- **Goal:** Surface recent runs (from `runs.json`) inside the main dashboard.
- **Deliverables:**
  - New component(s) in `pages/layouts.py` for run history.
  - Callback wiring retrieving data via `core/model_runs.py`.
  - Documentation updates describing UX change.
- **Acceptance:** Dashboard shows recent runs after training, navigation remains warning-free.

### 4.5 Label Provenance & Stats (Lower Priority)
- **Goal:** Display label counts/versions and tie runs to label snapshots.
- **Deliverables:**
  - Persistence updates recording label snapshots.
  - UI additions (home + experiment browser) showing counts.
  - Docs updated with provenance workflow guidance.
- **Acceptance:** Label counts visible, run entries show label version, tests updated where needed.

## 5. Execution Checklist per Task

1. Review relevant documentation sections.
2. Create/update tests if behavior changes.
3. Implement code changes following command + metadata patterns.
4. Run required test suites.
5. Capture excerpts of `/tmp/ssvae_dashboard.log` when training flows are touched.
6. Update docs (README, guides, collaboration notes, state plan) as appropriate.
7. Provide a concise summary (files touched, tests run, manual validation) in the final report.

## 6. Reporting Template

When the agent finishes a task, use the following structure:

```
Summary:
- <one-line outcome>

Changes:
- <file>: <short description>
- ...

Tests:
- poetry run pytest ...

Notes:
- <manual checks, log snippets, follow-ups>
```

## 7. Escalation Rules

Stop and request human guidance if any of the following occur:
- Inability to reconcile dashboard state with backend expectations (e.g., optimizer mismatch after changes).
- Need to modify theoretical specs or mathematical logic in `docs/theory/*`.
- Blocking dependency updates or environment changes.

## 8. Ready-to-Start Summary

- All source files follow ASCII format and command-based state mutations.
- Documentation refreshed November 2025.
- Tests to trust: `tests/test_dashboard_config.py`, `tests/test_dashboard_integration.py`.
- Launch command: `poetry run python use_cases/dashboard/app.py` (set `DASHBOARD_LOG_LEVEL=DEBUG` if debugging).

With this spec, the autonomous agent should be able to select a task from the roadmap, execute it end-to-end, and report results with minimal human intervention.
