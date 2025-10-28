# Project State Summary

**Date:** 2025-10-28  
**Branch:** multi-model-system  
**Status:** Core working, UI issues remain

---

## Documentation Structure

### Root Level
- **DASHBOARD_SYSTEM_SPEC.md** (MAIN) - Complete system spec, issues, dev workflow
- **PROJECT_STATE.md** (this file) - Quick overview

### Dashboard Docs (`use_cases/dashboard/`)
- **README.md** - Quick start, features, routes
- **DEVELOPER_GUIDE.md** - Architecture internals (state, commands, training)
- **AGENT_GUIDE.md** - Extension patterns

### Tests
- **tests/run_dashboard_tests.py** - Standalone test runner (8 tests, all passing)
- **tests/test_dashboard_integration.py** - Pytest version (requires pytest install)

---

## Quick Commands

```bash
# Start
cd /workspaces/active_learning
poetry run python use_cases/dashboard/app.py

# Logs
tail -f /tmp/ssvae_dashboard.log

# Tests
poetry run python tests/run_dashboard_tests.py

# Check state
ls artifacts/models/
```

---

## Current Issues (Priority Order)

### 1. Delete Button Race Condition (CRITICAL)
- **Symptom:** Delete triggers both delete_model and open_model callbacks
- **Evidence:** Logs show both fire simultaneously
- **Fix:** Add confirmation modal (see DASHBOARD_SYSTEM_SPEC.md § Next Session #1)

### 2. Training Not Starting
- **Status:** Not yet debugged with logging
- **Fix:** Enable DEBUG logs, test, check logs

### 3. Visualization Empty on First Load
- **Status:** Suspected race condition
- **Fix:** Add loading state or ensure MNIST loaded first

---

## What's Working

✅ Multi-model architecture  
✅ Model creation with name sanitization  
✅ Model loading and switching  
✅ Isolated state per model  
✅ Structured logging system  
✅ Integration tests (8/8 passing)  
✅ Command pattern for state updates  
✅ Thread-safe state management  

---

## Key Files Modified

### Directory Reorganization (2025-10-28)
**Structure:** Organized into `core/`, `pages/`, `callbacks/`, `utils/` directories

```
use_cases/dashboard/
├── app.py               # Entry point
├── core/                # Infrastructure
│   ├── state.py         # State management
│   ├── commands.py      # State mutations
│   ├── state_models.py  # Data classes
│   ├── model_manager.py # File I/O
│   └── logging_config.py # Logging
├── pages/               # UI layouts
├── callbacks/           # Event handlers
└── utils/               # Helpers
```

### New Infrastructure
- `core/logging_config.py` - Structured logging
- `utils/callback_utils.py` - Callback decorators
- `core/model_manager.py` - File I/O
- `pages/home.py` - Model list UI
- `callbacks/home_callbacks.py` - Home page logic

### Core Updates
- `app.py` - Logging init, routing, query params
- `core/state.py` - RLock, multi-model state
- `core/state_models.py` - ModelMetadata, ModelState
- `core/commands.py` - Create/Load/Delete commands + logging
- All callbacks - Use active_model pattern, updated imports

---

## For Next Session

**Provide:**
1. `DASHBOARD_SYSTEM_SPEC.md`
2. Codebase
3. Prompt: "Read DASHBOARD_SYSTEM_SPEC.md. Fix delete button race condition (§ Next Session #1)."

**Don't need:**
- This file (PROJECT_STATE.md) - info is in spec
- Implementation history docs (removed)
- Detailed guides (keep DEVELOPER_GUIDE.md)

---

## Architecture at a Glance

```
AppState
├── models: Dict[id → ModelMetadata]  # Registry
├── active_model: ActiveModelState    # Loaded model
│   ├── model, trainer, config
│   ├── data, training, history
│   └── model_id, metadata
└── cache: Dict                       # Shared (MNIST)

Commands → Dispatcher (with lock) → New State
Callbacks → @logged_callback → Logs
```

**Logging:** `/tmp/ssvae_dashboard.log` (DEBUG level in file, INFO in console)

**Tests:** 8 integration tests covering creation, deletion, loading, consistency

**Routes:** `/` (home), `/model/{id}` (dashboard), `/model/{id}/training-hub`, `/model/{id}/configure-training`
