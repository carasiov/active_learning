# SSVAE Dashboard System Specification

**Last Updated:** 2025-10-28  
**Status:** Partial Implementation - Core Issues Remain  
**Branch:** multi-model-system

---

## System Overview

Multi-model SSVAE dashboard allowing users to create, manage, train, and visualize multiple semi-supervised VAE models independently. Each model has isolated state (checkpoints, labels, training history).

### Architecture
```
use_cases/dashboard/
├── app.py                          # Main Dash app, routing
├── core/                           # Core infrastructure
│   ├── state.py                    # Global state management
│   ├── state_models.py             # Data classes (AppState, ModelMetadata, etc.)
│   ├── commands.py                 # Command pattern for state mutations
│   ├── model_manager.py            # File I/O for models
│   └── logging_config.py           # Structured logging system
├── pages/                          # Page layouts
│   ├── home.py                     # Home page with model list
│   ├── training_hub.py             # Training controls
│   ├── training.py                 # Training config page
│   └── layouts.py                  # Main dashboard layout
├── callbacks/                      # All Dash callbacks
│   ├── home_callbacks.py
│   ├── training_callbacks.py
│   └── ...
├── utils/                          # Utility functions
│   ├── visualization.py            # Plotting helpers
│   ├── callback_utils.py           # Callback decorators
│   └── training_callback.py        # Training metrics callback
├── assets/                         # Static files (CSS, images)
└── docs/                           # Documentation
```

---

## Key Design Decisions

### 1. Model Isolation
- Each model stored in: `artifacts/models/{model_id}/`
  - `metadata.json` - Model metadata
  - `checkpoint.ckpt` - Model weights
  - `labels.csv` - Labeled samples
  - `history.json` - Training history
- Model directory name = sanitized user-provided name (e.g., "My Model" → `my_model`)

### 2. State Management
- **Global State:** `dashboard_state.app_state` (thread-locked)
  - `models`: Dict[model_id, ModelMetadata] - All available models
  - `active_model`: Optional[ActiveModelState] - Currently loaded model
  - `cache`: Dict - Shared data (MNIST dataset)
  
- **Command Pattern:** All state mutations via `Command.execute(state) → (new_state, message)`
  - `CreateModelCommand` - Create new model
  - `LoadModelCommand` - Load model as active
  - `DeleteModelCommand` - Delete model
  - `StartTrainingCommand` - Initiate training
  - `UpdateLabelsCommand` - Update labels

### 3. Routing
- **Home:** `/` - Model list, create/delete models
- **Model Dashboard:** `/model/{id}` - Active model view
- **Training Hub:** `/model/{id}/training-hub` - Training controls
- **Config:** `/model/{id}/configure-training` - Hyperparameters

### 4. Logging
- Structured logging to `/tmp/ssvae_dashboard.log`
- Decorator `@logged_callback` logs all callback invocations
- Log levels: DEBUG (file), INFO (console)
- View logs: `tail -f /tmp/ssvae_dashboard.log`

---

## Current Status: Known Issues

### ❌ CRITICAL: Delete Button Not Working
**Symptom:** Clicking "Delete" on model does nothing or triggers "Open" instead  
**Root Cause:** Both `delete_model` and `open_model` callbacks fire simultaneously when delete is clicked  
**Evidence:** Logs show both callbacks triggered:
```
[20:51:04] [INFO] delete_model SUCCESS | result=/?refresh=1
[20:51:04] [INFO] open_model SUCCESS | result=/model/test123
```
**Attempted Fixes:**
- Added `?refresh=1` to force page reload - didn't fix simultaneous firing
- Added CSS `pointer-events: auto` - insufficient for Dash callbacks
- Need: Event propagation stop or separate delete button from card

**Next Steps:**
- Move delete button outside button group OR
- Add confirmation modal that blocks other clicks OR
- Use different callback output (not url.pathname)

### ⚠️ Training Not Starting
**Symptom:** Click "Start Training", modal appears, click confirm, nothing happens  
**Status:** Not yet debugged with new logging system  
**Check:** `grep "training" /tmp/ssvae_dashboard.log` after attempting training

### ⚠️ Data Visualization Issues
**Symptom:** Scatter plot empty on first load, appears after refresh  
**Status:** Suspected race condition with MNIST loading or prediction computation

---

## Testing Infrastructure

### Integration Tests
**Location:** `tests/run_dashboard_tests.py` (standalone, no pytest needed)

**Run Tests:**
```bash
cd /workspaces/active_learning
poetry run python tests/run_dashboard_tests.py
```

**Coverage:** 8 tests, 100% passing
- Model creation (simple names, sanitization, duplicates)
- Model deletion (specific model, first in list, active model protection)
- Model loading
- State/filesystem consistency

**Add New Test:**
```python
def test_your_feature(temp_models_dir):
    """Test description."""
    dashboard_state.initialize_app_state()
    # test code
    assert something == expected

# Register in main()
runner.run_test(test_your_feature, "Description")
```

---

## Development Workflow

### Start Dashboard
```bash
pkill -9 -f "python.*dashboard" 2>/dev/null
cd /workspaces/active_learning
poetry run python ./use_cases/dashboard/app.py > /tmp/dash.log 2>&1 &
```

### View Logs (Real-time)
```bash
tail -f /tmp/ssvae_dashboard.log
```

### Debug Specific Issue
```bash
# After performing action in UI
grep "callback_name" /tmp/ssvae_dashboard.log | tail -20

# Example: After clicking delete
grep "delete_model\|open_model" /tmp/ssvae_dashboard.log | tail -10
```

### Run Tests
```bash
poetry run python tests/run_dashboard_tests.py
```

### Check State vs Filesystem
```bash
# List models on disk
ls -la artifacts/models/

# Check state
poetry run python -c "
from use_cases.dashboard.core import state as dashboard_state
dashboard_state.initialize_app_state()
with dashboard_state.state_lock:
    print(list(dashboard_state.app_state.models.keys()))
"
```

---

## Code Patterns

### Adding New Callback
```python
from use_cases.dashboard.utils.callback_utils import logged_callback

@app.callback(
    Output(...),
    Input(...),
    prevent_initial_call=True,
)
@logged_callback("callback_name")
def my_callback(...):
    # Callback logic
    pass
```

### Adding New Command
```python
@dataclass
class MyCommand(Command):
    param: str
    
    def validate(self, state: AppState) -> Optional[str]:
        # Return error message or None
        if invalid:
            return "Error message"
        return None
    
    def execute(self, state: AppState) -> Tuple[AppState, str]:
        logger.info(f"Executing MyCommand: param={self.param}")
        # Mutate state
        new_state = ...
        return new_state, "Success message"
```

### Reading Logs Programmatically
```python
from use_cases.dashboard.core.logging_config import get_logger

logger = get_logger('component_name')  # 'callbacks', 'commands', 'state', etc.

logger.debug("Detailed debug info")
logger.info("Important event")
logger.warning("Warning condition")
logger.error("Error occurred", exc_info=True)  # Includes traceback
```

---

## Next Session Priorities

### 1. Fix Delete Button (CRITICAL)
**Recommended Approach:** Add confirmation modal that blocks UI
```python
# In home_callbacks.py
@app.callback(
    Output("delete-confirm-modal", "is_open"),
    Output("delete-confirm-store", "data"),
    Input({"type": "home-delete-model", "model_id": ALL}, "n_clicks"),
    ...
)
def show_delete_confirmation(...):
    # Show modal, store model_id
    return True, {"model_id": model_id}

@app.callback(
    Output("url", "pathname", allow_duplicate=True),
    Input("confirm-delete-button", "n_clicks"),
    State("delete-confirm-store", "data"),
    ...
)
def confirm_delete(...):
    # Actually delete
    command = DeleteModelCommand(model_id=stored_model_id)
    ...
```

This decouples delete from open callback and adds user confirmation.

### 2. Debug Training Issue
Enable debug logging:
```python
# In app.py line 23
DashboardLogger.setup(console_level=logging.DEBUG)
```

Then test training and check:
```bash
grep "training\|StartTraining" /tmp/ssvae_dashboard.log
```

### 3. Fix Visualization Race Condition
Add loading state to scatter plot callback or ensure MNIST loads before rendering.

---

## Files Modified in This Session

### Directory Reorganization (2025-10-28)
- **Created `core/`**: Moved state management, commands, model I/O, logging
- **Created `pages/`**: Consolidated all page layouts
- **Created `utils/`**: Grouped visualization and callback utilities
- **Result**: Cleaner abstraction boundaries, 13 files → 3 organized directories + entry point

### New Files Created
- `use_cases/dashboard/core/logging_config.py` - Structured logging
- `use_cases/dashboard/utils/callback_utils.py` - Callback decorators
- `tests/run_dashboard_tests.py` - Integration tests (8/8 passing)
- `tests/test_dashboard_integration.py` - Pytest tests (unused, requires pytest)

### Modified Files
- `use_cases/dashboard/app.py` - Added logging initialization, updated imports
- `use_cases/dashboard/core/commands.py` - Added logging to commands
- `use_cases/dashboard/callbacks/home_callbacks.py` - Added @logged_callback, attempted delete fix
- `use_cases/dashboard/pages/home.py` - Show model_id, className for delete button
- `use_cases/dashboard/core/state_models.py` - Field name mapping for old metadata

### No Longer Needed (can delete)
- Any `test_*.py` files in root
- Any `*_debug.py` files
- `/tmp/dash_*.log` (except ssvae_dashboard.log)

---

## Quick Reference Commands

```bash
# Start dashboard
cd /workspaces/active_learning && poetry run python use_cases/dashboard/app.py &

# View logs
tail -f /tmp/ssvae_dashboard.log

# Run tests
poetry run python tests/run_dashboard_tests.py

# Kill dashboard
pkill -9 -f "python.*dashboard"

# Check what's on disk vs state
ls artifacts/models/ && poetry run python -c "from use_cases.dashboard import state; state.initialize_app_state(); print(list(state.app_state.models.keys()))"
```

---

## Key Insights from This Session

1. **Logging is essential** - Print statements weren't enough, structured logging caught the simultaneous callback issue
2. **Tests prevent regressions** - All 8 tests pass, proving delete logic works (UI issue, not logic)
3. **Dash callback conflicts** - Multiple callbacks with same Output can race
4. **State refresh issues** - Returning same pathname doesn't trigger re-render
5. **Model naming** - Directory names should match display names (now sanitized)

---

## For Next Chat: Provide This Prompt

```
I'm working on a multi-model SSVAE dashboard. Read DASHBOARD_SYSTEM_SPEC.md for full context.

CRITICAL ISSUE: Delete button triggers both delete_model and open_model callbacks simultaneously. 
Evidence in logs:
[20:51:04] [INFO] delete_model SUCCESS | result=/?refresh=1
[20:51:04] [INFO] open_model SUCCESS | result=/model/test123

Both fire when delete is clicked, causing open to override delete navigation.

Recommended fix: Add confirmation modal to decouple callbacks.
See "Next Session Priorities #1" in spec doc.

Also need to debug: Training not starting, visualization empty on first load.

System is functional except these UI issues. Tests pass. Logs working.
```
