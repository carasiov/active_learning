# SSVAE Dashboard - Refactoring Tasks

**Status:** Phase 2 in progress (Phase 1 complete)  
**Agent Workflow:** Implement tasks sequentially, test after each, commit when verified.

**Before starting:** Read `DASHBOARD_ARCHITECTURE.md` for context and constraints.

---

## Phase 1: Code Organization (No Functionality Changes)

### Task 1: Extract Utility Functions *(completed)*

**Objective:** Move visualization helper functions to a standalone `utils.py` module.

**Files to create:**
- `use_cases/dashboard/utils.py`

**Functions to extract from `app.py`:**
- `_colorize_user_labels(labels: np.ndarray) -> List[str]`
- `_colorize_numeric(values: np.ndarray) -> List[str]`
- `array_to_base64(arr: np.ndarray) -> str`
- `_build_hover_text(...) -> List[str]`

**Constraints:**
- Do not modify function signatures or behavior
- Keep all imports these functions need (numpy, PIL, matplotlib)
- Add proper docstrings to each function
- Do not modify `app.py` yet (that's Task 5)

**Verification:**
```bash
# File exists and imports work
python -c "from use_cases.dashboard.utils import array_to_base64; print('OK')"

# Functions have correct signatures
python -c "from use_cases.dashboard.utils import _colorize_user_labels, _colorize_numeric, array_to_base64, _build_hover_text; print('OK')"
```

---

### Task 2: Extract State Management *(completed)*

**Objective:** Move state initialization and state-related functions to `state.py`.

**Files to create:**
- `use_cases/dashboard/state.py`

**What to extract from `app.py`:**
- `app_state` dictionary definition
- `state_lock` threading.Lock
- `metrics_queue` Queue definition
- Constants: `CHECKPOINT_PATH`, `LABELS_PATH`, `COOLWARM_CMAP`, `MAX_STATUS_MESSAGES`
- Functions:
  - `initialize_model_and_data()`
  - `_append_status_message(message: str)`
  - `_append_status_message_locked(message: str)`
  - `_update_history_with_epoch(payload: Dict)`
  - `_update_label(sample_idx: int, label_value: int | None)`
  - `_clear_metrics_queue()`

**Constraints:**
- Import `utils` functions where needed (e.g., `_build_hover_text`)
- Keep all docstrings and comments
- Maintain exact behavior - this is pure extraction, no refactoring
- Add module-level docstring explaining this is the state layer

**Verification:**
```bash
# File imports successfully
python -c "from use_cases.dashboard.state import app_state, state_lock, initialize_model_and_data; print('OK')"

# Check key functions exist
python -c "from use_cases.dashboard.state import _update_label, _append_status_message; print('OK')"
```

---

### Task 3: Extract Layout Components *(completed)*

**Objective:** Move UI layout construction to `layouts.py`.

**Files to create:**
- `use_cases/dashboard/layouts.py`

**What to extract from `app.py`:**
- The entire layout construction logic from inside `create_app()`
- Create a function: `def build_dashboard_layout() -> html.Div` that returns the full layout
- Include all the inline layout code (header, stores, controls, scatter plot, sample display, etc.)

**Constraints:**
- Import necessary components: `dash.html`, `dash.dcc`, `dash_bootstrap_components as dbc`, `plotly.graph_objects as go`
- Do not extract callback definitions - only layout
- Keep IDs identical (these are used by callbacks)
- Add docstring explaining this is the UI layout layer

**Key IDs to preserve:**
- `"latent-scatter"`, `"color-mode-radio"`, `"selected-sample-store"`, `"labels-store"`, `"latent-store"`
- `"original-image"`, `"reconstructed-image"`, `"selected-sample-header"`, `"prediction-info"`
- `{"type": "label-button", "label": i}` for i in range(10)
- `"delete-label-button"`, `"start-training-button"`, `"training-status"`, `"poll-interval"`
- All hyperparameter input IDs (sliders, number inputs)

**Verification:**
```bash
# File imports successfully
python -c "from use_cases.dashboard.layouts import build_dashboard_layout; print('OK')"

# Returns proper Dash component
python -c "from use_cases.dashboard.layouts import build_dashboard_layout; layout = build_dashboard_layout(); print(type(layout).__name__)"
# Should print: Div
```

---

### Task 4: Extract Callbacks *(completed)*

**Objective:** Organize callbacks into separate modules by functional area.

**Files to create:**
- `use_cases/dashboard/callbacks/__init__.py`
- `use_cases/dashboard/callbacks/training_callbacks.py`
- `use_cases/dashboard/callbacks/visualization_callbacks.py`
- `use_cases/dashboard/callbacks/labeling_callbacks.py`

**Task 4a: Training Callbacks**

**Move to `training_callbacks.py`:**
- `train_worker(num_epochs: int)` function
- `_configure_trainer_callbacks(trainer, target_epochs)` function
- Callbacks:
  - Start training button callback (decorated with `@app.callback(...)`)
  - Polling interval callback (decorated with `@app.callback(...)`)

**Pattern:**
```python
def register_training_callbacks(app):
    """Register all training-related callbacks."""
    
    @app.callback(...)
    def start_training_callback(...):
        ...
    
    @app.callback(...)
    def poll_training_progress(...):
        ...
```

**Task 4b: Visualization Callbacks**

**Move to `visualization_callbacks.py`:**
- Callbacks:
  - `update_scatter()` - updates the scatter plot
  - `handle_point_selection()` - handles click on scatter plot
  - `sync_color_mode()` - syncs color mode radio

**Task 4c: Labeling Callbacks**

**Move to `labeling_callbacks.py`:**
- Callbacks:
  - `update_sample_display()` - updates image displays and prediction info
  - `handle_label_actions()` - handles label button clicks

**Constraints for all callback files:**
- Import `from use_cases.dashboard.state import app_state, state_lock, metrics_queue, ...` as needed
- Import `from use_cases.dashboard.utils import ...` as needed
- Each module should have a `register_X_callbacks(app)` function
- Keep `@app.callback()` decorators inside the register function
- Do not modify callback logic - pure extraction
- Add module docstrings

**Verification:**
```bash
# All modules import
python -c "from use_cases.dashboard.callbacks import training_callbacks, visualization_callbacks, labeling_callbacks; print('OK')"

# Register functions exist
python -c "from use_cases.dashboard.callbacks.training_callbacks import register_training_callbacks; print('OK')"
python -c "from use_cases.dashboard.callbacks.visualization_callbacks import register_visualization_callbacks; print('OK')"
python -c "from use_cases.dashboard.callbacks.labeling_callbacks import register_labeling_callbacks; print('OK')"
```

---

### Task 5: Refactor app.py to Use Extracted Modules *(completed)*

**Objective:** Simplify `app.py` to be a thin orchestration layer.

**Files to modify:**
- `use_cases/dashboard/app.py`

**Target structure:**
```python
"""Dashboard entry point."""
from dash import Dash
import dash_bootstrap_components as dbc

from state import initialize_model_and_data
from layouts import build_dashboard_layout
from callbacks.training_callbacks import register_training_callbacks
from callbacks.visualization_callbacks import register_visualization_callbacks
from callbacks.labeling_callbacks import register_labeling_callbacks


def create_app() -> Dash:
    """Create and configure the Dash application."""
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )
    app.layout = build_dashboard_layout()
    
    register_training_callbacks(app)
    register_visualization_callbacks(app)
    register_labeling_callbacks(app)
    
    return app


app = create_app()


if __name__ == "__main__":
    initialize_model_and_data()
    app.run_server(debug=False, host="0.0.0.0", port=8050)
```

**Constraints:**
- Remove all extracted code
- Keep imports minimal
- Keep the `if __name__ == "__main__"` block
- Update any relative imports to work from the new structure

**Verification:**
```bash
# Dashboard starts without errors
python use_cases/dashboard/app.py &
sleep 5
curl http://localhost:8050 > /dev/null 2>&1 && echo "Dashboard is running" || echo "FAILED"
pkill -f "python use_cases/dashboard/app.py"

# Test a complete workflow
# 1. Open http://localhost:8050
# 2. Click a point in the scatter plot
# 3. Click a label button (0-9)
# 4. Verify the CSV updated
# 5. Change color mode and verify scatter updates
```

---

## Phase 2: Performance Optimizations

### Task 6: Optimize Scatter Plot Callback

**Objective:** Reduce unnecessary figure rebuilds in the scatter plot callback.

**Files to modify:**
- `use_cases/dashboard/callbacks/visualization_callbacks.py`

**Implementation approach:**
- Cache the base figure structure
- On color mode changes, use Plotly's `Patch()` to update only marker colors
- Only rebuild the full figure when latent coordinates actually change (latent_version change)
- Add a cache dict keyed by `(latent_version, color_mode)` to avoid recomputing colors

**Pattern:**
```python
# Add module-level cache
_figure_cache = {}

@app.callback(...)
def update_scatter(...):
    cache_key = (latent_version, color_mode)
    if cache_key in _figure_cache:
        return _figure_cache[cache_key]
    
    # Build figure...
    _figure_cache[cache_key] = figure
    return figure
```

**Constraints:**
- Do not change the visual output - should look identical
- Keep the selected point overlay working
- Clear cache when latent_version changes

**Verification:**
```bash
# Start dashboard and test
python use_cases/dashboard/app.py &
sleep 5

# Manual test:
# 1. Open dashboard
# 2. Switch between color modes rapidly - should feel instant
# 3. Start training, wait for completion - scatter should update
# 4. Color modes should still work after training

pkill -f "python use_cases/dashboard/app.py"
```

---

### Task 7: Optimize Polling Callback with dash.no_update

**Objective:** Make polling callback only update outputs that actually changed.

**Files to modify:**
- `use_cases/dashboard/callbacks/training_callbacks.py`

**Implementation:**
- Import `from dash import no_update`
- Track what actually changed during message processing
- Return `no_update` for outputs that haven't changed
- Only update status display when messages were processed
- Only update latent-store when version actually changed

**Current signature returns 8 outputs:**
```python
return (
    status_children,          # Always update if messages processed
    controls_disabled,        # Only update if training state changed
    controls_disabled,        # (repeated for multiple buttons)
    controls_disabled,
    controls_disabled,
    controls_disabled,
    latent_store_out,        # Only update if latent_version changed
    interval_disabled,       # Only update if idle state changed
)
```

**Constraints:**
- Must handle the initial state correctly (don't use no_update on first call)
- Keep the same logic, just optimize returns
- Test that training still works end-to-end

**Verification:**
```bash
# Start dashboard, start training, verify:
# 1. Status updates appear during training
# 2. Controls disable/enable correctly
# 3. Scatter plot updates on completion
# 4. No errors in console
```

---

### Task 8: Implement Smart Interval Disabling

**Objective:** Disable polling interval when truly idle to reduce CPU usage.

**Files to modify:**
- `use_cases/dashboard/callbacks/training_callbacks.py`

**Implementation:**
- Current: `interval_disabled = not active and not processed_messages and metrics_queue.empty()`
- This already exists! Just verify it's working correctly.
- If interval is currently always enabled, fix the logic to properly disable when:
  - Not training (`app_state["training"]["active"] == False`)
  - Queue is empty (`metrics_queue.empty() == True`)
  - No messages were just processed

**Constraints:**
- Must re-enable immediately when training starts
- Must process all queued messages before disabling

**Verification:**
```bash
# Check that polling stops when idle
python use_cases/dashboard/app.py &
sleep 5

# Manual verification:
# 1. Open browser console, watch network requests
# 2. With no training active, should not see poll requests
# 3. Start training, should see poll requests every ~1s
# 4. Training completes, poll requests should stop

pkill -f "python use_cases/dashboard/app.py"
```

---

## Phase 3: New Features

### Task 9: Add Loss Curve Visualization

**Objective:** Add a live-updating line chart showing training metrics.

**Files to modify:**
- `use_cases/dashboard/layouts.py` - add the graph component
- `use_cases/dashboard/callbacks/visualization_callbacks.py` - add callback

**Layout addition:**
```python
# Add inside build_dashboard_layout(), after training controls:
html.Div([
    html.H5("Training Progress", className="mt-4"),
    dcc.Graph(id="loss-curves", style={"height": "400px"}),
], className="mt-3"),
```

**Callback to add:**
```python
@app.callback(
    Output("loss-curves", "figure"),
    Input("latent-store", "data"),
)
def update_loss_curves(_latent_store: dict):
    """Update loss curves when training completes."""
    with state_lock:
        history = app_state["history"]
        epochs = history["epochs"]
    
    if not epochs:
        return go.Figure()  # Empty figure if no training yet
    
    fig = go.Figure()
    
    # Add traces for losses
    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], 
                             name="Train Loss", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], 
                             name="Val Loss", mode="lines+markers"))
    # Add component losses...
    
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig
```

**Constraints:**
- Show at least: train_loss, val_loss, train_reconstruction_loss, val_reconstruction_loss
- Use distinct colors for train vs val
- Keep the plot clean and readable
- Update only when latent-store changes (after training)

**Verification:**
```bash
# Start dashboard, train for a few epochs, verify:
# 1. Loss curves appear and show data
# 2. Curves update after training completes
# 3. Multiple training sessions accumulate history
```

---

### Task 10: Add Dataset Statistics Display

**Objective:** Show labeled sample counts and label distribution.

**Files to modify:**
- `use_cases/dashboard/layouts.py` - add statistics panel
- `use_cases/dashboard/callbacks/labeling_callbacks.py` - add callback

**Layout addition:**
```python
# Add inside build_dashboard_layout(), in sidebar or near controls:
html.Div([
    html.H6("Dataset Statistics"),
    html.Div(id="dataset-stats", className="small"),
], className="mb-3"),
```

**Callback to add:**
```python
@app.callback(
    Output("dataset-stats", "children"),
    Input("labels-store", "data"),
)
def update_dataset_stats(_labels_store: dict):
    """Display labeled sample counts and distribution."""
    with state_lock:
        labels = np.array(app_state["data"]["labels"])
    
    total_samples = len(labels)
    labeled_mask = ~np.isnan(labels)
    labeled_count = np.sum(labeled_mask)
    labeled_pct = (labeled_count / total_samples * 100) if total_samples > 0 else 0
    
    # Count per class
    label_counts = {}
    if labeled_count > 0:
        labeled_values = labels[labeled_mask].astype(int)
        for digit in range(10):
            label_counts[digit] = np.sum(labeled_values == digit)
    
    # Build display
    lines = [
        html.Div(f"Total samples: {total_samples}"),
        html.Div(f"Labeled: {labeled_count} ({labeled_pct:.1f}%)"),
        html.Div(f"Unlabeled: {total_samples - labeled_count}"),
    ]
    
    if label_counts:
        lines.append(html.Hr(className="my-2"))
        lines.append(html.Div("Label distribution:", className="fw-bold"))
        for digit, count in sorted(label_counts.items()):
            lines.append(html.Div(f"  {digit}: {count}"))
    
    return lines
```

**Constraints:**
- Update immediately when labels change
- Show: total samples, labeled count, unlabeled count, per-class distribution
- Keep formatting clean and compact

**Verification:**
```bash
# Start dashboard, verify:
# 1. Stats show current label counts
# 2. Stats update immediately when you label a sample
# 3. Distribution shows counts for each digit
```

---

## Phase 4: Final Polish (Optional)

### Task 11: Add Error Boundary for Training

**Objective:** Improve error handling when training fails.

**Files to modify:**
- `use_cases/dashboard/callbacks/training_callbacks.py`

**Implementation:**
- Wrap training start callback in try/except
- If training fails to start, show error in status
- If background thread fails, catch in `train_worker` and push error message

**Verification:**
```bash
# Test error handling:
# 1. Try to start training with invalid config (if possible)
# 2. Verify error message appears in status
# 3. Verify dashboard remains responsive
```

---

### Task 12: Add Keyboard Shortcuts for Labeling

**Objective:** Allow pressing 0-9 keys to label selected sample.

**Files to modify:**
- `use_cases/dashboard/layouts.py` - add clientside callback
- `use_cases/dashboard/callbacks/labeling_callbacks.py` - wire up handling

**Implementation:**
- Use Dash's `dcc.Input` with keypress detection or clientside callback
- When user presses 0-9, trigger same logic as clicking label button
- Only when a sample is selected

**Verification:**
```bash
# Start dashboard, verify:
# 1. Click a point to select it
# 2. Press a digit key (0-9)
# 3. Label is applied (same as clicking button)
```

---

## Completion Checklist

After all tasks are complete, verify:

- [ ] Dashboard starts without errors
- [ ] All 60k points render in scatter plot
- [ ] Clicking points updates image display
- [ ] Labeling works (buttons and keyboard if implemented)
- [ ] Labels persist to CSV immediately
- [ ] Color modes all work (user labels, pred class, true class, certainty)
- [ ] Training starts in background
- [ ] Status updates appear during training
- [ ] Loss curves update after training
- [ ] Latent space refreshes after training
- [ ] Dataset stats update when labels change
- [ ] Multiple training sessions work (optimizer state preserved)
- [ ] CLI scripts still work (`train.py`, `infer.py`)
- [ ] Code is organized into logical modules
- [ ] No performance regressions (clicking feels instant)

---

## Notes for Agent

- **Work sequentially**: Complete Task N before starting Task N+1
- **Preserve behavior**: Unless explicitly changing functionality, keep exact behavior
- **Read architecture doc**: Refer to `DASHBOARD_ARCHITECTURE.md` for context
- **Ask if unclear**: If a task is ambiguous, ask for clarification
- **Test thoroughly**: User will test after each task before committing
