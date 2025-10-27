# Dashboard Developer Guide

This guide explains the internal architecture of the SSVAE Active Learning Dashboard. Read this if you need to understand how the system works, debug issues, or modify core functionality.

**For usage:** See [README.md](README.md)  
**For extending:** See [AGENT_GUIDE.md](AGENT_GUIDE.md)

---

## State Management

### Immutable Architecture

All application state is stored as frozen dataclasses in `state_models.py`:

```python
@dataclass(frozen=True)
class AppState:
    model: SSVAE                # Model instance
    trainer: InteractiveTrainer # Training state
    config: SSVAEConfig         # Hyperparameters
    data: DataState             # Dataset, predictions, labels
    training: TrainingStatus    # Training state machine
    ui: UIState                 # Selected sample, color mode
    cache: Dict                 # Performance caching
    history: TrainingHistory    # Loss curves data
```

**Why immutable?**
- Prevents accidental state corruption
- Makes state changes explicit and auditable
- Eliminates subtle race conditions
- Enables atomic updates under lock

### Thread Safety

State is protected by two locks:

```python
state_lock = threading.Lock()      # Protects app_state reads/writes
_init_lock = threading.Lock()      # Ensures single initialization
```

**Access pattern:**
```python
from use_cases.dashboard import state as dashboard_state

# READ
with dashboard_state.state_lock:
    labels = dashboard_state.app_state.data.labels
    config = dashboard_state.app_state.config

# UPDATE (via command - see below)
command = LabelSampleCommand(sample_idx=42, label=7)
dashboard_state.dispatcher.execute(command)
```

**Critical:** Always import the module, not the variable:
```python
# WRONG - gets stale reference
from use_cases.dashboard.state import app_state

# CORRECT - always gets current value  
from use_cases.dashboard import state as dashboard_state
```

### State Updates

Never mutate state directly. Use commands:

```python
# WRONG - raises FrozenInstanceError
with dashboard_state.state_lock:
    dashboard_state.app_state.data.labels[0] = 5

# CORRECT - use command
command = LabelSampleCommand(sample_idx=0, label=5)
success, message = dashboard_state.dispatcher.execute(command)
```

---

## Command Pattern

All state-modifying operations are encapsulated as commands:

```python
@dataclass
class LabelSampleCommand(Command):
    sample_idx: int
    label: Optional[int]
    
    def validate(self, state: AppState) -> Optional[str]:
        # Check if operation is valid
        if self.sample_idx < 0:
            return "Invalid sample index"
        return None  # Valid
    
    def execute(self, state: AppState) -> Tuple[AppState, str]:
        # Perform state transition
        new_labels = state.data.labels.copy()
        new_labels[self.sample_idx] = self.label
        new_state = state.with_label_update(new_labels, ...)
        return new_state, "Sample labeled"
```

**Benefits:**
- Validation centralized in command
- State updates atomic (execute under lock)
- Audit trail (command history)
- Testable in isolation

**Existing commands:**
- `LabelSampleCommand` - Assign/remove label
- `StartTrainingCommand` - Queue training run
- `CompleteTrainingCommand` - Update predictions after training
- `StopTrainingCommand` - Request training halt
- `SelectSampleCommand` - Change selected sample
- `ChangeColorModeCommand` - Update scatter color mode

---

## Training System

### Background Worker

Training runs in a daemon thread to avoid blocking the UI:

```python
def train_worker(num_epochs: int) -> None:
    # Get data snapshot
    with dashboard_state.state_lock:
        x_train = np.array(dashboard_state.app_state.data.x_train)
        labels = np.array(dashboard_state.app_state.data.labels)
    
    # Train (outside lock - takes minutes)
    history = trainer.train_epochs(num_epochs, x_train, labels)
    
    # Update state (inside lock)
    latent, recon, pred_classes, pred_certainty = model.predict(x_train)
    command = CompleteTrainingCommand(latent, recon, ...)
    dashboard_state.dispatcher.execute(command)
```

**Key points:**
- Copies data before training (lock held briefly)
- Training happens outside lock (minutes)
- Results committed atomically via command

### Real-Time Updates

Training progress flows through a queue:

```python
metrics_queue = Queue()  # Thread-safe

# Training callback puts messages
def on_epoch_end(epoch, metrics, ...):
    metrics_queue.put({
        "type": "epoch_complete",
        "epoch": epoch + 1,
        "train_loss": float(metrics["train"]["loss"]),
        ...
    })

# UI polls and processes messages (every 2 seconds)
def poll_training_status(...):
    while True:
        try:
            message = metrics_queue.get_nowait()
            if message["type"] == "epoch_complete":
                _update_history_with_epoch(message)
                _append_status_message(f"Epoch {epoch}/{target}")
        except Empty:
            break
```

**Why queue + polling?**
- Training thread can't directly update Dash UI
- Queue is thread-safe, non-blocking
- Polling interval (2s) balances responsiveness and overhead
- UI can batch multiple messages per poll

### Stop Training

Stopping is cooperative, not forceful:

1. User clicks "Stop Training"
2. `StopTrainingCommand` sets `stop_requested=True` flag
3. Dashboard callback checks flag via `TrainingStoppedException`
4. Training completes current epoch, then halts gracefully
5. Partial results saved and displayed

**Why not kill the thread?**
- Thread termination is dangerous (corrupts state)
- Graceful stop ensures valid checkpoint saved
- Current epoch completes (only ~20 seconds)

---

## Page Architecture

### Main Page (`/`)

**Purpose:** Interactive labeling and quick training

**Layout:**
- Left: Minimal training controls (epochs, train button)
- Center: Latent space scatter plot (60k points)
- Right: Sample viewer and labeling controls

**Features:**
- Click points to select samples
- Label via keyboard (0-9) or buttons
- Quick training (modal confirmation)
- Color modes (user labels, predictions, certainty)

### Training Hub (`/training-hub`)

**Purpose:** Detailed training monitoring and configuration

**Layout:**
- Top: Status hero bar (running/idle/complete)
- Left (40%): Training configuration
  - Epochs input
  - Essential parameters (LR, recon weight, KL weight)
- Right (60%): Real-time monitoring
  - Loss curves (4 series)
  - Dataset statistics
  - Metrics table
- Bottom: Training terminal (dark theme)

**Features:**
- Real-time loss curves (update every 2s)
- Stop training mid-run
- Terminal with full training log
- Clear/download terminal output
- Loss smoothing toggle

### Configuration Page (`/configure-training`)

**Purpose:** Advanced hyperparameter tuning

**Features:**
- 17+ training parameters
- Architecture selection (encoder/decoder type)
- Optimizer settings
- Early stopping configuration
- Warning when changes require model restart

---

## Callback Organization

### File Structure

```
callbacks/
├── training_callbacks.py        # Main page training
├── training_hub_callbacks.py    # Training Hub monitoring
├── visualization_callbacks.py   # Scatter plot rendering
├── labeling_callbacks.py        # Sample selection/labeling
└── config_callbacks.py          # Configuration updates
```

### What Lives Where

**training_callbacks.py:**
- Modal toggle (train button → confirmation)
- Training confirmation (start worker thread)
- Polling loop (process metrics queue)
- Main page status display

**training_hub_callbacks.py:**
- Loss curves rendering
- Terminal output display
- Dataset statistics
- Metrics table updates
- Stop training button
- Hub-specific polling

**visualization_callbacks.py:**
- Scatter plot rendering
- Color mode changes
- Figure caching
- Sample selection from plot

**labeling_callbacks.py:**
- Sample image rendering
- Label button handlers
- Keyboard shortcuts (0-9)
- Dataset stats panel

**config_callbacks.py:**
- Parameter slider updates
- Architecture warnings
- Config persistence

---

## Performance Considerations

### Scatter Plot Caching

The 60k-point scatter plot is expensive to build (2-4s). We cache aggressively:

```python
# Cache structure
app_state.cache = {
    "base_figures": {
        (mode, version): plotly_figure,  # Full figure
        ...
    },
    "colors": {
        mode: ["#ff0000", ...],  # 60k color strings
        ...
    }
}
```

**Cache hits:** ~50ms (just update selection marker)  
**Cache misses:** ~2-4s (rebuild entire figure)

**Invalidation:**
- Color cache: Only when data version changes
- Figure cache: Only when mode or data changes
- Manual clear: Via config page if needed

### Training Updates

**Polling frequency:** 2 seconds (configurable via interval)

**Adaptive polling:**
- Enabled during training (fast updates)
- Disabled when idle (save resources)
- Re-enabled when messages in queue

**Update triggers:**
- Terminal: Every poll if messages processed
- Loss curves: Every poll during training
- Metrics table: Only when new epoch completes
- Scatter plot: Only when latent updated (end of training)

---

## Common Issues & Solutions

### Issue: Training button has no effect

**Symptoms:** Click "Train Model", nothing happens

**Diagnosis:**
1. Check browser console for JavaScript errors
2. Check terminal for "Train button clicked" message
3. Verify epochs input has a value (not None)

**Common causes:**
- Epochs input empty → modal doesn't open
- Modal blocked by browser popup blocker
- Callback error (check server logs)

**Fix:**
1. Ensure epochs input filled before clicking
2. Check server terminal for error messages
3. Look for modal dialog (may be behind window)

### Issue: Real-time updates not showing

**Symptoms:** Terminal/curves only update at end of training

**Diagnosis:**
1. Check if polling is enabled (should be during training)
2. Check if messages are in queue (server logs show "epoch_complete")
3. Check if callbacks have poll interval as input

**Fix:**
Ensure callbacks include poll trigger:
```python
@app.callback(
    Output("terminal", "children"),
    Input("poll", "n_intervals"),  # This triggers every 2s
    ...
)
```

### Issue: Scatter plot empty or slow

**Symptoms:** No points, or 2-4s delay on every interaction

**Diagnosis:**
1. Check if cache initialized in `state.py`
2. Check if figure cache hit (fast path)
3. Check browser console for WebGL errors

**Common causes:**
- Cache not initialized → rebuild every time
- Cache key wrong → never hit
- Browser memory full → cache cleared

**Fix:**
1. Verify `initialize_model_and_data()` creates cache
2. Check `visualization_callbacks.py` cache logic
3. Clear browser cache and reload

### Issue: Labels not persisting

**Symptoms:** Labels disappear after refresh

**Diagnosis:**
1. Check if `labels.csv` updated (timestamps)
2. Check server logs for CSV write errors
3. Verify label store version incremented

**Common causes:**
- File permissions (CSV not writable)
- CSV format corrupted
- Label command validation failed

**Fix:**
1. Check `data/mnist/labels.csv` exists and writable
2. Verify CSV has header: `Serial,label`
3. Check command history for label command errors

### Issue: Training hangs or doesn't stop

**Symptoms:** Stop button clicked, training continues forever

**Diagnosis:**
1. Check if `stop_requested` flag set (state inspection)
2. Check if callback checks flag (training_callbacks.py)
3. Check if epoch is very long (100k+ samples)

**Common causes:**
- Stop check not implemented in callback
- Training stuck in long epoch (will stop after)
- Thread deadlock (rare)

**Fix:**
1. Wait for current epoch to complete (~20s)
2. Check server logs for "Stop requested" message
3. If stuck >2 minutes, restart dashboard

---

## Development Tips

### Adding a New Command

1. Define command class in `commands.py`
2. Implement `validate()` and `execute()`
3. Add to callback in appropriate file
4. Test validation catches errors
5. Test execution updates state correctly

### Adding a New Callback

1. Identify which file (training/labeling/visualization/config)
2. Add `@app.callback` decorator
3. Include necessary Input/Output components
4. Access state with lock
5. Return updates or `no_update`

### Debugging State Issues

```python
# In callback, add temporary logging
with dashboard_state.state_lock:
    print(f"DEBUG: labels version = {dashboard_state.app_state.data.version}")
    print(f"DEBUG: training state = {dashboard_state.app_state.training.state}")
    print(f"DEBUG: labeled count = {np.sum(~np.isnan(dashboard_state.app_state.data.labels))}")
```

### Testing Without UI

```python
# Test commands directly
from use_cases.dashboard.commands import LabelSampleCommand
from use_cases.dashboard import state as dashboard_state

dashboard_state.initialize_model_and_data()

cmd = LabelSampleCommand(sample_idx=0, label=5)
error = cmd.validate(dashboard_state.app_state)
assert error is None, f"Validation failed: {error}"

success, msg = dashboard_state.dispatcher.execute(cmd)
assert success, f"Execution failed: {msg}"
print(f"✓ Command executed: {msg}")
```

---

## Architecture Decisions

### Why immutable dataclasses?

**Problem:** Mutable dicts allowed silent state corruption, race conditions, unclear mutation points

**Solution:** Frozen dataclasses make mutations impossible, force explicit updates via commands

**Tradeoff:** More verbose (can't just do `state["key"] = value`), but much safer

### Why command pattern?

**Problem:** State changes scattered across callbacks, no validation, no audit trail

**Solution:** Commands centralize validation, provide single execution path, enable logging

**Tradeoff:** More code (command class + validate + execute), but clearer and testable

### Why queue + polling?

**Problem:** Background training thread can't directly update Dash UI (not thread-safe)

**Solution:** Thread puts messages in queue, UI polls and processes them

**Tradeoff:** 2-second delay on updates (acceptable for training that takes minutes)

### Why two pages?

**Problem:** Main page crowded with both labeling controls AND training monitoring

**Solution:** Split into focused pages (labeling vs training)

**Tradeoff:** Navigation required, but each page is cleaner and more focused

---

## Further Reading

- **README.md** - Features, quick start, product vision
- **AGENT_GUIDE.md** - Patterns for extending the dashboard
- **State models** - `state_models.py` for dataclass definitions
- **Commands** - `commands.py` for all state-modifying operations
- **Callbacks** - `callbacks/*.py` for UI interaction handlers
