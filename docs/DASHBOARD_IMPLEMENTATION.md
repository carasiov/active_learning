# Dashboard Implementation Guide

## Overview

Build a web-based dashboard that centralizes the SSVAE active learning workflow: visualize latent space, label samples interactively, configure training parameters, and train in the background with live feedback.

**Goal:** Prototype dashboard for single-user localhost use. Pragmatic over polished.

**Tech Stack:**
- Dash + Plotly Scattergl (handles 60k points via WebGL)
- Background threading for training (simple, no async complexity)
- Queue-based metrics communication
- Global state with `threading.Lock()`

**File Structure:**
````
use_cases/dashboard/
  app.py                    # Main dashboard (start as single file)
  dashboard_callback.py     # Custom TrainingCallback for metrics queue
  README.md                 # Basic usage instructions
````

## Critical Constraints

### DO NOT MODIFY
- `src/ssvae/` - Model code
- `src/training/` - Training loop, losses, train state  
- `src/callbacks/` - Callback system (except adding DashboardMetricsCallback)
- `use_cases/scripts/` - CLI tools must keep working
- `data/mnist/mnist.py` - Data loaders

### MUST REUSE
````python
from ssvae import SSVAE, SSVAEConfig
from training.interactive_trainer import InteractiveTrainer
from callbacks import TrainingCallback
from data.mnist import load_train_images_for_ssvae

# InteractiveTrainer preserves optimizer state across train_epochs() calls
# SSVAE.predict(data) returns (latent, recon, pred_class, pred_certainty)
# labels.csv format: Serial (index), label (0-9 or NaN)
````

### THREAD SAFETY
````python
import threading
state_lock = threading.Lock()

# Always wrap state access
with state_lock:
    value = app_state["something"]
````

## State Structure
````python
app_state = {
    "model": None,              # SSVAE instance
    "trainer": None,            # InteractiveTrainer instance
    "config": None,             # SSVAEConfig instance
    "data": {
        "x_train": None,        # (60000, 28, 28) training images
        "labels": None,         # (60000,) with NaN for unlabeled
        "latent": None,         # (60000, 2) latent coords
        "reconstructed": None,
        "pred_classes": None,
        "pred_certainty": None
    },
    "training": {
        "active": False,        # Is training running?
        "thread": None,
        "target_epochs": 0
    },
    "ui": {
        "selected_sample": 0,
        "color_mode": "user_labels"
    },
    "history": {               # For live plotting (Phase 3)
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        # ... other metrics
    }
}

metrics_queue = Queue()  # Training thread → UI communication
````

---

# Phase 1: Static UI with Labeling

## User Capabilities
- Open dashboard → see latent space scatter (60k points)
- Click any point → view original + reconstructed images
- Press label button (0-9) → updates CSV immediately
- Delete label → removes from CSV
- Switch color modes (user labels / predictions / certainty)

## Implementation Notes

**Initialization:**
- Load model from `artifacts/checkpoints/ssvae.ckpt` if exists, else fresh model
- Load all 60k training images via `load_train_images_for_ssvae()`
- Load labels from `data/mnist/labels.csv` into numpy array (NaN for unlabeled)
- Run `model.predict()` once to generate initial latent space
- All of this in `initialize_model_and_data()` called before `app.run_server()`

**Layout Components:**
- Header: "SSVAE Active Learning Dashboard"
- Left (8 cols): Latent scatter plot + color mode radio buttons
- Right (4 cols): Selected sample header, original image, reconstructed image, prediction info, label buttons (0-9 + Delete)
- Use `dcc.Store(id="selected-sample-store")` to track selection

**Key Callbacks:**

1. **Update scatter plot** - Input: color mode + selected sample
   - Use `go.Scattergl()` (not `go.Scatter`) for performance
   - Color points based on mode (user_labels / predictions / certainty)
   - Highlight selected sample with red X marker overlay

2. **Handle clicks** - Input: latent-scatter clickData
   - Extract: `clickData["points"][0]["pointIndex"]`
   - Update selected-sample-store
   - Update app_state["ui"]["selected_sample"] with lock

3. **Update images** - Input: selected-sample-store
   - Get original + recon from app_state["data"]
   - Convert numpy arrays to base64 PNG for `<img src>`
   - Show prediction info: "Predicted: X (Y% confidence) | User Label: Z"

4. **Label buttons** - 10 buttons (0-9), each with callback
   - Update app_state["data"]["labels"][selected_sample] = label
   - Update CSV: `pd.read_csv()` → modify → `pd.to_csv()`
   - Both operations inside `with state_lock:`
   - Use `prevent_initial_call=True`

5. **Delete button** - Similar but set label to NaN, remove row from CSV

## Tricky Parts

**Image encoding to base64:**
````python
import io, base64
from PIL import Image

def array_to_base64(arr):
    # Normalize to 0-255 first
    arr_norm = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
    img = Image.fromarray(arr_norm, mode='L')
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return "data:image/png;base64," + base64.b64encode(buffer.read()).decode()
````

**Label button callbacks:**
- Can't dynamically create callbacks in loop - need separate function per button or use pattern matching
- Each callback needs State for selected_sample, not Input (don't want to trigger on selection change)

**CSV thread safety:**
- All `pd.read_csv()` and `to_csv()` must be inside `with state_lock:`
- Otherwise concurrent labels corrupt the file

## Verification

**Automated:** Create `test_phase1.py` that checks:
- Files exist (app.py, README.md)
- Imports work (no syntax errors)
- Dashboard starts without immediate crash (run for 10 seconds)

**Manual:** 
- Click point → images update
- Click label button → CSV gains entry
- Click different point + label → CSV updates correctly
- Delete button removes label from CSV

**Success Criteria:**
- 60k scatter renders smoothly (<2s)
- Click detection works
- Images display correctly
- Labeling persists to CSV
- No crashes or race conditions

---

# Phase 2: Training Integration

## User Capabilities
- Adjust hyperparameters via sliders (recon weight, KL weight, learning rate)
- Set number of epochs via input box
- Click "Start Training" → background training begins
- See live status updates (epoch counter, current losses)
- Training completes → latent space automatically refreshes
- Click "Start Training" again → continues from previous state

## Implementation Notes

**New Layout Components:**
- Training controls card (above latent space):
  - Sliders: recon_weight (0-5000), kl_weight (0-1), learning_rate (0.0001-0.01)
  - Input: num_epochs (1-200)
  - Button: "Start Training" (disabled during training)
  - Status div: shows current epoch + metrics

- Polling: `dcc.Interval(id="training-poll", interval=1000)` for live updates

**Dashboard Callback:**
Create `dashboard_callback.py`:
````python
class DashboardMetricsCallback(TrainingCallback):
    def __init__(self, queue):
        self.queue = queue
    
    def on_epoch_end(self, epoch, metrics, history, trainer):
        # Push metrics dict to queue (convert JAX arrays to float)
        self.queue.put({
            "type": "epoch_complete",
            "epoch": epoch + 1,
            "train_loss": float(metrics["train"]["loss"]),
            "val_loss": float(metrics["val"]["loss"]),
            # ... other metrics
        })
    
    def on_train_end(self, history, trainer):
        self.queue.put({"type": "training_complete"})
````

**Training Worker Function:**
````python
def train_worker(epochs):
    """Runs in background thread."""
    try:
        with state_lock:
            app_state["training"]["active"] = True
            trainer = app_state["trainer"]
            x_train = app_state["data"]["x_train"]
            labels = app_state["data"]["labels"]
        
        # Training happens OUTSIDE lock (it's slow)
        callback = DashboardMetricsCallback(metrics_queue)
        trainer.train_epochs(
            num_epochs=epochs,
            data=x_train,
            labels=labels,
            callbacks=[callback]
        )
        
        # Regenerate latent space
        with state_lock:
            model = app_state["model"]
        
        latent, recon, pred_cls, pred_cert = model.predict(x_train)
        
        with state_lock:
            app_state["data"]["latent"] = latent
            app_state["data"]["reconstructed"] = recon
            app_state["data"]["pred_classes"] = pred_cls
            app_state["data"]["pred_certainty"] = pred_cert
            app_state["training"]["active"] = False
        
        metrics_queue.put({"type": "latent_updated"})
        
    except Exception as e:
        with state_lock:
            app_state["training"]["active"] = False
        metrics_queue.put({"type": "error", "message": str(e)})
````

**Key Callbacks:**

1. **Start training** - Input: train-button clicks, State: epochs + slider values
   - Update config: `app_state["config"].recon_weight = slider_value`
   - Update `trainer.config` with new values
   - Start thread: `threading.Thread(target=train_worker, args=(epochs,), daemon=True).start()`
   - Return: button disabled=True, sliders disabled=True

2. **Update status** - Input: training-poll interval
   - Drain metrics_queue: `while not metrics_queue.empty(): msg = queue.get_nowait()`
   - Display epoch messages, show progress indicator when active
   - Keep last 10 messages visible

3. **Refresh scatter** - Add training-poll as Input to existing scatter callback
   - This triggers scatter redraw when latent updates

## Tricky Parts

**Config updates:**
- Must update both `app_state["config"]` AND `trainer.config`
- InteractiveTrainer reads config from its own attribute, not from model

**Lock management:**
- Hold lock only for quick reads/writes
- DON'T hold lock during `train_epochs()` (blocks UI for minutes)
- Pattern: read data with lock → train without lock → write results with lock

**Queue polling:**
- UI callback checks queue every second via Interval
- Training thread pushes to queue (thread-safe by default)
- Drain entire queue each poll to avoid lag

**Button state:**
- Disable button + sliders when `app_state["training"]["active"] == True`
- Re-enable when training completes

## Verification

**Automated:** Check that:
- dashboard_callback.py exists and imports correctly
- Training controls present in layout
- train_worker function exists
- Dashboard starts with training UI

**Manual:**
- Adjust sliders → set epochs to 5 → click "Start Training"
- Watch status update every second
- Verify button disabled during training
- After completion, click point → images should reflect new model
- Start training again → should continue (not reset)

**Success Criteria:**
- Training runs in background without freezing UI
- Status updates in real-time
- Latent space refreshes automatically when done
- Can train multiple times sequentially
- No deadlocks or race conditions

---

# Phase 3: Metrics Visualization and Polish

## User Capabilities
- See live loss curves (total + components) during training
- View dataset statistics (total samples, labeled count, %)
- Better error handling (messages shown, no silent failures)
- Visual polish (loading states, smooth transitions)

## Implementation Notes

**New Layout Components:**
- Metrics row (above latent space, below training controls):
  - Left (8 cols): Loss curves graph `dcc.Graph(id="loss-curves")`
  - Right (4 cols): Dataset info card (labeled count, percentages, status)

**History Tracking:**
- Add `"history"` dict to app_state with lists for each metric
- DashboardMetricsCallback pushes special "update_history" messages
- UI callback accumulates these into app_state["history"]

**Key Callbacks:**

1. **Update loss curves** - Input: training-poll
   - Read app_state["history"]
   - Plot train_loss + val_loss (bold lines)
   - Plot component losses (lighter, smaller) for detail
   - Return empty plot with message if no history yet

2. **Update dataset stats** - Input: training-poll  
   - Count labeled samples: `np.sum(~np.isnan(labels))`
   - Calculate percentages
   - Show training status (idle / training)

3. **Modify status callback** - Handle "update_history" message type
   - Extract metrics from message
   - Append to appropriate history lists with lock

**Error Handling:**
- Wrap train_worker entire body in try/except
- Push error messages to queue with type="error"
- Display errors in red in status div
- Set training.active = False on error

## Tricky Parts

**History accumulation:**
- Need to handle "update_history" messages separately from display messages
- Both go through same queue, filter by type
- Append to lists with lock to prevent corruption

**Empty plot handling:**
- When history is empty (no training yet), return placeholder figure
- Check `len(history["epochs"]) == 0` before plotting

**Plotly figure construction:**
- Use different line styles for train vs val (solid vs dash)
- Use opacity for component losses so they don't overwhelm total loss
- `hovermode="x unified"` for better hover experience

**Dataset stats refresh:**
- Needs to recount labels every second (during polling)
- Fast operation, no performance concern

## Verification

**Automated:** Check that:
- Loss curves and dataset stats present in layout
- History tracking in app_state
- Error handling in train_worker
- Dashboard starts with full UI

**Manual:**
- Start training → watch loss curves appear and update
- Verify all component losses visible
- Label a sample → verify dataset stats update immediately
- Cause an error (invalid config?) → verify error message displays

**Success Criteria:**
- Loss curves update in real-time during training
- Dataset stats accurate and responsive
- Errors displayed gracefully (no crashes)
- Complete workflow smooth: label → train → visualize → repeat

---

# Technical Reference

## Paths
````python
ROOT_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = ROOT_DIR / "artifacts" / "checkpoints" / "ssvae.ckpt"
LABELS_PATH = ROOT_DIR / "data" / "mnist" / "labels.csv"
````

## Dependencies to Add
````bash
poetry add dash dash-bootstrap-components plotly
````

## Common Gotchas

**Scattergl vs Scatter:**
- Must use `go.Scattergl()` not `go.Scatter()` for 60k points
- Scattergl uses WebGL, much faster for large datasets

**Callback prevent_initial_call:**
- Usually want `prevent_initial_call=True` for button callbacks
- Otherwise they trigger on page load with None values

**JAX array to Python:**
- Metrics from callbacks are JAX arrays
- Convert to Python: `float(jax_array)` before putting in queue or state

**CSV index handling:**
- labels.csv has Serial as index (row numbers)
- When modifying: `df.loc[idx, "label"] = value`
- When deleting: `df = df.drop(idx)` then `df.to_csv()`

**Lock duration:**
- Keep lock held as short as possible
- Pattern: lock → copy data out → unlock → process → lock → write back

**Queue emptying:**
- Always drain queue in loop: `while not metrics_queue.empty():`
- Don't assume only one message per poll

## Debugging Tips

**If scatter doesn't render:**
- Check browser console for JavaScript errors
- Verify data shapes: latent should be (N, 2)
- Ensure Scattergl has `mode="markers"`

**If training freezes UI:**
- Check that training is in background thread
- Verify lock is NOT held during train_epochs call

**If labels don't persist:**
- Add print statements around CSV write
- Check state_lock is used
- Verify LABELS_PATH is correct

**If callbacks don't trigger:**
- Check Input/Output component IDs match layout
- Verify callback decorator syntax correct
- Look for typos in component IDs (common issue)

---

# Implementation Strategy

**Commit 1: Phase 1**
- Implement static UI with labeling
- Verify manually in browser
- Commit when working

**Commit 2: Phase 2**  
- Add training integration
- Verify training works without freezing
- Commit when working

**Commit 3: Phase 3**
- Add metrics visualization
- Final polish and error handling
- Commit when complete

Each phase builds on previous without breaking it. Test after each phase before proceeding.

---

# README Template

Create `use_cases/dashboard/README.md`:
````markdown
# SSVAE Dashboard

Interactive dashboard for semi-supervised active learning.

## Quick Start
```bash
poetry run python use_cases/dashboard/app.py
```

Open http://localhost:8050

## Features

- Interactive 60k-point latent space visualization
- Click-to-label workflow (updates CSV immediately)
- Configure training parameters (loss weights, learning rate)
- Background training with live progress
- Real-time loss curve visualization
- Dataset statistics tracking

## Workflow

1. Browse latent space, click uncertain points
2. Label them (0-9)
3. Adjust training parameters if desired
4. Click "Start Training"
5. Watch metrics update
6. Repeat

## Architecture

- Single-user localhost deployment
- Background threading for training
- State preserved across training sessions
- Integrates with existing CLI tools (train.py, infer.py still work)
````
