Implementation Plan for Agent (already done as this current codebase state, this document is just for reference)

# Multi-Model Architecture Implementation Guide

## Project Context

You're extending an existing SSVAE Active Learning Dashboard to support multiple independent model experiments. Currently, the dashboard supports only a single model instance with global state. We're refactoring to allow researchers to create, manage, and switch between multiple models, each with isolated labels, training history, and checkpoints.

**Current Architecture Strengths:**
- Immutable state models (DataState, TrainingStatus, UIState, TrainingHistory)
- Command pattern with validation/execution separation
- Thread-safe state access via locks
- Clean callback organization

**What We're Building (MVP v1):**
- Multiple independent models (isolated state, labels, history, checkpoints)
- Home page with model cards (create, open, delete)
- One active model at a time (no side-by-side comparison yet)
- Training history persists across dashboard restarts
- MNIST only (generic dataset loading deferred to v2)

**Explicitly NOT Building Yet:**
- Model forking/cloning
- Side-by-side comparison views
- Batch export operations
- Custom active learning strategies

---

## Design Decisions

### File Structure
Models live in isolated directories:
```
artifacts/models/
├── baseline_001/
│   ├── checkpoint.ckpt       # Model weights
│   ├── labels.csv            # THIS model's labels (isolated)
│   ├── history.json          # Training curves (NEW: persisted)
│   └── metadata.json         # Created date, description, config
├── contrastive_002/
│   └── ...
```

**Why isolated labels?** Each model is an independent experiment. Labels are part of the experimental state, not shared across models. This allows A/B testing different labeling strategies.

### State Architecture
We're introducing a two-tier system:

**Tier 1: ModelMetadata** (lightweight, all models)
```python
@dataclass(frozen=True)
class ModelMetadata:
    model_id: str              # Unique ID (e.g., "baseline_001")
    name: str                  # User-friendly name
    created_at: str           
    last_modified: str        
    dataset: str               # "mnist"
    total_epochs: int          # Accumulated training
    labeled_count: int         # Current label count
    latest_loss: Optional[float]
```

**Tier 2: ModelState** (heavy, only active model)
```python
@dataclass(frozen=True)
class ModelState:
    model_id: str
    model: SSVAE                    # Full model in memory
    trainer: InteractiveTrainer
    config: SSVAEConfig
    data: DataState
    training: TrainingStatus
    ui: UIState
    history: TrainingHistory
```

**Tier 1** lives in the registry for fast home page rendering. **Tier 2** is loaded only when user opens a model. This keeps memory footprint reasonable.

### Auto-Save Strategy
- **Labels:** Immediate save after each label (existing behavior, now model-scoped)
- **History:** Save to `history.json` after each epoch completes (NEW)
- **Checkpoint:** Save after training completes (existing behavior)
- **Metadata:** Update `last_modified` on any state change (NEW)

### Model Naming
Hybrid approach:
- User provides optional friendly name (e.g., "Baseline Experiment")
- System generates unique ID (e.g., `model_001`, `model_002`)
- Directory uses ID, UI shows name
- If no name provided, use ID as name

### Config Presets (Creation Modal)
Offer three starting points:
1. **Default MNIST** (current defaults: recon=1000, kl=0.1, lr=0.001)
2. **High Reconstruction** (recon=5000, kl=0.01, lr=0.001)
3. **Classification Focus** (label_weight=10, recon=500, kl=0.1, lr=0.001)

User can still tweak via Advanced Config page after creation.

---

## Implementation Tasks

### Task 1: Create New State Models

**File:** `use_cases/dashboard/state_models.py`

**What to add:**

```python
@dataclass(frozen=True)
class ModelMetadata:
    """Lightweight model info for home page and registry."""
    model_id: str
    name: str
    created_at: str  # ISO format
    last_modified: str  # ISO format
    dataset: str
    total_epochs: int
    labeled_count: int
    latest_loss: Optional[float]
    
    @classmethod
    def from_dict(cls, data: dict) -> ModelMetadata:
        """Load from metadata.json"""
        return cls(**data)
    
    def to_dict(self) -> dict:
        """Serialize to metadata.json"""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "dataset": self.dataset,
            "total_epochs": self.total_epochs,
            "labeled_count": self.labeled_count,
            "latest_loss": self.latest_loss,
        }
```

**Refactor existing ModelState:**
- Rename current `AppState` internals to be nested inside `ModelState`
- ModelState becomes the full encapsulation of one model's state
- AppState becomes registry of models + active model

```python
@dataclass(frozen=True)
class ModelState:
    """Full state for one model - only loaded when active."""
    model_id: str
    metadata: ModelMetadata  # Embed metadata here too
    model: SSVAE
    trainer: InteractiveTrainer
    config: SSVAEConfig
    data: DataState
    training: TrainingStatus
    ui: UIState
    history: TrainingHistory
    
    def with_updated_metadata(self, **kwargs) -> ModelState:
        """Helper to update metadata fields."""
        new_metadata = replace(self.metadata, **kwargs)
        return replace(self, metadata=new_metadata)

@dataclass(frozen=True)
class AppState:
    """Root state - manages model registry."""
    models: Dict[str, ModelMetadata]  # All models (lightweight)
    active_model: Optional[ModelState]  # Only one loaded
    cache: Dict[str, object]  # Shared cache
    
    def with_active_model(self, model_state: ModelState) -> AppState:
        """Load a model as active."""
        # Update registry with latest metadata
        updated_models = dict(self.models)
        updated_models[model_state.model_id] = model_state.metadata
        return replace(
            self,
            models=updated_models,
            active_model=model_state,
            cache={"base_figures": {}, "colors": {}}  # Clear cache
        )
    
    def with_unloaded_model(self) -> AppState:
        """Unload active model."""
        return replace(self, active_model=None, cache={})
    
    def with_model_metadata(self, metadata: ModelMetadata) -> AppState:
        """Update registry entry."""
        updated_models = dict(self.models)
        updated_models[metadata.model_id] = metadata
        return replace(self, models=updated_models)
```

**Verification:**
```bash
poetry run python -c "
from use_cases.dashboard.state_models import ModelMetadata, ModelState, AppState
print('✓ New state models import successfully')
"
```

---

### Task 2: Create Model Persistence Layer

**New file:** `use_cases/dashboard/model_manager.py`

This handles all disk I/O for models. Keep it pure (no state mutations).

**What to implement:**

```python
"""Model persistence layer - handles save/load to disk."""

from pathlib import Path
from typing import Optional, Dict, List
import json
from datetime import datetime

import numpy as np
import pandas as pd

from use_cases.dashboard.state_models import ModelMetadata, ModelState, TrainingHistory
from ssvae import SSVAE, SSVAEConfig

MODELS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "models"


class ModelManager:
    """Handles model persistence to disk."""
    
    @staticmethod
    def ensure_models_dir() -> None:
        """Create models directory if missing."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def generate_model_id() -> str:
        """Generate unique model ID like 'model_001'."""
        ModelManager.ensure_models_dir()
        existing = [d.name for d in MODELS_DIR.iterdir() if d.is_dir()]
        
        # Extract numbers from existing IDs
        numbers = []
        for name in existing:
            if name.startswith("model_"):
                try:
                    num = int(name.split("_")[1])
                    numbers.append(num)
                except (IndexError, ValueError):
                    pass
        
        next_num = max(numbers, default=0) + 1
        return f"model_{next_num:03d}"
    
    @staticmethod
    def model_dir(model_id: str) -> Path:
        """Get directory path for model."""
        return MODELS_DIR / model_id
    
    @staticmethod
    def create_model_directory(model_id: str) -> Path:
        """Create model directory structure."""
        model_path = ModelManager.model_dir(model_id)
        model_path.mkdir(parents=True, exist_ok=True)
        return model_path
    
    @staticmethod
    def save_metadata(metadata: ModelMetadata) -> None:
        """Save metadata.json."""
        path = ModelManager.model_dir(metadata.model_id) / "metadata.json"
        with open(path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
    
    @staticmethod
    def load_metadata(model_id: str) -> Optional[ModelMetadata]:
        """Load metadata.json."""
        path = ModelManager.model_dir(model_id) / "metadata.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return ModelMetadata.from_dict(data)
    
    @staticmethod
    def save_history(model_id: str, history: TrainingHistory) -> None:
        """Save training history to history.json."""
        path = ModelManager.model_dir(model_id) / "history.json"
        data = {
            "epochs": list(history.epochs),
            "train_loss": list(history.train_loss),
            "val_loss": list(history.val_loss),
            "train_reconstruction_loss": list(history.train_reconstruction_loss),
            "val_reconstruction_loss": list(history.val_reconstruction_loss),
            "train_kl_loss": list(history.train_kl_loss),
            "val_kl_loss": list(history.val_kl_loss),
            "train_classification_loss": list(history.train_classification_loss),
            "val_classification_loss": list(history.val_classification_loss),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load_history(model_id: str) -> TrainingHistory:
        """Load training history from history.json."""
        path = ModelManager.model_dir(model_id) / "history.json"
        if not path.exists():
            return TrainingHistory.empty()
        
        with open(path, "r") as f:
            data = json.load(f)
        
        return TrainingHistory(
            epochs=data.get("epochs", []),
            train_loss=data.get("train_loss", []),
            val_loss=data.get("val_loss", []),
            train_reconstruction_loss=data.get("train_reconstruction_loss", []),
            val_reconstruction_loss=data.get("val_reconstruction_loss", []),
            train_kl_loss=data.get("train_kl_loss", []),
            val_kl_loss=data.get("val_kl_loss", []),
            train_classification_loss=data.get("train_classification_loss", []),
            val_classification_loss=data.get("val_classification_loss", []),
        )
    
    @staticmethod
    def checkpoint_path(model_id: str) -> Path:
        """Get checkpoint path for model."""
        return ModelManager.model_dir(model_id) / "checkpoint.ckpt"
    
    @staticmethod
    def labels_path(model_id: str) -> Path:
        """Get labels CSV path for model."""
        return ModelManager.model_dir(model_id) / "labels.csv"
    
    @staticmethod
    def list_all_models() -> Dict[str, ModelMetadata]:
        """Scan models directory and load all metadata."""
        ModelManager.ensure_models_dir()
        models = {}
        
        for model_dir in MODELS_DIR.iterdir():
            if not model_dir.is_dir():
                continue
            
            metadata = ModelManager.load_metadata(model_dir.name)
            if metadata:
                models[metadata.model_id] = metadata
        
        return models
    
    @staticmethod
    def delete_model(model_id: str) -> None:
        """Delete model directory and all contents."""
        import shutil
        model_path = ModelManager.model_dir(model_id)
        if model_path.exists():
            shutil.rmtree(model_path)
```

**Verification:**
```bash
poetry run python -c "
from use_cases.dashboard.model_manager import ModelManager
ModelManager.ensure_models_dir()
model_id = ModelManager.generate_model_id()
print(f'✓ Generated model ID: {model_id}')
assert model_id == 'model_001'
"
```

---

### Task 3: Refactor state.py for Multi-Model

**File:** `use_cases/dashboard/state.py`

**Current problem:** Everything assumes single global model. We need to support registry + active model pattern.

**Key changes:**

1. Remove old global paths (these become model-specific):
```python
# DELETE THESE (now in ModelManager)
CHECKPOINT_PATH = ...
LABELS_PATH = ...
```

2. Update initialization to load model registry:
```python
def initialize_app_state() -> None:
    """Initialize app state with model registry."""
    global app_state
    
    if app_state is not None:
        return
    
    with _init_lock:
        if app_state is not None:
            return
        
        from use_cases.dashboard.model_manager import ModelManager
        
        # Load all model metadata
        models = ModelManager.list_all_models()
        
        # Create empty registry (no active model)
        with state_lock:
            app_state = AppState(
                models=models,
                active_model=None,
                cache={}
            )

def initialize_model_and_data():
    """DEPRECATED: Use load_model(model_id) instead."""
    # Keep for backward compat during transition
    initialize_app_state()
```

3. Add model loading function:
```python
def load_model(model_id: str) -> None:
    """Load a specific model as active."""
    global app_state
    from use_cases.dashboard.model_manager import ModelManager
    from data.mnist import load_train_images_for_ssvae, load_mnist_splits
    
    # Ensure app state initialized
    initialize_app_state()
    
    with state_lock:
        # Don't reload if already active
        if app_state.active_model and app_state.active_model.model_id == model_id:
            return
        
        # Load metadata
        metadata = ModelManager.load_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not found")
        
        # Load model
        model = SSVAE(input_dim=(28, 28), config=SSVAEConfig())
        checkpoint_path = ModelManager.checkpoint_path(model_id)
        if checkpoint_path.exists():
            model.load_model_weights(str(checkpoint_path))
            model.weights_path = str(checkpoint_path)
        
        trainer = InteractiveTrainer(model)
        
        # Load data
        x_train = load_train_images_for_ssvae(dtype=np.float32)
        (_, true_labels), _ = load_mnist_splits(normalize=True, reshape=False, dtype=np.float32)
        true_labels = np.asarray(true_labels, dtype=np.int32)
        
        # Load history
        history = ModelManager.load_history(model_id)
        
        # Load labels
        labels_array = np.full(shape=(x_train.shape[0],), fill_value=np.nan, dtype=float)
        labels_path = ModelManager.labels_path(model_id)
        if labels_path.exists():
            stored_labels = pd.read_csv(labels_path)
            if not stored_labels.empty:
                stored_labels["Serial"] = pd.to_numeric(stored_labels["Serial"], errors="coerce")
                stored_labels = stored_labels.dropna(subset=["Serial"])
                stored_labels["Serial"] = stored_labels["Serial"].astype(int)
                stored_labels["label"] = pd.to_numeric(stored_labels.get("label"), errors="coerce").astype("Int64")
                serials = stored_labels["Serial"].to_numpy()
                label_values = stored_labels["label"].astype(int).to_numpy()
                valid_mask = (serials >= 0) & (serials < x_train.shape[0])
                labels_array[serials[valid_mask]] = label_values[valid_mask].astype(float)
        
        # Get predictions
        latent, recon, pred_classes, pred_certainty = model.predict(x_train)
        hover_metadata = _build_hover_metadata(pred_classes, pred_certainty, labels_array, true_labels)
        
        # Build ModelState
        data_state = DataState(
            x_train=x_train,
            labels=labels_array,
            true_labels=true_labels,
            latent=latent,
            reconstructed=recon,
            pred_classes=pred_classes,
            pred_certainty=pred_certainty,
            hover_metadata=hover_metadata,
            version=0
        )
        
        training_status = TrainingStatus(
            state=TrainingState.IDLE,
            target_epochs=0,
            status_messages=[],
            thread=None
        )
        
        ui_state = UIState(
            selected_sample=0,
            color_mode="user_labels"
        )
        
        model_state = ModelState(
            model_id=model_id,
            metadata=metadata,
            model=model,
            trainer=trainer,
            config=model.config,
            data=data_state,
            training=training_status,
            ui=ui_state,
            history=history
        )
        
        # Update app state
        app_state = app_state.with_active_model(model_state)
```

4. Update helper functions to use active model:
```python
def _load_labels_dataframe() -> pd.DataFrame:
    """Load labels CSV for ACTIVE model."""
    if app_state.active_model is None:
        return pd.DataFrame(columns=["label"])
    
    from use_cases.dashboard.model_manager import ModelManager
    labels_path = ModelManager.labels_path(app_state.active_model.model_id)
    
    # ... rest of existing logic using labels_path

def _persist_labels_dataframe(df: pd.DataFrame) -> None:
    """Save labels CSV for ACTIVE model."""
    if app_state.active_model is None:
        return
    
    from use_cases.dashboard.model_manager import ModelManager
    labels_path = ModelManager.labels_path(app_state.active_model.model_id)
    
    # ... rest of existing logic using labels_path
```

**Verification:**
```bash
poetry run python -c "
from use_cases.dashboard import state as dashboard_state
dashboard_state.initialize_app_state()
print('✓ App state initialized')
print(f'Models in registry: {len(dashboard_state.app_state.models)}')
"
```

---

### Task 4: Create New Commands

**File:** `use_cases/dashboard/commands.py`

Add three new commands for model lifecycle.

```python
@dataclass
class CreateModelCommand(Command):
    """Create a new model with fresh state."""
    name: Optional[str] = None  # User-friendly name (optional)
    config_preset: str = "default"  # "default", "high_recon", "classification"
    
    def validate(self, state: AppState) -> Optional[str]:
        """Validate preset exists."""
        valid_presets = {"default", "high_recon", "classification"}
        if self.config_preset not in valid_presets:
            return f"Invalid preset: {self.config_preset}. Must be one of {valid_presets}"
        return None
    
    def execute(self, state: AppState) -> Tuple[AppState, str]:
        """Create new model directory and metadata."""
        from use_cases.dashboard.model_manager import ModelManager
        from datetime import datetime
        from use_cases.dashboard import state as dashboard_state
        
        # Generate ID
        model_id = ModelManager.generate_model_id()
        display_name = self.name if self.name else model_id
        
        # Create directory
        ModelManager.create_model_directory(model_id)
        
        # Create config based on preset
        config = SSVAEConfig()
        if self.config_preset == "high_recon":
            config.recon_weight = 5000.0
            config.kl_weight = 0.01
        elif self.config_preset == "classification":
            config.label_weight = 10.0
            config.recon_weight = 500.0
        
        # Create metadata
        now = datetime.utcnow().isoformat()
        metadata = ModelMetadata(
            model_id=model_id,
            name=display_name,
            created_at=now,
            last_modified=now,
            dataset="mnist",
            total_epochs=0,
            labeled_count=0,
            latest_loss=None
        )
        
        # Save files
        ModelManager.save_metadata(metadata)
        ModelManager.save_history(model_id, TrainingHistory.empty())
        
        # Initialize model and save checkpoint
        model = SSVAE(input_dim=(28, 28), config=config)
        checkpoint_path = ModelManager.checkpoint_path(model_id)
        model.save_model_weights(str(checkpoint_path))
        
        # Create empty labels.csv
        labels_path = ModelManager.labels_path(model_id)
        pd.DataFrame(columns=["Serial", "label"]).to_csv(labels_path, index=False)
        
        # Update registry
        new_state = state.with_model_metadata(metadata)
        
        # Auto-load as active
        dashboard_state.load_model(model_id)
        
        return new_state, f"Created model: {display_name}"


@dataclass
class LoadModelCommand(Command):
    """Load a model as active."""
    model_id: str
    
    def validate(self, state: AppState) -> Optional[str]:
        """Check model exists."""
        if self.model_id not in state.models:
            return f"Model not found: {self.model_id}"
        
        # Don't reload if already active
        if state.active_model and state.active_model.model_id == self.model_id:
            return "Model already loaded"
        
        return None
    
    def execute(self, state: AppState) -> Tuple[AppState, str]:
        """Load model into active state."""
        from use_cases.dashboard import state as dashboard_state
        
        # Load via state.py helper (handles all the complexity)
        dashboard_state.load_model(self.model_id)
        
        # State is updated in-place by load_model, return current state
        return dashboard_state.app_state, f"Loaded model: {self.model_id}"


@dataclass
class DeleteModelCommand(Command):
    """Delete a model permanently."""
    model_id: str
    
    def validate(self, state: AppState) -> Optional[str]:
        """Check model exists and not active."""
        if self.model_id not in state.models:
            return f"Model not found: {self.model_id}"
        
        if state.active_model and state.active_model.model_id == self.model_id:
            return "Cannot delete active model. Switch to another model first."
        
        return None
    
    def execute(self, state: AppState) -> Tuple[AppState, str]:
        """Delete model files and remove from registry."""
        from use_cases.dashboard.model_manager import ModelManager
        
        # Get name for message
        model_name = state.models[self.model_id].name
        
        # Delete files
        ModelManager.delete_model(self.model_id)
        
        # Remove from registry
        updated_models = dict(state.models)
        del updated_models[self.model_id]
        new_state = replace(state, models=updated_models)
        
        return new_state, f"Deleted model: {model_name}"
```

**Important:** Update existing commands to work with active model. Example:

```python
# In LabelSampleCommand.execute():
def execute(self, state: AppState) -> Tuple[AppState, str]:
    """Execute label update on ACTIVE model."""
    if state.active_model is None:
        return state, "No model loaded"
    
    from use_cases.dashboard.model_manager import ModelManager
    from datetime import datetime
    
    # ... existing label update logic ...
    
    # Save labels to model-specific CSV
    labels_path = ModelManager.labels_path(state.active_model.model_id)
    # ... save logic ...
    
    # Update metadata
    labeled_count = int(np.sum(~np.isnan(new_labels)))
    updated_model = state.active_model.with_updated_metadata(
        labeled_count=labeled_count,
        last_modified=datetime.utcnow().isoformat()
    )
    
    # Save metadata
    ModelManager.save_metadata(updated_model.metadata)
    
    # Update state
    new_state = state.with_active_model(updated_model)
    return new_state, message
```

Apply similar pattern to `StartTrainingCommand`, `CompleteTrainingCommand`, etc. They should all check `state.active_model` and update metadata appropriately.

**Verification:**
```bash
poetry run python -c "
from use_cases.dashboard.commands import CreateModelCommand
from use_cases.dashboard import state as dashboard_state
dashboard_state.initialize_app_state()
cmd = CreateModelCommand(name='Test Model', config_preset='default')
success, msg = dashboard_state.dispatcher.execute(cmd)
print(f'✓ {msg}')
assert success
"
```

---

### Task 5: Build Home Page Layout

**New file:** `use_cases/dashboard/pages_home.py`

Create the home page UI with model cards.

```python
"""Home page - Model selection and creation."""

from dash import dcc, html
import dash_bootstrap_components as dbc

from use_cases.dashboard import state as dashboard_state


def build_home_layout() -> html.Div:
    """Build the home page with model cards."""
    dashboard_state.initialize_app_state()
    
    with dashboard_state.state_lock:
        models = dashboard_state.app_state.models
    
    # Empty state
    if not models:
        return _build_empty_state()
    
    # Model cards
    model_cards = []
    for metadata in sorted(models.values(), key=lambda m: m.last_modified, reverse=True):
        model_cards.append(_build_model_card(metadata))
    
    return html.Div(
        [
            # Header
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(
                                src="/assets/infoteam_logo_basic.png",
                                style={"height": "50px"},
                            ),
                        ],
                        style={"display": "inline-block", "marginRight": "32px"},
                    ),
                    html.Div(
                        [
                            html.H1("SSVAE Research Hub", style={
                                "fontSize": "28px",
                                "fontWeight": "700",
                                "margin": "0",
                                "color": "#000000",
                            }),
                            html.Div("Manage your semi-supervised learning experiments", style={
                                "fontSize": "15px",
                                "color": "#6F6F6F",
                            }),
                        ],
                        style={"display": "inline-block", "verticalAlign": "middle"},
                    ),
                ],
                style={
                    "padding": "24px 48px",
                    "backgroundColor": "#ffffff",
                },
            ),
            html.Div(style={"height": "4px", "backgroundColor": "#C10A27"}),
            
            # Action bar
            html.Div(
                [
                    dbc.Button(
                        "+ New Model",
                        id="home-new-model-btn",
                        n_clicks=0,
                        style={
                            "backgroundColor": "#C10A27",
                            "border": "none",
                            "borderRadius": "8px",
                            "padding": "12px 24px",
                            "fontSize": "15px",
                            "fontWeight": "700",
                            "color": "#ffffff",
                        },
                    ),
                ],
                style={"padding": "24px 48px", "backgroundColor": "#f5f5f5"},
            ),
            
            # Model grid
            html.Div(
                model_cards,
                style={
                    "padding": "32px 48px",
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fill, minmax(400px, 1fr))",
                    "gap": "24px",
                },
            ),
            
            # Create model modal
            _build_create_modal(),
        ],
        style={
            "fontFamily": "'Open Sans', Verdana, sans-serif",
            "backgroundColor": "#fafafa",
            "minHeight": "100vh",
        },
    )


def _build_empty_state() -> html.Div:
    """Empty state when no models exist."""
    return html.Div(
        [
            # Header (same as above)
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(
                                src="/assets/infoteam_logo_basic.png",
                                style={"height": "50px"},
                            ),
                        ],
                        style={"display": "inline-block", "marginRight": "32px"},
                    ),
                    html.Div(
                        [
                            html.H1("SSVAE Research Hub", style={
                                "fontSize": "28px",
                                "fontWeight": "700",
                                "margin": "0",
                                "color": "#000000",
                            }),
                            html.Div("Manage your semi-supervised learning experiments", style={
                                "fontSize": "15px",
                                "color": "#6F6F6F",
                            }),
                        ],
                        style={"display": "inline-block", "verticalAlign": "middle"},
                    ),
                ],
                style={
                    "padding": "24px 48px",
                    "backgroundColor": "#ffffff",
                },
            ),
            html.Div(style={"height": "4px", "backgroundColor": "#C10A27"}),
            
            # Empty state message
            html.Div(
                [
                    html.Div("🎯", style={"fontSize": "64px", "marginBottom": "24px"}),
                    html.H2("No Models Yet", style={
                        "fontSize": "24px",
                        "fontWeight": "700",
                        "color": "#000000",
                        "marginBottom": "12px",
                    }),
                    html.P(
                        "Create your first model to start experimenting with semi-supervised learning.",
                        style={"fontSize": "16px", "color": "#6F6F6F", "marginBottom": "32px"},
                    ),
                    dbc.Button(
                        "+ Create Your First Model",
                        id="home-new-model-btn",
                        n_clicks=0,
                        style={
                            "backgroundColor": "#C10A27",
                            "border": "none",
                            "borderRadius": "8px",
                            "padding": "16px 32px",
                            "fontSize": "16px",
                            "fontWeight": "700",
                            "color": "#ffffff",
                        },
                    ),
                ],
                style={
                    "textAlign": "center",
                    "padding": "120px 48px",
                },
            ),
            
            # Create model modal
            _build_create_modal(),
        ],
        style={
            "fontFamily": "'Open Sans', Verdana, sans-serif",
            "backgroundColor": "#fafafa",
            "minHeight": "100vh",
        },
    )


def _build_model_card(metadata) -> html.Div:
    """Build a single model card."""
    from datetime import datetime
    
    # Format last modified
    try:
        last_mod = datetime.fromisoformat(metadata.last_modified)
        now = datetime.utcnow()
        delta = now - last_mod
        
        if delta.days > 0:
            time_ago = f"{delta.days}d ago"
        elif delta.seconds > 3600:
            time_ago = f"{delta.seconds // 3600}h ago"
        elif delta.seconds > 60:
            time_ago = f"{delta.seconds // 60}m ago"
        else:
            time_ago = "just now"
    except:
        time_ago = "unknown"
    
    # Loss display
    loss_display = f"{metadata.latest_loss:.4f}" if metadata.latest_loss else "—"
    
    return html.Div(
        [
            html.Div(
                [
                    html.H3(metadata.name, style={
                        "fontSize": "18px",
                        "fontWeight": "700",
                        "color": "#000000",
                        "marginBottom": "8px",
                    }),
                    html.Div(
                        f"{metadata.dataset.upper()} • {metadata.labeled_count} labels • {metadata.total_epochs} epochs",
                        style={"fontSize": "13px", "color": "#6F6F6F", "marginBottom": "12px"},
                    ),
                    html.Div(
                        [
                            html.Span(f"Last: {time_ago}", style={"marginRight": "16px"}),
                            html.Span(f"Loss: {loss_display}"),
                        ],
                        style={"fontSize": "13px", "color": "#6F6F6F"},
                    ),
                ],
                style={"flex": "1"},
            ),
            html.Div(
                [
                    dbc.Button(
                        "Open",
                        id={"type": "home-open-model", "model_id": metadata.model_id},
                        n_clicks=0,
                        style={
                            "backgroundColor": "#45717A",
                            "border": "none",
                            "borderRadius": "6px",
                            "padding": "8px 20px",
                            "fontSize": "14px",
                            "fontWeight": "600",
                            "color": "#ffffff",
                            "marginRight": "8px",
                        },
                    ),
                    dbc.Button(
                        "Delete",
                        id={"type": "home-delete-model", "model_id": metadata.model_id},
                        n_clicks=0,
                        style={
                            "backgroundColor": "#ffffff",
                            "border": "1px solid #C6C6C6",
                            "borderRadius": "6px",
                            "padding": "8px 16px",
                            "fontSize": "14px",
                            "fontWeight": "600",
                            "color": "#6F6F6F",
                        },
                    ),
                ],
            ),
        ],
        style={
            "backgroundColor": "#ffffff",
            "border": "1px solid #C6C6C6",
            "borderRadius": "8px",
            "padding": "24px",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
        },
    )


def _build_create_modal() -> dbc.Modal:
    """Modal for creating new model."""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Create New Model")),
            dbc.ModalBody(
                [
                    html.Div(
                        [
                            html.Label("Model Name (optional)", style={
                                "fontSize": "14px",
                                "fontWeight": "600",
                                "marginBottom": "6px",
                                "display": "block",
                            }),
                            dcc.Input(
                                id="home-model-name-input",
                                type="text",
                                placeholder="e.g., Baseline Experiment",
                                style={
                                    "width": "100%",
                                    "padding": "10px 12px",
                                    "fontSize": "14px",
                                    "border": "1px solid #C6C6C6",
                                    "borderRadius": "6px",
                                },
                            ),
                        ],
                        style={"marginBottom": "20px"},
                    ),
                    html.Div(
                        [
                            html.Label("Configuration Preset", style={
                                "fontSize": "14px",
                                "fontWeight": "600",
                                "marginBottom": "12px",
                                "display": "block",
                            }),
                            dbc.RadioItems(
                                id="home-config-preset",
                                options=[
                                    {"label": "Default MNIST (balanced)", "value": "default"},
                                    {"label": "High Reconstruction (recon=5000)", "value": "high_recon"},
                                    {"label": "Classification Focus (label_weight=10)", "value": "classification"},
                                ],
                                value="default",
                                style={"fontSize": "14px"},
                            ),
                        ],
                        style={"marginBottom": "20px"},
                    ),
                    html.Div(
                        id="home-create-feedback",
                        style={"fontSize": "14px", "marginTop": "12px"},
                    ),
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "Cancel",
                        id="home-cancel-create",
                        n_clicks=0,
                        style={
                            "backgroundColor": "#ffffff",
                            "border": "1px solid #C6C6C6",
                            "color": "#6F6F6F",
                        },
                    ),
                    dbc.Button(
                        "Create Model",
                        id="home-confirm-create",
                        n_clicks=0,
                        style={
                            "backgroundColor": "#C10A27",
                            "border": "none",
                            "color": "#ffffff",
                            "marginLeft": "8px",
                        },
                    ),
                ]
            ),
        ],
        id="home-create-modal",
        is_open=False,
        centered=True,
    )
```

**Verification:**
```bash
poetry run python -c "
from use_cases.dashboard.pages_home import build_home_layout
layout = build_home_layout()
print('✓ Home layout renders')
"
```

---

### Task 6: Add Home Page Callbacks

**New file:** `use_cases/dashboard/callbacks/home_callbacks.py`

Wire up the home page interactions.

```python
"""Home page callbacks - model management."""

from dash import Dash, Input, Output, State, ALL
from dash.exceptions import PreventUpdate

from use_cases.dashboard import state as dashboard_state
from use_cases.dashboard.commands import CreateModelCommand, LoadModelCommand, DeleteModelCommand


def register_home_callbacks(app: Dash) -> None:
    """Register home page callbacks."""
    
    @app.callback(
        Output("home-create-modal", "is_open"),
        Output("home-model-name-input", "value"),
        Output("home-config-preset", "value"),
        Input("home-new-model-btn", "n_clicks"),
        Input("home-confirm-create", "n_clicks"),
        Input("home-cancel-create", "n_clicks"),
        State("home-create-modal", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_create_modal(new_clicks, confirm_clicks, cancel_clicks, is_open):
        """Open/close create model modal."""
        import dash
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if triggered_id == "home-new-model-btn":
            return True, "", "default"
        elif triggered_id in ["home-confirm-create", "home-cancel-create"]:
            return False, "", "default"
        
        raise PreventUpdate
    
    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Output("home-create-feedback", "children"),
        Input("home-confirm-create", "n_clicks"),
        State("home-model-name-input", "value"),
        State("home-config-preset", "value"),
        prevent_initial_call=True,
    )
    def create_model(n_clicks, model_name, preset):
        """Create new model and navigate to it."""
        if not n_clicks:
            raise PreventUpdate
        
        # Create model
        command = CreateModelCommand(
            name=model_name if model_name else None,
            config_preset=preset
        )
        success, message = dashboard_state.dispatcher.execute(command)
        
        if not success:
            return dash.no_update, html.Div(message, style={"color": "#C10A27"})
        
        # Get newly created model ID
        with dashboard_state.state_lock:
            model_id = dashboard_state.app_state.active_model.model_id
        
        # Navigate to model
        return f"/model/{model_id}", ""
    
    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Input({"type": "home-open-model", "model_id": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def open_model(n_clicks_list):
        """Navigate to model dashboard."""
        import dash
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        # Find which button was clicked
        triggered_id = ctx.triggered[0]["prop_id"]
        if ".n_clicks" not in triggered_id:
            raise PreventUpdate
        
        # Extract model_id from triggered_id JSON
        import json
        id_dict = json.loads(triggered_id.split(".")[0])
        model_id = id_dict["model_id"]
        
        # Load model
        command = LoadModelCommand(model_id=model_id)
        success, message = dashboard_state.dispatcher.execute(command)
        
        if not success:
            raise PreventUpdate
        
        # Navigate
        return f"/model/{model_id}"
    
    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Input({"type": "home-delete-model", "model_id": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def delete_model(n_clicks_list):
        """Delete model with confirmation."""
        import dash
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        # Find which button was clicked
        triggered_id = ctx.triggered[0]["prop_id"]
        if ".n_clicks" not in triggered_id:
            raise PreventUpdate
        
        # Extract model_id
        import json
        id_dict = json.loads(triggered_id.split(".")[0])
        model_id = id_dict["model_id"]
        
        # TODO: Add confirmation dialog (defer to later task)
        # For now, just delete
        command = DeleteModelCommand(model_id=model_id)
        success, message = dashboard_state.dispatcher.execute(command)
        
        # Reload home page
        return "/"
```

**Verification:**
Test by running dashboard and creating a model through UI.

---

### Task 7: Update app.py Routing

**File:** `use_cases/dashboard/app.py`

Update routing to support home page and model-scoped pages.

**Key changes:**

1. Add home page import:
```python
from use_cases.dashboard.pages_home import build_home_layout
from use_cases.dashboard.callbacks.home_callbacks import register_home_callbacks
```

2. Update `display_page` callback:
```python
@app.callback(
    Output('page-content', 'children'),
    Output('training-config-store', 'data'),
    Input('url', 'pathname'),
)
def display_page(pathname):
    """Route to appropriate page."""
    import re
    import dataclasses
    
    # Initialize app state (registry only)
    dashboard_state.initialize_app_state()
    
    # Home page
    if pathname == '/' or pathname is None:
        return build_home_layout(), {}
    
    # Model-scoped pages: /model/{id}, /model/{id}/training-hub, etc.
    model_match = re.match(r'^/model/([^/]+)(/.*)?$', pathname)
    if model_match:
        model_id = model_match.group(1)
        sub_path = model_match.group(2) or ''
        
        # Load model if not already active
        with dashboard_state.state_lock:
            if not dashboard_state.app_state.active_model or \
               dashboard_state.app_state.active_model.model_id != model_id:
                dashboard_state.load_model(model_id)
            
            config_dict = dataclasses.asdict(dashboard_state.app_state.active_model.config)
        
        # Route to sub-page
        if sub_path == '/training-hub':
            return build_training_hub_layout(), config_dict
        elif sub_path == '/configure-training':
            return build_training_config_page(), config_dict
        else:  # Default to main dashboard
            return build_dashboard_layout(), config_dict
    
    # Fallback to home
    return build_home_layout(), {}
```

3. Register home callbacks:
```python
register_home_callbacks(app)
```

4. Add breadcrumb navigation to existing pages (in layouts.py):
```python
# Add to build_dashboard_layout() header:
html.Div(
    [
        dcc.Link("Home", href="/", style={"color": "#45717A", "fontSize": "14px"}),
        html.Span(" / ", style={"color": "#C6C6C6", "margin": "0 8px"}),
        html.Span("Model Dashboard", style={"color": "#4A4A4A", "fontSize": "14px"}),
    ],
    style={"padding": "8px 32px", "backgroundColor": "#f5f5f5"},
)
```

**Verification:**
```bash
poetry run python use_cases/dashboard/app.py
# Navigate to http://localhost:8050
# Should see home page
# Create a model → should navigate to /model/model_001
```

---

### Task 8: Update Existing Commands for Multi-Model

**File:** `use_cases/dashboard/commands.py`

Update all existing commands to work with `active_model` and persist metadata.

**Pattern to apply:**

```python
def execute(self, state: AppState) -> Tuple[AppState, str]:
    """Execute on active model."""
    if state.active_model is None:
        return state, "No model loaded"
    
    from use_cases.dashboard.model_manager import ModelManager
    from datetime import datetime
    
    # ... existing logic ...
    
    # Update active model
    updated_model = state.active_model # ... apply changes
    
    # Update metadata
    updated_model = updated_model.with_updated_metadata(
        last_modified=datetime.utcnow().isoformat(),
        # ... other metadata fields as needed
    )
    
    # Save metadata to disk
    ModelManager.save_metadata(updated_model.metadata)
    
    # Update app state
    new_state = state.with_active_model(updated_model)
    return new_state, message
```

**Commands to update:**
1. `LabelSampleCommand` - update `labeled_count` in metadata
2. `StartTrainingCommand` - no metadata change needed
3. `CompleteTrainingCommand` - update `total_epochs`, `latest_loss` in metadata, save history
4. `SelectSampleCommand` - no metadata change needed
5. `ChangeColorModeCommand` - no metadata change needed

**Example for CompleteTrainingCommand:**

```python
def execute(self, state: AppState) -> Tuple[AppState, str]:
    """Update after training completes."""
    if state.active_model is None:
        return state, "No model loaded"
    
    from use_cases.dashboard.model_manager import ModelManager
    from datetime import datetime
    
    # ... existing prediction update logic ...
    
    # Get latest loss from history
    if self.history and len(self.history.val_loss) > 0:
        latest_loss = float(self.history.val_loss[-1])
    else:
        latest_loss = None
    
    # Update model with new predictions
    updated_model = state.active_model.with_training_complete(
        latent=self.latent,
        reconstructed=self.reconstructed,
        pred_classes=self.pred_classes,
        pred_certainty=self.pred_certainty,
        hover_metadata=self.hover_metadata
    )
    
    # Update metadata
    total_epochs = len(updated_model.history.epochs)
    updated_model = updated_model.with_updated_metadata(
        total_epochs=total_epochs,
        latest_loss=latest_loss,
        last_modified=datetime.utcnow().isoformat()
    )
    
    # Persist
    ModelManager.save_metadata(updated_model.metadata)
    ModelManager.save_history(updated_model.model_id, updated_model.history)
    
    # Update app state
    new_state = state.with_active_model(updated_model)
    return new_state, "Training complete"
```

**Verification:**
```bash
# After implementing, test full workflow:
# 1. Create model
# 2. Label samples → check metadata.labeled_count updates
# 3. Train → check metadata.total_epochs and latest_loss update
# 4. Check history.json persisted
```

---

## Final Testing Plan

After implementing all tasks, test the complete workflow:

1. **Start fresh:**
```bash
rm -rf artifacts/models/*  # Clear old data
poetry run python use_cases/dashboard/app.py
```

2. **Home page:**
- Should see empty state
- Click "Create Your First Model"
- Enter name "Baseline Test", select "Default MNIST"
- Should navigate to `/model/model_001`

3. **Label and train:**
- Click some points, label them (0-9)
- Set epochs to 5
- Click "Train Model"
- Watch terminal for epoch progress
- Check `artifacts/models/model_001/history.json` exists and updates

4. **Switch models:**
- Navigate back to home (`/`)
- Create second model "High Recon Test" with "High Reconstruction" preset
- Should see both models in home page
- Click "Open" on first model
- Should load first model's state (labels preserved)

5. **Delete model:**
- Go home
- Click "Delete" on second model
- Should remove from list
- Check `artifacts/models/model_002/` deleted

6. **Persistence:**
- Stop dashboard
- Restart dashboard
- Navigate to home
- Should see first model with correct stats (epochs, labels, loss)
- Open it
- Training history should load (loss curves populated)

---

## Architecture Notes for Agent

**Immutability:** All state models are frozen dataclasses. Never mutate fields directly. Always use `replace()` or `with_*()` helpers to create new instances.

**Thread safety:** All state access must be inside `with state_lock:`. Commands execute under lock via dispatcher.

**Persistence:** ModelManager handles all disk I/O. Keep it pure (no side effects in state models).

**Caching:** Figure cache is cleared when active model changes (`with_active_model()` does this automatically).

**Error handling:** Commands validate before executing. Return `(state, error_message)` for validation failures, not exceptions.

**File paths:** Never hardcode paths. Always use `ModelManager.model_dir(model_id)` and helpers.

**Metadata updates:** Whenever state changes, update metadata and persist via `ModelManager.save_metadata()`. This keeps home page accurate.

---

## What We're NOT Changing

- Existing callback structure (training, labeling, visualization)
- Command pattern infrastructure (Dispatcher, CommandHistoryEntry)
- UI styling (infoteam colors, layouts)
- SSVAE model or trainer implementations
- Training worker threading approach

These work well. We're extending, not rewriting.