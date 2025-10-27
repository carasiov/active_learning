# Dashboard State Architecture

## Overview

This document describes the state management architecture for the SSVAE Active Learning Dashboard. A migration from mutable dict-based state to immutable dataclasses was completed to improve type safety, thread safety, and maintainability.

## Previous Architecture (Pre-Migration)

### Structure
The dashboard previously used a single mutable dictionary to store all application state:

```python
app_state = {
    "model": SSVAE(),
    "trainer": InteractiveTrainer(),
    "config": SSVAEConfig(),
    "data": {
        "x_train": ndarray,
        "labels": ndarray,
        "latent": ndarray,
        # ... more nested dicts
    },
    "training": {
        "state": "IDLE",
        "thread": None,
        # ...
    },
    # ... more nested structures
}
```

### Issues
- **No type safety**: Dictionary access (`app_state["key"]`) provided no IDE autocomplete or type checking
- **Race conditions**: Multiple threads could modify nested dictionaries simultaneously
- **Unclear mutations**: No explicit API for state updates, leading to scattered mutations throughout callbacks
- **Versioning complexity**: Separate version counters for labels and latent data required manual synchronization

## Current Architecture (Post-Migration)

### Structure

State is now organized as a hierarchy of frozen dataclasses defined in `state_models.py`:

```python
@dataclass(frozen=True)
class AppState:
    model: SSVAE
    trainer: InteractiveTrainer
    config: SSVAEConfig
    data: DataState
    training: TrainingStatus
    ui: UIState
    cache: Dict[str, object]
    history: TrainingHistory
```

All nested state containers (`DataState`, `TrainingStatus`, `UIState`, `TrainingHistory`) are also frozen dataclasses.

### Key Components

#### State Models (`state_models.py`)
- `AppState`: Root container for all application state
- `DataState`: Training data, predictions, and metadata
- `TrainingStatus`: Training state machine (IDLE/QUEUED/RUNNING)
- `TrainingState`: Enum for training states
- `UIState`: User interface state (selected sample, color mode)
- `TrainingHistory`: Training metrics history

Each dataclass provides immutable update methods:
```python
def with_ui(self, **updates) -> AppState:
    """Returns new AppState with updated UI state."""
    return replace(self, ui=replace(self.ui, **updates))
```

#### State Management (`state.py`)

**Module-level variables:**
```python
app_state: Optional[AppState] = None
state_lock = threading.Lock()
_init_lock = threading.Lock()
```

**Initialization:**
```python
def initialize_model_and_data() -> None:
    """Thread-safe initialization using _init_lock.
    
    Only one thread performs initialization while others wait.
    Sets app_state atomically after all data is loaded.
    """
```

**Access Pattern:**
All modules import the state module and access through it:
```python
from use_cases.dashboard import state as dashboard_state

# Read
with dashboard_state.state_lock:
    config = dashboard_state.app_state.config
    
# Update
with dashboard_state.state_lock:
    dashboard_state.app_state = dashboard_state.app_state.with_ui(
        selected_sample=5
    )
```

### Thread Safety

**Initialization Lock (`_init_lock`)**
- Ensures only one thread initializes application state
- Other threads block and wait for initialization to complete
- Prevents race conditions during startup

**State Lock (`state_lock`)**
- Protects all reads and writes to `app_state`
- Must be held when accessing any state attributes
- Must be held when reassigning `app_state`

**Update Pattern:**
```python
with dashboard_state.state_lock:
    new_state = dashboard_state.app_state.with_training_complete(
        latent=latent,
        reconstructed=recon,
        pred_classes=pred_classes,
        pred_certainty=pred_certainty,
        hover_metadata=hover_metadata
    )
    dashboard_state.app_state = new_state
```

### Import Pattern

**Critical Implementation Detail:**

Python's import semantics require special handling. When you import a variable directly:
```python
from use_cases.dashboard.state import app_state  # WRONG
```
You get a reference to the object at import time (None). When `app_state` is later reassigned in `state.py`, your module still has the old reference.

**Correct pattern:**
```python
from use_cases.dashboard import state as dashboard_state  # CORRECT

# Access through module
value = dashboard_state.app_state.config
```

This ensures you always get the current value of the module-level variable.

## File Organization

### Core State Files
- `state_models.py`: Dataclass definitions and update methods
- `state.py`: State initialization, helper functions, locks

### Callback Files
All callback files follow the same pattern:
- `callbacks/training_callbacks.py`
- `callbacks/labeling_callbacks.py`
- `callbacks/visualization_callbacks.py`
- `callbacks/config_callbacks.py`

Each imports and accesses state via module reference.

### Layout Files
- `layouts.py`: Dashboard layout, uses module-based state access
- `app.py`: Application entry point, routing callbacks

## Migration Benefits

### Type Safety
- Full IDE autocomplete for all state attributes
- Type checker catches errors at development time
- Self-documenting state structure

### Thread Safety
- Immutability prevents accidental mutations
- Explicit locks make concurrency patterns clear
- Atomic updates prevent partial state changes

### Maintainability
- Clear update methods show all possible state transitions
- Unified version tracking (single `data.version` replaces separate counters)
- Reduced cognitive load when reading callback code

## Development Guidelines

### Reading State
```python
from use_cases.dashboard import state as dashboard_state

def my_callback():
    with dashboard_state.state_lock:
        # Read multiple values in one lock acquisition
        config = dashboard_state.app_state.config
        labels = dashboard_state.app_state.data.labels
        training_active = dashboard_state.app_state.training.is_active()
    
    # Process without holding lock
    result = process(labels)
    return result
```

### Updating State
```python
def my_callback():
    with dashboard_state.state_lock:
        # Create new state with updates
        dashboard_state.app_state = dashboard_state.app_state.with_ui(
            color_mode="predictions",
            selected_sample=10
        )
```

### Adding New State Fields

1. Add field to appropriate dataclass in `state_models.py`
2. Update initialization in `state.py` if needed
3. Add update method if complex logic required
4. Update all consumers to handle new field

### Common Patterns

**Conditional updates:**
```python
with dashboard_state.state_lock:
    current = dashboard_state.app_state
    if current.training.is_active():
        return no_update
    
    dashboard_state.app_state = current.with_training_queued(epochs)
```

**Multiple nested updates:**
```python
from dataclasses import replace

with dashboard_state.state_lock:
    dashboard_state.app_state = replace(
        dashboard_state.app_state,
        training=dashboard_state.app_state.training.with_running(worker),
        data=replace(dashboard_state.app_state.data, version=new_version)
    )
```

## Testing Considerations

- State initialization must complete before tests run
- Tests should not rely on mutable state between test cases
- Use `dashboard_state.app_state = None` to reset between tests
- Re-call `initialize_model_and_data()` for each test suite

## Performance Notes

- Dataclass creation is fast (minimal overhead vs dict)
- `replace()` performs shallow copy (shares unchanged nested objects)
- Lock contention is low due to short critical sections
- Version counter triggers reactive updates only when data changes

## Future Enhancements

Potential improvements to consider:

1. **Persistent configuration**: Save config to disk on changes
2. **State snapshots**: Serialize entire state for debugging/replay
3. **Event sourcing**: Log all state transitions for audit trail
4. **Optimistic locking**: Detect concurrent modifications with version checks
5. **Read-write lock**: Allow concurrent reads with exclusive writes

## Summary

The migration from mutable dicts to immutable dataclasses provides a robust foundation for the dashboard's state management. The architecture enforces thread safety through explicit locking, provides type safety through dataclasses, and makes state transitions explicit through update methods. Developers working with this codebase should follow the module-based import pattern and always acquire locks when accessing shared state.
