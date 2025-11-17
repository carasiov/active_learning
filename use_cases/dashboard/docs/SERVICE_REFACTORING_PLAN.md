# Dashboard Service Architecture Refactoring Plan

> **Goal:** Transform dashboard from global-state-based to service-based architecture, improving abstraction quality, robustness, and preparing for future scaling (process isolation, REST API).

## Table of Contents
- [Current State Analysis](#current-state-analysis)
- [Target Architecture](#target-architecture)
- [Migration Strategy](#migration-strategy)
- [Phase 1: Service Abstractions](#phase-1-service-abstractions)
- [Phase 2: Command-Service Integration](#phase-2-command-service-integration)
- [Phase 3: Dependency Injection](#phase-3-dependency-injection)
- [Phase 4: Testing Infrastructure](#phase-4-testing-infrastructure)
- [Phase 5: Process Isolation (Future)](#phase-5-process-isolation-future)
- [Phase 6: REST API Layer (Future)](#phase-6-rest-api-layer-future)
- [Success Criteria](#success-criteria)

---

## Current State Analysis

### Strengths ✅
1. **Command Pattern** - Well-implemented with `validate()` + `execute()` separation
2. **Immutable State Models** - `AppState`, `ModelState` use frozen dataclasses
3. **Layered Architecture** - Clear separation (UI → Commands → State → Persistence)
4. **Thread-Safe Dispatcher** - `CommandDispatcher` provides atomic execution

### Pain Points ❌
1. **Global Mutable State**
   - `dashboard_state.app_state: Optional[AppState] = None` (singleton)
   - Accessed from multiple modules (commands, callbacks, workers)
   - Hard to test in isolation (can't instantiate multiple states)

2. **Tight Coupling**
   - Commands import `from use_cases.dashboard.core import state as dashboard_state`
   - Training workers directly access `dashboard_state.app_state.active_model.trainer`
   - No abstraction boundary between UI and domain logic

3. **Duplicated Training Logic**
   - `train_worker()` in `training_callbacks.py`
   - `train_worker_hub()` in `training_hub_callbacks.py`
   - Nearly identical code, violates DRY

4. **Process Safety**
   - Threading-only (no process isolation)
   - Training crash can kill entire dashboard
   - GIL limits true parallelism

5. **Testing Challenges**
   - Global state makes unit tests difficult
   - Need to mock `dashboard_state` module
   - Can't run tests in parallel

---

## Target Architecture

### Service-Based Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         Dashboard UI                             │
│                  (Dash pages + callbacks)                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Command Dispatcher                            │
│              (Validates + Executes commands)                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
   │   Training  │  │    Model    │  │   Labeling  │
   │   Service   │  │   Service   │  │   Service   │
   └─────────────┘  └─────────────┘  └─────────────┘
           │               │               │
           └───────────────┼───────────────┘
                           ▼
                   ┌───────────────┐
                   │  Persistence  │
                   │    Layer      │
                   │ (ModelManager)│
                   └───────────────┘
```

### Key Principles
1. **Explicit Contracts** - Services define clear interfaces (Protocols)
2. **No Global State** - Services own their state
3. **Dependency Injection** - Services passed to commands/callbacks
4. **Swappable Implementations** - In-process → Multi-process → REST API
5. **Testable** - Can mock services for unit tests

---

## Migration Strategy

### Guiding Principles
- ✅ **Incremental** - Each phase independently valuable
- ✅ **Backward Compatible** - Dashboard keeps working during refactoring
- ✅ **Test-Driven** - Add tests before refactoring
- ✅ **Documented** - Update docs as we go

### Risk Mitigation
- Keep old code paths until new ones proven
- Feature flags for new vs old implementation
- Comprehensive integration tests
- Staged rollout (service by service)

---

## Phase 1: Service Abstractions

**Goal:** Extract service interfaces and initial implementations without changing existing behavior.

### 1.1 Create Service Protocols

**File:** `use_cases/dashboard/services/__init__.py`
```python
"""Service layer for dashboard operations."""
from .training_service import TrainingService, InProcessTrainingService
from .model_service import ModelService
from .labeling_service import LabelingService

__all__ = [
    "TrainingService",
    "InProcessTrainingService",
    "ModelService",
    "LabelingService",
]
```

### 1.2 Training Service

**File:** `use_cases/dashboard/services/training_service.py`
```python
"""Training execution service - abstracts how training runs."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, Dict, Any
from pathlib import Path
import threading
from queue import Queue

from rcmvae.application.runtime.interactive import InteractiveTrainer


@dataclass
class TrainingJob:
    """Specification for a training job."""
    job_id: str
    model_id: str
    trainer: InteractiveTrainer
    num_epochs: int
    x_train: Any  # numpy array
    labels: Any  # numpy array
    checkpoint_path: Path
    run_epoch_offset: int = 0


class JobStatus(Enum):
    """Training job lifecycle states."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobProgress:
    """Training job progress information."""
    status: JobStatus
    current_epoch: int
    total_epochs: int
    latest_metrics: Dict[str, float]
    error_message: Optional[str] = None


class TrainingService(ABC):
    """Abstract interface for training execution.

    Implementations:
    - InProcessTrainingService: Threading (current)
    - ProcessPoolTrainingService: Multiprocessing (Phase 5)
    - HTTPTrainingService: REST API client (Phase 6)
    """

    @abstractmethod
    def start_training(self, job: TrainingJob) -> str:
        """Start async training job.

        Args:
            job: Training job specification

        Returns:
            job_id: Unique identifier for tracking

        Raises:
            AlreadyTrainingError: If model already training
            ResourceError: If no workers available
        """
        pass

    @abstractmethod
    def get_progress(self, job_id: str) -> Optional[JobProgress]:
        """Get current training progress.

        Args:
            job_id: Job identifier

        Returns:
            JobProgress or None if job doesn't exist
        """
        pass

    @abstractmethod
    def stop_training(self, job_id: str) -> bool:
        """Request graceful training stop.

        Args:
            job_id: Job identifier

        Returns:
            True if stop signal sent, False if job not found
        """
        pass

    @abstractmethod
    def is_training(self, model_id: str) -> bool:
        """Check if model currently has active training job."""
        pass


class InProcessTrainingService(TrainingService):
    """Thread-based training service (current implementation)."""

    def __init__(self, metrics_queue: Queue):
        """Initialize service.

        Args:
            metrics_queue: Queue for pushing metrics to UI
        """
        self._metrics_queue = metrics_queue
        self._jobs: Dict[str, JobProgress] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._stop_flags: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def start_training(self, job: TrainingJob) -> str:
        """Start training in background thread."""
        with self._lock:
            # Check if model already training
            if self.is_training(job.model_id):
                raise AlreadyTrainingError(f"Model {job.model_id} already training")

            # Initialize job state
            self._jobs[job.job_id] = JobProgress(
                status=JobStatus.QUEUED,
                current_epoch=job.run_epoch_offset,
                total_epochs=job.run_epoch_offset + job.num_epochs,
                latest_metrics={},
            )

            # Create stop flag
            self._stop_flags[job.job_id] = threading.Event()

            # Start worker thread
            thread = threading.Thread(
                target=self._train_worker,
                args=(job,),
                daemon=True,
            )
            self._threads[job.job_id] = thread
            thread.start()

            return job.job_id

    def get_progress(self, job_id: str) -> Optional[JobProgress]:
        """Get job progress."""
        with self._lock:
            return self._jobs.get(job_id)

    def stop_training(self, job_id: str) -> bool:
        """Set stop flag for job."""
        with self._lock:
            if job_id in self._stop_flags:
                self._stop_flags[job_id].set()
                return True
            return False

    def is_training(self, model_id: str) -> bool:
        """Check if model has active job."""
        with self._lock:
            for job_id, progress in self._jobs.items():
                # Find job by model_id (need to store model_id in JobProgress)
                if progress.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                    # TODO: Store model_id in JobProgress for this check
                    return True  # Simplified for now
            return False

    def _train_worker(self, job: TrainingJob) -> None:
        """Background worker - executes training loop.

        This consolidates logic from train_worker() and train_worker_hub().
        """
        stop_flag = self._stop_flags[job.job_id]

        try:
            # Update status to RUNNING
            with self._lock:
                self._jobs[job.job_id].status = JobStatus.RUNNING

            # Push start message to UI
            self._metrics_queue.put({
                "type": "start",
                "job_id": job.job_id,
                "model_id": job.model_id,
            })

            # Configure trainer callbacks
            def on_epoch_end(epoch: int, metrics: Dict[str, float]):
                # Update job progress
                with self._lock:
                    if job.job_id in self._jobs:
                        self._jobs[job.job_id].current_epoch = epoch
                        self._jobs[job.job_id].latest_metrics = metrics

                # Push to UI metrics queue
                self._metrics_queue.put({
                    "type": "epoch",
                    "job_id": job.job_id,
                    "epoch": epoch,
                    **metrics,
                })

                # Check stop flag
                if stop_flag.is_set():
                    raise TrainingStoppedException("User requested stop")

            # Run training
            job.trainer.train(
                x_train=job.x_train,
                labels=job.labels,
                num_epochs=job.num_epochs,
                checkpoint_path=str(job.checkpoint_path),
                on_epoch_end=on_epoch_end,
            )

            # Mark complete
            with self._lock:
                self._jobs[job.job_id].status = JobStatus.COMPLETED

            self._metrics_queue.put({
                "type": "complete",
                "job_id": job.job_id,
            })

        except TrainingStoppedException:
            with self._lock:
                self._jobs[job.job_id].status = JobStatus.CANCELLED

            self._metrics_queue.put({
                "type": "cancelled",
                "job_id": job.job_id,
            })

        except Exception as e:
            with self._lock:
                self._jobs[job.job_id].status = JobStatus.FAILED
                self._jobs[job.job_id].error_message = str(e)

            self._metrics_queue.put({
                "type": "error",
                "job_id": job.job_id,
                "message": str(e),
            })

            # Log full traceback
            import traceback
            from use_cases.dashboard.core.logging_config import get_logger
            logger = get_logger("training_service")
            logger.error(
                "Training failed | job_id=%s model_id=%s",
                job.job_id,
                job.model_id,
                exc_info=True,
            )


class AlreadyTrainingError(Exception):
    """Raised when attempting to start training on already-training model."""
    pass


class TrainingStoppedException(Exception):
    """Raised when training is stopped by user request."""
    pass
```

### 1.3 Model Service

**File:** `use_cases/dashboard/services/model_service.py`
```python
"""Model lifecycle management service."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path
import threading

from rcmvae.application.runtime.interactive import InteractiveTrainer
from rcmvae.application.model_api import SSVAE
from rcmvae.domain.config import SSVAEConfig

from use_cases.dashboard.core.state_models import ModelState, ModelMetadata
from use_cases.dashboard.core.model_manager import ModelManager


@dataclass
class CreateModelRequest:
    """Request to create new model."""
    name: str
    config: SSVAEConfig
    dataset_total_samples: int
    dataset_seed: int


@dataclass
class LoadModelRequest:
    """Request to load existing model."""
    model_id: str


class ModelService:
    """Service for model CRUD operations.

    Manages:
    - Model creation and initialization
    - Loading models from disk
    - Model metadata updates
    - Model deletion
    """

    def __init__(self, model_manager: ModelManager):
        """Initialize service.

        Args:
            model_manager: Persistence layer for models
        """
        self._manager = model_manager
        self._lock = threading.Lock()

    def create_model(self, request: CreateModelRequest) -> ModelState:
        """Create and initialize new model.

        Args:
            request: Model creation specification

        Returns:
            Initialized ModelState

        Raises:
            ValidationError: If config invalid
        """
        with self._lock:
            # Generate model ID
            model_id = self._manager.generate_model_id()

            # Create directory
            self._manager.create_model_directory(model_id)

            # Initialize metadata
            from datetime import datetime
            metadata = ModelMetadata(
                model_id=model_id,
                name=request.name,
                created_at=datetime.utcnow().isoformat(),
                last_modified=datetime.utcnow().isoformat(),
                dataset="mnist",
                total_epochs=0,
                labeled_count=0,
                latest_loss=None,
                dataset_total_samples=request.dataset_total_samples,
                dataset_seed=request.dataset_seed,
            )

            # Save metadata
            self._manager.save_metadata(metadata)

            # Save config
            self._manager.save_config(model_id, request.config)

            # Initialize model and trainer
            # (This part comes from CreateModelCommand logic)
            # TODO: Extract model initialization logic

            return model_state  # Return initialized ModelState

    def load_model(self, request: LoadModelRequest) -> Optional[ModelState]:
        """Load model from disk.

        Args:
            request: Load specification

        Returns:
            ModelState or None if not found
        """
        with self._lock:
            # Load metadata
            metadata = self._manager.load_metadata(request.model_id)
            if metadata is None:
                return None

            # Load config
            config = self._manager.load_config(request.model_id)
            if config is None:
                return None

            # Load history
            history = self._manager.load_history(request.model_id)

            # Initialize model and trainer
            # (This part comes from LoadModelCommand logic)
            # TODO: Extract model initialization logic

            return model_state  # Return loaded ModelState

    def delete_model(self, model_id: str) -> bool:
        """Delete model from disk.

        Args:
            model_id: Model identifier

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            model_dir = self._manager.model_dir(model_id)
            if not model_dir.exists():
                return False

            import shutil
            shutil.rmtree(model_dir)
            return True

    def list_models(self) -> List[ModelMetadata]:
        """List all models.

        Returns:
            List of model metadata
        """
        with self._lock:
            models = []
            for model_dir in ModelManager.MODELS_DIR.iterdir():
                if model_dir.is_dir():
                    metadata = self._manager.load_metadata(model_dir.name)
                    if metadata:
                        models.append(metadata)

            # Sort by last_modified descending
            models.sort(key=lambda m: m.last_modified, reverse=True)
            return models

    def update_metadata(self, metadata: ModelMetadata) -> None:
        """Update model metadata.

        Args:
            metadata: Updated metadata
        """
        with self._lock:
            self._manager.save_metadata(metadata)
```

### 1.4 Labeling Service

**File:** `use_cases/dashboard/services/labeling_service.py`
```python
"""Labeling operations service."""

from typing import Optional, Tuple
import threading
import numpy as np
import pandas as pd

from use_cases.dashboard.core.model_manager import ModelManager


class LabelingService:
    """Service for managing sample labels.

    Manages:
    - Label assignment/deletion
    - Label persistence (CSV)
    - Label statistics
    """

    def __init__(self, model_manager: ModelManager):
        """Initialize service.

        Args:
            model_manager: Persistence layer
        """
        self._manager = model_manager
        self._lock = threading.Lock()

    def set_label(
        self,
        model_id: str,
        sample_idx: int,
        label: Optional[int],
    ) -> bool:
        """Set or delete label for sample.

        Args:
            model_id: Model identifier
            sample_idx: Sample index
            label: Label value (None to delete)

        Returns:
            True if successful

        Raises:
            ValueError: If label invalid
        """
        with self._lock:
            # Validate label
            if label is not None:
                if not isinstance(label, int) or not (0 <= label <= 9):
                    raise ValueError(f"Invalid label: {label}")

            # Load labels CSV
            labels_path = self._manager.model_dir(model_id) / "labels.csv"
            if labels_path.exists():
                df = pd.read_csv(labels_path, index_col=0)
            else:
                df = pd.DataFrame(columns=["label"])

            # Update
            if label is None:
                if sample_idx in df.index:
                    df = df.drop(sample_idx)
            else:
                df.loc[sample_idx, "label"] = label

            # Save
            df.to_csv(labels_path)
            return True

    def get_label(self, model_id: str, sample_idx: int) -> Optional[int]:
        """Get label for sample.

        Args:
            model_id: Model identifier
            sample_idx: Sample index

        Returns:
            Label value or None if unlabeled
        """
        with self._lock:
            labels_path = self._manager.model_dir(model_id) / "labels.csv"
            if not labels_path.exists():
                return None

            df = pd.read_csv(labels_path, index_col=0)
            if sample_idx not in df.index:
                return None

            return int(df.loc[sample_idx, "label"])

    def get_labeled_count(self, model_id: str) -> int:
        """Get total number of labeled samples.

        Args:
            model_id: Model identifier

        Returns:
            Count of labeled samples
        """
        with self._lock:
            labels_path = self._manager.model_dir(model_id) / "labels.csv"
            if not labels_path.exists():
                return 0

            df = pd.read_csv(labels_path, index_col=0)
            return len(df)
```

### 1.5 Service Container

**File:** `use_cases/dashboard/services/container.py`
```python
"""Service container for dependency injection."""

from dataclasses import dataclass
from queue import Queue

from use_cases.dashboard.services.training_service import TrainingService, InProcessTrainingService
from use_cases.dashboard.services.model_service import ModelService
from use_cases.dashboard.services.labeling_service import LabelingService
from use_cases.dashboard.core.model_manager import ModelManager


@dataclass
class ServiceContainer:
    """Container holding all service instances.

    This enables dependency injection and makes testing easier.
    """
    training: TrainingService
    model: ModelService
    labeling: LabelingService

    @classmethod
    def create_default(cls, metrics_queue: Queue) -> "ServiceContainer":
        """Create container with default (in-process) implementations.

        Args:
            metrics_queue: Queue for training metrics

        Returns:
            ServiceContainer with all services initialized
        """
        model_manager = ModelManager()

        return cls(
            training=InProcessTrainingService(metrics_queue),
            model=ModelService(model_manager),
            labeling=LabelingService(model_manager),
        )
```

### 1.6 Deliverables

**Files to create:**
- [ ] `use_cases/dashboard/services/__init__.py`
- [ ] `use_cases/dashboard/services/training_service.py`
- [ ] `use_cases/dashboard/services/model_service.py`
- [ ] `use_cases/dashboard/services/labeling_service.py`
- [ ] `use_cases/dashboard/services/container.py`

**Tests to add:**
- [ ] `tests/dashboard/services/test_training_service.py`
- [ ] `tests/dashboard/services/test_model_service.py`
- [ ] `tests/dashboard/services/test_labeling_service.py`

**Success criteria:**
- [ ] All service interfaces compile
- [ ] `InProcessTrainingService` matches current `train_worker()` behavior
- [ ] Unit tests pass for each service
- [ ] No changes to existing dashboard code yet (services not used)

---

## Phase 2: Command-Service Integration

**Goal:** Refactor commands to use services instead of direct global state access.

### 2.1 Update Command Base Class

**File:** `use_cases/dashboard/core/commands.py`

Add service dependency to Command protocol:
```python
class Command(ABC):
    """Base class for all state-modifying commands."""

    @abstractmethod
    def validate(self, state: AppState, services: ServiceContainer) -> Optional[str]:
        """Validate if command can execute.

        Args:
            state: Current application state (read-only)
            services: Service container for domain operations

        Returns:
            Error message if invalid, None if valid
        """
        pass

    @abstractmethod
    def execute(
        self,
        state: AppState,
        services: ServiceContainer,
    ) -> Tuple[AppState, str]:
        """Execute command, producing new state.

        Args:
            state: Current application state (read-only)
            services: Service container for domain operations

        Returns:
            (new_state, status_message)
        """
        pass
```

### 2.2 Update CommandDispatcher

```python
class CommandDispatcher:
    """Central dispatcher for all state-modifying commands."""

    def __init__(
        self,
        state_lock: threading.RLock,
        services: ServiceContainer,
    ):
        """Initialize dispatcher.

        Args:
            state_lock: Lock protecting app_state
            services: Service container
        """
        self._state_lock = state_lock
        self._services = services  # NEW
        self._command_history: List[CommandHistoryEntry] = []
        self._history_lock = threading.Lock()

    def execute(self, command: Command) -> Tuple[bool, str]:
        """Execute command atomically."""
        from use_cases.dashboard.core import state as dashboard_state

        with self._state_lock:
            if dashboard_state.app_state is None:
                error_msg = "Application state not initialized"
                self._log_command(command, success=False, message=error_msg)
                return False, error_msg

            # Validate (now with services)
            error = command.validate(dashboard_state.app_state, self._services)
            if error:
                self._log_command(command, success=False, message=error)
                return False, error

            # Execute (now with services)
            try:
                new_state, message = command.execute(
                    dashboard_state.app_state,
                    self._services,
                )
                dashboard_state.app_state = new_state
                self._log_command(command, success=True, message=message)
                return True, message

            except Exception as e:
                error_msg = f"Command execution failed: {e}"
                self._log_command(command, success=False, message=error_msg)
                return False, error_msg
```

### 2.3 Refactor StartTrainingCommand

**Before:**
```python
@dataclass
class StartTrainingCommand(Command):
    num_epochs: int
    recon_weight: float
    kl_weight: float
    learning_rate: float

    def execute(self, state: AppState) -> Tuple[AppState, str]:
        # Updates config, sets state to QUEUED
        # Training started later by callback via train_worker()
        ...
```

**After:**
```python
@dataclass
class StartTrainingCommand(Command):
    num_epochs: int
    recon_weight: float
    kl_weight: float
    learning_rate: float

    def validate(
        self,
        state: AppState,
        services: ServiceContainer,
    ) -> Optional[str]:
        """Validate training can start."""
        if state.active_model is None:
            return "No model loaded"

        # Check if already training via service
        if services.training.is_training(state.active_model.model_id):
            return "Training already in progress"

        # Validate parameters...
        return None

    def execute(
        self,
        state: AppState,
        services: ServiceContainer,
    ) -> Tuple[AppState, str]:
        """Start training via service."""
        if state.active_model is None:
            return state, "No model loaded"

        # Update config with new hyperparameters
        updated_config = replace(
            state.active_model.model.config,
            reconstruction_loss_weight=self.recon_weight,
            kl_divergence_weight=self.kl_weight,
            learning_rate=self.learning_rate,
        )

        # Reinitialize trainer with updated config
        # (extract this logic into ModelService.update_trainer())

        # Create training job
        from use_cases.dashboard.services.training_service import TrainingJob
        import uuid

        job = TrainingJob(
            job_id=uuid.uuid4().hex,
            model_id=state.active_model.model_id,
            trainer=state.active_model.trainer,  # Updated trainer
            num_epochs=self.num_epochs,
            x_train=state.active_model.data.x_train,
            labels=state.active_model.data.labels,
            checkpoint_path=ModelManager.checkpoint_path(state.active_model.model_id),
            run_epoch_offset=len(state.active_model.history.epochs),
        )

        # Start training via service
        try:
            job_id = services.training.start_training(job)
        except AlreadyTrainingError as e:
            return state, str(e)

        # Update state to RUNNING
        updated_model = replace(
            state.active_model,
            training=replace(
                state.active_model.training,
                state=TrainingState.RUNNING,
                target_epochs=self.num_epochs,
            ),
        )

        new_state = state.with_active_model(updated_model)
        return new_state, f"Training started (job: {job_id})"
```

### 2.4 Refactor LabelSampleCommand

**After:**
```python
@dataclass
class LabelSampleCommand(Command):
    sample_idx: int
    label: Optional[int]

    def execute(
        self,
        state: AppState,
        services: ServiceContainer,
    ) -> Tuple[AppState, str]:
        """Execute label update via service."""
        if state.active_model is None:
            return state, "No model loaded"

        # Update via service (handles persistence)
        try:
            services.labeling.set_label(
                model_id=state.active_model.model_id,
                sample_idx=self.sample_idx,
                label=self.label,
            )
        except ValueError as e:
            return state, f"Label update failed: {e}"

        # Update in-memory state
        labels_array = state.active_model.data.labels.copy()
        if self.label is None:
            labels_array[self.sample_idx] = np.nan
        else:
            labels_array[self.sample_idx] = float(self.label)

        # Update hover metadata
        # ... (same as before)

        # Get updated labeled count from service
        labeled_count = services.labeling.get_labeled_count(
            state.active_model.model_id
        )

        # Update model state
        updated_model = state.active_model.with_label_update(
            labels_array,
            hover_metadata,
        ).with_updated_metadata(labeled_count=labeled_count)

        new_state = state.with_active_model(updated_model)
        return new_state, "Label updated"
```

### 2.5 Deliverables

**Files to update:**
- [ ] `use_cases/dashboard/core/commands.py` - All command classes
- [ ] Update each command to accept `services` parameter:
  - [ ] `LabelSampleCommand`
  - [ ] `StartTrainingCommand`
  - [ ] `CompleteTrainingCommand`
  - [ ] `StopTrainingCommand`
  - [ ] `CreateModelCommand`
  - [ ] `LoadModelCommand`
  - [ ] `DeleteModelCommand`
  - [ ] `UpdateModelConfigCommand`
  - [ ] Others as needed

**Tests to update:**
- [ ] All command unit tests to provide mock services

**Success criteria:**
- [ ] All commands compile with new signature
- [ ] Command tests pass with mock services
- [ ] Commands use services for domain operations (not direct state manipulation)

---

## Phase 3: Dependency Injection

**Goal:** Remove global state dependency, inject services into dispatcher.

### 3.1 Update app.py Initialization

**File:** `use_cases/dashboard/app.py`

**Before:**
```python
# Global imports trigger state initialization
from use_cases.dashboard.core import state as dashboard_state
from use_cases.dashboard.core.commands import dispatcher

# State is global module-level variable
dashboard_state.initialize_state()
```

**After:**
```python
from queue import Queue
from use_cases.dashboard.core.state import AppStateManager
from use_cases.dashboard.services.container import ServiceContainer
from use_cases.dashboard.core.commands import CommandDispatcher

# Create metrics queue
metrics_queue = Queue()

# Create services
services = ServiceContainer.create_default(metrics_queue)

# Create state manager (replaces global state)
state_manager = AppStateManager()

# Create dispatcher with services
dispatcher = CommandDispatcher(
    state_lock=state_manager.lock,
    services=services,
)

# Make available to app
app.server.state_manager = state_manager
app.server.dispatcher = dispatcher
app.server.services = services
app.server.metrics_queue = metrics_queue
```

### 3.2 Create AppStateManager

**File:** `use_cases/dashboard/core/state.py`

**Replace global state with manager:**
```python
class AppStateManager:
    """Manages application state without global variables.

    This replaces the module-level `app_state` singleton.
    """

    def __init__(self):
        """Initialize state manager."""
        self._state: Optional[AppState] = None
        self.lock = threading.RLock()

    def get_state(self) -> Optional[AppState]:
        """Get current state (thread-safe)."""
        with self.lock:
            return self._state

    def set_state(self, state: AppState) -> None:
        """Set state (thread-safe)."""
        with self.lock:
            self._state = state

    def initialize(self) -> None:
        """Initialize state to default."""
        with self.lock:
            self._state = AppState(
                models={},
                active_model=None,
                ui=UIState.default(),
            )

    def update(self, updater: Callable[[AppState], AppState]) -> None:
        """Update state via function (atomic).

        Args:
            updater: Function that takes old state, returns new state
        """
        with self.lock:
            if self._state is not None:
                self._state = updater(self._state)
```

### 3.3 Update Callbacks to Use Injected Services

**File:** `use_cases/dashboard/callbacks/training_callbacks.py`

**Before:**
```python
import dash
from dash import callback

from use_cases.dashboard.core import state as dashboard_state
from use_cases.dashboard.core.commands import StartTrainingCommand, dispatcher

@callback(...)
def start_training_callback(n_clicks, num_epochs):
    # Uses global dashboard_state and dispatcher
    cmd = StartTrainingCommand(num_epochs=num_epochs, ...)
    success, message = dispatcher.execute(cmd)
    return message
```

**After:**
```python
import dash
from dash import callback
from flask import current_app

from use_cases.dashboard.core.commands import StartTrainingCommand

@callback(...)
def start_training_callback(n_clicks, num_epochs):
    # Get dispatcher from app context
    dispatcher = current_app.server.dispatcher

    cmd = StartTrainingCommand(num_epochs=num_epochs, ...)
    success, message = dispatcher.execute(cmd)
    return message
```

### 3.4 Remove Global State Variables

**File:** `use_cases/dashboard/core/state.py`

**Delete:**
```python
# OLD - REMOVE
app_state: Optional[AppState] = None
state_lock = threading.RLock()
dispatcher = CommandDispatcher(state_lock)
```

**Keep only:**
```python
# NEW - Service-based
class AppStateManager:
    """Manages state without globals."""
    ...
```

### 3.5 Deliverables

**Files to update:**
- [ ] `use_cases/dashboard/app.py` - Initialize services and state manager
- [ ] `use_cases/dashboard/core/state.py` - Add `AppStateManager`, remove globals
- [ ] All callback files - Get dispatcher/services from `current_app.server`:
  - [ ] `callbacks/training_callbacks.py`
  - [ ] `callbacks/training_hub_callbacks.py`
  - [ ] `callbacks/labeling_callbacks.py`
  - [ ] `callbacks/home_callbacks.py`
  - [ ] `callbacks/config_callbacks.py`
  - [ ] Others as needed

**Tests to update:**
- [ ] All integration tests - Provide test state manager and services

**Success criteria:**
- [ ] No global state variables remain
- [ ] Services injected via `current_app.server`
- [ ] All callbacks work with injected dependencies
- [ ] Dashboard starts and runs correctly
- [ ] All tests pass

---

## Phase 4: Testing Infrastructure

**Goal:** Comprehensive test coverage with mockable services.

### 4.1 Service Mocks

**File:** `tests/dashboard/services/mocks.py`
```python
"""Mock service implementations for testing."""

from typing import Dict, Optional
from unittest.mock import Mock

from use_cases.dashboard.services.training_service import (
    TrainingService,
    JobStatus,
    JobProgress,
    TrainingJob,
)
from use_cases.dashboard.services.model_service import ModelService
from use_cases.dashboard.services.labeling_service import LabelingService


class MockTrainingService(TrainingService):
    """In-memory training service for testing."""

    def __init__(self):
        self._jobs: Dict[str, JobProgress] = {}
        self._start_count = 0
        self._stop_count = 0

    def start_training(self, job: TrainingJob) -> str:
        self._start_count += 1
        self._jobs[job.job_id] = JobProgress(
            status=JobStatus.RUNNING,
            current_epoch=0,
            total_epochs=job.num_epochs,
            latest_metrics={},
        )
        return job.job_id

    def get_progress(self, job_id: str) -> Optional[JobProgress]:
        return self._jobs.get(job_id)

    def stop_training(self, job_id: str) -> bool:
        if job_id in self._jobs:
            self._stop_count += 1
            self._jobs[job_id].status = JobStatus.CANCELLED
            return True
        return False

    def is_training(self, model_id: str) -> bool:
        return any(
            p.status in {JobStatus.QUEUED, JobStatus.RUNNING}
            for p in self._jobs.values()
        )


class MockModelService(ModelService):
    """In-memory model service for testing."""

    def __init__(self):
        self._models: Dict[str, ModelState] = {}

    # Implement mock versions of all methods
    ...
```

### 4.2 Command Tests with Mocks

**File:** `tests/dashboard/commands/test_start_training_command.py`
```python
"""Test StartTrainingCommand with mock services."""

import pytest
from use_cases.dashboard.core.commands import StartTrainingCommand
from use_cases.dashboard.services.container import ServiceContainer
from tests.dashboard.services.mocks import (
    MockTrainingService,
    MockModelService,
    MockLabelingService,
)


@pytest.fixture
def mock_services():
    """Create mock service container."""
    return ServiceContainer(
        training=MockTrainingService(),
        model=MockModelService(),
        labeling=MockLabelingService(),
    )


def test_start_training_success(mock_services, test_app_state):
    """Test successful training start."""
    cmd = StartTrainingCommand(
        num_epochs=5,
        recon_weight=1.0,
        kl_weight=1.0,
        learning_rate=0.001,
    )

    # Validate
    error = cmd.validate(test_app_state, mock_services)
    assert error is None

    # Execute
    new_state, message = cmd.execute(test_app_state, mock_services)

    # Verify
    assert "started" in message.lower()
    assert new_state.active_model.training.state == TrainingState.RUNNING
    assert mock_services.training._start_count == 1


def test_start_training_already_running(mock_services, test_app_state):
    """Test error when already training."""
    # Set up: model already training
    mock_services.training._jobs["existing"] = JobProgress(
        status=JobStatus.RUNNING,
        current_epoch=1,
        total_epochs=10,
        latest_metrics={},
    )

    cmd = StartTrainingCommand(num_epochs=5, ...)

    # Should fail validation
    error = cmd.validate(test_app_state, mock_services)
    assert error is not None
    assert "already" in error.lower()
```

### 4.3 Integration Tests

**File:** `tests/dashboard/test_integration.py`
```python
"""Integration tests for dashboard with services."""

import pytest
from queue import Queue

from use_cases.dashboard.core.state import AppStateManager
from use_cases.dashboard.services.container import ServiceContainer
from use_cases.dashboard.core.commands import (
    CommandDispatcher,
    CreateModelCommand,
    StartTrainingCommand,
)


@pytest.fixture
def integration_env():
    """Set up full integration environment."""
    metrics_queue = Queue()
    services = ServiceContainer.create_default(metrics_queue)
    state_manager = AppStateManager()
    state_manager.initialize()

    dispatcher = CommandDispatcher(
        state_lock=state_manager.lock,
        services=services,
    )

    return {
        "state_manager": state_manager,
        "services": services,
        "dispatcher": dispatcher,
        "metrics_queue": metrics_queue,
    }


def test_full_workflow(integration_env):
    """Test: Create model → Start training → Stop training."""
    env = integration_env
    dispatcher = env["dispatcher"]

    # 1. Create model
    create_cmd = CreateModelCommand(
        name="Test Model",
        # ... config
    )
    success, msg = dispatcher.execute(create_cmd)
    assert success

    # 2. Start training
    train_cmd = StartTrainingCommand(num_epochs=2, ...)
    success, msg = dispatcher.execute(train_cmd)
    assert success

    # 3. Check training started
    state = env["state_manager"].get_state()
    assert state.active_model.training.state == TrainingState.RUNNING

    # 4. Stop training
    stop_cmd = StopTrainingCommand()
    success, msg = dispatcher.execute(stop_cmd)
    assert success
```

### 4.4 Deliverables

**Files to create:**
- [ ] `tests/dashboard/services/mocks.py`
- [ ] `tests/dashboard/services/test_training_service.py`
- [ ] `tests/dashboard/services/test_model_service.py`
- [ ] `tests/dashboard/services/test_labeling_service.py`
- [ ] `tests/dashboard/commands/test_*.py` - One per command
- [ ] `tests/dashboard/test_integration.py`

**Success criteria:**
- [ ] 80%+ code coverage on services
- [ ] All commands have unit tests with mocks
- [ ] Integration tests cover main workflows
- [ ] Tests run in parallel (no global state conflicts)

---

## Phase 5: Process Isolation (Future)

**Goal:** Enable true process isolation for training (prevents dashboard crashes).

### 5.1 ProcessPoolTrainingService

**File:** `use_cases/dashboard/services/training_service.py`

Add new implementation:
```python
import multiprocessing as mp
from multiprocessing import Pool, Manager
from typing import Dict

class ProcessPoolTrainingService(TrainingService):
    """Multi-process training service with process isolation.

    Benefits:
    - Training crash doesn't kill dashboard
    - True parallelism (no GIL)
    - Resource limits per process

    Challenges:
    - Must serialize job state
    - Can't pass callbacks (need IPC)
    - More complex error handling
    """

    def __init__(self, metrics_queue: Queue, max_workers: int = 1):
        """Initialize process pool.

        Args:
            metrics_queue: Queue for metrics (must be multiprocessing.Queue)
            max_workers: Max concurrent training jobs (usually 1 for GPU)
        """
        self._metrics_queue = metrics_queue
        self._pool = Pool(processes=max_workers)
        self._manager = Manager()
        self._jobs = self._manager.dict()  # Shared dict across processes
        self._lock = mp.Lock()

    def start_training(self, job: TrainingJob) -> str:
        """Start training in worker process."""
        with self._lock:
            # Check already training
            if self.is_training(job.model_id):
                raise AlreadyTrainingError(f"Model {job.model_id} already training")

            # Initialize job state
            self._jobs[job.job_id] = {
                "status": JobStatus.QUEUED.value,
                "current_epoch": 0,
                "total_epochs": job.num_epochs,
                "model_id": job.model_id,
            }

            # Submit to pool
            async_result = self._pool.apply_async(
                _train_worker_process,  # Top-level function (picklable)
                args=(job, self._metrics_queue),
                callback=self._on_complete,
                error_callback=self._on_error,
            )

            return job.job_id

    def _on_complete(self, result):
        """Callback when training completes."""
        job_id, final_metrics = result
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = JobStatus.COMPLETED.value

    def _on_error(self, error):
        """Callback when training fails."""
        # Log error, update job status
        ...


def _train_worker_process(job: TrainingJob, metrics_queue) -> Tuple[str, Dict]:
    """Worker function for process pool (must be top-level for pickle).

    This is the entry point for each training process.
    """
    # Set up JAX in worker process
    import jax
    jax.config.update('jax_platform_name', 'cpu')  # Or 'gpu'

    # Run training
    # ... (similar to InProcessTrainingService._train_worker)

    return job.job_id, final_metrics
```

### 5.2 Serialization Challenges

**Problem:** Can't pickle JAX/Flax objects directly

**Solution:** Serialize to checkpoint format
```python
@dataclass
class SerializableTrainingJob:
    """Training job specification (serializable)."""
    job_id: str
    model_id: str
    config_dict: Dict[str, Any]  # SSVAEConfig as dict
    checkpoint_path: str
    num_epochs: int
    x_train: np.ndarray  # NumPy arrays are picklable
    labels: np.ndarray

    @classmethod
    def from_training_job(cls, job: TrainingJob) -> "SerializableTrainingJob":
        """Convert TrainingJob to serializable form."""
        # Save trainer checkpoint
        job.trainer.save_checkpoint(job.checkpoint_path)

        return cls(
            job_id=job.job_id,
            model_id=job.model_id,
            config_dict=asdict(job.trainer.config),
            checkpoint_path=str(job.checkpoint_path),
            num_epochs=job.num_epochs,
            x_train=job.x_train,
            labels=job.labels,
        )


def _train_worker_process(
    serializable_job: SerializableTrainingJob,
    metrics_queue,
) -> Tuple[str, Dict]:
    """Worker that reconstructs trainer from checkpoint."""
    # Reconstruct config
    config = SSVAEConfig(**serializable_job.config_dict)

    # Load trainer from checkpoint
    trainer = InteractiveTrainer.from_checkpoint(
        config,
        serializable_job.checkpoint_path,
    )

    # Run training
    # ...
```

### 5.3 Deliverables

**Files to update:**
- [ ] `use_cases/dashboard/services/training_service.py` - Add `ProcessPoolTrainingService`
- [ ] `use_cases/dashboard/services/container.py` - Add factory for process-based container

**Configuration:**
- [ ] Add environment variable: `DASHBOARD_TRAINING_MODE=process|thread`
- [ ] Default: `thread` (current), opt-in: `process`

**Success criteria:**
- [ ] Can start dashboard with `DASHBOARD_TRAINING_MODE=process`
- [ ] Training runs in separate process
- [ ] Training crash doesn't kill dashboard
- [ ] Metrics still flow to UI

---

## Phase 6: REST API Layer (Future)

**Goal:** Optional REST API for external integrations or multi-user scenarios.

### 6.1 FastAPI Server

**File:** `use_cases/dashboard/api/server.py`
```python
"""REST API server for dashboard services."""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List

from use_cases.dashboard.services.container import ServiceContainer
from use_cases.dashboard.services.training_service import TrainingJob, JobProgress


app = FastAPI(title="SSVAE Dashboard API")

# Dependency injection
def get_services() -> ServiceContainer:
    """Get service container (injected by app startup)."""
    return app.state.services


# ============================================================================
# Request/Response Models
# ============================================================================

class StartTrainingRequest(BaseModel):
    """Request to start training."""
    model_id: str
    num_epochs: int
    recon_weight: float = 1.0
    kl_weight: float = 1.0
    learning_rate: float = 0.001


class StartTrainingResponse(BaseModel):
    """Response with job ID."""
    job_id: str
    message: str


class JobStatusResponse(BaseModel):
    """Training job status."""
    job_id: str
    status: str
    current_epoch: int
    total_epochs: int
    latest_metrics: dict


# ============================================================================
# Training Endpoints
# ============================================================================

@app.post("/api/training/start", response_model=StartTrainingResponse)
async def start_training(
    request: StartTrainingRequest,
    services: ServiceContainer = Depends(get_services),
):
    """Start training job.

    POST /api/training/start
    {
        "model_id": "model_001",
        "num_epochs": 10,
        "recon_weight": 1.0,
        "kl_weight": 1.0,
        "learning_rate": 0.001
    }

    Response:
    {
        "job_id": "abc123",
        "message": "Training started"
    }
    """
    # Load model
    model_state = services.model.load_model(
        LoadModelRequest(model_id=request.model_id)
    )
    if model_state is None:
        raise HTTPException(status_code=404, detail="Model not found")

    # Create job
    import uuid
    job = TrainingJob(
        job_id=uuid.uuid4().hex,
        model_id=request.model_id,
        trainer=model_state.trainer,
        num_epochs=request.num_epochs,
        x_train=model_state.data.x_train,
        labels=model_state.data.labels,
        checkpoint_path=ModelManager.checkpoint_path(request.model_id),
    )

    # Start via service
    try:
        job_id = services.training.start_training(job)
        return StartTrainingResponse(
            job_id=job_id,
            message="Training started",
        )
    except AlreadyTrainingError:
        raise HTTPException(status_code=409, detail="Model already training")


@app.get("/api/training/status/{job_id}", response_model=JobStatusResponse)
async def get_training_status(
    job_id: str,
    services: ServiceContainer = Depends(get_services),
):
    """Get training job status.

    GET /api/training/status/abc123

    Response:
    {
        "job_id": "abc123",
        "status": "running",
        "current_epoch": 5,
        "total_epochs": 10,
        "latest_metrics": {"loss": 0.42}
    }
    """
    progress = services.training.get_progress(job_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job_id,
        status=progress.status.value,
        current_epoch=progress.current_epoch,
        total_epochs=progress.total_epochs,
        latest_metrics=progress.latest_metrics,
    )


@app.post("/api/training/stop/{job_id}")
async def stop_training(
    job_id: str,
    services: ServiceContainer = Depends(get_services),
):
    """Stop training job.

    POST /api/training/stop/abc123

    Response:
    {
        "message": "Training stopped"
    }
    """
    success = services.training.stop_training(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"message": "Training stopped"}


# ============================================================================
# Model Endpoints
# ============================================================================

@app.get("/api/models", response_model=List[dict])
async def list_models(
    services: ServiceContainer = Depends(get_services),
):
    """List all models."""
    models = services.model.list_models()
    return [m.to_dict() for m in models]


@app.post("/api/models")
async def create_model(
    request: CreateModelRequest,
    services: ServiceContainer = Depends(get_services),
):
    """Create new model."""
    model_state = services.model.create_model(request)
    return {"model_id": model_state.model_id}


@app.delete("/api/models/{model_id}")
async def delete_model(
    model_id: str,
    services: ServiceContainer = Depends(get_services),
):
    """Delete model."""
    success = services.model.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")

    return {"message": "Model deleted"}


# ============================================================================
# Labeling Endpoints
# ============================================================================

@app.put("/api/labels/{model_id}/{sample_idx}")
async def set_label(
    model_id: str,
    sample_idx: int,
    label: Optional[int],
    services: ServiceContainer = Depends(get_services),
):
    """Set or delete label for sample."""
    try:
        services.labeling.set_label(model_id, sample_idx, label)
        return {"message": "Label updated"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### 6.2 HTTP Training Service Client

**File:** `use_cases/dashboard/services/training_service.py`

Add HTTP client implementation:
```python
import httpx

class HTTPTrainingService(TrainingService):
    """Training service client that talks to REST API.

    Use this when dashboard UI and training server are separate processes.
    """

    def __init__(self, base_url: str):
        """Initialize HTTP client.

        Args:
            base_url: API server URL (e.g., "http://localhost:8000")
        """
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30.0)

    def start_training(self, job: TrainingJob) -> str:
        """Start training via HTTP POST."""
        # Serialize job to API request
        request = {
            "model_id": job.model_id,
            "num_epochs": job.num_epochs,
            # ... other fields
        }

        response = self._client.post(
            f"{self._base_url}/api/training/start",
            json=request,
        )

        if response.status_code == 409:
            raise AlreadyTrainingError("Model already training")

        response.raise_for_status()
        data = response.json()
        return data["job_id"]

    def get_progress(self, job_id: str) -> Optional[JobProgress]:
        """Get progress via HTTP GET."""
        response = self._client.get(
            f"{self._base_url}/api/training/status/{job_id}"
        )

        if response.status_code == 404:
            return None

        response.raise_for_status()
        data = response.json()

        return JobProgress(
            status=JobStatus(data["status"]),
            current_epoch=data["current_epoch"],
            total_epochs=data["total_epochs"],
            latest_metrics=data["latest_metrics"],
        )

    def stop_training(self, job_id: str) -> bool:
        """Stop training via HTTP POST."""
        response = self._client.post(
            f"{self._base_url}/api/training/stop/{job_id}"
        )

        if response.status_code == 404:
            return False

        response.raise_for_status()
        return True

    def is_training(self, model_id: str) -> bool:
        """Check if training via model metadata."""
        # Would need new endpoint: GET /api/models/{model_id}/training
        ...
```

### 6.3 Deployment Options

**Option A: Combined (current)**
```
Dash UI + Services (same process)
```

**Option B: Split UI and API**
```
Browser → Dash UI (port 8050)
              ↓
          FastAPI (port 8000) → Services → Training workers
```

**Option C: Multi-User**
```
Browser 1 ──┐
Browser 2 ──┼→ Load Balancer → Dash UI instances → FastAPI → Training queue
Browser 3 ──┘
```

### 6.4 Deliverables

**Files to create:**
- [ ] `use_cases/dashboard/api/__init__.py`
- [ ] `use_cases/dashboard/api/server.py`
- [ ] `use_cases/dashboard/api/models.py` - Pydantic request/response models
- [ ] `use_cases/dashboard/services/training_service.py` - Add `HTTPTrainingService`

**Configuration:**
- [ ] Add environment variable: `DASHBOARD_API_MODE=local|remote`
- [ ] If `remote`: `DASHBOARD_API_URL=http://api-server:8000`

**Documentation:**
- [ ] API documentation (OpenAPI/Swagger)
- [ ] Deployment guide for split architecture

**Success criteria:**
- [ ] Can run FastAPI server standalone
- [ ] Dash UI can connect to remote API
- [ ] API is fully documented (Swagger UI)
- [ ] End-to-end tests with HTTP client

---

## Success Criteria

### Phase 1 (Service Abstractions)
- [ ] All service interfaces defined
- [ ] `InProcessTrainingService` implemented
- [ ] Unit tests pass for all services
- [ ] No breaking changes to existing dashboard

### Phase 2 (Command Integration)
- [ ] All commands use services instead of direct state access
- [ ] Command tests use mock services
- [ ] Training starts via `TrainingService.start_training()`
- [ ] No duplicated training worker code

### Phase 3 (Dependency Injection)
- [ ] No global `app_state` variable
- [ ] `AppStateManager` replaces global state
- [ ] Services injected via `current_app.server`
- [ ] All callbacks updated
- [ ] Dashboard works end-to-end

### Phase 4 (Testing)
- [ ] 80%+ service code coverage
- [ ] All commands have unit tests
- [ ] Integration tests cover main workflows
- [ ] Tests can run in parallel

### Phase 5 (Process Isolation) - Future
- [ ] `ProcessPoolTrainingService` implemented
- [ ] Training runs in separate process
- [ ] Dashboard survives training crashes
- [ ] Opt-in via environment variable

### Phase 6 (REST API) - Future
- [ ] FastAPI server operational
- [ ] All endpoints documented
- [ ] `HTTPTrainingService` client works
- [ ] Can deploy UI and API separately

---

## Migration Checklist

### Before Starting
- [ ] Read this plan thoroughly
- [ ] Review current codebase ([DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md))
- [ ] Set up development environment
- [ ] Create feature branch: `feature/service-refactoring`

### During Each Phase
- [ ] Create phase-specific branch (e.g., `phase-1-services`)
- [ ] Implement changes incrementally
- [ ] Write tests before refactoring
- [ ] Run full test suite after each change
- [ ] Update documentation as you go
- [ ] Merge phase branch to feature branch when complete

### After Completion
- [ ] Full integration testing
- [ ] Performance testing (compare before/after)
- [ ] Update all documentation:
  - [ ] [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
  - [ ] [AGENT_GUIDE.md](AGENT_GUIDE.md)
  - [ ] [dashboard_state_plan.md](dashboard_state_plan.md)
- [ ] Merge feature branch to main
- [ ] Archive this plan as `docs/SERVICE_REFACTORING_COMPLETED.md`

---

## Notes

### Design Decisions

**Why services over direct domain access?**
- Abstraction: Commands don't need to know how training works
- Testability: Can mock services for unit tests
- Swappability: Can change implementation (thread → process → HTTP) without touching commands

**Why keep commands?**
- Commands handle state transitions (pure functions)
- Services handle domain operations (effectful)
- Clear separation of concerns

**Why dependency injection?**
- No global state (testable, parallel tests)
- Explicit dependencies (easier to reason about)
- Standard pattern (familiar to other developers)

### Risks

**Complexity increase:**
- More layers (commands → services → domain)
- Mitigation: Good documentation, clear interfaces

**Performance overhead:**
- Service indirection adds function calls
- Mitigation: Negligible compared to training time

**Migration effort:**
- Touching many files
- Mitigation: Incremental phases, each independently valuable

### Future Enhancements

Beyond this plan:
- **Job queue persistence** - Survive dashboard restarts
- **Priority scheduling** - User priority levels
- **Resource quotas** - Limit GPU hours per model
- **Audit logging** - Track all commands/operations
- **Metrics dashboard** - Grafana/Prometheus integration
- **Distributed training** - Multi-GPU via Ray/Dask

---

## Questions / Decisions Needed

- [ ] **Phase 1 timeline:** How many days/weeks per phase?
- [ ] **Testing strategy:** Unit tests only, or integration tests too?
- [ ] **Backward compatibility:** Support old checkpoints during migration?
- [ ] **Feature flags:** Use env vars or config file?
- [ ] **Process isolation:** Phase 5 priority (high/medium/low)?
- [ ] **REST API:** Phase 6 priority (high/medium/low)?

---

**Next Steps:**
1. Review this plan with team/stakeholders
2. Decide on phase priorities and timeline
3. Set up development branch
4. Begin Phase 1: Service Abstractions

**Estimated Effort:**
- Phase 1: 2-3 days
- Phase 2: 2-3 days
- Phase 3: 1-2 days
- Phase 4: 2-3 days
- **Total (Phases 1-4):** ~2 weeks

- Phase 5: 3-4 days (optional, future)
- Phase 6: 5-7 days (optional, future)

---

## AI-Assisted Implementation Guide

> **For efficient AI collaboration:** Each phase has clear target states and acceptance criteria. Focus entirely on reaching the next checkpoint robustly. Branch can be reset if needed.

### Quick Start for Next Session

**You say:** "Let's start Phase 1, Session 1: Create service skeleton"

**I do:**
1. Create directory structure
2. Write service protocol files
3. Run verification
4. Confirm checkpoint reached

**Branch safety:** You can always `git reset --hard` if session goes wrong.

### Phase 1: Detailed Session Plan

#### Session 1: Service Skeleton
**Target:** Empty service files that compile

**Deliverables:**
- `use_cases/dashboard/services/__init__.py`
- `use_cases/dashboard/services/training_service.py` (protocols only)
- `use_cases/dashboard/services/model_service.py` (stubs)
- `use_cases/dashboard/services/labeling_service.py` (stubs)
- `use_cases/dashboard/services/container.py`

**Verify:**
```bash
python -c "from use_cases.dashboard.services import ServiceContainer; print('✓')"
```

#### Session 2: TrainingService Implementation
**Target:** Working `InProcessTrainingService` with tests

**Deliverables:**
- Complete `InProcessTrainingService._train_worker()`
- Complete all `TrainingService` methods
- `tests/dashboard/services/test_training_service.py`

**Verify:**
```bash
pytest tests/dashboard/services/test_training_service.py -v
```

#### Session 3: ModelService + LabelingService
**Target:** All services implemented and tested

**Deliverables:**
- Complete `ModelService` methods
- Complete `LabelingService` methods
- Service tests

**Verify:**
```bash
pytest tests/dashboard/services/ -v
```

### Phase 2: Detailed Session Plan

#### Session 1: Update Command Protocol
**Target:** Commands accept services parameter

**Changes:**
- `Command` base class signature
- `CommandDispatcher` initialization and execute method

**Verify:** Code compiles (tests will fail - that's OK)

#### Session 2-4: Refactor Commands in Batches
**Target:** All commands use services

**Batch 1:** Training commands
**Batch 2:** Model commands
**Batch 3:** Labeling + remaining

**Verify after each:**
```bash
pytest tests/dashboard/commands/test_[command]_command.py -v
```

### Phase 3: Detailed Session Plan

#### Session 1: AppStateManager
**Target:** New state manager exists alongside old one

#### Session 2: Update app.py
**Target:** Services injected, both systems work

#### Session 3-4: Update Callbacks
**Target:** All callbacks use injection

#### Session 5: Delete Global State
**Target:** No globals, everything works

**Final verify:**
```bash
# Start dashboard
poetry run python use_cases/dashboard/app.py

# Check no globals remain
grep "^app_state.*=" use_cases/dashboard/core/state.py
```

### Efficiency Guidelines

**To save context:**
- Implement one session completely before next
- Minimal docstrings (code is self-documenting)
- Focus on target state, not perfection

**Verification pattern:**
After each change → Run tests → Confirm checkpoint → Move to next

**Recovery pattern:**
If stuck → Describe issue → I suggest: fix/rollback/skip → You decide

### Ready to Start?

When you're ready in the next session, just say:
**"Start Phase 1, Session 1"**

And I'll focus entirely on that specific target state.
