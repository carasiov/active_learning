"""Training execution service - abstracts how training runs."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
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
    model_id: str = ""
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
                model_id=job.model_id,
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
                if progress.model_id == model_id and progress.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                    return True
            return False

    def _train_worker(self, job: TrainingJob) -> None:
        """Background worker - executes training loop.

        This consolidates logic from train_worker() and train_worker_hub().
        """
        stop_flag = self._stop_flags[job.job_id]

        try:
            # Update status to RUNNING
            with self._lock:
                if job.job_id in self._jobs:
                    self._jobs[job.job_id] = JobProgress(
                        status=JobStatus.RUNNING,
                        current_epoch=self._jobs[job.job_id].current_epoch,
                        total_epochs=self._jobs[job.job_id].total_epochs,
                        latest_metrics={},
                        model_id=job.model_id,
                    )

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
                        self._jobs[job.job_id] = JobProgress(
                            status=JobStatus.RUNNING,
                            current_epoch=epoch,
                            total_epochs=self._jobs[job.job_id].total_epochs,
                            latest_metrics=metrics,
                            model_id=job.model_id,
                        )

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
                if job.job_id in self._jobs:
                    self._jobs[job.job_id] = JobProgress(
                        status=JobStatus.COMPLETED,
                        current_epoch=self._jobs[job.job_id].current_epoch,
                        total_epochs=self._jobs[job.job_id].total_epochs,
                        latest_metrics=self._jobs[job.job_id].latest_metrics,
                        model_id=job.model_id,
                    )

            self._metrics_queue.put({
                "type": "complete",
                "job_id": job.job_id,
            })

        except TrainingStoppedException:
            with self._lock:
                if job.job_id in self._jobs:
                    self._jobs[job.job_id] = JobProgress(
                        status=JobStatus.CANCELLED,
                        current_epoch=self._jobs[job.job_id].current_epoch,
                        total_epochs=self._jobs[job.job_id].total_epochs,
                        latest_metrics=self._jobs[job.job_id].latest_metrics,
                        model_id=job.model_id,
                    )

            self._metrics_queue.put({
                "type": "cancelled",
                "job_id": job.job_id,
            })

        except Exception as e:
            with self._lock:
                if job.job_id in self._jobs:
                    self._jobs[job.job_id] = JobProgress(
                        status=JobStatus.FAILED,
                        current_epoch=self._jobs[job.job_id].current_epoch,
                        total_epochs=self._jobs[job.job_id].total_epochs,
                        latest_metrics=self._jobs[job.job_id].latest_metrics,
                        model_id=job.model_id,
                        error_message=str(e),
                    )

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
