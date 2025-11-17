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
