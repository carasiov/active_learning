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
