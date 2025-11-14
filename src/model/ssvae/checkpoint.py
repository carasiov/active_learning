"""
CheckpointManager - Handles model state persistence.

Separates checkpoint I/O from model logic for easier testing and maintenance.
"""
from __future__ import annotations

from pathlib import Path

from flax.serialization import from_bytes, to_bytes

from model.training.train_state import SSVAETrainState


class CheckpointManager:
    """Manages saving and loading of SSVAE training state.

    This class isolates checkpoint I/O from the main model class,
    making it easier to:
    - Test checkpoint logic independently
    - Support different checkpoint formats
    - Add checkpointing features without modifying SSVAE
    """

    @staticmethod
    def save(state: SSVAETrainState, path: str) -> None:
        """Save training state to disk.

        Args:
            state: Training state to save
            path: Path to save checkpoint

        Example:
            >>> manager = CheckpointManager()
            >>> manager.save(state, "model.ckpt")
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "params": state.params,
            "opt_state": state.opt_state,
            "step": state.step,
        }

        data = to_bytes(payload)
        with open(path, "wb") as f:
            f.write(data)

    @staticmethod
    def load(state_template: SSVAETrainState, path: str) -> SSVAETrainState:
        """Load training state from disk.

        Args:
            state_template: Template state with correct structure
            path: Path to checkpoint file

        Returns:
            Updated training state with loaded parameters

        Example:
            >>> manager = CheckpointManager()
            >>> loaded_state = manager.load(initial_state, "model.ckpt")
        """
        with open(path, "rb") as f:
            payload_template = {
                "params": state_template.params,
                "opt_state": state_template.opt_state,
                "step": state_template.step,
            }
            payload = from_bytes(payload_template, f.read())

        return state_template.replace(
            params=payload["params"],
            opt_state=payload["opt_state"],
            step=payload["step"],
        )

    @staticmethod
    def checkpoint_exists(path: str) -> bool:
        """Check if checkpoint file exists.

        Args:
            path: Path to checkpoint

        Returns:
            True if checkpoint exists, False otherwise
        """
        return Path(path).exists()

    @staticmethod
    def get_checkpoint_info(path: str) -> dict:
        """Get metadata about a checkpoint without fully loading it.

        Args:
            path: Path to checkpoint

        Returns:
            Dictionary with checkpoint metadata
        """
        p = Path(path)
        if not p.exists():
            return {"exists": False}

        return {
            "exists": True,
            "path": str(p.absolute()),
            "size_bytes": p.stat().st_size,
            "modified": p.stat().st_mtime,
        }
