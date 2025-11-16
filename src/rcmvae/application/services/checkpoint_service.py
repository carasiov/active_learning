"""
CheckpointManager - Handles model state persistence.

Separates checkpoint I/O from model logic for easier testing and maintenance.
"""
from __future__ import annotations

from pathlib import Path

from collections.abc import Mapping, Sequence

from flax.serialization import from_bytes, to_bytes, msgpack_restore
from flax.core import FrozenDict, freeze, unfreeze

from rcmvae.application.runtime.state import SSVAETrainState


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
            data = f.read()

        try:
            payload = from_bytes(payload_template, data)
        except ValueError:
            # Fallback for checkpoints missing newer keys (e.g., prior params)
            raw_payload = msgpack_restore(data)

            def merge_structures(template, source):  # noqa: ANN001 - helper for nested merges
                if source is None:
                    return template

                # FrozenDict handling (params trees)
                if isinstance(template, FrozenDict):
                    target_dict = dict(template)
                    source_dict = dict(source) if isinstance(source, Mapping) else {}
                    merged = {}
                    for key, value in target_dict.items():
                        merged[key] = merge_structures(value, source_dict.get(key))
                    return FrozenDict(merged)

                # Mapping (plain dict)
                if isinstance(template, Mapping):
                    target_dict = dict(template)
                    source_dict = dict(source) if isinstance(source, Mapping) else {}
                    merged = {}
                    for key, value in target_dict.items():
                        merged[key] = merge_structures(value, source_dict.get(key))
                    return type(template)(merged)

                # Tuples / sequences (opt state chains)
                if isinstance(template, tuple):
                    source_seq = list(source) if isinstance(source, Sequence) else []
                    items = []
                    for idx, tmpl_item in enumerate(template):
                        src_item = source_seq[idx] if idx < len(source_seq) else None
                        items.append(merge_structures(tmpl_item, src_item))
                    return tuple(items)

                if isinstance(template, list):
                    source_seq = list(source) if isinstance(source, Sequence) else []
                    items = []
                    for idx, tmpl_item in enumerate(template):
                        src_item = source_seq[idx] if idx < len(source_seq) else None
                        items.append(merge_structures(tmpl_item, src_item))
                    return items

                # Scalars / arrays: prefer source when available
                return source

            params_template = payload_template["params"]
            opt_state_template = payload_template["opt_state"]
            raw_params = raw_payload.get("params")
            raw_opt_state = raw_payload.get("opt_state")

            merged_params = merge_structures(params_template, raw_params)
            try:
                merged_opt_state = merge_structures(opt_state_template, raw_opt_state)
            except Exception:
                merged_opt_state = opt_state_template

            payload = {
                "params": merged_params,
                "opt_state": merged_opt_state,
                "step": raw_payload.get("step", payload_template["step"]),
            }

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
