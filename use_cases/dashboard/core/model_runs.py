from __future__ import annotations

"""Lightweight manifest of dashboard training runs per model."""

import json
from pathlib import Path
from typing import Any, Dict, List

from use_cases.dashboard.core.model_manager import ModelManager
from use_cases.dashboard.core.state_models import RunRecord

_MANIFEST_NAME = "runs.json"


def _manifest_path(model_id: str) -> Path:
    return ModelManager.model_dir(model_id) / _MANIFEST_NAME


def load_run_manifest(model_id: str) -> List[Dict[str, Any]]:
    path = _manifest_path(model_id)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def load_run_records(model_id: str) -> List[RunRecord]:
    """Return manifest as RunRecord instances sorted most-recent first."""
    raw = load_run_manifest(model_id)
    records = [RunRecord.from_dict(item) for item in raw]
    records.sort(key=lambda rec: rec.timestamp, reverse=True)
    return records


def append_run_record(model_id: str, record: Dict[str, Any]) -> List[RunRecord]:
    runs = load_run_manifest(model_id)
    runs = [existing for existing in runs if existing.get("run_id") != record.get("run_id")]
    runs.append(record)
    runs.sort(key=lambda item: item.get("timestamp", ""), reverse=True)

    path = _manifest_path(model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(runs, handle, indent=2)
    return [RunRecord.from_dict(item) for item in runs]
