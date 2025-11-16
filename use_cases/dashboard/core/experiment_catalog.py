from __future__ import annotations

"""Filesystem scanners for experiment results consumed by the dashboard."""

from dataclasses import dataclass
import base64
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

RESULTS_ROOT = Path(__file__).resolve().parents[2] / "experiments" / "results"
_IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".svg")
_MAX_INLINE_IMAGE_BYTES = 4_500_000  # ~4.5 MB per inline image


@dataclass(frozen=True)
class RunListEntry:
    run_id: str
    experiment_name: str
    timestamp: str
    architecture_code: Optional[str]
    status: str
    path: Path
    model_id: Optional[str]
    tags: Sequence[str]
    metrics: Dict[str, Any]
    summary_path: Optional[Path]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "architecture_code": self.architecture_code,
            "status": self.status,
            "path": str(self.path),
            "model_id": self.model_id,
            "tags": list(self.tags),
            "metrics": self.metrics,
            "summary_path": str(self.summary_path) if self.summary_path else None,
        }


@dataclass(frozen=True)
class FigurePreview:
    category: str
    name: str
    data_url: Optional[str]
    relative_path: str
    size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "name": self.name,
            "data_url": self.data_url,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class RunDetail:
    entry: RunListEntry
    summary: Dict[str, Any]
    history: Dict[str, Any]
    config_text: Optional[str]
    report_path: Optional[Path]
    figures: List[FigurePreview]
    artifacts: Dict[str, List[str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry": self.entry.to_dict(),
            "summary": self.summary,
            "history": self.history,
            "config_text": self.config_text,
            "report_path": str(self.report_path) if self.report_path else None,
            "figures": [fig.to_dict() for fig in self.figures],
            "artifacts": self.artifacts,
        }


def list_runs(limit: Optional[int] = None) -> List[RunListEntry]:
    runs: List[RunListEntry] = []
    if not RESULTS_ROOT.exists():
        return runs

    for run_dir in sorted(_iter_run_dirs(RESULTS_ROOT), reverse=True):
        entry = _build_run_entry(run_dir)
        if entry:
            runs.append(entry)
            if limit is not None and len(runs) >= limit:
                break
    return runs


def load_run_detail(run_id: str, *, max_images_per_category: int = 4) -> Optional[RunDetail]:
    run_dir = RESULTS_ROOT / run_id
    if not run_dir.exists():
        return None

    entry = _build_run_entry(run_dir)
    if not entry:
        return None

    summary = _load_json(run_dir / "summary.json") or {}
    history = (
        _load_json(run_dir / "artifacts" / "diagnostics" / "history.json")
        or _load_json(run_dir / "history.json")
        or {}
    )
    config_text = _load_text(run_dir / "config.yaml")
    figures = _load_figures(run_dir, max_per_category=max_images_per_category)
    artifacts = _enumerate_artifacts(run_dir)
    report_path = (run_dir / "REPORT.md") if (run_dir / "REPORT.md").exists() else None

    return RunDetail(
        entry=entry,
        summary=summary,
        history=history,
        config_text=config_text,
        report_path=report_path,
        figures=figures,
        artifacts=artifacts,
    )


def serialize_run_list(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    return [entry.to_dict() for entry in list_runs(limit=limit)]


def _iter_run_dirs(root: Path) -> Iterable[Path]:
    for child in root.iterdir():
        if child.is_dir():
            yield child


def _build_run_entry(run_dir: Path) -> Optional[RunListEntry]:
    run_id = run_dir.name
    summary_path = run_dir / "summary.json"
    summary = _load_json(summary_path) or {}

    metadata = summary.get("metadata", {})
    experiment_name = metadata.get("experiment_name") or _infer_experiment_name(run_id)
    timestamp = metadata.get("timestamp") or _infer_timestamp(run_id)
    architecture_code = metadata.get("architecture_code") or _infer_architecture_code(run_id)
    model_id = metadata.get("model_id")
    tags = metadata.get("tags") or []
    status = metadata.get("status") or ("complete" if summary else "pending")
    metrics = _extract_primary_metrics(summary)

    return RunListEntry(
        run_id=run_id,
        experiment_name=experiment_name,
        timestamp=timestamp,
        architecture_code=architecture_code,
        status=status,
        path=run_dir,
        model_id=model_id,
        tags=tags,
        metrics=metrics,
        summary_path=summary_path if summary_path.exists() else None,
    )


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _load_text(path: Path, max_bytes: int = 64_000) -> Optional[str]:
    if not path.exists():
        return None
    try:
        data = path.read_bytes()
        if len(data) > max_bytes:
            data = data[:max_bytes]
        return data.decode("utf-8", errors="replace")
    except Exception:
        return None


def _infer_experiment_name(run_id: str) -> str:
    parts = run_id.split("__")
    if len(parts) >= 1 and parts[0]:
        return parts[0].replace("_", " ").title()
    return run_id


def _infer_architecture_code(run_id: str) -> Optional[str]:
    parts = run_id.split("__")
    if len(parts) >= 3:
        return parts[1]
    if len(parts) == 2:
        return parts[1]
    return None


def _infer_timestamp(run_id: str) -> str:
    parts = run_id.split("__")
    last = parts[-1]
    return last


def _extract_primary_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    training = summary.get("training", {})
    if training:
        for key in ("final_loss", "final_recon_loss", "final_kl_z", "epochs_completed"):
            if key in training:
                metrics[key] = training[key]

    classification = summary.get("classification", {})
    if classification:
        for key in ("final_accuracy", "final_classification_loss"):
            if key in classification:
                metrics[key] = classification[key]

    mixture = summary.get("mixture", {})
    if mixture:
        for key in ("K_eff", "active_components", "responsibility_confidence_mean"):
            if key in mixture:
                metrics[key] = mixture[key]

    tau_metrics = summary.get("tau_classifier", {})
    if tau_metrics:
        for key in ("certainty_mean", "num_free_channels", "avg_components_per_label"):
            if key in tau_metrics:
                metrics[key] = tau_metrics[key]

    return metrics


def _load_figures(run_dir: Path, *, max_per_category: int) -> List[FigurePreview]:
    figures_root = run_dir / "figures"
    previews: List[FigurePreview] = []
    if not figures_root.exists():
        return previews

    for category_dir in sorted(figures_root.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        count = 0
        for image_path in sorted(category_dir.glob("**/*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            relative = image_path.relative_to(run_dir)
            data_url = _encode_image(image_path)
            size_bytes = image_path.stat().st_size if image_path.exists() else 0
            previews.append(
                FigurePreview(
                    category=category,
                    name=image_path.stem.replace("_", " ").title(),
                    data_url=data_url,
                    relative_path=str(relative),
                    size_bytes=size_bytes,
                )
            )
            count += 1
            if count >= max_per_category:
                break
    return previews


def _encode_image(path: Path) -> Optional[str]:
    try:
        data = path.read_bytes()
    except Exception:
        return None

    if len(data) > _MAX_INLINE_IMAGE_BYTES:
        return None

    encoded = base64.b64encode(data).decode("ascii")
    mime = _infer_mime(path.suffix)
    return f"data:{mime};base64,{encoded}"


def _infer_mime(suffix: str) -> str:
    ext = suffix.lower().lstrip('.')
    if ext in {"jpg", "jpeg"}:
        return "image/jpeg"
    if ext == "svg":
        return "image/svg+xml"
    return "image/png"


def _enumerate_artifacts(run_dir: Path) -> Dict[str, List[str]]:
    artifacts_root = run_dir / "artifacts"
    if not artifacts_root.exists():
        return {}

    artifact_map: Dict[str, List[str]] = {}
    for subdir in sorted(artifacts_root.iterdir()):
        if not subdir.is_dir():
            continue
        relative_paths = [str(path.relative_to(run_dir)) for path in sorted(subdir.rglob("*")) if path.is_file()]
        if relative_paths:
            artifact_map[subdir.name] = relative_paths
    return artifact_map