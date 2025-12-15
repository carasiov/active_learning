# Experimentation Contracts

This document describes stable contracts for experimentation in this repo: what a run produces, where metrics/plots come from, and what “adding a new signal” means.

For practical iteration guidance (configs, which plots to check first), see:
- `use_cases/experiments/iteration_runbook.md`

## Run artifact contracts

Experiments produce a run directory with a stable structure (RunPaths). Concrete details live in:
- `use_cases/experiments/README.md`
- `src/infrastructure/runpaths/structure.py`

At minimum, tooling should assume:
- `config.yaml` is a snapshot of the run config
- `summary.json` is the structured metrics output
- `REPORT.md` is a human-readable summary referencing figures
- `figures/` contains visualizations (core/mixture/τ/uncertainty)
- `artifacts/` contains checkpoints and optional diagnostics arrays

## Metric naming contracts

Metric keys are treated as a stable API:
- Canonical key schema: `src/infrastructure/metrics/schema.py`
- Default providers: `src/infrastructure/metrics/providers/defaults.py`

Rules:
- Add new keys in `schema.py` (avoid ad-hoc strings in providers/plotters).
- Prefer additive changes; renaming/removing keys is breaking for dashboards and analysis scripts.
- If a metric is only meaningful under certain configurations, mark it as disabled/skipped at the provider level rather than emitting misleading zeros.

## Visualization contracts

Plots are registered and gated via:
- Registry + execution: `src/infrastructure/visualization/registry.py`
- Default plotters (gating logic): `src/infrastructure/visualization/plotters.py`
- Implementations: `src/infrastructure/visualization/{core,mixture,tau}/plots.py`

Rules:
- Plotters must be explicit about applicability (disabled/skipped) based on `config`.
- Plots should not silently “look fine” when inputs are incompatible (e.g., curriculum with inactive channels needs active-set-aware plotting).
- Plot outputs should live under `figures/` in stable subfolders (`core/`, `mixture/`, `tau/`, …).

## Adding a new signal (metric or plot)

A change is considered integrated when:
1. A stable key exists (for metrics) or a stable output path exists (for plots).
2. It is registered via the appropriate registry (metric provider or plotter).
3. It appears in `summary.json` / `REPORT.md` (directly or via the generic sections), or is explicitly marked disabled/skipped with a reason.
4. A small run can exercise it without manual wiring.

## Project-specific evaluation

Active project specs may define additional signals and acceptance criteria. Example:
- Decentralized latents channel curriculum: `docs/projects/decentralized_latents/channel_curriculum/README.md`
