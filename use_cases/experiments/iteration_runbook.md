# Iteration Runbook (Experiments)

This is a practical, change-friendly guide for how we iterate on the model via `use_cases/experiments/run_experiment.py`. It is intentionally allowed to be specific to the current tooling and defaults.

For stable “contracts” (registries, naming, artifact expectations), see `docs/development/experimentation_contracts.md`.

## Default workflow

1. Choose or edit a config under `use_cases/experiments/configs/` (often `use_cases/experiments/configs/quick.yaml`).
2. Run validate-only to catch wiring/config errors early:

```bash
poetry run python use_cases/experiments/run_experiment.py --config use_cases/experiments/configs/quick.yaml --validate-only
```

3. Run a small training job and inspect the report output directory:

```bash
JAX_PLATFORMS=cpu poetry run python use_cases/experiments/run_experiment.py --config use_cases/experiments/configs/quick.yaml
```

## What we check first in `REPORT.md`

- Loss curves and stability (training vs validation).
- Reconstructions (do we reconstruct at all; do we regress visually?).
- Mixture routing hardness and specialization (for mixture-based priors).
- τ plots/metrics only when `use_tau_classifier: true`.

If a plot is disabled, check the “plot status” section in the report; that usually means a config gate is not satisfied (e.g., `decoder_conditioning: none` disables component-conditioning plots).

## Common signals (current)

Metric names are defined in `src/infrastructure/metrics/schema.py`; the report summarizes a subset.

Mixture routing:
- `mixture.responsibility_confidence_mean`
- `mixture.component_entropy`
- `mixture.K_eff`, `mixture.active_components`
- `mixture.pi_entropy`, `mixture.pi_max`

Logit-MoG regularizer:
- `loss.kl_c_logit_mog`

## Practical notes for decentralized latents

For decentralized runs (one latent per component) you usually have:
- `latent_layout: decentralized`
- `use_gumbel_softmax: true` (often straight-through)

Important: this changes the *decoder routing behavior* (hard vs soft) but does not change which distribution is best for monitoring. Usage/sparsity monitoring is typically more stable on deterministic responsibilities than on sampled `component_selection`.

## Curriculum-facing iteration

If/when the channel curriculum (“pots”) is enabled, this runbook should be extended with:
- the recommended curriculum config block and defaults
- what unlock events look like in plots/metrics

Project-level evaluation criteria live in:
- `docs/projects/decentralized_latents/channel_curriculum/validation_signals.md`

