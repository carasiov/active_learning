# Logistic-Normal Mixture Regularizer Integration (2025-12-08)

This note captures the recent work to **add** a logistic-normal mixture prior on encoder logits as an alternative to the categorical KL on the discrete channel variable `c`, aligning with the supervisor directive for peaky responsibilities without relying on Dirichlet-style categorical sampling.

## Rationale

- Supervisor requested a Gaussian-only prior penalty in logit space that is easy to evaluate via log-sum-exp, avoiding Dirichlet/categorical KL complications (see AGENTS transcript, 2025‑12‑08).
- Encourages one-hot-like `q(c|x)` by fitting encoder logits against a mixture of isotropic Gaussians centred at +M·e_k before the softmax.
- Leaves the Dirichlet MAP on global π unchanged; the change applies only to the per-sample `c` regularizer.

## Mathematical Definition

Let `component_logits = y(x)` be the deterministic encoder outputs (pre-Gumbel, before any straight-through sampling). When the logit mixture prior is active we optimize

```
L_c_logit_mog = - λ · E_x [ log p_mix(y(x)) ]
```

where `p_mix(y) = (1/|A|) Σ_{k∈A} N(y; μ_k, σ²I)` with `μ_k = M · e_k`, and `A` is the set of **active** channels (for the curriculum setting; when there is no active-set curriculum, take `A={1,…,K}`).

This can be viewed as a KL to the prior if we model `q(y|x)` as a Dirac delta at `y(x)`. In practice it’s implemented as a negative log prior penalty evaluated at the deterministic logits. This penalty is applied before any Gumbel noise is added; downstream routing (soft responsibilities vs. straight-through Gumbel) is orthogonal.

Implementation detail: `log p_mix` is evaluated with a numerically stable `logsumexp` of the per-component log densities.

## Implementation Overview

| Concern | Location | Notes |
|---------|----------|-------|
| Config surface | `src/rcmvae/domain/config.py` | Added `c_regularizer ∈ {"categorical","logit_mog","both"}` plus `c_logit_prior_{weight,mean,sigma}`. Defaults stay `"categorical"` for backward compat. |
| Prior logic | `src/rcmvae/domain/priors/mixture.py` | Computes `kl_c_logit_mog` on raw encoder `component_logits` (pre-Gumbel). Means at +M·e_k, isotropic σ. |
| Loss aggregation | `src/rcmvae/application/services/loss_pipeline.py` | New term anneals with the same `kl_c_scale` schedule to avoid extra trainer plumbing. |
| Trainer history | `src/rcmvae/application/services/training_service.py` | Tracks `kl_c_logit_mog` for both train/val splits so histories, CSV exports, and callbacks see the new metric. |
| Experiment metrics/reporting | `src/infrastructure/metrics/{schema.py,providers/defaults.py}` | Canonical key `loss.kl_c_logit_mog` plus `final_kl_c_logit_mog` in summary outputs; REPORT.md now lists it under the Training section automatically. |
| Tests | `tests/test_mixture_losses.py::test_logit_mog_penalty_prefers_axis_aligned_logits` | Guards against regressions by ensuring axis-aligned logits get lower penalty than flat logits. |

## Experiment Configuration

- `use_cases/experiments/configs/quick.yaml` now opts into the new behaviour:
  - `c_regularizer: "logit_mog"`
  - `kl_c_weight: 0.00` (categorical KL disabled, but annealing still scales the logit penalty via `kl_c_scale`)
  - Tunable knobs for mean/sigma/weight (in `quick.yaml` today: mean=7.0, sigma=1.0, weight=1.0; in practice, peaky behaviour often needs larger `M/σ`).
    - Peakiness strength is governed by `M / σ`; increasing `c_logit_prior_mean` or decreasing `c_logit_prior_sigma` pushes responsibilities closer to one-hot.
- Gumbel-Softmax routing remains enabled (`use_gumbel_softmax: true`, straight-through) to support decentralised decoder selection; the regularizer operates on the logits before any Gumbel perturbation.
- Reminder: Supervisor expects **only** the logit mixture term (set `kl_c_weight≈0`). The `"both"` mode is available for experiments that want local peakiness **plus** global alignment to π, but weights may need tuning since the objectives can compete. Despite the `kl_` prefix, the `kl_c_logit_mog` metric is a negative log prior under the mixture applied to deterministic logits.
- When `c_regularizer="logit_mog"` and `kl_c_weight=0`, π influences training only via the Dirichlet MAP penalty; it no longer shapes per-sample responsibilities directly.

## Run / Validation Notes

- Validation: `poetry run python use_cases/experiments/run_experiment.py --config use_cases/experiments/configs/quick.yaml --validate-only`.
- Full CPU runs hit the 10 min timeout in this environment; expect ~>10 min compile time. Prefer GPU or relaxed timeout to finish a run and materialize REPORT with the new metric.
- Once a run completes, `summary.json` exposes `training.final_kl_c_logit_mog` and plotting/reporting surfaces the logit-regularizer metric without additional wiring.

## Follow-ups / Risks

1. **Metric consumers**: Any downstream dashboards that hard-code metric lists must include `training.final_kl_c_logit_mog` to avoid dropping the new signal.
2. **Docs alignment**: `docs/theory/conceptual_model.md` still references Dirichlet regularization; update Theory layer once logistic-normal is fully validated.
3. **Anneal schedule**: Currently shares `kl_c_anneal_epochs`. Monitor whether independent scheduling is needed once experiments scale.
4. **Config defaults**: Project default remains `"categorical"`; switch to `"logit_mog"` when ready to make it the global default.
