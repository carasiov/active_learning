# Curriculum for Decentralized Latent Channels — Implementation Mapping

This appendix maps the "Design Contract" (`docs/projects/decentralized_latents/channel_curriculum/design_contract.md:1`) onto the current codebase, and identifies the concrete insertion points needed to implement active-channel masking + unlocking ("pots").

## 0) Current state (what exists today)

> **UPDATE 2025-12-23:** Curriculum IS now implemented. The description below is **historical** (pre-implementation state). See `DELEGATION_V2.md` for implementation details and `experimental_findings_2025-12-23.md` for current experimental results.

~~- There is **no** runtime notion of "inactive / not-yet-unlocked" channels; all `K=num_components` always compete in routing (`src/rcmvae/domain/network.py:171`).~~
~~- Curriculum unlocking ("pots") is not implemented yet; this appendix documents the insertion points and minimal plumbing needed to add it.~~

**Current state (2025-12-23):**
- Curriculum controller implemented: `src/rcmvae/application/curriculum/controller.py`
- Curriculum hooks implemented: `src/rcmvae/application/curriculum/hooks.py`
- Active masking works; kick mechanism (logit bias + temperature) implemented
- **Known issue:** 2-channel coalition phenomenon — see `experimental_findings_2025-12-23.md`

- `logit_mog` exists and applies to raw logits (`src/rcmvae/domain/priors/mixture.py:107`) and is surfaced in experiment outputs (`src/infrastructure/metrics/schema.py:50`).

## 1) Where the active mask is stored / updated

Recommended (JAX-static-shape friendly):

- Keep `K_max = config.num_components` fixed.
- Maintain a Python-side training controller that tracks:
  - `K_active` (int)
  - `active_mask` (shape `[K_max]`, 0/1 or bool)
  - optional `kick_remaining` (int epochs/steps)

Best current wiring point for a runtime-controlled value is the **trainer loop hooks**:

- `TrainerLoopHooks` supports `batch_context_fn` and `eval_context_fn` that return `Dict[str, jnp.ndarray] | None` (`src/rcmvae/application/services/training_service.py:17`).
- Those contexts are forwarded into the train/eval step call sites (so the train step can pass them down into `compute_loss_and_metrics_v2` / model forward). This is the least invasive place to put curriculum state *without* changing parameter shapes.

If the mask must be checkpointed, store it in state (requires small extensions):

- `SSVAETrainState` is the explicit state object used in training (`src/rcmvae/application/runtime/state.py`).

## 2) Where logits are masked before softmax / Gumbel

Today:

- Soft responsibilities: `responsibilities = softmax(component_logits)` (`src/rcmvae/domain/network.py:176`).
- Routing distribution: `component_selection` from either Gumbel-softmax or softmax fallback (`src/rcmvae/domain/network.py:193`).

Curriculum masking should occur before both of these.

Proposed insertion:

- In `src/rcmvae/domain/network.py`, introduce a runtime input (e.g. `active_mask` and optional `logit_bias`) and apply a routing-only mask:
  - `routing_logits = jnp.where(active_mask, component_logits + logit_bias, -jnp.inf)`
  - then compute **both** `responsibilities` and `component_selection` from `routing_logits` (so diagnostics/usage match what the decoder can select).

Important nuance: `routing_logits` is for *softmax/Gumbel routing only* — do **not** feed masked logits with `-inf` into the logit-MoG Gaussian distance computation (see §3).

Note: `SSVAE.__call__` currently accepts `gumbel_temperature` as an override (`src/rcmvae/domain/network.py:159`). This is already a hook for a “kick” mechanism (temperature changes); masking/logit bias would need a similar override in the forward signature.

## 3) Where the logit-mixture prior sums over the active set

Today:

- `kl_c_logit_mog` is computed in `MixtureGaussianPrior.compute_kl_terms()` using `k = component_logits.shape[-1]` and a uniform mixture over all `k` (`src/rcmvae/domain/priors/mixture.py:112`).

Curriculum requirement:

- The mixture prior must sum over **active channels only** `|A|`, and the Gaussian means should be indexed accordingly.

Implementation options:

1. **Preferred (contract-aligned and numerically safe)**:
   - keep `component_logits = y(x)` as the raw, finite encoder output, and
   - restrict the logit-mixture sum to active components only (i.e., compute `logsumexp` over `k∈A` and normalize by `|A|`).

   Do **not** use `-inf`-masked logits in the Gaussian mixture distance calculation: the current implementation computes squared distances `(y - μ_k)^2`, and `-inf` entries would produce `inf`/`nan` and destabilize training.
2. **Alternative**: pass `active_mask` (or active indices / `K_active`) into `compute_kl_terms()` and implement active-set mixture evaluation there, still using the raw logits `component_logits`.

Related loss annealing:

- `compute_loss_and_metrics_v2()` already scales `kl_c_logit_mog` by `kl_c_scale` (`src/rcmvae/application/services/loss_pipeline.py:439`), so curriculum can reuse `kl_c_anneal_epochs` or add an independent schedule later.

## 4) Where recon weighting uses the masked routing distribution

Today:

- The decoder computes reconstructions for all `K` (`src/rcmvae/domain/network.py:270`) and then forms the expectation using `component_selection` (`src/rcmvae/domain/network.py:309`).
- The prior’s reconstruction loss uses `extras["component_selection"]` when present (`src/rcmvae/domain/priors/mixture.py:120`).

Therefore:

- If `component_selection` is computed from masked logits, inactive channels contribute exactly 0 to the expectation and to the reconstruction loss.
- Decoder compute will still run for all `K` unless further optimized (correctness first; optimization later).

## 5) Where the unlock trigger is computed and recorded

Best place to compute triggers:

- Inside the Python-side `Trainer.train()` loop, after metrics are available for the epoch (`src/rcmvae/application/services/training_service.py:160`), because:
  - plateau detection is inherently non-JIT and uses history/float comparisons,
  - unlock events should be written into history/diagnostics for reporting.

How to record it:

- Extend the trainer history to include:
  - `curriculum_k_active` (train/val if needed),
  - `curriculum_unlocked` (0/1 event flag),
  - `curriculum_kick_active` (0/1).

History plumbing exists centrally in the trainer (`src/rcmvae/application/services/training_service.py:113` and `_update_history()` at `src/rcmvae/application/services/training_service.py:501`).

Recommendation: if runs can be resumed from checkpoints, treat `K_active` and kick counters as **checkpointed state**, not “nice to have”. Otherwise a resumed run can silently diverge from the intended curriculum stage.

## 6) Where the kick window modifies routing temperature / ST / bias / weights

Existing hook:

- `compute_loss_and_metrics_v2` accepts a `gumbel_temperature` override (`src/rcmvae/application/services/loss_pipeline.py:375`) which is passed into the model forward (`src/rcmvae/application/services/loss_pipeline.py:394`).
- The model forward uses the override if provided (`src/rcmvae/domain/network.py:187`).

Therefore, the easiest “kick” to implement first is:

- **Temperature kick**: for `T_kick` epochs after unlock, set a higher `gumbel_temperature` via the trainer’s `batch_context_fn` / `eval_context_fn` → train/eval step → `compute_loss_and_metrics_v2(..., gumbel_temperature=...)`.

If we want the other kicks:

- **Logit bias kick** needs a forward override beyond temperature (add `logit_bias` and apply before softmax/Gumbel in `src/rcmvae/domain/network.py`).
- **Disable straight-through kick** similarly needs a runtime override (currently controlled by config `use_straight_through_gumbel`; see `src/rcmvae/domain/network.py:211`).
- **Regularizer relaxation kick** can be done without forward changes by temporarily setting an effective `c_logit_prior_weight` (but today it is read from config inside `MixtureGaussianPrior.compute_kl_terms()`).

## 7) Which metrics are logged for sparsity, usage, and unlock events

Already available (mixture metrics):

- `responsibility_confidence_mean` etc are computed and rendered by the visualization/metrics stack (see mixture metrics provider and plotters; entrypoints are `src/infrastructure/metrics/providers/defaults.py` and `src/infrastructure/visualization/plotters.py`).

Already tracked in training history:

- `component_entropy`, `pi_entropy`, `kl_c_logit_mog` (`src/rcmvae/application/services/training_service.py:113`).

To add for curriculum:

- Add `curriculum_k_active` + unlock/kick flags into trainer history and into metric providers:
  - training history: `src/rcmvae/application/services/training_service.py:113`
  - summary export: `src/infrastructure/metrics/providers/defaults.py:27`
  - canonical naming: `src/infrastructure/metrics/schema.py`

Monitoring nuance: prefer using masked **responsibilities** (deterministic) for usage/sparsity statistics and unlock decisions, rather than the potentially noisy `component_selection` when Gumbel sampling is active.

## 8) Notes / caveats to keep aligned with current code

- `top_m_gating` and `soft_embedding_warmup_epochs` exist in config but are not implemented; do not rely on them for curriculum behavior without adding explicit code support (`src/rcmvae/domain/config.py:201`).
- Visualization “Component Embedding” plotters run when `decoder_conditioning != "none"` (and are disabled for VampPrior because it has no component embeddings).

## 9) Experiment workflow implications (run_experiment.py + quick.yaml)

### 9.1 How experiments are run today

- `use_cases/experiments/run_experiment.py` loads YAML and constructs `SSVAEConfig(**model_config)` from the YAML `model:` section (`use_cases/experiments/run_experiment.py:85`).
- Training is orchestrated by `ExperimentService.run()` which constructs an `SSVAE` and calls `SSVAE.fit(...)` (`src/rcmvae/application/services/experiment_service.py:40`).
- `SSVAE.fit()` currently constructs loop hooks **only** for the τ-classifier (`SSVAE._build_tau_loop_hooks`) and passes them into `Trainer.train(..., loop_hooks=...)` (`src/rcmvae/application/model_api.py:288`).

Consequence: curriculum cannot be implemented “purely from YAML” today because there is no mechanism for experiments to inject additional trainer hooks beyond τ.

### 9.2 What must change to support curriculum runs

To approach the target state (active-channel curriculum), the experiments stack needs a path to:

- parse a **new** top-level `curriculum:` section in the YAML (must not be forwarded into `SSVAEConfig`, otherwise config construction will fail), and
- inject curriculum state into training via `TrainerLoopHooks.batch_context_fn` / `eval_context_fn` (and optionally `post_batch_fn`) (`src/rcmvae/application/services/training_service.py:17`).

Minimal plumbing changes (high-level):

1. Extend `ExperimentService.run()` to accept curriculum settings (or a `loop_hooks` argument) and forward it into `SSVAE.fit(...)`.
2. Extend `SSVAE.fit()` to accept **external** loop hooks (curriculum) and merge them with the existing τ hooks (if τ is enabled) before calling the trainer.
3. Extend the train/eval step plumbing to pass curriculum context through to:
   - model forward (masking + optional kick overrides),
   - prior KL computation (active-set logit-MoG evaluation).

## 10) Config delta: quick.yaml → curriculum-ready config

`use_cases/experiments/configs/quick.yaml` is currently a “static K” run: all `num_components` are active from epoch 0, and there is no concept of unlocking.

For curriculum, the safest approach is to keep the current `model:` section mostly intact, and add a new top-level `curriculum:` section in YAML (so it is not passed into `SSVAEConfig`).

### 10.1 Fields likely to change in `model:` for curriculum runs

- `component_diversity_weight`:
  - **current**: strongly negative (entropy reward → encourages many globally used components).
  - **curriculum default**: `0.0` (neutral) so “how many pots exist” is controlled by `K_active`, not by a conflicting global reward (`src/rcmvae/application/services/loss_pipeline.py:207`).
- `kl_c_anneal_epochs`:
  - currently scales `kl_c_logit_mog` because the loss pipeline multiplies `kl_c_logit_mog` by `kl_c_scale` (`src/rcmvae/application/services/loss_pipeline.py:436`).
  - consider whether we want to keep this coupling or introduce a separate anneal schedule for `logit_mog` later.
- `top_m_gating`, `soft_embedding_warmup_epochs`:
  - currently unused; curriculum should not rely on them unless implemented (`src/rcmvae/domain/config.py:201`).

### 10.2 New top-level `curriculum:` section (proposed schema)

Example (illustrative; not implemented yet):

```yaml
curriculum:
  enabled: true
  k_active_init: 1
  k_active_max: null            # null → use model.num_components

  unlock:
    policy: "plateau"           # plateau first; “normality” can come later
    monitor: "val_loss"         # or "val_reconstruction_loss"
    patience_epochs: 10
    min_delta: 0.001
    cooldown_epochs: 3          # avoid rapid repeated unlocks

  kick:
    enabled: true
    epochs: 5
    gumbel_temperature: 5.0     # uses existing override path
    # future (requires new runtime overrides):
    # disable_straight_through: true
    # logit_bias_new_channel: 1.0
    # logit_mog_weight_scale: 0.2
```

Key compatibility rule: the `curriculum:` section must be consumed by experiments/training orchestration and must not be forwarded into `SSVAEConfig(**model_config)`.

## 11) Visualization + reporting implications

### 11.1 What stays compatible

Most mixture visualizations remain meaningful if masking is implemented correctly (inactive channels get ~zero mass):

- channel latent responsibility grids
- responsibility histograms
- routing hardness (soft vs hard)
- selected vs weighted reconstruction
- mixture evolution (π and usage) — but should ideally annotate unlock epochs

### 11.2 What should be added for curriculum

To validate curriculum behavior, add:

- a new `figures/curriculum/k_active_over_time.png` plot (and/or annotate existing loss curves) showing `K_active` and unlock events.
- summary fields like:
  - `curriculum.final_k_active`
  - `curriculum.unlock_events` (count)
  - `curriculum.kick_epochs_total`

### 11.3 What needs adjustment under curriculum

- Any “active components” metric should be computed **within the active set**; otherwise it will be dominated by intentionally inactive channels.
- Channel grids should ideally render only active channels (or visually mark inactive ones as “closed”).
- Component embedding / per-component reconstruction plots should continue to gate on `decoder_conditioning != "none"`; under curriculum, they may additionally want to visually mark inactive channels.
