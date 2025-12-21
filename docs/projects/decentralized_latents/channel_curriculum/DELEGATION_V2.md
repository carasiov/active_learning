# Delegation: Channel Curriculum V2 (Kick Adoption + Observability)

This document is self-contained and intended for the next agent implementing fixes to the **channel curriculum (“pots”)** for decentralized latent channels.

## Canonical project docs (read first)

- `docs/projects/decentralized_latents/channel_curriculum/README.md`
- `docs/projects/decentralized_latents/channel_curriculum/high_level_context.md`
- `docs/projects/decentralized_latents/channel_curriculum/design_contract.md`
- `docs/projects/decentralized_latents/channel_curriculum/implementation_mapping.md`
- `docs/projects/decentralized_latents/channel_curriculum/validation_signals.md`
- `docs/projects/decentralized_latents/channel_curriculum/logit_mog_regularizer.md`

## Goal

Fix the curriculum failure mode where **newly unlocked channels are not adopted** (routing collapses to a single channel) and improve observability so we can verify curriculum dynamics without guesswork.

## Current implementation state (already landed)

### Curriculum + masking (end-to-end)

- **Routing masking**
  - `src/rcmvae/domain/network.py`: `SSVAENetwork.__call__(..., active_mask=...)` masks routing logits to `-jnp.inf` for inactive channels and computes both `responsibilities` and `component_selection` from the masked logits.
- **Logit-MoG invariant (“raw vs masked logits”)**
  - `src/rcmvae/domain/priors/mixture.py`: the logit-MoG regularizer uses **finite** `extras["raw_logits"]` and masks mixture-sum terms via `-inf` at the *log-prob* level (never feeds `-inf` into distance computations).
- **Curriculum hooks**
  - `src/rcmvae/application/curriculum/hooks.py`: always injects `active_mask`; during kick window it also sets:
    - `gumbel_temperature` override (kick temp)
    - `straight_through_gumbel=False` (soft routing during kick)
- **Trainer integration**
  - `src/rcmvae/application/services/training_service.py`: calls `loop_hooks.on_epoch_end_fn(epoch, flat_metrics)` after val metrics are computed.
- **Post-training artifact correctness (partially fixed)**
  - `src/rcmvae/application/services/experiment_service.py`: captures `final_active_mask`, regenerates diagnostics with it, and passes it to `predict_batched(...)` for predictions and mixture outputs.
  - `src/rcmvae/application/services/diagnostics_service.py`: supports `active_mask` for responsibilities/usage collection.

### Config

- Example config: `use_cases/experiments/configs/mnist_curriculum.yaml`
- Curriculum unlock monitor is recon-only in code and config:
  - `src/rcmvae/application/curriculum/controller.py` defaults to `val_reconstruction_loss`
  - `use_cases/experiments/configs/mnist_curriculum.yaml` uses `monitor: val_reconstruction_loss`

## Evidence: V1 failure persists (run log + artifacts)

Run:
- `use_cases/experiments/results/mnist_curriculum_test__mix10-dec-gbl_head_film__20251216_145412/`

Observed behavior:
- Unlock triggered at epoch index **78**:
  - log: `[Curriculum] Epoch 78: Unlocked channel 2 ...`
  - meaning: `k_active` increased **1 → 2** (new channel index is `k_active-1 = 1`, 0-based)
- Kick window active for 5 epochs with `straight_through_gumbel=False`, `temperature=5.0`.
- Routing still collapsed:
  - `summary.json` shows:
    - `mixture.component_usage ≈ [0.9998201, 0.0001799, 0, ...]`
    - `K_eff ≈ 1.0017`
    - `responsibility_confidence_mean ≈ 0.99982`
    - `final_component_entropy ≈ 0.0017`
  - Channel latent grid plot: only channel 0 has points; the new channel remains empty.

## Diagnosis (what’s going wrong)

1. **Inertia / logit lock-in**
   - Training for ~78 epochs with `k_active=1` teaches the encoder to output extremely confident logits for channel 0.
   - The `logit_mog` regularizer likely reinforces this by pulling `raw_logits` toward the active component’s mean in logit space.
2. **No diversity pressure**
   - Current runs have `model.component_diversity_weight` unset → defaults to `0.0`.
   - With no explicit penalty for collapse, “always channel 0” is a stable solution.
3. **Soft kick alone is insufficient**
   - Disabling ST allows gradients, but only if the forward allocates *meaningful* probability to the new channel.
   - If `P(new)` is extremely small, gradients into the new channel are too weak to recover.

## Critical correctness/observability gap to fix

Even if training is curriculum-correct, **some artifacts are not**:

- `src/rcmvae/application/callbacks/mixture_tracking.py` computes `usage_history.npy` via `state.apply_fn(..., training=False)` **without `active_mask`**.
  - This produces non-zero usage for “inactive” channels pre-unlock and makes time-series plots misleading.
- Multiple mixture plotters in `src/infrastructure/visualization/mixture/plots.py` call:
  - `model.predict_batched(..., return_mixture=True)` and/or `model._apply_fn(... training=False ...)`
  - also **without `active_mask`**, so mixture plots may violate curriculum invariants.
- `src/infrastructure/visualization/curriculum/plots.py::curriculum_usage_plotter` currently skips because `history["component_usage"]` is never populated.
  - But `usage_history.npy` exists and is the correct source for this plot (once it becomes curriculum-aware).

## V2 implementation: required changes (ordered)

### 1) Stronger kick: routing logit bias during kick (PRIMARY)

**Objective:** Guarantee the newly unlocked channel receives non-trivial routing mass during kick, so it can train and compete.

Spec:
- Add config: `curriculum.kick.logit_bias: float` (e.g., `10.0`).
- During kick, bias **only the newly unlocked channel** (index `k_active-1`) by adding +bias to `routing_logits`.

Key constraints:
- Apply bias to **`routing_logits` only** (post-mask), not to `raw_logits`.
  - Do **not** break the logit-MoG invariant in `src/rcmvae/domain/priors/mixture.py`.
- Bias must not resurrect inactive channels:
  - If mask sets inactive logits to `-inf`, then `-inf + bias = -inf` remains safe.

Suggested plumbing (minimal surface area):
- `src/rcmvae/application/curriculum/controller.py`
  - Track “newly unlocked channel index” in state (e.g., `last_unlocked_idx`).
  - Provide a method to return the bias vector during kick (shape `[K_max]`).
- `src/rcmvae/application/curriculum/hooks.py`
  - Include `routing_logit_bias` in `batch_context_fn` during kick.
  - Keep `straight_through_gumbel=False` and temperature override.
- `src/rcmvae/application/services/loss_pipeline.py`
  - Add an optional arg `routing_logit_bias: jnp.ndarray | None` to `compute_loss_and_metrics_v2`.
- `src/rcmvae/application/services/factory_service.py`
  - Thread `routing_logit_bias` through `_model_forward(...)` into the network apply call.
- `src/rcmvae/domain/network.py`
  - Add `routing_logit_bias: jnp.ndarray | None = None` to `SSVAENetwork.__call__`.
  - After masking and before softmax/Gumbel, do:
    - `routing_logits = routing_logits + routing_logit_bias[None, :]` (broadcast)
  - Keep `extras["raw_logits"] = component_logits` unchanged.

Acceptance signal (for this step alone):
- Immediately after the first unlock + kick window, the new channel’s mean usage should exceed a small threshold (e.g. `> 0.02`) and not instantly vanish.

### 2) Prevent early lock-in: attenuate/disable Logit-MoG when `k_active <= 1`

**Objective:** Avoid baking in a “channel 0 forever” solution during the long `k_active=1` phase.

Spec (simple and effective):
- In `src/rcmvae/domain/priors/mixture.py` (logit_mog branch), if curriculum `active_mask` is present and `k_active <= 1`, set `kl_c_logit_mog = 0`.
  - Optionally also attenuate during kick (e.g., scale by 0.1) to avoid fighting exploration.

### 3) Add traffic shaping: diversity reward (CONFIG CHANGE)

**Objective:** Make “use only channel 0” suboptimal.

Spec:
- Set `model.component_diversity_weight: -0.05` in `use_cases/experiments/configs/mnist_curriculum.yaml`.
  - Negative weight **rewards** higher entropy of batch-mean usage (see `src/rcmvae/domain/config.py` docstring).

Also recommended (config knobs):
- Increase `curriculum.kick.epochs` from 5 → **15–30** (cold components need time).
- Consider reducing `model.c_logit_prior_weight` while iterating (e.g. `0.02–0.05`) and/or increasing `c_logit_prior_sigma` (e.g. `2.0`) if exploration is still blocked.

### 4) Make artifacts curriculum-correct (must for debugging)

#### 4A) Curriculum-aware mixture history

Problem:
- `src/rcmvae/application/callbacks/mixture_tracking.py` currently logs usage without `active_mask`.

Spec:
- Ensure `MixtureHistoryTracker` uses the **current epoch’s** `active_mask` when calling `state.apply_fn(...)`.

Suggested implementation approach:
- In `src/rcmvae/application/services/training_service.py`, store the eval context dict on the trainer each epoch (e.g., `self._last_eval_context = eval_context`).
- In `MixtureHistoryTracker.on_epoch_end`, read `active_mask = getattr(trainer, "_last_eval_context", {}).get("active_mask")` and pass it to `state.apply_fn(..., active_mask=active_mask)`.

#### 4B) Curriculum-aware mixture plotters

Problem:
- Several plotters call `predict_batched`/`_apply_fn` without `active_mask`.

Spec:
- Thread `context.final_active_mask` into mixture plot calls so responsibilities/usage respect curriculum.
- Places to inspect/update:
  - `src/infrastructure/visualization/plotters.py` (wrapper layer)
  - `src/infrastructure/visualization/mixture/plots.py` (actual plotting helpers)

#### 4C) Curriculum usage plot should run

Problem:
- `src/infrastructure/visualization/curriculum/plots.py::curriculum_usage_plotter` skips because history lacks `component_usage`.

Spec:
- If `history["component_usage"]` is missing, load:
  - `usage_history.npy` and `tracked_epochs.npy` from the diagnostics dir (`model.last_diagnostics_dir` / same folder as `component_usage.npy`).
- Use `curriculum_history` to mask channels that are not active yet (as it already does conceptually).
- Output should be `figures/curriculum/active_usage_over_time.png`.

### 5) OPTIONAL: Early stopping patience reset after unlock

This is more invasive and may not be needed (the run above did not stop immediately after unlock), but it’s conceptually valid.

If implementing:
- Extend the hook contract so curriculum can signal “unlock happened”.
- Reset the `EarlyStoppingTracker` in `src/rcmvae/application/services/training_service.py` when that signal is seen.

### 6) OPTIONAL: Latent snapshots around unlock/kick

Desired snapshots (per unlock U):
- `U-1` (pre-unlock)
- `U + kick_epochs` (end of kick)
- `U + 10` (stabilization)

This likely requires adding a callback that can run plotting against the current model state and save images with epoch stamps.

## Verification plan

### Unit tests (minimum)

Add tests under `tests/` for:
1. **Logit bias application**
   - Bias affects routing probabilities for the newly unlocked channel.
   - Bias does not affect inactive channels (mask still forces 0 prob).
   - `raw_logits` stored in extras remains unchanged by bias.
2. **Logit-MoG gating**
   - With `active_mask` implying `k_active=1`, `kl_c_logit_mog` is zero (or attenuated as specified).
3. **Mixture tracking uses active_mask**
   - Ensure usage_history is ~0 for inactive channels when `k_active=1`.

### Experiment run (manual)

Validate config:
- `poetry run python use_cases/experiments/run_experiment.py --config use_cases/experiments/configs/mnist_curriculum.yaml --validate-only`

Run:
- `poetry run python use_cases/experiments/run_experiment.py --config use_cases/experiments/configs/mnist_curriculum.yaml`

Inspect:
- `use_cases/experiments/results/<run_id>/summary.json`
  - `curriculum.final_k_active`, `curriculum.unlock_epochs`
  - `mixture.component_usage`, `mixture.K_eff`, `mixture.active_components`
- `use_cases/experiments/results/<run_id>/artifacts/diagnostics/<checkpoint>/usage_history.npy`
- `use_cases/experiments/results/<run_id>/figures/curriculum/active_usage_over_time.png` (should exist after step 4C)
- `use_cases/experiments/results/<run_id>/figures/mixture/channel_latents/channel_latents_grid.png` (new channel should show non-empty alpha)

Success criteria (first unlock):
- Newly opened channel usage `> 0.02` shortly after unlock (stretch goal `> 0.05`) and remains non-trivial after kick ends.
- `K_eff` increases meaningfully above 1 (target: `> 1.2` in early iterations).
- Routing entropy increases and does not collapse back to ~0 immediately.

## Known operational issues

- Plotting/inference may emit GPU allocator “out of memory” warnings but still complete.
  - If it becomes a hard failure, reduce plotter sample sizes or prediction batch size in the relevant plotting functions.

