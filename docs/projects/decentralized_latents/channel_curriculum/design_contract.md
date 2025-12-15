# Curriculum for Decentralized Latent Channels with Logit‑Mixture Regularization

**Design Contract (math + invariants + policy)**  
**Date:** 2025‑12‑15

## 0) Purpose and supervisor intent

We want a VAE-like system with **decentralized latent spaces** (“channels” / “pots”) such that:

- **Per-sample sparsity:** each datapoint `x` should use **one channel (or very few)**; avoid explaining one sample as a mixture of many channels.
- **Channel coherence:** channels should become internally coherent; labels may span multiple channels, but channels should not mix unrelated modes.
- **Curriculum (“pots”):** start with a small number of active channels; **unlock new channels only when needed**, and stop unlocking when it no longer improves learning.
- **Gaussian-only c-regularization:** avoid Dirichlet / categorical KL pathologies by enforcing peakiness through a Gaussian-mixture prior in logit space.

This document specifies the **mathematical design** and the **curriculum mechanism** independent of code details. (See `docs/projects/decentralized_latents/channel_curriculum/implementation_mapping.md:1` for code pointers.)

---

## 1) Variables and model interface

Let `K_max` be the maximum number of channels supported by the architecture.

For each input `x`:

### Channel selection (discrete index / global routing)

- Encoder outputs **channel logits**:
  - `y(x) ∈ R^{K_max}`
- Responsibilities (soft assignment for diagnostics/usage):
  - `r(x) = softmax(y(x)) ∈ Δ^{K_max-1}`

Optionally, a separate **routing distribution** `s(x)` is used for weighting reconstructions and/or hard selection (e.g., Gumbel‑softmax with straight‑through). In all cases:

- `s(x) ∈ Δ^{K_max-1}`
- For hard routing, `argmax s(x)` selects the channel.

### Per-channel continuous latents

Encoder outputs per-channel posteriors:

`q_φ(z_k | x) = N(μ_k(x), diag(σ_k^2(x)))` for `k = 1..K_max`.

Sample:

`z_k ~ q_φ(z_k | x)`.

### Decoding

Each channel yields a component reconstruction:

`x̂_k = f_θ(z_k, k)`.

The final reconstruction used for loss is the **expectation** weighted by routing `s(x)`:

`x̂ = Σ_{k=1..K_max} s_k(x) x̂_k`.

---

## 2) Base objective (without curriculum)

Per-example loss (minimize):

`L(x) = L_recon(x, x̂) + β_z Σ_{k=1..K_max} KL(q_φ(z_k | x) || N(0, I)) + L_c(x)`.

Where `L_c` is the channel-selection regularization term (defined below).

---

## 3) Channel selection regularization via logit mixture prior (“logit_mog”)

### 3.1 Mixture prior on logits

Define a mixture-of-Gaussians prior on logits `y`:

`p_mix(y) = (1/|A|) Σ_{k∈A} N(y; μ_k, σ^2 I)`, with `μ_k = M e_k`.

Where:

- `e_k` is the k-th standard basis vector in `R^{K_max}`
- `M > 0` controls separation (“peakiness”)
- `σ > 0` controls spread
- `A ⊆ {1..K_max}` is the **active channel set** (curriculum; initially `|A|=1`)

### 3.2 Regularizer term

Regularize deterministic logits `y(x)` via a negative log prior penalty:

`L_{c,mog}(x) = -λ_c log p_mix(y(x))`.

Over a batch, use `E_x[·]`. Implementation should evaluate `log p_mix` using a numerically stable log-sum-exp across mixture components.

Note on naming: the codebase metric name `kl_c_logit_mog` is historical; this term should be interpreted as a negative log prior penalty on deterministic logits (not a categorical KL).

### 3.3 Interpretation and tuning intuition

- After softmax, logits near `M e_k` produce near one-hot responsibilities (“winner-takes-all”).
- Peakiness is primarily governed by `M/σ`:
  - increase `M` or decrease `σ` → harder selection
  - decrease `M` or increase `σ` → softer selection
- `λ_c` controls the strength; it should be **annealed** from 0 to target to avoid early collapse.

### 3.4 Categorical KL (legacy) and “both”

If desired for experiments:

- **categorical**: `L_c = λ_cat KL(q(c|x) || π)`
- **logit_mog** (supervisor-intended): `L_c = L_{c,mog}` and `λ_cat ≈ 0`
- **both**: `L_c = L_{c,mog} + λ_cat KL(q(c|x) || π)` (may require careful weighting)

Global priors on `π` (e.g., Dirichlet MAP) are separate and may remain unchanged.

---

## 4) Curriculum: active channels (“unlocking pots”)

### 4.1 Active channel set

Maintain a runtime active set `A_t` with:

- start: `A_0 = {1}`
- monotone growth: `A_{t+1} = A_t ∪ {k_new}` when unlocking

Keep `K_max` fixed; curriculum operates by masking.

### 4.2 Masking invariants (must hold everywhere)

Inactive channels must not participate in selection.

Define masked logits:

`y'_k(x) = y_k(x)` if `k∈A_t`, else `-∞`.

Then responsibilities/routing are computed from `y'` (softmax or Gumbel‑softmax). Consequences:

- `r(x)` and `s(x)` are distributions over **active channels only** (renormalized).
- Reconstruction weighting uses `s(x)`, so inactive channels contribute 0.
- The logit-mixture prior uses mixture components only over `A_t`.
- During curriculum, usage statistics should be computed from the masked responsibilities (i.e., `r(x)=softmax(y'(x))`) so monitoring/metrics reflect what can actually be selected.

Important nuance (numerical + contract alignment):

- It is acceptable for **routing only** to use `-∞`-masked logits `y'` (softmax/Gumbel-softmax).
- The logit-mixture penalty must be computed on the **finite raw logits** `y(x)` while restricting the **mixture sum** to `k∈A_t`. Do not feed `-∞`-masked logits into Gaussian distance computations `(y - μ_k)^2`.

This guarantees consistency between routing, regularization, and metrics.

---

## 5) Unlock policy (supervisor-aligned default)

This specification supports both a simple **plateau-based** unlock policy (v1) and a more structural **within-channel normality** unlock policy (v2). Plateau is recommended as the first implementation because it is robust and easy to validate.

Supported unlock policies (conceptual):

- **Plateau**: unlock when progress stalls (v1 default).
- **Within-channel normality**: unlock when within-channel latents appear “simple enough” (v2 refinement).

### 5.1 Trigger: plateau-based unlocking

Unlock a new channel when learning progress stalls, e.g.:

- Monitor validation (preferred) or training reconstruction improvement over a window.
- If improvement < threshold for `W` epochs/steps → unlock one new channel.

### 5.2 Stop criterion

Stop unlocking when:

- unlocking yields no measurable improvement after an adaptation window, or
- a maximum active channel count budget is reached.

(Optionally, add a “normality” criterion later, but plateau is the robust first trigger.)

### 5.3 Choosing the new channel index

Use the next unused index by default (simple, deterministic). If a “free channel” concept exists, choose an unused/free channel consistently.

---

## 6) “Kick” mechanism at unlock (required)

Unlocking increases capacity but does not guarantee the optimizer will use it. Therefore, after each unlock, apply a short exploration window of length `T_kick` where one or more of the following holds:

- **Routing softening:** increase temperature and/or disable straight-through to allow exploration.
- **Temporary logit bias:** add a small positive bias to the new channel’s logit to ensure it receives some assignments initially.
- **Regularizer relaxation:** temporarily reduce `λ_c` so mass can migrate.

After `T_kick`, restore normal settings.

---

## 7) Global usage control

During curriculum, avoid objectives that explicitly encourage using many components globally (this conflicts with “keep few pots until needed”).

If a global usage term exists (e.g., entropy of mean responsibilities):

- set it to neutral initially,
- optionally later penalize excessive spread *within the active set* if needed.

The active set itself is the primary control knob for “how many pots exist”.

---

## 8) Diagnostics and acceptance criteria

Track over time (especially around unlock events):

### Per-sample sparsity

- `E[max_k r_k(x)]`
- `E[H(r(x))]`

### Channel usage (within active set)

- `p̂(k) = E_x[r_k(x)]` for `k∈A`
- number of effectively-used channels (e.g., count with `p̂(k) > ε`)

### Unlock effectiveness

After unlock + kick window:

- new channel receives nontrivial mass,
- reconstruction improves or becomes more stable,
- specialization indicators improve (label purity if labels exist; τ-map confidence, etc.).

### Stability

- no immediate collapse to a single channel for all samples,
- no oscillation where the model never commits to a split.

---

## 9) Open knobs (explicitly allowed)

- Whether to use straight-through routing outside kick windows.
- Whether `M` is fixed globally or adjusted with `|A|` to keep a target peak probability.
- Kick mechanism choice (bias vs temperature vs regularizer relaxation).
- Plateau window and thresholds.
