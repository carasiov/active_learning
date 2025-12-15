# Decentralized Latent Spaces — Project Evolution

This project area covers the evolution from a **shared/global latent space** to **decentralized latent channels** (“pots”), and the current push to add a **channel-unlocking curriculum** on top.

The canonical “what we’re building now” specification lives in `docs/projects/decentralized_latents/channel_curriculum/`.

## Where To Start (Current Spec)

- `docs/projects/decentralized_latents/channel_curriculum/README.md` — reading order + glossary.
- `docs/projects/decentralized_latents/channel_curriculum/high_level_context.md` — rationale (why logit-MoG + pots).
- `docs/projects/decentralized_latents/channel_curriculum/design_contract.md` — the normative invariants + curriculum policy.
- `docs/projects/decentralized_latents/channel_curriculum/implementation_mapping.md` — how it maps onto the codebase + experiments.
- `docs/projects/decentralized_latents/channel_curriculum/logit_mog_regularizer.md` — logit-MoG mechanism note (already implemented).

## North Star Docs (Theory + Architecture)

- Conceptual intent: `docs/theory/conceptual_model.md`
- Mathematical contract: `docs/theory/mathematical_specification.md`
- Current system architecture patterns: `docs/development/architecture.md`
- Experiment/validation contracts: `docs/development/experimentation_contracts.md`
- Status snapshot: `docs/theory/implementation_roadmap.md`

## Glossary (Project Terms)

- `component_logits`: encoder logits `y(x) ∈ R^{K_max}` (pre-softmax).
- Responsibilities: `r(x) = softmax(y(x))` (deterministic; for monitoring/usage).
- Routing / selection: `s(x)` used for reconstruction weighting (softmax or Gumbel-softmax; may be straight-through).
- `K_max`: architectural maximum number of channels (fixed tensor shapes).
- Active set `A_t`: which channels are “open” at curriculum stage `t`; `|A_t| = K_active`.
- “Pots”: the channels; curriculum grows the number of open pots over training.
- Logit-MoG: Gaussian mixture prior/penalty on logits to make `q(c|x)` peaky.

## Evolution (How We Got Here)

### Phase 1 — Decentralized Latent Concept

Initial direction: replace a single global latent `z` with **K per-channel latents** `z_k` and a channel variable `c`. The intended behavior is: a datapoint should be explained primarily by one channel (pot), and the continuous latent captures variation within that channel.

### Phase 2 — Infrastructure Shipped (Decentralized Layout + Routing + Conditioning)

We implemented the infrastructure to support decentralized latents:

- encoder emits per-channel latents (`[B, K, D]`) and `component_logits`
- routing selects/weights components (softmax or Gumbel-softmax, optional straight-through)
- decoder supports component conditioning via embeddings (CIN/FiLM/Concat variants)
- diagnostics + visualization produce per-channel plots and mixture evolution artifacts

### Phase 3 — Regularization Reality

The original “prior on q(c|x) via simplex/Dirichlet” framing proved awkward to control and stabilize. In practice, the codebase uses:

- categorical KL to π (optional), plus
- Dirichlet MAP penalty on π (optional), plus
- a batch-usage entropy term (sign-controlled via `component_diversity_weight`)

This provides global mixture-shape control, but does not fully enforce the supervisor’s “one pot per datapoint” preference by itself.

### Phase 4 — Logit-MoG Added (Per-sample Peakiness)

To directly enforce per-sample sparsity in a smooth way, we added the **logit-mixture (logit-MoG) regularizer**: a negative log prior penalty on raw logits that encourages axis-aligned, winner-takes-all responsibilities.

This aligns strongly with the supervisor’s “avoid parts-based mixing across channels” requirement.

### Phase 5 — Curriculum Reframed as Hierarchical Splitting (“Pots”)

Even with peaky routing, letting all `K_max` channels compete from the start can cause uncontrolled fragmentation and makes “how many pots are needed” hard to manage.

The supervisor’s curriculum framing is a **hierarchical splitting process**:

- start with one open pot
- unlock a new empty pot when progress stalls or the current pots become “internally simple enough”
- apply a short “kick” so the new pot gets used
- stop unlocking when it no longer improves learning

This requires a missing capability today: a runtime **active channel set** `A_t` that restricts routing/regularization/metrics to the open pots.

## Current State (What Exists vs What’s Missing)

Implemented today:

- decentralized latent layout, routing, component-conditioned decoding, diagnostics/plots
- logit-MoG regularizer (per-sample peakiness on logits)

Missing for the curriculum target state:

- active-set masking (`A_t` / `K_active`) applied consistently to routing + monitoring + logit-MoG mixture sum
- unlock trigger policy (plateau/normality) and unlock event tracking
- a “kick” mechanism at unlock (temperature kick is the easiest first implementation)
- curriculum-specific metrics and plots (e.g., `K_active` vs epoch + unlock markers)

## Code Pointers (Common Entry Points)

- Network forward / routing: `src/rcmvae/domain/network.py`
- Config surface: `src/rcmvae/domain/config.py`
- Mixture prior + logit-MoG term: `src/rcmvae/domain/priors/mixture.py`
- Loss pipeline: `src/rcmvae/application/services/loss_pipeline.py`
- Training loop hooks: `src/rcmvae/application/services/training_service.py`
- Experiments runner: `use_cases/experiments/run_experiment.py`
