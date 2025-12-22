# Channel Curriculum (“Pots”)

Short entry point for the decentralized-latents curriculum work (“pots”).

## Overview

We’re building a mixture VAE with decentralized latent channels and a **channel-unlocking curriculum**:

- **Per-sample sparsity**: each datapoint should route to ~one channel.
- **Controlled growth (“pots”)**: start with one active channel; unlock more only when needed.
- **Logit-mixture regularization**: encourage peaky `q(c|x)` via a Gaussian mixture prior on encoder logits (logit-MoG).

## Glossary

- `component_logits`: encoder outputs `y(x) ∈ R^{K_max}` (pre-softmax).
- Responsibilities: `r(x) = softmax(y(x))` (diagnostics / usage statistics; deterministic).
- Routing / selection: `s(x)` used to weight reconstructions (softmax or Gumbel-softmax; can be straight-through).
- Active channels / pots: the subset `A_t ⊆ {1..K_max}` that is “open” during curriculum stage `t`.
- `K_max`: the architectural maximum number of channels (fixed).
- `K_active`: how many channels are currently open (runtime state; `|A_t|`).

## What to read first

1. `CONTEXT.md` — **comprehensive project briefing** (supervisor decisions, math characterization, priorities, gotchas)
2. `high_level_context.md` — intuition and rationale
3. `design_contract.md` — formal invariants + curriculum policy
4. `implementation_mapping.md` — how this maps onto the codebase and experiment workflow

## North Star References

These documents are the stable “north star” for intent, math, and architectural conventions. The curriculum docs in this folder may temporarily be more specific than theory docs while the curriculum is being implemented/validated.

- Conceptual intent: `docs/theory/conceptual_model.md`
- Mathematical contract: `docs/theory/mathematical_specification.md`
- Architecture patterns (hooks/registries): `docs/development/architecture.md`
- Experiment/validation contracts: `docs/development/experimentation_contracts.md`
- Status snapshot: `docs/theory/implementation_roadmap.md`

## Related docs

- `CONTEXT.md` — comprehensive project briefing for agents
- `DELEGATION_V2.md` — current implementation task spec
- `logit_mog_regularizer.md` — the existing logit-MoG regularizer (what's already implemented)
- `validation_signals.md` — what to measure/plot to validate curriculum behavior
