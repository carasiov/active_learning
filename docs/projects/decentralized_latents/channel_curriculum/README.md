---
status: current
updated: 2025-12-26
---

# Channel Curriculum ("Pots")

Curriculum-based channel unlocking for decentralized latent VAE.

## Quick Start

```bash
# Run curriculum experiment
poetry run python use_cases/experiments/run_experiment.py \
  --config use_cases/experiments/configs/mnist_curriculum_multi_unlock.yaml

# Results in: use_cases/experiments/results/<run_id>/
```

## Current Status (2025-12-26)

**V2 kick mechanism implemented.** Successfully breaks single-channel monopoly.

**Known issue:** Stable 2-channel coalition forms; channels 3+ cannot compete.

**Clarification (2025-12-26):** Supervisor's "L1 sparsity" = Logit-MoG (already implemented).
- Logit-MoG: per-sample peakiness (each r(x) → one-hot)
- Entropy reward: global diversity (spread usage across channels)
- These are **complementary**, not alternatives.

**Next step:** Run simultaneous training baseline to determine if 2-channel coalition is curriculum-specific.

See `FINDINGS.md` for details.

---

## Documents

| File | Purpose |
|------|---------|
| `SPEC.md` | Mathematical specification, supervisor decisions, invariants |
| `IMPLEMENTATION.md` | Code locations, config structure, data flow |
| `FINDINGS.md` | Experimental results and analysis |
| `archive/` | Historical documents (completed delegations, superseded docs) |

---

## Glossary

| Term | Definition |
|------|------------|
| `component_logits` | Encoder outputs y(x) ∈ R^K (pre-softmax) |
| Responsibilities | r(x) = softmax(y(x)) — deterministic soft assignment |
| Routing | s(x) — distribution used for decoder weighting (softmax or Gumbel) |
| Active channels | A_t ⊆ {1..K_max} — channels "open" at curriculum stage t |
| K_max | Architectural maximum channels (fixed) |
| K_active | Currently open channels (runtime state, \|A_t\|) |
| Kick | Exploration window after unlock (logit bias + temperature) |

---

## Key Configs

| Config | Purpose |
|--------|---------|
| `mnist_curriculum.yaml` | Standard curriculum (plateau unlock) |
| `mnist_curriculum_kick_diag.yaml` | Diagnostic: forced early unlock |
| `mnist_curriculum_multi_unlock.yaml` | Extended: 250 epochs, all channels |
| `mnist_simultaneous_k20.yaml` | **Baseline:** All K=20 from start, no curriculum |

---

## North Star References

- Conceptual intent: `docs/theory/conceptual_model.md`
- Mathematical contract: `docs/theory/mathematical_specification.md`
- Architecture patterns: `docs/development/architecture.md`
