# Channel Curriculum — Validation Signals

This document captures what to measure and visualize to validate the “pots” curriculum behavior once active-channel masking + unlocking is implemented.

It is intentionally project-specific and may evolve with experiments.

## Core acceptance criteria

1. **Inactive channels are truly inactive**
   - Inactive channels receive ~zero routing mass and do not influence reconstructions.
   - Monitoring metrics treat inactive channels as “closed” (not “dead”).

2. **Unlock events are observable**
   - `K_active` increases only at explicit unlock events.
   - Reports/plots make unlock epochs obvious.

3. **Kick is effective**
   - Newly unlocked channels receive nontrivial mass during the kick window.
   - After the kick window, mass either stabilizes (specialization) or returns to previous channels (failed split), and this is visible in metrics.

4. **Unlock improves something measurable**
   - Plateau-breaking: reconstruction or validation loss improves after an adaptation window.
   - Specialization improves (clearer channel ownership / lower within-channel mixing), when labels exist.

## Metrics to add (curriculum-specific)

These should appear in history + `summary.json` and be used by plots/reports:

- `curriculum.k_active` (time series)
- `curriculum.unlocked` (event flag per epoch/step)
- `curriculum.kick_active` (flag per epoch/step)
- `curriculum.unlock_count` (summary)
- `curriculum.final_k_active` (summary)

Curriculum-normalized mixture diagnostics:
- “active components” computed within the active set (avoid being dominated by intentionally inactive channels).

## Plot requirements

Minimal curriculum plots:
- `figures/curriculum/k_active_over_time.png` (with unlock markers and kick windows)

Adjust existing mixture plots:
- Mixture evolution (π/usage) should visually distinguish inactive channels (e.g., gray/hidden/annotated).
- Channel latent grids should either render only active channels or clearly mark inactive ones.

## Diagnostics around unlock transitions

Around each unlock event, capture:
- change in reconstruction / validation metric slopes (plateau detection sanity)
- routing mass assigned to the newly unlocked channel over time
- responsibility entropy changes (did routing become more/less decisive?)

## References

- Design contract: `docs/projects/decentralized_latents/channel_curriculum/design_contract.md`
- Implementation plan: `docs/projects/decentralized_latents/channel_curriculum/implementation_mapping.md`
- General experimentation contracts: `docs/development/experimentation_contracts.md`

