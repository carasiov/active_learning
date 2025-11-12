# Implementation Roadmap

> **Purpose** — describe the state of the rearchitected SSVAE, highlight what is already production-ready, and call out the next focused efforts.  
> **Theory:** [Conceptual Model](conceptual_model.md) · [Math Spec](mathematical_specification.md)  
> **Implementation:** [Architecture Guide](../development/architecture.md) · [Implementation Guide](../development/implementation.md)

---

## Current Snapshot · Nov 2025

| Pillar | Status | Key files / notes |
|--------|--------|-------------------|
| Mixture prior with entropy + Dirichlet controls | ✅ shipping | `src/ssvae/priors/mixture.py`, `src/training/losses.py` (usage penalty + Dirichlet) |
| Component-aware decoder (dense/conv, heteroscedastic variants) | ✅ shipping | `src/ssvae/components/decoders.py` |
| τ-classifier latent workflow (responsibility-based) | ✅ shipping | `src/ssvae/components/tau_classifier.py`, now enabled for **all** mixture-based priors |
| Heteroscedastic decoder + weighted loss | ✅ needs tuning knobs only | `src/ssvae/components/decoders.py`, `src/training/losses.py` |
| VampPrior (pseudo-input learning, MC-KL) | ✅ shipping | `src/ssvae/priors/vamp.py`, network now caches pseudo stats & supports pseudo-LR scaling |
| Geometric MoG (diagnostic/curriculum prior) | ✅ shipping | `src/ssvae/priors/geometric_mog.py` |
| OOD scoring via `r × τ` | 📋 ready once experiment wiring added |
| Dynamic label addition / active learning loop | 📋 design ready; needs workflow + UX |

Legend: ✅ production-ready · ⚠️ needs tuning · 📋 planned/ready-to-wire

---

## Completed Pillars

### Component-Aware Decoder
- **What**: learns component embeddings `e_c` that specialize the decoder (parallel paths for `z` and `e_c`, merge mid-way). Supports dense/conv + heteroscedastic heads.  
- **Why**: lets mixture components acquire functional roles without forcing latent separation.  
- **Where**: `ComponentAware*Decoder` classes; configured via `use_component_aware_decoder`.  
- **Validation**: ablations show improved recon quality and interpretable per-component outputs.

### Mixture Prior with Diversity Controls
- **What**: `MixtureGaussianPrior` handles `KL_z`, `KL_c`, optional Dirichlet MAP on π, and usage-entropy “diversity reward/punishment”.  
- **Extras**: learnable π (`config.learnable_pi`) with gradient masking when disabled; metrics surfaced via `compute_loss_and_metrics_v2`.  
- **Diagnostics**: callbacks + `DiagnosticsCollector` export component usage, entropies, π histories.

### τ-Classifier & Latent Workflow
- **What**: responsibility-based classifier substitutes the head: accumulates soft counts → τ-map → `p(y|x)=Σ_c q(c|x)τ_{c,y}`.  
- **New in this revision**: any **mixture-based prior** (`mixture`, `vamp`, `geometric_mog`) gets τ hooks automatically (`SSVAE.config.is_mixture_based_prior()`), so VampPrior experiments can stay latent-only.  
- **Files**: `src/ssvae/components/tau_classifier.py`, trainer hooks in `ssvae/models.py` and `training/trainer.py`.

### Heteroscedastic Decoder
- **What**: decoder predicts `(mean, σ)`; losses handle either per-sample (standard) or per-component (mixture) heteroscedasticity.  
- **Status**: stable, just needs experiment-level tuning of `sigma_min/max` and loss scaling.  
- **Files**: decoders + `heteroscedastic_reconstruction_loss()` utilities.

### VampPrior Subsystem
- **What**: pseudo-input prior with Monte Carlo KL. Network now re-encodes pseudo-inputs every forward pass and caches `pseudo_z_mean`/`pseudo_z_log_var` in `EncoderOutput.extras`, so the prior remains stateless.  
- **Training hygiene**: `vamp_pseudo_lr_scale` scales gradients for `params['prior']['pseudo_inputs']` inside the JIT train step (see `_scale_vamp_pseudo_gradients()` in `ssvae/factory.py`).  
- **Features**: random or k-means pseudo init, optional multi-sample KL, uniform π for now.  
- **Status**: production-ready for spatial visualization + component-free decoding.

### Geometric Mixture of Gaussians
- **What**: fixed centers (circle/grid) with analytical KL; acts as a curriculum/debug prior.  
- **Safeguards**: validation enforces grid square counts and warns about induced topology.  
- **Status**: shipping but flagged “diagnostic only”.

---

## Tooling & Infrastructure

- **Factory + Prior registry** — `SSVAEFactory` builds networks, optimizers (with gradient masks), and PriorMode instances; new priors just register via `ssvae/priors/__init__.py`.
- **Loss pipeline** — `compute_loss_and_metrics_v2` delegates reconstruction + KL to the active prior and merges τ losses, keeping trainer logic agnostic.  
- **Diagnostics** — `DiagnosticsCollector` + callbacks capture π/usage histories, component entropies, per-component reconstructions, and latent dumps for 2-D runs.  
- **Experiments** — configs live under `use_cases/experiments/configs/`; runners log to timestamped result dirs, feeding dashboards/plots.

---

## Next Focus Areas

1. **OOD & Active Learning Loop**
   - Wire the existing metrics (`max_c r_c`, τ certainty) into experiment scripts for acquisition and reporting.
   - Surface `get_ood_score()` and responsibility entropy in the CLI/dashboard.

2. **Dynamic Label Addition**
   - Build workflow that monitors free channels (low usage + low τ confidence) and spawns new labels/components when thresholds hit.
   - Update τ-classifier persistence / checkpointing to handle label-space expansion.

3. **Prior Research Tracks**
   - Learnable π for VampPrior / hybrid priors (requires extending PriorMode interface with optional state).  
   - Flow-based or hierarchical priors once metrics confirm VampPrior + mixture cover the needed regimes.

---

## File Reference

- **Config / validation** — `src/ssvae/config.py`
- **Network + prior parameters** — `src/ssvae/network.py`
- **Priors** — `src/ssvae/priors/{standard,mixture,vamp,geometric_mog}.py`
- **Losses** — `src/training/losses.py`
- **Trainer / hooks** — `src/training/trainer.py`
- **Tau classifier** — `src/ssvae/components/tau_classifier.py`
- **Diagnostics** — `src/ssvae/diagnostics.py`
- **Experiments** — `use_cases/experiments/…`

Use this roadmap with the architecture + implementation guides to stay aligned with the project’s invariants while iterating.
