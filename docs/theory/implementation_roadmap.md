# Implementation Roadmap

> **Purpose** — describe the state of the rearchitected SSVAE, highlight what is already production-ready, and call out the next focused efforts.  
> **Theory:** [Conceptual Model](conceptual_model.md) · [Math Spec](mathematical_specification.md)  
> **Implementation:** [Architecture Guide](../development/architecture.md) · [Implementation Guide](../development/implementation.md)

---

## Current Snapshot · Nov 2025

| Pillar | Status | Key files / notes |
|--------|--------|-------------------|
| Mixture prior with entropy + Dirichlet controls | ✅ shipping | `src/rcmvae/domain/priors/mixture.py`, `src/rcmvae/application/services/loss_pipeline.py` (usage penalty + Dirichlet) |
| Modular decoder architecture (conditioning + backbone + output) | ✅ shipping | `src/rcmvae/domain/components/decoder_modules/`, modular decoders in `decoders.py`, `decoder_conditioning` config option |
| Decoder conditioning (CIN, FiLM, Concat, None) | ✅ shipping | `conditioning.py`: CIN (recommended), FiLM, ConcatConditioner, NoopConditioner |
| τ-classifier latent workflow (responsibility-based) | ✅ shipping | `src/rcmvae/domain/components/tau_classifier.py`, now enabled for **all** mixture-based priors |
| Heteroscedastic decoder + weighted loss | ✅ needs tuning knobs only | `src/rcmvae/domain/components/decoders.py`, `src/rcmvae/application/services/loss_pipeline.py` |
| VampPrior (pseudo-input learning, MC-KL) | ✅ shipping | `src/rcmvae/domain/priors/vamp.py`, network now caches pseudo stats & supports pseudo-LR scaling |
| Geometric MoG (diagnostic/curriculum prior) | ✅ shipping | `src/rcmvae/domain/priors/geometric_mog.py` |
| OOD scoring via `r × τ` | 📋 ready once experiment wiring added |
| Dynamic label addition / active learning loop | 📋 design ready; needs workflow + UX |

Legend: ✅ production-ready · ⚠️ needs tuning · 📋 planned/ready-to-wire

---

## Completed Pillars

### Modular Decoder Architecture
- **What**: composition-based decoders: `conditioner + backbone + output_head`, supporting all conditioning × output combinations.  
- **Why**: avoids class explosion, enables all feature combinations, and makes testing/extension module-scoped.  
- **Where**: `src/rcmvae/domain/components/decoder_modules/{conditioning,backbones,outputs}.py`; composed via `ModularConvDecoder`/`ModularDenseDecoder` in `decoders.py`.

### Decoder Conditioning
- **What**: Unified `decoder_conditioning` config option replaces legacy flags.
- **Options**:
  - `"cin"` — Conditional Instance Normalization (recommended): normalizes then modulates
  - `"film"` — FiLM: scale + shift without normalization
  - `"concat"` — Concatenate projected embedding
  - `"none"` — Pass-through (for standard/vamp priors)
- **Valid priors**: `mixture`, `geometric_mog` support all; `vamp`, `standard` use `"none"` only.
- **Migration**: The legacy `use_component_aware_decoder` flag is deprecated and has no effect. Use `decoder_conditioning` instead.

### Decentralized Latent Layout
- **What**: Support for `latent_layout="decentralized"` (K independent latents) vs "shared" (single global latent).
- **Mechanism**: Encoder outputs `[B, K, D]`; Gumbel-Softmax routing selects active path; Decoder processes active component.
- **Status**: ✅ shipping (core infrastructure complete).
- **Validation**: Validated in `mix10-dir-dec-gbl_tau_film-het` experiment (see `refactor_validation_report.md`).

### Mixture Prior with Diversity Controls
- **What**: `MixtureGaussianPrior` handles `KL_z`, `KL_c`, optional Dirichlet MAP on π, and usage-entropy “diversity reward/punishment”.  
- **Extras**: learnable π (`config.learnable_pi`) with gradient masking when disabled; metrics surfaced via `compute_loss_and_metrics_v2`.  
- **Diagnostics**: callbacks + `DiagnosticsCollector` export component usage, entropies, π histories.

### τ-Classifier & Latent Workflow
- **What**: responsibility-based classifier substitutes the head: accumulates soft counts → τ-map → `p(y|x)=Σ_c q(c|x)τ_{c,y}`.  
- **New in this revision**: any **mixture-based prior** (`mixture`, `vamp`, `geometric_mog`) gets τ hooks automatically (`SSVAE.config.is_mixture_based_prior()`), so VampPrior experiments can stay latent-only.  
- **Files**: `src/rcmvae/domain/components/tau_classifier.py`, trainer hooks in `src/rcmvae/application/model_api.py` and `src/rcmvae/application/services/training_service.py`.

### Heteroscedastic Decoder
- **What**: decoder predicts `(mean, σ)`; losses handle either per-sample (standard) or per-component (mixture) heteroscedasticity.  
- **Status**: stable, just needs experiment-level tuning of `sigma_min/max` and loss scaling.  
- **Files**: decoders + `heteroscedastic_reconstruction_loss()` utilities.

### VampPrior Subsystem
- **What**: pseudo-input prior with Monte Carlo KL. Network now re-encodes pseudo-inputs every forward pass and caches `pseudo_z_mean`/`pseudo_z_log_var` in `EncoderOutput.extras`, so the prior remains stateless.  
- **Training hygiene**: `vamp_pseudo_lr_scale` scales gradients for `params['prior']['pseudo_inputs']` inside the JIT train step (see `_scale_vamp_pseudo_gradients()` in `rcmvae/application/services/factory_service.py`).  
- **Features**: random or k-means pseudo init, optional multi-sample KL, uniform π for now.  
- **Status**: production-ready for spatial visualization + component-free decoding.

### Geometric Mixture of Gaussians
- **What**: fixed centers (circle/grid) with analytical KL; acts as a curriculum/debug prior.  
- **Safeguards**: validation enforces grid square counts and warns about induced topology.  
- **Status**: shipping but flagged “diagnostic only”.

---

## Tooling & Infrastructure

- **Factory + Prior registry** — `ModelFactoryService` builds networks, optimizers (with gradient masks), and PriorMode instances; new priors just register via `src/rcmvae/domain/priors/__init__.py`.
- **Loss pipeline** — `compute_loss_and_metrics_v2` delegates reconstruction + KL to the active prior and merges τ losses, keeping trainer logic agnostic.  
- **Diagnostics** — `DiagnosticsCollector` + callbacks capture π/usage histories, component entropies, per-component reconstructions, and latent dumps for 2-D runs.  
- **Experiments** — configs live under `use_cases/experiments/configs/`; runners log to timestamped result dirs, feeding dashboards/plots.

---

## Future Focus Areas

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

- **Config / validation** — `src/rcmvae/domain/config.py`
- **Network + prior parameters** — `src/rcmvae/domain/network.py`
- **Priors** — `src/rcmvae/domain/priors/{standard,mixture,vamp,geometric_mog}.py`
- **Losses** — `src/rcmvae/application/services/loss_pipeline.py`
- **Trainer / hooks** — `src/rcmvae/application/services/training_service.py`
- **Tau classifier** — `src/rcmvae/domain/components/tau_classifier.py`
- **Diagnostics** — `src/rcmvae/application/services/diagnostics_service.py`
- **Experiments** — `use_cases/experiments/…`

Use this roadmap with the architecture + implementation guides to stay aligned with the project’s invariants while iterating.
