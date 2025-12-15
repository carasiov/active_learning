# Extending the System

> **Purpose** — hands-on guidance for evolving the refactored SSVAE. Read this alongside the [Architecture Guide](architecture.md) and [Implementation Guide](implementation.md) so changes stay consistent with the project’s invariants (see `AGENTS.md` for the knowledge-graph navigation model).

---

## Table of Contents

- [Tutorial 1 · Adding a New Prior](#tutorial-1--adding-a-new-prior)
- [Tutorial 2 · Modular Decoder Variants](#tutorial-2--modular-decoder-variants)
- [Tutorial 3 · τ-Classifier & Trainer Hooks](#tutorial-3--τ-classifier--trainer-hooks)
- [Tutorial 4 · Adding Custom Loss Terms](#tutorial-4--adding-custom-loss-terms)
- [General Extension Checklist](#general-extension-checklist)

The codebase relies on three pillars:

1. **Protocols and registries** (`PriorMode`, decoder builders, Trainer hooks)
2. **Configuration-first design** (`SSVAEConfig` carries every knob)
3. **Factory orchestration** (`ModelFactoryService` wires components, optimizers, and loss plumbing)

Each tutorial below mirrors that structure: define the abstraction, register it, expose config, then cover testing/documentation.

Active project specs that affect extension work:
- Decentralized latents channel curriculum (logit-MoG + “pots”): `docs/projects/decentralized_latents/channel_curriculum/README.md`

---

## Tutorial 1 · Adding a New Prior

### 1. Understand the PriorMode contract
`src/rcmvae/domain/priors/base.py` defines the protocol:

```python
class PriorMode(Protocol):
    def compute_kl_terms(self, encoder_output: EncoderOutput, config) -> Dict[str, jnp.ndarray]: ...
    def compute_reconstruction_loss(self, x_true, x_recon, encoder_output, config) -> jnp.ndarray: ...
    def get_prior_type(self) -> str: ...
    def requires_component_embeddings(self) -> bool: ...
```

`EncoderOutput.extras` is the extensibility surface. Mixture-style priors expect `responsibilities`, π, embeddings, etc. VampPrior adds `pseudo_z_mean` / `pseudo_z_log_var` (see below).

### 2. Implement the prior
- Place the class under `src/rcmvae/domain/priors/`.
- Use the shared loss helpers from `src/rcmvae/application/services/loss_pipeline.py` (e.g., `kl_divergence`, `weighted_reconstruction_loss_mse`, `usage_sparsity_penalty`) so monitoring metrics stay consistent.
- Return every metric key requested by `compute_loss_and_metrics_v2` even if the value is zeroed (`kl_z`, `kl_c`, `dirichlet_penalty`, `component_diversity`, diagnostics).

**Example: VampPrior KL core (excerpt)**
```python
def compute_kl_terms(self, encoder_output, config):
    extras = encoder_output.extras or {}
    pseudo_mean = extras["pseudo_z_mean"]
    pseudo_log_var = extras["pseudo_z_log_var"]
    z_samples = encoder_output.z if self.num_samples_kl == 1 else self._draw_samples(...)

    log_q = self._log_gaussian_prob(z_samples, encoder_output.z_mean, encoder_output.z_log_var)
    log_p = self._compute_log_prior_prob(z_samples, pseudo_mean, pseudo_log_var)
    kl_z = config.kl_weight * jnp.mean(log_q - log_p)
    ...
    return {"kl_z": kl_z, "kl_c": kl_c, "dirichlet_penalty": ..., "component_diversity": ..., ...}
```

### 3. Register & configure
1. Add the class to `PRIOR_REGISTRY` in `src/rcmvae/domain/priors/__init__.py`.
2. Update `ModelFactoryService._create_prior` so the new identifier maps to your constructor.
3. Introduce any config needed in `src/rcmvae/domain/config.py` with validation inside `__post_init__()`. Examples already in tree:
   - `vamp_num_samples_kl`, `vamp_pseudo_lr_scale`, `vamp_pseudo_init_method`
   - `geometric_arrangement`, `geometric_radius`
4. When the prior needs extra tensors (e.g., pseudo-input stats), extend `ModelFactoryService` / `SSVAENetwork` to provide them via `EncoderOutput.extras`. VampPrior re-encodes pseudo-inputs every forward pass and caches the latent stats so the prior stays stateless.

### 4. Training quirks
Special learning-rate needs can be handled in `_scale_vamp_pseudo_gradients()` (see `src/rcmvae/application/services/factory_service.py`). The helper scales `prior/pseudo_inputs` grads before `state.apply_gradients()` so you avoid optimizer forks.

### 5. Tests & docs
Add regression coverage under `tests/` (see `tests/test_vamp_prior.py` and `tests/test_prior_abstraction.py` for patterns). Document the new prior in:
- `docs/theory/implementation_roadmap.md` (Status table)
- `docs/development/implementation.md` (module reference)
- This file (summary of behavior / extension steps)

---

## Tutorial 2 · Modular Decoder Variants

Decoding for mixture-style priors is implemented via a **modular decoder composition**:

`Decoder = Conditioner + Backbone + OutputHead`

This supports component-conditional specialization without hard-coding “component-aware” decoder subclasses.

To add or modify a decoder variant:

1. **Choose the layer you’re extending**
   - **Conditioner** (recommended): `src/rcmvae/domain/components/decoder_modules/conditioning.py` (e.g., CIN / FiLM / concat / noop).
   - **Backbone**: `src/rcmvae/domain/components/decoder_modules/backbones.py` (dense/conv feature stacks).
   - **Output head**: `src/rcmvae/domain/components/decoder_modules/outputs.py` (`StandardHead` or `HeteroscedasticHead`-style).

2. **Wire it into the decoder factory**
   - Update `src/rcmvae/domain/components/factory.py::build_decoder`.
   - Conditioning selection is controlled by `config.decoder_conditioning ∈ {"cin","film","concat","none"}`.
   - Heteroscedastic outputs are controlled by `config.use_heteroscedastic_decoder` (plus `sigma_min`/`sigma_max`).
   - The component embedding size is `config.component_embedding_dim`.

3. **Keep “planned but not implemented” knobs in mind**
   - `config.top_m_gating` and `config.soft_embedding_warmup_epochs` exist in `SSVAEConfig`, but are not currently implemented in the network/loss pipeline. Don’t rely on them without adding explicit code support.

4. **Network integration contract**
   - `SSVAENetwork.__call__()` decodes **all** components, then forms the expected reconstruction using `component_selection` (softmax or Gumbel-softmax depending on config).
   - If `use_gumbel_softmax` is disabled, `component_selection` falls back to the deterministic softmax distribution.
   - For heteroscedastic decoders, recon is `(mean, sigma)` and `extras["recon_per_component"]` becomes `(mean_per_component, sigma_per_component)`.
   - If you change output shapes, ensure the `extras` keys stay consistent; losses, diagnostics, and plots depend on them.

5. **Testing**
   - Follow existing mixture tests (`tests/test_mixture_encoder.py`, `tests/test_mixture_losses.py`, `tests/test_mixture_integration.py`): focus on shape invariants, NaN-safety, and gradient flow through both the latent path and the conditioning path.

---

## Tutorial 3 · τ-Classifier & Trainer Hooks

The τ-classifier replaces the supervised head whenever `config.is_mixture_based_prior()` and `config.use_tau_classifier` are both true. The integration has three pieces:

1. **Stateful helper** — `src/rcmvae/domain/components/tau_classifier.py` tracks soft counts `s_{c,y}` and exposes:
   - `update_counts(responsibilities, labels, labeled_mask)`
   - `get_tau()` for use inside the loss
   - `predict()` / `get_certainty()` for inference paths
2. **Trainer hooks** — `SSVAE._build_tau_loop_hooks()` creates a `TrainerLoopHooks` bundle:
   - `batch_context_fn` injects the current τ matrix into `compute_loss_and_metrics_v2` (so losses can call `tau_classification_loss`)
   - `post_batch_fn` retrieves responsibilities from the forward pass and feeds them back into `TauClassifier.update_counts`
   - `eval_context_fn` mirrors the training context for validation metrics
3. **Loss routing** — `compute_loss_and_metrics_v2` switches between the standard classifier loss and the τ loss depending on whether `tau` and `responsibilities` are present. No special casing is needed once hooks provide the context.

When adding a new mixture-like prior, ensure it supplies `responsibilities` in `EncoderOutput.extras` so the τ workflow continues to operate (VampPrior and Geometric MoG already follow this pattern).

---

## Tutorial 4 · Adding Custom Loss Terms

Custom regularizers should plug into `compute_loss_and_metrics_v2` (or helper functions in `src/rcmvae/application/services/loss_pipeline.py`) so they appear in the trainer metrics. Example: channel-embedding repulsion. Existing patterns: `usage_sparsity_penalty`, `dirichlet_map_penalty`, `l1_penalty`, heteroscedastic NLL helpers.

```python
def channel_repulsion_loss(embeddings: jnp.ndarray, weight: float) -> jnp.ndarray:
    norm = embeddings / jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
    cosine = norm @ norm.T
    off_diag = cosine - jnp.diag(jnp.diag(cosine))
    return weight * jnp.sum(off_diag ** 2)
```

Integration checklist:
1. Compute the term inside the loss function (after recon/kl/classification).
2. Add it to the totals and to the metrics dict (`metrics["channel_repulsion"] = repulsion_loss`).
3. Expose a config knob (`repulsion_weight`) with validation/defaults in `SSVAEConfig`.
4. If the term touches specific parameters (e.g., decoder embeddings), use optimizer masks (`_make_weight_decay_mask`) or gradient scaling helpers as needed.

Existing regularizers you can reference: `usage_sparsity_penalty`, `dirichlet_map_penalty`, heteroscedastic NLL helpers.

---

## General Extension Checklist

1. **Design first** — confirm the change aligns with the theory/architecture docs (per `AGENTS.md` navigation order).
2. **Implement in modules** — add code under `src/rcmvae/domain/**` using the established structure.
3. **Wire through the factory/config** — no feature should require manual instantiation; everything flows through `SSVAEConfig`.
4. **Update tests** — add or extend coverage under `tests/` (unit + integration as appropriate).
5. **Document** — update this guide, the implementation guide, and the roadmap/status tables so the knowledge graph stays accurate.

Following this loop keeps the system coherent while letting us introduce new priors, decoders, loss terms, or trainer hooks without breaking the architectural guarantees.
