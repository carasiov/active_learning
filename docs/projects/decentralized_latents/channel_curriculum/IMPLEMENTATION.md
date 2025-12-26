---
status: current
updated: 2025-12-23
purpose: Code locations and implementation details
---

# Channel Curriculum — Implementation Guide

## 1. Current State

Curriculum is **implemented and functional**. The kick mechanism (logit bias + temperature) successfully breaks single-channel monopoly.

**Known issue:** 2-channel coalition phenomenon. See `FINDINGS.md`.

---

## 2. Code Locations

### Core Curriculum

| Component | Location |
|-----------|----------|
| Controller (state, unlock logic) | `src/rcmvae/application/curriculum/controller.py` |
| Hooks (batch/eval context injection) | `src/rcmvae/application/curriculum/hooks.py` |
| Config schema | `src/rcmvae/domain/config.py` (curriculum section) |

### Routing & Masking

| Component | Location |
|-----------|----------|
| Logit masking | `src/rcmvae/domain/network.py` — `SSVAENetwork.__call__` |
| Gumbel-softmax routing | `src/rcmvae/domain/network.py:187` |
| Logit bias application | `src/rcmvae/domain/network.py` (routing_logit_bias arg) |

### Regularization

| Component | Location |
|-----------|----------|
| Logit-MoG computation | `src/rcmvae/domain/priors/mixture.py` |
| Loss aggregation | `src/rcmvae/application/services/loss_pipeline.py` |
| Diversity term | `src/rcmvae/application/services/loss_pipeline.py:207` |

### Training Integration

| Component | Location |
|-----------|----------|
| Hook wiring | `src/rcmvae/application/services/training_service.py` |
| Experiment orchestration | `src/rcmvae/application/services/experiment_service.py` |
| Artifact generation | `src/rcmvae/application/services/diagnostics_service.py` |

---

## 3. Config Structure

Example curriculum config (`use_cases/experiments/configs/mnist_curriculum.yaml`):

```yaml
curriculum:
  enabled: true
  k_active_init: 1

  unlock:
    policy: plateau
    monitor: val_reconstruction_loss
    patience_epochs: 5
    min_delta: 1.0
    cooldown_epochs: 2

  kick:
    enabled: true
    epochs: 15
    gumbel_temperature: 5.0
    logit_bias: 5.0
```

Key model settings that interact with curriculum:

```yaml
model:
  num_components: 10              # K_max
  use_straight_through_gumbel: true
  c_regularizer: logit_mog
  c_logit_prior_weight: 0.1
  c_logit_prior_mean: 5.0
  c_logit_prior_sigma: 1.0
  component_diversity_weight: -0.05  # negative = reward entropy
```

---

## 4. Data Flow

### Training Step

```
batch_context_fn() → active_mask, kick settings
    ↓
compute_loss_and_metrics_v2(..., active_mask, routing_logit_bias, ...)
    ↓
SSVAENetwork.__call__(..., active_mask, routing_logit_bias, ...)
    ↓
  1. Compute raw logits y(x)
  2. Apply mask: routing_logits = where(active, y, -inf)
  3. Apply kick bias: routing_logits += logit_bias (if in kick)
  4. Compute responsibilities from routing_logits
  5. Store extras["raw_logits"] = y (unmasked, for logit-MoG)
    ↓
MixtureGaussianPrior.compute_kl_terms(raw_logits, active_mask)
    ↓
  - Logit-MoG on raw_logits with mixture sum over active set only
```

### Unlock Decision

```
on_epoch_end(epoch, metrics)
    ↓
controller.maybe_unlock(metrics["val_reconstruction_loss"])
    ↓
  - Check plateau: improvement < min_delta for patience_epochs?
  - If yes: k_active += 1, start kick window
    ↓
batch_context_fn now returns updated active_mask + kick settings
```

---

## 5. Key Implementation Details

### Masking Invariant

```python
# Routing uses -inf masking
routing_logits = jnp.where(active_mask, component_logits, -jnp.inf)

# Logit-MoG uses finite raw logits with restricted sum
# In mixture.py: logsumexp over active indices only
log_probs = []
for k in active_indices:
    log_probs.append(log_gaussian(raw_logits, mu_k, sigma))
log_p_mix = logsumexp(log_probs) - log(len(active_indices))
```

### Kick Bias Application

```python
# Only during kick window, only to newly unlocked channel
if routing_logit_bias is not None:
    routing_logits = routing_logits + routing_logit_bias[None, :]
```

### Logit-MoG Gating

```python
# Disabled when k_active <= 1 to prevent early lock-in
if k_active <= 1:
    kl_c_logit_mog = 0.0
```

---

## 6. Testing

```bash
# Validate config
poetry run python use_cases/experiments/run_experiment.py \
  --config use_cases/experiments/configs/mnist_curriculum.yaml \
  --validate-only

# Run experiment
poetry run python use_cases/experiments/run_experiment.py \
  --config use_cases/experiments/configs/mnist_curriculum.yaml
```

Verify in results:
- `summary.json`: `curriculum.final_k_active`, `mixture.K_eff`, `mixture.component_usage`
- `figures/mixture/channel_latents/channel_latents_grid.png`
- `figures/mixture/model_evolution.png`
