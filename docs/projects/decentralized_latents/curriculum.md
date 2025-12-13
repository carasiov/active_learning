# Channel Unlocking Curriculum

This document describes the **channel unlocking curriculum** feature for progressive training of mixture-of-VAEs models. The curriculum gradually exposes mixture components during training, preventing collapse and enabling stable learning.

## Overview

Channel unlocking is a training strategy where mixture components are progressively "unlocked" (made available for routing) rather than all being active from the start. This prevents:

1. **Mode collapse**: Early training with all K components often leads to a few components dominating
2. **Premature routing decisions**: Hard routing before latent spaces are structured causes instability
3. **Dead components**: Components that never receive gradients become permanently inactive

## Unlock Modes

The curriculum supports two unlock modes:

### 1. Epoch-Based Unlock (`curriculum_unlock_mode: "epoch"`)

Components are unlocked at fixed epoch intervals:

```yaml
curriculum_enabled: true
curriculum_unlock_mode: "epoch"
curriculum_start_k_active: 1      # Start with 1 component
curriculum_unlock_every_epochs: 10  # Unlock new component every 10 epochs
curriculum_max_k_active: 10       # Maximum active components
```

**Schedule example** (K=10, start=1, unlock_every=10):
- Epochs 0-9: k_active = 1
- Epochs 10-19: k_active = 2
- Epochs 20-29: k_active = 3
- ... and so on until k_active = 10

### 2. Trigger-Based Unlock (`curriculum_unlock_mode: "trigger"`)

Components are unlocked when training plateaus AND latent distributions have settled:

```yaml
curriculum_enabled: true
curriculum_unlock_mode: "trigger"
curriculum_start_k_active: 1
curriculum_min_epochs_per_channel: 10  # Minimum epochs before unlock allowed
curriculum_plateau_window_epochs: 8    # Look at last 8 epochs
curriculum_plateau_min_improvement: 0.005  # <0.5% improvement = plateau
curriculum_plateau_metric: "reconstruction_loss"
curriculum_normality_threshold: 0.5    # Normality score threshold (lower = stricter)
```

**Trigger conditions** (ALL must be met):
1. **Minimum epochs**: At least `curriculum_min_epochs_per_channel` epochs since last unlock
2. **Plateau detected**: Relative improvement in reconstruction loss over the window is below threshold
3. **Normality OK**: Active latent channels are close to N(0,I)

#### Normality Score

The normality score measures how close the posterior q(z|x) is to N(0,I):

```
S_k = mean(||μ_k||²) + mean(|exp(logvar_k) - 1|²)
```

- Score of 0 = perfect N(0,I)
- Higher scores indicate deviation from standard normal
- Only computed over **active** channels

## Migration Window

After each unlock event, there's an optional "migration window" period where routing is softened to allow examples to migrate to the newly unlocked component:

```yaml
curriculum_migration_epochs: 3      # Duration of migration window
curriculum_soft_routing_during_migration: true  # Disable straight-through
curriculum_temp_boost_during_migration: 1.5    # Multiply Gumbel temp
curriculum_logit_mog_scale_during_migration: 0.5  # Scale logit-MoG weight
```

**During migration window**:
- Straight-through is disabled (soft routing)
- Gumbel temperature is boosted (softer selections)
- Logit-MoG weight is scaled down (less peakiness pressure)

This allows gradients to flow to the newly unlocked component without hard routing decisions.

## Implementation Details

### Masking Mechanism

Inactive components are masked out in the forward pass:

```python
# In network.py: apply_curriculum_mask()
if k_active < K:
    # Mask weights for inactive components
    mask = jnp.where(jnp.arange(K) < k_active, 1.0, 0.0)
    weights = weights * mask
    weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-10)
```

### Top-M Gating (Optional)

For large K, you can limit computation to top-M components within the active set:

```yaml
top_m_gating: 3  # Only use top 3 components by weight (0 = use all)
```

This applies **after** curriculum masking, so effective_m = min(top_m, k_active).

### History Tracking

The training service tracks curriculum metrics:

- `k_active`: Number of active components per epoch
- `in_migration_window`: Whether migration window is active
- `normality_score`: Latent normality proxy (trigger mode)
- `plateau_detected`: Whether plateau condition was met (trigger mode)
- `plateau_improvement`: Relative improvement value (trigger mode)
- `unlock_triggered`: Whether unlock was triggered (trigger mode)

### Visualization

The curriculum metrics plotter generates:

1. **k_active evolution**: Step plot showing unlocks over time
2. **Migration window indicator**: Shows when soft routing is active
3. **Normality score** (trigger mode): Latent normality over time
4. **Plateau diagnostics** (trigger mode): Improvement and trigger events

Output: `figures/curriculum/curriculum_metrics.png`

## Configuration Reference

### Required Settings

| Parameter | Type | Description |
|-----------|------|-------------|
| `curriculum_enabled` | bool | Enable curriculum learning |
| `curriculum_start_k_active` | int | Initial number of active components (≥1) |
| `curriculum_max_k_active` | int | Maximum active components (≤num_components) |

### Epoch Mode Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `curriculum_unlock_every_epochs` | int | 5 | Epochs between unlocks |

### Trigger Mode Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `curriculum_min_epochs_per_channel` | int | 0 | Minimum epochs at each k_active before unlock (0=disabled) |
| `curriculum_plateau_window_epochs` | int | 5 | Window for plateau detection |
| `curriculum_plateau_min_improvement` | float | 0.01 | Minimum improvement to not be plateau |
| `curriculum_plateau_metric` | str | "reconstruction_loss" | Metric for plateau detection |
| `curriculum_normality_threshold` | float | 1.0 | Max normality score for unlock |

### Migration Window Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `curriculum_migration_epochs` | int | 0 | Duration of migration window (0=disabled) |
| `curriculum_soft_routing_during_migration` | bool | true | Disable ST during migration |
| `curriculum_temp_boost_during_migration` | float | 1.0 | Gumbel temp multiplier |
| `curriculum_logit_mog_scale_during_migration` | float | 1.0 | Logit-MoG weight scale |

## Example Configs

See the ready-to-run example configurations:

- [`configs/curriculum_epoch.yaml`](../../../use_cases/experiments/configs/curriculum_epoch.yaml) - Epoch-based unlock
- [`configs/curriculum_trigger.yaml`](../../../use_cases/experiments/configs/curriculum_trigger.yaml) - Trigger-based unlock

## Best Practices

1. **Start with epoch mode**: Simpler to debug and understand
2. **Use migration window**: Helps components stabilize after unlock
3. **Monitor normality scores**: High scores indicate latent collapse
4. **Tune plateau threshold**: 0.01-0.05 works well for most cases
5. **Balance unlock pace**: Too fast → collapse; too slow → underfitting
6. **Use min epochs constraint**: Set `curriculum_min_epochs_per_channel: 10` in trigger mode to prevent rapid consecutive unlocks before channels develop
