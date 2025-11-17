# Model Creation & Configuration Redesign

## Problem Statement

### Current Design Flaw

**Issue discovered**: Models created from homepage use hardcoded `SSVAEConfig()` defaults, but training hub allows editing architectural parameters that require model recreation.

**Root Cause**:
1. Homepage creates models with `SSVAEConfig()` → **default `prior_type="standard"`**
2. Training hub exposes ALL config fields including structural parameters (encoder_type, decoder_type, prior_type, etc.)
3. Changing structural parameters only updates config, NOT the model architecture
4. Results in architecture mismatch: config says "mixture" but model has `DenseEncoder` (3 outputs) instead of `MixtureDenseEncoder` (4 outputs)

**User Impact**:
- Error when training: `ValueError: Mixture responsibilities unavailable`
- Confusing UX: users can change parameters that appear to work but actually break the model
- Warning message about restarting is easy to miss

### Current Flow

```
Homepage → Create Model
  ├─ Inputs: name, num_samples, num_labeled, seed
  ├─ Hardcoded: config = SSVAEConfig()  # prior_type="standard"
  └─ Result: Model with standard prior architecture

Training Hub → Edit Config
  ├─ Exposes: ALL parameters (including structural)
  ├─ Changes: config.prior_type = "mixture"
  ├─ Warning: "Structural changes require restarting"
  └─ Problem: Architecture NOT rebuilt (still has DenseEncoder)

Training → Execute
  └─ Error: Config says mixture but encoder doesn't output component_logits
```

## Solution Design

### Parameter Classification

**Structural Parameters** (require model recreation):
```python
STRUCTURAL_PARAMS = {
    # Architecture
    "encoder_type",           # dense vs conv
    "decoder_type",           # dense vs conv
    "latent_dim",            # changes network layer sizes
    "hidden_dims",           # defines network depth/width

    # Prior (affects encoder output)
    "prior_type",            # standard vs mixture vs vamp vs geometric_mog
    "num_components",        # mixture size
    "component_embedding_dim",  # embedding layer dimensions
    "use_component_aware_decoder",  # changes decoder architecture

    # Loss (affects decoder output)
    "use_heteroscedastic_decoder",  # decoder outputs mean+sigma vs just mean
    "reconstruction_loss",   # affects output activation
}
```

**Modifiable Parameters** (safe to change after creation):
```python
MODIFIABLE_PARAMS = {
    # Training hyperparameters
    "batch_size",
    "learning_rate",
    "max_epochs",
    "patience",
    "monitor_metric",
    "random_seed",
    "dropout_rate",

    # Loss weights
    "recon_weight",
    "kl_weight",
    "label_weight",
    "kl_c_weight",
    "dirichlet_weight",
    "component_diversity_weight",
    "contrastive_weight",

    # Regularization
    "weight_decay",
    "grad_clip_norm",
    "dirichlet_alpha",

    # Training behavior
    "kl_c_anneal_epochs",
    "soft_embedding_warmup_epochs",
    "top_m_gating",
    "mixture_history_log_every",

    # Switches for modifiable features
    "use_tau_classifier",      # doesn't change network architecture
    "tau_smoothing_alpha",
    "learnable_pi",
    "use_contrastive",

    # Heteroscedastic bounds (architecture already set)
    "sigma_min",
    "sigma_max",

    # VampPrior learning params
    "vamp_num_samples_kl",
    "vamp_pseudo_lr_scale",

    # Geometric params (only if prior is geometric_mog)
    "geometric_arrangement",   # WARNING: still structural but for geometric_mog only
    "geometric_radius",
}
```

### Proposed Solution: Two-Stage Model Creation

#### Stage 1: Configuration (New Homepage Flow)

**New homepage form sections**:

```
┌─────────────────────────────────────────────────┐
│  Create New Model                                │
├─────────────────────────────────────────────────┤
│                                                  │
│  Model Name (optional)                          │
│  ┌─────────────────────────────────────────┐   │
│  │ e.g., Baseline Experiment               │   │
│  └─────────────────────────────────────────┘   │
│                                                  │
│  ═══════════════════════════════════════════   │
│  Dataset                                        │
│  ───────────────────────────────────────────   │
│                                                  │
│  Total Samples:      [1024  ▼]                 │
│  Labeled Samples:    [128   ▼]                 │
│  Sampling Seed:      [0     ▼]                 │
│                                                  │
│  ═══════════════════════════════════════════   │
│  Model Architecture (cannot be changed later)   │
│  ───────────────────────────────────────────   │
│                                                  │
│  Prior Type:                                    │
│  ( ) Standard    ( ) Mixture    ( ) Vamp       │
│  ( ) Geometric                                  │
│                                                  │
│  [if mixture/vamp/geometric selected:]          │
│  Number of Components: [10  ▼]                 │
│  ☑ Component-Aware Decoder                     │
│                                                  │
│  Encoder/Decoder:                               │
│  ( ) Dense (MLP)    (•) Convolutional          │
│                                                  │
│  Latent Dimension:   [2     ▼]                 │
│                                                  │
│  [Advanced Architecture ▼]                      │
│    Hidden Layers: [256,128,64]                 │
│    ☐ Heteroscedastic Decoder                   │
│    Component Embedding Dim: [auto]             │
│    Reconstruction Loss: ( ) BCE (•) MSE        │
│                                                  │
│  ═══════════════════════════════════════════   │
│  Quick Training Presets (can be changed later)  │
│  ───────────────────────────────────────────   │
│                                                  │
│  Learning Rate:  [0.001  ▼]                    │
│  Batch Size:     [128    ▼]                    │
│  Max Epochs:     [200    ▼]                    │
│                                                  │
│  [Cancel]                   [Create Model]     │
│                                                  │
└─────────────────────────────────────────────────┘
```

**Key Features**:
1. **Clear separation**: "Architecture (cannot be changed later)" vs "Quick Training Presets"
2. **Conditional fields**: Show mixture-specific options only when mixture prior selected
3. **Collapsible advanced**: Most users can use defaults, advanced users can expand
4. **Presets**: Common configs like "Baseline (Standard)", "Mixture (10 components)", "VampPrior"

#### Stage 2: Training Hub (Restricted Config)

**Modified training hub config sections**:

```python
# config_metadata.py changes:

_FIELD_SPECS_TRAINING_HUB = (
    # Only include MODIFIABLE_PARAMS
    # Remove all structural params from this list
)

_FIELD_SPECS_CREATION = (
    # Include ALL params for model creation
    # Used only in homepage modal
)

# Add immutable display for structural params
_STRUCTURAL_DISPLAY_SPECS = (
    # Read-only display of current architecture
    FieldSpec(
        key="prior_type",
        label="Prior Type",
        display_only=True,  # New flag
        ...
    ),
    # ... other structural params
)
```

**Training hub layout**:
```
┌─────────────────────────────────────────────────┐
│  Training Configuration                          │
├─────────────────────────────────────────────────┤
│                                                  │
│  Current Architecture (read-only)               │
│  ───────────────────────────────────────────   │
│  Prior: Mixture (10 components)                 │
│  Encoder/Decoder: Convolutional                 │
│  Latent Dimension: 2                            │
│  Component-Aware Decoder: Yes                   │
│                                                  │
│  ═══════════════════════════════════════════   │
│  Training & Data                                │
│  ───────────────────────────────────────────   │
│  [editable fields...]                           │
│                                                  │
│  ═══════════════════════════════════════════   │
│  Prior & Mixture Tuning                         │
│  ───────────────────────────────────────────   │
│  [editable fields...]                           │
│                                                  │
└─────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Prevent the Bug (Quick Fix) ✅ DONE

**Status**: Already completed with the fallback fix in `training_callbacks.py`

```python
# Gracefully handle architecture mismatches
except ValueError as e:
    error_msg = str(e).lower()
    if "mixture" in error_msg or "responsibilities" in error_msg:
        logger.warning(f"Mixture data unavailable ({e}), falling back to non-mixture prediction")
        latent_val, recon_val, preds, cert = model.predict_batched(data)
        return latent_val, recon_val, preds, cert, None, None
    raise
```

This prevents crashes but doesn't fix the root cause.

### Phase 2: Immediate Warning (Quick Win)

**Goal**: Make it impossible to miss when users create architecture mismatches

**Changes**:
1. Add prominent warning in training hub when structural params differ from model
2. Disable "Train Model" button when architecture mismatch detected
3. Show clear instructions to create new model

```python
# In UpdateConfigCommand.execute()

architecture_mismatch = any(
    getattr(current_config, field) != getattr(new_config, field)
    for field in STRUCTURAL_PARAMS
)

if architecture_mismatch:
    # Don't just warn - block training
    updated_model = current_model.with_architecture_warning(True)
    return state.with_active_model(updated_model), (
        "⚠️ ARCHITECTURE MISMATCH: Cannot train with these changes. "
        "These parameters require creating a new model:\n"
        f"Changed: {', '.join(changed_fields)}\n\n"
        "Please create a new model with the desired architecture."
    )
```

### Phase 3: Enhanced Model Creation (Main Fix)

**Goal**: Allow users to configure architecture at creation time

**Tasks**:

1. **Create `ModelCreationWizard` component** (`pages/model_creation.py`)
   - Multi-section form with structural params
   - Conditional rendering (show mixture params only for mixture prior)
   - Preset configurations
   - Validation before creation

2. **Update `CreateModelCommand`**
   ```python
   @dataclass
   class CreateModelCommand(Command):
       name: Optional[str] = None
       num_samples: int = 1024
       num_labeled: int = 128
       seed: Optional[int] = None
       config: SSVAEConfig = None  # NEW: Accept full config instead of using defaults

       def execute(self, state: AppState, services: Any):
           config = self.config or SSVAEConfig()  # Use provided or default
           request = CreateModelRequest(
               name=self.name or "Unnamed Model",
               config=config,  # Use the provided config
               dataset_total_samples=self.num_samples,
               dataset_seed=rng_seed,
           )
           # ... rest of creation logic
   ```

3. **Split config metadata**
   ```python
   # config_metadata.py

   def get_structural_field_specs() -> Tuple[FieldSpec, ...]:
       """Fields that require model recreation (for creation wizard)."""
       return _STRUCTURAL_SPECS

   def get_modifiable_field_specs() -> Tuple[FieldSpec, ...]:
       """Fields safe to change after creation (for training hub)."""
       return _MODIFIABLE_SPECS

   def get_structural_display_specs() -> Tuple[FieldSpec, ...]:
       """Read-only display of current architecture (for training hub)."""
       return _STRUCTURAL_DISPLAY_SPECS
   ```

4. **Update training hub layout** (`training_hub.py`)
   - Add read-only architecture summary at top
   - Remove structural params from editable config
   - Show warning if mismatch detected (shouldn't happen with new flow)

5. **Update homepage**
   - Replace simple modal with full creation wizard
   - OR: Keep simple modal, add "Configure Architecture" button that opens wizard
   - Provide quick presets: "Standard", "Mixture (10)", "VampPrior", "Custom"

### Phase 4: Migration & Testing

**Goal**: Ensure existing models continue to work

1. **Audit existing models**: Script to check for config/architecture mismatches
2. **Migration warning**: Show notice on dashboard load about new creation flow
3. **Testing**:
   - Create model with each prior type
   - Verify training works without errors
   - Verify config changes only allow modifiable params
   - Test architecture mismatch detection

## Configuration Presets

Provide common starting points:

```python
PRESETS = {
    "standard_baseline": SSVAEConfig(
        prior_type="standard",
        encoder_type="conv",
        decoder_type="conv",
        latent_dim=2,
    ),
    "mixture_10": SSVAEConfig(
        prior_type="mixture",
        num_components=10,
        encoder_type="conv",
        decoder_type="conv",
        latent_dim=2,
        use_component_aware_decoder=True,
        use_tau_classifier=True,
        learnable_pi=True,
    ),
    "vamp_prior": SSVAEConfig(
        prior_type="vamp",
        num_components=10,
        encoder_type="conv",
        decoder_type="conv",
        latent_dim=2,
        vamp_pseudo_init_method="kmeans",
    ),
    "geometric_diagnostic": SSVAEConfig(
        prior_type="geometric_mog",
        num_components=10,
        encoder_type="conv",
        decoder_type="conv",
        latent_dim=2,
        geometric_arrangement="circle",
    ),
}
```

## Migration Notes

### For Existing Models

Models created before this change:
- Continue to work as-is (no breaking changes)
- May have config/architecture mismatches if users changed structural params
- Show warning banner: "This model was created before architecture locking. Consider recreating for full compatibility."

### For Users

Clear communication:
```
🎉 New Feature: Full Architecture Control

You can now configure your model's architecture (prior type, encoder/decoder,
latent dimension, etc.) when creating a model.

⚠️ Important: Architecture cannot be changed after creation. To use a different
architecture, create a new model.

✅ Training parameters (learning rate, batch size, loss weights, etc.) can still
be modified anytime.
```

## Alternative Approaches Considered

### Alternative 1: Allow Architecture Changes with Auto-Rebuild

**Pros**: More flexible, users don't need to create new models
**Cons**:
- Complex implementation (need to preserve training state, checkpoints)
- Risk of data loss
- Confusing UX (what happens to trained weights?)

**Decision**: Too complex, better to keep creation separate

### Alternative 2: Live Architecture Migration

**Pros**: Seamless transition between architectures
**Cons**:
- Extremely complex (transfer learning between different architectures)
- Lossy (can't perfectly preserve all information)
- High risk of bugs

**Decision**: Not worth the complexity for this use case

### Alternative 3: Keep Current Flow, Just Add Validation

**Pros**: Minimal changes
**Cons**:
- Doesn't solve the root problem (users still can't configure architecture)
- Still confusing UX (why can't I change prior type?)

**Decision**: Doesn't address user needs adequately

## Success Criteria

1. ✅ No more `ValueError: Mixture responsibilities unavailable` errors
2. ✅ Users can create models with any prior type from homepage
3. ✅ Training hub clearly shows which parameters are modifiable
4. ✅ Architecture mismatches are impossible (or caught with clear error)
5. ✅ Existing models continue to work
6. ✅ UX is clear about what can/cannot be changed

## Open Questions

1. **Should we allow "cloning" a model with different architecture?**
   - Clone button: "Create new model with same dataset but different architecture"
   - Could preserve dataset + labels, just rebuild model

2. **How to handle VampPrior pseudo-input initialization?**
   - Need data before creating model
   - Could: Create model → Load data → Initialize pseudo-inputs → Save
   - OR: Defer pseudo-input init to first training run

3. **Should geometric params be locked too?**
   - `geometric_arrangement` and `geometric_radius` technically don't change architecture
   - But they fundamentally change the prior distribution
   - Recommendation: Lock them (treat as structural)

4. **Dashboard restart requirement**
   - Current warning says "restart dashboard" for structural changes
   - With new design, this becomes impossible
   - Should remove this warning path entirely

## References

- Current issue: use_cases/dashboard/core/commands.py:611-648
- Default config: src/rcmvae/domain/config.py:168 (prior_type="standard")
- Config metadata: use_cases/dashboard/core/config_metadata.py
- Homepage: use_cases/dashboard/pages/home.py:427-616
- Training hub: use_cases/dashboard/pages/training_hub.py
