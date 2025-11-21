# Decentralized Latents: Implementation Specification

> **Purpose**: This document specifies the implementation of **decentralized latent spaces** (mixture-of-VAEs). The core infrastructure change is refactoring the decoder to support modular composition, enabling all necessary feature combinations (FiLM, heteroscedastic, Gumbel-Softmax) for the decentralized architecture.

---

## Table of Contents

1. [Architectural Context](#architectural-context)
2. [Current State Audit](#current-state-audit)
3. [Design Decision: Modular Composition](#design-decision-modular-composition)
4. [Implementation Tasks](#implementation-tasks)
5. [Validation & Experiments](#validation--experiments)
6. [Success Criteria](#success-criteria)
7. [References](#references)

---

## Architectural Context

### Vision: Decentralized Latent Spaces (Mixture of VAEs)

We are transitioning from a **shared-latent mixture-of-Gaussians** to a **decentralized-latent mixture-of-VAEs** architecture.

#### What We Had (Shared Latent)

```
Encoder: x → [component_logits, z_mean, z_log_var, z]
         where z ∈ R^[B,D] (single shared latent)

Prior:   p(z) = Σ_k π_k N(0,I)  (mixture of identical Gaussians)

Decoder: p(x|z,c) receives:
         - z (shared across all components)
         - e_c (component embedding)
         - Method: Concatenation [z; e_c] → shared weights
```

**Limitation**: All components encode to the same $z$ space, limiting specialization.

#### What We're Building (Decentralized Latent)

```
Encoder: x → [component_logits, {z_mean_k, z_log_var_k, z_k}_{k=1}^K]
         where z_k ∈ R^[B,K,D] (one latent per component)

Structure: K independent channels, each dimension d=2
           - Class information → channel index c
           - Intra-class variation → continuous latent z_c

Selection: c ~ Gumbel-Softmax(q(c|x))  (differentiable discrete sampling)
           z_c ~ N(μ_c, σ_c)           (continuous sampling for selected channel)

Decoder: p(x|z_c, c) receives:
         - z_c (selected component's latent)
         - c (component index or embedding e_c)
         - Method: FiLM modulation
           Features = γ_c · Norm(Backbone(z_c)) + β_c
```

**Key Insight**: Each component learns its own encoding strategy. Channel 1 might encode "digit identity" while Channel 2 encodes "rotation angle."

### Loss Function (All Channels)

```python
# Reconstruction: using sampled channel c
recon_loss = -log p(x | z_c, c)

# Latent KL: sum over ALL channels (encourages unused channels to stay prior-like)
kl_z = Σ_{k=1}^K KL(q(z_k|x) || N(0,I))

# Component KL: regularize component selection
kl_c = KL(q(c|x) || Dirichlet(α))

# Total
loss = recon_loss + β_z * kl_z + β_c * kl_c
```

**Critical**: We compute KL for **all** $K$ channels, even though only one is used for reconstruction. This prevents unused channels from diverging.

### Curriculum Training Strategy

To prevent "compromise" solutions where channels don't specialize:

**Phase 1: Warm-up**
- High reconstruction weight, low KL
- Establishes basic encoding capability

**Phase 2: KL Annealing**
- Gradually increase β_z and β_c
- Encourages latent regularization

**Phase 3: Channel Curriculum** (Key Innovation)
- Start with K_active = 1 (only one channel unlocked)
- Gradually unlock: 1 → 2 → ... → K channels
- When unlocking channel k:
  - **Boost KL weight** briefly to force usage
  - **Mask logits** for locked channels (force q(c|x)=0 for c > K_active)
  - **Soft reconstruction warm-up** (optional)

**Visualization Goal**: K separate 2D scatter plots, one per channel, showing clean cluster separation.

### Complementary Features

1. **Gumbel-Softmax Routing** (`use_gumbel_softmax`)
   - Differentiable hard selection: c ~ Gumbel-Softmax(logits, τ)
   - Straight-through estimator: forward is one-hot, backward is soft
   - Temperature annealing: τ starts high, decays to make routing crisper

2. **FiLM Conditioning** (`use_film_decoder`)
   - Component embedding e_c → MLP → (γ, β) affine parameters
   - Modulates decoder features: γ ⊙ Norm(h) + β
   - **Stronger specialization than concatenation**

3. **Heteroscedastic Output** (`use_heteroscedastic_decoder`)
   - Decoder predicts (mean, σ) per image
   - Captures aleatoric uncertainty
   - NLL loss: ||x - mean||²/(2σ²) + log σ
   - σ clamped to [σ_min, σ_max] for stability

---

## Current State Audit

### What Exists

From [`src/rcmvae/domain/components/decoders.py`](../../../src/rcmvae/domain/components/decoders.py):

**10 Decoder Classes** (combinatorial explosion):
- `DenseDecoder`, `HeteroscedasticDenseDecoder`
- `ConvDecoder`, `HeteroscedasticConvDecoder`
- `ComponentAwareDenseDecoder`, `ComponentAwareHeteroscedasticDenseDecoder`
- `ComponentAwareConvDecoder`, `ComponentAwareHeteroscedasticConvDecoder`
- `FiLMDenseDecoder`, `FiLMConvDecoder`

**Missing**: `FiLMHeteroscedasticConvDecoder`, `FiLMHeteroscedasticDenseDecoder`

### Critical Issues

1. **Silent Override in Factory** ([`factory.py:99-103`](../../../src/rcmvae/domain/components/factory.py#L99-L103)):
   ```python
   use_film_decoder = (
       config.prior_type in {"mixture", "geometric_mog"} and
       config.use_film_decoder and
       not use_heteroscedastic  # ← SILENT OVERRIDE
   )
   ```
   If user sets both flags, FiLM is silently disabled.

2. **Duplicated Logic in Network** ([`network.py:281-285`](../../../src/rcmvae/domain/network.py#L281-L285)):
   Same override logic duplicated in forward pass.

3. **Unreachable Error** ([`factory.py:147-148`](../../../src/rcmvae/domain/components/factory.py#L147-L148)):
   ```python
   if use_film_decoder and use_heteroscedastic:
       raise ValueError("FiLM conv decoder does not yet support heteroscedastic outputs.")
   ```
   This error never fires due to line 102 override.

4. **Tau Classifier Silent Fallback** ([`loss_pipeline.py:428-433`](../../../src/rcmvae/application/services/loss_pipeline.py#L428-L433)):
   If `use_tau_classifier=True` but `tau` is None, silently falls back to standard classifier.

5. **Config Validation Gap** ([`config.py`](../../../src/rcmvae/domain/config.py)):
   Allows `use_film_decoder=True` + `use_heteroscedastic_decoder=True` without warning.

### Current Uncertainties (Not Bugs)

1. **Decentralized Latents**: Code exists, but unknown if it produces per-component specialization
2. **Gumbel-Softmax**: Unknown if straight-through gradients flow correctly
3. **FiLM Effectiveness**: No empirical comparison vs. concatenation
4. **Feature Combos**: Untested which combinations work scientifically

### Documentation Drift

Existing docs ([`conceptual_model.md`](../../theory/conceptual_model.md), [`architecture.md`](../../development/architecture.md)) describe the **pre-decentralized** architecture. They need updates for:
- Decentralized latent formulation
- Gumbel-Softmax sampling
- FiLM conditioning math
- Curriculum training strategy

---

## Design Decision: Modular Composition

### Rejected Alternatives

**Option A: Universal Decoder (Conditionals)**
- Single class with `if use_film: ...` branches
- ❌ Con: Hard to test features in isolation
- ❌ Con: Mixed concerns (conditioning + backbone + output)

**Option B: Static Dispatch (JAX JIT)**
- Use `static_argnums` to compile separate paths
- ❌ Con: Complex typing, harder to debug

**Option C: Minimal Fix (Add Missing Classes)**
- Add `FiLMHeteroscedasticConvDecoder` class
- ❌ Con: Now 12 classes; next feature → 24 classes

### Chosen: Modular Composition

**Pattern**: Separate modules for orthogonal concerns, composed at runtime.

```
Decoder = Conditioner + Backbone + OutputHead
```

**Modules**:
1. **Conditioners**: `FiLMLayer`, `ConcatConditioner`, `NoopConditioner`
2. **Backbones**: `ConvBackbone`, `DenseBackbone`
3. **Output Heads**: `HeteroscedasticHead`, `StandardHead`

**Benefits**:
- ✅ **Testable**: Each module tested independently
- ✅ **Extensible**: New conditioning method = new conditioner (no class explosion)
- ✅ **AI-agent friendly**: Clear boundaries, obvious extension points
- ✅ **Idiomatic JAX/Flax**: Composable modules are the Flax pattern
- ✅ **Scalable**: 10+ features without 1000 classes

**Trade-off**: More abstraction layers (3 modules instead of 1 monolithic class), but the cost is worth it for a research framework.

---

## Implementation Tasks

### Task 1: Extract Decoder Modules

#### 1.1 Create Module Directory

```bash
mkdir -p src/rcmvae/domain/components/decoder_modules
touch src/rcmvae/domain/components/decoder_modules/__init__.py
touch src/rcmvae/domain/components/decoder_modules/conditioning.py
touch src/rcmvae/domain/components/decoder_modules/backbones.py
touch src/rcmvae/domain/components/decoder_modules/outputs.py
```

#### 1.2 Extract Conditioning Modules

**File**: `src/rcmvae/domain/components/decoder_modules/conditioning.py`

**Extract from existing code**:

- [ ] **`FiLMLayer`** (extract from [`FiLMConvDecoder`](../../../src/rcmvae/domain/components/decoders.py))
  - Signature: `__call__(self, z, component_embedding) -> features`
  - Logic: Generate γ, β from embedding; modulate z features
  - Test: Shape correctness, gradient flow

- [ ] **`ConcatConditioner`** (extract from [`ComponentAwareConvDecoder`](../../../src/rcmvae/domain/components/decoders.py))
  - Signature: `__call__(self, z, component_embedding) -> features`
  - Logic: `jnp.concatenate([z, component_embedding], axis=-1)`
  - Test: Shape correctness

- [ ] **`NoopConditioner`** (new, for standard decoders)
  - Signature: `__call__(self, z, component_embedding) -> z`
  - Logic: Return z unchanged
  - Test: Identity operation

#### 1.3 Extract Output Modules

**File**: `src/rcmvae/domain/components/decoder_modules/outputs.py`

- [ ] **`HeteroscedasticHead`** (extract from [`HeteroscedasticConvDecoder`](../../../src/rcmvae/domain/components/decoders.py))
  - Signature: `__call__(self, features) -> (mean, sigma)`
  - Logic: Two conv heads (mean, log_sigma); sigma clamping
  - Test: Shape, sigma bounds, gradient flow

- [ ] **`StandardHead`** (extract from [`ConvDecoder`](../../../src/rcmvae/domain/components/decoders.py))
  - Signature: `__call__(self, features) -> mean`
  - Logic: Single conv head with sigmoid
  - Test: Shape, output range [0,1]

#### 1.4 Extract Backbone Modules

**File**: `src/rcmvae/domain/components/decoder_modules/backbones.py`

- [ ] **`ConvBackbone`** (extract conv transpose layers from existing decoders)
  - Signature: `__call__(self, z) -> features`
  - Architecture: ConvTranspose → ReLU → ... → feature map
  - Test: Shape transformation

- [ ] **`DenseBackbone`** (extract from dense decoders)
  - Signature: `__call__(self, z) -> features`
  - Architecture: Dense → ReLU → ... → flattened features
  - Test: Shape transformation

**Deliverable**: 7 module files with unit tests for each.

---

### Task 2: Build Modular Decoders

**File**: `src/rcmvae/domain/components/decoders.py` (add to existing file)

#### 2.1 ModularConvDecoder

```python
class ModularConvDecoder(nn.Module):
    """Composable convolutional decoder for mixture-of-VAEs.
    
    Supports all combinations of:
    - Conditioning: "film" | "concat" | "none"
    - Output: "standard" | "heteroscedastic"
    
    Args:
        conditioning_method: How to incorporate component embeddings
        output_type: Standard mean-only or heteroscedastic (mean, sigma)
        latent_dim: Latent dimension
        output_hw: Output image shape (H, W)
        component_embedding_dim: Dimension of component embeddings (if used)
        sigma_min, sigma_max: Variance clamps for heteroscedastic output
    """
    conditioning_method: str
    output_type: str
    latent_dim: int
    output_hw: Tuple[int, int]
    component_embedding_dim: int | None = None
    sigma_min: float = 0.05
    sigma_max: float = 0.5
    
    def setup(self):
        # Select conditioning strategy
        if self.conditioning_method == "film":
            self.conditioner = FiLMLayer(self.component_embedding_dim)
        elif self.conditioning_method == "concat":
            self.conditioner = ConcatConditioner()
        else:
            self.conditioner = NoopConditioner()
        
        # Backbone
        self.backbone = ConvBackbone(self.latent_dim, self.output_hw)
        
        # Output head
        if self.output_type == "heteroscedastic":
            self.output_head = HeteroscedasticHead(
                self.sigma_min, self.sigma_max
            )
        else:
            self.output_head = StandardHead()
    
    def __call__(self, z, component_embedding=None):
        features = self.conditioner(z, component_embedding)
        decoded = self.backbone(features)
        return self.output_head(decoded)
```

- [ ] Implement `ModularConvDecoder`
- [ ] Implement `ModularDenseDecoder` (same pattern, Dense backbone)
- [ ] Write integration tests for all 6 combinations:
  - (film, concat, none) × (standard, heteroscedastic)
- [ ] Benchmark: ensure no performance regression vs. legacy decoders

**Deliverable**: 2 modular decoder classes with full integration tests.

---

### Task 3: Migrate Factory & Network

#### 3.1 Update Factory

**File**: [`src/rcmvae/domain/components/factory.py`](../../../src/rcmvae/domain/components/factory.py)

**Replace `build_decoder` function**:

```python
def build_decoder(config: SSVAEConfig, *, input_hw=None):
    """Build decoder using modular composition.
    
    Maps config flags to conditioning method and output type.
    No silent overrides - all combinations are valid.
    """
    resolved_hw = _resolve_input_hw(config, input_hw)
    
    # Determine conditioning method
    if config.use_film_decoder and config.prior_type in {"mixture", "geometric_mog"}:
        conditioning = "film"
    elif config.use_component_aware_decoder and config.prior_type in {"mixture", "geometric_mog"}:
        conditioning = "concat"
    else:
        conditioning = "none"
    
    # Determine output type
    output_type = "heteroscedastic" if config.use_heteroscedastic_decoder else "standard"
    
    # Instantiate modular decoder
    if config.decoder_type == "conv":
        return ModularConvDecoder(
            conditioning_method=conditioning,
            output_type=output_type,
            latent_dim=config.latent_dim,
            output_hw=resolved_hw,
            component_embedding_dim=config.component_embedding_dim,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
        )
    elif config.decoder_type == "dense":
        hidden_dims = _resolve_encoder_hidden_dims(config, resolved_hw)
        decoder_hidden_dims = tuple(reversed(hidden_dims)) or (config.latent_dim,)
        return ModularDenseDecoder(
            conditioning_method=conditioning,
            output_type=output_type,
            hidden_dims=decoder_hidden_dims,
            output_hw=resolved_hw,
            component_embedding_dim=config.component_embedding_dim,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
        )
    
    raise ValueError(f"Unknown decoder type: {config.decoder_type}")
```

Tasks:
- [ ] Delete lines 99-103 (silent override logic)
- [ ] Delete lines 147-148 (unreachable error)
- [ ] Rewrite `build_decoder` as shown above
- [ ] Remove imports of old decoder classes (keep for now, deprecate later)

#### 3.2 Update Network

**File**: [`src/rcmvae/domain/network.py`](../../../src/rcmvae/domain/network.py)

- [ ] Delete lines 281-285 (duplicated override logic)
- [ ] Verify decoder calling convention matches modular signature
- [ ] Test forward pass with all combinations

**Deliverable**: Factory and network use modular decoders; no silent overrides.

---

### Task 4: Deprecation & Testing

#### 4.1 Deprecate Old Classes

**File**: `src/rcmvae/domain/components/decoders.py`

For each old decoder class:
```python
import warnings

@deprecated("Use ModularConvDecoder with conditioning_method='film' and output_type='standard'")
class FiLMConvDecoder(nn.Module):
    """Deprecated. Use ModularConvDecoder instead.
    
    Migration:
        # Old
        decoder = FiLMConvDecoder(...)
        
        # New
        decoder = ModularConvDecoder(
            conditioning_method="film",
            output_type="standard",
            ...
        )
    """
    ...
```

- [ ] Add `@deprecated` decorator to all 10 old classes
- [ ] Add migration guide in docstrings
- [ ] **Do not delete** (backward compatibility)

#### 4.2 Testing

**Test Suite**:
- [ ] Unit tests for each module pass
- [ ] Integration tests for all 6 decoder combinations pass
- [ ] All existing tests still pass (no regressions)
- [ ] Run experiment: FiLM + Heteroscedastic config trains successfully

**Performance Benchmark**:
- [ ] Measure forward/backward pass time for modular vs. legacy
- [ ] Document: acceptable if within 5% (JIT should optimize)

**Deliverable**: All tests passing, FiLM + Heteroscedastic validated.

---

## Validation & Experiments

### Experiment 1: Decentralized Latent Validation

**Goal**: Confirm decentralized mode produces per-component specialization.

**Config**:
```yaml
# configs/validate_decentralized.yaml
latent_layout: "decentralized"
latent_dim: 2  # d=2 for visualization
num_components: 10  # K=10 channels
use_gumbel_softmax: true
use_straight_through_gumbel: true
gumbel_temperature: 1.0
use_film_decoder: true
use_heteroscedastic_decoder: false  # Start simple
```

**Tasks**:
- [ ] Train on MNIST for 50 epochs
- [ ] **Visualize per-component latent spaces**:
  - Use `predict_batched(..., return_mixture=True)`
  - Extract `z_mean_per_component` from extras
  - Plot K separate 2D scatter plots (one per channel)
- [ ] **Verify specialization**:
  - Do channels show different cluster patterns?
  - Are some channels digit-specific?
- [ ] **Compare to shared baseline**:
  - Train with `latent_layout="shared"`, same hyperparams
  - Compare: reconstruction, KL_z, component usage, τ accuracy

**Success Metric**: Decentralized shows cleaner per-channel separation than shared.

---

### Experiment 2: FiLM vs. Concatenation Ablation

**Goal**: Empirically determine which conditioning method is better.

**Hypothesis**: FiLM provides stronger specialization than concatenation.

**Configs**:
```yaml
# A: FiLM
use_film_decoder: true
use_component_aware_decoder: false

# B: Concat
use_film_decoder: false
use_component_aware_decoder: true
```

**Tasks**:
- [ ] Train both configs (same hyperparams, 3 seeds each)
- [ ] Compare metrics:
  - Reconstruction loss (MSE)
  - Component usage entropy
  - τ-classifier accuracy
  - Visual: do channels serve distinct roles?
- [ ] Document findings in artifact

**Success Metric**: FiLM shows ≥5% improvement in reconstruction or specialization.

---

### Experiment 3: FiLM + Heteroscedastic Validation

**Goal**: Validate the "ultimate" config now enabled by modular decoders.

**Config**:
```yaml
# configs/ultimate_decoder.yaml
latent_layout: "decentralized"
use_film_decoder: true
use_heteroscedastic_decoder: true
use_gumbel_softmax: true
```

**Tasks**:
- [ ] Train without errors
- [ ] Verify outputs: `(mean, sigma)` tuples per component
- [ ] Visualize learned variance maps (σ per image)
- [ ] Check: Do different components learn different uncertainty patterns?
  - E.g., Channel 1 (clean digits) → low σ, Channel 2 (rotated) → high σ

**Success Metric**: Training converges, uncertainty estimates are sensible.

---

### Experiment 4: Curriculum Training (If Implemented)

**Goal**: Test channel unlocking strategy from supervisor spec.

**Config**:
```yaml
curriculum_enabled: true  # If implemented
k_active_schedule: [1, 2, 5, 10]  # Unlock at epochs [0, 10, 20, 30]
kl_boost_on_unlock: 2.0  # Boost KL weight when unlocking
```

**Tasks**:
- [ ] Train with curriculum
- [ ] Visualize: Do new channels appear as unlocked?
- [ ] Compare to no-curriculum baseline

**Success Metric**: Curriculum prevents mode collapse, cleaner specialization.

---

## Success Criteria

### Phase 1: Modular Decoders (Implementation)

- [ ] 7 reusable modules implemented and tested
- [ ] `ModularConvDecoder` and `ModularDenseDecoder` working
- [ ] All 6 combinations (3 conditioning × 2 outputs) tested
- [ ] FiLM + Heteroscedastic combo working (previously blocked)
- [ ] No silent overrides in factory or network
- [ ] No performance regression (within 5% of legacy)
- [ ] Old decoder classes deprecated (not deleted)
- [ ] All existing tests pass

### Phase 2: Validation & Documentation

- [ ] Decentralized latents produce per-component specialization (or issues documented)
- [ ] Gumbel-Softmax straight-through gradient flow confirmed
- [ ] FiLM vs. Concatenation empirically compared
- [ ] FiLM + Heteroscedastic validated experimentally
- [ ] Documentation updated (see [References](#references) for files to update)
- [ ] AI agents can execute from docs alone

---

## References

### Code Files

**Core Implementation**:
- [`src/rcmvae/domain/components/decoders.py`](../../../src/rcmvae/domain/components/decoders.py) - Current decoder classes
- [`src/rcmvae/domain/components/factory.py`](../../../src/rcmvae/domain/components/factory.py) - Silent overrides (lines 99-103, 147-148)
- [`src/rcmvae/domain/network.py`](../../../src/rcmvae/domain/network.py) - Duplicated logic (lines 281-285)
- [`src/rcmvae/domain/config.py`](../../../src/rcmvae/domain/config.py) - Configuration dataclass
- [`src/rcmvae/application/services/loss_pipeline.py`](../../../src/rcmvae/application/services/loss_pipeline.py) - Loss functions

**Documentation (To Update)**:
- [`docs/theory/conceptual_model.md`](../../theory/conceptual_model.md) - Add decentralized latents section
- [`docs/theory/mathematical_specification.md`](../../theory/mathematical_specification.md) - Add Gumbel-Softmax, FiLM math, all-channel KL
- [`docs/development/architecture.md`](../../development/architecture.md) - Update decoder section with modular pattern
- [`docs/development/implementation.md`](../../development/implementation.md) - Document new modules
- [`docs/development/extending.md`](../../development/extending.md) - Add tutorials for new conditioning methods
- [`docs/theory/implementation_roadmap.md`](../../theory/implementation_roadmap.md) - Mark decentralized features as validated

### Related Artifacts

- `decoder_refactor_context.md` - Original analysis (silent overrides discovery)
- `decoder_architecture_analysis.md` - Comparison of design options
