# Heteroscedastic Decoder Implementation Plan

> **Status**: Ready for Implementation (Nov 10, 2025)
> **Priority**: NEXT PRIORITY after τ-classifier completion
> **Session ID**: claude/heteroscedastic-decoder-review-011CUzhjm2QDuKwX8fgtMVsQ

---

## Executive Summary

This document defines the implementation plan for adding heteroscedastic decoder support to the RCM-VAE codebase. The heteroscedastic decoder learns per-input variance σ(x) to capture aleatoric (observation) uncertainty, complementing the epistemic uncertainty already captured in the latent space.

**Key Benefits:**
- **Aleatoric uncertainty quantification**: Model noise inherent in observations
- **Improved reconstruction quality**: Adaptive noise modeling for clean vs noisy inputs
- **Better OOD detection**: Combine reconstruction confidence with latent-based scores
- **Calibrated uncertainty**: Separate "what" (epistemic via z) from "how noisy" (aleatoric via σ)

---

## Documentation Review Summary

### 1. Conceptual Model ([conceptual_model.md](docs/theory/conceptual_model.md))

**Key Insights:**
- Aleatoric uncertainty lives in **heteroscedastic decoder variance** σ²(x) (clamped for stability)
- Epistemic uncertainty in **z** (and model parameters)
- Clear separation: discrete ambiguity through q(c|x), observation noise through σ²(x)

**Non-Negotiables:**
> "Train a **heteroscedastic** decoder with clamped σ²(x)." (Line 75)

**Default Variance:**
> "Default is a per-image σ(x) (clamped) for stability; a per-pixel head is optional and can be enabled later." (Line 87)

### 2. Mathematical Specification ([mathematical_specification.md](docs/theory/mathematical_specification.md))

**Section 4 - Objective:**
> "**Decoder variance stability:** per-image scalar σ(x)=σ_min+softplus(s_θ(x)), clamp σ(x)∈[0.05,0.5]; optional small penalty λ_σ(logσ(x)-μ_σ)² (default off)." (Lines 74-75)

**Section 8 - Defaults:**
> "**Decoder variance:** per-image scalar, σ_min=0.05, clamp [0.05,0.5]." (Line 136)

**Reconstruction Loss Formula:**
```
L_recon = ||x - x̂||² / (2σ²) + log σ
```

This is the **negative log-likelihood** of a Gaussian observation model:
```
p(x|x̂,σ) = N(x; x̂, σ²I)
-log p(x|x̂,σ) = (1/2σ²)||x - x̂||² + log σ + const
```

### 3. Implementation Roadmap ([implementation_roadmap.md](docs/theory/implementation_roadmap.md))

**Status at a Glance (Line 20):**
```
| **Heteroscedastic decoder** σ(x) | 🎯 **Next priority** | [Math Spec §4] |
```

**Near-Term Enhancement (Lines 151-156):**
```
**Heteroscedastic Decoder:**
- Add variance head: σ(x) = σ_min + softplus(s_θ(x))
- Clamp σ(x) ∈ [0.05, 0.5] for stability
- Reconstruction loss: ||x - x̂||²/(2σ²) + log σ
- **Enables:** Aleatoric uncertainty quantification per input
```

### 4. System Architecture ([architecture.md](docs/development/architecture.md))

**Design Principles (Lines 12-18):**
1. Protocol-based abstractions
2. Factory pattern for component creation
3. Configuration-driven
4. Separation of concerns
5. Immutability (JAX functional patterns)

**Current Decoders (Lines 242-266):**
- `DenseDecoder`: Fully connected layers
- `ConvDecoder`: Transposed convolutional layers
- Component-aware variants: `ComponentAwareDenseDecoder`, `ComponentAwareConvDecoder`

**Interface Pattern:**
```python
class Decoder(nn.Module):
    def __call__(self, z, deterministic=True):
        # Returns: reconstructed x
        ...
```

### 5. Current Loss Computation ([losses.py](src/training/losses.py))

**Current Reconstruction Loss (Lines 15-23):**
```python
def reconstruction_loss_mse(x: jnp.ndarray, recon: jnp.ndarray, weight: float):
    """Mean squared error reconstruction loss."""
    diff = jnp.square(x - recon)
    per_sample = jnp.mean(diff, axis=axes)
    return weight * jnp.mean(per_sample)
```

**Current Status:** Only mean reconstruction, no variance prediction

---

## Success Criteria

### Primary Goals

#### 1. Functional Requirements
- ✅ **Variance Head Output**: Decoders output (x̂, σ) tuple
- ✅ **Variance Parameterization**: σ(x) = σ_min + softplus(s_θ(x))
- ✅ **Clamping**: Hard clamp σ(x) ∈ [0.05, 0.5]
- ✅ **Per-Image Scalar**: Single σ value per image (not per-pixel)
- ✅ **Heteroscedastic Loss**: L = ||x - x̂||²/(2σ²) + log σ
- ✅ **Backward Compatible**: Existing decoders continue to work

#### 2. Architecture Coverage
- ✅ **Dense Decoder**: HeteroscedasticDenseDecoder
- ✅ **Conv Decoder**: HeteroscedasticConvDecoder
- ✅ **Component-Aware Dense**: ComponentAwareHeteroscedasticDenseDecoder
- ✅ **Component-Aware Conv**: ComponentAwareHeteroscedasticConvDecoder

#### 3. Integration Points
- ✅ **Factory Integration**: Auto-select based on config.use_heteroscedastic_decoder
- ✅ **Loss Integration**: Prior.compute_reconstruction_loss() handles heteroscedastic
- ✅ **Configuration**: New parameters in SSVAEConfig
- ✅ **Model Integration**: Forward pass returns variance

#### 4. Testing & Validation
- ✅ **Unit Tests**: Decoder output shapes, variance bounds, loss computation
- ✅ **Integration Tests**: Full training loop with heteroscedastic decoder
- ✅ **Ablation Experiment**: Compare with/without heteroscedastic variance
- ✅ **Variance Analysis**: Visualize learned σ(x) across different inputs

### Secondary Goals (Nice to Have)

#### 1. Advanced Features
- ⏸️ **Per-Pixel Variance**: Optional σ(x) ∈ ℝ^(H×W) (deferred per spec)
- ⏸️ **Variance Regularization**: Optional λ_σ(log σ - μ_σ)² penalty (default off per spec)

#### 2. Diagnostics & Visualization
- ✅ **Variance Histograms**: Distribution of σ values
- ✅ **Uncertainty Maps**: Visualize high/low variance regions
- ✅ **Correlation Analysis**: σ vs reconstruction error

---

## Implementation Plan

### Phase 1: Core Decoder Implementation

#### Task 1.1: Heteroscedastic Dense Decoder
**File**: `src/ssvae/components/decoders.py`

**Add Classes:**
```python
class HeteroscedasticDenseDecoder(nn.Module):
    """Dense decoder with learned per-image variance."""
    hidden_dims: Tuple[int, ...]
    output_hw: Tuple[int, int]
    sigma_min: float = 0.05
    sigma_max: float = 0.5

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Decode latent to mean and variance.

        Returns:
            mean: Reconstructed image [batch, H, W]
            sigma: Per-image std deviation [batch,]
        """
        # Shared trunk
        x = z
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f"hidden_{i}")(x)
            x = nn.leaky_relu(x)

        # Mean head
        h, w = self.output_hw
        mean = nn.Dense(h * w, name="mean_head")(x)
        mean = mean.reshape((-1, h, w))

        # Variance head (scalar per image)
        log_sigma_raw = nn.Dense(1, name="sigma_head")(x)  # [batch, 1]
        log_sigma_raw = log_sigma_raw.squeeze(-1)  # [batch,]
        sigma = self.sigma_min + jax.nn.softplus(log_sigma_raw)
        sigma = jnp.clip(sigma, self.sigma_min, self.sigma_max)

        return mean, sigma
```

#### Task 1.2: Heteroscedastic Conv Decoder
**File**: `src/ssvae/components/decoders.py`

Similar structure, but:
- Convolutional trunk for spatial features
- Global average pooling before sigma head
- Variance head outputs single scalar per image

#### Task 1.3: Component-Aware Heteroscedastic Variants

**Add:**
- `ComponentAwareHeteroscedasticDenseDecoder`
- `ComponentAwareHeteroscedasticConvDecoder`

**Pattern**: Extend component-aware decoders to output (mean, sigma) tuples

### Phase 2: Loss Function Implementation

#### Task 2.1: Heteroscedastic Reconstruction Loss
**File**: `src/training/losses.py`

**Add Function:**
```python
def heteroscedastic_reconstruction_loss(
    x: jnp.ndarray,           # [batch, H, W]
    mean: jnp.ndarray,        # [batch, H, W]
    sigma: jnp.ndarray,       # [batch,]
    weight: float,
) -> jnp.ndarray:
    """Heteroscedastic reconstruction loss with learned variance.

    Loss = ||x - mean||² / (2σ²) + log σ

    This is the negative log-likelihood under Gaussian observation model:
    p(x|mean,σ) = N(x; mean, σ²I)

    Args:
        x: Ground truth images
        mean: Predicted mean reconstructions
        sigma: Predicted per-image standard deviations
        weight: Loss scaling factor

    Returns:
        Weighted scalar loss
    """
    # Compute squared error per image
    diff = jnp.square(x - mean)
    if diff.ndim > 1:
        axes = tuple(range(1, diff.ndim))
        se_per_image = jnp.sum(diff, axis=axes)  # [batch,]
    else:
        se_per_image = diff

    # Negative log-likelihood
    # NLL = (1/2σ²) ||x - mean||² + log σ + const
    sigma_safe = jnp.maximum(sigma, 1e-6)  # Numerical stability
    nll = se_per_image / (2 * sigma_safe ** 2) + jnp.log(sigma_safe)

    return weight * jnp.mean(nll)
```

#### Task 2.2: Weighted Heteroscedastic Loss (Mixture Prior)
**File**: `src/training/losses.py`

**Add Function:**
```python
def weighted_heteroscedastic_reconstruction_loss(
    x: jnp.ndarray,                    # [batch, H, W]
    mean_components: jnp.ndarray,      # [batch, K, H, W]
    sigma_components: jnp.ndarray,     # [batch, K]
    responsibilities: jnp.ndarray,     # [batch, K]
    weight: float,
) -> jnp.ndarray:
    """Expected heteroscedastic reconstruction loss under q(c|x).

    Loss = E_q(c|x) [ ||x - mean_c||²/(2σ_c²) + log σ_c ]
         = Σ_c q(c|x) [ ||x - mean_c||²/(2σ_c²) + log σ_c ]
    """
    # Compute per-component squared errors
    diff = jnp.square(x[:, None, ...] - mean_components)  # [batch, K, H, W]
    axes = tuple(range(2, diff.ndim))
    se_per_component = jnp.sum(diff, axis=axes)  # [batch, K]

    # Compute per-component NLL
    sigma_safe = jnp.maximum(sigma_components, 1e-6)
    nll_per_component = (
        se_per_component / (2 * sigma_safe ** 2) + jnp.log(sigma_safe)
    )  # [batch, K]

    # Weight by responsibilities
    weighted_nll = jnp.sum(responsibilities * nll_per_component, axis=1)  # [batch,]

    return weight * jnp.mean(weighted_nll)
```

### Phase 3: Prior Integration

#### Task 3.1: Update StandardPrior
**File**: `src/ssvae/priors/standard.py`

**Modify `compute_reconstruction_loss`:**
```python
def compute_reconstruction_loss(
    self,
    x_true: jnp.ndarray,
    x_recon: jnp.ndarray | Tuple[jnp.ndarray, jnp.ndarray],  # (mean, sigma) or just mean
    encoder_output: EncoderOutput,
    config,
) -> jnp.ndarray:
    """Compute reconstruction loss (heteroscedastic or standard)."""

    # Check if heteroscedastic (tuple output)
    if isinstance(x_recon, tuple):
        mean, sigma = x_recon
        return heteroscedastic_reconstruction_loss(
            x_true, mean, sigma, config.recon_weight
        )
    else:
        # Standard reconstruction (backward compatible)
        return reconstruction_loss(
            x_true, x_recon, config.recon_weight, config.reconstruction_loss
        )
```

#### Task 3.2: Update MixturePrior
**File**: `src/ssvae/priors/mixture.py`

Similar update to handle both heteroscedastic and standard reconstructions.

### Phase 4: Configuration & Factory

#### Task 4.1: Add Configuration Parameters
**File**: `src/ssvae/config.py`

**Add Fields:**
```python
@dataclass
class SSVAEConfig:
    # ... existing fields ...

    # Heteroscedastic decoder
    use_heteroscedastic_decoder: bool = False  # Enable learned variance
    sigma_min: float = 0.05                    # Minimum allowed σ
    sigma_max: float = 0.5                     # Maximum allowed σ

    # Optional: variance regularization (default off per spec)
    use_sigma_regularization: bool = False
    sigma_regularization_weight: float = 0.0
    sigma_target_mean: float = 0.1
```

**Add to INFORMATIVE_HPARAMETERS:**
```python
INFORMATIVE_HPARAMETERS = (
    # ... existing ...
    "use_heteroscedastic_decoder",
    "sigma_min",
    "sigma_max",
)
```

#### Task 4.2: Update Factory
**File**: `src/ssvae/factory.py`

**Modify `build_decoder`:**
```python
def build_decoder(config: SSVAEConfig, input_shape, key):
    """Create decoder based on config."""

    # Determine decoder class
    if config.decoder_type == "dense":
        if config.use_heteroscedastic_decoder:
            if config.use_component_aware_decoder and config.prior_type == "mixture":
                decoder_cls = ComponentAwareHeteroscedasticDenseDecoder
            else:
                decoder_cls = HeteroscedasticDenseDecoder
        else:
            # Standard (backward compatible)
            if config.use_component_aware_decoder and config.prior_type == "mixture":
                decoder_cls = ComponentAwareDenseDecoder
            else:
                decoder_cls = DenseDecoder

    elif config.decoder_type == "conv":
        # Similar logic for conv decoders
        ...

    # Create decoder with appropriate parameters
    decoder = decoder_cls(
        hidden_dims=...,
        output_hw=...,
        sigma_min=config.sigma_min if config.use_heteroscedastic_decoder else None,
        sigma_max=config.sigma_max if config.use_heteroscedastic_decoder else None,
        ...
    )

    return decoder
```

### Phase 5: Testing

#### Task 5.1: Unit Tests
**File**: `tests/test_heteroscedastic_decoder.py` (new)

**Test Cases:**
1. **Output Shape Test**: Verify (mean, sigma) tuple shapes
2. **Variance Bounds Test**: Check σ ∈ [σ_min, σ_max]
3. **Gradient Flow Test**: Verify gradients flow through both heads
4. **Loss Computation Test**: Verify heteroscedastic loss formula
5. **Backward Compatibility Test**: Ensure standard decoders still work

#### Task 5.2: Integration Tests
**File**: `tests/test_integration_workflows.py`

**Add Test:**
```python
def test_heteroscedastic_training_loop():
    """Test full training with heteroscedastic decoder."""
    config = SSVAEConfig(
        use_heteroscedastic_decoder=True,
        sigma_min=0.05,
        sigma_max=0.5,
        max_epochs=5,
    )
    model = SSVAE(input_dim=(28, 28), config=config)
    # ... train and verify loss decreases
```

#### Task 5.3: Validation Experiment
**File**: `use_cases/experiments/configs/heteroscedastic_validation.yaml`

**Configuration:**
```yaml
model:
  prior_type: "mixture"
  num_components: 10
  use_component_aware_decoder: true
  use_heteroscedastic_decoder: true
  sigma_min: 0.05
  sigma_max: 0.5
  latent_dim: 16

training:
  max_epochs: 50
  batch_size: 128

experiment:
  name: "heteroscedastic_decoder_validation"
  description: "Validate heteroscedastic decoder vs standard decoder"
```

### Phase 6: Visualization & Diagnostics

#### Task 6.1: Variance Visualization
**File**: `use_cases/experiments/src/visualization/plotters.py`

**Add Function:**
```python
def plot_learned_variances(
    variances: np.ndarray,      # [n_samples,]
    labels: np.ndarray,         # [n_samples,]
    recon_errors: np.ndarray,   # [n_samples,]
    save_path: Path,
):
    """Visualize learned per-image variances.

    Creates:
    1. Histogram of σ values
    2. σ vs reconstruction error scatter
    3. σ distribution per class
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Histogram
    axes[0].hist(variances, bins=50, edgecolor='black')
    axes[0].axvline(variances.mean(), color='r', linestyle='--', label=f'Mean: {variances.mean():.3f}')
    axes[0].set_xlabel('Learned σ')
    axes[0].set_ylabel('Count')
    axes[0].legend()

    # Scatter: σ vs error
    axes[1].scatter(recon_errors, variances, alpha=0.3)
    axes[1].set_xlabel('Reconstruction Error')
    axes[1].set_ylabel('Learned σ')

    # Box plot per class
    class_data = [variances[labels == i] for i in np.unique(labels)]
    axes[2].boxplot(class_data)
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('Learned σ')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
```

---

## Expected Outcomes

### Quantitative Metrics

1. **Reconstruction Quality**:
   - Lower NLL on test set (proper probabilistic scoring)
   - Similar or better MSE/MAE (mean predictions)

2. **Uncertainty Calibration**:
   - High σ for ambiguous/noisy inputs
   - Low σ for clean, easy inputs
   - Correlation between σ and actual reconstruction error

3. **OOD Detection**:
   - Improved AUROC when combining latent + reconstruction uncertainty
   - High σ on out-of-distribution samples

### Qualitative Analysis

1. **Variance Patterns**:
   - Digit boundaries: higher σ
   - Solid regions: lower σ
   - Between-class examples: higher σ

2. **Ablation Study**:
   - Standard decoder: uniform implicit variance
   - Heteroscedastic decoder: adaptive variance
   - Component-aware + heteroscedastic: best of both

---

## Risk Mitigation

### Potential Issues

1. **Variance Collapse**: σ → σ_min everywhere
   - **Mitigation**: Proper initialization, monitor σ distribution
   - **Check**: Variance histogram should show spread, not concentration at bounds

2. **Variance Explosion**: σ → σ_max everywhere
   - **Mitigation**: Hard clamping, proper loss weighting
   - **Check**: Loss should decrease, not increase

3. **Mean-Variance Trade-off**: Model uses σ to explain away reconstruction errors
   - **Mitigation**: Balance recon_weight appropriately
   - **Check**: Visual inspection of reconstructions

4. **Integration Complexity**: Many decoder variants to maintain
   - **Mitigation**: Shared base classes, factory pattern
   - **Check**: All tests pass, no regressions

### Rollback Plan

If heteroscedastic decoder causes issues:
1. Set `use_heteroscedastic_decoder: false` (default)
2. System reverts to standard decoders (backward compatible)
3. No data loss, no model corruption

---

## Timeline Estimate

**Total**: ~4-6 hours implementation + 2-3 hours testing/validation

### Breakdown:
- Phase 1 (Decoders): 1.5 hours
- Phase 2 (Loss): 1 hour
- Phase 3 (Prior): 0.5 hours
- Phase 4 (Config/Factory): 0.5 hours
- Phase 5 (Tests): 1.5 hours
- Phase 6 (Viz): 1 hour

---

## Dependencies

**Completed:**
- ✅ Component-aware decoder (Nov 9, 2025)
- ✅ τ-classifier (Nov 10, 2025)
- ✅ Prior-based loss delegation
- ✅ Factory pattern
- ✅ Comprehensive test suite

**Blockers:** None

**Enables:**
- 📋 OOD detection (requires heteroscedastic + τ-classifier)
- 📋 Uncertainty-aware active learning
- 📋 Calibration analysis

---

## Related Documentation

- **[Conceptual Model](docs/theory/conceptual_model.md)** - Design vision and invariants
- **[Mathematical Specification](docs/theory/mathematical_specification.md)** - Precise formulations
- **[Implementation Roadmap](docs/theory/implementation_roadmap.md)** - Current status
- **[System Architecture](docs/development/architecture.md)** - Design patterns
- **[Extending the System](docs/development/extending.md)** - Extension tutorials

---

## Sign-off Checklist

Before starting implementation:
- [x] Reviewed all relevant documentation
- [x] Understood mathematical specification
- [x] Identified all integration points
- [x] Defined clear success criteria
- [x] Estimated timeline
- [x] Planned testing strategy
- [x] Identified risks and mitigations

**Status**: ✅ Ready to proceed with implementation

**Next Action**: Begin Phase 1 - Core Decoder Implementation
