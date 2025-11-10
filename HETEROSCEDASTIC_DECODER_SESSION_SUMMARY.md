# Heteroscedastic Decoder Implementation - Session Summary

**Session Date**: November 10, 2025
**Branch**: `claude/heteroscedastic-decoder-review-011CUzhjm2QDuKwX8fgtMVsQ`
**Status**: ✅ **COMPLETE - Ready for Testing**

---

## What Was Implemented

### 1. Core Heteroscedastic Decoder Implementation

**4 New Decoder Classes** (src/ssvae/components/decoders.py, lines 196-563):
- `HeteroscedasticDenseDecoder` - Dense architecture with dual heads (mean, σ)
- `HeteroscedasticConvDecoder` - Convolutional architecture with global pooling for σ
- `ComponentAwareHeteroscedasticDenseDecoder` - Component-aware + heteroscedastic
- `ComponentAwareHeteroscedasticConvDecoder` - Conv variant with both features

**Key Implementation Details**:
- Variance parameterization: `σ = σ_min + softplus(s_θ(x))`
- Hard clamping: `σ ∈ [0.05, 0.5]`
- Per-image scalar variance (not per-pixel)
- All decoders return `(mean, sigma)` tuples

### 2. Heteroscedastic Loss Functions

**2 New Loss Functions** (src/training/losses.py, lines 106-199):
- `heteroscedastic_reconstruction_loss()` - Standard NLL: `||x - mean||²/(2σ²) + log σ`
- `weighted_heteroscedastic_reconstruction_loss()` - Expectation over mixture components

**Properties**:
- Numerical stability via `sigma_safe = max(sigma, EPS)`
- Prevents trivial solutions through `log σ` regularization
- Proper Gaussian likelihood formulation

### 3. Prior Integration

**Updated Both Priors** for heteroscedastic support:
- `StandardPrior.compute_reconstruction_loss()` (src/ssvae/priors/standard.py:52-87)
  - Auto-detects tuple outputs: `isinstance(x_recon, tuple)`
  - Falls back to standard MSE/BCE for backward compatibility

- `MixturePrior.compute_reconstruction_loss()` (src/ssvae/priors/mixture.py:117-181)
  - Handles per-component `(mean, sigma)` tuples
  - Expectation over responsibilities with heteroscedastic loss

### 4. Configuration & Factory

**Configuration** (src/ssvae/config.py):
- Added 3 parameters (lines 163-165):
  - `use_heteroscedastic_decoder: bool = False` (backward compatible default)
  - `sigma_min: float = 0.05`
  - `sigma_max: float = 0.5`
- Added validation (lines 214-221): σ_min > 0, σ_max > σ_min
- Added to `INFORMATIVE_HPARAMETERS` (lines 64-66)

**Factory** (src/ssvae/components/factory.py):
- Updated `build_decoder()` to handle 8 decoder variants (lines 54-146)
- Auto-selects based on: decoder_type × component_aware × heteroscedastic
- Imports all 4 heteroscedastic classes (lines 9-18)

### 5. Testing & Validation

**Unit Tests** (tests/test_heteroscedastic_decoder.py):
- 25+ comprehensive test cases
- Coverage: output shapes, variance bounds, gradient flow, loss correctness
- Factory integration tests, config validation
- **Note**: File exists locally but gitignored (tests/ in .gitignore)

**Validation Experiments**:
- `use_cases/experiments/configs/heteroscedastic_validation.yaml` - Full config with learned variance
- `use_cases/experiments/configs/heteroscedastic_baseline.yaml` - Baseline for comparison

### 6. Documentation

**Planning Document** (HETEROSCEDASTIC_DECODER_PLAN.md):
- 608 lines of comprehensive implementation plan
- All 6 phases documented with code examples
- Success criteria, risk mitigation, timeline estimates

---

## Git Commits (All Pushed to Remote)

```bash
8673186 - Add comprehensive heteroscedastic decoder implementation plan (608 lines)
f7ae021 - Implement heteroscedastic decoder for aleatoric uncertainty quantification (600 lines code)
6768076 - Add validation experiment configurations for heteroscedastic decoder (2 configs)
```

**Branch Status**: Clean working tree ✅
**Remote**: Synchronized ✅

---

## Files Modified (Summary)

| File | Lines Changed | Status |
|------|---------------|--------|
| src/ssvae/components/decoders.py | +375 | ✅ Linted |
| src/training/losses.py | +95 | ✅ Linted |
| src/ssvae/priors/standard.py | +15 | ✅ Linted |
| src/ssvae/priors/mixture.py | +20 | ✅ Linted |
| src/ssvae/config.py | +15 | ✅ Linted |
| src/ssvae/components/factory.py | +70 | ✅ Linted |
| tests/test_heteroscedastic_decoder.py | +550 | Created (gitignored) |
| use_cases/experiments/configs/*.yaml | +145 | ✅ Committed |
| HETEROSCEDASTIC_DECODER_PLAN.md | +608 | ✅ Committed |

**Total**: ~1,900 lines of new code + documentation

---

## Verification Checklist

All success criteria from HETEROSCEDASTIC_DECODER_PLAN.md met:

- ✅ Variance head outputs (mean, σ) tuples
- ✅ Variance parameterization: σ = σ_min + softplus(s_θ(x))
- ✅ Clamping: σ ∈ [0.05, 0.5]
- ✅ Per-image scalar variance
- ✅ Heteroscedastic loss: ||x - x̂||²/(2σ²) + log σ
- ✅ Backward compatible (default: use_heteroscedastic_decoder=False)
- ✅ All 4 architecture variants implemented
- ✅ Factory integration complete
- ✅ Configuration with validation
- ✅ Prior integration (both standard and mixture)
- ✅ Comprehensive tests written
- ✅ Validation experiments configured

---

## Test Results ✅

**Unit Tests**: All 25 tests PASSED! (13.08s runtime)

**Test Command**:
```bash
JAX_PLATFORMS=cpu poetry run pytest tests/test_heteroscedastic_decoder.py -v
```

**Test Coverage** (25 tests):
- ✅ HeteroscedasticDenseDecoder (4 tests): output shape, variance bounds, gradient flow, deterministic
- ✅ HeteroscedasticConvDecoder (3 tests): output shape, variance bounds, validation
- ✅ ComponentAwareHeteroscedasticDenseDecoder (3 tests): output shape, variance bounds, component influence
- ✅ ComponentAwareHeteroscedasticConvDecoder (2 tests): output shape, variance bounds
- ✅ Loss functions (5 tests): shape, error sensitivity, sigma sensitivity, weighted, responsibilities
- ✅ Factory integration (6 tests): all 8 decoder variants correctly selected
- ✅ Configuration validation (3 tests): sigma_min, sigma_max, defaults

**Key Validations**:
- Variance parameterization: σ = σ_min + softplus(s_θ(x)) ✅
- Variance bounds: σ ∈ [0.05, 0.5] enforced ✅
- Loss formula: ||x - x̂||²/(2σ²) + log σ ✅
- Gradient flow: Mean and sigma heads both differentiable ✅
- Factory selection: Correct decoder for all config combinations ✅

## How to Continue in Next Session

### Immediate Next Steps

**1. Run Unit Tests** ✅ **COMPLETED**
```bash
JAX_PLATFORMS=cpu poetry run pytest tests/test_heteroscedastic_decoder.py -v
Result: 25 passed, 6 warnings in 13.08s
```

**2. Run Validation Experiment**:
```bash
# Test heteroscedastic decoder
poetry run python use_cases/experiments/run_experiment.py \
    --config use_cases/experiments/configs/heteroscedastic_validation.yaml

# Compare with baseline
poetry run python use_cases/experiments/run_experiment.py \
    --config use_cases/experiments/configs/heteroscedastic_baseline.yaml
```

**3. Analyze Results**:
- Compare NLL (heteroscedastic should be lower)
- Check variance distribution (should be ~0.1-0.2, not collapsed)
- Visualize σ vs reconstruction error correlation
- Evaluate OOD detection with combined uncertainty

### Expected Validation Outcomes

**Quantitative**:
- Lower NLL on test set (better probabilistic scoring)
- Similar or better MSE/MAE (mean predictions)
- Variance spread: mean ~0.1-0.2, no collapse to bounds

**Qualitative**:
- High σ for ambiguous/noisy inputs
- Low σ for clean, confident inputs
- Per-class variance reflects aleatoric uncertainty

### Integration with Broader System

**This implementation enables**:
1. **OOD Detection**: Combine epistemic (z) + aleatoric (σ) uncertainty
2. **Uncertainty-Aware Active Learning**: Query high-uncertainty samples
3. **Calibration Analysis**: Proper uncertainty quantification
4. **Ablation Studies**: Compare heteroscedastic vs standard decoders

**Dependencies satisfied**:
- ✅ Component-aware decoder (completed Nov 9)
- ✅ τ-classifier (completed Nov 10)
- ✅ Prior abstraction
- ✅ Factory pattern

**Next features unlocked** (per implementation_roadmap.md):
- OOD detection with combined uncertainty
- Uncertainty visualization
- Calibration analysis

---

## Key Code Locations for Reference

**Variance Parameterization**:
- decoders.py:254-256 (HeteroscedasticDenseDecoder)
- decoders.py:359-361 (HeteroscedasticConvDecoder)
- decoders.py:440-442 (ComponentAwareHeteroscedasticDenseDecoder)
- decoders.py:560-562 (ComponentAwareHeteroscedasticConvDecoder)

**Heteroscedastic Loss**:
- losses.py:147 (main formula)
- losses.py:189-191 (weighted version)

**Factory Selection Logic**:
- factory.py:88-96 (dense variant selection)
- factory.py:119-126 (conv variant selection)

**Configuration**:
- config.py:163-165 (parameters)
- config.py:214-221 (validation)

---

## Mathematical Specification Reference

**Per math spec (docs/theory/mathematical_specification.md §4, §8)**:
- Variance: σ(x) = σ_min + softplus(s_θ(x))
- Bounds: σ ∈ [0.05, 0.5]
- Loss: -log p(x|mean,σ) = ||x - mean||²/(2σ²) + log σ + const
- Default: σ_min=0.05, σ_max=0.5, per-image scalar

**Implementation follows spec exactly** ✅

---

## Notes for Next Session

1. **Environment Setup**: Tests require JAX, Flax, pytest via poetry
2. **Test Execution**: Run tests to verify all 25+ cases pass
3. **Experiment Validation**: Compare heteroscedastic vs baseline performance
4. **Potential Issues to Watch**:
   - Variance collapse (σ → σ_min everywhere)
   - Variance explosion (σ → σ_max everywhere)
   - Mean-variance trade-off (check reconstructions visually)

5. **If Tests Fail**: Implementation is complete but may need model integration adjustments
6. **If Tests Pass**: Move to validation experiments and performance analysis

---

## Summary

**Implementation Status**: ✅ **COMPLETE**
- All code written, linted, committed, and pushed
- Backward compatible (default: use_heteroscedastic_decoder=False)
- Follows AGENTS.md policy throughout
- Follows mathematical specification exactly
- Ready for testing and validation

**Next Session Goal**: Validate implementation through tests and experiments

**Branch**: `claude/heteroscedastic-decoder-review-011CUzhjm2QDuKwX8fgtMVsQ`
**Last Commit**: `6e52df8` (integration fixes for network and visualization)

---

## Validation Experiment Results (Session 2)

### Integration Fixes Applied

**Issue**: Heteroscedastic decoder outputs `(mean, sigma)` tuples not handled in:
- Network forward pass (mixture prior component-wise processing)
- Model predict methods (numpy conversion)
- Visualization plotters (reconstruction display)

**Solution** (Commit `6e52df8`):
1. **Network (src/ssvae/network.py)**:
   - Updated `ForwardOutput.recon` type annotation to support tuples
   - Added tuple detection in mixture prior path
   - Separate reshaping and expectation for mean and sigma
   - Store `(mean_per_component, sigma_per_component)` in extras

2. **Models (src/ssvae/models.py)**:
   - Handle tuple outputs in `_predict_deterministic()`
   - Handle tuple outputs in `_predict_with_sampling()`
   - Stack means and sigmas separately when sampling

3. **Visualization (use_cases/experiments/src/visualization/plotters.py)**:
   - Extract mean from tuple for reconstruction plots
   - Maintains backward compatibility

### Experiment Comparison

**Configuration**:
- Both use same architecture: mixture prior (K=10), component-aware decoder, τ-classifier
- Same hyperparameters: latent_dim=16, hidden_dims=[256,128,64], 50 labeled samples
- Only difference: `use_heteroscedastic_decoder` (True vs False)

**Results**:

| Metric | Heteroscedastic | Baseline | Ratio |
|--------|----------------|----------|-------|
| **Training** |
| Epochs | 51 | 118 | 0.43x |
| Time (s) | 176.9 | 370.3 | 0.48x |
| Final Loss | 15,834 | 117.6 | **135x** |
| Recon Loss | 15,370 | 25.6 | **600x** |
| KL(z) | 372.3 | 0.025 | **14,892x** |
| **Performance** |
| Accuracy | 9.5% | 37.0% | 0.26x |
| Certainty (mean) | 0.182 | 0.450 | 0.40x |
| Certainty (range) | [0.180, 0.182] | [0.150, 0.861] | - |
| **Mixture Quality** |
| K_eff | 1.00 | 9.90 | **0.10x** |
| Active Components | 1 | 10 | 0.10x |
| Component Entropy | 0.001 | 0.031 | 0.03x |
| Max Component Usage | 99.98% | 12.5% | 8.0x |
| **τ-Classifier** |
| Label Coverage | 2/10 | 9/10 | 0.22x |
| Avg Comp/Label | 0.30 | 2.10 | 0.14x |
| τ Sparsity | 0.04 | 0.61 | 0.07x |

### Critical Issues Identified

#### 1. Component Collapse ⚠️
**Heteroscedastic model collapsed to 1 active component**:
- Component 5: 99.98% usage
- All other components: <0.02% usage each
- K_eff = 1.00 (should be ~10)
- This defeats the purpose of the mixture prior entirely

**Root Cause**: Extremely high reconstruction loss creates unstable gradients that prevent mixture learning.

#### 2. Reconstruction Loss Scale Mismatch ⚠️
**Heteroscedastic reconstruction loss is 600x higher than baseline**:

**Standard MSE Loss**:
```
L_recon = recon_weight * ||x - x̂||²
       ≈ 500 * 0.05  (typical pixel-wise MSE)
       ≈ 25
```

**Heteroscedastic NLL Loss**:
```
L_het = recon_weight * (||x - x̂||²/(2σ²) + log σ)
      ≈ 500 * (0.05/(2*0.05²) + log(0.05))
      ≈ 500 * (0.05/0.005 + (-3.0))
      ≈ 500 * (10 - 3)
      ≈ 3,500  (before considering actual errors)
```

When σ is near σ_min = 0.05:
- Division by σ² = 0.0025 amplifies errors by 400x
- This creates massive gradient magnitudes
- Training becomes unstable, mixture collapses

#### 3. KL Divergence Anomaly
**KL(z) is 14,892x higher for heteroscedastic** (372 vs 0.025):
- This suggests the encoder is producing very different latent distributions
- Likely a consequence of the reconstruction loss instability
- The model may be "fleeing" to extreme latent values to try to reduce reconstruction loss

### Analysis & Recommendations

#### Issue: Loss Scale Incompatibility
The heteroscedastic NLL formula has fundamentally different magnitude than MSE:
- MSE scales linearly with squared error
- NLL includes division by σ², creating non-linear amplification
- Using the same `recon_weight=500` for both is incorrect

#### Recommended Fixes

**Option 1: Reduce recon_weight for heteroscedastic** (Quick Fix)
```yaml
# heteroscedastic_validation.yaml
recon_weight: 50.0  # 10x reduction from 500.0
```
- Reduces heteroscedastic loss to similar magnitude as standard MSE
- May require tuning to find optimal value

**Option 2: Normalize heteroscedastic loss** (Better Fix)
Modify `heteroscedastic_reconstruction_loss()` to normalize:
```python
# Compute NLL
nll_per_image = se_per_image / (2 * sigma_safe ** 2) + jnp.log(sigma_safe)

# Normalize by dividing by typical sigma value to match MSE scale
# This makes the loss magnitude similar to standard MSE
normalized_nll = nll_per_image / jnp.log(1.0 / sigma_min)

return weight * jnp.mean(normalized_nll)
```

**Option 3: Separate weight parameter** (Most Flexible)
Add new config parameter:
```python
heteroscedastic_recon_weight: float = 50.0  # Separate from recon_weight
```
- Use `recon_weight` for standard decoders
- Use `heteroscedastic_recon_weight` for heteroscedastic decoders
- Allows independent tuning

#### Next Steps
1. Implement one of the loss scaling fixes above
2. Re-run heteroscedastic validation experiment
3. Verify:
   - Reconstruction loss comparable to baseline (~25-50 range)
   - All components active (K_eff > 8)
   - Better accuracy (>30%)
   - Reasonable certainty range
4. Add variance analysis:
   - Check σ distribution (should be ~0.1-0.2, not collapsed)
   - Correlate σ with reconstruction error
   - Visualize learned uncertainty

### Positive Outcomes Despite Issues

**What Worked**:
- ✅ Integration successful - all code paths handle tuples correctly
- ✅ Training runs without errors - no crashes or exceptions
- ✅ Faster convergence (51 vs 118 epochs) - though to wrong solution
- ✅ Backward compatibility maintained - standard decoder still works perfectly
- ✅ All visualizations generated successfully

**What Needs Fixing**:
- ⚠️ Loss scale mismatch causing component collapse
- ⚠️ Need appropriate recon_weight for heteroscedastic loss
- ⚠️ May need additional regularization to stabilize mixture training

---

## Session 2 Git Commits

```bash
6e52df8 - Add heteroscedastic decoder integration for network and visualization
```

**Files Modified**:
- src/ssvae/network.py: Tuple handling in forward pass
- src/ssvae/models.py: Tuple handling in predict methods
- use_cases/experiments/src/visualization/plotters.py: Tuple handling in plots

**Branch Status**: All commits pushed ✅

---

## Updated Summary

**Implementation Status**: ✅ **INTEGRATION COMPLETE, NEEDS TUNING**
- Core heteroscedastic decoder: ✅ Implemented
- Unit tests: ✅ All 25 tests passing
- Integration: ✅ Network, models, visualization
- Validation experiments: ⚠️ **Requires loss scaling fix**

**Validation Results**: ⚠️ **COMPONENT COLLAPSE DUE TO LOSS SCALE**
- Heteroscedastic model collapsed to 1 component (K_eff = 1.00)
- Reconstruction loss 600x higher than baseline
- Need to reduce `recon_weight` or normalize loss

**Next Steps**:
1. Implement loss scaling fix (Option 1, 2, or 3 above)
2. Re-run validation experiment
3. Analyze variance distributions
4. Compare uncertainty quantification quality
5. Document final results

**Branch**: `claude/heteroscedastic-decoder-review-011CUzhjm2QDuKwX8fgtMVsQ`
**Last Commit**: `6e52df8` (integration fixes)
