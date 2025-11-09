# Vision to Implementation Gap

> **Purpose:** High-level comparison between the RCM-VAE vision ([Conceptual Model](conceptual_model.md), [Math Spec](mathematical_specification.md)) and current implementation status.
>
> **For detailed status:** See [Implementation Status](../development/STATUS.md)

---

## Overview

The project implements a **responsibility-conditioned mixture VAE** (RCM-VAE) for semi-supervised learning with active learning and OOD detection. This document tracks the gap between the full vision and current implementation.

**Current State (Nov 2025):** Core mixture VAE complete with component-aware decoding. τ-classifier is the next priority to unlock full RCM-VAE functionality.

---

## Architecture Status

| Component | Vision | Current Status | Next Steps |
|-----------|--------|----------------|------------|
| **Prior** | Interchangeable (MoG/Vamp/Flow) | ✅ Mixture of Gaussians<br>📋 VampPrior ready (optional) | Implement VampPrior if spatial clustering needed |
| **Encoder** | $q_\phi(c\|x)$ responsibilities + $q_\phi(z\|x,c)$ | ✅ Complete | - |
| **Decoder** | Component-aware $p_\theta(x\|z,c)$ | ✅ Complete (Nov 9) | - |
| **Classifier** | τ-based: $p(y\|x) = \sum_c q(c\|x)\tau_{c,y}$ | 🎯 **Next priority** | Replace separate classifier head |
| **Variance** | Heteroscedastic $\sigma(x)$ | 📋 Planned | Add after τ-classifier |
| **OOD Detection** | $1 - \max_c r_c \cdot \max_y \tau_{c,y}$ | 📋 Blocked by τ-classifier | Implement OOD scoring |
| **Dynamic Labels** | Free channel detection + assignment | 📋 Blocked by OOD | Enable incremental label addition |

**Legend:** ✅ Complete | 🎯 In progress | 📋 Planned

---

## Training Features

| Feature | Vision | Current Status |
|---------|--------|----------------|
| **Semi-supervised** | Labeled + unlabeled data | ✅ Complete |
| **Mixture Prior** | K-component mixture with diversity control | ✅ Complete |
| **Component-Aware Decoder** | Specialized decoding per component | ✅ Complete |
| **KL Annealing** | Gradual KL weight increase | ✅ Complete |
| **Early Stopping** | Validation-based stopping | ✅ Complete |
| **Callback System** | Extensible observability | ✅ Complete |
| **Top-M Gating** | Efficient component selection | 📋 Planned (efficiency) |

---

## Key Capabilities

| Capability | Vision | Current Status |
|------------|--------|----------------|
| **Classification** | Latent-only via responsibilities + τ map | 🎯 Needs τ-classifier |
| **Uncertainty** | Aleatoric (σ) + Epistemic (sampling) | Partial (epistemic only) |
| **OOD Detection** | Responsibility × label confidence | 📋 Blocked by τ-classifier |
| **Active Learning** | Query disagreement + OOD | Partial (infrastructure ready) |
| **Multimodality** | Multiple components per label | ✅ Supported by architecture |
| **Interpretability** | 2D visualization, component analysis | ✅ Complete |

---

## Implementation Priorities

**Immediate (unlocks full RCM-VAE):**
1. **τ-classifier** - Latent-only classification via responsibility-label map
2. **OOD detection** - Leverage τ-classifier for out-of-distribution scoring

**Near-term (enhancements):**
3. **Heteroscedastic decoder** - Per-input variance for aleatoric uncertainty
4. **Top-M gating** - Efficiency improvement for large K

**Future (optional):**
5. **VampPrior** - Alternative prior for spatial clustering
6. **Dynamic label addition** - Incremental label assignment to free channels

---

## Dependency Graph

```
✅ Mixture prior → ✅ Entropy reward → ✅ Component-aware decoder
                                              ↓
                                    🎯 τ-classifier
                                      ↙        ↘
                          📋 OOD detection   📋 Heteroscedastic decoder
                                  ↓
                          📋 Dynamic labels
```

---

## Validation Status

**Component-Aware Decoder (Nov 9, 2025):**
- ✅ Embeddings diverge and form natural families
- ✅ Per-component reconstructions show distinct patterns
- ✅ Reconstruction improvement: +1.7% vs standard decoder
- ⚠️ Classification gap: -18% (expected - needs τ-classifier)

**Mixture Prior Stability:**
- ✅ Maintains 6-9/10 active components with entropy reward
- ✅ K_eff = 5.8-6.7 (healthy diversity)
- ✅ No mode collapse with proper configuration

**Key Finding:** Components specialize by visual features (thickness, curvature), not digit labels. This validates the need for τ-based classification to map feature-components → labels.

---

## What "Complete" Looks Like

The full RCM-VAE system will provide:

1. **Latent-only classification** - No separate classifier head, predictions from responsibilities + τ map
2. **Uncertainty quantification** - Aleatoric (σ) + epistemic (latent sampling)
3. **OOD detection** - Identify samples not owned by any labeled component
4. **Active learning** - Query high-uncertainty and OOD samples
5. **Multimodal support** - Multiple components per class naturally handled
6. **Dynamic growth** - Add new labels to free components over time

**Current gap:** Items 1, 3, 4, 6 blocked by τ-classifier. Item 2 needs heteroscedastic decoder.

---

## Related Documentation

**Theory (stable reference):**
- **[Conceptual Model](conceptual_model.md)** - Mental model and core invariants
- **[Mathematical Specification](mathematical_specification.md)** - Precise formulations

**Implementation (changing):**
- **[Implementation Status](../development/STATUS.md)** - Detailed feature status and recent updates
- **[Implementation Decisions](../development/DECISIONS.md)** - Architectural choices and rationale
- **[System Architecture](../development/architecture.md)** - Design patterns in the codebase
