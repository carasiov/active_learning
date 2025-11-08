# Implementation Roadmap

> **Purpose:** Bridge document connecting the current codebase implementation to the full RCM-VAE vision described in the [Conceptual Model](conceptual_model.md) and [Mathematical Specification](mathematical_specification.md).

---

## Current Implementation Status

### What We Have

The current codebase implements a **foundational semi-supervised VAE** with the following features:

**Core Architecture:**
- **Mixture of Gaussians prior** with learnable mixture weights $\pi$
- **Standard encoder-decoder** architecture (dense and convolutional options)
- **Separate classifier head** for supervised classification
- **Component factory pattern** via `PriorMode` protocol for pluggable priors
- **Modular design:** Clean separation of components (encoders, decoders, classifiers, priors)

**Training Infrastructure:**
- Semi-supervised training loop with labeled and unlabeled data
- Loss function components: reconstruction, KL divergence, classification
- Usage sparsity regularization on mixture components
- Optional Dirichlet prior on mixture weights $\pi$
- Callback-based training observability (logging, plotting)
- Early stopping and checkpoint management

**Experimentation Tools:**
- Comparison tool for systematic model evaluation
- Interactive dashboard for active learning workflows
- Comprehensive test suite (49 tests: unit, integration, regression)

**Key Design Decisions:**
- **Protocol-based prior abstraction:** The `PriorMode` protocol enables easy integration of new priors (VampPrior, flows, etc.) without modifying core model code
- **Configuration-driven:** All hyperparameters exposed via `SSVAEConfig` dataclass
- **Factory pattern:** Centralized component creation with validation

### What We're Building Toward

The full RCM-VAE system as specified in the theoretical documents includes:

**Key Additions:**
1. **Component-aware decoder** $p_\theta(x|z,c)$ - decoder receives channel embedding
2. **Latent-only classifier** via responsibilities $r_c(x)$ and channel→label map $\tau_{c,y}$
3. **OOD scoring** using $1 - \max_c r_c(z) \cdot \max_y \tau_{c,y}$
4. **Dynamic label addition** with free channel detection
5. **VampPrior** support (architecture ready, needs implementation)
6. **Heteroscedastic decoder** with per-image variance $\sigma(x)$

---

## Architectural Readiness

### What's Architecturally Ready

**PriorMode Protocol:**
The current architecture already supports pluggable priors through the `PriorMode` protocol:

```python
# From src/ssvae/components/priors/protocol.py
class PriorMode(Protocol):
    """Protocol defining interface for different prior modes."""

    def kl_divergence(self, z_mean, z_logvar, component_logits=None) -> Array:
        """Compute KL divergence for this prior mode."""
        ...

    def sample(self, key, latent_dim, num_samples=1) -> Array:
        """Sample from the prior distribution."""
        ...
```

This means:
- ✅ Adding VampPrior requires implementing the protocol, not refactoring core code
- ✅ Component-aware features can be added incrementally
- ✅ Existing mixture prior provides foundation for responsibility-based methods

**Modular Components:**
- Encoders, decoders, and classifiers are separate modules
- Easy to extend decoder to accept channel embeddings
- Classifier can be replaced with latent-only $\tau$-based predictor

### What Needs Development

**Near-term (Foundation):**
1. **Component-aware decoder:**
   - Extend decoder to accept channel index/embedding
   - Implement Top-$M$ gating for efficient training
   - Add soft-embedding warm-up option

2. **Responsibility-based classifier:**
   - Replace separate classifier head with $\tau$ map computation
   - Implement soft count accumulation $s_{c,y}$
   - Add stop-gradient treatment in supervised loss

3. **Heteroscedastic decoder:**
   - Add variance head to decoder (per-image scalar)
   - Implement variance clamping
   - Integrate into reconstruction loss

**Medium-term (Advanced Features):**
4. **VampPrior implementation:**
   - Pseudo-input learning
   - Prior shaping (MMD/MC-KL)
   - Integration via existing `PriorMode` protocol

5. **OOD detection:**
   - Implement $r \times \tau$ scoring
   - Integrate with active learning workflows
   - Add to comparison tool metrics

6. **Dynamic label addition:**
   - Free channel detection logic
   - Channel claiming mechanism
   - Dashboard integration for interactive labeling

---

## Design Continuity

The current implementation was explicitly designed to support the RCM-VAE vision:

- **Protocol-based priors:** Enable VampPrior without refactoring
- **Factory pattern:** Centralized place to add component-aware features
- **Modular losses:** Easy to add $\tau$-based supervised loss
- **Comprehensive testing:** Safety net for incremental changes
- **Comparison tool:** Ready to evaluate new features systematically

The path forward is **incremental enhancement**, not refactoring. Each RCM-VAE feature can be added as an option while maintaining backward compatibility with the current standard/mixture prior modes.

---

## Related Documentation

- **[Conceptual Model](conceptual_model.md)** - High-level vision and mental model
- **[Mathematical Specification](mathematical_specification.md)** - Precise mathematical formulations
- **[System Architecture](../development/architecture.md)** - Current implementation design patterns
- **[Implementation Guide](../development/implementation.md)** - Module reference and code organization
- **[Extending the System](../development/extending.md)** - How to add new components (VampPrior, component-aware decoder, etc.)
