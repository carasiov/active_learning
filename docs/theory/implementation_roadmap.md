# Implementation Roadmap

> **Purpose:** Bridge document connecting the current codebase implementation to the full RCM-VAE vision described in the [Conceptual Model](conceptual_model.md) and [Mathematical Specification](mathematical_specification.md).

---

## Current Implementation Status

### What We Have

The current codebase implements a **foundational semi-supervised VAE** with the following features:

**Core Architecture:**
- **Mixture of Gaussians prior** with learnable mixture weights $\pi$ and identical $p(z|c) = \mathcal{N}(0, I)$ for all components
- **Standard encoder-decoder** architecture (dense and convolutional options)
- **Separate classifier head** for supervised classification
- **Component factory pattern** via `PriorMode` protocol for pluggable priors
- **Modular design:** Clean separation of components (encoders, decoders, classifiers, priors)

**Training Infrastructure:**
- Semi-supervised training loop with labeled and unlabeled data
- Loss function components: reconstruction, KL divergence, classification
- **Entropy-based diversity regularization** on mixture components (negative usage sparsity weight)
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

**Recent Achievements (Nov 2025):**
- ✅ **Resolved mode collapse:** Successfully tuned entropy reward (`component_diversity_weight: -0.05`) and Dirichlet prior (`alpha: 5.0`) to maintain 9/10 active components
- ✅ **Validated parsimony mechanisms:** K_eff = 6.7 with confident responsibility assignments (mean = 0.87)
- ✅ **Demonstrated multimodal capability:** Multiple components naturally map to same labels via $\tau$
- ⚠️ **Identified architectural limitation:** Without component-aware decoder or VampPrior, components lack spatial separation in latent space (all encode near origin due to identical $\mathcal{N}(0,I)$ priors)

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

**CRITICAL NEXT STEP (Immediate Priority):**

**🎯 Component-aware decoder** - **STATUS: URGENT**

**Why This Is Critical:**
Current experiments (Nov 2025) show components have **no spatial separation** in latent space—all points cluster near origin with overlapping component assignments. This is because:
- All priors are identical: $p(z|c) = \mathcal{N}(0, I)$ for every component
- Decoder doesn't see component identity: $p_\theta(x|z)$ (not $p_\theta(x|z,c)$)
- Components can only differentiate through $\tau$ mappings, not functional specialization

**Impact:** Components exist as "soft labels" without architectural grounding. They cannot develop specialized decoding strategies (e.g., "component 3 decodes thick strokes well"). This limits representation power and makes latent space visualizations uninformative.

**Implementation Plan:**
1. **Decoder modification:**
   ```python
   # Current: decode(z) → x
   # Target:  decode([z; e_c]) → x  where e_c is learned component embedding
   ```
   - Add `component_embeddings` parameter to decoder (shape: `[K, embedding_dim]`)
   - Concatenate `e_c` with `z` before first decoder layer
   - Keep embedding_dim small (4-16) to avoid overwhelming latent information

2. **Training with Top-$M$ gating (efficiency):**
   - Compute reconstruction as weighted sum over top-M components (default M=5)
   - Full K-way sum is expensive; top-M approximation validated in spec
   - Implementation: `top_m_indices = jnp.argsort(q_c)[-M:]`

3. **Optional soft-embedding warm-up:**
   - First 5-10 epochs: use soft weighting `z_tilde = [z; sum_c q(c|x) * e_c]`
   - Later epochs: sample hard c and use `z_tilde = [z; e_c]`
   - Helps early training stability (similar to Gumbel-Softmax annealing)

4. **Testing & validation:**
   - Verify components develop distinct decoding strategies (check `component_embeddings` divergence)
   - Expect improved reconstruction and classification accuracy
   - Monitor for over-fragmentation (components becoming too specialized)

**Expected Outcomes:**
- Components gain **functional identity** beyond soft label assignments
- Better reconstruction quality (specialized decoders per component)
- Improved classification (components aligned with semantic features)
- Latent space may remain spatially overlapping (this is acceptable per spec)

**Reference:** [Math Spec Section 3.3](mathematical_specification.md) - Component-Aware Decoder

---

**Near-term (Foundation - After Component-Aware Decoder):**

2. **Responsibility-based classifier:**
   - Replace separate classifier head with $\tau$ map computation
   - Implement soft count accumulation $s_{c,y}$
   - Add stop-gradient treatment in supervised loss
   - **Blocker:** Should be done AFTER component-aware decoder to ensure components have meaningful specialization

3. **Heteroscedastic decoder:**
   - Add variance head to decoder (per-image scalar)
   - Implement variance clamping $\sigma(x) \in [0.05, 0.5]$
   - Integrate into reconstruction loss: $-\mathbb{E}[\log p(x|z,c)] = \frac{||x - \hat{x}||^2}{2\sigma^2} + \log \sigma$
   - Enables aleatoric uncertainty quantification

**Medium-term (Advanced Features):**

4. **VampPrior implementation (Alternative to Component-Aware Decoder):**
   - **Purpose:** If spatial clustering is desired, VampPrior provides learned per-component centers
   - Pseudo-input learning: $u_1, \ldots, u_K$ where $p(z|c) = q_\phi(z|u_c)$
   - Prior shaping (MMD/MC-KL) to match target distributions
   - Integration via existing `PriorMode` protocol
   - **Trade-off:** More complex than component-aware decoder; induces topology in latent space
   - **Decision Point:** Choose ONE of {component-aware decoder, VampPrior} based on whether spatial clustering is needed

5. **OOD detection:**
   - Implement $r \times \tau$ scoring: $s_{\text{OOD}} = 1 - \max_c r_c(z) \cdot \max_y \tau_{c,y}$
   - Integrate with active learning workflows
   - Add to comparison tool metrics
   - Test on Fashion-MNIST and unseen MNIST digits

6. **Dynamic label addition:**
   - Free channel detection logic: `is_free = (usage < 1e-3) OR (max_y tau_{c,y} < 0.05)`
   - Channel claiming mechanism for new labels
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

## Implementation Notes from Nov 2025 Experiments

### Key Finding: Components Need Functional Differentiation

**Observation:** With identical priors ($p(z|c) = \mathcal{N}(0,I)$ for all $c$) and no component-aware decoder, all components encode to the same spatial region (near origin). Components differentiate only through:
- Soft label assignments via $\tau$ map
- Subtle directional biases in latent space (visible in plots but weak)

**Implication:** This validates the spec's design choice: components should specialize via **decoder pathways** (component-aware decoder) OR **prior centers** (VampPrior), not just responsibilities.

**Visualization Impact:** "Latent by Component" plots show heavy overlap—this is expected and acceptable per [Math Spec Section 6](mathematical_specification.md): *"When all $p(z|c)$ are identical standard normals, latent density alone is uninformative by design."*

### Entropy Reward Discovery

**Effective configuration for preventing collapse:**
```yaml
component_diversity_weight: -0.05  # NEGATIVE = entropy reward (not penalty)
dirichlet_alpha: 5.0               # Stronger than default 1.0
kl_c_weight: 0.001                 # 2× baseline to prevent concentration
```

**Naming clarification:** The parameter was renamed from `usage_sparsity_weight` to `component_diversity_weight` for clarity. When negative, it implements **diversity encouragement**. The spec's "minimize entropy" phrasing assumes positive weight (minimize $-H$ = maximize $H$); our negative weight achieves same effect but inverts the algebra.

### Component-Label Alignment

**Observed:** Component majority labels `[9,5,6,7,7,6,5,6,6,6]` show natural multimodality (multiple components per digit) without explicit supervision. This validates the soft-count $s_{c,y} \to \tau_{c,y}$ mechanism.

**Negative ARI = -0.096:** Components don't align with digit identity because:
1. No architectural pressure to cluster by label (encoder is mostly unsupervised)
2. Decoder doesn't differentiate by component (no specialization incentive)
3. This will improve with component-aware decoder + stronger $\tau$-based supervision

---

## Related Documentation

- **[Conceptual Model](conceptual_model.md)** - High-level vision and mental model
- **[Mathematical Specification](mathematical_specification.md)** - Precise mathematical formulations
- **[System Architecture](../development/architecture.md)** - Current implementation design patterns
- **[Implementation Guide](../development/implementation.md)** - Module reference and code organization
- **[Extending the System](../development/extending.md)** - How to add new components (VampPrior, component-aware decoder, etc.)
