# Implementation Roadmap

> **Purpose:** Bridge between the RCM-VAE vision ([Conceptual Model](conceptual_model.md), [Math Spec](mathematical_specification.md)) and current implementation status.
>
> **For theory:** See [Conceptual Model](conceptual_model.md) | **For math:** See [Mathematical Specification](mathematical_specification.md)  
> **For code:** See [Architecture](../development/architecture.md) | **For usage:** See [Experiment Guide](../../EXPERIMENT_GUIDE.md)


---

## Status at a Glance (Nov 2025)

**Current State:** ✅ τ-Classifier integrated | 🎯 Validation experiments next | 📊 Full RCM-VAE architecture ready

| Feature | Status | Reference |
|---------|--------|-----------|
| **Mixture prior with diversity control** | ✅ Production | [Entropy reward](#entropy-reward-configuration) |
| **Component-aware decoder** $p_\theta(x\|z,c)$ | ✅ Complete (Nov 9) | [Details](#-component-aware-decoder-completed) |
| **Latent-only classifier** via $\tau$ | ✅ **Complete (Nov 10)** | [Details](#-tau-classifier-completed) |
| **Heteroscedastic decoder** $\sigma(x)$ | 🎯 **Next priority** | [Math Spec §4](mathematical_specification.md) |
| **OOD detection** via $r \times \tau$ | 📋 Ready (τ-classifier available) | [Math Spec §6](mathematical_specification.md) |
| **VampPrior** (optional) | 📋 Alternative prior mode | [Details](#vampprior-optional) |
| **Dynamic label addition** | 📋 Ready (free channel detection available) | [Math Spec §7](mathematical_specification.md) |

**Legend:** ✅ Complete | 🚧 In progress | 🎯 Next up | 📋 Planned | 🔬 Research | ⏸️ Deferred

---

## Core Architecture (What's in the Codebase)

### Production Features

The current codebase implements a **foundational semi-supervised VAE** with the following features:

**Mixture VAE with Component-Aware Decoding:**
- **Prior:** K-channel mixture with $p(z|c) = \mathcal{N}(0, I)$ for all components ([Math Spec §3.1](mathematical_specification.md))
- **Encoder:** Amortized $q_\phi(c|x)$ (responsibilities) and $q_\phi(z|x,c)$ ([Math Spec §3.2](mathematical_specification.md))
- **Decoder:** Component-aware with separate pathways for $z$ and $e_c$ ([Details below](#-component-aware-decoder-completed))
- **Classifier:** Separate head on $z$ (to be replaced with τ-based predictor)
- **Channel weights:** Fixed uniform $\pi$ (learnable option available)


**Training Infrastructure:**
- Semi-supervised training loop with labeled and unlabeled data
- Loss function components: reconstruction, KL divergence, classification
- **Entropy-based diversity regularization** on mixture components ([Details](#mode-collapse-prevention-resolved))
- Optional Dirichlet prior on mixture weights $\pi$
- Callback-based training observability (logging, plotting)
- Early stopping and checkpoint management

**Experimentation Tools:**
- Single-experiment runner with timestamped outputs
- Comprehensive visualizations (loss curves, latent spaces, reconstructions, mixture evolution)
- Component divergence analysis (embedding heatmaps, per-component reconstructions)
- Interactive dashboard for active learning (scaffold in place)
- Comprehensive test suite (49 tests: unit, integration, regression)

### Design Patterns

**Protocol-based extensibility:**
```python
class PriorMode(Protocol):
    """Pluggable prior interface - add VampPrior without refactoring."""
    def kl_divergence(...) -> Array: ...
    def sample(...) -> Array: ...
```

**Factory pattern for components:**
- `build_encoder()`, `build_decoder()`, `build_classifier()`
- Automatically selects component-aware vs standard decoder based on config
- Centralized validation and parameter resolution

**Configuration-driven:**
- All hyperparameters in `SSVAEConfig` dataclass (25+ parameters)
- YAML-based experiment configs with inheritance support
- Backward compatibility maintained for legacy parameters

---

## Implementation Status

### ✅ Component-Aware Decoder (Completed)

**Status:** Production-ready as of Nov 9, 2025

**What it does:**  
Gives each component a learnable embedding $e_c$ that specializes the decoding pathway. Components develop functional identity (e.g., "decodes thick strokes well") without requiring spatial separation in latent space.

**Implementation:**
- `ComponentAwareDenseDecoder` and `ComponentAwareConvDecoder` classes
- Separate processing: $z \to \text{Dense}(\text{hidden}/2)$ and $e_c \to \text{Dense}(\text{hidden}/2)$, then concatenate
- Component embeddings stored in prior: `params['prior']['component_embeddings']` (shape: `[K, embedding_dim]`)
- Factory auto-selects based on `use_component_aware_decoder: true` (default for mixture prior)

**Validation (see `COMPONENT_AWARE_DECODER_ABLATION.md`):**
- ✅ Embeddings diverge: mean pairwise distance = 0.54 (std = 0.19)
- ✅ Form natural families (block structure in distance heatmap)
- ✅ Per-component reconstructions show distinct visual patterns
- ✅ Reconstruction improvement: +1.7% vs standard decoder
- ⚠️ Classification gap: -18% (expected - current classifier ignores components)

**Key finding:** Components specialize by **visual features** (thickness, roundness), not digit labels. This validates the need for τ-classifier.

**Theory:** [Conceptual Model - "Why This Shape"](conceptual_model.md) | **Math:** [Math Spec §3.3](mathematical_specification.md)

---

### ✅ τ-Classifier (Completed)

**Status:** Production-ready as of Nov 10, 2025

**What it does:**
Maps components to labels via soft count statistics, replacing the separate classifier head. This enables the model to leverage component specialization for classification while supporting natural multimodality (multiple components per label).

**Implementation:**
- `TauClassifier` class in `src/ssvae/components/tau_classifier.py`
- Soft count accumulation: $s_{c,y} \leftarrow s_{c,y} + q(c|x) \cdot \mathbf{1}\{y=y_i\}$
- Normalized probability map: $\tau_{c,y} = (s_{c,y} + \alpha_0) / \sum_{y'} (s_{c,y'} + \alpha_0)$
- Prediction: $p(y|x) = \sum_c q(c|x) \cdot \tau_{c,y}$
- Stop-gradient on τ in supervised loss (gradients flow through $q(c|x)$ only)
- Trainer integration via `TrainerLoopHooks` (batch/eval context + post-batch updates)
- Vectorized count updates (`responsibilities.T @ one_hot(labels)`) for scalability
- Training-time guardrails: `num_components >= num_classes`, labeled-regime logging/warnings
- Configuration: `use_tau_classifier: true` (default for mixture prior), `tau_smoothing_alpha: 1.0`

**Validation (see `TAU_CLASSIFIER_IMPLEMENTATION_REPORT.md`):**
- ✅ 49 comprehensive unit tests (initialization, count updates, multimodality, stop-gradient)
- ✅ Full training loop integration with count updates after each batch
- ✅ Prediction methods updated to use τ-based classification
- ✅ Backward compatible (standard classifier used when τ disabled)
- 📋 Pending: Validation experiments to confirm accuracy recovery

**Unlocked Capabilities:**
- **OOD Detection:** $1 - \max_c (r_c \times \max_y \tau_{c,y})$ via `get_ood_score()`
- **Free Channel Detection:** Identify components available for new labels via `get_free_channels()`
- **Active Learning:** Uncertainty-based acquisition via responsibility entropy + τ confidence
- **Diagnostics:** Component→label associations via `get_diagnostics()`

**Next Steps:**
- Run ablation experiments (`tau_classifier_validation.yaml`)
- Compare accuracy vs standard classifier head
- Analyze τ matrix structure (sparse but multi-hot)
- Validate component-label alignment improvement

**Theory:** [Conceptual Model - "How We Classify"](conceptual_model.md) | **Math:** [Math Spec §5](mathematical_specification.md)

---

### 📋 Near-Term Enhancements (After τ-Classifier)

**Heteroscedastic Decoder:**
- Add variance head: $\sigma(x) = \sigma_{\min} + \text{softplus}(s_\theta(x))$
- Clamp $\sigma(x) \in [0.05, 0.5]$ for stability
- Reconstruction loss: $\frac{\|x - \hat{x}\|^2}{2\sigma^2} + \log \sigma$
- **Enables:** Aleatoric uncertainty quantification per input
- **Theory:** [Conceptual Model - "Why This Shape"](conceptual_model.md) | **Math:** [Math Spec §4](mathematical_specification.md)

**Top-M Gating (Efficiency):**
- Currently: decode all K components (expensive for large K)
- **Proposal:** Only decode top-M components by responsibility
- Config: `top_m_gating: 5` (0 = use all)
- **Expected:** 2-3× speedup with minimal accuracy loss
- **Math:** [Math Spec §3.3](mathematical_specification.md) - mentioned as default $M=5$

---

### 📋 Medium-Term Features (Blocked by Dependencies)

**OOD Detection:** (requires τ-classifier first)
- Score: $s_{\text{OOD}} = 1 - \max_c r_c(z) \cdot \max_y \tau_{c,y}$
- "Not owned by any labeled component"
- Integrate with active learning workflows
- Test on Fashion-MNIST and unseen digits
- **Theory:** [Conceptual Model - "How We Detect OOD"](conceptual_model.md) | **Math:** [Math Spec §6](mathematical_specification.md)

**Dynamic Label Addition:** (requires τ-classifier + OOD)
- Free channel detection: `(usage < 1e-3) OR (max_y τ_{c,y} < 0.05)`
- New label claims 1-3 free channels via high-responsibility examples
- Dashboard integration for interactive labeling
- **Math:** [Math Spec §7](mathematical_specification.md)

**VampPrior (Optional):**
- **Purpose:** Spatial clustering in latent space (alternative to component-aware decoder)
- **Note:** With component-aware decoder working, VampPrior is optional
- Learns pseudo-inputs $u_1, \ldots, u_K$ where $p(z|c) = q_\phi(z|u_c)$
- **Trade-off:** More complex; induces spatial topology vs functional specialization
- **Use case:** When spatial visualization is priority
- **Theory:** [Conceptual Model](conceptual_model.md) | **Math:** [Math Spec §3.1(B)](mathematical_specification.md)

---

## Key Findings & Decisions

### Mode Collapse Prevention (Resolved)

**Problem:** Early experiments showed catastrophic collapse to 1/10 components (K_eff = 1.0)

**Solution:** Entropy reward (not sparsity penalty)
```yaml
component_diversity_weight: -0.05  # NEGATIVE = reward diversity
dirichlet_alpha: 5.0               # Stronger prior (vs default 1.0)
kl_c_weight: 0.001                 # 2× baseline
```

**Result:** ✅ Maintains 6-9/10 active components, K_eff = 5.8-6.7

**Why it works:**  
Negative weight on $-H(\hat{p}(c))$ encourages high entropy → diverse component usage. Dirichlet prior prevents extreme concentration. Note: `component_diversity_weight` is a misnomer when negative (should be "diversity reward").

**Evolution:**

| Configuration | Active Components | K_eff | Outcome |
|---------------|-------------------|-------|---------|
| Positive weight (+0.1) | 1/10 | 1.0 | Collapse |
| Negative weight (-0.05) | 6-9/10 | 5.8-6.7 | ✅ Healthy |

---

### Components Cluster by Features, Not Labels

**Observed:** Component majority labels `[8, 0, 7, 0, 0, 1, 0, 0, 1, 9]` — four components prefer "0"!

**Analysis:**  
Components don't organize by digit identity. They specialize by **visual features**:
- Stroke thickness (thin vs bold)
- Aspect ratio (tall vs wide)  
- Curvature characteristics

**Implication:**
- A "thick 0" and "thick 8" may share the same component
- This is **architecturally correct** per [Conceptual Model](conceptual_model.md): components find useful structure, not necessarily class labels
- τ-classifier will bridge: "this feature-component → this label"

**Metrics:**
- NMI = 0.76 (moderate mutual information with labels)
- ARI ≈ 0 (clusters don't align with digit boundaries)
- This is **expected and acceptable** with only 50 labeled samples

**Why ARI = 0 is OK:**
1. Encoder is mostly unsupervised → optimizes reconstruction, not classification
2. Visual features span multiple digit classes
3. τ-classifier will map feature-components to labels explicitly

**Expected with τ-classifier:**
- Components maintain feature-based organization (good for reconstruction)
- τ map learns which features → which labels (good for classification)
- Classification improves without forcing spatial clustering

---

### Latent Space Structure

**Observation:** "Latent by Component" plots show heavy spatial overlap (all components near origin)

**Why this happens:**  
All priors are identical: $p(z|c) = \mathcal{N}(0, I)$ for every component. The KL term pulls all encodings toward origin. No spatial separation is enforced.

**Is this a problem?** ❌ No, it's by design.

**Per [Math Spec §6](mathematical_specification.md):**  
> "When all $p(z|c)$ are identical standard normals, latent density alone is uninformative by design; $r$, $\tau$, and reconstruction carry the signal."

**What differentiates components?**
- **Decoder pathways:** Each $e_c$ specializes the reconstruction (component-aware decoder)
- **Responsibilities:** Encoder assigns $q(c|x)$ based on which component reconstructs best
- **τ mappings:** Components associate with labels via soft counts

**Spatial separation is NOT required** for functional specialization. If spatial clustering is desired, use VampPrior instead.

---

## Configuration & Tuning

### Entropy Reward Configuration

**Effective settings for preventing collapse:**
```yaml
model:
  component_diversity_weight: -0.05  # Negative = reward entropy
  dirichlet_alpha: 5.0               # Stronger than default 1.0
  kl_c_weight: 0.001                 # Component KL weight
```

**Naming clarification:**  
`component_diversity_weight` is a misnomer. When negative, it rewards diversity (maximizes entropy). When positive, it enforces sparsity (minimizes entropy). Consider renaming to `component_diversity_reward` or adding `encourage_diversity: bool`.

### Component-Aware Decoder Configuration

**Recommended settings:**
```yaml
model:
  prior_type: "mixture"
  use_component_aware_decoder: true  # Default for mixture prior
  component_embedding_dim: 8         # Small (4-16) to not overwhelm latent
  num_components: 10                 # K value
```

**Optional tuning:**
- Larger `component_embedding_dim` (16) for more expressive embeddings
- Asymmetric pathway split (75% for z, 25% for e_c) — not yet implemented
- Soft-embedding warm-up (first 5-10 epochs) — not yet implemented

---

## Dependency Graph

**Current path (locked in):**
```
✅ Mixture prior → ✅ Entropy reward → ✅ Component-aware decoder
```

**Next steps (sequential dependencies):**
```
🎯 τ-classifier → 📋 OOD detection → 📋 Dynamic labels
              ↘ 📋 Heteroscedastic decoder (parallel track)
              ↘ 📋 Top-M gating (efficiency, parallel track)
```

**Optional enhancements (no dependencies):**
```
📋 VampPrior (alternative to component-aware decoder for spatial clustering)
```

---

## Related Documentation

### Theory & Vision
- **[Conceptual Model](conceptual_model.md)** — Stable mental model and core invariants
- **[Mathematical Specification](mathematical_specification.md)** — Precise objectives, training protocol, math formulations

### Implementation & Code
- **[System Architecture](../development/architecture.md)** — Design patterns and component structure
- **[Implementation Guide](../development/implementation.md)** — Module-by-module code reference
- **[Extending the System](../development/extending.md)** — How to add new features (step-by-step tutorials)

### Usage & Experimentation
- **[Experiment Guide](../../EXPERIMENT_GUIDE.md)** — Configuration → execution → interpretation workflow
- **[Verification Checklist](../../VERIFICATION_CHECKLIST.md)** — Comprehensive regression testing

### Experiment Reports
- **`COMPONENT_AWARE_DECODER_ABLATION.md`** — Controlled ablation study (Nov 9, 2025)
- **`experiments/runs/PROGRESS_REPORT_20251109.md`** — Overall progress assessment
