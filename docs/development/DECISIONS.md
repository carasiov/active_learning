# Implementation Decisions

> **Purpose:** Document key architectural choices, trade-offs considered, and rationale behind design decisions.
>
> **Last Updated:** November 9, 2025

---

## Overview

This document captures significant implementation decisions made during development of the SSVAE/RCM-VAE system. Each decision includes:
- **Context:** What problem we were solving
- **Options considered:** Alternative approaches
- **Decision:** What we chose
- **Rationale:** Why we made this choice
- **Trade-offs:** What we gained and lost
- **Outcome:** How it worked out

---

## Design Decisions

### Mode Collapse Prevention: Entropy Reward vs. Sparsity Penalty

**Context:**

Early experiments with mixture prior showed catastrophic mode collapse - all data assigned to 1 out of 10 components (K_eff = 1.0).

**Options Considered:**

1. **Sparsity penalty** (positive weight): Minimize $-H(\hat{p}(c))$ to encourage sparse component usage
2. **Entropy reward** (negative weight): Maximize $H(\hat{p}(c))$ to encourage diverse component usage
3. **Uniform KL penalty**: Penalize deviation from uniform prior
4. **No regularization**: Let components organize naturally

**Decision:**

Use **entropy reward** with negative `component_diversity_weight`:

```yaml
component_diversity_weight: -0.05  # NEGATIVE = reward high entropy
dirichlet_alpha: 5.0                # Stronger Dirichlet prior
kl_c_weight: 0.001                  # Mild component KL
```

**Rationale:**

- Sparsity penalty (option 1) **caused** the collapse - it encouraged using fewer components
- Entropy reward (option 2) directly opposes collapse by rewarding diverse usage
- Dirichlet prior provides soft preference for uniform π
- Component KL prevents components from becoming too confident

**Trade-offs:**

✅ **Gained:**
- Stable training with 6-9/10 active components
- K_eff = 5.8-6.7 (healthy diversity)
- No mode collapse across multiple experiments

⚠️ **Lost:**
- Can't enforce hard sparsity (all K components get some usage)
- Parameter tuning required (diversity weight, Dirichlet alpha)

**Outcome:**

✅ **Success.** Mixture prior is now stable and production-ready. The entropy reward configuration is recommended for all mixture experiments.

**Reference:** See [STATUS.md - Entropy Reward Configuration](STATUS.md#entropy-reward-for-mode-collapse-prevention)

---

### Components Cluster by Features, Not Labels

**Context:**

After implementing component-aware decoder, we observed component majority labels like `[8, 0, 7, 0, 0, 1, 0, 0, 1, 9]` - **four components prefer digit "0"**. This seemed like a failure initially.

**Analysis:**

Components don't organize by digit identity. They specialize by **visual features**:
- Stroke thickness (thin vs. bold)
- Aspect ratio (tall vs. wide)
- Curvature characteristics (rounded vs. angular)

**Decision:**

**This is correct behavior, not a bug.** We will:
1. Accept that components cluster by features, not labels
2. Use τ-classifier to map feature-components → labels
3. Allow multiple components per label (natural multimodality)

**Rationale:**

From [Conceptual Model](../theory/conceptual_model.md):
> "Components find useful structure, not necessarily class labels. The decoder conditions on c to specialize reconstruction, and τ maps channels to labels."

**Why this makes sense:**
- Encoder is mostly **unsupervised** (optimizes reconstruction, not classification)
- Only 50 labeled samples out of 10,000 - not enough to force label-based clustering
- Visual features (thickness, curvature) **span multiple digits**
- A "thick 0" and "thick 8" sharing a component is reasonable
- τ-classifier explicitly handles component→label mapping

**Trade-offs:**

✅ **Gained:**
- Natural multimodality (multiple components per digit)
- Better reconstruction (components optimize for visual patterns)
- Cleaner separation of concerns (reconstruction vs. classification)

⚠️ **Accepted:**
- Low NMI and ARI metrics (clusters don't align with digit boundaries)
- Components don't have obvious "semantic" labels
- Need τ-classifier to recover classification performance

**Outcome:**

✅ **Validated as correct.** The -18% classification drop with component-aware decoder confirms components ignore labels. τ-classifier will bridge the gap.

**Metrics:**
- NMI = 0.76 (moderate mutual information with labels)
- ARI ≈ 0 (clusters don't align with digit boundaries)
- Both are **expected and acceptable** with semi-supervised learning (50 labels / 10k samples)

**Next Steps:**
- Implement τ-classifier to leverage feature-based components
- Expect classification to recover (or exceed) standard decoder performance
- Visualize τ matrix to see component→label associations

---

### Latent Space Spatial Overlap is Intentional

**Context:**

"Latent by Component" visualizations show all components heavily overlapping near the origin. This appeared problematic initially.

**Observation:**

All priors are identical: $p(z|c) = \mathcal{N}(0, I)$ for every component. The KL term $\text{KL}(q(z|x,c) || p(z|c))$ pulls all encodings toward the standard normal origin.

**Question:**

Is spatial overlap a bug? Should components be spatially separated in latent space?

**Decision:**

**Spatial overlap is intentional and by design.** Components differentiate via decoder pathways and responsibilities, not spatial location.

**Rationale:**

From [Math Spec §6](../theory/mathematical_specification.md):
> "When all $p(z|c)$ are identical standard normals, latent density alone is uninformative by design; $r$, $\tau$, and reconstruction carry the signal."

**How components differentiate without spatial separation:**

1. **Decoder pathways:** Each component $c$ has embedding $e_c$ that specializes decoding $p(x|z,c)$
2. **Responsibilities:** Encoder assigns $q(c|x)$ based on which component reconstructs best
3. **τ mappings:** Components associate with labels via soft counts from labeled data

**Spatial separation is NOT required for functional specialization.**

**Trade-offs:**

✅ **Gained:**
- Simpler prior (all $p(z|c)$ identical)
- Standard KL divergence computation
- Functional specialization via decoder, not latent geometry
- Easier to implement and understand

⚠️ **Lost:**
- 2D latent visualization doesn't show component separation
- Can't use latent density for OOD detection
- Less intuitive for visualization

**Alternatives:**

If spatial clustering is desired:
- **VampPrior:** Learn pseudo-inputs $u_k$ where $p(z|c) = q(z|u_k)$ induces spatial centers
- **Geometric MoG:** Place Gaussian centers on grid/circle (not recommended - induces artificial topology)

**Outcome:**

✅ **Correct by design.** Spatial overlap is expected and doesn't indicate a problem. OOD detection will use $r \times \tau$ confidence, not latent density.

---

### Protocol-Based Priors vs. Inheritance Hierarchy

**Context:**

Need to support multiple prior types (standard Gaussian, mixture, VampPrior, flows) without modifying core SSVAE code.

**Options Considered:**

1. **Protocol (structural typing):** Define `PriorMode` protocol with required methods
2. **Abstract base class:** Define `BasePrior` ABC with abstract methods
3. **Duck typing:** No formal interface, just expect certain methods to exist
4. **Single prior class:** Use configuration flags to switch behavior

**Decision:**

Use **Protocol-based** abstraction:

```python
from typing import Protocol

class PriorMode(Protocol):
    def kl_divergence(self, z_mean, z_logvar, ...) -> Array: ...
    def sample(self, key, latent_dim, num_samples) -> Array: ...
```

**Rationale:**

- **Static type checking:** MyPy/Pyright can verify implementations without runtime overhead
- **No inheritance:** Priors don't need to inherit from base class
- **Structural typing:** "If it has these methods, it's a valid prior"
- **Flexibility:** Easy to add new priors (VampPrior, flows) without modifying interface
- **JAX-friendly:** Protocols work well with functional programming style

**Trade-offs:**

✅ **Gained:**
- Clean separation of concerns
- Easy to add new priors
- No runtime inheritance overhead
- Static type safety

⚠️ **Lost:**
- No shared implementation (each prior implements everything)
- Less familiar to developers used to OOP inheritance

**Outcome:**

✅ **Success.** Adding `MixturePrior` after `StandardPrior` required zero changes to SSVAE core code. Ready for VampPrior implementation.

**See:** [Architecture - PriorMode Protocol](architecture.md#1-priormode-protocol)

---

### Factory Pattern for Component Creation

**Context:**

Need centralized, validated creation of encoders, decoders, classifiers, and priors with compatible configurations.

**Options Considered:**

1. **Factory pattern:** Centralized `SSVAEFactory.create_*()` methods
2. **Builder pattern:** Fluent API like `SSVAE.builder().with_encoder(...).build()`
3. **Direct instantiation:** User creates components manually and passes to SSVAE
4. **Configuration-based:** SSVAE reads config and creates components internally

**Decision:**

Use **Factory pattern** with `SSVAEFactory`:

```python
factory = SSVAEFactory()
model, state, train_fn, eval_fn, rng, prior = factory.create_model(input_dim, config)
```

**Rationale:**

- **Single source of truth:** All component creation in one place
- **Validation:** Factory ensures compatible configurations (e.g., latent_dim matches architecture)
- **Consistency:** Same creation logic for all SSVAE instances
- **Testability:** Can test factory in isolation
- **Extensibility:** Easy to add new component types

**Trade-offs:**

✅ **Gained:**
- Centralized validation
- Consistent component creation
- Easy to test
- Clear extension point for new components

⚠️ **Lost:**
- Extra indirection (factory instead of direct instantiation)
- More files to navigate

**Outcome:**

✅ **Success.** Factory makes it easy to switch between standard/component-aware decoders, add new encoder types, etc. Validation catches config errors early.

**See:** [Architecture - SSVAEFactory](architecture.md#2-ssvaefactory)

---

### Configuration-Driven Design (SSVAEConfig Dataclass)

**Context:**

Need to expose 25+ hyperparameters with type safety, defaults, and serialization.

**Options Considered:**

1. **Dataclass:** `@dataclass SSVAEConfig` with type annotations
2. **Dictionary:** Plain `dict` with defaults
3. **Pydantic:** Validation library with automatic type checking
4. **Hydra/OmegaConf:** Configuration framework with CLI integration

**Decision:**

Use **dataclass** with explicit types:

```python
@dataclass
class SSVAEConfig:
    latent_dim: int = 2
    prior_type: str = "standard"
    # ... 25+ parameters
```

**Rationale:**

- **Built-in:** No external dependencies (unlike Pydantic, Hydra)
- **Type safety:** Static type checkers work out of the box
- **Serialization:** Easy to convert to dict/YAML
- **IDE support:** Autocomplete, type hints work automatically
- **Simplicity:** Straightforward to understand and extend

**Trade-offs:**

✅ **Gained:**
- Type safety with no dependencies
- Clear defaults
- IDE autocomplete
- Easy serialization

⚠️ **Lost:**
- No advanced validation (e.g., "latent_dim must be > 0")
- No automatic CLI generation
- Manual serialization logic

**Outcome:**

✅ **Success.** Config dataclass makes it easy to create experiments, share configurations, and catch type errors early. Works well with YAML-based experiment configs.

**See:** [API Reference - config.py](api_reference.md#configpy---configuration)

---

### Callback Pattern for Training Observability

**Context:**

Need extensible hooks into training loop for logging, plotting, mixture tracking, etc.

**Options Considered:**

1. **Callback pattern:** Observer pattern with `on_epoch_end()`, etc.
2. **Event emitter:** Publish-subscribe with event bus
3. **Direct integration:** Hard-code logging/plotting into Trainer
4. **Decorator pattern:** Wrap training loop with decorators

**Decision:**

Use **Callback pattern** with base class:

```python
class TrainingCallback:
    def on_train_begin(self): pass
    def on_epoch_end(self, epoch, history): pass
    def on_train_end(self): pass

# Usage
callbacks = [ConsoleLogger(), LossCurvePlotter()]
model.fit(X, y, path, callbacks=callbacks)
```

**Rationale:**

- **Familiar:** Standard pattern (Keras, PyTorch Lightning use it)
- **Extensible:** Easy to add custom callbacks
- **Composable:** Multiple callbacks can run simultaneously
- **Separation of concerns:** Trainer doesn't know about logging/plotting details

**Trade-offs:**

✅ **Gained:**
- Clean separation (Trainer vs. observability)
- Easy to add custom callbacks
- Composable (multiple callbacks)
- Familiar pattern

⚠️ **Lost:**
- Callbacks can't modify training behavior (by design)
- Extra abstraction layer

**Outcome:**

✅ **Success.** Callbacks make it easy to add mixture tracking, custom logging, etc. without touching Trainer code.

**See:** [Architecture - Callback Pattern](architecture.md#5-callback-pattern)

---

### Component-Aware Decoder: Separate Pathways vs. Concatenation

**Context:**

Need to condition decoder on both latent $z$ and component $c$. Two main approaches.

**Options Considered:**

1. **Separate pathways:** Process $z$ and $e_c$ separately, then concatenate
2. **Simple concatenation:** Concatenate $[z; e_c]$ and feed through decoder
3. **Additive:** Add component bias to latent: $z + e_c$
4. **Multiplicative:** Scale latent by component: $z \odot e_c$

**Decision:**

Use **separate pathways**:

```python
z_features = Dense(hidden_dim // 2)(z)
c_features = Dense(hidden_dim // 2)(e_c)
combined = concatenate([z_features, c_features])
# ... rest of decoder
```

**Rationale:**

- **Symmetry:** Both $z$ and $c$ get equal processing capacity
- **Specialization:** $z$ pathway learns instance-level features, $c$ pathway learns component-level features
- **Flexibility:** Can adjust split ratio (e.g., 75% for $z$, 25% for $c$)
- **Better than simple concat:** Gives each input dedicated capacity before combining

**Trade-offs:**

✅ **Gained:**
- Better functional specialization
- More expressive than simple concatenation
- Can tune z vs. c contribution

⚠️ **Lost:**
- Slightly more parameters (~10% increase)
- Slightly slower than simple concatenation

**Outcome:**

✅ **Success.** Component-aware decoder shows clear embedding divergence and functional specialization. +1.7% reconstruction improvement vs. standard decoder.

**Reference:** See [STATUS.md - Component-Aware Decoder](STATUS.md#-component-aware-decoder)

---

## Future Decisions to Make

### τ-Classifier Update Strategy

**Question:** How should τ map be updated during training?

**Options:**
1. **After each epoch:** Update $s_{c,y}$ with all labeled data
2. **After each batch:** Incremental updates
3. **Periodic:** Every N epochs
4. **EMA:** Exponential moving average of counts

**Status:** 🎯 To be decided during implementation

---

### Top-M Gating: Soft vs. Hard Selection

**Question:** How to select top-M components for decoding?

**Options:**
1. **Hard selection:** Take top-M by responsibility, renormalize
2. **Soft gating:** Multiply reconstruction by responsibility, keep all K
3. **Adaptive M:** Choose M dynamically based on responsibility entropy

**Status:** 📋 To be decided when implementing Top-M gating

---

### VampPrior Pseudo-Input Initialization

**Question:** How to initialize pseudo-inputs for VampPrior?

**Options:**
1. **Random data points:** Sample K points from dataset
2. **K-means:** Use K-means cluster centers
3. **Learned from scratch:** Random initialization, let gradients optimize
4. **Hybrid:** K-means init + gradient refinement

**Status:** 📋 To be decided if/when implementing VampPrior

---

## Lessons Learned

### Start with Simplest Prior (Standard Gaussian)

**Lesson:** Implemented standard Gaussian prior first, validated training, then added mixture complexity.

**Why it helped:**
- Baseline to compare against
- Easier to debug issues
- Validated training infrastructure before adding complexity

**Recommendation:** Always implement simplest version first, validate, then add features.

---

### Diversity Reward Requires Careful Tuning

**Lesson:** Entropy reward weight needs tuning for each dataset/K value.

**What we learned:**
- Too strong (-0.5): Forces uniform usage, hurts reconstruction
- Too weak (-0.001): Doesn't prevent collapse
- Sweet spot (-0.05): Balances diversity and reconstruction

**Recommendation:** Start with -0.05, adjust based on K_eff metrics.

---

### Components Don't Need Spatial Separation

**Lesson:** Initial concern about latent overlap was unfounded.

**What we learned:**
- Spatial clustering is NOT required for functional specialization
- Component-aware decoder enables specialization via pathways, not geometry
- Visualization can be misleading (2D projection of high-D space)

**Recommendation:** Don't force spatial separation. Use component-aware decoder + τ-classifier instead.

---

## Related Documentation

- **[Status](STATUS.md)** - Current implementation status and feature details
- **[Architecture](architecture.md)** - Design patterns and philosophy
- **[Vision Gap](../theory/vision_gap.md)** - What's complete vs. planned
- **[Conceptual Model](../theory/conceptual_model.md)** - Theoretical foundation
- **[Mathematical Specification](../theory/mathematical_specification.md)** - Precise formulations
