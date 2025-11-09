---
**Status:** Current implementation as of November 2025
**Reflects:** Component-aware decoder complete, τ-classifier next
**For planned features:** See [Implementation Roadmap](../theory/implementation_roadmap.md)
**For contributing:** See [CONTRIBUTING.md](CONTRIBUTING.md)
---

# System Architecture

> **Purpose:** Understand why the SSVAE system is designed this way
> **Audience:** Researchers, code reviewers, developers seeking design rationale
> **For implementation details:** See code docstrings and [CONTRIBUTING.md](CONTRIBUTING.md)

---

## Overview

The system implements a semi-supervised variational autoencoder (SSVAE) with a modular,
protocol-based architecture designed for extensibility and experimentation. The core
philosophy is **configuration-driven components** with **pluggable abstractions** for
priors, encoders, and decoders.

**Current state (November 2025):**
- ✅ Mixture prior with K components
- ✅ Component-aware decoder (separate z and e_c pathways)
- ✅ Usage-entropy diversity regularization
- 🚧 τ-classifier implementation (next priority)

For current implementation status, see [Implementation Roadmap](../theory/implementation_roadmap.md).

### Design Principles

1. **Protocol-based abstractions:** Components implement protocols (interfaces) rather than concrete inheritance
2. **Factory pattern:** Centralized component creation with validation
3. **Configuration-driven:** All hyperparameters exposed through dataclasses
4. **Separation of concerns:** Clear boundaries between model, training, and observability
5. **Immutability:** JAX functional programming patterns (pure functions, explicit state)

---

## System Components

### High-Level Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    SSVAE Model (Public API)                  │
│                     src/ssvae/models.py                      │
│                                                               │
│  • fit(data, labels) → train model                           │
│  • predict(data) → latent, recon, predictions, certainty     │
│  • load_model_weights(path) → restore checkpoint            │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Factory    │ │  Checkpoint  │ │ Diagnostics  │
│              │ │   Manager    │ │  Collector   │
│ Component    │ │              │ │              │
│  creation    │ │ Save/restore │ │ Metrics      │
│  validation  │ │    state     │ │ collection   │
└──────┬───────┘ └──────────────┘ └──────────────┘
       │
       │ Creates
       ▼
┌─────────────────────────────────────────────┐
│         Network Components                   │
├─────────────────────────────────────────────┤
│  • Encoders (dense, conv)                   │
│  • Decoders (dense, conv)                   │
│  • Classifiers (softmax)                    │
│  • Priors (standard, mixture - PriorMode)   │
└─────────────────────────────────────────────┘
       │
       │ Used by
       ▼
┌─────────────────────────────────────────────┐
│         Training Infrastructure              │
├─────────────────────────────────────────────┤
│  • Trainer: training loop, early stopping   │
│  • Losses: reconstruction, KL, classification│
│  • Callbacks: logging, plotting             │
└─────────────────────────────────────────────┘
```

---

## Core Abstractions

### 1. PriorMode Protocol

**Purpose:** Enable pluggable prior distributions without modifying core model code.

**Location:** `src/ssvae/components/priors/protocol.py`

**Interface contract:**
See `src/ssvae/priors/base.py` for protocol definition.

**Implementations:**
- `StandardPrior` in `priors/standard.py` - Simple N(0,I)
- `MixturePrior` in `priors/mixture.py` - Mixture with embeddings

**Design Rationale:**
- Protocol (not abstract base class) for static type checking without runtime overhead
- New priors (VampPrior, flows) can be added by implementing this protocol
- No changes to `SSVAE` class required when adding new priors

### 2. SSVAEFactory

**Purpose:** Centralized component creation with validation and consistency checks.

**Location:** `src/ssvae/factory.py`

**Key responsibilities:**
- Create encoders, decoders, classifiers
- Validate configuration consistency
- Initialize network weights

See `src/ssvae/factory.py` for implementation details.

**Design Rationale:**
- Single source of truth for component creation
- Early validation prevents runtime errors
- Easy to extend with new component types
- Testable in isolation

### 3. SSVAEConfig

**Purpose:** Centralized configuration with type safety and defaults.

**Location:** `src/ssvae/config.py`

**Purpose:** Type-safe configuration with defaults

See `src/ssvae/config.py` for all 25+ parameters with descriptions.

**Design Rationale:**
- Dataclass provides free validation, serialization, equality
- Explicit types catch errors at config time
- Defaults enable quick experimentation
- Easy to extend without breaking existing code

### 4. CheckpointManager

**Purpose:** Handle model state persistence and restoration.

**Location:** `src/ssvae/checkpoint_manager.py`

**Responsibilities:**
- Save model parameters to disk
- Load parameters and restore model state
- Handle version compatibility
- Manage checkpoint metadata

**Design Rationale:**
- JAX requires explicit state management (no implicit parameter storage)
- Checkpoints include full state (params, optimizer, config)
- Enables model deployment and experiment reproduction

### 5. DiagnosticsCollector

**Purpose:** Collect and organize training metrics.

**Location:** `src/ssvae/diagnostics.py`

**Responsibilities:**
- Compute mixture prior diagnostics (usage, entropy, π values)
- Organize metrics for logging and visualization
- Handle metric aggregation across batches

**Design Rationale:**
- Separates metric computation from training loop
- Easy to add new metrics without touching core code
- Provides consistent metric reporting across tools

---

## Component Structure

### Encoders

**Location:** `src/ssvae/components/encoders/`

**Types:**
- `DenseEncoder`: Fully connected layers
- `ConvEncoder`: Convolutional layers (for image data)

**Key properties:**
- Produce distribution parameters (mean, log-variance)
- Mixture encoder adds component logits
- Dropout support for regularization

See `src/ssvae/components/encoders.py` for implementation.

### Decoders

**Location:** `src/ssvae/components/decoders/`

**Types:**
- `DenseDecoder`: Fully connected layers
- `ConvDecoder`: Transposed convolutional layers

**Key properties:**
- Mirrors encoder architecture (symmetric VAE)
- Output shape matches input data
- Sigmoid activation for pixel values

See `src/ssvae/components/decoders.py` for implementation.

### Classifiers

**Location:** `src/ssvae/components/classifiers/`

**Current Implementation:**
- `DenseClassifier`: Simple MLP with softmax output

**Note:** This will be replaced by latent-only τ-based classification in the RCM-VAE architecture.

See `src/ssvae/components/classifier.py` for implementation.

### Priors

**Location:** `src/ssvae/components/priors/`

**Implementations:**

**StandardPrior** (`standard.py`):
- Simple $\mathcal{N}(0, I)$ prior
- KL divergence: analytical formula
- No learnable parameters

**MixturePrior** (`mixture.py`):
- Mixture of Gaussians: $p(z) = \sum_k \pi_k \mathcal{N}(0, I)$
- Learnable mixture weights $\pi$ (via logits)
- Optional Dirichlet prior on $\pi$
- Usage sparsity regularization
- Component-wise KL: $\text{KL}(q(c|x) || \pi)$

**Design:**
- Both implement `PriorMode` protocol
- Stateless (pure functions)
- Easy to add VampPrior, flows, etc.

---

## Data Flow

### Training Forward Pass

```
Input x (image)
    │
    ▼
┌─────────────┐
│   Encoder   │ → z_mean, z_logvar, [component_logits]
└─────────────┘
    │
    ▼
┌─────────────┐
│ Reparameterization │ → z = μ + σ * ε
└─────────────┘
    │
    ├─────────────┐
    ▼             ▼
┌─────────┐  ┌──────────┐
│ Decoder │  │Classifier│
└─────────┘  └──────────┘
    │             │
    ▼             ▼
   x̂          class logits
    │             │
    └──────┬──────┘
           ▼
    ┌──────────────┐
    │ Loss Compute │
    └──────────────┘
           │
           ▼
    Recon + KL + Classification
```

### Inference

```
Input x
    │
    ▼
Encoder → z_mean (no sampling for predictions)
    │
    ├─────────────┐
    ▼             ▼
Decoder      Classifier
    │             │
    ▼             ▼
Reconstruction  Predictions + Certainty
```

---

## Training Infrastructure

### Trainer

**Location:** `src/training/trainer.py`

**Responsibilities:**
- Batch iteration and gradient updates
- Early stopping based on validation loss
- Training history tracking
- Checkpoint saving

**Key Design:**
- Pure functional training loop (JAX style)
- Explicit state passing (no hidden state)
- Callback hooks for observability

### Losses

**Location:** `src/training/losses.py`

**Components:**
- **Reconstruction loss:** BCE or MSE
- **KL divergence:** Prior-specific computation
- **Classification loss:** Cross-entropy on labeled data
- **Regularization:** Usage sparsity, Dirichlet prior

**Design:**
- Composable loss functions
- Weight-based loss scaling
- Separate labeled/unlabeled handling

### Callbacks

**Location:** `src/callbacks/`

**Types:**
- `LoggingCallback`: Console and CSV logging
- `PlottingCallback`: Loss curve visualization

**Design:**
- Observer pattern for training events
- Stateful (can accumulate history)
- Easy to add custom callbacks

---

## Extension Philosophy

**When to extend vs. configure:**

| Goal | Approach | Example |
|------|----------|---------|
| Change hyperparameter | Configure | Adjust `kl_weight` in config |
| Add prior distribution | Extend | Implement `PriorMode` protocol |
| Add regularizer | Extend | Add loss term to `losses.py` |
| Add architecture | Extend | Register in factory |
| Add observability | Extend | Create callback class |

**Extension patterns:**
1. **Protocol implementation** - Priors (see [CONTRIBUTING.md](CONTRIBUTING.md#adding-a-prior))
2. **Factory registration** - Components (see [CONTRIBUTING.md](CONTRIBUTING.md#adding-components))
3. **Callback hooks** - Observability (see [CONTRIBUTING.md](CONTRIBUTING.md#adding-a-callback))
4. **Configuration parameters** - Hyperparameters

For step-by-step guides, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Testing Strategy

### Test Organization

- **Unit tests** (`tests/test_network_components.py`): Individual components in isolation
- **Integration tests** (`tests/test_integration_workflows.py`): End-to-end workflows
- **Regression tests** (`tests/test_mixture_prior_regression.py`): Prior behavior validation

### Key Tests

**Component validation:**
- Output shapes correct
- Deterministic vs stochastic modes
- Gradient flow

**Integration validation:**
- Full training loop works
- Checkpoint save/load
- Configuration changes

**Regression validation:**
- Mixture prior metrics stable
- Loss values in expected ranges

---

## Design Patterns Summary

| Pattern | Purpose | Location | Benefit |
|---------|---------|----------|---------|
| **Protocol** | Pluggable priors | `priors/protocol.py` | Add priors without modifying core code |
| **Factory** | Component creation | `factory.py` | Centralized validation and consistency |
| **Dataclass Config** | Type-safe configuration | `config.py` | Early error detection, serialization |
| **Dependency Injection** | Pass components to SSVAE | `models.py` | Testability, flexibility |
| **Callback Pattern** | Training observability | `callbacks/` | Extensible logging/plotting |
| **Immutable State** | JAX functional style | Throughout | Reproducibility, parallelization |

---

## Related Documentation

- **[Conceptual Model](../theory/conceptual_model.md)** - Theoretical foundation and mental model
- **[Mathematical Specification](../theory/mathematical_specification.md)** - Precise objectives and formulations
- **[Implementation Roadmap](../theory/implementation_roadmap.md)** - Current status and next steps
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to extend and modify the system (step-by-step guides)
