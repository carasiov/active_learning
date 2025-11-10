# System Architecture

> **Purpose:** This document describes the design patterns, component structure, and architectural decisions in the current SSVAE codebase. For theoretical foundations, see [Conceptual Model](../theory/conceptual_model.md). For module-level implementation details, see [Implementation Guide](implementation.md).

---

## Overview

The system implements a semi-supervised variational autoencoder (SSVAE) with a modular, protocol-based architecture designed for extensibility and experimentation. The core philosophy is **configuration-driven components** with **pluggable abstractions** for priors, encoders, and decoders.

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

**Interface:**
```python
class PriorMode(Protocol):
    """Protocol defining interface for different prior modes."""

    def kl_divergence(
        self,
        z_mean: Array,
        z_logvar: Array,
        component_logits: Optional[Array] = None
    ) -> Array:
        """Compute KL divergence for this prior mode."""
        ...

    def sample(
        self,
        key: PRNGKey,
        latent_dim: int,
        num_samples: int = 1
    ) -> Array:
        """Sample from the prior distribution."""
        ...
```

**Implementations:**
- `StandardPrior`: Simple $\mathcal{N}(0, I)$ Gaussian
- `MixturePrior`: Mixture of Gaussians with learnable weights $\pi$

**Design Rationale:**
- Protocol (not abstract base class) for static type checking without runtime overhead
- New priors (VampPrior, flows) can be added by implementing this protocol
- No changes to `SSVAE` class required when adding new priors

### 2. SSVAEFactory

**Purpose:** Centralized component creation with validation and consistency checks.

**Location:** `src/ssvae/factory.py`

**Responsibilities:**
- Create encoders, decoders, classifiers based on configuration
- Validate hyperparameters (e.g., latent_dim matches architecture)
- Ensure compatible component combinations
- Initialize network weights

**Example:**
```python
network, variables = SSVAEFactory.create_network(
    config=config,
    input_shape=(28, 28),
    key=jax.random.PRNGKey(42)
)
```

**Design Rationale:**
- Single source of truth for component creation
- Early validation prevents runtime errors
- Easy to extend with new component types
- Testable in isolation

### 3. SSVAEConfig

**Purpose:** Centralized configuration with type safety and defaults.

**Location:** `src/ssvae/config.py`

**Key Parameters:**
```python
@dataclass
class SSVAEConfig:
    # Architecture
    latent_dim: int = 2
    hidden_dims: Tuple[int, ...] = (256, 128, 64)
    num_classes: int = 10

    # Prior
    prior_type: str = "standard"  # or "mixture"
    num_components: int = 10  # for mixture prior

    # Loss weights
    recon_weight: float = 1.0
    kl_weight: float = 1.0
    label_weight: float = 1.0
    kl_c_weight: float = 1.0  # mixture KL weight

    # Regularization
    component_diversity_weight: float = 0.0
    dirichlet_alpha: Optional[float] = None
    dirichlet_weight: float = 1.0

    # Training
    batch_size: int = 128
    learning_rate: float = 1e-3
    max_epochs: int = 100
    patience: int = 10
```

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

**Interface:**
```python
class Encoder(nn.Module):
    latent_dim: int
    hidden_dims: Tuple[int, ...]

    def __call__(self, x, deterministic=True):
        # Returns: z_mean, z_logvar, [component_logits]
        ...
```

**Design:**
- Encoders produce distribution parameters, not samples
- Separate mean and log-variance outputs
- Mixture encoder adds component logits output
- Dropout support via `deterministic` flag

### Decoders

**Location:** `src/ssvae/components/decoders/`

**Types:**
- `DenseDecoder`: Fully connected layers
- `ConvDecoder`: Transposed convolutional layers

**Interface:**
```python
class Decoder(nn.Module):
    output_shape: Tuple[int, ...]
    hidden_dims: Tuple[int, ...]

    def __call__(self, z, deterministic=True):
        # Returns: reconstructed x
        ...
```

**Design:**
- Mirrors encoder architecture (symmetric VAE)
- Output shape matches input data shape
- Sigmoid activation for [0,1] pixel values

### Classifiers

**Location:** `src/ssvae/components/classifiers/`

**Current Implementation:**
- `DenseClassifier`: Simple MLP with softmax output

**Interface:**
```python
class Classifier(nn.Module):
    num_classes: int
    hidden_dim: int

    def __call__(self, z):
        # Returns: class logits
        ...
```

**Note:** This will be replaced by latent-only $\tau$-based classification in the RCM-VAE architecture.

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
- Optional `TrainerLoopHooks` structure extends the loop with three touch points:
  - `batch_context_fn` supplies extra kwargs (e.g., τ matrix) to each `train_step`.
  - `post_batch_fn` runs outside JIT to mutate Python-side state (τ count updates).
  - `eval_context_fn` mirrors those kwargs during validation.
  This keeps τ-classifier (and future stateful components) integrated without duplicating the core training logic.

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

## Extension Points

### Adding a New Prior

1. **Implement PriorMode protocol:**
```python
class VampPrior:
    def kl_divergence(self, z_mean, z_logvar, component_logits=None):
        # Your VampPrior KL computation
        ...

    def sample(self, key, latent_dim, num_samples=1):
        # Sample from learned pseudo-inputs
        ...
```

2. **Register in factory:**
```python
# In factory.py
if config.prior_type == "vamp":
    prior = VampPrior(...)
```

3. **No changes to SSVAE class required!**

### Adding a New Encoder

1. **Create encoder module:**
```python
class TransformerEncoder(nn.Module):
    def __call__(self, x, deterministic=True):
        # Your architecture
        return z_mean, z_logvar
```

2. **Register in factory:**
```python
if config.encoder_type == "transformer":
    encoder = TransformerEncoder(...)
```

### Adding Component-Aware Features

1. **Extend decoder to accept channel:**
```python
class ComponentAwareDecoder(nn.Module):
    def __call__(self, z, channel_embedding, deterministic=True):
        combined = jnp.concatenate([z, channel_embedding], axis=-1)
        # Decode combined representation
        ...
```

2. **Update factory and SSVAE to pass channel info**

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
- **[Implementation Guide](implementation.md)** - Module-by-module reference
- **[Extending the System](extending.md)** - Step-by-step tutorials for adding features
- **[Implementation Roadmap](../theory/implementation_roadmap.md)** - Path to full RCM-VAE system
