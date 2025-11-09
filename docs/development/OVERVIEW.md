# Core Model Overview

> **Purpose:** Quick introduction to the `/src/` codebase for developers extending or understanding the core SSVAE model.

---

## What Is This?

The `/src/` directory contains a **reusable JAX/Flax implementation** of a semi-supervised variational autoencoder (SSVAE) with mixture prior support. It's designed as a library that can be used by:

- **Batch experimentation** (`/use_cases/experiments/`) - Train and evaluate models
- **Interactive dashboard** (`/use_cases/dashboard/`) - Active learning interface
- **Your own applications** - Import and use the model

This document provides a quick orientation to the codebase architecture.

---

## Architecture at a Glance

```
src/
├── ssvae/                   # 🧠 Core Model (JAX/Flax)
│   ├── models.py            #    SSVAE class (public API)
│   ├── config.py            #    SSVAEConfig (hyperparameters)
│   ├── factory.py           #    Component creation
│   ├── network.py           #    Neural network architecture
│   ├── checkpoint.py        #    Save/load model state
│   ├── diagnostics.py       #    Metrics collection
│   ├── components/          #    Neural network components
│   │   ├── encoders/        #       Dense, Conv encoders
│   │   ├── decoders/        #       Dense, Conv, ComponentAware decoders
│   │   └── classifiers/     #       Classification heads
│   └── priors/              #    Prior distributions
│       ├── standard.py      #       Standard Gaussian N(0,I)
│       └── mixture.py       #       Mixture of Gaussians
│
├── training/                # 🔄 Training Infrastructure
│   ├── trainer.py           #    Main training loop
│   ├── interactive_trainer.py #  Stateful trainer for active learning
│   ├── losses.py            #    Loss functions
│   └── train_state.py       #    Training state management
│
├── callbacks/               # 📊 Training Observability
│   ├── logging.py           #    Console & CSV logging
│   ├── plotting.py          #    Loss visualization
│   └── mixture_tracker.py   #    Mixture dynamics tracking
│
└── utils/                   # 🔧 Utilities
    └── device.py            #    JAX device configuration
```

---

## Key Modules

### `ssvae/` - Core Model

**Entry point:** `models.py` provides the public `SSVAE` class

**Key responsibilities:**
- Model initialization and configuration
- Training via `.fit(data, labels, path)`
- Inference via `.predict(data)`
- Checkpoint save/load

**Configuration:** All hyperparameters live in `config.py` as `SSVAEConfig` dataclass

**Components:** Encoders, decoders, classifiers, and priors are created by the factory pattern

### `training/` - Training Infrastructure

**Entry point:** `trainer.py` provides the `Trainer` class

**Key responsibilities:**
- Training loop with early stopping
- Data splitting (train/validation)
- Metric computation and history tracking
- Checkpoint management during training

**Interactive variant:** `interactive_trainer.py` for active learning workflows (preserves optimizer state across sessions)

### `callbacks/` - Observability

**Purpose:** Hook into training events for logging and visualization

**Available callbacks:**
- `ConsoleLogger` - Print metrics to console
- `CSVExporter` - Export metrics to CSV
- `LossCurvePlotter` - Generate loss plots
- `MixtureHistoryTracker` - Track mixture prior evolution

---

## Design Philosophy

The codebase follows these principles:

### 1. **Protocol-Based Abstractions**
Components implement protocols (interfaces) rather than concrete inheritance. This enables pluggable behavior.

**Example:** `PriorMode` protocol allows adding new priors (VampPrior, flows) without modifying core code.

### 2. **Factory Pattern**
Centralized component creation with validation ensures consistency.

**Example:** `SSVAEFactory.create_network()` validates configuration and creates compatible encoder/decoder/classifier.

### 3. **Configuration-Driven**
All hyperparameters are explicit in `SSVAEConfig` dataclass - no hidden defaults.

**Example:**
```python
config = SSVAEConfig(
    latent_dim=2,
    prior_type="mixture",
    num_components=10,
    kl_weight=0.5
)
model = SSVAE(input_dim=(28, 28), config=config)
```

### 4. **Separation of Concerns**
Clear boundaries between model (what), training (how), and observability (tracking).

**Example:** Model doesn't know about checkpointing or logging - that's handled by `CheckpointManager` and callbacks.

### 5. **Immutability (JAX Style)**
Pure functions with explicit state passing - no hidden global state.

**Example:** Training state is passed explicitly, not stored in class attributes.

---

## Common Workflows

### Creating and Training a Model

```python
from ssvae import SSVAE, SSVAEConfig

# Configure
config = SSVAEConfig(
    latent_dim=2,
    prior_type="mixture",
    num_components=10
)

# Create model
model = SSVAE(input_dim=(28, 28), config=config)

# Train
history = model.fit(X_train, y_train, weights_path="model.ckpt")

# Predict
z, recon, preds, certainty = model.predict(X_test)
```

### Using Custom Callbacks

```python
from callbacks import ConsoleLogger, LossCurvePlotter

callbacks = [
    ConsoleLogger(log_every=10),
    LossCurvePlotter(save_path="loss_curves.png")
]

history = model.fit(X, y, "model.ckpt", callbacks=callbacks)
```

---

## Next Steps

**Understand the design patterns:**
→ [Architecture Guide](architecture.md) - Why we chose protocols, factory, config-driven design

**Find specific modules and classes:**
→ [API Reference](api_reference.md) - Concise module-by-module guide

**Check current implementation status:**
→ [Status](STATUS.md) - What's complete, in progress, and planned

**Understand design decisions:**
→ [Decisions](DECISIONS.md) - Why we made specific architectural choices

**Add new features:**
→ [Extending Guide](extending.md) - Step-by-step tutorials for common extension tasks

**Understand the theory:**
→ [Conceptual Model](../theory/conceptual_model.md) - Mental model and invariants
→ [Math Specification](../theory/mathematical_specification.md) - Precise formulations

---

## Testing

The codebase includes comprehensive tests:

```bash
# Run all tests
pytest tests/

# Specific test categories
pytest tests/test_network_components.py      # Unit tests
pytest tests/test_integration_workflows.py   # End-to-end tests
pytest tests/test_mixture_prior_regression.py # Prior behavior tests
```

---

## Getting Help

- **Code questions:** See [API Reference](api_reference.md)
- **Design questions:** See [Architecture](architecture.md) and [Decisions](DECISIONS.md)
- **Theory questions:** See [Conceptual Model](../theory/conceptual_model.md)
- **Usage questions:** See use case guides ([Experiments](../../use_cases/experiments/README.md) | [Dashboard](../../use_cases/dashboard/README.md))
