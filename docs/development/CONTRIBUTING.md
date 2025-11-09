# Contributing to SSVAE

> **For architecture concepts:** See [ARCHITECTURE.md](ARCHITECTURE.md)
> **For theory:** See [../theory/conceptual_model.md](../theory/conceptual_model.md)
> **Last updated:** 2025-11-09
> **Status:** Component-aware decoder complete, τ-classifier next

---

## Table of Contents
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Codebase Navigation](#codebase-navigation)
- [Common Tasks](#common-tasks)
  - [Adding a Prior](#adding-a-prior)
  - [Adding a Loss Term](#adding-a-loss-term)
  - [Adding Components](#adding-components)
  - [Adding a Callback](#adding-a-callback)
- [Testing](#testing)
- [Code Style](#code-style)
- [API Quick Reference](#api-quick-reference)
- [Getting Help](#getting-help)

---

## Setup

```bash
# Clone repository
git clone https://github.com/your-org/active-learning-showcase.git
cd active-learning-showcase

# Install in development mode
pip install -e ".[dev]"

# Verify installation
pytest tests/
python experiments/run_experiment.py --config configs/baseline.yaml
```

**Requirements:**
- Python 3.10+
- JAX (CPU or GPU)
- See `pyproject.toml` for full dependencies

---

## Quick Start

### 1. Run an Experiment (5 minutes)

The fastest way to understand the system is to run it:

```bash
# Run baseline experiment
python experiments/run_experiment.py --config configs/baseline.yaml

# Outputs saved to: experiments/runs/<timestamp>/
```

**What you'll see:**
- Training progress with loss curves
- Latent space visualizations
- Component diagnostics (for mixture prior)
- Saved checkpoints

### 2. Modify a Configuration (10 minutes)

Edit `experiments/configs/baseline.yaml`:

```yaml
model:
  latent_dim: 10  # Change from 2 to 10
  num_components: 20  # Change from 10 to 20
```

Run again and observe differences in results.

### 3. Make a Code Change (15 minutes)

Try adding a simple penalty:

```python
# Edit src/training/losses.py
def latent_l2_penalty(z: jnp.ndarray, weight: float) -> jnp.ndarray:
    """Penalize large latent values."""
    return weight * jnp.mean(jnp.sum(z**2, axis=-1))

# Add to compute_loss_and_metrics_v2() around line 250
l2_penalty = latent_l2_penalty(z, config.get("latent_l2_weight", 0.0))
total = recon_loss + total_kl + cls_loss_weighted + l2_penalty
metrics["latent_l2"] = l2_penalty
```

Add config parameter:
```python
# Edit src/ssvae/config.py
@dataclass
class SSVAEConfig:
    # ... existing fields ...
    latent_l2_weight: float = 0.01  # Add this
```

Test it:
```bash
pytest tests/  # Should still pass
python experiments/run_experiment.py --config configs/baseline.yaml
```

---

## Codebase Navigation

### Directory Structure

```
src/
├── ssvae/              # Core model
│   ├── models.py       # 👈 START HERE - Public API
│   ├── network.py      # Neural network architecture
│   ├── config.py       # All hyperparameters
│   ├── factory.py      # Component creation & validation
│   ├── checkpoint.py   # Save/load model state
│   ├── diagnostics.py  # Mixture prior diagnostics
│   ├── components/     # Neural network components
│   │   ├── encoders.py
│   │   ├── decoders.py
│   │   ├── classifier.py
│   │   └── factory.py
│   └── priors/         # Prior distributions
│       ├── base.py     # Protocol definition
│       ├── standard.py # Simple N(0,I) prior
│       └── mixture.py  # Mixture prior
│
├── training/           # Training infrastructure
│   ├── trainer.py      # Main training loop
│   ├── losses.py       # 👈 Loss functions here
│   ├── train_state.py  # JAX training state
│   └── interactive_trainer.py  # Incremental training
│
├── callbacks/          # Training observability
│   ├── base_callback.py
│   ├── logging.py
│   ├── plotting.py
│   └── mixture_tracking.py
│
└── utils/
    └── device.py       # JAX device configuration

experiments/
├── run_experiment.py   # Primary experiment runner
├── configs/            # YAML configurations
└── runs/               # Experiment outputs (timestamped)

tests/
├── test_network_components.py
├── test_integration_workflows.py
└── test_mixture_prior_regression.py
```

### Key Files by Task

| I want to... | Look at... |
|--------------|-----------|
| Understand the API | `src/ssvae/models.py` |
| Add a loss term | `src/training/losses.py` |
| Add a prior | `src/ssvae/priors/` + `factory.py` |
| Add a component | `src/ssvae/components/` + `factory.py` |
| Modify training | `src/training/trainer.py` |
| Add a callback | `src/callbacks/` |
| Configure hyperparameters | `src/ssvae/config.py` |
| Run experiments | `experiments/run_experiment.py` |

---

## Common Tasks

### Adding a Prior

**Goal:** Implement a new prior distribution (e.g., VampPrior, flow-based prior)

**Steps:**

1. **Create prior class** implementing `PriorMode` protocol:

```python
# src/ssvae/priors/my_prior.py
from ssvae.priors.base import PriorMode, EncoderOutput

class MyPrior:
    """My custom prior distribution."""

    def compute_kl_terms(self, encoder_output: EncoderOutput, config) -> dict:
        """Compute KL divergence terms."""
        # Your KL computation here
        return {"kl_z": kl_value}

    def compute_reconstruction_loss(self, x_true, x_recon, encoder_output, config):
        """Compute reconstruction loss."""
        # Your reconstruction logic here
        return loss_value

    def get_prior_type(self) -> str:
        return "my_prior"

    def requires_component_embeddings(self) -> bool:
        return False  # True if decoder needs component embeddings
```

2. **Register** in `src/ssvae/priors/__init__.py`:

```python
from .my_prior import MyPrior

PRIOR_REGISTRY = {
    "standard": StandardGaussianPrior,
    "mixture": MixtureGaussianPrior,
    "my_prior": MyPrior,  # Add this
}
```

3. **Add config parameters** (if needed) in `src/ssvae/config.py`:

```python
@dataclass
class SSVAEConfig:
    # ... existing fields ...
    my_prior_param: float = 1.0  # Add custom parameters
```

4. **Write tests** in `tests/test_my_prior.py`:

```python
def test_my_prior_kl():
    """Test KL computation."""
    prior = MyPrior()
    encoder_output = EncoderOutput(...)
    kl_terms = prior.compute_kl_terms(encoder_output, config)
    assert "kl_z" in kl_terms
    assert kl_terms["kl_z"] >= 0
```

**Reference:**
- Protocol definition: `src/ssvae/priors/base.py`
- Working example: `src/ssvae/priors/mixture.py`
- Test pattern: `tests/test_mixture_prior_regression.py`

---

### Adding a Loss Term

**Goal:** Add a custom loss or regularization term

**Steps:**

1. **Define function** in `src/training/losses.py`:

```python
def my_custom_loss(z: jnp.ndarray, weight: float) -> jnp.ndarray:
    """Description of what this loss does."""
    # Your loss computation
    return weight * loss_value
```

2. **Add to total loss** in `compute_loss_and_metrics_v2()`:

```python
# Around line 250 in losses.py
my_loss = my_custom_loss(z, config.my_loss_weight)
total = recon_loss + total_kl + cls_loss_weighted + my_loss
metrics["my_loss"] = my_loss
```

3. **Add weight parameter** to `SSVAEConfig`:

```python
@dataclass
class SSVAEConfig:
    # ... existing fields ...
    my_loss_weight: float = 0.1
```

4. **Test:**

```python
def test_my_loss():
    """Test loss is non-negative and differentiable."""
    z = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    loss = my_custom_loss(z, weight=1.0)
    assert loss >= 0
    assert jnp.isfinite(loss)
```

**Reference:**
- Existing losses: `src/training/losses.py`
- Example: `usage_sparsity_penalty()` around line 100

---

### Adding Components

**Goal:** Add a new encoder, decoder, or classifier architecture

**Steps:**

1. **Create component class** in `src/ssvae/components/encoders.py` (or decoders/classifiers):

```python
class MyEncoder(nn.Module):
    """My custom encoder architecture."""
    hidden_dims: Tuple[int, ...]
    latent_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, training: bool):
        # Your architecture here
        z_mean = nn.Dense(self.latent_dim)(x)
        z_log = nn.Dense(self.latent_dim)(x)

        # Reparameterization
        if self.has_rng("reparam"):
            eps = random.normal(self.make_rng("reparam"), z_mean.shape)
            z = z_mean + jnp.exp(0.5 * z_log) * eps
        else:
            z = z_mean

        return z_mean, z_log, z
```

2. **Register in factory** in `src/ssvae/components/factory.py`:

```python
def build_encoder(config: SSVAEConfig, *, input_hw: Tuple[int, int] | None = None):
    # ... existing code ...

    if config.encoder_type == "my_encoder":
        return MyEncoder(
            hidden_dims=hidden_dims,
            latent_dim=config.latent_dim
        )
```

3. **Add config option:**

```python
@dataclass
class SSVAEConfig:
    encoder_type: str = "dense"  # Now accepts "my_encoder" too
```

4. **Test shape outputs:**

```python
def test_my_encoder():
    """Test encoder produces correct output shapes."""
    encoder = MyEncoder(hidden_dims=(128, 64), latent_dim=10)
    x = jnp.zeros((32, 28, 28))
    z_mean, z_log, z = encoder(x, training=True)
    assert z_mean.shape == (32, 10)
    assert z_log.shape == (32, 10)
    assert z.shape == (32, 10)
```

**Reference:**
- Encoder examples: `src/ssvae/components/encoders.py`
- Factory pattern: `src/ssvae/components/factory.py`

---

### Adding a Callback

**Goal:** Add custom logging, plotting, or diagnostics during training

**Steps:**

1. **Create callback class** inheriting from `TrainingCallback`:

```python
# src/callbacks/my_callback.py
from callbacks.base_callback import TrainingCallback

class MyCallback(TrainingCallback):
    """My custom callback."""

    def on_train_start(self, trainer):
        """Called once before training."""
        print("Training started!")

    def on_epoch_end(self, epoch, metrics, history, trainer):
        """Called after each epoch."""
        print(f"Epoch {epoch}: loss={metrics['train']['loss']:.4f}")

    def on_train_end(self, history, trainer):
        """Called once after training."""
        print("Training complete!")
```

2. **Use in training:**

```python
from callbacks.my_callback import MyCallback

model = SSVAE(input_dim=(28, 28), config=config)
history = model.fit(
    X, y, "model.ckpt",
    callbacks=[MyCallback()]
)
```

**Reference:**
- Base class: `src/callbacks/base_callback.py`
- Examples: `src/callbacks/logging.py`, `src/callbacks/plotting.py`

---

## Testing

### Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_network_components.py

# Specific test
pytest tests/test_network_components.py::test_dense_encoder

# With coverage
pytest --cov=src tests/

# Verbose output
pytest -v tests/
```

### Test Organization

- `test_network_components.py` - Unit tests for encoders, decoders, classifiers
- `test_integration_workflows.py` - End-to-end training workflows
- `test_mixture_prior_regression.py` - Mixture prior behavior validation

### Writing Tests

```python
import jax.numpy as jnp
from ssvae import SSVAE, SSVAEConfig

def test_my_feature():
    """Test that my feature works correctly.

    Test pattern:
    1. Setup: Create config and model
    2. Action: Call the method being tested
    3. Assert: Check outputs are correct
    """
    # Setup
    config = SSVAEConfig(my_param=True)
    model = SSVAE(input_dim=(28, 28), config=config)

    # Action
    result = model.my_method(...)

    # Assert
    assert result.shape == expected_shape
    assert not jnp.isnan(result).any()
    assert result >= 0  # If result should be non-negative
```

### Validation Checklist

Before submitting a pull request:

- [ ] All tests pass: `pytest tests/`
- [ ] Added test for new feature
- [ ] Code follows style guide (see below)
- [ ] Updated docstrings
- [ ] No `print()` statements (use `logging` instead)
- [ ] Type hints added to new functions
- [ ] No warnings from pytest

---

## Code Style

### General Principles

- Follow PEP 8
- Use type hints everywhere
- Write descriptive docstrings (Google style)
- Keep functions small and focused
- Avoid global state

### Imports

```python
from __future__ import annotations  # Always at top

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ssvae import SSVAE, SSVAEConfig
from training.losses import kl_divergence
```

**Order:** Future imports → Standard library → Third-party → Local imports

### Type Hints

```python
def compute_loss(
    x: jnp.ndarray,
    y: jnp.ndarray,
    weight: float = 1.0
) -> jnp.ndarray:
    """Always include type hints for parameters and return values."""
    ...
```

### Docstrings

Use Google style:

```python
def my_function(x: jnp.ndarray, weight: float) -> jnp.ndarray:
    """One-line summary of what this function does.

    Optional longer description explaining the function in more detail.
    Include mathematical formulas if relevant.

    Args:
        x: Description of x with shape [batch, dim]
        weight: Scaling factor for the output

    Returns:
        Weighted output with same shape as x

    Raises:
        ValueError: If weight is negative

    Example:
        >>> x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> result = my_function(x, weight=2.0)
        >>> result.shape
        (2, 2)
    """
```

### Naming Conventions

- **Classes:** `PascalCase` (e.g., `SSVAEConfig`, `MixturePrior`)
- **Functions/variables:** `snake_case` (e.g., `compute_loss`, `z_mean`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `MAX_EPOCHS`, `DEFAULT_SEED`)
- **Private:** `_leading_underscore` (e.g., `_internal_helper`)

### JAX Conventions

```python
# ✅ Good: Pure functions
def compute_kl(z_mean: jnp.ndarray, z_log: jnp.ndarray) -> jnp.ndarray:
    return -0.5 * jnp.sum(1 + z_log - z_mean**2 - jnp.exp(z_log))

# ❌ Bad: In-place operations
def bad_function(x):
    x[0] = 5  # JAX arrays are immutable!
    return x

# ✅ Good: Return new arrays
def good_function(x):
    return x.at[0].set(5)

# ✅ Good: Use JIT for performance-critical code
@jax.jit
def train_step(state, batch):
    ...
```

### Code Organization

```python
# Module structure
"""Module docstring explaining purpose."""

from __future__ import annotations

# Imports
...

# Constants
MAX_ITERATIONS = 1000

# Private helpers
def _internal_helper(...):
    """Private functions start with underscore."""
    ...

# Public API
class MyClass:
    """Public classes and functions."""
    ...

def public_function(...):
    """Public functions."""
    ...
```

---

## API Quick Reference

### Creating a Model

```python
from ssvae import SSVAE, SSVAEConfig

# Standard VAE
config = SSVAEConfig(latent_dim=2, prior_type="standard")
model = SSVAE(input_dim=(28, 28), config=config)

# Mixture VAE (recommended)
config = SSVAEConfig(
    latent_dim=10,
    prior_type="mixture",
    num_components=50,
    hidden_dims=(512, 256, 128)
)
model = SSVAE(input_dim=(28, 28), config=config)
```

### Training

```python
import numpy as np

# Semi-supervised labels (NaN = unlabeled)
labels = np.array([0, 1, np.nan, np.nan, 2, ...])

# Train
history = model.fit(X_train, labels, "model.ckpt")

# Access training history
print(history["loss"])
print(history["val_loss"])
```

### Prediction

```python
# Deterministic prediction (use mean)
z, recon, predictions, certainty = model.predict(X_test)

# Stochastic prediction (sample from distribution)
z, recon, predictions, certainty = model.predict(
    X_test,
    sample=True,
    num_samples=10
)

# Mixture-specific outputs
if config.prior_type == "mixture":
    z, recon, preds, cert, q_c, pi = model.predict(
        X_test,
        return_mixture=True
    )
```

### Configuration

```python
# See all parameters
config = SSVAEConfig(
    # Architecture
    latent_dim=16,
    hidden_dims=(512, 256, 128),
    num_classes=10,
    encoder_type="dense",  # or "conv"
    decoder_type="dense",  # or "conv"

    # Prior
    prior_type="mixture",  # or "standard"
    num_components=50,
    use_component_aware_decoder=True,
    component_embedding_dim=8,

    # Loss weights
    recon_weight=500.0,
    kl_weight=5.0,
    kl_c_weight=0.001,
    label_weight=0.0,

    # Regularization
    component_diversity_weight=-0.05,  # Negative = encourage diversity
    dirichlet_alpha=5.0,

    # Training
    batch_size=128,
    learning_rate=1e-3,
    max_epochs=100,
    patience=10
)
```

**For all parameters:** See `src/ssvae/config.py`

### Loading Checkpoints

```python
# Load trained model
model = SSVAE(input_dim=(28, 28), config=config)
model.load_model_weights("model.ckpt")

# Continue training
history = model.fit(X_new, y_new, "model.ckpt")
```

---

## Getting Help

### Documentation

- **This file**: How to contribute (tasks, code style)
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Why the system is designed this way
- **[../theory/conceptual_model.md](../theory/conceptual_model.md)**: Theoretical foundations
- **[../theory/mathematical_specification.md](../theory/mathematical_specification.md)**: Mathematical details
- **[../theory/implementation_roadmap.md](../theory/implementation_roadmap.md)**: Current status and future plans

### Code

- **Public API**: Start with `src/ssvae/models.py`
- **Protocols**: Check `src/ssvae/priors/base.py`
- **Examples**: See `src/ssvae/priors/mixture.py`, `src/callbacks/logging.py`

### Community

- **Bug reports**: GitHub Issues
- **Feature requests**: GitHub Discussions
- **Questions**: GitHub Discussions or team chat

---

## Next Steps

1. **Setup**: Install dependencies and run tests
2. **Explore**: Run an experiment to see the system in action
3. **Modify**: Try the Quick Start code change
4. **Choose a task**: Pick from Common Tasks above
5. **Submit**: Create PR with your changes

Thank you for contributing! 🎉
