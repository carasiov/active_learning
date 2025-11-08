# Implementation Guide

> **Purpose:** Module-by-module reference for developers working with the codebase. For high-level design patterns, see [System Architecture](architecture.md). For tutorials on adding features, see [Extending the System](extending.md).

---

## Module Organization

```
src/
├── ssvae/              # Core SSVAE model
│   ├── models.py       # Public API (SSVAE class)
│   ├── config.py       # Configuration dataclass
│   ├── factory.py      # Component creation
│   ├── checkpoint.py   # State persistence
│   ├── diagnostics.py  # Metrics collection
│   ├── components/     # Neural network components
│   │   ├── encoders.py
│   │   ├── decoders.py
│   │   ├── classifier.py
│   │   └── factory.py
│   └── priors/         # Prior distributions
│       ├── base.py     # PriorMode protocol
│       ├── standard.py
│       └── mixture.py
│
├── training/           # Training infrastructure
│   ├── trainer.py      # Main training loop
│   ├── losses.py       # Loss functions
│   └── train_state.py  # Training state management
│
├── callbacks/          # Training observability
│   ├── base_callback.py
│   ├── logging.py
│   └── plotting.py
│
└── utils/              # Utilities
    └── device.py       # JAX device selection
```

---

## Core Model (`src/ssvae/`)

### `models.py` - Public API

**Purpose:** Main user-facing interface for SSVAE model.

**Key Class: `SSVAE`**

```python
class SSVAE:
    """Semi-Supervised Variational Autoencoder with pluggable priors."""

    def __init__(self, input_dim: Tuple[int, ...], config: SSVAEConfig):
        """Initialize SSVAE model.

        Args:
            input_dim: Shape of input data (e.g., (28, 28) for MNIST)
            config: Model configuration
        """
```

**Public Methods:**

- **`fit(data, labels, weights_path, callbacks=None)`**
  - Train the model on semi-supervised data
  - `labels`: Can contain NaN for unlabeled samples
  - Returns: Training history dict

- **`predict(data, sample=False, num_samples=1)`**
  - Run inference on data
  - Returns: `(latent, reconstruction, predictions, certainty)`

- **`load_model_weights(weights_path)`**
  - Load model parameters from checkpoint

**Internal Structure:**
```python
self.config: SSVAEConfig          # Model configuration
self.input_shape: Tuple[int, ...] # Data shape
self.network: SSVAENetwork        # Neural network (from factory)
self.variables: FrozenDict        # JAX parameters
```

**Usage Example:**
```python
from ssvae import SSVAE, SSVAEConfig

config = SSVAEConfig(latent_dim=2, prior_type="mixture")
model = SSVAE(input_dim=(28, 28), config=config)
history = model.fit(X_train, y_train, "model.ckpt")
z, recon, preds, cert = model.predict(X_test)
```

---

### `config.py` - Configuration

**Purpose:** Type-safe configuration with validation and defaults.

**Key Class: `SSVAEConfig`**

**Parameter Categories:**

**Architecture:**
```python
latent_dim: int = 2                      # Latent space dimensionality
hidden_dims: Tuple[int, ...] = (256, 128, 64)  # Layer sizes
num_classes: int = 10                    # Number of output classes
encoder_type: str = "dense"              # "dense" or "conv"
decoder_type: str = "dense"              # "dense" or "conv"
```

**Prior:**
```python
prior_type: str = "standard"             # "standard" or "mixture"
num_components: int = 10                 # Mixture components (if mixture)
```

**Loss Weights:**
```python
recon_weight: float = 1.0                # Reconstruction loss weight
kl_weight: float = 1.0                   # KL divergence weight
label_weight: float = 1.0                # Classification loss weight
kl_c_weight: float = 1.0                 # Component KL weight (mixture)
reconstruction_loss: str = "bce"         # "bce" or "mse"
```

**Regularization:**
```python
usage_sparsity_weight: float = 0.0       # Channel usage sparsity
dirichlet_alpha: Optional[float] = None  # Dirichlet prior concentration
dirichlet_weight: float = 1.0            # Dirichlet prior weight
```

**Training:**
```python
batch_size: int = 128
learning_rate: float = 1e-3
max_epochs: int = 100
patience: int = 10                       # Early stopping patience
random_seed: int = 42
```

**Usage:**
```python
# Default configuration
config = SSVAEConfig()

# Custom configuration
config = SSVAEConfig(
    latent_dim=10,
    prior_type="mixture",
    num_components=20,
    learning_rate=5e-4
)
```

---

### `factory.py` - Component Creation

**Purpose:** Centralized factory for creating and validating model components.

**Key Class: `SSVAEFactory`**

**Main Methods:**

**`create_network(config, input_shape, key)`**
- Creates complete neural network with all components
- Returns: `(network, variables)`
- Validates configuration consistency

**`create_encoder(config, key)`**
- Creates encoder based on `config.encoder_type`
- Supports: dense, convolutional

**`create_decoder(config, input_shape, key)`**
- Creates decoder based on `config.decoder_type`
- Ensures output shape matches input

**`create_classifier(config, key)`**
- Creates classifier head
- Output: `num_classes` logits

**`create_prior(config)`**
- Creates prior distribution
- Returns: `PriorMode` implementation

**Design Pattern:**
```python
# Factory validates and creates compatible components
network, variables = SSVAEFactory.create_network(
    config=config,
    input_shape=(28, 28),
    key=jax.random.PRNGKey(42)
)
```

---

### `checkpoint.py` - State Persistence

**Purpose:** Save and load model state.

**Key Class: `CheckpointManager`**

**Methods:**

**`save_checkpoint(variables, config, path)`**
- Saves model parameters and configuration
- Uses pickle for serialization
- Stores metadata (timestamp, version)

**`load_checkpoint(path)`**
- Loads model state from disk
- Returns: `(variables, config)`
- Handles version compatibility

**File Format:**
```python
{
    'variables': FrozenDict,  # JAX parameters
    'config': SSVAEConfig,    # Model configuration
    'metadata': {
        'timestamp': str,
        'version': str
    }
}
```

---

### `diagnostics.py` - Metrics Collection

**Purpose:** Collect and organize training metrics, especially for mixture priors.

**Key Class: `DiagnosticsCollector`**

**Methods:**

**`collect_mixture_diagnostics(variables, data, network, config)`**
- Computes mixture prior metrics
- Returns: Dict with usage, entropy, π values

**Collected Metrics:**
- `component_usage`: Empirical usage $\hat{p}(c)$
- `component_entropy`: Responsibility entropy
- `pi`: Mixture weights (softmax of logits)
- `pi_entropy`: Entropy of π distribution

**Usage:**
```python
diagnostics = DiagnosticsCollector.collect_mixture_diagnostics(
    variables=model.variables,
    data=validation_data,
    network=model.network,
    config=model.config
)
print(f"Active components: {(diagnostics['component_usage'] > 0.01).sum()}")
```

---

## Network Components (`src/ssvae/components/`)

### `encoders.py`

**Purpose:** Encode inputs to latent distribution parameters.

**Classes:**

**`DenseEncoder(nn.Module)`**
- Fully connected encoder
- Architecture: input → hidden layers → (z_mean, z_logvar)
- For mixture prior: also outputs component_logits

**Parameters:**
```python
latent_dim: int                   # Latent space dimension
hidden_dims: Tuple[int, ...]      # Layer sizes (e.g., (256, 128, 64))
num_components: Optional[int]     # Mixture components (if applicable)
dropout_rate: float = 0.0         # Dropout probability
```

**Output:**
```python
# Standard prior
z_mean, z_logvar = encoder(x, deterministic=True)

# Mixture prior
z_mean, z_logvar, component_logits = encoder(x, deterministic=True)
```

**`ConvEncoder(nn.Module)`**
- Convolutional encoder for image data
- Architecture: Conv layers → Flatten → Dense → (z_mean, z_logvar)
- Useful for larger images (CIFAR-10, etc.)

---

### `decoders.py`

**Purpose:** Decode latent vectors to reconstructions.

**Classes:**

**`DenseDecoder(nn.Module)`**
- Fully connected decoder
- Architecture: z → hidden layers → output_shape
- Sigmoid activation for [0, 1] pixels

**Parameters:**
```python
output_shape: Tuple[int, ...]     # Reconstructed data shape
hidden_dims: Tuple[int, ...]      # Layer sizes (reversed from encoder)
dropout_rate: float = 0.0
```

**Output:**
```python
reconstruction = decoder(z, deterministic=True)  # Shape: output_shape
```

**`ConvDecoder(nn.Module)`**
- Transposed convolutional decoder
- Mirrors ConvEncoder architecture
- Dense → ConvTranspose layers → output_shape

---

### `classifier.py`

**Purpose:** Map latent vectors to class predictions.

**Class: `DenseClassifier(nn.Module)`**

**Parameters:**
```python
num_classes: int        # Number of output classes
hidden_dim: int = 64    # Intermediate layer size
```

**Output:**
```python
logits = classifier(z)  # Shape: (batch_size, num_classes)
```

**Note:** In the RCM-VAE architecture, this will be replaced by latent-only classification via the $\tau$ map.

---

### `components/factory.py`

**Purpose:** Create network components from configuration.

Similar to top-level `factory.py` but focused on individual components. Handles:
- Component initialization
- Weight initialization
- Shape validation

---

## Prior Distributions (`src/ssvae/priors/`)

### `base.py` - PriorMode Protocol

**Purpose:** Define interface for prior distributions.

**Protocol:**
```python
class PriorMode(Protocol):
    """Interface for prior distributions."""

    def kl_divergence(
        self,
        z_mean: Array,
        z_logvar: Array,
        component_logits: Optional[Array] = None
    ) -> Array:
        """Compute KL divergence KL(q(z|x) || p(z))."""
        ...

    def sample(
        self,
        key: PRNGKey,
        latent_dim: int,
        num_samples: int = 1
    ) -> Array:
        """Sample from prior distribution."""
        ...
```

---

### `standard.py` - Standard Gaussian Prior

**Purpose:** Simple $\mathcal{N}(0, I)$ prior.

**Class: `StandardPrior`**

**KL Divergence:**
```python
# Analytical formula
kl = -0.5 * jnp.sum(1 + z_logvar - z_mean**2 - jnp.exp(z_logvar), axis=-1)
```

**Sampling:**
```python
samples = jax.random.normal(key, shape=(num_samples, latent_dim))
```

**No learnable parameters.**

---

### `mixture.py` - Mixture of Gaussians Prior

**Purpose:** Mixture prior $p(z) = \sum_k \pi_k \mathcal{N}(0, I)$.

**Class: `MixturePrior`**

**Parameters:**
```python
num_components: int                   # Number of mixture components (K)
dirichlet_alpha: Optional[float]      # Dirichlet prior concentration
dirichlet_weight: float               # Dirichlet regularization weight
```

**Learnable Parameters:**
- `pi_logits`: Mixture weights (before softmax)

**KL Divergence:**
```python
# Two components:
# 1. KL_z: Standard Gaussian KL (weighted by responsibilities)
# 2. KL_c: KL(q(c|x) || π)
kl_z = # Per-component Gaussian KL
kl_c = # Component assignment KL
total_kl = kl_z + kl_c_weight * kl_c
```

**Regularization:**
- **Dirichlet prior:** Encourages sparse π for α < 1
- **Usage sparsity:** Minimizes entropy of empirical component usage

**Key Functions:**

**`get_mixture_weights(params)`**
```python
pi = jax.nn.softmax(params['pi_logits'])  # Normalized mixture weights
```

**`compute_component_kl(component_logits, pi)`**
```python
q_c = jax.nn.softmax(component_logits)  # q(c|x)
kl_c = jnp.sum(q_c * (jnp.log(q_c) - jnp.log(pi)), axis=-1)
```

---

## Training Infrastructure (`src/training/`)

### `trainer.py`

**Purpose:** Main training loop with early stopping.

**Key Class: `Trainer`**

**Methods:**

**`train(network, variables, train_data, train_labels, config, callbacks=None)`**
- Main training loop
- Handles batching, gradient updates, early stopping
- Returns: `(final_variables, history)`

**Training Loop:**
```python
for epoch in range(max_epochs):
    # Training batches
    for batch_x, batch_y in train_loader:
        variables, loss = train_step(variables, batch_x, batch_y)

    # Validation
    val_loss = validate(variables, val_data, val_labels)

    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

    # Callbacks
    for callback in callbacks:
        callback.on_epoch_end(epoch, history)
```

**Optimizer:** Adam with configurable learning rate

---

### `losses.py` - Loss Functions

**Purpose:** Compute individual loss components.

**Key Functions:**

**`reconstruction_loss(x, x_recon, loss_type="bce")`**
- BCE: Binary cross-entropy (for [0,1] images)
- MSE: Mean squared error

**`kl_divergence(z_mean, z_logvar, prior)`**
- Delegates to prior's `kl_divergence()` method
- Handles standard and mixture priors

**`classification_loss(logits, labels)`**
- Cross-entropy on labeled samples
- Ignores NaN labels (unlabeled)

**`total_loss(x, x_recon, z_mean, z_logvar, logits, labels, prior, config)`**
- Combines all losses with configured weights
- Returns: total loss + individual components dict

**Loss Computation:**
```python
loss_dict = {
    'reconstruction': recon_weight * recon_loss,
    'kl': kl_weight * kl_div,
    'classification': label_weight * class_loss,
    'total': recon + kl + class_loss
}
```

---

### `losses_v2.py` - Enhanced Loss Functions

**Purpose:** Extended loss functions with mixture prior support.

**Additional Features:**
- Usage sparsity regularization
- Dirichlet prior on mixture weights
- Auxiliary metrics (loss without global priors)

**Key Function:**

**`compute_ssvae_loss_v2(...)`**
- Computes all loss components for mixture prior
- Returns comprehensive metrics dict:
  ```python
  {
      'loss': total_loss,
      'reconstruction_loss': recon,
      'kl_z': kl_latent,
      'kl_c': kl_component,
      'classification_loss': class_loss,
      'usage_sparsity_loss': sparsity,
      'component_entropy': entropy,
      'pi_entropy': pi_ent
  }
  ```

---

### `train_state.py`

**Purpose:** Manage training state (parameters, optimizer state).

**Key Class: `TrainState`**

```python
@dataclass
class TrainState:
    step: int
    params: FrozenDict
    opt_state: optax.OptState
    rng: PRNGKey
```

**Usage:**
- Encapsulates all mutable training state
- Enables functional training loop
- Easy to checkpoint/restore

---

## Callbacks (`src/callbacks/`)

### `base_callback.py`

**Purpose:** Base class for training callbacks.

**Class: `Callback`**

**Methods:**
```python
def on_train_begin(self): pass
def on_epoch_begin(self, epoch): pass
def on_epoch_end(self, epoch, history): pass
def on_train_end(self): pass
```

---

### `logging.py`

**Purpose:** Log training metrics to console and CSV.

**Class: `LoggingCallback`**

**Features:**
- Console output with formatted metrics
- CSV file export for post-analysis
- Configurable metric selection

**Example Output:**
```
Epoch 10/100 | Train Loss: 125.3 | Val Loss: 130.2 | Recon: 100.1 | KL: 25.2
```

---

### `plotting.py`

**Purpose:** Visualize training progress.

**Class: `PlottingCallback`**

**Features:**
- Loss curves (train/val)
- Real-time plot updates
- Save figures to disk

---

## Utilities (`src/utils/`)

### `device.py`

**Purpose:** JAX device selection and management.

**Functions:**

**`select_device(prefer_gpu=True)`**
- Selects JAX device (GPU if available, else CPU)
- Handles JAX_PLATFORMS environment variable

**`get_device_info()`**
- Returns device type and count
- Useful for debugging

---

## Working with the Code

### Common Patterns

**Creating a model:**
```python
from ssvae import SSVAE, SSVAEConfig

config = SSVAEConfig(latent_dim=2, prior_type="mixture")
model = SSVAE(input_dim=(28, 28), config=config)
```

**Training:**
```python
history = model.fit(X_train, y_train, "model.ckpt")
```

**Inference:**
```python
z, recon, preds, cert = model.predict(X_test)
```

**Custom configuration:**
```python
config = SSVAEConfig(
    latent_dim=10,
    hidden_dims=(512, 256, 128),
    prior_type="mixture",
    num_components=20,
    kl_c_weight=0.5,
    usage_sparsity_weight=1.0
)
```

### Adding New Components

See [Extending the System](extending.md) for step-by-step tutorials on:
- Adding new priors (VampPrior)
- Adding new encoders/decoders
- Adding custom loss terms
- Implementing component-aware features

---

## Testing

### Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_network_components.py

# With coverage
pytest --cov=src tests/
```

### Test Organization

- `tests/test_network_components.py` - Component unit tests
- `tests/test_integration_workflows.py` - End-to-end tests
- `tests/test_mixture_prior_regression.py` - Prior behavior tests

---

## Related Documentation

- **[System Architecture](architecture.md)** - Design patterns and principles
- **[Extending the System](extending.md)** - How-to tutorials
- **[Conceptual Model](../theory/conceptual_model.md)** - Theoretical foundation
- **[Usage Guide](../guides/usage.md)** - User-facing workflows
