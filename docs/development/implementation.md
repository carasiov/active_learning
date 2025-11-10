# Implementation Guide

> **Purpose:** Module-by-module reference for developers working with the codebase. For high-level design patterns, see [System Architecture](architecture.md). For tutorials on adding features, see [Extending the System](extending.md).

---

## Module Organization

```
src/
├── ssvae/              # Core SSVAE model
│   ├── models.py       # Public API (SSVAE class)
│   ├── network.py      # Neural network architecture
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

### `network.py` - Neural Network Architecture

**Purpose:** Core neural network architecture components used by the SSVAE model.

**Key Classes:**

**`ForwardOutput` (NamedTuple)**

Standardized output format from network forward pass.

```python
class ForwardOutput(NamedTuple):
    component_logits: Optional[jnp.ndarray]  # Mixture component logits (if mixture prior)
    z_mean: jnp.ndarray                      # Latent mean
    z_log: jnp.ndarray                       # Latent log-variance
    z: jnp.ndarray                           # Sampled latent vector
    recon: jnp.ndarray                       # Reconstruction
    class_logits: jnp.ndarray                # Classification logits
    extras: Dict[str, jnp.ndarray]           # Additional outputs (e.g., mixture info)
```

**`MixturePriorParameters` (nn.Module)**

Learnable parameters for mixture prior distributions.

```python
class MixturePriorParameters(nn.Module):
    num_components: int     # Number of mixture components
    embed_dim: int          # Embedding dimension per component

    def __call__(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Returns: (component_embeddings, pi_logits)
```

Creates two learnable parameter arrays:
- `component_embeddings`: Component-specific embeddings `[K, latent_dim]`
- `pi_logits`: Mixture weights before softmax `[K]`

**`SSVAENetwork` (nn.Module)**

Main neural network module combining encoder, decoder, and classifier.

**Parameters:**
```python
config: SSVAEConfig                          # Model configuration
input_hw: Tuple[int, int]                    # Input dimensions
encoder_hidden_dims: Tuple[int, ...]         # Encoder layer sizes
decoder_hidden_dims: Tuple[int, ...]         # Decoder layer sizes
classifier_hidden_dims: Tuple[int, ...]      # Classifier layer sizes
classifier_dropout_rate: float               # Dropout rate
latent_dim: int                              # Latent space dimension
output_hw: Tuple[int, int]                   # Output dimensions
encoder_type: str                            # "dense" or "conv"
decoder_type: str                            # "dense" or "conv"
classifier_type: str                         # "dense"
```

**Forward Pass:**
```python
def __call__(self, x: jnp.ndarray, *, training: bool) -> ForwardOutput:
    """
    Args:
        x: Input images [batch, H, W]
        training: Whether in training mode

    Returns:
        ForwardOutput with all network outputs
    """
```

**Standard Prior Flow:**
1. Encoder: `x → (z_mean, z_log, z)`
2. Decoder: `z → reconstruction`
3. Classifier: `z → class_logits`

**Mixture Prior Flow:**
1. Encoder: `x → (component_logits, z_mean, z_log, z)`
2. Get mixture parameters: `prior_module() → (embeddings, pi_logits)`
3. Compute responsibilities: `q(c|x) = softmax(component_logits)`
4. Decoder with components:
   - Tile z for each component: `[batch, K, latent_dim]`
   - Concatenate with embeddings: `[batch, K, latent_dim + embed_dim]`
   - Decode per-component: `recon_per_component [batch, K, H, W]`
   - Weight by responsibilities: `recon = Σ_k q(c_k|x) * recon_k`
5. Classifier: `z → class_logits`

**Utility Functions:**

**`_make_weight_decay_mask(params)`**

Creates a mask for selective weight decay application.

```python
def _make_weight_decay_mask(params: Dict) -> Dict:
    """
    Create mask matching params tree structure for weight decay.

    Returns:
        Mask dict where True = apply decay, False = no decay

    No decay applied to:
    - bias terms
    - scale terms (BatchNorm, LayerNorm)
    - prior parameters (pi_logits, component_embeddings)
    """
```

**Usage:**
```python
# Used by SSVAEFactory to configure optimizer
decay_mask = _make_weight_decay_mask(params)
optimizer = optax.adamw(
    learning_rate=config.learning_rate,
    weight_decay=config.weight_decay,
    mask=decay_mask
)
```

**Design Notes:**
- `ForwardOutput` provides type-safe interface for network outputs
- `MixturePriorParameters` keeps mixture prior learnable params organized
- `SSVAENetwork` orchestrates all components with prior-specific logic
- Weight decay masking prevents regularization of bias/prior parameters

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
component_diversity_weight: float = 0.0  # Component usage diversity (negative = reward)
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

**Purpose:** Main training loop with early stopping, data splitting, and checkpoint management.

**Key Classes:**

**`DataSplits` (dataclass)**
- Encapsulates train/validation split
- Fields: `x_train`, `y_train`, `x_val`, `y_val`, `train_size`, `val_size`, `total_samples`, `labeled_count`
- Property: `has_labels` - checks if any labeled samples exist
- Method: `with_train()` - create new split with updated training data

**`TrainingSetup` (dataclass)**
- Training configuration container
- Fields: `batch_size`, `eval_batch_size`, `max_epochs`, `patience`, `monitor_metric`

**`EarlyStoppingTracker` (dataclass)**
- Tracks early stopping state
- Fields: `monitor_metric`, `patience`, `best_val`, `wait`, `checkpoint_saved`, `halted_early`
- Method: `update()` - update tracker and save checkpoint if improved

**`Trainer`**
- Main training orchestrator
- Works with `SSVAETrainState` (not raw variables)
- Integrates with callbacks for observability
- Extensible via `TrainerLoopHooks` for per-batch context, post-batch side effects, and evaluation context (used by τ-classifier integration)

**Key Methods:**

**`train(state, data, labels, weights_path, shuffle_rng, train_step_fn, eval_metrics_fn, save_fn, callbacks, num_epochs, patience)`**
- Main training loop with functional state passing
- Handles data splitting, batching, early stopping
- Returns: `(final_state, final_rng, history)`

**Training Loop Structure:**
```python
# Split data
splits = self._prepare_data_splits(data, labels)

# Initialize history
history = self._init_history()

# Training loop
for epoch in range(max_epochs):
    # Train epoch
    state, train_metrics = train_epoch(state, splits, train_step_fn)

    # Validation
    val_metrics = eval_metrics_fn(state.params, splits.x_val, splits.y_val)

    # Update history
    history = update_history(history, train_metrics, val_metrics)

    # Callbacks
    for callback in callbacks:
        callback.on_epoch_end(epoch, history)

    # Early stopping check
    if early_stopping.update(val_metrics, state=state, save_fn=save_fn):
        break

return state, shuffle_rng, history
```

**Features:**
- Functional state management (JAX style)
- 80/20 train/validation split
- Early stopping on configurable metric (default: `val_loss`)
- Automatic checkpoint saving on improvement
- Comprehensive metric tracking (20+ metrics in history)
- Optional `TrainerLoopHooks` (dataclass with `batch_context_fn`, `post_batch_fn`, `eval_context_fn`) let features like the τ-classifier update Python-side state after each batch while still feeding data (e.g., the current τ matrix) into compiled `train_step`/`eval_metrics` calls. `SSVAE._build_tau_loop_hooks()` wires these hooks automatically when `use_tau_classifier=True`, and standard training runs simply omit them (no overhead).

---

### `interactive_trainer.py`

**Purpose:** Stateful trainer for incremental/interactive training sessions.

**Key Class: `InteractiveTrainer`**

**Use Case:** Active learning workflows where training happens in multiple rounds with incrementally added labels.

**Features:**
- Preserves optimizer state across training sessions
- Supports incremental label addition
- Mirrors SSVAE interface while maintaining state
- Used by dashboard for interactive training

**Key Methods:**

**`__init__(model, export_history=False, callbacks=None)`**
- Wraps an SSVAE model instance
- Maintains reference to model's state and RNG

**`train_epochs(num_epochs, data, labels, weights_path=None, patience=None)`**
- Train for N epochs while preserving optimizer state
- Returns: Training history dict
- Automatically resumes from current state
- Delegates to `Trainer.train(..., loop_hooks=model._build_tau_loop_hooks())`, so τ-enabled models continue to update counts via the same hook pathway even during incremental sessions.

**`get_latent_space(data)`**
- Returns deterministic latent coordinates for visualization
- Uses current model state

**`predict(data, sample=False, num_samples=1)`**
- Run inference using current state
- Delegates to wrapped SSVAE model

**`save_checkpoint(path)` / `load_checkpoint(path)`**
- Persist/restore model state and optimizer state

**Usage Pattern:**
```python
from ssvae import SSVAE
from training.interactive_trainer import InteractiveTrainer

# Create model and trainer
model = SSVAE(input_dim=(28, 28), config=config)
trainer = InteractiveTrainer(model)

# Round 1: Train with initial labels
history1 = trainer.train_epochs(10, X, y_initial)

# Round 2: Add more labels, continue training
y_updated = add_labels(y_initial, new_labels)
history2 = trainer.train_epochs(10, X, y_updated)  # Resumes from previous state

# Get results
z = trainer.get_latent_space(X)
preds = trainer.predict(X_test)
```

**Design:**
- Wraps `Trainer` internally for actual training logic
- Maintains state between calls (not functional like `Trainer`)
- Ideal for interactive/dashboard workflows

---

### `losses.py` - Loss Functions

**Purpose:** Loss computation functions with protocol-based abstraction.

**Key Components:**

**Utility Functions:**
- `reconstruction_loss_mse()` / `reconstruction_loss_bce()` - Simple reconstruction losses
- `weighted_reconstruction_loss_mse()` / `weighted_reconstruction_loss_bce()` - For mixture priors
- `kl_divergence()` - Standard Gaussian KL
- `categorical_kl()` - Component assignment KL
- `dirichlet_map_penalty()` - Dirichlet prior regularization
- `usage_sparsity_penalty()` - Component diversity regularization
- `classification_loss()` - Cross-entropy on labeled samples

**Main Loss Function:**

**`compute_loss_and_metrics_v2(params, batch_x, batch_y, model_apply_fn, config, prior, rng, training, kl_c_scale)`**

Protocol-based loss computation that delegates to priors for their specific logic.

**Args:**
- `params`: Model parameters
- `batch_x`: Input images
- `batch_y`: Labels (NaN for unlabeled)
- `model_apply_fn`: Forward pass function
- `config`: SSVAEConfig
- `prior`: PriorMode instance (handles prior-specific logic)
- `rng`: Random key (None for deterministic)
- `training`: Training mode flag
- `kl_c_scale`: Annealing factor for component KL

**Returns:**
```python
(total_loss, metrics_dict)

# metrics_dict contains:
{
    'loss': total_loss,
    'reconstruction_loss': recon,
    'kl_z': kl_latent,
    'kl_c': kl_component,
    'kl_loss': kl_z + kl_c,
    'classification_loss': cls_unweighted,
    'weighted_classification_loss': cls_weighted,
    'component_diversity': diversity,
    'component_entropy': entropy,
    'pi_entropy': pi_entropy,
    'dirichlet_penalty': dirichlet,
    'loss_no_global_priors': recon + kl_z + kl_c + cls,
    'contrastive_loss': 0.0  # placeholder
}
```

**Design Pattern:**
```python
# Prior handles its own KL computation
encoder_output = EncoderOutput(z_mean, z_log_var, z, component_logits, extras)
kl_terms = prior.compute_kl_terms(encoder_output, config)

# Prior handles reconstruction weighting (if needed)
recon_loss = prior.compute_reconstruction_loss(
    batch_x, recon, encoder_output, config
)

# Assemble total loss
total = recon_loss + sum(kl_terms.values()) + classification_loss
```

**Usage in Factory:**
```python
# SSVAEFactory automatically uses protocol-based losses
factory.create_model(input_dim, config)
```

---

### `train_state.py`

**Purpose:** Manage training state with RNG tracking.

**Key Class: `SSVAETrainState`**

Extends Flax's `train_state.TrainState` to include RNG metadata.

```python
class SSVAETrainState(train_state.TrainState):
    """TrainState carrying RNG metadata required during training."""

    rng: jax.Array  # Additional field for random number generation

    # Inherited from train_state.TrainState:
    # - step: int (training step counter)
    # - apply_fn: Callable (model forward pass)
    # - params: FrozenDict (model parameters)
    # - tx: optax.GradientTransformation (optimizer)
    # - opt_state: optax.OptState (optimizer state)
```

**Class Method:**

**`create(apply_fn, params, tx, rng, **kwargs)`**
- Factory method to instantiate training state
- Initializes optimizer state automatically
- Returns: `SSVAETrainState` instance

**Usage:**
```python
import optax
from training.train_state import SSVAETrainState

# Create training state
state = SSVAETrainState.create(
    apply_fn=network.apply,
    params=initial_params,
    tx=optax.adam(learning_rate=1e-3),
    rng=jax.random.PRNGKey(42)
)

# Use in training
new_state = state.replace(
    step=state.step + 1,
    params=updated_params,
    opt_state=updated_opt_state
)
```

**Design:**
- Encapsulates all mutable training state
- Enables pure functional training loop
- RNG tracking ensures reproducibility
- Easy to checkpoint/restore (all state in one object)

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
    component_diversity_weight=1.0
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
- **[Experiment Guide](../../use_cases/experiments/README.md)** - Primary experimentation workflow
- **[Dashboard Guide](../../use_cases/dashboard/README.md)** - Interactive active learning interface
