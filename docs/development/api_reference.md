# API Reference

> **Purpose:** Concise module-by-module reference for the `/src/` codebase. For design patterns, see [Architecture](architecture.md). For usage examples, see [Extending](extending.md).

---

## Module Organization

```
src/
├── ssvae/              # Core SSVAE model
├── training/           # Training infrastructure
├── callbacks/          # Training observability
└── utils/              # Utilities
```

---

## Core Model (`src/ssvae/`)

### `models.py` - Public API

**Main Class:** `SSVAE`

**Purpose:** User-facing interface for creating, training, and using SSVAE models.

**Constructor:**
```python
SSVAE(input_dim: Tuple[int, int], config: SSVAEConfig | None = None)
```
- `input_dim`: Shape of input data (e.g., `(28, 28)` for MNIST)
- `config`: Model configuration (uses defaults if None)

**Key Methods:**
- `fit(data, labels, weights_path, callbacks=None)` → Train model and save checkpoint
- `predict(data, sample=False, num_samples=1)` → Run inference, returns `(latent, recon, predictions, certainty)`
- `load_model_weights(weights_path)` → Load parameters from checkpoint

**Example:**
```python
from ssvae import SSVAE, SSVAEConfig

config = SSVAEConfig(latent_dim=2, prior_type="mixture")
model = SSVAE(input_dim=(28, 28), config=config)
history = model.fit(X_train, y_train, "model.ckpt")
z, recon, preds, cert = model.predict(X_test)
```

---

### `config.py` - Configuration

**Main Class:** `SSVAEConfig` (dataclass)

**Purpose:** Type-safe configuration with validation and defaults.

**Key Parameter Groups:**

**Architecture:**
- `latent_dim: int = 2` - Latent space dimensionality
- `hidden_dims: Tuple[int, ...] = (256, 128, 64)` - Layer sizes
- `num_classes: int = 10` - Number of output classes
- `encoder_type: str = "dense"` - "dense" or "conv"
- `decoder_type: str = "dense"` - "dense" or "conv"

**Prior:**
- `prior_type: str = "standard"` - "standard" or "mixture"
- `num_components: int = 10` - Mixture components (if mixture)
- `use_component_aware_decoder: bool = True` - Enable component-aware decoding

**Loss Weights:**
- `recon_weight: float = 1.0` - Reconstruction loss weight
- `kl_weight: float = 1.0` - KL divergence weight
- `label_weight: float = 1.0` - Classification loss weight
- `kl_c_weight: float = 1.0` - Component KL weight (mixture only)

**Regularization:**
- `component_diversity_weight: float = 0.0` - Component usage diversity (negative = reward)
- `dirichlet_alpha: float | None = None` - Dirichlet prior concentration
- `weight_decay: float = 0.0` - L2 regularization

**Training:**
- `batch_size: int = 128`
- `learning_rate: float = 1e-3`
- `max_epochs: int = 100`
- `patience: int = 10` - Early stopping patience

**See:** [src/ssvae/config.py](../../src/ssvae/config.py) for full parameter list

---

### `network.py` - Neural Network Architecture

**Main Classes:**

**`SSVAENetwork` (nn.Module)**

**Purpose:** Main neural network combining encoder, decoder, and classifier.

**Key Method:**
- `__call__(x, training: bool)` → `ForwardOutput`

**Output Structure:** `ForwardOutput` (NamedTuple)
- `component_logits: Array | None` - Mixture component logits
- `z_mean: Array` - Latent mean
- `z_log: Array` - Latent log-variance
- `z: Array` - Sampled latent vector
- `recon: Array` - Reconstruction
- `class_logits: Array` - Classification logits
- `extras: Dict` - Additional outputs (e.g., mixture info)

**`MixturePriorParameters` (nn.Module)**

**Purpose:** Learnable parameters for mixture prior.

**Returns:** `(component_embeddings, pi_logits)`
- `component_embeddings: Array` - Component-specific embeddings `[K, latent_dim]`
- `pi_logits: Array` - Mixture weights before softmax `[K]`

---

### `factory.py` - Component Creation

**Main Class:** `SSVAEFactory`

**Purpose:** Centralized factory for creating and validating model components.

**Key Methods:**

- `create_model(input_dim, config)` → Complete model with all components
- `create_network(config, input_shape, key)` → Neural network
- `create_encoder(config, key)` → Encoder module
- `create_decoder(config, input_shape, key)` → Decoder module
- `create_classifier(config, key)` → Classifier head
- `create_prior(config)` → Prior distribution

**Example:**
```python
factory = SSVAEFactory()
model, state, train_step, eval_fn, rng, prior = factory.create_model(
    input_dim=(28, 28),
    config=config
)
```

---

### `checkpoint.py` - State Persistence

**Main Class:** `CheckpointManager`

**Purpose:** Save and load model state to/from disk.

**Key Methods:**

- `save_checkpoint(state, config, path)` → Save model
- `load_checkpoint(path)` → Load model, returns `(state, config)`

**File Format:** Pickle with `{state, config, metadata}`

---

### `diagnostics.py` - Metrics Collection

**Main Class:** `DiagnosticsCollector`

**Purpose:** Collect and organize training metrics, especially for mixture priors.

**Key Methods:**

- `collect_mixture_diagnostics(state, data, config)` → Mixture metrics dict

**Collected Metrics:**
- `component_usage` - Empirical usage distribution
- `component_entropy` - Responsibility entropy
- `pi` - Mixture weights (softmax of logits)
- `pi_entropy` - Entropy of π distribution
- `K_eff` - Effective number of components used

---

## Network Components (`src/ssvae/components/`)

### Encoders (`encoders/`)

**Purpose:** Encode inputs to latent distribution parameters.

**Classes:**

**`DenseEncoder(nn.Module)`**
- Fully connected encoder
- Returns: `(z_mean, z_logvar)` for standard prior
- Returns: `(z_mean, z_logvar, component_logits)` for mixture prior

**`ConvEncoder(nn.Module)`**
- Convolutional encoder for image data
- Similar output structure to DenseEncoder

**Key Parameters:**
- `latent_dim` - Latent space dimension
- `hidden_dims` - Layer sizes
- `num_components` - Mixture components (if applicable)

---

### Decoders (`decoders/`)

**Purpose:** Decode latent vectors to reconstructions.

**Classes:**

**`DenseDecoder(nn.Module)`**
- Fully connected decoder
- Returns: `reconstruction` (same shape as input)

**`ConvDecoder(nn.Module)`**
- Transposed convolutional decoder
- Mirrors ConvEncoder architecture

**`ComponentAwareDenseDecoder(nn.Module)`**
- Decoder conditioned on both z and component c
- Separate pathways for latent and component embedding
- Returns: Weighted reconstruction based on responsibilities

**`ComponentAwareConvDecoder(nn.Module)`**
- Convolutional variant of component-aware decoder

**Key Parameters:**
- `output_shape` - Reconstructed data shape
- `hidden_dims` - Layer sizes
- `component_embedding_dim` - Size of component embeddings (component-aware only)

---

### Classifiers (`classifiers/`)

**Purpose:** Map latent vectors to class predictions.

**Class:** `DenseClassifier(nn.Module)`

**Returns:** `logits` of shape `(batch_size, num_classes)`

**Note:** This will be replaced by τ-based classification in the full RCM-VAE architecture.

---

## Prior Distributions (`src/ssvae/priors/`)

### `base.py` - Protocol

**`PriorMode` (Protocol)**

**Purpose:** Interface for pluggable prior distributions.

**Required Methods:**
- `kl_divergence(z_mean, z_logvar, ...)` → KL divergence
- `sample(key, latent_dim, num_samples)` → Samples from prior

---

### `standard.py` - Standard Gaussian

**Class:** `StandardPrior`

**Prior:** $p(z) = \mathcal{N}(0, I)$

**KL Divergence:** Analytical formula
```python
kl = -0.5 * sum(1 + z_logvar - z_mean^2 - exp(z_logvar))
```

**No learnable parameters.**

---

### `mixture.py` - Mixture of Gaussians

**Class:** `MixturePrior`

**Prior:** $p(z) = \sum_k \pi_k \mathcal{N}(0, I)$

**Key Parameters:**
- `num_components` - Number of mixture components (K)
- `dirichlet_alpha` - Dirichlet prior concentration (optional)
- `dirichlet_weight` - Dirichlet regularization weight

**Learnable Parameters:**
- `pi_logits` - Mixture weights (before softmax)
- `component_embeddings` - Component-specific embeddings (if component-aware decoder)

**KL Components:**
1. KL_z: Standard Gaussian KL (weighted by responsibilities)
2. KL_c: Component assignment KL(q(c|x) || π)

**Key Methods:**
- `get_mixture_weights(params)` → π (normalized weights)
- `compute_component_kl(component_logits, pi)` → KL(q(c|x) || π)

---

## Training Infrastructure (`src/training/`)

### `trainer.py` - Main Training Loop

**Main Class:** `Trainer`

**Purpose:** Orchestrate training with early stopping and data management.

**Key Classes:**

**`DataSplits` (dataclass)**
- Encapsulates train/validation split
- Fields: `x_train`, `y_train`, `x_val`, `y_val`, sizes, labeled count

**`TrainingSetup` (dataclass)**
- Training configuration: batch size, epochs, patience

**`EarlyStoppingTracker` (dataclass)**
- Tracks early stopping state and manages checkpoints

**`Trainer` (main class)**

**Key Method:**
```python
train(state, data, labels, weights_path, shuffle_rng,
      train_step_fn, eval_metrics_fn, save_fn,
      callbacks, num_epochs, patience)
```

**Returns:** `(final_state, final_rng, history)`

**Features:**
- 80/20 train/validation split
- Early stopping on validation loss
- Automatic checkpoint saving
- Callback integration

---

### `interactive_trainer.py` - Stateful Trainer

**Main Class:** `InteractiveTrainer`

**Purpose:** Stateful wrapper for incremental training (active learning workflows).

**Key Methods:**
- `train_epochs(num_epochs, data, labels, ...)` → Train while preserving state
- `get_latent_space(data)` → Deterministic latent coordinates
- `predict(data, ...)` → Inference
- `save_checkpoint(path)` / `load_checkpoint(path)` → Persistence

**Use Case:** Dashboard and active learning where training happens in multiple rounds.

---

### `losses.py` - Loss Functions

**Purpose:** Composable loss functions for SSVAE training.

**Key Functions:**

**Reconstruction:**
- `reconstruction_loss_mse(x, recon)` → MSE
- `reconstruction_loss_bce(x, recon)` → Binary cross-entropy
- `weighted_reconstruction_loss_*()` → For mixture priors

**KL Divergence:**
- `kl_divergence(z_mean, z_logvar)` → Standard Gaussian KL
- `categorical_kl(q, pi)` → Component assignment KL

**Regularization:**
- `dirichlet_map_penalty(pi, alpha)` → Dirichlet prior
- `usage_sparsity_penalty(usage)` → Component diversity
- `classification_loss(logits, labels, mask)` → Cross-entropy

**Main Loss:**
```python
compute_loss_and_metrics_v2(params, batch_x, batch_y,
                            model_apply_fn, config, prior,
                            rng, training, kl_c_scale)
```

**Returns:** `(total_loss, metrics_dict)` with 10+ metrics

---

### `train_state.py` - Training State

**Main Class:** `SSVAETrainState`

**Purpose:** Extends Flax's `TrainState` with RNG tracking.

**Fields:**
- `step` - Training step counter (inherited)
- `apply_fn` - Model forward pass (inherited)
- `params` - Model parameters (inherited)
- `tx` - Optimizer (inherited)
- `opt_state` - Optimizer state (inherited)
- `rng` - Random key (added)

**Class Method:**
```python
SSVAETrainState.create(apply_fn, params, tx, rng)
```

**Usage:** Encapsulates all training state for pure functional training loop.

---

## Callbacks (`src/callbacks/`)

### Base Callback (`base_callback.py`)

**Class:** `TrainingCallback`

**Methods:**
- `on_train_begin()` - Called before training
- `on_epoch_begin(epoch)` - Called before each epoch
- `on_epoch_end(epoch, history)` - Called after each epoch
- `on_train_end()` - Called after training completes

---

### Logging (`logging.py`)

**Classes:**

**`ConsoleLogger(TrainingCallback)`**
- Prints metrics to console
- Parameters: `log_every` (default: 1)

**`CSVExporter(TrainingCallback)`**
- Exports metrics to CSV file
- Parameters: `csv_path`, `metrics_to_log`

---

### Plotting (`plotting.py`)

**Class:** `LossCurvePlotter(TrainingCallback)`

**Purpose:** Visualize training progress with loss curves.

**Parameters:**
- `save_path` - Where to save plot
- `show_plot` - Display interactively

---

### Mixture Tracking (`mixture_tracker.py`)

**Class:** `MixtureHistoryTracker(TrainingCallback)`

**Purpose:** Track mixture prior evolution (π, usage) over time.

**Parameters:**
- `log_every` - Track every N epochs
- `export_dir` - Where to save history arrays

**Exports:**
- `pi_history.npy` - π values over time
- `usage_history.npy` - Component usage over time
- `tracked_epochs.npy` - Which epochs were tracked

---

## Utilities (`src/utils/`)

### `device.py` - JAX Device Management

**Functions:**

- `configure_jax_device(prefer_gpu=True)` → Configure JAX device
- `print_device_banner()` → Print device info banner

**Usage:**
```python
from utils import configure_jax_device
configure_jax_device()  # Call before importing jax.numpy
```

---

## Related Documentation

- **[Development Overview](OVERVIEW.md)** - Quick intro to codebase
- **[Architecture](architecture.md)** - Design patterns and philosophy
- **[Status](STATUS.md)** - Current implementation status
- **[Decisions](DECISIONS.md)** - Architectural choices and rationale
- **[Extending](extending.md)** - How to add new features
- **[Conceptual Model](../theory/conceptual_model.md)** - Theoretical foundation
- **[Math Specification](../theory/mathematical_specification.md)** - Mathematical formulations
