# Implementation Guide

> **Purpose:** Module-by-module reference for developers working with the codebase. For high-level design patterns, see [System Architecture](architecture.md). For tutorials on adding features, see [Extending the System](extending.md).

---

## Table of Contents

- [Developer Workflow (at a glance)](#developer-workflow-at-a-glance)
- [Module Organization](#module-organization)
- [Core Model (`src/rcmvae/`)](#core-model-srcrcmvae)
- [Network Components (`src/rcmvae/domain/components/`)](#network-components-srcrcmvaedomaincomponents)
- [Prior Distributions (`src/rcmvae/domain/priors/`)](#prior-distributions-srcrcmvaedomainpriors)
- [Training Infrastructure (`src/rcmvae/application/services/training_service.py`)](#training-infrastructure-srcrcmvaeapplicationservicestraining_servicepy)
- [Callbacks (`src/rcmvae/application/callbacks/`)](#callbacks-srcrcmvaeapplicationcallbacks)
- [Utilities (`src/rcmvae/utils/`)](#utilities-srcrcmvaeutils)
- [Working with the Code](#working-with-the-code)
- [Testing](#testing)
- [Related Documentation](#related-documentation)

---

## Developer Workflow (at a glance)

When making changes or additions, a typical flow is:

1. **Orient** – skim `AGENTS.md` and `docs/development/architecture.md` to confirm the design intent.
2. **Locate modules** – use the Module Organization below to find the relevant package under `src/rcmvae` or `src/infrastructure`.
3. **Cross-check usage** – see how experiments or the dashboard call into your target module via `use_cases/experiments` or `use_cases/dashboard`.
4. **Implement & wire** – make changes in `domain/` or `application/`, wire them through `ModelFactoryService` and `SSVAEConfig`.
5. **Validate** – run targeted tests (e.g. mixture/τ suites) and a small `run_experiment.py` config to observe behavior and visualizations.

---

## Module Organization

```
src/
├── rcmvae/                     # Core model layer
│   ├── domain/                 # Configuration, components, priors, network math
│   ├── application/            # Public API, runtime/, services/ (factory/trainer/diagnostics)
│   ├── adapters/               # Bridges into CLI/dashboard tooling
│   └── utils/                  # Device helpers (JAX runtime setup)
│
└── infrastructure/             # Shared infrastructure for experiments & dashboard
    ├── logging/                # Structured logging setup
    ├── metrics/                # Metric registry + default providers
    ├── visualization/          # Plot registry + concrete plotters
    ├── runpaths/               # Experiment run directory schema helpers
    └── utils/                  # (Future) shared infrastructure utilities
```

Experiment entrypoints and the dashboard live under `use_cases/` and are documented in:
- `use_cases/experiments/README.md` – experiment pipeline and CLI usage
- `use_cases/dashboard/README.md` – interactive app and development notes

---

## Core Model (`src/rcmvae/`)

### `application/model_api.py` - Public API

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

**Runtime-Oriented Structure:**
```python
self.config: SSVAEConfig               # Model configuration
self.input_shape: Tuple[int, ...]      # Data shape
self.runtime: ModelRuntime             # Network, train state, compiled fns, prior, RNG
self.prior: PriorMode                  # Convenience alias (from runtime.prior)
self._apply_fn: Callable               # Wrapped apply for predictions
```

`ModelRuntime` (in `src/rcmvae/application/runtime.py`) is the single source of truth for the current network and optimizer state. Services update it immutably (e.g., `Trainer` returns a new runtime after each training session), which keeps orchestration pure and simplifies checkpointing and diagnostics.

**Usage Example:**
```python
from rcmvae.application.model_api import SSVAE
from rcmvae.domain.config import SSVAEConfig

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
- `component_embeddings`: Component-specific embeddings `[K, embed_dim]`
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
   - Decode per-component via conditioning: `recon_k = decoder(z_k, embedding_k)`
   - Form the expected reconstruction using `component_selection`:
     - deterministic softmax if `use_gumbel_softmax=false`
     - (straight-through) Gumbel-softmax sample if enabled
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
# Used by ModelFactoryService to configure optimizer
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

### `factory_service.py` - Component Creation

**Location:** `src/rcmvae/application/services/factory_service.py`

**Purpose:** Centralized factory for creating and validating model components.

**Key Class: `ModelFactoryService`**

**Main Entry Point:**

**`build_runtime(input_dim, config, random_seed=None, init_data=None)`**
- Creates the full `ModelRuntime` (network, parameters, optimizers, compiled train/eval fns, prior, shuffle RNG)
- Validates configuration consistency before returning
- Handles VampPrior pseudo-input initialization when `init_data` provided

Lower-level component builders (encoders/decoders/classifiers) live under `src/rcmvae/domain/components/factory.py`. They remain importable for advanced extensions, but the preferred workflow is to request a ready-to-train runtime from `ModelFactoryService`.

**Design Pattern:**
```python
runtime = ModelFactoryService.build_runtime(
    input_dim=(28, 28),
    config=config,
    random_seed=42,
)
state = runtime.state
train_step = runtime.train_step_fn
```

---

### `checkpoint_service.py` - State Persistence

**Location:** `src/rcmvae/application/services/checkpoint_service.py`

**Purpose:** Save and load model training state.

**Key Class: `CheckpointManager`**

- `save(state, path)` – serializes `SSVAETrainState` (params, opt_state, step) to disk.
- `load(state_template, path)` – restores a state tree matching the given template.
- `checkpoint_exists(path)` / `get_checkpoint_info(path)` – lightweight existence and metadata checks.

---

### `diagnostics_service.py` - Metrics Collection

**Location:** `src/rcmvae/application/services/diagnostics_service.py`

**Purpose:** Collect and organize training diagnostics, especially for mixture priors.

**Key Class: `DiagnosticsCollector`**

- `collect_mixture_diagnostics(...)` – computes component usage, entropies, and saves statistics (including latent dumps for 2D latents) into a diagnostics directory.
- `last_output_dir` / `load_component_usage` / `load_latent_data` – helpers for downstream visualization modules.

---

## Network Components (`src/rcmvae/domain/components/`)

### `encoders.py`

**Purpose:** Encode inputs to latent distribution parameters.

**Classes:**

**`DenseEncoder(nn.Module)`**
- Fully connected encoder
- Architecture: input → hidden layers → (z_mean, z_logvar)

**`MixtureDenseEncoder(nn.Module)`**
- Dense variant used whenever `prior_type="mixture"` with dense encoders
- Emits `component_logits` in addition to `(z_mean, z_logvar, z)`
- Shares the same MLP trunk as `DenseEncoder`

**`ConvEncoder(nn.Module)`**
- Convolutional encoder for image data
- Architecture: Conv blocks → Flatten → Dense → (z_mean, z_logvar)
- Useful for larger images (CIFAR-10, etc.)

**`MixtureConvEncoder(nn.Module)`**
- Convolutional counterpart to `MixtureDenseEncoder`
- Reuses the conv stack but adds a logits head so conv architectures can drive mixture priors
- Output signature: `(component_logits, z_mean, z_logvar, z)`

---

### `decoders.py`

**Purpose:** Decode latent vectors to reconstructions.

**Modular decoders (preferred)**
- Composition pattern: `conditioner + backbone + output_head`
- Implementations: `ModularConvDecoder`, `ModularDenseDecoder`
- Modules live in `src/rcmvae/domain/components/decoder_modules/{conditioning,backbones,outputs}.py`
  - Conditioners: `ConditionalInstanceNorm`, `FiLMLayer`, `ConcatConditioner`, `NoopConditioner`
  - Backbones: `ConvBackbone`, `DenseBackbone`
  - Output heads: `StandardHead`, `HeteroscedasticHead`
- Conditioning selection is controlled by `config.decoder_conditioning ∈ {"cin","film","concat","none"}`; VampPrior and standard prior force `"none"`. See `src/rcmvae/domain/components/factory.py::build_decoder`.

**Implementing a new decoder module**
1) Choose concern:
   - New conditioner: accepts `(features, component_embedding)` → features (match shape; handle spatial broadcasting).
   - New backbone: latent → intermediate features (no conditioning/output logic).
   - New output head: features → mean or `(mean, sigma)` (apply clamping if variance).
2) Place in `decoder_modules/`, export in `__init__.py`.
3) Tests: add shape + gradient flow checks following existing mixture patterns (`tests/test_mixture_losses.py`, `tests/test_mixture_integration.py`).
4) Wire into factory if it should be selectable via config.

---

### `classifier.py`

**Purpose:** Map latent vectors to class predictions.

**Class: `Classifier(nn.Module)`**

**Parameters:**
```python
hidden_dims: Tuple[int, ...]   # Hidden MLP layers
num_classes: int               # Number of output classes
dropout_rate: float = 0.0      # Optional dropout in hidden layers
```

**Output:**
```python
logits = classifier(z, training=training)  # Shape: (batch_size, num_classes)
```

**Note:** In the RCM-VAE architecture, this will be replaced by latent-only classification via the $\tau$ map.

---

### `components/factory.py`

**Location:** `src/rcmvae/domain/components/factory.py`

**Purpose:** Create network components from configuration.

Handles:
- Component initialization
- Weight initialization
- Shape validation

---

## Prior Distributions (`src/rcmvae/domain/priors/`)

### `base.py` - PriorMode Protocol

**Purpose:** Define interface for prior distributions using protocol-based design.

**Key Classes:**
- `EncoderOutput`: Structured output from encoder (z_mean, z_log_var, z, component_logits, extras)
- `PriorMode`: Protocol defining interface all priors must implement

**Protocol methods:**
```python
class PriorMode(Protocol):
    def compute_kl_terms(self, encoder_output: EncoderOutput, config) -> Dict[str, Array]:
        """Compute all KL divergence and regularization terms."""
        
    def compute_reconstruction_loss(self, x_true, x_recon, encoder_output, config) -> Array:
        """Compute reconstruction loss (handles heteroscedastic decoders)."""
        
    def get_prior_type(self) -> str:
        """Return identifier string."""
        
    def requires_component_embeddings(self) -> bool:
        """Whether decoder needs component embeddings."""
```

### Prior Implementations

**`standard.py` - StandardGaussianPrior:**
- Simple N(0,I) Gaussian prior
- Analytical KL: 0.5 * Σ(σ² + μ² - 1 - log σ²)
- No component structure
- Use for: Baseline VAE experiments

**`mixture.py` - MixtureGaussianPrior:**
- K-channel mixture with identical N(0,I) per component
- Learnable π (mixture weights) and component embeddings
- Requires: Mixture encoder outputting responsibilities
- Use for: Production semi-supervised learning with functional specialization

**`vamp.py` - VampPrior:**
- Learned pseudo-inputs {u₁, ..., uₖ} where p(z) = Σₖ πₖ q(z|uₖ)
- Monte Carlo KL estimation
- Encoder dependency: `SSVAENetwork` re-encodes pseudo-inputs each forward pass and
  stores statistics in `EncoderOutput.extras`
- Pseudo-inputs stored in `params['prior']['pseudo_inputs']`
- `requires_component_embeddings() = False` (spatial separation is sufficient)
- Use for: Spatial clustering visualization, alternative to component-aware decoder

**`geometric_mog.py` - GeometricMixtureOfGaussiansPrior:**
- Fixed geometric centers (circle or grid arrangement)
- Analytical KL with non-zero mean priors
- WARNING: Induces artificial topology
- Use for: Diagnostics, curriculum learning, quick visualization only

### Factory Pattern

**`__init__.py` - Prior Registry:**
```python
PRIOR_REGISTRY = {
    "standard": StandardGaussianPrior,
    "mixture": MixtureGaussianPrior,
    "vamp": VampPrior,
    "geometric_mog": GeometricMixtureOfGaussiansPrior,
}

def get_prior(prior_type: str, **kwargs) -> PriorMode:
    """Create prior instance based on type string."""
```

**Key design:**
- All priors implement `PriorMode` protocol (duck typing)
- Factory enables dynamic prior selection via `config.prior_type`
- Mixture-based priors (mixture, vamp, geometric_mog) require mixture encoder
- VampPrior automatically consumes cached pseudo-input statistics provided by the network
- Learnable parameters stored in `params['prior']` (embeddings, π, pseudo-inputs)
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

## Training Infrastructure (`src/rcmvae/application/services/training_service.py`)

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

**`train(runtime, data, labels, weights_path, save_fn, callbacks, num_epochs, patience)`**
- Accepts a `ModelRuntime` (state + compiled train/eval functions + shuffle RNG)
- Handles data splitting, batching, early stopping
- Returns: `(updated_runtime, history)`

### `experiment_service.py` - Experiment Orchestration

**Purpose:** Provide a high-level façade for experiment workflows so CLI/dashboard layers don’t need to know the details of model instantiation, training, or batched inference.

**Key Pieces:**
- `ExperimentService`: builds an `SSVAE`, kicks off training (via `SSVAE.fit()`), then runs batched predictions for downstream metrics/visualizations.
- `TrainingArtifacts` dataclass: packages the trained `model`, training `history`, latent/reconstruction arrays, prediction confidence, optional responsibilities/π, training time, and diagnostics directory.

**Typical Usage:**
```python
service = ExperimentService(input_dim=(28, 28))
artifacts = service.run(
    config=SSVAEConfig(**model_config),
    x_train=X_train,
    y_train=y_semi,
    weights_path=run_paths.artifacts / "checkpoint.ckpt",
)
# Downstream layers consume artifacts.latent / artifacts.history etc.
```

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

### `runtime/interactive.py` – Interactive Trainer

**Location:** `src/rcmvae/application/runtime/interactive.py`

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
from rcmvae.application.model_api import SSVAE
from rcmvae.application.runtime.interactive import InteractiveTrainer

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
- `l1_penalty()` - Masked L1 regularization (shares weight-decay mask)
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
    'l1_penalty': l1_penalty,  # masked L1 on params, config.l1_weight scales it
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
# ModelFactoryService automatically uses protocol-based losses
runtime = ModelFactoryService.build_runtime(input_dim=input_dim, config=config)
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

## Callbacks (`src/rcmvae/application/callbacks/`)

### `base.py`

**Purpose:** Defines `TrainingCallback`, the observer interface the trainer invokes at key lifecycle events (`on_train_start`, `on_epoch_end`, `on_train_end`).

---

### `logging.py`

**Purpose:** Concrete callbacks for console logging and CSV export.

**Classes:**
- `ConsoleLogger`: prints tidy metric tables each epoch.
- `CSVExporter`: writes the training/validation curves to disk for later analysis.


---

### `plotting.py`

**Purpose:** Generates loss-curve figures at the end of training via Matplotlib (`LossCurvePlotter`).

---

## Utilities (`src/rcmvae/utils/`)

### `device.py`

**Purpose:** Centralized helpers for probing/initializing the active JAX backend.

**Key functions:**
- `configure_jax_device()`: initializes JAX (with graceful CPU fallback) and caches the detected platform.
- `get_device_info()`: returns `(device_type, device_count)` for status displays.
- `print_device_banner()`: human-friendly summary printed at startup when desired.

---

## Working with the Code

### Common Patterns

**Creating a model:**
```python
from rcmvae.application.model_api import SSVAE
from rcmvae.domain.config import SSVAEConfig

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

# Example: focus on mixture + τ behavior
pytest tests/test_mixture_encoder.py tests/test_mixture_losses.py \
       tests/test_tau_classifier.py tests/test_tau_integration.py
```

### Test Organization (selected examples)

- Mixture & priors: `tests/test_mixture_encoder.py`, `tests/test_mixture_losses.py`,
  `tests/test_prior_abstraction.py`, `tests/test_vamp_prior.py`, `tests/test_geometric_mog_prior.py`
- τ-classifier: `tests/test_tau_classifier.py`, `tests/test_tau_integration.py`, `tests/test_tau_validations.py`
- System/regression: `tests/test_phase1.py`, `tests/test_refactor_safety.py`, `tests/test_refactor_safety_v2.py`,
  `tests/test_backward_compatibility.py`, `tests/test_legacy_checkpoint.py`
- Experiments & infra: `tests/test_experiment_naming.py`, `tests/test_experiment_validation.py`,
  `tests/test_logging_setup.py`

---

## Related Documentation

- **[System Architecture](architecture.md)** - Design patterns and principles
- **[Extending the System](extending.md)** - How-to tutorials
- **[Experimentation Contracts](experimentation_contracts.md)** - Stable run/metric/plot contracts
- **[Conceptual Model](../theory/conceptual_model.md)** - Theoretical foundation
- **[Experiment Guide](../../use_cases/experiments/README.md)** - Primary experimentation workflow
- **[Dashboard Guide](../../use_cases/dashboard/README.md)** - Interactive active learning interface
- Active project spec (decentralized latents): `docs/projects/decentralized_latents/channel_curriculum/README.md`
