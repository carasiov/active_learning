# SSVAE Implementation Guide

This document explains the core `/src` implementation: architecture, APIs, and extension points.

---

## Architecture Overview

The SSVAE follows a modular JAX/Flax design with clear separation between model structure, training logic, and observability.

```
Input (28×28) 
  ↓
Encoder (Dense/Conv) → produces (z_mean, z_log_var) + optional component_logits
  ↓
Latent Space (z) ← sampled via reparameterization trick
  ↓ ↓
  ↓ └→ Classifier → class_logits
  ↓
Decoder (Dense/Conv) → reconstruction
  ↓
Loss = recon_weight × L_recon + kl_weight × L_kl + label_weight × L_class
```

**Key Design Principles:**
- **Configuration-driven**: Architecture determined by `SSVAEConfig` at initialization
- **Pure functions**: Loss computation separated from model forward pass
- **Factory pattern**: Components built via `build_encoder/decoder/classifier`
- **Immutable training state**: JAX functional style with explicit state threading

---

## Core Components

### 1. Model API (`src/ssvae/models.py`)

**`SSVAE`** — Primary public interface

```python
from ssvae import SSVAE, SSVAEConfig

config = SSVAEConfig(latent_dim=2, max_epochs=50)
vae = SSVAE(input_dim=(28, 28), config=config)

# Train with semi-supervised labels (NaN = unlabeled)
labels = np.full(1000, np.nan)
labels[:10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Label 10 samples
history = vae.fit(X_train, labels, "model.ckpt")

# Inference returns: (latent, reconstruction, class_logits, certainty)
z, recon, logits, cert = vae.predict(X_test)
```

**Key Methods:**
- `fit(data, labels, weights_path)` → `HistoryDict`: Train and save best checkpoint
- `predict(data, sample=False, num_samples=1)` → 4-tuple: Inference with optional sampling
- `load_model_weights(weights_path)`: Restore from checkpoint

**Semi-supervised Label Format:**
- Use `np.nan` for unlabeled samples
- Classification loss computed only over labeled indices
- Supports 0% to 100% labeled data

**`SSVAENetwork`** — Flax module (internal)
- Composes encoder, decoder, classifier
- Returns 6-tuple: `(component_logits, z_mean, z_log_var, z, recon, class_logits)`
- Used internally by `SSVAE`, rarely instantiated directly

---

### 2. Configuration (`src/ssvae/config.py`)

**`SSVAEConfig`** — Frozen dataclass controlling all behavior

**Architecture Selection:**
```python
# Dense encoder (fully connected)
SSVAEConfig(encoder_type="dense", hidden_dims=(256, 128, 64))

# Convolutional encoder (for images)
SSVAEConfig(encoder_type="conv", hidden_dims=(32, 64, 128))

# Mixture prior (better clustering)
SSVAEConfig(prior_type="mixture", num_components=10)
```

**Loss Configuration:**
```python
# Binary images (MNIST): BCE with low weight
SSVAEConfig(reconstruction_loss="bce", recon_weight=1.0)

# Natural images: MSE with high weight
SSVAEConfig(reconstruction_loss="mse", recon_weight=500.0)
```

**Critical Parameters:**
- `prior_type`: `"standard"` (Gaussian) or `"mixture"` (mixture of Gaussians)
- `reconstruction_loss`: `"mse"` or `"bce"`
- `recon_weight`, `kl_weight`, `label_weight`: Loss term weights
- `encoder_type`, `decoder_type`: `"dense"` or `"conv"`
- `num_components`: Number of mixture components (only with `prior_type="mixture"`)

**Constraints:**
- Mixture prior requires `encoder_type="dense"`
- `recon_weight` scale depends on `reconstruction_loss` (BCE ~1.0, MSE ~500)
- Architecture changes require new `SSVAE` instance (no dynamic reconfiguration)

---

### 3. Component Factory (`src/ssvae/components/`)

**Pattern:** Configuration → Factory → Flax Module

```python
from ssvae.components import build_encoder, build_decoder, build_classifier

# Factory handles prior type selection and validation
encoder = build_encoder(config, input_hw=(28, 28))
decoder = build_decoder(config, input_hw=(28, 28))
classifier = build_classifier(config)
```

**Available Components:**

**Encoders:**
- `DenseEncoder`: Fully connected, supports standard Gaussian prior
- `MixtureDenseEncoder`: Fully connected, outputs component logits for mixture prior
- `ConvEncoder`: Convolutional (standard prior only)

**Decoders:**
- `DenseDecoder`: Fully connected
- `ConvDecoder`: Transposed convolutions

**Classifier:**
- `Classifier`: Dense layers operating on latent space

**Extending with Custom Architectures:**

1. Create new module in `src/ssvae/components/{encoders,decoders,classifier}.py`
2. Update factory logic in `factory.py`:
   ```python
   def build_encoder(config, *, input_hw=None):
       if config.encoder_type == "custom":
           return CustomEncoder(...)
       # ... existing logic
   ```
3. Add `encoder_type="custom"` to `SSVAEConfig`

**Factory Rationale:**
- Centralizes validation (e.g., mixture + conv = error)
- Handles dimension calculation automatically
- Provides clean extension point without modifying core model

---

### 4. Training Infrastructure (`src/training/`)

**`Trainer`** — Stateless training loop

```python
from training import Trainer

trainer = Trainer(model, config)
history = trainer.train(
    train_data, train_labels,
    val_data, val_labels,
    callbacks=[...]
)
```

**Features:**
- Early stopping via `patience` parameter
- Validation split with `monitor_metric` ("loss" or "val_loss")
- Gradient clipping and weight decay
- Checkpoint saving (best model only)

**`InteractiveTrainer`** — Stateful wrapper for incremental training

```python
from training import InteractiveTrainer

# Preserves optimizer state across calls
interactive = InteractiveTrainer(model, config)
interactive.train_epochs(data, labels, num_epochs=10)  # Train 10 epochs
interactive.train_epochs(data, labels, num_epochs=5)   # Continue for 5 more
```

**Use Case:** Dashboard needs incremental training without resetting optimizer

**`SSVAETrainState`** — Flax TrainState + RNG threading

```python
@dataclass
class SSVAETrainState:
    step: int
    apply_fn: Callable
    params: PyTree
    tx: optax.GradientTransformation
    opt_state: PyTree
    rng: jax.random.PRNGKey  # ← Added for reproducible sampling
```

**RNG Management:**
- Three separate keys: params init, data shuffle, training (sampling)
- Checkpoint includes RNG state for exact reproducibility

---

### 5. Loss Functions (`src/training/losses.py`)

**Pure functions** — Decoupled from model architecture

**`compute_loss_and_metrics`** — Main loss aggregation
```python
total_loss, metrics = compute_loss_and_metrics(
    component_logits, z_mean, z_log_var, z,
    x_recon, x_true, class_logits, labels,
    config
)
```

**Loss Components:**

1. **Reconstruction Loss**
   - `reconstruction_loss_mse`: Mean squared error (continuous values)
   - `reconstruction_loss_bce`: Binary cross-entropy (binary images)
   - Weighted by `config.recon_weight`

2. **KL Divergence**
   - `kl_divergence_standard`: Standard Gaussian prior
   - `kl_divergence_mixture`: Mixture of Gaussians prior
   - Weighted by `config.kl_weight`

3. **Classification Loss**
   - Cross-entropy over labeled samples (mask out NaN labels)
   - Weighted by `config.label_weight`

**MSE vs BCE Tradeoff:**
- **BCE**: Numerically stable for binary data, pixel-wise sum → use `recon_weight=1.0`
- **MSE**: Better for continuous values, pixel-wise mean → use `recon_weight=500.0`
- Scale difference requires different weights for balanced training

**Mixture Prior:**
- Encoder outputs `component_logits` (shape: `[batch, K]`)
- KL computed as: `log p(z|x) - log Σ_k π_k N(z; 0, I)`
- Better latent clustering for multi-modal data
- Only works with dense encoder (conv lacks mixture support)

---

### 6. Callbacks (`src/callbacks/`)

**Observer Pattern** — Decouple training events from I/O

```python
class TrainingCallback:
    def on_train_start(self, state, config): ...
    def on_epoch_end(self, epoch, metrics, state): ...
    def on_train_end(self, history, state): ...
```

**Built-in Callbacks:**
- `ConsoleLogger`: Table-formatted console output
- `CSVExporter`: Write history to CSV
- `LossCurvePlotter`: Matplotlib visualization

**Custom Callback Example:**
```python
from callbacks import TrainingCallback

class WandBLogger(TrainingCallback):
    def on_epoch_end(self, epoch, metrics, state):
        import wandb
        # Convert JAX arrays to Python scalars
        wandb.log({k: float(v) for k, v in metrics.items()})
```

**Integration:**
```python
# Option 1: Pass to Trainer
trainer.train(..., callbacks=[WandBLogger()])

# Option 2: Override SSVAE._build_callbacks
class CustomSSVAE(SSVAE):
    def _build_callbacks(self):
        return super()._build_callbacks() + [WandBLogger()]
```

**Thread Safety:** Callbacks execute sequentially, not thread-safe by default. Dashboard uses background training → ensure callback I/O doesn't block.

---

## Common Patterns

### Pattern 1: Standard Semi-supervised Training
```python
# 1. Prepare data (NaN = unlabeled)
labels = np.full(len(X_train), np.nan)
labels[labeled_indices] = y_train[labeled_indices]

# 2. Configure and train
config = SSVAEConfig(latent_dim=2, max_epochs=50)
vae = SSVAE(input_dim=(28, 28), config=config)
history = vae.fit(X_train, labels, "model.ckpt")

# 3. Inference
z, recon, logits, cert = vae.predict(X_test)
predictions = logits.argmax(axis=1)
```

### Pattern 2: Active Learning Loop
```python
# 1. Train on initial labels
vae = SSVAE(input_dim=(28, 28), config=config)
vae.fit(X_pool, labels, "model.ckpt")

# 2. Find uncertain samples
z, _, logits, cert = vae.predict(X_pool)
uncertain_indices = cert.argsort()[:10]  # 10 most uncertain

# 3. Label uncertain samples
labels[uncertain_indices] = oracle_label(uncertain_indices)

# 4. Retrain (or use InteractiveTrainer for incremental)
vae.fit(X_pool, labels, "model.ckpt")  # Repeat from step 2
```

### Pattern 3: Architecture Comparison
```python
configs = [
    SSVAEConfig(prior_type="standard"),
    SSVAEConfig(prior_type="mixture", num_components=10),
]

for config in configs:
    vae = SSVAE(input_dim=(28, 28), config=config)
    history = vae.fit(X_train, labels, f"model_{config.prior_type}.ckpt")
    # Compare histories...
```

### Pattern 4: Incremental Training (Dashboard)
```python
trainer = InteractiveTrainer(model, config)

# User labels 5 samples → train 10 epochs
trainer.train_epochs(X_pool, labels, num_epochs=10)

# User labels 5 more → continue training
trainer.train_epochs(X_pool, labels, num_epochs=10)

# Optimizer state preserved across calls
```

---

## Technical Details

### Checkpoint Format
```python
# Saved by orbax-checkpoint (Flax's default)
{
    'params': {...},          # Model parameters
    'opt_state': {...},       # Optimizer state (momentum, etc.)
    'rng': PRNGKey,          # RNG state for reproducibility
    'step': int              # Training step count
}
```

### Data Flow Through Training
```
1. Batch sampling: (X, y) ← random shuffle with RNG
2. Forward pass: (component_logits, z_mean, z_log_var, z, recon, logits) ← model(X)
3. Loss computation: total_loss, metrics ← compute_loss_and_metrics(...)
4. Gradient update: state ← optimizer.update(grads, state)
5. Callback hooks: on_epoch_end(epoch, metrics, state)
```

### JAX/Flax Considerations
- **JIT compilation**: `@jax.jit` applied to training step (first epoch slower)
- **GPU memory**: Batch size × image size must fit in VRAM
- **RNG threading**: Explicit PRNG key splitting for reproducibility
- **Immutable state**: Functional updates, no in-place mutations

---

## Extension Points

### 1. Add New Architecture Type
**File:** `src/ssvae/components/{encoders,decoders}.py`

```python
class ResNetEncoder(nn.Module):
    latent_dim: int
    
    @nn.compact
    def __call__(self, x):
        # ... ResNet blocks ...
        z_mean = nn.Dense(self.latent_dim)(h)
        z_log_var = nn.Dense(self.latent_dim)(h)
        return None, z_mean, z_log_var  # component_logits=None for standard prior
```

**Update factory:** Add `encoder_type="resnet"` case in `build_encoder`

### 2. Add New Loss Term
**File:** `src/training/losses.py`

```python
def perceptual_loss(recon, true):
    # ... compute perceptual loss ...
    return loss_value

# In compute_loss_and_metrics:
if config.use_perceptual:
    total_loss += config.perceptual_weight * perceptual_loss(x_recon, x_true)
```

**Update config:** Add `use_perceptual: bool = False` and `perceptual_weight: float = 1.0`

### 3. Add Custom Callback
**File:** `src/callbacks/custom.py`

```python
from callbacks import TrainingCallback

class TensorBoardLogger(TrainingCallback):
    def on_train_start(self, state, config):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter()
    
    def on_epoch_end(self, epoch, metrics, state):
        for k, v in metrics.items():
            self.writer.add_scalar(k, float(v), epoch)
    
    def on_train_end(self, history, state):
        self.writer.close()
```

**Export:** Add to `src/callbacks/__init__.py`

### 4. Add Configuration Preset
**File:** `src/ssvae/config.py`

```python
@staticmethod
def for_cifar10():
    """Preset for CIFAR-10 (32×32 RGB)."""
    return SSVAEConfig(
        input_hw=(32, 32),
        encoder_type="conv",
        hidden_dims=(64, 128, 256),
        reconstruction_loss="mse",
        recon_weight=500.0,
    )
```

**Usage:** `config = SSVAEConfig.for_cifar10()`

---

## Troubleshooting

### "Mixture prior not supported with conv encoder"
- **Cause:** `MixtureConvEncoder` not implemented
- **Fix:** Use `encoder_type="dense"` with `prior_type="mixture"`

### Training loss plateau with BCE
- **Cause:** `recon_weight` too low
- **Fix:** BCE is pixel-wise sum → use `recon_weight=1.0` (not 500)

### Training loss plateau with MSE
- **Cause:** `recon_weight` too high
- **Fix:** MSE is pixel-wise mean → use `recon_weight=500.0` (not 1.0)

### Out of memory on GPU
- **Cause:** Batch size too large
- **Fix:** Reduce `config.batch_size` (default 128 → try 64 or 32)

### Slow first epoch
- **Cause:** JIT compilation overhead
- **Fix:** Expected behavior, subsequent epochs fast

### Classification accuracy stuck at ~10%
- **Cause:** Insufficient labeled samples or `label_weight` too low
- **Fix:** Increase labeled samples or set `label_weight=1000.0`

---

## Summary

**Core abstractions:**
- `SSVAE`: High-level API (fit/predict)
- `SSVAEConfig`: Single source of truth for behavior
- Component factory: Extensible architecture selection
- Pure loss functions: Composable without coupling
- Callback system: Pluggable observability

**Key design decisions:**
- Configuration immutability → architecture changes require rebuild
- Semi-supervised via NaN masking → transparent to loss functions  
- Factory pattern → validate constraints early (e.g., mixture + conv)
- Separate `Trainer` and `InteractiveTrainer` → support both batch and incremental workflows

**When to extend:**
- New architecture type → Add component + factory case
- New loss term → Add function + config flag
- Custom observability → Subclass `TrainingCallback`
- New data modality → Create config preset

For usage examples, see `scripts/compare_models.py` and `use_cases/dashboard/`.
