# Contributing Guide

Guide for extending and modifying the SSVAE project.

> **Before adding features:** See [Verification Methodology](VERIFICATION.md) for testing and validation process.

---

## Development Workflow

**Philosophy:** Validate functionality quickly through experimentation before integrating into user-facing interfaces.

### Recommended Cycle

```
┌─────────────┐
│ 1. Implement│  Add feature in src/ (encoder, loss, prior, callback)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 2. Validate │  Test with: scripts/compare_models.py
└──────┬──────┘  • Does it converge?
       │         • Better than baseline?
       │         • Expected behavior?
       ▼
┌─────────────┐
│ 3. Document │  Update docs/IMPLEMENTATION.md if API changes
└──────┬──────┘  • Public API modified?
       │         • New extension pattern?
       │         • Config options added?
       ▼
┌─────────────┐
│ 4. Integrate│  Add to dashboard once behavior confirmed
└─────────────┘  • UI controls
                 • Validation logic
                 • User docs
```

### Why This Approach?

✅ **Fast feedback** - `compare_models.py` provides immediate results without UI complexity  
✅ **Regression detection** - Automated comparisons catch unexpected behavior  
✅ **Thorough analysis** - Visualizations and metrics enable deep understanding  
✅ **Easy integration** - Dashboard integration straightforward after validation

---

## Extending the Model

### Adding a New Encoder Type

**Example:** Implementing a ResNet encoder

**Step 1: Implement the encoder**

Create new class in `src/ssvae/components/encoders.py`:

```python
class ResNetEncoder(nn.Module):
    """ResNet-style encoder with skip connections."""
    latent_dim: int
    hidden_dims: Tuple[int, ...] = (64, 128, 256)
    
    @nn.compact
    def __call__(self, x, *, training: bool):
        # Flatten input
        h = x.reshape((x.shape[0], -1))
        
        # Residual blocks
        for dim in self.hidden_dims:
            identity = h
            h = nn.Dense(dim)(h)
            h = nn.relu(h)
            h = nn.Dense(dim)(h)
            if identity.shape[-1] == dim:
                h = h + identity  # Skip connection
            h = nn.relu(h)
        
        # Output heads
        z_mean = nn.Dense(self.latent_dim)(h)
        z_log_var = nn.Dense(self.latent_dim)(h)
        
        return None, z_mean, z_log_var  # component_logits=None for standard prior
```

**Step 2: Update the factory**

Modify `build_encoder()` in `src/ssvae/components/factory.py`:

```python
def build_encoder(config: SSVAEConfig, *, input_hw: Tuple[int, int] | None = None) -> nn.Module:
    # ... existing code ...
    
    if config.encoder_type == "resnet":
        if config.prior_type == "mixture":
            raise ValueError("ResNet encoder does not support mixture prior yet")
        return ResNetEncoder(
            latent_dim=config.latent_dim,
            hidden_dims=config.hidden_dims
        )
    
    # ... rest of function ...
```

**Step 3: Add configuration option**

In `src/ssvae/config.py`, update the docstring:

```python
@dataclass(frozen=True)
class SSVAEConfig:
    # ...
    encoder_type: str = "dense"  # "dense", "conv", or "resnet"
```

**Step 4: Create test configuration**

Create `configs/comparisons/test_resnet.yaml`:

```yaml
description: "Testing ResNet encoder"

data:
  num_samples: 5000
  num_labeled: 50
  epochs: 30

models:
  Dense:
    encoder_type: dense
    hidden_dims: [256, 128, 64]
  
  ResNet:
    encoder_type: resnet
    hidden_dims: [64, 128, 256]
```

**Step 5: Validate**

```bash
# Run comparison
poetry run python scripts/compare_models.py --config configs/comparisons/test_resnet.yaml

# Check results in artifacts/comparisons/latest/
# - Does ResNet converge?
# - How does loss compare to Dense?
# - Is latent space well-structured?
```

**Step 6: Document**

If the API is stable, add to `docs/IMPLEMENTATION.md`:

```markdown
### Available Encoders

- `DenseEncoder`: Fully connected layers
- `ConvEncoder`: Convolutional layers for images
- `ResNetEncoder`: Residual connections for deeper networks
```

**Step 7: Integrate into dashboard (optional)**

Once validated, add to dashboard configuration UI in `use_cases/dashboard/pages/configure_training.py`.

---

### Adding a New Loss Function

**Example:** Implementing perceptual loss

**Step 1: Implement the loss**

Add to `src/training/losses.py`:

```python
def perceptual_loss(
    x_recon: jnp.ndarray,
    x_true: jnp.ndarray,
    feature_extractor_fn: Callable
) -> jnp.ndarray:
    """
    Perceptual loss using pre-trained feature extractor.
    
    Args:
        x_recon: Reconstructed images, shape (batch, H, W)
        x_true: Original images, shape (batch, H, W)
        feature_extractor_fn: Function mapping images to features
    
    Returns:
        Scalar loss value
    """
    features_recon = feature_extractor_fn(x_recon)
    features_true = feature_extractor_fn(x_true)
    return jnp.mean((features_recon - features_true) ** 2)
```

**Step 2: Integrate into total loss**

Modify `compute_loss_and_metrics()` in `src/training/losses.py`:

```python
def compute_loss_and_metrics(...):
    # ... existing losses ...
    
    # Add perceptual loss if enabled
    if config.use_perceptual:
        perc_loss = perceptual_loss(x_recon, x_true, feature_extractor)
        total_loss += config.perceptual_weight * perc_loss
        metrics['perceptual_loss'] = perc_loss
    
    return total_loss, metrics
```

**Step 3: Add config options**

In `src/ssvae/config.py`:

```python
@dataclass(frozen=True)
class SSVAEConfig:
    # ... existing fields ...
    use_perceptual: bool = False
    perceptual_weight: float = 1.0
```

**Step 4: Test**

```yaml
# configs/comparisons/test_perceptual.yaml
models:
  MSE:
    reconstruction_loss: mse
    use_perceptual: false
  
  MSE+Perceptual:
    reconstruction_loss: mse
    use_perceptual: true
    perceptual_weight: 0.1
```

```bash
poetry run python scripts/compare_models.py --config configs/comparisons/test_perceptual.yaml
```

---

### Adding a New Callback

**Example:** TensorBoard logger

**Step 1: Implement callback**

Create `src/callbacks/tensorboard.py`:

```python
from callbacks import TrainingCallback

class TensorBoardLogger(TrainingCallback):
    """Log training metrics to TensorBoard."""
    
    def __init__(self, log_dir: str = "runs"):
        self.log_dir = log_dir
        self.writer = None
    
    def on_train_start(self, state, config):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.log_dir)
        
    def on_epoch_end(self, epoch: int, metrics: dict, state):
        # Convert JAX arrays to Python scalars
        for key, value in metrics.items():
            self.writer.add_scalar(key, float(value), epoch)
    
    def on_train_end(self, history: dict, state):
        if self.writer:
            self.writer.close()
```

**Step 2: Export**

Add to `src/callbacks/__init__.py`:

```python
from .tensorboard import TensorBoardLogger

__all__ = [..., "TensorBoardLogger"]
```

**Step 3: Use in training**

```python
from callbacks import TensorBoardLogger

# Option 1: Pass to Trainer
trainer.train(..., callbacks=[TensorBoardLogger()])

# Option 2: Extend SSVAE._build_callbacks
class CustomSSVAE(SSVAE):
    def _build_callbacks(self, **kwargs):
        return super()._build_callbacks(**kwargs) + [TensorBoardLogger()]
```

---

## Testing

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_model_commands.py

# Dashboard tests
poetry run python tests/run_dashboard_tests.py
```

### Writing Tests

**Model tests:**

```python
def test_new_encoder():
    config = SSVAEConfig(encoder_type="resnet")
    vae = SSVAE(input_dim=(28, 28), config=config)
    
    # Test forward pass
    X = np.random.rand(10, 28, 28).astype(np.float32)
    y = np.full(10, np.nan)
    
    history = vae.fit(X, y, "test.ckpt")
    
    assert "loss" in history
    assert history["loss"][-1] < history["loss"][0]  # Loss decreases
```

**Integration tests:**

Use `compare_models.py` as integration test:

```bash
# This tests the full pipeline
poetry run python scripts/compare_models.py --models standard --epochs 5 --num-samples 500
```

---

## Documentation Updates

### When to Update Documentation

**Update `docs/IMPLEMENTATION.md` when:**
- ✅ Public API changes (new methods, parameters)
- ✅ New component types added (encoder, loss, callback)
- ✅ Extension patterns change
- ✅ Configuration options added

**Update `docs/USAGE.md` when:**
- ✅ New usage patterns emerge
- ✅ Tools gain new features
- ✅ Common workflows change

**Update `docs/GETTING_STARTED.md` when:**
- ✅ Installation requirements change
- ✅ Setup process changes
- ✅ Quick start example needs update

**Update dashboard docs when:**
- ✅ Dashboard features added/changed
- ✅ Extension patterns for dashboard modified

### Documentation Style

- **Be concise:** Focus on essential information
- **Include examples:** Show, don't just tell
- **Link generously:** Reference related docs
- **Keep it practical:** Emphasize common use cases
- **Update cross-references:** Ensure links stay valid

---

## Code Style

### Python Conventions

- **Type hints:** Use throughout (JAX/Flax compatible)
- **Docstrings:** Use for public APIs (NumPy style)
- **Naming:** `snake_case` for functions/variables, `PascalCase` for classes
- **Imports:** Group by stdlib, third-party, local

### JAX/Flax Patterns

- **Pure functions:** Keep training logic functional
- **Immutable state:** Use `dataclasses(frozen=True)`
- **Explicit RNG:** Thread PRNG keys explicitly
- **JIT-friendly:** Avoid Python control flow in JIT-compiled functions

---

## Common Contribution Scenarios

### Scenario 1: Fix a Bug

1. **Reproduce:** Create minimal test case
2. **Fix:** Implement fix in appropriate module
3. **Verify:** Run tests and comparison tool
4. **Document:** Add to troubleshooting if user-facing

### Scenario 2: Add Architecture Variant

1. **Implement:** New component in `src/ssvae/components/`
2. **Factory:** Update `factory.py` with new type
3. **Config:** Add option to `SSVAEConfig`
4. **Test:** Create comparison YAML and validate
5. **Document:** Add to `IMPLEMENTATION.md` if stable

### Scenario 3: Improve Dashboard

1. **Read guides:** Start with `use_cases/dashboard/docs/AGENT_GUIDE.md`
2. **Follow patterns:** Use established command/callback patterns
3. **Test manually:** Verify UI changes work
4. **Consider model impact:** If model changes needed, validate first with comparison tool

### Scenario 4: Add Example/Tutorial

1. **Create notebook:** In `examples/` (create if needed)
2. **Document clearly:** Explain each step
3. **Test thoroughly:** Ensure reproducible
4. **Link from README:** Update documentation map

---

## Getting Help

**Questions about:**
- **Model architecture** → See `docs/IMPLEMENTATION.md`
- **Comparison tool** → See `configs/comparisons/README.md`
- **Dashboard internals** → See `use_cases/dashboard/docs/DEVELOPER_GUIDE.md`
- **Dashboard extensions** → See `use_cases/dashboard/docs/AGENT_GUIDE.md`
- **Setup issues** → See `.devcontainer/README.md` or `docs/GETTING_STARTED.md`

**Still stuck?**
- Check logs (`/tmp/ssvae_dashboard.log` for dashboard)
- Review existing code for patterns
- Create minimal reproduction case
- [Contact maintainers or open issue]

---

## Contribution Checklist

Before submitting changes:

- [ ] Code follows established patterns
- [ ] Tests pass (`poetry run pytest`)
- [ ] New features validated with comparison tool
- [ ] Documentation updated if API changed
- [ ] Cross-references updated if docs restructured
- [ ] Examples work and are reproducible
- [ ] Commit messages are clear and descriptive

---

Thank you for contributing to the SSVAE project! 🎉
