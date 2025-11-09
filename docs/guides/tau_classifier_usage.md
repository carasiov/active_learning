# τ-Classifier Usage Guide

## Overview

The τ-classifier is a key component of the RCM-VAE that leverages component specialization for classification. Instead of using a separate MLP head on the latent space, it uses a component→label association matrix (τ) learned from soft counts.

## Mathematical Background

From the Mathematical Specification (§5):

```
Soft counts:    s_{c,y} ← s_{c,y} + q(c|x) · 1{y=y_i}
Normalize:      τ_{c,y} = (s_{c,y} + α_0) / Σ_y' (s_{c,y'} + α_0)
Prediction:     p(y|x) = Σ_c q(c|x) · τ_{c,y}
Loss:           L_sup = -log Σ_c q(c|x) τ_{c,y_true}  [with stop-grad on τ]
```

**Key insight:** Multiple components can serve the same label, specializing by visual features rather than semantic categories.

## Configuration

Enable the τ-classifier in your config:

```python
from ssvae import SSVAEConfig

config = SSVAEConfig(
    prior_type="mixture",          # Required: τ-classifier only works with mixture prior
    num_components=10,             # Number of mixture components
    use_tau_classifier=True,       # Enable τ-classifier
    tau_alpha_0=1.0,               # Dirichlet smoothing parameter
    use_component_aware_decoder=True,  # Recommended for best results
    # ... other settings ...
)
```

## Architecture Changes

When `use_tau_classifier=True`:

1. **Classifier input:** Uses responsibilities `q(c|x)` instead of latent `z`
2. **Classifier output:** Computes `p(y|x) = Σ_c q(c|x) · τ_{c,y}`
3. **Loss computation:** Uses stop-grad on τ (no gradients flow to τ)
4. **τ updates:** Accumulated from soft counts, updated periodically

## Training Loop Integration

The τ matrix must be updated from soft counts during training. Here's the recommended approach:

### Option 1: Update τ Every Epoch (Recommended)

```python
import numpy as np
from ssvae.components.tau_classifier import (
    accumulate_soft_counts,
    update_tau_in_params,
)

# Initialize soft counts
soft_counts = np.zeros((config.num_components, config.num_classes))

# Training loop
for epoch in range(num_epochs):
    epoch_soft_counts = np.zeros((config.num_components, config.num_classes))

    for batch_x, batch_y in train_batches:
        # Forward pass
        output = model.apply(state.params, batch_x, training=True)

        # Extract responsibilities from output
        if output.extras and 'responsibilities' in output.extras:
            responsibilities = output.extras['responsibilities']

            # Accumulate soft counts for labeled samples
            mask = ~np.isnan(batch_y)
            if np.any(mask):
                batch_counts = accumulate_soft_counts(
                    responsibilities[mask],
                    batch_y[mask],
                    config.num_classes,
                )
                epoch_soft_counts += np.array(batch_counts)

        # ... normal training step (gradient update) ...

    # Update τ at end of epoch
    soft_counts += epoch_soft_counts
    new_params = update_tau_in_params(
        state.params,
        soft_counts,
        alpha_0=config.tau_alpha_0,
    )
    state = state.replace(params=new_params)

    print(f"Epoch {epoch}: Updated τ from {np.sum(epoch_soft_counts):.0f} labeled samples")
```

### Option 2: Update τ Every N Batches

For faster adaptation, update τ more frequently:

```python
# Update τ every N batches
update_frequency = 10  # batches
batch_count = 0

for batch_x, batch_y in train_batches:
    # ... forward pass and accumulate counts ...

    batch_count += 1
    if batch_count % update_frequency == 0:
        new_params = update_tau_in_params(
            state.params,
            soft_counts,
            alpha_0=config.tau_alpha_0,
        )
        state = state.replace(params=new_params)
```

## Prediction

When making predictions with a τ-classifier model:

```python
from ssvae.components.tau_classifier import predict_from_tau, extract_tau_from_params

# Get responsibilities from forward pass
output = model.apply(state.params, batch_x, training=False)
responsibilities = output.extras['responsibilities']

# Extract current τ from parameters
tau = extract_tau_from_params(state.params)

# Make predictions
predictions, class_probs = predict_from_tau(responsibilities, tau)

print(f"Predicted classes: {predictions}")
print(f"Class probabilities shape: {class_probs.shape}")  # [batch, num_classes]
```

## OOD Detection

The τ-classifier enables natural OOD detection:

```python
from ssvae.components.tau_classifier import get_ood_score, get_certainty

# Compute OOD scores
ood_scores = get_ood_score(responsibilities, tau)  # Higher = more OOD

# Or get certainty (inverse of OOD score)
certainty = get_certainty(responsibilities, tau)  # Higher = more certain

# Filter OOD samples
ood_threshold = 0.7
ood_mask = ood_scores > ood_threshold
print(f"Detected {np.sum(ood_mask)} OOD samples")
```

## Dynamic Label Addition

Identify free channels for new labels:

```python
from ssvae.components.tau_classifier import get_free_channels

# Compute empirical component usage
usage = np.mean(responsibilities, axis=0)  # Average over dataset

# Find free channels
free_channels = get_free_channels(
    usage,
    tau,
    usage_threshold=1e-3,
    tau_threshold=0.05,
)

print(f"Free channels: {np.where(free_channels)[0]}")
print(f"Available capacity: {np.sum(free_channels)} / {config.num_components}")
```

## Monitoring τ Matrix

Visualize component→label associations:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Extract τ from parameters
tau = extract_tau_from_params(state.params)

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    np.array(tau),
    annot=True,
    fmt='.2f',
    cmap='YlOrRd',
    xticklabels=[f'Class {i}' for i in range(config.num_classes)],
    yticklabels=[f'Comp {i}' for i in range(config.num_components)],
)
plt.title('Component → Label Associations (τ matrix)')
plt.xlabel('Labels')
plt.ylabel('Components')
plt.tight_layout()
plt.savefig('tau_matrix.png')
```

## Expected Results

With a properly configured τ-classifier, you should observe:

1. **Multiple components per label:** Several components may specialize on different visual features within the same class
2. **Improved classification:** Accuracy should match or exceed z-based classifier
3. **Better uncertainty:** OOD detection via component-label confidence
4. **Interpretable specialization:** Components group by visual similarity, not just labels

Example for MNIST digit "0":
- Component 3: Thin, vertical "0"s
- Component 7: Thick, rounded "0"s
- Component 9: Oval-shaped "0"s

All three have high τ_{c,0} but specialize by visual features.

## Troubleshooting

### Issue: τ matrix stays uniform

**Cause:** Soft counts not being accumulated or updated

**Solution:**
- Verify `use_tau_classifier=True` in config
- Check that responsibilities are being extracted correctly
- Ensure `update_tau_in_params()` is called regularly

### Issue: Classification accuracy drops

**Cause:** τ not updated frequently enough early in training

**Solution:**
- Update τ more frequently (every N batches instead of every epoch)
- Initialize with more smoothing (higher `tau_alpha_0`)
- Ensure component-aware decoder is enabled

### Issue: All components collapse to one label

**Cause:** Insufficient component diversity

**Solution:**
- Enable diversity regularization: `component_diversity_weight=-0.05` (negative encourages diversity)
- Increase number of components
- Use KL_c annealing: `kl_c_anneal_epochs=10`

## Implementation Details

### File Structure

- `src/ssvae/components/tau_classifier.py` - TauClassifier module and utility functions
- `src/ssvae/network.py` - Network integration (lines 174-182)
- `src/ssvae/config.py` - Configuration options (lines 161-162, 198-202)
- `src/ssvae/components/factory.py` - Classifier factory (lines 96-125)

### Key Functions

- `TauClassifier.__call__()` - Forward pass with stop-grad on τ
- `accumulate_soft_counts()` - Compute batch count increments
- `compute_tau_from_counts()` - Normalize counts to τ matrix
- `update_tau_in_params()` - Update τ in model parameters
- `predict_from_tau()` - Make predictions using τ
- `get_ood_score()` - Compute OOD confidence
- `get_free_channels()` - Find available components for new labels

## References

- [Mathematical Specification](../theory/mathematical_specification.md) §5 - τ-classifier derivation
- [Conceptual Model](../theory/conceptual_model.md) - High-level intuition
- [Implementation Roadmap](../theory/implementation_roadmap.md) - Development status
- [Theory-to-Code Mapping](../analysis/theory_to_code_mapping.md) - Detailed analysis
