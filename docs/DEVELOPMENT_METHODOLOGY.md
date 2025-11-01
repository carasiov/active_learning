# Development Methodology for Active Learning SSVAE

## Purpose

This document defines the **incremental development process** for extending the project with features described in CONTEXT.md. It balances rigor with the prototype nature of the project.

---

## Core Principle: Feature → Validate → Integrate

```
1. Minimal Implementation (1-2 weeks)
   └─ Smallest working version that can be tested

2. Isolated Validation (1 week)
   └─ Comparison tool: baseline vs. baseline+feature
   └─ Quantitative + qualitative assessment

3. Ablation Study (optional, 2-3 days)
   └─ Vary feature hyperparameters
   └─ Understand sensitivity

4. Integration (1 week)
   └─ Merge to main if validated
   └─ Update docs and dashboard
   └─ Move to next feature
```

**Timeline per feature:** 2-4 weeks for major features, 1 week for minor additions.

---

## Feature Roadmap (Prioritized)

Based on CONTEXT.md, here's a suggested implementation order:

### **Phase A: Foundation (Highest Impact)**

| Feature | Complexity | Success Metric | Timeline |
|---------|-----------|----------------|----------|
| 1. External Label Store + Dirichlet | Medium | Classification accuracy ≥ baseline | 2 weeks |
| 2. Decoder Variance (μ, σ outputs) | Medium | Reconstruction NLL < MSE loss | 2 weeks |
| 3. OOD/Query Scoring | Low | Visual validation on test set | 1 week |

### **Phase B: Refinement (Medium Impact)**

| Feature | Complexity | Success Metric | Timeline |
|---------|-----------|----------------|----------|
| 4. Sparsity Regularization | Low | <5 active components (of K=10) | 1 week |
| 5. VampPrior | High | Latent space quality > mixture | 3 weeks |
| 6. Contrastive Loss | Medium | Cluster tightness +10-20% | 2 weeks |

### **Phase C: Polish (Nice to Have)**

| Feature | Complexity | Success Metric | Timeline |
|---------|-----------|----------------|----------|
| 7. Training Curriculum (auto phase switching) | Medium | Convergence speed +20% | 2 weeks |
| 8. Pseudo-input Regularization (VampPrior) | Low | If VampPrior implemented | 1 week |

---

## Validation Framework

### Tier 1: Sanity Checks (Required)

Every feature must pass these before being considered functional:

```python
# Training stability
assert not np.isnan(history['total_loss']).any(), "Training diverged"
assert not np.isinf(history['total_loss']).any(), "Numerical overflow"

# Reconstruction quality
final_recon_loss = history['reconstruction_loss'][-1]
assert final_recon_loss < 100.0, "Poor reconstruction (MNIST baseline: ~50)"

# Latent space non-collapse
z_std = np.std(latent_embeddings, axis=0).mean()
assert z_std > 0.1, "Latent collapse (everything mapped to same point)"

# Classification baseline (if labels present)
if num_labeled > 0:
    assert classification_accuracy > 1.0 / num_classes, "Worse than random"
```

### Tier 2: Comparative Metrics (Primary Evaluation)

Compare baseline vs. feature using the comparison tool:

```bash
poetry run python scripts/compare_models.py \
  --config configs/comparisons/feature_test.yaml
```

**Key Metrics:**

| Metric | How to Compute | Success Threshold |
|--------|----------------|-------------------|
| Reconstruction Loss | Final epoch loss | ≤ baseline (or ≤ 1.05× if trade-off) |
| Classification Accuracy | % correct on labeled validation | ≥ baseline - 0.05 |
| Latent Cluster Quality | Silhouette score on true labels | ≥ baseline |
| KL Divergence | Final KL term | Stable (not → 0 or → ∞) |
| Training Time | Wall-clock minutes | < baseline × 2.0 |

**Automated Extraction:**

```python
# In scripts/compare_models.py
def evaluate_feature(baseline_metrics, feature_metrics):
    """Return pass/fail for feature validation."""
    checks = {
        'recon_loss': feature_metrics['recon_loss'] <= baseline_metrics['recon_loss'] * 1.05,
        'accuracy': feature_metrics['accuracy'] >= baseline_metrics['accuracy'] - 0.05,
        'silhouette': feature_metrics['silhouette'] >= baseline_metrics['silhouette'],
        'training_stable': not np.isnan(feature_metrics['loss_curve']).any(),
    }
    return all(checks.values()), checks
```

### Tier 3: Qualitative Assessment (Human Judgment)

Visual inspection using comparison tool outputs:

1. **Latent Space Quality** (`latent_spaces.png`)
   - Are clusters more separated?
   - Are cluster shapes reasonable (not linear/collapsed)?
   - Do class colors form meaningful groups?

2. **Loss Curves** (`loss_comparison.png`)
   - Smooth convergence (no wild oscillations)?
   - Does new feature show clear benefit in any phase?

3. **Uncertainty Scores** (if applicable)
   - Do high-uncertainty points fall on cluster boundaries?
   - Do OOD points get flagged correctly?

**Decision Rule:**
- If Tier 1 fails → debug immediately
- If Tier 2 metrics are worse → investigate why, may need tuning
- If Tier 2 metrics are ≥95% of baseline + Tier 3 looks better → accept feature
- If Tier 2 metrics improve + Tier 3 looks similar → accept feature

---

## Feature-Specific Evaluation

### 1. External Label Store + Dirichlet Smoothing

**Implementation:**
- `src/ssvae/label_store.py` - LabelStore class
- Modify `src/ssvae/models.py` to use store instead of classifier head (optional: keep both, make it a config flag)

**Success Criteria:**

```yaml
# configs/comparisons/label_store.yaml
data:
  num_samples: 10000
  num_labeled: 50  # Very few labels (5 per class)
  epochs: 100

models:
  DirectClassifier:
    use_label_store: false
    label_weight: 1.0

  LabelStore_alpha0.1:
    use_label_store: true
    alpha0: 0.1

  LabelStore_alpha1.0:
    use_label_store: true
    alpha0: 1.0

  LabelStore_alpha10:
    use_label_store: true
    alpha0: 10.0
```

**Expected Results:**
- Classification accuracy ≥ baseline (especially with few labels)
- Ablation over alpha0 shows smooth behavior (not super sensitive)
- Latent space shows cleaner component-label alignment

**Validation Script:**
```bash
poetry run python scripts/compare_models.py \
  --config configs/comparisons/label_store.yaml \
  --output artifacts/comparisons/label_store_$(date +%Y%m%d)
```

---

### 2. Decoder Variance (μ, σ outputs)

**Implementation:**
- Modify decoder to output 2× channels: `(mean, log_var)`
- Replace MSE loss with Gaussian NLL loss:
  ```python
  def gaussian_nll_loss(x, mean, log_var):
      return 0.5 * (log_var + ((x - mean) ** 2) / jnp.exp(log_var))
  ```

**Success Criteria:**

```yaml
# configs/comparisons/decoder_variance.yaml
models:
  MSE_Baseline:
    reconstruction_loss: mse
    decoder_outputs_variance: false

  GaussianNLL:
    reconstruction_loss: gaussian_nll
    decoder_outputs_variance: true
```

**Expected Results:**
- Reconstruction NLL < MSE loss (shows model is capturing uncertainty)
- Latent space has better separation (aleatoric uncertainty no longer pushed to latent)
- Variance map shows high variance on ambiguous regions (e.g., 3 vs. 8 boundary)

**Visual Check:**
Plot decoder variance for test samples:
```python
# Add to comparison_utils.py
def plot_decoder_variance(model, X_test):
    mean, log_var = model.decode(z_test)
    std = np.exp(0.5 * log_var)

    fig, axes = plt.subplots(3, 10)
    for i in range(10):
        axes[0, i].imshow(X_test[i])  # Original
        axes[1, i].imshow(mean[i])     # Reconstruction
        axes[2, i].imshow(std[i])      # Uncertainty (should be high on edges)
```

---

### 3. OOD / Query Scoring

**Implementation:**
- `src/ssvae/uncertainty.py` - compute_ood_score()
- Requires: component responsibilities r_c(x) from encoder
- Requires: label store τ_{c,y}

**Success Criteria:**

```python
# Visual validation (no automated metric yet)
def validate_ood_scoring():
    # Train on MNIST 0-8
    train_data = mnist[mnist_labels < 9]

    # Compute OOD scores on:
    test_normal = mnist[mnist_labels < 9]   # In-distribution
    test_ood = mnist[mnist_labels == 9]     # OOD (never seen "9")

    scores_normal = compute_ood_score(test_normal)
    scores_ood = compute_ood_score(test_ood)

    # Success: OOD scores should be significantly higher
    assert scores_ood.mean() > scores_normal.mean() + 0.2

    # Plot histogram
    plt.hist(scores_normal, alpha=0.5, label='In-distribution')
    plt.hist(scores_ood, alpha=0.5, label='OOD')
    plt.legend()
```

**Expected Results:**
- OOD samples (held-out class) have higher scores than in-distribution
- Boundary samples (e.g., 3 vs. 8) have medium scores
- High-confidence samples have low scores

---

### 4. Sparsity Regularization

**Implementation:**
- Add entropy regularization to loss:
  ```python
  p_c = jnp.mean(component_logits, axis=0)  # Empirical usage
  sparsity_loss = lambda_sparse * jnp.sum(p_c * jnp.log(p_c + 1e-8))
  ```

**Success Criteria:**

```yaml
# configs/comparisons/sparsity.yaml
models:
  NoSparsity:
    use_sparsity: false

  Sparsity_low:
    use_sparsity: true
    lambda_sparse: 0.01

  Sparsity_high:
    use_sparsity: true
    lambda_sparse: 0.1
```

**Expected Results:**
- Active components: K_active < K (e.g., 3-5 of 10 used)
- Reconstruction quality maintained (not degraded by sparsity)
- Component usage plot shows concentration:
  ```python
  p_c = responsibilities.mean(axis=0)
  plt.bar(range(K), p_c)
  plt.title(f"Active components: {(p_c > 0.05).sum()} of {K}")
  ```

---

### 5. VampPrior

**Implementation:**
- High complexity: requires learnable pseudo-inputs
- See [VampPrior paper](https://arxiv.org/abs/1705.07120) for details

**Success Criteria:**

```yaml
# configs/comparisons/vampprior.yaml
models:
  StandardPrior:
    prior_type: standard

  MixturePrior:
    prior_type: mixture
    num_components: 10

  VampPrior:
    prior_type: vamp
    num_pseudoinputs: 50
```

**Expected Results:**
- Latent space quality (silhouette) > mixture prior
- KL divergence converges smoothly (not → 0)
- Pseudo-inputs are diverse (not collapsed)

**Visual Check:**
```python
# Plot pseudo-inputs to verify they're meaningful
fig, axes = plt.subplots(5, 10)
for i, u_k in enumerate(pseudo_inputs[:50]):
    axes.flatten()[i].imshow(u_k.reshape(28, 28))
plt.suptitle("VampPrior Pseudo-Inputs (should look like diverse digits)")
```

---

### 6. Contrastive Loss

**Implementation:**
- Add contrastive term to pull same-class samples together in latent space:
  ```python
  def contrastive_loss(z, labels):
      # For each labeled pair (i, j) with same label:
      # L = ||z_i - z_j||^2  (minimize)
      # For each labeled pair (i, j) with different labels:
      # L = max(0, margin - ||z_i - z_j||^2)  (maximize up to margin)
  ```

**Success Criteria:**

```yaml
# configs/comparisons/contrastive.yaml
data:
  num_labeled: 200  # Need enough labels for pairs

models:
  Baseline:
    use_contrastive: false

  Contrastive_weak:
    use_contrastive: true
    contrastive_weight: 0.1

  Contrastive_strong:
    use_contrastive: true
    contrastive_weight: 1.0
```

**Expected Results:**
- Silhouette score improves by 10-20%
- Within-cluster variance decreases
- Between-cluster distance increases
- Classification accuracy improves (tighter clusters = cleaner boundaries)

---

## Ablation Studies

For each feature, vary its primary hyperparameter to understand sensitivity:

```python
# Example: Label store smoothing
alpha0_values = [0.01, 0.1, 1.0, 10.0, 100.0]

# Expect: smooth trend, not super sensitive
# If very sensitive (e.g., α=1.0 works, α=0.9 fails) → implementation bug
```

**When to run ablations:**
- Feature has configurable hyperparameter
- You're unsure about default value
- You want to demonstrate robustness in paper/report

**When to skip ablations:**
- Feature has obvious default (e.g., enable/disable flag)
- Prototype stage and time-constrained

---

## Documentation Requirements

For each merged feature, update:

1. **IMPLEMENTATION.md** - API changes
   ```markdown
   ### LabelStore

   External label store with Dirichlet smoothing for semi-supervised classification.

   Usage:
   ```python
   config = SSVAEConfig(use_label_store=True, alpha0=1.0)
   ```
   ```

2. **CONTEXT.md** - Mark as implemented
   ```markdown
   ### 3.2 Label Store ✅ **IMPLEMENTED**
   ```

3. **Config example** - Add preset
   ```python
   # src/ssvae/config.py
   @staticmethod
   def label_store_preset():
       return SSVAEConfig(use_label_store=True, alpha0=1.0, ...)
   ```

4. **Test case** - Add regression test
   ```python
   # tests/test_label_store.py
   def test_label_store_accuracy():
       assert accuracy > 0.8
   ```

---

## Anti-Patterns to Avoid

### ❌ Don't: Implement multiple features at once
**Problem:** Can't isolate what caused improvement or regression.

**Instead:** One feature at a time, validate, merge, move on.

---

### ❌ Don't: Optimize hyperparameters excessively
**Problem:** Overfitting to MNIST, wastes time in prototype stage.

**Instead:** Pick reasonable defaults, do 1-2 ablation points, move on.

---

### ❌ Don't: Skip comparison with baseline
**Problem:** No idea if feature helps or hurts.

**Instead:** Always run `compare_models.py` with baseline vs. feature.

---

### ❌ Don't: Implement paper-perfect versions
**Problem:** Takes 3× longer, not needed for prototype.

**Instead:** Minimal working version first, refine if needed.

---

## Example: Full Feature Development Cycle

Let's walk through **decoder variance** end-to-end:

### Week 1: Implementation

```python
# src/ssvae/components/decoders.py
class DenseDecoderWithVariance(nn.Module):
    """Decoder that outputs (mean, log_var) for each pixel."""

    @nn.compact
    def __call__(self, z, training=False):
        x = nn.Dense(128)(z)
        x = nn.relu(x)
        x = nn.Dense(784 * 2)(x)  # 2× output channels

        mean, log_var = jnp.split(x, 2, axis=-1)
        return mean.reshape(-1, 28, 28), log_var.reshape(-1, 28, 28)
```

```python
# src/training/losses.py
def gaussian_nll_loss(x, mean, log_var):
    """Negative log-likelihood for Gaussian decoder."""
    return 0.5 * jnp.mean(log_var + ((x - mean) ** 2) / jnp.exp(log_var))
```

```python
# src/ssvae/config.py
@dataclass
class SSVAEConfig:
    # ...
    reconstruction_loss: str = "mse"  # Options: "mse", "bce", "gaussian_nll"
    decoder_outputs_variance: bool = False
```

### Week 2: Validation

```bash
# Create comparison config
cat > configs/comparisons/decoder_variance.yaml <<EOF
description: "Validate decoder variance modeling"

data:
  num_samples: 10000
  num_labeled: 100
  epochs: 100

models:
  MSE:
    reconstruction_loss: mse
    decoder_outputs_variance: false
    recon_weight: 500

  GaussianNLL:
    reconstruction_loss: gaussian_nll
    decoder_outputs_variance: true
    recon_weight: 1.0
EOF

# Run comparison
poetry run python scripts/compare_models.py \
  --config configs/comparisons/decoder_variance.yaml
```

**Check results:**
```bash
cd artifacts/comparisons/decoder_variance_20251101_143022/

# 1. Sanity check
cat summary.json | jq '.GaussianNLL.final_loss'  # Should be finite

# 2. Metric comparison
cat COMPARISON_REPORT.md  # Read human-friendly summary

# 3. Visual inspection
open latent_spaces.png  # Clusters more separated?
open loss_comparison.png  # Convergence smooth?
```

### Week 3: Refinement (if needed)

If GaussianNLL shows improvement:
```python
# Add visualization of uncertainty maps
def plot_uncertainty(model, X_test):
    mean, log_var = model.predict_with_variance(X_test)
    std = np.exp(0.5 * log_var)

    fig, axes = plt.subplots(10, 3)
    for i in range(10):
        axes[i, 0].imshow(X_test[i], cmap='gray')
        axes[i, 0].set_title('Original')

        axes[i, 1].imshow(mean[i], cmap='gray')
        axes[i, 1].set_title('Reconstruction')

        axes[i, 2].imshow(std[i], cmap='hot')
        axes[i, 2].set_title('Uncertainty')

    plt.savefig('uncertainty_visualization.png')
```

### Week 3-4: Integration

```bash
# Merge to main
git add src/ssvae/components/decoders.py src/training/losses.py
git commit -m "feat: add decoder variance modeling with Gaussian NLL loss"

# Update docs
# - Add to IMPLEMENTATION.md (API reference)
# - Update CONTEXT.md (mark as implemented)
# - Add example to USAGE.md

# Add test
cat > tests/test_decoder_variance.py <<EOF
def test_decoder_variance():
    config = SSVAEConfig(reconstruction_loss='gaussian_nll')
    vae = SSVAE((28, 28), config)
    # ... test that variance output is reasonable
EOF

pytest tests/test_decoder_variance.py
```

### Done! Move to next feature.

---

## Summary: Practical Rules

1. **One feature at a time** - Isolate changes
2. **Use comparison tool** - Your secret weapon for fast validation
3. **Define success upfront** - Write criteria before implementing
4. **Visual + quantitative** - Numbers + human judgment
5. **Minimal first, refine later** - Don't over-engineer
6. **Document as you go** - Future you will thank you
7. **3-tier validation** - Sanity → metrics → qualitative
8. **2-4 weeks per feature** - Don't rush, but don't overthink
9. **Ablate if unsure** - Vary hyperparameters to understand
10. **Keep baseline running** - Always have a reference point

---

## Time Estimates

**Full feature roadmap (CONTEXT.md):**
- Phase A (foundation): 5 weeks
- Phase B (refinement): 6 weeks
- Phase C (polish): 3 weeks
- **Total: ~14 weeks** (3.5 months) at 1 feature at a time

**Prototype-friendly version:**
- Implement Phases A + B only: 11 weeks
- Skip ablations for low-risk features: -2 weeks
- Parallel work on docs: -1 week
- **Realistic: ~8-10 weeks** (2-2.5 months)

This is very reasonable for a prototype research project.

---

## Questions to Ask at Each Stage

**Before implementing:**
- Can this be done in <2 weeks?
- Is there a simpler version I can start with?
- What's the minimal test case?

**During validation:**
- Did metrics improve or stay neutral?
- Is the behavior consistent across runs (same seed)?
- Can I explain why this helps (or doesn't)?

**Before merging:**
- Did I update docs?
- Did I add a test?
- Would someone else understand this code in 6 months?

**After merging:**
- What did I learn?
- What would I do differently next time?
- Is this good enough for the prototype goals?
