# SSVAE Framework Issues - Fix Plan

**Session:** Verification Phase
**Date:** 2025-11-12

## Issues Found

### 1. Loss Plots Missing Validation Data ⚠️ HIGH PRIORITY

**Location:** `use_cases/experiments/src/visualization/plotters.py:19` (plot_loss_comparison)

**Problem:**
- Only plots training metrics: `loss`, `reconstruction_loss`, `kl_loss`, `classification_loss`
- SSVAE trainer provides validation metrics: `val_loss`, `val_reconstruction_loss`, etc.
- Plots should show both train and val on same axes

**Fix:**
```python
# Current: Only plots training
for metric, title in metrics:
    ax.plot(epochs, history[metric], label="Train")

# Fixed: Plot both train and validation
for metric, title in metrics:
    if metric in history:
        ax.plot(epochs, history[metric], label="Train", linewidth=2)
    val_metric = f"val_{metric}"
    if val_metric in history:
        ax.plot(epochs, history[val_metric], label="Val", linewidth=2, linestyle="--")
```

**Impact:** Loss comparison plots will show training vs validation, making it easy to spot overfitting.

---

### 2. Latent Space by Components Shows Fewer Points ⚠️ MEDIUM PRIORITY

**Location:** `use_cases/experiments/src/visualization/plotters.py:272` (plot_latent_by_component)

**Problem:**
- `plot_latent_spaces` (by class): Uses `model.predict_batched(X_data)` → ALL data
- `plot_latent_by_component`: Loads diagnostics `latent.npz` → validation data only
- Diagnostics saved during training only contain validation batches (src/ssvae/diagnostics.py:89-91)
- Result: By-component plot has ~10% of points (validation split)

**Root Cause:**
The diagnostics are computed during training on validation data:
```python
# In trainer.py, diagnostics only see validation batches
for batch in val_loader:
    z, resp = model.encode(batch)
    z_records.append(z)  # Only validation data
    resp_records.append(resp)
```

**Fix Options:**

**Option A: Recompute on full dataset (RECOMMENDED)**
```python
# In plot_latent_by_component, don't use diagnostics
# Instead, call model.predict_batched(X_data, return_mixture=True)
latent, _, _, _, responsibilities, _ = model.predict_batched(X_data, return_mixture=True)
component_assignments = responsibilities.argmax(axis=1)

# Plot all data, not just validation subset
for c in range(n_components):
    mask = component_assignments == c
    ax.scatter(latent[mask, 0], latent[mask, 1], label=f'C{c}')
```

**Option B: Save full dataset diagnostics separately**
- Add method to save diagnostics on arbitrary data
- Call after training with full X_data
- More complex, requires SSVAE changes

**Recommendation:** Option A - Recompute during plotting (simpler, more robust)

**Impact:** By-component latent space will show all data points, matching by-class plot.

---

### 3. Terminal Output Unclear ℹ️ LOW PRIORITY (UX)

**Location:** `src/training/trainer.py` (SSVAE training loop output)

**Problem:**
Column names in training table are unclear:
```
Train.loss_np | Train.rec | Train.kl | Train.cls | Train.con
```

**Issues:**
- `loss_np` - What does `_np` mean? (no priors?)
- `rec` - Abbreviation not immediately clear (reconstruction)
- `kl` - KL divergence of what? (z? c? both?)
- `cls` - Classification?
- `con` - Contrastive?

**Fix:**
Update column headers for clarity:
```python
# Current
Epoch | Train.loss | Train.rec | Train.kl | Train.cls

# Proposed
Epoch | Total Loss | Reconstruction | KL Divergence | Classification
      | Train | Val | Train | Val | Train | Val | Train | Val
```

Or simpler:
```
Epoch | Loss (T/V) | Recon (T/V) | KL (T/V) | Classif (T/V)
```

**Note:** This requires changes in SSVAE `src/training/trainer.py`, not experiment framework.

**Impact:** Easier to understand training progress at a glance.

---

## Implementation Priority

1. **FIRST: Loss plots** - Quick fix, high impact
2. **SECOND: Latent space data** - Moderate effort, fixes visual discrepancy
3. **THIRD: Terminal output** - SSVAE change, can be separate PR

---

## Testing Plan

After fixes:
1. Run experiment with mixture prior (more validation opportunities)
2. Check loss plots show both train/val curves
3. Verify latent space plots have same number of points
4. Check terminal output (if fixed)

---

## Notes

- Issues #1 and #2 are in the **experiment framework** (use_cases/experiments/)
- Issue #3 is in the **SSVAE library** (src/ssvae/)
- Can fix #1 and #2 immediately
- Issue #3 might need separate consideration
