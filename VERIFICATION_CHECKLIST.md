# Verification Test Checklist

Run this checklist after implementing new features to ensure end-to-end functionality.

## Quick Smoke Test

```bash
poetry run python scripts/run_experiment.py --config configs/quick.yaml
```

**Expected results (1-2 minutes):**
- [ ] Training completes without errors
- [ ] Output directory created: `artifacts/experiments/quick_test_YYYYMMDD_HHMMSS/`
- [ ] Files generated (see below)

## Full Mixture Test

```bash
poetry run python scripts/run_experiment.py --config configs/mixture_example.yaml
```

**Expected results (5-10 minutes):**
- [ ] Training completes without errors
- [ ] Output directory created: `artifacts/experiments/mixture_k10_YYYYMMDD_HHMMSS/`
- [ ] All mixture-specific artifacts generated

## Artifact Verification

### Required Files (All Experiments)

- [ ] `config.yaml` - Copy of config used
- [ ] `REPORT.md` - Experiment report
- [ ] `summary.json` - Structured metrics
- [ ] `checkpoint.ckpt` - Model weights
- [ ] `loss_comparison.png` - Loss curves
- [ ] `latent_spaces.png` - Latent by class
- [ ] `{model}_reconstructions.png` - Reconstructions

### Mixture-Specific Files (mixture_example.yaml)

- [ ] `latent_by_component.png` - Latent by component assignment
- [ ] `responsibility_histogram.png` - max_c q(c|x) distribution
- [ ] `visualizations/mixture/model_evolution.png` - π and usage evolution
- [ ] `diagnostics/component_usage.npy` - Final component usage
- [ ] `diagnostics/pi.npy` - Final mixture weights
- [ ] `diagnostics/latent.npz` - Latent data with responsibilities
- [ ] `diagnostics/pi_history.npy` - π evolution over epochs
- [ ] `diagnostics/usage_history.npy` - Usage evolution over epochs
- [ ] `diagnostics/tracked_epochs.npy` - Which epochs were tracked

## Metrics Verification (summary.json)

### Required Sections (All Models)

```json
{
  "training": {
    "final_loss": <number>,
    "final_recon_loss": <number>,
    "final_kl_z": <number>,
    "final_kl_c": <number>,
    "training_time_sec": <number>,
    "epochs_completed": <number>
  },
  "classification": {
    "final_accuracy": <number>,  // NEW!
    "final_classification_loss": <number>
  }
}
```

- [ ] `training` section populated
- [ ] `classification` section populated
- [ ] `final_accuracy` is between 0 and 1

### Mixture-Specific Sections

```json
{
  "mixture": {
    "K": 10,
    "K_eff": <number>,  // NEW! Should be < K
    "active_components": <number>,  // NEW! Components with >1% usage
    "responsibility_confidence_mean": <number>,  // NEW! Mean of max_c q(c|x)
    "final_component_entropy": <number>,
    "final_pi_entropy": <number>,
    "pi_max": <number>,
    "pi_min": <number>,
    "pi_argmax": <number>,
    "pi_values": [...]
  },
  "clustering": {  // NEW! Only for latent_dim=2
    "nmi": <number>,
    "ari": <number>
  }
}
```

- [ ] `mixture` section populated
- [ ] `K_eff` < `K` (effective components < total)
- [ ] `active_components` ≤ `K`
- [ ] `responsibility_confidence_mean` between 0 and 1
- [ ] `clustering` section populated (latent_dim=2 only)
- [ ] `nmi` and `ari` between -1 and 1

## Visualization Verification

### Loss Curves (loss_comparison.png)
- [ ] 4 panels: total loss, reconstruction, KL, classification
- [ ] Training and validation curves visible
- [ ] Axes labeled, legend present

### Latent Spaces (latent_spaces.png)
- [ ] 2D scatter plot
- [ ] Points colored by digit class (0-9)
- [ ] Legend with 10 classes
- [ ] Clear clustering visible

### Latent by Component (latent_by_component.png) - Mixture Only
- [ ] 2D scatter plot
- [ ] Points colored by component assignment (argmax q(c|x))
- [ ] Legend shows components (C0, C1, ...)
- [ ] Compare to latent_spaces.png to see encoder's learned structure

### Responsibility Histogram (responsibility_histogram.png) - Mixture Only
- [ ] Histogram of max_c q(c|x)
- [ ] 50 bins, values between 0 and 1
- [ ] Red dashed line showing mean
- [ ] Mean value displayed in legend

### Mixture Evolution (visualizations/mixture/model_evolution.png) - Mixture Only
- [ ] 2 subplots stacked vertically
- [ ] Top: π evolution over epochs (K lines)
- [ ] Bottom: Component usage evolution over epochs (K lines)
- [ ] X-axis: Epoch numbers
- [ ] Legends with component labels

### Reconstructions
- [ ] 2 rows: original (top), reconstruction (bottom)
- [ ] 8 sample images
- [ ] Clear visual similarity between rows

## Report Verification (REPORT.md)

### Required Sections
- [ ] Experiment metadata (name, description, tags)
- [ ] Configuration (data + model)
- [ ] Summary metrics table (grouped by category)
- [ ] All visualizations embedded with captions
- [ ] Mixture evolution section (if applicable)

### Metrics Table Check
- [ ] Training metrics row
- [ ] Classification metrics row (with accuracy)
- [ ] Mixture metrics row (K_eff, active_components, etc.) - if applicable
- [ ] Clustering metrics row (NMI, ARI) - if latent_dim=2 and mixture

## Regression Indicators

### ⚠️ Potential Issues to Watch For:

1. **Component Collapse**
   - K_eff << K (e.g., K_eff=2 when K=10)
   - Only 1-2 active_components
   - Uniform responsibility distribution (histogram flat)

2. **Training Instability**
   - Loss curves oscillating wildly
   - NaN or Inf values in metrics
   - Extremely high KL values

3. **Poor Clustering**
   - NMI < 0.3 (bad correspondence between components and classes)
   - ARI < 0.1 (poor cluster quality)

4. **Missing Artifacts**
   - Any required file missing
   - Empty or corrupt .npy files
   - Broken image links in REPORT.md

## Success Criteria

**Minimum passing criteria:**
- ✅ All required files present
- ✅ summary.json has all required sections
- ✅ Accuracy > 0.5 (better than random for 10-class MNIST)
- ✅ All visualizations render without errors
- ✅ REPORT.md displays correctly in markdown viewer

**Mixture-specific passing criteria:**
- ✅ K_eff > 3 (at least some component diversity)
- ✅ active_components ≥ K/2 (most components used)
- ✅ responsibility_confidence_mean > 0.4 (encoder somewhat confident)
- ✅ NMI > 0.3, ARI > 0.1 (reasonable clustering)
- ✅ Evolution plots show convergence (not flat lines)

## Quick Commands

```bash
# Run quick test
poetry run python scripts/run_experiment.py --config configs/quick.yaml

# Run full mixture test
poetry run python scripts/run_experiment.py --config configs/mixture_example.yaml

# Check latest results
ls -lht artifacts/experiments/ | head -n 5

# View summary
cat artifacts/experiments/<latest>/summary.json | jq '.'

# View report
cat artifacts/experiments/<latest>/REPORT.md
```

## Notes

- Quick test should complete in 1-2 minutes
- Full mixture test should complete in 5-10 minutes
- If verification fails, check error messages and logs
- Save passing outputs as regression baseline
