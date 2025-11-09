# τ-Classifier Integration Plan for Experiment Framework

**Status**: Ready for implementation
**Created**: 2025-11-09
**Purpose**: Integrate τ-based classifier into the standard `experiments/run_experiment.py` workflow

---

## Executive Summary

The τ-classifier is currently validated in a standalone script (`experiments/validate_tau_classifier.py`). This plan integrates it into the standard experiment framework so you can run τ-classifier experiments with simple YAML configs alongside your existing mixture prior experiments.

**Key Benefits:**
- ✅ Use τ-classifier with familiar `poetry run python experiments/run_experiment.py --config <file>`
- ✅ Automatic τ metrics in `summary.json` and `REPORT.md`
- ✅ τ matrix visualization alongside existing mixture plots
- ✅ Direct comparison with z-based classifier in same report

---

## Current Framework Analysis

### Existing Structure

```
experiments/
├── run_experiment.py          # Main entry point
├── experiment_utils.py        # Visualization utilities
├── configs/                   # YAML configs
│   ├── quick.yaml
│   └── mixture_example.yaml
└── runs/                      # Timestamped outputs
    └── <name>_<timestamp>/
        ├── config.yaml        # Config snapshot
        ├── checkpoint.ckpt    # Model weights
        ├── summary.json       # Structured metrics
        ├── REPORT.md          # Human-readable report
        ├── diagnostics/
        │   └── checkpoint/
        │       ├── pi.npy, usage.npy, etc.
        │       └── (NEW) tau_matrix.npy
        └── visualizations/
            └── mixture/
                ├── model_evolution.png
                └── (NEW) tau_analysis.png
```

### Current Metrics Structure

**summary.json format:**
```json
{
  "training": {...},
  "classification": {...},
  "mixture": {
    "K_eff": 5.6,
    "active_components": 6,
    "responsibility_confidence_mean": 0.92,
    "pi_entropy": 2.30
  },
  "clustering": {...}
}
```

---

## Integration Plan

### Phase 1: Configuration Support ✅ READY

**No changes needed** - τ-classifier params already exist in `SSVAEConfig`:
- `use_tau_classifier: bool = False`
- `tau_alpha_0: float = 1.0`

**Example config to create:**

```yaml
# experiments/configs/tau_classifier_example.yaml
experiment:
  name: "tau_classifier_test"
  description: "Testing τ-based classifier with mixture prior"
  tags: ["tau", "mixture", "k20"]

data:
  num_samples: 5000
  num_labeled: 1000  # τ works best with more labels
  seed: 42

model:
  # Architecture (same as standard mixture)
  encoder_type: "dense"
  decoder_type: "dense"
  latent_dim: 16
  hidden_dims: [256, 128, 64]

  # Loss
  reconstruction_loss: "bce"
  recon_weight: 1.0
  kl_weight: 1.0
  label_weight: 100.0  # Higher weight for classification

  # Mixture prior (REQUIRED for τ-classifier)
  prior_type: "mixture"
  num_components: 20  # More components = more specialization potential
  kl_c_weight: 0.0  # No KL penalty on components
  kl_c_anneal_epochs: 0

  # Component-aware decoder (RECOMMENDED)
  use_component_aware_decoder: true
  component_embedding_dim: 8

  # τ-CLASSIFIER CONFIGURATION (NEW!)
  use_tau_classifier: true    # Enable τ-based classification
  tau_alpha_0: 1.0             # Dirichlet smoothing parameter

  # Regularization (encourage component diversity)
  component_diversity_weight: -0.05  # Negative = encourage diversity
  dirichlet_alpha: 1.0

  # Mixture tracking
  mixture_history_log_every: 1

  # Training
  learning_rate: 0.001
  batch_size: 128
  max_epochs: 50
  patience: 15
  random_seed: 42

  # Note: weight_decay and grad_clip_norm auto-disabled for τ-classifier
  weight_decay: 0.0001  # Factory.py ignores this for τ
  grad_clip_norm: 1.0    # Factory.py ignores this for τ
```

---

### Phase 2: Metrics Collection 🔧 IMPLEMENT

**File**: `experiments/run_experiment.py`, function `train_model()` (lines 86-193)

**Current mixture metrics:**
```python
if config.prior_type == 'mixture':
    mixture_summary = {
        'K': config.num_components,
        'K_eff': mixture_metrics.get('K_eff', 0.0),
        'active_components': mixture_metrics.get('active_components', 0),
        'responsibility_confidence_mean': mixture_metrics.get('responsibility_confidence_mean', 0.0),
        ...
    }
```

**Add τ-specific metrics:**
```python
if config.use_tau_classifier:
    tau_summary = {
        'tau_alpha_0': config.tau_alpha_0,
        'soft_count_total': 0.0,  # Total accumulated soft counts
        'tau_specialization_scores': [],  # Per-component specialization
        'tau_label_coverage': {},  # Labels -> num components specialized
        'tau_dominant_confidences': [],  # Confidence of dominant label per component
    }

    # Load τ matrix and compute metrics
    diag_dir = model.last_diagnostics_dir
    if diag_dir:
        tau_path = Path(diag_dir) / "tau_matrix.npy"
        if tau_path.exists():
            tau = np.load(tau_path)  # Shape: (K, num_classes)

            # Specialization score: entropy of τ[c, :] distribution
            from scipy.stats import entropy
            tau_summary['tau_specialization_scores'] = [
                float(entropy(tau[c, :])) for c in range(tau.shape[0])
            ]

            # Dominant label per component
            dominant_labels = tau.argmax(axis=1)
            dominant_confidences = tau.max(axis=1)
            tau_summary['tau_dominant_labels'] = dominant_labels.tolist()
            tau_summary['tau_dominant_confidences'] = dominant_confidences.tolist()

            # Label coverage: how many components specialize in each label?
            threshold = 0.3  # Component "specializes" if τ[c,y] > threshold
            label_coverage = {}
            for label in range(tau.shape[1]):
                count = int(np.sum(tau[:, label] > threshold))
                label_coverage[int(label)] = count
            tau_summary['tau_label_coverage'] = label_coverage

        # Load soft count total
        soft_count_path = Path(diag_dir) / "soft_count_total.npy"
        if soft_count_path.exists():
            tau_summary['soft_count_total'] = float(np.load(soft_count_path))

    summary['tau'] = tau_summary
```

---

### Phase 3: Visualization Integration 🔧 IMPLEMENT

**File**: `experiments/experiment_utils.py`

**New function to add:**

```python
def plot_tau_analysis(
    models: Dict[str, object],
    output_dir: Path
):
    """Generate τ matrix analysis visualization.

    Creates a heatmap of the τ matrix showing component-to-label associations,
    plus bar charts showing specialization and label coverage.

    Args:
        models: Dictionary of model_name -> model
        output_dir: Directory to save plots
    """
    tau_models = {
        name: model for name, model in models.items()
        if hasattr(model.config, 'use_tau_classifier') and model.config.use_tau_classifier
    }

    if not tau_models:
        return

    import seaborn as sns
    from scipy.stats import entropy

    # Create subdirectory for tau plots
    tau_dir = output_dir / 'visualizations' / 'tau'
    tau_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in tau_models.items():
        diag_dir = model.last_diagnostics_dir
        if not diag_dir:
            continue

        try:
            tau_path = Path(diag_dir) / "tau_matrix.npy"
            if not tau_path.exists():
                continue

            tau = np.load(tau_path)  # Shape: (K, num_classes)
            K, num_classes = tau.shape

            # Create figure with 3 subplots
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])

            # Main heatmap: τ matrix
            ax_main = fig.add_subplot(gs[0, 0])
            sns.heatmap(tau, annot=True, fmt='.2f', cmap='YlOrRd',
                       xticklabels=range(num_classes),
                       yticklabels=[f'C{c}' for c in range(K)],
                       ax=ax_main, cbar_kws={'label': 'τ (probability)'})
            ax_main.set_xlabel('Label')
            ax_main.set_ylabel('Component')
            ax_main.set_title(f'{model_name}: τ Matrix (Component → Label Association)')

            # Right subplot: Specialization scores (entropy)
            ax_right = fig.add_subplot(gs[0, 1])
            specialization = [entropy(tau[c, :]) for c in range(K)]
            ax_right.barh(range(K), specialization, color='steelblue')
            ax_right.set_yticks(range(K))
            ax_right.set_yticklabels([f'C{c}' for c in range(K)])
            ax_right.set_xlabel('Entropy (lower = more specialized)')
            ax_right.set_title('Component Specialization')
            ax_right.grid(True, alpha=0.3)

            # Bottom subplot: Label coverage
            ax_bottom = fig.add_subplot(gs[1, 0])
            threshold = 0.3
            label_coverage = [np.sum(tau[:, label] > threshold) for label in range(num_classes)]
            ax_bottom.bar(range(num_classes), label_coverage, color='coral')
            ax_bottom.set_xlabel('Label')
            ax_bottom.set_ylabel('# Components Specialized')
            ax_bottom.set_title(f'Label Coverage (threshold τ > {threshold})')
            ax_bottom.set_xticks(range(num_classes))
            ax_bottom.grid(True, alpha=0.3)

            # Statistics text box
            ax_stats = fig.add_subplot(gs[1, 1])
            ax_stats.axis('off')

            # Compute statistics
            mean_spec = np.mean(specialization)
            labels_with_zero_coverage = sum(1 for c in label_coverage if c == 0)
            max_tau = np.max(tau)

            stats_text = (
                f"Statistics:\n"
                f"─────────────\n"
                f"Mean specialization: {mean_spec:.3f}\n"
                f"Labels w/ 0 coverage: {labels_with_zero_coverage}\n"
                f"Max τ value: {max_tau:.3f}\n"
                f"α₀ (smoothing): {model.config.tau_alpha_0}\n"
            )
            ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes,
                         fontsize=11, verticalalignment='center',
                         family='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()

            # Save
            safe_name = "".join(c.lower() if c.isalnum() else "_" for c in model_name).strip("_")
            output_path = tau_dir / f'{safe_name}_tau_analysis.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {output_path}")
            plt.close()

        except Exception as e:
            print(f"Warning: Could not plot τ analysis for {model_name}: {e}")
```

**Add to visualization call in `run_experiment.py`:**
```python
def generate_visualizations(model, history, X_train, y_true, output_dir: Path):
    """Generate all visualizations for single model."""
    print("\nGenerating visualizations...")

    models = {'Model': model}
    histories = {'Model': history}

    plot_loss_comparison(histories, output_dir)
    plot_latent_spaces(models, X_train, y_true, output_dir)
    plot_latent_by_component(models, X_train, y_true, output_dir)
    plot_responsibility_histogram(models, output_dir)
    plot_mixture_evolution(models, output_dir)
    plot_tau_analysis(models, output_dir)  # NEW!
    recon_paths = plot_reconstructions(models, X_train, output_dir)
    plot_component_embedding_divergence(models, output_dir)
    plot_reconstruction_by_component(models, X_train, output_dir)

    return recon_paths
```

---

### Phase 4: Report Generation 🔧 IMPLEMENT

**File**: `experiments/run_experiment.py`, function `generate_report()` (lines 218-353)

**Add τ section after mixture section:**

```python
        # τ-classifier metrics (NEW!)
        if 'tau' in summary:
            tau = summary['tau']
            f.write("### τ-Classifier Metrics\n\n")

            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| α₀ (smoothing) | {tau.get('tau_alpha_0', 1.0)} |\n")
            f.write(f"| Soft count total | {tau.get('soft_count_total', 0):.0f} |\n")

            # Mean specialization
            if 'tau_specialization_scores' in tau:
                mean_spec = np.mean(tau['tau_specialization_scores'])
                f.write(f"| Mean specialization entropy | {mean_spec:.3f} |\n")

            # Label coverage
            if 'tau_label_coverage' in tau:
                coverage = tau['tau_label_coverage']
                labels_with_zero = sum(1 for c in coverage.values() if c == 0)
                f.write(f"| Labels with 0 coverage | {labels_with_zero} |\n")

            f.write("\n")

            # Detailed component-label associations
            if 'tau_dominant_labels' in tau and 'tau_dominant_confidences' in tau:
                f.write("**Component Specializations:**\n\n")
                f.write("| Component | Dominant Label | Confidence |\n")
                f.write("|-----------|----------------|------------|\n")

                for c, (label, conf) in enumerate(zip(
                    tau['tau_dominant_labels'],
                    tau['tau_dominant_confidences']
                )):
                    f.write(f"| C{c} | {label} | {conf:.3f} |\n")
                f.write("\n")

        # ... existing visualizations section ...

        # Add τ visualization
        tau_viz_dir = output_dir / 'visualizations' / 'tau'
        if tau_viz_dir.exists():
            tau_plots = list(tau_viz_dir.glob('*_tau_analysis.png'))
            if tau_plots:
                f.write("### τ Matrix Analysis\n\n")
                f.write("Component-to-label association learned from soft counts:\n\n")
                for plot_path in sorted(tau_plots):
                    rel_path = plot_path.relative_to(output_dir)
                    f.write(f"![τ Analysis]({rel_path})\n\n")
```

---

### Phase 5: Diagnostics Persistence 🔧 IMPLEMENT

**File**: `src/ssvae/diagnostics.py`

**Ensure τ matrix is saved alongside other diagnostics:**

```python
# In DiagnosticsCollector.save() method, add:

if self.config.use_tau_classifier:
    # Save τ matrix
    if hasattr(self, '_tau_matrix') and self._tau_matrix is not None:
        tau_path = output_dir / "tau_matrix.npy"
        np.save(tau_path, self._tau_matrix)

    # Save accumulated soft counts total
    if hasattr(self, '_soft_count_total'):
        np.save(output_dir / "soft_count_total.npy", self._soft_count_total)
```

**Access τ from model state:**
```python
# Extract τ from model parameters
if 'classifier' in params and 'tau' in params['classifier']:
    self._tau_matrix = np.array(params['classifier']['tau'])
```

---

## Implementation Checklist

### Core Integration
- [ ] **Config**: Create `experiments/configs/tau_classifier_example.yaml`
- [ ] **Metrics**: Add τ metrics collection to `run_experiment.py::train_model()`
- [ ] **Visualization**: Add `plot_tau_analysis()` to `experiment_utils.py`
- [ ] **Report**: Add τ section to `generate_report()` in `run_experiment.py`
- [ ] **Diagnostics**: Ensure τ matrix saved in `DiagnosticsCollector.save()`

### Testing
- [ ] Run quick test: `poetry run python experiments/run_experiment.py --config experiments/configs/tau_classifier_example.yaml`
- [ ] Verify `summary.json` contains `tau` section
- [ ] Verify `REPORT.md` contains τ analysis
- [ ] Verify `visualizations/tau/` directory created with plots
- [ ] Compare with baseline z-classifier

### Documentation
- [ ] Update `experiments/README.md` with τ-classifier section
- [ ] Add τ-specific metrics to interpretation guide
- [ ] Document τ hyperparameters (tau_alpha_0, num_labeled requirements)

---

## Expected Outcomes

### After Implementation

**Run τ experiment:**
```bash
poetry run python experiments/run_experiment.py --config experiments/configs/tau_classifier_example.yaml
```

**Output structure:**
```
experiments/runs/tau_classifier_test_<timestamp>/
├── REPORT.md               # Now includes τ section
├── summary.json            # Now includes {"tau": {...}}
├── diagnostics/checkpoint/
│   ├── tau_matrix.npy      # NEW: saved τ matrix
│   └── soft_count_total.npy  # NEW: total counts
└── visualizations/
    └── tau/
        └── model_tau_analysis.png  # NEW: τ heatmap + analysis
```

**summary.json example:**
```json
{
  "training": {...},
  "classification": {...},
  "mixture": {...},
  "tau": {
    "tau_alpha_0": 1.0,
    "soft_count_total": 45000,
    "tau_specialization_scores": [0.8, 0.3, ...],
    "tau_dominant_labels": [1, 1, 8, 3, ...],
    "tau_dominant_confidences": [0.95, 0.62, ...],
    "tau_label_coverage": {
      "0": 2,
      "1": 4,
      "7": 0,  // Problem: no components for label 7
      "8": 0
    }
  }
}
```

---

## Usage Guide (Post-Integration)

### Quick Start
```bash
# 1. Create config (or use template)
cp experiments/configs/mixture_example.yaml experiments/configs/my_tau_test.yaml

# 2. Enable τ-classifier in config
#    Add these lines to model section:
#      use_tau_classifier: true
#      tau_alpha_0: 1.0

# 3. Run experiment
poetry run python experiments/run_experiment.py --config experiments/configs/my_tau_test.yaml

# 4. View results
cat experiments/runs/my_tau_test_*/REPORT.md
```

### Comparing τ vs Z Classifiers

Create two configs differing only in `use_tau_classifier`:
```bash
# Run both
poetry run python experiments/run_experiment.py --config experiments/configs/baseline_z.yaml
poetry run python experiments/run_experiment.py --config experiments/configs/test_tau.yaml

# Compare metrics
jq '.classification.final_accuracy' experiments/runs/baseline_z_*/summary.json
jq '.classification.final_accuracy' experiments/runs/test_tau_*/summary.json

# Visual comparison
open experiments/runs/baseline_z_*/latent_spaces.png
open experiments/runs/test_tau_*/latent_spaces.png
```

---

## Performance Tuning Guide

### If τ-Classifier Underperforms

**Symptom**: Low accuracy, many labels with 0 coverage

**Fixes**:
1. **Increase labeled data**: τ learns from soft counts, needs more labels
   ```yaml
   data:
     num_labeled: 2000  # Try 1000-5000 instead of 50-100
   ```

2. **Increase component diversity**:
   ```yaml
   model:
     component_diversity_weight: -0.10  # More negative = more diversity
     dirichlet_alpha: 1.0  # Lower = less prior bias
   ```

3. **Reduce component KL pressure**:
   ```yaml
   model:
     kl_c_weight: 0.0  # Allow components to specialize freely
   ```

4. **Add τ analysis to debug**:
   - Check `tau_label_coverage` in summary.json
   - Look at τ heatmap - are components actually specializing?
   - Check `tau_specialization_scores` - lower is more specialized

---

## Future Enhancements

### Potential Additions
1. **τ evolution tracking**: Save τ matrix at each epoch like π evolution
2. **Multi-model comparison**: Direct τ vs z comparison plots
3. **Soft count visualization**: Show how counts accumulate over training
4. **Label-aware metrics**: Per-label accuracy broken down by component
5. **Auto-tuning**: Suggest optimal `tau_alpha_0` based on label coverage

---

## Summary

This integration plan makes the τ-classifier a **first-class citizen** in your experiment framework:

✅ **Simple to use**: Just add `use_tau_classifier: true` to any mixture config
✅ **Automatic metrics**: τ statistics appear in summary.json and reports
✅ **Rich visualization**: τ heatmap shows component specialization
✅ **Production-ready**: Saves τ matrix for post-hoc analysis

**Next Step**: Implement Phase 2-5 following the code snippets provided above.
