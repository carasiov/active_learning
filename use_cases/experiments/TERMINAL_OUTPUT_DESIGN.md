# Terminal Output Cleanup - Design Document

**Goal:** Professional, organized console output that's robust and extensible.

## Design Principles

### 1. Progressive Disclosure
- Show summary in console
- Full details in logs only
- User can find details if needed, but console stays clean

### 2. Robustness
- Handle all prior types (standard, mixture, vamp, geometric_mog)
- Handle all classifier modes (τ-classifier, standard classifier)
- Handle all decoder types (vanilla, component-aware, heteroscedastic)
- Gracefully handle missing/optional features

### 3. Extensibility
- Easy to add new configuration types
- Clear interfaces for formatting
- Centralized formatting logic

### 4. Consistency
- Single separator style throughout
- Consistent indentation (2 spaces)
- Aligned columns where appropriate

---

## Output Structure

```
================================================================
Experiment: <name>
================================================================
Architecture:  <architecture_code>
Run ID:        <full_run_id>
Output:        <relative_path>

Data Configuration:
  Dataset:     <dataset_name> (<total> samples, <labeled> labeled)
  Split:       <train_size> train / <val_size> validation

Model Configuration:
  Prior:       <prior_description>
  Encoder:     <encoder_type> (latent_dim=<dim>)
  Decoder:     <decoder_description>
  Classifier:  <classifier_description>

Training Configuration:
  Device:      <device_type> (<count> devices)
  Optimizer:   Adam (lr=<lr>, weight_decay=<wd>)
  Batch size:  <batch>
  Epochs:      <max> (patience=<patience>)
  Monitoring:  <metric>

================================================================
Training Progress
================================================================
<training_table>
```

---

## Component Descriptions (Extensible)

### Prior Types

**Standard:**
```
Prior:       Standard Gaussian N(0, I)
```

**Mixture:**
```
Prior:       Mixture of Gaussians (K=10, learnable π)
             Regularization: Dirichlet(α=0.3), diversity=-0.1
```

**VampPrior:**
```
Prior:       VampPrior (K=20, pseudo-inputs via k-means)
             KL samples: 1, learning rate scale: 0.1
```

**Geometric MoG:**
```
Prior:       Geometric MoG (K=9, grid arrangement)
             Radius: 2.0 (fixed positions)
```

### Classifier Types

**τ-classifier (mixture priors only):**
```
Classifier:  τ-classifier (latent-only, K components → C classes)
             Smoothing: α=1.0
```

**Standard classifier:**
```
Classifier:  Standard (dense, 2 layers)
             Dropout: 0.2
```

### Decoder Types

**Vanilla:**
```
Decoder:     Convolutional
```

**Component-aware:**
```
Decoder:     Convolutional, component-aware (embed_dim=8)
```

**Heteroscedastic:**
```
Decoder:     Convolutional, heteroscedastic (σ ∈ [0.05, 0.5])
```

**Combined:**
```
Decoder:     Convolutional, component-aware, heteroscedastic
             Component embed: 8, σ range: [0.05, 0.5]
```

---

## Implementation Strategy

### Phase 1: Create Formatter Module
**File:** `use_cases/experiments/src/utils/formatters.py`

```python
def format_experiment_header(config, run_id, architecture_code, paths):
    """Format clean experiment header."""

def format_data_config(data_config, split_info):
    """Format data configuration section."""

def format_model_config(model_config):
    """Format model configuration section (robust to all configs)."""

def format_training_config(training_config, device_info):
    """Format training configuration section."""
```

**Key features:**
- Conditional formatting based on config
- Graceful handling of missing keys
- Extensible with new prior/classifier/decoder types

### Phase 2: Update Experiment CLI
**File:** `use_cases/experiments/src/cli/run.py`

```python
# After config loading
print_experiment_header(config, run_id, architecture_code, paths)

# Suppress SSVAE verbose output
# Capture data info for header instead

# After training
print_experiment_footer(summary, paths)
```

### Phase 3: Silence SSVAE Verbose Output
**Files to modify:**
- `src/ssvae/models.py` - Remove model config print
- `src/training/trainer.py` - Remove hyperparameter dump, keep only essential
- Keep all details in experiment.log (for debugging)

### Phase 4: Consolidate Messages
- Remove scattered INFO/Note messages from console
- Keep warnings/errors visible
- Move informational messages to logs

---

## Robustness Checklist

### Prior Type Variations
- [ ] Standard prior
- [ ] Mixture prior (with Dirichlet)
- [ ] Mixture prior (without Dirichlet)
- [ ] VampPrior (k-means init)
- [ ] VampPrior (random init)
- [ ] Geometric MoG (grid)
- [ ] Geometric MoG (circle)

### Classifier Variations
- [ ] τ-classifier (mixture priors)
- [ ] Standard classifier (all priors)
- [ ] With/without dropout

### Decoder Variations
- [ ] Vanilla decoder
- [ ] Component-aware only
- [ ] Heteroscedastic only
- [ ] Component-aware + heteroscedastic
- [ ] Conv vs dense

### Edge Cases
- [ ] No validation split (val_split=0)
- [ ] All unlabeled data (num_labeled=0)
- [ ] Small datasets (< batch_size)
- [ ] CPU training (no GPU)
- [ ] Single device vs multi-device

---

## Extensibility Guidelines

### Adding New Prior Type

1. Add formatting in `format_model_config()`:
```python
elif prior_type == "new_prior":
    prior_desc = f"New Prior (param={value})"
    if optional_feature:
        prior_desc += f"\n             Feature: {feature_value}"
```

2. Works automatically with existing system

### Adding New Classifier Type

1. Add formatting in `format_model_config()`:
```python
elif classifier_type == "new_classifier":
    clf_desc = f"New Classifier ({params})"
```

### Adding New Configuration Section

1. Add formatter function:
```python
def format_new_section(config):
    lines = ["New Section:"]
    lines.append(f"  Param: {config.get('param', 'default')}")
    return "\n".join(lines)
```

2. Call in experiment header

---

## Logging Strategy

### Console Output
- Summary information only
- Training progress table
- Warnings and errors
- Final results summary

### experiment.log
- Full configuration dump
- All hyperparameters
- Detailed training logs
- Diagnostic messages

### training.log
- Training progress (filtered)
- Epoch metrics
- Checkpoint saves

---

## Testing Plan

### Functional Testing
1. Run with default.yaml (mixture prior)
2. Run with standard prior config
3. Run with VampPrior config
4. Run with different data sizes
5. Run on CPU (no GPU)

### Regression Testing
- Ensure all experiments still work
- Verify logs contain full details
- Check backward compatibility

### Visual Testing
- Review console output for clarity
- Verify alignment and spacing
- Check with long experiment names
- Test with very small/large K values

---

## Migration Path

### Phase 1: Add formatters (no breaking changes)
- Create formatting module
- Add tests for formatters
- Verify output matches expectations

### Phase 2: Update experiment CLI (incremental)
- Switch to new header
- Test with existing experiments
- Keep fallback to old format if needed

### Phase 3: Silence SSVAE (coordinated)
- Add quiet mode to SSVAE
- Update experiment CLI to use quiet mode
- Ensure logs capture everything

### Phase 4: Polish (final touches)
- Fine-tune spacing
- Optimize table width
- Final visual review

---

## Success Criteria

✅ Console output fits on standard terminal (80-100 cols)
✅ Key information visible at a glance
✅ Works with all configuration types
✅ Easy to extend for new features
✅ Detailed info preserved in logs
✅ No loss of debugging capability
✅ Professional, clean appearance

---

## Implementation Priority

**HIGH (Do First):**
1. Create formatter module with robust prior/classifier/decoder handling
2. Update experiment CLI to use formatters
3. Test with all prior types

**MEDIUM (Do Second):**
4. Silence SSVAE verbose output
5. Remove trainer hyperparameter dump
6. Consolidate scattered messages

**LOW (Polish):**
7. Fine-tune spacing and alignment
8. Optimize for different terminal widths
9. Add color (optional, future enhancement)
