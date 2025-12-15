# Experiment Guide

**Primary tool for training and evaluating SSVAE models.**

> **New location:** The experimentation toolkit now lives under `use_cases/experiments/` alongside the dashboard so all user-facing workflows share a common root.
>
> **Implementation vs. results:** CLI and orchestration code lives under `use_cases/experiments/src/` and `src/rcmvae/adapters/experiments/`, while shared metrics/visualization registries live under `src/infrastructure/`. Generated artifacts for each run land in `use_cases/experiments/results/`.

This guide covers the practical workflow: edit config → run experiment → interpret results.

For a more “engineering loop” view (what signals we rely on, which plots matter, how to validate changes), see:
- Stable contracts: `docs/development/experimentation_contracts.md`
- Practical iteration: `use_cases/experiments/iteration_runbook.md`

---

## Run Directory Layout

Each run uses the standardized [RunPaths](src/infrastructure/runpaths/structure.py) schema so downstream tools know where to look:

```
use_cases/experiments/results/<name>_<timestamp>/
├── config.yaml                # Snapshot of experiment config
├── summary.json               # Structured metrics
├── REPORT.md                  # Human-readable report (links into figures/)
├── artifacts/                 # Checkpoints + diagnostics
├── figures/                   # All plots (loss curves, latents, mixture/τ/uncertainty, ...)
└── logs/                      # Reserved for log files
```

The layout lives in `src/infrastructure/runpaths/structure.py`, so any new tooling can reuse the same helper.

### Channel-Wise Latent Diagnostic

Mixture runs now emit `figures/mixture/channel_latents/channel_XX.png` plus a grid view. Each scatter fixes (x, y) to the true 2-D latent, colors points by the ground-truth label palette, and sets opacity to that channel’s responsibility (`r_c(x)`). This directly checks the conceptual invariant from `docs/theory/conceptual_model.md` §How-We-Classify: channels should specialize so confident regions glow in a single label color while diffuse/multi-label regions fade out. Use the grid for a global sanity check and individual files to deep-dive into problematic channels called out in review discussions.

---

## Configuration

### Config Structure

```yaml
experiment:
  name: "my_experiment"           # Short identifier
  description: "Testing..."       # What you're investigating
  tags: ["baseline", "debug"]     # For organizing results

data:
  num_samples: 10000              # Total dataset size
  num_labeled: 100                # Labeled samples for classification
  seed: 42                        # Random seed for data sampling
  dataset_variant: "mnist"       # "mnist" (full MNIST) or "digits" (sklearn fallback)

model:
  # ──────────────────────────────────────────────────────────────
  # Core Architecture
  # ──────────────────────────────────────────────────────────────
  num_classes: 10                 # Output classes (10 for MNIST)
  latent_dim: 2                   # Latent space dimensions
  
  encoder_type: "conv"            # "conv" for images, "dense" for tabular
  decoder_type: "conv"            # Should match encoder_type
  classifier_type: "dense"        # Classifier architecture
  
  hidden_dims: [256, 128, 64]     # Layer sizes (ignored for conv)

  # ──────────────────────────────────────────────────────────────
  # Loss Configuration
  # ──────────────────────────────────────────────────────────────
  reconstruction_loss: "bce"      # "bce" for MNIST, "mse" for natural images
  recon_weight: 1.0               # Reconstruction loss weight
  kl_weight: 1.0                  # KL divergence weight (VAE β)
  label_weight: 1.0               # Classification loss weight

  # ──────────────────────────────────────────────────────────────
  # Training Configuration
  # ──────────────────────────────────────────────────────────────
  learning_rate: 0.001            # Adam optimizer learning rate
  batch_size: 128                 # Samples per batch
  max_epochs: 100                 # Maximum training epochs
  patience: 20                    # Early stopping patience
  val_split: 0.1                  # Validation set fraction
  
  random_seed: 42                 # Random seed for reproducibility
  grad_clip_norm: 5.0             # Gradient clipping (null to disable)
  weight_decay: 0.0001            # L2 regularization
  dropout_rate: 0.2               # Dropout in classifier
  
  monitor_metric: "loss"          # Early stopping metric: "loss", "val_loss", "classification_loss"

  # ──────────────────────────────────────────────────────────────
  # Prior Configuration
  # ──────────────────────────────────────────────────────────────
  prior_type: "standard"          # "standard", "mixture", "vamp", "geometric_mog"
  num_components: 10              # Number of mixture components (K)

  # Decoder Architecture (mixture/vamp/geometric_mog only)
  use_component_aware_decoder: false  # Decoder variant:
                                    # - false: concat [z, e_c] (shared weights)
                                    # - true: separate Dense(z) and Dense(e_c) pathways
  component_embedding_dim: 8      # Embedding size for component conditioning

  # VampPrior-specific settings (prior_type="vamp" only)
  vamp_pseudo_init_method: "kmeans"  # "kmeans" or "random"
  vamp_num_samples_kl: 1          # Monte Carlo samples for KL (1, 5, or 10)
  vamp_pseudo_lr_scale: 0.1       # Learning rate multiplier for pseudo-inputs

  # Geometric MoG settings (prior_type="geometric_mog" only)
  geometric_arrangement: "grid"   # "circle" or "grid"
  geometric_radius: 2.0           # Spatial radius

  # Mixture/VampPrior regularization
  kl_c_weight: 1.0                # Component assignment KL weight
  kl_c_anneal_epochs: 50          # Anneal kl_c_weight from 0→1 over N epochs
  
  component_diversity_weight: -0.10  # Entropy reward (negative = encourage diversity)
  
  learnable_pi: false             # Learn mixture weights π
  dirichlet_alpha: null           # Dirichlet prior strength (null = disabled)
  dirichlet_weight: 0.05          # Dirichlet regularization weight

  # ──────────────────────────────────────────────────────────────
  # Classification Strategy
  # ──────────────────────────────────────────────────────────────
  use_tau_classifier: false       # Use τ-based latent-only classification
  tau_smoothing_alpha: 1.0        # Laplace smoothing for τ counts

  # ──────────────────────────────────────────────────────────────
  # Uncertainty Estimation
  # ──────────────────────────────────────────────────────────────
  use_heteroscedastic_decoder: false  # Learn per-image variance
  sigma_min: 0.05                 # Minimum variance
  sigma_max: 0.5                  # Maximum variance

  # ──────────────────────────────────────────────────────────────
  # Advanced Options
  # ──────────────────────────────────────────────────────────────
  top_m_gating: 0                 # Use only top-M components (0 = use all)
  soft_embedding_warmup_epochs: 10  # Soft-weighted embeddings warmup
  
  use_contrastive: false          # Enable contrastive loss
  contrastive_weight: 0.0         # Contrastive loss weight
  
  mixture_history_log_every: 1    # Log π and usage every N epochs
```

**For detailed parameter descriptions**, see annotated configs in `configs/`.

**For model architecture theory**, see `/docs/theory/`.

**New:** `configs/decentralized_mixture.yaml` shows the decentralized Mixture-of-VAEs setup
(one latent per component, Gumbel-Softmax routing) ready for MNIST-sized runs.

> **Note:** When `use_tau_classifier: true`, the config validator enforces `num_components >= num_classes`. During training you'll also see an INFO block summarizing the labeled-data regime (zero / few-shot / low-data / standard) plus warnings if there are too few or no labeled samples—these guardrails help interpret τ-classifier results.

---

## Understanding Output

### Directory Structure

The logical structure mirrors `RunPaths` from `src/infrastructure/runpaths/structure.py`:

```
use_cases/experiments/results/<name>_<timestamp>/
├── config.yaml
├── summary.json
├── REPORT.md
├── artifacts/
│   ├── checkpoints/
│   │   └── checkpoint.ckpt                  # Trained model weights
│   └── diagnostics/
│       ├── latent.npz                       # z_mean, labels, q_c (if mixture)
│       ├── pi.npy
│       ├── component_usage.npy
│       ├── component_entropy.npy
│       ├── pi_history.npy
│       ├── usage_history.npy
│       └── tracked_epochs.npy
├── figures/
│   ├── core/
│   │   ├── loss_comparison.png              # Training/validation curves
│   │   ├── latent_spaces.png                # Latent space by class
│   │   └── reconstructions.png              # Input/output samples
│   ├── mixture/
│   │   ├── latent_by_component.png          # Latent space colored by component
│   │   ├── responsibility_histogram.png     # max_c q(c|x) distribution
│   │   ├── *_evolution.png                  # π and usage over time
│   │   └── channel_latents/
│   │       ├── channel_latents_grid.png     # All channels in a grid
│   │       └── channel_XX.png               # Per-channel lens (color=label, alpha=resp)
│   └── tau/
│       ├── tau_matrix_heatmap.png
│       ├── tau_per_class_accuracy.png
│       └── tau_certainty_analysis.png
└── logs/
    └── (reserved for log files / CSV histories)
```

### Key Files

**`REPORT.md`** - Start here! Human-readable summary with embedded visualizations.

**`summary.json`** - Structured metrics for programmatic analysis

**`checkpoint.ckpt`** - Model weights for inference or resuming training.

---

## Module Layout & Extension Points

The experimentation toolkit is thin glue on top of the core model and infrastructure:

| Module | Purpose | Typical changes |
|--------|---------|-----------------|
| `use_cases/experiments/run_experiment.py` | CLI entrypoint: argument parsing, config loading/validation, run directory creation | Add CLI flags, new validation modes, or metadata in the header |
| `use_cases/experiments/src/config.py` | YAML loading and metadata augmentation | Rarely changed; extend if you add top-level config sections |
| `use_cases/experiments/src/data.py` | MNIST loading, sub-sampling, semi-supervised label assignment | Change dataset size, labeling regime, or dataset_variant behavior |
| `src/rcmvae/adapters/experiments/runner.py` | Connects `SSVAEConfig` + `ExperimentService` to metrics and visualization | Change overall training pipeline behavior, add new summary metadata |
| `src/infrastructure/metrics/` | Metric registry + default providers | Register new providers with `@register_metric` in `providers/defaults.py` |
| `src/infrastructure/visualization/` | Plot registry + concrete plotters | Register new plotters with `@register_plotter` in `visualization/plotters.py` |

### Adding a Metric

1. Implement a function that accepts `src.metrics.registry.MetricContext`.
2. Decorate it with `@register_metric`.
3. Return a nested dict fragment; it will be deep-merged into `summary.json`.

`MetricContext` already exposes the trained model, histories, latent samples, π/responsibilities, and diagnostics directory so you can compute arbitrary aggregates.

### Adding a Plot

1. Implement a function that accepts `src.visualization.registry.VisualizationContext`.
2. Decorate it with `@register_plotter`.
3. Save artifacts under `context.figures_dir`; return any metadata (e.g., reconstruction filenames) that the report should mention.

Plotters run sequentially after training, so new visualizations can be dropped in without touching the CLI or pipeline.

---

### 1. Example: Train a Default Model
The default model is the current default configuration from experimentation. This has no further meaning but just reflects the current configuration for experimentation. 

```bash
poetry run python use_cases/experiments/run_experiment.py --config use_cases/experiments/configs/default.yaml
```

**Expect:**
- Runtime: ~2 minutes
- Accuracy: Low (semi-supervised with minimal labels)
- `latent_spaces.png`: Some cluster structure visible

**Use for:** Reference point before experimenting with mixture priors.

---

### 2. Example: How to Create Custom Experiment

```bash
# Run
poetry run python use_cases/experiments/run_experiment.py --config use_cases/experiments/configs/my_experiment.yaml

# Compare results
# Check use_cases/experiments/results/ for outputs
```
---

## Troubleshooting

### JAX Device Errors

**Error:** `No GPU/TPU found`

**Solution:** Use CPU mode:
```bash
JAX_PLATFORMS=cpu poetry run python use_cases/experiments/run_experiment.py --config <config>
```

---

### Out of Memory (OOM)

**Error:** `CUDA out of memory` or similar

**Solutions:**
1. Reduce batch size: `batch_size: 64` → `batch_size: 32`
2. Reduce samples: `num_samples: 10000` → `num_samples: 5000`
3. Use CPU mode: `JAX_PLATFORMS=cpu`

---

### Evolution Plots Missing

**Expected files not created:**
- `pi_history.npy`
- `usage_history.npy`
- `visualizations/mixture/model_evolution.png`

**Solution:** Verify mixture config has:
```yaml
model:
  prior_type: "mixture"
  mixture_history_log_every: 1
```

---

### Slow Training

**Solutions:**
1. Use GPU (remove `JAX_PLATFORMS=cpu`)
2. Increase batch size: `batch_size: 256`
3. Reduce dataset: `num_samples: 5000`
4. Use quick config for testing

---

## Advanced Topics


### History Tracking Configuration

Control evolution plot granularity:

```yaml
model:
  mixture_history_log_every: 5  # Track every 5 epochs (faster, less storage)
```

**Trade-offs:**
- `log_every: 1` - Detailed evolution plots, more I/O overhead
- `log_every: 10` - Coarse evolution plots, minimal overhead

---


## Related Documentation

- **Theory & Architecture:** `docs/theory/` - Mathematical specification and design vision
- **Development Docs:** `docs/development/` - Architecture, implementation, and extension guides
- **Experiment CLI Guide:** `use_cases/experiments/README.md` (this file)
- **Dashboard:** `use_cases/dashboard/README.md` - Web UI for browsing and steering experiments
