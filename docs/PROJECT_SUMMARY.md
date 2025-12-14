# Project Summary: RCM-VAE

> **Last verified**: 2025-12-14 against codebase commit 4453ba0

Based on the documentation and codebase, here's where the project stands across three dimensions:

---

## 1. Mathematical Model

**Core idea:** RCM-VAE (Responsibility-Conditioned Mixture VAE) is a semi-supervised generative classifier where *semantics live in latent space*—classification emerges from the model's mixture structure rather than a separate discriminative head.

### Latent Structure
Two latent variables work together:
- **Discrete channel** $c \in \{1, \ldots, K\}$: A "which expert" selector that captures global modes (e.g., digit identity)
- **Continuous latent** $z_c \in \mathbb{R}^d$: Per-channel variation (e.g., stroke thickness, rotation)

Two layout modes:
| Layout | Description |
|--------|-------------|
| **Shared** | Single global $z$; components compete in the same space |
| **Decentralized** (target) | K independent latents $\{z_1, \ldots, z_K\}$; each channel owns its own 2D space |

### Generative Story
$$c \sim \text{Cat}(\pi), \quad z \sim p(z|c) = \mathcal{N}(0, I), \quad x \sim p_\theta(x|z, c)$$

### Objective (Minimization Form)
Per-sample ELBO:
$$\mathcal{L}(x) = \underbrace{-\mathbb{E}_{q(c,z|x)}[\log p_\theta(x|z,c)]}_{\text{Reconstruction}} + \text{KL}_z + \beta_c \cdot \text{KL}(q(c|x) \| \pi)$$

Where $\text{KL}_z$ depends on layout:
- **Shared:** Weighted sum over components
- **Decentralized:** Sum of all K independent KLs: $\sum_k \text{KL}(q(z_k|x) \| p(z_k))$

**Additional terms:**
- Supervised loss via τ-classifier: $-\log \sum_c q(c|x) \cdot \tau_{c,y}$
- Usage sparsity: entropy penalty/reward on empirical channel usage $\hat{p}(c)$
- Optional: Dirichlet MAP on π, **Logit-MoG regularizer** (Gaussian mixture in logit space), contrastive losses

### τ-Classifier (Latent-Only Classification)
Instead of a classifier head, labels propagate through responsibilities:
1. Accumulate soft counts: $s_{c,y} \leftarrow s_{c,y} + q(c|x)$ for labeled $(x, y)$
2. Normalize to get channel→label map: $\tau_{c,y} = \frac{s_{c,y} + \alpha}{\sum_{y'}(s_{c,y'} + \alpha)}$
3. Predict: $p(y|x) = \sum_c q(c|x) \cdot \tau_{c,y}$

**Implementation:** `src/rcmvae/domain/components/tau_classifier.py`

### Prior Options
| Prior | Use Case | Implementation |
|-------|----------|----------------|
| **Standard** | Baseline VAE (no mixture) | `priors/standard.py` |
| **Mixture** | Production semi-supervised learning | `priors/mixture.py` |
| **VampPrior** | Spatial visualization via learned pseudo-inputs | `priors/vamp.py` |
| **Geometric MoG** | Debugging/curriculum (fixed centers) | `priors/geometric_mog.py` |

> **Note on VampPrior:** Current implementation has limitations—does not use component embeddings, π is fixed uniform, and pseudo-inputs may require manual initialization from data. See `vamp.py` header for details.

### Logit-MoG Regularizer
The `c_regularizer: "logit_mog"` option pushes responsibilities toward one-hot via a **mixture of Gaussians in logit space** (not logistic-normal). Each component k has a Gaussian centered at $M \cdot e_k$ where $e_k$ is the k-th standard basis vector:

$$\log p(\text{logits}) = \log \sum_k \frac{1}{K} \mathcal{N}(\text{logits}; M \cdot e_k, \sigma^2 I)$$

**Implementation:** `src/rcmvae/domain/priors/mixture.py:106-124`

---

## 2. Architecture

The codebase follows a **layered, domain-driven design** with JAX/Flax functional patterns.

### High-Level Organization
```
src/rcmvae/
├── domain/           # The math: config, components, priors, network
├── application/      # Orchestration: model API, services, runtime
├── adapters/         # External interfaces (CLI, experiments)
└── utils/            # Device helpers

src/infrastructure/   # Shared tooling for experiments/dashboard
├── metrics/          # Registry + providers for summary.json
├── visualization/    # Registry + plotters
└── runpaths/         # Standardized output directory schema

use_cases/
├── experiments/      # CLI runner, configs, results
└── dashboard/        # Interactive web UI
```

### Design Patterns
| Pattern | Purpose | Location |
|---------|---------|----------|
| **Protocol** | Pluggable priors via `PriorMode` | `domain/priors/base.py` |
| **Factory** | Centralized component creation | `application/services/factory_service.py` |
| **Dataclass Config** | Type-safe configuration (80+ fields) | `domain/config.py` |
| **Composition** | Modular decoders | `domain/components/decoder_modules/` |

### Key Abstractions

**PriorMode Protocol:**
```python
class PriorMode(Protocol):
    def compute_kl_terms(...) -> Dict[str, Array]
    def compute_reconstruction_loss(...) -> Array
    def get_prior_type() -> str
    def requires_component_embeddings() -> bool
```

**Modular Decoder = Conditioner + Backbone + OutputHead**
- **Conditioners:** CIN (Conditional Instance Norm, recommended), FiLM, Concat, None
- **Backbones:** ConvBackbone, DenseBackbone
- **Output heads:** StandardHead, HeteroscedasticHead (mean + σ)

**ModelRuntime:** Single source of truth for network state, optimizer, compiled train/eval functions, and RNG.
```python
@dataclass(frozen=True)
class ModelRuntime:
    network: SSVAENetwork
    state: SSVAETrainState
    train_step_fn: TrainStepFn
    eval_metrics_fn: EvalMetricsFn
    prior: PriorMode
    shuffle_rng: jax.Array
```

### Data Flow (Training)
```
x → Encoder → (q(c|x), {μ_k, σ_k}) → Reparameterize all K z_k
    ↓
    Decode all K with embeddings e_k → {recon_k}
    ↓
    Weight by q(c|x) → final recon
    ↓
    Compute losses (recon + KL_z + KL_c + classification)
```

---

## 3. Experimental Workflow

### Config → Run → Results

**Config structure** (YAML):
```yaml
experiment:
  name: "my_run"
  tags: ["baseline"]

data:
  num_samples: 10000
  num_labeled: 100
  dataset_variant: "mnist"

model:
  # Architecture
  latent_dim: 2
  latent_layout: "decentralized"
  encoder_type: "conv"

  # Prior
  prior_type: "mixture"
  num_components: 10
  decoder_conditioning: "cin"

  # Routing
  use_gumbel_softmax: true
  gumbel_temperature: 2.0

  # Regularization
  c_regularizer: "logit_mog"      # Gaussian mixture on logits
  component_diversity_weight: -3.0  # Negative = encourage diversity

  # Training
  max_epochs: 100
  batch_size: 128
```

### CLI Usage
```bash
# Validate config
poetry run python use_cases/experiments/run_experiment.py --config quick.yaml --validate-only

# Run experiment
poetry run python use_cases/experiments/run_experiment.py --config quick.yaml
```

### Output Structure
```
results/<name>_<timestamp>/
├── config.yaml          # Frozen config
├── summary.json         # Structured metrics
├── REPORT.md            # Human-readable report
├── artifacts/
│   ├── checkpoints/     # Model checkpoints
│   ├── diagnostics/     # latent.npz, pi_history.npy, ...
│   ├── tau/             # τ-classifier artifacts
│   ├── ood/             # OOD scoring data
│   └── uncertainty/     # Heteroscedastic outputs
├── figures/
│   ├── core/            # Loss curves, latent spaces, reconstructions
│   ├── mixture/         # Channel ownership, routing hardness, evolution
│   ├── tau/             # τ matrix heatmap, per-class accuracy
│   ├── uncertainty/     # Variance maps
│   └── ood/             # OOD distributions
└── logs/
```

### Key Metrics & Visualizations

**Metrics (in summary.json):**
- Training: final losses (recon, KL_z, KL_c, kl_c_logit_mog), accuracy
- Component: usage entropy, π values, responsibility distribution
- τ-classifier: per-class accuracy, certainty

**Visualizations:**
| Plot | What it Shows |
|------|---------------|
| `channel_latents_grid.png` | K separate 2D plots—each channel's view of the latent space |
| `channel_ownership_heatmap.png` | Which channels "own" which labels |
| `routing_hardness.png` | Soft vs Gumbel routing comparison |
| `tau_matrix_heatmap.png` | Channel→label map τ |

### Current Default Config (`quick.yaml`)
- **Decentralized layout** with 10 components, latent_dim=2
- **Gumbel-Softmax** routing with straight-through
- **CIN conditioning** on decoder
- **Logit-MoG regularizer**: pushes responsibilities toward one-hot via Gaussian mixture in logit space
- **Heteroscedastic decoder** enabled
- **τ-classifier disabled** (classification via separate head)

---

## Summary Table

| Dimension | Status | Notes |
|-----------|--------|-------|
| **Mixture prior + diversity controls** | ✅ Shipping | `mixture.py`, `component_diversity_weight` |
| **Decentralized latent layout** | ✅ Shipping | `config.latent_layout`, `network.py` |
| **Modular decoder (CIN/FiLM/Concat)** | ✅ Shipping | `decoder_modules/conditioning.py` |
| **Gumbel-Softmax routing** | ✅ Shipping | `network.py:193-226` |
| **τ-classifier** | ✅ Shipping (optional) | `tau_classifier.py` |
| **Heteroscedastic decoder** | ✅ Shipping | `outputs.py:35-75` |
| **VampPrior** | ✅ Shipping | Has caveats—see `vamp.py` header |
| **Logit-MoG regularizer** | ✅ Shipping | `mixture.py:106-124` |
| **OOD scoring** | 📋 Infrastructure ready | `tau_classifier.py:207-220` (not fully wired) |
| **Dynamic label addition** | 📋 Infrastructure ready | `get_free_channels()` exists |

---

## Key Implementation Files

| Concept | Primary File |
|---------|--------------|
| Configuration | `src/rcmvae/domain/config.py` |
| Network forward pass | `src/rcmvae/domain/network.py` |
| Loss computation | `src/rcmvae/application/services/loss_pipeline.py` |
| Prior protocol | `src/rcmvae/domain/priors/base.py` |
| τ-classifier | `src/rcmvae/domain/components/tau_classifier.py` |
| Factory service | `src/rcmvae/application/services/factory_service.py` |
| Decoder conditioning | `src/rcmvae/domain/components/decoder_modules/conditioning.py` |
| Output structure | `src/infrastructure/runpaths/structure.py` |

---

The project is architecturally mature with a clean separation between domain logic, application orchestration, and external adapters. The main experimental focus is on getting the decentralized mixture model to produce clean, interpretable per-channel latent spaces where each channel specializes to a semantic concept.
