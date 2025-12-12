# Design Context: Decentralized Latent Spaces (Mixture of VAEs)

> **Status**: Active Specification  
> **Source**: Supervisor Meeting Notes (Nov 2025)  
> **Last Verified**: November 2025

---

## 1. Core Concept

**Goal**: Move from a single global latent space to $K$ separate latent spaces ("channels").

- **Structure**: $K$ independent channels, each dimension $d=2$.
- **Behavior**: Class information lives in the channel index $c$. Continuous latents $z_c$ encode intra-class variation.
- **Visualization**: Instead of one cluttered 2D plot, we have $K$ clean 2D plots.

## 2. Architecture Specification

### A. Encoder (Shared Trunk + Multi-Head)

- **Input**: Image $x$.
- **Trunk**: Shared convolutional network $\rightarrow$ Feature Map $F$.
- **Heads**:
    1. **Component Head**: $F \rightarrow$ Softmax $\rightarrow q_\phi(c|x)$ ($K$ probabilities).
    2. **Latent Heads**: $K$ separate heads (or one large projection), each outputting $\mu_c, \sigma_c$ for channel $c$.
- **Output**:
    - Component probabilities: $[B, K]$
    - Latent parameters: $[B, K, D]$ (Means), $[B, K, D]$ (LogVars)

### B. Sampling (Gumbel-Softmax)

- **Discrete $c$**: Sample $c \sim \text{Gumbel-Softmax}(q_\phi(c|x))$ to allow gradients to flow.
- **Continuous $z$**: Sample $z_c \sim \mathcal{N}(\mu_c, \sigma_c)$ for _all_ channels (or just the active one, depending on implementation, but loss implies all).
- **Decoder Input**: The decoder receives the $z_c$ corresponding to the sampled $c$.

### C. Decoder (Conditional Normalization)

- **Mechanism**: "Channel-Conditioned" decoding.
- **Implementation**:
    - Input: $z_{sampled}$ (shape $[B, D]$).
    - Conditioning: The chosen channel $c$ (one-hot or embedding).
    - **FiLM / Conditional Norm**: Convolutional layers are modulated by $\gamma_c, \beta_c$ derived from $c$. $$ \text{Norm}(h, c) = \gamma_c \cdot \frac{h - \mu}{\sigma} + \beta_c $$

## 3. Loss Function

The objective is a sum of terms:

1. **Reconstruction**: $-\log p_\theta(x | z_c, c)$ (using the sampled channel).
2. **Latent KL**: Sum over **all** channels (encourages unused channels to stay prior-like). $$ \sum_{k=1}^K D_{KL}(q_\phi(z_k|x) | \mathcal{N}(0, I)) $$
3. **Component KL/Regularization**:
    - Prior on $q(c|x)$: Dirichlet-like prior to encourage sparsity per sample but usage across batch.
    - Term: $D_{KL}(q_\phi(c|x) | \text{Dirichlet}(\alpha))$ or similar.

---

# Implementation Verification

## Summary

| Category | Status |
|----------|--------|
| **Core Architecture** | ✅ Fully implemented |
| **Encoder (Shared Trunk + Multi-Head)** | ✅ Fully implemented |
| **Gumbel-Softmax Sampling** | ✅ Fully implemented |
| **FiLM Decoder Conditioning** | ⚠️ Scale/add only (normalization unstable) |
| **Latent KL (all K channels)** | ✅ Fully implemented |
| **Component KL** | ✅ Implemented (via categorical KL) |
| **Dirichlet Regularization** | ⚠️ Different formulation (MAP on π) |
| **Class↔Channel Binding** | ⚠️ Optional (τ-classifier disabled by default) |
| **Visualization (K plots)** | ✅ Fully implemented |

---

## Detailed Verification

### Core Concept

| Requirement | Status | Implementation |
|------------|--------|----------------|
| K decentralized latents | ✅ | `latent_layout="decentralized"` validated in [`SSVAEConfig.__post_init__`](../../../src/rcmvae/domain/config.py); per-component latents produced in [`MixtureConvEncoder`](../../../src/rcmvae/domain/components/encoders.py) |
| Latent dimension d=2 | ✅ | Config-driven `latent_dim=2` default; `quick.yaml` uses `latent_dim: 2` |
| Class info in channel index | ⚠️ | Responsibilities q(c\|x) available ([`network.py`](../../../src/rcmvae/domain/network.py)), but classification defaults to separate head. τ-classifier (latent-only) is optional and **disabled by default** in `quick.yaml` (`use_tau_classifier: false`) |
| K clean 2D plots | ✅ | [`plot_channel_latent_responsibility`](../../../src/infrastructure/visualization/mixture/plots.py) generates grid + individual per-channel PNGs |

### Encoder (Shared Trunk + Multi-Head)

| Requirement | Status | Implementation |
|------------|--------|----------------|
| Shared conv trunk | ✅ | [`MixtureConvEncoder.__call__`](../../../src/rcmvae/domain/components/encoders.py): 3 conv layers (32→64→128) then flatten |
| Component head | ✅ | `Dense(num_components)` → `component_logits [B, K]`, softmaxed to responsibilities in [`SSVAENetwork`](../../../src/rcmvae/domain/network.py) |
| Per-channel latent heads | ✅ | When decentralized: `Dense(latent_dim * num_components)` → reshape to `[B, K, D]` for means and log-vars |
| Reparameterized sampling | ✅ | All K channels sampled: `z [B, K, D]` |

### Sampling (Gumbel-Softmax)

| Requirement | Status | Implementation |
|------------|--------|----------------|
| Gumbel-Softmax for c | ✅ | [`_component_selection`](../../../src/rcmvae/domain/network.py): `y_soft = softmax((logits + gumbel_noise) / temp)` |
| Straight-through estimator | ✅ | `y = y_hard - stop_gradient(y_soft) + y_soft` (forward=one-hot, backward=soft) |
| Temperature annealing | ✅ | `gumbel_temperature` → `gumbel_temperature_min` over `gumbel_temperature_anneal_epochs` in [`Trainer.train`](../../../src/rcmvae/application/services/training_service.py) |
| Decoder receives z_c | ✅ | `z = sum(component_selection[..., None] * z_per_component, axis=1)` |

**Note on reconstruction**: With straight-through Gumbel enabled, reconstruction uses the one-hot selected channel. Without it, reconstruction is a soft expectation over all components.

### Decoder (Channel-Conditioned)

| Requirement | Status | Implementation |
|------------|--------|----------------|
| FiLM modulation | ✅ | [`FiLMLayer`](../../../src/rcmvae/domain/components/decoder_modules/conditioning.py): `Dense(2 * feature_dim)` → split to (γ, β) → `features * γ + β` |
| Component embeddings | ✅ | [`MixturePriorParameters`](../../../src/rcmvae/domain/network.py): learned `[K, embed_dim]` embeddings tiled and passed to decoder |
| Modular architecture | ✅ | [`build_decoder`](../../../src/rcmvae/domain/components/factory.py) composes Conditioner + Backbone + OutputHead |
| Heteroscedastic output | ✅ | [`HeteroscedasticHead`](../../../src/rcmvae/domain/components/decoder_modules/outputs.py): returns `(mean, sigma)` with clamped σ |

#### ⚠️ FiLM Normalization Status

The specification shows: $\text{Norm}(h, c) = \gamma_c \cdot \frac{h - \mu}{\sigma} + \beta_c$

**Current implementation**:
- Default path uses **scale/add only**: `h' = γ ⊙ h + β` (stable)
- Optional `normalize=True` branch exists in [`FiLMLayer`](../../../src/rcmvae/domain/components/decoder_modules/conditioning.py) that applies `(h - μ) / σ` before modulation

**Known issue**: The `normalize=True` path causes training instability—KL and responsibilities collapse to ~0, decoder outputs become near-constant, reconstruction overwhelms other losses.

**Potential fixes** (not yet implemented):
1. Use standard LayerNorm/GroupNorm before FiLM (γ init=1, β=0 for identity start).
2. Residual mix between the raw and normalized features.
3. Warm-start: train with scale/add first, then enable normalization with a small mixing weight.

### Loss Function

| Requirement | Status | Implementation |
|------------|--------|----------------|
| Reconstruction (sampled channel) | ✅ | Weighted by `component_selection` in [`MixtureGaussianPrior.compute_reconstruction_loss`](../../../src/rcmvae/domain/priors/mixture.py) |
| Latent KL (all K channels) | ✅ | [`kl_divergence`](../../../src/rcmvae/application/services/loss_pipeline.py) sums over `[B, K, D]` via `sum(kl, axis=(1, 2))` |
| Component KL | ✅ | [`categorical_kl`](../../../src/rcmvae/domain/priors/mixture.py): KL(q(c\|x) ‖ π) |
| Dirichlet regularization | ⚠️ | See note below |

#### ⚠️ Dirichlet Prior Formulation

The specification calls for: $D_{KL}(q(c|x) \| \text{Dirichlet}(\alpha))$

**Current implementation** uses a different but functionally similar approach:
1. **`categorical_kl`**: KL between q(c|x) and π (learned or fixed mixture weights)
2. **`dirichlet_map_penalty`**: MAP penalty on π itself (not on q(c|x))
3. **`usage_sparsity_penalty`**: Entropy reward/penalty on **batch-averaged** component usage (via `component_diversity_weight`)

This achieves the same goals (per-sample sparsity + batch diversity) through different math.

**New option**: A logistic-normal mixture prior on component logits (`c_regularizer="logit_mog"`) replaces the categorical KL when enabled. It pushes q(c|x) toward one-hot vectors via a Gaussian mixture in logit space (means at +M·e_k), leaving the global π MAP penalty unchanged.

---

## Configuration Reference

| Spec Concept | Config Key | Default (`quick.yaml`) |
|-------------|-----------|------------------------|
| K channels | `num_components` | 10 |
| Latent dim d | `latent_dim` | 2 |
| Decentralized mode | `latent_layout` | `"decentralized"` |
| Gumbel-Softmax | `use_gumbel_softmax` | `true` |
| Straight-through | `use_straight_through_gumbel` | `true` |
| Initial temperature | `gumbel_temperature` | 2.0 |
| Min temperature | `gumbel_temperature_min` | 0.5 |
| Anneal epochs | `gumbel_temperature_anneal_epochs` | 100 |
| Decoder conditioning | `decoder_conditioning` | `"cin"` (Conditional Instance Norm) |
| Heteroscedastic | `use_heteroscedastic_decoder` | `true` |
| Component embedding dim | `component_embedding_dim` | 32 |
| KL_c weight | `kl_c_weight` | 0.05 |
| Diversity weight | `component_diversity_weight` | -5.0 (negative = encourage) |
| τ-classifier | `use_tau_classifier` | `false` |

**Decoder Conditioning Options** (`decoder_conditioning`):
- `"cin"`: Conditional Instance Normalization — normalizes features then applies γ/β from component embedding
- `"film"`: FiLM — applies γ/β modulation without normalization
- `"concat"`: Concatenates projected component embedding with features
- `"none"`: No conditioning (for standard prior)

---

## Visualization

All mixture plots saved to `results/<run>/mixture/`:

| Plot | Function | Description |
|------|----------|-------------|
| `channel_latents_grid.png` | [`plot_channel_latent_responsibility`](../../../src/infrastructure/visualization/mixture/plots.py) | **K separate 2D plots** (main deliverable) |
| `channel_{k:02d}.png` | (same) | Individual per-channel PNGs |
| `latent_by_component.png` | [`plot_latent_by_component`](../../../src/infrastructure/visualization/mixture/plots.py) | Single 2D plot colored by component |
| `*_channel_ownership.png` | [`plot_channel_ownership_heatmap`](../../../src/infrastructure/visualization/mixture/plots.py) | Channel vs label ownership matrix |
| `*_component_kl.png` | [`plot_component_kl_heatmap`](../../../src/infrastructure/visualization/mixture/plots.py) | Per-component KL (spot dead channels) |
| `*_routing_hardness.png` | [`plot_routing_hardness`](../../../src/infrastructure/visualization/mixture/plots.py) | Soft vs Gumbel routing comparison |
| `*_recon_selected.png` | [`plot_selected_vs_weighted_reconstruction`](../../../src/infrastructure/visualization/mixture/plots.py) | Selected vs weighted reconstruction |
| `*_reconstruction_by_component.png` | [`plot_reconstruction_by_component`](../../../src/infrastructure/visualization/mixture/plots.py) | Per-component reconstruction |

---

## Key Files

| Component | Location |
|-----------|----------|
| Encoder | [`src/rcmvae/domain/components/encoders.py`](../../../src/rcmvae/domain/components/encoders.py) |
| Decoder | [`src/rcmvae/domain/components/decoders.py`](../../../src/rcmvae/domain/components/decoders.py) |
| FiLM Layer | [`src/rcmvae/domain/components/decoder_modules/conditioning.py`](../../../src/rcmvae/domain/components/decoder_modules/conditioning.py) |
| Output Heads | [`src/rcmvae/domain/components/decoder_modules/outputs.py`](../../../src/rcmvae/domain/components/decoder_modules/outputs.py) |
| Network | [`src/rcmvae/domain/network.py`](../../../src/rcmvae/domain/network.py) |
| Factory | [`src/rcmvae/domain/components/factory.py`](../../../src/rcmvae/domain/components/factory.py) |
| Mixture Prior | [`src/rcmvae/domain/priors/mixture.py`](../../../src/rcmvae/domain/priors/mixture.py) |
| Loss Pipeline | [`src/rcmvae/application/services/loss_pipeline.py`](../../../src/rcmvae/application/services/loss_pipeline.py) |
| Training | [`src/rcmvae/application/services/training_service.py`](../../../src/rcmvae/application/services/training_service.py) |
| Config | [`src/rcmvae/domain/config.py`](../../../src/rcmvae/domain/config.py) |
| Visualizations | [`src/infrastructure/visualization/mixture/plots.py`](../../../src/infrastructure/visualization/mixture/plots.py) |
| Quick Config | [`use_cases/experiments/configs/quick.yaml`](../../../use_cases/experiments/configs/quick.yaml) |

---

## Known Issues & Future Work

### FiLM Normalization (Unstable)

The `normalize=True` option in `FiLMLayer` applies per-sample normalization before γ/β modulation but causes training collapse. Future directions:
- Replace with standard LayerNorm/GroupNorm + FiLM.
- Residual mixing between the raw and normalized features.
- Warm-start training strategy.

### Class↔Channel Alignment

Currently optional via `use_tau_classifier`. When disabled (default), classification uses a separate dense head on pooled latent z, so class information doesn't strictly live in channel index c.

### Dirichlet Prior

Uses MAP penalty on π and batch-averaged entropy, not the exact $D_{KL}(q(c|x) \| \text{Dirichlet})$ from the spec.

### Curriculum Training

Config placeholders exist but channel unlocking is not actively implemented (deferred).
