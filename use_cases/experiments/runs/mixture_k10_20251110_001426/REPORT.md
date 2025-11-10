# Experiment Report

**Experiment:** mixture_k10

**Description:** Mixture VAE with K=10 components, usage sparsity, and full tracking

**Tags:** mixture, k10, 2d-latent

**Generated:** 20251110_001426

## Configuration

### Data

- num_samples: 5000
- num_labeled: 50
- seed: 42

### Model

- Prior: mixture
- Latent dim: 2
- Hidden dims: (256, 128, 64)
- Components (K): 10
- Reconstruction loss: bce
- Learning rate: 0.001
- Batch size: 128
- Max epochs: 50

## Results

### Summary Metrics

| Category | Metric | Value |
|----------|--------|-------|
| Training | Loss | 228.3712 |
| Training | Recon Loss | 132.5337 |
| Training | Kl Z | 3.4228 |
| Training | Kl C | 0.0021 |
| Training | Training Time Sec | 31.6848 |
| Training | Epochs Completed | 50 |
| Classification | Accuracy | 0.4712 |
| Classification | Classification Loss | 0.3959 |
| Mixture | K | 10 |
| Mixture | Component Entropy | 0.1602 |
| Mixture | Pi Entropy | 2.3026 |
| Mixture | K Eff | 5.8421 |
| Mixture | Active Components | 6 |
| Mixture | Responsibility Confidence Mean | 0.9144 |
| Mixture | Component Majority Labels | [8, 0, 7, 0, 0, 1, 0, 0, 1, 9] |
| Mixture | Component Majority Confidence | [0.47211533784866333, 0.25378841161727905, 0.9701563715934753, 0.032780688256025314, 0.7544224858283997, 0.9905977249145508, 0.9881797432899475, 0.02392127923667431, 0.5195245146751404, 0.5026559233665466] |
| Mixture | Pi Max | 0.1000 |
| Mixture | Pi Min | 0.1000 |
| Mixture | Pi Argmax | 5 |
| Clustering | NMI | 0.7602 |
| Clustering | ARI | 0.0000 |

## Visualizations

### Loss Curves

![Loss Comparison](loss_comparison.png)

### Latent Space

**By Class Label:**

![Latent Spaces](latent_spaces.png)

**By Component Assignment:**

![Latent by Component](latent_by_component.png)

### Responsibility Confidence

Distribution of max_c q(c|x):

![Responsibility Histogram](responsibility_histogram.png)

### Reconstructions

![Reconstructions](model_reconstructions.png)

### Mixture Evolution

![Mixture Evolution](visualizations/mixture/model_evolution.png)

### Component Embedding Divergence

Pairwise distances between learned component embeddings:

![Component Embedding Divergence](component_embedding_divergence.png)

### Reconstruction by Component

How each component reconstructs individual inputs:

![Reconstruction by Component](model_reconstruction_by_component.png)

