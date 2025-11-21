# Baseline Experiment Analysis: Decentralized Latents

> **Experiment ID**: `prior_mixture__mix10-dec-gbl_head_film-het__20251121_204351`
> **Date**: 2025-11-21
> **Purpose**: Establish baseline behavior for decentralized latent space (Mixture-of-VAEs) before modular refactor.

## 1. Context & Configuration

This experiment attempted to run the target architecture (FiLM + Heteroscedastic) but inadvertently established a **Concatenation + Heteroscedastic** baseline due to a known configuration bug.

### Hyperparameters
- **Prior**: Mixture of Gaussians ($K=10$, fixed $\pi$)
- **Latent Dim**: 2 per component (Total latent capacity = 20 dims)
- **Decoder**: Component-Aware (Concatenation) + Heteroscedastic
  - *Intended*: FiLM + Heteroscedastic
  - *Actual*: Concatenation (due to silent override in `factory.py`)
- **Regularization**: Diversity weight = -30.0
- **Dataset**: MNIST (70k)

## 2. Key Findings

### A. Configuration Anomaly (Confirmed)
The terminal output confirmed the silent override identified in the [Implementation Spec](./implementation_spec.md):
```text
Model Configuration:
  Decoder: Conv, component-aware (embed=32), heteroscedastic (σ ∈ [0.05, 0.5])
```
The system silently fell back to `component-aware` (concatenation) despite the experiment name implying FiLM. This validates the critical need for the **Modular Decoder Refactor** to enable the true target architecture.

### B. Decentralized Latent Behavior
Despite the fallback, the core "Mixture of VAEs" concept **is functioning**:

1.  **Specialization**: Specific channels successfully captured specific digits (e.g., Channel 1 → "1", Channel 5 → "2", Channel 6 → "3").
2.  **Mode Collapse / Garbage Channels**: Several channels (0, 2, 3, 4, 7) failed to specialize, acting as "garbage collectors" with scattered points.
    *   *Implication*: This strongly validates the need for the **Curriculum Training** strategy (gradually unlocking channels) specified in [Design Context](./design_context.md).

### C. Performance
- **Reconstruction**: Blurry but recognizable. The concatenation decoder struggles to sharpen outputs compared to expected FiLM performance.
- **Stability**: Converged (Epoch 78), best validation loss ~17.38.
- **Resources**: Significant memory pressure during visualization (`Allocator ran out of memory`), suggesting optimization needed for 10-channel tracking.

## 3. Visual Evidence

### Latent Space Specialization
![Latent Space by Channel](../../../use_cases/experiments/results/prior_mixture__mix10-dec-gbl_head_film-het__20251121_204351/figures/mixture/channel_latents/channel_latents_grid.png)
*Shows clear specialization in some channels vs. noise in others.*

### Global Latent Structure
![Global Latent Space](../../../use_cases/experiments/results/prior_mixture__mix10-dec-gbl_head_film-het__20251121_204351/figures/core/latent_spaces.png)
*Shows the aggregated latent structure.*

### Reconstructions
![Reconstructions](../../../use_cases/experiments/results/prior_mixture__mix10-dec-gbl_head_film-het__20251121_204351/figures/core/model_reconstructions.png)
*Baseline reconstruction quality (Concatenation decoder).*

## 4. Implications for Project

1.  **Refactor is Critical**: We cannot test the hypothesis (FiLM > Concat) until the silent override is fixed via the modular refactor.
2.  **Curriculum is Necessary**: The "garbage channel" phenomenon confirms that $K=10$ is too hard to learn simultaneously; the $K_{active} = 1 \to 10$ curriculum is essential.
3.  **Baseline Established**: We now have a concrete performance target (Loss ~17.38) to beat with the Modular FiLM architecture.

---
*Reference*: [Implementation Spec](./implementation_spec.md) | [Design Context](./design_context.md)
