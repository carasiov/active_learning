# Design Context: Decentralized Latent Spaces (Mixture of VAEs)

> **Status**: Active Specification  
> **Source**: Supervisor Meeting Notes (Nov 2025)

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
    - **FiLM / Conditional Norm**: Convolutional layers are modulated by $\gamma_c, \beta_c$ derived from $c$. 
      $$
      \text{Norm}(h, c) = \gamma_c \cdot \frac{h - \mu}{\sigma} + \beta_c
      $$

## 3. Loss Function

The objective is a sum of terms:

1. **Reconstruction**: $-\log p_\theta(x | z_c, c)$ (using the sampled channel).
2. **Latent KL**: Sum over **all** channels (encourages unused channels to stay prior-like). 
   $$
   \sum_{k=1}^K D_{KL}(q_\phi(z_k|x) \| \mathcal{N}(0, I))
   $$
3. **Component KL/Regularization**:
    - Prior on $q(c|x)$: Dirichlet-like prior to encourage sparsity per sample but usage across batch.
    - Term: $D_{KL}(q_\phi(c|x) \| \text{Dirichlet}(\alpha))$ or similar.

## 4. Training Strategy (Curriculum)

To prevent "compromise" solutions:

1. **Phase 1: Reconstruction**: High recon weight, low KL.
2. **Phase 2: KL Annealing**: Ramp up $\beta$ for $z$ and $c$.
3. **Phase 3: Channel Curriculum**:
    - Start with $K_{active}=1$.
    - Gradually unlock more channels ($1 \rightarrow 2 \rightarrow \dots \rightarrow K$).
    - When adding a channel, boost prior/KL weight briefly to force usage.

## 5. Dashboard Requirements

- **Multi-View**: Display $K$ separate 2D scatter plots.
- **Evolution**: Show plots appearing as the curriculum unlocks channels.
- **Discovery**: User sees "Channel 7 = Fives", "Channel 12 = Sevens".
