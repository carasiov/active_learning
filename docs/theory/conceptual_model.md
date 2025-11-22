# RCM-VAE Conceptual Model

> This document explains the *ideas* behind the RCM-VAE (Responsibility-Conditioned Mixture VAE).  
> It is built around the original project intent: an **interactive, semi-supervised, generative classifier** whose semantics live in latent space.  
> For exact math and implementation details, see the [Mathematical Specification](mathematical_specification.md).

---

## 1. Overall Goal

We want one model that can:

- Learn a **generative latent model** of the data (like a VAE).
- Organize data into **discrete components/channels** (clusters) in latent space.
- Use **few labels** + **many unlabeled samples** to turn those components into a **classifier**.
- Provide **uncertainty** and **OOD (out-of-distribution) detection** from its latent structure.
- Support **dynamic labels**: new labels can be introduced during training without redesigning the model.
- Expose a **2D visualization** for interactive, active learning: you explore the latent space, pick points to label, and watch the structure evolve.

The key design choice:

> **Semantics should live in latent space, not in a separate classifier head.**  
> The model is a *generative classifier* whose decision boundaries emerge from its mixture structure.

---

## 2. Core Latent Model: Channels and Local Latents

### 2.1 Two types of latent variables

RCM-VAE uses:

- A **discrete channel** $c \in \{1,\dots,K\}$:
  - Think: “Which expert handles this datapoint?”
  - Each channel is a semantic *slot* that can eventually align with a digit, style, or other coherent concept.
- A **continuous latent** $z_c$ associated with the selected channel:
  - Think: “Within this expert’s regime, what are the exact details of this datapoint?”
  - Captures pose, stroke thickness, style, etc.

In the **decentralized layout** [↗](mathematical_specification.md#31-latent-layouts) (our target mode):

- There is a set of per-channel latents:
  - $Z = \{z_1, \dots, z_K\}$, each $z_k \in \mathbb{R}^d$.
- During training, **all K latents are decoded**:
  - Each $z_k$ is passed through the decoder with embedding $e_k$ to produce reconstruction $\text{recon}_k$.
  - The final reconstruction is a weighted combination: $\sum_k q(c{=}k\mid x) \cdot \text{recon}_k$.

Conceptually, the generative story is:

> Sample a channel $c$, then sample a latent $z$ for that channel, then generate $x$.  
> In training, the encoder [`encoders.py`](../../src/rcmvae/domain/components/encoders.py) maintains a full set $\{z_k\}$ for amortized inference. All K are decoded [`network.py`](../../src/rcmvae/domain/network.py#L159-L341) and weighted by responsibilities to compute the [reconstruction loss](mathematical_specification.md#4-objective-minimize-form). The KL term sums over all K posteriors.

### 2.2 Simple priors, flexible structure (with sparsity)

Priors are intentionally simple and modular:

- **Channels:** $c \sim \text{Categorical}(\pi)$.
- **Latents:** Often  
  $z_k \sim \mathcal{N}(0, I)$ for each channel $k$.

Optionally:

- **VampPrior** [`vamp.py`](../../src/rcmvae/domain/priors/vamp.py): $p(z\mid c)$ defined via pseudo-inputs.
- **Geometric priors** [`geometric_mog.py`](../../src/rcmvae/domain/priors/geometric_mog.py): components arranged in a structured way in latent space.

Crucially, we **regularize channel usage** so that only as many channels as needed become active, for example by:

- Putting a sparsity-biased prior on $\pi$, or
- Adding an [entropy/sparsity penalty](mathematical_specification.md#4-objective-minimize-form) on empirical usage $\hat p(c) \approx \mathbb{E}_x[q_\phi(c\mid x)]$.

Intuitively:

> We want a reasonably large “pool” of potential channels, but we encourage the model to actually use only the subset that helps explain the data and labels.

---

## 3. Encoder and Decoder: A Mixture-of-Expert VAE

### 3.1 Encoder: responsibilities and per-channel latents

The encoder performs **amortized inference**:

- It predicts **responsibilities**:
  - $q_\phi(c \mid x)$ — a soft assignment of datapoint $x$ to channels.
  - These are the *responsibility weights* $r_c(x)$.
- In decentralized mode, it also outputs **per-channel latent parameters**:
  - Means and variances for each $z_k$, shape $[B, K, d]$.

So for each $x$, the encoder effectively says:

> “If this datapoint belonged to channel 1, its latent would be $z_1$;  
> if it belonged to channel 2, its latent would be $z_2$; …  
> and overall I think channels 2 and 7 are most likely.”

#### Gumbel-Softmax as an implementation option

Conceptually, we have a categorical distribution $q_\phi(c\mid x)$ over channels.

Implementation options:

- Use **soft responsibilities directly**:
  - Treat the model as a mixture and weight contributions by $q_\phi(c\mid x)$ (e.g., in reconstruction and KL).
- Or use **Gumbel-Softmax / straight-through**:
  - Sample an approximate one-hot from $q_\phi(c\mid x)$ during training to simulate hard routing while keeping gradients.

[Gumbel-Softmax](mathematical_specification.md#33-approximate-posterior) is thus an **implementation detail** for making channel selection more discrete; the core concept is the categorical responsibilities $q_\phi(c\mid x)$.

### 3.2 Decoder: component-aware via FiLM

The decoder is **conditioned** on the active channel:

- It always decodes from:
  - The **active latent** $z_c$.
  - A learned **channel embedding** $e_c$.

To make each channel behave like its own expert, the decoder uses **FiLM (Feature-wise Linear Modulation)** [`conditioning.py`](../../src/rcmvae/domain/components/decoder_modules/conditioning.py):

- At each layer, feature maps are modulated by parameters $(\gamma_c, \beta_c)$ computed from $e_c$:
  - $h' = \gamma_c \odot h + \beta_c$.

### 3.3 Decoder as a likelihood model

The decoder produces a **likelihood**:

- For continuous data:
  $$p_\theta(x\mid z,c) = \mathcal{N}\big(x; \mu_\theta(z,c), \Sigma_\theta(z,c)\big),$$
  usually with $\Sigma_\theta(z,c) = \text{diag}(\sigma_\theta^2(z,c))$ [`outputs.py`](../../src/rcmvae/domain/components/decoder_modules/outputs.py).
- For binary-ish data:
  $$p_\theta(x\mid z,c) = \text{Bernoulli}\big(x; \sigma(\text{logit}_\theta(z,c))\big).$$

---

## 4. Labels, Channels, and a Latent-Space Classifier

We **never** attach a discriminative head.  
Classification happens entirely through channels and responsibilities.

### 4.1 Building soft label counts

For labeled data $(x_i, y_i)$:

- Use responsibilities $r_c(x_i) = q_\phi(c\mid x_i)$ to accumulate soft counts [`tau_classifier.py`](../../src/rcmvae/domain/components/tau_classifier.py#L61-L97):
  $$s_{c,y} \leftarrow s_{c,y} + r_c(x_i).$$

### 4.2 The channel → label map $\tau$

We convert counts to a probability distribution:

- $\tau_{c,y}$ = probability that channel $c$ corresponds to label $y$ (see [`tau_classifier.py`](../../src/rcmvae/domain/components/tau_classifier.py)).

### 4.3 Purely latent-space classifier

Prediction:

$$p(y\mid x) = \sum_c q_\phi(c\mid x)\,\tau_{c,y}.$$

---

## 5. Uncertainty and OOD

Uncertainty emerges from:

- $q_\phi(c\mid x)$,
- $p_\theta(x\mid z,c)$,
- $\tau_{c,\cdot}$.

### 5.1 Epistemic uncertainty

Arises when responsibilities are ambiguous or predictions vary across samples of $(c,z_c)$.

### 5.2 Aleatoric uncertainty

Comes from decoder variance $\sigma_\theta(z,c)$ and diffuse $\tau_{c,\cdot}$.

### 5.3 OOD scoring

A simple OOD score:

$$s_{\text{OOD}}(x) \approx 1 - \max_c\bigl(q_\phi(c\mid x)\,\max_y \tau_{c,y}\bigr).$$ (See [`tau_classifier.py`](../../src/rcmvae/domain/components/tau_classifier.py#L207-L220) and [Mathematical Specification](mathematical_specification.md#6-ood-scoring).)

---

## 6. Dynamic Labels and Free Channels

Channels start as unlabeled; labeled data shapes $\tau$.

### 6.1 Adding new labels

Use free or weakly aligned channels (see [`tau_classifier.py`](../../src/rcmvae/domain/components/tau_classifier.py#L222-L251)), update counts, and assign them to the new label.

---

## 7. Interaction, Visualization, and Active Learning

### 7.1 2D projection

Use $v = g_\psi(z) \in \mathbb{R}^2$ or a 2D latent.

### 7.2 Active learning

Use predictive entropy, epistemic uncertainty, or boundary points.

---

## 8. Contrastive Learning and Structure

Optional contrastive losses help align channels and labels.

---

## 9. One-Sentence View

RCM-VAE is an **interactive, semi-supervised mixture-VAE** where discrete latent channels are sparsely used semantic slots; they are bound to labels via responsibilities and a simple channel→label map, support new labels, and provide generative modeling, classification, uncertainty, OOD detection, and 2D exploration from one latent space.

