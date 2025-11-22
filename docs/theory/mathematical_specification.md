# RCM-VAE Mathematical Specification

> **Purpose:** This is the precise, math-forward contract for the responsibility-conditioned mixture VAE. For intuition and the stable mental model, see [Conceptual Model](conceptual_model.md). For the current implementation status, see [Implementation Roadmap](implementation_roadmap.md).
>
> **Note:** All math is written with `$...$` (inline) and `$$...$$` (display), which renders in standard Markdown viewers with KaTeX/MathJax support.

---

## 1. Abstract

We present a responsibility-conditioned mixture VAE for semi-supervised classification with OOD awareness and dynamic label growth. A discrete channel $c$ captures global modes; a channel-specific continuous latent $z_c$ captures within-mode variation. This is a **structured variational autoencoder** with a discrete global latent ($c$) and continuous local latents ($z_c$), where inference is performed via **amortized variational inference**: a neural encoder predicts $q_\phi(c\mid x)$ and $q_\phi(z_c\mid x,c)$. The decoder is component-aware, $p_\theta(x\mid z,c)$. A latent-only classifier arises from responsibilities $r_c(x)=q_\phi(c\mid x)$ and a component→label map $\tau$. We support interchangeable priors (mixture of Gaussians, VampPrior, geometric arrangements). Channel usage is sparsified to keep only as many channels as needed; free channels serve OOD/new labels.

---

## 2. Introduction

This model achieves three goals: (i) classification from latent space; (ii) [uncertainty awareness](conceptual_model.md#5-uncertainty-and-ood) (aleatoric via decoder variance, epistemic via latent sampling); and (iii) dynamic label addition over time. We factor global vs. local variability via $c$ and $z$, then supervise the latent space lightly via responsibility-weighted label counts.

This model employs a **structured variational posterior** of the form $q_\phi(c, z \mid x)$. Training maximizes an **evidence lower bound (ELBO)** comprising reconstruction likelihood, KL divergence terms, and optional regularizers, as in standard variational inference. The discrete channel $c$ captures global mixture-style modes (e.g., digit identity in MNIST), while the continuous latent $z_c$ captures local within-mode variation (e.g., stroke thickness, rotation).

---

## 3. Model

### 3.1 Latent Layouts

The system supports two latent layouts (configured via `latent_layout` in [`config.py`](../../src/rcmvae/domain/config.py)):

**1. Shared Layout (Legacy/Baseline):**
Single global latent $z \in \mathbb{R}^D$. Components compete to explain data in this shared space.
$$
c \sim \mathrm{Cat}(\pi), \quad z \mid c \sim \mathcal{N}(0, I), \quad x \sim p_\theta(x \mid z, c)
$$

**2. Decentralized Layout (Target):**
The generative model samples one channel and one latent:
$$
c \sim \mathrm{Cat}(\pi), \quad z \sim p(z \mid c) = \mathcal{N}(0, I), \quad x \sim p_\theta(x \mid z, c)
$$

**Training implementation (hybrid approach):** The encoder [`encoders.py`](../../src/rcmvae/domain/components/encoders.py) produces $K$ independent posteriors $q_\phi(z_k \mid x)$ for $k=1,\dots,K$. During training, **all K latents are decoded** [`network.py`](../../src/rcmvae/domain/network.py#L159-L341) to produce per-component reconstructions, which are then weighted by $q_\phi(c\mid x)$. The KL term sums over all K posteriors. This treats the K latents as amortized inference variables with independent priors $p(z_k) = \mathcal{N}(0,I)$, forming a structured approximation to $q_\phi(c, z \mid x)$.

### 3.2 Prior Modes

**Prior Modes (for $\pi$ and $z$ structure):**
*   **Mixture:** Learned $\pi$, standard normal $z$ [`mixture.py`](../../src/rcmvae/domain/priors/mixture.py).
*   **VampPrior:** $p(z|c)$ defined by pseudo-inputs [`vamp.py`](../../src/rcmvae/domain/priors/vamp.py).
*   **Geometric:** Fixed spatial arrangement of components [`geometric_mog.py`](../../src/rcmvae/domain/priors/geometric_mog.py).

### 3.3 Approximate Posterior

**Shared:**
$$ q_\phi(c, z \mid x) = q_\phi(c \mid x) \, q_\phi(z \mid x) $$
(Single encoder head for $z$).

**Decentralized:**
The encoder produces $K$ independent Gaussian posteriors:
$$ q_\phi(z_k \mid x) = \mathcal{N}(\mu_k(x), \sigma_k^2(x)) \quad \text{for } k=1,\dots,K. $$
This forms a structured approximation to $q_\phi(c, z \mid x) = q_\phi(c \mid x) \, q_\phi(z \mid x, c)$.

**Training procedure:** The encoder outputs all $K$ sets of parameters (shape $[B,K,D]$). During training:
1. All $K$ latents $z_k \sim q_\phi(z_k \mid x)$ are sampled and decoded: $\text{recon}_k = p_\theta(x \mid z_k, k)$
2. Reconstruction loss: $\mathcal{L}_{\text{recon}} = \sum_{k=1}^K q_\phi(c{=}k\mid x) \cdot \mathcal{L}(x, \text{recon}_k)$ (weighted expectation)
3. KL loss: $\text{KL}_z = \sum_{k=1}^K \text{KL}(q_\phi(z_k\mid x) \| p(z_k))$ (sum over all K)

**Discrete Relaxation (Gumbel-Softmax):**
To differentiate through component selection $c$, we use the [Gumbel-Softmax trick](conceptual_model.md#gumbel-softmax-as-an-implementation-option) (implemented in [`network.py`](../../src/rcmvae/domain/network.py#L182-L227)):
$$ y_k = \frac{\exp((\log \pi_k + g_k) / \tau)}{\sum_j \exp((\log \pi_j + g_j) / \tau)} $$
where $g_k \sim \mathrm{Gumbel}(0, 1)$ and $\tau$ is temperature.
*   **Training:** Use soft samples $y$ (or straight-through hard samples) to weight decoder inputs/losses.
*   **Inference:** Hard sampling $c = \arg\max y$.

### 3.4 Decoder Architectures

Implemented in [`decoders.py`](../../src/rcmvae/domain/components/decoders.py) using modules from [`decoder_modules/`](../../src/rcmvae/domain/components/decoder_modules/).

**Standard (concatenated):** Embed component $c$ as $e_c$, then concatenate with $z$: $\tilde z=[z; e_c]$, so $p_\theta(x\mid z,c)=p_\theta(x\mid \tilde z)$ with shared decoder weights.

**Component-aware:** Separate transformation pathways for $z$ and $e_c$ before fusion:
$$z_{\text{path}} = W_z(z), \quad e_{\text{path}} = W_e(e_c), \quad \tilde z = [z_{\text{path}}; e_{\text{path}}], \quad p_\theta(x\mid z,c)=p_\theta(x\mid \tilde z).$$
This enables component-specific feature learning while both architectures receive embedding context.

**FiLM conditioning (current):** Generate affine parameters from embedding: $(\gamma,\beta)=g_\theta(e_c)$, apply feature-wise modulation $h'=\gamma\odot h + \beta$ inside the decoder ([`conditioning.py`](../../src/rcmvae/domain/components/decoder_modules/conditioning.py)). This strictly dominates concatenation when component embeddings are available.

**Decentralized training detail:** In decentralized layout, the decoder processes **all K latents** $(z_1, \dots, z_K)$ with their corresponding embeddings $(e_1, \dots, e_K)$ to produce $K$ reconstructions. The final output is a weighted combination using responsibilities/component selection.

**Conditioning policy:** Train by evaluating the reconstruction term as a **weighted sum over channels** (expectation under $q(c\mid x)$); for efficiency we enable **Top-$M$ gating (default $M{=}5$)** and keep $\mathrm{KL}_c$ (if used) over all $K$. Optional: a short **soft-embedding warm-up** (replace $e_c$ by $\sum_c q(c\mid x)e_c$) in the first epochs; at **generation** time, sample a hard $c$ and decode with $e_c$.

**Heteroscedastic output (current):** Decoder emits $(\mu,\sigma)$ with $\sigma = \operatorname{clip}\big(\sigma_{\min} + \operatorname{softplus}(s),\, \sigma_{\min},\, \sigma_{\max}\big)$ for stability ([`outputs.py`](../../src/rcmvae/domain/components/decoder_modules/outputs.py)); likelihood term uses $\|x-\mu\|^2/(2\sigma^2)+\log\sigma$ ([`loss_pipeline.py`](../../src/rcmvae/application/services/loss_pipeline.py#L106-L149)).

---

## 4. Objective (Minimize Form)

**Convention.** We minimize losses; all regularizers are written as positive penalties.

Per-example objective (ELBO) (implemented in [`loss_pipeline.py`](../../src/rcmvae/application/services/loss_pipeline.py)):
$$
\mathcal L(x) = \underbrace{-\mathbb{E}_{q_\phi(c\mid x)}\big[\mathbb{E}_{q_\phi(z\mid x,c)}[\log p_\theta(x\mid z,c)]\big]}_{\text{Recon}} + \text{KL}_z + \underbrace{\beta_c\,\mathrm{KL}\big(q_\phi(c\mid x)\,\|\,\pi\big)}_{\text{$c$-KL}}.
$$
In the decentralized layout, $q_\phi(z\mid x,c)$ refers specifically to the posterior of the active latent $z_c$, i.e., the $c$-th latent in the set $\{z_k\}$ produced by the encoder.

**Latent KL ($\text{KL}_z$) depends on layout:**

*   **Shared:** Weighted sum against component priors.
    $$ \text{KL}_z = \sum_c q_\phi(c\mid x)\,\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z\mid c)\big) $$
*   **Decentralized:** Sum of independent KLs for all components.
    $$ \text{KL}_z = \sum_{k=1}^K \mathrm{KL}\big(q_\phi(z_k\mid x)\,\|\,p(z_k)\big) $$

**Supervised latent loss (labeled $(x,y)$):**
$$
\mathcal L_{\text{sup}}(x,y)=-\log\sum_c q_\phi(c\mid x)\,\tau_{c,y}.
$$
(See [`tau_classifier.py`](../../src/rcmvae/domain/components/tau_classifier.py)).

**Channel-usage sparsity (EMA $\hat p$):**
$$
\mathcal R_{\text{usage}}=\lambda_u\Big(-\sum_c \hat p(c)\log \hat p(c)\Big)\quad\text{(minimize entropy)}.
$$

**Decoder variance stability:** per-image scalar $\sigma(x)=\sigma_{\min}+\mathrm{softplus}(s_\theta(x))$, clamp $\sigma(x)\in[0.05,0.5]$; optional small penalty $\lambda_\sigma(\log\sigma(x)-\mu_\sigma)^2$ (default off).

**Prior on channel weights.** Off by default (fixed uniform $\pi$). If $\pi$ is learnable, add $-\lambda_\pi\log p(\pi)$ (e.g., Dirichlet prior).

**Optional prior shaping (VampPrior only):** distance $D(q_{\text{mix}},p_{\text{target}})$ via MMD or MC-KL; apply after recon stabilizes.

**Total loss:**
$$
\min_\Theta\ \mathbb{E}_x[\mathcal L(x)] + \lambda_{\text{sup}}\,\mathbb{E}_{(x,y)}[\mathcal L_{\text{sup}}]
\;+\; \mathcal R_{\text{usage}}
\;+\; \lambda_\pi\,\mathcal R_\pi
\;+\; \lambda_{\text{shape}}\,D(q_{\text{mix}},p_{\text{target}})
\;+\; \text{(optional: contrastive, repulsion)}.
$$

---

## 5. Responsibilities → $\tau$ → Latent Classifier

Maintain soft counts per channel/label:
$$
s_{c,y}\leftarrow s_{c,y}+q_\phi(c\mid x)\,\mathbf{1}\{y_i=y\},\qquad \tau_{c,y}=\frac{s_{c,y}+\alpha_0}{\sum_{y'}(s_{c,y'}+\alpha_0)}.
$$
Predict with $\ p(y\mid x)=\sum_c q_\phi(c\mid x)\,\tau_{c,y}$. Implementation: update $\tau$ from responsibility-weighted counts; treat $\tau$ as **stop-grad** in $\mathcal L_{\text{sup}}$ (see [`tau_classifier.py`](../../src/rcmvae/domain/components/tau_classifier.py)). Multiple channels per label are allowed. Channels with low $\max_y\tau_{c,y}$ are candidates for OOD/new labels.

---

## 6. OOD Scoring

**Default:**
$$
s_{\text{OOD}}(x) = 1 - \max_c \Bigl( q_\phi(c \mid x) \cdot \max_y \tau_{c,y} \Bigr).
$$

**Variant (optional):**
$$
s^{\text{mix}}_{\text{OOD}} = w_1\, s_{\text{OOD}} + w_2\, \mathrm{ReconError}(x),\quad w_1 + w_2 = 1.
$$

---

## 7. Training Protocol

1. **Encode** logits for $q_\phi(c\mid x)$ and per-channel $\mu_\phi(x,c),\log\sigma^2_\phi(x,c)$. Maintain EMA for $\hat p(c)$.

2. **Reconstruction as expectation over channels** with **Top-$M$ gating** (default $M{=}5$). Compute $z$-KL on the same set; compute $c$-KL over **all $K$** if enabled.

3. **Anneal** the $z$-KL weight linearly 0→1 over the first ~10k steps; keep **$\beta_c{=}0$ by default**.

4. **Decode** with $[z; e_c]$. Optional short **soft-embedding warm-up**; at generation, sample a hard $c$.

5. **Optimize** the total loss; consider mild repulsion between $e_c$ to avoid duplicate channels.

6. **Dynamic labels.** **Free channel:** a channel is free if $\hat p(c){<}10^{-3}$ **or** $\max_y\tau_{c,y}{<}0.05$ (see [`tau_classifier.py`](../../src/rcmvae/domain/components/tau_classifier.py#L222-L251)). A new label claims **1–3** free channels chosen by highest responsibilities of its first labeled examples; initialize counts with those examples.


---

## Additional Notes

**Sparsity (default).** We use usage-entropy on empirical channel usage as the default; a Dirichlet prior on $\pi$ is optional.

**Responsibility convention.** Responsibilities are always derived from encoder outputs: $r_c(x) = q_\phi(c\mid x)$. When referring to latent-space points $z$, we use $r_c(z)$ to denote the same value inherited from the corresponding input $x$.

**OOD scoring.** Use responsibility×label-map confidence, e.g., $1-\max_c r_c(z)\,\max_y \tau_{c,y}$ (see [Section 6](#6-ood-scoring)), optionally blended with reconstruction checks.

**Decoder variance.** Default is a per-image $\sigma(x)$ (clamped) for stability; a per-pixel head is optional and can be enabled later.

---

## Related Documentation

- **[Conceptual Model](conceptual_model.md)** - High-level intuition and stable mental model
- **[Implementation Roadmap](implementation_roadmap.md)** - Current implementation status and next steps
- **[System Architecture](../development/architecture.md)** - Design patterns and component structure in the codebase
