# RCM-VAE Mathematical Specification

> **Purpose:** This is the precise, math-forward contract for the responsibility-conditioned mixture VAE. For intuition and the stable mental model, see [Conceptual Model](conceptual_model.md). For the current implementation status, see [Implementation Roadmap](implementation_roadmap.md).
>
> **Note:** All math is written with `$...$` (inline) and `$$...$$` (display), which renders in standard Markdown viewers with KaTeX/MathJax support.

---

## 1. Abstract

We present a responsibility-conditioned mixture VAE for semi-supervised classification with OOD awareness and dynamic label growth. A discrete channel $c$ captures global modes; a continuous latent $z$ captures within-mode variation. The decoder is component-aware, $p_\theta(x\mid z,c)$. A latent-only classifier arises from responsibilities $r_c(x)=q_\phi(c\mid x)$ and a component→label map $\tau$. We support interchangeable priors. **Default:** K-channels with $z\mid c\sim\mathcal N(0,I)$ and fixed uniform $\pi$. Channel usage is sparsified to keep only as many channels as needed; free channels serve OOD/new labels.

---

## 2. Introduction

**Default path.** Unless stated otherwise, we use **K-channels** with $z\mid c\sim\mathcal N(0,I)$ and **fixed uniform** $\pi$; amortized $q_\phi(c\mid x),q_\phi(z\mid x,c)$; a **component-aware decoder** $p_\theta(x\mid[z;e_c])$; a **per-image scalar** $\sigma(x)$ (clamped); a **latent-only classifier** via $\tau$ (stop-grad); **usage-entropy sparsity** on $\hat p(c)$; and **no $c$-KL** by default.

Goal: a single model that (i) classifies from latent space; (ii) is uncertainty-aware (aleatoric via decoder variance, epistemic via latent sampling); and (iii) can add labels over time. We factor global vs. local variability via $c$ and $z$, then supervise the latent space lightly via responsibility-weighted label counts.

---

## 3. Model

### 3.1 Prior Modes (Interchangeable)

**(A) K-channels (Default):**
$$
c\sim\mathrm{Cat}(\pi),\quad z\mid c\sim\mathcal N(0,I_d),\quad x\mid z,c\sim p_\theta(x\mid z,c),\quad \pi_c\equiv 1/K.
$$

**(B) VampPrior:** learn pseudo-inputs $u_1,\dots,u_K\in\mathcal X$ and define
$$
p(c)=\mathrm{Cat}(\pi),\quad p(z\mid c)=q_\phi(z\mid u_c),\quad p(z)=\sum_{c=1}^K \pi_c\, q_\phi(z\mid u_c),\quad x\mid z,c\sim p_\theta(x\mid z,c).
$$

**(C) Fixed geometric MoG :** centers arranged on circle/grid, uniform $\pi$; 

### 3.2 Approximate Posterior

$$
q_\phi(c,z\mid x)=q_\phi(c\mid x)\,q_\phi(z\mid x,c),\quad q_\phi(z\mid x,c)=\mathcal{N}\big(\mu_\phi(x,c),\operatorname{diag}(\sigma_\phi^2(x,c))\big).
$$

### 3.3 Component-Aware Decoder

Concatenate an embedding $e_c$ with $z$: $\tilde z=[z; e_c]$, so $p_\theta(x\mid z,c)=p_\theta(x\mid \tilde z)$. **Conditioning policy:** train by evaluating the reconstruction term as a **weighted sum over channels** (expectation under $q(c\mid x)$); for efficiency we enable **Top-$M$ gating (default $M{=}5$)** and keep $\mathrm{KL}_c$ (if used) over all $K$. Optional: a short **soft-embedding warm-up** (replace $e_c$ by $\sum_c q(c\mid x)e_c$) in the first epochs; at **generation** time, sample a hard $c$ and decode with $e_c$.

---

## 4. Objective (Minimize Form)

**Convention.** We minimize losses; all regularizers are written as positive penalties.

Per-example objective:
$$
\mathcal L(x)= \underbrace{-\sum_c q_\phi(c\mid x)\,\mathbb{E}_{q_\phi(z\mid x,c)}\big[\log p_\theta(x\mid z,c)\big]}_{\text{Recon}}
\;+\;
\underbrace{\sum_c q_\phi(c\mid x)\,\mathrm{KL}\big(q_\phi(z\mid x,c)\,\|\,p(z\mid c)\big)}_{\text{$z$-KL}}
\;+\;
\underbrace{\beta_c\,\mathrm{KL}\big(q_\phi(c\mid x)\,\|\,\pi\big)}_{\text{$c$-KL (default }\beta_c{=}0\text{)}}.
$$

**Supervised latent loss (labeled $(x,y)$):**
$$
\mathcal L_{\text{sup}}(x,y)=-\log\sum_c q_\phi(c\mid x)\,\tau_{c,y}.
$$

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
Predict with $\ p(y\mid x)=\sum_c q_\phi(c\mid x)\,\tau_{c,y}$. Implementation: update $\tau$ from responsibility-weighted counts; treat $\tau$ as **stop-grad** in $\mathcal L_{\text{sup}}$. Multiple channels per label are allowed. Channels with low $\max_y\tau_{c,y}$ are candidates for OOD/new labels.

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

6. **Dynamic labels.** **Free channel:** a channel is free if $\hat p(c){<}10^{-3}$ **or** $\max_y\tau_{c,y}{<}0.05$. A new label claims **1–3** free channels chosen by highest responsibilities of its first labeled examples; initialize counts with those examples.

---

## 8. Defaults (MNIST-like)

- Latent $d{=}16$ (or $d{=}2$ for direct viz), $K\in[50,100]$.

- **Decoder variance:** per-image scalar, $\sigma_{\min}{=}0.05$, clamp $[0.05,0.5]$.

- **Channel weights:** $\pi$ fixed uniform by default (no $\pi$-prior term). If learnable: Dirichlet prior optional.

- **Top-$M$ gating:** default $M{=}5$.

- **Label smoothing prior:** $\alpha_0\approx 1$. **EMA** momentum $0.9$–$0.99$ for $\hat p(c)$.

- **$c$-KL weight:** $\beta_c{=}0$ by default.

---

## 9. Experiment Plan

- **Datasets:** MNIST (base), Fashion-MNIST (stress), CIFAR-10 grayscale (stretch).

- **Protocols:** few-label regime (e.g., 10–50 labels/class), class-imbalance, dynamic label addition, OOD with unseen digits/fashion classes.

- **Metrics:** Accuracy/NLL/ECE; OOD AUROC/AUPR; recon-NLL; $K_{\text{eff}}$ via $H(\hat p(c))$; NMI/ARI; calibration plots.

- **Baselines:** standard VAE + softmax; mixture VAE without component-aware decoder; evidential softmax; contrastive SSL + linear probe.

---

## 10. Ablations

- K-channels vs VampPrior; with/without prior-shaping.

- With/without component-aware decoding.

- Sparsity schedules; with/without repulsion.

- Soft-embedding warm-up on/off; Top-$M$ gating $M\in\{3,5\}$.

- Learned-$\sigma$ vs fixed-$\sigma$ decoder.

---

## 11. Limitations & Risks

- Over-fragmentation if sparsity too weak; over-pruning if too strong.

- Fixed geometric MoG can mislead if its topology mismatches data.

- PoE/tempering (if tried) can collapse modes without careful normalization.

- Top-$M$ gating is a biased estimator (good trade-off empirically).

---

## Appendix A — Prior Shaping (VampPrior)

- **MMD** between ${\mathrm{Enc}(u_k)}$ (or samples from $q_{\text{mix}}$) and target samples; use RBF kernels, anneal weight after recon stabilizes.

- **MC-KL** $\mathrm{KL}\big(q_{\text{mix}}\,\|\,p_{\text{target}}\big)$ via log-sum-exp; sample a few points per component.

---

## Appendix B — PoE Prior (Experimental)

Tempered product $p(z)\propto \prod_c p_c(z)^{\tau}$ with $\tau\in(0,1]$. Monitor normalization, entropy, and collapse.

---

## Appendix C — Contrastive Add-On (Optional)

Supervised contrastive in latent/projection space; cluster-prototype term using $e_c$ or latent centroids; weight modestly alongside ELBO.

---

## Appendix D — VampPrior Hygiene

Pseudo-inputs: diverse init (data/k-means), smaller LR, mild repulsion between ${\mathrm{Enc}(u_i)}$, optional annealed diversity loss.

---

## Appendix E — Encoder Variants

Default: shared trunk with small per-channel heads for $q(z\mid x,c)$. Heavy K-branch encoders are supported as experiments but compute-heavy.

---

## Additional Notes

**Sparsity (default).** We use usage-entropy on empirical channel usage as the default; a Dirichlet prior on $\pi$ is optional.

**Responsibility convention.** We write $r_c(z)$ for latent-space responsibilities. In practice we compute $r_c(z)$ via the encoder as $q_{\phi}(c\mid x)$ for the input that produced $z$.

**OOD scoring.** Use responsibility×label-map confidence, e.g., $1-\max_c r_c(z)\,\max_y \tau_{c,y}$ (optionally blended with reconstruction checks).

**Decoder variance.** Default is a per-image $\sigma(x)$ (clamped) for stability; a per-pixel head is optional and can be enabled later.

---

## Related Documentation

- **[Conceptual Model](conceptual_model.md)** - High-level intuition and stable mental model
- **[Implementation Roadmap](implementation_roadmap.md)** - Current implementation status and next steps
- **[System Architecture](../development/architecture.md)** - Design patterns and component structure in the codebase
