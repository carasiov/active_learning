# RCM-VAE — Conceptual Model (North Star)

> **Purpose:** This document establishes the stable mental model and core invariants for the responsibility-conditioned mixture VAE architecture. For precise mathematical formulations and training protocols, see [Mathematical Specification](mathematical_specification.md). For the current implementation status, see [Implementation Roadmap](implementation_roadmap.md).

---

## Background

We set out to model **predictive uncertainty** in classification and to turn a VAE's latent space into a **label-efficient, uncertainty-aware classifier**. Our baseline toolkit combines **evidential/Dirichlet heads** (predict class probs and confidence via $\alpha = S p$), **ensembles/MC-dropout** for epistemic spread, and **heteroscedastic decoders** for aleatoric noise, with simple **calibration** (temperature/conformal). The disentanglement is intentional: the mean says _which class_, the strength ($S$) says _how sure_; $S$ is per-input (optionally per-class).

We connect this to VAEs by placing aleatoric variance in the **decoder** ($\sigma^2(x)$) (clamped) and epistemic in **$z$** (and model parameters). To expose semantics, we introduce **discrete channels** ($c$) for global modes and keep **continuous** ($z$) for within-mode detail, with a **component-aware decoder** $p_\theta(x\mid z,c)$. The encoder yields **responsibilities** $r_c(x)=q(c\mid x)$. From labeled points we accumulate soft counts $s_{c,y}$ and normalize to a **channel→label map** $\tau_{c,y}$. Classification is **latent-only**:
$$
p(y\mid x)=\sum_c q_\phi(c\mid x)\cdot\tau_{c,y} \quad \text{(use stop-grad on }\tau\text{ when used in the loss).}
$$
This separates _where the point sits_ (via $r$) from _how channels map to labels_ (via $\tau$), and naturally supports **multiple channels per label** (multimodality).

**ELBO (component-aware decoder)**
$$
\mathcal L(x)=
\mathbb E_{q_\phi(c\mid x)}\Big[
\mathbb E_{q_\phi(z\mid x,c)}[-\log p_\theta(x\mid z,c)]
+\mathrm{KL}\big(q_\phi(z\mid x,c)\,\|\,p(z\mid c)\big)
\Big]
+\mathrm{KL}\big(q_\phi(c\mid x)\,\|\,\pi\big).
$$

For **OOD**, we define "not owned by labeled channels" and score
$$
\text{OOD}(x)=1-\max_c r_c(z)\cdot\max_y \tau_{c,y},
$$
optionally blended with reconstruction error. (With identical $p(z\mid c)=\mathcal N(0,I)$, latent density alone is uninformative by design; $r,\tau$, and reconstruction carry the signal.)

On **priors**, we keep options open: fixed **Mixture-of-Gaussians**, **VampPrior**, or flows. We view VampPrior's effect as **restriction to prototype posteriors**, producing soft cluster attraction; when we need a target latent shape, we match it (e.g., **MMD**, **MC-KL**, moment matching). As a default uses usage-entropy sparsity; optional Dirichlet prior on $\pi$. We enforce **parsimony** with usage-entropy/Dirichlet penalties and mild **repulsion** to avoid duplicate channels, keeping **free channels** for new labels/OOD.

We drive **label efficiency** with an active loop: query points where $r$ and $\tau$ disagree or reconstruction is poor; use a **2D visualization** (projection of $z$) for interactive selection; and add **contrastive learning** (supervised + semi-supervised via responsibilities) to sharpen cluster geometry.

For **training hygiene**, we use KL/temperature **anneals** (especially on $q(c\mid x)$), **variance clamps** for $\sigma^2(x)$, **Top-$M$ gating** for channels, and responsibility **thresholding** when updating $s_{c,y}$. Our roadmap: baseline VAE → conditional VAE → mixture prior/VampPrior → responsibilities→labels → contrastive+repulsion → component-aware decoder → OOD & free channels → dynamic label addition → 2D viz + active learning → optional adaptive-$K$ (merge/split).

---

## One Story, Two Lenses

Our generative story is simple: data come from a small set of **global modes** and rich **within-mode** variation. We represent the mode with a discrete **channel** ($c$) and the within-mode details with a continuous latent **$z$**. The decoder **knows $c$**, so each channel can specialize; $z$ refines instance-level detail. The **conceptual model** holds this mental model steady. The **specification** is the contract that implements and evaluates it (objectives, metrics, protocol). When details change—loss weights, training tricks—that's spec land; when the picture of how $c$ and $z$ carve reality changes, that's conceptual model land.

## Core Mental Model (Stable Invariants)

Sampling is $c\sim\pi,\ z\sim\mathcal N(0,I_d),\ x\sim p_\theta(x\mid z,c)$. Inference provides **responsibilities** $r_c(x)=q_\phi(c\mid x)$. We keep a **latent-only classifier** by accumulating soft counts from labeled data into $s_{c,y}$ and normalizing to $\tau_{c,y}$; predictions are
$$
p(y\mid x)=\sum_c q_\phi(c\mid x)\cdot\tau_{c,y}.
$$
Aleatoric uncertainty lives in a **heteroscedastic** decoder variance $\sigma^2(x)$ (clamped for stability). We prefer **parsimony**: use only as many channels as needed (sparse $\pi$, usage penalties), allow multiple channels per label (multimodality), and keep **free channels** for new labels or OOD. Optional priors (fixed MoG, VampPrior, flows) may shape $p(z)$; the default is $p(z\mid c)=\mathcal N(0,I)$ with conditioning in the decoder.

## How We Classify and Detect OOD

Classification uses latents only. Responsibilities summarize where a point sits across channels; $\tau$ maps channels to labels. The product $r\times\tau$ yields posterior label mass in latent space without peeking at pixels. OOD is "not owned by any labeled channel": use the score $1-\max_c r_c(z)\cdot\max_y\tau_{c,y}$, optionally blended with reconstruction checks. When all $p(z\mid c)$ are identical standard normals, raw latent density is uninformative for OOD—confidence from $r$ and $\tau$ carries the signal.

## Why This Shape

Giving the decoder $c$ separates **global** structure (which expert) from **local** variation (how this instance looks). That yields cleaner clusters, natural multimodality (several channels per label), and interpretable uncertainty: discrete ambiguity through $q(c\mid x)$, observation noise through $\sigma^2(x)$. The soft-count map $s\!\to\!\tau$ turns a handful of labels into a robust channel–label taxonomy and scales well with active learning.

## Guardrails and Failure Modes

Encourage **sparse usage** so we don't spray mass across many small channels; mild **repulsion** between channels prevents duplicates. Anneal the $c$-KL or temperature so early responsibilities aren't mushy, and clamp $\log\sigma^2$ to avoid variance collapse. If channels fragment a label, that's acceptable multimodality; if many channels mirror each other, increase sparsity and repulsion. If OODs look "confident," tighten the decision by combining the OOD score with reconstruction deviation.

## What "Good" Looks Like

Active channels align with real modes; several can serve one label without cross-label bleed. Calibration is tight (low ECE), and OOD AUROC is strong with human-legible errors in the 2D view. Label efficiency is steep under active learning. Training is stable: responsibilities sharpen over time; no posterior or variance collapse.

## Change Routing and Pointers

If we change the meaning of $c$ or abandon latent-only classification, the **conceptual model** changes. If we swap a loss term, tweak temperatures, adopt Top-$M$ gating, add a MoG/VampPrior/flow, or adjust acquisition rules, the **specification** changes. The conceptual model should point **down** to the spec for mechanics (losses, training, metrics); the spec can point **up** here for intuition and rationale. Keep overlap minimal and purposeful.

## Non-Negotiables (Minimal Contract)

Use a **component-aware decoder** $p_\theta(x\mid z,c)$. Serve a **latent-only** classifier via $r$ and $\tau$ with stop-grad on $\tau$. Enforce **usage sparsity** and **repulsion**; keep **free channels** for discovery. Train a **heteroscedastic** decoder with clamped $\sigma^2(x)$. Provide the stated **OOD score** and integrate it into active learning.

## Notation Lock

We freeze the core symbols: $c$ (channel), $z$ (latent), $r$ (responsibilities), $\tau$ (channel→label), $\pi$ (channel weights), $\sigma^2$ (decoder variance). The spec may introduce extra symbols; these do not alter the conceptual model's set.

**Sparsity (default).** We use usage-entropy on empirical channel usage as the default; a Dirichlet prior on $\pi$ is optional.

**Responsibility convention.** We write $r_c(z)$ for latent-space responsibilities. In practice we compute $r_c(z)$ via the encoder as $q_{\phi}(c\mid x)$ for the input that produced $z$.

**OOD scoring.** Use responsibility×label-map confidence, e.g., $1-\max_c r_c(z)\,\max_y \tau_{c,y}$ (optionally blended with reconstruction checks).

**Decoder variance.** Default is a per-image $\sigma(x)$ (clamped) for stability; a per-pixel head is optional and can be enabled later.

**Product-of-Experts (PoE).** A tempered PoE prior is experimental and may be ablated later; we default to mixtures/channels for stability.

---

## Related Documentation

- **[Mathematical Specification](mathematical_specification.md)** - Precise objectives, training protocol, and mathematical formulations
- **[Vision Gap](vision_gap.md)** - Current implementation vs. target state
- **[System Architecture](../development/architecture.md)** - Design patterns and component structure in the codebase
- **[Implementation Status](../development/STATUS.md)** - Detailed feature status and recent updates
