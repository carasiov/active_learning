# Channel Curriculum ("Pots") — Project Context

> **Purpose:** Comprehensive briefing for any agent working on the channel curriculum.
> **Last updated:** 2025-12-23

**Recent updates:**
- 2025-12-23: Added §11 (Experimental Findings), §12 (Supervisor Meeting Dec 3), updated §6 (Priorities)

---

## 1. Supervisor Decisions & Constraints (authoritative)

These are settled decisions from the supervisor. Do not revisit without explicit direction.

* **Replace "Dirichlet / categorical sampling tricks" for per-sample c sparsity** with a **Gaussian-only prior in logit space**: encode into R^K, then apply **softmax** to obtain a probability simplex vector. Regularize the *pre-softmax logits* with a **mixture of Gaussians** so that after softmax the resulting q(c|x) is **peaky / near one-hot**.

* **Keep KL / regularization computable and stable** by using **Normal distributions** (and log-sum-exp for mixtures) rather than relying on Dirichlet sampling or categorical sampling behavior.

* **Per-sample "winner-takes-all"** is the intended behavior for c: one channel dominates for a given x. **No parts-based mixing across channels** as the default target.

* **Softmax-on-logits is the key mechanism**: do not treat c as a Dirichlet variable if that complicates training; operate in R^K and map to the simplex via softmax.

* **Mixture prior geometry**: mixture components centered along coordinate axes (e.g., M * e_k) to encourage mass on individual components after softmax.

* **Curriculum is part of the supervisor's intended setup**: start with few channels and **unlock additional channels over training** to avoid compromise solutions and to allow channels to specialize.

---

## 2. Mathematical Characterization (verified against codebase)

> Source: High-level agent synthesis + intermediate agent verification (2025-12-22)

The implemented model is a **mixture-of-experts VAE with per-component latents**, not a VampPrior-driven latent mixture.

### Generative Model

```
z_1, ..., z_K  ~iid  N(0, I)
c  ~  Cat(π)           where π = softmax(pi_logits)
x  ~  p_θ(x | z_c, c)  implemented as mixing per-component reconstructions
```

### Inference Factorization

```
q_φ(c, z_{1:K} | x) = q_φ(c | x) ∏_k q_φ(z_k | x)
```

All K posteriors are amortized in parallel. This is the **"sample all z_k, then select one"** variant—not "sample c, then sample z_c".

### Objective Structure

```
L = L_recon
  + β_z * Σ_k KL(q(z_k|x) || N(0,I))   # summed over ALL k, not weighted by responsibilities
  + λ_cat * KL(q(c|x) || π)            # categorical KL (optional)
  + λ_mog * L_logit_mog                 # logit-space Gaussian mixture penalty
  + dirichlet_penalty(π)                # optional sparsity on π
  + diversity_term(q(c|x))              # optional batch-usage entropy
```

**Key implication:** Continuous KL is not responsibility-weighted. Inactive channels naturally push q(z_k|x) → N(0,I) to minimize cost.

### Three Forces Shaping Routing

1. **Reconstruction advantage** — specialization improves reconstruction
2. **KL-to-π + π-prior** — usage budget, global mixture shape
3. **Logit-MoG geometry** — axis-aligned Gaussian prior on encoder logits

Force (3) is a **strong modeling choice** that can dominate clustering behavior. Treat it as such when debugging.

### Label Integration (τ-classifier)

Labels influence training through EM-like soft count accumulation, not direct gradient flow:

```
s_{c,y} ← s_{c,y} + q(c|x) * 1{y=y_i}   # accumulate counts
τ_{c,y} = normalize(s_{c,y})             # component → label map
loss = -log Σ_c q(c|x) * stop_gradient(τ_{c,y})
```

Gradients flow through q(c|x) only. Labels steer routing, not decoder/embeddings directly.

### VampPrior vs Mixture Prior

The codebase supports both, but **current curriculum uses `prior_type="mixture"`** (direct component embeddings), not VampPrior pseudo-inputs. The system is closer to a **conditional MoE-VAE** than a classic VampPrior VAE.

---

## 3. Research Direction

The system is moving from a single global latent space to **K decentralized latent "channels"**, where each channel has its own continuous latent space z_k. A discrete variable c (implemented via responsibilities r(x) = q(c|x)) decides which channel is responsible for a given input.

**Goal:** Class identity lives primarily in the channel index (e.g., one or few channels per digit), while intra-class variation (stroke thickness, slant) is captured in the continuous latent within the selected channel.

The "pots" curriculum is the stabilization mechanism:
1. Begin with K_active = 1 so the model learns reconstruction and a stable latent
2. As training plateaus, unlock channels one at a time
3. Each newly unlocked channel should become competitive and specialize

**Core intuition:** Specialization should emerge through routing pressure + decoder specialization, not by mixing parts across channels.

---

## 4. Invariants (must not be violated)

* **Static K_max shape** — curriculum is implemented by masking, not resizing tensors
* **Routing masking to -∞** is allowed only for distributions (softmax/Gumbel-softmax) used for routing/selection
* **Logit-MoG must never use -∞ masked logits** — compute on finite raw logits and restrict mixture sum to active set
* **Monitoring/usage statistics** should be based on deterministic responsibilities (masked softmax), not noisy Gumbel samples

---

## 5. Open Questions (needs supervisor input)

* **Unlock trigger long-term:** v1 uses plateau on validation metric; supervisor mentioned "channels look normal" criterion. Define what "normal enough" means.

* **Canonical kick mechanism:** Temperature-only kicks are insufficient under ST routing. Decide between: disable ST during kick, logit bias to new channel, relax logit-MoG strength, or combination.

* **Global usage shaping:** If system collapses to one channel globally, do we want within-active-set diversity term, or pure architectural pressure?

* **Channels per label:** Is 1:1 mapping the goal, or "few channels per class" acceptable?

* **When to stop unlocking:** Fixed K_max, or early stop based on saturation criterion?

---

## 6. Priorities (ordered)

> **Status update (2025-12-23):** V2 kick mechanism implemented and tested. New failure mode discovered: 2-channel coalition. See §11 for experimental findings.

1. ~~**Make unlocking cause new channels to be used**~~ **PARTIALLY SOLVED**
   - V2 kick (logit bias + temperature) successfully breaks single-channel monopoly
   - **New issue:** Stable 2-channel coalition forms; channels 3+ cannot compete
   - See `experimental_findings_2025-12-23.md` for details

2. **Test simultaneous training baseline** (NEW - highest priority)
   - Supervisor recommended: K=20-30, L1 sparsity, no curriculum
   - Purpose: Establish what unsupervised specialization is achievable
   - See §12 for supervisor guidance on L1

3. **Implement perturbation at unlock** (if baseline shows specialization)
   - Supervisor predicted local minimum risk; suggested perturbation
   - Options: noise to incumbent weights, incumbent LR decay, reconstruction-based routing

4. **Consider hybrid approach**
   - Simultaneous training for initial specialization
   - Curriculum only for capacity expansion after specialization achieved

5. ~~**Ensure curriculum invariants apply to evaluation + artifacts**~~ **DONE**
   - Implemented in V2

6. **Add L1 sparsity term** (supervisor recommendation)
   - `λ · Σ|w_c|` on channel weights
   - May replace or complement current entropy-based diversity term

---

## 7. Gotchas (do not re-learn)

### Straight-through Gumbel + temperature kick pitfall
With ST hard routing, argmax is effectively argmax(logits + g); temperature does not reliably change which component wins. Temperature mostly affects soft gradients. A "temperature-only kick" often fails to redistribute usage after unlock.

### Masking vs. logit-MoG numerical stability
Routing logits may be -∞ masked; **logit-MoG must not use these** in squared-distance computations or it produces inf/NaN. Compute logit-MoG on finite raw logits and restrict mixture sum to active indices.

### Metric naming confusion
`kl_c_logit_mog` is not a categorical KL; it's a **negative log prior penalty** on deterministic logits. Interpret accordingly.

### Unlock + early stopping interaction
If early stopping patience isn't reset after unlock, training can stop before the newly unlocked channel has time to learn.

### Post-training artifacts must respect active set
If prediction/plot generation recomputes routing without active_mask, artifacts can violate curriculum assumptions and mislead debugging.

### Peakiness alone doesn't imply multi-channel usage
Logit-MoG enforces per-sample one-hotness but is symmetric across channels; without exploration/kick or mild usage pressure, global collapse to a single channel can remain stable.

---

## 8. Monitoring Checklist

When verifying curriculum behavior, check:

- [ ] **K_eff** = exp(H(p̂(c))) — effective number of channels used
- [ ] **Per-channel KL** — inactive channels should have ~0 KL(q(z_k|x) || N(0,I))
- [ ] **Routing sharpness** — H(q(c|x)) distribution, with/without ST Gumbel
- [ ] **τ matrix** — visualize τ_{c,y}, expect low-entropy rows, multiple rows per label
- [ ] **Component duplication** — compare embeddings e_k or decoder outputs
- [ ] **Logit-MoG influence** — does it impose clusters that conflict with reconstruction/supervision?

---

## 9. Code Pointers

| Concern | Location |
|---------|----------|
| Encoder (logits + per-component z) | `src/rcmvae/domain/components/encoders.py` |
| Routing, masking, decoding | `src/rcmvae/domain/network.py:159-401` |
| Logit-MoG + KL terms | `src/rcmvae/domain/priors/mixture.py` |
| τ-classifier | `src/rcmvae/domain/components/tau_classifier.py` |
| Curriculum controller | `src/rcmvae/application/curriculum/controller.py` |
| Curriculum hooks | `src/rcmvae/application/curriculum/hooks.py` |
| Loss pipeline | `src/rcmvae/application/services/loss_pipeline.py` |
| Config surface | `src/rcmvae/domain/config.py` |

---

## 10. Related Documents

- `README.md` — entry point and glossary
- `high_level_context.md` — intuition and rationale
- `design_contract.md` — formal invariants and curriculum policy
- `implementation_mapping.md` — code locations and wiring
- `DELEGATION_V2.md` — **HISTORICAL** (V2 implemented; see §11 for current state)
- `validation_signals.md` — what to measure/plot
- `experimental_findings_2025-12-23.md` — **NEW** experimental results and analysis

---

## 11. Experimental Findings (2025-12-23)

> **Full details:** See `experimental_findings_2025-12-23.md`

### Summary

V2 kick mechanism (logit bias + temperature) was implemented and tested. Key findings:

| Experiment | Unlocks | Final K_eff | Result |
|------------|---------|-------------|--------|
| Kick diagnostic | 2 | 2.12 | **Kick works** — broke single-channel monopoly |
| Multi-unlock (all 10) | 9 | 2.04 | **Coalition stable** — channels 3-9 collapsed |

### The 2-Channel Coalition Phenomenon

1. First unlock (epoch 33): C_0 drops from 100% → 50%, C_1 rises to 50%
2. Subsequent unlocks: Each new channel briefly spikes, then **immediately collapses to 0%**
3. Final state: C_0 (55%) + C_1 (45%) share all 10 digit classes; K_eff ≈ 2

### Root Cause Hypothesis

Curriculum creates asymmetric competition where incumbents have insurmountable advantage:
- First-mover advantage: C_0 + C_1 already offer good reconstruction everywhere
- Gradient asymmetry: ST-Gumbel gives incumbents all forward samples; newcomers only get soft gradients
- Diversity saturation: `-0.05 * entropy` is satisfied at K_eff ≈ 2

### Implication

**Curriculum ≠ simultaneous training** for inducing specialization. Curriculum may be better suited for capacity expansion of an already-specialized model.

---

## 12. Supervisor Meeting Notes (2025-12-03)

> **Source:** MS Teams meeting, ~33 min. Supervisor provides conceptual guidance without codebase access.

### Key Concepts from Supervisor

| Concept | Description |
|---------|-------------|
| **Structural primitives** | Channels = mid-level structures (circle, stroke, loop); latents = fine detail (thickness, slant) |
| **Bucket analogy** | Start with one bucket; open new empty bucket; move structurally distinct subsets; iterate until buckets are "normal filled" |
| **L1 sparsity** | `λ · Σ\|w_c\|` on channel weights — push many toward zero; try before Dirichlet |
| **Perturbation at unlock** | "May require controlled perturbation of network parameters when adding channels so the model actually uses the new capacity" |
| **Local minimum risk** | "Changing the model/prior can trap training in a local minimum" |

### Supervisor's Recommended Experiment

"Fast experiment: increase channels (20-30), increase sparsity strength, observe if channels specialize" — this is **simultaneous training**, not curriculum.

### Translation Gaps

| Supervisor Concept | Implementation Question |
|--------------------|------------------------|
| L1 on channel weights | With ST-Gumbel one-hot forward pass, does L1 on soft q(c\|x) translate? |
| Perturbation | What form? Noise to weights? Reset encoder? Freeze incumbents? |
| Structural primitives | Are FiLM-conditioned channels capable of this? |

### What Supervisor Hasn't Seen

- The 2-channel coalition phenomenon (discovered 2025-12-23)
- ST-Gumbel gradient asymmetry specifics
- Our specific logit-MoG parameterization
