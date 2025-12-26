---
status: current
updated: 2025-12-26
purpose: Mathematical specification and supervisor decisions
---

# Channel Curriculum — Specification

## 1. Supervisor Decisions (Authoritative)

These are settled decisions. Do not revisit without explicit direction.

| Decision | Rationale |
|----------|-----------|
| **Logit-space regularization** | Encode into R^K, softmax to simplex, regularize pre-softmax logits with mixture-of-Gaussians |
| **Per-sample winner-takes-all** | One channel dominates per x; no parts-based mixing |
| **Curriculum unlock** | Start with K_active=1, unlock channels when progress stalls |
| **Gaussian-only priors** | Avoid Dirichlet/categorical KL complications |

**Supervisor's mental model (Dec 3, 2025):**
- Channels = structural primitives (circle, stroke, loop)
- Latents = fine detail (thickness, slant)
- "Bucket analogy": coarsely sort into one bucket, then split into new buckets iteratively

**Clarification (2025-12-26):** "L1 sparsity" = Logit-MoG (per-sample peakiness).
See `FINDINGS.md` §5f for details on mechanism distinction.

---

## 2. Mathematical Formulation

### Variables

For each input x with K_max channels:

| Variable | Definition |
|----------|------------|
| `y(x) ∈ R^K_max` | Encoder logits (pre-softmax) |
| `r(x) = softmax(y(x))` | Responsibilities (soft assignment) |
| `s(x)` | Routing distribution (softmax or Gumbel-softmax) |
| `q_φ(z_k \| x) = N(μ_k(x), σ_k²(x))` | Per-channel latent posterior |
| `x̂_k = f_θ(z_k, k)` | Per-channel reconstruction |
| `x̂ = Σ_k s_k(x) x̂_k` | Final reconstruction |

### Objective

```
L(x) = L_recon(x, x̂)
     + β_z · Σ_k KL(q(z_k|x) || N(0,I))   # all k, not responsibility-weighted
     + L_c(x)                              # channel selection regularization
```

### Logit-MoG Regularization (L_c)

Mixture-of-Gaussians prior on logits:

```
p_mix(y) = (1/|A|) Σ_{k∈A} N(y; μ_k, σ²I)
where μ_k = M · e_k  (axis-aligned)

L_c(x) = -λ · log p_mix(y(x))
```

Parameters:
- `M` (c_logit_prior_mean): separation between modes
- `σ` (c_logit_prior_sigma): spread of each mode
- `λ` (c_logit_prior_weight): regularization strength
- Peakiness governed by ratio M/σ

### Channel Regularization Mechanisms (Per-sample vs Global)

| Mechanism | Target | Effect | Config |
|-----------|--------|--------|--------|
| **Logit-MoG** | Per-sample | Each r(x) → one-hot | `c_regularizer: logit_mog` |
| **Entropy reward** | Global (batch) | Usage spread across channels | `component_diversity_weight < 0` |

These are **complementary**:
- Logit-MoG ensures each sample strongly prefers one channel
- Entropy reward ensures different samples use different channels

Note: Literal L1 on r(x) is useless since ||softmax(y)||₁ = 1 always.

---

## 3. Curriculum Mechanism

### Active Channel Set

- `A_t ⊆ {1..K_max}` — channels open at stage t
- Start: `A_0 = {1}` (single channel)
- Growth: `A_{t+1} = A_t ∪ {k_new}` on unlock

### Masking Invariants

1. **Routing**: Inactive channels masked to -∞ before softmax
2. **Logit-MoG**: Computed on finite raw logits; mixture sum restricted to A_t
3. **Reconstruction**: Inactive channels contribute 0 weight

### Unlock Policy

**Plateau-based** (current implementation):
- Monitor `val_reconstruction_loss`
- Unlock when improvement < threshold for `patience_epochs`

### Kick Mechanism

After unlock, for `kick.epochs`:
- Logit bias: +`kick.logit_bias` to newly unlocked channel
- Temperature: increase to `kick.gumbel_temperature`
- ST disabled: soft routing during kick

---

## 4. Key Invariants

| Invariant | Enforcement |
|-----------|-------------|
| Static K_max shape | Curriculum operates by masking, not resizing |
| Routing -∞ masking | Only for softmax/Gumbel distributions |
| Logit-MoG finite | Never feed -∞ into distance computations |
| Metrics use masked responsibilities | Usage stats reflect what can be selected |

---

## 5. Known Issues

### 2-Channel Coalition (discovered 2025-12-23)

**Symptom:** First unlock works (K_eff: 1→2), subsequent unlocks fail (K_eff stays ~2)

**Cause:** Curriculum creates asymmetric competition; incumbents have insurmountable advantage

**Status:** Open. See `FINDINGS.md` for experimental evidence.

---

## 6. References

| Topic | Location |
|-------|----------|
| Curriculum controller | `src/rcmvae/application/curriculum/controller.py` |
| Curriculum hooks | `src/rcmvae/application/curriculum/hooks.py` |
| Logit-MoG prior | `src/rcmvae/domain/priors/mixture.py` |
| Network routing | `src/rcmvae/domain/network.py` |
| Config schema | `src/rcmvae/domain/config.py` |

For historical context: `archive/high_level_context.md`, `archive/DELEGATION_V2.md`
