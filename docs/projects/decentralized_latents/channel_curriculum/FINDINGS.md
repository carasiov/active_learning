---
status: current
updated: 2025-12-26
purpose: Experimental results and analysis
---

# Experimental Findings: Channel Curriculum Dynamics

> **Date:** 2025-12-23
> **Status:** Active investigation
> **Purpose:** Document curriculum experiments, relate to supervisor guidance, identify next steps

---

## 1. Executive Summary

- **Kick mechanism works** — successfully breaks single-channel monopoly and establishes multi-channel usage
- **Stable 2-channel equilibrium emerges** — subsequent unlocks fail to gain traction; channels 2-9 collapse despite 150+ epochs available
- **No class-based specialization** — both active channels share all 10 digit classes; split appears arbitrary
- **Curriculum ≠ simultaneous training** — fundamentally different optimization landscape; incumbents have insurmountable advantage
- **Supervisor predicted this** — "local minimum" risk when unlocking; suggested perturbation as remedy

---

## 2. Experiments Conducted

| Experiment | Config | Epochs | Unlocks | Final K_eff | Key Finding |
|------------|--------|--------|---------|-------------|-------------|
| Original validation | `mnist_curriculum.yaml` | 100 | 1 (epoch 97) | 1.00 | Unlock too late; kick had no time |
| Kick diagnostic | `mnist_curriculum_kick_diag.yaml` | 50 | 2 | 2.12 | **Kick works** — C1 captured 47% |
| Multi-unlock | `mnist_curriculum_multi_unlock.yaml` | 250 | 9 (all 10 channels) | 2.04 | **Coalition stable** — C0+C1 defend against all newcomers |

**Result directories:**
- `use_cases/experiments/results/mnist_curriculum_kick_diagnostic__mix10-dec-gbl_head_film__20251222_233509/`
- `use_cases/experiments/results/mnist_curriculum_multi_unlock__mix10-dec-gbl_head_film__20251222_235906/`

---

## 3. Key Visualizations & What They Show

### 3a. Channel Latents Grid (multi-unlock experiment)
![Reference: figures/mixture/channel_latents/channel_latents_grid.png]

- **Channels 0 & 1:** Dense, structured latent spaces containing all 10 digit classes
- **Channels 2-9:** Empty (collapsed to prior) — never captured samples despite being unlocked
- **Interpretation:** Only 2 channels learned; other 8 are vestigial

### 3b. Component Usage Evolution
![Reference: figures/mixture/model_evolution.png]

- **Epoch 0-33:** C_0 at 100% (single-channel phase)
- **Epoch 33 (first unlock):** Sharp transition — C_0 drops to ~50%, C_1 rises to ~50%
- **Epoch 47+ (subsequent unlocks):** Each new channel briefly spikes, then **immediately collapses to 0%**
- **Epoch 100-250:** Stable oscillation between C_0 (~55%) and C_1 (~45%)
- **Interpretation:** First unlock establishes duopoly; all subsequent newcomers fail to compete

### 3c. Channel Ownership Heatmap
![Reference: figures/mixture/model_channel_ownership.png]

- Both channels have ownership across all 10 digit classes
- Some differentiation: C_0 favors {0, 2, 8, 9}, C_1 favors {1, 3, 4}
- **No clean class separation** — the 50/50 split is not semantically meaningful
- Channels 2-9: zero ownership (black)

---

## 4. Mathematical Interpretation

### Why Does a 2-Channel Equilibrium Form?

1. **Kick breaks monopoly:** Logit bias (+5.0) shifts P(c=new|x) from ~0 to competitive, allowing C_1 to receive samples and learn

2. **Diversity reward saturates at K=2:** With `-0.05 * entropy`, the reward is ~satisfied at H ≈ 0.69 nats (2-channel uniform)

3. **Reconstruction capacity sufficient:** 2D latent × 2 channels × FiLM decoder may already have enough capacity for MNIST; no pressure for more

### Why Can't Channels 3+ Compete?

1. **First-mover advantage:** C_0 and C_1 already offer good reconstruction everywhere; newcomers can't offer better

2. **Gradient asymmetry (ST-Gumbel):** Forward pass is argmax — incumbents receive all samples. Newcomers only get soft gradient signal, which is insufficient to bootstrap

3. **Kick window insufficient against coalition:** 12 epochs competing against 2 established channels (vs. 1 in the first unlock) — the bar is higher

### Curriculum vs. Simultaneous Training

| Aspect | Curriculum | Simultaneous |
|--------|------------|--------------|
| Initial state | 1 channel trained on everything | K channels compete from random init |
| Competition | Asymmetric (newcomer vs incumbent) | Symmetric |
| Optimization landscape | Sequential local minima | Single joint optimization |
| Expected outcome | Stable small coalition | Potential for K-way specialization |

---

## 5. Supervisor Guidance & Our Interpretation

### 5a. Supervisor's Theoretical Input (Dec 3 meeting)

The supervisor provides conceptual/mathematical direction without codebase access. Key concepts from his guidance:

| Concept | His Description |
|---------|-----------------|
| **Bucket analogy** | Start with one bucket, data coarsely sorted; open new empty bucket; move structurally distinct subsets; iterate until buckets are "normal filled" |
| **Structural primitives** | Channels = mid-level structures (circle, stroke, loop); latents = fine detail (thickness, slant) |
| **L1 sparsity** | ~~`λ · Σ|w_c|` on channel weights~~ → **Clarified:** means Logit-MoG (per-sample peakiness via Gaussian mixture on logits). See §5f. |
| **Perturbation at unlock** | "May require controlled perturbation of network parameters when adding channels so the model actually uses the new capacity" |
| **Local minimum risk** | "Changing the model/prior can trap training in a local minimum" |
| **Stopping criterion** | "Check how many channels are active; stop opening more if they are not being used" |

### 5b. How We Operationalized These Concepts

| Supervisor Concept | Our Implementation |
|--------------------|-------------------|
| Curriculum / bucket filling | Plateau-based unlock policy; k_active starts at 1, increments on validation loss plateau |
| Sparsity / per-sample one-hot | ST-Gumbel for hard routing; logit-MoG prior encouraging peaky q(c\|x) |
| "Kick" for new channel usage | Logit bias (+5.0 to newly unlocked channel) + temperature increase (5.0) during kick window |
| Diversity pressure | `-0.05 * batch_entropy` — rewards spreading usage across channels |
| Perturbation | **Not yet implemented** — current kick is additive bias, not weight perturbation |

### 5c. Where Supervisor's Predictions Match Our Findings

| His Prediction | Our Observation | Status |
|----------------|-----------------|--------|
| "Local minimum when unlocking" | 2-channel coalition stable against all newcomers | **Confirmed** |
| "May need perturbation" | Logit bias alone insufficient for K>2 | **Suggests he's right** |
| "Stop if channels not being used" | Channels 3-9 never used despite unlocking | Would trigger his stopping rule |

### 5d. What Supervisor Hasn't Seen (implementation specifics)

- ST-Gumbel produces one-hot forward pass (not soft mixing)
- Logit-MoG geometry: axis-aligned Gaussians at `M * e_k` in logit space
- Specific kick mechanism (logit bias + temperature, not weight perturbation)
- The 2-channel coalition phenomenon and evolution dynamics

### 5e. Translation Gaps / Open Questions

| Supervisor Concept | Implementation Question |
|--------------------|------------------------|
| ~~"L1 on channel weights"~~ | **RESOLVED (2025-12-26):** Supervisor meant Logit-MoG, not literal L1. See §5f. |
| "Perturbation when unlocking" | What form? Noise to incumbent weights? Reset parts of encoder? Freeze incumbents temporarily? |
| "Channels look normal" | Need to implement normality criterion for unlock/stop decisions |
| "Structural primitives" | Are FiLM-conditioned channels capable of representing this? Or is architecture insufficient? |

### 5f. Clarification from High-Level Agent (2025-12-26)

After consulting the high-level agent (ChatGPT) with the supervisor's original German notes, we clarified:

**What "L1 sparsity" actually meant:**
> "Regularize pre-softmax logits y(x) ∈ R^K with a mixture of Gaussians centered at M·e_k,
> so post-softmax r(x) becomes peaky (one-hot-ish). This avoids Dirichlet/categorical
> sampling complications while using only Gaussian KL terms."

This is **exactly** Logit-MoG, which is already implemented.

**Mechanism distinction (critical insight):**

| Mechanism | Target | Effect | Config |
|-----------|--------|--------|--------|
| Logit-MoG | Per-sample | Each r(x) → one-hot | `c_regularizer: logit_mog` |
| Entropy reward | Global (batch) | Usage spread across channels | `component_diversity_weight < 0` |

These are **complementary**, not alternatives:
- Logit-MoG ensures each sample strongly prefers one channel
- Entropy reward ensures different samples use different channels

**Why literal L1 on r(x) is useless:**
Since r(x) = softmax(y(x)), we have ||r(x)||₁ = 1 always. L1 penalty provides no gradient signal.

---

## 6. Conclusions

### What Works
- Curriculum + kick successfully breaks single-channel monopoly
- Usage redistribution from 100/0 to ~50/50 is achievable
- Plateau-based unlock triggers appropriately

### What Doesn't Work (Yet)
- Scaling beyond 2 channels — coalition defends against newcomers
- Class-based specialization — both channels share all classes
- Semantic structure — split appears arbitrary, not structural

### Root Cause Hypothesis
Curriculum creates asymmetric competition where incumbents have insurmountable advantage. This is a **fundamentally different optimization landscape** than simultaneous training where all channels start equal.

---

## 7. Next Steps (Prioritized)

### Immediate: Simultaneous Training Baseline
- [x] ~~Implement L1 sparsity~~ → **Already done** via Logit-MoG (see §5f)
- [ ] Run simultaneous training with K=20 (`mnist_simultaneous_k20.yaml`)
- [ ] Observe if >2 channels emerge without curriculum
- **Purpose:** Determine if 2-channel coalition is curriculum-specific or architectural

### If Simultaneous Training Shows K>2 Specialization:
- [ ] Problem is curriculum-specific (incumbent advantage)
- [ ] Implement perturbation at unlock (noise injection, LR decay for incumbents)
- [ ] Test hybrid: simultaneous init → curriculum expansion

### If Simultaneous Training Also Shows K≈2:
- [ ] Problem is architectural (Logit-MoG geometry, capacity, etc.)
- [ ] Revisit M/σ ratio, latent_dim, decoder conditioning
- [ ] Consult supervisor on conceptual model revision

---

## 8. Questions for Supervisor (Next Meeting)

1. **On perturbation:** What form do you envision? Noise to weights? Reset? Freeze incumbents? Or something else?

2. ~~**On L1 vs. entropy:**~~ **RESOLVED (2025-12-26):** L1 = Logit-MoG (per-sample), entropy = global diversity. Complementary mechanisms. See §5f.

3. **On the 2-channel phenomenon:** We see stable coalition that defends against newcomers. Is this an acceptable intermediate state, or must we break it?

4. **On structural primitives:** The channels we observe share all classes. Is it possible they represent structural primitives (as you described) but we can't see it because we color by class? What visualization would reveal structural specialization?

5. **On hybrid approach:** Would you support: (a) simultaneous training for initial specialization, then (b) curriculum for capacity expansion? Or is pure curriculum the research target?

---

## 9. Appendix: Configuration Details

### Kick Diagnostic Config (successful 2-channel)
```yaml
curriculum:
  k_active_init: 1
  unlock:
    patience_epochs: 5
    min_delta: 2.0
  kick:
    epochs: 15
    gumbel_temperature: 5.0
    logit_bias: 5.0
```

### Multi-Unlock Config (coalition phenomenon)
```yaml
max_epochs: 250
curriculum:
  k_active_init: 1
  unlock:
    patience_epochs: 5
    min_delta: 1.5
    cooldown_epochs: 2
  kick:
    epochs: 12
    gumbel_temperature: 5.0
    logit_bias: 5.0
```

### Key Model Settings (both experiments)
```yaml
num_components: 10
latent_dim: 2
prior_type: mixture
decoder_conditioning: film
use_straight_through_gumbel: true
c_regularizer: logit_mog
component_diversity_weight: -0.05
```

---

*Document created: 2025-12-23*
*Last updated: 2025-12-23*
