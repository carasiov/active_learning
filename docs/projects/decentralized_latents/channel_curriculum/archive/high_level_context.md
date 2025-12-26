# `high_level_context.md`

## High-level context: why we use a logit-mixture prior and how it connects to the “pots” curriculum

### 1) What we are building

We are building a responsibility-conditioned mixture VAE with **decentralized latent channels**. Each channel is intended to represent a coherent mode (“pot”), and each datapoint should be explained by (mostly) **one** of these pots. The continuous latent inside that pot captures intra-mode variation.

At a high level, the system has two jobs:

1. Decide **which channel** explains a datapoint (the channel variable (c)).
2. Encode **how** it looks within that channel (the continuous latent (z_c)).

If this works, channels become interpretable units that can later be mapped to labels, used for uncertainty/OOD signals, and expanded over time.

---

### 2) The original problem: regularizing (c) so it behaves like “choose a pot”

The channel posterior (q(c\mid x)) is a probability vector over (K) channels. The key requirement from the supervisor is **per-sample sparsity**:

* For a given input (x), (q(c\mid x)) should be **peaky** (near one-hot).
* Otherwise, the model can represent a single datapoint as a mixture of multiple channels (“parts-based mixing”), which undermines the idea that channels correspond to coherent modes.

So the question becomes: how do we push (q(c\mid x)) toward “winner-takes-all” behavior in a stable way?

---

### 3) Why we moved away from Dirichlet / categorical tricks

A natural approach is to place a prior directly on the simplex (Dirichlet-like ideas) or to regularize the categorical distribution (q(c\mid x)) directly. In practice, these approaches can be inconvenient and brittle:

* Discrete sampling (or hard categorical choices) tends to create gradient difficulties.
* Priors on the simplex often require extra approximations or tuning to behave as intended.
* It can be hard to control “how peaky” the distribution becomes without side effects.

The supervisor’s direction was to simplify: prefer **Gaussian-based** priors/penalties so the regularization remains smooth and easy to evaluate.

---

### 4) The key trick: enforce peakiness in logit space (logit-mixture / logistic-normal mixture idea)

Instead of putting a prior directly on the probability vector (q(c\mid x)), we work with **logits**.

Let the encoder output logits:
$$
y(x)\in\mathbb{R}^{K_{\max}}
$$
and define responsibilities by:
$$
q(c\mid x)=\text{softmax}(y(x)).
$$

Now we choose a prior on (y) such that, after softmax, the resulting probability vector is near one-hot.

In the curriculum setting, only a runtime active set of channels is allowed to compete (the active pots). You can still think of the logits as living in $\mathbb{R}^{K_{\max}}$, with selection restricted to the active subset.

#### Logit-mixture prior (axis-aligned mixture of Gaussians)

We use a mixture of isotropic Gaussians in logit space:

* one Gaussian per channel,
* centered at a basis direction:
  $$
  \mu_k = M\,e_k
  $$
  with covariance (\sigma^2 I).

Intuition:

* If (y(x)) lands near (\mu_k), then the (k)-th logit dominates.
* Softmax turns that into “channel (k) wins.”

#### What we optimize in the current implementation

In the current system, logits (y(x)) are treated as deterministic encoder outputs and we add a **negative log prior** penalty:
$$
L_{c,\text{logit\_mog}} = -\lambda\,\mathbb{E}_x\big[\log p_{\text{mix}}(y(x))\big].
$$

This is not a categorical KL and not a Dirichlet prior; it is a smooth log-density evaluation under a Gaussian mixture.

#### What the parameters mean

* (M): separates the mixture modes. Larger (M) tends to make responsibilities more one-hot.
* (\sigma): spreads each mode. Smaller (\sigma) makes selection harder/sharper but can become brittle.
* (\lambda): strength of the regularization. Typically annealed from 0 upward so reconstruction learns first.

A useful mental model: peakiness is mainly governed by the ratio (M/\sigma).

---

### 5) Responsibilities vs routing: why there are two “distributions”

In the implementation there are conceptually two related objects derived from logits:

* **Responsibilities**: a deterministic softmax over logits. This is the stable “belief” distribution used for diagnostics and usage statistics.
* **Routing / selection distribution**: what the decoder uses to weight channel reconstructions (can be softmax or Gumbel-softmax, optionally straight-through).

Important point:

* The logit-mixture penalty is applied on **raw logits before any Gumbel noise**.
* Routing hardness (soft vs straight-through) is an orthogonal choice: it affects how sharply the decoder is routed, but does not change the definition of the logit prior penalty.

---

### 6) Why we need a curriculum (“pots”) on top of logit-mixture regularization

Even if each datapoint selects one channel confidently, if we allow all (K) channels to compete from the beginning, we can get undesirable behavior:

* premature fragmentation (too many channels used too early),
* redundant channels that split arbitrarily,
* difficulty controlling how many channels are actually needed.

The supervisor’s mental model is a **hierarchical splitting process**:

* Start with one pot.
* Train until the representation inside the pot becomes “simple enough” or progress stalls.
* Open a new empty pot and let the model split the data into simpler subsets.
* Repeat until additional pots stop helping.

This curriculum is the mechanism that turns “a mixture model with many components” into “a controlled growth process that discovers structure gradually.”

---

### 7) What “unlocking pots” means operationally

Operationally, the curriculum introduces a runtime notion of an **active channel set**:

* Keep a fixed maximum (K_{\max}) for architecture and tensor shapes.
* Maintain an active subset (\mathcal{A}\subseteq{1,\dots,K_{\max}}).
* Only channels in (\mathcal{A}) are allowed to compete for selection.

Key consistency requirement:

* Masking is applied for **routing** (softmax/Gumbel-softmax) so inactive channels cannot be selected.
* The logit-mixture penalty is evaluated on **finite raw logits** (y(x)), but the mixture sum is restricted to **active** components only. This keeps the penalty well-defined and aligned with the curriculum.

This is the concrete “pots are closed vs open” mechanism.

---

### 8) When to unlock and why a “kick” is needed

The supervisor discussed two practical ideas for unlocking:

* **Stagnation/plateau**: if training progress stalls, the current stage is “done.”
* **Normality inside channels**: a more structural criterion that the latent codes within channels look approximately normal (channel weights can vary; the important part is within-channel simplicity).

In practice, a plateau trigger is often used as a robust first implementation, with “normality” added later as a refinement.

#### Why a kick is needed

When a new channel is unlocked, the optimizer may continue using the old channels because that is a stable local optimum. Therefore, unlocking typically needs a short exploration phase (“kick”), for example:

* temporarily soften routing (higher temperature / no straight-through),
* temporarily bias logits toward the newly unlocked channel,
* temporarily relax the logit-mixture weight to allow redistribution.

The goal of the kick is not to force a split permanently, but to ensure the new pot receives enough data to start specializing.

---

### 9) How to navigate the project documents

* **`design_contract.md`**: the formal specification (math, invariants, curriculum policy).
* **`implementation_mapping.md`**: where the pieces go in the codebase (masking insertion points, hook wiring, metrics, YAML plumbing).
* **`high_level_context.md`** (this document): intuition and rationale, tying together the logit-mixture idea and the curriculum.

For implementation details of the existing logit-MoG regularizer, see `docs/projects/decentralized_latents/channel_curriculum/logit_mog_regularizer.md`.
