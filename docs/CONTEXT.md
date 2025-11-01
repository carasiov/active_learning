# Context and Motivation

**Core intent**: Build an interactive tool **as a showcase** to explore and label data with few labels, using strong unsupervised structure learning. Don't just classify—tell me _how sure_ you are and when you've _never seen_ something like this before.

---

## 1. Motivating Application

**Background domain**: High-dimensional text/notification embeddings (≈1,500D device notifications/error messages) where:
- Only a small subset has expert labels (labeling is expensive)
- New messages appear as slight rephrasings of existing ones
- System must rely on semantic proximity in embedding space
- **MNIST is the visual proof-of-concept** for development/demonstration

**Key challenge**: Data is too high-dimensional to cluster directly → need learned latent space (12–64D) that is structured, clusterable, and visualizable (2D).

---

## 2. Core Capabilities

The system must:
1. **Encode and cluster** unlabeled data into meaningful latent space using a VAE
2. **Let users label** few points directly in 2D visualization
3. **Instantly propagate** labels through latent structure (few labels → many colored points)
4. **Show confidence** and **flag OOD/unknown** points with uncertainty scores
5. **Let users steer training** via sliders (reconstruction/KL/classification/prior weights)
6. **Propose points to label** using uncertainty scores → active learning loop
7. **Support interactive training** (1–5 epochs at a time) with immediate visualization updates

---

## 3. Architecture Overview

### 3.1 Generative Core: Component-Based VAE

**Discrete component variable** c ∈ {1,...,K}:
- **Inference**: q(c,z|x) = q(c|x)·q(z|x,c) where q(c|x) are responsibilities r_c(x)
- **Generation**: sample c ~ π (learned, sparse) → z ~ N(0,I) → x ~ p(x|z,c)
- **Decoder is component-conditional** so components become semantically meaningful

**Why components?** Standard VAE with N(0,I) prior pushes everything to origin → bad for interactive labeling. Components create **clustered but connected** latent space.

### 3.2 Label Store (External to Model)

Keep counts s[c][y] = how much component c has seen label y. With Dirichlet smoothing α₀:

```
τ_{c,y} = (s_{c,y} + α₀) / Σ_y' (s_{c,y'} + α₀)
```

**Classification without separate head**:
```
p(y|x) = Σ_c r_c(x) · τ_{c,y}
```

This connects unlabeled structure directly to labels—exactly as designed in meetings.

### 3.3 Uncertainty: Aleatoric vs Epistemic

**Key distinction**:
- **Aleatoric** (data noise) → modeled in decoder output (μ, σ per pixel)
- **Epistemic** (model uncertainty) → visible in latent/component space

**Why decoder variance matters**: If decoder only outputs means (MSE loss), it's forced to pack data noise into latent → uncertainties get confounded. Solution: decoder outputs (μ, log σ) with Gaussian NLL loss.

### 3.4 OOD / Query Score

```
s_OOD(x) = 1 - max_c (r_c(x) · max_y τ_{c,y})
```

Interpretation:
- If any component wants x strongly (high r_c) AND that component knows a label well (high max_y τ_{c,y}) → low score → in-distribution
- Otherwise → high score → **suggest for labeling**

### 3.5 Sparsity: Use Only Needed Components

Track empirical usage: p̂(c) = (1/N) Σ_i r_c(x_i)

Penalize entropy to concentrate on few active components:
```
L_sparse = λ Σ_c p̂(c) log(p̂(c) + ε)
```

Leaves unused components available for future labels/OOD data.

---

## 4. Training Strategy (Curriculum)

**Phase 1 – Reconstruction first**
- KL weight ≈ 0 (get "non-bullshit" reconstructions)

**Phase 2 – Turn on prior**
- Gradually increase KL/prior weight
- Optionally switch from standard Normal to VampPrior/mixture

**Phase 3 – Add classification/contrastive**
- Add supervised term when labels exist
- Add contrastive term to tighten clusters semantically

**Throughout – Interactive/short runs**
- User presses "train 1/5 epochs"
- Model updates plots & losses
- User tweaks sliders (active learning for hyperparameters)

---

## 5. Priors: Why "Fancy"?

**Problem**: Standard N(0,I) prior over-compresses → bad latent structure for interactive use.

**Solution directions**:
1. **VampPrior**: p(z) = (1/K) Σ_k q(z|u_k) where u_k are learnable pseudo-inputs
   - Creates data-shaped prior
   - Supervisor suggested regularizing pseudo-latents toward N(0,I) for global shape

2. **Mixture prior**: K channels of standard normals with learned mixing weights
   - Natural fit with component architecture
   - Enable sparsity via mixing weight regularization

Both create **clustered but connected** latent spaces suitable for few-shot labeling.
