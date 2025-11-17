# Training Hub Redesign

## Overview

Redesign the Training Hub to provide a clean, organized, and contextually-aware training configuration interface. Parameters should be grouped logically, with prior-specific settings shown conditionally based on the model's architecture.

## Design Philosophy

**Core Principles**:
1. **Contextual Intelligence**: Show only relevant parameters for the current model's prior type
2. **Information Hierarchy**: Group parameters by purpose, not alphabetically
3. **Visual Coherence**: Match the overall dashboard design (typography, colors, spacing)
4. **Progressive Disclosure**: Essential parameters visible, advanced in collapsible sections
5. **Semantic Clarity**: Use terminology from conceptual model (channels, responsibilities, τ-classifier)

**Visual Language** (from existing app):
- **Primary Color**: #C10A27 (red accent, buttons)
- **Secondary Color**: #45717A (teal, secondary actions)
- **Neutral Dark**: #000000 (headings)
- **Neutral Medium**: #6F6F6F (body text, labels)
- **Neutral Light**: #C6C6C6 (borders)
- **Background**: #ffffff (cards), #f5f5f5 (page), #fafafa (sections)
- **Font**: 'Open Sans', Verdana, sans-serif
- **Monospace**: ui-monospace, monospace (numbers, technical values)

## Layout Structure

```
┌──────────────────────────────────────────────────────────────┐
│  Training Hub                                                 │
│  Model: experiment-name-123                                   │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────┐  ┌───────────────────────────────┐ │
│  │ Left Panel (40%)    │  │ Right Panel (60%)             │ │
│  │                     │  │                               │ │
│  │ Architecture        │  │ Training Progress             │ │
│  │ (read-only summary) │  │ - Loss curves                 │ │
│  │                     │  │ - Metrics                     │ │
│  │ Training Setup      │  │ - Status                      │ │
│  │ - Epochs            │  │                               │ │
│  │ - Learning rate     │  │ Recent Runs                   │ │
│  │ - Batch size        │  │ - Last 5 runs                 │ │
│  │                     │  │ - Quick access                │ │
│  │ Loss Weights        │  │                               │ │
│  │ - Reconstruction    │  │ Mixture Diagnostics           │ │
│  │ - KL divergence     │  │ (if mixture prior)            │ │
│  │ - Classification    │  │ - π values chart              │ │
│  │                     │  │ - Component usage             │ │
│  │ [Prior-Specific]    │  │                               │ │
│  │ (conditional)       │  │                               │ │
│  │                     │  │                               │ │
│  │ Regularization      │  │                               │ │
│  │ (collapsible)       │  │                               │ │
│  │                     │  │                               │ │
│  │ [Train Model]       │  │                               │ │
│  │ [Stop Training]     │  │                               │ │
│  │ [Configure More...] │  │                               │ │
│  └─────────────────────┘  └───────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## Left Panel: Configuration Sections

### 1. Architecture Summary (Read-Only, Always Visible)

**Purpose**: Show locked structural parameters

```
┌────────────────────────────────────────────┐
│ Architecture (fixed at creation)           │
├────────────────────────────────────────────┤
│ Prior:       Mixture (10 components)      │
│ Encoder:     Convolutional                │
│ Latent Dim:  2                            │
│ Recon Loss:  BCE (binary)                 │
│                                            │
│ Component-aware decoder: Yes              │
│ Heteroscedastic decoder: No               │
└────────────────────────────────────────────┘
```

**Styling**:
- Light gray background (#fafafa)
- Small text (13px)
- Subtle border
- Icon: 🔒 next to title

### 2. Training Setup (Always Visible)

**Purpose**: Core training loop configuration

```
┌────────────────────────────────────────────┐
│ Training Setup                             │
├────────────────────────────────────────────┤
│                                            │
│ Epoch Budget              [200 ▼]         │
│ How many epochs to train                   │
│                                            │
│ Early Stop Patience       [20  ▼]         │
│ Stop after N epochs without improvement    │
│                                            │
│ Learning Rate             [0.001 ▼]       │
│ Adam optimizer learning rate               │
│                                            │
│ Batch Size                [128 ▼]         │
│ Samples per optimization step              │
│                                            │
│ Random Seed               [42  ▼]         │
│ For reproducibility                        │
│                                            │
└────────────────────────────────────────────┘
```

**Fields**:
- `max_epochs` (1-500, default: 200)
- `patience` (1-100, default: 20)
- `learning_rate` (1e-5 to 1e-1, default: 1e-3)
- `batch_size` (32-4096 step 32, default: 128)
- `random_seed` (0-10000, default: 42)

**Validation**:
- Monitor metric selector (if needed): "auto" | "loss" | "classification_loss"

### 3. Loss Weights (Always Visible)

**Purpose**: Balance loss components

```
┌────────────────────────────────────────────┐
│ Loss Weights                               │
├────────────────────────────────────────────┤
│                                            │
│ Reconstruction            [1.0  ▼]        │
│ Pixel reconstruction term (BCE scale)      │
│                                            │
│ KL Divergence            [1.0  ▼]         │
│ Latent regularization (β in β-VAE)        │
│                                            │
│ Classification           [1.0  ▼]         │
│ Supervised label loss weight               │
│                                            │
└────────────────────────────────────────────┘
```

**Fields**:
- `recon_weight` (0.0-10000, default depends on reconstruction_loss)
  - BCE: default 1.0
  - MSE: default 500.0
- `kl_weight` (0.0-20.0, default: 1.0)
- `label_weight` (0.0-50.0, default: 1.0)

**Helper Text**:
- Show current reconstruction loss type (from architecture)
- Recommend typical weights for that loss type

### 4. Prior-Specific Settings (Conditional)

#### 4A. Standard Prior (No Additional Settings)

**Show**: Simple message
```
┌────────────────────────────────────────────┐
│ Prior: Standard                            │
├────────────────────────────────────────────┤
│ Simple N(0,I) Gaussian prior.              │
│ No additional configuration needed.        │
└────────────────────────────────────────────┘
```

#### 4B. Mixture Prior Settings

```
┌────────────────────────────────────────────┐
│ Mixture Prior Configuration                │
├────────────────────────────────────────────┤
│                                            │
│ ☑ τ-Classifier                            │
│ Use latent-only classification via r×τ     │
│                                            │
│ τ Smoothing (α₀)         [1.0  ▼]        │
│ Laplace smoothing for unseen c→y pairs    │
│                                            │
│ Component KL Weight      [1.0  ▼]         │
│ Weight on KL(q(c|x) || π)                 │
│                                            │
│ KL Anneal Epochs         [0    ▼]         │
│ Ramp component KL from 0 over N epochs    │
│                                            │
│ ☑ Learnable π                             │
│ Allow mixture weights to adapt             │
│                                            │
│ Usage Entropy Weight     [-0.1 ▼]         │
│ Entropy H[p̂_c]: negative = reward diversity │
│                                            │
│ ─── Advanced Mixture Options ▼ ───        │
│                                            │
│ Dirichlet α (π prior)    [blank]          │
│ MAP regularization strength (optional)     │
│                                            │
│ Dirichlet Weight         [1.0  ▼]         │
│ Scaling for Dirichlet penalty              │
│                                            │
│ Top-M Gating             [0    ▼]         │
│ Reconstruct with top M components (0=all)  │
│                                            │
│ Soft Embedding Warmup    [0    ▼]         │
│ Use soft embeddings for first N epochs    │
│                                            │
└────────────────────────────────────────────┘
```

**Fields**:
- `use_tau_classifier` (boolean, default: True)
- `tau_smoothing_alpha` (>0, default: 1.0)
- `kl_c_weight` (0.0-10.0, default: 1.0)
- `kl_c_anneal_epochs` (0-500, default: 0)
- `learnable_pi` (boolean, default: True)
- `component_diversity_weight` (-10.0 to 10.0, default: -0.1)
  - Label: "Usage Entropy Weight"
  - Description: "Entropy H[p̂_c]: negative = reward diversity"
- **Advanced** (collapsible):
  - `dirichlet_alpha` (0.1-10.0 or blank, default: None)
  - `dirichlet_weight` (0.0-10.0, default: 1.0)
  - `top_m_gating` (0-num_components, default: 0)
  - `soft_embedding_warmup_epochs` (0-500, default: 0)

**Semantic Notes**:
- Use terminology from conceptual model: "channels" = "components", "responsibilities" = r, "τ-classifier"
- Usage entropy: H[p̂_c] where p̂_c is empirical component usage. Negative weight = entropy reward (encourage diverse usage)
- Show component count from architecture summary

#### 4C. VampPrior Settings

```
┌────────────────────────────────────────────┐
│ VampPrior Configuration                    │
├────────────────────────────────────────────┤
│                                            │
│ ☑ τ-Classifier                            │
│ Use latent-only classification via r×τ     │
│                                            │
│ τ Smoothing (α₀)         [1.0  ▼]        │
│ Laplace smoothing for unseen c→y pairs    │
│                                            │
│ Component KL Weight      [1.0  ▼]         │
│ Weight on KL(q(c|x) || π)                 │
│                                            │
│ KL Anneal Epochs         [0    ▼]         │
│ Ramp component KL from 0 over N epochs    │
│                                            │
│ Usage Entropy Weight     [-0.1 ▼]         │
│ Entropy H[p̂_c]: negative = reward diversity │
│                                            │
│ ─── VampPrior-Specific ▼ ───              │
│                                            │
│ KL Samples (MC)          [1    ▼]         │
│ Monte Carlo samples for KL estimation      │
│                                            │
│ Pseudo-Input LR Scale    [0.1  ▼]         │
│ Learning rate multiplier for u_k           │
│                                            │
│ Note: Pseudo-inputs initialized at model   │
│ creation. Π is uniform for VampPrior.     │
│                                            │
└────────────────────────────────────────────┘
```

**Fields**:
- `use_tau_classifier` (boolean, default: True)
- `tau_smoothing_alpha` (>0, default: 1.0)
- `kl_c_weight` (0.0-10.0, default: 1.0)
- `kl_c_anneal_epochs` (0-500, default: 0)
- `component_diversity_weight` (-10.0 to 10.0, default: -0.1)
  - Label: "Usage Entropy Weight"
  - Description: "Entropy H[p̂_c]: negative = reward diversity"
- **VampPrior-Specific**:
  - `vamp_num_samples_kl` (1-10, default: 1)
  - `vamp_pseudo_lr_scale` (0.01-1.0, default: 0.1)

**Info Box**:
- Note that π is uniform (not learnable) for VampPrior
- Pseudo-inputs are initialized at creation time using method specified in architecture

#### 4D. Geometric MoG Settings

```
┌────────────────────────────────────────────┐
│ Geometric MoG Configuration                │
├────────────────────────────────────────────┤
│                                            │
│ ☑ τ-Classifier                            │
│ Use latent-only classification via r×τ     │
│                                            │
│ τ Smoothing (α₀)         [1.0  ▼]        │
│ Laplace smoothing for unseen c→y pairs    │
│                                            │
│ Component KL Weight      [1.0  ▼]         │
│ Weight on KL(q(c|x) || π)                 │
│                                            │
│ ☑ Learnable π                             │
│ Allow mixture weights to adapt             │
│                                            │
│ Usage Entropy Weight     [-0.1 ▼]         │
│ Entropy H[p̂_c]: negative = reward diversity │
│                                            │
│ Note: Components arranged geometrically   │
│ (circle/grid) with fixed spacing. See     │
│ architecture summary for arrangement.      │
│                                            │
└────────────────────────────────────────────┘
```

**Fields**:
- `use_tau_classifier` (boolean, default: True)
- `tau_smoothing_alpha` (>0, default: 1.0)
- `kl_c_weight` (0.0-10.0, default: 1.0)
- `learnable_pi` (boolean, default: True)
- `component_diversity_weight` (-10.0 to 10.0, default: -0.1)
  - Label: "Usage Entropy Weight"
  - Description: "Entropy H[p̂_c]: negative = reward diversity"

**Info Note**:
- Factual note about geometric arrangement being structural (locked at creation)

### 5. Regularization (Collapsible, Always Available)

```
┌────────────────────────────────────────────┐
│ Regularization ▼                           │
├────────────────────────────────────────────┤
│                                            │
│ Weight Decay             [0.0001 ▼]       │
│ L2 regularization (AdamW)                  │
│                                            │
│ Gradient Clip Norm       [1.0  ▼]         │
│ Clip gradients by global norm (0=off)     │
│                                            │
│ Classifier Dropout       [0.2  ▼]         │
│ Dropout rate in classifier head            │
│                                            │
│ ─── Heteroscedastic Decoder ───           │
│ (only if enabled in architecture)          │
│                                            │
│ σ Min                    [0.05 ▼]         │
│ Minimum decoder variance                   │
│                                            │
│ σ Max                    [0.5  ▼]         │
│ Maximum decoder variance                   │
│                                            │
│ ─── Contrastive Learning ───              │
│                                            │
│ ☐ Enable Contrastive                      │
│ Add supervised contrastive loss            │
│                                            │
│ Contrastive Weight       [0.0  ▼]         │
│ Scaling for contrastive term               │
│                                            │
└────────────────────────────────────────────┘
```

**Fields**:
- `weight_decay` (0.0-0.1, default: 1e-4)
- `grad_clip_norm` (0.0-100.0, default: 1.0, 0=disabled)
- `dropout_rate` (0.0-0.8, default: 0.2)
- **Heteroscedastic** (if architecture has it):
  - `sigma_min` (1e-5 to 5.0, default: 0.05)
  - `sigma_max` (0.01-10.0, default: 0.5)
- **Contrastive**:
  - `use_contrastive` (boolean, default: False)
  - `contrastive_weight` (0.0-50.0, default: 0.0)

### 6. Action Buttons (Always Visible)

```
┌────────────────────────────────────────────┐
│                                            │
│ [  Train Model for N Epochs  ]            │
│ Start/resume training                      │
│                                            │
│ [  Stop Training  ]                        │
│ Gracefully halt (if running)               │
│                                            │
│ [  Full Configuration...  ]               │
│ Open detailed config editor                │
│                                            │
└────────────────────────────────────────────┘
```

**Buttons**:
- **Train**: Primary red (#C10A27), full width, shows "Training..." when active
- **Stop**: Secondary gray, full width, disabled when idle
- **Full Config**: Outline button, links to `/model/{id}/configure-training`

## Implementation Strategy

**Note on Mockup Values**: The values shown in mockups (e.g., epochs=200, kl_weight=1.0) are recommendations for typical mixture model workflows. The actual implementation should:
1. Pull current values from `ModelState.config` (the source of truth)
2. Display the model's actual configured values, not hardcoded defaults
3. Some config defaults (e.g., `recon_weight=500` for MSE) will differ from what's shown in mockups (which assume BCE)

### Phase 1: Architecture Summary & Parameter Grouping

1. Add read-only architecture display at top of left panel
2. Group existing parameters into logical sections (Training Setup, Loss Weights)
3. Update styling to match design system

### Phase 2: Conditional Prior Sections

1. Implement prior type detection from `ModelState.config.prior_type`
2. Create component for each prior type's settings
3. Add conditional rendering logic
4. Test with each prior type

### Phase 3: Advanced Options & Polish

1. Implement collapsible sections (Regularization, Advanced Mixture)
2. Add helper text and tooltips
3. Validate parameter interdependencies
4. Polish spacing, typography, colors

### Phase 4: Full Configuration Link

1. Keep existing `/configure-training` page for power users
2. Add "Full Configuration..." button linking to it
3. Ensure both interfaces sync state properly

## Parameter Organization by Purpose

### Essential (Always Visible)
- Training loop: epochs, patience, learning rate, batch size
- Loss balance: recon_weight, kl_weight, label_weight

### Prior-Specific (Conditional)
- **Mixture**: τ-classifier, component KL, usage entropy, learnable π
- **VampPrior**: τ-classifier, component KL, usage entropy, pseudo-input LR
- **Geometric**: τ-classifier, component KL, usage entropy, learnable π
- **Standard**: (none - just a note)

### Advanced (Collapsible)
- Regularization: weight decay, grad clip, dropout
- Heteroscedastic: sigma bounds (if enabled)
- Contrastive: enable + weight
- Mixture advanced: Dirichlet prior, gating, warmup

## Semantic Alignment with Conceptual Model

**Terminology Mapping**:
- ✅ "Component" or "Channel" (not "cluster" or "mode")
- ✅ "Responsibilities" r (not "assignments")
- ✅ "τ-Classifier" (channel→label map)
- ✅ "Usage Entropy" H[p̂_c] (not "diversity")
- ✅ "π" (mixture weights)
- ✅ "Latent-only classification" (via r×τ)

**Conceptual Guidance**:
- Explain that τ-classifier uses responsibilities to classify
- Note that usage entropy H[p̂_c]: negative weight = entropy reward (encourage diverse component usage)
- Clarify that component-aware decoder was set at creation
- Link parameters to objectives (KL on c, entropy on usage)

## Visual Design Spec

### Typography
```css
{
  /* Section headings */
  fontSize: "17px",
  fontWeight: "700",
  color: "#000000",
  fontFamily: "'Open Sans', Verdana, sans-serif",

  /* Parameter labels */
  fontSize: "14px",
  fontWeight: "600",
  color: "#6F6F6F",
  fontFamily: "'Open Sans', Verdana, sans-serif",

  /* Helper text */
  fontSize: "12px",
  fontWeight: "400",
  color: "#6F6F6F",
  fontFamily: "'Open Sans', Verdana, sans-serif",

  /* Numeric values */
  fontSize: "14px",
  fontFamily: "ui-monospace, monospace",
}
```

### Colors
```css
{
  /* Primary action */
  backgroundColor: "#C10A27",
  color: "#ffffff",

  /* Secondary action */
  backgroundColor: "#45717A",
  color: "#ffffff",

  /* Card backgrounds */
  backgroundColor: "#ffffff",
  border: "1px solid #C6C6C6",

  /* Page background */
  backgroundColor: "#f5f5f5",

  /* Input fields */
  border: "1px solid #C6C6C6",
  borderRadius: "6px",

  /* Architecture summary (locked) */
  backgroundColor: "#fafafa",
  border: "1px solid #E6E6E6",
}
```

### Spacing
```css
{
  /* Section margins */
  marginBottom: "24px",

  /* Card padding */
  padding: "24px",

  /* Input groups */
  marginBottom: "16px",

  /* Label-input gap */
  marginBottom: "6px",

  /* Input padding */
  padding: "10px 12px",
}
```

## Migration Notes

### For Users
- Training Hub now shows only relevant parameters for your model's prior type
- Structural parameters (prior, encoder, latent dim) are read-only - shown at top
- Full configuration editor still available via "Full Configuration..." button
- All settings persist between training runs

### For Developers
- Prior-specific sections use conditional rendering based on `config.prior_type`
- Architecture summary pulls from locked structural parameters
- Helper text uses terminology from conceptual model
- Visual design matches homepage and overall app style
