# Dashboard Redesign: Model Creation & Training Hub
## Complete Implementation Guide

> **Purpose**: Single source of truth for implementing the model creation and training hub redesign. Consolidates all design decisions, specifications, and implementation steps.
>
> **Status**: Ready for implementation
>
> **Related Commits**:
> - `a65e7b4` - Fallback fix for mixture data unavailability (temporary mitigation)
> - `38a5925` - Initial design documentation
> - `644f2e2` - UI refinements (removed advanced options)
> - `4c2849f` - Mathematical terminology corrections

---

## Table of Contents

1. [Context & Problem Statement](#context--problem-statement)
2. [Solution Overview](#solution-overview)
3. [Model Creation Redesign](#model-creation-redesign)
4. [Training Hub Redesign](#training-hub-redesign)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Code References & Patterns](#code-references--patterns)
7. [Testing & Validation](#testing--validation)
8. [Migration Strategy](#migration-strategy)

---

## Context & Problem Statement

### Background

**Read First**:
- `AGENTS.md` - Documentation architecture and navigation principles
- `use_cases/dashboard/README.md` - Dashboard structure and current state
- `docs/theory/conceptual_model.md` - Mental model and terminology (τ-classifier, responsibilities, usage entropy)
- `src/rcmvae/domain/config.py` - SSVAEConfig parameter definitions

### The Problem

**Root Cause**: Architecture mismatch between model creation and configuration.

**Current Flow** (Broken):
```
Homepage: Create Model
  ├─ Input: name, num_samples, num_labeled, seed
  ├─ Hardcoded: config = SSVAEConfig()  # prior_type="standard"
  └─ Creates: Model with standard prior architecture (DenseEncoder)

Training Hub: Edit Config
  ├─ User changes: prior_type = "mixture"
  ├─ Config updated: model.config.prior_type = "mixture"
  ├─ Architecture unchanged: Still has DenseEncoder (3 outputs)
  └─ Error: Model expects component_logits but encoder doesn't provide it
```

**User Impact**:
- Cannot create mixture/vamp/geometric models from UI
- Changing prior_type in training hub breaks model
- Error: `ValueError: Mixture responsibilities unavailable`
- Confusing UX: parameters appear editable but actually require model recreation

**Current Mitigation** (`a65e7b4`):
```python
# In training_callbacks.py:_predict_outputs()
except ValueError as e:
    if "mixture" in error_msg or "responsibilities" in error_msg:
        logger.warning(f"Mixture data unavailable, falling back...")
        return latent, recon, preds, cert, None, None  # No mixture features
```

This prevents crashes but mixture features don't work.

### Parameter Classification

**Code Reference**: `use_cases/dashboard/core/commands.py:611-648` (UpdateConfigCommand)

**Structural Parameters** (require model recreation):
```python
STRUCTURAL_PARAMS = {
    # Architecture
    "encoder_type",           # dense vs conv
    "decoder_type",           # dense vs conv
    "latent_dim",            # network layer sizes
    "hidden_dims",           # network depth/width

    # Prior (affects encoder outputs)
    "prior_type",            # standard/mixture/vamp/geometric_mog
    "num_components",        # mixture size
    "component_embedding_dim",  # embedding dimensions
    "use_component_aware_decoder",  # decoder architecture

    # Loss (affects decoder outputs)
    "use_heteroscedastic_decoder",  # decoder outputs (mean+sigma vs mean)
    "reconstruction_loss",   # output activation (BCE vs MSE)
}
```

**Modifiable Parameters** (safe to change after creation):
- Training: `batch_size`, `learning_rate`, `max_epochs`, `patience`, `random_seed`
- Loss weights: `recon_weight`, `kl_weight`, `label_weight`, `kl_c_weight`
- Regularization: `weight_decay`, `grad_clip_norm`, `dropout_rate`
- Prior behavior: `use_tau_classifier`, `learnable_pi`, `component_diversity_weight`
- Advanced: All anneal/warmup/gating parameters, Dirichlet prior, contrastive learning

**Principle**: If it changes the network architecture (layers, inputs, outputs), it's structural.

---

## Solution Overview

### Two-Part Solution

**Part 1: Model Creation Redesign**
- Expand homepage modal to include all structural parameters
- Users configure architecture BEFORE model creation
- Parameters organized logically: Dataset → Architecture → Prior
- Conditional rendering (show mixture params only for mixture priors)

**Part 2: Training Hub Redesign**
- Show architecture summary (read-only) at top
- Expose only modifiable parameters
- Conditional sections based on prior type (contextually intelligent)
- Group parameters by purpose, not alphabetically

### Design Principles

1. **Contextual Intelligence**: Show only relevant parameters for current architecture
2. **Information Hierarchy**: Essential parameters visible, advanced collapsible
3. **Visual Coherence**: Match existing app design (colors, typography, spacing)
4. **Semantic Clarity**: Use conceptual model terminology (channels, r, τ, H[p̂_c])
5. **Progressive Disclosure**: Core parameters in main flow, advanced in collapsible sections

### Visual Language

**Code Reference**: Existing styles in `use_cases/dashboard/pages/home.py`, `training_hub.py`

**Colors**:
- Primary: `#C10A27` (red - primary actions, Train button)
- Secondary: `#45717A` (teal - secondary actions)
- Neutral Dark: `#000000` (headings)
- Neutral Medium: `#6F6F6F` (labels, body text)
- Neutral Light: `#C6C6C6` (borders)
- Backgrounds: `#ffffff` (cards), `#f5f5f5` (page), `#fafafa` (read-only sections)

**Typography**:
- Font: `'Open Sans', Verdana, sans-serif`
- Monospace: `ui-monospace, monospace` (numeric values)
- Section headings: 17px bold
- Labels: 14px semi-bold
- Helper text: 12px regular
- Inputs: 14px monospace (for numbers)

**Spacing**:
- Section margins: `24px`
- Card padding: `24px`
- Input groups: `16px` gap
- Label-input gap: `6px`
- Input padding: `10px 12px`
- Border radius: `6px` (inputs, buttons, cards), `8px` (large cards)

---

## Model Creation Redesign

### Current Implementation

**Code Reference**: `use_cases/dashboard/pages/home.py:427-616` (_build_create_modal)

**Current Fields** (minimal):
```python
# pages/home.py:_build_create_modal()
- Model Name (optional)
- Total Samples (32-70000)
- Labeled Samples (0-70000)
- Sampling Seed (default: 0)
```

**Command**: `use_cases/dashboard/core/commands.py:662-736` (CreateModelCommand)
```python
@dataclass
class CreateModelCommand(Command):
    name: Optional[str] = None
    num_samples: int = 1024
    num_labeled: int = 128
    seed: Optional[int] = None

    def execute(self, state: AppState, services: Any):
        config = SSVAEConfig()  # ← Always uses defaults!
        request = CreateModelRequest(
            name=self.name or "Unnamed Model",
            config=config,  # ← prior_type="standard"
            dataset_total_samples=self.num_samples,
            dataset_seed=rng_seed,
        )
        model_id = services.model.create_model(request)
```

### New Modal Design

**Layout** (single-page form with conditional rendering):

```
┌─────────────────────────────────────────────────┐
│  Create New Model                                │
├─────────────────────────────────────────────────┤
│                                                  │
│  Model Name (optional)                          │
│  ┌─────────────────────────────────────────┐   │
│  │ e.g., Baseline Experiment               │   │
│  └─────────────────────────────────────────┘   │
│                                                  │
│  ═══════════════════════════════════════════   │
│  Dataset                                        │
│  ───────────────────────────────────────────   │
│                                                  │
│  Total Samples:      [1024  ▼]                 │
│  Labeled Samples:    [128   ▼]                 │
│  Sampling Seed:      [0     ▼]                 │
│                                                  │
│  ═══════════════════════════════════════════   │
│  Model Architecture (cannot be changed later)   │
│  ───────────────────────────────────────────   │
│                                                  │
│  Encoder/Decoder:                               │
│  ( ) Dense (MLP)    (•) Convolutional          │
│                                                  │
│  [if Dense selected:]                           │
│  Hidden Layers: [256,128,64]                   │
│                                                  │
│  Latent Dimension:   [2     ▼]                 │
│                                                  │
│  Reconstruction Loss:                           │
│  (•) BCE (binary)    ( ) MSE (continuous)      │
│                                                  │
│  ☐ Heteroscedastic Decoder                     │
│                                                  │
│  ─── Prior Configuration ───                    │
│                                                  │
│  Prior Type:                                    │
│  ( ) Standard    (•) Mixture    ( ) Vamp       │
│  ( ) Geometric                                  │
│                                                  │
│  [if mixture/vamp/geometric selected:]          │
│  Number of Components:   [10  ▼]               │
│  Component Embedding Dim: [auto ▼]             │
│  ☑ Component-Aware Decoder                     │
│                                                  │
│  [Cancel]                   [Create Model]     │
│                                                  │
└─────────────────────────────────────────────────┘
```

**Rationale for Organization**:
1. **Dataset**: Foundation - determines what data the model trains on
2. **Encoder/Decoder**: Most fundamental - determines base network structure
3. **Latent Dimension**: Core parameter affecting all downstream components
4. **Reconstruction Loss**: Affects decoder output activation (BCE for binary, MSE for continuous)
5. **Heteroscedastic Decoder**: Structural (decoder outputs mean+variance vs just mean)
6. **Prior Configuration**: Builds on encoder (determines encoder outputs)

This matches the dependency chain: Base network → Latent space → Loss structure → Prior structure.

### Implementation Specifications

#### Section 1: Dataset (No Changes)

**Fields** (existing):
- `name` (string, optional)
- `num_samples` (32-70000, default: 1024)
- `num_labeled` (0-num_samples, default: 128)
- `seed` (integer, default: 0)

**Validation** (existing in CreateModelCommand.validate):
```python
if total <= 0:
    errors.append("Total samples must be greater than zero")
if labeled < 0:
    errors.append("Labeled samples must be non-negative")
if total > 70000:
    errors.append("Total samples must be at most 70000 for MNIST")
if labeled > total:
    errors.append("Labeled samples cannot exceed total samples")
```

#### Section 2: Architecture (New)

**Encoder/Decoder Type**:
- Component: `dbc.RadioItems`
- Options: `[{"label": "Dense (MLP)", "value": "dense"}, {"label": "Convolutional", "value": "conv"}]`
- Default: `"conv"` (better for MNIST)
- Maps to: `config.encoder_type`, `config.decoder_type`

**Hidden Layers** (conditional on Dense):
- Component: `dcc.Input(type="text")`
- Show if: `encoder_type == "dense"`
- Format: Comma-separated integers (e.g., "256,128,64")
- Default: `"256,128,64"`
- Validation: Parse and validate positive integers
- Maps to: `config.hidden_dims` (tuple of ints)

**Latent Dimension**:
- Component: `dcc.Input(type="number")`
- Range: 2-256
- Default: `2`
- Maps to: `config.latent_dim`

**Reconstruction Loss**:
- Component: `dbc.RadioItems`
- Options: `[{"label": "BCE (binary)", "value": "bce"}, {"label": "MSE (continuous)", "value": "mse"}]`
- Default: `"bce"` (standard for MNIST)
- Helper: "BCE for binary images (MNIST), MSE for continuous"
- Maps to: `config.reconstruction_loss`

**Heteroscedastic Decoder**:
- Component: `dbc.Checkbox`
- Label: "Heteroscedastic Decoder"
- Helper: "Learn per-image variance σ(x) for uncertainty quantification"
- Default: `False`
- Maps to: `config.use_heteroscedastic_decoder`

#### Section 3: Prior Configuration (New)

**Prior Type**:
- Component: `dbc.RadioItems`
- Options:
  ```python
  [
      {"label": "Standard", "value": "standard"},
      {"label": "Mixture", "value": "mixture"},
      {"label": "Vamp", "value": "vamp"},
      {"label": "Geometric", "value": "geometric_mog"}
  ]
  ```
- Default: `"mixture"` (most feature-rich)
- Maps to: `config.prior_type`

**Number of Components** (conditional):
- Show if: `prior_type in ["mixture", "vamp", "geometric_mog"]`
- Component: `dcc.Input(type="number")`
- Range: 1-64
- Default: `10`
- Maps to: `config.num_components`

**Component Embedding Dim** (conditional):
- Show if: `prior_type in ["mixture", "vamp", "geometric_mog"]`
- Component: `dcc.Input(type="number", placeholder="auto")`
- Range: 1-128 or blank
- Default: `None` (auto = latent_dim)
- Helper: "Default: same as latent dimension"
- Maps to: `config.component_embedding_dim`

**Component-Aware Decoder** (conditional):
- Show if: `prior_type in ["mixture", "geometric_mog"]` (not vamp)
- Component: `dbc.Checkbox`
- Label: "Component-Aware Decoder"
- Helper: "Separate decoder pathways per component (recommended)"
- Default: `True`
- Maps to: `config.use_component_aware_decoder`

### Updated CreateModelCommand

**Code Changes** (`use_cases/dashboard/core/commands.py`):

```python
@dataclass
class CreateModelCommand(Command):
    """Create a new model with full architectural configuration."""
    # Existing fields
    name: Optional[str] = None
    num_samples: int = 1024
    num_labeled: int = 128
    seed: Optional[int] = None

    # NEW: Architecture configuration
    encoder_type: str = "conv"
    decoder_type: str = "conv"
    hidden_dims: Optional[str] = None  # "256,128,64" format
    latent_dim: int = 2
    reconstruction_loss: str = "bce"
    use_heteroscedastic_decoder: bool = False

    # NEW: Prior configuration
    prior_type: str = "mixture"
    num_components: int = 10
    component_embedding_dim: Optional[int] = None
    use_component_aware_decoder: bool = True

    def validate(self, state: AppState, services: Any) -> Optional[str]:
        """Validate dataset sizing and architecture choices."""
        errors: List[str] = []

        # Existing dataset validation...

        # NEW: Architecture validation
        if self.encoder_type not in ["dense", "conv"]:
            errors.append("Encoder type must be 'dense' or 'conv'")
        if self.decoder_type not in ["dense", "conv"]:
            errors.append("Decoder type must be 'dense' or 'conv'")
        if self.latent_dim < 2 or self.latent_dim > 256:
            errors.append("Latent dimension must be between 2 and 256")
        if self.reconstruction_loss not in ["bce", "mse"]:
            errors.append("Reconstruction loss must be 'bce' or 'mse'")

        # Validate hidden_dims if dense
        if self.encoder_type == "dense" and self.hidden_dims:
            try:
                dims = [int(d.strip()) for d in self.hidden_dims.split(",")]
                if not all(d > 0 for d in dims):
                    errors.append("Hidden dimensions must be positive integers")
            except ValueError:
                errors.append("Hidden dimensions must be comma-separated integers")

        # Validate prior configuration
        if self.prior_type not in ["standard", "mixture", "vamp", "geometric_mog"]:
            errors.append("Invalid prior type")
        if self.prior_type in ["mixture", "vamp", "geometric_mog"]:
            if self.num_components < 1 or self.num_components > 64:
                errors.append("Number of components must be between 1 and 64")
            if self.component_embedding_dim is not None:
                if self.component_embedding_dim < 1 or self.component_embedding_dim > 128:
                    errors.append("Component embedding dim must be between 1 and 128")

        if errors:
            return "; ".join(errors)
        return None

    def execute(self, state: AppState, services: Any) -> Tuple[AppState, str]:
        """Create new model with user-specified architecture."""
        from rcmvae.domain.config import SSVAEConfig

        # Parse hidden_dims
        hidden_dims_tuple = None
        if self.encoder_type == "dense" and self.hidden_dims:
            hidden_dims_tuple = tuple(int(d.strip()) for d in self.hidden_dims.split(","))

        # Build config from user choices
        config = SSVAEConfig(
            # Architecture
            encoder_type=self.encoder_type,
            decoder_type=self.decoder_type,
            hidden_dims=hidden_dims_tuple or (256, 128, 64),  # Default if not specified
            latent_dim=self.latent_dim,
            reconstruction_loss=self.reconstruction_loss,
            use_heteroscedastic_decoder=self.use_heteroscedastic_decoder,

            # Prior
            prior_type=self.prior_type,
            num_components=self.num_components,
            component_embedding_dim=self.component_embedding_dim,
            use_component_aware_decoder=self.use_component_aware_decoder,

            # Defaults for other parameters (modifiable in training hub)
            batch_size=128,
            learning_rate=1e-3,
            max_epochs=200,
            patience=20,
            random_seed=42,
        )

        # Rest of existing logic...
        rng_seed = int(self.seed if self.seed is not None else 0)
        request = CreateModelRequest(
            name=self.name or "Unnamed Model",
            config=config,  # ← Now uses user-specified config!
            dataset_total_samples=self.num_samples,
            dataset_seed=rng_seed,
        )

        model_id = services.model.create_model(request)

        # Add initial labels (existing logic)...

        return new_state, model_id
```

### Callback Implementation

**New Callback** (`use_cases/dashboard/callbacks/home_callbacks.py` - create if doesn't exist):

```python
@app.callback(
    Output("home-hidden-layers-input", "style"),
    Input("home-encoder-type-radio", "value"),
)
def toggle_hidden_layers(encoder_type: str):
    """Show/hide hidden layers input based on encoder type."""
    if encoder_type == "dense":
        return {"display": "block", "marginBottom": "16px"}
    else:
        return {"display": "none"}


@app.callback(
    Output("home-prior-options", "style"),
    Input("home-prior-type-radio", "value"),
)
def toggle_prior_options(prior_type: str):
    """Show/hide prior-specific options based on prior type."""
    if prior_type in ["mixture", "vamp", "geometric_mog"]:
        return {"display": "block"}
    else:
        return {"display": "none"}


@app.callback(
    Output("home-component-aware-decoder-option", "style"),
    Input("home-prior-type-radio", "value"),
)
def toggle_component_aware_option(prior_type: str):
    """Show component-aware decoder option only for mixture/geometric."""
    if prior_type in ["mixture", "geometric_mog"]:
        return {"display": "block"}
    else:
        return {"display": "none"}
```

---

## Training Hub Redesign

### Current Implementation

**Code Reference**: `use_cases/dashboard/pages/training_hub.py:450-650`

**Current Structure**:
- Left panel (40%): "Training Configuration" with collapsible "Essential Parameters"
- Right panel (60%): Progress charts, recent runs
- Shows ALL config parameters without contextual filtering
- No architecture summary
- No distinction between structural and modifiable parameters

**Current Parameters Exposed**:
- Epochs (via `_render_quick_control("max_epochs")`)
- Learning rate, recon weight, KL weight (in collapsible section)
- Full config accessible via link to `/configure-training`

### New Training Hub Layout

```
┌──────────────────────────────────────────────────────────────┐
│  Training Hub                                                 │
│  Model: experiment-name-123                                   │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────┐  ┌───────────────────────────────┐ │
│  │ LEFT (40%)          │  │ RIGHT (60%)                   │ │
│  │                     │  │                               │ │
│  │ 1. Architecture     │  │ Training Progress             │ │
│  │    Summary 🔒       │  │ (existing charts)             │ │
│  │                     │  │                               │ │
│  │ 2. Training Setup   │  │ Recent Runs                   │ │
│  │                     │  │ (existing section)            │ │
│  │ 3. Loss Weights     │  │                               │ │
│  │                     │  │ Mixture Diagnostics           │ │
│  │ 4. Prior-Specific   │  │ (existing - if mixture)       │ │
│  │    (conditional)    │  │                               │ │
│  │                     │  │                               │ │
│  │ 5. Regularization   │  │                               │ │
│  │    (collapsible)    │  │                               │ │
│  │                     │  │                               │ │
│  │ [Train Model]       │  │                               │ │
│  │ [Stop Training]     │  │                               │ │
│  │ [Full Config...]    │  │                               │ │
│  └─────────────────────┘  └───────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### Section Specifications

#### 1. Architecture Summary (New, Read-Only)

**Purpose**: Show locked structural parameters

**Code Reference**: Get values from `state.active_model.config`

```
┌────────────────────────────────────────────┐
│ Architecture 🔒 (fixed at creation)        │
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

**Implementation**:
```python
def _build_architecture_summary(config: SSVAEConfig) -> html.Div:
    """Build read-only architecture summary."""
    # Prior description
    if config.prior_type == "standard":
        prior_desc = "Standard N(0,I)"
    elif config.prior_type in ["mixture", "vamp", "geometric_mog"]:
        prior_desc = f"{config.prior_type.title()} ({config.num_components} components)"

    # Encoder description
    encoder_desc = "Convolutional" if config.encoder_type == "conv" else "Dense (MLP)"

    # Decoder flags
    component_aware = "Yes" if config.use_component_aware_decoder else "No"
    heteroscedastic = "Yes" if config.use_heteroscedastic_decoder else "No"

    return html.Div([
        html.Div("Architecture 🔒 (fixed at creation)", style={
            "fontSize": "14px",
            "fontWeight": "700",
            "color": "#000000",
            "marginBottom": "12px",
        }),
        html.Div([
            _build_summary_row("Prior:", prior_desc),
            _build_summary_row("Encoder:", encoder_desc),
            _build_summary_row("Latent Dim:", str(config.latent_dim)),
            _build_summary_row("Recon Loss:", f"{config.reconstruction_loss.upper()}"),
            html.Div(style={"height": "8px"}),  # Spacer
            _build_summary_row("Component-aware decoder:", component_aware),
            _build_summary_row("Heteroscedastic decoder:", heteroscedastic),
        ]),
    ], style={
        "padding": "16px",
        "backgroundColor": "#fafafa",
        "border": "1px solid #E6E6E6",
        "borderRadius": "6px",
        "marginBottom": "24px",
    })

def _build_summary_row(label: str, value: str) -> html.Div:
    """Build a single summary row."""
    return html.Div([
        html.Span(label, style={"fontSize": "13px", "color": "#6F6F6F", "marginRight": "8px"}),
        html.Span(value, style={"fontSize": "13px", "color": "#000000", "fontWeight": "600"}),
    ], style={"marginBottom": "4px"})
```

#### 2. Training Setup (Always Visible)

**Fields** (`config_metadata.py` references):
- `max_epochs` (1-500, default: 200)
- `patience` (1-100, default: 20)
- `learning_rate` (1e-5 to 1e-1, default: 1e-3)
- `batch_size` (32-4096 step 32, default: 128)
- `random_seed` (0-10000, default: 42)

**Implementation**: Use existing `_render_quick_control()` helper

#### 3. Loss Weights (Always Visible)

**Fields**:
- `recon_weight` (0.0-10000)
  - Helper text: Show recommended values based on `reconstruction_loss`
  - "Typical: 1.0 for BCE, 500 for MSE"
- `kl_weight` (0.0-20.0, default: 1.0)
- `label_weight` (0.0-50.0, default: 1.0)

#### 4. Prior-Specific Sections (Conditional)

**Conditional Rendering Logic**:
```python
def _build_prior_section(config: SSVAEConfig) -> html.Div:
    """Build prior-specific configuration section."""
    prior_type = config.prior_type

    if prior_type == "standard":
        return _build_standard_prior_note()
    elif prior_type == "mixture":
        return _build_mixture_prior_section(config)
    elif prior_type == "vamp":
        return _build_vamp_prior_section(config)
    elif prior_type == "geometric_mog":
        return _build_geometric_prior_section(config)
    else:
        return html.Div()
```

##### 4A. Standard Prior

```python
def _build_standard_prior_note() -> html.Div:
    """Simple note for standard prior."""
    return html.Div([
        html.Div("Prior: Standard", style={"fontSize": "14px", "fontWeight": "700"}),
        html.Div("Simple N(0,I) Gaussian prior. No additional configuration needed.",
                 style={"fontSize": "13px", "color": "#6F6F6F", "marginTop": "8px"}),
    ], style={"marginBottom": "24px"})
```

##### 4B. Mixture Prior

**Semantic Reference**: `docs/theory/conceptual_model.md` - terminology (r, τ, H[p̂_c])

**Fields**:
- `use_tau_classifier` (boolean, default: True)
- `tau_smoothing_alpha` (>0, default: 1.0) - "τ Smoothing (α₀)"
- `kl_c_weight` (0.0-10.0, default: 1.0) - "Component KL Weight"
- `kl_c_anneal_epochs` (0-500, default: 0) - "KL Anneal Epochs"
- `learnable_pi` (boolean, default: True) - "Learnable π"
- `component_diversity_weight` (-10.0 to 10.0, default: -0.1)
  - **Label**: "Usage Entropy Weight"
  - **Helper**: "Entropy H[p̂_c]: negative = reward diversity"
- **Advanced** (collapsible):
  - `dirichlet_alpha` (0.1-10.0 or blank, default: None)
  - `dirichlet_weight` (0.0-10.0, default: 1.0)
  - `top_m_gating` (0-num_components, default: 0)
  - `soft_embedding_warmup_epochs` (0-500, default: 0)

**Mockup**:
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

##### 4C. VampPrior

**Fields** (shared with mixture):
- `use_tau_classifier`, `tau_smoothing_alpha`, `kl_c_weight`, `kl_c_anneal_epochs`
- `component_diversity_weight` (usage entropy)

**VampPrior-Specific**:
- `vamp_num_samples_kl` (1-10, default: 1) - "KL Samples (MC)"
- `vamp_pseudo_lr_scale` (0.01-1.0, default: 0.1) - "Pseudo-Input LR Scale"

**Info Box**:
- Note: π is uniform (not learnable) for VampPrior
- Pseudo-inputs initialized at creation time

##### 4D. Geometric MoG

**Fields** (same as mixture basic):
- `use_tau_classifier`, `tau_smoothing_alpha`, `kl_c_weight`
- `learnable_pi`, `component_diversity_weight`

**Info Note**:
- "Components arranged geometrically (circle/grid) with fixed spacing."
- No yellow warning (it's a valid prior choice)

#### 5. Regularization (Collapsible)

**Fields**:
- `weight_decay` (0.0-0.1, default: 1e-4)
- `grad_clip_norm` (0.0-100.0, default: 1.0, 0=disabled)
- `dropout_rate` (0.0-0.8, default: 0.2)

**Heteroscedastic** (only if `use_heteroscedastic_decoder=True`):
- `sigma_min` (1e-5 to 5.0, default: 0.05)
- `sigma_max` (0.01-10.0, default: 0.5)

**Contrastive**:
- `use_contrastive` (boolean, default: False)
- `contrastive_weight` (0.0-50.0, default: 0.0)

### Updated config_metadata.py

**Code Changes**:

```python
# Split field specs into two sets

def get_structural_field_specs() -> Tuple[FieldSpec, ...]:
    """Parameters that require model recreation (for creation modal)."""
    return tuple(spec for spec in _FIELD_SPECS if spec.key in STRUCTURAL_PARAMS)

def get_modifiable_field_specs() -> Tuple[FieldSpec, ...]:
    """Parameters safe to change after creation (for training hub)."""
    return tuple(spec for spec in _FIELD_SPECS if spec.key not in STRUCTURAL_PARAMS)

def get_prior_specific_specs(prior_type: str) -> Tuple[FieldSpec, ...]:
    """Get fields relevant for the given prior type."""
    if prior_type == "standard":
        return ()

    # Common mixture-based fields
    base_mixture_keys = {
        "use_tau_classifier",
        "tau_smoothing_alpha",
        "kl_c_weight",
        "kl_c_anneal_epochs",
        "component_diversity_weight",
    }

    if prior_type == "mixture":
        mixture_keys = base_mixture_keys | {
            "learnable_pi",
            "dirichlet_alpha",
            "dirichlet_weight",
            "top_m_gating",
            "soft_embedding_warmup_epochs",
        }
        return tuple(spec for spec in _FIELD_SPECS if spec.key in mixture_keys)

    elif prior_type == "vamp":
        vamp_keys = base_mixture_keys | {
            "vamp_num_samples_kl",
            "vamp_pseudo_lr_scale",
        }
        return tuple(spec for spec in _FIELD_SPECS if spec.key in vamp_keys)

    elif prior_type == "geometric_mog":
        geometric_keys = base_mixture_keys | {"learnable_pi"}
        return tuple(spec for spec in _FIELD_SPECS if spec.key in geometric_keys)

    return ()
```

### Terminology Alignment

**Reference**: `docs/theory/conceptual_model.md` - §Notation Lock

**Correct Terms**:
- ✅ "Component" or "Channel" (not "cluster" or "mode")
- ✅ "Responsibilities" r = q(c|x) from encoder
- ✅ "τ-Classifier" (channel→label map)
- ✅ "Usage Entropy" H[p̂_c] (NOT "diversity")
- ✅ "π" (mixture weights)
- ✅ "Latent-only classification" (via r×τ)

**Parameter Labels**:
```python
# In FieldSpec definitions
FieldSpec(
    key="component_diversity_weight",
    label="Usage Entropy Weight",  # ← Mathematical term
    description="Entropy H[p̂_c]: negative = reward diversity",  # ← Explain direction
)

FieldSpec(
    key="tau_smoothing_alpha",
    label="τ Smoothing (α₀)",  # ← Use mathematical notation
    description="Laplace smoothing for unseen c→y pairs",
)

FieldSpec(
    key="kl_c_weight",
    label="Component KL Weight",
    description="Weight on KL(q(c|x) || π)",  # ← Show mathematical objective
)
```

---

## Implementation Roadmap

### Prerequisites

**Before Starting**:
1. Read `AGENTS.md` - understand documentation structure
2. Review `use_cases/dashboard/README.md` - current architecture
3. Check `docs/theory/conceptual_model.md` - terminology and semantics
4. Verify current state: `git log --oneline -10` (should see commits a65e7b4, 38a5925, 644f2e2, 4c2849f)

### Phase 1: Model Creation Enhancement (3-5 days)

**Goal**: Allow users to configure architecture at creation time

**Files to Modify**:
- `use_cases/dashboard/pages/home.py:427-616` (_build_create_modal)
- `use_cases/dashboard/core/commands.py:662-736` (CreateModelCommand)
- `use_cases/dashboard/callbacks/home_callbacks.py` (create new file)

**Steps**:

1. **Add Architecture Fields to Modal** (1 day)
   - Add encoder/decoder radio buttons
   - Add hidden layers input (conditional on Dense)
   - Add latent dimension input
   - Add reconstruction loss radio buttons
   - Add heteroscedastic decoder checkbox
   - Style to match existing modal

2. **Add Prior Configuration Section** (1 day)
   - Add prior type radio buttons
   - Add number of components input (conditional)
   - Add component embedding dim input (conditional)
   - Add component-aware decoder checkbox (conditional)
   - Implement conditional rendering logic

3. **Update CreateModelCommand** (1 day)
   - Add new parameters to dataclass
   - Implement validation for architecture choices
   - Build SSVAEConfig from user inputs
   - Test with each prior type

4. **Add Conditional Rendering Callbacks** (0.5 day)
   - Show/hide hidden layers based on encoder type
   - Show/hide prior options based on prior type
   - Show/hide component-aware decoder based on prior type

5. **Testing & Polish** (0.5 day)
   - Create model with each prior type
   - Verify config is correctly set
   - Verify model architecture matches config
   - Test edge cases (validation)

**Acceptance Criteria**:
- ✅ Can create models with all 4 prior types from homepage
- ✅ Conditional fields show/hide correctly
- ✅ Validation catches invalid inputs
- ✅ Created models have correct architecture
- ✅ Visual style matches existing app

### Phase 2: Training Hub Architecture Summary (1 day)

**Goal**: Show read-only architecture summary at top of training hub

**Files to Modify**:
- `use_cases/dashboard/pages/training_hub.py:450-650`

**Steps**:

1. **Create Architecture Summary Component** (0.5 day)
   - Implement `_build_architecture_summary(config)`
   - Implement `_build_summary_row(label, value)`
   - Style with locked icon 🔒
   - Light gray background (#fafafa)

2. **Add to Training Hub Layout** (0.5 day)
   - Insert at top of left column
   - Verify spacing and alignment
   - Test with different prior types
   - Ensure values pull from `state.active_model.config`

**Acceptance Criteria**:
- ✅ Summary shows current architecture
- ✅ Clearly marked as read-only (locked icon)
- ✅ Shows different info for each prior type
- ✅ Values are accurate (match model.config)

### Phase 3: Training Hub Parameter Reorganization (2 days)

**Goal**: Group parameters logically and conditionally

**Files to Modify**:
- `use_cases/dashboard/pages/training_hub.py`
- `use_cases/dashboard/core/config_metadata.py`

**Steps**:

1. **Split config_metadata.py** (0.5 day)
   - Add `get_modifiable_field_specs()`
   - Add `get_prior_specific_specs(prior_type)`
   - Add `STRUCTURAL_PARAMS` constant
   - Update existing functions

2. **Create Section Builders** (1 day)
   - `_build_training_setup_section()` - epochs, patience, LR, batch size
   - `_build_loss_weights_section()` - recon, KL, label weights
   - `_build_standard_prior_note()` - simple note
   - `_build_mixture_prior_section()` - mixture parameters
   - `_build_vamp_prior_section()` - VampPrior parameters
   - `_build_geometric_prior_section()` - Geometric parameters
   - `_build_regularization_section()` - collapsible advanced

3. **Update Main Layout** (0.5 day)
   - Replace existing parameter rendering
   - Use new section builders
   - Add conditional logic for prior type
   - Ensure proper spacing

**Acceptance Criteria**:
- ✅ Training setup always visible
- ✅ Loss weights always visible
- ✅ Prior-specific section changes based on prior type
- ✅ Regularization is collapsible
- ✅ Mathematical terminology used correctly

### Phase 4: UpdateConfigCommand Validation (1 day)

**Goal**: Prevent users from changing structural parameters

**Files to Modify**:
- `use_cases/dashboard/core/commands.py:562-651` (UpdateConfigCommand)

**Steps**:

1. **Add Architecture Mismatch Detection** (0.5 day)
   ```python
   def validate(self, state: AppState, services: Any):
       current_config = state.active_model.config
       new_config = self._new_config

       # Check for structural changes
       structural_changes = []
       for field in STRUCTURAL_PARAMS:
           if getattr(current_config, field) != getattr(new_config, field):
               structural_changes.append(field)

       if structural_changes:
           return (
               f"Cannot change structural parameters: {', '.join(structural_changes)}. "
               "These require creating a new model."
           )

       return None
   ```

2. **Testing** (0.5 day)
   - Try changing prior_type in training hub → Should block
   - Try changing latent_dim → Should block
   - Try changing learning_rate → Should work
   - Verify error messages are clear

**Acceptance Criteria**:
- ✅ Changing structural params is blocked with clear error
- ✅ Changing modifiable params works normally
- ✅ Error message explains what to do (create new model)

### Phase 5: Polish & Documentation (1 day)

**Goal**: Final polish and update documentation

**Files to Modify**:
- `use_cases/dashboard/ROADMAP.md`
- `use_cases/dashboard/docs/collaboration_notes.md`

**Steps**:

1. **Visual Polish** (0.5 day)
   - Verify all spacing matches design spec
   - Check typography consistency
   - Test color contrast
   - Verify responsive layout

2. **Update Documentation** (0.5 day)
   - Update ROADMAP.md with completion notes
   - Add known issues section if any
   - Document new model creation flow
   - Note any breaking changes

**Acceptance Criteria**:
- ✅ Visual design matches specification
- ✅ ROADMAP.md updated
- ✅ No regressions in existing functionality

---

## Code References & Patterns

### File Organization

**Reference**: `use_cases/dashboard/README.md` - Project Structure

```
use_cases/dashboard/
├── core/
│   ├── state_manager.py       # AppStateManager (state operations)
│   ├── state_models.py        # Immutable state dataclasses
│   ├── commands.py            # Command pattern (UPDATE THESE)
│   ├── model_manager.py       # File I/O for models
│   └── config_metadata.py     # Parameter metadata (UPDATE THIS)
├── services/
│   ├── training_service.py    # Training execution
│   ├── model_service.py       # Model CRUD (uses CreateModelRequest)
│   ├── labeling_service.py    # Label persistence
│   └── container.py           # Dependency injection
├── callbacks/
│   ├── training_callbacks.py       # Main dashboard training
│   ├── training_hub_callbacks.py   # Training hub page (UPDATE THIS)
│   ├── home_callbacks.py           # Home page (CREATE THIS)
│   └── visualization_callbacks.py  # Plot updates
├── pages/
│   ├── home.py               # Model selection (UPDATE THIS)
│   ├── training_hub.py       # Training workbench (UPDATE THIS)
│   └── layouts.py            # Main dashboard layout
└── utils/
    ├── visualization.py       # Plotting utilities
    └── training_callback.py   # JAX→Dashboard bridge
```

### Existing Patterns

**Command Pattern** (`core/commands.py`):
```python
@dataclass
class SomeCommand(Command):
    param1: str
    param2: int

    def validate(self, state: AppState, services: Any) -> Optional[str]:
        """Return error message or None."""
        if self.param1 not in valid_values:
            return "Invalid param1"
        return None

    def execute(self, state: AppState, services: Any) -> Tuple[AppState, str]:
        """Return (new_state, message)."""
        # Perform action
        new_state = state.with_something(...)
        return new_state, "Success message"
```

**State Updates** (immutable pattern):
```python
# CORRECT: Create new state
new_model = current_model.with_config(new_config)
new_state = state.with_active_model(new_model)

# WRONG: Mutate in place
state.active_model.config = new_config  # DON'T DO THIS
```

**Conditional Rendering** (Dash callbacks):
```python
@app.callback(
    Output("some-div", "style"),
    Input("some-dropdown", "value"),
)
def toggle_visibility(dropdown_value: str):
    if dropdown_value == "show":
        return {"display": "block"}
    else:
        return {"display": "none"}
```

**Field Rendering** (existing helper):
```python
# Use existing _render_quick_control helper
_render_quick_control(
    field_key="learning_rate",  # Key in config_metadata.py
    custom_id="training-hub-lr",  # Component ID
    config=config,  # Current config
    label_style={...},  # Label styling
    input_style={...},  # Input styling
)
```

### Key Backend Classes

**SSVAEConfig** (`src/rcmvae/domain/config.py:77-250`):
```python
@dataclass
class SSVAEConfig:
    # Architecture
    encoder_type: str = "dense"
    decoder_type: str = "dense"
    latent_dim: int = 2
    hidden_dims: Tuple[int, ...] = (256, 128, 64)
    reconstruction_loss: str = "mse"
    use_heteroscedastic_decoder: bool = False

    # Prior
    prior_type: str = "standard"
    num_components: int = 10
    component_embedding_dim: int | None = None
    use_component_aware_decoder: bool = True

    # ... many more parameters

    def is_mixture_based_prior(self) -> bool:
        return self.prior_type in {"mixture", "vamp", "geometric_mog"}
```

**ModelState** (`use_cases/dashboard/core/state_models.py`):
```python
@dataclass(frozen=True)
class ModelState:
    model_id: str
    model: Any  # SSVAE instance
    trainer: Any  # InteractiveTrainer
    config: SSVAEConfig  # ← Source of truth
    data: DataState
    training: TrainingState
    history: HistoryState
    ui: UIState
    metadata: ModelMetadata

    def with_config(self, new_config: SSVAEConfig) -> ModelState:
        return replace(self, config=new_config, ...)
```

### Style Constants

**Reference**: Extracted from `pages/home.py`, `training_hub.py`

```python
# Define once, reuse everywhere
COLORS = {
    "primary": "#C10A27",
    "secondary": "#45717A",
    "text_dark": "#000000",
    "text_medium": "#6F6F6F",
    "text_light": "#999999",
    "border": "#C6C6C6",
    "border_light": "#E6E6E6",
    "bg_white": "#ffffff",
    "bg_light": "#f5f5f5",
    "bg_lighter": "#fafafa",
}

TYPOGRAPHY = {
    "font_family": "'Open Sans', Verdana, sans-serif",
    "font_mono": "ui-monospace, monospace",
    "heading_large": "17px",
    "heading_medium": "15px",
    "label": "14px",
    "body": "13px",
    "small": "12px",
}

SPACING = {
    "section": "24px",
    "card": "24px",
    "group": "16px",
    "label": "6px",
}
```

---

## Testing & Validation

### Unit Testing

**Create**: `tests/dashboard/test_model_creation.py`

```python
def test_create_model_with_mixture_prior():
    """Test creating model with mixture prior."""
    cmd = CreateModelCommand(
        name="Test Mixture Model",
        num_samples=1024,
        num_labeled=128,
        encoder_type="conv",
        decoder_type="conv",
        latent_dim=2,
        prior_type="mixture",
        num_components=10,
    )

    # Validation should pass
    error = cmd.validate(mock_state, mock_services)
    assert error is None

    # Execute and verify
    new_state, msg = cmd.execute(mock_state, mock_services)
    model = new_state.active_model
    assert model.config.prior_type == "mixture"
    assert model.config.num_components == 10


def test_prevent_structural_config_change():
    """Test that UpdateConfigCommand blocks structural changes."""
    # Create model with standard prior
    state = create_model_with_config(prior_type="standard")

    # Try to change to mixture
    new_config = replace(state.active_model.config, prior_type="mixture")
    cmd = UpdateConfigCommand(config=new_config)

    # Should be blocked
    error = cmd.validate(state, mock_services)
    assert error is not None
    assert "structural parameters" in error.lower()
    assert "prior_type" in error
```

### Integration Testing

**Manual Test Plan**:

1. **Model Creation Flow**:
   - Go to homepage
   - Click "Create New Model"
   - Select Dense encoder → Verify hidden layers input appears
   - Select Conv encoder → Verify hidden layers input disappears
   - Select Mixture prior → Verify component options appear
   - Enter all fields → Click "Create Model"
   - Verify model created with correct config

2. **Training Hub Flow**:
   - Load mixture model → Verify mixture section appears
   - Load standard model → Verify simple note appears
   - Verify architecture summary shows correct values
   - Try changing learning rate → Should work
   - Try changing prior_type → Should be blocked

3. **Edge Cases**:
   - Create model with 64 components → Verify no errors
   - Create model with Dense + empty hidden layers → Should use defaults
   - Create VampPrior model → Verify pseudo-input note appears
   - Load old model (created before this change) → Should work with warning

### Validation Checklist

Before marking complete:

- [ ] Can create models with all 4 prior types
- [ ] Conditional fields show/hide correctly
- [ ] Training hub shows architecture summary
- [ ] Training hub shows correct prior-specific section
- [ ] Cannot change structural parameters from training hub
- [ ] Existing models still load and train
- [ ] Visual design matches specification
- [ ] Mathematical terminology used correctly
- [ ] No console errors or warnings
- [ ] Code follows existing patterns

---

## Migration Strategy

### For Existing Models

**Models created before this change**:
- Continue to work without modification
- May have config/architecture mismatches if users previously changed structural params
- Show informational banner on first load

**Banner Implementation**:
```python
# In training_hub.py
def _check_architecture_mismatch(model_state: ModelState) -> bool:
    """Check if model has config/architecture mismatch."""
    # Check if model was created before structural locking
    if model_state.metadata.created_at < "2025-11-17":  # This redesign date
        # Check for mismatch indicators
        config = model_state.config
        # If config says mixture but no responsibilities in data, likely mismatch
        if config.is_mixture_based_prior() and model_state.data.responsibilities is None:
            return True
    return False

# In layout
if _check_architecture_mismatch(state.active_model):
    banner = dbc.Alert([
        html.Strong("Architecture Note: "),
        "This model was created before architecture locking. ",
        "If you experience issues, consider creating a new model with the desired architecture.",
    ], color="info", dismissable=True)
```

### Communication to Users

**In-App Message** (first load after update):
```
🎉 New Feature: Full Architecture Control

You can now configure your model's architecture when creating a model:
- Prior type (Standard, Mixture, VampPrior, Geometric)
- Encoder/Decoder type (Dense or Convolutional)
- Latent dimension, reconstruction loss, and more

⚠️ Important: Architecture cannot be changed after creation.
To use a different architecture, create a new model.

✅ Training parameters (learning rate, batch size, loss weights) can still
be modified anytime in the Training Hub.
```

### Breaking Changes

**None** - This is a non-breaking change:
- Old models continue to work
- New functionality is additive (expanded creation modal)
- Training hub restrictions prevent errors that would have occurred anyway

### Rollback Plan

If issues arise:
1. Revert commits: `git revert 4c2849f 644f2e2 38a5925` (keep a65e7b4 fallback fix)
2. Or: Keep new creation flow, temporarily re-enable all params in training hub
3. Document issues in `use_cases/dashboard/docs/collaboration_notes.md`

---

## References & Links

### Documentation

**Essential Reading** (before implementation):
- `AGENTS.md` - How to navigate the documentation
- `use_cases/dashboard/README.md` - Dashboard overview
- `docs/theory/conceptual_model.md` - Terminology and semantics
- `src/rcmvae/domain/config.py` - Parameter definitions (inline docs)

**Implementation Guides** (during implementation):
- `use_cases/dashboard/docs/DEVELOPER_GUIDE.md` - Architecture patterns
- `docs/development/architecture.md` - Design patterns
- `use_cases/experiments/README.md` - Configuration examples

**Related Work**:
- `use_cases/dashboard/ROADMAP.md` - Project roadmap and status
- `use_cases/dashboard/docs/collaboration_notes.md` - Current work

### Code Files by Phase

**Phase 1 (Model Creation)**:
- `use_cases/dashboard/pages/home.py:427-616` - Modal to update
- `use_cases/dashboard/core/commands.py:662-736` - CreateModelCommand
- `use_cases/dashboard/callbacks/home_callbacks.py` - Create this file

**Phase 2 (Architecture Summary)**:
- `use_cases/dashboard/pages/training_hub.py:450-650` - Add summary

**Phase 3 (Training Hub Reorganization)**:
- `use_cases/dashboard/pages/training_hub.py` - Section builders
- `use_cases/dashboard/core/config_metadata.py` - Split specs

**Phase 4 (Validation)**:
- `use_cases/dashboard/core/commands.py:562-651` - UpdateConfigCommand

**Phase 5 (Documentation)**:
- `use_cases/dashboard/ROADMAP.md` - Update status
- `use_cases/dashboard/docs/collaboration_notes.md` - Document changes

### Configuration Reference

**Prior Types** (`src/rcmvae/domain/config.py:168`):
- `"standard"` - Simple N(0,I) Gaussian
- `"mixture"` - Mixture of Gaussians with component-aware decoder
- `"vamp"` - Variational Mixture of Posteriors (learned pseudo-inputs)
- `"geometric_mog"` - Fixed geometric arrangement (circle/grid)

**Encoder Types** (`src/rcmvae/domain/config.py:162`):
- `"dense"` - MLP encoder (uses `hidden_dims`)
- `"conv"` - Convolutional encoder (hardcoded for 28x28)

**Reconstruction Losses** (`src/rcmvae/domain/config.py:147`):
- `"bce"` - Binary cross-entropy (for binary/MNIST), typical weight: 1.0
- `"mse"` - Mean squared error (for continuous), typical weight: 500.0

### Mathematical Notation

**Reference**: `docs/theory/conceptual_model.md` - §Notation Lock

- **c** - Component/channel index
- **z** - Continuous latent variable
- **r** or **q(c|x)** - Responsibilities (component probabilities from encoder)
- **τ_{c,y}** - τ-classifier map (component → label probabilities)
- **π** - Mixture weights
- **H[p̂_c]** - Usage entropy (empirical component usage)
- **α₀** - Laplace smoothing parameter for τ-classifier

### Visual Design System

**Colors**: See §Visual Language
**Typography**: See §Visual Language
**Spacing**: See §Visual Language

**Component Library** (Dash Bootstrap Components):
- `dbc.Input` - Text/number inputs
- `dbc.RadioItems` - Radio button groups
- `dbc.Checkbox` - Checkboxes
- `dbc.Button` - Buttons
- `dbc.Alert` - Alert banners
- `html.Div` - Generic containers

---

## Quick Start Checklist

Before starting implementation:

- [ ] Read this entire document
- [ ] Review `AGENTS.md` and `docs/theory/conceptual_model.md`
- [ ] Check current git status: `git log --oneline -10`
- [ ] Verify on correct branch: `claude/read-review-code-015t5G3dNEAWya6kiLQTvHSX`
- [ ] Pull latest changes: `git pull`
- [ ] Create feature branch: `git checkout -b feature/model-creation-redesign`
- [ ] Set up development environment: `poetry install`
- [ ] Run dashboard to verify current state: `poetry run python use_cases/dashboard/app.py`
- [ ] Review existing modal: Navigate to homepage, click "Create New Model"
- [ ] Review existing training hub: Load a model, go to Training Hub

During implementation:

- [ ] Follow phases in order (1 → 2 → 3 → 4 → 5)
- [ ] Test after each phase before moving to next
- [ ] Commit after each phase with clear messages
- [ ] Reference this document for specifications
- [ ] Use existing patterns from codebase
- [ ] Match visual design to specification

After implementation:

- [ ] Run full validation checklist
- [ ] Test all 4 prior types (standard, mixture, vamp, geometric)
- [ ] Verify existing models still work
- [ ] Update ROADMAP.md
- [ ] Create pull request with summary
- [ ] Reference commits from this design work (a65e7b4, 38a5925, 644f2e2, 4c2849f)

---

**End of Implementation Guide**
