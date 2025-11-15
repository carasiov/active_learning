"""Architecture code generation for experiment naming.

This module auto-generates human-readable architecture codes from config,
following the naming format: {name}__{architecture_code}__{timestamp}

Architecture code structure: {prior}_{classifier}_{decoder}

Design principles (from AGENTS.md):
- Single source of truth: Config → architecture code (auto-generated)
- Extensible: New features add encoding functions
- Validated: Fail fast with clear errors for invalid combinations
- Stable: Code format changes are breaking changes (document carefully)

Adding a new feature:
1. Add encoding logic to appropriate function (encode_prior/classifier/decoder)
2. Update NAMING_LEGEND_TEMPLATE below
3. Add validation rule to validation.py if needed
4. Update tests in tests/test_naming.py

Example:
    from use_cases.experiments.src.naming import generate_architecture_code

    code = generate_architecture_code(config)
    # → "mix10-dir_tau_ca-het" for mixture prior with τ-classifier
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model.ssvae.config import SSVAEConfig


def generate_architecture_code(config: SSVAEConfig) -> str:
    """Auto-generate architecture code from config.

    Args:
        config: SSVAE configuration object

    Returns:
        Architecture code string like "mix10-dir_tau_ca-het"

    Raises:
        ValueError: If config has unknown or invalid architecture choices

    Example:
        >>> config = SSVAEConfig(
        ...     prior_type="mixture",
        ...     num_components=10,
        ...     dirichlet_alpha=5.0,
        ...     use_tau_classifier=True,
        ...     use_component_aware_decoder=True,
        ...     use_heteroscedastic_decoder=True
        ... )
        >>> generate_architecture_code(config)
        'mix10-dir_tau_ca-het'
    """
    prior_code = _encode_prior(config)
    classifier_code = _encode_classifier(config)
    decoder_code = _encode_decoder(config)

    return f"{prior_code}_{classifier_code}_{decoder_code}"


def _encode_prior(config: SSVAEConfig) -> str:
    """Encode prior type with parameters and modifiers.

    Format:
    - standard: "std"
    - mixture: "mix{K}" with optional "-dir" modifier
    - vamp: "vamp{K}-{km|rand}"
    - geometric_mog: "geo{K}-{circle|grid}"

    Args:
        config: SSVAE configuration

    Returns:
        Prior code string

    Raises:
        ValueError: If prior_type is unknown or configuration is invalid
    """
    prior_type = config.prior_type

    if prior_type == "standard":
        return "std"

    elif prior_type == "mixture":
        code = f"mix{config.num_components}"

        # Add Dirichlet modifier if enabled
        if config.dirichlet_alpha is not None and config.dirichlet_alpha > 0:
            code += "-dir"

        return code

    elif prior_type == "vamp":
        code = f"vamp{config.num_components}"

        # VampPrior requires initialization method
        init_method = config.vamp_pseudo_init_method
        if init_method == "kmeans":
            code += "-km"
        elif init_method == "random":
            code += "-rand"
        else:
            raise ValueError(
                f"VampPrior requires vamp_pseudo_init_method='kmeans' or 'random', "
                f"got '{init_method}'"
            )

        return code

    elif prior_type == "geometric_mog":
        code = f"geo{config.num_components}"

        # Geometric MoG requires arrangement
        arrangement = config.geometric_arrangement
        if arrangement == "circle":
            code += "-circle"
        elif arrangement == "grid":
            code += "-grid"
        else:
            raise ValueError(
                f"Geometric MoG requires geometric_arrangement='circle' or 'grid', "
                f"got '{arrangement}'"
            )

        return code

    else:
        raise ValueError(f"Unknown prior_type: '{prior_type}'")


def _encode_classifier(config: SSVAEConfig) -> str:
    """Encode classifier type.

    Format:
    - τ-classifier: "tau"
    - Standard head: "head"

    Args:
        config: SSVAE configuration

    Returns:
        Classifier code string
    """
    if config.use_tau_classifier:
        return "tau"
    else:
        return "head"


def _encode_decoder(config: SSVAEConfig) -> str:
    """Encode decoder features.

    Format builds up modifiers:
    - Component-aware: adds "ca"
    - Heteroscedastic: adds "het"
    - Plain (no features): "plain"

    Modifiers are joined with "-" in canonical order: ca-het

    Future extensions (examples):
    - Contrastive: add "contr" → "ca-het-contr"
    - Other features: append to list

    Args:
        config: SSVAE configuration

    Returns:
        Decoder code string
    """
    features = []

    # Order matters for consistency (alphabetical for simplicity)
    if config.use_component_aware_decoder:
        features.append("ca")

    if config.use_heteroscedastic_decoder:
        features.append("het")

    # Future: Add contrastive when fully integrated
    # if config.use_contrastive:
    #     features.append("contr")

    if not features:
        return "plain"

    return "-".join(features)


def generate_naming_legend() -> str:
    """Generate naming legend documentation.

    This creates a markdown document explaining the architecture code format.
    Should be regenerated whenever naming rules change.

    Returns:
        Markdown-formatted legend string

    Usage:
        Save to results/NAMING_LEGEND.md at experiment startup:

        >>> legend = generate_naming_legend()
        >>> Path("results/NAMING_LEGEND.md").write_text(legend)
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return NAMING_LEGEND_TEMPLATE.format(timestamp=timestamp)


# Naming legend template (updated when encoding rules change)
NAMING_LEGEND_TEMPLATE = """# Experiment Naming Legend

**Auto-generated:** {timestamp}

This document describes the experiment directory naming convention used by
the experiment management system.

## Directory Format

```
{{experiment-name}}__{{architecture_code}}__{{timestamp}}/
```

**Example:**
```
baseline__mix10-dir_tau_ca-het__20251112_143022/
```

This translates to:
- **Experiment series:** baseline
- **Architecture:** Mixture prior (K=10) with Dirichlet, τ-classifier, component-aware heteroscedastic decoder
- **Run time:** November 12, 2025 at 14:30:22

---

## Architecture Code Structure

Architecture code has three components: `{{prior}}_{{classifier}}_{{decoder}}`

### Prior Codes

Encodes the prior distribution p(z) and p(c):

| Code | Meaning | Config |
|------|---------|--------|
| `std` | Standard Gaussian N(0,I) | `prior_type: "standard"` |
| `mix{{K}}` | Mixture of Gaussians, K components | `prior_type: "mixture"`, `num_components: K` |
| `mix{{K}}-dir` | Mixture with Dirichlet prior on π | Above + `dirichlet_alpha: > 0` |
| `vamp{{K}}-km` | VampPrior, k-means init | `prior_type: "vamp"`, `vamp_pseudo_init_method: "kmeans"` |
| `vamp{{K}}-rand` | VampPrior, random init | `prior_type: "vamp"`, `vamp_pseudo_init_method: "random"` |
| `geo{{K}}-circle` | Geometric MoG, circle arrangement | `prior_type: "geometric_mog"`, `geometric_arrangement: "circle"` |
| `geo{{K}}-grid` | Geometric MoG, grid arrangement | `prior_type: "geometric_mog"`, `geometric_arrangement: "grid"` |

**Notes:**
- K is the number of mixture components (`num_components`)
- Dirichlet modifier appears only when `dirichlet_alpha` is set and positive
- VampPrior always includes initialization method suffix
- Geometric MoG always includes arrangement suffix

### Classifier Codes

Encodes the classification strategy:

| Code | Meaning | Config |
|------|---------|--------|
| `tau` | τ-classifier (latent-only, responsibility-based) | `use_tau_classifier: true` |
| `head` | Standard classifier head on z | `use_tau_classifier: false` |

**Requirements:**
- `tau` requires mixture-based prior (`mixture`, `vamp`, or `geometric_mog`)
- `tau` requires `num_components >= num_classes`

### Decoder Codes

Encodes decoder features (modifiers are cumulative):

| Code | Meaning | Config |
|------|---------|--------|
| `plain` | Standard decoder (no special features) | Default |
| `ca` | Component-aware decoder | `use_component_aware_decoder: true` |
| `het` | Heteroscedastic (learns σ²) | `use_heteroscedastic_decoder: true` |
| `ca-het` | Both component-aware and heteroscedastic | Both enabled |

**Notes:**
- Modifiers combine in canonical order: `ca-het` (not `het-ca`)
- Component-aware requires mixture-based prior
- Future features will extend this list (e.g., `ca-het-contr` for contrastive)

---

## Validation Rules

The system enforces these constraints at config load time:

1. **τ-classifier requires mixture prior:**
   - Cannot use `tau` with `std` prior
   - Valid: `mix*_tau_*`, `vamp*_tau_*`, `geo*_tau_*`

2. **Component-aware decoder requires mixture prior:**
   - Cannot use `ca` with `std` prior
   - Valid: `mix*_*_ca*`, `vamp*_*_ca*`, `geo*_*_ca*`

3. **VampPrior requires initialization method:**
   - Must specify `-km` or `-rand`
   - Invalid: `vamp10` (missing suffix)

4. **Geometric MoG requires arrangement:**
   - Must specify `-circle` or `-grid`
   - Invalid: `geo9` (missing suffix)

5. **Grid arrangement requires perfect square K:**
   - Valid: `geo4-grid`, `geo9-grid`, `geo16-grid`
   - Invalid: `geo10-grid` (10 is not a perfect square)

---

## Example Architectures

### Standard Configurations

**Basic VAE:**
```
quick__std_head_plain__20251112_143022
```
- Standard Gaussian prior
- Standard classifier head
- Plain decoder

**Full mixture model:**
```
baseline__mix10-dir_tau_ca-het__20251112_143022
```
- Mixture prior with 10 components and Dirichlet regularization
- τ-classifier for latent-only classification
- Component-aware + heteroscedastic decoder

### VampPrior Experiments

**Spatial clustering with k-means:**
```
vamp-test__vamp20-km_tau_ca-het__20251112_150000
```
- VampPrior with 20 pseudo-inputs, k-means initialization
- τ-classifier
- Component-aware + heteroscedastic decoder

**Random initialization baseline:**
```
vamp-baseline__vamp20-rand_tau_ca__20251112_151500
```
- VampPrior with random initialization (for comparison)
- Component-aware only (no heteroscedastic)

### Geometric Debugging

**Circle arrangement for visualization:**
```
debug__geo8-circle_tau_plain__20251112_160000
```
- 8 components arranged in circle (2D latent space)
- τ-classifier
- Plain decoder (no component-aware for visualization)

---

## Extending the Naming System

When adding a new architectural feature:

1. **Update encoder function:** Add logic to `use_cases/experiments/src/naming.py`
2. **Update validation:** Add constraint to `use_cases/experiments/src/validation.py`
3. **Regenerate legend:** This file is auto-generated, no manual edits
4. **Update tests:** Add test cases to `tests/test_naming.py`
5. **Document in README:** Update experiment guide with usage examples

**Breaking changes:** Changing the code format affects existing result directories.
Plan carefully and version-bump if necessary.

---

## Related Documentation

- **Config reference:** `src/model/ssvae/config.py::SSVAEConfig`
- **Theory:** `docs/theory/conceptual_model.md`
- **Implementation:** `docs/development/implementation.md`
- **Experiments:** `use_cases/experiments/README.md`
"""
