# Verification Methodology

**Purpose:** Establish systematic verification for core functionality and future features.

**Philosophy:** Test what you implement. Observe what it learns.

---

## Quick Start

1. **Run core tests:** `poetry run pytest tests/test_core_correctness.py`
2. **Run feature tests:** `poetry run pytest tests/test_mixture_prior.py`
3. **View example baseline:** `artifacts/golden/mixture_prior/`

---

## Core Principle

The verification methodology is built on two pillars:

1. **Mathematical correctness** - Verify the implementation matches the theory
2. **Observable behavior** - Ensure the model learns what we expect

This is NOT about testing one feature. It's about establishing a **verification culture** where every feature follows a clear process: implement → test → diagnose → validate.

---

## For Core Functionality (One-Time Foundation)

### What We Test

The core tests in `tests/test_core_correctness.py` guard against breaking fundamental operations that ALL models and experiments depend on:

**Reconstruction Losses (MSE/BCE)**
- Used by every VAE configuration
- Tests: perfect reconstruction, error scaling, weight scaling, numerical stability

**KL Divergence (Standard)**
- Foundation for all VAE models
- Tests: zero KL for standard Gaussian, positive KL for non-standard, formula correctness

**Semi-Supervised Masking**
- Core mechanism for labeled/unlabeled splits
- Tests: masks unlabeled examples, only counts labeled samples, handles all-unlabeled case

**Training Loop Correctness**
- Ensures train/val splits don't corrupt experiments
- Tests: label ratio preservation, no data leakage

### Running Core Tests

```bash
# Run all core tests
poetry run pytest tests/test_core_correctness.py -v

# Run specific test
poetry run pytest tests/test_core_correctness.py::test_standard_kl_divergence_zero_for_standard_gaussian -v
```

**When to run:**
- Before committing changes to `src/training/losses.py`
- Before committing changes to `src/training/trainer.py`
- As part of CI/CD pipeline

**What failure means:**
If any core test fails, STOP. The foundation is broken, and all downstream work is suspect.

---

## For New Features (Repeatable Pattern)

Use the **mixture prior** as the reference example. Follow this exact pattern when adding:
- Label store
- OOD scoring
- VampPrior
- Curriculum learning
- Any other feature

### Step 1: Implement Feature

Normal development process in `src/`.

### Step 2: Add Feature Tests

**File:** `tests/test_{feature_name}.py`

Test YOUR feature's logic, not the core (that's already tested):

**Three categories:**
1. **Mathematical correctness** - Does it reduce to known cases?
2. **Shape/type correctness** - Are outputs the right shape and type?
3. **Key invariants** - What must always be true?

**Example: Mixture Prior** (`tests/test_mixture_prior.py`)

```python
# Mathematical correctness
def test_mixture_kl_reduces_to_standard_when_K_equals_1():
    """When K=1, mixture should equal standard KL."""
    # ... test implementation

# Shape/type correctness
def test_mixture_encoder_output_shapes():
    """Verify encoder returns (component_logits, z_mean, z_log, z)."""
    # ... test implementation

# Key invariants
def test_responsibilities_sum_to_one():
    """Responsibilities must sum to 1.0 (fundamental property)."""
    # ... test implementation
```

**Pattern for future features:**
```python
# tests/test_label_store.py

def test_label_store_retrieval_correctness():
    """Nearest neighbor retrieval returns closest stored examples."""
    pass

def test_label_store_output_shapes():
    """Retrieved representations have correct shape."""
    pass

def test_label_store_size_bounds():
    """Store never exceeds max_size parameter."""
    pass
```

### Step 3: Add Feature Diagnostics

**File:** `scripts/diagnostics.py`

Create a diagnostic class that exports observable behavior:

```python
class MyFeatureDiagnostics(DiagnosticExporter):
    """Diagnostic exporter for my_feature.

    Exports:
        - observable_1.npy: Description
        - observable_2.npy: Description
        - summary.txt: Human-readable summary
    """

    def should_export(self, config: SSVAEConfig) -> bool:
        """Return True if feature is enabled."""
        return config.use_my_feature

    def export(self, model, data, labels, output_dir):
        """Export diagnostics to output_dir/diagnostics/."""
        # Compute observables
        # Save to .npy files
        # Generate summary text
        pass

# Add to registry
DIAGNOSTICS = [
    MixtureDiagnostics(),
    MyFeatureDiagnostics(),  # <-- Add here
]
```

**Example: Mixture Prior Diagnostics**

Exports:
- `component_usage.npy` - How many samples assigned to each component?
- `component_entropy.npy` - How uncertain are component assignments?
- `per_class_component_usage.npy` - Do components specialize to classes?
- `mixture_summary.txt` - Human-readable summary with warnings

These help answer:
- ✓ Are all components being used?
- ✓ Is the model collapsing to fewer components?
- ✓ Do components correlate with semantic classes?

**Integration:**

The `scripts/compare_models.py` tool automatically runs all applicable diagnostics:

```python
for diagnostic in DIAGNOSTICS:
    if diagnostic.should_export(config):
        diagnostic.export(model, X_val, y_val, output_dir)
```

No need to modify the comparison tool - just add your diagnostic class and it runs automatically!

### Step 4: Document

Add section to this file documenting:
- What the feature does
- What tests exist
- What diagnostics are exported
- What "healthy" behavior looks like

---



## Future Feature Checklist

When adding **label store** / **OOD scoring** / **VampPrior** / etc:

- [ ] **Implement** in `src/`
- [ ] **Test** in `tests/test_{feature}.py` (follow mixture example)
  - [ ] Mathematical correctness test
  - [ ] Shape/type correctness test
  - [ ] Key invariant test
- [ ] **Diagnose** in `scripts/diagnostics.py`
  - [ ] Add `{Feature}Diagnostics` class
  - [ ] Export observable behaviors
  - [ ] Add to `DIAGNOSTICS` registry
- [ ] **Validate** with advisor
  - [ ] Create `configs/{feature}_baseline.yaml`
  - [ ] Run experiment and review outputs
  - [ ] Save to `artifacts/golden/{feature}/`
- [ ] **Document** in this file
  - [ ] Add "Golden Baseline: {Feature}" section
  - [ ] Document validated behavior
  - [ ] Include reproduction instructions

---

## Relationship to Other Docs

**`CONTRIBUTING.md`:**
Before adding features, see verification methodology here.

**`IMPLEMENTATION.md`:**
All loss functions have tests in `tests/test_core_correctness.py`.

**`README.md`:**
Links to this doc in documentation map.

---

## FAQ

**Q: Do I need to test everything?**
A: Test what you implement. Core tests cover the foundation. Your feature tests should cover YOUR logic.

**Q: How do I know if my feature is working correctly?**
A: If tests pass AND diagnostics match expected behavior (validated with advisor).

**Q: Can I skip the advisor validation step?**
A: Not for the first implementation of a feature. You need ground truth to know what "correct" looks like. After that, you can compare against the golden baseline.

**Q: What if my feature doesn't have obvious diagnostics?**
A: Every feature has observable behavior. Talk with advisor to identify what to measure.

**Q: How often should I run core tests?**
A: Before every commit to `src/training/`. Consider adding as pre-commit hook.

**Q: This seems like a lot of work!**
A: The first feature (mixture prior) establishes the pattern. After that, it's ~15-30 minutes per feature to add tests + diagnostics. Much faster than debugging mysterious failures later!

---

## Key Insight

This isn't "test mixture prior" - it's **"establish verification culture"**.

Mixture is just the first feature to go through the process. Every future feature follows the same pattern:

```
Implement → Test → Diagnose → Validate → Document
```

The result: **Confidence to build quickly without breaking things.**
