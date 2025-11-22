# Documentation Update Specification
**Context**: Modular Decoder Refactor (Phases 1-2 Complete)
**Goal**: Update codebase documentation to reflect the new modular decoder architecture

---

## Summary of Changes

The modular decoder refactor introduced:
1. **New Architecture Pattern**: Composition-based decoders (conditioning + backbone + output)
2. **Feature Combination Support**: FiLM + Heteroscedastic now possible (was blocked by silent override)
3. **New Modules**: `decoder_modules` package with `conditioning.py`, `backbones.py`, `outputs.py`
4. **Config Validation**: Added strict validation to prevent ambiguous/incompatible flags
5. **Deprecated Classes**: 10 legacy decoder classes marked deprecated
6. **Validation**: 29% reconstruction improvement with FiLM vs concatenation baseline

**References**:
-  [refactor_validation_report.md](file:///home/acarasiov/.gemini/antigravity/brain/6e810ef4-4c1e-4ae8-b704-5f172834d2e1/refactor_validation_report.md)
- `src/rcmvae/domain/components/decoder_modules/` (new code)
- `docs/projects/decentralized_latents/implementation_spec.md` (refactor plan)

---

## Documentation Files to Update

### 1. **docs/development/architecture.md** (HIGH PRIORITY)
**Current State**: Describes legacy decoder classes and factory pattern
**Changes Needed**:
- **Add Section**: "Modular Decoder Architecture"
  - Explain composition pattern (conditioner + backbone + output_head)
  - Document the 3 module types and their implementations
  - Show example: How `ModularConvDecoder` composes modules
- **Update Section**: "Component Structure → Decoders"
  - Mark legacy classes as deprecated
  - Explain migration path to modular decoders
- **Update Section**: "Factory Pattern"
  - Document new `build_decoder` logic (FiLM > Concat > Noop priority)
  - Explain config validation

**Code References**:
- `src/rcmvae/domain/components/decoder_modules/__init__.py`
- `src/rcmvae/domain/components/decoders.py` (ModularConvDecoder/ModularDenseDecoder)
- `src/rcmvae/domain/components/factory.py` (build_decoder function)

---

### 2. **docs/development/implementation.md** (MEDIUM PRIORITY)
**Current State**: Implementation details for various components
**Changes Needed**:
- **Add Section**: "Implementing a New Decoder Module"
  - How to create a new conditioner (protocol requirements)
  - How to create a new backbone
  - How to create a new output head
  - Testing requirements (shape preservation, gradient flow)
- **Update Section**: "Decoder Implementation"
  - Note that new decoders should use modular composition
  - Deprecation timeline for legacy classes

**Code References**:
- `src/rcmvae/domain/components/decoder_modules/conditioning.py` (FiLMLayer example)
- `tests/test_decoder_conditioning_modules.py` (test examples)

---

### 3. **docs/theory/conceptual_model.md** (MEDIUM PRIORITY)
**Current State**: Describes "component-aware" decoders conceptually
**Changes Needed**:
- **Update Section**: "Decoder Architectures"
  - Add "FiLM Conditioning" as a conditioning method
  - Explain the distinction: Conditioning (how?) vs Output (what?)
  - Note that FiLM + Heteroscedastic is now supported

**Conceptual Addition**:
```markdown
### Decoder Conditioning Methods
1. **No Conditioning**: Standard VAE decoder (ignores component identity)
2. **Concatenation**: Appends component embedding to latent
3. **FiLM (Feature-wise Linear Modulation)**: Modulates decoder features via γ, β parameters
```

---

### 4. **docs/theory/mathematical_specification.md** (LOW PRIORITY)
**Current State**: Mathematical definitions of loss, KL, etc.
**Changes Needed**:
- **Add Section**: "FiLM Conditioning Equations"
  ```
  γ, β = MLP(e_k)        # Component embedding → modulation params
  h' = γ ⊙ h + β         # Feature-wise affine transform
  ```
- **Add Section**: "Heteroscedastic Output"
  ```
  μ, σ = Decoder(z)
  σ = clip(softplus(σ_raw) + σ_min, σ_min, σ_max)
  p(x|z) = N(x; μ, σ²I)
  ```

**Note**: Only add if mathematical rigor is important to the user. Otherwise, skip.

---

### 5. **docs/theory/implementation_roadmap.md** (HIGH PRIORITY)
**Current State**: Progress tracker for implementation pillars
**Changes Needed**:
- **Update Section**: "Completed Pillars"
  - Add "Modular Decoder Architecture" as completed
  - Mark "Component-Aware Decoder" as legacy (superseded by modular)
- **Add Note**: FiLM + Heteroscedastic combination validated (29% improvement)

---

### 6. **AGENTS.md** (ALREADY UPDATED ✅)
**Status**: Already reflects the decentralized latents project and known issues
**No further changes needed**

---

## Execution Plan for Agent

**Step 1**: Review References
- Read `refactor_validation_report.md`
- Skim `docs/projects/decentralized_latents/implementation_spec.md`
- View code in `src/rcmvae/domain/components/decoder_modules/`

**Step 2**: Update High Priority Docs
- `architecture.md`: Add modular decoder section, update factory logic
- `implementation_roadmap.md`: Mark modular architecture as complete

**Step 3**: Update Medium Priority Docs
- `implementation.md`: Add "Implementing a New Decoder Module" guide
- `conceptual_model.md`: Update decoder architecture section with FiLM

**Step 4**: (Optional) Update Low Priority
- `mathematical_specification.md`: Add FiLM/Heteroscedastic equations

**Step 5**: Verification
- Ensure all internal links work (use relative paths)
- Confirm no references to "silent override bug" remain (it's fixed)
- Check that deprecated classes are clearly marked

---

## Success Criteria
- [ ] `architecture.md` has a complete "Modular Decoder Architecture" section
- [ ] All references to legacy decoders note deprecation
- [ ] Factory priority logic (FiLM > Concat > Noop) is documented
- [ ] Agent guide for creating new decoder modules exists
- [ ] `implementation_roadmap.md` reflects modular decoder completion
