# Kickstart Prompt for Model Creation & Training Hub Redesign

Use this prompt to start implementation in a fresh context window:

---

## Task

Implement the SSVAE Dashboard model creation and training hub redesign according to the complete specifications in `use_cases/dashboard/docs/IMPLEMENTATION_GUIDE.md`.

## Context

**Problem**: Users cannot create mixture/vamp/geometric models from the UI. Models are always created with hardcoded `SSVAEConfig()` defaults (prior_type="standard"). When users change structural parameters like `prior_type` in the training hub or configuration page, only the config is updated—the model architecture is NOT rebuilt. This causes `ValueError: Mixture responsibilities unavailable` when training.

**Solution**: Two-part redesign:
1. **Model Creation**: Expand homepage modal to expose all structural parameters (encoder type, prior type, latent dim, etc.) BEFORE model creation. Once created, these are locked forever.
2. **Training Configuration System**: Unified backend redesign for both Training Hub (quick controls) and Full Configuration Page (comprehensive view):
   - Show structural parameters as **read-only** with lock icon (architecture summary)
   - Show ALL modifiable parameters as **editable** (filtered from structural ones)
   - Conditional sections/tabs based on prior type
   - Correct mathematical terminology throughout

**Current Branch**: `claude/read-review-code-015t5G3dNEAWya6kiLQTvHSX`

**Design Commits**:
- `a65e7b4` - Temporary fallback fix (prevents crashes)
- `38a5925` - Initial design documentation
- `644f2e2` - UI refinements (removed advanced options)
- `4c2849f` - Mathematical terminology corrections
- `de1559b` - Consolidated implementation guide
- `8ea13eb` - Kickstart prompt
- `7d972a9` - Added Full Configuration Page redesign specs

## Your Mission

**Read the complete implementation guide first**:
```bash
# This is your single source of truth
use_cases/dashboard/docs/IMPLEMENTATION_GUIDE.md
```

Then implement the redesign in 6 phases:

### Phase 1: Model Creation Enhancement (3-5 days)
- Expand `use_cases/dashboard/pages/home.py` modal with all structural parameters
- Update `CreateModelCommand` to accept architecture configuration
- Add conditional rendering callbacks for show/hide fields
- **Files**: `pages/home.py`, `core/commands.py`, `callbacks/home_callbacks.py` (new)

### Phase 2: Training Hub Architecture Summary (1 day)
- Add read-only architecture summary at top of training hub
- Show locked structural parameters with 🔒 icon
- **Files**: `pages/training_hub.py`

### Phase 3: Training Hub Parameter Reorganization (2 days)
- Split `config_metadata.py` into structural vs modifiable parameters
- Create conditional section builders based on prior type
- Reorganize training hub with logical sections
- **Files**: `pages/training_hub.py`, `core/config_metadata.py`

### Phase 4: Full Configuration Page Redesign (2-3 days)
- Add architecture summary at top (read-only structural params with lock icon)
- Show ALL modifiable parameters in editable tabs (use `get_modifiable_field_specs()`)
- Add conditional tabs based on prior type
- Reorganize into logical sections/tabs
- Apply same backend logic as Training Hub (unified system)
- **Files**: `core/config_metadata.py`, `pages/training.py`

### Phase 5: UpdateConfigCommand Validation (1 day)
- Block changes to structural parameters after creation
- Add clear error messages
- **Files**: `core/commands.py`

### Phase 6: Polish & Documentation (1 day)
- Visual polish and consistency check
- Update ROADMAP.md
- **Files**: `ROADMAP.md`, `docs/collaboration_notes.md`

## Key Requirements

**Terminology** (from `docs/theory/conceptual_model.md`):
- ✅ "Usage Entropy Weight" with H[p̂_c] notation (NOT "component diversity")
- ✅ "Component" or "Channel" (NOT "cluster")
- ✅ "Responsibilities" r = q(c|x)
- ✅ "τ-Classifier" (latent-only classification)

**Visual Design** (match existing):
- Colors: Primary #C10A27, Secondary #45717A, Neutrals #000000/#6F6F6F/#C6C6C6
- Typography: 'Open Sans', ui-monospace for numbers
- Spacing: 24px sections, 16px groups, 6px label-input

**Conditional Rendering**:
- Show hidden layers input only when Dense encoder selected
- Show component options only when mixture/vamp/geometric prior selected
- Show component-aware decoder only for mixture/geometric (not vamp)
- Training hub shows different sections based on prior type

## Success Criteria

- ✅ Can create models with all 4 prior types from homepage
- ✅ Conditional fields show/hide correctly in model creation modal
- ✅ Training hub shows architecture summary (read-only with lock icon)
- ✅ Training hub shows only essential modifiable parameters
- ✅ Training hub has conditional sections based on prior type
- ✅ Full configuration page shows architecture summary (read-only with lock icon)
- ✅ Full configuration page shows ALL modifiable parameters in editable tabs
- ✅ Full configuration page has conditional tabs based on prior type
- ✅ Structural parameters visible but clearly locked (not editable)
- ✅ Cannot save changes to structural parameters
- ✅ Existing models still load and work
- ✅ Mathematical terminology used correctly
- ✅ Visual design matches specification

## Getting Started

1. **Read the implementation guide in full**: `use_cases/dashboard/docs/IMPLEMENTATION_GUIDE.md`
2. **Review essential docs**:
   - `use_cases/dashboard/README.md` - Dashboard structure
   - `docs/theory/conceptual_model.md` - Terminology
   - `src/rcmvae/domain/config.py` - Parameter definitions
3. **Verify current state**:
   ```bash
   git status
   git log --oneline -5
   # Should see: 7d972a9 (full config specs), 8ea13eb (kickstart), de1559b (guide)
   ```
4. **Start with Phase 1**: Model creation enhancement
5. **Test after each phase** before moving to next
6. **Commit after each phase** with clear messages

## Important Notes

- Follow existing patterns from codebase (Command pattern, immutable state, conditional callbacks)
- Reference IMPLEMENTATION_GUIDE.md for detailed specifications, code snippets, and file locations
- Each section includes line number references (e.g., `pages/home.py:427-616`)
- Use existing helpers like `_render_quick_control()` where applicable
- Test with all 4 prior types: standard, mixture, vamp, geometric_mog

## Questions?

If anything is unclear:
1. Check IMPLEMENTATION_GUIDE.md first (1935 lines, very comprehensive)
2. Reference existing code patterns in the files mentioned
3. Review conceptual model for terminology questions
4. Ask user for clarification on design decisions

---

**Now begin Phase 1: Model Creation Enhancement**

Start by reading the Model Creation Redesign section in IMPLEMENTATION_GUIDE.md (lines 164-517), then update the homepage modal in `use_cases/dashboard/pages/home.py`.
