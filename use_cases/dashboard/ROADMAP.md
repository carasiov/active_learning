# Dashboard Integration Roadmap

**Purpose**: Guide the evolution of the dashboard from a basic training interface to a full active learning workbench that leverages the rich experiment infrastructure.

**Context**: This document bridges the gap between the CLI experiment workflow (`use_cases/experiments/`) and the interactive dashboard (`use_cases/dashboard/`), enabling the dashboard to generate experiment-quality outputs and visualizations.

## 📚 Related Documentation

**Before starting work**, review these guides for context:

### Essential Reading
- **[AGENTS.md](../../../AGENTS.md)** - How to navigate the documentation ecosystem and understand system architecture
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - Dashboard internals, backend integration points, debugging toolkit, extension workflow
- **[Collaboration Notes](docs/collaboration_notes.md)** - Current system snapshot, debugging playbook, recent work
- **[Dashboard README](README.md)** - Quick start, project structure, routes

### Background Context
- **[Experiments Guide](../../experiments/README.md)** - Experiment outputs (REPORT.md, figures), run directory structure
- **[Conceptual Model](../../../docs/theory/conceptual_model.md)** - Mixture model theory, component specialization, τ-classifier, OOD scoring
- **[Mathematical Specification](../../../docs/theory/mathematical_specification.md)** - Formal definitions of responsibilities, τ-matrix, uncertainty
- **[Architecture Guide](../../../docs/development/architecture.md)** - Design patterns, service layer, state management

### Implementation References
- **[State Management Plan](docs/dashboard_state_plan.md)** - State architecture, AppStateManager design
- **[Config Metadata](core/config_metadata.py)** - How training parameters are defined and validated
- **[State Models](core/state_models.py)** - Immutable dataclasses representing app state

### Quick References
- **Extension Checklist**: [DEVELOPER_GUIDE.md §5](docs/DEVELOPER_GUIDE.md) - Step-by-step workflow for adding features
- **Debugging Playbook**: [collaboration_notes.md §Debugging](docs/collaboration_notes.md) - Logs, tests, common issues
- **Backend Touchpoints**: [DEVELOPER_GUIDE.md §2](docs/DEVELOPER_GUIDE.md) - Where dashboard interacts with SSVAE backend

---

## Executive Summary

### Current State (As of November 2025)

**✅ Completed**:
- **Phase 0: Stabilization** (Complete)
  - Service architecture refactoring
  - `TrainingService`, `ModelService`, `LabelingService` abstractions
  - `AppStateManager` eliminates global state
  - Commands use dependency injection
  - Training state lifecycle fixes
  - Defensive error handling

- **Phase 1.1 & 1.2: Experiment Integration** (Complete, commit `7ceec67`)
  - ✅ REPORT.md generation for dashboard runs
  - ✅ Run records persisted to `ModelState.runs` and `runs.json`
  - ✅ "Recent Runs" section in Training Hub (last 5 runs)
  - ✅ Run Viewer page at `/model/{id}/run/{run_id}`
  - ✅ Displays run metadata: epochs, labeled samples, metrics
  - ⚠️ Image rendering in reports (known limitation, workaround provided)

- **Mixture Model Support**:
  - Captures `responsibilities` and `pi_values` during training
  - Enables component specialization analysis
  - Data ready for visualization (Phase 1.3)

- **Basic Active Learning Loop**:
  - Label samples via click → Train → See updated latent space → View run report

**🚧 In Progress**:
- Phase 1.3: Expose Mixture Diagnostics in UI (Next)

**❌ Missing**:
- τ-matrix (component→label mapping) visualizations in UI
- π evolution tracking across training sessions in UI
- Uncertainty/OOD highlighting for strategic labeling
- Component specialization diagnostics in UI
- Real-time mixture updates during training

### Vision: Dashboard as Active Learning Workbench

The target system enables this workflow:

1. **Diagnose**: Visualize component specialization, τ-matrix, uncertainty regions
2. **Strategize**: Identify high-impact samples using OOD scores, component ambiguity
3. **Label**: Click to label strategically chosen samples
4. **Train**: Run training with real-time component evolution tracking
5. **Analyze**: Review experiment-quality REPORT.md with full diagnostics
6. **Iterate**: Repeat with insights from previous run

**Key Principle**: Dashboard training runs should produce the same rich outputs as CLI experiments, just with interactive labeling in the loop.

---

## Phase 0: Stabilization ✅ COMPLETED

**Goal**: Make the current dashboard reliable and bug-free.

**Status**: ✅ Complete (See `PHASE0_TESTING.md` for verification checklist)

### 0.1 Fix Training State Lifecycle ✅

**Implementation**:
- ✅ All `state_manager.update_state()` calls verified (commits: `a7dde90`, `414778e`)
- ✅ State transition logging added to `AppStateManager.update_state()` (commit: `9784986`)
- ✅ Enhanced `StartTrainingCommand.validate()` with thread safety check (commit: `9784986`)
- ✅ Consistent use of `update_state()` throughout codebase

**Results**:
- Training lifecycle works: start → stop → start, start → complete → start
- State always returns to IDLE after training
- Clear logging for debugging state issues

### 0.2 Fix UI Inconsistencies ✅

**Implementation**:
- ✅ Added `prevent_initial_call=True` to action callbacks (commit: `45800ba`)
- ✅ Epochs input investigated - behavior is expected (resets to config on navigation)
- ✅ Input validation present in commands
- ✅ Navigation paths tested and working

**Results**:
- Navigation smooth: home → model → training-hub → config
- Form inputs behave predictably
- No unexpected resets during usage

### 0.3 Add Defensive Error Handling ✅

**Implementation**:
- ✅ Try-catch around `model.predict_batched()` in both training workers (commit: `9784986`)
- ✅ Shape validation for `responsibilities` and `pi_values` (commit: `9784986`)
- ✅ Null checks audited - all callbacks safe
- ✅ Comprehensive error logging and user messages
- ✅ Robust model loading with architecture mismatch handling (commit: `082cef7`)

**Results**:
- No crashes from prediction failures
- Models with incompatible checkpoints load gracefully
- User-friendly error messages with actionable guidance
- System remains stable after any error

### 0.4 Mixture Model Support ✅

**Implementation**:
- ✅ `_predict_outputs()` helper with validation (commit: `9784986`)
- ✅ Graceful handling of None responsibilities
- ✅ Shape validation prevents crashes
- ✅ Training hub worker has full mixture support (commit: `2eef358`)

**Results**:
- Mixture models work end-to-end
- Non-mixture models unaffected (responsibilities=None handled)
- Defensive error handling prevents crashes

**Key Commits**:
- `9784986` - Comprehensive error handling and logging
- `414778e` - Fixed state update in clear_hub_terminal
- `082cef7` - Robust model loading with mismatch handling
- `45800ba` - UI consistency improvements
- `2eef358` - Mixture model support in training hub
- `a7dde90` - State update fixes

**Testing**: See `PHASE0_TESTING.md` for comprehensive test checklist

---

## Phase 1: Experiment Integration Foundation

**Goal**: Enable dashboard training runs to generate experiment-quality outputs.

### 1.1 Leverage `generate_dashboard_run()` Infrastructure ✅

**Status**: ✅ Complete (Pre-existing implementation verified November 2025)

**Background**: The function `use_cases/dashboard/core/run_generation.py::generate_dashboard_run()` already exists and creates full experiment outputs. It's called in `CompleteTrainingCommand` and properly saves run records.

**See Also**: [Experiments Guide - Run Directory Layout](../../experiments/README.md) for output structure details

**Implementation Details**:
- `CompleteTrainingCommand.execute()` (lines 453-483) calls `generate_dashboard_run()`
- Run records are saved via `append_run_record()` to `{model_dir}/runs.json`
- `ModelState.runs` is updated with the new run record (lines 470-473)
- All metadata fields present: `run_id`, `timestamp`, `report_path`, `results_root`, `metrics`, etc.

**Design Decision**:
- ✅ Using `generate_dashboard_run()` - no duplication of experiment infrastructure
- ✅ Run records stored in `ModelState.runs` for UI access
- ✅ Links to generated REPORT.md instead of duplicating content

**Acceptance Criteria**:
- ✅ After training, `ModelState.runs` contains new `RunRecord`
- ✅ `RunRecord` includes paths to REPORT.md and figures
- ✅ Ready for UI integration (completed in Phase 1.2)

### 1.2 Surface Experiment Outputs in Training Hub ✅

**Status**: ✅ Complete (Implemented November 2025, commit `7ceec67`)

**Goal**: Make REPORT.md and figures accessible from the UI.

**Implementation**:
1. ✅ Created `run_viewer.py` page to display REPORT.md content
2. ✅ Added route `/model/{id}/run/{run_id}` in `app.py`
3. ✅ Added "Recent Runs" section to training hub showing last 5 runs
4. ✅ Each run displays: run ID, timestamp, epochs, labeled samples, validation loss
5. ✅ "View Report →" button links to run viewer page

**Files Modified**:
- `use_cases/dashboard/pages/run_viewer.py` (new)
- `use_cases/dashboard/app.py` (routing)
- `use_cases/dashboard/pages/training_hub.py` (Recent Runs section)

**UI Design** (Implemented):
```
Training Hub
├── Training Controls (existing)
├── Loss Curves (existing)
└── Recent Runs (NEW) ✅
    ├── Run {run_id} • {timestamp} [View Report →]
    │   └── Epochs: {start}→{end} | Labeled: {count}/{total} | Val Loss: {loss}
    └── Shows total run count, handles empty state gracefully
```

**Acceptance Criteria**:
- ✅ User can click "View Report" to see full REPORT.md
- ⚠️ Images in report: Not yet rendered (known limitation, see below)
- ✅ Run metadata shows in UI (timestamp, epochs, best metrics)

**Known Limitations**:
- **Image Rendering**: REPORT.md references images with relative paths (e.g., `figures/core/loss_comparison.png`). These don't render in the web viewer because:
  - Images are not served as static assets
  - Dash's `dcc.Markdown` component can't resolve relative file paths
- **Workaround**: Info banner displays the report file path for users to view locally
- **Future Enhancement** (not in current roadmap):
  - Option 1: Serve experiment results directories as static assets via Flask
  - Option 2: Convert images to base64 and embed in HTML
  - Option 3: Create custom image viewer component that reads from filesystem

### 1.3 Expose Mixture Diagnostics in UI

**Goal**: Surface component specialization information during and after training.

**Background**: The experiment workflow generates:
- `latent_by_component.png`: Latent scatter colored by component assignment
- `responsibility_histogram.png`: Distribution of component ownership
- `tau_matrix_heatmap.png`: Component→label mapping visualization
- π evolution plots

**See Also**:
- [Conceptual Model §How-We-Classify](../../../docs/theory/conceptual_model.md) - Component specialization theory
- [Experiments Guide §Channel-Wise Latent Diagnostic](../../experiments/README.md) - Visualization details
- [Mathematical Specification](../../../docs/theory/mathematical_specification.md) - Formal definition of τ-matrix

**Tasks**:
1. Create new "Component Analysis" tab in training hub
2. Display τ-matrix heatmap (if mixture model)
3. Show latent space colored by component (not just by label)
4. Add toggle: "Color by Label" vs "Color by Component"
5. Display current π values as bar chart

**UI Layout**:
```
Component Analysis Tab
├── Component Assignment (Latent Space)
│   └── Scatter plot colored by argmax(responsibilities)
├── τ-Matrix Heatmap
│   └── Shows which components map to which labels
├── Mixture Weights (π)
│   └── Bar chart of current component probabilities
└── Responsibility Distribution
    └── Histogram of max responsibilities (certainty measure)
```

**Acceptance Criteria**:
- Mixture models show all component visualizations
- Non-mixture models show message "Not a mixture model"
- Visualizations update after each training run
- Can toggle between label-colored and component-colored latent views

**Estimated Duration**: 3-4 days

---

## Phase 2: Active Learning Intelligence

**Goal**: Guide users to label strategically impactful samples.

### 2.1 Implement OOD Scoring

**Background**: The conceptual model defines OOD score as:
```
OOD(x) = 1 - max_c [r_c(x) · max_y τ_{c,y}]
```

This identifies points not well-owned by any labeled component.

**See Also**: [Conceptual Model §How-We-Classify](../../../docs/theory/conceptual_model.md) - OOD scoring derivation and intuition

**Tasks**:
1. Add `compute_ood_scores()` to `ModelState` (uses responsibilities + τ)
2. Store OOD scores in `DataState.ood_scores` array
3. Update `_build_hover_metadata()` to include OOD score
4. Color latent scatter by OOD score (high OOD = red, low = blue)

**Validation**:
- Unlabeled, ambiguous regions should have high OOD scores
- Well-classified, certain regions should have low OOD scores
- Samples from labeled classes within cluster cores: OOD ≈ 0

**Acceptance Criteria**:
- Latent visualization has "Color by OOD Score" mode
- Hover shows OOD score for each point
- High-OOD regions are visually obvious (red highlight)

### 2.2 Strategic Labeling Suggestions

**Goal**: Automatically suggest which samples to label next.

**Strategies**:
1. **High OOD**: Label samples far from labeled component distributions
2. **Component Ambiguity**: Label samples with low `max(responsibilities)`
3. **Label Diversity**: Ensure all classes have minimum representation
4. **Boundary Samples**: Points near decision boundaries (low certainty)

**Tasks**:
1. Implement `suggest_labeling_candidates()` function
2. Takes `DataState`, `τ-matrix`, current label counts
3. Returns ranked list of sample indices with rationale
4. Add "Suggested Labels" panel to UI
5. Clicking suggestion centers view on that point

**UI Design**:
```
Labeling Suggestions Panel
├── Strategy: [High OOD ▼]
├── Top Candidates:
│   ├── Sample 1247 - OOD: 0.87, Component: uncertain
│   ├── Sample 0892 - OOD: 0.79, Near boundary
│   └── Sample 2103 - OOD: 0.74, Class 7 underrepresented
└── [Apply Label] [View Sample] [Skip]
```

**Acceptance Criteria**:
- Suggestions change based on selected strategy
- Clicking suggestion highlights point in latent space
- Can label directly from suggestion panel
- Suggestions update after labeling/training

### 2.3 Track π Evolution Across Sessions

**Goal**: Show how mixture weights evolve with iterative labeling/training.

**Background**: The experiment workflow saves `pi_history.npy` showing π evolution during training. We need session-level tracking across multiple training runs.

**Tasks**:
1. Extend `RunRecord` to include `pi_final` values
2. Create `MixtureEvolutionHistory` in `ModelState`
3. After each run, append (timestamp, π_values) to history
4. Add "π Evolution" plot showing multi-run trajectory
5. Highlight when components become specialized (low entropy)

**Visualization**:
```
π Evolution (Multi-Run View)
─────────────────────────────────
 1.0 ┤     Component 0 ──────
     │     Component 1 ─ ─ ─
 π   │     Component 2 ······
     │
 0.0 └──────────────────────────
     Run 1   Run 2   Run 3   Run 4
     (0 labels) (10 labels) (25 labels)
```

**Acceptance Criteria**:
- Can see π trajectory across multiple training runs
- Runs annotated with label count at that point
- Clear visual when a component "locks in" to a label

**Estimated Duration**: 4-5 days

---

## Phase 3: Advanced Diagnostics

**Goal**: Replicate full experiment diagnostics in dashboard.

### 3.1 Real-Time τ-Matrix Updates

**Current**: τ-matrix computed once per training run
**Target**: Live updates during training via `TrainerLoopHooks`

**Tasks**:
1. Modify `DashboardMetricsCallback` to capture τ updates per epoch
2. Send τ-matrix snapshots via metrics queue
3. Update UI to show "current" vs "starting" τ-matrix
4. Animate transitions (optional, nice-to-have)

**Acceptance Criteria**:
- τ-matrix updates in real-time during training
- Can see component→label assignments converging
- Final τ-matrix matches what's saved in diagnostics

### 3.2 Per-Component Latent Views

**Background**: Experiments generate `channel_latents/` with per-component views where opacity = responsibility.

**Tasks**:
1. Add component selector dropdown
2. For selected component, dim points with low responsibility
3. Opacity = `responsibilities[:, selected_component]`
4. Highlight high-responsibility regions

**UI**:
```
Component View: [Component 3 ▼]
[Latent space where Component 3 owns points appears bright,
 other points are transparent/dim]
```

**Acceptance Criteria**:
- Can view each component's "territory" clearly
- High-responsibility regions stand out
- Works for all mixture model types

### 3.3 Uncertainty Decomposition

**Goal**: Separate aleatoric (data noise) from epistemic (model uncertainty).

**Background**: Heteroscedastic decoder provides σ²(x) for aleatoric uncertainty.

**Tasks**:
1. Store σ² values in `DataState.aleatoric_uncertainty`
2. Compute epistemic uncertainty from responsibility entropy: `-Σ r_c log r_c`
3. Add "Uncertainty Analysis" visualization
4. Color latent space by uncertainty type

**Visualization**:
```
Uncertainty Decomposition
├── Aleatoric (σ²): Inherent data noise
│   └── [Latent scatter colored by σ²]
├── Epistemic (H(r)): Model uncertainty about component
│   └── [Latent scatter colored by entropy]
└── Combined Risk
    └── [2D heatmap: epistemic × aleatoric]
```

**Acceptance Criteria**:
- Can distinguish noisy data from uncertain model regions
- High epistemic + low aleatoric = needs more labels
- High aleatoric = inherently ambiguous data

**Estimated Duration**: 5-6 days

---

## Phase 4: Production Readiness

**Goal**: Make dashboard robust for extended research sessions.

### 4.1 Model Versioning & Checkpointing

**Tasks**:
1. Track model version across training runs
2. Enable rollback to previous checkpoint
3. Add "Compare Runs" view (side-by-side τ-matrices)
4. Export model with full diagnostics

### 4.2 Session Persistence

**Tasks**:
1. Save dashboard state to disk periodically
2. Restore labeling progress after reload
3. Cache latent embeddings for fast page loads
4. Add "Resume Session" on dashboard start

### 4.3 Batch Labeling Operations

**Tasks**:
1. Multi-select mode for labeling
2. "Label all in region" via lasso selection
3. Undo/redo for labeling actions
4. Import/export labels as CSV

### 4.4 Performance Optimization

**Tasks**:
1. Lazy-load large visualizations
2. Reduce data sent to frontend (downsample scatter plots)
3. Server-side rendering for complex plots
4. Add loading spinners with progress

**Estimated Duration**: 6-8 days

---

## Architecture Decisions & Patterns

### Key Principles

1. **Reuse Experiment Infrastructure**: Don't duplicate visualization/metrics code
   - ✅ Use `generate_dashboard_run()` for outputs
   - ✅ Leverage existing metrics registry
   - ✅ Call experiment plotting functions directly

2. **Service Layer for Domain Logic**: Keep commands thin
   - ✅ Services handle model operations
   - ✅ Commands orchestrate services
   - ✅ No business logic in callbacks

3. **Immutable State with Explicit Updates**: Never mutate `AppState` directly
   - ✅ Always use `state.with_active_model(updated_model)`
   - ✅ Call `state_manager.update_state()` to persist
   - ❌ Never assign `state_manager.state = ...`

4. **Mixture-Aware by Default**: All code should handle both mixture and non-mixture models
   - ✅ Check `if responsibilities is not None:` before using
   - ✅ Gracefully degrade for non-mixture models
   - ✅ Use `_predict_outputs()` helper for consistent handling

### Critical Code Patterns

#### Pattern 1: Updating Model State

```python
# ✅ CORRECT
with state_manager.state_lock:
    if state_manager.state.active_model:
        updated_model = state_manager.state.active_model.with_training(
            state=TrainingState.IDLE,
            # ... other updates
        )
        state_manager.update_state(
            state_manager.state.with_active_model(updated_model)
        )

# ❌ WRONG - direct assignment doesn't persist
state_manager.state = new_state
```

#### Pattern 2: Mixture Model Prediction

```python
# ✅ CORRECT - handles both mixture and non-mixture
def _predict_outputs(model, data):
    try:
        mixture_mode = bool(model.config.is_mixture_based_prior())
    except AttributeError:
        mixture_mode = False

    if mixture_mode:
        latent, recon, preds, cert, resp, pi = model.predict_batched(
            data, return_mixture=True
        )
        return latent, recon, preds, cert, resp, pi

    latent, recon, preds, cert = model.predict_batched(data)
    return latent, recon, preds, cert, None, None

# ❌ WRONG - crashes on non-mixture models
latent, recon, preds, cert, resp, pi = model.predict_batched(data, return_mixture=True)
```

#### Pattern 3: Null-Safe Visualization

```python
# ✅ CORRECT
if responsibilities is not None:
    # Show component-colored view
    colors = responsibilities.argmax(axis=1)
else:
    # Fall back to label-colored view
    colors = labels

# ❌ WRONG - crashes when responsibilities is None
colors = responsibilities.argmax(axis=1)
```

### File Organization

```
use_cases/dashboard/
├── core/
│   ├── state_manager.py       # AppStateManager (all state operations)
│   ├── state_models.py        # Immutable state dataclasses
│   ├── commands.py            # Command pattern (orchestration)
│   ├── model_manager.py       # File I/O for models
│   └── run_generation.py      # Experiment output generation
├── services/
│   ├── training_service.py    # Training execution
│   ├── model_service.py       # Model CRUD
│   ├── labeling_service.py    # Label persistence
│   └── container.py           # Dependency injection
├── callbacks/
│   ├── training_callbacks.py       # Main dashboard training
│   ├── training_hub_callbacks.py   # Training hub page
│   ├── labeling_callbacks.py       # Labeling interactions
│   └── visualization_callbacks.py  # Plot updates
├── pages/
│   ├── home.py               # Model selection
│   ├── training_hub.py       # Training workbench
│   └── layouts.py            # Main dashboard layout
└── utils/
    ├── visualization.py       # Plotting utilities
    └── training_callback.py   # JAX→Dashboard bridge
```

### Data Flow

```
User Action (Click/Train)
    ↓
Dash Callback
    ↓
Create Command (with params)
    ↓
dispatcher.execute(command)
    ↓
Command.validate(state, services) → Error or None
    ↓
Command.execute(state, services) → (new_state, message)
    ↓
state_manager.update_state(new_state)
    ↓
UI Updates (via polling or reactive callbacks)
```

### Testing Strategy

1. **Unit Tests**: Services and commands in isolation
2. **Integration Tests**: Full command execution with mock state
3. **UI Tests**: Critical user flows (label → train → view results)
4. **Mixture Model Tests**: Verify both mixture and non-mixture paths

---

## Migration Guide for Next Session

### Quick Start Checklist

When resuming work:

1. **Read this document** to understand vision and current status
2. **Check recent commits** for any changes since this was written
3. **Run the dashboard** and test basic flow:
   - Load model
   - Label a few samples
   - Start training
   - Stop training
   - Start again (should work without errors)
4. **Identify current phase** based on what's working/broken
5. **Pick next task** from the appropriate phase

### Common Pitfalls to Avoid

1. **Don't bypass state_manager**: Always use `update_state()`, never direct assignment
2. **Don't assume mixture model**: Check `responsibilities is not None` before using
3. **Don't duplicate experiment code**: Import and call existing functions
4. **Don't modify state during read**: Use `with state_lock:` for snapshots only
5. **Don't skip validation**: Commands should validate before executing

### Key Files to Reference

**Core Documentation**:
- **[AGENTS.md](../../../AGENTS.md)** - Documentation navigation guide
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - Dashboard architecture, backend touchpoints
- **[Architecture](../../../docs/development/architecture.md)** - Design patterns, service layer
- **[Conceptual Model](../../../docs/theory/conceptual_model.md)** - Mixture models, τ-classifier, OOD theory

**Implementation Details**:
- **[Experiments Guide](../../experiments/README.md)** - CLI workflow, REPORT.md structure
- **[State Models](core/state_models.py)** - Immutable dataclasses (AppState, ModelState, etc.)
- **[Config Metadata](core/config_metadata.py)** - Training parameter definitions
- **[Collaboration Notes](docs/collaboration_notes.md)** - Recent work, debugging playbook

### Debugging Tips

1. **Training stuck**: Check `state.active_model.training.state` - should be IDLE when not training
2. **Missing mixture data**: Verify `_predict_outputs()` is called, not direct `model.predict()`
3. **UI not updating**: Check metrics queue is being drained in polling callback
4. **State inconsistencies**: Add logging to `AppStateManager.update_state()`

---

## Success Metrics

### Phase 0 (Stabilization)
- ✅ Can complete 5 train→stop→train cycles without errors
- ✅ All mixture model tests pass
- ✅ No silent failures in logs

### Phase 1 (Experiment Integration)
- ✅ REPORT.md generated for every training run
- ✅ Can view past run reports from UI
- ✅ Mixture visualizations appear in experiment browser

### Phase 2 (Active Learning)
- ✅ OOD scores correlate with uncertainty
- ✅ Suggested labels guide user to high-impact samples
- ✅ π evolution shows component specialization over time

### Phase 3 (Advanced Diagnostics)
- ✅ Real-time τ-matrix updates during training
- ✅ Per-component latent views isolate specialization
- ✅ Can decompose uncertainty into aleatoric/epistemic

### Phase 4 (Production)
- ✅ Session persists across browser refreshes
- ✅ Can handle 10+ training runs without performance degradation
- ✅ Batch labeling speeds up curriculum building

---

## Appendix: Technical Debt Tracking

### Known Issues (To Fix During Stabilization)

1. **FAST_DASHBOARD_MODE bypasses real predictions**: Should be deprecated once performance is optimized
2. **Labels stored in CSV and in-memory**: Single source of truth should be LabelingService
3. **Global `dashboard_state` module import**: Should pass `state_manager` explicitly where possible
4. **Polling callbacks**: Could be replaced with WebSocket for lower latency
5. **Mixture data not always persisted**: `responsibilities` and `pi_values` not saved to disk

### Future Enhancements (Beyond This Roadmap)

1. **Multi-user support**: Track per-user labeling sessions
2. **Distributed training**: Offload training to backend workers
3. **Curriculum scheduling**: Auto-adjust hyperparameters based on label distribution
4. **Export to paper**: Generate LaTeX-ready figures from dashboard
5. **Model comparison**: Side-by-side view of multiple model variants

---

**Last Updated**: November 17, 2025
**Status**: Phase 0 ✅ Complete | Phase 1.1 & 1.2 ✅ Complete | Phase 1.3 🚧 In Progress
**Next Milestone**: Complete Phase 1.3 (Mixture Diagnostics UI)
