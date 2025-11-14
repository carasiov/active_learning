# AGENTS.md - Working with This Codebase

## Purpose
This repo has a **comprehensive documentation network** designed for deep understanding, not just task execution. This guide explains how that network is structured and how to navigate it effectively during collaborative planning and implementation.

**Key principle**: Documentation forms a **knowledge graph**. Navigate by following conceptual links, not by reading linearly.

---

## 1. Documentation Architecture

### The Web Structure
```
README.md (entry point)
    ↓
    ├─→ Theory Layer (WHY + WHAT)
    │   • Stable foundations - conceptual model, math spec
    │   • Change slowly, anchor the project
    │
    ├─→ Implementation Layer (HOW + WHERE)  
    │   • Current state - architecture, implementation guide
    │   • Evolve with code, track patterns
    │
    └─→ Usage Layer (WORKFLOW)
        • Experiments - config → run → interpret
        • Changes frequently, documents current practices
```

### Navigation Principle
**Start broad, narrow down:**
1. Theory docs → Understand vision and invariants
2. Architecture docs → Understand patterns and structure  
3. Implementation docs → Understand current code organization
4. Usage docs → Understand workflow and practices

**Don't skip theory** - Making changes without understanding design rationale leads to violations of core invariants.

---

## 2. Where to Find What Type of Information

### Stable Knowledge (Read First, Changes Rarely)
- **Design vision & mental model**: `docs/theory/conceptual_model.md`
- **Mathematical foundations**: `docs/theory/mathematical_specification.md`
- **Architecture patterns**: `docs/development/architecture.md`

**These are "north star" documents** - Changes here indicate fundamental redesigns, not incremental work.

### Current State (Check Before Planning)
- **Implementation status**: `docs/theory/implementation_roadmap.md`
  - §Status-at-a-Glance table shows what's done vs planned
  - §Recent completions shows what just landed (under validation)
  - §Key-Findings documents known behaviors and solutions

**This is your "what's happening now" reference** - Always check before proposing changes.

### How-To Knowledge (Reference During Work)
- **Design patterns & abstractions**: `docs/development/architecture.md`
- **Module-by-module reference**: `docs/development/implementation.md`  
- **Extension tutorials**: `docs/development/extending.md`
- **Experiment extension points**: `use_cases/experiments/src/{pipeline,metrics,visualization,io}`
- **Experiment workflow**: `use_cases/experiments/README.md`

**These explain mechanics** - How to extend, where files live, how to run experiments.

### Grounded Details (Source of Truth)
- **Configuration parameters**: `src/model/ssvae/config.py::SSVAEConfig` (inline docstrings)
- **Example experiments**: `use_cases/experiments/configs/*.yaml` (real configurations)
- **Test patterns**: `tests/` (usage examples and edge cases)

**Code is authoritative** - When docs and code conflict, code is correct.

---

## 3. Implicit Knowledge (Not Obvious from Linear Reading)

### 3A. Documentation Conventions

**Linking patterns:**
- Theory docs use **conceptual references**: "See §How-We-Classify"
- Implementation docs use **file paths**: "`src/model/ssvae/priors/mixture.py`"
- Both link bidirectionally: theory ↔ implementation

**Cross-references indicate relationships:**
- "For intuition see X, for math see Y, for code see Z"
- Follow these to build complete understanding

**Section references are stable:**
- Docs use §Section-Names not page numbers
- Safe to bookmark specific sections

### 3B. Stability Indicators

**How to tell if something is established vs experimental:**

| Indicator | Meaning |
|-----------|---------|
| In conceptual_model.md | Core design principle (stable) |
| In math_specification.md | Precise formulation (stable) |
| In architecture.md | Established pattern (stable) |
| In roadmap.md §Status table | Implementation status (check marks = done, empty = planned) |
| In roadmap.md §Recent completions | Just landed, under validation |

**Example**: τ-classifier shows in roadmap §Status with ✅ but also in §Recent-Completions → Implemented but still validating.

### 3C. Unique Codebase Patterns

**This repo differs from typical ML codebases in key ways:**

**Protocol-based extension** (not inheritance):
- Explained: `architecture.md` §Core-Abstractions §PriorMode-Protocol
- Why different: Enables pluggability without modifying core classes
- Example: `priors/mixture.py` implements protocol, registered in `priors/__init__.py`

**Functional JAX style** (explicit state):
- Explained: `architecture.md` §Design-Principles
- Key difference: State is explicit (`SSVAETrainState`), not hidden in objects
- Gotcha: RNG must be threaded explicitly (`rng, subkey = jax.random.split(rng)`)

**Hook-based training extensions**:
- Explained: `extending.md` §Tutorial-3, real example in `roadmap.md` §τ-Classifier
- Pattern: `TrainerLoopHooks` provide touch points outside JIT boundaries
- Use case: τ-classifier updates Python-side state after each batch

**Registry-driven experiment toolkit**:
- Source: `use_cases/experiments/src/infrastructure/metrics/registry.py`, `use_cases/experiments/src/infrastructure/visualization/registry.py`
- Why: Keeps CLI thin and lets agents add metrics/plots by registering new providers
- Gotcha: Outputs are routed through `io/structure.py`, so follow that schema when emitting artifacts

**Understanding these patterns is critical** - They're architectural choices that influence all extensions.

### 3D. Configuration Interdependencies

**Some parameters interact in non-obvious ways:**

Where to learn about them:
- Inline docs: `config.py::SSVAEConfig` has detailed docstrings
- Validation: `__post_init__()` catches some violations
- Context: `use_cases/experiments/README.md` §Configuration discusses patterns

**Historical naming gotchas:**
- `component_diversity_weight` negative = entropy *reward* (misnomer)
- Explained in: `roadmap.md` §Entropy-Reward-Configuration

**When in doubt about parameter interactions:**
1. Check `config.py` docstrings
2. Look at example configs in `use_cases/experiments/configs/`
3. Check roadmap.md for known interaction patterns

---

## 4. Navigating During Different Work Modes

### Planning / Brainstorming Phase

**Goal: Understand design space and constraints**

1. **Check current state**: `roadmap.md` §Status-at-a-Glance
   - What's done? What's validated? What's planned?

2. **Understand design rationale**: `conceptual_model.md`
   - Why is it designed this way?
   - What are the invariants?

3. **Check mathematical constraints**: `mathematical_specification.md`
   - What are the formal requirements?
   - What are the trade-offs?

4. **Look for related work**: `extending.md`
   - Has someone done something similar?
   - What patterns exist?

### Implementation Phase

**Goal: Understand current code structure and patterns**

1. **Understand architecture**: `architecture.md`
   - What patterns should I follow?
   - Where do different concerns live?

2. **Locate relevant modules**: `implementation.md`
   - Which files do I need to touch?
   - How are they structured?

3. **Follow extension tutorials**: `extending.md`
   - Step-by-step for similar changes

4. **Reference grounded examples**: Code files + tests
   - How is it actually done?

### Validation Phase

**Goal: Verify changes work as intended**

1. **Check experiment workflow**: `use_cases/experiments/README.md`
   - How to run tests?
   - How to interpret results?

2. **Verify against spec**: `mathematical_specification.md`
   - Does it satisfy formal requirements?

3. **Check for regressions**: Test suite patterns
   - What should I test?
   - What are edge cases?

---

## 5. Key Documentation Cross-References

### Concept → Multiple Perspectives

**τ-classifier** (example of complete cross-referencing):
- Theory: `conceptual_model.md` §How-We-Classify
- Math: `mathematical_specification.md` §5
- Status: `roadmap.md` §τ-Classifier-Completed
- Implementation: `implementation.md` §tau_classifier.py
- Tutorial: `extending.md` §Tutorial-3
- Code: `src/model/ssvae/components/tau_classifier.py`

**Protocol-based priors**:
- Design: `architecture.md` §Core-Abstractions §PriorMode-Protocol
- Math: `mathematical_specification.md` §3.1
- Tutorial: `extending.md` §Tutorial-1 (VampPrior)
- Example: `src/model/ssvae/priors/mixture.py`



---

## 6. What This Documentation Network Doesn't Do

**Intentionally NOT documented:**

**Implementation details that change frequently:**
- Don't duplicate code in docs
- Code is source of truth
- Docs point to relevant files

**Specific experiment results:**
- Results in `use_cases/experiments/results/*/REPORT.md` after running
- Not committed to repo (too large, too variable)

**Future features not started:**
- Roadmap shows what's planned
- No detailed docs until implementation begins

**Step-by-step for every task:**
- Docs provide patterns and examples
- Not exhaustive recipes (you collaborate with human to plan)

---

## 7. Orientation Checklist for New Work

**Before proposing changes:**
- [ ] Read `conceptual_model.md` to understand design vision
- [ ] Check `roadmap.md` §Status to see current state
- [ ] Verify in `architecture.md` that pattern exists or understand why new pattern needed
- [ ] Look in `extending.md` for similar tutorials

**During implementation:**
- [ ] Reference `implementation.md` for module organization
- [ ] Check `config.py` for parameter definitions
- [ ] Look at existing code for patterns
- [ ] Run relevant tests

**After implementation:**
- [ ] Follow `use_cases/experiments/README.md` workflow for validation
- [ ] Compare results to expectations in `mathematical_specification.md`
- [ ] Check for regressions in `tests/`

---

## 8. Quick Reference (Minimal Pointers)

### Entry Points by Intent
- Understand design vision → `conceptual_model.md`
- Understand current state → `roadmap.md` §Status-at-a-Glance
- Understand architecture → `architecture.md`
- Add new feature → `extending.md` + relevant architecture section
- Run experiments → `use_cases/experiments/README.md` §Quick-Start
- Understand module → `implementation.md` §Module-Organization

**Note:** This project uses [Poetry](https://python-poetry.org/) for Python dependency management and running commands. Ensure Poetry is installed and use `poetry run` for all CLI operations.

### Common Commands
```bash
# Quick sanity check
JAX_PLATFORMS=cpu poetry run python use_cases/experiments/run_experiment.py --config use_cases/experiments/configs/quick.yaml

# Full experiment
poetry run python use_cases/experiments/run_experiment.py --config use_cases/experiments/configs/mixture_example.yaml

# Run tests  
pytest tests/
```

### Critical Files for Reference
- Configuration: `src/model/ssvae/config.py::SSVAEConfig`
- Loss computation: `src/model/training/losses.py::compute_loss_and_metrics_v2()`
- Training loop: `src/model/training/trainer.py::Trainer`
- Example configs: `use_cases/experiments/configs/*.yaml`
- Experiment pipeline: `use_cases/experiments/src/pipeline/train.py`
- Metrics registry: `use_cases/experiments/src/infrastructure/metrics/registry.py`
- Visualization registry: `use_cases/experiments/src/infrastructure/visualization/registry.py`

---

## Summary

**This repo prioritizes deep understanding over quick execution.**

The documentation network is designed to support collaborative planning by providing:
- **Stable foundations** (theory) to anchor discussions
- **Current state** (roadmap) to inform decisions
- **Patterns and examples** (architecture, extending) to guide implementation
- **Cross-references** to build complete understanding

Navigate by following conceptual links based on what you need to understand, not by reading linearly.

When in doubt, start with `conceptual_model.md` to understand the "why" before diving into the "how" or "what".
