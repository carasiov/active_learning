# AGENTS.md - Working with This Codebase

> **Companion to README.md**: README provides the narrative overview and entry points. This document explains how to traverse the documentation network effectively, what to trust, and implicit knowledge not obvious from linear reading.

---

## 1. Documentation Network Structure

This repo's documentation forms a **layered knowledge graph**:

```
Theory Layer (docs/theory/)
  └─ Stable foundations: design vision, mathematical formulation
     Changes rarely, anchors all decisions

Implementation Layer (docs/development/)
  └─ Current patterns: architecture, module organization, tutorials
     Evolves with code

Usage Layer (experiments/)
  └─ Workflow: configuration, execution, interpretation
     Changes frequently with practices

Code Layer (src/, tests/)
  └─ Ground truth: when docs conflict with code, code wins
```

### Navigation Principles

**Start broad, then narrow**: Theory → Implementation → Usage → Code

**Trust hierarchy**: Theory > Implementation docs > Code > Experiments

**Cross-reference conventions**:
- Theory docs use §Section-Names (stable anchors)
- Implementation docs use `file/paths.py` (may move)
- Links are bidirectional for complete understanding

---

## 2. Trust and Precedence

### Stability Hierarchy

| Source | Trust Level | Use For |
|--------|-------------|---------|
| [`docs/theory/conceptual_model.md`](docs/theory/conceptual_model.md) | Invariants | Understanding "why" |
| [`docs/theory/mathematical_specification.md`](docs/theory/mathematical_specification.md) | Constraints | Formal requirements |
| [`docs/development/architecture.md`](docs/development/architecture.md) | Patterns | Design decisions |
| [`docs/theory/implementation_roadmap.md`](docs/theory/implementation_roadmap.md) §Status ✅ | Current state | What's validated |
| [`docs/theory/implementation_roadmap.md`](docs/theory/implementation_roadmap.md) §Recent | Provisional | What's under validation |
| Code | Authority | When docs unclear |

### Resolving Conflicts

- **Theory says X, code does Y**: Code may be incorrect or transitional
- **Roadmap says "under validation"**: Feature complete but not battle-tested
- **Config exists but roadmap silent**: May be exploratory, not established

---

## 3. Architectural Patterns

### Key Differences from Typical ML Repos

**Protocol-based extension** (not inheritance):
- Location: [`docs/development/architecture.md`](docs/development/architecture.md) §Core-Abstractions §PriorMode-Protocol
- Pattern: Implement interface, register in PRIOR_REGISTRY
- Example: `priors/mixture.py`

**Functional JAX style** (explicit state):
- State in SSVAETrainState, RNG threaded explicitly
- Gotcha: No Python control flow inside JIT boundaries
- Reference: [`docs/development/architecture.md`](docs/development/architecture.md) §Design-Principles

**Hook-based training extensions**:
- Use case: Update Python-side state during training (outside JIT)
- Pattern: TrainerLoopHooks with `batch_context_fn`, `post_batch_fn`, `eval_context_fn`
- Example: [`docs/theory/implementation_roadmap.md`](docs/theory/implementation_roadmap.md) §τ-Classifier
- Tutorial: [`docs/development/extending.md`](docs/development/extending.md) §Tutorial-3

### Configuration System

Parameters interact subtly. Documentation locations:
- Primary: [`src/ssvae/config.py`](src/ssvae/config.py) `::SSVAEConfig` (inline docstrings)
- Patterns: [`experiments/README.md`](experiments/README.md) §Configuration
- Examples: `experiments/configs/*.yaml`

### Concept Cross-Reference Pattern

Example using τ-classifier (applies to most concepts):

```
Theory       → docs/theory/conceptual_model.md §How-We-Classify
Math         → docs/theory/mathematical_specification.md §5
Status       → docs/theory/implementation_roadmap.md §τ-Classifier-Completed
Architecture → docs/development/implementation.md §tau_classifier.py
Tutorial     → docs/development/extending.md §Tutorial-3
Code         → src/ssvae/components/tau_classifier.py
```

---

## 4. Quick Reference: Finding Things

| Task | Location |
|------|----------|
| **Current status** | [`docs/theory/implementation_roadmap.md`](docs/theory/implementation_roadmap.md) §Status-at-a-Glance |
| **Commands** | [`experiments/README.md`](experiments/README.md) §Quick-Start, §Common-Workflows |
| **Math symbol → code** | [`docs/development/implementation.md`](docs/development/implementation.md) §Module-Organization |
| **Config parameters** | [`src/ssvae/config.py`](src/ssvae/config.py) `::SSVAEConfig` docstrings |
| **Module purpose** | [`docs/development/implementation.md`](docs/development/implementation.md) §Module-Organization |
| **Design patterns** | [`docs/development/architecture.md`](docs/development/architecture.md) (design)<br>[`docs/development/extending.md`](docs/development/extending.md) (tutorials) |
| **Known issues** | [`docs/theory/implementation_roadmap.md`](docs/theory/implementation_roadmap.md) §Key-Findings |

### When Stuck

1. Known issues/solutions: [`docs/theory/implementation_roadmap.md`](docs/theory/implementation_roadmap.md) §Key-Findings
2. Similar examples: [`docs/development/extending.md`](docs/development/extending.md) tutorials
3. Trace cross-references: Follow theory → implementation → code chain

---

## 6. Intentional Boundaries

This documentation network does **not** provide:

- Step-by-step instructions for every task (patterns exist, specifics are context-dependent)
- Frequently-changing implementation details (code is the source of truth)
- Experiment results (generated at runtime in `experiments/runs/`)
- Unstarted features (roadmap shows intent, details come with implementation)

---

## Summary

**Navigate by**: Theory → Implementation → Usage → Code

**Trust by**: Stability hierarchy (theory > architecture > current implementation)

**Resolve conflicts**: Code wins when docs unclear; theory guides when code seems wrong

**Start here**: [`docs/theory/conceptual_model.md`](docs/theory/conceptual_model.md) for "why" before "how"