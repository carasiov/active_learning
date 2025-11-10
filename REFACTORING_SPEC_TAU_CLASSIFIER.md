# τ-Classifier Refactoring Specification

**Date:** November 10, 2025  
**Status:** Active  
**Priority:** High (architectural debt)  
**Assignee:** Engineering Team  

---

## Context

The τ-classifier implementation (completed Nov 10) is **functionally correct and validated** but has architectural debt that needs addressing:

1. **Training loop duplication** - 130 lines duplicated between `SSVAE._fit_with_tau_classifier()` and `Trainer.train()`
2. **Performance bottleneck** - Count updates use Python loops instead of vectorized operations
3. **Missing guardrails** - No validation for minimum data requirements

These issues won't affect correctness but will cause maintenance problems and limit scalability.

---

## Objectives

**Primary Goals:**
- Eliminate code duplication in training loop
- Improve count update performance for scalability to K>50
- Add user-facing validation to prevent common pitfalls

**Non-Goals:**
- Changing the τ-classifier algorithm (it's validated and working)
- Optimizing JIT-compiled functions (already efficient)
- Rewriting tests (keep as regression suite)

**Success Metrics:**
- Zero duplicated training logic
- Count updates <1ms for K=50, batch=128
- All existing tests pass
- Validation experiment results match (±1% accuracy)

---

## Task 1: Vectorize Count Updates ⚡

**Priority:** High  
**Estimated Effort:** 1-2 hours  
**Risk Level:** Low  

### Problem
`TauClassifier.update_counts()` uses nested Python loops (O(K × C × N)):
```python
for c in range(num_components):
    for y in range(num_classes):
        count = jnp.sum(...)
        self.s_cy = self.s_cy.at[c, y].add(count)
```

This creates K×C temporary arrays and doesn't scale beyond K=50-100.

### Requirements

**Must Have:**
- Replace loops with vectorized JAX operations (matmul or equivalent)
- Numerical equivalence: new implementation matches old to machine precision
- Performance improvement: measurably faster for K≥10
- Preserve API: `update_counts(responsibilities, labels, mask)` signature unchanged

**Suggested Approach:**
- Use one-hot encoding + matrix multiplication: `responsibilities.T @ one_hot_labels`
- Single array update instead of K×C incremental updates
- Consider using `jax.numpy.bincount` or `jax.ops.segment_sum` for alternative approaches

**Testing:**
- Add `test_update_counts_vectorized_equivalence()` comparing old vs new
- Benchmark both implementations (log timing in test output)
- Verify all existing `test_tau_classifier.py` tests still pass

**Deliverables:**
- Modified `TauClassifier.update_counts()` in `src/ssvae/components/tau_classifier.py`
- New equivalence test
- Performance benchmark results (comment or test output)

---

## Task 2: Refactor Training Loop Integration 🔧

**Priority:** High  
**Estimated Effort:** 3-5 hours  
**Risk Level:** Medium-High  

### Problem
`SSVAE._fit_with_tau_classifier()` duplicates ~130 lines from `Trainer.train()`:
- Data splitting, shuffling, epoch iteration
- Early stopping logic, callback orchestration
- Metrics aggregation

Any bug fix or feature in `Trainer` won't propagate to τ-classifier training.

### Requirements

**Must Have:**
- Single training loop handles both τ and non-τ cases
- τ count updates happen after each batch (before next train step)
- Updated τ matrix available to next batch's loss computation
- Zero regression in training behavior (loss curves, early stopping, callbacks)
- Backward compatibility: non-τ models train identically

**Design Constraints:**
- τ updates must happen **outside JIT** (Python object state)
- τ matrix must pass **into JIT** (as array parameter)
- Cannot break JAX functional purity in compiled functions

**Suggested Approaches:**

*Option A: Post-batch callback (recommended)*
```python
# Add to Trainer.train()
def train(..., post_batch_callback=None):
    for batch in batches:
        context = getattr(self, '_batch_context', {})
        state, metrics = train_step(state, batch_x, batch_y, **context)
        
        if post_batch_callback:
            self._batch_context = post_batch_callback(state, batch_x, batch_y)
```

*Option B: Trainer subclass*
```python
class TauTrainer(Trainer):
    def _train_one_batch(self, ...):
        result = super()._train_one_batch(...)
        # Add tau update here
        return result
```

*Option C: Strategy pattern*
```python
class TrainingStrategy(Protocol):
    def pre_batch(...) -> dict  # Returns context for train_step
    def post_batch(...) -> None
```

**Choose the approach that feels cleanest** - all can work if implemented correctly.

**Testing:**
- `test_tau_training_matches_original()` - compare loss curves before/after refactor
- `test_non_tau_unchanged()` - verify standard training unaffected
- `test_callback_not_called_for_standard_prior()` - no overhead when unused
- Run full `tau_classifier_validation.yaml` experiment - results should match

**Deliverables:**
- Modified `Trainer` class with extension mechanism
- Simplified `SSVAE.fit()` using extended Trainer
- Remove or deprecate `_fit_with_tau_classifier()`
- Updated integration tests
- Brief design doc (2-3 paragraphs) explaining chosen approach

---

## Task 3: Add Data Requirement Validation ⚠️

**Priority:** Medium  
**Estimated Effort:** 1 hour  
**Risk Level:** Low  

### Problem
Users can train with limited labeled data and may hit component collapse without understanding the tradeoffs. The model should provide guidance about expected behavior with different data regimes.

### Requirements

**Must Have:**
- Informational logging about labeled sample count at training start
- Guidance on expected behavior (e.g., "Training with N labeled samples across K classes - component specialization may be limited")
- Config validation: `num_components >= num_classes` when using τ-classifier
- Optional: Warning if labeled data is extremely sparse (e.g., < 10 samples total)

**Nice to Have:**
- Suggest reducing `num_components` if data insufficient
- Link to documentation explaining component collapse
- Optional strict mode that raises error instead of warning

**Example Logging:**
```
INFO: τ-classifier training with 100 labeled samples across 10 classes.
      Component specialization will develop gradually - more labels will
      improve separation. Current regime: few-shot learning mode.
      
Note: Baseline testing uses ~500 labeled samples. Performance improves
      with more labels, but model is designed for label-efficient learning.
```

**Testing:**
- Test warning triggered with low data
- Test warning suppressed with adequate data
- Test config validation catches invalid configurations

**Deliverables:**
- Validation in `SSVAE.fit()` or `SSVAEConfig.__post_init__()`
- Test coverage for validation logic
- Update experiment configs to document requirements

---

## Task 4: Simplify Prediction Logic 🧹

**Priority:** Low  
**Estimated Effort:** 30-60 minutes  
**Risk Level:** Low  

### Problem
`_predict_deterministic()` and `_predict_with_sampling()` have complex nested conditionals for choosing between τ-classifier and standard classifier.

### Requirements

**Must Have:**
- Extract prediction strategy into helper method
- Reduce duplication between deterministic and sampling paths
- Maintain exact same behavior (no regression in predictions)

**Suggested Refactoring:**
```python
def _get_predictions(self, logits, extras, axis_hint=None):
    """Unified prediction logic - tries tau, falls back to standard."""
    # Implementation details up to you
```

**Testing:**
- Verify predictions unchanged for both τ and non-τ models
- Test both deterministic and sampling modes

**Deliverables:**
- Cleaner `_predict_deterministic()` and `_predict_with_sampling()`
- Helper method(s) as appropriate
- No new tests required (existing tests validate correctness)

---

## Implementation Guidelines

### Ordering
**Recommended sequence:**
1. **Task 1** (vectorization) - Low risk, quick win
2. **Task 3** (validation) - Low risk, high user value
3. **Task 2** (training loop) - Highest risk, take your time
4. **Task 4** (prediction cleanup) - Optional polish

Tasks 1 and 3 are independent and can be done in parallel or any order.

### Development Workflow
```bash
# Create feature branch
git checkout -b refactor/tau-classifier-optimization

# After each task:
pytest tests/test_tau_classifier.py tests/test_tau_integration.py -v
python -m experiments.run experiments/configs/tau_classifier_validation.yaml

# Integration validation (final check)
# Should produce accuracy within ±1% of baseline (34.4% ± 0.3%)
```

### Testing Philosophy
- **Existing tests are your safety net** - they validate correctness
- Add new tests for **new behavior** (validation warnings, performance)
- Use **integration tests as regression tests** - run full experiment
- Don't remove old tests until new tests prove equivalence

### Code Review Checklist
Before submitting PR:
- [ ] All existing tests pass
- [ ] No performance regression (count updates faster or same)
- [ ] Validation experiment results match baseline
- [ ] Code coverage maintained or improved
- [ ] No new TODO/FIXME comments
- [ ] Docstrings updated where behavior changed

---

## Out of Scope

**Don't change these** (unless you find a bug):
- τ-classifier algorithm (`TauClassifier.predict()`, `get_tau()`, etc.)
- Stop-gradient placement in `tau_classification_loss()`
- Loss computation logic in `compute_loss_and_metrics_v2()`
- Soft count accumulation formula
- Test assertions in `test_tau_classifier.py`

These are **validated and correct** - refactoring is about structure, not algorithm.

---

## Questions & Clarifications

**Q: Can I change the Trainer API?**  
A: Yes, but keep backward compatibility. Existing code using `Trainer.train()` without τ should work unchanged.

**Q: What if I find a better approach than suggested?**  
A: Go for it! The suggestions are guidelines, not requirements. Just ensure tests pass and document your reasoning.

**Q: Should I optimize other parts I notice?**  
A: Stick to the spec unless you find actual bugs. Log improvement ideas in separate issues.

**Q: How much faster should count updates be?**  
A: 10-50× is achievable with vectorization. <1ms for K=50, batch=128 is the target. If you hit this, you're good.

**Q: What if the training loop refactor is taking too long?**  
A: Timebox it to 5 hours. If it's getting complex, document the challenge and we'll pair on it.

---

## Success Criteria Summary

**Minimum Viable Refactor:**
- ✅ Count updates vectorized and tested
- ✅ Training loop refactored or clear plan documented
- ✅ Data validation warnings in place
- ✅ All tests passing
- ✅ Validation experiment matches baseline

**Stretch Goals:**
- ✅ Prediction logic simplified
- ✅ Performance benchmarks documented
- ✅ Design doc for training loop approach

---

## References

- **Theory:** `docs/theory/mathematical_specification.md` Section 5
- **Original Implementation:** `src/ssvae/components/tau_classifier.py`
- **Training Loop:** `src/training/trainer.py`, `src/ssvae/models.py`
- **Tests:** `tests/test_tau_classifier.py`, `tests/test_tau_integration.py`
- **Validation Results:** `TAU_CLASSIFIER_VALIDATION_RESULTS.md`

---

**Note to implementer:** You're working with validated, correct code. The goal is to make it maintainable and scalable, not to fix bugs. Take your time with the training loop refactor - it's the most delicate part. Feel free to ask questions or propose alternative approaches.

Good luck! 🚀
