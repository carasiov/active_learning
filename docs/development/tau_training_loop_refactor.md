# τ-Classifier Training Loop Refactor

**Goal:** Remove the bespoke `SSVAE._fit_with_tau_classifier()` loop and let the canonical `Trainer` orchestrate every training regime (τ-enabled or not) without duplicating logic.

## Approach

We introduced a lightweight extension surface for the trainer via `TrainerLoopHooks`. The hooks expose three touch points:

1. `batch_context_fn` — executes just before each call to the JIT-ed `train_step` and can return keyword arguments (e.g., the current τ matrix) that should be forwarded to the training step.
2. `post_batch_fn` — runs after the JIT step completes, while Python still holds the batch tensors. This is where we can safely update mutable Python-side state such as the τ-classifier counts.
3. `eval_context_fn` — evaluated once per epoch to provide keyword arguments for validation metrics (so evaluation sees the same τ that was used during training).

Because these hooks are optional, the standard prior path runs the identical loop it always has; if no hooks are provided the trainer charges ahead with zero overhead.

## SSVAE Integration

`SSVAE` now builds τ-specific hooks via `_build_tau_loop_hooks()`. The batch context hook hands the latest `TauClassifier.get_tau()` array to `train_step`, mirroring the previous custom loop. The post-batch hook performs a deterministic forward pass (outside JIT) to grab responsibilities from the network extras and update `TauClassifier.update_counts()` before the next batch begins. The eval hook simply mirrors the current τ into the validation pass. Because τ-updates still live in `SSVAE`, we keep encapsulation around the classifier object and avoid pushing τ-specific knowledge into the generic trainer.

This design buys us three things:

- **Zero duplication:** Trainer owns all shuffling, early stopping, callbacks, and history logic again.
- **Extensibility:** Future stateful components (e.g., curriculum schedulers) can reuse the same hook surface.
- **Backwards compatibility:** Models without τ never instantiate hooks, so the hot path stays untouched.
