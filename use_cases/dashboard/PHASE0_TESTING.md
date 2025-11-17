# Phase 0 Stabilization - Testing Checklist

This document provides a checklist for verifying Phase 0 stabilization work completed in [ROADMAP.md Phase 0](ROADMAP.md).

## 📚 Before Testing

**Familiarize yourself with**:
- **[Dashboard README](README.md)** - Quick start instructions, routes, development tips
- **[ROADMAP.md Phase 0](ROADMAP.md)** - Stabilization goals and implementation details
- **[Collaboration Notes §Debugging](docs/collaboration_notes.md)** - How to capture logs and debug issues
- **[Developer Guide §4](docs/DEVELOPER_GUIDE.md)** - Debugging toolkit (log levels, file locations)

**Key Concepts Tested**:
- **State Lifecycle**: [ROADMAP.md §0.1](ROADMAP.md) - State transitions, update patterns
- **Error Handling**: [ROADMAP.md §0.3](ROADMAP.md) - Defensive error handling, model loading
- **Mixture Support**: [ROADMAP.md §0.4](ROADMAP.md) - Mixture model predictions, responsibilities

## Test Environment Setup

1. Start the dashboard:
   ```bash
   poetry run python use_cases/dashboard/app.py
   ```

2. Open browser to http://localhost:8050

## 1. Model Loading Robustness ✓

**Objective**: Verify models with architecture mismatches load gracefully

### Test Cases:

- [ ] **TC1.1**: Open model with architecture mismatch (e.g., `test1`)
  - **Expected**: Model loads successfully
  - **Expected**: Warning message appears in training terminal
  - **Expected**: Dashboard UI remains functional (no crashes)
  - **Expected**: Visualization shows placeholder/zero data

- [ ] **TC1.2**: Open valid model (e.g., `adv_model`)
  - **Expected**: Model loads successfully
  - **Expected**: No warning messages
  - **Expected**: Predictions and visualizations appear correctly

## 2. Training Lifecycle: Start → Stop → Start ✓

**Objective**: Verify training can be stopped and restarted without errors

### Test Procedure:

1. Open any model
2. Navigate to Training Hub (`/model/{id}/training-hub`)
3. Configure training:
   - Set epochs to 10
   - Verify hyperparameters are set (LR, recon weight, KL weight)
4. Click "Start Training"
5. Confirm in modal
6. Wait for training to begin (status = RUNNING)
7. Click "Stop Training" after 2-3 epochs
8. Wait for training to stop (status returns to IDLE)
9. Click "Start Training" again
10. Confirm in modal

### Expected Results:

- [ ] Training starts successfully (first time)
- [ ] Status changes: IDLE → QUEUED → RUNNING
- [ ] Terminal shows epoch progress (e.g., "Epoch 1/10 | train 0.xxxx")
- [ ] Stop button becomes enabled during training
- [ ] Training stops when requested
- [ ] Status returns to IDLE
- [ ] No error messages in terminal
- [ ] Second training run starts successfully
- [ ] No "Training already in progress" error
- [ ] No "Thread still active" error

## 3. Training Lifecycle: Start → Complete → Start ✓

**Objective**: Verify training can be restarted after natural completion

### Test Procedure:

1. Open any model
2. Navigate to Training Hub
3. Set epochs to **2** (short run)
4. Click "Start Training" and confirm
5. Wait for training to complete (all epochs)
6. Observe status change to COMPLETE
7. Wait a few seconds
8. Click "Start Training" again
9. Set epochs to **2** again
10. Confirm in modal

### Expected Results:

- [ ] First training run completes successfully
- [ ] Status changes through: IDLE → QUEUED → RUNNING → COMPLETE → IDLE
- [ ] Terminal shows "Training complete" message
- [ ] Loss curves update with epoch data
- [ ] Status automatically returns to IDLE after completion
- [ ] Second training run starts without errors
- [ ] No state inconsistencies

## 4. Navigation Paths ✓

**Objective**: Verify smooth navigation without state confusion

### Test Procedure:

1. Start at home (`/`)
2. Click "Open" on any model
3. Should navigate to `/model/{id}` (main dashboard)
4. Click "Training Hub" link
5. Should navigate to `/model/{id}/training-hub`
6. Click "Model Architecture & Advanced Settings"
7. Should navigate to `/model/{id}/configure-training`
8. Click "← Back to Latent Viewer" (from training hub)
9. Should return to `/model/{id}`
10. Navigate to home via logo click
11. Open a different model

### Expected Results:

- [ ] All navigation links work
- [ ] No "No model loaded" errors
- [ ] Form values don't reset unexpectedly
- [ ] Selected model persists across navigation
- [ ] Status messages persist in training terminal
- [ ] Loss curves persist
- [ ] Latent scatter plot remains consistent

## 5. UI Consistency ✓

**Objective**: Verify form inputs behave predictably

### Test Cases:

- [ ] **TC5.1**: Epochs input value persists
  - Set epochs to 50 in Training Hub
  - Navigate away and back to Training Hub
  - **Expected**: Value remains 50 (or shows config default)
  - **Note**: May reset to config value on page reload (expected)

- [ ] **TC5.2**: Color mode selection persists
  - Select "Predictions" color mode
  - Navigate to Training Hub and back
  - **Expected**: Color mode persists

- [ ] **TC5.3**: Sample selection persists
  - Click on a point in latent scatter
  - Navigate to Training Hub and back
  - **Expected**: Same sample remains selected

## 6. Error Handling ✓

**Objective**: Verify defensive error handling prevents crashes

### Test Cases:

- [ ] **TC6.1**: Training with 0 labeled samples
  - Create model with 0 labeled samples
  - Start training for 2 epochs
  - **Expected**: Training completes quickly
  - **Expected**: Warning about "0 labeled samples" (if implemented)
  - **Expected**: No crashes

- [ ] **TC6.2**: Rapid start/stop cycles
  - Start training
  - Immediately stop training (within 1 second)
  - Repeat 3 times
  - **Expected**: No crashes or state corruption
  - **Expected**: Status always returns to IDLE

- [ ] **TC6.3**: Training with mixture model
  - Open mixture-based model (if available)
  - Train for 2 epochs
  - **Expected**: Mixture visualizations appear (if implemented)
  - **Expected**: No "responsibilities unavailable" errors
  - **Expected**: Training completes successfully

## 7. State Transition Logging ✓

**Objective**: Verify logging provides useful debugging information

### Test Procedure:

1. Check terminal output during training lifecycle tests
2. Look for log messages in format:
   ```
   [HH:MM:SS] [INFO] Training state transition | model={id} | IDLE → QUEUED
   [HH:MM:SS] [INFO] Training state transition | model={id} | QUEUED → RUNNING
   [HH:MM:SS] [INFO] Training state transition | model={id} | RUNNING → IDLE
   ```

### Expected Results:

- [ ] State transitions logged to terminal
- [ ] Log messages include model ID
- [ ] Log messages include old and new state
- [ ] Logs appear at correct times (when state changes)

## Troubleshooting

If tests fail, consult these resources:

**Debugging Guide**:
- **[Collaboration Notes §Debugging Playbook](docs/collaboration_notes.md)** - Step-by-step debugging workflow
- **[Developer Guide §4 Debugging Toolkit](docs/DEVELOPER_GUIDE.md)** - Log locations, test commands
- **[README §Development](README.md)** - Enabling DEBUG logs, runtime log location

**Common Issues**:
1. **Training stuck in RUNNING**: See [ROADMAP.md §Debugging Tips](ROADMAP.md) - Check state and thread status
2. **Architecture mismatch errors**: See [ROADMAP.md §0.3](ROADMAP.md) - Should load gracefully with warnings
3. **Missing mixture visualizations**: See [ROADMAP.md §0.4](ROADMAP.md) - Check `_predict_outputs()` implementation
4. **State not resetting**: See [ROADMAP.md §Architecture Patterns](ROADMAP.md) - Must use `update_state()`

**Log Collection**:
```bash
# Enable debug logs
export DASHBOARD_LOG_LEVEL=DEBUG
poetry run python use_cases/dashboard/app.py

# Collect runtime logs
tail -n 200 /tmp/ssvae_dashboard.log
```

**Report Issues**:
When reporting test failures, include:
- Test case number (e.g., TC1.1, TC2.3)
- Terminal output (console logs)
- Runtime logs (`/tmp/ssvae_dashboard.log`)
- Screenshots showing the issue
- Browser console errors (F12 → Console)

## Summary

**Phase 0 is complete when**:
- All test cases pass ✓
- No crashes during normal usage ✓
- Training lifecycle works reliably ✓
- Navigation is smooth ✓
- Error messages are user-friendly ✓

**Known Limitations** (acceptable for Phase 0):
- Epochs input may reset to config default on navigation (not a bug)
- Models with architecture mismatch show zero predictions (by design)
- Fast training completion with 0 labeled samples (expected behavior)

---

**Testing Date**: _________________

**Tested By**: _________________

**Notes**:
