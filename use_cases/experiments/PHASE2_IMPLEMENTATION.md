# Phase 2: Logging System - Implementation Complete

**Date:** 2025-11-12
**Status:** ✅ Complete

## Overview

Phase 2 implements a structured logging system with dual output for clean terminal monitoring and detailed file debugging. All implementations follow AGENTS.md principles for explicit communication and persistent storage.

---

## What Was Built

### 1. Structured Logging System (`src/logging/setup.py`)

**Multiple handlers for different purposes:**
- **Console (stdout):** Clean output without timestamps for monitoring
- **experiment.log:** Complete detailed log with DEBUG level
- **training.log:** Filtered log containing only training progress
- **errors.log:** Warnings and errors only for quick diagnosis

**Key features:**
```python
logger = setup_experiment_logging(log_dir)
logger.info("Message")     # → stdout + experiment.log
logger.debug("Detail")      # → experiment.log only
logger.warning("Issue")     # → stdout + experiment.log + errors.log
```

**TrainingLogFilter:**
- Automatically extracts training-related messages
- Keywords: epoch, loss, accuracy, learning rate, batch
- Enables clean separation of training progress from other logs

### 2. Logging Helper Functions

**`log_section_header(logger, title)`**
- Creates visual separators with centered titles
- Uses `═` character for clear section breaks
- Standardizes formatting across experiment logs

**`log_model_initialization(logger, config, architecture_code)`**
- Structured logging of model configuration
- Displays: prior config, classifier type, decoder features, training settings
- Auto-formats based on config (shows relevant details only)

**Example output:**
```
════════════════════════════════════════════════════════════════════════════════
                              Model Initialization
════════════════════════════════════════════════════════════════════════════════
Architecture Code: mix10-dir_tau_ca-het

Prior Configuration:
  Type: mixture
  Components (K): 10
  Dirichlet α: 5.0
  Diversity weight: -0.10 (encourages diversity)

Classifier Configuration:
  Type: τ-classifier (latent-only)
  Smoothing α: 1.0

Decoder Configuration:
  Component-aware: ✓
  Heteroscedastic: ✓
  σ range: [0.05, 0.5]
```

**`log_training_epoch(logger, epoch, max_epochs, losses, metrics)`**
- Compact one-line progress logging
- Formats: `Epoch 5/100 | loss.total=198.30 | loss.recon=185.20 | acc=0.870`
- Automatically routes to training.log via filter

### 3. Enhanced Directory Structure (`src/io/structure.py`)

**RunPaths dataclass extended with subdirectories:**

```python
@dataclass
class RunPaths:
    # Top-level
    root, config, summary, report, artifacts, figures, logs

    # Artifact subdirectories
    artifacts_checkpoints: Path     # Model checkpoints
    artifacts_diagnostics: Path     # Latent dumps, histories
    artifacts_tau: Path             # τ-classifier state
    artifacts_ood: Path             # OOD scoring data
    artifacts_uncertainty: Path     # Heteroscedastic outputs

    # Figure subdirectories
    figures_core: Path              # Loss curves, latent spaces
    figures_mixture: Path           # Mixture evolution plots
    figures_tau: Path               # τ-classifier visualizations
    figures_uncertainty: Path       # Variance maps
    figures_ood: Path               # OOD distributions
```

**All subdirectories created automatically:**
```python
timestamp, paths = create_run_paths("baseline")
paths.ensure()  # Creates all directories

# Use specific subdirectories
checkpoint_path = paths.artifacts_checkpoints / "final.ckpt"
plot_path = paths.figures_mixture / "evolution.png"
```

### 4. Complete Example (`examples/logging_example.py`)

**Demonstrates:**
- Setting up logging with all handlers
- Logging model initialization
- Logging training progress
- Using different log levels
- Structured section headers

**Run it:**
```bash
python use_cases/experiments/examples/logging_example.py
```

### 5. Comprehensive Tests (`tests/test_logging_setup.py`)

**19 tests covering:**
- Log directory and file creation
- Message routing to correct handlers
- Log level filtering
- Training log filter behavior
- Helper function formatting
- Full experiment workflow

**All tests passing:** ✅ 19/19

---

## Directory Structure After Phase 2

```
{experiment_name}_{timestamp}/
├── config.yaml
├── summary.json
├── REPORT.md
├── artifacts/
│   ├── checkpoints/          # Model checkpoints
│   ├── diagnostics/          # Latent dumps, π/usage histories
│   ├── tau/                  # τ-classifier artifacts
│   ├── ood/                  # OOD scoring data
│   └── uncertainty/          # Heteroscedastic outputs
├── figures/
│   ├── core/                 # Loss curves, latent spaces, reconstructions
│   ├── mixture/              # Evolution plots, responsibility histograms
│   ├── tau/                  # Heatmaps, per-class accuracy
│   ├── uncertainty/          # Variance maps
│   └── ood/                  # OOD score distributions
└── logs/
    ├── experiment.log        # Complete detailed log (DEBUG level)
    ├── training.log          # Training progress only (filtered)
    └── errors.log            # Warnings and errors only
```

---

## Design Decisions

### 1. Standard Python `logging` Module

**Why not custom logger?**
- Familiar to Python developers
- Well-tested and robust
- Flexible handler/filter system
- Easy to extend

**Benefits:**
- No new dependencies
- Standard configuration patterns
- Integration with existing tools

### 2. Clean Console + Detailed Files

**Dual output pattern:**
- **Console:** Clean, timestamp-free for monitoring
- **Files:** Detailed with timestamps for debugging

**Rationale:**
- Terminal: Quick status checks, no clutter
- Files: Full history, searchable, persistent

### 3. Filtered Training Log

**Separate training.log:**
- Contains only epoch/loss/accuracy messages
- Easy to tail during training
- Clean time-series for analysis

**Alternative considered:** CSV metrics file
- Deferred to Phase 3 (will add alongside training.log)

### 4. Component-Based Subdirectories

**Organized by component:**
- Easier to find related artifacts
- Natural grouping for cleanup
- Scales with new components

**Example:** When τ-classifier is disabled, `artifacts/tau/` is empty but exists (explicit vs implicit).

---

## Usage Examples

### Basic Setup

```python
from use_cases.experiments.src.logging import setup_experiment_logging

logger = setup_experiment_logging(run_paths.logs)
logger.info("Experiment starting...")
```

### Model Initialization

```python
from use_cases.experiments.src.core.naming import generate_architecture_code
from use_cases.experiments.src.logging.setup import log_model_initialization

arch_code = generate_architecture_code(config)
log_model_initialization(logger, config, arch_code)
```

### Training Loop

```python
from use_cases.experiments.src.logging.setup import (
    log_section_header,
    log_training_epoch,
)

log_section_header(logger, "Training Progress")

for epoch in range(1, config.max_epochs + 1):
    # ... training code ...

    losses = {
        "loss.total": total_loss,
        "loss.recon": recon_loss,
        "loss.kl_z": kl_z_loss,
    }
    metrics = {"acc": accuracy}

    log_training_epoch(logger, epoch, config.max_epochs, losses, metrics)
```

### Error Handling

```python
try:
    train_model()
except Exception as e:
    logger.error(f"Training failed: {e}", exc_info=True)
    raise
```

---

## Integration Points

### With Phase 1 (Core Infrastructure)

**Architecture codes logged:**
```python
arch_code = generate_architecture_code(config)  # Phase 1
log_model_initialization(logger, config, arch_code)  # Phase 2
```

**Canonical metric names in logs:**
```python
from use_cases.experiments.src.metrics.schema import LossKeys

losses = {
    LossKeys.TOTAL: total_loss,
    LossKeys.RECON: recon_loss,
    LossKeys.KL_Z: kl_z_loss,
}
log_training_epoch(logger, epoch, max_epochs, losses)
```

### With Existing Experiment Pipeline

**RunPaths already used:**
```python
# Existing code
timestamp, paths = create_run_paths(config.experiment.name)

# New logging
logger = setup_experiment_logging(paths.logs)
```

**Backward compatible:**
- Existing code doesn't need paths.artifacts_checkpoints
- Can use paths.artifacts (parent dir) as before
- Gradual migration to subdirectories

---

## Testing

### Running Tests

```bash
# All logging tests
JAX_PLATFORMS=cpu poetry run pytest tests/test_logging_setup.py -v

# Specific test class
JAX_PLATFORMS=cpu poetry run pytest tests/test_logging_setup.py::TestLoggingSetup -v

# Run example
python use_cases/experiments/examples/logging_example.py
```

### Test Coverage

**19 tests covering:**

1. **Setup tests (9):**
   - Directory creation
   - Log file creation
   - Logger naming
   - Level configuration
   - Message routing to files

2. **Filter tests (5):**
   - Training keyword detection
   - Non-training message blocking
   - Case insensitivity

3. **Helper tests (4):**
   - Section header formatting
   - Model initialization logging
   - Training epoch logging

4. **Integration test (1):**
   - Full experiment workflow

**All passing:** ✅ 19/19

---

## Files Created

```
use_cases/experiments/src/
├── logging/
│   ├── __init__.py
│   └── setup.py                      # Logging configuration + helpers

use_cases/experiments/src/io/
└── structure.py                       # Updated with subdirectories

use_cases/experiments/examples/
└── logging_example.py                 # Complete usage example

tests/
└── test_logging_setup.py              # 19 unit tests

use_cases/experiments/
└── PHASE2_IMPLEMENTATION.md           # This document
```

---

## Benefits Delivered

### For Developers

**Clean development experience:**
- Monitor training in terminal (no log clutter)
- Debug issues in experiment.log (full detail)
- Quick error diagnosis (errors.log)

**Structured logging:**
- Helper functions enforce consistency
- Section headers improve readability
- Automatic filtering reduces noise

### For Research

**Reproducibility:**
- Every experiment has complete log history
- Configuration logged at initialization
- Training progress preserved

**Debugging:**
- Separate logs for different concerns
- Timestamp precision in files
- Stack traces for errors

### For Operations

**Scalability:**
- Organized subdirectories
- Easy to archive/cleanup
- Component-specific paths

**Monitoring:**
- tail -f logs/training.log for progress
- grep logs/errors.log for issues
- Structured format for parsing

---

## Comparison: Before vs After

### Before Phase 2

```python
# Ad-hoc printing
print(f"Epoch {epoch}: loss={loss}")
print(f"Accuracy: {acc}")

# No persistent logs
# No structured output
# Mixed debug and progress messages
```

**Problems:**
- Output lost when terminal closes
- No separation of concerns
- Hard to grep/filter
- No timestamps
- Cluttered terminal

### After Phase 2

```python
# Structured logging
log_training_epoch(logger, epoch, max_epochs, losses, metrics)

# Clean terminal
# Detailed file logs
# Filtered training log
# Organized subdirectories
```

**Benefits:**
- Persistent logs survive disconnection
- Clean terminal (no timestamps)
- Easy to find specific messages
- Organized by component
- Professional logging

---

## Next Steps

### Immediate

**Phase 3: Status Objects for Metrics (2 sessions)**
- Update metric registry to handle `ComponentResult`
- Migrate metric providers to return status objects
- Update `summary.json` generation
- Add status fields to reports

### Short-term

**CSV metrics export:**
- Add `logs/metrics.csv` for time-series analysis
- Complement training.log with structured data
- Enable plotting with pandas/matplotlib

**Progress bars:**
- Integrate tqdm or Rich progress bars
- Show in terminal without cluttering logs
- Optional (can disable for batch jobs)

### Future

**Integration with experiment runner:**
- Wire logging into `run_experiment.py`
- Use canonical metric keys from Phase 1
- Log model initialization automatically

**Remote logging:**
- Optional handler for remote log aggregation
- Send to Weights & Biases, TensorBoard, etc.
- Maintain local logs as source of truth

---

## Compliance with AGENTS.md

### Explicit Communication ✅

- No silent operations (all important messages logged)
- Clear section headers (visual structure)
- Structured output (readable and parseable)

### Persistent State ✅

- Logs survive terminal disconnection
- Detailed history in files
- Organized subdirectories

### Fail-Fast Principle ✅

- Errors logged immediately with stack traces
- Warnings highlighted (errors.log)
- Easy to diagnose issues

### Theory-First Approach ✅

- Design documented before implementation
- Clear rationale for decisions
- Extensible patterns

---

## Summary

**Phase 2 delivers:**
- ✅ Structured logging with multiple handlers
- ✅ Clean terminal + detailed file logs
- ✅ Filtered training log (progress only)
- ✅ Helper functions for consistent formatting
- ✅ Organized subdirectories for artifacts/figures
- ✅ Complete example demonstrating usage
- ✅ 19 unit tests (all passing)

**Key features:**
- Dual output (monitor in terminal, debug in files)
- Automatic filtering (training vs other messages)
- Component-based organization (subdirectories)
- Professional logging patterns (standard library)

**Ready for:**
- Phase 3: Migrate metrics to use ComponentResult
- Integration with existing experiment pipeline
- Production use with real training runs

**Quality:**
- All tests passing (19/19)
- Well-documented (examples + inline docs)
- Backward compatible (optional features)
- Extensible (easy to add handlers/filters)
