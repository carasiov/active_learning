# Phase 6: Configuration Metadata Augmentation - Implementation Complete

**Date:** 2025-11-12
**Status:** ✅ Complete

## Overview

Phase 6 completes the experiment management system by adding automatic metadata augmentation to configurations and directory naming. Experiments are now fully self-documenting with architecture codes embedded in directory names, configuration files, and summary outputs.

---

## What Was Built

### 1. Enhanced Directory Naming (`src/io/structure.py`)

**Updated `create_run_paths()` to include architecture code:**
- New signature: `create_run_paths(experiment_name, architecture_code) → (run_id, timestamp, paths)`
- Directory format: `{name}__{architecture_code}__{timestamp}` (double underscore separators)
- Backward compatible: Falls back to `{name}_{timestamp}` if architecture_code is None
- Returns run_id (full directory name) for use in metadata

**Key changes:**
```python
def create_run_paths(
    experiment_name: str | None,
    architecture_code: str | None = None,
) -> Tuple[str, str, RunPaths]:
    """Create run directory structure with architecture code.

    Phase 6: Enhanced to include architecture code in directory naming.

    Returns:
        Tuple of (run_id, timestamp, RunPaths)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = sanitize_name(experiment_name)

    # Phase 6: Include architecture code in directory name
    if architecture_code:
        run_id = f"{slug}__{architecture_code}__{timestamp}"
    else:
        # Backward compatibility
        run_id = f"{slug}_{timestamp}"

    run_root = RESULTS_DIR / run_id
    # ... create paths
    return run_id, timestamp, paths
```

**Example directory names:**
- Before: `baseline_experiment_20241112_143027`
- After: `baseline_experiment__mix10-dir_tau_ca-het__20241112_143027`

### 2. Config Metadata Augmentation (`src/pipeline/config.py`)

**New `augment_config_metadata()` function:**
- Adds run_id, architecture_code, timestamp to config dict
- Preserves all existing config sections
- Top-level metadata keys for easy access
- Self-documenting provenance

**Implementation:**
```python
def augment_config_metadata(
    config: Dict[str, Any],
    run_id: str,
    architecture_code: str,
    timestamp: str,
) -> Dict[str, Any]:
    """Augment config with metadata for self-documenting experiments.

    Phase 6: Adds run_id, architecture_code, and timestamp to config.

    Example:
        >>> config = augment_config_metadata(
        ...     config,
        ...     "baseline__mix10-dir__20241112_143027",
        ...     "mix10-dir_tau_ca-het",
        ...     "20241112_143027"
        ... )
        >>> config["run_id"]
        'baseline__mix10-dir__20241112_143027'
    """
    config["run_id"] = run_id
    config["architecture_code"] = architecture_code
    config["timestamp"] = timestamp
    return config
```

**Exported from `src/pipeline/__init__.py`** for CLI access.

### 3. Enhanced CLI (`src/cli/run.py`)

**Updated workflow to generate and use architecture codes:**

1. **Load config** from YAML
2. **Create SSVAEConfig early** (for validation and code generation)
3. **Generate architecture code** using `generate_architecture_code()`
4. **Create run paths** with architecture code
5. **Augment config** with metadata
6. **Save augmented config** to config.yaml
7. **Run training** (SSVAEConfig re-created internally)
8. **Enhance summary** with metadata
9. **Save summary** and report

**Key changes:**
```python
# Phase 6: Generate architecture code for directory naming
_model_config = {**model_config}
if isinstance(_model_config.get("hidden_dims"), list):
    _model_config["hidden_dims"] = tuple(_model_config["hidden_dims"])

ssvae_config = SSVAEConfig(**_model_config)
architecture_code = generate_architecture_code(ssvae_config)

# Phase 6: Create run paths with architecture code
run_id, timestamp, run_paths = create_run_paths(
    exp_meta.get("name"),
    architecture_code,
)
print(f"Architecture: {architecture_code}")
print(f"Output directory: {run_paths.root}")

# Phase 6: Augment config with metadata
experiment_config = augment_config_metadata(
    experiment_config,
    run_id,
    architecture_code,
    timestamp,
)
write_config_copy(experiment_config, run_paths)

# ... run training ...

# Phase 6: Enhance summary with metadata
summary["metadata"] = {
    "run_id": run_id,
    "architecture_code": architecture_code,
    "timestamp": timestamp,
    "experiment_name": exp_meta.get("name", "experiment"),
}
write_summary(summary, run_paths)
```

### 4. Self-Documenting Outputs

**config.yaml now includes:**
```yaml
experiment:
  name: "baseline_experiment"
  description: "..."
  tags: [...]

data:
  dataset: "mnist"
  # ...

model:
  prior_type: "mixture"
  num_components: 10
  # ...

# Phase 6: Auto-generated metadata
run_id: "baseline_experiment__mix10-dir_tau_ca-het__20241112_143027"
architecture_code: "mix10-dir_tau_ca-het"
timestamp: "20241112_143027"
```

**summary.json now includes:**
```json
{
  "training": {
    "final_loss": 100.5,
    ...
  },
  "classification": {
    "accuracy": 0.85,
    ...
  },
  "metadata": {
    "run_id": "baseline_experiment__mix10-dir_tau_ca-het__20241112_143027",
    "architecture_code": "mix10-dir_tau_ca-het",
    "timestamp": "20241112_143027",
    "experiment_name": "baseline_experiment"
  }
}
```

---

## Design Decisions

### 1. Double Underscore Separators

**Why `__` instead of single `_`?**
- Easy parsing: `name, arch_code, timestamp = dir_name.split("__")`
- Experiment names can contain single underscores
- Clear visual separation in directory listings
- Consistent with Python conventions (e.g., `__init__`)

**Example:**
```
my_experiment__mix10-dir_tau_ca-het__20241112_143027
└─┬──────────┘ └───────┬────────────┘ └──────┬──────┘
  name           architecture_code        timestamp
```

### 2. SSVAEConfig Created Twice

**Why create SSVAEConfig in both CLI and pipeline?**
- **CLI creation:** For validation and architecture code generation only
- **Pipeline creation:** For actual training (clean separation of concerns)
- **Cost:** Negligible (config creation is fast)
- **Benefit:** Fail-fast validation + clean architecture

**Alternative considered:** Pass SSVAEConfig to pipeline
- **Rejected:** Would tightly couple CLI to SSVAE (harder to extend)

### 3. Metadata at Top Level

**Why add metadata to top level of config dict?**
- Easy access: `config["run_id"]` instead of `config["metadata"]["run_id"]`
- No nesting: Simpler for tools and scripts
- Backward compatible: Doesn't affect existing config sections

**Alternative considered:** Nested under `config["metadata"]`
- **Rejected:** More verbose, less convenient

### 4. Backward Compatibility

**Why support None architecture_code?**
- Allows gradual migration
- Tests can use simple directory names
- External tools can opt-out if needed

**Behavior:**
- `create_run_paths("test", "mix10")` → `test__mix10__20241112_143027`
- `create_run_paths("test", None)` → `test_20241112_143027`

---

## Files Modified/Created

### Modified
```
use_cases/experiments/src/io/
└── structure.py                  # Enhanced create_run_paths() with arch code

use_cases/experiments/src/pipeline/
├── config.py                     # Added augment_config_metadata()
└── __init__.py                   # Export augment_config_metadata

use_cases/experiments/src/cli/
└── run.py                        # Generate arch code, augment config, enhance summary
```

### Created
```
use_cases/experiments/
├── PHASE6_IMPLEMENTATION.md      # This document
└── test_phase6.py                # Verification tests
```

---

## Benefits Delivered

### For Researchers

**Self-documenting experiments:**
- Directory name shows architecture at a glance
- No need to open config to know what was run
- Easy to organize and compare experiments
- Searchable by architecture code

**Reproducibility:**
- Full provenance in config.yaml
- Summary includes all metadata
- Can reconstruct experiment from any output file

**Organization:**
```
results/
├── baseline__std_noclf_vanilla__20241112_120000/    # Standard prior
├── baseline__mix10-dir_tau_ca-het__20241112_130000/  # Mixture prior
└── baseline__vamp20-km_tau_vanilla__20241112_140000/ # VampPrior
```

### For Development

**Debugging:**
- Know exactly which architecture was run
- Metadata embedded everywhere
- No ambiguity about experiment identity

**Automation:**
- Parse directory names programmatically
- Filter/group experiments by architecture
- Build analysis dashboards from metadata

**Clarity:**
- No more "what config was this?" questions
- Self-contained experiment outputs
- Professional presentation

### For Operations

**Management:**
- Easy to identify old/failed experiments
- Can infer resource usage from architecture code
- Automated cleanup policies based on metadata

**Monitoring:**
- Track which architectures are being used
- Identify popular configurations
- Spot configuration drift

---

## Integration with Previous Phases

### Phase 1: Core Infrastructure
- Uses `generate_architecture_code()` from Phase 1
- Leverages SSVAEConfig validation
- Follows naming conventions designed in Phase 1

### Phase 2: Logging System
- Metadata included in logs
- Directory structure from Phase 2 unchanged
- RunPaths dataclass extended, not replaced

### Phase 3: Metrics Status Objects
- Summary metadata complements metrics
- Same dict-based approach
- Status tracking independent of metadata

### Phase 4: Plotters Status Objects
- Plot status and metadata both in summary
- Orthogonal concerns (status vs identity)
- Compatible data structures

### Phase 5: Storage Organization
- Directory naming doesn't affect subdirectory structure
- Component-based organization preserved
- Relative paths still work

---

## Usage Examples

### Basic Usage

```python
from src.core.naming import generate_architecture_code
from src.pipeline.config import augment_config_metadata, load_experiment_config
from src.io.structure import create_run_paths
from ssvae import SSVAEConfig

# Load config
config = load_experiment_config("configs/my_experiment.yaml")

# Generate architecture code
model_config = {**config["model"]}
if isinstance(model_config.get("hidden_dims"), list):
    model_config["hidden_dims"] = tuple(model_config["hidden_dims"])
ssvae_config = SSVAEConfig(**model_config)
arch_code = generate_architecture_code(ssvae_config)
# → "mix10-dir_tau_ca-het"

# Create run paths
run_id, timestamp, paths = create_run_paths(
    config["experiment"]["name"],
    arch_code,
)
# run_id: "my_experiment__mix10-dir_tau_ca-het__20241112_143027"

# Augment config with metadata
config = augment_config_metadata(config, run_id, arch_code, timestamp)
# config["run_id"]: "my_experiment__mix10-dir_tau_ca-het__20241112_143027"
```

### Parsing Directory Names

```python
from pathlib import Path

def parse_run_directory(dir_path: Path) -> dict:
    """Extract metadata from run directory name."""
    dir_name = dir_path.name

    # Check if it has architecture code (double underscores)
    if "__" in dir_name:
        parts = dir_name.split("__")
        if len(parts) == 3:
            return {
                "experiment_name": parts[0],
                "architecture_code": parts[1],
                "timestamp": parts[2],
            }

    # Legacy format (single underscore)
    parts = dir_name.rsplit("_", 1)
    return {
        "experiment_name": parts[0],
        "architecture_code": None,
        "timestamp": parts[1] if len(parts) == 2 else None,
    }

# Example
info = parse_run_directory(Path("baseline__mix10-dir__20241112_143027"))
# {"experiment_name": "baseline", "architecture_code": "mix10-dir", "timestamp": "20241112_143027"}
```

### Filtering Experiments

```python
from pathlib import Path

def find_experiments_by_architecture(arch_code: str) -> list[Path]:
    """Find all experiment runs using a specific architecture."""
    results_dir = Path("results")
    return [
        d for d in results_dir.iterdir()
        if d.is_dir() and f"__{arch_code}__" in d.name
    ]

# Example
mixture_runs = find_experiments_by_architecture("mix10-dir_tau_ca-het")
# [Path("baseline__mix10-dir_tau_ca-het__20241112_143027"), ...]
```

---

## Testing Status

**Verification tests created:** `test_phase6.py`

Tests cover:
- ✅ Directory naming with architecture code
- ✅ Backward compatibility (None architecture code)
- ✅ Config metadata augmentation
- ✅ Summary metadata enhancement
- ✅ Name sanitization

**All tests passing:**
```
============================================================
Phase 6 Verification Tests
============================================================

1. Testing directory naming with architecture code:
   ✓ Directory naming works correctly

2. Testing backward compatibility (no architecture code):
   ✓ Backward compatibility maintained

3. Testing config metadata augmentation:
   ✓ Config augmentation works correctly

4. Testing summary metadata enhancement:
   ✓ Summary enhancement works correctly

5. Testing name sanitization:
   ✓ Name sanitization works correctly

All Phase 6 verification tests passed! ✓
```

**Unit tests needed:**
- Test create_run_paths with various experiment names
- Test architecture code generation integration
- Test config augmentation edge cases
- Test summary metadata serialization

---

## Comparison: Before vs After

### Before Phase 6

**Directory structure:**
```
results/
├── baseline_experiment_20241112_120000/
├── baseline_experiment_20241112_130000/  # Which config was this?
└── baseline_experiment_20241112_140000/  # Can't tell without opening config
```

**config.yaml:**
```yaml
experiment:
  name: "baseline_experiment"
data:
  dataset: "mnist"
model:
  prior_type: "mixture"
  num_components: 10
  # ...
```

**summary.json:**
```json
{
  "training": {"final_loss": 100.5},
  "classification": {"accuracy": 0.85}
}
```

**Problems:**
- Can't identify experiments without opening files
- No provenance in outputs
- Hard to organize/compare experiments
- Manual record-keeping required

### After Phase 6

**Directory structure:**
```
results/
├── baseline__std_noclf_vanilla__20241112_120000/     # Standard prior
├── baseline__mix10-dir_tau_ca-het__20241112_130000/  # Mixture + τ-classifier
└── baseline__vamp20-km_tau_vanilla__20241112_140000/ # VampPrior
```

**config.yaml:**
```yaml
experiment:
  name: "baseline_experiment"
data:
  dataset: "mnist"
model:
  prior_type: "mixture"
  num_components: 10
  # ...

# Phase 6: Auto-generated metadata
run_id: "baseline__mix10-dir_tau_ca-het__20241112_130000"
architecture_code: "mix10-dir_tau_ca-het"
timestamp: "20241112_130000"
```

**summary.json:**
```json
{
  "training": {"final_loss": 100.5},
  "classification": {"accuracy": 0.85},
  "metadata": {
    "run_id": "baseline__mix10-dir_tau_ca-het__20241112_130000",
    "architecture_code": "mix10-dir_tau_ca-het",
    "timestamp": "20241112_130000",
    "experiment_name": "baseline_experiment"
  }
}
```

**Benefits:**
- ✅ Architecture visible in directory name
- ✅ Complete provenance in all files
- ✅ Easy organization and comparison
- ✅ Self-documenting outputs
- ✅ Programmatic analysis enabled

---

## Compliance with AGENTS.md

### Explicit Communication ✅

- Architecture code printed during run: `Architecture: mix10-dir_tau_ca-het`
- Directory name shows architecture immediately
- All metadata visible in outputs

### Persistent State ✅

- Metadata in config.yaml (persistent)
- Metadata in summary.json (persistent)
- Directory name encodes metadata (filesystem-level persistence)
- Survives terminal disconnection

### Fail-Fast Principle ✅

- SSVAEConfig created early (validation before directory creation)
- Invalid configs caught before creating run directory
- Architecture code generation can fail early with clear errors

### Theory-First Approach ✅

- Design documented before implementation
- Clear rationale for decisions
- Examples and usage patterns provided
- Integration with previous phases explained

---

## Next Steps

### Immediate

**Phase 7: Hyperparameter Sweep Infrastructure (2-3 sessions)**
- Grid search over config parameters
- Parallel experiment execution
- Automatic results aggregation
- Comparative analysis tools

### Short-term

**Unit tests for Phase 6:**
- Test directory parsing with edge cases
- Test metadata augmentation with various configs
- Test backward compatibility thoroughly
- Integration tests with full experiment runs

**Documentation:**
- Update README with Phase 6 features
- Add architecture code legend to REPORT.md
- Create migration guide for existing experiments

### Future

**Enhanced metadata:**
- Git commit hash at experiment time
- Hardware/environment information
- Dependency versions (JAX, FLAX, etc.)
- Execution time estimates

**Tooling:**
- CLI for querying experiments by architecture
- Dashboard for visualizing experiment history
- Automatic experiment comparison reports
- Cleanup tools with metadata filtering

---

## Summary

**Phase 6 delivers:**
- ✅ Architecture codes in directory names
- ✅ Config augmentation with metadata
- ✅ Enhanced summary.json with provenance
- ✅ Self-documenting experiments
- ✅ Backward compatibility maintained

**Key features:**
- Double underscore separators for easy parsing
- Run ID embedded in all outputs
- Architecture code visible at filesystem level
- Complete metadata in config and summary

**Ready for:**
- Phase 7: Hyperparameter sweep infrastructure
- Production use with full metadata tracking
- Programmatic experiment analysis
- Long-term experiment management

**Quality:**
- All verification tests passing
- Backward compatible
- Well-documented with examples
- Clean integration with previous phases

**Impact:**
- Professional experiment management
- Easy experiment identification
- Full reproducibility
- Streamlined research workflow

