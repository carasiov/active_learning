# Experiment Management System Design

> **Purpose:** Design document for research-grade experiment management. Addresses robustness, reproducibility, and observability for the SSVAE experiment pipeline.
>
> **Status:** Design proposal (2025-11-12)
>
> **Related:** [Experiment Guide](../../use_cases/experiments/README.md) · [Architecture](architecture.md)

---

## Design Principles

1. **Explicit over implicit**: Missing outputs must be explained, not silent
2. **Lightweight over heavyweight**: Simple files and conventions, not complex frameworks
3. **Durable by default**: All runs preserve full provenance without manual effort
4. **Comparable without tools**: Humans can grep/diff configs and compare metrics without special tooling

---

## Dimension A: Naming & Provenance Model

### Current State
```
results/{slug}_{timestamp}/
├── config.yaml          # Full config snapshot
├── summary.json         # Metrics
└── REPORT.md            # Human report
```

### Design Decision: Architecture-Encoded Naming + Metadata

**Directory naming structure:**
```
{experiment-name}__{architecture-code}__{timestamp}/
```

Where:
- `experiment-name`: User-chosen (e.g., "baseline", "kl-ablation", "quick")
- `architecture-code`: Auto-generated from config: `{prior}_{classifier}_{decoder}`
- `timestamp`: `YYYYMMDD_HHMMSS` for uniqueness

**Architecture code format:**
```
prior:
  std              standard Gaussian
  mix{K}           mixture K components (e.g., mix10)
  mix{K}-dir       mixture + Dirichlet prior
  vamp{K}-km       VampPrior K components, k-means init
  vamp{K}-rand     VampPrior, random init
  geo{K}-circle    geometric circle arrangement
  geo{K}-grid      geometric grid arrangement

classifier:
  tau              τ-classifier (latent-only, responsibility-based)
  head             standard classifier head on z

decoder:
  plain            plain decoder
  ca               component-aware
  het              heteroscedastic
  ca-het           component-aware + heteroscedastic

decoder modifiers (append):
  -contr           contrastive learning enabled
```

**Validation rules:**
1. τ-classifier requires mixture-based prior (mix/vamp/geo, not std)
2. Component-aware decoder requires mixture-based prior
3. VampPrior requires init method (-km or -rand)
4. Geometric requires arrangement (-circle or -grid)

**Examples:**
```
baseline__mix10-dir_tau_ca-het__20251112_143022/
quick__std_head_plain__20251112_145030/
vamprior-test__vamp20-km_tau_ca-het__20251112_144022/
contrastive__mix10-dir_tau_ca-het-contr__20251113_091500/
```

**Structured metadata in `config.yaml` header:**

```yaml
experiment:
  name: "baseline"
  description: "Testing reduced KL weight impact on component usage"
  tags: ["ablation", "kl-tuning", "2025-11"]
  status: "exploratory"              # exploratory | keep | baseline | failed

  # Provenance (auto-populated at runtime)
  run_id: "baseline__mix10-dir_tau_ca-het__20251112_143022"
  architecture_code: "mix10-dir_tau_ca-het"
  timestamp: "2025-11-12T14:30:22"

  # For resumed runs (optional, manual specification)
  resumed_from: null                 # run_id of parent run
  resumed_epoch: null
  config_notes: ""                   # Free-form notes about config changes

# Rest of config follows...
data:
  dataset: "mnist"
  ...
```

**Auto-generated legend file** (`results/NAMING_LEGEND.md`):
Documents all architecture codes and validation rules. Updated automatically as new patterns are used.

**Benefits:**
- **Self-describing**: Directory name encodes architecture, no need to open config
- **Greppable**: `ls results/ | grep "vamp.*tau"` finds all VampPrior + τ-classifier runs
- **Comparable**: Easy to spot architectural differences at a glance
- **User control**: Experiment name remains human-chosen and meaningful
- **No collisions**: Timestamp ensures uniqueness
- **Status tracking**: Can filter/cleanup exploratory runs

---

## Dimension B: Configuration Layer

### Current State
- Hand-authored YAML files in `use_cases/experiments/configs/`
- Full config snapshot saved to each run directory
- No structured metadata or architecture encoding

### Design Decision: Enhanced Config Metadata (File-Based Only)

**File structure (no change):**
```
use_cases/experiments/configs/
├── default.yaml                    # Recommended baseline
├── quick.yaml                      # Fast sanity check
├── my_experiment.yaml              # User experiments
└── ...
```

**Workflow:**
1. User edits/creates YAML config file
2. Runs: `poetry run python use_cases/experiments/run_experiment.py --config configs/my_experiment.yaml`
3. System generates architecture code from config
4. System saves full config snapshot with metadata to run directory

**Saved config.yaml includes:**
```yaml
experiment:
  name: "baseline"
  description: "Testing KL weight impact"
  tags: ["ablation", "kl-tuning"]
  status: "exploratory"

  # Auto-populated at runtime
  run_id: "baseline__mix10-dir_tau_ca-het__20251112_143022"
  architecture_code: "mix10-dir_tau_ca-het"
  timestamp: "2025-11-12T14:30:22"

# Full config follows (same as source file)
data:
  dataset: "mnist"
  ...
model:
  kl_weight: 1.0
  ...
```

**Benefits:**
- Simple: edit config files directly (familiar workflow)
- Complete: full snapshot preserved in each run
- Discoverable: architecture code auto-generated and validated
- Greppable: `grep 'kl_weight: 1.0' results/*/config.yaml`

---

## Dimension C: Logging & Observability

### Current State
```python
print(f"\n{'=' * 60}\nTraining Model\n{'=' * 60}")
print(f"Training complete in {train_time:.1f}s")
```
- No structured logging
- No loss component breakdown
- No persistent log file

### Design Decision: Hierarchical Structured Logging

**Hierarchical metric naming convention:**
```
loss.total
loss.recon
loss.kl.z                # KL(q(z|x,c) || p(z|c))
loss.kl.c                # KL(q(c|x) || π)
loss.classifier          # Classification loss (if enabled)
loss.diversity           # Entropy reward/penalty
loss.dirichlet           # Dirichlet regularization

metric.accuracy.train
metric.accuracy.val
metric.k_eff             # Effective components
metric.responsibility.confidence
```

**Log levels:**
- `INFO`: Normal progress (model init, epoch summaries, component status)
- `WARNING`: Concerning but non-fatal (component collapse, low accuracy)
- `ERROR`: Fatal errors

**Log output routing:**
```python
# Both stdout and logs/train.log
logger.info("Building SSVAE...")
logger.info("  Encoder: ConvEncoder (3.2M params)")
logger.info("  Prior: MixtureGaussianPrior (K=10, learnable_pi=True)")

# Component status messages
logger.info("τ-Classifier: ENABLED (num_components=10 >= num_classes=10)")
logger.info("Heteroscedastic decoder: ENABLED (sigma_min=0.05, sigma_max=0.5)")
```

**Training progress format:**
```
Epoch   10/100 | loss.total: 198.3 | loss.recon: 180.5 | loss.kl.z: 15.2 | loss.kl.c: 2.1 | time: 5.2s
Epoch   20/100 | loss.total: 185.7 | loss.recon: 170.2 | loss.kl.z: 13.8 | loss.kl.c: 1.7 | time: 5.1s
```

**File structure:**
```
results/run_20251112_143022/
├── logs/
│   ├── train.log                   # All INFO/WARNING/ERROR messages
│   └── metrics.csv                 # Epoch-level metrics (for plotting)
```

**Implementation approach:**
- Use Python `logging` module (stdlib, lightweight)
- Create `ExperimentLogger` wrapper with dual handlers (console + file)
- Update `history` dict to use hierarchical keys
- Update `summary.json` to use hierarchical keys

**Benefits:**
- Grep logs for specific losses: `grep 'loss.kl.c' logs/train.log`
- Consistent naming everywhere (logs, history, summary.json, plots)
- Persistent logs for debugging failed runs
- Clear component status eliminates "why is this missing?" questions

---

## Dimension D: Artifact Schema & Modularity

### Current State (Problem)
```python
@register_metric
def tau_classifier_metrics(context: MetricContext) -> Optional[MetricResult]:
    if not context.config.use_tau_classifier:
        return None  # ← Silent disappearance!

    return {"tau_classifier": {"label_coverage": ...}}
```

**Result:** When `use_tau_classifier=false`, τ metrics are absent from `summary.json` and REPORT.md. User can't tell if component is disabled or broken.

### Design Decision: Explicit Component Status Protocol

**New provider return type:**
```python
@dataclass
class ComponentResult:
    status: Literal["success", "disabled", "failed"]
    reason: Optional[str] = None       # Why disabled/failed?
    data: Optional[Dict[str, Any]] = None  # Metrics/plots if success
```

**Updated provider pattern:**
```python
@register_metric
def tau_classifier_metrics(context: MetricContext) -> ComponentResult:
    if not context.config.use_tau_classifier:
        return ComponentResult(
            status="disabled",
            reason="use_tau_classifier=false",
            data=None
        )

    try:
        metrics = compute_tau_metrics(context)
        return ComponentResult(
            status="success",
            data={"tau_classifier": metrics}
        )
    except Exception as e:
        logger.error(f"τ-classifier metrics failed: {e}")
        return ComponentResult(
            status="failed",
            reason=str(e),
            data=None
        )
```

**Updated REPORT.md template:**
```markdown
## Classification: τ-Classifier

**Status:** Disabled
**Reason:** `use_tau_classifier=false`
**To enable:** Set `model.use_tau_classifier: true` and ensure `num_components >= num_classes`

---

## Mixture Prior Evolution

**Status:** Success

![Evolution](figures/mixture/model_evolution.png)

**Metrics:**
- K_eff: 8.3
- Active components: 9/10
- ...
```

**Updated summary.json structure:**
```json
{
  "components": {
    "tau_classifier": {
      "status": "disabled",
      "reason": "use_tau_classifier=false"
    },
    "mixture_prior": {
      "status": "success",
      "data": {
        "k_eff": 8.3,
        "active_components": 9
      }
    },
    "heteroscedastic_decoder": {
      "status": "success",
      "data": {
        "sigma_mean": 0.12,
        "sigma_std": 0.03
      }
    }
  },
  "training": {
    "final_loss": 198.3,
    ...
  }
}
```

**Benefits:**
- **No silent failures**: Every component reports its status
- **Self-documenting**: Reports explain why things are missing
- **Debuggable**: Can distinguish "disabled by config" from "crashed"
- **Robust**: Adding a component always produces output (even if just "disabled")

---

## Implementation Roadmap

### Phase 1: Logging Infrastructure (Foundational)
1. Create `ExperimentLogger` class with dual output (stdout + file)
2. Update `history` dict keys to hierarchical format (`loss.recon` → `loss.kl.z`)
3. Add structured model initialization logging
4. Add training progress table with all loss components

**Why first:** Logging is cross-cutting; needed for all other phases. Hierarchical names must be consistent from the start.

### Phase 2: Component Status Protocol (Robustness)
1. Define `ComponentResult` dataclass
2. Update `MetricProvider` and `Plotter` type signatures
3. Refactor existing providers to return `ComponentResult`
4. Update `collect_metrics()` and `render_all_plots()` to handle new return type
5. Generate `components` section in `summary.json`

**Why second:** Fixes the "silent disappearance" pain point immediately.

### Phase 3: Architecture-Encoded Naming & Provenance
1. Implement architecture code generator from config
2. Add validation for architecture code combinations
3. Implement new directory naming: `{name}__{arch}__{timestamp}`
4. Expand `experiment` metadata section in config.yaml (architecture_code, status, tags)
5. Generate `results/NAMING_LEGEND.md` documentation

**Why third:** Builds on existing config structure; improves discoverability without breaking existing runs.

### Phase 4: Updated REPORT.md Template (Polish)
1. Update report generator to include component status sections
2. Add explicit "disabled" markers with enable instructions
3. Improve metric organization (group by component)

**Why last:** Depends on Phase 2 (component status). Nice-to-have polish.

**Note:** CLI overrides (`--set`) are not implemented. User works exclusively with config files.

---

## Migration Strategy

**Backward compatibility:**
- Existing configs work as-is (new metadata fields are optional/auto-generated)
- Existing `summary.json` structure preserved (new `components` section is additive)
- Existing REPORT.md format remains readable (new sections append)

**Deprecation path:**
- Phase 1-2: Old-style providers (return `None` or dict) still work, but log deprecation warnings
- After 2-3 weeks: Remove compatibility shims, require new return types

---

## Design Decisions Summary

**Finalized:**
- ✅ Architecture-encoded naming: `{name}__{arch}__{timestamp}`
- ✅ Hierarchical metric names: `loss.recon`, `loss.kl.z`, `metric.accuracy.val`
- ✅ Component status protocol: All components report `success|disabled|failed`
- ✅ Contrastive learning encoded in decoder segment: `ca-het-contr`
- ✅ No git tracking (local-only workflow)
- ✅ Run status: `exploratory | keep | baseline | failed`

**To decide during implementation:**
- Logging verbosity: Epoch-level progress (default), optionally batch-level via flag

---

## Success Criteria

**You'll know this design works when:**
1. ✅ Directory names are self-describing: `baseline__mix10-dir_tau_ca-het__20251112_143022`
2. ✅ Toggling any component always produces an explicit status message (no silent disappearance)
3. ✅ All loss components are visible during training: `loss.recon | loss.kl.z | loss.kl.c`
4. ✅ Hierarchical metric names are consistent everywhere (logs, history, summary.json, plots)
5. ✅ You can grep for experiments: `ls results/ | grep "vamp.*tau"`
6. ✅ Logs answer "why is X missing?" without re-running or reading code
7. ✅ The system scales from 10 throwaway runs/day to 100s of archived experiments

---

## Related Documentation

After implementation, update:
- `use_cases/experiments/README.md` §Configuration (document CLI overrides)
- `use_cases/experiments/README.md` §Understanding-Output (explain component status)
- `extending.md` §Adding-a-Metric (show new ComponentResult pattern)
