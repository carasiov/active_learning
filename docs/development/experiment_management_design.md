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

### Design Decision: Enhanced Metadata, Same Structure

**Keep:** Simple timestamped directories (works well, no naming collisions)

**Add:** Structured provenance in `config.yaml` header:

```yaml
experiment:
  name: "mixture_ablation_kl05"
  description: "Testing reduced KL weight impact on component usage"
  tags: ["ablation", "kl-tuning", "2025-11"]
  status: "exploratory"              # exploratory | keep | baseline | failed

  # Provenance (auto-populated)
  run_id: "mixture_ablation_kl05_20251112_143022"
  timestamp: "2025-11-12T14:30:22"
  git_commit: "f8b5a17"              # Short hash of current commit
  git_dirty: false                   # Uncommitted changes present?

  # For resumed runs (optional)
  resumed_from: null                 # run_id of parent run
  resumed_epoch: null
  config_changes: {}                 # Dict of changed params

# Rest of config follows...
data:
  dataset: "mnist"
  ...
```

**Benefits:**
- Every run is self-documenting (provenance embedded in config.yaml)
- Can grep by tag: `grep -r '"ablation"' results/*/config.yaml`
- Git hash allows exact code reconstruction
- Status field enables cleanup scripts: `find results/ -name config.yaml | xargs grep 'status: "exploratory"'`

---

## Dimension B: Configuration Layer

### Current State
- Hand-authored YAML files
- No override mechanism
- No base config tracking

### Design Decision: YAML + CLI Overrides + Provenance

**File structure (no change):**
```
use_cases/experiments/configs/
├── default.yaml                    # Recommended baseline
├── quick.yaml                      # Fast sanity check
├── my_experiment.yaml              # User experiments
└── ...
```

**CLI override syntax:**
```bash
poetry run python use_cases/experiments/run_experiment.py \
  --config configs/default.yaml \
  --set model.kl_weight=1.0 \
  --set model.num_components=20 \
  --tag ablation \
  --tag kl-sweep
```

**Saved config.yaml records everything:**
```yaml
experiment:
  name: "default+conv+mixture"
  base_config: "configs/default.yaml"
  cli_overrides:
    model.kl_weight: 1.0
    model.num_components: 20
  tags: ["ablation", "kl-sweep"]

# Full resolved config below (base + overrides applied)
data:
  dataset: "mnist"
  ...
model:
  kl_weight: 1.0          # ← Override applied
  num_components: 20      # ← Override applied
  ...
```

**Implementation:**
- Simple dotted-key override parser (no Hydra dependency)
- Validate overrides against SSVAEConfig schema
- Save both original base path and final merged config

**Benefits:**
- Quick experimentation without editing files
- Full provenance (can see exactly what changed)
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

### Phase 3: Enhanced Provenance (Reproducibility)
1. Add git commit/dirty detection utility
2. Expand `experiment` metadata section in config.yaml
3. Record provenance automatically when creating run directory
4. Add `status` field and tagging support

**Why third:** Builds on existing config structure; no breaking changes.

### Phase 4: CLI Overrides (Workflow)
1. Implement dotted-key override parser
2. Add `--set key=value` CLI argument
3. Add `--tag` CLI argument
4. Record `base_config` and `cli_overrides` in saved config.yaml

**Why fourth:** Convenience feature; doesn't block other improvements.

### Phase 5: Updated REPORT.md Template (Polish)
1. Update report generator to include component status sections
2. Add explicit "disabled" markers with enable instructions
3. Improve metric organization (group by component)

**Why last:** Depends on Phase 2 (component status). Nice-to-have polish.

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

## Open Questions for User Verification

1. **Logging verbosity:** Epoch-level progress only, or also batch-level (every N batches)?
2. **CLI override syntax:** `--set model.kl_weight=1.0` or `--override model.kl_weight=1.0`?
3. **Run status values:** `exploratory | keep | baseline | failed` sufficient, or need more?
4. **Git dirty warning:** Should the system refuse to run if uncommitted changes exist, or just warn?

---

## Success Criteria

**You'll know this design works when:**
1. ✅ Toggling any component always produces an explicit status message (no silent disappearance)
2. ✅ All loss components are visible during training and in final summary
3. ✅ Every run has full provenance (can reproduce exact code + config state)
4. ✅ You can run a quick override without editing config files
5. ✅ Logs answer "why is X missing?" without re-running or reading code
6. ✅ The system scales from 10 throwaway runs/day to 100s of archived experiments

---

## Related Documentation

After implementation, update:
- `use_cases/experiments/README.md` §Configuration (document CLI overrides)
- `use_cases/experiments/README.md` §Understanding-Output (explain component status)
- `extending.md` §Adding-a-Metric (show new ComponentResult pattern)
