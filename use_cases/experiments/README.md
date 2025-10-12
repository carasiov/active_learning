# Experiments

This directory contains reproducible experiment utilities and their outputs, kept self-contained within `use_cases/experiments/`.

## Label Efficiency

Evaluate how many labeled examples are required for strong downstream classification performance.

Tools provided:

- `generate_labels.py` – deterministically sample balanced label sets of varying sizes (`use_cases/experiments/label_sets/labels_<N>.csv`).
- `run_label_efficiency.sh` – orchestrate training runs across label budgets and store checkpoints, histories, and run metadata under `use_cases/experiments/artifacts/label_efficiency/`.
- `collect_results.py` – gather the last epoch classification metrics from each run into a summary table.

### Quickstart

```bash
# Generate a balanced label file (defaults to use_cases/experiments/label_sets/)
python use_cases/experiments/generate_labels.py --num-labels 50

# Run all label efficiency trials with the dense encoder/decoder baseline
bash use_cases/experiments/run_label_efficiency.sh dense

# Summarise the resulting training histories for a specific run tag
python use_cases/experiments/collect_results.py --experiment-dir use_cases/experiments/artifacts/label_efficiency/<run_tag>
```

Each script accepts additional flags documented via `--help`. All experiment artefacts remain in the `use_cases/experiments/` tree so the core training pipeline stays uncluttered.
