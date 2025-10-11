# Experiments

This directory contains reproducible experiment utilities and their outputs, kept self-contained within `experiments/`.

## Label Efficiency

Evaluate how many labeled examples are required for strong downstream classification performance.

Tools provided:

- `generate_labels.py` – deterministically sample balanced label sets of varying sizes (`experiments/label_sets/labels_<N>.csv`).
- `run_label_efficiency.sh` – orchestrate training runs across label budgets and store checkpoints, histories, and run metadata under `experiments/artifacts/label_efficiency/`.
- `collect_results.py` – gather the last epoch classification metrics from each run into a summary table.

### Quickstart

```bash
# Generate a balanced label file (defaults to experiments/label_sets/)
python experiments/generate_labels.py --num-labels 50

# Run all label efficiency trials with the dense encoder/decoder baseline
bash experiments/run_label_efficiency.sh dense

# Summarise the resulting training histories for a specific run tag
python experiments/collect_results.py --experiment-dir experiments/artifacts/label_efficiency/<run_tag>
```

Each script accepts additional flags documented via `--help`. All experiment artefacts remain in the `experiments/` tree so the core training pipeline stays uncluttered.
