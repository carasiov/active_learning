Active Learning – Semi-Supervised VAE (JAX/Flax)
================================================

This repository delivers a modular Semi-Supervised Variational Autoencoder (SSVAE) built on JAX, Flax, and Optax. It demonstrates how to learn useful latent structure from predominantly unlabeled data and fine-tune a classifier with only a handful of labels. 

Highlights
---------
- **Canonical JAX/Flax implementation:** `src/ssvae/` exposes the public API.
- **Composable architecture:** encoder/decoder/classifier components live under `src/model_components/`, enabling easy swaps via config.
- **Modular observability:** `src/callbacks/` provides training callbacks for console logging, CSV export, and loss plotting.
- **Pure training loop:** `src/training/` houses loss functions, the trainer, train state wrapper, and an interactive trainer for incremental labeling sessions.
- **Use-case bundles:** CLI entry points under `use_cases/scripts/` feed generated outputs into `artifacts/` (checkpoints, run histories, showcase figures).
- **End-to-end showcase notebook:** `use_cases/notebooks/showcase_ssvae.ipynb` walks through the three stages of semi-supervised learning.



Repository Structure
--------------------

```
active_learning/
├── artifacts/              # Generated outputs (checkpoints, per-run history, legacy TF weights)
├── data/                   # Labels.csv and generated inference outputs
├── docs/                   # Structure documentation, cleanup notes
├── src/                    # Installable packages (configs, ssvae, model_components, training)
│   ├── callbacks/
│   ├── configs/
│   ├── model_components/
│   ├── ssvae/
│   └── training/
├── use_cases/              # Reproducible workflows built on the library
│   ├── experiments/        # Experiment runners and their artifacts
│   ├── notebooks/          # Showcase and exploratory notebooks
│   └── scripts/            # CLI entry points (train, infer, viewers)
└── ROOT/                   # Refactor spec and progress tracker
```


Development Setup
-----------------

### Option 1: Using Devcontainer (Recommended)

The project includes a devcontainer configuration for reproducible GPU-accelerated development.

**Prerequisites:**
- VS Code with "Remote - SSH" and "Dev Containers" extensions
- Access to a machine with Docker and NVIDIA GPUs (e.g., the cluster)

**Setup:**

1. Connect to cluster via VS Code Remote SSH
2. Open project folder
3. Click "Reopen in Container" when prompted
4. Wait for initial build (~5-10 minutes first time)
5. Verify GPU: `python -c "import jax; print(jax.devices())"`

See `.devcontainer/README.md` for detailed instructions and troubleshooting.

**Benefits:**
- Automatic dependency installation via Poetry
- GPU access configured out of the box
- Consistent environment across team members
- Isolated from system Python

### Option 2: Manual Setup with Poetry

If you prefer not to use devcontainers:

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# For GPU support, ensure you have CUDA 12 installed
# JAX will automatically detect GPU if available

# Run code
poetry run python use_cases/scripts/train.py
```


Configuration
-------------

`src/configs/base.py` defines `SSVAEConfig`, covering:

- Architecture: `latent_dim`, `hidden_dims`, `encoder_type`, `decoder_type`, `classifier_type`.
- Loss weights: `recon_weight`, `kl_weight`, `label_weight`, optional `use_contrastive`, `contrastive_weight`.
- Training: `batch_size`, `learning_rate`, `max_epochs`, `patience`,  `grad_clip_norm`, `weight_decay`.

Example:

```python
from configs.base import SSVAEConfig
from ssvae import SSVAE

config = SSVAEConfig(
    latent_dim=2,
    hidden_dims=(256, 128, 64, 32),
    max_epochs=75,
    patience=10,
)
vae = SSVAE(input_dim=(28, 28), config=config)
```

The refactored training loop honors `patience` and checkpoint paths exactly once per improvement, ensuring weight formats stay consistent.

---

Training & Inference Scripts
----------------------------

### 1. Prepare Labels

Labels live in `data/mnist/labels.csv` with columns `Serial` and `label`. Unlabeled data can either be absent or marked as NaN. The interactive viewer updates this CSV directly.

### 2. Train

```bash
python use_cases/scripts/train.py \
  --labels data/mnist/labels.csv \
  --weights artifacts/checkpoints/ssvae.ckpt
```

- Preprocessing: `MinMaxScaler` followed by binarization (`> 0.5 -> 1.0`).
- Outputs (saved alongside the `--weights` path):
  - Checkpoint: `artifacts/checkpoints/ssvae.ckpt` (Flax serialization of params/opt state/step).
  - History CSV: `artifacts/checkpoints/ssvae_history.csv`.
  - Training plot: `artifacts/checkpoints/ssvae_loss.png`.

### 3. Inference

```bash
python use_cases/scripts/infer.py \
  --weights artifacts/checkpoints/ssvae.ckpt \
  --output data/output_latent.npz \
  --split train  # or test
```

`data/output_latent.npz` (consumed by the viewer) contains:

- `input` – normalized/binarized images.
- `latent` – 2D latent means.
- `tsne` – currently identical to `latent` (placeholder for future t-SNE).
- `reconstruted` – reconstructed images.
- `labels` – true MNIST digits.
- `pred_classes`, `pred_certainty` – classifier predictions and confidences.

### 4. Interactive Viewer

```bash
python use_cases/scripts/view_latent.py
```

Features:

- Scatter plot of the latent space.
- Click to select a point; view original and reconstruction.
- Press `0`–`9` to label, `d` to remove label.
- Color by user labels, predicted classes, true classes, or certainty.
- Updates `data/mnist/labels.csv` immediately.

### 5. Incremental Training (Optional)

`training/interactive_trainer.py` supports label–train cycles without reinitializing optimizers:

```python
from training.interactive_trainer import InteractiveTrainer
trainer = InteractiveTrainer(vae)
history = trainer.train_epochs(num_epochs=10, data=x_train, labels=labels)
latent = trainer.get_latent_space(x_train)
```

This is ideal for dashboards or manual labeling sessions where you add labels iteratively.

When you need custom observability (e.g., streaming to dashboards), pass a list of callbacks to `InteractiveTrainer` or directly to `Trainer.train()`. The default stack (console logging + CSV + plotting) is built in `SSVAE._build_callbacks`.

---

Callback System
---------------

Training observability is handled by the callback package (`src/callbacks/`):

- `TrainingCallback` (base class) defines the hook surface: `on_train_start`, `on_epoch_end`, and `on_train_end`.
- `ConsoleLogger` mirrors the legacy console table, including optional contrastive columns.
- `CSVExporter` writes run history with the same schema consumed by experiment collectors.
- `LossCurvePlotter` renders loss curves when Matplotlib is available (otherwise it no-ops).

Callbacks are instantiated in `SSVAE._build_callbacks`, so CLI scripts and notebooks automatically receive the default behavior without additional wiring.

### Extending Callbacks

1. Create a new callback in `src/callbacks/` (either a new module or alongside the existing ones) that subclasses `TrainingCallback`.
2. Override only the hooks you need; remember to convert JAX arrays to Python scalars (`float(...)`) before logging or serializing values.
3. Export the callback from `src/callbacks/__init__.py` if you want downstream code to import it via `from callbacks import YourCallback`.
4. Inject it by passing a callback list to `Trainer.train(..., callbacks=[...])`, to `InteractiveTrainer(..., callbacks=[...])`, or by extending `SSVAE._build_callbacks` in your own wrapper.

This architecture keeps the training loop free of I/O concerns while making it straightforward to add integrations like streaming loggers, experiment trackers, or custom visualizations.

---

Showcase Notebook
-----------------

`use_cases/notebooks/showcase_ssvae.ipynb` demonstrates the three stages of semi-supervised learning on a MNIST subset (≈12k samples) and runs in under 10 minutes on CPU:

1. **Stage 1 – Untrained:** random latent scatter.
2. **Stage 2 – Unsupervised:** clusters emerge from reconstruction-only training.
3. **Stage 3 – Semi-Supervised:** only 50 labels (5 per digit) yield meaningful classification boundaries.

Artifacts (saved automatically to `artifacts/showcase/`):

- `stage1_latent.png`, `stage2_latent.png`, `stage3_latent.png` – latent plots at each stage.
- `stage2_recon.png` – original vs reconstructed samples.
- `stage3_predictions.png`, `stage3_certainty.png` – classifier results.
- `comparison.png` – side-by-side visualization of all stages.
- `stage2_unsupervised.ckpt`, `stage3_semi_supervised.ckpt` – checkpoints between stages.

The notebook inserts `ROOT/src` into `sys.path`, so it can be run directly from `use_cases/notebooks/`:

```bash
source .venv/bin/activate
jupyter notebook use_cases/notebooks/showcase_ssvae.ipynb
# Kernel → Restart & Run All
```
