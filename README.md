Active Learning – Semi-Supervised VAE (JAX/Flax)
================================================

This repository delivers a modular Semi-Supervised Variational Autoencoder (SSVAE) built on JAX, Flax, and Optax. It demonstrates how to learn useful latent structure from predominantly unlabeled data and fine-tune a classifier with only a handful of labels. The codebase evolved through a four-phase /ROOT refactor mission and now serves as an experimentation platform for active learning and interactive training workflows.

Highlights
---------
- **Canonical JAX/Flax implementation:** `ssvae/` exposes the public API; TensorFlow code and shims have been removed.
- **Composable architecture:** encoder/decoder/classifier components live under `model_components/`, enabling easy swaps via config.
- **Pure training loop:** `training/` houses loss functions, the trainer, train state wrapper, and an interactive trainer for incremental labeling sessions.
- **Scripts and artifacts:** CLI entry points under `scripts/` feed generated outputs into `artifacts/` (checkpoints, progress plots/CSVs, showcase figures).
- **End-to-end showcase notebook:** `notebooks/showcase_ssvae.ipynb` walks through the three stages of semi-supervised learning.

The repository is ready for experimentation, presentation, and further research extensions.

---

Repository Structure
--------------------

```
active_learning/
├── artifacts/              # Generated outputs (checkpoints, progress, showcase)
├── configs/                # Configuration dataclasses (SSVAEConfig)
├── data/                   # Labels.csv and generated inference outputs
├── docs/                   # Structure documentation, cleanup notes
├── model_components/       # Flax encoders/decoders/classifier + factory helpers
├── notebooks/              # Showcase and experimental notebooks
├── scripts/                # CLI entry points (train, infer, view, utilities)
├── ssvae/                  # Public JAX SSVAE models
├── training/               # Losses, trainer, train state, interactive trainer
└── ROOT/                   # Refactor spec and progress tracker
```

- `model_components/` replaced the old `models/` directory to avoid confusion with saved artifacts.
- `artifacts/` is the canonical location for generated checkpoints (`artifacts/checkpoints/`) and training curves (`artifacts/progress/`).
- Legacy TensorFlow weights remain under `models/` only for archival purposes and are ignored by `.gitignore`.
- `docs/STRUCTURE.md` and `docs/CLEANUP_NOTES.md` contain additional structure notes and follow-up suggestions.

---

Environment & Dependencies
---------------------------

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

- Requires Python 3.9+.
- Depends on CPU-compatible JAX/Flax/Optax builds (GPU optional).
- Downloads MNIST via `sklearn.datasets.fetch_openml`; the first run fetches the dataset to the OpenML cache.

---

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
poetry run python scripts/train.py
```

### Option 3: Legacy (requirements.txt)

**Note:** This is deprecated. Use Poetry for new setups.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

Configuration
-------------

`configs/base.py` defines `SSVAEConfig`, covering:

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

Labels live in `data/labels.csv` with columns `Serial` and `label`. Unlabeled data can either be absent or marked as NaN. The interactive viewer updates this CSV directly.

### 2. Train

```bash
python scripts/train.py \
  --labels data/labels.csv \
  --weights artifacts/checkpoints/ssvae.ckpt
```

- Preprocessing: `MinMaxScaler` followed by binarization (`> 0.5 -> 1.0`).
- Outputs:
  - Checkpoint: `artifacts/checkpoints/ssvae.ckpt` (Flax serialization of params/opt state/step).
  - Progress CSV: `artifacts/progress/ssvae_history.csv`.
  - Training plot: `artifacts/progress/ssvae_loss_plot.png`.

### 3. Inference

```bash
python scripts/infer.py \
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
python scripts/view_latent.py
```

Features:

- Scatter plot of the latent space.
- Click to select a point; view original and reconstruction.
- Press `0`–`9` to label, `d` to remove label.
- Color by user labels, predicted classes, true classes, or certainty.
- Updates `data/labels.csv` immediately.

### 5. Incremental Training (Optional)

`training/interactive_trainer.py` supports label–train cycles without reinitializing optimizers:

```python
from training.interactive_trainer import InteractiveTrainer
trainer = InteractiveTrainer(vae)
history = trainer.train_epochs(num_epochs=10, data=x_train, labels=labels)
latent = trainer.get_latent_space(x_train)
```

This is ideal for dashboards or manual labeling sessions where you add labels iteratively.

---

Showcase Notebook
-----------------

`notebooks/showcase_ssvae.ipynb` demonstrates the three stages of semi-supervised learning on a MNIST subset (≈12k samples) and runs in under 10 minutes on CPU:

1. **Stage 1 – Untrained:** random latent scatter.
2. **Stage 2 – Unsupervised:** clusters emerge from reconstruction-only training.
3. **Stage 3 – Semi-Supervised:** only 50 labels (5 per digit) yield meaningful classification boundaries.

Artifacts (saved automatically to `artifacts/showcase/`):

- `stage1_latent.png`, `stage2_latent.png`, `stage3_latent.png` – latent plots at each stage.
- `stage2_recon.png` – original vs reconstructed samples.
- `stage3_predictions.png`, `stage3_certainty.png` – classifier results.
- `comparison.png` – side-by-side visualization of all stages.
- `stage2_unsupervised.ckpt`, `stage3_semi_supervised.ckpt` – checkpoints between stages.

The notebook inserts the project root into `sys.path`, so it can be run directly from `notebooks/`:

```bash
source .venv/bin/activate
jupyter notebook notebooks/showcase_ssvae.ipynb
# Kernel → Restart & Run All
```

---

Verifying the Installation
---------------------------

To validate your environment manually (mirrors the notebook tests):

1. **Sanity import:**
   ```bash
   python -c "import ssvae; print(ssvae.SSVAE)"
   ```
2. **Train on MNIST (subset) and check `artifacts/progress/` + `artifacts/checkpoints/` outputs.**
3. **Infer latent space:** confirm `data/output_latent.npz` contains the expected keys.
4. **Interactive trainer smoke test:** run the small script in `docs/CLEANUP_NOTES.md` to ensure the latent space responds to new labels.

The comprehensive test plan (and its successful execution) is logged in `ROOT/progress_tracker.txt` under the “End-to-End Verification Test” entry dated 2025-10-09.

---

Development Notes & Future Work
-------------------------------

Accomplished (per /ROOT mission & follow-ups):

- Extracted configuration, losses, components, and trainer into modular packages.
- Added `InteractiveTrainer` for incremental training workflows.
- Replaced TensorFlow implementation with a canonical JAX/Flax model; removed `src_tf/` entirely.
- Standardized imports (`from ssvae import SSVAE`) and artifact layout (`artifacts/`).
- Provided a showcase notebook and detailed documentation (this README and `docs/STRUCTURE.md`).

Suggested next steps:

- Implement convolutional encoder/decoder variants and expose them via config toggles.
- Replace the contrastive loss stub with a real implementation (e.g., supervised contrastive, InfoNCE).
- Add a data-loader utility (e.g., `data/mnist.py`) so scripts/notebooks share preprocessing code.
- Integrate the interactive trainer into a UI (Streamlit/Panel) for rapid human-in-the-loop labeling.
- Add automated tests for loss functions, trainer loops, and component shapes.
- Migrate legacy `.h5` weights from `models/` into an archival directory or convert them if needed.
