# Getting Started

Complete guide to installation, setup, and running your first experiment.

> **Note:** This guide covers the legacy `compare_models.py` workflow and general Python API usage. For the current primary experimentation workflow using `run_experiment.py`, see the **[Experiment Guide](../../EXPERIMENT_GUIDE.md)**.

---

## Requirements

### System Requirements

- **Python:** 3.11 or higher
- **Operating System:** Linux (Ubuntu 22.04 tested), macOS, Windows with WSL2
- **Memory:** 8GB RAM minimum (16GB recommended for full MNIST dataset)
- **GPU (optional):** NVIDIA GPU with CUDA 12+ for acceleration
  - CPU-only mode fully supported (slower training)
  - JAX automatically detects and uses available GPU

### Key Dependencies

Automatically installed via Poetry:
- **JAX/JAXlib:** Numerical computing and auto-differentiation
- **Flax:** Neural network library built on JAX
- **Optax:** Gradient-based optimization
- **NumPy:** Array operations
- **Matplotlib/Seaborn:** Visualization (for comparison tool)

### Optional Dependencies

- **PyYAML:** For YAML-based configuration files (comparison tool)
- **Dash/Plotly:** For interactive dashboard (included by default)

---

## Installation

### Option 1: Devcontainer (Recommended for GPU Development)

The devcontainer provides a reproducible environment with GPU passthrough and all dependencies pre-configured.

**Prerequisites:**
- VS Code with extensions: "Remote - SSH" and "Dev Containers"
- Access to a machine with Docker and NVIDIA GPUs

**Setup Steps:**

1. **Connect to cluster via Remote SSH:**
   ```
   Open VS Code → F1 → "Remote-SSH: Connect to Host"
   ```

2. **Open project folder:**
   ```
   File → Open Folder → Navigate to project directory
   ```

3. **Reopen in container:**
   ```
   VS Code detects .devcontainer/ → Click "Reopen in Container"
   OR: F1 → "Dev Containers: Reopen in Container"
   ```

4. **Wait for build** (~5-10 minutes first time):
   - Downloads CUDA 12.1 base image
   - Installs Python 3.11 and Poetry
   - Runs `poetry install` automatically
   - Configures GPU passthrough

5. **Verify installation:**
   ```bash
   # Check GPU access
   nvidia-smi
   
   # Verify JAX sees GPUs
   poetry run python -c "import jax; print(jax.devices())"
   # Expected: [cuda(id=0), cuda(id=1)] or similar
   ```

**Daily usage:** After initial build, reopening in container takes ~10 seconds. VS Code remembers the container configuration.

**Troubleshooting:** See [`.devcontainer/README.md`](../.devcontainer/README.md) for GPU issues, rebuild instructions, and device selection.

---

### Option 2: Manual Setup with Poetry

For local development or when devcontainer is not available.

**Step 1: Install Poetry**

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Step 2: Install project dependencies**

```bash
cd /path/to/active_learning_showcase
poetry install
```

This creates a virtual environment in `.venv/` and installs all dependencies.

**Step 3: Verify installation**

```bash
# Check Python environment
poetry run python --version

# Verify JAX installation
poetry run python -c "import jax; print(jax.devices())"
```

**GPU Support:**
- JAX automatically detects CUDA if available (requires CUDA 12+)
- If no GPU detected, JAX uses CPU (fully functional, just slower)
- To force CPU mode: `JAX_PLATFORMS=cpu poetry run python ...`

**Troubleshooting:**
- **Poetry not found:** Add to PATH or use full path to `poetry` binary
- **CUDA version mismatch:** Install CUDA 12+ or use CPU mode
- **Import errors:** Ensure you're using `poetry run` or activate the virtualenv

---

## Quick Start

### Run Your First Comparison

Train two models and compare their behavior:

```bash
poetry run python scripts/compare_models.py --models standard mixture_k10 --epochs 10
```

**What this does:**
1. Loads MNIST (5,000 samples, 50 labeled)
2. Trains a standard Gaussian prior model
3. Trains a mixture-of-10-Gaussians prior model
4. Generates comparison visualizations

**Expected output:** `artifacts/comparisons/20241031_143022/`
```
├── loss_comparison.png        # Training/validation loss curves
├── latent_spaces.png          # 2D latent space visualizations
├── summary.json               # Final metrics (loss, accuracy, training time)
├── COMPARISON_REPORT.md       # Comprehensive analysis
├── standard_checkpoint.ckpt   # Model weights
└── mixture_k10_checkpoint.ckpt
```

**Typical results:**
- Both models converge in 10 epochs
- Mixture model often shows better-separated clusters in latent space
- Training time: ~1 minute on GPU, ~5 minutes on CPU

---

### Verify Your Setup

**Check 1: GPU acceleration working?**

```bash
poetry run python -c "import jax; print('GPU available:', jax.devices()[0].platform == 'gpu')"
```

**Check 2: Compare training speeds**

```bash
# Force CPU mode
JAX_PLATFORMS=cpu poetry run python scripts/compare_models.py --epochs 5 --num-samples 1000

# Use GPU (if available)
poetry run python scripts/compare_models.py --epochs 5 --num-samples 1000
```

GPU should be 3-5× faster for typical workloads.

**Check 3: Dashboard works?**

```bash
poetry run python use_cases/dashboard/app.py
```

Open http://localhost:8050 and verify the interface loads.

---

## What Just Happened?

When you ran `compare_models.py`:

1. **Data preparation:** MNIST loaded, 5,000 samples randomly selected (seed=42)
2. **Label masking:** Only 50 samples kept their labels (rest set to NaN)
3. **Model initialization:** Two SSVAE models created with different priors
4. **Training:** Each model trained for 10 epochs with early stopping
5. **Evaluation:** Loss metrics computed on validation set (20% of data)
6. **Visualization:** Loss curves and latent spaces plotted
7. **Reporting:** Summary metrics and checkpoints saved

**Key insight:** Both models learned from 99% unlabeled data, demonstrating semi-supervised learning.

---

## Next Steps

**Explore the tools:**
- 📖 [Usage Guide](usage.md) - Detailed examples for all tools
- 🔬 [Comparison Tool](../../configs/comparisons/README.md) - YAML configs, advanced options
- 🎛️ [Dashboard](../../use_cases/dashboard/README.md) - Interactive interface

**Understand the model:**
- 📚 [Implementation Guide](../development/implementation.md) - Architecture, API reference, internals

**Start experimenting:**
- Try different priors: `--models standard mixture_k5 mixture_k20`
- Adjust labeled data: `--num-labeled 10` (extreme few-shot) or `--num-labeled 500`
- Test architectures: Modify YAML configs to use convolutional encoders
- Hyperparameter search: Create custom YAML configs

**Common first experiments:**

```bash
# How few labels can we use?
poetry run python scripts/compare_models.py --num-labeled 10 --epochs 20

# Does more data help?
poetry run python scripts/compare_models.py --num-samples 20000 --epochs 30

# Compare many mixture sizes
poetry run python scripts/compare_models.py --models standard mixture_k5 mixture_k10 mixture_k20
```

---

## Troubleshooting

### Installation Issues

**"Poetry command not found"**
- Ensure Poetry is in PATH: `export PATH="$HOME/.local/bin:$PATH"`
- Or use full path: `~/.local/bin/poetry install`

**"Python 3.11 not found"**
- Install Python 3.11 or higher
- Ubuntu: `sudo apt install python3.11`
- macOS: `brew install python@3.11`

**"poetry install fails with dependency conflicts"**
- Update Poetry: `poetry self update`
- Clear cache: `poetry cache clear --all pypi`
- Retry: `poetry install`

### GPU Issues

**"JAX not detecting GPU"**
- Verify CUDA: `nvidia-smi` should show GPU
- Check CUDA version: Must be 12.0 or higher
- Reinstall JAX with CUDA: See JAX installation docs
- Fallback to CPU: `JAX_PLATFORMS=cpu` (works fine, just slower)

**"Out of memory errors"**
- Reduce batch size in config: `batch_size: 64` (default 128)
- Use fewer samples: `--num-samples 1000`
- Use CPU mode: `JAX_PLATFORMS=cpu`

### Runtime Issues

**"ModuleNotFoundError: No module named 'ssvae'"**
- Ensure you're using `poetry run` prefix
- Or activate environment: `source .venv/bin/activate`

**"Comparison tool hangs or crashes"**
- Check logs in terminal
- Try CPU mode: `JAX_PLATFORMS=cpu`
- Reduce problem size: `--num-samples 1000 --epochs 5`

**Need more help?**
- Check specialized guides in [Documentation Map](../README.md#documentation-map)
- Review logs in `/tmp/ssvae_dashboard.log` (for dashboard)
- See `.devcontainer/README.md` for GPU-specific troubleshooting
