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

### Run Your First Experiment

Train a model and generate a comprehensive analysis:

```bash
# Quick test (7 seconds)
JAX_PLATFORMS=cpu poetry run python scripts/run_experiment.py --config configs/quick.yaml

# View results
cat artifacts/experiments/baseline_quick_*/REPORT.md
```

**What this does:**
1. Loads MNIST (1,000 samples for quick test)
2. Trains a standard VAE model
3. Generates visualizations, metrics, and a human-readable report

**Expected output:** `artifacts/experiments/baseline_quick_<timestamp>/`
```
├── REPORT.md                  # Human-readable summary with embedded visualizations
├── config.yaml                # Configuration snapshot
├── checkpoint.ckpt            # Trained model weights
├── summary.json               # Structured metrics
├── visualizations/
│   ├── loss_comparison.png    # Training curves
│   ├── latent_spaces.png      # 2D latent space by class
│   └── model_reconstructions.png  # Input/output samples
└── diagnostics/checkpoint/
    └── latent.npz             # Latent embeddings
```

**Typical results:**
- Model converges in ~10 epochs
- Loss decreases smoothly
- Runtime: ~7 seconds on CPU, ~3 seconds on GPU

**Next:** Try a full experiment with `configs/default.yaml` or `configs/mixture_example.yaml`

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

When you ran `run_experiment.py`:

1. **Configuration loading:** YAML config parsed and validated
2. **Data preparation:** MNIST loaded with specified sample size (1,000 for quick test)
3. **Label masking:** Only specified labeled samples kept (rest set to NaN)
4. **Model initialization:** SSVAE model created based on config (architecture, prior type)
5. **Training:** Model trained with early stopping and progress tracking
6. **Evaluation:** Comprehensive metrics computed (loss, clustering, mixture diagnostics)
7. **Visualization:** Multiple plots generated (losses, latent spaces, reconstructions)
8. **Reporting:** Human-readable report and structured JSON saved

**Key insight:** The model learned from mostly unlabeled data, demonstrating semi-supervised learning. The experiment workflow is configuration-driven, making it easy to reproduce and modify.

---

## Next Steps

**Master the primary workflow:**
- 📖 [Experiment Guide](../../EXPERIMENT_GUIDE.md) - Complete workflow: configuration → execution → interpretation
- 🔬 Try different configs in `configs/` directory

**Explore other tools:**
- 📖 [Usage Guide](usage.md) - Dashboard, Python API, and legacy comparison tool
- 🎛️ [Dashboard](../../use_cases/dashboard/README.md) - Interactive interface for active learning

**Understand the model:**
- 📚 [Conceptual Model](../theory/conceptual_model.md) - High-level vision and design rationale
- 📚 [Implementation Guide](../development/implementation.md) - Architecture, API reference, internals

**Start experimenting:**
- Try mixture prior: `poetry run python scripts/run_experiment.py --config configs/mixture_example.yaml`
- Adjust hyperparameters: Copy and edit configs (e.g., `kl_weight`, `num_components`)
- Test architectures: Modify YAML configs to use convolutional encoders
- Check status: Review [Implementation Roadmap](../theory/implementation_roadmap.md) for available features

**Common first experiments:**

```bash
# Full baseline with standard prior
poetry run python scripts/run_experiment.py --config configs/default.yaml

# Mixture model with evolution tracking
poetry run python scripts/run_experiment.py --config configs/mixture_example.yaml

# Create custom experiment
cp configs/default.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml, then:
poetry run python scripts/run_experiment.py --config configs/my_experiment.yaml
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
