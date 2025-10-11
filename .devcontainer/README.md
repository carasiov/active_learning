# Devcontainer Configuration

This directory contains the VS Code devcontainer configuration for GPU-accelerated development.

## What This Does

- Creates a Docker container with CUDA 12.1 (compatible with cluster's CUDA 12.4)
- Installs Python 3.11
- Installs Poetry and all project dependencies
- Passes through GPU access
- Configures VS Code with Python and Jupyter extensions

## Prerequisites

**On the cluster:**
- Docker installed ✓ (verified: Docker 27.5.0)
- NVIDIA GPU available ✓ (verified: 2× GTX 1080 Ti)
- nvidia-docker runtime (usually comes with Docker on GPU systems)

**On your local machine:**
- VS Code with extensions:
  - Remote - SSH
  - Dev Containers

## Usage

### First Time Setup

1. Connect to cluster via VS Code Remote SSH:
   - Open VS Code
   - Press `F1` or `Cmd+Shift+P`
   - Type "Remote-SSH: Connect to Host"
   - Enter: `acarasiov@inv241286`

2. Open project folder:
   - File → Open Folder
   - Navigate to `/mnt/morty/acarasiov/projects/active_learning`

3. Reopen in container:
   - VS Code detects `.devcontainer/`
   - Click "Reopen in Container" when prompted
   - OR: Press `F1` → "Dev Containers: Reopen in Container"

4. Wait for build (first time takes ~5-10 minutes):
   - Downloads CUDA base image
   - Installs Python and Poetry
   - Runs `poetry install`

5. Verify GPU access:
   ```bash
   python -c "import jax; print(jax.devices())"
   # Should show: [cuda(id=0), cuda(id=1)]
   ```

### Daily Usage

After the first build, reopening in container is fast (~10 seconds).

- Connect to cluster via Remote SSH
- VS Code remembers you're using a container
- Reopen in container when prompted

### Running Code

Inside the container:

```bash
# Train with GPU
python scripts/test_conv.py
# Should be much faster than CPU!

# Full training
python scripts/train.py

# Notebooks work too
jupyter notebook
```

### Rebuilding the Container

If you modify `.devcontainer/devcontainer.json`:

1. Press `F1` → "Dev Containers: Rebuild Container"
2. Wait for rebuild
3. Test GPU access again

## Troubleshooting

**"Could not connect to container"**
- Check Docker is running: `docker ps` on cluster
- Check you have permissions: `docker run hello-world`

**"GPU not detected inside container"**
- Verify outside container: `nvidia-smi` on cluster
- Check docker has GPU access: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`

**"Poetry install fails"**
- Check internet access from container
- Try rebuilding: "Dev Containers: Rebuild Container"

**"Container build is slow"**
- First build is always slow (downloading images)
- Subsequent builds reuse cached layers
- If stuck, check Docker disk space: `docker system df`

## Alternative: Local Development

You can also use this devcontainer locally (without GPU) for testing:

1. Open project folder in VS Code locally
2. "Reopen in Container"
3. JAX will use CPU mode (slower, but works for small tests)

The same container config works both places!

