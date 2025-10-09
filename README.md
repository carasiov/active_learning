Active Learning – JAX SSVAE

Overview
- Modular JAX/Flax implementation of a semi‑supervised VAE (SSVAE) with interactive training.
- Canonical API lives under `ssvae/`; legacy `src_tf/` remains as a shim for existing scripts.

Quickstart
1) Create a virtual environment and install requirements
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt`

2) Train
   - `python scripts/train.py --labels data/labels.csv --weights artifacts/checkpoints/ssvae.ckpt`

3) Infer latent space
   - `python scripts/infer.py --weights artifacts/checkpoints/ssvae.ckpt --output data/output_latent.npz --split train`

4) Explore and label
   - `python scripts/view_latent.py`

Canonical Imports
- `from ssvae import SSVAE`

Docs
- Structure: `docs/STRUCTURE.md`
- Refactor plan and progress: `ROOT/`
