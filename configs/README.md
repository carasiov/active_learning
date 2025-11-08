# Comparison Configs

Optional YAML configurations for `compare_models.py`.

## Usage

**Option 1: Command-line (No config needed)**
```bash
poetry run python scripts/compare_models.py \
  --models standard mixture_k5 \
  --num-samples 1000 \
  --num-labeled 50 \
  --epochs 10
```

**Option 2: YAML config**
```bash
poetry run python scripts/compare_models.py --config configs/comparison_quick_test.yaml
```

## Available Configs

### `comparison_quick_test.yaml`
- **Purpose**: Quick testing of refactored architecture
- **Samples**: 1,000 (50 labeled)
- **Epochs**: 10
- **Models**: Standard, Mixture (K=5)
- **Time**: ~2-3 minutes on CPU

### `comparison_full.yaml`
- **Purpose**: Comprehensive comparison
- **Samples**: 5,000 (100 labeled)
- **Epochs**: 30
- **Models**: Standard, Mixture (K=5, 10, 20)
- **Features**: Tests Dirichlet regularization, usage sparsity
- **Time**: ~15-20 minutes on CPU

## Config Format

```yaml
data:
  num_samples: 1000
  num_labeled: 50
  epochs: 10
  seed: 42

models:
  Model Name:
    prior_type: standard|mixture
    num_components: 10  # For mixture only
    latent_dim: 2
    hidden_dims: [256, 128, 64]
    # ... any SSVAEConfig parameter
```

## Predefined Models (Command-line)

If using `--models` without config:
- `standard` - Standard Gaussian prior
- `mixture_k5` - Mixture with K=5
- `mixture_k10` - Mixture with K=10
- `mixture_k20` - Mixture with K=20

Example:
```bash
poetry run python scripts/compare_models.py --models standard mixture_k10 --epochs 20
```
