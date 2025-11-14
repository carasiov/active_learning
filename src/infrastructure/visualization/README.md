# Visualization Module

## Overview

The visualization module provides a comprehensive suite of diagnostic plots for VAE experiments, including loss curves, latent space analysis, and model-specific visualizations for mixture priors and semi-supervised learning.

## Architecture

The module uses a **registry pattern** for dynamic plot discovery and execution based on model configuration. The implementation is organized into domain-specific modules:

```
src/visualization/
├── __init__.py              # Public API exports
├── registry.py              # Orchestration and registry management
├── status.py                # Status objects for component results
│
├── plotters.py              # Registry bindings (~280 lines)
│   └─ All @register_plotter decorators
│
├── plot_utils.py            # Shared utilities (~270 lines)
│   ├─ Helper functions (sanitize names, color palettes, etc.)
│   └─ PlotGrid class for subplot management
│
├── core_plots.py            # Basic visualizations (~350 lines)
│   ├─ Loss curves
│   ├─ Latent space scatter plots
│   ├─ Reconstructions
│   └─ Report generation
│
├── mixture_plots.py         # Mixture-specific plots (~570 lines)
│   ├─ Latent by component
│   ├─ Channel responsibility visualization
│   ├─ Responsibility histograms
│   ├─ Mixture evolution (π dynamics)
│   ├─ Component embedding divergence
│   └─ Reconstruction by component
│
└── tau_plots.py             # τ-classifier plots (~280 lines)
    ├─ τ matrix heatmap
    ├─ Per-class accuracy
    └─ Certainty vs accuracy analysis
```

## Module Responsibilities

### `plotters.py` - Registry Bindings
Entry point for the visualization system. Contains:
- `@register_plotter` decorated functions that bind to the registry
- Wrapper functions (`_single_model_dict`, `_single_history_dict`) for API compatibility
- Conditional logic for enabling/disabling plots based on model config

### `plot_utils.py` - Shared Utilities
Common functions used across all plotting modules:
- `_sanitize_model_name()`: Convert model names to safe filenames
- `_build_label_palette()`: Generate consistent color palettes for class labels
- `_downsample_points()`: Subsample large datasets for plotting
- `_compute_limits()`: Calculate consistent axis limits across subplots
- `_extract_component_recon()`: Parse component-wise reconstructions from model output
- `_prep_image()`: Prepare image arrays for matplotlib
- `PlotGrid`: Helper class for creating multi-panel figures
- `safe_save_plot()`: Error-handling wrapper for saving figures

### `core_plots.py` - Basic Diagnostics
Fundamental visualizations applicable to all VAE models:
- **Loss Comparison**: Training/validation curves for all loss components
- **Latent Spaces**: 2D scatter plots colored by class labels
- **Reconstructions**: Original vs reconstructed image grids
- **Report Generation**: Markdown summary with embedded figures

### `mixture_plots.py` - Mixture Prior Analysis
Visualizations specific to mixture prior VAEs:
- **Latent by Component**: Scatter plots colored by component assignment
- **Channel Latent Responsibility**: Per-component latent space with alpha-blended responsibilities
- **Responsibility Histogram**: Distribution of max_c q(c|x) confidence
- **Mixture Evolution**: π weights and component usage over training
- **Component Embedding Divergence**: Pairwise distances between component embeddings
- **Reconstruction by Component**: How each component reconstructs inputs

### `tau_plots.py` - Semi-Supervised Diagnostics
Visualizations for τ-classifier semi-supervised models:
- **τ Matrix Heatmap**: Components-to-labels mapping with sparsity metrics
- **Per-Class Accuracy**: Classification performance breakdown by class
- **Certainty Analysis**: Calibration plot of prediction confidence vs accuracy

## Usage

### From Training Pipeline

```python
from visualization import VisualizationContext, render_all_plots

# Create context with experiment data
viz_context = VisualizationContext(
    model=trained_model,
    config=model_config,
    history=training_history,
    x_train=train_data,
    y_true=train_labels,
    figures_dir=output_dir / 'figures'
)

# Execute all applicable plots
visualization_meta = render_all_plots(viz_context)
```

### Adding New Plots

1. **Choose the appropriate module** based on plot category:
   - Basic VAE diagnostic → `core_plots.py`
   - Mixture prior specific → `mixture_plots.py`
   - τ-classifier specific → `tau_plots.py`

2. **Implement the plotting function**:
   ```python
   # In mixture_plots.py
   def plot_my_new_analysis(
       models: Dict[str, object],
       X_data: np.ndarray,
       output_dir: Path
   ):
       """Your plot implementation."""
       # ... plotting code ...
       safe_save_plot(fig, output_path)
   ```

3. **Register in `plotters.py`**:
   ```python
   @register_plotter
   def my_new_plotter(context: VisualizationContext) -> ComponentResult:
       """Registry wrapper for my new plot."""
       if not _should_run(context):
           return ComponentResult.disabled(reason="...")

       try:
           plot_my_new_analysis(
               _single_model_dict(context.model),
               context.x_train,
               context.figures_dir
           )
           return ComponentResult.success(data={})
       except Exception as e:
           return ComponentResult.failed(reason="...", error=e)
   ```

## Component Status System

Each plotter returns a `ComponentResult` indicating execution status:

- **`success`**: Plot generated successfully
- **`disabled`**: Not applicable for this model configuration (e.g., mixture plots for standard VAE)
- **`skipped`**: Preconditions not met (e.g., 2D latent space required but model has 10D)
- **`failed`**: Error occurred during generation

This status system enables:
- Graceful handling of inapplicable plots
- Detailed reporting in `summary.json`
- Easy debugging of visualization failures

## Output Structure

Plots are organized into subdirectories by category:

```
figures/
├── core/
│   ├── loss_comparison.png
│   ├── latent_spaces.png
│   └── {model_name}_reconstructions.png
│
├── mixture/
│   ├── latent_by_component.png
│   ├── responsibility_histogram.png
│   ├── {model_name}_evolution.png
│   ├── component_embedding_divergence.png
│   ├── {model_name}_reconstruction_by_component.png
│   └── channel_latents/
│       ├── channel_latents_grid.png
│       └── channel_XX.png
│
└── tau/
    ├── tau_matrix_heatmap.png
    ├── tau_per_class_accuracy.png
    └── tau_certainty_analysis.png
```

## Design Principles

1. **Separation of Concerns**: Implementation split by domain (core/mixture/tau)
2. **Registry Pattern**: Dynamic discovery without hardcoded dependencies
3. **Graceful Degradation**: Plots skip gracefully when not applicable
4. **Consistent API**: All functions accept `Dict[str, model]` for multi-model support
5. **Extensibility**: Easy to add new plots without modifying existing code
6. **Error Handling**: Failures in one plot don't prevent others from running

## Benefits of Refactoring

**Before** (monolithic `plotters.py`):
- 1,643 lines in single file
- Mixed concerns (core/mixture/tau logic interleaved)
- Difficult to navigate and modify
- Repeated utility code
- No unit testing

**After** (modular structure):
- Largest module: 570 lines (mixture_plots.py)
- Clear domain separation
- Easy to locate specific functionality
- Shared utilities in dedicated module
- Ready for unit testing

## Migration Notes

This refactoring maintains **100% backward compatibility**:
- Public API (`VisualizationContext`, `register_plotter`, `render_all_plots`) unchanged
- Registry auto-discovery still works via `from . import plotters`
- All plots produce identical output to pre-refactor version
- No changes needed in `train.py` or other consumers

## Testing

To validate the refactoring:

```bash
# Run quick experiment to verify all plots generate correctly
poetry run python run_experiment.py --config use_cases/experiments/configs/quick.yaml

# Check output
ls results/*/figures/core/
ls results/*/figures/mixture/
ls results/*/figures/tau/

# Verify summary includes plot status
cat results/*/summary.json | grep -A 20 "_plot_status"
```
