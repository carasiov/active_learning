"""Diagnostic framework for feature-specific model analysis.

This module provides an extensible system for exporting diagnostic artifacts
for different SSVAE features. Each feature can register a diagnostic exporter
that generates relevant analysis outputs.

Usage:
    from diagnostics import DIAGNOSTICS

    for diagnostic in DIAGNOSTICS:
        if diagnostic.should_export(config):
            diagnostic.export(model, data, labels, output_dir)

Adding a new diagnostic:
    1. Create a subclass of DiagnosticExporter
    2. Implement should_export() and export() methods
    3. Add instance to DIAGNOSTICS registry

Example features:
    - Mixture prior → MixtureDiagnostics (component usage, entropy)
    - Label store → LabelStoreDiagnostics (stored representations, retrieval stats)
    - OOD scoring → OODDiagnostics (score distributions, threshold analysis)
    - VampPrior → VampPriorDiagnostics (pseudo-input visualizations)
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ssvae import SSVAE, SSVAEConfig


class DiagnosticExporter:
    """Base class for feature-specific diagnostic exporters.

    Subclasses should implement:
    - should_export(): Return True if diagnostic applies to the given config
    - export(): Generate and save diagnostic artifacts
    """

    def should_export(self, config: SSVAEConfig) -> bool:
        """Return True if this diagnostic applies to the config.

        Args:
            config: The model configuration

        Returns:
            True if diagnostic should run, False otherwise
        """
        raise NotImplementedError

    def export(
        self,
        model: SSVAE,
        data: np.ndarray,
        labels: np.ndarray,
        output_dir: Path,
    ) -> None:
        """Export diagnostic artifacts to output directory.

        Args:
            model: Trained SSVAE model
            data: Validation data (for computing diagnostics)
            labels: True labels (for analysis, may contain NaN)
            output_dir: Directory to save artifacts
        """
        raise NotImplementedError


class MixtureDiagnostics(DiagnosticExporter):
    """Diagnostic exporter for mixture prior models.

    Exports:
        - component_usage.npy: Component assignment counts [K,]
        - component_responsibilities.npy: Average responsibilities per component [K,]
        - component_entropy.npy: Entropy of component distribution per sample [N,]
        - component_assignments.npy: Most likely component per sample [N,]
        - per_class_component_usage.npy: Component usage per class [num_classes, K]

    These diagnostics help answer:
        - Are all components being used?
        - Do components specialize to certain classes?
        - Is the model collapsing to fewer components?
    """

    def should_export(self, config: SSVAEConfig) -> bool:
        """Export if using mixture prior."""
        return config.prior_type == "mixture"

    def export(
        self,
        model: SSVAE,
        data: np.ndarray,
        labels: np.ndarray,
        output_dir: Path,
    ) -> None:
        """Export mixture prior diagnostics."""
        import jax.numpy as jnp

        # Forward pass to get component assignments
        x_jnp = jnp.array(data, dtype=jnp.float32)
        component_logits, _, _, _, _, _ = model._apply_fn(
            model.state.params,
            x_jnp,
            training=False,
        )

        # Convert to numpy for analysis
        component_logits_np = np.array(component_logits)

        # Compute responsibilities (softmax)
        import jax
        responsibilities = jax.nn.softmax(component_logits, axis=-1)
        responsibilities_np = np.array(responsibilities)

        # 1. Component usage (hard assignments)
        component_assignments = np.argmax(component_logits_np, axis=1)
        K = component_logits_np.shape[1]
        component_usage = np.bincount(component_assignments, minlength=K)

        # 2. Average responsibilities per component
        avg_responsibilities = np.mean(responsibilities_np, axis=0)

        # 3. Component entropy per sample
        log_resp = np.log(responsibilities_np + 1e-10)
        component_entropy = -np.sum(responsibilities_np * log_resp, axis=1)

        # 4. Per-class component usage (if labels available)
        labeled_mask = ~np.isnan(labels)
        if np.any(labeled_mask):
            labeled_labels = labels[labeled_mask].astype(int)
            labeled_assignments = component_assignments[labeled_mask]
            num_classes = int(np.max(labeled_labels)) + 1

            per_class_usage = np.zeros((num_classes, K), dtype=np.int32)
            for cls in range(num_classes):
                class_mask = (labeled_labels == cls)
                if np.any(class_mask):
                    class_assignments = labeled_assignments[class_mask]
                    per_class_usage[cls] = np.bincount(class_assignments, minlength=K)
        else:
            per_class_usage = None

        # Save all diagnostics
        diagnostics_dir = output_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        np.save(diagnostics_dir / "component_usage.npy", component_usage)
        np.save(diagnostics_dir / "component_responsibilities.npy", avg_responsibilities)
        np.save(diagnostics_dir / "component_entropy.npy", component_entropy)
        np.save(diagnostics_dir / "component_assignments.npy", component_assignments)

        if per_class_usage is not None:
            np.save(diagnostics_dir / "per_class_component_usage.npy", per_class_usage)

        # Generate summary text
        summary_path = diagnostics_dir / "mixture_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Mixture Prior Diagnostics\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Number of components (K): {K}\n")
            f.write(f"Samples analyzed: {len(data)}\n\n")

            f.write("Component Usage (hard assignments):\n")
            for k in range(K):
                percentage = 100 * component_usage[k] / len(data)
                f.write(f"  Component {k}: {component_usage[k]:4d} samples ({percentage:5.1f}%)\n")

            f.write(f"\nAverage Component Entropy: {np.mean(component_entropy):.4f}\n")
            f.write(f"  (Range: 0.0 = deterministic, {np.log(K):.4f} = uniform)\n\n")

            # Check for component collapse
            active_components = np.sum(component_usage > 0)
            if active_components < K:
                f.write(f"WARNING: Only {active_components}/{K} components active!\n")

            # Check for balanced usage
            max_usage = np.max(component_usage)
            min_usage = np.min(component_usage)
            if max_usage > 10 * min_usage and min_usage > 0:
                f.write(f"WARNING: Imbalanced component usage (max/min = {max_usage/min_usage:.1f}x)\n")

            if per_class_usage is not None:
                f.write("\nPer-Class Component Usage:\n")
                for cls in range(num_classes):
                    total_cls = np.sum(per_class_usage[cls])
                    if total_cls > 0:
                        dominant_component = np.argmax(per_class_usage[cls])
                        dominant_pct = 100 * per_class_usage[cls, dominant_component] / total_cls
                        f.write(f"  Class {cls}: dominant component {dominant_component} "
                               f"({dominant_pct:.1f}% of class samples)\n")

        print(f"  Mixture diagnostics saved to {diagnostics_dir}")
        print(f"    - component_usage.npy")
        print(f"    - component_responsibilities.npy")
        print(f"    - component_entropy.npy")
        print(f"    - component_assignments.npy")
        if per_class_usage is not None:
            print(f"    - per_class_component_usage.npy")
        print(f"    - mixture_summary.txt")


class ReconstructionDiagnostics(DiagnosticExporter):
    """Diagnostic exporter for reconstruction quality (applies to all models).

    Exports:
        - reconstructions.png: Grid showing original vs reconstruction (10 samples)
        - reconstruction_indices.npy: Indices of samples used (for reproducibility)

    These diagnostics help answer:
        - Is the model reconstructing inputs accurately?
        - Are there systematic failures (e.g., certain digits)?
        - Visual confirmation of reconstruction loss values
    """

    def should_export(self, config: SSVAEConfig) -> bool:
        """Always export reconstruction diagnostics."""
        return True

    def export(
        self,
        model: SSVAE,
        data: np.ndarray,
        labels: np.ndarray,
        output_dir: Path,
    ) -> None:
        """Export reconstruction diagnostics."""
        import matplotlib.pyplot as plt

        # Select samples: one per class (0-9) for reproducibility
        indices = []
        for digit in range(10):
            mask = labels == digit
            if mask.sum() > 0:
                indices.append(np.where(mask)[0][0])

        num_samples = len(indices)
        if num_samples == 0:
            print("  Warning: No labeled samples for reconstruction diagnostics")
            return

        X_samples = data[indices]

        # Get reconstructions
        _, reconstructions, _, _ = model.predict(X_samples)

        # Create diagnostics directory
        diagnostics_dir = output_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        # Create visualization: 2 columns (original, reconstruction), N rows (samples)
        fig, axes = plt.subplots(num_samples, 2, figsize=(4, 2 * num_samples))

        # Handle single sample case
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for idx in range(num_samples):
            # Original
            axes[idx, 0].imshow(X_samples[idx], cmap='gray')
            axes[idx, 0].axis('off')
            if idx == 0:
                axes[idx, 0].set_title('Original', fontsize=10)

            # Reconstruction
            axes[idx, 1].imshow(reconstructions[idx], cmap='gray')
            axes[idx, 1].axis('off')
            if idx == 0:
                axes[idx, 1].set_title('Reconstruction', fontsize=10)

        plt.tight_layout()
        recon_path = diagnostics_dir / 'reconstructions.png'
        plt.savefig(recon_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Save indices for reproducibility
        np.save(diagnostics_dir / 'reconstruction_indices.npy', np.array(indices))

        print(f"  Reconstruction diagnostics saved to {diagnostics_dir}")
        print(f"    - reconstructions.png")
        print(f"    - reconstruction_indices.npy")


# =============================================================================
# Future Diagnostics (Template Examples)
# =============================================================================

# class LabelStoreDiagnostics(DiagnosticExporter):
#     """Diagnostic exporter for label store feature.
#
#     Exports:
#         - stored_representations.npy: Stored latent representations
#         - retrieval_distances.npy: Distances between queries and retrieved items
#         - label_distribution.npy: Distribution of labels in store
#     """
#
#     def should_export(self, config: SSVAEConfig) -> bool:
#         return config.use_label_store
#
#     def export(self, model, data, labels, output_dir):
#         # Implementation here
#         pass


# class OODDiagnostics(DiagnosticExporter):
#     """Diagnostic exporter for OOD scoring feature.
#
#     Exports:
#         - ood_scores.npy: OOD scores for validation data
#         - score_histogram.png: Distribution of scores
#         - threshold_analysis.txt: Recommended thresholds
#     """
#
#     def should_export(self, config: SSVAEConfig) -> bool:
#         return config.use_ood_scoring
#
#     def export(self, model, data, labels, output_dir):
#         # Implementation here
#         pass


# =============================================================================
# Registry
# =============================================================================

# Global registry of all diagnostic exporters
# Add new diagnostics here as they are implemented
DIAGNOSTICS = [
    ReconstructionDiagnostics(),  # Always-on: visual verification of reconstruction quality
    MixtureDiagnostics(),         # Feature-specific: mixture prior analysis
    # LabelStoreDiagnostics(),    # Future
    # OODDiagnostics(),           # Future
    # VampPriorDiagnostics(),     # Future
]
