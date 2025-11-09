"""
DiagnosticsCollector - Generates model diagnostics and visualizations.

Separates diagnostic logic from model training for cleaner architecture.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

import jax.numpy as jnp
import numpy as np

from ssvae.config import SSVAEConfig

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class DiagnosticsCollector:
    """Collects and saves model diagnostics, especially for mixture priors.

    This class handles:
    - Component usage statistics
    - Latent space visualizations
    - Responsibility distributions
    - π (mixture weights) tracking

    Separating this from the main model makes it easier to:
    - Test diagnostic logic independently
    - Add new diagnostics without touching model code
    - Disable diagnostics in production
    """

    def __init__(self, config: SSVAEConfig):
        """Initialize diagnostics collector.

        Args:
            config: Model configuration
        """
        self.config = config
        self._last_output_dir: Path | None = None

    def collect_mixture_stats(
        self,
        apply_fn: Callable,
        params: Dict,
        data: np.ndarray,
        labels: np.ndarray,
        output_dir: Path,
        *,
        batch_size: int = 1024,
    ) -> Dict[str, float]:
        """Collect and save mixture prior diagnostics.

        Args:
            apply_fn: Model forward function
            params: Model parameters
            data: Input data (images)
            labels: Labels for data
            output_dir: Directory to save diagnostics
            batch_size: Batch size for processing

        Returns:
            Dictionary with computed metrics:
                - K_eff: Effective number of components
                - responsibility_confidence_mean: Mean of max_c q(c|x)
                - active_components: Number of components with usage > 1%

        Saves:
            - component_usage.npy: Mean usage per component
            - component_entropy.npy: Mean entropy of responsibilities
            - pi.npy: Learned mixture weights (if available)
            - latent.npz: Latent codes, labels, and responsibilities (if latent_dim=2)
        """
        if self.config.prior_type != "mixture":
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._last_output_dir = output_dir

        eps = 1e-8
        usage_sum: np.ndarray | None = None
        entropy_sum = 0.0
        count = 0
        component_label_counts: np.ndarray | None = None
        num_classes = int(self.config.num_classes)

        z_records: List[np.ndarray] = []
        resp_records: List[np.ndarray] = []
        label_records: List[np.ndarray] = []
        pi_array: np.ndarray | None = None

        # Process in batches
        total = data.shape[0]
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_inputs = jnp.asarray(data[start:end])

            # Forward pass
            forward_output = apply_fn(params, batch_inputs, training=False)
            component_logits, z_mean, _, _, _, _, extras = forward_output

            # Extract mixture-specific outputs
            responsibilities = extras.get("responsibilities") if hasattr(extras, "get") else None
            if responsibilities is None:
                return

            resp_np = np.asarray(responsibilities)

            # Accumulate usage statistics
            if usage_sum is None:
                usage_sum = np.zeros(resp_np.shape[1], dtype=np.float64)
            usage_sum += resp_np.sum(axis=0)

            # Compute entropy
            entropy_batch = -resp_np * np.log(resp_np + eps)
            entropy_sum += entropy_batch.sum()
            count += resp_np.shape[0]

            # Accumulate component-label statistics (ignore unlabeled samples)
            batch_labels = labels[start:end]
            if component_label_counts is None:
                num_components = resp_np.shape[1]
                component_label_counts = np.zeros((num_components, num_classes), dtype=np.float64)
            valid_mask = ~np.isnan(batch_labels)
            if valid_mask.any():
                label_int = batch_labels[valid_mask].astype(np.int32)
                label_int = np.clip(label_int, 0, num_classes - 1)
                resp_valid = resp_np[valid_mask]
                one_hot = np.zeros((label_int.size, num_classes), dtype=np.float64)
                one_hot[np.arange(label_int.size), label_int] = 1.0
                component_label_counts += resp_valid.T @ one_hot

            # Collect latent space data (for visualization)
            z_records.append(np.asarray(z_mean))
            resp_records.append(resp_np)
            label_records.append(labels[start:end])

            # Extract π (mixture weights) once
            if pi_array is None:
                pi_val = extras.get("pi") if hasattr(extras, "get") else None
                if pi_val is not None:
                    pi_array = np.asarray(pi_val)

        if usage_sum is None or count == 0:
            return {}

        # Compute final statistics
        component_usage = (usage_sum / count).astype(np.float32)
        component_entropy = np.array(entropy_sum / count, dtype=np.float32)

        # Save component statistics
        np.save(output_dir / "component_usage.npy", component_usage)
        np.save(output_dir / "component_entropy.npy", component_entropy)

        if pi_array is not None:
            np.save(output_dir / "pi.npy", pi_array.astype(np.float32))

        if component_label_counts is not None:
            component_label_counts = component_label_counts.astype(np.float32)
            np.save(output_dir / "component_label_counts.npy", component_label_counts)
            csv_path = output_dir / "component_label_counts.csv"
            totals = component_label_counts.sum(axis=1, keepdims=True)
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("component,label,count,proportion\n")
                for comp_idx in range(component_label_counts.shape[0]):
                    total = totals[comp_idx, 0]
                    for label_idx in range(num_classes):
                        count_val = component_label_counts[comp_idx, label_idx]
                        proportion = (count_val / total) if total > 0 else 0.0
                        f.write(f"{comp_idx},{label_idx},{count_val:.6f},{proportion:.6f}\n")

        # Compute derived metrics
        eps = 1e-8
        p_c = component_usage / (component_usage.sum() + eps)  # Normalize to probabilities
        H_c = -np.sum(p_c * np.log(p_c + eps))
        K_eff = float(np.exp(H_c))

        # Active components (usage > 1%)
        active_components = int(np.sum(component_usage > 0.01))

        # Responsibility confidence (mean of max_c q(c|x))
        if resp_records:
            all_responsibilities = np.concatenate(resp_records, axis=0)
            max_responsibilities = all_responsibilities.max(axis=1)
            responsibility_confidence_mean = float(max_responsibilities.mean())
        else:
            responsibility_confidence_mean = 0.0

        # Save latent visualization data (only for 2D latent)
        if self.config.latent_dim == 2 and z_records:
            z_array = np.concatenate(z_records, axis=0).astype(np.float32)
            resp_array = np.concatenate(resp_records, axis=0).astype(np.float32)
            labels_array = np.concatenate(label_records, axis=0)

            np.savez(
                output_dir / "latent.npz",
                z_mean=z_array,
                labels=labels_array,
                q_c=resp_array,
            )

        # Return computed metrics
        metrics = {
            "K_eff": K_eff,
            "active_components": active_components,
            "responsibility_confidence_mean": responsibility_confidence_mean,
        }

        if component_label_counts is not None:
            totals = component_label_counts.sum(axis=1)
            majority_labels = component_label_counts.argmax(axis=1).astype(int)
            majority_conf = np.divide(
                component_label_counts[np.arange(component_label_counts.shape[0]), majority_labels],
                np.maximum(totals, eps),
                out=np.zeros_like(totals),
                where=totals > 0,
            )
            metrics["component_majority_labels"] = majority_labels.tolist()
            metrics["component_majority_confidence"] = majority_conf.tolist()

        return metrics

    @property
    def last_output_dir(self) -> Path | None:
        """Get the last directory where diagnostics were saved."""
        return self._last_output_dir

    def load_component_usage(self, output_dir: Path | None = None) -> np.ndarray | None:
        """Load component usage statistics.

        Args:
            output_dir: Directory to load from (uses last if None)

        Returns:
            Component usage array or None if not found
        """
        dir_path = output_dir or self._last_output_dir
        if dir_path is None:
            return None

        usage_path = Path(dir_path) / "component_usage.npy"
        if not usage_path.exists():
            return None

        return np.load(usage_path)

    def load_latent_data(self, output_dir: Path | None = None) -> Dict[str, np.ndarray] | None:
        """Load latent space visualization data.

        Args:
            output_dir: Directory to load from (uses last if None)

        Returns:
            Dictionary with 'z_mean', 'labels', 'q_c' or None if not found
        """
        dir_path = output_dir or self._last_output_dir
        if dir_path is None:
            return None

        latent_path = Path(dir_path) / "latent.npz"
        if not latent_path.exists():
            return None

        data = np.load(latent_path)
        return {
            "z_mean": data["z_mean"],
            "labels": data["labels"],
            "q_c": data["q_c"],
        }

    @staticmethod
    def compute_accuracy(
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Compute classification accuracy.

        Args:
            predictions: Predicted class indices (shape: [N])
            labels: True class labels (shape: [N])

        Returns:
            Accuracy as a float in [0, 1]
        """
        # Filter out unlabeled samples (NaN labels)
        valid_mask = ~np.isnan(labels)
        if not valid_mask.any():
            return 0.0

        predictions_valid = predictions[valid_mask]
        labels_valid = labels[valid_mask]

        correct = (predictions_valid == labels_valid).sum()
        total = len(labels_valid)

        return float(correct / total) if total > 0 else 0.0

    @staticmethod
    def compute_clustering_metrics(
        component_assignments: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """Compute clustering quality metrics (NMI and ARI).

        Args:
            component_assignments: Component assignments from argmax_c q(c|x) (shape: [N])
            labels: True class labels (shape: [N])

        Returns:
            Dictionary with 'nmi' and 'ari' scores, or empty dict if sklearn unavailable
        """
        try:
            from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
        except ImportError:
            return {}

        # Filter out unlabeled samples (NaN labels)
        valid_mask = ~np.isnan(labels)
        if not valid_mask.any():
            return {}

        component_assignments_valid = component_assignments[valid_mask]
        labels_valid = labels[valid_mask]

        nmi = normalized_mutual_info_score(labels_valid, component_assignments_valid)
        ari = adjusted_rand_score(labels_valid, component_assignments_valid)

        return {
            "nmi": float(nmi),
            "ari": float(ari),
        }

    def visualize_tau_matrix(
        self,
        params: Dict,
        output_dir: Path | None = None,
        *,
        title: str = "Component → Label Associations (τ matrix)",
        annotate_values: bool = True,
        figsize: tuple = (12, 8),
    ) -> Path | None:
        """Visualize τ matrix showing component→label associations.

        This visualization shows the τ_{c,y} matrix where each row is a component
        and each column is a label. High values indicate strong associations.

        Args:
            params: Model parameters containing τ matrix
            output_dir: Directory to save figure (uses last if None)
            title: Figure title
            annotate_values: Whether to annotate cells with values
            figsize: Figure size (width, height)

        Returns:
            Path to saved figure or None if plotting unavailable

        Example:
            >>> diagnostics = DiagnosticsCollector(config)
            >>> diagnostics.visualize_tau_matrix(
            ...     model.state.params,
            ...     output_dir=Path("results"),
            ... )
        """
        if not PLOTTING_AVAILABLE:
            print("Warning: matplotlib/seaborn not available. Skipping τ matrix visualization.")
            return None

        if not self.config.use_tau_classifier:
            print("Warning: τ-classifier not enabled. Skipping τ matrix visualization.")
            return None

        dir_path = Path(output_dir) if output_dir else self._last_output_dir
        if dir_path is None:
            print("Warning: No output directory specified. Skipping τ matrix visualization.")
            return None

        dir_path.mkdir(parents=True, exist_ok=True)

        # Extract τ matrix from parameters
        try:
            from ssvae.components.tau_classifier import extract_tau_from_params
            tau = extract_tau_from_params(params)
            tau_np = np.array(tau)
        except (ValueError, KeyError) as e:
            print(f"Warning: Could not extract τ matrix: {e}")
            return None

        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)

        # Use seaborn for nice heatmap
        sns.heatmap(
            tau_np,
            annot=annotate_values,
            fmt='.3f' if annotate_values else None,
            cmap='YlOrRd',
            cbar_kws={'label': 'Association Strength τ_{c,y}'},
            xticklabels=[f'{i}' for i in range(self.config.num_classes)],
            yticklabels=[f'{i}' for i in range(self.config.num_components)],
            ax=ax,
            vmin=0,
            vmax=1,
        )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Label (y)', fontsize=12)
        ax.set_ylabel('Component (c)', fontsize=12)

        # Add grid for better readability
        ax.set_xticks(np.arange(self.config.num_classes) + 0.5, minor=False)
        ax.set_yticks(np.arange(self.config.num_components) + 0.5, minor=False)

        plt.tight_layout()

        # Save figure
        output_path = dir_path / "tau_matrix.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"τ matrix visualization saved to: {output_path}")
        return output_path

    def visualize_tau_with_usage(
        self,
        params: Dict,
        output_dir: Path | None = None,
        *,
        title: str = "Component Specialization Analysis",
        figsize: tuple = (16, 8),
    ) -> Path | None:
        """Visualize τ matrix alongside component usage statistics.

        Creates a two-panel figure showing:
        1. τ matrix heatmap
        2. Component usage bar chart

        This helps identify which components are active and their label specializations.

        Args:
            params: Model parameters containing τ matrix
            output_dir: Directory to save figure (uses last if None)
            title: Figure title
            figsize: Figure size (width, height)

        Returns:
            Path to saved figure or None if plotting unavailable
        """
        if not PLOTTING_AVAILABLE:
            print("Warning: matplotlib/seaborn not available. Skipping visualization.")
            return None

        if not self.config.use_tau_classifier:
            print("Warning: τ-classifier not enabled. Skipping visualization.")
            return None

        dir_path = Path(output_dir) if output_dir else self._last_output_dir
        if dir_path is None:
            print("Warning: No output directory specified. Skipping visualization.")
            return None

        dir_path.mkdir(parents=True, exist_ok=True)

        # Extract τ matrix
        try:
            from ssvae.components.tau_classifier import extract_tau_from_params
            tau = extract_tau_from_params(params)
            tau_np = np.array(tau)
        except (ValueError, KeyError) as e:
            print(f"Warning: Could not extract τ matrix: {e}")
            return None

        # Load component usage
        usage = self.load_component_usage(dir_path)
        if usage is None:
            print("Warning: Component usage not available. Falling back to simple visualization.")
            return self.visualize_tau_matrix(params, output_dir, title=title)

        # Create two-panel figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [3, 1]})

        # Panel 1: τ matrix heatmap
        sns.heatmap(
            tau_np,
            annot=False,  # Too crowded with many components
            cmap='YlOrRd',
            cbar_kws={'label': 'τ_{c,y}'},
            xticklabels=[f'{i}' for i in range(self.config.num_classes)],
            yticklabels=[f'{i}' for i in range(self.config.num_components)],
            ax=ax1,
            vmin=0,
            vmax=1,
        )

        ax1.set_title('Component → Label Associations (τ)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Label', fontsize=10)
        ax1.set_ylabel('Component', fontsize=10)

        # Panel 2: Component usage bar chart
        component_indices = np.arange(self.config.num_components)
        colors = ['green' if u > 0.01 else 'gray' for u in usage]

        ax2.barh(component_indices, usage, color=colors, alpha=0.7)
        ax2.set_ylim(-0.5, self.config.num_components - 0.5)
        ax2.invert_yaxis()  # Match τ matrix orientation
        ax2.set_xlabel('Usage', fontsize=10)
        ax2.set_title('Component Usage', fontsize=12, fontweight='bold')
        ax2.axvline(0.01, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Active threshold')
        ax2.legend(fontsize=8)

        # Add overall title
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        # Save figure
        output_path = dir_path / "tau_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"τ analysis visualization saved to: {output_path}")
        return output_path

    def save_tau_summary(
        self,
        params: Dict,
        output_dir: Path | None = None,
    ) -> Path | None:
        """Save τ matrix summary statistics to text file.

        Args:
            params: Model parameters containing τ matrix
            output_dir: Directory to save summary (uses last if None)

        Returns:
            Path to saved summary file or None if unavailable
        """
        if not self.config.use_tau_classifier:
            return None

        dir_path = Path(output_dir) if output_dir else self._last_output_dir
        if dir_path is None:
            return None

        dir_path.mkdir(parents=True, exist_ok=True)

        # Extract τ matrix
        try:
            from ssvae.components.tau_classifier import extract_tau_from_params
            tau = extract_tau_from_params(params)
            tau_np = np.array(tau)
        except (ValueError, KeyError) as e:
            print(f"Warning: Could not extract τ matrix: {e}")
            return None

        # Compute summary statistics
        max_tau_per_component = tau_np.max(axis=1)
        dominant_label_per_component = tau_np.argmax(axis=1)
        components_per_label = [np.sum(dominant_label_per_component == label) for label in range(self.config.num_classes)]

        # Write summary
        summary_path = dir_path / "tau_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("τ MATRIX SUMMARY\n")
            f.write("="*80 + "\n\n")

            f.write(f"Matrix shape: [{self.config.num_components} components, {self.config.num_classes} labels]\n")
            f.write(f"Smoothing parameter α₀: {self.config.tau_alpha_0}\n\n")

            f.write("Component Specializations:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Component':<12} {'Dominant Label':<15} {'Confidence':<12} {'Label Distribution'}\n")
            f.write("-"*80 + "\n")

            for c in range(self.config.num_components):
                dominant = dominant_label_per_component[c]
                confidence = max_tau_per_component[c]
                distribution = ', '.join([f"{label}:{tau_np[c, label]:.3f}" for label in range(self.config.num_classes)])
                f.write(f"{c:<12} {dominant:<15} {confidence:<12.3f} {distribution}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("Label Coverage:\n")
            f.write("-"*80 + "\n")
            for label in range(self.config.num_classes):
                count = components_per_label[label]
                f.write(f"Label {label}: {count} component(s) specialized\n")

            f.write("\n" + "="*80 + "\n")

        print(f"τ summary saved to: {summary_path}")
        return summary_path
