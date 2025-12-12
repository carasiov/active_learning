"""
DiagnosticsCollector - Generates model diagnostics and visualizations.

Separates diagnostic logic from model training for cleaner architecture.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

import jax.numpy as jnp
import numpy as np

from rcmvae.domain.config import SSVAEConfig

COMPONENT_PRIORS = {"mixture", "geometric_mog", "vamp"}


class DiagnosticsCollector:
    """Collects and saves model diagnostics, especially for component-based priors.

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
        """Collect and save diagnostics for any component-based prior.

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
        if self.config.prior_type not in COMPONENT_PRIORS:
            return {}

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
        z_per_component_records: List[np.ndarray] = []

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
            if self.config.latent_dim == 2 and extras is not None:
                z_mean_per_component = extras.get("z_mean_per_component") if hasattr(extras, "get") else None
                if z_mean_per_component is not None:
                    z_per_component_records.append(np.asarray(z_mean_per_component))
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

            latent_payload = {
                "z_mean": z_array,
                "labels": labels_array,
                "q_c": resp_array,
            }
            if z_per_component_records:
                latent_payload["z_mean_per_component"] = np.concatenate(z_per_component_records, axis=0).astype(np.float32)

            np.savez(output_dir / "latent.npz", **latent_payload)

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
            Dictionary with 'z_mean', 'labels', 'q_c' and optional
            'z_mean_per_component' when saved, or None if not found
        """
        dir_path = output_dir or self._last_output_dir
        if dir_path is None:
            return None

        latent_path = Path(dir_path) / "latent.npz"
        if not latent_path.exists():
            return None

        data = np.load(latent_path)
        result = {
            "z_mean": data["z_mean"],
            "labels": data["labels"],
            "q_c": data["q_c"],
        }
        if "z_mean_per_component" in data.files:
            result["z_mean_per_component"] = data["z_mean_per_component"]
        return result

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
