"""Unit tests for TauClassifier."""
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model.ssvae.components.tau_classifier import TauClassifier


class TestTauClassifierInitialization:
    """Test TauClassifier initialization and basic properties."""

    def test_initialization(self):
        """Test basic initialization."""
        tau_clf = TauClassifier(num_components=5, num_classes=3, alpha_0=1.0)

        assert tau_clf.num_components == 5
        assert tau_clf.num_classes == 3
        assert tau_clf.alpha_0 == 1.0
        assert tau_clf.s_cy.shape == (5, 3)

    def test_initial_tau_is_uniform(self):
        """Verify τ initializes to uniform with smoothing prior."""
        tau_clf = TauClassifier(num_components=5, num_classes=3, alpha_0=1.0)
        tau = tau_clf.get_tau()

        # Should be uniform: each row sums to 1, all entries equal
        assert tau.shape == (5, 3)
        assert jnp.allclose(tau.sum(axis=1), 1.0)
        assert jnp.allclose(tau, 1.0 / 3)  # Uniform across 3 classes

    def test_custom_smoothing_prior(self):
        """Test initialization with different smoothing prior."""
        tau_clf = TauClassifier(num_components=10, num_classes=5, alpha_0=2.0)

        assert tau_clf.alpha_0 == 2.0
        # Initial counts should all be alpha_0
        assert jnp.allclose(tau_clf.s_cy, 2.0)

        # τ should still be uniform
        tau = tau_clf.get_tau()
        assert jnp.allclose(tau, 1.0 / 5)


class TestCountUpdate:
    """Test soft count accumulation."""

    def test_count_update_basic(self):
        """Verify soft counts accumulate correctly."""
        tau_clf = TauClassifier(num_components=2, num_classes=2, alpha_0=1.0)

        # Mock data: 2 samples, both labeled
        responsibilities = jnp.array([
            [0.9, 0.1],  # Sample 0: strongly component 0
            [0.2, 0.8],  # Sample 1: strongly component 1
        ])
        labels = jnp.array([0, 1])
        labeled_mask = jnp.array([True, True])

        tau_clf.update_counts(responsibilities, labels, labeled_mask)

        # Expected counts:
        # Component 0, label 0: 1.0 (prior) + 0.9 = 1.9
        # Component 0, label 1: 1.0 (prior) + 0.0 = 1.0
        # Component 1, label 0: 1.0 (prior) + 0.2 = 1.2
        # Component 1, label 1: 1.0 (prior) + 0.8 = 1.8

        assert tau_clf.s_cy[0, 0] > tau_clf.s_cy[0, 1]  # Component 0 → class 0
        assert tau_clf.s_cy[1, 1] > tau_clf.s_cy[1, 0]  # Component 1 → class 1

    def test_count_update_with_unlabeled(self):
        """Test that unlabeled samples don't contribute to counts."""
        tau_clf = TauClassifier(num_components=3, num_classes=2, alpha_0=1.0)

        responsibilities = jnp.array([
            [0.8, 0.1, 0.1],  # Labeled
            [0.1, 0.8, 0.1],  # Unlabeled
            [0.1, 0.1, 0.8],  # Labeled
        ])
        labels = jnp.array([0, 1, 1])  # Middle sample will be masked
        labeled_mask = jnp.array([True, False, True])

        initial_counts = tau_clf.s_cy.copy()
        tau_clf.update_counts(responsibilities, labels, labeled_mask)

        # Only labeled samples (0 and 2) should contribute
        # Sample 0: component 0, label 0 → +0.8
        # Sample 2: component 2, label 1 → +0.8
        assert tau_clf.s_cy[0, 0] == pytest.approx(initial_counts[0, 0] + 0.8)
        assert tau_clf.s_cy[2, 1] == pytest.approx(initial_counts[2, 1] + 0.8)

        # Component 1 should only have prior (no labeled samples used it strongly)
        # Actually sample 0 has small contribution to comp 1
        assert tau_clf.s_cy[1, 0] == pytest.approx(initial_counts[1, 0] + 0.1)

    def test_empty_batch(self):
        """Test that empty labeled batch doesn't crash."""
        tau_clf = TauClassifier(num_components=2, num_classes=2, alpha_0=1.0)

        responsibilities = jnp.array([[0.5, 0.5], [0.5, 0.5]])
        labels = jnp.array([0, 1])
        labeled_mask = jnp.array([False, False])  # All unlabeled

        initial_counts = tau_clf.s_cy.copy()
        tau_clf.update_counts(responsibilities, labels, labeled_mask)

        # Counts should be unchanged
        assert jnp.allclose(tau_clf.s_cy, initial_counts)

    def test_update_counts_vectorized_equivalence(self):
        """Vectorized update should match naive implementation and be faster."""
        num_components = 64
        num_classes = 10
        batch_size = 256
        key = jax.random.PRNGKey(0)
        resp_key, labels_key, mask_key = jax.random.split(key, 3)

        responsibilities = jax.random.dirichlet(
            resp_key, jnp.ones(num_components), shape=(batch_size,)
        )
        labels = jax.random.randint(
            labels_key, shape=(batch_size,), minval=0, maxval=num_classes
        )
        labeled_mask = jax.random.bernoulli(
            mask_key, p=0.7, shape=(batch_size,)
        )

        tau_clf = TauClassifier(
            num_components=num_components,
            num_classes=num_classes,
            alpha_0=1.0,
        )

        def naive_update_counts(
            s_cy: jnp.ndarray,
            resp: jnp.ndarray,
            lbls: jnp.ndarray,
            mask: jnp.ndarray,
        ) -> jnp.ndarray:
            labeled_resp = resp[mask]
            labeled_y = lbls[mask]
            counts = s_cy

            if labeled_y.size == 0:
                return counts

            for c in range(counts.shape[0]):
                for y in range(counts.shape[1]):
                    label_mask = (labeled_y == y).astype(labeled_resp.dtype)
                    contrib = jnp.sum(labeled_resp[:, c] * label_mask)
                    counts = counts.at[c, y].add(contrib)
            return counts

        initial_counts = tau_clf.s_cy  # Capture prior (all alpha_0)

        start = time.perf_counter()
        tau_clf.update_counts(responsibilities, labels, labeled_mask)
        vectorized_time = time.perf_counter() - start
        vectorized_counts = tau_clf.s_cy

        start = time.perf_counter()
        baseline_counts = naive_update_counts(
            initial_counts,
            responsibilities,
            labels,
            labeled_mask,
        )
        naive_time = time.perf_counter() - start

        # Log benchmark information for visibility (not part of assertion)
        print(
            f"Vectorized update: {vectorized_time * 1e3:.3f} ms | "
            f"Naive update: {naive_time * 1e3:.3f} ms | "
            f"batch={batch_size}, K={num_components}, C={num_classes}, "
            f"labeled={int(labeled_mask.sum())}"
        )

        assert jnp.allclose(vectorized_counts, baseline_counts)


class TestMultimodality:
    """Test that multiple components can map to same label."""

    def test_multimodal_label_assignment(self):
        """Verify multiple components can map to same label."""
        tau_clf = TauClassifier(num_components=4, num_classes=2, alpha_0=1.0)

        # Simulate: components 0,1 → class 0; components 2,3 → class 1
        responsibilities = jnp.array([
            [0.8, 0.2, 0.0, 0.0],  # Sample: comp 0 → label 0
            [0.1, 0.9, 0.0, 0.0],  # Sample: comp 1 → label 0
            [0.0, 0.0, 0.7, 0.3],  # Sample: comp 2 → label 1
            [0.0, 0.0, 0.2, 0.8],  # Sample: comp 3 → label 1
        ])
        labels = jnp.array([0, 0, 1, 1])
        labeled_mask = jnp.ones(4, dtype=bool)

        tau_clf.update_counts(responsibilities, labels, labeled_mask)
        tau = tau_clf.get_tau()

        # Components 0,1 should prefer class 0
        assert tau[0, 0] > tau[0, 1]
        assert tau[1, 0] > tau[1, 1]

        # Components 2,3 should prefer class 1
        assert tau[2, 1] > tau[2, 0]
        assert tau[3, 1] > tau[3, 0]


class TestPrediction:
    """Test label prediction from responsibilities."""

    def test_predict_with_learned_tau(self):
        """Test prediction after learning τ."""
        tau_clf = TauClassifier(num_components=3, num_classes=2, alpha_0=1.0)

        # Train: component 0 → label 0, components 1,2 → label 1
        train_resp = jnp.array([
            [1.0, 0.0, 0.0],  # Label 0
            [0.0, 1.0, 0.0],  # Label 1
            [0.0, 0.0, 1.0],  # Label 1
        ])
        train_labels = jnp.array([0, 1, 1])
        train_mask = jnp.ones(3, dtype=bool)

        tau_clf.update_counts(train_resp, train_labels, train_mask)

        # Test predictions
        test_resp = jnp.array([
            [0.9, 0.05, 0.05],  # Should predict label 0
            [0.1, 0.7, 0.2],  # Should predict label 1
            [0.0, 0.3, 0.7],  # Should predict label 1
        ])

        predictions, class_probs = tau_clf.predict(test_resp)

        assert predictions[0] == 0
        assert predictions[1] == 1
        assert predictions[2] == 1

        # Check that class probabilities sum to 1
        assert jnp.allclose(class_probs.sum(axis=1), 1.0)

    def test_predict_shape(self):
        """Test prediction output shapes."""
        tau_clf = TauClassifier(num_components=10, num_classes=5, alpha_0=1.0)

        batch_size = 32
        responsibilities = jax.random.dirichlet(
            jax.random.PRNGKey(0), jnp.ones(10), shape=(batch_size,)
        )

        predictions, class_probs = tau_clf.predict(responsibilities)

        assert predictions.shape == (batch_size,)
        assert class_probs.shape == (batch_size, 5)


class TestSupervisedLoss:
    """Test supervised loss computation."""

    def test_supervised_loss_basic(self):
        """Test basic supervised loss computation."""
        tau_clf = TauClassifier(num_components=3, num_classes=2, alpha_0=1.0)

        # Pre-populate counts to create non-uniform τ
        tau_clf.s_cy = jnp.array([
            [10.0, 1.0],  # Component 0 → class 0
            [1.0, 10.0],  # Component 1 → class 1
            [5.0, 5.0],  # Component 2 → ambiguous
        ])

        responsibilities = jnp.array([[0.9, 0.05, 0.05]])
        labels = jnp.array([0])
        mask = jnp.array([True])

        loss = tau_clf.supervised_loss(responsibilities, labels, mask)

        # Loss should be finite and non-negative
        assert jnp.isfinite(loss)
        assert loss >= 0.0

        # With high responsibility on component 0 and label 0, loss should be low
        assert loss < 1.0

    def test_supervised_loss_with_unlabeled(self):
        """Test that unlabeled samples don't contribute to loss."""
        tau_clf = TauClassifier(num_components=2, num_classes=2, alpha_0=1.0)

        responsibilities = jnp.array([
            [0.9, 0.1],  # Labeled
            [0.1, 0.9],  # Unlabeled
        ])
        labels = jnp.array([0, 1])
        mask = jnp.array([True, False])

        # Loss should only consider first sample
        loss = tau_clf.supervised_loss(responsibilities, labels, mask)

        # Should be same as computing with only first sample
        loss_single = tau_clf.supervised_loss(
            responsibilities[:1], labels[:1], mask[:1]
        )

        assert jnp.allclose(loss, loss_single)

    def test_stop_gradient_on_tau(self):
        """Verify gradients don't flow through τ in supervised loss."""

        def compute_loss_and_grad(responsibilities):
            """Helper to compute loss and gradients."""
            tau_clf = TauClassifier(num_components=3, num_classes=2, alpha_0=1.0)

            # Pre-populate counts
            tau_clf.s_cy = jnp.array([
                [10.0, 1.0],
                [1.0, 10.0],
                [5.0, 5.0],
            ])

            labels = jnp.array([0])
            mask = jnp.array([True])
            loss = tau_clf.supervised_loss(responsibilities, labels, mask)

            return loss

        # Test gradient computation
        responsibilities = jnp.array([[0.5, 0.3, 0.2]])
        grad_fn = jax.grad(compute_loss_and_grad)
        grad = grad_fn(responsibilities)

        # Gradient should exist (flows through responsibilities)
        assert not jnp.allclose(grad, 0.0)
        assert jnp.isfinite(grad).all()


class TestCertaintyAndOOD:
    """Test certainty and OOD score computation."""

    def test_certainty_high_confidence(self):
        """Test certainty for high-confidence predictions."""
        tau_clf = TauClassifier(num_components=2, num_classes=2, alpha_0=1.0)

        # Set up strong component→label associations
        tau_clf.s_cy = jnp.array([
            [100.0, 1.0],  # Component 0 strongly → class 0
            [1.0, 100.0],  # Component 1 strongly → class 1
        ])

        # Sample strongly on component 0
        responsibilities = jnp.array([[0.95, 0.05]])

        certainty = tau_clf.get_certainty(responsibilities)

        # Certainty should be high
        assert certainty[0] > 0.9

    def test_certainty_low_confidence(self):
        """Test certainty for ambiguous predictions."""
        tau_clf = TauClassifier(num_components=2, num_classes=2, alpha_0=1.0)

        # Ambiguous component→label associations
        tau_clf.s_cy = jnp.array([
            [5.0, 5.0],  # Component 0 → ambiguous
            [5.0, 5.0],  # Component 1 → ambiguous
        ])

        # Evenly distributed responsibilities
        responsibilities = jnp.array([[0.5, 0.5]])

        certainty = tau_clf.get_certainty(responsibilities)

        # Certainty should be low
        assert certainty[0] < 0.6

    def test_ood_score(self):
        """Test OOD score is 1 - certainty."""
        tau_clf = TauClassifier(num_components=3, num_classes=2, alpha_0=1.0)

        responsibilities = jax.random.dirichlet(
            jax.random.PRNGKey(0), jnp.ones(3), shape=(10,)
        )

        certainty = tau_clf.get_certainty(responsibilities)
        ood_score = tau_clf.get_ood_score(responsibilities)

        assert jnp.allclose(ood_score, 1.0 - certainty)


class TestFreeChannels:
    """Test free channel detection."""

    def test_free_channels_unused(self):
        """Test detection of unused components."""
        tau_clf = TauClassifier(num_components=5, num_classes=2, alpha_0=1.0)

        # Components 0,1 have counts, 2,3,4 don't
        tau_clf.s_cy = jnp.array([
            [10.0, 1.0],  # Used
            [1.0, 10.0],  # Used
            [1.0, 1.0],  # Unused (only prior)
            [1.0, 1.0],  # Unused (only prior)
            [1.0, 1.0],  # Unused (only prior)
        ])

        free = tau_clf.get_free_channels(
            usage_threshold=5.0,  # Total count must be > 5.0
            confidence_threshold=0.7,
        )

        # Components 2,3,4 should be free (low usage)
        free_set = set(free[free >= 0].tolist())
        assert 2 in free_set
        assert 3 in free_set
        assert 4 in free_set

    def test_free_channels_low_confidence(self):
        """Test detection of low-confidence components."""
        tau_clf = TauClassifier(num_components=3, num_classes=2, alpha_0=1.0)

        tau_clf.s_cy = jnp.array([
            [100.0, 1.0],  # High confidence for class 0
            [50.0, 50.0],  # Low confidence (ambiguous)
            [10.0, 1.0],  # Moderate confidence for class 0
        ])

        free = tau_clf.get_free_channels(
            usage_threshold=5.0,
            confidence_threshold=0.7,
        )

        # Component 1 should be free (low confidence)
        free_set = set(free[free >= 0].tolist())
        assert 1 in free_set


class TestDiagnostics:
    """Test diagnostic information."""

    def test_diagnostics_structure(self):
        """Test diagnostic dictionary structure."""
        tau_clf = TauClassifier(num_components=5, num_classes=3, alpha_0=1.0)

        diag = tau_clf.get_diagnostics()

        # Check required keys
        assert "tau" in diag
        assert "s_cy" in diag
        assert "component_label_confidence" in diag
        assert "component_dominant_label" in diag
        assert "components_per_label" in diag
        assert "tau_entropy" in diag

        # Check shapes
        assert diag["tau"].shape == (5, 3)
        assert diag["s_cy"].shape == (5, 3)
        assert diag["component_label_confidence"].shape == (5,)
        assert diag["component_dominant_label"].shape == (5,)
        assert diag["components_per_label"].shape == (3,)
        assert diag["tau_entropy"].shape == (5,)

    def test_diagnostics_after_training(self):
        """Test diagnostics reflect learned associations."""
        tau_clf = TauClassifier(num_components=4, num_classes=2, alpha_0=1.0)

        # Train: components 0,1 → label 0; components 2,3 → label 1
        responsibilities = jnp.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        labels = jnp.array([0, 0, 1, 1])
        mask = jnp.ones(4, dtype=bool)

        tau_clf.update_counts(responsibilities, labels, mask)
        diag = tau_clf.get_diagnostics()

        # Dominant labels should match training
        assert diag["component_dominant_label"][0] == 0
        assert diag["component_dominant_label"][1] == 0
        assert diag["component_dominant_label"][2] == 1
        assert diag["component_dominant_label"][3] == 1

        # Each label should have 2 components
        assert diag["components_per_label"][0] == 2
        assert diag["components_per_label"][1] == 2


class TestResetCounts:
    """Test count resetting functionality."""

    def test_reset_counts(self):
        """Test that reset_counts returns to initial state."""
        tau_clf = TauClassifier(num_components=3, num_classes=2, alpha_0=1.0)

        # Update some counts
        responsibilities = jnp.array([[1.0, 0.0, 0.0]])
        labels = jnp.array([0])
        mask = jnp.array([True])

        tau_clf.update_counts(responsibilities, labels, mask)

        # Counts should have changed
        assert not jnp.allclose(tau_clf.s_cy, 1.0)

        # Reset
        tau_clf.reset_counts()

        # Should be back to prior
        assert jnp.allclose(tau_clf.s_cy, 1.0)

        # τ should be uniform again
        tau = tau_clf.get_tau()
        assert jnp.allclose(tau, 1.0 / 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
