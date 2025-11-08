# Extending the System

> **Purpose:** Step-by-step tutorials for extending the SSVAE codebase with new features. For architectural overview, see [System Architecture](architecture.md). For module reference, see [Implementation Guide](implementation.md).

---

## Overview

The codebase is designed for extensibility through:
- **Protocol-based abstractions** (PriorMode)
- **Factory pattern** (centralized component creation)
- **Configuration-driven** design (add parameters without touching core code)

This guide provides practical tutorials for common extension tasks.

---

## Tutorial 1: Adding a New Prior (VampPrior)

### Goal
Implement VampPrior (Variational Mixture of Posteriors) as described in the [Mathematical Specification](../theory/mathematical_specification.md).

### Steps

#### Step 1: Create Prior Module

Create `src/ssvae/priors/vamp.py`:

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional
from jax import Array
from jax.random import PRNGKey

class VampPrior:
    """Variational Mixture of Posteriors prior.

    Prior is defined as p(z) = sum_k pi_k * q(z|u_k)
    where u_k are learned pseudo-inputs.
    """

    def __init__(
        self,
        num_components: int,
        input_shape: tuple,
        encoder: nn.Module,
        uniform_weights: bool = True
    ):
        """Initialize VampPrior.

        Args:
            num_components: Number of pseudo-inputs (K)
            input_shape: Shape of pseudo-inputs (e.g., (28, 28))
            encoder: Encoder network to get q(z|u_k)
            uniform_weights: If True, use uniform pi; else learnable
        """
        self.num_components = num_components
        self.input_shape = input_shape
        self.encoder = encoder
        self.uniform_weights = uniform_weights

    def initialize_pseudo_inputs(self, key: PRNGKey, data_sample: Array):
        """Initialize pseudo-inputs from data.

        Strategy: Select K random data points or use k-means centers.
        """
        # Simple initialization: random data points
        indices = jax.random.choice(
            key,
            data_sample.shape[0],
            shape=(self.num_components,),
            replace=False
        )
        pseudo_inputs = data_sample[indices]
        return pseudo_inputs

    def kl_divergence(
        self,
        z_mean: Array,
        z_logvar: Array,
        pseudo_inputs: Array,
        encoder_params: dict
    ) -> Array:
        """Compute KL(q(z|x) || p(z)) where p(z) = sum_k pi_k q(z|u_k).

        Uses Monte Carlo estimation:
        KL = E_q[log q(z|x) - log p(z)]
           = E_q[log q(z|x) - log sum_k pi_k q(z|u_k)]

        Args:
            z_mean: Encoded mean from x
            z_logvar: Encoded log-variance from x
            pseudo_inputs: Learned pseudo-inputs u_k
            encoder_params: Encoder parameters for q(z|u_k)
        """
        # Sample z ~ q(z|x)
        eps = jax.random.normal(
            jax.random.PRNGKey(0),
            shape=z_mean.shape
        )
        z = z_mean + jnp.exp(0.5 * z_logvar) * eps

        # Compute log q(z|x)
        log_q_z_x = self._log_gaussian_prob(z, z_mean, z_logvar)

        # Compute log p(z) = log sum_k pi_k q(z|u_k)
        log_p_z = self._log_prior_prob(z, pseudo_inputs, encoder_params)

        # KL divergence
        kl = log_q_z_x - log_p_z
        return kl.mean()

    def _log_gaussian_prob(self, z: Array, mean: Array, logvar: Array) -> Array:
        """Log probability of z under Gaussian(mean, var)."""
        var = jnp.exp(logvar)
        log_prob = -0.5 * (
            jnp.sum(jnp.log(2 * jnp.pi * var), axis=-1)
            + jnp.sum((z - mean)**2 / var, axis=-1)
        )
        return log_prob

    def _log_prior_prob(
        self,
        z: Array,
        pseudo_inputs: Array,
        encoder_params: dict
    ) -> Array:
        """Compute log p(z) = log sum_k pi_k q(z|u_k)."""
        K = self.num_components

        # Get q(z|u_k) for all pseudo-inputs
        # Shape: (K, latent_dim)
        u_means = []
        u_logvars = []
        for k in range(K):
            u_k = pseudo_inputs[k]
            # Encode pseudo-input
            mean_k, logvar_k = self.encoder.apply(
                encoder_params,
                u_k[None, ...],  # Add batch dim
                deterministic=True
            )
            u_means.append(mean_k[0])
            u_logvars.append(logvar_k[0])

        u_means = jnp.stack(u_means)      # (K, latent_dim)
        u_logvars = jnp.stack(u_logvars)  # (K, latent_dim)

        # Compute log q(z|u_k) for each component
        # Broadcasting: z is (batch, latent_dim), means are (K, latent_dim)
        log_q_z_uk = jax.vmap(
            lambda mean, logvar: self._log_gaussian_prob(
                z[:, None, :],  # (batch, 1, latent_dim)
                mean[None, :],  # (1, latent_dim)
                logvar[None, :]  # (1, latent_dim)
            )
        )(u_means, u_logvars)  # (K, batch)

        # log sum_k pi_k q(z|u_k)
        # Assuming uniform pi_k = 1/K
        log_pi = jnp.log(1.0 / K)
        log_p_z = jax.scipy.special.logsumexp(
            log_pi + log_q_z_uk,
            axis=0
        )  # (batch,)

        return log_p_z

    def sample(self, key: PRNGKey, latent_dim: int, num_samples: int = 1) -> Array:
        """Sample from VampPrior p(z) = sum_k pi_k q(z|u_k).

        Strategy:
        1. Sample component k ~ Cat(pi)
        2. Sample z ~ q(z|u_k)
        """
        # For now, return standard normal (placeholder)
        # Full implementation requires access to pseudo-inputs
        return jax.random.normal(key, shape=(num_samples, latent_dim))
```

#### Step 2: Integrate into Factory

Update `src/ssvae/factory.py`:

```python
from ssvae.priors.vamp import VampPrior

class SSVAEFactory:
    @staticmethod
    def create_prior(config: SSVAEConfig, encoder=None, input_shape=None):
        """Create prior distribution based on config."""
        if config.prior_type == "standard":
            return StandardPrior()

        elif config.prior_type == "mixture":
            return MixturePrior(
                num_components=config.num_components,
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_weight=config.dirichlet_weight
            )

        elif config.prior_type == "vamp":
            # VampPrior requires encoder and input shape
            if encoder is None or input_shape is None:
                raise ValueError(
                    "VampPrior requires encoder and input_shape"
                )
            return VampPrior(
                num_components=config.num_components,
                input_shape=input_shape,
                encoder=encoder,
                uniform_weights=config.get("vamp_uniform_weights", True)
            )

        else:
            raise ValueError(f"Unknown prior_type: {config.prior_type}")
```

#### Step 3: Update Configuration

Add VampPrior parameters to `src/ssvae/config.py`:

```python
@dataclass
class SSVAEConfig:
    # ... existing fields ...

    # VampPrior specific
    vamp_uniform_weights: bool = True       # Use uniform or learnable pi
    vamp_pseudo_init: str = "random"        # "random" or "kmeans"
    vamp_prior_shaping: bool = False        # Enable MMD/MC-KL shaping
    vamp_shaping_weight: float = 0.0        # Prior shaping weight
```

#### Step 4: Update Loss Computation

Modify `src/training/losses.py` to handle VampPrior's KL:

```python
def kl_divergence(
    z_mean, z_logvar, prior,
    component_logits=None,
    pseudo_inputs=None,  # New: for VampPrior
    encoder_params=None  # New: for VampPrior
):
    """Compute KL divergence based on prior type."""
    if isinstance(prior, VampPrior):
        return prior.kl_divergence(
            z_mean, z_logvar,
            pseudo_inputs, encoder_params
        )
    else:
        # Standard or Mixture
        return prior.kl_divergence(z_mean, z_logvar, component_logits)
```

#### Step 5: Test VampPrior

Create `tests/test_vamp_prior.py`:

```python
import jax
import jax.numpy as jnp
from ssvae import SSVAE, SSVAEConfig

def test_vamp_prior_basic():
    """Test VampPrior initialization and training."""
    config = SSVAEConfig(
        latent_dim=2,
        prior_type="vamp",
        num_components=10,
        max_epochs=5
    )

    model = SSVAE(input_dim=(28, 28), config=config)

    # Create dummy data
    X = jax.random.normal(jax.random.PRNGKey(0), (100, 28, 28))
    y = jnp.array([0] * 100)

    # Should train without errors
    history = model.fit(X, y, "test_vamp.ckpt")

    assert "loss" in history
    assert history["loss"][-1] < history["loss"][0]  # Loss should decrease
```

#### Step 6: Use VampPrior

```python
from ssvae import SSVAE, SSVAEConfig

config = SSVAEConfig(
    latent_dim=10,
    prior_type="vamp",
    num_components=50,
    vamp_uniform_weights=True,
    max_epochs=100
)

model = SSVAE(input_dim=(28, 28), config=config)
history = model.fit(X_train, y_train, "vamp_model.ckpt")
```

---

## Tutorial 2: Adding Component-Aware Decoder

### Goal
Implement decoder conditioning $p_\theta(x|z,c)$ as specified in [Mathematical Specification](../theory/mathematical_specification.md).

### Steps

#### Step 1: Create Component-Aware Decoder

Update `src/ssvae/components/decoders.py`:

```python
class ComponentAwareDecoder(nn.Module):
    """Decoder conditioned on both z and component c.

    Implements p(x|z,c) by concatenating channel embedding with z.
    """
    output_shape: Tuple[int, ...]
    hidden_dims: Tuple[int, ...]
    num_components: int
    embedding_dim: int = 16  # Dimension of channel embeddings
    dropout_rate: float = 0.0

    def setup(self):
        # Channel embeddings (one per component)
        self.channel_embeddings = self.param(
            'channel_embeddings',
            nn.initializers.normal(stddev=0.02),
            (self.num_components, self.embedding_dim)
        )

        # Decoder layers (input is z + embedding)
        input_dim = self.hidden_dims[0] + self.embedding_dim

        layers = []
        for i, hidden_dim in enumerate(self.hidden_dims[1:]):
            layers.append(nn.Dense(hidden_dim))
            layers.append(nn.relu)
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(rate=self.dropout_rate))

        # Output layer
        output_dim = int(jnp.prod(jnp.array(self.output_shape)))
        layers.append(nn.Dense(output_dim))
        layers.append(nn.sigmoid)

        self.decoder_net = nn.Sequential(layers)

    def __call__(
        self,
        z: Array,
        component_index: Optional[Array] = None,
        component_probs: Optional[Array] = None,
        deterministic: bool = True
    ):
        """Decode with component awareness.

        Args:
            z: Latent vector (batch, latent_dim)
            component_index: Hard component index (batch,) - for generation
            component_probs: Soft probabilities q(c|x) (batch, K) - for training
            deterministic: Whether to use dropout

        Returns:
            Reconstruction (batch, *output_shape)
        """
        if component_index is not None:
            # Hard component selection (generation)
            embeddings = self.channel_embeddings[component_index]  # (batch, emb_dim)
            combined = jnp.concatenate([z, embeddings], axis=-1)
            return self._decode(combined, deterministic)

        elif component_probs is not None:
            # Soft component weighting (training with Top-M gating)
            # Take top M components for efficiency
            top_m = 5
            top_indices = jnp.argsort(component_probs, axis=-1)[:, -top_m:]
            top_probs = jnp.take_along_axis(
                component_probs, top_indices, axis=-1
            )
            # Renormalize
            top_probs = top_probs / top_probs.sum(axis=-1, keepdims=True)

            # Compute reconstruction for each top component
            batch_size = z.shape[0]
            reconstructions = []

            for m in range(top_m):
                comp_idx = top_indices[:, m]  # (batch,)
                embeddings = self.channel_embeddings[comp_idx]
                combined = jnp.concatenate([z, embeddings], axis=-1)
                recon_m = self._decode(combined, deterministic)
                reconstructions.append(recon_m)

            # Weighted average
            reconstructions = jnp.stack(reconstructions, axis=1)  # (batch, M, *)
            weights = top_probs[..., None, None]  # (batch, M, 1, 1)
            final_recon = (reconstructions * weights).sum(axis=1)

            return final_recon

        else:
            raise ValueError("Must provide component_index or component_probs")

    def _decode(self, combined: Array, deterministic: bool):
        """Apply decoder network."""
        output = self.decoder_net(combined, deterministic=deterministic)
        return output.reshape(-1, *self.output_shape)
```

#### Step 2: Update Factory

Modify `src/ssvae/factory.py`:

```python
@staticmethod
def create_decoder(config: SSVAEConfig, input_shape, key):
    """Create decoder based on config."""
    if config.decoder_type == "dense":
        if config.get("component_aware_decoder", False):
            # Component-aware decoder
            return ComponentAwareDecoder(
                output_shape=input_shape,
                hidden_dims=tuple(reversed(config.hidden_dims)),
                num_components=config.num_components,
                embedding_dim=config.get("channel_embedding_dim", 16),
                dropout_rate=config.dropout_rate
            )
        else:
            # Standard decoder
            return DenseDecoder(...)

    # ... other decoder types
```

#### Step 3: Update Configuration

```python
@dataclass
class SSVAEConfig:
    # ... existing fields ...

    # Component-aware decoder
    component_aware_decoder: bool = False
    channel_embedding_dim: int = 16
    top_m_gating: int = 5  # Number of top components for training
```

#### Step 4: Update Training

Modify forward pass in `src/ssvae/models.py`:

```python
def _forward(self, x, deterministic=True):
    """Forward pass through network."""
    # Encode
    if self.config.prior_type == "mixture":
        z_mean, z_logvar, component_logits = self.encoder(x, deterministic)
        component_probs = jax.nn.softmax(component_logits)
    else:
        z_mean, z_logvar = self.encoder(x, deterministic)
        component_probs = None

    # Sample z
    z = self._reparameterize(z_mean, z_logvar)

    # Decode (component-aware or standard)
    if self.config.component_aware_decoder:
        recon = self.decoder(
            z,
            component_probs=component_probs,
            deterministic=deterministic
        )
    else:
        recon = self.decoder(z, deterministic=deterministic)

    # ... rest of forward pass
```

---

## Tutorial 3: Implementing Latent-Only Classifier

### Goal
Replace separate classifier head with $\tau$-based classification using responsibilities.

### Steps

#### Step 1: Create Tau Map Module

Create `src/ssvae/components/tau_classifier.py`:

```python
import jax.numpy as jnp
from jax import Array
from typing import Dict, Tuple

class TauClassifier:
    """Latent-only classifier via responsibility-label map.

    Implements p(y|x) = sum_c q(c|x) * tau_{c,y}
    where tau is built from soft counts s_{c,y}.
    """

    def __init__(
        self,
        num_components: int,
        num_classes: int,
        alpha_0: float = 1.0  # Smoothing prior
    ):
        self.num_components = num_components
        self.num_classes = num_classes
        self.alpha_0 = alpha_0

        # Initialize soft counts
        self.s_cy = jnp.ones((num_components, num_classes)) * alpha_0

    def update_counts(
        self,
        component_probs: Array,  # q(c|x): (batch, K)
        labels: Array,           # y: (batch,)
        mask: Array              # labeled mask: (batch,)
    ):
        """Update soft counts from labeled data.

        Args:
            component_probs: Responsibilities q(c|x)
            labels: True labels (only for labeled samples)
            mask: Boolean mask for labeled samples
        """
        # Filter to labeled samples
        labeled_probs = component_probs[mask]  # (n_labeled, K)
        labeled_y = labels[mask]               # (n_labeled,)

        # Accumulate soft counts: s_{c,y} += sum_i q(c|x_i) * 1{y_i=y}
        for c in range(self.num_components):
            for y in range(self.num_classes):
                count = jnp.sum(
                    labeled_probs[:, c] * (labeled_y == y)
                )
                self.s_cy = self.s_cy.at[c, y].add(count)

    def get_tau(self) -> Array:
        """Compute tau_{c,y} from soft counts.

        Returns:
            tau: (num_components, num_classes) probability map
        """
        # Normalize: tau_{c,y} = s_{c,y} / sum_y' s_{c,y'}
        tau = self.s_cy / self.s_cy.sum(axis=1, keepdims=True)
        return tau

    def predict(self, component_probs: Array) -> Tuple[Array, Array]:
        """Predict labels from responsibilities.

        Args:
            component_probs: q(c|x) of shape (batch, K)

        Returns:
            predictions: Predicted class indices (batch,)
            probabilities: Class probabilities (batch, num_classes)
        """
        tau = self.get_tau()  # (K, num_classes)

        # p(y|x) = sum_c q(c|x) * tau_{c,y}
        # Shape: (batch, K) @ (K, num_classes) = (batch, num_classes)
        class_probs = component_probs @ tau

        predictions = jnp.argmax(class_probs, axis=-1)
        return predictions, class_probs

    def supervised_loss(
        self,
        component_probs: Array,
        labels: Array,
        mask: Array
    ) -> Array:
        """Compute supervised loss using stop-grad on tau.

        Loss: -log sum_c q(c|x) * tau_{c,y_true}

        Args:
            component_probs: q(c|x) of shape (batch, K)
            labels: True labels (batch,)
            mask: Labeled sample mask (batch,)
        """
        tau = jax.lax.stop_gradient(self.get_tau())  # Stop-grad!

        # Get p(y|x) for true labels
        # tau[c, y] indexed by labels
        tau_for_labels = tau[:, labels]  # (K, batch)
        prob_true_class = jnp.sum(
            component_probs * tau_for_labels.T,  # (batch, K)
            axis=-1
        )  # (batch,)

        # Negative log likelihood (only for labeled)
        nll = -jnp.log(prob_true_class + 1e-8)
        return jnp.where(mask, nll, 0.0).sum() / mask.sum()
```

#### Step 2: Integrate into SSVAE

Modify `src/ssvae/models.py`:

```python
class SSVAE:
    def __init__(self, input_dim, config):
        # ... existing initialization ...

        # Use tau classifier for mixture prior
        if config.prior_type == "mixture":
            self.tau_classifier = TauClassifier(
                num_components=config.num_components,
                num_classes=config.num_classes,
                alpha_0=config.get("tau_alpha_0", 1.0)
            )
        else:
            self.tau_classifier = None

    def fit(self, data, labels, weights_path, callbacks=None):
        """Training with tau updates."""
        # ... training loop ...

        # In each epoch, update tau from labeled data
        if self.tau_classifier is not None:
            _, _, component_logits = self.encoder.apply(...)
            component_probs = jax.nn.softmax(component_logits)

            labeled_mask = ~jnp.isnan(labels)
            self.tau_classifier.update_counts(
                component_probs, labels, labeled_mask
            )

    def predict(self, data, sample=False, num_samples=1):
        """Prediction using tau classifier."""
        z_mean, z_logvar, component_logits = self.encoder.apply(...)
        component_probs = jax.nn.softmax(component_logits)

        if self.tau_classifier is not None:
            predictions, class_probs = self.tau_classifier.predict(
                component_probs
            )
            # Certainty from max(r_c * max_y tau_{c,y})
            tau = self.tau_classifier.get_tau()
            certainty = jnp.max(
                component_probs[:, :, None] * tau[None, :, :],
                axis=(1, 2)
            )
        else:
            # Use standard classifier
            logits = self.classifier.apply(...)
            predictions = jnp.argmax(logits, axis=-1)
            class_probs = jax.nn.softmax(logits)
            certainty = jnp.max(class_probs, axis=-1)

        return z_mean, reconstruction, predictions, certainty
```

---

## Tutorial 4: Adding Custom Loss Terms

### Goal
Add a custom regularization term (e.g., contrastive loss, repulsion).

### Example: Channel Repulsion Loss

```python
# In src/training/losses.py

def channel_repulsion_loss(
    channel_embeddings: Array,  # (K, embedding_dim)
    repulsion_weight: float = 0.1
) -> Array:
    """Encourage diverse channel embeddings.

    Penalizes high similarity between different channels.
    Uses cosine similarity as distance metric.
    """
    # Normalize embeddings
    normalized = channel_embeddings / jnp.linalg.norm(
        channel_embeddings, axis=-1, keepdims=True
    )

    # Compute pairwise cosine similarity
    # (K, emb_dim) @ (emb_dim, K) = (K, K)
    similarity = normalized @ normalized.T

    # Mask diagonal (self-similarity)
    mask = 1 - jnp.eye(similarity.shape[0])
    masked_sim = similarity * mask

    # Repulsion loss: penalize high similarity
    # Want all off-diagonal similarities close to 0
    repulsion = repulsion_weight * jnp.sum(masked_sim ** 2)

    return repulsion
```

**Usage in total loss:**

```python
def total_loss_with_repulsion(...):
    # Standard losses
    recon = reconstruction_loss(...)
    kl = kl_divergence(...)
    classification = classification_loss(...)

    # Add repulsion
    if config.component_aware_decoder:
        channel_emb = get_channel_embeddings(decoder_params)
        repulsion = channel_repulsion_loss(
            channel_emb,
            repulsion_weight=config.repulsion_weight
        )
    else:
        repulsion = 0.0

    total = recon + kl + classification + repulsion
    return total, {
        'reconstruction': recon,
        'kl': kl,
        'classification': classification,
        'repulsion': repulsion,
        'total': total
    }
```

---

## General Extension Guidelines

### Adding New Components

1. **Create module** in appropriate directory (`src/ssvae/components/`, `src/ssvae/priors/`)
2. **Implement interface** (Protocol if applicable)
3. **Update factory** to create new component
4. **Add configuration** parameters to `SSVAEConfig`
5. **Write tests** to verify component behavior
6. **Update documentation** (this file!)

### Testing New Features

Always add tests for new features:

```python
# tests/test_my_feature.py
def test_my_new_component():
    """Test that new component works correctly."""
    # Setup
    config = SSVAEConfig(my_new_param=True)
    model = SSVAE(input_dim=(28, 28), config=config)

    # Test
    result = model.my_new_method(...)

    # Assertions
    assert result.shape == expected_shape
    assert not jnp.isnan(result).any()
```

### Documentation Updates

When adding features, update:
- This file (extending.md) with tutorial
- [Implementation Guide](implementation.md) with module reference
- [System Architecture](architecture.md) if design patterns change
- [Implementation Roadmap](../theory/implementation_roadmap.md) if moving toward RCM-VAE

---

## Related Documentation

- **[System Architecture](architecture.md)** - Design patterns and principles
- **[Implementation Guide](implementation.md)** - Module reference
- **[Mathematical Specification](../theory/mathematical_specification.md)** - Theoretical foundations
- **[Implementation Roadmap](../theory/implementation_roadmap.md)** - Path to full RCM-VAE
