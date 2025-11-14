"""Unit tests for mixture encoder implementation."""
from __future__ import annotations

import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from callbacks.mixture_tracking import MixtureHistoryTracker
from model.ssvae import SSVAE
from model.ssvae.components.encoders import DenseEncoder, MixtureConvEncoder, MixtureDenseEncoder
sys.path.append(str(Path(__file__).resolve().parents[1]))
from use_cases.experiments.src.visualization.plotters import _extract_component_recon


def test_mixture_encoder_output_shapes():
    """Verify MixtureDenseEncoder returns correct output shapes."""
    batch_size = 32
    K = 5
    latent_dim = 2
    hidden_dims = (128, 64)
    
    encoder = MixtureDenseEncoder(
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        num_components=K,
    )
    
    rng = random.PRNGKey(0)
    params_key, reparam_key = random.split(rng)
    dummy_input = jnp.zeros((batch_size, 28, 28), dtype=jnp.float32)
    
    variables = encoder.init(
        {"params": params_key, "reparam": reparam_key},
        dummy_input,
        training=True,
    )
    
    component_logits, z_mean, z_log, z = encoder.apply(
        variables,
        dummy_input,
        training=True,
        rngs={"reparam": reparam_key},
    )
    
    assert component_logits.shape == (batch_size, K)
    assert z_mean.shape == (batch_size, latent_dim)
    assert z_log.shape == (batch_size, latent_dim)
    assert z.shape == (batch_size, latent_dim)


def test_mixture_encoder_responsibilities_sum_to_one():
    """Verify responsibilities from component_logits sum to 1.0."""
    batch_size = 16
    K = 10
    
    encoder = MixtureDenseEncoder(
        hidden_dims=(64,),
        latent_dim=2,
        num_components=K,
    )
    
    rng = random.PRNGKey(42)
    params_key, reparam_key = random.split(rng)
    dummy_input = jnp.zeros((batch_size, 28, 28), dtype=jnp.float32)
    
    variables = encoder.init(
        {"params": params_key, "reparam": reparam_key},
        dummy_input,
        training=True,
    )
    
    component_logits, _, _, _ = encoder.apply(
        variables,
        dummy_input,
        training=False,
    )
    
    responsibilities = jax.nn.softmax(component_logits, axis=-1)
    resp_sums = jnp.sum(responsibilities, axis=-1)
    
    assert jnp.allclose(resp_sums, 1.0, atol=1e-6)


def test_standard_encoder_unchanged():
    """Verify DenseEncoder still returns 3-tuple format."""
    batch_size = 8
    latent_dim = 2
    
    encoder = DenseEncoder(
        hidden_dims=(128, 64),
        latent_dim=latent_dim,
    )
    
    rng = random.PRNGKey(0)
    params_key, reparam_key = random.split(rng)
    dummy_input = jnp.zeros((batch_size, 28, 28), dtype=jnp.float32)
    
    variables = encoder.init(
        {"params": params_key, "reparam": reparam_key},
        dummy_input,
        training=True,
    )
    
    output = encoder.apply(
        variables,
        dummy_input,
        training=True,
        rngs={"reparam": reparam_key},
    )
    
    assert len(output) == 3
    z_mean, z_log, z = output
    assert z_mean.shape == (batch_size, latent_dim)
    assert z_log.shape == (batch_size, latent_dim)
    assert z.shape == (batch_size, latent_dim)


def test_mixture_encoder_without_reparam_rng():
    """Verify mixture encoder works without reparam RNG (returns z_mean)."""
    batch_size = 4
    
    encoder = MixtureDenseEncoder(
        hidden_dims=(64,),
        latent_dim=2,
        num_components=3,
    )
    
    rng = random.PRNGKey(0)
    dummy_input = jnp.zeros((batch_size, 28, 28), dtype=jnp.float32)
    
    variables = encoder.init(
        {"params": rng},
        dummy_input,
        training=False,
    )
    
    component_logits, z_mean, z_log, z = encoder.apply(
        variables,
        dummy_input,
        training=False,
    )
    
    # Without reparam RNG, z should equal z_mean
    assert jnp.allclose(z, z_mean, atol=1e-6)


def test_mixture_conv_encoder_output_shapes():
    """MixtureConvEncoder should emit logits and latent stats with correct shapes."""
    batch_size = 16
    num_components = 4
    latent_dim = 3

    encoder = MixtureConvEncoder(latent_dim=latent_dim, num_components=num_components)

    rng = random.PRNGKey(7)
    params_key, reparam_key = random.split(rng)
    dummy_input = jnp.zeros((batch_size, 28, 28), dtype=jnp.float32)

    variables = encoder.init(
        {"params": params_key, "reparam": reparam_key},
        dummy_input,
        training=True,
    )

    component_logits, z_mean, z_log, z = encoder.apply(
        variables,
        dummy_input,
        training=True,
        rngs={"reparam": reparam_key},
    )

    assert component_logits.shape == (batch_size, num_components)
    assert z_mean.shape == (batch_size, latent_dim)
    assert z_log.shape == (batch_size, latent_dim)
    assert z.shape == (batch_size, latent_dim)


def test_mixture_conv_encoder_softmax_normalized():
    """Responsibilities derived from conv logits should sum to 1."""
    batch_size = 5
    num_components = 6
    encoder = MixtureConvEncoder(latent_dim=2, num_components=num_components)

    rng = random.PRNGKey(123)
    dummy_input = jnp.zeros((batch_size, 28, 28), dtype=jnp.float32)
    variables = encoder.init({"params": rng}, dummy_input, training=False)

    component_logits, _, _, _ = encoder.apply(
        variables,
        dummy_input,
        training=False,
    )

    responsibilities = jax.nn.softmax(component_logits, axis=-1)
    sums = responsibilities.sum(axis=-1)

    assert jnp.allclose(sums, 1.0, atol=1e-6)


class _DummyState:
    """Minimal stand-in for trainer state to record apply_fn batch sizes."""

    def __init__(self):
        self.params = {}
        self.batch_sizes: list[int] = []

    def apply_fn(self, params, batch, training):
        del params, training
        self.batch_sizes.append(batch.shape[0])
        extras = {
            "responsibilities": jnp.ones((batch.shape[0], 2), dtype=jnp.float32) / 2.0,
            "pi": jnp.array([0.6, 0.4], dtype=jnp.float32),
        }
        output = SimpleNamespace(extras=extras)
        return output


def test_mixture_history_tracker_batches_inputs(tmp_path):
    """MixtureHistoryTracker should chunk inputs to avoid oversized GPU batches."""
    tracker = MixtureHistoryTracker(
        tmp_path / "diag",
        log_every=1,
        max_samples=8,
        eval_batch_size=3,
    )
    tracker.on_train_start(trainer=None)

    x_train = np.random.rand(10, 28, 28).astype(np.float32)
    splits = SimpleNamespace(
        x_train=x_train,
        y_train=np.zeros(10, dtype=np.float32),
    )
    trainer = SimpleNamespace(
        latest_splits=splits,
        _current_state=_DummyState(),
    )

    tracker.on_epoch_end(epoch=0, metrics={}, history={}, trainer=trainer)

    assert trainer._current_state.batch_sizes == [3, 3, 2]
    assert len(tracker.tracked_epochs) == 1
    assert len(tracker.usage_history) == 1
    assert len(tracker.pi_history) == 1


def test_predict_batched_handles_heteroscedastic_outputs():
    """predict_batched should concatenate heteroscedastic recon tuples batch-wise."""
    dummy_model = object.__new__(SSVAE)
    batches = [
        (
            np.zeros((3, 2), dtype=np.float32),
            (np.zeros((3, 4, 4), dtype=np.float32), np.full((3,), 0.1, dtype=np.float32)),
            np.zeros(3, dtype=np.int32),
            np.zeros(3, dtype=np.float32),
        ),
        (
            np.ones((2, 2), dtype=np.float32),
            (np.ones((2, 4, 4), dtype=np.float32), np.full((2,), 0.2, dtype=np.float32)),
            np.ones(2, dtype=np.int32),
            np.ones(2, dtype=np.float32),
        ),
    ]
    call_idx = {"value": 0}

    def fake_predict(self, batch, *, sample=False, num_samples=1, return_mixture=False):
        assert not sample and not return_mixture
        out = batches[call_idx["value"]]
        call_idx["value"] += 1
        return out

    dummy_model.predict = MethodType(fake_predict, dummy_model)

    data = np.zeros((5, 4, 4), dtype=np.float32)
    latent, recon, preds, cert = SSVAE.predict_batched(dummy_model, data, batch_size=3)

    assert latent.shape == (5, 2)
    assert isinstance(recon, tuple)
    mean, sigma = recon
    assert mean.shape == (5, 4, 4)
    assert sigma.shape == (5,)
    assert np.allclose(sigma[:3], 0.1) and np.allclose(sigma[3:], 0.2)
    assert preds.shape == (5,)
    assert cert.shape == (5,)


def test_extract_component_recon_handles_heteroscedastic():
    mean = np.ones((1, 2, 4, 4), dtype=np.float32)
    sigma = np.full((1, 2), 0.3, dtype=np.float32)
    extras = {"recon_per_component": (mean, sigma)}
    recon, sigma_out, err = _extract_component_recon(extras)
    assert err is None
    assert recon.shape == (2, 4, 4)
    assert sigma_out.shape == (2,)
    assert np.allclose(sigma_out, 0.3)

    extras_bad = {"recon_per_component": np.zeros((2, 2, 4, 4))}
    recon_bad, _, err_bad = _extract_component_recon(extras_bad)
    assert recon_bad is None
    assert err_bad is not None
