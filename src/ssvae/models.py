from __future__ import annotations
"""SSVAE models module (JAX)."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ssvae.config import SSVAEConfig
from ssvae.components.factory import build_classifier, build_decoder, build_encoder, get_architecture_dims
from callbacks import CSVExporter, ConsoleLogger, LossCurvePlotter, TrainingCallback
from training.losses import compute_loss_and_metrics
from training.train_state import SSVAETrainState
from training.trainer import Trainer

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    from jax.nn import softmax
    import optax
    from flax import linen as nn
    from flax import traverse_util
    from flax.core import freeze, FrozenDict
    from flax.serialization import from_bytes, to_bytes
except Exception as e:  # pragma: no cover
    raise ImportError(
        "ssvae requires JAX, Flax, and Optax. Please install jax/jaxlib, flax, and optax."
    ) from e


def _make_weight_decay_mask(params: Dict[str, Dict[str, jnp.ndarray]]):
    """Create a mask matching ``params`` tree type for weight decay.

    Optax's masked wrappers require mask and params to share the same
    pytree structure and node types (e.g., dict vs FrozenDict). Build the
    mask and wrap it to match the container type of ``params``.
    """
    flat_params = traverse_util.flatten_dict(params)
    mask = {}
    for key in flat_params:
        param_name = key[-1]
        mask[key] = param_name not in ("bias", "scale")
    unflat = traverse_util.unflatten_dict(mask)
    if isinstance(params, FrozenDict):
        return freeze(unflat)
    return unflat


class SSVAENetwork(nn.Module):
    config: SSVAEConfig
    input_hw: Tuple[int, int]
    encoder_hidden_dims: Tuple[int, ...]
    decoder_hidden_dims: Tuple[int, ...]
    classifier_hidden_dims: Tuple[int, ...]
    classifier_dropout_rate: float
    latent_dim: int
    output_hw: Tuple[int, int]
    encoder_type: str
    decoder_type: str
    classifier_type: str

    def setup(self):
        self.encoder = build_encoder(self.config, input_hw=self.input_hw)
        self.decoder = build_decoder(self.config, input_hw=self.input_hw)
        self.classifier = build_classifier(self.config, input_hw=self.input_hw)

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        training: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        z_mean, z_log, z = self.encoder(x, training=training)
        recon = self.decoder(z)
        logits = self.classifier(z, training=training)
        return z_mean, z_log, z, recon, logits


class SSVAE:
    """
    Modular JAX SSVAE with a stable public API used by use_cases/scripts/train.py and use_cases/scripts/infer.py.

    Methods:
    - prepare_data_for_keras_model(data)
    - load_model_weights(path)
    - predict(data)
    - fit(data, labels, weights_path)
    """

    def __init__(self, input_dim: Tuple[int, int], config: SSVAEConfig | None = None):
        self.input_dim = input_dim
        self.config = config or SSVAEConfig()
        self.latent_dim = self.config.latent_dim
        self.weights_path: str | None = None

        self._out_hw = (input_dim[0], input_dim[1])
        if self.config.input_hw is None:
            self.config.input_hw = self._out_hw

        enc_dims, dec_dims, clf_dims = get_architecture_dims(self.config, input_hw=self._out_hw)
        self.model = SSVAENetwork(
            config=self.config,
            input_hw=self._out_hw,
            encoder_hidden_dims=enc_dims,
            decoder_hidden_dims=dec_dims,
            classifier_hidden_dims=clf_dims,
            classifier_dropout_rate=self.config.dropout_rate,
            latent_dim=self.latent_dim,
            output_hw=self._out_hw,
            encoder_type=self.config.encoder_type,
            decoder_type=self.config.decoder_type,
            classifier_type=self.config.classifier_type,
        )
        self._rng = random.PRNGKey(self.config.random_seed)
        params_key, sample_key, dropout_key, self._rng = random.split(self._rng, 4)
        dummy_input = jnp.zeros((1, *self._out_hw), dtype=jnp.float32)
        variables = self.model.init(
            {"params": params_key, "reparam": sample_key, "dropout": dropout_key},
            dummy_input,
            training=True,
        )
        decay_mask = _make_weight_decay_mask(variables["params"])

        opt_core = (
            optax.adamw(
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                mask=decay_mask,
            )
            if self.config.weight_decay > 0.0
            else optax.adam(self.config.learning_rate)
        )
        tx_steps = []
        if self.config.grad_clip_norm is not None:
            tx_steps.append(optax.clip_by_global_norm(self.config.grad_clip_norm))
        tx_steps.append(opt_core)
        self.tx = optax.chain(*tx_steps) if len(tx_steps) > 1 else tx_steps[0]

        self.state = SSVAETrainState.create(
            apply_fn=self.model.apply,
            params=variables["params"],
            tx=self.tx,
            rng=self._rng,
        )
        self._shuffle_rng = random.PRNGKey(self.config.random_seed + 1)

        cfg = self.config
        model_apply = self.state.apply_fn

        def apply_fn(params, *args, **kwargs):
            return model_apply({"params": params}, *args, **kwargs)
        self._apply_fn = apply_fn

        def _model_forward(
            params: Dict[str, Dict[str, jnp.ndarray]],
            batch_x: jnp.ndarray,
            *,
            training: bool,
            key: jax.Array | None,
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            if key is None:
                return apply_fn(params, batch_x, training=training)
            reparam_key, dropout_key = random.split(key)
            return apply_fn(
                params,
                batch_x,
                training=training,
                rngs={"reparam": reparam_key, "dropout": dropout_key},
            )

        def _loss_and_metrics(
            params: Dict[str, Dict[str, jnp.ndarray]],
            batch_x: jnp.ndarray,
            batch_y: jnp.ndarray,
            key: jax.Array | None,
            training: bool,
        ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
            rng = key if training else None
            return compute_loss_and_metrics(
                params,
                batch_x,
                batch_y,
                _model_forward,
                cfg,
                rng,
                training=training,
            )

        train_loss_and_grad = jax.value_and_grad(_train_loss_fn := (lambda p, x, y, k: _loss_and_metrics(p, x, y, k, True)), argnums=0, has_aux=True)

        @jax.jit
        def train_step(
            state: SSVAETrainState,
            batch_x: jnp.ndarray,
            batch_y: jnp.ndarray,
            key: jax.Array,
        ):
            (loss, metrics), grads = train_loss_and_grad(state.params, batch_x, batch_y, key)
            new_state = state.apply_gradients(grads=grads)
            return new_state, metrics

        self._train_step = train_step
        self._eval_metrics = jax.jit(lambda p, x, y: _loss_and_metrics(p, x, y, None, False)[1])

    def prepare_data_for_keras_model(self, data: np.ndarray) -> np.ndarray:
        return np.where(data == 0, 0.0, 1.0)

    def _save_weights(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "params": self.state.params,
            "opt_state": self.state.opt_state,
            "step": self.state.step,
        }
        data = to_bytes(payload)
        with open(path, "wb") as f:
            f.write(data)

    def _load_weights(self, path: str):
        with open(path, "rb") as f:
            payload_template = {
                "params": self.state.params,
                "opt_state": self.state.opt_state,
                "step": self.state.step,
            }
            payload = from_bytes(payload_template, f.read())
        self.state = self.state.replace(
            params=payload["params"],
            opt_state=payload["opt_state"],
            step=payload["step"],
        )

    def load_model_weights(self, weights_path: str):
        self.weights_path = str(weights_path)
        self._load_weights(self.weights_path)

    def _build_callbacks(self, *, weights_path: str | None, export_history: bool) -> List[TrainingCallback]:
        callbacks: List[TrainingCallback] = [ConsoleLogger()]
        if not export_history:
            return callbacks

        base_path = Path(weights_path) if weights_path else DEFAULT_CHECKPOINT_PATH
        history_path = base_path.with_name(f"{base_path.stem}_history.csv")
        plot_path = base_path.with_name(f"{base_path.stem}_loss.png")
        callbacks.append(CSVExporter(history_path))
        callbacks.append(LossCurvePlotter(plot_path))
        return callbacks

    def predict(
        self,
        data: np.ndarray,
        *,
        sample: bool = False,
        num_samples: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = jnp.array(data, dtype=jnp.float32)
        if sample:
            num_samples = max(1, int(num_samples))
            latent_samples = []
            recon_samples = []
            logits_samples = []
            for _ in range(num_samples):
                self._rng, subkey = random.split(self._rng)
                z_mean, _, z, recon, logits = self._apply_fn(
                    self.state.params,
                    x,
                    training=False,
                    rngs={"reparam": subkey},
                )
                latent_samples.append(z)
                recon_samples.append(recon)
                logits_samples.append(logits)
            latent_stack = jnp.stack(latent_samples) if num_samples > 1 else latent_samples[0]
            recon_stack = jnp.stack(recon_samples) if num_samples > 1 else recon_samples[0]
            logits_stack = jnp.stack(logits_samples) if num_samples > 1 else logits_samples[0]
            probs = softmax(logits_stack, axis=-1)
            if num_samples > 1:
                pred_class = jnp.argmax(probs, axis=-1)
                pred_certainty = jnp.max(probs, axis=-1)
            else:
                pred_class = jnp.argmax(probs, axis=1)
                pred_certainty = jnp.max(probs, axis=1)
            return (
                np.array(latent_stack),
                np.array(recon_stack),
                np.array(pred_class, dtype=np.int32),
                np.array(pred_certainty),
            )
        else:
            z_mean, _, _, recon, logits = self._apply_fn(self.state.params, x, training=False)
            probs = softmax(logits, axis=1)
            pred_class = jnp.argmax(probs, axis=1)
            pred_certainty = jnp.max(probs, axis=1)
            return (
                np.array(z_mean),
                np.array(recon),
                np.array(pred_class, dtype=np.int32),
                np.array(pred_certainty),
            )

    def fit(self, data: np.ndarray, labels: np.ndarray, weights_path: str):
        self.weights_path = str(weights_path)
        callbacks = self._build_callbacks(weights_path=self.weights_path, export_history=True)
        trainer = Trainer(self.config)
        self.state, self._shuffle_rng, history = trainer.train(
            self.state,
            data=data,
            labels=labels,
            weights_path=self.weights_path,
            shuffle_rng=self._shuffle_rng,
            train_step_fn=self._train_step,
            eval_metrics_fn=self._eval_metrics,
            save_fn=self._save_weights,
            callbacks=callbacks,
        )
        self._rng = self.state.rng
        return history


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "artifacts" / "checkpoints" / "ssvae.ckpt"
