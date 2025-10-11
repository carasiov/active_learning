from __future__ import annotations
"""SSVAE models module (JAX)."""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from configs.base import SSVAEConfig
from model_components.classifier import Classifier
from model_components.decoders import ConvDecoder, DenseDecoder
from model_components.encoders import ConvEncoder, DenseEncoder
from model_components.factory import get_architecture_dims
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
    from flax.core import freeze
    from flax.serialization import from_bytes, to_bytes
except Exception as e:  # pragma: no cover
    raise ImportError(
        "ssvae requires JAX, Flax, and Optax. Please install jax/jaxlib, flax, and optax."
    ) from e


def _make_weight_decay_mask(params: Dict[str, Dict[str, jnp.ndarray]]) -> Dict[str, Dict[str, bool]]:
    flat_params = traverse_util.flatten_dict(params)
    mask = {}
    for key in flat_params:
        param_name = key[-1]
        mask[key] = param_name not in ("bias", "scale")
    return freeze(traverse_util.unflatten_dict(mask))


class SSVAENetwork(nn.Module):
    encoder_hidden_dims: Tuple[int, ...]
    decoder_hidden_dims: Tuple[int, ...]
    classifier_hidden_dims: Tuple[int, ...]
    latent_dim: int
    output_hw: Tuple[int, int]
    encoder_type: str
    decoder_type: str
    classifier_type: str

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        training: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if self.encoder_type == "dense":
            encoder = DenseEncoder(hidden_dims=self.encoder_hidden_dims, latent_dim=self.latent_dim)
        elif self.encoder_type == "conv":
            encoder = ConvEncoder(latent_dim=self.latent_dim)
        else:
            raise ValueError(f"Unsupported encoder_type: {self.encoder_type}")

        if self.decoder_type == "dense":
            decoder = DenseDecoder(hidden_dims=self.decoder_hidden_dims, output_hw=self.output_hw)
        elif self.decoder_type == "conv":
            decoder = ConvDecoder(latent_dim=self.latent_dim, output_hw=self.output_hw)
        else:
            raise ValueError(f"Unsupported decoder_type: {self.decoder_type}")

        if self.classifier_type == "dense":
            classifier = Classifier(self.classifier_hidden_dims)
        else:
            raise ValueError(f"Unsupported classifier_type: {self.classifier_type}")

        z_mean, z_log, z = encoder(x, training=training)
        recon = decoder(z)
        logits = classifier(z)
        return z_mean, z_log, z, recon, logits


class SSVAE:
    """
    Modular JAX SSVAE with a stable public API used by scripts/train.py and scripts/infer.py.

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
            encoder_hidden_dims=enc_dims,
            decoder_hidden_dims=dec_dims,
            classifier_hidden_dims=clf_dims,
            latent_dim=self.latent_dim,
            output_hw=self._out_hw,
            encoder_type=self.config.encoder_type,
            decoder_type=self.config.decoder_type,
            classifier_type=self.config.classifier_type,
        )
        self._rng = random.PRNGKey(self.config.random_seed)
        params_key, sample_key, self._rng = random.split(self._rng, 3)
        dummy_input = jnp.zeros((1, *self._out_hw), dtype=jnp.float32)
        variables = self.model.init({"params": params_key, "reparam": sample_key}, dummy_input, training=True)
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
            return apply_fn(params, batch_x, training=training, rngs={"reparam": key})

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

    def _log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, jnp.ndarray],
        val_metrics: Dict[str, jnp.ndarray],
        history: Dict[str, list[float]],
    ) -> None:
        metric_columns = [
            ("Train.loss", train_metrics, "loss"),
            ("Val.loss", val_metrics, "loss"),
            ("Train.rec", train_metrics, "reconstruction_loss"),
            ("Val.rec", val_metrics, "reconstruction_loss"),
            ("Train.kl", train_metrics, "kl_loss"),
            ("Val.kl", val_metrics, "kl_loss"),
            ("Train.cls", train_metrics, "classification_loss"),
            ("Val.cls", val_metrics, "classification_loss"),
        ]
        if "contrastive_loss" in train_metrics and "contrastive_loss" in val_metrics:
            metric_columns.extend(
                [
                    ("Train.con", train_metrics, "contrastive_loss"),
                    ("Val.con", val_metrics, "contrastive_loss"),
                ]
            )

        header_parts = [f"{'Epoch':>5}"]
        row_parts = [f"{epoch+1:>5d}"]
        for label, source, key in metric_columns:
            header_parts.append(f"{label:>12}")
            row_parts.append(f"{float(source[key]):>12.4f}")

        if epoch == 0:
            header_line = " | ".join(header_parts)
            divider = "-" * len(header_line)
            print(header_line, flush=True)
            print(divider, flush=True)

        print(" | ".join(row_parts), flush=True)

    def load_model_weights(self, weights_path: str):
        self.weights_path = str(weights_path)
        self._load_weights(self.weights_path)

    def _export_history(self, history: Dict[str, list[float]]):
        try:
            PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
            csv_path = PROGRESS_DIR / "ssvae_history.csv"
            headers = [
                "epoch",
                "loss",
                "val_loss",
                "reconstruction_loss",
                "val_reconstruction_loss",
                "kl_loss",
                "val_kl_loss",
                "classification_loss",
                "val_classification_loss",
            ]
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(",".join(headers) + "\n")
                epochs = len(history["loss"])
                for i in range(epochs):
                    row = [
                        str(i + 1),
                        f"{history['loss'][i]:.8f}",
                        f"{history['val_loss'][i]:.8f}",
                        f"{history['reconstruction_loss'][i]:.8f}",
                        f"{history['val_reconstruction_loss'][i]:.8f}",
                        f"{history['kl_loss'][i]:.8f}",
                        f"{history['val_kl_loss'][i]:.8f}",
                        f"{history['classification_loss'][i]:.8f}",
                        f"{history['val_classification_loss'][i]:.8f}",
                    ]
                    f.write(",".join(row) + "\n")

            if _HAS_PLT:
                fig, axes = plt.subplots(2, 1, figsize=(10, 8))
                axes[0].plot(history["loss"], label="Training Loss")
                axes[0].plot(history["val_loss"], label="Validation Loss")
                axes[0].set_xlabel("Epochs")
                axes[0].set_ylabel("Loss")
                axes[0].set_title("Loss and Validation Loss")
                axes[0].legend()
                axes[0].grid(True)

                axes[1].plot(history["reconstruction_loss"], label="Reconstruction", color="blue")
                axes[1].plot(history["val_reconstruction_loss"], label="Val Reconstruction", color="cyan")
                axes[1].plot(history["kl_loss"], label="KL", color="red")
                axes[1].plot(history["val_kl_loss"], label="Val KL", color="orange")
                axes[1].plot(history["classification_loss"], label="Classification", color="green")
                axes[1].plot(history["val_classification_loss"], label="Val Classification", color="lime")
                axes[1].set_xlabel("Epochs")
                axes[1].set_ylabel("Loss")
                axes[1].set_title("Component Losses")
                axes[1].legend()
                axes[1].grid(True)
                fig.tight_layout()
                fig.savefig(PROGRESS_PLOT_PATH)
                plt.close(fig)
        except Exception:
            pass

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
        trainer = Trainer(self.config)
        self.state, self._shuffle_rng, history = trainer.train(
            self.state,
            data=data,
            labels=labels,
            weights_path=self.weights_path,
            shuffle_rng=self._shuffle_rng,
            train_step_fn=self._train_step,
            eval_metrics_fn=self._eval_metrics,
            log_fn=self._log_epoch_metrics,
            save_fn=self._save_weights,
            export_history_fn=self._export_history,
        )
        self._rng = self.state.rng
        return history


class SSCVAE(SSVAE):
    def __init__(self, input_dim: Tuple[int, int, int] = (28, 28, 1), config: SSVAEConfig | None = None):
        super().__init__(input_dim=input_dim[:2], config=config)


BASE_DIR = Path(__file__).resolve().parents[1]
PROGRESS_DIR = BASE_DIR / "artifacts" / "progress"
PROGRESS_PLOT_PATH = PROGRESS_DIR / "ssvae_loss_plot.png"

try:  # optional plotting
    import matplotlib.pyplot as plt  # type: ignore
    _HAS_PLT = True
except Exception:  # pragma: no cover
    _HAS_PLT = False
