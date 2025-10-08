from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from configs.base import SSVAEConfig
from models.classifier import Classifier
from models.decoders import DenseDecoder as Decoder
from models.encoders import DenseEncoder as Encoder
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
except Exception as e:  # pragma: no cover - import-time guard for environments without JAX/Flax
    raise ImportError(
        "vae_model_jax requires JAX, Flax, and Optax. Please install jax/jaxlib, flax, and optax to use this backend."
    ) from e


def _make_weight_decay_mask(params: Dict[str, Dict[str, jnp.ndarray]]) -> Dict[str, Dict[str, bool]]:
    """Create a mask to exclude bias/scale parameters from weight decay."""
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

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        training: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        encoder = Encoder(self.encoder_hidden_dims, self.latent_dim)
        decoder = Decoder(self.decoder_hidden_dims, self.output_hw)
        classifier = Classifier(self.classifier_hidden_dims)
        z_mean, z_log, z = encoder(x, training=training)
        recon = decoder(z)
        logits = classifier(z)
        return z_mean, z_log, z, recon, logits


class SSVAE:
    """
    JAX implementation of the TF SSVAE with an API compatible with scripts/train_tf.py and scripts/infer_tf.py.

    Methods kept compatible:
    - prepare_data_for_keras_model(data) -> np.ndarray
    - load_model_weights(weights_path)
    - predict(data) -> (latent, recon, pred_class, pred_certainty)
    - fit(data, labels, weights_path)

    Hyperparameters are configurable via SSVAEConfig while preserving defaults that
    match the TensorFlow reference implementation.
    """

    def __init__(self, input_dim: Tuple[int, int], config: SSVAEConfig | None = None):
        self.input_dim = input_dim
        self.config = config or SSVAEConfig()
        self.latent_dim = self.config.latent_dim
        self.weights_path: str | None = None

        self._out_hw = (input_dim[0], input_dim[1])

        flat = self._out_hw[0] * self._out_hw[1]
        encoder_hidden_dims = self.config.hidden_dims or (flat,)
        decoder_hidden_dims = tuple(reversed(encoder_hidden_dims)) or (self.latent_dim,)
        last_hidden = encoder_hidden_dims[-1] if encoder_hidden_dims else self.latent_dim
        classifier_hidden_dims = (last_hidden, last_hidden)  # mirrors TF classifier (two hidden layers)

        self.model = SSVAENetwork(
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            classifier_hidden_dims=classifier_hidden_dims,
            latent_dim=self.latent_dim,
            output_hw=self._out_hw,
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

        def _train_loss_fn(
            params: Dict[str, Dict[str, jnp.ndarray]],
            batch_x: jnp.ndarray,
            batch_y: jnp.ndarray,
            key: jax.Array,
        ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
            return _loss_and_metrics(params, batch_x, batch_y, key, training=True)

        def _eval_metrics_fn(
            params: Dict[str, Dict[str, jnp.ndarray]],
            batch_x: jnp.ndarray,
            batch_y: jnp.ndarray,
        ) -> Dict[str, jnp.ndarray]:
            return _loss_and_metrics(params, batch_x, batch_y, None, training=False)[1]

        train_loss_and_grad = jax.value_and_grad(_train_loss_fn, argnums=0, has_aux=True)

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
        self._eval_metrics = jax.jit(_eval_metrics_fn)

    # Data prep kept for compatibility
    def prepare_data_for_keras_model(self, data: np.ndarray) -> np.ndarray:
        """Binarize the inputs (0/1) to mirror the TensorFlow preprocessing."""
        return np.where(data == 0, 0.0, 1.0)

    # ------------
    # IO helpers
    # ------------
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
        print(
            f"Epoch {epoch+1:03d} "
            f"loss={float(train_metrics['loss']):.4f} val_loss={float(val_metrics['loss']):.4f} "
            f"rec={float(train_metrics['reconstruction_loss']):.4f} "
            f"kl={float(train_metrics['kl_loss']):.4f} "
            f"cls={float(train_metrics['classification_loss']):.4f}",
            flush=True,
        )

    def load_model_weights(self, weights_path: str):
        self.weights_path = str(weights_path)
        self._load_weights(self.weights_path)

    def _export_history(self, history: Dict[str, list[float]]):
        try:
            PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
            csv_path = PROGRESS_DIR / "ssvae_jax_history.csv"
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
            # Non-fatal: keep training if plotting/filesystem fail
            pass

    # --------
    # Inference
    # --------
    def predict(
        self,
        data: np.ndarray,
        *,
        sample: bool = False,
        num_samples: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict latent codes, reconstructions, and classifier outputs.

        By default this mirrors the TensorFlow implementation and uses the latent
        mean deterministically. Set `sample=True` to draw from the latent posterior.
        """
        x = jnp.array(data, dtype=jnp.float32)
        if sample:
            num_samples = max(1, int(num_samples))
            latent_samples = []
            recon_samples = []
            logits_samples = []
            for _ in range(num_samples):
                self._rng, subkey = random.split(self._rng)
                # Sampling path mirrors training-time reparameterization instead of the TF
                # public inference path, which always uses the posterior mean.
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
            # Deterministic inference that matches TensorFlow's default behavior.
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

    # -----
    # Train
    # -----
    def fit(self, data: np.ndarray, labels: np.ndarray, weights_path: str):
        """Train the model with TensorFlow-parity loss aggregation (classification averaged over labeled examples only)."""
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
    """
    Placeholder SSCVAE (convolutional variant) implemented as a fallback to the dense SSVAE
    to keep the API surface identical. For MNIST training/inference in this repo, `SSVAE` is used.

    If you need true conv/transpose-conv in JAX, we can extend this in a follow-up.
    """

    def __init__(self, input_dim: Tuple[int, int, int] = (28, 28, 1), config: SSVAEConfig | None = None):
        super().__init__(input_dim=input_dim[:2], config=config)


BASE_DIR = Path(__file__).resolve().parents[1]
PROGRESS_DIR = BASE_DIR / "models" / "progress"
PROGRESS_PLOT_PATH = PROGRESS_DIR / "ssvae_loss_plot.png"

try:  # optional plotting
    import matplotlib.pyplot as plt  # type: ignore
    _HAS_PLT = True
except Exception:  # pragma: no cover
    _HAS_PLT = False
