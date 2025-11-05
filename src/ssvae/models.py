from __future__ import annotations
"""SSVAE models module (JAX)."""

from pathlib import Path

from typing import Dict, List, NamedTuple, Optional, Tuple

from utils import configure_jax_device, print_device_banner

configure_jax_device()

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


class ForwardOutput(NamedTuple):
    component_logits: Optional[jnp.ndarray]
    z_mean: jnp.ndarray
    z_log: jnp.ndarray
    z: jnp.ndarray
    recon: jnp.ndarray
    class_logits: jnp.ndarray
    extras: Dict[str, jnp.ndarray]


class MixturePriorParameters(nn.Module):
    """Container for learnable mixture prior parameters."""

    num_components: int
    embed_dim: int

    @nn.compact
    def __call__(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        embeddings = self.param(
            "component_embeddings",
            nn.initializers.normal(stddev=0.02),
            (self.num_components, self.embed_dim),
        )
        pi_logits = self.param(
            "pi_logits",
            nn.initializers.zeros,
            (self.num_components,),
        )
        return embeddings, pi_logits


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
        apply_decay = param_name not in ("bias", "scale")
        if "prior" in key and param_name == "pi_logits":
            apply_decay = False
        mask[key] = apply_decay
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
        self.prior_module: MixturePriorParameters | None = None
        if self.config.prior_type == "mixture":
            self.prior_module = MixturePriorParameters(
                name="prior",
                num_components=self.config.num_components,
                embed_dim=self.latent_dim,
            )

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        training: bool,
    ) -> ForwardOutput:
        """Forward pass returning latent statistics, reconstructions, and classifier logits."""
        encoder_output = self.encoder(x, training=training)
        extras: Dict[str, jnp.ndarray] = {}

        if self.config.prior_type == "mixture":
            component_logits, z_mean, z_log, z = encoder_output
            if self.prior_module is None:
                raise ValueError("Mixture prior selected but prior_module was not initialized.")
            embeddings, pi_logits = self.prior_module()
            responsibilities = softmax(component_logits, axis=-1)
            pi = softmax(pi_logits, axis=-1)

            batch_size = z.shape[0]
            num_components = self.config.num_components

            z_tiled = jnp.broadcast_to(z[:, None, :], (batch_size, num_components, self.latent_dim))
            embed_tiled = jnp.broadcast_to(embeddings[None, :, :], (batch_size, num_components, embeddings.shape[-1]))
            decoder_inputs = jnp.concatenate([z_tiled, embed_tiled], axis=-1)
            decoder_inputs_flat = decoder_inputs.reshape((batch_size * num_components, -1))
            recon_per_component_flat = self.decoder(decoder_inputs_flat)
            recon_per_component = recon_per_component_flat.reshape(
                (batch_size, num_components, *self.output_hw)
            )
            expected_recon = jnp.sum(
                responsibilities[..., None, None] * recon_per_component,
                axis=1,
            )
            recon = expected_recon
            extras = {
                "recon_per_component": recon_per_component,
                "responsibilities": responsibilities,
                "pi_logits": pi_logits,
                "pi": pi,
                "component_embeddings": embeddings,
            }
        else:
            z_mean, z_log, z = encoder_output
            component_logits = None
            recon = self.decoder(z)

        logits = self.classifier(z, training=training)
        return ForwardOutput(component_logits, z_mean, z_log, z, recon, logits, extras)


class SSVAE:
    """
    Modular JAX SSVAE with a stable public API used by use_cases/scripts/train.py and use_cases/scripts/infer.py.

    Methods:
    - prepare_data_for_keras_model(data)
    - load_model_weights(path)
    - predict(data)
    - fit(data, labels, weights_path)
    """

    _DEVICE_BANNER_PRINTED = False

    def __init__(self, input_dim: Tuple[int, int], config: SSVAEConfig | None = None):
        self.input_dim = input_dim
        self.config = config or SSVAEConfig()
        self.latent_dim = self.config.latent_dim
        self.weights_path: str | None = None
        self._last_diagnostics_dir: Path | None = None

        if not SSVAE._DEVICE_BANNER_PRINTED:
            print_device_banner()
            SSVAE._DEVICE_BANNER_PRINTED = True

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
        ) -> ForwardOutput:
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
            kl_c_scale: float,
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
                kl_c_scale=kl_c_scale,
            )

        train_loss_and_grad = jax.value_and_grad(
            lambda p, x, y, k, scale: _loss_and_metrics(p, x, y, k, True, scale),
            argnums=0,
            has_aux=True,
        )

        @jax.jit
        def train_step(
            state: SSVAETrainState,
            batch_x: jnp.ndarray,
            batch_y: jnp.ndarray,
            key: jax.Array,
            kl_c_scale: float,
        ):
            (loss, metrics), grads = train_loss_and_grad(state.params, batch_x, batch_y, key, kl_c_scale)
            new_state = state.apply_gradients(grads=grads)
            return new_state, metrics

        self._train_step = train_step
        self._eval_metrics = jax.jit(lambda p, x, y: _loss_and_metrics(p, x, y, None, False, 1.0)[1])

    def prepare_data_for_keras_model(self, data: np.ndarray) -> np.ndarray:
        return np.where(data == 0, 0.0, 1.0)

    def _save_weights(self, state: SSVAETrainState, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "params": state.params,
            "opt_state": state.opt_state,
            "step": state.step,
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
        """Load params and optimizer state from checkpoint."""
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

    def _diagnostics_output_dir(self) -> Path:
        base = Path(self.weights_path) if self.weights_path else DEFAULT_CHECKPOINT_PATH
        diag_dir = base.parent / "diagnostics" / base.stem
        diag_dir.mkdir(parents=True, exist_ok=True)
        self._last_diagnostics_dir = diag_dir
        return diag_dir

    def _save_mixture_diagnostics(self, splits: Trainer.DataSplits | None) -> None:
        if self.config.prior_type != "mixture" or splits is None:
            return

        val_x = np.asarray(splits.x_val)
        val_y = np.asarray(splits.y_val)
        if val_x.size == 0:
            val_x = np.asarray(splits.x_train)
            val_y = np.asarray(splits.y_train)
        if val_x.size == 0:
            return

        batch_size = min(self.config.batch_size, 1024)
        eps = 1e-8

        usage_sum: np.ndarray | None = None
        entropy_sum = 0.0
        count = 0
        z_records: List[np.ndarray] = []
        resp_records: List[np.ndarray] = []
        label_records: List[np.ndarray] = []
        pi_array: np.ndarray | None = None

        total = val_x.shape[0]
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_inputs = jnp.asarray(val_x[start:end])
            forward = self._apply_fn(self.state.params, batch_inputs, training=False)
            component_logits, z_mean, _, _, _, _, extras = forward
            responsibilities = extras.get("responsibilities") if hasattr(extras, "get") else None
            if responsibilities is None:
                return
            resp_np = np.asarray(responsibilities)
            if usage_sum is None:
                usage_sum = np.zeros(resp_np.shape[1], dtype=np.float64)
            usage_sum += resp_np.sum(axis=0)
            entropy_batch = -resp_np * np.log(resp_np + eps)
            entropy_sum += entropy_batch.sum()
            count += resp_np.shape[0]
            z_records.append(np.asarray(z_mean))
            resp_records.append(resp_np)
            label_records.append(val_y[start:end])
            if pi_array is None:
                pi_val = extras.get("pi") if hasattr(extras, "get") else None
                if pi_val is not None:
                    pi_array = np.asarray(pi_val)

        if usage_sum is None or count == 0:
            return

        component_usage = (usage_sum / count).astype(np.float32)
        component_entropy = np.array(entropy_sum / count, dtype=np.float32)

        diag_dir = self._diagnostics_output_dir()
        np.save(diag_dir / "component_usage.npy", component_usage)
        np.save(diag_dir / "component_entropy.npy", component_entropy)
        if pi_array is not None:
            np.save(diag_dir / "pi.npy", pi_array.astype(np.float32))

        if self.latent_dim == 2 and z_records:
            z_array = np.concatenate(z_records, axis=0).astype(np.float32)
            resp_array = np.concatenate(resp_records, axis=0).astype(np.float32)
            labels_array = np.concatenate(label_records, axis=0)
            np.savez(diag_dir / "latent.npz", z_mean=z_array, labels=labels_array, q_c=resp_array)

    def predict(
        self,
        data: np.ndarray,
        *,
        sample: bool = False,
        num_samples: int = 1,
        return_mixture: bool = False,
    ) -> Tuple:
        """Returns (latent, reconstruction, class_predictions, certainty).

        When ``return_mixture`` is True and the model uses a mixture prior, the tuple will
        include two additional entries: responsibilities ``q_c`` and the learned ``pi``.
        """
        x = jnp.array(data, dtype=jnp.float32)
        mixture_active = self.config.prior_type == "mixture"
        if return_mixture and not mixture_active:
            raise ValueError("return_mixture=True is only supported for mixture priors.")

        if sample:
            num_samples = max(1, int(num_samples))
            latent_samples = []
            recon_samples = []
            logits_samples = []
            resp_samples = []
            pi_value = None
            for _ in range(num_samples):
                self._rng, subkey = random.split(self._rng)
                forward = self._apply_fn(
                    self.state.params,
                    x,
                    training=False,
                    rngs={"reparam": subkey},
                )
                component_logits, z_mean, _, z, recon, logits, extras = forward
                latent_samples.append(z)
                recon_samples.append(recon)
                logits_samples.append(logits)
                if return_mixture:
                    responsibilities = extras.get("responsibilities") if hasattr(extras, "get") else None
                    if responsibilities is None:
                        raise ValueError("Mixture responsibilities unavailable during predict().")
                    resp_samples.append(responsibilities)
                    if pi_value is None:
                        pi_val = extras.get("pi") if hasattr(extras, "get") else None
                        if pi_val is not None:
                            pi_value = pi_val

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
            result = (
                np.array(latent_stack),
                np.array(recon_stack),
                np.array(pred_class, dtype=np.int32),
                np.array(pred_certainty),
            )
            if return_mixture:
                q_stack = jnp.stack(resp_samples) if num_samples > 1 else resp_samples[0]
                pi_np = np.array(pi_value) if pi_value is not None else None
                result += (np.array(q_stack), pi_np)
            return result
        else:
            forward = self._apply_fn(self.state.params, x, training=False)
            component_logits, z_mean, _, _, recon, logits, extras = forward
            probs = softmax(logits, axis=1)
            pred_class = jnp.argmax(probs, axis=1)
            pred_certainty = jnp.max(probs, axis=1)
            result = (
                np.array(z_mean),
                np.array(recon),
                np.array(pred_class, dtype=np.int32),
                np.array(pred_certainty),
            )
            if return_mixture:
                responsibilities = extras.get("responsibilities") if hasattr(extras, "get") else None
                pi_val = extras.get("pi") if hasattr(extras, "get") else None
                if responsibilities is None or pi_val is None:
                    raise ValueError("Mixture responsibilities unavailable during predict().")
                result += (np.array(responsibilities), np.array(pi_val))
            return result

    def fit(self, data: np.ndarray, labels: np.ndarray, weights_path: str):
        """Train with semi-supervised labels (NaN = unlabeled). Returns history dict."""
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
        if self.config.prior_type == "mixture":
            self._save_mixture_diagnostics(trainer.latest_splits)
        return history


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "artifacts" / "checkpoints" / "ssvae.ckpt"
