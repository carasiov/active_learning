from __future__ import annotations

"""Metadata helpers for dashboard configuration UI."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from rcmvae.domain.config import SSVAEConfig
from rcmvae.domain.priors import PRIOR_REGISTRY


UNSET = object()


@dataclass(frozen=True)
class OptionSpec:
    label: str
    value: Any
    description: str | None = None

    def to_dash_option(self) -> Dict[str, Any]:
        option = {"label": self.label, "value": self.value}
        if self.description:
            option["title"] = self.description
        return option


@dataclass(frozen=True)
class FieldSpec:
    key: str
    component_id: str
    section: str
    label: str
    description: str
    control: str
    default: Any = None
    options: Tuple[OptionSpec, ...] = field(default_factory=tuple)
    props: Dict[str, Any] = field(default_factory=dict)
    width: int = 12
    extract: Callable[[Dict[str, Any]], Any] | None = None
    transform: Callable[[Any, Dict[str, Any]], Any] | None = None

    def extract_value(self, config_dict: Dict[str, Any] | None) -> Any:
        if not config_dict:
            return self.default
        if self.extract:
            return self.extract(config_dict)
        return config_dict.get(self.key, self.default)

    def apply_transform(self, value: Any, current_config: Dict[str, Any]) -> Any:
        if self.transform:
            return self.transform(value, current_config)
        return value


@dataclass(frozen=True)
class SectionSpec:
    id: str
    title: str
    description: str
    field_keys: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def _int_transform(
    name: str,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
    allow_none: bool = False,
) -> Callable[[Any, Dict[str, Any]], Any]:
    def _transform(value: Any, _config: Dict[str, Any]) -> Any:
        if value in (None, ""):
            if allow_none:
                return None
            raise ValueError(f"{name} is required.")
        try:
            converted = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be an integer.") from None
        if min_value is not None and converted < min_value:
            raise ValueError(f"{name} must be at least {min_value}.")
        if max_value is not None and converted > max_value:
            raise ValueError(f"{name} must be at most {max_value}.")
        return converted

    return _transform


def _float_transform(
    name: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    allow_none: bool = False,
    none_if_zero: bool = False,
) -> Callable[[Any, Dict[str, Any]], Any]:
    def _transform(value: Any, _config: Dict[str, Any]) -> Any:
        if value in (None, ""):
            if allow_none:
                return None
            raise ValueError(f"{name} is required.")
        try:
            converted = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number.") from None
        if none_if_zero and converted == 0:
            return None
        if min_value is not None and converted < min_value:
            raise ValueError(f"{name} must be at least {min_value}.")
        if max_value is not None and converted > max_value:
            raise ValueError(f"{name} must be at most {max_value}.")
        return converted

    return _transform


def _choice_transform(name: str, options: Sequence[str]) -> Callable[[Any, Dict[str, Any]], Any]:
    allowed = set(options)

    def _transform(value: Any, _config: Dict[str, Any]) -> Any:
        if value in (None, ""):
            raise ValueError(f"{name} is required.")
        if value not in allowed:
            raise ValueError(f"{name} must be one of: {', '.join(sorted(allowed))}.")
        return value

    return _transform


def _hidden_dims_extract(config_dict: Dict[str, Any]) -> str:
    dims = config_dict.get("hidden_dims", (256, 128, 64))
    if isinstance(dims, (list, tuple)):
        return ",".join(str(int(d)) for d in dims)
    if isinstance(dims, str):
        return dims
    return "256,128,64"


def _hidden_dims_transform(value: Any, _config: Dict[str, Any]) -> Tuple[int, ...]:
    if value in (None, ""):
        raise ValueError("Hidden Layers must be provided (e.g., '256,128,64').")
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        raise ValueError("Hidden Layers must include at least one dimension.")
    dims: List[int] = []
    for part in parts:
        try:
            dim = int(part)
        except ValueError:
            raise ValueError("Hidden Layers must contain integers only.") from None
        if dim <= 0:
            raise ValueError("Hidden Layers must be positive integers.")
        dims.append(dim)
    return tuple(dims)


def _dirichlet_alpha_transform(value: Any, _config: Dict[str, Any]) -> Any:
    if value in (None, ""):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        raise ValueError("Dirichlet α must be a number or blank.") from None
    if converted <= 0:
        raise ValueError("Dirichlet α must be positive or blank to disable.")
    return converted


# ---------------------------------------------------------------------------
# Option builders
# ---------------------------------------------------------------------------


def _build_prior_options() -> Tuple[OptionSpec, ...]:
    options: List[OptionSpec] = []
    for key, prior_cls in PRIOR_REGISTRY.items():
        doc = (prior_cls.__doc__ or "").strip().splitlines()
        description = doc[0].strip() if doc else None
        label = key.replace("_", " ").title()
        options.append(OptionSpec(label=label, value=key, description=description))
    return tuple(options)


def _build_monitor_metric_options() -> Tuple[OptionSpec, ...]:
    return (
        OptionSpec(label="Auto", value="auto", description="Choose based on label availability."),
        OptionSpec(label="Validation Loss", value="loss"),
        OptionSpec(label="Classification Loss", value="classification_loss"),
    )


def _build_reconstruction_loss_options() -> Tuple[OptionSpec, ...]:
    return (
        OptionSpec(label="Binary Cross-Entropy (MNIST)", value="bce"),
        OptionSpec(label="Mean Squared Error (continuous)", value="mse"),
    )


def _build_encoder_options() -> Tuple[OptionSpec, ...]:
    return (
        OptionSpec(label="Dense (MLP)", value="dense"),
        OptionSpec(label="Convolutional", value="conv"),
    )


# ---------------------------------------------------------------------------
# Field & Section definitions
# ---------------------------------------------------------------------------

_FIELD_SPECS: Tuple[FieldSpec, ...] = (
    # Training & data
    FieldSpec(
        key="batch_size",
        component_id="cfg-batch-size",
        section="training",
        label="Batch Size",
        description="Samples per optimizer step.",
        control="number",
        default=128,
        props={"min": 32, "max": 4096, "step": 32},
        transform=_int_transform("Batch size", min_value=32, max_value=4096),
    ),
    FieldSpec(
        key="learning_rate",
        component_id="cfg-learning-rate",
        section="training",
        label="Learning Rate",
        description="Adam/AdamW base learning rate.",
        control="number",
        default=1e-3,
        props={"min": 1e-5, "max": 1e-1, "step": 1e-4},
        transform=_float_transform("Learning rate", min_value=1e-5, max_value=1e-1),
    ),
    FieldSpec(
        key="max_epochs",
        component_id="cfg-max-epochs",
        section="training",
        label="Epoch Budget",
        description="Maximum training epochs before early stopping.",
        control="number",
        default=200,
        props={"min": 1, "max": 500},
        transform=_int_transform("Epoch budget", min_value=1, max_value=500),
    ),
    FieldSpec(
        key="patience",
        component_id="cfg-patience",
        section="training",
        label="Early-Stopping Patience",
        description="Epochs without improvement before stopping.",
        control="number",
        default=20,
        props={"min": 1, "max": 100},
        transform=_int_transform("Patience", min_value=1, max_value=100),
    ),
    FieldSpec(
        key="monitor_metric",
        component_id="cfg-monitor-metric",
        section="training",
        label="Monitored Metric",
        description="Validation metric used for early stopping.",
        control="dropdown",
        default="auto",
        options=_build_monitor_metric_options(),
        transform=_choice_transform("Monitored metric", ["auto", "loss", "classification_loss"]),
        width=6,
    ),
    FieldSpec(
        key="random_seed",
        component_id="cfg-random-seed",
        section="training",
        label="Random Seed",
        description="Seed for initialisation (blank = unchanged).",
        control="number",
        default=42,
        props={"min": 0, "max": 10_000, "step": 1},
        transform=_int_transform("Random seed", min_value=0, max_value=10_000),
        width=6,
    ),
    # Architecture
    FieldSpec(
        key="encoder_type",
        component_id="cfg-encoder-type",
        section="architecture",
        label="Encoder",
        description="Latent encoder family.",
        control="radio",
        default="conv",
        options=_build_encoder_options(),
        transform=_choice_transform("Encoder type", ["dense", "conv"]),
        width=6,
    ),
    FieldSpec(
        key="decoder_type",
        component_id="cfg-decoder-type",
        section="architecture",
        label="Decoder",
        description="Reconstruction network type (typically mirrors encoder).",
        control="radio",
        default="conv",
        options=_build_encoder_options(),
        transform=_choice_transform("Decoder type", ["dense", "conv"]),
        width=6,
    ),
    FieldSpec(
        key="latent_dim",
        component_id="cfg-latent-dim",
        section="architecture",
        label="Latent Dimension",
        description="Dimensionality of latent z (2 for visualization).",
        control="number",
        default=2,
        props={"min": 2, "max": 256},
        transform=_int_transform("Latent dimension", min_value=2, max_value=256),
        width=6,
    ),
    FieldSpec(
        key="hidden_dims",
        component_id="cfg-hidden-dims",
        section="architecture",
        label="Hidden Layers",
        description="Comma-separated layer sizes for dense encoder." ,
        control="text",
        default="256,128,64",
        extract=_hidden_dims_extract,
        transform=_hidden_dims_transform,
        width=6,
    ),
    FieldSpec(
        key="dropout_rate",
        component_id="cfg-dropout-rate",
        section="architecture",
        label="Classifier Dropout",
        description="Dropout rate applied inside classifier head.",
        control="number",
        default=0.2,
        props={"min": 0.0, "max": 0.8, "step": 0.05},
        transform=_float_transform("Dropout rate", min_value=0.0, max_value=0.8),
        width=6,
    ),
    # Prior & latent controls
    FieldSpec(
        key="prior_type",
        component_id="cfg-prior-type",
        section="prior",
        label="Prior",
        description="Latent prior distribution.",
        control="dropdown",
        default="mixture",
        options=_build_prior_options(),
        transform=_choice_transform("Prior", list(PRIOR_REGISTRY.keys())),
        width=6,
    ),
    FieldSpec(
        key="num_components",
        component_id="cfg-num-components",
        section="prior",
        label="Mixture Components",
        description="Number of prior components (>= num classes).",
        control="number",
        default=10,
        props={"min": 1, "max": 64},
        transform=_int_transform("Mixture components", min_value=1, max_value=128),
        width=6,
    ),
    FieldSpec(
        key="component_embedding_dim",
        component_id="cfg-component-embedding-dim",
        section="prior",
        label="Component Embedding Dim",
        description="Dimension of component embeddings (blank = latent dim).",
        control="number",
        default=None,
        props={"min": 1, "max": 128},
        transform=_int_transform("Component embedding dimension", min_value=1, max_value=256, allow_none=True),
        width=6,
    ),
    FieldSpec(
        key="use_component_aware_decoder",
        component_id="cfg-use-component-aware",
        section="prior",
        label="Component-aware Decoder",
        description="Separate decoder pathways per component (mixture/geometric).",
        control="switch",
        default=True,
        transform=lambda v, _cfg: bool(v),
        width=6,
    ),
    FieldSpec(
        key="use_tau_classifier",
        component_id="cfg-use-tau-classifier",
        section="prior",
        label="τ-classifier",
        description="Latent responsibility-based classifier (mixture/vamp).",
        control="switch",
        default=True,
        transform=lambda v, _cfg: bool(v),
        width=6,
    ),
    FieldSpec(
        key="tau_smoothing_alpha",
        component_id="cfg-tau-smoothing-alpha",
        section="prior",
        label="τ Laplace α",
        description="Smoothing prior for τ counts ( > 0 ).",
        control="number",
        default=1.0,
        props={"min": 0.01, "step": 0.1},
        transform=_float_transform("τ Laplace α", min_value=0.0001),
        width=6,
    ),
    FieldSpec(
        key="kl_c_weight",
        component_id="cfg-kl-c-weight",
        section="prior",
        label="Component KL Weight",
        description="Weight on KL(q(c|x) || π)",
        control="number",
        default=1.0,
        props={"min": 0.0, "max": 10.0, "step": 0.05},
        transform=_float_transform("Component KL weight", min_value=0.0, max_value=10.0),
        width=6,
    ),
    FieldSpec(
        key="kl_c_anneal_epochs",
        component_id="cfg-kl-c-anneal",
        section="prior",
        label="KL Component Anneal",
        description="Epochs to ramp kl_c_weight from 0 → target.",
        control="number",
        default=0,
        props={"min": 0, "max": 200},
        transform=_int_transform("KL component anneal epochs", min_value=0, max_value=500),
        width=6,
    ),
    FieldSpec(
        key="component_diversity_weight",
        component_id="cfg-component-diversity",
        section="prior",
        label="Component Diversity Weight",
        description="Negative encourages component usage diversity.",
        control="number",
        default=-0.1,
        props={"step": 0.01},
        transform=_float_transform("Component diversity weight", min_value=-10.0, max_value=10.0),
        width=6,
    ),
    FieldSpec(
        key="learnable_pi",
        component_id="cfg-learnable-pi",
        section="prior",
        label="Learn Mixture Weights",
        description="Enable learnable π for mixture/geometric priors.",
        control="switch",
        default=True,
        transform=lambda v, _cfg: bool(v),
        width=6,
    ),
    FieldSpec(
        key="dirichlet_alpha",
        component_id="cfg-dirichlet-alpha",
        section="prior",
        label="Dirichlet α (π prior)",
        description="Blank disables MAP prior; positive value strengthens regularization.",
        control="number",
        default=None,
        props={"min": 0.1, "max": 10.0, "step": 0.1},
        transform=_dirichlet_alpha_transform,
        width=6,
    ),
    FieldSpec(
        key="dirichlet_weight",
        component_id="cfg-dirichlet-weight",
        section="prior",
        label="Dirichlet Weight",
        description="Scaling factor for Dirichlet prior when enabled.",
        control="number",
        default=1.0,
        props={"min": 0.0, "max": 10.0, "step": 0.1},
        transform=_float_transform("Dirichlet weight", min_value=0.0, max_value=10.0),
        width=6,
    ),
    FieldSpec(
        key="top_m_gating",
        component_id="cfg-top-m-gating",
        section="prior",
        label="Top-M Gating",
        description="Reconstruct using only top-M components (0 = all).",
        control="number",
        default=0,
        props={"min": 0, "max": 64},
        transform=_int_transform("Top-M gating", min_value=0, max_value=128),
        width=6,
    ),
    FieldSpec(
        key="soft_embedding_warmup_epochs",
        component_id="cfg-soft-embedding-warmup",
        section="prior",
        label="Soft Embedding Warmup",
        description="Epochs to use soft embeddings before hard assignments.",
        control="number",
        default=0,
        props={"min": 0, "max": 200},
        transform=_int_transform("Soft embedding warmup", min_value=0, max_value=500),
        width=6,
    ),
    FieldSpec(
        key="mixture_history_log_every",
        component_id="cfg-mixture-log-every",
        section="prior",
        label="Mixture History Interval",
        description="Log π / responsibilities every N epochs (0 = disable).",
        control="number",
        default=1,
        props={"min": 0, "max": 50},
        transform=_int_transform("Mixture history interval", min_value=0, max_value=100),
        width=6,
    ),
    # Loss & regularisation
    FieldSpec(
        key="reconstruction_loss",
        component_id="cfg-recon-loss",
        section="loss",
        label="Reconstruction Loss",
        description="Pixel reconstruction objective.",
        control="radio",
        default="bce",
        options=_build_reconstruction_loss_options(),
        transform=_choice_transform("Reconstruction loss", ["bce", "mse"]),
        width=6,
    ),
    FieldSpec(
        key="recon_weight",
        component_id="cfg-recon-weight",
        section="loss",
        label="Recon Weight",
        description="Scaling for reconstruction term.",
        control="number",
        default=1.0,
        props={"min": 0.0, "max": 5000, "step": 0.5},
        transform=_float_transform("Recon weight", min_value=0.0, max_value=10_000.0),
        width=6,
    ),
    FieldSpec(
        key="kl_weight",
        component_id="cfg-kl-weight",
        section="loss",
        label="KL Weight",
        description="β coefficient for latent KL.",
        control="number",
        default=1.0,
        props={"min": 0.0, "max": 10.0, "step": 0.05},
        transform=_float_transform("KL weight", min_value=0.0, max_value=20.0),
        width=6,
    ),
    FieldSpec(
        key="label_weight",
        component_id="cfg-label-weight",
        section="loss",
        label="Label Loss Weight",
        description="Scaling for supervised classification loss.",
        control="number",
        default=1.0,
        props={"min": 0.0, "max": 50.0, "step": 0.5},
        transform=_float_transform("Label weight", min_value=0.0, max_value=50.0),
        width=6,
    ),
    FieldSpec(
        key="use_heteroscedastic_decoder",
        component_id="cfg-use-heteroscedastic",
        section="loss",
        label="Heteroscedastic Decoder",
        description="Learn per-sample variance σ(x) for reconstructions.",
        control="switch",
        default=False,
        transform=lambda v, _cfg: bool(v),
        width=6,
    ),
    FieldSpec(
        key="sigma_min",
        component_id="cfg-sigma-min",
        section="loss",
        label="σ Min",
        description="Lower bound on decoder variance.",
        control="number",
        default=0.05,
        props={"min": 1e-3, "max": 1.0, "step": 0.01},
        transform=_float_transform("σ min", min_value=1e-5, max_value=5.0),
        width=6,
    ),
    FieldSpec(
        key="sigma_max",
        component_id="cfg-sigma-max",
        section="loss",
        label="σ Max",
        description="Upper bound on decoder variance.",
        control="number",
        default=0.5,
        props={"min": 0.01, "max": 5.0, "step": 0.01},
        transform=_float_transform("σ max", min_value=0.01, max_value=10.0),
        width=6,
    ),
    FieldSpec(
        key="weight_decay",
        component_id="cfg-weight-decay",
        section="loss",
        label="Weight Decay",
        description="L2 regularisation via AdamW.",
        control="number",
        default=1e-4,
        props={"min": 0.0, "max": 1e-1, "step": 1e-5},
        transform=_float_transform("Weight decay", min_value=0.0, max_value=0.1),
        width=6,
    ),
    FieldSpec(
        key="grad_clip_norm",
        component_id="cfg-grad-clip",
        section="loss",
        label="Gradient Clip Norm",
        description="Set to 0/blank to disable clipping.",
        control="number",
        default=1.0,
        props={"min": 0.0, "max": 50.0, "step": 0.5},
        transform=_float_transform("Gradient clip norm", min_value=0.0, max_value=100.0, allow_none=True, none_if_zero=True),
        width=6,
    ),
    FieldSpec(
        key="use_contrastive",
        component_id="cfg-use-contrastive",
        section="loss",
        label="Contrastive Aux Loss",
        description="Enable optional contrastive term.",
        control="switch",
        default=False,
        transform=lambda v, _cfg: bool(v),
        width=6,
    ),
    FieldSpec(
        key="contrastive_weight",
        component_id="cfg-contrastive-weight",
        section="loss",
        label="Contrastive Weight",
        description="Scaling for contrastive loss.",
        control="number",
        default=0.0,
        props={"min": 0.0, "max": 10.0, "step": 0.1},
        transform=_float_transform("Contrastive weight", min_value=0.0, max_value=50.0),
        width=6,
    ),
)


_SECTION_SPECS: Tuple[SectionSpec, ...] = (
    SectionSpec(
        id="training",
        title="Training & Data",
        description="Core training schedule and optimisation controls.",
        field_keys=(
            "batch_size",
            "learning_rate",
            "max_epochs",
            "patience",
            "monitor_metric",
            "random_seed",
        ),
    ),
    SectionSpec(
        id="architecture",
        title="Architecture",
        description="Encoder/decoder families and latent dimensionality.",
        field_keys=(
            "encoder_type",
            "decoder_type",
            "latent_dim",
            "hidden_dims",
            "dropout_rate",
        ),
    ),
    SectionSpec(
        id="prior",
        title="Latent Prior & Responsibilities",
        description="Mixture structure, τ-classifier, and component regularisation.",
        field_keys=(
            "prior_type",
            "num_components",
            "component_embedding_dim",
            "use_component_aware_decoder",
            "use_tau_classifier",
            "tau_smoothing_alpha",
            "kl_c_weight",
            "kl_c_anneal_epochs",
            "component_diversity_weight",
            "learnable_pi",
            "dirichlet_alpha",
            "dirichlet_weight",
            "top_m_gating",
            "soft_embedding_warmup_epochs",
            "mixture_history_log_every",
        ),
    ),
    SectionSpec(
        id="loss",
        title="Losses & Regularisation",
        description="Objective weights and auxiliary regularisers.",
        field_keys=(
            "reconstruction_loss",
            "recon_weight",
            "kl_weight",
            "label_weight",
            "use_heteroscedastic_decoder",
            "sigma_min",
            "sigma_max",
            "weight_decay",
            "grad_clip_norm",
            "use_contrastive",
            "contrastive_weight",
        ),
    ),
)

_FIELD_BY_KEY: Dict[str, FieldSpec] = {spec.key: spec for spec in _FIELD_SPECS}
_FIELD_BY_ID: Dict[str, FieldSpec] = {spec.component_id: spec for spec in _FIELD_SPECS}


def get_field_specs() -> Tuple[FieldSpec, ...]:
    return _FIELD_SPECS


def get_field_by_id(component_id: str) -> FieldSpec:
    return _FIELD_BY_ID[component_id]


def get_section_specs() -> Tuple[SectionSpec, ...]:
    return _SECTION_SPECS


def extract_initial_values(config_dict: Dict[str, Any] | None) -> List[Any]:
    current = config_dict or {}
    values: List[Any] = []
    for spec in _FIELD_SPECS:
        values.append(spec.extract_value(current))
    return values


def build_updates(values: Sequence[Any], current_config: Dict[str, Any]) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    for spec, raw in zip(_FIELD_SPECS, values):
        transformed = spec.apply_transform(raw, current_config)
        if transformed is UNSET:
            continue
        updates[spec.key] = transformed
    return updates


def default_config_dict() -> Dict[str, Any]:
    base = SSVAEConfig()
    return {name: getattr(base, name) for name in base.__dataclass_fields__}