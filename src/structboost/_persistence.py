"""Checkpoint serialization for a fitted :class:`~structboost._model.BAE`.

The payload written here contains **only** tensors and Python primitives. That
is a deliberate constraint, not an accident of implementation: it is what lets
``BAE.load`` call ``torch.load(..., weights_only=True)``, so reading a
structboost checkpoint can never execute arbitrary code from the file. PyTorch's
restricted unpickler (``torch/_weights_only_unpickler.py``) admits tensors,
``torch.device``, ``torch.Size``, ``OrderedDict``, ``set``, ``bytes`` and the
builtin scalar/container types — but *not* NumPy arrays or NumPy scalars, which
is why everything crossing this boundary is converted explicitly.

Two things a fitted model carries are intentionally not written:

``ObsCovariateEncoding.encoded``
    The training-set covariate design matrix, shape ``(n_cells, n_columns)``. It
    is read only inside ``BAE.fit``, which re-encodes from the incoming
    ``AnnData`` before using it, so nothing needs it after a load. Dropping it
    keeps cell-level training data out of a shipped model file and keeps the
    file size independent of the training set.
optimizer state
    A loaded model is deployable, not resumable mid-run. ``fit`` rebuilds the
    decoder and its optimizer unconditionally anyway.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ._model import BAE
    from ._utils import ObsCovariateEncoding

#: Identifies the file as a structboost checkpoint. A ``.pt`` file is otherwise
#: indistinguishable from any other torch payload, and a confusing
#: ``KeyError`` deep inside the restore is a worse diagnostic than a refusal.
MAGIC = "structboost.bae"

#: Bumped whenever the payload layout changes incompatibly. ``load`` refuses a
#: file written by a newer format than it understands rather than guessing.
#:
#: 2 — added ``layer``. The bump is for *forward* compatibility: an older install
#: would ignore the key and silently read ``adata.X`` instead of the layer the
#: model was fitted on, which is a wrong answer rather than a missing one.
#:
#: 3 — the two covariate encodings became one, alongside the ``batch_key`` /
#: ``batch_integration_mode`` API. Formats 1 and 2 predate the first public
#: release, so no migration is written and the loader refuses them by name
#: rather than failing on a missing key.
#:
#: 4 — ``BAEConfig.boosting_mode`` was dropped with ``allboost``'s refine mode.
#: ``restore_payload`` splats the stored config into ``BAEConfig``, so a format-3
#: file would otherwise die on an unexpected keyword argument. Formats 1-3 predate
#: the first public release; 4 was released as 0.2.0.
#:
#: 5 — four ``BAEConfig`` fields dropped (``decoder_dropout_rate``,
#: ``decoder_use_batch_norm``, ``disentanglement_standardize``,
#: ``standardize_targets``) plus the ``balance_obs`` payload key. Same splatting
#: hazard as format 4. Unlike every earlier bump this one *does* break files
#: written by a released version (0.2.0), and no migration is written: the method
#: is under active development and a checkpoint is cheap to regenerate, whereas a
#: compatibility shim for options that no longer exist is not.
CHECKPOINT_FORMAT = 5
#: Oldest format this install can read.
MIN_CHECKPOINT_FORMAT = 5

#: Category element types that survive the round trip with their comparison
#: semantics intact. ``bool`` precedes ``int`` because ``bool`` is a subclass of
#: ``int`` and would otherwise be recorded as one.
_CATEGORY_KINDS: tuple[tuple[str, type], ...] = (
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("str", str),
)


def _to_primitive(obj: Any, *, coerce_unknown: bool = False) -> Any:
    """Convert `obj` into tensors-free primitives that ``weights_only`` accepts.

    NumPy scalars become Python scalars and NumPy arrays become nested lists;
    tuples become lists, since pickle would otherwise be the only thing keeping
    them apart and the distinction does not survive anyway.

    Parameters
    ----------
    obj
        Value to convert.
    coerce_unknown
        If True, anything not otherwise convertible is stringified. This is used
        only for provenance dictionaries, matching the best-effort
        ``json.dumps(..., default=str)`` policy in :mod:`structboost._io`. For
        values that take part in computation it stays False, so an unexpected
        type raises instead of silently becoming its ``repr``.
    """
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_primitive(v, coerce_unknown=coerce_unknown) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_primitive(v, coerce_unknown=coerce_unknown) for v in obj]
    if coerce_unknown:
        return str(obj)
    raise TypeError(f"cannot serialize object of type {type(obj).__name__!r} into a checkpoint")


def _tensor(array: Any) -> Any:
    """Convert an array-like to a CPU tensor, preserving dtype."""
    import torch

    return torch.as_tensor(np.asarray(array)).cpu()


def _array(tensor: Any) -> np.ndarray:
    """Convert a checkpoint tensor back to a NumPy array."""
    return tensor.cpu().numpy()


def _encode_categories(column: str, categories: list) -> dict[str, Any]:
    """Record a categorical column's levels with their type.

    The levels come from ``list(series.cat.categories)`` or
    ``list(pd.unique(series))``, so they can be NumPy scalars. Their *type*
    matters downstream: ``transform_obs_covariates`` both tests
    ``value not in categories`` and builds ``pd.Categorical(series,
    categories=...)``, and a level list whose dtype no longer matches the incoming
    column maps every cell to NaN, yielding an all-zero dummy block — a wrong
    reconstruction with no error raised. NumPy scalars are therefore unwrapped to
    true Python scalars, which compare equal to their NumPy counterparts, and
    anything outside the supported set is refused here rather than corrupting a
    reconstruction later.
    """
    values = [v.item() if isinstance(v, np.generic) else v for v in categories]
    for kind, py_type in _CATEGORY_KINDS:
        if all(isinstance(v, py_type) for v in values):
            return {"kind": kind, "values": values}
    offending = next(
        (type(v).__name__ for v in values if not isinstance(v, (bool, int, float, str))),
        type(values[0]).__name__ if values else "unknown",
    )
    raise ValueError(
        f"obs column {column!r} has categories of type {offending!r}, which cannot be "
        "persisted in a BAE checkpoint. Convert the column to string, integer or "
        "boolean before fitting."
    )


def _decode_categories(payload: dict[str, Any]) -> list:
    """Rebuild a category list with the element type it was saved with."""
    kind = payload["kind"]
    caster = {"bool": bool, "int": int, "float": float, "str": str}[kind]
    return [caster(v) for v in payload["values"]]


def _encode_encoding(encoding: ObsCovariateEncoding | None) -> dict[str, Any] | None:
    """Serialize an :class:`~structboost._utils.ObsCovariateEncoding`.

    ``encoded`` is dropped; see the module docstring.
    """
    if encoding is None:
        return None
    column_info: dict[str, Any] = {}
    for column, info in encoding.column_info.items():
        entry = {
            "type": info["type"],
            "n_dummies": int(info["n_dummies"]),
            "encoded_columns": [str(c) for c in info["encoded_columns"]],
        }
        if info["type"] == "categorical":
            entry["categories"] = _encode_categories(column, list(info["categories"]))
        column_info[column] = entry
    return {
        "column_info": column_info,
        "mean": _tensor(encoding.mean),
        "std": _tensor(encoding.std),
        "obs_columns": [str(c) for c in encoding.obs_columns],
        "n_columns": int(encoding.n_columns),
        "encoded_columns": [str(c) for c in encoding.encoded_columns],
    }


def _decode_encoding(payload: dict[str, Any] | None) -> ObsCovariateEncoding | None:
    """Rebuild an encoding, with a zero-row ``encoded`` placeholder."""
    if payload is None:
        return None
    from ._utils import ObsCovariateEncoding

    column_info: dict[str, dict] = {}
    for column, info in payload["column_info"].items():
        entry: dict[str, Any] = {
            "type": info["type"],
            "n_dummies": info["n_dummies"],
            "encoded_columns": list(info["encoded_columns"]),
        }
        if info["type"] == "categorical":
            entry["categories"] = _decode_categories(info["categories"])
        column_info[column] = entry
    n_columns = int(payload["n_columns"])
    return ObsCovariateEncoding(
        # Not the training design matrix: it is not persisted, and a zero-row
        # array makes any accidental use fail on shape instead of quietly
        # contributing wrong numbers.
        encoded=np.empty((0, n_columns), dtype=np.float64),
        column_info=column_info,
        mean=_array(payload["mean"]),
        std=_array(payload["std"]),
        obs_columns=list(payload["obs_columns"]),
        n_columns=n_columns,
        encoded_columns=list(payload["encoded_columns"]),
    )


def build_payload(model: BAE) -> dict[str, Any]:
    """Assemble the checkpoint dictionary for a fitted model."""
    from dataclasses import asdict

    config = asdict(model.config)
    config["device"] = str(model.config.device)
    config["decoder_hidden_dims"] = [int(d) for d in model.config.decoder_hidden_dims]

    report = model._training_report
    prior = model._prior_weights
    scaling = model._latent_scaling

    return {
        "magic": MAGIC,
        "format_version": CHECKPOINT_FORMAT,
        "structboost_version": _package_version(),
        "n_genes": int(model.n_genes),
        "config": _to_primitive(config),
        "encoder_state": model.encoder.state_dict(),
        "decoder_state": model.decoder.state_dict(),
        # Read off the decoder itself rather than re-derived from the encoding,
        # so the rebuilt geometry cannot drift from what the weights expect.
        "decoder_n_covariates": int(model.decoder.n_covariates),
        "is_fitted": bool(model._is_fitted),
        "layer": model._layer,
        "var_names": None if model._var_names is None else [str(v) for v in model._var_names],
        "training_history": {
            str(k): [float(x) for x in v] for k, v in model._training_history.items()
        },
        "training_report": (
            None if report is None else {k: _tensor(v) for k, v in report.to_dict().items()}
        ),
        "latent_init": _to_primitive(model._latent_init),
        "batch_encoding": _encode_encoding(model._batch_encoding),
        "batch_integration_mode": str(model._batch_integration_mode),
        "batch_weights": (None if model._batch_weights is None else _tensor(model._batch_weights)),
        "mandatory_genes": _to_primitive(model._mandatory_genes),
        "prior_weights": None if prior is None else _tensor(prior),
        "prior_info": _to_primitive(model._prior_info, coerce_unknown=True),
        "latent_scaling": (
            None
            if scaling is None
            else {"mean": _tensor(scaling["mean"]), "scale": _tensor(scaling["scale"])}
        ),
    }


def restore_payload(cls: type[BAE], payload: dict[str, Any], device: Any) -> BAE:
    """Rebuild a model from a checkpoint payload on `device`."""
    from ._decoder import BAEDecoder
    from ._types import BAEConfig, TrainingReport

    config_kwargs = dict(payload["config"])
    # Must be a tuple again: BAEConfig equality and `dataclasses.replace`
    # round-trips both depend on it, and a list would quietly propagate.
    config_kwargs["decoder_hidden_dims"] = tuple(config_kwargs["decoder_hidden_dims"])
    config_kwargs["device"] = device
    config = BAEConfig(**config_kwargs)

    model = cls(int(payload["n_genes"]), config)

    # Rebuild the decoder at the geometry the saved weights were trained with;
    # `__init__` builds an unconditioned one. Mirrors BAE.fit.
    decoder_input = 2 * config.latent_dim if config.split_softmax else config.latent_dim
    model.decoder = BAEDecoder(
        model.n_genes,
        config,
        input_dim_override=decoder_input,
        n_covariates=int(payload["decoder_n_covariates"]),
    ).to(config.device)

    model.encoder.load_state_dict(payload["encoder_state"])
    model.decoder.load_state_dict(payload["decoder_state"])
    model.to(config.device)

    model._is_fitted = bool(payload["is_fitted"])
    # `.get`: format-1 checkpoints predate layer support and were fitted on `.X`.
    model._layer = payload.get("layer")
    var_names = payload["var_names"]
    model._var_names = None if var_names is None else np.asarray(var_names, dtype=object)
    model._training_history = {k: list(v) for k, v in payload["training_history"].items()}
    report = payload["training_report"]
    model._training_report = (
        None
        if report is None
        else TrainingReport.from_dict({k: _array(v) for k, v in report.items()})
    )
    model._latent_init = dict(payload["latent_init"])

    model._batch_encoding = _decode_encoding(payload["batch_encoding"])
    model._batch_integration_mode = payload["batch_integration_mode"]
    batch_weights = payload["batch_weights"]
    model._batch_weights = None if batch_weights is None else _array(batch_weights)
    model._mandatory_genes = payload["mandatory_genes"]

    prior = payload["prior_weights"]
    model._prior_weights = None if prior is None else _array(prior)
    model._prior_info = dict(payload["prior_info"])
    scaling = payload["latent_scaling"]
    model._latent_scaling = (
        None
        if scaling is None
        else {"mean": _array(scaling["mean"]), "scale": _array(scaling["scale"])}
    )
    return model


def _package_version() -> str:
    """Best-effort package version for provenance; never fatal."""
    try:
        from . import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - provenance only
        return "unknown"
