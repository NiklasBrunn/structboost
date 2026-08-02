"""BAE main model class."""

from __future__ import annotations

import functools
import warnings
from dataclasses import fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ._boosting import allboost
from ._decoder import BAEDecoder
from ._encoder import BAEEncoder, SplitSoftmax
from ._types import BAEConfig, TrainingReport

if TYPE_CHECKING:
    from anndata import AnnData

    from ._types import Device

# Diagnostic series collected per iteration when `diagnostics=True`.
# "iteration" is generated at the end, not accumulated.
_REPORT_FIELDS: tuple[str, ...] = tuple(TrainingReport.__dataclass_fields__)

# Per-group reconstruction losses are reported only for obs columns with at most
# this many levels; beyond it the breakdown is per-cell noise rather than a
# summary. Columns above the limit are skipped with a warning, never silently.
_MAX_GROUP_LEVELS = 50

# The correlation method includes a small relative-variance barrier so that
# reducing correlation cannot be achieved solely by making dimensions constant.
# These are deliberately scale-relative: BAE's latent magnitude is set by
# boosting shrinkage and should not be forced to an arbitrary unit variance.
_DISENTANGLEMENT_VARIANCE_WEIGHT = 0.1
_DISENTANGLEMENT_MIN_STD_RATIO = 0.25
_DISENTANGLEMENT_RELATIVE_EPS = 1e-4


#: ``var`` columns that conventionally hold a gene identifier. Real datasets index
#: on one convention and keep the other in a column — CellRanger and scanpy put
#: symbols in ``var_names`` and Ensembl accessions in ``var["gene_ids"]``, while
#: cellxgene census does the reverse via ``feature_name``. A prior written with
#: Ensembl as the join key would otherwise fail to match a symbol-indexed dataset
#: even though both identifier sets are present on both sides.
_IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "gene_ids",
    "gene_id",
    "gene_symbols",
    "gene_symbol",
    "feature_name",
    "feature_id",
    "ensembl_id",
    "symbol",
)


def _identifier_sets(adata: AnnData) -> dict[str, np.ndarray]:
    """Every usable gene identifier column on an AnnData, keyed by name."""
    sets: dict[str, np.ndarray] = {"var_names": np.asarray(adata.var_names, dtype=object)}
    for column in _IDENTIFIER_COLUMNS:
        if column in adata.var.columns:
            sets[column] = np.asarray(adata.var[column], dtype=object)
    return sets


def _choose_join(
    prior_sets: dict[str, np.ndarray],
    target_sets: dict[str, np.ndarray],
) -> tuple[str, str, np.ndarray, np.ndarray]:
    """Pick the identifier pair that actually matches, not the first one offered.

    Scored by number of matched genes over every (prior, target) identifier pair.
    Ties keep the earlier prior key, so the reference's own join key wins whenever
    it works and the fallback only engages when it does not.
    """
    best: tuple[int, str, str] = (-1, "", "")
    for prior_key, prior_ids in prior_sets.items():
        prior_lookup = {str(v) for v in prior_ids.tolist()}
        for target_key, target_ids in target_sets.items():
            overlap = sum(1 for v in target_ids.tolist() if str(v) in prior_lookup)
            if overlap > best[0]:
                best = (overlap, prior_key, target_key)
    _, prior_key, target_key = best
    return prior_key, target_key, prior_sets[prior_key], target_sets[target_key]


def _resolve_prior_weights(
    reference: object,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    """Extract ``(weights, identifiers, metadata)`` from any accepted prior form.

    Several identifier sets may be returned — a Parquet file carries Ensembl
    accessions *and* symbols, an AnnData carries ``var_names`` plus any identifier
    columns — so the caller can join on whichever actually matches the target.

    A bare array is refused: without gene identifiers the matrix cannot be
    aligned to another dataset, and silently assuming positional correspondence
    between two gene panels would produce a plausible-looking but meaningless
    encoder.
    """
    from pathlib import Path

    if isinstance(reference, (str, Path)):
        from ._io import read_encoder_weights

        frame = read_encoder_weights(reference)
        meta = dict(frame.attrs.get("metadata", {}))
        meta["source"] = str(reference)
        sets = {frame.attrs.get("join_key", "gene_id"): frame.index.to_numpy(dtype=object)}
        if "gene_symbol" in frame.attrs:
            sets.setdefault("gene_symbol", np.asarray(frame.attrs["gene_symbol"], dtype=object))
        return frame.to_numpy(dtype=np.float64), sets, meta

    if isinstance(reference, BAE):
        if reference._var_names is None:
            raise ValueError(
                "This BAE carries no gene names, so its encoder cannot be aligned to "
                "another dataset. Gene names are recorded by `fit`; pass the AnnData "
                "the reference was fitted on instead, or an unfitted model is not a "
                "usable prior."
            )
        weights = reference.get_encoder_weights().astype(np.float64)
        return (
            weights,
            {"var_names": np.asarray(reference._var_names, dtype=object)},
            {"source": "BAE"},
        )

    # AnnData: duck-typed so this module keeps working without anndata installed.
    if hasattr(reference, "varm") and hasattr(reference, "var_names"):
        if "BAE_encoder_weights" not in reference.varm:
            raise ValueError(
                "AnnData has no varm['BAE_encoder_weights']. Fit a BAE on it first, "
                "or pass the weight matrix directly."
            )
        return (
            np.asarray(reference.varm["BAE_encoder_weights"], dtype=np.float64),
            _identifier_sets(reference),
            {"source": "AnnData"},
        )

    # DataFrame: duck-typed on the index/columns pair.
    if hasattr(reference, "index") and hasattr(reference, "columns"):
        return (
            reference.to_numpy(dtype=np.float64),
            {str(reference.index.name or "index"): np.asarray(reference.index, dtype=object)},
            {"source": "DataFrame"},
        )

    if isinstance(reference, np.ndarray):
        raise TypeError(
            "A bare array cannot be used as a prior encoder matrix: it carries no "
            "gene identifiers, so it cannot be aligned to the target panel. Pass a "
            "gene-indexed DataFrame, an AnnData with varm['BAE_encoder_weights'], a "
            "fitted BAE, or a .parquet/.csv written by write_encoder_weights."
        )
    raise TypeError(f"Unsupported reference type for a prior encoder matrix: {type(reference)!r}")


def _align_prior_to_panel(
    weights: np.ndarray,
    identifiers: np.ndarray,
    target: np.ndarray,
    *,
    min_coverage: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Reindex a prior weight matrix onto the target gene panel.

    Genes absent from the target contribute zero, which attenuates the affected
    program rather than breaking it. Per-dimension coverage — the share of each
    column's absolute weight mass that survived — reports by how much, since a
    program that lost half its genes is not the same program.

    Returns
    -------
    aligned
        Shape ``(len(target), latent_dim)``.
    coverage
        Shape ``(latent_dim,)``, retained absolute weight mass per dimension.
    n_matched
        How many reference genes were found in the target panel.
    """
    if weights.ndim != 2:
        raise ValueError(f"Prior weights must be 2-D (n_genes, latent_dim), got {weights.shape}")
    if weights.shape[0] != identifiers.shape[0]:
        raise ValueError(
            f"Prior weights has {weights.shape[0]} rows but {identifiers.shape[0]} "
            "gene identifiers were supplied"
        )

    # First occurrence wins: duplicated gene names are common in real panels and
    # a dict-based lookup keeps this total rather than raising on the duplicate.
    position = {}
    for index, name in enumerate(target.tolist()):
        position.setdefault(str(name), index)

    aligned = np.zeros((target.shape[0], weights.shape[1]), dtype=np.float64)
    matched = np.zeros(identifiers.shape[0], dtype=bool)
    for row, name in enumerate(identifiers.tolist()):
        index = position.get(str(name))
        if index is not None:
            aligned[index] = weights[row]
            matched[row] = True

    total_mass = np.abs(weights).sum(axis=0)
    kept_mass = np.abs(weights[matched]).sum(axis=0)
    coverage = np.divide(kept_mass, total_mass, out=np.ones_like(total_mass), where=total_mass > 0)

    if total_mass.min() == 0:
        dead = np.flatnonzero(total_mass == 0).tolist()
        raise ValueError(
            f"Prior dimensions {dead} have no non-zero weights, so they encode nothing "
            "and cannot be transferred. Drop them from the reference matrix."
        )
    if coverage.min() < min_coverage:
        worst = int(np.argmin(coverage))
        raise ValueError(
            f"Only {coverage.min():.1%} of the weight mass of prior dimension {worst} "
            f"is present in the target panel, below min_coverage={min_coverage:.0%}. "
            f"{int(matched.sum())}/{matched.size} reference genes matched overall. "
            "Identifier sets on both sides were tried automatically, so this is a "
            "genuine panel difference rather than an Ensembl-vs-symbol mismatch. "
            "Pass join_on= to force a particular adata.var column, or lower "
            "min_coverage to accept the attenuated program."
        )
    if coverage.min() < 0.9:
        warnings.warn(
            f"Prior dimension {int(np.argmin(coverage))} retains only "
            f"{coverage.min():.1%} of its weight mass on the target panel; its latent "
            "values are attenuated accordingly. Per-dimension coverage is recorded in "
            "adata.uns['bae_transfer']['prior_coverage'].",
            UserWarning,
            stacklevel=3,
        )
    return aligned, coverage, int(matched.sum())


def _build_allboost_mandatory(
    resolved_mandatory: np.ndarray | list[np.ndarray] | None,
    n_dummies: int,
    n_genes: int,
) -> np.ndarray | list[np.ndarray] | None:
    """Combine gene mandatory indices with obs covariate mandatory indices."""
    obs_indices = np.arange(n_genes, n_genes + n_dummies, dtype=np.intp) if n_dummies > 0 else None

    if resolved_mandatory is None and obs_indices is None:
        return None

    if resolved_mandatory is None:
        return obs_indices

    if obs_indices is None:
        return resolved_mandatory

    # Both present: merge
    if isinstance(resolved_mandatory, list):
        # Per-target: append obs indices to each sub-list
        return [np.concatenate([sub, obs_indices]) for sub in resolved_mandatory]
    return np.concatenate([resolved_mandatory, obs_indices])


class _FitLayer:
    """Sentinel: "whichever layer this model was fitted on"."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "<fit-time layer>"


#: Default for the ``layer`` argument of :meth:`BAE.transform` and
#: :meth:`BAE.reconstruct`. A plain ``None`` default cannot express this, because
#: ``None`` already means ``adata.X`` — which a model fitted on a layer must be
#: able to be told explicitly.
FIT_LAYER = _FitLayer()

#: Sentinel default for ``batch_integration_mode``. Behaves as ``"both"``, but is
#: distinguishable from a caller who typed ``"both"``, so that a mode given
#: without a ``batch_key`` can be reported as the mistake it is instead of
#: silently doing nothing.
_MODE_UNSET = "__unset__"

#: Which mechanisms each mode switches on. The covariate is never an encoder
#: *input*: ``"encoder"`` means it enters the boosting fit as a mandatory
#: regressor, so gene selection is not confounded by it.
_BATCH_MODES: dict[str, tuple[bool, bool]] = {
    # mode: (conditions the decoder, regresses inside the boosting fit)
    "none": (False, False),
    "decoder": (True, False),
    "encoder": (False, True),
    "both": (True, True),
}


def _expression_matrix(adata: AnnData, layer: str | None):
    """Return the matrix BAE should read, raw and possibly sparse.

    Every read of expression data in this module goes through here, so the choice
    of layer is made in exactly one place and cannot drift between ``fit`` and
    the methods that consume a fitted model.
    """
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        # anndata >= 0.13 exposes ``.X`` as ``layers[None]``, so iterating the
        # mapping yields a ``None`` key alongside the real names. Sorting that
        # mixed list raises TypeError and buries this KeyError, turning a clear
        # "no such layer" message into a comparison error from the error path.
        available = sorted(name for name in adata.layers if name is not None)
        raise KeyError(f"adata has no layer {layer!r}; available layers: {available}")
    return adata.layers[layer]


def _torch_rng_state() -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Snapshot the global torch RNG state, CPU and every visible CUDA device."""
    cuda_states = (
        [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else []
    )
    return torch.get_rng_state(), cuda_states


def _restore_torch_rng_state(state: tuple[torch.Tensor, list[torch.Tensor]]) -> None:
    """Put back a state captured by :func:`_torch_rng_state`."""
    cpu_state, cuda_states = state
    torch.set_rng_state(cpu_state)
    for device, device_state in enumerate(cuda_states):
        torch.cuda.set_rng_state(device_state, device)


def _isolates_torch_rng(*, config_fallback: bool = False):
    """Restore the caller's global torch RNG state around a *seeded* call.

    Seeding is what makes a BAE fit reproducible, and it necessarily goes through
    the global torch generator: the decoder's ``reset_parameters`` and the
    ``DataLoader``'s shuffling both draw from it, and neither accepts a local
    generator without changing the draws. Left alone, that seeding leaks — a
    caller who fits a model finds their own ``torch`` stream reset underneath
    them. This wrapper puts the stream back afterwards, so the isolation is
    invisible in the fitted model and every seed keeps producing exactly the
    results it did before.

    Deliberately inert when no seed is in effect. An unseeded call does not seed
    either; restoring state around it would make two successive unseeded fits on
    the same model return *identical* results, quietly removing the variability
    an unseeded fit is supposed to have.

    Parameters
    ----------
    config_fallback
        Whether an absent ``seed`` argument falls back to ``self.config.seed``.
        True for :meth:`BAE.fit`, which applies that fallback itself; False for
        :meth:`BAE._iteration_support_frequency`, whose seed comes only from
        :meth:`BAE.stability_selection` and never from the config.
    """

    def decorate(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            seed = kwargs.get("seed")
            if seed is None and config_fallback:
                seed = self.config.seed
            if seed is None:
                return method(self, *args, **kwargs)
            state = _torch_rng_state()
            try:
                return method(self, *args, **kwargs)
            finally:
                _restore_torch_rng_state(state)

        return wrapper

    return decorate


def _mandatory_genes_for_uns(spec: object) -> np.ndarray | dict[str, np.ndarray]:
    """Render a ``mandatory_genes`` specification into an h5ad-writable value.

    A per-dimension specification is a list of lists, and the sub-lists need not
    be the same length. Storing that raw makes the whole ``AnnData`` unwritable:
    ``adata.write_h5ad`` tries to build one array from it and raises
    ``ValueError: setting an array element with a sequence``. Equal-length
    sub-lists happen to coerce to a 2-D array and survive, so the failure only
    appears once a user asks for different marker counts per dimension.

    Per-dimension specifications therefore become a dict of one array per
    dimension, keyed ``dim_0``, ``dim_1``, ... — the same key convention the
    encoder weight files use (:mod:`structboost._io`). A flat specification stays
    a single array, which is what it already round-tripped as.
    """
    is_per_dimension = (
        isinstance(spec, (list, tuple))
        and len(spec) > 0
        and all(isinstance(sub, (list, tuple, np.ndarray)) for sub in spec)
    )
    if is_per_dimension:
        return {f"dim_{i}": np.asarray(sub) for i, sub in enumerate(spec)}
    return np.asarray(spec)


class BAE(nn.Module):
    """Boosting Autoencoder for interpretable dimensionality reduction.

    Combines a linear encoder (optimized via componentwise boosting) with an
    MLP decoder (optimized via SGD). The hybrid training procedure alternates
    between boosting-based encoder updates and gradient-based decoder updates.

    Parameters
    ----------
    n_genes
        Number of input genes.
    config
        BAE configuration. If None, uses defaults.

    Examples
    --------
    >>> import anndata as ad
    >>> adata = ad.read_h5ad("data.h5ad")
    >>> model = BAE(adata.n_vars)
    >>> model.fit(adata)  # Stores results in adata.obsm["X_bae"]
    >>> latent = adata.obsm["X_bae"]
    """

    def __init__(self, n_genes: int, config: BAEConfig | None = None) -> None:
        super().__init__()
        self.config = config or BAEConfig()
        self.n_genes = n_genes

        # Build components
        self.encoder = BAEEncoder(n_genes, self.config)
        self.split_softmax_layer = SplitSoftmax() if self.config.split_softmax else None
        decoder_input_dim = 2 * self.config.latent_dim if self.config.split_softmax else None
        self.decoder = BAEDecoder(n_genes, self.config, input_dim_override=decoder_input_dim)

        # State
        self._is_fitted = False
        self._training_history: dict[str, list[float]] = {
            "train_loss": [],
            "selection_loss": [],
        }
        self._training_report: TrainingReport | None = None
        self._latent_init: dict[str, str | int] = {"method": "zero", "pretrain_epochs": 0}
        #: One encoding drives both mechanisms. Which of them are active is
        #: recorded by `_batch_integration_mode` rather than by two attributes.
        self._batch_encoding = None
        self._batch_integration_mode: str = "none"
        self._batch_weights: np.ndarray | None = None
        self._balance_obs: str | None = None
        self._mandatory_genes = None
        #: Layer `fit` read expression from; None means ``adata.X``. Later calls
        #: default to it so a model always reads the representation it learned on.
        self._layer: str | None = None
        self._var_names: np.ndarray | None = None
        # Transfer state: set by `from_reference`, None for an ordinary model.
        self._prior_weights: np.ndarray | None = None
        self._prior_info: dict[str, object] = {}
        self._latent_scaling: dict[str, np.ndarray] | None = None

        # Move to device
        self.to(self.config.device)

    @classmethod
    def from_reference(
        cls,
        reference: object,
        adata: AnnData,
        *,
        n_additional_dims: int = 5,
        prior_mode: Literal["frozen", "anchored"] | None = None,
        min_coverage: float = 0.5,
        join_on: str | None = None,
        config: BAEConfig | None = None,
    ) -> BAE:
        """Build a model that carries a reference encoder matrix onto new data.

        The transferable product of a BAE fit is its encoder weight matrix: k0
        sparse gene programs. This constructor aligns such a matrix to ``adata``'s
        gene panel, places it in the first k0 latent dimensions, and appends
        ``n_additional_dims`` zero-initialized dimensions for variance the prior
        programs cannot explain.

        Fitting then runs in two phases. The decoder is first trained against the
        prior programs alone (``fit(..., decoder_warmup_epochs=...)``), which is
        what makes the added dimensions *residual*: the boosting target is
        ``z* = z - lr * dL/dz``, so until the decoder has converged against the
        prior programs the gradient still carries signal those programs could
        explain, and the new dimensions would merely re-learn them. Boosting then
        starts, restricted to the new dimensions or anchored on the prior ones
        according to ``prior_mode``.

        Parameters
        ----------
        reference
            The prior encoder matrix. Accepts a fitted :class:`BAE`, an
            :class:`~anndata.AnnData` carrying ``varm["BAE_encoder_weights"]``, a
            gene-indexed :class:`pandas.DataFrame`, or a path to a ``.parquet`` /
            ``.csv`` written by :func:`~structboost.write_encoder_weights`. A bare
            array is rejected: it carries no gene identifiers, so aligning it to
            another panel would be guesswork.
        adata
            Target dataset. Required here rather than at ``fit`` time because the
            encoder's shape depends on its gene panel, and because a coverage
            failure should surface before any training happens.
        n_additional_dims
            Number of new latent dimensions, default 5. ``0`` is valid and useful:
            it adapts the decoder (and, under ``"anchored"``, the programs) to the
            new data without adding capacity.

            This is a user choice, and the default is a pragmatic starting point
            rather than one derived from the data — how much residual structure a
            dataset holds is not knowable in advance. Erring high is the cheaper
            mistake under ``prior_mode="frozen"``, where the transferred programs
            stay bitwise fixed no matter how many dimensions are added, and each
            novel column's gene set can be read or ignored independently without
            refitting. Too small a value silently misses structure instead.

            To check the choice after fitting, read
            ``novel_variance_share_per_dim`` and the novel dimensions' stability:
            a dimension contributing almost nothing is surplus. Evaluate on
            held-out cells via :meth:`transfer_diagnostics` — in-sample every
            dimension appears to contribute, because free dimensions always reduce
            training error.
        prior_mode
            ``"frozen"`` (default) or ``"anchored"``; see :class:`BAEConfig`.
            Overrides ``config.prior_mode`` when given.
        min_coverage
            Minimum share of a prior dimension's absolute weight mass that must be
            present in ``adata``. Below this the transfer raises rather than
            silently returning an attenuated program.
        join_on
            Column of ``adata.var`` to align on. By default every identifier set
            available on both sides is tried and the one matching the most genes
            wins — a reference keyed on Ensembl accessions therefore aligns to a
            symbol-indexed dataset without intervention, provided either side
            carries the other convention (CellRanger and scanpy put symbols in
            ``var_names`` and accessions in ``var["gene_ids"]``). The pair used is
            recorded in ``adata.uns["bae_transfer"]["join_key"]``. Set this
            explicitly when a dataset carries several identifier columns and the
            automatic choice must not be trusted.
        config
            Base configuration. ``latent_dim`` is overwritten with
            ``k0 + n_additional_dims``, since the layout is determined by the
            reference matrix rather than chosen. Defaults to a fresh
            :class:`BAEConfig` and is **not** inherited from the reference: the
            reference's ``seed``, ``max_iterations`` and stopping rule describe how
            that model was fitted, not how this one should be. Pass the reference's
            config explicitly to reuse its boosting hyperparameters.

        Returns
        -------
        An unfitted model with the aligned prior installed.
        """
        if n_additional_dims < 0:
            raise ValueError(f"n_additional_dims must be >= 0, got {n_additional_dims}")
        if not 0.0 < min_coverage <= 1.0:
            raise ValueError(f"min_coverage must be in (0, 1], got {min_coverage}")

        weights, prior_sets, info = _resolve_prior_weights(reference)
        if join_on is not None:
            if join_on not in adata.var.columns:
                raise ValueError(
                    f"join_on={join_on!r} is not a column of adata.var. Available: "
                    f"{list(adata.var.columns)}"
                )
            target_sets = {join_on: np.asarray(adata.var[join_on], dtype=object)}
        else:
            target_sets = _identifier_sets(adata)
        prior_key, target_key, identifiers, target = _choose_join(prior_sets, target_sets)
        info["join_key"] = f"{prior_key}->{target_key}"
        aligned, coverage, n_matched = _align_prior_to_panel(
            weights, identifiers, target, min_coverage=min_coverage
        )

        n_prior = aligned.shape[1]
        latent_dim = n_prior + n_additional_dims
        if config is not None and config.latent_dim != latent_dim:
            # Mirrors the warm-start path, which warns for the same reason: a
            # silently overridden latent_dim is a model of a different size than
            # the caller asked for.
            warnings.warn(
                f"latent_dim was changed from {config.latent_dim} to {latent_dim} to "
                f"match the reference matrix: {n_prior} prior dimensions plus "
                f"n_additional_dims={n_additional_dims}. The layout is determined by "
                "the reference, so latent_dim is not independently settable here.",
                UserWarning,
                stacklevel=2,
            )
        prior_mode_kw = {"prior_mode": prior_mode} if prior_mode is not None else {}
        resolved = replace(config or BAEConfig(), latent_dim=latent_dim, **prior_mode_kw)
        if config is None and isinstance(reference, BAE):
            # Only `latent_dim` is carried over from the prior, so the call reads
            # as "the reference plus k dimensions" while every other setting
            # silently reverts to `BAEConfig()`. A reference tuned to
            # `max_iterations=200` transfers at 1000 with nothing to show for it.
            #
            # Inheriting instead was rejected: a Path or array reference carries
            # no config, so the same prior would behave differently depending on
            # whether it was passed as a model or as its exported weights.
            #
            # Diffing two `replace` results rather than the raw configs drops
            # `latent_dim` and `prior_mode` on its own, since both sides force
            # them identically. No exclusion list to keep in sync.
            inherited = replace(reference.config, latent_dim=latent_dim, **prior_mode_kw)
            differing = [
                (f.name, getattr(inherited, f.name), getattr(resolved, f.name))
                for f in fields(BAEConfig)
                if getattr(inherited, f.name) != getattr(resolved, f.name)
            ]
            if differing:
                changes = ", ".join(f"{name} {was!r} -> {now!r}" for name, was, now in differing)
                warnings.warn(
                    f"from_reference used BAEConfig defaults, not the reference's: {changes}.",
                    UserWarning,
                    stacklevel=2,
                )
        model = cls(adata.n_vars, resolved)
        model._prior_weights = aligned
        model._prior_info = {
            **info,
            "n_prior_dims": n_prior,
            "n_additional_dims": int(n_additional_dims),
            "prior_coverage": coverage.tolist(),
            "n_reference_genes": int(weights.shape[0]),
            "n_matched_genes": n_matched,
            "min_coverage": float(min_coverage),
        }
        return model

    def save(self, path: str | Path) -> Path:
        """Write the fitted model to a single checkpoint file.

        The checkpoint holds everything needed to reproduce :meth:`transform`
        and :meth:`reconstruct`, to use the model as a ``reference`` for
        :meth:`from_reference`, and to read back ``training_history`` /
        ``training_report``.

        Two things are deliberately excluded. The decoder's optimizer state is
        not written, so a loaded model is deployable but cannot resume a
        training run mid-flight — a fresh :meth:`fit` still works, since it
        rebuilds the decoder and optimizer regardless. And the training-set
        covariate design matrix (``ObsCovariateEncoding.encoded``) is not
        written, so a shipped model file carries no cell-level training data and
        its size does not grow with the training set; only the encoding
        *parameters* needed to encode new data are kept.

        The payload contains only tensors and plain Python values, which is what
        lets :meth:`load` read it with ``weights_only=True``: loading a
        structboost checkpoint cannot execute code from the file.

        Parameters
        ----------
        path
            Destination file. ``.pt`` is the conventional suffix. Parent
            directories are created if needed.

        Returns
        -------
        The path written.

        See Also
        --------
        BAE.load : Read a checkpoint back.
        structboost.write_encoder_weights : Persist only the gene programs, in a
            readable, shareable format.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        ValueError
            If a conditioned fit used an obs column whose categorical levels
            cannot be persisted (for example datetimes). Converting the column
            to string, integer or boolean before fitting resolves it.

        Examples
        --------
        >>> model.fit(adata)  # doctest: +SKIP
        >>> model.save("bae_model.pt")  # doctest: +SKIP
        """
        from ._persistence import build_payload

        if not self._is_fitted:
            raise RuntimeError(
                "Model not fitted; there is nothing to save. Call fit() first, or use "
                "write_encoder_weights() to persist the prior encoder matrix of a "
                "from_reference() model."
            )
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(build_payload(self), destination)
        return destination

    @classmethod
    def load(cls, path: str | Path, *, device: Device | None = None) -> BAE:
        """Load a model written by :meth:`save`.

        Parameters
        ----------
        path
            Checkpoint file.
        device
            Device to place the model on. If None, the device recorded at save
            time is used when it is available, and CPU otherwise — the fallback
            warns rather than relocating silently.

        Returns
        -------
        The restored model, ready for :meth:`transform` and :meth:`reconstruct`.

        Raises
        ------
        FileNotFoundError
            If `path` does not exist.
        ValueError
            If the file is not a structboost checkpoint, or was written by a
            newer checkpoint format than this version understands.

        Examples
        --------
        >>> model = BAE.load("bae_model.pt")  # doctest: +SKIP
        >>> latent = model.transform(adata)  # doctest: +SKIP
        """
        from ._persistence import (
            CHECKPOINT_FORMAT,
            MAGIC,
            MIN_CHECKPOINT_FORMAT,
            restore_payload,
        )

        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"No BAE checkpoint at {source}")

        # Always read onto CPU first: the file may name a device this machine
        # does not have, and `restore_payload` moves the model afterwards.
        payload = torch.load(source, map_location="cpu", weights_only=True)

        if not isinstance(payload, dict) or payload.get("magic") != MAGIC:
            raise ValueError(f"{source} is not a structboost BAE checkpoint (written by BAE.save).")
        written_format = int(payload.get("format_version", 0))
        if written_format > CHECKPOINT_FORMAT:
            raise ValueError(
                f"{source} uses checkpoint format {written_format}, written by structboost "
                f"{payload.get('structboost_version', 'unknown')}; this install understands "
                f"up to {CHECKPOINT_FORMAT}. Upgrade structboost to load it."
            )
        if written_format < MIN_CHECKPOINT_FORMAT:
            raise ValueError(
                f"{source} uses checkpoint format {written_format}, which predates the first "
                f"public release; this install reads format {MIN_CHECKPOINT_FORMAT} and newer. "
                "Refit the model to write a current checkpoint."
            )
        resolved = cls._resolve_load_device(device, source, payload)
        return restore_payload(cls, payload, resolved)

    @staticmethod
    def _resolve_load_device(device: Device | None, source: Path, payload: dict) -> torch.device:
        """Pick the device for a load, warning when the saved one is unavailable."""
        if device is not None:
            return torch.device(device)
        requested = torch.device(payload.get("config", {}).get("device", "cpu"))
        mps = getattr(torch.backends, "mps", None)
        available = {
            "cuda": torch.cuda.is_available(),
            "mps": bool(mps is not None and mps.is_available()),
        }
        if not available.get(requested.type, True):
            warnings.warn(
                f"{source} was saved on {requested}, which is not available here; "
                "loading onto CPU instead. Pass device= to choose explicitly.",
                UserWarning,
                stacklevel=3,
            )
            return torch.device("cpu")
        return requested

    def _warn_on_panel_mismatch(self, adata: AnnData, method: str) -> None:
        """Warn when `adata`'s gene panel differs from the one fitted on.

        The encoder is a plain matrix product against gene columns in a fixed
        order, so a panel of the same width in a different order produces
        plausible, wrong numbers rather than an error. That was hard to trigger
        while a fitted model lived only in the session that produced it; a model
        loaded from disk months later makes it easy.
        """
        if self._var_names is None:
            return
        incoming = np.asarray(adata.var_names, dtype=object)
        if incoming.shape == self._var_names.shape and np.array_equal(incoming, self._var_names):
            return
        if incoming.shape != self._var_names.shape:
            detail = f"{incoming.size} genes against {self._var_names.size} at fit time"
        else:
            positions = np.flatnonzero(incoming != self._var_names)
            shown = ", ".join(
                f"{int(i)}: {self._var_names[i]!r}->{incoming[i]!r}" for i in positions[:3]
            )
            more = "" if positions.size <= 3 else f" (+{positions.size - 3} more)"
            detail = f"{positions.size} positions differ ({shown}{more})"
        warnings.warn(
            f"BAE.{method} received a gene panel that does not match the fitted one: "
            f"{detail}. The encoder maps gene columns by position, so the result is "
            "only meaningful if the panel is identical. Use BAE.from_reference to align "
            "a model to a different panel.",
            UserWarning,
            stacklevel=3,
        )

    def _transfer_boosting_inputs(
        self,
        targets: np.ndarray,
        n_features: int,
        prior_dims: int,
        mandatory: np.ndarray | list[np.ndarray] | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | list[np.ndarray] | None]:
        """Boosting inputs for one iteration: targets, offset, mandatory spec.

        ``"frozen"`` withholds the prior columns from boosting entirely.
        ``"anchored"`` boosts them, but from the fixed original matrix as a
        boosting offset, so the fitted correction is bounded by ``boosting_stepno``
        and is discarded at the start of the next iteration rather than compounding.

        A per-dimension ``mandatory`` list is sliced to match the columns actually
        boosted; ``allboost`` requires one entry per target.
        """
        if self._prior_weights is None:
            return targets, None, mandatory
        if self.config.prior_mode == "frozen":
            sliced = mandatory[prior_dims:] if isinstance(mandatory, list) else mandatory
            return targets[:, prior_dims:], None, sliced
        beta_init = np.zeros((self.config.latent_dim, n_features), dtype=np.float64)
        beta_init[:prior_dims, : self.n_genes] = self._prior_weights.T
        return targets, beta_init, mandatory

    def _expand_transfer_betamat(
        self, betamat: np.ndarray, n_features: int, prior_dims: int
    ) -> np.ndarray:
        """Restore the full ``(latent_dim, n_features)`` layout after frozen boosting.

        Nuisance coefficients stay zero for the prior rows: those dimensions were
        never fitted this iteration, so there is no nuisance estimate to report for
        them.
        """
        if self._prior_weights is None or self.config.prior_mode != "frozen":
            return betamat
        full = np.zeros((self.config.latent_dim, n_features), dtype=np.float64)
        full[:prior_dims, : self.n_genes] = self._prior_weights.T
        full[prior_dims:] = betamat
        return full

    def _validate_transfer_fit(
        self, adata: AnnData, *, init_obsm: str | None, init_pca: bool
    ) -> None:
        """Reject fit settings whose semantics conflict with a prior encoder matrix."""
        if adata.n_vars != self.n_genes:
            raise ValueError(
                f"This model was aligned to a {self.n_genes}-gene panel by "
                f"from_reference, but adata has {adata.n_vars} genes. Rebuild with "
                "from_reference against the dataset you intend to fit."
            )
        if self.config.split_softmax:
            raise ValueError(
                "split_softmax is not supported with a prior encoder matrix. It "
                "applies one softmax across all 2*latent_dim entries, so adding "
                "dimensions dilutes every prior entry by a data-dependent amount — a "
                "frozen matrix would no longer mean the prior programs act unchanged."
            )
        if self.config.standardize_targets and self.config.prior_mode == "anchored":
            raise ValueError(
                "standardize_targets is not supported with prior_mode='anchored'. "
                "Rescaling each target column to unit variance puts the fitted "
                "coefficients on a per-iteration scale that the fixed anchor does not "
                "share. Use prior_mode='frozen', or standardize_targets=False."
            )
        if init_pca or init_obsm is not None:
            raise ValueError(
                "init_pca/init_obsm cannot be combined with a prior encoder matrix: "
                "they override the latent state on the first boosting iteration, which "
                "is exactly where the prior programs define the target. Use "
                "decoder_warmup_epochs to settle the decoder against the prior instead."
            )

    @property
    def prior_weights(self) -> np.ndarray | None:
        """The *anchor*: aligned reference matrix ``(n_genes, k0)``, or None.

        This is what was transferred in, not what was fitted. Under
        ``prior_mode="frozen"`` the two are identical; under ``"anchored"`` compare
        against :attr:`fitted_prior_weights` to see how far the data moved the
        programs.
        """
        return None if self._prior_weights is None else self._prior_weights.copy()

    @property
    def n_prior_dims(self) -> int:
        """Number of transferred dimensions; ``0`` for an ordinary model."""
        return 0 if self._prior_weights is None else int(self._prior_weights.shape[1])

    @property
    def fitted_prior_weights(self) -> np.ndarray | None:
        """Fitted transferred block ``(n_genes, k0)``, or None if not a transfer."""
        if self._prior_weights is None:
            return None
        return self.get_encoder_weights()[:, : self.n_prior_dims]

    @property
    def novel_weights(self) -> np.ndarray | None:
        """Fitted novel block ``(n_genes, k1)``, or None if not a transfer.

        These are the gene programs the transfer *added* — the residual structure
        the reference matrix could not represent. Empty (``k1 == 0``) when the
        model was built for decoder-only adaptation.
        """
        if self._prior_weights is None:
            return None
        return self.get_encoder_weights()[:, self.n_prior_dims :]

    @property
    def training_history(self) -> dict[str, list[float]]:
        """Per-iteration reconstruction and checkpoint-selection losses."""
        return self._training_history

    @property
    def training_report(self) -> TrainingReport | None:
        """Per-iteration diagnostics, or None if ``diagnostics`` was off."""
        return self._training_report

    def forward(
        self,
        x: torch.Tensor,
        obs_covariates: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode then decode.

        Parameters
        ----------
        x
            Input tensor of shape (n_cells, n_genes).
        obs_covariates
            Optional obs covariate tensor for cVAE conditioning.

        Returns
        -------
        x_recon
            Reconstructed input.
        z
            Latent representation.
        """
        z = self.encoder(x)
        h = self.split_softmax_layer(z) if self.split_softmax_layer else z
        x_recon = self.decoder(h, obs_covariates)
        return x_recon, z

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation.

        Parameters
        ----------
        x
            Input tensor of shape (n_cells, n_genes).

        Returns
        -------
        Latent representation of shape (n_cells, latent_dim).
        """
        self.eval()
        with torch.no_grad():
            return self.encoder(x)

    # --- Hybrid Training Helpers ---

    @staticmethod
    def _recon_loss(
        x_recon: torch.Tensor,
        x: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruction MSE, optionally weighted per cell.

        Without weights this is a single fused reduction over all elements. That
        matters beyond speed: reducing per cell and then averaging reassociates the
        float32 sum and shifts results by ~1e-7 relative against releases that
        predate per-cell weighting, for every user who never asked for weights.
        The per-cell path is taken only when weights are actually supplied.
        """
        if sample_weights is None:
            return nn.functional.mse_loss(x_recon, x)
        per_cell = (x_recon - x).square().mean(dim=1)
        return (per_cell * sample_weights).sum() / sample_weights.sum()

    @staticmethod
    def _correlation_disentanglement_loss(
        z: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Squared latent correlation and a relative anti-collapse barrier.

        The first term is the mean squared off-diagonal correlation. Unlike a
        covariance penalty, it cannot be reduced merely by shrinking every latent
        dimension and letting the decoder compensate with larger weights. The
        second term discourages an individual dimension from becoming nearly
        constant relative to the mean latent standard deviation, without imposing
        an arbitrary unit-variance scale.

        This adapts the covariance-reduction and variance-preservation principles
        of DeCov [1]_ and VICReg [2]_ to BAE's deterministic latent codes.

        Parameters
        ----------
        z
            Latent codes of shape ``(n_cells, latent_dim)``.
        sample_weights
            Optional non-negative cell weights. When supplied, the weighted mean
            and covariance describe the same balanced pseudo-population used by
            the reconstruction target loss.

        Returns
        -------
        correlation_loss
            Mean squared off-diagonal latent correlation.
        variance_loss
            Relative minimum-variance barrier.

        References
        ----------
        .. [1] Cogswell et al. (2016), "Reducing Overfitting in Deep Networks by
           Decorrelating Representations", arXiv:1511.06068.
        .. [2] Bardes, Ponce & LeCun (2022), "VICReg: Variance-Invariance-
           Covariance Regularization for Self-Supervised Learning", ICLR 2022.
        """
        if z.ndim != 2:
            raise ValueError(f"z must be 2-D, got shape {tuple(z.shape)}")

        if sample_weights is None:
            centered = z - z.mean(dim=0, keepdim=True)
            covariance = centered.T @ centered / z.shape[0]
        else:
            if sample_weights.ndim != 1 or sample_weights.shape[0] != z.shape[0]:
                raise ValueError(
                    f"sample_weights must have shape (n_cells,), got {tuple(sample_weights.shape)}"
                )
            weight_sum = sample_weights.sum()
            mean = (z * sample_weights[:, None]).sum(dim=0, keepdim=True) / weight_sum
            centered = z - mean
            covariance = centered.T @ (centered * sample_weights[:, None]) / weight_sum

        variances = torch.diagonal(covariance)
        relative_eps = variances.mean().detach() * _DISENTANGLEMENT_RELATIVE_EPS + z.new_tensor(
            1e-12
        )
        denominator = torch.sqrt(
            (variances[:, None] + relative_eps) * (variances[None, :] + relative_eps)
        )
        correlation = covariance / denominator
        off_diagonal = correlation - torch.diag_embed(torch.diagonal(correlation))
        n_pairs = max(z.shape[1] * (z.shape[1] - 1), 1)
        correlation_loss = off_diagonal.square().sum() / n_pairs

        std = torch.sqrt(variances + relative_eps)
        reference_std = std.mean().detach()
        minimum_std = _DISENTANGLEMENT_MIN_STD_RATIO * reference_std
        variance_loss = torch.relu(minimum_std - std).square().mean() / (
            reference_std.square() + z.new_tensor(1e-12)
        )
        return correlation_loss, variance_loss

    def _compute_boosting_targets(
        self,
        X: torch.Tensor,
        *,
        lr: float = 1.0,
        obs_covariates: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
        stats: dict[str, float] | None = None,
        z_override: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Compute boosting targets via single gradient step from current z.

        Computes targets = z_current - lr * ∂L_target/∂z, representing one step of
        functional gradient descent on the latent representation. With
        ``config.disentanglement="correlation"``, ``L_target`` additionally
        includes the soft latent-correlation and relative-variance terms returned
        by :meth:`_correlation_disentanglement_loss`.

        ``L_target`` sums the squared error over cells and averages it over genes::

            L_target = (1 / n_genes) * sum_cells sum_genes (x_hat - x)^2
                     = n_cells * L_mean

        where ``L_mean`` is the plain elementwise MSE used for the decoder update
        and reported reconstruction loss. Checkpoint selection also uses
        ``L_mean`` unless correlation disentanglement is active, in which case it
        adds the normalized constraint. The two reconstruction conventions differ
        only by the constant ``n_cells``, but that constant matters here: under
        ``L_mean`` each cell's share of the gradient is divided by ``n_cells``, so
        the same cell in a larger dataset receives a proportionally smaller target
        step, and ``target_optim_lr`` silently means something different at every
        dataset size. Summing over cells makes each cell's step depend only on its
        own reconstruction error; averaging over genes keeps it stable when
        ``n_genes`` changes.

        This holds per cell only because the decoder runs in eval mode below, so
        batch-norm uses running statistics and does not couple cells together.

        Parameters
        ----------
        X
            Full input data tensor of shape (n_cells, n_genes).
        lr
            Step size for gradient descent direction.
        obs_covariates
            Optional obs covariate tensor for cVAE decoder conditioning.
        sample_weights
            Optional non-negative per-cell weights with mean one.
        stats
            Optional dict filled in place with ``"target_grad_norm"`` and
            ``"loss_pre_boost"``. Both quantities are already computed here, so
            collecting them costs nothing. ``"loss_pre_boost"`` is the *mean* MSE,
            keeping it comparable with ``loss_post_boost`` and
            ``loss_post_decoder``; ``"target_grad_norm"`` is the norm of the
            ``L_target`` gradient that the encoder is actually fitted against.
        z_override
            Latent state to start from instead of ``encoder(X)``. Used once, on
            the first iteration, to warm-start training from a supplied
            representation.

        Returns
        -------
        Target latent codes as numpy array of shape (n_cells, latent_dim).
        """
        self.decoder.eval()
        # Get current encoder output (detached) and create leaf tensor for grad
        if z_override is not None:
            z_current = z_override
        else:
            with torch.no_grad():
                z_current = self.encoder(X)
        z = z_current.clone().detach().requires_grad_(True)

        # Compute gradient of reconstruction loss w.r.t. z (through split-softmax)
        h = self.split_softmax_layer(z) if self.split_softmax_layer else z
        x_recon = self.decoder(h, obs_covariates)
        loss = self._recon_loss(x_recon, X, sample_weights)
        # Sum over cells, mean over genes. See the docstring: this is what makes a
        # cell's target step independent of how many other cells are in the dataset.
        # Unweighted, that is exactly `n_cells` times the elementwise mean.
        target_loss = (
            loss * X.shape[0]
            if sample_weights is None
            else ((x_recon - X).square().mean(dim=1) * sample_weights).sum()
        )
        # Only dL/dz is needed. `torch.autograd.grad` skips accumulating gradients
        # into the decoder parameters, which `loss.backward()` would compute and
        # leave in `.grad` for the decoder optimizer to discard on its next
        # `zero_grad()`. Same convention as `_full_recon_loss`.
        if self.config.disentanglement == "correlation":
            correlation_loss, variance_loss = self._correlation_disentanglement_loss(
                z, sample_weights
            )
            # The reconstruction target loss sums over cells. Correlation and
            # variance are population averages, so multiply them by n_cells to
            # keep the per-cell regularizer gradient, and therefore lambda's
            # meaning, independent of dataset size.
            target_loss = target_loss + X.shape[0] * self.config.disentanglement_lambda * (
                correlation_loss + _DISENTANGLEMENT_VARIANCE_WEIGHT * variance_loss
            )
        (z_grad,) = torch.autograd.grad(target_loss, z)

        # Target = current z moved in negative gradient direction
        with torch.no_grad():
            targets = z - lr * z_grad
            if stats is not None:
                stats["target_grad_norm"] = float(z_grad.norm())
                # Report the mean MSE, not `target_loss`: the A/B/C loss
                # decomposition in TrainingReport compares this against
                # `loss_post_boost` and `loss_post_decoder`, which are both means.
                stats["loss_pre_boost"] = float(loss.detach())

        return targets.cpu().numpy()

    def _update_decoder(
        self,
        X: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        k_steps: int,
        *,
        D_train: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
    ) -> float:
        """Update decoder via minibatch SGD.

        Parameters
        ----------
        X
            Full input data tensor.
        optimizer
            Optimizer for decoder parameters.
        k_steps
            Number of SGD steps to perform.
        D_train
            Optional obs covariate tensor for cVAE decoder conditioning.
        sample_weights
            Optional non-negative per-cell weights with mean one.

        Returns
        -------
        Average loss over the k steps.
        """
        self.decoder.train()
        tensors = [X]
        if D_train is not None:
            tensors.append(D_train)
        if sample_weights is not None:
            tensors.append(sample_weights)
        dataset = TensorDataset(*tensors)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        total_loss = 0.0
        steps_done = 0

        for _ in range(k_steps):
            for batch_data in loader:
                if D_train is not None:
                    x_batch, d_batch = batch_data[0], batch_data[1]
                    w_batch = batch_data[2] if sample_weights is not None else None
                else:
                    x_batch = batch_data[0]
                    d_batch = None
                    w_batch = batch_data[1] if sample_weights is not None else None
                optimizer.zero_grad()
                # The encoder is fitted by boosting and is in no optimizer, so its
                # gradient is never read. Detaching keeps backward from computing a
                # (latent_dim, n_genes) gradient per minibatch and from accumulating
                # it into `encoder.linear.weight.grad`, which nothing ever zeroes.
                with torch.no_grad():
                    z = self.encoder(x_batch)
                h = self.split_softmax_layer(z) if self.split_softmax_layer else z
                x_recon = self.decoder(h, d_batch)
                loss = self._recon_loss(x_recon, x_batch, w_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps_done += 1
                if steps_done >= k_steps:
                    break
            if steps_done >= k_steps:
                break

        return total_loss / max(steps_done, 1)

    def _checkpoint_selection_loss(
        self,
        train_loss: float,
        X: torch.Tensor,
        sample_weights: torch.Tensor | None,
    ) -> float:
        """Objective used for early stopping and best-state restoration.

        Reconstruction-only fits preserve the historical ``train_loss`` criterion.
        Correlation-constrained fits add the same dimensionless disentanglement
        penalty used to construct their boosting targets; otherwise restoring the
        best reconstruction checkpoint can silently undo the constraint.
        """
        if self.config.disentanglement != "correlation":
            return train_loss
        with torch.no_grad():
            z = self.encoder(X)
            correlation_loss, variance_loss = self._correlation_disentanglement_loss(
                z, sample_weights
            )
            penalty = self.config.disentanglement_lambda * (
                correlation_loss + _DISENTANGLEMENT_VARIANCE_WEIGHT * variance_loss
            )
        return train_loss + float(penalty)

    def _resolve_latent_init(
        self,
        adata: AnnData,
        X_np: np.ndarray,
        *,
        init_obsm: str | None,
        init_pca: bool,
        is_standardized: bool,
    ) -> tuple[np.ndarray | None, str]:
        """Resolve the requested warm-start latent state.

        Returns the initial latent matrix (or None for the default zero start)
        together with a short provenance string.
        """
        if init_obsm is not None and init_pca:
            raise ValueError("init_obsm and init_pca are mutually exclusive; pass only one")

        if init_pca:
            from ._utils import _pca_scores

            k = self.config.latent_dim
            n_max = min(X_np.shape)
            if k > n_max:
                raise ValueError(
                    f"init_pca needs latent_dim <= min(n_cells, n_genes) = {n_max}, "
                    f"got latent_dim={k}"
                )
            # Never rescale, and skip centering when the data is already z-transformed
            # so that no further transformation is applied to it.
            return _pca_scores(X_np, k, center=not is_standardized), "pca"

        if init_obsm is None:
            return None, "zero"

        if init_obsm not in adata.obsm:
            raise ValueError(
                f"init_obsm key {init_obsm!r} not found in adata.obsm. "
                f"Available keys: {list(adata.obsm.keys())}"
            )
        z_init = np.asarray(adata.obsm[init_obsm], dtype=np.float64)
        if z_init.ndim != 2:
            raise ValueError(f"adata.obsm[{init_obsm!r}] must be 2-D, got shape {z_init.shape}")
        if z_init.shape[0] != adata.n_obs:
            raise ValueError(
                f"adata.obsm[{init_obsm!r}] has {z_init.shape[0]} rows but adata has "
                f"{adata.n_obs} observations"
            )
        if z_init.shape[1] < 1:
            raise ValueError(f"adata.obsm[{init_obsm!r}] must have at least one column")
        if not np.isfinite(z_init).all():
            raise ValueError(f"adata.obsm[{init_obsm!r}] contains non-finite values")
        return z_init, f"obsm:{init_obsm}"

    def _pretrain_decoder(
        self,
        Z: torch.Tensor,
        X: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        n_epochs: int,
        *,
        D: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
        verbose: bool = False,
        desc: str = "Pre-training decoder",
    ) -> float:
        """Fit the decoder to map a fixed latent state to X, before BAE training.

        The encoder is not involved: `Z` is held fixed, so this teaches the decoder
        to read the warm-start latent space. Must be called after the decoder is
        final (`fit` re-initializes it under the seed) and after `optimizer` has
        been built over its parameters.

        Parameters
        ----------
        verbose
            Show a per-epoch progress bar with the running reconstruction loss.
            Worth having: a transfer's warm-up can run for hundreds of epochs
            before the training loop's own bar appears, and silence there is
            indistinguishable from a hang.
        desc
            Progress bar label.

        Returns
        -------
        Mean reconstruction loss over the final epoch.
        """
        tensors = [Z, X]
        if D is not None:
            tensors.append(D)
        if sample_weights is not None:
            tensors.append(sample_weights)
        dataset = TensorDataset(*tensors)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        self.decoder.train()
        epoch_loss = 0.0
        progress = tqdm(range(n_epochs), desc=desc, disable=not verbose, unit="epoch")
        for _ in progress:
            total, steps = 0.0, 0
            for batch in loader:
                z_batch, x_batch = batch[0], batch[1]
                h = self.split_softmax_layer(z_batch) if self.split_softmax_layer else z_batch
                optimizer.zero_grad()
                covariates = batch[2] if D is not None else None
                weight_index = 3 if D is not None else 2
                weights = batch[weight_index] if sample_weights is not None else None
                loss = self._recon_loss(self.decoder(h, covariates), x_batch, weights)
                loss.backward()
                optimizer.step()
                total += loss.item()
                steps += 1
            epoch_loss = total / max(steps, 1)
            progress.set_postfix(loss=f"{epoch_loss:.4f}")
        progress.close()
        return epoch_loss

    def _full_recon_loss(
        self,
        X: torch.Tensor,
        D: torch.Tensor | None,
        *,
        sample_weights: torch.Tensor | None = None,
        grad: bool = False,
    ) -> tuple[float, float]:
        """Full-data reconstruction MSE, and optionally the decoder gradient norm.

        Always runs the decoder in eval mode so dropout and batch-norm statistics
        are untouched and no RNG is consumed; gradients are taken with
        ``torch.autograd.grad`` so nothing accumulates into ``.grad``. Together
        these keep diagnostics read-only with respect to the fitted model.
        """
        was_training = self.decoder.training
        self.decoder.eval()
        try:
            if not grad:
                with torch.no_grad():
                    z = self.encoder(X)
                    h = self.split_softmax_layer(z) if self.split_softmax_layer else z
                    loss = self._recon_loss(self.decoder(h, D), X, sample_weights)
                    return float(loss), float("nan")

            z = self.encoder(X)
            h = self.split_softmax_layer(z) if self.split_softmax_layer else z
            loss = self._recon_loss(self.decoder(h, D), X, sample_weights)
            params = [p for p in self.decoder.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, retain_graph=False)
            norm = float(torch.sqrt(sum((g**2).sum() for g in grads)))
            return float(loss.detach()), norm
        finally:
            self.decoder.train(was_training)

    def _collect_diagnostics(
        self,
        series: dict[str, list],
        stats: dict[str, float],
        *,
        X: torch.Tensor,
        D: torch.Tensor | None,
        sample_weights: torch.Tensor | None,
        targets: np.ndarray,
        prev_W: np.ndarray | None,
    ) -> np.ndarray:
        """Record one iteration of diagnostics; returns the current encoder weights.

        Called after the encoder has been re-fit and the decoder updated, so the
        boosting-target comparison uses the same targets the encoder was fitted to.
        """
        # .copy() is required: numpy() shares storage with the tensor, and
        # set_weights writes in place, so a view would silently track the
        # current weights and make every convergence metric read as "no change".
        W = self.encoder.linear.weight.detach().cpu().numpy().copy()

        # How well did boosting fit the targets it was given?
        with torch.no_grad():
            z = self.encoder(X).cpu().numpy()
        ss_res = float(((targets - z) ** 2).sum())
        ss_tot = float(((targets - targets.mean(axis=0)) ** 2).sum())
        series["boosting_r2"].append(1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))
        series["latent_var_per_dim"].append(z.var(axis=0))

        # Sparsity and encoder scale.
        selected = np.abs(W).sum(axis=0) > 0
        series["n_selected"].append(int(selected.sum()))
        series["n_selected_per_dim"].append((np.abs(W) > 0).sum(axis=1))
        w_norm = float(np.abs(W).mean())
        series["encoder_weight_norm"].append(w_norm)

        # Convergence against the previous iteration. The encoder magnitude grows
        # by orders of magnitude during training, so the weight change is only
        # meaningful relative to it.
        if prev_W is None:
            series["weight_change_rel"].append(float("nan"))
            series["support_jaccard"].append(float("nan"))
            series["min_dim_cosine"].append(float("nan"))
        else:
            delta = float(np.abs(W - prev_W).mean())
            series["weight_change_rel"].append(delta / w_norm if w_norm > 0 else float("nan"))
            prev_sel = np.abs(prev_W).sum(axis=0) > 0
            union = int((selected | prev_sel).sum())
            series["support_jaccard"].append(
                float((selected & prev_sel).sum()) / union if union else float("nan")
            )
            num = (W * prev_W).sum(axis=1)
            den = np.linalg.norm(W, axis=1) * np.linalg.norm(prev_W, axis=1)
            cos = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
            series["min_dim_cosine"].append(
                float(np.nanmin(cos)) if np.isfinite(cos).any() else float("nan")
            )

        # Loss decomposition across the two halves of the alternation.
        loss_c, dec_grad = self._full_recon_loss(X, D, sample_weights=sample_weights, grad=True)
        loss_a = stats["loss_pre_boost"]
        loss_b = stats["loss_post_boost"]
        series["loss_pre_boost"].append(loss_a)
        series["loss_post_boost"].append(loss_b)
        series["loss_post_decoder"].append(loss_c)
        series["encoder_delta"].append(loss_b - loss_a)
        series["decoder_delta"].append(loss_c - loss_b)
        series["target_grad_norm"].append(stats["target_grad_norm"])
        series["decoder_grad_norm"].append(dec_grad)
        return W

    @staticmethod
    def _check_standardized(X: np.ndarray, tol: float = 0.1) -> bool:
        """Check if data appears to be standardized (z-transformed)."""
        col_means = np.abs(X.mean(axis=0))
        col_stds = X.std(axis=0)
        mean_ok = np.median(col_means) < tol
        std_ok = np.abs(np.median(col_stds) - 1.0) < tol
        return mean_ok and std_ok

    #: Columns with a standard deviation below this are treated as constant and
    #: left unscaled. Matches the guard in ``disentangle_boosting_targets``.
    _STD_EPS: float = 1e-12

    @staticmethod
    def _standardize_targets(targets: np.ndarray) -> np.ndarray:
        """Standardize targets to zero mean and unit variance per column.

        Columns whose standard deviation falls below ``_STD_EPS`` are left
        unscaled. Testing ``std == 0`` is not enough: a latent dimension carrying
        almost no signal has a tiny but nonzero standard deviation, and dividing
        by it inflates pure numerical noise to unit variance, handing a dead
        dimension the same weight in the boosting step as a real one.

        Parameters
        ----------
        targets
            Target matrix of shape (n_samples, latent_dim).

        Returns
        -------
        Standardized target matrix.
        """
        mean = targets.mean(axis=0, keepdims=True)
        std = targets.std(axis=0, keepdims=True)
        std = np.where(std < BAE._STD_EPS, 1.0, std)
        return (targets - mean) / std

    # --- AnnData Integration (scverse conventions) ---

    @staticmethod
    def _to_tensor(X: np.ndarray | sp.spmatrix, device: torch.device) -> torch.Tensor:
        """Convert array or sparse matrix to tensor."""
        if sp.issparse(X):
            X = X.toarray()
        return torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)

    @_isolates_torch_rng(config_fallback=True)
    def fit(
        self,
        adata: AnnData,
        *,
        layer: str | None = None,
        mandatory_genes: list[str] | list[int] | np.ndarray | list | None = None,
        batch_key: str | list[str] | None = None,
        batch_integration_mode: Literal["encoder", "decoder", "both"] = _MODE_UNSET,  # type: ignore[assignment]
        balance_obs: str | None = None,
        max_iterations: int | None = None,
        early_stopping_patience: int | None = None,
        enable_early_stopping: bool | None = None,
        seed: int | None = None,
        verbose: bool = True,
        diagnostics: bool | None = None,
        init_obsm: str | None = None,
        init_pca: bool = False,
        init_pretrain_epochs: int = 0,
        decoder_warmup_epochs: int = 0,
        stability_selection: str | bool | None = None,
    ) -> BAE:
        """Fit BAE using hybrid boosting+SGD training.

        The training loop alternates between:

        1. Computing boosting targets via gradient step: z* = z - lr * ∂L_target/∂z,
           where L_target sums the squared error over cells and averages it over
           genes (see ``_compute_boosting_targets``)
        2. (Optional) Applying leave-one-out target residualization
        3. (Optional) Standardizing targets for optimal boosting convergence
        4. Resetting encoder weights to zero
        5. Fitting encoder via allboost to map X → z*
        6. Updating decoder via minibatch SGD

        Parameters
        ----------
        adata
            AnnData object with gene expression. Data should be standardized
            (z-transformed).
        layer
            Read expression from ``adata.layers[layer]`` instead of ``adata.X``.
            The choice is remembered: :meth:`transform`, :meth:`reconstruct` and
            the diagnostics default to the same layer, so a fitted model always
            reads the representation it learned on. Pass the same layer to
            :func:`~structboost.linear_ceiling` when comparing reconstruction
            quality against its achievable maximum.
        mandatory_genes
            Gene names (str) or column indices (int) placed in the unpenalized
            adjustment block of the boosting fit, so they are never subject to
            competitive selection. Can be a flat list (applied to all latent dims)
            or a list of lists (per latent dimension).

            This forces them into the model *specification*, not into the fitted
            support: a gene whose contribution is estimated as zero still ends up
            with a zero encoder weight. Do not rely on this to guarantee that a
            marker appears in the selected gene set.
        batch_key
            Obs column, or several, holding the covariate to integrate over.
            ``None`` means no integration is performed.
        batch_integration_mode
            Which of the two mechanisms to apply. Defaults to ``"both"``.

            ``"decoder"``
                The encoded covariate is concatenated to the decoder input, so
                the decoder can explain covariate-driven variation directly and
                the latent code does not have to carry it.
            ``"encoder"``
                The encoded covariate is added to the boosting design as a
                mandatory regressor, so a covariate-correlated gene is not
                selected *because of* the covariate.
            ``"both"``
                Both of the above. This is the usual choice.

            **The covariate is never an encoder input.** ``"encoder"`` names the
            half of the model it protects, not a tensor it is fed to.
            :meth:`transform` stays gene-only and needs no covariate labels under
            any mode, which is what makes a fitted encoder deployable on data
            carrying no covariate annotation. :meth:`reconstruct` needs them only
            under ``"decoder"`` and ``"both"``.

            Passing a mode without a ``batch_key`` raises, rather than silently
            integrating nothing.

            The ridge that stabilizes the ``"encoder"`` mechanism when covariates
            are near-collinear lives on :class:`BAEConfig` as ``nuisance_ridge``.
        balance_obs
            Optional categorical obs column used to inverse-frequency weight the
            *reconstruction losses*, giving each observed level equal total weight
            so that a large group cannot dominate the fit.

            Scope, precisely: the weights enter the boosting-target gradient, the
            decoder update, the reported losses and the diagnostics. They do **not**
            enter the ``allboost`` fit itself, which remains ordinary (unweighted)
            least squares. So the targets the encoder chases are balanced, but the
            projection of those targets onto genes is not, and gene selection still
            leans toward the larger group. Making it a true weighted least squares
            would require weighted column norms, weighted inner products and a
            weight-dependent covariance cache throughout ``_boosting.py``.

            Measured effect on a deliberately imbalanced 500/90/45 design: the
            spread in per-group reconstruction MSE fell from 0.231 to 0.150.
            Check ``adata.uns["bae"]["reconstruction_loss_by_obs"]`` to see whether
            it helped on your data — and check first whether group size actually
            predicts fit quality, because uneven per-group reconstruction has causes
            other than imbalance.
        max_iterations
            Maximum training iterations (overrides config).
        early_stopping_patience
            Early stopping patience (overrides config).
        enable_early_stopping
            Enable/disable early stopping (overrides config).
        seed
            Random seed for reproducibility (overrides config).
        verbose
            Print training progress.
        diagnostics
            Collect per-iteration training diagnostics (overrides config). Adds
            full-data forward/backward passes per iteration; see
            :attr:`training_report`. Also adds the relative encoder weight change
            (``dW``, which converges toward 0) and the number of selected genes
            (``n_sel``) to the progress bar. Does not change the fitted model.
        init_obsm
            Warm-start the latent state from ``adata.obsm[init_obsm]`` instead of
            from zero. If that representation has a different number of columns
            than ``config.latent_dim``, the representation wins: ``latent_dim`` is
            overwritten for this fit and a ``UserWarning`` is emitted. Mutually
            exclusive with ``init_pca``.
        init_pca
            Warm-start from a PCA of ``adata.X`` keeping ``config.latent_dim``
            components. The data is never rescaled, and it is mean-centered only
            when it is not already z-transformed, so no transformation is applied
            on top of one the caller already performed. Mutually exclusive with
            ``init_obsm``.
        init_pretrain_epochs
            Before training, fit the decoder to map the warm-start latent state to
            ``adata.X`` for this many full passes over the cells. Only valid with
            ``init_obsm`` or ``init_pca``; passing a positive value without one
            raises. Default 0 (no pre-training); 20 is a conservative value.
            Measured trade-off: this lowers the initial loss substantially but does
            not improve the converged loss, and large values noticeably reduce
            gene-selection precision, because the decoder is tuned to a latent code
            the sparse encoder cannot exactly reproduce.
        decoder_warmup_epochs
            Transfer models only (:meth:`from_reference`). Trains the decoder
            against the frozen prior programs for this many passes before boosting
            starts. This is what makes the added dimensions *residual*: until the
            decoder has converged against the prior, ``dL/dz`` still carries signal
            those programs could explain, and the new dimensions would re-learn it.
        stability_selection
            Run :meth:`stability_selection` once after training and store its
            results in ``adata``. One of:

            ``None`` (default)
                Skip it.
            ``"subsample"``
                Meinshausen-Bühlmann cell resampling; error-controlled.
            ``"iteration"``
                Frequency over further training iterations; targets the larger
                variance source on measured data but provides no error bound.

            A single argument rather than a flag plus a mode, so the combination
            "disabled, but with a mode" cannot be expressed. ``True`` is accepted as
            a deprecated alias for ``"subsample"``. Call the method directly for
            control over ``n_runs`` and ``threshold``.

        Returns
        -------
        Self for method chaining.

        Notes
        -----
        The warm start is applied once, on the first iteration only: it sets the
        boosting targets, so the encoder learns the supplied representation and the
        decoder is then trained against the encoder's output. With
        ``standardize_targets=True`` the warm-start targets are standardized like
        any others, preserving the structure of the representation but not its scale.

        With ``config.disentanglement="correlation"``, the soft penalty is applied
        inside step 1 rather than as a target transformation. Early stopping and
        best-state restoration then use reconstruction loss plus the weighted
        disentanglement penalty. ``training_history["train_loss"]`` remains the
        unregularized decoder MSE; ``training_history["selection_loss"]`` records
        the checkpoint-selection objective.
        """
        # Resolve mandatory gene names to indices
        from ._utils import resolve_mandatory_genes

        # Resolve the batch arguments. No `batch_key` means no integration, and a
        # mode named without one is a mistake worth reporting rather than a
        # silent no-op. An additive conditioning alternative was evaluated and
        # rejected as strictly dominated; the CHANGELOG records the measurements.
        if batch_integration_mode is not _MODE_UNSET and batch_integration_mode not in _BATCH_MODES:
            raise ValueError(
                f"batch_integration_mode must be one of 'encoder', 'decoder', 'both', "
                f"got {batch_integration_mode!r}"
            )
        if batch_key is None:
            if batch_integration_mode is not _MODE_UNSET:
                raise ValueError(
                    f"batch_integration_mode={batch_integration_mode!r} was given without a "
                    "batch_key, so there is no covariate to integrate over. Pass "
                    "batch_key, or drop the mode."
                )
            batch_columns: list[str] | None = None
            batch_mode = "none"
        else:
            batch_columns = [batch_key] if isinstance(batch_key, str) else list(batch_key)
            if not batch_columns:
                raise ValueError("batch_key must name at least one obs column")
            batch_mode = "both" if batch_integration_mode is _MODE_UNSET else batch_integration_mode
        conditions_decoder, regresses_encoder = _BATCH_MODES[batch_mode]
        nuisance_ridge = self.config.nuisance_ridge
        if stability_selection not in (None, False, True, "subsample", "iteration"):
            raise ValueError(
                "stability_selection must be None, 'subsample' or 'iteration', "
                f"got {stability_selection!r}"
            )

        resolved_mandatory = resolve_mandatory_genes(mandatory_genes, adata)
        self._mandatory_genes = mandatory_genes
        # Recorded before the first read below, which resolves through it, so the
        # whole fit and every later call share one source of expression.
        self._layer = layer

        # Resolve parameters. Compare against None rather than truthiness so that an
        # explicit 0 overrides the config instead of silently falling back to it.
        max_iter = max_iterations if max_iterations is not None else self.config.max_iterations
        patience = (
            early_stopping_patience
            if early_stopping_patience is not None
            else self.config.early_stopping_patience
        )
        use_early_stopping = (
            enable_early_stopping
            if enable_early_stopping is not None
            else self.config.enable_early_stopping
        )
        collect_diagnostics = diagnostics if diagnostics is not None else self.config.diagnostics
        seed = seed if seed is not None else self.config.seed

        # Mirror the BAEConfig bounds, which these arguments bypass.
        if max_iter < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iter}")
        if patience < 1:
            raise ValueError(f"early_stopping_patience must be >= 1, got {patience}")
        if decoder_warmup_epochs < 0:
            raise ValueError(f"decoder_warmup_epochs must be >= 0, got {decoder_warmup_epochs}")

        is_transfer = self._prior_weights is not None
        if is_transfer:
            self._validate_transfer_fit(adata, init_obsm=init_obsm, init_pca=init_pca)
        elif decoder_warmup_epochs:
            raise ValueError(
                "decoder_warmup_epochs only applies to a model built by "
                "BAE.from_reference; there is no prior encoder matrix to warm the "
                "decoder against. Use init_pretrain_epochs with init_pca/init_obsm."
            )

        # Set random seeds for reproducibility. The decoder is *not* re-initialized
        # here: it is rebuilt further down (its input width depends on conditioning
        # and on a warm start that may change latent_dim) and seeded there, so that
        # its weights depend only on `seed` and not on how much RNG the intervening
        # construction happened to consume.
        #
        # Only torch is seeded. Nothing in this package draws from the global NumPy
        # generator — `_boosting.py` is deterministic, and `_stability.py`,
        # `_simulation.py` and `_explorer.py` all use their own `default_rng` — so
        # seeding it changed no result here and only clobbered the caller's stream.
        # `@_isolates_torch_rng` puts the torch stream back when the fit returns.
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Prepare data
        matrix = _expression_matrix(adata, self._layer)
        X_np = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
        X_np = X_np.astype(np.float32)

        # Warn if data doesn't appear standardized. The result also decides whether
        # a PCA warm start needs to center the data (it must not re-transform data
        # that is already z-scored).
        is_standardized = self._check_standardized(X_np)
        if not is_standardized:
            warnings.warn(
                "Input data does not appear to be standardized (mean≈0, std≈1). "
                "Standardization is recommended for optimal performance.",
                UserWarning,
                stacklevel=2,
            )

        # Use full dataset for training (no validation split)
        X_train = self._to_tensor(X_np, self.config.device)
        raw_weights = self._balance_weights(adata, balance_obs)
        sample_weights: torch.Tensor | None = (
            torch.from_numpy(raw_weights).to(self.config.device)
            if raw_weights is not None
            else None
        )

        # Covariance cache for boosting (uses training data only)
        X_train_np = X_train.cpu().numpy()

        # --- Latent state initialization (warm start) ---
        if init_pretrain_epochs < 0:
            raise ValueError(f"init_pretrain_epochs must be >= 0, got {init_pretrain_epochs}")
        z_init_np, init_desc = self._resolve_latent_init(
            adata,
            X_np,
            init_obsm=init_obsm,
            init_pca=init_pca,
            is_standardized=is_standardized,
        )
        if init_pretrain_epochs > 0 and z_init_np is None:
            raise ValueError(
                "init_pretrain_epochs requires a warm start; pass init_obsm or init_pca"
            )

        if z_init_np is not None and z_init_np.shape[1] != self.config.latent_dim:
            new_dim = int(z_init_np.shape[1])
            warnings.warn(
                f"latent_dim was changed from {self.config.latent_dim} to {new_dim} for "
                f"training to match the {new_dim} columns of the supplied representation "
                f"({init_desc}).",
                UserWarning,
                stacklevel=2,
            )
            self.config.latent_dim = new_dim
            # The encoder bakes latent_dim into its linear layer, so it must be rebuilt.
            self.encoder = BAEEncoder(self.n_genes, self.config).to(self.config.device)

        z_init = (
            torch.from_numpy(z_init_np.astype(np.float32)).to(self.config.device)
            if z_init_np is not None
            else None
        )

        # --- Batch covariates: one encoding, two mechanisms it can drive ---
        self._batch_encoding = None
        self._batch_integration_mode = batch_mode
        D_condition: torch.Tensor | None = None
        D_nuisance_np: np.ndarray | None = None
        n_nuisance = 0
        if batch_columns is not None:
            from ._utils import encode_obs_covariates

            self._batch_encoding = encode_obs_covariates(adata, batch_columns)
            if conditions_decoder:
                condition_np = self._batch_encoding.encoded.astype(np.float32)
                D_condition = torch.from_numpy(condition_np).to(self.config.device)
            if regresses_encoder:
                D_nuisance_np = self._batch_encoding.encoded.astype(np.float32)
                n_nuisance = self._batch_encoding.n_columns

        # Always rebuild so repeated fits cannot retain a stale conditioning shape.
        decoder_input = (
            2 * self.config.latent_dim if self.config.split_softmax else self.config.latent_dim
        )
        n_condition = self._batch_encoding.n_columns if conditions_decoder else 0
        self.decoder = BAEDecoder(
            self.n_genes,
            self.config,
            input_dim_override=decoder_input,
            n_covariates=n_condition,
        ).to(self.config.device)

        if seed is not None:
            # Re-seed immediately before the reset so the decoder's initial weights
            # are a function of `seed` alone. Constructing BAEDecoder above draws
            # from the global RNG, and how much it draws depends on the conditioning
            # width; without this, enabling conditioning would perturb the weights of
            # the *unconditioned* part of the model as a side effect.
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            self.decoder.reset_parameters()

        # Build augmented sourcemat for allboost
        if D_nuisance_np is not None:
            sourcemat_aug = np.hstack([X_train_np, D_nuisance_np])
        else:
            sourcemat_aug = X_train_np

        # Build combined mandatory indices (genes + obs covariates)
        allboost_mandatory = _build_allboost_mandatory(resolved_mandatory, n_nuisance, self.n_genes)
        mandatory_ridge = np.zeros(sourcemat_aug.shape[1], dtype=np.float64)
        if n_nuisance:
            mandatory_ridge[self.n_genes :] = nuisance_ridge

        if self.config.boosting_precompute_covcache:
            from ._utils import compute_covariance_cache

            covcache = compute_covariance_cache(sourcemat_aug)
        else:
            covcache = None  # Lazy computation during allboost

        # Optimizer for decoder only
        decoder_optimizer = torch.optim.AdamW(
            self.decoder.parameters(),
            lr=self.config.decoder_lr,
            weight_decay=self.config.decoder_weight_decay,
        )

        # Optional decoder pre-training on the warm-start latent state. This must run
        # after the decoder is final: `fit` re-initializes it under the seed above and
        # may have just rebuilt it, either of which would discard earlier pre-training.
        if init_pretrain_epochs > 0 and z_init is not None:
            self._pretrain_decoder(
                z_init,
                X_train,
                decoder_optimizer,
                init_pretrain_epochs,
                D=D_condition,
                sample_weights=sample_weights,
                verbose=verbose,
                desc="Decoder pre-training (warm start)",
            )

        # --- Phase 1: install the prior encoder matrix and settle the decoder ---
        # Deliberately outside the training loop below. The decoder converges here
        # against the prior programs alone, so this phase's loss is typically lower
        # than the first phase-2 iterations, where the new dimensions switch on and
        # temporarily worsen the reconstruction. Inside the loop it would win
        # checkpoint selection, and `fit` would restore an encoder whose novel
        # columns are all zero — returning the prior unchanged, with no error, and
        # reading as "no novel structure found".
        prior_dims = self._prior_weights.shape[1] if is_transfer else 0
        warmup_loss: float | None = None
        if is_transfer:
            W_start = np.zeros((self.config.latent_dim, self.n_genes), dtype=np.float32)
            W_start[:prior_dims] = self._prior_weights.T
            self.encoder.set_weights(torch.from_numpy(W_start).to(self.config.device))

            if decoder_warmup_epochs > 0:
                self.eval()
                with torch.no_grad():
                    z_prior = self.encoder(X_train)
                warmup_loss = self._pretrain_decoder(
                    z_prior,
                    X_train,
                    decoder_optimizer,
                    decoder_warmup_epochs,
                    D=D_condition,
                    sample_weights=sample_weights,
                    verbose=verbose,
                    desc="Decoder warm-up (prior only)",
                )

        # Early stopping state (based on the checkpoint-selection objective)
        best_selection_loss = float("inf")
        patience_counter = 0
        best_encoder_weights = None
        best_decoder_state = None
        best_batch_weights = None

        # Training loop
        self._training_history = {"train_loss": [], "selection_loss": []}
        # Recorded separately so `train_loss` stays comparable with an ordinary
        # fit and is never mistaken for an iteration of the alternation.
        if warmup_loss is not None:
            self._training_history["warmup_loss"] = [warmup_loss]
        self._training_report = None
        diag_series: dict[str, list] = {f: [] for f in _REPORT_FIELDS if f != "iteration"}
        prev_W: np.ndarray | None = None

        # Progress bar setup
        pbar = tqdm(
            range(max_iter),
            desc="Training BAE",
            disable=not verbose,
            unit="iter",
        )

        for iteration in pbar:
            # STEP 1: Compute boosting targets via gradient step from current z
            # targets = z - lr * ∂L_target/∂z (functional gradient descent).
            # The warm start applies once: it seeds z on the first iteration only.
            diag_stats: dict[str, float] = {}
            targets = self._compute_boosting_targets(
                X_train,
                lr=self.config.target_optim_lr,
                obs_covariates=D_condition,
                sample_weights=sample_weights,
                stats=diag_stats if collect_diagnostics else None,
                z_override=z_init if iteration == 0 else None,
            )

            # STEP 2 (optional): retain the earlier leave-one-out residualization
            # method. The recommended correlation method is already part of the
            # differentiable target objective computed in step 1.
            if self.config.disentanglement == "leave_one_out":
                from ._utils import disentangle_boosting_targets

                targets = disentangle_boosting_targets(
                    targets, standardize=self.config.disentanglement_standardize
                )

            # STEP 3 (optional): Standardize targets for optimal boosting convergence
            if self.config.standardize_targets:
                targets = self._standardize_targets(targets)

            # STEP 4: Reset encoder weights before boosting (rebuild from scratch)
            self.encoder.reset_weights()

            # STEP 5: Fit encoder via boosting to map X → targets. On a transfer
            # model the prior columns are either withheld ("frozen") or boosted
            # from the fixed original matrix as an offset ("anchored").
            fit_targets, beta_init, fit_mandatory = self._transfer_boosting_inputs(
                targets, sourcemat_aug.shape[1], prior_dims, allboost_mandatory
            )
            if fit_targets.shape[1] == 0:
                # Frozen transfer with no additional dimensions: nothing competes
                # for selection, and only the decoder adapts to the new data.
                betamat = np.zeros((0, sourcemat_aug.shape[1]), dtype=np.float64)
            elif covcache is None:
                betamat, covcache = allboost(
                    sourcemat_aug,
                    fit_targets,
                    covcache=covcache,
                    stepno=self.config.boosting_stepno,
                    nu=self.config.boosting_nu,
                    csf=self.config.boosting_csf,
                    independent=self.config.boosting_independent,
                    mandatory_features=fit_mandatory,
                    mandatory_ridge=mandatory_ridge,
                    beta_init=beta_init,
                    return_covcache=True,
                )
            else:
                betamat = allboost(
                    sourcemat_aug,
                    fit_targets,
                    covcache=covcache,
                    stepno=self.config.boosting_stepno,
                    nu=self.config.boosting_nu,
                    csf=self.config.boosting_csf,
                    independent=self.config.boosting_independent,
                    mandatory_features=fit_mandatory,
                    mandatory_ridge=mandatory_ridge,
                    beta_init=beta_init,
                )
            betamat = self._expand_transfer_betamat(betamat, sourcemat_aug.shape[1], prior_dims)

            # Extract gene weights only; obs weights are nuisance (discarded)
            W_genes = betamat[:, : self.n_genes]
            batch_weights = betamat[:, self.n_genes :].copy() if D_nuisance_np is not None else None
            W = torch.from_numpy(W_genes.astype(np.float32)).to(self.config.device)
            self.encoder.set_weights(W)

            # Loss with the new encoder but the old decoder: isolates the
            # boosting step's effect from the decoder's.
            if collect_diagnostics:
                diag_stats["loss_post_boost"] = self._full_recon_loss(
                    X_train, D_condition, sample_weights=sample_weights
                )[0]

            # STEP 6: Update decoder via minibatch SGD
            train_loss = self._update_decoder(
                X_train,
                decoder_optimizer,
                self.config.decoder_updates_per_iteration,
                D_train=D_condition,
                sample_weights=sample_weights,
            )
            self._training_history["train_loss"].append(train_loss)
            selection_loss = self._checkpoint_selection_loss(train_loss, X_train, sample_weights)
            self._training_history["selection_loss"].append(selection_loss)

            if collect_diagnostics:
                diag_series["train_loss"].append(train_loss)
                prev_W = self._collect_diagnostics(
                    diag_series,
                    diag_stats,
                    X=X_train,
                    D=D_condition,
                    sample_weights=sample_weights,
                    targets=targets,
                    prev_W=prev_W,
                )

            # STEP 7: Check early stopping criteria and store the best model. For
            # correlation-constrained fits, selection_loss includes the constraint
            # so restoration cannot quietly select a less-disentangled checkpoint.
            if selection_loss < best_selection_loss:
                best_selection_loss = selection_loss
                patience_counter = 0
                best_encoder_weights = self.encoder.linear.weight.detach().clone()
                best_decoder_state = {k: v.clone() for k, v in self.decoder.state_dict().items()}
                best_batch_weights = batch_weights
            else:
                patience_counter += 1

            postfix = {"train_loss": f"{train_loss:.4f}"}
            if self.config.disentanglement == "correlation":
                postfix["selection_loss"] = f"{selection_loss:.4f}"
            if collect_diagnostics:
                # Two of the 17 diagnostic series are worth watching live: relative
                # encoder weight change (converges toward 0) and how many genes are
                # selected. dW has no predecessor on the first iteration.
                dw = diag_series["weight_change_rel"][-1]
                postfix["dW"] = "n/a" if not np.isfinite(dw) else f"{dw:.3f}"
                postfix["n_sel"] = str(diag_series["n_selected"][-1])
            if use_early_stopping:
                postfix["patience"] = f"{patience_counter}/{patience}"
            pbar.set_postfix(**postfix)

            if use_early_stopping and patience_counter >= patience:
                pbar.set_description("Training BAE (early stop)")
                break

        pbar.close()

        self._latent_init = {"method": init_desc, "pretrain_epochs": int(init_pretrain_epochs)}

        if collect_diagnostics:
            n_iter = len(diag_series["train_loss"])
            int_fields = {"n_selected", "n_selected_per_dim"}
            self._training_report = TrainingReport(
                iteration=np.arange(n_iter, dtype=np.intp),
                **{
                    name: np.asarray(values, dtype=np.intp if name in int_fields else np.float64)
                    for name, values in diag_series.items()
                },
            )

        # Restore best model state
        if best_encoder_weights is not None:
            self.encoder.set_weights(best_encoder_weights)
        if best_decoder_state is not None:
            self.decoder.load_state_dict(best_decoder_state)
        self._batch_weights = best_batch_weights
        self._balance_obs = balance_obs

        self._is_fitted = True

        # Store results in AnnData
        self._store_results(adata)

        # Optional stability selection of the encoder's gene sets. Off by default:
        # it adds a fraction of one fit's cost (see BAE.stability_selection).
        if stability_selection is True:
            warnings.warn(
                "stability_selection=True is deprecated; pass the mode explicitly, "
                'e.g. stability_selection="subsample".',
                FutureWarning,
                stacklevel=2,
            )
            stability_selection = "subsample"
        if stability_selection:
            self.stability_selection(adata, mode=stability_selection, verbose=verbose)
        return self

    @property
    def _conditions_decoder(self) -> bool:
        """Whether the batch covariate is concatenated to the decoder input."""
        return _BATCH_MODES[self._batch_integration_mode][0]

    @property
    def _regresses_encoder(self) -> bool:
        """Whether the batch covariate enters the boosting design as a regressor."""
        return _BATCH_MODES[self._batch_integration_mode][1]

    def _resolve_layer(self, layer: str | None | _FitLayer) -> str | None:
        """Resolve a per-call ``layer`` override against the fit-time layer."""
        return self._layer if isinstance(layer, _FitLayer) else layer

    def transform(self, adata: AnnData, *, layer: str | None | _FitLayer = FIT_LAYER) -> np.ndarray:
        """Transform data to the gene-only latent space.

        Parameters
        ----------
        adata
            AnnData object with the same genes used during fitting. Batch or
            other obs labels are deliberately not required because they are not
            part of the deployable encoder.
        layer
            Where to read expression from. Defaults to the layer the model was
            fitted on, so a model applied to new data reads the same
            representation it was trained on. Pass a name to override, or
            ``None`` to force ``adata.X``.

        Returns
        -------
        Latent representation of shape (n_cells, latent_dim).

        See Also
        --------
        BAE.reconstruct : Reconstruct expression; needs the conditioning columns.
        BAE.fit_transform : Fit and return the latent in one call.

        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self._warn_on_panel_mismatch(adata, "transform")
        matrix = _expression_matrix(adata, self._resolve_layer(layer))
        X = self._to_tensor(matrix, self.config.device)

        z = self.get_latent(X)
        latent = z.cpu().numpy()

        # Update adata. The scaling statistics are *reused*, never re-estimated:
        # new cells must go through the same map the fitted model defines, and a
        # small query set would estimate its own moments badly.
        adata.obsm["X_bae"] = latent
        self._store_scaled_latent(adata, latent)
        return latent

    def reconstruct(
        self, adata: AnnData, *, layer: str | None | _FitLayer = FIT_LAYER
    ) -> np.ndarray:
        """Reconstruct expression using the fitted, conditioned decoder.

        Unlike :meth:`transform`, this method requires the decoder-conditioning
        obs columns used during fitting and rejects unseen categorical levels.

        Parameters
        ----------
        adata
            AnnData object with expression and, when applicable, the fitted
            batch columns, when the mode conditions the decoder.
        layer
            Where to read expression from. Defaults to the layer the model was
            fitted on. Pass a name to override, or ``None`` to force ``adata.X``.

        Returns
        -------
        Reconstructed expression of shape ``(n_cells, n_genes)``.

        See Also
        --------
        BAE.transform : Gene-only latent projection; needs no obs labels.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self._warn_on_panel_mismatch(adata, "reconstruct")
        D: torch.Tensor | None = None
        if self._conditions_decoder:
            from ._utils import transform_obs_covariates

            encoded = transform_obs_covariates(adata, self._batch_encoding)
            D = self._to_tensor(encoded, self.config.device)
        matrix = _expression_matrix(adata, self._resolve_layer(layer))
        X = self._to_tensor(matrix, self.config.device)
        self.eval()
        with torch.no_grad():
            reconstruction, _ = self.forward(X, D)
        return reconstruction.cpu().numpy()

    @staticmethod
    def _balance_weights(adata: AnnData, balance_obs: str | None) -> np.ndarray | None:
        """Inverse-frequency per-cell weights giving each level equal total weight."""
        if balance_obs is None:
            return None
        if balance_obs not in adata.obs.columns:
            raise ValueError(f"Column {balance_obs!r} not found in adata.obs")
        groups = adata.obs[balance_obs]
        if groups.isna().any():
            raise ValueError(f"Column {balance_obs!r} contains missing values")
        counts = groups.value_counts(sort=False)
        if len(counts) < 2:
            raise ValueError(f"Column {balance_obs!r} must contain at least two levels")
        if len(counts) > 50:
            raise ValueError(
                f"Column {balance_obs!r} has {len(counts)} levels; "
                "balance_obs must identify a discrete batch variable"
            )
        weights = {lvl: len(groups) / (len(counts) * c) for lvl, c in counts.items()}
        return groups.map(weights).to_numpy(dtype=np.float32)

    @_isolates_torch_rng()
    def _iteration_support_frequency(
        self,
        adata: AnnData,
        *,
        n_iterations: int,
        seed: int | None,
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Selection frequency over ``n_iterations`` further training iterations.

        Continues the alternating optimization from the fitted state, recording the
        encoder support after each boosting step, then **restores the model**, so the
        call is non-destructive like :meth:`stability_selection` in subsample mode.

        Why this exists: the encoder support does not converge even when the
        reconstruction loss does. On measured data the loss plateaus at ~95% of the
        achievable linear ceiling while consecutive iterations share only about a
        third of their selected genes, with the support autocorrelation still ~0.45
        at lag 100 and no periodicity — a slow random walk over a plateau rather than
        a limit cycle. A single fit therefore reports one arbitrary position on that
        walk. Averaging over iterations targets exactly that variance, which cell
        subsampling cannot reach because it holds the model fixed.

        Frequencies are counted **per latent dimension**, but only after each
        iteration's dimensions are matched to the fitted model's by maximum absolute
        cosine similarity (Hungarian assignment). Anchoring to the fitted model is
        what gives a dimension index a stable meaning: it answers "how often does
        *this* dimension of the model I have select gene g", rather than comparing
        indices that could permute. On measured data the mean matched similarity is
        ~0.84 and the identity permutation is retained in every iteration, so the
        matching is usually a no-op — but it is cheap insurance, and the realized
        quality is returned so a caller can tell when the anchoring did not hold.

        Returns
        -------
        frequency
            Shape ``(n_genes, latent_dim)`` — fraction of recorded iterations in
            which each gene had a nonzero weight in each matched dimension.
        avg_selected
            Mean support size per dimension across the recorded iterations.
        dim_match_quality
            Mean matched absolute cosine similarity to the fitted model.
        coefficients
            ``(conditional mean, conditional sd, sign consistency)``, each
            ``(n_genes, latent_dim)``, from streaming accumulators — the full
            coefficient trace would be ``n_iterations x n_genes x latent_dim``
            floats, which is infeasible at realistic sizes.

        Notes
        -----
        The decoder optimizer is rebuilt here, so AdamW moment estimates start from
        zero rather than continuing those from ``fit``. The alternation is otherwise
        identical to the training loop; ``tests/test_stability.py`` pins that
        equivalence so the two cannot silently diverge.
        """
        from ._utils import resolve_mandatory_genes, transform_obs_covariates

        if n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {n_iterations}")

        if seed is not None:
            torch.manual_seed(seed)

        matrix = _expression_matrix(adata, self._layer)
        X_np = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
        X_np = X_np.astype(np.float32)
        X_train = self._to_tensor(X_np, self.config.device)

        D_condition = None
        if self._conditions_decoder:
            D_condition = self._to_tensor(
                transform_obs_covariates(adata, self._batch_encoding), self.config.device
            )
        raw_weights = self._balance_weights(adata, self._balance_obs)
        sample_weights = (
            torch.from_numpy(raw_weights).to(self.config.device)
            if raw_weights is not None
            else None
        )

        sourcemat_aug = X_np.astype(np.float64)
        n_nuisance = 0
        if self._regresses_encoder:
            D_nuisance = transform_obs_covariates(adata, self._batch_encoding)
            sourcemat_aug = np.hstack([sourcemat_aug, np.asarray(D_nuisance, dtype=np.float64)])
            n_nuisance = self._batch_encoding.n_columns

        resolved_mandatory = resolve_mandatory_genes(self._mandatory_genes, adata)
        allboost_mandatory = _build_allboost_mandatory(resolved_mandatory, n_nuisance, self.n_genes)
        mandatory_ridge = np.zeros(sourcemat_aug.shape[1], dtype=np.float64)
        if n_nuisance:
            mandatory_ridge[self.n_genes :] = self.config.nuisance_ridge

        # Snapshot so the fitted model is unchanged when this returns.
        saved_encoder = self.encoder.linear.weight.detach().clone()
        saved_decoder = {k: v.detach().clone() for k, v in self.decoder.state_dict().items()}

        decoder_optimizer = torch.optim.AdamW(
            self.decoder.parameters(),
            lr=self.config.decoder_lr,
            weight_decay=self.config.decoder_weight_decay,
        )

        from scipy.optimize import linear_sum_assignment

        reference = saved_encoder.detach().cpu().numpy()

        def _unit_rows(matrix: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            return matrix / np.where(norms > 0, norms, 1.0)

        reference_unit = _unit_rows(reference)

        covcache = None
        latent_dim = reference.shape[0]
        prior_dims = self._prior_weights.shape[1] if self._prior_weights is not None else 0
        counts = np.zeros((self.n_genes, latent_dim), dtype=np.float64)
        coef_sum = np.zeros((self.n_genes, latent_dim), dtype=np.float64)
        coef_sq_sum = np.zeros((self.n_genes, latent_dim), dtype=np.float64)
        positive_count = np.zeros((self.n_genes, latent_dim), dtype=np.float64)
        support_sizes: list[np.ndarray] = []
        match_scores: list[float] = []

        try:
            for _ in tqdm(
                range(n_iterations),
                desc="Stability selection (iteration)",
                disable=not verbose,
                unit="run",
            ):
                # Mirrors the fit loop, steps 1-6. Kept in the same order; see
                # `fit` for the authoritative sequence.
                targets = self._compute_boosting_targets(
                    X_train,
                    lr=self.config.target_optim_lr,
                    obs_covariates=D_condition,
                    sample_weights=sample_weights,
                )
                if self.config.disentanglement == "leave_one_out":
                    from ._utils import disentangle_boosting_targets

                    targets = disentangle_boosting_targets(
                        targets, standardize=self.config.disentanglement_standardize
                    )
                if self.config.standardize_targets:
                    targets = self._standardize_targets(targets)

                self.encoder.reset_weights()
                fit_targets, beta_init, fit_mandatory = self._transfer_boosting_inputs(
                    targets, sourcemat_aug.shape[1], prior_dims, allboost_mandatory
                )
                if fit_targets.shape[1] == 0:
                    betamat = np.zeros((0, sourcemat_aug.shape[1]), dtype=np.float64)
                else:
                    result = allboost(
                        sourcemat_aug,
                        fit_targets,
                        covcache=covcache,
                        stepno=self.config.boosting_stepno,
                        nu=self.config.boosting_nu,
                        csf=self.config.boosting_csf,
                        independent=self.config.boosting_independent,
                        mandatory_features=fit_mandatory,
                        mandatory_ridge=mandatory_ridge,
                        beta_init=beta_init,
                        return_covcache=covcache is None,
                    )
                    if covcache is None:
                        betamat, covcache = result
                    else:
                        betamat = result
                betamat = self._expand_transfer_betamat(betamat, sourcemat_aug.shape[1], prior_dims)

                W_genes = betamat[:, : self.n_genes]
                self.encoder.set_weights(
                    torch.from_numpy(W_genes.astype(np.float32)).to(self.config.device)
                )

                # Match this iteration's dimensions to the fitted model's before
                # counting; comparing raw indices would conflate dimensions whenever
                # the optimizer permutes them. On a transfer model the prior
                # dimensions are pinned to their own indices and only the novel
                # block is matched: prior dimensions cannot permute (they are the
                # reference matrix), and matching them anyway would let a novel
                # dimension be assigned into a prior slot.
                signed = _unit_rows(W_genes) @ reference_unit.T
                similarity = np.abs(signed)
                if prior_dims:
                    sub_rows, sub_cols = linear_sum_assignment(
                        -similarity[prior_dims:, prior_dims:]
                    )
                    pinned = np.arange(prior_dims)
                    rows = np.concatenate([pinned, sub_rows + prior_dims])
                    cols = np.concatenate([pinned, sub_cols + prior_dims])
                    # Reported over the novel block only: the pinned prior
                    # dimensions score 1.0 against themselves and would inflate it.
                    novel = similarity[rows[prior_dims:], cols[prior_dims:]]
                    match_scores.append(float(novel.mean()) if novel.size else 1.0)
                else:
                    rows, cols = linear_sum_assignment(-similarity)
                    match_scores.append(float(similarity[rows, cols].mean()))

                selected = np.abs(W_genes) > 0  # (latent_dim, n_genes)
                if prior_dims:
                    # `|W| > 0` is tautologically true across the prior support:
                    # frozen columns never change, and anchored columns start from
                    # the anchor's non-zero coefficients. Counting it would report
                    # frequency 1.0 for every prior gene and read as overwhelming
                    # evidence. A prior dimension's selection is therefore movement
                    # away from the anchor — identically zero under "frozen", which
                    # is the honest answer for a dimension that cannot vary.
                    selected[:prior_dims] = W_genes[:prior_dims] != self._prior_weights.T
                counts[:, cols] += selected[rows].T
                support_sizes.append(selected.sum(axis=1)[rows][np.argsort(cols)])

                # Coefficients are sign-aligned to the fitted encoder before being
                # accumulated: a dimension whose best match is negative would
                # otherwise cancel itself out under averaging.
                flip = np.sign(signed[rows, cols])
                flip[flip == 0] = 1.0
                aligned = (W_genes[rows] * flip[:, None]).T  # (n_genes, matched dim)
                coef_sum[:, cols] += aligned
                coef_sq_sum[:, cols] += aligned**2
                positive_count[:, cols] += aligned > 0

                self._update_decoder(
                    X_train,
                    decoder_optimizer,
                    self.config.decoder_updates_per_iteration,
                    D_train=D_condition,
                    sample_weights=sample_weights,
                )
        finally:
            self.encoder.set_weights(saved_encoder)
            self.decoder.load_state_dict(saved_decoder)

        quality = float(np.mean(match_scores)) if match_scores else float("nan")
        if np.isfinite(quality) and quality < 0.5:
            warnings.warn(
                f"Latent dimensions matched the fitted model poorly across iterations "
                f"(mean |cosine| {quality:.2f}). Per-dimension frequencies are unreliable "
                "here; use frequency.max(axis=1) for the flat union instead.",
                UserWarning,
                stacklevel=3,
            )
        from ._stability import _coefficient_statistics

        cond_mean, cond_sd, sign_consistency = _coefficient_statistics(
            coef_sum, coef_sq_sum, positive_count, counts
        )
        return (
            counts / n_iterations,
            np.mean(np.vstack(support_sizes), axis=0),
            quality,
            (cond_mean, cond_sd, sign_consistency),
        )

    def stability_selection(
        self,
        adata: AnnData,
        *,
        mode: str = "iteration",
        n_runs: int = 300,
        subsample_frac: float = 0.5,
        threshold: float = 0.7,
        seed: int | None = None,
        n_subsamples: int | None = None,
        n_iterations: int | None = None,
        verbose: bool = True,
    ):
        """Stability-select genes for each latent dimension of the fitted model.

        The model is frozen. Its functional-gradient targets ``z*`` are recomputed
        once, then :func:`~structboost.stability_selection` re-runs the encoder's
        boosting problem on many subsamples of the cells and records, per latent
        dimension, how often each gene is selected. This costs a fraction of one
        fit and needs no refitting — a single converged model is enough.

        Targets are ``z*`` rather than the latent codes: the codes are a sparse
        linear function of the already-selected genes, so selecting them back is
        nearly circular, whereas ``z*`` carries the decoder's full reconstruction
        gradient. The resampled problem reuses the model's boosting
        hyperparameters, mandatory genes, and (if fitted) nuisance regressors,
        conditioning and balancing, so it matches the selection the encoder solved.

        Scope. This measures stability under *cell resampling, conditional on the
        learned representation*. It does not capture the variability from
        re-initializing and refitting the autoencoder to a different local optimum,
        which full refits would. On simulated data the per-gene selection
        frequencies track a full-refit gold standard closely (correlation ~0.95),
        so it is a good, far cheaper proxy; but where the model has several
        competing optima, high-stakes marker claims may still warrant a handful of
        full refits as a cross-check.

        Parameters
        ----------
        adata
            The data the model was fitted on (same genes; obs columns required only
            if the model used conditioning, nuisance or balancing covariates).
        mode
            Which source of instability to measure.

            ``"iteration"`` (default)
                Selection frequency across ``n_runs`` further training
                iterations. Answers "would these genes still be selected if the
                optimizer had stopped somewhere else on its loss plateau?" The two
                are complementary, not alternatives: they measure different variance
                sources, and on measured data the second is the larger one, because
                the encoder support keeps moving long after the loss has converged.

                Costs ``n_runs`` further training steps and needs no refitting or
                subsampling. It is the default because it is the mode that measures
                the variance that actually dominates: on measured data the
                reconstruction loss converges while the encoder support keeps
                moving. On simulated data with exact ground truth it also gives the
                lowest false-discovery rate of the three readouts (0.26, against
                0.31 for subsampling and 0.38 for a single fit), and its
                frequencies are empirically well calibrated — genes selected in
                90-100% of iterations are markers 88% of the time.

                Provides **no formal error control**; see
                :class:`~structboost.StabilitySelectionResult`.
            ``"subsample"``
                Meinshausen-Bühlmann cell subsampling with the model frozen. Answers
                "would these genes still be selected on a different sample of cells?"

                .. warning::
                   ``expected_false_positives`` reports the Meinshausen-Bühlmann
                   bound, and on simulated data where the truth is known that bound
                   is **violated by roughly an order of magnitude** (mean realized
                   2.57 false positives per dimension against a bound of 0.20;
                   satisfied in 15 of 60 dimensions). The likely cause is a scope
                   error rather than an implementation bug: the bound assumes
                   exchangeable subsamples of an inference problem fixed in advance,
                   whereas the targets ``z*`` here come from a model fitted on the
                   same cells being resampled. Treat the number as a diagnostic,
                   not a guarantee. This mode is retained for investigation.
        n_runs
            How many resampling runs to average over — cell subsamples in subsample
            mode, further training iterations in iteration mode. One name because it
            plays the same role in both.

            The default of 300 is set by iteration mode's requirement: the support
            autocorrelation decays slowly (still ~0.45 at lag 100 on measured data),
            so short windows give highly correlated, near-duplicate samples.
            Subsample mode is well served by fewer — 100 was this method's previous
            default — so pass ``n_runs=100`` there if the threefold cost matters.
        subsample_frac
            Subsample mode only. ``0.5`` is the value Meinshausen & Bühlmann (2010)
            derive the bound for.
        threshold
            Selection-frequency cutoff for the stable support, in both modes.
        seed
            Seeds the subsampling RNG (subsample mode) or torch (iteration mode).
        n_subsamples, n_iterations
            Deprecated aliases for ``n_runs``, kept so existing calls keep working.
        verbose
            Show a progress bar. On by default: the iteration mode's ``n_runs``
            defaults to 300 further training steps, which is a long silence.

        Returns
        -------
        StabilitySelectionResult
            Subsample mode stores ``adata.varm["BAE_selection_frequency"]``;
            iteration mode stores ``adata.varm["BAE_iteration_frequency"]``. Both
            are ``(n_genes, latent_dim)``: iteration mode matches each iteration's
            dimensions to the fitted model's before counting, so a dimension index
            keeps its meaning — see ``dim_match_quality``, and fall back to
            ``frequency.max(axis=1)`` when it is low. Both write a summary under
            ``adata.uns["bae"]["stability_selection"]``, tagged with the mode that
            produced it.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if mode not in {"subsample", "iteration"}:
            raise ValueError(f"mode must be 'subsample' or 'iteration', got {mode!r}")
        for old_name, old_value in (("n_subsamples", n_subsamples), ("n_iterations", n_iterations)):
            if old_value is not None:
                warnings.warn(
                    f"{old_name} is deprecated; use n_runs, which names the same "
                    "quantity in both modes.",
                    FutureWarning,
                    stacklevel=2,
                )
                n_runs = old_value
        if n_runs < 1:
            raise ValueError(f"n_runs must be >= 1, got {n_runs}")
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")

        if mode == "iteration":
            from ._stability import StabilitySelectionResult

            frequency, avg_selected, quality, coefficients = self._iteration_support_frequency(
                adata, n_iterations=n_runs, seed=seed, verbose=verbose
            )
            result = StabilitySelectionResult(
                frequency=frequency,
                stable_support=frequency >= threshold,
                threshold=float(threshold),
                avg_selected=avg_selected,
                # Training iterations are neither independent nor exchangeable, so
                # the Meinshausen-Bühlmann bound does not apply. Deliberately NaN
                # rather than a number that would look like error control.
                expected_false_positives=np.full(frequency.shape[1], np.nan),
                n_subsamples=0,
                subsample_frac=float("nan"),
                mode="iteration",
                n_iterations=int(n_runs),
                dim_match_quality=quality,
                coefficient_cond_mean=coefficients[0],
                coefficient_sd=coefficients[1],
                sign_consistency=coefficients[2],
            )
            adata.varm["BAE_iteration_frequency"] = result.frequency
            uns = adata.uns.setdefault("bae", {})
            uns["stability_selection"] = {
                "mode": "iteration",
                "threshold": result.threshold,
                "n_iterations": result.n_iterations,
                "n_stable_per_dim": result.stable_support.sum(axis=0).astype(np.intp),
                "avg_selected_per_dim": result.avg_selected,
                "dim_match_quality": quality,
                "expected_false_positives_per_dim": result.expected_false_positives,
            }
            return result

        from ._stability import stability_selection as _run_stability
        from ._utils import resolve_mandatory_genes, transform_obs_covariates

        matrix = _expression_matrix(adata, self._layer)
        X_np = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
        X_np = X_np.astype(np.float32)
        X_t = torch.from_numpy(X_np).to(self.config.device)

        # Rebuild the same covariate context the encoder was fitted under, so the
        # resampled selection problem is the one the model actually solved.
        D_condition = None
        if self._conditions_decoder:
            D_condition = self._to_tensor(
                transform_obs_covariates(adata, self._batch_encoding), self.config.device
            )
        weights_np = self._balance_weights(adata, self._balance_obs)
        weights_t = (
            torch.from_numpy(weights_np).to(self.config.device) if weights_np is not None else None
        )

        targets = self._compute_boosting_targets(
            X_t,
            lr=self.config.target_optim_lr,
            obs_covariates=D_condition,
            sample_weights=weights_t,
        )

        sourcemat = X_np.astype(np.float64)
        n_nuisance = 0
        if self._regresses_encoder:
            D_nuisance = transform_obs_covariates(adata, self._batch_encoding)
            sourcemat = np.hstack([sourcemat, np.asarray(D_nuisance, dtype=np.float64)])
            n_nuisance = self._batch_encoding.n_columns

        resolved_mandatory = resolve_mandatory_genes(self._mandatory_genes, adata)
        mandatory = _build_allboost_mandatory(resolved_mandatory, n_nuisance, self.n_genes)
        mandatory_ridge = np.zeros(sourcemat.shape[1], dtype=np.float64)
        if n_nuisance:
            mandatory_ridge[self.n_genes :] = self.config.nuisance_ridge

        result = _run_stability(
            sourcemat,
            targets,
            n_genes=self.n_genes,
            mandatory_features=mandatory,
            mandatory_ridge=mandatory_ridge,
            n_subsamples=n_runs,
            subsample_frac=subsample_frac,
            threshold=threshold,
            stepno=self.config.boosting_stepno,
            nu=self.config.boosting_nu,
            csf=self.config.boosting_csf,
            independent=self.config.boosting_independent,
            seed=seed,
            verbose=verbose,
        )

        adata.varm["BAE_selection_frequency"] = result.frequency
        uns = adata.uns.setdefault("bae", {})
        uns["stability_selection"] = {
            "mode": "subsample",
            "threshold": result.threshold,
            "n_subsamples": result.n_subsamples,
            "subsample_frac": result.subsample_frac,
            "n_stable_per_dim": result.stable_support.sum(axis=0).astype(np.intp),
            "avg_selected_per_dim": result.avg_selected,
            "expected_false_positives_per_dim": result.expected_false_positives,
        }
        return result

    def apply_encoder(
        self,
        weights: np.ndarray,
        adata: AnnData | None = None,
        *,
        preserve_prior: bool = True,
    ) -> BAE:
        """Install aggregated encoder weights, replacing the fitted ones.

        Deliberately separate from :meth:`stability_selection`, which stays
        non-destructive. A diagnostic that silently swapped the encoder would make
        ``fit()`` followed by a reliability check produce a different model than
        ``fit()`` alone, and would compound if called twice. It is also not a
        strictly better encoder but a **choice**: ``"masked_cond_mean"`` buys
        precision (0.74 vs 0.62 for the fitted encoder on simulated data) at the
        cost of recall (0.27 vs 0.31), and that trade belongs to the caller — tune
        it with ``stability_selection(threshold=...)``.

        The decoder is left untouched.
        :meth:`~structboost.StabilitySelectionResult.stable_encoder` preserves the
        latent scale, and on simulated data the aggregate *improved* reconstruction
        relative to the fitted encoder (59% of the linear ceiling against 55%), so
        no refit is required.

        Parameters
        ----------
        weights
            Shape ``(n_genes, latent_dim)`` — the orientation returned by
            ``stable_encoder`` and stored in ``adata.varm``.
        adata
            If given, refresh ``varm["BAE_encoder_weights"]`` and ``obsm["X_bae"]``
            so the stored results match the installed encoder rather than the
            superseded one.
        preserve_prior
            On a model built by :meth:`from_reference`, keep the transferred block
            rather than taking it from ``weights``. Default ``True``, because the
            obvious call — installing
            :meth:`~structboost.StabilitySelectionResult.stable_encoder` — would
            otherwise **delete the transferred programs**: that estimator zeroes
            every entry outside the stable support, and under
            ``prior_mode="frozen"`` the prior columns have selection frequency
            zero by construction (they cannot vary, so there is nothing to be
            stable about). The result would be an encoder whose prior block is
            all zeros, with no error raised. Pass ``False`` only to overwrite the
            transferred programs deliberately.

        Returns
        -------
        Self, for chaining.

        Examples
        --------
        >>> res = model.stability_selection(adata)
        >>> model.apply_encoder(res.stable_encoder(), adata)  # doctest: +SKIP
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        weights = np.asarray(weights)
        expected = (self.n_genes, self.config.latent_dim)
        if weights.shape != expected:
            raise ValueError(
                f"weights must have shape {expected} (n_genes, latent_dim), got {weights.shape}"
            )
        if preserve_prior and self._prior_weights is not None:
            weights = weights.copy()
            weights[:, : self.n_prior_dims] = self.fitted_prior_weights
        if not np.isfinite(weights).all():
            raise ValueError("weights must be finite")

        previous = self.encoder.linear.weight.detach().clone()
        self.encoder.set_weights(
            torch.from_numpy(np.ascontiguousarray(weights.T, dtype=np.float32)).to(
                self.config.device
            )
        )
        if adata is not None:
            self._warn_if_integration_degraded(adata, previous)
            adata.varm["BAE_encoder_weights"] = weights.astype(np.float32)
            matrix = _expression_matrix(adata, self._layer)
            latent = self.get_latent(self._to_tensor(matrix, self.config.device)).cpu().numpy()
            adata.obsm["X_bae"] = latent
            # The encoder changed, so the scaling statistics belong to a
            # superseded model. Re-estimating here is what stops X_bae_scaled
            # from silently describing weights that are no longer installed.
            if self._prior_weights is not None:
                self._refit_latent_scaling(latent)
                self._store_scaled_latent(adata, latent)
            adata.uns.setdefault("bae", {})["encoder_source"] = "aggregated"
        return self

    def transfer_diagnostics(self, adata: AnnData) -> dict[str, object]:
        """Transfer diagnostics evaluated on ``adata``, which need not be the fit data.

        ``fit`` writes these for the cells it trained on, where
        ``novel_variance_share`` is positive by construction: k free dimensions
        reduce in-sample reconstruction error whether or not the data contains
        anything the prior programs missed.

        Passing **held-out cells** is what turns the number into evidence. Capacity
        that merely fits noise does not generalize, so a novel dimension carrying
        real structure keeps its share out of sample while one that does not
        collapses. This needs nothing but the model and your own data — no access
        to the reference dataset, and no marker ground truth — which matters
        because a prior encoder matrix is often all that is shared.

        Parameters
        ----------
        adata
            Cells to evaluate on, over the same gene panel the model was aligned
            to. Hold these out of ``fit`` for an out-of-sample reading.

        Returns
        -------
        The same mapping ``fit`` stores in ``adata.uns["bae_transfer"]``.

        Raises
        ------
        ValueError
            If the model carries no prior matrix, or the panel does not match.
        """
        if self._prior_weights is None:
            raise ValueError(
                "transfer_diagnostics is only defined for a model built by "
                "BAE.from_reference; an ordinary fit has no prior/novel split."
            )
        if adata.n_vars != self.n_genes:
            raise ValueError(
                f"adata has {adata.n_vars} genes but the model was aligned to "
                f"{self.n_genes}. Pass cells over the same panel."
            )
        X = self._to_tensor(_expression_matrix(adata, self._layer), self.config.device)
        weights = self.encoder.linear.weight.detach().cpu().numpy().T
        return self._transfer_diagnostics(adata, X, weights)

    #: obsm key for the per-dimension standardized latent, written only by
    #: transfer models. See ``_refit_latent_scaling``.
    SCALED_LATENT_KEY: str = "X_bae_scaled"

    def _refit_latent_scaling(self, latent: np.ndarray) -> None:
        """Re-estimate the per-dimension standardization from a training latent.

        Called whenever the *encoder* changes on the data it was fitted to — at
        the end of ``fit`` and on ``apply_encoder``. Not called by ``transform``:
        mapping new cells must reuse these statistics, or the transformation
        would differ per call and be poorly estimated on small held-out sets.
        """
        mean = latent.mean(axis=0)
        scale = latent.std(axis=0)
        scale = np.where(scale < BAE._STD_EPS, 1.0, scale)
        self._latent_scaling = {"mean": mean, "scale": scale}

    def _store_scaled_latent(self, adata: AnnData, latent: np.ndarray) -> None:
        """Write the standardized latent alongside the raw one, transfers only.

        ``obsm["X_bae"]`` stays exactly ``X @ varm["BAE_encoder_weights"]``, which
        is what makes the encoder auditable and the frozen-prior guarantee
        checkable end to end. It is deliberately *not* overwritten here.

        A transfer's two blocks are on incomparable scales: hand-authored prior
        coefficients are written on a human scale, boosted ones land wherever the
        gradient targets put them. Measured on the Tasic transfer, the prior
        dimensions had a median latent SD of 9.98 against 0.043 for the novel
        ones — a 232x gap, and because distance is squared the novel block
        contributed ~0.00% of the total. Every Euclidean consumer inherits that:
        neighbour graphs, Leiden, kNN, kBET/LISI. This key is the one to hand
        those tools.

        Centering also cannot be folded back into the weights, since the encoder
        has no bias term — another reason this is a separate key rather than a
        rescaled ``X_bae``.
        """
        if self._prior_weights is None or self._latent_scaling is None:
            return
        scaling = self._latent_scaling
        adata.obsm[BAE.SCALED_LATENT_KEY] = (latent - scaling["mean"]) / scaling["scale"]
        info = adata.uns.setdefault("bae_transfer", {})
        info["latent_mean"] = scaling["mean"]
        info["latent_scale"] = scaling["scale"]

    def _transfer_diagnostics(
        self, adata: AnnData, X: torch.Tensor, encoder_weights: np.ndarray
    ) -> dict[str, object]:
        """Summary of a transfer fit, written to ``adata.uns["bae_transfer"]``.

        ``novel_variance_share`` is the rise in reconstruction MSE when the added
        dimensions are zeroed. **On the fitting data it is not evidence of novel
        biology:** k free dimensions always reduce in-sample reconstruction error,
        so the number is positive even when the target contains nothing the prior
        programs missed — measured on gene-shuffled data with no recoverable
        structure at all, it still reads +0.002 to +0.008 as k grows from 1 to 4.
        It also grows with k under the null, so raw values are not comparable
        across different ``n_additional_dims``.

        Use :meth:`transfer_diagnostics` on **held-out cells** to make it
        interpretable; that requires only the model and your own data. Where the
        reference dataset is also available, a transfer onto held-out *reference*
        cells gives a second, stricter null (see
        ``paper_analysis/a9_transfer_null.py``).

        ``novel_variance_share_per_dim`` is the same quantity computed by dropping
        one novel dimension at a time. The entries do not sum to the aggregate:
        the dimensions are not orthogonal, so structure carried by several of them
        is counted in each. A dimension near zero is a candidate for reducing
        ``n_additional_dims``.

        Support sizes are reported separately for the prior and novel blocks
        because the union is dominated by the prior: a sparsity check against the
        combined count would be measuring the reference matrix, not this fit.
        """
        assert self._prior_weights is not None
        prior_dims = self._prior_weights.shape[1]

        covariates = None
        if self._conditions_decoder:
            from ._utils import transform_obs_covariates

            covariates = self._to_tensor(
                transform_obs_covariates(adata, self._batch_encoding), self.config.device
            )

        def reconstruction_mse(weights: torch.Tensor) -> float:
            saved = self.encoder.linear.weight.detach().clone()
            try:
                self.encoder.set_weights(weights)
                self.eval()
                with torch.no_grad():
                    z = self.encoder(X)
                    h = self.split_softmax_layer(z) if self.split_softmax_layer else z
                    return float(nn.functional.mse_loss(self.decoder(h, covariates), X))
            finally:
                self.encoder.set_weights(saved)

        full = self.encoder.linear.weight.detach().clone()
        prior_only = full.clone()
        prior_only[prior_dims:] = 0.0
        mse_full = reconstruction_mse(full)
        mse_prior_only = reconstruction_mse(prior_only)

        # Leave-one-out per novel dimension: the aggregate share cannot say which
        # added dimension is doing the work, and a dimension contributing ~0 is a
        # candidate for reducing `n_additional_dims`. These do not sum to the
        # aggregate — the dimensions are not orthogonal, so shared structure is
        # counted by every dimension carrying it.
        per_dim = []
        for j in range(prior_dims, full.shape[0]):
            dropped = full.clone()
            dropped[j] = 0.0
            per_dim.append(reconstruction_mse(dropped) - mse_full)

        prior_block = encoder_weights[:, :prior_dims]
        novel_block = encoder_weights[:, prior_dims:]
        return {
            **{k: v for k, v in self._prior_info.items() if k != "reference_panel"},
            "prior_mode": self.config.prior_mode,
            "n_selected_prior": int((np.abs(prior_block) > 0).any(axis=1).sum()),
            "n_selected_novel": int((np.abs(novel_block) > 0).any(axis=1).sum()),
            "n_selected_novel_per_dim": [int(c) for c in (np.abs(novel_block) > 0).sum(axis=0)],
            "variance_explained": 1.0 - mse_full,
            "variance_explained_prior_only": 1.0 - mse_prior_only,
            "novel_variance_share": mse_prior_only - mse_full,
            "novel_variance_share_per_dim": per_dim,
            # Compared at float32, the encoder's storage precision. A float64
            # prior read from Parquet generally is NOT float32-representable
            # (2.6 becomes 2.5999999046325684), so comparing at float64 reports
            # drift for every such prior even though nothing moved.
            "prior_weights_unchanged": bool(
                np.array_equal(
                    prior_block.astype(np.float32),
                    self._prior_weights.astype(np.float32),
                )
            ),
        }

    def _warn_if_integration_degraded(
        self, adata: AnnData, previous_weights: torch.Tensor, tolerance: float = 0.05
    ) -> None:
        """Warn when an aggregated encoder reintroduces covariate signal.

        Batch integration is a property of the encoder *as a whole*: the nuisance
        regressors absorb covariate signal during boosting, and the surviving gene
        coefficients partially cancel each other's residual. Thresholding that
        vector is therefore **not a covariate-neutral operation** — dropping
        low-frequency genes can remove exactly the terms providing the cancellation.

        Whether that actually happens depends on the fit. A first, underpowered
        check (60 training iterations, 10 stability runs) showed the
        ``"masked_cond_mean"`` encoder raising the maximum per-dimension batch R^2
        from 0.021 to 0.281. A better-powered rerun (300 iterations, 100 runs) did
        **not** reproduce it: every aggregate slightly *lowered* batch R^2 relative
        to the fitted encoder (0.228 fitted, 0.200 masked, 0.166 unmasked). The
        likely reading is that a coarse frequency estimate from few runs produces a
        crude mask, not that masking is inherently unsafe.

        Which is exactly why this measures the change per call instead of warning
        unconditionally: the hazard is real but conditional, so a caller whose
        integration survives is never nagged.
        """
        encoding = self._batch_encoding
        if encoding is None:
            return
        from ._utils import transform_obs_covariates

        try:
            design = transform_obs_covariates(adata, encoding)
        except Exception:  # noqa: BLE001 - diagnostic only; never block installation
            return

        def covariate_r2(weights: torch.Tensor) -> float:
            saved = self.encoder.linear.weight.detach().clone()
            try:
                self.encoder.set_weights(weights)
                matrix = _expression_matrix(adata, self._layer)
                z = self.get_latent(self._to_tensor(matrix, self.config.device)).cpu().numpy()
            finally:
                self.encoder.set_weights(saved)
            centered = z - z.mean(axis=0, keepdims=True)
            fitted = design @ np.linalg.lstsq(design, centered, rcond=None)[0]
            ss_total = (centered**2).sum(axis=0)
            ss_residual = ((centered - fitted) ** 2).sum(axis=0)
            # `where=` has to guard the division itself. Writing
            # `np.divide(1.0 - ss_residual / ss_total, 1.0, where=ss_total > 0)`
            # evaluates the inner `/` eagerly over every column, so a dead latent
            # dimension still divides by zero and warns before `where` discards
            # the result. Same values, no warning.
            has_variance = ss_total > 0
            ratio = np.divide(
                ss_residual, ss_total, out=np.zeros_like(ss_total), where=has_variance
            )
            r2 = np.where(has_variance, 1.0 - ratio, 0.0)
            return float(np.max(r2))

        before = covariate_r2(previous_weights)
        after = covariate_r2(self.encoder.linear.weight.detach().clone())
        if after - before > tolerance:
            warnings.warn(
                f"The installed encoder reintroduced covariate signal into the latent "
                f"space: maximum per-dimension R^2 against the conditioned covariates "
                f"rose from {before:.3f} to {after:.3f}. Integration is a property of the "
                "whole coefficient vector, so thresholding it can remove terms that were "
                "cancelling residual covariate effects. Consider a lower "
                "stability_selection(threshold=...), which keeps more of them, and "
                'check adata.uns["bae"]["latent_obs_r2_per_dim"].',
                UserWarning,
                stacklevel=3,
            )

    def transform_splitsoftmax(self, adata: AnnData) -> np.ndarray:
        """Transform data to split-softmax representation.

        Computes the split-softmax compositional representation h in Delta^{2d-1}
        from the encoder output z. Each latent dimension z_i is paired with its
        negation -z_i (interleaved) and softmax-normalized:

            h = softmax((z_1, -z_1, ..., z_d, -z_d))

        Can be called on models trained with or without split_softmax=True.
        When split_softmax was not enabled during training, a warning is emitted
        and the transformation is applied post-hoc.

        Parameters
        ----------
        adata
            AnnData object.

        Returns
        -------
        Split-softmax representation of shape (n_cells, 2 * latent_dim).

        Raises
        ------
        RuntimeError
            If model is not fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        ssm = self.split_softmax_layer
        if ssm is None:
            warnings.warn(
                "Model was not trained with split_softmax=True. "
                "Applying split-softmax transformation post-hoc.",
                UserWarning,
                stacklevel=2,
            )
            ssm = SplitSoftmax()

        X = self._to_tensor(_expression_matrix(adata, self._layer), self.config.device)
        self.eval()
        with torch.no_grad():
            z = self.encoder(X)
            h = ssm(z)

        h_np = h.cpu().numpy()
        adata.obsm["X_bae_splitsoftmax"] = h_np

        # Store clipped split-softmax encoder weights
        adata.varm["bae_program_weights"] = self.get_splitsoftmax_encoder_weights(
            clip_negative=True, as_numpy=True
        )

        return h_np

    def fit_transform(self, adata: AnnData, **fit_kwargs) -> np.ndarray:
        """Fit model and return latent representation.

        Parameters
        ----------
        adata
            AnnData object.
        **fit_kwargs
            Arguments passed to fit().

        Returns
        -------
        Latent representation.
        """
        self.fit(adata, **fit_kwargs)
        return adata.obsm["X_bae"]

    def _reconstruction_stats(self, adata: AnnData) -> tuple[np.ndarray, float]:
        """Per-cell reconstruction MSE, and the fraction of variance explained.

        Streams over the cells in ``batch_size`` chunks. The obvious
        implementation — reconstruct everything, densify ``adata.X``, subtract —
        holds two ``(n_cells, n_genes)`` arrays at once, which is ~8 GB at 100,000
        cells by 20,000 genes and is allocated at the very *end* of an otherwise
        successful fit. Streaming keeps the peak at one chunk.

        Returns
        -------
        residual
            Mean squared error per cell, shape ``(n_cells,)``.
        variance_explained
            ``1 - SSE/SST``, where SST is computed about the per-gene mean. On the
            z-transformed input BAE expects, SST per element is ~1, so this is
            close to ``1 - MSE``; computing it properly keeps the number honest if
            the input was not standardized.
        """
        from ._utils import transform_obs_covariates

        D_all = None
        if self._conditions_decoder:
            D_all = transform_obs_covariates(adata, self._batch_encoding)

        X = _expression_matrix(adata, self._layer)
        col_means = np.asarray(X.mean(axis=0)).ravel()
        means_t = torch.from_numpy(col_means.astype(np.float32)).to(self.config.device)

        residual = np.empty(adata.n_obs, dtype=np.float64)
        ss_error = 0.0
        ss_total = 0.0
        step = max(int(self.config.batch_size), 1)

        self.eval()
        with torch.no_grad():
            for start in range(0, adata.n_obs, step):
                stop = min(start + step, adata.n_obs)
                chunk = X[start:stop]
                x = self._to_tensor(chunk, self.config.device)
                d = (
                    self._to_tensor(D_all[start:stop], self.config.device)
                    if D_all is not None
                    else None
                )
                recon, _ = self.forward(x, d)
                squared_error = (recon - x).square()
                residual[start:stop] = squared_error.mean(dim=1).cpu().numpy()
                ss_error += float(squared_error.sum())
                ss_total += float((x - means_t).square().sum())

        explained = 1.0 - ss_error / ss_total if ss_total > 0 else float("nan")
        return residual, explained

    def _store_results(self, adata: AnnData) -> None:
        """Store results in AnnData (scverse convention).

        Two of the recorded metrics answer different questions and are easy to
        confuse:

        ``latent_obs_r2_per_dim``
            Fraction of each latent dimension's variance explained by the
            conditioned obs columns. This is the **integration** metric — it says
            whether the covariate signal is gone from the representation. Values
            near zero are the goal; a single high dimension is a residual
            covariate axis worth inspecting.
        ``reconstruction_loss_by_obs``
            Reconstruction MSE per group. This is a **fairness** metric — it says
            whether the model fits all groups comparably, not whether it
            integrated them. A group can be reconstructed poorly in a perfectly
            integrated model, and vice versa.

        ``variance_explained`` anchors the raw losses, which are otherwise
        unreadable: on the z-transformed input BAE expects, an MSE of 1.0 is what
        predicting zero everywhere scores. Compare it with
        :func:`~structboost.linear_ceiling` for the achievable maximum.
        """
        X = self._to_tensor(_expression_matrix(adata, self._layer), self.config.device)

        # Recorded so a fitted model is itself a usable prior for a later transfer:
        # `from_reference` needs gene identifiers, which the weight matrix lacks.
        self._var_names = np.asarray(adata.var_names, dtype=object)

        # Latent embedding: adata.obsm["X_bae"] — purely gene-based
        z = self.get_latent(X)
        latent = z.cpu().numpy()
        adata.obsm["X_bae"] = latent
        if self._prior_weights is not None:
            self._refit_latent_scaling(latent)

        # Encoder weights: adata.varm["BAE_encoder_weights"]
        # Shape: (n_genes, latent_dim) — transposed from (latent_dim, n_genes)
        encoder_weights = self.encoder.linear.weight.detach().cpu().numpy().T
        adata.varm["BAE_encoder_weights"] = encoder_weights

        # Metadata: adata.uns["bae"]
        uns_dict: dict = {
            "latent_dim": self.config.latent_dim,
            "is_fitted": self._is_fitted,
            "training_history": self._training_history,
            "latent_init": self._latent_init,
            "disentanglement": self.config.disentanglement,
        }
        # Which matrix the fit read. Absent when it was `adata.X`: `uns` cannot
        # hold None, and no key is the honest encoding of "the default".
        if self._layer is not None:
            uns_dict["layer"] = self._layer
        if self.config.disentanglement == "correlation":
            uns_dict["disentanglement_lambda"] = self.config.disentanglement_lambda
        elif self.config.disentanglement == "leave_one_out":
            uns_dict["disentanglement_standardize"] = self.config.disentanglement_standardize
        if self._training_report is not None:
            uns_dict["training_report"] = self._training_report.to_dict()
        if hasattr(self, "_mandatory_genes") and self._mandatory_genes is not None:
            uns_dict["mandatory_genes"] = _mandatory_genes_for_uns(self._mandatory_genes)
        uns_dict["batch_integration_mode"] = self._batch_integration_mode
        if self._batch_encoding is not None:
            uns_dict["batch_key"] = self._batch_encoding.obs_columns
            uns_dict["batch_columns"] = self._batch_encoding.encoded_columns
        if self._batch_weights is not None:
            uns_dict["batch_weights"] = self._batch_weights
            uns_dict["nuisance_ridge"] = self.config.nuisance_ridge
        if self._balance_obs is not None:
            uns_dict["balance_obs"] = self._balance_obs
        diagnostic_encoding = self._batch_encoding
        if diagnostic_encoding is not None:
            from ._utils import transform_obs_covariates

            design = transform_obs_covariates(adata, diagnostic_encoding)
            centered_z = adata.obsm["X_bae"] - adata.obsm["X_bae"].mean(axis=0, keepdims=True)
            fitted_z = design @ np.linalg.lstsq(design, centered_z, rcond=None)[0]
            ss_total = (centered_z**2).sum(axis=0)
            ss_residual = ((centered_z - fitted_z) ** 2).sum(axis=0)
            latent_r2 = np.divide(
                ss_residual,
                ss_total,
                out=np.full_like(ss_total, np.nan),
                where=ss_total > 0,
            )
            uns_dict["latent_obs_r2_per_dim"] = 1.0 - latent_r2

        group_columns = set()
        if self._batch_encoding is not None:
            group_columns.update(self._batch_encoding.obs_columns)
        if self._balance_obs is not None:
            group_columns.add(self._balance_obs)
        if group_columns:
            residual, explained = self._reconstruction_stats(adata)
            # A bare MSE is not interpretable on its own: on the z-transformed input
            # BAE expects, 1.0 is what predicting zero everywhere scores, so "0.85"
            # reads as a good fit when it means 15% of variance explained. Record the
            # normalized figure next to it. See `linear_ceiling` for the companion
            # question of how much a latent_dim-dimensional model could explain.
            uns_dict["variance_explained"] = float(explained)
            kept = sorted(c for c in group_columns if adata.obs[c].nunique() <= _MAX_GROUP_LEVELS)
            skipped = sorted(set(group_columns) - set(kept))
            if skipped:
                warnings.warn(
                    "Per-group reconstruction losses were not computed for "
                    f"{skipped}: each has more than {_MAX_GROUP_LEVELS} levels. "
                    "adata.uns['bae']['reconstruction_loss_by_obs'] therefore does "
                    "not cover every conditioned column.",
                    UserWarning,
                    stacklevel=2,
                )
            uns_dict["reconstruction_loss_by_obs"] = {
                column: {
                    str(level): float(residual[np.asarray(adata.obs[column] == level)].mean())
                    for level in adata.obs[column].unique()
                }
                for column in kept
            }
        adata.uns["bae"] = uns_dict

        if self._prior_weights is not None:
            adata.uns["bae_transfer"] = self._transfer_diagnostics(adata, X, encoder_weights)
            # After uns["bae_transfer"] exists, so the statistics land beside the
            # rest of the transfer metadata rather than creating a stub dict.
            self._store_scaled_latent(adata, latent)

    def get_encoder_weights(self, as_numpy: bool = True) -> np.ndarray | torch.Tensor:
        """Get encoder weight matrix.

        Parameters
        ----------
        as_numpy
            If True, return numpy array; else torch tensor.

        Returns
        -------
        Encoder weights of shape (n_genes, latent_dim).
        """
        W = self.encoder.linear.weight.detach().T  # (latent_dim, n_genes) -> (n_genes, latent_dim)
        if as_numpy:
            return W.cpu().numpy()
        return W

    def get_splitsoftmax_encoder_weights(
        self, *, clip_negative: bool = True, as_numpy: bool = True
    ) -> np.ndarray | torch.Tensor:
        """Get encoder weights mapped to split-softmax dimensions.

        Returns the effective per-gene weights for each of the 2 * latent_dim
        split-softmax dimensions. For split-softmax dimension 2i (positive
        direction of latent dim i), the weights are W[:, i]. For dimension
        2i+1 (negative direction), the weights are -W[:, i].

        The columns are interleaved in the same order as the split-softmax
        output: (W_1, -W_1, W_2, -W_2, ..., W_d, -W_d).

        Can be called on models trained with or without split_softmax=True.
        When split_softmax was not enabled during training, a warning is emitted.

        Parameters
        ----------
        clip_negative
            If True (default), clip all negative weights to zero. This retains
            only the genes that positively contribute to each split-softmax
            dimension, improving interpretability.
        as_numpy
            If True, return numpy array; else torch tensor.

        Returns
        -------
        Encoder weights of shape (n_genes, 2 * latent_dim).
        """
        if self.split_softmax_layer is None:
            warnings.warn(
                "Model was not trained with split_softmax=True. "
                "Returning split-softmax encoder weights computed post-hoc.",
                UserWarning,
                stacklevel=2,
            )

        # W shape: (n_genes, latent_dim)
        W = self.encoder.linear.weight.detach().T

        # Interleave: (W_1, -W_1, W_2, -W_2, ..., W_d, -W_d)
        W_split = torch.stack([W, -W], dim=-1).reshape(W.shape[0], -1)

        if clip_negative:
            W_split = torch.clamp(W_split, min=0.0)

        if as_numpy:
            return W_split.cpu().numpy()
        return W_split
