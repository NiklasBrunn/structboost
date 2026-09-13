"""BAE main model class."""

from __future__ import annotations

import copy
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

from ._boosting import allboost, column_norms_sq
from ._decoder import BAEDecoder
from ._encoder import BAEEncoder, SplitSoftmax
from ._types import BAEConfig, TrainingReport

if TYPE_CHECKING:
    from anndata import AnnData

    from ._types import Device

# Diagnostic series collected per iteration when `diagnostics=True`.
# "iteration" is generated at the end, not accumulated.
_REPORT_FIELDS: tuple[str, ...] = tuple(TrainingReport.__dataclass_fields__)

# Below this mean matched |cosine| between an iteration's latent dimensions and the
# fitted model's, iteration-mode stability frequencies stop describing a single
# representation and a frequency threshold becomes destructive. Raised from 0.5 in
# 0.4.0: measured runs sitting at 0.70-0.79 already lost a quarter to a half of
# their recovered markers at the default threshold, while runs at 0.93 and above
# lost none.
_DIM_MATCH_WARN: float = 0.85

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
        self._mandatory_genes = None
        #: Design guidance (`fit(design_key=...)`): per design variable, in order,
        #: its latent dimensions and its strength, plus the strata; the
        #: decomposition's variables and strata; and, for the restored iteration,
        #: the exact split of the encoder weights and the per-step selection trace.
        #: The subspace bases are rebuilt from the data wherever they are needed.
        self._design_blocks: dict[str, np.ndarray] | None = None
        self._design_lambdas: dict[str, float] | None = None
        self._design_within: list[str] | None = None
        self._decompose_columns: list[str] | None = None
        self._decompose_within: list[str] | None = None
        self._encoder_components: dict[str, np.ndarray] | None = None
        self._selection_trace: dict | None = None
        self._selection_path: dict | None = None
        #: Layer `fit` read expression from; None means ``adata.X``. Later calls
        #: default to it so a model always reads the representation it learned on.
        self._layer: str | None = None
        self._var_names: np.ndarray | None = None
        # Transfer state: set by `from_reference`, None for an ordinary model.
        self._prior_weights: np.ndarray | None = None
        self._prior_info: dict[str, object] = {}
        self._latent_scaling: dict[str, np.ndarray] | None = None
        #: Whether the last fit built a full covariance matrix; recorded in uns.
        self._precomputed_covcache: bool = False
        #: AdamW state from the *best* iteration of the last fit, for
        #: `stability_selection(continue_optimizer=True)`. In-memory only: `save`
        #: deliberately excludes optimizer state, so a loaded model has None here.
        self._decoder_optimizer_state: dict | None = None

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

        .. admonition:: Exploratory
           :class: caution

           Under active development. Alignment, freezing and the diagnostics work,
           but the prior and novel latent blocks land on **incomparable scales** (a
           232x gap in per-dimension standard deviation was measured), so every
           Euclidean consumer must be handed ``obsm["X_bae_scaled"]``.

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
            # Deliberately no migration: formats are dropped when a BAEConfig field
            # goes, and `restore_payload` splats the stored config into BAEConfig,
            # so an old file carries settings the class no longer has. Refitting is
            # cheap; a shim for options that no longer exist is not.
            raise ValueError(
                f"{source} uses checkpoint format {written_format}, written by structboost "
                f"{payload.get('structboost_version', 'unknown')}; this install reads format "
                f"{MIN_CHECKPOINT_FORMAT} and newer, and no migration is provided. Refit the "
                "model to write a current checkpoint."
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

    def _design_subspaces(
        self, adata: AnnData, columns: list[str], within: list[str] | None
    ) -> dict[frozenset, np.ndarray]:
        """Orthonormal bases of the design subspaces the filters and the decomposition use.

        Keyed by subsets of ``columns``: the joint design ``J``, every leave-one-out
        design ``J \\ {v}``, and the empty design. Without ``within`` the empty
        design is the intercept and a subset's basis spans the intercept plus its
        encoded columns (through ``encode_obs_covariates``, so numeric columns
        work). With ``within`` -- the stratified form -- the empty design spans the
        stratum indicators and every other subset the stratum-by-design
        interaction, so ``P_S - P_∅`` is the design effect *inside* the strata with
        the stratum main effect, typically cell type, removed rather than kept.
        Bases come from a rank-revealing SVD because empty stratum-design cells
        make the interaction design rank deficient. Every subset is nested in
        ``J``, which is what makes ``P_J - P_{J\\v}`` a projector and the parts of
        the decomposition additive.
        """
        from scipy.linalg import orth

        from ._utils import encode_obs_covariates

        n = adata.n_obs
        strata = None
        if within:
            import pandas as pd

            labels = adata.obs[list(within)].astype(str).agg("|".join, axis=1)
            strata = pd.get_dummies(labels).to_numpy(dtype=np.float64)

        def basis(subset: tuple[str, ...]) -> np.ndarray:
            if not subset:
                return orth(strata) if strata is not None else np.full((n, 1), n**-0.5)
            design = np.asarray(
                encode_obs_covariates(adata, list(subset)).encoded, dtype=np.float64
            )
            if strata is None:
                return np.linalg.qr(np.column_stack([np.ones(n), design]))[0]
            return orth(
                np.hstack([strata, *(strata * design[:, [j]] for j in range(design.shape[1]))])
            )

        subsets = [tuple(columns), *(tuple(c for c in columns if c != v) for v in columns), ()]
        return {frozenset(subset): basis(subset) for subset in dict.fromkeys(subsets)}

    @staticmethod
    def _design_keep(Q_keep: np.ndarray, Q_drop: np.ndarray | None, T: np.ndarray) -> np.ndarray:
        """The design-explained part of the columns of ``T`` : ``P_keep T - P_drop T``."""
        kept = Q_keep @ (Q_keep.T @ T)
        return kept if Q_drop is None else kept - Q_drop @ (Q_drop.T @ T)

    @staticmethod
    def _design_r2(Q_keep: np.ndarray, Q_drop: np.ndarray | None, Z: np.ndarray) -> np.ndarray:
        """Share of each latent column's (within-stratum) variance the design explains."""
        Z = np.asarray(Z, dtype=np.float64)
        centred = Z - Z.mean(axis=0) if Q_drop is None else Z - Q_drop @ (Q_drop.T @ Z)
        kept = Q_keep @ (Q_keep.T @ centred)
        total = (centred**2).sum(axis=0)
        return np.divide(
            (kept**2).sum(axis=0), total, out=np.full_like(total, np.nan), where=total > 0
        )

    def _validate_transfer_fit(
        self,
        adata: AnnData,
        *,
        init_obsm: str | None,
        init_pca: bool,
        design_key: str | list[str] | None = None,
    ) -> None:
        """Reject fit settings whose semantics conflict with a prior encoder matrix."""
        if design_key is not None:
            raise ValueError(
                "design_key cannot be combined with a prior encoder matrix in this "
                "release: the transferred columns would need their own component in "
                "the weight attribution. Fit the design-guided model from scratch."
            )
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
    def _recon_loss(x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction MSE, as a single fused reduction over all elements.

        Deliberately not reassociated into a per-cell mean followed by an average:
        that shifts the float32 result by ~1e-7 relative for no gain.
        """
        return nn.functional.mse_loss(x_recon, x)

    @staticmethod
    def _correlation_disentanglement_loss(
        z: torch.Tensor,
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

        centered = z - z.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / z.shape[0]

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
        stats: dict[str, float] | None = None,
        z_override: torch.Tensor | None = None,
        components: bool = False,
        decompose: dict[frozenset, np.ndarray] | None = None,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray] | None]:
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

        This holds per cell: the decoder is a deterministic per-cell function, and
        it is run in eval mode below so that any future stateful layer cannot
        quietly couple cells into each other's targets.

        Parameters
        ----------
        X
            Full input data tensor of shape (n_cells, n_genes).
        lr
            Step size for gradient descent direction.
        obs_covariates
            Optional obs covariate tensor for cVAE decoder conditioning.
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
        components
            Also return the additive parts of the target, each
            ``(n_cells, latent_dim)``: ``"carry"`` (the current code ``z``),
            ``"recon"`` (the reconstruction gradient step) and, under correlation
            disentanglement, ``"correlation"``. ``targets == sum(parts)`` up to
            float32 rounding. Used by the weight attribution.
        decompose
            Bases from :meth:`_design_subspaces` for ``_decompose_columns``.
            Splits the reconstruction step by design variables instead of
            returning it whole. With ``R = D(z) - X`` and the target-loss
            convention ``L = ||R||²_F / n_genes``, the residual sum of squares a
            design subspace ``S`` explains is ``G_S = ||Q_Sᵀ R||²_F / n_genes``.
            Pythagoras gives ``L = G_J + (L - G_J)`` exactly and gradients are
            additive, so the parts are ``between_<v> = G_J - G_{J\\v}`` (what
            variable ``v`` explains beyond the others), ``shared`` (the rest of
            ``G_J - G_∅``, only with several variables; can be negative under
            suppression and large when variables are confounded), ``mean`` or
            ``strata`` (``G_∅``) and ``within = L - G_J``. One backward pass per
            subset. The total gradient is still the one the plain fit takes, so
            the fit is unchanged. Also returns each gene's residual sum of
            squares split the same way, as shares.

        Returns
        -------
        Target latent codes as numpy array of shape (n_cells, latent_dim); with
        ``components``, also the parts dict and the per-gene residual shares
        (``None`` without ``decompose``).
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
        loss = self._recon_loss(x_recon, X)
        # Sum over cells, mean over genes. See the docstring: this is what makes a
        # cell's target step independent of how many other cells are in the dataset.
        # That is exactly `n_cells` times the elementwise mean.
        target_loss = loss * X.shape[0]
        # Only dL/dz is needed. `torch.autograd.grad` skips accumulating gradients
        # into the decoder parameters, which `loss.backward()` would compute and
        # leave in `.grad` for the decoder optimizer to discard on its next
        # `zero_grad()`. Same convention as `_full_recon_loss`.
        penalty = None
        if self.config.disentanglement == "correlation":
            correlation_loss, variance_loss = self._correlation_disentanglement_loss(z)
            # The reconstruction target loss sums over cells. Correlation and
            # variance are population averages, so multiply them by n_cells to
            # keep the per-cell regularizer gradient, and therefore lambda's
            # meaning, independent of dataset size.
            penalty = (
                X.shape[0]
                * self.config.disentanglement_lambda
                * (correlation_loss + _DISENTANGLEMENT_VARIANCE_WEIGHT * variance_loss)
            )
        # The total gradient is always taken from the total loss, exactly as a
        # fit without attribution takes it; the parts below are derived from it
        # and from extra backward passes, so attribution never changes the fit.
        total = target_loss if penalty is None else target_loss + penalty
        need_graph = components and (penalty is not None or decompose is not None)
        (z_grad,) = torch.autograd.grad(total, z, retain_graph=need_graph)

        grads: dict[str, torch.Tensor] = {}
        shares: dict[str, np.ndarray] | None = None
        if components:
            recon_grad = z_grad
            if penalty is not None:
                # The penalty's share of the step is its own part; reconstruction's
                # is the remainder.
                grads["correlation"] = torch.autograd.grad(
                    penalty, z, retain_graph=decompose is not None
                )[0]
                recon_grad = z_grad - grads["correlation"]
            if decompose is None:
                grads["recon"] = recon_grad
            else:
                columns, stratified = self._decompose_columns, self._decompose_within is not None
                R = x_recon - X
                bases = {
                    key: torch.as_tensor(Q, dtype=X.dtype, device=X.device)
                    for key, Q in decompose.items()
                }

                def explained(key: frozenset) -> torch.Tensor:
                    value = ((bases[key].T @ R) ** 2).sum() / X.shape[1]
                    return torch.autograd.grad(value, z, retain_graph=True)[0]

                joint, empty = frozenset(columns), frozenset()
                g_joint, g_empty = explained(joint), explained(empty)
                unique = {v: g_joint - explained(joint - {v}) for v in columns}
                grads.update({f"between_{v}": g for v, g in unique.items()})
                if len(columns) > 1:
                    grads["shared"] = g_joint - g_empty - sum(unique.values())
                grads["strata" if stratified else "mean"] = g_empty
                grads["within"] = recon_grad - g_joint
                with torch.no_grad():
                    # The same split per gene, as shares of its residual sum of squares.
                    ss = {key: ((Q.T @ R) ** 2).sum(dim=0) for key, Q in bases.items()}
                    r_sq = (R**2).sum(dim=0)
                    cols = {f"between_{v}": ss[joint] - ss[joint - {v}] for v in columns}
                    if len(columns) > 1:
                        cols["shared"] = ss[joint] - ss[empty] - sum(cols.values())
                    cols["strata" if stratified else "mean"] = ss[empty]
                    cols["within"] = r_sq - ss[joint]
                    shares = {n: (c / r_sq.clamp_min(1e-30)).cpu().numpy() for n, c in cols.items()}

        # Target = current z moved in negative gradient direction
        with torch.no_grad():
            targets = z - lr * z_grad
            if stats is not None:
                stats["target_grad_norm"] = float(z_grad.norm())
                # Report the mean MSE, not `target_loss`: the A/B/C loss
                # decomposition in TrainingReport compares this against
                # `loss_post_boost` and `loss_post_decoder`, which are both means.
                stats["loss_pre_boost"] = float(loss.detach())
            if components:
                parts = {"carry": z.detach().cpu().numpy()}
                parts.update({name: (-lr * g).cpu().numpy() for name, g in grads.items()})
                return targets.cpu().numpy(), parts, shares

        return targets.cpu().numpy()

    def _boost_encoder(
        self,
        X: torch.Tensor,
        D: torch.Tensor | None,
        boost: dict,
        *,
        z_override: torch.Tensor | None = None,
        path: bool = False,
        design: dict[frozenset, np.ndarray] | None = None,
        decompose: dict[frozenset, np.ndarray] | None = None,
        stats: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray] | None, dict | None]:
        """The boosting half of one training iteration: steps 1-4 of :meth:`fit`.

        Targets by one gradient step on the latent code, optional Löwdin
        orthogonalization, optional design filter, encoder reset, and the
        ``allboost`` fit. Shared by :meth:`fit` and
        :meth:`_iteration_support_frequency`, so stability selection continues
        exactly the alternation that produced the model.

        Parameters
        ----------
        boost
            Fixed per-fit boosting inputs from :meth:`_boost_inputs`. Its
            ``"covcache"`` entry is filled in place on the first call when the
            cache is lazy.
        design, decompose
            Subspace bases from :meth:`_design_subspaces` for the design filter
            (``_design_blocks``) and for the decomposition of the reconstruction
            step (``_decompose_columns``, forwarded to
            :meth:`_compute_boosting_targets`). Either one also makes the fitted
            encoder split into the additive parts of its target: ``carry`` (the
            current code ``z``), ``recon`` or its decomposition, ``correlation``
            where that mode is on, and ``design``, what the filter removed.
            Boosting is linear in the target once the selection path is fixed,
            so each part is fitted along the path of the total
            (``allboost(selection_from=...)``) and the parts sum to the encoder
            exactly; the part not replayed is taken as the remainder, so the
            identity holds to the last bit rather than to float32 rounding.

            Deliberately the split of *this iteration's* target, not an
            accumulation over the fit. An accumulated split (shadow matrices
            carried through the recursion) is exact too but ill-conditioned:
            wherever reconstruction keeps re-injecting what the design term keeps
            removing, the two shadows grow while cancelling -- measured at 400x
            the weight on planted data at ``design_lambda=0.5``.
            The filter of block ``v`` keeps ``P_J T - P_{J\\v} T``, what ``v``
            explains beyond the other design variables, so a confounded part is
            filtered from every block, and removes the fraction
            ``target_optim_lr * design_lambda[v]`` of the rest. Applied after
            orthogonalization, so the constraint is exact and the orthogonality
            between constrained and free dimensions is the approximate one.
        path
            Record the selected gene and its increment per step even without
            attribution (``track_selection_path``). Free: the history is passive.

        Returns
        -------
        betamat
            Full ``(latent_dim, n_features)`` coefficient matrix, genes first.
        targets
            The targets the encoder was fitted against, after all transforms.
        parts
            ``{name: (latent_dim, n_genes)}`` summing to the gene weights, or None.
        trace
            Per-step record: ``gene``; ``delta`` per part; per replayed part the
            ``counterfactual_gene`` it would have picked and the ``rank`` of the
            applied gene under its criterion; and, for the total and each
            replayed part, the ``fit_score`` of the applied update against that
            part's residual (1 minus the relative residual left, Eq. 13-15 of the
            PerturbBoost supplement), the cosine ``alignment`` of the applied gene
            with that residual (Eq. 16-17) and the first-order ``gain`` (Eq. 20);
            plus ``target_norm`` per part. With only ``path``, just ``gene`` and
            ``delta["total"]``. None otherwise.
        """
        attribute = design is not None or decompose is not None
        out = self._compute_boosting_targets(
            X,
            lr=self.config.target_optim_lr,
            obs_covariates=D,
            stats=stats,
            z_override=z_override,
            components=attribute,
            decompose=decompose,
        )
        targets, parts, shares = out if attribute else (out, {}, None)
        # Replayed followers: the target without the design step (whose pick is
        # the counterfactual that matters, when there is a design step) and each
        # gradient part alone.
        steps = [n for n in parts if n != "carry"]
        followers: dict[str, np.ndarray] = {}
        if attribute and design is not None:
            followers["no_design"] = sum(parts.values())
        followers.update({n: parts[n] for n in steps})
        follower_targets = list(followers.values())

        if self.config.disentanglement == "orthogonal":
            from ._utils import disentangle_boosting_targets

            alpha = self.config.disentanglement_alpha
            if attribute:
                targets, follower_targets = disentangle_boosting_targets(
                    targets, alpha=alpha, components=follower_targets
                )
            else:
                targets = disentangle_boosting_targets(targets, alpha=alpha)

        if design is not None:
            joint = frozenset(self._design_blocks)
            for variable, dims in self._design_blocks.items():
                block = targets[:, dims].astype(np.float64)
                kept = self._design_keep(design[joint], design[joint - {variable}], block)
                factor = self.config.target_optim_lr * self._design_lambdas[variable]
                targets[:, dims] = (block - factor * (block - kept)).astype(targets.dtype)
            if attribute:
                # The design part is replayed too, for its scores and counterfactual;
                # its *weights* are still taken as the exact remainder below.
                followers["design"] = targets - follower_targets[0]
                follower_targets.append(followers["design"])

        self.encoder.reset_weights()
        n_features = boost["sourcemat"].shape[1]
        prior_dims = boost["prior_dims"]
        fit_targets, beta_init, fit_mandatory = self._transfer_boosting_inputs(
            targets, n_features, prior_dims, boost["mandatory"]
        )
        k = fit_targets.shape[1]
        if k == 0:
            # Frozen transfer with no additional dimensions: nothing competes for
            # selection, and only the decoder adapts to the new data.
            betamat = np.zeros((0, n_features), dtype=np.float64)
            hist = None
        else:
            kwargs = dict(
                covcache=boost["covcache"],
                col_norms_sq=boost["col_norms_sq"],
                stepno=self.config.boosting_stepno,
                nu=self.config.boosting_nu,
                csf=self.config.boosting_csf,
                independent=self.config.boosting_independent,
                mandatory_features=fit_mandatory,
                mandatory_ridge=boost["ridge"],
                beta_init=beta_init,
                return_covcache=boost["covcache"] is None,
            )
            if attribute or path:
                kwargs["return_history"] = "steps"
            if attribute:
                # Followers ride along as extra target columns: one predictor-target
                # product and one covariance cache for the total and its parts.
                kwargs["selection_from"] = np.tile(np.arange(k), 1 + len(followers))
                if isinstance(fit_mandatory, list):
                    kwargs["mandatory_features"] = fit_mandatory * (1 + len(followers))
                fit_targets = np.hstack([fit_targets, *follower_targets])
            result = allboost(boost["sourcemat"], fit_targets, **kwargs)
            betamat, *rest = result if isinstance(result, tuple) else (result,)
            hist = rest[0] if (attribute or path) else None
            if boost["covcache"] is None:
                boost["covcache"] = rest[-1]
        if not attribute:
            trace = (
                {"gene": hist.selection.copy(), "delta": {"total": hist.update.copy()}}
                if hist is not None
                else None
            )
            return (
                self._expand_transfer_betamat(betamat, n_features, prior_dims),
                targets,
                None,
                trace,
            )

        def split(matrix: np.ndarray) -> dict[str, np.ndarray]:
            """Blocks of a stacked (leader, followers...) array as the additive parts.

            The remainder goes to the part that was not replayed: ``carry`` when
            there is no design step, else ``design`` (with ``carry`` taken off the
            no-design target). Either way the parts sum to the leader exactly.
            """
            block = {n: matrix[k * (i + 1) : k * (i + 2)] for i, n in enumerate(followers)}
            out = {n: block[n].copy() for n in steps}
            if "no_design" in block:
                out["carry"] = block["no_design"] - sum(block[n] for n in steps)
                out["design"] = matrix[:k] - block["no_design"]
            else:
                out["carry"] = matrix[:k] - sum(block[n] for n in steps)
            return out

        gene_cols = slice(None, self.n_genes)
        new_parts = {n: v[:, gene_cols] for n, v in split(betamat).items()}
        delta = {"total": hist.update[:k], **split(hist.update)}
        stacked = np.vstack([targets.T, *(t.T for t in follower_targets)])
        norm = {"total": np.linalg.norm(targets, axis=0)}
        norm.update({n: np.linalg.norm(v, axis=1) for n, v in split(stacked).items()})

        # Per-step scores of the applied update against each part's residual, all
        # from the O(1) bookkeeping allboost keeps: with u the applied increment,
        # s_c = x_j' r_c and q_c = ||r_c||^2 just before the step,
        #   fit_score = 1 - ||r_c - u x_j||^2 / ||r_c||^2 = (2 u s_c - u^2 ||x_j||^2) / q_c,
        #   alignment = s_c / (||x_j|| ||r_c||),   gain = u s_c.
        gene = hist.selection[:k]
        rows = {"total": 0, **{n: i + 1 for i, n in enumerate(followers)}}
        u = hist.update[:k]
        xnorm_sq = np.where(gene >= 0, boost["col_norms_sq"][np.maximum(gene, 0)], 0.0).astype(
            np.float64
        )
        eps = np.finfo(np.float64).tiny
        fit_score, alignment, gain = {}, {}, {}
        for name, r in rows.items():
            s_c = hist.score[k * r : k * (r + 1)]
            q_c = hist.residual_sq[k * r : k * (r + 1)]
            fit_score[name] = (2.0 * u * s_c - u**2 * xnorm_sq) / (q_c + eps)
            alignment[name] = s_c / (np.sqrt(xnorm_sq * q_c) + eps)
            gain[name] = u * s_c
        trace = {
            "gene": gene.copy(),
            "delta": delta,
            "counterfactual_gene": {
                n: hist.selection[k * (i + 1) : k * (i + 2)].copy() for i, n in enumerate(followers)
            },
            "rank": {
                n: hist.rank[k * (i + 1) : k * (i + 2)].copy() for i, n in enumerate(followers)
            },
            "fit_score": fit_score,
            "alignment": alignment,
            "gain": gain,
            "target_norm": norm,
        }
        if shares is not None:
            trace["residual_share"] = shares
        return betamat[:k], targets, new_parts, trace

    def _boost_inputs(
        self,
        sourcemat: np.ndarray,
        mandatory: np.ndarray | list[np.ndarray] | None,
        ridge: np.ndarray,
        *,
        precompute: bool,
        prior_dims: int,
    ) -> dict:
        """Everything about the boosting problem that is fixed for a whole fit.

        ``sourcemat`` is fixed, so its column norms and covariance are too;
        computing them here, once, is what ``allboost`` would otherwise redo on
        every call. A lazy cache starts as ``None`` and is filled by the first
        :meth:`_boost_encoder` call.
        """
        from ._utils import compute_covariance_cache

        return {
            "sourcemat": sourcemat,
            "mandatory": mandatory,
            "ridge": ridge,
            "prior_dims": prior_dims,
            "col_norms_sq": column_norms_sq(sourcemat),
            "covcache": compute_covariance_cache(sourcemat) if precompute else None,
        }

    def _update_decoder(
        self,
        X: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        *,
        D_train: torch.Tensor | None = None,
    ) -> float:
        """Update the decoder with one shuffled pass over the cells.

        The cells are partitioned into ``ceil(n_cells / batch_size)`` minibatches
        and each contributes exactly one AdamW step, so the decoder half of the
        alternation consumes the same data the boosting half does: ``z*`` is
        computed on every cell and the encoder is re-solved on every cell.

        The step *count* is therefore not a setting. It was one until 0.5.0, and a
        fixed count made a cell's participation depend on dataset size while the
        boosting half stayed full-batch at every size. See ``BAEConfig.batch_size``.

        Parameters
        ----------
        X
            Full input data tensor.
        optimizer
            Optimizer for decoder parameters.
        D_train
            Optional obs covariate tensor for cVAE decoder conditioning.

        Returns
        -------
        Mean minibatch loss over the pass.
        """
        self.decoder.train()
        tensors = [X] if D_train is None else [X, D_train]
        dataset = TensorDataset(*tensors)
        # `drop_last` is left at False, so a final short batch still contributes
        # rather than silently excluding up to `batch_size - 1` cells from the
        # pass. Safe because the decoder is a deterministic per-cell function --
        # dropout and batch norm went in 0.3.0 -- so a small batch is a noisier
        # step, not an invalid one. `_pretrain_decoder` has always done the same.
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        total_loss = 0.0
        steps_done = 0

        for batch_data in loader:
            x_batch = batch_data[0]
            d_batch = batch_data[1] if D_train is not None else None
            optimizer.zero_grad()
            # The encoder is fitted by boosting and is in no optimizer, so its
            # gradient is never read. Detaching keeps backward from computing a
            # (latent_dim, n_genes) gradient per minibatch and from accumulating
            # it into `encoder.linear.weight.grad`, which nothing ever zeroes.
            with torch.no_grad():
                z = self.encoder(x_batch)
            h = self.split_softmax_layer(z) if self.split_softmax_layer else z
            x_recon = self.decoder(h, d_batch)
            loss = self._recon_loss(x_recon, x_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps_done += 1

        return total_loss / max(steps_done, 1)

    def _checkpoint_selection_loss(self, train_loss: float, X: torch.Tensor) -> float:
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
            correlation_loss, variance_loss = self._correlation_disentanglement_loss(z)
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
        tensors = [Z, X] if D is None else [Z, X, D]
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
                loss = self._recon_loss(self.decoder(h, covariates), x_batch)
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
        grad: bool = False,
    ) -> tuple[float, float]:
        """Full-data reconstruction MSE, and optionally the decoder gradient norm.

        Always runs the decoder in eval mode, so a future stateful layer cannot
        make a diagnostic pass mutate the model or consume RNG; gradients are taken
        with ``torch.autograd.grad`` so nothing accumulates into ``.grad``. Together
        these keep diagnostics read-only with respect to the fitted model.
        """
        was_training = self.decoder.training
        self.decoder.eval()
        try:
            if not grad:
                with torch.no_grad():
                    z = self.encoder(X)
                    h = self.split_softmax_layer(z) if self.split_softmax_layer else z
                    loss = self._recon_loss(self.decoder(h, D), X)
                    return float(loss), float("nan")

            z = self.encoder(X)
            h = self.split_softmax_layer(z) if self.split_softmax_layer else z
            loss = self._recon_loss(self.decoder(h, D), X)
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
        loss_c, dec_grad = self._full_recon_loss(X, D, grad=True)
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
        design_key: str | list[str] | dict[str, list[int]] | None = None,
        design_within: str | list[str] | None = None,
        decompose_key: str | list[str] | None = None,
        decompose_within: str | list[str] | None = None,
        track_selection_path: bool = False,
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
        stability_selection: bool = False,
    ) -> BAE:
        """Fit BAE using hybrid boosting+SGD training.

        The training loop alternates between:

        1. Computing boosting targets via gradient step: z* = z - lr * ∂L_target/∂z,
           where L_target sums the squared error over cells and averages it over
           genes (see ``_compute_boosting_targets``)
        2. (Optional) Orthogonalizing the targets across latent dimensions
        3. Resetting encoder weights to zero
        4. Fitting encoder via allboost to map X → z*
        5. Updating the decoder with one shuffled pass over the cells, i.e.
           ``ceil(n_cells / batch_size)`` AdamW steps

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
        design_key
            **Exploratory**, under active development. Obs column, or several,
            holding an experimental design variable (condition, timepoint, ...)
            that a block of latent dimensions should separate. Adds the loss
            ``½ Σ_k ||(I - P) z_k||²`` on those dimensions — the latent
            variance the design does *not* explain, ``P`` projecting onto the
            encoded design columns — weighted by ``BAEConfig.design_lambda``; see
            there for how the step is taken and what it costs.

            Every variable gets one block of dimensions. A name or a list assigns
            consecutive blocks in order, each as wide as the variable's encoded
            columns (levels minus one per categorical column, one per numeric
            column), since the subspace a variable spans has that rank; a dict
            ``{variable: [dims, ...]}`` assigns them explicitly. The other
            dimensions stay reconstruction-only. A block keeps the part of the
            target its variable explains *beyond* the other variables
            (``P_J - P_{J \\ v}``), so what two confounded variables share is
            filtered out of both blocks. ``design_lambda`` may be a dict with one
            strength per variable, missing ones defaulting to 1.
            ``uns["bae"]["design_blocks"]`` reports the assignment and
            ``latent_design_r2_per_dim`` each dimension's R² on its block's
            variable (on the joint design for the free dimensions).

            Turning this on also makes the fit **split every encoder weight
            exactly** into the parts of the target that produced it in the
            restored iteration: ``varm["BAE_encoder_weights_carry"]`` (the code
            carried over from the previous iteration), ``..._recon`` (the
            reconstruction gradient step) and ``..._design`` (what the design step
            removed), which sum to ``varm["BAE_encoder_weights"]`` to the last bit.
            ``uns["bae"]["selection_trace"]`` records, for the same iteration and
            every boosting step, the selected gene, its coefficient increment
            split the same way, the norms of the target parts, and the gene that
            would have been selected at that step *without* the design step
            (``counterfactual_gene["no_design"]``) or by the reconstruction
            gradient alone (``["recon"]``), given the genes already entered. The
            split is exact because boosting is linear in its target once the
            selection path is fixed.

            Read it with the mechanism in mind. The design step is a *filter*: it
            removes within-group variation from the target rather than adding
            between-group variation, so on the genes that win, its own additive
            share is a small shrinkage and the between-group signal it protects
            sits in the reconstruction part. Whether the design term *decided* a
            selection is therefore the counterfactual column, not the sign of the
            design part. On planted data at ``design_lambda=1`` the constrained
            dimension recovered all ten condition genes, the design part opposed
            the weight's sign on every one of them, and 79% of its boosting steps
            would have gone to another gene without the design step.

            Not combinable with a transfer model, a warm start or
            ``boosting_independent=False`` in this release. The covariate is
            never an encoder input, so :meth:`transform` stays gene-only.
        design_within
            Obs column(s) defining strata, typically the cell type. With it the
            kept part of the target is the design effect *within* each stratum:
            the stratum main effect is removed together with the within-cell
            residual, so a constrained dimension separates conditions inside every
            cell type rather than separating cell types whose composition differs
            between conditions. ``latent_design_r2_per_dim`` then reports the share
            of within-stratum variance the design explains. Requires ``design_key``.
        decompose_key
            **Exploratory.** Obs column(s) naming design variables by which the
            *reconstruction* gradient is split, as a diagnostic that leaves the
            fit unchanged. The reconstruction loss splits exactly into the residual
            variance the design explains and the rest, and so does its gradient,
            so the trace then records for every boosting step how much of the
            selected gene's score came from between-design residual variance
            (``between_<variable>``, plus ``shared`` for several variables and
            ``mean`` for the residual's column means), how much from ``within``,
            and which gene each part alone would have picked. With no
            ``design_key`` this measures what a design term would be up against;
            with one it splits the reconstruction part of that fit. The weight
            parts land in ``varm["BAE_encoder_weights_<part>"]`` and each gene's
            residual sum of squares is split the same way in
            ``varm["BAE_residual_variance_share"]``. Costs one extra backward pass
            per subset of the variables.
        decompose_within
            Strata for the decomposition, as ``design_within`` for the design
            term: the between parts become the design effect inside each stratum
            and a ``strata`` part holds the stratum main effect.
        track_selection_path
            Also store, for *every* training iteration, the gene selected at each
            boosting step and its increment, as ``uns["bae"]["selection_path"]``
            with arrays of shape ``(n_iterations, latent_dim, stepno)``. This is
            the lightweight selection-stability log; the full per-step scores are
            kept for the restored iteration only. Works with or without a
            ``design_key`` and does not change the fit.
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
            **Exploratory**, under active development; see :class:`BAE` notes on
            warm starts. Warm-start the latent state from ``adata.obsm[init_obsm]``
            instead of from zero. If that representation has a different number of columns
            than ``config.latent_dim``, the representation wins: ``latent_dim`` is
            overwritten for this fit and a ``UserWarning`` is emitted. Mutually
            exclusive with ``init_pca``.
        init_pca
            **Exploratory**, under active development. Warm-start from a PCA of
            ``adata.X`` keeping ``config.latent_dim`` components. The data is never rescaled, and it is mean-centered only
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
            results in ``adata``. Off by default; call the method directly for
            control over ``n_runs`` and ``threshold``.

        Returns
        -------
        Self for method chaining.

        Notes
        -----
        The warm start is applied once, on the first iteration only: it sets the
        boosting targets, so the encoder learns the supplied representation and the
        decoder is then trained against the encoder's output.

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

        if design_key is None and design_within is not None:
            raise ValueError(
                "design_within was given without a design_key; there is nothing to stratify"
            )
        if decompose_key is None and decompose_within is not None:
            raise ValueError(
                "decompose_within was given without a decompose_key; there is nothing to stratify"
            )
        if design_key is not None or decompose_key is not None:
            what = "design_key" if design_key is not None else "decompose_key"
            if init_pca or init_obsm is not None:
                raise ValueError(
                    f"{what} cannot be combined with init_pca/init_obsm: the warm "
                    "start would need its own component in the weight attribution"
                )
            if not self.config.boosting_independent:
                raise ValueError(
                    f"{what} requires boosting_independent=True: the attribution "
                    "replays the selection path per latent dimension, which shared "
                    "boosting state across dimensions would corrupt"
                )

        is_transfer = self._prior_weights is not None
        if is_transfer:
            self._validate_transfer_fit(
                adata,
                init_obsm=init_obsm,
                init_pca=init_pca,
                design_key=design_key if design_key is not None else decompose_key,
            )
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

        # A frozen transfer with no added dimensions never calls allboost at all
        # (there is nothing left to select), so building a covariance matrix for
        # it would be pure waste -- 8*p^2 bytes and an O(n*p^2) product for a
        # cache no one reads.
        boosts_anything = not (
            is_transfer
            and self.config.prior_mode == "frozen"
            and self._prior_weights.shape[1] >= self.config.latent_dim
        )
        from ._utils import resolve_precompute_covcache

        precompute_covcache = boosts_anything and resolve_precompute_covcache(
            self.config.boosting_precompute_covcache, sourcemat_aug.shape[1]
        )
        self._precomputed_covcache = precompute_covcache
        prior_dims = self._prior_weights.shape[1] if is_transfer else 0
        boost = self._boost_inputs(
            sourcemat_aug,
            allboost_mandatory,
            mandatory_ridge,
            precompute=precompute_covcache,
            prior_dims=prior_dims,
        )

        # --- Design guidance and the reconstruction-gradient decomposition ---
        self._design_blocks = self._design_lambdas = self._design_within = None
        self._decompose_columns = self._decompose_within = None
        design = decompose = None

        def resolve_columns(spec, name: str) -> list[str]:
            columns = [spec] if isinstance(spec, str) else list(spec)
            if not columns:
                raise ValueError(f"{name} must name at least one obs column")
            missing = [c for c in columns if c not in adata.obs.columns]
            if missing:
                raise ValueError(f"{name} columns {missing} not found in adata.obs")
            if len(set(columns)) != len(columns):
                raise ValueError(f"{name} names a column twice: {columns}")
            return columns

        def resolve_within(spec, columns: list[str], name: str) -> list[str] | None:
            if spec is None:
                return None
            within = resolve_columns(spec, name)
            if set(within) & set(columns):
                raise ValueError(f"{name} and its design variables must name different columns")
            return within

        if design_key is not None:
            from ._utils import encode_obs_covariates

            latent_dim = self.config.latent_dim
            if isinstance(design_key, dict):
                columns = resolve_columns(list(design_key), "design_key")
                blocks = {v: np.asarray(design_key[v], dtype=np.intp).ravel() for v in columns}
            else:
                columns = resolve_columns(design_key, "design_key")
                edges = np.cumsum(
                    [0, *(encode_obs_covariates(adata, [v]).n_columns for v in columns)]
                )
                blocks = {
                    v: np.arange(edges[i], min(edges[i + 1], latent_dim))
                    for i, v in enumerate(columns)
                }
            dims = np.concatenate(list(blocks.values()))
            if (
                any(d.size == 0 for d in blocks.values())
                or np.unique(dims).size != dims.size
                or ((dims < 0) | (dims >= latent_dim)).any()
            ):
                raise ValueError(
                    "design_key blocks must be non-empty, disjoint sets of dimensions in "
                    f"[0, {latent_dim}), got { {v: d.tolist() for v, d in blocks.items()} }"
                )
            spec = self.config.design_lambda
            if isinstance(spec, dict) and set(spec) - set(columns):
                raise ValueError(
                    "design_lambda names variables that are not in design_key: "
                    f"{sorted(set(spec) - set(columns))}"
                )
            lambdas = {
                v: float(spec.get(v, 1.0) if isinstance(spec, dict) else spec) for v in columns
            }
            lr = self.config.target_optim_lr
            for v, lam in lambdas.items():
                if lr * lam > 1.0:
                    raise ValueError(
                        f"target_optim_lr * design_lambda = {lr * lam:.3g} for {v!r} exceeds 1. "
                        "One step then overshoots the design subspace: the within-group "
                        "residual comes back with its sign flipped, and since the selection "
                        "criterion is squared its competitors regain their scores. Use a "
                        "value in [0, 1/target_optim_lr]; 1 removes the residual completely."
                    )
            self._design_blocks, self._design_lambdas = blocks, lambdas
            self._design_within = resolve_within(design_within, columns, "design_within")
            design = self._design_subspaces(adata, columns, self._design_within)

        if decompose_key is not None:
            self._decompose_columns = resolve_columns(decompose_key, "decompose_key")
            self._decompose_within = resolve_within(
                decompose_within, self._decompose_columns, "decompose_within"
            )
            decompose = self._design_subspaces(
                adata, self._decompose_columns, self._decompose_within
            )

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
                    verbose=verbose,
                    desc="Decoder warm-up (prior only)",
                )

        # Early stopping state (based on the checkpoint-selection objective)
        best_selection_loss = float("inf")
        patience_counter = 0
        best_encoder_weights = None
        best_decoder_state = None
        best_optimizer_state = None
        best_batch_weights = None
        best_components = None
        best_trace = None
        path_log: dict[str, list[np.ndarray]] = {"gene": [], "update": []}

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
            # STEPS 1-4: targets by a gradient step on z (the warm start seeds z on
            # the first iteration only), orthogonalization, design filter, encoder
            # reset and the boosting fit -- see `_boost_encoder`. With a design key
            # the encoder is also split exactly into the parts of its target.
            diag_stats: dict[str, float] = {}
            betamat, targets, components, trace = self._boost_encoder(
                X_train,
                D_condition,
                boost,
                z_override=z_init if iteration == 0 else None,
                path=track_selection_path,
                design=design,
                decompose=decompose,
                stats=diag_stats if collect_diagnostics else None,
            )
            if track_selection_path and trace is not None:
                path_log["gene"].append(trace["gene"])
                path_log["update"].append(trace["delta"]["total"])

            # Extract gene weights only; obs weights are nuisance (discarded)
            W_genes = betamat[:, : self.n_genes]
            batch_weights = betamat[:, self.n_genes :].copy() if D_nuisance_np is not None else None
            W = torch.from_numpy(W_genes.astype(np.float32)).to(self.config.device)
            self.encoder.set_weights(W)

            # Loss with the new encoder but the old decoder: isolates the
            # boosting step's effect from the decoder's.
            if collect_diagnostics:
                diag_stats["loss_post_boost"] = self._full_recon_loss(X_train, D_condition)[0]

            # STEP 5: Update decoder with one shuffled pass over the cells
            train_loss = self._update_decoder(
                X_train,
                decoder_optimizer,
                D_train=D_condition,
            )
            self._training_history["train_loss"].append(train_loss)
            selection_loss = self._checkpoint_selection_loss(train_loss, X_train)
            self._training_history["selection_loss"].append(selection_loss)

            if collect_diagnostics:
                diag_series["train_loss"].append(train_loss)
                prev_W = self._collect_diagnostics(
                    diag_series,
                    diag_stats,
                    X=X_train,
                    D=D_condition,
                    targets=targets,
                    prev_W=prev_W,
                )

            # STEP 6: Check early stopping criteria and store the best model. For
            # correlation-constrained fits, selection_loss includes the constraint
            # so restoration cannot quietly select a less-disentangled checkpoint.
            if selection_loss < best_selection_loss:
                best_selection_loss = selection_loss
                patience_counter = 0
                best_encoder_weights = self.encoder.linear.weight.detach().clone()
                best_decoder_state = {k: v.clone() for k, v in self.decoder.state_dict().items()}
                # Snapshotted *here*, not at the end of the loop: `fit` returns the
                # best iteration's weights, so end-of-loop moments would belong to a
                # decoder that is not the one being returned. Pairing them would
                # manufacture an inconsistency rather than continue a state.
                best_optimizer_state = copy.deepcopy(decoder_optimizer.state_dict())
                best_batch_weights = batch_weights
                # The attribution must describe the encoder `fit` returns, which is
                # this iteration's, not the last one's.
                best_components = components
                best_trace = trace
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
        self._decoder_optimizer_state = best_optimizer_state
        self._batch_weights = best_batch_weights
        self._encoder_components = best_components
        self._selection_trace = best_trace
        self._selection_path = (
            {name: np.stack(values) for name, values in path_log.items()}
            if path_log["gene"]
            else None
        )

        self._is_fitted = True

        # Store results in AnnData
        self._store_results(adata)

        # Optional stability selection of the encoder's gene sets. Off by default:
        # it adds a fraction of one fit's cost (see BAE.stability_selection).
        if stability_selection:
            self.stability_selection(adata, verbose=verbose)
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

    @_isolates_torch_rng()
    def _iteration_support_frequency(
        self,
        adata: AnnData,
        *,
        n_iterations: int,
        seed: int | None,
        verbose: bool = True,
        continue_optimizer: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Selection frequency over ``n_iterations`` further training iterations.

        Continues the alternating optimization from the fitted state, recording the
        encoder support after each boosting step, then **restores the model**, so the
        call is non-destructive.

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
        from ._utils import (
            resolve_mandatory_genes,
            resolve_precompute_covcache,
            transform_obs_covariates,
        )

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
        # float32, matching `fit`. This path used to promote to float64 on top of
        # the float32 copy above -- three copies of the expression matrix resident
        # at once -- buying a precision difference measured at about one gene in
        # 380, far inside the run-to-run support variation this method documents.
        sourcemat_aug = X_np
        n_nuisance = 0
        if self._regresses_encoder:
            D_nuisance = transform_obs_covariates(adata, self._batch_encoding)
            sourcemat_aug = np.hstack([sourcemat_aug, np.asarray(D_nuisance, dtype=np.float32)])
            n_nuisance = self._batch_encoding.n_columns

        resolved_mandatory = resolve_mandatory_genes(self._mandatory_genes, adata)
        allboost_mandatory = _build_allboost_mandatory(resolved_mandatory, n_nuisance, self.n_genes)
        mandatory_ridge = np.zeros(sourcemat_aug.shape[1], dtype=np.float64)
        if n_nuisance:
            mandatory_ridge[self.n_genes :] = self.config.nuisance_ridge
        prior_dims = self._prior_weights.shape[1] if self._prior_weights is not None else 0
        # Honour the same covariance-cache setting `fit` resolved. Without this,
        # the method documented as mirroring the fit loop would run a different
        # cache strategy from the fit it is analysing.
        boost = self._boost_inputs(
            sourcemat_aug,
            allboost_mandatory,
            mandatory_ridge,
            precompute=resolve_precompute_covcache(
                self.config.boosting_precompute_covcache, sourcemat_aug.shape[1]
            ),
            prior_dims=prior_dims,
        )
        design = (
            self._design_subspaces(adata, list(self._design_blocks), self._design_within)
            if self._design_blocks is not None
            else None
        )

        # Snapshot so the fitted model is unchanged when this returns.
        saved_encoder = self.encoder.linear.weight.detach().clone()
        saved_decoder = {k: v.detach().clone() for k, v in self.decoder.state_dict().items()}

        decoder_optimizer = torch.optim.AdamW(
            self.decoder.parameters(),
            lr=self.config.decoder_lr,
            weight_decay=self.config.decoder_weight_decay,
        )
        if continue_optimizer:
            if self._decoder_optimizer_state is None:
                # Never silent: a loaded checkpoint carries no optimizer state (see
                # `save`), so the same call would otherwise return different numbers
                # depending on whether the model came from disk or from `fit`.
                warnings.warn(
                    "continue_optimizer=True but no AdamW state is available; the "
                    "moments are only kept in memory by `fit` and are not written to "
                    "a checkpoint. Falling back to a fresh optimizer, which is the "
                    "default behaviour. Re-fit in this session to use continuation.",
                    UserWarning,
                    stacklevel=3,
                )
            else:
                decoder_optimizer.load_state_dict(copy.deepcopy(self._decoder_optimizer_state))
                # `load_state_dict` restores `param_groups` too, including the
                # learning rate that was in force during `fit`. That would silently
                # override a caller who lowered `config.decoder_lr` for this phase --
                # the documented way to damp the decoder while counting -- so the
                # current config is re-applied over the restored groups. Only the
                # moment estimates are carried across, which is the point.
                for group in decoder_optimizer.param_groups:
                    group["lr"] = self.config.decoder_lr
                    group["weight_decay"] = self.config.decoder_weight_decay

        from scipy.optimize import linear_sum_assignment

        reference = saved_encoder.detach().cpu().numpy()

        def _unit_rows(matrix: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            return matrix / np.where(norms > 0, norms, 1.0)

        reference_unit = _unit_rows(reference)

        latent_dim = reference.shape[0]
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
                # The same alternation as `fit`, steps 1-6; the boosting half is
                # literally the same code. The recon/design split is not carried
                # here: the targets do not depend on it, only the readout does.
                betamat, _, _, _ = self._boost_encoder(X_train, D_condition, boost, design=design)

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
                    D_train=D_condition,
                )
        finally:
            self.encoder.set_weights(saved_encoder)
            self.decoder.load_state_dict(saved_decoder)

        quality = float(np.mean(match_scores)) if match_scores else float("nan")
        if np.isfinite(quality) and quality < _DIM_MATCH_WARN:
            warnings.warn(
                f"Latent dimensions matched the fitted model poorly across iterations "
                f"(mean |cosine| {quality:.2f}, below {_DIM_MATCH_WARN}). The counted "
                "iterations are not describing one representation, so a frequency "
                "threshold prunes genes that are simply attached to a different "
                "version of the latent space: measured on real data this removed "
                "21-52% of recovered markers at the default threshold. Lower "
                "`threshold` to 0.3-0.5, and prefer frequency.max(axis=1), the flat "
                "union, over the per-dimension split.",
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
        n_runs: int = 300,
        threshold: float = 0.5,
        seed: int | None = None,
        verbose: bool = True,
        continue_optimizer: bool = False,
    ):
        """Stability-select genes for each latent dimension of the fitted model.

        .. admonition:: Exploratory
           :class: caution

           Under active development. This provides **no formal error control**, and
           its per-dimension frequencies are only interpretable when
           ``dim_match_quality`` is high. Defaults have already moved once
           (``threshold`` went from 0.7 to 0.5 in 0.4.0).

        Records how often each gene is selected across ``n_runs`` further training
        iterations continued from the fitted state, then restores the model, so the
        call is non-destructive. It answers: *would these genes still be selected if
        the optimizer had stopped somewhere else on its loss plateau?*

        That is the variance source that dominates here. The encoder support does
        not converge even when the reconstruction loss does — on measured data the
        loss plateaus at ~95% of the achievable linear ceiling while consecutive
        iterations share only about a third of their selected genes. A single fit
        reports one arbitrary position on that walk.

        On simulated data with exact ground truth this gives the lowest
        false-discovery rate of the available readouts (0.26, against 0.31 for cell
        subsampling and 0.38 for a single fit), and its frequencies are empirically
        well calibrated: genes selected in 90-100% of iterations are markers 88% of
        the time. It provides **no formal error control** —
        ``expected_false_positives`` is deliberately ``NaN`` rather than a number
        that would look like a guarantee, because training iterations are neither
        independent nor exchangeable.

        Scope. This measures stability *conditional on the learned representation*.
        It does not capture the variability from re-initializing and refitting the
        autoencoder to a different local optimum, which full refits would. For
        high-stakes marker claims, a handful of full refits remains a worthwhile
        cross-check.

        Parameters
        ----------
        adata
            The data the model was fitted on (same genes; obs columns required only
            if the model used conditioning or nuisance covariates).
        n_runs
            How many further training iterations to average over. The default of
            300 is set by the support autocorrelation, which decays slowly (still
            ~0.45 at lag 100 on measured data), so short windows give highly
            correlated, near-duplicate samples.
        threshold
            Selection-frequency cutoff for the stable support. Default 0.5,
            lowered from 0.7 in 0.4.0.

            0.7 was measured to be too aggressive whenever the latent
            representation is still moving: across three real datasets it removed
            21-52% of recovered marker genes relative to the fitted encoder, and
            0-7% even when the representation had settled. At 0.5 the worst loss
            over the same six runs was 6%. Raising it back toward 0.7-0.9 buys
            precision and is reasonable when ``dim_match_quality`` is high; see
            that field on the result before doing so.

            Note the standalone :func:`structboost.stability_selection` keeps 0.7,
            because its Meinshausen-Buhlmann bound is undefined at or below 0.5.
        seed
            Seeds torch for the continued training iterations.
        verbose
            Show a progress bar. On by default: ``n_runs`` defaults to 300 further
            training steps, which is a long silence.
        continue_optimizer
            Carry the decoder's AdamW moment estimates over from :meth:`fit` instead
            of starting them at zero. Default False, which preserves the behaviour
            every earlier result was measured under.

            The reset is not free. A fresh AdamW restarts its step counter, so bias
            correction begins again and ``exp_avg_sq`` needs on the order of
            ``1/(1 - beta2) = 1000`` steps to become a usable variance estimate.
            An iteration takes ``ceil(n_cells / batch_size)`` steps, so that
            transient spans roughly ``1000 * batch_size / n_cells`` of the counted
            iterations — brief on a large dataset (about 31 of ``n_runs=300`` at
            16,000 cells) but most of the window on a small one (about 250 of 300
            at 2,000 cells), and it falls at the end where the support is furthest
            from equilibrium. Continuing removes it, and the smaller the dataset
            the more it is worth doing.

            The state is the one from the iteration ``fit`` *restored*, not from its
            last iteration, so the moments belong to the decoder actually returned.
            It is held in memory only: :meth:`save` deliberately excludes optimizer
            state, so a model read back from a checkpoint has none and this argument
            warns and falls back rather than silently changing what it measures.

            A caller who lowered ``config.decoder_lr`` for this phase keeps that
            change; only the moment estimates are carried across.

        Returns
        -------
        StabilitySelectionResult
            Stores ``adata.varm["BAE_iteration_frequency"]``, shape
            ``(n_genes, latent_dim)``. Each iteration's dimensions are matched to
            the fitted model's before counting, so a dimension index keeps its
            meaning — see ``dim_match_quality``, and fall back to
            ``frequency.max(axis=1)`` when it is low. A summary is written under
            ``adata.uns["bae"]["stability_selection"]``.

        See Also
        --------
        structboost.stability_selection : The standalone ``allboost``-level
            function, which resamples cells in the Meinshausen-Buhlmann scheme.
            Available for supervised boosting problems that have no training loop
            to iterate over.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if n_runs < 1:
            raise ValueError(f"n_runs must be >= 1, got {n_runs}")
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")

        from ._stability import StabilitySelectionResult

        frequency, avg_selected, quality, coefficients = self._iteration_support_frequency(
            adata,
            n_iterations=n_runs,
            seed=seed,
            verbose=verbose,
            continue_optimizer=continue_optimizer,
        )
        result = StabilitySelectionResult(
            frequency=frequency,
            stable_support=frequency >= threshold,
            threshold=float(threshold),
            avg_selected=avg_selected,
            # Training iterations are neither independent nor exchangeable, so
            # the Meinshausen-Buhlmann bound does not apply. Deliberately NaN
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
            "continue_optimizer": bool(continue_optimizer),
            "threshold": result.threshold,
            "n_iterations": result.n_iterations,
            "n_stable_per_dim": result.stable_support.sum(axis=0).astype(np.intp),
            "avg_selected_per_dim": result.avg_selected,
            "dim_match_quality": quality,
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

    #: Columns with a standard deviation below this are treated as constant and
    #: left unscaled. Testing ``std == 0`` is not enough: a latent dimension
    #: carrying almost no signal has a tiny but nonzero standard deviation, and
    #: dividing by it inflates pure numerical noise to unit variance, handing a
    #: dead dimension the same weight as a real one.
    _STD_EPS: float = 1e-12

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
        cells gives a second, stricter null: the prior programs were fitted on
        that population, so whatever share the novel dimensions still claim there
        is capacity fitting noise rather than structure the prior missed.

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
        :func:`~structboost.linear_ceiling` for the achievable maximum. It is
        written by every fit; only ``reconstruction_loss_by_obs`` and
        ``latent_obs_r2_per_dim`` require a covariate.
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
        if self._training_report is not None:
            uns_dict["training_report"] = self._training_report.to_dict()
        if hasattr(self, "_mandatory_genes") and self._mandatory_genes is not None:
            uns_dict["mandatory_genes"] = _mandatory_genes_for_uns(self._mandatory_genes)
        uns_dict["batch_integration_mode"] = self._batch_integration_mode
        # The *resolved* decision, not the setting: "auto" is the default, so
        # without this a run does not record which strategy it actually used.
        uns_dict["boosting_precompute_covcache"] = bool(self._precomputed_covcache)
        if self._batch_encoding is not None:
            uns_dict["batch_key"] = self._batch_encoding.obs_columns
            uns_dict["batch_columns"] = self._batch_encoding.encoded_columns
        if self._batch_weights is not None:
            uns_dict["batch_weights"] = self._batch_weights
            uns_dict["nuisance_ridge"] = self.config.nuisance_ridge
        from ._utils import latent_r2_per_dim, transform_obs_covariates

        if self._batch_encoding is not None:
            uns_dict["latent_obs_r2_per_dim"] = latent_r2_per_dim(
                transform_obs_covariates(adata, self._batch_encoding), latent
            )
        if self._design_blocks is not None:
            # Same statistic as the batch R² above, opposite goal: near one on the
            # constrained dimensions means the design term did its job. A block's
            # dimensions are scored against what its variable explains beyond the
            # others (the part the filter kept), the free ones against the joint design.
            blocks = self._design_blocks
            Q = self._design_subspaces(adata, list(blocks), self._design_within)
            joint = frozenset(blocks)
            r2 = self._design_r2(Q[joint], Q[frozenset()], latent)
            for v, dims in blocks.items():
                r2[dims] = self._design_r2(Q[joint], Q[joint - {v}], latent[:, dims])
            uns_dict["design_key"] = list(blocks)
            uns_dict["design_blocks"] = {v: d.copy() for v, d in blocks.items()}
            uns_dict["design_dims"] = np.concatenate(list(blocks.values()))
            uns_dict["design_lambda"] = dict(self._design_lambdas)
            if self._design_within is not None:
                uns_dict["design_within"] = list(self._design_within)
            uns_dict["latent_design_r2_per_dim"] = r2
        if self._decompose_columns is not None:
            uns_dict["decompose_key"] = list(self._decompose_columns)
            if self._decompose_within is not None:
                uns_dict["decompose_within"] = list(self._decompose_within)
        if self._selection_path is not None:
            uns_dict["selection_path"] = {k: v.copy() for k, v in self._selection_path.items()}
        if self._encoder_components is not None:
            # Exact split of the encoder: these sum to varm["BAE_encoder_weights"] to
            # within its float32 rounding. Kept in float64: the parts are float64 and
            # casting each separately would break the identity in the last bit.
            for name, W_part in self._encoder_components.items():
                adata.varm[f"BAE_encoder_weights_{name}"] = np.ascontiguousarray(W_part.T)
            uns_dict["attribution_parts"] = list(self._encoder_components)
            trace = copy.deepcopy(self._selection_trace)
            shares = trace.pop("residual_share", None)
            if shares is not None:
                import pandas as pd

                adata.varm["BAE_residual_variance_share"] = pd.DataFrame(
                    shares, index=adata.var_names
                )
            uns_dict["selection_trace"] = trace

        # Always recorded. A bare MSE is not interpretable on its own: on the
        # z-transformed input BAE expects, 1.0 is what predicting zero everywhere
        # scores, so "0.85" reads as a good fit when it means 15% of variance
        # explained. See `linear_ceiling` for the companion question of how much a
        # latent_dim-dimensional model could explain. This used to be written only
        # for covariate fits, which left the documented quality workflow raising
        # KeyError on a plain `fit(adata)` — the metric has nothing to do with
        # covariates, and only the per-group breakdown below does.
        residual, explained = self._reconstruction_stats(adata)
        uns_dict["variance_explained"] = float(explained)

        group_columns = set()
        if self._batch_encoding is not None:
            group_columns.update(self._batch_encoding.obs_columns)
        if group_columns:
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
