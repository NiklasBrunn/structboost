"""Componentwise L2 boosting (pure NumPy port of Julia allboost)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray

_CovarianceCache = NDArray[np.floating] | dict[int, NDArray[np.float64]]


def column_norms_sq(sourcemat: NDArray[np.floating]) -> NDArray[np.floating]:
    """Squared column norms ``||x_j||^2`` of the boosting design matrix.

    Exposed so a caller that invokes :func:`allboost` repeatedly against the
    *same* ``sourcemat`` -- which is what :meth:`structboost.BAE.fit` does, once
    per training iteration -- can compute this once and pass it back in. It is
    otherwise recomputed on every call, an O(n*p) pass that also allocates a full
    ``(n_samples, n_features)`` temporary.

    Using this helper rather than open-coding the expression is what guarantees a
    hoisted value is bit-for-bit the one ``allboost`` would have computed itself.

    ``einsum`` rather than ``(sourcemat**2).sum(axis=0)``: the latter materializes
    a full ``(n_samples, n_features)`` squared copy, 1.4 GB at 60k x 6k, purely to
    reduce it away again.
    """
    return np.einsum("ij,ij->j", sourcemat, sourcemat)


@dataclass
class AllboostHistory:
    """Trace information returned by `allboost(..., return_history=True)`.

    Attributes
    ----------
    selection
        Selected feature index at each step, shape (n_targets, stepno). For a
        target that *replays* another's path (``selection_from``) this holds the
        feature its own criterion would have chosen at that step -- the
        counterfactual pick -- while the leader's pick is what was applied.
    update
        Coefficient increment applied to the selected feature at each step,
        shape (n_targets, stepno). Steps that never ran (see ``selection`` = -1)
        hold 0. Excludes the mandatory pre-step, whose coefficients are not
        selected.
    score
        ``x_j' r`` of the *applied* feature ``j`` against this target's residual
        ``r`` just before the update, shape (n_targets, stepno). For a follower
        this is the applied (leader's) feature scored against the follower's own
        residual, which is what the fit, alignment and gain scores of
        :meth:`structboost.BAE.fit` are built from.
    residual_sq
        ``||r||^2`` of this target's residual just before the update, shape
        (n_targets, stepno). Tracked in O(1) per step from the same quantities.
    rank
        Rank of the applied feature under this target's own selection criterion
        (0 = it was this target's own first choice), shape (n_targets, stepno).
        Always 0 for a target that selects for itself.
    beta_path
        Coefficient path after each step, shape (n_targets, stepno, n_features).
        ``None`` when ``return_history="steps"``: the full path costs
        ``n_targets * stepno * n_features`` floats, which the per-step trace
        above does not.
    """

    selection: NDArray[np.int64]
    update: NDArray[np.float64]
    score: NDArray[np.float64]
    residual_sq: NDArray[np.float64]
    rank: NDArray[np.int64]
    beta_path: NDArray[np.float64] | None = None


def _validate_mandatory_features(
    mandatory_features: NDArray[np.intp] | list[NDArray[np.intp]] | None,
    n_features: int,
    n_targets: int,
) -> list[NDArray[np.intp]]:
    """Normalize and validate mandatory feature indices for each target."""
    if mandatory_features is None:
        return [np.array([], dtype=np.intp) for _ in range(n_targets)]

    if isinstance(mandatory_features, list):
        if len(mandatory_features) != n_targets:
            raise ValueError(
                "mandatory_features list length must match number of targets; "
                f"got {len(mandatory_features)} and {n_targets}"
            )
        raw_per_target = mandatory_features
    else:
        raw_per_target = [mandatory_features] * n_targets

    validated: list[NDArray[np.intp]] = []
    for mand in raw_per_target:
        mand_arr = np.asarray(mand)
        if mand_arr.ndim != 1:
            raise ValueError("mandatory_features entries must be 1D integer arrays")
        # An empty selection carries no dtype information (``np.asarray([])`` is
        # float64), so it is accepted as-is; anything non-empty must be integral.
        if mand_arr.size:
            if mand_arr.dtype == np.bool_:
                raise ValueError(
                    "mandatory_features must contain integer positions, not a boolean "
                    "mask. A mask is silently reinterpreted as the indices 0/1, which "
                    "selects the wrong features. Use np.flatnonzero(mask) instead."
                )
            if not np.issubdtype(mand_arr.dtype, np.integer):
                raise ValueError(
                    "mandatory_features must have an integer dtype, got "
                    f"{mand_arr.dtype}. Non-integer values would be truncated toward "
                    "zero (1.9 -> 1), silently selecting a different feature."
                )
        mand_arr = mand_arr.astype(np.intp, copy=False)
        if np.unique(mand_arr).size != mand_arr.size:
            raise ValueError("mandatory_features contains duplicate indices")
        if ((mand_arr < 0) | (mand_arr >= n_features)).any():
            raise ValueError("mandatory_features index out of range")
        validated.append(mand_arr)

    return validated


class _MandatoryBlock:
    """The per-target constants of the mandatory pre-step.

    ``mand_idx`` is fixed for the whole of a target's boosting run, so the
    covariance columns of the mandatory features, the Gram submatrix among them
    and the ridge penalty do not change between steps -- only the right-hand side
    does. Building them once per target instead of once per *step* is what this
    exists for: the ``column_stack`` alone costs ~270 ms per training iteration at
    20 mandatory features and p=4000, repeated ``stepno`` times for nothing.
    """

    __slots__ = ("cov", "lhs", "penalty")

    def __init__(
        self,
        mand_idx: NDArray[np.intp],
        col_norms_sq: NDArray[np.floating],
        get_covariance_column: Callable[[int], NDArray[np.floating]],
        ridge: NDArray[np.float64],
    ) -> None:
        self.cov = np.column_stack([get_covariance_column(int(j)) for j in mand_idx])
        self.penalty = ridge[mand_idx] * col_norms_sq[mand_idx]
        # The solve's left-hand side is constant across steps too. Deliberately
        # kept as a matrix rather than a factorization: `np.linalg.solve` is
        # O(n_mandatory^3) and therefore free next to the block construction,
        # while swapping in a Cholesky would change results in the last bits.
        self.lhs = self.cov[mand_idx] + np.diag(self.penalty)


def _mandatory_prestep(
    mand_idx: NDArray[np.intp],
    actualnom: NDArray[np.floating],
    beta: NDArray[np.floating],
    col_norms_sq: NDArray[np.floating],
    block: _MandatoryBlock,
    target_index: int = 0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Joint mandatory update, optionally ridge-stabilized.

    The unpenalized case is Binder & Schumacher (2008), Approach B. Ridge
    penalties are expressed relative to each predictor's squared norm.
    """
    if mand_idx.size == 0:
        return actualnom, beta, 0.0

    # Include the derivative of the penalty at the current coefficient. Without
    # this term, repeatedly applying a ridge pre-step would converge back to the
    # unpenalized OLS solution as boosting proceeds.
    nom_mand = actualnom[mand_idx] * col_norms_sq[mand_idx] - block.penalty * beta[mand_idx]
    try:
        gamma_mand = np.linalg.solve(block.lhs, nom_mand)
    except np.linalg.LinAlgError as exc:
        # Deliberately not falling back to a pseudo-inverse or auto-adding ridge:
        # either silently fits a different model than the caller specified.
        raise ValueError(
            f"The mandatory feature block is singular for target {target_index}. "
            f"Mandatory features {mand_idx.tolist()} are exactly collinear (for "
            "example a redundant dummy level, or a covariate duplicated across "
            "columns), so their joint unpenalized update has no unique solution. "
            "Drop the redundant column(s), or pass an explicit mandatory_ridge to "
            "stabilize the block — note that ridge changes the estimates, so it "
            "is not applied automatically."
        ) from exc
    # Change of ||r||^2 from r <- r - X_m gamma, from quantities already at hand:
    # x_m' r is `actualnom * norms` on the block, and the Gram block is `lhs`
    # without its ridge diagonal.
    xr = actualnom[mand_idx] * col_norms_sq[mand_idx]
    gram = block.lhs - np.diag(block.penalty)
    residual_delta = float(gamma_mand @ gram @ gamma_mand - 2.0 * gamma_mand @ xr)
    beta[mand_idx] += gamma_mand
    actualnom -= (block.cov @ gamma_mand) / col_norms_sq

    return actualnom, beta, residual_delta


@overload
def allboost(
    sourcemat: NDArray[np.floating],
    targetmat: NDArray[np.floating],
    *,
    mandatory_features: NDArray[np.intp] | list[NDArray[np.intp]] | None = None,
    mandatory_ridge: float | NDArray[np.floating] = 0.0,
    beta_init: NDArray[np.floating] | None = None,
    covcache: _CovarianceCache | None = None,
    col_norms_sq: NDArray[np.floating] | None = None,
    selection_from: NDArray[np.intp] | None = None,
    stepno: int = 20,
    nu: float = 0.1,
    csf: float = 0.9,
    independent: bool = True,
    return_history: Literal[False] = False,
    return_covcache: Literal[False] = False,
) -> NDArray[np.floating]: ...


@overload
def allboost(
    sourcemat: NDArray[np.floating],
    targetmat: NDArray[np.floating],
    *,
    mandatory_features: NDArray[np.intp] | list[NDArray[np.intp]] | None = None,
    mandatory_ridge: float | NDArray[np.floating] = 0.0,
    beta_init: NDArray[np.floating] | None = None,
    covcache: _CovarianceCache | None = None,
    col_norms_sq: NDArray[np.floating] | None = None,
    selection_from: NDArray[np.intp] | None = None,
    stepno: int = 20,
    nu: float = 0.1,
    csf: float = 0.9,
    independent: bool = True,
    return_history: Literal[True, "steps"],
    return_covcache: Literal[False] = False,
) -> tuple[NDArray[np.floating], AllboostHistory]: ...


@overload
def allboost(
    sourcemat: NDArray[np.floating],
    targetmat: NDArray[np.floating],
    *,
    mandatory_features: NDArray[np.intp] | list[NDArray[np.intp]] | None = None,
    mandatory_ridge: float | NDArray[np.floating] = 0.0,
    beta_init: NDArray[np.floating] | None = None,
    covcache: _CovarianceCache | None = None,
    col_norms_sq: NDArray[np.floating] | None = None,
    selection_from: NDArray[np.intp] | None = None,
    stepno: int = 20,
    nu: float = 0.1,
    csf: float = 0.9,
    independent: bool = True,
    return_history: Literal[False] = False,
    return_covcache: Literal[True] = ...,
) -> tuple[NDArray[np.floating], _CovarianceCache]: ...


@overload
def allboost(
    sourcemat: NDArray[np.floating],
    targetmat: NDArray[np.floating],
    *,
    mandatory_features: NDArray[np.intp] | list[NDArray[np.intp]] | None = None,
    mandatory_ridge: float | NDArray[np.floating] = 0.0,
    beta_init: NDArray[np.floating] | None = None,
    covcache: _CovarianceCache | None = None,
    col_norms_sq: NDArray[np.floating] | None = None,
    selection_from: NDArray[np.intp] | None = None,
    stepno: int = 20,
    nu: float = 0.1,
    csf: float = 0.9,
    independent: bool = True,
    return_history: Literal[True, "steps"],
    return_covcache: Literal[True],
) -> tuple[NDArray[np.floating], AllboostHistory, _CovarianceCache]: ...


def allboost(
    sourcemat: NDArray[np.floating],
    targetmat: NDArray[np.floating],
    *,
    mandatory_features: NDArray[np.intp] | list[NDArray[np.intp]] | None = None,
    mandatory_ridge: float | NDArray[np.floating] = 0.0,
    beta_init: NDArray[np.floating] | None = None,
    covcache: _CovarianceCache | None = None,
    col_norms_sq: NDArray[np.floating] | None = None,
    selection_from: NDArray[np.intp] | None = None,
    stepno: int = 20,
    nu: float = 0.1,
    csf: float = 0.9,
    independent: bool = True,
    return_history: bool | Literal["steps"] = False,
    return_covcache: bool = False,
) -> (
    NDArray[np.floating]
    | tuple[NDArray[np.floating], AllboostHistory]
    | tuple[NDArray[np.floating], _CovarianceCache]
    | tuple[NDArray[np.floating], AllboostHistory, _CovarianceCache]
):
    """Componentwise L2 boosting for multivariate regression.

    Parameters
    ----------
    sourcemat : ndarray of shape (n_samples, n_features)
        Predictor matrix. Standardization (z-transform) is recommended but not
        required; the algorithm uses actual column norms internally.
    targetmat : ndarray of shape (n_samples, n_targets)
        Target matrix. For optimal boosting performance, targets should be
        standardized (zero mean, unit variance per column) before passing to
        this function.
    mandatory_features : ndarray of shape (n_mandatory,) or list of ndarrays, optional
        Features that are forcibly updated in each boosting step using a joint
        multivariate OLS pre-step (Binder & Schumacher, 2008, Approach B).
        Must be integer positions; boolean masks and float arrays are rejected
        rather than silently reinterpreted.
        If an ndarray is provided, it is applied to all targets. If a list is
        provided, it must have length n_targets with one 1D index array per target.

        These features are always part of the unpenalized adjustment block, so
        they are never subject to competitive selection. That is a statement about
        the model specification, not about the fitted values: a mandatory feature
        whose contribution is estimated as zero will have a zero coefficient. If
        you need a guaranteed non-zero support, this is not that mechanism.
    mandatory_ridge : float or ndarray of shape (n_features,), default=0
        Optional non-negative ridge penalty for mandatory coefficients, relative
        to each predictor's squared norm. A scalar applies to all mandatory
        features; an array permits selective stabilization. Zero preserves the
        Binder & Schumacher (2008) unpenalized mandatory update.
    beta_init : ndarray of shape (n_targets, n_features), optional
        Starting coefficients per target. Boosting normally begins from the zero
        model; supplying ``beta_init`` starts it from the offset model
        ``F_0 = sourcemat @ beta_init[t]`` instead, so the procedure fits the
        *correction* to an externally supplied model rather than the model itself.
        This is the classical boosting offset (Bühlmann & Hothorn 2007, Sec. 2)
        and is used by :meth:`BAE.from_reference` to anchor transferred encoder
        weights.

        The learning-rate and penalty state (``nuvec``/``penvec``) still starts
        fresh, so features carrying a non-zero initial coefficient compete as
        though never selected. Pre-ageing the csf state would require the
        selection counts that produced ``beta_init``, which are not part of the
        coefficient matrix. Deviation from ``beta_init`` is therefore bounded by
        ``stepno`` and does not accumulate across repeated calls.
    covcache : ndarray or dict, optional
        Predictor covariance cache. A pre-computed ``X.T @ X`` ndarray uses the
        full-cache fast path. A dict maps feature indices to covariance columns
        and grows only when a feature is selected. If None, an empty column
        cache is created. Reuse the returned cache with the same sourcemat only.
    col_norms_sq : ndarray of shape (n_features,), optional
        Pre-computed squared column norms, as returned by :func:`column_norms_sq`.
        Computed internally when None. Supplying it skips an O(n*p) pass and a
        full ``(n_samples, n_features)`` temporary per call, which matters when
        ``allboost`` is called repeatedly against an unchanging ``sourcemat`` --
        once per training iteration, in :meth:`structboost.BAE.fit`. It carries
        the same staleness contract as ``covcache``: reuse it with the same
        ``sourcemat`` only.
    selection_from : ndarray of shape (n_targets,), optional
        Path replay. Target ``t`` with ``selection_from[t] != t`` does not choose
        its own features: at every step it applies the feature selected for the
        *leader* target ``selection_from[t]`` (which must come earlier, i.e. have
        a lower index), while its own criterion is still evaluated and its argmax
        recorded in ``history.selection[t]`` as the counterfactual pick. The
        learning-rate and penalty adaptation follow the path, so a follower is
        fitted by exactly the linear operator the leader's path defines.

        That linearity is the point: if the leader's target is a sum
        ``t_1 + t_2`` and two followers carry ``t_1`` and ``t_2``, their
        coefficients sum to the leader's, so a coefficient can be attributed
        exactly to the parts of the target that produced it. This is how
        :meth:`structboost.BAE.fit` splits encoder weights into reconstruction
        and design contributions. Requires ``independent=True`` (a follower's
        adaptation would otherwise leak into the shared state) and the same
        ``mandatory_features`` for a follower and its leader.
    stepno : int, default=20
        Number of boosting iterations per target.
    nu : float, default=0.1
        Learning rate. Adapts per feature via csf after each selection.
    csf : float, default=0.9
        Cumulative shrinkage factor.
        After selection: nuvec[j] = 1 - (1 - nuvec[j])^csf.
        csf < 1 promotes diversity, csf > 1 reinforces selected features.
    independent : bool, default=True
        If True, reset learning rate and penalty vectors for each target (no
        cross-target effects). Recommended for marker gene discovery.
        Note: the internal predictor–predictor covariance cache depends only on
        `sourcemat` and is therefore shared across targets for efficiency.
        If False, parameters persist across targets.
    return_history : bool or str, default=False
        If True, also return an `AllboostHistory` object containing the selected
        feature at each step, the increment it received, and the coefficient path
        (after each step). ``"steps"`` returns the same without ``beta_path``,
        which is the memory-heavy part.
    return_covcache : bool, default=False
        If True, also return the (possibly lazily computed) covariance cache.
        Useful when covcache was not provided and you want to reuse it later.

    Returns
    -------
    betamat : ndarray of shape (n_targets, n_features)
        Coefficient matrix (always returned).
    history : AllboostHistory
        Returned as second element if return_history=True.
    covcache_out : ndarray or dict
        Returned as last element if return_covcache=True. When no cache was
        supplied, this is a dict containing only computed covariance columns.

    Notes
    -----
    Features are selected by the penalized variance-reduction (score) criterion
    ``(x_j' r)^2 / (||x_j||^2 + penvec_j)`` — the reduction in penalized RSS from a
    ridge step on feature ``j``. This is scale-invariant in the predictor columns
    and gives unbiased model selection (Hofner et al. 2011).

    The unbiasedness is worth spelling out, because it rests on the *penalty*
    rather than on the criterion alone. Hofner et al. show that boosting selects
    without bias when the base-learners are comparable in degrees of freedom.
    Initializing ``penvec_j = ||x_j||^2 * (1/nu - 1)`` does exactly that: the
    effective ridge degrees of freedom of feature ``j`` are
    ``||x_j||^2 / (||x_j||^2 + penvec_j) = nu``, the same for every feature
    whatever its column norm, and the criterion reduces to
    ``nu * (x_j' r)^2 / ||x_j||^2``. Penalty adaptation (``csf``) then moves
    features off that common footing deliberately, which is the diversity
    mechanism rather than a selection bias. An earlier formulation selected on
    the squared shrunken coefficient ``(x_j' r / (||x_j||^2 + penvec_j))^2``,
    which favors low-norm features and matches the score criterion only for equal
    column norms (standardized predictors). The coefficient *update* is unchanged.

    References
    ----------
    Binder, H. & Schumacher, M. (2009). Incorporating pathway information into
    boosting estimation of high-dimensional risk prediction models.
    *BMC Bioinformatics* 10, 18. (Source of the algorithm: componentwise L2
    boosting with the ``nu``/``csf`` penalty-adaptation mechanism.)

    Binder, H. & Schumacher, M. (2008). Allowing for mandatory covariates in
    boosting estimation of sparse high-dimensional survival models.
    *BMC Bioinformatics* 9, 14. (Mandatory-covariate pre-step used by
    ``mandatory_features``.)

    Hofner, B., Hothorn, T., Kneib, T. & Schmid, M. (2011). A Framework for
    Unbiased Model Selection Based on Boosting. *JCGS* 20(4). (Penalized
    variance-reduction selection criterion.)

    Bühlmann, P. & Hothorn, T. (2007). Boosting Algorithms: Regularization,
    Prediction and Model Fitting. *Statistical Science* 22(4), 477-505.
    (Offset / initial-model formulation used by ``beta_init``.)
    """
    if stepno < 1:
        # Without this, `range(stepno)` simply never runs and the caller gets an
        # all-zero betamat -- including zero coefficients for mandatory features,
        # which look like a fitted model that selected nothing.
        raise ValueError(f"stepno must be >= 1, got {stepno}")

    if return_history not in (True, False, "steps"):
        raise ValueError(f"return_history must be True, False or 'steps', got {return_history!r}")

    n, p = sourcemat.shape
    k = targetmat.shape[1]

    if targetmat.shape[0] != n:
        raise ValueError(
            f"sourcemat and targetmat must have same n_samples, got {n} and {targetmat.shape[0]}"
        )

    # Match the target's dtype to the predictors'. Left alone, a float32
    # `sourcemat` against a float64 `targetmat` makes numpy promote *sourcemat*
    # inside every predictor-target product -- materializing a full float64 copy
    # of the design matrix, measured 20-25x slower than the matched-dtype path.
    # Casting the (n_samples, n_targets) target instead is the cheap direction,
    # and it fixes the precision of the fit at the predictors', which is where
    # the O(n) accumulations that actually limit accuracy happen.
    if (
        np.issubdtype(sourcemat.dtype, np.floating)
        and np.issubdtype(targetmat.dtype, np.floating)
        and sourcemat.dtype != targetmat.dtype
    ):
        targetmat = targetmat.astype(sourcemat.dtype, copy=False)

    mandatory_per_target = _validate_mandatory_features(mandatory_features, p, k)
    ridge = np.asarray(mandatory_ridge, dtype=np.float64)
    if ridge.ndim == 0:
        ridge = np.full(p, float(ridge), dtype=np.float64)
    if ridge.shape != (p,):
        raise ValueError(f"mandatory_ridge must be scalar or shape ({p},), got {ridge.shape}")
    if not np.isfinite(ridge).all() or (ridge < 0).any():
        raise ValueError("mandatory_ridge must contain finite, non-negative values")

    if beta_init is not None:
        beta_init = np.asarray(beta_init, dtype=np.float64)
        if beta_init.shape != (k, p):
            raise ValueError(
                f"beta_init must have shape ({k}, {p}) (n_targets, n_features), "
                f"got {beta_init.shape}"
            )
        if not np.isfinite(beta_init).all():
            raise ValueError("beta_init must contain only finite values")

    # Column squared norms (used for penalty scaling and residual updates). A
    # caller looping over the same `sourcemat` can hoist this out with
    # `column_norms_sq` and hand it back; it is otherwise an O(n*p) pass plus a
    # full (n_samples, n_features) temporary on every single call.
    if col_norms_sq is None:
        col_norms_sq = column_norms_sq(sourcemat)
    else:
        col_norms_sq = np.asarray(col_norms_sq)
        if col_norms_sq.shape != (p,):
            raise ValueError(f"col_norms_sq must have shape ({p},), got {col_norms_sq.shape}")

    # Check for zero-variance columns
    if (col_norms_sq == 0).any():
        raise ValueError(
            "sourcemat contains zero-variance columns. "
            "Remove constant features before calling allboost."
        )

    if selection_from is not None:
        selection_from = np.asarray(selection_from)
        if selection_from.shape != (k,) or not np.issubdtype(selection_from.dtype, np.integer):
            raise ValueError(f"selection_from must be an integer array of shape ({k},)")
        if ((selection_from < 0) | (selection_from > np.arange(k))).any():
            raise ValueError(
                "selection_from[t] must be in [0, t]: a follower can only replay a target "
                "that comes before it"
            )
        if not independent:
            raise ValueError("selection_from requires independent=True")
        for t, leader in enumerate(selection_from):
            if leader != t and not np.array_equal(
                np.sort(mandatory_per_target[t]), np.sort(mandatory_per_target[leader])
            ):
                raise ValueError(
                    f"target {t} replays target {leader} but their mandatory_features differ"
                )
    else:
        selection_from = np.arange(k)

    betamat = np.zeros((k, p), dtype=np.float64)

    # The selection path is always recorded: a follower reads its leader's.
    selection_hist = np.full((k, stepno), -1, dtype=np.int64)
    applied = np.full((k, stepno), -1, dtype=np.int64)
    update_hist = np.zeros((k, stepno), dtype=np.float64)
    score_hist = np.zeros((k, stepno), dtype=np.float64)
    residual_hist = np.zeros((k, stepno), dtype=np.float64)
    rank_hist = np.zeros((k, stepno), dtype=np.int64)
    beta_path: NDArray[np.float64] | None = None
    if return_history is True:
        beta_path = np.zeros((k, stepno, p), dtype=np.float64)

    # Covariance cache: full ndarrays retain the precomputed fast path; the
    # default dict stores only columns that are actually requested.
    if covcache is None:
        covcache = {}

    if isinstance(covcache, dict):
        _covcache = covcache
        for j, column in _covcache.items():
            if not isinstance(j, (int, np.integer)) or not 0 <= j < p:
                raise ValueError(f"covcache column index out of range: {j!r}")
            if np.asarray(column).shape != (p,):
                raise ValueError(
                    f"covcache column {j} must have shape ({p},), got {np.asarray(column).shape}"
                )

        def get_covariance_column(j: int) -> NDArray[np.floating]:
            column = _covcache.get(j)
            if column is None:
                column = np.asarray(sourcemat.T @ sourcemat[:, j], dtype=np.float64)
                _covcache[j] = column
            return column

    else:
        if covcache.shape != (p, p):
            raise ValueError(f"covcache must have shape ({p}, {p}), got {covcache.shape}")
        _covcache = covcache

        # A column needs its NaN scan only once per call: after the first look it
        # is either clean or has just been filled in below, and neither state can
        # revert -- a later fill of some other column writes values, never NaNs.
        # Re-scanning on every access spends an O(p) pass per boosting step for
        # nothing, and the mandatory block alone touches its columns `stepno`
        # times per target.
        _verified = np.zeros(p, dtype=bool)

        def get_covariance_column(j: int) -> NDArray[np.floating]:
            column = _covcache[:, j]
            if not _verified[j]:
                nan_mask = np.isnan(column)
                if nan_mask.any():
                    computed = sourcemat[:, nan_mask].T @ sourcemat[:, j]
                    column[nan_mask] = computed
                    _covcache[j, nan_mask] = computed
                _verified[j] = True
            return column

    # Starting residual correlations for *every* target at once. One
    # (n_features, n_targets) matrix product instead of one matrix-vector product
    # per target: same quantity, but a compute-bound gemm rather than k
    # memory-bound gemv passes over sourcemat, measured ~3.8x faster. With
    # `beta_init` the residual is taken at the offset model rather than at zero --
    # otherwise the first selection step would re-fit signal the offset already
    # explains -- and that too batches into a single product.
    # Leaders and followers go through separate products: a gemm's rounding
    # depends on its width, and a leader must come out bitwise as it would
    # without its followers, or the attribution would perturb the fit it explains.
    leaders = selection_from == np.arange(k)
    groups = [slice(None)] if leaders.all() else [np.flatnonzero(leaders), np.flatnonzero(~leaders)]
    blocks = []
    for idx in groups:
        residuals = targetmat[:, idx]
        if beta_init is not None:
            residuals = residuals - sourcemat @ beta_init[idx].T
        nom = (sourcemat.T @ residuals) / col_norms_sq[:, None]
        blocks.append((idx, nom, np.einsum("ij,ij->j", residuals, residuals, dtype=np.float64)))
    if len(blocks) == 1:
        _, initial_nom, initial_residual_sq = blocks[0]
    else:
        initial_nom = np.empty((p, k), dtype=np.result_type(*(nom for _, nom, _ in blocks)))
        initial_residual_sq = np.empty(k, dtype=np.float64)
        for idx, nom, residual_sq in blocks:
            initial_nom[:, idx] = nom
            initial_residual_sq[idx] = residual_sq

    # Initialize shared state (used if independent=False)
    if not independent:
        nuvec = np.full(p, nu, dtype=np.float64)
        penvec = col_norms_sq * (1.0 / nu - 1.0)

    for t_idx in range(k):
        # Reset parameters for each target if independent
        if independent:
            nuvec = np.full(p, nu, dtype=np.float64)
            penvec = col_norms_sq * (1.0 / nu - 1.0)

        # `.copy()` is required: actualnom is updated in place below.
        actualnom = initial_nom[:, t_idx].copy()
        residual_sq = float(initial_residual_sq[t_idx])
        beta = np.zeros(p, dtype=np.float64) if beta_init is None else beta_init[t_idx].copy()
        mand_idx = mandatory_per_target[t_idx]
        # Built once per target: nothing in it depends on the boosting step.
        mand_block = (
            _MandatoryBlock(mand_idx, col_norms_sq, get_covariance_column, ridge)
            if mand_idx.size > 0
            else None
        )

        for step in range(stepno):
            if mand_block is not None:
                actualnom, beta, residual_delta = _mandatory_prestep(
                    mand_idx,
                    actualnom,
                    beta,
                    col_norms_sq,
                    mand_block,
                    t_idx,
                )
                residual_sq += residual_delta

            if mand_idx.size >= p:
                # Every predictor is mandatory, so there is no eligible candidate to
                # select. The pre-step above has already fitted them jointly; going on
                # would take argmax over an all -inf criterion, land on an arbitrary
                # mandatory index and record it in `selection` as though it had been
                # chosen. The coefficients would barely move (the pre-step drives
                # `actualnom` to zero at mandatory positions, so the spurious update is
                # ~0), but the trace would be wrong and the work wasted.
                if beta_path is not None:
                    beta_path[t_idx, step:, :] = beta
                break

            # Selection criterion: penalized variance-reduction (score) criterion
            #   criterion_j = (x_j' r)^2 / (||x_j||^2 + penvec_j),
            # with x_j' r = beta_j * ||x_j||^2 (= actualnom * col_norms_sq).
            # This is the reduction in the penalized RSS achieved by a ridge step
            # on feature j (equivalently the penalized score statistic). It is
            # scale-invariant in the predictor columns and yields unbiased feature
            # selection (Hofner et al. 2011).
            #
            # An earlier formulation selected on the *squared shrunken
            # coefficient*:
            #   criterion_j = (beta_j * ||x_j||^2 / (||x_j||^2 + penvec_j))^2
            #               = ((x_j' r) / (||x_j||^2 + penvec_j))^2,
            # which carries an extra 1/(||x_j||^2 + penvec_j) factor that favors
            # low-norm features. The two criteria rank features identically only
            # when column norms are equal (standardized predictors) and diverge
            # otherwise. Only the *selection* changed; the coefficient update
            # (nuvec * actualnom) is unchanged.
            numer = actualnom * col_norms_sq  # x_j' r
            criterion = numer**2 / (col_norms_sq + penvec)
            criterion[mand_idx] = -np.inf
            actualsel = int(np.argmax(criterion))
            selection_hist[t_idx, step] = actualsel
            if selection_from[t_idx] != t_idx:
                # Replay: the leader's pick is applied, this target's own pick was
                # recorded above as the counterfactual.
                actualsel = int(applied[selection_from[t_idx], step])
                if actualsel < 0:
                    break
                rank_hist[t_idx, step] = int((criterion > criterion[actualsel]).sum())
            applied[t_idx, step] = actualsel

            # Update the winner only. `actualnom` is refreshed through the cached
            # covariance column rather than by recomputing the residual.
            actualupdate = nuvec[actualsel] * actualnom[actualsel]
            score = float(numer[actualsel])
            update_hist[t_idx, step] = actualupdate
            score_hist[t_idx, step] = score
            residual_hist[t_idx, step] = residual_sq
            # r <- r - u x_j, so ||r||^2 <- ||r||^2 - 2 u x_j'r + u^2 ||x_j||^2.
            residual_sq += actualupdate * (actualupdate * col_norms_sq[actualsel] - 2.0 * score)
            beta[actualsel] += actualupdate
            cov_col = get_covariance_column(actualsel)
            actualnom -= actualupdate * cov_col / col_norms_sq
            # Update adaptive parameters
            nuvec[actualsel] = 1.0 - (1.0 - nuvec[actualsel]) ** csf
            penvec[actualsel] = col_norms_sq[actualsel] * (1.0 / nuvec[actualsel] - 1.0)

            if beta_path is not None:
                beta_path[t_idx, step, :] = beta

        betamat[t_idx, :] = beta

    history = AllboostHistory(
        selection=selection_hist,
        update=update_hist,
        score=score_hist,
        residual_sq=residual_hist,
        rank=rank_hist,
        beta_path=beta_path,
    )
    if return_history and return_covcache:
        return betamat, history, _covcache
    if return_history:
        return betamat, history
    if return_covcache:
        return betamat, _covcache
    return betamat
