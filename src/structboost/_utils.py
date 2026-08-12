"""Utility functions for structboost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from anndata import AnnData


def compute_covariance_cache(
    sourcemat: NDArray[np.floating],
    *,
    out: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Compute the predictor covariance matrix used by allboost.

    This function computes X.T @ X (the Gram matrix), which allboost uses
    internally to update regression coefficients. Pre-computing this matrix
    can provide significant speedups when calling allboost multiple times
    on the same sourcemat (e.g., during cross-validation or hyperparameter
    tuning).

    Note: The output matrix is O(p²) in memory, which can be substantial for
    high-dimensional data.

    Parameters
    ----------
    sourcemat : ndarray of shape (n_samples, n_features)
        Predictor matrix. Standardization is recommended but not required.
    out : ndarray of shape (n_features, n_features), optional
        Pre-allocated output array. If provided, the result is written
        in-place. Must have dtype float64.

    Returns
    -------
    covcache : ndarray of shape (n_features, n_features)
        The covariance/Gram matrix X.T @ X.

    Examples
    --------
    >>> import numpy as np
    >>> from structboost import allboost, compute_covariance_cache
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal((100, 50))
    >>> X = (X - X.mean(axis=0)) / X.std(axis=0)
    >>> covcache = compute_covariance_cache(X)
    >>> # Reuse one cache across calls. Targets are (n_samples, n_targets).
    >>> targets = rng.standard_normal((100, 3))
    >>> beta1 = allboost(X, targets, covcache=covcache)
    >>> beta2 = allboost(X, targets[:, :2], covcache=covcache)
    >>> beta1.shape
    (3, 50)
    """
    p = sourcemat.shape[1]

    if out is not None:
        if out.shape != (p, p):
            raise ValueError(f"out array must have shape ({p}, {p}), got {out.shape}")
        np.dot(sourcemat.T, sourcemat, out=out)
        return out

    # Fortran order, so that ``covcache[:, j]`` -- the only way allboost ever
    # reads this matrix -- is a contiguous vector rather than a stride of 8*p
    # bytes. Reading a column of the C-ordered equivalent touches one cache line
    # per element and measured 4-9x slower inside the boosting loop.
    #
    # The values are identical either way; only the layout differs. numpy will
    # not write a gemm result straight into an F-ordered ``out``, so this costs
    # one transient copy of the p x p matrix at build time -- negligible against
    # the O(n*p^2) product itself, but it does mean the peak allocation here is
    # briefly twice the returned size.
    return np.asfortranarray(np.dot(sourcemat.T, sourcemat), dtype=np.float64)


#: Fraction of total system memory the automatic covariance-cache decision is
#: willing to spend on the p x p Gram matrix. Deliberately conservative: the
#: matrix is transiently doubled while it is built (see
#: :func:`compute_covariance_cache`), and a fit needs room for the expression
#: matrix and the decoder besides.
_PRECOMPUTE_MEMORY_FRACTION = 0.25

#: Used when the platform does not expose its physical memory (non-POSIX).
_PRECOMPUTE_FALLBACK_BUDGET = 2 * 1024**3


def _total_memory_bytes() -> int | None:
    """Physical memory, or None where the platform will not say."""
    try:
        import os

        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, AttributeError, OSError):  # pragma: no cover - platform dependent
        return None


def resolve_precompute_covcache(setting: bool | str, n_features: int) -> bool:
    """Decide whether to build the full covariance matrix up front.

    ``True``/``False`` are honoured exactly; ``"auto"`` precomputes whenever the
    ``8 * n_features**2`` byte matrix fits the memory budget above.

    Precomputing is not merely a way to avoid recomputing columns. Building all
    ``p`` columns at once is a single compute-bound matrix product running near
    hardware peak, whereas fetching them one at a time is a sequence of
    memory-bound matrix-vector products; measured on real data the former wins
    from roughly ``p/59`` distinct selected features onwards, and a BAE fit
    passes that in its first iteration (``boosting_stepno * latent_dim``
    candidates). The decision here is therefore about memory, not about how many
    columns will be needed.
    """
    if setting is True or setting is False:
        return bool(setting)
    if setting != "auto":
        raise ValueError(
            f"boosting_precompute_covcache must be True, False or 'auto', got {setting!r}"
        )
    total = _total_memory_bytes()
    budget = (
        int(total * _PRECOMPUTE_MEMORY_FRACTION)
        if total is not None
        else _PRECOMPUTE_FALLBACK_BUDGET
    )
    return 8 * int(n_features) ** 2 <= budget


def disentangle_boosting_targets(
    targets: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Orthogonalize gradient vectors across latent dimensions.

    For each gradient vector (column), computes the residual after removing
    the linear projection onto the subspace spanned by all other columns.
    This encourages the subsequent boosting step to learn encoder weights
    that produce disentangled latent representations.

    The regression carries **no intercept**: columns are projected through the
    origin, not about their means. On centered targets that is the textbook
    residualization — in two dimensions it flips the correlation's sign and
    preserves its magnitude exactly. On targets with a substantial common offset
    it is not: a shared mean dominates the projection, and measured on a synthetic
    pair a shift of +5 turned a correlation of +0.64 into -0.999 rather than the
    -0.64 the centered case gives. BAE's targets ``z*`` are near-centered because
    ``z = X @ W`` on z-scored ``X`` is, so this is ordinarily a non-issue; it is
    documented because nothing enforces it.

    Parameters
    ----------
    targets
        Boosting target matrix of shape (n_samples, latent_dim).
        Each column is the negative gradient for one latent dimension.

    Returns
    -------
    Orthogonalized target matrix of shape (n_samples, latent_dim).
    """
    n_dims = targets.shape[1]
    if n_dims == 1:
        return targets.copy()

    work = targets
    result = np.empty_like(work)
    for j in range(n_dims):
        other_cols = np.delete(work, j, axis=1)
        y = work[:, j]
        coeffs, *_ = np.linalg.lstsq(other_cols, y, rcond=None)
        fitted = other_cols @ coeffs
        result[:, j] = y - fitted

    return result.astype(targets.dtype, copy=False)


def _pca_scores(
    X: NDArray[np.floating], n_components: int, *, center: bool = True
) -> NDArray[np.float64]:
    """Top-``n_components`` principal component scores of X, via SVD.

    Columns are **never** rescaled. BAE's input contract is z-transformed data, so
    dividing by column standard deviations here would silently apply a second
    transformation on top of one the caller already performed — and on data that is
    *not* standardized it would mask the condition that
    ``BAE._check_standardized`` exists to warn about.

    Parameters
    ----------
    X
        Data matrix of shape (n_samples, n_features).
    n_components
        Number of components to keep. Must not exceed ``min(X.shape)``.
    center
        Subtract the column means before decomposing. Required for PCA to be
        meaningful on raw data, but pass ``center=False`` when X is already
        z-transformed so that no further transformation is applied to it.

    Returns
    -------
    ndarray of shape (n_samples, n_components)
        Component scores, ordered by decreasing explained variance.

    Raises
    ------
    ValueError
        If ``n_components`` is not in ``[1, min(X.shape)]``.
    """
    n_max = min(X.shape)
    if not 1 <= n_components <= n_max:
        raise ValueError(
            f"n_components must be in [1, {n_max}] for data of shape {X.shape}, got {n_components}"
        )
    work = X - X.mean(axis=0, keepdims=True) if center else X
    u, s, _ = np.linalg.svd(np.asarray(work, dtype=np.float64), full_matrices=False)
    return u[:, :n_components] * s[:n_components]


@dataclass
class ObsCovariateEncoding:
    """Stores encoding parameters for obs covariates.

    Attributes
    ----------
    encoded
        Standardized encoded matrix, shape (n_samples, n_dummy_cols).
    column_info
        Per-column metadata: type ("categorical"/"numeric"), categories, dummy col indices.
    mean
        Column means used for standardization.
    std
        Column stds used for standardization.
    obs_columns
        Original obs column names.
    n_columns
        Total number of encoded columns.
    encoded_columns
        Human-readable names of the encoded columns.
    """

    encoded: NDArray[np.floating]
    column_info: dict[str, dict]
    mean: NDArray[np.floating]
    std: NDArray[np.floating]
    obs_columns: list[str]
    n_columns: int
    encoded_columns: list[str]


def encode_obs_covariates(
    adata: AnnData,
    obs_columns: list[str],
) -> ObsCovariateEncoding:
    """Encode obs covariates as a standardized numeric matrix.

    Categorical, string, object, and boolean columns are dummy-encoded using
    the first observed category as reference. Numeric columns are kept as-is.
    The result is z-standardized.

    Parameters
    ----------
    adata
        AnnData object with obs DataFrame.
    obs_columns
        Column names in adata.obs to encode.

    Returns
    -------
    ObsCovariateEncoding with the encoded matrix and metadata.

    Raises
    ------
    ValueError
        If a column is missing, contains missing/non-finite values, is constant,
        or if the resulting design matrix is rank-deficient.
    """
    import pandas as pd

    missing = [c for c in obs_columns if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"Columns {missing} not found in adata.obs")

    if not obs_columns:
        raise ValueError("obs_columns must contain at least one column")

    parts: list[NDArray[np.floating]] = []
    column_info: dict[str, dict] = {}
    encoded_columns: list[str] = []

    for col in obs_columns:
        series = adata.obs[col]
        if series.isna().any():
            raise ValueError(f"Column {col!r} contains missing values")
        is_categorical = (
            isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series.dtype)
            or pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
        )
        if is_categorical:
            if isinstance(series.dtype, pd.CategoricalDtype):
                cats = list(series.cat.categories)
            else:
                cats = list(pd.unique(series))
            if len(cats) < 2:
                raise ValueError(f"Column {col!r} is constant")
            categorical = pd.Series(
                pd.Categorical(series, categories=cats),
                index=series.index,
            )
            dummies = pd.get_dummies(categorical, drop_first=True).to_numpy(dtype=np.float64)
            names = [f"{col}[{cat}]" for cat in cats[1:]]
            column_info[col] = {
                "type": "categorical",
                "categories": cats,
                "n_dummies": dummies.shape[1],
                "encoded_columns": names,
            }
            parts.append(dummies)
            encoded_columns.extend(names)
        else:
            try:
                vals = series.to_numpy(dtype=np.float64).reshape(-1, 1)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Column {col!r} cannot be encoded as numeric") from exc
            if not np.isfinite(vals).all():
                raise ValueError(f"Column {col!r} contains non-finite values")
            if vals.std() < 1e-12:
                raise ValueError(f"Column {col!r} is constant")
            column_info[col] = {
                "type": "numeric",
                "n_dummies": 1,
                "encoded_columns": [col],
            }
            parts.append(vals)
            encoded_columns.append(col)

    raw = np.hstack(parts)
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    if np.any(std < 1e-12):
        raise ValueError("Encoded covariate design contains a constant column")
    encoded = (raw - mean) / std
    rank = np.linalg.matrix_rank(encoded)
    if rank < encoded.shape[1]:
        raise ValueError(
            f"Encoded covariate design is rank-deficient (rank {rank} < {encoded.shape[1]} columns)"
        )

    return ObsCovariateEncoding(
        encoded=encoded,
        column_info=column_info,
        mean=mean,
        std=std,
        obs_columns=list(obs_columns),
        n_columns=encoded.shape[1],
        encoded_columns=encoded_columns,
    )


def transform_obs_covariates(
    adata: AnnData,
    encoding: ObsCovariateEncoding,
) -> NDArray[np.floating]:
    """Apply a stored obs covariate encoding to new data.

    Parameters
    ----------
    adata
        New AnnData object with the same obs columns.
    encoding
        Encoding from a previous ``encode_obs_covariates`` call.

    Returns
    -------
    Encoded matrix of shape (n_samples, n_columns), standardized with
    the training mean/std.

    Raises
    ------
    ValueError
        If required columns are missing, contain missing/non-finite values, or
        contain a categorical level not observed during fitting.
    """
    import pandas as pd

    missing = [c for c in encoding.obs_columns if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"Columns {missing} not found in adata.obs")

    parts: list[NDArray[np.floating]] = []
    for col in encoding.obs_columns:
        info = encoding.column_info[col]
        series = adata.obs[col]
        if series.isna().any():
            raise ValueError(f"Column {col!r} contains missing values")
        if info["type"] == "categorical":
            cats = info["categories"]
            unknown = [value for value in pd.unique(series) if value not in cats]
            if unknown:
                raise ValueError(
                    f"Column {col!r} contains levels not seen during fitting: {unknown}"
                )
            categorical = pd.Series(
                pd.Categorical(series, categories=cats),
                index=series.index,
            )
            dummies = pd.get_dummies(categorical, drop_first=True).to_numpy(dtype=np.float64)
            parts.append(dummies)
        else:
            vals = series.to_numpy(dtype=np.float64).reshape(-1, 1)
            if not np.isfinite(vals).all():
                raise ValueError(f"Column {col!r} contains non-finite values")
            parts.append(vals)

    raw = np.hstack(parts)
    return (raw - encoding.mean) / encoding.std


def _resolve_flat_mandatory(
    entries: list[str] | list[int] | NDArray[np.intp],
    adata: AnnData,
) -> NDArray[np.intp]:
    """Resolve a flat list of gene names or indices to intp array."""
    if isinstance(entries, np.ndarray) and entries.dtype == np.intp:
        return entries

    indices: list[int] = []
    for entry in entries:
        if isinstance(entry, str):
            if entry not in adata.var_names:
                raise ValueError(f"Gene '{entry}' not found in adata.var_names")
            indices.append(adata.var_names.get_loc(entry))
        else:
            indices.append(int(entry))
    return np.array(indices, dtype=np.intp)


def resolve_mandatory_genes(
    mandatory_genes: list[str]
    | list[int]
    | NDArray[np.intp]
    | list[list[str] | list[int] | NDArray[np.intp]]
    | None,
    adata: AnnData,
) -> NDArray[np.intp] | list[NDArray[np.intp]] | None:
    """Resolve mandatory gene specifications to integer indices.

    Parameters
    ----------
    mandatory_genes
        Gene names (str) or column indices (int), or per-target lists thereof.
        None means no mandatory genes.
    adata
        AnnData with var_names for name resolution.

    Returns
    -------
    Resolved indices as a 1D intp array (global) or list of 1D intp arrays
    (per-target). None if input is None.
    """
    if mandatory_genes is None:
        return None

    # Check if it's a list of lists (per-target)
    if (
        isinstance(mandatory_genes, list)
        and len(mandatory_genes) > 0
        and isinstance(mandatory_genes[0], (list, np.ndarray))
    ):
        return [_resolve_flat_mandatory(sub, adata) for sub in mandatory_genes]

    return _resolve_flat_mandatory(mandatory_genes, adata)


def linear_ceiling(
    adata: AnnData,
    n_components: int,
    *,
    layer: str | None = None,
) -> float:
    """Variance explainable by the best ``n_components``-dimensional linear model.

    A reconstruction MSE means little on its own. On the z-transformed input BAE
    expects, predicting zero everywhere scores exactly 1.0, so an MSE of 0.85 is
    15% of variance explained, not "close to perfect". Even that number needs a
    reference: most per-gene variance in scRNA-seq is dropout and sampling noise,
    so no ``n_components``-dimensional model can reach 1. This returns that
    reference — the fraction captured by the leading ``n_components`` principal
    components, which upper-bounds any linear encoder of the same width.

    Compare it against ``adata.uns["bae"]["variance_explained"]``. A fit sitting
    near the ceiling is doing as well as its latent budget allows and should be
    given more dimensions rather than more iterations; one far below it is
    underfitting, and ``boosting_stepno`` or ``max_iterations`` is the lever.

    Parameters
    ----------
    adata
        AnnData object. Uses the same matrix BAE was fitted on.
    n_components
        Latent width to compare against, i.e. ``config.latent_dim``.
    layer
        Optional ``adata.layers`` key to use instead of ``adata.X``. Pass the
        same layer the model was fitted with (``adata.uns["bae"]["layer"]``, or
        absent when the fit read ``adata.X``); a ceiling computed on a different
        matrix is not comparable with the model's reconstruction.

    Returns
    -------
    Fraction of total variance in ``[0, 1]`` captured by the top
    ``n_components`` principal components.

    See Also
    --------
    structboost.BAE.fit : Writes ``uns["bae"]["variance_explained"]`` to compare
        against this ceiling.

    Examples
    --------
    >>> ceiling = linear_ceiling(adata, adata.uns["bae"]["latent_dim"])
    >>> got = adata.uns["bae"]["variance_explained"]
    >>> print(f"BAE reaches {100 * got / ceiling:.0f}% of the linear ceiling")
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import LinearOperator, svds

    matrix = adata.layers[layer] if layer is not None else adata.X
    n_obs, n_vars = matrix.shape
    max_rank = min(n_obs, n_vars)
    if not 1 <= n_components <= max_rank:
        raise ValueError(
            f"n_components must be in [1, {max_rank}] for data of shape "
            f"{(n_obs, n_vars)}, got {n_components}"
        )

    mean = np.asarray(matrix.mean(axis=0), dtype=np.float64).ravel()
    # Total sum of squares about the per-gene mean, without densifying: for sparse
    # X, ||X - 1 mu'||_F^2 = ||X||_F^2 - n_obs ||mu||^2.
    if sp.issparse(matrix):
        sum_squares = float(matrix.multiply(matrix).sum())
    else:
        sum_squares = float(np.square(np.asarray(matrix, dtype=np.float64)).sum())
    ss_total = sum_squares - n_obs * float(mean @ mean)
    if ss_total <= 0:
        return float("nan")

    # `svds` cannot return every singular value, so fall back to a dense
    # decomposition when the full spectrum is requested (only feasible, and only
    # asked for, on small data).
    if n_components >= max_rank - 1:
        dense = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
        singular = np.linalg.svd(np.asarray(dense, dtype=np.float64) - mean, compute_uv=False)
    else:
        # Centering a sparse matrix would densify it; apply it as an operator.
        def _matvec(v):
            return matrix @ v - mean @ v

        def _rmatvec(v):
            return np.asarray(matrix.T @ v).ravel() - mean * v.sum()

        centered = LinearOperator(
            (n_obs, n_vars), matvec=_matvec, rmatvec=_rmatvec, dtype=np.float64
        )
        singular = svds(centered, k=n_components, return_singular_vectors=False)

    top = np.sort(singular)[::-1][:n_components]
    return float(min(np.square(top).sum() / ss_total, 1.0))
