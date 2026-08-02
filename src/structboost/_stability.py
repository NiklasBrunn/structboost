"""Stability selection for boosting-based gene selection.

Implements Meinshausen & Bühlmann (2010) stability selection wrapped around
``allboost``. The selector is re-run on many random subsamples of the cells and a
gene is called *stable* for a latent dimension when it is selected in at least a
fraction ``threshold`` of the subsamples. This turns a single, seed-dependent
gene list into a per-gene selection frequency plus a bound on the expected number
of false selections.

Following the paper, subsamples are drawn of size ``floor(subsample_frac * n)``
**without replacement** (``subsample_frac=0.5`` is the value the theory is derived
for). Bootstrap resampling is deliberately not offered: sampling with replacement
breaks the exchangeability the error bound relies on.

Reference
---------
Meinshausen, N. & Bühlmann, P. (2010). Stability selection. *Journal of the Royal
Statistical Society: Series B*, 72(4), 417-473.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._boosting import allboost

#: Boolean mask array. Named rather than spelled ``NDArray[np.bool_]`` inline:
#: the trailing underscore in ``np.bool_`` is valid Python but reads as reference
#: syntax to the documentation builder, which then reports a broken target on
#: every page that renders the annotation.
BoolArray = NDArray[np.bool_]


def _mandatory_gene_counts(
    mandatory_features: NDArray[np.intp] | list[NDArray[np.intp]] | None,
    latent_dim: int,
    p_genes: int,
) -> NDArray[np.float64]:
    """Per-dimension count of forced features that fall in the gene columns.

    Mirrors ``allboost``'s ``mandatory_features`` shapes: ``None``, a single array
    applied to every target, or one array per target. Only indices ``< p_genes``
    are counted (nuisance columns beyond the genes are already excluded elsewhere).
    """
    counts = np.zeros(latent_dim, dtype=np.float64)
    if mandatory_features is None:
        return counts
    if isinstance(mandatory_features, list):
        for j, mand in enumerate(mandatory_features):
            counts[j] = int((np.asarray(mand) < p_genes).sum())
    else:
        counts[:] = int((np.asarray(mandatory_features) < p_genes).sum())
    return counts


def _coefficient_statistics(
    coef_sum: NDArray[np.float64],
    coef_sq_sum: NDArray[np.float64],
    positive_count: NDArray[np.float64],
    counts: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Conditional mean, sd and sign consistency from streaming accumulators.

    All three condition on the runs in which an entry was actually selected, so an
    entry selected twice out of a hundred reports the spread of those two draws
    rather than being diluted by ninety-eight zeros.
    """
    safe = np.maximum(counts, 1.0)
    cond_mean = np.divide(coef_sum, safe, out=np.zeros_like(coef_sum), where=counts > 0)
    variance = np.maximum(coef_sq_sum / safe - cond_mean**2, 0.0)
    cond_sd = np.where(counts > 1, np.sqrt(variance), 0.0)
    modal = np.maximum(positive_count, counts - positive_count)
    sign_consistency = np.divide(modal, safe, out=np.ones_like(coef_sum), where=counts > 0)
    return cond_mean, cond_sd, sign_consistency


@dataclass(frozen=True)
class StabilitySelectionResult:
    """Result of :func:`structboost.stability_selection`.

    Attributes
    ----------
    frequency
        Per-gene, per-dimension selection frequency, shape ``(n_genes, latent_dim)``.
        Entry ``(g, j)`` is the fraction of subsamples in which gene ``g`` received
        a nonzero coefficient for latent dimension ``j``.
    stable_support
        Boolean mask ``frequency >= threshold``, shape ``(n_genes, latent_dim)``.
    threshold
        The selection-frequency threshold ``pi`` used for ``stable_support``.
    avg_selected
        Mean number of genes selected per subsample, per dimension, shape
        ``(latent_dim,)``. This is the ``q`` in the error bound below.
    expected_false_positives
        Meinshausen-Bühlmann upper bound on the expected number of falsely stable
        genes per dimension, ``E[V] <= q^2 / ((2*threshold - 1) * p)``, shape
        ``(latent_dim,)``. Here ``q`` is the average number of *competitively*
        selected genes and ``p`` the number of candidate genes; forced (mandatory)
        genes are excluded from both, since they are always selected by
        construction and are not candidates. ``NaN`` when ``threshold <= 0.5`` (the
        bound is undefined there) or when every gene is mandatory. The bound holds
        under the paper's exchangeability and "no worse than random guessing"
        assumptions, which real data may violate; treat it as a guideline, not a
        guarantee.
    n_subsamples
        Number of subsamples drawn. ``0`` for ``mode="iteration"``, which draws no
        subsamples.
    subsample_frac
        Fraction of cells in each subsample. ``NaN`` for ``mode="iteration"``.
    mode
        Which resampling scheme produced the frequencies.

        ``"subsample"``
            Meinshausen-Bühlmann cell subsampling with the model frozen. Targets
            variance under *cell resampling*. ``frequency`` is per latent dimension
            and ``expected_false_positives`` carries the paper's error bound.
        ``"iteration"``
            Selection frequency across additional *training iterations* of a single
            fit. Targets variance from the optimizer's position on its loss plateau,
            which is a different — and on measured data, larger — source of
            instability. ``frequency`` is per latent dimension, with each
            iteration's dimensions matched to the fitted model's before counting —
            see ``dim_match_quality``. ``expected_false_positives`` is ``NaN``:
            training iterations are neither independent nor exchangeable, so the
            Meinshausen-Bühlmann bound does not apply and no error control is
            claimed.
    n_iterations
        Number of training iterations averaged over. ``0`` for ``mode="subsample"``.
    coefficient_cond_mean
        Shape ``(n_genes, latent_dim)``, or ``None`` when coefficients were not
        collected. Mean coefficient over the runs in which each entry was
        *selected* — reliability-blind but scale-preserving, since its expectation
        equals the per-run coefficient. Feeds :meth:`stable_encoder`.
    coefficient_sd
        Standard deviation of the same conditional distribution. This is a
        **spread, not a standard error**: subsample runs share half their cells by
        construction and iteration runs are autocorrelated, so the effective sample
        size is far below ``n_runs`` and ``sd / sqrt(n_runs)`` would be badly
        overconfident. Reporting it as a precision would require a block bootstrap.
    sign_consistency
        Fraction of selecting runs in which each coefficient took its modal sign.
        A gene selected every time with a sign that flips is not stable, which a
        selection frequency alone cannot reveal. Measured at ~0.999 on real data,
        so in practice a guard rather than a headline.
    dim_match_quality
        ``mode="iteration"`` only: mean absolute cosine similarity between each
        iteration's latent dimensions and the fitted model's, after optimal
        matching. Per-dimension frequencies are only meaningful when dimensions
        keep their identity across iterations, and this quantifies that rather than
        assuming it. Near 1 means the anchoring held; low values mean the
        per-dimension split should not be trusted and ``frequency.max(axis=1)``
        (the flat union) is the safer readout. ``NaN`` for ``mode="subsample"``.
    """

    frequency: NDArray[np.float64]
    stable_support: BoolArray
    threshold: float
    avg_selected: NDArray[np.float64]
    expected_false_positives: NDArray[np.float64]
    n_subsamples: int
    subsample_frac: float
    mode: str = "subsample"
    n_iterations: int = 0
    dim_match_quality: float = float("nan")
    coefficient_cond_mean: NDArray[np.float64] | None = None
    coefficient_sd: NDArray[np.float64] | None = None
    sign_consistency: NDArray[np.float64] | None = None

    @property
    def n_runs(self) -> int:
        """Number of resampling runs, whichever mode produced this result."""
        return self.n_iterations if self.mode == "iteration" else self.n_subsamples

    @property
    def coefficient_mean(self) -> NDArray[np.float64] | None:
        """Mean coefficient over *all* runs, counting unselected runs as zero.

        Equals ``coefficient_cond_mean * frequency``, so a gene selected in 40% of
        runs is shrunk to 40% of its typical weight — shrinkage proportional to
        reliability. Statistically the more principled estimator, but it changes the
        latent scale the decoder was trained against, which is why
        :meth:`stable_encoder` does not offer it.
        """
        if self.coefficient_cond_mean is None:
            return None
        return self.coefficient_cond_mean * self.frequency

    def stable_encoder(self) -> NDArray[np.float64]:
        """Aggregate the per-run coefficients into a more reproducible encoder.

        Among stably selected entries the run-to-run coefficient spread is large
        (CV ~0.34 median on measured data), so the *support* of a fit is more
        reproducible than its *weights*. This returns the conditional mean
        coefficient restricted to the stable support: the frequency threshold
        decides *which* genes, the conditional mean decides *how much*.

        Measured on simulated data against exact ground truth, with the decoder
        left untouched, it improves on the fitted encoder on every axis that
        matters for selection:

        ==================  ==========  =========  ======
        encoder             precision   FDR        genes
        ==================  ==========  =========  ======
        fitted              0.62        0.38       156
        this                **0.74**    **0.26**   116
        ==================  ==========  =========  ======

        and raises reconstruction from 55% to 59% of the linear ceiling. The same
        ordering holds under batch conditioning (precision 0.754 vs 0.691, FDR
        0.246 vs 0.309). Because the latent scale is preserved, the result can be
        installed with :meth:`BAE.apply_encoder` **without refitting the decoder**.

        To trade precision for recall, lower ``threshold`` when calling
        :meth:`BAE.stability_selection` rather than reaching for a different
        estimator — the threshold *is* that dial. An unmasked conditional mean is
        simply this with ``threshold`` at zero, and it was measured to be a worse
        selector at every setting tried (precision 0.47 unconditioned, 0.56
        conditioned) as well as numerically fragile on batch-dominated fits, where
        it produced a reconstruction worse than predicting zero.

        Returns
        -------
        ndarray of shape ``(n_genes, latent_dim)``
            Same orientation as ``frequency`` and ``adata.varm``.

        Raises
        ------
        ValueError
            If coefficients were not collected for this result.
        """
        if self.coefficient_cond_mean is None:
            raise ValueError(
                "coefficients were not collected for this result; "
                "stable_encoder() needs a result produced by BAE.stability_selection"
            )
        return np.where(self.stable_support, self.coefficient_cond_mean, 0.0)


def stability_selection(
    sourcemat: NDArray[np.floating],
    targetmat: NDArray[np.floating],
    *,
    n_genes: int | None = None,
    mandatory_features: NDArray[np.intp] | list[NDArray[np.intp]] | None = None,
    mandatory_ridge: float | NDArray[np.floating] = 0.0,
    n_subsamples: int = 100,
    subsample_frac: float = 0.5,
    threshold: float = 0.7,
    stepno: int = 20,
    nu: float = 0.1,
    csf: float = 0.9,
    independent: bool = True,
    seed: int | None = None,
    verbose: bool = False,
) -> StabilitySelectionResult:
    """Run stability selection over subsamples of ``allboost``.

    Parameters
    ----------
    sourcemat
        Predictor matrix, shape ``(n_samples, n_features)``. May include trailing
        nuisance columns; only the first ``n_genes`` columns are scored.
    targetmat
        Target matrix, shape ``(n_samples, latent_dim)``. For BAE these are the
        functional-gradient targets ``z*`` (see :meth:`BAE.stability_selection`),
        not the latent codes themselves — the codes are a sparse linear function of
        the already-selected genes, which makes selecting them nearly circular.
    n_genes
        Number of leading columns of ``sourcemat`` that are genes. Frequencies are
        reported only for these. Defaults to all columns.
    mandatory_features, mandatory_ridge
        Forwarded to :func:`allboost`, so the resampled problem matches the one the
        encoder solved (e.g. batch nuisance regressors). Mandatory columns beyond
        ``n_genes`` never appear in the reported frequencies.
    n_subsamples
        Number of subsamples (``B``). More reduces Monte-Carlo noise in the
        frequencies; cost is linear.
    subsample_frac
        Fraction of cells per subsample, drawn without replacement. ``0.5`` is the
        value Meinshausen-Bühlmann derive the error bound for; other values give
        frequencies but weaken the bound's justification.
    threshold
        Stability threshold ``pi`` in ``(0.5, 1]``. Genes selected in at least this
        fraction of subsamples form the stable support. Must exceed ``0.5`` for the
        error bound to be defined.
    stepno, nu, csf, independent
        Boosting hyperparameters, forwarded to :func:`allboost`. Use the same
        values the encoder was fitted with.
    seed
        Seed for the subsampling RNG.
    verbose
        Show a progress bar over the subsamples. Defaults to False so that
        calling this function directly stays silent;
        :meth:`structboost.BAE.stability_selection` passes its own ``verbose``
        through.

    Returns
    -------
    StabilitySelectionResult
    """
    sourcemat = np.asarray(sourcemat, dtype=np.float64)
    targetmat = np.asarray(targetmat, dtype=np.float64)
    n = sourcemat.shape[0]
    if targetmat.shape[0] != n:
        raise ValueError(
            f"sourcemat and targetmat must share n_samples, got {n} and {targetmat.shape[0]}"
        )
    latent_dim = targetmat.shape[1]
    p_genes = sourcemat.shape[1] if n_genes is None else int(n_genes)
    if not 1 <= p_genes <= sourcemat.shape[1]:
        raise ValueError(f"n_genes must be in [1, {sourcemat.shape[1]}], got {p_genes}")
    if n_subsamples < 1:
        raise ValueError(f"n_subsamples must be >= 1, got {n_subsamples}")
    if not 0.0 < subsample_frac < 1.0:
        raise ValueError(f"subsample_frac must be in (0, 1), got {subsample_frac}")
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")

    sub_n = int(np.floor(subsample_frac * n))
    if sub_n < 2:
        raise ValueError(
            f"subsample_frac={subsample_frac} gives {sub_n} cells; too few to fit. "
            "Increase subsample_frac or use more cells."
        )

    rng = np.random.default_rng(seed)
    counts = np.zeros((p_genes, latent_dim), dtype=np.float64)
    selected_per_subsample = np.zeros(latent_dim, dtype=np.float64)
    # Streaming accumulators: the full coefficient trace would be
    # n_subsamples x p_genes x latent_dim floats, which is infeasible at scale.
    # Signs need no alignment here because every subsample fits the same targets.
    coef_sum = np.zeros((p_genes, latent_dim), dtype=np.float64)
    coef_sq_sum = np.zeros((p_genes, latent_dim), dtype=np.float64)
    positive_count = np.zeros((p_genes, latent_dim), dtype=np.float64)

    # tqdm ships in the `[bae]` extra while this module is part of the NumPy-only
    # core, so it is imported lazily *and* only when a bar was actually asked for.
    # Importing it unconditionally would make a quiet call fail on a core install.
    runs: Iterable[int] = range(n_subsamples)
    if verbose:
        from tqdm import tqdm

        runs = tqdm(runs, desc="Stability selection (subsample)", unit="run")

    for _ in runs:
        idx = rng.choice(n, size=sub_n, replace=False)
        # A fresh covcache per subsample is mandatory: the Gram matrix depends on
        # the rows, so the full-data cache would silently produce wrong updates.
        betamat = allboost(
            sourcemat[idx],
            targetmat[idx],
            mandatory_features=mandatory_features,
            mandatory_ridge=mandatory_ridge,
            stepno=stepno,
            nu=nu,
            csf=csf,
            independent=independent,
        )
        beta_genes = betamat[:, :p_genes].T  # (n_genes, latent_dim)
        selected = np.abs(betamat[:, :p_genes]) > 0  # (latent_dim, n_genes)
        counts += selected.T
        selected_per_subsample += selected.sum(axis=1)
        coef_sum += beta_genes
        coef_sq_sum += beta_genes**2
        positive_count += beta_genes > 0

    frequency = counts / n_subsamples
    avg_selected = selected_per_subsample / n_subsamples
    stable_support = frequency >= threshold

    if threshold > 0.5:
        # Forced (mandatory) genes are selected in every subsample by construction:
        # their frequency-1.0 is not evidence of stability, and they are not
        # candidates competing for selection. Exclude them from both the average
        # model size q and the candidate pool p so the bound reflects the genes
        # that were actually selected competitively.
        n_forced = _mandatory_gene_counts(mandatory_features, latent_dim, p_genes)
        q_eff = np.maximum(avg_selected - n_forced, 0.0)
        p_eff = p_genes - n_forced
        expected_false_positives = np.where(
            p_eff > 0, q_eff**2 / ((2.0 * threshold - 1.0) * p_eff), np.nan
        )
    else:
        warnings.warn(
            "threshold <= 0.5: the Meinshausen-Bühlmann error bound is undefined, "
            "so expected_false_positives is NaN. Use a threshold in (0.5, 1].",
            UserWarning,
            stacklevel=2,
        )
        expected_false_positives = np.full(latent_dim, np.nan)

    cond_mean, cond_sd, sign_consistency = _coefficient_statistics(
        coef_sum, coef_sq_sum, positive_count, counts
    )

    return StabilitySelectionResult(
        frequency=frequency,
        stable_support=stable_support,
        threshold=float(threshold),
        avg_selected=avg_selected,
        expected_false_positives=expected_false_positives,
        n_subsamples=int(n_subsamples),
        subsample_frac=float(subsample_frac),
        coefficient_cond_mean=cond_mean,
        coefficient_sd=cond_sd,
        sign_consistency=sign_consistency,
    )
