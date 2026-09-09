"""Plotting helpers for structboost.

These utilities are intentionally lightweight and optional. Everything here needs
only matplotlib and numpy, imported inside the functions so the module itself
stays importable without them -- with one exception:
:func:`plot_dimension_gene_umaps` draws through ``scanpy.pl.umap`` so its panels
match the rest of a scanpy figure, and says so if scanpy is absent.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np


def plot_top_boosting_coefficients(
    betamat: np.ndarray,
    gene_names: Sequence[str],
    cluster_labels: Sequence[str],
    *,
    top_m: int = 10,
    neg_color: str = "#1f77b4",
    pos_color: str = "#d62728",
):
    """Plot per-target dot/lollipop charts of top boosting coefficients.

    Parameters
    ----------
    betamat
        Array of shape (n_targets, n_features) with boosting coefficients.
    gene_names
        Feature names of length n_features (e.g., gene symbols).
    cluster_labels
        Target labels of length n_targets (e.g., cluster IDs).
    top_m
        Number of features to display per target (ranked by ``|beta|``).
    neg_color, pos_color
        Colors used for negative and positive coefficients, respectively.

    Returns
    -------
    figs
        List of matplotlib Figure objects, one per target.
    axes
        List of matplotlib Axes objects, one per target.
    """
    import matplotlib.pyplot as plt

    betamat = np.asarray(betamat, dtype=float)
    n_targets, n_features = betamat.shape

    if len(gene_names) != n_features:
        raise ValueError(f"gene_names must have length {n_features}, got {len(gene_names)}")
    if len(cluster_labels) != n_targets:
        raise ValueError(f"cluster_labels must have length {n_targets}, got {len(cluster_labels)}")
    if top_m < 1:
        raise ValueError("top_m must be >= 1")

    top_m_eff = min(top_m, n_features)
    figs = []
    axes = []

    for t, cluster in enumerate(cluster_labels):
        beta_all = betamat[t, :]

        nonzero = np.flatnonzero(beta_all)
        if nonzero.size > 0:
            idx_rank = nonzero[np.argsort(np.abs(beta_all[nonzero]))[::-1]]
        else:
            idx_rank = np.argsort(np.abs(beta_all))[::-1]

        feat_idx = idx_rank[:top_m_eff]
        beta_hat = beta_all[feat_idx]
        genes = [gene_names[j] for j in feat_idx]

        order = np.argsort(np.abs(beta_hat))[::-1]
        beta_hat = beta_hat[order]
        genes = [genes[o] for o in order]
        y_pos = np.arange(len(genes))

        colors = np.where(beta_hat >= 0, pos_color, neg_color)

        fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(genes) + 1.6))
        ax.hlines(y_pos, 0, beta_hat, color=colors, linewidth=4, alpha=0.25, zorder=1)
        ax.scatter(beta_hat, y_pos, s=90, c=colors, edgecolors="white", linewidths=0.9, zorder=2)
        ax.axvline(0, color="#9aa0a6", linestyle=(0, (3, 3)), linewidth=1.5, zorder=0)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(genes, fontsize=10)
        ax.set_xlabel("Boosting coefficient", fontsize=11)
        ax.set_title(
            f"Top {len(genes)} boosting coefficients (Cluster {cluster})", fontsize=12, pad=10
        )

        max_abs = float(np.max(np.abs(beta_hat))) if beta_hat.size else 1.0
        pad = 0.15 * max_abs + 1e-9
        ax.set_xlim(-(max_abs + pad), (max_abs + pad))

        ax.grid(axis="y", visible=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        for x, y, c in zip(beta_hat, y_pos, colors, strict=True):
            ha = "left" if x >= 0 else "right"
            ax.text(
                x + (0.02 * (max_abs + pad) if x >= 0 else -0.02 * (max_abs + pad)),
                y,
                f"{x:.3f}",
                va="center",
                ha=ha,
                fontsize=9,
                color=c,
                alpha=0.85,
            )

        ax.invert_yaxis()
        fig.tight_layout()

        figs.append(fig)
        axes.append(ax)

    return figs, axes


def plot_boosting_coefficient_paths(
    beta_path: np.ndarray,
    gene_names: Sequence[str],
    cluster_labels: Sequence[str],
    *,
    top_m: int = 10,
    linewidth: float = 1.6,
    alpha: float = 0.9,
):
    """Plot coefficient trajectories across boosting steps (per target).

    This visualizes how coefficients evolve over iterations. For each target, we
    select the top-``|beta|`` features at the **final** step and plot their paths.

    Parameters
    ----------
    beta_path
        Array of shape (n_targets, n_steps, n_features), typically
        `history.beta_path` from `allboost(..., return_history=True)`.
    gene_names
        Feature names of length n_features.
    cluster_labels
        Target labels of length n_targets.
    top_m
        Number of features to display per target (ranked by ``|beta|`` at final step).
    linewidth
        Line width for each feature trajectory.
    alpha
        Line alpha for each feature trajectory.
    Returns
    -------
    figs, axes
        Lists of matplotlib Figure/Axes objects, one per target.
    """
    import matplotlib.pyplot as plt

    beta_path = np.asarray(beta_path, dtype=float)
    n_targets, n_steps, n_features = beta_path.shape

    if len(gene_names) != n_features:
        raise ValueError(f"gene_names must have length {n_features}, got {len(gene_names)}")
    if len(cluster_labels) != n_targets:
        raise ValueError(f"cluster_labels must have length {n_targets}, got {len(cluster_labels)}")
    if top_m < 1:
        raise ValueError("top_m must be >= 1")

    steps = np.arange(1, n_steps + 1)
    top_m_eff = min(top_m, n_features)

    figs = []
    axes = []

    for t, cluster in enumerate(cluster_labels):
        beta_final = beta_path[t, -1, :]
        top_idx = np.argsort(np.abs(beta_final))[::-1][:top_m_eff]

        fig, ax = plt.subplots(figsize=(10, 4))
        for j in top_idx:
            ax.plot(
                steps, beta_path[t, :, j], label=gene_names[j], linewidth=linewidth, alpha=alpha
            )

        ax.axhline(0, color="black", linewidth=0.9, alpha=0.4)
        ax.set_title(f"Coefficient paths (Cluster {cluster})")
        ax.set_xlabel("Boosting step")
        ax.set_ylabel("Coefficient")
        ax.legend(ncols=2, fontsize=8, frameon=False)
        fig.tight_layout()

        figs.append(fig)
        axes.append(ax)

    return figs, axes


def plot_training_diagnostics(
    report,
    *,
    figsize: tuple[float, float] = (13.0, 9.0),
    log_scale: bool = True,
):
    """Plot a :class:`~structboost.TrainingReport` as a six-panel diagnostic figure.

    Parameters
    ----------
    report
        A :class:`~structboost.TrainingReport`, a fitted ``BAE`` (its
        ``training_report`` is used), or an ``AnnData`` carrying
        ``uns["bae"]["training_report"]``.
    figsize
        Figure size in inches.
    log_scale
        Use a log y-axis for the convergence and gradient panels, whose values
        span orders of magnitude.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Panels are: reconstruction loss, per-iteration contribution
        of the encoder and decoder halves, encoder convergence, selected genes,
        gradient norms, and boosting fit quality.

    Raises
    ------
    ValueError
        If no training report is available (``diagnostics=True`` was not set).
    """
    import matplotlib.pyplot as plt

    from ._types import TrainingReport

    if not isinstance(report, TrainingReport):
        candidate = getattr(report, "training_report", None)
        if candidate is None:
            uns = getattr(report, "uns", None)
            stored = uns.get("bae", {}).get("training_report") if uns is not None else None
            if stored is None:
                raise ValueError(
                    "no training report available; fit with diagnostics=True to collect one"
                )
            candidate = TrainingReport.from_dict(stored)
        report = candidate

    it = report.iteration
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("BAE training diagnostics", fontsize=13)

    ax = axes[0, 0]
    ax.plot(it, report.loss_post_decoder, lw=1.8, color="#1f77b4", label="full-data")
    ax.plot(it, report.train_loss, lw=1.0, color="#aaaaaa", alpha=0.9, label="minibatch running")
    ax.set_title("Reconstruction loss")
    ax.set_xlabel("iteration")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=8)

    # The two halves of the alternating optimizer, as loss change per iteration.
    ax = axes[0, 1]
    ax.axhline(0.0, color="#666666", lw=0.8)
    ax.plot(it, report.encoder_delta, lw=1.4, color="#d62728", label="encoder (boosting)")
    ax.plot(it, report.decoder_delta, lw=1.4, color="#2ca02c", label="decoder (SGD)")
    ax.set_title("Loss change per update\n(negative = improved)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("Δ MSE")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.plot(it, report.weight_change_rel, lw=1.6, color="#9467bd", label="rel. weight change")
    ax.plot(it, 1.0 - report.support_jaccard, lw=1.2, color="#ff7f0e", label="1 − support Jaccard")
    if log_scale:
        ax.set_yscale("log")
    ax.set_title("Encoder convergence")
    ax.set_xlabel("iteration")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(it, report.n_selected, lw=1.6, color="#8c564b")
    ax.set_title("Selected genes")
    ax.set_xlabel("iteration")
    ax.set_ylabel("n genes with nonzero weight")

    ax = axes[1, 1]
    ax.plot(it, report.target_grad_norm, lw=1.6, color="#e377c2", label="‖∂L/∂z‖ (targets)")
    ax.plot(it, report.decoder_grad_norm, lw=1.2, color="#7f7f7f", label="‖∇ decoder‖")
    if log_scale:
        ax.set_yscale("log")
    ax.set_title("Gradient norms")
    ax.set_xlabel("iteration")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.plot(it, report.boosting_r2, lw=1.6, color="#17becf")
    ax.set_title("Boosting fit to targets")
    ax.set_xlabel("iteration")
    ax.set_ylabel("R²")

    for ax in axes.ravel():
        ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    return fig


# --- Reading a fitted BAE, dimension by dimension ---------------------------
#
# Palettes are chosen by how many groups are shown, from established schemes
# rather than bespoke ones, so a figure from this package sits beside a scanpy
# figure without a jarring recolouring.
#
#   <= 5   the first five of `tab10`, reordered to blue / orange / red / purple /
#          green so the commonest small comparisons get the most separable hues.
#   <= 10  the rest of `tab10`, keeping those five as its prefix so a plot does
#          not recolour itself when a sixth group appears.
#   <= 20  scanpy's `default_20`, the de-facto standard in single cell.
#
# What each costs is measurable with :func:`palette_audit`, and only the first
# tier clears every bar: `tab10` pairs red with green, which is the colour-blind
# confusion axis, so its separation under simulated deuteranopia is 1.3 dE
# against a target of 8; at twenty hues nothing comes close, because twenty
# reliably distinguishable colours do not exist. Past five, colour is a
# *supporting* encoding here -- affordable because the group panel labels every
# row, and costly in the score panel, which has no labels.
_PALETTE_5: tuple[str, ...] = ("#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#2ca02c")
_PALETTE_10: tuple[str, ...] = _PALETTE_5 + (
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)
_PALETTE_20: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#279e68",
    "#d62728",
    "#aa40fc",
    "#8c564b",
    "#e377c2",
    "#b5bd61",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#dbdb8d",
    "#9edae5",
    "#ad494a",
    "#8c6d31",
)
#: Seven hues chosen for measured separation rather than convention: WCAG 3.0
#: contrast on white and 10.0 dE under the worst of normal vision and simulated
#: deuteranopia / protanopia / tritanopia. An eighth drops that to 7.3.
_PALETTE_SAFE: tuple[str, ...] = (
    "#0077BB",
    "#A67C00",
    "#882255",
    "#6699CC",
    "#332288",
    "#EE6677",
    "#CF1C90",
)
_PALETTES: dict[str, tuple[str, ...]] = {
    "tab5": _PALETTE_5,
    "tab10": _PALETTE_10,
    "scanpy20": _PALETTE_20,
    "safe": _PALETTE_SAFE,
}
#: Groups past the palette keep a violin -- that panel labels every row, so
#: identity survives without a hue -- but are grey wherever colour is the only
#: channel.
_OTHER = "#BBBBBB"
#: Negative and positive. One blue means one thing across every panel: cells
#: below zero, and genes whose encoder weight is negative.
_NEG, _POS = "#0072B2", "#D55E00"
_INK, _MUTED, _FAINT = "#1a1a1a", "#333333", "#b0b0b0"
#: One type scale for every panel; axis text drifting between sizes reads as
#: several figures pasted together.
_TICK, _LABEL, _TITLE, _LEGEND = 9.0, 9.5, 10.5, 11.0
#: Below this many cells a group gets no violin: a kernel density over a handful
#: of points draws a shape the data does not support.
_MIN_GROUP = 10
#: Relative panel widths, so a three-panel row still reads.
_WIDTH = {"scores": 1.35, "contributions": 1.0, "weights": 0.8, "shares": 1.0, "groups": 1.0}


def _palette_for(n_groups: int) -> tuple[str, ...]:
    """Smallest established scheme that covers ``n_groups``."""
    if n_groups <= len(_PALETTE_5):
        return _PALETTE_5
    if n_groups <= len(_PALETTE_10):
        return _PALETTE_10
    return _PALETTE_20


def gene_variance_shares(X: np.ndarray, w: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Each gene's share of one latent dimension's variance. Sums to 1 exactly.

    From ``Var(s) = Cov(s, s)`` with ``s = X @ w``::

        Var(s) = Cov(sum_g w_g X_g, s) = sum_g w_g Cov(X_g, s)

    so ``w_g Cov(X_g, s) / Var(s)`` is an exact additive decomposition, needing no
    orthogonality assumption and handling correlated genes correctly -- two
    redundant genes split a share rather than both claiming it.

    This is the honest ranking of genes within a dimension. ``|w_g|`` is not,
    because it ignores how much the gene actually varies: measured on one dataset
    the two rankings agree on 8.6 of 10 genes per dimension, and where they differ
    ``|w|`` promotes genes that barely move -- SCT ranked #4 by weight and #35 of
    38 by share, being near-absent in the tissue.

    The result is a *magnitude*: because a negative-weight gene is anti-correlated
    with the score, the product is positive either way, and only about 1% of genes
    come out negative (a suppressor, whose weight opposes its own correlation with
    the finished score). Direction lives in the sign of ``w``, not here.

    Parameters
    ----------
    X
        ``(n_cells, n_genes)`` expression the encoder was fitted on.
    w
        ``(n_genes,)`` encoder weights for one latent dimension.
    scores
        ``(n_cells,)`` that dimension's latent scores.

    Returns
    -------
    ndarray
        ``(n_genes,)`` shares, summing to 1 over a dimension's selected genes.
    """
    var = float(np.asarray(scores).var())
    if var <= 0:
        return np.zeros_like(np.asarray(w), dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    centered = np.asarray(scores, dtype=np.float64) - float(np.mean(scores))
    cov = (X - X.mean(axis=0)).T @ centered / X.shape[0]
    return np.asarray(np.asarray(w, dtype=np.float64) * cov / var, dtype=np.float64)


def palette_audit(palette) -> dict:
    """Measured separation and contrast for a palette, so a choice can be checked.

    Reports the minimum pairwise OKLab distance under normal vision and under
    simulated deuteranopia / protanopia / tritanopia (Machado et al. 2009), plus
    the worst WCAG contrast against white. Targets are 15 dE, 8 dE and 3.0; a
    palette failing them is not unusable, but colour is then a supporting encoding
    rather than the identity channel.

    Parameters
    ----------
    palette
        A key of the built-in palettes (``"tab5"``, ``"tab10"``, ``"scanpy20"``,
        ``"safe"``) or a sequence of hex colours.

    Returns
    -------
    dict
        ``n``, ``min_de_normal``, ``min_de_cvd``, ``min_contrast`` and
        ``n_below_contrast_floor``.
    """
    import itertools

    pal = _PALETTES.get(palette, palette) if isinstance(palette, str) else palette

    def to_linear(h):
        c = np.array([int(str(h).lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)])
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    m1 = np.array(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ]
    )
    m2 = np.array(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ]
    )
    cvd = {
        "deuteranopia": [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ],
        "protanopia": [
            [0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998],
        ],
        "tritanopia": [
            [1.255528, -0.076749, -0.178779],
            [-0.078411, 0.930809, 0.147602],
            [0.004733, 0.691367, 0.303900],
        ],
    }

    def oklab(v):
        return m2 @ np.cbrt(np.clip(m1 @ v, 0, None))

    def delta(a, b):
        return float(np.linalg.norm((oklab(a) - oklab(b)) * 100))

    lins = [to_linear(h) for h in pal]
    contrast = [1.05 / ((0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]) + 0.05) for v in lins]
    pairs = list(itertools.combinations(lins, 2))
    return {
        "n": len(list(pal)),
        "min_de_normal": min((delta(a, b) for a, b in pairs), default=float("inf")),
        "min_de_cvd": min(
            (delta(np.array(m) @ a, np.array(m) @ b) for m in cvd.values() for a, b in pairs),
            default=float("inf"),
        ),
        "min_contrast": min(contrast),
        "n_below_contrast_floor": sum(c < 3.0 for c in contrast),
    }


def _elide(text, limit: int) -> str:
    """Truncate with an ellipsis, so a shortened category reads as shortened rather
    than as a different category with a similar name."""
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    """`n` with a correctly inflected noun.

    Both counts here can legitimately be 1 -- a dimension can select a single gene,
    and the grey fold can be one group deep -- so "1 genes selected" is reachable
    rather than hypothetical.
    """
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def _point_size(n_cells: int) -> float:
    """Marker size for the score panel, larger on smaller datasets.

    Colour identity is the whole point of that panel, and a rare population is
    invisible below roughly 4 pt however distinct its hue -- the groups a sparse
    encoder detects can number a few dozen cells out of thousands.
    """
    return float(np.clip(250_000 / max(n_cells, 1), 8.0, 20.0))


def _zero(ax, axis: str) -> None:
    """Mark zero prominently.

    Not a gridline: the encoder is bias-free, so a score of zero means no selected
    gene departs from its mean in that cell, and it is where the dimension's sign --
    and therefore which tail a cell belongs to -- flips.
    """
    (ax.axhline if axis == "y" else ax.axvline)(
        0.0, color="black", lw=1.2, ls="--", zorder=3, alpha=0.85
    )


def _score_cmap():
    """Diverging blue-neutral-orange, for signed quantities.

    A sequential map has no midpoint, so its neutral colour lands at whatever the
    data mean happens to be -- while zero is a real level here, not a middling one.
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("bae_score", [_NEG, "#F2F2F2", _POS])


def _resolve_groups(values, Z, max_groups, palette):
    """Assign hues to the groups most relevant *to the dimensions*, not the largest.

    Abundance is the wrong rule: a sparse encoder preferentially encodes *rare*
    populations, so an abundance cut hides exactly what the figure exists to show --
    on one dataset it folded three 22-to-29-cell populations into grey and left 4 of
    10 dimensions with an empty group panel. A group earns a hue by how far its
    median sits from the overall median on the dimension where it is most extreme,
    in units of that dimension's spread.
    """
    values = np.asarray(values, dtype=object)
    levels, counts = np.unique(values, return_counts=True)
    count_of = dict(zip(levels, counts, strict=True))
    if len(levels) <= max_groups:
        keep = list(levels[np.argsort(counts)[::-1]])
    else:
        sd = Z.std(axis=0)
        sd[sd == 0] = 1.0
        overall = np.median(Z, axis=0)

        def extremity(level):
            if count_of[level] < _MIN_GROUP:
                return -np.inf
            return float(np.max(np.abs(np.median(Z[values == level], axis=0) - overall) / sd))

        keep = sorted(levels, key=lambda lv: (-extremity(lv), -count_of[lv]))[:max_groups]
    # Never cycled -- a repeated hue asserts an identity that is not there.
    lut = {lv: (palette[i] if i < len(palette) else _OTHER) for i, lv in enumerate(keep)}
    colors = np.array([lut.get(x, _OTHER) for x in values], dtype=object)
    return colors, keep, lut, len(levels) - len(keep)


def _split_violin(ax, data, positions, color, half):
    """One side of a split violin: draw, then clip each body to half its width.

    matplotlib has no split violin, and two offset violins would double the vertical
    space per gene and break the alignment with the score axis. Clipping the path is
    exact and keeps both halves on one row.
    """
    if not any(np.size(d) for d in data):
        return None
    # `widths` is the FULL violin width, so a clipped half reaches widths/2 from its
    # position while adjacent positions sit 1.0 apart. Anything above 1.0 makes
    # neighbouring halves overlap; 0.92 leaves a visible gap between genes.
    kw = dict(positions=positions, widths=0.92, showextrema=False, showmedians=False)
    try:
        parts = ax.violinplot(data, orientation="horizontal", **kw)
    except TypeError:  # matplotlib < 3.10
        parts = ax.violinplot(data, vert=False, **kw)
    for body, pos in zip(parts["bodies"], positions, strict=True):
        verts = body.get_paths()[0].vertices
        verts[:, 1] = (
            np.clip(verts[:, 1], pos, np.inf)
            if half == "upper"
            else np.clip(verts[:, 1], -np.inf, pos)
        )
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_linewidth(0.9)
        body.set_alpha(0.9)
    return parts


def _draw_scores(ax, z, colors, size, dim, rng) -> str:
    order = np.argsort(z)
    # Drawn in random order. Each cell keeps its own (rank, score) position -- the
    # curve is identical -- but markers are wider than the spacing between adjacent
    # ranks, so whatever is drawn last wins every overlap. Painting in score order
    # hands each local contest to the higher-scoring cell, and since score correlates
    # with group, one group systematically covers its neighbours. Seeded, so the
    # figure stays reproducible.
    shuffle = rng.permutation(z.size)
    ax.scatter(
        np.arange(z.size)[shuffle],
        z[order][shuffle],
        c=colors[order][shuffle],
        s=size,
        linewidths=0,
        alpha=0.8,
        rasterized=True,
    )
    _zero(ax, "y")
    # Per tail, not pooled. Most dimensions are one-sided -- a population at one end
    # and nothing coherent at the other -- and a single |centred| figure averages
    # that asymmetry away, which is the thing worth seeing.
    centered = np.abs(z - z.mean())
    total = max(centered.sum(), 1e-12)
    d = max(1, z.size // 10)
    ax.set_title(f"dim {dim}", fontsize=_TITLE, loc="left", color=_INK, fontweight="bold")
    ax.set_xlabel("cells, ranked by score", fontsize=_LABEL, color=_MUTED)
    ax.set_ylabel("latent score", fontsize=_LABEL, color=_MUTED)
    ax.text(
        0.02,
        0.96,
        f"decile mass — top {centered[order[-d:]].sum() / total:.0%} · "
        f"bottom {centered[order[:d]].sum() / total:.0%}",
        transform=ax.transAxes,
        fontsize=_LABEL,
        color=_MUTED,
        va="top",
    )
    return "y"


def _draw_contributions(
    ax, contrib, names, shares, weights, rank, n_genes, normalize, active, scores
) -> str:
    """Per-gene contribution distributions, split by the sign of the cell's score.

    Each row is one gene. The upper half is the distribution over cells at the
    dimension's **positive** end, the lower half over cells at its **negative** end.
    A gene can carry a large share while acting on only one tail, and the two halves
    say which; colour therefore encodes *which end the cells came from*, while the
    gene name's colour encodes the sign of its encoder weight.

    ``normalize="cell"`` divides by the cell's total *absolute* contribution rather
    than by its score. Dividing by the score -- the obvious reading of "fraction of
    this cell's score" -- is unusable: the flat plateau of a subgroup dimension is
    exactly the cells whose score is ~0, so the ratio diverges (measured -8,921 to
    +1,191 for one gene), flips sign with the score, and inflates without bound
    wherever positive and negative contributions cancel.
    """
    if normalize == "cell":
        denom = np.abs(contrib).sum(axis=1, keepdims=True)
        denom[denom == 0] = np.inf
        contrib = contrib / denom
    if active is not None:
        contrib, scores = contrib[active], scores[active]

    # Ranked by share, largest at the top, and not re-sorted by median contribution:
    # because X is centred every gene's mean contribution is exactly zero, so a
    # median only measures skew -- a marker expressed in few cells sits slightly
    # below zero in most of them precisely when its share is largest.
    order = np.argsort(np.abs(rank))[::-1][:n_genes][::-1]
    pos_cells, neg_cells = scores > 0, scores < 0
    positions = np.arange(len(order))
    _split_violin(ax, [contrib[pos_cells, i] for i in order], positions, _POS, "upper")
    _split_violin(ax, [contrib[neg_cells, i] for i in order], positions, _NEG, "lower")
    for row, i in enumerate(order):
        for mask, off in ((pos_cells, 0.42), (neg_cells, -0.42)):
            if mask.sum():
                med = float(np.median(contrib[mask, i]))
                ax.plot([med, med], [row, row + off], color="#222222", lw=1.0, zorder=4)
    _zero(ax, "x")
    ax.set_yticks(positions)
    # The share is a magnitude, so it carries no sign; direction is the weight's,
    # shown by the label colour.
    ax.set_yticklabels([f"{names[i]}  {abs(shares[i]):.0%}" for i in order], fontsize=_TICK)
    for label, i in zip(ax.get_yticklabels(), order, strict=True):
        label.set_color(_POS if weights[i] >= 0 else _NEG)
    # Robust limits: a handful of cells can sit ten times further out than the bulk,
    # and on raw limits every violin collapses onto the zero line. The clipped
    # fraction is stated rather than hidden.
    data = [contrib[:, i] for i in order]
    lo = min(float(np.percentile(d_, 0.5)) for d_ in data)
    hi = max(float(np.percentile(d_, 99.5)) for d_ in data)
    pad = 0.08 * max(hi - lo, 1e-9)
    beyond = sum(int(((d_ < lo) | (d_ > hi)).sum()) for d_ in data)
    total = sum(np.size(d_) for d_ in data)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(-0.8, len(order) - 0.2)
    what = (
        "fraction of the cell's gene activity" if normalize == "cell" else "contribution to score"
    )
    where = " · active cells" if active is not None else ""
    ax.set_title(
        f"gene contributions  (share of variance){where}", fontsize=_TITLE, loc="left", color=_INK
    )
    ax.set_xlabel(
        f"{what}  ({beyond / max(total, 1):.1%} of points beyond axis)",
        fontsize=_LABEL,
        color=_MUTED,
    )
    return "x"


def _draw_weights(ax, w, names, rank, n_genes) -> str:
    """Raw encoder coefficients, for when the coefficient itself is the question.

    Available but not default: ``w`` is what the model stores, while
    :func:`gene_variance_shares` is what the gene does to the code.
    """
    order = np.argsort(np.abs(rank))[::-1][:n_genes][::-1]
    ax.barh(
        np.arange(order.size),
        w[order],
        height=0.72,
        color=[_POS if v >= 0 else _NEG for v in w[order]],
    )
    _zero(ax, "x")
    ax.set_yticks(np.arange(order.size))
    ax.set_yticklabels([_elide(names[i], 20) for i in order], fontsize=_TICK)
    # Gene name in the colour of its weight's sign, as in every other gene panel.
    for label, i in zip(ax.get_yticklabels(), order, strict=True):
        label.set_color(_POS if w[i] >= 0 else _NEG)
    ax.set_title("encoder weights", fontsize=_TITLE, loc="left", color=_INK)
    ax.set_xlabel("weight", fontsize=_LABEL, color=_MUTED)
    return "x"


def _draw_shares(ax, shares, weights, sort_by, n_genes) -> str:
    """Every selected gene's share of the dimension, as a sorted profile.

    The contribution panel shows the top handful; this shows *all* of them, which
    is the only view that says whether a dimension rests on three genes or spreads
    evenly over forty. Both are legitimate and read differently -- a concentrated
    dimension is a marker programme, a flat one a diffuse signature -- and the
    shares sum to 1 either way, so the shape is the message.

    ``sort_by="weight"`` keeps the y axis on the share but orders by coefficient,
    which draws the disagreement between the two rankings directly: a monotone
    curve means they agree, and a ragged one means large coefficients sitting on
    genes that move no cells.
    """
    order = np.argsort(np.abs(weights) if sort_by == "weight" else shares)
    y = shares[order]
    x = np.arange(y.size)
    ax.axhline(0.0, color=_FAINT, lw=0.8, zorder=0)
    ax.scatter(
        x, y, s=26, linewidths=0, zorder=3, c=[_POS if weights[i] >= 0 else _NEG for i in order]
    )
    ax.set_xlim(-0.6, max(y.size - 0.4, 1.0))
    key = "|weight|" if sort_by == "weight" else "share"
    ax.set_xlabel(f"selected genes, ordered by {key}", fontsize=_LABEL, color=_MUTED)
    ax.set_ylabel("share of variance", fontsize=_LABEL, color=_MUTED)
    top = float(np.sort(shares)[::-1][:n_genes].sum())
    ax.set_title(
        f"{_plural(y.size, 'gene')} · top {n_genes} hold {top:.0%}",
        fontsize=_TITLE,
        loc="left",
        color=_INK,
    )
    return "y"


def _draw_groups(ax, z, labels, keep, lut, group_by) -> str:
    groups = [g for g in keep if (labels == g).sum() >= _MIN_GROUP]
    groups.sort(key=lambda g: float(np.median(z[labels == g])))
    if groups:
        kw = dict(
            positions=np.arange(len(groups)), widths=0.85, showextrema=False, showmedians=True
        )
        data = [z[labels == g] for g in groups]
        try:
            parts = ax.violinplot(data, orientation="horizontal", **kw)
        except TypeError:  # matplotlib < 3.10
            parts = ax.violinplot(data, vert=False, **kw)
        for body, g in zip(parts["bodies"], groups, strict=True):
            body.set_facecolor(lut[g])
            body.set_edgecolor("none")
            body.set_alpha(0.85)
        parts["cmedians"].set_color("#333333")
        parts["cmedians"].set_linewidth(1.0)
    _zero(ax, "x")
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels([_elide(g, 22) for g in groups], fontsize=_TICK)
    ax.set_title(f"by {group_by}, ordered by median", fontsize=_TITLE, loc="left", color=_INK)
    ax.set_xlabel("latent score", fontsize=_LABEL, color=_MUTED)
    return "x"


def plot_latent_dimensions(
    adata,
    *,
    dims: Sequence[int] | None = None,
    group_by: str | None = None,
    panels: Sequence[str] = ("scores", "contributions", "groups"),
    latent_key: str = "X_bae",
    weights_key: str = "BAE_encoder_weights",
    layer: str | None = None,
    gene_names: str | None = None,
    n_genes: int = 10,
    rank_by: Literal["share", "weight"] = "share",
    normalize: Literal["none", "cell"] = "none",
    active_quantile: float | None = None,
    max_groups: int = 12,
    palette: str | Sequence[str] = "auto",
    point_size: float | None = None,
    seed: int = 0,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
):
    """Plot per-dimension diagnostics for a fitted BAE: one row per dimension.

    Three panels, each answering a different question, any subset selectable:

    ``"scores"``
        Every cell's score, sorted -- the quantile function of the dimension. A
        smooth ramp is a graded, population-wide axis; a flat plateau with a hook
        is a subgroup axis. That shape decides which summary of the dimension is
        meaningful at all, and it is not a property of the model: it follows from
        whether the population the dimension encodes is rare or abundant.

    ``"contributions"``
        What each gene actually contributes to the score, ``X[:, g] * W[g, k]``, as
        a distribution over cells, split by the sign of the cell's score. **Not the
        encoder weight**: a gene's influence depends on how much it varies as well
        as on its weight, and the two diverge whenever the panel is not perfectly
        standardized. Ranked by :func:`gene_variance_shares`.

    ``"weights"``
        The raw coefficients. Available, not default.

    ``"shares"``
        Every selected gene's share of the dimension's variance, sorted -- the
        whole profile rather than the top handful, which is what says whether a
        dimension rests on three genes or spreads evenly over forty. Obeys
        ``rank_by``: ordering by ``"weight"`` while plotting the share exposes
        genes carrying a large coefficient that move no cells.

    ``"groups"``
        The same scores split by ``group_by``, violins ordered by median. Dropped
        when ``group_by`` is ``None``, because an ungrouped violin is the score
        panel rotated a quarter turn.

    Parameters
    ----------
    adata
        AnnData carrying a fitted BAE: ``obsm[latent_key]`` and
        ``varm[weights_key]``, both written by :meth:`structboost.BAE.fit`.
    dims
        Dimensions to draw, in order. ``None`` draws all.
    group_by
        ``obs`` column defining the groups for the ``"groups"`` panel.
    panels
        Which panels to draw per dimension, left to right.
    latent_key, weights_key, layer
        Where to read the code, the encoder and the expression. ``layer=None``
        means ``adata.X``; pass the layer the model was fitted on, or the
        contributions will not sum to the scores.
    gene_names
        ``var`` column holding display names. ``None`` uses ``var_names``.
    n_genes
        Genes per dimension in the gene panels.
    rank_by
        Which quantity picks them. ``"share"`` is the default and the honest one;
        ``"weight"`` reproduces a coefficient ranking.
    normalize
        ``"none"`` plots contributions in score units, so the violins are the terms
        that sum to the curve in the ``"scores"`` panel. ``"cell"`` divides each by
        that cell's total absolute contribution, a bounded per-cell share, at the
        cost of the additive link to the score axis.
    active_quantile
        Restrict the contribution panel to cells whose ``|score|`` is at or above
        this quantile. On a subgroup dimension most cells sit at zero and add only
        noise; on a ramp dimension there is no active set and this is arbitrary.
    max_groups
        How many groups appear at all. Those past the palette are grey but keep
        their labelled violin, so only the hue is lost.
    palette
        ``"auto"`` picks the smallest established scheme covering the groups shown;
        ``"tab5"``, ``"tab10"``, ``"scanpy20"``, ``"safe"``, or a sequence of
        colours. See :func:`palette_audit` for what each costs.
    point_size
        Marker size in the score panel. ``None`` scales it to the cell count.
    seed
        Seeds the draw-order permutation of the score panel.
    figsize, dpi
        Overrides for the computed size, and the figure's resolution.

    Returns
    -------
    fig, axes
        The figure and its ``(n_dims, n_panels)`` array of axes.

    Raises
    ------
    KeyError
        If a required ``obsm``/``varm``/``obs`` key is missing.
    ValueError
        If ``dims`` is out of range, a panel name is unknown, the palette is
        unknown, or no panel is left to draw.

    Examples
    --------
    >>> plot_latent_dimensions(adata, group_by="cell_type")           # doctest: +SKIP
    >>> plot_latent_dimensions(adata, dims=[0, 3], panels=("scores",))  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    panels = [p for p in panels if p != "groups" or group_by is not None]
    if not panels:
        raise ValueError(
            "no panels to draw: 'groups' was the only one requested and group_by is None"
        )
    unknown = set(panels) - set(_WIDTH)
    if unknown:
        raise ValueError(f"unknown panels {sorted(unknown)}; choose from {sorted(_WIDTH)}")
    if latent_key not in adata.obsm:
        raise KeyError(f"adata.obsm[{latent_key!r}] not found; fit the model first")

    Z = np.asarray(adata.obsm[latent_key], dtype=np.float64)
    dims = list(range(Z.shape[1])) if dims is None else list(dims)
    if any(d < 0 or d >= Z.shape[1] for d in dims):
        raise ValueError(f"dims out of range for a {Z.shape[1]}-dimensional code: {dims}")

    X = W = names = None
    if {"contributions", "weights", "shares"} & set(panels):
        if weights_key not in adata.varm:
            raise KeyError(f"adata.varm[{weights_key!r}] not found; needed for gene panels")
        W = np.asarray(adata.varm[weights_key], dtype=np.float64)
        # Left sparse: only a dimension's own selected genes are densified below,
        # which is tens of columns rather than the whole panel.
        X = adata.X if layer is None else adata.layers[layer]
        names = (
            np.asarray(adata.var[gene_names], dtype=str)
            if gene_names
            else np.asarray(adata.var_names, dtype=str)
        )

    labels = colors = keep = lut = None
    n_other = 0
    if group_by is not None:
        if group_by not in adata.obs:
            raise KeyError(f"adata.obs[{group_by!r}] not found")
        labels = np.asarray(adata.obs[group_by].astype(str), dtype=object)
        n_levels = len(np.unique(labels))
        if palette == "auto":
            pal = _palette_for(min(n_levels, max_groups))
        elif isinstance(palette, str):
            if palette not in _PALETTES:
                raise ValueError(
                    f"unknown palette {palette!r}; use 'auto', "
                    f"{sorted(_PALETTES)}, or a list of colours"
                )
            pal = _PALETTES[palette]
        else:
            pal = tuple(palette)
        colors, keep, lut, n_other = _resolve_groups(labels, Z[:, dims], max_groups, tuple(pal))
    if colors is None:
        colors = np.full(Z.shape[0], "#4C72B0", dtype=object)
    size = _point_size(Z.shape[0]) if point_size is None else point_size
    rng = np.random.default_rng(seed)

    # Row height follows the number of label rows a panel carries, not a constant:
    # the gene and group panels each need one tick per entry, and at a fixed height
    # their labels collide as soon as either exceeds about eight.
    rows_needed = max(
        n_genes if {"contributions", "weights", "shares"} & set(panels) else 0,
        max_groups if "groups" in panels else 0,
        6,
    )
    fig_size = figsize or (
        4.4 * sum(_WIDTH[p] for p in panels),
        (0.38 * rows_needed + 1.7) * len(dims),
    )
    # Margins in inches, not figure fractions: a fractional top margin grows with
    # the figure, opening dead space under the legend on a tall one.
    fig, axes = plt.subplots(
        len(dims),
        len(panels),
        squeeze=False,
        figsize=fig_size,
        dpi=dpi,
        gridspec_kw={
            "width_ratios": [_WIDTH[p] for p in panels],
            "hspace": 0.38,
            "wspace": 0.68,
            "top": 1 - 0.45 / fig_size[1],
            "bottom": 0.45 / fig_size[1],
        },
    )

    for row, dim in enumerate(dims):
        z = Z[:, dim]
        for col, panel in enumerate(panels):
            ax = axes[row][col]
            if panel == "scores":
                grid = _draw_scores(ax, z, colors, size, dim, rng)
            elif panel in ("contributions", "weights", "shares"):
                w = W[:, dim]
                nz = np.flatnonzero(w)
                if nz.size == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "no genes selected",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=_LABEL,
                        color=_MUTED,
                    )
                    ax.set_axis_off()
                    continue
                sub = X[:, nz]
                sub = np.asarray(
                    sub.toarray() if hasattr(sub, "toarray") else sub, dtype=np.float64
                )
                shares = gene_variance_shares(sub, w[nz], z)
                rank = shares if rank_by == "share" else w[nz]
                if panel == "weights":
                    grid = _draw_weights(ax, w[nz], names[nz], rank, n_genes)
                elif panel == "shares":
                    grid = _draw_shares(ax, shares, w[nz], rank_by, n_genes)
                else:
                    active = None
                    if active_quantile is not None:
                        active = np.abs(z) >= np.quantile(np.abs(z), active_quantile)
                    grid = _draw_contributions(
                        ax,
                        sub * w[nz],
                        names[nz],
                        shares,
                        w[nz],
                        rank,
                        n_genes,
                        normalize,
                        active,
                        z,
                    )
            else:
                grid = _draw_groups(ax, z, labels, keep, lut, group_by)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(_FAINT)
            # `color`/`labelcolor`, not `colors`: the latter sets both and would
            # overwrite the per-gene label colours the panels set below.
            ax.tick_params(color=_MUTED, labelsize=_TICK, length=3)
            ax.tick_params(axis="x", labelcolor=_MUTED)
            if panel not in ("contributions", "weights"):
                ax.tick_params(axis="y", labelcolor=_MUTED)
            ax.grid(axis=grid, color="#f2f2f2", lw=0.6)
            ax.set_axisbelow(True)

    # The contribution panel's two colours are a second encoding and need their own
    # key; without it the figure is not self-describing.
    split_key = (
        [
            Patch(facecolor=_POS, edgecolor="none", label="cells with score > 0"),
            Patch(facecolor=_NEG, edgecolor="none", label="cells with score < 0"),
        ]
        if "contributions" in panels
        else []
    )
    hued = [lv for lv in (keep or []) if lut.get(lv) != _OTHER]
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            markersize=8,
            markeredgecolor="none",
            markerfacecolor=lut[lv],
            label=_elide(lv, 30),
        )
        for lv in hued
    ]
    n_grey = len(keep or []) - len(hued) + n_other
    if n_grey:
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                markersize=8,
                markeredgecolor="none",
                markerfacecolor=_OTHER,
                label=(f"{_plural(n_grey, 'further group')} (labelled in the violins)"),
            )
        )
    handles = handles + split_key
    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=min(5, len(handles)),
            frameon=False,
            fontsize=_LEGEND,
            markerscale=1.5,
            handletextpad=0.5,
            columnspacing=1.6,
            labelspacing=0.7,
            bbox_to_anchor=(0.5, 1.0 + 1.45 / fig_size[1]),
        )
    # No tight_layout: the explicit gridspec spacing above is the intent, and
    # tight_layout both overrides it and warns about the figure-level legend.
    return fig, axes


def _display_values(x, scale: str) -> np.ndarray:
    """Expression prepared for colouring.

    ``"log1p"`` assumes the source holds *normalized, not yet logged* counts, which
    is what a scanpy pipeline leaves in a layer. ``"none"`` is for an already
    log-transformed layer -- logging twice is silent and wrong, so it is a choice
    rather than a guess. ``"zscore"`` standardizes per gene, making panels
    comparable at the cost of hiding how much signal a gene carries at all.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if scale == "log1p":
        if x.min() < 0:
            raise ValueError(
                "scale='log1p' needs non-negative expression, but the source has "
                f"values down to {x.min():.2f}. It is probably already scaled or "
                "z-scored -- pass scale='none', or point `layer` at counts."
            )
        return np.log1p(x)
    if scale == "zscore":
        sd = x.std()
        return (x - x.mean()) / (sd if sd > 0 else 1.0)
    if scale == "none":
        return x
    raise ValueError(f"unknown scale {scale!r}; use 'log1p', 'zscore' or 'none'")


def plot_dimension_gene_umaps(
    adata,
    *,
    dims: Sequence[int] | None = None,
    n_genes: int = 5,
    rank_by: Literal["share", "weight"] = "share",
    scale: Literal["log1p", "zscore", "none"] = "log1p",
    layer: str | None = None,
    weights_layer: str | None = None,
    basis: str = "X_umap",
    latent_key: str = "X_bae",
    weights_key: str = "BAE_encoder_weights",
    gene_names: str | None = None,
    include_score: bool = True,
    cmap: str = "viridis",
    score_cmap=None,
    symmetric_score: bool = True,
    point_size: float | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
):
    """A grid of UMAPs: one row per latent dimension, one column per top gene.

    Drawn with ``scanpy.pl.umap``, so the panels match the rest of a scanpy
    figure, and ranked by the same quantity :func:`plot_latent_dimensions` uses.
    Reading across a row shows whether a dimension's genes light up the *same*
    cells -- a coherent programme -- or different ones, a dimension summing
    unrelated signals. Neither the score curve nor the contribution violins can
    show that, because both have already summed over cells.

    Parameters
    ----------
    adata
        AnnData with a fitted BAE and a precomputed embedding in ``obsm[basis]``.
    dims
        Dimensions, one row each. ``None`` uses all.
    n_genes
        Genes per dimension, i.e. columns.
    rank_by
        ``"share"`` (default) or ``"weight"``. See :func:`gene_variance_shares`.
    scale
        Display transform for the expression: ``"log1p"`` (default), ``"zscore"``
        or ``"none"``.
    layer
        Layer holding the expression to colour by. ``None`` means ``adata.X`` --
        which under this package's contract is z-scored, so ``scale="log1p"`` will
        refuse it and say so rather than colour by something meaningless.
    weights_layer
        Layer the encoder was fitted on, if different from ``layer``. Affects only
        the variance shares, never the colouring.
    basis
        ``obsm`` key of the embedding.
    include_score
        Prepend a column colouring cells by the dimension's own latent score, so
        the genes can be compared against what they are meant to build.
    cmap
        Colour map for the gene panels when the values are non-negative.
    score_cmap
        Colour map for signed quantities. ``None`` uses a diverging
        blue-neutral-orange map matching the violin panels' sign colours.
    symmetric_score
        Centre signed colour scales on zero using symmetric limits, so the neutral
        colour marks zero rather than the data mean.
    latent_key, weights_key, gene_names, point_size, figsize, dpi
        As in :func:`plot_latent_dimensions`.

    Returns
    -------
    fig, axes
        The figure and its ``(n_dims, n_genes [+1])`` array of axes.

    Raises
    ------
    ImportError
        If scanpy is not installed.
    KeyError
        If the embedding, the code or the encoder is missing.
    ValueError
        If ``dims`` is out of range or ``scale`` is unknown.

    Examples
    --------
    >>> plot_dimension_gene_umaps(adata, dims=[0, 3], layer="lognorm")  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    try:
        import scanpy as sc
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "plot_dimension_gene_umaps draws through scanpy.pl.umap so its panels "
            "match a scanpy figure, but scanpy is not installed. Install it with "
            "`pip install scanpy`, or use plot_latent_dimensions, which needs only "
            "matplotlib."
        ) from exc
    from anndata import AnnData

    if basis not in adata.obsm:
        raise KeyError(
            f"adata.obsm[{basis!r}] not found. Compute an embedding first, e.g. "
            f"sc.pp.neighbors(adata, use_rep={latent_key!r}); sc.tl.umap(adata)."
        )
    if latent_key not in adata.obsm:
        raise KeyError(f"adata.obsm[{latent_key!r}] not found; fit the model first")
    if weights_key not in adata.varm:
        raise KeyError(f"adata.varm[{weights_key!r}] not found; fit the model first")

    Z = np.asarray(adata.obsm[latent_key], dtype=np.float64)
    W = np.asarray(adata.varm[weights_key], dtype=np.float64)
    dims = list(range(Z.shape[1])) if dims is None else list(dims)
    if any(d < 0 or d >= Z.shape[1] for d in dims):
        raise ValueError(f"dims out of range for a {Z.shape[1]}-dimensional code: {dims}")
    names = (
        np.asarray(adata.var[gene_names], dtype=str)
        if gene_names
        else np.asarray(adata.var_names, dtype=str)
    )

    fit_src = adata.X if weights_layer is None else adata.layers[weights_layer]
    colour_src = adata.X if layer is None else adata.layers[layer]

    # Panels are assembled into a throwaway AnnData carrying only the embedding and
    # the display columns, so the caller's object is never mutated -- not even
    # transiently, which matters if this raises midway.
    frame, titles, n_selected = {}, {}, {}
    for dim in dims:
        w = W[:, dim]
        nz = np.flatnonzero(w)
        n_selected[dim] = int(nz.size)
        if nz.size == 0:
            titles[dim] = []
            continue
        sub = fit_src[:, nz]
        sub = np.asarray(sub.toarray() if hasattr(sub, "toarray") else sub, dtype=np.float64)
        shares = gene_variance_shares(sub, w[nz], Z[:, dim])
        rank = shares if rank_by == "share" else w[nz]
        order = nz[np.argsort(np.abs(rank))[::-1][:n_genes]]
        share_of = dict(zip(nz, shares, strict=True))
        cols = []
        for gene in order:
            key = f"d{dim}:{names[gene]}"
            col = colour_src[:, gene]
            frame[key] = _display_values(
                col.toarray() if hasattr(col, "toarray") else np.asarray(col), scale
            )
            # Two facts on two channels. The percentage is a *magnitude*; because
            # `share = w * cov(X_g, s)` and a negative-weight gene is anti-correlated
            # with the score, the product is positive either way. Direction is the
            # sign of the encoder weight, and lives in the title colour.
            cols.append(
                (key, f"{names[gene]}  {abs(share_of[gene]):.0%}", _POS if w[gene] >= 0 else _NEG)
            )
        titles[dim] = cols
        if include_score:
            frame[f"d{dim}:score"] = Z[:, dim]

    tmp = AnnData(np.empty((adata.n_obs, 0)), obs=frame)
    tmp.obsm[basis] = np.asarray(adata.obsm[basis])

    ncol = n_genes + (1 if include_score else 0)
    fig_size = figsize or (2.9 * ncol, 3.0 * len(dims) + 1.1)
    fig, axes = plt.subplots(
        len(dims),
        ncol,
        squeeze=False,
        dpi=dpi,
        figsize=fig_size,
        gridspec_kw={
            "hspace": 0.30,
            "wspace": 0.22,
            "top": 1 - 0.85 / fig_size[1],
            "bottom": 0.42 / fig_size[1],
        },
    )
    size = point_size if point_size is not None else max(120_000 / max(adata.n_obs, 1), 2.0)
    for row, dim in enumerate(dims):
        panels = ([(f"d{dim}:score", f"dim {dim} score", _INK)] if include_score else []) + titles[
            dim
        ]
        for col in range(ncol):
            ax = axes[row][col]
            if col >= len(panels):
                ax.axis("off")
                continue
            key, title, title_color = panels[col]
            is_score = key.endswith(":score")
            # A signed quantity needs a diverging map centred on zero; a non-negative
            # magnitude needs a sequential one. The score is always signed, and the
            # gene panels become signed as soon as `scale="zscore"` -- at which point
            # a sequential map would put its neutral colour at the data mean.
            signed = is_score or scale == "zscore"
            kw = {}
            if signed:
                kw["cmap"] = score_cmap if score_cmap is not None else _score_cmap()
                if symmetric_score:
                    lim = float(np.abs(tmp.obs[key].to_numpy()).max())
                    kw["vmin"], kw["vmax"] = -lim, lim
            else:
                kw["cmap"] = cmap
            sc.pl.umap(
                tmp,
                color=key,
                ax=ax,
                show=False,
                size=size,
                frameon=False,
                colorbar_loc="right",
                title=title,
                **kw,
            )
            ax.set_title(
                title,
                fontsize=_TITLE if col == 0 else _LABEL,
                color=title_color,
                loc="center",
                fontweight="normal" if is_score else "bold",
                pad=18 if is_score else 6,
            )
            if is_score:
                # The count the percentages are shares *of*: a large share of a
                # 24-gene encoder and of a 300-gene one mean very different things.
                ax.text(
                    0.5,
                    1.0,
                    f"{_plural(n_selected.get(dim, 0), 'gene')} selected",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=_TICK,
                    color=_MUTED,
                )

    what = {"log1p": "log1p expression", "zscore": "z-scored expression", "none": "expression"}[
        scale
    ]
    # Aligned to the first panel's left edge, not the figure's, and queried after a
    # draw because scanpy inserts a colorbar per panel, shifting the axes from where
    # the gridspec put them.
    fig.canvas.draw()
    x_left = min(axes[r][0].get_position().x0 for r in range(len(dims)))
    fig.text(
        x_left,
        1 - 0.06 / fig_size[1],
        f"Top {n_genes} genes per latent dimension, coloured by {what}",
        ha="left",
        va="top",
        fontsize=_TITLE + 1,
        color=_INK,
        fontweight="bold",
    )
    fig.text(
        x_left,
        0.30 / fig_size[1],
        "%  share of dimension variance   ·   gene name  orange = positive weight, blue = negative",
        ha="left",
        va="top",
        fontsize=_TICK,
        color=_MUTED,
    )
    return fig, axes


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, column-wise. Spearman is Pearson on these.

    Done here rather than through ``scipy.stats`` so this module keeps needing only
    numpy and matplotlib; ties matter because a sparse encoder leaves many cells at
    exactly zero on a subgroup dimension, and ordinal ranks would break those ties
    arbitrarily and bias the correlation.
    """
    out = np.empty_like(x, dtype=np.float64)
    for j in range(x.shape[1]):
        col = x[:, j]
        order = np.argsort(col, kind="mergesort")
        ranks = np.empty(col.size, dtype=np.float64)
        ranks[order] = np.arange(1, col.size + 1, dtype=np.float64)
        # average the ranks within each run of equal values
        srt = col[order]
        start = 0
        for stop in range(1, srt.size + 1):
            if stop == srt.size or srt[stop] != srt[start]:
                if stop - start > 1:
                    ranks[order[start:stop]] = ranks[order[start:stop]].mean()
                start = stop
        out[:, j] = ranks
    return out


def plot_dimension_correlation(
    adata,
    *,
    dims: Sequence[int] | None = None,
    method: Literal["pearson", "spearman"] = "pearson",
    absolute: bool = False,
    latent_key: str = "X_bae",
    annotate: bool | None = None,
    cmap=None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
):
    """Heatmap of the correlation between latent dimensions.

    Whether two dimensions are near-duplicates decides whether they can be read as
    two findings or one, and nothing else in the readout answers it -- every other
    panel looks at one dimension at a time.

    It matters most when the fit does not enforce separation:
    :class:`~structboost.BAEConfig` defaults ``disentanglement`` to
    ``"orthogonal"``, and a fit that turns it off can carry two dimensions
    describing the same programme with nothing else flagging it.

    Parameters
    ----------
    adata
        AnnData with a fitted BAE, or an ``(n_cells, n_dims)`` array of scores.
    dims
        Dimensions to include. ``None`` uses all.
    method
        ``"pearson"`` (default) or ``"spearman"``. Spearman is computed as Pearson
        on tie-averaged ranks, which matters here: a sparse encoder leaves many
        cells at exactly zero on a subgroup dimension.
    absolute
        Plot ``|r|`` instead of ``r``. Signed values get a diverging map centred on
        zero, because the sign of a correlation is meaningful and a latent
        dimension's own sign is arbitrary; absolute values are a magnitude and get
        a sequential one.
    annotate
        Write each value into its cell. ``None`` annotates when there are at most
        12 dimensions, beyond which the numbers stop fitting.
    cmap
        Override the colour map chosen by ``absolute``.
    figsize, dpi
        Overrides for the computed size, and the figure's resolution.

    Returns
    -------
    fig, ax

    Raises
    ------
    KeyError
        If ``latent_key`` is missing from an AnnData input.
    ValueError
        If ``dims`` is out of range or ``method`` is unknown.

    Examples
    --------
    >>> plot_dimension_correlation(adata, method="spearman")      # doctest: +SKIP
    >>> plot_dimension_correlation(adata, absolute=True)          # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    if hasattr(adata, "obsm"):
        if latent_key not in adata.obsm:
            raise KeyError(f"adata.obsm[{latent_key!r}] not found; fit the model first")
        Z = np.asarray(adata.obsm[latent_key], dtype=np.float64)
    else:
        Z = np.asarray(adata, dtype=np.float64)
    dims = list(range(Z.shape[1])) if dims is None else list(dims)
    if any(d < 0 or d >= Z.shape[1] for d in dims):
        raise ValueError(f"dims out of range for a {Z.shape[1]}-dimensional code: {dims}")
    if method not in ("pearson", "spearman"):
        raise ValueError(f"unknown method {method!r}; use 'pearson' or 'spearman'")

    sub = Z[:, dims]
    corr = np.corrcoef((_average_ranks(sub) if method == "spearman" else sub), rowvar=False)
    corr = np.atleast_2d(corr)
    shown = np.abs(corr) if absolute else corr

    n = len(dims)
    fig, ax = plt.subplots(figsize=figsize or (0.55 * n + 3.0, 0.55 * n + 2.2), dpi=dpi)
    im = ax.imshow(
        shown,
        cmap=cmap if cmap is not None else ("viridis" if absolute else _score_cmap()),
        vmin=0.0 if absolute else -1.0,
        vmax=1.0,
    )
    ax.set_xticks(range(n), [str(d) for d in dims], fontsize=_TICK)
    ax.set_yticks(range(n), [str(d) for d in dims], fontsize=_TICK)
    ax.tick_params(color=_MUTED, labelcolor=_MUTED, length=3)
    ax.set_xlabel("latent dimension", fontsize=_LABEL, color=_MUTED)
    ax.set_ylabel("latent dimension", fontsize=_LABEL, color=_MUTED)
    if annotate is None:
        annotate = n <= 12
    if annotate:
        # Ink chosen from the cell's actual luminance, not from a threshold on the
        # value. A threshold has to assume how the colour map runs, and the two used
        # here run oppositely: the diverging map is palest in the middle and
        # saturated at both ends, viridis darkens monotonically toward zero. One
        # rule cannot serve both, and a user-supplied `cmap` could be anything.
        for i in range(n):
            for j in range(n):
                v = shown[i, j]
                r, g, b, _ = im.cmap(im.norm(v))
                lin = [
                    c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)
                ]
                luminance = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=_TICK - 1,
                    color="white" if luminance < 0.4 else _INK,
                )

    off = shown[~np.eye(n, dtype=bool)] if n > 1 else np.zeros(0)
    worst = float(np.abs(off).max()) if off.size else 0.0
    what = "|r|" if absolute else "r"
    ax.set_title(
        f"{method.capitalize()} correlation between latent dimensions ({what})\n"
        f"largest off-diagonal |r| = {worst:.2f}",
        fontsize=_TITLE,
        loc="left",
        color=_INK,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, ax
