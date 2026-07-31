"""Plotting helpers for structboost.

These utilities are intentionally lightweight and optional: they only depend on
matplotlib and numpy.
"""

from __future__ import annotations

from collections.abc import Sequence

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
