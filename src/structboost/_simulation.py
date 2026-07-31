"""Simulation utilities for generating synthetic scRNA-seq count data.

Generates negative-binomial UMI counts with block/stage structure, useful for
testing and benchmarking BAE and boosting algorithms. Cells are assigned to
contiguous stages; each stage over-expresses a sliding window of marker genes.
Optional technical effects (library size, batch, ambient RNA, dropout) can be
layered on top, and two orthogonal knobs control the noise level:
``effect_size`` (signal strength) and ``dispersion`` (overdispersion).

References
----------
Hess, M. et al. (2020) Bioinformatics — original block/stage simulation design
that the marker-window geometry follows.

Zappia, L., Phipson, B. & Oshlack, A. (2017) Splatter: simulation of single-cell
RNA sequencing data. Genome Biology 18:174 — gamma-distributed gene means,
gamma-Poisson counts, log-normal library sizes, multiplicative batch effects,
and the mean-dependent logistic dropout model.

Young, M. D. & Behjati, S. (2020) SoupX removes ambient RNA contamination from
droplet-based single-cell RNA sequencing data. GigaScience 9(12) — the pooled
ambient "soup" contamination model.

Notes
-----
Counts are generated in row chunks of ``_ROW_CHUNK``. This constant is part of
the reproducibility contract: changing it changes the random stream for a fixed
seed, even though the distribution is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

#: Boolean mask array. Named rather than spelled ``NDArray[np.bool_]`` inline:
#: the trailing underscore in ``np.bool_`` is valid Python but reads as reference
#: syntax to the documentation builder, which then reports a broken target on
#: every page that renders the annotation.
BoolArray = NDArray[np.bool_]


if TYPE_CHECKING:
    import anndata as ad

_ROW_CHUNK = 2048


@dataclass(frozen=True)
class SimulationResult:
    """Simulated counts together with the ground truth used to generate them.

    Attributes
    ----------
    counts
        Integer UMI counts, shape (n, n_genes).
    stage_labels
        Stage index per cell, shape (n,). ``-1`` marks leftover background cells
        that belong to no stage (possible when ``n`` is not divisible by
        ``stageno``).
    stage_sizes
        Number of cells per stage, shape (stageno,).
    gene_level
        Shape ``(n_genes,)``. Which level of the cell-type hierarchy each gene
        marks — ``0`` is the broadest — or ``-1`` for a noise gene. With
        ``hierarchy=None`` every marker is level ``0``.
    leaf_paths
        Shape ``(stageno, n_levels)``. Ancestry of each leaf population, so
        ``leaf_paths[k, d]`` is the node index of population ``k`` at depth ``d``.
        These become the per-level cell labels in the AnnData wrapper.
    hierarchy
        The branching factors used, or ``None`` for the flat layout.
    marker_mask
        Boolean marker indicator, shape (stageno, n_genes). ``marker_mask[k, j]``
        is True when gene j is a marker of stage k. Genes in the overlap between
        two consecutive stages are True in both rows.
    gene_means
        Baseline expression mean per gene (lambda_j), shape (n_genes,).
    size_factors
        Per-cell library size factor, shape (n,). All exactly 1.0 when
        ``lib_size_sd == 0``.
    batch_labels
        Batch index per cell, shape (n,).
    params
        Scalar simulation parameters. Contains no ``None`` values so it can be
        stored in ``adata.uns`` and round-tripped through h5ad.
    """

    counts: NDArray[np.int32]
    stage_labels: NDArray[np.intp]
    stage_sizes: NDArray[np.intp]
    marker_mask: BoolArray
    gene_means: NDArray[np.float64]
    size_factors: NDArray[np.float64]
    batch_labels: NDArray[np.intp]
    params: dict[str, int | float | bool | str]
    gene_level: NDArray[np.intp] = None  # type: ignore[assignment]
    leaf_paths: NDArray[np.intp] = None  # type: ignore[assignment]
    hierarchy: tuple[int, ...] | None = None

    @property
    def marker_genes(self) -> BoolArray:
        """Genes that are a marker of at least one stage, shape (n_genes,)."""
        return self.marker_mask.any(axis=0)


def _marker_mask(stageno: int, n_genes: int, stagep: int, stageoverlap: int) -> BoolArray:
    """Build the (stageno, n_genes) marker indicator.

    Stage ``k`` covers the window ``[curp, curp + stagep)``, where ``curp``
    advances by ``stagep - stageoverlap`` between stages, so consecutive stages
    share exactly ``stageoverlap`` marker genes.
    """
    mask = np.zeros((stageno, n_genes), dtype=np.bool_)
    curp = 0
    for k in range(stageno):
        mask[k, curp : curp + stagep] = True
        curp += stagep - stageoverlap
    return mask


def _hierarchical_marker_mask(
    hierarchy: tuple[int, ...],
    n_genes: int,
    markers_per_level: tuple[int, ...],
) -> tuple[BoolArray, NDArray[np.intp], NDArray[np.intp]]:
    """Build a nested marker structure over a cell-type tree.

    Real cell taxonomies are hierarchical: a marker such as *Vip* labels an entire
    subclass, while *Mybpc1* distinguishes one type within it — which is why Tasic
    et al. (2016) name their clusters ``<subclass> <marker>``. This reproduces that
    structure, in contrast to the flat layout built by :func:`_marker_mask`, where
    populations only share genes with their immediate neighbours (a trajectory
    model rather than a taxonomy).

    ``hierarchy`` gives the branching factor at each level, outermost first, so
    ``(2, 5)`` is two classes of five types each and produces ``2 * 5 = 10`` leaf
    populations. Every leaf expresses the union of its ancestors' marker blocks:
    a cell of type ``(1, 3)`` carries class 1's markers *and* type (1,3)'s markers.
    Marker blocks are disjoint across the whole tree, so a gene belongs to exactly
    one node and its level is unambiguous.

    Parameters
    ----------
    hierarchy
        Branching factors, outermost first. ``len(hierarchy)`` is the tree depth.
    n_genes
        Total genes; anything not assigned to a node is a noise gene.
    markers_per_level
        Marker-block size at each level, same length as ``hierarchy``.

    Returns
    -------
    mask
        ``(n_leaves, n_genes)`` boolean marker indicator, unions taken over ancestors.
    gene_level
        ``(n_genes,)`` level index each gene marks, ``-1`` for noise genes.
    leaf_paths
        ``(n_leaves, n_levels)`` ancestry, so ``leaf_paths[k, d]`` is the node index
        of leaf ``k`` at depth ``d``. Used to build the per-level cell labels.
    """
    n_levels = len(hierarchy)
    n_leaves = int(np.prod(hierarchy))

    # Ancestry of every leaf: the mixed-radix expansion of its index.
    leaf_paths = np.zeros((n_leaves, n_levels), dtype=np.intp)
    for k in range(n_leaves):
        rest = k
        for d in range(n_levels - 1, -1, -1):
            leaf_paths[k, d] = rest % hierarchy[d]
            rest //= hierarchy[d]

    mask = np.zeros((n_leaves, n_genes), dtype=np.bool_)
    gene_level = np.full(n_genes, -1, dtype=np.intp)

    cursor = 0
    for depth in range(n_levels):
        block = markers_per_level[depth]
        # Nodes at this depth are the distinct ancestry prefixes of length depth+1.
        prefixes = np.unique(leaf_paths[:, : depth + 1], axis=0)
        for prefix in prefixes:
            members = np.all(leaf_paths[:, : depth + 1] == prefix, axis=1)
            mask[members, cursor : cursor + block] = True
            gene_level[cursor : cursor + block] = depth
            cursor += block

    return mask, gene_level, leaf_paths


def _default_hierarchy(stageno: int) -> tuple[int, ...]:
    """A balanced two-level taxonomy over ``stageno`` leaf populations.

    Picks the divisor of ``stageno`` closest to its square root, so ten
    populations become two classes of five rather than a lopsided split. Prime
    counts have no non-trivial factorisation and fall back to a single level,
    which still gives disjoint marker blocks per population — the hierarchy is
    simply flat for that count.
    """
    if stageno < 4:
        return (stageno,)
    divisors = [d for d in range(2, stageno) if stageno % d == 0]
    if not divisors:
        return (stageno,)
    target = stageno**0.5
    best = min(divisors, key=lambda d: (abs(d - target), d))
    return (best, stageno // best)


def _resolve_markers_per_level(
    hierarchy: tuple[int, ...], n_genes: int, stagep: int
) -> tuple[int, ...]:
    """Split the per-population marker budget across hierarchy levels.

    A hierarchy needs many more distinct genes than a flat layout: each leaf owns
    a private block *and* inherits one from every ancestor, so the total is
    ``sum_d n_nodes(d) * block(d)`` rather than one window per population. An even
    split across levels therefore overflows ``n_genes`` easily.

    Starts from an even split holding ``sum(block) == stagep`` (so each population
    carries the requested number of markers) and, while the total does not fit,
    shifts one marker from the deepest level to the broadest. That direction is
    what makes the layout cheaper: the deepest level has the most nodes, so each
    marker moved out of it frees the most genes, and shared programs grow rather
    than shrink — which is the biologically sensible way to spend a tight budget.

    Raises
    ------
    ValueError
        If no allocation fits, i.e. even one private marker per leaf plus one
        shared marker per higher-level node exceeds ``n_genes``.
    """
    depth = len(hierarchy)
    nodes = [int(np.prod(hierarchy[: d + 1])) for d in range(depth)]

    def total(blocks: list[int]) -> int:
        return sum(nodes[d] * blocks[d] for d in range(depth))

    base = max(1, stagep // depth)
    blocks = [base] * depth
    blocks[0] += max(0, stagep - base * depth)

    while total(blocks) > n_genes:
        # Only levels below the root can donate: moving a marker from level 0 to
        # level 0 is a no-op and would spin forever.
        deepest = max((d for d in range(1, depth) if blocks[d] > 1), default=None)
        if deepest is None:
            raise ValueError(
                f"hierarchical marker layout does not fit: {hierarchy} needs at least "
                f"{total([1] * depth)} genes for one marker per node, got n_genes={n_genes}. "
                "Increase n_genes, or reduce the hierarchy depth or branching."
            )
        blocks[deepest] -= 1
        blocks[0] += 1
    return tuple(blocks)


def _resolve_stage_sizes(
    n: int, stageno: int, stagen: int | None, imbalanced: bool, rng: np.random.Generator
) -> NDArray[np.intp]:
    """Cells per stage: Dirichlet-distributed when imbalanced, else uniform."""
    if imbalanced:
        proportions = rng.dirichlet(np.ones(stageno))
        sizes = np.maximum(1, np.round(proportions * n).astype(np.intp))
        diff = n - sizes.sum()
        if diff != 0:
            order = np.argsort(sizes)[::-1]
            for i in range(abs(diff)):
                sizes[order[i % stageno]] += np.sign(diff)
        return sizes
    if stagen is None:
        stagen = n // stageno
    return np.full(stageno, stagen, dtype=np.intp)


def _validate(
    *,
    n: int,
    n_genes: int,
    stageno: int,
    stagep: int | None,
    stagen: int | None,
    stageoverlap: int,
    imbalanced: bool,
    hierarchy: tuple[int, ...] | None = None,
    markers_per_level: tuple[int, ...] | None = None,
    base_mean: float,
    gene_mean_shape: float,
    effect_size: float,
    effect_size_sd: float,
    dispersion: float,
    lib_size_sd: float,
    n_batches: int,
    batch_effect_sd: float,
    ambient_frac: float,
    dropout_mid: float | None,
    dropout_shape: float,
) -> int:
    """Validate all parameters before any random number is drawn.

    Returns the resolved ``stagep``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n_genes < 1:
        raise ValueError(f"n_genes must be >= 1, got {n_genes}")
    if stageno < 1:
        raise ValueError(f"stageno must be >= 1, got {stageno}")
    if stageno > n:
        raise ValueError(f"stageno must be <= n, got stageno={stageno} and n={n}")
    if stagep is None:
        stagep = n_genes // stageno
    if stagep < 1:
        raise ValueError(
            f"stagep must be >= 1, got {stagep} "
            f"(n_genes={n_genes} // stageno={stageno} is 0; increase n_genes)"
        )
    if stagep > n_genes:
        raise ValueError(f"stagep must be <= n_genes, got stagep={stagep}, n_genes={n_genes}")
    if hierarchy is not None:
        if any(b < 1 for b in hierarchy) or len(hierarchy) < 1:
            raise ValueError(f"hierarchy branching factors must all be >= 1, got {hierarchy}")
        if int(np.prod(hierarchy)) != stageno:
            raise ValueError(
                f"hierarchy {hierarchy} implies {int(np.prod(hierarchy))} leaf populations "
                f"but stageno={stageno}. Set one or the other, not both."
            )
        if markers_per_level is not None:
            if len(markers_per_level) != len(hierarchy):
                raise ValueError(
                    f"markers_per_level must have one entry per level: got "
                    f"{len(markers_per_level)} for a depth-{len(hierarchy)} hierarchy"
                )
            if any(m < 1 for m in markers_per_level):
                raise ValueError(f"markers_per_level entries must be >= 1, got {markers_per_level}")
            needed = sum(
                int(np.prod(hierarchy[: d + 1])) * markers_per_level[d]
                for d in range(len(hierarchy))
            )
            if needed > n_genes:
                raise ValueError(
                    f"hierarchical marker layout does not fit: {hierarchy} with "
                    f"{markers_per_level} markers per level needs n_genes >= {needed}, "
                    f"got {n_genes}"
                )
    else:
        # Flat sliding-window layout only; `stageoverlap` has no meaning otherwise.
        if not 0 <= stageoverlap < stagep:
            raise ValueError(
                f"stageoverlap must satisfy 0 <= stageoverlap < stagep, "
                f"got stageoverlap={stageoverlap} and stagep={stagep}"
            )
        needed = (stageno - 1) * (stagep - stageoverlap) + stagep
        if needed > n_genes:
            raise ValueError(
                f"stage layout does not fit: {stageno} stages of {stagep} genes with "
                f"overlap {stageoverlap} need n_genes >= {needed}, got {n_genes}"
            )
    if imbalanced and stagen is not None:
        raise ValueError("stagen must be None when imbalanced=True; stage sizes are drawn")
    if stagen is not None:
        if stagen < 1:
            raise ValueError(f"stagen must be >= 1, got {stagen}")
        if stagen * stageno > n:
            raise ValueError(f"stagen * stageno must be <= n, got {stagen} * {stageno} > {n}")
    if base_mean <= 0:
        raise ValueError(f"base_mean must be > 0, got {base_mean}")
    if gene_mean_shape <= 0:
        raise ValueError(f"gene_mean_shape must be > 0, got {gene_mean_shape}")
    if effect_size <= 0:
        raise ValueError(f"effect_size must be > 0, got {effect_size}")
    if effect_size_sd < 0:
        raise ValueError(f"effect_size_sd must be >= 0, got {effect_size_sd}")
    if dispersion < 0:
        raise ValueError(f"dispersion must be >= 0, got {dispersion}")
    if lib_size_sd < 0:
        raise ValueError(f"lib_size_sd must be >= 0, got {lib_size_sd}")
    if not 1 <= n_batches <= n:
        raise ValueError(f"n_batches must satisfy 1 <= n_batches <= n, got {n_batches}")
    if batch_effect_sd < 0:
        raise ValueError(f"batch_effect_sd must be >= 0, got {batch_effect_sd}")
    if not 0.0 <= ambient_frac < 1.0:
        raise ValueError(f"ambient_frac must be in [0, 1), got {ambient_frac}")
    if dropout_mid is not None and not np.isfinite(dropout_mid):
        raise ValueError(f"dropout_mid must be finite or None, got {dropout_mid}")
    if not np.isfinite(dropout_shape):
        raise ValueError(f"dropout_shape must be finite, got {dropout_shape}")
    return stagep


def sim_scrnaseq_data(
    *,
    n: int = 1000,
    n_genes: int = 50,
    stageno: int = 10,
    stagep: int | None = None,
    stagen: int | None = None,
    stageoverlap: int | None = None,
    hierarchy: tuple[int, ...] | bool | None = True,
    markers_per_level: tuple[int, ...] | None = None,
    imbalanced: bool = False,
    base_mean: float = 2.0,
    gene_mean_shape: float = 4.0,
    effect_size: float = 10.0,
    effect_size_sd: float = 0.0,
    dispersion: float = 0.2,
    lib_size_sd: float = 0.0,
    n_batches: int = 1,
    batch_effect_sd: float = 0.0,
    ambient_frac: float = 0.0,
    dropout_mid: float | None = None,
    dropout_shape: float = -1.0,
    seed: int = 1,
) -> SimulationResult:
    """Simulate negative-binomial scRNA-seq counts with block/stage structure.

    Cells are assigned to ``stageno`` contiguous stages in row order. Each stage
    over-expresses a sliding window of ``stagep`` marker genes by a factor of
    ``effect_size``; consecutive windows share ``stageoverlap`` genes. Counts are
    drawn from a gamma-Poisson (negative binomial) mixture, so
    ``Var[c] = mu + dispersion * mu**2``.

    All technical effects are neutral by default, so the base call produces clean,
    well-separated block structure.

    Parameters
    ----------
    n
        Number of cells.
    n_genes
        Number of genes.
    stageno
        Number of stages (cell populations).
    stagep
        Marker genes per stage. Defaults to ``n_genes // stageno``.
    stagen
        Cells per stage. Defaults to ``n // stageno``. Must be None when
        ``imbalanced=True``.
    stageoverlap
        Marker genes shared between consecutive stages. Must be < ``stagep``.
    imbalanced
        If True, draw stage sizes from a symmetric Dirichlet instead of using
        equal sizes.
    base_mean
        Expected baseline expression mean across genes. Gene means are drawn as
        ``Gamma(gene_mean_shape, base_mean / gene_mean_shape)``.
    gene_mean_shape
        Shape of the gene-mean gamma; the coefficient of variation of gene means
        is ``1 / sqrt(gene_mean_shape)``.
    effect_size
        Marker-to-background mean ratio. This is the signal-strength knob: lower
        values make stages harder to separate.
    effect_size_sd
        Log-normal spread of the per-gene fold change around ``effect_size``.
        0.0 gives every marker gene exactly ``effect_size``.
    dispersion
        Negative-binomial overdispersion; the biological coefficient of variation
        is ``sqrt(dispersion)``. This is the noise knob, orthogonal to
        ``effect_size``. 0.0 gives exact Poisson counts.
    lib_size_sd
        Log-normal standard deviation of the per-cell size factor. 0.0 gives
        uniform sequencing depth.
    n_batches
        Number of technical batches. Cells are assigned round-robin, so batch is
        orthogonal to stage.
    batch_effect_sd
        Log-normal standard deviation of the per-gene, per-batch multiplicative
        shift. The shifts are normalized to unit geometric mean per gene, so
        enabling batch effects does not move marginal gene means.
    ambient_frac
        Fraction of each cell's counts replaced by draws from a pooled ambient
        profile. Library size is preserved exactly.
    dropout_mid
        Midpoint of the mean-dependent logistic dropout curve, on the log-mean
        scale. ``None`` disables dropout entirely.
    dropout_shape
        Steepness of the dropout curve. Negative values (the default) mean
        high-expression genes drop out less.
    seed
        Random seed. Independent sub-streams are spawned per component, so
        changing one parameter does not perturb the others' draws.

    Returns
    -------
    SimulationResult
        Counts plus the ground truth (stage labels, marker mask, gene means).

    Raises
    ------
    ValueError
        If any parameter is out of range or the stage layout does not fit in
        ``n_genes``.

    References
    ----------
    The count model reimplemented here — gamma-distributed gene means,
    gamma-Poisson counts, log-normal library sizes, multiplicative batch effects
    and mean-dependent logistic dropout — follows Splatter: Zappia, L.,
    Phipson, B. & Oshlack, A. (2017). *Splatter: simulation of single-cell RNA
    sequencing data.* Genome Biology 18, 174.

    Examples
    --------
    >>> from structboost import sim_scrnaseq_data
    >>> res = sim_scrnaseq_data(n=100, n_genes=20, stageno=4, seed=0)
    >>> res.counts.shape
    (100, 20)
    >>> res.marker_mask.shape
    (4, 20)
    """
    if hierarchy is True:
        hierarchy = _default_hierarchy(stageno)
    elif hierarchy is False:
        hierarchy = None
    hierarchical = hierarchy is not None
    if hierarchical and stageoverlap not in (None, 0):
        raise ValueError(
            "stageoverlap is not used when hierarchy is set: overlap between "
            "populations comes from shared ancestor markers instead. Pass "
            "hierarchy=None for the flat sliding-window layout."
        )
    stageoverlap = 0 if hierarchical else (2 if stageoverlap is None else stageoverlap)

    stagep = _validate(
        n=n,
        n_genes=n_genes,
        stageno=stageno,
        stagep=stagep,
        stagen=stagen,
        stageoverlap=stageoverlap,
        imbalanced=imbalanced,
        hierarchy=hierarchy,
        markers_per_level=markers_per_level,
        base_mean=base_mean,
        gene_mean_shape=gene_mean_shape,
        effect_size=effect_size,
        effect_size_sd=effect_size_sd,
        dispersion=dispersion,
        lib_size_sd=lib_size_sd,
        n_batches=n_batches,
        batch_effect_sd=batch_effect_sd,
        ambient_frac=ambient_frac,
        dropout_mid=dropout_mid,
        dropout_shape=dropout_shape,
    )

    # Independent sub-streams so each knob can be swept without perturbing others.
    seeds = np.random.SeedSequence(seed).spawn(7)
    rng_struct, rng_gene, rng_effect, rng_tech, rng_count, rng_ambient, rng_dropout = (
        np.random.default_rng(s) for s in seeds
    )

    # --- Phase A: deterministic structure ------------------------------------
    stage_sizes = _resolve_stage_sizes(n, stageno, stagen, imbalanced, rng_struct)
    stage_labels = np.full(n, -1, dtype=np.intp)
    start = 0
    for k, size in enumerate(stage_sizes):
        stop = min(start + int(size), n)
        stage_labels[start:stop] = k
        start = stop

    if hierarchical:
        if markers_per_level is None:
            markers_per_level = _resolve_markers_per_level(hierarchy, n_genes, stagep)
            _validate(
                n=n,
                n_genes=n_genes,
                stageno=stageno,
                stagep=stagep,
                stagen=stagen,
                stageoverlap=stageoverlap,
                imbalanced=imbalanced,
                base_mean=base_mean,
                gene_mean_shape=gene_mean_shape,
                effect_size=effect_size,
                effect_size_sd=effect_size_sd,
                dispersion=dispersion,
                lib_size_sd=lib_size_sd,
                n_batches=n_batches,
                batch_effect_sd=batch_effect_sd,
                ambient_frac=ambient_frac,
                dropout_mid=dropout_mid,
                dropout_shape=dropout_shape,
                hierarchy=hierarchy,
                markers_per_level=markers_per_level,
            )
        mask, gene_level, leaf_paths = _hierarchical_marker_mask(
            hierarchy, n_genes, markers_per_level
        )
    else:
        mask = _marker_mask(stageno, n_genes, stagep, stageoverlap)
        gene_level = np.where(mask.any(axis=0), 0, -1).astype(np.intp)
        leaf_paths = np.arange(stageno, dtype=np.intp)[:, None]

    gene_means = rng_gene.gamma(gene_mean_shape, base_mean / gene_mean_shape, size=n_genes)

    # Fold-change matrix; row 0 is the background row, rows 1.. are the stages.
    z_eff = rng_effect.normal(size=(stageno, n_genes))
    fold_stage = effect_size * np.exp(effect_size_sd * z_eff - effect_size_sd**2 / 2)
    fold = np.ones((stageno + 1, n_genes), dtype=np.float64)
    fold[1:] = np.where(mask, fold_stage, 1.0)

    batch_labels = (np.arange(n) % n_batches).astype(np.intp)
    log_batch = batch_effect_sd * rng_tech.normal(size=(n_batches, n_genes))
    log_batch -= log_batch.mean(axis=0, keepdims=True)  # unit geometric mean per gene
    batch_factor = np.exp(log_batch)

    size_factors = np.exp(lib_size_sd * rng_tech.normal(size=n) - lib_size_sd**2 / 2)
    size_factors /= size_factors.mean()

    if not np.isfinite(fold).all() or not np.isfinite(batch_factor).all():
        raise ValueError(
            "non-finite expression multipliers; reduce effect_size_sd or batch_effect_sd"
        )

    # Ambient soup profile: the mean expression per gene, in closed form.
    soup_profile: NDArray[np.float64] | None = None
    if ambient_frac > 0.0:
        flat = (stage_labels + 1) * n_batches + batch_labels
        weights = np.bincount(
            flat, weights=size_factors, minlength=(stageno + 1) * n_batches
        ).reshape(stageno + 1, n_batches)
        mean_expr = gene_means / n * np.einsum("sg,sj,gj->j", weights, fold, batch_factor)
        total = mean_expr.sum()
        soup_profile = mean_expr / total if total > 0 else np.full(n_genes, 1.0 / n_genes)

    # --- Phase B: stochastic pipeline, chunked over rows ---------------------
    counts = np.empty((n, n_genes), dtype=np.int32)
    for a in range(0, n, _ROW_CHUNK):
        b = min(a + _ROW_CHUNK, n)
        mu = (
            gene_means[None, :]
            * fold[stage_labels[a:b] + 1, :]
            * batch_factor[batch_labels[a:b], :]
            * size_factors[a:b, None]
        )

        rate = rng_count.gamma(1.0 / dispersion, dispersion * mu) if dispersion > 0 else mu
        chunk = rng_count.poisson(rate)

        if soup_profile is not None:
            kept = rng_ambient.binomial(chunk, 1.0 - ambient_frac)
            n_ambient = chunk.sum(axis=1) - kept.sum(axis=1)
            chunk = kept + rng_ambient.multinomial(n_ambient, soup_profile)

        if dropout_mid is not None:
            mu_tilde = (1.0 - ambient_frac) * mu
            if soup_profile is not None:
                mu_tilde = mu_tilde + ambient_frac * mu.sum(axis=1, keepdims=True) * soup_profile
            logit = dropout_shape * (np.log(np.maximum(mu_tilde, 1e-12)) - dropout_mid)
            keep_prob = 1.0 / (1.0 + np.exp(logit))  # == 1 - sigmoid(logit)
            chunk = chunk * (rng_dropout.random(chunk.shape) < keep_prob)

        counts[a:b] = chunk.astype(np.int32, copy=False)

    params: dict[str, int | float | bool | str] = {
        "n": int(n),
        "n_genes": int(n_genes),
        "stageno": int(stageno),
        "stagep": int(stagep),
        "stageoverlap": int(stageoverlap),
        "hierarchy": str(hierarchy),
        "markers_per_level": str(markers_per_level),
        "imbalanced": bool(imbalanced),
        "base_mean": float(base_mean),
        "gene_mean_shape": float(gene_mean_shape),
        "effect_size": float(effect_size),
        "effect_size_sd": float(effect_size_sd),
        "dispersion": float(dispersion),
        "lib_size_sd": float(lib_size_sd),
        "n_batches": int(n_batches),
        "batch_effect_sd": float(batch_effect_sd),
        "ambient_frac": float(ambient_frac),
        "dropout": bool(dropout_mid is not None),
        "dropout_mid": float("nan") if dropout_mid is None else float(dropout_mid),
        "dropout_shape": float(dropout_shape),
        "seed": int(seed),
        "design": "negative-binomial",
    }

    return SimulationResult(
        counts=counts,
        stage_labels=stage_labels,
        gene_level=gene_level,
        leaf_paths=leaf_paths,
        hierarchy=tuple(hierarchy) if hierarchy is not None else None,
        stage_sizes=stage_sizes,
        marker_mask=mask,
        gene_means=gene_means,
        size_factors=size_factors,
        batch_labels=batch_labels,
        params=params,
    )


def sim_scrnaseq_anndata(
    *,
    n: int = 1000,
    n_genes: int = 50,
    stageno: int = 10,
    stagep: int | None = None,
    stagen: int | None = None,
    stageoverlap: int | None = None,
    hierarchy: tuple[int, ...] | bool | None = True,
    markers_per_level: tuple[int, ...] | None = None,
    imbalanced: bool = False,
    base_mean: float = 2.0,
    gene_mean_shape: float = 4.0,
    effect_size: float = 10.0,
    effect_size_sd: float = 0.0,
    dispersion: float = 0.2,
    lib_size_sd: float = 0.0,
    n_batches: int = 1,
    batch_effect_sd: float = 0.0,
    ambient_frac: float = 0.0,
    dropout_mid: float | None = None,
    dropout_shape: float = -1.0,
    seed: int = 1,
    target_sum: float = 1e4,
    standardize: bool = True,
) -> ad.AnnData:
    """Simulate scRNA-seq counts and package them as an AnnData object.

    Wraps :func:`sim_scrnaseq_data` and applies the standard preprocessing
    pipeline. Raw counts and library-size-normalized log1p values are always
    kept as layers, so ``adata.X`` can be swapped without re-simulating.

    Only the two preprocessing arguments are documented below; every other
    parameter is passed through to :func:`sim_scrnaseq_data` unchanged.

    Parameters
    ----------
    target_sum
        Library size each cell is normalized to before ``log1p``.
    standardize
        If True, ``adata.X`` holds the per-gene z-score of the log1p-normalized
        values, ready for :class:`~structboost.BAE`. If False, ``adata.X`` holds
        the log1p-normalized values themselves.

    Returns
    -------
    anndata.AnnData
        Shape (n, n_genes) with:

        - ``X`` — z-scored log1p values, or log1p values when
          ``standardize=False``
        - ``layers["counts"]`` — raw integer UMI counts
        - ``layers["lognorm"]`` — log1p of library-size-normalized counts
        - ``obs["stage"]`` — ground-truth stage label per cell
        - ``obs["stage_id"]``, ``obs["batch"]``, ``obs["size_factor"]``,
          ``obs["total_counts"]``
        - ``var["is_marker"]``, ``var["marker_stages"]``,
          ``var["n_marker_stages"]``, ``var["base_mean"]``
        - ``varm["marker_mask"]`` — (n_genes, stageno) ground-truth marker matrix
        - ``uns["simulation"]`` — simulation parameters

    Raises
    ------
    ImportError
        If anndata is not installed.
    ValueError
        If ``target_sum <= 0`` or any parameter of :func:`sim_scrnaseq_data` is
        out of range.

    Examples
    --------
    >>> from structboost import sim_scrnaseq_anndata
    >>> adata = sim_scrnaseq_anndata(n=100, n_genes=20, stageno=4, seed=0)
    >>> adata.obs["stage"].nunique()
    4
    """
    try:
        import anndata as ad
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised only without anndata
        raise ImportError(
            "anndata is required for sim_scrnaseq_anndata. "
            "Install with: pip install structboost[bae]"
        ) from exc

    if target_sum <= 0:
        raise ValueError(f"target_sum must be > 0, got {target_sum}")

    result = sim_scrnaseq_data(
        n=n,
        n_genes=n_genes,
        stageno=stageno,
        stagep=stagep,
        stagen=stagen,
        stageoverlap=stageoverlap,
        hierarchy=hierarchy,
        markers_per_level=markers_per_level,
        imbalanced=imbalanced,
        base_mean=base_mean,
        gene_mean_shape=gene_mean_shape,
        effect_size=effect_size,
        effect_size_sd=effect_size_sd,
        dispersion=dispersion,
        lib_size_sd=lib_size_sd,
        n_batches=n_batches,
        batch_effect_sd=batch_effect_sd,
        ambient_frac=ambient_frac,
        dropout_mid=dropout_mid,
        dropout_shape=dropout_shape,
        seed=seed,
    )

    counts = result.counts
    totals = counts.sum(axis=1, keepdims=True).astype(np.float64)
    lognorm = np.log1p(counts / np.maximum(totals, 1.0) * target_sum).astype(np.float32)

    adata = ad.AnnData(lognorm.copy())
    adata.obs_names = [f"Cell_{i}" for i in range(n)]
    adata.var_names = [f"Gene_{j}" for j in range(counts.shape[1])]
    adata.layers["counts"] = counts
    adata.layers["lognorm"] = lognorm

    adata.obs["stage"] = pd.Categorical(
        ["Background" if label < 0 else f"Stage_{label}" for label in result.stage_labels]
    )
    adata.obs["stage_id"] = result.stage_labels.astype(np.int32)
    if result.hierarchy is not None:
        # One categorical per hierarchy level, so subgroup detection can be scored
        # at every resolution the taxonomy defines rather than only at the leaves.
        # Labels carry the full ancestry path: the within-parent index alone would
        # collide across parents, silently merging distinct subgroups.
        for depth in range(result.leaf_paths.shape[1]):
            labels = []
            for stage in result.stage_labels:
                if stage < 0:
                    labels.append("Background")
                else:
                    path = result.leaf_paths[stage, : depth + 1]
                    joined = "-".join(str(int(v)) for v in path)
                    labels.append(f"L{depth}_{joined}")
            adata.obs[f"level_{depth}"] = pd.Categorical(labels)

    adata.obs["batch"] = pd.Categorical([f"Batch_{g}" for g in result.batch_labels])
    adata.obs["size_factor"] = result.size_factors.astype(np.float32)
    adata.obs["total_counts"] = totals.ravel().astype(np.float32)

    mask = result.marker_mask
    adata.var["base_mean"] = result.gene_means.astype(np.float32)
    adata.var["is_marker"] = mask.any(axis=0)
    adata.var["marker_level"] = result.gene_level.astype(np.int32)
    adata.var["n_marker_stages"] = mask.sum(axis=0).astype(np.int32)
    adata.var["marker_stages"] = [
        ",".join(f"Stage_{k}" for k in np.flatnonzero(mask[:, j])) for j in range(mask.shape[1])
    ]
    adata.varm["marker_mask"] = mask.T.copy()

    if standardize:
        mean = lognorm.mean(axis=0, keepdims=True)
        std = lognorm.std(axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)  # constant genes stay all-zero
        adata.X = ((lognorm - mean) / std).astype(np.float32)

    adata.uns["simulation"] = {
        **result.params,
        "stage_sizes": np.asarray(result.stage_sizes, dtype=np.int64),
        "target_sum": float(target_sum),
        "standardize": bool(standardize),
        "x_content": "zscore" if standardize else "lognorm",
    }
    return adata
