"""Annotation helpers for BAE latent dimension interpretation.

Provides extraction of ranked gene lists per latent dimension from fitted
AnnData objects and writing functional annotations back to h5ad files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GeneRanking:
    """Ranked gene list with weights.

    Attributes
    ----------
    gene_names
        Gene names ordered by descending magnitude.
    weights
        Corresponding encoder weights (same order as gene_names).
    """

    gene_names: list[str]
    weights: list[float]


@dataclass
class DimensionGeneRanking:
    """Ranked gene lists (positive and negative) for a single latent dimension.

    Attributes
    ----------
    dimension
        Index of the latent dimension.
    n_selected_genes
        Total number of nonzero genes in this dimension.
    positive_genes
        Top-k positively weighted genes, ranked by weight descending.
    negative_genes
        Top-k negatively weighted genes, ranked by absolute weight descending.
    """

    dimension: int
    n_selected_genes: int
    positive_genes: GeneRanking
    negative_genes: GeneRanking


@dataclass
class DimensionAnnotation:
    """Functional annotation for a single latent dimension.

    Attributes
    ----------
    dimension
        Index of the latent dimension.
    positive_annotation
        Functional description of the positive gene program.
    negative_annotation
        Functional description of the negative gene program.
    overall_annotation
        Overall biological axis / cell type assignment for this dimension.
    database_references
        Relevant database URLs supporting the annotation.
    """

    dimension: int
    positive_annotation: str
    negative_annotation: str
    overall_annotation: str
    database_references: list[str] = field(default_factory=list)


def extract_gene_rankings(
    adata_path: str | Path,
    model_key: str = "BAE_encoder_weights",
    dimensions: list[int] | None = None,
    top_k: int = 50,
) -> list[DimensionGeneRanking]:
    """Extract ranked gene lists (pos/neg) per latent dimension from .h5ad.

    For each dimension, genes are split by sign of their encoder weight.
    Within each group, genes are ranked by magnitude (descending).
    At most ``min(top_k, n_available)`` genes are returned per group.

    Parameters
    ----------
    adata_path
        Path to a .h5ad file containing fitted BAE results.
    model_key
        Key in ``adata.varm`` holding the encoder weight matrix,
        e.g. ``"BAE_encoder_weights"``.
    dimensions
        Indices of latent dimensions to extract. ``None`` extracts all.
    top_k
        Maximum number of genes per sign group (positive/negative).
        Clamped to ``min(top_k, n_available)`` when fewer genes exist.

    Returns
    -------
    list[DimensionGeneRanking]
        Ranked gene lists for each requested dimension.

    Raises
    ------
    FileNotFoundError
        If ``adata_path`` does not exist.
    KeyError
        If ``model_key`` is not found in ``adata.varm``.
    """
    import anndata as ad
    import numpy as np

    adata_path = Path(adata_path)
    if not adata_path.exists():
        raise FileNotFoundError(f"File not found: {adata_path}")

    adata = ad.read_h5ad(adata_path)

    if model_key not in adata.varm:
        raise KeyError(
            f"{model_key!r} not found in adata.varm. Available keys: {list(adata.varm.keys())}"
        )

    W = np.asarray(adata.varm[model_key])  # (n_genes, latent_dim)
    gene_names = list(adata.var_names)
    _n_genes, latent_dim = W.shape

    if dimensions is None:
        dimensions = list(range(latent_dim))

    rankings: list[DimensionGeneRanking] = []
    for dim in dimensions:
        w = W[:, dim]

        # Positive genes: w > 0, sorted by weight descending
        pos_mask = w > 0
        pos_indices = np.where(pos_mask)[0]
        pos_order = np.argsort(-w[pos_indices])
        pos_indices = pos_indices[pos_order][:top_k]
        pos_names = [gene_names[i] for i in pos_indices]
        pos_weights = [float(w[i]) for i in pos_indices]

        # Negative genes: w < 0, sorted by |weight| descending
        neg_mask = w < 0
        neg_indices = np.where(neg_mask)[0]
        neg_order = np.argsort(w[neg_indices])  # most negative first
        neg_indices = neg_indices[neg_order][:top_k]
        neg_names = [gene_names[i] for i in neg_indices]
        neg_weights = [float(w[i]) for i in neg_indices]

        n_selected = int(pos_mask.sum() + neg_mask.sum())

        rankings.append(
            DimensionGeneRanking(
                dimension=dim,
                n_selected_genes=n_selected,
                positive_genes=GeneRanking(gene_names=pos_names, weights=pos_weights),
                negative_genes=GeneRanking(gene_names=neg_names, weights=neg_weights),
            )
        )

    return rankings


_KNOWN_MODEL_KEYS = ("BAE_encoder_weights",)


def detect_model_keys(adata_path: str | Path) -> list[str]:
    """Detect which BAE encoder weight keys are present in a .h5ad file.

    Parameters
    ----------
    adata_path
        Path to a .h5ad file.

    Returns
    -------
    list[str]
        Model keys found in ``adata.varm`` (subset of
        ``["BAE_encoder_weights"]``).
    """
    import anndata as ad

    adata = ad.read_h5ad(Path(adata_path))
    return [k for k in _KNOWN_MODEL_KEYS if k in adata.varm]


def extract_metadata(adata_path: str | Path) -> dict[str, str | None]:
    """Extract biological metadata from adata.uns if present.

    Looks for common keys: ``organism``, ``tissue``, ``species``, ``organ``,
    ``experiment``. Falls back to ``"species"`` for ``"organism"`` and
    ``"organ"`` for ``"tissue"`` if the primary keys are absent.

    Parameters
    ----------
    adata_path
        Path to a .h5ad file.

    Returns
    -------
    dict[str, str | None]
        Keys: ``"organism"``, ``"tissue"``, ``"experiment"``. Values are
        ``None`` if not found.
    """
    import anndata as ad

    adata = ad.read_h5ad(Path(adata_path))
    uns = adata.uns

    organism = uns.get("organism") or uns.get("species")
    tissue = uns.get("tissue") or uns.get("organ")
    experiment = uns.get("experiment")

    return {
        "organism": str(organism) if organism is not None else None,
        "tissue": str(tissue) if tissue is not None else None,
        "experiment": str(experiment) if experiment is not None else None,
    }


def write_annotations_to_h5ad(
    adata_path: str | Path,
    annotations: list[DimensionAnnotation],
) -> None:
    """Write functional annotations into adata.uns of a .h5ad file.

    Annotations are stored under ``adata.uns["bae_dimension_annotations"]``
    as a dict keyed by dimension index (string).

    .. warning::
       The file is read, mutated and written back over the same path. Keep a copy
       if the original matters.

    See Also
    --------
    extract_gene_rankings : Produce the ranked gene lists to annotate.

    Parameters
    ----------
    adata_path
        Path to a .h5ad file.
    annotations
        List of :class:`DimensionAnnotation` objects to write.
    """
    import anndata as ad

    adata_path = Path(adata_path)
    adata = ad.read_h5ad(adata_path)

    uns_key = "bae_dimension_annotations"

    annotation_dict: dict[str, dict[str, str | list[str]]] = {}
    for ann in annotations:
        annotation_dict[str(ann.dimension)] = {
            "positive_annotation": ann.positive_annotation,
            "negative_annotation": ann.negative_annotation,
            "overall_annotation": ann.overall_annotation,
            "database_references": ann.database_references,
        }

    adata.uns[uns_key] = annotation_dict
    adata.write_h5ad(adata_path)
