"""structboost - A scverse-ecosystem package for Boosting Autoencoders.

This module keeps imports lightweight by lazily importing optional / heavy
dependencies (notably `torch`) only when needed.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from ._annotation import (
    DimensionAnnotation,
    DimensionGeneRanking,
    GeneRanking,
    extract_gene_rankings,
    write_annotations_to_h5ad,
)
from ._boosting import AllboostHistory, allboost
from ._io import looks_like_ensembl, read_encoder_weights, write_encoder_weights
from ._simulation import SimulationResult, sim_scrnaseq_anndata, sim_scrnaseq_data
from ._stability import StabilitySelectionResult, stability_selection
from ._types import BAEConfig, TrainingReport
from ._utils import (
    ObsCovariateEncoding,
    compute_covariance_cache,
    encode_obs_covariates,
    linear_ceiling,
    transform_obs_covariates,
)

if TYPE_CHECKING:  # pragma: no cover
    # For type checkers only (avoids importing torch at runtime)
    from ._model import BAE as BAE

try:
    __version__ = version("structboost")
except PackageNotFoundError:  # pragma: no cover
    # Fallback for editable checkouts where metadata is not available.
    __version__ = "0.1.0"

__all__ = [
    "AllboostHistory",
    "BAE",
    "BAEConfig",
    "DimensionAnnotation",
    "DimensionGeneRanking",
    "GeneRanking",
    "ObsCovariateEncoding",
    "SimulationResult",
    "StabilitySelectionResult",
    "TrainingReport",
    "allboost",
    "compute_covariance_cache",
    "encode_obs_covariates",
    "export_interactive_html",
    "extract_gene_rankings",
    "linear_ceiling",
    "looks_like_ensembl",
    "plot_boosting_coefficient_paths",
    "plot_top_boosting_coefficients",
    "plot_training_diagnostics",
    "read_encoder_weights",
    "sim_scrnaseq_anndata",
    "sim_scrnaseq_data",
    "stability_selection",
    "transform_obs_covariates",
    "write_annotations_to_h5ad",
    "write_encoder_weights",
]

_PLOT_FUNCTIONS = frozenset(
    {
        "plot_boosting_coefficient_paths",
        "plot_top_boosting_coefficients",
        "plot_training_diagnostics",
    }
)


def __getattr__(name: str) -> Any:  # PEP 562
    if name == "BAE":
        from ._model import BAE as _BAE

        return _BAE
    if name == "export_interactive_html":
        from ._explorer import export_interactive_html as _export_interactive_html

        return _export_interactive_html
    if name in _PLOT_FUNCTIONS:
        from . import _plotting

        return getattr(_plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
