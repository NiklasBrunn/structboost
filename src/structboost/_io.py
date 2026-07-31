"""Read and write BAE encoder weight matrices as standalone files.

An encoder weight matrix is the transferable, interpretable product of a BAE fit:
``(n_genes, latent_dim)`` sparse gene loadings. Shipping it to another dataset
requires the gene identifiers to travel with it, which a bare array cannot do.

Parquet is the recommended format. Beyond preserving dtypes and the index, it is
immune to the spreadsheet round-trip that silently rewrites gene symbols such as
``SEPT2``, ``MARCH1`` and ``DEC1`` as dates — found in roughly a fifth of papers
carrying Excel supplements [1]_. CSV/TSV is accepted on read for interoperability
but is not the recommended write format.

Two identifier columns are carried:

``gene_id``
    Ensembl accession (``ENSG…``, ``ENSMUSG…``). Stable across annotation
    releases and unambiguous, so it is the default join key.
``gene_symbol``
    Human-readable label. Carried but never authoritative: symbols are revised
    between releases and collide through aliases and species case conventions.

References
----------
.. [1] Ziemann, M., Eren, Y. & El-Osta, A. (2016). Gene name errors are
   widespread in the scientific literature. *Genome Biology* 17, 177.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

#: Ensembl gene accessions: ``ENS`` + optional species code + ``G`` + digits,
#: with an optional ``.version`` suffix (``ENSG00000141510.17``).
_ENSEMBL_GENE_RE = re.compile(r"^ENS[A-Z]{0,6}G\d{5,}(\.\d+)?$")

#: Column name prefix for the latent dimensions in the on-disk wide layout.
DIM_PREFIX = "dim_"

_PARQUET_SUFFIXES = frozenset({".parquet", ".pq"})
_TEXT_SUFFIXES = frozenset({".csv", ".tsv", ".txt"})


def looks_like_ensembl(identifiers: object, *, min_fraction: float = 0.5) -> bool:
    """Whether a set of gene identifiers is predominantly Ensembl accessions.

    Parameters
    ----------
    identifiers
        Iterable of gene identifiers.
    min_fraction
        Fraction that must match the Ensembl pattern. Defaults to 0.5 rather than
        1.0 because real panels carry a tail of spike-ins and custom features
        that never match, and rejecting a whole panel over those would be wrong.

    Returns
    -------
    True if at least ``min_fraction`` of the identifiers are Ensembl accessions.
    """
    values = [str(value) for value in np.asarray(identifiers, dtype=object).ravel()]
    if not values:
        return False
    hits = sum(1 for value in values if _ENSEMBL_GENE_RE.match(value))
    return hits / len(values) >= min_fraction


def _resolve_format(path: Path, fmt: str | None) -> str:
    if fmt is not None:
        if fmt not in ("parquet", "text"):
            raise ValueError(f"format must be 'parquet' or 'text', got {fmt!r}")
        return fmt
    suffixes = [s.lower() for s in path.suffixes]
    # `.csv.gz` and friends: the compression suffix is not the format.
    meaningful = [s for s in suffixes if s not in (".gz", ".bz2", ".xz", ".zst")]
    suffix = meaningful[-1] if meaningful else ""
    if suffix in _PARQUET_SUFFIXES:
        return "parquet"
    if suffix in _TEXT_SUFFIXES:
        return "text"
    raise ValueError(
        f"Cannot infer format from {path.name!r}. Expected one of "
        f"{sorted(_PARQUET_SUFFIXES | _TEXT_SUFFIXES)}, or pass format= explicitly."
    )


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def write_encoder_weights(
    weights: NDArray[np.floating] | pd.DataFrame,
    path: str | Path,
    *,
    gene_ids: object | None = None,
    gene_symbols: object | None = None,
    metadata: dict[str, Any] | None = None,
    drop_zero_rows: bool = False,
    format: str | None = None,
) -> Path:
    """Write an encoder weight matrix with its gene identifiers.

    Parameters
    ----------
    weights
        Array of shape ``(n_genes, latent_dim)``, or a DataFrame indexed by gene
        identifier whose remaining columns are the latent dimensions.
    path
        Destination. Format is inferred from the suffix unless ``format`` is given.
    gene_ids
        Ensembl accessions, length ``n_genes``. The join key on read.
    gene_symbols
        Gene symbols, length ``n_genes``. Carried as a label only.
    metadata
        Provenance recorded with the matrix. ``structboost_version`` and
        ``latent_dim`` are filled in automatically. Callers should add the
        species and **annotation release**: without the release the Ensembl
        accessions are only probably joinable to another dataset.
    drop_zero_rows
        Omit genes whose weights are zero in every dimension. Safe for the
        coverage guard (a zero weight contributes nothing to either side of the
        ratio), but the file then no longer distinguishes the reference *panel*
        from the reference *support*, so the full panel is stored in the metadata.
    format
        ``"parquet"`` or ``"text"``. Inferred from the suffix when omitted.

    Returns
    -------
    The path written.
    """
    import pandas as pd

    path = Path(path)
    resolved = _resolve_format(path, format)

    if isinstance(weights, pd.DataFrame):
        frame = weights.copy()
        if gene_ids is None and frame.index.name in ("gene_id", None):
            gene_ids = frame.index.to_numpy()
        matrix = frame.to_numpy(dtype=np.float64)
    else:
        matrix = np.asarray(weights, dtype=np.float64)

    if matrix.ndim != 2:
        raise ValueError(f"weights must be 2-D (n_genes, latent_dim), got {matrix.shape}")
    n_genes, latent_dim = matrix.shape

    def _check(name: str, values: object) -> NDArray[np.object_] | None:
        if values is None:
            return None
        array = np.asarray(values, dtype=object).ravel()
        if array.shape[0] != n_genes:
            raise ValueError(f"{name} has length {array.shape[0]} but weights has {n_genes} rows")
        return array

    ids = _check("gene_ids", gene_ids)
    symbols = _check("gene_symbols", gene_symbols)
    if ids is None and symbols is None:
        raise ValueError(
            "At least one of gene_ids or gene_symbols is required. An encoder "
            "weight matrix without gene identifiers cannot be aligned to another "
            "dataset, which is the only reason to write one out."
        )

    full_panel = [str(v) for v in (ids if ids is not None else symbols)]

    columns: dict[str, Any] = {}
    if ids is not None:
        columns["gene_id"] = [str(v) for v in ids]
    if symbols is not None:
        columns["gene_symbol"] = [str(v) for v in symbols]
    for j in range(latent_dim):
        columns[f"{DIM_PREFIX}{j}"] = matrix[:, j]
    frame = pd.DataFrame(columns)

    meta: dict[str, Any] = dict(metadata or {})
    meta.setdefault("latent_dim", int(latent_dim))
    meta.setdefault("n_genes_written", int(n_genes))
    try:
        from . import __version__

        meta.setdefault("structboost_version", __version__)
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        pass

    if drop_zero_rows:
        keep = (matrix != 0).any(axis=1)
        frame = frame.loc[keep].reset_index(drop=True)
        meta["reference_panel"] = full_panel
        meta["n_genes_written"] = int(keep.sum())

    path.parent.mkdir(parents=True, exist_ok=True)
    if resolved == "parquet":
        _write_parquet(frame, path, meta)
    else:
        sep = "\t" if ".tsv" in [s.lower() for s in path.suffixes] else ","
        frame.to_csv(path, index=False, sep=sep)
        _sidecar_path(path).write_text(json.dumps(meta, indent=2, default=str))
    return path


def _write_parquet(frame: pd.DataFrame, path: Path, meta: dict[str, Any]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Writing Parquet needs pyarrow. Install it with "
            "`pip install 'structboost[io]'`, or write a .csv instead."
        ) from exc

    table = pa.Table.from_pandas(frame, preserve_index=False)
    existing = table.schema.metadata or {}
    encoded = {
        **existing,
        b"structboost": json.dumps(meta, default=str).encode("utf-8"),
    }
    pq.write_table(table.replace_schema_metadata(encoded), path)


def read_encoder_weights(
    path: str | Path,
    *,
    format: str | None = None,
) -> pd.DataFrame:
    """Read an encoder weight matrix written by :func:`structboost.write_encoder_weights`.

    Parameters
    ----------
    path
        Source file. Format is inferred from the suffix unless ``format`` is given.
    format
        ``"parquet"`` or ``"text"``. Inferred from the suffix when omitted.

    Returns
    -------
    DataFrame of shape ``(n_genes, latent_dim)`` indexed by the join identifier —
    ``gene_id`` when the file carries Ensembl accessions, otherwise
    ``gene_symbol``. Any symbol column is preserved in ``frame.attrs["gene_symbol"]``
    rather than as a data column, so every remaining column is a latent dimension.
    Provenance is available in ``frame.attrs["metadata"]`` and the join key used in
    ``frame.attrs["join_key"]``.
    """
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No encoder weight file at {path}")
    resolved = _resolve_format(path, format)

    meta: dict[str, Any] = {}
    if resolved == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "Reading Parquet needs pyarrow. Install it with `pip install 'structboost[io]'`."
            ) from exc
        table = pq.read_table(path)
        raw = (table.schema.metadata or {}).get(b"structboost")
        if raw is not None:
            meta = json.loads(raw.decode("utf-8"))
        frame = table.to_pandas()
    else:
        sep = "\t" if ".tsv" in [s.lower() for s in path.suffixes] else ","
        frame = pd.read_csv(path, sep=sep)
        sidecar = _sidecar_path(path)
        if sidecar.exists():
            meta = json.loads(sidecar.read_text())

    dim_columns = [c for c in frame.columns if str(c).startswith(DIM_PREFIX)]
    if not dim_columns:
        raise ValueError(
            f"{path.name} has no latent-dimension columns (expected names starting "
            f"with {DIM_PREFIX!r}). Columns found: {list(frame.columns)}"
        )
    # Sort numerically: dim_10 must not order before dim_2.
    dim_columns.sort(key=lambda c: int(str(c)[len(DIM_PREFIX) :]))

    has_id = "gene_id" in frame.columns
    has_symbol = "gene_symbol" in frame.columns
    if not has_id and not has_symbol:
        raise ValueError(
            f"{path.name} carries no gene_id or gene_symbol column, so its weights "
            "cannot be aligned to a dataset."
        )
    join_key = "gene_id" if has_id else "gene_symbol"

    out = frame[dim_columns].astype(np.float64)
    out.index = pd.Index(frame[join_key].astype(str), name=join_key)
    out.attrs["metadata"] = meta
    out.attrs["join_key"] = join_key
    if has_symbol:
        out.attrs["gene_symbol"] = frame["gene_symbol"].astype(str).to_numpy()
    return out
