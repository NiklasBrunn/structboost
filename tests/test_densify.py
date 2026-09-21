"""Contracts for :func:`structboost._utils.densify`.

The helper exists to cut peak memory, so the thing worth pinning is that it buys
that without moving a single bit: every expression matrix a fit reads goes
through it, and a changed byte there changes the fitted model. These tests
therefore compare raw bytes rather than using ``allclose``.

They also pin the property the helper introduced on top of that: a fit no longer
depends on which sparse format ``adata.X`` happens to be stored in.
"""

import numpy as np
import pytest

from structboost._utils import densify

sp = pytest.importorskip("scipy.sparse")

#: Every sparse layout an AnnData is realistically carrying, plus both dtypes.
FORMATS = ("csr", "csc", "coo")
DTYPES = (np.float32, np.float64)


def _reference(matrix, dtype=np.float32):
    """What the call sites did before ``densify`` existed."""
    dense = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    return dense.astype(dtype)


@pytest.fixture
def matrix():
    rng = np.random.default_rng(11)
    x = rng.standard_normal((257, 83))
    # Zeros are what makes a sparse format worth using, and they are also the
    # entries `toarray` writes rather than scatters.
    x[rng.random(x.shape) < 0.6] = 0.0
    return x


def _as_sparse(x, fmt, dtype):
    return getattr(sp, f"{fmt}_matrix")(x.astype(dtype))


class TestByteIdentity:
    @pytest.mark.parametrize("fmt", FORMATS)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_matches_unchunked_conversion_exactly(self, matrix, fmt, dtype):
        sparse = _as_sparse(matrix, fmt, dtype)
        assert densify(sparse).tobytes() == _reference(sparse).tobytes()

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_dense_input_matches(self, matrix, dtype):
        dense = matrix.astype(dtype)
        assert densify(dense).tobytes() == _reference(dense).tobytes()

    @pytest.mark.parametrize("block_bytes", [1, 64, 1024, 1 << 20])
    def test_block_size_never_changes_the_bytes(self, matrix, block_bytes):
        """The block loop is a pure partition of the rows: no reassociation, so
        every block size -- including one row at a time -- gives the same array."""
        sparse = _as_sparse(matrix, "csr", np.float64)
        whole = densify(sparse)
        assert densify(sparse, block_bytes=block_bytes).tobytes() == whole.tobytes()

    def test_block_size_below_one_row_still_works(self, matrix):
        """A panel wider than the block budget must not produce a zero-row step."""
        sparse = _as_sparse(matrix, "csr", np.float64)
        assert densify(sparse, block_bytes=0).tobytes() == densify(sparse).tobytes()


class TestWritingIntoAView:
    """``out=`` is what lets the boosting design be one allocation.

    ``fit`` builds ``(n_cells, n_genes + n_nuisance)`` up front and reads the
    panel into its gene block, instead of densifying separately and copying the
    result in with ``hstack``.
    """

    @pytest.mark.parametrize("fmt", (*FORMATS, "dense"))
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_column_block_of_a_wider_array(self, matrix, fmt, dtype):
        payload = matrix.astype(dtype) if fmt == "dense" else _as_sparse(matrix, fmt, dtype)
        n_rows, n_cols = matrix.shape
        design = np.empty((n_rows, n_cols + 3), dtype=np.float32)
        block = densify(payload, out=design[:, :n_cols])

        assert block.base is design
        assert not block.flags["C_CONTIGUOUS"]  # a strided view, by construction
        # Same values as the allocating form, which is what keeps a fit identical.
        assert block.tobytes() == _reference(payload).tobytes()

    def test_nuisance_columns_are_untouched(self, matrix):
        n_rows, n_cols = matrix.shape
        design = np.zeros((n_rows, n_cols + 2), dtype=np.float32)
        design[:, n_cols:] = 7.5
        densify(_as_sparse(matrix, "csr", np.float64), out=design[:, :n_cols])
        assert (design[:, n_cols:] == 7.5).all()

    def test_out_dtype_wins_over_dtype_argument(self, matrix):
        out = np.empty(matrix.shape, dtype=np.float64)
        assert densify(_as_sparse(matrix, "csr", np.float64), dtype=np.float32, out=out) is out
        assert out.dtype == np.float64

    def test_shape_mismatch_is_rejected(self, matrix):
        """Silently filling the wrong block would corrupt the design matrix."""
        sparse = _as_sparse(matrix, "csr", np.float64)
        with pytest.raises(ValueError, match="out must have shape"):
            densify(sparse, out=np.empty((matrix.shape[0], matrix.shape[1] + 1), np.float32))

    @pytest.mark.parametrize("block_bytes", [1, 4096, 1 << 20])
    def test_block_size_never_changes_the_bytes(self, matrix, block_bytes):
        sparse = _as_sparse(matrix, "csr", np.float64)
        n_rows, n_cols = matrix.shape
        design = np.empty((n_rows, n_cols + 3), dtype=np.float32)
        block = densify(sparse, out=design[:, :n_cols], block_bytes=block_bytes)
        assert block.tobytes() == densify(sparse).tobytes()

    def test_torch_reads_the_block_without_copying_it(self, matrix):
        """The assumption the single-allocation design rests on.

        The gene block has a row stride of ``n_genes + n_nuisance``, which BLAS
        takes as a leading dimension, so the encoder's matmul runs on it directly.
        If torch ever materialized a contiguous copy instead, the second full
        panel this construction exists to avoid would silently come back.
        """
        torch = pytest.importorskip("torch")
        n_rows, n_cols = matrix.shape
        design = np.empty((n_rows, n_cols + 4), dtype=np.float32)
        block = densify(_as_sparse(matrix, "csr", np.float64), out=design[:, :n_cols])

        strided = torch.from_numpy(block)
        contiguous = torch.from_numpy(np.ascontiguousarray(block))
        assert not strided.is_contiguous()

        torch.manual_seed(0)
        layer = torch.nn.Linear(n_cols, 3, bias=False)
        with torch.no_grad():
            from_view, from_copy = layer(strided), layer(contiguous)
        assert from_view.numpy().tobytes() == from_copy.numpy().tobytes()


class TestOutputContract:
    @pytest.mark.parametrize("fmt", FORMATS)
    def test_sparse_input_is_c_contiguous(self, matrix, fmt):
        """Row-major whatever the source format.

        ``csc.toarray()`` returns a Fortran-ordered array, which made the fitted
        model depend on the storage format (different BLAS tiling) and handed
        torch a non-contiguous training tensor. One layout for every input.
        """
        result = densify(_as_sparse(matrix, fmt, np.float64))
        assert result.flags["C_CONTIGUOUS"]

    @pytest.mark.parametrize("fmt", FORMATS)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_dtype_and_shape(self, matrix, fmt, dtype):
        result = densify(_as_sparse(matrix, fmt, dtype))
        assert result.dtype == np.float32
        assert result.shape == matrix.shape

    def test_honours_requested_dtype(self, matrix):
        assert densify(_as_sparse(matrix, "csr", np.float64), dtype=np.float64).dtype == np.float64

    def test_matching_dense_input_is_not_copied(self, matrix):
        """`np.asarray` semantics: no needless duplicate of an already-dense panel."""
        dense = matrix.astype(np.float32)
        assert densify(dense) is dense

    @pytest.mark.parametrize("shape", [(0, 5), (5, 0), (0, 0)])
    def test_empty(self, shape):
        result = densify(sp.csr_matrix(np.zeros(shape)))
        assert result.shape == shape
        assert result.dtype == np.float32


class TestStorageFormatIndependence:
    """The same data must give the same model however AnnData stored it."""

    @staticmethod
    def _fit_weights(x, fmt):
        from anndata import AnnData

        from structboost import BAE, BAEConfig

        # AnnData stores CSR and CSC only, which is also the whole population of
        # formats this property can be violated by.
        payload = x if fmt == "dense" else getattr(sp, f"{fmt}_matrix")(x)
        adata = AnnData(X=payload)
        model = BAE(adata.n_vars, BAEConfig(latent_dim=3, max_iterations=6, seed=5, batch_size=64))
        model.fit(adata, verbose=False)
        return model.get_encoder_weights()

    def test_csr_csc_dense_agree_bitwise(self):
        """Regression test: ``csc.toarray()`` is Fortran-ordered, so before
        ``densify`` the storage format changed the BLAS reduction order and with
        it the fitted weights, by ~2e-9 on a short fit."""
        pytest.importorskip("torch")
        pytest.importorskip("anndata")
        rng = np.random.default_rng(3)
        x = rng.standard_normal((120, 40))
        x = (x - x.mean(axis=0)) / x.std(axis=0)
        reference = self._fit_weights(x, "csr")
        for fmt in ("csc", "dense"):
            assert self._fit_weights(x, fmt).tobytes() == reference.tobytes(), fmt
