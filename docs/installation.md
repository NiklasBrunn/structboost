# Installation

Requires Python 3.10 or newer.

## Extras

The core package depends on NumPy alone. Everything heavier is opt-in, and the
imports are lazy. Importing `structboost` does not import torch.

| Extra | Install | Brings in | Needed for |
| --- | --- | --- | --- |
| *(none)* | `pip install structboost` | numpy | {func}`~structboost.allboost`, {func}`~structboost.stability_selection` |
| `bae` | `pip install "structboost[bae]"` | torch, anndata, scipy, pandas, tqdm | {class}`~structboost.BAE`, the simulator, everything AnnData |
| `plot` | `pip install "structboost[plot]"` | matplotlib | the `plot_*` functions |
| `io` | `pip install "structboost[io]"` | pyarrow | Parquet encoder-weight files |

Most users of the BAE want:

```bash
pip install "structboost[bae,plot]"
```

:::{note}
Pre-releases are published to **TestPyPI only**, so a version such as `0.6.0rc1`
never occupies a number on PyPI. To install one:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ "structboost[bae]"
```

The `--extra-index-url` is required: TestPyPI does not carry the dependencies, so
pip needs PyPI to resolve torch and the rest.
:::

## From source

```bash
git clone https://github.com/NiklasBrunn/structboost
cd structboost
pip install -e ".[bae,plot,io]"
```

For development, add the tooling extras and install the hooks:

```bash
pip install -e ".[bae,plot,io,dev,test,docs]"
pre-commit install
```

See {doc}`contributing` for the full development workflow.

## Verifying the install

```python
import structboost

print(structboost.__version__)

# Simulate data with known marker genes and fit a small model end to end.
from structboost import BAE, BAEConfig, sim_scrnaseq_anndata

adata = sim_scrnaseq_anndata(n=300, n_genes=200, stageno=4, seed=0)
model = BAE(adata.n_vars, BAEConfig(latent_dim=4, max_iterations=10))
model.fit(adata, verbose=False)

print(adata.obsm["X_bae"].shape)                    # (300, 4)
print((adata.varm["BAE_encoder_weights"] != 0).sum())  # a small number
```

If the last line prints a number close to `200 * 4`, the encoder is not sparse
and something is wrong. See {doc}`guide/concepts/standardization`, since the
usual cause is input that was never z-scored.

## GPU

Set `device` on the config:

```python
BAEConfig(latent_dim=10, device="cuda")
```

Only the decoder's gradient steps run on the accelerator. The boosting fit that
selects genes is NumPy on the CPU, so the speed-up is bounded by how much of your
runtime the decoder accounts for. `BAE.load` warns and falls back to CPU when a
checkpoint names a device the machine does not have.
