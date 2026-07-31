# The standardization contract

**BAE expects `adata.X` to be z-scored per gene: mean ≈ 0, standard deviation ≈ 1.**

The usual scanpy path gets you there:

```python
import scanpy as sc

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
sc.pp.scale(adata)          # <- this is the one that matters
```

{func}`~structboost.sim_scrnaseq_anndata` produces standardized `X` by default.

## What happens if you skip it

`fit` checks and warns. It does not refuse:

```
UserWarning: Input data does not appear to be standardized (mean≈0, std≈1).
Standardization is recommended for optimal performance.
```

The check is a heuristic on the matrix being fitted: the median absolute column
mean must be below 0.1, and the median column standard deviation within 0.1 of 1.

Ignoring the warning degrades the method in a specific way. Boosting's selection
criterion is scale-invariant in the *predictors*, so it will not simply pick the
highest-expressed genes, but the reconstruction loss is not scale-invariant, so
high-variance genes dominate the target gradient, and every reported number stops
being interpretable (see {doc}`reading-quality`).

## The second job this contract does

The check is not only a warning. Its result decides **whether a PCA warm start
centers the data**.

`init_pca=True` runs a PCA to initialize the latent code. If the input is already
z-scored, centering it again would be a second transformation on top of one you
already applied. So the warm start centers only when the data does *not* look
standardized.

Columns are never rescaled by the warm start, in either case. Rescaling would
silently apply a second standardization, and on data that is genuinely not
standardized, it would mask the very condition the warning exists to report.

## Why an MSE of 0.85 is not good

On z-scored input, predicting zero everywhere scores an MSE of exactly **1.0**.
That is the entire reason the contract earns its keep: it fixes the scale so that
a loss can be read at all.

So an MSE of 0.85 is not "close to perfect". It is 15% of variance explained.
{doc}`reading-quality` covers what to compare it against instead.

## If your data lives in a layer

Pass `layer=`. The contract applies to whatever matrix is read:

```python
adata.layers["scaled"] = sc.pp.scale(adata, copy=True).X
model.fit(adata, layer="scaled")
```

The layer is remembered, so `transform`, `reconstruct` and the diagnostics all
default to it afterwards, so a fitted model always reads the representation it
learned on. See {doc}`anndata-contract`.
