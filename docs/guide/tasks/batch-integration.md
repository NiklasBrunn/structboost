# Batches and unwanted covariates

Name the covariate once. The mode decides which of the two mechanisms act on it.

```python
model.fit(adata, batch_key="batch")                              # both, the default
model.fit(adata, batch_key="batch", batch_integration_mode="decoder")
model.fit(adata, batch_key=["batch", "donor"])                    # several columns
```

No `batch_key` means no integration. Naming a mode without a `batch_key` raises,
rather than quietly integrating nothing.

## The two mechanisms

`"decoder"`
: The encoded covariate is concatenated to the decoder input. The decoder can
  then explain covariate-driven variation directly, so the latent code does not
  have to carry it.

`"encoder"`
: The encoded covariate is added to the boosting design as a mandatory
  regressor, so a covariate-correlated gene is not selected *because of* the
  covariate. Its coefficients are kept for diagnostics in
  `adata.uns["bae"]["batch_weights"]` and excluded from the encoder, which stays
  gene-only.

`"both"` (default)
: Both. This is the usual choice, and the one the measurements below describe.

:::{important}
**The covariate is never an encoder input.** `"encoder"` names the half of the
model the mechanism protects, not a tensor it is fed to.

{meth}`~structboost.BAE.transform` is gene-only and needs no covariate labels
under any mode:

```python
latent = model.transform(query_adata)     # no batch column required
```

That is the point. The deployable artifact works on data with no covariate
annotation at all. {meth}`~structboost.BAE.reconstruct` needs the columns only
under `"decoder"` and `"both"`, and rejects levels it did not see when fitting.
:::

## Why the split exists at all

A conditioning-only fit removes covariate signal from the latent, but nothing
stops a batch-correlated gene from being selected in the first place. A
regression-only fit protects selection, but leaves the decoder to account for the
covariate through the latent code. They target different failures, which is why
the default applies both and why each remains separately selectable.

An additive alternative to concatenative conditioning was evaluated and rejected
as strictly dominated. On simulated data with a known batch effect it left *more*
batch variance in the latent than doing nothing: R² 0.38 against a 0.35
uncorrected baseline, versus 0.006 for concatenation.

## If you subset to highly variable genes first

With several batches, prefer `flavor="seurat_v3_paper"`:

```python
sc.pp.highly_variable_genes(
    adata, n_top_genes=2000, flavor="seurat_v3_paper",
    layer="counts", batch_key="batch",
)
```

scanpy's `"seurat_v3"` ranks a gene only in the batches where it registers, so
one variable in a single batch can outrank one variable in all of them. Such a
gene is nearly a batch indicator, and boosting is greedy enough to spend a whole
latent dimension on it. Neither mechanism above stops that: `"encoder"` removes
only the covariate's linear effect from the design.

## Near-collinear covariates

Boosting refits the mandatory block jointly at every step, and that block is the
mandatory genes together with the batch columns. Exactly collinear entries raise.
Merely close-to-collinear ones return a large, sign-unstable answer and raise
nothing, which is the case worth knowing about.

`BAEConfig.nuisance_ridge` is the remedy:

```python
BAEConfig(latent_dim=10, nuisance_ridge=1e-3)
```

It is relative to each encoded column's squared norm, and mandatory *gene*
coefficients stay unpenalized. It is never applied automatically, because ridge
changes the estimates and doing so silently would fit a different model than the
one asked for.

## Did it work?

```python
adata.uns["bae"]["latent_obs_r2_per_dim"]      # integration: near zero is good
adata.uns["bae"]["reconstruction_loss_by_obs"] # fairness: even across groups?
```

These answer different questions and are easy to confuse, see
{doc}`../concepts/anndata-contract`. A single high entry in
`latent_obs_r2_per_dim` is a residual covariate axis worth inspecting rather than
a failure of the whole fit.

Note that `"encoder"` alone targets gene selection rather than the latent, so it
need not move `latent_obs_r2_per_dim` much. Judge it by which genes were
selected.

## Encoding rules

Covariate encoding is strict and fails rather than guessing: missing values,
constant columns and rank-deficient designs all raise. Categorical columns are
dummy-encoded against the first observed level, numeric columns are used as-is,
and everything is then standardized.

The encoding parameters, not the training design matrix, are what a saved model
carries, so `reconstruct` works on new data. See {doc}`persistence`.

## Related

- {func}`~structboost.encode_obs_covariates`, {func}`~structboost.transform_obs_covariates`
- {doc}`../concepts/anndata-contract`
