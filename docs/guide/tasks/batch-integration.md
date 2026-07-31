# Batches and unwanted covariates

Two mechanisms, deliberately independent. Passing one does not give you the
other.

```python
model.fit(
    adata,
    condition_obs=["batch"],   # decoder-side: explains reconstruction variation
    nuisance_obs=["batch"],    # boosting-side: protects gene selection
    nuisance_ridge=1e-3,       # optional, for correlated covariates
    balance_obs="batch",       # optional, equal weight per group
)
```

## `condition_obs`,  decoder conditioning

The encoded covariates are concatenated to the decoder input. The decoder can
then explain batch-driven variation directly, so the latent code does not have to
carry it.

Passing the argument is what enables conditioning. There is no mode to select.
(An additive alternative was evaluated and rejected as strictly dominated: on simulated data with a known batch effect it
left *more* batch variance in the latent than doing nothing: R² 0.38 against a
0.35 uncorrected baseline, versus 0.006 for concatenation.)

The effect on the encoder is indirect: conditioning removes covariate signal from
the latent only insofar as the decoder no longer needs it there.

## `nuisance_obs`,  protecting gene selection

The encoded covariates are appended to the boosting design matrix and marked
mandatory, so the covariate is regressed out *inside* the boosting fit. This is
what stops a batch-correlated gene from being selected because of the batch.

Their coefficients are kept for diagnostics in
`adata.uns["bae"]["nuisance_weights"]` but excluded from the encoder, which stays
gene-only.

`nuisance_ridge` stabilizes the joint mandatory block when covariates are
correlated. It is relative to each encoded column's squared norm, and gene
mandatory coefficients stay unpenalized. The default `0` preserves exact OLS, and
ridge is never applied automatically even when the block is singular, because
ridge changes the estimates.

## Why they are separate

`transform` is gene-only and needs **no** covariate labels:

```python
latent = model.transform(query_adata)     # no batch column required
```

That is the point of keeping conditioning out of the encoder: the deployable
artifact works on data with no covariate annotation at all. `reconstruct`, by
contrast, requires the fitted `condition_obs` columns and rejects levels it did
not see during fitting.

Keeping the two mechanisms separate also means each can be ablated
independently, which is how they were benchmarked in the first place.

## `balance_obs`,  equal weight per group

Inverse-frequency per-cell weights so a large group cannot dominate.

:::{warning}
**The weights do not reach gene selection.** They enter the boosting-target
gradient, the decoder update, the reported losses and the diagnostics. They do
**not** enter the `allboost` fit itself, which remains ordinary unweighted least
squares.

So the targets the encoder chases are balanced, but the projection of those
targets onto genes is not, and **gene selection still leans toward the larger
group**.
:::

Measured on a deliberately imbalanced 500/90/45 design: the spread in per-group
reconstruction MSE fell from 0.231 to 0.150.

Check whether it helped on your data:

```python
adata.uns["bae"]["reconstruction_loss_by_obs"]
```

And check first whether group size actually predicts fit quality, since uneven
per-group reconstruction has causes other than imbalance.

The column must be discrete: at most 50 levels, no missing values, at least two
levels.

## Did it work?

```python
adata.uns["bae"]["latent_obs_r2_per_dim"]      # integration: near zero is good
adata.uns["bae"]["reconstruction_loss_by_obs"] # fairness: even across groups?
```

These answer different questions and are easy to confuse. See
{doc}`../concepts/anndata-contract`. A single high entry in
`latent_obs_r2_per_dim` is a residual covariate axis worth inspecting rather than
a failure of the whole fit.

## Encoding rules

Covariate encoding is strict, and fails rather than guessing: missing values,
constant columns and rank-deficient designs all raise. Categorical columns are
dummy-encoded against the first observed level, numeric columns are used as-is,
everything is then standardized.

The encoding parameters, not the training design matrix, are what a saved model
carries, so `reconstruct` works on new data. See {doc}`persistence`.

## Related

- {func}`~structboost.encode_obs_covariates`, {func}`~structboost.transform_obs_covariates`
- {doc}`../concepts/anndata-contract`
