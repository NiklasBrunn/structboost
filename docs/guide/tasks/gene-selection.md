# Getting a gene list you can trust

A single fit gives you one gene list per latent dimension. That list is **not
reproducible**, and the reason is statistical rather than a defect of the
optimizer.

Gene expression is high-dimensional and strongly correlated, with co-expressed
genes carrying nearly the same information. Under those conditions the encoder
support is not identifiable: many different sparse gene sets reconstruct the data
about equally well, so which one a fit returns is settled by small and essentially
arbitrary differences rather than by the data. The fitted latent space can be
stable while the gene list that produced it is not.

The optimizer makes that visible. Measured on real data, the loss plateaus at
~95% of the achievable linear ceiling while consecutive iterations share only
about a third of their selected genes, with the support autocorrelation still
~0.45 at lag 100 and no periodicity. It is a slow random walk over a plateau
rather than a limit cycle, and a single fit reports one arbitrary position on
that walk.

So: quantify the instability rather than hoping it is small.

## Forcing genes in

```python
# every latent dimension
model.fit(adata, mandatory_genes=["ACTB", "GAPDH"])

# per dimension, sub-lists may differ in length
model.fit(adata, mandatory_genes=[["CD3D", "CD3E"], ["MS4A1"], []])
```

Mandatory genes enter boosting's unpenalized adjustment block, Approach B of
Binder & Schumacher [^binder2008], so they are never subject to competitive
selection. Names or integer column indices both work.

:::{warning}
**This forces genes into the model *specification*, not into the fitted support.**
A gene whose contribution is estimated as zero still ends up with a zero encoder
weight. Do not use this to guarantee that a marker appears in your selected gene
set, that is not what it does.
:::

A collinear mandatory block raises rather than silently regularizing (a redundant
dummy level, for instance). Drop the redundant column, or pass `nuisance_ridge`
explicitly. Ridge changes the estimates, so it is never applied automatically.

## Stability selection

A single fit gives one gene list, and that list is not reproducible: in a
high-dimensional space with correlated genes the encoder support is not
identifiable, and a fit returns one of many sets that reconstruct about equally
well. Stability selection turns that one list into a per-gene frequency.

> *Would these genes still be selected if the optimizer had stopped somewhere
> else on its loss plateau?*

```python
res = model.stability_selection(adata, n_runs=300, threshold=0.7)

adata.varm["BAE_iteration_frequency"]   # (n_genes, latent_dim)
res.dim_match_quality                   # did dimensions keep their identity?
```

This continues training from the fitted state for `n_runs` further iterations,
recording the encoder support after each boosting step, then restores the model,
so the call is non-destructive.

That is the variance source that dominates. The encoder support does not converge
even when the reconstruction loss does: on measured data the loss plateaus at ~95%
of the achievable linear ceiling while consecutive iterations share only about a
third of their selected genes, with the support autocorrelation still ~0.45 at lag
100 and no periodicity, a slow random walk over a plateau. A single fit reports one
arbitrary position on that walk.

Measured as the best of the available readouts: FDR 0.26, against 0.31 for cell
subsampling and 0.38 for a single fit. Genes selected in 90–100% of iterations are
true markers 88% of the time.

Dimensions are Hungarian-matched to the fitted model by maximum absolute cosine
before counting; that anchoring is what gives a dimension index a stable meaning.
`dim_match_quality` reports how well it held (~0.84 measured). Below 0.5 a warning
fires and `frequency.max(axis=1)`, the flat union, is the safer readout.

`n_runs=300` is the default because the support autocorrelation decays slowly.
Short windows give near-duplicate samples.

:::{warning}
Stability selection here provides **no formal error control**.
`expected_false_positives` is deliberately `NaN` rather than a number that would
look like a guarantee: training iterations are neither independent nor
exchangeable, so the Meinshausen–Bühlmann bound does not apply.

For a bound with a derivation behind it, the standalone
{func}`structboost.stability_selection` resamples cells in the Meinshausen–Bühlmann
scheme [^mb2010]. It operates on an `allboost` problem directly rather than on a
fitted BAE. Note that on simulated data where the truth is known, that bound was
measured to be **violated by roughly an order of magnitude** in the BAE setting
(mean realized 2.57 false positives per dimension against a bound of 0.20) — the
targets `z*` come from a model fitted on the same cells being resampled, which is
not the fixed-in-advance inference problem the theory assumes.
:::

### What this does not capture

It measures stability *conditional on the learned representation*. It does not
capture the variability from re-initializing and refitting the autoencoder to a
different local optimum.

On simulated data the per-gene frequencies track a full-refit gold standard
closely (correlation ~0.95), so this is usually acceptable. But where the model
has several competing optima, high-stakes marker claims still warrant a handful
of full refits as a cross-check.

## Reading the result

{class}`~structboost.StabilitySelectionResult` carries more than frequencies:

```python
res.frequency          # (n_genes, latent_dim)
res.stable_support     # frequency >= threshold
res.avg_selected       # mean genes selected per run, per dimension
res.sign_consistency   # fraction of runs taking the modal sign
res.coefficient_sd     # spread of the selected coefficients
```

`sign_consistency` catches something a frequency cannot: a gene selected every
single time, with a sign that flips, is not stable. Measured at ~0.999 on real
data, so in practice a guard rather than a headline.

:::{note}
`coefficient_sd` is a **spread, not a standard error**. Iteration runs are
autocorrelated, so the effective sample size is far below `n_runs`, and
`sd / sqrt(n_runs)` would be badly overconfident. Reporting it as a precision
would require a block bootstrap.
:::

Per-dimension gene sets:

```python
genes = [
    adata.var_names[res.stable_support[:, j]]
    for j in range(res.frequency.shape[1])
]
```

## Installing the aggregated encoder

The *support* of a fit is more reproducible than its *weights*. `stable_encoder()`
builds an encoder from the stable support, which you can install:

```python
model.apply_encoder(res.stable_encoder(), adata)
```

Measured against ground truth: precision 0.62 → 0.74, FDR 0.38 → 0.26, 156 → 116
genes, reconstruction 55% → 59% of the linear ceiling. The decoder is untouched.

This is a separate step on purpose. A diagnostic that silently swapped the
encoder would make `fit()` followed by a reliability check produce a different
model than `fit()` alone, and would compound if called twice. It is also not a
strictly better encoder but a **choice**, precision bought with recall, and
that trade belongs to you.

To move along that trade-off, lower `threshold` rather than reaching for a
different estimator. The threshold *is* the dial.

:::{danger}
On a **frozen transfer model**, `apply_encoder(res.stable_encoder())` would
delete the transferred programs: that estimator zeroes every entry outside the
stable support, and frozen prior columns have selection frequency zero by
construction. They cannot vary, so there is nothing for them to be stable about.
The result is an encoder whose prior block is all zeros, with no error raised.
Pass `preserve_prior=True` (the default). See {doc}`transfer`.
:::

## Running one automatically after training

```python
model.fit(adata, stability_selection="iteration")
```

This uses the method defaults (`n_runs=300`, `threshold=0.7`). Call the method
directly when you want control over them.

## References

[^binder2008]: Binder, H. & Schumacher, M. (2008). *Allowing for mandatory
    covariates in boosting estimation of sparse high-dimensional survival
    models.* BMC Bioinformatics 9, 14.
    <https://doi.org/10.1186/1471-2105-9-14>. Provides the offset-based update that
    admits unpenalized mandatory covariates.

[^mb2010]: Meinshausen, N. & Bühlmann, P. (2010). *Stability selection.* Journal
    of the Royal Statistical Society: Series B 72(4), 417–473.
    <https://doi.org/10.1111/j.1467-9868.2010.00740.x>. Provides subsampling scheme and
    the error bound this implementation reports (and measurably violates, see
    above).

## Related

- {meth}`~structboost.BAE.stability_selection`, {class}`~structboost.StabilitySelectionResult`
- {func}`structboost.stability_selection`: the standalone `allboost`-level
  function of the same name, which resamples cells rather than iterating, and has
  its own defaults
- {doc}`interpreting`: what the selected genes mean
