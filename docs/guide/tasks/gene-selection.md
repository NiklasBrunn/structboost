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

## Two stability modes

Both freeze the model, recompute the boosting targets `z*` once, and re-solve the
encoder's boosting problem many times. Both cost a fraction of one fit and need
no refitting. Both are non-destructive, and the model is restored afterwards.

They measure **different** sources of instability, so they complement rather than
replace each other.

### Iteration mode (the default)

> *Would these genes still be selected if the optimizer had stopped somewhere
> else on its loss plateau?*

```python
res = model.stability_selection(adata, mode="iteration", n_runs=300, threshold=0.7)

adata.varm["BAE_iteration_frequency"]   # (n_genes, latent_dim)
res.dim_match_quality                   # did dimensions keep their identity?
```

This runs `n_runs` further training iterations from the fitted state, recording
the support after each boosting step. Measured as the best of the three readouts:
FDR 0.26, against 0.31 for subsampling and 0.38 for a single fit. Genes selected
in 90–100% of iterations are true markers 88% of the time.

Dimensions are Hungarian-matched to the fitted model by maximum absolute cosine
before counting, that anchoring is what gives a dimension index a stable
meaning. `dim_match_quality` reports how well it held (~0.84 measured). Below
0.5 a warning fires and `frequency.max(axis=1)`, the flat union, is the safer
readout.

`n_runs=300` is the default because the support autocorrelation decays slowly.
Short windows give near-duplicate samples.

:::{warning}
Iteration mode provides **no formal error control**. `expected_false_positives`
is deliberately `NaN` rather than a number that would look like a guarantee, 
training iterations are neither independent nor exchangeable.
:::

### Subsample mode

> *Would these genes still be selected on a different sample of cells?*

```python
res = model.stability_selection(adata, mode="subsample", n_runs=100, threshold=0.7)

adata.varm["BAE_selection_frequency"]
res.expected_false_positives             # Meinshausen-Bühlmann bound
```

Meinshausen–Bühlmann subsampling [^mb2010] without replacement at half the cells.
`0.5` is the fraction their theory is derived for. Bootstrap resampling is
deliberately not offered, because sampling with replacement breaks the exchangeability
the bound relies on.

:::{danger}
**The error bound does not hold here.** On simulated data where the truth is
known, the Meinshausen–Bühlmann bound is violated by roughly an order of
magnitude: mean realized 2.57 false positives per dimension against a bound of
0.20, satisfied in 15 of 60 dimensions.

The likely cause is a scope error rather than an implementation bug. The bound
assumes exchangeable subsamples of an inference problem fixed in advance, whereas
the targets `z*` here come from a model fitted on the same cells being resampled.

**Treat `expected_false_positives` as a diagnostic, not a guarantee.** This mode
is retained for investigation.
:::

### What neither mode captures

Both measure stability *conditional on the learned representation*. Neither
captures the variability from re-initializing and refitting the autoencoder to a
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
`coefficient_sd` is a **spread, not a standard error**. Subsample runs share half
their cells by construction and iteration runs are autocorrelated, so the
effective sample size is far below `n_runs`. `sd / sqrt(n_runs)` would be badly
overconfident. Reporting it as a precision would require a block bootstrap.
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
  function of the same name, subsample-only, with different defaults
- {doc}`interpreting`: what the selected genes mean
