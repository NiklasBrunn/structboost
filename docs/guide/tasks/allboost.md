# Sparse supervised boosting

{func}`~structboost.allboost` is the componentwise L2 boosting routine that fits
the BAE encoder, usable on its own, with no autoencoder and no AnnData. It is
pure NumPy and needs only the core install.

```python
import numpy as np
from structboost import allboost

# targets should be standardized, predictors need not be
targets_std = (targets - targets.mean(axis=0)) / targets.std(axis=0)

betamat = allboost(
    sourcemat,        # (n_samples, n_features)
    targets_std,      # (n_samples, n_targets)
    stepno=20,
    nu=0.1,
    csf=0.9,
    mode="standard",
    independent=True,
)
# betamat: (n_targets, n_features) -- note the orientation
```

A natural single-cell use is regressing cluster indicator columns on genes, which
gives a sparse marker signature per cluster directly.

## How it relates to the BAE

It *is* the encoder fitter. Every iteration of {meth}`~structboost.BAE.fit` calls
it with `sourcemat = [X | nuisance]` and `targetmat = z*`, and the config maps
one-to-one:

| `BAEConfig` | `allboost` |
| --- | --- |
| `boosting_stepno` | `stepno` |
| `boosting_nu` | `nu` |
| `boosting_csf` | `csf` |
| `boosting_independent` | `independent` |
| `mandatory_genes` | `mandatory_features` |
| `prior_mode="anchored"` | `beta_init` |

So everything in {doc}`../concepts/how-it-works` about sparsity applies here too:
`stepno` caps how many distinct features can enter.

## The selection criterion

Each step picks the feature maximizing penalized variance reduction,
`(xⱼ'r)² / (‖xⱼ‖² + penaltyⱼ)`. This is scale-invariant in the predictor columns
and does not systematically favour low-norm features, which is why predictor
standardization is recommended but not required. The algorithm uses actual
column norms.

That the selection is *unbiased* rests on the penalty, not on the criterion
alone. Boosting selects without bias when its base-learners are comparable in
degrees of freedom [^hofner2011], and the initial penalty
`penaltyⱼ = ‖xⱼ‖²(1/nu − 1)` makes the effective ridge degrees of freedom equal
to `nu` for every feature, whatever its column norm. `csf` then moves features
off that common footing on purpose, that is the diversity mechanism, not a
bias.

`csf` adapts the per-feature learning rate after each selection:
`nu_j ← 1 - (1 - nu_j)^csf`. Below 1 it promotes diversity. Above 1 it reinforces
already-selected features. Adapting per-feature penalties across boosting steps
is the mechanism Binder & Schumacher [^binder2009] use to fold external biological
knowledge into the fit: pathway membership in their case, mandatory markers or
a chosen `csf` here.

## Forcing features in

```python
betamat = allboost(
    sourcemat,
    targets_std,
    stepno=20,
    mandatory_features=np.array([0, 5, 12], dtype=np.intp),
)
```

Mandatory features enter an unpenalized adjustment block via a joint OLS
pre-step, the mandatory-covariate mechanism of Binder & Schumacher
[^binder2008], so they escape competitive selection. As with the BAE, **this does not
guarantee a non-zero coefficient**. If the contribution is estimated as zero, the
coefficient is zero.

:::{warning}
Boolean masks and float index arrays are rejected outright. A mask silently
reinterpreted as the indices `0`/`1` would select the wrong features. Use
`np.flatnonzero(mask)`.
:::

## Inspecting the path

```python
betamat, hist = allboost(sourcemat, targets, stepno=200, return_history=True)

hist.selection   # (n_targets, stepno) which feature was chosen each step
hist.beta_path   # (n_targets, stepno, n_features) -- can be memory-intensive
```

```python
from structboost import plot_boosting_coefficient_paths

plot_boosting_coefficient_paths(hist.beta_path, gene_names=feature_names)
```

## Reusing the covariance cache

Repeated fits on the same `sourcemat` (cross-validation, parameter sweeps) can
share the predictor Gram matrix:

```python
betamat, covcache = allboost(sourcemat, targets, stepno=20, return_covcache=True)
betamat2 = allboost(sourcemat, other_targets, stepno=20, covcache=covcache)
```

:::{danger}
Reuse a cache **only with the same `sourcemat`**. The Gram matrix depends on the
rows, so a cache from different rows produces silently wrong updates. This is why
stability selection builds a fresh cache per subsample.
:::

{func}`~structboost.compute_covariance_cache` builds one eagerly. It is *O(p²)* in
memory.

## References

[^hofner2011]: Hofner, B., Hothorn, T., Kneib, T. & Schmid, M. (2011). *A
    Framework for Unbiased Model Selection Based on Boosting.* Journal of
    Computational and Graphical Statistics 20(4), 956–971.
    <https://doi.org/10.1198/jcgs.2011.09220>. Comparable base-learner degrees of
    freedom are the condition for unbiased selection.

[^binder2009]: Binder, H. & Schumacher, M. (2009). *Incorporating pathway
    information into boosting estimation of high-dimensional risk prediction
    models.* BMC Bioinformatics 10, 18.
    <https://doi.org/10.1186/1471-2105-10-18>. Adapts single-feature penalties over
    the course of boosting, which is what `csf` implements. This is the paper to
    cite for `allboost` itself.

[^binder2008]: Binder, H. & Schumacher, M. (2008). *Allowing for mandatory
    covariates in boosting estimation of sparse high-dimensional survival
    models.* BMC Bioinformatics 9, 14.
    <https://doi.org/10.1186/1471-2105-9-14>. Introduces the offset-based update
    that admits unpenalized mandatory covariates.

## Related

- {func}`~structboost.allboost`
- {class}`~structboost.AllboostHistory`, {func}`~structboost.compute_covariance_cache`
- {func}`structboost.stability_selection`: the standalone subsample-mode
  stability wrapper around `allboost`
